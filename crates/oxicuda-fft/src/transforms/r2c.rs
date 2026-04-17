//! Real-to-complex FFT execution.
//!
//! Implements the R2C (real-to-complex) transform: given N real-valued
//! samples, produces N/2 + 1 complex frequency bins exploiting Hermitian
//! symmetry.
//!
//! # Algorithm
//!
//! 1. Pack the N real values into N/2 complex values (even-indexed reals
//!    become the real parts, odd-indexed become imaginary parts).
//! 2. Execute a C2C forward FFT of length N/2.
//! 3. Post-process the N/2 complex results to recover the N/2 + 1 unique
//!    frequency bins of the original real sequence.
#![allow(dead_code)]

use oxicuda_driver::Stream;
use oxicuda_driver::ffi::CUdeviceptr;
use std::f32::consts::PI as PI_F32;
use std::f64::consts::PI as PI_F64;

use crate::error::{FftError, FftResult};
use crate::plan::FftPlan;
use crate::transforms::c2c;
use crate::types::Complex;
use crate::types::{FftDirection, FftType};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Executes a real-to-complex FFT.
///
/// # Arguments
///
/// * `plan`   - A compiled [`FftPlan`] with `transform_type == FftType::R2C`.
/// * `input`  - Device pointer to N real-valued elements.
/// * `output` - Device pointer to N/2 + 1 complex elements.
/// * `stream` - CUDA [`Stream`] for asynchronous execution.
///
/// # Errors
///
/// Returns [`FftError::UnsupportedTransform`] if the plan is not R2C.
/// Returns [`FftError::InvalidSize`] if the FFT size is odd (R2C requires
/// even N).
pub fn fft_r2c(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    stream: &Stream,
) -> FftResult<()> {
    validate_r2c_plan(plan)?;

    let n = plan.sizes[0];

    // Step 1: Pack real data as N/2 complex values
    // The packing is implicit if the real data is contiguous: we
    // reinterpret the N reals as N/2 complex (re, im) pairs.
    let half_n = n / 2;

    // Step 2: Execute C2C forward FFT of length N/2
    execute_half_length_c2c(plan, input, output, half_n, stream)?;

    // Step 3: Post-process to extract Hermitian-symmetric output
    post_process_r2c(plan, output, n, stream)?;

    Ok(())
}

/// Executes a batched real-to-complex FFT.
///
/// # Errors
///
/// Returns [`FftError::UnsupportedTransform`] if the plan is not R2C.
/// Returns [`FftError::InvalidBatch`] if `batch_count` is zero.
pub fn fft_r2c_batched(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    batch_count: usize,
    stream: &Stream,
) -> FftResult<()> {
    if batch_count == 0 {
        return Err(FftError::InvalidBatch(
            "batch_count must be >= 1".to_string(),
        ));
    }
    let mut exec_plan = plan.clone();
    exec_plan.batch = batch_count;
    fft_r2c(&exec_plan, input, output, stream)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates that the plan is suitable for an R2C transform.
fn validate_r2c_plan(plan: &FftPlan) -> FftResult<()> {
    if plan.transform_type != FftType::R2C {
        return Err(FftError::UnsupportedTransform(format!(
            "expected R2C plan, got {}",
            plan.transform_type
        )));
    }

    let n = plan.sizes[0];
    if n % 2 != 0 {
        return Err(FftError::InvalidSize(format!(
            "R2C requires even FFT size, got {n}"
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal steps
// ---------------------------------------------------------------------------

/// Executes the half-length C2C FFT on the packed real data.
///
/// The input is reinterpreted as N/2 complex pairs and a forward C2C
/// transform is applied.
fn execute_half_length_c2c(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    half_n: usize,
    stream: &Stream,
) -> FftResult<()> {
    let mut half_plan = FftPlan::new_1d(half_n, FftType::C2C, plan.batch)?
        .with_precision(plan.precision)
        .with_direction(FftDirection::Forward);

    // Reuse already compiled kernels when the source plan layout matches.
    if plan.transform_type == FftType::C2C && plan.sizes.len() == 1 && plan.sizes[0] == half_n {
        half_plan.strategy = plan.strategy.clone();
        half_plan.compiled_kernels = plan.compiled_kernels.clone();
        half_plan.temp_buffer_bytes = plan.temp_buffer_bytes;
        half_plan.kernel_variant = plan.kernel_variant;
    }

    c2c::fft_c2c(&half_plan, input, output, FftDirection::Forward, stream)
}

/// Post-processes the half-length C2C output to produce the full R2C result.
///
/// Given the C2C output `X[k]` for k = 0..N/2-1, computes the R2C output
/// `Y[k]` for k = 0..N/2 using:
///
/// ```text
/// Y[k]     = 0.5 * (X[k] + conj(X[N/2 - k]))
///          - 0.5j * W_N^k * (X[k] - conj(X[N/2 - k]))
/// Y[N/2]   = Re(X[0]) - Im(X[0])
/// Y[0]     = Re(X[0]) + Im(X[0])
/// ```
///
/// where `W_N^k = exp(-2*pi*i*k/N)` is the twiddle factor.
fn post_process_r2c(
    plan: &FftPlan,
    output: CUdeviceptr,
    n: usize,
    stream: &Stream,
) -> FftResult<()> {
    let half_n = n / 2;
    let batch = plan.batch;

    if half_n == 0 || batch == 0 {
        return Ok(());
    }

    match plan.precision {
        crate::types::FftPrecision::Single => {
            for b in 0..batch {
                let in_off = (b * half_n * std::mem::size_of::<Complex<f32>>()) as u64;
                let mut x = vec![Complex::<f32>::zero(); half_n];
                copy_dtoh_async(&mut x, output + in_off, stream)?;

                let y = postprocess_host_f32(&x, n);

                let out_off = (b * (half_n + 1) * std::mem::size_of::<Complex<f32>>()) as u64;
                copy_htod_async(output + out_off, &y, stream)?;
            }
        }
        crate::types::FftPrecision::Double => {
            for b in 0..batch {
                let in_off = (b * half_n * std::mem::size_of::<Complex<f64>>()) as u64;
                let mut x = vec![Complex::<f64>::zero(); half_n];
                copy_dtoh_async(&mut x, output + in_off, stream)?;

                let y = postprocess_host_f64(&x, n);

                let out_off = (b * (half_n + 1) * std::mem::size_of::<Complex<f64>>()) as u64;
                copy_htod_async(output + out_off, &y, stream)?;
            }
        }
    }

    stream.synchronize()?;

    Ok(())
}

fn copy_dtoh_async<T: Copy>(dst: &mut [T], src: CUdeviceptr, stream: &Stream) -> FftResult<()> {
    let api = oxicuda_driver::try_driver()?;
    let byte_count = std::mem::size_of_val(dst);
    let rc = unsafe {
        (api.cu_memcpy_dtoh_async_v2)(
            dst.as_mut_ptr().cast::<std::ffi::c_void>(),
            src,
            byte_count,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)?;
    Ok(())
}

fn copy_htod_async<T: Copy>(dst: CUdeviceptr, src: &[T], stream: &Stream) -> FftResult<()> {
    let api = oxicuda_driver::try_driver()?;
    let byte_count = std::mem::size_of_val(src);
    let rc = unsafe {
        (api.cu_memcpy_htod_async_v2)(
            dst,
            src.as_ptr().cast::<std::ffi::c_void>(),
            byte_count,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)?;
    Ok(())
}

fn postprocess_host_f32(x: &[Complex<f32>], n: usize) -> Vec<Complex<f32>> {
    let half_n = n / 2;
    let mut y = vec![Complex::<f32>::zero(); half_n + 1];
    if half_n == 0 {
        return y;
    }

    let x0 = x[0];
    y[0] = Complex::<f32>::new(x0.re + x0.im, 0.0);
    y[half_n] = Complex::<f32>::new(x0.re - x0.im, 0.0);

    for k in 1..half_n {
        let a = x[k];
        let b = x[half_n - k].conj();
        let sum = Complex::<f32>::new(0.5 * (a.re + b.re), 0.5 * (a.im + b.im));
        let diff = Complex::<f32>::new(0.5 * (a.re - b.re), 0.5 * (a.im - b.im));

        let angle = -2.0_f32 * PI_F32 * (k as f32) / (n as f32);
        let w = Complex::<f32>::new(angle.cos(), angle.sin());
        let t = w * diff;
        let corr = Complex::<f32>::new(t.im, -t.re); // -j * t

        y[k] = sum + corr;
    }

    y
}

fn postprocess_host_f64(x: &[Complex<f64>], n: usize) -> Vec<Complex<f64>> {
    let half_n = n / 2;
    let mut y = vec![Complex::<f64>::zero(); half_n + 1];
    if half_n == 0 {
        return y;
    }

    let x0 = x[0];
    y[0] = Complex::<f64>::new(x0.re + x0.im, 0.0);
    y[half_n] = Complex::<f64>::new(x0.re - x0.im, 0.0);

    for k in 1..half_n {
        let a = x[k];
        let b = x[half_n - k].conj();
        let sum = Complex::<f64>::new(0.5 * (a.re + b.re), 0.5 * (a.im + b.im));
        let diff = Complex::<f64>::new(0.5 * (a.re - b.re), 0.5 * (a.im - b.im));

        let angle = -2.0_f64 * PI_F64 * (k as f64) / (n as f64);
        let w = Complex::<f64>::new(angle.cos(), angle.sin());
        let t = w * diff;
        let corr = Complex::<f64>::new(t.im, -t.re); // -j * t

        y[k] = sum + corr;
    }

    y
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r2c_rejects_c2c_plan() {
        let plan = FftPlan::new_1d(256, FftType::C2C, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_r2c_plan(&p);
            assert!(matches!(result, Err(FftError::UnsupportedTransform(_))));
        }
    }

    #[test]
    fn r2c_rejects_odd_size() {
        let plan = FftPlan::new_1d(255, FftType::R2C, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_r2c_plan(&p);
            assert!(matches!(result, Err(FftError::InvalidSize(_))));
        }
    }

    #[test]
    fn r2c_accepts_even_size() {
        let plan = FftPlan::new_1d(256, FftType::R2C, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_r2c_plan(&p);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn r2c_postprocess_dc_nyquist_f32() {
        let x = vec![Complex::<f32>::new(3.0, 1.0), Complex::<f32>::new(0.0, 0.0)];
        let y = postprocess_host_f32(&x, 4);
        assert_eq!(y.len(), 3);
        assert!((y[0].re - 4.0).abs() < 1e-6);
        assert!(y[0].im.abs() < 1e-6);
        assert!((y[2].re - 2.0).abs() < 1e-6);
        assert!(y[2].im.abs() < 1e-6);
    }
}
