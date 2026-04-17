//! Complex-to-real FFT execution.
//!
//! Implements the C2R (complex-to-real) transform, which is the inverse
//! of the R2C transform.  Given N/2 + 1 complex frequency bins with
//! Hermitian symmetry, produces N real-valued time-domain samples.
//!
//! # Algorithm
//!
//! 1. Pre-process the N/2 + 1 complex bins back to N/2 complex values
//!    suitable for a half-length inverse C2C FFT.
//! 2. Execute an inverse C2C FFT of length N/2.
//! 3. Unpack the N/2 complex results into N real values.
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

/// Executes a complex-to-real inverse FFT.
///
/// # Arguments
///
/// * `plan`   - A compiled [`FftPlan`] with `transform_type == FftType::C2R`.
/// * `input`  - Device pointer to N/2 + 1 complex elements (Hermitian).
/// * `output` - Device pointer to N real elements.
/// * `stream` - CUDA [`Stream`] for asynchronous execution.
///
/// # Errors
///
/// Returns [`FftError::UnsupportedTransform`] if the plan is not C2R.
/// Returns [`FftError::InvalidSize`] if the FFT size is odd.
pub fn fft_c2r(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    stream: &Stream,
) -> FftResult<()> {
    validate_c2r_plan(plan)?;

    let n = plan.sizes[0];
    let half_n = n / 2;

    // Step 1: Pre-process Hermitian input into N/2 complex values
    pre_process_c2r(plan, input, output, n, stream)?;

    // Step 2: Execute inverse C2C FFT of length N/2
    execute_half_length_inverse_c2c(plan, output, output, half_n, stream)?;

    // Step 3: Unpack N/2 complex values into N real values
    unpack_real_output(plan, output, n, stream)?;

    Ok(())
}

/// Executes a batched complex-to-real inverse FFT.
///
/// # Errors
///
/// Returns [`FftError::UnsupportedTransform`] if the plan is not C2R.
/// Returns [`FftError::InvalidBatch`] if `batch_count` is zero.
pub fn fft_c2r_batched(
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
    fft_c2r(&exec_plan, input, output, stream)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates that the plan is suitable for a C2R transform.
fn validate_c2r_plan(plan: &FftPlan) -> FftResult<()> {
    if plan.transform_type != FftType::C2R {
        return Err(FftError::UnsupportedTransform(format!(
            "expected C2R plan, got {}",
            plan.transform_type
        )));
    }

    let n = plan.sizes[0];
    if n % 2 != 0 {
        return Err(FftError::InvalidSize(format!(
            "C2R requires even FFT size, got {n}"
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal steps
// ---------------------------------------------------------------------------

/// Pre-processes the Hermitian-symmetric input to prepare for the
/// half-length inverse C2C FFT.
///
/// Given input `Y[k]` for k = 0..N/2, computes `X[k]` for k = 0..N/2-1:
///
/// ```text
/// X[k] = 0.5 * (Y[k] + conj(Y[N/2 - k]))
///      + 0.5j * conj(W_N^k) * (Y[k] - conj(Y[N/2 - k]))
/// X[0] = (Re(Y[0]) + Re(Y[N/2]), Re(Y[0]) - Re(Y[N/2]))
/// ```
///
/// This is the exact inverse of the R2C post-processing step.
fn pre_process_c2r(
    plan: &FftPlan,
    input: CUdeviceptr,
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
                let in_off = (b * (half_n + 1) * std::mem::size_of::<Complex<f32>>()) as u64;
                let mut y = vec![Complex::<f32>::zero(); half_n + 1];
                copy_dtoh_async(&mut y, input + in_off, stream)?;

                let x = preprocess_host_f32(&y, n);

                let out_off = (b * half_n * std::mem::size_of::<Complex<f32>>()) as u64;
                copy_htod_async(output + out_off, &x, stream)?;
            }
        }
        crate::types::FftPrecision::Double => {
            for b in 0..batch {
                let in_off = (b * (half_n + 1) * std::mem::size_of::<Complex<f64>>()) as u64;
                let mut y = vec![Complex::<f64>::zero(); half_n + 1];
                copy_dtoh_async(&mut y, input + in_off, stream)?;

                let x = preprocess_host_f64(&y, n);

                let out_off = (b * half_n * std::mem::size_of::<Complex<f64>>()) as u64;
                copy_htod_async(output + out_off, &x, stream)?;
            }
        }
    }

    stream.synchronize()?;

    Ok(())
}

/// Executes the half-length inverse C2C FFT.
fn execute_half_length_inverse_c2c(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    half_n: usize,
    stream: &Stream,
) -> FftResult<()> {
    let mut half_plan = FftPlan::new_1d(half_n, FftType::C2C, plan.batch)?
        .with_precision(plan.precision)
        .with_direction(FftDirection::Inverse);

    if plan.transform_type == FftType::C2C && plan.sizes.len() == 1 && plan.sizes[0] == half_n {
        half_plan.strategy = plan.strategy.clone();
        half_plan.compiled_kernels = plan.compiled_kernels.clone();
        half_plan.temp_buffer_bytes = plan.temp_buffer_bytes;
        half_plan.kernel_variant = plan.kernel_variant;
    }

    c2c::fft_c2c(&half_plan, input, output, FftDirection::Inverse, stream)
}

/// Unpacks the N/2 complex results into N contiguous real values.
///
/// After the inverse C2C, the result is N/2 complex values whose real
/// and imaginary parts, interleaved, form the original N real samples.
/// This step copies them into a contiguous real output buffer.
fn unpack_real_output(
    plan: &FftPlan,
    _output: CUdeviceptr,
    _n: usize,
    _stream: &Stream,
) -> FftResult<()> {
    // When the output buffer is the same as the C2C output, the real
    // values are already in the correct interleaved order (re0, im0 = x0, x1).
    // No additional copy is needed.

    let _precision = plan.precision;

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

fn preprocess_host_f32(y: &[Complex<f32>], n: usize) -> Vec<Complex<f32>> {
    let half_n = n / 2;
    let mut x = vec![Complex::<f32>::zero(); half_n];
    if half_n == 0 {
        return x;
    }

    x[0] = Complex::<f32>::new(y[0].re + y[half_n].re, y[0].re - y[half_n].re);

    for k in 1..half_n {
        let a = y[k];
        let b = y[half_n - k].conj();
        let sum = Complex::<f32>::new(0.5 * (a.re + b.re), 0.5 * (a.im + b.im));
        let diff = Complex::<f32>::new(0.5 * (a.re - b.re), 0.5 * (a.im - b.im));

        let angle = 2.0_f32 * PI_F32 * (k as f32) / (n as f32);
        let wc = Complex::<f32>::new(angle.cos(), angle.sin()); // conj(W)
        let t = wc * diff;
        let corr = Complex::<f32>::new(-t.im, t.re); // j * t

        x[k] = sum + corr;
    }

    x
}

fn preprocess_host_f64(y: &[Complex<f64>], n: usize) -> Vec<Complex<f64>> {
    let half_n = n / 2;
    let mut x = vec![Complex::<f64>::zero(); half_n];
    if half_n == 0 {
        return x;
    }

    x[0] = Complex::<f64>::new(y[0].re + y[half_n].re, y[0].re - y[half_n].re);

    for k in 1..half_n {
        let a = y[k];
        let b = y[half_n - k].conj();
        let sum = Complex::<f64>::new(0.5 * (a.re + b.re), 0.5 * (a.im + b.im));
        let diff = Complex::<f64>::new(0.5 * (a.re - b.re), 0.5 * (a.im - b.im));

        let angle = 2.0_f64 * PI_F64 * (k as f64) / (n as f64);
        let wc = Complex::<f64>::new(angle.cos(), angle.sin()); // conj(W)
        let t = wc * diff;
        let corr = Complex::<f64>::new(-t.im, t.re); // j * t

        x[k] = sum + corr;
    }

    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c2r_rejects_c2c_plan() {
        let plan = FftPlan::new_1d(256, FftType::C2C, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_c2r_plan(&p);
            assert!(matches!(result, Err(FftError::UnsupportedTransform(_))));
        }
    }

    #[test]
    fn c2r_rejects_odd_size() {
        let plan = FftPlan::new_1d(255, FftType::C2R, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_c2r_plan(&p);
            assert!(matches!(result, Err(FftError::InvalidSize(_))));
        }
    }

    #[test]
    fn c2r_accepts_even_size() {
        let plan = FftPlan::new_1d(256, FftType::C2R, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_c2r_plan(&p);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn c2r_preprocess_dc_nyquist_f32() {
        // Inverse of r2c_postprocess_dc_nyquist_f32 with N=4
        let y = vec![
            Complex::<f32>::new(4.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
            Complex::<f32>::new(2.0, 0.0),
        ];
        let x = preprocess_host_f32(&y, 4);
        assert_eq!(x.len(), 2);
        assert!((x[0].re - 6.0).abs() < 1e-6);
        assert!((x[0].im - 2.0).abs() < 1e-6);
    }
}
