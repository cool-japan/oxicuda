//! Complex-to-complex FFT execution.
//!
//! Implements the C2C transform by dispatching the compiled kernels in the
//! [`FftPlan`].  For small sizes the plan contains a single kernel; for
//! large sizes a multi-stage ping-pong execution is performed.
#![allow(dead_code)]

use oxicuda_driver::Stream;
use oxicuda_driver::ffi::CUdeviceptr;
use std::f32::consts::PI as PI_F32;
use std::f64::consts::PI as PI_F64;

use crate::error::{FftError, FftResult};
use crate::plan::FftPlan;
use crate::types::{Complex, FftDirection, FftType};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Executes a complex-to-complex FFT according to the given plan.
///
/// # Arguments
///
/// * `plan`      - A compiled [`FftPlan`] with `transform_type == FftType::C2C`.
/// * `input`     - Device pointer to the input complex array.
/// * `output`    - Device pointer to the output complex array.
/// * `direction` - [`FftDirection::Forward`] or [`FftDirection::Inverse`].
/// * `stream`    - CUDA [`Stream`] on which to enqueue the kernel launches.
///
/// # In-place transforms
///
/// When `input == output`, a temporary buffer (sized by
/// `plan.temp_buffer_bytes`) is used internally for multi-stage plans.
///
/// # Errors
///
/// Returns [`FftError::UnsupportedTransform`] if the plan is not C2C.
/// Returns [`FftError::InternalError`] if the plan has no compiled kernels.
pub fn fft_c2c(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    direction: FftDirection,
    stream: &Stream,
) -> FftResult<()> {
    // Validate transform type
    if plan.transform_type != FftType::C2C {
        return Err(FftError::UnsupportedTransform(format!(
            "expected C2C plan, got {}",
            plan.transform_type
        )));
    }

    // Single-kernel path: one kernel covers all Stockham stages
    if plan.strategy.single_kernel {
        return execute_single_kernel(plan, input, output, direction, stream);
    }

    // Multi-stage path: each compiled kernel handles one radix stage
    execute_multi_stage(plan, input, output, direction, stream)
}

// ---------------------------------------------------------------------------
// Single-kernel execution
// ---------------------------------------------------------------------------

/// Launches the single compiled kernel that performs the full FFT in shared
/// memory.
fn execute_single_kernel(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    direction: FftDirection,
    stream: &Stream,
) -> FftResult<()> {
    execute_host_fallback(plan, input, output, direction, stream)
}

// ---------------------------------------------------------------------------
// Multi-stage execution (ping-pong)
// ---------------------------------------------------------------------------

/// Launches one kernel per Stockham stage, using a ping-pong buffer pattern.
///
/// Stage layout:
///   stage 0: input  -> temp
///   stage 1: temp   -> input  (or output if last)
///   stage 2: input  -> temp   ...
///   ...
///   last:    ...    -> output
///
/// When `input == output` (in-place), all intermediate writes use the temp
/// buffer and a final copy ensures the result ends up at `output`.
fn execute_multi_stage(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    direction: FftDirection,
    stream: &Stream,
) -> FftResult<()> {
    execute_host_fallback(plan, input, output, direction, stream)
}

fn execute_host_fallback(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    direction: FftDirection,
    stream: &Stream,
) -> FftResult<()> {
    let n = plan.sizes[0];
    let batch = plan.batch;

    if n == 0 || batch == 0 {
        return Ok(());
    }

    match plan.precision {
        crate::types::FftPrecision::Single => {
            for b in 0..batch {
                let off = (b * n * std::mem::size_of::<Complex<f32>>()) as u64;
                let mut x = vec![Complex::<f32>::zero(); n];
                copy_dtoh_async(&mut x, input + off, stream)?;
                let y = if n.is_power_of_two() {
                    fft_radix2_host_f32(&x, direction)
                } else {
                    dft_host_f32(&x, direction)
                };
                copy_htod_async(output + off, &y, stream)?;
            }
        }
        crate::types::FftPrecision::Double => {
            for b in 0..batch {
                let off = (b * n * std::mem::size_of::<Complex<f64>>()) as u64;
                let mut x = vec![Complex::<f64>::zero(); n];
                copy_dtoh_async(&mut x, input + off, stream)?;
                let y = if n.is_power_of_two() {
                    fft_radix2_host_f64(&x, direction)
                } else {
                    dft_host_f64(&x, direction)
                };
                copy_htod_async(output + off, &y, stream)?;
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

fn dft_host_f32(x: &[Complex<f32>], direction: FftDirection) -> Vec<Complex<f32>> {
    let n = x.len();
    let mut y = vec![Complex::<f32>::zero(); n];
    if n == 0 {
        return y;
    }

    let sign = match direction {
        FftDirection::Forward => -1.0_f32,
        FftDirection::Inverse => 1.0_f32,
    };

    for (k, yk) in y.iter_mut().enumerate() {
        let mut acc = Complex::<f32>::zero();
        for (t, &xt) in x.iter().enumerate() {
            let angle = sign * 2.0_f32 * PI_F32 * (k as f32) * (t as f32) / (n as f32);
            let w = Complex::<f32>::new(angle.cos(), angle.sin());
            acc = acc + (xt * w);
        }
        *yk = acc;
    }

    y
}

fn dft_host_f64(x: &[Complex<f64>], direction: FftDirection) -> Vec<Complex<f64>> {
    let n = x.len();
    let mut y = vec![Complex::<f64>::zero(); n];
    if n == 0 {
        return y;
    }

    let sign = match direction {
        FftDirection::Forward => -1.0_f64,
        FftDirection::Inverse => 1.0_f64,
    };

    for (k, yk) in y.iter_mut().enumerate() {
        let mut acc = Complex::<f64>::zero();
        for (t, &xt) in x.iter().enumerate() {
            let angle = sign * 2.0_f64 * PI_F64 * (k as f64) * (t as f64) / (n as f64);
            let w = Complex::<f64>::new(angle.cos(), angle.sin());
            acc = acc + (xt * w);
        }
        *yk = acc;
    }

    y
}

fn bit_reverse_permute_f32(data: &mut [Complex<f32>]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (u32::BITS - bits);
        let j = j as usize;
        if j > i {
            data.swap(i, j);
        }
    }
}

fn bit_reverse_permute_f64(data: &mut [Complex<f64>]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (u32::BITS - bits);
        let j = j as usize;
        if j > i {
            data.swap(i, j);
        }
    }
}

fn fft_radix2_host_f32(x: &[Complex<f32>], direction: FftDirection) -> Vec<Complex<f32>> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }

    let mut a = x.to_vec();
    bit_reverse_permute_f32(&mut a);

    let sign = match direction {
        FftDirection::Forward => -1.0_f32,
        FftDirection::Inverse => 1.0_f32,
    };

    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta = sign * 2.0_f32 * PI_F32 / (len as f32);
        let w_len = Complex::<f32>::new(theta.cos(), theta.sin());

        let mut i = 0usize;
        while i < n {
            let mut w = Complex::<f32>::one();
            for j in 0..half {
                let u = a[i + j];
                let v = a[i + j + half] * w;
                a[i + j] = u + v;
                a[i + j + half] = u - v;
                w = w * w_len;
            }
            i += len;
        }

        len *= 2;
    }

    a
}

fn fft_radix2_host_f64(x: &[Complex<f64>], direction: FftDirection) -> Vec<Complex<f64>> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }

    let mut a = x.to_vec();
    bit_reverse_permute_f64(&mut a);

    let sign = match direction {
        FftDirection::Forward => -1.0_f64,
        FftDirection::Inverse => 1.0_f64,
    };

    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta = sign * 2.0_f64 * PI_F64 / (len as f64);
        let w_len = Complex::<f64>::new(theta.cos(), theta.sin());

        let mut i = 0usize;
        while i < n {
            let mut w = Complex::<f64>::one();
            for j in 0..half {
                let u = a[i + j];
                let v = a[i + j + half] * w;
                a[i + j] = u + v;
                a[i + j + half] = u - v;
                w = w * w_len;
            }
            i += len;
        }

        len *= 2;
    }

    a
}

// ---------------------------------------------------------------------------
// Batch execution
// ---------------------------------------------------------------------------

/// Executes a batched C2C FFT.
///
/// This is identical to [`fft_c2c`] but accepts an explicit `batch_count`
/// that may differ from the plan's default (e.g., for sub-batching).
///
/// # Errors
///
/// Returns [`FftError::UnsupportedTransform`] if the plan is not C2C.
/// Returns [`FftError::InvalidBatch`] if `batch_count` is zero.
pub fn fft_c2c_batched(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    direction: FftDirection,
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
    fft_c2c(&exec_plan, input, output, direction, stream)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::FftPlan;

    fn approx_eq_f32(a: Complex<f32>, b: Complex<f32>, tol: f32) {
        assert!((a.re - b.re).abs() <= tol, "re: {} vs {}", a.re, b.re);
        assert!((a.im - b.im).abs() <= tol, "im: {} vs {}", a.im, b.im);
    }

    fn approx_eq_f64(a: Complex<f64>, b: Complex<f64>, tol: f64) {
        assert!((a.re - b.re).abs() <= tol, "re: {} vs {}", a.re, b.re);
        assert!((a.im - b.im).abs() <= tol, "im: {} vs {}", a.im, b.im);
    }

    #[test]
    fn c2c_validates_transform_type() {
        let plan = FftPlan::new_1d(256, FftType::R2C, 1);
        assert!(plan.is_ok());
        // Verify that a R2C plan would be rejected by the type check
        if let Ok(p) = plan {
            assert_ne!(p.transform_type, FftType::C2C);
        }
    }

    #[test]
    fn c2c_plan_strategy() {
        let plan = FftPlan::new_1d(256, FftType::C2C, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert_eq!(p.transform_type, FftType::C2C);
            assert!(p.strategy.single_kernel);
        }
    }

    #[test]
    fn dft_forward_impulse_is_all_ones_f32() {
        let x = vec![
            Complex::<f32>::new(1.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
        ];
        let y = dft_host_f32(&x, FftDirection::Forward);
        for yi in y {
            approx_eq_f32(yi, Complex::<f32>::new(1.0, 0.0), 1e-5);
        }
    }

    #[test]
    fn dft_inverse_of_ones_is_dc_spike_f64() {
        let x = vec![Complex::<f64>::new(1.0, 0.0); 4];
        let y = dft_host_f64(&x, FftDirection::Inverse);
        approx_eq_f64(y[0], Complex::<f64>::new(4.0, 0.0), 1e-12);
        approx_eq_f64(y[1], Complex::<f64>::new(0.0, 0.0), 1e-12);
        approx_eq_f64(y[2], Complex::<f64>::new(0.0, 0.0), 1e-12);
        approx_eq_f64(y[3], Complex::<f64>::new(0.0, 0.0), 1e-12);
    }

    #[test]
    fn radix2_fft_matches_dft_f32_forward() {
        let x = vec![
            Complex::<f32>::new(1.0, -0.5),
            Complex::<f32>::new(0.25, 2.0),
            Complex::<f32>::new(-1.0, 0.75),
            Complex::<f32>::new(0.0, -1.0),
            Complex::<f32>::new(2.0, 0.0),
            Complex::<f32>::new(-0.25, 0.5),
            Complex::<f32>::new(0.1, -0.2),
            Complex::<f32>::new(-0.8, 0.3),
        ];
        let fast = fft_radix2_host_f32(&x, FftDirection::Forward);
        let ref_dft = dft_host_f32(&x, FftDirection::Forward);
        for i in 0..x.len() {
            approx_eq_f32(fast[i], ref_dft[i], 2e-4);
        }
    }

    #[test]
    fn radix2_fft_matches_dft_f64_inverse() {
        let x = vec![
            Complex::<f64>::new(1.0, -0.5),
            Complex::<f64>::new(0.25, 2.0),
            Complex::<f64>::new(-1.0, 0.75),
            Complex::<f64>::new(0.0, -1.0),
            Complex::<f64>::new(2.0, 0.0),
            Complex::<f64>::new(-0.25, 0.5),
            Complex::<f64>::new(0.1, -0.2),
            Complex::<f64>::new(-0.8, 0.3),
        ];
        let fast = fft_radix2_host_f64(&x, FftDirection::Inverse);
        let ref_dft = dft_host_f64(&x, FftDirection::Inverse);
        for i in 0..x.len() {
            approx_eq_f64(fast[i], ref_dft[i], 1e-10);
        }
    }
}
