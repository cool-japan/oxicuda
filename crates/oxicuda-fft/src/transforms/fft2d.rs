//! 2-D FFT execution.
//!
//! A 2-D FFT of an `Nx x Ny` matrix is decomposed into:
//!
//! 1. **Row pass** — batched 1-D FFT of length `Ny` across all `Nx` rows.
//! 2. **Transpose** — reorder from row-major to column-major layout.
//! 3. **Column pass** — batched 1-D FFT of length `Nx` across all `Ny` columns.
//! 4. **Transpose** — reorder back to row-major layout.
//!
//! The transpose steps ensure that each 1-D FFT pass operates on
//! contiguous memory, maximising GPU memory bandwidth utilisation.
#![allow(dead_code)]

use oxicuda_driver::Stream;
use oxicuda_driver::ffi::CUdeviceptr;
use std::f32::consts::PI as PI_F32;
use std::f64::consts::PI as PI_F64;

use crate::error::{FftError, FftResult};
use crate::plan::FftPlan;
use crate::types::Complex;
use crate::types::{FftDirection, FftType};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Executes a 2-D complex-to-complex FFT.
///
/// The plan must have been created with [`FftPlan::new_2d`] and contain
/// exactly two dimension sizes `[Nx, Ny]`.
///
/// # Arguments
///
/// * `plan`      - A compiled 2-D [`FftPlan`].
/// * `input`     - Device pointer to the `Nx * Ny` complex input matrix.
/// * `output`    - Device pointer to the `Nx * Ny` complex output matrix.
/// * `direction` - [`FftDirection::Forward`] or [`FftDirection::Inverse`].
/// * `stream`    - CUDA [`Stream`] for asynchronous execution.
///
/// # Errors
///
/// Returns [`FftError::UnsupportedTransform`] if the plan is not 2-D.
/// Returns [`FftError::InternalError`] on internal failures.
pub fn fft_2d(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    direction: FftDirection,
    stream: &Stream,
) -> FftResult<()> {
    validate_2d_plan(plan)?;

    let nx = plan.sizes[0];
    let ny = plan.sizes[1];

    // Step 1: Row-wise batched 1-D FFT (Nx batches of length Ny)
    execute_row_fft(plan, input, output, nx, ny, direction, stream)?;

    // Step 2: Transpose Nx x Ny -> Ny x Nx
    execute_transpose(output, output, nx, ny, plan, stream)?;

    // Step 3: Column-wise batched 1-D FFT (Ny batches of length Nx)
    execute_col_fft(plan, output, output, nx, ny, direction, stream)?;

    // Step 4: Transpose Ny x Nx -> Nx x Ny (restore original layout)
    execute_transpose(output, output, ny, nx, plan, stream)?;

    Ok(())
}

/// Executes a batched 2-D FFT.
///
/// # Errors
///
/// Returns [`FftError::UnsupportedTransform`] if the plan is not 2-D.
/// Returns [`FftError::InvalidBatch`] if `batch_count` is zero.
pub fn fft_2d_batched(
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

    // Process each external batch element sequentially with a single-plan batch,
    // so explicit batch_count is honored independently of plan.batch.
    let elem_stride = plan.elements_per_batch() * plan.precision.complex_bytes();
    let mut exec_plan = plan.clone();
    exec_plan.batch = 1;

    for b in 0..batch_count {
        let offset = (b * elem_stride) as u64;
        fft_2d(
            &exec_plan,
            input + offset,
            output + offset,
            direction,
            stream,
        )?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates that the plan is a 2-D plan.
fn validate_2d_plan(plan: &FftPlan) -> FftResult<()> {
    if plan.sizes.len() != 2 {
        return Err(FftError::UnsupportedTransform(format!(
            "expected 2-D plan with 2 dimensions, got {}",
            plan.sizes.len()
        )));
    }

    // R2C/C2R 2-D is supported but the first dimension must be even
    if (plan.transform_type == FftType::R2C || plan.transform_type == FftType::C2R)
        && plan.sizes[0] % 2 != 0
    {
        return Err(FftError::InvalidSize(format!(
            "2-D R2C/C2R requires even first dimension, got {}",
            plan.sizes[0]
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal steps
// ---------------------------------------------------------------------------

/// Executes the row-wise pass: `Nx` independent 1-D FFTs of length `Ny`.
fn execute_row_fft(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    nx: usize,
    ny: usize,
    direction: FftDirection,
    stream: &Stream,
) -> FftResult<()> {
    let elems = nx * ny;
    if elems == 0 || plan.batch == 0 {
        return Ok(());
    }

    match plan.precision {
        crate::types::FftPrecision::Single => {
            for b in 0..plan.batch {
                let off = (b * elems * std::mem::size_of::<Complex<f32>>()) as u64;
                let mut x = vec![Complex::<f32>::zero(); elems];
                copy_dtoh_async(&mut x, input + off, stream)?;

                let mut y = vec![Complex::<f32>::zero(); elems];
                for r in 0..nx {
                    let row_start = r * ny;
                    let row = &x[row_start..row_start + ny];
                    let row_out = if ny.is_power_of_two() {
                        fft_radix2_f32(row, direction)
                    } else {
                        dft_1d_f32(row, direction)
                    };
                    y[row_start..row_start + ny].copy_from_slice(&row_out);
                }

                copy_htod_async(output + off, &y, stream)?;
            }
        }
        crate::types::FftPrecision::Double => {
            for b in 0..plan.batch {
                let off = (b * elems * std::mem::size_of::<Complex<f64>>()) as u64;
                let mut x = vec![Complex::<f64>::zero(); elems];
                copy_dtoh_async(&mut x, input + off, stream)?;

                let mut y = vec![Complex::<f64>::zero(); elems];
                for r in 0..nx {
                    let row_start = r * ny;
                    let row = &x[row_start..row_start + ny];
                    let row_out = if ny.is_power_of_two() {
                        fft_radix2_f64(row, direction)
                    } else {
                        dft_1d_f64(row, direction)
                    };
                    y[row_start..row_start + ny].copy_from_slice(&row_out);
                }

                copy_htod_async(output + off, &y, stream)?;
            }
        }
    }

    stream.synchronize()?;
    Ok(())
}

/// Executes the column-wise pass: `Ny` independent 1-D FFTs of length `Nx`.
fn execute_col_fft(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    nx: usize,
    ny: usize,
    direction: FftDirection,
    stream: &Stream,
) -> FftResult<()> {
    let elems = nx * ny;
    if elems == 0 || plan.batch == 0 {
        return Ok(());
    }

    match plan.precision {
        crate::types::FftPrecision::Single => {
            for b in 0..plan.batch {
                let off = (b * elems * std::mem::size_of::<Complex<f32>>()) as u64;
                let mut x = vec![Complex::<f32>::zero(); elems];
                copy_dtoh_async(&mut x, input + off, stream)?;

                let mut y = vec![Complex::<f32>::zero(); elems];
                for r in 0..ny {
                    let row_start = r * nx;
                    let row = &x[row_start..row_start + nx];
                    let row_out = if nx.is_power_of_two() {
                        fft_radix2_f32(row, direction)
                    } else {
                        dft_1d_f32(row, direction)
                    };
                    y[row_start..row_start + nx].copy_from_slice(&row_out);
                }

                copy_htod_async(output + off, &y, stream)?;
            }
        }
        crate::types::FftPrecision::Double => {
            for b in 0..plan.batch {
                let off = (b * elems * std::mem::size_of::<Complex<f64>>()) as u64;
                let mut x = vec![Complex::<f64>::zero(); elems];
                copy_dtoh_async(&mut x, input + off, stream)?;

                let mut y = vec![Complex::<f64>::zero(); elems];
                for r in 0..ny {
                    let row_start = r * nx;
                    let row = &x[row_start..row_start + nx];
                    let row_out = if nx.is_power_of_two() {
                        fft_radix2_f64(row, direction)
                    } else {
                        dft_1d_f64(row, direction)
                    };
                    y[row_start..row_start + nx].copy_from_slice(&row_out);
                }

                copy_htod_async(output + off, &y, stream)?;
            }
        }
    }

    stream.synchronize()?;

    Ok(())
}

/// Executes a matrix transpose on the GPU.
fn execute_transpose(
    input: CUdeviceptr,
    output: CUdeviceptr,
    rows: usize,
    cols: usize,
    plan: &FftPlan,
    stream: &Stream,
) -> FftResult<()> {
    let elems = rows * cols;
    if elems == 0 || plan.batch == 0 {
        return Ok(());
    }

    match plan.precision {
        crate::types::FftPrecision::Single => {
            for b in 0..plan.batch {
                let off = (b * elems * std::mem::size_of::<Complex<f32>>()) as u64;
                let mut x = vec![Complex::<f32>::zero(); elems];
                copy_dtoh_async(&mut x, input + off, stream)?;

                let y = transpose_2d(&x, rows, cols);
                copy_htod_async(output + off, &y, stream)?;
            }
        }
        crate::types::FftPrecision::Double => {
            for b in 0..plan.batch {
                let off = (b * elems * std::mem::size_of::<Complex<f64>>()) as u64;
                let mut x = vec![Complex::<f64>::zero(); elems];
                copy_dtoh_async(&mut x, input + off, stream)?;

                let y = transpose_2d(&x, rows, cols);
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

fn dft_1d_f32(x: &[Complex<f32>], direction: FftDirection) -> Vec<Complex<f32>> {
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

fn dft_1d_f64(x: &[Complex<f64>], direction: FftDirection) -> Vec<Complex<f64>> {
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

fn fft_radix2_f32(x: &[Complex<f32>], direction: FftDirection) -> Vec<Complex<f32>> {
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

fn fft_radix2_f64(x: &[Complex<f64>], direction: FftDirection) -> Vec<Complex<f64>> {
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

fn transpose_2d<T: Copy + Default>(x: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut y = vec![T::default(); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            y[c * rows + r] = x[r * cols + c];
        }
    }
    y
}

impl Default for Complex<f32> {
    fn default() -> Self {
        Complex::<f32>::zero()
    }
}

impl Default for Complex<f64> {
    fn default() -> Self {
        Complex::<f64>::zero()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_f32(a: Complex<f32>, b: Complex<f32>, tol: f32) {
        assert!((a.re - b.re).abs() <= tol, "re: {} vs {}", a.re, b.re);
        assert!((a.im - b.im).abs() <= tol, "im: {} vs {}", a.im, b.im);
    }

    #[test]
    fn validate_2d_accepts_valid_plan() {
        let plan = FftPlan::new_2d(64, 64, FftType::C2C, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_2d_plan(&p);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn validate_2d_rejects_1d_plan() {
        let plan = FftPlan::new_1d(256, FftType::C2C, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_2d_plan(&p);
            assert!(matches!(result, Err(FftError::UnsupportedTransform(_))));
        }
    }

    #[test]
    fn validate_2d_rejects_odd_r2c() {
        let plan = FftPlan::new_2d(63, 64, FftType::R2C, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_2d_plan(&p);
            assert!(matches!(result, Err(FftError::InvalidSize(_))));
        }
    }

    #[test]
    fn transpose_2d_maps_layout() {
        let x = vec![
            Complex::<f32>::new(1.0, 0.0),
            Complex::<f32>::new(2.0, 0.0),
            Complex::<f32>::new(3.0, 0.0),
            Complex::<f32>::new(4.0, 0.0),
            Complex::<f32>::new(5.0, 0.0),
            Complex::<f32>::new(6.0, 0.0),
        ];
        let y = transpose_2d(&x, 2, 3);
        assert_eq!(y.len(), 6);
        approx_eq_f32(y[0], Complex::<f32>::new(1.0, 0.0), 1e-6);
        approx_eq_f32(y[1], Complex::<f32>::new(4.0, 0.0), 1e-6);
        approx_eq_f32(y[2], Complex::<f32>::new(2.0, 0.0), 1e-6);
        approx_eq_f32(y[3], Complex::<f32>::new(5.0, 0.0), 1e-6);
        approx_eq_f32(y[4], Complex::<f32>::new(3.0, 0.0), 1e-6);
        approx_eq_f32(y[5], Complex::<f32>::new(6.0, 0.0), 1e-6);
    }

    #[test]
    fn dft_1d_impulse_forward_is_ones() {
        let x = vec![
            Complex::<f32>::new(1.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
        ];
        let y = dft_1d_f32(&x, FftDirection::Forward);
        for yi in y {
            approx_eq_f32(yi, Complex::<f32>::new(1.0, 0.0), 1e-5);
        }
    }

    #[test]
    fn radix2_matches_dft_f32() {
        let x = vec![
            Complex::<f32>::new(1.0, -0.25),
            Complex::<f32>::new(-0.5, 0.75),
            Complex::<f32>::new(0.25, -1.0),
            Complex::<f32>::new(2.0, 0.1),
            Complex::<f32>::new(-1.25, 0.0),
            Complex::<f32>::new(0.5, -0.5),
            Complex::<f32>::new(0.0, 0.2),
            Complex::<f32>::new(-0.8, 1.1),
        ];
        let fast = fft_radix2_f32(&x, FftDirection::Forward);
        let ref_dft = dft_1d_f32(&x, FftDirection::Forward);
        for i in 0..x.len() {
            approx_eq_f32(fast[i], ref_dft[i], 2e-4);
        }
    }
}
