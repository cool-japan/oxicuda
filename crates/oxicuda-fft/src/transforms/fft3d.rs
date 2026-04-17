//! 3-D FFT execution.
//!
//! A 3-D FFT of an `Nx x Ny x Nz` volume is decomposed into three passes
//! of 1-D FFTs along each axis, with transpose operations between passes
//! to ensure contiguous memory access:
//!
//! 1. **Z-axis pass** — batched 1-D FFT of length `Nz` across `Nx * Ny` slices.
//! 2. **Transpose** — reorder so Y-axis data is contiguous.
//! 3. **Y-axis pass** — batched 1-D FFT of length `Ny` across `Nx * Nz` slices.
//! 4. **Transpose** — reorder so X-axis data is contiguous.
//! 5. **X-axis pass** — batched 1-D FFT of length `Nx` across `Ny * Nz` slices.
//! 6. **Transpose** — restore original `Nx x Ny x Nz` layout.
#![allow(dead_code)]

use oxicuda_driver::Stream;
use oxicuda_driver::ffi::CUdeviceptr;
use std::f32::consts::PI as PI_F32;
use std::f64::consts::PI as PI_F64;

use crate::error::{FftError, FftResult};
use crate::plan::FftPlan;
use crate::types::Complex;
use crate::types::FftDirection;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Executes a 3-D complex-to-complex FFT.
///
/// The plan must have been created with [`FftPlan::new_3d`] and contain
/// exactly three dimension sizes `[Nx, Ny, Nz]`.
///
/// # Arguments
///
/// * `plan`      - A compiled 3-D [`FftPlan`].
/// * `input`     - Device pointer to the `Nx * Ny * Nz` complex input.
/// * `output`    - Device pointer to the `Nx * Ny * Nz` complex output.
/// * `direction` - [`FftDirection::Forward`] or [`FftDirection::Inverse`].
/// * `stream`    - CUDA [`Stream`] for asynchronous execution.
///
/// # Errors
///
/// Returns [`FftError::UnsupportedTransform`] if the plan is not 3-D.
pub fn fft_3d(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    direction: FftDirection,
    stream: &Stream,
) -> FftResult<()> {
    validate_3d_plan(plan)?;

    let nx = plan.sizes[0];
    let ny = plan.sizes[1];
    let nz = plan.sizes[2];

    // Pass 1: Z-axis — Nx*Ny batches of length Nz
    execute_axis_fft(
        plan,
        input,
        output,
        nz,
        nx * ny * plan.batch,
        direction,
        stream,
    )?;

    // Transpose: make Y-axis contiguous
    execute_3d_transpose(output, output, nx, ny, nz, Axis::ZToY, plan, stream)?;

    // Pass 2: Y-axis — Nx*Nz batches of length Ny
    execute_axis_fft(
        plan,
        output,
        output,
        ny,
        nx * nz * plan.batch,
        direction,
        stream,
    )?;

    // Transpose: make X-axis contiguous
    execute_3d_transpose(output, output, nx, ny, nz, Axis::YToX, plan, stream)?;

    // Pass 3: X-axis — Ny*Nz batches of length Nx
    execute_axis_fft(
        plan,
        output,
        output,
        nx,
        ny * nz * plan.batch,
        direction,
        stream,
    )?;

    // Final transpose: restore Nx x Ny x Nz layout
    execute_3d_transpose(output, output, nx, ny, nz, Axis::XToOriginal, plan, stream)?;

    Ok(())
}

/// Executes a batched 3-D FFT.
///
/// # Errors
///
/// Returns [`FftError::UnsupportedTransform`] if the plan is not 3-D.
/// Returns [`FftError::InvalidBatch`] if `batch_count` is zero.
pub fn fft_3d_batched(
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

    let elem_stride = plan.elements_per_batch() * plan.precision.complex_bytes();
    let mut exec_plan = plan.clone();
    exec_plan.batch = 1;

    for b in 0..batch_count {
        let offset = (b * elem_stride) as u64;
        fft_3d(
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

/// Validates that the plan is a 3-D plan.
fn validate_3d_plan(plan: &FftPlan) -> FftResult<()> {
    if plan.sizes.len() != 3 {
        return Err(FftError::UnsupportedTransform(format!(
            "expected 3-D plan with 3 dimensions, got {}",
            plan.sizes.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Axis descriptors
// ---------------------------------------------------------------------------

/// Describes which axis reordering a 3-D transpose performs.
#[derive(Debug, Clone, Copy)]
enum Axis {
    /// Reorder from Z-contiguous to Y-contiguous.
    ZToY,
    /// Reorder from Y-contiguous to X-contiguous.
    YToX,
    /// Restore original `Nx x Ny x Nz` (row-major) layout.
    XToOriginal,
}

// ---------------------------------------------------------------------------
// Internal steps
// ---------------------------------------------------------------------------

/// Executes a 1-D FFT along a single axis of the 3-D volume.
///
/// `fft_length` is the size of the 1-D FFT, and `batch` is the total
/// number of independent 1-D FFTs to perform (product of the other two
/// dimensions times the plan's batch count).
fn execute_axis_fft(
    plan: &FftPlan,
    input: CUdeviceptr,
    output: CUdeviceptr,
    fft_length: usize,
    batch: usize,
    direction: FftDirection,
    stream: &Stream,
) -> FftResult<()> {
    if fft_length == 0 || batch == 0 {
        return Ok(());
    }

    let elems = fft_length * batch;
    match plan.precision {
        crate::types::FftPrecision::Single => {
            let mut x = vec![Complex::<f32>::zero(); elems];
            copy_dtoh_async(&mut x, input, stream)?;
            let mut y = vec![Complex::<f32>::zero(); elems];

            for seg in 0..batch {
                let start = seg * fft_length;
                let slice = &x[start..start + fft_length];
                let out = if fft_length.is_power_of_two() {
                    fft_radix2_f32(slice, direction)
                } else {
                    dft_1d_f32(slice, direction)
                };
                y[start..start + fft_length].copy_from_slice(&out);
            }

            copy_htod_async(output, &y, stream)?;
        }
        crate::types::FftPrecision::Double => {
            let mut x = vec![Complex::<f64>::zero(); elems];
            copy_dtoh_async(&mut x, input, stream)?;
            let mut y = vec![Complex::<f64>::zero(); elems];

            for seg in 0..batch {
                let start = seg * fft_length;
                let slice = &x[start..start + fft_length];
                let out = if fft_length.is_power_of_two() {
                    fft_radix2_f64(slice, direction)
                } else {
                    dft_1d_f64(slice, direction)
                };
                y[start..start + fft_length].copy_from_slice(&out);
            }

            copy_htod_async(output, &y, stream)?;
        }
    }

    stream.synchronize()?;

    Ok(())
}

/// Executes a 3-D transpose to reorder axes.
///
/// For a volume stored as a flat array in memory, this permutes the
/// linearisation order so that the next FFT axis is contiguous.
#[allow(clippy::too_many_arguments)]
fn execute_3d_transpose(
    input: CUdeviceptr,
    output: CUdeviceptr,
    nx: usize,
    ny: usize,
    nz: usize,
    axis: Axis,
    plan: &FftPlan,
    stream: &Stream,
) -> FftResult<()> {
    let elems_per_batch = nx * ny * nz;
    if elems_per_batch == 0 || plan.batch == 0 {
        return Ok(());
    }

    match plan.precision {
        crate::types::FftPrecision::Single => {
            for b in 0..plan.batch {
                let off = (b * elems_per_batch * std::mem::size_of::<Complex<f32>>()) as u64;
                let mut x = vec![Complex::<f32>::zero(); elems_per_batch];
                copy_dtoh_async(&mut x, input + off, stream)?;
                let y = permute_3d_f32(&x, nx, ny, nz, axis);
                copy_htod_async(output + off, &y, stream)?;
            }
        }
        crate::types::FftPrecision::Double => {
            for b in 0..plan.batch {
                let off = (b * elems_per_batch * std::mem::size_of::<Complex<f64>>()) as u64;
                let mut x = vec![Complex::<f64>::zero(); elems_per_batch];
                copy_dtoh_async(&mut x, input + off, stream)?;
                let y = permute_3d_f64(&x, nx, ny, nz, axis);
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

fn idx_xyz(_nx: usize, ny: usize, nz: usize, x: usize, y: usize, z: usize) -> usize {
    (x * ny + y) * nz + z
}

fn idx_xzy(_nx: usize, ny: usize, nz: usize, x: usize, z: usize, y: usize) -> usize {
    (x * nz + z) * ny + y
}

fn idx_yzx(nx: usize, _ny: usize, nz: usize, y: usize, z: usize, x: usize) -> usize {
    (y * nz + z) * nx + x
}

fn permute_3d_f32(
    data: &[Complex<f32>],
    nx: usize,
    ny: usize,
    nz: usize,
    axis: Axis,
) -> Vec<Complex<f32>> {
    let mut out = vec![Complex::<f32>::zero(); data.len()];
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let src_xyz = idx_xyz(nx, ny, nz, x, y, z);
                let src_xzy = idx_xzy(nx, ny, nz, x, z, y);
                let src_yzx = idx_yzx(nx, ny, nz, y, z, x);
                match axis {
                    Axis::ZToY => {
                        out[src_xzy] = data[src_xyz];
                    }
                    Axis::YToX => {
                        out[src_yzx] = data[src_xzy];
                    }
                    Axis::XToOriginal => {
                        out[src_xyz] = data[src_yzx];
                    }
                }
            }
        }
    }
    out
}

fn permute_3d_f64(
    data: &[Complex<f64>],
    nx: usize,
    ny: usize,
    nz: usize,
    axis: Axis,
) -> Vec<Complex<f64>> {
    let mut out = vec![Complex::<f64>::zero(); data.len()];
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let src_xyz = idx_xyz(nx, ny, nz, x, y, z);
                let src_xzy = idx_xzy(nx, ny, nz, x, z, y);
                let src_yzx = idx_yzx(nx, ny, nz, y, z, x);
                match axis {
                    Axis::ZToY => {
                        out[src_xzy] = data[src_xyz];
                    }
                    Axis::YToX => {
                        out[src_yzx] = data[src_xzy];
                    }
                    Axis::XToOriginal => {
                        out[src_xyz] = data[src_yzx];
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FftType;

    fn approx_eq_f32(a: Complex<f32>, b: Complex<f32>, tol: f32) {
        assert!((a.re - b.re).abs() <= tol, "re: {} vs {}", a.re, b.re);
        assert!((a.im - b.im).abs() <= tol, "im: {} vs {}", a.im, b.im);
    }

    #[test]
    fn validate_3d_accepts_valid_plan() {
        let plan = FftPlan::new_3d(32, 32, 32, FftType::C2C, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_3d_plan(&p);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn validate_3d_rejects_2d_plan() {
        let plan = FftPlan::new_2d(64, 64, FftType::C2C, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_3d_plan(&p);
            assert!(matches!(result, Err(FftError::UnsupportedTransform(_))));
        }
    }

    #[test]
    fn validate_3d_rejects_1d_plan() {
        let plan = FftPlan::new_1d(256, FftType::C2C, 1);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let result = validate_3d_plan(&p);
            assert!(matches!(result, Err(FftError::UnsupportedTransform(_))));
        }
    }

    #[test]
    fn axis_permutations_roundtrip_f32() {
        let nx = 2;
        let ny = 2;
        let nz = 3;
        let mut xyz = Vec::with_capacity(nx * ny * nz);
        for i in 0..(nx * ny * nz) {
            xyz.push(Complex::<f32>::new(i as f32, 0.0));
        }

        let xzy = permute_3d_f32(&xyz, nx, ny, nz, Axis::ZToY);
        let yzx = permute_3d_f32(&xzy, nx, ny, nz, Axis::YToX);
        let back = permute_3d_f32(&yzx, nx, ny, nz, Axis::XToOriginal);

        assert_eq!(back.len(), xyz.len());
        for i in 0..xyz.len() {
            approx_eq_f32(back[i], xyz[i], 1e-6);
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
        let fast = fft_radix2_f32(&x, FftDirection::Inverse);
        let ref_dft = dft_1d_f32(&x, FftDirection::Inverse);
        for i in 0..x.len() {
            approx_eq_f32(fast[i], ref_dft[i], 2e-4);
        }
    }
}
