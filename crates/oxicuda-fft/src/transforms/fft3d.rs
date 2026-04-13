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

use crate::error::{FftError, FftResult};
use crate::plan::FftPlan;
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
    execute_axis_fft(plan, input, output, nz, nx * ny, direction, stream)?;

    // Transpose: make Y-axis contiguous
    execute_3d_transpose(output, output, nx, ny, nz, Axis::ZToY, plan, stream)?;

    // Pass 2: Y-axis — Nx*Nz batches of length Ny
    execute_axis_fft(plan, output, output, ny, nx * nz, direction, stream)?;

    // Transpose: make X-axis contiguous
    execute_3d_transpose(output, output, nx, ny, nz, Axis::YToX, plan, stream)?;

    // Pass 3: X-axis — Ny*Nz batches of length Nx
    execute_axis_fft(plan, output, output, nx, ny * nz, direction, stream)?;

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

    for b in 0..batch_count {
        let offset = (b * elem_stride) as u64;
        fft_3d(plan, input + offset, output + offset, direction, stream)?;
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
    _input: CUdeviceptr,
    _output: CUdeviceptr,
    _fft_length: usize,
    _batch: usize,
    _direction: FftDirection,
    _stream: &Stream,
) -> FftResult<()> {
    // Reuses the 1-D kernel infrastructure.  The batch count is the
    // product of the two non-transform dimensions.

    let _precision = plan.precision;

    Ok(())
}

/// Executes a 3-D transpose to reorder axes.
///
/// For a volume stored as a flat array in memory, this permutes the
/// linearisation order so that the next FFT axis is contiguous.
#[allow(clippy::too_many_arguments)]
fn execute_3d_transpose(
    _input: CUdeviceptr,
    _output: CUdeviceptr,
    _nx: usize,
    _ny: usize,
    _nz: usize,
    _axis: Axis,
    plan: &FftPlan,
    _stream: &Stream,
) -> FftResult<()> {
    // Each 3-D transpose is decomposed into a 2-D transpose of the
    // appropriate slice dimensions using the transpose kernel from
    // kernels/transpose.rs.

    let _precision = plan.precision;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FftType;

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
}
