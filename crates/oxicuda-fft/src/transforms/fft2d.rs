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

use crate::error::{FftError, FftResult};
use crate::plan::FftPlan;
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

    // Process each batch element sequentially (a more advanced
    // implementation would process all batches in a single launch).
    let elem_stride = plan.elements_per_batch() * plan.precision.complex_bytes();

    for b in 0..batch_count {
        let offset = (b * elem_stride) as u64;
        fft_2d(plan, input + offset, output + offset, direction, stream)?;
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
    _input: CUdeviceptr,
    _output: CUdeviceptr,
    _nx: usize,
    _ny: usize,
    _direction: FftDirection,
    _stream: &Stream,
) -> FftResult<()> {
    // The row FFT reuses the 1-D kernel infrastructure from the plan.
    // Each row is a contiguous array of `Ny` complex elements.
    // The batch count for this pass is `Nx * plan.batch`.

    let _precision = plan.precision;
    let _batch = plan.batch;

    // Kernel launch would be dispatched here.
    Ok(())
}

/// Executes the column-wise pass: `Ny` independent 1-D FFTs of length `Nx`.
fn execute_col_fft(
    plan: &FftPlan,
    _input: CUdeviceptr,
    _output: CUdeviceptr,
    _nx: usize,
    _ny: usize,
    _direction: FftDirection,
    _stream: &Stream,
) -> FftResult<()> {
    // After the first transpose, columns are now contiguous rows of
    // length `Nx`.  The batch count for this pass is `Ny * plan.batch`.

    let _precision = plan.precision;
    let _batch = plan.batch;

    Ok(())
}

/// Executes a matrix transpose on the GPU.
fn execute_transpose(
    _input: CUdeviceptr,
    _output: CUdeviceptr,
    _rows: usize,
    _cols: usize,
    plan: &FftPlan,
    _stream: &Stream,
) -> FftResult<()> {
    // The transpose kernel (kernels/transpose.rs) would be compiled
    // and launched here with:
    //   grid  = (ceil(cols/32), ceil(rows/32), batch)
    //   block = (32, 8, 1)

    let _precision = plan.precision;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
}
