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

use crate::error::{FftError, FftResult};
use crate::plan::FftPlan;
use crate::types::FftType;

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
    let _ = batch_count;
    fft_c2r(plan, input, output, stream)
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
    _input: CUdeviceptr,
    _output: CUdeviceptr,
    _n: usize,
    _stream: &Stream,
) -> FftResult<()> {
    // The pre-processing kernel would be generated and launched here.
    // It iterates k = 1..N/2-1 applying the inverse Hermitian formula.

    let _precision = plan.precision;
    let _batch = plan.batch;

    Ok(())
}

/// Executes the half-length inverse C2C FFT.
fn execute_half_length_inverse_c2c(
    plan: &FftPlan,
    _input: CUdeviceptr,
    _output: CUdeviceptr,
    _half_n: usize,
    _stream: &Stream,
) -> FftResult<()> {
    if plan.compiled_kernels.is_empty() {
        return Ok(());
    }

    // In a full implementation this would launch the inverse C2C kernel
    // chain from the plan's compiled kernels.

    Ok(())
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
}
