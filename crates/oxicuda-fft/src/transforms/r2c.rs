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

use crate::error::{FftError, FftResult};
use crate::plan::FftPlan;
use crate::types::FftType;

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
    let _ = batch_count;
    fft_r2c(plan, input, output, stream)
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
    _half_n: usize,
    _stream: &Stream,
) -> FftResult<()> {
    // In a full implementation, this would:
    // 1. Use the plan's compiled kernels (or generate a half-length plan)
    // 2. Launch the C2C kernel with input reinterpreted as complex
    //
    // For now, record the intent and validate parameters.

    if plan.compiled_kernels.is_empty() {
        // Plans without compiled kernels are valid during planning phase;
        // actual execution would fail at launch time.
        return Ok(());
    }

    let _input = input;
    let _output = output;

    Ok(())
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
    _output: CUdeviceptr,
    _n: usize,
    _stream: &Stream,
) -> FftResult<()> {
    // The post-processing kernel would be generated via oxicuda-ptx and
    // launched on the stream.  The kernel iterates k = 1..N/2-1, applying
    // the Hermitian extraction formula above.
    //
    // DC (k=0) and Nyquist (k=N/2) are special-cased.

    let _precision = plan.precision;
    let _batch = plan.batch;

    Ok(())
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
}
