//! Public GEMM API.
//!
//! Provides the [`gemm`] function — the primary entry point for performing
//! general matrix multiplication on the GPU:
//!
//! `C = alpha * op(A) * op(B) + beta * C`
//!
//! The function validates dimensions, constructs a [`GemmProblem`], and
//! delegates to the [`GemmDispatcher`] for kernel selection and launch.

use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{GpuFloat, MatrixDesc, MatrixDescMut, Transpose};

use super::gemm::dispatch::{GemmDispatcher, GemmProblem};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Performs a general matrix multiplication on the GPU.
///
/// Computes `C = alpha * op(A) * op(B) + beta * C`, where `op(X)` is `X`,
/// `X^T`, or `X^H` depending on the transpose flag.
///
/// # Type parameters
///
/// * `T` — element type (must implement [`GpuFloat`]).
///
/// # Arguments
///
/// * `handle` — the BLAS handle providing context, stream, and SM version.
/// * `trans_a` — transpose mode for matrix A.
/// * `trans_b` — transpose mode for matrix B.
/// * `alpha` — scaling factor for the product `op(A) * op(B)`.
/// * `a` — descriptor for input matrix A.
/// * `b` — descriptor for input matrix B.
/// * `beta` — scaling factor for the existing contents of C.
/// * `c` — descriptor for the output matrix C (read-write).
///
/// # Dimension rules
///
/// After applying the transpose modes:
/// - `op(A)` is M x K
/// - `op(B)` is K x N
/// - `C` must be M x N
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if any dimension is zero.
/// Returns [`BlasError::DimensionMismatch`] if the inner dimensions of
/// `op(A)` and `op(B)` do not agree, or if C does not have the right shape.
/// Returns other [`BlasError`] variants on PTX generation or launch failure.
///
/// # Example
///
/// ```rust,no_run
/// # use oxicuda_blas::level3::gemm_api::gemm;
/// # use oxicuda_blas::handle::BlasHandle;
/// # use oxicuda_blas::types::*;
/// # fn main() -> Result<(), oxicuda_blas::error::BlasError> {
/// # let handle: BlasHandle = unimplemented!();
/// # let a: MatrixDesc<f32> = unimplemented!();
/// # let b: MatrixDesc<f32> = unimplemented!();
/// # let mut c: MatrixDescMut<f32> = unimplemented!();
/// gemm(&handle, Transpose::NoTrans, Transpose::NoTrans, 1.0f32, &a, &b, 0.0f32, &mut c)?;
/// # Ok(())
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: GpuFloat>(
    handle: &BlasHandle,
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: T,
    a: &MatrixDesc<T>,
    b: &MatrixDesc<T>,
    beta: T,
    c: &mut MatrixDescMut<T>,
) -> BlasResult<()> {
    // Extract effective dimensions after transpose.
    let (m, k_a) = a.effective_dims(trans_a);
    let (k_b, n) = b.effective_dims(trans_b);

    // Validate non-zero dimensions.
    if m == 0 || n == 0 || k_a == 0 {
        return Err(BlasError::InvalidDimension(
            "GEMM dimensions must be non-zero".into(),
        ));
    }

    // Validate inner dimension agreement.
    if k_a != k_b {
        return Err(BlasError::DimensionMismatch(format!(
            "inner dimensions of op(A) ({k_a}) and op(B) ({k_b}) do not match"
        )));
    }
    let k = k_a;

    // Validate output matrix dimensions.
    if c.rows != m || c.cols != n {
        return Err(BlasError::DimensionMismatch(format!(
            "C is {}x{} but GEMM produces {}x{}",
            c.rows, c.cols, m, n
        )));
    }

    // Build the problem description.
    let problem = GemmProblem {
        m,
        n,
        k,
        trans_a,
        trans_b,
        input_type: T::PTX_TYPE,
        output_type: accumulator_ptx_type::<T>(),
        math_mode: handle.math_mode(),
    };

    // Create (or reuse) the dispatcher.
    let dispatcher = GemmDispatcher::new(handle.sm_version());

    // Convert scalar arguments to bit representation for the kernel.
    let alpha_bits = alpha.to_bits_u64();
    let beta_bits = beta.to_bits_u64();

    dispatcher.dispatch(
        &problem,
        a.ptr,
        b.ptr,
        c.ptr,
        alpha_bits,
        beta_bits,
        handle.stream(),
    )
}

/// Returns the PTX type of the accumulator for type `T`.
///
/// For half-precision types the accumulator is F32; for F32 it's F32;
/// for F64 it's F64.
fn accumulator_ptx_type<T: GpuFloat>() -> PtxType {
    <T::Accumulator as GpuFloat>::PTX_TYPE
}

// ---------------------------------------------------------------------------
// Dimension helpers (useful for callers)
// ---------------------------------------------------------------------------

/// Computes the expected output dimensions for a GEMM operation.
///
/// Returns `(M, N)` — the dimensions of the output matrix C after
/// applying the transpose modes to A (rows x K) and B (K x cols).
pub fn gemm_output_dims(
    a_rows: u32,
    a_cols: u32,
    trans_a: Transpose,
    b_rows: u32,
    b_cols: u32,
    trans_b: Transpose,
) -> (u32, u32) {
    let m = match trans_a {
        Transpose::NoTrans => a_rows,
        Transpose::Trans | Transpose::ConjTrans => a_cols,
    };
    let n = match trans_b {
        Transpose::NoTrans => b_cols,
        Transpose::Trans | Transpose::ConjTrans => b_rows,
    };
    (m, n)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_dims_no_trans() {
        let (m, n) = gemm_output_dims(128, 64, Transpose::NoTrans, 64, 256, Transpose::NoTrans);
        assert_eq!((m, n), (128, 256));
    }

    #[test]
    fn output_dims_trans_a() {
        let (m, n) = gemm_output_dims(64, 128, Transpose::Trans, 128, 256, Transpose::NoTrans);
        assert_eq!((m, n), (128, 256));
    }

    #[test]
    fn output_dims_trans_b() {
        let (m, n) = gemm_output_dims(128, 64, Transpose::NoTrans, 256, 64, Transpose::Trans);
        assert_eq!((m, n), (128, 256));
    }

    #[test]
    fn output_dims_both_trans() {
        let (m, n) = gemm_output_dims(64, 128, Transpose::Trans, 256, 128, Transpose::Trans);
        assert_eq!((m, n), (128, 256));
    }

    #[test]
    fn accumulator_ptx_type_f32() {
        assert_eq!(accumulator_ptx_type::<f32>(), PtxType::F32);
    }

    #[test]
    fn accumulator_ptx_type_f64() {
        assert_eq!(accumulator_ptx_type::<f64>(), PtxType::F64);
    }
}
