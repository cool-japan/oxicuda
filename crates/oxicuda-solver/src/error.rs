//! Error types for OxiCUDA Solver operations.
//!
//! Provides [`SolverError`] covering all failure modes for GPU-accelerated
//! matrix decompositions and linear solvers — singular matrices, convergence
//! failures, workspace issues, and underlying CUDA/BLAS/PTX errors.

use oxicuda_blas::BlasError;
use oxicuda_driver::CudaError;
use oxicuda_ptx::PtxGenError;
use thiserror::Error;

/// Solver-specific error type.
///
/// Every fallible solver operation returns [`SolverResult<T>`] which uses this
/// enum as its error variant. The variants are ordered by failure mode:
/// upstream errors first, then solver-specific conditions.
#[derive(Debug, Error)]
pub enum SolverError {
    /// A CUDA driver call failed.
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] CudaError),

    /// A BLAS operation failed.
    #[error("BLAS error: {0}")]
    Blas(#[from] BlasError),

    /// PTX kernel source generation failed.
    #[error("PTX generation error: {0}")]
    PtxGeneration(#[from] PtxGenError),

    /// The matrix is singular (pivot is exactly zero or numerically zero).
    #[error("singular matrix detected")]
    SingularMatrix,

    /// The matrix is not positive definite (Cholesky decomposition failed).
    #[error("matrix is not positive definite")]
    NotPositiveDefinite,

    /// Operand dimensions are incompatible.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// An iterative solver failed to converge within the allowed iterations.
    #[error("convergence failure after {iterations} iterations (residual = {residual:.6e})")]
    ConvergenceFailure {
        /// Number of iterations performed before giving up.
        iterations: u32,
        /// The residual norm at termination.
        residual: f64,
    },

    /// The operation requires a workspace of at least the specified size.
    #[error("workspace of at least {0} bytes required")]
    WorkspaceRequired(usize),

    /// An internal logic error that should not occur under normal conditions.
    #[error("internal solver error: {0}")]
    InternalError(String),
}

/// Convenience alias for solver operations.
pub type SolverResult<T> = Result<T, SolverError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_singular_matrix() {
        let err = SolverError::SingularMatrix;
        assert!(err.to_string().contains("singular"));
    }

    #[test]
    fn display_convergence_failure() {
        let err = SolverError::ConvergenceFailure {
            iterations: 100,
            residual: 1e-3,
        };
        let msg = err.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("1.0"));
    }

    #[test]
    fn from_cuda_error() {
        let cuda_err = CudaError::NotInitialized;
        let solver_err: SolverError = cuda_err.into();
        assert!(matches!(solver_err, SolverError::Cuda(_)));
    }

    #[test]
    fn from_blas_error() {
        let blas_err = BlasError::InvalidDimension("test".into());
        let solver_err: SolverError = blas_err.into();
        assert!(matches!(solver_err, SolverError::Blas(_)));
    }

    #[test]
    fn display_workspace_required() {
        let err = SolverError::WorkspaceRequired(4096);
        assert!(err.to_string().contains("4096"));
    }
}
