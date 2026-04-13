//! Error types for OxiCUDA Sparse operations.
//!
//! Provides [`SparseError`] covering all failure modes for GPU-accelerated
//! sparse matrix routines -- format validation, dimension mismatches, PTX
//! generation, and underlying CUDA / BLAS errors.

use oxicuda_blas::BlasError;
use oxicuda_driver::CudaError;
use oxicuda_ptx::PtxGenError;
use thiserror::Error;

/// Sparse-specific error type.
///
/// Every fallible sparse operation returns [`SparseResult<T>`] which uses this
/// enum as its error variant.
#[derive(Debug, Error)]
pub enum SparseError {
    /// A CUDA driver call failed.
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] CudaError),

    /// A BLAS operation failed.
    #[error("BLAS error: {0}")]
    Blas(#[from] BlasError),

    /// PTX kernel generation failed.
    #[error("PTX generation error: {0}")]
    PtxGeneration(String),

    /// The sparse matrix format is invalid (e.g. mismatched array lengths).
    #[error("invalid sparse format: {0}")]
    InvalidFormat(String),

    /// Matrix dimensions are incompatible for the requested operation.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// The number of non-zeros is zero, which is invalid for this operation.
    #[error("matrix has zero non-zeros")]
    ZeroNnz,

    /// The matrix is singular and cannot be factored or solved.
    #[error("singular matrix detected")]
    SingularMatrix,

    /// An iterative algorithm failed to converge within the iteration limit.
    #[error("convergence failure: {0}")]
    ConvergenceFailure(String),

    /// An internal logic error (should not happen in correct code).
    #[error("internal error: {0}")]
    InternalError(String),

    /// A function argument is invalid.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// An I/O operation failed (e.g. PTX cache).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<PtxGenError> for SparseError {
    fn from(err: PtxGenError) -> Self {
        Self::PtxGeneration(err.to_string())
    }
}

/// Convenience alias used throughout the sparse crate.
pub type SparseResult<T> = Result<T, SparseError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_zero_nnz() {
        let err = SparseError::ZeroNnz;
        assert!(err.to_string().contains("zero non-zeros"));
    }

    #[test]
    fn display_dimension_mismatch() {
        let err = SparseError::DimensionMismatch("A.cols != B.rows".to_string());
        assert!(err.to_string().contains("A.cols != B.rows"));
    }

    #[test]
    fn from_cuda_error() {
        let cuda_err = CudaError::NotInitialized;
        let sparse_err: SparseError = cuda_err.into();
        assert!(matches!(sparse_err, SparseError::Cuda(_)));
    }
}
