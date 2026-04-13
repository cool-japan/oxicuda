//! Error types for OxiCUDA BLAS operations.
//!
//! Provides [`BlasError`] covering all failure modes for GPU-accelerated
//! BLAS routines — dimension mismatches, buffer validation, PTX generation,
//! kernel launch issues, and underlying CUDA driver errors.

use oxicuda_driver::CudaError;
use thiserror::Error;

/// BLAS-specific error type.
///
/// Every fallible BLAS operation returns [`BlasResult<T>`] which uses this
/// enum as its error variant. The variants are ordered roughly by how early
/// in the call chain they are likely to appear: argument validation first,
/// then PTX/launch errors, then driver-level failures.
#[derive(Debug, Error)]
pub enum BlasError {
    /// A CUDA driver call failed.
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] CudaError),

    /// A matrix or vector dimension is invalid (e.g. zero rows).
    #[error("invalid matrix dimensions: {0}")]
    InvalidDimension(String),

    /// A device buffer is too small for the requested operation.
    #[error("buffer too small: expected at least {expected} elements, got {actual}")]
    BufferTooSmall {
        /// Minimum number of elements required.
        expected: usize,
        /// Actual number of elements in the buffer.
        actual: usize,
    },

    /// Two operands have incompatible dimensions (e.g. inner dims of A and B
    /// in a GEMM do not match).
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// The requested operation or precision is not supported on this device.
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// PTX kernel source generation failed.
    #[error("PTX generation error: {0}")]
    PtxGeneration(String),

    /// A kernel launch (grid/block configuration or driver call) failed.
    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),

    /// A caller-provided argument is invalid.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The autotuner encountered an error while profiling kernel variants.
    #[error("autotuner error: {0}")]
    AutotuneError(String),
}

// -- Conversions from dependency error types -----------------------------------

impl From<oxicuda_ptx::PtxGenError> for BlasError {
    fn from(err: oxicuda_ptx::PtxGenError) -> Self {
        Self::PtxGeneration(err.to_string())
    }
}

/// Convenience alias used throughout the BLAS crate.
pub type BlasResult<T> = Result<T, BlasError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_buffer_too_small() {
        let err = BlasError::BufferTooSmall {
            expected: 1024,
            actual: 512,
        };
        assert!(err.to_string().contains("1024"));
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn display_dimension_mismatch() {
        let err = BlasError::DimensionMismatch("A.cols != B.rows".to_string());
        assert!(err.to_string().contains("A.cols != B.rows"));
    }

    #[test]
    fn from_cuda_error() {
        let cuda_err = CudaError::NotInitialized;
        let blas_err: BlasError = cuda_err.into();
        assert!(matches!(blas_err, BlasError::Cuda(_)));
    }
}
