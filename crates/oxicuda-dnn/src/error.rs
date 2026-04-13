//! Error types for the DNN crate.
//!
//! [`DnnError`] unifies failures from the CUDA driver, BLAS layer, PTX
//! generation, and DNN-specific validation into a single error enum.
//! All public DNN functions return [`DnnResult<T>`].

use oxicuda_blas::BlasError;
use oxicuda_driver::CudaError;
use thiserror::Error;

/// Errors that can arise during DNN operations.
#[derive(Debug, Error)]
pub enum DnnError {
    /// A low-level CUDA driver call failed.
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] CudaError),

    /// A BLAS operation failed (e.g. during convolution via im2col + GEMM).
    #[error("BLAS error: {0}")]
    Blas(#[from] BlasError),

    /// PTX kernel generation or compilation failed.
    #[error("PTX generation error: {0}")]
    PtxGeneration(String),

    /// Tensor dimensions are invalid (e.g. zero-sized, negative, or
    /// inconsistent with the expected layout).
    #[error("invalid tensor dimensions: {0}")]
    InvalidDimension(String),

    /// The supplied device buffer is too small for the described tensor.
    #[error("buffer too small: expected {expected} bytes, got {actual} bytes")]
    BufferTooSmall {
        /// Minimum required buffer size in bytes.
        expected: usize,
        /// Actual buffer size in bytes.
        actual: usize,
    },

    /// The requested operation is not supported for the given configuration
    /// (e.g. unsupported data type or algorithm on a particular SM version).
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// A function argument failed validation.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The operation requires a workspace buffer of at least the given size,
    /// but none (or too small) was provided.
    #[error("workspace required: need at least {0} bytes")]
    WorkspaceRequired(usize),

    /// A GPU kernel launch failed.
    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),

    /// An I/O error occurred (e.g. when reading/writing the PTX cache).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<oxicuda_ptx::PtxGenError> for DnnError {
    fn from(e: oxicuda_ptx::PtxGenError) -> Self {
        Self::PtxGeneration(e.to_string())
    }
}

/// Convenience alias for `Result<T, DnnError>`.
pub type DnnResult<T> = Result<T, DnnError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_buffer_too_small() {
        let e = DnnError::BufferTooSmall {
            expected: 4096,
            actual: 1024,
        };
        assert!(e.to_string().contains("4096"));
        assert!(e.to_string().contains("1024"));
    }

    #[test]
    fn display_workspace_required() {
        let e = DnnError::WorkspaceRequired(8192);
        assert!(e.to_string().contains("8192"));
    }

    #[test]
    fn from_cuda_error() {
        let cuda_err = CudaError::InvalidValue;
        let dnn_err: DnnError = cuda_err.into();
        assert!(matches!(dnn_err, DnnError::Cuda(_)));
    }
}
