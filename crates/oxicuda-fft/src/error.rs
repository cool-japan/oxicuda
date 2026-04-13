//! Error types for OxiCUDA FFT operations.
//!
//! Provides [`FftError`] covering all failure modes for GPU-accelerated
//! FFT routines -- invalid sizes, workspace requirements, PTX generation,
//! and underlying CUDA driver errors.

use oxicuda_driver::CudaError;
use oxicuda_ptx::error::PtxGenError;
use thiserror::Error;

/// FFT-specific error type.
///
/// Every fallible FFT operation returns [`FftResult<T>`] which uses this
/// enum as its error variant.
#[derive(Debug, Error)]
pub enum FftError {
    /// A CUDA driver call failed.
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] CudaError),

    /// PTX kernel generation failed.
    #[error("PTX generation error: {0}")]
    PtxGeneration(#[from] PtxGenError),

    /// The FFT size is invalid (zero, negative, or not factorizable).
    #[error("invalid FFT size: {0}")]
    InvalidSize(String),

    /// The batch count is invalid (must be >= 1).
    #[error("invalid batch count: {0}")]
    InvalidBatch(String),

    /// A temporary workspace buffer is required but was not provided.
    #[error("workspace required: {0} bytes")]
    WorkspaceRequired(usize),

    /// The requested transform type is not supported for the given
    /// configuration (e.g. C2R with odd size).
    #[error("unsupported transform: {0}")]
    UnsupportedTransform(String),

    /// An internal error that should not occur under normal operation.
    #[error("internal FFT error: {0}")]
    InternalError(String),
}

/// Convenience alias used throughout the FFT crate.
pub type FftResult<T> = Result<T, FftError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_invalid_size() {
        let err = FftError::InvalidSize("size must be > 0".to_string());
        assert!(err.to_string().contains("size must be > 0"));
    }

    #[test]
    fn display_workspace_required() {
        let err = FftError::WorkspaceRequired(1024);
        assert!(err.to_string().contains("1024"));
    }

    #[test]
    fn from_cuda_error() {
        let cuda_err = CudaError::NotInitialized;
        let fft_err: FftError = cuda_err.into();
        assert!(matches!(fft_err, FftError::Cuda(_)));
    }
}
