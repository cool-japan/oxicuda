//! Error types for OxiCUDA Rand operations.
//!
//! Provides [`RandError`] covering all failure modes for GPU-accelerated
//! random number generation -- seed validation, size checks, PTX generation,
//! kernel launch issues, and underlying CUDA driver errors.

use oxicuda_driver::CudaError;
use oxicuda_ptx::error::PtxGenError;
use thiserror::Error;

/// RNG-specific error type.
///
/// Every fallible RNG operation returns [`RandResult<T>`] which uses this
/// enum as its error variant.
#[derive(Debug, Error)]
pub enum RandError {
    /// A CUDA driver call failed.
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] CudaError),

    /// PTX kernel source generation failed.
    #[error("PTX generation error: {0}")]
    PtxGeneration(#[from] PtxGenError),

    /// The requested output size is invalid (e.g., zero elements, or not
    /// a multiple of the engine's natural output width).
    #[error("invalid output size: {0}")]
    InvalidSize(String),

    /// The provided seed value is invalid.
    #[error("invalid seed: {0}")]
    InvalidSeed(String),

    /// One or more distribution parameters are invalid.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// The requested distribution is not supported for this engine or
    /// precision combination.
    #[error("unsupported distribution: {0}")]
    UnsupportedDistribution(String),

    /// An internal error that should not occur under normal operation.
    #[error("internal error: {0}")]
    InternalError(String),
}

/// Convenience alias used throughout the rand crate.
pub type RandResult<T> = Result<T, RandError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_invalid_size() {
        let err = RandError::InvalidSize("must be a multiple of 4".to_string());
        assert!(err.to_string().contains("multiple of 4"));
    }

    #[test]
    fn display_unsupported_distribution() {
        let err = RandError::UnsupportedDistribution("poisson f64".to_string());
        assert!(err.to_string().contains("poisson f64"));
    }

    #[test]
    fn display_invalid_parameter() {
        let err = RandError::InvalidParameter("lambda must be >= 0".to_string());
        assert!(err.to_string().contains("lambda must be >= 0"));
    }

    #[test]
    fn from_cuda_error() {
        let cuda_err = CudaError::NotInitialized;
        let rand_err: RandError = cuda_err.into();
        assert!(matches!(rand_err, RandError::Cuda(_)));
    }
}
