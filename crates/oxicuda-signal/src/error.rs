//! Error types for `oxicuda-signal`.

use oxicuda_driver::CudaError;
use thiserror::Error;

/// Signal processing error variants.
#[derive(Debug, Error)]
pub enum SignalError {
    /// Underlying CUDA driver error.
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] CudaError),

    /// Invalid transform size (e.g. zero, or non-power-of-2 for DCT).
    #[error("Invalid transform size: {0}")]
    InvalidSize(String),

    /// Invalid parameter value (e.g. negative filter order).
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Transform size not supported on this architecture.
    #[error("Unsupported size {size} for operation '{op}'")]
    UnsupportedSize {
        /// Transform size.
        size: usize,
        /// Operation name.
        op: &'static str,
    },

    /// PTX generation or JIT compilation failed.
    #[error("PTX generation failed: {0}")]
    PtxGeneration(String),

    /// Dimension mismatch between buffers.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension description.
        expected: String,
        /// Actual dimension description.
        got: String,
    },

    /// Insufficient workspace memory.
    #[error("Insufficient workspace: need {need} bytes, have {have} bytes")]
    InsufficientWorkspace {
        /// Required bytes.
        need: usize,
        /// Available bytes.
        have: usize,
    },

    /// Operation not yet supported on this platform.
    #[error("Unsupported platform: {0}")]
    UnsupportedPlatform(String),
}

/// Convenience `Result` alias for signal processing operations.
pub type SignalResult<T> = Result<T, SignalError>;
