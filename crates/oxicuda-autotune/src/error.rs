//! Error types for the autotune engine.
//!
//! [`AutotuneError`] covers all failure modes encountered during
//! configuration search, benchmarking, and result persistence.

/// Error type for autotuning operations.
///
/// This enum captures every class of failure that the tuning engine
/// can encounter: driver-level CUDA errors, benchmark execution
/// failures, missing viable configurations, I/O problems when
/// persisting results, and serialization issues.
#[derive(Debug, thiserror::Error)]
pub enum AutotuneError {
    /// A CUDA driver call failed.
    #[error("CUDA error: {0}")]
    Cuda(#[from] oxicuda_driver::CudaError),

    /// A benchmark iteration failed with a descriptive message.
    #[error("benchmark failed: {0}")]
    BenchmarkFailed(String),

    /// The search space was exhausted without finding any valid configuration.
    #[error("no viable configuration found")]
    NoViableConfig,

    /// An I/O error occurred (e.g. reading/writing the result database).
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// A serialization or deserialization error occurred.
    #[error("serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    /// PTX code generation failed.
    #[error("PTX generation error: {0}")]
    PtxError(String),
}

/// Convenience alias for `Result<T, AutotuneError>`.
pub type AutotuneResult<T> = Result<T, AutotuneError>;
