//! Error types for the OxiCUDA training engine.

use thiserror::Error;

/// All errors produced by `oxicuda-train`.
#[derive(Debug, Error)]
pub enum TrainError {
    /// Underlying CUDA driver error.
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] oxicuda_driver::CudaError),

    /// Optimizer state has not been initialised yet.
    #[error("optimizer state not initialised — call init_states() or step() first")]
    StateNotInitialised,

    /// Parameter / gradient count mismatch between calls.
    #[error("parameter count mismatch: expected {expected}, got {got}")]
    ParamCountMismatch {
        /// Expected parameter count.
        expected: usize,
        /// Actual parameter count.
        got: usize,
    },

    /// A parameter's gradient is `None`.
    #[error("gradient is None for parameter at index {index}")]
    NoGradient {
        /// Zero-based index of the parameter without a gradient.
        index: usize,
    },

    /// Shape mismatch between two buffers.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },

    /// Gradient norm was NaN or Inf.
    #[error("gradient norm is NaN or Inf — training may have diverged")]
    InvalidGradNorm,

    /// Learning rate must be strictly positive.
    #[error("learning rate must be positive, got {lr}")]
    InvalidLearningRate {
        /// The invalid learning rate value.
        lr: f64,
    },

    /// Loss scale overflow: all gradients were Inf/NaN.
    #[error("loss scale overflow detected (scale={scale})")]
    LossScaleOverflow {
        /// The scale factor in use when overflow was detected.
        scale: f64,
    },

    /// Checkpoint segment count exceeded configured maximum.
    #[error("checkpoint buffer overflow: max_segments={max}")]
    CheckpointOverflow {
        /// Configured maximum number of segments.
        max: usize,
    },

    /// ZeRO rank out of range.
    #[error("ZeRO rank {rank} out of range (world_size={world_size})")]
    InvalidRank {
        /// The rank that was out of range.
        rank: usize,
        /// The total number of workers.
        world_size: usize,
    },

    /// An empty parameter list was passed to an operation that requires at
    /// least one parameter.
    #[error("empty parameter list")]
    EmptyParams,

    /// A required feature or algorithm variant is not yet supported.
    #[error("not supported: {msg}")]
    NotSupported {
        /// Description of the unsupported feature.
        msg: String,
    },

    /// Internal invariant violation.
    #[error("internal error: {msg}")]
    Internal {
        /// Internal error message.
        msg: String,
    },

    /// AMP GradScaler detected invalid operation state.
    #[error("AMP invalid state: {0}")]
    InvalidState(String),

    /// AMP scale has dropped to or below the minimum allowed value.
    #[error("AMP scale {0} is at or below minimum — training is unstable")]
    AmpMinScaleReached(f64),
}

/// Convenience alias for `Result<T, TrainError>`.
pub type TrainResult<T> = Result<T, TrainError>;
