//! Error types for `oxicuda-rl`.

use thiserror::Error;

/// All errors produced by the OxiCUDA RL library.
#[derive(Debug, Error)]
pub enum RlError {
    /// Underlying CUDA driver error.
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] oxicuda_driver::CudaError),

    /// Replay buffer has zero capacity.
    #[error("replay buffer capacity must be > 0")]
    ZeroCapacity,

    /// Replay buffer does not yet contain enough transitions to sample.
    #[error("not enough transitions in buffer: have {have}, need {need}")]
    InsufficientTransitions {
        /// Number of transitions currently stored.
        have: usize,
        /// Number requested.
        need: usize,
    },

    /// Dimension mismatch between observation, action, or reward tensors.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        got: usize,
    },

    /// Invalid hyperparameter value.
    #[error("invalid hyperparameter `{name}`: {msg}")]
    InvalidHyperparameter {
        /// Parameter name.
        name: String,
        /// Description of why it is invalid.
        msg: String,
    },

    /// Policy returned a probability vector that does not sum to 1 within
    /// tolerance.
    #[error("invalid probability distribution: sum={sum:.6}, expected 1.0 ± {tol}")]
    InvalidDistribution {
        /// Actual sum of the probability vector.
        sum: f32,
        /// Tolerance used for the check.
        tol: f32,
    },

    /// Attempt to sample from an empty categorical distribution.
    #[error("cannot sample from empty categorical distribution")]
    EmptyDistribution,

    /// N-step buffer flush requested before enough steps were collected.
    #[error("n-step buffer incomplete: have {have} steps, need {need}")]
    NStepIncomplete {
        /// Steps currently accumulated.
        have: usize,
        /// Steps required (= n).
        need: usize,
    },

    /// Priority sum in segment tree dropped to zero (all priorities are zero).
    #[error("priority segment tree sum is zero — all priorities must be > 0")]
    ZeroPrioritySum,

    /// Feature not yet implemented.
    #[error("not supported: {0}")]
    NotSupported(String),

    /// Internal logic error — should never surface in correct code.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience alias.
pub type RlResult<T> = Result<T, RlError>;
