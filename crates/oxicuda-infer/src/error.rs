//! Error types for the OxiCUDA inference engine.

use thiserror::Error;

// ─── InferError ───────────────────────────────────────────────────────────────

/// All errors that can occur during inference engine operation.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum InferError {
    /// KV cache block allocation failed — not enough free blocks.
    #[error(
        "KV cache block allocation failed: needed {needed} blocks but only {available} available"
    )]
    BlockAllocFailed { needed: usize, available: usize },

    /// Operation on an already-finished sequence.
    #[error("sequence {0} has already finished")]
    SequenceFinished(u64),

    /// Unknown sequence identifier.
    #[error("invalid sequence id: {0}")]
    InvalidSequenceId(u64),

    /// Batch contains no sequences to process.
    #[error("batch is empty")]
    EmptyBatch,

    /// Sequence exceeded its configured maximum new-token count.
    #[error("sequence {seq_id} exceeded maximum length {max_len}")]
    MaxLengthExceeded { seq_id: u64, max_len: usize },

    /// Block table for a sequence reached its capacity limit.
    #[error("block table for sequence {seq_id} is full ({blocks} blocks allocated)")]
    BlockTableFull { seq_id: u64, blocks: usize },

    /// A sampling-related failure (e.g., empty distribution after filtering).
    #[error("sampling error: {0}")]
    SamplingError(String),

    /// Beam search internal failure.
    #[error("beam search failed: {0}")]
    BeamSearchError(String),

    /// Draft and target lengths mismatched in speculative decoding.
    #[error("speculative decoding length mismatch: expected {expected} positions, got {got}")]
    SpeculativeDecodingMismatch { expected: usize, got: usize },

    /// Cache manager operation failed.
    #[error("cache manager error: {0}")]
    CacheManagerError(String),

    /// Prefix cache operation failed.
    #[error("prefix cache error: {0}")]
    PrefixCacheError(String),

    /// Tensor dimension disagreement.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Bad configuration parameter.
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),

    /// NaN detected in model logits.
    #[error("NaN detected in model logits")]
    NanLogits,

    /// Other unclassified inference error.
    #[error("inference error: {0}")]
    Other(String),
}

/// Convenience alias for `Result<T, InferError>`.
pub type InferResult<T> = Result<T, InferError>;

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_alloc_failed_message() {
        let e = InferError::BlockAllocFailed {
            needed: 4,
            available: 2,
        };
        let msg = e.to_string();
        assert!(msg.contains("needed 4"), "got: {msg}");
        assert!(msg.contains("only 2"), "got: {msg}");
    }

    #[test]
    fn sequence_finished_message() {
        let e = InferError::SequenceFinished(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn max_length_exceeded_message() {
        let e = InferError::MaxLengthExceeded {
            seq_id: 1,
            max_len: 512,
        };
        let msg = e.to_string();
        assert!(msg.contains("512"), "got: {msg}");
    }

    #[test]
    fn speculative_mismatch_message() {
        let e = InferError::SpeculativeDecodingMismatch {
            expected: 5,
            got: 3,
        };
        let msg = e.to_string();
        assert!(msg.contains("expected 5"), "got: {msg}");
        assert!(msg.contains("got 3"), "got: {msg}");
    }

    #[test]
    fn nan_logits_message() {
        let e = InferError::NanLogits;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn infer_result_ok_err() {
        let ok: InferResult<u32> = Ok(99);
        assert!(ok.is_ok());
        let err: InferResult<u32> = Err(InferError::EmptyBatch);
        assert!(err.is_err());
    }
}
