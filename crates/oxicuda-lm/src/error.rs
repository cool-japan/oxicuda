//! Error types for the `oxicuda-lm` crate.

use thiserror::Error;

/// All errors that can arise from LLM inference operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LmError {
    /// Tensor or slice dimension does not match expectation.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Model or layer configuration is invalid.
    #[error("invalid config: {msg}")]
    InvalidConfig { msg: String },

    /// An input that must be non-empty was empty.
    #[error("empty input: {context}")]
    EmptyInput { context: &'static str },

    /// A token id is outside the vocabulary range.
    #[error("out-of-vocabulary token id {token}")]
    OutOfVocab { token: u32 },

    /// Tokenizer was used before being properly initialised.
    #[error("tokenizer is not initialised")]
    TokenizerUninitialized,

    /// Token byte sequence is not valid UTF-8.
    #[error("UTF-8 decode error for token {token}")]
    Utf8Decode { token: u32 },

    /// A named weight entry is absent from `ModelWeights`.
    #[error("weight '{name}' not found")]
    WeightNotFound { name: String },

    /// A named weight has an unexpected shape.
    #[error("weight '{name}' shape mismatch: expected {expected:?}, got {got:?}")]
    WeightShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Transformer layer index is out of range.
    #[error("layer index {idx} is out of range [0, {n_layers})")]
    LayerIndexOutOfRange { idx: usize, n_layers: usize },

    /// Hidden dimension is not divisible by the number of attention heads.
    #[error(
        "head dimension mismatch: hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}"
    )]
    HeadDimMismatch { hidden_dim: usize, n_heads: usize },

    /// KV cache contains a different number of past tokens than expected.
    #[error("KV cache length mismatch: expected past_len={past_len}, got {got}")]
    KvCacheLengthMismatch { past_len: usize, got: usize },

    /// Input sequence length exceeds the model's `max_position_embeddings`.
    #[error("sequence too long: total={total_len} > max_position_embeddings={max_pos}")]
    SequenceTooLong { total_len: usize, max_pos: usize },

    /// A BPE merge references tokens that are not in the vocabulary.
    #[error("invalid BPE merge pair: tokens {a} and {b} not both present in vocabulary")]
    InvalidMergePair { a: u32, b: u32 },

    /// Vocabulary size does not match model configuration.
    #[error("vocab size mismatch: expected {expected}, got {got}")]
    VocabSizeMismatch { expected: usize, got: usize },

    /// n_kv_heads does not divide n_heads evenly (GQA requirement).
    #[error(
        "GQA constraint violated: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
    )]
    GqaHeadMismatch { n_heads: usize, n_kv_heads: usize },

    /// A weight tensor element count does not match its declared shape.
    #[error("weight data length {data_len} does not match shape {shape:?} product {expected}")]
    WeightDataLengthMismatch {
        data_len: usize,
        shape: Vec<usize>,
        expected: usize,
    },

    /// Catch-all for internal invariant violations.
    #[error("internal error: {msg}")]
    Internal { msg: String },
}

/// Convenience alias.
pub type LmResult<T> = Result<T, LmError>;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = LmError::DimensionMismatch {
            expected: 4,
            got: 3,
        };
        assert!(e.to_string().contains("expected 4"));
        assert!(e.to_string().contains("got 3"));
    }

    #[test]
    fn error_display_invalid_config() {
        let e = LmError::InvalidConfig {
            msg: "n_heads must be positive".into(),
        };
        assert!(e.to_string().contains("n_heads must be positive"));
    }

    #[test]
    fn error_display_weight_not_found() {
        let e = LmError::WeightNotFound {
            name: "layer.0.attn.w_q".into(),
        };
        assert!(e.to_string().contains("layer.0.attn.w_q"));
    }

    #[test]
    fn error_display_weight_shape_mismatch() {
        let e = LmError::WeightShapeMismatch {
            name: "embed".into(),
            expected: vec![128, 64],
            got: vec![64, 128],
        };
        let s = e.to_string();
        assert!(s.contains("embed"));
        assert!(s.contains("[128, 64]"));
        assert!(s.contains("[64, 128]"));
    }

    #[test]
    fn error_display_out_of_vocab() {
        let e = LmError::OutOfVocab { token: 99999 };
        assert!(e.to_string().contains("99999"));
    }

    #[test]
    fn error_display_sequence_too_long() {
        let e = LmError::SequenceTooLong {
            total_len: 2048,
            max_pos: 1024,
        };
        let s = e.to_string();
        assert!(s.contains("2048"));
        assert!(s.contains("1024"));
    }

    #[test]
    fn error_display_gqa_mismatch() {
        let e = LmError::GqaHeadMismatch {
            n_heads: 32,
            n_kv_heads: 5,
        };
        let s = e.to_string();
        assert!(s.contains("32"));
        assert!(s.contains("5"));
    }

    #[test]
    fn error_display_invalid_merge_pair() {
        let e = LmError::InvalidMergePair { a: 10, b: 20 };
        let s = e.to_string();
        assert!(s.contains("10"));
        assert!(s.contains("20"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(LmError::Internal { msg: "test".into() });
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn error_clone_and_eq() {
        let a = LmError::EmptyInput {
            context: "token_ids",
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}
