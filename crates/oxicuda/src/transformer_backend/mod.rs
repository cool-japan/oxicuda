//! TrustformeRS Transformer GPU Backend for OxiCUDA.
//!
//! This module provides transformer model inference infrastructure suitable
//! for the TrustformeRS project, implementing cutting-edge techniques:
//!
//! - **Paged KV-Cache** — memory-efficient key-value caching with block tables,
//!   copy-on-write for shared prefixes, and configurable eviction policies.
//! - **Attention Dispatch** — selects optimal attention kernel (Flash, Paged,
//!   Hopper async, sliding window) based on hardware and sequence config.
//! - **Continuous Batching** — iteration-level scheduling with preemption,
//!   priority queues, and prefill/decode separation.
//! - **Speculative Decoding** — draft-verify pipeline with rejection sampling,
//!   typical acceptance, and tree verification.
//! - **Token Sampling** — top-k, top-p, min-p, temperature, repetition/frequency/
//!   presence penalties, beam search, and Mirostat.
//! - **Quantized Inference** — INT8, INT4, FP8, GPTQ, AWQ, SmoothQuant with
//!   per-layer config and on-the-fly dequantization.
//!
//! # Feature Gate
//!
//! This module is behind the `transformer-backend` feature flag:
//!
//! ```toml
//! [dependencies]
//! oxicuda = { version = "0.3.0", features = ["transformer-backend"] }
//! ```
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

pub mod attention;
pub mod kv_cache;
pub mod quantize;
pub mod sampling;
pub mod scheduler;
pub mod speculative;

// ─── Public re-exports ──────────────────────────────────────

pub use attention::{AttentionConfig, AttentionDispatch, AttentionKind, HeadConfig};
pub use kv_cache::{
    BlockId, CacheEvictionPolicy, CacheStats, PagedKvCache, PagedKvCacheConfig, SequenceId,
};
pub use quantize::{QuantConfig, QuantMethod, QuantizedTensor, QuantizedWeight};
pub use sampling::{BeamSearchConfig, MirostatState, SamplingOutput, SamplingParams, TokenSampler};
pub use scheduler::{
    ContinuousBatchScheduler, Priority, SchedulerBudget, SchedulerConfig, SchedulerOutput,
    Sequence, SequenceGroup, SequenceStatus,
};
pub use speculative::{
    AcceptanceMethod, DraftModelConfig, SpeculativeDecoder, SpeculativeDecoderConfig,
    SpeculativeOutput,
};

/// Error type for transformer backend operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerError {
    /// KV-cache error (e.g. out of blocks).
    CacheError(String),
    /// Scheduler error.
    SchedulerError(String),
    /// Attention dispatch error.
    AttentionError(String),
    /// Sampling error.
    SamplingError(String),
    /// Quantization error.
    QuantizationError(String),
    /// Speculative decoding error.
    SpeculativeError(String),
    /// Invalid configuration.
    InvalidConfig(String),
}

impl std::fmt::Display for TransformerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CacheError(msg) => write!(f, "cache error: {msg}"),
            Self::SchedulerError(msg) => write!(f, "scheduler error: {msg}"),
            Self::AttentionError(msg) => write!(f, "attention error: {msg}"),
            Self::SamplingError(msg) => write!(f, "sampling error: {msg}"),
            Self::QuantizationError(msg) => write!(f, "quantization error: {msg}"),
            Self::SpeculativeError(msg) => write!(f, "speculative error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for TransformerError {}

/// Result type for transformer backend operations.
pub type TransformerResult<T> = Result<T, TransformerError>;
