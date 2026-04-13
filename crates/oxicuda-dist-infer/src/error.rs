//! Error types for oxicuda-dist-infer.

use thiserror::Error;

/// All errors produced by the distributed inference engine.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DistInferError {
    // ── General ──────────────────────────────────────────────────────────────
    /// World-size must be a power-of-two for ring algorithms or ≥ 1 for any op.
    #[error("invalid world size {world_size}: {reason}")]
    InvalidWorldSize {
        world_size: usize,
        reason: &'static str,
    },

    /// Rank is out of range for the given world-size.
    #[error("rank {rank} out of range for world_size {world_size}")]
    RankOutOfRange { rank: usize, world_size: usize },

    /// Feature requires at least `needed` ranks.
    #[error("need at least {needed} ranks (world_size={world_size})")]
    TooFewRanks { needed: usize, world_size: usize },

    // ── Tensor Parallelism ────────────────────────────────────────────────────
    /// Output feature dimension is not evenly divisible by tensor-parallel degree.
    #[error("output_features {features} not divisible by tp_degree {degree}")]
    TpFeaturesMisaligned { features: usize, degree: usize },

    /// Input feature dimension is not evenly divisible by tensor-parallel degree.
    #[error("input_features {features} not divisible by tp_degree {degree}")]
    TpInputMisaligned { features: usize, degree: usize },

    /// Shard weight has wrong shape.
    #[error("shard weight shape [{rows}×{cols}] does not match expected [{exp_rows}×{exp_cols}]")]
    ShardShapeMismatch {
        rows: usize,
        cols: usize,
        exp_rows: usize,
        exp_cols: usize,
    },

    // ── Sequence Parallelism ──────────────────────────────────────────────────
    /// Sequence length is not divisible by sp_degree.
    #[error("seq_len {seq_len} not divisible by sp_degree {degree}")]
    SpSeqLenMisaligned { seq_len: usize, degree: usize },

    /// Local chunk size is zero (sequence too short for sp_degree).
    #[error("local chunk size is zero (seq_len={seq_len}, sp_degree={degree})")]
    EmptyChunk { seq_len: usize, degree: usize },

    // ── Expert Parallelism ────────────────────────────────────────────────────
    /// Expert count is not divisible by expert-parallel degree.
    #[error("n_experts {n_experts} not divisible by ep_degree {degree}")]
    EpExpertsMisaligned { n_experts: usize, degree: usize },

    /// Token routing produced an empty expert assignment.
    #[error("expert {expert_id} received no tokens after routing")]
    EmptyExpertBatch { expert_id: usize },

    // ── Distributed KV Cache ──────────────────────────────────────────────────
    /// Sequence is not managed by this rank.
    #[error("sequence {seq_id} is not owned by rank {rank}")]
    SequenceNotOwned { seq_id: u64, rank: usize },

    /// Sequence migration target rank is invalid.
    #[error("migration target rank {target} invalid (world_size={world_size})")]
    MigrationTargetInvalid { target: usize, world_size: usize },

    /// KV cache block pool exhausted on rank.
    #[error("block pool exhausted on rank {rank}: no free blocks")]
    BlockPoolExhausted { rank: usize },

    // ── Request Router ────────────────────────────────────────────────────────
    /// All ranks are at capacity.
    #[error("all {n_ranks} ranks at capacity, cannot route request")]
    AllRanksAtCapacity { n_ranks: usize },

    /// Routing policy requires non-zero token count.
    #[error("cannot route empty token sequence")]
    EmptyTokenSequence,

    /// Prefix affinity lookup failed — no matching cache entry found.
    #[error("no prefix affinity entry for token hash {token_hash:#018x}")]
    NoPrefixAffinity { token_hash: u64 },

    // ── Dimension / Shape ────────────────────────────────────────────────────
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    // ── Other ────────────────────────────────────────────────────────────────
    #[error("internal error: {0}")]
    Internal(&'static str),
}

/// Convenience alias.
pub type DistInferResult<T> = Result<T, DistInferError>;
