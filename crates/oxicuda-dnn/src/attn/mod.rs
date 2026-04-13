//! Attention mechanisms for transformer models.
//!
//! This module provides GPU-accelerated attention implementations:
//!
//! | Sub-module          | Description                                          |
//! |---------------------|------------------------------------------------------|
//! | [`mha`]             | Naive multi-head attention (reference / small seqs)  |
//! | [`flash_attn`]      | FlashAttention-2 forward, backward, paged, decode   |
//! | [`rope`]            | Rotary Positional Embedding (RoPE)                   |
//! | [`fused_rope_attn`] | Fused RoPE + attention (single kernel, less BW)      |
//! | [`block_sparse`]    | Block-sparse attention for long-context transformers |
//! | [`ring_attention`]  | Ring attention for sequence parallelism (multi-GPU)  |
//! | [`speculative_decode`]| Speculative decoding KV-cache with rollback        |
//! | [`kv_cache`]        | KV-cache management for autoregressive decoding     |

pub mod block_sparse;
pub mod flash_attn;
pub mod fused_rope_attn;
pub mod gqa;
pub mod kv_cache;
pub mod mha;
pub mod ring_attention;
pub mod rope;
pub mod sliding_window;
pub mod speculative_decode;

// Re-exports for convenience.
pub use block_sparse::{BlockSparseAttentionPlan, BlockSparseConfig, BlockSparsePattern};
pub use flash_attn::backward::flash_attention_backward;
pub use flash_attn::decode::single_query_decode_attention;
pub use flash_attn::forward::{FlashAttentionConfig, flash_attention_forward};
pub use flash_attn::paged::{PagedAttentionConfig, paged_attention_decode};
pub use fused_rope_attn::{FusedRopeAttnConfig, FusedRopeAttnPlan};
pub use gqa::{GqaConfig, gqa_forward};
pub use kv_cache::{KvCache, KvCacheConfig};
pub use mha::multi_head_attention;
pub use ring_attention::{
    RingAttentionConfig, RingAttentionDtype, RingAttentionPlan, RingAttentionStats, RingCommPlan,
    RingStep,
};
pub use rope::apply_rope;
pub use sliding_window::{SlidingWindowConfig, sliding_window_attention};
// ---------------------------------------------------------------------------
// Tcgen05AttentionConfig
// ---------------------------------------------------------------------------

/// Configuration for 5th-generation Tensor Core attention kernels (`tcgen05`).
///
/// `tcgen05` instructions are available on Blackwell (sm_100+) GPUs. They
/// provide a wider 128×256 output tile compared to the 128×128 tile of
/// Hopper's `wgmma`, enabling higher throughput on Blackwell hardware.
///
/// # FP4 support
///
/// Blackwell (sm_100+) natively supports FP4 (NV FP4 / OCP MXFP4)
/// as an input precision for tensor core operations, reducing memory
/// bandwidth by 2× compared to FP8 and 4× compared to FP16.
///
/// # Validity
///
/// The configuration is only valid when `sm_version >= 100`. Creating a
/// config with a lower SM version is allowed for introspection, but
/// [`is_valid`](Self::is_valid) will return `false`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tcgen05AttentionConfig {
    /// Target SM version (must be ≥ 100 for valid tcgen05 support).
    pub sm_version: u32,
    /// M dimension of the tcgen05 output tile (128 for Blackwell).
    pub m_tile: usize,
    /// N dimension of the tcgen05 output tile (256 for Blackwell).
    pub n_tile: usize,
    /// Whether FP4 input precision is supported (`sm_version >= 100`).
    pub supports_fp4: bool,
}

impl Tcgen05AttentionConfig {
    /// Constructs a `tcgen05` attention configuration for the given SM version.
    ///
    /// The tile shape is fixed at M=128, N=256 per the Blackwell ISA specification.
    /// FP4 support is enabled for sm_100 and above.
    #[must_use]
    pub fn new(sm_version: u32) -> Self {
        Self {
            sm_version,
            m_tile: 128,
            n_tile: 256,
            supports_fp4: sm_version >= 100,
        }
    }

    /// Returns `true` if this SM version supports `tcgen05` tensor core instructions.
    ///
    /// `tcgen05` requires Blackwell (sm_100+).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.sm_version >= 100
    }
}

// ---------------------------------------------------------------------------

pub use speculative_decode::{
    KvCheckpoint, KvManager, SpecDecConfig, SpecDecOutput, SpeculativeDecodeConfig,
    SpeculativeDecodePlan, SpeculativeDecoder, SpeculativeKvManager, TokenVerificationResult,
    VerificationResult, accept_token,
};

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Task 4: Tcgen05AttentionConfig — 5th-gen Tensor Core attention
    // -----------------------------------------------------------------------

    /// sm_100 (Blackwell) is a valid target for tcgen05 attention.
    #[test]
    fn tcgen05_valid_for_sm100() {
        let cfg = Tcgen05AttentionConfig::new(100);
        assert!(cfg.is_valid(), "sm_100 must be valid for tcgen05");
    }

    /// sm_120 (Blackwell Ultra) is also valid for tcgen05 attention.
    #[test]
    fn tcgen05_valid_for_sm120() {
        let cfg = Tcgen05AttentionConfig::new(120);
        assert!(cfg.is_valid(), "sm_120 must be valid for tcgen05");
    }

    /// sm_90 (Hopper) does NOT support tcgen05 (that uses wgmma instead).
    #[test]
    fn tcgen05_invalid_for_sm90() {
        let cfg = Tcgen05AttentionConfig::new(90);
        assert!(!cfg.is_valid(), "sm_90 should not be valid for tcgen05");
    }

    /// sm_80 (Ampere) does NOT support tcgen05.
    #[test]
    fn tcgen05_invalid_for_sm80() {
        let cfg = Tcgen05AttentionConfig::new(80);
        assert!(!cfg.is_valid(), "sm_80 should not be valid for tcgen05");
    }

    /// The output tile for tcgen05 on Blackwell is M=128, N=256.
    #[test]
    fn tcgen05_tile_shape_m128_n256() {
        let cfg = Tcgen05AttentionConfig::new(100);
        assert_eq!(cfg.m_tile, 128, "tcgen05 M tile must be 128");
        assert_eq!(cfg.n_tile, 256, "tcgen05 N tile must be 256");
    }

    /// FP4 input precision is supported on Blackwell (sm_100+).
    #[test]
    fn tcgen05_supports_fp4_on_blackwell() {
        let cfg = Tcgen05AttentionConfig::new(100);
        assert!(
            cfg.supports_fp4,
            "Blackwell (sm_100) should support FP4 input"
        );
    }

    /// FP4 input precision is NOT supported on Hopper (sm_90).
    #[test]
    fn tcgen05_does_not_support_fp4_on_hopper() {
        let cfg = Tcgen05AttentionConfig::new(90);
        assert!(
            !cfg.supports_fp4,
            "Hopper (sm_90) should not support FP4 input"
        );
    }

    /// sm_version field is stored as provided.
    #[test]
    fn tcgen05_sm_version_stored_correctly() {
        let cfg = Tcgen05AttentionConfig::new(100);
        assert_eq!(cfg.sm_version, 100);
    }

    /// Tile shape is the same for all valid Blackwell SM versions.
    #[test]
    fn tcgen05_tile_shape_consistent_across_blackwell() {
        let sm100 = Tcgen05AttentionConfig::new(100);
        let sm120 = Tcgen05AttentionConfig::new(120);
        assert_eq!(
            sm100.m_tile, sm120.m_tile,
            "M tile must be the same for all Blackwell"
        );
        assert_eq!(
            sm100.n_tile, sm120.n_tile,
            "N tile must be the same for all Blackwell"
        );
    }
}
