//! Transformer blocks and full-model KV cache.
//!
//! This module provides:
//!
//! - [`PastKvCache`]: KV cache across all layers for one sequence.
//! - [`GptBlock`]: GPT-2 style pre-LayerNorm transformer block.
//! - [`LlamaBlock`]: LLaMA-2/3 pre-RMSNorm transformer block with RoPE.

use crate::error::{LmError, LmResult};
use crate::layer::{
    attention::{LayerKvCache, MultiHeadAttention},
    embedding::RotaryEmbedding,
    ffn::{MlpFfn, SwiGluFfn},
    norm::{LayerNorm, RmsNorm},
};

// ─── PastKvCache ─────────────────────────────────────────────────────────────

/// Full KV cache for one sequence, spanning all transformer layers.
#[derive(Debug, Clone)]
pub struct PastKvCache {
    layer_caches: Vec<LayerKvCache>,
}

impl PastKvCache {
    /// Create an empty cache for `n_layers` layers.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            layer_caches: (0..n_layers)
                .map(|_| LayerKvCache::new(n_kv_heads, head_dim))
                .collect(),
        }
    }

    /// Borrow the cache for layer `idx`.
    pub fn layer(&self, idx: usize) -> LmResult<&LayerKvCache> {
        self.layer_caches
            .get(idx)
            .ok_or(LmError::LayerIndexOutOfRange {
                idx,
                n_layers: self.layer_caches.len(),
            })
    }

    /// Mutably borrow the cache for layer `idx`.
    pub fn layer_mut(&mut self, idx: usize) -> LmResult<&mut LayerKvCache> {
        let n = self.layer_caches.len();
        self.layer_caches
            .get_mut(idx)
            .ok_or(LmError::LayerIndexOutOfRange { idx, n_layers: n })
    }

    /// Number of past tokens already in the cache (from any layer).
    pub fn past_len(&self) -> usize {
        self.layer_caches.first().map_or(0, |c| c.past_len)
    }

    /// Number of layers covered by this cache.
    pub fn n_layers(&self) -> usize {
        self.layer_caches.len()
    }
}

// ─── Helper: add residual connection ─────────────────────────────────────────

fn add_residual(acc: &mut [f32], delta: &[f32]) {
    for (a, &d) in acc.iter_mut().zip(delta.iter()) {
        *a += d;
    }
}

// ─── GptBlock ────────────────────────────────────────────────────────────────

/// GPT-2 style transformer block.
///
/// ```text
/// h = x + attn(ln_1(x))
/// y = h + ffn(ln_2(h))
/// ```
///
/// Both LayerNorm operations are pre-normalisation (applied to the input, not
/// the output — original GPT-2 used post-norm, but most modern GPT-2
/// implementations use pre-norm for stability).
#[derive(Debug, Clone)]
pub struct GptBlock {
    /// First layer normalisation (before attention).
    pub ln_1: LayerNorm,
    /// Multi-head self-attention.
    pub attn: MultiHeadAttention,
    /// Second layer normalisation (before FFN).
    pub ln_2: LayerNorm,
    /// Feed-forward network.
    pub ffn: MlpFfn,
}

impl GptBlock {
    /// Construct with zero-initialised weights.
    pub fn new(
        hidden_dim: usize,
        n_heads: usize,
        ffn_intermediate: usize,
        norm_eps: f32,
    ) -> LmResult<Self> {
        Ok(Self {
            ln_1: LayerNorm::new(hidden_dim, norm_eps)?,
            attn: MultiHeadAttention::new(n_heads, n_heads, hidden_dim, true)?,
            ln_2: LayerNorm::new(hidden_dim, norm_eps)?,
            ffn: MlpFfn::new(hidden_dim, ffn_intermediate)?,
        })
    }

    /// Forward pass with optional KV cache.
    ///
    /// Returns `(output [seq_len × hidden_dim], updated LayerKvCache)`.
    pub fn forward(
        &self,
        x: &[f32],
        seq_len: usize,
        past_kv: Option<&LayerKvCache>,
    ) -> LmResult<(Vec<f32>, LayerKvCache)> {
        // Pre-norm → attention → residual
        let normed_1 = self.ln_1.forward(x, seq_len)?;
        let (attn_out, new_kv) = self.attn.forward(&normed_1, seq_len, past_kv, None)?;
        let mut h = x.to_vec();
        add_residual(&mut h, &attn_out);

        // Pre-norm → FFN → residual
        let normed_2 = self.ln_2.forward(&h, seq_len)?;
        let ffn_out = self.ffn.forward(&normed_2, seq_len)?;
        add_residual(&mut h, &ffn_out);

        Ok((h, new_kv))
    }
}

// ─── LlamaBlock ──────────────────────────────────────────────────────────────

/// LLaMA-2/3 transformer block.
///
/// ```text
/// h = x + attn(rms_norm(x))   # RoPE applied inside attn
/// y = h + ffn(rms_norm(h))
/// ```
///
/// Uses RMSNorm (no bias) and SwiGLU FFN.
#[derive(Debug, Clone)]
pub struct LlamaBlock {
    /// RMSNorm before attention.
    pub attn_norm: RmsNorm,
    /// Multi-head attention with GQA support.
    pub attn: MultiHeadAttention,
    /// RMSNorm before FFN.
    pub ffn_norm: RmsNorm,
    /// SwiGLU feed-forward network.
    pub ffn: SwiGluFfn,
    /// Rotary positional embedding.
    pub rope: RotaryEmbedding,
}

impl LlamaBlock {
    /// Construct with zero-initialised weights.
    pub fn new(
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        intermediate_dim: usize,
        max_positions: usize,
        rope_theta: f32,
        rms_eps: f32,
    ) -> LmResult<Self> {
        if hidden_dim % n_heads != 0 {
            return Err(LmError::HeadDimMismatch {
                hidden_dim,
                n_heads,
            });
        }
        let head_dim = hidden_dim / n_heads;
        Ok(Self {
            attn_norm: RmsNorm::new(hidden_dim, rms_eps)?,
            attn: MultiHeadAttention::new(n_heads, n_kv_heads, hidden_dim, true)?,
            ffn_norm: RmsNorm::new(hidden_dim, rms_eps)?,
            ffn: SwiGluFfn::new(hidden_dim, intermediate_dim)?,
            rope: RotaryEmbedding::new(head_dim, max_positions, rope_theta)?,
        })
    }

    /// Forward pass with optional KV cache.
    ///
    /// Returns `(output [seq_len × hidden_dim], updated LayerKvCache)`.
    pub fn forward(
        &self,
        x: &[f32],
        seq_len: usize,
        past_kv: Option<&LayerKvCache>,
    ) -> LmResult<(Vec<f32>, LayerKvCache)> {
        // Pre-norm → attention (with RoPE) → residual
        let normed_1 = self.attn_norm.forward(x, seq_len)?;
        let (attn_out, new_kv) =
            self.attn
                .forward(&normed_1, seq_len, past_kv, Some(&self.rope))?;
        let mut h = x.to_vec();
        add_residual(&mut h, &attn_out);

        // Pre-norm → SwiGLU FFN → residual
        let normed_2 = self.ffn_norm.forward(&h, seq_len)?;
        let ffn_out = self.ffn.forward(&normed_2, seq_len)?;
        add_residual(&mut h, &ffn_out);

        Ok((h, new_kv))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PastKvCache ───────────────────────────────────────────────────────

    #[test]
    fn past_kv_cache_empty() {
        let c = PastKvCache::new(4, 2, 8);
        assert_eq!(c.n_layers(), 4);
        assert_eq!(c.past_len(), 0);
    }

    #[test]
    fn past_kv_cache_layer_access() {
        let c = PastKvCache::new(3, 2, 4);
        assert!(c.layer(0).is_ok());
        assert!(c.layer(2).is_ok());
        assert!(matches!(
            c.layer(3),
            Err(LmError::LayerIndexOutOfRange { idx: 3, .. })
        ));
    }

    #[test]
    fn past_kv_cache_layer_mut() {
        let mut c = PastKvCache::new(2, 2, 4);
        let lc = c.layer_mut(0).unwrap();
        lc.append(&[1.0_f32; 2 * 4], &[1.0_f32; 2 * 4], 1);
        assert_eq!(c.past_len(), 1);
    }

    // ── GptBlock ──────────────────────────────────────────────────────────

    #[test]
    fn gpt_block_output_shape() {
        let block = GptBlock::new(8, 2, 16, 1e-5).unwrap();
        let x = vec![0.5_f32; 2 * 8]; // 2 tokens, hidden=8
        let (out, kv) = block.forward(&x, 2, None).unwrap();
        assert_eq!(out.len(), 2 * 8);
        assert_eq!(kv.past_len, 2);
    }

    #[test]
    fn gpt_block_residual_connection() {
        // With all weights = 0, attn and FFN outputs are 0 → output = x
        let block = GptBlock::new(4, 2, 8, 1e-5).unwrap();
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let (out, _) = block.forward(&x, 1, None).unwrap();
        // residual: out = x + 0 = x
        for (&o, &xi) in out.iter().zip(x.iter()) {
            assert!((o - xi).abs() < 1e-5, "residual failed: o={o} xi={xi}");
        }
    }

    #[test]
    fn gpt_block_kv_cache_extends() {
        let block = GptBlock::new(4, 2, 8, 1e-5).unwrap();
        let x = vec![0.0_f32; 4];
        let (_, kv1) = block.forward(&x, 1, None).unwrap();
        let (_, kv2) = block.forward(&x, 1, Some(&kv1)).unwrap();
        assert_eq!(kv2.past_len, 2);
    }

    // ── LlamaBlock ────────────────────────────────────────────────────────

    #[test]
    fn llama_block_output_shape() {
        let block = LlamaBlock::new(8, 4, 2, 12, 32, 10_000.0, 1e-5).unwrap();
        let x = vec![0.5_f32; 2 * 8]; // 2 tokens
        let (out, kv) = block.forward(&x, 2, None).unwrap();
        assert_eq!(out.len(), 2 * 8);
        assert_eq!(kv.past_len, 2);
    }

    #[test]
    fn llama_block_residual_connection() {
        let block = LlamaBlock::new(4, 2, 2, 8, 32, 10_000.0, 1e-5).unwrap();
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let (out, _) = block.forward(&x, 1, None).unwrap();
        // Zero weights → attn and ffn outputs are 0 → output = x
        for (&o, &xi) in out.iter().zip(x.iter()) {
            assert!((o - xi).abs() < 1e-5, "residual failed: o={o} xi={xi}");
        }
    }

    #[test]
    fn llama_block_kv_cache_incremental() {
        let block = LlamaBlock::new(4, 2, 2, 8, 32, 10_000.0, 1e-5).unwrap();
        let x = vec![0.0_f32; 4];
        let (_, kv1) = block.forward(&x, 1, None).unwrap();
        let (_, kv2) = block.forward(&x, 1, Some(&kv1)).unwrap();
        assert_eq!(kv2.past_len, 2);
    }

    #[test]
    fn llama_block_gqa() {
        // 4 query heads, 2 KV heads
        let block = LlamaBlock::new(8, 4, 2, 12, 32, 10_000.0, 1e-5).unwrap();
        let x = vec![0.0_f32; 3 * 8]; // 3 tokens
        let (out, _) = block.forward(&x, 3, None).unwrap();
        assert_eq!(out.len(), 3 * 8);
    }

    #[test]
    fn llama_block_head_mismatch_error() {
        // hidden_dim=8 not divisible by n_heads=3
        assert!(LlamaBlock::new(8, 3, 1, 12, 32, 10_000.0, 1e-5).is_err());
    }
}
