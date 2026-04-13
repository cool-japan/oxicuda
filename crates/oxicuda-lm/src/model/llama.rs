//! LLaMA-2 / LLaMA-3 / Mistral style causal language model.
//!
//! # Architecture
//!
//! ```text
//! token_ids
//!     │
//!     └─► TokenEmbedding
//!     │
//!     ├─► LlamaBlock × n_layers  (RMSNorm → GQA-MHA+RoPE → residual → RMSNorm → SwiGLU → residual)
//!     │
//!     ├─► RMSNorm (output_norm)
//!     │
//!     └─► LM head (independent weight, not tied to embedding)
//!
//! Output: logits [seq_len × vocab_size]
//! ```
//!
//! Unlike GPT-2, LLaMA uses **independent** LM head weights (not weight-tied).
//! The `lm_head` field stores its own `[vocab_size × hidden_dim]` matrix.

use crate::config::LlamaConfig;
use crate::error::{LmError, LmResult};
use crate::layer::{
    embedding::TokenEmbedding,
    norm::RmsNorm,
    transformer::{LlamaBlock, PastKvCache},
};
use crate::model::gpt::argmax_f32;
use crate::weights::WeightTensor;

// ─── LlamaModel ──────────────────────────────────────────────────────────────

/// LLaMA-2/3/Mistral causal language model.
#[derive(Debug, Clone)]
pub struct LlamaModel {
    /// Model hyperparameters.
    pub config: LlamaConfig,
    /// Token embedding table: `[vocab_size × hidden_dim]`.
    pub embed: TokenEmbedding,
    /// Transformer blocks.
    pub blocks: Vec<LlamaBlock>,
    /// Output RMSNorm.
    pub norm: RmsNorm,
    /// LM head weight: `[vocab_size × hidden_dim]`.
    pub lm_head: WeightTensor,
}

impl LlamaModel {
    // ── Constructor ──────────────────────────────────────────────────────

    /// Construct a LLaMA model with zero-initialised weights.
    pub fn new(config: LlamaConfig) -> LmResult<Self> {
        config.validate()?;
        let embed = TokenEmbedding::new(config.vocab_size, config.hidden_dim)?;

        let blocks: Vec<LlamaBlock> = (0..config.n_layers)
            .map(|_| {
                LlamaBlock::new(
                    config.hidden_dim,
                    config.n_heads,
                    config.n_kv_heads,
                    config.intermediate_dim,
                    config.max_position_embeddings,
                    config.rope_theta,
                    config.rms_norm_eps,
                )
            })
            .collect::<LmResult<_>>()?;

        let norm = RmsNorm::new(config.hidden_dim, config.rms_norm_eps)?;
        let lm_head = WeightTensor::zeros(&[config.vocab_size, config.hidden_dim]);

        Ok(Self {
            config,
            embed,
            blocks,
            norm,
            lm_head,
        })
    }

    // ── Forward pass ─────────────────────────────────────────────────────

    /// Full forward pass.
    ///
    /// # Arguments
    ///
    /// - `token_ids`: token sequence (length ≥ 1).
    /// - `past_kv`: optional KV cache from previous decode steps.
    ///
    /// # Returns
    ///
    /// `(logits [seq_len × vocab_size], updated PastKvCache)`.
    pub fn forward(
        &self,
        token_ids: &[u32],
        past_kv: Option<&PastKvCache>,
    ) -> LmResult<(Vec<f32>, PastKvCache)> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(LmError::EmptyInput {
                context: "LlamaModel::forward token_ids",
            });
        }
        let past_len = past_kv.map_or(0, |c| c.past_len());
        let total_len = past_len + seq_len;
        if total_len > self.config.max_position_embeddings {
            return Err(LmError::SequenceTooLong {
                total_len,
                max_pos: self.config.max_position_embeddings,
            });
        }

        // ── 1. Token embeddings ───────────────────────────────────────────
        let mut h = self.embed.forward(token_ids)?;

        // ── 2. Transformer blocks ─────────────────────────────────────────
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim();
        let mut new_kv = PastKvCache::new(self.config.n_layers, n_kv_heads, head_dim);

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let past_layer = past_kv.and_then(|c| c.layer(layer_idx).ok());
            let (block_out, layer_kv) = block.forward(&h, seq_len, past_layer)?;
            h = block_out;
            *new_kv.layer_mut(layer_idx)? = layer_kv;
        }

        // ── 3. Output RMSNorm ─────────────────────────────────────────────
        let h = self.norm.forward(&h, seq_len)?;

        // ── 4. LM head ────────────────────────────────────────────────────
        let logits = self.apply_lm_head(&h, seq_len)?;

        Ok((logits, new_kv))
    }

    /// Greedy next-token prediction from the last position's logits.
    ///
    /// Returns `(next_token_id, updated PastKvCache)`.
    pub fn next_token(
        &self,
        token_ids: &[u32],
        past_kv: Option<&PastKvCache>,
    ) -> LmResult<(u32, PastKvCache)> {
        let (logits, new_kv) = self.forward(token_ids, past_kv)?;
        let seq_len = token_ids.len();
        let last_start = (seq_len - 1) * self.config.vocab_size;
        let last_logits = &logits[last_start..last_start + self.config.vocab_size];
        let next_tok = argmax_f32(last_logits)?;
        Ok((next_tok, new_kv))
    }

    // ── Utility ──────────────────────────────────────────────────────────

    /// Approximate total parameter count.
    pub fn n_params(&self) -> usize {
        let embed_params = self.config.vocab_size * self.config.hidden_dim;
        let norm_params = self.config.hidden_dim; // weight only (RMSNorm has no bias)
        let lm_head_params = self.config.vocab_size * self.config.hidden_dim;

        let block_params: usize = self
            .blocks
            .iter()
            .map(|_| {
                let hd = self.config.hidden_dim;
                let id = self.config.intermediate_dim;
                let kv = self.config.n_kv_heads * self.config.head_dim();
                // norms (no bias) + attn projections + ffn projections
                let norms = 2 * hd;
                let attn = hd * hd   // w_q
                + kv * hd         // w_k
                + kv * hd         // w_v
                + hd * hd; // w_o
                let ffn = id * hd   // w_gate
                + id * hd         // w_up
                + hd * id; // w_down
                norms + attn + ffn
            })
            .sum();

        embed_params + norm_params + lm_head_params + block_params
    }

    // ── Private ───────────────────────────────────────────────────────────

    /// Apply the LM head weight matrix.
    ///
    /// `h`: `[seq_len × hidden_dim]`; output: `[seq_len × vocab_size]`.
    fn apply_lm_head(&self, h: &[f32], seq_len: usize) -> LmResult<Vec<f32>> {
        let hd = self.config.hidden_dim;
        let vs = self.config.vocab_size;
        if h.len() != seq_len * hd {
            return Err(LmError::DimensionMismatch {
                expected: seq_len * hd,
                got: h.len(),
            });
        }
        // logits[t, v] = sum_d h[t, d] * lm_head[v, d]
        let head_data = &self.lm_head.data;
        let mut logits = vec![0.0_f32; seq_len * vs];
        for t in 0..seq_len {
            let h_row = &h[t * hd..(t + 1) * hd];
            let l_row = &mut logits[t * vs..(t + 1) * vs];
            for v in 0..vs {
                let lh_row = &head_data[v * hd..(v + 1) * hd];
                l_row[v] = h_row
                    .iter()
                    .zip(lh_row.iter())
                    .map(|(&hi, &li)| hi * li)
                    .sum();
            }
        }
        Ok(logits)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlamaConfig;

    fn tiny_model() -> LlamaModel {
        LlamaModel::new(LlamaConfig::tiny()).unwrap()
    }

    #[test]
    fn llama_model_constructs() {
        let m = tiny_model();
        assert_eq!(m.blocks.len(), 2);
        assert_eq!(m.config.n_kv_heads, 2);
    }

    #[test]
    fn llama_forward_output_shape() {
        let m = tiny_model();
        let (logits, kv) = m.forward(&[0, 1, 2], None).unwrap();
        assert_eq!(logits.len(), 3 * m.config.vocab_size);
        assert_eq!(kv.past_len(), 3);
        assert_eq!(kv.n_layers(), 2);
    }

    #[test]
    fn llama_forward_empty_error() {
        let m = tiny_model();
        assert!(matches!(
            m.forward(&[], None),
            Err(LmError::EmptyInput { .. })
        ));
    }

    #[test]
    fn llama_forward_too_long_error() {
        let mut cfg = LlamaConfig::tiny();
        cfg.max_position_embeddings = 4;
        let m = LlamaModel::new(cfg).unwrap();
        let ids: Vec<u32> = (0..5).collect();
        assert!(matches!(
            m.forward(&ids, None),
            Err(LmError::SequenceTooLong { .. })
        ));
    }

    #[test]
    fn llama_kv_cache_incremental() {
        let m = tiny_model();
        let (_, kv1) = m.forward(&[0, 1], None).unwrap();
        let (logits2, kv2) = m.forward(&[2], Some(&kv1)).unwrap();
        assert_eq!(logits2.len(), m.config.vocab_size);
        assert_eq!(kv2.past_len(), 3);
    }

    #[test]
    fn llama_next_token_valid_id() {
        let m = tiny_model();
        let (tok, _) = m.next_token(&[0], None).unwrap();
        assert!((tok as usize) < m.config.vocab_size);
    }

    #[test]
    fn llama_n_params_positive() {
        let m = tiny_model();
        assert!(m.n_params() > 0);
    }

    #[test]
    fn llama_incremental_vs_full_last_position() {
        // Full forward of [0,1] then check token-1 logits match incremental.
        let m = tiny_model();
        let (logits_full, _) = m.forward(&[0, 1], None).unwrap();
        let last_full = &logits_full[m.config.vocab_size..];

        let (_, kv0) = m.forward(&[0], None).unwrap();
        let (logits_incr, _) = m.forward(&[1], Some(&kv0)).unwrap();

        for (&full_v, &incr_v) in last_full.iter().zip(logits_incr.iter()) {
            assert!(
                (full_v - incr_v).abs() < 1e-4,
                "full={full_v} incr={incr_v}"
            );
        }
    }

    #[test]
    fn llama_lm_head_ones_gives_nonzero_logits() {
        let mut m = tiny_model();
        // Set embedding and lm_head to ones so we get non-zero output
        m.embed.weight.data = vec![1.0_f32; m.config.vocab_size * m.config.hidden_dim];
        m.lm_head.data = vec![1.0_f32; m.config.vocab_size * m.config.hidden_dim];
        // Forward: after all-zero block weights, h ≈ embed(token) = ones
        // Logits ≈ ones · ones^T = sum = hidden_dim
        let (logits, _) = m.forward(&[0], None).unwrap();
        // Not all zero
        let all_zero = logits.iter().all(|&v| v.abs() < 1e-6);
        assert!(!all_zero, "expected non-zero logits with ones weights");
    }

    #[test]
    fn llama_gqa_factor_consistent() {
        // Verify GQA config is used: 4 query heads, 2 KV heads
        let m = tiny_model();
        assert_eq!(m.config.gqa_factor(), 2);
        // Model forward should not error
        let (_, _) = m.forward(&[0], None).unwrap();
    }
}
