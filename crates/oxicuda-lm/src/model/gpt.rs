//! GPT-2 style causal language model.
//!
//! # Architecture
//!
//! ```text
//! token_ids
//!     │
//!     ├─► TokenEmbedding       + LearnedPositionalEmbedding
//!     │   (add embeddings)
//!     │
//!     ├─► GptBlock × n_layers  (pre-LayerNorm, MHA + MLP)
//!     │
//!     ├─► LayerNorm (ln_f)
//!     │
//!     └─► LM head = token_embed.weight^T   (weight tying)
//!
//! Output: logits [seq_len × vocab_size]
//! ```
//!
//! Weight tying: the LM head shares the token embedding matrix, so parameters
//! are not double-counted.

use crate::config::GptConfig;
use crate::error::{LmError, LmResult};
use crate::layer::{
    embedding::{LearnedPositionalEmbedding, TokenEmbedding},
    norm::LayerNorm,
    transformer::{GptBlock, PastKvCache},
};

// ─── Gpt2Model ───────────────────────────────────────────────────────────────

/// GPT-2 causal language model.
///
/// All weights are initialised to zero/identity defaults.
/// Use [`crate::model::weights::load_gpt2_block`] to populate from a checkpoint.
#[derive(Debug, Clone)]
pub struct Gpt2Model {
    /// Model hyperparameters.
    pub config: GptConfig,
    /// Token embedding table: `[vocab_size × n_embd]`.
    pub token_embed: TokenEmbedding,
    /// Positional embedding table: `[n_positions × n_embd]`.
    pub pos_embed: LearnedPositionalEmbedding,
    /// Transformer blocks.
    pub blocks: Vec<GptBlock>,
    /// Final layer normalisation.
    pub ln_f: LayerNorm,
}

impl Gpt2Model {
    // ── Constructor ──────────────────────────────────────────────────────

    /// Construct a GPT-2 model with zero-initialised weights.
    pub fn new(config: GptConfig) -> LmResult<Self> {
        config.validate()?;
        let token_embed = TokenEmbedding::new(config.vocab_size, config.n_embd)?;
        let pos_embed = LearnedPositionalEmbedding::new(config.n_positions, config.n_embd)?;
        let blocks: Vec<GptBlock> = (0..config.n_layers)
            .map(|_| {
                GptBlock::new(
                    config.n_embd,
                    config.n_heads,
                    config.ffn_intermediate,
                    config.layer_norm_eps,
                )
            })
            .collect::<LmResult<_>>()?;
        let ln_f = LayerNorm::new(config.n_embd, config.layer_norm_eps)?;
        Ok(Self {
            config,
            token_embed,
            pos_embed,
            blocks,
            ln_f,
        })
    }

    // ── Forward pass ─────────────────────────────────────────────────────

    /// Full forward pass.
    ///
    /// # Arguments
    ///
    /// - `token_ids`: token sequence to process.
    /// - `past_kv`: optional KV cache from previous steps.
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
                context: "Gpt2Model::forward token_ids",
            });
        }
        let past_len = past_kv.map_or(0, |c| c.past_len());
        let total_len = past_len + seq_len;
        if total_len > self.config.n_positions {
            return Err(LmError::SequenceTooLong {
                total_len,
                max_pos: self.config.n_positions,
            });
        }

        // ── 1. Embeddings ─────────────────────────────────────────────────
        let tok_emb = self.token_embed.forward(token_ids)?;
        let pos_emb = self.pos_embed.forward(seq_len, past_len)?;
        // h = tok_emb + pos_emb
        let mut h: Vec<f32> = tok_emb
            .iter()
            .zip(pos_emb.iter())
            .map(|(&t, &p)| t + p)
            .collect();

        // ── 2. Transformer blocks ─────────────────────────────────────────
        let n_kv_heads = self.config.n_heads; // GPT-2 is full MHA
        let head_dim = self.config.head_dim();
        let mut new_kv = PastKvCache::new(self.config.n_layers, n_kv_heads, head_dim);

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let past_layer = past_kv.and_then(|c| c.layer(layer_idx).ok());
            let (block_out, layer_kv) = block.forward(&h, seq_len, past_layer)?;
            h = block_out;
            *new_kv.layer_mut(layer_idx)? = layer_kv;
        }

        // ── 3. Final LayerNorm ────────────────────────────────────────────
        let h = self.ln_f.forward(&h, seq_len)?;

        // ── 4. LM head (weight-tied to token_embed) ───────────────────────
        let logits = self.lm_head(&h, seq_len)?;

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
        // Logits for the last position
        let last_start = (seq_len - 1) * self.config.vocab_size;
        let last_logits = &logits[last_start..last_start + self.config.vocab_size];
        let next_tok = argmax_f32(last_logits)?;
        Ok((next_tok, new_kv))
    }

    // ── Utility ──────────────────────────────────────────────────────────

    /// Total parameter count (embedding tables + transformer blocks + ln_f).
    ///
    /// Note: the LM head is weight-tied to the token embedding, so it is
    /// counted only once.
    pub fn n_params(&self) -> usize {
        let embed_params = self.token_embed.vocab_size * self.token_embed.embed_dim
            + self.pos_embed.max_positions * self.pos_embed.embed_dim
            + self.ln_f.dim * 2; // weight + bias

        let block_params: usize = self
            .blocks
            .iter()
            .map(|b| {
                // ln_1 + ln_2 (weight+bias) + attn (4 proj) + ffn (2 proj+bias)
                let ln = 2 * b.ln_1.dim * 2;
                let hd = self.config.n_embd;
                let ffd = self.config.ffn_intermediate;
                let attn = 4 * hd * hd                 // w_q, w_k, w_v, w_o
                + 3 * hd + hd; // b_q, b_k, b_v, b_o
                let ffn = ffd * hd + ffd               // w_fc + b_fc
                + hd * ffd + hd; // w_proj + b_proj
                ln + attn + ffn
            })
            .sum();

        embed_params + block_params
    }

    // ── Private ───────────────────────────────────────────────────────────

    /// Apply the weight-tied LM head.
    ///
    /// `h` is `[seq_len × n_embd]`; output is `[seq_len × vocab_size]`.
    fn lm_head(&self, h: &[f32], seq_len: usize) -> LmResult<Vec<f32>> {
        let hd = self.config.n_embd;
        let vs = self.config.vocab_size;
        if h.len() != seq_len * hd {
            return Err(LmError::DimensionMismatch {
                expected: seq_len * hd,
                got: h.len(),
            });
        }
        // logits[t, v] = sum_d h[t, d] * embed[v, d]
        let embed = &self.token_embed.weight.data;
        let mut logits = vec![0.0_f32; seq_len * vs];
        for t in 0..seq_len {
            let h_row = &h[t * hd..(t + 1) * hd];
            let l_row = &mut logits[t * vs..(t + 1) * vs];
            for v in 0..vs {
                let emb_row = &embed[v * hd..(v + 1) * hd];
                l_row[v] = h_row
                    .iter()
                    .zip(emb_row.iter())
                    .map(|(&hi, &ei)| hi * ei)
                    .sum();
            }
        }
        Ok(logits)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Argmax of a `f32` slice; returns the index of the maximum element.
pub(crate) fn argmax_f32(x: &[f32]) -> LmResult<u32> {
    if x.is_empty() {
        return Err(LmError::EmptyInput {
            context: "argmax_f32",
        });
    }
    let (idx, _) =
        x.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
                if v > best_v { (i, v) } else { (best_i, best_v) }
            });
    Ok(idx as u32)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GptConfig;

    fn tiny_model() -> Gpt2Model {
        Gpt2Model::new(GptConfig::tiny()).unwrap()
    }

    #[test]
    fn gpt2_model_constructs() {
        let m = tiny_model();
        assert_eq!(m.blocks.len(), 2);
    }

    #[test]
    fn gpt2_forward_output_shape() {
        let m = tiny_model();
        let (logits, kv) = m.forward(&[0, 1, 2], None).unwrap();
        assert_eq!(logits.len(), 3 * m.config.vocab_size);
        assert_eq!(kv.n_layers(), 2);
        assert_eq!(kv.past_len(), 3);
    }

    #[test]
    fn gpt2_forward_empty_error() {
        let m = tiny_model();
        assert!(matches!(
            m.forward(&[], None),
            Err(LmError::EmptyInput { .. })
        ));
    }

    #[test]
    fn gpt2_forward_sequence_too_long_error() {
        let mut cfg = GptConfig::tiny();
        cfg.n_positions = 4;
        let m = Gpt2Model::new(cfg).unwrap();
        let ids: Vec<u32> = (0..5).collect();
        assert!(matches!(
            m.forward(&ids, None),
            Err(LmError::SequenceTooLong { .. })
        ));
    }

    #[test]
    fn gpt2_forward_kv_cache_incremental() {
        let m = tiny_model();
        let (_, kv1) = m.forward(&[0, 1], None).unwrap();
        let (logits2, kv2) = m.forward(&[2], Some(&kv1)).unwrap();
        assert_eq!(logits2.len(), m.config.vocab_size);
        assert_eq!(kv2.past_len(), 3);
    }

    #[test]
    fn gpt2_next_token_returns_valid_id() {
        let m = tiny_model();
        let (tok, _) = m.next_token(&[0], None).unwrap();
        assert!((tok as usize) < m.config.vocab_size);
    }

    #[test]
    fn gpt2_weight_tied_lm_head() {
        // If embedding weight = ones, logits for each vocab item = sum of h row
        let mut m = Gpt2Model::new(GptConfig::tiny()).unwrap();
        // set embedding to ones
        m.token_embed.weight.data = vec![1.0_f32; m.config.vocab_size * m.config.n_embd];
        // set all block weights to zero (already done) so h = tok+pos emb = 0
        let (logits, _) = m.forward(&[0], None).unwrap();
        // h = 0, logits = 0 @ embed = 0
        assert!(logits.iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn gpt2_n_params_positive() {
        let m = tiny_model();
        assert!(m.n_params() > 0);
    }

    #[test]
    fn gpt2_incremental_vs_full_last_position() {
        // Full forward of [0,1] then compare last-position logits
        // with incremental [0] then [1] with cache.
        // Both should be identical for the token-1 position.
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
    fn argmax_f32_correct() {
        assert_eq!(argmax_f32(&[0.1, 0.9, 0.5]).unwrap(), 1);
        assert_eq!(argmax_f32(&[5.0, 3.0]).unwrap(), 0);
    }

    #[test]
    fn argmax_f32_empty_error() {
        assert!(matches!(argmax_f32(&[]), Err(LmError::EmptyInput { .. })));
    }
}
