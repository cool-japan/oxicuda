//! `oxicuda-lm` — Large Language Model inference primitives.
//!
//! This crate provides the **model-layer abstractions** for LLM inference:
//! a BPE tokenizer, transformer layer building blocks with KV-cache, and
//! complete GPT-2 and LLaMA-2/3 model implementations.
//!
//! # Architecture overview
//!
//! ```text
//!  ┌─────────────────────────────────────────────────┐
//!  │               oxicuda-lm                        │
//!  │                                                 │
//!  │  ┌────────────┐  ┌──────────────────────────┐  │
//!  │  │ tokenizer  │  │       layer              │  │
//!  │  │            │  │  ┌──────────────────────┐│  │
//!  │  │ BpeTokenizer│  │  │ TokenEmbedding       ││  │
//!  │  │ Vocab      │  │  │ RotaryEmbedding (RoPE)││  │
//!  │  └────────────┘  │  │ MultiHeadAttention   ││  │
//!  │                  │  │   + LayerKvCache      ││  │
//!  │  ┌────────────┐  │  │ MlpFfn / SwiGluFfn   ││  │
//!  │  │  config    │  │  │ RmsNorm / LayerNorm   ││  │
//!  │  │            │  │  │ GptBlock / LlamaBlock ││  │
//!  │  │ GptConfig  │  │  │ PastKvCache          ││  │
//!  │  │ LlamaConfig│  │  └──────────────────────┘│  │
//!  │  └────────────┘  └──────────────────────────┘  │
//!  │                                                 │
//!  │  ┌────────────────────────────────────────────┐│
//!  │  │                 model                      ││
//!  │  │  Gpt2Model  ─── forward → logits + cache   ││
//!  │  │  LlamaModel ─── forward → logits + cache   ││
//!  │  └────────────────────────────────────────────┘│
//!  │                                                 │
//!  │  ┌────────────────────────────────────────────┐│
//!  │  │  ptx_kernels (5 GPU kernel PTX strings)    ││
//!  │  │  weights (ModelWeights, WeightTensor)       ││
//!  │  └────────────────────────────────────────────┘│
//!  └─────────────────────────────────────────────────┘
//! ```
//!
//! # Design
//!
//! - **Pure Rust**: no C/CUDA SDK at compile time.
//! - **CPU reference implementations**: all forward passes are pure-Rust
//!   CPU implementations suitable for testing.  GPU acceleration is provided
//!   by the PTX kernel strings (see [`ptx_kernels`]) once a CUDA driver is
//!   available at runtime.
//! - **No unwrap()** in library code.
//! - **KV cache**: all attention layers return an updated [`layer::PastKvCache`]
//!   so incremental decoding is fully supported.

pub mod config;
pub mod error;
pub mod handle;
pub mod layer;
pub mod model;
pub mod ptx_kernels;
pub mod tokenizer;
pub mod weights;

// ── Convenient top-level re-exports ─────────────────────────────────────────

pub use config::{GptConfig, LlamaConfig};
pub use error::{LmError, LmResult};
pub use handle::{LmHandle, SmVersion};
pub use layer::{
    LayerKvCache, LayerNorm, LearnedPositionalEmbedding, MlpFfn, MultiHeadAttention, PastKvCache,
    RmsNorm, RotaryEmbedding, SwiGluFfn, TokenEmbedding,
};
pub use model::{Gpt2Model, LlamaModel};
pub use tokenizer::{BpeBuilder, BpeTokenizer, Vocab};
pub use weights::{ModelWeights, WeightTensor};

// ─── Integration tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── E2E 1: GPT-2 tiny forward pass ───────────────────────────────────

    #[test]
    fn e2e_gpt2_tiny_forward() {
        // Minimal GPT-2: 2 layers, 2 heads, hidden=8, vocab=16, max_pos=32
        let cfg = GptConfig::tiny();
        let m = Gpt2Model::new(cfg).unwrap();
        let token_ids: Vec<u32> = vec![0, 3, 7, 2, 5];
        let (logits, kv) = m.forward(&token_ids, None).unwrap();
        // logits shape: [seq_len × vocab_size] = [5 × 16]
        assert_eq!(logits.len(), 5 * 16);
        assert_eq!(kv.past_len(), 5);
        assert_eq!(kv.n_layers(), 2);
        // With zero weights, logits should all be zero (embeddings are zero).
        assert!(logits.iter().all(|&v| v.abs() < 1e-6));
    }

    // ── E2E 2: LLaMA tiny forward pass ───────────────────────────────────

    #[test]
    fn e2e_llama_tiny_forward() {
        let cfg = LlamaConfig::tiny();
        let m = LlamaModel::new(cfg).unwrap();
        let token_ids: Vec<u32> = vec![0, 1, 2, 3];
        let (logits, kv) = m.forward(&token_ids, None).unwrap();
        assert_eq!(logits.len(), 4 * 16);
        assert_eq!(kv.past_len(), 4);
        assert_eq!(kv.n_layers(), 2);
    }

    // ── E2E 3: GPT-2 incremental decode consistency ───────────────────────

    #[test]
    fn e2e_gpt2_incremental_decode_consistent() {
        // Full 3-token pass vs token-by-token with KV cache:
        // The last-position logits must match.
        let m = Gpt2Model::new(GptConfig::tiny()).unwrap();

        // Full pass
        let full_ids = vec![1u32, 2, 3];
        let (logits_full, _) = m.forward(&full_ids, None).unwrap();
        let vs = m.config.vocab_size;
        let last_full = logits_full[2 * vs..].to_vec();

        // Incremental
        let (_, kv0) = m.forward(&[1u32], None).unwrap();
        let (_, kv1) = m.forward(&[2u32], Some(&kv0)).unwrap();
        let (logits_3, _) = m.forward(&[3u32], Some(&kv1)).unwrap();

        assert_eq!(logits_3.len(), vs);
        for (&full_v, &incr_v) in last_full.iter().zip(logits_3.iter()) {
            assert!(
                (full_v - incr_v).abs() < 1e-4,
                "GPT-2 incremental mismatch: full={full_v} incr={incr_v}"
            );
        }
    }

    // ── E2E 4: LLaMA incremental decode consistency ───────────────────────

    #[test]
    fn e2e_llama_incremental_decode_consistent() {
        let m = LlamaModel::new(LlamaConfig::tiny()).unwrap();

        let full_ids = vec![0u32, 5, 10];
        let (logits_full, _) = m.forward(&full_ids, None).unwrap();
        let vs = m.config.vocab_size;
        let last_full = logits_full[2 * vs..].to_vec();

        let (_, kv0) = m.forward(&[0u32], None).unwrap();
        let (_, kv1) = m.forward(&[5u32], Some(&kv0)).unwrap();
        let (logits_3, _) = m.forward(&[10u32], Some(&kv1)).unwrap();

        for (&full_v, &incr_v) in last_full.iter().zip(logits_3.iter()) {
            assert!(
                (full_v - incr_v).abs() < 1e-4,
                "LLaMA incremental mismatch: full={full_v} incr={incr_v}"
            );
        }
    }

    // ── E2E 5: BPE tokenizer encode/decode round-trip ─────────────────────

    #[test]
    fn e2e_bpe_encode_decode_roundtrip() {
        // Build a small BPE tokenizer on top of 256 byte tokens.
        let t = BpeBuilder::new()
            .add_merge(b"h", b"e") // "he" → id 256
            .add_merge(b"l", b"l") // "ll" → id 257
            .add_merge(b"he", b"ll") // "hell" → id 258
            .add_merge(b"hell", b"o") // "hello" → id 259
            .build()
            .unwrap();

        let original = "hello";
        let ids = t.encode(original).unwrap();
        let decoded = t.decode(&ids).unwrap();
        assert_eq!(
            &decoded, original,
            "BPE round-trip failed: '{original}' → {ids:?} → '{decoded}'"
        );
        // Should be fully merged to a single token
        assert_eq!(
            ids,
            vec![259u32],
            "Expected full merge to one token, got {ids:?}"
        );
    }

    // ── E2E 6: RMSNorm and LayerNorm correctness ──────────────────────────

    #[test]
    fn e2e_rms_norm_and_layer_norm_correctness() {
        use crate::layer::{LayerNorm, RmsNorm};

        let dim = 8;
        // Random-ish input with a known structure
        let x: Vec<f32> = (0..dim).map(|i| i as f32 - 3.5).collect();
        // [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]

        // RMSNorm with weight=1: rms = sqrt(mean(x^2)) = sqrt(10.5) ≈ 3.240
        let rms_norm = RmsNorm::new(dim, 1e-8).unwrap();
        let rms_out = rms_norm.forward(&x, 1).unwrap();
        let expected_rms = 1.0 / (x.iter().map(|&v| v * v).sum::<f32>() / dim as f32 + 1e-8).sqrt();
        for (&o, &xi) in rms_out.iter().zip(x.iter()) {
            assert!(
                (o - xi * expected_rms).abs() < 1e-5,
                "RMSNorm out[i]={o} expected {}",
                xi * expected_rms
            );
        }

        // LayerNorm with weight=1, bias=0: output should have mean≈0, var≈1
        let ln = LayerNorm::new(dim, 1e-8).unwrap();
        let ln_out = ln.forward(&x, 1).unwrap();
        let mu: f32 = ln_out.iter().sum::<f32>() / dim as f32;
        let var: f32 = ln_out.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / dim as f32;
        assert!(mu.abs() < 1e-5, "LayerNorm mean={mu}");
        assert!((var - 1.0).abs() < 1e-4, "LayerNorm var={var}");
    }

    // ── E2E 7: PTX kernels for all SM versions ────────────────────────────

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        use crate::ptx_kernels::*;
        let sms = [75u32, 80, 86, 90, 100, 120];
        for sm in sms {
            let p1 = embedding_forward_ptx(sm);
            let p2 = rope_apply_ptx(sm);
            let p3 = silu_gate_ptx(sm);
            let p4 = rms_norm_ptx(sm);
            let p5 = causal_attn_softmax_ptx(sm);
            for (name, ptx) in [
                ("embedding_forward", &p1),
                ("rope_apply", &p2),
                ("silu_gate", &p3),
                ("rms_norm", &p4),
                ("causal_attn_softmax", &p5),
            ] {
                let target = format!("sm_{sm}");
                assert!(
                    ptx.contains(&target),
                    "SM {sm}: kernel '{name}' missing target directive"
                );
            }
        }
    }

    // ── E2E 8: GQA with LLaMA-3 style (4Q / 2KV) multi-step decode ────────

    #[test]
    fn e2e_llama_gqa_multistep_decode() {
        let m = LlamaModel::new(LlamaConfig::tiny()).unwrap();
        // Prefill phase: 4 tokens
        let prefill_ids = vec![0u32, 1, 2, 3];
        let (_, kv) = m.forward(&prefill_ids, None).unwrap();
        assert_eq!(kv.past_len(), 4);

        // Decode: 3 more tokens one at a time
        let mut cur_kv = kv;
        for step_tok in [4u32, 5, 6] {
            let (logits, new_kv) = m.forward(&[step_tok], Some(&cur_kv)).unwrap();
            assert_eq!(logits.len(), m.config.vocab_size);
            cur_kv = new_kv;
        }
        assert_eq!(cur_kv.past_len(), 7);
    }

    // ── E2E 9: Vocab special token round-trip ────────────────────────────

    #[test]
    fn e2e_vocab_special_token_roundtrip() {
        use std::collections::HashMap;
        let tokens = vec![vec![b'a'], vec![b'b'], vec![1u8, 0], vec![2u8, 0]];
        let special: HashMap<String, u32> = [("<bos>".into(), 2u32), ("<eos>".into(), 3u32)]
            .into_iter()
            .collect();
        let v = Vocab::from_tokens(tokens, special).unwrap();
        assert_eq!(v.special_id("<bos>"), Some(2));
        assert_eq!(v.special_id("<eos>"), Some(3));
        assert_eq!(v.bytes_to_id(b"a"), Some(0));
        assert_eq!(v.decode_token(0).unwrap(), "a");
    }

    // ── E2E 10: GPT-2 next_token greedy decode loop ───────────────────────

    #[test]
    fn e2e_gpt2_greedy_decode_loop() {
        // Run 5 steps of greedy decode with KV cache.
        let m = Gpt2Model::new(GptConfig::tiny()).unwrap();
        let mut token_ids = vec![0u32]; // start token
        let (_, mut kv) = m.forward(&token_ids, None).unwrap();

        for _ in 0..4 {
            let last_tok = *token_ids.last().unwrap();
            let (next_tok, new_kv) = m.next_token(&[last_tok], Some(&kv)).unwrap();
            token_ids.push(next_tok);
            kv = new_kv;
        }

        // Should have generated 5 tokens total (1 initial + 4 decoded)
        assert_eq!(token_ids.len(), 5);
        // All generated token ids must be in vocab range
        for &t in &token_ids {
            assert!((t as usize) < m.config.vocab_size);
        }
    }
}
