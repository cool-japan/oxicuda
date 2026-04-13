//! OxiCUDA Inference Engine — Vol.11.
//!
//! `oxicuda-infer` provides a production-grade GPU inference engine built on
//! the OxiCUDA stack.  It implements the core algorithms required for
//! efficient large-language-model serving:
//!
//! # Architecture
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────┐
//!  │               ContinuousBatcher                      │  ← orchestrator
//!  └────────────────────────┬─────────────────────────────┘
//!         ┌─────────────────┼──────────────────┐
//!  ┌──────▼──────┐  ┌───────▼──────┐  ┌────────▼──────┐
//!  │  Scheduler  │  │ CacheManager │  │ Sampling Suite │
//!  │  (FCFS +    │  │ PagedKvCache │  │  greedy        │
//!  │  preemption)│  │ PrefixCache  │  │  top-k / top-p │
//!  └─────────────┘  └──────────────┘  │  beam search   │
//!                                      │  speculative   │
//!                                      └────────────────┘
//!                                      ┌────────────────┐
//!                                      │  Executor      │
//!                                      │  ModelRunner   │
//!                                      │  PagedAttnCPU  │
//!                                      └────────────────┘
//! ```
//!
//! # Key Algorithms
//!
//! ## PagedAttention (Kwon et al., 2023)
//!
//! KV cache stored in fixed-size physical blocks accessed via a per-sequence
//! block table.  Enables non-contiguous memory allocation and O(1) prefix
//! sharing for prompt caching.
//!
//! ## Continuous Batching (vLLM, Yu et al., 2022)
//!
//! Sequences join and leave the GPU batch at token granularity rather than
//! waiting for the entire batch to finish.  Dramatically improves throughput
//! for variable-length generation.
//!
//! ## Speculative Decoding (Chen et al., 2023)
//!
//! A fast draft model generates `k` candidate tokens; the slow target model
//! verifies all `k+1` positions in a single pass.  Provably correct:
//! the output distribution is identical to target-only sampling.
//!
//! # Quick Start
//!
//! ```rust
//! use oxicuda_infer::batch::{BatcherConfig, ContinuousBatcher, SamplingParams};
//! use oxicuda_infer::cache::kv_cache::PagedKvCache;
//! use oxicuda_infer::error::InferResult;
//!
//! // 1. Build a KV cache (4 layers, 4 kv-heads, head_dim=64, block_size=16, 128 blocks).
//! let kv_cache = PagedKvCache::new(4, 4, 64, 16, 128);
//!
//! // 2. Construct the continuous batcher.
//! let mut batcher = ContinuousBatcher::new(BatcherConfig::default_test(), kv_cache);
//!
//! // 3. Submit a generation request.
//! let params = SamplingParams { max_new_tokens: 4, eos_token_id: Some(1), ..Default::default() };
//! let _seq_id = batcher.add_request(vec![10, 20, 30], params);
//!
//! // 4. Run decode steps until all sequences finish.
//! // (In a real engine, `model_fn` would call the actual GPU model.)
//! let model_fn = |tokens: &[u32], _btables: &[Vec<u32>], _lens: &[usize]| -> InferResult<Vec<Vec<f32>>> {
//!     Ok(tokens.iter().map(|_| {
//!         let mut v = vec![0.0_f32; 256];
//!         v[1] = 10.0;   // always predict EOS
//!         v
//!     }).collect())
//! };
//!
//! let output = batcher.step(model_fn).unwrap();
//! assert!(!output.is_empty());
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::module_name_repetitions)]

pub mod batch;
pub mod cache;
pub mod error;
pub mod executor;
pub mod handle;
pub mod ptx_kernels;
pub mod sampling;

// Re-export the most commonly used types.
pub use batch::{
    BatcherConfig, ContinuousBatcher, FinishReason, GenerationOutput, SamplingParams,
    ScheduledBatch, Scheduler, SchedulerConfig, Sequence, SequenceId, SequenceStatus,
};
pub use cache::{BlockId, CacheManager, KvBlock, PagedKvCache, PrefixCache, PrefixEntry};
pub use error::{InferError, InferResult};
pub use executor::{
    AttentionConfig, MockModelRunner, ModelRunner, RunnerStats, paged_attention_cpu,
};
pub use handle::InferHandle;
pub use sampling::{
    BeamHypothesis, BeamSearchConfig, BeamSearchState, Rng, greedy_sample, greedy_sample_batch,
    speculative_verify, top_k_filter, top_k_sample, top_p_filter, top_p_sample,
};

// ─── Integration tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end greedy generation: batcher generates until EOS.
    #[test]
    fn e2e_greedy_until_eos() {
        let vocab = 32_usize;
        let kv = PagedKvCache::new(2, 2, 16, 8, 64);
        let cfg = BatcherConfig {
            vocab_size: vocab,
            ..BatcherConfig::default_test()
        };
        let mut b = ContinuousBatcher::new(cfg, kv);
        let params = SamplingParams {
            eos_token_id: Some(5),
            max_new_tokens: 64,
            ..Default::default()
        };
        b.add_request(vec![1, 2, 3], params);

        // Model: always returns high logit for token 5 (EOS)
        let model_fn =
            |tokens: &[u32], _: &[Vec<u32>], _: &[usize]| -> InferResult<Vec<Vec<f32>>> {
                Ok(tokens
                    .iter()
                    .map(|_| {
                        let mut v = vec![0.0_f32; 32];
                        v[5] = 10.0;
                        v
                    })
                    .collect())
            };

        let outputs = b.step(model_fn).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].finish_reason, FinishReason::EosToken(5));
        assert!(!b.has_unfinished());
    }

    /// End-to-end: max_new_tokens termination.
    #[test]
    fn e2e_max_tokens_termination() {
        let kv = PagedKvCache::new(2, 2, 16, 8, 64);
        let cfg = BatcherConfig {
            vocab_size: 16,
            ..BatcherConfig::default_test()
        };
        let mut b = ContinuousBatcher::new(cfg, kv);
        let params = SamplingParams {
            max_new_tokens: 1,
            ..Default::default()
        };
        b.add_request(vec![0], params);

        let model_fn =
            |tokens: &[u32], _: &[Vec<u32>], _: &[usize]| -> InferResult<Vec<Vec<f32>>> {
                Ok(tokens.iter().map(|_| vec![1.0_f32; 16]).collect())
            };

        let outputs = b.step(model_fn).unwrap();
        assert_eq!(outputs[0].finish_reason, FinishReason::MaxLength);
    }

    /// Beam search completes on EOS.
    #[test]
    fn e2e_beam_search_completes() {
        let cfg = BeamSearchConfig {
            beam_width: 2,
            eos_token_id: 0,
            max_new_tokens: 8,
            length_penalty: 0.6,
        };
        let mut state = BeamSearchState::new(cfg);
        // All logits peak at token 0 (EOS) → all beams finish in one step.
        let logits: Vec<Vec<f32>> = (0..2)
            .map(|_| {
                let mut v = vec![0.0_f32; 8];
                v[0] = 10.0;
                v
            })
            .collect();
        let done = state.step(&logits).unwrap();
        assert!(done);
        assert!(!state.completed.is_empty());
    }

    /// Speculative decoding: when draft == target, all drafts accepted.
    #[test]
    fn e2e_speculative_all_accepted() {
        let vocab = 8_usize;
        let k = 3_usize;
        let probs: Vec<Vec<f32>> = (0..k)
            .map(|i| {
                let mut v = vec![0.0_f32; vocab];
                v[i % vocab] = 1.0;
                v
            })
            .collect();
        let mut target = probs.clone();
        target.push({
            let mut v = vec![0.0_f32; vocab];
            v[k % vocab] = 1.0;
            v
        });
        let draft: Vec<u32> = (0..k as u32).map(|i| i % vocab as u32).collect();
        let mut rng = Rng::new(0);
        let (accepted, _bonus) = speculative_verify(&draft, &probs, &target, &mut rng).unwrap();
        assert_eq!(accepted.len(), k);
    }

    /// PagedAttention: single-token sequence output equals V.
    #[test]
    fn e2e_paged_attention_single_token() {
        let n_h = 2;
        let hd = 4;
        let bs = 4;
        let mut cache = PagedKvCache::new(1, n_h, hd, bs, 4);
        let id = cache.alloc_block().unwrap();
        let kv = vec![1.0_f32; n_h * hd];
        cache.append_token(id, 0, &kv, &kv).unwrap();
        let q = vec![1.0_f32; n_h * hd];
        let out = paged_attention_cpu(&q, &cache, &[id], 1, 0, n_h, n_h, hd, bs, 1.0).unwrap();
        for &v in &out {
            assert!((v - 1.0_f32).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    /// Prefix cache: hit rate is correctly computed across hits and misses.
    #[test]
    fn e2e_prefix_cache_hit_rate() {
        let mut cache = PrefixCache::new(32);
        let t = vec![1_u32, 2, 3, 4];
        cache.insert(&t, vec![BlockId(0), BlockId(1)]);
        cache.lookup(&[99_u32, 88]); // miss
        cache.lookup(&t); // hit
        cache.lookup(&t); // hit
        // 3 total queries, 2 hits → hit_rate = 2/3
        assert!(
            (cache.hit_rate() - 2.0 / 3.0).abs() < 0.01,
            "got {}",
            cache.hit_rate()
        );
    }

    /// MockModelRunner: decode returns correct shape.
    #[test]
    fn e2e_mock_runner_decode() {
        let runner = MockModelRunner::new(64, 0);
        let logits = runner
            .decode(&[1, 2, 3], &[vec![], vec![], vec![]], &[0, 0, 0])
            .unwrap();
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|row| row.len() == 64));
    }
}
