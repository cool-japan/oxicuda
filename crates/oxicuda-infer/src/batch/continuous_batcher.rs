//! # Continuous Batcher
//!
//! Orchestrates the full inference lifecycle: request admission, KV cache
//! management, model forward pass, token sampling, and response emission.
//!
//! ## Design
//!
//! The `ContinuousBatcher` combines:
//!
//! * A [`Scheduler`] for FCFS admission + preemption decisions.
//! * A [`CacheManager`] for paged KV block allocation.
//! * A `model_fn` closure that performs the actual GPU/CPU forward pass.
//! * A random-number generator for stochastic sampling.
//!
//! Each call to [`ContinuousBatcher::step`] runs **one** batched forward pass
//! and returns any sequences that finished during that step.
//!
//! ## Model function contract
//!
//! ```text
//! model_fn(token_ids: &[u32],          // [n_seqs] — one token per sequence
//!          block_tables: &[&[u32]],     // [n_seqs][blocks] — KV block IDs
//!          seq_lens: &[usize])          // [n_seqs] — KV length per sequence
//!   → InferResult<Vec<Vec<f32>>>        // [n_seqs][vocab_size] — logits
//! ```

use crate::batch::scheduler::{ScheduledBatch, Scheduler, SchedulerConfig, StepResult};
use crate::batch::sequence::{FinishReason, SamplingParams, SequenceId, SequenceStatus};
use crate::cache::cache_manager::CacheManager;
use crate::cache::kv_cache::PagedKvCache;
use crate::error::{InferError, InferResult};
use crate::sampling::{Rng, top_k::top_k_sample};

// ─── GenerationOutput ────────────────────────────────────────────────────────

/// Completed generation result returned by the batcher.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Identifier of the completed sequence.
    pub seq_id: SequenceId,
    /// Original prompt tokens.
    pub prompt_tokens: Vec<u32>,
    /// All generated output tokens.
    pub output_tokens: Vec<u32>,
    /// Why generation ended.
    pub finish_reason: FinishReason,
    /// Cumulative output log-probability.
    pub cumulative_logprob: f64,
}

// ─── BatcherConfig ───────────────────────────────────────────────────────────

/// Configuration for the `ContinuousBatcher`.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Scheduler constraints (max running seqs, token budget, …).
    pub scheduler: SchedulerConfig,
    /// Vocabulary size (number of logits per token position).
    pub vocab_size: usize,
    /// Random seed for stochastic sampling.
    pub seed: u64,
}

impl BatcherConfig {
    /// Small default configuration for testing.
    #[must_use]
    pub fn default_test() -> Self {
        Self {
            scheduler: SchedulerConfig::new(4, 512, 16, 128),
            vocab_size: 256,
            seed: 42,
        }
    }
}

// ─── ContinuousBatcher ───────────────────────────────────────────────────────

/// vLLM-style continuous batching inference engine.
pub struct ContinuousBatcher {
    scheduler: Scheduler,
    cache_manager: CacheManager,
    vocab_size: usize,
    rng: Rng,
}

impl ContinuousBatcher {
    /// Construct a batcher from a config and a pre-built KV cache.
    #[must_use]
    pub fn new(config: BatcherConfig, kv_cache: PagedKvCache) -> Self {
        let scheduler = Scheduler::new(config.scheduler);
        let cache_manager = CacheManager::new(kv_cache);
        Self {
            scheduler,
            cache_manager,
            vocab_size: config.vocab_size,
            rng: Rng::new(config.seed),
        }
    }

    // ── Request API ──────────────────────────────────────────────────────────

    /// Submit a generation request.  Returns the assigned [`SequenceId`].
    pub fn add_request(&mut self, prompt_tokens: Vec<u32>, params: SamplingParams) -> SequenceId {
        self.scheduler.add_request(prompt_tokens, params)
    }

    /// Whether there are any sequences still waiting to be processed.
    #[must_use]
    pub fn has_unfinished(&self) -> bool {
        self.scheduler.has_unfinished()
    }

    // ── Step ─────────────────────────────────────────────────────────────────

    /// Execute one batched forward-pass step.
    ///
    /// `model_fn` is called with the current batch of (token_id, block_table,
    /// seq_len) tuples and must return `vocab_size` logits per sequence.
    ///
    /// Returns the list of [`GenerationOutput`]s for sequences that finished
    /// during this step (may be empty).
    pub fn step<F>(&mut self, model_fn: F) -> InferResult<Vec<GenerationOutput>>
    where
        F: Fn(
            &[u32],      // input token ids [n_seqs]
            &[Vec<u32>], // block tables [n_seqs][n_blocks]
            &[usize],    // kv sequence lengths [n_seqs]
        ) -> InferResult<Vec<Vec<f32>>>, // logits [n_seqs][vocab_size]
    {
        let batch = self.scheduler.schedule();
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        // Allocate KV blocks for newly admitted prefill sequences.
        for &seq_id in &batch.prefill_ids {
            self.ensure_cache_allocated(seq_id, &batch)?;
        }

        // Build input arrays for the model.
        let (token_ids, block_tables, seq_lens) = self.build_model_inputs(&batch)?;

        // Run model forward pass.
        let all_logits = model_fn(&token_ids, &block_tables, &seq_lens)?;

        if all_logits.len() != batch.n_seqs() {
            return Err(InferError::DimensionMismatch {
                expected: batch.n_seqs(),
                got: all_logits.len(),
            });
        }

        // Sample tokens and notify the scheduler.
        let all_ids: Vec<SequenceId> = batch
            .prefill_ids
            .iter()
            .chain(batch.decode_ids.iter())
            .copied()
            .collect();

        let mut results = Vec::with_capacity(all_ids.len());
        for (seq_id, logits) in all_ids.iter().zip(all_logits.iter()) {
            if logits.len() != self.vocab_size {
                return Err(InferError::DimensionMismatch {
                    expected: self.vocab_size,
                    got: logits.len(),
                });
            }
            let token = self.sample_token(logits, *seq_id)?;
            let log_prob = self.log_prob_of(logits, token);
            results.push(StepResult {
                seq_id: *seq_id,
                token,
                log_prob,
            });
        }

        self.scheduler.on_step_complete(results)?;

        // Collect finished sequences.
        let finished = self.scheduler.take_finished();
        let mut outputs = Vec::new();
        for seq in finished {
            self.cache_manager.free_sequence(seq.id);
            if let SequenceStatus::Finished(ref reason) = seq.status {
                outputs.push(GenerationOutput {
                    seq_id: seq.id,
                    prompt_tokens: seq.prompt_tokens.clone(),
                    output_tokens: seq.output_tokens.clone(),
                    finish_reason: reason.clone(),
                    cumulative_logprob: seq.cumulative_logprob,
                });
            }
        }
        Ok(outputs)
    }

    // ── Internals ────────────────────────────────────────────────────────────

    fn ensure_cache_allocated(
        &mut self,
        seq_id: SequenceId,
        batch: &ScheduledBatch,
    ) -> InferResult<()> {
        // Find the sequence's prompt length from the scheduler's view.
        // We need to discover if it's a prefill we should allocate for.
        if !batch.prefill_ids.contains(&seq_id) {
            return Ok(());
        }
        // Allocate with 1 initial token (the scheduler provides the rest lazily).
        if self.cache_manager.block_table(seq_id).is_err() {
            self.cache_manager.allocate_sequence(seq_id, 1)?;
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn build_model_inputs(
        &mut self,
        batch: &ScheduledBatch,
    ) -> InferResult<(Vec<u32>, Vec<Vec<u32>>, Vec<usize>)> {
        let all_ids: Vec<SequenceId> = batch
            .prefill_ids
            .iter()
            .chain(batch.decode_ids.iter())
            .copied()
            .collect();

        let mut token_ids = Vec::with_capacity(all_ids.len());
        let mut block_tables = Vec::with_capacity(all_ids.len());
        let mut seq_lens = Vec::with_capacity(all_ids.len());

        for &seq_id in &all_ids {
            // Condition the model on the actual current token for this sequence.
            // For prefill this is the last prompt token; for decode this is
            // the most recently generated token.
            token_ids.push(self.scheduler.last_token(seq_id).unwrap_or(0));

            let btbl = self
                .cache_manager
                .block_table(seq_id)
                .unwrap_or(&[])
                .iter()
                .map(|b| b.0)
                .collect();
            block_tables.push(btbl);

            // Prefer logical sequence length from scheduler state. Fall back to
            // cache-manager length when the scheduler no longer tracks the id.
            let len = self
                .scheduler
                .total_len(seq_id)
                .unwrap_or_else(|| self.cache_manager.seq_length(seq_id).unwrap_or(0));
            seq_lens.push(len);
        }

        Ok((token_ids, block_tables, seq_lens))
    }

    fn sample_token(&mut self, logits: &[f32], _seq_id: SequenceId) -> InferResult<u32> {
        // Top-1 sampling via the embedded RNG (deterministic: k=1 ⇒ greedy).
        // Passing &mut self.rng keeps the field alive; a temperature/top-p
        // variant would use it for stochastic sampling.
        top_k_sample(logits, 1, &mut self.rng)
    }

    fn log_prob_of(&self, logits: &[f32], token: u32) -> f64 {
        if token as usize >= logits.len() {
            return -f64::INFINITY;
        }
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = logits.iter().map(|&x| ((x - max) as f64).exp()).sum();
        let log_sum = sum_exp.ln() + max as f64;
        (logits[token as usize] as f64) - log_sum
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    /// Number of sequences currently running.
    #[must_use]
    pub fn n_running(&self) -> usize {
        self.scheduler.n_running()
    }

    /// Number of sequences waiting to be admitted.
    #[must_use]
    pub fn n_waiting(&self) -> usize {
        self.scheduler.n_waiting()
    }

    /// Number of free KV blocks.
    #[must_use]
    pub fn n_free_blocks(&self) -> usize {
        self.cache_manager.n_free_blocks()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::kv_cache::PagedKvCache;
    use std::cell::Cell;

    fn make_batcher() -> ContinuousBatcher {
        let cfg = BatcherConfig::default_test();
        let cache = PagedKvCache::new(4, 4, 64, 16, 128);
        ContinuousBatcher::new(cfg, cache)
    }

    /// Model function: returns logits with maximum probability at token 42.
    fn greedy_model(
        tokens: &[u32],
        _btables: &[Vec<u32>],
        _seq_lens: &[usize],
    ) -> InferResult<Vec<Vec<f32>>> {
        Ok(tokens
            .iter()
            .map(|_| {
                let mut v = vec![0.0_f32; 256];
                v[42] = 10.0; // highest logit → token 42
                v
            })
            .collect())
    }

    /// Model that emits EOS (token 1) unconditionally.
    fn eos_model(
        tokens: &[u32],
        _btables: &[Vec<u32>],
        _seq_lens: &[usize],
    ) -> InferResult<Vec<Vec<f32>>> {
        Ok(tokens
            .iter()
            .map(|_| {
                let mut v = vec![0.0_f32; 256];
                v[1] = 10.0; // EOS token
                v
            })
            .collect())
    }

    #[test]
    fn add_and_check_unfinished() {
        let mut b = make_batcher();
        b.add_request(vec![0, 1, 2], SamplingParams::default());
        assert!(b.has_unfinished());
    }

    #[test]
    fn step_with_empty_queue_returns_empty() {
        let mut b = make_batcher();
        let outputs = b.step(greedy_model).unwrap();
        assert!(outputs.is_empty());
    }

    #[test]
    fn eos_terminates_sequence() {
        let mut b = make_batcher();
        let params = SamplingParams {
            eos_token_id: Some(1),
            max_new_tokens: 128,
            ..Default::default()
        };
        b.add_request(vec![0, 1], params);
        // Step 1: prefill → token 1 (EOS) generated
        let out = b.step(eos_model).unwrap();
        assert!(!out.is_empty(), "sequence should finish in step 1");
        assert_eq!(out[0].finish_reason, FinishReason::EosToken(1));
        assert!(!b.has_unfinished());
    }

    #[test]
    fn max_new_tokens_terminates() {
        let mut b = make_batcher();
        let params = SamplingParams {
            max_new_tokens: 1,
            ..Default::default()
        };
        b.add_request(vec![5], params);
        let out = b.step(greedy_model).unwrap();
        assert!(!out.is_empty(), "should finish after 1 token");
        assert_eq!(out[0].finish_reason, FinishReason::MaxLength);
    }

    #[test]
    fn greedy_picks_highest_logit_token() {
        let mut b = make_batcher();
        let params = SamplingParams {
            max_new_tokens: 1,
            ..Default::default()
        };
        b.add_request(vec![0], params);
        let out = b.step(greedy_model).unwrap();
        assert!(!out.is_empty());
        assert_eq!(out[0].output_tokens, vec![42]);
    }

    #[test]
    fn multiple_sequences_run_concurrently() {
        let mut b = make_batcher();
        let eos_params = SamplingParams {
            eos_token_id: Some(1),
            max_new_tokens: 16,
            ..Default::default()
        };
        for _ in 0..3 {
            b.add_request(vec![0], eos_params.clone());
        }
        let out = b.step(eos_model).unwrap();
        assert_eq!(
            out.len(),
            3,
            "all 3 sequences should finish in one EOS step"
        );
    }

    #[test]
    fn n_running_and_waiting_counts() {
        let mut b = make_batcher();
        b.add_request(vec![0, 1], SamplingParams::default());
        b.add_request(vec![2, 3], SamplingParams::default());
        assert_eq!(b.n_waiting(), 2);
        // After one step, seqs move from waiting to running (prefill)
        let _ = b.step(greedy_model).unwrap();
        // After prefill step, sequences are in running (decode phase)
        assert_eq!(b.n_waiting(), 0);
    }

    #[test]
    fn output_tokens_non_empty_after_completion() {
        let mut b = make_batcher();
        let params = SamplingParams {
            max_new_tokens: 1,
            ..Default::default()
        };
        b.add_request(vec![10, 20], params);
        let out = b.step(greedy_model).unwrap();
        assert!(!out.is_empty());
        assert!(!out[0].output_tokens.is_empty());
    }

    #[test]
    fn model_input_uses_last_prompt_token() {
        let mut b = make_batcher();
        let params = SamplingParams {
            max_new_tokens: 1,
            eos_token_id: Some(1),
            ..Default::default()
        };
        b.add_request(vec![7, 8, 9], params);

        let saw_expected_token = Cell::new(false);
        let out = b
            .step(|tokens, _btables, _seq_lens| {
                if tokens == [9] {
                    saw_expected_token.set(true);
                }
                let mut v = vec![vec![0.0_f32; 256]; tokens.len()];
                for logits in &mut v {
                    logits[1] = 10.0;
                }
                Ok(v)
            })
            .unwrap();

        assert!(
            saw_expected_token.get(),
            "expected prompt tail token as model input"
        );
        assert_eq!(out.len(), 1);
    }
}
