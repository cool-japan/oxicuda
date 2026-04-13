//! # Model Runner
//!
//! Defines the [`ModelRunner`] trait and provides a [`MockModelRunner`] for
//! testing the inference pipeline without a real GPU model.
//!
//! ## Trait contract
//!
//! A `ModelRunner` receives:
//!
//! * `token_ids`    — one input token per sequence (`[n_seqs]`).
//! * `block_tables` — per-sequence KV cache block table (`[n_seqs][n_blocks]`).
//! * `seq_lens`     — number of tokens already in the KV cache per sequence.
//!
//! It returns logits of shape `[n_seqs][vocab_size]`.
//!
//! A separate `prefill` entry point takes `seq_starts` offsets so that
//! variable-length prompts can be processed in a single batched pass.

use crate::error::{InferError, InferResult};

// ─── ModelRunner trait ────────────────────────────────────────────────────────

/// Abstraction over a transformer forward pass for the inference engine.
pub trait ModelRunner: Send {
    /// Vocabulary size produced by this model.
    fn vocab_size(&self) -> usize;

    /// Process a batch of sequences in **decode** mode (one input token each).
    ///
    /// # Parameters
    ///
    /// * `token_ids`    — `[n_seqs]` last token for each sequence.
    /// * `block_tables` — `[n_seqs][n_blocks]` physical KV block IDs.
    /// * `seq_lens`     — `[n_seqs]` number of KV tokens already cached.
    ///
    /// # Returns
    ///
    /// `[n_seqs][vocab_size]` logits.
    fn decode(
        &self,
        token_ids: &[u32],
        block_tables: &[Vec<u32>],
        seq_lens: &[usize],
    ) -> InferResult<Vec<Vec<f32>>>;

    /// Process a batch of sequences in **prefill** mode (full prompt).
    ///
    /// # Parameters
    ///
    /// * `token_ids`  — concatenated prompt tokens for all sequences.
    /// * `seq_starts` — start offsets into `token_ids` for each sequence
    ///   (length = `n_seqs + 1`; last element = total tokens).
    /// * `block_tables` — `[n_seqs][n_blocks]` physical KV block IDs.
    ///
    /// # Returns
    ///
    /// `[n_seqs][vocab_size]` logits (one row = logits at last prompt token).
    fn prefill(
        &self,
        token_ids: &[u32],
        seq_starts: &[usize],
        block_tables: &[Vec<u32>],
    ) -> InferResult<Vec<Vec<f32>>>;
}

// ─── MockModelRunner ─────────────────────────────────────────────────────────

/// Deterministic mock model runner for unit-testing the inference pipeline.
///
/// Produces logits determined by `token_id % vocab_size` — i.e., the
/// sequence `0, 1, 2, …, vocab_size-1, 0, 1, …` always peaking at a
/// predictable token.
///
/// This makes test expectations reproducible without any randomness or
/// real model weights.
pub struct MockModelRunner {
    pub vocab_size: usize,
    /// Bias: the "hot" token for position `i` is `(token_id + bias) % vocab_size`.
    pub bias: u32,
}

impl MockModelRunner {
    /// Create a mock runner that peaks at `(input_token + bias) % vocab_size`.
    #[must_use]
    pub fn new(vocab_size: usize, bias: u32) -> Self {
        Self { vocab_size, bias }
    }

    fn make_logits_for_token(&self, token_id: u32) -> Vec<f32> {
        let mut logits = vec![0.0_f32; self.vocab_size];
        let hot = (token_id as usize + self.bias as usize) % self.vocab_size;
        logits[hot] = 10.0;
        logits
    }
}

impl ModelRunner for MockModelRunner {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn decode(
        &self,
        token_ids: &[u32],
        block_tables: &[Vec<u32>],
        seq_lens: &[usize],
    ) -> InferResult<Vec<Vec<f32>>> {
        if token_ids.is_empty() {
            return Err(InferError::EmptyBatch);
        }
        if token_ids.len() != block_tables.len() || token_ids.len() != seq_lens.len() {
            return Err(InferError::DimensionMismatch {
                expected: token_ids.len(),
                got: block_tables.len(),
            });
        }
        Ok(token_ids
            .iter()
            .map(|&t| self.make_logits_for_token(t))
            .collect())
    }

    fn prefill(
        &self,
        token_ids: &[u32],
        seq_starts: &[usize],
        block_tables: &[Vec<u32>],
    ) -> InferResult<Vec<Vec<f32>>> {
        if token_ids.is_empty() || seq_starts.len() < 2 {
            return Err(InferError::EmptyBatch);
        }
        let n_seqs = seq_starts.len() - 1;
        if n_seqs != block_tables.len() {
            return Err(InferError::DimensionMismatch {
                expected: n_seqs,
                got: block_tables.len(),
            });
        }
        // Return logits based on the last token of each sequence.
        let mut out = Vec::with_capacity(n_seqs);
        for i in 0..n_seqs {
            let start = seq_starts[i];
            let end = seq_starts[i + 1];
            if start >= end || end > token_ids.len() {
                return Err(InferError::DimensionMismatch {
                    expected: token_ids.len(),
                    got: end,
                });
            }
            let last_token = token_ids[end - 1];
            out.push(self.make_logits_for_token(last_token));
        }
        Ok(out)
    }
}

// ─── RunnerStats ─────────────────────────────────────────────────────────────

/// Aggregate statistics collected during a multi-step inference run.
#[derive(Debug, Default, Clone)]
pub struct RunnerStats {
    /// Total forward-pass calls.
    pub n_steps: u64,
    /// Total tokens processed (sum of batch sizes).
    pub total_tokens: u64,
    /// Total sequences completed.
    pub sequences_completed: u64,
}

impl RunnerStats {
    /// Average tokens per step.
    #[must_use]
    pub fn avg_batch_size(&self) -> f64 {
        if self.n_steps == 0 {
            0.0
        } else {
            self.total_tokens as f64 / self.n_steps as f64
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runner() -> MockModelRunner {
        MockModelRunner::new(32, 0)
    }

    #[test]
    fn vocab_size_correct() {
        let r = make_runner();
        assert_eq!(r.vocab_size(), 32);
    }

    #[test]
    fn decode_returns_one_row_per_seq() {
        let r = make_runner();
        let logits = r
            .decode(&[0, 1, 2], &[vec![], vec![], vec![]], &[0, 1, 2])
            .unwrap();
        assert_eq!(logits.len(), 3);
        assert_eq!(logits[0].len(), 32);
    }

    #[test]
    fn decode_peaks_at_expected_token() {
        let r = MockModelRunner::new(8, 3);
        // token_id=2, bias=3 → hot = (2+3)%8 = 5
        let logits = r.decode(&[2], &[vec![]], &[0]).unwrap();
        let argmax = logits[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(argmax, 5);
    }

    #[test]
    fn decode_empty_error() {
        let r = make_runner();
        assert!(matches!(
            r.decode(&[], &[], &[]),
            Err(InferError::EmptyBatch)
        ));
    }

    #[test]
    fn decode_dimension_mismatch() {
        let r = make_runner();
        assert!(r.decode(&[0, 1], &[vec![]], &[0, 1]).is_err());
    }

    #[test]
    fn prefill_returns_one_row_per_seq() {
        let r = make_runner();
        // 2 sequences: first has tokens [0,1,2], second has [3,4]
        let tokens = vec![0_u32, 1, 2, 3, 4];
        let starts = vec![0, 3, 5];
        let btables = vec![vec![], vec![]];
        let logits = r.prefill(&tokens, &starts, &btables).unwrap();
        assert_eq!(logits.len(), 2);
    }

    #[test]
    fn prefill_uses_last_token() {
        // Sequence tokens [10, 20, 99]; last = 99 → hot = 99%32 = 3.
        let r = MockModelRunner::new(32, 0);
        let tokens = vec![10_u32, 20, 99];
        let starts = vec![0, 3];
        let btables = vec![vec![]];
        let logits = r.prefill(&tokens, &starts, &btables).unwrap();
        let argmax = logits[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(argmax, 99 % 32);
    }

    #[test]
    fn runner_stats_avg_batch() {
        let s = RunnerStats {
            n_steps: 10,
            total_tokens: 40,
            ..RunnerStats::default()
        };
        assert!((s.avg_batch_size() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn runner_stats_zero_steps() {
        let s = RunnerStats::default();
        assert_eq!(s.avg_batch_size(), 0.0);
    }
}
