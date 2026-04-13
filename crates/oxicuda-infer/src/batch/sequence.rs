//! # Sequence State Machine
//!
//! Models a single inference request as a `Sequence` that transitions through
//! a well-defined state machine:
//!
//! ```text
//!  Waiting ─► Prefill ─► Decode ─► Finished
//!     ▲                    │
//!     │        ◄── Preempted ◄┘   (KV eviction, re-added to Waiting on resume)
//! ```
//!
//! * **Waiting**   — request is queued but the KV cache is not yet allocated.
//! * **Prefill**   — prompt tokens are being processed (one iteration).
//! * **Decode**    — autoregressively generating one token per step.
//! * **Preempted** — KV blocks were reclaimed; sequence must be re-prefilled.
//! * **Finished**  — generation is complete (EOS, max tokens, or stop string).

use crate::cache::kv_cache::BlockId;

// ─── SequenceId ───────────────────────────────────────────────────────────────

/// Unique identifier for a generation request.
pub type SequenceId = u64;

// ─── FinishReason ─────────────────────────────────────────────────────────────

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Model emitted the designated end-of-sequence token.
    EosToken(u32),
    /// The `max_new_tokens` limit was reached.
    MaxLength,
    /// A stop-string was detected in the output (token-level approximation).
    StopToken(u32),
}

// ─── SequenceStatus ──────────────────────────────────────────────────────────

/// Current lifecycle phase of a `Sequence`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Queued, awaiting KV cache allocation.
    Waiting,
    /// Prompt tokens are being prefilled.
    Prefill,
    /// Generating one token per decode step.
    Decode,
    /// Temporarily removed from the GPU; will be re-admitted when memory allows.
    Preempted,
    /// Generation complete.
    Finished(FinishReason),
}

impl SequenceStatus {
    /// Is the sequence currently runnable (Prefill or Decode)?
    #[must_use]
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Prefill | Self::Decode)
    }

    /// Has the sequence finished?
    #[must_use]
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Finished(_))
    }
}

// ─── SamplingParams ──────────────────────────────────────────────────────────

/// Per-request sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Softmax temperature (1.0 = no scaling; < 1.0 sharpens, > 1.0 flattens).
    pub temperature: f32,
    /// Top-K filter: keep only the K most likely tokens, or `None` to disable.
    pub top_k: Option<usize>,
    /// Nucleus (top-p) filter: smallest set with cumulative prob ≥ p, or `None`.
    pub top_p: Option<f32>,
    /// Maximum number of new tokens to generate (does not include prompt).
    pub max_new_tokens: usize,
    /// Token ID that signals end-of-sequence; generation stops on this token.
    pub eos_token_id: Option<u32>,
    /// Repetition penalty ≥ 1.0 (1.0 = no penalty; > 1.0 discourages repeats).
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            max_new_tokens: 512,
            eos_token_id: None,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingParams {
    /// Greedy (argmax) sampling configuration.
    #[must_use]
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            temperature: 0.0,
            max_new_tokens,
            ..Default::default()
        }
    }

    /// Standard top-p / nucleus sampling.
    #[must_use]
    pub fn nucleus(top_p: f32, temperature: f32, max_new_tokens: usize) -> Self {
        Self {
            temperature,
            top_p: Some(top_p),
            max_new_tokens,
            ..Default::default()
        }
    }
}

// ─── Sequence ────────────────────────────────────────────────────────────────

/// A single generation request tracked through its full lifecycle.
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique request identifier.
    pub id: SequenceId,
    /// Current lifecycle status.
    pub status: SequenceStatus,
    /// Input prompt tokens (immutable after construction).
    pub prompt_tokens: Vec<u32>,
    /// Autoregressively generated output tokens (grows during Decode).
    pub output_tokens: Vec<u32>,
    /// Physical KV cache blocks assigned to this sequence (block table).
    pub block_table: Vec<BlockId>,
    /// Sampling configuration for this request.
    pub sampling_params: SamplingParams,
    /// Cumulative log-probability of the output sequence (for scoring).
    pub cumulative_logprob: f64,
}

impl Sequence {
    /// Create a new sequence in the `Waiting` state.
    #[must_use]
    pub fn new(id: SequenceId, prompt_tokens: Vec<u32>, sampling_params: SamplingParams) -> Self {
        Self {
            id,
            status: SequenceStatus::Waiting,
            prompt_tokens,
            output_tokens: Vec::new(),
            block_table: Vec::new(),
            sampling_params,
            cumulative_logprob: 0.0,
        }
    }

    /// Total tokens in the KV cache (prompt + generated).
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// The last token that the model should condition on.
    ///
    /// During prefill this is the last prompt token; during decode it is the
    /// most recently generated token.
    #[must_use]
    pub fn last_token(&self) -> u32 {
        if let Some(&t) = self.output_tokens.last() {
            t
        } else {
            *self.prompt_tokens.last().unwrap_or(&0)
        }
    }

    /// Append one generated token and update cumulative log-probability.
    pub fn append_token(&mut self, token: u32, log_prob: f64) {
        self.output_tokens.push(token);
        self.cumulative_logprob += log_prob;
    }

    /// Check whether the most recent output token triggers termination.
    ///
    /// Returns the finish reason if generation should stop, `None` otherwise.
    #[must_use]
    pub fn check_finish(&self) -> Option<FinishReason> {
        let n_out = self.output_tokens.len();
        if n_out == 0 {
            return None;
        }
        let last = *self.output_tokens.last().expect("n_out > 0");

        // EOS token check.
        if let Some(eos) = self.sampling_params.eos_token_id {
            if last == eos {
                return Some(FinishReason::EosToken(eos));
            }
        }
        // Max new tokens check.
        if n_out >= self.sampling_params.max_new_tokens {
            return Some(FinishReason::MaxLength);
        }
        None
    }

    /// Transition to Finished with the given reason.
    pub fn finish(&mut self, reason: FinishReason) {
        self.status = SequenceStatus::Finished(reason);
    }

    /// Transition from Prefill → Decode (called after the prompt is processed).
    pub fn start_decode(&mut self) {
        if self.status == SequenceStatus::Prefill {
            self.status = SequenceStatus::Decode;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seq(id: u64, n_prompt: usize) -> Sequence {
        Sequence::new(
            id,
            (0..n_prompt as u32).collect(),
            SamplingParams::default(),
        )
    }

    #[test]
    fn initial_status_is_waiting() {
        let s = make_seq(1, 4);
        assert_eq!(s.status, SequenceStatus::Waiting);
        assert!(!s.status.is_running());
        assert!(!s.status.is_finished());
    }

    #[test]
    fn last_token_during_prefill() {
        let s = make_seq(1, 4);
        assert_eq!(s.last_token(), 3); // last prompt token = n_prompt-1
    }

    #[test]
    fn last_token_during_decode() {
        let mut s = make_seq(1, 4);
        s.append_token(99, -0.5);
        assert_eq!(s.last_token(), 99);
    }

    #[test]
    fn append_token_grows_output_and_logprob() {
        let mut s = make_seq(1, 4);
        s.append_token(10, -1.0);
        s.append_token(20, -2.0);
        assert_eq!(s.output_tokens, vec![10, 20]);
        assert!((s.cumulative_logprob - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn check_finish_eos() {
        let params = SamplingParams {
            eos_token_id: Some(2),
            max_new_tokens: 512,
            ..Default::default()
        };
        let mut s = Sequence::new(1, vec![0, 1], params);
        s.append_token(2, -0.1);
        assert_eq!(s.check_finish(), Some(FinishReason::EosToken(2)));
    }

    #[test]
    fn check_finish_max_length() {
        let params = SamplingParams {
            max_new_tokens: 2,
            ..Default::default()
        };
        let mut s = Sequence::new(1, vec![0], params);
        s.append_token(5, 0.0);
        assert_eq!(s.check_finish(), None);
        s.append_token(6, 0.0);
        assert_eq!(s.check_finish(), Some(FinishReason::MaxLength));
    }

    #[test]
    fn total_len() {
        let mut s = make_seq(1, 3);
        assert_eq!(s.total_len(), 3);
        s.append_token(100, 0.0);
        assert_eq!(s.total_len(), 4);
    }

    #[test]
    fn start_decode_transitions() {
        let mut s = make_seq(1, 2);
        s.status = SequenceStatus::Prefill;
        s.start_decode();
        assert_eq!(s.status, SequenceStatus::Decode);
        assert!(s.status.is_running());
    }

    #[test]
    fn finish_marks_as_done() {
        let mut s = make_seq(1, 2);
        s.finish(FinishReason::MaxLength);
        assert!(s.status.is_finished());
    }
}
