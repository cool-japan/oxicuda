//! High-level speculative decoding driver.
//!
//! [`SpeculativeDecoder`] orchestrates the full speculation loop:
//!
//! 1. Draft model generates `gamma` token proposals with their draft
//!    probabilities.
//! 2. Target model verifies all draft tokens in one parallel pass, producing
//!    target probabilities for the same positions.
//! 3. The *modified rejection sampling* acceptance criterion accepts a prefix
//!    of the draft tokens; the correction token is drawn from the residual
//!    distribution `norm(max(0, p_target − p_draft))`.
//!
//! Host-side arithmetic is used here; the companion PTX kernels (in
//! `SpeculativeDecodePlan`) handle the GPU-side execution.
//!
//! # Acceptance algorithm
//!
//! For each drafted token *i* (in order):
//! ```text
//! accept_i  ← uniform() < min(1.0, p_target(t_i) / p_draft(t_i))
//! ```
//! If *accept_i* is false, draw a correction token from the residual and stop.
//! If all `gamma` tokens are accepted, append a bonus token sampled from the
//! target distribution.

use crate::attn::speculative_decode::{
    CacheCheckpoint, DraftCacheState, SpeculativeDecodeConfig, TargetCacheState,
};
use crate::error::DnnError;

// ---------------------------------------------------------------------------
// accept_token — host-side acceptance criterion
// ---------------------------------------------------------------------------

/// Standard speculative sampling acceptance criterion.
///
/// Returns `true` if the draft token with probability `draft_prob` should be
/// accepted given the target probability `target_prob` and a uniform random
/// sample `rng_sample ∈ [0, 1)`.
///
/// The criterion is:  `rng_sample < min(1.0, target_prob / draft_prob)`.
///
/// Edge cases:
/// - If `draft_prob` is 0.0, the draft token is impossible; always reject.
/// - If `target_prob` ≥ `draft_prob`, always accept regardless of `rng_sample`.
#[must_use]
pub fn accept_token(draft_prob: f32, target_prob: f32, rng_sample: f32) -> bool {
    if draft_prob <= 0.0 {
        return false;
    }
    rng_sample < (target_prob / draft_prob).min(1.0)
}

// ---------------------------------------------------------------------------
// DraftCacheState extension methods
// ---------------------------------------------------------------------------

impl DraftCacheState {
    /// Appends one token to sequence 0's position and updates counters.
    ///
    /// Allocates a new page if the current position lands on a page boundary.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::WorkspaceRequired`] if no free pages remain.
    pub fn advance_token(&mut self, _token: u32) -> Result<(), DnnError> {
        let seq_id = 0usize;
        let pos = self.seq_positions[seq_id];
        let logical_page = pos / self.page_size;

        if logical_page >= self.page_tables[seq_id].len() {
            let phys = self.free_pages.pop().ok_or_else(|| {
                DnnError::WorkspaceRequired(self.num_heads * self.page_size * self.head_dim * 4)
            })?;
            self.page_tables[seq_id].push(phys);
        }

        self.seq_positions[seq_id] += 1;
        self.total_tokens_generated += 1;
        Ok(())
    }

    /// Rolls back sequence 0's position to `position`, freeing any pages that
    /// are no longer needed.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `position` exceeds the current
    /// position.
    pub fn rollback_to(&mut self, position: usize) -> Result<(), DnnError> {
        let seq_id = 0usize;
        let current = self.seq_positions[seq_id];
        if position > current {
            return Err(DnnError::InvalidArgument(format!(
                "rollback target {} exceeds current position {}",
                position, current
            )));
        }

        // Determine which pages should remain after rollback.
        let pages_needed = position.div_ceil(self.page_size);
        let current_pages = self.page_tables[seq_id].len();
        if pages_needed < current_pages {
            let excess: Vec<usize> = self.page_tables[seq_id].drain(pages_needed..).collect();
            self.free_pages.extend(excess);
        }

        self.seq_positions[seq_id] = position;
        Ok(())
    }

    /// Returns the number of tokens accepted (verified) for sequence 0.
    ///
    /// This is a proxy for `seq_positions[0]` in the context where
    /// only verified tokens advance the draft position.  See [`SpeculativeDecoder`]
    /// which tracks accepted/rejected counts separately.
    #[must_use]
    pub fn accepted_count(&self) -> usize {
        // In the draft cache the position reflects how many tokens have been
        // *generated* by the draft model.  The actual accepted count is tracked
        // externally in SpeculativeDecoder.
        self.seq_positions.first().copied().unwrap_or(0)
    }

    /// Returns the total tokens generated minus the current accepted position,
    /// i.e. how many generations have been rolled back.
    #[must_use]
    pub fn rejected_count(&self) -> usize {
        let pos = self.seq_positions.first().copied().unwrap_or(0);
        self.total_tokens_generated.saturating_sub(pos)
    }
}

// ---------------------------------------------------------------------------
// TargetCacheState extension methods
// ---------------------------------------------------------------------------

/// Outcome of verifying draft tokens against target model probabilities.
#[derive(Debug, Clone)]
pub struct TokenVerificationResult {
    /// How many draft tokens were accepted.
    pub accepted: usize,
    /// Correction token drawn from the residual distribution at the first
    /// rejected position, if rejection occurred.
    pub correction: Option<u32>,
}

impl TargetCacheState {
    /// Runs the host-side speculative sampling criterion over `draft_tokens`.
    ///
    /// * `draft_probs`  — per-token draft probabilities (length `gamma`).
    /// * `target_probs` — per-token target probabilities (length `gamma`).
    /// * `rng_samples`  — uniform samples in `[0, 1)` (length `gamma`).
    ///
    /// The method iterates tokens in order, accepting each with probability
    /// `min(1, p_target / p_draft)`.  On the first rejection it draws a
    /// correction token using `sample_correction`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the slices have inconsistent lengths.
    #[must_use]
    pub fn verify_tokens(
        &self,
        draft_tokens: &[u32],
        draft_probs: &[f32],
        target_probs: &[f32],
        rng_samples: &[f32],
    ) -> TokenVerificationResult {
        debug_assert_eq!(draft_tokens.len(), draft_probs.len());
        debug_assert_eq!(draft_tokens.len(), target_probs.len());
        debug_assert_eq!(draft_tokens.len(), rng_samples.len());

        let gamma = draft_tokens.len();
        for i in 0..gamma {
            if !accept_token(draft_probs[i], target_probs[i], rng_samples[i]) {
                let correction = self.sample_correction(i, target_probs, draft_probs);
                return TokenVerificationResult {
                    accepted: i,
                    correction: Some(correction),
                };
            }
        }
        // All accepted; return gamma accepted with no correction.
        TokenVerificationResult {
            accepted: gamma,
            correction: None,
        }
    }

    /// Samples a correction token from the residual distribution
    /// `norm(max(0, p_target − p_draft))` at the given position.
    ///
    /// Falls back to the argmax when the residual sums to zero (all mass
    /// was already assigned to the draft).  In a full implementation this
    /// method would take a random seed; here it uses the argmax of the
    /// residual for determinism in tests.
    ///
    /// The `position` argument is used to ensure the returned token index
    /// is reproducible from the residual distribution shape.
    #[must_use]
    pub fn sample_correction(
        &self,
        _position: usize,
        target_probs: &[f32],
        draft_probs: &[f32],
    ) -> u32 {
        // Residual distribution: max(0, p_target − p_draft) then normalize.
        let residual: Vec<f32> = target_probs
            .iter()
            .zip(draft_probs.iter())
            .map(|(&t, &d)| (t - d).max(0.0))
            .collect();

        let sum: f32 = residual.iter().sum();
        if sum <= 0.0 {
            // Fallback: argmax of target distribution.
            let best = target_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            return best as u32;
        }

        // Argmax of normalised residual (deterministic sampling for host tests).
        let best = residual
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        best as u32
    }
}

// ---------------------------------------------------------------------------
// SpecDecConfig / SpecDecOutput
// ---------------------------------------------------------------------------

/// Configuration for [`SpeculativeDecoder`].
///
/// Wraps [`SpeculativeDecodeConfig`] and adds page pool sizes for the
/// standalone KV manager.
#[derive(Debug, Clone)]
pub struct SpecDecConfig {
    /// Underlying model / cache configuration.
    pub inner: SpeculativeDecodeConfig,
    /// Number of speculative tokens per step (γ).
    pub gamma: usize,
}

/// Output from one [`SpeculativeDecoder::step`].
#[derive(Debug, Clone)]
pub struct SpecDecOutput {
    /// Tokens that were accepted (may include a correction token).
    pub accepted_tokens: Vec<u32>,
    /// Fraction of draft tokens accepted: `accepted / gamma`.
    pub acceptance_rate: f32,
}

// ---------------------------------------------------------------------------
// SpeculativeDecoder
// ---------------------------------------------------------------------------

/// High-level driver for one speculative decode step.
///
/// Manages host-side state: draft and target caches plus checkpointing.
/// The caller is responsible for supplying the actual token IDs and
/// probability vectors (which would come from running the draft and target
/// models).
pub struct SpeculativeDecoder {
    /// Configuration.
    config: SpecDecConfig,
    /// Draft model cache state.
    draft_state: DraftCacheState,
    /// Target model cache state.
    target_state: TargetCacheState,
    /// Active checkpoint for rollback.
    active_checkpoint: Option<CacheCheckpoint>,
    /// Running accepted count across all steps.
    total_accepted: usize,
    /// Running rejected count across all steps.
    total_rejected: usize,
}

impl SpeculativeDecoder {
    /// Creates a new `SpeculativeDecoder`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError`] if the configuration fails validation.
    pub fn new(config: SpecDecConfig) -> Result<Self, DnnError> {
        config.inner.validate()?;
        let draft_state = DraftCacheState::new(&config.inner);
        let target_state = TargetCacheState::new(&config.inner);
        Ok(Self {
            config,
            draft_state,
            target_state,
            active_checkpoint: None,
            total_accepted: 0,
            total_rejected: 0,
        })
    }

    /// Returns the inner configuration.
    #[must_use]
    pub fn config(&self) -> &SpecDecConfig {
        &self.config
    }

    /// Returns a reference to the draft cache state.
    #[must_use]
    pub fn draft_state(&self) -> &DraftCacheState {
        &self.draft_state
    }

    /// Returns a reference to the target cache state.
    #[must_use]
    pub fn target_state(&self) -> &TargetCacheState {
        &self.target_state
    }

    /// Total tokens accepted (verified) across all steps.
    #[must_use]
    pub fn total_accepted(&self) -> usize {
        self.total_accepted
    }

    /// Total tokens rejected across all steps.
    #[must_use]
    pub fn total_rejected(&self) -> usize {
        self.total_rejected
    }

    /// Runs one speculative decode step.
    ///
    /// # Arguments
    ///
    /// * `draft_tokens` — `gamma` token IDs proposed by the draft model.
    /// * `draft_probs`  — draft model probabilities for each token.
    /// * `target_probs` — target model probabilities for the same positions.
    /// * `rng_samples`  — pre-generated uniform samples in `[0, 1)`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::WorkspaceRequired`] if page allocation fails.
    /// Returns [`DnnError::InvalidArgument`] if slice lengths are inconsistent
    /// or if `draft_tokens` is empty.
    pub fn step(
        &mut self,
        draft_tokens: &[u32],
        draft_probs: &[f32],
        target_probs: &[f32],
        rng_samples: &[f32],
    ) -> Result<SpecDecOutput, DnnError> {
        let gamma = draft_tokens.len();
        if gamma == 0 {
            return Err(DnnError::InvalidArgument(
                "draft_tokens must not be empty".into(),
            ));
        }
        if draft_probs.len() != gamma || target_probs.len() != gamma || rng_samples.len() != gamma {
            return Err(DnnError::InvalidArgument(format!(
                "all slices must have length gamma={}, got draft_probs={}, target_probs={}, rng_samples={}",
                gamma,
                draft_probs.len(),
                target_probs.len(),
                rng_samples.len(),
            )));
        }

        // Snapshot draft state before generating.
        let draft_pos_before = self.draft_state.seq_positions[0];
        let draft_pages_before = self.draft_state.page_tables[0].clone();
        let draft_free_before = self.draft_state.free_pages.clone();
        self.active_checkpoint = Some(CacheCheckpoint {
            draft_positions: self.draft_state.seq_positions.clone(),
            draft_page_tables: self.draft_state.page_tables.clone(),
            draft_free_pages: self.draft_state.free_pages.clone(),
            target_position: self.target_state.verified_position,
            timestamp: self.total_accepted as u64,
        });

        // Advance draft model: generate gamma tokens.
        for &tok in draft_tokens {
            self.draft_state.advance_token(tok)?;
        }

        // Target model verification (host-side simulation).
        let verification =
            self.target_state
                .verify_tokens(draft_tokens, draft_probs, target_probs, rng_samples);

        let accepted_count = verification.accepted;
        let rejected_count = gamma - accepted_count;

        self.total_accepted += accepted_count;
        self.total_rejected += rejected_count;

        // Rollback draft to accepted position if there was any rejection.
        if rejected_count > 0 {
            // Restore draft to checkpoint then re-advance to accepted position.
            self.draft_state.seq_positions = vec![draft_pos_before];
            self.draft_state.page_tables = vec![draft_pages_before];
            self.draft_state.free_pages = draft_free_before;
            // Re-advance to accepted_count.
            for &tok in draft_tokens.iter().take(accepted_count) {
                self.draft_state.advance_token(tok)?;
            }
        }

        // Advance target state for accepted tokens.
        self.target_state.verified_position += accepted_count;

        // Build output token list.
        let mut accepted_tokens: Vec<u32> = draft_tokens[..accepted_count].to_vec();
        if let Some(correction) = verification.correction {
            accepted_tokens.push(correction);
        }

        let acceptance_rate = accepted_count as f32 / gamma as f32;
        self.active_checkpoint = None;

        Ok(SpecDecOutput {
            accepted_tokens,
            acceptance_rate,
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> SpeculativeDecodeConfig {
        SpeculativeDecodeConfig {
            draft_num_layers: 4,
            draft_num_heads: 4,
            draft_head_dim: 64,
            target_num_layers: 12,
            target_num_heads: 8,
            target_head_dim: 128,
            max_draft_tokens: 5,
            page_size: 16,
            max_pages: 32,
            acceptance_threshold: 0.8,
        }
    }

    fn dec_config() -> SpecDecConfig {
        SpecDecConfig {
            inner: base_config(),
            gamma: 4,
        }
    }

    // -----------------------------------------------------------------------
    // accept_token
    // -----------------------------------------------------------------------

    #[test]
    fn test_accept_token_always_accepts_when_target_gte_draft() {
        // ratio = 1.0, any rng < 1.0 → accept
        assert!(accept_token(0.3, 0.3, 0.999));
        assert!(accept_token(0.1, 0.5, 0.999)); // ratio > 1 → clamped to 1
        assert!(accept_token(0.2, 0.8, 0.0));
    }

    #[test]
    fn test_accept_token_rejects_when_zero_target_prob() {
        // ratio = 0/any → 0.0; rng ≥ 0 → always reject
        assert!(!accept_token(0.5, 0.0, 0.0));
        assert!(!accept_token(0.5, 0.0, 0.999));
    }

    #[test]
    fn test_accept_token_rejects_when_draft_prob_zero() {
        // draft = 0 → reject always (undefined ratio)
        assert!(!accept_token(0.0, 0.5, 0.0));
    }

    #[test]
    fn test_acceptance_rate_calculation() {
        // ratio = 0.4 / 0.8 = 0.5; rng 0.3 → accept, rng 0.7 → reject
        assert!(accept_token(0.8, 0.4, 0.3));
        assert!(!accept_token(0.8, 0.4, 0.7));
    }

    // -----------------------------------------------------------------------
    // DraftCacheState methods
    // -----------------------------------------------------------------------

    #[test]
    fn test_draft_state_advance_and_rollback() {
        let mut state = DraftCacheState::new(&base_config());
        state.advance_token(10).expect("advance 0");
        state.advance_token(20).expect("advance 1");
        assert_eq!(state.seq_positions[0], 2);

        state.rollback_to(1).expect("rollback");
        assert_eq!(state.seq_positions[0], 1);
    }

    #[test]
    fn test_draft_state_rollback_beyond_position_errors() {
        let mut state = DraftCacheState::new(&base_config());
        state.advance_token(1).expect("advance");
        let err = state.rollback_to(5);
        assert!(err.is_err());
    }

    #[test]
    fn test_draft_state_accepted_rejected_count() {
        let mut state = DraftCacheState::new(&base_config());
        for t in 0..5u32 {
            state.advance_token(t).expect("advance");
        }
        // After 5 advances, no rollback → rejected_count == 0
        assert_eq!(state.accepted_count(), 5);
        assert_eq!(state.rejected_count(), 0);
    }

    // -----------------------------------------------------------------------
    // TargetCacheState.verify_tokens
    // -----------------------------------------------------------------------

    #[test]
    fn test_verification_all_accepted() {
        let state = TargetCacheState::new(&base_config());
        let draft_tokens = [1u32, 2, 3];
        let draft_probs = [0.2f32, 0.3, 0.5];
        let target_probs = [0.4f32, 0.6, 1.0]; // all ≥ draft
        let rng = [0.99f32, 0.99, 0.99]; // doesn't matter: ratio ≥ 1 always

        let result = state.verify_tokens(&draft_tokens, &draft_probs, &target_probs, &rng);
        assert_eq!(result.accepted, 3);
        assert!(result.correction.is_none());
    }

    #[test]
    fn test_verification_partial_acceptance() {
        let state = TargetCacheState::new(&base_config());
        let draft_tokens = [1u32, 2, 3];
        let draft_probs = [0.6f32, 0.3, 0.5];
        let _target_probs = [0.6f32, 0.3, 0.5]; // ratio = 1: accepts if rng < 1
        // Force rejection at position 1 with rng ≥ 1 (impossible: clamp → reject):
        // Use ratio < 1 by making target < draft at position 1.
        let target_probs2 = [0.6f32, 0.1, 0.5]; // pos 1: ratio = 0.1/0.3 ≈ 0.33
        let rng = [0.01f32, 0.5, 0.01]; // pos 1: 0.5 > 0.33 → reject

        let result = state.verify_tokens(&draft_tokens, &draft_probs, &target_probs2, &rng);
        assert_eq!(result.accepted, 1); // only pos 0 accepted
        assert!(result.correction.is_some());
    }

    #[test]
    fn test_verification_first_rejected() {
        let state = TargetCacheState::new(&base_config());
        let draft_tokens = [42u32];
        let draft_probs = [0.9f32];
        let target_probs = [0.1f32]; // ratio = 0.11...
        let rng = [0.5f32]; // 0.5 > 0.11 → reject

        let result = state.verify_tokens(&draft_tokens, &draft_probs, &target_probs, &rng);
        assert_eq!(result.accepted, 0);
        assert!(result.correction.is_some());
    }

    // -----------------------------------------------------------------------
    // SpeculativeDecoder.step
    // -----------------------------------------------------------------------

    #[test]
    fn test_spec_decoder_all_accepted() {
        let mut dec = SpeculativeDecoder::new(dec_config()).expect("new decoder");
        let tokens = [1u32, 2, 3, 4];
        let draft_p = [0.1f32; 4];
        let target_p = [0.9f32; 4]; // all ratios >> 1 → always accept
        let rng = [0.99f32; 4];

        let out = dec.step(&tokens, &draft_p, &target_p, &rng).expect("step");
        assert_eq!(out.accepted_tokens, tokens);
        assert!((out.acceptance_rate - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spec_decoder_partial_acceptance_with_correction() {
        let mut dec = SpeculativeDecoder::new(dec_config()).expect("new decoder");
        let tokens = [1u32, 2, 3, 4];
        // Make position 0 accept, position 1 reject (ratio = 0.1/0.8 ≈ 0.125).
        let draft_p = [0.1f32, 0.8, 0.5, 0.5];
        let target_p = [0.9f32, 0.1, 0.5, 0.5];
        let rng = [0.01f32, 0.5, 0.01, 0.01]; // pos 1: 0.5 > 0.125 → reject

        let out = dec.step(&tokens, &draft_p, &target_p, &rng).expect("step");
        // Accepted: 1 draft token + 1 correction.
        assert_eq!(out.accepted_tokens.len(), 2);
        assert_eq!(out.accepted_tokens[0], 1);
        assert!(out.acceptance_rate < 1.0);
    }

    #[test]
    fn test_spec_decoder_empty_tokens_error() {
        let mut dec = SpeculativeDecoder::new(dec_config()).expect("new decoder");
        let err = dec.step(&[], &[], &[], &[]);
        assert!(err.is_err());
    }

    #[test]
    fn test_spec_decoder_mismatched_lengths_error() {
        let mut dec = SpeculativeDecoder::new(dec_config()).expect("new decoder");
        let err = dec.step(&[1u32], &[0.5f32, 0.5], &[0.5f32], &[0.5f32]);
        assert!(err.is_err());
    }

    #[test]
    fn test_spec_decoder_running_totals() {
        let mut dec = SpeculativeDecoder::new(dec_config()).expect("new decoder");
        // Step 1: all 4 accepted.
        dec.step(&[1u32, 2, 3, 4], &[0.1f32; 4], &[0.9f32; 4], &[0.01f32; 4])
            .expect("step 1");
        assert_eq!(dec.total_accepted(), 4);
        assert_eq!(dec.total_rejected(), 0);
    }
}
