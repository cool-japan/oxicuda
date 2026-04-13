//! # Speculative Decoding
//!
//! Implements the **rejection sampling** acceptance algorithm from:
//!
//! > Chen et al., "Accelerating Large Language Model Decoding with Speculative
//! > Sampling" (arXiv 2302.01318, 2023).
//!
//! ## Concept
//!
//! A small **draft model** autoregressively generates `k` candidate tokens
//! `d_1, ..., d_k` with probabilities `q(d_i | ctx, d_1,...,d_{i-1})`.  A
//! large **target model** evaluates all `k+1` positions in a single parallel
//! forward pass, producing `p(· | ctx, d_1,...,d_{i-1})` for each position.
//!
//! The acceptance criterion:
//!
//! ```text
//! u_i ~ Uniform(0, 1)
//! accept d_i  iff  u_i ≤ min(1, p(d_i) / q(d_i))
//! ```
//!
//! If `d_i` is rejected, a correction token is sampled from:
//! ```text
//! p'(t) = max(0, p(t) - q(t))  (normalised)
//! ```
//!
//! If all `k` drafts are accepted, one more token is sampled from
//! `p(· | ctx, d_1,...,d_k)` (the last target position).
//!
//! ## Correctness guarantee
//!
//! The output distribution is **identical** to sampling directly from the
//! target model (in expectation), while producing β·k+1 tokens per step on
//! average where β is the acceptance rate.

use crate::error::{InferError, InferResult};
use crate::sampling::{Rng, categorical_sample};

// ─── speculative_verify ──────────────────────────────────────────────────────

/// Run the rejection-sampling acceptance test for one speculative-decode step.
///
/// # Parameters
///
/// * `draft_tokens`  — `k` tokens proposed by the draft model.
/// * `draft_probs`   — `[k][vocab_size]` probability arrays from the draft.
/// * `target_probs`  — `[k+1][vocab_size]` probability arrays from the target.
///   Positions 0..k correspond to the same contexts as `draft_probs`;
///   position k is used if all drafts are accepted.
/// * `rng`           — random-number generator.
///
/// # Returns
///
/// `(accepted_tokens, bonus_token)` where:
///
/// * `accepted_tokens` is the longest accepted prefix of `draft_tokens`.
/// * `bonus_token` is the correction / continuation token sampled after
///   the last accepted position.
///
/// The caller should append `accepted_tokens ++ [bonus_token]` to the
/// sequence.
///
/// # Errors
///
/// * [`InferError::SpeculativeDecodingMismatch`] — if `target_probs.len()` ≠ `k+1`.
/// * [`InferError::EmptyBatch`]                  — if `draft_tokens` is empty.
/// * [`InferError::DimensionMismatch`]           — vocab size inconsistency.
/// * [`InferError::SamplingError`]               — correction distribution is all-zero.
pub fn speculative_verify(
    draft_tokens: &[u32],
    draft_probs: &[Vec<f32>],
    target_probs: &[Vec<f32>],
    rng: &mut Rng,
) -> InferResult<(Vec<u32>, u32)> {
    let k = draft_tokens.len();
    if k == 0 {
        return Err(InferError::EmptyBatch);
    }
    if target_probs.len() != k + 1 {
        return Err(InferError::SpeculativeDecodingMismatch {
            expected: k + 1,
            got: target_probs.len(),
        });
    }
    if draft_probs.len() != k {
        return Err(InferError::SpeculativeDecodingMismatch {
            expected: k,
            got: draft_probs.len(),
        });
    }

    let vocab_size = target_probs[0].len();
    for p in draft_probs.iter().chain(target_probs.iter()) {
        if p.len() != vocab_size {
            return Err(InferError::DimensionMismatch {
                expected: vocab_size,
                got: p.len(),
            });
        }
    }

    let mut accepted_tokens: Vec<u32> = Vec::with_capacity(k);

    for i in 0..k {
        let d_i = draft_tokens[i] as usize;
        let p_di = target_probs[i].get(d_i).copied().unwrap_or(0.0);
        let q_di = draft_probs[i].get(d_i).copied().unwrap_or(0.0).max(1e-10);

        let acceptance_ratio = (p_di / q_di).min(1.0);
        let u = rng.next_f32();

        if u <= acceptance_ratio {
            // Accept draft token.
            accepted_tokens.push(draft_tokens[i]);
        } else {
            // Reject: sample correction token from max(0, p - q) / Z.
            let bonus = sample_correction(&target_probs[i], &draft_probs[i], rng)?;
            return Ok((accepted_tokens, bonus));
        }
    }

    // All k drafts accepted: sample one more from target_probs[k].
    let bonus = categorical_sample_from_probs(&target_probs[k], rng)?;
    Ok((accepted_tokens, bonus))
}

// ─── Speculative helpers ─────────────────────────────────────────────────────

/// Sample from the correction distribution `max(0, p - q)` (normalised).
fn sample_correction(p: &[f32], q: &[f32], rng: &mut Rng) -> InferResult<u32> {
    let mut correction: Vec<f32> = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).max(0.0))
        .collect();

    let total: f32 = correction.iter().sum();
    if total < 1e-12 {
        // p ≈ q everywhere; fall back to sampling from p.
        return categorical_sample_from_probs(p, rng);
    }
    let inv = 1.0 / total;
    correction.iter_mut().for_each(|x| *x *= inv);
    Ok(categorical_sample(&correction, rng) as u32)
}

/// Sample from a raw probability vector (must sum to approximately 1).
fn categorical_sample_from_probs(probs: &[f32], rng: &mut Rng) -> InferResult<u32> {
    if probs.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    let total: f32 = probs.iter().sum();
    if total < 1e-12 {
        return Err(InferError::SamplingError(
            "speculative: all probabilities are zero".to_owned(),
        ));
    }
    Ok(categorical_sample(probs, rng) as u32)
}

// ─── SpeculativeDecoder ───────────────────────────────────────────────────────

/// Stateful wrapper around speculative decoding for a sequence.
///
/// Accumulates draft tokens from a fast draft model and verifies them
/// in batches against the slow target model.
pub struct SpeculativeDecoder {
    /// Number of draft tokens to generate before each verification.
    pub lookahead: usize,
    /// Accepted tokens so far (from all verification steps).
    pub output_tokens: Vec<u32>,
    /// Total draft tokens proposed.
    pub n_draft_proposed: u64,
    /// Total draft tokens accepted.
    pub n_draft_accepted: u64,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder with the given lookahead window.
    ///
    /// # Panics
    /// Panics if `lookahead == 0`.
    #[must_use]
    pub fn new(lookahead: usize) -> Self {
        assert!(lookahead > 0, "lookahead must be ≥ 1");
        Self {
            lookahead,
            output_tokens: Vec::new(),
            n_draft_proposed: 0,
            n_draft_accepted: 0,
        }
    }

    /// Process one verification step.
    ///
    /// Calls [`speculative_verify`] and updates internal statistics.
    pub fn verify_step(
        &mut self,
        draft_tokens: &[u32],
        draft_probs: &[Vec<f32>],
        target_probs: &[Vec<f32>],
        rng: &mut Rng,
    ) -> InferResult<()> {
        let (accepted, bonus) = speculative_verify(draft_tokens, draft_probs, target_probs, rng)?;
        self.n_draft_proposed += draft_tokens.len() as u64;
        self.n_draft_accepted += accepted.len() as u64;
        self.output_tokens.extend_from_slice(&accepted);
        self.output_tokens.push(bonus);
        Ok(())
    }

    /// Empirical acceptance rate over all verification steps.
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        if self.n_draft_proposed == 0 {
            0.0
        } else {
            self.n_draft_accepted as f64 / self.n_draft_proposed as f64
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_probs(n: usize) -> Vec<f32> {
        vec![1.0 / n as f32; n]
    }

    fn peaked_probs(n: usize, peak: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        v[peak] = 1.0;
        v
    }

    #[test]
    fn all_accept_when_draft_eq_target() {
        // When p == q, acceptance ratio = 1 → always accept.
        let vocab = 8;
        let k = 3;
        let probs: Vec<Vec<f32>> = (0..k).map(|i| peaked_probs(vocab, i)).collect();
        // target_probs has k+1 rows; last row for the bonus token
        let mut target_probs = probs.clone();
        target_probs.push(peaked_probs(vocab, k));

        let mut rng = Rng::new(0);
        let draft_tokens: Vec<u32> = (0..k as u32).collect();
        let (accepted, _bonus) =
            speculative_verify(&draft_tokens, &probs, &target_probs, &mut rng).unwrap();
        assert_eq!(accepted.len(), k, "all {k} drafts should be accepted");
    }

    #[test]
    fn reject_when_draft_differs_from_target() {
        // Draft always proposes token 0; target strongly prefers token 1.
        // → acceptance ratio p(0)/q(0) ≈ 0 → immediate rejection.
        let vocab = 4;
        let k = 3;
        let draft_probs: Vec<Vec<f32>> = (0..k).map(|_| peaked_probs(vocab, 0)).collect();
        let target_probs: Vec<Vec<f32>> = (0..k + 1).map(|_| peaked_probs(vocab, 1)).collect();
        let draft_tokens: Vec<u32> = vec![0; k];

        let mut rng = Rng::new(123);
        let (accepted, bonus) =
            speculative_verify(&draft_tokens, &draft_probs, &target_probs, &mut rng).unwrap();
        // With p(0)=0 and q(0)=1, acceptance = 0 → reject immediately.
        assert!(accepted.is_empty(), "should reject first draft immediately");
        // Bonus token should come from correction distribution ∝ max(0, p-q)
        // p = [0,1,0,0], q = [1,0,0,0] → correction ∝ [0,1,0,0] → token 1
        assert_eq!(bonus, 1);
    }

    #[test]
    fn wrong_target_length_error() {
        let vocab = 4;
        let k = 2;
        let draft = vec![0_u32, 1];
        let dp = vec![uniform_probs(vocab); k];
        let tp = vec![uniform_probs(vocab); k]; // should be k+1
        let mut r = Rng::new(0);
        assert!(matches!(
            speculative_verify(&draft, &dp, &tp, &mut r),
            Err(InferError::SpeculativeDecodingMismatch { .. })
        ));
    }

    #[test]
    fn empty_draft_error() {
        let vocab = 4;
        let dp = vec![uniform_probs(vocab)];
        let tp = vec![uniform_probs(vocab); 1];
        let mut r = Rng::new(0);
        assert!(matches!(
            speculative_verify(&[], &dp, &tp, &mut r),
            Err(InferError::EmptyBatch)
        ));
    }

    #[test]
    fn decoder_accumulates_output_tokens() {
        let vocab = 4;
        let k = 2;
        let draft = vec![0_u32, 1];
        let dp = vec![uniform_probs(vocab); k];
        let mut tp = vec![uniform_probs(vocab); k + 1];
        tp[0][0] = 0.9; // make token 0 likely for position 0
        tp[1][1] = 0.9; // make token 1 likely for position 1

        let mut decoder = SpeculativeDecoder::new(k);
        let mut rng = Rng::new(42);
        decoder.verify_step(&draft, &dp, &tp, &mut rng).unwrap();
        assert!(!decoder.output_tokens.is_empty());
    }

    #[test]
    fn acceptance_rate_bounds() {
        let vocab = 4;
        let k = 4;
        let probs: Vec<Vec<f32>> = (0..k).map(|_| uniform_probs(vocab)).collect();
        let mut target = probs.clone();
        target.push(uniform_probs(vocab));
        let draft: Vec<u32> = (0..k as u32).map(|i| i % vocab as u32).collect();

        let mut decoder = SpeculativeDecoder::new(k);
        let mut rng = Rng::new(7);
        for _ in 0..10 {
            decoder
                .verify_step(&draft, &probs, &target, &mut rng)
                .unwrap();
        }
        let rate = decoder.acceptance_rate();
        assert!(
            (0.0..=1.0).contains(&rate),
            "acceptance rate out of range: {rate}"
        );
    }

    #[test]
    fn bonus_token_always_returned() {
        // Even when all drafts are rejected, a bonus token must be returned.
        let vocab = 4;
        let k = 3;
        let dp = vec![peaked_probs(vocab, 0); k];
        let tp = vec![peaked_probs(vocab, 1); k + 1]; // target prefers token 1
        let draft = vec![0_u32; k];
        let mut r = Rng::new(0);
        let (accepted, bonus) = speculative_verify(&draft, &dp, &tp, &mut r).unwrap();
        // accepted may be 0 (first token rejected) or more if some get through.
        let _ = accepted;
        // bonus must be valid (< vocab)
        assert!((bonus as usize) < vocab);
    }
}
