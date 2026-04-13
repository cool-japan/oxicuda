//! # Beam Search
//!
//! Maintains `beam_width` candidate hypotheses simultaneously, expanding
//! each by the full vocabulary and retaining only the top-K continuations.
//!
//! ## Algorithm (one decode step)
//!
//! Given the current set of `beam_width` beams and `vocab_size` logits for
//! each, the step:
//!
//! 1. Compute log-probabilities from logits (log-softmax).
//! 2. For each live beam `b`, form all `vocab_size` expansions:
//!    `score(b, t) = beam.score + log_prob(t)`.
//! 3. Collect all `beam_width × vocab_size` candidates, sort by score.
//! 4. Keep the top `beam_width` candidates as new live beams.
//! 5. Sequences ending with `eos_token_id` are moved to `completed`.
//! 6. Apply length-penalty normalisation when comparing hypotheses for
//!    selection: `norm_score = score / (len^alpha)`.
//!
//! ## References
//!
//! * Wu et al., "Google's Neural Machine Translation System" (2016)
//!   introduced length normalisation: `α ∈ [0, 1]`.

use crate::error::{InferError, InferResult};

// ─── BeamSearchConfig ────────────────────────────────────────────────────────

/// Beam search hyperparameters.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of parallel beams (search width).
    pub beam_width: usize,
    /// Length-penalty exponent `α` (0.0 = no penalty, 1.0 = divide by length).
    pub length_penalty: f32,
    /// Maximum new tokens per beam.
    pub max_new_tokens: usize,
    /// Token ID that signals the end of a hypothesis.
    pub eos_token_id: u32,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            length_penalty: 0.6,
            max_new_tokens: 256,
            eos_token_id: 2,
        }
    }
}

// ─── BeamHypothesis ──────────────────────────────────────────────────────────

/// One candidate sequence in the beam.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token sequence generated so far (not including prompt).
    pub tokens: Vec<u32>,
    /// Accumulated unnormalised log-probability.
    pub score: f64,
    /// Completed (ended with EOS).
    pub completed: bool,
}

impl BeamHypothesis {
    /// Length-normalised score for ranking.
    #[must_use]
    pub fn normalised_score(&self, alpha: f32) -> f64 {
        let len = self.tokens.len().max(1) as f64;
        self.score / len.powf(alpha as f64)
    }
}

// ─── BeamSearchState ─────────────────────────────────────────────────────────

/// Mutable state of an ongoing beam search.
pub struct BeamSearchState {
    pub config: BeamSearchConfig,
    /// Live (not yet EOS) beams.
    pub beams: Vec<BeamHypothesis>,
    /// Completed beams (ended with EOS or max length reached).
    pub completed: Vec<BeamHypothesis>,
}

impl BeamSearchState {
    /// Initialise a new beam search from a single empty hypothesis.
    #[must_use]
    pub fn new(config: BeamSearchConfig) -> Self {
        let initial_beam = BeamHypothesis {
            tokens: Vec::new(),
            score: 0.0,
            completed: false,
        };
        Self {
            beams: vec![initial_beam; config.beam_width],
            completed: Vec::new(),
            config,
        }
    }

    /// Is the search finished?  True when all beams have completed or
    /// the maximum generation length is reached.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.beams.is_empty()
    }

    /// Return the hypothesis with the highest length-normalised score,
    /// considering both completed and still-live beams.
    #[must_use]
    pub fn best_hypothesis(&self) -> Option<&BeamHypothesis> {
        let alpha = self.config.length_penalty;
        self.completed
            .iter()
            .chain(self.beams.iter())
            .max_by(|a, b| {
                a.normalised_score(alpha)
                    .partial_cmp(&b.normalised_score(alpha))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    // ── Step ─────────────────────────────────────────────────────────────────

    /// Expand one decode step.
    ///
    /// `all_logits` has shape `[n_live_beams][vocab_size]` — one row per
    /// currently live beam.  The method expands all beams, prunes to
    /// `beam_width`, and moves finished beams to `completed`.
    ///
    /// Returns `true` if the search is done after this step.
    ///
    /// # Errors
    ///
    /// * [`InferError::BeamSearchError`] — `all_logits` length ≠ live beam count.
    /// * [`InferError::EmptyBatch`]      — `all_logits` is empty.
    pub fn step(&mut self, all_logits: &[Vec<f32>]) -> InferResult<bool> {
        if all_logits.is_empty() {
            return Err(InferError::EmptyBatch);
        }
        if all_logits.len() != self.beams.len() {
            return Err(InferError::BeamSearchError(format!(
                "expected {} logit rows (one per live beam), got {}",
                self.beams.len(),
                all_logits.len(),
            )));
        }

        let vocab_size = all_logits[0].len();
        if vocab_size == 0 {
            return Err(InferError::BeamSearchError("empty logits row".to_owned()));
        }

        // --- Build candidate list -------------------------------------------
        // Each candidate = (beam_idx, token_id, new_score)
        let mut candidates: Vec<(usize, u32, f64)> =
            Vec::with_capacity(self.beams.len() * vocab_size);

        for (beam_idx, (beam, logits)) in self.beams.iter().zip(all_logits.iter()).enumerate() {
            let log_probs = log_softmax(logits);
            for (tok_id, &lp) in log_probs.iter().enumerate() {
                let new_score = beam.score + lp as f64;
                candidates.push((beam_idx, tok_id as u32, new_score));
            }
        }

        // --- Sort candidates by unnormalised score (descending) ---
        candidates
            .sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // --- Select top `beam_width` next beams ----------------------------
        let mut next_beams: Vec<BeamHypothesis> = Vec::with_capacity(self.config.beam_width);
        for (beam_idx, tok_id, new_score) in candidates.iter().copied() {
            if next_beams.len() >= self.config.beam_width {
                break;
            }
            let mut tokens = self.beams[beam_idx].tokens.clone();
            tokens.push(tok_id);
            let completed =
                tok_id == self.config.eos_token_id || tokens.len() >= self.config.max_new_tokens;
            next_beams.push(BeamHypothesis {
                tokens,
                score: new_score,
                completed,
            });
        }

        // --- Move completed beams to the completed list --------------------
        let mut still_live = Vec::with_capacity(self.config.beam_width);
        for hyp in next_beams {
            if hyp.completed {
                self.completed.push(hyp);
            } else {
                still_live.push(hyp);
            }
        }
        self.beams = still_live;

        Ok(self.is_done())
    }
}

// ─── log_softmax ─────────────────────────────────────────────────────────────

/// Numerically stable log-softmax.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
    logits.iter().map(|&x| (x - max) - log_sum_exp).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_logits(n: usize) -> Vec<f32> {
        vec![1.0_f32; n]
    }

    fn peaked_logits(n: usize, peak: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        v[peak] = 10.0;
        v
    }

    fn make_config() -> BeamSearchConfig {
        BeamSearchConfig {
            beam_width: 3,
            length_penalty: 0.6,
            max_new_tokens: 16,
            eos_token_id: 0,
        }
    }

    #[test]
    fn initialises_with_beam_width_beams() {
        let state = BeamSearchState::new(make_config());
        assert_eq!(state.beams.len(), 3);
        assert!(!state.is_done());
    }

    #[test]
    fn step_expands_beams() {
        let mut state = BeamSearchState::new(make_config());
        let logits: Vec<Vec<f32>> = (0..3).map(|_| uniform_logits(8)).collect();
        state.step(&logits).unwrap();
        // Live beams still ≤ beam_width.
        assert!(state.beams.len() + state.completed.len() <= 3);
    }

    #[test]
    fn eos_token_completes_beam() {
        let mut state = BeamSearchState::new(make_config());
        // Peaked logits that always choose token 0 = EOS.
        let logits: Vec<Vec<f32>> = (0..3).map(|_| peaked_logits(8, 0)).collect();
        let done = state.step(&logits).unwrap();
        assert!(done, "all beams should finish on EOS");
        assert!(!state.completed.is_empty());
    }

    #[test]
    fn length_normalised_score() {
        let hyp = BeamHypothesis {
            tokens: vec![1, 2, 3],
            score: -3.0,
            completed: false,
        };
        let score_alpha0 = hyp.normalised_score(0.0);
        let score_alpha1 = hyp.normalised_score(1.0);
        // alpha=0: score/1 = -3; alpha=1: score/3 = -1
        assert!((score_alpha0 - (-3.0)).abs() < 1e-6);
        assert!((score_alpha1 - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn best_hypothesis_prefers_completed() {
        let mut state = BeamSearchState::new(make_config());
        state.completed.push(BeamHypothesis {
            tokens: vec![1],
            score: -1.0,
            completed: true,
        });
        state.beams.push(BeamHypothesis {
            tokens: vec![],
            score: 100.0, // very high but not completed
            completed: false,
        });
        // best_hypothesis returns best normalised score across both pools
        let best = state.best_hypothesis().unwrap();
        // The live beam with score 100 wins on normalised score.
        // (completed: -1/1^0.6 = -1, live: 100/1^0.6 = 100)
        assert!((best.score - 100.0).abs() < 1e-6);
    }

    #[test]
    fn max_new_tokens_terminates() {
        let config = BeamSearchConfig {
            beam_width: 2,
            length_penalty: 0.0,
            max_new_tokens: 2,
            eos_token_id: 999,
        };
        let mut state = BeamSearchState::new(config);
        for _ in 0..3 {
            let logits: Vec<Vec<f32>> = (0..state.beams.len())
                .map(|_| peaked_logits(10, 5))
                .collect();
            if state.is_done() {
                break;
            }
            state.step(&logits).unwrap();
        }
        assert!(
            state.is_done(),
            "search should terminate after max_new_tokens"
        );
    }

    #[test]
    fn empty_logits_error() {
        let mut state = BeamSearchState::new(make_config());
        assert!(matches!(state.step(&[]), Err(InferError::EmptyBatch)));
    }

    #[test]
    fn wrong_beam_count_error() {
        let mut state = BeamSearchState::new(make_config());
        let logits = vec![uniform_logits(8)]; // only 1 row, need 3
        assert!(matches!(
            state.step(&logits),
            Err(InferError::BeamSearchError(_))
        ));
    }

    #[test]
    fn log_softmax_sums_correctly() {
        let logits = vec![0.0_f32, 1.0, 2.0];
        let lsm = log_softmax(&logits);
        let sum_exp: f64 = lsm.iter().map(|&x| (x as f64).exp()).sum();
        assert!(
            (sum_exp - 1.0).abs() < 1e-5,
            "sum of exp(log_softmax) = {sum_exp}"
        );
    }
}
