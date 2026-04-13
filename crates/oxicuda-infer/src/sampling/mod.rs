//! Token sampling strategies for autoregressive language model decoding.
//!
//! # Modules
//!
//! | Module | Algorithm |
//! |--------|-----------|
//! | [`greedy`] | Argmax — always pick the highest-probability token |
//! | [`top_k`]  | Filter to top-K tokens before sampling |
//! | [`top_p`]  | Nucleus sampling: smallest set with cumprob ≥ p |
//! | [`beam_search`] | Maintain K parallel hypotheses with length normalisation |
//! | [`speculative`] | Chen et al. 2023 speculative decoding with rejection sampling |
//!
//! # Shared RNG
//!
//! All stochastic samplers accept a [`Rng`] for reproducible testing.
//! The RNG is a 64-bit LCG with Knuth's constants.

pub mod beam_search;
pub mod greedy;
pub mod speculative;
pub mod top_k;
pub mod top_p;

pub use beam_search::{BeamHypothesis, BeamSearchConfig, BeamSearchState};
pub use greedy::{greedy_sample, greedy_sample_batch};
pub use speculative::speculative_verify;
pub use top_k::{top_k_filter, top_k_sample};
pub use top_p::{top_p_filter, top_p_sample};

// ─── Rng ─────────────────────────────────────────────────────────────────────

/// Simple 64-bit linear-congruential generator for reproducible sampling.
///
/// Uses Knuth's constants:
/// * multiplier = 6364136223846793005
/// * increment  = 1442695040888963407
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new RNG with the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        } // avoid all-zero state
    }

    /// Advance the state and return the next 64-bit integer.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Sample a float uniformly in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Use the upper 24 bits for f32 mantissa precision.
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }

    /// Sample a uniformly random index in `0..n`.
    pub fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

// ─── Shared utilities ────────────────────────────────────────────────────────

/// Compute softmax probabilities from logits (numerically stable).
///
/// Returns a new `Vec<f32>` with the same length as `logits`.
pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    probs.iter_mut().for_each(|p| *p *= inv);
    probs
}

/// Sample a categorical index from a probability distribution using the rng.
///
/// Requires `probs` to be non-negative and sum to approximately 1.
pub(crate) fn categorical_sample(probs: &[f32], rng: &mut Rng) -> usize {
    let u = rng.next_f32();
    let mut cumsum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum > u {
            return i;
        }
    }
    // Fallback: return last non-zero element.
    probs.iter().rposition(|&p| p > 0.0).unwrap_or(0)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_different_seeds_different_values() {
        let mut r1 = Rng::new(1);
        let mut r2 = Rng::new(2);
        assert_ne!(r1.next_u64(), r2.next_u64());
    }

    #[test]
    fn rng_f32_in_range() {
        let mut r = Rng::new(42);
        for _ in 0..1000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "got {v}");
        }
    }

    #[test]
    fn rng_usize_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            let v = r.next_usize(10);
            assert!(v < 10, "got {v}");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 0.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }

    #[test]
    fn softmax_largest_logit_has_highest_prob() {
        let logits = vec![0.0_f32, 5.0, 1.0];
        let probs = softmax(&logits);
        let argmax = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(argmax, 1);
    }

    #[test]
    fn categorical_sample_all_mass_on_one() {
        let mut r = Rng::new(0);
        let mut probs = vec![0.0_f32; 8];
        probs[3] = 1.0;
        for _ in 0..20 {
            assert_eq!(categorical_sample(&probs, &mut r), 3);
        }
    }
}
