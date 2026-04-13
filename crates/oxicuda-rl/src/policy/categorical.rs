//! # Categorical Policy (Discrete Actions)
//!
//! A policy over a discrete action space with `n_actions` choices.
//!
//! ## Operations
//!
//! * **Softmax**: convert raw logits to a probability distribution.
//! * **Sample**: draw an action proportional to probabilities.
//! * **Log-probability**: `log P(a|s)` for REINFORCE / PPO ratio.
//! * **Entropy**: `H = -Σ p_a log p_a` for entropy regularisation.
//! * **Greedy**: `argmax p(a|s)` for evaluation / exploitation.

use crate::error::{RlError, RlResult};
use crate::handle::RlHandle;

// ─── CategoricalPolicy ───────────────────────────────────────────────────────

/// Discrete action policy backed by a categorical distribution.
#[derive(Debug, Clone)]
pub struct CategoricalPolicy {
    /// Number of discrete actions.
    n_actions: usize,
    /// Temperature for softmax (lower → more deterministic).
    temperature: f32,
}

impl CategoricalPolicy {
    /// Create a categorical policy with `n_actions` actions.
    ///
    /// Default temperature: 1.0.
    #[must_use]
    pub fn new(n_actions: usize) -> Self {
        assert!(n_actions > 0, "n_actions must be > 0");
        Self {
            n_actions,
            temperature: 1.0,
        }
    }

    /// Set the softmax temperature.
    #[must_use]
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Number of actions.
    #[must_use]
    #[inline]
    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    /// Compute softmax probabilities from logits.
    ///
    /// Numerically stable: subtracts `max(logits)` before exponentiation.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `logits.len() != n_actions`.
    pub fn softmax(&self, logits: &[f32]) -> RlResult<Vec<f32>> {
        if logits.len() != self.n_actions {
            return Err(RlError::DimensionMismatch {
                expected: self.n_actions,
                got: logits.len(),
            });
        }
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits
            .iter()
            .map(|&l| ((l - max_logit) / self.temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        Ok(probs)
    }

    /// Sample an action index from the given probability distribution.
    ///
    /// Uses the Gumbel-max trick for efficient sampling: add Gumbel(0,1) noise
    /// to log-probs and return the argmax.
    ///
    /// # Errors
    ///
    /// * [`RlError::EmptyDistribution`] if `probs` is empty.
    /// * [`RlError::InvalidDistribution`] if `probs` does not sum to ≈ 1.
    pub fn sample_action(&self, probs: &[f32], handle: &mut RlHandle) -> RlResult<usize> {
        if probs.is_empty() {
            return Err(RlError::EmptyDistribution);
        }
        let sum: f32 = probs.iter().sum();
        if (sum - 1.0).abs() > 0.01 {
            return Err(RlError::InvalidDistribution { sum, tol: 0.01 });
        }
        let rng = handle.rng_mut();
        // Gumbel-max: argmax(log(p) + G)  where G ~ Gumbel(0,1) = -log(-log(U)), U~Uniform
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            let u = (rng.next_f32() + 1e-10).min(1.0 - 1e-10);
            let gumbel = -((-u.ln()).ln());
            let val = p.ln() + gumbel;
            if val > best_val {
                best_val = val;
                best_idx = i;
            }
        }
        Ok(best_idx)
    }

    /// Sample an action from raw logits (convenience: softmax + sample).
    pub fn sample_from_logits(&self, logits: &[f32], handle: &mut RlHandle) -> RlResult<usize> {
        let probs = self.softmax(logits)?;
        self.sample_action(&probs, handle)
    }

    /// Log-probability of action `a` under distribution `probs`.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `probs.len() != n_actions`.
    pub fn log_prob(&self, probs: &[f32], action: usize) -> RlResult<f32> {
        if probs.len() != self.n_actions {
            return Err(RlError::DimensionMismatch {
                expected: self.n_actions,
                got: probs.len(),
            });
        }
        let p = probs[action].max(1e-10);
        Ok(p.ln())
    }

    /// Log-probabilities for a batch of (probs, action) pairs.
    ///
    /// * `probs_batch` — flat `[batch_size × n_actions]` slice.
    /// * `actions` — `[batch_size]` integer slice.
    ///
    /// Returns a `Vec<f32>` of length `batch_size`.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if sizes are inconsistent.
    pub fn log_prob_batch(&self, probs_batch: &[f32], actions: &[usize]) -> RlResult<Vec<f32>> {
        let batch_size = actions.len();
        if probs_batch.len() != batch_size * self.n_actions {
            return Err(RlError::DimensionMismatch {
                expected: batch_size * self.n_actions,
                got: probs_batch.len(),
            });
        }
        let mut out = Vec::with_capacity(batch_size);
        for (b, &a) in actions.iter().enumerate() {
            let p = probs_batch[b * self.n_actions + a].max(1e-10);
            out.push(p.ln());
        }
        Ok(out)
    }

    /// Shannon entropy `H = -Σ p_a * log(p_a)`.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `probs.len() != n_actions`.
    pub fn entropy(&self, probs: &[f32]) -> RlResult<f32> {
        if probs.len() != self.n_actions {
            return Err(RlError::DimensionMismatch {
                expected: self.n_actions,
                got: probs.len(),
            });
        }
        let h = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        Ok(h)
    }

    /// Greedy action: `argmax p(a|s)`.
    ///
    /// # Errors
    ///
    /// * [`RlError::EmptyDistribution`] if `probs` is empty.
    pub fn greedy_action(&self, probs: &[f32]) -> RlResult<usize> {
        if probs.is_empty() {
            return Err(RlError::EmptyDistribution);
        }
        let (idx, _) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("probs is non-empty");
        Ok(idx)
    }

    /// KL divergence `KL(p || q)`.
    ///
    /// Both `p` and `q` must be valid probability distributions.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if lengths differ.
    pub fn kl_divergence(&self, p: &[f32], q: &[f32]) -> RlResult<f32> {
        if p.len() != q.len() {
            return Err(RlError::DimensionMismatch {
                expected: p.len(),
                got: q.len(),
            });
        }
        let kl = p
            .iter()
            .zip(q.iter())
            .filter(|&(&pi, _)| pi > 0.0)
            .map(|(&pi, &qi)| pi * (pi.ln() - qi.max(1e-10).ln()))
            .sum();
        Ok(kl)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> CategoricalPolicy {
        CategoricalPolicy::new(4)
    }

    // ── softmax ──────────────────────────────────────────────────────────────

    #[test]
    fn softmax_sums_to_one() {
        let p = policy();
        let probs = p.softmax(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let s: f32 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "softmax sum={s}");
    }

    #[test]
    fn softmax_max_logit_has_max_prob() {
        let p = policy();
        let probs = p.softmax(&[1.0, 5.0, 2.0, 3.0]).unwrap();
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 1, "max logit at idx=1 should have max prob");
    }

    #[test]
    fn softmax_dimension_mismatch() {
        let p = policy();
        assert!(p.softmax(&[1.0, 2.0]).is_err());
    }

    // ── sample ───────────────────────────────────────────────────────────────

    #[test]
    fn sample_in_range() {
        let p = policy();
        let probs = p.softmax(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut handle = RlHandle::default_handle();
        for _ in 0..100 {
            let a = p.sample_action(&probs, &mut handle).unwrap();
            assert!(a < 4, "action out of range: {a}");
        }
    }

    #[test]
    fn sample_deterministic_peaked() {
        // Peaked distribution → almost always sample action 2
        let p = CategoricalPolicy::new(3).with_temperature(0.01);
        let probs = p.softmax(&[0.0, 0.0, 10.0]).unwrap();
        let mut handle = RlHandle::default_handle();
        let mut counts = [0_usize; 3];
        for _ in 0..50 {
            let a = p.sample_action(&probs, &mut handle).unwrap();
            counts[a] += 1;
        }
        assert!(
            counts[2] > 45,
            "peaked dist should mostly pick action 2, counts={counts:?}"
        );
    }

    #[test]
    fn sample_invalid_distribution() {
        let p = policy();
        let mut handle = RlHandle::default_handle();
        // probs don't sum to 1
        assert!(p.sample_action(&[0.1, 0.1, 0.1, 0.1], &mut handle).is_err());
    }

    // ── log_prob ─────────────────────────────────────────────────────────────

    #[test]
    fn log_prob_uniform_distribution() {
        let p = CategoricalPolicy::new(4);
        let probs = vec![0.25; 4];
        let lp = p.log_prob(&probs, 0).unwrap();
        assert!((lp - (-2.0_f32.ln() * 2.0)).abs() < 1e-5, "log_prob={lp}");
    }

    #[test]
    fn log_prob_batch_correct() {
        let p = CategoricalPolicy::new(2);
        let probs = vec![0.7, 0.3, 0.4, 0.6]; // 2 examples, 2 classes
        let actions = vec![0_usize, 1];
        let lps = p.log_prob_batch(&probs, &actions).unwrap();
        assert!((lps[0] - 0.7_f32.ln()).abs() < 1e-5);
        assert!((lps[1] - 0.6_f32.ln()).abs() < 1e-5);
    }

    // ── entropy ──────────────────────────────────────────────────────────────

    #[test]
    fn entropy_uniform_is_max() {
        let p = CategoricalPolicy::new(4);
        let uniform = vec![0.25; 4];
        let peaked = vec![0.97, 0.01, 0.01, 0.01];
        let h_u = p.entropy(&uniform).unwrap();
        let h_p = p.entropy(&peaked).unwrap();
        assert!(h_u > h_p, "uniform entropy {h_u} should be > peaked {h_p}");
    }

    #[test]
    fn entropy_deterministic_is_zero() {
        let p = CategoricalPolicy::new(2);
        let probs = vec![1.0, 0.0];
        let h = p.entropy(&probs).unwrap();
        assert!(
            h.abs() < 1e-6,
            "deterministic entropy should be ≈0, got {h}"
        );
    }

    // ── greedy ───────────────────────────────────────────────────────────────

    #[test]
    fn greedy_selects_max() {
        let p = policy();
        let probs = vec![0.1, 0.5, 0.3, 0.1];
        let a = p.greedy_action(&probs).unwrap();
        assert_eq!(a, 1, "greedy should pick idx=1");
    }

    // ── kl_divergence ────────────────────────────────────────────────────────

    #[test]
    fn kl_div_identical_distributions_zero() {
        let p = policy();
        let probs = vec![0.25; 4];
        let kl = p.kl_divergence(&probs, &probs).unwrap();
        assert!(kl.abs() < 1e-5, "KL(p||p) should be 0, got {kl}");
    }

    #[test]
    fn kl_div_non_negative() {
        let p = policy();
        let p_dist = vec![0.4, 0.3, 0.2, 0.1];
        let q_dist = vec![0.1, 0.2, 0.3, 0.4];
        let kl = p.kl_divergence(&p_dist, &q_dist).unwrap();
        assert!(kl > 0.0, "KL divergence should be > 0");
    }
}
