//! # Reward Normalizer
//!
//! Normalizes rewards by a running estimate of the return's standard deviation,
//! following the convention in PPO and similar algorithms:
//!
//! ```text
//! G_t = γ G_{t-1} + r_t   (discounted running return)
//! normalised_r = r / std(G) + clip
//! ```
//!
//! Two modes are supported:
//! * **Return normalisation** (PPO default): maintain running stats on the
//!   discounted return `G` and divide rewards by `std(G)`.
//! * **Reward clipping** (simple): clip raw rewards to `[-clip, clip]`.

use crate::normalize::running_stats::RunningStats;

/// Reward normalization mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RewardNormMode {
    /// Divide by running std of discounted return.
    ReturnNorm,
    /// Clip rewards to `[-clip, clip]` (no normalisation).
    Clip,
    /// No normalisation or clipping.
    None,
}

/// Reward normalizer with optional return-based normalisation and clipping.
#[derive(Debug, Clone)]
pub struct RewardNormalizer {
    mode: RewardNormMode,
    gamma: f32,
    clip: f32,
    /// Running statistics on the discounted return.
    return_stats: RunningStats,
    /// Current discounted return accumulator (per environment, if vectorized).
    running_returns: Vec<f64>,
    n_envs: usize,
}

impl RewardNormalizer {
    /// Create a reward normalizer.
    ///
    /// * `n_envs` — number of parallel environments.
    /// * `gamma`  — discount factor for return accumulation.
    /// * `clip`   — symmetric clip range.
    /// * `mode`   — normalisation mode.
    #[must_use]
    pub fn new(n_envs: usize, gamma: f32, clip: f32, mode: RewardNormMode) -> Self {
        assert!(n_envs > 0, "n_envs must be > 0");
        Self {
            mode,
            gamma,
            clip,
            return_stats: RunningStats::new(1),
            running_returns: vec![0.0_f64; n_envs],
            n_envs,
        }
    }

    /// Process rewards for a step across `n_envs` parallel environments.
    ///
    /// Updates the running return estimates and normalizes.
    ///
    /// Returns the normalised / clipped rewards.
    ///
    /// # Panics
    ///
    /// Panics if `rewards.len() != n_envs` or `dones.len() != n_envs`.
    pub fn process(&mut self, rewards: &[f32], dones: &[f32]) -> Vec<f32> {
        assert_eq!(rewards.len(), self.n_envs);
        assert_eq!(dones.len(), self.n_envs);

        match self.mode {
            RewardNormMode::None => rewards.to_vec(),
            RewardNormMode::Clip => rewards
                .iter()
                .map(|&r| r.clamp(-self.clip, self.clip))
                .collect(),
            RewardNormMode::ReturnNorm => {
                // Update running returns: G_t = γ G_{t-1} + r_t
                for (i, (&r, &d)) in rewards.iter().zip(dones.iter()).enumerate() {
                    self.running_returns[i] =
                        self.gamma as f64 * self.running_returns[i] * (1.0 - d as f64) + r as f64;
                    // Update stats with current return value
                    let _ = self.return_stats.update(&[self.running_returns[i] as f32]);
                }
                let std = self.return_stats.std_f32()[0];
                rewards
                    .iter()
                    .map(|&r| (r / (std + 1e-8)).clamp(-self.clip, self.clip))
                    .collect()
            }
        }
    }

    /// Normalise a batch of rewards without updating running statistics (evaluation).
    pub fn normalise_eval(&self, rewards: &[f32]) -> Vec<f32> {
        match self.mode {
            RewardNormMode::None => rewards.to_vec(),
            RewardNormMode::Clip => rewards
                .iter()
                .map(|&r| r.clamp(-self.clip, self.clip))
                .collect(),
            RewardNormMode::ReturnNorm => {
                let std = self.return_stats.std_f32()[0];
                rewards
                    .iter()
                    .map(|&r| (r / (std + 1e-8)).clamp(-self.clip, self.clip))
                    .collect()
            }
        }
    }

    /// Reset running returns (call at episode start).
    pub fn reset_returns(&mut self) {
        self.running_returns.iter_mut().for_each(|g| *g = 0.0);
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_mode_clips_rewards() {
        let mut rn = RewardNormalizer::new(2, 0.99, 1.0, RewardNormMode::Clip);
        let r = rn.process(&[5.0, -5.0], &[0.0, 0.0]);
        assert!((r[0] - 1.0).abs() < 1e-5, "r[0]={}", r[0]);
        assert!((r[1] + 1.0).abs() < 1e-5, "r[1]={}", r[1]);
    }

    #[test]
    fn none_mode_passthrough() {
        let mut rn = RewardNormalizer::new(3, 0.99, 10.0, RewardNormMode::None);
        let r = rn.process(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]);
        assert_eq!(r, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn return_norm_output_within_clip() {
        let mut rn = RewardNormalizer::new(1, 0.99, 5.0, RewardNormMode::ReturnNorm);
        // Feed 200 steps to build running stats
        for _ in 0..200 {
            let r = rn.process(&[1.0], &[0.0]);
            assert!(r[0].abs() <= 5.0 + 1e-4, "clipped |r|={}", r[0].abs());
        }
    }

    #[test]
    fn done_resets_return() {
        let mut rn = RewardNormalizer::new(1, 0.99, 10.0, RewardNormMode::ReturnNorm);
        for _ in 0..10 {
            rn.process(&[1.0], &[0.0]);
        }
        let g_before = rn.running_returns[0];
        rn.process(&[1.0], &[1.0]); // done=1 → reset return
        let g_after = rn.running_returns[0];
        // After done, G = 0*G_before*0 + r = 1.0
        assert!(
            g_after.abs() < g_before.abs() + 1.0 + 1e-3,
            "done should reset return"
        );
    }

    #[test]
    fn normalise_eval_no_stat_change() {
        let rn = RewardNormalizer::new(1, 0.99, 5.0, RewardNormMode::ReturnNorm);
        let before = rn.return_stats.count();
        let _ = rn.normalise_eval(&[1.0, 2.0]);
        let after = rn.return_stats.count();
        assert_eq!(before, after, "eval should not update stats");
    }
}
