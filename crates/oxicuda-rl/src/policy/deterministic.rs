//! # Deterministic Policy (DDPG / TD3)
//!
//! Models a deterministic policy `μ_θ(s)` that maps states to actions directly.
//! Used in DDPG and TD3 where the critic provides the gradient signal.
//!
//! ## Target policy smoothing (TD3)
//!
//! TD3 adds clipped Gaussian noise to the target policy during Q-value
//! computation to prevent overfitting to sharp peaks in the Q-function:
//!
//! ```text
//! ã = clip(μ'(s') + clip(ε, -c, c), a_lo, a_hi)
//! where ε ~ N(0, σ²)
//! ```
//!
//! ## Exploration noise
//!
//! During training an **Ornstein-Uhlenbeck** process is commonly added to
//! action outputs for temporally correlated exploration:
//!
//! ```text
//! dx_t = θ(μ - x_t)dt + σ dW_t
//! ```
//! approximated as: `x_{t+1} = x_t + θ(μ - x_t) + σ N(0, 1)`.

use crate::error::{RlError, RlResult};
use crate::handle::RlHandle;

// ─── DeterministicPolicy ─────────────────────────────────────────────────────

/// Deterministic policy wrapper for DDPG/TD3.
///
/// Holds only the action bounds and exploration noise configuration; the
/// network parameters (`μ_θ(s)`) are managed externally.
#[derive(Debug, Clone)]
pub struct DeterministicPolicy {
    action_dim: usize,
    /// Action lower bound.
    action_low: f32,
    /// Action upper bound.
    action_high: f32,
}

impl DeterministicPolicy {
    /// Create a policy with symmetric action bounds `[-1, 1]`.
    #[must_use]
    pub fn new(action_dim: usize) -> Self {
        Self::with_bounds(action_dim, -1.0, 1.0)
    }

    /// Create a policy with custom action bounds.
    #[must_use]
    pub fn with_bounds(action_dim: usize, action_low: f32, action_high: f32) -> Self {
        assert!(action_dim > 0, "action_dim must be > 0");
        assert!(action_low < action_high, "action_low must be < action_high");
        Self {
            action_dim,
            action_low,
            action_high,
        }
    }

    /// Number of action dimensions.
    #[must_use]
    #[inline]
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Clip `action` to `[action_low, action_high]`.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `action.len() != action_dim`.
    pub fn clip_action(&self, action: &[f32]) -> RlResult<Vec<f32>> {
        if action.len() != self.action_dim {
            return Err(RlError::DimensionMismatch {
                expected: self.action_dim,
                got: action.len(),
            });
        }
        Ok(action
            .iter()
            .map(|&a| a.clamp(self.action_low, self.action_high))
            .collect())
    }

    /// Add Gaussian exploration noise and clip.
    ///
    /// Returns `clip(action + N(0, σ²), low, high)`.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `action.len() != action_dim`.
    pub fn exploration_action(
        &self,
        action: &[f32],
        sigma: f32,
        handle: &mut RlHandle,
    ) -> RlResult<Vec<f32>> {
        if action.len() != self.action_dim {
            return Err(RlError::DimensionMismatch {
                expected: self.action_dim,
                got: action.len(),
            });
        }
        let rng = handle.rng_mut();
        let noisy: Vec<f32> = action
            .iter()
            .map(|&a| {
                let u1 = (rng.next_f32() + 1e-10).min(1.0 - 1e-10);
                let u2 = rng.next_f32();
                let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                (a + sigma * noise).clamp(self.action_low, self.action_high)
            })
            .collect();
        Ok(noisy)
    }

    /// TD3 target policy smoothing: add clipped noise to target actions.
    ///
    /// ```text
    /// ã = clip(action + clip(N(0, σ²), -c, c), low, high)
    /// ```
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `action.len() != action_dim`.
    pub fn smooth_target_action(
        &self,
        action: &[f32],
        sigma: f32,
        clip_c: f32,
        handle: &mut RlHandle,
    ) -> RlResult<Vec<f32>> {
        if action.len() != self.action_dim {
            return Err(RlError::DimensionMismatch {
                expected: self.action_dim,
                got: action.len(),
            });
        }
        let rng = handle.rng_mut();
        let smoothed: Vec<f32> = action
            .iter()
            .map(|&a| {
                let u1 = (rng.next_f32() + 1e-10).min(1.0 - 1e-10);
                let u2 = rng.next_f32();
                let noise_raw = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                let noise = (sigma * noise_raw).clamp(-clip_c, clip_c);
                (a + noise).clamp(self.action_low, self.action_high)
            })
            .collect();
        Ok(smoothed)
    }
}

// ─── Ornstein-Uhlenbeck noise ────────────────────────────────────────────────

/// Ornstein-Uhlenbeck noise process for temporally correlated exploration.
///
/// Implements the discrete-time approximation:
/// ```text
/// x_{t+1} = x_t + θ(μ - x_t) + σ * N(0, 1)
/// ```
#[derive(Debug, Clone)]
pub struct OrnsteinUhlenbeck {
    action_dim: usize,
    /// Mean-reversion level (default 0).
    mu: Vec<f32>,
    /// Mean-reversion speed θ.
    theta: f32,
    /// Noise scale σ.
    sigma: f32,
    /// Current state.
    state: Vec<f32>,
}

impl OrnsteinUhlenbeck {
    /// Create an OU process with zero mean.
    ///
    /// * `theta` — mean-reversion speed (typical: 0.15).
    /// * `sigma` — noise scale (typical: 0.2).
    #[must_use]
    pub fn new(action_dim: usize, theta: f32, sigma: f32) -> Self {
        Self {
            action_dim,
            mu: vec![0.0; action_dim],
            theta,
            sigma,
            state: vec![0.0; action_dim],
        }
    }

    /// Reset the OU process to its mean.
    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Advance the OU process by one step and return the noise vector.
    pub fn sample(&mut self, handle: &mut RlHandle) -> Vec<f32> {
        let rng = handle.rng_mut();
        let mut out = Vec::with_capacity(self.action_dim);
        let mut k = 0;
        // Box-Muller pairs
        while k < self.action_dim {
            let u1 = (rng.next_f32() + 1e-10).min(1.0 - 1e-10);
            let u2 = rng.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            out.push(r * theta.cos());
            if k + 1 < self.action_dim {
                out.push(r * theta.sin());
            }
            k += 2;
        }
        out.truncate(self.action_dim);

        for (x, (&mu, &w)) in self.state.iter_mut().zip(self.mu.iter().zip(out.iter())) {
            *x += self.theta * (mu - *x) + self.sigma * w;
        }
        self.state.clone()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DeterministicPolicy ──────────────────────────────────────────────────

    #[test]
    fn clip_action_within_bounds() {
        let p = DeterministicPolicy::new(3);
        let clipped = p.clip_action(&[-2.0, 0.0, 2.0]).unwrap();
        assert_eq!(clipped, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn exploration_action_stays_in_bounds() {
        let p = DeterministicPolicy::new(4);
        let mut handle = RlHandle::default_handle();
        for _ in 0..100 {
            let a = p.exploration_action(&[0.0; 4], 0.3, &mut handle).unwrap();
            for v in a {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "exploration action out of bounds: {v}"
                );
            }
        }
    }

    #[test]
    fn smooth_target_action_within_bounds() {
        let p = DeterministicPolicy::new(2);
        let mut handle = RlHandle::default_handle();
        for _ in 0..100 {
            let a = p
                .smooth_target_action(&[0.5, -0.5], 0.2, 0.5, &mut handle)
                .unwrap();
            for v in a {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "smoothed action out of bounds: {v}"
                );
            }
        }
    }

    #[test]
    fn clip_action_dimension_error() {
        let p = DeterministicPolicy::new(3);
        assert!(p.clip_action(&[0.0; 2]).is_err());
    }

    // ── OrnsteinUhlenbeck ────────────────────────────────────────────────────

    #[test]
    fn ou_sample_correct_dim() {
        let mut ou = OrnsteinUhlenbeck::new(4, 0.15, 0.2);
        let mut handle = RlHandle::default_handle();
        let noise = ou.sample(&mut handle);
        assert_eq!(noise.len(), 4);
    }

    #[test]
    fn ou_reset_zeroes_state() {
        let mut ou = OrnsteinUhlenbeck::new(2, 0.15, 0.2);
        let mut handle = RlHandle::default_handle();
        ou.sample(&mut handle);
        ou.reset();
        assert_eq!(ou.state, vec![0.0, 0.0]);
    }

    #[test]
    fn ou_mean_reversion() {
        // With θ=1.0 (fast reversion) and large initial noise, state should approach 0
        let mut ou = OrnsteinUhlenbeck::new(1, 1.0, 0.01);
        ou.state = vec![100.0];
        let mut handle = RlHandle::default_handle();
        for _ in 0..50 {
            ou.sample(&mut handle);
        }
        assert!(
            ou.state[0].abs() < 5.0,
            "OU should mean-revert, state={}",
            ou.state[0]
        );
    }
}
