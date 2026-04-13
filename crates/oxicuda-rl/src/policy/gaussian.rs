//! # Diagonal Gaussian Policy (Continuous Actions)
//!
//! Models the policy as a product of independent Gaussians:
//!
//! ```text
//! π(a|s) = N(μ(s), σ(s)²)  — elementwise independent
//! ```
//!
//! Action sampling uses the **reparameterisation trick**:
//! ```text
//! a = μ + σ ⊙ ε,   ε ~ N(0, I)
//! ```
//! allowing gradients to flow through the sample.
//!
//! The **log-probability** is:
//! ```text
//! log π(a|s) = -0.5 Σ [ ((a_i - μ_i)/σ_i)² + 2 log σ_i + log(2π) ]
//! ```
//!
//! For policies with a **Tanh squashing** (SAC-style), an additional
//! log-determinant-Jacobian correction is applied:
//! ```text
//! log π̃(ã|s) = log π(a|s) - Σ log(1 - tanh²(a_i))
//! ```

use crate::error::{RlError, RlResult};
use crate::handle::RlHandle;

/// Diagonal Gaussian policy for continuous action spaces.
#[derive(Debug, Clone)]
pub struct GaussianPolicy {
    /// Action dimension.
    action_dim: usize,
    /// Whether to apply Tanh squashing (SAC-style bounded actions).
    squash: bool,
    /// Minimum log standard deviation (clamps for numerical stability).
    log_std_min: f32,
    /// Maximum log standard deviation.
    log_std_max: f32,
}

impl GaussianPolicy {
    /// Create a Gaussian policy.
    ///
    /// * `action_dim` — number of continuous action dimensions.
    /// * `squash`     — if `true`, apply `tanh` to actions and adjust log-prob
    ///   (SAC-style bounded actions).
    #[must_use]
    pub fn new(action_dim: usize, squash: bool) -> Self {
        assert!(action_dim > 0, "action_dim must be > 0");
        Self {
            action_dim,
            squash,
            log_std_min: -20.0,
            log_std_max: 2.0,
        }
    }

    /// Set log-std bounds.
    #[must_use]
    pub fn with_log_std_bounds(mut self, min: f32, max: f32) -> Self {
        self.log_std_min = min;
        self.log_std_max = max;
        self
    }

    /// Action dimension.
    #[must_use]
    #[inline]
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Convert raw log-std output from a network to clipped standard deviations.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `log_std.len() != action_dim`.
    pub fn to_std(&self, log_std: &[f32]) -> RlResult<Vec<f32>> {
        if log_std.len() != self.action_dim {
            return Err(RlError::DimensionMismatch {
                expected: self.action_dim,
                got: log_std.len(),
            });
        }
        Ok(log_std
            .iter()
            .map(|&ls| ls.clamp(self.log_std_min, self.log_std_max).exp())
            .collect())
    }

    /// Sample an action using the reparameterisation trick.
    ///
    /// Generates `ε ~ N(0, I)` using the Box-Muller transform, then computes
    /// `a = μ + σ ⊙ ε`.  If squashing is enabled the output is passed through
    /// `tanh`.
    ///
    /// Returns `(action, epsilon)` where `epsilon` is the raw Gaussian noise
    /// (needed for log-prob computation without a full re-evaluation).
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `mu.len() != action_dim` or
    ///   `std.len() != action_dim`.
    pub fn sample(
        &self,
        mu: &[f32],
        std: &[f32],
        handle: &mut RlHandle,
    ) -> RlResult<(Vec<f32>, Vec<f32>)> {
        if mu.len() != self.action_dim {
            return Err(RlError::DimensionMismatch {
                expected: self.action_dim,
                got: mu.len(),
            });
        }
        if std.len() != self.action_dim {
            return Err(RlError::DimensionMismatch {
                expected: self.action_dim,
                got: std.len(),
            });
        }
        let rng = handle.rng_mut();
        let mut epsilon = Vec::with_capacity(self.action_dim);
        // Box-Muller transform: produce pairs of N(0,1) samples
        let mut k = 0;
        while k < self.action_dim {
            let u1 = (rng.next_f32() + 1e-10).min(1.0 - 1e-10);
            let u2 = rng.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            epsilon.push(r * theta.cos());
            if k + 1 < self.action_dim {
                epsilon.push(r * theta.sin());
            }
            k += 2;
        }
        epsilon.truncate(self.action_dim);

        let mut action: Vec<f32> = mu
            .iter()
            .zip(std.iter())
            .zip(epsilon.iter())
            .map(|((&m, &s), &e)| m + s * e)
            .collect();

        if self.squash {
            for a in action.iter_mut() {
                *a = a.tanh();
            }
        }
        Ok((action, epsilon))
    }

    /// Log-probability of `action` under `N(mu, std²)`.
    ///
    /// If squashing is enabled, applies the Tanh Jacobian correction.
    ///
    /// Returns the **total** log-prob (sum over action dimensions).
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if lengths mismatch.
    pub fn log_prob(&self, action: &[f32], mu: &[f32], std: &[f32]) -> RlResult<f32> {
        if action.len() != self.action_dim
            || mu.len() != self.action_dim
            || std.len() != self.action_dim
        {
            return Err(RlError::DimensionMismatch {
                expected: self.action_dim,
                got: action.len(),
            });
        }
        let log_2pi = (2.0 * std::f32::consts::PI).ln();
        let mut log_p = 0.0_f32;
        for ((&a, &m), &s) in action.iter().zip(mu.iter()).zip(std.iter()) {
            let s_safe = s.max(1e-8);
            let z = (a - m) / s_safe;
            log_p += -0.5 * (z * z + 2.0 * s_safe.ln() + log_2pi);
        }
        if self.squash {
            // Correction: -Σ log(1 - tanh²(a_pre))
            // Since a = tanh(a_pre), we need a_pre = atanh(a).
            for &a in action.iter() {
                let a_clip = a.clamp(-1.0 + 1e-6, 1.0 - 1e-6);
                let correction = (1.0 - a_clip * a_clip).max(1e-10).ln();
                log_p -= correction;
            }
        }
        Ok(log_p)
    }

    /// Entropy of a diagonal Gaussian `N(0, σ²)`:
    /// ```text
    /// H = 0.5 Σ (1 + ln(2πe σ_i²))
    /// ```
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `std.len() != action_dim`.
    pub fn entropy(&self, std: &[f32]) -> RlResult<f32> {
        if std.len() != self.action_dim {
            return Err(RlError::DimensionMismatch {
                expected: self.action_dim,
                got: std.len(),
            });
        }
        let log_2pie = (2.0 * std::f32::consts::PI * std::f32::consts::E).ln();
        let h: f32 = std
            .iter()
            .map(|&s| 0.5 * (log_2pie + 2.0 * s.max(1e-8).ln()))
            .sum();
        Ok(h)
    }

    /// Clamp action values to `[-1, 1]` (for squashed policies) or
    /// `[-action_scale, action_scale]` in general.
    #[must_use]
    pub fn clip_action(&self, action: &[f32], scale: f32) -> Vec<f32> {
        action.iter().map(|&a| a.clamp(-scale, scale)).collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> GaussianPolicy {
        GaussianPolicy::new(3, false)
    }

    // ── to_std ───────────────────────────────────────────────────────────────

    #[test]
    fn to_std_positive() {
        let p = policy();
        let std = p.to_std(&[0.0, 1.0, -1.0]).unwrap();
        for &s in &std {
            assert!(s > 0.0, "std must be > 0");
        }
    }

    #[test]
    fn to_std_clamps_log_std() {
        let p = GaussianPolicy::new(1, false).with_log_std_bounds(-5.0, 0.0);
        let std = p.to_std(&[-100.0]).unwrap(); // clamped to -5
        assert!((std[0] - (-5.0_f32).exp()).abs() < 1e-5);
    }

    // ── sample ───────────────────────────────────────────────────────────────

    #[test]
    fn sample_correct_dim() {
        let p = policy();
        let mut handle = RlHandle::default_handle();
        let (a, e) = p.sample(&[0.0; 3], &[1.0; 3], &mut handle).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(e.len(), 3);
    }

    #[test]
    fn sample_squashed_in_range() {
        let p = GaussianPolicy::new(4, true);
        let mut handle = RlHandle::default_handle();
        for _ in 0..50 {
            let (a, _) = p.sample(&[0.0; 4], &[1.0; 4], &mut handle).unwrap();
            for v in a {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "squashed action out of [-1,1]: {v}"
                );
            }
        }
    }

    #[test]
    fn sample_mean_approx_mu() {
        // With σ → 0, samples should ≈ μ
        let p = GaussianPolicy::new(1, false);
        let mut handle = RlHandle::default_handle();
        let mu = 3.0_f32;
        let sigma = 0.001;
        let mut sum = 0.0_f32;
        let n = 500;
        for _ in 0..n {
            let (a, _) = p.sample(&[mu], &[sigma], &mut handle).unwrap();
            sum += a[0];
        }
        let mean = sum / n as f32;
        assert!(
            (mean - mu).abs() < 0.05,
            "sample mean {mean} should be ≈ {mu}"
        );
    }

    // ── log_prob ─────────────────────────────────────────────────────────────

    #[test]
    fn log_prob_mode_is_highest() {
        let p = GaussianPolicy::new(1, false);
        let mu = [2.0];
        let std = [1.0];
        let lp_mode = p.log_prob(&mu, &mu, &std).unwrap();
        let lp_off = p.log_prob(&[5.0], &mu, &std).unwrap();
        assert!(lp_mode > lp_off, "mode should have higher log-prob");
    }

    #[test]
    fn log_prob_dimension_error() {
        let p = policy();
        assert!(p.log_prob(&[0.0; 2], &[0.0; 3], &[1.0; 3]).is_err());
    }

    // ── entropy ──────────────────────────────────────────────────────────────

    #[test]
    fn entropy_positive() {
        let p = GaussianPolicy::new(2, false);
        let h = p.entropy(&[1.0, 2.0]).unwrap();
        assert!(h > 0.0, "entropy of Gaussian should be > 0");
    }

    #[test]
    fn entropy_larger_std_more_entropy() {
        let p = GaussianPolicy::new(1, false);
        let h1 = p.entropy(&[1.0]).unwrap();
        let h2 = p.entropy(&[2.0]).unwrap();
        assert!(h2 > h1, "larger std should have more entropy");
    }
}
