//! # TD(λ) Multi-step Returns
//!
//! TD(λ) computes a weighted mixture of n-step returns via backward
//! accumulation (the "online" / "eligibility trace" form), equivalent to:
//!
//! ```text
//! G_t^λ = (1-λ) Σ_{n=1}^{T-t} λ^{n-1} G_t^(n) + λ^{T-t} G_t^(T-t)
//! ```
//!
//! The **backward** equivalent used here:
//! ```text
//! G_T = v_T
//! G_t = r_t + γ * [ (1-λ) * v_{t+1} + λ * G_{t+1} ] * (1 - done_t)
//!      + r_t * done_t
//! ```
//!
//! Special cases:
//! * `λ = 0` → one-step TD: `G_t = r_t + γ * v_{t+1} * (1-done)`
//! * `λ = 1` → Monte Carlo: `G_t = r_t + γ * G_{t+1} * (1-done)`

use crate::error::{RlError, RlResult};

/// Configuration for TD(λ) return computation.
#[derive(Debug, Clone, Copy)]
pub struct TdConfig {
    /// Discount factor γ.
    pub gamma: f32,
    /// TD(λ) mixing coefficient.
    pub lambda: f32,
}

impl Default for TdConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
        }
    }
}

/// Compute TD(λ) returns for a flat trajectory.
///
/// # Arguments
///
/// * `rewards`     — `[T]` rewards.
/// * `values`      — `[T+1]` value estimates including `v_{T}` (bootstrap).
/// * `dones`       — `[T]` done flags.
///
/// Returns `[T]` TD(λ) returns `G_t^λ`.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] if `values.len() != T+1`.
pub fn compute_td_lambda(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    cfg: TdConfig,
) -> RlResult<Vec<f32>> {
    let t = rewards.len();
    if values.len() != t + 1 {
        return Err(RlError::DimensionMismatch {
            expected: t + 1,
            got: values.len(),
        });
    }
    if dones.len() != t {
        return Err(RlError::DimensionMismatch {
            expected: t,
            got: dones.len(),
        });
    }
    let mut returns = vec![0.0_f32; t];
    // Bootstrap from the last value
    let mut g = values[t];
    for i in (0..t).rev() {
        let mask = 1.0 - dones[i];
        // G_t = r_t + γ * mask * [(1-λ)*v_{t+1} + λ*G_{t+1}]
        let mixed = (1.0 - cfg.lambda) * values[i + 1] + cfg.lambda * g;
        g = rewards[i] + cfg.gamma * mask * mixed;
        returns[i] = g;
    }
    Ok(returns)
}

/// Compute TD(λ) advantages `A_t = G_t^λ - v_t`.
///
/// # Errors
///
/// Same as [`compute_td_lambda`].
pub fn compute_td_advantages(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    cfg: TdConfig,
) -> RlResult<Vec<f32>> {
    let returns = compute_td_lambda(rewards, values, dones, cfg)?;
    let advantages: Vec<f32> = returns
        .iter()
        .zip(&values[..rewards.len()])
        .map(|(&g, &v)| g - v)
        .collect();
    Ok(advantages)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn td_lambda_zero_equals_one_step() {
        let cfg = TdConfig {
            gamma: 0.99,
            lambda: 0.0,
        };
        let r = vec![1.0_f32; 3];
        let v = vec![0.0_f32; 4]; // T+1 values
        let d = vec![0.0_f32; 3];
        let g = compute_td_lambda(&r, &v, &d, cfg).unwrap();
        // G_t = r_t + γ * v_{t+1} = 1 + 0 = 1 for all t
        for &gi in &g {
            assert!((gi - 1.0).abs() < 1e-5, "λ=0 return={gi}");
        }
    }

    #[test]
    fn td_lambda_one_is_monte_carlo() {
        let cfg = TdConfig {
            gamma: 1.0,
            lambda: 1.0,
        };
        let r = vec![1.0_f32; 3];
        let v = vec![0.0_f32; 4];
        let d = vec![0.0_f32; 3];
        let g = compute_td_lambda(&r, &v, &d, cfg).unwrap();
        // G_0 = 1 + 1 + 1 = 3, G_1 = 1 + 1 = 2, G_2 = 1
        assert!((g[0] - 3.0).abs() < 1e-5, "G_0={}", g[0]);
        assert!((g[1] - 2.0).abs() < 1e-5, "G_1={}", g[1]);
        assert!((g[2] - 1.0).abs() < 1e-5, "G_2={}", g[2]);
    }

    #[test]
    fn td_lambda_done_truncates() {
        let cfg = TdConfig {
            gamma: 0.99,
            lambda: 0.95,
        };
        let r = vec![1.0, 1.0, 1.0];
        let v = vec![0.5, 0.5, 0.5, 0.5];
        let d = vec![0.0, 1.0, 0.0]; // done at t=1
        let g = compute_td_lambda(&r, &v, &d, cfg).unwrap();
        // At t=1: done=1 → G_1 = r_1 (no future bootstrap)
        // G_2 should be ≈ 1 + 0.99 * 0.5 = 1.495
        // G_1 = 1 (done)
        // G_0 depends on G_1=1 with λ-mixing: G_0 ≈ 1 + 0.99 * mixed
        assert!(
            (g[1] - r[1]).abs() < 0.1,
            "G_1 at done should ≈ r={}, got {}",
            r[1],
            g[1]
        );
    }

    #[test]
    fn td_advantages_length() {
        let r = vec![1.0_f32; 5];
        let v = vec![0.5_f32; 6];
        let d = vec![0.0_f32; 5];
        let a = compute_td_advantages(&r, &v, &d, TdConfig::default()).unwrap();
        assert_eq!(a.len(), 5);
    }

    #[test]
    fn td_dimension_mismatch_values() {
        let r = vec![1.0_f32; 3];
        let v = vec![0.5_f32; 3]; // should be 4
        let d = vec![0.0_f32; 3];
        assert!(compute_td_lambda(&r, &v, &d, TdConfig::default()).is_err());
    }

    #[test]
    fn td_returns_decrease_with_later_steps() {
        // In a constant-reward undiscounted MC trajectory, earlier steps get larger returns
        let cfg = TdConfig {
            gamma: 1.0,
            lambda: 1.0,
        };
        let r = vec![1.0_f32; 5];
        let v = vec![0.0_f32; 6];
        let d = vec![0.0_f32; 5];
        let g = compute_td_lambda(&r, &v, &d, cfg).unwrap();
        assert!(
            g[0] > g[1] && g[1] > g[2],
            "returns should decrease for later steps"
        );
    }
}
