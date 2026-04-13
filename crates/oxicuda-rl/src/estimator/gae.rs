//! # Generalized Advantage Estimation (GAE)
//!
//! Schulman et al. (2015), "High-Dimensional Continuous Control Using
//! Generalized Advantage Estimation", ICLR 2016.
//!
//! ## Algorithm
//!
//! Given a trajectory of `T` transitions `(r_t, v_t, v_{t+1}, done_t)` and
//! hyperparameters `γ` (discount) and `λ` (bias-variance trade-off), GAE
//! computes the advantage estimate via a **reversed scan**:
//!
//! ```text
//! δ_t = r_t + γ * v_{t+1} * (1 - done_t) - v_t
//!
//! A_t = δ_t + (γλ) * (1 - done_t) * A_{t+1}
//! ```
//!
//! Special cases:
//! * `λ = 0` → standard one-step TD advantage: `A_t = δ_t`
//! * `λ = 1` → undiscounted Monte Carlo advantage
//!
//! The **return** (value target) is computed as `G_t = A_t + v_t`.

use crate::error::{RlError, RlResult};

/// Hyperparameters for GAE computation.
#[derive(Debug, Clone, Copy)]
pub struct GaeConfig {
    /// Discount factor γ ∈ (0, 1].
    pub gamma: f32,
    /// GAE λ ∈ [0, 1].
    pub lambda: f32,
    /// Whether to normalise advantages to have zero mean and unit variance.
    pub normalise: bool,
}

impl Default for GaeConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            normalise: true,
        }
    }
}

/// Computed GAE output for a single trajectory.
#[derive(Debug, Clone)]
pub struct GaeOutput {
    /// Advantage estimates `A_t` for each timestep.
    pub advantages: Vec<f32>,
    /// Value targets (returns) `G_t = A_t + v_t`.
    pub returns: Vec<f32>,
}

/// Compute GAE advantages and returns for a flat trajectory.
///
/// # Arguments
///
/// * `rewards`    — `[T]` reward at each timestep.
/// * `values`     — `[T]` value estimate `V(s_t)`.
/// * `next_values`— `[T]` value estimate `V(s_{t+1})` (last entry can be 0
///   for terminal).
/// * `dones`      — `[T]` termination flag (1.0 = done, 0.0 = ongoing).
/// * `cfg`        — GAE configuration.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] if slice lengths differ.
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    next_values: &[f32],
    dones: &[f32],
    cfg: GaeConfig,
) -> RlResult<GaeOutput> {
    let t = rewards.len();
    if values.len() != t || next_values.len() != t || dones.len() != t {
        return Err(RlError::DimensionMismatch {
            expected: t,
            got: values.len(),
        });
    }
    let gamma_lambda = cfg.gamma * cfg.lambda;
    let mut advantages = vec![0.0_f32; t];
    let mut gae = 0.0_f32;

    // Backward scan over the trajectory
    for i in (0..t).rev() {
        let mask = 1.0 - dones[i];
        let delta = rewards[i] + cfg.gamma * next_values[i] * mask - values[i];
        gae = delta + gamma_lambda * mask * gae;
        advantages[i] = gae;
    }

    if cfg.normalise && t > 1 {
        let mean = advantages.iter().sum::<f32>() / t as f32;
        let var = advantages
            .iter()
            .map(|&a| (a - mean) * (a - mean))
            .sum::<f32>()
            / t as f32;
        let std = (var + 1e-8).sqrt();
        for a in advantages.iter_mut() {
            *a = (*a - mean) / std;
        }
    }

    let returns: Vec<f32> = advantages
        .iter()
        .zip(values.iter())
        .map(|(&a, &v)| a + v)
        .collect();

    Ok(GaeOutput {
        advantages,
        returns,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ones_trajectory(t: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let rewards = vec![1.0_f32; t];
        let values = vec![0.5_f32; t];
        let next_values = vec![0.5_f32; t];
        let dones = vec![0.0_f32; t];
        (rewards, values, next_values, dones)
    }

    #[test]
    fn gae_output_length() {
        let (r, v, nv, d) = ones_trajectory(10);
        let out = compute_gae(&r, &v, &nv, &d, GaeConfig::default()).unwrap();
        assert_eq!(out.advantages.len(), 10);
        assert_eq!(out.returns.len(), 10);
    }

    #[test]
    fn gae_dimension_mismatch() {
        let r = vec![1.0; 5];
        let v = vec![0.5; 4]; // wrong length
        let nv = vec![0.5; 5];
        let d = vec![0.0; 5];
        assert!(compute_gae(&r, &v, &nv, &d, GaeConfig::default()).is_err());
    }

    #[test]
    fn gae_lambda_zero_is_td() {
        // λ=0 → GAE = one-step TD advantage = δ_t
        let cfg = GaeConfig {
            gamma: 0.99,
            lambda: 0.0,
            normalise: false,
        };
        let r = vec![1.0_f32; 3];
        let v = vec![0.0_f32; 3];
        let nv = vec![0.0_f32; 3];
        let d = vec![0.0_f32; 3];
        let out = compute_gae(&r, &v, &nv, &d, cfg).unwrap();
        // δ = 1 + 0.99*0 - 0 = 1.0 for all t; no λ accumulation
        for &a in &out.advantages {
            assert!((a - 1.0).abs() < 1e-5, "A={a}");
        }
    }

    #[test]
    fn gae_done_resets_accumulation() {
        let cfg = GaeConfig {
            gamma: 0.99,
            lambda: 0.95,
            normalise: false,
        };
        // Last step is terminal
        let r = vec![1.0, 1.0, 1.0];
        let v = vec![0.0, 0.0, 0.0];
        let nv = vec![0.0, 0.0, 0.0];
        let d = vec![0.0, 0.0, 1.0]; // done at t=2
        let out = compute_gae(&r, &v, &nv, &d, cfg).unwrap();
        // At t=2 (last): δ = 1 + 0.99*0*(1-1) - 0 = 1; gae = 1
        // At t=1: δ = 1 + 0.99*0 - 0 = 1; gae = 1 + 0.99*0.95*(1-0)*1 = 1.94..
        // At t=0: gae should be > 1
        assert!(
            out.advantages[0] > out.advantages[2],
            "earlier steps with future returns should have higher advantage"
        );
    }

    #[test]
    fn gae_normalise_zero_mean() {
        let (r, v, nv, d) = ones_trajectory(20);
        let cfg = GaeConfig {
            normalise: true,
            ..GaeConfig::default()
        };
        let out = compute_gae(&r, &v, &nv, &d, cfg).unwrap();
        let mean = out.advantages.iter().sum::<f32>() / 20.0;
        assert!(
            mean.abs() < 1e-4,
            "normalised mean should be ≈0, got {mean}"
        );
    }

    #[test]
    fn gae_normalise_unit_std() {
        let (r, v, nv, d) = ones_trajectory(50);
        let cfg = GaeConfig {
            normalise: true,
            ..GaeConfig::default()
        };
        let out = compute_gae(&r, &v, &nv, &d, cfg).unwrap();
        let mean = out.advantages.iter().sum::<f32>() / 50.0;
        let var: f32 = out
            .advantages
            .iter()
            .map(|&a| (a - mean).powi(2))
            .sum::<f32>()
            / 50.0;
        let std = var.sqrt();
        assert!(
            (std - 1.0).abs() < 0.05,
            "normalised std should be ≈1, got {std}"
        );
    }

    #[test]
    fn gae_returns_equal_advantage_plus_value() {
        let (r, v, nv, d) = ones_trajectory(5);
        let cfg = GaeConfig {
            normalise: false,
            ..GaeConfig::default()
        };
        let out = compute_gae(&r, &v, &nv, &d, cfg).unwrap();
        for (i, (&ret, (&a, &vi))) in out
            .returns
            .iter()
            .zip(out.advantages.iter().zip(v.iter()))
            .enumerate()
        {
            assert!(
                (ret - (a + vi)).abs() < 1e-5,
                "G[{i}] = A+V failed: {ret} vs {}",
                a + vi
            );
        }
    }
}
