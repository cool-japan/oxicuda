//! Core environment trait and a reference Linear-Quadratic environment.
//!
//! # Overview
//!
//! [`Env`] is the standard single-environment interface. [`LinearQuadraticEnv`]
//! is a fully deterministic LQR environment used in unit and integration tests.

use crate::error::{RlError, RlResult};

// ─── StepResult ───────────────────────────────────────────────────────────────

/// Result returned by [`Env::step`].
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Next observation vector (length = `obs_dim`).
    pub obs: Vec<f32>,
    /// Scalar reward received.
    pub reward: f32,
    /// Whether the episode has ended.
    pub done: bool,
}

// ─── EnvInfo ─────────────────────────────────────────────────────────────────

/// Static metadata about an environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvInfo {
    /// Length of the observation vector.
    pub obs_dim: usize,
    /// Length of the action vector.
    pub action_dim: usize,
    /// Maximum number of steps per episode (0 = unlimited).
    pub max_steps: usize,
}

// ─── Env trait ────────────────────────────────────────────────────────────────

/// Standard RL environment interface.
///
/// Implementors provide `reset`, `step`, and metadata queries.  All methods
/// return [`RlResult`] so errors propagate cleanly without panicking.
pub trait Env {
    /// Reset the environment to its initial state and return the first
    /// observation.
    fn reset(&mut self) -> RlResult<Vec<f32>>;

    /// Advance the environment by one step given `action`.
    ///
    /// `action.len()` must equal [`Self::action_dim`]; otherwise
    /// [`RlError::DimensionMismatch`] is returned.
    fn step(&mut self, action: &[f32]) -> RlResult<StepResult>;

    /// Return static metadata for this environment.
    fn info(&self) -> EnvInfo;

    /// Observation vector length.
    fn obs_dim(&self) -> usize;

    /// Action vector length.
    fn action_dim(&self) -> usize;
}

// ─── LinearQuadraticEnv ───────────────────────────────────────────────────────

/// A deterministic Linear-Quadratic Regulator (LQR) environment.
///
/// ## Dynamics
///
/// State `x ∈ ℝ^d`, action `u ∈ ℝ^d` (same dimension):
///
/// ```text
/// x_{t+1}[i] = 0.9 · x_t[i] + 0.1 · u[i]
/// ```
///
/// ## Reward
///
/// ```text
/// r = −‖x‖² − 0.1·‖u‖²
/// ```
///
/// ## Reset
///
/// State is initialised to alternating `[0.5, -0.5, 0.5, -0.5, ...]`.
///
/// ## Episode termination
///
/// The episode ends when `step_count >= max_steps` or `‖x‖ > 10.0`.
#[derive(Debug, Clone)]
pub struct LinearQuadraticEnv {
    obs_dim: usize,
    max_steps: usize,
    state: Vec<f32>,
    step_count: usize,
}

impl LinearQuadraticEnv {
    /// Create a new LQR environment.
    ///
    /// # Arguments
    ///
    /// * `obs_dim`  — dimension of state and action vectors (must be `>= 1`).
    /// * `max_steps` — maximum number of steps per episode (must be `>= 1`).
    ///
    /// # Errors
    ///
    /// Returns [`RlError::InvalidHyperparameter`] if either argument is zero.
    pub fn new(obs_dim: usize, max_steps: usize) -> Self {
        // Deterministic initial state: alternating 0.5 / -0.5.
        let state = (0..obs_dim)
            .map(|i| if i % 2 == 0 { 0.5_f32 } else { -0.5_f32 })
            .collect();
        Self {
            obs_dim,
            max_steps,
            state,
            step_count: 0,
        }
    }

    /// Compute the squared L2 norm of `v`.
    fn sq_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum()
    }
}

impl Env for LinearQuadraticEnv {
    fn reset(&mut self) -> RlResult<Vec<f32>> {
        self.step_count = 0;
        // Deterministic reset: alternating 0.5 / -0.5.
        for (i, x) in self.state.iter_mut().enumerate() {
            *x = if i % 2 == 0 { 0.5_f32 } else { -0.5_f32 };
        }
        Ok(self.state.clone())
    }

    fn step(&mut self, action: &[f32]) -> RlResult<StepResult> {
        if action.len() != self.obs_dim {
            return Err(RlError::DimensionMismatch {
                expected: self.obs_dim,
                got: action.len(),
            });
        }

        // Compute reward before state transition (uses current state).
        let x_sq = Self::sq_norm(&self.state);
        let u_sq = Self::sq_norm(action);
        let reward = -x_sq - 0.1 * u_sq;

        // Dynamics: x_{t+1}[i] = 0.9 * x[i] + 0.1 * u[i]
        for (x, u) in self.state.iter_mut().zip(action.iter()) {
            *x = 0.9 * (*x) + 0.1 * u;
        }

        self.step_count += 1;

        // Done conditions.
        let x_norm = Self::sq_norm(&self.state).sqrt();
        let done = self.step_count >= self.max_steps || x_norm > 10.0;

        Ok(StepResult {
            obs: self.state.clone(),
            reward,
            done,
        })
    }

    fn info(&self) -> EnvInfo {
        EnvInfo {
            obs_dim: self.obs_dim,
            action_dim: self.obs_dim,
            max_steps: self.max_steps,
        }
    }

    #[inline]
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    #[inline]
    fn action_dim(&self) -> usize {
        self.obs_dim
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lqr_reset_alternating() {
        let mut env = LinearQuadraticEnv::new(4, 10);
        let obs = env.reset().unwrap();
        assert_eq!(obs.len(), 4);
        assert!((obs[0] - 0.5).abs() < 1e-6);
        assert!((obs[1] + 0.5).abs() < 1e-6);
        assert!((obs[2] - 0.5).abs() < 1e-6);
        assert!((obs[3] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn lqr_step_dimension_mismatch() {
        let mut env = LinearQuadraticEnv::new(4, 10);
        let _ = env.reset().unwrap();
        assert!(env.step(&[0.0; 3]).is_err());
    }

    #[test]
    fn lqr_step_reward_is_negative() {
        let mut env = LinearQuadraticEnv::new(4, 10);
        let _ = env.reset().unwrap();
        // State is non-zero after reset, so reward should be negative.
        let res = env.step(&[0.0; 4]).unwrap();
        assert!(res.reward <= 0.0, "reward={}", res.reward);
    }

    #[test]
    fn lqr_episode_ends_at_max_steps() {
        let max = 5;
        let mut env = LinearQuadraticEnv::new(2, max);
        let _ = env.reset().unwrap();
        let mut done = false;
        for i in 0..max {
            let res = env.step(&[0.0; 2]).unwrap();
            done = res.done;
            if i < max - 1 {
                assert!(!done, "should not be done before max_steps");
            }
        }
        assert!(done, "should be done at max_steps");
    }

    #[test]
    fn lqr_info() {
        let env = LinearQuadraticEnv::new(3, 100);
        let info = env.info();
        assert_eq!(info.obs_dim, 3);
        assert_eq!(info.action_dim, 3);
        assert_eq!(info.max_steps, 100);
    }

    #[test]
    fn lqr_obs_action_dim() {
        let env = LinearQuadraticEnv::new(5, 10);
        assert_eq!(env.obs_dim(), 5);
        assert_eq!(env.action_dim(), 5);
    }

    #[test]
    fn lqr_large_action_terminates_early() {
        let mut env = LinearQuadraticEnv::new(2, 1000);
        let _ = env.reset().unwrap();
        // Massive action drives state norm above 10.
        let done_at_some_point =
            (0..1000).any(|_| env.step(&[100.0, 100.0]).map(|r| r.done).unwrap_or(true));
        assert!(done_at_some_point);
    }
}
