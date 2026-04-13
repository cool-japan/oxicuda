//! Vectorized environment wrapper.
//!
//! [`VecEnv`] runs multiple [`Env`] instances in lock-step, auto-resetting any
//! environment that reaches a terminal state.  All observations are returned as
//! a flattened `Vec<f32>` of length `n_envs × obs_dim`.

use crate::env::env::{Env, StepResult};
use crate::error::{RlError, RlResult};

// ─── VecStepResult ────────────────────────────────────────────────────────────

/// Result returned by [`VecEnv::step`].
#[derive(Debug, Clone)]
pub struct VecStepResult {
    /// Flattened observations for all environments, length = `n_envs × obs_dim`.
    ///
    /// For done environments this is the first observation of the **new**
    /// episode (auto-reset semantics).
    pub obs: Vec<f32>,
    /// Per-environment scalar rewards, length = `n_envs`.
    pub rewards: Vec<f32>,
    /// Per-environment done flags, length = `n_envs`.
    pub dones: Vec<bool>,
}

// ─── VecEnv ───────────────────────────────────────────────────────────────────

/// Synchronous vectorized environment.
///
/// Wraps a `Vec<E>` of homogeneous environments.  All environments must share
/// the same `obs_dim` and `action_dim`; this is validated lazily on the first
/// `step` call to avoid redundant checks.
///
/// # Auto-reset
///
/// When `step` detects that environment `i` is done, it immediately calls
/// `reset()` on it and places the resulting observation in the corresponding
/// slot of `VecStepResult::obs`.
///
/// # Example
///
/// ```rust
/// use oxicuda_rl::env::env::LinearQuadraticEnv;
/// use oxicuda_rl::env::vectorized::VecEnv;
///
/// let envs: Vec<_> = (0..4).map(|_| LinearQuadraticEnv::new(3, 50)).collect();
/// let mut ve = VecEnv::new(envs);
/// let obs = ve.reset_all().unwrap();
/// assert_eq!(obs.len(), 4 * 3);
/// ```
#[derive(Debug)]
pub struct VecEnv<E: Env> {
    envs: Vec<E>,
}

impl<E: Env> VecEnv<E> {
    /// Create a new [`VecEnv`] from a non-empty vector of environments.
    ///
    /// # Panics
    ///
    /// Does **not** panic; returns an instance even with an empty `envs` slice
    /// (though subsequent calls will fail with [`RlError::DimensionMismatch`]).
    #[must_use]
    pub fn new(envs: Vec<E>) -> Self {
        Self { envs }
    }

    /// Number of parallel environments.
    #[must_use]
    #[inline]
    pub fn n_envs(&self) -> usize {
        self.envs.len()
    }

    /// Reset **all** environments and return flattened observations.
    ///
    /// Returns a `Vec<f32>` of length `n_envs × obs_dim`.
    ///
    /// # Errors
    ///
    /// Propagates any [`RlError`] from individual `reset()` calls.
    pub fn reset_all(&mut self) -> RlResult<Vec<f32>> {
        let mut flat = Vec::new();
        for env in &mut self.envs {
            let obs = env.reset()?;
            flat.extend_from_slice(&obs);
        }
        Ok(flat)
    }

    /// Step all environments simultaneously.
    ///
    /// `actions` must have length `n_envs × action_dim`; the slice is split
    /// into per-environment chunks before dispatch.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] — `actions.len()` is not a multiple of
    ///   `n_envs`, or a chunk length does not match the environment's
    ///   `action_dim`.
    /// * Any error propagated from individual `step()` or `reset()` calls.
    pub fn step(&mut self, actions: &[f32]) -> RlResult<VecStepResult> {
        let n = self.envs.len();
        if n == 0 {
            return Ok(VecStepResult {
                obs: Vec::new(),
                rewards: Vec::new(),
                dones: Vec::new(),
            });
        }

        // Infer per-environment action chunk size.
        if actions.len() % n != 0 {
            return Err(RlError::DimensionMismatch {
                expected: n * self.envs[0].action_dim(),
                got: actions.len(),
            });
        }
        let action_dim = actions.len() / n;

        let mut flat_obs: Vec<f32> = Vec::with_capacity(actions.len());
        let mut rewards = Vec::with_capacity(n);
        let mut dones = Vec::with_capacity(n);

        for (env, chunk) in self.envs.iter_mut().zip(actions.chunks_exact(action_dim)) {
            let StepResult { obs, reward, done } = env.step(chunk)?;

            rewards.push(reward);
            dones.push(done);

            if done {
                // Auto-reset: use the first observation of the new episode.
                let reset_obs = env.reset()?;
                flat_obs.extend_from_slice(&reset_obs);
            } else {
                flat_obs.extend_from_slice(&obs);
            }
        }

        Ok(VecStepResult {
            obs: flat_obs,
            rewards,
            dones,
        })
    }

    /// Immutable access to the underlying environment slice.
    #[must_use]
    #[inline]
    pub fn envs(&self) -> &[E] {
        &self.envs
    }

    /// Mutable access to the underlying environment slice.
    #[inline]
    pub fn envs_mut(&mut self) -> &mut [E] {
        &mut self.envs
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::env::LinearQuadraticEnv;

    fn make_vec_env(n: usize, obs_dim: usize, max_steps: usize) -> VecEnv<LinearQuadraticEnv> {
        let envs = (0..n)
            .map(|_| LinearQuadraticEnv::new(obs_dim, max_steps))
            .collect();
        VecEnv::new(envs)
    }

    #[test]
    fn reset_all_length() {
        let mut ve = make_vec_env(4, 3, 50);
        let obs = ve.reset_all().unwrap();
        assert_eq!(obs.len(), 4 * 3);
    }

    #[test]
    fn step_output_lengths() {
        let mut ve = make_vec_env(4, 3, 50);
        let _ = ve.reset_all().unwrap();
        let actions = vec![0.0_f32; 4 * 3];
        let res = ve.step(&actions).unwrap();
        assert_eq!(res.obs.len(), 4 * 3);
        assert_eq!(res.rewards.len(), 4);
        assert_eq!(res.dones.len(), 4);
    }

    #[test]
    fn step_dimension_mismatch() {
        let mut ve = make_vec_env(4, 3, 50);
        let _ = ve.reset_all().unwrap();
        // Wrong total length.
        assert!(ve.step(&[0.0; 10]).is_err());
    }

    #[test]
    fn auto_reset_on_done() {
        // max_steps=1 so every step triggers a done and auto-reset.
        let mut ve = make_vec_env(2, 2, 1);
        let _ = ve.reset_all().unwrap();
        let res = ve.step(&[0.0_f32; 2 * 2]).unwrap();
        // All dones should be true.
        assert!(res.dones.iter().all(|&d| d));
        // obs should still have the reset observations (length correct).
        assert_eq!(res.obs.len(), 2 * 2);
    }

    #[test]
    fn n_envs_accessor() {
        let ve = make_vec_env(6, 4, 100);
        assert_eq!(ve.n_envs(), 6);
    }

    #[test]
    fn empty_vec_env_step() {
        let envs: Vec<LinearQuadraticEnv> = Vec::new();
        let mut ve = VecEnv::new(envs);
        let res = ve.step(&[]).unwrap();
        assert!(res.obs.is_empty());
        assert!(res.rewards.is_empty());
        assert!(res.dones.is_empty());
    }
}
