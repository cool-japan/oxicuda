//! # DQN and Double-DQN Bellman Error
//!
//! Mnih et al. (2015) and van Hasselt et al. (2016).
//!
//! ## Standard DQN
//!
//! ```text
//! target = r + γ * max_a Q_target(s', a) * (1 - done)
//! loss   = E[ (Q(s, a) - target)² ]
//! ```
//!
//! ## Double-DQN
//!
//! Action selection uses the *online* network; Q-value estimation uses the
//! *target* network, reducing maximisation bias:
//!
//! ```text
//! a* = argmax_a Q_online(s', a)
//! target = r + γ * Q_target(s', a*) * (1 - done)
//! ```
//!
//! ## Huber Loss
//!
//! The Huber (smooth-L1) loss is used by default to clip gradient magnitudes:
//! ```text
//! L(δ) = 0.5 δ²        if |δ| ≤ κ
//!        κ(|δ| - κ/2)  otherwise
//! ```

use crate::error::{RlError, RlResult};

/// DQN hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct DqnConfig {
    /// Discount factor γ.
    pub gamma: f32,
    /// Huber loss κ (`None` = use MSE).
    pub huber_kappa: Option<f32>,
    /// Whether to use importance-sampling weights (PER).
    pub use_is_weights: bool,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            huber_kappa: Some(1.0),
            use_is_weights: false,
        }
    }
}

/// DQN loss output.
#[derive(Debug, Clone)]
pub struct DqnLoss {
    /// Mean Bellman error (loss value to minimise).
    pub loss: f32,
    /// Per-sample TD errors `|target - Q(s,a)|` (for PER priority update).
    pub td_errors: Vec<f32>,
}

/// Huber loss for a scalar error.
#[inline]
fn huber(delta: f32, kappa: f32) -> f32 {
    if delta.abs() <= kappa {
        0.5 * delta * delta
    } else {
        kappa * (delta.abs() - 0.5 * kappa)
    }
}

/// Compute the standard DQN Bellman loss.
///
/// # Arguments
///
/// * `q_sa`        — `[B]` Q(s_t, a_t) from online network.
/// * `rewards`     — `[B]` rewards.
/// * `max_q_next`  — `[B]` max_a Q_target(s_{t+1}, a) from target network.
/// * `dones`       — `[B]` done flags.
/// * `is_weights`  — `[B]` importance-sampling weights (all 1 if not using PER).
/// * `cfg`         — DQN configuration.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] for inconsistent lengths.
pub fn dqn_loss(
    q_sa: &[f32],
    rewards: &[f32],
    max_q_next: &[f32],
    dones: &[f32],
    is_weights: &[f32],
    cfg: DqnConfig,
) -> RlResult<DqnLoss> {
    let b = q_sa.len();
    if rewards.len() != b || max_q_next.len() != b || dones.len() != b || is_weights.len() != b {
        return Err(RlError::DimensionMismatch {
            expected: b,
            got: b.wrapping_sub(1),
        });
    }
    let mut loss = 0.0_f32;
    let mut td_errors = Vec::with_capacity(b);
    for i in 0..b {
        let target = rewards[i] + cfg.gamma * max_q_next[i] * (1.0 - dones[i]);
        let delta = target - q_sa[i];
        td_errors.push(delta.abs());
        let elem_loss = match cfg.huber_kappa {
            Some(k) => huber(delta, k),
            None => 0.5 * delta * delta,
        };
        loss += is_weights[i] * elem_loss;
    }
    loss /= b as f32;
    Ok(DqnLoss { loss, td_errors })
}

/// Compute the Double-DQN Bellman loss.
///
/// # Arguments
///
/// * `q_sa`         — `[B]` Q_online(s_t, a_t).
/// * `rewards`      — `[B]` rewards.
/// * `q_next_online`— `[B × A]` Q_online(s_{t+1}, ·) for all actions.
/// * `q_next_target`— `[B × A]` Q_target(s_{t+1}, ·) for all actions.
/// * `n_actions`    — number of discrete actions A.
/// * `dones`        — `[B]` done flags.
/// * `is_weights`   — `[B]` IS weights.
/// * `cfg`          — DQN configuration.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] for length mismatches.
#[allow(clippy::too_many_arguments)]
pub fn double_dqn_loss(
    q_sa: &[f32],
    rewards: &[f32],
    q_next_online: &[f32],
    q_next_target: &[f32],
    n_actions: usize,
    dones: &[f32],
    is_weights: &[f32],
    cfg: DqnConfig,
) -> RlResult<DqnLoss> {
    let b = q_sa.len();
    if rewards.len() != b
        || q_next_online.len() != b * n_actions
        || q_next_target.len() != b * n_actions
        || dones.len() != b
        || is_weights.len() != b
    {
        return Err(RlError::DimensionMismatch {
            expected: b * n_actions,
            got: q_next_online.len(),
        });
    }
    // For each sample: a* = argmax_a Q_online(s', a)
    let max_q_next: Vec<f32> = (0..b)
        .map(|i| {
            let slice = &q_next_online[i * n_actions..(i + 1) * n_actions];
            let a_star = slice
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);
            q_next_target[i * n_actions + a_star]
        })
        .collect();
    dqn_loss(q_sa, rewards, &max_q_next, dones, is_weights, cfg)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huber_small_delta() {
        let l = huber(0.5, 1.0);
        assert!((l - 0.125).abs() < 1e-5, "huber(0.5)={l}");
    }

    #[test]
    fn huber_large_delta() {
        let l = huber(2.0, 1.0);
        // κ*(|δ| - κ/2) = 1*(2 - 0.5) = 1.5
        assert!((l - 1.5).abs() < 1e-5, "huber(2.0)={l}");
    }

    #[test]
    fn dqn_loss_zero_when_perfect() {
        // If Q(s,a) = r + γ*max_Q_next, loss = 0
        let cfg = DqnConfig {
            gamma: 1.0,
            huber_kappa: None,
            use_is_weights: false,
        };
        let q = vec![2.0_f32; 4];
        let r = vec![1.0_f32; 4];
        let max_q_next = vec![1.0_f32; 4];
        let done = vec![0.0_f32; 4];
        let w = vec![1.0_f32; 4];
        let l = dqn_loss(&q, &r, &max_q_next, &done, &w, cfg).unwrap();
        assert!(l.loss.abs() < 1e-5, "loss should be 0, got {}", l.loss);
    }

    #[test]
    fn dqn_loss_positive_when_off() {
        let cfg = DqnConfig::default();
        let q = vec![0.0_f32; 4];
        let r = vec![1.0_f32; 4];
        let max_q_next = vec![1.0_f32; 4];
        let done = vec![0.0_f32; 4];
        let w = vec![1.0_f32; 4];
        let l = dqn_loss(&q, &r, &max_q_next, &done, &w, cfg).unwrap();
        assert!(l.loss > 0.0, "loss should be > 0");
    }

    #[test]
    fn dqn_loss_done_stops_future() {
        let cfg = DqnConfig {
            gamma: 1.0,
            huber_kappa: None,
            use_is_weights: false,
        };
        // With done=1, target = r, so loss = (q - r)^2 / 2
        let q = vec![0.0_f32];
        let r = vec![2.0_f32];
        let max_q_next = vec![100.0_f32]; // should be ignored
        let done = vec![1.0_f32];
        let w = vec![1.0_f32];
        let l = dqn_loss(&q, &r, &max_q_next, &done, &w, cfg).unwrap();
        // target=2, delta=2, MSE=2.0
        assert!((l.loss - 2.0).abs() < 1e-5, "loss={}", l.loss);
    }

    #[test]
    fn dqn_td_errors_correct_length() {
        let cfg = DqnConfig::default();
        let n = 8;
        let q = vec![0.0_f32; n];
        let r = vec![1.0_f32; n];
        let max_q = vec![0.5_f32; n];
        let done = vec![0.0_f32; n];
        let w = vec![1.0_f32; n];
        let l = dqn_loss(&q, &r, &max_q, &done, &w, cfg).unwrap();
        assert_eq!(l.td_errors.len(), n);
    }

    #[test]
    fn double_dqn_selects_online_argmax() {
        let cfg = DqnConfig {
            gamma: 1.0,
            huber_kappa: None,
            use_is_weights: false,
        };
        // 1 sample, 3 actions
        // online Q_next: [1, 5, 2] → a* = 1
        // target Q_next: [10, 3, 10] → Q_target(s', a*=1) = 3
        let q_sa = vec![0.0_f32];
        let r = vec![0.0_f32];
        let q_next_on = vec![1.0, 5.0, 2.0]; // [B × A]
        let q_next_tgt = vec![10.0, 3.0, 10.0];
        let done = vec![0.0_f32];
        let w = vec![1.0_f32];
        let l = double_dqn_loss(&q_sa, &r, &q_next_on, &q_next_tgt, 3, &done, &w, cfg).unwrap();
        // target = 0 + 1.0 * 3.0 = 3.0; loss = 0.5 * (0 - 3)^2 = 4.5
        assert!((l.loss - 4.5).abs() < 1e-4, "loss={}", l.loss);
    }

    #[test]
    fn dqn_dimension_mismatch() {
        let cfg = DqnConfig::default();
        let q = vec![0.0_f32; 4];
        let r = vec![1.0_f32; 3];
        let max_q = vec![0.5_f32; 4];
        let done = vec![0.0_f32; 4];
        let w = vec![1.0_f32; 4];
        assert!(dqn_loss(&q, &r, &max_q, &done, &w, cfg).is_err());
    }
}
