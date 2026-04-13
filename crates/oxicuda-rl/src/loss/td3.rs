//! # TD3 (Twin Delayed DDPG) Loss Functions
//!
//! Fujimoto et al. (2018), "Addressing Function Approximation Error in
//! Actor-Critic Methods".
//!
//! TD3 makes three improvements over DDPG:
//! 1. **Twin critics** — use min(Q1, Q2) target to reduce overestimation.
//! 2. **Delayed policy updates** — actor updates every `d` critic updates.
//! 3. **Target policy smoothing** — add clipped Gaussian noise to target actions.
//!
//! ## Critic Loss
//!
//! ```text
//! y_t = r_t + γ (1-done_t) min(Q1', Q2')(s_{t+1}, ã_{t+1})
//! L_Q = (Q1(s_t,a_t) - y)² + (Q2(s_t,a_t) - y)²
//! ```
//!
//! ## Actor Loss
//!
//! ```text
//! L_π = -Q1(s_t, μ_θ(s_t))
//! ```
//! (Maximise Q1 by gradient ascent.)

use crate::error::{RlError, RlResult};

/// TD3 hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct Td3Config {
    /// Discount factor γ.
    pub gamma: f32,
    /// Whether to use Huber loss (else MSE) for critic.
    pub huber: bool,
}

impl Default for Td3Config {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            huber: false,
        }
    }
}

/// TD3 loss output.
#[derive(Debug, Clone)]
pub struct Td3Loss {
    /// Total critic loss (Q1 + Q2).
    pub critic_loss: f32,
    /// Q1 loss.
    pub q1_loss: f32,
    /// Q2 loss.
    pub q2_loss: f32,
    /// Per-sample TD errors (from Q1) for PER.
    pub td_errors: Vec<f32>,
}

/// Compute TD3 critic loss using twin Q-networks.
///
/// # Arguments
///
/// * `q1_sa`, `q2_sa` — `[B]` Q1 and Q2 predictions at `(s_t, a_t)`.
/// * `rewards`        — `[B]` rewards.
/// * `dones`          — `[B]` done flags.
/// * `min_q_next`     — `[B]` min(Q1', Q2') at target smoothed `(s_{t+1}, ã)`.
/// * `is_weights`     — `[B]` importance-sampling weights.
/// * `cfg`            — TD3 configuration.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] for inconsistent lengths.
pub fn td3_critic_loss(
    q1_sa: &[f32],
    q2_sa: &[f32],
    rewards: &[f32],
    dones: &[f32],
    min_q_next: &[f32],
    is_weights: &[f32],
    cfg: Td3Config,
) -> RlResult<Td3Loss> {
    let b = q1_sa.len();
    if q2_sa.len() != b
        || rewards.len() != b
        || dones.len() != b
        || min_q_next.len() != b
        || is_weights.len() != b
    {
        return Err(RlError::DimensionMismatch {
            expected: b,
            got: b.wrapping_sub(1),
        });
    }
    let mut q1_loss = 0.0_f32;
    let mut q2_loss = 0.0_f32;
    let mut td_errors = Vec::with_capacity(b);
    for i in 0..b {
        let mask = 1.0 - dones[i];
        let target = rewards[i] + cfg.gamma * mask * min_q_next[i];
        let d1 = target - q1_sa[i];
        let d2 = target - q2_sa[i];
        td_errors.push(d1.abs());
        let l1 = if cfg.huber {
            if d1.abs() <= 1.0 {
                0.5 * d1 * d1
            } else {
                d1.abs() - 0.5
            }
        } else {
            0.5 * d1 * d1
        };
        let l2 = if cfg.huber {
            if d2.abs() <= 1.0 {
                0.5 * d2 * d2
            } else {
                d2.abs() - 0.5
            }
        } else {
            0.5 * d2 * d2
        };
        q1_loss += is_weights[i] * l1;
        q2_loss += is_weights[i] * l2;
    }
    let b_f = b as f32;
    q1_loss /= b_f;
    q2_loss /= b_f;
    Ok(Td3Loss {
        critic_loss: q1_loss + q2_loss,
        q1_loss,
        q2_loss,
        td_errors,
    })
}

/// Compute TD3 actor loss.
///
/// ```text
/// L_π = -mean(Q1(s_t, μ_θ(s_t)))
/// ```
///
/// The gradient flows through `q1_mu` with respect to the actor parameters.
///
/// # Arguments
///
/// * `q1_mu` — `[B]` Q1(s_t, μ_θ(s_t)) evaluated at current policy actions.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] if `q1_mu` is empty.
pub fn td3_actor_loss(q1_mu: &[f32]) -> RlResult<f32> {
    if q1_mu.is_empty() {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    Ok(-q1_mu.iter().sum::<f32>() / q1_mu.len() as f32)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    type Td3Batch = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    fn make_td3_inputs(b: usize) -> Td3Batch {
        let q1 = vec![1.0_f32; b];
        let q2 = vec![1.0_f32; b];
        let r = vec![0.5_f32; b];
        let d = vec![0.0_f32; b];
        let min_q_next = vec![1.0_f32; b];
        let w = vec![1.0_f32; b];
        (q1, q2, r, d, min_q_next, w)
    }

    #[test]
    fn td3_critic_loss_zero_when_perfect() {
        let cfg = Td3Config {
            gamma: 1.0,
            huber: false,
        };
        // target = r + γ * min_q_next = 0.5 + 1.0 = 1.5
        // Q1=Q2=1.5 → loss=0
        let (_, _, r, d, min_qn, w) = make_td3_inputs(4);
        let q = vec![1.5_f32; 4];
        let l = td3_critic_loss(&q, &q, &r, &d, &min_qn, &w, cfg).unwrap();
        assert!(l.critic_loss.abs() < 1e-5, "critic_loss={}", l.critic_loss);
    }

    #[test]
    fn td3_critic_loss_positive_when_off() {
        let (q1, q2, r, d, min_qn, w) = make_td3_inputs(4);
        let l = td3_critic_loss(&q1, &q2, &r, &d, &min_qn, &w, Td3Config::default()).unwrap();
        assert!(l.critic_loss > 0.0, "loss should be > 0");
    }

    #[test]
    fn td3_twin_q_both_contribute() {
        let (q1, _, r, d, min_qn, w) = make_td3_inputs(4);
        let q2_bad = vec![100.0_f32; 4]; // very wrong Q2
        let l = td3_critic_loss(&q1, &q2_bad, &r, &d, &min_qn, &w, Td3Config::default()).unwrap();
        assert!(l.q2_loss > l.q1_loss, "Q2 loss should be larger");
    }

    #[test]
    fn td3_done_stops_bootstrap() {
        let cfg = Td3Config {
            gamma: 1.0,
            huber: false,
        };
        let q1 = vec![2.0_f32];
        let q2 = vec![2.0_f32];
        let r = vec![2.0_f32];
        let d = vec![1.0_f32]; // done
        let min_qn = vec![100.0_f32]; // should be masked
        let w = vec![1.0_f32];
        let l = td3_critic_loss(&q1, &q2, &r, &d, &min_qn, &w, cfg).unwrap();
        // target = r = 2.0, Q1=Q2=2.0 → loss=0
        assert!(l.critic_loss.abs() < 1e-5, "done should mask future Q");
    }

    #[test]
    fn td3_actor_loss_negative_when_high_q() {
        let q1 = vec![5.0_f32; 4];
        let l = td3_actor_loss(&q1).unwrap();
        assert!(l < 0.0, "actor loss should be negative (we maximise Q1)");
        assert!((l - (-5.0)).abs() < 1e-5, "actor_loss={l}");
    }

    #[test]
    fn td3_actor_loss_empty_error() {
        assert!(td3_actor_loss(&[]).is_err());
    }

    #[test]
    fn td3_td_errors_all_positive() {
        let (q1, q2, r, d, min_qn, w) = make_td3_inputs(8);
        let l = td3_critic_loss(&q1, &q2, &r, &d, &min_qn, &w, Td3Config::default()).unwrap();
        for (i, &e) in l.td_errors.iter().enumerate() {
            assert!(e >= 0.0, "td_error[{i}]={e}");
        }
    }

    #[test]
    fn td3_huber_reduces_large_errors() {
        // With a large error, Huber should give smaller loss than MSE
        let q1 = vec![0.0_f32];
        let q2 = vec![0.0_f32];
        let r = vec![10.0_f32]; // large reward → large error
        let d = vec![1.0_f32]; // done, target=10
        let min_qn = vec![0.0_f32];
        let w = vec![1.0_f32];
        let l_mse = td3_critic_loss(
            &q1,
            &q2,
            &r,
            &d,
            &min_qn,
            &w,
            Td3Config {
                gamma: 1.0,
                huber: false,
            },
        )
        .unwrap();
        let l_hub = td3_critic_loss(
            &q1,
            &q2,
            &r,
            &d,
            &min_qn,
            &w,
            Td3Config {
                gamma: 1.0,
                huber: true,
            },
        )
        .unwrap();
        assert!(
            l_hub.critic_loss < l_mse.critic_loss,
            "Huber < MSE for large errors"
        );
    }
}
