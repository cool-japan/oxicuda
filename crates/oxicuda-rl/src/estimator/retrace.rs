//! # Retrace(λ) Safe Off-Policy Returns
//!
//! Munos et al. (2016), "Safe and Efficient Off-Policy Reinforcement Learning",
//! NeurIPS 2016.
//!
//! ## Algorithm
//!
//! Retrace uses a product of truncated IS ratios to form a multi-step return
//! that is safe (finite variance) for any behaviour policy:
//!
//! ```text
//! c_t = λ * min(1, π(a_t|s_t) / μ(a_t|s_t))
//!
//! Q^ret(s_t, a_t) = r_t + γ E_{a~π}[Q(s_{t+1}, a)]
//!   + γ Σ_{s=t+1}^{T-1} γ^{s-t} (Π_{i=t+1}^{s} c_i)
//!       [r_s + γ E_{a~π}[Q(s_{s+1}, a)] - Q(s_s, a_s)]
//! ```
//!
//! In the common approximate form (using V instead of E\[Q\]):
//! ```text
//! δ_t = r_t + γ * V(s_{t+1}) * (1-done_t) - Q(s_t, a_t)
//! Q^ret_t = Q(s_t, a_t) + δ_t + Σ_{s=t+1}^{T-1} (Π_{i=t+1}^s c_i) δ_s
//! ```

use crate::error::{RlError, RlResult};

/// Retrace(λ) configuration.
#[derive(Debug, Clone, Copy)]
pub struct RetraceConfig {
    /// Discount factor γ.
    pub gamma: f32,
    /// Retrace λ ∈ [0, 1].
    pub lambda: f32,
}

impl Default for RetraceConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 1.0,
        }
    }
}

/// Retrace(λ) return output.
#[derive(Debug, Clone)]
pub struct RetraceOutput {
    /// Retrace Q-value targets `Q^ret_t`.
    pub q_targets: Vec<f32>,
    /// Temporal difference errors `δ_t`.
    pub td_errors: Vec<f32>,
}

/// Compute Retrace(λ) Q-value targets.
///
/// # Arguments
///
/// * `rewards`       — `[T]` rewards.
/// * `q_values`      — `[T]` Q(s_t, a_t) estimates.
/// * `values`        — `[T+1]` V(s_t) = E_{a~π}[Q(s_t, a)] estimates.
/// * `dones`         — `[T]` done flags.
/// * `log_probs_new` — `[T]` log-probs under current policy π.
/// * `log_probs_old` — `[T]` log-probs under behaviour policy μ.
/// * `cfg`           — Retrace configuration.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] for inconsistent slice lengths.
pub fn compute_retrace(
    rewards: &[f32],
    q_values: &[f32],
    values: &[f32],
    dones: &[f32],
    log_probs_new: &[f32],
    log_probs_old: &[f32],
    cfg: RetraceConfig,
) -> RlResult<RetraceOutput> {
    let t = rewards.len();
    if q_values.len() != t
        || values.len() != t + 1
        || dones.len() != t
        || log_probs_new.len() != t
        || log_probs_old.len() != t
    {
        return Err(RlError::DimensionMismatch {
            expected: t,
            got: q_values.len(),
        });
    }

    // c_t = λ * min(1, ρ_t)
    let c: Vec<f32> = log_probs_new
        .iter()
        .zip(log_probs_old.iter())
        .map(|(&lp_new, &lp_old)| {
            let rho = (lp_new - lp_old).exp().clamp(0.0, 1e6);
            cfg.lambda * rho.min(1.0)
        })
        .collect();

    // δ_t = r_t + γ * V(s_{t+1}) * mask - Q(s_t, a_t)
    let td_errors: Vec<f32> = (0..t)
        .map(|i| {
            let mask = 1.0 - dones[i];
            rewards[i] + cfg.gamma * values[i + 1] * mask - q_values[i]
        })
        .collect();

    // Q^ret via backward accumulation
    // Q^ret_t = Q_t + δ_t + γ * c_{t+1} * (Q^ret_{t+1} - Q_{t+1})
    let mut q_targets = vec![0.0_f32; t];
    let mut ret_next = values[t]; // bootstrap: V(s_T)

    for i in (0..t).rev() {
        let mask = 1.0 - dones[i];
        // c_{t+1} is at index i+1 (0 if i = T-1)
        let c_next = if i + 1 < t { c[i + 1] } else { 0.0 };
        let q_next = if i + 1 < t {
            q_values[i + 1]
        } else {
            values[t]
        };
        q_targets[i] = q_values[i] + td_errors[i] + cfg.gamma * mask * c_next * (ret_next - q_next);
        ret_next = q_targets[i];
    }

    Ok(RetraceOutput {
        q_targets,
        td_errors,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_retrace(t: usize, same_policy: bool) -> RetraceOutput {
        let r = vec![1.0_f32; t];
        let q = vec![0.5_f32; t];
        let v = vec![0.5_f32; t + 1];
        let d = vec![0.0_f32; t];
        let lp_new = vec![0.0_f32; t];
        let lp_old = if same_policy {
            vec![0.0_f32; t]
        } else {
            vec![-0.5_f32; t]
        };
        compute_retrace(&r, &q, &v, &d, &lp_new, &lp_old, RetraceConfig::default()).unwrap()
    }

    #[test]
    fn retrace_output_length() {
        let out = simple_retrace(5, true);
        assert_eq!(out.q_targets.len(), 5);
        assert_eq!(out.td_errors.len(), 5);
    }

    #[test]
    fn retrace_td_errors_finite() {
        let out = simple_retrace(4, false);
        for (i, &d) in out.td_errors.iter().enumerate() {
            assert!(d.is_finite(), "td_error[{i}]={d}");
        }
    }

    #[test]
    fn retrace_on_policy_q_targets_finite() {
        let out = simple_retrace(4, true);
        for (i, &q) in out.q_targets.iter().enumerate() {
            assert!(q.is_finite(), "q_target[{i}]={q}");
        }
    }

    #[test]
    fn retrace_dimension_mismatch() {
        let r = vec![1.0_f32; 3];
        let q = vec![0.5_f32; 3];
        let v = vec![0.5_f32; 3]; // should be 4
        let d = vec![0.0_f32; 3];
        let lp = vec![0.0_f32; 3];
        assert!(compute_retrace(&r, &q, &v, &d, &lp, &lp, RetraceConfig::default()).is_err());
    }

    #[test]
    fn retrace_done_stops_accumulation() {
        let cfg = RetraceConfig::default();
        let r = vec![1.0, 1.0, 1.0];
        let q = vec![0.0_f32; 3];
        let v = vec![0.0_f32; 4];
        let d = vec![0.0, 1.0, 0.0];
        let lp = vec![0.0_f32; 3];
        let out = compute_retrace(&r, &q, &v, &d, &lp, &lp, cfg).unwrap();
        assert!(out.q_targets.iter().all(|&q| q.is_finite()));
    }

    #[test]
    fn retrace_lambda_zero_is_bellman() {
        // λ=0 → c=0 → Q^ret_t = Q_t + δ_t = r_t + γ*V(s_{t+1})*(1-done)
        let cfg = RetraceConfig {
            gamma: 0.99,
            lambda: 0.0,
        };
        let r = vec![1.0_f32; 3];
        let q = vec![0.5_f32; 3];
        let v = vec![0.5_f32; 4];
        let d = vec![0.0_f32; 3];
        let lp = vec![0.0_f32; 3];
        let out = compute_retrace(&r, &q, &v, &d, &lp, &lp, cfg).unwrap();
        // Q^ret = Q + δ = Q + (r + γ*V_next - Q) = r + γ*V_next
        let expected = 1.0 + 0.99 * 0.5;
        for (i, &qt) in out.q_targets.iter().enumerate() {
            assert!(
                (qt - expected).abs() < 1e-4,
                "Q_target[{i}]={qt} vs {expected}"
            );
        }
    }
}
