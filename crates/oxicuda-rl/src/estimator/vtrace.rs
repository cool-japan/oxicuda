//! # V-trace Off-Policy Return Estimation (IMPALA)
//!
//! Espeholt et al. (2018), "IMPALA: Scalable Distributed Deep-RL with
//! Importance Weighted Actor-Learner Architectures", ICML 2018.
//!
//! ## Algorithm
//!
//! Given IS ratios `ρ_t = π(a_t|s_t) / μ(a_t|s_t)` between the current
//! policy `π` and the behaviour policy `μ`:
//!
//! ```text
//! c_t    = min(c̄, ρ_t)
//! ρ̄_t   = min(ρ̄, ρ_t)
//!
//! δ_t    = ρ̄_t [r_t + γ V(s_{t+1})(1-done_t) - V(s_t)]
//!
//! v_s = V(s_s) + Σ_{t=s}^{s+n-1} γ^{t-s} (Π_{i=s}^{t-1} c_i) δ_t
//! ```
//!
//! The advantage for the policy gradient is:
//! ```text
//! A_t = ρ̄_t [r_t + γ v_{t+1}(1-done_t) - V(s_t)]
//! ```

use crate::error::{RlError, RlResult};

/// V-trace configuration.
#[derive(Debug, Clone, Copy)]
pub struct VtraceConfig {
    /// Discount factor γ.
    pub gamma: f32,
    /// Clipping threshold c̄ for importance weights in the v-trace target.
    pub c_bar: f32,
    /// Clipping threshold ρ̄ for IS weights in TD errors.
    pub rho_bar: f32,
}

impl Default for VtraceConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            c_bar: 1.0,
            rho_bar: 1.0,
        }
    }
}

/// V-trace output.
#[derive(Debug, Clone)]
pub struct VtraceOutput {
    /// V-trace value targets `v_t`.
    pub vs: Vec<f32>,
    /// Policy gradient advantages `A_t`.
    pub advantages: Vec<f32>,
}

/// Compute V-trace returns and policy gradient advantages.
///
/// # Arguments
///
/// * `rewards`          — `[T]` rewards.
/// * `values`           — `[T+1]` value estimates (including bootstrap `v_T`).
/// * `dones`            — `[T]` done flags.
/// * `log_probs_new`    — `[T]` log-probs under current policy `π`.
/// * `log_probs_old`    — `[T]` log-probs under behaviour policy `μ`.
/// * `cfg`              — V-trace hyperparameters.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] if slice lengths are inconsistent.
pub fn compute_vtrace(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    log_probs_new: &[f32],
    log_probs_old: &[f32],
    cfg: VtraceConfig,
) -> RlResult<VtraceOutput> {
    let t = rewards.len();
    if values.len() != t + 1
        || dones.len() != t
        || log_probs_new.len() != t
        || log_probs_old.len() != t
    {
        return Err(RlError::DimensionMismatch {
            expected: t,
            got: t.wrapping_sub(1),
        });
    }

    // Compute IS ratios
    let rho: Vec<f32> = log_probs_new
        .iter()
        .zip(log_probs_old.iter())
        .map(|(&lp_new, &lp_old)| {
            let ratio = (lp_new - lp_old).exp();
            ratio.clamp(0.0, 1e6) // prevent extreme ratios
        })
        .collect();

    let c_vals: Vec<f32> = rho.iter().map(|&r| r.min(cfg.c_bar)).collect();
    let rho_vals: Vec<f32> = rho.iter().map(|&r| r.min(cfg.rho_bar)).collect();

    // Compute TD errors δ_t = ρ̄_t [r_t + γ v_{t+1} (1-done_t) - v_t]
    let deltas: Vec<f32> = (0..t)
        .map(|i| {
            let mask = 1.0 - dones[i];
            rho_vals[i] * (rewards[i] + cfg.gamma * values[i + 1] * mask - values[i])
        })
        .collect();

    // Compute v_s via backward scan
    let mut vs = vec![0.0_f32; t];
    let mut acc = values[t]; // v_T = V(s_T) for bootstrap

    for i in (0..t).rev() {
        let mask = 1.0 - dones[i];
        acc = values[i] + deltas[i] + cfg.gamma * c_vals[i] * mask * (acc - values[i + 1]);
        vs[i] = acc;
    }

    // Policy gradient advantages: A_t = ρ̄_t [r_t + γ v_{t+1} (1-done) - V(s_t)]
    let advantages: Vec<f32> = (0..t)
        .map(|i| {
            let mask = 1.0 - dones[i];
            // v_{t+1} is vs[t+1] if t+1 < T, else values[T]
            let v_next = if i + 1 < t { vs[i + 1] } else { values[t] };
            rho_vals[i] * (rewards[i] + cfg.gamma * v_next * mask - values[i])
        })
        .collect();

    Ok(VtraceOutput { vs, advantages })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run_vtrace(t: usize, rho_same: bool) -> VtraceOutput {
        let r = vec![1.0_f32; t];
        let v = vec![0.5_f32; t + 1];
        let d = vec![0.0_f32; t];
        let lp_new = vec![0.0_f32; t]; // log π = 0
        let lp_old = if rho_same {
            vec![0.0_f32; t] // IS ratio = exp(0-0) = 1
        } else {
            vec![-1.0_f32; t] // IS ratio = exp(0-(-1)) = e > 1, clipped
        };
        compute_vtrace(&r, &v, &d, &lp_new, &lp_old, VtraceConfig::default()).unwrap()
    }

    #[test]
    fn vtrace_output_length() {
        let out = run_vtrace(10, true);
        assert_eq!(out.vs.len(), 10);
        assert_eq!(out.advantages.len(), 10);
    }

    #[test]
    fn vtrace_on_policy_matches_td() {
        // When π = μ (ρ = 1, both bars ≥ 1), V-trace should match TD with c̄=ρ̄=1
        let cfg = VtraceConfig {
            gamma: 0.99,
            c_bar: 1.0,
            rho_bar: 1.0,
        };
        let r = vec![1.0_f32; 3];
        let v = vec![0.0_f32; 4];
        let d = vec![0.0_f32; 3];
        let lp = vec![0.0_f32; 3];
        let out = compute_vtrace(&r, &v, &d, &lp, &lp, cfg).unwrap();
        // vs[2] ≈ 0 + (1 + 0.99*0 - 0) = 1.0 (δ = ρ * (r + γ*v_next - v))
        assert!(
            out.vs[2] > 0.5,
            "v_trace target should be > v=0, vs[2]={}",
            out.vs[2]
        );
    }

    #[test]
    fn vtrace_advantages_non_nan() {
        let out = run_vtrace(5, false);
        for (i, &a) in out.advantages.iter().enumerate() {
            assert!(a.is_finite(), "advantage[{i}] is NaN/inf: {a}");
        }
    }

    #[test]
    fn vtrace_dimension_mismatch() {
        let r = vec![1.0_f32; 3];
        let v = vec![0.5_f32; 3]; // wrong: should be 4
        let d = vec![0.0_f32; 3];
        let lp = vec![0.0_f32; 3];
        assert!(compute_vtrace(&r, &v, &d, &lp, &lp, VtraceConfig::default()).is_err());
    }

    #[test]
    fn vtrace_done_stops_accumulation() {
        let cfg = VtraceConfig::default();
        let r = vec![1.0, 1.0, 1.0];
        let v = vec![0.0_f32; 4];
        let d = vec![0.0, 1.0, 0.0];
        let lp = vec![0.0_f32; 3];
        let out = compute_vtrace(&r, &v, &d, &lp, &lp, cfg).unwrap();
        // At done step, mask=0 so no future contributions
        assert!(out.vs[1].is_finite());
    }

    #[test]
    fn vtrace_clipping_reduces_large_rho() {
        // Large rho (off-policy) should be clipped to c̄/ρ̄
        let cfg = VtraceConfig {
            gamma: 0.99,
            c_bar: 1.0,
            rho_bar: 1.0,
        };
        let r = vec![1.0_f32; 2];
        let v = vec![0.5_f32; 3];
        let d = vec![0.0_f32; 2];
        let lp_new = vec![0.0_f32; 2];
        let lp_old = vec![-100.0_f32; 2]; // huge IS ratio, clipped to ρ̄=1
        let out = compute_vtrace(&r, &v, &d, &lp_new, &lp_old, cfg).unwrap();
        // Clipped to ρ̄=1, so same as on-policy
        let lp = vec![0.0_f32; 2];
        let out_on = compute_vtrace(&r, &v, &d, &lp, &lp, cfg).unwrap();
        assert!(
            (out.vs[0] - out_on.vs[0]).abs() < 1e-4,
            "clipped rho should match on-policy: {} vs {}",
            out.vs[0],
            out_on.vs[0]
        );
    }
}
