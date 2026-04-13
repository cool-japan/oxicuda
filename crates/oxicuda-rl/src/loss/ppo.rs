//! # Proximal Policy Optimization (PPO) Loss
//!
//! Schulman et al. (2017), "Proximal Policy Optimization Algorithms".
//!
//! ## Combined PPO Loss
//!
//! ```text
//! L^CLIP(θ) = E_t[ min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t) ]
//! L^VF(θ)   = E_t[ (V_θ(s_t) - V_t^targ)² ]
//! L^ENT(θ)  = E_t[ H(π_θ(·|s_t)) ]
//! L(θ) = -L^CLIP + c_v L^VF - c_e L^ENT
//! ```

use crate::error::{RlError, RlResult};

/// PPO hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct PpoConfig {
    /// Clip threshold ε (default 0.2).
    pub clip_eps: f32,
    /// Value loss coefficient c_v.
    pub value_coeff: f32,
    /// Entropy bonus coefficient c_e.
    pub entropy_coeff: f32,
    /// Optional clip range for value function (`None` = no clip).
    pub value_clip: Option<f32>,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_eps: 0.2,
            value_coeff: 0.5,
            entropy_coeff: 0.01,
            value_clip: None,
        }
    }
}

/// PPO loss output.
#[derive(Debug, Clone)]
pub struct PpoLoss {
    /// Combined PPO loss (minimise this).
    pub total: f32,
    /// Clipped surrogate policy loss (positive = policy loss).
    pub policy_loss: f32,
    /// Value function MSE loss.
    pub value_loss: f32,
    /// Mean entropy (for monitoring).
    pub entropy: f32,
    /// Fraction of time the clip is active (for monitoring).
    pub clip_fraction: f32,
    /// Approximate KL divergence `E[log(π/π_old)]`.
    pub approx_kl: f32,
}

/// Compute the PPO loss for a mini-batch.
///
/// # Arguments
///
/// * `log_probs_new` — `[B]` log π_θ(a_t|s_t) under the current policy.
/// * `log_probs_old` — `[B]` log π_old(a_t|s_t) stored at collection time.
/// * `advantages`    — `[B]` advantage estimates (pre-normalised).
/// * `value_preds`   — `[B]` current value predictions `V_θ(s_t)`.
/// * `value_targets` — `[B]` value targets (e.g. from GAE returns).
/// * `entropies`     — `[B]` per-sample entropy (may be all-zeros).
/// * `old_vpreds`    — `[B]` old value predictions (used for optional clip).
/// * `cfg`           — PPO configuration.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] for inconsistent lengths.
#[allow(clippy::too_many_arguments)]
pub fn ppo_loss(
    log_probs_new: &[f32],
    log_probs_old: &[f32],
    advantages: &[f32],
    value_preds: &[f32],
    value_targets: &[f32],
    entropies: &[f32],
    old_vpreds: &[f32],
    cfg: PpoConfig,
) -> RlResult<PpoLoss> {
    let b = log_probs_new.len();
    if log_probs_old.len() != b
        || advantages.len() != b
        || value_preds.len() != b
        || value_targets.len() != b
        || entropies.len() != b
        || old_vpreds.len() != b
    {
        return Err(RlError::DimensionMismatch {
            expected: b,
            got: b.wrapping_sub(1),
        });
    }
    let b_f = b as f32;

    let mut policy_loss = 0.0_f32;
    let mut value_loss = 0.0_f32;
    let mut entropy_sum = 0.0_f32;
    let mut clip_count = 0_usize;
    let mut kl_sum = 0.0_f32;

    for i in 0..b {
        let ratio = (log_probs_new[i] - log_probs_old[i]).exp();
        let adv = advantages[i];
        // Clipped surrogate
        let surr1 = ratio * adv;
        let surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv;
        let pol = surr1.min(surr2);
        policy_loss += pol;

        // Clipping indicator
        if (ratio - 1.0).abs() > cfg.clip_eps {
            clip_count += 1;
        }

        // Value loss
        let vf = match cfg.value_clip {
            Some(vclip) => {
                let vf_unclipped = (value_preds[i] - value_targets[i]).powi(2);
                let vf_clipped = (old_vpreds[i]
                    + (value_preds[i] - old_vpreds[i]).clamp(-vclip, vclip)
                    - value_targets[i])
                    .powi(2);
                vf_unclipped.max(vf_clipped)
            }
            None => (value_preds[i] - value_targets[i]).powi(2),
        };
        value_loss += vf;

        entropy_sum += entropies[i];
        kl_sum += log_probs_old[i] - log_probs_new[i];
    }

    let policy_loss_mean = policy_loss / b_f;
    let value_loss_mean = value_loss / b_f;
    let entropy_mean = entropy_sum / b_f;

    // Total loss: negate policy_loss (we want to maximise it), minimise value_loss
    let total =
        -policy_loss_mean + cfg.value_coeff * value_loss_mean - cfg.entropy_coeff * entropy_mean;

    Ok(PpoLoss {
        total,
        policy_loss: -policy_loss_mean,
        value_loss: value_loss_mean,
        entropy: entropy_mean,
        clip_fraction: clip_count as f32 / b_f,
        approx_kl: kl_sum / b_f,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ppo_batch(b: usize, ratio: f32, adv: f32) -> PpoLoss {
        let lp_new = vec![ratio.ln() + 0.0_f32; b];
        let lp_old = vec![0.0_f32; b];
        let adv_v = vec![adv; b];
        let vp = vec![1.0_f32; b];
        let vt = vec![1.0_f32; b];
        let ent = vec![0.5_f32; b];
        let ovp = vec![1.0_f32; b];
        ppo_loss(
            &lp_new,
            &lp_old,
            &adv_v,
            &vp,
            &vt,
            &ent,
            &ovp,
            PpoConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn ppo_loss_zero_value_loss_when_match() {
        let l = make_ppo_batch(4, 1.0, 1.0);
        assert!(l.value_loss.abs() < 1e-5, "v_loss={}", l.value_loss);
    }

    #[test]
    fn ppo_loss_positive_when_bad_policy() {
        // Large ratio > 1+ε should be clipped, policy loss should not explode
        let l = make_ppo_batch(8, 10.0, 1.0); // ratio=10, clipped to 1.2
        assert!(l.total.is_finite(), "loss should be finite");
    }

    #[test]
    fn ppo_clip_fraction_all_clipped() {
        // ratio=10 >> 1+eps=1.2 → all clipped
        let l = make_ppo_batch(8, 10.0, 1.0);
        assert!(
            (l.clip_fraction - 1.0).abs() < 1e-5,
            "clip_fraction={}",
            l.clip_fraction
        );
    }

    #[test]
    fn ppo_clip_fraction_none_clipped() {
        // ratio=1.0 → no clipping
        let l = make_ppo_batch(8, 1.0, 1.0);
        assert!(l.clip_fraction < 1e-5, "clip_fraction={}", l.clip_fraction);
    }

    #[test]
    fn ppo_dimension_mismatch() {
        let b = vec![0.0_f32; 4];
        let b3 = vec![0.0_f32; 3];
        assert!(ppo_loss(&b, &b, &b, &b, &b, &b, &b3, PpoConfig::default()).is_err());
    }

    #[test]
    fn ppo_entropy_reduces_loss() {
        // Higher entropy → lower total loss (entropy bonus)
        let b = 8;
        let lp_new = vec![0.0_f32; b];
        let lp_old = vec![0.0_f32; b];
        let adv = vec![0.0_f32; b];
        let vp = vec![0.0_f32; b];
        let vt = vec![0.0_f32; b];
        let ovp = vec![0.0_f32; b];
        let ent_low = vec![0.1_f32; b];
        let ent_high = vec![2.0_f32; b];
        let l_low = ppo_loss(
            &lp_new,
            &lp_old,
            &adv,
            &vp,
            &vt,
            &ent_low,
            &ovp,
            PpoConfig::default(),
        )
        .unwrap();
        let l_high = ppo_loss(
            &lp_new,
            &lp_old,
            &adv,
            &vp,
            &vt,
            &ent_high,
            &ovp,
            PpoConfig::default(),
        )
        .unwrap();
        assert!(
            l_high.total < l_low.total,
            "higher entropy should reduce total loss"
        );
    }
}
