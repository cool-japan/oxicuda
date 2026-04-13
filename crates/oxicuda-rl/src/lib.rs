//! # OxiCUDA-RL — GPU-Accelerated Reinforcement Learning Primitives (Vol.9)
//!
//! `oxicuda-rl` provides a comprehensive set of GPU-ready RL building blocks:
//!
//! ## Replay Buffers
//!
//! * [`buffer::UniformReplayBuffer`] — fixed-capacity circular buffer with
//!   uniform random sampling (DQN, SAC, TD3).
//! * [`buffer::PrioritizedReplayBuffer`] — segment-tree PER with IS weight
//!   computation (PER-DQN, PER-SAC).
//! * [`buffer::NStepBuffer`] — n-step return accumulation with configurable
//!   discount and episode-boundary handling.
//!
//! ## Policy Distributions
//!
//! * [`policy::CategoricalPolicy`] — discrete actions with Gumbel-max
//!   sampling, log-probability, entropy, KL-divergence.
//! * [`policy::GaussianPolicy`] — diagonal Gaussian for continuous actions
//!   with reparameterisation trick and optional Tanh squashing (SAC).
//! * [`policy::DeterministicPolicy`] — DDPG/TD3 with exploration noise and
//!   TD3 target policy smoothing.
//!
//! ## Return / Advantage Estimators
//!
//! * [`estimator::compute_gae`] — GAE advantages and value targets (PPO, A3C).
//! * [`estimator::compute_td_lambda`] — TD(λ) multi-step returns.
//! * [`estimator::compute_vtrace`] — V-trace off-policy correction (IMPALA).
//! * [`estimator::compute_retrace`] — Retrace(λ) safe off-policy Q-targets.
//!
//! ## Loss Functions
//!
//! * [`loss::ppo_loss`] — PPO clip + value + entropy combined loss.
//! * [`loss::dqn_loss`] / [`loss::double_dqn_loss`] — Bellman MSE / Huber.
//! * [`loss::sac_critic_loss`] / [`loss::sac_actor_loss`] — SAC soft Q and
//!   policy losses with automatic temperature tuning.
//! * [`loss::td3_critic_loss`] / [`loss::td3_actor_loss`] — TD3 twin-Q critic
//!   and deterministic actor losses.
//!
//! ## Normalization
//!
//! * [`normalize::ObservationNormalizer`] — running mean/variance with clip.
//! * [`normalize::RewardNormalizer`] — return-based or clip normalization.
//! * [`normalize::RunningStats`] — Welford online statistics tracker.
//!
//! ## Environment Abstractions
//!
//! * [`env::Env`] — standard RL environment trait (`reset`, `step`).
//! * [`env::VecEnv`] — vectorized multi-environment wrapper with auto-reset.
//! * [`env::env::LinearQuadraticEnv`] — reference LQ environment for testing.
//!
//! ## PTX Kernels
//!
//! * [`ptx_kernels`] — GPU PTX source strings for TD-error, PPO ratio, SAC
//!   target, PER IS weight computation, and advantage normalisation.
//!
//! ## Quick Start
//!
//! ```rust
//! use oxicuda_rl::buffer::UniformReplayBuffer;
//! use oxicuda_rl::policy::CategoricalPolicy;
//! use oxicuda_rl::estimator::{GaeConfig, compute_gae};
//! use oxicuda_rl::loss::{PpoConfig, ppo_loss};
//! use oxicuda_rl::handle::RlHandle;
//!
//! // Set up replay buffer
//! let mut buf = UniformReplayBuffer::new(10_000, 8, 4);
//! let mut handle = RlHandle::default_handle();
//!
//! // Push some experience
//! for i in 0..100_usize {
//!     buf.push(
//!         vec![i as f32; 8],
//!         vec![0.0_f32; 4],
//!         1.0,
//!         vec![i as f32 + 1.0; 8],
//!         false,
//!     );
//! }
//!
//! // Sample a mini-batch
//! let batch = buf.sample(32, &mut handle).unwrap();
//! assert_eq!(batch.len(), 32);
//!
//! // Compute GAE for a 5-step rollout
//! let rewards    = vec![1.0_f32; 5];
//! let values     = vec![0.5_f32; 5];
//! let next_vals  = vec![0.5_f32; 5];
//! let dones      = vec![0.0_f32; 5];
//! let gae = compute_gae(&rewards, &values, &next_vals, &dones, GaeConfig::default()).unwrap();
//! assert_eq!(gae.advantages.len(), 5);
//! ```
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::module_inception)]
#![allow(clippy::wildcard_imports)]

// ─── Public modules ──────────────────────────────────────────────────────────

/// Error types and result alias.
pub mod error;

/// RL session handle: SM version, device info, seeded RNG.
pub mod handle;

/// PTX kernel sources for GPU-accelerated RL operations.
pub mod ptx_kernels;

/// Experience replay buffers.
pub mod buffer;

/// Policy distributions for discrete and continuous action spaces.
pub mod policy;

/// Return and advantage estimators.
pub mod estimator;

/// RL algorithm loss functions.
pub mod loss;

/// Observation and reward normalization.
pub mod normalize;

/// Environment abstractions.
pub mod env;

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use error::{RlError, RlResult};

/// Convenience prelude: imports the most commonly used types.
pub mod prelude {
    pub use crate::buffer::{
        NStepBuffer, NStepTransition, PrioritizedReplayBuffer, PrioritySample, Transition,
        UniformReplayBuffer,
    };
    pub use crate::env::env::{Env, EnvInfo, LinearQuadraticEnv, StepResult};
    pub use crate::env::vectorized::{VecEnv, VecStepResult};
    pub use crate::error::{RlError, RlResult};
    pub use crate::estimator::{
        GaeConfig, RetraceConfig, TdConfig, VtraceConfig, VtraceOutput, compute_gae,
        compute_retrace, compute_td_lambda, compute_vtrace,
    };
    pub use crate::handle::{LcgRng, RlHandle, SmVersion};
    pub use crate::loss::{
        DqnConfig, DqnLoss, PpoConfig, PpoLoss, SacConfig, SacLoss, Td3Config, Td3Loss,
        double_dqn_loss, dqn_loss, ppo_loss, sac_actor_loss, sac_critic_loss, sac_temperature_loss,
        td3_actor_loss, td3_critic_loss,
    };
    pub use crate::normalize::{ObservationNormalizer, RewardNormalizer, RunningStats};
    pub use crate::policy::{
        CategoricalPolicy, DeterministicPolicy, GaussianPolicy, deterministic::OrnsteinUhlenbeck,
    };
}

// ─── Integration tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::prelude::*;

    /// End-to-end DQN-style training loop simulation.
    #[test]
    fn e2e_dqn_style_loop() {
        let obs_dim = 4;
        let n_actions = 2;
        let mut buf = UniformReplayBuffer::new(1000, obs_dim, 1);
        let mut handle = RlHandle::default_handle();
        let mut env = LinearQuadraticEnv::new(obs_dim, 200);
        let policy = CategoricalPolicy::new(n_actions);

        let mut obs = env.reset().unwrap();
        // Collect 200 transitions
        for _ in 0..200 {
            // Dummy logits
            let logits = obs.iter().take(n_actions).copied().collect::<Vec<_>>();
            let probs = policy.softmax(&logits).unwrap();
            let _action = policy.sample_action(&probs, &mut handle).unwrap();
            let result = env.step(&[0.0; 4]).unwrap();
            buf.push(
                obs.clone(),
                vec![_action as f32],
                result.reward,
                result.obs.clone(),
                result.done,
            );
            obs = if result.done {
                env.reset().unwrap()
            } else {
                result.obs
            };
        }
        assert!(buf.len() >= 32, "should have enough transitions");

        // Sample and compute loss
        let batch = buf.sample(32, &mut handle).unwrap();
        let q_sa: Vec<f32> = batch.iter().map(|t| t.reward).collect();
        let rewards: Vec<f32> = batch.iter().map(|t| t.reward).collect();
        let max_q_next: Vec<f32> = batch.iter().map(|_| 0.0).collect();
        let dones: Vec<f32> = batch
            .iter()
            .map(|t| if t.done { 1.0 } else { 0.0 })
            .collect();
        let is_w = vec![1.0_f32; 32];
        let l = dqn_loss(
            &q_sa,
            &rewards,
            &max_q_next,
            &dones,
            &is_w,
            DqnConfig::default(),
        )
        .unwrap();
        assert!(l.loss.is_finite(), "DQN loss should be finite");
    }

    /// End-to-end PPO-style advantage computation + loss.
    #[test]
    fn e2e_ppo_gae_loss() {
        let t = 128;
        let rewards: Vec<f32> = (0..t)
            .map(|i| if i % 10 == 9 { -1.0 } else { 0.1 })
            .collect();
        let values: Vec<f32> = vec![0.5; t];
        let next_vals: Vec<f32> = vec![0.5; t];
        let dones: Vec<f32> = (0..t)
            .map(|i| if i % 10 == 9 { 1.0 } else { 0.0 })
            .collect();

        let gae = compute_gae(&rewards, &values, &next_vals, &dones, GaeConfig::default()).unwrap();
        assert_eq!(gae.advantages.len(), t);

        // Simulate PPO mini-batch update
        let lp_new = vec![-0.693_f32; t]; // ln(0.5)
        let lp_old = vec![-0.693_f32; t];
        let vp = vec![0.5_f32; t];
        let ent = vec![0.693_f32; t];
        let ovp = vec![0.5_f32; t];
        let l = ppo_loss(
            &lp_new,
            &lp_old,
            &gae.advantages,
            &vp,
            &gae.returns,
            &ent,
            &ovp,
            PpoConfig::default(),
        )
        .unwrap();
        assert!(
            l.total.is_finite(),
            "PPO loss should be finite: {}",
            l.total
        );
        assert!(l.clip_fraction >= 0.0 && l.clip_fraction <= 1.0);
    }

    /// End-to-end SAC-style off-policy update.
    #[test]
    fn e2e_sac_style_update() {
        let mut buf = PrioritizedReplayBuffer::new(256, 8, 2, 0.6, 0.4);
        let mut handle = RlHandle::default_handle();
        for i in 0..256_usize {
            buf.push(
                vec![i as f32 * 0.01; 8],
                vec![0.1_f32; 2],
                (i % 5) as f32 * 0.2,
                vec![(i + 1) as f32 * 0.01; 8],
                i % 20 == 19,
            );
        }
        let batch = buf.sample(32, &mut handle).unwrap();
        let q: Vec<f32> = batch.iter().map(|s| s.transition.reward).collect();
        let r: Vec<f32> = batch.iter().map(|s| s.transition.reward).collect();
        let d: Vec<f32> = batch
            .iter()
            .map(|s| if s.transition.done { 1.0 } else { 0.0 })
            .collect();
        let min_qn = vec![0.5_f32; 32];
        let lp_next = vec![-0.5_f32; 32];
        let is_w: Vec<f32> = batch.iter().map(|s| s.weight).collect();
        let (cl, _) =
            sac_critic_loss(&q, &r, &d, &min_qn, &lp_next, &is_w, SacConfig::default()).unwrap();
        assert!(cl.is_finite(), "SAC critic loss should be finite");
    }

    /// VecEnv with observation normalization.
    #[test]
    fn e2e_vecenv_with_obs_norm() {
        let envs: Vec<_> = (0..4).map(|_| LinearQuadraticEnv::new(3, 50)).collect();
        let mut ve = VecEnv::new(envs);
        let mut norm = ObservationNormalizer::new(3);
        let init_obs = ve.reset_all().unwrap();
        for chunk in init_obs.chunks_exact(3) {
            norm.process_one(chunk).unwrap();
        }
        let actions = vec![0.01_f32; 4 * 3];
        for _ in 0..20 {
            let result = ve.step(&actions).unwrap();
            for chunk in result.obs.chunks_exact(3) {
                let _norm_obs = norm.process_one(chunk).unwrap();
            }
        }
        assert!(norm.count() > 0);
    }

    /// N-step buffer integration.
    #[test]
    fn e2e_n_step_buffer() {
        let mut nsbuf = NStepBuffer::new(3, 0.99);
        let mut transitions = Vec::new();
        for i in 0..20_usize {
            if let Some(t) = nsbuf.push([i as f32], [0.0], 1.0, [(i + 1) as f32], false) {
                transitions.push(t);
            }
        }
        // Should have transitions after first 3 steps
        assert!(!transitions.is_empty(), "n-step should produce transitions");
        for t in &transitions {
            assert_eq!(t.actual_n, 3);
            // R = 1 + 0.99 + 0.99^2 ≈ 2.9701
            assert!(
                (t.n_step_return - (1.0 + 0.99 + 0.99_f32 * 0.99)).abs() < 0.01,
                "n_step_return={}",
                t.n_step_return
            );
        }
    }
}
