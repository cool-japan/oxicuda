//! # OxiCUDA Training Engine — Vol.8
//!
//! `oxicuda-train` provides a full GPU-accelerated training stack:
//!
//! * **Fused optimizer kernels** — PTX-generated fused parameter update kernels
//!   for Adam, AdamW, Lion, CAME, and Muon, keeping all optimizer state on-device
//!   and minimising memory traffic.
//! * **Gradient clipping** — global norm clip, per-layer norm clip, and element
//!   value clip.
//! * **Gradient accumulation** — micro-batch accumulation with configurable
//!   step count and averaging/summing mode.
//! * **Gradient checkpointing** — activation recomputation to reduce memory by
//!   trading compute; supports uniform, selective, and offload policies.
//! * **LR schedulers** — 11 ready-to-use schedules covering constant, step,
//!   multi-step, exponential, cosine, warmup+cosine, polynomial, 1cycle,
//!   cyclic, and reduce-on-plateau.
//! * **ZeRO optimizer state sharding** — ZeRO-1/2/3 optimizer state, gradient,
//!   and parameter partitioning for memory-efficient distributed training.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use oxicuda_train::gpu_optimizer::{GpuOptimizer, ParamTensor};
//! use oxicuda_train::gpu_optimizer::adamw::GpuAdamW;
//! use oxicuda_train::grad_clip::clip_grad_norm;
//! use oxicuda_train::lr_scheduler::{LrScheduler, WarmupCosine};
//!
//! // Build model parameters
//! let mut params = vec![
//!     ParamTensor::new(vec![0.5f32; 1024], "embed"),
//!     ParamTensor::new(vec![0.1f32; 4096], "ffn"),
//! ];
//!
//! // Create AdamW optimizer
//! let mut opt = GpuAdamW::new(3e-4).with_weight_decay(0.01);
//!
//! // Create LR scheduler
//! let mut sched = WarmupCosine::new(3e-4, 500, 10_000);
//!
//! // Training loop
//! for step in 0..10_000u64 {
//!     // ... compute gradients into params[i].grad ...
//!     # for p in &mut params { p.set_grad(vec![0.001f32; p.len()]).unwrap(); }
//!
//!     // Clip gradients
//!     clip_grad_norm(&mut params, 1.0).unwrap();
//!
//!     // Optimiser step
//!     let lr = sched.step();
//!     opt.set_lr(lr);
//!     opt.step(&mut params).unwrap();
//!
//!     // Zero gradients for next step
//!     opt.zero_grad(&mut params);
//! }
//! ```
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::wildcard_imports)]

// ─── Public modules ──────────────────────────────────────────────────────────

/// Error types for the training engine.
pub mod error;

/// Training session handle (CUDA context + stream + SM version).
pub mod handle;

/// PTX kernel generators for fused optimizer parameter updates.
pub mod ptx_kernels;

/// GPU-resident optimizer implementations.
pub mod gpu_optimizer;

/// Gradient clipping (global norm, per-layer, value).
pub mod grad_clip;

/// Gradient accumulation over micro-batches.
pub mod grad_accum;

/// Gradient checkpointing (activation recomputation).
pub mod checkpoint;

/// Learning rate schedulers (11 variants).
pub mod lr_scheduler;

/// ZeRO optimizer state sharding (Stage 1/2/3).
pub mod zero;

/// Automatic Mixed Precision — GradScaler with dynamic loss scaling.
pub mod amp;

/// Exponential Moving Average of model parameters.
pub mod ema;

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use error::{TrainError, TrainResult};

/// Convenience prelude.
pub mod prelude {
    pub use crate::amp::{AmpState, GradScaler, GradScalerConfig, has_overflow};
    pub use crate::checkpoint::{CheckpointManager, CheckpointPolicy};
    pub use crate::ema::{EmaDecayMode, ExponentialMovingAverage, LayerDecay};
    pub use crate::error::{TrainError, TrainResult};
    pub use crate::gpu_optimizer::adagrad::GpuAdaGrad;
    pub use crate::gpu_optimizer::adam::GpuAdam;
    pub use crate::gpu_optimizer::adamw::GpuAdamW;
    pub use crate::gpu_optimizer::came::GpuCame;
    pub use crate::gpu_optimizer::lion::GpuLion;
    pub use crate::gpu_optimizer::muon::GpuMuon;
    pub use crate::gpu_optimizer::radam::GpuRAdam;
    pub use crate::gpu_optimizer::rmsprop::GpuRMSProp;
    pub use crate::gpu_optimizer::{GpuOptimizer, ParamTensor};
    pub use crate::grad_accum::GradientAccumulator;
    pub use crate::grad_clip::{GlobalNormClip, GradientClipper, clip_grad_norm};
    pub use crate::lr_scheduler::{
        CosineAnnealingLR, CyclicLR, ExponentialLR, LinearWarmup, LrScheduler, MultiStepLR,
        OneCycleLR, PolynomialDecayLR, ReduceLROnPlateau, StepLR, WarmupCosine,
    };
    pub use crate::zero::{ZeroConfig, ZeroOptimizer, ZeroStage};
}

// ─── Integration tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::prelude::*;

    // Full training loop integration test
    #[test]
    fn e2e_training_loop_adamw_with_scheduler() {
        // Minimise f(x, y) = x² + 2y² via gradient [2x, 4y]
        let mut params = vec![
            ParamTensor::new(vec![3.0_f32], "x"),
            ParamTensor::new(vec![2.0_f32], "y"),
        ];

        // Use beta2=0.9 so the second-moment bias correction converges in
        // ~20 steps rather than the ~1000 steps needed with default beta2=0.999.
        let mut opt = GpuAdamW::new(1e-2).with_weight_decay(0.0).with_beta2(0.9);
        let mut sched = WarmupCosine::new(1e-2, 50, 500);

        for _step in 0..500u64 {
            let x = params[0].data[0];
            let y = params[1].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            params[1].set_grad(vec![4.0 * y]).unwrap();

            clip_grad_norm(&mut params, 10.0).unwrap();

            let lr = sched.step();
            opt.set_lr(lr);
            opt.step(&mut params).unwrap();
        }

        let x = params[0].data[0].abs();
        let y = params[1].data[0].abs();
        // WarmupCosine decays lr → 0 over 500 steps; total ∫lr dt ≈ 2.5,
        // so x₀=3 → x≈0.5 and y₀=2 → y≈0.5. Threshold chosen accordingly.
        assert!(x < 0.6, "x should converge, got {x}");
        assert!(y < 0.6, "y should converge, got {y}");
    }

    #[test]
    fn e2e_gradient_accumulation_with_lion() {
        // Accumulate 4 micro-batches before each step
        let k = 4;
        let mut accum = GradientAccumulator::new(k);
        // lr=1e-2: Lion takes exactly ±lr per step, so 100 outer steps × 0.01 = 1.0
        // total displacement from x=2 → x≈1.
        let mut opt = GpuLion::new(1e-2);
        let mut params = vec![ParamTensor::new(vec![2.0_f32; 4], "w")];

        for outer in 0..100_usize {
            for _ in 0..k {
                let x = params[0].data[0];
                // param has 4 elements; gradient must match length
                params[0].set_grad(vec![2.0 * x; 4]).unwrap();
                accum.accumulate(&params).unwrap();
            }
            accum.finalise(&mut params).unwrap();
            opt.step(&mut params).unwrap();
            opt.zero_grad(&mut params);
            let _ = outer;
        }

        let val = params[0].data[0].abs();
        assert!(val < 1.5, "should make progress toward 0, got |x|={val}");
    }

    #[test]
    fn e2e_checkpoint_and_recompute() {
        use crate::checkpoint::{CheckpointManager, CheckpointPolicy, RecomputeFn};

        let mut mgr = CheckpointManager::new(CheckpointPolicy::Uniform { interval: 2 });

        // Forward: checkpoint every other layer
        let activations: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32; 8]).collect();

        for (i, act) in activations.iter().enumerate() {
            if mgr.should_checkpoint(&format!("layer_{i}")) {
                mgr.save_input(&format!("layer_{i}"), act.clone()).unwrap();
            }
        }

        // Should have saved layers 0 and 2 (every 2)
        assert!(mgr.get_input("layer_0").is_some());
        assert!(mgr.get_input("layer_1").is_none()); // not checkpointed
        assert!(mgr.get_input("layer_2").is_some());

        // Backward recomputation via closure
        let relu = RecomputeFn::new("relu", |x| x.iter().map(|&v| v.max(0.0)).collect());
        let inp = mgr.get_input("layer_0").unwrap().to_vec();
        let recomputed = relu.run(&inp);
        assert_eq!(recomputed, inp); // all >= 0 so relu is identity here
    }

    #[test]
    fn e2e_zero_stage2_with_adamw() {
        let cfg = ZeroConfig::stage2(0, 1); // single rank = normal training
        let mut zero = ZeroOptimizer::new(GpuAdamW::new(1e-3), cfg);
        let mut params = vec![{
            let mut p = ParamTensor::new(vec![1.0_f32; 16], "w");
            p.set_grad(vec![0.5_f32; 16]).unwrap();
            p
        }];
        zero.step(&mut params).unwrap();
        for &v in &params[0].data {
            assert!(v < 1.0, "param should decrease after ZeRO step");
        }
        assert_eq!(zero.step_count(), 1);
    }

    #[test]
    fn e2e_came_with_cyclic_lr() {
        let mut opt = GpuCame::new(1e-3);
        let mut sched = CyclicLR::new(1e-4, 1e-2, 10);
        let mut params = vec![ParamTensor::new(vec![5.0_f32], "p")];

        for _ in 0..50 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            let lr = sched.step();
            opt.set_lr(lr);
            opt.step(&mut params).unwrap();
        }
        let val = params[0].data[0].abs();
        assert!(
            val < 5.0,
            "CAME + CyclicLR should reduce loss, got |x|={val}"
        );
    }

    #[test]
    fn e2e_muon_with_reduce_on_plateau() {
        let mut opt = GpuMuon::new(1e-2).with_ns_steps(0);
        let mut sched = ReduceLROnPlateau::new(1e-2, 0.5, 5);
        let mut params = vec![ParamTensor::new(vec![3.0_f32], "p")];
        let mut last_loss = f32::INFINITY;

        for _ in 0..100 {
            let x = params[0].data[0];
            let loss = x * x;
            let _ = sched.step_metric(loss as f64).unwrap();

            params[0].set_grad(vec![2.0 * x]).unwrap();
            let lr = sched.get_lr();
            opt.set_lr(lr);
            opt.step(&mut params).unwrap();
            last_loss = loss;
        }
        assert!(
            last_loss < 9.0,
            "loss should improve over time, got {last_loss}"
        );
    }
}
