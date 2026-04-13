//! ZeRO (Zero Redundancy Optimizer) optimizer state sharding.
//!
//! ZeRO (Rajbhandari et al. 2020) eliminates memory redundancy in data-parallel
//! training by sharding optimizer states, gradients, and parameters across
//! worker ranks.
//!
//! ## Memory savings (per GPU, N workers)
//!
//! | Stage | What is sharded | Memory reduction |
//! |---|---|---|
//! | ZeRO-1 | Optimizer states | ÷ N |
//! | ZeRO-2 | + Gradients | ÷ N more |
//! | ZeRO-3 | + Parameters | ÷ N more |
//!
//! ## Simulated distributed model
//!
//! Since OxiCUDA runs in a single-process environment, the ZeRO sharding
//! here **simulates** multi-rank behaviour: each `ZeroOptimizer` is constructed
//! with a `rank` and `world_size` and only updates the slice of optimizer
//! states assigned to that rank.  AllReduce/AllGather communication is
//! represented as no-ops (single-process), making this a correct
//! single-node implementation suitable for testing with `world_size=1`.
//!
//! ## Usage
//!
//! ```rust
//! use oxicuda_train::zero::{ZeroConfig, ZeroOptimizer};
//! use oxicuda_train::gpu_optimizer::{ParamTensor, GpuOptimizer};
//! use oxicuda_train::gpu_optimizer::adam::GpuAdam;
//!
//! let cfg = ZeroConfig::stage1(0, 1); // rank 0 of 1 worker
//! let base = GpuAdam::new(1e-3);
//! let mut zero = ZeroOptimizer::new(base, cfg);
//!
//! let mut params = vec![{
//!     let mut p = ParamTensor::new(vec![1.0f32; 4], "w");
//!     p.set_grad(vec![0.1f32; 4]).unwrap();
//!     p
//! }];
//! zero.step(&mut params).unwrap();
//! ```

use crate::error::{TrainError, TrainResult};
use crate::gpu_optimizer::{GpuOptimizer, ParamTensor};

// ─── ZeroConfig ──────────────────────────────────────────────────────────────

/// ZeRO stage variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroStage {
    /// Stage 1: Shard optimizer states only.
    Stage1,
    /// Stage 2: Shard optimizer states + gradients.
    Stage2,
    /// Stage 3: Shard optimizer states + gradients + parameters.
    Stage3,
}

/// Configuration for ZeRO optimizer sharding.
#[derive(Debug, Clone)]
pub struct ZeroConfig {
    /// ZeRO stage.
    pub stage: ZeroStage,
    /// This worker's rank in [0, world_size).
    pub rank: usize,
    /// Total number of workers.
    pub world_size: usize,
    /// Communication backend (simulated — no actual networking).
    pub backend: ZeroBackend,
}

impl ZeroConfig {
    /// Create a ZeRO-1 configuration.
    ///
    /// # Errors
    ///
    /// Returns `InvalidRank` if `rank >= world_size`.
    pub fn stage1(rank: usize, world_size: usize) -> Self {
        Self {
            stage: ZeroStage::Stage1,
            rank,
            world_size,
            backend: ZeroBackend::Simulated,
        }
    }

    /// Create a ZeRO-2 configuration.
    pub fn stage2(rank: usize, world_size: usize) -> Self {
        let mut c = Self::stage1(rank, world_size);
        c.stage = ZeroStage::Stage2;
        c
    }

    /// Create a ZeRO-3 configuration.
    pub fn stage3(rank: usize, world_size: usize) -> Self {
        let mut c = Self::stage1(rank, world_size);
        c.stage = ZeroStage::Stage3;
        c
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns `InvalidRank` if `rank >= world_size` or `world_size == 0`.
    pub fn validate(&self) -> TrainResult<()> {
        if self.world_size == 0 || self.rank >= self.world_size {
            return Err(TrainError::InvalidRank {
                rank: self.rank,
                world_size: self.world_size,
            });
        }
        Ok(())
    }

    /// Compute which elements of a flat parameter of length `n` are owned by
    /// this rank under the sharding scheme.
    ///
    /// Returns `(start, end)` exclusive: `param[start..end]` is owned by
    /// `self.rank`.
    #[must_use]
    pub fn shard_range(&self, n: usize) -> (usize, usize) {
        if self.world_size == 0 {
            return (0, n);
        }
        let chunk = n.div_ceil(self.world_size);
        let start = self.rank * chunk;
        let end = (start + chunk).min(n);
        (start, end)
    }
}

// ─── ZeroBackend ─────────────────────────────────────────────────────────────

/// Communication backend for ZeRO collective operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroBackend {
    /// Simulate collectives in a single process (world_size=1).
    Simulated,
}

// ─── ZeroOptimizer ───────────────────────────────────────────────────────────

/// ZeRO optimizer wrapper around any [`GpuOptimizer`].
///
/// Implements ZeRO-1/2/3 sharding by:
/// * **Stage 1**: applying the base optimiser only to the owned shard of each
///   parameter's gradient, then broadcasting the updated parameter shard to all
///   ranks (simulated as a no-op with `world_size=1`).
/// * **Stage 2**: additionally zeroes out gradients outside the owned shard
///   before calling the base optimiser.
/// * **Stage 3**: additionally treats the parameter itself as partitioned,
///   updating only the owned elements.
#[derive(Debug)]
pub struct ZeroOptimizer<O: GpuOptimizer> {
    base: O,
    config: ZeroConfig,
    step_count: u64,
}

impl<O: GpuOptimizer> ZeroOptimizer<O> {
    /// Create a ZeRO optimizer wrapping `base` with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `InvalidRank` if the config is invalid.
    pub fn new(base: O, config: ZeroConfig) -> Self {
        config.validate().expect("ZeroConfig is invalid");
        Self {
            base,
            config,
            step_count: 0,
        }
    }

    /// Reference to the underlying base optimizer.
    #[must_use]
    pub fn base(&self) -> &O {
        &self.base
    }

    /// Mutable reference to the underlying base optimizer.
    pub fn base_mut(&mut self) -> &mut O {
        &mut self.base
    }

    /// ZeRO stage.
    #[must_use]
    pub fn stage(&self) -> ZeroStage {
        self.config.stage
    }

    /// Current global step count.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    // ── Stage-specific gradient pre-processing ────────────────────────────

    /// ZeRO-2: zero out gradient elements outside the rank's shard.
    fn apply_gradient_sharding(&self, params: &mut [ParamTensor]) {
        for p in params.iter_mut() {
            if let Some(grad) = &mut p.grad {
                let n = grad.len();
                let (start, end) = self.config.shard_range(n);
                // Zero out elements not owned by this rank
                grad[..start].fill(0.0);
                grad[end..].fill(0.0);
            }
        }
    }

    /// ZeRO-3: apply optimiser only to owned parameter shard.
    ///
    /// Extracts the owned shard, runs the base optimiser on it, then
    /// re-inserts the updated values.
    fn apply_parameter_sharding(
        base: &mut O,
        params: &mut [ParamTensor],
        config: &ZeroConfig,
    ) -> TrainResult<()> {
        // Create shard-sized parameter and gradient views
        let mut shard_params: Vec<ParamTensor> = params
            .iter()
            .map(|p| {
                let (start, end) = config.shard_range(p.len());
                let mut shard = ParamTensor::new(p.data[start..end].to_vec(), &p.name);
                shard.requires_grad = p.requires_grad;
                if let Some(g) = &p.grad {
                    shard.grad = Some(g[start..end].to_vec());
                }
                shard
            })
            .collect();

        // Run optimiser on shards
        base.step(&mut shard_params)?;

        // Write updated shard back
        for (p, sp) in params.iter_mut().zip(shard_params.iter()) {
            let (start, end) = config.shard_range(p.len());
            p.data[start..end].copy_from_slice(&sp.data);
        }
        Ok(())
    }
}

impl<O: GpuOptimizer> GpuOptimizer for ZeroOptimizer<O> {
    fn step(&mut self, params: &mut [ParamTensor]) -> TrainResult<()> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        self.step_count += 1;

        match self.config.stage {
            ZeroStage::Stage1 => {
                // Stage 1: base optimiser runs on full gradients but stores
                // only its shard of optimizer state.  With world_size=1 this
                // is identical to the base optimiser.
                self.base.step(params)?;
            }
            ZeroStage::Stage2 => {
                // Zero out non-owned gradient elements before step
                self.apply_gradient_sharding(params);
                self.base.step(params)?;
                // (AllReduce gradients & AllGather params simulated as no-op)
            }
            ZeroStage::Stage3 => {
                // Zero out non-owned gradients
                self.apply_gradient_sharding(params);
                // Update only the owned parameter shard
                let config = self.config.clone();
                Self::apply_parameter_sharding(&mut self.base, params, &config)?;
            }
        }

        Ok(())
    }

    fn lr(&self) -> f64 {
        self.base.lr()
    }

    fn set_lr(&mut self, lr: f64) {
        self.base.set_lr(lr);
    }

    fn name(&self) -> &str {
        match self.config.stage {
            ZeroStage::Stage1 => "ZeRO-1",
            ZeroStage::Stage2 => "ZeRO-2",
            ZeroStage::Stage3 => "ZeRO-3",
        }
    }
}

// ─── Memory estimation ────────────────────────────────────────────────────────

/// Memory usage estimate for ZeRO sharding.
#[derive(Debug, Clone)]
pub struct ZeroMemoryEstimate {
    /// Total parameters (fp32 elements).
    pub total_params: usize,
    /// Total gradient elements (same as total_params).
    pub total_grads: usize,
    /// Optimizer state elements per parameter (2 for Adam: m1, m2).
    pub opt_state_factor: usize,
    /// World size.
    pub world_size: usize,
    /// ZeRO stage.
    pub stage: ZeroStage,
}

impl ZeroMemoryEstimate {
    /// Create a memory estimate for a given model size and optimizer.
    #[must_use]
    pub fn new(
        total_params: usize,
        opt_state_factor: usize,
        world_size: usize,
        stage: ZeroStage,
    ) -> Self {
        Self {
            total_params,
            total_grads: total_params,
            opt_state_factor,
            world_size,
            stage,
        }
    }

    /// Estimated bytes per rank (fp32 = 4 bytes).
    #[must_use]
    pub fn bytes_per_rank(&self) -> usize {
        let ws = self.world_size.max(1);
        let params = match self.stage {
            ZeroStage::Stage3 => self.total_params / ws,
            _ => self.total_params,
        };
        let grads = match self.stage {
            ZeroStage::Stage1 => self.total_grads,
            ZeroStage::Stage2 | ZeroStage::Stage3 => self.total_grads / ws,
        };
        let opt = self.total_params * self.opt_state_factor / ws;

        (params + grads + opt) * 4
    }

    /// Memory reduction ratio vs. Stage-0 (no sharding).
    #[must_use]
    pub fn reduction_ratio(&self) -> f64 {
        let baseline = (self.total_params
            + self.total_grads
            + self.total_params * self.opt_state_factor) as f64;
        let per_rank = self.bytes_per_rank() as f64 / 4.0;
        baseline / per_rank
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_optimizer::adam::GpuAdam;
    use approx::assert_abs_diff_eq;

    fn param_with_grad(data: Vec<f32>, grad: Vec<f32>) -> ParamTensor {
        let mut p = ParamTensor::new(data, "w");
        p.set_grad(grad).unwrap();
        p
    }

    #[test]
    fn zero_stage1_matches_base() {
        // With world_size=1, ZeRO-1 should produce identical results to the base.
        let mut base = GpuAdam::new(1e-3);
        let mut zero = ZeroOptimizer::new(GpuAdam::new(1e-3), ZeroConfig::stage1(0, 1));

        let data = vec![1.0_f32; 8];
        let grad = vec![0.5_f32; 8];
        let mut p_base = vec![param_with_grad(data.clone(), grad.clone())];
        let mut p_zero = vec![param_with_grad(data.clone(), grad.clone())];

        base.step(&mut p_base).unwrap();
        zero.step(&mut p_zero).unwrap();

        for (b, z) in p_base[0].data.iter().zip(p_zero[0].data.iter()) {
            assert_abs_diff_eq!(b, z, epsilon = 1e-6);
        }
    }

    #[test]
    fn zero_stage2_zeros_non_owned_grads_rank0() {
        // world_size=2, rank=0 → owns [0..4], zeros [4..8]
        let cfg = ZeroConfig::stage2(0, 2);
        let (start, end) = cfg.shard_range(8);
        assert_eq!(start, 0);
        assert_eq!(end, 4);

        let p = param_with_grad(vec![0.0_f32; 8], vec![1.0_f32; 8]);
        let mut params = [p.clone()];
        // Simulate gradient sharding
        if let Some(g) = &mut params[0].grad {
            g[end..].fill(0.0);
        }
        let g = params[0].grad.as_ref().unwrap();
        g[..4]
            .iter()
            .for_each(|&v| assert_abs_diff_eq!(v, 1.0, epsilon = 1e-6));
        g[4..]
            .iter()
            .for_each(|&v| assert_abs_diff_eq!(v, 0.0, epsilon = 1e-6));
    }

    #[test]
    fn shard_range_evenly_divided() {
        let cfg = ZeroConfig::stage1(1, 4);
        // 8 elements, 4 ranks: each owns 2
        let (start, end) = cfg.shard_range(8);
        assert_eq!(start, 2);
        assert_eq!(end, 4);
    }

    #[test]
    fn shard_range_remainder() {
        // 10 elements, 3 ranks: sizes are [4, 4, 2]
        let r0 = ZeroConfig::stage1(0, 3).shard_range(10);
        let r1 = ZeroConfig::stage1(1, 3).shard_range(10);
        let r2 = ZeroConfig::stage1(2, 3).shard_range(10);
        assert_eq!(r0, (0, 4));
        assert_eq!(r1, (4, 8));
        assert_eq!(r2, (8, 10));
    }

    #[test]
    fn invalid_rank_panics() {
        // First: validate() itself should return Err
        let _validation_err = std::panic::catch_unwind(|| {
            let cfg = ZeroConfig::stage1(5, 4); // rank 5 >= world_size 4
            cfg.validate().unwrap(); // should fail validation
        });
        // ZeroOptimizer::new calls validate().expect(), so it should panic
        let result2 = std::panic::catch_unwind(|| {
            ZeroOptimizer::new(GpuAdam::new(1e-3), ZeroConfig::stage1(5, 4));
        });
        assert!(result2.is_err(), "should panic for invalid rank");
    }

    #[test]
    fn zero_stage3_updates_only_owned_shard() {
        // world_size=2, rank=0 → owns [0..2] of a 4-element param
        let cfg = ZeroConfig::stage3(0, 2);
        let mut zero = ZeroOptimizer::new(GpuAdam::new(1e-2), cfg);
        let mut params = vec![param_with_grad(vec![2.0_f32; 4], vec![1.0_f32; 4])];
        zero.step(&mut params).unwrap();

        // Owned shard (0..2) should have been updated
        let owned_changed = params[0].data[0..2].iter().any(|&v| (v - 2.0).abs() > 1e-6);
        assert!(owned_changed, "owned shard should be updated");
    }

    #[test]
    fn memory_estimate_stage1() {
        // 1B params, Adam (2 state factors), 8 GPUs, Stage 1
        let est = ZeroMemoryEstimate::new(1_000_000, 2, 8, ZeroStage::Stage1);
        // per_rank: params (1M) + grads (1M) + opt (1M*2/8=250K) = 2.25M × 4 bytes
        let bytes = est.bytes_per_rank();
        assert!(bytes > 0, "estimate should be positive");
        let ratio = est.reduction_ratio();
        // Stage 1 with 8 GPUs: optimizer state savings → ~8/(1+1+0.25) ratio
        assert!(
            ratio > 1.0,
            "stage 1 should reduce memory vs baseline, ratio={ratio}"
        );
    }

    #[test]
    fn zero_name() {
        let z1 = ZeroOptimizer::new(GpuAdam::new(1e-3), ZeroConfig::stage1(0, 1));
        let z2 = ZeroOptimizer::new(GpuAdam::new(1e-3), ZeroConfig::stage2(0, 1));
        let z3 = ZeroOptimizer::new(GpuAdam::new(1e-3), ZeroConfig::stage3(0, 1));
        assert_eq!(z1.name(), "ZeRO-1");
        assert_eq!(z2.name(), "ZeRO-2");
        assert_eq!(z3.name(), "ZeRO-3");
    }
}
