//! Expert capacity factor tuning.
//!
//! Implements dynamic capacity adjustment for Mixture-of-Experts layers.
//! The capacity factor controls how many tokens each expert can accept:
//! `capacity = ceil(tokens_per_batch / num_experts * capacity_factor)`.
//!
//! When the number of tokens routed to an expert exceeds its capacity,
//! overflow tokens are dropped (not processed). This module provides:
//!
//! - Static capacity computation from configuration.
//! - An overflow-mask PTX kernel that marks dropped tokens.
//! - A dynamic capacity PTX kernel that adjusts capacity based on observed
//!   routing distributions at runtime.
//!
//! # Reference
//!
//! Lepikhin et al., "GShard: Scaling Giant Models with Conditional
//! Computation and Automatic Sharding", 2020.

use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for expert capacity management.
#[derive(Debug, Clone)]
pub struct CapacityConfig {
    /// Number of expert networks in the MoE layer.
    pub num_experts: u32,
    /// Capacity multiplier (default 1.25). Must be >= 1.0.
    pub capacity_factor: f32,
    /// Minimum allowed capacity per expert (floor).
    pub min_capacity: u32,
    /// Maximum allowed capacity per expert (ceiling, 0 = no limit).
    pub max_capacity: u32,
    /// Total tokens in a single batch.
    pub tokens_per_batch: u32,
    /// Target SM architecture for PTX generation.
    pub sm_version: SmVersion,
}

impl CapacityConfig {
    /// Validates that the configuration parameters are consistent.
    pub fn validate(&self) -> DnnResult<()> {
        if self.num_experts == 0 {
            return Err(DnnError::InvalidArgument(
                "num_experts must be positive".into(),
            ));
        }
        if self.tokens_per_batch == 0 {
            return Err(DnnError::InvalidArgument(
                "tokens_per_batch must be positive".into(),
            ));
        }
        if self.capacity_factor < 1.0 {
            return Err(DnnError::InvalidArgument(
                "capacity_factor must be >= 1.0".into(),
            ));
        }
        if self.capacity_factor.is_nan() || self.capacity_factor.is_infinite() {
            return Err(DnnError::InvalidArgument(
                "capacity_factor must be finite".into(),
            ));
        }
        if self.max_capacity != 0 && self.max_capacity < self.min_capacity {
            return Err(DnnError::InvalidArgument(format!(
                "max_capacity ({}) must be >= min_capacity ({})",
                self.max_capacity, self.min_capacity
            )));
        }
        Ok(())
    }
}

impl Default for CapacityConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            capacity_factor: 1.25,
            min_capacity: 4,
            max_capacity: 0,
            tokens_per_batch: 1024,
            sm_version: SmVersion::Sm80,
        }
    }
}

// ---------------------------------------------------------------------------
// CapacityPlan
// ---------------------------------------------------------------------------

/// Execution plan for expert capacity management kernels.
///
/// Provides both static capacity computation and PTX kernels for
/// runtime overflow masking and dynamic capacity adjustment.
#[derive(Debug, Clone)]
pub struct CapacityPlan {
    config: CapacityConfig,
}

impl CapacityPlan {
    /// Creates a new capacity plan, validating the configuration.
    pub fn new(config: CapacityConfig) -> DnnResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Returns a reference to the underlying configuration.
    pub fn config(&self) -> &CapacityConfig {
        &self.config
    }

    /// Computes the static expert capacity.
    ///
    /// `capacity = clamp(ceil(tokens_per_batch / num_experts * capacity_factor), min, max)`
    pub fn expert_capacity(&self) -> u32 {
        let base = self.config.tokens_per_batch as f64 / self.config.num_experts as f64;
        let raw = (base * self.config.capacity_factor as f64).ceil() as u32;
        let clamped = raw.max(self.config.min_capacity);
        if self.config.max_capacity > 0 {
            clamped.min(self.config.max_capacity)
        } else {
            clamped
        }
    }

    /// Generates PTX for the overflow mask kernel.
    ///
    /// Given per-token expert assignments and a running per-expert count,
    /// this kernel marks tokens that exceed the expert capacity as dropped
    /// by setting `overflow_mask[token_idx] = 1`.
    ///
    /// # Kernel parameters
    ///
    /// - `expert_assignments` (u64 ptr): Per-token expert ID, length `num_tokens`.
    /// - `expert_counts` (u64 ptr): Per-expert running token count (atomically updated).
    /// - `overflow_mask` (u64 ptr): Output mask, 1 = dropped, length `num_tokens`.
    /// - `capacity` (u32): Maximum tokens per expert.
    /// - `num_tokens` (u32): Total tokens.
    pub fn generate_overflow_mask_ptx(&self) -> DnnResult<String> {
        let kernel_name = "moe_overflow_mask";

        let ptx = KernelBuilder::new(kernel_name)
            .target(self.config.sm_version)
            .param("expert_assignments", PtxType::U64)
            .param("expert_counts", PtxType::U64)
            .param("overflow_mask", PtxType::U64)
            .param("capacity", PtxType::U32)
            .param("num_tokens", PtxType::U32)
            .body(|b| {
                let gid = b.global_thread_id_x();
                let n_tok = b.load_param_u32("num_tokens");

                let exit_lbl = b.fresh_label("exit");
                let p_exit = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_exit}, {gid}, {n_tok};"));
                b.branch_if(p_exit, &exit_lbl);

                let assign_ptr = b.load_param_u64("expert_assignments");
                let counts_ptr = b.load_param_u64("expert_counts");
                let mask_ptr = b.load_param_u64("overflow_mask");
                let cap = b.load_param_u32("capacity");

                // Load expert_id = expert_assignments[gid]
                let assign_addr = b.byte_offset_addr(assign_ptr, gid.clone(), 4);
                let expert_id = b.load_global_u32(assign_addr);

                // Atomically increment expert_counts[expert_id]
                let count_addr = b.byte_offset_addr(counts_ptr, expert_id, 4);
                let old_count = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "atom.global.add.u32 {old_count}, [{count_addr}], 1;"
                ));

                // If old_count >= capacity, mark as overflow
                let is_overflow = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {is_overflow}, {old_count}, {cap};"));

                let mask_val = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("selp.u32 {mask_val}, 1, 0, {is_overflow};"));

                let mask_addr = b.byte_offset_addr(mask_ptr, gid, 4);
                b.store_global_u32(mask_addr, mask_val);

                b.label(&exit_lbl);
                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Generates PTX for the dynamic capacity adjustment kernel.
    ///
    /// Given observed routing counts from a previous forward pass, this kernel
    /// computes adjusted per-expert capacities based on actual usage patterns.
    /// Experts receiving more tokens get higher capacity; under-used experts
    /// donate capacity to busier ones while maintaining the global token budget.
    ///
    /// The adjustment formula per expert:
    /// `new_cap_i = clamp(base_cap * (observed_i / expected), min_cap, max_cap)`
    /// where `expected = tokens / num_experts`.
    ///
    /// # Kernel parameters
    ///
    /// - `observed_counts` (u64 ptr): Actual tokens routed per expert (from previous batch).
    /// - `adjusted_capacities` (u64 ptr): Output per-expert capacities, length `num_experts`.
    /// - `num_experts` (u32): Number of experts.
    /// - `base_capacity` (u32): Static base capacity (from `expert_capacity()`).
    /// - `min_capacity` (u32): Minimum capacity floor.
    /// - `max_capacity` (u32): Maximum capacity ceiling (0 = no limit).
    /// - `tokens_per_batch` (u32): Total tokens for computing expected count.
    pub fn generate_dynamic_capacity_ptx(&self) -> DnnResult<String> {
        let kernel_name = "moe_dynamic_capacity";

        let ptx = KernelBuilder::new(kernel_name)
            .target(self.config.sm_version)
            .param("observed_counts", PtxType::U64)
            .param("adjusted_capacities", PtxType::U64)
            .param("num_experts", PtxType::U32)
            .param("base_capacity", PtxType::U32)
            .param("min_capacity", PtxType::U32)
            .param("max_capacity", PtxType::U32)
            .param("tokens_per_batch", PtxType::U32)
            .body(|b| {
                let gid = b.global_thread_id_x();
                let n_exp = b.load_param_u32("num_experts");

                let exit_lbl = b.fresh_label("exit");
                let p_exit = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_exit}, {gid}, {n_exp};"));
                b.branch_if(p_exit, &exit_lbl);

                let obs_ptr = b.load_param_u64("observed_counts");
                let adj_ptr = b.load_param_u64("adjusted_capacities");
                let base_cap = b.load_param_u32("base_capacity");
                let min_cap = b.load_param_u32("min_capacity");
                let max_cap = b.load_param_u32("max_capacity");
                let n_tok = b.load_param_u32("tokens_per_batch");

                // Load observed count for this expert
                let obs_addr = b.byte_offset_addr(obs_ptr, gid.clone(), 4);
                let obs_u32 = b.load_global_u32(obs_addr);

                // expected = tokens_per_batch / num_experts (float)
                let ntok_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {ntok_f}, {n_tok};"));
                let nexp_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {nexp_f}, {n_exp};"));
                let expected_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {expected_f}, {ntok_f}, {nexp_f};"));

                // ratio = observed / expected
                let obs_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {obs_f}, {obs_u32};"));
                let ratio = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {ratio}, {obs_f}, {expected_f};"));

                // new_cap_f = base_capacity * ratio
                let base_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {base_f}, {base_cap};"));
                let new_cap_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {new_cap_f}, {base_f}, {ratio};"));

                // Round up (ceil)
                let new_cap_ceil = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rpi.f32.f32 {new_cap_ceil}, {new_cap_f};"));

                // Convert to u32
                let new_cap_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("cvt.rzi.u32.f32 {new_cap_u32}, {new_cap_ceil};"));

                // Clamp: max(min_cap, min(new_cap, max_cap))
                let clamped = b.max_u32(new_cap_u32, min_cap);

                // Only apply max if max_capacity > 0
                let has_max = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.gt.u32 {has_max}, {max_cap}, 0;"));
                let capped = b.min_u32(clamped.clone(), max_cap);
                let final_cap = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "selp.u32 {final_cap}, {capped}, {clamped}, {has_max};"
                ));

                // Store result
                let adj_addr = b.byte_offset_addr(adj_ptr, gid, 4);
                b.store_global_u32(adj_addr, final_cap);

                b.label(&exit_lbl);
                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CapacityConfig {
        CapacityConfig {
            num_experts: 8,
            capacity_factor: 1.25,
            min_capacity: 4,
            max_capacity: 0,
            tokens_per_batch: 1024,
            sm_version: SmVersion::Sm80,
        }
    }

    #[test]
    fn config_validate_ok() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn config_validate_zero_experts() {
        let mut cfg = default_config();
        cfg.num_experts = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_tokens() {
        let mut cfg = default_config();
        cfg.tokens_per_batch = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_low_capacity_factor() {
        let mut cfg = default_config();
        cfg.capacity_factor = 0.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_nan_capacity_factor() {
        let mut cfg = default_config();
        cfg.capacity_factor = f32::NAN;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_inf_capacity_factor() {
        let mut cfg = default_config();
        cfg.capacity_factor = f32::INFINITY;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_max_lt_min() {
        let mut cfg = default_config();
        cfg.min_capacity = 10;
        cfg.max_capacity = 5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_default_values() {
        let cfg = CapacityConfig::default();
        assert_eq!(cfg.num_experts, 8);
        assert!((cfg.capacity_factor - 1.25).abs() < 1e-6);
        assert_eq!(cfg.min_capacity, 4);
        assert_eq!(cfg.max_capacity, 0);
        assert_eq!(cfg.tokens_per_batch, 1024);
    }

    #[test]
    fn expert_capacity_basic() {
        // capacity = ceil(1024 / 8 * 1.25) = ceil(160.0) = 160
        let plan = CapacityPlan::new(default_config()).expect("valid config");
        assert_eq!(plan.expert_capacity(), 160);
    }

    #[test]
    fn expert_capacity_with_min() {
        let cfg = CapacityConfig {
            num_experts: 1024,
            capacity_factor: 1.0,
            min_capacity: 4,
            max_capacity: 0,
            tokens_per_batch: 8,
            sm_version: SmVersion::Sm80,
        };
        // base = 8/1024 = 0.0078125, raw = ceil(0.0078125) = 1, clamped to min=4
        let plan = CapacityPlan::new(cfg).expect("valid config");
        assert_eq!(plan.expert_capacity(), 4);
    }

    #[test]
    fn expert_capacity_with_max() {
        let cfg = CapacityConfig {
            num_experts: 2,
            capacity_factor: 2.0,
            min_capacity: 1,
            max_capacity: 100,
            tokens_per_batch: 1024,
            sm_version: SmVersion::Sm80,
        };
        // base = 512, raw = ceil(1024.0) = 1024, clamped to max=100
        let plan = CapacityPlan::new(cfg).expect("valid config");
        assert_eq!(plan.expert_capacity(), 100);
    }

    #[test]
    fn overflow_mask_ptx_generates() {
        let plan = CapacityPlan::new(default_config()).expect("valid config");
        let ptx = plan.generate_overflow_mask_ptx();
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains(".entry moe_overflow_mask"));
        assert!(text.contains("atom.global.add.u32"));
    }

    #[test]
    fn dynamic_capacity_ptx_generates() {
        let plan = CapacityPlan::new(default_config()).expect("valid config");
        let ptx = plan.generate_dynamic_capacity_ptx();
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains(".entry moe_dynamic_capacity"));
        assert!(text.contains("div.rn.f32"));
    }

    #[test]
    fn overflow_mask_ptx_contains_selp() {
        let plan = CapacityPlan::new(default_config()).expect("valid config");
        let text = plan.generate_overflow_mask_ptx().unwrap_or_default();
        assert!(text.contains("selp.u32"));
    }

    #[test]
    fn dynamic_capacity_ptx_contains_clamp() {
        let plan = CapacityPlan::new(default_config()).expect("valid config");
        let text = plan.generate_dynamic_capacity_ptx().unwrap_or_default();
        // Should contain min/max clamping
        assert!(text.contains("max.u32") || text.contains("min.u32"));
    }

    #[test]
    fn plan_config_accessor() {
        let plan = CapacityPlan::new(default_config()).expect("valid config");
        assert_eq!(plan.config().num_experts, 8);
    }

    #[test]
    fn expert_capacity_various_factors() {
        for factor in [1.0f32, 1.25, 1.5, 2.0] {
            let cfg = CapacityConfig {
                num_experts: 8,
                capacity_factor: factor,
                min_capacity: 1,
                max_capacity: 0,
                tokens_per_batch: 1024,
                sm_version: SmVersion::Sm80,
            };
            let plan = CapacityPlan::new(cfg).expect("valid config");
            let cap = plan.expert_capacity();
            let expected = (1024.0 / 8.0 * factor as f64).ceil() as u32;
            assert_eq!(cap, expected, "factor={factor}");
        }
    }
}
