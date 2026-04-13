//! MoE auxiliary loss computation.
//!
//! Implements Switch Transformer style load-balancing losses to encourage
//! uniform expert usage across the Mixture-of-Experts layer.
//!
//! Two loss types are provided:
//!
//! - **Load balance loss** — `alpha * num_experts * sum(f_i * P_i)` where
//!   `f_i` is the fraction of tokens routed to expert `i` and `P_i` is the
//!   mean routing probability for expert `i`.
//!
//! - **Z-loss** — `(1/n) * sum(log(sum(exp(router_logits)))^2)` which
//!   penalises large logit magnitudes to stabilise training.
//!
//! # Reference
//!
//! Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
//! with Simple and Efficient Sparsity", 2021.

use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for MoE auxiliary loss computation.
#[derive(Debug, Clone)]
pub struct AuxLossConfig {
    /// Number of expert networks in the MoE layer.
    pub num_experts: u32,
    /// Total number of tokens in the batch.
    pub num_tokens: u32,
    /// Loss weight (typically 0.01 for Switch Transformer).
    pub alpha: f32,
    /// Target SM architecture for PTX generation.
    pub sm_version: SmVersion,
}

impl AuxLossConfig {
    /// Validates that the configuration parameters are consistent.
    pub fn validate(&self) -> DnnResult<()> {
        if self.num_experts == 0 {
            return Err(DnnError::InvalidArgument(
                "num_experts must be positive".into(),
            ));
        }
        if self.num_tokens == 0 {
            return Err(DnnError::InvalidArgument(
                "num_tokens must be positive".into(),
            ));
        }
        if self.alpha < 0.0 {
            return Err(DnnError::InvalidArgument(
                "alpha must be non-negative".into(),
            ));
        }
        if self.alpha.is_nan() {
            return Err(DnnError::InvalidArgument("alpha must not be NaN".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AuxLossPlan
// ---------------------------------------------------------------------------

/// Execution plan for MoE auxiliary loss kernels.
///
/// Encapsulates validated configuration and provides PTX generation for both
/// load-balance loss and z-loss kernels.
#[derive(Debug, Clone)]
pub struct AuxLossPlan {
    config: AuxLossConfig,
}

impl AuxLossPlan {
    /// Creates a new auxiliary loss plan, validating the configuration.
    pub fn new(config: AuxLossConfig) -> DnnResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Returns a reference to the underlying configuration.
    pub fn config(&self) -> &AuxLossConfig {
        &self.config
    }

    /// Generates PTX for the load-balance auxiliary loss kernel.
    ///
    /// The kernel computes:
    ///
    /// 1. Per-expert fraction: `f_i = count_i / num_tokens`
    /// 2. Per-expert mean probability: `P_i = sum(router_prob_i) / num_tokens`
    /// 3. Loss = `alpha * num_experts * sum(f_i * P_i)`
    ///
    /// # Kernel parameters
    ///
    /// - `expert_counts` (u64 ptr): Per-expert token counts, length `num_experts`.
    /// - `expert_probs` (u64 ptr): Per-expert summed routing probabilities, length `num_experts`.
    /// - `loss_out` (u64 ptr): Single-element output for the scalar loss.
    /// - `num_experts` (u32): Number of experts.
    /// - `num_tokens` (u32): Total tokens in batch.
    ///
    /// The kernel uses a single thread block with `num_experts` threads (capped
    /// at 1024). Each thread computes `f_i * P_i` for one expert, then a
    /// shared-memory reduction sums the partial products. Thread 0 writes the
    /// final scaled loss.
    pub fn generate_load_balance_loss_ptx(&self) -> DnnResult<String> {
        let alpha = self.config.alpha;
        let alpha_bits = alpha.to_bits();

        let kernel_name = "moe_load_balance_loss";

        let ptx = KernelBuilder::new(kernel_name)
            .target(self.config.sm_version)
            .param("expert_counts", PtxType::U64)
            .param("expert_probs", PtxType::U64)
            .param("loss_out", PtxType::U64)
            .param("num_experts", PtxType::U32)
            .param("num_tokens", PtxType::U32)
            .body(move |b| {
                let tid = b.thread_id_x();
                let n_exp = b.load_param_u32("num_experts");
                let n_tok = b.load_param_u32("num_tokens");

                // Early exit if tid >= num_experts
                let exit_lbl = b.fresh_label("exit");
                let p_exit = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_exit}, {tid}, {n_exp};"));
                b.branch_if(p_exit, &exit_lbl);

                // Load pointers
                let counts_ptr = b.load_param_u64("expert_counts");
                let probs_ptr = b.load_param_u64("expert_probs");
                let loss_ptr = b.load_param_u64("loss_out");

                // Load count_i (u32) and convert to float
                let count_addr = b.byte_offset_addr(counts_ptr, tid.clone(), 4);
                let count_u32 = b.load_global_u32(count_addr);
                let count_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {count_f}, {count_u32};"));

                // Convert num_tokens to float
                let ntok_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {ntok_f}, {n_tok};"));

                // f_i = count_i / num_tokens
                let f_i = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {f_i}, {count_f}, {ntok_f};"));

                // Load summed probability for expert i
                let prob_addr = b.byte_offset_addr(probs_ptr, tid.clone(), 4);
                let prob_sum = b.load_global_f32(prob_addr);

                // P_i = prob_sum / num_tokens
                let p_i = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {p_i}, {prob_sum}, {ntok_f};"));

                // partial = f_i * P_i
                let partial = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {partial}, {f_i}, {p_i};"));

                // Warp-level shuffle reduction for partial sums
                // Use shared memory for cross-warp reduction
                // First: intra-warp reduction
                for offset in [16u32, 8, 4, 2, 1] {
                    let shfl_val = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!(
                        "shfl.sync.down.b32 {shfl_val}, {partial}, {offset}, 31, 0xFFFFFFFF;"
                    ));
                    b.raw_ptx(&format!("add.rn.f32 {partial}, {partial}, {shfl_val};"));
                }

                // Lane 0 of each warp does atomic add to loss_out
                let lane = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("and.b32 {lane}, {tid}, 31;"));

                let skip_write = b.fresh_label("skip_write");
                let p_lane0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ne.u32 {p_lane0}, {lane}, 0;"));
                b.branch_if(p_lane0, &skip_write);

                // Scale: loss_contribution = alpha * num_experts * partial
                let alpha_reg = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {alpha_reg}, 0F{alpha_bits:08X};"));

                let nexp_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {nexp_f}, {n_exp};"));

                let scaled = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {scaled}, {alpha_reg}, {nexp_f};"));
                b.raw_ptx(&format!("mul.rn.f32 {scaled}, {scaled}, {partial};"));

                // Atomic add to loss_out[0]
                let _old = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!(
                    "atom.global.add.f32 {_old}, [{loss_ptr}], {scaled};"
                ));

                b.label(&skip_write);
                b.label(&exit_lbl);
                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Generates PTX for the z-loss (router logit stabilisation) kernel.
    ///
    /// Computes: `z_loss = (1/n) * sum_i(log(sum_j(exp(logit_ij)))^2)`
    ///
    /// Each thread processes one token row of router logits. The thread
    /// iterates over experts, computes the log-sum-exp of the router logits,
    /// squares it, and atomically adds to a global accumulator.
    ///
    /// # Kernel parameters
    ///
    /// - `router_logits` (u64 ptr): Logit tensor of shape `[num_tokens, num_experts]`.
    /// - `loss_out` (u64 ptr): Single-element output for the scalar z-loss.
    /// - `num_tokens` (u32): Number of tokens (rows).
    /// - `num_experts` (u32): Number of experts (columns).
    pub fn generate_z_loss_ptx(&self) -> DnnResult<String> {
        let kernel_name = "moe_z_loss";

        let ptx = KernelBuilder::new(kernel_name)
            .target(self.config.sm_version)
            .param("router_logits", PtxType::U64)
            .param("loss_out", PtxType::U64)
            .param("num_tokens", PtxType::U32)
            .param("num_experts", PtxType::U32)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_tok = b.load_param_u32("num_tokens");

                let exit_lbl = b.fresh_label("exit");
                let p_exit = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_exit}, {gid}, {n_tok};"));
                b.branch_if(p_exit, &exit_lbl);

                let logits_ptr = b.load_param_u64("router_logits");
                let loss_ptr = b.load_param_u64("loss_out");
                let n_exp = b.load_param_u32("num_experts");

                // Row base = logits_ptr + gid * num_experts * sizeof(f32)
                let row_stride = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {row_stride}, {n_exp}, 4;"));
                let row_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {row_offset}, {gid}, {row_stride};"));
                let row_offset_64 = b.cvt_u32_to_u64(row_offset);
                let logits_base = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!(
                    "add.u64 {logits_base}, {logits_ptr}, {row_offset_64};"
                ));

                // Pass 1: find max logit for numerical stability
                let max_val = b.alloc_reg(PtxType::F32);
                let neg_inf_bits = f32::NEG_INFINITY.to_bits();
                b.raw_ptx(&format!("mov.b32 {max_val}, 0F{neg_inf_bits:08X};"));

                let e_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {e_reg}, 0;"));
                let lp1_head = b.fresh_label("lp1");
                let lp1_end = b.fresh_label("lp1e");
                b.label(&lp1_head);
                let p_lp1 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_lp1}, {e_reg}, {n_exp};"));
                b.branch_if(p_lp1, &lp1_end);

                let addr1 = b.byte_offset_addr(logits_base.clone(), e_reg.clone(), 4);
                let val1 = b.load_global_f32(addr1);
                b.raw_ptx(&format!("max.f32 {max_val}, {max_val}, {val1};"));

                b.raw_ptx(&format!("add.u32 {e_reg}, {e_reg}, 1;"));
                b.branch(&lp1_head);
                b.label(&lp1_end);

                // Pass 2: sum(exp(logit - max))
                let sum_exp = b.alloc_reg(PtxType::F32);
                let zero_bits = 0.0f32.to_bits();
                b.raw_ptx(&format!("mov.b32 {sum_exp}, 0F{zero_bits:08X};"));

                b.raw_ptx(&format!("mov.u32 {e_reg}, 0;"));
                let lp2_head = b.fresh_label("lp2");
                let lp2_end = b.fresh_label("lp2e");
                b.label(&lp2_head);
                let p_lp2 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_lp2}, {e_reg}, {n_exp};"));
                b.branch_if(p_lp2, &lp2_end);

                let addr2 = b.byte_offset_addr(logits_base.clone(), e_reg.clone(), 4);
                let val2 = b.load_global_f32(addr2);
                // shifted = logit - max
                let shifted = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("sub.rn.f32 {shifted}, {val2}, {max_val};"));
                // exp(shifted) via ex2.approx(shifted * log2(e))
                // log2(e) ≈ 1.4426950408889634
                let log2e_bits = std::f32::consts::LOG2_E.to_bits();
                let scaled_val = b.alloc_reg(PtxType::F32);
                let log2e_reg = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {log2e_reg}, 0F{log2e_bits:08X};"));
                b.raw_ptx(&format!("mul.rn.f32 {scaled_val}, {shifted}, {log2e_reg};"));
                let exp_val = b.ex2_approx_f32(scaled_val);

                b.raw_ptx(&format!("add.rn.f32 {sum_exp}, {sum_exp}, {exp_val};"));

                b.raw_ptx(&format!("add.u32 {e_reg}, {e_reg}, 1;"));
                b.branch(&lp2_head);
                b.label(&lp2_end);

                // log_sum_exp = log(sum_exp) + max = lg2(sum_exp) / lg2(e) + max
                // log(sum_exp) = lg2(sum_exp) / log2(e)
                let lg2_sum = b.lg2_approx_f32(sum_exp);
                let ln_sum = b.alloc_reg(PtxType::F32);
                let inv_log2e_bits = (1.0f32 / std::f32::consts::LOG2_E).to_bits();
                let inv_log2e = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {inv_log2e}, 0F{inv_log2e_bits:08X};"));
                b.raw_ptx(&format!("mul.rn.f32 {ln_sum}, {lg2_sum}, {inv_log2e};"));
                let lse = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("add.rn.f32 {lse}, {ln_sum}, {max_val};"));

                // z_contribution = lse^2
                let z_sq = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {z_sq}, {lse}, {lse};"));

                // Divide by num_tokens: contribution = z_sq / n
                let ntok_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {ntok_f}, {n_tok};"));
                let contribution = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {contribution}, {z_sq}, {ntok_f};"));

                // Atomic add to loss_out
                let _old = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!(
                    "atom.global.add.f32 {_old}, [{loss_ptr}], {contribution};"
                ));

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

    fn default_config() -> AuxLossConfig {
        AuxLossConfig {
            num_experts: 8,
            num_tokens: 1024,
            alpha: 0.01,
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
        cfg.num_tokens = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_negative_alpha() {
        let mut cfg = default_config();
        cfg.alpha = -0.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_nan_alpha() {
        let mut cfg = default_config();
        cfg.alpha = f32::NAN;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn plan_creation_ok() {
        let plan = AuxLossPlan::new(default_config());
        assert!(plan.is_ok());
    }

    #[test]
    fn plan_creation_invalid() {
        let mut cfg = default_config();
        cfg.num_experts = 0;
        assert!(AuxLossPlan::new(cfg).is_err());
    }

    #[test]
    fn load_balance_loss_ptx_generates() {
        let plan = AuxLossPlan::new(default_config()).expect("valid config");
        let ptx = plan.generate_load_balance_loss_ptx();
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains(".entry moe_load_balance_loss"));
        assert!(text.contains("atom.global.add.f32"));
    }

    #[test]
    fn z_loss_ptx_generates() {
        let plan = AuxLossPlan::new(default_config()).expect("valid config");
        let ptx = plan.generate_z_loss_ptx();
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains(".entry moe_z_loss"));
        assert!(text.contains("ex2.approx"));
    }

    #[test]
    fn load_balance_loss_ptx_contains_div() {
        let plan = AuxLossPlan::new(default_config()).expect("valid config");
        let text = plan.generate_load_balance_loss_ptx().unwrap_or_default();
        assert!(text.contains("div.rn.f32"));
    }

    #[test]
    fn z_loss_ptx_contains_log_sum_exp() {
        let plan = AuxLossPlan::new(default_config()).expect("valid config");
        let text = plan.generate_z_loss_ptx().unwrap_or_default();
        assert!(text.contains("lg2.approx"));
    }

    #[test]
    fn plan_config_accessor() {
        let plan = AuxLossPlan::new(default_config()).expect("valid config");
        assert_eq!(plan.config().num_experts, 8);
        assert_eq!(plan.config().num_tokens, 1024);
    }

    #[test]
    fn load_balance_loss_different_expert_counts() {
        for num_experts in [2, 4, 16, 32, 64] {
            let cfg = AuxLossConfig {
                num_experts,
                num_tokens: 512,
                alpha: 0.01,
                sm_version: SmVersion::Sm80,
            };
            let plan = AuxLossPlan::new(cfg).expect("valid config");
            assert!(plan.generate_load_balance_loss_ptx().is_ok());
        }
    }

    #[test]
    fn z_loss_different_expert_counts() {
        for num_experts in [2, 4, 16, 32, 64] {
            let cfg = AuxLossConfig {
                num_experts,
                num_tokens: 256,
                alpha: 0.01,
                sm_version: SmVersion::Sm80,
            };
            let plan = AuxLossPlan::new(cfg).expect("valid config");
            assert!(plan.generate_z_loss_ptx().is_ok());
        }
    }

    #[test]
    fn config_validate_zero_alpha_ok() {
        let mut cfg = default_config();
        cfg.alpha = 0.0;
        assert!(cfg.validate().is_ok());
    }
}
