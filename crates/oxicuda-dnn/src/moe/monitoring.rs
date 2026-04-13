//! MoE load balancing monitoring.
//!
//! Provides runtime expert utilization tracking for Mixture-of-Experts layers.
//! GPU kernels compute per-expert token counts and an imbalance score
//! (coefficient of variation = std / mean) to detect routing skew.
//!
//! # Imbalance score interpretation
//!
//! | CV range   | Interpretation        |
//! |------------|-----------------------|
//! | < 0.1      | Balanced              |
//! | 0.1 – 0.3  | Moderate imbalance    |
//! | > 0.3      | Severe imbalance      |

use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Imbalance thresholds
// ---------------------------------------------------------------------------

/// CV threshold below which experts are considered balanced.
pub const BALANCED_THRESHOLD: f32 = 0.1;

/// CV threshold above which experts are considered severely imbalanced.
pub const SEVERE_IMBALANCE_THRESHOLD: f32 = 0.3;

// ---------------------------------------------------------------------------
// Imbalance classification
// ---------------------------------------------------------------------------

/// Classification of expert load balance quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImbalanceLevel {
    /// CV < 0.1 — experts receive roughly equal token counts.
    Balanced,
    /// 0.1 <= CV <= 0.3 — some skew but within acceptable bounds.
    Moderate,
    /// CV > 0.3 — significant routing imbalance, may hurt model quality.
    Severe,
}

impl ImbalanceLevel {
    /// Classifies an imbalance score (coefficient of variation) into a level.
    pub fn from_cv(cv: f32) -> Self {
        if cv < BALANCED_THRESHOLD {
            Self::Balanced
        } else if cv <= SEVERE_IMBALANCE_THRESHOLD {
            Self::Moderate
        } else {
            Self::Severe
        }
    }
}

// ---------------------------------------------------------------------------
// MoeUtilizationReport
// ---------------------------------------------------------------------------

/// Summary of expert utilization statistics for a single forward pass.
#[derive(Debug, Clone)]
pub struct MoeUtilizationReport {
    /// Token count per expert.
    pub per_expert_counts: Vec<u32>,
    /// Total tokens processed across all experts.
    pub total_tokens: u32,
    /// Coefficient of variation (std / mean) of per-expert counts.
    pub imbalance_score: f32,
    /// Index of the most loaded expert.
    pub most_loaded: u32,
    /// Index of the least loaded expert.
    pub least_loaded: u32,
}

impl MoeUtilizationReport {
    /// Creates a report from per-expert counts.
    ///
    /// Returns `Err` if `counts` is empty.
    pub fn from_counts(counts: &[u32]) -> DnnResult<Self> {
        if counts.is_empty() {
            return Err(DnnError::InvalidArgument(
                "per-expert counts must not be empty".into(),
            ));
        }

        let total: u64 = counts.iter().map(|&c| c as u64).sum();
        let n = counts.len() as f64;
        let mean = total as f64 / n;

        // Compute standard deviation
        let variance = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        let cv = if mean > 0.0 {
            (std_dev / mean) as f32
        } else {
            0.0
        };

        let (mut most_loaded_idx, mut most_loaded_val) = (0u32, counts[0]);
        let (mut least_loaded_idx, mut least_loaded_val) = (0u32, counts[0]);
        for (i, &c) in counts.iter().enumerate().skip(1) {
            if c > most_loaded_val {
                most_loaded_val = c;
                most_loaded_idx = i as u32;
            }
            if c < least_loaded_val {
                least_loaded_val = c;
                least_loaded_idx = i as u32;
            }
        }

        Ok(Self {
            per_expert_counts: counts.to_vec(),
            total_tokens: total.min(u32::MAX as u64) as u32,
            imbalance_score: cv,
            most_loaded: most_loaded_idx,
            least_loaded: least_loaded_idx,
        })
    }

    /// Returns the imbalance classification.
    pub fn imbalance_level(&self) -> ImbalanceLevel {
        ImbalanceLevel::from_cv(self.imbalance_score)
    }
}

// ---------------------------------------------------------------------------
// MoeMonitor
// ---------------------------------------------------------------------------

/// Runtime monitor for MoE expert utilization.
///
/// Generates GPU kernels to count per-expert token assignments and compute
/// imbalance metrics entirely on device.
#[derive(Debug, Clone)]
pub struct MoeMonitor {
    /// Number of experts to monitor.
    pub num_experts: u32,
    /// Target SM architecture for PTX generation.
    pub sm_version: SmVersion,
}

impl MoeMonitor {
    /// Creates a new monitor for the given expert count.
    pub fn new(num_experts: u32, sm_version: SmVersion) -> DnnResult<Self> {
        if num_experts == 0 {
            return Err(DnnError::InvalidArgument(
                "num_experts must be positive".into(),
            ));
        }
        Ok(Self {
            num_experts,
            sm_version,
        })
    }

    /// Generates PTX for the utilization counting kernel.
    ///
    /// Given per-token expert assignments, atomically counts tokens per expert.
    ///
    /// # Kernel parameters
    ///
    /// - `expert_assignments` (u64 ptr): Expert ID per token, length `num_tokens`.
    /// - `expert_counts` (u64 ptr): Output per-expert counts (must be zero-initialised),
    ///   length `num_experts`.
    /// - `num_tokens` (u32): Total number of tokens.
    /// - `num_experts` (u32): Number of experts (for bounds checking).
    pub fn generate_utilization_ptx(&self) -> DnnResult<String> {
        let kernel_name = "moe_utilization_count";

        let ptx = KernelBuilder::new(kernel_name)
            .target(self.sm_version)
            .param("expert_assignments", PtxType::U64)
            .param("expert_counts", PtxType::U64)
            .param("num_tokens", PtxType::U32)
            .param("num_experts", PtxType::U32)
            .body(|b| {
                let gid = b.global_thread_id_x();
                let n_tok = b.load_param_u32("num_tokens");

                let exit_lbl = b.fresh_label("exit");
                let p_exit = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_exit}, {gid}, {n_tok};"));
                b.branch_if(p_exit, &exit_lbl);

                let assign_ptr = b.load_param_u64("expert_assignments");
                let counts_ptr = b.load_param_u64("expert_counts");
                let n_exp = b.load_param_u32("num_experts");

                // Load expert_id = expert_assignments[gid]
                let assign_addr = b.byte_offset_addr(assign_ptr, gid, 4);
                let expert_id = b.load_global_u32(assign_addr);

                // Bounds check: expert_id < num_experts
                let in_bounds = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lt.u32 {in_bounds}, {expert_id}, {n_exp};"));
                let skip_lbl = b.fresh_label("skip");
                let p_oob = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {p_oob}, {in_bounds};"));
                b.branch_if(p_oob, &skip_lbl);

                // Atomic increment expert_counts[expert_id]
                let count_addr = b.byte_offset_addr(counts_ptr, expert_id, 4);
                let _old = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("atom.global.add.u32 {_old}, [{count_addr}], 1;"));

                b.label(&skip_lbl);
                b.label(&exit_lbl);
                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Generates PTX for the imbalance score kernel.
    ///
    /// Computes the coefficient of variation (CV = std / mean) of per-expert
    /// token counts in two passes:
    ///
    /// 1. Thread 0 computes the mean from `expert_counts`.
    /// 2. All threads (one per expert) compute `(count_i - mean)^2` and
    ///    atomically sum into a shared variance accumulator.
    /// 3. Thread 0 computes `sqrt(variance / N) / mean` and writes the result.
    ///
    /// # Kernel parameters
    ///
    /// - `expert_counts` (u64 ptr): Per-expert token counts, length `num_experts`.
    /// - `imbalance_out` (u64 ptr): Single-element output for the CV score.
    /// - `num_experts` (u32): Number of experts.
    /// - `total_tokens` (u32): Sum of all expert counts.
    pub fn generate_imbalance_score_ptx(&self) -> DnnResult<String> {
        let kernel_name = "moe_imbalance_score";

        let ptx = KernelBuilder::new(kernel_name)
            .target(self.sm_version)
            .param("expert_counts", PtxType::U64)
            .param("imbalance_out", PtxType::U64)
            .param("num_experts", PtxType::U32)
            .param("total_tokens", PtxType::U32)
            .body(|b| {
                let tid = b.thread_id_x();
                let n_exp = b.load_param_u32("num_experts");

                let exit_lbl = b.fresh_label("exit");
                let p_exit = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_exit}, {tid}, {n_exp};"));
                b.branch_if(p_exit, &exit_lbl);

                let counts_ptr = b.load_param_u64("expert_counts");
                let imb_ptr = b.load_param_u64("imbalance_out");
                let n_tok = b.load_param_u32("total_tokens");

                // Compute mean = total_tokens / num_experts
                let ntok_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {ntok_f}, {n_tok};"));
                let nexp_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {nexp_f}, {n_exp};"));
                let mean = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {mean}, {ntok_f}, {nexp_f};"));

                // Load count_i and compute (count_i - mean)^2
                let count_addr = b.byte_offset_addr(counts_ptr, tid.clone(), 4);
                let count_u32 = b.load_global_u32(count_addr);
                let count_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {count_f}, {count_u32};"));
                let diff = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("sub.rn.f32 {diff}, {count_f}, {mean};"));
                let diff_sq = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {diff_sq}, {diff}, {diff};"));

                // Warp-level reduction of diff_sq
                for offset in [16u32, 8, 4, 2, 1] {
                    let shfl_val = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!(
                        "shfl.sync.down.b32 {shfl_val}, {diff_sq}, {offset}, 31, 0xFFFFFFFF;"
                    ));
                    b.raw_ptx(&format!("add.rn.f32 {diff_sq}, {diff_sq}, {shfl_val};"));
                }

                // Lane 0 computes final CV and writes result
                let lane = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("and.b32 {lane}, {tid}, 31;"));
                let skip_lbl = b.fresh_label("skip");
                let p_lane0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ne.u32 {p_lane0}, {lane}, 0;"));
                b.branch_if(p_lane0, &skip_lbl);

                // variance = sum_diff_sq / num_experts
                let variance = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {variance}, {diff_sq}, {nexp_f};"));

                // std_dev = sqrt(variance)
                let std_dev = b.sqrt_rn_f32(variance);

                // cv = std_dev / mean (guard against zero mean)
                let p_zero_mean = b.alloc_reg(PtxType::Pred);
                let zero_bits = 0.0f32.to_bits();
                let zero_reg = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {zero_reg}, 0F{zero_bits:08X};"));
                b.raw_ptx(&format!("setp.eq.f32 {p_zero_mean}, {mean}, {zero_reg};"));

                let cv = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {cv}, {std_dev}, {mean};"));
                // If mean == 0, set cv = 0
                b.raw_ptx(&format!("selp.f32 {cv}, {zero_reg}, {cv}, {p_zero_mean};"));

                // Atomic add to imbalance_out (supports multi-warp case)
                let _old = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("atom.global.add.f32 {_old}, [{imb_ptr}], {cv};"));

                b.label(&skip_lbl);
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

    #[test]
    fn imbalance_level_balanced() {
        assert_eq!(ImbalanceLevel::from_cv(0.0), ImbalanceLevel::Balanced);
        assert_eq!(ImbalanceLevel::from_cv(0.05), ImbalanceLevel::Balanced);
        assert_eq!(ImbalanceLevel::from_cv(0.099), ImbalanceLevel::Balanced);
    }

    #[test]
    fn imbalance_level_moderate() {
        assert_eq!(ImbalanceLevel::from_cv(0.1), ImbalanceLevel::Moderate);
        assert_eq!(ImbalanceLevel::from_cv(0.2), ImbalanceLevel::Moderate);
        assert_eq!(ImbalanceLevel::from_cv(0.3), ImbalanceLevel::Moderate);
    }

    #[test]
    fn imbalance_level_severe() {
        assert_eq!(ImbalanceLevel::from_cv(0.31), ImbalanceLevel::Severe);
        assert_eq!(ImbalanceLevel::from_cv(1.0), ImbalanceLevel::Severe);
    }

    #[test]
    fn report_from_counts_uniform() {
        let counts = vec![100, 100, 100, 100];
        let report = MoeUtilizationReport::from_counts(&counts).expect("valid counts");
        assert_eq!(report.total_tokens, 400);
        assert!(report.imbalance_score < 1e-6);
        assert_eq!(report.imbalance_level(), ImbalanceLevel::Balanced);
    }

    #[test]
    fn report_from_counts_skewed() {
        let counts = vec![400, 0, 0, 0];
        let report = MoeUtilizationReport::from_counts(&counts).expect("valid counts");
        assert_eq!(report.total_tokens, 400);
        assert!(report.imbalance_score > SEVERE_IMBALANCE_THRESHOLD);
        assert_eq!(report.imbalance_level(), ImbalanceLevel::Severe);
        assert_eq!(report.most_loaded, 0);
    }

    #[test]
    fn report_from_counts_moderate() {
        // 8 experts with mild skew
        let counts = vec![110, 90, 105, 95, 115, 85, 100, 100];
        let report = MoeUtilizationReport::from_counts(&counts).expect("valid counts");
        assert_eq!(report.total_tokens, 800);
        // CV should be moderate
        assert!(report.imbalance_score > 0.0);
        assert!(report.imbalance_score < 0.5);
    }

    #[test]
    fn report_from_empty_counts() {
        let counts: Vec<u32> = vec![];
        assert!(MoeUtilizationReport::from_counts(&counts).is_err());
    }

    #[test]
    fn report_most_and_least_loaded() {
        let counts = vec![10, 50, 5, 30];
        let report = MoeUtilizationReport::from_counts(&counts).expect("valid counts");
        assert_eq!(report.most_loaded, 1);
        assert_eq!(report.least_loaded, 2);
    }

    #[test]
    fn monitor_creation_ok() {
        let monitor = MoeMonitor::new(8, SmVersion::Sm80);
        assert!(monitor.is_ok());
    }

    #[test]
    fn monitor_creation_zero_experts() {
        let monitor = MoeMonitor::new(0, SmVersion::Sm80);
        assert!(monitor.is_err());
    }

    #[test]
    fn utilization_ptx_generates() {
        let monitor = MoeMonitor::new(8, SmVersion::Sm80).expect("valid monitor");
        let ptx = monitor.generate_utilization_ptx();
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains(".entry moe_utilization_count"));
        assert!(text.contains("atom.global.add.u32"));
    }

    #[test]
    fn imbalance_score_ptx_generates() {
        let monitor = MoeMonitor::new(8, SmVersion::Sm80).expect("valid monitor");
        let ptx = monitor.generate_imbalance_score_ptx();
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains(".entry moe_imbalance_score"));
        assert!(text.contains("sqrt.rn.f32"));
    }

    #[test]
    fn utilization_ptx_bounds_check() {
        let monitor = MoeMonitor::new(16, SmVersion::Sm80).expect("valid monitor");
        let text = monitor.generate_utilization_ptx().unwrap_or_default();
        // Should contain bounds check for expert_id < num_experts
        assert!(text.contains("setp.lt.u32"));
    }

    #[test]
    fn imbalance_score_ptx_contains_shuffle() {
        let monitor = MoeMonitor::new(8, SmVersion::Sm80).expect("valid monitor");
        let text = monitor.generate_imbalance_score_ptx().unwrap_or_default();
        assert!(text.contains("shfl.sync.down"));
    }

    #[test]
    fn report_all_zeros() {
        let counts = vec![0, 0, 0, 0];
        let report = MoeUtilizationReport::from_counts(&counts).expect("valid counts");
        assert_eq!(report.total_tokens, 0);
        // CV should be 0 when mean is 0
        assert!(report.imbalance_score.abs() < 1e-6);
    }

    #[test]
    fn utilization_ptx_different_expert_counts() {
        for n in [2, 4, 16, 32, 64] {
            let monitor = MoeMonitor::new(n, SmVersion::Sm80).expect("valid monitor");
            assert!(monitor.generate_utilization_ptx().is_ok());
        }
    }
}
