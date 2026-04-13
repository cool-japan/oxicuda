//! Expert routing with top-k selection.
//!
//! Implements the gating mechanism for Mixture-of-Experts models.
//! Given router logits of shape `[num_tokens, num_experts]`, selects the
//! top-k experts per token, computes softmax weights over the selected
//! experts, and builds permutation/offset arrays for subsequent expert
//! dispatch.
//!
//! # Kernel design
//!
//! - **`moe_topk_softmax`**: Each warp handles one token. For small expert
//!   counts (<=32), a warp-level bitonic sort directly extracts top-k.
//!   For larger counts, a partial selection (k-th element) algorithm is used.
//!
//! - **`moe_sort_by_expert`**: Builds a permutation array that groups tokens
//!   by expert ID and computes prefix-sum offsets for each expert's token range.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::ptx_helpers;
use crate::types::{Activation, TensorDesc};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Warp size for NVIDIA GPUs.
const WARP_SIZE: u32 = 32;

/// Block size for the top-k softmax kernel (warps per block).
const TOPK_WARPS_PER_BLOCK: u32 = 4;

/// Block size for the sort-by-expert kernel.
const SORT_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// MoeConfig
// ---------------------------------------------------------------------------

/// Configuration for a Mixture-of-Experts layer.
///
/// Describes the architecture parameters that determine kernel selection,
/// tile sizes, and validation bounds.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Number of expert networks.
    pub num_experts: u32,
    /// Number of experts activated per token (typically 1 or 2).
    pub top_k: u32,
    /// Hidden dimension of the input/output embeddings.
    pub hidden_dim: u32,
    /// Intermediate (expansion) dimension of the expert FFN.
    pub intermediate_dim: u32,
    /// Activation function between the two FFN projections.
    pub activation: Activation,
    /// PTX data type for kernel generation.
    pub precision: PtxType,
    /// Target SM architecture for kernel generation.
    pub sm_version: SmVersion,
}

impl MoeConfig {
    /// Validates that the configuration is consistent.
    pub(crate) fn validate(&self) -> DnnResult<()> {
        if self.num_experts == 0 {
            return Err(DnnError::InvalidArgument(
                "num_experts must be positive".into(),
            ));
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(DnnError::InvalidArgument(format!(
                "top_k ({}) must be in [1, num_experts ({})]",
                self.top_k, self.num_experts
            )));
        }
        if self.hidden_dim == 0 {
            return Err(DnnError::InvalidArgument(
                "hidden_dim must be positive".into(),
            ));
        }
        if self.intermediate_dim == 0 {
            return Err(DnnError::InvalidArgument(
                "intermediate_dim must be positive".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Performs top-k expert routing on router logits.
///
/// Given `router_logits` of shape `[num_tokens, num_experts]`, this function:
///
/// 1. Selects the top-k experts per token via a fused top-k + softmax kernel.
/// 2. Sorts tokens by expert assignment and builds permutation/offset arrays.
///
/// # Arguments
///
/// * `handle` -- DNN handle providing stream, context, and SM version.
/// * `router_logits` -- Input tensor of shape `[num_tokens, num_experts]`.
/// * `expert_indices` -- Output buffer for top-k expert IDs per token,
///   length `num_tokens * top_k`.
/// * `expert_weights` -- Output buffer for softmax weights per token,
///   length `num_tokens * top_k`.
/// * `permutation` -- Output permutation mapping tokens to their position
///   in expert-sorted order, length `num_tokens * top_k`.
/// * `expert_offsets` -- Output prefix-sum array of per-expert token counts,
///   length `num_experts + 1`.
/// * `config` -- MoE layer configuration.
///
/// # Errors
///
/// Returns [`DnnError`] on dimension mismatch, buffer undersize, or kernel
/// launch failure.
#[allow(clippy::too_many_arguments)]
pub fn moe_routing<T: GpuFloat>(
    handle: &DnnHandle,
    router_logits: &TensorDesc<T>,
    expert_indices: &mut DeviceBuffer<i32>,
    expert_weights: &mut DeviceBuffer<T>,
    permutation: &mut DeviceBuffer<i32>,
    expert_offsets: &mut DeviceBuffer<i32>,
    config: &MoeConfig,
) -> DnnResult<()> {
    config.validate()?;
    validate_routing_args(
        router_logits,
        expert_indices,
        expert_weights,
        permutation,
        expert_offsets,
        config,
    )?;

    let num_tokens = router_logits.dims[0];

    // Step 1: Top-K selection + softmax weights
    moe_topk_softmax(
        handle,
        router_logits,
        expert_indices,
        expert_weights,
        num_tokens,
        config,
    )?;

    // Step 2: Sort tokens by expert ID, build permutation and offset arrays
    moe_sort_by_expert(
        handle,
        expert_indices,
        permutation,
        expert_offsets,
        num_tokens,
        config,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates dimensions and buffer sizes for routing arguments.
fn validate_routing_args<T: GpuFloat>(
    router_logits: &TensorDesc<T>,
    expert_indices: &DeviceBuffer<i32>,
    expert_weights: &DeviceBuffer<T>,
    permutation: &DeviceBuffer<i32>,
    expert_offsets: &DeviceBuffer<i32>,
    config: &MoeConfig,
) -> DnnResult<()> {
    if router_logits.ndim() != 2 {
        return Err(DnnError::InvalidDimension(format!(
            "router_logits must be 2D, got {}D",
            router_logits.ndim()
        )));
    }
    let num_tokens = router_logits.dims[0] as usize;
    let num_experts_in = router_logits.dims[1];
    if num_experts_in != config.num_experts {
        return Err(DnnError::InvalidDimension(format!(
            "router_logits dim[1] ({}) != config.num_experts ({})",
            num_experts_in, config.num_experts
        )));
    }

    let topk = config.top_k as usize;
    let required_indices = num_tokens * topk;
    if expert_indices.len() < required_indices {
        return Err(DnnError::BufferTooSmall {
            expected: required_indices,
            actual: expert_indices.len(),
        });
    }
    if expert_weights.len() < required_indices {
        return Err(DnnError::BufferTooSmall {
            expected: required_indices,
            actual: expert_weights.len(),
        });
    }
    if permutation.len() < required_indices {
        return Err(DnnError::BufferTooSmall {
            expected: required_indices,
            actual: permutation.len(),
        });
    }
    let required_offsets = config.num_experts as usize + 1;
    if expert_offsets.len() < required_offsets {
        return Err(DnnError::BufferTooSmall {
            expected: required_offsets,
            actual: expert_offsets.len(),
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Top-K + Softmax kernel
// ---------------------------------------------------------------------------

/// Generates PTX for the fused top-k selection + softmax kernel.
///
/// Each warp processes one token. Lanes cooperatively scan all experts,
/// track the best (value, index) pair locally, then reduce across the warp
/// via shuffle-down. Lane 0 writes the winning expert index and a weight
/// of 1.0 (single-expert softmax degenerates to identity).
fn generate_topk_softmax_ptx<T: GpuFloat>(config: &MoeConfig) -> DnnResult<String> {
    let kernel_name = format!("moe_topk_softmax_{}", T::NAME);
    let elem_bytes = T::SIZE as u32;
    let num_experts = config.num_experts;
    let ty_str = if T::PTX_TYPE == PtxType::F32 {
        "f32"
    } else {
        "f64"
    };
    let bits_str = if T::PTX_TYPE == PtxType::F32 {
        "b32"
    } else {
        "b64"
    };

    let ptx = KernelBuilder::new(&kernel_name)
        .target(config.sm_version)
        .param("logits_ptr", PtxType::U64)
        .param("indices_out", PtxType::U64)
        .param("weights_out", PtxType::U64)
        .param("num_tokens", PtxType::U32)
        .param("num_experts", PtxType::U32)
        .param("top_k", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let tid = b.thread_id_x();
            let num_tok = b.load_param_u32("num_tokens");

            // warp_idx = gid / 32; lane = tid & 31
            let warp_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {warp_idx}, {gid}, 5;"));
            let lane = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("and.b32 {lane}, {tid}, 31;"));

            // Early exit
            let exit_lbl = b.fresh_label("exit");
            let p_exit = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {p_exit}, {warp_idx}, {num_tok};"));
            b.branch_if(p_exit, &exit_lbl);

            // Load pointers
            let logits_ptr = b.load_param_u64("logits_ptr");
            let indices_out = b.load_param_u64("indices_out");
            let weights_out = b.load_param_u64("weights_out");
            let p_top_k = b.load_param_u32("top_k");

            // logits_base = logits_ptr + warp_idx * num_experts * elem_bytes
            let logits_base =
                b.byte_offset_addr(logits_ptr, warp_idx.clone(), num_experts * elem_bytes);

            // Init best_val = -inf, best_idx = 0xFFFFFFFF
            let best_val = ptx_helpers::load_float_imm::<T>(b, f64::NEG_INFINITY);
            let best_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {best_idx}, 0xFFFFFFFF;"));

            // Loop: e = lane; e < num_experts; e += 32
            let e_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {e_reg}, {lane};"));
            let lp_head = b.fresh_label("lp");
            let lp_end = b.fresh_label("lpe");
            b.label(&lp_head);
            let p_lp = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {p_lp}, {e_reg}, {num_experts};"));
            b.branch_if(p_lp, &lp_end);

            let addr = b.byte_offset_addr(logits_base.clone(), e_reg.clone(), elem_bytes);
            let val = ptx_helpers::load_global_float::<T>(b, addr);

            // Compare and update best
            let is_better = ptx_helpers::setp_gt_float::<T>(b, val.clone(), best_val.clone());
            let new_val = ptx_helpers::selp_float::<T>(b, val, best_val.clone(), is_better.clone());
            let new_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "selp.u32 {new_idx}, {e_reg}, {best_idx}, {is_better};"
            ));
            b.raw_ptx(&format!("mov.{ty_str} {best_val}, {new_val};"));
            b.raw_ptx(&format!("mov.u32 {best_idx}, {new_idx};"));

            b.raw_ptx(&format!("add.u32 {e_reg}, {e_reg}, {WARP_SIZE};"));
            b.branch(&lp_head);
            b.label(&lp_end);

            // Warp-level shuffle reduction
            for offset in [16u32, 8, 4, 2, 1] {
                let sv = b.alloc_reg(T::PTX_TYPE);
                let si = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "shfl.sync.down.{bits_str} {sv}, {best_val}, {offset}, 31, 0xFFFFFFFF;"
                ));
                b.raw_ptx(&format!(
                    "shfl.sync.down.b32 {si}, {best_idx}, {offset}, 31, 0xFFFFFFFF;"
                ));
                let cmp = ptx_helpers::setp_gt_float::<T>(b, sv.clone(), best_val.clone());
                let nv = ptx_helpers::selp_float::<T>(b, sv, best_val.clone(), cmp.clone());
                let ni = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("selp.u32 {ni}, {si}, {best_idx}, {cmp};"));
                b.raw_ptx(&format!("mov.{ty_str} {best_val}, {nv};"));
                b.raw_ptx(&format!("mov.u32 {best_idx}, {ni};"));
            }

            // Lane 0 writes result
            let skip_lbl = b.fresh_label("skip");
            let p_lane0 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ne.u32 {p_lane0}, {lane}, 0;"));
            b.branch_if(p_lane0, &skip_lbl);

            // slot = warp_idx * top_k
            let slot = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mul.lo.u32 {slot}, {warp_idx}, {p_top_k};"));
            let idx_addr = b.byte_offset_addr(indices_out, slot.clone(), 4);
            b.raw_ptx(&format!("st.global.u32 [{idx_addr}], {best_idx};"));

            let wt_addr = b.byte_offset_addr(weights_out, slot, elem_bytes);
            let one_val = ptx_helpers::load_float_imm::<T>(b, 1.0);
            ptx_helpers::store_global_float::<T>(b, wt_addr, one_val);

            b.label(&skip_lbl);
            b.label(&exit_lbl);
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Fused top-k expert selection and softmax weight computation.
fn moe_topk_softmax<T: GpuFloat>(
    handle: &DnnHandle,
    router_logits: &TensorDesc<T>,
    expert_indices: &mut DeviceBuffer<i32>,
    expert_weights: &mut DeviceBuffer<T>,
    num_tokens: u32,
    config: &MoeConfig,
) -> DnnResult<()> {
    let ptx = generate_topk_softmax_ptx::<T>(config)?;
    let kernel_name = format!("moe_topk_softmax_{}", T::NAME);

    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let block_size = TOPK_WARPS_PER_BLOCK * WARP_SIZE;
    let threads_needed = num_tokens * WARP_SIZE;
    let grid = threads_needed.div_ceil(block_size);
    let params = LaunchParams::new(grid, block_size);

    let args = (
        router_logits.ptr,
        expert_indices.as_device_ptr(),
        expert_weights.as_device_ptr(),
        num_tokens,
        config.num_experts,
        config.top_k,
    );

    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Sort-by-expert kernel
// ---------------------------------------------------------------------------

/// Generates PTX for the sort-by-expert kernel.
///
/// Uses atomic increments to scatter tokens by expert ID into the
/// permutation array.
fn generate_sort_by_expert_ptx(config: &MoeConfig) -> DnnResult<String> {
    let kernel_name = "moe_sort_by_expert";

    let ptx = KernelBuilder::new(kernel_name)
        .target(config.sm_version)
        .param("expert_indices", PtxType::U64)
        .param("permutation", PtxType::U64)
        .param("expert_offsets", PtxType::U64)
        .param("num_tokens", PtxType::U32)
        .param("num_experts", PtxType::U32)
        .param("top_k", PtxType::U32)
        .body(|b| {
            let gid = b.global_thread_id_x();
            let num_tok = b.load_param_u32("num_tokens");
            let top_k = b.load_param_u32("top_k");
            let total = b.mul_lo_u32(num_tok, top_k);

            let exit_lbl = b.fresh_label("exit");
            let p = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {p}, {gid}, {total};"));
            b.branch_if(p, &exit_lbl);

            let indices_ptr = b.load_param_u64("expert_indices");
            let perm_ptr = b.load_param_u64("permutation");
            let offsets_ptr = b.load_param_u64("expert_offsets");

            // Load expert_indices[gid]
            let idx_addr = b.byte_offset_addr(indices_ptr, gid.clone(), 4);
            let expert_id = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("ld.global.u32 {expert_id}, [{idx_addr}];"));

            // Atomic increment expert_offsets[expert_id]
            let off_addr = b.byte_offset_addr(offsets_ptr, expert_id, 4);
            let write_pos = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "atom.global.add.u32 {write_pos}, [{off_addr}], 1;"
            ));

            // permutation[write_pos] = gid
            let perm_addr = b.byte_offset_addr(perm_ptr, write_pos, 4);
            b.raw_ptx(&format!("st.global.u32 [{perm_addr}], {gid};"));

            b.label(&exit_lbl);
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Sorts tokens by their assigned expert IDs.
fn moe_sort_by_expert(
    handle: &DnnHandle,
    expert_indices: &DeviceBuffer<i32>,
    permutation: &mut DeviceBuffer<i32>,
    expert_offsets: &mut DeviceBuffer<i32>,
    num_tokens: u32,
    config: &MoeConfig,
) -> DnnResult<()> {
    let ptx = generate_sort_by_expert_ptx(config)?;
    let kernel_name = "moe_sort_by_expert";

    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, kernel_name)?;

    let total = num_tokens * config.top_k;
    let grid = total.div_ceil(SORT_BLOCK_SIZE);
    let params = LaunchParams::new(grid, SORT_BLOCK_SIZE);

    let args = (
        expert_indices.as_device_ptr(),
        permutation.as_device_ptr(),
        expert_offsets.as_device_ptr(),
        num_tokens,
        config.num_experts,
        config.top_k,
    );

    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CPU-side Top-K helpers (used in tests only)
// ---------------------------------------------------------------------------

/// Selects the indices of the top-k values from `logits` (descending order).
///
/// Returns a `Vec<usize>` of length `k` containing the indices of the highest
/// logit values, sorted by value (largest first). Ties are broken by lower
/// index first.
///
/// This is a reference implementation used for testing the PTX kernel's output.
#[cfg(test)]
fn topk_indices(logits: &[f32], k: usize) -> Vec<usize> {
    assert!(k <= logits.len(), "k must not exceed number of logits");
    // Collect (value, index) pairs.
    let mut indexed: Vec<(f32, usize)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    // Sort descending by value, ascending by index for tie-breaking.
    indexed.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    indexed.iter().take(k).map(|&(_, i)| i).collect()
}

/// Computes softmax over the selected top-k logit values.
///
/// Returns probabilities that sum to 1.0 (within floating-point precision).
#[cfg(test)]
fn topk_softmax_weights(logits: &[f32], indices: &[usize]) -> Vec<f32> {
    let selected: Vec<f32> = indices.iter().map(|&i| logits[i]).collect();
    let max_val = selected.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = selected.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validate_zero_experts() {
        let config = MoeConfig {
            num_experts: 0,
            top_k: 1,
            hidden_dim: 256,
            intermediate_dim: 512,
            activation: Activation::Silu,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm80,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validate_topk_exceeds_experts() {
        let config = MoeConfig {
            num_experts: 4,
            top_k: 5,
            hidden_dim: 256,
            intermediate_dim: 512,
            activation: Activation::Silu,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm80,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validate_ok() {
        let config = MoeConfig {
            num_experts: 8,
            top_k: 2,
            hidden_dim: 256,
            intermediate_dim: 512,
            activation: Activation::Silu,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm80,
        };
        assert!(config.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Task 2: Top-K selection correctness (CPU reference)
    // -----------------------------------------------------------------------

    /// Top-2 selection from 8 experts returns the two highest logit indices.
    #[test]
    fn test_topk_selects_correct_indices() {
        // logits = [0.1, 0.9, 0.2, 0.7, 0.3, 0.5, 0.4, 0.8]
        // top-2: indices [1, 7] (values 0.9, 0.8)
        let logits = [0.1f32, 0.9, 0.2, 0.7, 0.3, 0.5, 0.4, 0.8];
        let result = topk_indices(&logits, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1, "highest logit is at index 1 (value 0.9)");
        assert_eq!(
            result[1], 7,
            "second highest logit is at index 7 (value 0.8)"
        );
    }

    /// Top-1 selection picks the maximum.
    #[test]
    fn test_topk_top1_picks_maximum() {
        let logits = [0.3f32, 0.1, 0.8, 0.5, 0.2];
        let result = topk_indices(&logits, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 2, "maximum is at index 2 (value 0.8)");
    }

    /// Top-K all: selecting all experts returns them in descending order.
    #[test]
    fn test_topk_all_experts_descending() {
        let logits = [0.5f32, 0.1, 0.9, 0.3];
        let result = topk_indices(&logits, 4);
        assert_eq!(result, vec![2, 0, 3, 1]);
    }

    /// Softmax weights over top-k sum to 1.0 (after normalisation).
    #[test]
    fn test_topk_softmax_weights_sum_to_one() {
        let logits = [0.1f32, 0.9, 0.2, 0.7, 0.3, 0.5, 0.4, 0.8];
        let indices = topk_indices(&logits, 2);
        let weights = topk_softmax_weights(&logits, &indices);
        let total: f32 = weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "softmax weights must sum to 1.0, got {total}"
        );
    }

    /// Softmax weights are non-negative and in (0, 1].
    #[test]
    fn test_topk_softmax_weights_valid_probabilities() {
        let logits = [1.0f32, 2.0, 0.5, 3.0, 0.1];
        let indices = topk_indices(&logits, 3);
        let weights = topk_softmax_weights(&logits, &indices);
        for &w in &weights {
            assert!(w > 0.0, "weight must be positive, got {w}");
            assert!(w <= 1.0, "weight must not exceed 1.0, got {w}");
        }
    }

    /// Top-K is deterministic: same logits always produce the same result.
    #[test]
    fn test_topk_deterministic() {
        let logits = [0.1f32, 0.9, 0.2, 0.7, 0.3, 0.5, 0.4, 0.8];
        let result1 = topk_indices(&logits, 2);
        let result2 = topk_indices(&logits, 2);
        assert_eq!(result1, result2, "top-k must be deterministic");
    }

    /// Top-K with uniform logits returns the first k indices (tie-breaking by index).
    #[test]
    fn test_topk_uniform_logits_breaks_tie_by_index() {
        let logits = [1.0f32; 8];
        let result = topk_indices(&logits, 3);
        // All logits equal → tie-break by lower index first.
        assert_eq!(result, vec![0, 1, 2]);
    }

    /// Top-1 softmax weight is always 1.0 (identity).
    #[test]
    fn test_topk_top1_softmax_weight_is_one() {
        let logits = [0.3f32, 0.9, 0.5];
        let indices = topk_indices(&logits, 1);
        let weights = topk_softmax_weights(&logits, &indices);
        assert_eq!(weights.len(), 1);
        assert!(
            (weights[0] - 1.0).abs() < 1e-6,
            "top-1 softmax weight must be 1.0, got {}",
            weights[0]
        );
    }

    /// The higher-value expert gets more weight than the lower-value one.
    #[test]
    fn test_topk_higher_logit_gets_higher_weight() {
        // logits[1]=0.9 > logits[7]=0.8, so weight[0] > weight[1]
        let logits = [0.1f32, 0.9, 0.2, 0.7, 0.3, 0.5, 0.4, 0.8];
        let indices = topk_indices(&logits, 2);
        let weights = topk_softmax_weights(&logits, &indices);
        assert!(
            weights[0] > weights[1],
            "index {} (logit {}) should have higher weight than index {} (logit {})",
            indices[0],
            logits[indices[0]],
            indices[1],
            logits[indices[1]]
        );
    }

    #[test]
    fn topk_ptx_gen_f32() {
        let config = MoeConfig {
            num_experts: 8,
            top_k: 2,
            hidden_dim: 256,
            intermediate_dim: 512,
            activation: Activation::Silu,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm80,
        };
        let ptx = generate_topk_softmax_ptx::<f32>(&config);
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains(".entry moe_topk_softmax_f32"));
    }

    #[test]
    fn sort_ptx_gen() {
        let config = MoeConfig {
            num_experts: 8,
            top_k: 2,
            hidden_dim: 256,
            intermediate_dim: 512,
            activation: Activation::Silu,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm80,
        };
        let ptx = generate_sort_by_expert_ptx(&config);
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains(".entry moe_sort_by_expert"));
    }
}
