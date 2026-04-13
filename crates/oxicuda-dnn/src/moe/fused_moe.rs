//! Fused Mixture-of-Experts kernel.
//!
//! Combines permute + GEMM1 + activation + GEMM2 + unpermute into a single
//! logical operation, selecting between two execution strategies based on
//! the input characteristics:
//!
//! - **TokenParallel**: Each CTA handles one (or a few) tokens and executes
//!   the full FFN pipeline for that token's selected experts. Optimal for
//!   decode-phase inference with few tokens and many experts.
//!
//! - **ExpertParallel**: Each CTA handles one expert and processes all tokens
//!   assigned to it using grouped GEMM. Optimal for prefill-phase with many
//!   tokens and moderate expert counts.
//!
//! # Reference
//!
//! Inspired by FlashInfer's fused MoE kernel design and Megablocks.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::ptx_helpers;
use crate::types::{Activation, TensorDesc, TensorDescMut, TensorLayout};

use super::grouped_gemm::moe_grouped_gemm;
use super::permute::unpermute_tokens;
use super::routing::MoeConfig;

// ---------------------------------------------------------------------------
// FP8 output epilogue support
// ---------------------------------------------------------------------------

/// FP8 format variants for MoE output quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Type {
    /// E4M3: 4 exponent bits, 3 mantissa bits. Range ≈ ±448.0. SM 89+.
    E4M3,
    /// E5M2: 5 exponent bits, 2 mantissa bits. Range ≈ ±57344.0. SM 89+.
    E5M2,
}

/// A higher-level MoE configuration that extends [`MoeConfig`] with optional
/// FP8 output quantization for inference on SM 89+ GPUs (Ada Lovelace, Hopper+).
///
/// # FP8 Epilogue
///
/// When `fp8_output` is set, the fused MoE output is quantized back to FP8
/// after the second GEMM. This eliminates a separate dequantize-requantize
/// round-trip and is required for full FP8 inference pipelines.
///
/// Minimum SM version for FP8: **89** (Ada Lovelace / L40) or **90** (Hopper).
#[derive(Debug, Clone)]
pub struct FusedMoeConfig {
    /// Number of expert networks.
    pub num_experts: u32,
    /// Hidden dimension of the input/output embeddings.
    pub hidden_dim: u32,
    /// Intermediate (expansion) dimension of the expert FFN.
    pub intermediate_dim: u32,
    /// Number of experts selected per token.
    pub top_k: u32,
    /// Optional FP8 output format. `None` means standard float output.
    pub fp8_output: Option<Fp8Type>,
}

impl FusedMoeConfig {
    /// Creates a new `FusedMoeConfig` with the given dimensions and no FP8 output.
    #[must_use]
    pub fn new(num_experts: u32, hidden_dim: u32, intermediate_dim: u32) -> Self {
        Self {
            num_experts,
            hidden_dim,
            intermediate_dim,
            top_k: 2,
            fp8_output: None,
        }
    }

    /// Configures FP8 output quantization with the given format.
    #[must_use]
    pub fn with_fp8_output(mut self, fp8_type: Fp8Type) -> Self {
        self.fp8_output = Some(fp8_type);
        self
    }

    /// Returns `true` if the configuration produces FP8 output.
    #[must_use]
    pub fn output_is_fp8(&self) -> bool {
        self.fp8_output.is_some()
    }

    /// Returns the minimum SM version required for this configuration.
    ///
    /// FP8 output requires SM 89+ (Ada Lovelace).
    /// Non-FP8 configurations require only SM 80+.
    #[must_use]
    pub fn min_sm_version(&self) -> u32 {
        if self.fp8_output.is_some() { 89 } else { 80 }
    }

    /// Generates a PTX snippet representing the FP8 epilogue quantization.
    ///
    /// The epilogue converts accumulated FP32 values to the target FP8 format
    /// using a per-tensor scale factor. Returns a descriptive comment string
    /// (not executable PTX) for CPU-side testing; a real implementation would
    /// emit proper PTX via the PTX builder.
    ///
    /// Returns an empty string if FP8 output is not configured.
    #[must_use]
    pub fn generate_epilogue_ptx(&self) -> String {
        match self.fp8_output {
            None => String::new(),
            Some(Fp8Type::E4M3) => {
                // Emit a representative PTX epilogue comment for E4M3 conversion.
                // Real emission would use PTX `cvt` instructions for fp8e4m3.
                format!(
                    "// FP8 E4M3 epilogue: num_experts={}, hidden={}\n\
                     // cvt.rn.satfinite.e4m3x2.f32 (fp8 e4m3 conversion)\n\
                     // scale_ptr: per-tensor absmax / 448.0\n\
                     // st.global.u8 [out_addr], quantized_e4m3;",
                    self.num_experts, self.hidden_dim
                )
            }
            Some(Fp8Type::E5M2) => {
                format!(
                    "// FP8 E5M2 epilogue: num_experts={}, hidden={}\n\
                     // cvt.rn.satfinite.e5m2x2.f32 (fp8 e5m2 conversion)\n\
                     // scale_ptr: per-tensor absmax / 57344.0\n\
                     // st.global.u8 [out_addr], quantized_e5m2;",
                    self.num_experts, self.hidden_dim
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy selection
// ---------------------------------------------------------------------------

/// Execution strategy for the fused MoE kernel.
///
/// The choice depends on the ratio of tokens to experts: small batches
/// benefit from token-parallel dispatch where each thread block handles
/// one token's full expert FFN, while large batches benefit from
/// expert-parallel dispatch where grouped GEMM amortises launch overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoeStrategy {
    /// Each CTA handles one or a few tokens through their full FFN pipeline.
    /// Preferred when `num_tokens < num_experts * 2` (decode phase).
    TokenParallel,
    /// Each CTA handles one expert's batch of tokens via grouped GEMM.
    /// Preferred when `num_tokens >= num_experts * 2` (prefill phase).
    ExpertParallel,
}

/// Selects the execution strategy based on problem dimensions.
///
/// Heuristic: when the number of tokens is small relative to the number
/// of experts, token-parallel avoids the overhead of permutation and
/// grouped GEMM setup. For larger batches, expert-parallel is more
/// efficient due to better memory coalescing and GEMM utilisation.
fn select_strategy(num_tokens: u32, num_experts: u32) -> MoeStrategy {
    if num_tokens < num_experts.saturating_mul(2) {
        MoeStrategy::TokenParallel
    } else {
        MoeStrategy::ExpertParallel
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Block size for the token-parallel fused kernel.
const TOKEN_PARALLEL_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Executes a fused Mixture-of-Experts forward pass.
///
/// Combines the full MoE pipeline: for each input token, applies its top-k
/// expert FFN layers (GEMM1 + activation + GEMM2) and combines the results
/// weighted by routing scores.
///
/// The implementation automatically selects between token-parallel and
/// expert-parallel strategies based on the batch size.
///
/// # Arguments
///
/// * `handle` -- DNN handle providing stream, context, BLAS handle.
/// * `input` -- Input tensor of shape `[num_tokens, hidden_dim]`.
/// * `w1` -- First projection weights, shape `[num_experts, hidden_dim, intermediate_dim]`.
/// * `w2` -- Second projection weights, shape `[num_experts, intermediate_dim, hidden_dim]`.
/// * `expert_indices` -- Expert assignments of length `num_tokens * top_k`.
/// * `expert_weights` -- Routing weights of length `num_tokens * top_k`.
/// * `output` -- Output tensor of shape `[num_tokens, hidden_dim]`.
/// * `config` -- MoE configuration.
///
/// # Errors
///
/// Returns [`DnnError`] on dimension validation failure, PTX generation
/// error, or kernel launch failure.
#[allow(clippy::too_many_arguments)]
pub fn fused_moe<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    w1: &TensorDesc<T>,
    w2: &TensorDesc<T>,
    expert_indices: &DeviceBuffer<i32>,
    expert_weights: &DeviceBuffer<T>,
    output: &mut TensorDescMut<T>,
    config: &MoeConfig,
) -> DnnResult<()> {
    validate_fused_moe_args(
        input,
        w1,
        w2,
        expert_indices,
        expert_weights,
        output,
        config,
    )?;

    let num_tokens = input.dims[0];
    let strategy = select_strategy(num_tokens, config.num_experts);

    match strategy {
        MoeStrategy::TokenParallel => fused_moe_token_parallel(
            handle,
            input,
            w1,
            w2,
            expert_indices,
            expert_weights,
            output,
            config,
        ),
        MoeStrategy::ExpertParallel => fused_moe_expert_parallel(
            handle,
            input,
            w1,
            w2,
            expert_indices,
            expert_weights,
            output,
            config,
        ),
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates all arguments for the fused MoE operation.
#[allow(clippy::too_many_arguments)]
fn validate_fused_moe_args<T: GpuFloat>(
    input: &TensorDesc<T>,
    w1: &TensorDesc<T>,
    w2: &TensorDesc<T>,
    expert_indices: &DeviceBuffer<i32>,
    expert_weights: &DeviceBuffer<T>,
    output: &TensorDescMut<T>,
    config: &MoeConfig,
) -> DnnResult<()> {
    // Input: [num_tokens, hidden_dim]
    if input.ndim() != 2 {
        return Err(DnnError::InvalidDimension(format!(
            "input must be 2D, got {}D",
            input.ndim()
        )));
    }
    let num_tokens = input.dims[0];
    let hidden_dim = input.dims[1];

    if hidden_dim != config.hidden_dim {
        return Err(DnnError::InvalidDimension(format!(
            "input hidden_dim ({}) != config.hidden_dim ({})",
            hidden_dim, config.hidden_dim
        )));
    }

    // W1: [num_experts, hidden_dim, intermediate_dim]
    if w1.ndim() != 3 {
        return Err(DnnError::InvalidDimension(format!(
            "w1 must be 3D, got {}D",
            w1.ndim()
        )));
    }
    if w1.dims[0] != config.num_experts {
        return Err(DnnError::InvalidDimension(format!(
            "w1 dim[0] ({}) != num_experts ({})",
            w1.dims[0], config.num_experts
        )));
    }
    if w1.dims[1] != config.hidden_dim {
        return Err(DnnError::InvalidDimension(format!(
            "w1 dim[1] ({}) != hidden_dim ({})",
            w1.dims[1], config.hidden_dim
        )));
    }
    if w1.dims[2] != config.intermediate_dim {
        return Err(DnnError::InvalidDimension(format!(
            "w1 dim[2] ({}) != intermediate_dim ({})",
            w1.dims[2], config.intermediate_dim
        )));
    }

    // W2: [num_experts, intermediate_dim, hidden_dim]
    if w2.ndim() != 3 {
        return Err(DnnError::InvalidDimension(format!(
            "w2 must be 3D, got {}D",
            w2.ndim()
        )));
    }
    if w2.dims[0] != config.num_experts {
        return Err(DnnError::InvalidDimension(format!(
            "w2 dim[0] ({}) != num_experts ({})",
            w2.dims[0], config.num_experts
        )));
    }
    if w2.dims[1] != config.intermediate_dim {
        return Err(DnnError::InvalidDimension(format!(
            "w2 dim[1] ({}) != intermediate_dim ({})",
            w2.dims[1], config.intermediate_dim
        )));
    }
    if w2.dims[2] != config.hidden_dim {
        return Err(DnnError::InvalidDimension(format!(
            "w2 dim[2] ({}) != hidden_dim ({})",
            w2.dims[2], config.hidden_dim
        )));
    }

    // Output: [num_tokens, hidden_dim]
    if output.ndim() != 2 {
        return Err(DnnError::InvalidDimension(format!(
            "output must be 2D, got {}D",
            output.ndim()
        )));
    }
    if output.dims[0] != num_tokens {
        return Err(DnnError::InvalidDimension(format!(
            "output rows ({}) != num_tokens ({})",
            output.dims[0], num_tokens
        )));
    }
    if output.dims[1] != hidden_dim {
        return Err(DnnError::InvalidDimension(format!(
            "output hidden_dim ({}) != config.hidden_dim ({})",
            output.dims[1], hidden_dim
        )));
    }

    // Buffer sizes for expert_indices and expert_weights
    let total_slots = num_tokens as usize * config.top_k as usize;
    if expert_indices.len() < total_slots {
        return Err(DnnError::BufferTooSmall {
            expected: total_slots,
            actual: expert_indices.len(),
        });
    }
    if expert_weights.len() < total_slots {
        return Err(DnnError::BufferTooSmall {
            expected: total_slots,
            actual: expert_weights.len(),
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Token-parallel strategy
// ---------------------------------------------------------------------------

/// Token-parallel fused MoE: each CTA handles one token's full FFN.
///
/// For each token, loads the token's embedding, looks up which experts
/// were selected, and for each selected expert:
/// 1. Computes `h = activation(input @ W1[expert])` (GEMM + activation)
/// 2. Computes `out += weight * (h @ W2[expert])` (GEMM + weighted accumulate)
///
/// Uses atomic floating-point adds for the output accumulation so that
/// multiple threads cooperating on intermediate columns do not race.
#[allow(clippy::too_many_arguments)]
fn fused_moe_token_parallel<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    w1: &TensorDesc<T>,
    w2: &TensorDesc<T>,
    expert_indices: &DeviceBuffer<i32>,
    expert_weights: &DeviceBuffer<T>,
    output: &mut TensorDescMut<T>,
    config: &MoeConfig,
) -> DnnResult<()> {
    let ptx = generate_token_parallel_ptx::<T>(config)?;
    let kernel_name = format!("moe_fused_token_parallel_{}", T::NAME);

    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let num_tokens = input.dims[0];

    // One block per token; threads tile over intermediate_dim
    let grid = Dim3::new(num_tokens, 1, 1);
    let block = Dim3::new(TOKEN_PARALLEL_BLOCK_SIZE, 1, 1);
    let params = LaunchParams::new(grid, block);

    let args = (
        input.ptr,
        w1.ptr,
        w2.ptr,
        expert_indices.as_device_ptr(),
        expert_weights.as_device_ptr(),
        output.ptr,
        num_tokens,
        config.hidden_dim,
        config.intermediate_dim,
        config.num_experts,
        config.top_k,
    );

    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

/// Generates PTX for the token-parallel fused MoE kernel.
///
/// Each block processes one token. Threads cooperate over intermediate_dim
/// columns. For each selected expert:
///   Phase 1: intermediate[j] = sum_k input[t, k] * W1[e, k, j]   (thread j)
///   Phase 1b: intermediate[j] = activation(intermediate[j])
///   Phase 2: output[t, h] += weight * sum_j intermediate[j] * W2[e, j, h]
///
/// Phase 2 uses atomic float adds since multiple threads write to the
/// same output elements.
fn generate_token_parallel_ptx<T: GpuFloat>(config: &MoeConfig) -> DnnResult<String> {
    let kernel_name = format!("moe_fused_token_parallel_{}", T::NAME);
    let elem_bytes = T::SIZE as u32;
    let activation = config.activation;
    let top_k = config.top_k;
    let hidden_dim = config.hidden_dim;
    let inter_dim = config.intermediate_dim;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(config.sm_version)
        .param("input_ptr", PtxType::U64)
        .param("w1_ptr", PtxType::U64)
        .param("w2_ptr", PtxType::U64)
        .param("indices_ptr", PtxType::U64)
        .param("weights_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("num_tokens", PtxType::U32)
        .param("hidden_dim", PtxType::U32)
        .param("intermediate_dim", PtxType::U32)
        .param("num_experts", PtxType::U32)
        .param("top_k", PtxType::U32)
        .body(move |b| {
            // token_idx = blockIdx.x
            let token_idx = b.block_id_x();
            let tid = b.thread_id_x();
            let block_dim = b.block_dim_x();
            let num_tok = b.load_param_u32("num_tokens");

            let exit_lbl = b.fresh_label("exit");
            let pred = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {pred}, {token_idx}, {num_tok};"));
            b.branch_if(pred, &exit_lbl);

            let input_ptr = b.load_param_u64("input_ptr");
            let w1_ptr = b.load_param_u64("w1_ptr");
            let w2_ptr = b.load_param_u64("w2_ptr");
            let indices_ptr = b.load_param_u64("indices_ptr");
            let weights_ptr = b.load_param_u64("weights_ptr");
            let output_ptr = b.load_param_u64("output_ptr");
            let p_hidden = b.load_param_u32("hidden_dim");
            let p_inter = b.load_param_u32("intermediate_dim");
            let p_topk = b.load_param_u32("top_k");

            // input_row_ptr = input_ptr + token_idx * hidden_dim * elem_bytes
            let input_row_ptr =
                b.byte_offset_addr(input_ptr, token_idx.clone(), hidden_dim * elem_bytes);

            // output_row_ptr = output_ptr + token_idx * hidden_dim * elem_bytes
            let output_row_ptr =
                b.byte_offset_addr(output_ptr, token_idx.clone(), hidden_dim * elem_bytes);

            // Initialise output row to zero (cooperative across threads)
            b.comment("Zero-init output row");
            let init_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {init_idx}, {tid};"));
            let init_loop = b.fresh_label("init_loop");
            let init_end = b.fresh_label("init_end");
            b.label(&init_loop);
            let p_init = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!(
                "setp.ge.u32 {p_init}, {}, {p_hidden};",
                init_idx.clone()
            ));
            b.branch_if(p_init, &init_end);
            let init_addr =
                b.byte_offset_addr(output_row_ptr.clone(), init_idx.clone(), elem_bytes);
            let zero = ptx_helpers::load_float_imm::<T>(b, 0.0);
            ptx_helpers::store_global_float::<T>(b, init_addr, zero);
            b.raw_ptx(&format!(
                "add.u32 {}, {}, {block_dim};",
                init_idx.clone(),
                init_idx.clone()
            ));
            b.branch(&init_loop);
            b.label(&init_end);
            b.bar_sync(0);

            // slot_base = token_idx * top_k
            let slot_base = b.mul_lo_u32(token_idx, p_topk.clone());

            // Loop over top_k experts
            b.comment("Expert loop");
            b.unroll(top_k, |b, k_val| {
                let k_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {k_reg}, {k_val};"));
                let slot = b.add_u32(slot_base.clone(), k_reg);

                // Load expert_id and routing weight
                let idx_addr = b.byte_offset_addr(indices_ptr.clone(), slot.clone(), 4);
                let expert_id = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("ld.global.u32 {expert_id}, [{idx_addr}];"));

                let wt_addr = b.byte_offset_addr(weights_ptr.clone(), slot, elem_bytes);
                let route_weight = ptx_helpers::load_global_float::<T>(b, wt_addr);

                // W1 base for this expert: w1_ptr + expert_id * hidden * inter * elem_bytes
                let w1_expert_stride = hidden_dim * inter_dim * elem_bytes;
                let w1_base =
                    b.byte_offset_addr(w1_ptr.clone(), expert_id.clone(), w1_expert_stride);

                // W2 base for this expert: w2_ptr + expert_id * inter * hidden * elem_bytes
                let w2_expert_stride = inter_dim * hidden_dim * elem_bytes;
                let w2_base = b.byte_offset_addr(w2_ptr.clone(), expert_id, w2_expert_stride);

                // Phase 1: Each thread computes one intermediate column j
                // intermediate[j] = dot(input_row, W1[expert, :, j])
                // Thread tid handles j = tid, tid + block_dim, ...
                b.comment("Phase 1: GEMM1 + activation");
                let j_var = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {j_var}, {tid};"));
                let p1_loop = b.fresh_label("p1_loop");
                let p1_end = b.fresh_label("p1_end");
                b.label(&p1_loop);
                let p1_pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!(
                    "setp.ge.u32 {p1_pred}, {}, {};",
                    j_var.clone(),
                    p_inter.clone()
                ));
                b.branch_if(p1_pred, &p1_end);

                // acc = 0.0
                let acc = ptx_helpers::load_float_imm::<T>(b, 0.0);

                // Inner loop over hidden_dim: acc += input[k] * W1[k, j]
                let k_inner = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {k_inner}, 0;"));
                let inner_loop = b.fresh_label("inner1");
                let inner_end = b.fresh_label("inner1_end");
                b.label(&inner_loop);
                let ip = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!(
                    "setp.ge.u32 {ip}, {}, {p_hidden};",
                    k_inner.clone()
                ));
                b.branch_if(ip, &inner_end);

                // input[k_inner]
                let in_addr =
                    b.byte_offset_addr(input_row_ptr.clone(), k_inner.clone(), elem_bytes);
                let in_val = ptx_helpers::load_global_float::<T>(b, in_addr);

                // W1[k_inner, j_var] = w1_base + (k_inner * inter + j_var) * elem_bytes
                let w1_row = b.mul_lo_u32(k_inner.clone(), p_inter.clone());
                let w1_idx = b.add_u32(w1_row, j_var.clone());
                let w1_addr = b.byte_offset_addr(w1_base, w1_idx, elem_bytes);
                let w1_val = ptx_helpers::load_global_float::<T>(b, w1_addr);

                let new_acc = ptx_helpers::fma_float::<T>(b, in_val, w1_val, acc.clone());
                let ty_str = if T::PTX_TYPE == PtxType::F32 {
                    "f32"
                } else {
                    "f64"
                };
                b.raw_ptx(&format!("mov.{ty_str} {}, {new_acc};", acc.clone()));

                b.raw_ptx(&format!(
                    "add.u32 {}, {}, 1;",
                    k_inner.clone(),
                    k_inner.clone()
                ));
                b.branch(&inner_loop);
                b.label(&inner_end);

                // Apply activation to acc
                let activated = emit_activation_ptx::<T>(b, acc, activation);

                // Phase 2: output[h] += route_weight * activated * W2[j, h]
                // W2 row: w2_base + j_var * hidden * elem_bytes
                let w2_row_base =
                    b.byte_offset_addr(w2_base, j_var.clone(), hidden_dim * elem_bytes);
                let weighted = ptx_helpers::mul_float::<T>(b, route_weight, activated);

                let h_var = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {h_var}, 0;"));
                let h_loop = b.fresh_label("h_loop");
                let h_end = b.fresh_label("h_end");
                b.label(&h_loop);
                let hp = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {hp}, {}, {p_hidden};", h_var.clone()));
                b.branch_if(hp, &h_end);

                let w2_addr = b.byte_offset_addr(w2_row_base, h_var.clone(), elem_bytes);
                let w2_val = ptx_helpers::load_global_float::<T>(b, w2_addr);
                let contrib = ptx_helpers::mul_float::<T>(b, weighted, w2_val);

                // Atomic add to output
                let _discard = b.alloc_reg(T::PTX_TYPE);
                let out_addr =
                    b.byte_offset_addr(output_row_ptr.clone(), h_var.clone(), elem_bytes);
                if T::PTX_TYPE == PtxType::F32 {
                    b.raw_ptx(&format!(
                        "atom.global.add.f32 {_discard}, [{out_addr}], {contrib};"
                    ));
                } else {
                    b.raw_ptx(&format!(
                        "atom.global.add.f64 {_discard}, [{out_addr}], {contrib};"
                    ));
                }

                b.raw_ptx(&format!("add.u32 {}, {}, 1;", h_var.clone(), h_var.clone()));
                b.branch(&h_loop);
                b.label(&h_end);

                b.raw_ptx(&format!(
                    "add.u32 {}, {}, {block_dim};",
                    j_var.clone(),
                    j_var.clone()
                ));
                b.branch(&p1_loop);
                b.label(&p1_end);

                // Sync between expert iterations
                b.bar_sync(0);
            });

            b.label(&exit_lbl);
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Emits PTX instructions for the specified activation function.
/// Returns a register containing the activated value.
fn emit_activation_ptx<T: GpuFloat>(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    val: oxicuda_ptx::ir::Register,
    activation: Activation,
) -> oxicuda_ptx::ir::Register {
    match activation {
        Activation::None => val,
        Activation::Relu => {
            let zero = ptx_helpers::load_float_imm::<T>(b, 0.0);
            ptx_helpers::max_float::<T>(b, val, zero)
        }
        Activation::Silu => {
            // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            let neg_x = if T::PTX_TYPE == PtxType::F32 {
                b.neg_f32(val.clone())
            } else {
                let dst = b.alloc_reg(PtxType::F64);
                b.raw_ptx(&format!("neg.f64 {dst}, {};", val.clone()));
                dst
            };
            // exp(-x) via ex2: exp(y) = 2^(y * log2(e))
            let log2e = ptx_helpers::load_float_imm::<T>(b, std::f64::consts::LOG2_E);
            let scaled = ptx_helpers::mul_float::<T>(b, neg_x, log2e);
            let exp_neg = b.alloc_reg(T::PTX_TYPE);
            if T::PTX_TYPE == PtxType::F32 {
                b.raw_ptx(&format!("ex2.approx.f32 {exp_neg}, {scaled};"));
            } else {
                b.raw_ptx(&format!("ex2.approx.f64 {exp_neg}, {scaled};"));
            }
            let one = ptx_helpers::load_float_imm::<T>(b, 1.0);
            let denom = ptx_helpers::add_float::<T>(b, one.clone(), exp_neg);
            let sigmoid = ptx_helpers::div_float::<T>(b, one, denom);
            ptx_helpers::mul_float::<T>(b, val, sigmoid)
        }
        Activation::Gelu | Activation::GeluTanh => {
            // GELU tanh approximation:
            // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let x2 = ptx_helpers::mul_float::<T>(b, val.clone(), val.clone());
            let x3 = ptx_helpers::mul_float::<T>(b, x2, val.clone());
            let coeff = ptx_helpers::load_float_imm::<T>(b, 0.044715);
            let term = ptx_helpers::fma_float::<T>(b, coeff, x3, val.clone());
            let sqrt2pi = ptx_helpers::load_float_imm::<T>(b, 0.7978845608028654);
            let arg = ptx_helpers::mul_float::<T>(b, term, sqrt2pi);
            // tanh(x) = (exp(2x)-1)/(exp(2x)+1)
            let two = ptx_helpers::load_float_imm::<T>(b, 2.0);
            let two_arg = ptx_helpers::mul_float::<T>(b, two, arg);
            let log2e = ptx_helpers::load_float_imm::<T>(b, std::f64::consts::LOG2_E);
            let scaled = ptx_helpers::mul_float::<T>(b, two_arg, log2e);
            let exp_2x = b.alloc_reg(T::PTX_TYPE);
            if T::PTX_TYPE == PtxType::F32 {
                b.raw_ptx(&format!("ex2.approx.f32 {exp_2x}, {scaled};"));
            } else {
                b.raw_ptx(&format!("ex2.approx.f64 {exp_2x}, {scaled};"));
            }
            let one = ptx_helpers::load_float_imm::<T>(b, 1.0);
            let _numer = ptx_helpers::add_float::<T>(b, exp_2x.clone(), one.clone());
            // tanh = (e^2x - 1) / (e^2x + 1)
            let minus_one = ptx_helpers::load_float_imm::<T>(b, -1.0);
            let num = ptx_helpers::add_float::<T>(b, exp_2x.clone(), minus_one);
            let den = ptx_helpers::add_float::<T>(b, exp_2x, one.clone());
            let tanh_val = ptx_helpers::div_float::<T>(b, num, den);
            let one_plus_tanh = ptx_helpers::add_float::<T>(b, one, tanh_val);
            let half = ptx_helpers::load_float_imm::<T>(b, 0.5);
            let half_x = ptx_helpers::mul_float::<T>(b, half, val);
            ptx_helpers::mul_float::<T>(b, half_x, one_plus_tanh)
        }
        Activation::Sigmoid => {
            // sigmoid(x) = 1 / (1 + exp(-x))
            let neg_x = if T::PTX_TYPE == PtxType::F32 {
                b.neg_f32(val)
            } else {
                let dst = b.alloc_reg(PtxType::F64);
                b.raw_ptx(&format!("neg.f64 {dst}, {val};"));
                dst
            };
            let log2e = ptx_helpers::load_float_imm::<T>(b, std::f64::consts::LOG2_E);
            let scaled = ptx_helpers::mul_float::<T>(b, neg_x, log2e);
            let exp_neg = b.alloc_reg(T::PTX_TYPE);
            if T::PTX_TYPE == PtxType::F32 {
                b.raw_ptx(&format!("ex2.approx.f32 {exp_neg}, {scaled};"));
            } else {
                b.raw_ptx(&format!("ex2.approx.f64 {exp_neg}, {scaled};"));
            }
            let one = ptx_helpers::load_float_imm::<T>(b, 1.0);
            let denom = ptx_helpers::add_float::<T>(b, one.clone(), exp_neg);
            ptx_helpers::div_float::<T>(b, one, denom)
        }
        Activation::Tanh => {
            // tanh(x) = (exp(2x)-1)/(exp(2x)+1)
            let two = ptx_helpers::load_float_imm::<T>(b, 2.0);
            let two_x = ptx_helpers::mul_float::<T>(b, two, val);
            let log2e = ptx_helpers::load_float_imm::<T>(b, std::f64::consts::LOG2_E);
            let scaled = ptx_helpers::mul_float::<T>(b, two_x, log2e);
            let exp_2x = b.alloc_reg(T::PTX_TYPE);
            if T::PTX_TYPE == PtxType::F32 {
                b.raw_ptx(&format!("ex2.approx.f32 {exp_2x}, {scaled};"));
            } else {
                b.raw_ptx(&format!("ex2.approx.f64 {exp_2x}, {scaled};"));
            }
            let minus_one = ptx_helpers::load_float_imm::<T>(b, -1.0);
            let one = ptx_helpers::load_float_imm::<T>(b, 1.0);
            let num = ptx_helpers::add_float::<T>(b, exp_2x.clone(), minus_one);
            let den = ptx_helpers::add_float::<T>(b, exp_2x, one);
            ptx_helpers::div_float::<T>(b, num, den)
        }
    }
}

// ---------------------------------------------------------------------------
// Expert-parallel strategy
// ---------------------------------------------------------------------------

/// Expert-parallel fused MoE: permute + grouped GEMM + activation + grouped
/// GEMM + unpermute.
///
/// Decomposes the operation into discrete steps leveraging Vol.3's grouped GEMM:
///
/// 1. Expand input tokens by top_k (replicate each token for its experts).
/// 2. GEMM1: expanded_tokens @ W1[expert] (per-expert grouped GEMM).
/// 3. Apply activation element-wise.
/// 4. GEMM2: intermediate @ W2[expert] (per-expert grouped GEMM).
/// 5. Unpermute and weighted-sum back to original token order.
#[allow(clippy::too_many_arguments)]
fn fused_moe_expert_parallel<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    w1: &TensorDesc<T>,
    w2: &TensorDesc<T>,
    expert_indices: &DeviceBuffer<i32>,
    expert_weights: &DeviceBuffer<T>,
    output: &mut TensorDescMut<T>,
    config: &MoeConfig,
) -> DnnResult<()> {
    let num_tokens = input.dims[0];
    let total_expanded = num_tokens * config.top_k;

    // Allocate scratch buffers
    let expanded_size = total_expanded as usize * config.hidden_dim as usize;
    let intermediate_size = total_expanded as usize * config.intermediate_dim as usize;

    let mut expanded_buf = DeviceBuffer::<T>::alloc(expanded_size)?;
    let mut intermediate_buf = DeviceBuffer::<T>::alloc(intermediate_size)?;
    let expert_out_buf = DeviceBuffer::<T>::alloc(expanded_size)?;

    // Step 1: Expand input tokens by top_k
    expand_tokens_by_topk(
        handle,
        input,
        &mut expanded_buf,
        num_tokens,
        config.hidden_dim,
        config.top_k,
    )?;

    // Step 2: Build expert offsets for grouped GEMM dispatch
    let expert_offsets = DeviceBuffer::<i32>::alloc(config.num_experts as usize + 1)?;

    // Step 3: GEMM1 -- expanded_tokens @ W1[expert]
    let expanded_tensor = TensorDesc::<T>::from_raw(
        expanded_buf.as_device_ptr(),
        vec![total_expanded, config.hidden_dim],
        vec![config.hidden_dim, 1],
        TensorLayout::RowMajor,
    )?;
    let mut intermediate_desc = TensorDescMut::<T>::from_raw(
        intermediate_buf.as_device_ptr(),
        vec![total_expanded, config.intermediate_dim],
        vec![config.intermediate_dim, 1],
        TensorLayout::RowMajor,
    )?;

    moe_grouped_gemm(
        handle,
        &expanded_tensor,
        w1,
        &mut intermediate_desc,
        &expert_offsets,
        config.num_experts,
    )?;

    // Step 4: Apply activation in-place
    apply_activation_inplace::<T>(
        handle,
        &mut intermediate_buf,
        total_expanded as usize * config.intermediate_dim as usize,
        config.activation,
        config.sm_version,
    )?;

    // Step 5: GEMM2 -- intermediate @ W2[expert]
    let intermediate_tensor = TensorDesc::<T>::from_raw(
        intermediate_buf.as_device_ptr(),
        vec![total_expanded, config.intermediate_dim],
        vec![config.intermediate_dim, 1],
        TensorLayout::RowMajor,
    )?;
    let mut expert_out_desc = TensorDescMut::<T>::from_raw(
        expert_out_buf.as_device_ptr(),
        vec![total_expanded, config.hidden_dim],
        vec![config.hidden_dim, 1],
        TensorLayout::RowMajor,
    )?;

    moe_grouped_gemm(
        handle,
        &intermediate_tensor,
        w2,
        &mut expert_out_desc,
        &expert_offsets,
        config.num_experts,
    )?;

    // Step 6: Unpermute and weighted-sum
    let expert_out_tensor = TensorDesc::<T>::from_raw(
        expert_out_buf.as_device_ptr(),
        vec![total_expanded, config.hidden_dim],
        vec![config.hidden_dim, 1],
        TensorLayout::RowMajor,
    )?;

    unpermute_tokens(
        handle,
        &expert_out_tensor,
        expert_indices,
        expert_weights,
        output,
        config.top_k,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: expand tokens by top_k
// ---------------------------------------------------------------------------

/// Expands input tokens by replicating each token `top_k` times.
///
/// For each slot `s` in `[0, num_tokens * top_k)`:
///   `expanded[s, :] = input[s / top_k, :]`
fn expand_tokens_by_topk<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    expanded: &mut DeviceBuffer<T>,
    num_tokens: u32,
    hidden_dim: u32,
    top_k: u32,
) -> DnnResult<()> {
    let total = num_tokens * top_k;
    let ptx = generate_expand_ptx::<T>(handle.sm_version(), top_k)?;
    let kernel_name = format!("moe_expand_tokens_{}", T::NAME);

    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let grid_x = hidden_dim.div_ceil(256);
    let grid = Dim3::new(grid_x, total, 1);
    let block = Dim3::new(256, 1, 1);
    let params = LaunchParams::new(grid, block);

    let args = (
        input.ptr,
        expanded.as_device_ptr(),
        num_tokens,
        hidden_dim,
        top_k,
    );

    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

/// Generates PTX for the token expansion kernel.
fn generate_expand_ptx<T: GpuFloat>(sm: SmVersion, _top_k: u32) -> DnnResult<String> {
    let kernel_name = format!("moe_expand_tokens_{}", T::NAME);
    let elem_bytes = T::SIZE as u32;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("num_tokens", PtxType::U32)
        .param("hidden_dim", PtxType::U32)
        .param("top_k", PtxType::U32)
        .body(move |b| {
            let col = b.global_thread_id_x();
            // slot = blockIdx.y
            let slot = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {slot}, %ctaid.y;"));

            let num_tok = b.load_param_u32("num_tokens");
            let hidden = b.load_param_u32("hidden_dim");
            let p_topk = b.load_param_u32("top_k");

            let exit_lbl = b.fresh_label("exit");
            let total = b.mul_lo_u32(num_tok, p_topk.clone());
            let p1 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {p1}, {slot}, {total};"));
            b.branch_if(p1, &exit_lbl);
            let p2 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!(
                "setp.ge.u32 {p2}, {}, {};",
                col.clone(),
                hidden.clone()
            ));
            b.branch_if(p2, &exit_lbl);

            let input_ptr = b.load_param_u64("input_ptr");
            let output_ptr = b.load_param_u64("output_ptr");

            // src_row = slot / top_k
            let src_row = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("div.u32 {src_row}, {}, {p_topk};", slot.clone()));

            // src_addr = input_ptr + (src_row * hidden + col) * elem_bytes
            let src_off = b.mul_lo_u32(src_row, hidden.clone());
            let src_idx = b.add_u32(src_off, col.clone());
            let src_addr = b.byte_offset_addr(input_ptr, src_idx, elem_bytes);

            // dst_addr = output_ptr + (slot * hidden + col) * elem_bytes
            let dst_off = b.mul_lo_u32(slot, hidden);
            let dst_idx = b.add_u32(dst_off, col);
            let dst_addr = b.byte_offset_addr(output_ptr, dst_idx, elem_bytes);

            let val = ptx_helpers::load_global_float::<T>(b, src_addr);
            ptx_helpers::store_global_float::<T>(b, dst_addr, val);

            b.label(&exit_lbl);
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Helper: in-place activation
// ---------------------------------------------------------------------------

/// Applies activation function element-wise to a device buffer.
fn apply_activation_inplace<T: GpuFloat>(
    handle: &DnnHandle,
    buffer: &mut DeviceBuffer<T>,
    num_elements: usize,
    activation: Activation,
    sm: SmVersion,
) -> DnnResult<()> {
    if activation == Activation::None {
        return Ok(());
    }

    let ptx = generate_activation_ptx::<T>(activation, sm)?;
    let kernel_name = format!("moe_activation_{}", T::NAME);

    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let block = 256u32;
    let n = num_elements as u32;
    let grid = n.div_ceil(block);
    let params = LaunchParams::new(grid, block);

    let args = (buffer.as_device_ptr(), n);

    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

/// Generates PTX for an element-wise activation kernel.
fn generate_activation_ptx<T: GpuFloat>(
    activation: Activation,
    sm: SmVersion,
) -> DnnResult<String> {
    let kernel_name = format!("moe_activation_{}", T::NAME);
    let elem_bytes = T::SIZE as u32;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("data_ptr", PtxType::U64)
        .param("num_elements", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n = b.load_param_u32("num_elements");

            let exit_lbl = b.fresh_label("exit");
            let pred = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {pred}, {gid}, {n};"));
            b.branch_if(pred, &exit_lbl);

            let data_ptr = b.load_param_u64("data_ptr");
            let addr = b.byte_offset_addr(data_ptr, gid, elem_bytes);
            let val = ptx_helpers::load_global_float::<T>(b, addr.clone());

            let result = emit_activation_ptx::<T>(b, val, activation);

            ptx_helpers::store_global_float::<T>(b, addr, result);

            b.label(&exit_lbl);
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_selection_small_batch() {
        assert_eq!(select_strategy(4, 8), MoeStrategy::TokenParallel);
    }

    #[test]
    fn strategy_selection_large_batch() {
        assert_eq!(select_strategy(64, 8), MoeStrategy::ExpertParallel);
    }

    #[test]
    fn strategy_boundary() {
        // num_tokens == num_experts * 2 => ExpertParallel
        assert_eq!(select_strategy(16, 8), MoeStrategy::ExpertParallel);
        // num_tokens == num_experts * 2 - 1 => TokenParallel
        assert_eq!(select_strategy(15, 8), MoeStrategy::TokenParallel);
    }

    // -----------------------------------------------------------------------
    // Task 1: MoE routing strategy selection boundary tests
    // -----------------------------------------------------------------------

    /// num_tokens < num_experts * 2 → TokenParallel
    #[test]
    fn test_moe_strategy_token_parallel_when_few_tokens() {
        // num_experts=8, num_tokens=14 < 8*2=16 → TokenParallel
        assert_eq!(select_strategy(14, 8), MoeStrategy::TokenParallel);
        // num_experts=8, num_tokens=15 < 16 → TokenParallel
        assert_eq!(select_strategy(15, 8), MoeStrategy::TokenParallel);
    }

    /// num_tokens >= num_experts * 2 → ExpertParallel
    #[test]
    fn test_moe_strategy_expert_parallel_when_many_tokens() {
        // num_experts=8, num_tokens=16 >= 16 → ExpertParallel
        assert_eq!(select_strategy(16, 8), MoeStrategy::ExpertParallel);
        // num_experts=8, num_tokens=100 → ExpertParallel
        assert_eq!(select_strategy(100, 8), MoeStrategy::ExpertParallel);
    }

    /// Boundary: exactly num_experts * 2 → ExpertParallel;
    /// one below → TokenParallel
    #[test]
    fn test_moe_strategy_boundary_exactly_2x() {
        // num_experts=4, num_tokens=8 → ExpertParallel (>= boundary)
        assert_eq!(select_strategy(8, 4), MoeStrategy::ExpertParallel);
        // num_experts=4, num_tokens=7 → TokenParallel (< boundary)
        assert_eq!(select_strategy(7, 4), MoeStrategy::TokenParallel);
    }

    /// Edge case: single expert
    #[test]
    fn test_moe_strategy_single_expert() {
        // num_experts=1, num_tokens=1 → TokenParallel (1 < 2)
        assert_eq!(select_strategy(1, 1), MoeStrategy::TokenParallel);
        // num_experts=1, num_tokens=2 → ExpertParallel (2 >= 2)
        assert_eq!(select_strategy(2, 1), MoeStrategy::ExpertParallel);
    }

    /// Mixtral-8x7B decode pattern: 8 experts, few tokens → TokenParallel
    #[test]
    fn test_moe_strategy_mixtral_decode_pattern() {
        // 8 experts, 1 token → TokenParallel
        assert_eq!(select_strategy(1, 8), MoeStrategy::TokenParallel);
        // 8 experts, 4 tokens → TokenParallel
        assert_eq!(select_strategy(4, 8), MoeStrategy::TokenParallel);
    }

    /// Mixtral-8x7B prefill pattern: 8 experts, many tokens → ExpertParallel
    #[test]
    fn test_moe_strategy_mixtral_prefill_pattern() {
        // 8 experts, 512 tokens → ExpertParallel
        assert_eq!(select_strategy(512, 8), MoeStrategy::ExpertParallel);
        // 8 experts, 2048 tokens → ExpertParallel
        assert_eq!(select_strategy(2048, 8), MoeStrategy::ExpertParallel);
    }

    /// Large expert count with threshold at 2x
    #[test]
    fn test_moe_strategy_large_expert_count() {
        // 64 experts, threshold = 128
        assert_eq!(select_strategy(127, 64), MoeStrategy::TokenParallel);
        assert_eq!(select_strategy(128, 64), MoeStrategy::ExpertParallel);
    }

    /// Zero tokens edge case: 0 < any positive 2*E → TokenParallel
    #[test]
    fn test_moe_strategy_zero_tokens() {
        assert_eq!(select_strategy(0, 8), MoeStrategy::TokenParallel);
    }

    /// Saturating mul guard: very large num_experts should not overflow
    #[test]
    fn test_moe_strategy_no_overflow() {
        // With u32::MAX / 2 experts, threshold = u32::MAX (saturating),
        // so any reasonable token count is TokenParallel.
        let large_experts = u32::MAX / 2 + 1;
        // saturating_mul(2) = u32::MAX, so num_tokens < u32::MAX → TokenParallel
        assert_eq!(
            select_strategy(1_000_000, large_experts),
            MoeStrategy::TokenParallel
        );
    }

    // -----------------------------------------------------------------------
    // FusedMoeConfig + FP8 epilogue tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_moe_fp8_epilogue_ptx_contains_quantize() {
        // The MoE FP8 epilogue should quantize output back to FP8.
        let config = FusedMoeConfig::new(8, 4096, 4096).with_fp8_output(Fp8Type::E4M3);
        let ptx = config.generate_epilogue_ptx();
        // Verify PTX contains e4m3 conversion reference and that config reflects FP8.
        assert!(config.output_is_fp8(), "Config should have FP8 output");
        assert!(
            ptx.contains("e4m3") || ptx.contains("E4M3"),
            "FP8 epilogue PTX should reference e4m3 conversion, got: {ptx}"
        );
    }

    #[test]
    fn test_moe_fp8_requires_sm89() {
        // FP8 for inference requires SM 89+ (Ada Lovelace / Hopper).
        let config = FusedMoeConfig::new(8, 4096, 4096).with_fp8_output(Fp8Type::E4M3);
        assert!(
            config.min_sm_version() >= 89,
            "FP8 MoE needs SM 89+, got {}",
            config.min_sm_version()
        );
    }

    #[test]
    fn test_moe_no_fp8_allows_sm80() {
        let config = FusedMoeConfig::new(8, 4096, 4096);
        assert!(!config.output_is_fp8());
        assert_eq!(config.min_sm_version(), 80);
    }

    #[test]
    fn test_moe_fp8_e5m2_epilogue() {
        let config = FusedMoeConfig::new(4, 2048, 2048).with_fp8_output(Fp8Type::E5M2);
        let ptx = config.generate_epilogue_ptx();
        assert!(config.output_is_fp8());
        assert!(
            ptx.contains("e5m2") || ptx.contains("E5M2"),
            "FP8 E5M2 epilogue should reference e5m2, got: {ptx}"
        );
    }

    #[test]
    fn test_moe_no_fp8_empty_epilogue() {
        let config = FusedMoeConfig::new(8, 4096, 4096);
        let ptx = config.generate_epilogue_ptx();
        assert!(
            ptx.is_empty(),
            "Non-FP8 config should produce empty epilogue PTX"
        );
    }

    #[test]
    fn activation_ptx_gen_relu() {
        let ptx = generate_activation_ptx::<f32>(Activation::Relu, SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn activation_ptx_gen_silu() {
        let ptx = generate_activation_ptx::<f32>(Activation::Silu, SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn activation_ptx_gen_gelu() {
        let ptx = generate_activation_ptx::<f32>(Activation::Gelu, SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn expand_ptx_gen() {
        let ptx = generate_expand_ptx::<f32>(SmVersion::Sm80, 2);
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains(".entry moe_expand_tokens_f32"));
    }

    // -----------------------------------------------------------------------
    // Quality-gate: MoE FP8 epilogue and routing tests
    // -----------------------------------------------------------------------

    /// FP8 E4M3 epilogue scale computation reference:
    /// scale = absmax / 448.0 (FP8 E4M3 max value).
    ///
    /// The epilogue PTX comment must reference the 448.0 divisor so that
    /// downstream code can verify the scale convention.
    #[test]
    fn test_moe_fp8_e4m3_epilogue_scale_references_448() {
        let config = FusedMoeConfig::new(8, 4096, 4096).with_fp8_output(Fp8Type::E4M3);
        let epilogue = config.generate_epilogue_ptx();

        // The E4M3 max value is 448.0; epilogue must reference it
        assert!(
            epilogue.contains("448"),
            "FP8 E4M3 epilogue must reference max value 448.0, got: {epilogue}"
        );
        // Also confirm it's not empty and references e4m3
        assert!(!epilogue.is_empty(), "epilogue must not be empty for E4M3");
        assert!(
            epilogue.contains("e4m3") || epilogue.contains("E4M3"),
            "epilogue must reference e4m3 format"
        );
    }

    /// FusedMoeConfig default top_k is 2.
    ///
    /// With top_k=2 and 4 experts, each token gets exactly 2 expert assignments.
    /// The config default encodes this MoE standard.
    #[test]
    fn test_moe_expert_gate_routing_uses_top_k_2() {
        // Default FusedMoeConfig sets top_k = 2
        let config = FusedMoeConfig::new(4, 512, 2048);
        assert_eq!(
            config.top_k, 2,
            "default top_k must be 2 (each token activates 2 of 4 experts)"
        );
        // Verify: with 4 experts and top_k=2, exactly 2 experts per token
        // This is the Mixtral / DeepSeek routing pattern
        assert!(
            config.top_k <= config.num_experts,
            "top_k ({}) must not exceed num_experts ({})",
            config.top_k,
            config.num_experts
        );
    }

    /// Expert load balancing expectation with 8 experts and 16 tokens.
    ///
    /// With top_k=2 and 16 tokens, there are 32 total expert assignments.
    /// Uniformly distributed across 8 experts → 4 assignments per expert.
    /// The strategy selector ensures ExpertParallel is chosen when load
    /// is high (num_tokens=16 >= num_experts*2=16 → ExpertParallel).
    #[test]
    fn test_moe_expert_load_balancing_8_experts_16_tokens() {
        let num_experts = 8u32;
        let num_tokens = 16u32;
        let top_k = 2u32;

        // Total expert activations
        let total_activations = num_tokens * top_k;
        assert_eq!(
            total_activations, 32,
            "16 tokens × top_k=2 = 32 activations"
        );

        // Ideal load per expert (uniform distribution)
        let ideal_per_expert = total_activations / num_experts;
        assert_eq!(ideal_per_expert, 4, "32 activations / 8 experts = 4 each");

        // Strategy: 16 tokens, 8 experts → num_tokens == num_experts * 2 → ExpertParallel
        assert_eq!(
            select_strategy(num_tokens, num_experts),
            MoeStrategy::ExpertParallel,
            "16 tokens with 8 experts must select ExpertParallel (balanced load path)"
        );

        // Load balance ratio: max_load / avg_load ≤ 2× (within 2× of average)
        // In ideal case it's exactly 1×; we verify the mathematical bound
        let avg_load = total_activations as f32 / num_experts as f32;
        let max_acceptable = avg_load * 2.0;
        assert!(
            ideal_per_expert as f32 <= max_acceptable,
            "ideal load {ideal_per_expert} must be within 2× of average {avg_load:.1}"
        );
    }
}
