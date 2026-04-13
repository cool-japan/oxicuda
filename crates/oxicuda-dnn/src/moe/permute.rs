//! Token permutation and unpermutation for MoE dispatch.
//!
//! After routing assigns each token to its top-k experts, the input tokens
//! must be permuted (scattered) so that all tokens for the same expert are
//! contiguous in memory. After expert computation, the results are
//! unpermuted (gathered) back to original token order and combined with
//! the routing weights.
//!
//! # Kernels
//!
//! - **`permute_tokens`**: Scatters input rows according to the permutation
//!   array. Each thread copies one element of one row.
//! - **`unpermute_tokens`**: Gathers expert output rows back to original
//!   order, multiplying each by its routing weight, and accumulates across
//!   the top-k expert contributions.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::ptx_helpers;
use crate::types::{TensorDesc, TensorDescMut};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Block size (X dimension) for permutation kernels.
const PERM_BLOCK_X: u32 = 256;

// ---------------------------------------------------------------------------
// Permute
// ---------------------------------------------------------------------------

/// Generates PTX for the token permutation kernel.
///
/// Grid: `(ceil(hidden_dim/256), num_rows, 1)`. Each thread copies one element.
/// Kernel: `output[perm[row], col] = input[row, col]`.
fn generate_permute_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let kernel_name = format!("moe_permute_tokens_{}", T::NAME);
    let elem_bytes = T::SIZE as u32;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("perm_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("num_rows", PtxType::U32)
        .param("hidden_dim", PtxType::U32)
        .body(move |b| {
            // col = blockIdx.x * blockDim.x + threadIdx.x
            let col = b.global_thread_id_x();
            // row = blockIdx.y
            let row = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {row}, %ctaid.y;"));

            let num_rows = b.load_param_u32("num_rows");
            let hidden = b.load_param_u32("hidden_dim");

            // Bounds check: row < num_rows && col < hidden_dim
            let exit_lbl = b.fresh_label("exit");
            let pred_row = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {pred_row}, {row}, {num_rows};"));
            b.branch_if(pred_row, &exit_lbl);
            let pred_col = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {pred_col}, {col}, {hidden};"));
            b.branch_if(pred_col, &exit_lbl);

            let input_ptr = b.load_param_u64("input_ptr");
            let perm_ptr = b.load_param_u64("perm_ptr");
            let output_ptr = b.load_param_u64("output_ptr");

            // Load permutation[row] => dest_row
            let perm_addr = b.byte_offset_addr(perm_ptr, row.clone(), 4);
            let dest_row = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("ld.global.u32 {dest_row}, [{perm_addr}];"));

            // src_addr = input_ptr + (row * hidden + col) * elem_bytes
            let row_off = b.mul_lo_u32(row, hidden.clone());
            let src_idx = b.add_u32(row_off, col.clone());
            let src_addr = b.byte_offset_addr(input_ptr, src_idx, elem_bytes);

            // dst_addr = output_ptr + (dest_row * hidden + col) * elem_bytes
            let dst_off = b.mul_lo_u32(dest_row, hidden);
            let dst_idx = b.add_u32(dst_off, col);
            let dst_addr = b.byte_offset_addr(output_ptr, dst_idx, elem_bytes);

            // Copy element
            let val = ptx_helpers::load_global_float::<T>(b, src_addr);
            ptx_helpers::store_global_float::<T>(b, dst_addr, val);

            b.label(&exit_lbl);
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Permutes tokens by expert assignment.
///
/// Rearranges input rows according to the permutation array so that tokens
/// assigned to the same expert are contiguous in the output buffer.
///
/// # Arguments
///
/// * `handle` -- DNN handle providing stream and context.
/// * `input` -- Input tensor of shape `[num_rows, hidden_dim]`.
/// * `permutation` -- Permutation indices of length `num_rows`.
/// * `output` -- Output tensor of shape `[num_rows, hidden_dim]`.
///
/// # Errors
///
/// Returns [`DnnError`] on dimension mismatch or kernel launch failure.
pub fn permute_tokens<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    permutation: &DeviceBuffer<i32>,
    output: &mut TensorDescMut<T>,
) -> DnnResult<()> {
    // Validate dimensions
    if input.ndim() != 2 {
        return Err(DnnError::InvalidDimension(format!(
            "input must be 2D, got {}D",
            input.ndim()
        )));
    }
    if output.ndim() != 2 {
        return Err(DnnError::InvalidDimension(format!(
            "output must be 2D, got {}D",
            output.ndim()
        )));
    }

    let num_rows = input.dims[0];
    let hidden_dim = input.dims[1];

    if output.dims[0] != num_rows || output.dims[1] != hidden_dim {
        return Err(DnnError::InvalidDimension(format!(
            "output dims [{}, {}] must match input dims [{}, {}]",
            output.dims[0], output.dims[1], num_rows, hidden_dim
        )));
    }
    if permutation.len() < num_rows as usize {
        return Err(DnnError::BufferTooSmall {
            expected: num_rows as usize,
            actual: permutation.len(),
        });
    }

    let ptx = generate_permute_ptx::<T>(handle.sm_version())?;
    let kernel_name = format!("moe_permute_tokens_{}", T::NAME);

    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let grid_x = hidden_dim.div_ceil(PERM_BLOCK_X);
    let grid = Dim3::new(grid_x, num_rows, 1);
    let block = Dim3::new(PERM_BLOCK_X, 1, 1);
    let params = LaunchParams::new(grid, block);

    let args = (
        input.ptr,
        permutation.as_device_ptr(),
        output.ptr,
        num_rows,
        hidden_dim,
    );

    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Unpermute
// ---------------------------------------------------------------------------

/// Generates PTX for the weighted unpermutation kernel.
///
/// Grid: `(ceil(hidden_dim/256), num_tokens, 1)`.
///
/// For each token `t` and column `c`:
/// ```text
/// output[t, c] = sum_{j=0}^{top_k-1} weight[t*top_k+j] * expert_out[perm[t*top_k+j], c]
/// ```
fn generate_unpermute_ptx<T: GpuFloat>(sm: SmVersion, top_k: u32) -> DnnResult<String> {
    let kernel_name = format!("moe_unpermute_tokens_{}", T::NAME);
    let elem_bytes = T::SIZE as u32;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("expert_out_ptr", PtxType::U64)
        .param("perm_ptr", PtxType::U64)
        .param("weights_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("num_tokens", PtxType::U32)
        .param("hidden_dim", PtxType::U32)
        .param("top_k", PtxType::U32)
        .body(move |b| {
            let col = b.global_thread_id_x();
            let token = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {token}, %ctaid.y;"));

            let num_tok = b.load_param_u32("num_tokens");
            let hidden = b.load_param_u32("hidden_dim");
            let p_topk = b.load_param_u32("top_k");

            let exit_lbl = b.fresh_label("exit");
            let pred_tok = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {pred_tok}, {token}, {num_tok};"));
            b.branch_if(pred_tok, &exit_lbl);
            let pred_col = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {pred_col}, {col}, {hidden};"));
            b.branch_if(pred_col, &exit_lbl);

            let expert_out = b.load_param_u64("expert_out_ptr");
            let perm_ptr = b.load_param_u64("perm_ptr");
            let weights_ptr = b.load_param_u64("weights_ptr");
            let output_ptr = b.load_param_u64("output_ptr");

            // acc = 0.0
            let acc = ptx_helpers::load_float_imm::<T>(b, 0.0);

            // base_slot = token * top_k
            let base_slot = b.mul_lo_u32(token.clone(), p_topk);

            // Unrolled loop over top_k experts
            b.unroll(top_k, |b, j| {
                let j_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {j_reg}, {j};"));
                let slot = b.add_u32(base_slot.clone(), j_reg);

                // Load weight[slot]
                let wt_addr = b.byte_offset_addr(weights_ptr.clone(), slot.clone(), elem_bytes);
                let weight = ptx_helpers::load_global_float::<T>(b, wt_addr);

                // Load perm[slot] => src_row
                let perm_addr = b.byte_offset_addr(perm_ptr.clone(), slot, 4);
                let src_row = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("ld.global.u32 {src_row}, [{perm_addr}];"));

                // Load expert_out[src_row * hidden + col]
                let row_off = b.mul_lo_u32(src_row, hidden.clone());
                let idx = b.add_u32(row_off, col.clone());
                let src_addr = b.byte_offset_addr(expert_out.clone(), idx, elem_bytes);
                let val = ptx_helpers::load_global_float::<T>(b, src_addr);

                // acc += weight * val
                let contribution = ptx_helpers::fma_float::<T>(b, weight, val, acc.clone());
                let ty_str = if T::PTX_TYPE == PtxType::F32 {
                    "f32"
                } else {
                    "f64"
                };
                b.raw_ptx(&format!("mov.{ty_str} {}, {contribution};", acc.clone()));
            });

            // Store output[token * hidden + col] = acc
            let out_off = b.mul_lo_u32(token, hidden);
            let out_idx = b.add_u32(out_off, col);
            let out_addr = b.byte_offset_addr(output_ptr, out_idx, elem_bytes);
            ptx_helpers::store_global_float::<T>(b, out_addr, acc);

            b.label(&exit_lbl);
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Unpermutes and weighted-sums expert outputs back to original token order.
///
/// For each output token, accumulates contributions from its top-k expert
/// outputs, weighted by the routing scores:
///
/// ```text
/// output[t, :] = sum_{j=0}^{top_k-1} weight[t*top_k+j] * expert_out[perm[t*top_k+j], :]
/// ```
///
/// # Arguments
///
/// * `handle` -- DNN handle providing stream and context.
/// * `expert_output` -- Expert computation results, shape
///   `[num_tokens * top_k, hidden_dim]`.
/// * `permutation` -- Permutation indices of length `num_tokens * top_k`.
/// * `expert_weights` -- Routing weights of length `num_tokens * top_k`.
/// * `output` -- Output tensor of shape `[num_tokens, hidden_dim]`.
/// * `top_k` -- Number of experts per token.
///
/// # Errors
///
/// Returns [`DnnError`] on dimension mismatch or kernel launch failure.
pub fn unpermute_tokens<T: GpuFloat>(
    handle: &DnnHandle,
    expert_output: &TensorDesc<T>,
    permutation: &DeviceBuffer<i32>,
    expert_weights: &DeviceBuffer<T>,
    output: &mut TensorDescMut<T>,
    top_k: u32,
) -> DnnResult<()> {
    if expert_output.ndim() != 2 {
        return Err(DnnError::InvalidDimension(format!(
            "expert_output must be 2D, got {}D",
            expert_output.ndim()
        )));
    }
    if output.ndim() != 2 {
        return Err(DnnError::InvalidDimension(format!(
            "output must be 2D, got {}D",
            output.ndim()
        )));
    }
    if top_k == 0 {
        return Err(DnnError::InvalidArgument("top_k must be positive".into()));
    }

    let num_tokens = output.dims[0];
    let hidden_dim = output.dims[1];

    if expert_output.dims[1] != hidden_dim {
        return Err(DnnError::InvalidDimension(format!(
            "expert_output hidden_dim ({}) != output hidden_dim ({})",
            expert_output.dims[1], hidden_dim
        )));
    }

    let total_slots = num_tokens as usize * top_k as usize;
    if permutation.len() < total_slots {
        return Err(DnnError::BufferTooSmall {
            expected: total_slots,
            actual: permutation.len(),
        });
    }
    if expert_weights.len() < total_slots {
        return Err(DnnError::BufferTooSmall {
            expected: total_slots,
            actual: expert_weights.len(),
        });
    }

    let ptx = generate_unpermute_ptx::<T>(handle.sm_version(), top_k)?;
    let kernel_name = format!("moe_unpermute_tokens_{}", T::NAME);

    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let grid_x = hidden_dim.div_ceil(PERM_BLOCK_X);
    let grid = Dim3::new(grid_x, num_tokens, 1);
    let block = Dim3::new(PERM_BLOCK_X, 1, 1);
    let params = LaunchParams::new(grid, block);

    let args = (
        expert_output.ptr,
        permutation.as_device_ptr(),
        expert_weights.as_device_ptr(),
        output.ptr,
        num_tokens,
        hidden_dim,
        top_k,
    );

    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permute_ptx_gen_f32() {
        let ptx = generate_permute_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains(".entry moe_permute_tokens_f32"));
    }

    #[test]
    fn unpermute_ptx_gen_f32() {
        let ptx = generate_unpermute_ptx::<f32>(SmVersion::Sm80, 2);
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains(".entry moe_unpermute_tokens_f32"));
    }
}
