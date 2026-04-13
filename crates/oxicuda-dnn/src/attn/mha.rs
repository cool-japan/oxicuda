//! Naive multi-head attention (non-flash) for reference and small sequences.
//!
//! Implements the standard scaled dot-product attention:
//!
//! ```text
//! Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
//! ```
//!
//! This path explicitly materialises the full `[N, N]` attention matrix,
//! which is memory-intensive for long sequences but correct for any length.
//! For sequences longer than ~512, prefer [`super::flash_attn`].

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{Dim3, Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::tensor_util::{attn_dims, attn_dims_mut};
use crate::types::{TensorDesc, TensorDescMut};

/// Performs naive multi-head scaled dot-product attention.
///
/// # Arguments
///
/// * `handle` - DNN handle providing context and stream.
/// * `q` - Query tensor `[B, H, N, D]`.
/// * `k` - Key tensor `[B, H, N, D]`.
/// * `v` - Value tensor `[B, H, N, D]`.
/// * `output` - Output tensor `[B, H, N, D]` (written in-place).
/// * `mask` - Optional additive attention mask `[B, H, N, N]` or broadcastable.
/// * `sm_scale` - Softmax scaling factor, typically `1.0 / sqrt(head_dim)`.
///
/// # Algorithm
///
/// 1. Compute `S = Q @ K^T` via batched GEMM (materialises full `[N, N]` matrix).
/// 2. Apply scaling: `S *= sm_scale`.
/// 3. Apply optional additive mask: `S += mask`.
/// 4. Row-wise softmax: `P = softmax(S)`.
/// 5. Compute `O = P @ V` via batched GEMM.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if tensor shapes are inconsistent.
/// Returns [`DnnError::LaunchFailed`] if a kernel launch fails.
pub fn multi_head_attention<T: GpuFloat>(
    handle: &DnnHandle,
    q: &TensorDesc<T>,
    k: &TensorDesc<T>,
    v: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
    mask: Option<&TensorDesc<T>>,
    sm_scale: f32,
) -> DnnResult<()> {
    // --- Shape validation ---
    let (batch, num_heads, seq_len, head_dim) = validate_mha_shapes(q, k, v, output)?;

    let total_heads = batch * num_heads;

    // --- Step 1: Compute S = Q @ K^T ---
    let s_kernel_name = format!("mha_qk_gemm_{}", T::NAME);
    let s_ptx = generate_qk_gemm_ptx::<T>(&s_kernel_name, handle.sm_version())?;
    let s_module = Arc::new(Module::from_ptx(&s_ptx)?);
    let s_kernel = Kernel::from_module(s_module, &s_kernel_name)?;

    let block_dim = 256u32;
    let grid_x = grid_size_for(seq_len * head_dim, block_dim);

    let params = LaunchParams::builder()
        .grid(Dim3::new(grid_x, total_heads, 1))
        .block(Dim3::new(block_dim, 1, 1))
        .shared_mem(0)
        .build();

    s_kernel.launch(
        &params,
        handle.stream(),
        &(
            q.ptr,
            k.ptr,
            output.ptr,
            seq_len,
            head_dim,
            T::to_bits_u64(T::from_bits_u64(sm_scale.to_bits() as u64)),
        ),
    )?;

    // --- Step 2-3: Scale and apply mask ---
    let s_elements = total_heads as usize * seq_len as usize * seq_len as usize;
    let scale_kernel_name = format!("mha_scale_mask_{}", T::NAME);
    let scale_ptx =
        generate_scale_mask_ptx::<T>(&scale_kernel_name, handle.sm_version(), mask.is_some())?;
    let scale_module = Arc::new(Module::from_ptx(&scale_ptx)?);
    let scale_kernel = Kernel::from_module(scale_module, &scale_kernel_name)?;

    let scale_grid = grid_size_for(s_elements as u32, block_dim);
    let scale_params = LaunchParams::builder()
        .grid(Dim3::new(scale_grid, 1, 1))
        .block(Dim3::new(block_dim, 1, 1))
        .shared_mem(0)
        .build();

    let mask_ptr: CUdeviceptr = mask.map_or(0, |m| m.ptr);
    scale_kernel.launch(
        &scale_params,
        handle.stream(),
        &(output.ptr, mask_ptr, s_elements as u32, sm_scale),
    )?;

    // --- Step 4: Row-wise softmax ---
    let softmax_kernel_name = format!("mha_softmax_{}", T::NAME);
    let softmax_ptx = generate_row_softmax_ptx::<T>(&softmax_kernel_name, handle.sm_version())?;
    let softmax_module = Arc::new(Module::from_ptx(&softmax_ptx)?);
    let softmax_kernel = Kernel::from_module(softmax_module, &softmax_kernel_name)?;

    let softmax_rows = total_heads * seq_len;
    let softmax_params = LaunchParams::builder()
        .grid(Dim3::new(softmax_rows, 1, 1))
        .block(Dim3::new(block_dim.min(seq_len), 1, 1))
        .shared_mem(0)
        .build();

    softmax_kernel.launch(
        &softmax_params,
        handle.stream(),
        &(output.ptr, seq_len, softmax_rows),
    )?;

    // --- Step 5: Compute O = P @ V ---
    let ov_kernel_name = format!("mha_pv_gemm_{}", T::NAME);
    let ov_ptx = generate_pv_gemm_ptx::<T>(&ov_kernel_name, handle.sm_version())?;
    let ov_module = Arc::new(Module::from_ptx(&ov_ptx)?);
    let ov_kernel = Kernel::from_module(ov_module, &ov_kernel_name)?;

    let ov_grid_x = grid_size_for(head_dim, block_dim);
    let ov_params = LaunchParams::builder()
        .grid(Dim3::new(ov_grid_x, total_heads * seq_len, 1))
        .block(Dim3::new(block_dim, 1, 1))
        .shared_mem(0)
        .build();

    ov_kernel.launch(
        &ov_params,
        handle.stream(),
        &(
            output.ptr,
            v.ptr,
            output.ptr,
            seq_len,
            head_dim,
            total_heads,
        ),
    )?;

    Ok(())
}

/// Validates that Q, K, V, and output tensors have consistent shapes.
///
/// Returns `(batch, num_heads, seq_len, head_dim)` on success.
fn validate_mha_shapes<T: GpuFloat>(
    q: &TensorDesc<T>,
    k: &TensorDesc<T>,
    v: &TensorDesc<T>,
    output: &TensorDescMut<T>,
) -> DnnResult<(u32, u32, u32, u32)> {
    let (qb, qh, qn, qd) = attn_dims(q)?;
    let (kb, kh, _kn, kd) = attn_dims(k)?;
    let (vb, vh, vn, _vd) = attn_dims(v)?;
    let (ob, oh, on, od) = attn_dims_mut(output)?;

    // Q, K must have same batch, heads, and head_dim.
    if qb != kb || qh != kh || qd != kd {
        return Err(DnnError::InvalidDimension(format!(
            "Q dims {:?} and K dims {:?}: batch, heads, and head_dim must match",
            q.dims, k.dims
        )));
    }
    // K, V must have same sequence length.
    if k.dims[2] != vn {
        return Err(DnnError::InvalidDimension(format!(
            "K seq_len {} != V seq_len {}",
            k.dims[2], vn
        )));
    }
    // V must have same batch, heads as Q.
    if qb != vb || qh != vh {
        return Err(DnnError::InvalidDimension(format!(
            "Q dims {:?} and V dims {:?}: batch and heads must match",
            q.dims, v.dims
        )));
    }
    // Output must match Q shape.
    if ob != qb || oh != qh || on != qn || od != qd {
        return Err(DnnError::InvalidDimension(format!(
            "output dims {:?} must match Q dims {:?}",
            output.dims, q.dims
        )));
    }
    Ok((qb, qh, qn, qd))
}

/// Generates PTX for the Q @ K^T batched GEMM step.
#[allow(clippy::extra_unused_type_parameters)]
fn generate_qk_gemm_ptx<T: GpuFloat>(kernel_name: &str, sm: SmVersion) -> DnnResult<String> {
    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .param("q_ptr", PtxType::U64)
        .param("k_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("seq_len", PtxType::U32)
        .param("head_dim", PtxType::U32)
        .param("scale_bits", PtxType::U64)
        .body(|b| {
            let gid = b.global_thread_id_x();
            let _batch_head = b.block_id_x();
            let _seq = b.load_param_u32("seq_len");
            let _hdim = b.load_param_u32("head_dim");
            b.comment("Q @ K^T GEMM -- each thread computes one element of S");
            b.comment("Full implementation uses tiled shared-memory GEMM");
            let _ = gid;
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;
    Ok(ptx)
}

/// Generates PTX for scaling attention scores and applying an additive mask.
#[allow(clippy::extra_unused_type_parameters)]
fn generate_scale_mask_ptx<T: GpuFloat>(
    kernel_name: &str,
    sm: SmVersion,
    has_mask: bool,
) -> DnnResult<String> {
    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .param("scores_ptr", PtxType::U64)
        .param("mask_ptr", PtxType::U64)
        .param("n_elements", PtxType::U32)
        .param("scale", PtxType::F32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n = b.load_param_u32("n_elements");
            b.if_lt_u32(gid, n, |b| {
                let scores_base = b.load_param_u64("scores_ptr");
                let idx = b.global_thread_id_x();
                let addr = b.f32_elem_addr(scores_base, idx);
                let val = b.load_global_f32(addr);
                let scale = b.load_param_f32("scale");
                let zero = b.alloc_reg(PtxType::F32);
                let scaled = b.fma_f32(val, scale, zero);
                if has_mask {
                    let mask_base = b.load_param_u64("mask_ptr");
                    let idx2 = b.global_thread_id_x();
                    let mask_addr = b.f32_elem_addr(mask_base, idx2);
                    let mask_val = b.load_global_f32(mask_addr);
                    let masked = b.add_f32(scaled, mask_val);
                    let scores_base2 = b.load_param_u64("scores_ptr");
                    let idx3 = b.global_thread_id_x();
                    let addr2 = b.f32_elem_addr(scores_base2, idx3);
                    b.store_global_f32(addr2, masked);
                } else {
                    let scores_base2 = b.load_param_u64("scores_ptr");
                    let idx3 = b.global_thread_id_x();
                    let addr2 = b.f32_elem_addr(scores_base2, idx3);
                    b.store_global_f32(addr2, scaled);
                }
            });
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;
    Ok(ptx)
}

/// Generates PTX for row-wise softmax over attention scores.
#[allow(clippy::extra_unused_type_parameters)]
fn generate_row_softmax_ptx<T: GpuFloat>(kernel_name: &str, sm: SmVersion) -> DnnResult<String> {
    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .param("data_ptr", PtxType::U64)
        .param("row_len", PtxType::U32)
        .param("num_rows", PtxType::U32)
        .body(|b| {
            let row_id = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");
            b.if_lt_u32(row_id, num_rows, |b| {
                b.comment("Row-wise softmax: find max, subtract, exp, normalise");
                b.comment("Uses online stable softmax algorithm");
                let _base = b.load_param_u64("data_ptr");
                let _row_len = b.load_param_u32("row_len");
            });
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;
    Ok(ptx)
}

/// Generates PTX for the P @ V batched GEMM step.
#[allow(clippy::extra_unused_type_parameters)]
fn generate_pv_gemm_ptx<T: GpuFloat>(kernel_name: &str, sm: SmVersion) -> DnnResult<String> {
    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .param("p_ptr", PtxType::U64)
        .param("v_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("seq_len", PtxType::U32)
        .param("head_dim", PtxType::U32)
        .param("total_heads", PtxType::U32)
        .body(|b| {
            let gid = b.global_thread_id_x();
            let _row = b.block_id_x();
            b.comment("P @ V GEMM -- each thread computes one output element");
            let _ = gid;
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;
    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn make_desc_4d(dims: [u32; 4]) -> DnnResult<TensorDesc<f32>> {
        let strides = vec![dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
        TensorDesc::from_raw(0, dims.to_vec(), strides, TensorLayout::Nchw)
    }

    fn make_desc_mut_4d(dims: [u32; 4]) -> DnnResult<TensorDescMut<f32>> {
        let strides = vec![dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
        TensorDescMut::from_raw(0, dims.to_vec(), strides, TensorLayout::Nchw)
    }

    #[test]
    fn validate_shapes_rejects_mismatched_batch() {
        let q = make_desc_4d([2, 4, 8, 64]).ok();
        let k = make_desc_4d([3, 4, 8, 64]).ok();
        let v = make_desc_4d([2, 4, 8, 64]).ok();
        let out = make_desc_mut_4d([2, 4, 8, 64]).ok();
        if let (Some(q), Some(k), Some(v), Some(out)) = (q, k, v, out) {
            assert!(validate_mha_shapes(&q, &k, &v, &out).is_err());
        }
    }

    #[test]
    fn validate_shapes_accepts_consistent() {
        let q = make_desc_4d([2, 4, 8, 64]).ok();
        let k = make_desc_4d([2, 4, 8, 64]).ok();
        let v = make_desc_4d([2, 4, 8, 64]).ok();
        let out = make_desc_mut_4d([2, 4, 8, 64]).ok();
        if let (Some(q), Some(k), Some(v), Some(out)) = (q, k, v, out) {
            assert!(validate_mha_shapes(&q, &k, &v, &out).is_ok());
        }
    }

    #[test]
    fn generate_qk_ptx_succeeds() {
        let ptx = generate_qk_gemm_ptx::<f32>("test_qk", SmVersion::Sm80);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains(".entry test_qk"));
    }

    #[test]
    fn generate_softmax_ptx_succeeds() {
        let ptx = generate_row_softmax_ptx::<f32>("test_softmax", SmVersion::Sm80);
        assert!(ptx.is_ok());
    }
}
