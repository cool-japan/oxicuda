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

use oxicuda_blas::GpuFloat;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::builder::BodyBuilder;
use oxicuda_ptx::ir::Register;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::kernel_cache::{cache_key, cache_key_with};
use crate::ptx_helpers::{
    load_float_imm, load_global_float, mul_float, store_global_float, sub_float,
};
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
    let block_dim = 256u32;

    // Scratch buffer for the S/P attention-score matrix, `[total_heads,
    // seq_len, seq_len]` flattened. This is generally a *different* size
    // than the `[total_heads, seq_len, head_dim]` output tensor (unless
    // seq_len == head_dim) -- reusing `output.ptr` to hold S would silently
    // write out of bounds of the output allocation for any other shape, so a
    // dedicated allocation is used instead (freed automatically when this
    // function returns; `cuMemFree` implicitly waits for the kernels below to
    // finish using it first).
    let s_elements = total_heads as usize * seq_len as usize * seq_len as usize;
    let scores = DeviceBuffer::<T>::alloc(s_elements)?;
    let scores_ptr: CUdeviceptr = scores.as_device_ptr();

    // --- Step 1: Compute S = Q @ K^T (unscaled; scale+mask applied next) ---
    let s_kernel_name = format!("mha_qk_gemm_{}", T::NAME);
    let s_kernel = handle.get_or_compile_kernel(
        &cache_key(&s_kernel_name, handle.sm_version()),
        &s_kernel_name,
        || generate_qk_gemm_ptx::<T>(&s_kernel_name, handle.sm_version()),
    )?;

    let qk_grid = grid_size_for(s_elements as u32, block_dim);
    let qk_params = LaunchParams::new(qk_grid, block_dim);

    s_kernel.kernel().launch(
        &qk_params,
        handle.stream(),
        &(
            q.ptr,
            k.ptr,
            scores_ptr,
            seq_len,
            head_dim,
            s_elements as u32,
        ),
    )?;

    // --- Step 2-3: Scale and apply mask (in-place on the scratch buffer) ---
    let scale_kernel_name = format!("mha_scale_mask_{}", T::NAME);
    // `has_mask` selects a different epilogue inside the generated PTX but is
    // absent from the entry name, so it has to be part of the cache key.
    let has_mask = mask.is_some();
    let scale_kernel = handle.get_or_compile_kernel(
        &cache_key_with(
            &scale_kernel_name,
            handle.sm_version(),
            &format!("mask={has_mask}"),
        ),
        &scale_kernel_name,
        || generate_scale_mask_ptx::<T>(&scale_kernel_name, handle.sm_version(), has_mask),
    )?;

    let scale_grid = grid_size_for(s_elements as u32, block_dim);
    let scale_params = LaunchParams::new(scale_grid, block_dim);

    let mask_ptr: CUdeviceptr = mask.map_or(0, |m| m.ptr);
    scale_kernel.kernel().launch(
        &scale_params,
        handle.stream(),
        &(scores_ptr, mask_ptr, s_elements as u32, sm_scale),
    )?;

    // --- Step 4: Row-wise softmax (in-place on the scratch buffer) ---
    let softmax_kernel_name = format!("mha_softmax_{}", T::NAME);
    let softmax_kernel = handle.get_or_compile_kernel(
        &cache_key(&softmax_kernel_name, handle.sm_version()),
        &softmax_kernel_name,
        || generate_row_softmax_ptx::<T>(&softmax_kernel_name, handle.sm_version()),
    )?;

    let softmax_rows = total_heads * seq_len;
    let softmax_grid = grid_size_for(softmax_rows, block_dim);
    let softmax_params = LaunchParams::new(softmax_grid, block_dim);

    softmax_kernel.kernel().launch(
        &softmax_params,
        handle.stream(),
        &(scores_ptr, seq_len, softmax_rows),
    )?;

    // --- Step 5: Compute O = P @ V ---
    let ov_kernel_name = format!("mha_pv_gemm_{}", T::NAME);
    let ov_kernel = handle.get_or_compile_kernel(
        &cache_key(&ov_kernel_name, handle.sm_version()),
        &ov_kernel_name,
        || generate_pv_gemm_ptx::<T>(&ov_kernel_name, handle.sm_version()),
    )?;

    let ov_elements = total_heads as usize * seq_len as usize * head_dim as usize;
    let ov_grid = grid_size_for(ov_elements as u32, block_dim);
    let ov_params = LaunchParams::new(ov_grid, block_dim);

    ov_kernel.kernel().launch(
        &ov_params,
        handle.stream(),
        &(
            scores_ptr,
            v.ptr,
            output.ptr,
            seq_len,
            head_dim,
            ov_elements as u32,
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
    let (kb, kh, kn, kd) = attn_dims(k)?;
    let (vb, vh, vn, _vd) = attn_dims(v)?;
    let (ob, oh, on, od) = attn_dims_mut(output)?;

    // Q, K must have same batch, heads, sequence length, and head_dim. This
    // naive (non-flash) path materializes a single `[seq_len, seq_len]` score
    // matrix per head, so it does not support cross-attention shapes where Q
    // and K/V have different sequence lengths.
    if qb != kb || qh != kh || qd != kd || qn != kn {
        return Err(DnnError::InvalidDimension(format!(
            "Q dims {:?} and K dims {:?}: batch, heads, seq_len, and head_dim must match",
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

/// Emits `dst = dst * a + dst`-style in-place accumulation: `acc += a * bv`.
///
/// The dot-product loops below are single runtime loops (label + branch), so
/// the accumulator must be a *stable* register that is read-modified-written
/// each iteration. A typed op like [`fma_float`](crate::ptx_helpers::fma_float)
/// allocates a fresh destination register on every Rust-side call, which
/// would only see the update on the *next* emitted instruction, not the next
/// *runtime* loop iteration -- so accumulation must go through the same named
/// register on every pass, exactly like `rnn::lstm::fma_acc_inplace`.
fn fma_acc_inplace<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    acc: &Register,
    a: &Register,
    bv: &Register,
) {
    let ty = if T::PTX_TYPE == PtxType::F32 {
        "f32"
    } else {
        "f64"
    };
    b.raw_ptx(&format!("fma.rn.{ty} {acc}, {a}, {bv}, {acc};"));
}

/// Decodes a flat thread/element index `gid` into `(row, col, batch_head)`
/// for a `[total_heads * seq_len, col_count]` flattened output matrix, where
/// `row` is itself `batch_head * seq_len + row_in_head`.
///
/// Both the QK^T (`col_count == seq_len`) and PV (`col_count == head_dim`)
/// GEMMs need exactly this decomposition, and in both cases `gid` is already
/// the correct flat element offset into the output buffer (`row * col_count +
/// col == gid` by construction), so no separate output-address computation is
/// needed.
fn decode_row_col_batch_head(
    b: &mut BodyBuilder<'_>,
    gid: &Register,
    col_count: &Register,
    seq_len: &Register,
) -> (Register, Register, Register) {
    let row = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {row}, {gid}, {col_count};"));
    let col = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {col}, {gid}, {col_count};"));
    let batch_head = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {batch_head}, {row}, {seq_len};"));
    (row, col, batch_head)
}

/// Generates PTX for the Q @ K^T batched GEMM step.
///
/// Computes, for every `(batch_head, i, j)` triple flattened into the launch
/// grid, `S[batch_head, i, j] = sum_d Q[batch_head, i, d] * K[batch_head, j,
/// d]`. No scaling or masking is applied here -- the caller runs
/// [`generate_scale_mask_ptx`] over the result afterwards.
fn generate_qk_gemm_ptx<T: GpuFloat>(kernel_name: &str, sm: SmVersion) -> DnnResult<String> {
    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .max_threads_per_block(256)
        .param("q_ptr", PtxType::U64)
        .param("k_ptr", PtxType::U64)
        .param("s_ptr", PtxType::U64)
        .param("seq_len", PtxType::U32)
        .param("head_dim", PtxType::U32)
        .param("n_elements", PtxType::U32)
        .body(|b| {
            let gid = b.global_thread_id_x();
            let n = b.load_param_u32("n_elements");
            b.if_lt_u32(gid.clone(), n, move |b| {
                let seq_len_reg = b.load_param_u32("seq_len");
                let head_dim_reg = b.load_param_u32("head_dim");
                let (row, col, batch_head) =
                    decode_row_col_batch_head(b, &gid, &seq_len_reg, &seq_len_reg);

                let q_ptr = b.load_param_u64("q_ptr");
                let k_ptr = b.load_param_u64("k_ptr");
                let s_ptr = b.load_param_u64("s_ptr");

                // Q's row for this thread is `row` directly (Q is flattened
                // as `[total_heads*seq_len, head_dim]`, same as `row`'s
                // definition). K's row lives in the same batch_head's block:
                // `batch_head*seq_len + col`.
                let q_row_off = b.mul_lo_u32(row, head_dim_reg.clone());
                let k_row = b.mad_lo_u32(batch_head, seq_len_reg, col);
                let k_row_off = b.mul_lo_u32(k_row, head_dim_reg.clone());

                let acc = load_float_imm::<T>(b, 0.0);
                let d_ctr = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {d_ctr}, 0;"));
                let loop_start = b.fresh_label("qk_dot_loop");
                let loop_end = b.fresh_label("qk_dot_end");
                b.label(&loop_start);
                let p = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p}, {d_ctr}, {head_dim_reg};"));
                b.branch_if(p, &loop_end);

                let q_off = b.add_u32(q_row_off.clone(), d_ctr.clone());
                let q_addr = b.byte_offset_addr(q_ptr.clone(), q_off, T::size_u32());
                let q_val = load_global_float::<T>(b, q_addr);

                let k_off = b.add_u32(k_row_off.clone(), d_ctr.clone());
                let k_addr = b.byte_offset_addr(k_ptr.clone(), k_off, T::size_u32());
                let k_val = load_global_float::<T>(b, k_addr);

                fma_acc_inplace::<T>(b, &acc, &q_val, &k_val);

                b.raw_ptx(&format!("add.u32 {d_ctr}, {d_ctr}, 1;"));
                b.branch(&loop_start);
                b.label(&loop_end);

                // `gid == row*seq_len + col` by construction, which is
                // exactly S's flat offset (S is `[total_heads*seq_len,
                // seq_len]`).
                let s_addr = b.byte_offset_addr(s_ptr, gid.clone(), T::size_u32());
                store_global_float::<T>(b, s_addr, acc);
            });
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;
    Ok(ptx)
}

/// Generates PTX for scaling attention scores and applying an additive mask.
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) fn generate_scale_mask_ptx<T: GpuFloat>(
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
                // Initialise the FMA addend to +0.0: an `alloc_reg` register is
                // undefined until written, so `fma(val, scale, zero)` would add
                // a garbage term without this `mov`.
                b.raw_ptx(&format!("mov.f32 {zero}, 0f00000000;"));
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

/// Emits `exp(x) = ex2.approx(x * log2(e))`, matching
/// `rnn::lstm::emit_approx_exp`: `f32` uses `ex2.approx.f32` directly; `f64`
/// round-trips through `f32` since `ex2.approx.f64` does not exist.
fn approx_exp<T: GpuFloat>(b: &mut BodyBuilder<'_>, x: Register, log2e: &Register) -> Register {
    let scaled = mul_float::<T>(b, x, log2e.clone());
    if T::PTX_TYPE == PtxType::F32 {
        b.ex2_approx_f32(scaled)
    } else {
        let f32_val = b.cvt_f64_to_f32(scaled);
        let exp_f32 = b.ex2_approx_f32(f32_val);
        b.cvt_f32_to_f64(exp_f32)
    }
}

/// Emits in-place `acc = max(acc, val)` (see [`fma_acc_inplace`] for why the
/// accumulator must be a stable, re-used register across loop iterations).
fn max_acc_inplace<T: GpuFloat>(b: &mut BodyBuilder<'_>, acc: &Register, val: &Register) {
    let ty = if T::PTX_TYPE == PtxType::F32 {
        "f32"
    } else {
        "f64"
    };
    b.raw_ptx(&format!("max.{ty} {acc}, {acc}, {val};"));
}

/// Emits in-place `acc = acc + val`.
fn add_acc_inplace<T: GpuFloat>(b: &mut BodyBuilder<'_>, acc: &Register, val: &Register) {
    let ty = if T::PTX_TYPE == PtxType::F32 {
        "f32"
    } else {
        "f64"
    };
    b.raw_ptx(&format!("add.{ty} {acc}, {acc}, {val};"));
}

/// Generates PTX for a numerically-stable row-wise softmax over attention
/// scores, in place: for each row `i`, `data[i, :] = softmax(data[i, :])`.
///
/// One thread processes one row sequentially through the standard 3-pass
/// stable softmax (max, exp-sum, normalize) -- no shared memory or block
/// cooperation is needed since there is no cross-thread reduction.
fn generate_row_softmax_ptx<T: GpuFloat>(kernel_name: &str, sm: SmVersion) -> DnnResult<String> {
    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .max_threads_per_block(256)
        .param("data_ptr", PtxType::U64)
        .param("row_len", PtxType::U32)
        .param("num_rows", PtxType::U32)
        .body(|b| {
            let row_id = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");
            b.if_lt_u32(row_id.clone(), num_rows, move |b| {
                let row_len_reg = b.load_param_u32("row_len");
                let data_ptr = b.load_param_u64("data_ptr");
                let row_off = b.mul_lo_u32(row_id, row_len_reg.clone());
                let log2e = load_float_imm::<T>(b, std::f64::consts::LOG2_E);

                // -- Pass 1: row max -----------------------------------------
                let max_reg = load_float_imm::<T>(b, f64::NEG_INFINITY);
                let ctr1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {ctr1}, 0;"));
                let max_loop = b.fresh_label("mha_sm_max_loop");
                let max_end = b.fresh_label("mha_sm_max_end");
                b.label(&max_loop);
                let p1 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p1}, {ctr1}, {row_len_reg};"));
                b.branch_if(p1, &max_end);
                let off1 = b.add_u32(row_off.clone(), ctr1.clone());
                let addr1 = b.byte_offset_addr(data_ptr.clone(), off1, T::size_u32());
                let val1 = load_global_float::<T>(b, addr1);
                max_acc_inplace::<T>(b, &max_reg, &val1);
                b.raw_ptx(&format!("add.u32 {ctr1}, {ctr1}, 1;"));
                b.branch(&max_loop);
                b.label(&max_end);

                // -- Pass 2: sum of exp(x - max) ------------------------------
                let sum_reg = load_float_imm::<T>(b, 0.0);
                let ctr2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {ctr2}, 0;"));
                let sum_loop = b.fresh_label("mha_sm_sum_loop");
                let sum_end = b.fresh_label("mha_sm_sum_end");
                b.label(&sum_loop);
                let p2 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p2}, {ctr2}, {row_len_reg};"));
                b.branch_if(p2, &sum_end);
                let off2 = b.add_u32(row_off.clone(), ctr2.clone());
                let addr2 = b.byte_offset_addr(data_ptr.clone(), off2, T::size_u32());
                let val2 = load_global_float::<T>(b, addr2);
                let diff2 = sub_float::<T>(b, val2, max_reg.clone());
                let exp2 = approx_exp::<T>(b, diff2, &log2e);
                add_acc_inplace::<T>(b, &sum_reg, &exp2);
                b.raw_ptx(&format!("add.u32 {ctr2}, {ctr2}, 1;"));
                b.branch(&sum_loop);
                b.label(&sum_end);

                // -- Pass 3: normalize in place --------------------------------
                let ty_name = if T::PTX_TYPE == PtxType::F32 {
                    "f32"
                } else {
                    "f64"
                };
                let recip = b.alloc_reg(T::PTX_TYPE);
                b.raw_ptx(&format!("rcp.approx.{ty_name} {recip}, {sum_reg};"));
                let ctr3 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {ctr3}, 0;"));
                let norm_loop = b.fresh_label("mha_sm_norm_loop");
                let norm_end = b.fresh_label("mha_sm_norm_end");
                b.label(&norm_loop);
                let p3 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p3}, {ctr3}, {row_len_reg};"));
                b.branch_if(p3, &norm_end);
                let off3 = b.add_u32(row_off.clone(), ctr3.clone());
                let addr3 = b.byte_offset_addr(data_ptr.clone(), off3, T::size_u32());
                let val3 = load_global_float::<T>(b, addr3.clone());
                let diff3 = sub_float::<T>(b, val3, max_reg.clone());
                let exp3 = approx_exp::<T>(b, diff3, &log2e);
                let normalized = mul_float::<T>(b, exp3, recip.clone());
                store_global_float::<T>(b, addr3, normalized);
                b.raw_ptx(&format!("add.u32 {ctr3}, {ctr3}, 1;"));
                b.branch(&norm_loop);
                b.label(&norm_end);
            });
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;
    Ok(ptx)
}

/// Generates PTX for the P @ V batched GEMM step.
///
/// Computes, for every `(batch_head, i, d)` triple flattened into the launch
/// grid, `O[batch_head, i, d] = sum_j P[batch_head, i, j] * V[batch_head, j,
/// d]`, where `P` is the post-softmax attention-probability matrix produced
/// by [`generate_row_softmax_ptx`].
fn generate_pv_gemm_ptx<T: GpuFloat>(kernel_name: &str, sm: SmVersion) -> DnnResult<String> {
    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .max_threads_per_block(256)
        .param("p_ptr", PtxType::U64)
        .param("v_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("seq_len", PtxType::U32)
        .param("head_dim", PtxType::U32)
        .param("n_elements", PtxType::U32)
        .body(|b| {
            let gid = b.global_thread_id_x();
            let n = b.load_param_u32("n_elements");
            b.if_lt_u32(gid.clone(), n, move |b| {
                let seq_len_reg = b.load_param_u32("seq_len");
                let head_dim_reg = b.load_param_u32("head_dim");
                let (row, d, batch_head) =
                    decode_row_col_batch_head(b, &gid, &head_dim_reg, &seq_len_reg);

                let p_ptr = b.load_param_u64("p_ptr");
                let v_ptr = b.load_param_u64("v_ptr");
                let out_ptr = b.load_param_u64("out_ptr");

                // P's row for this thread is `row` directly (P is flattened
                // as `[total_heads*seq_len, seq_len]`, same row indexing as
                // O). V's row for key/value position `j` lives in this
                // thread's batch_head block: `batch_head*seq_len + j`.
                let p_row_off = b.mul_lo_u32(row, seq_len_reg.clone());

                let acc = load_float_imm::<T>(b, 0.0);
                let j_ctr = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {j_ctr}, 0;"));
                let loop_start = b.fresh_label("pv_dot_loop");
                let loop_end = b.fresh_label("pv_dot_end");
                b.label(&loop_start);
                let p_pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_pred}, {j_ctr}, {seq_len_reg};"));
                b.branch_if(p_pred, &loop_end);

                let p_off = b.add_u32(p_row_off.clone(), j_ctr.clone());
                let p_addr = b.byte_offset_addr(p_ptr.clone(), p_off, T::size_u32());
                let p_val = load_global_float::<T>(b, p_addr);

                let v_row = b.mad_lo_u32(batch_head.clone(), seq_len_reg.clone(), j_ctr.clone());
                let v_row_off = b.mul_lo_u32(v_row, head_dim_reg.clone());
                let v_off = b.add_u32(v_row_off, d.clone());
                let v_addr = b.byte_offset_addr(v_ptr.clone(), v_off, T::size_u32());
                let v_val = load_global_float::<T>(b, v_addr);

                fma_acc_inplace::<T>(b, &acc, &p_val, &v_val);

                b.raw_ptx(&format!("add.u32 {j_ctr}, {j_ctr}, 1;"));
                b.branch(&loop_start);
                b.label(&loop_end);

                // `gid == row*head_dim + d` by construction, which is
                // exactly O's flat offset (O is `[total_heads*seq_len,
                // head_dim]`).
                let out_addr = b.byte_offset_addr(out_ptr, gid.clone(), T::size_u32());
                store_global_float::<T>(b, out_addr, acc);
            });
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
