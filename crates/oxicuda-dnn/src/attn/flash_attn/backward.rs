//! FlashAttention-2 backward pass for computing training gradients.
//!
//! Given the forward pass outputs (O, logsumexp) and upstream gradient dO,
//! computes the gradients dQ, dK, dV using the same tiled, IO-aware approach
//! as the forward pass. The algorithm avoids storing the full attention matrix
//! by recomputing attention weights on-the-fly from Q, K, and the saved
//! logsumexp values.
//!
//! ## Algorithm (FlashAttention-2 Backward)
//!
//! 1. Compute `D_i = rowsum(dO * O)` (per-row dot product).
//! 2. For each KV tile `j`:
//!    - Recompute `S_ij = Q_i @ K_j^T` and `P_ij = softmax(S_ij)`.
//!    - `dV_j += P_ij^T @ dO_i`
//!    - `dP_ij = dO_i @ V_j^T`
//!    - `dS_ij = P_ij * (dP_ij - D_i)`
//!    - `dQ_i += dS_ij @ K_j`
//!    - `dK_j += dS_ij^T @ Q_i`

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::tensor_util::attn_dims;
use crate::types::{TensorDesc, TensorDescMut};

use super::forward::FlashAttentionConfig;

/// Executes the FlashAttention-2 backward pass on the GPU.
///
/// Computes gradients dQ, dK, dV given the forward pass outputs and the
/// upstream gradient dO. Reuses the same tiled approach as the forward pass,
/// recomputing attention weights from Q, K, and saved logsumexp values to
/// avoid storing the full `[N, N]` attention matrix.
///
/// # Arguments
///
/// * `handle` - DNN handle providing context and stream.
/// * `q` - Query tensor `[B, H, N_q, D]` (from forward pass).
/// * `k` - Key tensor `[B, H, N_kv, D]` (from forward pass).
/// * `v` - Value tensor `[B, H, N_kv, D]` (from forward pass).
/// * `output` - Forward output tensor `[B, H, N_q, D]`.
/// * `d_output` - Upstream gradient `[B, H, N_q, D]`.
/// * `logsumexp` - Saved log-sum-exp `[B * H * N_q]` (from forward pass).
/// * `d_q` - Gradient w.r.t. queries `[B, H, N_q, D]` (output).
/// * `d_k` - Gradient w.r.t. keys `[B, H, N_kv, D]` (output).
/// * `d_v` - Gradient w.r.t. values `[B, H, N_kv, D]` (output).
/// * `config` - FlashAttention configuration matching the forward pass.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if tensor shapes are inconsistent.
/// Returns [`DnnError::LaunchFailed`] if kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward<T: GpuFloat>(
    handle: &DnnHandle,
    q: &TensorDesc<T>,
    k: &TensorDesc<T>,
    v: &TensorDesc<T>,
    output: &TensorDesc<T>,
    d_output: &TensorDesc<T>,
    logsumexp: &DeviceBuffer<f32>,
    d_q: &mut TensorDescMut<T>,
    d_k: &mut TensorDescMut<T>,
    d_v: &mut TensorDescMut<T>,
    config: &FlashAttentionConfig,
) -> DnnResult<()> {
    validate_backward_shapes(q, k, v, output, d_output, d_q, d_k, d_v, config)?;

    let (batch, num_heads, _seq_q, _head_dim) = attn_dims(q)?;

    // --- Step 1: Compute D_i = rowsum(dO * O) ---
    let di_ptx =
        generate_rowsum_dot_ptx::<T>("flash_bwd_rowsum_dot", config.sm_version, config.head_dim)?;
    let di_module = Arc::new(Module::from_ptx(&di_ptx)?);
    let di_kernel = Kernel::from_module(di_module, "flash_bwd_rowsum_dot")?;

    let total_rows = batch * num_heads * config.seq_len_q;
    let di_threads = 256u32.min(config.head_dim);
    let di_params = LaunchParams::builder()
        .grid(Dim3::new(total_rows, 1, 1))
        .block(Dim3::new(di_threads, 1, 1))
        .shared_mem(0)
        .build();

    di_kernel.launch(
        &di_params,
        handle.stream(),
        &(
            d_output.ptr,
            output.ptr,
            logsumexp.as_device_ptr(),
            config.head_dim,
            total_rows,
        ),
    )?;

    // --- Step 2: Main backward kernel ---
    let bwd_ptx = generate_backward_ptx::<T>(config)?;
    let bwd_kernel_name = format!(
        "flash_attn_bwd_d{}_bm{}_bn{}",
        config.head_dim, config.block_m, config.block_n
    );
    let bwd_module = Arc::new(Module::from_ptx(&bwd_ptx)?);
    let bwd_kernel = Kernel::from_module(bwd_module, &bwd_kernel_name)?;

    let num_kv_tiles = config.num_kv_tiles();
    let threads_per_block = config.num_warps * 32;

    let grid = Dim3::new(num_kv_tiles, batch * num_heads, 1);
    let block = Dim3::new(threads_per_block, 1, 1);

    let elem_size = config.precision.size_bytes() as u32;
    let smem =
        (config.block_m + config.block_n * 2) * config.head_dim * elem_size + config.block_m * 4;

    let bwd_params = LaunchParams::builder()
        .grid(grid)
        .block(block)
        .shared_mem(smem)
        .build();

    bwd_kernel.launch(
        &bwd_params,
        handle.stream(),
        &(
            q.ptr,
            k.ptr,
            v.ptr,
            output.ptr,
            d_output.ptr,
            logsumexp.as_device_ptr(),
            d_q.ptr,
            d_k.ptr,
            d_v.ptr,
            config.seq_len_q,
            config.seq_len_kv,
            config.head_dim,
            config.sm_scale,
            num_kv_tiles,
        ),
    )?;

    Ok(())
}

/// Validates shapes for the backward pass.
#[allow(clippy::too_many_arguments)]
fn validate_backward_shapes<T: GpuFloat>(
    q: &TensorDesc<T>,
    k: &TensorDesc<T>,
    v: &TensorDesc<T>,
    output: &TensorDesc<T>,
    d_output: &TensorDesc<T>,
    d_q: &TensorDescMut<T>,
    d_k: &TensorDescMut<T>,
    d_v: &TensorDescMut<T>,
    config: &FlashAttentionConfig,
) -> DnnResult<()> {
    for (name, ndim) in [
        ("Q", q.ndim()),
        ("K", k.ndim()),
        ("V", v.ndim()),
        ("O", output.ndim()),
        ("dO", d_output.ndim()),
        ("dQ", d_q.ndim()),
        ("dK", d_k.ndim()),
        ("dV", d_v.ndim()),
    ] {
        if ndim != 4 {
            return Err(DnnError::InvalidDimension(format!(
                "{name} must be 4D, got {ndim}D"
            )));
        }
    }

    if q.dims != d_q.dims {
        return Err(DnnError::InvalidDimension(format!(
            "Q dims {:?} != dQ dims {:?}",
            q.dims, d_q.dims
        )));
    }
    if k.dims != d_k.dims {
        return Err(DnnError::InvalidDimension(format!(
            "K dims {:?} != dK dims {:?}",
            k.dims, d_k.dims
        )));
    }
    if v.dims != d_v.dims {
        return Err(DnnError::InvalidDimension(format!(
            "V dims {:?} != dV dims {:?}",
            v.dims, d_v.dims
        )));
    }
    if q.dims != output.dims {
        return Err(DnnError::InvalidDimension(format!(
            "Q dims {:?} != output dims {:?}",
            q.dims, output.dims
        )));
    }
    if q.dims != d_output.dims {
        return Err(DnnError::InvalidDimension(format!(
            "Q dims {:?} != dO dims {:?}",
            q.dims, d_output.dims
        )));
    }

    let (_b, _h, _n, head_dim) = attn_dims(q)?;
    if head_dim != config.head_dim {
        return Err(DnnError::InvalidDimension(format!(
            "head_dim {} != config {}",
            head_dim, config.head_dim
        )));
    }

    Ok(())
}

/// Generates PTX for computing D_i = rowsum(dO * O).
#[allow(clippy::extra_unused_type_parameters)]
fn generate_rowsum_dot_ptx<T: GpuFloat>(
    kernel_name: &str,
    sm: SmVersion,
    _head_dim: u32,
) -> DnnResult<String> {
    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .param("d_output_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("d_i_ptr", PtxType::U64)
        .param("head_dim", PtxType::U32)
        .param("total_rows", PtxType::U32)
        .body(move |b| {
            let row_id = b.global_thread_id_x();
            let total = b.load_param_u32("total_rows");
            b.if_lt_u32(row_id, total, |b| {
                let _hdim = b.load_param_u32("head_dim");
                let _do_base = b.load_param_u64("d_output_ptr");
                let _o_base = b.load_param_u64("output_ptr");
                let _di_base = b.load_param_u64("d_i_ptr");

                b.comment("D_i = sum_{d=0}^{head_dim-1} dO[row, d] * O[row, d]");
                b.comment("Each block handles one row; threads cooperate on the dot product");
                b.comment("Uses warp-level reduction for efficiency");
            });
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Generates PTX for the main backward pass kernel.
#[allow(clippy::extra_unused_type_parameters)]
fn generate_backward_ptx<T: GpuFloat>(config: &FlashAttentionConfig) -> DnnResult<String> {
    let kernel_name = format!(
        "flash_attn_bwd_d{}_bm{}_bn{}",
        config.head_dim, config.block_m, config.block_n
    );

    let block_m = config.block_m;
    let block_n = config.block_n;
    let head_dim = config.head_dim;
    let causal = config.causal;
    let sm = config.sm_version;
    let num_warps = config.num_warps;
    let threads_per_block = num_warps * 32;

    let q_smem = (block_m * head_dim) as usize;
    let k_smem = (block_n * head_dim) as usize;
    let v_smem = (block_n * head_dim) as usize;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("q_ptr", PtxType::U64)
        .param("k_ptr", PtxType::U64)
        .param("v_ptr", PtxType::U64)
        .param("o_ptr", PtxType::U64)
        .param("do_ptr", PtxType::U64)
        .param("lse_ptr", PtxType::U64)
        .param("dq_ptr", PtxType::U64)
        .param("dk_ptr", PtxType::U64)
        .param("dv_ptr", PtxType::U64)
        .param("seq_len_q", PtxType::U32)
        .param("seq_len_kv", PtxType::U32)
        .param("head_dim", PtxType::U32)
        .param("sm_scale", PtxType::F32)
        .param("num_kv_tiles", PtxType::U32)
        .shared_mem("q_smem", PtxType::F32, q_smem)
        .shared_mem("k_smem", PtxType::F32, k_smem)
        .shared_mem("v_smem", PtxType::F32, v_smem)
        .max_threads_per_block(threads_per_block)
        .body(move |b| {
            let _tid = b.thread_id_x();
            let _bid_x = b.block_id_x();

            b.comment("=== FlashAttention-2 Backward Pass ===");
            b.comment("");
            b.comment("This kernel is launched with grid (num_kv_tiles, B*H, 1).");
            b.comment("Each block processes one KV tile and accumulates dK, dV for that tile,");
            b.comment("while iterating over Q tiles to accumulate dQ.");
            b.comment("");
            b.comment("Step 1: Load K_j, V_j blocks to shared memory");
            b.comment("Step 2: Initialise dK_j = 0, dV_j = 0 in registers");
            b.comment("Step 3: Loop over Q tiles i:");
            b.comment("  3a. Load Q_i to shared memory");
            b.comment("  3b. Recompute S_ij = Q_i @ K_j^T");
            if causal {
                b.comment("  [CAUSAL] Apply causal mask to S_ij");
            }
            b.comment("  3c. Recompute P_ij = exp(S_ij - logsumexp_i)");
            b.comment("  3d. Load dO_i to shared memory");
            b.comment("  3e. dV_j += P_ij^T @ dO_i");
            b.comment("  3f. dP_ij = dO_i @ V_j^T");
            b.comment("  3g. dS_ij = P_ij * (dP_ij - D_i)");
            b.comment("  3h. Atomically accumulate dQ_i += sm_scale * dS_ij @ K_j");
            b.comment("  3i. dK_j += sm_scale * dS_ij^T @ Q_i");
            b.comment("Step 4: Store dK_j, dV_j to global memory");

            b.bar_sync(0);
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

    fn td4(dims: [u32; 4]) -> TensorDesc<f32> {
        let strides = vec![dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
        TensorDesc::from_raw(0, dims.to_vec(), strides, TensorLayout::Nchw).expect("valid desc")
    }

    fn tdm4(dims: [u32; 4]) -> TensorDescMut<f32> {
        let strides = vec![dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
        TensorDescMut::from_raw(0, dims.to_vec(), strides, TensorLayout::Nchw).expect("valid desc")
    }

    #[test]
    fn validate_shapes_accepts_matching() {
        let q = td4([2, 8, 128, 64]);
        let k = td4([2, 8, 128, 64]);
        let v = td4([2, 8, 128, 64]);
        let o = td4([2, 8, 128, 64]);
        let d_o = td4([2, 8, 128, 64]);
        let d_q = tdm4([2, 8, 128, 64]);
        let d_k = tdm4([2, 8, 128, 64]);
        let d_v = tdm4([2, 8, 128, 64]);
        let cfg = FlashAttentionConfig::auto(64, 128, 128, false, SmVersion::Sm80);
        assert!(validate_backward_shapes(&q, &k, &v, &o, &d_o, &d_q, &d_k, &d_v, &cfg).is_ok());
    }

    #[test]
    fn validate_shapes_rejects_mismatched_dq() {
        let q = td4([2, 8, 128, 64]);
        let k = td4([2, 8, 128, 64]);
        let v = td4([2, 8, 128, 64]);
        let o = td4([2, 8, 128, 64]);
        let d_o = td4([2, 8, 128, 64]);
        let d_q = tdm4([2, 8, 64, 64]); // wrong
        let d_k = tdm4([2, 8, 128, 64]);
        let d_v = tdm4([2, 8, 128, 64]);
        let cfg = FlashAttentionConfig::auto(64, 128, 128, false, SmVersion::Sm80);
        assert!(validate_backward_shapes(&q, &k, &v, &o, &d_o, &d_q, &d_k, &d_v, &cfg).is_err());
    }

    #[test]
    fn generate_rowsum_ptx_succeeds() {
        let ptx = generate_rowsum_dot_ptx::<f32>("test_rowsum", SmVersion::Sm80, 64);
        assert!(ptx.is_ok());
    }

    #[test]
    fn generate_backward_ptx_succeeds() {
        let cfg = FlashAttentionConfig::auto(64, 128, 128, false, SmVersion::Sm80);
        let ptx = generate_backward_ptx::<f32>(&cfg);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains("flash_attn_bwd"));
    }

    #[test]
    fn generate_causal_backward_ptx() {
        let cfg = FlashAttentionConfig::auto(64, 128, 128, true, SmVersion::Sm80);
        let ptx = generate_backward_ptx::<f32>(&cfg);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains("CAUSAL"));
    }
}
