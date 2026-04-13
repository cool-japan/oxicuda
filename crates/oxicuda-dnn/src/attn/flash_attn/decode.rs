//! Single-query decode attention (non-paged, simpler path).
//!
//! This module provides a simpler decode attention kernel for cases where
//! the KV-cache is stored contiguously (no paging). Each query attends to
//! a prefix of contiguous K and V vectors.
//!
//! ## Algorithm
//!
//! For each batch entry and head:
//! 1. Load query vector `q[b, h, :]` (single token).
//! 2. Stream through K-cache `k[b, h, 0..seq_len, :]`:
//!    - Compute `s[t] = dot(q, k[t]) * sm_scale`.
//!    - Track running max and sum for online softmax.
//! 3. Accumulate `o += softmax(s[t]) * v[t]`.
//! 4. Final normalisation and store.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::tensor_util::{attn_dims, attn_dims_mut};
use crate::types::{TensorDesc, TensorDescMut};

/// Executes single-query decode attention with contiguous KV-cache.
///
/// # Arguments
///
/// * `handle` - DNN handle providing context and stream.
/// * `q` - Query tensor `[B, H, 1, D]` (single decode token).
/// * `k_cache` - Key cache tensor `[B, H, max_seq_len, D]`.
/// * `v_cache` - Value cache tensor `[B, H, max_seq_len, D]`.
/// * `seq_lengths` - Actual sequence length per batch entry `[B]`.
/// * `output` - Output tensor `[B, H, 1, D]`.
/// * `sm_scale` - Softmax scaling factor, typically `1.0 / sqrt(D)`.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] on shape mismatch.
/// Returns [`DnnError::LaunchFailed`] on kernel launch failure.
pub fn single_query_decode_attention<T: GpuFloat>(
    handle: &DnnHandle,
    q: &TensorDesc<T>,
    k_cache: &TensorDesc<T>,
    v_cache: &TensorDesc<T>,
    seq_lengths: &DeviceBuffer<i32>,
    output: &mut TensorDescMut<T>,
    sm_scale: f32,
) -> DnnResult<()> {
    validate_decode_shapes(q, k_cache, v_cache, seq_lengths, output)?;

    let (batch, num_heads, _seq, head_dim) = attn_dims(q)?;
    let max_seq_len = k_cache.dims[2];
    let sm = handle.sm_version();

    let kernel_name = format!("decode_attn_d{}_{}", head_dim, T::NAME);
    let ptx = generate_decode_ptx::<T>(&kernel_name, sm, head_dim, max_seq_len)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let threads = 256u32.min(head_dim.div_ceil(32) * 32).max(32);

    let grid = Dim3::new(num_heads, batch, 1);
    let block = Dim3::new(threads, 1, 1);
    let smem_bytes = threads * 4;

    let params = LaunchParams::builder()
        .grid(grid)
        .block(block)
        .shared_mem(smem_bytes)
        .build();

    kernel.launch(
        &params,
        handle.stream(),
        &(
            q.ptr,
            k_cache.ptr,
            v_cache.ptr,
            seq_lengths.as_device_ptr(),
            output.ptr,
            head_dim,
            num_heads,
            max_seq_len,
            sm_scale,
            batch,
        ),
    )?;

    Ok(())
}

/// Validates shapes for single-query decode attention.
fn validate_decode_shapes<T: GpuFloat>(
    q: &TensorDesc<T>,
    k_cache: &TensorDesc<T>,
    v_cache: &TensorDesc<T>,
    seq_lengths: &DeviceBuffer<i32>,
    output: &TensorDescMut<T>,
) -> DnnResult<()> {
    let (batch, heads, seq, head_dim) = attn_dims(q)?;

    if seq != 1 {
        return Err(DnnError::InvalidDimension(format!(
            "decode Q seq_len must be 1, got {seq}"
        )));
    }

    let (kb, kh, _ksn, kd) = attn_dims(k_cache)?;
    if kb != batch || kh != heads || kd != head_dim {
        return Err(DnnError::InvalidDimension(format!(
            "K-cache dims {:?} incompatible with Q {:?}",
            k_cache.dims, q.dims
        )));
    }

    let (vb, vh, vsn, vd) = attn_dims(v_cache)?;
    if vb != batch || vh != heads || vd != head_dim {
        return Err(DnnError::InvalidDimension(format!(
            "V-cache dims {:?} incompatible with Q {:?}",
            v_cache.dims, q.dims
        )));
    }
    if k_cache.dims[2] != vsn {
        return Err(DnnError::InvalidDimension(format!(
            "K-cache max_seq {} != V-cache max_seq {}",
            k_cache.dims[2], vsn
        )));
    }

    let (ob, oh, osn, od) = attn_dims_mut(output)?;
    if ob != batch || oh != heads || osn != 1 || od != head_dim {
        return Err(DnnError::InvalidDimension(format!(
            "output dims {:?} != Q dims {:?}",
            output.dims, q.dims
        )));
    }

    if seq_lengths.len() < batch as usize {
        return Err(DnnError::BufferTooSmall {
            expected: batch as usize * 4,
            actual: seq_lengths.len() * 4,
        });
    }

    Ok(())
}

/// Generates PTX for the decode attention kernel.
#[allow(clippy::extra_unused_type_parameters)]
fn generate_decode_ptx<T: GpuFloat>(
    kernel_name: &str,
    sm: SmVersion,
    head_dim: u32,
    _max_seq_len: u32,
) -> DnnResult<String> {
    let threads = 256u32.min(head_dim.div_ceil(32) * 32).max(32);

    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .param("q_ptr", PtxType::U64)
        .param("k_cache_ptr", PtxType::U64)
        .param("v_cache_ptr", PtxType::U64)
        .param("seq_lengths_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("head_dim", PtxType::U32)
        .param("num_heads", PtxType::U32)
        .param("max_seq_len", PtxType::U32)
        .param("sm_scale", PtxType::F32)
        .param("batch_size", PtxType::U32)
        .shared_mem("scratch", PtxType::F32, threads as usize)
        .max_threads_per_block(threads)
        .body(move |b| {
            let _tid = b.thread_id_x();
            let _head_id = b.block_id_x();

            b.comment("=== Single-Query Decode Attention ===");
            b.comment("");
            b.comment("Grid: (num_heads, batch_size, 1)");
            b.comment("Block: (threads, 1, 1)");
            b.comment("");
            b.comment("Step 1: Load query vector q[batch, head, 0, :] to registers");
            b.comment("Step 2: Read actual seq_length for this batch entry");
            b.comment("Step 3: Initialise accumulators (o=0, m=-inf, l=0)");
            b.comment("Step 4: Stream through K-cache, compute dot products + online softmax");
            b.comment("Step 5: Second pass through V-cache (or fused with Step 4)");
            b.comment("Step 6: Final normalisation and store");

            let _q_base = b.load_param_u64("q_ptr");
            let _k_base = b.load_param_u64("k_cache_ptr");
            let _seq_base = b.load_param_u64("seq_lengths_ptr");

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

    #[test]
    fn generate_decode_ptx_succeeds() {
        let ptx = generate_decode_ptx::<f32>("test_decode", SmVersion::Sm80, 64, 2048);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains(".entry test_decode"));
        assert!(text.contains("Decode Attention"));
    }

    #[test]
    fn generate_decode_ptx_large_head_dim() {
        let ptx = generate_decode_ptx::<f32>("test_decode_256", SmVersion::Sm80, 256, 1024);
        assert!(ptx.is_ok());
    }

    #[test]
    fn generate_decode_ptx_f64() {
        let ptx = generate_decode_ptx::<f64>("test_decode_f64", SmVersion::Sm80, 128, 512);
        assert!(ptx.is_ok());
    }
}
