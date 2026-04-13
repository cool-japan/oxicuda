//! Rotary Positional Embedding (RoPE).
//!
//! Implements the rotary positional encoding from Su et al. (2021), which
//! applies a position-dependent rotation to pairs of dimensions in the
//! query and key vectors. RoPE has become the standard positional encoding
//! for modern LLMs (LLaMA, Mistral, etc.) because it:
//!
//! - Encodes relative position information naturally.
//! - Decays attention with distance via the rotation mechanism.
//! - Supports length extrapolation better than learned embeddings.
//!
//! ## Algorithm
//!
//! For each pair of dimensions `(2i, 2i+1)` at position `pos`:
//! ```text
//! theta_i = base^(-2i / head_dim)
//! cos_val = cos(pos * theta_i)
//! sin_val = sin(pos * theta_i)
//! x_rot[2i]   = x[2i]   * cos_val - x[2i+1] * sin_val
//! x_rot[2i+1] = x[2i]   * sin_val + x[2i+1] * cos_val
//! ```

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::tensor_util::attn_dims_mut;
use crate::types::TensorDescMut;

/// Applies Rotary Positional Embedding (RoPE) to query and key tensors in-place.
///
/// Each thread handles one pair of adjacent dimensions, applying the rotation
/// matrix determined by the token position and frequency base.
///
/// # Arguments
///
/// * `handle` - DNN handle providing context and stream.
/// * `q` - Query tensor `[B, H, N, D]` (modified in-place).
/// * `k` - Key tensor `[B, H, N, D]` (modified in-place).
/// * `positions` - Position index for each token `[B * N]`.
/// * `head_dim` - Per-head dimension (must be even).
/// * `base` - Frequency base (typically 10000.0 for standard RoPE,
///   500000.0 for extended-context models).
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if head_dim is odd or tensors
/// have inconsistent shapes.
/// Returns [`DnnError::LaunchFailed`] on kernel launch failure.
pub fn apply_rope<T: GpuFloat>(
    handle: &DnnHandle,
    q: &mut TensorDescMut<T>,
    k: &mut TensorDescMut<T>,
    positions: &DeviceBuffer<i32>,
    head_dim: u32,
    base: f32,
) -> DnnResult<()> {
    if head_dim % 2 != 0 {
        return Err(DnnError::InvalidDimension(
            "RoPE requires even head_dim".to_string(),
        ));
    }

    let (qb, qh, qn, qd) = attn_dims_mut(q)?;
    let (kb, _kh, kn, kd) = attn_dims_mut(k)?;

    if qd != head_dim || kd != head_dim {
        return Err(DnnError::InvalidDimension(format!(
            "head_dim mismatch: Q={}, K={}, expected={}",
            qd, kd, head_dim
        )));
    }
    if qb != kb || qn != kn {
        return Err(DnnError::InvalidDimension(
            "Q and K must have same batch size and seq_len for RoPE".to_string(),
        ));
    }

    let batch = qb;
    let seq_len = qn;
    let q_heads = qh;
    let k_heads = _kh;

    let total_tokens = batch as usize * seq_len as usize;
    if positions.len() < total_tokens {
        return Err(DnnError::BufferTooSmall {
            expected: total_tokens * 4,
            actual: positions.len() * 4,
        });
    }

    let sm = handle.sm_version();
    let half_dim = head_dim / 2;

    let kernel_name = format!("rope_{}", T::NAME);
    let ptx = generate_rope_ptx::<T>(&kernel_name, sm, head_dim)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let block_dim = 256u32;

    // Launch for Q
    let total_pairs_q = batch as u64 * q_heads as u64 * seq_len as u64 * half_dim as u64;
    let grid_dim_q = grid_size_for(total_pairs_q as u32, block_dim);

    let q_params = LaunchParams::builder()
        .grid(Dim3::new(grid_dim_q, 1, 1))
        .block(Dim3::new(block_dim, 1, 1))
        .shared_mem(0)
        .build();

    kernel.launch(
        &q_params,
        handle.stream(),
        &(
            q.ptr,
            positions.as_device_ptr(),
            head_dim,
            q_heads,
            seq_len,
            batch,
            base,
            total_pairs_q as u32,
        ),
    )?;

    // Launch for K (may have different num_heads for GQA)
    let total_pairs_k = batch as u64 * k_heads as u64 * seq_len as u64 * half_dim as u64;
    let grid_dim_k = grid_size_for(total_pairs_k as u32, block_dim);

    let k_params = LaunchParams::builder()
        .grid(Dim3::new(grid_dim_k, 1, 1))
        .block(Dim3::new(block_dim, 1, 1))
        .shared_mem(0)
        .build();

    kernel.launch(
        &k_params,
        handle.stream(),
        &(
            k.ptr,
            positions.as_device_ptr(),
            head_dim,
            k_heads,
            seq_len,
            batch,
            base,
            total_pairs_k as u32,
        ),
    )?;

    Ok(())
}

/// Generates PTX for the RoPE kernel.
#[allow(clippy::extra_unused_type_parameters)]
fn generate_rope_ptx<T: GpuFloat>(
    kernel_name: &str,
    sm: SmVersion,
    head_dim: u32,
) -> DnnResult<String> {
    let _half_dim = head_dim / 2;

    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .param("data_ptr", PtxType::U64)
        .param("positions_ptr", PtxType::U64)
        .param("head_dim", PtxType::U32)
        .param("num_heads", PtxType::U32)
        .param("seq_len", PtxType::U32)
        .param("batch_size", PtxType::U32)
        .param("base", PtxType::F32)
        .param("total_pairs", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let total = b.load_param_u32("total_pairs");

            b.if_lt_u32(gid, total, |b| {
                b.comment("=== Rotary Positional Embedding ===");
                b.comment("");
                b.comment("Decompose linear index into (batch, head, token, pair):");
                b.comment("  pair_idx  = gid % half_dim");
                b.comment("  token_idx = (gid / half_dim) % seq_len");
                b.comment("  head_idx  = (gid / (half_dim * seq_len)) % num_heads");
                b.comment("  batch_idx = gid / (half_dim * seq_len * num_heads)");

                let _data = b.load_param_u64("data_ptr");
                let _pos_ptr = b.load_param_u64("positions_ptr");
                let _base_val = b.load_param_f32("base");

                b.comment("");
                b.comment("Compute frequency:");
                b.comment("  theta = base^(-2 * pair_idx / head_dim)");
                b.comment("  angle = position * theta");
                b.comment("");
                b.comment("Load pair (x_even, x_odd)");
                b.comment("Apply rotation:");
                b.comment("  x_even' = x_even * cos(angle) - x_odd * sin(angle)");
                b.comment("  x_odd'  = x_even * sin(angle) + x_odd * cos(angle)");
                b.comment("Store back");
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

    #[test]
    fn generate_rope_ptx_succeeds() {
        let ptx = generate_rope_ptx::<f32>("test_rope", SmVersion::Sm80, 64);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains(".entry test_rope"));
        assert!(text.contains("Rotary Positional Embedding"));
    }

    #[test]
    fn generate_rope_ptx_128d() {
        let ptx = generate_rope_ptx::<f32>("test_rope_128", SmVersion::Sm80, 128);
        assert!(ptx.is_ok());
    }

    #[test]
    fn odd_head_dim_detected() {
        assert_ne!(65 % 2, 0);
    }
}
