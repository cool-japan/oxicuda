//! GPT-NeoX half-split partial-rotary RoPE (device kernel).
//!
//! This module emits a real PTX kernel for the GPT-NeoX flavour of rotary
//! position embedding, the device-resident counterpart of
//! [`crate::position::apply_rope_neox_half_split`].
//!
//! ## Half-split pairing
//!
//! Unlike the interleaved RoPE in [`crate::attn::rope`] (which pairs adjacent
//! lanes `2i` / `2i+1`), the GPT-NeoX layout splits the rotated block in half:
//! lane `i` is rotated against lane `i + rotary_dim/2`. With `half =
//! rotary_dim / 2`:
//!
//! ```text
//! out[i]      = x[i]·cosθ − x[i+half]·sinθ
//! out[i+half] = x[i]·sinθ + x[i+half]·cosθ
//! ```
//!
//! ## Frequencies
//!
//! The inverse frequency denominator is `rotary_dim` (the number of rotated
//! lanes), **not** `head_dim`:
//!
//! ```text
//! freq_i = base^(−2i / rotary_dim) = exp2( (−2i / rotary_dim) · log2(base) )
//! θ      = pos · freq_i
//! ```
//!
//! The `base^x` is computed on-device with the `lg2`/`ex2` approximation
//! instructions (there is no `powf` on the device), exactly as
//! `base^x = exp2(x · log2(base))`.
//!
//! ## Partial rotary
//!
//! Only the leading `rotary_dim` lanes are rotated; the tail
//! `[rotary_dim, head_dim)` is copied through unchanged (the `pair_idx == 0`
//! thread of each head performs that copy, unrolled over the compile-time tail
//! range).
//!
//! ## Layout
//!
//! The tensor is flat row-major `[seq_len, num_heads, head_dim]` (no batch
//! dimension), positions are implicit `0..seq_len`, and the operation is
//! out-of-place (`input` → `output`). This matches trustformers' `rope_f32`.

use oxicuda_blas::GpuFloat;
use oxicuda_launch::{Dim3, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::kernel_cache::cache_key_with;

/// Applies GPT-NeoX half-split partial-rotary RoPE to a device tensor.
///
/// The kernel rotates the leading `rotary_dim` lanes of each head using the
/// half-split pairing (`i` ↔ `i + rotary_dim/2`), writing the result to a
/// separate `output` buffer. Lanes in the tail `[rotary_dim, head_dim)` are
/// copied through unchanged. One thread is launched per `(pos, head, pair)`
/// triple where `pair < rotary_dim/2`.
///
/// # Arguments
///
/// * `handle` - DNN handle providing the SM version and CUDA stream.
/// * `input` - Source tensor, flat row-major `[seq_len, num_heads, head_dim]`.
/// * `output` - Destination tensor of identical shape (out-of-place).
/// * `seq_len` - Sequence length (positions are implicit `0..seq_len`).
/// * `num_heads` - Number of attention heads.
/// * `head_dim` - Per-head feature dimension (full head width).
/// * `rotary_dim` - Number of leading lanes to rotate; must be even, non-zero,
///   and `<= head_dim`.
/// * `base` - Frequency base (typically `10000.0`).
///
/// # Errors
///
/// * [`DnnError::InvalidDimension`] if `head_dim == 0`.
/// * [`DnnError::InvalidArgument`] if `rotary_dim` is zero, odd, or greater
///   than `head_dim`, or if `seq_len` / `num_heads` is zero.
/// * [`DnnError::BufferTooSmall`] if `input` or `output` is undersized.
/// * [`DnnError::PtxGeneration`] if PTX generation fails, or a launch/driver
///   error if the kernel cannot be built or launched.
// The 8-argument signature mirrors the trustformers `rope_f32` host call
// (input, output, seq_len, num_heads, head_dim, rotary_dim, base) plus the
// handle; each is a distinct scalar with no natural grouping struct.
#[allow(clippy::too_many_arguments)]
pub fn rope_neox_half_split_f32(
    handle: &DnnHandle,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    base: f32,
) -> DnnResult<()> {
    if head_dim == 0 {
        return Err(DnnError::InvalidDimension(
            "rope_neox: head_dim must be non-zero".into(),
        ));
    }
    if rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
        return Err(DnnError::InvalidArgument(format!(
            "rope_neox: rotary_dim ({rotary_dim}) must be even, non-zero, and <= head_dim ({head_dim})"
        )));
    }
    if seq_len == 0 || num_heads == 0 {
        return Err(DnnError::InvalidArgument(
            "rope_neox: seq_len and num_heads must be non-zero".into(),
        ));
    }

    let total = (seq_len as usize) * (num_heads as usize) * (head_dim as usize);
    if input.len() < total {
        return Err(DnnError::BufferTooSmall {
            expected: total * 4,
            actual: input.len() * 4,
        });
    }
    if output.len() < total {
        return Err(DnnError::BufferTooSmall {
            expected: total * 4,
            actual: output.len() * 4,
        });
    }

    let half = rotary_dim / 2;
    let total_pairs = (seq_len as u64) * (num_heads as u64) * (half as u64);

    let sm = handle.sm_version();
    let kernel_name = "rope_neox_half_split_f32".to_string();
    // The entry name is a fixed literal, yet the generator *unrolls* the
    // pass-through tail `[rotary_dim, head_dim)` at code-generation time. Both
    // dimensions must therefore discriminate the cache key.
    let kernel = handle.get_or_compile_kernel(
        &cache_key_with(&kernel_name, sm, &format!("hd={head_dim},rd={rotary_dim}")),
        &kernel_name,
        || generate_rope_neox_ptx::<f32>(&kernel_name, sm, head_dim, rotary_dim),
    )?;

    let block_dim = 256u32;
    let grid_x = grid_size_for(total_pairs as u32, block_dim);

    let params = LaunchParams::builder()
        .grid(Dim3::new(grid_x, 1, 1))
        .block(Dim3::new(block_dim, 1, 1))
        .shared_mem(0)
        .build();

    kernel.kernel().launch(
        &params,
        handle.stream(),
        &(
            input.as_device_ptr(),
            output.as_device_ptr(),
            seq_len,
            num_heads,
            head_dim,
            rotary_dim,
            base,
            total_pairs as u32,
        ),
    )?;

    Ok(())
}

/// Generates the GPT-NeoX half-split partial-rotary RoPE PTX kernel.
///
/// Each thread handles one `(pos, head, pair)` triple, where `pair` ranges over
/// `[0, rotary_dim/2)`. The thread:
///
/// 1. Decomposes its linear id over `[seq_len, num_heads, half]` (pair fastest).
/// 2. Computes the base element offset `pos·(num_heads·head_dim) +
///    head·head_dim` for the `[seq_len, num_heads, head_dim]` layout.
/// 3. Computes `freq = base^(−2·pair/rotary_dim) = exp2((−2·pair/rotary_dim)·
///    log2(base))` and `θ = pos·freq` using the on-device `lg2`/`ex2`/`sin`/
///    `cos` approximations.
/// 4. Loads `xi = in[base+pair]`, `xj = in[base+pair+half]`, and writes the
///    rotated pair to `out`.
/// 5. For `pair == 0`, copies the unrotated tail `[rotary_dim, head_dim)`
///    (unrolled at codegen over the compile-time tail range).
#[allow(clippy::too_many_lines, clippy::extra_unused_type_parameters)]
fn generate_rope_neox_ptx<T: GpuFloat>(
    kernel_name: &str,
    sm: SmVersion,
    head_dim: u32,
    rotary_dim: u32,
) -> DnnResult<String> {
    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("seq_len", PtxType::U32)
        .param("num_heads", PtxType::U32)
        .param("head_dim", PtxType::U32)
        .param("rotary_dim", PtxType::U32)
        .param("base", PtxType::F32)
        .param("total_pairs", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let total = b.load_param_u32("total_pairs");

            b.if_lt_u32(gid, total, |b| {
                b.comment("=== GPT-NeoX half-split partial-rotary RoPE ===");
                b.comment(
                    "half-split pairing: i <-> i + rotary_dim/2; freq denom = rotary_dim; tail [rotary_dim,head_dim) pass-through",
                );

                let gid = b.global_thread_id_x();
                let rotary_dim_reg = b.load_param_u32("rotary_dim");
                let num_heads_reg = b.load_param_u32("num_heads");
                let head_dim_reg = b.load_param_u32("head_dim");

                let half = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shr.u32 {half}, {rotary_dim_reg}, 1;"));

                // Decompose gid over [seq_len, num_heads, half] (pair fastest).
                let pair_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {pair_idx}, {gid}, {half};"));
                let tmp = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {tmp}, {gid}, {half};"));
                let head_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {head_idx}, {tmp}, {num_heads_reg};"));
                let pos = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {pos}, {tmp}, {num_heads_reg};"));

                // base_off = pos*(num_heads*head_dim) + head_idx*head_dim
                let nh_hd = b.mul_lo_u32(num_heads_reg.clone(), head_dim_reg.clone());
                let pos_off = b.mul_lo_u32(pos.clone(), nh_hd);
                let head_off = b.mul_lo_u32(head_idx.clone(), head_dim_reg.clone());
                let base_off = b.add_u32(pos_off, head_off);

                // idx_i = base_off + pair_idx ; idx_j = base_off + pair_idx + half
                let idx_i = b.add_u32(base_off.clone(), pair_idx.clone());
                let bp = b.add_u32(base_off.clone(), pair_idx.clone());
                let idx_j = b.add_u32(bp, half.clone());

                // freq = base^(-2*pair/rotary_dim) = exp2( (-2*pair/rotary_dim) * log2(base) )
                let pair_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {pair_f}, {pair_idx};"));
                let rot_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {rot_f}, {rotary_dim_reg};"));
                let two = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {two}, 0F{bits:08X};", bits = 2.0f32.to_bits()));
                let two_pair = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {two_pair}, {pair_f}, {two};"));
                let ratio = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {ratio}, {two_pair}, {rot_f};"));
                let neg_ratio = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("neg.f32 {neg_ratio}, {ratio};"));
                let base_val = b.load_param_f32("base");
                let log2_base = b.lg2_approx_f32(base_val);
                let exp_arg = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {exp_arg}, {neg_ratio}, {log2_base};"));
                let freq = b.ex2_approx_f32(exp_arg);

                // angle = pos * freq
                let pos_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {pos_f}, {pos};"));
                let angle = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {angle}, {pos_f}, {freq};"));

                let c = b.cos_approx_f32(angle.clone());
                let s = b.sin_approx_f32(angle);

                // Load inputs (re-load in_ptr per access).
                let in_base = b.load_param_u64("in_ptr");
                let addr_i = b.f32_elem_addr(in_base, idx_i.clone());
                let xi = b.load_global_f32(addr_i);
                let in_base2 = b.load_param_u64("in_ptr");
                let addr_j = b.f32_elem_addr(in_base2, idx_j.clone());
                let xj = b.load_global_f32(addr_j);

                // out_i = xi*c - xj*s
                let m1 = {
                    let r = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {r}, {xi}, {c};"));
                    r
                };
                let m2 = {
                    let r = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {r}, {xj}, {s};"));
                    r
                };
                let out_i = b.sub_f32(m1, m2);
                // out_j = xi*s + xj*c  (fma: xj*c + (xi*s))
                let m3 = {
                    let r = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {r}, {xi}, {s};"));
                    r
                };
                let out_j = b.fma_f32(xj, c, m3);

                // Store outputs (re-load out_ptr; idx_i/idx_j were cloned above).
                let out_base = b.load_param_u64("out_ptr");
                let oaddr_i = b.f32_elem_addr(out_base, idx_i);
                b.store_global_f32(oaddr_i, out_i);
                let out_base2 = b.load_param_u64("out_ptr");
                let oaddr_j = b.f32_elem_addr(out_base2, idx_j);
                b.store_global_f32(oaddr_j, out_j);

                // Pass-through tail when pair_idx == 0, unrolled over [rotary_dim, head_dim).
                if rotary_dim < head_dim {
                    let one = b.mov_imm_u32(1);
                    b.if_lt_u32(pair_idx.clone(), one, |b| {
                        b.comment(
                            "pair_idx==0 thread copies the unrotated tail [rotary_dim, head_dim)",
                        );
                        for k in rotary_dim..head_dim {
                            let num_heads_reg = b.load_param_u32("num_heads");
                            let head_dim_reg = b.load_param_u32("head_dim");
                            let gid = b.global_thread_id_x();
                            let rotary_dim_reg = b.load_param_u32("rotary_dim");
                            let half = b.alloc_reg(PtxType::U32);
                            b.raw_ptx(&format!("shr.u32 {half}, {rotary_dim_reg}, 1;"));
                            let tmp = b.alloc_reg(PtxType::U32);
                            b.raw_ptx(&format!("div.u32 {tmp}, {gid}, {half};"));
                            let head_idx = b.alloc_reg(PtxType::U32);
                            b.raw_ptx(&format!("rem.u32 {head_idx}, {tmp}, {num_heads_reg};"));
                            let pos = b.alloc_reg(PtxType::U32);
                            b.raw_ptx(&format!("div.u32 {pos}, {tmp}, {num_heads_reg};"));
                            let nh_hd = b.mul_lo_u32(num_heads_reg, head_dim_reg.clone());
                            let pos_off = b.mul_lo_u32(pos, nh_hd);
                            let head_off = b.mul_lo_u32(head_idx, head_dim_reg);
                            let base_off = b.add_u32(pos_off, head_off);
                            let k_reg = b.mov_imm_u32(k);
                            let idx_k = b.add_u32(base_off, k_reg);
                            let in_base = b.load_param_u64("in_ptr");
                            let addr_k = b.f32_elem_addr(in_base, idx_k.clone());
                            let val_k = b.load_global_f32(addr_k);
                            let out_base = b.load_param_u64("out_ptr");
                            let oaddr_k = b.f32_elem_addr(out_base, idx_k);
                            b.store_global_f32(oaddr_k, val_k);
                        }
                    });
                }
            });

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
    fn neox_ptx_generation_succeeds() {
        let ptx = generate_rope_neox_ptx::<f32>("test_rope_neox", SmVersion::Sm80, 8, 8);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains(".entry test_rope_neox"));
        assert!(text.contains("NeoX half-split"));
        assert!(text.contains("rotary_dim"));
    }

    #[test]
    fn neox_ptx_generation_partial() {
        let ptx = generate_rope_neox_ptx::<f32>("test_rope_neox_partial", SmVersion::Sm80, 8, 4);
        assert!(ptx.is_ok());
    }
}
