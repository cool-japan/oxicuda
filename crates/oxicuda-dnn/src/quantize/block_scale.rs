//! Block-scaled quantization for Blackwell FP4.
//!
//! Each block of `block_size` consecutive elements shares a single scale
//! factor. This provides a middle ground between per-tensor and per-channel
//! quantization, enabling Blackwell-era micro-scaling for FP4/FP8 inference.
//!
//! Quantization:
//! 1. For each block, compute `absmax` of the block.
//! 2. `scale[block_idx] = absmax / max_repr`.
//! 3. `output[i] = round(clamp(input[i] / scale[block_idx], -max_repr, max_repr))`.
//!
//! Dequantization:
//! `output[i] = (float)input[i] * scale[block_idx]`.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::ptx_helpers::*;
use crate::types::TensorDesc;

/// Block size for the quantization kernel launch.
const BS_QUANT_BLOCK: u32 = 256;

/// Maximum representable value for Blackwell FP4 (simplified to 6.0 for
/// the 4-bit E2M1 format).
const FP4_MAX: f64 = 6.0;

/// Performs block-scaled quantization.
///
/// The input tensor is divided into contiguous blocks of `block_size` elements.
/// Each block gets its own scale factor stored in `scales`. The quantized
/// values are stored as u8 in `output`.
///
/// # Arguments
///
/// * `handle` — DNN handle.
/// * `input` — Input tensor.
/// * `output` — Output buffer for quantized u8 values.
/// * `scales` — Output buffer for per-block scale factors.
/// * `block_size` — Number of elements per block (must be power of 2, min 16).
///
/// # Errors
///
/// Returns [`DnnError::InvalidArgument`] if block_size is invalid.
/// Returns [`DnnError::BufferTooSmall`] if buffers are too small.
pub fn quantize_block_scaled<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<f32>,
    block_size: u32,
) -> DnnResult<()> {
    let n = input.numel();
    if n == 0 {
        return Ok(());
    }

    if block_size == 0 || !block_size.is_power_of_two() {
        return Err(DnnError::InvalidArgument(format!(
            "block_size must be a non-zero power of 2, got {block_size}"
        )));
    }
    if block_size < 16 {
        return Err(DnnError::InvalidArgument(format!(
            "block_size must be >= 16, got {block_size}"
        )));
    }

    if output.len() < n {
        return Err(DnnError::BufferTooSmall {
            expected: n,
            actual: output.len(),
        });
    }

    let n_u32 = n as u32;
    let num_blocks = n_u32.div_ceil(block_size);

    if scales.len() < num_blocks as usize {
        return Err(DnnError::BufferTooSmall {
            expected: num_blocks as usize * std::mem::size_of::<f32>(),
            actual: scales.len() * std::mem::size_of::<f32>(),
        });
    }

    // Step 1: Per-block absmax + scale computation
    let scale_ptx = generate_block_scale_ptx::<T>(handle.sm_version())?;
    let scale_mod = Arc::new(Module::from_ptx(&scale_ptx)?);
    let scale_name = format!("dnn_block_scale_{}", T::NAME);
    let scale_kernel = Kernel::from_module(scale_mod, &scale_name)?;

    // One thread block per quantization block
    let params1 = LaunchParams::new(num_blocks, BS_QUANT_BLOCK.min(block_size));
    let args1 = (
        input.ptr,
        scales.as_device_ptr(),
        n_u32,
        block_size,
        num_blocks,
    );

    scale_kernel
        .launch(&params1, handle.stream(), &args1)
        .map_err(|e| DnnError::LaunchFailed(format!("block_scale: {e}")))?;

    // Step 2: Per-element quantization using block scales
    let quant_ptx = generate_block_quant_ptx::<T>(handle.sm_version())?;
    let quant_mod = Arc::new(Module::from_ptx(&quant_ptx)?);
    let quant_name = format!("dnn_block_quantize_{}", T::NAME);
    let quant_kernel = Kernel::from_module(quant_mod, &quant_name)?;

    let grid = grid_size_for(n_u32, BS_QUANT_BLOCK);
    let params2 = LaunchParams::new(grid, BS_QUANT_BLOCK);
    let args2 = (
        input.ptr,
        output.as_device_ptr(),
        scales.as_device_ptr(),
        n_u32,
        block_size,
    );

    quant_kernel
        .launch(&params2, handle.stream(), &args2)
        .map_err(|e| DnnError::LaunchFailed(format!("block_quantize: {e}")))?;

    Ok(())
}

/// Generates PTX for per-block scale computation.
///
/// Each thread block processes one quantization block. Threads cooperatively
/// compute the absmax via reduction in shared memory, then thread 0 writes
/// `absmax / FP4_MAX` to the scales buffer.
fn generate_block_scale_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_block_scale_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(BS_QUANT_BLOCK)
        .shared_mem("smem", PtxType::F32, BS_QUANT_BLOCK as usize)
        .param("in_ptr", PtxType::U64)
        .param("scales_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("block_size", PtxType::U32)
        .param("num_blocks", PtxType::U32)
        .body(move |b| {
            let bid = b.block_id_x();
            let tid = b.thread_id_x();
            let bdim = b.block_dim_x();
            let n_reg = b.load_param_u32("n");
            let blk_sz = b.load_param_u32("block_size");
            let num_blk = b.load_param_u32("num_blocks");
            let in_ptr = b.load_param_u64("in_ptr");

            b.if_lt_u32(bid.clone(), num_blk, move |b| {
                // Block start offset
                let blk_start = b.mul_lo_u32(bid.clone(), blk_sz.clone());

                // Each thread processes elements strided by bdim within this block
                let partial = load_float_imm::<f32>(b, 0.0);
                let i = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {i}, {tid};"));

                let loop_lbl = b.fresh_label("bsc_loop");
                let end_lbl = b.fresh_label("bsc_end");
                b.label(&loop_lbl);
                let p_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_done}, {i}, {blk_sz};"));
                b.branch_if(p_done, &end_lbl);

                let global_idx = b.add_u32(blk_start.clone(), i.clone());
                // bounds check
                let p_bounds = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_bounds}, {global_idx}, {n_reg};"));
                let skip = b.fresh_label("bsc_skip");
                b.branch_if(p_bounds, &skip);

                let addr = b.byte_offset_addr(in_ptr.clone(), global_idx, T::size_u32());
                let val = load_global_float::<T>(b, addr);
                let val_f32 = if T::PTX_TYPE == PtxType::F64 {
                    b.cvt_f64_to_f32(val)
                } else {
                    val
                };
                let abs_v = b.abs_f32(val_f32);
                let new_p = b.max_f32(partial.clone(), abs_v);
                b.raw_ptx(&format!("mov.f32 {partial}, {new_p};"));

                b.label(&skip);
                b.raw_ptx(&format!("add.u32 {i}, {i}, {bdim};"));
                b.branch(&loop_lbl);
                b.label(&end_lbl);

                b.raw_ptx(&format!("st.shared.f32 [smem + {tid} * 4], {partial};"));
                b.bar_sync(0);

                // Tree reduction
                let stride = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shr.u32 {stride}, {bdim}, 1;"));
                let red_loop = b.fresh_label("bsc_red");
                let red_end = b.fresh_label("bsc_red_end");
                b.label(&red_loop);
                let p_s = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {p_s}, {stride}, 0;"));
                b.branch_if(p_s, &red_end);
                let p_a = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lt.u32 {p_a}, {tid}, {stride};"));
                let skip_r = b.fresh_label("bsc_skip_r");
                let inv = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {inv}, {p_a};"));
                b.branch_if(inv, &skip_r);
                let other = b.add_u32(tid.clone(), stride.clone());
                let a = b.alloc_reg(PtxType::F32);
                let bv = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.shared.f32 {a}, [smem + {tid} * 4];"));
                b.raw_ptx(&format!("ld.shared.f32 {bv}, [smem + {other} * 4];"));
                let m = b.max_f32(a, bv);
                b.raw_ptx(&format!("st.shared.f32 [smem + {tid} * 4], {m};"));
                b.label(&skip_r);
                b.bar_sync(0);
                b.raw_ptx(&format!("shr.u32 {stride}, {stride}, 1;"));
                b.branch(&red_loop);
                b.label(&red_end);

                // Thread 0: write scale
                let p_t0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {p_t0}, {tid}, 0;"));
                let skip_w = b.fresh_label("bsc_skip_w");
                let inv_t0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {inv_t0}, {p_t0};"));
                b.branch_if(inv_t0, &skip_w);

                let absmax = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.shared.f32 {absmax}, [smem];"));
                let fp4_max = load_float_imm::<f32>(b, FP4_MAX);
                let sc = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {sc}, {absmax}, {fp4_max};"));
                let eps = load_float_imm::<f32>(b, 1e-12);
                let safe_sc = b.max_f32(sc, eps);

                let scales_ptr = b.load_param_u64("scales_ptr");
                let sc_addr = b.byte_offset_addr(scales_ptr, bid, 4u32);
                b.raw_ptx(&format!("st.global.f32 [{sc_addr}], {safe_sc};"));

                b.label(&skip_w);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("block_scale: {e}")))?;

    Ok(ptx)
}

/// Generates PTX for per-element block-scaled quantization.
fn generate_block_quant_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_block_quantize_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(BS_QUANT_BLOCK)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("scales_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("block_size", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let blk_sz = b.load_param_u32("block_size");

                // Which block does this element belong to?
                let blk_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {blk_idx}, {gid}, {blk_sz};"));

                // Load per-block scale
                let scales_ptr = b.load_param_u64("scales_ptr");
                let sc_addr = b.byte_offset_addr(scales_ptr, blk_idx, 4u32);
                let scale = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.global.f32 {scale}, [{sc_addr}];"));

                // Load input
                let in_ptr = b.load_param_u64("in_ptr");
                let addr = b.byte_offset_addr(in_ptr, gid.clone(), T::size_u32());
                let val = load_global_float::<T>(b, addr);
                let val_f32 = if T::PTX_TYPE == PtxType::F64 {
                    b.cvt_f64_to_f32(val)
                } else {
                    val
                };

                // Quantize
                let scaled = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {scaled}, {val_f32}, {scale};"));

                let max_v = load_float_imm::<f32>(b, FP4_MAX);
                let neg_max = b.neg_f32(max_v.clone());
                let cl = b.max_f32(scaled, neg_max);
                let cl = b.min_f32(cl, max_v);

                let rounded = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rni.f32.f32 {rounded}, {cl};"));
                let as_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("cvt.rzi.s32.f32 {as_s32}, {rounded};"));

                // Bias to u8 range: val + 6 maps [-6, 6] to [0, 12]
                let bias = b.alloc_reg(PtxType::S32);
                let fp4_max_i = FP4_MAX as i32;
                b.raw_ptx(&format!("mov.s32 {bias}, {fp4_max_i};"));
                let biased = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("add.s32 {biased}, {as_s32}, {bias};"));

                // Clamp to [0, 255]
                let zero_s = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {zero_s}, 0;"));
                let max255 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {max255}, 255;"));
                let cl2 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("max.s32 {cl2}, {biased}, {zero_s};"));
                let cl3 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("min.s32 {cl3}, {cl2}, {max255};"));

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, 1u32);
                b.raw_ptx(&format!("st.global.u8 [{out_addr}], {cl3};"));
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("block_quantize: {e}")))?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Fp4BlockQuantizer — NVFP4 / OCP MXFP4 host-side helper
// ---------------------------------------------------------------------------

/// FP4 E2M1 block-scaled quantizer (NVFP4 / OCP MXFP4 compatible).
///
/// NVFP4 encodes each value in 4 bits using the E2M1 format:
/// 1 sign bit + 2 exponent bits + 1 mantissa bit.  With a bias of 1,
/// the maximum positive value representable is:
///
/// ```text
/// 0_11_1 → exponent = 3 − 1 = 2, mantissa = 1.5
/// value  = 1.5 × 2² = 6.0
/// ```
///
/// Block scaling assigns one `f32` scale factor per 32-element block
/// (mandated by both NVIDIA NVFP4 and the OCP MXFP4 standard).
/// The scale is chosen so that the maximum absolute value in the block
/// maps exactly to [`MAX_VALUE`](Self::MAX_VALUE):
///
/// ```text
/// scale = absmax(block) / 6.0
/// quantized[i] = round(clamp(input[i] / scale, -6.0, 6.0))
/// ```
///
/// # OCP MXFP4 compatibility
///
/// The OCP MXFP4 (Microscaling FP4) standard specifies the same 32-element
/// block size, making this implementation compatible with both NVIDIA NVFP4
/// hardware and the open OCP standard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fp4BlockQuantizer {
    /// Number of elements per quantization block. Must be 32 for NVFP4.
    pub block_size: usize,
}

impl Fp4BlockQuantizer {
    /// Maximum positive value representable in E2M1 FP4 format.
    ///
    /// `0_11_1` → exponent = 3 − 1 = 2 (bias = 1), mantissa = 1.5
    /// → value = 1.5 × 2² = **6.0**
    pub const MAX_VALUE: f32 = 6.0;

    /// The block size mandated by NVFP4 and OCP MXFP4.
    pub const REQUIRED_BLOCK_SIZE: usize = 32;

    /// Creates a new `Fp4BlockQuantizer`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `block_size` is not 32,
    /// which is the only block size permitted by the NVFP4 / OCP MXFP4 standard.
    pub fn new(block_size: usize) -> DnnResult<Self> {
        if block_size != Self::REQUIRED_BLOCK_SIZE {
            return Err(DnnError::InvalidArgument(format!(
                "NVFP4/OCP MXFP4 block size must be {}, got {block_size}",
                Self::REQUIRED_BLOCK_SIZE
            )));
        }
        Ok(Self { block_size })
    }

    /// Computes the per-block scale factor.
    ///
    /// The scale maps the maximum absolute value in the block to
    /// [`MAX_VALUE`](Self::MAX_VALUE):
    ///
    /// ```text
    /// scale = absmax(block) / 6.0
    /// ```
    ///
    /// If all elements are zero the scale is returned as `1.0` to avoid
    /// division by zero during dequantization.
    #[must_use]
    pub fn compute_scale(&self, block: &[f32]) -> f32 {
        let max_abs = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_abs == 0.0 {
            1.0
        } else {
            max_abs / Self::MAX_VALUE
        }
    }

    /// Returns the number of scale factors needed for `n` input elements.
    ///
    /// Each block of [`block_size`](Self::block_size) elements requires
    /// exactly one scale; a partial final block still requires a scale.
    #[must_use]
    pub fn num_scales(&self, n: usize) -> usize {
        n.div_ceil(self.block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_scale_ptx_f32() {
        let ptx = generate_block_scale_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let s = ptx.expect("should gen");
        assert!(s.contains("dnn_block_scale_f32"));
    }

    #[test]
    fn block_quant_ptx_f32() {
        let ptx = generate_block_quant_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn block_scale_ptx_f64() {
        let ptx = generate_block_scale_ptx::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn block_quant_ptx_f64() {
        let ptx = generate_block_quant_ptx::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    // -----------------------------------------------------------------------
    // Task 3: Fp4BlockQuantizer — FP4 / NVFP4 / OCP MXFP4 block-scaled quantization
    // -----------------------------------------------------------------------

    /// E2M1 maximum representable value is exactly 6.0.
    #[test]
    fn fp4_e2m1_max_value_is_6() {
        assert_eq!(
            Fp4BlockQuantizer::MAX_VALUE,
            6.0,
            "E2M1 max (0_11_1) must be 6.0"
        );
    }

    /// Constructing with block_size=32 succeeds.
    #[test]
    fn fp4_block_size_32_succeeds() {
        let q = Fp4BlockQuantizer::new(32);
        assert!(q.is_ok(), "new(32) must succeed");
    }

    /// Constructing with block_size=16 returns an error.
    #[test]
    fn fp4_block_size_must_be_32() {
        let q = Fp4BlockQuantizer::new(16);
        assert!(
            q.is_err(),
            "new(16) must return Err — NVFP4 only allows block_size=32"
        );
    }

    /// Constructing with block_size=64 returns an error.
    #[test]
    fn fp4_block_size_64_rejected() {
        let q = Fp4BlockQuantizer::new(64);
        assert!(
            q.is_err(),
            "new(64) must return Err — NVFP4 only allows block_size=32"
        );
    }

    /// A block of all zeros yields scale 1.0 (not 0.0, to avoid div-by-zero).
    #[test]
    fn fp4_scale_for_all_zeros_is_one() {
        let q = Fp4BlockQuantizer::new(32).expect("new(32) must succeed");
        let block = vec![0.0f32; 32];
        assert_eq!(
            q.compute_scale(&block),
            1.0,
            "all-zero block scale must be 1.0"
        );
    }

    /// A block whose maximum absolute value is exactly MAX_VALUE=6.0 yields scale 1.0.
    #[test]
    fn fp4_scale_for_max_input_is_1() {
        let q = Fp4BlockQuantizer::new(32).expect("new(32) must succeed");
        let block = vec![6.0f32; 32];
        let scale = q.compute_scale(&block);
        assert!(
            (scale - 1.0).abs() < 1e-6,
            "block with max=6.0 should yield scale=1.0, got {scale}"
        );
    }

    /// A block whose maximum absolute value is 12.0 yields scale 2.0.
    #[test]
    fn fp4_scale_for_large_input_is_2() {
        let q = Fp4BlockQuantizer::new(32).expect("new(32) must succeed");
        let block = vec![12.0f32; 32];
        let scale = q.compute_scale(&block);
        assert!(
            (scale - 2.0).abs() < 1e-6,
            "block with max=12.0 should yield scale=2.0, got {scale}"
        );
    }

    /// num_scales for exactly one block (32 elements) is 1.
    #[test]
    fn fp4_num_scales_32_elements() {
        let q = Fp4BlockQuantizer::new(32).expect("new(32) must succeed");
        assert_eq!(q.num_scales(32), 1);
    }

    /// num_scales for 3 full blocks (96 elements) is 3.
    #[test]
    fn fp4_num_scales_96_elements() {
        let q = Fp4BlockQuantizer::new(32).expect("new(32) must succeed");
        assert_eq!(q.num_scales(96), 3);
    }

    /// num_scales for 33 elements rounds up to 2 (partial block still needs a scale).
    #[test]
    fn fp4_num_scales_33_elements() {
        let q = Fp4BlockQuantizer::new(32).expect("new(32) must succeed");
        assert_eq!(q.num_scales(33), 2, "partial block must still get a scale");
    }

    /// num_scales for 0 elements is 0.
    #[test]
    fn fp4_num_scales_zero_elements() {
        let q = Fp4BlockQuantizer::new(32).expect("new(32) must succeed");
        assert_eq!(q.num_scales(0), 0);
    }

    /// Absmax scale: block with values [0..=31] scaled negatively — absmax is 31.
    #[test]
    fn fp4_absmax_scale_correct() {
        let q = Fp4BlockQuantizer::new(32).expect("new(32) must succeed");
        // Build a block whose maximum absolute value is 6.0
        let mut block = vec![0.0f32; 32];
        block[0] = -6.0; // max abs value
        block[1] = 3.0;
        block[2] = 1.5;
        let scale = q.compute_scale(&block);
        assert!(
            (scale - 1.0).abs() < 1e-6,
            "absmax=6.0 → scale=6.0/6.0=1.0, got {scale}"
        );
    }

    /// OCP MXFP4 compatibility: block_size must be 32.
    #[test]
    fn fp4_mxfp4_ocp_compatible() {
        let q = Fp4BlockQuantizer::new(32).expect("new(32) must succeed");
        assert_eq!(
            q.block_size,
            Fp4BlockQuantizer::REQUIRED_BLOCK_SIZE,
            "OCP MXFP4 requires block_size == {}",
            Fp4BlockQuantizer::REQUIRED_BLOCK_SIZE
        );
    }

    /// block_size field is stored as provided.
    #[test]
    fn fp4_block_size_stored_correctly() {
        let q = Fp4BlockQuantizer::new(32).expect("new(32) must succeed");
        assert_eq!(q.block_size, 32);
    }
}
