//! INT8 symmetric quantization and dequantization.
//!
//! Uses symmetric quantization: `scale = absmax / 127.0`, then each value
//! is divided by scale and clamped to [-127, 127].

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::ptx_helpers::*;
use crate::types::{TensorDesc, TensorDescMut};

/// Block size for INT8 quantization kernels.
const INT8_QUANT_BLOCK: u32 = 256;

/// INT8 maximum representable magnitude (symmetric).
const INT8_MAX: f64 = 127.0;

/// Quantizes a floating-point tensor to INT8 using symmetric quantization.
///
/// The scale factor is computed as `absmax(input) / 127.0` and stored in
/// `scale[0]`. Each element is then `round(clamp(input[i] / scale, -127, 127))`.
///
/// # Errors
///
/// Returns errors if buffers are too small or PTX generation fails.
pub fn quantize_to_int8<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut DeviceBuffer<i8>,
    scale: &mut DeviceBuffer<f32>,
) -> DnnResult<()> {
    let n = input.numel();
    if n == 0 {
        return Ok(());
    }

    if output.len() < n {
        return Err(DnnError::BufferTooSmall {
            expected: n,
            actual: output.len(),
        });
    }
    if scale.is_empty() {
        return Err(DnnError::BufferTooSmall {
            expected: 1,
            actual: 0,
        });
    }

    let n_u32 = n as u32;

    // Step 1: Absmax (reuse FP8's absmax kernel pattern, but with INT8 scale divisor)
    let absmax_ptx = generate_int8_absmax_ptx::<T>(handle.sm_version())?;
    let absmax_mod = Arc::new(Module::from_ptx(&absmax_ptx)?);
    let absmax_name = format!("dnn_int8_absmax_{}", T::NAME);
    let absmax_kernel = Kernel::from_module(absmax_mod, &absmax_name)?;

    let params1 = LaunchParams::new(1u32, INT8_QUANT_BLOCK);
    let args1 = (input.ptr, scale.as_device_ptr(), n_u32);

    absmax_kernel
        .launch(&params1, handle.stream(), &args1)
        .map_err(|e| DnnError::LaunchFailed(format!("int8 absmax: {e}")))?;

    // Step 2: Quantize
    let quant_ptx = generate_int8_quant_ptx::<T>(handle.sm_version())?;
    let quant_mod = Arc::new(Module::from_ptx(&quant_ptx)?);
    let quant_name = format!("dnn_int8_quantize_{}", T::NAME);
    let quant_kernel = Kernel::from_module(quant_mod, &quant_name)?;

    let grid = grid_size_for(n_u32, INT8_QUANT_BLOCK);
    let params2 = LaunchParams::new(grid, INT8_QUANT_BLOCK);
    let args2 = (
        input.ptr,
        output.as_device_ptr(),
        scale.as_device_ptr(),
        n_u32,
    );

    quant_kernel
        .launch(&params2, handle.stream(), &args2)
        .map_err(|e| DnnError::LaunchFailed(format!("int8 quantize: {e}")))?;

    Ok(())
}

/// Dequantizes INT8 data back to floating-point.
///
/// `out[i] = (float)in[i] * scale[0]`
///
/// # Errors
///
/// Returns errors if buffers are too small.
pub fn dequantize_from_int8<T: GpuFloat>(
    handle: &DnnHandle,
    input: &DeviceBuffer<i8>,
    scale: &DeviceBuffer<f32>,
    output: &mut TensorDescMut<T>,
    n: u32,
) -> DnnResult<()> {
    if n == 0 {
        return Ok(());
    }

    let n_usize = n as usize;
    if input.len() < n_usize {
        return Err(DnnError::BufferTooSmall {
            expected: n_usize,
            actual: input.len(),
        });
    }
    if scale.is_empty() {
        return Err(DnnError::BufferTooSmall {
            expected: 1,
            actual: 0,
        });
    }
    if output.numel() < n_usize {
        return Err(DnnError::BufferTooSmall {
            expected: n_usize * T::SIZE,
            actual: output.numel() * T::SIZE,
        });
    }

    let ptx = generate_int8_dequant_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let name = format!("dnn_int8_dequantize_{}", T::NAME);
    let kernel = Kernel::from_module(module, &name)?;

    let grid = grid_size_for(n, INT8_QUANT_BLOCK);
    let params = LaunchParams::new(grid, INT8_QUANT_BLOCK);
    let args = (input.as_device_ptr(), scale.as_device_ptr(), output.ptr, n);

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("int8 dequantize: {e}")))?;

    Ok(())
}

/// Generates PTX for INT8 absmax + scale computation.
fn generate_int8_absmax_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_int8_absmax_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(INT8_QUANT_BLOCK)
        .shared_mem("smem", PtxType::F32, INT8_QUANT_BLOCK as usize)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let bdim = b.block_dim_x();
            let n_reg = b.load_param_u32("n");
            let in_ptr = b.load_param_u64("in_ptr");

            let partial = load_float_imm::<f32>(b, 0.0);
            let i = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {i}, {tid};"));

            let loop_lbl = b.fresh_label("i8abs_loop");
            let end_lbl = b.fresh_label("i8abs_end");
            b.label(&loop_lbl);
            let p_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {p_done}, {i}, {n_reg};"));
            b.branch_if(p_done, &end_lbl);

            let addr = b.byte_offset_addr(in_ptr.clone(), i.clone(), T::size_u32());
            let val = load_global_float::<T>(b, addr);
            let val_f32 = if T::PTX_TYPE == PtxType::F64 {
                b.cvt_f64_to_f32(val)
            } else {
                val
            };
            let abs_v = b.abs_f32(val_f32);
            let new_p = b.max_f32(partial.clone(), abs_v);
            b.raw_ptx(&format!("mov.f32 {partial}, {new_p};"));
            b.raw_ptx(&format!("add.u32 {i}, {i}, {bdim};"));
            b.branch(&loop_lbl);
            b.label(&end_lbl);

            b.raw_ptx(&format!("st.shared.f32 [smem + {tid} * 4], {partial};"));
            b.bar_sync(0);

            // Tree reduction
            let stride = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {stride}, {bdim}, 1;"));
            let red_loop = b.fresh_label("i8abs_red");
            let red_end = b.fresh_label("i8abs_red_end");
            b.label(&red_loop);
            let p_s = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.eq.u32 {p_s}, {stride}, 0;"));
            b.branch_if(p_s, &red_end);
            let p_a = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lt.u32 {p_a}, {tid}, {stride};"));
            let skip = b.fresh_label("i8abs_skip");
            let inv = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("not.pred {inv}, {p_a};"));
            b.branch_if(inv, &skip);
            let other = b.add_u32(tid.clone(), stride.clone());
            let a = b.alloc_reg(PtxType::F32);
            let bv = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("ld.shared.f32 {a}, [smem + {tid} * 4];"));
            b.raw_ptx(&format!("ld.shared.f32 {bv}, [smem + {other} * 4];"));
            let m = b.max_f32(a, bv);
            b.raw_ptx(&format!("st.shared.f32 [smem + {tid} * 4], {m};"));
            b.label(&skip);
            b.bar_sync(0);
            b.raw_ptx(&format!("shr.u32 {stride}, {stride}, 1;"));
            b.branch(&red_loop);
            b.label(&red_end);

            // Thread 0: scale = absmax / 127
            let p_t0 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.eq.u32 {p_t0}, {tid}, 0;"));
            let skip_w = b.fresh_label("i8abs_skip_w");
            let inv_t0 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("not.pred {inv_t0}, {p_t0};"));
            b.branch_if(inv_t0, &skip_w);

            let absmax = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("ld.shared.f32 {absmax}, [smem];"));
            let int8_max = load_float_imm::<f32>(b, INT8_MAX);
            let sc = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("div.rn.f32 {sc}, {absmax}, {int8_max};"));
            let eps = load_float_imm::<f32>(b, 1e-12);
            let safe_sc = b.max_f32(sc, eps);

            let out_ptr = b.load_param_u64("out_ptr");
            b.raw_ptx(&format!("st.global.f32 [{out_ptr}], {safe_sc};"));

            b.label(&skip_w);
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("int8_absmax: {e}")))?;

    Ok(ptx)
}

/// Generates PTX for INT8 quantization kernel.
fn generate_int8_quant_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_int8_quantize_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(INT8_QUANT_BLOCK)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("scale_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let in_ptr = b.load_param_u64("in_ptr");
                let scale_ptr = b.load_param_u64("scale_ptr");

                let scale = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.global.f32 {scale}, [{scale_ptr}];"));

                let addr = b.byte_offset_addr(in_ptr, gid.clone(), T::size_u32());
                let val = load_global_float::<T>(b, addr);
                let val_f32 = if T::PTX_TYPE == PtxType::F64 {
                    b.cvt_f64_to_f32(val)
                } else {
                    val
                };

                let scaled = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {scaled}, {val_f32}, {scale};"));

                let max_v = load_float_imm::<f32>(b, INT8_MAX);
                let neg_max = b.neg_f32(max_v.clone());
                let cl = b.max_f32(scaled, neg_max);
                let cl = b.min_f32(cl, max_v);

                let rounded = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rni.f32.f32 {rounded}, {cl};"));
                let as_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("cvt.rzi.s32.f32 {as_s32}, {rounded};"));

                // Store as i8 (s8)
                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, 1u32);
                b.raw_ptx(&format!("st.global.s8 [{out_addr}], {as_s32};"));
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("int8_quantize: {e}")))?;

    Ok(ptx)
}

/// Generates PTX for INT8 dequantization.
fn generate_int8_dequant_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_int8_dequantize_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(INT8_QUANT_BLOCK)
        .param("in_ptr", PtxType::U64)
        .param("scale_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let in_ptr = b.load_param_u64("in_ptr");
                let scale_ptr = b.load_param_u64("scale_ptr");

                let scale = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.global.f32 {scale}, [{scale_ptr}];"));

                let in_addr = b.byte_offset_addr(in_ptr, gid.clone(), 1u32);
                let raw = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("ld.global.s8 {raw}, [{in_addr}];"));

                let float_val = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.s32 {float_val}, {raw};"));
                let result_f32 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {result_f32}, {float_val}, {scale};"));

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, T::size_u32());
                if T::PTX_TYPE == PtxType::F64 {
                    let r64 = b.cvt_f32_to_f64(result_f32);
                    store_global_float::<T>(b, out_addr, r64);
                } else {
                    store_global_float::<T>(b, out_addr, result_f32);
                }
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("int8_dequantize: {e}")))?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// BlockQuantizedInt8
// ---------------------------------------------------------------------------

/// Block-quantized INT8 with per-block scale factors (for GPTQ/AWQ).
///
/// Each contiguous block of `block_size` elements shares one `f32` scale
/// factor. Quantization is symmetric: `scale = absmax(block) / 127.0`.
///
/// # Memory layout
///
/// ```text
/// data:   [q0, q1, ..., q_{n-1}]          — i8, n elements
/// scales: [s0, s1, ..., s_{m-1}]          — f32, m = ceil(n / block_size)
/// ```
///
/// # Compression ratio
///
/// Each original `f32` (4 bytes) becomes one `i8` (1 byte) plus a shared
/// scale (`f32 / block_size` amortised), giving approximately 3.7× compression
/// for `block_size = 64`.
///
/// # Example
///
/// ```rust
/// use oxicuda_dnn::quantize::int8_quantize::BlockQuantizedInt8;
/// let input: Vec<f32> = (0..64).map(|i| i as f32).collect();
/// let q = BlockQuantizedInt8::quantize(&input, 64);
/// let deq = q.dequantize();
/// // round-trip error < 0.5 * scale
/// ```
#[derive(Debug, Clone)]
pub struct BlockQuantizedInt8 {
    /// INT8 quantized values.
    pub data: Vec<i8>,
    /// Per-block scale factors (`absmax / 127.0`).
    pub scales: Vec<f32>,
    /// Number of elements per block.
    pub block_size: usize,
    /// Original number of elements (before padding to block boundary).
    pub orig_len: usize,
}

impl BlockQuantizedInt8 {
    /// Quantizes a slice of `f32` values to INT8 using per-block symmetric quantization.
    ///
    /// Each block of `block_size` contiguous elements is quantized with its own
    /// scale factor derived from the block's absolute maximum value.
    ///
    /// # Panics
    ///
    /// Panics if `block_size == 0`.
    #[must_use]
    pub fn quantize(input: &[f32], block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        let n = input.len();
        let num_blocks = n.div_ceil(block_size);
        let mut data = Vec::with_capacity(n);
        let mut scales = Vec::with_capacity(num_blocks);

        for block in input.chunks(block_size) {
            let max_abs = block.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales.push(scale);
            for &x in block {
                let q = (x / scale).round().clamp(-128.0, 127.0) as i8;
                data.push(q);
            }
        }

        Self {
            data,
            scales,
            block_size,
            orig_len: n,
        }
    }

    /// Dequantizes the INT8 data back to `f32` using the stored per-block scales.
    ///
    /// Returns exactly `orig_len` elements.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.orig_len);
        for (block_idx, block) in self.data.chunks(self.block_size).enumerate() {
            if block_idx >= self.scales.len() {
                break;
            }
            let scale = self.scales[block_idx];
            for &q in block {
                output.push(q as f32 * scale);
            }
        }
        output.truncate(self.orig_len);
        output
    }

    /// Returns the compression ratio: original bytes / quantized bytes.
    ///
    /// Original: `orig_len × 4` bytes (f32).
    /// Quantized: `data.len() × 1` + `scales.len() × 4` bytes.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let quantized_bytes = self.data.len() + self.scales.len() * 4;
        let original_bytes = self.orig_len * 4;
        if quantized_bytes == 0 {
            return 1.0;
        }
        original_bytes as f32 / quantized_bytes as f32
    }

    /// Returns the number of blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.scales.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_absmax_ptx_f32() {
        let ptx = generate_int8_absmax_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn int8_quant_ptx_f32() {
        let ptx = generate_int8_quant_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn int8_dequant_ptx_f32() {
        let ptx = generate_int8_dequant_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn int8_quant_ptx_f64() {
        let ptx = generate_int8_quant_ptx::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    // -----------------------------------------------------------------------
    // BlockQuantizedInt8 tests (CPU-only, no GPU required)
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_int8_quantize_round_trip() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect();
        let quant = BlockQuantizedInt8::quantize(&input, 64);
        let deq = quant.dequantize();
        assert_eq!(
            deq.len(),
            input.len(),
            "dequantize must return orig_len elements"
        );
        for (i, (&orig, &restored)) in input.iter().zip(deq.iter()).enumerate() {
            assert!(
                (orig - restored).abs() < 0.05,
                "Round-trip error at {}: {} vs {}",
                i,
                orig,
                restored
            );
        }
    }

    #[test]
    fn test_block_int8_compression_ratio() {
        // 1024 f32 → 1024 i8 + 16 f32 scales (block_size=64)
        // quantized_bytes = 1024 + 16*4 = 1088
        // original_bytes = 1024*4 = 4096
        // ratio = 4096 / 1088 ≈ 3.76
        let input = vec![1.0f32; 1024];
        let quant = BlockQuantizedInt8::quantize(&input, 64);
        let ratio = quant.compression_ratio();
        assert!(ratio > 3.0, "Expected >3× compression, got {ratio}×");
    }

    #[test]
    fn test_block_int8_constant_block() {
        let input = vec![std::f32::consts::PI; 128];
        let quant = BlockQuantizedInt8::quantize(&input, 64);
        let deq = quant.dequantize();
        for &v in &deq {
            assert!(
                (v - std::f32::consts::PI).abs() < 0.03,
                "Constant block should round-trip cleanly, got {v}"
            );
        }
    }

    #[test]
    fn test_block_int8_zero_input() {
        let input = vec![0.0f32; 64];
        let quant = BlockQuantizedInt8::quantize(&input, 64);
        let deq = quant.dequantize();
        assert!(
            deq.iter().all(|&v| v == 0.0),
            "Zero input should produce zero output"
        );
    }

    #[test]
    fn test_block_int8_scale_count() {
        // 256 elements, block_size=64 → 4 blocks → 4 scales
        let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let quant = BlockQuantizedInt8::quantize(&input, 64);
        assert_eq!(quant.num_blocks(), 4);
        assert_eq!(quant.scales.len(), 4);
    }

    #[test]
    fn test_block_int8_non_multiple_len() {
        // 100 elements, block_size=64 → 2 blocks (64 + 36)
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let quant = BlockQuantizedInt8::quantize(&input, 64);
        let deq = quant.dequantize();
        assert_eq!(deq.len(), 100, "dequantize must return orig_len=100");
        assert_eq!(quant.num_blocks(), 2);
    }

    #[test]
    fn test_block_int8_max_value_saturation() {
        // Very large values should be clamped to ±127.
        let input = vec![1e30f32, -1e30f32, 0.0f32, 1e30f32];
        let quant = BlockQuantizedInt8::quantize(&input, 4);
        // All values should survive round-trip (scale = 1e30/127)
        let deq = quant.dequantize();
        assert_eq!(deq.len(), 4);
        assert!(deq[0] > 0.0, "large positive should round-trip positive");
        assert!(deq[1] < 0.0, "large negative should round-trip negative");
        assert_eq!(deq[2], 0.0, "zero should round-trip as zero");
    }

    // -----------------------------------------------------------------------
    // Quality-gate: INT8 block scaling tests (GPTQ/AWQ style)
    // -----------------------------------------------------------------------

    /// For block_size=128 and n=256 elements, exactly 2 scale values are
    /// computed (one per 128-element group).
    #[test]
    fn test_block_int8_block_size_128_scale_count() {
        // 256 elements, block_size=128 → exactly 2 scale values
        let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let q = BlockQuantizedInt8::quantize(&input, 128);
        assert_eq!(
            q.scales.len(),
            2,
            "256 elements with block_size=128 must have 2 scale factors"
        );
        assert_eq!(q.num_blocks(), 2, "num_blocks() must match scales.len()");
    }

    /// GPTQ-style dequant precision: given known INT8 weights and scale,
    /// verify dequantized values match expected within floating-point precision.
    ///
    /// weights: [-128, 0, 127], scale: 0.1
    /// expected dequantized: [-12.8, 0.0, 12.7]
    #[test]
    fn test_block_int8_gptq_dequant_precision() {
        // Construct BlockQuantizedInt8 directly with known weights and scale.
        let bq = BlockQuantizedInt8 {
            data: vec![-128i8, 0i8, 127i8],
            scales: vec![0.1f32],
            block_size: 3,
            orig_len: 3,
        };

        let deq = bq.dequantize();
        assert_eq!(deq.len(), 3, "dequantized length must match orig_len");

        // -128 * 0.1 = -12.8
        let expected = [-12.8f32, 0.0f32, 12.7f32];
        for (i, (&got, &exp)) in deq.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "element {i}: got {got}, expected {exp} (scale=0.1)"
            );
        }
    }

    /// For n_elements=256, block_size=128 → exactly 2 scale values.
    /// Verifies boundary: the split occurs at exactly 128 elements.
    #[test]
    fn test_block_int8_block_boundary_256_elements() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.27).collect();
        let q = BlockQuantizedInt8::quantize(&input, 128);

        assert_eq!(
            q.scales.len(),
            2,
            "n=256, block_size=128 must produce exactly 2 scale values"
        );
        // Scale for block 0 is derived from elements [0..128]
        // Scale for block 1 is derived from elements [128..256]
        // Both must be positive (absmax > 0 for these inputs)
        assert!(q.scales[0] > 0.0, "block 0 scale must be positive");
        assert!(q.scales[1] > 0.0, "block 1 scale must be positive");
    }

    /// Verify that block scales are independent per-block, not shared globally.
    /// Two blocks with very different magnitudes should have very different scales.
    #[test]
    fn test_block_int8_per_block_independent_scales() {
        // Block 0: values near 0 (small magnitude)
        // Block 1: values near 127 (large magnitude)
        let mut input = vec![0.001f32; 64]; // small values
        input.extend(vec![100.0f32; 64]); // large values

        let q = BlockQuantizedInt8::quantize(&input, 64);
        assert_eq!(q.scales.len(), 2, "2 blocks expected");

        // Block 0 scale should be much smaller than block 1 scale
        let scale_ratio = q.scales[1] / q.scales[0];
        assert!(
            scale_ratio > 10.0,
            "block 1 scale should be much larger than block 0 scale (ratio: {scale_ratio:.1})"
        );
    }

    /// Round-trip with non-power-of-two block size: verifies correct block
    /// boundaries and that dequantize returns exactly orig_len elements.
    #[test]
    fn test_block_int8_block_size_100_round_trip() {
        let input: Vec<f32> = (0..300).map(|i| (i as f32 - 150.0) / 5.0).collect();
        let q = BlockQuantizedInt8::quantize(&input, 100);

        // 300 elements / 100 = exactly 3 blocks
        assert_eq!(q.num_blocks(), 3, "300 / 100 = 3 blocks");

        let deq = q.dequantize();
        assert_eq!(deq.len(), 300, "dequantize must return orig_len=300");

        // Round-trip error must be within 0.5 * max_scale
        let max_scale = q.scales.iter().cloned().fold(0.0f32, f32::max);
        for (i, (&orig, &restored)) in input.iter().zip(deq.iter()).enumerate() {
            assert!(
                (orig - restored).abs() <= max_scale * 0.5 + 1e-5,
                "element {i}: round-trip error {:.4} exceeds tolerance {:.4}",
                (orig - restored).abs(),
                max_scale * 0.5
            );
        }
    }
}
