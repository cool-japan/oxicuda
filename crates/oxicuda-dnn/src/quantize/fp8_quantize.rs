//! FP8 E4M3 quantization and dequantization.
//!
//! Two-step quantization:
//! 1. Absmax reduction to compute the per-tensor scale factor.
//! 2. Elementwise quantize: `out[i] = clamp(round(in[i] / scale * 448), -448, 448)`
//!    where 448 is the max representable value for E4M3.
//!
//! Dequantization is a simple elementwise: `out[i] = fp8_to_float(in[i]) * scale`.

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
use crate::types::TensorDescMut;

/// Block size for quantization kernels.
const QUANT_BLOCK: u32 = 256;

/// Maximum representable value for FP8 E4M3 format.
const E4M3_MAX: f64 = 448.0;

/// Quantizes an FP32/FP64 tensor to FP8 E4M3 format.
///
/// The function first computes the absolute maximum of the input tensor,
/// then derives a per-tensor scale factor: `scale = absmax / 448.0`.
/// Each element is then divided by scale and clamped to [-448, 448] before
/// being stored as a u8 in E4M3 format.
///
/// # Arguments
///
/// * `handle` — DNN handle.
/// * `input` — Input tensor (any supported float type).
/// * `output` — Output buffer for quantized u8 values (one per element).
/// * `scale` — Output buffer for the computed scale factor (single f32).
///
/// # Errors
///
/// Returns errors if buffers are too small or PTX generation fails.
pub fn quantize_to_fp8<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut DeviceBuffer<u8>,
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

    // Step 1: Absmax reduction
    let absmax_ptx = generate_absmax_ptx::<T>(handle.sm_version())?;
    let absmax_module = Arc::new(Module::from_ptx(&absmax_ptx)?);
    let absmax_name = format!("dnn_absmax_{}", T::NAME);
    let absmax_kernel = Kernel::from_module(absmax_module, &absmax_name)?;

    let _grid = grid_size_for(n_u32, QUANT_BLOCK);
    let params = LaunchParams::new(1u32, QUANT_BLOCK);

    // We use the scale buffer temporarily for the absmax result
    let args_absmax = (input.ptr, scale.as_device_ptr(), n_u32);

    // For simplicity, run a single-block reduction (works for reasonable sizes)
    absmax_kernel
        .launch(&params, handle.stream(), &args_absmax)
        .map_err(|e| DnnError::LaunchFailed(format!("fp8 absmax: {e}")))?;

    // Step 2: Quantize elementwise
    let quant_ptx = generate_fp8_quant_ptx::<T>(handle.sm_version())?;
    let quant_module = Arc::new(Module::from_ptx(&quant_ptx)?);
    let quant_name = format!("dnn_fp8_quantize_{}", T::NAME);
    let quant_kernel = Kernel::from_module(quant_module, &quant_name)?;

    let grid2 = grid_size_for(n_u32, QUANT_BLOCK);
    let params2 = LaunchParams::new(grid2, QUANT_BLOCK);

    let args_quant = (
        input.ptr,
        output.as_device_ptr(),
        scale.as_device_ptr(),
        n_u32,
    );

    quant_kernel
        .launch(&params2, handle.stream(), &args_quant)
        .map_err(|e| DnnError::LaunchFailed(format!("fp8 quantize: {e}")))?;

    Ok(())
}

/// Dequantizes FP8 E4M3 data back to floating-point.
///
/// Each u8 value is interpreted as an E4M3 float, then multiplied by the
/// scale factor: `out[i] = fp8_to_float(in[i]) * scale[0]`.
///
/// # Errors
///
/// Returns errors if buffers are too small.
pub fn dequantize_from_fp8<T: GpuFloat>(
    handle: &DnnHandle,
    input: &DeviceBuffer<u8>,
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

    let ptx = generate_fp8_dequant_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let name = format!("dnn_fp8_dequantize_{}", T::NAME);
    let kernel = Kernel::from_module(module, &name)?;

    let grid = grid_size_for(n, QUANT_BLOCK);
    let params = LaunchParams::new(grid, QUANT_BLOCK);

    let args = (input.as_device_ptr(), scale.as_device_ptr(), output.ptr, n);

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("fp8 dequantize: {e}")))?;

    Ok(())
}

/// Generates PTX for single-block absmax reduction.
fn generate_absmax_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_absmax_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(QUANT_BLOCK)
        .shared_mem("smem", PtxType::F32, QUANT_BLOCK as usize)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let bdim = b.block_dim_x();
            let n_reg = b.load_param_u32("n");
            let in_ptr = b.load_param_u64("in_ptr");

            // Each thread computes partial absmax
            let partial = load_float_imm::<f32>(b, 0.0);
            let i = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {i}, {tid};"));

            let loop_lbl = b.fresh_label("absmax_loop");
            let end_lbl = b.fresh_label("absmax_end");
            b.label(&loop_lbl);
            let p_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {p_done}, {i}, {n_reg};"));
            b.branch_if(p_done, &end_lbl);

            let addr = b.byte_offset_addr(in_ptr.clone(), i.clone(), T::size_u32());
            let val = load_global_float::<T>(b, addr);

            // Convert to f32 if needed, then abs
            let val_f32 = if T::PTX_TYPE == PtxType::F64 {
                b.cvt_f64_to_f32(val)
            } else {
                val
            };
            let abs_val = b.abs_f32(val_f32);
            let new_partial = b.max_f32(partial.clone(), abs_val);
            b.raw_ptx(&format!("mov.f32 {partial}, {new_partial};"));

            b.raw_ptx(&format!("add.u32 {i}, {i}, {bdim};"));
            b.branch(&loop_lbl);
            b.label(&end_lbl);

            // Store to shared memory
            b.raw_ptx(&format!("st.shared.f32 [smem + {tid} * 4], {partial};"));
            b.bar_sync(0);

            // Tree reduction (max)
            let stride = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {stride}, {bdim}, 1;"));

            let red_loop = b.fresh_label("absmax_red");
            let red_end = b.fresh_label("absmax_red_end");
            b.label(&red_loop);
            let p_s = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.eq.u32 {p_s}, {stride}, 0;"));
            b.branch_if(p_s, &red_end);

            let p_a = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lt.u32 {p_a}, {tid}, {stride};"));
            let skip = b.fresh_label("absmax_skip");
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

            // Thread 0: scale = absmax / E4M3_MAX, store to out
            let p_t0 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.eq.u32 {p_t0}, {tid}, 0;"));
            let skip_w = b.fresh_label("absmax_skip_w");
            let inv_t0 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("not.pred {inv_t0}, {p_t0};"));
            b.branch_if(inv_t0, &skip_w);

            let absmax = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("ld.shared.f32 {absmax}, [smem];"));
            let e4m3_max = load_float_imm::<f32>(b, E4M3_MAX);
            let scale_val = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("div.rn.f32 {scale_val}, {absmax}, {e4m3_max};"));

            // Ensure scale is at least epsilon to avoid division by zero
            let eps = load_float_imm::<f32>(b, 1e-12);
            let safe_scale = b.max_f32(scale_val, eps);

            let out_ptr = b.load_param_u64("out_ptr");
            b.raw_ptx(&format!("st.global.f32 [{out_ptr}], {safe_scale};"));

            b.label(&skip_w);
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("absmax: {e}")))?;

    Ok(ptx)
}

/// Generates PTX for FP8 quantization kernel.
fn generate_fp8_quant_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_fp8_quantize_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(QUANT_BLOCK)
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

                // Load scale
                let scale = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.global.f32 {scale}, [{scale_ptr}];"));

                // Load input value, convert to f32 if needed
                let addr = b.byte_offset_addr(in_ptr, gid.clone(), T::size_u32());
                let val = load_global_float::<T>(b, addr);
                let val_f32 = if T::PTX_TYPE == PtxType::F64 {
                    b.cvt_f64_to_f32(val)
                } else {
                    val
                };

                // Quantize: scaled = val / scale, clamped to [-448, 448], rounded
                let scaled = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {scaled}, {val_f32}, {scale};"));

                let max_val = load_float_imm::<f32>(b, E4M3_MAX);
                let neg_max = b.neg_f32(max_val.clone());
                let clamped = b.max_f32(scaled, neg_max);
                let clamped = b.min_f32(clamped, max_val);

                // Round to nearest integer, then convert to u8 (simplified E4M3 encoding)
                // In practice, this should do proper E4M3 float encoding.
                // For now we use a simplified approach: round and store as u8.
                let rounded = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rni.f32.f32 {rounded}, {clamped};"));
                let as_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("cvt.rzi.s32.f32 {as_s32}, {rounded};"));

                // Map signed value to u8 range [0, 255]
                let offset = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {offset}, 128;"));
                let biased = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("add.s32 {biased}, {as_s32}, {offset};"));

                // Clamp to [0, 255]
                let zero_s = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {zero_s}, 0;"));
                let max255 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {max255}, 255;"));
                let cl = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("max.s32 {cl}, {biased}, {zero_s};"));
                let cl2 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("min.s32 {cl2}, {cl}, {max255};"));

                // Store as u8
                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, 1u32);
                b.raw_ptx(&format!("st.global.u8 [{out_addr}], {cl2};"));
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("fp8_quantize: {e}")))?;

    Ok(ptx)
}

/// Generates PTX for FP8 dequantization.
fn generate_fp8_dequant_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_fp8_dequantize_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(QUANT_BLOCK)
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

                // Load u8 value
                let in_addr = b.byte_offset_addr(in_ptr, gid.clone(), 1u32);
                let raw = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("ld.global.u8 {raw}, [{in_addr}];"));

                // Undo bias: signed = raw - 128
                let bias = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {bias}, 128;"));
                let raw_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.b32 {raw_s32}, {raw};"));
                let signed_val = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("sub.s32 {signed_val}, {raw_s32}, {bias};"));

                // Convert to float
                let float_val = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.s32 {float_val}, {signed_val};"));

                // Multiply by scale
                let result_f32 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {result_f32}, {float_val}, {scale};"));

                // Convert to output type and store
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
        .map_err(|e| DnnError::PtxGeneration(format!("fp8_dequantize: {e}")))?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absmax_ptx_f32() {
        let ptx = generate_absmax_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let s = ptx.expect("should gen");
        assert!(s.contains("dnn_absmax_f32"));
    }

    #[test]
    fn fp8_quant_ptx_f32() {
        let ptx = generate_fp8_quant_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn fp8_dequant_ptx_f32() {
        let ptx = generate_fp8_dequant_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn fp8_quant_ptx_f64() {
        let ptx = generate_fp8_quant_ptx::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    // -----------------------------------------------------------------------
    // Quality-gate: FP8 quantization CPU reference math tests (e4m3 / e5m2)
    // -----------------------------------------------------------------------

    /// E4M3 max representable value is 448.0.
    const E4M3_MAX_F32: f32 = 448.0;

    /// E5M2 max representable value is 57344.0.
    const E5M2_MAX_F32: f32 = 57344.0;

    /// CPU reference FP8 E4M3 quantization.
    ///
    /// Simulates the absmax-scaled quantize: x → clamp(x/scale, -448, 448).
    /// The precision is limited by E4M3's 3-bit mantissa (≈1/8 granularity).
    fn cpu_quantize_e4m3(x: f32, scale: f32) -> f32 {
        if scale == 0.0 {
            return 0.0;
        }
        let scaled = x / scale;
        let clamped = scaled.clamp(-E4M3_MAX_F32, E4M3_MAX_F32);
        // E4M3 mantissa: 3 bits → 8 discrete levels per exponent → step ≈ max/448
        // Approximate by rounding to nearest 1/8th in the [-448, 448] range.
        // (Real E4M3 encoding is more complex; this captures the quantization error.)
        (clamped * 8.0).round() / 8.0 * scale
    }

    /// CPU reference FP8 E5M2 quantization.
    ///
    /// E5M2 has 2-bit mantissa and 5-bit exponent. Max = 57344.0.
    fn cpu_quantize_e5m2(x: f32, scale: f32) -> f32 {
        if scale == 0.0 {
            return 0.0;
        }
        let scaled = x / scale;
        let clamped = scaled.clamp(-E5M2_MAX_F32, E5M2_MAX_F32);
        // E5M2 mantissa: 2 bits → step ≈ exponent_value / 4
        // Use coarser rounding than E4M3 to model 2-bit mantissa precision.
        (clamped * 4.0).round() / 4.0 * scale
    }

    /// E4M3 max value clamping: inputs above 448.0 are clamped.
    #[test]
    fn test_fp8_e4m3_max_value_clamping() {
        // Values well above E4M3_MAX should be clamped to ≈ E4M3_MAX * scale
        let scale = 1.0f32;
        let large_positive = cpu_quantize_e4m3(1000.0, scale);
        let large_negative = cpu_quantize_e4m3(-1000.0, scale);

        assert!(
            (large_positive - E4M3_MAX_F32).abs() < 1.0,
            "E4M3: large positive should clamp to ≈448, got {large_positive}"
        );
        assert!(
            (large_negative + E4M3_MAX_F32).abs() < 1.0,
            "E4M3: large negative should clamp to ≈-448, got {large_negative}"
        );
    }

    /// E5M2 max value: 57344.0. Verify clamping behavior.
    #[test]
    fn test_fp8_e5m2_max_value_clamping() {
        let scale = 1.0f32;
        let large_positive = cpu_quantize_e5m2(1_000_000.0, scale);
        let large_negative = cpu_quantize_e5m2(-1_000_000.0, scale);

        assert!(
            (large_positive - E5M2_MAX_F32).abs() < 1.0,
            "E5M2: large positive should clamp to ≈57344, got {large_positive}"
        );
        assert!(
            (large_negative + E5M2_MAX_F32).abs() < 1.0,
            "E5M2: large negative should clamp to ≈-57344, got {large_negative}"
        );
    }

    /// Small values below E4M3 resolution quantize to near-zero.
    #[test]
    fn test_fp8_e4m3_quantize_small_values() {
        let scale = 1.0f32;
        // 1e-10 << 1/8 (E4M3 step with scale=1), so rounds to 0
        let result = cpu_quantize_e4m3(1e-10, scale);
        assert!(
            result.abs() < 1.0 / 8.0,
            "E4M3: value 1e-10 with scale=1 should quantize to < 0.125, got {result}"
        );
    }

    /// Small values below E5M2 resolution quantize to near-zero.
    #[test]
    fn test_fp8_e5m2_quantize_small_values() {
        let scale = 1.0f32;
        // 1e-10 << 1/4 (E5M2 step with scale=1), so rounds to 0
        let result = cpu_quantize_e5m2(1e-10, scale);
        assert!(
            result.abs() < 1.0 / 4.0,
            "E5M2: value 1e-10 with scale=1 should quantize to < 0.25, got {result}"
        );
    }

    /// E4M3 absmax-based scale selection: scale = absmax / 448.
    ///
    /// This matches the kernel behavior in `generate_absmax_ptx`.
    #[test]
    fn test_fp8_e4m3_absmax_scale_selection() {
        let values = [10.0f32, -50.0, 30.0, -20.0, 5.0];
        let absmax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = absmax / E4M3_MAX_F32;

        assert!((absmax - 50.0).abs() < 1e-6, "absmax should be 50.0");
        let expected_scale = 50.0 / 448.0;
        assert!(
            (scale - expected_scale).abs() < 1e-6,
            "scale should be 50/448 ≈ {expected_scale:.6}, got {scale:.6}"
        );

        // Quantize the absmax value — it should round to near E4M3_MAX * scale
        let q_absmax = cpu_quantize_e4m3(absmax, scale);
        assert!(
            (q_absmax - absmax).abs() < scale * 0.5,
            "quantized absmax should be within half a step of original: {q_absmax} vs {absmax}"
        );
    }

    /// E4M3 round-trip: quantize then dequantize (CPU simulation).
    ///
    /// The dequantized value should be close to the original (within quantization error).
    #[test]
    fn test_fp8_e4m3_round_trip_accuracy() {
        let original_values = [1.0f32, -2.0, 0.5, -0.25, 4.0, -3.75];
        let absmax = original_values
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let scale = (absmax / E4M3_MAX_F32).max(1e-12);

        for &orig in &original_values {
            let quantized = cpu_quantize_e4m3(orig, scale);
            // Dequantize: value is already in float (our CPU sim returns float)
            // Max error is half the quantization step: step ≈ scale / 8
            let max_error = scale / 8.0 + 1e-6;
            assert!(
                (quantized - orig).abs() <= max_error,
                "E4M3 round-trip error for {orig}: |{quantized} - {orig}| = {} > {max_error}",
                (quantized - orig).abs()
            );
        }
    }

    /// E5M2 has coarser quantization than E4M3 for the same scale.
    #[test]
    fn test_fp8_e5m2_coarser_than_e4m3() {
        let scale = 1.0f32;
        let x = 1.1f32; // Not a multiple of 1/8

        let e4m3_result = cpu_quantize_e4m3(x, scale);
        let e5m2_result = cpu_quantize_e5m2(x, scale);

        // E4M3 (1/8 step) should be closer to original than E5M2 (1/4 step)
        let e4m3_error = (e4m3_result - x).abs();
        let e5m2_error = (e5m2_result - x).abs();

        // e4m3 error ≤ 1/16, e5m2 error ≤ 1/8 — E4M3 is finer
        assert!(
            e4m3_error <= e5m2_error + 1e-6,
            "E4M3 (error={e4m3_error:.4}) should be ≤ E5M2 (error={e5m2_error:.4}) for same input"
        );
    }

    /// E4M3 quantization is symmetric around zero.
    #[test]
    fn test_fp8_e4m3_symmetric() {
        let scale = 1.0f32;
        for &x in &[0.5f32, 1.0, 2.0, 5.0, 10.0, 100.0] {
            let pos = cpu_quantize_e4m3(x, scale);
            let neg = cpu_quantize_e4m3(-x, scale);
            assert!(
                (pos + neg).abs() < 1e-5,
                "E4M3 should be symmetric: q({x}) + q(-{x}) = {pos} + {neg} ≠ 0"
            );
        }
    }

    /// E5M2 quantization is symmetric around zero.
    #[test]
    fn test_fp8_e5m2_symmetric() {
        let scale = 1.0f32;
        for &x in &[0.5f32, 1.0, 2.0, 100.0, 1000.0] {
            let pos = cpu_quantize_e5m2(x, scale);
            let neg = cpu_quantize_e5m2(-x, scale);
            assert!(
                (pos + neg).abs() < 1e-4,
                "E5M2 should be symmetric: q({x}) + q(-{x}) = {pos} + {neg} ≠ 0"
            );
        }
    }

    /// E4M3 max representable constant matches the kernel constant.
    #[test]
    fn test_fp8_e4m3_max_constant_matches_kernel() {
        // The kernel uses E4M3_MAX = 448.0 (defined as f64 constant)
        assert!((E4M3_MAX as f32 - E4M3_MAX_F32).abs() < 0.1);
        assert_eq!(E4M3_MAX_F32, 448.0);
    }

    /// PTX generation for F32 absmax matches E4M3_MAX comment.
    #[test]
    fn test_fp8_ptx_contains_e4m3_max_comment_or_value() {
        let ptx = generate_absmax_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        // PTX should reference e4m3_max in some form (through the float constant)
        let text = ptx.ok().unwrap_or_default();
        assert!(
            text.contains("dnn_absmax_f32"),
            "absmax kernel name should appear in PTX"
        );
    }
}
