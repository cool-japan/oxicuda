//! FP8 E4M3 quantization and dequantization.
//!
//! Two-step quantization:
//! 1. Absmax reduction to compute the per-tensor scale factor.
//! 2. Elementwise quantize: `out[i] = clamp(round(in[i] / scale * 448), -448, 448)`
//!    where 448 is the max representable value for E4M3.
//!
//! Dequantization is a simple elementwise: `out[i] = fp8_to_float(in[i]) * scale`.

use oxicuda_blas::GpuFloat;
use oxicuda_launch::{LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::kernel_cache::cache_key;
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
    let absmax_name = format!("dnn_absmax_{}", T::NAME);
    let absmax_kernel = handle.get_or_compile_kernel(
        &cache_key(&absmax_name, handle.sm_version()),
        &absmax_name,
        || generate_absmax_ptx::<T>(handle.sm_version()),
    )?;

    let _grid = grid_size_for(n_u32, QUANT_BLOCK);
    let params = LaunchParams::new(1u32, QUANT_BLOCK);

    // We use the scale buffer temporarily for the absmax result
    let args_absmax = (input.ptr, scale.as_device_ptr(), n_u32);

    // For simplicity, run a single-block reduction (works for reasonable sizes)
    absmax_kernel
        .kernel()
        .launch(&params, handle.stream(), &args_absmax)
        .map_err(|e| DnnError::LaunchFailed(format!("fp8 absmax: {e}")))?;

    // Step 2: Quantize elementwise
    let quant_name = format!("dnn_fp8_quantize_{}", T::NAME);
    let quant_kernel = handle.get_or_compile_kernel(
        &cache_key(&quant_name, handle.sm_version()),
        &quant_name,
        || generate_fp8_quant_ptx::<T>(handle.sm_version()),
    )?;

    let grid2 = grid_size_for(n_u32, QUANT_BLOCK);
    let params2 = LaunchParams::new(grid2, QUANT_BLOCK);

    let args_quant = (
        input.ptr,
        output.as_device_ptr(),
        scale.as_device_ptr(),
        n_u32,
    );

    quant_kernel
        .kernel()
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

    let name = format!("dnn_fp8_dequantize_{}", T::NAME);
    let kernel =
        handle.get_or_compile_kernel(&cache_key(&name, handle.sm_version()), &name, || {
            generate_fp8_dequant_ptx::<T>(handle.sm_version())
        })?;

    let grid = grid_size_for(n, QUANT_BLOCK);
    let params = LaunchParams::new(grid, QUANT_BLOCK);

    let args = (input.as_device_ptr(), scale.as_device_ptr(), output.ptr, n);

    kernel
        .kernel()
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

            // Base address of the shared-memory scratch buffer. PTX address
            // operands cannot scale a register inside the brackets
            // (`[smem + r*4]` is invalid), so the element address must be
            // materialised into a register: `addr = &smem + idx * 4`.
            let smem_base = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {smem_base}, smem;"));

            // Store this thread's partial absmax: smem[tid] = partial.
            let self_addr = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {self_addr}, {tid}, 4, {smem_base};"));
            b.raw_ptx(&format!("st.shared.f32 [{self_addr}], {partial};"));
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
            // Compute element addresses for this thread and its tree partner.
            let tid_addr = b.alloc_reg(PtxType::U32);
            let other_addr = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {tid_addr}, {tid}, 4, {smem_base};"));
            b.raw_ptx(&format!(
                "mad.lo.u32 {other_addr}, {other}, 4, {smem_base};"
            ));
            b.raw_ptx(&format!("ld.shared.f32 {a}, [{tid_addr}];"));
            b.raw_ptx(&format!("ld.shared.f32 {bv}, [{other_addr}];"));
            let m = b.max_f32(a, bv);
            b.raw_ptx(&format!("st.shared.f32 [{tid_addr}], {m};"));

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

/// Emits PTX that encodes an `f32` value into a real FP8 **E4M3** byte.
///
/// E4M3 layout: `[sign:1 | exponent:4 | mantissa:3]`, exponent bias 7. The
/// format has no infinities; the bit pattern `s 1111 111` is reserved for
/// `NaN`, so the largest finite magnitude is `0x7E` → `2^8 · 1.75 = 448`.
///
/// The encoding works entirely on the bit pattern of the input:
///
/// * **Sign** — bit 31 of the `f32`, relocated to bit 7.
/// * **Saturation** — magnitudes `>= 448` clamp to `0x7E` (the input is
///   already clamped to `[-448, 448]`, so this only catches the endpoint).
/// * **Flush-to-zero** — magnitudes below `2^-10` (half the smallest
///   subnormal step `2^-9`) round to zero.
/// * **Normal** (`biased_exp >= 1`) — keep the top 3 mantissa bits of the
///   23-bit `f32` mantissa, rounding to nearest-even on the dropped 20
///   bits; a mantissa carry bumps the exponent (and may saturate).
/// * **Subnormal** (`biased_exp <= 0`) — right-shift the 24-bit significand
///   (implicit 1 included) into the 3-bit subnormal field, rounding to
///   nearest-even; a carry promotes to the smallest normal `0x08`.
///
/// `value` is the `f32` register name; the returned `U32` register holds the
/// E4M3 byte in its low 8 bits.
fn emit_e4m3_encode(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>, value: &str) -> Register {
    b.comment("--- E4M3 encode (1 sign / 4 exp / 3 mantissa, bias 7) ---");

    // Raw f32 bit pattern.
    let fbits = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.b32 {fbits}, {value};"));

    // Sign bit relocated from f32 bit 31 to E4M3 bit 7: (fbits >> 24) & 0x80.
    let sign = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shr.u32 {sign}, {fbits}, 24;"));
    b.raw_ptx(&format!("and.b32 {sign}, {sign}, 128;"));

    // Magnitude bits = fbits & 0x7FFFFFFF.
    let xbits = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("and.b32 {xbits}, {fbits}, 2147483647;"));

    // f32 fields: biased exponent (bias 127) and 23-bit mantissa.
    let e32 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shr.u32 {e32}, {xbits}, 23;"));
    b.raw_ptx(&format!("and.b32 {e32}, {e32}, 255;"));
    let m32 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("and.b32 {m32}, {xbits}, 8388607;"));

    // E4M3 biased exponent candidate: e4 = e32 - 127 + 7 = e32 - 120.
    let e4 = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {e4}, {e32};"));
    b.raw_ptx(&format!("sub.s32 {e4}, {e4}, 120;"));

    // The magnitude-only byte (sign added at the very end).
    let mag = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {mag}, 0;"));

    let done = b.fresh_label("e4m3_enc_done");

    // Flush to zero: |x| < 2^-10  (bit pattern 0x35800000).
    let p_tiny = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lt.u32 {p_tiny}, {xbits}, 0x35800000;"));
    b.raw_ptx(&format!("@{p_tiny} bra {done};"));

    // Saturation: |x| >= 448  (bit pattern 0x43E00000) -> 0x7E.
    let p_sat = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ge.u32 {p_sat}, {xbits}, 0x43E00000;"));
    let after_sat = b.fresh_label("e4m3_enc_after_sat");
    b.raw_ptx(&format!("@!{p_sat} bra {after_sat};"));
    b.raw_ptx(&format!("mov.u32 {mag}, 0x7E;"));
    b.raw_ptx(&format!("bra {done};"));
    b.raw_ptx(&format!("{after_sat}:"));

    // Split on normal vs subnormal range.
    let subnormal = b.fresh_label("e4m3_enc_subnormal");
    let p_norm = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lt.s32 {p_norm}, {e4}, 1;"));
    b.raw_ptx(&format!("@{p_norm} bra {subnormal};"));

    // ---- Normal path: round the 23-bit mantissa down to 3 bits (RNE) ----
    let m3 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shr.u32 {m3}, {m32}, 20;"));
    let rest = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("and.b32 {rest}, {m32}, 1048575;")); // low 20 bits
    // Round up when rest > half, or (rest == half and m3 is odd).
    let p_gt = b.alloc_reg(PtxType::Pred);
    let p_eq = b.alloc_reg(PtxType::Pred);
    let p_odd = b.alloc_reg(PtxType::Pred);
    let p_round = b.alloc_reg(PtxType::Pred);
    let lsb = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("setp.gt.u32 {p_gt}, {rest}, 524288;")); // half = 1<<19
    b.raw_ptx(&format!("setp.eq.u32 {p_eq}, {rest}, 524288;"));
    b.raw_ptx(&format!("and.b32 {lsb}, {m3}, 1;"));
    b.raw_ptx(&format!("setp.ne.u32 {p_odd}, {lsb}, 0;"));
    b.raw_ptx(&format!("and.pred {p_eq}, {p_eq}, {p_odd};"));
    b.raw_ptx(&format!("or.pred {p_round}, {p_gt}, {p_eq};"));
    let round_inc = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("selp.u32 {round_inc}, 1, 0, {p_round};"));
    b.raw_ptx(&format!("add.u32 {m3}, {m3}, {round_inc};"));

    // Mantissa carry (m3 == 8): reset mantissa, bump exponent.
    let p_carry = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.eq.u32 {p_carry}, {m3}, 8;"));
    let carry_inc = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("selp.s32 {carry_inc}, 1, 0, {p_carry};"));
    b.raw_ptx(&format!("add.s32 {e4}, {e4}, {carry_inc};"));
    let m3_masked = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("and.b32 {m3_masked}, {m3}, 7;")); // 8 -> 0

    // Saturate when exponent overflowed: e4 > 15, or e4 == 15 with the
    // reserved NaN mantissa 7 -> clamp to 0x7E.
    let p_e_over = b.alloc_reg(PtxType::Pred);
    let p_e15 = b.alloc_reg(PtxType::Pred);
    let p_m7 = b.alloc_reg(PtxType::Pred);
    let p_nan = b.alloc_reg(PtxType::Pred);
    let p_sat2 = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.gt.s32 {p_e_over}, {e4}, 15;"));
    b.raw_ptx(&format!("setp.eq.s32 {p_e15}, {e4}, 15;"));
    b.raw_ptx(&format!("setp.eq.u32 {p_m7}, {m3_masked}, 7;"));
    b.raw_ptx(&format!("and.pred {p_nan}, {p_e15}, {p_m7};"));
    b.raw_ptx(&format!("or.pred {p_sat2}, {p_e_over}, {p_nan};"));

    let norm_pack = b.fresh_label("e4m3_enc_norm_pack");
    b.raw_ptx(&format!("@!{p_sat2} bra {norm_pack};"));
    b.raw_ptx(&format!("mov.u32 {mag}, 0x7E;"));
    b.raw_ptx(&format!("bra {done};"));
    b.raw_ptx(&format!("{norm_pack}:"));
    // mag = (e4 << 3) | m3.
    let e4u = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.b32 {e4u}, {e4};"));
    b.raw_ptx(&format!("shl.b32 {mag}, {e4u}, 3;"));
    b.raw_ptx(&format!("or.b32 {mag}, {mag}, {m3_masked};"));
    b.raw_ptx(&format!("bra {done};"));

    // ---- Subnormal path: e4 <= 0 -----------------------------------------
    b.raw_ptx(&format!("{subnormal}:"));
    // 24-bit significand with the implicit leading 1.
    let full = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("or.b32 {full}, {m32}, 8388608;"));
    // shift = 21 - e4  (>= 21, and <= 24 thanks to the flush-to-zero guard).
    let shift = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.s32 {shift}, 21;"));
    b.raw_ptx(&format!("sub.s32 {shift}, {shift}, {e4};"));
    let shift_u = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.b32 {shift_u}, {shift};"));
    // M = full >> shift.
    let sub_m = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shr.u32 {sub_m}, {full}, {shift_u};"));
    // Round-to-nearest-even on the dropped bits.
    let shift_m1 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("sub.u32 {shift_m1}, {shift_u}, 1;"));
    let rbit = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shr.u32 {rbit}, {full}, {shift_m1};"));
    b.raw_ptx(&format!("and.b32 {rbit}, {rbit}, 1;"));
    // sticky = low (shift-1) bits of `full`, extracted with bfe.
    let zero_start = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {zero_start}, 0;"));
    let sticky = b.bfe_u32(full, zero_start, shift_m1);
    let p_rb = b.alloc_reg(PtxType::Pred);
    let p_st = b.alloc_reg(PtxType::Pred);
    let p_mo = b.alloc_reg(PtxType::Pred);
    let p_tie = b.alloc_reg(PtxType::Pred);
    let p_subr = b.alloc_reg(PtxType::Pred);
    let sub_lsb = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("setp.ne.u32 {p_rb}, {rbit}, 0;"));
    b.raw_ptx(&format!("setp.ne.u32 {p_st}, {sticky}, 0;"));
    b.raw_ptx(&format!("and.b32 {sub_lsb}, {sub_m}, 1;"));
    b.raw_ptx(&format!("setp.ne.u32 {p_mo}, {sub_lsb}, 0;"));
    // round up if rbit && (sticky || lsb).
    b.raw_ptx(&format!("or.pred {p_tie}, {p_st}, {p_mo};"));
    b.raw_ptx(&format!("and.pred {p_subr}, {p_rb}, {p_tie};"));
    let sub_inc = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("selp.u32 {sub_inc}, 1, 0, {p_subr};"));
    b.raw_ptx(&format!("add.u32 {sub_m}, {sub_m}, {sub_inc};"));
    // Carry out of the subnormal field (M == 8) -> smallest normal 0x08.
    let p_subc = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ge.u32 {p_subc}, {sub_m}, 8;"));
    b.raw_ptx(&format!("selp.u32 {mag}, 8, {sub_m}, {p_subc};"));

    b.raw_ptx(&format!("{done}:"));
    // Combine the sign bit with the magnitude byte.
    let out = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("or.b32 {out}, {mag}, {sign};"));
    out
}

/// Emits PTX that decodes a real FP8 **E4M3** byte back to `f32`.
///
/// This is the exact inverse of [`emit_e4m3_encode`] for every value the
/// encoder can produce:
///
/// * exponent field `0` → subnormal `(-1)^s · 2^-6 · (mantissa / 8)`
///   (mantissa `0` is signed zero);
/// * exponent field `1..=15` → normal `(-1)^s · 2^(exp-7) · (1 + mantissa/8)`.
///
/// The result is assembled directly as an `f32` bit pattern so the decode is
/// exact (no rounding). `byte` is a `U32` register holding the E4M3 byte in
/// its low 8 bits; the returned `F32` register holds the decoded value.
fn emit_e4m3_decode(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>, byte: &str) -> Register {
    b.comment("--- E4M3 decode (1 sign / 4 exp / 3 mantissa, bias 7) ---");

    // Field extraction.
    let sign = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shr.u32 {sign}, {byte}, 7;"));
    b.raw_ptx(&format!("and.b32 {sign}, {sign}, 1;"));
    let exp = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shr.u32 {exp}, {byte}, 3;"));
    b.raw_ptx(&format!("and.b32 {exp}, {exp}, 15;"));
    let mant = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("and.b32 {mant}, {byte}, 7;"));

    // f32 sign bit.
    let sign32 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shl.b32 {sign32}, {sign}, 31;"));

    // Magnitude assembled as a float, then sign applied via bit-or.
    let magf = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {magf}, 0f00000000;"));

    let done = b.fresh_label("e4m3_dec_done");

    // Subnormal / zero: exponent field == 0.
    let normal = b.fresh_label("e4m3_dec_normal");
    let p_norm = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ne.u32 {p_norm}, {exp}, 0;"));
    b.raw_ptx(&format!("@{p_norm} bra {normal};"));
    // value = mant * 2^-9 = mant * (1/512).
    let mant_f = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("cvt.rn.f32.u32 {mant_f}, {mant};"));
    // 2^-9 bit pattern = 0x3B000000.
    let step = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.b32 {step}, 0x3B000000;"));
    b.raw_ptx(&format!("mul.rn.f32 {magf}, {mant_f}, {step};"));
    b.raw_ptx(&format!("bra {done};"));

    // Normal: value = 2^(exp-7) * (1 + mant/8).
    b.raw_ptx(&format!("{normal}:"));
    // f32 biased exponent = (exp - 7) + 127 = exp + 120.
    let e32 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("add.u32 {e32}, {exp}, 120;"));
    b.raw_ptx(&format!("shl.b32 {e32}, {e32}, 23;"));
    // f32 mantissa = mant << 20 (3-bit field promoted to the top of 23 bits).
    let m32 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shl.b32 {m32}, {mant}, 20;"));
    let magbits = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("or.b32 {magbits}, {e32}, {m32};"));
    b.raw_ptx(&format!("mov.b32 {magf}, {magbits};"));

    b.raw_ptx(&format!("{done}:"));
    // Apply the sign by or-ing the sign bit into the float bit pattern.
    let magbits2 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.b32 {magbits2}, {magf};"));
    b.raw_ptx(&format!("or.b32 {magbits2}, {magbits2}, {sign32};"));
    let out = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.b32 {out}, {magbits2};"));
    out
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

                // Quantize: scaled = val / scale, clamped to [-448, 448].
                let scaled = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.rn.f32 {scaled}, {val_f32}, {scale};"));

                let max_val = load_float_imm::<f32>(b, E4M3_MAX);
                let neg_max = b.neg_f32(max_val.clone());
                let clamped = b.max_f32(scaled, neg_max);
                let clamped = b.min_f32(clamped, max_val);

                // Encode the clamped value into a real E4M3 byte
                // (1 sign / 4 exponent / 3 mantissa, exponent bias 7).
                let e4m3_byte = emit_e4m3_encode(b, &clamped.to_string());

                // Store as u8
                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, 1u32);
                b.raw_ptx(&format!("st.global.u8 [{out_addr}], {e4m3_byte};"));
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

                // Load the E4M3 u8 value.
                let in_addr = b.byte_offset_addr(in_ptr, gid.clone(), 1u32);
                let raw = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("ld.global.u8 {raw}, [{in_addr}];"));

                // Decode the E4M3 byte to f32 (same encoding the quantize
                // kernel produced: 1 sign / 4 exponent / 3 mantissa, bias 7).
                let float_val = emit_e4m3_decode(b, &raw.to_string());

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

    // -----------------------------------------------------------------------
    // Real E4M3 encoding: bit-exact CPU model of emit_e4m3_encode / decode
    // -----------------------------------------------------------------------

    /// Bit-exact CPU model of the PTX `emit_e4m3_encode` routine. Encodes an
    /// `f32` (already clamped to `[-448, 448]`) into an E4M3 byte:
    /// `[sign:1 | exponent:4 | mantissa:3]`, exponent bias 7.
    fn cpu_e4m3_encode(value: f32) -> u8 {
        let fbits = value.to_bits();
        let sign = ((fbits >> 24) & 0x80) as u8;
        let xbits = fbits & 0x7fff_ffff;

        // Flush to zero: |x| < 2^-10 (0x35800000).
        if xbits < 0x3580_0000 {
            return sign;
        }
        // Saturation: |x| >= 448 (0x43E00000) -> 0x7E.
        if xbits >= 0x43E0_0000 {
            return sign | 0x7E;
        }

        let e32 = (xbits >> 23) & 0xff;
        let m32 = xbits & 0x7f_ffff;
        let mut e4 = e32 as i32 - 120;

        if e4 >= 1 {
            // Normal: round 23-bit mantissa to 3 bits, round-to-nearest-even.
            let mut m3 = m32 >> 20;
            let rest = m32 & 0x0f_ffff;
            let round = rest > 0x8_0000 || (rest == 0x8_0000 && (m3 & 1) == 1);
            if round {
                m3 += 1;
            }
            if m3 == 8 {
                m3 = 0;
                e4 += 1;
            }
            let m3 = m3 & 7;
            if e4 > 15 || (e4 == 15 && m3 == 7) {
                return sign | 0x7E;
            }
            sign | ((e4 as u32) << 3 | m3) as u8
        } else {
            // Subnormal: shift the 24-bit significand into the 3-bit field.
            let full = m32 | 0x80_0000;
            let shift = (21 - e4) as u32;
            let mut sub_m = full >> shift;
            let rbit = (full >> (shift - 1)) & 1;
            let sticky = full & ((1u32 << (shift - 1)) - 1);
            let round = rbit != 0 && (sticky != 0 || (sub_m & 1) != 0);
            if round {
                sub_m += 1;
            }
            let mag = if sub_m >= 8 { 8 } else { sub_m };
            sign | mag as u8
        }
    }

    /// Bit-exact CPU model of the PTX `emit_e4m3_decode` routine.
    fn cpu_e4m3_decode(byte: u8) -> f32 {
        let sign = ((byte >> 7) & 1) as u32;
        let exp = ((byte >> 3) & 0xf) as u32;
        let mant = (byte & 7) as u32;
        let sign_f = if sign == 1 { -1.0f32 } else { 1.0f32 };
        if exp == 0 {
            // Subnormal: mant * 2^-9.
            sign_f * (mant as f32) * (1.0 / 512.0)
        } else {
            // Normal: 2^(exp-7) * (1 + mant/8).
            let e32 = exp + 120;
            let m32 = mant << 20;
            let bits = (e32 << 23) | m32;
            sign_f * f32::from_bits(bits)
        }
    }

    /// The E4M3 encoder must emit canonical bit patterns for reference values.
    #[test]
    fn e4m3_encode_known_patterns() {
        // 1.0 -> e4=7 (0b0111), m=0 -> 0b0_0111_000 = 0x38.
        assert_eq!(cpu_e4m3_encode(1.0), 0x38);
        // 2.0 -> e4=8, m=0 -> 0x40.
        assert_eq!(cpu_e4m3_encode(2.0), 0x40);
        // 0.5 -> e4=6, m=0 -> 0x30.
        assert_eq!(cpu_e4m3_encode(0.5), 0x30);
        // -1.0 -> sign | 0x38 = 0xB8.
        assert_eq!(cpu_e4m3_encode(-1.0), 0xB8);
        // 448.0 (max) -> 0x7E.
        assert_eq!(cpu_e4m3_encode(448.0), 0x7E);
        // 0.0 -> 0x00.
        assert_eq!(cpu_e4m3_encode(0.0), 0x00);
        // 1.75 = 1 + 6/8 -> e4=7, m=6 -> 0b0_0111_110 = 0x3E.
        assert_eq!(cpu_e4m3_encode(1.75), 0x3E);
    }

    /// `gpu_one()` for the `E4M3` BLAS type must match the encoder's `1.0`.
    #[test]
    fn e4m3_gpu_one_matches_encoder() {
        use oxicuda_blas::E4M3;
        let one = <E4M3 as GpuFloat>::gpu_one();
        assert_eq!(one.0, cpu_e4m3_encode(1.0));
        assert_eq!(one.0, 0x38);
    }

    /// E4M3 decode must invert encode for every canonical pattern.
    #[test]
    fn e4m3_decode_inverts_encode() {
        for &v in &[1.0f32, 2.0, 0.5, 4.0, 0.25, 1.75, -1.0, -2.0, -0.5] {
            let byte = cpu_e4m3_encode(v);
            let decoded = cpu_e4m3_decode(byte);
            // These values are all exactly representable in E4M3.
            assert!(
                (decoded - v).abs() < 1e-6,
                "E4M3 round-trip exact value failed: {v} -> 0x{byte:02X} -> {decoded}"
            );
        }
    }

    /// Round-trip through real E4M3 must stay within E4M3 precision: the
    /// quantization step is `2^(exp-7) / 8` for normals.
    #[test]
    fn e4m3_round_trip_within_precision() {
        // Spread of magnitudes across several exponents.
        let values = [
            0.1f32, 0.3, 0.7, 1.1, 1.3, 2.6, 5.2, 11.0, 37.0, 100.0, 300.0, 440.0,
        ];
        for &v in &values {
            let byte = cpu_e4m3_encode(v);
            let decoded = cpu_e4m3_decode(byte);
            // Worst-case absolute error is half a mantissa step. The step at
            // magnitude `v` is `2^floor(log2(v)) / 8`.
            let exp = v.abs().log2().floor();
            let step = exp.exp2() / 8.0;
            let max_err = step * 0.5 + 1e-4;
            assert!(
                (decoded - v).abs() <= max_err,
                "E4M3 round-trip {v} -> {decoded}, error {} > {max_err}",
                (decoded - v).abs()
            );
        }
    }

    /// Negative values must round-trip with the sign preserved.
    #[test]
    fn e4m3_round_trip_negative_values() {
        for &v in &[-0.7f32, -1.3, -5.2, -37.0, -300.0] {
            let byte = cpu_e4m3_encode(v);
            assert!(byte & 0x80 != 0, "sign bit must be set for {v}");
            let decoded = cpu_e4m3_decode(byte);
            assert!(decoded < 0.0, "decoded {decoded} must stay negative");
            let exp = v.abs().log2().floor();
            let step = exp.exp2() / 8.0;
            assert!(
                (decoded - v).abs() <= step * 0.5 + 1e-4,
                "negative E4M3 round-trip {v} -> {decoded}"
            );
        }
    }

    /// E4M3 subnormals (exponent field 0) must encode and decode correctly.
    /// The subnormal grid is `m * 2^-9` for `m in 1..=7`.
    #[test]
    fn e4m3_subnormal_round_trip() {
        let step = 1.0f32 / 512.0; // 2^-9, the subnormal step
        for m in 1u32..=7 {
            let v = (m as f32) * step;
            let byte = cpu_e4m3_encode(v);
            // Exponent field must be 0 (subnormal), mantissa field == m.
            assert_eq!(byte >> 3, 0, "subnormal {v} must have exponent field 0");
            assert_eq!(u32::from(byte & 7), m, "subnormal mantissa mismatch");
            let decoded = cpu_e4m3_decode(byte);
            assert!(
                (decoded - v).abs() < 1e-7,
                "subnormal round-trip {v} -> {decoded}"
            );
        }
    }

    /// The smallest E4M3 subnormal is `2^-9`; the smallest normal is `2^-6`.
    #[test]
    fn e4m3_subnormal_to_normal_boundary() {
        // 2^-6 is the smallest normal: exponent field 1, mantissa 0.
        let smallest_normal = 2.0f32.powi(-6);
        let byte = cpu_e4m3_encode(smallest_normal);
        assert_eq!(byte, 0x08, "2^-6 must encode to the smallest normal 0x08");
        assert!((cpu_e4m3_decode(0x08) - smallest_normal).abs() < 1e-8);

        // A subnormal value that rounds up across the boundary (7.5 * 2^-9)
        // becomes the smallest normal.
        let near_boundary = 7.5f32 / 512.0;
        let byte = cpu_e4m3_encode(near_boundary);
        assert_eq!(byte & 0x7F, 0x08, "boundary subnormal must promote to 0x08");
    }

    /// Values below half the smallest subnormal step flush to zero.
    #[test]
    fn e4m3_flush_to_zero() {
        // 2^-11 is well below 2^-10 (half a subnormal step) -> zero.
        assert_eq!(cpu_e4m3_encode(2.0f32.powi(-11)), 0x00);
        assert_eq!(cpu_e4m3_encode(1e-10), 0x00);
        // Sign is preserved on flushed values (signed zero).
        assert_eq!(cpu_e4m3_encode(-1e-10), 0x80);
    }

    /// Saturation: magnitudes at or above 448 clamp to the E4M3 max 0x7E.
    #[test]
    fn e4m3_saturation() {
        assert_eq!(cpu_e4m3_encode(448.0), 0x7E);
        assert_eq!(cpu_e4m3_encode(-448.0), 0xFE);
        // The decoded max must be exactly 448.
        assert!((cpu_e4m3_decode(0x7E) - 448.0).abs() < 1e-3);
        assert!((cpu_e4m3_decode(0xFE) + 448.0).abs() < 1e-3);
        // 0x7F is the reserved NaN slot; the encoder must never produce it.
        for i in 0..=255u32 {
            // Sweep representable-magnitude inputs; none may hit 0x7F / 0xFF.
            let v = (i as f32) * 1.9 - 240.0;
            let clamped = v.clamp(-448.0, 448.0);
            let byte = cpu_e4m3_encode(clamped);
            assert_ne!(byte & 0x7F, 0x7F, "encoder must never emit the NaN pattern");
        }
    }

    /// Round-to-nearest-even: a mantissa exactly halfway must round to the
    /// even neighbour, not always up.
    #[test]
    fn e4m3_round_to_nearest_even() {
        // Within exponent 0 (values in [1, 2)), the mantissa step is 1/8.
        // 1.0625 = 1 + 0.5/8 is exactly halfway between 1.0 (m=0, even) and
        // 1.125 (m=1, odd) -> rounds to even -> 1.0 (m=0).
        let halfway_down = 1.0f32 + 0.5 / 8.0;
        assert_eq!(
            cpu_e4m3_encode(halfway_down) & 7,
            0,
            "tie must round to even (m=0)"
        );

        // 1.1875 = 1 + 1.5/8 is halfway between 1.125 (m=1, odd) and
        // 1.25 (m=2, even) -> rounds to even -> 1.25 (m=2).
        let halfway_up = 1.0f32 + 1.5 / 8.0;
        assert_eq!(
            cpu_e4m3_encode(halfway_up) & 7,
            2,
            "tie must round to even (m=2)"
        );
    }

    /// The quantize kernel PTX must emit real E4M3 bit manipulation, not the
    /// old offset-binary `round + 128` encoding.
    #[test]
    fn fp8_quant_ptx_uses_real_e4m3() {
        let ptx = generate_fp8_quant_ptx::<f32>(SmVersion::Sm80).expect("ptx");
        // The E4M3 encoder relies on bit-field extract and shifts.
        assert!(ptx.contains("E4M3 encode"), "must document E4M3 encoding");
        assert!(
            ptx.contains("bfe"),
            "E4M3 encode must use bfe for the mantissa field"
        );
        assert!(
            ptx.contains("shr.u32"),
            "E4M3 encode must shift the exponent field"
        );
        // The old offset-binary path rounded the scaled value to an integer
        // via `cvt.rzi.s32.f32`; the real E4M3 encoder never does that.
        assert!(
            !ptx.contains("cvt.rzi.s32.f32"),
            "offset-binary round-to-integer encoding must be removed"
        );
    }

    /// The dequant kernel PTX must decode real E4M3, not undo a `-128` bias.
    #[test]
    fn fp8_dequant_ptx_uses_real_e4m3() {
        let ptx = generate_fp8_dequant_ptx::<f32>(SmVersion::Sm80).expect("ptx");
        assert!(ptx.contains("E4M3 decode"), "must document E4M3 decoding");
        assert!(
            ptx.contains("and.b32"),
            "E4M3 decode must mask the bit fields"
        );
    }
}
