//! Bicubic resize.
//!
//! Each output pixel is interpolated from a 4x4 neighborhood of input pixels
//! using cubic weights. This provides higher-quality upsampling than bilinear
//! at the cost of more memory accesses and arithmetic per output element.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::ptx_helpers::*;
use crate::tensor_util::{nchw_dims, nchw_dims_mut};
use crate::types::{TensorDesc, TensorDescMut};

/// Block size for bicubic resize kernels.
const BICUBIC_BLOCK: u32 = 256;

/// Resizes a 4-D NCHW tensor using bicubic interpolation.
///
/// Uses the Keys cubic convolution kernel (a = -0.75) which provides
/// C1 continuity. Each output pixel reads from a 4x4 input neighborhood.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if batch or channel counts differ.
pub fn resize_bicubic<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
    align_corners: bool,
) -> DnnResult<()> {
    let (in_n, in_c, in_h, in_w) = nchw_dims(input)?;
    let (out_n, out_c, out_h, out_w) = nchw_dims_mut(output)?;

    if in_n != out_n || in_c != out_c {
        return Err(DnnError::InvalidDimension(format!(
            "resize_bicubic: batch/channel mismatch: in=({in_n},{in_c}), out=({out_n},{out_c})"
        )));
    }

    let total = output.numel() as u32;
    if total == 0 {
        return Ok(());
    }

    let ptx = generate_bicubic_ptx::<T>(handle.sm_version(), align_corners)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let suffix = if align_corners { "ac" } else { "noac" };
    let name = format!("dnn_resize_bicubic_{suffix}_{}", T::NAME);
    let kernel = Kernel::from_module(module, &name)?;

    let grid = grid_size_for(total, BICUBIC_BLOCK);
    let params = LaunchParams::new(grid, BICUBIC_BLOCK);

    let args = (
        input.ptr, output.ptr, in_n, in_c, in_h, in_w, out_h, out_w, total,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("resize_bicubic: {e}")))?;

    Ok(())
}

/// Generates PTX for bicubic interpolation.
///
/// The kernel computes source coordinates using the same formulas as bilinear
/// (with or without align_corners), then samples a 4x4 window using Keys
/// cubic weights (a = -0.75).
///
/// Cubic weight function:
/// ```text
/// w(x) = (a+2)|x|^3 - (a+3)|x|^2 + 1       for |x| <= 1
/// w(x) = a|x|^3 - 5a|x|^2 + 8a|x| - 4a     for 1 < |x| < 2
/// w(x) = 0                                    otherwise
/// ```
fn generate_bicubic_ptx<T: GpuFloat>(sm: SmVersion, align_corners: bool) -> DnnResult<String> {
    let suffix = if align_corners { "ac" } else { "noac" };
    let name = format!("dnn_resize_bicubic_{suffix}_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(BICUBIC_BLOCK)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("batch", PtxType::U32)
        .param("channels", PtxType::U32)
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("out_h", PtxType::U32)
        .param("out_w", PtxType::U32)
        .param("total", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let total = b.load_param_u32("total");

            b.if_lt_u32(gid.clone(), total, move |b| {
                let out_w = b.load_param_u32("out_w");
                let out_h = b.load_param_u32("out_h");
                let channels = b.load_param_u32("channels");
                let in_h = b.load_param_u32("in_h");
                let in_w = b.load_param_u32("in_w");

                // Decompose gid
                let ow_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {ow_idx}, {gid}, {out_w};"));
                let tmp1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {tmp1}, {gid}, {out_w};"));
                let oh_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {oh_idx}, {tmp1}, {out_h};"));
                let tmp2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {tmp2}, {tmp1}, {out_h};"));
                let c_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {c_idx}, {tmp2}, {channels};"));
                let n_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {n_idx}, {tmp2}, {channels};"));

                // Convert to float for coordinate computation (all in f32)
                let oh_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {oh_f}, {oh_idx};"));
                let ow_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {ow_f}, {ow_idx};"));
                let in_h_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {in_h_f}, {in_h};"));
                let in_w_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {in_w_f}, {in_w};"));
                let out_h_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {out_h_f}, {out_h};"));
                let out_w_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {out_w_f}, {out_w};"));

                let one_f = b.alloc_reg(PtxType::F32);
                let bits1 = 1.0f32.to_bits();
                b.raw_ptx(&format!("mov.b32 {one_f}, 0F{bits1:08X};"));
                let half_f = b.alloc_reg(PtxType::F32);
                let bits05 = 0.5f32.to_bits();
                b.raw_ptx(&format!("mov.b32 {half_f}, 0F{bits05:08X};"));

                let (src_h, src_w) = if align_corners {
                    let ih_m1 = b.sub_f32(in_h_f, one_f.clone());
                    let oh_m1 = b.sub_f32(out_h_f, one_f.clone());
                    let iw_m1 = b.sub_f32(in_w_f, one_f.clone());
                    let ow_m1 = b.sub_f32(out_w_f, one_f.clone());
                    let sh = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("div.rn.f32 {sh}, {ih_m1}, {oh_m1};"));
                    let sw = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("div.rn.f32 {sw}, {iw_m1}, {ow_m1};"));
                    let h = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {h}, {oh_f}, {sh};"));
                    let w = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {w}, {ow_f}, {sw};"));
                    (h, w)
                } else {
                    let oh_c = b.add_f32(oh_f, half_f.clone());
                    let ow_c = b.add_f32(ow_f, half_f.clone());
                    let hs = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {hs}, {oh_c}, {in_h_f};"));
                    let hd = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("div.rn.f32 {hd}, {hs}, {out_h_f};"));
                    let h = b.sub_f32(hd, half_f.clone());
                    let ws = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {ws}, {ow_c}, {in_w_f};"));
                    let wd = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("div.rn.f32 {wd}, {ws}, {out_w_f};"));
                    let w = b.sub_f32(wd, half_f);
                    (h, w)
                };

                // Floor to get the center pixel
                let h_floor = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rmi.f32.f32 {h_floor}, {src_h};"));
                let w_floor = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rmi.f32.f32 {w_floor}, {src_w};"));

                let frac_h = b.sub_f32(src_h, h_floor.clone());
                let frac_w = b.sub_f32(src_w, w_floor.clone());

                let h_center_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("cvt.rzi.s32.f32 {h_center_s32}, {h_floor};"));
                let w_center_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("cvt.rzi.s32.f32 {w_center_s32}, {w_floor};"));

                let in_h_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {in_h_s32}, {in_h};"));
                let in_w_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {in_w_s32}, {in_w};"));

                // Base offset
                let in_hw = b.mul_lo_u32(in_h, in_w.clone());
                let c_hw = b.mul_lo_u32(c_idx, in_hw.clone());
                let chw = b.mul_lo_u32(channels, in_hw);
                let n_off = b.mul_lo_u32(n_idx, chw);
                let base = b.add_u32(n_off, c_hw);

                let in_ptr = b.load_param_u64("in_ptr");

                // Accumulate over 4x4 window using cubic weights
                // We compute 4 h-weights and 4 w-weights, then do the
                // double-weighted sum. For simplicity, we use a loop
                // approach: iterate dy in -1..3, dx in -1..3
                let result = load_float_imm::<T>(b, 0.0);

                let dy = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {dy}, -1;"));
                let dy_end = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {dy_end}, 3;"));

                let loop_dy = b.fresh_label("bic_dy");
                let end_dy = b.fresh_label("bic_dy_end");
                b.label(&loop_dy);
                let p_dy = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.s32 {p_dy}, {dy}, {dy_end};"));
                b.branch_if(p_dy, &end_dy);

                // Compute h weight: cubic_weight(frac_h - dy)
                // t = |frac_h - dy|
                let dy_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.s32 {dy_f}, {dy};"));
                let th = b.sub_f32(frac_h.clone(), dy_f);
                let th_abs = b.abs_f32(th);

                // cubic_weight(t): using a = -0.75
                // if t <= 1: (1.25)*t^3 - (2.25)*t^2 + 1
                // if 1 < t < 2: -0.75*t^3 + 3.75*t^2 - 6*t + 3
                // else: 0
                let t2h = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {t2h}, {th_abs}, {th_abs};"));
                let t3h = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {t3h}, {t2h}, {th_abs};"));

                // Branch: t <= 1 vs 1 < t < 2 vs t >= 2
                let c125 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {c125}, 0F{:08X};", 1.25f32.to_bits()));
                let c225 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {c225}, 0F{:08X};", 2.25f32.to_bits()));
                let c075 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {c075}, 0F{:08X};", 0.75f32.to_bits()));
                let c375 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {c375}, 0F{:08X};", 3.75f32.to_bits()));
                let c6 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {c6}, 0F{:08X};", 6.0f32.to_bits()));
                let c3 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {c3}, 0F{:08X};", 3.0f32.to_bits()));
                let c2 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {c2}, 0F{:08X};", 2.0f32.to_bits()));

                // w_near = 1.25*t^3 - 2.25*t^2 + 1
                let zero_acc = load_float_imm::<f32>(b, 0.0);
                let wn_h = b.fma_f32(c125.clone(), t3h.clone(), zero_acc);
                let neg_c225 = b.neg_f32(c225.clone());
                let wn_h = b.fma_f32(neg_c225.clone(), t2h.clone(), wn_h);
                let wn_h = b.add_f32(wn_h, one_f.clone());

                // w_far = -0.75*t^3 + 3.75*t^2 - 6*t + 3
                let neg_c075 = b.neg_f32(c075.clone());
                let zero_acc2 = load_float_imm::<f32>(b, 0.0);
                let wf_h = b.fma_f32(neg_c075.clone(), t3h.clone(), zero_acc2);
                let wf_h = b.fma_f32(c375.clone(), t2h.clone(), wf_h);
                let neg_c6 = b.neg_f32(c6.clone());
                let wf_h = b.fma_f32(neg_c6.clone(), th_abs.clone(), wf_h);
                let wf_h = b.add_f32(wf_h, c3.clone());

                // Select based on |t|
                let p_le1_h = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.le.f32 {p_le1_h}, {th_abs}, {one_f};"));
                let p_lt2_h = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lt.f32 {p_lt2_h}, {th_abs}, {c2};"));
                let zero_f32 = load_float_imm::<f32>(b, 0.0);
                let w_h_far = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!(
                    "selp.f32 {w_h_far}, {wf_h}, {zero_f32}, {p_lt2_h};"
                ));
                let w_h = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("selp.f32 {w_h}, {wn_h}, {w_h_far}, {p_le1_h};"));

                // Inner loop over dx
                let dx = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {dx}, -1;"));
                let dx_end = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {dx_end}, 3;"));

                let loop_dx = b.fresh_label("bic_dx");
                let end_dx = b.fresh_label("bic_dx_end");
                b.label(&loop_dx);
                let p_dx = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.s32 {p_dx}, {dx}, {dx_end};"));
                b.branch_if(p_dx, &end_dx);

                // Compute w weight
                let dx_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.s32 {dx_f}, {dx};"));
                let tw = b.sub_f32(frac_w.clone(), dx_f);
                let tw_abs = b.abs_f32(tw);

                let t2w = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {t2w}, {tw_abs}, {tw_abs};"));
                let t3w = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {t3w}, {t2w}, {tw_abs};"));

                let zero_acc3 = load_float_imm::<f32>(b, 0.0);
                let wn_w = b.fma_f32(c125.clone(), t3w.clone(), zero_acc3);
                let wn_w = b.fma_f32(neg_c225.clone(), t2w.clone(), wn_w);
                let wn_w = b.add_f32(wn_w, one_f.clone());

                let zero_acc4 = load_float_imm::<f32>(b, 0.0);
                let wf_w = b.fma_f32(neg_c075.clone(), t3w, zero_acc4);
                let wf_w = b.fma_f32(c375.clone(), t2w, wf_w);
                let wf_w = b.fma_f32(neg_c6.clone(), tw_abs.clone(), wf_w);
                let wf_w = b.add_f32(wf_w, c3.clone());

                let p_le1_w = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.le.f32 {p_le1_w}, {tw_abs}, {one_f};"));
                let p_lt2_w = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lt.f32 {p_lt2_w}, {tw_abs}, {c2};"));
                let zero_f32b = load_float_imm::<f32>(b, 0.0);
                let w_w_far = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!(
                    "selp.f32 {w_w_far}, {wf_w}, {zero_f32b}, {p_lt2_w};"
                ));
                let w_w = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("selp.f32 {w_w}, {wn_w}, {w_w_far}, {p_le1_w};"));

                // Combined weight
                let weight_f32 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {weight_f32}, {w_h}, {w_w};"));

                // Source pixel: clamp(h_center + dy, 0, in_h-1), clamp(w_center + dx, 0, in_w-1)
                let sy = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("add.s32 {sy}, {h_center_s32}, {dy};"));
                let sx = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("add.s32 {sx}, {w_center_s32}, {dx};"));

                // Clamp to [0, in_dim - 1]
                let zero_s = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {zero_s}, 0;"));
                let ih_m1 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("sub.s32 {ih_m1}, {in_h_s32}, 1;"));
                let iw_m1 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("sub.s32 {iw_m1}, {in_w_s32}, 1;"));

                let sy_c = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("max.s32 {sy_c}, {sy}, {zero_s};"));
                let sy_cc = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("min.s32 {sy_cc}, {sy_c}, {ih_m1};"));
                let sx_c = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("max.s32 {sx_c}, {sx}, {zero_s};"));
                let sx_cc = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("min.s32 {sx_cc}, {sx_c}, {iw_m1};"));

                let sy_u = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {sy_u}, {sy_cc};"));
                let sx_u = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {sx_u}, {sx_cc};"));

                let row = b.mul_lo_u32(sy_u, in_w.clone());
                let hw = b.add_u32(row, sx_u);
                let idx = b.add_u32(base.clone(), hw);
                let addr = b.byte_offset_addr(in_ptr.clone(), idx, T::size_u32());
                let val = load_global_float::<T>(b, addr);

                // Convert weight to T precision and accumulate
                let weight_t = if T::PTX_TYPE == PtxType::F64 {
                    b.cvt_f32_to_f64(weight_f32)
                } else {
                    weight_f32
                };

                let contrib = mul_float::<T>(b, weight_t, val);
                let new_result = add_float::<T>(b, result.clone(), contrib);
                b.raw_ptx(&format!(
                    "mov.{ptx} {result}, {new_result};",
                    ptx = crate::ptx_helpers::ptx_type_name::<T>()
                ));

                b.raw_ptx(&format!("add.s32 {dx}, {dx}, 1;"));
                b.branch(&loop_dx);
                b.label(&end_dx);

                b.raw_ptx(&format!("add.s32 {dy}, {dy}, 1;"));
                b.branch(&loop_dy);
                b.label(&end_dy);

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, T::size_u32());
                store_global_float::<T>(b, out_addr, result);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("resize_bicubic: {e}")))?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_ptx_f32_align() {
        let ptx = generate_bicubic_ptx::<f32>(SmVersion::Sm80, true);
        assert!(ptx.is_ok());
        let s = ptx.expect("should gen");
        assert!(s.contains("dnn_resize_bicubic_ac_f32"));
    }

    #[test]
    fn bicubic_ptx_f32_no_align() {
        let ptx = generate_bicubic_ptx::<f32>(SmVersion::Sm80, false);
        assert!(ptx.is_ok());
    }

    #[test]
    fn bicubic_ptx_f64() {
        let ptx = generate_bicubic_ptx::<f64>(SmVersion::Sm80, true);
        assert!(ptx.is_ok());
    }
}
