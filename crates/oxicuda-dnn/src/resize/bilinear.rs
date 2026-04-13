//! Bilinear resize.
//!
//! Each output pixel is interpolated from its 4 nearest neighbors in the
//! input using bilinear weights. Supports `align_corners` mode which
//! changes the coordinate mapping formula.

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

/// Block size for bilinear resize kernels.
const BILINEAR_BLOCK: u32 = 256;

/// Resizes a 4-D NCHW tensor using bilinear interpolation.
///
/// When `align_corners` is true, the corner pixels of input and output are
/// exactly aligned. When false, the pixel centers are uniformly spaced (the
/// PyTorch default for `align_corners=False`).
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if batch or channel counts differ.
pub fn resize_bilinear<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
    align_corners: bool,
) -> DnnResult<()> {
    let (in_n, in_c, in_h, in_w) = nchw_dims(input)?;
    let (out_n, out_c, out_h, out_w) = nchw_dims_mut(output)?;

    if in_n != out_n || in_c != out_c {
        return Err(DnnError::InvalidDimension(format!(
            "resize_bilinear: batch/channel mismatch: in=({in_n},{in_c}), out=({out_n},{out_c})"
        )));
    }

    let total = output.numel() as u32;
    if total == 0 {
        return Ok(());
    }

    let ptx = generate_bilinear_ptx::<T>(handle.sm_version(), align_corners)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let suffix = if align_corners { "ac" } else { "noac" };
    let name = format!("dnn_resize_bilinear_{suffix}_{}", T::NAME);
    let kernel = Kernel::from_module(module, &name)?;

    let grid = grid_size_for(total, BILINEAR_BLOCK);
    let params = LaunchParams::new(grid, BILINEAR_BLOCK);

    let args = (
        input.ptr, output.ptr, in_n, in_c, in_h, in_w, out_h, out_w, total,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("resize_bilinear: {e}")))?;

    Ok(())
}

/// Generates PTX for bilinear interpolation.
///
/// For each output pixel at (oh, ow):
/// - Compute source floating-point coordinates (src_h, src_w)
/// - Determine the 4 neighbors: (h0,w0), (h0,w1), (h1,w0), (h1,w1)
/// - Compute bilinear weights and interpolate
fn generate_bilinear_ptx<T: GpuFloat>(sm: SmVersion, align_corners: bool) -> DnnResult<String> {
    let suffix = if align_corners { "ac" } else { "noac" };
    let name = format!("dnn_resize_bilinear_{suffix}_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(BILINEAR_BLOCK)
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

                // Decompose gid -> (n, c, oh, ow)
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

                // Compute source coordinates as floats
                // All bilinear math is done in f32 regardless of T
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

                // Compute scale and src coordinate
                let (src_h, src_w) = if align_corners {
                    // src_h = oh * (in_h - 1) / (out_h - 1)
                    let one_f = b.alloc_reg(PtxType::F32);
                    let bits_1 = 1.0f32.to_bits();
                    b.raw_ptx(&format!("mov.b32 {one_f}, 0F{bits_1:08X};"));

                    let ih_m1 = b.sub_f32(in_h_f.clone(), one_f.clone());
                    let oh_m1 = b.sub_f32(out_h_f, one_f.clone());
                    let iw_m1 = b.sub_f32(in_w_f.clone(), one_f.clone());
                    let ow_m1 = b.sub_f32(out_w_f, one_f);

                    // Avoid division by zero when output is 1
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
                    // src_h = (oh + 0.5) * in_h / out_h - 0.5
                    let half = b.alloc_reg(PtxType::F32);
                    let bits_05 = 0.5f32.to_bits();
                    b.raw_ptx(&format!("mov.b32 {half}, 0F{bits_05:08X};"));

                    let oh_c = b.add_f32(oh_f, half.clone());
                    let ow_c = b.add_f32(ow_f, half.clone());

                    let h_scaled = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {h_scaled}, {oh_c}, {in_h_f};"));
                    let h_div = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("div.rn.f32 {h_div}, {h_scaled}, {out_h_f};"));
                    let h = b.sub_f32(h_div, half.clone());

                    let w_scaled = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {w_scaled}, {ow_c}, {in_w_f};"));
                    let w_div = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("div.rn.f32 {w_div}, {w_scaled}, {out_w_f};"));
                    let w = b.sub_f32(w_div, half);
                    (h, w)
                };

                // Clamp and compute floor
                let zero_f = b.alloc_reg(PtxType::F32);
                let bits_0 = 0.0f32.to_bits();
                b.raw_ptx(&format!("mov.b32 {zero_f}, 0F{bits_0:08X};"));
                let src_h_clamped = b.max_f32(src_h, zero_f.clone());
                let src_w_clamped = b.max_f32(src_w, zero_f);

                // h0 = floor(src_h), h1 = h0 + 1 (clamped to in_h - 1)
                let h0_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rmi.f32.f32 {h0_f}, {src_h_clamped};"));
                let w0_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rmi.f32.f32 {w0_f}, {src_w_clamped};"));

                let h0 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("cvt.rzi.u32.f32 {h0}, {h0_f};"));
                let w0 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("cvt.rzi.u32.f32 {w0}, {w0_f};"));

                // Fractional parts
                let fh = b.sub_f32(src_h_clamped.clone(), h0_f.clone());
                let fw = b.sub_f32(src_w_clamped, w0_f);

                // h1 = min(h0 + 1, in_h - 1), clamp h0 too
                let h0p1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {h0p1}, {h0}, 1;"));
                let in_h_m1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("sub.u32 {in_h_m1}, {in_h}, 1;"));
                let h1 = b.min_u32(h0p1, in_h_m1.clone());
                let h0_c = b.min_u32(h0, in_h_m1);

                let w0p1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {w0p1}, {w0}, 1;"));
                let in_w_m1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("sub.u32 {in_w_m1}, {in_w}, 1;"));
                let w1 = b.min_u32(w0p1, in_w_m1.clone());
                let w0_c = b.min_u32(w0, in_w_m1);

                // Base offset for (n, c) plane
                let in_hw = b.mul_lo_u32(in_h, in_w.clone());
                let c_hw = b.mul_lo_u32(c_idx, in_hw.clone());
                let chw = b.mul_lo_u32(channels, in_hw);
                let n_off = b.mul_lo_u32(n_idx, chw);
                let base = b.add_u32(n_off, c_hw);

                let in_ptr = b.load_param_u64("in_ptr");

                // Load 4 neighbors: (h0,w0), (h0,w1), (h1,w0), (h1,w1)
                let r00 = b.mul_lo_u32(h0_c.clone(), in_w.clone());
                let hw00 = b.add_u32(r00, w0_c.clone());
                let idx00 = b.add_u32(base.clone(), hw00);
                let a00 = b.byte_offset_addr(in_ptr.clone(), idx00, T::size_u32());
                let v00 = load_global_float::<T>(b, a00);

                let r01 = b.mul_lo_u32(h0_c, in_w.clone());
                let hw01 = b.add_u32(r01, w1.clone());
                let idx01 = b.add_u32(base.clone(), hw01);
                let a01 = b.byte_offset_addr(in_ptr.clone(), idx01, T::size_u32());
                let v01 = load_global_float::<T>(b, a01);

                let r10 = b.mul_lo_u32(h1.clone(), in_w.clone());
                let hw10 = b.add_u32(r10, w0_c);
                let idx10 = b.add_u32(base.clone(), hw10);
                let a10 = b.byte_offset_addr(in_ptr.clone(), idx10, T::size_u32());
                let v10 = load_global_float::<T>(b, a10);

                let r11 = b.mul_lo_u32(h1, in_w);
                let hw11 = b.add_u32(r11, w1);
                let idx11 = b.add_u32(base, hw11);
                let a11 = b.byte_offset_addr(in_ptr, idx11, T::size_u32());
                let v11 = load_global_float::<T>(b, a11);

                // Bilinear interpolation (done in native T precision)
                // result = (1-fh)*(1-fw)*v00 + (1-fh)*fw*v01 + fh*(1-fw)*v10 + fh*fw*v11
                // Convert fh/fw to T precision if needed
                let (fh_t, fw_t) = if T::PTX_TYPE == PtxType::F64 {
                    let fh64 = b.cvt_f32_to_f64(fh);
                    let fw64 = b.cvt_f32_to_f64(fw);
                    (fh64, fw64)
                } else {
                    (fh, fw)
                };

                let one_t = load_float_imm::<T>(b, 1.0);
                let one_mfh = if T::PTX_TYPE == PtxType::F32 {
                    b.sub_f32(one_t.clone(), fh_t.clone())
                } else {
                    b.sub_f64(one_t.clone(), fh_t.clone())
                };
                let one_mfw = if T::PTX_TYPE == PtxType::F32 {
                    b.sub_f32(one_t, fw_t.clone())
                } else {
                    b.sub_f64(one_t, fw_t.clone())
                };

                // w00 = (1-fh) * (1-fw)
                let w00 = mul_float::<T>(b, one_mfh.clone(), one_mfw.clone());
                let w01 = mul_float::<T>(b, one_mfh, fw_t.clone());
                let w10 = mul_float::<T>(b, fh_t.clone(), one_mfw);
                let w11 = mul_float::<T>(b, fh_t, fw_t);

                // Weighted sum
                let t0 = mul_float::<T>(b, w00, v00);
                let t1 = fma_float::<T>(b, w01, v01, t0);
                let t2 = fma_float::<T>(b, w10, v10, t1);
                let result = fma_float::<T>(b, w11, v11, t2);

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, T::size_u32());
                store_global_float::<T>(b, out_addr, result);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("resize_bilinear: {e}")))?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_ptx_f32_align() {
        let ptx = generate_bilinear_ptx::<f32>(SmVersion::Sm80, true);
        assert!(ptx.is_ok());
        let s = ptx.expect("should gen");
        assert!(s.contains("dnn_resize_bilinear_ac_f32"));
    }

    #[test]
    fn bilinear_ptx_f32_no_align() {
        let ptx = generate_bilinear_ptx::<f32>(SmVersion::Sm80, false);
        assert!(ptx.is_ok());
    }

    #[test]
    fn bilinear_ptx_f64() {
        let ptx = generate_bilinear_ptx::<f64>(SmVersion::Sm80, false);
        assert!(ptx.is_ok());
    }
}
