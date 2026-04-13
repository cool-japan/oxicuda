//! Average pooling 2D operation.
//!
//! Each thread computes one output element as the mean of elements within
//! the kernel window. The `count_include_pad` flag controls whether
//! zero-padded elements contribute to the denominator.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::ptx_helpers::*;
use crate::tensor_util::{nchw_dims, nchw_dims_mut};
use crate::types::{TensorDesc, TensorDescMut, pool_output_size};

/// Block size for average pooling kernels.
const AVG_POOL_BLOCK: u32 = 256;

/// Performs 2D average pooling.
///
/// Each output element is the arithmetic mean of the elements within the
/// corresponding kernel window. When `count_include_pad` is true, padded
/// zeros are included in the element count (denominator); when false,
/// only valid (non-padded) elements contribute.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if dimensions are inconsistent.
pub fn avg_pool2d<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
    kernel_size: (u32, u32),
    stride: (u32, u32),
    padding: (u32, u32),
    count_include_pad: bool,
) -> DnnResult<()> {
    let (in_n, in_c, in_h, in_w) = nchw_dims(input)?;
    let (out_n, out_c, out_h, out_w) = nchw_dims_mut(output)?;

    let expected_oh =
        pool_output_size(in_h, kernel_size.0, stride.0, padding.0).ok_or_else(|| {
            DnnError::InvalidDimension(format!(
                "avg_pool2d: invalid output height for h={in_h}, kh={}, sh={}, ph={}",
                kernel_size.0, stride.0, padding.0
            ))
        })?;
    let expected_ow =
        pool_output_size(in_w, kernel_size.1, stride.1, padding.1).ok_or_else(|| {
            DnnError::InvalidDimension(format!(
                "avg_pool2d: invalid output width for w={in_w}, kw={}, sw={}, pw={}",
                kernel_size.1, stride.1, padding.1
            ))
        })?;

    if out_n != in_n || out_c != in_c || out_h != expected_oh || out_w != expected_ow {
        return Err(DnnError::InvalidDimension(format!(
            "avg_pool2d: output ({out_n},{out_c},{out_h},{out_w}) != expected ({in_n},{in_c},{expected_oh},{expected_ow})"
        )));
    }

    let total_output = output.numel() as u32;
    if total_output == 0 {
        return Ok(());
    }

    let ptx = generate_avg_pool2d_ptx::<T>(handle.sm_version(), count_include_pad)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let name = avg_pool2d_kernel_name::<T>(count_include_pad);
    let kernel = Kernel::from_module(module, &name)?;

    let grid = grid_size_for(total_output, AVG_POOL_BLOCK);
    let params = LaunchParams::new(grid, AVG_POOL_BLOCK);

    let args = (
        input.ptr,
        output.ptr,
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_size.0,
        kernel_size.1,
        stride.0,
        stride.1,
        padding.0,
        padding.1,
        total_output,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("avg_pool2d: {e}")))?;

    Ok(())
}

fn avg_pool2d_kernel_name<T: GpuFloat>(count_include_pad: bool) -> String {
    let suffix = if count_include_pad { "cip" } else { "nocip" };
    format!("dnn_avg_pool2d_{suffix}_{}", T::NAME)
}

/// Generates PTX for 2D average pooling.
fn generate_avg_pool2d_ptx<T: GpuFloat>(
    sm: SmVersion,
    count_include_pad: bool,
) -> DnnResult<String> {
    let name = avg_pool2d_kernel_name::<T>(count_include_pad);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(AVG_POOL_BLOCK)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("batch", PtxType::U32)
        .param("channels", PtxType::U32)
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("out_h", PtxType::U32)
        .param("out_w", PtxType::U32)
        .param("kh", PtxType::U32)
        .param("kw", PtxType::U32)
        .param("sh", PtxType::U32)
        .param("sw", PtxType::U32)
        .param("ph", PtxType::U32)
        .param("pw", PtxType::U32)
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

                // Decompose gid -> (n, c, oh, ow) for NCHW
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

                let sh = b.load_param_u32("sh");
                let sw = b.load_param_u32("sw");
                let ph = b.load_param_u32("ph");
                let pw = b.load_param_u32("pw");
                let kh = b.load_param_u32("kh");
                let kw = b.load_param_u32("kw");

                let h_start_raw = b.mul_lo_u32(oh_idx, sh);
                let w_start_raw = b.mul_lo_u32(ow_idx, sw);
                let h_start = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("sub.s32 {h_start}, {h_start_raw}, {ph};"));
                let w_start = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("sub.s32 {w_start}, {w_start_raw}, {pw};"));

                // Base offset for this (n, c) channel plane
                let in_hw_val = b.mul_lo_u32(in_h.clone(), in_w.clone());
                let c_hw = b.mul_lo_u32(c_idx, in_hw_val.clone());
                let chw = b.mul_lo_u32(channels, in_hw_val);
                let n_offset = b.mul_lo_u32(n_idx, chw);
                let base_offset = b.add_u32(n_offset, c_hw);

                let in_ptr = b.load_param_u64("in_ptr");

                let sum = load_float_imm::<T>(b, 0.0);
                let count = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {count}, 0;"));

                let h_end = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("add.s32 {h_end}, {h_start}, {kh};"));
                let w_end = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("add.s32 {w_end}, {w_start}, {kw};"));
                let in_h_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {in_h_s32}, {in_h};"));
                let in_w_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {in_w_s32}, {in_w};"));

                let ih = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {ih}, {h_start};"));

                let loop_h = b.fresh_label("avg_h");
                let end_h = b.fresh_label("avg_h_end");
                b.label(&loop_h);
                let ph_cmp = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.s32 {ph_cmp}, {ih}, {h_end};"));
                b.branch_if(ph_cmp, &end_h);

                let iw = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {iw}, {w_start};"));

                let loop_w = b.fresh_label("avg_w");
                let end_w = b.fresh_label("avg_w_end");
                b.label(&loop_w);
                let pw_cmp = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.s32 {pw_cmp}, {iw}, {w_end};"));
                b.branch_if(pw_cmp, &end_w);

                if count_include_pad {
                    // Always increment count
                    b.raw_ptx(&format!("add.u32 {count}, {count}, 1;"));
                }

                // Check bounds
                let h_ok = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.and.s32 {h_ok}, {ih}, 0, {{true}};"));
                let h_ok2 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!(
                    "setp.lt.and.s32 {h_ok2}, {ih}, {in_h_s32}, {h_ok};"
                ));
                let w_ok = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.and.s32 {w_ok}, {iw}, 0, {{true}};"));
                let w_ok2 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!(
                    "setp.lt.and.s32 {w_ok2}, {iw}, {in_w_s32}, {w_ok};"
                ));
                let hw_ok = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("and.pred {hw_ok}, {h_ok2}, {w_ok2};"));

                let skip = b.fresh_label("avg_skip");
                let inv = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {inv}, {hw_ok};"));
                b.branch_if(inv, &skip);

                // Load and accumulate
                let ih_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {ih_u32}, {ih};"));
                let iw_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {iw_u32}, {iw};"));
                let row_off = b.mul_lo_u32(ih_u32, in_w.clone());
                let hw_off = b.add_u32(row_off, iw_u32);
                let elem_idx = b.add_u32(base_offset.clone(), hw_off);
                let addr = b.byte_offset_addr(in_ptr.clone(), elem_idx, T::size_u32());
                let val = load_global_float::<T>(b, addr);
                let new_sum = add_float::<T>(b, sum.clone(), val);
                b.raw_ptx(&format!(
                    "mov.{ptx} {sum}, {new_sum};",
                    ptx = crate::ptx_helpers::ptx_type_name::<T>()
                ));

                if !count_include_pad {
                    b.raw_ptx(&format!("add.u32 {count}, {count}, 1;"));
                }

                b.label(&skip);
                b.raw_ptx(&format!("add.s32 {iw}, {iw}, 1;"));
                b.branch(&loop_w);
                b.label(&end_w);

                b.raw_ptx(&format!("add.s32 {ih}, {ih}, 1;"));
                b.branch(&loop_h);
                b.label(&end_h);

                // Divide sum by count
                let count_f = cvt_u32_to_float::<T>(b, count);
                let result = div_float::<T>(b, sum, count_f);

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, T::size_u32());
                store_global_float::<T>(b, out_addr, result);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("avg_pool2d: {e}")))?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avg_pool2d_ptx_f32_cip() {
        let ptx = generate_avg_pool2d_ptx::<f32>(SmVersion::Sm80, true);
        assert!(ptx.is_ok());
        let s = ptx.expect("should gen");
        assert!(s.contains("dnn_avg_pool2d_cip_f32"));
    }

    #[test]
    fn avg_pool2d_ptx_f32_nocip() {
        let ptx = generate_avg_pool2d_ptx::<f32>(SmVersion::Sm80, false);
        assert!(ptx.is_ok());
    }

    #[test]
    fn avg_pool2d_ptx_f64() {
        let ptx = generate_avg_pool2d_ptx::<f64>(SmVersion::Sm80, true);
        assert!(ptx.is_ok());
    }
}
