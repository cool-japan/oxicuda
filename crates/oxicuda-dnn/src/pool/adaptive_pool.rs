//! Adaptive pooling operations.
//!
//! Adaptive pooling automatically computes the kernel size and stride
//! needed to produce the desired output spatial dimensions. For
//! `output_size = (1, 1)`, the operation degenerates to global pooling.

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

/// Block size for adaptive pooling kernels.
const ADAPTIVE_BLOCK: u32 = 256;

/// Performs adaptive average pooling to the target `output_size`.
///
/// For each output element, the corresponding input window is computed as:
/// ```text
/// h_start = floor(oh * in_h / out_h)
/// h_end   = ceil((oh + 1) * in_h / out_h)
/// ```
/// This ensures all input elements are covered exactly once.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if the output tensor does not
/// have shape `(N, C, output_size.0, output_size.1)` matching the input
/// batch and channels.
pub fn adaptive_avg_pool2d<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
    output_size: (u32, u32),
) -> DnnResult<()> {
    let (in_n, in_c, in_h, in_w) = nchw_dims(input)?;
    let (out_n, out_c, out_h, out_w) = nchw_dims_mut(output)?;

    if out_n != in_n || out_c != in_c || out_h != output_size.0 || out_w != output_size.1 {
        return Err(DnnError::InvalidDimension(format!(
            "adaptive_avg_pool2d: output ({out_n},{out_c},{out_h},{out_w}) != expected ({in_n},{in_c},{},{})",
            output_size.0, output_size.1
        )));
    }
    if output_size.0 == 0 || output_size.1 == 0 {
        return Err(DnnError::InvalidDimension(
            "adaptive_avg_pool2d: output_size must be non-zero".into(),
        ));
    }

    let total = output.numel() as u32;
    if total == 0 {
        return Ok(());
    }

    let ptx = generate_adaptive_avg_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let name = format!("dnn_adaptive_avg_pool2d_{}", T::NAME);
    let kernel = Kernel::from_module(module, &name)?;

    let grid = grid_size_for(total, ADAPTIVE_BLOCK);
    let params = LaunchParams::new(grid, ADAPTIVE_BLOCK);

    let args = (
        input.ptr, output.ptr, in_n, in_c, in_h, in_w, out_h, out_w, total,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("adaptive_avg_pool2d: {e}")))?;

    Ok(())
}

/// Performs adaptive max pooling to the target `output_size`.
///
/// Uses the same adaptive window computation as [`adaptive_avg_pool2d`]
/// but selects the maximum rather than the mean.
///
/// # Errors
///
/// Same as [`adaptive_avg_pool2d`].
pub fn adaptive_max_pool2d<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
    output_size: (u32, u32),
) -> DnnResult<()> {
    let (in_n, in_c, in_h, in_w) = nchw_dims(input)?;
    let (out_n, out_c, out_h, out_w) = nchw_dims_mut(output)?;

    if out_n != in_n || out_c != in_c || out_h != output_size.0 || out_w != output_size.1 {
        return Err(DnnError::InvalidDimension(format!(
            "adaptive_max_pool2d: output ({out_n},{out_c},{out_h},{out_w}) != expected ({in_n},{in_c},{},{})",
            output_size.0, output_size.1
        )));
    }
    if output_size.0 == 0 || output_size.1 == 0 {
        return Err(DnnError::InvalidDimension(
            "adaptive_max_pool2d: output_size must be non-zero".into(),
        ));
    }

    let total = output.numel() as u32;
    if total == 0 {
        return Ok(());
    }

    let ptx = generate_adaptive_max_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let name = format!("dnn_adaptive_max_pool2d_{}", T::NAME);
    let kernel = Kernel::from_module(module, &name)?;

    let grid = grid_size_for(total, ADAPTIVE_BLOCK);
    let params = LaunchParams::new(grid, ADAPTIVE_BLOCK);

    let args = (
        input.ptr, output.ptr, in_n, in_c, in_h, in_w, out_h, out_w, total,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("adaptive_max_pool2d: {e}")))?;

    Ok(())
}

/// Generates PTX for adaptive average pooling.
///
/// Each thread computes one output element. The adaptive window boundaries
/// are computed via integer division to replicate PyTorch semantics:
/// ```text
/// h_start = oh * in_h / out_h
/// h_end   = (oh + 1) * in_h / out_h
/// ```
fn generate_adaptive_avg_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_adaptive_avg_pool2d_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(ADAPTIVE_BLOCK)
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

                // Adaptive window: h_start = oh * in_h / out_h
                let h_s_num = b.mul_lo_u32(oh_idx.clone(), in_h.clone());
                let h_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {h_start}, {h_s_num}, {out_h};"));
                let oh_plus1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {oh_plus1}, {oh_idx}, 1;"));
                let h_e_num = b.mul_lo_u32(oh_plus1, in_h.clone());
                let h_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {h_end}, {h_e_num}, {out_h};"));

                let w_s_num = b.mul_lo_u32(ow_idx.clone(), in_w.clone());
                let w_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {w_start}, {w_s_num}, {out_w};"));
                let ow_plus1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {ow_plus1}, {ow_idx}, 1;"));
                let w_e_num = b.mul_lo_u32(ow_plus1, in_w.clone());
                let w_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {w_end}, {w_e_num}, {out_w};"));

                // Base offset for (n, c) plane
                let in_hw = b.mul_lo_u32(in_h.clone(), in_w.clone());
                let c_hw = b.mul_lo_u32(c_idx, in_hw.clone());
                let chw = b.mul_lo_u32(channels, in_hw);
                let n_off = b.mul_lo_u32(n_idx, chw);
                let base = b.add_u32(n_off, c_hw);

                let in_ptr = b.load_param_u64("in_ptr");
                let sum = load_float_imm::<T>(b, 0.0);
                let count = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {count}, 0;"));

                let ih = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {ih}, {h_start};"));
                let loop_h = b.fresh_label("aavg_h");
                let end_h = b.fresh_label("aavg_h_end");
                b.label(&loop_h);
                let p_h = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_h}, {ih}, {h_end};"));
                b.branch_if(p_h, &end_h);

                let jw = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {jw}, {w_start};"));
                let loop_w = b.fresh_label("aavg_w");
                let end_w = b.fresh_label("aavg_w_end");
                b.label(&loop_w);
                let p_w = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_w}, {jw}, {w_end};"));
                b.branch_if(p_w, &end_w);

                let row = b.mul_lo_u32(ih.clone(), in_w.clone());
                let hw = b.add_u32(row, jw.clone());
                let idx = b.add_u32(base.clone(), hw);
                let addr = b.byte_offset_addr(in_ptr.clone(), idx, T::size_u32());
                let val = load_global_float::<T>(b, addr);
                let new_sum = add_float::<T>(b, sum.clone(), val);
                b.raw_ptx(&format!(
                    "mov.{ptx} {sum}, {new_sum};",
                    ptx = crate::ptx_helpers::ptx_type_name::<T>()
                ));
                b.raw_ptx(&format!("add.u32 {count}, {count}, 1;"));

                b.raw_ptx(&format!("add.u32 {jw}, {jw}, 1;"));
                b.branch(&loop_w);
                b.label(&end_w);

                b.raw_ptx(&format!("add.u32 {ih}, {ih}, 1;"));
                b.branch(&loop_h);
                b.label(&end_h);

                let count_f = cvt_u32_to_float::<T>(b, count);
                let result = div_float::<T>(b, sum, count_f);

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, T::size_u32());
                store_global_float::<T>(b, out_addr, result);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("adaptive_avg_pool2d: {e}")))?;

    Ok(ptx)
}

/// Generates PTX for adaptive max pooling.
fn generate_adaptive_max_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_adaptive_max_pool2d_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(ADAPTIVE_BLOCK)
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

                // Adaptive window boundaries
                let h_s_num = b.mul_lo_u32(oh_idx.clone(), in_h.clone());
                let h_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {h_start}, {h_s_num}, {out_h};"));
                let oh_p1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {oh_p1}, {oh_idx}, 1;"));
                let h_e_num = b.mul_lo_u32(oh_p1, in_h.clone());
                let h_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {h_end}, {h_e_num}, {out_h};"));

                let w_s_num = b.mul_lo_u32(ow_idx.clone(), in_w.clone());
                let w_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {w_start}, {w_s_num}, {out_w};"));
                let ow_p1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {ow_p1}, {ow_idx}, 1;"));
                let w_e_num = b.mul_lo_u32(ow_p1, in_w.clone());
                let w_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {w_end}, {w_e_num}, {out_w};"));

                let in_hw = b.mul_lo_u32(in_h, in_w.clone());
                let c_hw = b.mul_lo_u32(c_idx, in_hw.clone());
                let chw = b.mul_lo_u32(channels, in_hw);
                let n_off = b.mul_lo_u32(n_idx, chw);
                let base = b.add_u32(n_off, c_hw);

                let in_ptr = b.load_param_u64("in_ptr");
                let max_val = load_float_imm::<T>(b, f64::NEG_INFINITY);

                let ih = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {ih}, {h_start};"));
                let loop_h = b.fresh_label("amax_h");
                let end_h = b.fresh_label("amax_h_end");
                b.label(&loop_h);
                let p_h = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_h}, {ih}, {h_end};"));
                b.branch_if(p_h, &end_h);

                let jw = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {jw}, {w_start};"));
                let loop_w = b.fresh_label("amax_w");
                let end_w = b.fresh_label("amax_w_end");
                b.label(&loop_w);
                let p_w = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_w}, {jw}, {w_end};"));
                b.branch_if(p_w, &end_w);

                let row = b.mul_lo_u32(ih.clone(), in_w.clone());
                let hw = b.add_u32(row, jw.clone());
                let idx = b.add_u32(base.clone(), hw);
                let addr = b.byte_offset_addr(in_ptr.clone(), idx, T::size_u32());
                let val = load_global_float::<T>(b, addr);
                let is_gt = setp_gt_float::<T>(b, val.clone(), max_val.clone());
                let new_max = selp_float::<T>(b, val, max_val.clone(), is_gt);
                b.raw_ptx(&format!(
                    "mov.{ptx} {max_val}, {new_max};",
                    ptx = crate::ptx_helpers::ptx_type_name::<T>()
                ));

                b.raw_ptx(&format!("add.u32 {jw}, {jw}, 1;"));
                b.branch(&loop_w);
                b.label(&end_w);

                b.raw_ptx(&format!("add.u32 {ih}, {ih}, 1;"));
                b.branch(&loop_h);
                b.label(&end_h);

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, T::size_u32());
                store_global_float::<T>(b, out_addr, max_val);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("adaptive_max_pool2d: {e}")))?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_avg_ptx_f32() {
        let ptx = generate_adaptive_avg_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let s = ptx.expect("should gen");
        assert!(s.contains("dnn_adaptive_avg_pool2d_f32"));
    }

    #[test]
    fn adaptive_max_ptx_f32() {
        let ptx = generate_adaptive_max_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn adaptive_avg_ptx_f64() {
        let ptx = generate_adaptive_avg_ptx::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }
}
