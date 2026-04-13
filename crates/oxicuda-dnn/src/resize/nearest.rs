//! Nearest-neighbor resize.
//!
//! Each output thread computes its source pixel via floor division of
//! the scaled coordinates. No interpolation is performed.

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

/// Block size for resize kernels.
const RESIZE_BLOCK: u32 = 256;

/// Resizes a 4-D NCHW tensor using nearest-neighbor interpolation.
///
/// The output spatial dimensions are taken from `output.dims`. The input
/// and output must share the same batch size and channel count.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if batch or channel counts differ.
pub fn resize_nearest<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
) -> DnnResult<()> {
    let (in_n, in_c, in_h, in_w) = nchw_dims(input)?;
    let (out_n, out_c, out_h, out_w) = nchw_dims_mut(output)?;

    if in_n != out_n || in_c != out_c {
        return Err(DnnError::InvalidDimension(format!(
            "resize_nearest: batch/channel mismatch: in=({in_n},{in_c}), out=({out_n},{out_c})"
        )));
    }

    let total = output.numel() as u32;
    if total == 0 {
        return Ok(());
    }

    let ptx = generate_nearest_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let name = format!("dnn_resize_nearest_{}", T::NAME);
    let kernel = Kernel::from_module(module, &name)?;

    let grid = grid_size_for(total, RESIZE_BLOCK);
    let params = LaunchParams::new(grid, RESIZE_BLOCK);

    let args = (
        input.ptr, output.ptr, in_n, in_c, in_h, in_w, out_h, out_w, total,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("resize_nearest: {e}")))?;

    Ok(())
}

/// Generates PTX for nearest-neighbor resize.
///
/// For each output element at (n, c, oh, ow):
///   ih = oh * in_h / out_h
///   iw = ow * in_w / out_w
fn generate_nearest_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_resize_nearest_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(RESIZE_BLOCK)
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

                // Compute source coordinates: ih = oh * in_h / out_h
                let ih_num = b.mul_lo_u32(oh_idx, in_h.clone());
                let ih = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {ih}, {ih_num}, {out_h};"));
                let iw_num = b.mul_lo_u32(ow_idx, in_w.clone());
                let iw = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {iw}, {iw_num}, {out_w};"));

                // Input index: n * C*H*W + c * H*W + ih * W + iw
                let in_hw = b.mul_lo_u32(in_h, in_w.clone());
                let c_hw = b.mul_lo_u32(c_idx, in_hw.clone());
                let chw = b.mul_lo_u32(channels, in_hw);
                let n_off = b.mul_lo_u32(n_idx, chw);
                let base = b.add_u32(n_off, c_hw);
                let row = b.mul_lo_u32(ih, in_w);
                let hw = b.add_u32(row, iw);
                let src_idx = b.add_u32(base, hw);

                let in_ptr = b.load_param_u64("in_ptr");
                let addr = b.byte_offset_addr(in_ptr, src_idx, T::size_u32());
                let val = load_global_float::<T>(b, addr);

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, T::size_u32());
                store_global_float::<T>(b, out_addr, val);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("resize_nearest: {e}")))?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_ptx_f32() {
        let ptx = generate_nearest_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let s = ptx.expect("should gen");
        assert!(s.contains("dnn_resize_nearest_f32"));
    }

    #[test]
    fn nearest_ptx_f64() {
        let ptx = generate_nearest_ptx::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }
}
