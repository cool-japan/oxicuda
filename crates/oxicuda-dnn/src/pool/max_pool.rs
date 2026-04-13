//! MaxPool2D forward and backward operations.
//!
//! Each thread in the forward pass computes one output element by scanning
//! the pooling window for the maximum value. Optionally stores the index of
//! the max element for use in the backward pass.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::ptx_helpers::*;
use crate::tensor_util::{nchw_dims, nchw_dims_mut};
use crate::types::{TensorDesc, TensorDescMut, pool_output_size};

/// Block size for pooling kernels.
const POOL_BLOCK_SIZE: u32 = 256;

/// Performs 2D max pooling (forward pass).
///
/// Each output element is the maximum value within the corresponding kernel
/// window of the input tensor. When `indices` is provided, the linear index
/// within the H*W plane of the argmax is stored for backpropagation.
///
/// # Tensor layout
///
/// Both input and output must be 4-D NCHW tensors. The output spatial
/// dimensions must equal `floor((input_dim + 2*padding - kernel_size) / stride) + 1`.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if tensor shapes or pooling
/// parameters are inconsistent. Returns [`DnnError::BufferTooSmall`] if
/// the index buffer (when provided) is too small.
pub fn max_pool2d<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
    indices: Option<&mut DeviceBuffer<i32>>,
    kernel_size: (u32, u32),
    stride: (u32, u32),
    padding: (u32, u32),
) -> DnnResult<()> {
    let (in_n, in_c, in_h, in_w) = nchw_dims(input)?;
    let (out_n, out_c, out_h, out_w) = nchw_dims_mut(output)?;

    let expected_oh =
        pool_output_size(in_h, kernel_size.0, stride.0, padding.0).ok_or_else(|| {
            DnnError::InvalidDimension(format!(
                "max_pool2d: invalid output height for h={in_h}, kh={}, sh={}, ph={}",
                kernel_size.0, stride.0, padding.0
            ))
        })?;
    let expected_ow =
        pool_output_size(in_w, kernel_size.1, stride.1, padding.1).ok_or_else(|| {
            DnnError::InvalidDimension(format!(
                "max_pool2d: invalid output width for w={in_w}, kw={}, sw={}, pw={}",
                kernel_size.1, stride.1, padding.1
            ))
        })?;

    if out_n != in_n || out_c != in_c || out_h != expected_oh || out_w != expected_ow {
        return Err(DnnError::InvalidDimension(format!(
            "max_pool2d: output shape ({out_n},{out_c},{out_h},{out_w}) != expected ({in_n},{in_c},{expected_oh},{expected_ow})"
        )));
    }

    if let Some(ref idx) = indices {
        let required = output.numel();
        if idx.len() < required {
            return Err(DnnError::BufferTooSmall {
                expected: required * std::mem::size_of::<i32>(),
                actual: idx.len() * std::mem::size_of::<i32>(),
            });
        }
    }

    let total_output = output.numel() as u32;
    if total_output == 0 {
        return Ok(());
    }

    let has_indices = indices.is_some();
    let ptx = generate_max_pool2d_ptx::<T>(handle.sm_version(), has_indices)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel_name = max_pool2d_kernel_name::<T>(has_indices);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let grid = grid_size_for(total_output, POOL_BLOCK_SIZE);
    let params = LaunchParams::new(grid, POOL_BLOCK_SIZE);

    let idx_ptr = indices.map_or(0u64, |idx| idx.as_device_ptr());

    let args = (
        input.ptr,
        output.ptr,
        idx_ptr,
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
        .map_err(|e| DnnError::LaunchFailed(format!("max_pool2d: {e}")))?;

    Ok(())
}

/// Performs the backward pass for 2D max pooling.
///
/// Scatters gradient values from `grad_output` back to `grad_input` at the
/// positions recorded in `indices` during the forward pass. Uses atomic
/// addition to handle overlapping pooling windows.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if tensor shapes are inconsistent.
pub fn max_pool2d_backward<T: GpuFloat>(
    handle: &DnnHandle,
    grad_output: &TensorDesc<T>,
    indices: &DeviceBuffer<i32>,
    grad_input: &mut TensorDescMut<T>,
) -> DnnResult<()> {
    let (go_n, go_c, _go_h, _go_w) = nchw_dims(grad_output)?;
    let (gi_n, gi_c, gi_h, gi_w) = nchw_dims_mut(grad_input)?;

    if go_n != gi_n || go_c != gi_c {
        return Err(DnnError::InvalidDimension(format!(
            "max_pool2d_backward: batch/channel mismatch: grad_out=({go_n},{go_c}), grad_in=({gi_n},{gi_c})"
        )));
    }

    let total = grad_output.numel() as u32;
    if total == 0 {
        return Ok(());
    }

    if indices.len() < grad_output.numel() {
        return Err(DnnError::BufferTooSmall {
            expected: grad_output.numel() * std::mem::size_of::<i32>(),
            actual: indices.len() * std::mem::size_of::<i32>(),
        });
    }

    let ptx = generate_max_pool2d_backward_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel_name = format!("dnn_max_pool2d_bwd_{}", T::NAME);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let grid = grid_size_for(total, POOL_BLOCK_SIZE);
    let params = LaunchParams::new(grid, POOL_BLOCK_SIZE);

    let in_hw = gi_h * gi_w;
    let args = (
        grad_output.ptr,
        indices.as_device_ptr(),
        grad_input.ptr,
        go_n,
        go_c,
        gi_h,
        gi_w,
        in_hw,
        total,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("max_pool2d_backward: {e}")))?;

    Ok(())
}

fn max_pool2d_kernel_name<T: GpuFloat>(has_indices: bool) -> String {
    if has_indices {
        format!("dnn_max_pool2d_idx_{}", T::NAME)
    } else {
        format!("dnn_max_pool2d_{}", T::NAME)
    }
}

/// Generates PTX for the forward max-pool kernel.
///
/// Thread `gid` computes output[gid]. It decomposes gid into (n, c, oh, ow)
/// indices, scans the kernel window, and writes the max value (and optionally
/// the argmax index).
fn generate_max_pool2d_ptx<T: GpuFloat>(sm: SmVersion, has_indices: bool) -> DnnResult<String> {
    let name = max_pool2d_kernel_name::<T>(has_indices);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(POOL_BLOCK_SIZE)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("idx_ptr", PtxType::U64)
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

                // Decompose gid -> (n, c, oh_idx, ow_idx)
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

                // Start position (signed for negative padding offsets)
                let h_start_raw = b.mul_lo_u32(oh_idx, sh);
                let w_start_raw = b.mul_lo_u32(ow_idx, sw);
                let h_start = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("sub.s32 {h_start}, {h_start_raw}, {ph};"));
                let w_start = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("sub.s32 {w_start}, {w_start_raw}, {pw};"));

                // Base offset: n * C*H*W + c * H*W
                let in_hw_val = b.mul_lo_u32(in_h.clone(), in_w.clone());
                let c_hw = b.mul_lo_u32(c_idx, in_hw_val.clone());
                let chw = b.mul_lo_u32(channels, in_hw_val);
                let n_offset = b.mul_lo_u32(n_idx, chw);
                let base_offset = b.add_u32(n_offset, c_hw);

                let in_ptr = b.load_param_u64("in_ptr");

                // Initialize max_val = -inf, max_idx = -1
                let max_val = load_float_imm::<T>(b, f64::NEG_INFINITY);
                let max_idx = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {max_idx}, -1;"));

                let h_end = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("add.s32 {h_end}, {h_start}, {kh};"));
                let w_end = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("add.s32 {w_end}, {w_start}, {kw};"));

                let ih = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {ih}, {h_start};"));
                let in_h_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {in_h_s32}, {in_h};"));
                let in_w_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {in_w_s32}, {in_w};"));

                // Outer loop over kernel height
                let loop_h = b.fresh_label("pool_h");
                let end_h = b.fresh_label("pool_h_end");
                b.label(&loop_h);
                let ph_cmp = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.s32 {ph_cmp}, {ih}, {h_end};"));
                b.branch_if(ph_cmp, &end_h);

                // Bounds check for h
                let h_valid = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.and.s32 {h_valid}, {ih}, 0, {{true}};"));
                let h_valid2 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!(
                    "setp.lt.and.s32 {h_valid2}, {ih}, {in_h_s32}, {h_valid};"
                ));
                let skip_h = b.fresh_label("skip_h");
                let inv_h = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {inv_h}, {h_valid2};"));
                b.branch_if(inv_h, &skip_h);

                // Inner loop over kernel width
                let iw = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {iw}, {w_start};"));
                let loop_w = b.fresh_label("pool_w");
                let end_w = b.fresh_label("pool_w_end");
                b.label(&loop_w);
                let pw_cmp = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.s32 {pw_cmp}, {iw}, {w_end};"));
                b.branch_if(pw_cmp, &end_w);

                let w_valid = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.and.s32 {w_valid}, {iw}, 0, {{true}};"));
                let w_valid2 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!(
                    "setp.lt.and.s32 {w_valid2}, {iw}, {in_w_s32}, {w_valid};"
                ));
                let skip_w = b.fresh_label("skip_w");
                let inv_w = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {inv_w}, {w_valid2};"));
                b.branch_if(inv_w, &skip_w);

                // Load element: in_ptr[base_offset + ih * in_w + iw]
                let ih_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {ih_u32}, {ih};"));
                let iw_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {iw_u32}, {iw};"));
                let row_off = b.mul_lo_u32(ih_u32, in_w.clone());
                let hw_off = b.add_u32(row_off, iw_u32);
                let elem_idx = b.add_u32(base_offset.clone(), hw_off.clone());
                let addr = b.byte_offset_addr(in_ptr.clone(), elem_idx, T::size_u32());
                let val = load_global_float::<T>(b, addr);

                // Update max
                let is_gt = setp_gt_float::<T>(b, val.clone(), max_val.clone());
                let new_max = selp_float::<T>(b, val, max_val.clone(), is_gt.clone());
                b.raw_ptx(&format!(
                    "mov.{ptx} {max_val}, {new_max};",
                    ptx = crate::ptx_helpers::ptx_type_name::<T>()
                ));

                if has_indices {
                    let hw_off_s32 = b.alloc_reg(PtxType::S32);
                    b.raw_ptx(&format!("mov.b32 {hw_off_s32}, {hw_off};"));
                    let new_idx = b.alloc_reg(PtxType::S32);
                    b.raw_ptx(&format!(
                        "selp.s32 {new_idx}, {hw_off_s32}, {max_idx}, {is_gt};"
                    ));
                    b.raw_ptx(&format!("mov.s32 {max_idx}, {new_idx};"));
                }

                b.label(&skip_w);
                b.raw_ptx(&format!("add.s32 {iw}, {iw}, 1;"));
                b.branch(&loop_w);
                b.label(&end_w);

                b.label(&skip_h);
                b.raw_ptx(&format!("add.s32 {ih}, {ih}, 1;"));
                b.branch(&loop_h);
                b.label(&end_h);

                // Store output
                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid.clone(), T::size_u32());
                store_global_float::<T>(b, out_addr, max_val);

                if has_indices {
                    let idx_ptr = b.load_param_u64("idx_ptr");
                    let idx_addr = b.byte_offset_addr(idx_ptr, gid, 4u32);
                    b.raw_ptx(&format!("st.global.s32 [{idx_addr}], {max_idx};"));
                }
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("max_pool2d: {e}")))?;

    Ok(ptx)
}

/// Generates PTX for the backward max-pool kernel.
fn generate_max_pool2d_backward_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_max_pool2d_bwd_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(POOL_BLOCK_SIZE)
        .param("grad_out_ptr", PtxType::U64)
        .param("idx_ptr", PtxType::U64)
        .param("grad_in_ptr", PtxType::U64)
        .param("batch", PtxType::U32)
        .param("channels", PtxType::U32)
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("in_hw", PtxType::U32)
        .param("total", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let total = b.load_param_u32("total");

            b.if_lt_u32(gid.clone(), total.clone(), move |b| {
                let channels = b.load_param_u32("channels");
                let in_hw = b.load_param_u32("in_hw");

                // Load gradient value
                let grad_out_ptr = b.load_param_u64("grad_out_ptr");
                let go_addr = b.byte_offset_addr(grad_out_ptr, gid.clone(), T::size_u32());
                let grad_val = load_global_float::<T>(b, go_addr);

                // Load index (position in the H*W plane)
                let idx_ptr = b.load_param_u64("idx_ptr");
                let idx_addr = b.byte_offset_addr(idx_ptr, gid.clone(), 4u32);
                let hw_idx = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("ld.global.s32 {hw_idx}, [{idx_addr}];"));

                // Determine which (n, c) plane this thread belongs to.
                // For grad_output shaped (N, C, OH, OW), the nc plane index is
                // computed from the output's spatial extents. But we only need
                // the ratio: nc_idx = gid / (OH * OW). We get OH*OW from
                // total / (N*C). However, it is simpler to compute nc_idx
                // from the flat index of the *grad_output* since we know
                // total = N * C * OH * OW.
                //
                // nc_idx = gid / out_hw, where out_hw = total / (N * C).
                // But we do not pass out_hw directly. Instead, note that the
                // destination base offset = nc_idx * in_hw, and
                // nc_idx = gid / out_hw.
                //
                // A simpler approach: pass the output spatial product as in_hw
                // of the *output* tensor. But we have in_hw = gi_h * gi_w of
                // *input*. We actually need to figure out the (n, c) index
                // from the flat gid within the output tensor.
                //
                // Use: nc_idx = floor(gid * (N*C) / total) -- not exact.
                // Better: pass out_hw as a separate param.
                //
                // Actually, the simplest correct approach:
                // nc_idx = gid / (total / (N * C))
                // But integer division is tricky. Let us just pass out_hw.
                //
                // Since we don't have out_hw, we compute it:
                // out_hw = total / (batch * channels)
                let batch = b.load_param_u32("batch");
                let nc_total = b.mul_lo_u32(batch, channels.clone());
                let out_hw = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {out_hw}, {total}, {nc_total};"));

                let nc_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {nc_idx}, {gid}, {out_hw};"));

                // Check index validity (>= 0 means valid)
                let valid = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.s32 {valid}, {hw_idx}, 0;"));
                let skip_label = b.fresh_label("bwd_skip");
                let inv = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {inv}, {valid};"));
                b.branch_if(inv, &skip_label);

                // Destination: nc_idx * in_hw + hw_idx
                let nc_off = b.mul_lo_u32(nc_idx, in_hw);
                let hw_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {hw_u32}, {hw_idx};"));
                let dst_idx = b.add_u32(nc_off, hw_u32);

                let grad_in_ptr = b.load_param_u64("grad_in_ptr");
                let dst_addr = b.byte_offset_addr(grad_in_ptr, dst_idx, T::size_u32());

                // Atomic add for overlapping windows
                if T::PTX_TYPE == PtxType::F32 {
                    let tmp = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!(
                        "atom.global.add.f32 {tmp}, [{dst_addr}], {grad_val};"
                    ));
                } else {
                    let tmp = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!(
                        "atom.global.add.f64 {tmp}, [{dst_addr}], {grad_val};"
                    ));
                }

                b.label(&skip_label);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("max_pool2d_backward: {e}")))?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_pool2d_ptx_generates_f32() {
        let ptx = generate_max_pool2d_ptx::<f32>(SmVersion::Sm80, false);
        assert!(ptx.is_ok());
        let s = ptx.expect("should generate");
        assert!(s.contains("dnn_max_pool2d_f32"));
    }

    #[test]
    fn max_pool2d_ptx_with_indices_f32() {
        let ptx = generate_max_pool2d_ptx::<f32>(SmVersion::Sm80, true);
        assert!(ptx.is_ok());
        let s = ptx.expect("should generate");
        assert!(s.contains("dnn_max_pool2d_idx_f32"));
    }

    #[test]
    fn max_pool2d_backward_ptx_f32() {
        let ptx = generate_max_pool2d_backward_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let s = ptx.expect("should generate");
        assert!(s.contains("dnn_max_pool2d_bwd_f32"));
    }

    #[test]
    fn max_pool2d_ptx_f64() {
        let ptx = generate_max_pool2d_ptx::<f64>(SmVersion::Sm80, false);
        assert!(ptx.is_ok());
    }
}
