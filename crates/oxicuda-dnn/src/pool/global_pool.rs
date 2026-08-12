//! Global pooling operations.
//!
//! Global pooling reduces each (N, C) plane from (H, W) to a single scalar.
//! This is equivalent to adaptive pooling with `output_size = (1, 1)`.
//!
//! Uses a dedicated kernel that performs a channel-wise reduction over the
//! entire spatial extent, with one thread block per (N, C) plane.

use oxicuda_blas::GpuFloat;
use oxicuda_launch::LaunchParams;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::kernel_cache::cache_key;
use crate::ptx_helpers::*;
use crate::tensor_util::{nchw_dims, nchw_dims_mut};
use crate::types::{TensorDesc, TensorDescMut};

/// Block size for global pooling (reduction) kernels.
const GLOBAL_POOL_BLOCK: u32 = 256;

/// Performs global average pooling: each (N, C) plane -> single mean value.
///
/// The output tensor must have shape `(N, C, 1, 1)`.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if the output is not `(N, C, 1, 1)`.
pub fn global_avg_pool2d<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
) -> DnnResult<()> {
    let (in_n, in_c, in_h, in_w) = nchw_dims(input)?;
    let (out_n, out_c, out_h, out_w) = nchw_dims_mut(output)?;

    if out_n != in_n || out_c != in_c || out_h != 1 || out_w != 1 {
        return Err(DnnError::InvalidDimension(format!(
            "global_avg_pool2d: output ({out_n},{out_c},{out_h},{out_w}) must be ({in_n},{in_c},1,1)"
        )));
    }

    let nc = in_n * in_c;
    if nc == 0 {
        return Ok(());
    }
    let hw = in_h * in_w;

    let name = format!("dnn_global_avg_pool2d_{}", T::NAME);
    let kernel =
        handle.get_or_compile_kernel(&cache_key(&name, handle.sm_version()), &name, || {
            generate_global_avg_ptx::<T>(handle.sm_version())
        })?;

    // One block per (N, C) plane
    let params = LaunchParams::new(nc, GLOBAL_POOL_BLOCK);

    let args = (input.ptr, output.ptr, hw, nc);

    kernel
        .kernel()
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("global_avg_pool2d: {e}")))?;

    Ok(())
}

/// Performs global max pooling: each (N, C) plane -> single max value.
///
/// The output tensor must have shape `(N, C, 1, 1)`.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if the output is not `(N, C, 1, 1)`.
pub fn global_max_pool2d<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
) -> DnnResult<()> {
    let (in_n, in_c, in_h, in_w) = nchw_dims(input)?;
    let (out_n, out_c, out_h, out_w) = nchw_dims_mut(output)?;

    if out_n != in_n || out_c != in_c || out_h != 1 || out_w != 1 {
        return Err(DnnError::InvalidDimension(format!(
            "global_max_pool2d: output ({out_n},{out_c},{out_h},{out_w}) must be ({in_n},{in_c},1,1)"
        )));
    }

    let nc = in_n * in_c;
    if nc == 0 {
        return Ok(());
    }
    let hw = in_h * in_w;

    let name = format!("dnn_global_max_pool2d_{}", T::NAME);
    let kernel =
        handle.get_or_compile_kernel(&cache_key(&name, handle.sm_version()), &name, || {
            generate_global_max_ptx::<T>(handle.sm_version())
        })?;

    let params = LaunchParams::new(nc, GLOBAL_POOL_BLOCK);

    let args = (input.ptr, output.ptr, hw, nc);

    kernel
        .kernel()
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("global_max_pool2d: {e}")))?;

    Ok(())
}

/// Generates PTX for global average pooling.
///
/// Each block handles one (N, C) plane. Threads cooperatively load elements,
/// accumulate into shared memory using tree reduction, and the first thread
/// writes `sum / hw` to the output.
fn generate_global_avg_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_global_avg_pool2d_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(GLOBAL_POOL_BLOCK)
        .shared_mem("smem", T::PTX_TYPE, GLOBAL_POOL_BLOCK as usize)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("hw", PtxType::U32)
        .param("nc", PtxType::U32)
        .body(move |b| {
            let bid = b.block_id_x();
            let tid = b.thread_id_x();
            let bdim = b.block_dim_x();
            let hw = b.load_param_u32("hw");
            let nc = b.load_param_u32("nc");

            // Guard: block id < nc
            b.if_lt_u32(bid.clone(), nc, move |b| {
                // Base address for this plane: in_ptr + bid * hw * sizeof(T)
                let in_ptr = b.load_param_u64("in_ptr");
                let plane_off = b.mul_lo_u32(bid.clone(), hw.clone());
                let base_addr = b.byte_offset_addr(in_ptr, plane_off, T::size_u32());

                // Base address of the shared array. PTX forbids
                // `[sym + reg*imm]` operands, so all shared accesses index
                // off this register-held base.
                let smem_base = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {smem_base}, smem;"));

                // Thread accumulates partial sum across the plane
                let partial = load_float_imm::<T>(b, 0.0);
                let i = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {i}, {tid};"));

                let loop_lbl = b.fresh_label("gavg_loop");
                let end_lbl = b.fresh_label("gavg_end");
                b.label(&loop_lbl);
                let p_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_done}, {i}, {hw};"));
                b.branch_if(p_done, &end_lbl);

                let addr = b.byte_offset_addr(base_addr.clone(), i.clone(), T::size_u32());
                let val = load_global_float::<T>(b, addr);
                let new_partial = add_float::<T>(b, partial.clone(), val);
                b.raw_ptx(&format!(
                    "mov.{ptx} {partial}, {new_partial};",
                    ptx = crate::ptx_helpers::ptx_type_name::<T>()
                ));

                b.raw_ptx(&format!("add.u32 {i}, {i}, {bdim};"));
                b.branch(&loop_lbl);
                b.label(&end_lbl);

                // Store partial to shared memory (address pre-computed)
                if T::PTX_TYPE == PtxType::F32 {
                    let st_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mad.lo.u32 {st_addr}, {tid}, 4, {smem_base};"));
                    b.raw_ptx(&format!("st.shared.f32 [{st_addr}], {partial};"));
                } else {
                    let st_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mad.lo.u32 {st_addr}, {tid}, 8, {smem_base};"));
                    b.raw_ptx(&format!("st.shared.f64 [{st_addr}], {partial};"));
                }

                b.bar_sync(0);

                // Tree reduction in shared memory
                let stride_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shr.u32 {stride_reg}, {bdim}, 1;"));

                let red_loop = b.fresh_label("gavg_red");
                let red_end = b.fresh_label("gavg_red_end");
                b.label(&red_loop);
                let p_red = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {p_red}, {stride_reg}, 0;"));
                b.branch_if(p_red, &red_end);

                let p_active = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lt.u32 {p_active}, {tid}, {stride_reg};"));
                let skip_red = b.fresh_label("gavg_skip_red");
                let inv_active = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {inv_active}, {p_active};"));
                b.branch_if(inv_active, &skip_red);

                // Load smem[tid] and smem[tid + stride]
                let other_idx = b.add_u32(tid.clone(), stride_reg.clone());
                if T::PTX_TYPE == PtxType::F32 {
                    let self_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mad.lo.u32 {self_addr}, {tid}, 4, {smem_base};"));
                    let other_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {other_addr}, {other_idx}, 4, {smem_base};"
                    ));
                    let a = b.alloc_reg(PtxType::F32);
                    let bv = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("ld.shared.f32 {a}, [{self_addr}];"));
                    b.raw_ptx(&format!("ld.shared.f32 {bv}, [{other_addr}];"));
                    let s = b.add_f32(a, bv);
                    b.raw_ptx(&format!("st.shared.f32 [{self_addr}], {s};"));
                } else {
                    let self_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mad.lo.u32 {self_addr}, {tid}, 8, {smem_base};"));
                    let other_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {other_addr}, {other_idx}, 8, {smem_base};"
                    ));
                    let a = b.alloc_reg(PtxType::F64);
                    let bv = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!("ld.shared.f64 {a}, [{self_addr}];"));
                    b.raw_ptx(&format!("ld.shared.f64 {bv}, [{other_addr}];"));
                    let s = b.add_f64(a, bv);
                    b.raw_ptx(&format!("st.shared.f64 [{self_addr}], {s};"));
                }

                b.label(&skip_red);
                b.bar_sync(0);
                b.raw_ptx(&format!("shr.u32 {stride_reg}, {stride_reg}, 1;"));
                b.branch(&red_loop);
                b.label(&red_end);

                // Thread 0 writes result
                let p_tid0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {p_tid0}, {tid}, 0;"));
                let skip_write = b.fresh_label("gavg_skip_write");
                let inv_tid0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {inv_tid0}, {p_tid0};"));
                b.branch_if(inv_tid0, &skip_write);

                let final_sum = if T::PTX_TYPE == PtxType::F32 {
                    let r = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("ld.shared.f32 {r}, [{smem_base}];"));
                    r
                } else {
                    let r = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!("ld.shared.f64 {r}, [{smem_base}];"));
                    r
                };

                let hw_f = cvt_u32_to_float::<T>(b, hw);
                let mean = div_float::<T>(b, final_sum, hw_f);

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, bid, T::size_u32());
                store_global_float::<T>(b, out_addr, mean);

                b.label(&skip_write);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("global_avg_pool2d: {e}")))?;

    Ok(ptx)
}

/// Generates PTX for global max pooling.
fn generate_global_max_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let name = format!("dnn_global_max_pool2d_{}", T::NAME);

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(GLOBAL_POOL_BLOCK)
        .shared_mem("smem", T::PTX_TYPE, GLOBAL_POOL_BLOCK as usize)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("hw", PtxType::U32)
        .param("nc", PtxType::U32)
        .body(move |b| {
            let bid = b.block_id_x();
            let tid = b.thread_id_x();
            let bdim = b.block_dim_x();
            let hw = b.load_param_u32("hw");
            let nc = b.load_param_u32("nc");

            b.if_lt_u32(bid.clone(), nc, move |b| {
                let in_ptr = b.load_param_u64("in_ptr");
                let plane_off = b.mul_lo_u32(bid.clone(), hw.clone());
                let base_addr = b.byte_offset_addr(in_ptr, plane_off, T::size_u32());

                // Base address of the shared array (see global_avg note).
                let smem_base = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {smem_base}, smem;"));

                let partial = load_float_imm::<T>(b, f64::NEG_INFINITY);
                let i = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {i}, {tid};"));

                let loop_lbl = b.fresh_label("gmax_loop");
                let end_lbl = b.fresh_label("gmax_end");
                b.label(&loop_lbl);
                let p_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_done}, {i}, {hw};"));
                b.branch_if(p_done, &end_lbl);

                let addr = b.byte_offset_addr(base_addr.clone(), i.clone(), T::size_u32());
                let val = load_global_float::<T>(b, addr);
                let new_partial = max_float::<T>(b, partial.clone(), val);
                b.raw_ptx(&format!(
                    "mov.{ptx} {partial}, {new_partial};",
                    ptx = crate::ptx_helpers::ptx_type_name::<T>()
                ));

                b.raw_ptx(&format!("add.u32 {i}, {i}, {bdim};"));
                b.branch(&loop_lbl);
                b.label(&end_lbl);

                // Store to shared memory (address pre-computed)
                if T::PTX_TYPE == PtxType::F32 {
                    let st_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mad.lo.u32 {st_addr}, {tid}, 4, {smem_base};"));
                    b.raw_ptx(&format!("st.shared.f32 [{st_addr}], {partial};"));
                } else {
                    let st_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mad.lo.u32 {st_addr}, {tid}, 8, {smem_base};"));
                    b.raw_ptx(&format!("st.shared.f64 [{st_addr}], {partial};"));
                }
                b.bar_sync(0);

                // Tree reduction (max)
                let stride_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shr.u32 {stride_reg}, {bdim}, 1;"));

                let red_loop = b.fresh_label("gmax_red");
                let red_end = b.fresh_label("gmax_red_end");
                b.label(&red_loop);
                let p_red = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {p_red}, {stride_reg}, 0;"));
                b.branch_if(p_red, &red_end);

                let p_active = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lt.u32 {p_active}, {tid}, {stride_reg};"));
                let skip_red = b.fresh_label("gmax_skip_red");
                let inv_a = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {inv_a}, {p_active};"));
                b.branch_if(inv_a, &skip_red);

                let other_idx = b.add_u32(tid.clone(), stride_reg.clone());
                if T::PTX_TYPE == PtxType::F32 {
                    let self_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mad.lo.u32 {self_addr}, {tid}, 4, {smem_base};"));
                    let other_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {other_addr}, {other_idx}, 4, {smem_base};"
                    ));
                    let a = b.alloc_reg(PtxType::F32);
                    let bv = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("ld.shared.f32 {a}, [{self_addr}];"));
                    b.raw_ptx(&format!("ld.shared.f32 {bv}, [{other_addr}];"));
                    let m = b.max_f32(a, bv);
                    b.raw_ptx(&format!("st.shared.f32 [{self_addr}], {m};"));
                } else {
                    let self_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mad.lo.u32 {self_addr}, {tid}, 8, {smem_base};"));
                    let other_addr = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {other_addr}, {other_idx}, 8, {smem_base};"
                    ));
                    let a = b.alloc_reg(PtxType::F64);
                    let bv = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!("ld.shared.f64 {a}, [{self_addr}];"));
                    b.raw_ptx(&format!("ld.shared.f64 {bv}, [{other_addr}];"));
                    let m = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!("max.f64 {m}, {a}, {bv};"));
                    b.raw_ptx(&format!("st.shared.f64 [{self_addr}], {m};"));
                }

                b.label(&skip_red);
                b.bar_sync(0);
                b.raw_ptx(&format!("shr.u32 {stride_reg}, {stride_reg}, 1;"));
                b.branch(&red_loop);
                b.label(&red_end);

                // Thread 0 writes result
                let p_tid0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {p_tid0}, {tid}, 0;"));
                let skip_write = b.fresh_label("gmax_skip_w");
                let inv_t0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {inv_t0}, {p_tid0};"));
                b.branch_if(inv_t0, &skip_write);

                let final_max = if T::PTX_TYPE == PtxType::F32 {
                    let r = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("ld.shared.f32 {r}, [{smem_base}];"));
                    r
                } else {
                    let r = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!("ld.shared.f64 {r}, [{smem_base}];"));
                    r
                };

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, bid, T::size_u32());
                store_global_float::<T>(b, out_addr, final_max);

                b.label(&skip_write);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("global_max_pool2d: {e}")))?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_avg_ptx_f32() {
        let ptx = generate_global_avg_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let s = ptx.expect("should gen");
        assert!(s.contains("dnn_global_avg_pool2d_f32"));
    }

    #[test]
    fn global_max_ptx_f32() {
        let ptx = generate_global_max_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn global_avg_ptx_f64() {
        let ptx = generate_global_avg_ptx::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }
}
