//! IAMAX — `result = argmax |x_i|` (index of maximum absolute value)
//!
//! Finds the index of the element with the largest absolute value. Uses a
//! two-phase reduction where each block finds its local maximum and index,
//! then a second kernel selects the global winner.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

use super::axpy::{abs_float, load_global_float};
use super::dot::{
    load_shared_float, shared_mem_addr_for_tid, shared_mem_base_addr, store_shared_float,
};
use super::{L1_BLOCK_SIZE, required_elements};

/// Computes `result = argmax_i |x_i|` (IAMAX) on the GPU.
///
/// Returns the 0-based index of the element with the largest absolute value.
/// The index is written to `result_idx[0]` as a `u32`.
///
/// Ties are broken by selecting the smallest index (matching standard BLAS
/// behaviour).
///
/// # Arguments
///
/// * `handle` — BLAS handle.
/// * `n` — number of elements.
/// * `x` — input vector.
/// * `incx` — stride for `x`. Must be positive.
/// * `result_idx` — output buffer for the index (at least 1 element, u32).
///
/// # Errors
///
/// Returns an error if the buffer is too small or the increment is non-positive.
pub fn iamax<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    x: &DeviceBuffer<T>,
    incx: i32,
    result_idx: &mut DeviceBuffer<u32>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    if incx <= 0 {
        return Err(BlasError::InvalidArgument(format!(
            "incx must be positive, got {incx}"
        )));
    }

    let x_required = required_elements(n, incx);
    if x.len() < x_required {
        return Err(BlasError::BufferTooSmall {
            expected: x_required,
            actual: x.len(),
        });
    }
    if result_idx.is_empty() {
        return Err(BlasError::BufferTooSmall {
            expected: 1,
            actual: 0,
        });
    }

    let sm = handle.sm_version();
    let num_blocks = grid_size_for(n, L1_BLOCK_SIZE);

    // Phase 1: each block finds its local (value, index) pair.
    let partial_vals = DeviceBuffer::<T>::zeroed(num_blocks as usize)?;
    let partial_idxs = DeviceBuffer::<u32>::zeroed(num_blocks as usize)?;

    let ptx_p1 = generate_iamax_phase1_ptx::<T>(sm)?;
    let module_p1 = Arc::new(Module::from_ptx(&ptx_p1)?);
    let kernel_p1 = Kernel::from_module(module_p1, &iamax_phase1_name::<T>())?;

    let params_p1 = LaunchParams::new(num_blocks, L1_BLOCK_SIZE).with_shared_mem(
        L1_BLOCK_SIZE * (T::size_u32() + 4), // float + u32 per thread
    );

    let args_p1 = (
        x.as_device_ptr(),
        partial_vals.as_device_ptr(),
        partial_idxs.as_device_ptr(),
        n,
        incx as u32,
    );
    kernel_p1.launch(&params_p1, handle.stream(), &args_p1)?;

    // Phase 2: reduce partial (val, idx) pairs to the global maximum.
    let ptx_p2 = generate_iamax_phase2_ptx::<T>(sm)?;
    let module_p2 = Arc::new(Module::from_ptx(&ptx_p2)?);
    let kernel_p2 = Kernel::from_module(module_p2, &iamax_phase2_name::<T>())?;

    let params_p2 =
        LaunchParams::new(1u32, L1_BLOCK_SIZE).with_shared_mem(L1_BLOCK_SIZE * (T::size_u32() + 4));

    let args_p2 = (
        partial_vals.as_device_ptr(),
        partial_idxs.as_device_ptr(),
        result_idx.as_device_ptr(),
        num_blocks,
    );
    kernel_p2.launch(&params_p2, handle.stream(), &args_p2)?;

    Ok(())
}

fn iamax_phase1_name<T: GpuFloat>() -> String {
    format!("blas_iamax_phase1_{}", T::NAME)
}

fn iamax_phase2_name<T: GpuFloat>() -> String {
    format!("blas_iamax_phase2_{}", T::NAME)
}

/// Phase 1: each block finds the (|x_i|, i) pair with the maximum absolute value.
///
/// Shared memory layout: `[float_values: bs * elem_bytes] [u32_indices: bs * 4]`
fn generate_iamax_phase1_ptx<T: GpuFloat>(sm: SmVersion) -> BlasResult<String> {
    let name = iamax_phase1_name::<T>();
    let float_ty = T::PTX_TYPE;
    let bs = L1_BLOCK_SIZE;
    let val_smem_bytes = (bs as usize) * T::SIZE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(bs)
        .shared_mem("smem_val", float_ty, bs as usize)
        .shared_mem("smem_idx", PtxType::U32, bs as usize)
        .param("x_ptr", PtxType::U64)
        .param("out_val_ptr", PtxType::U64)
        .param("out_idx_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("incx", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let tid = b.thread_id_x();
            let n_reg = b.load_param_u32("n");

            // Initialize value to negative infinity, index to max u32.
            let neg_inf = b.alloc_reg(float_ty);
            if float_ty == PtxType::F32 {
                b.raw_ptx(&format!("mov.b32 {neg_inf}, 0fFF800000;"));
            } else {
                b.raw_ptx(&format!("mov.b64 {neg_inf}, 0dFFF0000000000000;"));
            }
            let max_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {max_idx}, 4294967295;"));

            let my_val = b.alloc_reg(float_ty);
            b.raw_ptx(&format!(
                "mov{} {my_val}, {neg_inf};",
                float_ty.as_ptx_str()
            ));
            let my_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {my_idx}, {max_idx};"));

            b.if_lt_u32(gid.clone(), n_reg, |b| {
                let x_ptr = b.load_param_u64("x_ptr");
                let incx = b.load_param_u32("incx");
                let x_i = b.mul_lo_u32(gid.clone(), incx);
                let x_addr = b.byte_offset_addr(x_ptr, x_i, T::size_u32());
                let xv = load_global_float::<T>(b, x_addr);
                let abs_val = abs_float::<T>(b, xv);
                b.raw_ptx(&format!(
                    "mov{} {my_val}, {abs_val};",
                    float_ty.as_ptx_str()
                ));
                b.raw_ptx(&format!("mov.u32 {my_idx}, {gid};"));
            });

            // Store to val shared memory.
            let smem_val_addr = val_smem_addr::<T>(b, tid.clone());
            store_shared_float::<T>(b, smem_val_addr, my_val.clone());
            // Store to idx shared memory.
            let smem_idx_addr = idx_smem_addr(b, tid.clone(), val_smem_bytes);
            b.raw_ptx(&format!("st.shared.u32 [{smem_idx_addr}], {my_idx};"));
            b.bar_sync(0);

            // Tree reduction: compare-and-swap (val, idx) pairs.
            emit_iamax_reduction::<T>(b, tid, bs, val_smem_bytes);

            // Thread 0 writes block results.
            let tid2 = b.thread_id_x();
            let one_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {one_reg}, 1;"));
            b.if_lt_u32(tid2, one_reg, |b| {
                let out_val = b.load_param_u64("out_val_ptr");
                let out_idx = b.load_param_u64("out_idx_ptr");
                let bid = b.block_id_x();

                let val_addr = val_smem_base(b);
                let final_val = load_shared_float::<T>(b, val_addr);
                let gout_val = b.byte_offset_addr(out_val, bid.clone(), T::size_u32());
                super::axpy::store_global_float::<T>(b, gout_val, final_val);

                let idx_addr = idx_smem_base(b, val_smem_bytes);
                let final_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("ld.shared.u32 {final_idx}, [{idx_addr}];"));
                let gout_idx = b.byte_offset_addr(out_idx, bid, 4);
                b.raw_ptx(&format!("st.global.u32 [{gout_idx}], {final_idx};"));
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Phase 2: reduce partial (val, idx) results to a single global winner.
fn generate_iamax_phase2_ptx<T: GpuFloat>(sm: SmVersion) -> BlasResult<String> {
    let name = iamax_phase2_name::<T>();
    let float_ty = T::PTX_TYPE;
    let bs = L1_BLOCK_SIZE;
    let val_smem_bytes = (bs as usize) * T::SIZE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(bs)
        .shared_mem("smem_val", float_ty, bs as usize)
        .shared_mem("smem_idx", PtxType::U32, bs as usize)
        .param("val_ptr", PtxType::U64)
        .param("idx_ptr", PtxType::U64)
        .param("out_idx_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let n_reg = b.load_param_u32("n");

            // Initialize to negative infinity.
            let neg_inf = b.alloc_reg(float_ty);
            if float_ty == PtxType::F32 {
                b.raw_ptx(&format!("mov.b32 {neg_inf}, 0fFF800000;"));
            } else {
                b.raw_ptx(&format!("mov.b64 {neg_inf}, 0dFFF0000000000000;"));
            }

            let my_val = b.alloc_reg(float_ty);
            b.raw_ptx(&format!(
                "mov{} {my_val}, {neg_inf};",
                float_ty.as_ptx_str()
            ));
            let my_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {my_idx}, 4294967295;"));

            // Load if tid < n.
            b.if_lt_u32(tid.clone(), n_reg, |b| {
                let val_ptr = b.load_param_u64("val_ptr");
                let idx_ptr = b.load_param_u64("idx_ptr");
                let v_addr = b.byte_offset_addr(val_ptr, tid.clone(), T::size_u32());
                let i_addr = b.byte_offset_addr(idx_ptr, tid.clone(), 4);
                let v = load_global_float::<T>(b, v_addr);
                b.raw_ptx(&format!("mov{} {my_val}, {v};", float_ty.as_ptx_str()));
                let i = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("ld.global.u32 {i}, [{i_addr}];"));
                b.raw_ptx(&format!("mov.u32 {my_idx}, {i};"));
            });

            let smem_val_addr = val_smem_addr::<T>(b, tid.clone());
            store_shared_float::<T>(b, smem_val_addr, my_val);
            let smem_idx_addr = idx_smem_addr(b, tid.clone(), val_smem_bytes);
            b.raw_ptx(&format!("st.shared.u32 [{smem_idx_addr}], {my_idx};"));
            b.bar_sync(0);

            emit_iamax_reduction::<T>(b, tid, bs, val_smem_bytes);

            // Thread 0 writes the winning index.
            let tid2 = b.thread_id_x();
            let one_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {one_reg}, 1;"));
            b.if_lt_u32(tid2, one_reg, |b| {
                let out_ptr = b.load_param_u64("out_idx_ptr");
                let idx_addr = idx_smem_base(b, val_smem_bytes);
                let winner_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("ld.shared.u32 {winner_idx}, [{idx_addr}];"));
                b.raw_ptx(&format!("st.global.u32 [{out_ptr}], {winner_idx};"));
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Shared memory address helpers for dual-array (val + idx) layout
// ---------------------------------------------------------------------------

fn val_smem_addr<T: GpuFloat>(b: &mut BodyBuilder<'_>, tid: Register) -> Register {
    shared_mem_addr_for_tid::<T>(b, tid)
}

fn val_smem_base(b: &mut BodyBuilder<'_>) -> Register {
    shared_mem_base_addr(b)
}

fn idx_smem_addr(b: &mut BodyBuilder<'_>, tid: Register, _val_smem_bytes: usize) -> Register {
    let base = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {base}, smem_idx;"));
    let tid64 = b.cvt_u32_to_u64(tid);
    let stride = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {stride}, 4;"));
    let offset = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mul.lo.u64 {offset}, {tid64}, {stride};"));
    b.add_u64(base, offset)
}

fn idx_smem_base(b: &mut BodyBuilder<'_>, _val_smem_bytes: usize) -> Register {
    let base = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {base}, smem_idx;"));
    base
}

/// Tree reduction for (value, index) pairs: keep the larger absolute value.
/// On ties, keep the smaller index (BLAS convention).
fn emit_iamax_reduction<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    tid: Register,
    block_size: u32,
    val_smem_bytes: usize,
) {
    let float_ty = T::PTX_TYPE;
    let mut stride = block_size / 2;

    while stride > 0 {
        let stride_reg = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("mov.u32 {stride_reg}, {stride};"));

        b.if_lt_u32(tid.clone(), stride_reg, |b| {
            // Load my (val, idx).
            let my_val_addr = val_smem_addr::<T>(b, tid.clone());
            let my_val = load_shared_float::<T>(b, my_val_addr.clone());
            let my_idx_addr = idx_smem_addr(b, tid.clone(), val_smem_bytes);
            let my_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("ld.shared.u32 {my_idx}, [{my_idx_addr}];"));

            // Load partner (val, idx).
            let partner_tid = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("add.u32 {partner_tid}, {tid}, {stride};"));
            let p_val_addr = val_smem_addr::<T>(b, partner_tid.clone());
            let p_val = load_shared_float::<T>(b, p_val_addr);
            let p_idx_addr = idx_smem_addr(b, partner_tid, val_smem_bytes);
            let p_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("ld.shared.u32 {p_idx}, [{p_idx_addr}];"));

            // Compare: if partner > my OR (partner == my AND p_idx < my_idx),
            // then take partner.
            let use_partner = b.alloc_reg(PtxType::Pred);
            let gt_pred = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!(
                "setp.gt{} {gt_pred}, {p_val}, {my_val};",
                float_ty.as_ptx_str()
            ));
            let eq_pred = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!(
                "setp.eq{} {eq_pred}, {p_val}, {my_val};",
                float_ty.as_ptx_str()
            ));
            let idx_lt = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lt.u32 {idx_lt}, {p_idx}, {my_idx};"));
            let eq_and_lt = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("and.pred {eq_and_lt}, {eq_pred}, {idx_lt};"));
            b.raw_ptx(&format!("or.pred {use_partner}, {gt_pred}, {eq_and_lt};"));

            // Conditional update.
            let new_val = b.alloc_reg(float_ty);
            b.raw_ptx(&format!(
                "selp{} {new_val}, {p_val}, {my_val}, {use_partner};",
                float_ty.as_ptx_str()
            ));
            let new_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "selp.u32 {new_idx}, {p_idx}, {my_idx}, {use_partner};"
            ));

            store_shared_float::<T>(b, my_val_addr, new_val);
            b.raw_ptx(&format!("st.shared.u32 [{my_idx_addr}], {new_idx};"));
        });

        b.bar_sync(0);
        stride /= 2;
    }
}
