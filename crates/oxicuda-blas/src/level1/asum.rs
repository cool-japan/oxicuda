//! ASUM — `result = sum |x_i|` (absolute sum / L1 norm)
//!
//! Computes the sum of absolute values of all elements in vector `x`.
//! Uses a two-phase parallel reduction identical in structure to [`fn@super::dot`]
//! and [`fn@super::nrm2`], but the per-element operation is `|x_i|`
//! instead of `x_i * y_i` or `x_i^2`.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

use super::axpy::{abs_float, load_global_float, reinterpret_bits_to_float, store_global_float};
use super::dot::{
    emit_shared_mem_reduction, generate_reduce_sum_phase2_ptx, load_shared_float,
    reduce_sum_phase2_name, shared_mem_addr_for_tid, shared_mem_base_addr, store_shared_float,
};
use super::{L1_BLOCK_SIZE, required_elements};

/// Computes `result = sum |x_i|` (ASUM / L1 norm) on the GPU.
///
/// The result is written to `result[0]`.
///
/// # Arguments
///
/// * `handle` — BLAS handle.
/// * `n` — number of elements.
/// * `x` — input vector.
/// * `incx` — stride for `x`. Must be positive.
/// * `result` — output buffer (at least 1 element).
///
/// # Errors
///
/// Returns an error if the buffer is too small or the increment is non-positive.
pub fn asum<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    x: &DeviceBuffer<T>,
    incx: i32,
    result: &mut DeviceBuffer<T>,
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
    if result.is_empty() {
        return Err(BlasError::BufferTooSmall {
            expected: 1,
            actual: 0,
        });
    }

    let sm = handle.sm_version();
    let num_blocks = grid_size_for(n, L1_BLOCK_SIZE);

    // Phase 1: partial absolute sums per block.
    let partials = DeviceBuffer::<T>::zeroed(num_blocks as usize)?;

    let ptx_p1 = generate_asum_phase1_ptx::<T>(sm)?;
    let module_p1 = Arc::new(Module::from_ptx(&ptx_p1)?);
    let kernel_p1 = Kernel::from_module(module_p1, &asum_phase1_name::<T>())?;

    let params_p1 =
        LaunchParams::new(num_blocks, L1_BLOCK_SIZE).with_shared_mem(L1_BLOCK_SIZE * T::size_u32());

    let args_p1 = (x.as_device_ptr(), partials.as_device_ptr(), n, incx as u32);
    kernel_p1.launch(&params_p1, handle.stream(), &args_p1)?;

    // Phase 2: reduce partials to final scalar (reuse dot's phase 2).
    let ptx_p2 = generate_reduce_sum_phase2_ptx::<T>(sm)?;
    let module_p2 = Arc::new(Module::from_ptx(&ptx_p2)?);
    let kernel_p2 = Kernel::from_module(module_p2, &reduce_sum_phase2_name::<T>())?;

    let params_p2 =
        LaunchParams::new(1u32, L1_BLOCK_SIZE).with_shared_mem(L1_BLOCK_SIZE * T::size_u32());

    let args_p2 = (partials.as_device_ptr(), result.as_device_ptr(), num_blocks);
    kernel_p2.launch(&params_p2, handle.stream(), &args_p2)?;

    Ok(())
}

fn asum_phase1_name<T: GpuFloat>() -> String {
    format!("blas_asum_phase1_{}", T::NAME)
}

/// Phase 1: each block computes sum(|x[i]|) for its range.
fn generate_asum_phase1_ptx<T: GpuFloat>(sm: SmVersion) -> BlasResult<String> {
    let name = asum_phase1_name::<T>();
    let float_ty = T::PTX_TYPE;
    let bs = L1_BLOCK_SIZE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(bs)
        .shared_mem("smem", float_ty, bs as usize)
        .param("x_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("incx", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let tid = b.thread_id_x();
            let n_reg = b.load_param_u32("n");

            let zr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("mov.u64 {zr}, 0;"));
            let zero_float = reinterpret_bits_to_float::<T>(b, zr);

            let val = b.alloc_reg(float_ty);
            b.raw_ptx(&format!(
                "mov{} {val}, {zero_float};",
                float_ty.as_ptx_str()
            ));

            b.if_lt_u32(gid.clone(), n_reg, |b| {
                let x_ptr = b.load_param_u64("x_ptr");
                let incx = b.load_param_u32("incx");
                let x_idx = b.mul_lo_u32(gid, incx);
                let x_addr = b.byte_offset_addr(x_ptr, x_idx, T::size_u32());
                let xv = load_global_float::<T>(b, x_addr);
                let abs_val = abs_float::<T>(b, xv);
                b.raw_ptx(&format!("mov{} {val}, {abs_val};", float_ty.as_ptx_str()));
            });

            let smem_addr = shared_mem_addr_for_tid::<T>(b, tid.clone());
            store_shared_float::<T>(b, smem_addr, val);
            b.bar_sync(0);

            emit_shared_mem_reduction::<T>(b, tid, bs);

            let tid2 = b.thread_id_x();
            let one_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {one_reg}, 1;"));
            b.if_lt_u32(tid2, one_reg, |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let bid = b.block_id_x();
                let out_addr = b.byte_offset_addr(out_ptr, bid, T::size_u32());
                let smem_base = shared_mem_base_addr(b);
                let final_val = load_shared_float::<T>(b, smem_base);
                store_global_float::<T>(b, out_addr, final_val);
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}
