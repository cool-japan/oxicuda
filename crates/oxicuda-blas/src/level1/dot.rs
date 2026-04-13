//! DOT — `result = x . y` (dot product)
//!
//! Computes the inner product of vectors `x` and `y`. Uses a two-phase
//! parallel reduction:
//!
//! 1. **Phase 1**: Each thread block computes a partial dot product using
//!    shared memory tree reduction. Partial results are written to an
//!    intermediate buffer.
//! 2. **Phase 2**: A single block reduces the partial results to the final
//!    scalar, written to `result`.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

use super::axpy::{
    add_float, load_global_float, mul_float, reinterpret_bits_to_float, store_global_float,
};
use super::{L1_BLOCK_SIZE, required_elements};

/// Computes `result = x . y` (dot product) on the GPU.
///
/// The result is written to `result`, a device buffer of at least 1 element.
/// Uses a two-phase parallel reduction for numerically stable summation across
/// potentially millions of elements.
///
/// # Arguments
///
/// * `handle` — BLAS handle.
/// * `n` — number of elements in the dot product.
/// * `x` — first input vector.
/// * `incx` — stride for `x`. Must be positive.
/// * `y` — second input vector.
/// * `incy` — stride for `y`. Must be positive.
/// * `result` — output buffer (at least 1 element) receiving the scalar result.
///
/// # Errors
///
/// Returns an error if buffers are too small or increments are non-positive.
pub fn dot<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    x: &DeviceBuffer<T>,
    incx: i32,
    y: &DeviceBuffer<T>,
    incy: i32,
    result: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }

    validate_positive_inc(incx, "incx")?;
    validate_positive_inc(incy, "incy")?;

    let x_required = required_elements(n, incx);
    let y_required = required_elements(n, incy);
    if x.len() < x_required {
        return Err(BlasError::BufferTooSmall {
            expected: x_required,
            actual: x.len(),
        });
    }
    if y.len() < y_required {
        return Err(BlasError::BufferTooSmall {
            expected: y_required,
            actual: y.len(),
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

    // Phase 1: partial dot products per block.
    let partials = DeviceBuffer::<T>::zeroed(num_blocks as usize)?;

    let ptx_p1 = generate_dot_phase1_ptx::<T>(sm)?;
    let module_p1 = Arc::new(Module::from_ptx(&ptx_p1)?);
    let kernel_p1 = Kernel::from_module(module_p1, &dot_phase1_name::<T>())?;

    let params_p1 =
        LaunchParams::new(num_blocks, L1_BLOCK_SIZE).with_shared_mem(L1_BLOCK_SIZE * T::size_u32());

    let args_p1 = (
        x.as_device_ptr(),
        y.as_device_ptr(),
        partials.as_device_ptr(),
        n,
        incx as u32,
        incy as u32,
    );
    kernel_p1.launch(&params_p1, handle.stream(), &args_p1)?;

    // Phase 2: reduce partial results to a single scalar.
    let ptx_p2 = generate_reduce_sum_phase2_ptx::<T>(sm)?;
    let module_p2 = Arc::new(Module::from_ptx(&ptx_p2)?);
    let kernel_p2 = Kernel::from_module(module_p2, &reduce_sum_phase2_name::<T>())?;

    let p2_n = num_blocks;
    let p2_blocks = grid_size_for(p2_n, L1_BLOCK_SIZE);
    let params_p2 = if p2_blocks > 1 {
        // For very large grids, we may need iterative reduction.
        // For typical L1 ops, num_blocks < 65536, so a single block suffices
        // after one reduction step. We handle the general case with a loop.
        LaunchParams::new(1u32, L1_BLOCK_SIZE).with_shared_mem(L1_BLOCK_SIZE * T::size_u32())
    } else {
        LaunchParams::new(1u32, L1_BLOCK_SIZE).with_shared_mem(L1_BLOCK_SIZE * T::size_u32())
    };

    let args_p2 = (partials.as_device_ptr(), result.as_device_ptr(), p2_n);
    kernel_p2.launch(&params_p2, handle.stream(), &args_p2)?;

    Ok(())
}

fn validate_positive_inc(inc: i32, name: &str) -> BlasResult<()> {
    if inc <= 0 {
        return Err(BlasError::InvalidArgument(format!(
            "{name} must be positive, got {inc}"
        )));
    }
    Ok(())
}

fn dot_phase1_name<T: GpuFloat>() -> String {
    format!("blas_dot_phase1_{}", T::NAME)
}

/// Phase 2 reduction kernel name — reusable by nrm2, asum, etc.
pub(crate) fn reduce_sum_phase2_name<T: GpuFloat>() -> String {
    format!("blas_reduce_sum_phase2_{}", T::NAME)
}

/// Generates PTX for Phase 1 of dot product.
///
/// Each block computes `sum(x[i*incx] * y[i*incy])` for its range of `i`
/// values, writing the partial sum to `out[blockIdx.x]`.
fn generate_dot_phase1_ptx<T: GpuFloat>(sm: SmVersion) -> BlasResult<String> {
    let name = dot_phase1_name::<T>();
    let float_ty = T::PTX_TYPE;
    let bs = L1_BLOCK_SIZE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(bs)
        .shared_mem("smem", float_ty, bs as usize)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("incx", PtxType::U32)
        .param("incy", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let tid = b.thread_id_x();
            let n_reg = b.load_param_u32("n");

            // Initialize shared memory element to zero.
            let zero_bits: u64 = 0;
            let zero_reg = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("mov.u64 {zero_reg}, {zero_bits};"));
            let zero_float = reinterpret_bits_to_float::<T>(b, zero_reg);

            // Compute shared memory address for this thread.
            let smem_addr = shared_mem_addr_for_tid::<T>(b, tid.clone());

            // If gid < n, compute x[gid*incx] * y[gid*incy]; else store 0.
            let prod = b.alloc_reg(float_ty);
            b.raw_ptx(&format!(
                "mov{} {prod}, {zero_float};",
                float_ty.as_ptx_str()
            ));

            b.if_lt_u32(gid.clone(), n_reg, |b| {
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let incx = b.load_param_u32("incx");
                let incy = b.load_param_u32("incy");

                let x_idx = b.mul_lo_u32(gid.clone(), incx);
                let y_idx = b.mul_lo_u32(gid, incy);

                let x_addr = b.byte_offset_addr(x_ptr, x_idx, T::size_u32());
                let y_addr = b.byte_offset_addr(y_ptr, y_idx, T::size_u32());

                let xv = load_global_float::<T>(b, x_addr);
                let yv = load_global_float::<T>(b, y_addr);

                let p = mul_float::<T>(b, xv, yv);
                b.raw_ptx(&format!("mov{} {prod}, {p};", float_ty.as_ptx_str()));
            });

            // Store to shared memory.
            store_shared_float::<T>(b, smem_addr, prod);
            b.bar_sync(0);

            // Tree reduction in shared memory.
            emit_shared_mem_reduction::<T>(b, tid, bs);

            // Thread 0 writes the block partial to global memory.
            let tid_check = b.thread_id_x();
            let one_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {one_reg}, 1;"));
            b.if_lt_u32(tid_check, one_reg, |b| {
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

/// Generates PTX for Phase 2: sum-reduce an array of partial results.
///
/// A single block reads up to `n` partial values (looping if n > blockDim),
/// accumulates them locally, then performs a shared memory tree reduction
/// to produce the final scalar.
pub(crate) fn generate_reduce_sum_phase2_ptx<T: GpuFloat>(sm: SmVersion) -> BlasResult<String> {
    let name = reduce_sum_phase2_name::<T>();
    let float_ty = T::PTX_TYPE;
    let bs = L1_BLOCK_SIZE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(bs)
        .shared_mem("smem", float_ty, bs as usize)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let n_reg = b.load_param_u32("n");
            let in_ptr = b.load_param_u64("in_ptr");

            // Each thread accumulates multiple elements with stride = blockDim.
            let acc = b.alloc_reg(float_ty);
            let zero_bits: u64 = 0;
            let zr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("mov.u64 {zr}, {zero_bits};"));
            let zf = reinterpret_bits_to_float::<T>(b, zr);
            b.raw_ptx(&format!("mov{} {acc}, {zf};", float_ty.as_ptx_str()));

            // Loop: i = tid; i < n; i += blockDim.x
            let loop_label = b.fresh_label("p2_loop");
            let done_label = b.fresh_label("p2_done");
            let i_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {i_reg}, {tid};"));

            b.label(&loop_label);
            // if i >= n, break
            let cmp = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {cmp}, {i_reg}, {n_reg};"));
            b.branch_if(cmp, &done_label);

            // Load partials[i]
            let elem_addr = b.byte_offset_addr(in_ptr.clone(), i_reg.clone(), T::size_u32());
            let val = load_global_float::<T>(b, elem_addr);
            let new_acc = add_float::<T>(b, acc.clone(), val);
            b.raw_ptx(&format!("mov{} {acc}, {new_acc};", float_ty.as_ptx_str()));

            // i += blockDim.x
            let bdim = b.block_dim_x();
            let next_i = b.add_u32(i_reg.clone(), bdim);
            b.raw_ptx(&format!("mov.u32 {i_reg}, {next_i};"));
            b.branch(&loop_label);

            b.label(&done_label);

            // Store accumulated value to shared memory.
            let smem_addr = shared_mem_addr_for_tid::<T>(b, tid.clone());
            store_shared_float::<T>(b, smem_addr, acc);
            b.bar_sync(0);

            // Tree reduction in shared memory.
            emit_shared_mem_reduction::<T>(b, tid, bs);

            // Thread 0 writes final result.
            let tid2 = b.thread_id_x();
            let one_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {one_reg}, 1;"));
            b.if_lt_u32(tid2, one_reg, |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let smem_base = shared_mem_base_addr(b);
                let final_val = load_shared_float::<T>(b, smem_base);
                store_global_float::<T>(b, out_ptr, final_val);
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Shared memory helpers used by all reduction kernels
// ---------------------------------------------------------------------------

/// Computes the shared memory address for thread `tid`: `&smem[tid * elem_bytes]`.
pub(crate) fn shared_mem_addr_for_tid<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    tid: Register,
) -> Register {
    let base = shared_mem_base_addr(b);
    let tid64 = b.cvt_u32_to_u64(tid);
    let stride = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {stride}, {};", T::size_u32()));
    let offset = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mul.lo.u64 {offset}, {tid64}, {stride};"));
    b.add_u64(base, offset)
}

/// Returns the base address of the `smem` shared memory allocation.
pub(crate) fn shared_mem_base_addr(b: &mut BodyBuilder<'_>) -> Register {
    let addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {addr}, smem;"));
    addr
}

/// Loads a float from shared memory.
pub(crate) fn load_shared_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, addr: Register) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.load_shared_f32(addr)
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("ld.shared.f64 {dst}, [{addr}];"));
        dst
    }
}

/// Stores a float to shared memory.
pub(crate) fn store_shared_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    addr: Register,
    val: Register,
) {
    if T::PTX_TYPE == PtxType::F32 {
        b.store_shared_f32(addr, val);
    } else {
        b.raw_ptx(&format!("st.shared.f64 [{addr}], {val};"));
    }
}

/// Emits a tree-reduction loop in shared memory.
///
/// Reduces `smem[0..block_size]` in-place. After this, `smem[0]` contains the
/// sum of all elements. `tid` must be the thread-local ID.
///
/// Pattern:
/// ```text
/// for stride in [block_size/2, block_size/4, ..., 1]:
///     if tid < stride:
///         smem[tid] += smem[tid + stride]
///     bar.sync
/// ```
pub(crate) fn emit_shared_mem_reduction<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    tid: Register,
    block_size: u32,
) {
    let _float_ty = T::PTX_TYPE;
    let mut stride = block_size / 2;
    while stride > 0 {
        let stride_reg = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("mov.u32 {stride_reg}, {stride};"));

        b.if_lt_u32(tid.clone(), stride_reg, |b| {
            // my_addr = &smem[tid]
            let my_addr = shared_mem_addr_for_tid::<T>(b, tid.clone());
            // partner_tid = tid + stride
            let partner_tid = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("add.u32 {partner_tid}, {tid}, {stride};"));
            let partner_addr = shared_mem_addr_for_tid::<T>(b, partner_tid);

            let my_val = load_shared_float::<T>(b, my_addr.clone());
            let partner_val = load_shared_float::<T>(b, partner_addr);

            let combined = add_float::<T>(b, my_val, partner_val);
            store_shared_float::<T>(b, my_addr, combined);
        });

        b.bar_sync(0);
        stride /= 2;
    }
}
