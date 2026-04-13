//! SWAP — swap two vectors
//!
//! Exchanges the elements of vectors `x` and `y` element-wise:
//! ```text
//! for i in 0..n:
//!     temp = x[i*incx]
//!     x[i*incx] = y[i*incy]
//!     y[i*incy] = temp
//! ```

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

use super::axpy::{load_global_float, store_global_float};
use super::{L1_BLOCK_SIZE, required_elements};

/// Swaps vectors `x` and `y` element-wise on the GPU.
///
/// After this call, `x` contains the previous contents of `y` and vice versa.
///
/// # Arguments
///
/// * `handle` — BLAS handle.
/// * `n` — number of elements to swap.
/// * `x` — first vector (modified in-place).
/// * `incx` — stride for `x`. Must be positive.
/// * `y` — second vector (modified in-place).
/// * `incy` — stride for `y`. Must be positive.
///
/// # Errors
///
/// Returns an error if buffers are too small or increments are non-positive.
pub fn swap<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    x: &mut DeviceBuffer<T>,
    incx: i32,
    y: &mut DeviceBuffer<T>,
    incy: i32,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    if incx <= 0 {
        return Err(BlasError::InvalidArgument(format!(
            "incx must be positive, got {incx}"
        )));
    }
    if incy <= 0 {
        return Err(BlasError::InvalidArgument(format!(
            "incy must be positive, got {incy}"
        )));
    }

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

    let ptx = generate_swap_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &swap_kernel_name::<T>())?;

    let grid = grid_size_for(n, L1_BLOCK_SIZE);
    let params = LaunchParams::new(grid, L1_BLOCK_SIZE);

    let args = (
        x.as_device_ptr(),
        y.as_device_ptr(),
        n,
        incx as u32,
        incy as u32,
    );

    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

fn swap_kernel_name<T: GpuFloat>() -> String {
    format!("blas_swap_{}", T::NAME)
}

/// Generates PTX for vector swap.
fn generate_swap_ptx<T: GpuFloat>(sm: SmVersion) -> BlasResult<String> {
    let name = swap_kernel_name::<T>();

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(L1_BLOCK_SIZE)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("incx", PtxType::U32)
        .param("incy", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let incx = b.load_param_u32("incx");
                let incy = b.load_param_u32("incy");

                let x_idx = b.mul_lo_u32(gid.clone(), incx);
                let y_idx = b.mul_lo_u32(gid, incy);

                let x_addr = b.byte_offset_addr(x_ptr.clone(), x_idx.clone(), T::size_u32());
                let y_addr = b.byte_offset_addr(y_ptr.clone(), y_idx.clone(), T::size_u32());

                // Load both values.
                let x_val = load_global_float::<T>(b, x_addr);
                let y_val = load_global_float::<T>(b, y_addr);

                // Store swapped: x gets y's value, y gets x's value.
                let x_store = b.byte_offset_addr(x_ptr, x_idx, T::size_u32());
                let y_store = b.byte_offset_addr(y_ptr, y_idx, T::size_u32());
                store_global_float::<T>(b, x_store, y_val);
                store_global_float::<T>(b, y_store, x_val);
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}
