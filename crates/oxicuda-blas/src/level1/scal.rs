//! SCAL — `x = alpha * x`
//!
//! Scales every element of vector `x` by scalar `alpha` in-place.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

use super::axpy::{load_global_float, reinterpret_bits_to_float, store_global_float};
use super::{L1_BLOCK_SIZE, required_elements};

/// Computes `x = alpha * x` (SCAL) on the GPU.
///
/// Scales every element of `x` by the scalar `alpha` in-place. If `n` is zero,
/// the function returns immediately. If `alpha` is 1.0, no kernel is launched.
///
/// # Arguments
///
/// * `handle` — BLAS handle providing the CUDA stream and device info.
/// * `n` — number of elements to scale.
/// * `alpha` — scalar multiplier.
/// * `x` — vector to scale in-place.
/// * `incx` — stride between consecutive elements. Must be positive.
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if `x` is too short for `n` and `incx`.
/// Returns [`BlasError::InvalidArgument`] if `incx` is non-positive.
pub fn scal<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    alpha: T,
    x: &mut DeviceBuffer<T>,
    incx: i32,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    if alpha == T::gpu_one() {
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

    let ptx = generate_scal_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &scal_kernel_name::<T>())?;

    let grid = grid_size_for(n, L1_BLOCK_SIZE);
    let params = LaunchParams::new(grid, L1_BLOCK_SIZE);

    let args = (x.as_device_ptr(), alpha.to_bits_u64(), n, incx as u32);

    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

/// Returns the kernel function name for SCAL.
fn scal_kernel_name<T: GpuFloat>() -> String {
    format!("blas_scal_{}", T::NAME)
}

/// Generates PTX for the SCAL kernel: `x[i*incx] = alpha * x[i*incx]`.
fn generate_scal_ptx<T: GpuFloat>(sm: SmVersion) -> BlasResult<String> {
    let name = scal_kernel_name::<T>();

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(L1_BLOCK_SIZE)
        .param("x_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("n", PtxType::U32)
        .param("incx", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let x_ptr = b.load_param_u64("x_ptr");
                let incx = b.load_param_u32("incx");

                let x_idx = b.mul_lo_u32(gid, incx);
                let x_addr = b.byte_offset_addr(x_ptr.clone(), x_idx.clone(), T::size_u32());

                let alpha_bits = b.load_param_u64("alpha_bits");
                let alpha = reinterpret_bits_to_float::<T>(b, alpha_bits);

                let x_val = load_global_float::<T>(b, x_addr);

                // result = alpha * x_val
                let result = super::axpy::mul_float::<T>(b, alpha, x_val);

                let x_store = b.byte_offset_addr(x_ptr, x_idx, T::size_u32());
                store_global_float::<T>(b, x_store, result);
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}
