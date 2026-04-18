//! Device-fill operation: write a scalar to every element of a DeviceBuffer.
//!
//! This module provides the [`fill`] function which performs the equivalent of
//! `for i in 0..n { dst[i] = value; }` but executes entirely on the GPU using
//! a PTX kernel generated at runtime via [`ElementwiseTemplate`].

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::templates::elementwise::{ElementwiseOp, ElementwiseTemplate};

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

/// Standard block size for the fill kernel.
const BLOCK_SIZE: u32 = 256;

/// Fills every element of `dst[0..n]` with `value` on the GPU.
///
/// Equivalent to `for i in 0..n { dst[i] = value; }` but executed in parallel
/// on the device. The scalar `value` is passed as a kernel parameter, so a
/// single PTX `ld.param` reads it once per thread — no extra allocation is needed.
///
/// # Arguments
///
/// * `handle` — BLAS handle bound to a CUDA context and stream.
/// * `dst` — Target device buffer. Must have at least `n` elements.
/// * `value` — The fill value (broadscasted to all `n` positions).
/// * `n` — Number of elements to fill. If `0`, returns immediately.
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if `dst.len() < n`.
/// Returns [`BlasError::PtxGeneration`] if kernel source generation fails.
/// Returns [`BlasError::LaunchFailed`] if module load or kernel launch fails.
pub fn fill<T: GpuFloat>(
    handle: &BlasHandle,
    dst: &mut DeviceBuffer<T>,
    value: T,
    n: u32,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    if dst.len() < n as usize {
        return Err(BlasError::BufferTooSmall {
            expected: n as usize,
            actual: dst.len(),
        });
    }

    let template = ElementwiseTemplate::new(ElementwiseOp::Fill, T::PTX_TYPE, handle.sm_version());
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("fill: {e}")))?;
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| BlasError::LaunchFailed(format!("fill module load: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("fill kernel lookup: {e}")))?;

    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);

    // Fill kernel signature: (dst_ptr: u64, value_bits: u64, n: u32)
    // The value is passed as raw bits (u64) matching the PTX scalar param convention
    // used throughout BLAS (mirrors the scale() scalar-passing pattern).
    let value_bits = value.to_bits_u64();
    let args = (dst.as_device_ptr(), value_bits, n);

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("fill launch: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_is_power_of_two() {
        assert!(BLOCK_SIZE.is_power_of_two());
        const { assert!(BLOCK_SIZE >= 32) };
    }

    #[test]
    fn ptx_template_generates_fill_f32() {
        let template = ElementwiseTemplate::new(
            ElementwiseOp::Fill,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template.generate().expect("fill PTX generation failed");
        assert!(ptx.contains("elementwise_fill_f32"), "wrong kernel name");
        assert!(
            ptx.contains("st.global.f32"),
            "must contain store instruction"
        );
        assert!(ptx.contains("ld.param.f32"), "must load scalar from param");
    }

    #[test]
    fn ptx_template_generates_fill_f64() {
        let template = ElementwiseTemplate::new(
            ElementwiseOp::Fill,
            oxicuda_ptx::ir::PtxType::F64,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template.generate().expect("fill f64 PTX generation failed");
        assert!(ptx.contains("elementwise_fill_f64"), "wrong kernel name");
        assert!(ptx.contains("st.global.f64"), "must contain f64 store");
    }
}
