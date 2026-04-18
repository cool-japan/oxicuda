//! Per-axis reduction over N-dimensional tensors.
//!
//! Provides [`reduce_axis`], which reduces a single axis of a contiguous
//! N-dimensional tensor stored in row-major order. The tensor is viewed as
//! `[outer, axis_len, inner]`, and one GPU thread block handles each
//! `(outer_idx, inner_idx)` output pair.
//!
//! # Supported operations
//!
//! All [`ReductionOp`] variants are supported. For [`ReductionOp::Mean`], the
//! kernel receives `inv_axis_len = 1 / axis_len` as a `f32` parameter and
//! multiplies the accumulated sum before writing the output.
//!
//! # Example (PTX-generation-only, no GPU required)
//!
//! ```
//! use oxicuda_ptx::templates::reduction::{PerAxisReductionTemplate, ReductionOp as PtxReductionOp};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = PerAxisReductionTemplate {
//!     op: PtxReductionOp::Sum,
//!     precision: PtxType::F32,
//!     target: SmVersion::Sm80,
//!     block_size: 256,
//! };
//! let ptx = template.generate().expect("PTX generation must succeed");
//! assert!(ptx.contains("reduce_axis_sum_f32_bs256"));
//! ```

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::templates::reduction::{PerAxisReductionTemplate, ReductionOp as PtxReductionOp};

use super::ops::ReductionOp;
use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

/// Default block size for per-axis reduction kernels (must be a power of 2, >= 32).
const AXIS_BLOCK_SIZE: u32 = 256;

/// Builds a per-axis reduction kernel from the PTX template, loads it, and returns it.
fn build_axis_kernel(
    handle: &BlasHandle,
    ptx_op: PtxReductionOp,
    ptx_type: oxicuda_ptx::ir::PtxType,
) -> BlasResult<(Kernel, String)> {
    let template = PerAxisReductionTemplate {
        op: ptx_op,
        precision: ptx_type,
        target: handle.sm_version(),
        block_size: AXIS_BLOCK_SIZE,
    };
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("reduce_axis_{}: {e}", ptx_op.as_str())))?;
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| BlasError::LaunchFailed(format!("module load for {kernel_name}: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Reduce over a single axis of an N-D tensor (row-major, contiguous).
///
/// The input is viewed as `[outer, axis_len, inner]`. Each output element
/// at `(outer_idx, inner_idx)` gets the reduction of
/// `input[outer_idx, 0..axis_len, inner_idx]`.
///
/// For [`ReductionOp::Mean`], the sum is divided by `axis_len` in-kernel
/// via `inv_axis_len = 1.0 / axis_len`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle bound to a CUDA context and stream.
/// * `op` -- reduction operation.
/// * `outer` -- product of dimensions before the reduction axis.
/// * `axis_len` -- size of the reduction axis.
/// * `inner` -- product of dimensions after the reduction axis.
/// * `input` -- device buffer; must have at least `outer * axis_len * inner` elements.
/// * `output` -- device buffer; must have at least `outer * inner` elements.
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if the buffers are too small.
/// Returns a PTX/launch error if kernel generation or execution fails.
/// Returns [`BlasError::InvalidArgument`] if dimension products overflow.
pub fn reduce_axis<T: GpuFloat>(
    handle: &BlasHandle,
    op: ReductionOp,
    outer: u32,
    axis_len: u32,
    inner: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if axis_len == 0 {
        return Ok(());
    }

    let input_needed = (outer as usize)
        .checked_mul(axis_len as usize)
        .and_then(|x| x.checked_mul(inner as usize))
        .ok_or_else(|| {
            BlasError::InvalidArgument(
                "reduce_axis: outer * axis_len * inner overflows usize".into(),
            )
        })?;
    let output_needed = (outer as usize)
        .checked_mul(inner as usize)
        .ok_or_else(|| {
            BlasError::InvalidArgument("reduce_axis: outer * inner overflows usize".into())
        })?;

    if input.len() < input_needed {
        return Err(BlasError::BufferTooSmall {
            expected: input_needed,
            actual: input.len(),
        });
    }
    if output.len() < output_needed {
        return Err(BlasError::BufferTooSmall {
            expected: output_needed,
            actual: output.len(),
        });
    }

    let ptx_op = op.to_ptx_op();
    let (kernel, _) = build_axis_kernel(handle, ptx_op, T::PTX_TYPE)?;

    let grid = outer.checked_mul(inner).ok_or_else(|| {
        BlasError::InvalidArgument("reduce_axis: outer * inner grid overflows u32".into())
    })?;
    let params = LaunchParams::new(grid, AXIS_BLOCK_SIZE);

    if op == ReductionOp::Mean {
        let inv_axis_len = 1.0_f32 / axis_len as f32;
        let args = (
            input.as_device_ptr(),
            output.as_device_ptr(),
            axis_len,
            inner,
            inv_axis_len,
        );
        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| BlasError::LaunchFailed(format!("reduce_axis mean: {e}")))?;
    } else {
        let args = (
            input.as_device_ptr(),
            output.as_device_ptr(),
            axis_len,
            inner,
        );
        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| BlasError::LaunchFailed(format!("reduce_axis {}: {e}", op.as_str())))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;
    use oxicuda_ptx::ir::PtxType;

    #[test]
    fn axis_block_size_is_valid() {
        assert!(AXIS_BLOCK_SIZE.is_power_of_two());
        const { assert!(AXIS_BLOCK_SIZE >= 32) };
    }

    #[test]
    fn ptx_generates_reduce_axis_sum_f32() {
        let template = PerAxisReductionTemplate {
            op: PtxReductionOp::Sum,
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            block_size: 256,
        };
        let ptx = template.generate().expect("generate must not fail");
        assert!(ptx.contains("reduce_axis_sum_f32_bs256"));
        assert!(ptx.contains("param_axis_len"));
        assert!(ptx.contains("param_inner"));
    }

    #[test]
    fn ptx_generates_reduce_axis_mean_f32() {
        let template = PerAxisReductionTemplate {
            op: PtxReductionOp::Mean,
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            block_size: 256,
        };
        let ptx = template.generate().expect("generate must not fail");
        assert!(ptx.contains("param_inv_axis_len"));
    }

    #[test]
    fn overflow_protection_outer_axis_inner() {
        // u32::MAX * u32::MAX * u32::MAX must overflow usize (64-bit)
        let big: u32 = u32::MAX;
        let result = (big as usize)
            .checked_mul(big as usize)
            .and_then(|x| x.checked_mul(big as usize));
        assert!(
            result.is_none(),
            "3-way u32::MAX multiply must overflow u64"
        );
    }
}
