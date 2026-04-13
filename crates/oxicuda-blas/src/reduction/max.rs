//! Parallel max reduction over device buffers.
//!
//! Uses the same two-phase block-level reduction strategy as [`super::sum`],
//! but with `max` as the combining operation and `-INF` as the identity.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::templates::reduction::{ReductionOp as PtxReductionOp, ReductionTemplate};

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

/// Block size for reduction kernels.
const REDUCE_BLOCK_SIZE: u32 = 256;

/// Builds a max-reduction kernel from the PTX template.
fn build_max_kernel(
    handle: &BlasHandle,
    ptx_type: oxicuda_ptx::ir::PtxType,
) -> BlasResult<(Kernel, String)> {
    let template = ReductionTemplate {
        op: PtxReductionOp::Max,
        precision: ptx_type,
        target: handle.sm_version(),
        block_size: REDUCE_BLOCK_SIZE,
    };
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("reduce_max: {e}")))?;
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| BlasError::LaunchFailed(format!("module load for reduce_max: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Computes the maximum element in the input buffer.
///
/// `result[0] = max(input[0], input[1], ..., input[n-1])`
///
/// Two-phase reduction: phase 1 produces one block-level max per block;
/// phase 2 reduces the partial maxima to a single scalar.
///
/// # Arguments
///
/// * `handle` -- BLAS handle bound to a CUDA context and stream.
/// * `n` -- number of elements to reduce.
/// * `input` -- device buffer containing the input array (at least `n` elements).
/// * `result` -- device buffer for the scalar result (at least 1 element; at
///   least `ceil(n / block_size)` elements for multi-block reductions).
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if buffers are too small, or a
/// PTX/launch error on generation or execution failure.
pub fn max<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    result: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }

    let n_usize = n as usize;
    if input.len() < n_usize {
        return Err(BlasError::BufferTooSmall {
            expected: n_usize,
            actual: input.len(),
        });
    }
    if result.is_empty() {
        return Err(BlasError::BufferTooSmall {
            expected: 1,
            actual: 0,
        });
    }

    let num_blocks = grid_size_for(n, REDUCE_BLOCK_SIZE);

    if num_blocks == 1 {
        let (kernel, _) = build_max_kernel(handle, T::PTX_TYPE)?;
        let params = LaunchParams::new(1u32, REDUCE_BLOCK_SIZE);
        let args = (input.as_device_ptr(), result.as_device_ptr(), n);
        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| BlasError::LaunchFailed(format!("max phase 1: {e}")))?;
    } else {
        let partials_needed = num_blocks as usize;
        if result.len() < partials_needed {
            return Err(BlasError::BufferTooSmall {
                expected: partials_needed,
                actual: result.len(),
            });
        }

        // Phase 1
        let (kernel1, _) = build_max_kernel(handle, T::PTX_TYPE)?;
        let params1 = LaunchParams::new(num_blocks, REDUCE_BLOCK_SIZE);
        let args1 = (input.as_device_ptr(), result.as_device_ptr(), n);
        kernel1
            .launch(&params1, handle.stream(), &args1)
            .map_err(|e| BlasError::LaunchFailed(format!("max phase 1: {e}")))?;

        // Phase 2
        let phase2_blocks = grid_size_for(num_blocks, REDUCE_BLOCK_SIZE);
        if phase2_blocks == 1 {
            let (kernel2, _) = build_max_kernel(handle, T::PTX_TYPE)?;
            let params2 = LaunchParams::new(1u32, REDUCE_BLOCK_SIZE);
            let args2 = (result.as_device_ptr(), result.as_device_ptr(), num_blocks);
            kernel2
                .launch(&params2, handle.stream(), &args2)
                .map_err(|e| BlasError::LaunchFailed(format!("max phase 2: {e}")))?;
        } else {
            return Err(BlasError::UnsupportedOperation(format!(
                "max: input size {n} requires more than two reduction phases"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_template_generates_max_f32() {
        let template = ReductionTemplate {
            op: PtxReductionOp::Max,
            precision: oxicuda_ptx::ir::PtxType::F32,
            target: oxicuda_ptx::arch::SmVersion::Sm80,
            block_size: 256,
        };
        let ptx = template
            .generate()
            .expect("max PTX generation should succeed");
        assert!(ptx.contains("reduce_max_f32_bs256"));
        assert!(ptx.contains("max.f32"));
    }
}
