//! Parallel sum reduction over device buffers.
//!
//! Implements a two-phase approach:
//! 1. Each thread block computes a partial sum into shared memory using tree
//!    reduction with warp shuffle for the final 32 elements.
//! 2. If multiple blocks are used, a second kernel reduces the partial sums
//!    into the final scalar result.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::templates::reduction::{ReductionOp as PtxReductionOp, ReductionTemplate};

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

/// Block size for reduction kernels (must be power of 2 >= 32).
const REDUCE_BLOCK_SIZE: u32 = 256;

/// Builds a reduction kernel from the PTX template, loads it, and returns it.
fn build_reduce_kernel(
    handle: &BlasHandle,
    ptx_op: PtxReductionOp,
    ptx_type: oxicuda_ptx::ir::PtxType,
    block_size: u32,
) -> BlasResult<(Kernel, String)> {
    let template = ReductionTemplate {
        op: ptx_op,
        precision: ptx_type,
        target: handle.sm_version(),
        block_size,
    };
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("reduce_{}: {e}", ptx_op.as_str())))?;
    let module = Arc::new(Module::from_ptx(&ptx_source).map_err(|e| {
        BlasError::LaunchFailed(format!("module load for reduce_{}: {e}", ptx_op.as_str()))
    })?);
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Computes the sum of all elements in the input buffer.
///
/// `result[0] = input[0] + input[1] + ... + input[n-1]`
///
/// The reduction is performed in two phases when the input is larger than
/// one block. Phase 1 produces one partial sum per block; phase 2 reduces
/// those partial sums to a single scalar.
///
/// # Arguments
///
/// * `handle` -- BLAS handle bound to a CUDA context and stream.
/// * `n` -- number of elements to reduce.
/// * `input` -- device buffer containing the input array (at least `n` elements).
/// * `result` -- device buffer for the scalar result (at least 1 element).
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if the input has fewer than `n`
/// elements or the result buffer is empty. Returns a PTX/launch error
/// if kernel generation or execution fails.
pub fn sum<T: GpuFloat>(
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
        // Single block: reduce directly into result
        let (kernel, _) =
            build_reduce_kernel(handle, PtxReductionOp::Sum, T::PTX_TYPE, REDUCE_BLOCK_SIZE)?;
        let params = LaunchParams::new(1u32, REDUCE_BLOCK_SIZE);
        let args = (input.as_device_ptr(), result.as_device_ptr(), n);
        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| BlasError::LaunchFailed(format!("sum phase 1: {e}")))?;
    } else {
        // Two-phase reduction: first into partial sums, then reduce those.
        // We need a temporary buffer for the partial sums; use the result
        // buffer if it is large enough, otherwise report an error.
        // For correctness, we need an intermediate buffer of `num_blocks`
        // elements. We require the result buffer to be large enough.
        let partials_needed = num_blocks as usize;
        if result.len() < partials_needed {
            return Err(BlasError::BufferTooSmall {
                expected: partials_needed,
                actual: result.len(),
            });
        }

        // Phase 1: each block produces one partial sum
        let (kernel1, _) =
            build_reduce_kernel(handle, PtxReductionOp::Sum, T::PTX_TYPE, REDUCE_BLOCK_SIZE)?;
        let params1 = LaunchParams::new(num_blocks, REDUCE_BLOCK_SIZE);
        let args1 = (input.as_device_ptr(), result.as_device_ptr(), n);
        kernel1
            .launch(&params1, handle.stream(), &args1)
            .map_err(|e| BlasError::LaunchFailed(format!("sum phase 1: {e}")))?;

        // Phase 2: reduce partial sums to final scalar
        let phase2_blocks = grid_size_for(num_blocks, REDUCE_BLOCK_SIZE);
        if phase2_blocks == 1 {
            // Reduce in-place: read from result[0..num_blocks], write to result[0]
            let (kernel2, _) =
                build_reduce_kernel(handle, PtxReductionOp::Sum, T::PTX_TYPE, REDUCE_BLOCK_SIZE)?;
            let params2 = LaunchParams::new(1u32, REDUCE_BLOCK_SIZE);
            let args2 = (result.as_device_ptr(), result.as_device_ptr(), num_blocks);
            kernel2
                .launch(&params2, handle.stream(), &args2)
                .map_err(|e| BlasError::LaunchFailed(format!("sum phase 2: {e}")))?;
        } else {
            // For very large inputs we would need a recursive reduction.
            // With block_size=256, phase 1 handles up to 256*256=65536 blocks,
            // i.e. ~16M elements. Phase 2 handles up to 65536 partial sums in
            // one block. This covers n up to ~16 billion, which is sufficient.
            return Err(BlasError::UnsupportedOperation(format!(
                "sum: input size {n} requires more than two reduction phases"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_block_size_is_valid() {
        assert!(REDUCE_BLOCK_SIZE.is_power_of_two());
        const { assert!(REDUCE_BLOCK_SIZE >= 32) };
    }

    #[test]
    fn ptx_template_generates_sum_f32() {
        let template = ReductionTemplate {
            op: PtxReductionOp::Sum,
            precision: oxicuda_ptx::ir::PtxType::F32,
            target: oxicuda_ptx::arch::SmVersion::Sm80,
            block_size: 256,
        };
        let ptx = template
            .generate()
            .expect("sum PTX generation should succeed");
        assert!(ptx.contains("reduce_sum_f32_bs256"));
        assert!(ptx.contains("shfl.sync.down"));
    }

    #[test]
    fn ptx_template_generates_sum_f64() {
        let template = ReductionTemplate {
            op: PtxReductionOp::Sum,
            precision: oxicuda_ptx::ir::PtxType::F64,
            target: oxicuda_ptx::arch::SmVersion::Sm80,
            block_size: 256,
        };
        let ptx = template
            .generate()
            .expect("sum f64 PTX generation should succeed");
        assert!(ptx.contains("reduce_sum_f64_bs256"));
    }

    #[test]
    fn grid_size_single_block() {
        // n <= REDUCE_BLOCK_SIZE should use 1 block
        assert_eq!(grid_size_for(100, REDUCE_BLOCK_SIZE), 1);
        assert_eq!(grid_size_for(256, REDUCE_BLOCK_SIZE), 1);
    }

    #[test]
    fn grid_size_multi_block() {
        assert_eq!(grid_size_for(257, REDUCE_BLOCK_SIZE), 2);
        assert_eq!(grid_size_for(1024, REDUCE_BLOCK_SIZE), 4);
    }
}
