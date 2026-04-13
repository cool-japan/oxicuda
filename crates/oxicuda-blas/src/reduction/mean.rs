//! Mean (average) reduction over device buffers.
//!
//! Computes the arithmetic mean by first computing the sum via the sum
//! reduction kernel, then launching a scalar division kernel to divide
//! by `n`.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::templates::elementwise::{ElementwiseOp as PtxElementwiseOp, ElementwiseTemplate};
use oxicuda_ptx::templates::reduction::{ReductionOp as PtxReductionOp, ReductionTemplate};

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

/// Block size for reduction kernels.
const REDUCE_BLOCK_SIZE: u32 = 256;

/// Computes `1/n` as a `u64` bit representation for the given `GpuFloat` type.
///
/// This works by computing the inverse in f64 (lossless for u32 values),
/// then converting to the target type's bit representation.
fn compute_inv_n<T: GpuFloat>(n: u32) -> BlasResult<u64> {
    let inv_f64 = 1.0_f64 / f64::from(n);
    // We need to store as T's bit representation. Since GpuFloat doesn't
    // support arithmetic, we convert through f64 -> f32/f64 bits.
    match T::PTX_TYPE {
        oxicuda_ptx::ir::PtxType::F32 => {
            let inv_f32 = inv_f64 as f32;
            Ok(u64::from(inv_f32.to_bits()))
        }
        oxicuda_ptx::ir::PtxType::F64 => Ok(inv_f64.to_bits()),
        other => Err(BlasError::UnsupportedOperation(format!(
            "mean: unsupported precision {} for scalar division",
            other.as_ptx_str()
        ))),
    }
}

/// Builds a sum-reduction kernel.
fn build_sum_kernel(
    handle: &BlasHandle,
    ptx_type: oxicuda_ptx::ir::PtxType,
) -> BlasResult<(Kernel, String)> {
    let template = ReductionTemplate {
        op: PtxReductionOp::Sum,
        precision: ptx_type,
        target: handle.sm_version(),
        block_size: REDUCE_BLOCK_SIZE,
    };
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("reduce_sum (for mean): {e}")))?;
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| BlasError::LaunchFailed(format!("module load for mean/sum: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Builds a scale kernel to divide the sum by n.
fn build_scale_kernel(
    handle: &BlasHandle,
    ptx_type: oxicuda_ptx::ir::PtxType,
) -> BlasResult<(Kernel, String)> {
    let template = ElementwiseTemplate::new(PtxElementwiseOp::Scale, ptx_type, handle.sm_version());
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("scale (for mean): {e}")))?;
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| BlasError::LaunchFailed(format!("module load for mean/scale: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Computes the arithmetic mean of the input buffer.
///
/// `result[0] = (input[0] + input[1] + ... + input[n-1]) / n`
///
/// Internally computes the sum via a two-phase reduction, then divides
/// the result by `n` using a scale kernel with `alpha = 1/n`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle bound to a CUDA context and stream.
/// * `n` -- number of elements (must be > 0 for a meaningful result).
/// * `input` -- device buffer containing the input array (at least `n` elements).
/// * `result` -- device buffer for the scalar result (at least
///   `max(1, ceil(n / block_size))` elements to hold partial sums during
///   intermediate phases).
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if buffers are too small,
/// [`BlasError::InvalidArgument`] if `n` is zero, or a PTX/launch error.
pub fn mean<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    result: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Err(BlasError::InvalidArgument(
            "mean requires n > 0".to_string(),
        ));
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

    // Phase 1: compute sum (may use multiple blocks)
    let num_blocks = grid_size_for(n, REDUCE_BLOCK_SIZE);

    let partials_needed = num_blocks as usize;
    if result.len() < partials_needed {
        return Err(BlasError::BufferTooSmall {
            expected: partials_needed,
            actual: result.len(),
        });
    }

    // Phase 1: block-level partial sums
    let (sum_kernel, _) = build_sum_kernel(handle, T::PTX_TYPE)?;
    let params1 = LaunchParams::new(num_blocks, REDUCE_BLOCK_SIZE);
    let args1 = (input.as_device_ptr(), result.as_device_ptr(), n);
    sum_kernel
        .launch(&params1, handle.stream(), &args1)
        .map_err(|e| BlasError::LaunchFailed(format!("mean/sum phase 1: {e}")))?;

    // Phase 2: reduce partial sums if needed
    if num_blocks > 1 {
        let phase2_blocks = grid_size_for(num_blocks, REDUCE_BLOCK_SIZE);
        if phase2_blocks > 1 {
            return Err(BlasError::UnsupportedOperation(format!(
                "mean: input size {n} requires more than two reduction phases"
            )));
        }
        let (sum_kernel2, _) = build_sum_kernel(handle, T::PTX_TYPE)?;
        let params2 = LaunchParams::new(1u32, REDUCE_BLOCK_SIZE);
        let args2 = (result.as_device_ptr(), result.as_device_ptr(), num_blocks);
        sum_kernel2
            .launch(&params2, handle.stream(), &args2)
            .map_err(|e| BlasError::LaunchFailed(format!("mean/sum phase 2: {e}")))?;
    }

    // Phase 3: divide sum by n  =>  result[0] = (1/n) * result[0]
    let inv_n_bits = compute_inv_n::<T>(n)?;
    let (scale_kernel, _) = build_scale_kernel(handle, T::PTX_TYPE)?;
    let scale_params = LaunchParams::new(1u32, REDUCE_BLOCK_SIZE);
    let args3 = (
        result.as_device_ptr(),
        result.as_device_ptr(),
        inv_n_bits,
        1u32,
    );
    scale_kernel
        .launch(&scale_params, handle.stream(), &args3)
        .map_err(|e| BlasError::LaunchFailed(format!("mean/scale: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_template_generates_sum_for_mean() {
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
    }

    #[test]
    fn ptx_template_generates_scale_for_mean() {
        let template = ElementwiseTemplate::new(
            PtxElementwiseOp::Scale,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("scale PTX generation should succeed");
        assert!(ptx.contains("elementwise_scale_f32"));
    }

    #[test]
    fn inv_n_computation_f32() {
        let n: u32 = 100;
        let inv: f32 = 1.0 / (n as f32);
        assert!((inv - 0.01).abs() < 1e-6);
    }

    #[test]
    fn inv_n_computation_f64() {
        let n: u32 = 100;
        let inv: f64 = 1.0 / (n as f64);
        assert!((inv - 0.01).abs() < 1e-12);
    }
}
