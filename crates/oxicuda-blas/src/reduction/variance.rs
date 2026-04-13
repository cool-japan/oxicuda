//! Variance reduction over device buffers.
//!
//! Computes the population variance using a two-pass approach:
//! 1. Compute the mean via sum reduction + scale.
//! 2. Compute the mean of squared deviations `(x[i] - mean)^2` by
//!    launching an elementwise kernel that subtracts the mean and squares,
//!    followed by another sum reduction and division by `n`.
//!
//! This module uses the existing reduction and elementwise PTX templates
//! rather than generating a custom fused kernel.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;
use oxicuda_ptx::templates::elementwise::{ElementwiseOp as PtxElementwiseOp, ElementwiseTemplate};
use oxicuda_ptx::templates::reduction::{ReductionOp as PtxReductionOp, ReductionTemplate};

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

/// Block size for reduction and elementwise kernels.
const BLOCK_SIZE: u32 = 256;

/// Computes `1/n` as a `u64` bit representation for the given `GpuFloat` type.
fn compute_inv_n<T: GpuFloat>(n: u32) -> BlasResult<u64> {
    let inv_f64 = 1.0_f64 / f64::from(n);
    match T::PTX_TYPE {
        PtxType::F32 => {
            let inv_f32 = inv_f64 as f32;
            Ok(u64::from(inv_f32.to_bits()))
        }
        PtxType::F64 => Ok(inv_f64.to_bits()),
        other => Err(BlasError::UnsupportedOperation(format!(
            "variance: unsupported precision {} for scalar division",
            other.as_ptx_str()
        ))),
    }
}

/// Builds a sum-reduction kernel.
fn build_sum_kernel(handle: &BlasHandle, ptx_type: PtxType) -> BlasResult<(Kernel, String)> {
    let template = ReductionTemplate {
        op: PtxReductionOp::Sum,
        precision: ptx_type,
        target: handle.sm_version(),
        block_size: BLOCK_SIZE,
    };
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("reduce_sum (for variance): {e}")))?;
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| BlasError::LaunchFailed(format!("module load for variance/sum: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Builds a scale kernel.
fn build_scale_kernel(handle: &BlasHandle, ptx_type: PtxType) -> BlasResult<(Kernel, String)> {
    let template = ElementwiseTemplate::new(PtxElementwiseOp::Scale, ptx_type, handle.sm_version());
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("scale (for variance): {e}")))?;
    let module =
        Arc::new(Module::from_ptx(&ptx_source).map_err(|e| {
            BlasError::LaunchFailed(format!("module load for variance/scale: {e}"))
        })?);
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Generates a PTX kernel that computes `(x[i] - scalar)^2` for each element.
///
/// Kernel signature: `(input_ptr: u64, output_ptr: u64, mean_ptr: u64, n: u32)`
/// The mean is loaded from a device pointer (1 element) so the variance can
/// be computed without a host-device synchronization.
fn generate_squared_diff_ptx(sm: SmVersion, ptx_type: PtxType) -> BlasResult<(String, String)> {
    let ty = ptx_type.as_ptx_str();
    let byte_size = ptx_type.size_bytes();
    let type_label = ptx_type.as_ptx_str().trim_start_matches('.');
    let kernel_name = format!("squared_diff_{type_label}");

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("mean_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            let tid = b.global_thread_id_x();
            let tid_name = tid.to_string();
            let n_reg = b.load_param_u32("n");
            b.if_lt_u32(tid, n_reg, move |b| {
                let input_ptr = b.load_param_u64("input_ptr");
                let output_ptr = b.load_param_u64("output_ptr");
                let mean_ptr = b.load_param_u64("mean_ptr");

                // Load mean from device memory (single element)
                b.raw_ptx(&format!("ld.global{ty} %f_mean, [{mean_ptr}];"));

                // Compute byte offset and load input element
                b.raw_ptx(&format!(
                    "cvt.u64.u32 %rd_off, {tid_name};\n    \
                     mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                     add.u64 %rd_in, {input_ptr}, %rd_off;\n    \
                     add.u64 %rd_out, {output_ptr}, %rd_off;"
                ));

                b.raw_ptx(&format!(
                    "ld.global{ty} %f_x, [%rd_in];\n    \
                     sub{ty} %f_diff, %f_x, %f_mean;\n    \
                     mul{ty} %f_sq, %f_diff, %f_diff;\n    \
                     st.global{ty} [%rd_out], %f_sq;"
                ));
            });
            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(format!("squared_diff: {e}")))?;

    Ok((ptx, kernel_name))
}

/// Computes the population variance of the input buffer.
///
/// `result[0] = sum((input[i] - mean)^2) / n`
///
/// Uses a two-pass algorithm:
/// 1. Compute the mean via sum reduction + scale by `1/n`.
/// 2. Compute `(input[i] - mean)^2` for each element.
/// 3. Sum the squared deviations and divide by `n`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle bound to a CUDA context and stream.
/// * `n` -- number of elements (must be > 0).
/// * `input` -- device buffer containing the input array (at least `n` elements).
/// * `work` -- temporary device buffer for intermediate results (at least `n` elements).
/// * `result` -- device buffer for the scalar result (at least
///   `max(1, ceil(n / block_size))` elements).
///
/// # Errors
///
/// Returns [`BlasError::InvalidArgument`] if `n` is zero,
/// [`BlasError::BufferTooSmall`] if any buffer is too small, or a
/// PTX/launch error on generation or execution failure.
pub fn variance<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    work: &mut DeviceBuffer<T>,
    result: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Err(BlasError::InvalidArgument(
            "variance requires n > 0".to_string(),
        ));
    }

    let n_usize = n as usize;
    if input.len() < n_usize {
        return Err(BlasError::BufferTooSmall {
            expected: n_usize,
            actual: input.len(),
        });
    }
    if work.len() < n_usize {
        return Err(BlasError::BufferTooSmall {
            expected: n_usize,
            actual: work.len(),
        });
    }
    let num_blocks = grid_size_for(n, BLOCK_SIZE);
    let partials_needed = num_blocks as usize;
    if result.len() < partials_needed {
        return Err(BlasError::BufferTooSmall {
            expected: partials_needed,
            actual: result.len(),
        });
    }

    // -----------------------------------------------------------------------
    // Pass 1: compute mean => result[0]
    // -----------------------------------------------------------------------

    // Phase 1a: sum reduction
    let (sum_kernel, _) = build_sum_kernel(handle, T::PTX_TYPE)?;
    let params_sum = LaunchParams::new(num_blocks, BLOCK_SIZE);
    let args_sum = (input.as_device_ptr(), result.as_device_ptr(), n);
    sum_kernel
        .launch(&params_sum, handle.stream(), &args_sum)
        .map_err(|e| BlasError::LaunchFailed(format!("variance/sum phase 1: {e}")))?;

    // Phase 1b: reduce partial sums if multi-block
    if num_blocks > 1 {
        let phase2_blocks = grid_size_for(num_blocks, BLOCK_SIZE);
        if phase2_blocks > 1 {
            return Err(BlasError::UnsupportedOperation(format!(
                "variance: input size {n} requires more than two reduction phases"
            )));
        }
        let (sum_kernel2, _) = build_sum_kernel(handle, T::PTX_TYPE)?;
        let params2 = LaunchParams::new(1u32, BLOCK_SIZE);
        let args2 = (result.as_device_ptr(), result.as_device_ptr(), num_blocks);
        sum_kernel2
            .launch(&params2, handle.stream(), &args2)
            .map_err(|e| BlasError::LaunchFailed(format!("variance/sum phase 2: {e}")))?;
    }

    // Phase 1c: divide by n => mean in result[0]
    let inv_n_bits = compute_inv_n::<T>(n)?;
    let (scale_kernel, _) = build_scale_kernel(handle, T::PTX_TYPE)?;
    let scale_params = LaunchParams::new(1u32, BLOCK_SIZE);
    let args_scale = (
        result.as_device_ptr(),
        result.as_device_ptr(),
        inv_n_bits,
        1u32,
    );
    scale_kernel
        .launch(&scale_params, handle.stream(), &args_scale)
        .map_err(|e| BlasError::LaunchFailed(format!("variance/mean scale: {e}")))?;

    // -----------------------------------------------------------------------
    // Pass 2: compute squared deviations => work[i] = (input[i] - mean)^2
    // -----------------------------------------------------------------------
    let (sq_diff_ptx, sq_diff_name) = generate_squared_diff_ptx(handle.sm_version(), T::PTX_TYPE)?;
    let sq_diff_module = Arc::new(
        Module::from_ptx(&sq_diff_ptx)
            .map_err(|e| BlasError::LaunchFailed(format!("module load for squared_diff: {e}")))?,
    );
    let sq_diff_kernel = Kernel::from_module(sq_diff_module, &sq_diff_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {sq_diff_name}: {e}")))?;

    let grid = grid_size_for(n, BLOCK_SIZE);
    let sq_params = LaunchParams::new(grid, BLOCK_SIZE);
    // mean_ptr = result[0]
    let sq_args = (
        input.as_device_ptr(),
        work.as_device_ptr(),
        result.as_device_ptr(),
        n,
    );
    sq_diff_kernel
        .launch(&sq_params, handle.stream(), &sq_args)
        .map_err(|e| BlasError::LaunchFailed(format!("variance/squared_diff: {e}")))?;

    // -----------------------------------------------------------------------
    // Pass 3: sum squared deviations and divide by n
    // -----------------------------------------------------------------------
    let (sum_kernel3, _) = build_sum_kernel(handle, T::PTX_TYPE)?;
    let params3 = LaunchParams::new(num_blocks, BLOCK_SIZE);
    let args3 = (work.as_device_ptr(), result.as_device_ptr(), n);
    sum_kernel3
        .launch(&params3, handle.stream(), &args3)
        .map_err(|e| BlasError::LaunchFailed(format!("variance/sum2 phase 1: {e}")))?;

    if num_blocks > 1 {
        let phase2_blocks = grid_size_for(num_blocks, BLOCK_SIZE);
        if phase2_blocks > 1 {
            return Err(BlasError::UnsupportedOperation(format!(
                "variance: input size {n} requires more than two reduction phases"
            )));
        }
        let (sum_kernel4, _) = build_sum_kernel(handle, T::PTX_TYPE)?;
        let params4 = LaunchParams::new(1u32, BLOCK_SIZE);
        let args4 = (result.as_device_ptr(), result.as_device_ptr(), num_blocks);
        sum_kernel4
            .launch(&params4, handle.stream(), &args4)
            .map_err(|e| BlasError::LaunchFailed(format!("variance/sum2 phase 2: {e}")))?;
    }

    // Divide by n => variance
    let (scale_kernel2, _) = build_scale_kernel(handle, T::PTX_TYPE)?;
    let args_final = (
        result.as_device_ptr(),
        result.as_device_ptr(),
        inv_n_bits,
        1u32,
    );
    scale_kernel2
        .launch(&scale_params, handle.stream(), &args_final)
        .map_err(|e| BlasError::LaunchFailed(format!("variance/final scale: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_diff_ptx_generates_f32() {
        let (ptx, name) = generate_squared_diff_ptx(SmVersion::Sm80, PtxType::F32)
            .expect("squared_diff PTX should generate");
        assert_eq!(name, "squared_diff_f32");
        assert!(ptx.contains("squared_diff_f32"));
        assert!(ptx.contains("sub.f32"));
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn squared_diff_ptx_generates_f64() {
        let (ptx, name) = generate_squared_diff_ptx(SmVersion::Sm80, PtxType::F64)
            .expect("squared_diff PTX should generate");
        assert_eq!(name, "squared_diff_f64");
        assert!(ptx.contains("sub.f64"));
    }
}
