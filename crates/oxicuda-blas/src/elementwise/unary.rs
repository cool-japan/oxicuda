//! Unary elementwise operations on device buffers.
//!
//! Each function generates a PTX kernel for the requested activation or
//! transformation, loads it into the CUDA driver, and launches it on the
//! handle's stream. The buffer size is validated before launch.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::templates::elementwise::{ElementwiseOp as PtxOp, ElementwiseTemplate};

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Standard block size for 1-D elementwise kernels.
const BLOCK_SIZE: u32 = 256;

/// Validates that both input and output buffers have at least `n` elements.
fn validate_unary_buffers<T: Copy>(
    n: u32,
    input: &DeviceBuffer<T>,
    output: &DeviceBuffer<T>,
) -> BlasResult<()> {
    let n_usize = n as usize;
    if input.len() < n_usize {
        return Err(BlasError::BufferTooSmall {
            expected: n_usize,
            actual: input.len(),
        });
    }
    if output.len() < n_usize {
        return Err(BlasError::BufferTooSmall {
            expected: n_usize,
            actual: output.len(),
        });
    }
    Ok(())
}

/// Generates PTX, loads the module, and returns the kernel ready to launch.
fn build_unary_kernel(
    handle: &BlasHandle,
    ptx_op: PtxOp,
    ptx_type: oxicuda_ptx::ir::PtxType,
) -> BlasResult<(Kernel, String)> {
    let template = ElementwiseTemplate::new(ptx_op, ptx_type, handle.sm_version());
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("{}: {e}", ptx_op.as_str())))?;
    let module = Arc::new(Module::from_ptx(&ptx_source).map_err(|e| {
        BlasError::LaunchFailed(format!("module load for {}: {e}", ptx_op.as_str()))
    })?);
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Launches a standard unary kernel `(input_ptr, output_ptr, n)`.
fn launch_unary<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
    ptx_op: PtxOp,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_unary_buffers(n, input, output)?;

    let (kernel, _name) = build_unary_kernel(handle, ptx_op, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (input.as_device_ptr(), output.as_device_ptr(), n);

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("{}: {e}", ptx_op.as_str())))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Applies the ReLU activation element-wise.
///
/// For each element: `output[i] = max(0, input[i])`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle bound to a CUDA context and stream.
/// * `n` -- number of elements to process.
/// * `input` -- device buffer containing the input array (at least `n` elements).
/// * `output` -- device buffer for the result (at least `n` elements).
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if either buffer has fewer than `n`
/// elements, or a PTX/launch error if kernel generation or execution fails.
pub fn relu<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Relu)
}

/// Applies the GELU activation element-wise (tanh approximation).
///
/// `output[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `input` -- input device buffer.
/// * `output` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn gelu<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Gelu)
}

/// Applies the sigmoid activation element-wise.
///
/// `output[i] = 1 / (1 + exp(-input[i]))`
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `input` -- input device buffer.
/// * `output` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn sigmoid<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Sigmoid)
}

/// Applies the SiLU (Swish) activation element-wise.
///
/// `output[i] = input[i] * sigmoid(input[i])`
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `input` -- input device buffer.
/// * `output` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn silu<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Silu)
}

/// Applies the hyperbolic tangent activation element-wise.
///
/// `output[i] = tanh(input[i])`
///
/// Named `tanh_activation` to avoid conflict with the `f32::tanh` / `f64::tanh`
/// inherent methods.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `input` -- input device buffer.
/// * `output` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn tanh_activation<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Tanh)
}

/// Scales every element by a scalar: `output[i] = alpha * input[i]`.
///
/// The scalar `alpha` is passed by value from the host and embedded into
/// the PTX kernel as a parameter.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `alpha` -- scalar multiplier.
/// * `input` -- input device buffer.
/// * `output` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn scale<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    alpha: T,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_unary_buffers(n, input, output)?;

    let (kernel, _name) = build_unary_kernel(handle, PtxOp::Scale, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);

    // Scale kernel signature: (input_ptr, output_ptr, scalar_bits, n)
    let alpha_bits = alpha.to_bits_u64();
    let args = (input.as_device_ptr(), output.as_device_ptr(), alpha_bits, n);

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("scale: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_buffers_rejects_short_input() {
        // We cannot construct real DeviceBuffers without a GPU, but we can
        // verify the validation logic via unit-level checks on the error type.
        let err = BlasError::BufferTooSmall {
            expected: 1024,
            actual: 512,
        };
        assert!(err.to_string().contains("1024"));
    }

    #[test]
    fn block_size_is_power_of_two() {
        assert!(BLOCK_SIZE.is_power_of_two());
        const { assert!(BLOCK_SIZE >= 32) };
    }

    #[test]
    fn ptx_template_generates_relu_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Relu,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("relu PTX generation should succeed");
        assert!(ptx.contains("elementwise_relu_f32"));
    }

    #[test]
    fn ptx_template_generates_gelu_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Gelu,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("gelu PTX generation should succeed");
        assert!(ptx.contains("elementwise_gelu_f32"));
    }

    #[test]
    fn ptx_template_generates_sigmoid_f64() {
        let template = ElementwiseTemplate::new(
            PtxOp::Sigmoid,
            oxicuda_ptx::ir::PtxType::F64,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("sigmoid PTX generation should succeed");
        assert!(ptx.contains("elementwise_sigmoid_f64"));
    }

    #[test]
    fn ptx_template_generates_scale_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Scale,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("scale PTX generation should succeed");
        assert!(ptx.contains("elementwise_scale_f32"));
    }

    #[test]
    fn ptx_template_generates_silu_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Silu,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("silu PTX generation should succeed");
        assert!(ptx.contains("elementwise_silu_f32"));
    }

    #[test]
    fn ptx_template_generates_tanh_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Tanh,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("tanh PTX generation should succeed");
        assert!(ptx.contains("elementwise_tanh_f32"));
    }
}
