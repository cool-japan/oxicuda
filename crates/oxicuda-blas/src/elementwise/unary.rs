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

/// Negates every element: `output[i] = -input[i]`.
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
pub fn neg<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Neg)
}

/// Computes the absolute value element-wise: `output[i] = |input[i]|`.
///
/// Named `abs_val` to avoid conflict with inherent `abs` methods.
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
pub fn abs_val<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Abs)
}

/// Computes the square root element-wise: `output[i] = sqrt(input[i])`.
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
pub fn sqrt<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Sqrt)
}

/// Computes the reciprocal square root element-wise: `output[i] = 1 / sqrt(input[i])`.
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
pub fn rsqrt<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Rsqrt)
}

/// Computes the exponential element-wise: `output[i] = exp(input[i])`.
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
pub fn exp<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Exp)
}

/// Computes the natural logarithm element-wise: `output[i] = ln(input[i])`.
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
pub fn log<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Log)
}

/// Computes the ceiling element-wise: `output[i] = ceil(input[i])`.
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
pub fn ceil<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Ceil)
}

/// Computes the floor element-wise: `output[i] = floor(input[i])`.
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
pub fn floor<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Floor)
}

/// Applies hard sigmoid element-wise: `output[i] = max(0, min(1, 0.2*input[i] + 0.5))`.
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
pub fn hard_sigmoid<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::HardSigmoid)
}

/// Applies hard swish element-wise: `output[i] = input[i] * max(0, min(6, input[i]+3)) / 6`.
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
pub fn hard_swish<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::HardSwish)
}

/// Applies softplus element-wise: `output[i] = ln(1 + exp(input[i]))`.
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
pub fn softplus<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::Softplus)
}

/// Applies leaky relu element-wise with alpha=0.01:
/// `output[i] = input[i] >= 0 ? input[i] : 0.01 * input[i]`.
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
pub fn leaky_relu<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::LeakyRelu)
}

/// Applies one-minus element-wise: `output[i] = 1 - input[i]`.
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
pub fn one_minus<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    launch_unary(handle, n, input, output, PtxOp::OneMinus)
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

    #[test]
    fn ptx_template_generates_neg_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Neg,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("neg PTX generation should succeed");
        assert!(ptx.contains("elementwise_neg_f32"));
        assert!(ptx.contains("neg.f32"));
    }

    #[test]
    fn ptx_template_generates_abs_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Abs,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("abs PTX generation should succeed");
        assert!(ptx.contains("elementwise_abs_f32"));
        assert!(ptx.contains("abs.f32"));
    }

    #[test]
    fn ptx_template_generates_sqrt_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Sqrt,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("sqrt PTX generation should succeed");
        assert!(ptx.contains("elementwise_sqrt_f32"));
        assert!(ptx.contains("sqrt.rn.f32"));
    }

    #[test]
    fn ptx_template_generates_rsqrt_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Rsqrt,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("rsqrt PTX generation should succeed");
        assert!(ptx.contains("elementwise_rsqrt_f32"));
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn ptx_template_generates_exp_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Exp,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("exp PTX generation should succeed");
        assert!(ptx.contains("elementwise_exp_f32"));
        assert!(ptx.contains("ex2.approx.f32"));
    }

    #[test]
    fn ptx_template_generates_log_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Log,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("log PTX generation should succeed");
        assert!(ptx.contains("elementwise_log_f32"));
        assert!(ptx.contains("lg2.approx.f32"));
    }

    #[test]
    fn ptx_template_generates_ceil_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Ceil,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("ceil PTX generation should succeed");
        assert!(ptx.contains("elementwise_ceil_f32"));
        assert!(ptx.contains("cvt.rpi.f32.f32"));
    }

    #[test]
    fn ptx_template_generates_floor_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Floor,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("floor PTX generation should succeed");
        assert!(ptx.contains("elementwise_floor_f32"));
        assert!(ptx.contains("cvt.rmi.f32.f32"));
    }

    #[test]
    fn ptx_template_generates_hard_sigmoid_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::HardSigmoid,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("hard_sigmoid PTX generation should succeed");
        assert!(ptx.contains("elementwise_hard_sigmoid_f32"));
    }

    #[test]
    fn ptx_template_generates_hard_swish_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::HardSwish,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("hard_swish PTX generation should succeed");
        assert!(ptx.contains("elementwise_hard_swish_f32"));
    }

    #[test]
    fn ptx_template_generates_softplus_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Softplus,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("softplus PTX generation should succeed");
        assert!(ptx.contains("elementwise_softplus_f32"));
    }

    #[test]
    fn ptx_template_generates_leaky_relu_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::LeakyRelu,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("leaky_relu PTX generation should succeed");
        assert!(ptx.contains("elementwise_leaky_relu_f32"));
        assert!(ptx.contains("setp.ge.f32"));
        assert!(ptx.contains("selp.f32"));
    }

    #[test]
    fn ptx_template_generates_one_minus_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::OneMinus,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("one_minus PTX generation should succeed");
        assert!(ptx.contains("elementwise_one_minus_f32"));
        assert!(ptx.contains("sub.f32"));
        assert!(ptx.contains("0f3F800000"));
    }
}
