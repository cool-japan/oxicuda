//! Public convolution API.
//!
//! High-level functions for convolution operations that automatically select
//! the optimal algorithm and dispatch to the appropriate kernel engine.
//! These are the primary entry points for end users of the DNN crate.

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::types::{Activation, ConvAlgorithm, ConvolutionDescriptor, TensorDesc, TensorDescMut};

use super::descriptor::ConvProblem;
use super::dgrad::implicit_gemm::DgradImplicitGemm;
use super::fprop::direct::{Conv1x1, DepthwiseConv};
use super::fprop::im2col_gemm::Im2colGemmConv;
use super::fprop::implicit_gemm::ImplicitGemmConv;
use super::fprop::winograd::WinogradConv;
use super::fused::{FusedBnParams, FusedConvBnAct};
use super::wgrad::implicit_gemm::WgradImplicitGemm;

// ---------------------------------------------------------------------------
// conv_forward
// ---------------------------------------------------------------------------

/// Performs a forward convolution.
///
/// Automatically selects the optimal algorithm based on problem dimensions
/// and target architecture. The caller may optionally provide a workspace
/// buffer for algorithms that require one (im2col, Winograd, FFT).
///
/// # Arguments
///
/// * `handle` — DNN handle providing CUDA context, stream, and BLAS access.
/// * `input` — Input tensor descriptor `[N, C, H, W]` or `[N, C, D, H, W]`.
/// * `filter` — Filter tensor descriptor `[K, C/g, R, S]`.
/// * `output` — Mutable output tensor descriptor `[N, K, P, Q]`.
/// * `conv_desc` — Convolution parameters (padding, stride, dilation, groups).
/// * `workspace` — Optional workspace buffer for algorithms that need it.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if tensor shapes are inconsistent.
/// Returns [`DnnError::WorkspaceRequired`] if the selected algorithm needs
/// workspace but none (or too small) was provided.
/// Returns other errors from PTX generation, module loading, or kernel launch.
///
/// # Example
///
/// ```rust,no_run
/// use oxicuda_dnn::conv::api::conv_forward;
/// use oxicuda_dnn::types::{ConvolutionDescriptor, TensorDesc, TensorDescMut};
/// use oxicuda_dnn::handle::DnnHandle;
/// # fn example(handle: &DnnHandle,
/// #            input: &TensorDesc<f32>,
/// #            filter: &TensorDesc<f32>,
/// #            output: &mut TensorDescMut<f32>) -> Result<(), oxicuda_dnn::error::DnnError> {
/// let conv_desc = ConvolutionDescriptor::conv2d(1, 1, 1, 1, 1, 1, 1)?;
/// conv_forward(handle, input, filter, output, &conv_desc, None)?;
/// # Ok(())
/// # }
/// ```
pub fn conv_forward<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    filter: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
    conv_desc: &ConvolutionDescriptor,
    workspace: Option<&mut DeviceBuffer<u8>>,
) -> DnnResult<()> {
    let problem = ConvProblem::from_descriptors(input, filter, output, conv_desc)?;
    problem.validate()?;

    let algo = problem.select_algorithm(handle.sm_version());

    match algo {
        ConvAlgorithm::Direct if problem.is_1x1() => {
            let engine = Conv1x1::new(problem, handle.sm_version())?;
            engine.execute(handle, input, filter, output)
        }
        ConvAlgorithm::Direct => {
            // Depthwise convolution
            let engine = DepthwiseConv::new(problem, handle.sm_version())?;
            engine.execute(handle, input, filter, output)
        }
        ConvAlgorithm::ImplicitGemm => {
            let engine = ImplicitGemmConv::new(problem, handle.sm_version());
            engine.execute(handle, input, filter, output)
        }
        ConvAlgorithm::Im2colGemm => {
            let ws = workspace.ok_or(DnnError::WorkspaceRequired(
                Im2colGemmConv::new(problem.clone(), handle.sm_version())
                    .workspace_bytes()
                    .unwrap_or(0),
            ))?;
            let engine = Im2colGemmConv::new(problem, handle.sm_version());
            engine.execute(handle, input, filter, output, ws)
        }
        ConvAlgorithm::Winograd => {
            let ws = workspace.ok_or_else(|| {
                DnnError::WorkspaceRequired(
                    WinogradConv::new(problem.clone(), handle.sm_version())
                        .and_then(|w| w.workspace_bytes())
                        .unwrap_or(0),
                )
            })?;
            let engine = WinogradConv::new(problem, handle.sm_version())?;
            engine.execute(handle, input, filter, output, ws)
        }
        ConvAlgorithm::FftConv => {
            // FFT-based convolution requires Vol.5 (oxicuda-fft).
            Err(DnnError::UnsupportedOperation(
                "FFT-based convolution not yet implemented (requires Vol.5 oxicuda-fft)".into(),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// conv_backward_data
// ---------------------------------------------------------------------------

/// Computes the gradient of the loss with respect to the input tensor.
///
/// This is the backward data pass (dgrad) used during training. It
/// implements the transposed convolution: for each input position, it
/// accumulates contributions from all overlapping filter positions in
/// the gradient output.
///
/// # Arguments
///
/// * `handle` — DNN handle.
/// * `filter` — Filter weights `[K, C/g, R, S]`.
/// * `grad_output` — Gradient from the layer above `[N, K, P, Q]`.
/// * `grad_input` — Output: gradient w.r.t. the input `[N, C, H, W]`.
/// * `conv_desc` — Convolution parameters.
/// * `_workspace` — Reserved for future algorithms.
///
/// # Errors
///
/// Same as [`conv_forward`].
pub fn conv_backward_data<T: GpuFloat>(
    handle: &DnnHandle,
    filter: &TensorDesc<T>,
    grad_output: &TensorDesc<T>,
    grad_input: &mut TensorDescMut<T>,
    conv_desc: &ConvolutionDescriptor,
    _workspace: Option<&mut DeviceBuffer<u8>>,
) -> DnnResult<()> {
    // Construct the forward problem from the perspective of the dgrad.
    // grad_input has the shape of the forward input.
    let problem = build_dgrad_problem::<T>(filter, grad_output, grad_input, conv_desc)?;
    problem.validate()?;

    let engine = DgradImplicitGemm::new(problem, handle.sm_version());
    engine.execute(handle, grad_output, filter, grad_input)
}

// ---------------------------------------------------------------------------
// conv_backward_filter
// ---------------------------------------------------------------------------

/// Computes the gradient of the loss with respect to the filter weights.
///
/// This is the backward filter pass (wgrad) used during training. It
/// cross-correlates the input tensor with the gradient output to produce
/// the filter gradient.
///
/// # Arguments
///
/// * `handle` — DNN handle.
/// * `input` — Forward input tensor `[N, C, H, W]`.
/// * `grad_output` — Gradient from the layer above `[N, K, P, Q]`.
/// * `grad_filter` — Output: gradient w.r.t. the filter `[K, C/g, R, S]`.
/// * `conv_desc` — Convolution parameters.
/// * `_workspace` — Reserved for future algorithms.
///
/// # Errors
///
/// Same as [`conv_forward`].
pub fn conv_backward_filter<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    grad_output: &TensorDesc<T>,
    grad_filter: &mut TensorDescMut<T>,
    conv_desc: &ConvolutionDescriptor,
    _workspace: Option<&mut DeviceBuffer<u8>>,
) -> DnnResult<()> {
    let problem = build_wgrad_problem::<T>(input, grad_output, grad_filter, conv_desc)?;
    problem.validate()?;

    let engine = WgradImplicitGemm::new(problem, handle.sm_version());
    engine.execute(handle, input, grad_output, grad_filter)
}

// ---------------------------------------------------------------------------
// conv_bn_relu
// ---------------------------------------------------------------------------

/// Performs fused convolution + batch normalisation + ReLU.
///
/// This fusion eliminates two extra memory round-trips compared to running
/// convolution, BN, and ReLU as separate operations. The BN parameters
/// must be pre-computed into fused scale/bias form.
///
/// # Arguments
///
/// * `handle` — DNN handle.
/// * `input` — Input tensor.
/// * `filter` — Filter weights.
/// * `output` — Mutable output tensor.
/// * `conv_desc` — Convolution parameters.
/// * `bn_params` — Fused BN scale and bias (device pointers).
/// * `activation` — Activation function to apply after BN.
///
/// # Errors
///
/// Same as [`conv_forward`].
pub fn conv_bn_relu<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    filter: &TensorDesc<T>,
    output: &mut TensorDescMut<T>,
    conv_desc: &ConvolutionDescriptor,
    bn_params: &FusedBnParams,
    activation: Activation,
) -> DnnResult<()> {
    let problem = ConvProblem::from_descriptors(input, filter, output, conv_desc)?;
    problem.validate()?;

    if bn_params.channels != problem.out_channels {
        return Err(DnnError::InvalidArgument(format!(
            "BN channels ({}) != out_channels ({})",
            bn_params.channels, problem.out_channels
        )));
    }

    let engine = FusedConvBnAct::new(problem, activation, handle.sm_version());
    engine.execute(handle, input, filter, output, bn_params)
}

// ---------------------------------------------------------------------------
// Helper: build ConvProblem for dgrad
// ---------------------------------------------------------------------------

/// Constructs a [`ConvProblem`] for the backward data pass.
///
/// The "input" for dgrad is the forward input shape (grad_input),
/// and the "output" is the forward output shape (grad_output).
fn build_dgrad_problem<T: GpuFloat>(
    filter: &TensorDesc<T>,
    _grad_output: &TensorDesc<T>,
    grad_input: &TensorDescMut<T>,
    conv_desc: &ConvolutionDescriptor,
) -> DnnResult<ConvProblem> {
    let layout = grad_input.layout;
    let ndim = layout.expected_ndim();
    let spatial = layout.spatial_dims();

    if grad_input.dims.len() != ndim {
        return Err(DnnError::InvalidDimension(format!(
            "grad_input has {} dims, expected {ndim}",
            grad_input.dims.len()
        )));
    }

    let batch = grad_input.dims[0];
    let in_channels = grad_input.dims[1];
    let in_dims = grad_input.dims[2..].to_vec();

    let out_channels = filter.dims[0];
    let filter_dims = if filter.dims.len() >= 2 + spatial {
        filter.dims[2..2 + spatial].to_vec()
    } else {
        return Err(DnnError::InvalidDimension("filter dims too short".into()));
    };

    Ok(ConvProblem {
        batch,
        in_channels,
        in_dims,
        out_channels,
        filter_dims,
        padding: conv_desc.padding.clone(),
        stride: conv_desc.stride.clone(),
        dilation: conv_desc.dilation.clone(),
        groups: conv_desc.groups,
        input_type: T::PTX_TYPE,
        output_type: T::PTX_TYPE,
        layout,
    })
}

/// Constructs a [`ConvProblem`] for the backward filter pass.
fn build_wgrad_problem<T: GpuFloat>(
    input: &TensorDesc<T>,
    _grad_output: &TensorDesc<T>,
    grad_filter: &TensorDescMut<T>,
    conv_desc: &ConvolutionDescriptor,
) -> DnnResult<ConvProblem> {
    let layout = input.layout;
    let spatial = layout.spatial_dims();

    let batch = input.dims[0];
    let in_channels = input.dims[1];
    let in_dims = input.dims[2..].to_vec();

    let out_channels = grad_filter.dims[0];
    let filter_dims = if grad_filter.dims.len() >= 2 + spatial {
        grad_filter.dims[2..2 + spatial].to_vec()
    } else {
        return Err(DnnError::InvalidDimension(
            "grad_filter dims too short".into(),
        ));
    };

    Ok(ConvProblem {
        batch,
        in_channels,
        in_dims,
        out_channels,
        filter_dims,
        padding: conv_desc.padding.clone(),
        stride: conv_desc.stride.clone(),
        dilation: conv_desc.dilation.clone(),
        groups: conv_desc.groups,
        input_type: T::PTX_TYPE,
        output_type: T::PTX_TYPE,
        layout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    /// Verifies that algorithm selection works through the API path.
    #[test]
    fn select_algorithm_through_problem() {
        let problem = ConvProblem {
            batch: 1,
            in_channels: 64,
            in_dims: vec![32, 32],
            out_channels: 128,
            filter_dims: vec![1, 1],
            padding: vec![0, 0],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            input_type: oxicuda_ptx::ir::PtxType::F32,
            output_type: oxicuda_ptx::ir::PtxType::F32,
            layout: TensorLayout::Nchw,
        };
        let algo = problem.select_algorithm(oxicuda_ptx::arch::SmVersion::Sm80);
        assert_eq!(algo, ConvAlgorithm::Direct);
    }

    #[test]
    fn build_dgrad_problem_validates_dims() {
        // This test verifies the helper constructs a valid problem.
        let filter = TensorDesc::<f32>::from_raw(
            0,
            vec![128, 64, 3, 3],
            vec![576, 9, 3, 1],
            TensorLayout::Nchw,
        );
        let grad_out = TensorDesc::<f32>::from_raw(
            0,
            vec![1, 128, 32, 32],
            vec![131072, 1024, 32, 1],
            TensorLayout::Nchw,
        );
        let grad_in = TensorDescMut::<f32>::from_raw(
            0,
            vec![1, 64, 32, 32],
            vec![65536, 1024, 32, 1],
            TensorLayout::Nchw,
        );
        let conv_desc = ConvolutionDescriptor::conv2d(1, 1, 1, 1, 1, 1, 1);

        if let (Ok(f), Ok(go), Ok(gi), Ok(cd)) = (filter, grad_out, grad_in, conv_desc) {
            let problem = build_dgrad_problem::<f32>(&f, &go, &gi, &cd);
            assert!(problem.is_ok());
            if let Ok(p) = problem {
                assert_eq!(p.batch, 1);
                assert_eq!(p.in_channels, 64);
                assert_eq!(p.out_channels, 128);
            }
        }
    }
}
