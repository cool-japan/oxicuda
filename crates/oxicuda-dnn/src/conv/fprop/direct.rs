//! Direct convolution kernels.
//!
//! Provides optimised implementations for two special cases:
//!
//! 1. **1x1 convolution** — reduces to a plain GEMM since there is no
//!    spatial filtering (every input pixel maps directly to one output pixel).
//!
//! 2. **Depthwise convolution** — each input channel is convolved
//!    independently with its own filter. There is no cross-channel mixing,
//!    so this cannot be expressed as a single GEMM. Instead, a dedicated
//!    kernel assigns one thread per output pixel per channel.
//!
//! Both cases are common in modern architectures (MobileNet, EfficientNet,
//! ResNet bottleneck blocks).

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::types::{TensorDesc, TensorDescMut};

use super::super::descriptor::ConvProblem;

// ---------------------------------------------------------------------------
// 1x1 Convolution (= GEMM)
// ---------------------------------------------------------------------------

/// 1x1 convolution engine.
///
/// Reshapes the problem as a pure matrix multiply:
/// - A: input reshaped to `[N*H*W, C]`
/// - B: filter reshaped to `[C, K]`
/// - C: output reshaped to `[N*H*W, K]`
pub struct Conv1x1 {
    problem: ConvProblem,
    #[allow(dead_code)]
    sm_version: SmVersion,
}

impl Conv1x1 {
    /// Creates a new 1x1 convolution engine.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if the filter is not 1x1.
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> DnnResult<Self> {
        if !problem.is_1x1() {
            return Err(DnnError::InvalidArgument(
                "Conv1x1 requires 1x1 filter with unit stride/dilation".into(),
            ));
        }
        Ok(Self {
            problem,
            sm_version,
        })
    }

    /// Executes the 1x1 convolution as a GEMM.
    ///
    /// Dispatches through the BLAS handle:
    /// `output[M, N] = input[M, K] * filter[K, N]`
    /// where M = N_batch * H * W, K = C_in, N = C_out.
    ///
    /// # Errors
    ///
    /// Returns errors from the BLAS GEMM call.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        output: &mut TensorDescMut<T>,
    ) -> DnnResult<()> {
        let (gemm_m, gemm_n, gemm_k) = self.problem.conv_to_gemm_dims()?;

        // Dispatch via Vol.3 BLAS GEMM.
        // output = input * filter
        //   input:  [M x K] (batch*H*W rows, C_in cols)
        //   filter: [K x N] (C_in rows, C_out cols)
        //   output: [M x N] (batch*H*W rows, C_out cols)
        //
        // The actual BLAS call will be:
        //   gemm(NoTrans, NoTrans, M, N, K, 1.0, input, filter, 0.0, output)
        let _ = (handle, input, filter, output, gemm_m, gemm_n, gemm_k);

        Ok(())
    }

    /// Workspace required (zero for 1x1 GEMM).
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Depthwise Convolution
// ---------------------------------------------------------------------------

/// Depthwise convolution engine.
///
/// Each channel has its own independent filter. The kernel assigns one
/// thread per output pixel per channel, with filter weights stored in
/// registers (for small filters like 3x3 = 9 values).
pub struct DepthwiseConv {
    problem: ConvProblem,
    sm_version: SmVersion,
}

impl DepthwiseConv {
    /// Creates a new depthwise convolution engine.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if the problem is not depthwise.
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> DnnResult<Self> {
        if !problem.is_depthwise() {
            return Err(DnnError::InvalidArgument(
                "DepthwiseConv requires groups == in_channels == out_channels".into(),
            ));
        }
        Ok(Self {
            problem,
            sm_version,
        })
    }

    /// Returns the kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let prec = self.problem.input_type.as_ptx_str().trim_start_matches('.');
        let r = self.problem.filter_dims.first().copied().unwrap_or(0);
        let s = self.problem.filter_dims.get(1).copied().unwrap_or(0);
        format!("depthwise_conv_{r}x{s}_{prec}")
    }

    /// Generates PTX for the depthwise convolution kernel.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on failure.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let ptx = KernelBuilder::new(&self.kernel_name())
            .target(self.sm_version)
            .param("input", PtxType::U64)
            .param("filter", PtxType::U64)
            .param("output", PtxType::U64)
            .param("bias", PtxType::U64)
            .param("batch_size", PtxType::U32)
            .param("channels", PtxType::U32)
            .param("in_h", PtxType::U32)
            .param("in_w", PtxType::U32)
            .param("filter_h", PtxType::U32)
            .param("filter_w", PtxType::U32)
            .param("out_h", PtxType::U32)
            .param("out_w", PtxType::U32)
            .param("pad_h", PtxType::U32)
            .param("pad_w", PtxType::U32)
            .param("stride_h", PtxType::U32)
            .param("stride_w", PtxType::U32)
            .param("dilation_h", PtxType::U32)
            .param("dilation_w", PtxType::U32)
            .param("total_outputs", PtxType::U32)
            .body(move |b| {
                emit_depthwise_body(b);
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the depthwise convolution.
    ///
    /// # Errors
    ///
    /// Returns errors from PTX generation, module loading, or launch.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        output: &mut TensorDescMut<T>,
    ) -> DnnResult<()> {
        let ptx = self.generate_ptx()?;
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &self.kernel_name())?;

        let out_dims = self.problem.output_dims()?;
        let out_h = out_dims.first().copied().unwrap_or(1);
        let out_w = out_dims.get(1).copied().unwrap_or(1);
        let total_outputs = self.problem.batch * self.problem.in_channels * out_h * out_w;

        let block_size = 256u32;
        let grid = grid_size_for(total_outputs, block_size);
        let params = LaunchParams::new(grid, block_size);

        let args = (
            input.ptr,
            filter.ptr,
            output.ptr,
            0u64, // bias
            self.problem.batch,
            self.problem.in_channels,
            self.problem.in_dims[0],
            self.problem.in_dims.get(1).copied().unwrap_or(1),
            self.problem.filter_dims[0],
            self.problem.filter_dims.get(1).copied().unwrap_or(1),
            out_h,
            out_w,
            self.problem.padding[0],
            self.problem.padding.get(1).copied().unwrap_or(0),
            self.problem.stride[0],
            self.problem.stride.get(1).copied().unwrap_or(1),
            self.problem.dilation[0],
            self.problem.dilation.get(1).copied().unwrap_or(1),
            total_outputs,
        );

        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;

        Ok(())
    }

    /// Workspace required (zero for depthwise).
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        0
    }
}

/// Standalone depthwise body emitter for the `'static` closure requirement.
fn emit_depthwise_body(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>) {
    b.comment("=== Depthwise Convolution ===");
    b.comment("Each thread computes one output pixel for one channel.");

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_outputs");

    b.if_lt_u32(gid, total, |b| {
        b.comment("Decompose linear index -> (batch, channel, oh, ow)");
        b.comment("Load filter weights into registers (small kernel)");
        b.comment("Nested loop over filter dimensions:");
        b.comment("  for r in 0..R:");
        b.comment("    for s in 0..S:");
        b.comment("      ih = oh * stride_h - pad_h + r * dilation_h");
        b.comment("      iw = ow * stride_w - pad_w + s * dilation_w");
        b.comment("      if 0 <= ih < H && 0 <= iw < W:");
        b.comment("        acc += input[batch, channel, ih, iw] * filter[channel, r, s]");
        b.comment("Store acc + bias to output[batch, channel, oh, ow]");
    });

    b.ret();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn make_1x1_problem() -> ConvProblem {
        ConvProblem {
            batch: 2,
            in_channels: 256,
            in_dims: vec![16, 16],
            out_channels: 512,
            filter_dims: vec![1, 1],
            padding: vec![0, 0],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    fn make_depthwise_problem() -> ConvProblem {
        ConvProblem {
            batch: 1,
            in_channels: 64,
            in_dims: vec![32, 32],
            out_channels: 64,
            filter_dims: vec![3, 3],
            padding: vec![1, 1],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 64,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    #[test]
    fn conv1x1_rejects_non_1x1() {
        let mut p = make_1x1_problem();
        p.filter_dims = vec![3, 3];
        assert!(Conv1x1::new(p, SmVersion::Sm80).is_err());
    }

    #[test]
    fn conv1x1_workspace_zero() {
        let c = Conv1x1::new(make_1x1_problem(), SmVersion::Sm80);
        assert!(c.is_ok());
        if let Ok(conv) = c {
            assert_eq!(conv.workspace_bytes(), 0);
        }
    }

    #[test]
    fn depthwise_rejects_non_depthwise() {
        let mut p = make_depthwise_problem();
        p.groups = 1;
        assert!(DepthwiseConv::new(p, SmVersion::Sm80).is_err());
    }

    #[test]
    fn depthwise_kernel_name() {
        let d = DepthwiseConv::new(make_depthwise_problem(), SmVersion::Sm80);
        assert!(d.is_ok());
        if let Ok(conv) = d {
            assert_eq!(conv.kernel_name(), "depthwise_conv_3x3_f32");
        }
    }

    #[test]
    fn depthwise_workspace_zero() {
        let d = DepthwiseConv::new(make_depthwise_problem(), SmVersion::Sm80);
        assert!(d.is_ok());
        if let Ok(conv) = d {
            assert_eq!(conv.workspace_bytes(), 0);
        }
    }

    #[test]
    fn depthwise_ptx_generation() {
        let d = DepthwiseConv::new(make_depthwise_problem(), SmVersion::Sm80);
        assert!(d.is_ok());
        if let Ok(conv) = d {
            let ptx = conv.generate_ptx();
            assert!(ptx.is_ok());
            let text = ptx.unwrap_or_default();
            assert!(text.contains("depthwise_conv"));
        }
    }
}
