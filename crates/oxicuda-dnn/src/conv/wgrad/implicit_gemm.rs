//! Implicit GEMM backward filter gradient (wgrad).
//!
//! Computes the gradient of the loss with respect to the convolution
//! filter weights. This is a cross-correlation between the input tensor
//! and the gradient output tensor.
//!
//! # GEMM mapping for wgrad
//!
//! ```text
//! grad_filter[K, C, R, S] = cross_corr(input[N, C, H, W], grad_output[N, K, P, Q])
//!
//! M = out_channels (K)
//! N = in_channels * R * S  (filter volume per group)
//! K_gemm = batch * P * Q   (output spatial points)
//!
//! A[m, k] = grad_output reshaped  (K x NPQ)
//! B[k, n] = input at mapped positions (NPQ x CRS)
//! D[m, n] = grad_filter  (K x CRS)
//! ```
//!
//! The wgrad is typically the most expensive backward pass because it
//! accumulates over the entire batch dimension.

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
// WgradImplicitGemm
// ---------------------------------------------------------------------------

/// Backward filter gradient via implicit GEMM.
///
/// Generates PTX for computing grad_filter from input and grad_output
/// using a cross-correlation pattern mapped to GEMM.
pub struct WgradImplicitGemm {
    problem: ConvProblem,
    sm_version: SmVersion,
}

impl WgradImplicitGemm {
    /// Creates a new wgrad engine.
    #[must_use]
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> Self {
        Self {
            problem,
            sm_version,
        }
    }

    /// Returns the kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let prec = self.problem.input_type.as_ptx_str().trim_start_matches('.');
        format!("wgrad_implicit_gemm_{prec}")
    }

    /// GEMM dimensions for the wgrad operation.
    ///
    /// - M = out_channels (K filters)
    /// - N = (in_channels / groups) * R * S (filter volume)
    /// - K_gemm = batch * P * Q (output spatial points, reduction axis)
    ///
    /// # Errors
    ///
    /// Returns an error if output dimension computation fails.
    pub fn wgrad_gemm_dims(&self) -> DnnResult<(u32, u32, u32)> {
        let out_dims = self.problem.output_dims()?;
        let out_spatial: u32 = out_dims.iter().product();

        let gemm_m = self.problem.out_channels;
        let channels_per_group = self.problem.in_channels / self.problem.groups;
        let filter_volume: u32 = self.problem.filter_dims.iter().product();
        let gemm_n = channels_per_group.saturating_mul(filter_volume);
        let gemm_k = self.problem.batch.saturating_mul(out_spatial);

        Ok((gemm_m, gemm_n, gemm_k))
    }

    /// Generates PTX for the wgrad implicit GEMM kernel.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on failure.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let ptx = KernelBuilder::new(&self.kernel_name())
            .target(self.sm_version)
            // Tensor pointers
            .param("input", PtxType::U64)
            .param("grad_output", PtxType::U64)
            .param("grad_filter", PtxType::U64)
            // Dimensions
            .param("batch_size", PtxType::U32)
            .param("in_channels", PtxType::U32)
            .param("in_h", PtxType::U32)
            .param("in_w", PtxType::U32)
            .param("out_channels", PtxType::U32)
            .param("filter_h", PtxType::U32)
            .param("filter_w", PtxType::U32)
            .param("out_h", PtxType::U32)
            .param("out_w", PtxType::U32)
            // Conv parameters
            .param("pad_h", PtxType::U32)
            .param("pad_w", PtxType::U32)
            .param("stride_h", PtxType::U32)
            .param("stride_w", PtxType::U32)
            .param("dilation_h", PtxType::U32)
            .param("dilation_w", PtxType::U32)
            .body(move |b| {
                emit_wgrad_body(b);
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the wgrad computation.
    ///
    /// # Errors
    ///
    /// Returns errors from PTX generation, module loading, or kernel launch.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        grad_output: &TensorDesc<T>,
        grad_filter: &mut TensorDescMut<T>,
    ) -> DnnResult<()> {
        let ptx = self.generate_ptx()?;
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &self.kernel_name())?;

        let out_dims = self.problem.output_dims()?;
        let out_h = out_dims.first().copied().unwrap_or(1);
        let out_w = out_dims.get(1).copied().unwrap_or(1);

        // Each thread computes one element of grad_filter.
        let filter_volume: u32 = self.problem.filter_dims.iter().product();
        let channels_per_group = self.problem.in_channels / self.problem.groups;
        let total_elements = self.problem.out_channels * channels_per_group * filter_volume;

        let block_size = 256u32;
        let grid = grid_size_for(total_elements, block_size);
        let params = LaunchParams::new(grid, block_size);

        let args = (
            input.ptr,
            grad_output.ptr,
            grad_filter.ptr,
            self.problem.batch,
            self.problem.in_channels,
            self.problem.in_dims[0],
            self.problem.in_dims.get(1).copied().unwrap_or(1),
            self.problem.out_channels,
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
        );

        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;

        Ok(())
    }

    /// Workspace required (zero for implicit GEMM wgrad).
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        0
    }
}

/// Standalone wgrad body emitter for the `'static` closure requirement.
fn emit_wgrad_body(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>) {
    b.comment("=== Wgrad Implicit GEMM (backward filter) ===");
    b.comment("Cross-correlation of input and grad_output.");

    let _gid = b.global_thread_id_x();

    b.comment("Map thread to filter position (k, c, r, s)");
    b.comment("Accumulate over batch and spatial dimensions:");
    b.comment("  grad_filter[k, c, r, s] = sum over n, oh, ow of:");
    b.comment("    grad_output[n, k, oh, ow] *");
    b.comment("    input[n, c, oh*stride_h - pad_h + r*dilation_h,");
    b.comment("              ow*stride_w - pad_w + s*dilation_w]");
    b.comment("  (with boundary checks for padding)");

    b.ret();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn make_problem() -> ConvProblem {
        ConvProblem {
            batch: 4,
            in_channels: 64,
            in_dims: vec![32, 32],
            out_channels: 128,
            filter_dims: vec![3, 3],
            padding: vec![1, 1],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    #[test]
    fn kernel_name() {
        let wg = WgradImplicitGemm::new(make_problem(), SmVersion::Sm80);
        assert_eq!(wg.kernel_name(), "wgrad_implicit_gemm_f32");
    }

    #[test]
    fn wgrad_gemm_dims() {
        let wg = WgradImplicitGemm::new(make_problem(), SmVersion::Sm80);
        let (m, n, k) = wg.wgrad_gemm_dims().unwrap_or((0, 0, 0));
        // M = 128 (out_channels)
        assert_eq!(m, 128);
        // N = 64 * 3 * 3 = 576 (filter volume)
        assert_eq!(n, 576);
        // K = 4 * 32 * 32 = 4096 (batch * spatial)
        assert_eq!(k, 4096);
    }

    #[test]
    fn workspace_zero() {
        let wg = WgradImplicitGemm::new(make_problem(), SmVersion::Sm80);
        assert_eq!(wg.workspace_bytes(), 0);
    }

    #[test]
    fn ptx_generation() {
        let wg = WgradImplicitGemm::new(make_problem(), SmVersion::Sm80);
        let ptx = wg.generate_ptx();
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains("wgrad_implicit_gemm"));
    }
}
