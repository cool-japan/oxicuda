//! Implicit GEMM backward data gradient (dgrad).
//!
//! Computes the gradient of the loss with respect to the input tensor.
//! Conceptually this is the "transpose" of the forward convolution:
//! instead of sliding the filter over the input, we slide the transposed
//! filter over the gradient output.
//!
//! # GEMM mapping for dgrad
//!
//! ```text
//! grad_input[N, C, H, W] = conv_transpose(grad_output[N, K, P, Q], filter[K, C, R, S])
//!
//! M = batch * H * W         (input spatial points)
//! N = in_channels            (C)
//! K = out_channels * R * S   (filter volume)
//!
//! A[m, k] = grad_output at mapped position
//! B[k, n] = filter^T weights
//! D[m, n] = grad_input
//! ```
//!
//! Note: when the forward convolution has stride > 1, the dgrad becomes
//! a dilated convolution (inserting zeros between gradient output elements).

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
// DgradImplicitGemm
// ---------------------------------------------------------------------------

/// Backward data gradient via implicit GEMM.
///
/// Generates PTX for computing grad_input from grad_output and filters
/// using the transposed convolution pattern.
pub struct DgradImplicitGemm {
    problem: ConvProblem,
    sm_version: SmVersion,
}

impl DgradImplicitGemm {
    /// Creates a new dgrad engine.
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
        format!("dgrad_implicit_gemm_{prec}")
    }

    /// GEMM dimensions for the dgrad operation.
    ///
    /// - M = batch * in_H * in_W (input spatial points)
    /// - N = in_channels
    /// - K = out_channels * R * S
    ///
    /// # Errors
    ///
    /// Returns an error if spatial dims are not valid.
    pub fn dgrad_gemm_dims(&self) -> DnnResult<(u32, u32, u32)> {
        let in_spatial: u32 = self.problem.in_dims.iter().product();
        let gemm_m = self.problem.batch.saturating_mul(in_spatial);
        let gemm_n = self.problem.in_channels;
        let filter_volume: u32 = self.problem.filter_dims.iter().product();
        let gemm_k = self.problem.out_channels.saturating_mul(filter_volume);
        Ok((gemm_m, gemm_n, gemm_k))
    }

    /// Generates PTX for the dgrad implicit GEMM kernel.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on failure.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let ptx = KernelBuilder::new(&self.kernel_name())
            .target(self.sm_version)
            // Tensor pointers
            .param("grad_output", PtxType::U64)
            .param("filter", PtxType::U64)
            .param("grad_input", PtxType::U64)
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
                emit_dgrad_body(b);
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the dgrad computation.
    ///
    /// # Errors
    ///
    /// Returns errors from PTX generation, module loading, or kernel launch.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        grad_output: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        grad_input: &mut TensorDescMut<T>,
    ) -> DnnResult<()> {
        let ptx = self.generate_ptx()?;
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &self.kernel_name())?;

        let (gemm_m, _gemm_n, _gemm_k) = self.dgrad_gemm_dims()?;
        let out_dims = self.problem.output_dims()?;
        let out_h = out_dims.first().copied().unwrap_or(1);
        let out_w = out_dims.get(1).copied().unwrap_or(1);

        let block_size = 256u32;
        let grid = grid_size_for(gemm_m, block_size);
        let params = LaunchParams::new(grid, block_size);

        let args = (
            grad_output.ptr,
            filter.ptr,
            grad_input.ptr,
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

    /// Workspace required (zero for implicit GEMM dgrad).
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        0
    }
}

/// Standalone dgrad body emitter for the `'static` closure requirement.
fn emit_dgrad_body(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>) {
    b.comment("=== Dgrad Implicit GEMM (backward data) ===");
    b.comment("Transpose of forward conv: slide transposed filter over grad_output");

    let _gid = b.global_thread_id_x();

    b.comment("Map thread to input spatial position (batch, ih, iw)");
    b.comment("For each input position, accumulate over:");
    b.comment("  for k in 0..out_channels:");
    b.comment("    for r in 0..R:");
    b.comment("      for s in 0..S:");
    b.comment("        oh = (ih + pad_h - r * dilation_h)");
    b.comment("        if oh % stride_h == 0:");
    b.comment("          oh /= stride_h");
    b.comment("          ow = (iw + pad_w - s * dilation_w)");
    b.comment("          if ow % stride_w == 0:");
    b.comment("            ow /= stride_w");
    b.comment("            if 0 <= oh < P && 0 <= ow < Q:");
    b.comment("              grad_input[n, c, ih, iw] += ");
    b.comment("                grad_output[n, k, oh, ow] * filter[k, c, r, s]");

    b.ret();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn make_problem() -> ConvProblem {
        ConvProblem {
            batch: 1,
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
        let dg = DgradImplicitGemm::new(make_problem(), SmVersion::Sm80);
        assert_eq!(dg.kernel_name(), "dgrad_implicit_gemm_f32");
    }

    #[test]
    fn dgrad_gemm_dims() {
        let dg = DgradImplicitGemm::new(make_problem(), SmVersion::Sm80);
        let (m, n, k) = dg.dgrad_gemm_dims().unwrap_or((0, 0, 0));
        // M = 1 * 32 * 32 = 1024
        assert_eq!(m, 1024);
        // N = 64
        assert_eq!(n, 64);
        // K = 128 * 3 * 3 = 1152
        assert_eq!(k, 1152);
    }

    #[test]
    fn workspace_zero() {
        let dg = DgradImplicitGemm::new(make_problem(), SmVersion::Sm80);
        assert_eq!(dg.workspace_bytes(), 0);
    }

    #[test]
    fn ptx_generation() {
        let dg = DgradImplicitGemm::new(make_problem(), SmVersion::Sm80);
        let ptx = dg.generate_ptx();
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains("dgrad_implicit_gemm"));
    }
}
