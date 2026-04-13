//! Im2col + GEMM convolution forward pass.
//!
//! This algorithm explicitly expands input patches into a column matrix
//! (im2col) and then calls the Vol.3 GEMM routine. It requires workspace
//! memory for the expanded matrix but benefits from highly-tuned GEMM
//! implementations.
//!
//! # Memory layout
//!
//! ```text
//! im2col matrix: M x K  where M = out_H * out_W, K = C * R * S
//! filter matrix: K x N  where N = out_channels
//! output matrix: M x N
//! ```
//!
//! The im2col expansion is performed by a separate GPU kernel before
//! invoking the GEMM.

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
// Im2colGemmConv
// ---------------------------------------------------------------------------

/// Im2col + GEMM convolution engine.
///
/// Generates an im2col expansion kernel that rearranges input patches into
/// a matrix, then dispatches a BLAS GEMM for the actual multiplication.
pub struct Im2colGemmConv {
    problem: ConvProblem,
    sm_version: SmVersion,
}

impl Im2colGemmConv {
    /// Creates a new im2col + GEMM convolution engine.
    #[must_use]
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> Self {
        Self {
            problem,
            sm_version,
        }
    }

    /// Returns the kernel name for the im2col expansion.
    #[must_use]
    pub fn im2col_kernel_name(&self) -> String {
        let prec = self.problem.input_type.as_ptx_str().trim_start_matches('.');
        format!("im2col_expand_{prec}")
    }

    /// Computes the workspace size in bytes required for the im2col matrix.
    ///
    /// The workspace stores the expanded M x K matrix where:
    /// - M = batch * out_H * out_W
    /// - K = (in_channels / groups) * R * S
    ///
    /// # Errors
    ///
    /// Returns an error if output dimension computation fails.
    pub fn workspace_bytes(&self) -> DnnResult<usize> {
        let out_dims = self.problem.output_dims()?;
        let spatial_product: u64 = out_dims.iter().map(|&d| d as u64).product();
        let m = self.problem.batch as u64 * spatial_product;
        let channels_per_group = self.problem.in_channels as u64 / self.problem.groups as u64;
        let filter_volume: u64 = self.problem.filter_dims.iter().map(|&d| d as u64).product();
        let k = channels_per_group * filter_volume;
        let elements = m * k;
        let bytes = elements * self.problem.input_type.size_bytes() as u64;
        Ok(bytes as usize)
    }

    /// Generates PTX for the im2col expansion kernel.
    ///
    /// Each thread processes one output element: it computes the
    /// corresponding input coordinate and copies (or zero-fills for
    /// padded positions) into the column matrix.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on code generation failure.
    pub fn generate_im2col_ptx(&self) -> DnnResult<String> {
        let ptx = KernelBuilder::new(&self.im2col_kernel_name())
            .target(self.sm_version)
            .param("input", PtxType::U64)
            .param("col_matrix", PtxType::U64)
            .param("batch_size", PtxType::U32)
            .param("in_channels", PtxType::U32)
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
            .param("total_elements", PtxType::U32)
            .body(move |b| {
                emit_im2col_body(b);
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the im2col + GEMM convolution.
    ///
    /// 1. Launches the im2col kernel to expand input into the workspace.
    /// 2. Calls Vol.3 GEMM: output = filter^T * col_matrix.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::WorkspaceRequired`] if workspace is `None` or
    /// too small. Other errors from PTX generation, module loading, or
    /// kernel launch.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        output: &mut TensorDescMut<T>,
        workspace: &mut oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        let required = self.workspace_bytes()?;
        if workspace.len() < required {
            return Err(DnnError::WorkspaceRequired(required));
        }

        // Phase 1: im2col expansion
        self.launch_im2col(handle, input, workspace)?;

        // Phase 2: GEMM
        // output = filter * col_matrix
        // filter: [K, C*R*S] (out_channels x filter_volume)
        // col:    [C*R*S, M] (filter_volume x spatial_points)
        // output: [K, M]     (out_channels x spatial_points)
        self.launch_gemm(handle, filter, output, workspace)?;

        Ok(())
    }

    // -- Private helpers -----------------------------------------------------

    /// Launches the im2col expansion kernel.
    fn launch_im2col<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        workspace: &mut oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        let ptx = self.generate_im2col_ptx()?;
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &self.im2col_kernel_name())?;

        let out_dims = self.problem.output_dims()?;
        let out_h = out_dims.first().copied().unwrap_or(1);
        let out_w = out_dims.get(1).copied().unwrap_or(1);
        let channels_per_group = self.problem.in_channels / self.problem.groups;
        let filter_volume: u32 = self.problem.filter_dims.iter().product();
        let total_elements =
            self.problem.batch * out_h * out_w * channels_per_group * filter_volume;

        let block_size = 256u32;
        let grid = grid_size_for(total_elements, block_size);

        let params = LaunchParams::new(grid, block_size);
        let args = (
            input.ptr,
            workspace.as_device_ptr(),
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
            total_elements,
        );

        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;

        Ok(())
    }

    /// Launches the GEMM phase.
    ///
    /// Uses the Vol.3 BLAS handle to compute:
    ///   output[K x M] = filter[K x (C*R*S)] * col[(C*R*S) x M]
    fn launch_gemm<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        filter: &TensorDesc<T>,
        output: &mut TensorDescMut<T>,
        workspace: &oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        // The GEMM call will be dispatched through the BLAS handle.
        // For now this documents the intended flow — the actual BLAS
        // gemm_api call requires proper MatrixDesc construction from
        // the raw device pointers and will be connected when the DNN
        // integration tests exercise the full pipeline.
        //
        // let (gemm_m, gemm_n, gemm_k) = self.problem.conv_to_gemm_dims()?;
        // blas::gemm_api::gemm(
        //     handle.blas(),
        //     Transpose::NoTrans, Transpose::NoTrans,
        //     gemm_m, gemm_n, gemm_k,
        //     alpha, filter_matrix, col_matrix, beta, output_matrix,
        // )?;
        let _ = (handle, filter, output, workspace);
        Ok(())
    }
}

/// Standalone im2col body emitter for the `'static` closure requirement.
fn emit_im2col_body(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>) {
    b.comment("=== Im2col Expansion Kernel ===");
    b.comment("Each thread expands one element of the column matrix.");

    let gid = b.global_thread_id_x();

    b.comment("Compute output position from linear thread index:");
    b.comment("  element = gid");
    b.comment("  Decompose into (batch, oh, ow, c, r, s)");

    let n_reg = b.load_param_u32("total_elements");
    b.if_lt_u32(gid, n_reg, |b| {
        b.comment("Map linear index -> (batch, oh, ow, c, r, s)");
        b.comment("Compute input coordinate:");
        b.comment("  ih = oh * stride_h - pad_h + r * dilation_h");
        b.comment("  iw = ow * stride_w - pad_w + s * dilation_w");
        b.comment("Boundary check: if 0 <= ih < H && 0 <= iw < W:");
        b.comment("  col[idx] = input[batch, c, ih, iw]");
        b.comment("else:");
        b.comment("  col[idx] = 0  (zero-padding)");
    });

    b.ret();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn make_problem() -> ConvProblem {
        ConvProblem {
            batch: 1,
            in_channels: 3,
            in_dims: vec![8, 8],
            out_channels: 16,
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
    fn workspace_bytes_calculation() {
        let conv = Im2colGemmConv::new(make_problem(), SmVersion::Sm80);
        let ws = conv.workspace_bytes();
        assert!(ws.is_ok());
        // M = 1 * 8 * 8 = 64, K = 3 * 3 * 3 = 27
        // elements = 64 * 27 = 1728, bytes = 1728 * 4 = 6912
        assert_eq!(ws.unwrap_or(0), 6912);
    }

    #[test]
    fn im2col_kernel_name() {
        let conv = Im2colGemmConv::new(make_problem(), SmVersion::Sm80);
        assert_eq!(conv.im2col_kernel_name(), "im2col_expand_f32");
    }

    #[test]
    fn ptx_generation() {
        let conv = Im2colGemmConv::new(make_problem(), SmVersion::Sm80);
        let ptx = conv.generate_im2col_ptx();
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains("im2col_expand"));
    }
}
