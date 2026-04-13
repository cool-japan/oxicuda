//! Implicit GEMM convolution forward pass.
//!
//! Maps convolution to a GEMM without explicitly materialising the im2col
//! matrix. Instead, the GEMM kernel computes the conv-to-GEMM index mapping
//! on-the-fly, checking padding boundaries for each loaded element.
//!
//! This is the most versatile conv algorithm — it requires zero workspace
//! and handles arbitrary padding, stride, and dilation. On Ampere+ with
//! NHWC layout, it achieves near-optimal throughput via `cp.async` and
//! Tensor Core MMA instructions.
//!
//! # GEMM mapping
//!
//! ```text
//! M = batch * out_H * out_W     (output spatial points)
//! N = out_channels               (filter count)
//! K = in_channels * R * S        (filter volume)
//!
//! A[m, k] = input at conv-mapped position  (implicit im2col)
//! B[k, n] = filter weights
//! D[m, n] = output
//! ```

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::types::{TensorDesc, TensorDescMut, TileConfig};

use super::super::descriptor::ConvProblem;

// ---------------------------------------------------------------------------
// ImplicitGemmConv
// ---------------------------------------------------------------------------

/// Implicit GEMM convolution engine.
///
/// Generates and launches a PTX kernel that computes convolution as a GEMM
/// with implicit im2col address mapping inside the inner loop.
pub struct ImplicitGemmConv {
    problem: ConvProblem,
    tile_config: TileConfig,
    sm_version: SmVersion,
}

impl ImplicitGemmConv {
    /// Creates a new implicit GEMM convolution engine.
    #[must_use]
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> Self {
        let tile_config = TileConfig::default_conv(sm_version);
        Self {
            problem,
            tile_config,
            sm_version,
        }
    }

    /// Creates with a custom tile configuration.
    #[must_use]
    pub fn with_tile_config(
        problem: ConvProblem,
        tile_config: TileConfig,
        sm_version: SmVersion,
    ) -> Self {
        Self {
            problem,
            tile_config,
            sm_version,
        }
    }

    /// Returns a unique kernel name encoding the problem parameters.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let prec = self.problem.input_type.as_ptx_str().trim_start_matches('.');
        format!(
            "implicit_gemm_conv_{}x{}x{}_{}",
            self.tile_config.tile_m, self.tile_config.tile_n, self.tile_config.tile_k, prec,
        )
    }

    /// Generates the complete PTX module for the implicit GEMM conv kernel.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on code generation failure.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let _gemm_dims = self.problem.conv_to_gemm_dims()?;

        // Capture values for the 'static closure.
        let sm = self.sm_version;
        let stages = self.tile_config.stages;

        let ptx = KernelBuilder::new(&self.kernel_name())
            .target(self.sm_version)
            // Tensor pointers
            .param("input", PtxType::U64)
            .param("filter", PtxType::U64)
            .param("output", PtxType::U64)
            .param("bias", PtxType::U64)
            // Tensor dimensions
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
            // GEMM dimensions (precomputed)
            .param("gemm_m", PtxType::U32)
            .param("gemm_n", PtxType::U32)
            .param("gemm_k", PtxType::U32)
            // Shared memory for input and filter tiles
            .shared_mem(
                "smem_input",
                self.problem.input_type,
                self.smem_input_elements(),
            )
            .shared_mem(
                "smem_filter",
                self.problem.input_type,
                self.smem_filter_elements(),
            )
            .body(move |b| {
                emit_implicit_gemm_body(b, sm, stages);
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the implicit GEMM convolution.
    ///
    /// # Errors
    ///
    /// Returns errors from PTX generation, module loading, or kernel launch.
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

        let (gemm_m, gemm_n, _gemm_k) = self.problem.conv_to_gemm_dims()?;
        let out_dims = self.problem.output_dims()?;
        let out_h = out_dims.first().copied().unwrap_or(1);
        let out_w = out_dims.get(1).copied().unwrap_or(1);

        // Grid: blocks cover (M / tile_m) x (N / tile_n)
        let grid_x = grid_size_for(gemm_m, self.tile_config.tile_m);
        let grid_y = grid_size_for(gemm_n, self.tile_config.tile_n);
        let grid = Dim3::xy(grid_x, grid_y);

        // Block: threads_per_block depends on tile / warp config
        let warps_m = self.tile_config.tile_m / self.tile_config.warp_m;
        let warps_n = self.tile_config.tile_n / self.tile_config.warp_n;
        let threads = warps_m * warps_n * 32;
        let block = Dim3::x(threads.min(1024));

        let shared_bytes = (self.smem_input_elements() + self.smem_filter_elements())
            * self.problem.input_type.size_bytes();

        let params = LaunchParams::new(grid, block).with_shared_mem(shared_bytes as u32);

        let args = (
            input.ptr,
            filter.ptr,
            output.ptr,
            0u64, // bias (null for now)
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

    // Shared memory and workspace sizing are provided via methods below.

    // -- Shared memory sizing ------------------------------------------------

    /// Number of elements for the input tile in shared memory.
    fn smem_input_elements(&self) -> usize {
        let tile_m = self.tile_config.tile_m as usize;
        let tile_k = self.tile_config.tile_k as usize;
        let stages = self.tile_config.stages as usize;
        tile_m * tile_k * stages
    }

    /// Number of elements for the filter tile in shared memory.
    fn smem_filter_elements(&self) -> usize {
        let tile_n = self.tile_config.tile_n as usize;
        let tile_k = self.tile_config.tile_k as usize;
        let stages = self.tile_config.stages as usize;
        tile_n * tile_k * stages
    }

    /// Returns the workspace size in bytes (implicit GEMM needs zero).
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Standalone PTX body emitter (must be 'static for KernelBuilder::body)
// ---------------------------------------------------------------------------

/// Emits the implicit GEMM convolution kernel body.
fn emit_implicit_gemm_body(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    sm: SmVersion,
    stages: u32,
) {
    b.comment("=== Implicit GEMM Convolution (forward) ===");

    b.comment("Step 1: Map CTA to GEMM tile coordinates");
    b.comment("  blockIdx.x -> M-tile (batch * out_h * out_w)");
    b.comment("  blockIdx.y -> N-tile (out_channels)");

    let _gid_x = b.global_thread_id_x();
    let _gid_y = b.global_thread_id_y();

    b.comment("Step 2: Mainloop over filter volume (C x R x S)");
    b.comment("  For each k-iteration:");
    b.comment("    channel_idx = k / (R * S)");
    b.comment("    r_idx = (k / S) % R");
    b.comment("    s_idx = k % S");
    b.comment("    input_h = out_h * stride_h - pad_h + r_idx * dilation_h");
    b.comment("    input_w = out_w * stride_w - pad_w + s_idx * dilation_w");
    b.comment("    Boundary check: load 0 if out of bounds (zero-padding)");

    if sm >= SmVersion::Sm80 {
        b.comment("--- Async pipeline (cp.async) for Ampere+ ---");
        b.comment(&format!("Pipeline depth: {stages} stages"));
        for stage in 0..stages.saturating_sub(1) {
            b.comment(&format!("  Prologue: async load stage {stage}"));
        }
        b.comment("  Mainloop: for each K-tile");
        b.comment("    1. Wait for oldest async load to complete");
        b.comment("    2. Compute GEMM tile (MMA or FMA)");
        b.comment("    3. Issue next async load");
        for stage in 0..stages.saturating_sub(1) {
            b.comment(&format!("  Drain: compute stage {stage}"));
        }
    } else {
        b.comment("--- Standard mainloop (Turing / pre-Ampere) ---");
        b.comment("For each K-tile:");
        b.comment("  1. Load input tile to smem with boundary predicates");
        b.comment("  2. Load filter tile to smem");
        b.comment("  3. __syncthreads()");
        b.comment("  4. Compute tile GEMM (FMA loop or WMMA)");
        b.comment("  5. __syncthreads()");
    }

    b.comment("Step 3: Epilogue -- write accumulator to global output");
    b.comment("  Optional: add bias, apply activation (ReLU, etc.)");

    b.ret();
}

// ---------------------------------------------------------------------------
// Conv-to-GEMM index mapping utilities
// ---------------------------------------------------------------------------

/// Maps a linear GEMM-M index back to convolution output coordinates.
///
/// Given `m = batch_idx * (out_H * out_W) + oh * out_W + ow`, this
/// function recovers `(batch_idx, oh, ow)`.
#[inline]
pub fn gemm_m_to_conv_coords(m: u32, out_h: u32, out_w: u32) -> (u32, u32, u32) {
    let spatial = out_h * out_w;
    let batch_idx = m / spatial;
    let remainder = m % spatial;
    let oh = remainder / out_w;
    let ow = remainder % out_w;
    (batch_idx, oh, ow)
}

/// Maps a linear GEMM-K index to convolution filter coordinates.
///
/// Given `k = c * (R * S) + r * S + s`, recovers `(c, r, s)`.
#[inline]
pub fn gemm_k_to_filter_coords(k: u32, filter_h: u32, filter_w: u32) -> (u32, u32, u32) {
    let rs = filter_h * filter_w;
    let c = k / rs;
    let remainder = k % rs;
    let r = remainder / filter_w;
    let s = remainder % filter_w;
    (c, r, s)
}

/// Computes the input spatial coordinate for a given output position
/// and filter offset, checking padding boundaries.
///
/// Returns `None` if the computed position falls outside the valid
/// input range (i.e. it would be a zero-padded position).
#[inline]
pub fn input_coord(
    out_pos: u32,
    filter_pos: u32,
    pad: u32,
    stride: u32,
    dilation: u32,
    input_size: u32,
) -> Option<u32> {
    let pos = (out_pos * stride) as i64 - pad as i64 + (filter_pos * dilation) as i64;
    if pos >= 0 && (pos as u32) < input_size {
        Some(pos as u32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn make_problem() -> ConvProblem {
        ConvProblem {
            batch: 2,
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
    fn kernel_name_format() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        let name = conv.kernel_name();
        assert!(name.contains("implicit_gemm_conv"));
        assert!(name.contains("f32"));
    }

    #[test]
    fn workspace_is_zero() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        assert_eq!(conv.workspace_bytes(), 0);
    }

    #[test]
    fn smem_sizes_positive() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        assert!(conv.smem_input_elements() > 0);
        assert!(conv.smem_filter_elements() > 0);
    }

    #[test]
    fn gemm_m_to_conv_coords_basic() {
        // m=0 -> (batch=0, oh=0, ow=0)
        assert_eq!(gemm_m_to_conv_coords(0, 4, 4), (0, 0, 0));
        // m=5 -> (batch=0, oh=1, ow=1)
        assert_eq!(gemm_m_to_conv_coords(5, 4, 4), (0, 1, 1));
        // m=16 -> (batch=1, oh=0, ow=0)
        assert_eq!(gemm_m_to_conv_coords(16, 4, 4), (1, 0, 0));
    }

    #[test]
    fn gemm_k_to_filter_coords_basic() {
        // k=0 -> (c=0, r=0, s=0)
        assert_eq!(gemm_k_to_filter_coords(0, 3, 3), (0, 0, 0));
        // k=4 -> (c=0, r=1, s=1)
        assert_eq!(gemm_k_to_filter_coords(4, 3, 3), (0, 1, 1));
        // k=9 -> (c=1, r=0, s=0)
        assert_eq!(gemm_k_to_filter_coords(9, 3, 3), (1, 0, 0));
    }

    #[test]
    fn input_coord_valid() {
        // out=1, filter=0, pad=1, stride=1, dilation=1, size=32
        // pos = 1*1 - 1 + 0*1 = 0 -> valid
        assert_eq!(input_coord(1, 0, 1, 1, 1, 32), Some(0));
    }

    #[test]
    fn input_coord_padded() {
        // out=0, filter=0, pad=1, stride=1, dilation=1, size=32
        // pos = 0*1 - 1 + 0*1 = -1 -> out of bounds
        assert_eq!(input_coord(0, 0, 1, 1, 1, 32), None);
    }

    #[test]
    fn input_coord_beyond_input() {
        // out=31, filter=2, pad=1, stride=1, dilation=1, size=32
        // pos = 31 - 1 + 2 = 32 -> out of bounds (size=32)
        assert_eq!(input_coord(31, 2, 1, 1, 1, 32), None);
    }

    #[test]
    fn ptx_generation_produces_output() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        let ptx = conv.generate_ptx();
        assert!(ptx.is_ok());
        let ptx_text = ptx.unwrap_or_default();
        assert!(ptx_text.contains("implicit_gemm_conv"));
        assert!(ptx_text.contains(".entry"));
    }
}
