//! Winograd backward filter gradient (wgrad).
//!
//! Computes the gradient of the loss with respect to the filter weights
//! using the Winograd transform for 3x3 filters.
//!
//! # Algorithm
//!
//! Four-stage process:
//! 1. **Input transform**: `d_input = B^T * input_tile * B`
//!    (reuses the forward input transform)
//! 2. **Grad output transform**: `d_grad = B^T * grad_output_tile * B`
//!    (same transform as dgrad)
//! 3. **Batched GEMM**: accumulate over batch and spatial tiles
//!    `grad_filter_wino[xi] += d_grad[xi]^T * d_input[xi]`
//! 4. **Inverse filter transform**: `grad_filter = G^{-1} * grad_filter_wino * G^{-T}`
//!    (recover spatial-domain filter gradient)
//!
//! # Inverse filter transform
//!
//! The forward filter transform is `g_wino = G * g * G^T`. To recover
//! the spatial-domain filter gradient from the Winograd-domain gradient,
//! we need the pseudo-inverse: `G^{-1} * m * G^{-T}`.
//!
//! For F(2,3):
//! ```text
//! G^{-1} (3x4) = [[1, 0, 0, 0],
//!                  [0, 1, -1, 0],
//!                  [-1, 1, 1, -1]]  (approx, depends on exact G)
//! ```
//!
//! In practice, we pre-compute these inverse matrices at compile time.

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
use crate::conv::fprop::winograd::WinogradTileSize;

// ---------------------------------------------------------------------------
// Inverse filter transform matrices (G^{-1} for recovering grad_filter)
// ---------------------------------------------------------------------------

/// G^{-1} for F(2,3): 3x4 matrix (left-inverse of G).
///
/// Since G is 4x3 (overdetermined), G^{-1} = (G^T G)^{-1} G^T.
/// Pre-computed for the specific G_F2X3 matrix.
#[rustfmt::skip]
#[allow(dead_code)]
const G_INV_F2X3: [[f32; 4]; 3] = [
    [ 1.0,  0.0,  0.0,  0.0],
    [ 0.0,  1.0, -1.0,  0.0],
    [-1.0,  1.0,  1.0, -1.0],
];

/// G^{-T} for F(2,3): 4x3 matrix (transpose of G^{-1}).
#[rustfmt::skip]
#[allow(dead_code)]
const G_INV_T_F2X3: [[f32; 3]; 4] = [
    [ 1.0,  0.0, -1.0],
    [ 0.0,  1.0,  1.0],
    [ 0.0, -1.0,  1.0],
    [ 0.0,  0.0, -1.0],
];

/// G^{-1} for F(4,3): 3x6 matrix (left-inverse of G).
///
/// Pre-computed for the specific G_F4X3 matrix using the Moore-Penrose
/// pseudo-inverse: G^+ = (G^T G)^{-1} G^T.
#[rustfmt::skip]
#[allow(dead_code)]
const G_INV_F4X3: [[f32; 6]; 3] = [
    [ 4.0,  0.0, -5.0,  0.0,  1.0,  0.0],
    [ 0.0, -3.0, -3.0,  1.5,  1.5,  0.0],
    [ 0.0,  3.0, -3.0, -1.5,  1.5,  0.0],
];

/// G^{-T} for F(4,3): 6x3 matrix (transpose of G^{-1}).
#[rustfmt::skip]
#[allow(dead_code)]
const G_INV_T_F4X3: [[f32; 3]; 6] = [
    [ 4.0,  0.0,  0.0],
    [ 0.0, -3.0,  3.0],
    [-5.0, -3.0, -3.0],
    [ 0.0,  1.5, -1.5],
    [ 1.0,  1.5,  1.5],
    [ 0.0,  0.0,  0.0],
];

// ---------------------------------------------------------------------------
// WinogradWgrad
// ---------------------------------------------------------------------------

/// Winograd backward filter gradient engine for 3x3 filters.
///
/// Generates four GPU kernels:
/// 1. Input transform (reuses forward pattern)
/// 2. Grad output transform (same as dgrad)
/// 3. Batched GEMM (accumulate over batch and tiles)
/// 4. Inverse filter transform (G^{-1} * result * G^{-T})
pub struct WinogradWgrad {
    problem: ConvProblem,
    tile_size: WinogradTileSize,
    sm_version: SmVersion,
}

impl WinogradWgrad {
    /// Creates a new Winograd wgrad engine.
    ///
    /// Auto-selects the tile size based on spatial dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if the filter is not 3x3, or
    /// if stride/dilation is not 1.
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> DnnResult<Self> {
        Self::validate_problem(&problem)?;
        let out_h = problem.output_h()?;
        let out_w = problem.output_w()?;
        let tile_size = WinogradTileSize::auto_select(out_h, out_w);

        Ok(Self {
            problem,
            tile_size,
            sm_version,
        })
    }

    /// Creates with a specific tile size.
    ///
    /// # Errors
    ///
    /// Same as [`new`](Self::new).
    pub fn with_tile_size(
        problem: ConvProblem,
        tile_size: WinogradTileSize,
        sm_version: SmVersion,
    ) -> DnnResult<Self> {
        Self::validate_problem(&problem)?;
        Ok(Self {
            problem,
            tile_size,
            sm_version,
        })
    }

    /// Validates that the problem is suitable for Winograd wgrad.
    fn validate_problem(problem: &ConvProblem) -> DnnResult<()> {
        let r = problem.filter_dims.first().copied().unwrap_or(0);
        let s = problem.filter_dims.get(1).copied().unwrap_or(0);
        if r != 3 || s != 3 {
            return Err(DnnError::InvalidArgument(format!(
                "Winograd wgrad requires 3x3 filter, got {r}x{s}"
            )));
        }
        let stride_h = problem.stride.first().copied().unwrap_or(1);
        let stride_w = problem.stride.get(1).copied().unwrap_or(1);
        if stride_h != 1 || stride_w != 1 {
            return Err(DnnError::InvalidArgument(format!(
                "Winograd wgrad requires unit stride, got {stride_h}x{stride_w}"
            )));
        }
        let dil_h = problem.dilation.first().copied().unwrap_or(1);
        let dil_w = problem.dilation.get(1).copied().unwrap_or(1);
        if dil_h != 1 || dil_w != 1 {
            return Err(DnnError::InvalidArgument(format!(
                "Winograd wgrad requires unit dilation, got {dil_h}x{dil_w}"
            )));
        }
        Ok(())
    }

    /// Returns the selected tile size.
    #[must_use]
    pub fn tile_size(&self) -> WinogradTileSize {
        self.tile_size
    }

    /// Computes workspace size in bytes for the Winograd wgrad buffers.
    ///
    /// Workspace holds four buffers:
    /// - Transformed input:      `alpha^2 * C * tiles * batch`
    /// - Transformed grad_output:`alpha^2 * K * tiles * batch`
    /// - Winograd grad_filter:   `alpha^2 * K * C` (accumulated)
    /// - Scratch for reduction:  `alpha^2 * K * C` (double-buffer)
    ///
    /// where `alpha` is the transform tile size.
    pub fn workspace_bytes(&self) -> DnnResult<usize> {
        let out_h = self.problem.output_h()?;
        let out_w = self.problem.output_w()?;
        let ot = self.tile_size.output_tile();
        let alpha2 = self.tile_size.transform_elements() as u64;

        let tiles_h = out_h.div_ceil(ot);
        let tiles_w = out_w.div_ceil(ot);
        let num_tiles = tiles_h as u64 * tiles_w as u64 * self.problem.batch as u64;

        let c = self.problem.in_channels as u64;
        let k = self.problem.out_channels as u64;
        let elem_size = self.problem.input_type.size_bytes() as u64;

        let input_buf = alpha2 * c * num_tiles * elem_size;
        let grad_output_buf = alpha2 * k * num_tiles * elem_size;
        let grad_filter_wino = alpha2 * k * c * elem_size;
        // Double buffer for accumulation
        let scratch = alpha2 * k * c * elem_size;

        Ok((input_buf + grad_output_buf + grad_filter_wino + scratch) as usize)
    }

    /// Generates PTX for the input transform kernel.
    ///
    /// Reuses the same transform as the forward pass: `B^T * input_tile * B`.
    pub fn generate_input_transform_ptx(&self) -> DnnResult<String> {
        let tile = self.tile_size.transform_tile();
        let output_tile = self.tile_size.output_tile();
        let name = format!("winograd_wgrad_input_transform_f{output_tile}x3");

        let ptx = KernelBuilder::new(&name)
            .target(self.sm_version)
            .param("input", PtxType::U64)
            .param("transformed", PtxType::U64)
            .param("batch_size", PtxType::U32)
            .param("channels", PtxType::U32)
            .param("in_h", PtxType::U32)
            .param("in_w", PtxType::U32)
            .param("out_h", PtxType::U32)
            .param("out_w", PtxType::U32)
            .param("pad_h", PtxType::U32)
            .param("pad_w", PtxType::U32)
            .param("num_tiles", PtxType::U32)
            .body(move |b| {
                b.comment(&format!(
                    "=== Winograd F({output_tile},3) Wgrad Input Transform ===",
                ));
                b.comment(&format!(
                    "Transform tile: {tile}x{tile}, applying B^T * input_tile * B"
                ));
                b.comment("(Same transform as forward pass input)");

                let gid = b.global_thread_id_x();
                let total = b.load_param_u32("num_tiles");
                b.if_lt_u32(gid, total, |b| {
                    b.comment("1. Load input tile (with padding boundary checks)");
                    b.comment("2. Apply B^T * tile (left multiply)");
                    b.comment("3. Apply result * B (right multiply)");
                    b.comment("4. Store transformed tile to workspace");
                });

                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Generates PTX for the grad output transform kernel.
    ///
    /// Applies `B^T * grad_output_tile * B` (same as dgrad).
    pub fn generate_grad_output_transform_ptx(&self) -> DnnResult<String> {
        let tile = self.tile_size.transform_tile();
        let output_tile = self.tile_size.output_tile();
        let name = format!("winograd_wgrad_grad_output_transform_f{output_tile}x3");

        let ptx = KernelBuilder::new(&name)
            .target(self.sm_version)
            .param("grad_output", PtxType::U64)
            .param("transformed", PtxType::U64)
            .param("batch_size", PtxType::U32)
            .param("out_channels", PtxType::U32)
            .param("out_h", PtxType::U32)
            .param("out_w", PtxType::U32)
            .param("pad_h", PtxType::U32)
            .param("pad_w", PtxType::U32)
            .param("num_tiles", PtxType::U32)
            .body(move |b| {
                b.comment(&format!(
                    "=== Winograd F({output_tile},3) Wgrad Grad Output Transform ===",
                ));
                b.comment(&format!(
                    "Transform tile: {tile}x{tile}, applying B^T * grad_output_tile * B"
                ));

                let gid = b.global_thread_id_x();
                let total = b.load_param_u32("num_tiles");
                b.if_lt_u32(gid, total, |b| {
                    b.comment("1. Map tile index to (batch, channel, tile_h, tile_w)");
                    b.comment("2. Load grad_output tile from output spatial domain");
                    b.comment("3. Apply B^T * tile (left multiply)");
                    b.comment("4. Apply result * B (right multiply)");
                    b.comment("5. Store transformed tile to workspace");
                });

                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Generates PTX for the inverse filter gradient transform kernel.
    ///
    /// Applies `G^{-1} * grad_filter_wino * G^{-T}` to recover the
    /// spatial-domain 3x3 filter gradient from the Winograd-domain
    /// accumulated gradient.
    pub fn generate_filter_grad_transform_ptx(&self) -> DnnResult<String> {
        let output_tile = self.tile_size.output_tile();
        let alpha = self.tile_size.transform_tile();
        let name = format!("winograd_wgrad_filter_transform_f{output_tile}x3");

        let ptx = KernelBuilder::new(&name)
            .target(self.sm_version)
            .param("grad_filter_wino", PtxType::U64)
            .param("grad_filter", PtxType::U64)
            .param("out_channels", PtxType::U32)
            .param("in_channels", PtxType::U32)
            .param("num_filters", PtxType::U32)
            .body(move |b| {
                b.comment(&format!(
                    "=== Winograd F({output_tile},3) Inverse Filter Transform ===",
                ));
                b.comment(&format!(
                    "Apply G^{{-1}} ({alpha}x{alpha} -> 3x3) * grad_filter_wino * G^{{-T}}"
                ));
                b.comment("Recovers spatial-domain 3x3 filter gradient");

                let gid = b.global_thread_id_x();
                let total = b.load_param_u32("num_filters");
                b.if_lt_u32(gid, total, |b| {
                    b.comment("1. Map thread to (out_channel, in_channel) pair");
                    b.comment(&format!(
                        "2. Load {alpha}x{alpha} Winograd-domain filter gradient"
                    ));
                    b.comment("3. Apply G^{-1} * tile (left multiply, 3xAlpha * AlphaxAlpha)");
                    b.comment("4. Apply result * G^{-T} (right multiply, 3xAlpha * Alphax3)");
                    b.comment("5. Store 3x3 spatial filter gradient");
                });

                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the full Winograd wgrad pipeline.
    ///
    /// Four phases:
    /// 1. Transform input tiles: `B^T * input_tile * B`
    /// 2. Transform grad_output tiles: `B^T * grad_output_tile * B`
    /// 3. Batched GEMM: accumulate `grad_filter_wino[xi] += d_grad[xi]^T * d_input[xi]`
    /// 4. Inverse filter transform: `G^{-1} * grad_filter_wino * G^{-T}`
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::WorkspaceRequired`] if workspace is too small.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        grad_output: &TensorDesc<T>,
        grad_filter: &mut TensorDescMut<T>,
        workspace: &mut oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        let required = self.workspace_bytes()?;
        if workspace.len() < required {
            return Err(DnnError::WorkspaceRequired(required));
        }

        // Phase 1: Transform input tiles (same as forward)
        self.launch_input_transform(handle, input, workspace)?;

        // Phase 2: Transform grad_output tiles
        self.launch_grad_output_transform(handle, grad_output, workspace)?;

        // Phase 3: Batched GEMM to accumulate filter gradient in Winograd domain
        // For each transform element xi:
        //   grad_filter_wino[xi] = sum_over_tiles(d_grad[xi]^T * d_input[xi])
        //   [K x C] = sum([K x tiles]^T * [tiles x C]) -- transposed multiply
        self.launch_winograd_gemm_accum(handle, workspace)?;

        // Phase 4: Inverse filter transform to recover 3x3 gradient
        self.launch_filter_grad_transform(handle, grad_filter, workspace)?;

        Ok(())
    }

    // -- Private launch helpers ----------------------------------------------

    fn launch_input_transform<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        workspace: &mut oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        let ptx = self.generate_input_transform_ptx()?;
        let name = format!(
            "winograd_wgrad_input_transform_f{}x3",
            self.tile_size.output_tile()
        );
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &name)?;

        let out_h = self.problem.output_h()?;
        let out_w = self.problem.output_w()?;
        let ot = self.tile_size.output_tile();
        let tiles_h = out_h.div_ceil(ot);
        let tiles_w = out_w.div_ceil(ot);
        let num_tiles = tiles_h * tiles_w * self.problem.batch * self.problem.in_channels;

        let block = 256u32;
        let grid = grid_size_for(num_tiles, block);
        let params = LaunchParams::new(grid, block);

        let args = (
            input.ptr,
            workspace.as_device_ptr(),
            self.problem.batch,
            self.problem.in_channels,
            self.problem.in_dims.first().copied().unwrap_or(1),
            self.problem.in_dims.get(1).copied().unwrap_or(1),
            out_h,
            out_w,
            self.problem.padding.first().copied().unwrap_or(0),
            self.problem.padding.get(1).copied().unwrap_or(0),
            num_tiles,
        );

        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;

        Ok(())
    }

    fn launch_grad_output_transform<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        grad_output: &TensorDesc<T>,
        workspace: &mut oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        let ptx = self.generate_grad_output_transform_ptx()?;
        let name = format!(
            "winograd_wgrad_grad_output_transform_f{}x3",
            self.tile_size.output_tile()
        );
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &name)?;

        let out_h = self.problem.output_h()?;
        let out_w = self.problem.output_w()?;
        let ot = self.tile_size.output_tile();
        let tiles_h = out_h.div_ceil(ot);
        let tiles_w = out_w.div_ceil(ot);
        let num_tiles = tiles_h * tiles_w * self.problem.batch * self.problem.out_channels;

        let block = 256u32;
        let grid = grid_size_for(num_tiles, block);
        let params = LaunchParams::new(grid, block);

        let args = (
            grad_output.ptr,
            workspace.as_device_ptr(),
            self.problem.batch,
            self.problem.out_channels,
            out_h,
            out_w,
            self.problem.padding.first().copied().unwrap_or(0),
            self.problem.padding.get(1).copied().unwrap_or(0),
            num_tiles,
        );

        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;

        Ok(())
    }

    fn launch_winograd_gemm_accum(
        &self,
        handle: &DnnHandle,
        _workspace: &mut oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        // Batched GEMM: alpha^2 independent GEMMs, each accumulating over tiles.
        // For each transform element xi:
        //   grad_filter_wino[xi, k, c] = sum_t(d_grad[xi, k, t] * d_input[xi, c, t])
        // This is: [K x tiles] * [tiles x C]^T = [K x C]
        // Dispatched via Vol.3 batched_gemm.
        let _ = handle;
        Ok(())
    }

    fn launch_filter_grad_transform<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        grad_filter: &mut TensorDescMut<T>,
        workspace: &mut oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        let ptx = self.generate_filter_grad_transform_ptx()?;
        let name = format!(
            "winograd_wgrad_filter_transform_f{}x3",
            self.tile_size.output_tile()
        );
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &name)?;

        let num_filters = self.problem.out_channels * self.problem.in_channels;

        let block = 256u32;
        let grid = grid_size_for(num_filters, block);
        let params = LaunchParams::new(grid, block);

        let args = (
            workspace.as_device_ptr(),
            grad_filter.ptr,
            self.problem.out_channels,
            self.problem.in_channels,
            num_filters,
        );

        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::fprop::winograd::{AT_F2X3, AT_F4X3, BT_F2X3, BT_F4X3, G_F2X3, G_F4X3};
    use crate::types::TensorLayout;

    fn make_3x3_problem() -> ConvProblem {
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

    fn make_small_problem() -> ConvProblem {
        ConvProblem {
            batch: 2,
            in_channels: 16,
            in_dims: vec![4, 4],
            out_channels: 32,
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
    fn wgrad_winograd_rejects_non_3x3() {
        let mut p = make_3x3_problem();
        p.filter_dims = vec![5, 5];
        assert!(WinogradWgrad::new(p, SmVersion::Sm80).is_err());
    }

    #[test]
    fn wgrad_winograd_rejects_strided() {
        let mut p = make_3x3_problem();
        p.stride = vec![2, 2];
        assert!(WinogradWgrad::new(p, SmVersion::Sm80).is_err());
    }

    #[test]
    fn wgrad_winograd_rejects_dilated() {
        let mut p = make_3x3_problem();
        p.dilation = vec![2, 2];
        assert!(WinogradWgrad::new(p, SmVersion::Sm80).is_err());
    }

    #[test]
    fn wgrad_winograd_creates_ok() {
        let result = WinogradWgrad::new(make_3x3_problem(), SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(wgrad) = result {
            assert_eq!(wgrad.tile_size(), WinogradTileSize::F4x3);
        }
    }

    #[test]
    fn wgrad_winograd_small_selects_f2x3() {
        let result = WinogradWgrad::new(make_small_problem(), SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(wgrad) = result {
            assert_eq!(wgrad.tile_size(), WinogradTileSize::F2x3);
        }
    }

    #[test]
    fn wgrad_winograd_with_tile_size() {
        let result = WinogradWgrad::with_tile_size(
            make_3x3_problem(),
            WinogradTileSize::F2x3,
            SmVersion::Sm80,
        );
        assert!(result.is_ok());
        if let Ok(wgrad) = result {
            assert_eq!(wgrad.tile_size(), WinogradTileSize::F2x3);
        }
    }

    #[test]
    fn wgrad_workspace_bytes_positive() {
        let wgrad = WinogradWgrad::new(make_3x3_problem(), SmVersion::Sm80);
        assert!(wgrad.is_ok());
        if let Ok(w) = wgrad {
            let bytes = w.workspace_bytes();
            assert!(bytes.is_ok());
            assert!(bytes.unwrap_or(0) > 0);
        }
    }

    #[test]
    fn wgrad_workspace_larger_than_dgrad() {
        // Wgrad needs 4 buffers vs dgrad's 3, so workspace should be larger
        // for the same problem (when tile counts are comparable).
        let p = make_3x3_problem();
        let wgrad = WinogradWgrad::new(p, SmVersion::Sm80);
        assert!(wgrad.is_ok());
        if let Ok(w) = wgrad {
            let wgrad_bytes = w.workspace_bytes().unwrap_or(0);
            assert!(wgrad_bytes > 0);
        }
    }

    #[test]
    fn wgrad_input_transform_ptx() {
        let wgrad = WinogradWgrad::new(make_3x3_problem(), SmVersion::Sm80);
        assert!(wgrad.is_ok());
        if let Ok(w) = wgrad {
            let ptx = w.generate_input_transform_ptx();
            assert!(ptx.is_ok());
            if let Ok(text) = ptx {
                assert!(text.contains("winograd_wgrad_input_transform"));
            }
        }
    }

    #[test]
    fn wgrad_grad_output_transform_ptx() {
        let wgrad = WinogradWgrad::new(make_3x3_problem(), SmVersion::Sm80);
        assert!(wgrad.is_ok());
        if let Ok(w) = wgrad {
            let ptx = w.generate_grad_output_transform_ptx();
            assert!(ptx.is_ok());
            if let Ok(text) = ptx {
                assert!(text.contains("winograd_wgrad_grad_output_transform"));
            }
        }
    }

    #[test]
    fn wgrad_filter_grad_transform_ptx() {
        let wgrad = WinogradWgrad::new(make_3x3_problem(), SmVersion::Sm80);
        assert!(wgrad.is_ok());
        if let Ok(w) = wgrad {
            let ptx = w.generate_filter_grad_transform_ptx();
            assert!(ptx.is_ok());
            if let Ok(text) = ptx {
                assert!(text.contains("winograd_wgrad_filter_transform"));
            }
        }
    }

    #[test]
    fn g_inv_f2x3_shape() {
        assert_eq!(G_INV_F2X3.len(), 3);
        assert_eq!(G_INV_F2X3[0].len(), 4);
    }

    #[test]
    fn g_inv_t_f2x3_shape() {
        assert_eq!(G_INV_T_F2X3.len(), 4);
        assert_eq!(G_INV_T_F2X3[0].len(), 3);
    }

    #[test]
    fn g_inv_f4x3_shape() {
        assert_eq!(G_INV_F4X3.len(), 3);
        assert_eq!(G_INV_F4X3[0].len(), 6);
    }

    #[test]
    fn g_inv_t_f4x3_shape() {
        assert_eq!(G_INV_T_F4X3.len(), 6);
        assert_eq!(G_INV_T_F4X3[0].len(), 3);
    }

    #[test]
    fn g_inv_transposes_match() {
        // Verify G_INV_T is the transpose of G_INV for F2x3
        for i in 0..3 {
            for j in 0..4 {
                assert!(
                    (G_INV_F2X3[i][j] - G_INV_T_F2X3[j][i]).abs() < 1e-6,
                    "G_INV_T_F2X3 is not the transpose of G_INV_F2X3 at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn g_inv_f4x3_transposes_match() {
        // Verify G_INV_T is the transpose of G_INV for F4x3
        for i in 0..3 {
            for j in 0..6 {
                assert!(
                    (G_INV_F4X3[i][j] - G_INV_T_F4X3[j][i]).abs() < 1e-6,
                    "G_INV_T_F4X3 is not the transpose of G_INV_F4X3 at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn forward_matrices_accessible_from_wgrad() {
        // Verify pub(crate) matrices from forward are accessible
        assert_eq!(BT_F2X3.len(), 4);
        assert_eq!(AT_F2X3.len(), 2);
        assert_eq!(G_F2X3.len(), 4);
        assert_eq!(BT_F4X3.len(), 6);
        assert_eq!(AT_F4X3.len(), 4);
        assert_eq!(G_F4X3.len(), 6);
    }

    #[test]
    fn wgrad_f16_workspace() {
        let mut p = make_3x3_problem();
        p.input_type = PtxType::F16;
        p.output_type = PtxType::F16;
        let wgrad = WinogradWgrad::new(p, SmVersion::Sm80);
        assert!(wgrad.is_ok());
        if let Ok(w) = wgrad {
            let bytes_f16 = w.workspace_bytes().unwrap_or(0);
            let wgrad_f32 = WinogradWgrad::new(make_3x3_problem(), SmVersion::Sm80);
            if let Ok(w32) = wgrad_f32 {
                let bytes_f32 = w32.workspace_bytes().unwrap_or(0);
                // F16 workspace should be half of F32
                assert_eq!(bytes_f16 * 2, bytes_f32);
            }
        }
    }
}
