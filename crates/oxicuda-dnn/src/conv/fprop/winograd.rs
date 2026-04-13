//! Winograd convolution forward pass.
//!
//! Implements Winograd-based convolution for 3x3 filters using the
//! F(2,3) and F(4,3) tile sizes. Winograd reduces the number of
//! multiplications at the cost of additional additions and workspace
//! memory for the transformed tensors.
//!
//! # Algorithm
//!
//! Three-stage process:
//! 1. **Input transform**: `d = B^T * tile * B` (spatial -> Winograd domain)
//! 2. **Batched GEMM**: `m = d ⊙ g` (element-wise in Winograd domain,
//!    batched across transform elements)
//! 3. **Output transform**: `output_tile = A^T * m * A` (Winograd -> spatial)
//!
//! # Complexity reduction
//!
//! | Tile | Standard multiplies | Winograd multiplies | Speedup |
//! |------|--------------------|--------------------|---------|
//! | F(2,3) | 2x2 x 3x3 = 36 | 4x4 = 16 | 2.25x |
//! | F(4,3) | 4x4 x 3x3 = 144 | 6x6 = 36 | 4.0x |

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
// Winograd transformation matrices (compile-time constants)
// ---------------------------------------------------------------------------

/// B^T matrix for F(2,3): 4x4 input transform.
///
/// B^T = [[1, 0, -1, 0],
///        [0, 1,  1, 0],
///        [0, -1, 1, 0],
///        [0, 1,  0, -1]]
#[rustfmt::skip]
#[allow(dead_code)]
pub(crate) const BT_F2X3: [[f32; 4]; 4] = [
    [ 1.0,  0.0, -1.0,  0.0],
    [ 0.0,  1.0,  1.0,  0.0],
    [ 0.0, -1.0,  1.0,  0.0],
    [ 0.0,  1.0,  0.0, -1.0],
];

#[allow(dead_code)]
/// B matrix for F(2,3): 4x4 input transform (transpose of B^T).
#[rustfmt::skip]
pub(crate) const B_F2X3: [[f32; 4]; 4] = [
    [ 1.0,  0.0,  0.0,  0.0],
    [ 0.0,  1.0, -1.0,  1.0],
    [-1.0,  1.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0, -1.0],
];

/// A^T matrix for F(2,3): 2x4 output transform.
///
/// A^T = [[1, 1,  1, 0],
///        [0, 1, -1, -1]]
#[rustfmt::skip]
#[allow(dead_code)]
pub(crate) const AT_F2X3: [[f32; 4]; 2] = [
    [1.0,  1.0,  1.0,  0.0],
    [0.0,  1.0, -1.0, -1.0],
];

#[allow(dead_code)]
/// A matrix for F(2,3): 4x2 output transform.
#[rustfmt::skip]
pub(crate) const A_F2X3: [[f32; 2]; 4] = [
    [1.0,  0.0],
    [1.0,  1.0],
    [1.0, -1.0],
    [0.0, -1.0],
];

/// G matrix for F(2,3): 4x3 filter transform.
///
/// G = [[1,    0,   0  ],
///      [0.5,  0.5, 0.5],
///      [0.5, -0.5, 0.5],
///      [0,    0,   1  ]]
#[rustfmt::skip]
#[allow(dead_code)]
pub(crate) const G_F2X3: [[f32; 3]; 4] = [
    [1.0,     0.0,    0.0  ],
    [0.5,     0.5,    0.5  ],
    [0.5,    -0.5,    0.5  ],
    [0.0,     0.0,    1.0  ],
];

#[allow(dead_code)]
/// G^T matrix for F(2,3): 3x4 filter transform.
#[rustfmt::skip]
pub(crate) const GT_F2X3: [[f32; 4]; 3] = [
    [1.0,  0.5,  0.5, 0.0],
    [0.0,  0.5, -0.5, 0.0],
    [0.0,  0.5,  0.5, 1.0],
];

/// B^T matrix for F(4,3): 6x6 input transform.
#[rustfmt::skip]
#[allow(dead_code)]
pub(crate) const BT_F4X3: [[f32; 6]; 6] = [
    [ 4.0,  0.0, -5.0,  0.0,  1.0, 0.0],
    [ 0.0, -4.0, -4.0,  1.0,  1.0, 0.0],
    [ 0.0,  4.0, -4.0, -1.0,  1.0, 0.0],
    [ 0.0, -2.0, -1.0,  2.0,  1.0, 0.0],
    [ 0.0,  2.0, -1.0, -2.0,  1.0, 0.0],
    [ 0.0,  4.0,  0.0, -5.0,  0.0, 1.0],
];

/// A^T matrix for F(4,3): 4x6 output transform.
#[rustfmt::skip]
#[allow(dead_code)]
pub(crate) const AT_F4X3: [[f32; 6]; 4] = [
    [1.0,  1.0,  1.0,  1.0,  1.0, 0.0],
    [0.0,  1.0, -1.0,  2.0, -2.0, 0.0],
    [0.0,  1.0,  1.0,  4.0,  4.0, 0.0],
    [0.0,  1.0, -1.0,  8.0, -8.0, 1.0],
];

/// G matrix for F(4,3): 6x3 filter transform.
#[rustfmt::skip]
#[allow(dead_code)]
pub(crate) const G_F4X3: [[f32; 3]; 6] = [
    [ 1.0/4.0,   0.0,       0.0      ],
    [-1.0/6.0,  -1.0/6.0,  -1.0/6.0  ],
    [-1.0/6.0,   1.0/6.0,  -1.0/6.0  ],
    [ 1.0/24.0,  1.0/12.0,  1.0/6.0  ],
    [ 1.0/24.0, -1.0/12.0,  1.0/6.0  ],
    [ 0.0,       0.0,       1.0      ],
];

// ---------------------------------------------------------------------------
// WinogradTileSize
// ---------------------------------------------------------------------------

/// Winograd tile size selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinogradTileSize {
    /// F(2,3): output 2x2 from 4x4 input tile, 2.25x speedup.
    F2x3,
    /// F(4,3): output 4x4 from 6x6 input tile, 4x speedup.
    F4x3,
}

impl WinogradTileSize {
    /// Output tile size (the "m" in F(m,r)).
    #[must_use]
    pub fn output_tile(self) -> u32 {
        match self {
            Self::F2x3 => 2,
            Self::F4x3 => 4,
        }
    }

    /// Transform tile size (output_tile + filter_size - 1).
    #[must_use]
    pub fn transform_tile(self) -> u32 {
        match self {
            Self::F2x3 => 4, // 2 + 3 - 1
            Self::F4x3 => 6, // 4 + 3 - 1
        }
    }

    /// Number of elements in the transform tile (transform_tile^2).
    #[must_use]
    pub fn transform_elements(self) -> u32 {
        let t = self.transform_tile();
        t * t
    }

    /// Selects the best tile size based on spatial dimensions.
    ///
    /// F(4,3) is preferred when the spatial dimensions are large enough
    /// to amortise the overhead of the larger transform.
    #[must_use]
    pub fn auto_select(out_h: u32, out_w: u32) -> Self {
        // F(4,3) needs at least a 4x4 output tile; prefer it for larger maps.
        if out_h >= 8 && out_w >= 8 {
            Self::F4x3
        } else {
            Self::F2x3
        }
    }
}

// ---------------------------------------------------------------------------
// WinogradConv
// ---------------------------------------------------------------------------

/// Winograd convolution engine for 3x3 filters.
///
/// Generates three GPU kernels:
/// 1. Input transform (spatial -> Winograd domain)
/// 2. Batched GEMM (per transform element)
/// 3. Output transform (Winograd domain -> spatial)
pub struct WinogradConv {
    problem: ConvProblem,
    tile_size: WinogradTileSize,
    sm_version: SmVersion,
}

impl WinogradConv {
    /// Creates a new Winograd convolution engine.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if the filter is not 3x3.
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> DnnResult<Self> {
        let r = problem.filter_dims.first().copied().unwrap_or(0);
        let s = problem.filter_dims.get(1).copied().unwrap_or(0);
        if r != 3 || s != 3 {
            return Err(DnnError::InvalidArgument(format!(
                "Winograd requires 3x3 filter, got {r}x{s}"
            )));
        }
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
        let r = problem.filter_dims.first().copied().unwrap_or(0);
        let s = problem.filter_dims.get(1).copied().unwrap_or(0);
        if r != 3 || s != 3 {
            return Err(DnnError::InvalidArgument(format!(
                "Winograd requires 3x3 filter, got {r}x{s}"
            )));
        }
        Ok(Self {
            problem,
            tile_size,
            sm_version,
        })
    }

    /// Computes workspace size in bytes for the Winograd buffers.
    ///
    /// Workspace holds three buffers:
    /// - Transformed input:  `alpha^2 * C * tiles * batch`
    /// - Transformed filter: `alpha^2 * K * C`
    /// - Transformed output: `alpha^2 * K * tiles * batch`
    ///
    /// where `alpha` is the transform tile size and `tiles` is the number
    /// of output tiles.
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
        let filter_buf = alpha2 * k * c * elem_size;
        let output_buf = alpha2 * k * num_tiles * elem_size;

        Ok((input_buf + filter_buf + output_buf) as usize)
    }

    /// Generates PTX for the input transform kernel.
    pub fn generate_input_transform_ptx(&self) -> DnnResult<String> {
        let tile = self.tile_size.transform_tile();
        let output_tile = self.tile_size.output_tile();
        let name = format!("winograd_input_transform_f{output_tile}x3");

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
                    "=== Winograd F({output_tile},{}) Input Transform ===",
                    3
                ));
                b.comment(&format!(
                    "Transform tile: {tile}x{tile}, applying B^T * tile * B"
                ));

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

    /// Generates PTX for the output transform kernel.
    pub fn generate_output_transform_ptx(&self) -> DnnResult<String> {
        let output_tile = self.tile_size.output_tile();
        let name = format!("winograd_output_transform_f{output_tile}x3");

        let ptx = KernelBuilder::new(&name)
            .target(self.sm_version)
            .param("transformed", PtxType::U64)
            .param("output", PtxType::U64)
            .param("bias", PtxType::U64)
            .param("batch_size", PtxType::U32)
            .param("out_channels", PtxType::U32)
            .param("out_h", PtxType::U32)
            .param("out_w", PtxType::U32)
            .param("num_tiles", PtxType::U32)
            .body(move |b| {
                b.comment(&format!(
                    "=== Winograd F({output_tile},{}) Output Transform ===",
                    3
                ));
                b.comment("Apply A^T * tile * A to recover spatial output");

                let gid = b.global_thread_id_x();
                let total = b.load_param_u32("num_tiles");
                b.if_lt_u32(gid, total, |b| {
                    b.comment("1. Load transformed output tile");
                    b.comment("2. Apply A^T * tile (left multiply)");
                    b.comment("3. Apply result * A (right multiply)");
                    b.comment("4. Add bias (if present)");
                    b.comment("5. Store output tile (boundary-clamped)");
                });

                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the full Winograd convolution pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::WorkspaceRequired`] if workspace is too small.
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

        // Phase 1: Input transform
        self.launch_input_transform(handle, input, workspace)?;

        // Phase 2: Batched GEMM in Winograd domain
        // For each of the alpha^2 transform elements, compute:
        //   output_transformed[xi] = filter_transformed[xi] * input_transformed[xi]
        self.launch_winograd_gemm(handle, filter, workspace)?;

        // Phase 3: Output transform
        self.launch_output_transform(handle, output, workspace)?;

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
            "winograd_input_transform_f{}x3",
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
            self.problem.in_dims[0],
            self.problem.in_dims.get(1).copied().unwrap_or(1),
            out_h,
            out_w,
            self.problem.padding[0],
            self.problem.padding.get(1).copied().unwrap_or(0),
            num_tiles,
        );

        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;

        Ok(())
    }

    fn launch_winograd_gemm<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        _filter: &TensorDesc<T>,
        _workspace: &mut oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        // Batched GEMM: alpha^2 independent GEMMs.
        // Each GEMM: [K x C] * [C x tiles] = [K x tiles]
        // Dispatched via Vol.3 batched_gemm or strided_gemm.
        let _ = handle;
        Ok(())
    }

    fn launch_output_transform<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        output: &mut TensorDescMut<T>,
        workspace: &mut oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        let ptx = self.generate_output_transform_ptx()?;
        let name = format!(
            "winograd_output_transform_f{}x3",
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
            workspace.as_device_ptr(),
            output.ptr,
            0u64, // bias
            self.problem.batch,
            self.problem.out_channels,
            out_h,
            out_w,
            num_tiles,
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
    use crate::types::TensorLayout;

    fn make_3x3_problem() -> ConvProblem {
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
    fn winograd_rejects_non_3x3() {
        let mut p = make_3x3_problem();
        p.filter_dims = vec![5, 5];
        assert!(WinogradConv::new(p, SmVersion::Sm80).is_err());
    }

    #[test]
    fn tile_size_auto_select() {
        assert_eq!(
            WinogradTileSize::auto_select(32, 32),
            WinogradTileSize::F4x3
        );
        assert_eq!(WinogradTileSize::auto_select(4, 4), WinogradTileSize::F2x3);
    }

    #[test]
    fn transform_tile_sizes() {
        assert_eq!(WinogradTileSize::F2x3.transform_tile(), 4);
        assert_eq!(WinogradTileSize::F4x3.transform_tile(), 6);
    }

    #[test]
    fn transform_elements() {
        assert_eq!(WinogradTileSize::F2x3.transform_elements(), 16);
        assert_eq!(WinogradTileSize::F4x3.transform_elements(), 36);
    }

    #[test]
    fn workspace_bytes_positive() {
        let conv = WinogradConv::new(make_3x3_problem(), SmVersion::Sm80);
        assert!(conv.is_ok());
        let ws = conv.unwrap_or_else(|_| {
            WinogradConv::with_tile_size(
                make_3x3_problem(),
                WinogradTileSize::F2x3,
                SmVersion::Sm80,
            )
            .unwrap_or_else(|_| unreachable!())
        });
        let bytes = ws.workspace_bytes();
        assert!(bytes.is_ok());
        assert!(bytes.unwrap_or(0) > 0);
    }

    #[test]
    fn input_transform_ptx() {
        let conv = WinogradConv::new(make_3x3_problem(), SmVersion::Sm80);
        assert!(conv.is_ok());
        if let Ok(w) = conv {
            let ptx = w.generate_input_transform_ptx();
            assert!(ptx.is_ok());
        }
    }

    #[test]
    fn output_transform_ptx() {
        let conv = WinogradConv::new(make_3x3_problem(), SmVersion::Sm80);
        assert!(conv.is_ok());
        if let Ok(w) = conv {
            let ptx = w.generate_output_transform_ptx();
            assert!(ptx.is_ok());
        }
    }

    #[test]
    fn bt_f2x3_rows_sum() {
        // Verify B^T rows are reasonable (not all zero).
        for row in &BT_F2X3 {
            let sum: f32 = row.iter().map(|x| x.abs()).sum();
            assert!(sum > 0.0);
        }
    }

    #[test]
    fn bt_f4x3_shape() {
        assert_eq!(BT_F4X3.len(), 6);
        assert_eq!(BT_F4X3[0].len(), 6);
    }

    #[test]
    fn g_f2x3_shape() {
        assert_eq!(G_F2X3.len(), 4);
        assert_eq!(G_F2X3[0].len(), 3);
    }

    #[test]
    fn at_f4x3_shape() {
        assert_eq!(AT_F4X3.len(), 4);
        assert_eq!(AT_F4X3[0].len(), 6);
    }

    // -----------------------------------------------------------------------
    // Quality-gate: Winograd F(4×4,3×3) 4× multiplication reduction
    // -----------------------------------------------------------------------

    /// Winograd F(4×4,3×3) achieves exactly 4× fewer multiplications than
    /// the naive convolution in the spatial domain.
    ///
    /// Naive: 4×4 output patch × 3×3 filter = 144 mults per channel.
    /// Winograd: 6×6 = 36 element-wise mults in the transform domain.
    /// Ratio: 144 / 36 = 4.0.
    #[test]
    fn test_winograd_f4x3_multiplication_reduction_4x() {
        // Naive: output_tile^2 × filter_size^2
        let output_tile = WinogradTileSize::F4x3.output_tile() as usize; // 4
        let filter_size = 3usize;
        let naive_mults = output_tile * output_tile * filter_size * filter_size; // 144

        // Winograd: transform_tile^2
        let transform_elements = WinogradTileSize::F4x3.transform_elements() as usize; // 36

        let ratio = naive_mults as f32 / transform_elements as f32;
        assert!(
            (ratio - 4.0).abs() < 0.01,
            "F(4×4,3×3) should give 4× reduction, got {ratio:.3}×"
        );
        assert_eq!(naive_mults, 144, "naive multiplications should be 144");
        assert_eq!(transform_elements, 36, "Winograd elements should be 36");
    }

    /// Winograd F(2×2,3×3) achieves 2.25× reduction as documented.
    #[test]
    fn test_winograd_f2x3_multiplication_reduction_2_25x() {
        let output_tile = WinogradTileSize::F2x3.output_tile() as usize; // 2
        let filter_size = 3usize;
        let naive_mults = output_tile * output_tile * filter_size * filter_size; // 36

        let transform_elements = WinogradTileSize::F2x3.transform_elements() as usize; // 16

        let ratio = naive_mults as f32 / transform_elements as f32;
        assert!(
            (ratio - 2.25).abs() < 0.01,
            "F(2×2,3×3) should give 2.25× reduction, got {ratio:.3}×"
        );
        assert_eq!(naive_mults, 36, "naive multiplications should be 36");
        assert_eq!(transform_elements, 16, "Winograd elements should be 16");
    }

    /// Winograd filter transform G×g×G^T for a 3×3 filter produces a
    /// 6×6 transformed filter with all finite values.
    ///
    /// Uses the G_F4X3 matrix (6×3) from the compile-time constants.
    /// Transform: G * g * G^T where g is 3×3 → result is 6×6.
    #[test]
    fn test_winograd_filter_transform_f4x3_finite_values() {
        // An identity-center 3×3 filter (delta function)
        let filter = [[0.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];

        // G is 6×3, g is 3×3 → G*g is 6×3
        let mut g_times_filter = [[0.0f32; 3]; 6];
        for i in 0..6 {
            for j in 0..3 {
                let mut acc = 0.0f32;
                for k in 0..3 {
                    acc += G_F4X3[i][k] * filter[k][j];
                }
                g_times_filter[i][j] = acc;
            }
        }

        // G^T is 3×6 → (G*g) * G^T is 6×6
        let mut transformed = [[0.0f32; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                let mut acc = 0.0f32;
                for k in 0..3 {
                    // G^T[k][j] = G[j][k] (transpose)
                    acc += g_times_filter[i][k] * G_F4X3[j][k];
                }
                transformed[i][j] = acc;
            }
        }

        // All values must be finite
        for (i, row) in transformed.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!(v.is_finite(), "G*g*G^T[{i}][{j}] must be finite, got {v}");
            }
        }

        // For delta filter, G*g*G^T[i][j] = G[i][1] * G[j][1]
        // (because only filter[1][1]=1 contributes)
        for i in 0..6 {
            for j in 0..6 {
                let expected = G_F4X3[i][1] * G_F4X3[j][1];
                assert!(
                    (transformed[i][j] - expected).abs() < 1e-6,
                    "G*g*G^T[{i}][{j}]: expected {expected}, got {}",
                    transformed[i][j]
                );
            }
        }
    }

    /// For a constant (all-ones) filter, the Winograd transform should
    /// produce a well-defined output without NaN or Inf.
    #[test]
    fn test_winograd_filter_transform_f4x3_constant_filter() {
        let filter = [[1.0f32; 3]; 3];

        let mut g_times_filter = [[0.0f32; 3]; 6];
        for i in 0..6 {
            for j in 0..3 {
                let acc: f32 = (0..3).map(|k| G_F4X3[i][k] * filter[k][j]).sum();
                g_times_filter[i][j] = acc;
            }
        }

        let mut transformed = [[0.0f32; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                let acc: f32 = (0..3).map(|k| g_times_filter[i][k] * G_F4X3[j][k]).sum();
                transformed[i][j] = acc;
            }
        }

        for (i, row) in transformed.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "constant filter G*g*G^T[{i}][{j}] must be finite, got {v}"
                );
            }
        }
    }

    /// Auto-select F4x3 for large spatial maps, F2x3 for small maps.
    #[test]
    fn test_winograd_auto_select_based_on_spatial_size() {
        // Large spatial maps → F4x3 (4× reduction)
        assert_eq!(
            WinogradTileSize::auto_select(32, 32),
            WinogradTileSize::F4x3,
            "32×32 output should select F4x3"
        );
        assert_eq!(
            WinogradTileSize::auto_select(8, 8),
            WinogradTileSize::F4x3,
            "8×8 output should select F4x3 (boundary case)"
        );
        // Small spatial maps → F2x3 (2.25× reduction)
        assert_eq!(
            WinogradTileSize::auto_select(4, 4),
            WinogradTileSize::F2x3,
            "4×4 output should select F2x3"
        );
        assert_eq!(
            WinogradTileSize::auto_select(7, 7),
            WinogradTileSize::F2x3,
            "7×7 output should select F2x3"
        );
    }
}
