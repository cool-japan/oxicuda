//! Winograd convolution forward pass — F(2x2, 3x3), NCHW, FP32.
//!
//! Winograd trades multiplications for additions. A 2x2 output tile from a 3x3
//! filter costs `2*2*3*3 = 36` multiplies directly, but only `4*4 = 16` in the
//! transform domain — a 2.25x reduction in the multiply count that dominates a
//! 3x3 stride-1 layer.
//!
//! # Pipeline
//!
//! With `alpha = 4`, `P = N * tiles_h * tiles_w` output tiles and `e` in
//! `0..16` indexing the transform domain:
//!
//! | Stage | Kernel | Work |
//! |---|---|---|
//! | 1 | `kernels::input_transform_ptx` | `V[e][c][p] = (B^T d B)[e]` |
//! | 2 | `kernels::filter_transform_ptx` | `U[e][k][c] = (G g G^T)[e]` |
//! | 3 | `kernels::batched_gemm_ptx` | `M[e] = U[e] (KxC) * V[e] (CxP)` |
//! | 4 | `kernels::output_transform_ptx` | `Y = A^T m A`, `+ bias`, store |
//!
//! All four run on [`DnnHandle::stream`] — stage 3 is this crate's own tiled
//! kernel rather than a BLAS dispatch, so there is no second stream and no
//! cross-stream ordering to get wrong (see `DnnHandle::synchronize_all`'s doc
//! comment for what that bug looks like when it happens).
//!
//! # Supported region
//!
//! [`WinogradConv::supports`] is the single authority, and
//! `is_winograd_eligible` (in `super::super::algo_select`) defers to it:
//!
//! * 2-D NCHW, FP32 in and out
//! * 3x3 filter, stride 1, dilation 1, `groups == 1`
//! * padding 0 or 1 per spatial axis
//!
//! Anything else must use another engine. `H`/`W` need not be even: partial
//! output tiles are computed in full and stored under a per-element boundary
//! guard.
//!
//! # Numerics
//!
//! Winograd is *not* bit-identical to a direct convolution: the transforms
//! reassociate the sum and introduce halves, so cancellation differs. The
//! contract is a **relative L2 error below `1e-4`** against an `f64` direct
//! reference; the error measured by `gpu_tests::conv_winograd` on an RTX A4000
//! is `9.7e-8` to `1.7e-7` across the edge-case sweep and up to `1.4e-6`
//! against `ImplicitGemmConv` at `C = 512`. Callers needing bit-reproducibility
//! against the direct path must not use this engine.
//!
//! # Dispatch gate
//!
//! Whether [`conv_forward`](super::super::api::conv_forward) actually routes
//! here is decided by
//! [`winograd_forward_implemented`](super::super::algo_select::winograd_forward_implemented)
//! together with `algo_select`'s profitability threshold. Both are now
//! *performance* decisions rather than correctness ones: measured on an RTX
//! A4000, this engine is 1.5x to 3.8x faster than `ImplicitGemmConv` above
//! ~2e7 GEMM FLOPs and slower below ~5e6 (four kernel launches against one).
//! See `WINOGRAD_FLOP_THRESHOLD` for the full table.

pub(crate) mod kernels;
pub(crate) mod matrices;

use oxicuda_blas::GpuFloat;
use oxicuda_launch::{Dim3, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::kernel_cache::cache_key;
use crate::types::{TensorDesc, TensorDescMut, TensorLayout};

use super::super::descriptor::ConvProblem;

// ---------------------------------------------------------------------------
// WinogradTileSize
// ---------------------------------------------------------------------------

/// Winograd tile size selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinogradTileSize {
    /// F(2,3): a 2x2 output tile from a 4x4 input tile — 2.25x fewer multiplies.
    F2x3,
    /// F(4,3): a 4x4 output tile from a 6x6 input tile — 4x fewer multiplies.
    ///
    /// **No forward kernels**: the larger transform's coefficients
    /// (`1/24`, `1/12`, ...) also amplify FP32 round-off enough that it needs
    /// its own error budget, so it is deliberately not enabled by inheritance
    /// from F(2,3). [`WinogradTileSize::forward_supported`] reports `false`
    /// and [`WinogradConv::with_tile_size`] rejects it. The matrices and the
    /// tile arithmetic are retained because the dgrad/wgrad modules use them.
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

    /// Transform tile size (`output_tile + filter_size - 1`).
    #[must_use]
    pub fn transform_tile(self) -> u32 {
        match self {
            Self::F2x3 => 4, // 2 + 3 - 1
            Self::F4x3 => 6, // 4 + 3 - 1
        }
    }

    /// Number of elements in the transform tile (`transform_tile^2`).
    #[must_use]
    pub fn transform_elements(self) -> u32 {
        let t = self.transform_tile();
        t * t
    }

    /// Whether this crate has real *forward* kernels for this tile size.
    ///
    /// Only F(2,3) does. This is a capability fact about the emitters in
    /// `kernels`, not a heuristic — [`WinogradConv::with_tile_size`] refuses
    /// anything that returns `false` rather than launching kernels that do not
    /// exist.
    #[must_use]
    pub const fn forward_supported(self) -> bool {
        matches!(self, Self::F2x3)
    }

    /// Selects the theoretically-best tile size for a given output extent,
    /// ignoring which tile sizes actually have kernels.
    ///
    /// Retained for the dgrad/wgrad Winograd planners, which use it for their
    /// own workspace arithmetic. The **forward** engine does not call this —
    /// see [`Self::best_forward`], which only ever returns a tile size with
    /// real kernels.
    #[must_use]
    pub fn auto_select(out_h: u32, out_w: u32) -> Self {
        // F(4,3) needs at least a 4x4 output tile; prefer it for larger maps.
        if out_h >= 8 && out_w >= 8 {
            Self::F4x3
        } else {
            Self::F2x3
        }
    }

    /// Selects the best tile size **that has forward kernels**.
    ///
    /// Currently constant: F(2,3) is the only implemented forward tile size.
    /// Written as a filter over [`Self::auto_select`] rather than a bare
    /// `F2x3` so that adding F(4,3) kernels is a one-line change here and the
    /// preference order stays in one place.
    #[must_use]
    pub fn best_forward(out_h: u32, out_w: u32) -> Self {
        let preferred = Self::auto_select(out_h, out_w);
        if preferred.forward_supported() {
            preferred
        } else {
            Self::F2x3
        }
    }
}

// ---------------------------------------------------------------------------
// WinogradGeometry
// ---------------------------------------------------------------------------

/// Number of transform-domain positions for F(2,3): `alpha^2 = 16`.
const F2X3_POSITIONS: u32 = 4 * 4;

/// Derived launch geometry and workspace partitioning for one F(2,3) problem.
///
/// Computed once per [`WinogradConv::execute`] call and shared by all four
/// launches, so the four kernels cannot disagree about the tile count or the
/// workspace offsets.
#[derive(Debug, Clone, Copy)]
struct WinogradGeometry {
    in_channels: u32,
    out_channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    pad_h: u32,
    pad_w: u32,
    tiles_h: u32,
    tiles_w: u32,
    /// `P = batch * tiles_h * tiles_w`: the flattened Winograd tile index.
    tile_count: u32,
    /// Elements per transform-domain plane of the transformed input (`C * P`).
    v_plane: u32,
    /// Elements per plane of the transformed filter (`K * C`).
    u_plane: u32,
    /// Elements per plane of the transformed output (`K * P`).
    m_plane: u32,
}

impl WinogradGeometry {
    /// Byte offset of the transformed **input** region within the workspace.
    const fn v_offset(&self) -> u64 {
        0
    }

    /// Byte offset of the transformed **filter** region.
    const fn u_offset(&self) -> u64 {
        (F2X3_POSITIONS as u64) * (self.v_plane as u64) * 4
    }

    /// Byte offset of the transformed **output** region.
    const fn m_offset(&self) -> u64 {
        self.u_offset() + (F2X3_POSITIONS as u64) * (self.u_plane as u64) * 4
    }

    /// Total workspace requirement in bytes.
    const fn workspace_bytes(&self) -> usize {
        (self.m_offset() + (F2X3_POSITIONS as u64) * (self.m_plane as u64) * 4) as usize
    }
}

// ---------------------------------------------------------------------------
// WinogradConv
// ---------------------------------------------------------------------------

/// Winograd F(2x2, 3x3) forward convolution engine.
///
/// See the module documentation for the pipeline, the supported problem
/// region, and the numerical contract.
pub struct WinogradConv {
    problem: ConvProblem,
    tile_size: WinogradTileSize,
    sm_version: SmVersion,
}

impl WinogradConv {
    /// Returns `true` if this engine can compute `problem`.
    ///
    /// This is the single authority on the supported region — the algorithm
    /// selector defers to it, so a shape that reaches
    /// [`execute`](Self::execute) through the dispatcher is guaranteed to be
    /// one the kernels handle. It does **not** consider profitability; that is
    /// [`select_algorithm`](super::super::algo_select::select_algorithm)'s job.
    #[must_use]
    pub fn supports(problem: &ConvProblem) -> bool {
        problem.layout == TensorLayout::Nchw
            && problem.input_type == PtxType::F32
            && problem.output_type == PtxType::F32
            && problem.in_dims.len() == 2
            && problem.filter_dims.len() == 2
            && problem.filter_dims[0] == 3
            && problem.filter_dims[1] == 3
            && problem.groups == 1
            && problem.stride.len() == 2
            && problem.stride.iter().all(|&v| v == 1)
            && problem.dilation.len() == 2
            && problem.dilation.iter().all(|&v| v == 1)
            && problem.padding.len() == 2
            && problem.padding.iter().all(|&v| v <= 1)
    }

    /// Creates a Winograd engine for `problem`, choosing the best tile size
    /// that has forward kernels.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::UnsupportedOperation`] if [`Self::supports`] is
    /// `false` for `problem`, naming the specific constraint that failed.
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> DnnResult<Self> {
        Self::validate_problem(&problem)?;
        let out_h = problem.output_h()?;
        let out_w = problem.output_w()?;
        let tile_size = WinogradTileSize::best_forward(out_h, out_w);
        Ok(Self {
            problem,
            tile_size,
            sm_version,
        })
    }

    /// Creates a Winograd engine with an explicit tile size.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], plus [`DnnError::UnsupportedOperation`] if
    /// `tile_size` has no forward kernels
    /// ([`WinogradTileSize::forward_supported`]).
    pub fn with_tile_size(
        problem: ConvProblem,
        tile_size: WinogradTileSize,
        sm_version: SmVersion,
    ) -> DnnResult<Self> {
        Self::validate_problem(&problem)?;
        if !tile_size.forward_supported() {
            return Err(DnnError::UnsupportedOperation(format!(
                "Winograd forward is implemented for F(2,3) only; {tile_size:?} has no kernels"
            )));
        }
        Ok(Self {
            problem,
            tile_size,
            sm_version,
        })
    }

    /// The tile size this engine will use.
    #[must_use]
    pub fn tile_size(&self) -> WinogradTileSize {
        self.tile_size
    }

    /// Computes the workspace size in bytes for the three transform buffers.
    ///
    /// ```text
    /// transformed input  : 16 * C * P
    /// transformed filter : 16 * K * C
    /// transformed output : 16 * K * P
    /// ```
    ///
    /// where `P = N * ceil(out_h/2) * ceil(out_w/2)`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] if the output extent cannot be
    /// derived, or if any transform buffer would exceed the kernels' 32-bit
    /// element indexing.
    pub fn workspace_bytes(&self) -> DnnResult<usize> {
        Ok(self.geometry()?.workspace_bytes())
    }

    /// Rejects a problem outside the engine's supported region, with a message
    /// that says which constraint failed rather than a bare "unsupported".
    fn validate_problem(problem: &ConvProblem) -> DnnResult<()> {
        if Self::supports(problem) {
            return Ok(());
        }
        let r = problem.filter_dims.first().copied().unwrap_or(0);
        let s = problem.filter_dims.get(1).copied().unwrap_or(0);
        Err(DnnError::UnsupportedOperation(format!(
            "Winograd F(2x2,3x3) forward requires 2-D NCHW FP32, a 3x3 filter, \
             stride 1, dilation 1, groups 1 and padding <= 1; got {r}x{s} filter, \
             stride {:?}, dilation {:?}, padding {:?}, groups {}, layout {:?}, \
             element type {}",
            problem.stride,
            problem.dilation,
            problem.padding,
            problem.groups,
            problem.layout,
            problem.input_type
        )))
    }

    /// Derives the launch geometry and workspace partitioning.
    fn geometry(&self) -> DnnResult<WinogradGeometry> {
        let out_h = self.problem.output_h()?;
        let out_w = self.problem.output_w()?;
        let ot = self.tile_size.output_tile();
        let tiles_h = out_h.div_ceil(ot);
        let tiles_w = out_w.div_ceil(ot);

        let batch = self.problem.batch;
        // `batch` is folded into `tile_count`; the kernels recover `n` by
        // decomposing the flat tile index, so it is never passed separately.
        let in_channels = self.problem.in_channels;
        let out_channels = self.problem.out_channels;

        let tile_count_u64 = u64::from(batch) * u64::from(tiles_h) * u64::from(tiles_w);
        let tile_count = fits_u32(tile_count_u64, "Winograd tile count")?;

        let v_plane = fits_u32(
            u64::from(in_channels) * tile_count_u64,
            "Winograd transformed-input plane",
        )?;
        let u_plane = fits_u32(
            u64::from(out_channels) * u64::from(in_channels),
            "Winograd transformed-filter plane",
        )?;
        let m_plane = fits_u32(
            u64::from(out_channels) * tile_count_u64,
            "Winograd transformed-output plane",
        )?;

        Ok(WinogradGeometry {
            in_channels,
            out_channels,
            in_h: self.problem.in_dims[0],
            in_w: self.problem.in_dims[1],
            out_h,
            out_w,
            pad_h: self.problem.padding[0],
            pad_w: self.problem.padding[1],
            tiles_h,
            tiles_w,
            tile_count,
            v_plane,
            u_plane,
            m_plane,
        })
    }

    /// Executes the Winograd convolution without a bias epilogue.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::WorkspaceRequired`] with the exact byte count when
    /// `workspace` is too small, [`DnnError::UnsupportedOperation`] for a
    /// non-FP32 element type, and any PTX-generation / launch error from the
    /// four stages.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        output: &mut TensorDescMut<T>,
        workspace: &mut DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        self.execute_with_bias(handle, input, filter, None, output, workspace)
    }

    /// Executes the Winograd convolution with an optional per-output-channel
    /// bias, added inside the output transform (no extra pass over the
    /// output tensor).
    ///
    /// # Errors
    ///
    /// As [`Self::execute`].
    pub fn execute_with_bias<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        bias: Option<&TensorDesc<T>>,
        output: &mut TensorDescMut<T>,
        workspace: &mut DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        if T::PTX_TYPE != PtxType::F32 {
            return Err(DnnError::UnsupportedOperation(format!(
                "Winograd F(2x2,3x3) forward is FP32-only; got {}",
                T::PTX_TYPE
            )));
        }
        let geom = self.geometry()?;
        let required = geom.workspace_bytes();
        if workspace.len() < required {
            return Err(DnnError::WorkspaceRequired(required));
        }

        let base = workspace.as_device_ptr();
        let v_ptr = base.wrapping_add(geom.v_offset());
        let u_ptr = base.wrapping_add(geom.u_offset());
        let m_ptr = base.wrapping_add(geom.m_offset());

        self.launch_input_transform(handle, input.ptr, v_ptr, &geom)?;
        self.launch_filter_transform(handle, filter.ptr, u_ptr, &geom)?;
        self.launch_gemm(handle, u_ptr, v_ptr, m_ptr, &geom)?;
        self.launch_output_transform(handle, m_ptr, output.ptr, bias, &geom)?;

        Ok(())
    }

    // -- Launch helpers ------------------------------------------------------

    /// Stage 1: `V = B^T d B`, one thread per `(in-channel, tile)` pair.
    fn launch_input_transform(
        &self,
        handle: &DnnHandle,
        input_ptr: u64,
        v_ptr: u64,
        geom: &WinogradGeometry,
    ) -> DnnResult<()> {
        let entry = kernels::INPUT_TRANSFORM_ENTRY;
        let sm = self.sm_version;
        let kernel = handle.get_or_compile_kernel(&cache_key(entry, sm), entry, move || {
            kernels::input_transform_ptx(sm)
        })?;
        let total = geom.v_plane;
        let params = kernel.launch_1d(total);
        let args = (
            input_ptr,
            v_ptr,
            geom.in_channels,
            geom.in_h,
            geom.in_w,
            geom.pad_h,
            geom.pad_w,
            geom.tiles_h,
            geom.tiles_w,
            geom.tile_count,
            total,
        );
        kernel
            .kernel()
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(format!("{entry}: {e}")))
    }

    /// Stage 2: `U = G g G^T`, one thread per `(out-channel, in-channel)` pair.
    fn launch_filter_transform(
        &self,
        handle: &DnnHandle,
        filter_ptr: u64,
        u_ptr: u64,
        geom: &WinogradGeometry,
    ) -> DnnResult<()> {
        let entry = kernels::FILTER_TRANSFORM_ENTRY;
        let sm = self.sm_version;
        let kernel = handle.get_or_compile_kernel(&cache_key(entry, sm), entry, move || {
            kernels::filter_transform_ptx(sm)
        })?;
        let total = geom.u_plane;
        let params = kernel.launch_1d(total);
        let args = (filter_ptr, u_ptr, total);
        kernel
            .kernel()
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(format!("{entry}: {e}")))
    }

    /// Stage 3: the 16 transform-domain products, one launch, `blockIdx.z = e`.
    ///
    /// This kernel is written as a fixed `GEMM_TILE x GEMM_TILE` thread block
    /// that sizes its shared-memory staging from that constant, so it must
    /// **not** go through
    /// [`CachedKernel::launch_1d`](crate::kernel_cache::CachedKernel::launch_1d)
    /// — that helper is explicitly documented as unsafe for kernels which
    /// assume a particular `blockDim`.
    fn launch_gemm(
        &self,
        handle: &DnnHandle,
        u_ptr: u64,
        v_ptr: u64,
        m_ptr: u64,
        geom: &WinogradGeometry,
    ) -> DnnResult<()> {
        let entry = kernels::GEMM_ENTRY;
        let sm = self.sm_version;
        let kernel = handle.get_or_compile_kernel(&cache_key(entry, sm), entry, move || {
            kernels::batched_gemm_ptx(sm)
        })?;
        let tile = kernels::GEMM_TILE;
        let grid = Dim3::new(
            geom.tile_count.div_ceil(tile).max(1),
            geom.out_channels.div_ceil(tile).max(1),
            F2X3_POSITIONS,
        );
        let block = Dim3::new(tile, tile, 1);
        let params = LaunchParams::new(grid, block);
        let args = (
            u_ptr,
            v_ptr,
            m_ptr,
            geom.out_channels,
            geom.in_channels,
            geom.tile_count,
        );
        kernel
            .kernel()
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(format!("{entry}: {e}")))
    }

    /// Stage 4: `Y = A^T m A` + bias, one thread per `(out-channel, tile)` pair.
    fn launch_output_transform<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        m_ptr: u64,
        output_ptr: u64,
        bias: Option<&TensorDesc<T>>,
        geom: &WinogradGeometry,
    ) -> DnnResult<()> {
        let entry = kernels::OUTPUT_TRANSFORM_ENTRY;
        let sm = self.sm_version;
        let kernel = handle.get_or_compile_kernel(&cache_key(entry, sm), entry, move || {
            kernels::output_transform_ptx(sm)
        })?;
        let total = geom.m_plane;
        let params = kernel.launch_1d(total);
        let args = (
            m_ptr,
            output_ptr,
            bias.map_or(0u64, |t| t.ptr),
            geom.out_channels,
            geom.out_h,
            geom.out_w,
            geom.tiles_h,
            geom.tiles_w,
            geom.tile_count,
            total,
        );
        kernel
            .kernel()
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(format!("{entry}: {e}")))
    }
}

/// Narrows a `u64` element count to the `u32` the kernels index with, or
/// reports the overflow instead of silently wrapping into a wild address.
fn fits_u32(value: u64, what: &str) -> DnnResult<u32> {
    u32::try_from(value).map_err(|_| {
        DnnError::InvalidDimension(format!(
            "{what} is {value} elements, which exceeds the kernels' 32-bit indexing limit"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_problem() -> ConvProblem {
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
    fn supports_the_canonical_3x3_shape() {
        assert!(WinogradConv::supports(&base_problem()));
    }

    /// Every constraint in the supported region must actually be enforced —
    /// a shape that slips through here would reach kernels that silently
    /// compute the wrong convolution (e.g. a stride-2 conv evaluated as
    /// stride 1).
    /// A named mutation of the baseline problem, for table-driven tests.
    type ProblemMutation = (&'static str, fn(&mut ConvProblem));

    #[test]
    fn supports_rejects_every_out_of_region_shape() {
        let cases: [ProblemMutation; 9] = [
            ("5x5 filter", |p| p.filter_dims = vec![5, 5]),
            ("stride 2", |p| p.stride = vec![2, 2]),
            ("dilation 2", |p| p.dilation = vec![2, 2]),
            ("groups 2", |p| p.groups = 2),
            ("padding 2", |p| p.padding = vec![2, 2]),
            ("NHWC", |p| p.layout = TensorLayout::Nhwc),
            ("f16 input", |p| p.input_type = PtxType::F16),
            ("f64 output", |p| p.output_type = PtxType::F64),
            ("3-D problem", |p| p.in_dims = vec![8, 16, 16]),
        ];
        for (label, mutate) in cases {
            let mut p = base_problem();
            mutate(&mut p);
            assert!(
                !WinogradConv::supports(&p),
                "{label} must be rejected by WinogradConv::supports"
            );
            assert!(
                WinogradConv::new(p, SmVersion::Sm86).is_err(),
                "{label} must be rejected by WinogradConv::new"
            );
        }
    }

    #[test]
    fn padding_zero_is_supported() {
        let mut p = base_problem();
        p.padding = vec![0, 0];
        assert!(WinogradConv::supports(&p));
    }

    /// F(4,3) has no forward kernels, so constructing an engine for it must
    /// fail loudly rather than produce one that launches nothing.
    #[test]
    fn f4x3_forward_is_rejected() {
        assert!(!WinogradTileSize::F4x3.forward_supported());
        assert!(WinogradTileSize::F2x3.forward_supported());
        assert!(
            WinogradConv::with_tile_size(base_problem(), WinogradTileSize::F4x3, SmVersion::Sm86)
                .is_err()
        );
        assert_eq!(
            WinogradTileSize::best_forward(32, 32),
            WinogradTileSize::F2x3
        );
        assert_eq!(WinogradTileSize::best_forward(4, 4), WinogradTileSize::F2x3);
    }

    /// `auto_select` keeps its original preference order for the dgrad/wgrad
    /// planners that still call it.
    #[test]
    fn auto_select_preference_order_is_unchanged() {
        assert_eq!(
            WinogradTileSize::auto_select(32, 32),
            WinogradTileSize::F4x3
        );
        assert_eq!(WinogradTileSize::auto_select(4, 4), WinogradTileSize::F2x3);
    }

    #[test]
    fn transform_tile_arithmetic() {
        assert_eq!(WinogradTileSize::F2x3.transform_tile(), 4);
        assert_eq!(WinogradTileSize::F2x3.transform_elements(), 16);
        assert_eq!(WinogradTileSize::F4x3.transform_tile(), 6);
        assert_eq!(WinogradTileSize::F4x3.transform_elements(), 36);
    }

    /// The workspace formula must match the layout the kernels address, and
    /// the three regions must partition it without overlap.
    #[test]
    fn workspace_layout_is_exact_and_disjoint() {
        // N=1, C=64, 32x32 pad 1 -> out 32x32 -> tiles 16x16 -> P = 256.
        let conv = WinogradConv::new(base_problem(), SmVersion::Sm86).expect("engine");
        let geom = conv.geometry().expect("geometry");
        assert_eq!(geom.tile_count, 256);
        assert_eq!(geom.v_plane, 64 * 256);
        assert_eq!(geom.u_plane, 128 * 64);
        assert_eq!(geom.m_plane, 128 * 256);

        let v_bytes = 16u64 * u64::from(geom.v_plane) * 4;
        let u_bytes = 16u64 * u64::from(geom.u_plane) * 4;
        let m_bytes = 16u64 * u64::from(geom.m_plane) * 4;
        assert_eq!(geom.v_offset(), 0);
        assert_eq!(geom.u_offset(), v_bytes);
        assert_eq!(geom.m_offset(), v_bytes + u_bytes);
        assert_eq!(
            conv.workspace_bytes().expect("bytes") as u64,
            v_bytes + u_bytes + m_bytes
        );
        // Every region must be 4-byte aligned for the f32 accesses.
        assert_eq!(geom.u_offset() % 4, 0);
        assert_eq!(geom.m_offset() % 4, 0);
    }

    /// Odd output extents round the tile count up; the extra half-tile is
    /// discarded by the output transform's boundary guard.
    #[test]
    fn odd_extents_round_tiles_up() {
        let mut p = base_problem();
        p.in_dims = vec![7, 5];
        p.padding = vec![0, 0];
        let conv = WinogradConv::new(p, SmVersion::Sm86).expect("engine");
        let geom = conv.geometry().expect("geometry");
        // out = 7-2 = 5, 5-2 = 3 -> tiles 3 x 2.
        assert_eq!((geom.out_h, geom.out_w), (5, 3));
        assert_eq!((geom.tiles_h, geom.tiles_w), (3, 2));
        assert_eq!(geom.tile_count, 6);
    }

    /// A problem whose transform buffers would overflow 32-bit element
    /// indexing must be reported, not wrapped.
    #[test]
    fn oversized_problem_is_rejected_not_wrapped() {
        let mut p = base_problem();
        p.batch = 4096;
        p.in_channels = 4096;
        p.in_dims = vec![256, 256];
        let conv = WinogradConv::new(p, SmVersion::Sm86).expect("engine");
        assert!(matches!(
            conv.workspace_bytes(),
            Err(DnnError::InvalidDimension(_))
        ));
    }

    #[test]
    fn fits_u32_boundary() {
        assert_eq!(fits_u32(u64::from(u32::MAX), "x").unwrap_or(0), u32::MAX);
        assert!(fits_u32(u64::from(u32::MAX) + 1, "x").is_err());
    }
}
