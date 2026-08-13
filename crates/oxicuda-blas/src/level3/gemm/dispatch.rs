//! GEMM kernel dispatcher — the brain of kernel selection.
//!
//! The [`GemmDispatcher`] classifies incoming GEMM problems, selects
//! architecture-aware tile configurations, generates PTX via
//! [`GemmTemplate`], and caches compiled modules for reuse.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::types::{FillMode, MathMode, Transpose};

use super::splitk::SplitKConfig;

// ---------------------------------------------------------------------------
// Problem description
// ---------------------------------------------------------------------------

/// Complete description of a GEMM problem for dispatch purposes.
///
/// Captures the matrix dimensions, transposition modes, element types, and
/// the math mode that controls whether Tensor Cores may be used.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct GemmProblem {
    /// Number of rows of the output matrix C (and of op(A)).
    pub m: u32,
    /// Number of columns of the output matrix C (and of op(B)).
    pub n: u32,
    /// Shared (inner) dimension: columns of op(A) / rows of op(B).
    pub k: u32,
    /// Whether matrix A is transposed.
    pub trans_a: Transpose,
    /// Whether matrix B is transposed.
    pub trans_b: Transpose,
    /// PTX type of the input matrices A and B.
    pub input_type: PtxType,
    /// PTX type of the output matrix C (and the accumulator).
    pub output_type: PtxType,
    /// Whether Tensor Core paths are permitted.
    pub math_mode: MathMode,
}

// ---------------------------------------------------------------------------
// Tile configuration
// ---------------------------------------------------------------------------

/// Tile dimensions and kernel tuning knobs for a GEMM launch.
///
/// The dispatcher selects a `TileConfig` based on the problem size and the
/// target architecture, then uses it to generate and launch the PTX kernel.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TileConfig {
    /// Block tile size in the M dimension (rows per CTA).
    pub tile_m: u32,
    /// Block tile size in the N dimension (columns per CTA).
    pub tile_n: u32,
    /// Block tile size in the K dimension (reduction step per iteration).
    pub tile_k: u32,
    /// Warp-level tile in M (rows computed per warp).
    pub warp_m: u32,
    /// Warp-level tile in N (columns computed per warp).
    pub warp_n: u32,
    /// Number of software pipeline stages for async global-to-shared loads.
    pub stages: u32,
    /// Whether to use Tensor Core instructions (WMMA / MMA / WGMMA).
    pub use_tensor_core: bool,
    /// Split-K factor (1 = no split, >1 = parallel K-reduction).
    pub split_k: u32,
}

// ---------------------------------------------------------------------------
// Problem classification
// ---------------------------------------------------------------------------

/// High-level classification of a GEMM problem shape.
///
/// The category drives the choice of tile configuration and, for some
/// categories, the kernel variant (e.g. split-K requires a separate
/// reduction pass).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmCategory {
    /// Normal square or moderately rectangular matrices.
    Standard,
    /// One of M or N is very small (< 32), making shared-memory tiling
    /// along that dimension wasteful.
    Skinny,
    /// K is much larger than M and N, benefiting from parallel K-reduction.
    SplitK,
    /// Hopper+ load-balanced streaming decomposition.
    StreamK,
    /// Hopper+ warp-specialized with producer/consumer warps.
    ///
    /// Splits warps into memory-loading producers and MMA-computing
    /// consumers, overlapping global memory latency with tensor-core
    /// compute. Requires SM >= 90 and a sufficiently large problem with
    /// half-precision (F16/BF16) or FP8 inputs.
    WarpSpecialized,
    /// Bandwidth-limited GEMM: low arithmetic intensity (small K or
    /// memory-bound shape). Uses wider vector loads, fewer pipeline stages,
    /// and prefetch tuning to maximise memory throughput.
    BandwidthLimited,
}

// ---------------------------------------------------------------------------
// Internal cache types
// ---------------------------------------------------------------------------

/// How a compiled GEMM kernel is launched (grid/block geometry and argument
/// tuple), which depends on which code generator produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmLaunchKind {
    /// Tiled [`GemmTemplate`] kernel — 8 args
    /// `(a, b, c, m, n, k, alpha, beta)`, tight row-major NoTrans A*B, launched
    /// with the tile-config grid/block and a grid-stride loop over M*N.
    Template,
    /// [`SimtGemmBuilder`](super::simt::SimtGemmBuilder) kernel — 11 args
    /// `(a, b, c, m, n, k, lda, ldb, ldc, alpha, beta)`, one thread per output
    /// element with a 16x16 block. Handles all four transpose combinations.
    Simt,
}

/// A compiled GEMM kernel together with its launch metadata.
struct CompiledGemm {
    /// The CUDA module that owns the compiled kernel.
    _module: Arc<Module>,
    /// The launchable kernel handle.
    kernel: Kernel,
    /// The tile config used to generate this kernel.
    tile_config: TileConfig,
    /// Dynamic shared memory requirement in bytes.
    shared_mem_bytes: u32,
    /// Launch geometry / argument shape for this kernel.
    launch_kind: GemmLaunchKind,
}

/// Key for the compiled-kernel cache.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct GemmKernelKey {
    input_type: PtxType,
    output_type: PtxType,
    trans_a: Transpose,
    trans_b: Transpose,
    /// Triangle-write mask baked into the (SIMT) kernel. `None` for a full
    /// write; distinguishes masked SYRK/SYR2K kernels from the plain GEMM
    /// kernel in the cache.
    fill_mode: Option<FillMode>,
    tile_config: TileConfig,
}

/// A compiled split-K partial-GEMM or reduction kernel (see
/// [`super::splitk`]), together with the module that owns it.
struct CompiledSplitK {
    _module: Arc<Module>,
    kernel: Kernel,
}

/// Owns the scratch workspace for a split-K launch, sized `split_factor *
/// m * n` accumulator-precision elements.
///
/// Allocated once per [`SplitKWorkspaceKey`] and then **kept for the
/// dispatcher's lifetime** — see [`GemmDispatcher::split_k_workspace`] for why
/// that is both a performance and a correctness-of-capture property.
enum SplitKWorkspace {
    F32(DeviceBuffer<f32>),
    F64(DeviceBuffer<f64>),
}

impl SplitKWorkspace {
    fn alloc(output_type: PtxType, elements: usize) -> BlasResult<Self> {
        match output_type {
            PtxType::F32 => Ok(Self::F32(DeviceBuffer::<f32>::alloc(elements).map_err(
                |e| BlasError::LaunchFailed(format!("split-K workspace alloc failed: {e}")),
            )?)),
            PtxType::F64 => Ok(Self::F64(DeviceBuffer::<f64>::alloc(elements).map_err(
                |e| BlasError::LaunchFailed(format!("split-K workspace alloc failed: {e}")),
            )?)),
            other => Err(BlasError::UnsupportedOperation(format!(
                "split-K workspace requires an F32 or F64 accumulator, got {}",
                other.as_ptx_str()
            ))),
        }
    }

    fn device_ptr(&self) -> u64 {
        match self {
            Self::F32(buf) => buf.as_device_ptr(),
            Self::F64(buf) => buf.as_device_ptr(),
        }
    }

    /// Bytes of device memory this workspace holds.
    fn bytes(&self) -> usize {
        match self {
            Self::F32(buf) => buf.len() * std::mem::size_of::<f32>(),
            Self::F64(buf) => buf.len() * std::mem::size_of::<f64>(),
        }
    }
}

/// Cache key for a reusable split-K workspace.
///
/// The stream is part of the identity, and that is the whole safety argument:
/// the partial and reduction kernels of one split-K launch communicate through
/// this buffer, so two launches sharing it must be ordered against each other.
/// Two launches on the *same* stream are ordered by stream semantics (and
/// [`GemmDispatcher::split_k_workspace`]'s lock keeps each launch's two
/// submissions adjacent in that order, so a second partial pass can never
/// slip between a first partial pass and its reduction). Two launches on
/// *different* streams have no such ordering — so they are given different
/// buffers rather than made to race.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SplitKWorkspaceKey {
    /// The stream the launch pair rides. See the type docs.
    stream: oxicuda_driver::ffi::CUstream,
    /// Accumulator precision of the workspace elements.
    output_type: PtxType,
    /// Exact element count. Keying on the exact count rather than a size class
    /// is deliberate: an entry, once created, is never resized or freed, so the
    /// device pointer for a given key is stable for the dispatcher's lifetime.
    elements: usize,
}

// ---------------------------------------------------------------------------
// GemmDispatcher
// ---------------------------------------------------------------------------

/// GEMM kernel dispatcher — selects, compiles, caches, and launches optimal
/// GEMM kernels.
///
/// The dispatcher is designed to be shared across BLAS calls via the
/// [`BlasHandle`](crate::handle::BlasHandle). It holds a read-write-locked
/// cache of compiled kernels keyed by (type, transpose, tile config).
pub struct GemmDispatcher {
    /// Target SM architecture, used for tile heuristics and PTX generation.
    sm_version: SmVersion,
    /// Number of streaming multiprocessors on the target device, used only
    /// to size the grid-stride `Template`-launch-kind GEMM launch (see
    /// [`Self::compute_grid`]) to enough concurrent threads to saturate the
    /// device.
    ///
    /// This is a hardware-*instance* fact, not implied by `sm_version`
    /// (compute capability 8.6 alone spans GPUs from 46 SMs, e.g. RTX
    /// A2000, up to 84, e.g. RTX 3090 Ti), so it is threaded in explicitly
    /// rather than looked up from an architecture table when a live device
    /// is available -- see [`Self::new_with_sm_count`]. [`Self::new`] falls
    /// back to a representative per-architecture value for contexts without
    /// one (tests, or a dispatcher built before a device is opened); an
    /// approximate count here still produces a *correct* launch, only a
    /// less precisely tuned one.
    sm_count: u32,
    /// Cache of compiled kernels.
    compiled: RwLock<HashMap<GemmKernelKey, Arc<CompiledGemm>>>,
    /// Cache of compiled split-K partial-GEMM kernels, keyed by accumulator
    /// type. Unlike the reduction kernel below, this kernel's PTX has no
    /// compile-time dependency on `split_factor` (`k_per_split`/`k_total`
    /// are ordinary runtime kernel arguments), so one compiled module per
    /// type serves every split factor.
    split_k_partial: RwLock<HashMap<PtxType, Arc<CompiledSplitK>>>,
    /// Cache of compiled split-K reduction kernels, keyed by (accumulator
    /// type, split factor) — the reduction loop is unrolled at PTX-generation
    /// time over `split_factor`, so each factor is a distinct kernel.
    split_k_reduce: RwLock<HashMap<(PtxType, u32), Arc<CompiledSplitK>>>,
    /// Reusable split-K reduction workspaces, keyed by
    /// [`SplitKWorkspaceKey`] and **never evicted, resized, or freed** while
    /// the dispatcher lives.
    ///
    /// # Why this is a cache rather than a per-call allocation
    ///
    /// Two reasons, and the second is the one that could not be worked around
    /// anywhere else:
    ///
    /// * **`cuMemFree` is a device-wide barrier.** `DeviceBuffer` frees through
    ///   the classic (non-stream-ordered) `cuMemFree`, which the driver defines
    ///   to block until every operation already submitted to every stream has
    ///   completed. Allocating a workspace per call therefore ended every
    ///   split-K GEMM with a full synchronisation the caller never asked for —
    ///   on a per-frame inference workload, once per skinny GEMM per frame.
    /// * **`cuMemAlloc` cannot be called during CUDA stream capture.** The
    ///   driver rejects it with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, which
    ///   made every split-K GEMM uncapturable — and split-K is exactly the
    ///   shape class (`m*n < 65536`, `k >= 512`) that repeated small-batch
    ///   inference GEMMs fall into. With the workspace resolved from this map
    ///   the launch pair contains nothing but two `cuLaunchKernel`s, so it
    ///   records into a graph cleanly.
    ///
    /// Keeping entries forever (rather than pooling them with reuse) is what
    /// makes the recorded pointer *stable*: a graph captured today replays
    /// against the same workspace tomorrow. See [`SplitKWorkspaceKey`] for the
    /// concurrency argument, and [`Self::SPLIT_K_WORKSPACE_MAX_ENTRIES`] /
    /// [`Self::SPLIT_K_WORKSPACE_MAX_BYTES`] for the bound on what that costs.
    ///
    /// A `Mutex` rather than an `RwLock` because the guard is deliberately held
    /// across *both* launches of a split-K pair — see
    /// [`Self::dispatch_skinny_split_k`].
    split_k_workspace: Mutex<HashMap<SplitKWorkspaceKey, SplitKWorkspace>>,
}

impl GemmDispatcher {
    /// Creates a new dispatcher targeting the given SM architecture.
    ///
    /// The streaming-multiprocessor *count* defaults to a representative
    /// value for `sm`'s architecture generation (see [`Self::sm_count`] on
    /// the field doc for why compute capability alone cannot give an exact
    /// count, and [`Self::new_with_sm_count`] to supply the real, live
    /// count from a `Device` instead -- [`crate::handle::BlasHandle`] does
    /// this automatically).
    #[must_use]
    pub fn new(sm: SmVersion) -> Self {
        Self::new_with_sm_count(sm, Self::typical_sm_count(sm))
    }

    /// Creates a new dispatcher targeting `sm`, with an explicit,
    /// caller-supplied streaming-multiprocessor count.
    ///
    /// Prefer this over [`Self::new`] whenever a live device is available:
    /// pass `device.multiprocessor_count()`. `sm_count` is clamped to at
    /// least 1 so a caller-supplied `0` (or a failed query defaulted to `0`)
    /// cannot make [`Self::compute_grid`] compute a zero-size launch.
    #[must_use]
    pub fn new_with_sm_count(sm: SmVersion, sm_count: u32) -> Self {
        Self {
            sm_version: sm,
            sm_count: sm_count.max(1),
            compiled: RwLock::new(HashMap::new()),
            split_k_partial: RwLock::new(HashMap::new()),
            split_k_reduce: RwLock::new(HashMap::new()),
            split_k_workspace: Mutex::new(HashMap::new()),
        }
    }

    /// Representative SM count for `sm`'s architecture *generation* -- a
    /// fallback for contexts without a live device (see [`Self::new`]).
    /// Mirrors `oxicuda_driver::occupancy_ext::DeviceOccupancyInfo::for_compute_capability`'s
    /// table (kept in sync by hand: that table is keyed by `(major, minor)`
    /// compute capability, not `SmVersion`, and pulling in a whole
    /// `DeviceOccupancyInfo` for one field would be a strange dependency to
    /// carry into a plain `u32` default).
    const fn typical_sm_count(sm: SmVersion) -> u32 {
        match sm {
            SmVersion::Sm75 => 68,
            SmVersion::Sm80 => 108,
            SmVersion::Sm86 => 84,
            SmVersion::Sm89 => 76,
            SmVersion::Sm90 | SmVersion::Sm90a => 132,
            SmVersion::Sm100 => 132,
            SmVersion::Sm120 => 148,
        }
    }

    /// Dispatches a GEMM operation: classify, select tile config, compile
    /// (if needed), compute grid/block, and launch the kernel.
    ///
    /// # Arguments
    ///
    /// * `problem` — the GEMM problem description.
    /// * `a_ptr` — device pointer to matrix A.
    /// * `b_ptr` — device pointer to matrix B.
    /// * `c_ptr` — device pointer to matrix C (output).
    /// * `alpha_bits` — alpha scalar as raw bits (`u64`).
    /// * `beta_bits` — beta scalar as raw bits (`u64`).
    /// * `fill_mode` — optional triangle-write mask. `None` (or `Some(Full)`)
    ///   writes the whole output; `Some(Upper)`/`Some(Lower)` leaves the
    ///   opposite triangle of `C` untouched (used by SYRK / SYR2K). Masked
    ///   requests always take the SIMT kernel.
    /// * `stream` — the CUDA stream for the launch.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError`] on PTX generation failure, module load failure,
    /// or kernel launch failure.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        problem: &GemmProblem,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        alpha_bits: u64,
        beta_bits: u64,
        fill_mode: Option<FillMode>,
        stream: &oxicuda_driver::Stream,
    ) -> BlasResult<()> {
        let category = self.classify(problem);

        // GEMV-shaped problems (tiny M*N, large K — e.g. ArcFace's
        // `1x25088 @ 25088x512` embedding projection, InSwapper's `1x512`
        // emap projection) are classified `Skinny`, whose tile config caps
        // the *single-pass* kernel at one thread per output element — for
        // M=1, N=512 that is exactly 512 threads, each then reducing all
        // 25088 K-elements serially. 512 threads is a rounding error next to
        // what an Ampere-class GPU can schedule concurrently (tens of
        // thousands), and no tile/grid tweak of the single-pass launch can
        // improve on it: with one thread doing the *entire* K reduction per
        // output element, M*N is a hard ceiling on useful parallelism.
        // Route these through a genuine two-pass split-K launch instead,
        // which parallelises the K reduction itself (see
        // `dispatch_skinny_split_k` / `super::splitk`) — every output
        // element still gets covered (this is not a substitute for grid
        // coverage, it is additional parallelism the single-pass launch
        // structurally cannot express.
        if category == GemmCategory::Skinny
            && fill_mode.is_none_or(|m| m == FillMode::Full)
            && problem.trans_a == Transpose::NoTrans
            && problem.trans_b == Transpose::NoTrans
            && Self::should_use_split_k_workspace(problem)
        {
            return self.dispatch_skinny_split_k(
                problem, a_ptr, b_ptr, c_ptr, alpha_bits, beta_bits, stream,
            );
        }

        let tile_config = self.heuristic_tile_config(problem, &category);
        let compiled = self.get_or_compile(problem, &tile_config, fill_mode)?;

        match compiled.launch_kind {
            GemmLaunchKind::Template => {
                let grid = Self::compute_grid(
                    problem,
                    &compiled.tile_config,
                    self.sm_version,
                    self.sm_count,
                );
                let block = Self::compute_block(&compiled.tile_config);
                let params =
                    LaunchParams::new(grid, block).with_shared_mem(compiled.shared_mem_bytes);

                // Kernel arguments: a_ptr, b_ptr, c_ptr, m, n, k, alpha, beta
                let args = (
                    a_ptr, b_ptr, c_ptr, problem.m, problem.n, problem.k, alpha_bits, beta_bits,
                );
                compiled
                    .kernel
                    .launch(&params, stream, &args)
                    .map_err(|e| {
                        BlasError::LaunchFailed(format!("GEMM kernel launch failed: {e}"))
                    })?;
            }
            GemmLaunchKind::Simt => {
                // Tight row-major leading dimensions for op(A)/op(B)/C. The SIMT
                // kernel reads A as `A[i*lda + j]` (NoTrans) / `A[j*lda + i]`
                // (Trans), so lda is the physical column count of the stored
                // matrix: k when A is untransposed (physical m x k), m when A is
                // transposed (physical k x m). Likewise for B and C (always
                // m x n → ldc = n).
                let lda = if problem.trans_a == Transpose::NoTrans {
                    problem.k
                } else {
                    problem.m
                };
                let ldb = if problem.trans_b == Transpose::NoTrans {
                    problem.n
                } else {
                    problem.k
                };
                let ldc = problem.n;

                const SIMT_TILE: u32 = 16;
                let grid = Dim3::new(
                    problem.n.div_ceil(SIMT_TILE),
                    problem.m.div_ceil(SIMT_TILE),
                    1,
                );
                let block = Dim3::new(SIMT_TILE, SIMT_TILE, 1);
                let params = LaunchParams::new(grid, block);

                // Kernel arguments: a, b, c, m, n, k, lda, ldb, ldc, alpha, beta
                let args = (
                    a_ptr, b_ptr, c_ptr, problem.m, problem.n, problem.k, lda, ldb, ldc,
                    alpha_bits, beta_bits,
                );
                compiled
                    .kernel
                    .launch(&params, stream, &args)
                    .map_err(|e| {
                        BlasError::LaunchFailed(format!("SIMT GEMM kernel launch failed: {e}"))
                    })?;
            }
        }

        Ok(())
    }

    /// Classifies a GEMM problem into a high-level category.
    ///
    /// The category drives tile selection: skinny problems use smaller tiles,
    /// split-K problems use parallel K-reduction, and standard problems use
    /// the largest tiles that fit in shared memory.
    pub fn classify(&self, problem: &GemmProblem) -> GemmCategory {
        let m = problem.m;
        let n = problem.n;
        let k = problem.k;

        // Skinny: one output dimension is very small.
        if m < 32 || n < 32 {
            return GemmCategory::Skinny;
        }

        // Split-K: K is much larger than both M and N.
        if k > 4 * m && k > 4 * n && k >= 1024 {
            return GemmCategory::SplitK;
        }

        // Bandwidth-limited: low arithmetic intensity (small K relative to
        // M×N). Check before standard/stream-K/warp-specialized since those
        // assume compute-bound workloads.
        {
            let elem_bytes = problem.input_type.size_bytes();
            if super::bandwidth_opt::is_bandwidth_limited(
                m as usize, n as usize, k as usize, elem_bytes,
            ) {
                return GemmCategory::BandwidthLimited;
            }
        }

        // Warp-specialized: Hopper+ with half-precision/FP8 inputs and large
        // enough problem that producer/consumer decomposition pays off.
        if super::warp_specialized::WarpSpecializedGemm::is_applicable(problem, self.sm_version) {
            return GemmCategory::WarpSpecialized;
        }

        // Stream-K: Hopper+ with large enough problem.
        if self.sm_version >= SmVersion::Sm90
            && u64::from(m) * u64::from(n) * u64::from(k) >= 64 * 1024 * 1024
        {
            return GemmCategory::StreamK;
        }

        GemmCategory::Standard
    }

    /// Selects a tile configuration using architecture-aware heuristics.
    ///
    /// The returned [`TileConfig`] is a best-effort default; the autotuner
    /// can later refine it with profiling data.
    pub fn heuristic_tile_config(
        &self,
        problem: &GemmProblem,
        category: &GemmCategory,
    ) -> TileConfig {
        let caps = self.sm_version.capabilities();

        // Determine whether Tensor Cores should be used.
        let use_tc = problem.math_mode == MathMode::TensorCore
            && caps.has_tensor_cores
            && super::tensor_core::TensorCoreValidator::is_supported(
                self.sm_version,
                problem.input_type,
                problem.output_type,
            );

        match category {
            GemmCategory::Standard => {
                // Use TileSelector for rectangular-aware tile selection.
                let selector = super::tiles::TileSelector::new(self.sm_version, use_tc);
                selector.select(problem.m, problem.n, problem.k)
            }
            GemmCategory::Skinny => self.skinny_tile_config(problem, use_tc),
            GemmCategory::SplitK => self.splitk_tile_config(problem, use_tc),
            GemmCategory::StreamK => self.streamk_tile_config(use_tc),
            GemmCategory::WarpSpecialized => self.warp_specialized_tile_config(problem),
            GemmCategory::BandwidthLimited => self.bandwidth_limited_tile_config(problem),
        }
    }

    /// Tile config for skinny (M or N < 32) problems.
    fn skinny_tile_config(&self, problem: &GemmProblem, use_tc: bool) -> TileConfig {
        let small_dim = problem.m.min(problem.n);
        let tile_small = if small_dim <= 8 {
            8
        } else if small_dim <= 16 {
            16
        } else {
            32
        };
        let tile_large = if use_tc { 128 } else { 64 };

        let (tile_m, tile_n) = if problem.m < problem.n {
            (tile_small, tile_large)
        } else {
            (tile_large, tile_small)
        };

        TileConfig {
            tile_m,
            tile_n,
            tile_k: if use_tc { 32 } else { 8 },
            warp_m: tile_m.min(32),
            warp_n: tile_n.min(32),
            stages: if use_tc && self.sm_version >= SmVersion::Sm80 {
                2
            } else {
                1
            },
            use_tensor_core: use_tc,
            split_k: 1,
        }
    }

    /// Tile config for split-K problems (K >> M, N).
    fn splitk_tile_config(&self, problem: &GemmProblem, use_tc: bool) -> TileConfig {
        // Choose split factor so each partition has ~256 K-elements.
        let target_k_per_split = 256u32;
        let split_k = (problem.k / target_k_per_split).clamp(2, 32);

        let base = if use_tc {
            TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: if self.sm_version >= SmVersion::Sm80 {
                    3
                } else {
                    2
                },
                use_tensor_core: true,
                split_k: 1,
            }
        } else {
            TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 8,
                warp_m: 32,
                warp_n: 32,
                stages: 1,
                use_tensor_core: false,
                split_k: 1,
            }
        };

        TileConfig { split_k, ..base }
    }

    /// Tile config for stream-K (Hopper+) problems.
    fn streamk_tile_config(&self, use_tc: bool) -> TileConfig {
        if use_tc {
            TileConfig {
                tile_m: 256,
                tile_n: 128,
                tile_k: 64,
                warp_m: 64,
                warp_n: 64,
                stages: 4,
                use_tensor_core: true,
                split_k: 1, // Stream-K handles its own decomposition.
            }
        } else {
            TileConfig {
                tile_m: 128,
                tile_n: 64,
                tile_k: 16,
                warp_m: 32,
                warp_n: 32,
                stages: 2,
                use_tensor_core: false,
                split_k: 1,
            }
        }
    }

    /// Tile config for warp-specialized (Hopper+) problems.
    ///
    /// Creates a default warp-specialized configuration and converts it to
    /// a [`TileConfig`]. The actual kernel uses the full
    /// [`WarpSpecializedGemm`](super::warp_specialized::WarpSpecializedGemm)
    /// struct for generation.
    fn warp_specialized_tile_config(&self, problem: &GemmProblem) -> TileConfig {
        // Pick pipeline stages based on problem size.
        let volume = u64::from(problem.m) * u64::from(problem.n) * u64::from(problem.k);
        let stages = if volume >= 256 * 1024 * 1024 { 4 } else { 3 };

        // Attempt to build a WarpSpecializedGemm; fall back to standard
        // TC config on any validation error.
        match super::warp_specialized::WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            stages,
            self.sm_version,
            problem.input_type,
            problem.output_type,
        ) {
            Ok(ws) => ws.to_tile_config(),
            Err(_) => {
                // Fallback: Hopper TC config.
                TileConfig {
                    tile_m: 256,
                    tile_n: 128,
                    tile_k: 64,
                    warp_m: 64,
                    warp_n: 64,
                    stages: 4,
                    use_tensor_core: true,
                    split_k: 1,
                }
            }
        }
    }

    /// Tile config for bandwidth-limited (memory-bound) problems.
    ///
    /// Delegates to [`select_bandwidth_tiles`] and converts the result to a
    /// [`TileConfig`] for the standard dispatch pipeline.
    fn bandwidth_limited_tile_config(&self, problem: &GemmProblem) -> TileConfig {
        let prec = match problem.input_type {
            PtxType::F16 => super::bandwidth_opt::BandwidthPrecision::F16,
            PtxType::BF16 => super::bandwidth_opt::BandwidthPrecision::BF16,
            PtxType::F64 => super::bandwidth_opt::BandwidthPrecision::F64,
            _ => super::bandwidth_opt::BandwidthPrecision::F32,
        };
        let cfg = super::bandwidth_opt::BandwidthGemmConfig {
            m: problem.m as usize,
            n: problem.n as usize,
            k: problem.k as usize,
            sm_version: self.sm_version,
            precision: prec,
            strategy: super::bandwidth_opt::BandwidthStrategy::Auto,
        };
        let bw = super::bandwidth_opt::select_bandwidth_tiles(&cfg);
        TileConfig {
            tile_m: bw.tile_m as u32,
            tile_n: bw.tile_n as u32,
            tile_k: bw.tile_k as u32,
            warp_m: (bw.tile_m / bw.warps_m.max(1)) as u32,
            warp_n: (bw.tile_n / bw.warps_n.max(1)) as u32,
            stages: bw.pipeline_stages as u32,
            use_tensor_core: false,
            split_k: 1,
        }
    }

    /// Dynamic shared-memory byte budget for a `Template`-launch-kind kernel
    /// compiled from `template`: `0` unless
    /// [`GemmTemplate::uses_shared_memory_tiles`] reports that `template`'s
    /// `generate()` output genuinely stages A/B tiles through shared memory
    /// (it does not, today -- see that method's doc comment), in which case
    /// it is `(tile_m*tile_k + tile_k*tile_n) * elem_bytes * stages`.
    ///
    /// A free function (not a `&self` method) and pure -- no PTX generation,
    /// no device access -- so it is directly unit-testable without a GPU.
    fn template_shared_mem_bytes(
        template: &GemmTemplate,
        tile_config: &TileConfig,
        elem_bytes: u32,
    ) -> u32 {
        if !template.uses_shared_memory_tiles() {
            return 0;
        }
        let smem_a = tile_config.tile_m * tile_config.tile_k * elem_bytes;
        let smem_b = tile_config.tile_k * tile_config.tile_n * elem_bytes;
        (smem_a + smem_b) * tile_config.stages
    }

    /// Retrieves a cached compiled kernel, or generates PTX and compiles it.
    fn get_or_compile(
        &self,
        problem: &GemmProblem,
        tile_config: &TileConfig,
        fill_mode: Option<FillMode>,
    ) -> BlasResult<Arc<CompiledGemm>> {
        // A triangle mask can only be honoured by the SIMT kernel (the tiled
        // `GemmTemplate` always writes a full tile), so a masked request is
        // normalised away for the plain full-write cache key.
        let mask = match fill_mode {
            Some(FillMode::Upper) | Some(FillMode::Lower) => fill_mode,
            None | Some(FillMode::Full) => None,
        };

        let key = GemmKernelKey {
            input_type: problem.input_type,
            output_type: problem.output_type,
            trans_a: problem.trans_a,
            trans_b: problem.trans_b,
            fill_mode: mask,
            tile_config: tile_config.clone(),
        };

        // Fast path: read lock.
        {
            let cache = self
                .compiled
                .read()
                .map_err(|_| BlasError::LaunchFailed("kernel cache lock poisoned".into()))?;
            if let Some(entry) = cache.get(&key) {
                return Ok(Arc::clone(entry));
            }
        }

        // Slow path: generate PTX and compile. The tiled `GemmTemplate` only
        // computes NoTrans A*B and always writes a full tile, so any transposed
        // operand OR any triangle-write mask is routed to the SIMT builder
        // (which honours lda/ldb/ldc for all four (trans_a, trans_b)
        // combinations and can skip stores outside the requested triangle).
        // Without this, a transposed GEMM silently returned the untransposed
        // product and a masked GEMM clobbered the off-triangle.
        let transposed =
            problem.trans_a != Transpose::NoTrans || problem.trans_b != Transpose::NoTrans;
        let use_simt = transposed || mask.is_some();

        let (module, kernel, shared_mem_bytes, launch_kind) = if use_simt {
            let builder = super::simt::SimtGemmBuilder::new(
                self.sm_version,
                problem.input_type,
                problem.output_type,
                problem.trans_a,
                problem.trans_b,
                mask,
            );
            let ptx = builder.generate()?;
            let kernel_name = builder.kernel_name();
            let module = Arc::new(
                Module::from_ptx(&ptx)
                    .map_err(|e| BlasError::LaunchFailed(format!("module load failed: {e}")))?,
            );
            let kernel = Kernel::from_module(Arc::clone(&module), &kernel_name)
                .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup failed: {e}")))?;
            (module, kernel, 0u32, GemmLaunchKind::Simt)
        } else {
            let template = GemmTemplate {
                tile_m: tile_config.tile_m,
                tile_n: tile_config.tile_n,
                tile_k: tile_config.tile_k,
                warp_m: tile_config.warp_m,
                warp_n: tile_config.warp_n,
                precision: problem.input_type,
                accumulator: problem.output_type,
                use_tensor_core: tile_config.use_tensor_core,
                stages: tile_config.stages,
                target: self.sm_version,
                epilogue: EpilogueKind::LinearCombination,
            };

            let ptx = template.generate().map_err(|e| {
                BlasError::PtxGeneration(format!("GEMM PTX generation failed: {e}"))
            })?;

            let kernel_name = template.kernel_name();
            let module = Arc::new(
                Module::from_ptx(&ptx)
                    .map_err(|e| BlasError::LaunchFailed(format!("module load failed: {e}")))?,
            );
            let kernel = Kernel::from_module(Arc::clone(&module), &kernel_name)
                .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup failed: {e}")))?;

            let elem_bytes = problem.input_type.size_bytes() as u32;
            let shared_mem_bytes =
                Self::template_shared_mem_bytes(&template, tile_config, elem_bytes);

            (module, kernel, shared_mem_bytes, GemmLaunchKind::Template)
        };

        let entry = Arc::new(CompiledGemm {
            _module: module,
            kernel,
            tile_config: tile_config.clone(),
            shared_mem_bytes,
            launch_kind,
        });

        // Insert into cache.
        {
            let mut cache = self
                .compiled
                .write()
                .map_err(|_| BlasError::LaunchFailed("kernel cache lock poisoned".into()))?;
            cache.insert(key, Arc::clone(&entry));
        }

        Ok(entry)
    }

    /// Number of independent grid-stride "waves" [`compute_grid`](Self::compute_grid)
    /// sizes a `Template`-launch-kind launch for: each wave is one full
    /// device's worth of resident threads (`sm_count *
    /// sm_version.max_threads_per_sm()`). More waves give the scheduler
    /// slack to hide the tail effect of unevenly-scheduled CTAs finishing
    /// at different times; 2-4 is the audited range (measured: 191 -> 747
    /// GFLOPS at 1024^3 F32 from this fix alone, on an RTX A4000), and 3 is
    /// the middle of it.
    const GRID_STRIDE_WAVES: u64 = 3;

    /// Computes the grid dimensions for a `Template`-launch-kind GEMM launch.
    ///
    /// `GemmTemplate::generate()` (the only kernel this launch kind ever
    /// runs) is **not** a tiled kernel: it is one thread per output element,
    /// with an internal grid-stride loop that lets a thread cover more than
    /// one element by striding `total_launched_threads` at a time (see its
    /// own doc comment). Sizing the grid as if it *were* tiled -- one CTA
    /// per `tile_m x tile_n` region, i.e. `ceil(n/tile_n) * ceil(m/tile_m)`
    /// CTAs, which is what this function computed before -- drastically
    /// under-provisions large problems: for a 1024x1024x1024 F32 GEMM at
    /// the `Standard` tile config (128x128), that was only 8x8=64 CTAs
    /// (8192 threads total on a device that can run tens of thousands
    /// concurrently), each thread then serially grid-striding through
    /// roughly 128 elements one at a time.
    ///
    /// The right sizing question for a grid-stride kernel is not "how many
    /// tiles" but "how many threads can usefully run at once": launch
    /// [`Self::GRID_STRIDE_WAVES`] full device occupancies, clamped to the
    /// number of output elements (`m * n`) -- extra threads beyond that do
    /// zero work (their very first bounds check exits immediately, per
    /// `generate()`'s `$TILE_LOOP`), so launching more only adds
    /// `cuLaunchKernel` overhead, never throughput. `tc.split_k` plays no
    /// role here: for this launch kind it was only ever a roundabout way to
    /// request "more threads than one CTA per tile gives", which this
    /// formula now computes directly and more precisely (the dedicated
    /// split-K *workspace* launch in [`Self::dispatch_skinny_split_k`] is
    /// the mechanism that gives K itself real parallelism).
    fn compute_grid(problem: &GemmProblem, tc: &TileConfig, sm: SmVersion, sm_count: u32) -> Dim3 {
        let block_threads = u64::from(Self::compute_block(tc).x.max(1));
        let device_threads = u64::from(sm_count.max(1)) * u64::from(sm.max_threads_per_sm());
        let total_elems = u64::from(problem.m) * u64::from(problem.n);
        let target_threads = (device_threads * Self::GRID_STRIDE_WAVES).min(total_elems);
        // `target_threads` is at most `total_elems`, itself at most
        // `u32::MAX * u32::MAX`; `ctas` (target_threads / block_threads,
        // block_threads >= 1) is therefore always far inside `u32`'s range
        // for any problem/launch shape this dispatcher actually produces
        // (grid.x's real CUDA ceiling is `2^31 - 1`), but the conversion is
        // still checked rather than assumed, with a saturating fallback
        // rather than a panic if some future caller ever proves that wrong.
        let ctas = target_threads.div_ceil(block_threads).max(1);
        let grid_x = u32::try_from(ctas).unwrap_or(u32::MAX);
        Dim3::new(grid_x, 1, 1)
    }

    /// Computes the block dimensions from the tile configuration.
    ///
    /// Each CTA (block) handles one `tile_m × tile_n` output tile using a
    /// flat 1-D thread layout.  The number of warps is:
    ///   `warps_m = tile_m / warp_m`
    ///   `warps_n = tile_n / warp_n`
    /// Total threads = `warps_m * warps_n * WARP_SIZE` (≤ 1 024).
    fn compute_block(tc: &TileConfig) -> Dim3 {
        const WARP_SIZE: u32 = 32;
        let warps_m = tc.tile_m / tc.warp_m.max(1);
        let warps_n = tc.tile_n / tc.warp_n.max(1);
        let threads = (warps_m * warps_n * WARP_SIZE).min(1024);
        Dim3::new(threads, 1, 1)
    }

    // -----------------------------------------------------------------------
    // Split-K workspace launch (GEMV-shaped Skinny problems)
    // -----------------------------------------------------------------------

    /// Below this `M*N`, the output alone cannot occupy a modern GPU's
    /// thread capacity (an RTX A4000: 48 SMs, up to 1536 resident threads
    /// each == ~73728 concurrent slots; even a high-end consumer/datacenter
    /// Ampere/Ada/Hopper part is in the same order of magnitude), so it is
    /// always worth spending extra parallelism on splitting K instead. Above
    /// it, the single-pass launch already has enough output elements to
    /// keep the device busy and a second reduction pass would only add
    /// overhead.
    const SPLIT_K_MN_THRESHOLD: u64 = 65_536;

    /// Splitting a short K into slivers adds a workspace allocation, a
    /// second kernel launch, and a reduction pass for little or no benefit;
    /// require at least two full `target_k_per_split`-sized partitions
    /// (mirrors [`Self::splitk_tile_config`]'s own `target_k_per_split`).
    const SPLIT_K_MIN_K: u32 = 512;

    /// Whether `problem` should take the split-K workspace path (see
    /// [`Self::dispatch_skinny_split_k`]) rather than the single-pass
    /// tiled/naive kernel.
    ///
    /// Restricted to homogeneous precision (`input_type == output_type`, and
    /// both `F32` or `F64`): the partial-sum kernel accumulates directly in
    /// that type with no `F16`/`BF16` <-> accumulator conversion path. The
    /// single-pass kernel already handles mixed precision correctly, so
    /// declining here is a missed optimisation, never a correctness gap —
    /// this targets the F32/F64 inference workload split-K actually helps.
    fn should_use_split_k_workspace(problem: &GemmProblem) -> bool {
        if problem.input_type != problem.output_type {
            return false;
        }
        if !matches!(problem.output_type, PtxType::F32 | PtxType::F64) {
            return false;
        }
        let mn = u64::from(problem.m) * u64::from(problem.n);
        mn > 0 && mn < Self::SPLIT_K_MN_THRESHOLD && problem.k >= Self::SPLIT_K_MIN_K
    }

    /// Dispatches a `Skinny`-category GEMM through a two-pass split-K
    /// launch: a partial-GEMM kernel with `gridDim.z == split_factor`
    /// reduces disjoint K sub-ranges into a scratch workspace (so the
    /// *reduction itself* is parallel, not just the M*N output-element
    /// coverage the single-pass kernel is capped by — see
    /// [`super::splitk::generate_splitk_partial_kernel`]), then a reduction
    /// kernel sums the partitions and applies `alpha`/`beta`
    /// ([`super::splitk::generate_splitk_reduction_kernel`]).
    ///
    /// Only ever called for `NoTrans` x `NoTrans`, full-write (no triangle
    /// mask) problems with a homogeneous `F32`/`F64` accumulator — see the
    /// call site in [`Self::dispatch`] and [`Self::should_use_split_k_workspace`].
    #[allow(clippy::too_many_arguments)]
    fn dispatch_skinny_split_k(
        &self,
        problem: &GemmProblem,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        alpha_bits: u64,
        beta_bits: u64,
        stream: &oxicuda_driver::Stream,
    ) -> BlasResult<()> {
        // Same "~256 K-elements per partition" target as `splitk_tile_config`,
        // clamped to [2, 32] partitions.
        let target_k_per_split = 256u32;
        let split_factor = (problem.k / target_k_per_split).clamp(2, 32);
        let cfg = SplitKConfig::new(problem.k, split_factor);

        let partial = self.get_or_compile_splitk_partial(problem.output_type)?;
        let reduce = self.get_or_compile_splitk_reduce(problem.output_type, cfg.split_factor)?;

        let mn = problem.m * problem.n;
        let ws_elements = cfg.workspace_elements(problem.m, problem.n);
        let ws_elements = usize::try_from(ws_elements).map_err(|_| {
            BlasError::LaunchFailed(format!(
                "split-K workspace of {ws_elements} elements overflows usize"
            ))
        })?;
        // Every `(z, row, col)` workspace slot is written exactly once by
        // the partial kernel below (see its doc comment), so an
        // uninitialised — or previously-used — allocation is safe: nothing is
        // ever read before it is written, this call or any earlier one.
        //
        // `cached` is held across BOTH launches below, which is what keeps the
        // partial/reduction pair adjacent in stream order. Without it, two
        // threads submitting split-K GEMMs of the same shape onto the same
        // stream could interleave as `partial(A), partial(B), reduce(A),
        // reduce(B)`, and `reduce(A)` would sum B's partial sums. It is a
        // submission-side lock only: it is released as soon as both launches
        // are *enqueued*, never held while the device runs them.
        let mut cached = self.lock_split_k_workspaces()?;
        let key = SplitKWorkspaceKey {
            stream: stream.raw(),
            output_type: problem.output_type,
            elements: ws_elements,
        };
        // A per-call allocation, used only when the cache is at its bound. It
        // must outlive both launches, so it is bound here rather than inside
        // the `else` arm. (This is also the only path that can still fail
        // under stream capture — `cuMemAlloc` is forbidden there — which is
        // reported to the caller as a launch failure exactly as before.)
        let overflow_workspace;
        let ws_ptr = if let Some(workspace) = cached.get(&key) {
            workspace.device_ptr()
        } else if Self::split_k_cache_has_room(&cached, problem.output_type, ws_elements) {
            let workspace = SplitKWorkspace::alloc(problem.output_type, ws_elements)?;
            let ptr = workspace.device_ptr();
            cached.insert(key, workspace);
            ptr
        } else {
            // Over the bound: fall back to the historical per-call allocation.
            // Correct, merely slower (and not capturable) — never a wrong
            // answer, and never an unbounded cache.
            overflow_workspace = SplitKWorkspace::alloc(problem.output_type, ws_elements)?;
            overflow_workspace.device_ptr()
        };

        // Partial pass: gridDim.z == split_factor selects the K-partition;
        // gridDim.x * blockDim.x grid-strides over the flattened M*N output
        // within each partition (see the kernel's own doc comment — the
        // grid-stride loop makes any positive thread count correct, so
        // sizing for full M*N coverage here is what buys the occupancy this
        // path exists for, not a correctness requirement).
        const PARTIAL_BLOCK: u32 = 256;
        let partial_grid = Dim3::new(mn.div_ceil(PARTIAL_BLOCK).max(1), 1, cfg.split_factor);
        let partial_block = Dim3::new(PARTIAL_BLOCK, 1, 1);
        let partial_params = LaunchParams::new(partial_grid, partial_block);
        let partial_args = (
            a_ptr,
            b_ptr,
            ws_ptr,
            problem.m,
            problem.n,
            problem.k,
            cfg.k_per_split,
        );
        partial
            .kernel
            .launch(&partial_params, stream, &partial_args)
            .map_err(|e| {
                BlasError::LaunchFailed(format!("split-K partial GEMM launch failed: {e}"))
            })?;

        // Reduction pass: one thread per output element (no grid-stride in
        // this kernel — see its doc comment), so `div_ceil` sizing here
        // *is* a correctness requirement, not just a perf choice.
        const REDUCE_BLOCK: u32 = 256;
        let reduce_grid = mn.div_ceil(REDUCE_BLOCK).max(1);
        let reduce_params = LaunchParams::new(reduce_grid, REDUCE_BLOCK);
        let reduce_args = (ws_ptr, c_ptr, mn, alpha_bits, beta_bits);
        reduce
            .kernel
            .launch(&reduce_params, stream, &reduce_args)
            .map_err(|e| {
                BlasError::LaunchFailed(format!("split-K reduction launch failed: {e}"))
            })?;

        // Both launches are enqueued; the submission lock may go now. A cached
        // workspace stays allocated (that is the point). An `overflow_workspace`
        // — the over-the-bound fallback — drops here instead, and `DeviceBuffer`
        // frees through the classic `cuMemFree`, which the driver defines to
        // block until every operation already submitted to every stream has
        // completed, so both launches above are guaranteed finished before that
        // memory is reclaimed. (The *caller* still owns synchronising `stream`
        // before reading `c_ptr` back to the host, exactly as for the
        // single-pass launch path.)
        drop(cached);
        Ok(())
    }

    /// Most distinct split-K workspaces kept alive at once.
    ///
    /// An inference session repeats a handful of skinny GEMM shapes forever, so
    /// this is generous for the workload it exists for while still bounding
    /// what an adversarial shape sweep can pin down.
    const SPLIT_K_WORKSPACE_MAX_ENTRIES: usize = 64;

    /// Most device memory the split-K workspace cache may hold, in bytes.
    ///
    /// A single workspace is at most `32 * 65535` accumulator elements
    /// (`split_factor` caps at 32, and `m*n` below
    /// [`Self::SPLIT_K_MN_THRESHOLD`]), i.e. ~8 MiB in F32 and ~16 MiB in F64,
    /// so this admits dozens of distinct shapes before the fallback engages.
    const SPLIT_K_WORKSPACE_MAX_BYTES: usize = 256 * 1024 * 1024;

    /// Acquire the split-K workspace map.
    ///
    /// A poisoned lock is reported rather than recovered: the map owns live
    /// device pointers that a captured CUDA graph may already have baked in,
    /// so continuing past a panic that happened while it was being mutated is
    /// not a risk worth taking for a cache.
    fn lock_split_k_workspaces(
        &self,
    ) -> BlasResult<std::sync::MutexGuard<'_, HashMap<SplitKWorkspaceKey, SplitKWorkspace>>> {
        self.split_k_workspace
            .lock()
            .map_err(|_| BlasError::LaunchFailed("split-K workspace cache lock poisoned".into()))
    }

    /// Whether a new `elements`-long workspace of `output_type` still fits
    /// inside both cache bounds.
    fn split_k_cache_has_room(
        cached: &HashMap<SplitKWorkspaceKey, SplitKWorkspace>,
        output_type: PtxType,
        elements: usize,
    ) -> bool {
        if cached.len() >= Self::SPLIT_K_WORKSPACE_MAX_ENTRIES {
            return false;
        }
        let element_bytes = match output_type {
            PtxType::F64 => std::mem::size_of::<f64>(),
            // Only F32/F64 ever reach here (`should_use_split_k_workspace`),
            // and `SplitKWorkspace::alloc` rejects anything else; F32 is the
            // right size for the only other reachable case.
            _ => std::mem::size_of::<f32>(),
        };
        let held: usize = cached.values().map(SplitKWorkspace::bytes).sum();
        held.saturating_add(elements.saturating_mul(element_bytes))
            <= Self::SPLIT_K_WORKSPACE_MAX_BYTES
    }

    /// Device bytes currently held by this dispatcher's split-K workspace
    /// cache.
    ///
    /// Zero until the first split-K GEMM; monotonically non-decreasing after
    /// that, by design — see [`Self::split_k_workspace`].
    ///
    /// # Errors
    ///
    /// [`BlasError::LaunchFailed`] if the cache lock is poisoned.
    pub fn split_k_workspace_bytes(&self) -> BlasResult<usize> {
        Ok(self
            .lock_split_k_workspaces()?
            .values()
            .map(SplitKWorkspace::bytes)
            .sum())
    }

    /// Retrieves (or compiles and caches) the split-K partial-GEMM kernel
    /// for `acc_type`. See [`super::splitk::generate_splitk_partial_kernel`].
    fn get_or_compile_splitk_partial(&self, acc_type: PtxType) -> BlasResult<Arc<CompiledSplitK>> {
        {
            let cache = self.split_k_partial.read().map_err(|_| {
                BlasError::LaunchFailed("split-K partial kernel cache lock poisoned".into())
            })?;
            if let Some(entry) = cache.get(&acc_type) {
                return Ok(Arc::clone(entry));
            }
        }

        let (kernel_name, ptx) =
            super::splitk::generate_splitk_partial_kernel(self.sm_version, acc_type)?;
        let module = Arc::new(
            Module::from_ptx(&ptx)
                .map_err(|e| BlasError::LaunchFailed(format!("module load failed: {e}")))?,
        );
        let kernel = Kernel::from_module(Arc::clone(&module), &kernel_name)
            .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup failed: {e}")))?;
        let entry = Arc::new(CompiledSplitK {
            _module: module,
            kernel,
        });

        let mut cache = self.split_k_partial.write().map_err(|_| {
            BlasError::LaunchFailed("split-K partial kernel cache lock poisoned".into())
        })?;
        cache.insert(acc_type, Arc::clone(&entry));
        Ok(entry)
    }

    /// Retrieves (or compiles and caches) the split-K reduction kernel for
    /// `(acc_type, split_factor)`. See
    /// [`super::splitk::generate_splitk_reduction_kernel`].
    fn get_or_compile_splitk_reduce(
        &self,
        acc_type: PtxType,
        split_factor: u32,
    ) -> BlasResult<Arc<CompiledSplitK>> {
        let key = (acc_type, split_factor);
        {
            let cache = self.split_k_reduce.read().map_err(|_| {
                BlasError::LaunchFailed("split-K reduction kernel cache lock poisoned".into())
            })?;
            if let Some(entry) = cache.get(&key) {
                return Ok(Arc::clone(entry));
            }
        }

        let (kernel_name, ptx) = super::splitk::generate_splitk_reduction_kernel(
            self.sm_version,
            acc_type,
            split_factor,
        )?;
        let module = Arc::new(
            Module::from_ptx(&ptx)
                .map_err(|e| BlasError::LaunchFailed(format!("module load failed: {e}")))?,
        );
        let kernel = Kernel::from_module(Arc::clone(&module), &kernel_name)
            .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup failed: {e}")))?;
        let entry = Arc::new(CompiledSplitK {
            _module: module,
            kernel,
        });

        let mut cache = self.split_k_reduce.write().map_err(|_| {
            BlasError::LaunchFailed("split-K reduction kernel cache lock poisoned".into())
        })?;
        cache.insert(key, Arc::clone(&entry));
        Ok(entry)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
