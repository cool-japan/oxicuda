//! CTA-tiled implicit-GEMM forward convolution.
//!
//! Same mathematics as [`ImplicitGemmConv`](super::implicit_gemm::ImplicitGemmConv)
//! -- the identical conv-to-GEMM index mapping, the identical cross-correlation
//! convention, the identical implicit zero padding -- but computed with a real
//! register-blocked, shared-memory-staged GEMM mainloop
//! ([`oxicuda_ptx::templates::tiled_mainloop`]) instead of one thread per
//! output element.
//!
//! # The mapping, and why it is transposed relative to the scalar engine
//!
//! The scalar engine enumerates output *elements*. This one enumerates a GEMM
//! whose orientation is chosen by what NCHW memory actually looks like:
//!
//! ```text
//! GEMM row    (m) = output channel k                    -> 0 .. C_out
//! GEMM column (n) = flattened output pixel (b, oh, ow)  -> 0 .. batch*P*Q
//! GEMM depth  (k) = filter tap (c, r, s)                -> 0 .. C_in*R*S
//!
//! A[m][kk] = filter[k][c][r][s]      row-major, contiguous in kk -- the
//!                                    filter tensor *is* this matrix
//! B[kk][n] = input[b][c][ih][iw]     implicit: never materialised
//! D[m][n]  = out[b][k][oh][ow]       contiguous in n within a batch item
//! ```
//!
//! Orienting it this way (rather than the textbook `M = pixels`) is what makes
//! both the `B` staging *and* the epilogue coalesced: consecutive `n` are
//! consecutive `ow`, hence consecutive addresses in both the input and the
//! output for unit stride. The 4-wide column groups the mainloop hands each
//! thread therefore become one `st.global.v4.f32` each.
//!
//! # Why `tile_k` is a multiple of `R*S`
//!
//! Decomposing a GEMM-depth index `kk` into `(c, r, s)` costs two divisions,
//! and doing that per element per k-step would swamp the FMAs. Choosing
//! `tile_k = channels_per_ktile * R * S` makes each staging slot's `(r, s)`
//! -- and therefore its input coordinates `(ih, iw)`, its padding-validity
//! predicate, and its address -- **invariant across the whole k-loop**: a
//! k-step advances the channel by exactly `channels_per_ktile`, i.e. the
//! address by a fixed `channels_per_ktile * H * W * 4` bytes. Every one of
//! those quantities is then computed once, in the mainloop's prologue, and the
//! loop body contains no convolution index arithmetic whatsoever.
//!
//! That is also why this engine takes over from the scalar one only where it
//! can be set up cleanly: see [`TiledConvPlan::for_problem`] for the exact
//! decline rules. Everything it declines still runs on the scalar engine,
//! which remains the numeric oracle both the unit tests and
//! `OXIONNX_CUDA_VERIFY` compare against.

use std::sync::OnceLock;

use oxicuda_blas::GpuFloat;
use oxicuda_launch::{Dim3, LaunchParams};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::{BodyBuilder, KernelBuilder};
use oxicuda_ptx::ir::{PtxType, Register};
use oxicuda_ptx::templates::tiled_mainloop::{
    GlobalTap, StageSlot, TILED_MAINLOOP_THREADS, TiledAccumulators, TiledMainloopConfig,
    emit_tiled_mainloop,
};

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::kernel_cache::cache_key;
use crate::types::{TensorDesc, TensorDescMut, TensorLayout};

use super::super::descriptor::ConvProblem;

// ---------------------------------------------------------------------------
// Profitability thresholds
// ---------------------------------------------------------------------------

/// Environment variable that takes the tiled kernel out of service entirely,
/// restoring the exact dispatch every consumer had before it existed.
///
/// Honoured in [`TiledConvPlan::for_problem`] -- the single place the
/// "is this shape tiled?" question is answered -- so setting it moves
/// `oxicuda-dnn`'s [`ImplicitGemmConv`](super::implicit_gemm::ImplicitGemmConv),
/// `algo_select::select_algorithm` *and* `oxionnx-cuda`'s `pick_engine` back
/// together, rather than leaving one of them routing to an engine that is no
/// longer tiled.
///
/// Exists so the tiling can be A/B-measured end to end (`oxiface`, `oxionnx`)
/// without rebuilding, and so a suspected miscompare can be pinned on it in
/// one run. Read once per process.
pub const DISABLE_TILED_ENV: &str = "OXICUDA_DISABLE_TILED_CONV";

/// Whether [`DISABLE_TILED_ENV`] is set to something other than `0`.
fn tiled_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED
        .get_or_init(|| std::env::var(DISABLE_TILED_ENV).is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// Smallest GEMM depth (`C_in/groups * R * S`) worth tiling.
///
/// Below this the k-loop is too short to amortise the staging prologue and the
/// two barriers per k-step, and the scalar engine -- which needs no staging at
/// all -- wins.
pub const MIN_GEMM_K: u32 = 64;

/// Output-pixel count (`batch * P * Q`) above which the tiling is taken
/// unconditionally.
///
/// A single sufficient condition, not a necessary one: what actually has to
/// hold is that the CTA grid can fill the device, which a wide output
/// guarantees on its own. See [`MIN_CTAS`] for the other way to satisfy it.
pub const MIN_GEMM_N: u32 = 4096;

/// CTA count above which the tiling is taken even for a *narrow* output.
///
/// The InSwapper generator's residual blocks are exactly this case and they
/// are the single most expensive layer in the face-swap pipeline: 1024 -> 1024
/// channels, 3x3, over a 32x32 output. That is only 1024 output pixels -- a
/// quarter of [`MIN_GEMM_N`] -- yet the GEMM it maps to is 1024 x 1024 x 9216,
/// which tiles into 8 x 8 = 64 CTAs and fills every SM of this class of device
/// with 19.3 GFLOP of work. Judging that shape by its pixel count alone would
/// leave the pipeline's heaviest convolution on the scalar engine.
///
/// 32 is deliberately below the 48 SMs of the development device: a CTA count
/// in that range still leaves the tiling comfortably ahead of a scalar kernel
/// that runs at a fixed ~1 TFLOPS no matter how the work is shaped.
pub const MIN_CTAS: u32 = 32;

/// Smallest output-channel count this engine will build a tile for.
///
/// The row tile shrinks to 32 for small channel counts (see
/// [`TiledConvPlan::for_problem`]); below 24 channels even a 32-row tile wastes
/// more than a quarter of every CTA.
pub const MIN_OUT_CHANNELS: u32 = 24;

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// The validated code-generation plan for one convolution problem.
///
/// Constructing this is the *entire* engine-selection decision, and it is
/// pure: no device, no allocation beyond the problem itself, so
/// [`TiledConvPlan::for_problem`] is unit-testable on any host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledConvPlan {
    /// Mainloop tiling.
    pub cfg: TiledMainloopConfig,
    /// Input channels consumed per k-step; `tile_k == channels_per_ktile * R * S`.
    pub channels_per_ktile: u32,
}

impl TiledConvPlan {
    /// Chooses a tiling for `problem`, or `None` when this engine declines it.
    ///
    /// Declines (each of which falls back to the scalar implicit-GEMM engine):
    ///
    /// * anything but f32 NCHW 2-D with `groups == 1` -- the transposed GEMM
    ///   orientation, the `st.global.v4.f32` epilogue and the `ld.shared.v4.f32`
    ///   fragments are all f32-NCHW-specific, and a grouped convolution would
    ///   need the group's channel offset folded into every staging address;
    /// * problems under the [`MIN_GEMM_K`] / [`MIN_GEMM_N`] /
    ///   [`MIN_OUT_CHANNELS`] profitability thresholds;
    /// * filter volumes whose `R*S` cannot produce a `tile_k` of at least 4
    ///   that also divides the channel count evenly (see
    ///   [`channels_per_ktile`](Self::channels_per_ktile));
    /// * filter volumes so large that even a 32-row tile overflows the static
    ///   shared-memory budget;
    /// * everything, when [`DISABLE_TILED_ENV`] is set.
    #[must_use]
    pub fn for_problem(problem: &ConvProblem) -> Option<Self> {
        if tiled_disabled() {
            return None;
        }
        if problem.input_type != PtxType::F32 || problem.output_type != PtxType::F32 {
            return None;
        }
        if problem.layout != TensorLayout::Nchw {
            return None;
        }
        if problem.groups != 1 {
            return None;
        }
        if problem.in_dims.len() != 2 || problem.filter_dims.len() != 2 {
            return None;
        }
        let out_dims = problem.output_dims().ok()?;
        let (out_h, out_w) = (*out_dims.first()?, *out_dims.get(1)?);

        let filter_h = problem.filter_dims[0];
        let filter_w = problem.filter_dims[1];
        let rs = filter_h.checked_mul(filter_w)?;
        let gemm_k = problem.in_channels.checked_mul(rs)?;
        let gemm_n = problem.batch.checked_mul(out_h)?.checked_mul(out_w)?;

        if gemm_k < MIN_GEMM_K {
            return None;
        }
        if problem.out_channels < MIN_OUT_CHANNELS {
            return None;
        }

        let channels_per_ktile = channels_per_ktile(problem.in_channels, rs)?;
        let tile_k = channels_per_ktile.checked_mul(rs)?;

        // Row tile: as tall as the channel count can keep busy, then shrunk
        // further if `tile_k` makes the staging tiles too big for static
        // shared memory (a 7x7 filter at 128 rows does not fit).
        let mut thread_m = if problem.out_channels >= 96 {
            8
        } else if problem.out_channels >= 48 {
            4
        } else {
            2
        };
        loop {
            let cfg = TiledMainloopConfig {
                thread_m,
                thread_n: 8,
                tile_k,
            };
            if cfg.validate().is_ok() {
                // Parallelism check, once the tile is known: either the output
                // is wide enough on its own, or the resulting grid is big
                // enough to fill the device anyway.
                let ctas = problem
                    .out_channels
                    .div_ceil(cfg.tile_m())
                    .saturating_mul(gemm_n.div_ceil(cfg.tile_n()));
                if gemm_n < MIN_GEMM_N && ctas < MIN_CTAS {
                    return None;
                }
                return Some(Self {
                    cfg,
                    channels_per_ktile,
                });
            }
            if thread_m <= 2 {
                return None;
            }
            thread_m /= 2;
        }
    }

    /// Number of k-steps the mainloop runs: `C_in / channels_per_ktile`,
    /// exactly (the divisibility is guaranteed by
    /// [`channels_per_ktile`](fn@channels_per_ktile)).
    #[must_use]
    pub const fn k_tiles(&self, in_channels: u32) -> u32 {
        in_channels / self.channels_per_ktile
    }
}

/// Chooses how many input channels one k-step consumes.
///
/// The constraint that matters is `tile_k == ct * R * S` **and** `ct` divides
/// `C_in`: the first makes every staging slot's `(r, s)` loop-invariant, the
/// second makes the k-loop trip count exact so no partial k-step needs a
/// per-iteration bounds test. Within those, the largest `ct` that keeps
/// `tile_k` in a sane range wins, because a longer k-step amortises the two
/// barriers over more FMAs.
///
/// Returns `None` when no admissible `ct` produces a `tile_k` of at least 4
/// (below which the barrier overhead is not worth paying).
#[must_use]
pub fn channels_per_ktile(in_channels: u32, rs: u32) -> Option<u32> {
    if rs == 0 || in_channels == 0 {
        return None;
    }
    if rs >= 8 {
        // R*S alone is already a healthy k-step (3x3 = 9, 5x5 = 25, 7x7 = 49).
        return Some(1);
    }
    let want = 8u32.div_ceil(rs).max(1);
    (1..=want)
        .rev()
        .find(|&ct| in_channels % ct == 0 && ct * rs >= 4)
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Tiled implicit-GEMM forward convolution engine.
pub struct TiledImplicitGemmConv {
    problem: ConvProblem,
    plan: TiledConvPlan,
    sm_version: SmVersion,
}

impl TiledImplicitGemmConv {
    /// Builds the engine for `problem`, or `None` if the tiling declines it.
    #[must_use]
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> Option<Self> {
        let plan = TiledConvPlan::for_problem(&problem)?;
        Some(Self {
            problem,
            plan,
            sm_version,
        })
    }

    /// The plan this engine was built with.
    #[must_use]
    pub const fn plan(&self) -> TiledConvPlan {
        self.plan
    }

    /// Kernel entry name, which is also the compiled-module cache key.
    ///
    /// # Cache-key contract
    ///
    /// Every constant [`Self::generate_ptx`] bakes into the instruction stream
    /// appears here. That is a longer list than the scalar engine's, because
    /// this kernel folds padding, stride and dilation into immediates as well
    /// (they select strength-reduced address arithmetic, and their cardinality
    /// across a real model is tiny -- a handful of modules, at roughly 200 us
    /// of JIT each, amortised over every frame of the session):
    ///
    /// | Code-gen constant | Encoded as |
    /// |---|---|
    /// | tile geometry | `{tile_m}x{tile_n}x{tile_k}` |
    /// | filter extent `R`, `S` (unrolled tap decomposition) | `{r}x{s}` |
    /// | `C_in`, `C_out` (address immediates, loop trip count) | `c{c}k{k}` |
    /// | padding / stride / dilation | `p`, `s`, `d` fields |
    /// | channels per k-step | `ct{n}` |
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let p = &self.problem;
        let cfg = self.plan.cfg;
        format!(
            "tiled_igemm_conv_{tm}x{tn}x{tk}_{r}x{s}_c{cin}k{cout}_p{ph}x{pw}_s{sh}x{sw}_d{dh}x{dw}_ct{ct}_f32_nchw",
            tm = cfg.tile_m(),
            tn = cfg.tile_n(),
            tk = cfg.tile_k,
            r = p.filter_dims[0],
            s = p.filter_dims[1],
            cin = p.in_channels,
            cout = p.out_channels,
            ph = p.padding[0],
            pw = p.padding.get(1).copied().unwrap_or(0),
            sh = p.stride[0],
            sw = p.stride.get(1).copied().unwrap_or(1),
            dh = p.dilation[0],
            dw = p.dilation.get(1).copied().unwrap_or(1),
            ct = self.plan.channels_per_ktile,
        )
    }

    /// Generates the complete PTX module.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if the mainloop emitter rejects the
    /// tiling or the kernel fails to assemble into text.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let geom = KernelGeometry::from_problem(&self.problem, self.plan);
        let cfg = self.plan.cfg;
        let kb = KernelBuilder::new(&self.kernel_name())
            .target(self.sm_version)
            .max_threads_per_block(TILED_MAINLOOP_THREADS)
            .param("input", PtxType::U64)
            .param("filter", PtxType::U64)
            .param("output", PtxType::U64)
            .param("bias", PtxType::U64)
            .param("in_h", PtxType::U32)
            .param("in_w", PtxType::U32)
            .param("out_h", PtxType::U32)
            .param("out_w", PtxType::U32)
            .param("batch", PtxType::U32)
            .shared_mem_aligned("conv_smem_a", PtxType::F32, cfg.smem_a_elems() as usize, 16)
            .shared_mem_aligned("conv_smem_b", PtxType::F32, cfg.smem_b_elems() as usize, 16);

        kb.body(move |b| emit_tiled_conv_body(b, geom))
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))
    }

    /// Executes the convolution, optionally adding a per-output-channel bias
    /// in the epilogue.
    ///
    /// # Errors
    ///
    /// Returns errors from PTX generation, module loading, or kernel launch.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        bias: Option<&TensorDesc<T>>,
        output: &mut TensorDescMut<T>,
    ) -> DnnResult<()> {
        let entry = self.kernel_name();
        let kernel =
            handle.get_or_compile_kernel(&cache_key(&entry, self.sm_version), &entry, || {
                self.generate_ptx()
            })?;

        let out_dims = self.problem.output_dims()?;
        let out_h = out_dims.first().copied().unwrap_or(1);
        let out_w = out_dims.get(1).copied().unwrap_or(1);
        let gemm_n = self
            .problem
            .batch
            .saturating_mul(out_h)
            .saturating_mul(out_w);

        let cfg = self.plan.cfg;
        let grid = Dim3::new(
            gemm_n.div_ceil(cfg.tile_n()).max(1),
            self.problem.out_channels.div_ceil(cfg.tile_m()).max(1),
            1,
        );
        let params = LaunchParams::new(grid, Dim3::new(TILED_MAINLOOP_THREADS, 1, 1));

        let args = (
            input.ptr,
            filter.ptr,
            output.ptr,
            bias.map_or(0u64, |b| b.ptr),
            self.problem.in_dims[0],
            self.problem.in_dims[1],
            out_h,
            out_w,
            self.problem.batch,
        );

        kernel
            .kernel()
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;
        Ok(())
    }

    /// Workspace bytes required (none: the staging tiles are static shared
    /// memory).
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Code-generation geometry
// ---------------------------------------------------------------------------

/// Every convolution constant the emitted kernel bakes in as an immediate.
///
/// `Copy` so the `'static` body closure can capture it by value.
#[derive(Debug, Clone, Copy)]
struct KernelGeometry {
    cfg: TiledMainloopConfig,
    /// Input channels per k-step.
    channels_per_ktile: u32,
    /// Filter width `S`: the divisor that splits a tap index into `(r, s)`.
    /// The height never appears on its own -- `R` only enters through `rs`.
    filter_w: u32,
    /// `R * S`, the tap count of one input channel.
    rs: u32,
    /// Input / output channel counts (`groups == 1`, so these are the whole
    /// tensors' channel dims).
    in_channels: u32,
    out_channels: u32,
    pad_h: u32,
    pad_w: u32,
    stride_h: u32,
    stride_w: u32,
    dil_h: u32,
    dil_w: u32,
}

impl KernelGeometry {
    fn from_problem(problem: &ConvProblem, plan: TiledConvPlan) -> Self {
        let filter_h = problem.filter_dims[0];
        let filter_w = problem.filter_dims.get(1).copied().unwrap_or(1);
        Self {
            cfg: plan.cfg,
            channels_per_ktile: plan.channels_per_ktile,
            filter_w,
            rs: filter_h * filter_w,
            in_channels: problem.in_channels,
            out_channels: problem.out_channels,
            pad_h: problem.padding[0],
            pad_w: problem.padding.get(1).copied().unwrap_or(0),
            stride_h: problem.stride[0],
            stride_w: problem.stride.get(1).copied().unwrap_or(1),
            dil_h: problem.dilation[0],
            dil_w: problem.dilation.get(1).copied().unwrap_or(1),
        }
    }

    /// GEMM depth: `C_in * R * S`.
    const fn gemm_k(&self) -> u32 {
        self.in_channels * self.rs
    }

    /// k-loop trip count.
    const fn k_tiles(&self) -> u32 {
        self.in_channels / self.channels_per_ktile
    }

    /// Whether the output-channel count divides the row tile exactly, in which
    /// case no CTA is ever partial along `m` and the row bounds tests can be
    /// dropped entirely.
    const fn rows_exact(&self) -> bool {
        self.out_channels % self.cfg.tile_m() == 0
    }
}

// ---------------------------------------------------------------------------
// Kernel body
// ---------------------------------------------------------------------------

/// Emits the whole kernel: preamble, mainloop taps, and epilogue.
fn emit_tiled_conv_body(b: &mut BodyBuilder<'_>, geom: KernelGeometry) {
    let cfg = geom.cfg;
    b.comment("=== Tiled implicit-GEMM convolution (cross-correlation, NCHW) ===");

    let input_ptr = b.load_param_u64("input");
    let filter_ptr = b.load_param_u64("filter");
    let output_ptr = b.load_param_u64("output");
    let bias_ptr = b.load_param_u64("bias");
    let in_h = b.load_param_u32("in_h");
    let in_w = b.load_param_u32("in_w");
    let out_h = b.load_param_u32("out_h");
    let out_w = b.load_param_u32("out_w");
    let batch = b.load_param_u32("batch");

    // Output pixels per batch item, and the total GEMM column count.
    let pq = b.mul_lo_u32(out_h, out_w.clone());
    let gemm_n = b.mul_lo_u32(batch, pq.clone());

    // CTA tile origin. The grid is (column tiles, row tiles).
    let block_col = b.alloc_reg(PtxType::U32);
    let block_row = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {block_col}, %ctaid.x;"));
    b.raw_ptx(&format!("mov.u32 {block_row}, %ctaid.y;"));
    let row0 = b.alloc_reg(PtxType::U32);
    let col0 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "mul.lo.u32 {row0}, {block_row}, {};",
        cfg.tile_m()
    ));
    b.raw_ptx(&format!(
        "mul.lo.u32 {col0}, {block_col}, {};",
        cfg.tile_n()
    ));

    let n_ktiles = b.mov_imm_u32(geom.k_tiles());

    // Byte stride between k-steps.
    //   filter: tile_k contiguous taps.
    //   input : channels_per_ktile whole feature-map planes.
    let a_step = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {a_step}, {};", u64::from(cfg.tile_k) * 4));
    let hw = b.mul_lo_u32(in_h.clone(), in_w.clone());
    let b_step32 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "mul.lo.u32 {b_step32}, {hw}, {};",
        geom.channels_per_ktile * 4
    ));
    let b_step = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {b_step}, {b_step32};"));

    let filter_taps = FilterTapCtx {
        geom,
        ptr: filter_ptr,
        row0: row0.clone(),
        step: a_step,
    };
    let input_taps = InputTapCtx {
        geom,
        ptr: input_ptr,
        col0: col0.clone(),
        gemm_n: gemm_n.clone(),
        pq: pq.clone(),
        out_w: out_w.clone(),
        in_h,
        in_w,
        step: b_step,
    };

    let acc = match emit_tiled_mainloop(
        b,
        cfg,
        "conv_smem_a",
        "conv_smem_b",
        &n_ktiles,
        |b, slots| emit_filter_taps(b, &filter_taps, slots),
        |b, slots| emit_input_taps(b, &input_taps, slots),
    ) {
        Ok(acc) => acc,
        Err(e) => {
            // Unreachable: `TiledConvPlan::for_problem` only ever produces
            // configurations `validate()` accepts, and it is the sole
            // constructor of this engine. Surfaced as a comment rather than a
            // panic so a future plan change fails visibly in the PTX instead of
            // aborting a caller's inference run.
            b.comment(&format!("tiled mainloop emission failed: {e}"));
            b.ret();
            return;
        }
    };

    emit_epilogue(
        b,
        geom,
        &acc,
        &EpilogueCtx {
            output_ptr,
            bias_ptr,
            row0,
            col0,
            pq,
            gemm_n,
        },
    );
    b.ret();
}

// ---------------------------------------------------------------------------
// A-operand (filter) taps
// ---------------------------------------------------------------------------

/// Everything the filter-side address callback needs.
struct FilterTapCtx {
    geom: KernelGeometry,
    ptr: Register,
    row0: Register,
    step: Register,
}

/// `A[m][kk] = filter[(row0 + m) * (C_in*R*S) + kk]` -- the filter tensor read
/// directly as a row-major `C_out x (C_in*R*S)` matrix, no rearrangement.
fn emit_filter_taps(
    b: &mut BodyBuilder<'_>,
    ctx: &FilterTapCtx,
    slots: &[StageSlot],
) -> Vec<GlobalTap> {
    let gemm_k = ctx.geom.gemm_k();
    let rows_exact = ctx.geom.rows_exact();
    let out_channels = ctx.geom.out_channels;
    slots
        .iter()
        .map(|slot| {
            let gm = b.add_u32(ctx.row0.clone(), slot.outer.clone());
            let valid = if rows_exact {
                None
            } else {
                let p = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {p}, {gm}, {out_channels};"));
                Some(p)
            };
            let gemm_k_reg = b.mov_imm_u32(gemm_k);
            let idx = b.mad_lo_u32(gm, gemm_k_reg, slot.inner.clone());
            let ptr = b.byte_offset_addr(ctx.ptr.clone(), idx, 4);
            GlobalTap {
                ptr,
                advance_bytes: ctx.step.clone(),
                valid,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// B-operand (implicit im2col) taps
// ---------------------------------------------------------------------------

/// Everything the activation-side address callback needs.
struct InputTapCtx {
    geom: KernelGeometry,
    ptr: Register,
    col0: Register,
    gemm_n: Register,
    pq: Register,
    out_w: Register,
    in_h: Register,
    in_w: Register,
    step: Register,
}

/// `B[kk][n] = input[b][c][ih][iw]`, the implicit im2col.
///
/// All of it -- the `(b, oh, ow)` decomposition of the column, the `(c, r, s)`
/// decomposition of the tap, the padded-input bounds predicate, the address --
/// is emitted **once**, outside the k-loop: see the module docs on why
/// `tile_k` is a multiple of `R*S`.
fn emit_input_taps(
    b: &mut BodyBuilder<'_>,
    ctx: &InputTapCtx,
    slots: &[StageSlot],
) -> Vec<GlobalTap> {
    let geom = ctx.geom;
    slots
        .iter()
        .map(|slot| {
            let gn = b.add_u32(ctx.col0.clone(), slot.outer.clone());
            let col_ok = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lo.u32 {col_ok}, {gn}, {};", ctx.gemm_n));

            // Column -> (batch item, oh, ow).
            let nb = b.alloc_reg(PtxType::U32);
            let rem = b.alloc_reg(PtxType::U32);
            let oh = b.alloc_reg(PtxType::U32);
            let ow = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("div.u32 {nb}, {gn}, {};", ctx.pq));
            b.raw_ptx(&format!("rem.u32 {rem}, {gn}, {};", ctx.pq));
            b.raw_ptx(&format!("div.u32 {oh}, {rem}, {};", ctx.out_w));
            b.raw_ptx(&format!("rem.u32 {ow}, {rem}, {};", ctx.out_w));

            // Tap -> (channel offset within the k-step, r, s). All divisors are
            // code-gen constants.
            let c_local = b.alloc_reg(PtxType::U32);
            let rs_local = b.alloc_reg(PtxType::U32);
            let r = b.alloc_reg(PtxType::U32);
            let s = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("div.u32 {c_local}, {}, {};", slot.inner, geom.rs));
            b.raw_ptx(&format!("rem.u32 {rs_local}, {}, {};", slot.inner, geom.rs));
            b.raw_ptx(&format!("div.u32 {r}, {rs_local}, {};", geom.filter_w));
            b.raw_ptx(&format!("rem.u32 {s}, {rs_local}, {};", geom.filter_w));

            // ih = oh*stride_h - pad_h + r*dilation_h, and likewise for iw. The
            // unsigned compare doubles as the `0 <= ih < H` bounds test: a
            // negative coordinate wraps to a huge unsigned value.
            let ih = emit_input_coord(b, &oh, &r, geom.stride_h, geom.pad_h, geom.dil_h);
            let iw = emit_input_coord(b, &ow, &s, geom.stride_w, geom.pad_w, geom.dil_w);
            let ih_ok = b.alloc_reg(PtxType::Pred);
            let iw_ok = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lo.u32 {ih_ok}, {ih}, {};", ctx.in_h));
            b.raw_ptx(&format!("setp.lo.u32 {iw_ok}, {iw}, {};", ctx.in_w));
            let spatial_ok = b.alloc_reg(PtxType::Pred);
            let valid = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("and.pred {spatial_ok}, {ih_ok}, {iw_ok};"));
            b.raw_ptx(&format!("and.pred {valid}, {spatial_ok}, {col_ok};"));

            // input[((nb * C_in + c_local) * H + ih) * W + iw]
            let cin_reg = b.mov_imm_u32(geom.in_channels);
            let plane = b.mad_lo_u32(nb, cin_reg, c_local);
            let row = b.mad_lo_u32(plane, ctx.in_h.clone(), ih);
            let idx = b.mad_lo_u32(row, ctx.in_w.clone(), iw);
            // Out-of-range coordinates must not produce an address that is
            // dereferenced -- the mainloop predicates the load off -- but the
            // *address* is still computed, so clamp it into the buffer to keep
            // the arithmetic well defined.
            let zero = b.mov_imm_u32(0);
            let safe_idx = b.selp(PtxType::U32, idx, zero, valid.clone());
            let ptr = b.byte_offset_addr(ctx.ptr.clone(), safe_idx, 4);
            GlobalTap {
                ptr,
                advance_bytes: ctx.step.clone(),
                valid: Some(valid),
            }
        })
        .collect()
}

/// `out_pos * stride - pad + tap * dilation`, as an unsigned value that wraps
/// on the negative (padded) side.
fn emit_input_coord(
    b: &mut BodyBuilder<'_>,
    out_pos: &Register,
    tap: &Register,
    stride: u32,
    pad: u32,
    dilation: u32,
) -> Register {
    let acc = b.alloc_reg(PtxType::U32);
    if stride == 1 {
        b.raw_ptx(&format!("mov.u32 {acc}, {out_pos};"));
    } else {
        b.raw_ptx(&format!("mul.lo.u32 {acc}, {out_pos}, {stride};"));
    }
    if dilation == 1 {
        b.raw_ptx(&format!("add.u32 {acc}, {acc}, {tap};"));
    } else {
        b.raw_ptx(&format!("mad.lo.u32 {acc}, {tap}, {dilation}, {acc};"));
    }
    if pad != 0 {
        b.raw_ptx(&format!("sub.u32 {acc}, {acc}, {pad};"));
    }
    acc
}

// ---------------------------------------------------------------------------
// Epilogue
// ---------------------------------------------------------------------------

/// Everything the epilogue needs from the preamble.
struct EpilogueCtx {
    output_ptr: Register,
    bias_ptr: Register,
    row0: Register,
    col0: Register,
    pq: Register,
    gemm_n: Register,
}

/// Emits the bias add and the output store.
///
/// Two store paths, chosen by one CTA-uniform branch:
///
/// * **interior** -- the CTA's whole tile is in range and `P*Q` is a multiple
///   of 4, so each 4-wide column group is 4 contiguous, 16-byte-aligned output
///   elements: one `st.global.v4.f32` per group per row.
/// * **boundary** -- fully predicated scalar stores.
///
/// The `P*Q % 4` term is what makes the fast path safe for `batch > 1`: it is
/// exactly the condition under which a 4-aligned column group cannot straddle
/// two batch items (which are `P*Q` apart in a channel-major output).
fn emit_epilogue(
    b: &mut BodyBuilder<'_>,
    geom: KernelGeometry,
    acc: &TiledAccumulators,
    ctx: &EpilogueCtx,
) {
    let cfg = geom.cfg;
    b.comment("=== Epilogue: bias, then NCHW store ===");

    // This thread's first output row / column within the whole GEMM.
    let gm0 = b.add_u32(ctx.row0.clone(), acc.row_base().clone());
    let gn0 = b.add_u32(ctx.col0.clone(), acc.col_base().clone());

    emit_bias_add(b, geom, acc, ctx, &gm0);

    // Interior test: the whole CTA tile is inside the GEMM, and the output's
    // spatial extent keeps 4-wide column groups contiguous and aligned.
    let row_end = b.alloc_reg(PtxType::U32);
    let col_end = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "add.u32 {row_end}, {}, {};",
        ctx.row0,
        cfg.tile_m()
    ));
    b.raw_ptx(&format!(
        "add.u32 {col_end}, {}, {};",
        ctx.col0,
        cfg.tile_n()
    ));
    let rows_in = b.alloc_reg(PtxType::Pred);
    let cols_in = b.alloc_reg(PtxType::Pred);
    let pq_aligned = b.alloc_reg(PtxType::Pred);
    let pq_low = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "setp.ls.u32 {rows_in}, {row_end}, {};",
        geom.out_channels
    ));
    b.raw_ptx(&format!(
        "setp.ls.u32 {cols_in}, {col_end}, {};",
        ctx.gemm_n
    ));
    b.raw_ptx(&format!("and.b32 {pq_low}, {}, 3;", ctx.pq));
    b.raw_ptx(&format!("setp.eq.u32 {pq_aligned}, {pq_low}, 0;"));
    let interior = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("and.pred {interior}, {rows_in}, {cols_in};"));
    b.raw_ptx(&format!("and.pred {interior}, {interior}, {pq_aligned};"));

    let boundary_label = b.fresh_label("conv_epi_boundary");
    let end_label = b.fresh_label("conv_epi_end");
    b.raw_ptx(&format!("@!{interior} bra {boundary_label};"));
    emit_vector_store(b, geom, acc, ctx, &gm0, &gn0);
    b.raw_ptx(&format!("bra {end_label};"));
    b.raw_ptx(&format!("{boundary_label}:"));
    emit_scalar_store(b, geom, acc, ctx, &gm0, &gn0);
    b.raw_ptx(&format!("{end_label}:"));
}

/// Adds `bias[k]` to every accumulator of row `k`, when a bias pointer was
/// supplied.
///
/// Branched rather than folded into an unconditional `+ 0.0` so a bias-free
/// convolution is bit-identical to the scalar engine's (which skips the add
/// entirely), signed zeros included.
fn emit_bias_add(
    b: &mut BodyBuilder<'_>,
    geom: KernelGeometry,
    acc: &TiledAccumulators,
    ctx: &EpilogueCtx,
    gm0: &Register,
) {
    let cfg = geom.cfg;
    let has_bias = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ne.u64 {has_bias}, {}, 0;", ctx.bias_ptr));
    let skip = b.fresh_label("conv_no_bias");
    b.raw_ptx(&format!("@!{has_bias} bra {skip};"));

    for m in 0..cfg.thread_m {
        let row = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!(
            "add.u32 {row}, {gm0}, {};",
            TiledAccumulators::row_offset(m)
        ));
        let val = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("mov.f32 {val}, 0f00000000;"));
        let row_ok = b.alloc_reg(PtxType::Pred);
        b.raw_ptx(&format!(
            "setp.lo.u32 {row_ok}, {row}, {};",
            geom.out_channels
        ));
        let zero = b.mov_imm_u32(0);
        let safe_row = b.selp(PtxType::U32, row, zero, row_ok.clone());
        let addr = b.byte_offset_addr(ctx.bias_ptr.clone(), safe_row, 4);
        b.raw_ptx(&format!("@{row_ok} ld.global.f32 {val}, [{addr}];"));
        for n in 0..cfg.thread_n {
            let dst = acc.acc(m, n);
            b.raw_ptx(&format!("add.rn.f32 {dst}, {dst}, {val};"));
        }
    }
    b.raw_ptx(&format!("{skip}:"));
}

/// Interior-CTA store: one `st.global.v4.f32` per 4-wide column group per row.
fn emit_vector_store(
    b: &mut BodyBuilder<'_>,
    geom: KernelGeometry,
    acc: &TiledAccumulators,
    ctx: &EpilogueCtx,
    gm0: &Register,
    gn0: &Register,
) {
    let cfg = geom.cfg;
    b.comment("Interior CTA: vectorised NCHW store");
    // Byte stride between consecutive output channels of the same batch item.
    let row_stride = b.alloc_reg(PtxType::U64);
    let row_stride32 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shl.b32 {row_stride32}, {}, 2;", ctx.pq));
    b.raw_ptx(&format!("cvt.u64.u32 {row_stride}, {row_stride32};"));

    for g in 0..acc.col_groups() {
        let col = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!(
            "add.u32 {col}, {gn0}, {};",
            TiledAccumulators::col_offset(g * 4)
        ));
        // (batch item, spatial offset) of this column group.
        let nb = b.alloc_reg(PtxType::U32);
        let sp = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("div.u32 {nb}, {col}, {};", ctx.pq));
        b.raw_ptx(&format!("rem.u32 {sp}, {col}, {};", ctx.pq));
        // out[((nb * C_out + gm0) * P*Q) + sp]
        let cout_reg = b.mov_imm_u32(geom.out_channels);
        let chan = b.mad_lo_u32(nb, cout_reg, gm0.clone());
        let idx = b.mad_lo_u32(chan, ctx.pq.clone(), sp);
        let addr = b.byte_offset_addr(ctx.output_ptr.clone(), idx, 4);
        for m in 0..cfg.thread_m {
            b.raw_ptx(&format!(
                "st.global.v4.f32 [{addr}], {{{}, {}, {}, {}}};",
                acc.acc(m, g * 4),
                acc.acc(m, g * 4 + 1),
                acc.acc(m, g * 4 + 2),
                acc.acc(m, g * 4 + 3)
            ));
            if m + 1 < cfg.thread_m {
                b.raw_ptx(&format!("add.u64 {addr}, {addr}, {row_stride};"));
            }
        }
    }
}

/// Boundary-CTA store: every element bounds-tested, one scalar store each.
fn emit_scalar_store(
    b: &mut BodyBuilder<'_>,
    geom: KernelGeometry,
    acc: &TiledAccumulators,
    ctx: &EpilogueCtx,
    gm0: &Register,
    gn0: &Register,
) {
    let cfg = geom.cfg;
    b.comment("Boundary CTA: fully predicated scalar store");
    for m in 0..cfg.thread_m {
        let row = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!(
            "add.u32 {row}, {gm0}, {};",
            TiledAccumulators::row_offset(m)
        ));
        let row_ok = b.alloc_reg(PtxType::Pred);
        b.raw_ptx(&format!(
            "setp.lo.u32 {row_ok}, {row}, {};",
            geom.out_channels
        ));
        for n in 0..cfg.thread_n {
            let col = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "add.u32 {col}, {gn0}, {};",
                TiledAccumulators::col_offset(n)
            ));
            let col_ok = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lo.u32 {col_ok}, {col}, {};", ctx.gemm_n));
            let ok = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("and.pred {ok}, {row_ok}, {col_ok};"));

            let nb = b.alloc_reg(PtxType::U32);
            let sp = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("div.u32 {nb}, {col}, {};", ctx.pq));
            b.raw_ptx(&format!("rem.u32 {sp}, {col}, {};", ctx.pq));
            let cout_reg = b.mov_imm_u32(geom.out_channels);
            let chan = b.mad_lo_u32(nb, cout_reg, row.clone());
            let idx = b.mad_lo_u32(chan, ctx.pq.clone(), sp);
            let zero = b.mov_imm_u32(0);
            let safe_idx = b.selp(PtxType::U32, idx, zero, ok.clone());
            let addr = b.byte_offset_addr(ctx.output_ptr.clone(), safe_idx, 4);
            b.raw_ptx(&format!("@{ok} st.global.f32 [{addr}], {};", acc.acc(m, n)));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tiled_implicit_gemm_tests.rs"]
mod tests;
