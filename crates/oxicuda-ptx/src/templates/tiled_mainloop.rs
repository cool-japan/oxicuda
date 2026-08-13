//! Reusable CTA-tiled f32 GEMM mainloop emitter.
//!
//! Every GEMM-shaped kernel in this workspace (the forward convolutions in
//! `oxicuda-dnn`, the `GemmDispatcher` in `oxicuda-blas`) was, until this
//! module existed, emitted as **one thread per output element**: a scalar
//! accumulation loop that re-reads both operands from global memory for every
//! multiply-add. On an RTX A4000 that codegen measures 927-1065 GFLOPS on the
//! real face-pipeline convolution shapes -- the FMA *issue-rate* ceiling for a
//! body that spends ~14 instructions per useful `fma.rn.f32`, not a memory or
//! math limit.
//!
//! This module emits the standard register-blocked, shared-memory-staged
//! mainloop instead, and is deliberately **operand-agnostic**: it knows how to
//! tile, stage, synchronise and accumulate, but nothing about where the two
//! operands live. The caller supplies, per staging slot, a global address and
//! a validity predicate through the [`GlobalTap`] callback interface -- which
//! is what lets one emitter serve both a plain row-major GEMM and a
//! convolution's *implicit* im2col (where the "B" operand is a padded,
//! strided, dilated view of an activation tensor that is never materialised).
//!
//! # Geometry
//!
//! ```text
//! CTA           : 256 threads, addressed as a 16 x 16 grid
//!                 tx = tid % 16   (column direction)
//!                 ty = tid / 16   (row direction)
//! CTA tile      : tile_m x tile_n  =  (16 * thread_m) x (16 * thread_n)
//! K step        : tile_k
//! Register tile : thread_m x thread_n accumulators per thread
//! ```
//!
//! The canonical configuration is `thread_m = thread_n = 8`, `tile_k = 8`:
//! a 128x128x8 CTA tile with an 8x8 register tile per thread, i.e. 64
//! accumulators live across the whole mainloop.
//!
//! # Why this exact index mapping
//!
//! Three separate hardware constraints pin the mapping down; changing any one
//! of them regresses throughput, so they are documented rather than left to be
//! rediscovered:
//!
//! * **Row fragments broadcast.** A warp spans `tid = 32w .. 32w+31`, i.e.
//!   exactly two `ty` values. All 16 threads sharing a `ty` read the *same*
//!   `A` fragment address, so the row-side `ld.shared` is a two-address
//!   broadcast and is bank-conflict-free for *any* row mapping. The rows a
//!   thread owns are therefore laid out contiguously (`ty * thread_m + i`),
//!   which is also what makes the epilogue's output addresses a fixed stride
//!   apart.
//! * **Column fragments must be split.** The column side has 16 distinct
//!   addresses per warp, and `ld.shared.v4` is serviced in phases of 8 threads
//!   (8 x 16 B = the 128 B shared crossbar width). A contiguous
//!   `tx * thread_n` mapping puts `tx` and `tx + 4` in the same banks -- a
//!   2-way conflict on every column load. Splitting each thread's columns into
//!   `thread_n / 4` groups of 4, group `g` starting at
//!   `g * 64 + tx * 4` ([`COL_GROUP_STRIDE`]), makes the 8 threads of a phase
//!   cover banks 0..31 exactly once. This is the same split the classic
//!   128x128 SGEMM kernels use, and [`TiledAccumulators::col_offset`] is the
//!   only place it is encoded.
//! * **The A staging tile is padded.** Staging writes `As[k][m]` with `k`
//!   varying fastest across consecutive threads (that is what makes the
//!   *global* read of a row-major `A` coalesced), so an unpadded `tile_m`
//!   pitch -- always a multiple of 32 floats -- would land every one of those
//!   writes in the same bank. [`SMEM_A_PAD`] shifts each row by 4 floats,
//!   which breaks the bank aliasing while keeping the pitch a multiple of 4
//!   floats so the mainloop's `ld.shared.v4.f32` stays 16-byte aligned.
//!
//! # What the caller owns
//!
//! * Declaring the two shared arrays, with
//!   [`KernelBuilder::shared_mem_aligned`](crate::builder::KernelBuilder::shared_mem_aligned)
//!   (`align = 16`, sizes from [`TiledMainloopConfig::smem_a_elems`] /
//!   [`smem_b_elems`](TiledMainloopConfig::smem_b_elems)).
//! * Turning each [`StageSlot`]'s `(outer, inner)` tile coordinates into a
//!   global address plus validity predicate ([`GlobalTap`]). Anything the
//!   caller can hoist -- a padded convolution's bounds test, a batch
//!   decomposition -- is computed *once* here, outside the k-loop, because the
//!   emitter only ever advances the returned pointer by a fixed byte stride.
//! * The epilogue: the emitter returns the live [`TiledAccumulators`] and does
//!   not store them, because what an epilogue must do (bias, activation,
//!   `alpha`/`beta`, an NCHW address remap) is entirely caller-specific.

use crate::builder::BodyBuilder;
use crate::error::PtxGenError;
use crate::ir::{PtxType, Register};

// ---------------------------------------------------------------------------
// Fixed geometry constants
// ---------------------------------------------------------------------------

/// Threads per CTA for every configuration this emitter produces.
///
/// Fixed rather than derived: the 16x16 thread addressing, the staging
/// stride, and the bank-conflict argument in the module docs are all stated
/// in terms of a 256-thread CTA spanning exactly 8 warps.
pub const TILED_MAINLOOP_THREADS: u32 = 256;

/// Threads along the column (`n`) direction of the CTA's thread grid.
pub const THREADS_X: u32 = 16;

/// Threads along the row (`m`) direction of the CTA's thread grid.
pub const THREADS_Y: u32 = 16;

/// Column distance between consecutive 4-wide column groups owned by one
/// thread (see the module docs' bank-conflict argument).
pub const COL_GROUP_STRIDE: u32 = THREADS_X * 4;

/// Padding, in `f32` elements, added to the shared `A` tile's row pitch.
///
/// A multiple of 4 so the padded pitch stays 16-byte aligned for
/// `ld.shared.v4.f32`.
pub const SMEM_A_PAD: u32 = 4;

/// Largest static shared-memory allocation a CTA may declare without opting
/// in to the dynamic carve-out (`cudaFuncAttributeMaxDynamicSharedMemorySize`),
/// which this emitter deliberately does not use.
pub const MAX_STATIC_SMEM_BYTES: u32 = 48 * 1024;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tiling configuration for [`emit_tiled_mainloop`].
///
/// The CTA tile is derived, not given: `tile_m = 16 * thread_m` and
/// `tile_n = 16 * thread_n`, because the 256-thread CTA is addressed as a
/// fixed 16x16 grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledMainloopConfig {
    /// Rows of the CTA tile owned by each thread (register tile height).
    pub thread_m: u32,
    /// Columns of the CTA tile owned by each thread (register tile width).
    /// Must be a multiple of 4 -- the column fragments are loaded with
    /// `ld.shared.v4.f32`.
    pub thread_n: u32,
    /// Depth of one mainloop step along the reduction dimension.
    pub tile_k: u32,
}

impl TiledMainloopConfig {
    /// The canonical 128x128x8 tile with an 8x8 register tile per thread.
    #[must_use]
    pub const fn canonical() -> Self {
        Self {
            thread_m: 8,
            thread_n: 8,
            tile_k: 8,
        }
    }

    /// Rows of the CTA tile.
    #[must_use]
    pub const fn tile_m(&self) -> u32 {
        THREADS_Y * self.thread_m
    }

    /// Columns of the CTA tile.
    #[must_use]
    pub const fn tile_n(&self) -> u32 {
        THREADS_X * self.thread_n
    }

    /// Row pitch, in `f32` elements, of the shared `A` tile (`[tile_k][tile_m]`
    /// plus [`SMEM_A_PAD`]).
    #[must_use]
    pub const fn smem_a_pitch(&self) -> u32 {
        self.tile_m() + SMEM_A_PAD
    }

    /// `f32` element count the caller must declare for the shared `A` tile.
    #[must_use]
    pub const fn smem_a_elems(&self) -> u32 {
        self.tile_k * self.smem_a_pitch()
    }

    /// `f32` element count the caller must declare for the shared `B` tile.
    #[must_use]
    pub const fn smem_b_elems(&self) -> u32 {
        self.tile_k * self.tile_n()
    }

    /// Total static shared memory the two staging tiles occupy.
    #[must_use]
    pub const fn smem_bytes(&self) -> u32 {
        4 * (self.smem_a_elems() + self.smem_b_elems())
    }

    /// Number of staging iterations each thread performs for the `A` tile.
    #[must_use]
    pub const fn a_stage_iters(&self) -> u32 {
        (self.tile_m() * self.tile_k).div_ceil(TILED_MAINLOOP_THREADS)
    }

    /// Number of staging iterations each thread performs for the `B` tile.
    #[must_use]
    pub const fn b_stage_iters(&self) -> u32 {
        (self.tile_k * self.tile_n()).div_ceil(TILED_MAINLOOP_THREADS)
    }

    /// Accumulator registers held per thread across the whole mainloop.
    #[must_use]
    pub const fn accumulators(&self) -> u32 {
        self.thread_m * self.thread_n
    }

    /// Validates the configuration against every assumption the emitted code
    /// makes.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] naming the violated
    /// constraint.
    pub fn validate(&self) -> Result<(), PtxGenError> {
        if self.thread_m == 0 || self.thread_m > 8 {
            return Err(PtxGenError::GenerationFailed(format!(
                "tiled mainloop thread_m must be in 1..=8, got {}",
                self.thread_m
            )));
        }
        if self.thread_n == 0 || self.thread_n > 8 || self.thread_n % 4 != 0 {
            return Err(PtxGenError::GenerationFailed(format!(
                "tiled mainloop thread_n must be 4 or 8 (a multiple of 4, at most 8), got {}",
                self.thread_n
            )));
        }
        if self.tile_k == 0 {
            return Err(PtxGenError::GenerationFailed(
                "tiled mainloop tile_k must be non-zero".into(),
            ));
        }
        if self.smem_bytes() > MAX_STATIC_SMEM_BYTES {
            return Err(PtxGenError::GenerationFailed(format!(
                "tiled mainloop staging tiles need {} B of static shared memory, over the {} B limit",
                self.smem_bytes(),
                MAX_STATIC_SMEM_BYTES
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Staging interface
// ---------------------------------------------------------------------------

/// One element of one staging tile, as seen by the caller's address callback.
///
/// A thread stages `a_stage_iters` (resp. `b_stage_iters`) elements per
/// mainloop step; the *tile coordinates* of those elements never change from
/// step to step -- only the k-offset does, and the emitter applies that itself
/// by advancing [`GlobalTap::advance_bytes`]. That is the whole reason this is
/// a callback taking pre-computed coordinates rather than an in-loop hook:
/// every address computation the caller performs here is hoisted out of the
/// k-loop for free.
pub struct StageSlot {
    /// Tile coordinate along the non-reduction dimension:
    /// `0..tile_m` for the `A` tile, `0..tile_n` for the `B` tile.
    pub outer: Register,
    /// Tile coordinate along the reduction dimension, `0..tile_k`.
    pub inner: Register,
    /// Runtime predicate that is true when this slot maps to a real element of
    /// the tile.
    ///
    /// `None` when the tile divides evenly into `256`-element staging passes,
    /// i.e. when the slot is *always* within the tile. A `Some` guard means
    /// `outer` / `inner` may be out of range for this thread, so the caller's
    /// address arithmetic may produce a nonsense (but never dereferenced)
    /// address -- the emitter ANDs the guard into the load predicate.
    pub guard: Option<Register>,
    /// Shared-memory byte address this slot's value is written to.
    smem_addr: Register,
}

/// The caller's answer for one [`StageSlot`]: where to read the element from.
pub struct GlobalTap {
    /// 64-bit global address of the element for the **first** k-step.
    ///
    /// The emitter mutates this register in place, advancing it by
    /// `advance_bytes` after every k-step, so it must be a register the caller
    /// does not reuse for anything else.
    pub ptr: Register,
    /// Byte distance between the addresses of consecutive k-steps.
    ///
    /// May be the *same* register for every slot of a tile (the emitter only
    /// reads it), which is how a caller whose slots share one stride pays for
    /// a single register instead of one per slot.
    pub advance_bytes: Register,
    /// Predicate that is true when the address is in bounds and the element is
    /// a real value.
    ///
    /// `None` means unconditionally valid. When false, zero is staged instead
    /// -- which is exactly what a convolution's zero padding needs, and what
    /// makes a partial K tile harmless.
    pub valid: Option<Register>,
}

// ---------------------------------------------------------------------------
// Accumulators
// ---------------------------------------------------------------------------

/// The live register tile produced by [`emit_tiled_mainloop`], plus the
/// per-thread tile coordinates an epilogue needs to address it.
pub struct TiledAccumulators {
    cfg: TiledMainloopConfig,
    regs: Vec<Register>,
    row_base: Register,
    col_base: Register,
}

impl TiledAccumulators {
    /// The configuration these accumulators were produced with.
    #[must_use]
    pub const fn config(&self) -> TiledMainloopConfig {
        self.cfg
    }

    /// The accumulator register holding CTA-tile element
    /// `(row_base + m, col_base + col_offset(n))`.
    ///
    /// # Panics
    ///
    /// Panics if `m >= thread_m` or `n >= thread_n`.
    #[must_use]
    pub fn acc(&self, m: u32, n: u32) -> &Register {
        assert!(
            m < self.cfg.thread_m && n < self.cfg.thread_n,
            "accumulator ({m}, {n}) is outside the {}x{} register tile",
            self.cfg.thread_m,
            self.cfg.thread_n
        );
        let idx = (m * self.cfg.thread_n + n) as usize;
        &self.regs[idx]
    }

    /// Register holding this thread's first CTA-tile **row**, `ty * thread_m`.
    #[must_use]
    pub const fn row_base(&self) -> &Register {
        &self.row_base
    }

    /// Register holding this thread's first CTA-tile **column**, `tx * 4`.
    #[must_use]
    pub const fn col_base(&self) -> &Register {
        &self.col_base
    }

    /// CTA-tile row of register-tile row `m`, relative to
    /// [`row_base`](Self::row_base). Rows are contiguous.
    #[must_use]
    pub const fn row_offset(m: u32) -> u32 {
        m
    }

    /// CTA-tile column of register-tile column `n`, relative to
    /// [`col_base`](Self::col_base).
    ///
    /// Columns are **not** contiguous: they come in groups of 4 spaced
    /// [`COL_GROUP_STRIDE`] apart -- see the module docs for why.
    #[must_use]
    pub const fn col_offset(n: u32) -> u32 {
        (n / 4) * COL_GROUP_STRIDE + (n % 4)
    }

    /// Number of 4-wide column groups each thread owns.
    #[must_use]
    pub const fn col_groups(&self) -> u32 {
        self.cfg.thread_n / 4
    }
}

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

/// Emits the complete staged, register-blocked mainloop.
///
/// On return the k-loop has run to completion and every accumulator holds the
/// full dot product for its output element; the caller emits the epilogue.
///
/// `smem_a` / `smem_b` name shared arrays the caller must have declared with
/// [`TiledMainloopConfig::smem_a_elems`] /
/// [`smem_b_elems`](TiledMainloopConfig::smem_b_elems) `f32` elements and
/// 16-byte alignment. `n_ktiles` is a **CTA-uniform** `u32` register giving the
/// number of k-steps: the loop contains `bar.sync`, so a per-thread trip count
/// would deadlock.
///
/// # Errors
///
/// Returns [`PtxGenError::GenerationFailed`] if `cfg` is invalid, or if either
/// callback returns a number of taps other than one per [`StageSlot`].
pub fn emit_tiled_mainloop<FA, FB>(
    b: &mut BodyBuilder<'_>,
    cfg: TiledMainloopConfig,
    smem_a: &str,
    smem_b: &str,
    n_ktiles: &Register,
    a_taps: FA,
    b_taps: FB,
) -> Result<TiledAccumulators, PtxGenError>
where
    FA: FnOnce(&mut BodyBuilder<'_>, &[StageSlot]) -> Vec<GlobalTap>,
    FB: FnOnce(&mut BodyBuilder<'_>, &[StageSlot]) -> Vec<GlobalTap>,
{
    cfg.validate()?;

    b.comment(&format!(
        "=== CTA-tiled mainloop: {}x{}x{} tile, {}x{} register tile, {} threads ===",
        cfg.tile_m(),
        cfg.tile_n(),
        cfg.tile_k,
        cfg.thread_m,
        cfg.thread_n,
        TILED_MAINLOOP_THREADS
    ));

    let tid = b.thread_id_x();
    let tx = b.alloc_reg(PtxType::U32);
    let ty = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("and.b32 {tx}, {tid}, {};", THREADS_X - 1));
    b.raw_ptx(&format!(
        "shr.u32 {ty}, {tid}, {};",
        THREADS_X.trailing_zeros()
    ));

    // Shared-window base addresses. `mov.u32 %r, <shared symbol>` yields the
    // symbol's offset inside the shared window, which is what `ld.shared` /
    // `st.shared` take; 32-bit is enough for the whole window and keeps the
    // per-slot address arithmetic in 32-bit registers.
    let as_base = b.alloc_reg(PtxType::U32);
    let bs_base = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {as_base}, {smem_a};"));
    b.raw_ptx(&format!("mov.u32 {bs_base}, {smem_b};"));

    // Staging slot coordinates (loop-invariant, computed once), then the
    // caller's addresses for them.
    let a = StagedTile::build(b, cfg, &tid, &as_base, TileSide::A, a_taps)?;
    let bb = StagedTile::build(b, cfg, &tid, &bs_base, TileSide::B, b_taps)?;

    // Accumulators, zeroed before the first k-step.
    b.comment("Zero the register tile");
    let regs: Vec<Register> = (0..cfg.accumulators() as usize)
        .map(|_| b.alloc_reg(PtxType::F32))
        .collect();
    for reg in &regs {
        b.raw_ptx(&format!("mov.f32 {reg}, 0f00000000;"));
    }

    // Fragment base addresses: rows are contiguous from `ty * thread_m`,
    // columns start at `tx * 4` (the group stride is folded into each
    // instruction's immediate offset).
    let a_frag_base = b.alloc_reg(PtxType::U32);
    let b_frag_base = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "mad.lo.u32 {a_frag_base}, {ty}, {}, {as_base};",
        cfg.thread_m * 4
    ));
    b.raw_ptx(&format!("mad.lo.u32 {b_frag_base}, {tx}, 16, {bs_base};"));

    emit_k_loop(
        b,
        cfg,
        &a,
        &bb,
        n_ktiles,
        &FragBases {
            a: a_frag_base,
            b: b_frag_base,
        },
        &regs,
    );

    let row_base = b.alloc_reg(PtxType::U32);
    let col_base = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {row_base}, {ty}, {};", cfg.thread_m));
    b.raw_ptx(&format!("shl.b32 {col_base}, {tx}, 2;"));

    Ok(TiledAccumulators {
        cfg,
        regs,
        row_base,
        col_base,
    })
}

/// Which of the two operands a staged tile is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TileSide {
    /// Row operand, staged transposed as `As[k][m]`.
    A,
    /// Column operand, staged as `Bs[k][n]`.
    B,
}

/// One operand's staging state: where each slot goes in shared memory, where
/// it comes from in global memory, and the register it passes through.
struct StagedTile {
    slots: Vec<StageSlot>,
    taps: Vec<GlobalTap>,
    /// Load predicate per slot: the tile guard combined (`and.pred`) with the
    /// caller's validity predicate, folded once outside the k-loop.
    load_pred: Vec<Option<Register>>,
    /// One register per slot, live across the whole pipeline.
    staged: Vec<Register>,
}

impl StagedTile {
    fn build<F>(
        b: &mut BodyBuilder<'_>,
        cfg: TiledMainloopConfig,
        tid: &Register,
        smem_base: &Register,
        side: TileSide,
        taps_fn: F,
    ) -> Result<Self, PtxGenError>
    where
        F: FnOnce(&mut BodyBuilder<'_>, &[StageSlot]) -> Vec<GlobalTap>,
    {
        let slots = match side {
            TileSide::A => build_a_slots(b, tid, cfg, smem_base),
            TileSide::B => build_b_slots(b, tid, cfg, smem_base),
        };
        b.comment("Caller-supplied global taps");
        let taps = taps_fn(b, &slots);
        if taps.len() != slots.len() {
            return Err(PtxGenError::GenerationFailed(format!(
                "tap callback returned {} taps for {} staging slots",
                taps.len(),
                slots.len()
            )));
        }
        // An out-of-tile slot must not dereference the (meaningless) address
        // its coordinates produced, so the tile guard joins the caller's
        // predicate here rather than inside the loop.
        let load_pred = fold_guards(b, &slots, &taps);
        let staged: Vec<Register> = (0..slots.len())
            .map(|_| b.alloc_reg(PtxType::F32))
            .collect();
        Ok(Self {
            slots,
            taps,
            load_pred,
            staged,
        })
    }

    /// Issues this tile's global loads for the slice the pointers currently
    /// address.
    fn load(&self, b: &mut BodyBuilder<'_>) {
        emit_stage_loads(b, &self.staged, &self.taps, &self.load_pred);
    }

    /// Commits the loaded slice to shared memory.
    fn commit(&self, b: &mut BodyBuilder<'_>) {
        emit_stage_stores(b, &self.slots, &self.staged);
    }

    /// Advances this tile's global pointers by one K slice.
    fn advance(&self, b: &mut BodyBuilder<'_>) {
        emit_advance(b, &self.taps);
    }
}

/// The two loop-invariant shared-memory addresses the register-tile update
/// reads its fragments from: this thread's first row of the `A` tile and its
/// first column of the `B` tile. Every fragment load is one of these plus an
/// immediate offset.
struct FragBases {
    a: Register,
    b: Register,
}

/// Emits the register-prefetch pipeline: a prologue that puts the first K
/// slice's loads in flight, then a loop whose body commits the previous
/// slice, issues the next one's loads, and only then runs the register-tile
/// update.
///
/// The ordering is the whole point. A straightforward
/// `load; store; barrier; compute` loop exposes a full global-memory latency
/// on every k-step, and no amount of occupancy hides it: a 64-accumulator
/// register tile costs ~140 registers, which is one CTA per SM, so there is no
/// second CTA to run while this one waits.
fn emit_k_loop(
    b: &mut BodyBuilder<'_>,
    cfg: TiledMainloopConfig,
    a: &StagedTile,
    bt: &StagedTile,
    n_ktiles: &Register,
    frags: &FragBases,
    acc: &[Register],
) {
    b.comment("Prefetch the first K slice");
    a.load(b);
    bt.load(b);
    a.advance(b);
    bt.advance(b);

    let kt = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {kt}, 0;"));
    let loop_top = b.fresh_label("tiled_k");
    let loop_end = b.fresh_label("tiled_k_end");
    let no_prefetch = b.fresh_label("tiled_k_tail");
    b.raw_ptx(&format!("{loop_top}:"));
    let done = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.hs.u32 {done}, {kt}, {n_ktiles};"));
    b.raw_ptx(&format!("@{done} bra {loop_end};"));

    b.comment("Commit the prefetched K slice to shared memory");
    a.commit(b);
    bt.commit(b);
    b.bar_sync(0);

    // Skipped on the final iteration, where the addresses would run one slice
    // past the end of both operands.
    let more = b.alloc_reg(PtxType::U32);
    let has_more = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("add.u32 {more}, {kt}, 1;"));
    b.raw_ptx(&format!("setp.lo.u32 {has_more}, {more}, {n_ktiles};"));
    b.raw_ptx(&format!("@!{has_more} bra {no_prefetch};"));
    b.comment("Prefetch the next K slice");
    a.load(b);
    bt.load(b);
    b.raw_ptx(&format!("{no_prefetch}:"));
    a.advance(b);
    bt.advance(b);

    emit_inner_product(b, cfg, &frags.a, &frags.b, acc);
    // Second barrier: the next iteration's staging overwrites the tiles this
    // one just read.
    b.bar_sync(0);

    b.raw_ptx(&format!("add.u32 {kt}, {kt}, 1;"));
    b.raw_ptx(&format!("bra {loop_top};"));
    b.raw_ptx(&format!("{loop_end}:"));
}

// ---------------------------------------------------------------------------
// Staging helpers
// ---------------------------------------------------------------------------

/// Builds the `A`-tile staging slots.
///
/// Slot `i` of thread `tid` covers flat tile index `tid + i * 256`, decomposed
/// as `outer = idx / tile_k`, `inner = idx % tile_k`. That order -- `inner`
/// varying fastest across consecutive threads -- is what makes the caller's
/// global read of a k-contiguous `A` coalesced.
fn build_a_slots(
    b: &mut BodyBuilder<'_>,
    tid: &Register,
    cfg: TiledMainloopConfig,
    as_base: &Register,
) -> Vec<StageSlot> {
    let total = cfg.tile_m() * cfg.tile_k;
    let pitch = cfg.smem_a_pitch();
    (0..cfg.a_stage_iters())
        .map(|i| {
            let idx = flat_index(b, tid, i);
            let outer = b.alloc_reg(PtxType::U32);
            let inner = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("div.u32 {outer}, {idx}, {};", cfg.tile_k));
            b.raw_ptx(&format!("rem.u32 {inner}, {idx}, {};", cfg.tile_k));
            // As[inner][outer] -- transposed, so the mainloop reads a row of
            // consecutive `m` for a fixed `k`.
            let off = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {off}, {inner}, {pitch}, {outer};"));
            let smem_addr = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shl.b32 {off}, {off}, 2;"));
            b.raw_ptx(&format!("add.u32 {smem_addr}, {as_base}, {off};"));
            StageSlot {
                outer,
                inner,
                guard: tile_guard(b, &idx, i, total),
                smem_addr,
            }
        })
        .collect()
}

/// Builds the `B`-tile staging slots.
///
/// Slot `i` covers `outer = idx % tile_n`, `inner = idx / tile_n`: `outer`
/// (the column) varies fastest, so consecutive threads read consecutive
/// columns -- coalesced for a row-major `B`, and for a convolution's implicit
/// im2col, consecutive output pixels of the same row.
fn build_b_slots(
    b: &mut BodyBuilder<'_>,
    tid: &Register,
    cfg: TiledMainloopConfig,
    bs_base: &Register,
) -> Vec<StageSlot> {
    let tile_n = cfg.tile_n();
    let total = cfg.tile_k * tile_n;
    let shift = tile_n.trailing_zeros();
    debug_assert!(tile_n.is_power_of_two(), "tile_n is 16 * thread_n");
    (0..cfg.b_stage_iters())
        .map(|i| {
            let idx = flat_index(b, tid, i);
            let outer = b.alloc_reg(PtxType::U32);
            let inner = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("and.b32 {outer}, {idx}, {};", tile_n - 1));
            b.raw_ptx(&format!("shr.u32 {inner}, {idx}, {shift};"));
            let off = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {off}, {inner}, {tile_n}, {outer};"));
            let smem_addr = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shl.b32 {off}, {off}, 2;"));
            b.raw_ptx(&format!("add.u32 {smem_addr}, {bs_base}, {off};"));
            StageSlot {
                outer,
                inner,
                guard: tile_guard(b, &idx, i, total),
                smem_addr,
            }
        })
        .collect()
}

/// `idx = tid + i * TILED_MAINLOOP_THREADS`.
fn flat_index(b: &mut BodyBuilder<'_>, tid: &Register, i: u32) -> Register {
    let idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "add.u32 {idx}, {tid}, {};",
        i * TILED_MAINLOOP_THREADS
    ));
    idx
}

/// Returns the runtime "this slot is inside the tile" predicate, or `None`
/// when staging pass `i` is entirely within the tile for every thread.
fn tile_guard(b: &mut BodyBuilder<'_>, idx: &Register, i: u32, total: u32) -> Option<Register> {
    if (i + 1) * TILED_MAINLOOP_THREADS <= total {
        return None;
    }
    let guard = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {guard}, {idx}, {total};"));
    Some(guard)
}

/// ANDs each slot's tile guard with the caller's validity predicate, producing
/// the load predicate (`None` when the load is unconditional).
fn fold_guards(
    b: &mut BodyBuilder<'_>,
    slots: &[StageSlot],
    taps: &[GlobalTap],
) -> Vec<Option<Register>> {
    slots
        .iter()
        .zip(taps.iter())
        .map(|(slot, tap)| match (&slot.guard, &tap.valid) {
            (None, None) => None,
            (Some(g), None) => Some(g.clone()),
            (None, Some(v)) => Some(v.clone()),
            (Some(g), Some(v)) => {
                let both = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("and.pred {both}, {g}, {v};"));
                Some(both)
            }
        })
        .collect()
}

/// Emits the predicated global loads for one tile's staging pass.
///
/// The destination is pre-zeroed rather than `selp`-ed afterwards, so an
/// invalid slot stages an exact `+0.0` without ever issuing the load.
fn emit_stage_loads(
    b: &mut BodyBuilder<'_>,
    staged: &[Register],
    taps: &[GlobalTap],
    preds: &[Option<Register>],
) {
    for ((dst, tap), pred) in staged.iter().zip(taps.iter()).zip(preds.iter()) {
        match pred {
            Some(p) => {
                b.raw_ptx(&format!("mov.f32 {dst}, 0f00000000;"));
                b.raw_ptx(&format!("@{p} ld.global.f32 {dst}, [{}];", tap.ptr));
            }
            None => {
                b.raw_ptx(&format!("ld.global.f32 {dst}, [{}];", tap.ptr));
            }
        }
    }
}

/// Emits one tile's shared stores.
///
/// The store is predicated on the *tile guard only*, never on the caller's
/// validity predicate: an out-of-bounds-but-in-tile element must still write
/// its zero, or the slot would silently keep the previous k-step's value.
fn emit_stage_stores(b: &mut BodyBuilder<'_>, slots: &[StageSlot], staged: &[Register]) {
    for (slot, src) in slots.iter().zip(staged.iter()) {
        match &slot.guard {
            Some(g) => b.raw_ptx(&format!("@{g} st.shared.f32 [{}], {src};", slot.smem_addr)),
            None => b.raw_ptx(&format!("st.shared.f32 [{}], {src};", slot.smem_addr)),
        }
    }
}

/// Advances one tile's global pointers by one K slice.
///
/// Unconditional: this is pure address arithmetic, never a dereference, so
/// running it once past the end of the K range on the final iteration costs an
/// integer add and reads nothing.
fn emit_advance(b: &mut BodyBuilder<'_>, taps: &[GlobalTap]) {
    for tap in taps {
        b.raw_ptx(&format!(
            "add.u64 {0}, {0}, {1};",
            tap.ptr, tap.advance_bytes
        ));
    }
}

// ---------------------------------------------------------------------------
// Inner product
// ---------------------------------------------------------------------------

/// Emits the fully unrolled `tile_k x thread_m x thread_n` register-tile
/// update from the two staged shared tiles.
///
/// Every shared read uses an immediate byte offset off a loop-invariant base
/// register, so the unrolled body contains only loads and FMAs -- no address
/// arithmetic at all.
fn emit_inner_product(
    b: &mut BodyBuilder<'_>,
    cfg: TiledMainloopConfig,
    a_frag_base: &Register,
    b_frag_base: &Register,
    acc: &[Register],
) {
    b.comment("Register-tile update over the staged K slice");
    let a_pitch_bytes = cfg.smem_a_pitch() * 4;
    let b_pitch_bytes = cfg.tile_n() * 4;

    for k in 0..cfg.tile_k {
        let a_row = k * a_pitch_bytes;
        let b_row = k * b_pitch_bytes;

        // Row fragment: contiguous, so `thread_m / 4` vector loads (plus a
        // scalar tail when `thread_m` is not a multiple of 4).
        let mut a_frag: Vec<Register> = Vec::with_capacity(cfg.thread_m as usize);
        let a_vec_groups = cfg.thread_m / 4;
        for g in 0..a_vec_groups {
            let regs: [Register; 4] = std::array::from_fn(|_| b.alloc_reg(PtxType::F32));
            b.raw_ptx(&format!(
                "ld.shared.v4.f32 {{{}, {}, {}, {}}}, [{a_frag_base}+{}];",
                regs[0],
                regs[1],
                regs[2],
                regs[3],
                a_row + g * 16
            ));
            a_frag.extend(regs);
        }
        for m in (a_vec_groups * 4)..cfg.thread_m {
            let reg = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!(
                "ld.shared.f32 {reg}, [{a_frag_base}+{}];",
                a_row + m * 4
            ));
            a_frag.push(reg);
        }

        // Column fragment: `thread_n / 4` groups of 4, spaced COL_GROUP_STRIDE
        // columns apart (the bank-conflict split -- see the module docs).
        let mut b_frag: Vec<Register> = Vec::with_capacity(cfg.thread_n as usize);
        for g in 0..(cfg.thread_n / 4) {
            let regs: [Register; 4] = std::array::from_fn(|_| b.alloc_reg(PtxType::F32));
            b.raw_ptx(&format!(
                "ld.shared.v4.f32 {{{}, {}, {}, {}}}, [{b_frag_base}+{}];",
                regs[0],
                regs[1],
                regs[2],
                regs[3],
                b_row + g * COL_GROUP_STRIDE * 4
            ));
            b_frag.extend(regs);
        }

        for m in 0..cfg.thread_m as usize {
            for n in 0..cfg.thread_n as usize {
                let dst = &acc[m * cfg.thread_n as usize + n];
                b.raw_ptx(&format!(
                    "fma.rn.f32 {dst}, {}, {}, {dst};",
                    a_frag[m], b_frag[n]
                ));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tiled_mainloop_tests.rs"]
mod tests;
