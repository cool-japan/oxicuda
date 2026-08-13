//! PTX emitters for the Winograd F(2x2, 3x3) forward pipeline.
//!
//! Four kernels, all NCHW / FP32 / stride 1 / dilation 1 / groups 1:
//!
//! 1. [`input_transform_ptx`] — `V = B^T d B` per (channel, tile).
//! 2. [`filter_transform_ptx`] — `U = G g G^T` per (out-channel, in-channel).
//! 3. [`batched_gemm_ptx`] — the 16 independent `[K x C] * [C x P]` products,
//!    one per transform-domain position, in a single shared-memory-tiled
//!    launch (`blockIdx.z` selects the position).
//! 4. [`output_transform_ptx`] — `Y = A^T m A`, bias, boundary-clamped store.
//!
//! # Workspace layout
//!
//! With `alpha = 4`, `P = N * tiles_h * tiles_w` (the flattened tile index)
//! and `e` in `0..16` the transform-domain position `xi*4 + nu`:
//!
//! ```text
//! V[e][c][p]  at  transformed_input [ e * (C*P) + c * P + p ]
//! U[e][k][c]  at  transformed_filter[ e * (K*C) + k * C + c ]
//! M[e][k][p]  at  transformed_output[ e * (K*P) + k * P + p ]
//! ```
//!
//! Each region is 16 contiguous *planes*; a thread that owns one `(c, p)` (or
//! `(k, c)`, or `(k, p)`) pair walks all 16 by adding one plane in bytes to a
//! running address. Consecutive threads own consecutive `p`, so every global
//! access in the transform kernels is coalesced along the fastest axis.
//!
//! # Why one emitter drives all three transforms
//!
//! Input, filter and output stages are all `out = M x M^T`
//! (see [`super::matrices`]). [`emit_congruence`] emits exactly that from the
//! matrix constants, so the kernels and the `f64` host reference in
//! `matrices::congruence` are driven by the same numbers — there is no second
//! transcription of the coefficients to get wrong.

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::{BodyBuilder, KernelBuilder};
use oxicuda_ptx::ir::{PtxType, Register};

use crate::error::{DnnError, DnnResult};

use super::matrices::{at_f2x3_view, bt_f2x3_view, g_f2x3_view};

/// Square tile edge of the transform-domain GEMM (both M and N), and the
/// K-step per shared-memory stage. The block is `GEMM_TILE x GEMM_TILE`
/// threads, so this must satisfy `GEMM_TILE^2 <= 1024`.
pub(crate) const GEMM_TILE: u32 = 16;

/// Entry-point name of the input-transform kernel.
pub(crate) const INPUT_TRANSFORM_ENTRY: &str = "winograd_input_transform_f2x3_f32";
/// Entry-point name of the filter-transform kernel.
pub(crate) const FILTER_TRANSFORM_ENTRY: &str = "winograd_filter_transform_f2x3_f32";
/// Entry-point name of the transform-domain batched GEMM kernel.
pub(crate) const GEMM_ENTRY: &str = "winograd_batched_gemm_f2x3_f32_t16";
/// Entry-point name of the output-transform kernel.
pub(crate) const OUTPUT_TRANSFORM_ENTRY: &str = "winograd_output_transform_f2x3_f32";

// ---------------------------------------------------------------------------
// Small emission helpers
// ---------------------------------------------------------------------------

/// Formats an `f32` as a PTX hexadecimal float literal.
fn f32_lit(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

/// Emits `dst = sum_k coeffs[k] * src[k]` into a fresh `f32` register.
///
/// Coefficients of `0`, `+1` and `-1` degrade to nothing, an add and a
/// subtract respectively — which is the entire reason F(2,3) is cheap: `B^T`
/// and `A^T` contain only those three values, so the input and output
/// transforms cost adds alone. `G`'s halves become a `mul`/`fma` with an
/// immediate.
fn emit_lincomb(b: &mut BodyBuilder<'_>, coeffs: &[f32], src: &[Register]) -> Register {
    let acc = b.alloc_reg(PtxType::F32);
    let mut started = false;
    for (k, &c) in coeffs.iter().enumerate() {
        if c == 0.0 {
            continue;
        }
        let s = &src[k];
        if started {
            if c == 1.0 {
                b.raw_ptx(&format!("add.rn.f32 {acc}, {acc}, {s};"));
            } else if c == -1.0 {
                b.raw_ptx(&format!("sub.rn.f32 {acc}, {acc}, {s};"));
            } else {
                b.raw_ptx(&format!("fma.rn.f32 {acc}, {s}, {}, {acc};", f32_lit(c)));
            }
        } else {
            if c == 1.0 {
                b.raw_ptx(&format!("mov.f32 {acc}, {s};"));
            } else if c == -1.0 {
                b.raw_ptx(&format!("neg.f32 {acc}, {s};"));
            } else {
                b.raw_ptx(&format!("mul.rn.f32 {acc}, {s}, {};", f32_lit(c)));
            }
            started = true;
        }
    }
    if !started {
        b.raw_ptx(&format!("mov.f32 {acc}, {};", f32_lit(0.0)));
    }
    acc
}

/// Emits the congruence transform `out = M * x * M^T`.
///
/// `m` is `rows x cols`; `x` is `cols x cols` given row-major in registers;
/// the returned vector is `rows x rows` row-major. Mirrors
/// [`matrices::congruence`](super::matrices::congruence) instruction for
/// instruction, including the two-stage factorisation (which is what turns an
/// O(rows^2 cols^2) dense product into two O(rows cols) passes).
fn emit_congruence(b: &mut BodyBuilder<'_>, m: &[&[f32]], x: &[Register]) -> Vec<Register> {
    let rows = m.len();
    let cols = m[0].len();

    // Stage 1: t[i][j] = sum_k M[i][k] * x[k][j]   (rows x cols)
    let mut t: Vec<Register> = Vec::with_capacity(rows * cols);
    for m_row in m.iter().take(rows) {
        for j in 0..cols {
            let column: Vec<Register> = (0..cols).map(|k| x[k * cols + j].clone()).collect();
            t.push(emit_lincomb(b, m_row, &column));
        }
    }

    // Stage 2: out[i][j] = sum_k t[i][k] * M[j][k]   (rows x rows)
    let mut out: Vec<Register> = Vec::with_capacity(rows * rows);
    for i in 0..rows {
        let row: Vec<Register> = (0..cols).map(|k| t[i * cols + k].clone()).collect();
        for m_row in m.iter().take(rows) {
            out.push(emit_lincomb(b, m_row, &row));
        }
    }
    out
}

/// Emits the standard `if (gid >= total) return;` prologue.
///
/// Returns `(gid, total, exit_label)`. `total` is handed back rather than
/// re-loaded by the caller because it doubles as the *plane stride* in every
/// transform kernel: the flat thread id indexes one plane, and consecutive
/// transform-domain positions are exactly `total` elements apart.
fn emit_bounds_guard(b: &mut BodyBuilder<'_>, total_param: &str) -> (Register, Register, String) {
    let gid = b.global_thread_id_x();
    let total = b.load_param_u32(total_param);
    let exit = b.fresh_label("wg_exit");
    let p_in = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.hs.u32 {p_in}, {gid}, {total};"));
    b.raw_ptx(&format!("@{p_in} bra {exit};"));
    (gid, total, exit)
}

/// Emits `q = a / d; r = a % d` as a pair of fresh `u32` registers.
fn emit_divmod(b: &mut BodyBuilder<'_>, a: &Register, d: &Register) -> (Register, Register) {
    let q = b.alloc_reg(PtxType::U32);
    let r = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {q}, {a}, {d};"));
    b.raw_ptx(&format!("rem.u32 {r}, {a}, {d};"));
    (q, r)
}

/// Emits `dst = base + e * plane_bytes` as a running 64-bit address, calling
/// `emit` once per transform-domain position `e` in `0..count`.
fn for_each_plane(
    b: &mut BodyBuilder<'_>,
    base: &Register,
    plane_bytes: &Register,
    count: usize,
    mut emit: impl FnMut(&mut BodyBuilder<'_>, usize, &Register),
) {
    let addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {addr}, {base};"));
    for e in 0..count {
        if e > 0 {
            b.raw_ptx(&format!("add.u64 {addr}, {addr}, {plane_bytes};"));
        }
        let cur = addr.clone();
        emit(b, e, &cur);
    }
}

/// Emits `plane_bytes = (u64)elements * 4`.
fn emit_plane_bytes(b: &mut BodyBuilder<'_>, elements: &Register) -> Register {
    let plane = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mul.wide.u32 {plane}, {elements}, 4;"));
    plane
}

// ---------------------------------------------------------------------------
// 1. Input transform:  V = B^T d B
// ---------------------------------------------------------------------------

/// Generates the input-transform kernel.
///
/// One thread owns one `(c, p)` pair, gathers the 4x4 input patch rooted at
/// `(2*tile_h - pad_h, 2*tile_w - pad_w)` with implicit zero padding, and
/// scatters the 16 transformed values across the 16 planes of
/// `transformed_input`.
///
/// # Errors
///
/// Returns [`DnnError::PtxGeneration`] if the builder fails.
pub(crate) fn input_transform_ptx(sm: SmVersion) -> DnnResult<String> {
    KernelBuilder::new(INPUT_TRANSFORM_ENTRY)
        .target(sm)
        .param("input", PtxType::U64)
        .param("transformed", PtxType::U64)
        .param("in_channels", PtxType::U32)
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("pad_h", PtxType::U32)
        .param("pad_w", PtxType::U32)
        .param("tiles_h", PtxType::U32)
        .param("tiles_w", PtxType::U32)
        .param("tile_count", PtxType::U32)
        .param("total", PtxType::U32)
        .body(|b| {
            b.comment("=== Winograd F(2x2,3x3) input transform: V = B^T d B ===");
            let (gid, total, exit) = emit_bounds_guard(b, "total");

            let input_ptr = b.load_param_u64("input");
            let out_ptr = b.load_param_u64("transformed");
            let in_channels = b.load_param_u32("in_channels");
            let in_h = b.load_param_u32("in_h");
            let in_w = b.load_param_u32("in_w");
            let pad_h = b.load_param_u32("pad_h");
            let pad_w = b.load_param_u32("pad_w");
            let tiles_h = b.load_param_u32("tiles_h");
            let tiles_w = b.load_param_u32("tiles_w");
            let tile_count = b.load_param_u32("tile_count");

            // gid = c * P + p; p = ((n * tiles_h) + th) * tiles_w + tw.
            b.comment("Decompose gid -> (c, n, tile_h, tile_w)");
            let (c, p) = emit_divmod(b, &gid, &tile_count);
            let (t1, tw) = emit_divmod(b, &p, &tiles_w);
            let (n, th) = emit_divmod(b, &t1, &tiles_h);

            // nc_base = ((n * C) + c) * in_h * in_w.
            let nc = b.mad_lo_u32(n, in_channels, c);
            let hw = b.mul_lo_u32(in_h.clone(), in_w.clone());
            let nc_base = b.mul_lo_u32(nc, hw);

            // Tile origin in input coordinates (signed: padding may go < 0).
            let two = b.mov_imm_u32(2);
            let th2 = b.mul_lo_u32(th, two.clone());
            let tw2 = b.mul_lo_u32(tw, two);
            let base_h = b.alloc_reg(PtxType::S32);
            let base_w = b.alloc_reg(PtxType::S32);
            b.raw_ptx(&format!("sub.s32 {base_h}, {th2}, {pad_h};"));
            b.raw_ptx(&format!("sub.s32 {base_w}, {tw2}, {pad_w};"));

            // Per-row and per-column validity + row bases. An unsigned compare
            // of a negative s32 wraps to a huge value, so `setp.lo.u32` is a
            // complete `0 <= x < limit` test in one instruction.
            b.comment("Row/column bounds (unsigned compare doubles as >= 0 test)");
            let mut row_base = Vec::with_capacity(4);
            let mut row_ok = Vec::with_capacity(4);
            for i in 0..4u32 {
                let ih = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("add.s32 {ih}, {base_h}, {i};"));
                let p_h = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {p_h}, {ih}, {in_h};"));
                let base = b.mad_lo_u32(ih, in_w.clone(), nc_base.clone());
                row_base.push(base);
                row_ok.push(p_h);
            }
            let mut col_idx = Vec::with_capacity(4);
            let mut col_ok = Vec::with_capacity(4);
            for j in 0..4u32 {
                let iw = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("add.s32 {iw}, {base_w}, {j};"));
                let p_w = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {p_w}, {iw}, {in_w};"));
                col_idx.push(iw);
                col_ok.push(p_w);
            }

            b.comment("Gather the 4x4 patch (out-of-range taps read as zero)");
            let fzero = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("mov.f32 {fzero}, {};", f32_lit(0.0)));
            let zero_idx = b.mov_imm_u32(0);
            let mut d = Vec::with_capacity(16);
            for i in 0..4 {
                for j in 0..4 {
                    let p_valid = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!(
                        "and.pred {p_valid}, {}, {};",
                        row_ok[i], col_ok[j]
                    ));
                    let idx = b.add_u32(row_base[i].clone(), col_idx[j].clone());
                    let safe = b.selp(PtxType::U32, idx, zero_idx.clone(), p_valid.clone());
                    let addr = b.byte_offset_addr(input_ptr.clone(), safe, 4);
                    let raw = b.load_global_f32(addr);
                    d.push(b.selp(PtxType::F32, raw, fzero.clone(), p_valid));
                }
            }

            b.comment("V = B^T d B");
            let bt = bt_f2x3_view();
            let v = emit_congruence(b, &bt, &d);

            b.comment("Scatter V across the 16 transform-domain planes");
            let base_addr = b.byte_offset_addr(out_ptr, gid, 4);
            let plane_bytes = emit_plane_bytes(b, &total);
            for_each_plane(b, &base_addr, &plane_bytes, 16, |b, e, addr| {
                b.raw_ptx(&format!("st.global.f32 [{addr}], {};", v[e]));
            });

            b.raw_ptx(&format!("{exit}:"));
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))
}

// ---------------------------------------------------------------------------
// 2. Filter transform:  U = G g G^T
// ---------------------------------------------------------------------------

/// Generates the filter-transform kernel.
///
/// One thread owns one `(k, c)` pair. Because the NCHW filter is stored
/// `[K, C, 3, 3]` contiguously and the transformed filter plane is `[K, C]`,
/// the flat thread id *is* both the source tap-block index (`gid * 9`) and the
/// destination offset within each plane — no decomposition is needed, and the
/// kernel therefore takes no channel-count parameters at all.
///
/// # Errors
///
/// Returns [`DnnError::PtxGeneration`] if the builder fails.
pub(crate) fn filter_transform_ptx(sm: SmVersion) -> DnnResult<String> {
    KernelBuilder::new(FILTER_TRANSFORM_ENTRY)
        .target(sm)
        .param("filter", PtxType::U64)
        .param("transformed", PtxType::U64)
        .param("total", PtxType::U32)
        .body(|b| {
            b.comment("=== Winograd F(2x2,3x3) filter transform: U = G g G^T ===");
            let (gid, total, exit) = emit_bounds_guard(b, "total");

            let filter_ptr = b.load_param_u64("filter");
            let out_ptr = b.load_param_u64("transformed");

            b.comment("Load the 3x3 tap block at filter[gid * 9 ..]");
            let nine = b.mov_imm_u32(9);
            let tap_base = b.mul_lo_u32(gid.clone(), nine);
            let block_addr = b.byte_offset_addr(filter_ptr, tap_base, 4);
            let mut g = Vec::with_capacity(9);
            for t in 0..9u32 {
                let val = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!(
                    "ld.global.f32 {val}, [{block_addr}+{}];",
                    t.saturating_mul(4)
                ));
                g.push(val);
            }

            b.comment("U = G g G^T");
            let gm = g_f2x3_view();
            let u = emit_congruence(b, &gm, &g);

            b.comment("Scatter U across the 16 transform-domain planes");
            let base_addr = b.byte_offset_addr(out_ptr, gid, 4);
            let plane_bytes = emit_plane_bytes(b, &total);
            for_each_plane(b, &base_addr, &plane_bytes, 16, |b, e, addr| {
                b.raw_ptx(&format!("st.global.f32 [{addr}], {};", u[e]));
            });

            b.raw_ptx(&format!("{exit}:"));
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))
}

// ---------------------------------------------------------------------------
// 3. Transform-domain batched GEMM
// ---------------------------------------------------------------------------

/// Generates the transform-domain batched GEMM kernel.
///
/// Computes, for every transform-domain position `e` in `0..16`:
///
/// ```text
/// M[e] (K x P) = U[e] (K x C) * V[e] (C x P)
/// ```
///
/// as a single launch with `blockIdx.z == e`, using the classic
/// shared-memory-tiled SGEMM: each `GEMM_TILE x GEMM_TILE` block stages one
/// `A` tile and one `B` tile per K-step, so each loaded element is reused
/// `GEMM_TILE` times. All 16 products issue from one kernel launch rather than
/// 16 BLAS dispatches, which matters precisely on the small shapes Winograd is
/// supposed to win (at `C=512, 8x8, K=512` the whole convolution is a few tens
/// of microseconds — 16 separate dispatches would be pure overhead).
///
/// Out-of-range rows/columns stage zeros into shared memory and skip the final
/// store, so `K`, `C` and `P` need not be multiples of `GEMM_TILE`.
///
/// # Errors
///
/// Returns [`DnnError::PtxGeneration`] if the builder fails.
pub(crate) fn batched_gemm_ptx(sm: SmVersion) -> DnnResult<String> {
    let tile = GEMM_TILE;
    let tile_elems = (tile * tile) as usize;
    KernelBuilder::new(GEMM_ENTRY)
        .target(sm)
        .shared_mem("wg_smem_a", PtxType::F32, tile_elems)
        .shared_mem("wg_smem_b", PtxType::F32, tile_elems)
        .param("u_ptr", PtxType::U64)
        .param("v_ptr", PtxType::U64)
        .param("m_ptr", PtxType::U64)
        .param("out_channels", PtxType::U32)
        .param("in_channels", PtxType::U32)
        .param("tile_count", PtxType::U32)
        .body(move |b| {
            b.comment("=== Winograd transform-domain batched GEMM (blockIdx.z = position) ===");

            let tx = b.alloc_reg(PtxType::U32);
            let ty = b.alloc_reg(PtxType::U32);
            let bx = b.alloc_reg(PtxType::U32);
            let by = b.alloc_reg(PtxType::U32);
            let bz = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {tx}, %tid.x;"));
            b.raw_ptx(&format!("mov.u32 {ty}, %tid.y;"));
            b.raw_ptx(&format!("mov.u32 {bx}, %ctaid.x;"));
            b.raw_ptx(&format!("mov.u32 {by}, %ctaid.y;"));
            b.raw_ptx(&format!("mov.u32 {bz}, %ctaid.z;"));

            let u_ptr = b.load_param_u64("u_ptr");
            let v_ptr = b.load_param_u64("v_ptr");
            let m_ptr = b.load_param_u64("m_ptr");
            let k_channels = b.load_param_u32("out_channels");
            let c_channels = b.load_param_u32("in_channels");
            let tile_count = b.load_param_u32("tile_count");

            // row indexes K (output channels), col indexes P (Winograd tiles).
            let row = b.alloc_reg(PtxType::U32);
            let col = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {row}, {by}, {tile}, {ty};"));
            b.raw_ptx(&format!("mad.lo.u32 {col}, {bx}, {tile}, {tx};"));

            b.comment("Per-position plane bases");
            let u_plane = b.mul_lo_u32(k_channels.clone(), c_channels.clone());
            let v_plane = b.mul_lo_u32(c_channels.clone(), tile_count.clone());
            let m_plane = b.mul_lo_u32(k_channels.clone(), tile_count.clone());
            let u_off = b.mul_lo_u32(bz.clone(), u_plane);
            let v_off = b.mul_lo_u32(bz.clone(), v_plane);
            let m_off = b.mul_lo_u32(bz, m_plane);
            let u_base = b.byte_offset_addr(u_ptr, u_off, 4);
            let v_base = b.byte_offset_addr(v_ptr, v_off, 4);
            let m_base = b.byte_offset_addr(m_ptr, m_off, 4);

            b.comment("Shared-memory tile bases (PTX forbids [sym + reg*imm])");
            let sa = b.alloc_reg(PtxType::U32);
            let sb = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {sa}, wg_smem_a;"));
            b.raw_ptx(&format!("mov.u32 {sb}, wg_smem_b;"));
            let lin = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {lin}, {ty}, {tile}, {tx};"));
            let sa_store = b.alloc_reg(PtxType::U32);
            let sb_store = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {sa_store}, {lin}, 4, {sa};"));
            b.raw_ptx(&format!("mad.lo.u32 {sb_store}, {lin}, 4, {sb};"));
            let sa_row = b.alloc_reg(PtxType::U32);
            let sb_col = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "mad.lo.u32 {sa_row}, {ty}, {}, {sa};",
                tile.saturating_mul(4)
            ));
            b.raw_ptx(&format!("mad.lo.u32 {sb_col}, {tx}, 4, {sb};"));

            let p_row = b.alloc_reg(PtxType::Pred);
            let p_col = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lo.u32 {p_row}, {row}, {k_channels};"));
            b.raw_ptx(&format!("setp.lo.u32 {p_col}, {col}, {tile_count};"));

            let fzero = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("mov.f32 {fzero}, {};", f32_lit(0.0)));
            let zero_idx = b.mov_imm_u32(0);
            let acc = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("mov.f32 {acc}, {};", f32_lit(0.0)));

            b.comment("K-loop over the transform-domain contraction (C)");
            let kt = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {kt}, 0;"));
            let loop_top = b.fresh_label("wg_gemm_k");
            let loop_end = b.fresh_label("wg_gemm_k_end");
            b.raw_ptx(&format!("{loop_top}:"));
            let p_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.hs.u32 {p_done}, {kt}, {c_channels};"));
            b.raw_ptx(&format!("@{p_done} bra {loop_end};"));

            // Stage A[row][kt + tx] and B[kt + ty][col].
            let a_col = b.add_u32(kt.clone(), tx.clone());
            let p_a_col = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lo.u32 {p_a_col}, {a_col}, {c_channels};"));
            let p_a = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("and.pred {p_a}, {p_row}, {p_a_col};"));
            let a_idx = b.mad_lo_u32(row.clone(), c_channels.clone(), a_col);
            let a_safe = b.selp(PtxType::U32, a_idx, zero_idx.clone(), p_a.clone());
            let a_addr = b.byte_offset_addr(u_base.clone(), a_safe, 4);
            let a_raw = b.load_global_f32(a_addr);
            let a_val = b.selp(PtxType::F32, a_raw, fzero.clone(), p_a);
            b.raw_ptx(&format!("st.shared.f32 [{sa_store}], {a_val};"));

            let b_row = b.add_u32(kt.clone(), ty.clone());
            let p_b_row = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lo.u32 {p_b_row}, {b_row}, {c_channels};"));
            let p_b = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("and.pred {p_b}, {p_col}, {p_b_row};"));
            let b_idx = b.mad_lo_u32(b_row, tile_count.clone(), col.clone());
            let b_safe = b.selp(PtxType::U32, b_idx, zero_idx.clone(), p_b.clone());
            let b_addr = b.byte_offset_addr(v_base.clone(), b_safe, 4);
            let b_raw = b.load_global_f32(b_addr);
            let b_val = b.selp(PtxType::F32, b_raw, fzero.clone(), p_b);
            b.raw_ptx(&format!("st.shared.f32 [{sb_store}], {b_val};"));

            b.bar_sync(0);
            b.comment("Accumulate the staged tile");
            for i in 0..tile {
                let av = b.alloc_reg(PtxType::F32);
                let bv = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!(
                    "ld.shared.f32 {av}, [{sa_row}+{}];",
                    i.saturating_mul(4)
                ));
                b.raw_ptx(&format!(
                    "ld.shared.f32 {bv}, [{sb_col}+{}];",
                    i.saturating_mul(tile).saturating_mul(4)
                ));
                b.raw_ptx(&format!("fma.rn.f32 {acc}, {av}, {bv}, {acc};"));
            }
            b.bar_sync(0);

            b.raw_ptx(&format!("add.u32 {kt}, {kt}, {tile};"));
            b.raw_ptx(&format!("bra {loop_top};"));
            b.raw_ptx(&format!("{loop_end}:"));

            b.comment("Guarded store of the accumulator");
            let p_store = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("and.pred {p_store}, {p_row}, {p_col};"));
            let out_idx = b.mad_lo_u32(row, tile_count, col);
            let out_addr = b.byte_offset_addr(m_base, out_idx, 4);
            b.raw_ptx(&format!("@{p_store} st.global.f32 [{out_addr}], {acc};"));
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))
}

// ---------------------------------------------------------------------------
// 4. Output transform:  Y = A^T m A
// ---------------------------------------------------------------------------

/// Generates the output-transform kernel.
///
/// One thread owns one `(k, p)` pair, gathers its 16 transform-domain values,
/// applies `A^T m A`, adds the optional per-output-channel bias, and stores the
/// resulting 2x2 patch with a per-element boundary guard (so `out_h`/`out_w`
/// need not be even).
///
/// A null `bias` pointer skips the bias load via a guarded branch, matching the
/// convention used by [`emit_standard_conv_body`](super::super::standard_conv::emit_standard_conv_body).
///
/// # Errors
///
/// Returns [`DnnError::PtxGeneration`] if the builder fails.
pub(crate) fn output_transform_ptx(sm: SmVersion) -> DnnResult<String> {
    KernelBuilder::new(OUTPUT_TRANSFORM_ENTRY)
        .target(sm)
        .param("transformed", PtxType::U64)
        .param("output", PtxType::U64)
        .param("bias", PtxType::U64)
        .param("out_channels", PtxType::U32)
        .param("out_h", PtxType::U32)
        .param("out_w", PtxType::U32)
        .param("tiles_h", PtxType::U32)
        .param("tiles_w", PtxType::U32)
        .param("tile_count", PtxType::U32)
        .param("total", PtxType::U32)
        .body(|b| {
            b.comment("=== Winograd F(2x2,3x3) output transform: Y = A^T m A ===");
            let (gid, total, exit) = emit_bounds_guard(b, "total");

            let in_ptr = b.load_param_u64("transformed");
            let out_ptr = b.load_param_u64("output");
            let bias_ptr = b.load_param_u64("bias");
            let out_channels = b.load_param_u32("out_channels");
            let out_h = b.load_param_u32("out_h");
            let out_w = b.load_param_u32("out_w");
            let tiles_h = b.load_param_u32("tiles_h");
            let tiles_w = b.load_param_u32("tiles_w");
            let tile_count = b.load_param_u32("tile_count");

            b.comment("Decompose gid -> (k, n, tile_h, tile_w)");
            let (k, p) = emit_divmod(b, &gid, &tile_count);
            let (t1, tw) = emit_divmod(b, &p, &tiles_w);
            let (n, th) = emit_divmod(b, &t1, &tiles_h);

            b.comment("Gather the 16 transform-domain values for this (k, tile)");
            let base_addr = b.byte_offset_addr(in_ptr, gid, 4);
            let plane_bytes = emit_plane_bytes(b, &total);
            let mut m = Vec::with_capacity(16);
            for_each_plane(b, &base_addr, &plane_bytes, 16, |b, _e, addr| {
                let val = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.global.f32 {val}, [{addr}];"));
                m.push(val);
            });

            b.comment("Y = A^T m A");
            let at = at_f2x3_view();
            let mut y = emit_congruence(b, &at, &m);

            b.comment("Guarded per-output-channel bias add");
            let no_bias = b.fresh_label("wg_no_bias");
            let p_has_bias = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.eq.u64 {p_has_bias}, {bias_ptr}, 0;"));
            b.raw_ptx(&format!("@{p_has_bias} bra {no_bias};"));
            let bias_addr = b.byte_offset_addr(bias_ptr, k.clone(), 4);
            let bias_val = b.load_global_f32(bias_addr);
            for y_elem in &mut y {
                b.raw_ptx(&format!("add.rn.f32 {y_elem}, {y_elem}, {bias_val};"));
            }
            b.raw_ptx(&format!("{no_bias}:"));

            b.comment("Boundary-clamped 2x2 store");
            let nk = b.mad_lo_u32(n, out_channels, k);
            let ohw = b.mul_lo_u32(out_h.clone(), out_w.clone());
            let out_base = b.mul_lo_u32(nk, ohw);
            let two = b.mov_imm_u32(2);
            let oh0 = b.mul_lo_u32(th, two.clone());
            let ow0 = b.mul_lo_u32(tw, two);
            for i in 0..2u32 {
                let oh = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {oh}, {oh0}, {i};"));
                let p_h = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {p_h}, {oh}, {out_h};"));
                let row_base = b.mad_lo_u32(oh, out_w.clone(), out_base.clone());
                for j in 0..2u32 {
                    let ow = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("add.u32 {ow}, {ow0}, {j};"));
                    let p_w = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.lo.u32 {p_w}, {ow}, {out_w};"));
                    let p_valid = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("and.pred {p_valid}, {p_h}, {p_w};"));
                    let idx = b.add_u32(row_base.clone(), ow);
                    let addr = b.byte_offset_addr(out_ptr.clone(), idx, 4);
                    b.raw_ptx(&format!(
                        "@{p_valid} st.global.f32 [{addr}], {};",
                        y[(i * 2 + j) as usize]
                    ));
                }
            }

            b.raw_ptx(&format!("{exit}:"));
            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every emitted module must declare exactly the entry point the launch
    /// site looks up — the kernel cache resolves `entry` inside the compiled
    /// module by name, so a mismatch is a hard `cuModuleGetFunction` failure.
    /// An entry-point name paired with the emitter that produces its module.
    type NamedEmitter = (&'static str, fn(SmVersion) -> DnnResult<String>);

    #[test]
    fn entries_match_emitted_names() {
        let cases: [NamedEmitter; 4] = [
            (INPUT_TRANSFORM_ENTRY, input_transform_ptx),
            (FILTER_TRANSFORM_ENTRY, filter_transform_ptx),
            (GEMM_ENTRY, batched_gemm_ptx),
            (OUTPUT_TRANSFORM_ENTRY, output_transform_ptx),
        ];
        for (entry, emit) in cases {
            let ptx = emit(SmVersion::Sm86).expect("ptx generation");
            assert!(
                ptx.contains(&format!(".visible .entry {entry}")),
                "{entry} is not the emitted entry point"
            );
            assert!(ptx.contains(".target sm_86"), "{entry} target");
        }
    }

    /// The transforms must actually load, compute and store. These assertions
    /// are the direct antidote to the pre-existing comment-only skeleton: a
    /// body that emits only `comment()` + `ret` fails every one of them.
    #[test]
    fn transforms_emit_real_work() {
        let inp = input_transform_ptx(SmVersion::Sm86).expect("input ptx");
        assert!(inp.contains("ld.global.f32"), "input transform must load");
        assert!(inp.contains("st.global.f32"), "input transform must store");
        assert!(
            inp.contains("add.rn.f32") && inp.contains("sub.rn.f32"),
            "B^T has +1 and -1 entries, so the transform must add and subtract"
        );

        let filt = filter_transform_ptx(SmVersion::Sm86).expect("filter ptx");
        assert!(filt.contains("ld.global.f32"), "filter transform must load");
        assert!(
            filt.contains("st.global.f32"),
            "filter transform must store"
        );
        assert!(
            filt.contains(&f32_lit(0.5)),
            "G's half coefficients must appear as immediates"
        );

        let outp = output_transform_ptx(SmVersion::Sm86).expect("output ptx");
        assert!(outp.contains("ld.global.f32"), "output transform must load");
        assert!(
            outp.contains("@%p") && outp.contains("st.global.f32"),
            "output store must be boundary-predicated"
        );
        assert!(
            outp.contains("setp.eq.u64"),
            "bias add must be guarded on a null pointer"
        );
    }

    /// The GEMM must be a real shared-memory-tiled product, not a stub.
    #[test]
    fn gemm_emits_tiled_product() {
        let ptx = batched_gemm_ptx(SmVersion::Sm86).expect("gemm ptx");
        assert!(ptx.contains(".shared .align 4 .b8 wg_smem_a[1024];"));
        assert!(ptx.contains(".shared .align 4 .b8 wg_smem_b[1024];"));
        assert!(ptx.contains("%ctaid.z"), "blockIdx.z selects the position");
        assert!(ptx.contains("bar.sync 0;"));
        assert_eq!(
            ptx.matches("fma.rn.f32").count(),
            GEMM_TILE as usize,
            "the inner product over one staged tile must be fully unrolled"
        );
    }

    /// The exact count of stores in the transform kernels: 16 scattered plane
    /// stores for the input/filter transforms, 4 boundary-guarded stores for
    /// the output transform. Catches an emitter that silently drops taps.
    #[test]
    fn transform_store_counts() {
        let inp = input_transform_ptx(SmVersion::Sm86).expect("input ptx");
        assert_eq!(inp.matches("st.global.f32").count(), 16);
        assert_eq!(inp.matches("ld.global.f32").count(), 16);

        let filt = filter_transform_ptx(SmVersion::Sm86).expect("filter ptx");
        assert_eq!(filt.matches("st.global.f32").count(), 16);
        assert_eq!(filt.matches("ld.global.f32").count(), 9);

        let outp = output_transform_ptx(SmVersion::Sm86).expect("output ptx");
        assert_eq!(outp.matches("st.global.f32").count(), 4);
        // 16 transform-domain gathers + 1 bias load.
        assert_eq!(outp.matches("ld.global.f32").count(), 17);
    }

    /// `f32_lit` must round-trip through the PTX hex-float encoding.
    #[test]
    fn f32_literal_encoding() {
        assert_eq!(f32_lit(0.0), "0f00000000");
        assert_eq!(f32_lit(1.0), "0f3F800000");
        assert_eq!(f32_lit(0.5), "0f3F000000");
        assert_eq!(f32_lit(-0.5), "0fBF000000");
    }
}
