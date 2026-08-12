//! Extended MSL (Metal Shading Language) kernel source generation.
//!
//! This module complements [`crate::msl`] with the neural-network and
//! extended-numeric kernels that are larger and more specialised:
//!
//! * `softmax_msl` — row-wise numerically-stable softmax.
//! * `layernorm_msl` — layer normalisation with affine `gamma`/`beta`.
//! * `scan_msl` — inclusive/exclusive prefix sum (Hillis-Steele in threadgroup).
//! * `simdgroup_gemm_msl` — GEMM using Apple `simdgroup_matrix<...>` MMA tiles
//!   (analogous to Tensor Cores; needs Metal 3 / Apple GPU family 7+).
//! * `gemm_msl_f64_ds` — double-single emulated FP64 GEMM (Metal has no native
//!   `double`; pairs of `float` carry the high/low limbs).
//! * `int8_quant_gemm_msl` — INT8 × INT8 → INT32 GEMM with per-tensor
//!   dequantisation back to `float` (dynamic-quantization inference path).
//!
//! Each function returns a complete, self-contained MSL translation unit as an
//! owned `String`.  They are unit-tested structurally (correct `kernel`
//! signatures, `[[buffer(n)]]` / `[[threadgroup(n)]]` attributes, the right
//! arithmetic and bounds guards) and, on macOS, compile-tested against a real
//! device when one is present.
//!
//! # v2 generators
//!
//! * [`attention_msl_v2`] — single-pass online-softmax attention with a runtime
//!   parameter buffer, replacing [`crate::msl::attention_msl`]'s two-pass,
//!   constant-baked, device-memory-accumulating kernel.
//! * [`simdgroup_gemm_msl_v2`] — `simdgroup_float8x8` MMA GEMM sharing the
//!   [`crate::msl::gemm_msl_v2`] parameter ABI, with correct ragged-edge
//!   handling for `m`/`n`/`k` that are not multiples of 8.

use crate::error::MetalResult;
use crate::msl::{GemmDtype, MslMathMode, with_math_mode};

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// MSL source for a row-wise, numerically-stable softmax kernel.
///
/// The input is treated as a `rows × cols` matrix in row-major order; each
/// threadgroup processes exactly one row.  The classic three-pass stable
/// softmax is used: (1) reduce the row maximum, (2) reduce `sum(exp(x - max))`,
/// (3) write `exp(x - max) / sum`.  Threadgroup memory holds the per-thread
/// partial maxima and sums.
///
/// Buffer layout:
/// * `[[buffer(0)]]` input  (`const float*`)
/// * `[[buffer(1)]]` output (`float*`)
/// * `[[buffer(2)]]` `rows` (`constant uint&`)
/// * `[[buffer(3)]]` `cols` (`constant uint&`)
/// * `[[threadgroup(0)]]` scratch (`threadgroup float*`, length
///   `threads_per_threadgroup` floats — i.e. `tg_size * 4` bytes)
///
/// # Dispatch contract
///
/// * one threadgroup per row: `threadgroups = rows`
/// * `threads_per_threadgroup` should be a power of two in `1..=1024`; validate
///   it with [`crate::msl::validate_threadgroup_size`]. The tree reductions
///   below no longer *require* it — they walk `n_active -> ceil(n_active / 2)`
///   with an `lid + half < n_active` guard, which is exact for any width,
///   replacing the `s >>= 1` halving loop that silently dropped partials for a
///   ragged width.
///
/// # Math mode
///
/// The row maximum is seeded with `-INFINITY` and the normalisation relies on
/// `exp(x - max) <= 1`; neither is guaranteed under Metal's default fast math.
/// Use [`softmax_msl_with_mode`] with
/// [`MslMathMode::Precise`] when that matters.
pub fn softmax_msl() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax_rows_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      rows   [[buffer(2)]],
    constant uint&      cols   [[buffer(3)]],
    threadgroup float*  scratch [[threadgroup(0)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tg_id >= rows) return;
    uint base = tg_id * cols;

    // Pass 1: row maximum (parallel reduction in threadgroup memory).
    float local_max = -INFINITY;
    for (uint c = lid; c < cols; c += tg_size) {
        local_max = max(local_max, input[base + c]);
    }
    scratch[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Width-agnostic tree: fold the upper ceil(n/2) lanes into the lower half.
    // Correct for any tg_size, unlike the power-of-two-only `s >>= 1` form.
    for (uint n_active = tg_size; n_active > 1u; ) {
        uint half_n = (n_active + 1u) / 2u;
        if (lid + half_n < n_active) {
            scratch[lid] = max(scratch[lid], scratch[lid + half_n]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        n_active = half_n;
    }
    float row_max = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: sum of exp(x - max).
    float local_sum = 0.0f;
    for (uint c = lid; c < cols; c += tg_size) {
        local_sum += exp(input[base + c] - row_max);
    }
    scratch[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n_active = tg_size; n_active > 1u; ) {
        uint half_n = (n_active + 1u) / 2u;
        if (lid + half_n < n_active) {
            scratch[lid] += scratch[lid + half_n];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        n_active = half_n;
    }
    float row_sum = scratch[0];
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // Pass 3: normalise.
    for (uint c = lid; c < cols; c += tg_size) {
        output[base + c] = exp(input[base + c] - row_max) * inv_sum;
    }
}
"#
}

/// [`softmax_msl`] compiled under an explicit [`MslMathMode`].
///
/// Softmax depends on `-INFINITY` as the max identity and on `exp(x - max)`
/// never exceeding one; [`MslMathMode::Precise`] requests the strict IEEE
/// semantics that guarantee both.
pub fn softmax_msl_with_mode(mode: MslMathMode) -> String {
    with_math_mode(softmax_msl(), mode)
}

/// [`softmax_msl`] generated for a specific threadgroup width, with the width
/// **validated at generation time** by
/// [`crate::msl::validate_threadgroup_size`].
///
/// Enforces the documented dispatch contract (a power of two in `1..=1024`)
/// rather than leaving it to the caller to read the doc comment.
pub fn softmax_msl_for_threadgroup(
    threads_per_threadgroup: usize,
    mode: MslMathMode,
) -> MetalResult<String> {
    crate::msl::validate_threadgroup_size(threads_per_threadgroup)?;
    Ok(softmax_msl_with_mode(mode))
}

// ─── Layer normalisation ───────────────────────────────────────────────────────

/// MSL source for a row-wise layer-normalisation kernel with affine transform.
///
/// For each row `r` of the `rows × cols` input:
/// `y = (x - mean) / sqrt(var + eps) * gamma + beta`
/// where `mean`/`var` are computed across the `cols` feature dimension.  Each
/// threadgroup processes one row; partial sums use threadgroup memory.
///
/// Buffer layout:
/// * `[[buffer(0)]]` input  (`const float*`)
/// * `[[buffer(1)]]` gamma  (`const float*`, length `cols`)
/// * `[[buffer(2)]]` beta   (`const float*`, length `cols`)
/// * `[[buffer(3)]]` output (`float*`)
/// * `[[buffer(4)]]` `rows` (`constant uint&`)
/// * `[[buffer(5)]]` `cols` (`constant uint&`)
/// * `[[buffer(6)]]` `eps`  (`constant float&`)
/// * `[[threadgroup(0)]]` scratch (`threadgroup float*`, length
///   `threads_per_threadgroup` floats — i.e. `tg_size * 4` bytes)
///
/// # Dispatch contract
///
/// * one threadgroup per row: `threadgroups = rows`
/// * `threads_per_threadgroup` should be a power of two in `1..=1024` (see
///   [`crate::msl::validate_threadgroup_size`]). As with [`softmax_msl`], the
///   tree reductions were rewritten to be exact for any width rather than only
///   for powers of two.
/// * `cols` must be non-zero — the mean and variance divide by it.
///
/// # Math mode
///
/// The mean/variance sums must not be reassociated for the two-pass variance to
/// stay accurate; use [`layernorm_msl_with_mode`] with
/// [`MslMathMode::Precise`] when that matters.
pub fn layernorm_msl() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void layernorm_rows_f32(
    device const float* input  [[buffer(0)]],
    device const float* gamma  [[buffer(1)]],
    device const float* beta   [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint&      rows   [[buffer(4)]],
    constant uint&      cols   [[buffer(5)]],
    constant float&     eps    [[buffer(6)]],
    threadgroup float*  scratch [[threadgroup(0)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tg_id >= rows) return;
    uint base = tg_id * cols;

    // Pass 1: mean.
    float local_sum = 0.0f;
    for (uint c = lid; c < cols; c += tg_size) {
        local_sum += input[base + c];
    }
    scratch[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n_active = tg_size; n_active > 1u; ) {
        uint half_n = (n_active + 1u) / 2u;
        if (lid + half_n < n_active) {
            scratch[lid] += scratch[lid + half_n];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        n_active = half_n;
    }
    float mean = scratch[0] / float(cols);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: variance.
    float local_var = 0.0f;
    for (uint c = lid; c < cols; c += tg_size) {
        float d = input[base + c] - mean;
        local_var += d * d;
    }
    scratch[lid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n_active = tg_size; n_active > 1u; ) {
        uint half_n = (n_active + 1u) / 2u;
        if (lid + half_n < n_active) {
            scratch[lid] += scratch[lid + half_n];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        n_active = half_n;
    }
    float var = scratch[0] / float(cols);
    float inv_std = rsqrt(var + eps);

    // Pass 3: affine normalise.
    for (uint c = lid; c < cols; c += tg_size) {
        float normed = (input[base + c] - mean) * inv_std;
        output[base + c] = normed * gamma[c] + beta[c];
    }
}
"#
}

/// [`layernorm_msl`] compiled under an explicit [`MslMathMode`].
///
/// The mean and the two-pass variance are sensitive to FP reassociation;
/// [`MslMathMode::Precise`] pins the summation order down.
pub fn layernorm_msl_with_mode(mode: MslMathMode) -> String {
    with_math_mode(layernorm_msl(), mode)
}

/// [`layernorm_msl`] generated for a specific threadgroup width, with the width
/// **validated at generation time** by
/// [`crate::msl::validate_threadgroup_size`].
pub fn layernorm_msl_for_threadgroup(
    threads_per_threadgroup: usize,
    mode: MslMathMode,
) -> MetalResult<String> {
    crate::msl::validate_threadgroup_size(threads_per_threadgroup)?;
    Ok(layernorm_msl_with_mode(mode))
}

// ─── Prefix sum (scan) ─────────────────────────────────────────────────────────

/// MSL source for a single-threadgroup Hillis-Steele inclusive/exclusive scan.
///
/// Computes a prefix sum over `n` elements within one threadgroup using
/// double-buffered threadgroup memory.  When `exclusive` is `true` the kernel
/// shifts the result right by one so element `i` holds the sum of `[0, i)`.
///
/// Buffer layout:
/// * `[[buffer(0)]]` input  (`const float*`)
/// * `[[buffer(1)]]` output (`float*`)
/// * `[[buffer(2)]]` `n`    (`constant uint&`)
/// * `[[threadgroup(0)]]` ping-pong scratch (`threadgroup float*`, length
///   `2 * threads_per_threadgroup` floats — i.e. `2 * tg_size * 4` bytes)
///
/// # Preconditions
///
/// * **Threadgroup memory is `2 * threads_per_threadgroup` floats, not `2 * n`.**
///   The kernel splits the scratch as `buf_a = scratch`,
///   `buf_b = scratch + tg_size` and writes `buf_b[lid]` for every `lid`, so a
///   caller who sized the allocation from `n` (the previous doc's claim) while
///   dispatching the usual `tg_size = next_power_of_two(n)` threads writes past
///   the end of threadgroup memory.
/// * **`n <= threads_per_threadgroup`.** This is a *single-threadgroup* scan
///   with no cross-threadgroup carry: the doubling loop stops at `tg_size`, so a
///   larger `n` yields a silently truncated result. A dispatcher must reject
///   `n > max_total_threads_per_threadgroup`; a multi-block scan needs the
///   standard three-kernel decomposition (per-block scan → scan of block sums →
///   add offsets), which is a separate feature.
/// * The threadgroup must be launched with a single 1-D grid whose global thread
///   id equals `lid` (one threadgroup only), because the kernel indexes `input`
///   with `gid` and the scratch with `lid`.
pub fn scan_msl(exclusive: bool) -> String {
    // For exclusive scan, seed each lane with its left neighbour (identity at 0).
    let seed = if exclusive {
        "(gid > 0u) ? input[gid - 1u] : 0.0f"
    } else {
        "input[gid]"
    };
    let kernel_name = if exclusive {
        "scan_exclusive_f32"
    } else {
        "scan_inclusive_f32"
    };
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void {kernel_name}(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      n      [[buffer(2)]],
    threadgroup float*  scratch [[threadgroup(0)]],
    uint gid     [[thread_position_in_grid]],
    uint lid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    // Two halves of `scratch` form the ping-pong buffers.
    threadgroup float* buf_a = scratch;
    threadgroup float* buf_b = scratch + tg_size;

    float v = (gid < n) ? ({seed}) : 0.0f;
    buf_a[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* src = buf_a;
    threadgroup float* dst = buf_b;
    for (uint offset = 1u; offset < tg_size; offset <<= 1u) {{
        if (lid >= offset) {{
            dst[lid] = src[lid] + src[lid - offset];
        }} else {{
            dst[lid] = src[lid];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* tmp = src;
        src = dst;
        dst = tmp;
    }}

    if (gid < n) {{
        output[gid] = src[lid];
    }}
}}
"#,
        kernel_name = kernel_name,
        seed = seed,
    )
}

/// Return the MSL function name for the requested scan variant.
pub fn scan_function_name(exclusive: bool) -> &'static str {
    if exclusive {
        "scan_exclusive_f32"
    } else {
        "scan_inclusive_f32"
    }
}

// ─── SIMD-group matrix GEMM ────────────────────────────────────────────────────

/// MSL source for a GEMM kernel using Apple `simdgroup_matrix` MMA tiles.
///
/// Each SIMD-group cooperatively multiplies `8×8` `float` tiles via
/// `simdgroup_float8x8` accumulators — the Metal analogue of NVIDIA Tensor
/// Cores.  This requires Metal 3 and Apple GPU family 7+ (M-series / A14+) to
/// execute, but the *source* can be generated and validated on any host.
///
/// Buffer layout matches [`crate::msl::gemm_msl`] (`a`, `b`, `c`, `GemmParams`
/// at `[[buffer(3)]]`), so `lda == k`, `ldb == n`, `ldc == n` and no transpose.
/// Use [`simdgroup_gemm_msl_v2`] for the runtime-strided/transposable variant.
///
/// # Ragged edges (fixed)
///
/// The previous version guarded only the *tile origin*
/// (`if (tile_row >= m || tile_col >= n) return;`) and then issued full 8-wide
/// `simdgroup_load`s straight out of `A`/`B` and stored all 64 cells of the
/// tile unconditionally. Whenever `m`, `n` or `k` was not a multiple of 8 that
/// read past the end of the operands and wrote past the end of `C` — and, for a
/// ragged `n`, across a row boundary into the next row of `C`. This version
/// stages each `8×8` A/B tile into threadgroup memory with **zero fill** past
/// the bounds (zeros contribute nothing to the dot product, so a `k` tail is
/// handled exactly) and guards every store with
/// `row < m && col < n`. No dimension has to be a multiple of 8.
///
/// # Scratch race (fixed)
///
/// The single shared 64-float scratch was raced by every SIMD-group in a
/// threadgroup larger than 32 threads. Each SIMD-group now owns a private
/// `192`-float slice at `simdgroup_index_in_threadgroup * 192`
/// (`a_stage | b_stage | c_stage`, 64 floats each).
///
/// # Dispatch contract
///
/// * threads per threadgroup: `32 * S` for `S = simdgroups_per_threadgroup`
///   (`S == 1`, i.e. 32 threads, is the simple case)
/// * threadgroups: `MTLSize::new(n.div_ceil(8 * S), m.div_ceil(8), 1)`
/// * `set_threadgroup_memory_length(0, 192 * S * 4)` bytes
/// * SIMD-group `s` of threadgroup `(x, y)` owns the output tile at
///   `row = y * 8`, `col = (x * S + s) * 8`
pub fn simdgroup_gemm_msl() -> &'static str {
    r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct GemmParams {
    uint m;
    uint n;
    uint k;
    float alpha;
    float beta;
};

// SIMD-group matrix GEMM. Each simdgroup computes one 8x8 output tile through
// zero-filled threadgroup staging, so ragged m/n/k need no special casing.
// Requires Metal 3 / Apple GPU family 7+ at dispatch time.
kernel void simdgroup_gemm_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float*       c [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    threadgroup float* tile [[threadgroup(0)]],
    uint2 tg_pos  [[threadgroup_position_in_grid]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint  sg_count [[simdgroups_per_threadgroup]],
    uint  lane    [[thread_index_in_simdgroup]]
) {
    const uint TILE = 8u;
    const uint CELLS = 64u;
    const uint LANES = 32u;

    uint tile_row = tg_pos.y * TILE;
    uint tile_col = (tg_pos.x * sg_count + sg_id) * TILE;
    if (tile_row >= params.m || tile_col >= params.n) return;

    // Private scratch for this simdgroup: A stage | B stage | C stage.
    threadgroup float* a_stage = tile + sg_id * (3u * CELLS);
    threadgroup float* b_stage = a_stage + CELLS;
    threadgroup float* c_stage = b_stage + CELLS;

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kk = 0u; kk < params.k; kk += TILE) {
        for (uint e = lane; e < CELLS; e += LANES) {
            uint r = e / TILE;
            uint col = e % TILE;
            uint ar = tile_row + r;
            uint ak = kk + col;
            a_stage[e] = (ar < params.m && ak < params.k)
                       ? a[ar * params.k + ak] : 0.0f;
            uint bk = kk + r;
            uint bc = tile_col + col;
            b_stage[e] = (bk < params.k && bc < params.n)
                       ? b[bk * params.n + bc] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a_frag;
        simdgroup_float8x8 b_frag;
        simdgroup_load(a_frag, a_stage, TILE);
        simdgroup_load(b_frag, b_stage, TILE);
        simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, c_stage, TILE);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = lane; e < CELLS; e += LANES) {
        uint r = e / TILE;
        uint col = e % TILE;
        uint row = tile_row + r;
        uint out_col = tile_col + col;
        if (row >= params.m || out_col >= params.n) continue;
        uint out_idx = row * params.n + out_col;
        // BLAS contract: beta==0 ⇒ C is not referenced (no 0*NaN poisoning).
        float prev = (params.beta == 0.0f) ? 0.0f : params.beta * c[out_idx];
        c[out_idx] = params.alpha * c_stage[e] + prev;
    }
}
"#
}

/// `simdgroup_float8x8` MMA GEMM sharing the [`crate::msl::gemm_msl_v2`]
/// parameter ABI.
///
/// This is the fast path for Apple GPU family 7+ (M1 and later, A14+): the
/// `8×8×8` matrix-multiply-accumulate unit replaces the scalar inner loop, so it
/// is typically several times faster than the threadgroup-tiled scalar kernel on
/// large shapes. Gate its selection on
/// [`MetalDeviceCapabilities::simdgroup_matrix`](crate::device_family::MetalDeviceCapabilities)
/// and fall back to [`crate::msl::gemm_msl_v2`] on older families.
///
/// # Parameter buffer `[[buffer(3)]]`
///
/// Byte-for-byte the `GemmParamsV2` layout documented on
/// [`crate::msl::gemm_msl_v2`] (40 bytes: `m, n, k, lda, ldb, ldc, trans_a,
/// trans_b` as `uint`, then `alpha, beta` as `float`), so a dispatcher can hand
/// the *same* parameter image to either kernel and pick between them purely on
/// device capability.
///
/// # Correctness for ragged shapes
///
/// Both operands are staged into threadgroup memory with zero fill before the
/// `simdgroup_load`, and every store is bounds-guarded, so `m`, `n` and `k` need
/// not be multiples of 8. Transposes and padded leading dimensions are resolved
/// during staging, so the MMA path never has to reason about them.
///
/// # Dispatch contract
///
/// Identical to [`simdgroup_gemm_msl`]:
///
/// * threads per threadgroup: `32 * S`, `S = simdgroups_per_threadgroup`
/// * threadgroups: `MTLSize::new(n.div_ceil(8 * S), m.div_ceil(8), 1)`
/// * `set_threadgroup_memory_length(0, 192 * S * 4)` bytes
///
/// Only [`GemmDtype::F32`] is supported: `simdgroup_half8x8` accumulates in
/// `half`, which is not an acceptable accumulator for a GEMM, and the mixed
/// `simdgroup_multiply_accumulate(float8x8, half8x8, half8x8, float8x8)` form is
/// not available on every family this crate targets. Passing
/// [`GemmDtype::F16`] returns
/// [`MetalError::Unsupported`](crate::error::MetalError::Unsupported).
pub fn simdgroup_gemm_msl_v2(dtype: GemmDtype) -> MetalResult<String> {
    if dtype != GemmDtype::F32 {
        return Err(crate::error::MetalError::Unsupported(format!(
            "simdgroup_matrix GEMM supports only f32 storage, got {dtype:?}"
        )));
    }
    Ok(r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct GemmParamsV2 {
    uint  m;
    uint  n;
    uint  k;
    uint  lda;
    uint  ldb;
    uint  ldc;
    uint  trans_a;
    uint  trans_b;
    float alpha;
    float beta;
};

// One 8x8 output tile per simdgroup, fed from zero-filled threadgroup staging
// so ragged m/n/k and both transposes are handled without special cases.
kernel void simdgroup_gemm_v2_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float*       c [[buffer(2)]],
    constant GemmParamsV2& params [[buffer(3)]],
    threadgroup float* tile [[threadgroup(0)]],
    uint2 tg_pos   [[threadgroup_position_in_grid]],
    uint  sg_id    [[simdgroup_index_in_threadgroup]],
    uint  sg_count [[simdgroups_per_threadgroup]],
    uint  lane     [[thread_index_in_simdgroup]]
) {
    const uint TILE = 8u;
    const uint CELLS = 64u;
    const uint LANES = 32u;

    uint tile_row = tg_pos.y * TILE;
    uint tile_col = (tg_pos.x * sg_count + sg_id) * TILE;
    if (tile_row >= params.m || tile_col >= params.n) return;

    threadgroup float* a_stage = tile + sg_id * (3u * CELLS);
    threadgroup float* b_stage = a_stage + CELLS;
    threadgroup float* c_stage = b_stage + CELLS;

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kk = 0u; kk < params.k; kk += TILE) {
        for (uint e = lane; e < CELLS; e += LANES) {
            uint r = e / TILE;
            uint col = e % TILE;

            uint ar = tile_row + r;
            uint ak = kk + col;
            float av = 0.0f;
            if (ar < params.m && ak < params.k) {
                uint off = (params.trans_a != 0u) ? (ak * params.lda + ar)
                                                  : (ar * params.lda + ak);
                av = a[off];
            }
            a_stage[e] = av;

            uint bk = kk + r;
            uint bc = tile_col + col;
            float bv = 0.0f;
            if (bk < params.k && bc < params.n) {
                uint off = (params.trans_b != 0u) ? (bc * params.ldb + bk)
                                                  : (bk * params.ldb + bc);
                bv = b[off];
            }
            b_stage[e] = bv;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a_frag;
        simdgroup_float8x8 b_frag;
        simdgroup_load(a_frag, a_stage, TILE);
        simdgroup_load(b_frag, b_stage, TILE);
        simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, c_stage, TILE);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = lane; e < CELLS; e += LANES) {
        uint r = e / TILE;
        uint col = e % TILE;
        uint row = tile_row + r;
        uint out_col = tile_col + col;
        if (row >= params.m || out_col >= params.n) continue;
        uint out_idx = row * params.ldc + out_col;
        float prev = (params.beta == 0.0f) ? 0.0f : params.beta * c[out_idx];
        c[out_idx] = params.alpha * c_stage[e] + prev;
    }
}
"#
    .to_string())
}

/// MSL function name generated by [`simdgroup_gemm_msl_v2`].
pub fn simdgroup_gemm_v2_function_name() -> &'static str {
    "simdgroup_gemm_v2_f32"
}

/// Threadgroup-memory bytes [`simdgroup_gemm_msl`] / [`simdgroup_gemm_msl_v2`]
/// need for `simdgroups_per_threadgroup` SIMD-groups.
///
/// Three private `8×8` `float` staging tiles per SIMD-group.
pub fn simdgroup_gemm_threadgroup_bytes(simdgroups_per_threadgroup: usize) -> usize {
    3 * 64 * 4 * simdgroups_per_threadgroup
}

// ─── Double-single emulated FP64 GEMM ──────────────────────────────────────────

/// MSL source for a double-single (`df64`) emulated FP64 GEMM kernel.
///
/// Metal has no native `double`.  This kernel represents each value as an
/// unevaluated sum of two `float` limbs (`hi + lo`), giving ~44 bits of
/// mantissa.  Products use Dekker's `two_prod` (via `fma`) and sums use
/// Knuth's `two_sum`, accumulating in extended precision before the result is
/// rounded back to a single `float` on store.
///
/// Storage is interleaved: buffers `a`/`b`/`c` are `float2` arrays where
/// `.x` is the high limb and `.y` the low limb.
///
/// Buffer layout matches [`crate::msl::gemm_msl`] (`GemmParams` at `[[buffer(3)]]`).
///
/// # Math mode — mandatory
///
/// This kernel is **only correct under strict IEEE semantics**. `two_sum`'s
/// `err = (a - (s - bb)) + (b - bb)` is algebraically zero, and a reassociating
/// compiler folds it away, silently collapsing the `df64` emulation to plain
/// `float`; `two_prod`'s `p = a*b; fma(a, b, -p)` likewise depends on `p` not
/// being contracted. Always compile it via
/// [`gemm_msl_f64_ds_with_mode`]`(`[`MslMathMode::Precise`]`)`, and on a
/// pre-Metal-3.2 toolchain additionally clear
/// `MTLCompileOptions.fastMathEnabled` on the host.
///
/// # Precision of `alpha` / `beta`
///
/// `GemmParams` carries `alpha`/`beta` as `float`, and the kernel lifts them via
/// `ds_from`, so the *scaling* is limited to `f32` precision even though the
/// accumulation itself carries ~44 mantissa bits. Widening them would change the
/// frozen `GemmParams` layout, so it is documented rather than fixed.
pub fn gemm_msl_f64_ds() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

struct GemmParams {
    uint m;
    uint n;
    uint k;
    float alpha;
    float beta;
};

// ── Double-single (df64) primitives ──
struct df64 { float hi; float lo; };

inline df64 ds_from(float a) { return df64{a, 0.0f}; }

// Knuth two-sum: returns rounded sum + exact error.
inline df64 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    float err = (a - (s - bb)) + (b - bb);
    return df64{s, err};
}

// Dekker two-product using fused multiply-add for the error term.
inline df64 two_prod(float a, float b) {
    float p = a * b;
    float err = fma(a, b, -p);
    return df64{p, err};
}

inline df64 ds_add(df64 a, df64 b) {
    df64 s = two_sum(a.hi, b.hi);
    float lo = s.lo + (a.lo + b.lo);
    df64 r = two_sum(s.hi, lo);
    return r;
}

inline df64 ds_mul(df64 a, df64 b) {
    df64 p = two_prod(a.hi, b.hi);
    float lo = p.lo + (a.hi * b.lo + a.lo * b.hi);
    df64 r = two_sum(p.hi, lo);
    return r;
}

kernel void gemm_f64_ds(
    device const float2* a [[buffer(0)]],
    device const float2* b [[buffer(1)]],
    device float2*       c [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.m || col >= params.n) return;

    df64 acc = ds_from(0.0f);
    for (uint i = 0u; i < params.k; i++) {
        float2 av = a[row * params.k + i];
        float2 bv = b[i * params.n + col];
        df64 ad = df64{av.x, av.y};
        df64 bd = df64{bv.x, bv.y};
        acc = ds_add(acc, ds_mul(ad, bd));
    }

    df64 scaled = ds_mul(acc, ds_from(params.alpha));
    uint out_idx = row * params.n + col;
    if (params.beta != 0.0f) {
        float2 cv = c[out_idx];
        df64 cd = ds_mul(df64{cv.x, cv.y}, ds_from(params.beta));
        scaled = ds_add(scaled, cd);
    }
    c[out_idx] = float2(scaled.hi, scaled.lo);
}
"#
}

/// [`gemm_msl_f64_ds`] compiled under an explicit [`MslMathMode`].
///
/// Use [`MslMathMode::Precise`]; see [`gemm_msl_f64_ds`] for why fast math
/// destroys the compensated arithmetic this kernel is built on.
pub fn gemm_msl_f64_ds_with_mode(mode: MslMathMode) -> String {
    with_math_mode(gemm_msl_f64_ds(), mode)
}

// ─── INT8 quantised GEMM ───────────────────────────────────────────────────────

/// MSL source for an INT8 × INT8 → dequantised-`float` GEMM kernel.
///
/// Per-tensor quantisation: the integer accumulation is scaled by
/// `scale_a * scale_b` on store.  An optional per-tensor zero point on each
/// operand is folded out algebraically (set both to 0 for symmetric
/// quantisation).
///
/// Buffers `a`/`b` are `char` (signed 8-bit); `c` is `float`.  The
/// `Int8GemmParams` constant buffer carries shapes, scales, and zero points.
///
/// # Overflow bound
///
/// The kernel used to accumulate `(a - z_a) * (b - z_b)` directly in a 32-bit
/// `int`. [`crate::numeric::Int8Quantizer::Asymmetric`] deliberately leaves the
/// affine zero point unclamped, so `|z|` reaches into the thousands and the
/// products reached ~10⁷ — overflowing (undefined behaviour in MSL) after only a
/// few hundred `K` iterations.
///
/// The zero-point terms are now folded out of the inner loop with the standard
/// identity
///
/// ```text
/// Σ (a - z_a)(b - z_b) = Σ a·b  -  z_b·Σ a  -  z_a·Σ b  +  K·z_a·z_b
/// ```
///
/// so the loop accumulates only raw `int8 × int8` products plus the two operand
/// sums, and the correction is applied once at the end in `float`. This is also
/// faster: two fewer subtractions per MAC.
///
/// # Precision of the correction, stated precisely
///
/// Each of the four terms is converted to `float` before being combined, so each
/// carries a *relative* error of about 2⁻²⁴. When the terms are of comparable
/// magnitude and do not cancel, that is ~10⁻⁷ relative on the result — far below
/// the INT8 quantisation error already present, and irrelevant.
///
/// **Under heavy cancellation it is not.** With a large asymmetric zero point
/// the individual terms can reach ~10¹⁰ while the true sum is near zero; the
/// absolute error then approaches `2⁻²⁴ · max|term|`, i.e. hundreds of integer
/// units before dequantisation. That regime means the quantisation itself is
/// badly conditioned (an asymmetric zero point thousands of units from the data
/// wastes almost the whole INT8 range), but a caller that needs the exact
/// integer sum there should use a symmetric quantiser, or split `K`, rather than
/// rely on this kernel. Widening the correction to 64-bit `long` would remove the
/// caveat and does compile on Apple Silicon, but 64-bit integers are not
/// available across every Metal family this crate targets.
///
/// The remaining integer bounds are therefore independent of the zero points:
///
/// * `Σ a·b` ≤ `127 · 127 · K`  → exact for `K ≤ 133_152`
/// * `Σ a`, `Σ b` ≤ `127 · K`   → exact for `K ≤ 16_909_320`
///
/// **`K` must not exceed 133_152**; a dispatcher should reject larger `K` (or
/// split it) rather than rely on wrap-around.
pub fn int8_quant_gemm_msl() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

struct Int8GemmParams {
    uint m;
    uint n;
    uint k;
    float scale_a;
    float scale_b;
    int  zero_a;
    int  zero_b;
};

kernel void int8_gemm(
    device const char* a [[buffer(0)]],
    device const char* b [[buffer(1)]],
    device float*      c [[buffer(2)]],
    constant Int8GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.m || col >= params.n) return;

    // Overflow-safe accumulation: keep the inner loop on RAW int8 products
    // (bounded by 127*127*K) and fold the zero points out algebraically —
    //   sum((a-za)(b-zb)) = sum(a*b) - zb*sum(a) - za*sum(b) + K*za*zb
    // — so an unclamped affine zero point can no longer overflow `int`.
    int acc = 0;
    int sum_a = 0;
    int sum_b = 0;
    for (uint i = 0u; i < params.k; i++) {
        int av = int(a[row * params.k + i]);
        int bv = int(b[i * params.n + col]);
        acc += av * bv;
        sum_a += av;
        sum_b += bv;
    }
    // The correction is applied in float: ~1e-7 relative per term, negligible
    // against the INT8 quantisation error UNLESS the terms cancel heavily (see
    // the Rust doc comment for that caveat).
    float za = float(params.zero_a);
    float zb = float(params.zero_b);
    float total = float(acc) - zb * float(sum_a) - za * float(sum_b)
                + float(params.k) * za * zb;
    uint out_idx = row * params.n + col;
    c[out_idx] = total * params.scale_a * params.scale_b;
}
"#
}

// ─── Attention v2 (single-pass online softmax) ─────────────────────────────────

/// Size in bytes of the `AttnParamsV2` constant buffer at `[[buffer(4)]]`.
pub const ATTN_PARAMS_V2_BYTES: usize = 24;

/// MSL function name generated by [`attention_msl_v2`].
pub fn attention_v2_function_name() -> &'static str {
    "attention_online_f32"
}

/// Threadgroup-memory bytes [`attention_msl_v2`] needs.
///
/// One `head_dim`-long `float` accumulator per SIMD-group in the threadgroup.
pub fn attention_v2_threadgroup_bytes(head_dim: usize, simdgroups_per_threadgroup: usize) -> usize {
    head_dim * simdgroups_per_threadgroup * 4
}

/// Single-pass, online-softmax scaled-dot-product attention with a **runtime**
/// parameter buffer.
///
/// Replaces [`crate::msl::attention_msl`], which
///
/// 1. bakes `BATCH_HEADS`/`SEQ_Q`/`SEQ_KV`/`HEAD_DIM`/`SCALE`/`CAUSAL` into the
///    source as `constant` globals — so every distinct shape compiles a fresh
///    `MTLLibrary` and adds an entry to the source-hash-keyed pipeline cache,
///    which grows without bound under varying batch/sequence sizes;
/// 2. runs **two** full passes over the keys, recomputing every `Q·K` dot
///    product from scratch in the second one — twice the FLOPs of the whole
///    operation; and
/// 3. accumulates the output *in device memory*
///    (`O[o_off + d] += w * V[v_off + d]` inside the key loop), i.e. a
///    read-modify-write round trip to global memory per key per head dimension.
///
/// This kernel makes a single pass, tracking the running maximum `m_i` and the
/// running denominator `l_i` and rescaling the accumulator by
/// `exp(m_old - m_new)` as each key arrives (the FlashAttention online-softmax
/// recurrence), and keeps the accumulator in threadgroup memory, writing `O`
/// exactly once at the end.
///
/// # Parallel decomposition
///
/// One **SIMD-group** (32 lanes) per query. The lanes split the `head_dim`
/// axis, so the `Q·K` dot product is one `simd_sum` — no threadgroup barrier —
/// and each lane owns a disjoint slice of the accumulator, so the rescale/update
/// needs no synchronisation either.
///
/// `simd_sum` and the `[[simdgroup_index_in_threadgroup]]` /
/// `[[simdgroups_per_threadgroup]]` attributes need MSL 2.0 with SIMD-group
/// reduction support — Apple GPU family 4+ (A11 and later, all Apple Silicon
/// Macs) or Mac family 2. Every query index, and therefore every loop bound and
/// early return in the kernel, is uniform across the SIMD-group, so `simd_sum`
/// is always reached convergently.
///
/// # Parameter buffer `[[buffer(4)]]` — exact byte layout
///
/// Total 24 bytes ([`ATTN_PARAMS_V2_BYTES`]), alignment 4, no padding:
///
/// | offset | size | MSL                | meaning |
/// |-------:|-----:|--------------------|---------|
/// | 0      | 4    | `uint  batch_heads`| `batch * heads` |
/// | 4      | 4    | `uint  seq_q`      | query positions |
/// | 8      | 4    | `uint  seq_kv`     | key/value positions |
/// | 12     | 4    | `uint  head_dim`   | per-head feature width |
/// | 16     | 4    | `uint  causal`     | `0` = dense, non-zero = causal |
/// | 20     | 4    | `float scale`      | usually `1/sqrt(head_dim)` |
///
/// # Buffer bindings
///
/// `[[buffer(0)]]` Q, `[[buffer(1)]]` K, `[[buffer(2)]]` V, `[[buffer(3)]]` O —
/// all `device float*`, laid out `[batch_heads][seq][head_dim]` row-major, as in
/// the v1 kernel and the CPU oracle.
///
/// # Dispatch contract
///
/// * threads per threadgroup: `32 * S` for `S = simdgroups_per_threadgroup`
/// * threadgroups: `(batch_heads * seq_q).div_ceil(S)`
/// * `set_threadgroup_memory_length(0, `[`attention_v2_threadgroup_bytes`]`(head_dim, S))`
/// * SIMD-group `s` of threadgroup `t` handles query
///   `q = t * S + s`, i.e. `sq = q % seq_q`, `bh = q / seq_q`
///
/// # Causal-mask alignment, and the KV-cache caveat
///
/// The mask is **top-left** aligned: query `sq` attends to keys `0..=sq`
/// (`masked ⟺ sk > sq`). This deliberately matches the three implementations
/// that already exist in the workspace — `oxicuda_backend`'s CPU `attention`,
/// `MetalBackend::attention`, and [`crate::msl::attention_msl`] — so the GPU
/// kernel and the CPU oracle agree and the existing tests stay valid.
///
/// **It is the wrong convention for KV-cache decode.** With `seq_q = 1` and
/// `seq_kv = N` (one new token attending to `N` cached keys), top-left alignment
/// masks everything except `sk = 0`, so the token attends only to the very first
/// key. FlashAttention and PyTorch use bottom-right alignment
/// (`masked ⟺ sk > sq + (seq_kv - seq_q)`) for exactly this reason. Switching
/// conventions is **not** a change to this kernel alone: `cpu.rs`,
/// `backend/trait_impls.rs` and [`crate::msl::attention_msl`] all encode
/// top-left, and the `ComputeBackend::attention` trait contract does not define
/// `causal` for a non-square score matrix at all. Until that contract is
/// extended (e.g. with a `causal_offset`), callers doing incremental decode must
/// pass `causal = 0` and mask on the host.
///
/// # Math mode
///
/// The running maximum is seeded with `-FLT_MAX` rather than `-INFINITY`, so the
/// kernel is correct even under a finite-math-only assumption, and the first key
/// is detected by `run_sum == 0` instead of relying on `exp(-inf)`.
/// [`MslMathMode::Precise`] additionally emits the strict-IEEE pragma and routes
/// `exp` through `metal::precise::exp`.
pub fn attention_msl_v2(mode: MslMathMode) -> String {
    let prelude = mode.prelude();
    let exp = format!("{}exp", mode.intrinsic_prefix());
    format!(
        r#"{prelude}
#include <metal_stdlib>
using namespace metal;

struct AttnParamsV2 {{
    uint  batch_heads;
    uint  seq_q;
    uint  seq_kv;
    uint  head_dim;
    uint  causal;
    float scale;
}};

// One simdgroup per query. Single pass over the keys with online-softmax
// rescaling: Q.K is never recomputed, and the output accumulator lives in
// threadgroup memory instead of being read-modify-written in device memory.
kernel void attention_online_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float*       O [[buffer(3)]],
    constant AttnParamsV2& params [[buffer(4)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint tg_id    [[threadgroup_position_in_grid]],
    uint sg_id    [[simdgroup_index_in_threadgroup]],
    uint sg_count [[simdgroups_per_threadgroup]],
    uint lane     [[thread_index_in_simdgroup]]
) {{
    const uint LANES = 32u;

    uint q_idx = tg_id * sg_count + sg_id;
    uint total = params.batch_heads * params.seq_q;
    if (q_idx >= total) return;

    uint sq = q_idx % params.seq_q;
    uint bh = q_idx / params.seq_q;
    uint hd = params.head_dim;
    uint q_off = (bh * params.seq_q + sq) * hd;

    threadgroup float* acc = scratch + sg_id * hd;
    for (uint d = lane; d < hd; d += LANES) {{
        acc[d] = 0.0f;
    }}

    // -FLT_MAX rather than -INFINITY: correct under finite-math-only too.
    float run_max = -FLT_MAX;
    float run_sum = 0.0f;

    // Top-left causal alignment: query sq attends to keys 0..=sq. Clamping the
    // loop bound is uniform across the simdgroup, so simd_sum stays convergent.
    uint kv_end = (params.causal != 0u) ? min(params.seq_kv, sq + 1u) : params.seq_kv;

    for (uint sk = 0u; sk < kv_end; ++sk) {{
        uint kv_off = (bh * params.seq_kv + sk) * hd;
        float partial = 0.0f;
        for (uint d = lane; d < hd; d += LANES) {{
            partial = fma(Q[q_off + d], K[kv_off + d], partial);
        }}
        float score = simd_sum(partial) * params.scale;

        float new_max = max(run_max, score);
        // First key: run_sum is still exactly zero, so nothing needs rescaling
        // and exp(-FLT_MAX - score) is never evaluated.
        float corr = (run_sum == 0.0f) ? 0.0f : {exp}(run_max - new_max);
        float w = {exp}(score - new_max);
        run_sum = run_sum * corr + w;
        run_max = new_max;

        for (uint d = lane; d < hd; d += LANES) {{
            acc[d] = fma(w, V[kv_off + d], acc[d] * corr);
        }}
    }}

    float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    for (uint d = lane; d < hd; d += LANES) {{
        O[q_off + d] = acc[d] * inv;
    }}
}}
"#
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// This file with the `#[cfg(test)]` module stripped off.
    ///
    /// Doc-pinning assertions must search only the non-test half: the literal
    /// they look for also appears in the assertion itself, so searching the
    /// whole file would make them trivially true.
    fn documentation_of_this_module() -> &'static str {
        let src = include_str!("msl_nn.rs");
        let marker = "#[cfg(te";
        match src.find(marker) {
            Some(at) => &src[..at],
            None => src,
        }
    }

    /// The doc-pinning helper is only meaningful if it really drops the test
    /// module — otherwise every `doc.contains(..)` assertion is satisfied by its
    /// own literal.
    #[test]
    fn doc_pinning_helper_excludes_the_test_module() {
        let doc = documentation_of_this_module();
        assert!(!doc.is_empty());
        assert!(
            !doc.contains("fn documentation_of_this_module"),
            "the helper failed to strip the test module"
        );
        assert!(
            doc.contains("pub fn softmax_msl()"),
            "docs were over-trimmed"
        );
    }

    // ── Softmax ──
    #[test]
    fn softmax_has_kernel_and_stable_passes() {
        let src = softmax_msl();
        assert!(src.contains("kernel void softmax_rows_f32"));
        assert!(src.contains("metal_stdlib"));
        // Numerically-stable softmax must subtract the row max before exp.
        assert!(src.contains("input[base + c] - row_max"));
        assert!(src.contains("threadgroup float*  scratch [[threadgroup(0)]]"));
        assert!(src.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
        // Bounds guard on the row index.
        assert!(src.contains("if (tg_id >= rows) return;"));
    }

    #[test]
    fn softmax_buffer_bindings() {
        let src = softmax_msl();
        assert!(src.contains("input  [[buffer(0)]]"));
        assert!(src.contains("output [[buffer(1)]]"));
        assert!(src.contains("rows   [[buffer(2)]]"));
        assert!(src.contains("cols   [[buffer(3)]]"));
    }

    // ── LayerNorm ──
    #[test]
    fn layernorm_has_kernel_and_affine() {
        let src = layernorm_msl();
        assert!(src.contains("kernel void layernorm_rows_f32"));
        // Affine transform: normed * gamma + beta.
        assert!(src.contains("normed * gamma[c] + beta[c]"));
        // Uses rsqrt(var + eps) for the inverse std.
        assert!(src.contains("rsqrt(var + eps)"));
        assert!(src.contains("eps    [[buffer(6)]]"));
    }

    #[test]
    fn layernorm_computes_mean_and_var() {
        let src = layernorm_msl();
        assert!(src.contains("scratch[0] / float(cols)"));
        // variance accumulates squared deviations
        assert!(src.contains("local_var += d * d;"));
    }

    // ── Scan ──
    #[test]
    fn scan_inclusive_seed_and_name() {
        let src = scan_msl(false);
        assert!(src.contains("kernel void scan_inclusive_f32"));
        assert!(src.contains("input[gid]"));
        // ping-pong double buffer
        assert!(src.contains("threadgroup float* buf_a = scratch;"));
        assert!(src.contains("threadgroup float* buf_b = scratch + tg_size;"));
        assert_eq!(scan_function_name(false), "scan_inclusive_f32");
    }

    #[test]
    fn scan_exclusive_shifts_right() {
        let src = scan_msl(true);
        assert!(src.contains("kernel void scan_exclusive_f32"));
        // exclusive scan seeds each lane with its left neighbour
        assert!(src.contains("(gid > 0u) ? input[gid - 1u] : 0.0f"));
        assert_eq!(scan_function_name(true), "scan_exclusive_f32");
    }

    #[test]
    fn scan_uses_hillis_steele_doubling() {
        let src = scan_msl(false);
        assert!(src.contains("for (uint offset = 1u; offset < tg_size; offset <<= 1u)"));
        assert!(src.contains("src[lid] + src[lid - offset]"));
    }

    // ── SIMD-group GEMM ──
    #[test]
    fn simdgroup_gemm_uses_mma_tiles() {
        let src = simdgroup_gemm_msl();
        assert!(src.contains("kernel void simdgroup_gemm_f32"));
        assert!(src.contains("simdgroup_float8x8"));
        assert!(src.contains("simdgroup_load"));
        assert!(src.contains("simdgroup_multiply_accumulate"));
        assert!(src.contains("simdgroup_store"));
        assert!(src.contains("GemmParams"));
    }

    #[test]
    fn simdgroup_gemm_walks_k_in_steps_of_8() {
        let src = simdgroup_gemm_msl();
        assert!(src.contains("kk += TILE"));
        assert!(src.contains("const uint TILE = 8u;"));
    }

    /// Regression for the ragged-edge OOB: the tile must be staged with zero
    /// fill and every store must be bounds-guarded, not just the tile origin.
    #[test]
    fn simdgroup_gemm_guards_ragged_edges_and_private_scratch() {
        for src in [
            simdgroup_gemm_msl().to_string(),
            simdgroup_gemm_msl_v2(GemmDtype::F32).expect("f32 simdgroup gemm"),
        ] {
            // Zero-filled staging instead of loading straight out of A/B.
            assert!(src.contains("a_stage"), "A is not staged: {src}");
            assert!(src.contains("b_stage"), "B is not staged: {src}");
            assert!(!src.contains("simdgroup_load(a_frag, a +"));
            assert!(!src.contains("simdgroup_load(b_frag, b +"));
            // Per-element store guard (the old kernel wrote all 64 cells blind).
            assert!(src.contains("if (row >= params.m || out_col >= params.n) continue;"));
            // Per-simdgroup private scratch instead of one shared 64-float tile.
            assert!(src.contains("tile + sg_id * (3u * CELLS)"));
            assert!(src.contains("[[simdgroup_index_in_threadgroup]]"));
            assert!(src.contains("[[simdgroups_per_threadgroup]]"));
            // beta == 0 must not read C.
            assert!(src.contains("(params.beta == 0.0f) ? 0.0f"));
        }
    }

    #[test]
    fn simdgroup_gemm_v2_uses_the_v2_param_abi_and_rejects_f16() {
        let src = simdgroup_gemm_msl_v2(GemmDtype::F32).expect("f32");
        assert!(src.contains("kernel void simdgroup_gemm_v2_f32"));
        assert_eq!(simdgroup_gemm_v2_function_name(), "simdgroup_gemm_v2_f32");
        // Same struct as crate::msl::gemm_msl_v2 so one param image serves both.
        for field in [
            "uint  m;",
            "uint  n;",
            "uint  k;",
            "uint  lda;",
            "uint  ldb;",
            "uint  ldc;",
            "uint  trans_a;",
            "uint  trans_b;",
            "float alpha;",
            "float beta;",
        ] {
            assert!(src.contains(field), "missing {field}");
        }
        assert!(src.contains("(params.trans_a != 0u) ? (ak * params.lda + ar)"));
        assert!(src.contains("(params.trans_b != 0u) ? (bc * params.ldb + bk)"));
        assert!(src.contains("row * params.ldc + out_col"));
        assert!(simdgroup_gemm_msl_v2(GemmDtype::F16).is_err());
    }

    #[test]
    fn simdgroup_threadgroup_budget_is_three_tiles_per_simdgroup() {
        assert_eq!(simdgroup_gemm_threadgroup_bytes(1), 3 * 64 * 4);
        assert_eq!(simdgroup_gemm_threadgroup_bytes(4), 4 * 3 * 64 * 4);
    }

    // ── Double-single FP64 GEMM ──
    #[test]
    fn f64_ds_has_dekker_primitives() {
        let src = gemm_msl_f64_ds();
        assert!(src.contains("kernel void gemm_f64_ds"));
        assert!(src.contains("struct df64"));
        assert!(src.contains("two_sum"));
        assert!(src.contains("two_prod"));
        // two_prod must use fma for the exact error term.
        assert!(src.contains("fma(a, b, -p)"));
        // Storage uses float2 limbs.
        assert!(src.contains("device const float2* a"));
    }

    #[test]
    fn f64_ds_accumulates_in_extended_precision() {
        let src = gemm_msl_f64_ds();
        assert!(src.contains("acc = ds_add(acc, ds_mul(ad, bd));"));
        assert!(src.contains("ds_mul(acc, ds_from(params.alpha))"));
    }

    // ── INT8 GEMM ──
    #[test]
    fn int8_gemm_dequantises_with_scales() {
        let src = int8_quant_gemm_msl();
        assert!(src.contains("kernel void int8_gemm"));
        assert!(src.contains("device const char* a"));
        // integer accumulation
        assert!(src.contains("int acc = 0;"));
        // dequant on store
        assert!(src.contains("total * params.scale_a * params.scale_b"));
        // zero points folded out of the inner loop
        assert!(src.contains("float za = float(params.zero_a);"));
        assert!(src.contains("float zb = float(params.zero_b);"));
    }

    /// Regression for the `int` overflow: the inner loop must accumulate RAW
    /// int8 products (bounded by 127*127*K) with the zero points folded out
    /// algebraically, not `(a - z_a) * (b - z_b)` whose magnitude grows with an
    /// unclamped affine zero point.
    #[test]
    fn int8_gemm_folds_zero_points_out_of_the_inner_loop() {
        let src = int8_quant_gemm_msl();
        assert!(src.contains("int av = int(a[row * params.k + i]);"));
        assert!(src.contains("int bv = int(b[i * params.n + col]);"));
        assert!(
            !src.contains("int(a[row * params.k + i]) - params.zero_a"),
            "zero point is still subtracted inside the inner loop"
        );
        assert!(src.contains("sum_a += av;"));
        assert!(src.contains("sum_b += bv;"));
        assert!(src.contains("float(params.k) * za * zb"));
        // The documented safe bound must stay in the doc comment.
        let doc = documentation_of_this_module();
        assert!(
            doc.contains("133_152"),
            "the K overflow bound is undocumented"
        );
    }

    /// The zero-point folding must be algebraically identical to the direct
    /// form; check it on the host with the same integer arithmetic the kernel
    /// performs.
    #[test]
    fn int8_zero_point_folding_matches_the_direct_form() {
        let (k, za, zb) = (2048usize, -1301i32, 977i32);
        let a: Vec<i32> = (0..k).map(|i| ((i * 37) % 255) as i32 - 128).collect();
        let b: Vec<i32> = (0..k).map(|i| ((i * 53) % 255) as i32 - 128).collect();

        let direct: i64 = (0..k)
            .map(|i| i64::from(a[i] - za) * i64::from(b[i] - zb))
            .sum();

        let acc: i32 = (0..k).map(|i| a[i] * b[i]).sum();
        let sum_a: i32 = a.iter().sum();
        let sum_b: i32 = b.iter().sum();
        let folded =
            i64::from(acc) - i64::from(zb) * i64::from(sum_a) - i64::from(za) * i64::from(sum_b)
                + k as i64 * i64::from(za) * i64::from(zb);

        assert_eq!(direct, folded);
        // And the raw accumulator stays inside i32, which the direct form does
        // not: (127 + |za|) * (127 + |zb|) * K overflows here.
        assert!(acc.checked_abs().is_some());
        let direct_bound = (127 + i64::from(za).abs()) * (127 + i64::from(zb).abs()) * k as i64;
        assert!(
            direct_bound > i64::from(i32::MAX),
            "this case must actually exercise the overflow the fix prevents"
        );
    }

    #[test]
    fn int8_gemm_has_bounds_guard() {
        let src = int8_quant_gemm_msl();
        assert!(src.contains("if (row >= params.m || col >= params.n) return;"));
    }

    // ── Tree-reduction hardening ──
    #[test]
    fn softmax_and_layernorm_trees_are_width_agnostic() {
        for (label, src) in [("softmax", softmax_msl()), ("layernorm", layernorm_msl())] {
            assert!(
                src.contains("for (uint n_active = tg_size; n_active > 1u; )"),
                "{label} still uses the power-of-two-only halving loop"
            );
            assert!(src.contains("if (lid + half_n < n_active)"), "{label}");
            assert!(
                !src.contains("s >>= 1u"),
                "{label} still contains the lossy halving loop"
            );
        }
    }

    #[test]
    fn scan_doc_states_the_real_scratch_size_and_the_n_limit() {
        // The doc used to promise `2*n` floats while the kernel indexes
        // `2*tg_size`; the source is the authority, so pin both.
        let src = scan_msl(false);
        assert!(src.contains("threadgroup float* buf_b = scratch + tg_size;"));
        let doc = documentation_of_this_module();
        assert!(doc.contains("`2 * threads_per_threadgroup` floats"));
        assert!(doc.contains("`n <= threads_per_threadgroup`"));
    }

    // ── Math-mode wrappers ──
    #[test]
    fn precise_wrappers_emit_the_strict_ieee_pragma() {
        for src in [
            softmax_msl_with_mode(MslMathMode::Precise),
            layernorm_msl_with_mode(MslMathMode::Precise),
            gemm_msl_f64_ds_with_mode(MslMathMode::Precise),
            attention_msl_v2(MslMathMode::Precise),
        ] {
            assert!(src.contains("#pragma METAL fp math_mode(safe)"), "{src}");
        }
        // Fast mode leaves the frozen sources untouched.
        assert_eq!(softmax_msl_with_mode(MslMathMode::Fast), softmax_msl());
        assert_eq!(layernorm_msl_with_mode(MslMathMode::Fast), layernorm_msl());
        assert_eq!(
            gemm_msl_f64_ds_with_mode(MslMathMode::Fast),
            gemm_msl_f64_ds()
        );
        assert!(!attention_msl_v2(MslMathMode::Fast).contains("#pragma"));
        // The threadgroup-validating generators enforce the dispatch contract.
        assert!(softmax_msl_for_threadgroup(256, MslMathMode::Fast).is_ok());
        assert!(softmax_msl_for_threadgroup(100, MslMathMode::Fast).is_err());
        assert!(layernorm_msl_for_threadgroup(1024, MslMathMode::Precise).is_ok());
        assert!(layernorm_msl_for_threadgroup(0, MslMathMode::Fast).is_err());
        assert!(layernorm_msl_for_threadgroup(2048, MslMathMode::Fast).is_err());
        // Precise mode routes exp through the precise namespace as well.
        assert!(attention_msl_v2(MslMathMode::Precise).contains("precise::exp"));
        assert!(!attention_msl_v2(MslMathMode::Fast).contains("precise::exp"));
    }

    // ── Attention v2 ──
    #[test]
    fn attention_v2_is_single_pass_with_runtime_params() {
        let src = attention_msl_v2(MslMathMode::Fast);
        assert!(src.contains("kernel void attention_online_f32"));
        assert_eq!(attention_v2_function_name(), "attention_online_f32");
        // Runtime parameters, not baked constants.
        assert!(!src.contains("constant uint BATCH_HEADS ="));
        for field in [
            "uint  batch_heads;",
            "uint  seq_q;",
            "uint  seq_kv;",
            "uint  head_dim;",
            "uint  causal;",
            "float scale;",
        ] {
            assert!(src.contains(field), "missing param field {field}");
        }
        assert_eq!(ATTN_PARAMS_V2_BYTES, std::mem::size_of::<[u32; 6]>());
        // Exactly one loop over the keys — the v1 kernel had two.
        assert_eq!(
            src.matches("for (uint sk = 0u; sk < kv_end; ++sk)").count(),
            1
        );
        // Online-softmax recurrence, accumulator in threadgroup memory.
        assert!(src.contains("float run_max = -FLT_MAX;"));
        assert!(src.contains("run_sum = run_sum * corr + w;"));
        assert!(src.contains("threadgroup float* acc = scratch + sg_id * hd;"));
        // The output must never be read-modify-written in device memory.
        assert!(!src.contains("O[o_off + d] +="));
        assert!(src.contains("simd_sum(partial)"));
    }

    #[test]
    fn attention_v2_threadgroup_budget() {
        assert_eq!(attention_v2_threadgroup_bytes(64, 1), 64 * 4);
        assert_eq!(attention_v2_threadgroup_bytes(64, 4), 4 * 64 * 4);
    }

    // ── macOS compile checks (skipped without a device) ──
    #[cfg(target_os = "macos")]
    #[test]
    fn nn_kernels_compile_on_macos() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        let sources = [
            softmax_msl().to_string(),
            layernorm_msl().to_string(),
            scan_msl(false),
            scan_msl(true),
            gemm_msl_f64_ds().to_string(),
            int8_quant_gemm_msl().to_string(),
            softmax_msl_with_mode(MslMathMode::Precise),
            layernorm_msl_with_mode(MslMathMode::Precise),
            gemm_msl_f64_ds_with_mode(MslMathMode::Precise),
            attention_msl_v2(MslMathMode::Fast),
            attention_msl_v2(MslMathMode::Precise),
        ];
        for src in &sources {
            if let Err(e) = device.new_library_with_source(src, &opts) {
                panic!("NN MSL failed to compile: {e}\n--- source ---\n{src}");
            }
        }
    }

    // `simdgroup_matrix` needs Metal 3; compile separately and tolerate older
    // toolchains by reporting rather than silently discarding the result.
    #[cfg(target_os = "macos")]
    #[test]
    fn simdgroup_gemm_compiles_on_metal3() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        let v2 = simdgroup_gemm_msl_v2(GemmDtype::F32).expect("f32 simdgroup gemm");
        let v1_ok = device
            .new_library_with_source(simdgroup_gemm_msl(), &opts)
            .is_ok();
        let v2_ok = device.new_library_with_source(&v2, &opts).is_ok();
        // Either the stack supports simdgroup_matrix (both compile) or it does
        // not (neither does) — one compiling without the other means the two
        // kernels have drifted apart.
        assert_eq!(
            v1_ok, v2_ok,
            "simdgroup GEMM v1/v2 disagree on compilability (v1={v1_ok}, v2={v2_ok})"
        );
    }

    /// The regression the simdgroup rewrite exists for: with `m`, `n` and `k`
    /// all non-multiples of 8 the old kernel read past `A`/`B` and wrote past
    /// the end of `C` (and across row boundaries). Guard bytes on both sides of
    /// the live region must survive, and the numbers must match an `f64` oracle.
    #[cfg(target_os = "macos")]
    #[test]
    fn simdgroup_gemm_v2_handles_ragged_shapes_without_oob() {
        use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
        let Some(device) = Device::system_default() else {
            return;
        };
        let queue = device.new_command_queue();
        let src = simdgroup_gemm_msl_v2(GemmDtype::F32).expect("f32 simdgroup gemm");
        let Ok(lib) = device.new_library_with_source(&src, &CompileOptions::new()) else {
            return; // pre-Metal-3 stack
        };
        let Ok(func) = lib.get_function("simdgroup_gemm_v2_f32", None) else {
            return;
        };
        let Ok(pso) = device.new_compute_pipeline_state_with_function(&func) else {
            return;
        };

        // Every dimension deliberately co-prime with the 8x8 MMA tile.
        let (m, n, k) = (13usize, 11usize, 5usize);
        let (lda, ldb, ldc) = (k + 2, n + 3, n + 4);
        const GUARD: usize = 64;
        const SENTINEL: f32 = -1234.5;
        let fill = |len: usize, seed: u32| -> Vec<f32> {
            let mut s = seed | 1;
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((s >> 8) as f32 / 8_388_608.0) - 1.0
                })
                .collect()
        };
        let a = fill(m * lda, 31);
        let b = fill(k * ldb, 37);
        let (alpha, beta) = (1.5f32, -0.25f32);

        for simdgroups in [1u64, 2, 4] {
            let mut c = fill(m * ldc, 41);
            c.extend(std::iter::repeat_n(SENTINEL, GUARD));
            let mk = |data: &[f32]| {
                device.new_buffer_with_data(
                    data.as_ptr() as *const std::ffi::c_void,
                    std::mem::size_of_val(data) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            };
            let (ba, bb, bc) = (mk(&a), mk(&b), mk(&c));
            let params: [u32; 10] = [
                m as u32,
                n as u32,
                k as u32,
                lda as u32,
                ldb as u32,
                ldc as u32,
                0,
                0,
                alpha.to_bits(),
                beta.to_bits(),
            ];
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&ba), 0);
            enc.set_buffer(1, Some(&bb), 0);
            enc.set_buffer(2, Some(&bc), 0);
            enc.set_bytes(3, 40, params.as_ptr() as *const std::ffi::c_void);
            enc.set_threadgroup_memory_length(
                0,
                simdgroup_gemm_threadgroup_bytes(simdgroups as usize) as u64,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (n as u64).div_ceil(8 * simdgroups),
                    (m as u64).div_ceil(8),
                    1,
                ),
                MTLSize::new(32 * simdgroups, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            // SAFETY: shared storage, GPU work complete, sized from `c`.
            let got = unsafe { std::slice::from_raw_parts(bc.contents() as *const f32, c.len()) };

            for r in 0..m {
                for col in 0..n {
                    let mut acc = 0.0f64;
                    for i in 0..k {
                        acc += f64::from(a[r * lda + i]) * f64::from(b[i * ldb + col]);
                    }
                    let want =
                        f64::from(alpha) * acc + f64::from(beta) * f64::from(c[r * ldc + col]);
                    let g = f64::from(got[r * ldc + col]);
                    assert!(
                        (g - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "S={simdgroups} at ({r},{col}): got {g} want {want}"
                    );
                }
                for col in n..ldc {
                    assert_eq!(
                        got[r * ldc + col],
                        c[r * ldc + col],
                        "S={simdgroups}: ldc padding at ({r},{col}) was overwritten"
                    );
                }
            }
            for g in 0..GUARD {
                assert_eq!(
                    got[m * ldc + g],
                    SENTINEL,
                    "S={simdgroups}: wrote {g} elements past the end of C"
                );
            }
        }
    }

    /// The df64 GEMM only delivers extended precision under strict IEEE rules.
    /// Run it on operands whose exact product needs more than 24 mantissa bits
    /// and assert the low limb survives when compiled with
    /// [`MslMathMode::Precise`].
    #[cfg(target_os = "macos")]
    #[test]
    fn df64_gemm_keeps_its_low_limb_under_precise_math() {
        use crate::numeric::{pack_df64, unpack_df64};
        use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

        let Some(device) = Device::system_default() else {
            return;
        };
        let queue = device.new_command_queue();

        // 1 + 2^-30 times itself: the exact product needs ~31 mantissa bits, so
        // a plain f32 GEMM returns exactly 1.0 and the low limb is the whole
        // signal.
        let x = 1.0f64 + 2.0f64.powi(-30);
        let a = pack_df64(&[x]);
        let b = pack_df64(&[x]);
        let c = vec![0.0f32; 2];

        let run = |src: &str| -> Option<Vec<f64>> {
            let lib = device
                .new_library_with_source(src, &CompileOptions::new())
                .ok()?;
            let func = lib.get_function("gemm_f64_ds", None).ok()?;
            let pso = device
                .new_compute_pipeline_state_with_function(&func)
                .ok()?;
            let mk = |d: &[f32]| {
                device.new_buffer_with_data(
                    d.as_ptr() as *const std::ffi::c_void,
                    std::mem::size_of_val(d) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            };
            let (ba, bb, bc) = (mk(&a), mk(&b), mk(&c));
            // GemmParams: m, n, k as uint then alpha, beta as float.
            let params: [u32; 5] = [1, 1, 1, 1.0f32.to_bits(), 0.0f32.to_bits()];
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&ba), 0);
            enc.set_buffer(1, Some(&bb), 0);
            enc.set_buffer(2, Some(&bc), 0);
            enc.set_bytes(3, 20, params.as_ptr() as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            // SAFETY: shared storage, GPU work complete, two floats allocated.
            let raw = unsafe { std::slice::from_raw_parts(bc.contents() as *const f32, 2) };
            assert_ne!(raw[0], 0.0, "the df64 GEMM did not run");
            unpack_df64(raw).ok()
        };

        let Some(precise) = run(&gemm_msl_f64_ds_with_mode(MslMathMode::Precise)) else {
            return;
        };
        let want = x * x;
        let f32_only = f64::from((x as f32) * (x as f32));
        assert!(
            (precise[0] - want).abs() < (f32_only - want).abs(),
            "precise df64 ({}) is no better than plain f32 ({f32_only}) vs {want}",
            precise[0]
        );
        assert!(
            (precise[0] - want).abs() < 1e-15,
            "df64 GEMM lost its extended precision: got {} want {want}",
            precise[0]
        );
    }

    /// Numeric validation of the online-softmax attention kernel against a CPU
    /// oracle that uses the same top-left causal alignment as `cpu.rs`.
    #[cfg(target_os = "macos")]
    #[test]
    fn attention_v2_matches_cpu_oracle() {
        use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
        let Some(device) = Device::system_default() else {
            return;
        };
        let queue = device.new_command_queue();
        let src = attention_msl_v2(MslMathMode::Fast);
        let Ok(lib) = device.new_library_with_source(&src, &CompileOptions::new()) else {
            return;
        };
        let Ok(func) = lib.get_function("attention_online_f32", None) else {
            return;
        };
        let Ok(pso) = device.new_compute_pipeline_state_with_function(&func) else {
            return;
        };

        // seq_q != seq_kv on purpose, and a head_dim that is not a lane multiple.
        let (bh, seq_q, seq_kv, hd) = (3usize, 5usize, 7usize, 20usize);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let fill = |len: usize, seed: u32| -> Vec<f32> {
            let mut s = seed | 1;
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((s >> 8) as f32 / 8_388_608.0) - 1.0
                })
                .collect()
        };
        let q = fill(bh * seq_q * hd, 3);
        let k = fill(bh * seq_kv * hd, 5);
        let v = fill(bh * seq_kv * hd, 7);

        for causal in [false, true] {
            let out = vec![0.0f32; bh * seq_q * hd];
            let mk = |data: &[f32]| {
                device.new_buffer_with_data(
                    data.as_ptr() as *const std::ffi::c_void,
                    std::mem::size_of_val(data) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            };
            let (bq, bk, bv, bo) = (mk(&q), mk(&k), mk(&v), mk(&out));
            let params: [u32; 6] = [
                bh as u32,
                seq_q as u32,
                seq_kv as u32,
                hd as u32,
                u32::from(causal),
                scale.to_bits(),
            ];
            let simdgroups = 4u64;
            let queries = (bh * seq_q) as u64;
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&bq), 0);
            enc.set_buffer(1, Some(&bk), 0);
            enc.set_buffer(2, Some(&bv), 0);
            enc.set_buffer(3, Some(&bo), 0);
            enc.set_bytes(
                4,
                ATTN_PARAMS_V2_BYTES as u64,
                params.as_ptr() as *const std::ffi::c_void,
            );
            enc.set_threadgroup_memory_length(
                0,
                attention_v2_threadgroup_bytes(hd, simdgroups as usize) as u64,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(queries.div_ceil(simdgroups), 1, 1),
                MTLSize::new(32 * simdgroups, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            // SAFETY: shared storage, GPU work complete, sized from `out`.
            let got = unsafe { std::slice::from_raw_parts(bo.contents() as *const f32, out.len()) };

            // CPU oracle: stable softmax, top-left causal mask.
            for b in 0..bh {
                for sq in 0..seq_q {
                    let q_off = (b * seq_q + sq) * hd;
                    let mut scores = Vec::with_capacity(seq_kv);
                    for sk in 0..seq_kv {
                        if causal && sk > sq {
                            continue;
                        }
                        let k_off = (b * seq_kv + sk) * hd;
                        let dot: f64 = (0..hd)
                            .map(|d| f64::from(q[q_off + d]) * f64::from(k[k_off + d]))
                            .sum();
                        scores.push((sk, dot * f64::from(scale)));
                    }
                    let max = scores.iter().fold(f64::NEG_INFINITY, |m, &(_, s)| m.max(s));
                    let sum: f64 = scores.iter().map(|&(_, s)| (s - max).exp()).sum();
                    for d in 0..hd {
                        let want: f64 = scores
                            .iter()
                            .map(|&(sk, s)| {
                                (s - max).exp() * f64::from(v[(b * seq_kv + sk) * hd + d])
                            })
                            .sum::<f64>()
                            / sum;
                        let g = f64::from(got[q_off + d]);
                        assert!(
                            (g - want).abs() <= 1e-4 * want.abs().max(1.0),
                            "causal={causal} b={b} sq={sq} d={d}: got {g} want {want}"
                        );
                    }
                }
            }
        }
    }
}
