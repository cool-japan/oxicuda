//! Additional WGSL shader-source generators that extend [`crate::shader`].
//!
//! Like the base module, every function here returns a complete, self-contained
//! WGSL source string ready for `device.create_shader_module()`.  None of these
//! require a GPU to *generate* or to test structurally; they are verified by
//! asserting on the emitted text (correct `@group`/`@binding`, `@workgroup_size`,
//! the right arithmetic, bounds guards, and — for softmax/layernorm — that the
//! numerically-stable max-subtraction / mean-centering steps are present).
//!
//! Covered kernels (all genuinely missing from `shader.rs`):
//!
//! * [`transpose_wgsl`] — tiled 2-D matrix transpose with a padded shared tile.
//! * [`softmax_wgsl`] — row-wise numerically-stable softmax (max-subtraction).
//! * [`scan_wgsl`] — Blelloch work-efficient inclusive/exclusive prefix scan.
//! * [`layernorm_wgsl`] — row-wise layer normalisation (Ba et al. 2016).
//! * [`subgroup_reduction_wgsl`] — warp-style subgroup reduction (Chrome 125+
//!   / Firefox 135+); the WGSL *source* and its `enable subgroups;` gate are
//!   generated and tested here even though on-device dispatch is HW-gated.
//! * [`f64_emul_add_wgsl`] — double-single (`vec2<f32>`) emulated f64 add for
//!   adapters that lack native FP64 (which is all of WebGPU).

/// Generate WGSL for a tiled 2-D matrix transpose: `out[c, r] = in[r, c]`.
///
/// `in` is a row-major `rows × cols` matrix; `out` is a row-major
/// `cols × rows` matrix.  A `tile × (tile + 1)` shared-memory staging array is
/// used so that the read and write phases are both coalesced and the `+1`
/// padding avoids shared-memory bank conflicts.
///
/// # Arguments
///
/// * `tile_size` — workgroup tile dimension (e.g. 8, 16, 32).
#[must_use]
pub fn transpose_wgsl(tile_size: u32) -> String {
    let padded = tile_size + 1;
    format!(
        r#"
struct TransposeParams {{
    rows: u32,
    cols: u32,
}}

@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             params: TransposeParams;

// Padded by +1 column to avoid shared-memory bank conflicts.
var<workgroup> tile: array<array<f32, {padded}>, {ts}>;

@compute @workgroup_size({ts}, {ts})
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {{
    let lr = lid.y;
    let lc = lid.x;

    // Read phase: coalesced load of a tile of the source.
    let in_r = wgid.y * {ts}u + lr;
    let in_c = wgid.x * {ts}u + lc;
    if (in_r < params.rows && in_c < params.cols) {{
        tile[lr][lc] = src[in_r * params.cols + in_c];
    }} else {{
        tile[lr][lc] = 0.0;
    }}
    workgroupBarrier();

    // Write phase: transposed coordinates, coalesced store to the destination.
    let out_r = wgid.x * {ts}u + lr;
    let out_c = wgid.y * {ts}u + lc;
    if (out_r < params.cols && out_c < params.rows) {{
        dst[out_r * params.rows + out_c] = tile[lc][lr];
    }}
}}
"#,
        ts = tile_size,
        padded = padded,
    )
}

/// Generate WGSL for a row-wise, numerically-stable softmax.
///
/// The input is a row-major `rows × cols` matrix; softmax is applied
/// independently to each of the `rows` rows.  Each row is handled by one
/// workgroup of 256 threads in three cooperative passes:
///
/// 1. row max via a shared-memory tree reduction (numerical stability),
/// 2. `sum(exp(x - max))` via a second tree reduction, and
/// 3. write `exp(x - max) / sum`.
///
/// Subtracting the row max before `exp` is what keeps the result finite for
/// large logits; a naïve `exp(x) / sum(exp(x))` would overflow.
#[must_use]
pub fn softmax_wgsl() -> String {
    r#"
struct SoftmaxParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: SoftmaxParams;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let row = wgid.x;
    if (row >= params.rows) { return; }
    let tid = lid.x;
    let base = row * params.cols;

    // Pass 1: per-thread partial max over a strided slice of the row.  True
    // negative infinity (not a finite `-1e38` sentinel), so a row whose real
    // maximum lies below that magnitude is never masked by the neutral
    // element.
    var local_max: f32 = bitcast<f32>(0xFF800000u);
    var i: u32 = tid;
    loop {
        if (i >= params.cols) { break; }
        local_max = max(local_max, input[base + i]);
        i = i + 256u;
    }
    shared_max[tid] = local_max;
    workgroupBarrier();
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let row_max = shared_max[0];
    workgroupBarrier();

    // Pass 2: per-thread partial sum of exp(x - row_max).
    var local_sum: f32 = 0.0;
    i = tid;
    loop {
        if (i >= params.cols) { break; }
        local_sum = local_sum + exp(input[base + i] - row_max);
        i = i + 256u;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let row_sum = shared_sum[0];
    let inv_sum = 1.0 / row_sum;
    workgroupBarrier();

    // Pass 3: write normalised probabilities.
    i = tid;
    loop {
        if (i >= params.cols) { break; }
        output[base + i] = exp(input[base + i] - row_max) * inv_sum;
        i = i + 256u;
    }
}
"#
    .to_string()
}

/// Whether a prefix scan is inclusive (`out[i]` includes `in[i]`) or exclusive
/// (`out[i]` is the sum of all strictly-earlier elements).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanKind {
    /// Inclusive scan: `out[i] = in[0] + ... + in[i]`.
    Inclusive,
    /// Exclusive scan: `out[i] = in[0] + ... + in[i-1]`, `out[0] = 0`.
    Exclusive,
}

/// Generate WGSL for a single-block Blelloch work-efficient prefix scan.
///
/// Implements the up-sweep (reduce) + down-sweep phases of the Blelloch
/// (1990) scan over a power-of-two-sized shared array of `block_size`
/// elements.  This is the per-block primitive; a multi-block scan composes
/// these with a block-sum carry pass at the host level.
///
/// `block_size` should be a power of two; the emitted `@workgroup_size` is
/// `block_size / 2` because each thread handles two elements (the canonical
/// Blelloch mapping).
///
/// For an exclusive scan the algorithm clears the last element before the
/// down-sweep; the inclusive variant additionally adds the original input back
/// after the exclusive down-sweep.
#[must_use]
pub fn scan_wgsl(block_size: u32, kind: ScanKind) -> String {
    let threads = (block_size / 2).max(1);
    // Inclusive = exclusive scan plus the original element added back.
    //
    // Each of the two elements a thread owns (`2*tid`, `2*tid + 1`) is
    // bounds-checked *independently* against `params.n`.  A single shared
    // guard on just the first element would let the second write land one
    // element past `n` whenever the block's tail is odd-length (i.e. `n`
    // falls strictly between `base + 2*tid` and `base + 2*tid + 1`).
    let inclusive_fixup = match kind {
        ScanKind::Inclusive => {
            "    // Inclusive: add the original input back to the exclusive result.\n    \
             if (base + 2u * tid < params.n) {\n        \
             output[base + 2u * tid] = shared_data[2u * tid] + input[base + 2u * tid];\n    \
             }\n    \
             if (base + 2u * tid + 1u < params.n) {\n        \
             output[base + 2u * tid + 1u] = shared_data[2u * tid + 1u] + input[base + 2u * tid + 1u];\n    \
             }"
        }
        ScanKind::Exclusive => {
            "    if (base + 2u * tid < params.n) {\n        \
             output[base + 2u * tid] = shared_data[2u * tid];\n    \
             }\n    \
             if (base + 2u * tid + 1u < params.n) {\n        \
             output[base + 2u * tid + 1u] = shared_data[2u * tid + 1u];\n    \
             }"
        }
    };
    let kind_comment = match kind {
        ScanKind::Inclusive => "inclusive",
        ScanKind::Exclusive => "exclusive",
    };

    format!(
        r#"
// Blelloch work-efficient {kind_comment} prefix scan (block size {bs}).
struct ScanParams {{
    n: u32,
}}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: ScanParams;

var<workgroup> shared_data: array<f32, {bs}>;

@compute @workgroup_size({threads})
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {{
    let tid  = lid.x;
    let base = wgid.x * {bs}u;

    // Load two elements per thread (zero-pad out-of-range).
    let i0 = 2u * tid;
    let i1 = 2u * tid + 1u;
    if (base + i0 < params.n) {{ shared_data[i0] = input[base + i0]; }} else {{ shared_data[i0] = 0.0; }}
    if (base + i1 < params.n) {{ shared_data[i1] = input[base + i1]; }} else {{ shared_data[i1] = 0.0; }}

    // Up-sweep (reduce) phase.
    var offset: u32 = 1u;
    var d: u32 = {bs}u >> 1u;
    loop {{
        workgroupBarrier();
        if (tid < d) {{
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            shared_data[bi] = shared_data[bi] + shared_data[ai];
        }}
        offset = offset << 1u;
        if (d == 1u) {{ break; }}
        d = d >> 1u;
    }}

    // Clear the last element (root) for the exclusive down-sweep.
    if (tid == 0u) {{ shared_data[{bs}u - 1u] = 0.0; }}

    // Down-sweep phase.
    d = 1u;
    loop {{
        offset = offset >> 1u;
        workgroupBarrier();
        if (tid < d) {{
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            let t = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] = shared_data[bi] + t;
        }}
        if (d == {bs}u >> 1u) {{ break; }}
        d = d << 1u;
    }}
    workgroupBarrier();

    // Write results (exclusive in shared_data; inclusive adds input back).
    // Bounds-checked per-element inside `inclusive_fixup` (see above).
{inclusive_fixup}
}}
"#,
        bs = block_size,
        threads = threads,
        kind_comment = kind_comment,
        inclusive_fixup = inclusive_fixup,
    )
}

/// Generate WGSL for row-wise layer normalisation (Ba, Kiros & Hinton 2016).
///
/// For each row of a row-major `rows × cols` matrix, computes
/// `y = (x - mean) / sqrt(var + eps) * gamma + beta`, where `mean` and `var`
/// are the per-row mean and (biased) variance.  `gamma` and `beta` are
/// per-column affine parameters of length `cols`.  `eps` is embedded as a
/// constant.
///
/// Each row is processed by one workgroup of 256 threads with two cooperative
/// tree reductions (sum, then sum-of-squares for variance).
///
/// # Arguments
///
/// * `eps` — numerical-stability epsilon added to the variance.
#[must_use]
pub fn layernorm_wgsl(eps: f32) -> String {
    format!(
        r#"
struct LayerNormParams {{
    rows: u32,
    cols: u32,
}}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read>       gamma:  array<f32>;
@group(0) @binding(2) var<storage, read>       beta:   array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             params: LayerNormParams;

var<workgroup> shared_acc: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {{
    let row = wgid.x;
    if (row >= params.rows) {{ return; }}
    let tid  = lid.x;
    let base = row * params.cols;
    let inv_n = 1.0 / f32(params.cols);

    // Pass 1: mean.
    var local_sum: f32 = 0.0;
    var i: u32 = tid;
    loop {{
        if (i >= params.cols) {{ break; }}
        local_sum = local_sum + input[base + i];
        i = i + 256u;
    }}
    shared_acc[tid] = local_sum;
    workgroupBarrier();
    var stride: u32 = 128u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (tid < stride) {{
            shared_acc[tid] = shared_acc[tid] + shared_acc[tid + stride];
        }}
        workgroupBarrier();
        stride = stride >> 1u;
    }}
    let mean = shared_acc[0] * inv_n;
    workgroupBarrier();

    // Pass 2: variance (mean of squared deviations).
    var local_var: f32 = 0.0;
    i = tid;
    loop {{
        if (i >= params.cols) {{ break; }}
        let d = input[base + i] - mean;
        local_var = local_var + d * d;
        i = i + 256u;
    }}
    shared_acc[tid] = local_var;
    workgroupBarrier();
    stride = 128u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (tid < stride) {{
            shared_acc[tid] = shared_acc[tid] + shared_acc[tid + stride];
        }}
        workgroupBarrier();
        stride = stride >> 1u;
    }}
    let variance = shared_acc[0] * inv_n;
    let inv_std = 1.0 / sqrt(variance + f32({eps}));
    workgroupBarrier();

    // Pass 3: normalise + affine.
    i = tid;
    loop {{
        if (i >= params.cols) {{ break; }}
        let norm = (input[base + i] - mean) * inv_std;
        output[base + i] = norm * gamma[i] + beta[i];
        i = i + 256u;
    }}
}}
"#,
        eps = eps,
    )
}

/// Generate WGSL for a warp-style subgroup reduction (P0 roadmap item).
///
/// Emits a compute shader that uses the WGSL `subgroups` extension and the
/// `subgroupAdd` / `subgroupMax` / `subgroupMin` built-ins (stabilising in
/// Chrome 125+ and Firefox 135+).  The shader reduces each subgroup's lane
/// values with a single built-in call, then the subgroup leaders combine their
/// partials through shared memory.
///
/// **Device note:** actually *dispatching* this requires an adapter that
/// reports `wgpu::Features::SUBGROUP`; the emitted source and its
/// `enable subgroups;` directive are generated and tested here on CPU, but
/// on-hardware execution is gated on a real GPU that supports subgroups.
///
/// # Arguments
///
/// * `op` — one of `"sum"`, `"max"`, `"min"` (unknown ops fall back to
///   `"sum"`).
/// * `chromium_experimental` — when `true`, emit the pre-standard
///   `enable chromium_experimental_subgroups;` directive instead of the
///   standard `enable subgroups;` (Chromium native path).
#[must_use]
pub fn subgroup_reduction_wgsl(op: &str, chromium_experimental: bool) -> String {
    let (subgroup_fn, neutral) = match op {
        "max" => ("subgroupMax", "bitcast<f32>(0xFF800000u)"),
        "min" => ("subgroupMin", "bitcast<f32>(0x7F800000u)"),
        _ => ("subgroupAdd", "f32(0.0)"),
    };
    // Combine across subgroup leaders in shared memory.
    let combine = match op {
        "max" => "max(acc, val)",
        "min" => "min(acc, val)",
        _ => "acc + val",
    };
    let enable = if chromium_experimental {
        "enable chromium_experimental_subgroups;"
    } else {
        "enable subgroups;"
    };

    format!(
        r#"
{enable}

struct SubgroupReduceParams {{
    n: u32,
}}

@group(0) @binding(0) var<storage, read>       input:        array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_sums: array<f32>;
@group(0) @binding(2) var<uniform>             params:       SubgroupReduceParams;

// Up to 256 lanes / min-subgroup-size of 4 = 64 leader slots, padded to 64.
var<workgroup> leader_vals: array<f32, 64>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id)   gid:  vec3<u32>,
    @builtin(local_invocation_id)    lid:  vec3<u32>,
    @builtin(workgroup_id)           wgid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id:   u32,
    @builtin(subgroup_size)          sg_size: u32,
) {{
    let tid = lid.x;
    var v: f32 = {neutral};
    if (gid.x < params.n) {{ v = input[gid.x]; }}

    // One built-in call reduces the whole subgroup.
    let sg_reduced = {subgroup_fn}(v);

    // Subgroup leaders publish their reduced value.
    let leader_index = tid / sg_size;
    if (sg_id == 0u) {{
        leader_vals[leader_index] = sg_reduced;
    }}
    workgroupBarrier();

    // Thread 0 folds the leader partials and writes the workgroup result.
    if (tid == 0u) {{
        let num_leaders = (256u + sg_size - 1u) / sg_size;
        var acc: f32 = {neutral};
        for (var i: u32 = 0u; i < num_leaders; i = i + 1u) {{
            let val = leader_vals[i];
            acc = {combine};
        }}
        partial_sums[wgid.x] = acc;
    }}
}}
"#,
        enable = enable,
        subgroup_fn = subgroup_fn,
        neutral = neutral,
        combine = combine,
    )
}

/// Generate WGSL for emulated double-precision **addition** using the
/// double-single ("double-float") technique (P2 roadmap item).
///
/// WebGPU has **no native FP64**.  Each logical f64 value is stored as a
/// `vec2<f32>` = `(hi, lo)` where `hi` is the leading f32 and `lo` is the
/// round-off residual, giving ~46 bits of mantissa.  The kernel adds two such
/// arrays element-wise using Knuth's TwoSum / Dekker error-free transformation
/// so the residual is carried correctly.
///
/// Buffers are laid out as interleaved `(hi, lo)` pairs, i.e. element `i`
/// occupies indices `2*i` (hi) and `2*i + 1` (lo).
#[must_use]
pub fn f64_emul_add_wgsl() -> String {
    r#"
// Double-single (emulated f64) element-wise add.  No native FP64 on WebGPU.
// Each value is a (hi, lo) pair: lo carries the round-off residual of hi.
struct DfParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read>       b:      array<f32>;
@group(0) @binding(2) var<storage, read_write> c:      array<f32>;
@group(0) @binding(3) var<uniform>             params: DfParams;

// Knuth TwoSum: returns (s, e) with a + b == s + e exactly (in f32).
fn two_sum(av: f32, bv: f32) -> vec2<f32> {
    let s = av + bv;
    let bb = s - av;
    let err = (av - (s - bb)) + (bv - bb);
    return vec2<f32>(s, err);
}

// Add two double-single numbers (hi, lo) + (hi, lo).
fn df_add(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let sh = two_sum(x.x, y.x);
    let sl = two_sum(x.y, y.y);
    var hi = sh.x;
    var lo = sh.y + sl.x;
    // Renormalise the high/low split.
    let r1 = two_sum(hi, lo);
    hi = r1.x;
    lo = r1.y + sl.y;
    let r2 = two_sum(hi, lo);
    return vec2<f32>(r2.x, r2.y);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let av = vec2<f32>(a[2u * i], a[2u * i + 1u]);
    let bv = vec2<f32>(b[2u * i], b[2u * i + 1u]);
    let r = df_add(av, bv);
    c[2u * i]      = r.x;
    c[2u * i + 1u] = r.y;
}
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── transpose_wgsl ────────────────────────────────────────────────────

    #[test]
    fn wgsl_transpose_contains_workgroup() {
        let src = transpose_wgsl(16);
        assert!(src.contains("@compute @workgroup_size(16, 16)"));
        assert!(src.contains("TransposeParams"));
    }

    #[test]
    fn wgsl_transpose_padded_tile_avoids_bank_conflict() {
        let src = transpose_wgsl(16);
        // Tile is padded to tile+1 columns.
        assert!(src.contains("array<array<f32, 17>, 16>"));
    }

    #[test]
    fn wgsl_transpose_swaps_indices() {
        let src = transpose_wgsl(8);
        // Read uses cols stride; write uses rows stride (the transpose).
        assert!(src.contains("src[in_r * params.cols + in_c]"));
        assert!(src.contains("dst[out_r * params.rows + out_c]"));
        // Write reads the tile with swapped local indices.
        assert!(src.contains("tile[lc][lr]"));
        assert!(src.contains("workgroupBarrier"));
    }

    #[test]
    fn wgsl_transpose_has_bounds_guards() {
        let src = transpose_wgsl(16);
        assert!(src.contains("in_r < params.rows && in_c < params.cols"));
        assert!(src.contains("out_r < params.cols && out_c < params.rows"));
    }

    // ── softmax_wgsl ──────────────────────────────────────────────────────

    #[test]
    fn wgsl_softmax_is_numerically_stable() {
        let src = softmax_wgsl();
        // Must subtract the row max before exp (stability).
        assert!(src.contains("input[base + i] - row_max"));
        assert!(src.contains("exp(input[base + i] - row_max)"));
        // The final write divides by the sum (probabilities), not raw exp.
        assert!(src.contains("inv_sum"));
        assert!(src.contains("* inv_sum"));
    }

    #[test]
    fn wgsl_softmax_does_not_naively_exp_then_divide_without_max() {
        let src = softmax_wgsl();
        // Guard: there must be NO `exp(input[base + i])` without the `- row_max`.
        // i.e. every exp call subtracts the max.
        assert!(!src.contains("exp(input[base + i])"));
    }

    #[test]
    fn wgsl_softmax_bindings_and_workgroup() {
        let src = softmax_wgsl();
        assert!(src.contains("@compute @workgroup_size(256)"));
        assert!(src.contains("var<storage, read>       input:"));
        assert!(src.contains("var<storage, read_write> output:"));
        assert!(src.contains("var<uniform>             params:"));
        // Two distinct reductions (max then sum).
        assert!(src.contains("shared_max"));
        assert!(src.contains("shared_sum"));
    }

    #[test]
    fn wgsl_softmax_row_per_workgroup() {
        let src = softmax_wgsl();
        assert!(src.contains("let row = wgid.x"));
        assert!(src.contains("if (row >= params.rows) { return; }"));
    }

    // ── scan_wgsl ─────────────────────────────────────────────────────────

    #[test]
    fn wgsl_scan_inclusive_adds_input_back() {
        let src = scan_wgsl(256, ScanKind::Inclusive);
        assert!(src.contains("inclusive"));
        // Inclusive = exclusive + original element.
        assert!(src.contains("shared_data[2u * tid] + input[base + 2u * tid]"));
        assert!(src.contains("shared_data[2u * tid + 1u] + input[base + 2u * tid + 1u]"));
    }

    #[test]
    fn wgsl_scan_exclusive_writes_shared_directly() {
        let src = scan_wgsl(256, ScanKind::Exclusive);
        assert!(src.contains("exclusive"));
        assert!(src.contains("output[base + 2u * tid] = shared_data[2u * tid];"));
        // Exclusive must NOT add the input back.
        assert!(!src.contains("shared_data[2u * tid] + input[base + 2u * tid]"));
    }

    #[test]
    fn wgsl_scan_write_stage_guards_each_element_independently() {
        // Regression for the odd-length-tail bug: the write stage previously
        // wrapped BOTH `output[base + 2*tid]` and `output[base + 2*tid + 1]`
        // in a single `if (base + 2*tid < n)` guard, so whenever
        // `base + 2*tid < n <= base + 2*tid + 1` (an odd remainder within the
        // block) the second write landed one element past `n`.  Each element
        // must now carry its own bound.
        for kind in [ScanKind::Inclusive, ScanKind::Exclusive] {
            let src = scan_wgsl(256, kind);
            assert!(
                src.contains("if (base + 2u * tid < params.n) {"),
                "{kind:?} scan lacks an independent guard for the first element"
            );
            assert!(
                src.contains("if (base + 2u * tid + 1u < params.n) {"),
                "{kind:?} scan lacks an independent guard for the second element"
            );
            // Structural count: the (unrelated, always-correct) load stage
            // contributes exactly 2 `if (base + ...)` guards (`i0`, `i1`).
            // The old buggy write stage added exactly 1 more (one shared
            // guard for both writes) for a total of 3; the fix adds 2 (one
            // per write) for a total of 4.  A regression back to the shared
            // guard would drop this count to 3.
            let guard_count = src.matches("if (base + ").count();
            assert_eq!(
                guard_count, 4,
                "{kind:?} scan: expected 2 load guards + 2 independent write \
                 guards (4 total), got {guard_count} — the write stage may have \
                 regressed to a single shared guard"
            );
        }
    }

    #[test]
    fn wgsl_scan_has_up_and_down_sweep() {
        let src = scan_wgsl(512, ScanKind::Inclusive);
        // Half as many threads as block size (two elements per thread).
        assert!(src.contains("@compute @workgroup_size(256)"));
        assert!(src.contains("array<f32, 512>"));
        // Blelloch clears the root before the down-sweep.
        assert!(src.contains("shared_data[512u - 1u] = 0.0"));
        assert!(src.contains("workgroupBarrier"));
    }

    #[test]
    fn wgsl_scan_block_size_64() {
        let src = scan_wgsl(64, ScanKind::Exclusive);
        assert!(src.contains("@compute @workgroup_size(32)"));
        assert!(src.contains("array<f32, 64>"));
    }

    // ── layernorm_wgsl ────────────────────────────────────────────────────

    #[test]
    fn wgsl_layernorm_centers_and_scales() {
        let src = layernorm_wgsl(1e-5);
        // Mean-centering then division by sqrt(var + eps).
        assert!(src.contains("input[base + i] - mean"));
        assert!(src.contains("sqrt(variance + f32(0.00001"));
        // Affine: norm * gamma + beta.
        assert!(src.contains("norm * gamma[i] + beta[i]"));
    }

    #[test]
    fn wgsl_layernorm_variance_is_mean_of_squared_dev() {
        let src = layernorm_wgsl(1e-5);
        assert!(src.contains("let d = input[base + i] - mean;"));
        assert!(src.contains("local_var = local_var + d * d;"));
        assert!(src.contains("let variance = shared_acc[0] * inv_n;"));
    }

    #[test]
    fn wgsl_layernorm_bindings() {
        let src = layernorm_wgsl(1e-6);
        assert!(src.contains("@group(0) @binding(0) var<storage, read>       input:"));
        assert!(src.contains("@group(0) @binding(1) var<storage, read>       gamma:"));
        assert!(src.contains("@group(0) @binding(2) var<storage, read>       beta:"));
        assert!(src.contains("@group(0) @binding(3) var<storage, read_write> output:"));
        assert!(src.contains("@group(0) @binding(4) var<uniform>             params:"));
        assert!(src.contains("@compute @workgroup_size(256)"));
    }

    #[test]
    fn wgsl_layernorm_embeds_eps() {
        // eps appears verbatim in the source.
        assert!(layernorm_wgsl(0.001).contains("0.001"));
    }

    // ── subgroup_reduction_wgsl ───────────────────────────────────────────

    #[test]
    fn wgsl_subgroup_sum_uses_subgroup_add() {
        let src = subgroup_reduction_wgsl("sum", false);
        assert!(src.contains("enable subgroups;"));
        assert!(src.contains("subgroupAdd(v)"));
        assert!(src.contains("acc + val"));
    }

    #[test]
    fn wgsl_subgroup_max_uses_subgroup_max() {
        let src = subgroup_reduction_wgsl("max", false);
        assert!(src.contains("subgroupMax(v)"));
        assert!(src.contains("max(acc, val)"));
        // True negative infinity, not a finite `-1e38` sentinel.
        assert!(src.contains("bitcast<f32>(0xFF800000u)"));
    }

    #[test]
    fn wgsl_subgroup_min_uses_subgroup_min() {
        let src = subgroup_reduction_wgsl("min", false);
        assert!(src.contains("subgroupMin(v)"));
        assert!(src.contains("min(acc, val)"));
    }

    #[test]
    fn wgsl_subgroup_chromium_experimental_directive() {
        let std_src = subgroup_reduction_wgsl("sum", false);
        assert!(std_src.contains("enable subgroups;"));
        assert!(!std_src.contains("chromium_experimental"));

        let exp_src = subgroup_reduction_wgsl("sum", true);
        assert!(exp_src.contains("enable chromium_experimental_subgroups;"));
    }

    #[test]
    fn wgsl_subgroup_uses_subgroup_builtins() {
        let src = subgroup_reduction_wgsl("sum", false);
        assert!(src.contains("@builtin(subgroup_invocation_id)"));
        assert!(src.contains("@builtin(subgroup_size)"));
        assert!(src.contains("@compute @workgroup_size(256)"));
    }

    // ── f64_emul_add_wgsl ─────────────────────────────────────────────────

    #[test]
    fn wgsl_f64_emul_uses_double_single() {
        let src = f64_emul_add_wgsl();
        // vec2<f32> = (hi, lo) representation.
        assert!(src.contains("vec2<f32>"));
        // Knuth TwoSum error-free transform.
        assert!(src.contains("fn two_sum"));
        assert!(src.contains("fn df_add"));
        // Interleaved (hi, lo) addressing.
        assert!(src.contains("a[2u * i]"));
        assert!(src.contains("a[2u * i + 1u]"));
    }

    #[test]
    fn wgsl_f64_emul_two_sum_is_error_free() {
        let src = f64_emul_add_wgsl();
        // The classic TwoSum residual computation.
        assert!(src.contains("let bb = s - av;"));
        assert!(src.contains("(av - (s - bb)) + (bv - bb)"));
    }

    #[test]
    fn wgsl_f64_emul_bindings_and_guard() {
        let src = f64_emul_add_wgsl();
        assert!(src.contains("@compute @workgroup_size(256)"));
        assert!(src.contains("if (i >= params.n) { return; }"));
        assert!(src.contains("var<storage, read_write> c:"));
    }
}
