//! WGSL shader source generation for common compute kernels.
//!
//! Each function returns a complete, self-contained WGSL source string
//! suitable for passing to `device.create_shader_module()`.

/// Generate WGSL source for a tiled GEMM kernel: `C = alpha * op(A) * op(B) + beta * C`.
///
/// Uses `tile_size × tile_size` workgroup tiles with shared-memory staging.
///
/// `op(A)` is the logical `m × k` left operand, `op(B)` the logical `k × n`
/// right operand.  The physical layout of the stored buffers depends on the
/// transpose flags carried in `GemmParams`:
///
/// * `trans_a == 0` — `a` is stored row-major as `m × k`; element `(r, i)` is
///   at `a[r * lda + i]`.
/// * `trans_a != 0` — `a` is stored row-major as `k × m` (the transpose of the
///   logical operand); element `(r, i)` of `op(A)` is at `a[i * lda + r]`.
/// * `trans_b == 0` — `b` is stored row-major as `k × n`; element `(i, c)` is
///   at `b[i * ldb + c]`.
/// * `trans_b != 0` — `b` is stored row-major as `n × k`; element `(i, c)` of
///   `op(B)` is at `b[c * ldb + i]`.
///
/// `lda` / `ldb` / `ldc` are the physical row strides (leading dimensions) of
/// the stored buffers, carried in `GemmParams`.  They default to the packed
/// width but may be larger to address a padded buffer or sub-matrix view; `C`
/// is written at `c[row * ldc + col]`.
///
/// The transpose flags are runtime uniforms, so a single shader module serves
/// all four NN / NT / TN / TT combinations.
///
/// # Arguments
///
/// * `tile_size` — workgroup tile dimension.  Because the workgroup is
///   `tile_size × tile_size`, `tile_size * tile_size` must not exceed WebGPU's
///   baseline `maxComputeInvocationsPerWorkgroup` of 256, so 16 is the portable
///   maximum (e.g. 8 or 16).
pub fn gemm_wgsl(tile_size: u32) -> String {
    format!(
        r#"
struct GemmParams {{
    m:       u32,
    n:       u32,
    k:       u32,
    alpha:   f32,
    beta:    f32,
    trans_a: u32,
    trans_b: u32,
    lda:     u32,
    ldb:     u32,
    ldc:     u32,
    _pad0:   u32,
    _pad1:   u32,
}}

@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read>       b:      array<f32>;
@group(0) @binding(2) var<storage, read_write> c:      array<f32>;
@group(0) @binding(3) var<uniform>             params: GemmParams;

var<workgroup> tile_a: array<array<f32, {ts}>, {ts}>;
var<workgroup> tile_b: array<array<f32, {ts}>, {ts}>;

// op(A)[r, i] — logical m×k left operand.  `lda` is the physical row stride of
// the stored buffer (>= the packed width), supporting padded / sub-matrix views.
fn load_a(r: u32, i: u32) -> f32 {{
    if (r >= params.m || i >= params.k) {{ return 0.0; }}
    if (params.trans_a == 0u) {{
        return a[r * params.lda + i];
    }}
    return a[i * params.lda + r];
}}

// op(B)[i, col] — logical k×n right operand.  `ldb` is the physical row stride.
fn load_b(i: u32, col: u32) -> f32 {{
    if (i >= params.k || col >= params.n) {{ return 0.0; }}
    if (params.trans_b == 0u) {{
        return b[i * params.ldb + col];
    }}
    return b[col * params.ldb + i];
}}

@compute @workgroup_size({ts}, {ts})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {{
    let row = gid.y;
    let col = gid.x;
    let lr  = lid.y;
    let lc  = lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (params.k + {ts}u - 1u) / {ts}u;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {{
        let a_col = t * {ts}u + lc;
        let b_row = t * {ts}u + lr;
        tile_a[lr][lc] = load_a(row, a_col);
        tile_b[lr][lc] = load_b(b_row, col);
        workgroupBarrier();

        for (var e: u32 = 0u; e < {ts}u; e = e + 1u) {{
            acc += tile_a[lr][e] * tile_b[e][lc];
        }}
        workgroupBarrier();
    }}

    if (row >= params.m || col >= params.n) {{ return; }}
    let idx = row * params.ldc + col;
    c[idx] = params.alpha * acc + params.beta * c[idx];
}}
"#,
        ts = tile_size
    )
}

/// Generate WGSL source for a batched (strided) GEMM kernel.
///
/// For each batch `b` in `0..batch_count`:
///   `C_b = alpha * op(A_b) * op(B_b) + beta * C_b`
/// where `A_b` starts at `a[b * stride_a]`, etc.
///
/// Uses `tile_size × tile_size` workgroup tiles with shared-memory staging and
/// Z = batch_count.  Transpose handling matches [`gemm_wgsl`]: the `trans_a` /
/// `trans_b` uniforms select a row-major (`m × k` / `k × n`) or column-major
/// (`k × m` / `n × k`) physical layout for each per-batch operand, with the
/// physical row strides carried as `lda` / `ldb` / `ldc`.
///
/// # Arguments
///
/// * `tile_size` — workgroup tile dimension.  `tile_size * tile_size` must not
///   exceed WebGPU's baseline `maxComputeInvocationsPerWorkgroup` of 256, so 16
///   is the portable maximum (e.g. 8 or 16).
pub fn batched_gemm_wgsl(tile_size: u32) -> String {
    format!(
        r#"
struct BatchedGemmParams {{
    m:        u32,
    n:        u32,
    k:        u32,
    alpha:    f32,
    beta:     f32,
    batch_count: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    trans_a:  u32,
    trans_b:  u32,
    lda:      u32,
    ldb:      u32,
    ldc:      u32,
    _pad0:    u32,
    _pad1:    u32,
}}

@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read>       b:      array<f32>;
@group(0) @binding(2) var<storage, read_write> c:      array<f32>;
@group(0) @binding(3) var<uniform>             params: BatchedGemmParams;

var<workgroup> tile_a: array<array<f32, {ts}>, {ts}>;
var<workgroup> tile_b: array<array<f32, {ts}>, {ts}>;

// op(A_b)[r, i] — logical m×k left operand for batch `a_offset`.  `lda` is the
// physical per-batch row stride of the stored buffer (>= the packed width).
fn load_a(a_offset: u32, r: u32, i: u32) -> f32 {{
    if (r >= params.m || i >= params.k) {{ return 0.0; }}
    if (params.trans_a == 0u) {{
        return a[a_offset + r * params.lda + i];
    }}
    return a[a_offset + i * params.lda + r];
}}

// op(B_b)[i, col] — logical k×n right operand for batch `b_offset`.  `ldb` is
// the physical per-batch row stride.
fn load_b(b_offset: u32, i: u32, col: u32) -> f32 {{
    if (i >= params.k || col >= params.n) {{ return 0.0; }}
    if (params.trans_b == 0u) {{
        return b[b_offset + i * params.ldb + col];
    }}
    return b[b_offset + col * params.ldb + i];
}}

@compute @workgroup_size({ts}, {ts})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {{
    let row = gid.y;
    let col = gid.x;
    let batch_index = gid.z;
    let lr  = lid.y;
    let lc  = lid.x;
    if (batch_index >= params.batch_count) {{ return; }}

    let a_offset = batch_index * params.stride_a;
    let b_offset = batch_index * params.stride_b;
    let c_offset = batch_index * params.stride_c;

    var acc: f32 = 0.0;
    let num_tiles = (params.k + {ts}u - 1u) / {ts}u;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {{
        let a_col = t * {ts}u + lc;
        let b_row = t * {ts}u + lr;
        tile_a[lr][lc] = load_a(a_offset, row, a_col);
        tile_b[lr][lc] = load_b(b_offset, b_row, col);
        workgroupBarrier();

        for (var e: u32 = 0u; e < {ts}u; e = e + 1u) {{
            acc += tile_a[lr][e] * tile_b[e][lc];
        }}
        workgroupBarrier();
    }}

    if (row >= params.m || col >= params.n) {{ return; }}
    let idx = c_offset + row * params.ldc + col;
    c[idx] = params.alpha * acc + params.beta * c[idx];
}}
"#,
        ts = tile_size
    )
}

/// Generate WGSL source for a tiled GEMM kernel using FP16 storage:
/// `C = alpha * op(A) * op(B) + beta * C`.
///
/// Uses `enable f16;` WGSL extension and `tile_size × tile_size` workgroup
/// tiles with shared-memory staging — the same tiling structure as
/// [`gemm_wgsl`], with two differences: the storage buffers *and* the
/// `var<workgroup>` tiles hold `f16` (halving both global- and
/// workgroup-memory traffic relative to staging `f32`), while the inner
/// accumulate loop widens each tile element to `f32` before multiplying, so
/// accumulation is still done in f32 for precision. `lda` / `ldb` / `ldc`
/// (physical row strides) and the `trans_a` / `trans_b` transpose flags are
/// runtime uniforms honoured via the same `load_a` / `load_b` indexing
/// convention as [`gemm_wgsl`] — see that function's doc for the exact
/// row-major / column-major index forms.
///
/// Requires the device to have enabled the `SHADER_F16` feature; otherwise the
/// module fails validation for the missing capability.
///
/// # Arguments
///
/// * `tile_size` — workgroup tile dimension.  Because the workgroup is
///   `tile_size × tile_size`, `tile_size * tile_size` must not exceed WebGPU's
///   baseline `maxComputeInvocationsPerWorkgroup` of 256, so 16 is the
///   portable maximum (e.g. 8 or 16).
pub fn gemm_wgsl_f16(tile_size: u32) -> String {
    format!(
        r#"
enable f16;

struct GemmParams {{
    m:       u32,
    n:       u32,
    k:       u32,
    alpha:   f32,
    beta:    f32,
    trans_a: u32,
    trans_b: u32,
    lda:     u32,
    ldb:     u32,
    ldc:     u32,
    _pad0:   u32,
    _pad1:   u32,
}}

@group(0) @binding(0) var<storage, read>       a:      array<f16>;
@group(0) @binding(1) var<storage, read>       b:      array<f16>;
@group(0) @binding(2) var<storage, read_write> c:      array<f16>;
@group(0) @binding(3) var<uniform>             params: GemmParams;

var<workgroup> tile_a: array<array<f16, {ts}>, {ts}>;
var<workgroup> tile_b: array<array<f16, {ts}>, {ts}>;

// op(A)[r, i] — logical m×k left operand.  `lda` is the physical row stride of
// the stored buffer (>= the packed width), supporting padded / sub-matrix
// views. The `r`/`i` bounds guard is required here — unlike a per-thread
// dot-product loop that only ever indexes `i < params.k` — because the tiled
// staging loop below reads `t * {ts}u + lc` / `t * {ts}u + lr`, which
// routinely exceeds `params.k` / `params.m` on the last (partial) tile; an
// unguarded read would silently pull in a neighbouring row/column via WGSL's
// robust-access semantics instead of contributing a proper zero pad.
fn load_a(r: u32, i: u32) -> f16 {{
    if (r >= params.m || i >= params.k) {{ return 0.0h; }}
    if (params.trans_a == 0u) {{
        return a[r * params.lda + i];
    }}
    return a[i * params.lda + r];
}}

// op(B)[i, col] — logical k×n right operand.  `ldb` is the physical row
// stride.  Same last-tile bounds guard as `load_a`.
fn load_b(i: u32, col: u32) -> f16 {{
    if (i >= params.k || col >= params.n) {{ return 0.0h; }}
    if (params.trans_b == 0u) {{
        return b[i * params.ldb + col];
    }}
    return b[col * params.ldb + i];
}}

@compute @workgroup_size({ts}, {ts})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {{
    let row = gid.y;
    let col = gid.x;
    let lr  = lid.y;
    let lc  = lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (params.k + {ts}u - 1u) / {ts}u;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {{
        let a_col = t * {ts}u + lc;
        let b_row = t * {ts}u + lr;
        tile_a[lr][lc] = load_a(row, a_col);
        tile_b[lr][lc] = load_b(b_row, col);
        workgroupBarrier();

        for (var e: u32 = 0u; e < {ts}u; e = e + 1u) {{
            acc += f32(tile_a[lr][e]) * f32(tile_b[e][lc]);
        }}
        workgroupBarrier();
    }}

    if (row >= params.m || col >= params.n) {{ return; }}
    let idx = row * params.ldc + col;
    let prev = f32(c[idx]);
    c[idx] = f16(params.alpha * acc + params.beta * prev);
}}
"#,
        ts = tile_size
    )
}

/// Generate WGSL source for an element-wise unary operation.
///
/// The shader reads `n` elements from `input`, applies the operation, and
/// writes the results to `output`.  Both buffers have `arrayLength` elements.
///
/// # Arguments
///
/// * `op` — one of: `"relu"`, `"sigmoid"`, `"tanh"`, `"exp"`, `"log"`,
///   `"sqrt"`, `"abs"`, `"neg"`.  Unknown ops are treated as identity.
pub fn elementwise_wgsl(op: &str) -> String {
    let op_expr = match op {
        "relu" => "max(x, 0.0)",
        "sigmoid" => "1.0 / (1.0 + exp(-x))",
        "tanh" => "tanh(x)",
        "exp" => "exp(x)",
        "log" => "log(x)",
        "sqrt" => "sqrt(x)",
        "abs" => "abs(x)",
        "neg" => "-x",
        _ => "x",
    };

    format!(
        r#"
@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&input)) {{ return; }}
    let x = input[i];
    output[i] = {op};
}}
"#,
        op = op_expr
    )
}

/// Generate WGSL source for an element-wise binary operation.
///
/// The shader reads `n` elements from two input buffers (`lhs` and `rhs`),
/// applies the operation, and writes the results to `output`.
///
/// # Arguments
///
/// * `op` — one of: `"add"`, `"sub"`, `"mul"`, `"div"`, `"max"`, `"min"`,
///   `"pow"`.  Unknown ops fall back to identity on `lhs`.
pub fn binary_wgsl(op: &str) -> String {
    let op_expr = match op {
        "add" => "a + b",
        "sub" => "a - b",
        "mul" => "a * b",
        "div" => "a / b",
        "max" => "max(a, b)",
        "min" => "min(a, b)",
        "pow" => "pow(a, b)",
        _ => "a",
    };

    format!(
        r#"
@group(0) @binding(0) var<storage, read>       lhs:    array<f32>;
@group(0) @binding(1) var<storage, read>       rhs:    array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&lhs)) {{ return; }}
    let a = lhs[i];
    let b = rhs[i];
    output[i] = {op};
}}
"#,
        op = op_expr
    )
}

/// Generate WGSL source for a parallel workgroup-level reduction.
///
/// Performs a two-pass approach: each workgroup of 256 threads reduces its
/// tile to a single value in shared memory, then the results are written to
/// a partial-sums buffer.  A second dispatch (with a single workgroup) then
/// reduces the partial-sums to the final scalar.
///
/// # Arguments
///
/// * `op` — one of: `"sum"`, `"max"`, `"min"`, `"mean"`.  `"mean"` behaves
///   like `"sum"` in the shader; the CPU is responsible for dividing by N.
///   Unknown ops fall back to `"sum"`.
pub fn reduction_wgsl(op: &str) -> String {
    // Neutral elements and combine expressions for each operation.  `max` /
    // `min` use the exact IEEE-754 infinities (`bitcast<f32>` from the
    // sign+all-ones-exponent+zero-mantissa bit pattern) rather than an
    // arbitrary finite `±1e38` sentinel, so a true extremum below/above that
    // magnitude is never masked by the neutral element.
    let (neutral, combine) = match op {
        "max" => ("bitcast<f32>(0xFF800000u)", "max(acc, val)"),
        "min" => ("bitcast<f32>(0x7F800000u)", "min(acc, val)"),
        // "sum" and "mean" use the same reduction body.
        _ => ("f32(0.0)", "acc + val"),
    };

    format!(
        r#"
// Reduction params: total element count.
struct ReduceParams {{
    n: u32,
}}

@group(0) @binding(0) var<storage, read>       input:        array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_sums: array<f32>;
@group(0) @binding(2) var<uniform>             params:       ReduceParams;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {{
    let tid         = lid.x;
    let global_idx  = gid.x;

    // Load or use neutral element when out of range.
    if (global_idx < params.n) {{
        shared_data[tid] = input[global_idx];
    }} else {{
        shared_data[tid] = {neutral};
    }}
    workgroupBarrier();

    // Parallel tree reduction within the workgroup.
    var stride: u32 = 128u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (tid < stride) {{
            let acc = shared_data[tid];
            let val = shared_data[tid + stride];
            shared_data[tid] = {combine};
        }}
        workgroupBarrier();
        stride = stride >> 1u;
    }}

    // Thread 0 writes the workgroup result to the partial-sums buffer.
    if (tid == 0u) {{
        partial_sums[wgid.x] = shared_data[0];
    }}
}}
"#,
        neutral = neutral,
        combine = combine,
    )
}

/// Generate a WGSL compute shader for 2D convolution in NCHW format.
///
/// The shader reads from `input` (NCHW) and `filter` (K×C×FH×FW), writing
/// the result to `output` (N×K×OH×OW).  Padding is handled via bounds
/// checking — out-of-range input positions contribute zero.
///
/// # Arguments
///
/// * `n` — batch size
/// * `c_in` — number of input channels
/// * `h_in`, `w_in` — spatial input dimensions
/// * `k_out` — number of output channels (filters)
/// * `fh`, `fw` — filter height / width
/// * `oh`, `ow` — output height / width
/// * `stride_h`, `stride_w` — convolution strides
/// * `pad_h`, `pad_w` — zero-padding applied to the input
#[allow(clippy::too_many_arguments)]
pub fn conv2d_wgsl(
    n: u32,
    c_in: u32,
    h_in: u32,
    w_in: u32,
    k_out: u32,
    fh: u32,
    fw: u32,
    oh: u32,
    ow: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
) -> String {
    format!(
        r#"
// Conv2D NCHW — generated by oxicuda-webgpu
// input   : [{n}, {c_in}, {h_in}, {w_in}]
// kernel_w: [{k_out}, {c_in}, {fh}, {fw}]
// output  : [{n}, {k_out}, {oh}, {ow}]

@group(0) @binding(0) var<storage, read>       input:    array<f32>;
@group(0) @binding(1) var<storage, read>       kernel_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> output:   array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    // gid.x = output x (ox mapped across batches*k_out*oh)
    // We flatten (batch, k, oy) into gid.y and ox into gid.x
    let ox = gid.x;
    let linear_y = gid.y;

    let batch_k_oh = {n}u * {k_out}u * {oh}u;
    if (ox >= {ow}u || linear_y >= batch_k_oh) {{ return; }}

    let b  = linear_y / ({k_out}u * {oh}u);
    let rem = linear_y % ({k_out}u * {oh}u);
    let kf = rem / {oh}u;
    let oy = rem % {oh}u;

    var acc: f32 = 0.0;
    for (var ci: u32 = 0u; ci < {c_in}u; ci = ci + 1u) {{
        for (var fy: u32 = 0u; fy < {fh}u; fy = fy + 1u) {{
            for (var fx: u32 = 0u; fx < {fw}u; fx = fx + 1u) {{
                let iy_raw = i32(oy * {stride_h}u + fy) - i32({pad_h}u);
                let ix_raw = i32(ox * {stride_w}u + fx) - i32({pad_w}u);
                if (iy_raw >= 0 && iy_raw < i32({h_in}u) && ix_raw >= 0 && ix_raw < i32({w_in}u)) {{
                    let iy = u32(iy_raw);
                    let ix = u32(ix_raw);
                    let in_idx = ((b * {c_in}u + ci) * {h_in}u + iy) * {w_in}u + ix;
                    let f_idx  = ((kf * {c_in}u + ci) * {fh}u + fy) * {fw}u + fx;
                    acc += input[in_idx] * kernel_w[f_idx];
                }}
            }}
        }}
    }}

    let o_idx = ((b * {k_out}u + kf) * {oh}u + oy) * {ow}u + ox;
    output[o_idx] = acc;
}}
"#,
        n = n,
        c_in = c_in,
        h_in = h_in,
        w_in = w_in,
        k_out = k_out,
        fh = fh,
        fw = fw,
        oh = oh,
        ow = ow,
        stride_h = stride_h,
        stride_w = stride_w,
        pad_h = pad_h,
        pad_w = pad_w,
    )
}

/// Generate a WGSL compute shader for scaled dot-product attention.
///
/// Implements: `O = softmax(Q·K^T * scale [+ causal_mask]) · V`
///
/// The softmax is numerically stable (subtracts max before exp).
/// When `causal` is true, positions where `sk > sq` are masked to −∞.
///
/// # Arguments
///
/// * `batch_heads` — combined batch × heads dimension
/// * `seq_q` — query sequence length
/// * `seq_kv` — key/value sequence length
/// * `head_dim` — dimension of each head
/// * `scale` — scaling factor (typically `1 / sqrt(head_dim)`)
/// * `causal` — whether to apply a causal (upper-triangular) mask
pub fn attention_wgsl(
    batch_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
    scale: f32,
    causal: bool,
) -> String {
    // True IEEE-754 negative infinity (naga-validated: `bitcast<f32>` from a
    // sign+all-ones-exponent+zero-mantissa u32 pattern), not an arbitrary
    // finite `-1e38` sentinel.  This makes the causal mask exact: a masked
    // score is genuinely unreachable by `max`, and if every key were somehow
    // masked (impossible today — `sk == 0` is never masked — but the shader
    // does not assume that invariant), `masked_score - max_score` becomes
    // `-inf - (-inf) == NaN`, and `NaN > 0.0` is `false`, so pass 3's `else`
    // branch (zero-fill) is a genuine backstop, not dead defensive code.
    let neg_inf = "bitcast<f32>(0xFF800000u)";

    let causal_check = if causal {
        "if (sk > sq) { score = bitcast<f32>(0xFF800000u); } else {"
    } else {
        "{"
    };

    format!(
        r#"
// Scaled dot-product attention — generated by oxicuda-webgpu
// Q, K, V : [{batch_heads}, seq, {head_dim}]
// O       : [{batch_heads}, {seq_q}, {head_dim}]
// scale   : {scale}
// causal  : {causal}

@group(0) @binding(0) var<storage, read>       q_buf: array<f32>;
@group(0) @binding(1) var<storage, read>       k_buf: array<f32>;
@group(0) @binding(2) var<storage, read>       v_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> o_buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let linear = gid.x;
    let total = {batch_heads}u * {seq_q}u;
    if (linear >= total) {{ return; }}

    let bh = linear / {seq_q}u;
    let sq = linear % {seq_q}u;

    let q_base = (bh * {seq_q}u + sq) * {head_dim}u;
    // Q and O share the same [batch_heads, seq_q, head_dim] shape/strides.
    let o_base = q_base;

    // Zero-initialise this thread's output row *before* accumulating below.
    // Pass 2 does a read-modify-write (`o_buf[..] += ..`) so a reused
    // (non-fresh) output buffer must not leak stale contents into the sum.
    for (var d: u32 = 0u; d < {head_dim}u; d = d + 1u) {{
        o_buf[o_base + d] = 0.0;
    }}

    // Pass 1: find max score for numerical stability
    var max_score: f32 = {neg_inf};
    for (var sk: u32 = 0u; sk < {seq_kv}u; sk = sk + 1u) {{
        var score: f32 = 0.0;
        {causal_check}
            let k_base = (bh * {seq_kv}u + sk) * {head_dim}u;
            for (var d: u32 = 0u; d < {head_dim}u; d = d + 1u) {{
                score += q_buf[q_base + d] * k_buf[k_base + d];
            }}
            score *= f32({scale});
        }}
        if (score > max_score) {{ max_score = score; }}
    }}

    // Pass 2: compute exp(score - max), accumulate weighted V
    var sum_exp: f32 = 0.0;
    for (var sk: u32 = 0u; sk < {seq_kv}u; sk = sk + 1u) {{
        var score: f32 = 0.0;
        {causal_check}
            let k_base = (bh * {seq_kv}u + sk) * {head_dim}u;
            for (var d: u32 = 0u; d < {head_dim}u; d = d + 1u) {{
                score += q_buf[q_base + d] * k_buf[k_base + d];
            }}
            score *= f32({scale});
        }}
        let w = exp(score - max_score);
        sum_exp += w;
        let v_base = (bh * {seq_kv}u + sk) * {head_dim}u;
        for (var d: u32 = 0u; d < {head_dim}u; d = d + 1u) {{
            // Accumulate in-place (we normalise after the loop).
            o_buf[o_base + d] += w * v_buf[v_base + d];
        }}
    }}

    // Pass 3: normalise, or write zeros if no key contributed (`sum_exp` is
    // `0.0` or `NaN`; `sum_exp > 0.0` is false for both, so this else branch
    // catches both without a separate NaN check).
    if (sum_exp > 0.0) {{
        for (var d: u32 = 0u; d < {head_dim}u; d = d + 1u) {{
            o_buf[o_base + d] /= sum_exp;
        }}
    }} else {{
        for (var d: u32 = 0u; d < {head_dim}u; d = d + 1u) {{
            o_buf[o_base + d] = 0.0;
        }}
    }}
}}
"#,
        batch_heads = batch_heads,
        seq_q = seq_q,
        seq_kv = seq_kv,
        head_dim = head_dim,
        scale = scale,
        causal = causal,
        causal_check = causal_check,
        neg_inf = neg_inf,
    )
}

/// Generate WGSL source for an N-D reduction along a single axis.
///
/// The tensor is logically reshaped to `[outer, dk, inner]`, where the reduce
/// axis spans `dk` elements, `outer` is the product of dimensions before the
/// axis, and `inner` is the product of dimensions after the axis.
///
/// Output shape is `[outer, inner]` (flattened to a 1-D buffer of length
/// `outer * inner`).  Each output slot is computed by a full workgroup of
/// 256 threads, which cooperatively reduce the `dk` elements via a strided
/// loop and a shared-memory tree reduction.
///
/// Dispatch must be 2-D: `(grid_x, ceil((outer * inner) / grid_x), 1)` to
/// stay below WebGPU's per-axis limit of 65 535 workgroups.  The shader
/// decodes its slot via `wgid.y * params.grid_x + wgid.x` and early-returns
/// if `slot >= outer * inner`.
///
/// For `Mean`, the shader divides each output by `dk` directly (no host-side
/// post-processing required).
///
/// # Arguments
///
/// * `op` — one of: `"sum"`, `"max"`, `"min"`, `"mean"`.  Unknown ops fall
///   back to `"sum"`.
pub fn reduction_nd_wgsl(op: &str) -> String {
    // The first form combines `acc` with `val` (per-thread strided loop).
    // The second form combines `acc2` with `val` (in-shared-memory tree
    // reduction).  Listing both explicitly is more robust than string-
    // substitution for future ops.
    let (neutral, combine, combine_alias) = match op {
        "max" => (
            "bitcast<f32>(0xFF800000u)",
            "max(acc, val)",
            "max(acc2, val)",
        ),
        "min" => (
            "bitcast<f32>(0x7F800000u)",
            "min(acc, val)",
            "min(acc2, val)",
        ),
        // "sum" and "mean" use the same combine; "mean" divides at the end.
        _ => ("f32(0.0)", "acc + val", "acc2 + val"),
    };

    // For "mean", divide the final reduced value by dk; otherwise pass-through.
    let final_expr = if op == "mean" {
        "shared_data[0] / f32(params.dk)"
    } else {
        "shared_data[0]"
    };

    format!(
        r#"
struct ReduceNdParams {{
    outer:        u32,
    dk:           u32,
    inner:        u32,
    outer_stride: u32,
    dk_stride:    u32,
    inner_stride: u32,
    grid_x:       u32,
    _pad:         u32,
}}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: ReduceNdParams;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {{
    let tid = lid.x;
    let total = params.outer * params.inner;

    // Decode 2-D workgroup id back to a linear output slot.
    let slot = wgid.y * params.grid_x + wgid.x;
    if (slot >= total) {{ return; }}

    let o = slot / params.inner;
    let j = slot % params.inner;
    let base = o * params.outer_stride + j * params.inner_stride;

    // Strided per-thread reduction across the dk axis.
    var acc: f32 = {neutral};
    var i: u32 = tid;
    loop {{
        if (i >= params.dk) {{ break; }}
        let val = input[base + i * params.dk_stride];
        acc = {combine};
        i = i + 256u;
    }}

    shared_data[tid] = acc;
    workgroupBarrier();

    // Tree reduction within the workgroup.
    var stride: u32 = 128u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (tid < stride) {{
            let acc2 = shared_data[tid];
            let val  = shared_data[tid + stride];
            shared_data[tid] = {combine_alias};
        }}
        workgroupBarrier();
        stride = stride >> 1u;
    }}

    if (tid == 0u) {{
        output[slot] = {final_expr};
    }}
}}
"#,
        neutral = neutral,
        combine = combine,
        combine_alias = combine_alias,
        final_expr = final_expr,
    )
}

/// Generate WGSL for the final scalar reduction of partial sums.
///
/// Takes a `partial_sums` array of length `num_groups` and reduces it to a
/// single value at `output[0]`.  Dispatched with a single workgroup of 256
/// threads: each thread first folds every partial it owns via a grid-stride
/// loop (`partial_sums[tid]`, `partial_sums[tid + 256]`, …) so that an
/// arbitrary `num_groups` — not just the first 256 — is reduced, then the 256
/// per-thread accumulators are combined with a shared-memory tree reduction.
pub fn reduction_final_wgsl(op: &str) -> String {
    let (neutral, combine) = match op {
        "max" => ("bitcast<f32>(0xFF800000u)", "max(acc, val)"),
        "min" => ("bitcast<f32>(0x7F800000u)", "min(acc, val)"),
        _ => ("f32(0.0)", "acc + val"),
    };

    format!(
        r#"
struct FinalReduceParams {{
    num_groups: u32,
}}

@group(0) @binding(0) var<storage, read>       partial_sums: array<f32>;
@group(0) @binding(1) var<storage, read_write> output:       array<f32>;
@group(0) @binding(2) var<uniform>             params:       FinalReduceParams;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let tid = lid.x;

    // Grid-stride fold: each of the 256 threads accumulates every partial at
    // index tid, tid+256, tid+512, …  Without this, partials beyond index 255
    // (num_groups > 256, i.e. > 65 536 input elements) would be silently
    // dropped.
    var acc: f32 = {neutral};
    var i: u32 = tid;
    loop {{
        if (i >= params.num_groups) {{ break; }}
        let val = partial_sums[i];
        acc = {combine};
        i = i + 256u;
    }}
    shared_data[tid] = acc;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (tid < stride) {{
            let acc = shared_data[tid];
            let val = shared_data[tid + stride];
            shared_data[tid] = {combine};
        }}
        workgroupBarrier();
        stride = stride >> 1u;
    }}

    if (tid == 0u) {{
        output[0] = shared_data[0];
    }}
}}
"#,
        neutral = neutral,
        combine = combine,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_gemm_contains_workgroup() {
        let src = gemm_wgsl(16);
        assert!(src.contains("@compute @workgroup_size(16, 16)"));
        assert!(src.contains("GemmParams"));
        assert!(src.contains("alpha"));
        assert!(src.contains("beta"));
    }

    #[test]
    fn wgsl_gemm_tile_size_embedded() {
        let src8 = gemm_wgsl(8);
        assert!(src8.contains("@workgroup_size(8, 8)"));
        let src32 = gemm_wgsl(32);
        assert!(src32.contains("@workgroup_size(32, 32)"));
    }

    #[test]
    fn wgsl_gemm_has_transpose_flags() {
        let src = gemm_wgsl(8);
        // Transpose flags live in the uniform struct.
        assert!(src.contains("trans_a: u32"));
        assert!(src.contains("trans_b: u32"));
        // Leading-dimension uniforms drive the physical row strides.
        assert!(src.contains("lda:     u32"));
        assert!(src.contains("ldb:     u32"));
        assert!(src.contains("ldc:     u32"));
        // Both row-major and column-major index forms must be present, keyed on
        // the leading dimensions rather than the packed extents.
        assert!(src.contains("a[r * params.lda + i]"));
        assert!(src.contains("a[i * params.lda + r]"));
        assert!(src.contains("b[i * params.ldb + col]"));
        assert!(src.contains("b[col * params.ldb + i]"));
        assert!(src.contains("row * params.ldc + col"));
    }

    #[test]
    fn wgsl_gemm_uses_shared_memory_tiling() {
        let src = gemm_wgsl(16);
        assert!(src.contains("var<workgroup> tile_a"));
        assert!(src.contains("var<workgroup> tile_b"));
        assert!(src.contains("workgroupBarrier"));
        // Tile dimension is embedded in the workgroup-array declaration.
        assert!(src.contains("array<array<f32, 16>, 16>"));
    }

    #[test]
    fn wgsl_elementwise_relu_contains_max() {
        let src = elementwise_wgsl("relu");
        assert!(src.contains("max(x, 0.0)"));
    }

    #[test]
    fn wgsl_elementwise_all_ops() {
        assert!(elementwise_wgsl("sigmoid").contains("exp(-x)"));
        assert!(elementwise_wgsl("tanh").contains("tanh(x)"));
        assert!(elementwise_wgsl("exp").contains("exp(x)"));
        assert!(elementwise_wgsl("log").contains("log(x)"));
        assert!(elementwise_wgsl("sqrt").contains("sqrt(x)"));
        assert!(elementwise_wgsl("abs").contains("abs(x)"));
        assert!(elementwise_wgsl("neg").contains("-x"));
        // Unknown op is identity.
        assert!(elementwise_wgsl("identity_op").contains("output[i] = x;"));
    }

    #[test]
    fn wgsl_reduction_sum_contains_addition() {
        let src = reduction_wgsl("sum");
        assert!(src.contains("acc + val"));
        assert!(src.contains("workgroupBarrier"));
    }

    #[test]
    fn wgsl_reduction_max_uses_max_fn() {
        let src = reduction_wgsl("max");
        assert!(src.contains("max(acc, val)"));
    }

    #[test]
    fn wgsl_reduction_min_uses_min_fn() {
        let src = reduction_wgsl("min");
        assert!(src.contains("min(acc, val)"));
    }

    #[test]
    fn wgsl_reduction_mean_same_as_sum() {
        // "mean" divides on the CPU side; the shader is identical to sum.
        let sum_src = reduction_wgsl("sum");
        let mean_src = reduction_wgsl("mean");
        assert_eq!(sum_src, mean_src);
    }

    #[test]
    fn wgsl_reduction_final_sum() {
        let src = reduction_final_wgsl("sum");
        assert!(src.contains("num_groups"));
        assert!(src.contains("output[0]"));
    }

    #[test]
    fn wgsl_reduction_final_grid_strides_over_all_groups() {
        // The final pass must fold *every* partial via a grid-stride loop, not
        // just the first 256 (which would drop reductions over > 65 536
        // elements).  Verify the strided loop is present for all ops.
        for op in ["sum", "max", "min", "mean"] {
            let src = reduction_final_wgsl(op);
            assert!(
                src.contains("i = i + 256u"),
                "final reduction for {op} lacks the grid-stride loop"
            );
            assert!(
                src.contains("if (i >= params.num_groups)"),
                "final reduction for {op} lacks the num_groups loop bound"
            );
        }
    }

    // ── reduction_nd_wgsl tests ───────────────────────────────────────────

    #[test]
    fn wgsl_reduction_nd_sum_contains_addition() {
        let src = reduction_nd_wgsl("sum");
        assert!(src.contains("acc + val"));
        // Tree-step reuses the same combine with renamed lhs.
        assert!(src.contains("acc2 + val"));
        assert!(src.contains("workgroupBarrier"));
        assert!(src.contains("ReduceNdParams"));
    }

    #[test]
    fn wgsl_reduction_nd_max_uses_max_fn() {
        let src = reduction_nd_wgsl("max");
        assert!(src.contains("max(acc, val)"));
        assert!(src.contains("max(acc2, val)"));
    }

    #[test]
    fn wgsl_reduction_nd_min_uses_min_fn() {
        let src = reduction_nd_wgsl("min");
        assert!(src.contains("min(acc, val)"));
        assert!(src.contains("min(acc2, val)"));
    }

    #[test]
    fn wgsl_reduction_nd_mean_divides_by_dk() {
        let src = reduction_nd_wgsl("mean");
        assert!(src.contains("shared_data[0] / f32(params.dk)"));
        assert!(src.contains("acc + val"));
    }

    #[test]
    fn wgsl_reduction_nd_sum_does_not_divide() {
        let src = reduction_nd_wgsl("sum");
        assert!(!src.contains("/ f32(params.dk)"));
    }

    #[test]
    fn wgsl_reduction_nd_decodes_2d_dispatch() {
        let src = reduction_nd_wgsl("sum");
        assert!(src.contains("wgid.y * params.grid_x + wgid.x"));
    }

    #[test]
    fn wgsl_reduction_nd_uses_strided_loop() {
        let src = reduction_nd_wgsl("sum");
        assert!(src.contains("i = i + 256u"));
    }

    // ── binary_wgsl tests ─────────────────────────────────────────────────

    #[test]
    fn wgsl_binary_add() {
        let src = binary_wgsl("add");
        assert!(src.contains("a + b"));
        assert!(src.contains("lhs"));
        assert!(src.contains("rhs"));
    }

    #[test]
    fn wgsl_binary_all_ops() {
        assert!(binary_wgsl("sub").contains("a - b"));
        assert!(binary_wgsl("mul").contains("a * b"));
        assert!(binary_wgsl("div").contains("a / b"));
        assert!(binary_wgsl("max").contains("max(a, b)"));
        assert!(binary_wgsl("min").contains("min(a, b)"));
        assert!(binary_wgsl("pow").contains("pow(a, b)"));
        // Unknown op is identity on lhs.
        assert!(binary_wgsl("unknown_op").contains("output[i] = a;"));
    }

    #[test]
    fn wgsl_binary_workgroup_size() {
        let src = binary_wgsl("add");
        assert!(src.contains("@workgroup_size(256)"));
    }

    // ── conv2d_wgsl tests ─────────────────────────────────────────────────

    #[test]
    fn wgsl_conv2d_contains_workgroup() {
        let src = conv2d_wgsl(1, 3, 32, 32, 16, 3, 3, 30, 30, 1, 1, 0, 0);
        assert!(src.contains("@compute @workgroup_size(8, 8)"));
    }

    #[test]
    fn wgsl_conv2d_contains_storage_bindings() {
        let src = conv2d_wgsl(1, 3, 32, 32, 16, 3, 3, 30, 30, 1, 1, 0, 0);
        assert!(src.contains("var<storage, read>       input:"));
        // "filter" is a reserved WGSL keyword; the binding was renamed to kernel_w.
        assert!(src.contains("var<storage, read>       kernel_w:"));
        assert!(src.contains("var<storage, read_write> output:"));
    }

    #[test]
    fn wgsl_conv2d_embeds_dimensions() {
        let src = conv2d_wgsl(2, 8, 64, 64, 32, 5, 5, 60, 60, 1, 1, 0, 0);
        // Check that the shape constants appear in the shader
        assert!(src.contains("8u")); // c_in
        assert!(src.contains("64u")); // h_in or w_in
        assert!(src.contains("32u")); // k_out
        assert!(src.contains("5u")); // fh or fw
        assert!(src.contains("60u")); // oh or ow
    }

    #[test]
    fn wgsl_conv2d_has_padding_check() {
        let src = conv2d_wgsl(1, 1, 8, 8, 1, 3, 3, 8, 8, 1, 1, 1, 1);
        // Padding check with signed comparison
        assert!(src.contains("iy_raw >= 0"));
        assert!(src.contains("ix_raw >= 0"));
    }

    #[test]
    fn wgsl_conv2d_has_stride() {
        let src = conv2d_wgsl(1, 1, 8, 8, 1, 3, 3, 3, 3, 2, 2, 0, 0);
        assert!(src.contains("2u")); // stride
    }

    // ── attention_wgsl tests ──────────────────────────────────────────────

    #[test]
    fn wgsl_attention_contains_workgroup() {
        let src = attention_wgsl(4, 8, 8, 64, 0.125, false);
        assert!(src.contains("@compute @workgroup_size(64)"));
    }

    #[test]
    fn wgsl_attention_contains_storage_bindings() {
        let src = attention_wgsl(4, 8, 8, 64, 0.125, false);
        assert!(src.contains("var<storage, read>       q_buf:"));
        assert!(src.contains("var<storage, read>       k_buf:"));
        assert!(src.contains("var<storage, read>       v_buf:"));
        assert!(src.contains("var<storage, read_write> o_buf:"));
    }

    #[test]
    fn wgsl_attention_stable_softmax() {
        let src = attention_wgsl(1, 4, 4, 32, 0.25, false);
        assert!(src.contains("max_score"));
        assert!(src.contains("exp(score - max_score)"));
        assert!(src.contains("sum_exp"));
    }

    #[test]
    fn wgsl_attention_causal_mask() {
        let src_causal = attention_wgsl(1, 4, 4, 32, 0.25, true);
        assert!(src_causal.contains("sk > sq"));

        let src_non_causal = attention_wgsl(1, 4, 4, 32, 0.25, false);
        assert!(!src_non_causal.contains("sk > sq"));
    }

    #[test]
    fn wgsl_attention_embeds_scale() {
        let src = attention_wgsl(2, 16, 16, 64, 0.125, false);
        assert!(src.contains("0.125"));
    }

    // ── batched_gemm_wgsl tests ────────────────────────────────────────────

    #[test]
    fn wgsl_batched_gemm_contains_batch_params() {
        let src = batched_gemm_wgsl(16);
        assert!(src.contains("batch_count"));
        assert!(src.contains("stride_a"));
        assert!(src.contains("stride_b"));
        assert!(src.contains("stride_c"));
    }

    #[test]
    fn wgsl_batched_gemm_contains_workgroup() {
        let src = batched_gemm_wgsl(16);
        assert!(src.contains("@compute @workgroup_size(16, 16)"));
        assert!(src.contains("BatchedGemmParams"));
    }

    #[test]
    fn wgsl_batched_gemm_uses_batch_index() {
        let src = batched_gemm_wgsl(8);
        assert!(src.contains("batch_index"));
        assert!(src.contains("gid.z"));
    }

    #[test]
    fn wgsl_batched_gemm_tile_size_embedded() {
        let src8 = batched_gemm_wgsl(8);
        assert!(src8.contains("@workgroup_size(8, 8)"));
        let src32 = batched_gemm_wgsl(32);
        assert!(src32.contains("@workgroup_size(32, 32)"));
    }

    #[test]
    fn wgsl_batched_gemm_has_transpose_flags() {
        let src = batched_gemm_wgsl(8);
        assert!(src.contains("trans_a:  u32"));
        assert!(src.contains("trans_b:  u32"));
        assert!(src.contains("lda:      u32"));
        assert!(src.contains("ldb:      u32"));
        assert!(src.contains("ldc:      u32"));
        // Per-batch offset is applied to every index form, keyed on the
        // per-batch leading dimensions.
        assert!(src.contains("a[a_offset + r * params.lda + i]"));
        assert!(src.contains("a[a_offset + i * params.lda + r]"));
        assert!(src.contains("b[b_offset + i * params.ldb + col]"));
        assert!(src.contains("b[b_offset + col * params.ldb + i]"));
        assert!(src.contains("row * params.ldc + col"));
    }

    #[test]
    fn wgsl_batched_gemm_uses_shared_memory_tiling() {
        let src = batched_gemm_wgsl(8);
        assert!(src.contains("var<workgroup> tile_a"));
        assert!(src.contains("var<workgroup> tile_b"));
        assert!(src.contains("workgroupBarrier"));
        assert!(src.contains("array<array<f32, 8>, 8>"));
    }

    // ── gemm_wgsl_f16 tests ─────────────────────────────────────────────

    #[test]
    fn wgsl_gemm_f16_enables_extension() {
        let src = gemm_wgsl_f16(16);
        assert!(src.contains("enable f16;"));
    }

    #[test]
    fn wgsl_gemm_f16_uses_f16_storage() {
        let src = gemm_wgsl_f16(16);
        assert!(src.contains("array<f16>"));
    }

    #[test]
    fn wgsl_gemm_f16_accumulates_in_f32() {
        let src = gemm_wgsl_f16(16);
        assert!(src.contains("var acc: f32 = 0.0;"));
        // The f16-typed tiles are widened to f32 at the point they are
        // multiplied — accumulation itself is f32, but the intermediate
        // shared-memory staging holds f16 (see `wgsl_gemm_f16_uses_shared_memory_tiling`).
        assert!(src.contains("f32(tile_a["));
        assert!(src.contains("f32(tile_b["));
    }

    #[test]
    fn wgsl_gemm_f16_uses_shared_memory_tiling() {
        let src = gemm_wgsl_f16(16);
        assert!(src.contains("var<workgroup> tile_a"));
        assert!(src.contains("var<workgroup> tile_b"));
        assert!(src.contains("workgroupBarrier"));
        // The tile itself is f16 (halves workgroup memory vs. staging f32),
        // and the tile dimension is embedded in the array declaration.
        assert!(src.contains("array<array<f16, 16>, 16>"));
    }

    #[test]
    fn wgsl_gemm_f16_load_helpers_guard_last_partial_tile() {
        // Regression: a per-thread dot-product loop only ever indexes
        // `i < params.k`, so the original (untiled) `load_a`/`load_b` had no
        // bounds guard. The tiled staging loop reads `t * ts + lc` /
        // `t * ts + lr`, which routinely exceeds `params.k` / `params.m` on
        // the last (partial) tile — an unguarded read would silently pull in
        // a neighbouring row/column instead of contributing a zero pad,
        // corrupting the result rather than erroring.
        let src = gemm_wgsl_f16(16);
        assert!(src.contains("if (r >= params.m || i >= params.k) { return 0.0h; }"));
        assert!(src.contains("if (i >= params.k || col >= params.n) { return 0.0h; }"));
    }

    #[test]
    fn wgsl_gemm_f16_contains_workgroup() {
        let src = gemm_wgsl_f16(8);
        assert!(src.contains("@compute @workgroup_size(8, 8)"));
        assert!(src.contains("GemmParams"));
    }

    #[test]
    fn wgsl_gemm_f16_has_transpose_flags() {
        let src = gemm_wgsl_f16(8);
        // Transpose flags and leading dimensions live in the uniform struct,
        // mirroring gemm_wgsl (see wgsl_gemm_has_transpose_flags).
        assert!(src.contains("trans_a: u32"));
        assert!(src.contains("trans_b: u32"));
        assert!(src.contains("lda:     u32"));
        assert!(src.contains("ldb:     u32"));
        assert!(src.contains("ldc:     u32"));
        // Both row-major and column-major index forms. `load_a`/`load_b`
        // return raw `f16` (the tiled staging loop stores them into an
        // f16-typed workgroup tile); the f16 -> f32 widening happens later,
        // at the point a tile element is multiplied — see
        // `wgsl_gemm_f16_accumulates_in_f32`.
        assert!(src.contains("return a[r * params.lda + i];"));
        assert!(src.contains("return a[i * params.lda + r];"));
        assert!(src.contains("return b[i * params.ldb + col];"));
        assert!(src.contains("return b[col * params.ldb + i];"));
        assert!(src.contains("row * params.ldc + col"));
    }

    #[test]
    fn wgsl_attention_embeds_dimensions() {
        let src = attention_wgsl(8, 32, 32, 128, 0.088, true);
        assert!(src.contains("128u")); // head_dim
        assert!(src.contains("32u")); // seq_q or seq_kv
        assert!(src.contains("8u")); // batch_heads
    }
}
