//! WGSL shader source generation for common compute kernels.
//!
//! Each function returns a complete, self-contained WGSL source string
//! suitable for passing to `device.create_shader_module()`.

/// Generate WGSL source for a tiled GEMM kernel: `C = alpha * A * B + beta * C`.
///
/// Uses `tile_size × tile_size` workgroups.  Both A and B are stored row-major.
///
/// # Arguments
///
/// * `tile_size` — workgroup tile dimension (e.g. 8, 16, 32).
pub fn gemm_wgsl(tile_size: u32) -> String {
    format!(
        r#"
struct GemmParams {{
    m:     u32,
    n:     u32,
    k:     u32,
    alpha: f32,
    beta:  f32,
}}

@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read>       b:      array<f32>;
@group(0) @binding(2) var<storage, read_write> c:      array<f32>;
@group(0) @binding(3) var<uniform>             params: GemmParams;

@compute @workgroup_size({ts}, {ts})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.y;
    let col = gid.x;
    if (row >= params.m || col >= params.n) {{ return; }}

    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {{
        acc += a[row * params.k + i] * b[i * params.n + col];
    }}

    let idx = row * params.n + col;
    c[idx] = params.alpha * acc + params.beta * c[idx];
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
    // Neutral elements and combine expressions for each operation.
    let (neutral, combine) = match op {
        "max" => ("f32(-1e38)", "max(acc, val)"),
        "min" => ("f32(1e38)", "min(acc, val)"),
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
// input : [{n}, {c_in}, {h_in}, {w_in}]
// filter: [{k_out}, {c_in}, {fh}, {fw}]
// output: [{n}, {k_out}, {oh}, {ow}]

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read>       filter: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

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
                    acc += input[in_idx] * filter[f_idx];
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
    let causal_check = if causal {
        "if (sk > sq) { score = f32(-1e38); } else {"
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

    // Pass 1: find max score for numerical stability
    var max_score: f32 = f32(-1e38);
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
        let o_base = (bh * {seq_q}u + sq) * {head_dim}u;
        for (var d: u32 = 0u; d < {head_dim}u; d = d + 1u) {{
            // Accumulate in-place (we normalise after the loop).
            o_buf[o_base + d] += w * v_buf[v_base + d];
        }}
    }}

    // Pass 3: normalise
    if (sum_exp > 0.0) {{
        let o_base = (bh * {seq_q}u + sq) * {head_dim}u;
        for (var d: u32 = 0u; d < {head_dim}u; d = d + 1u) {{
            o_buf[o_base + d] /= sum_exp;
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
    )
}

/// Generate WGSL for the final scalar reduction of partial sums.
///
/// Takes a `partial_sums` array of length `num_groups` and reduces it to a
/// single value at `output[0]`.  Should be dispatched with a single workgroup
/// of 256 threads.
pub fn reduction_final_wgsl(op: &str) -> String {
    let (neutral, combine) = match op {
        "max" => ("f32(-1e38)", "max(acc, val)"),
        "min" => ("f32(1e38)", "min(acc, val)"),
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

    if (tid < params.num_groups) {{
        shared_data[tid] = partial_sums[tid];
    }} else {{
        shared_data[tid] = {neutral};
    }}
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
        assert!(src.contains("var<storage, read>       filter:"));
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

    #[test]
    fn wgsl_attention_embeds_dimensions() {
        let src = attention_wgsl(8, 32, 32, 128, 0.088, true);
        assert!(src.contains("128u")); // head_dim
        assert!(src.contains("32u")); // seq_q or seq_kv
        assert!(src.contains("8u")); // batch_heads
    }
}
