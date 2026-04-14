//! MSL (Metal Shading Language) kernel source generation.
//!
//! Each function in this module returns a `&'static str` or `String` containing
//! a complete MSL translation unit.  The kernels are intentionally kept simple
//! and correct — they serve as reference implementations that can be replaced by
//! hand-tuned variants later.

// ─── GEMM ─────────────────────────────────────────────────────────────────────

/// MSL source for a single-precision GEMM kernel (`C = alpha*A*B + beta*C`).
///
/// Thread-group tiling is left to the hardware scheduler; this is a naive
/// reference implementation for correctness validation.
pub fn gemm_msl() -> &'static str {
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

kernel void gemm_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.m || col >= params.n) return;

    float acc = 0.0f;
    for (uint i = 0; i < params.k; i++) {
        acc += a[row * params.k + i] * b[i * params.n + col];
    }
    uint out_idx = row * params.n + col;
    c[out_idx] = params.alpha * acc + params.beta * c[out_idx];
}
"#
}

// ─── Elementwise unary ────────────────────────────────────────────────────────

/// MSL source for an element-wise unary kernel.
///
/// The element count is passed as a constant buffer at slot 2 so the kernel
/// can guard against out-of-bounds threads.  (Metal buffer pointers have no
/// `.get_elements()` method — they are raw pointers.)
///
/// Supported `op` values: `"relu"`, `"sigmoid"`, `"tanh"`, `"exp"`, `"log"`,
/// `"sqrt"`, `"abs"`, `"neg"`.  Unknown ops become identity.
pub fn elementwise_msl(op: &str) -> String {
    let op_expr = match op {
        "relu" => "max(x, 0.0f)",
        "sigmoid" => "1.0f / (1.0f + exp(-x))",
        "tanh" => "tanh(x)",
        "exp" => "exp(x)",
        "log" => "log(x)",
        "sqrt" => "sqrt(x)",
        "abs" => "abs(x)",
        "neg" => "-x",
        _ => "x", // identity fallback
    };
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= count) return;
    float x = input[gid];
    output[gid] = {op};
}}
"#,
        op = op_expr
    )
}

// ─── Binary elementwise ──────────────────────────────────────────────────────

/// MSL source for a binary element-wise kernel.
///
/// Supported `op` values: `"add"`, `"sub"`, `"mul"`, `"div"`, `"max"`,
/// `"min"`, `"pow"`.  Unknown ops fall back to copying `a`.
pub fn binary_msl(op: &str) -> String {
    let op_expr = match op {
        "add" => "a[tid] + b[tid]",
        "sub" => "a[tid] - b[tid]",
        "mul" => "a[tid] * b[tid]",
        "div" => "a[tid] / b[tid]",
        "max" => "max(a[tid], b[tid])",
        "min" => "min(a[tid], b[tid])",
        "pow" => "pow(a[tid], b[tid])",
        _ => "a[tid]", // identity fallback
    };
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void binary_f32(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= n) return;
    out[tid] = {op};
}}
"#,
        op = op_expr
    )
}

// ─── Reduction ────────────────────────────────────────────────────────────────

/// MSL source for a workgroup-based parallel reduction kernel.
///
/// Uses threadgroup (shared) memory for a tree-based parallel reduction.
/// Each threadgroup reduces one (outer, inner) slice of the input tensor
/// along the reduction axis and writes one output element.
///
/// Supported `op` values: `"sum"`, `"max"`, `"min"`, `"mean"`.
/// Unknown ops return an empty string.
pub fn reduction_msl(op: &str) -> String {
    let identity;
    let reduce_fn_body;
    let kernel_name;
    let is_mean;

    match op {
        "sum" => {
            identity = "0.0f";
            reduce_fn_body = "return a + b;";
            kernel_name = "reduce_sum_f32";
            is_mean = false;
        }
        "max" => {
            identity = "-INFINITY";
            reduce_fn_body = "return (a > b) ? a : b;";
            kernel_name = "reduce_max_f32";
            is_mean = false;
        }
        "min" => {
            identity = "INFINITY";
            reduce_fn_body = "return (a < b) ? a : b;";
            kernel_name = "reduce_min_f32";
            is_mean = false;
        }
        "mean" => {
            identity = "0.0f";
            reduce_fn_body = "return a + b;";
            kernel_name = "reduce_mean_f32";
            is_mean = true;
        }
        _ => return String::new(),
    }

    let final_expr = if is_mean {
        "sdata[0] / float(reduce_size)"
    } else {
        "sdata[0]"
    };

    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

inline float reduce_fn(float a, float b) {{
    {reduce_fn_body}
}}

kernel void {kernel_name}(
    device const float* input   [[buffer(0)]],
    device float*       output  [[buffer(1)]],
    constant uint& outer_size   [[buffer(2)]],
    constant uint& reduce_size  [[buffer(3)]],
    constant uint& inner_size   [[buffer(4)]],
    threadgroup float* sdata    [[threadgroup(0)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint lid    [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint outer_idx = tg_id / inner_size;
    uint inner_idx = tg_id % inner_size;
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    float acc = {identity};
    for (uint r = lid; r < reduce_size; r += tg_size) {{
        uint idx = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        acc = reduce_fn(acc, input[idx]);
    }}
    sdata[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {{
        if (lid < s) {{
            sdata[lid] = reduce_fn(sdata[lid], sdata[lid + s]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (lid == 0) {{
        output[outer_idx * inner_size + inner_idx] = {final_expr};
    }}
}}
"#,
        reduce_fn_body = reduce_fn_body,
        kernel_name = kernel_name,
        identity = identity,
        final_expr = final_expr,
    )
}

/// Return the MSL function name for the given reduction op.
pub fn reduction_function_name(op: &str) -> &'static str {
    match op {
        "sum" => "reduce_sum_f32",
        "max" => "reduce_max_f32",
        "min" => "reduce_min_f32",
        "mean" => "reduce_mean_f32",
        _ => "unknown",
    }
}

// ─── Conv2D ──────────────────────────────────────────────────────────────────

/// MSL source for a single-precision Conv2D forward kernel (NCHW layout).
///
/// All convolution parameters are embedded as compile-time constants so no
/// parameter buffer is needed.  Each thread computes one output element.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_msl(
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    k_out: usize,
    fh: usize,
    fw: usize,
    oh: usize,
    ow: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

constant uint N_BATCH = {n};
constant uint C_IN    = {c_in};
constant uint H_IN    = {h_in};
constant uint W_IN    = {w_in};
constant uint K_OUT   = {k_out};
constant uint FH      = {fh};
constant uint FW      = {fw};
constant uint OH      = {oh};
constant uint OW      = {ow};
constant uint STRIDE_H = {stride_h};
constant uint STRIDE_W = {stride_w};
constant uint PAD_H   = {pad_h};
constant uint PAD_W   = {pad_w};

kernel void conv2d_forward_f32(
    device const float* input  [[buffer(0)]],
    device const float* filter [[buffer(1)]],
    device float*       output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    uint total = N_BATCH * K_OUT * OH * OW;
    if (gid >= total) return;

    uint ox  = gid % OW;
    uint tmp = gid / OW;
    uint oy  = tmp % OH;
    tmp      = tmp / OH;
    uint kf  = tmp % K_OUT;
    uint b   = tmp / K_OUT;

    float acc = 0.0f;
    for (uint ci = 0; ci < C_IN; ci++) {{
        for (uint fy = 0; fy < FH; fy++) {{
            for (uint fx = 0; fx < FW; fx++) {{
                int iy = int(oy * STRIDE_H + fy) - int(PAD_H);
                int ix = int(ox * STRIDE_W + fx) - int(PAD_W);
                if (iy >= 0 && uint(iy) < H_IN && ix >= 0 && uint(ix) < W_IN) {{
                    uint in_idx = ((b * C_IN + ci) * H_IN + uint(iy)) * W_IN + uint(ix);
                    uint f_idx  = ((kf * C_IN + ci) * FH + fy) * FW + fx;
                    acc += input[in_idx] * filter[f_idx];
                }}
            }}
        }}
    }}
    output[gid] = acc;
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

// ─── Attention ───────────────────────────────────────────────────────────────

/// MSL source for a single-precision scaled dot-product attention kernel.
///
/// Each thread handles one (batch_head, query_position) pair.
/// Uses numerically stable softmax (subtract max before exp).
/// Optional causal masking skips positions where `sk > sq`.
#[allow(clippy::too_many_arguments)]
pub fn attention_msl(
    batch_heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> String {
    let causal_u32: u32 = u32::from(causal);
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

constant uint BATCH_HEADS = {batch_heads};
constant uint SEQ_Q       = {seq_q};
constant uint SEQ_KV      = {seq_kv};
constant uint HEAD_DIM    = {head_dim};
constant float SCALE      = {scale};
constant uint CAUSAL      = {causal};

kernel void attention_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float*       O [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    uint total = BATCH_HEADS * SEQ_Q;
    if (gid >= total) return;

    uint sq = gid % SEQ_Q;
    uint bh = gid / SEQ_Q;

    uint q_off = (bh * SEQ_Q + sq) * HEAD_DIM;

    // Pass 1: find max score for numerical stability
    float max_score = -INFINITY;
    for (uint sk = 0; sk < SEQ_KV; sk++) {{
        if (CAUSAL != 0 && sk > sq) continue;
        float dot = 0.0f;
        uint k_off = (bh * SEQ_KV + sk) * HEAD_DIM;
        for (uint d = 0; d < HEAD_DIM; d++) {{
            dot += Q[q_off + d] * K[k_off + d];
        }}
        float score = dot * SCALE;
        max_score = max(max_score, score);
    }}

    // Pass 2: softmax weights + accumulate output
    float sum_exp = 0.0f;
    uint o_off = (bh * SEQ_Q + sq) * HEAD_DIM;
    for (uint d = 0; d < HEAD_DIM; d++) {{
        O[o_off + d] = 0.0f;
    }}
    for (uint sk = 0; sk < SEQ_KV; sk++) {{
        if (CAUSAL != 0 && sk > sq) continue;
        float dot = 0.0f;
        uint k_off = (bh * SEQ_KV + sk) * HEAD_DIM;
        for (uint d = 0; d < HEAD_DIM; d++) {{
            dot += Q[q_off + d] * K[k_off + d];
        }}
        float w = exp(dot * SCALE - max_score);
        sum_exp += w;
        uint v_off = (bh * SEQ_KV + sk) * HEAD_DIM;
        for (uint d = 0; d < HEAD_DIM; d++) {{
            O[o_off + d] += w * V[v_off + d];
        }}
    }}

    // Normalize
    if (sum_exp > 0.0f) {{
        for (uint d = 0; d < HEAD_DIM; d++) {{
            O[o_off + d] /= sum_exp;
        }}
    }}
}}
"#,
        batch_heads = batch_heads,
        seq_q = seq_q,
        seq_kv = seq_kv,
        head_dim = head_dim,
        scale = scale,
        causal = causal_u32,
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msl_gemm_contains_kernel_name() {
        let src = gemm_msl();
        assert!(src.contains("gemm_f32"));
        assert!(src.contains("GemmParams"));
        assert!(src.contains("metal_stdlib"));
    }

    #[test]
    fn msl_elementwise_relu_contains_max() {
        let src = elementwise_msl("relu");
        assert!(src.contains("max(x, 0.0f)"));
        assert!(src.contains("elementwise_f32"));
        // The count parameter is present (spacing may vary due to alignment).
        assert!(src.contains("constant uint&"));
        assert!(src.contains("count"));
    }

    #[test]
    fn msl_elementwise_sigmoid_correct() {
        let src = elementwise_msl("sigmoid");
        assert!(src.contains("1.0f / (1.0f + exp(-x))"));
    }

    #[test]
    fn msl_elementwise_tanh_correct() {
        let src = elementwise_msl("tanh");
        assert!(src.contains("tanh(x)"));
    }

    #[test]
    fn msl_elementwise_exp_correct() {
        let src = elementwise_msl("exp");
        assert!(src.contains("exp(x)"));
    }

    #[test]
    fn msl_elementwise_log_correct() {
        let src = elementwise_msl("log");
        assert!(src.contains("log(x)"));
    }

    #[test]
    fn msl_elementwise_sqrt_correct() {
        let src = elementwise_msl("sqrt");
        assert!(src.contains("sqrt(x)"));
    }

    #[test]
    fn msl_elementwise_abs_correct() {
        let src = elementwise_msl("abs");
        assert!(src.contains("abs(x)"));
    }

    #[test]
    fn msl_elementwise_neg_correct() {
        let src = elementwise_msl("neg");
        assert!(src.contains("-x"));
    }

    #[test]
    fn msl_elementwise_unknown_op_identity() {
        let src = elementwise_msl("unknown_op");
        // Should fall through to identity — output[gid] = x
        assert!(src.contains("output[gid] = x;"));
    }

    #[test]
    fn msl_reduction_sum_threadgroup() {
        let src = reduction_msl("sum");
        assert!(src.contains("reduce_sum_f32"));
        assert!(src.contains("threadgroup float* sdata"));
        assert!(src.contains("threadgroup_barrier"));
        assert!(!src.contains("atomic"));
    }

    #[test]
    fn msl_reduction_unknown_empty() {
        assert!(reduction_msl("unknown_op").is_empty());
    }

    #[test]
    fn msl_reduction_max_contains_kernel() {
        let src = reduction_msl("max");
        assert!(src.contains("reduce_max_f32"));
        assert!(src.contains("-INFINITY"));
    }

    #[test]
    fn msl_reduction_min_contains_kernel() {
        let src = reduction_msl("min");
        assert!(src.contains("reduce_min_f32"));
        assert!(src.contains("INFINITY"));
    }

    #[test]
    fn msl_reduction_mean_contains_kernel() {
        let src = reduction_msl("mean");
        assert!(src.contains("reduce_mean_f32"));
        assert!(src.contains("float(reduce_size)"));
    }

    #[test]
    fn msl_reduction_function_names() {
        assert_eq!(reduction_function_name("sum"), "reduce_sum_f32");
        assert_eq!(reduction_function_name("max"), "reduce_max_f32");
        assert_eq!(reduction_function_name("min"), "reduce_min_f32");
        assert_eq!(reduction_function_name("mean"), "reduce_mean_f32");
        assert_eq!(reduction_function_name("xyz"), "unknown");
    }

    #[test]
    fn msl_binary_add_correct() {
        let src = binary_msl("add");
        assert!(src.contains("binary_f32"));
        assert!(src.contains("a[tid] + b[tid]"));
    }

    #[test]
    fn msl_binary_sub_correct() {
        let src = binary_msl("sub");
        assert!(src.contains("a[tid] - b[tid]"));
    }

    #[test]
    fn msl_binary_mul_correct() {
        let src = binary_msl("mul");
        assert!(src.contains("a[tid] * b[tid]"));
    }

    #[test]
    fn msl_binary_div_correct() {
        let src = binary_msl("div");
        assert!(src.contains("a[tid] / b[tid]"));
    }

    #[test]
    fn msl_binary_max_correct() {
        let src = binary_msl("max");
        assert!(src.contains("max(a[tid], b[tid])"));
    }

    #[test]
    fn msl_binary_min_correct() {
        let src = binary_msl("min");
        assert!(src.contains("min(a[tid], b[tid])"));
    }

    #[test]
    fn msl_binary_pow_correct() {
        let src = binary_msl("pow");
        assert!(src.contains("pow(a[tid], b[tid])"));
    }

    #[test]
    fn msl_binary_unknown_identity() {
        let src = binary_msl("unknown");
        assert!(src.contains("out[tid] = a[tid];"));
    }

    /// Validate that the GEMM MSL actually compiles on this machine.
    #[cfg(target_os = "macos")]
    #[test]
    fn msl_gemm_compiles_on_macos() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        match device.new_library_with_source(gemm_msl(), &opts) {
            Ok(_) => {} // success
            Err(e) => panic!("GEMM MSL failed to compile: {e}"),
        }
    }

    /// Validate that the elementwise MSL actually compiles on this machine.
    #[cfg(target_os = "macos")]
    #[test]
    fn msl_elementwise_compiles_on_macos() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        for op in &[
            "relu", "sigmoid", "tanh", "exp", "log", "sqrt", "abs", "neg",
        ] {
            let src = elementwise_msl(op);
            match device.new_library_with_source(&src, &opts) {
                Ok(_) => {}
                Err(e) => panic!("elementwise `{op}` MSL failed to compile: {e}"),
            }
        }
    }

    /// Validate that the binary MSL actually compiles on this machine.
    #[cfg(target_os = "macos")]
    #[test]
    fn msl_binary_compiles_on_macos() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        for op in &["add", "sub", "mul", "div", "max", "min", "pow"] {
            let src = binary_msl(op);
            match device.new_library_with_source(&src, &opts) {
                Ok(_) => {}
                Err(e) => panic!("binary `{op}` MSL failed to compile: {e}"),
            }
        }
    }

    /// Validate that the reduction MSL actually compiles on this machine.
    #[cfg(target_os = "macos")]
    #[test]
    fn msl_reduction_compiles_on_macos() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        for op in &["sum", "max", "min", "mean"] {
            let src = reduction_msl(op);
            match device.new_library_with_source(&src, &opts) {
                Ok(_) => {}
                Err(e) => panic!("reduction `{op}` MSL failed to compile: {e}"),
            }
        }
    }

    // ── Conv2D MSL tests ──────────────────────────────────────────────────────

    #[test]
    fn msl_conv2d_contains_kernel() {
        let src = conv2d_msl(1, 3, 8, 8, 16, 3, 3, 6, 6, 1, 1, 0, 0);
        assert!(src.contains("kernel void"));
        assert!(src.contains("conv2d_forward_f32"));
        assert!(src.contains("device float*"));
        assert!(src.contains("metal_stdlib"));
        assert!(src.contains("C_IN"));
        assert!(src.contains("K_OUT"));
    }

    #[test]
    fn msl_conv2d_embeds_params() {
        let src = conv2d_msl(2, 3, 32, 32, 64, 5, 5, 28, 28, 1, 1, 0, 0);
        assert!(src.contains("N_BATCH = 2"));
        assert!(src.contains("C_IN    = 3"));
        assert!(src.contains("K_OUT   = 64"));
        assert!(src.contains("FH      = 5"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn msl_conv2d_compiles_on_macos() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        let src = conv2d_msl(1, 1, 4, 4, 1, 3, 3, 2, 2, 1, 1, 0, 0);
        match device.new_library_with_source(&src, &opts) {
            Ok(_) => {}
            Err(e) => panic!("conv2d MSL failed to compile: {e}"),
        }
    }

    // ── Attention MSL tests ───────────────────────────────────────────────────

    #[test]
    fn msl_attention_contains_kernel() {
        let src = attention_msl(4, 8, 8, 64, 0.125, false);
        assert!(src.contains("kernel void"));
        assert!(src.contains("attention_f32"));
        assert!(src.contains("device float*"));
        assert!(src.contains("metal_stdlib"));
        assert!(src.contains("BATCH_HEADS"));
        assert!(src.contains("HEAD_DIM"));
    }

    #[test]
    fn msl_attention_causal_flag() {
        let src_no = attention_msl(1, 4, 4, 32, 0.25, false);
        assert!(src_no.contains("CAUSAL      = 0"));
        let src_yes = attention_msl(1, 4, 4, 32, 0.25, true);
        assert!(src_yes.contains("CAUSAL      = 1"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn msl_attention_compiles_on_macos() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        for causal in [false, true] {
            let src = attention_msl(2, 4, 4, 32, 0.125, causal);
            match device.new_library_with_source(&src, &opts) {
                Ok(_) => {}
                Err(e) => panic!("attention MSL (causal={causal}) failed to compile: {e}"),
            }
        }
    }
}
