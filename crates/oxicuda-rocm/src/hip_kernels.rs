//! HIP C++ kernel source generation for AMD ROCm GPUs.
//!
//! Each function returns a `String` containing a complete HIP C++ device
//! translation unit suitable for runtime compilation via `hiprtcCompileProgram`
//! or embedding as a pre-compiled binary.
//!
//! The generated kernels follow the same patterns as CUDA kernels but use HIP
//! intrinsics (`hipBlockDim_x`, `hipThreadIdx_x`, `hipBlockIdx_x`, etc.)
//! and the `__global__` qualifier for device entry points.

// ─── GEMM ─────────────────────────────────────────────────────────────────────

/// HIP C++ source for a single-precision GEMM kernel: `C = alpha * A * B + beta * C`.
///
/// Uses a naive per-element approach (no shared-memory tiling) — this is a
/// reference implementation for correctness validation.
///
/// Grid: `dim3((n+15)/16, (m+15)/16)`, Block: `dim3(16, 16)`.
pub fn gemm_hip(tile_size: u32) -> String {
    format!(
        r#"
extern "C" __global__ void gemm_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float*       __restrict__ c,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    float alpha,
    float beta
) {{
    unsigned int row = hipBlockIdx_y * {ts} + hipThreadIdx_y;
    unsigned int col = hipBlockIdx_x * {ts} + hipThreadIdx_x;
    if (row >= m || col >= n) return;

    float acc = 0.0f;
    for (unsigned int i = 0; i < k; ++i) {{
        acc += a[row * k + i] * b[i * n + col];
    }}

    unsigned int idx = row * n + col;
    c[idx] = alpha * acc + beta * c[idx];
}}
"#,
        ts = tile_size
    )
}

// ─── Batched GEMM ───────────────────────────────────────────────────────────

/// HIP C++ source for a batched single-precision GEMM kernel:
/// `C_b = alpha * A_b * B_b + beta * C_b` for each batch `b` in `0..batch_count`.
///
/// Uses `hipBlockIdx_z` as the batch index.  Each batch's matrices are located
/// at `a + batch_index * stride_a`, etc.
///
/// Grid: `dim3((n+ts-1)/ts, (m+ts-1)/ts, batch_count)`, Block: `dim3(ts, ts)`.
pub fn batched_gemm_hip(tile_size: u32) -> String {
    format!(
        r#"
extern "C" __global__ void batched_gemm_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float*       __restrict__ c,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    float alpha,
    float beta,
    unsigned int batch_count,
    unsigned int stride_a,
    unsigned int stride_b,
    unsigned int stride_c
) {{
    unsigned int batch_index = hipBlockIdx_z;
    if (batch_index >= batch_count) return;

    unsigned int row = hipBlockIdx_y * {ts} + hipThreadIdx_y;
    unsigned int col = hipBlockIdx_x * {ts} + hipThreadIdx_x;
    if (row >= m || col >= n) return;

    const float* a_batch = a + batch_index * stride_a;
    const float* b_batch = b + batch_index * stride_b;
    float*       c_batch = c + batch_index * stride_c;

    float acc = 0.0f;
    for (unsigned int i = 0; i < k; ++i) {{
        acc += a_batch[row * k + i] * b_batch[i * n + col];
    }}

    unsigned int idx = row * n + col;
    c_batch[idx] = alpha * acc + beta * c_batch[idx];
}}
"#,
        ts = tile_size
    )
}

// ─── Elementwise unary ────────────────────────────────────────────────────────

/// HIP C++ source for an element-wise unary kernel.
///
/// Grid: `((n + 255) / 256)`, Block: `256`.
///
/// Supported `op` values: `"relu"`, `"sigmoid"`, `"tanh"`, `"exp"`, `"log"`,
/// `"sqrt"`, `"abs"`, `"neg"`.  Unknown ops become identity.
pub fn elementwise_hip(op: &str) -> String {
    let op_expr = match op {
        "relu" => "fmaxf(x, 0.0f)",
        "sigmoid" => "1.0f / (1.0f + expf(-x))",
        "tanh" => "tanhf(x)",
        "exp" => "expf(x)",
        "log" => "logf(x)",
        "sqrt" => "sqrtf(x)",
        "abs" => "fabsf(x)",
        "neg" => "-x",
        _ => "x", // identity fallback
    };
    format!(
        r#"
extern "C" __global__ void elementwise_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    unsigned int n
) {{
    unsigned int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (gid >= n) return;
    float x = input[gid];
    output[gid] = {op};
}}
"#,
        op = op_expr
    )
}

// ─── Binary elementwise ──────────────────────────────────────────────────────

/// HIP C++ source for a binary element-wise kernel.
///
/// Grid: `((n + 255) / 256)`, Block: `256`.
///
/// Supported `op` values: `"add"`, `"sub"`, `"mul"`, `"div"`, `"max"`,
/// `"min"`.  Unknown ops fall back to copying `a`.
pub fn binary_hip(op: &str) -> String {
    let op_expr = match op {
        "add" => "a[tid] + b[tid]",
        "sub" => "a[tid] - b[tid]",
        "mul" => "a[tid] * b[tid]",
        "div" => "a[tid] / b[tid]",
        "max" => "fmaxf(a[tid], b[tid])",
        "min" => "fminf(a[tid], b[tid])",
        _ => "a[tid]", // identity fallback
    };
    format!(
        r#"
extern "C" __global__ void binary_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float*       __restrict__ out,
    unsigned int n
) {{
    unsigned int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (tid >= n) return;
    out[tid] = {op};
}}
"#,
        op = op_expr
    )
}

// ─── Reduction ────────────────────────────────────────────────────────────────

/// HIP C++ source for a workgroup-based parallel reduction kernel.
///
/// Each block reduces one `(outer, inner)` along the reduction axis using
/// shared memory.  The result is a single element per block.
///
/// Grid: `(outer_size * inner_size)`, Block: `block_size`.
///
/// Supported `op` values: `"sum"`, `"max"`, `"min"`, `"mean"`.
/// Unknown ops return an empty string.
pub fn reduction_hip(op: &str) -> String {
    let identity;
    let combine;
    let kernel_name;
    let is_mean;

    match op {
        "sum" => {
            identity = "0.0f";
            combine = "a + b";
            kernel_name = "reduce_sum_f32";
            is_mean = false;
        }
        "max" => {
            identity = "-HUGE_VALF";
            combine = "fmaxf(a, b)";
            kernel_name = "reduce_max_f32";
            is_mean = false;
        }
        "min" => {
            identity = "HUGE_VALF";
            combine = "fminf(a, b)";
            kernel_name = "reduce_min_f32";
            is_mean = false;
        }
        "mean" => {
            identity = "0.0f";
            combine = "a + b";
            kernel_name = "reduce_mean_f32";
            is_mean = true;
        }
        _ => return String::new(),
    }

    let final_expr = if is_mean {
        "sdata[0] / (float)reduce_size"
    } else {
        "sdata[0]"
    };

    format!(
        r#"
#include <math.h>

__device__ inline float reduce_op(float a, float b) {{
    return {combine};
}}

extern "C" __global__ void {kernel_name}(
    const float* __restrict__ input,
    float*       __restrict__ output,
    unsigned int outer_size,
    unsigned int reduce_size,
    unsigned int inner_size
) {{
    extern __shared__ float sdata[];

    unsigned int tg_id = hipBlockIdx_x;
    unsigned int lid   = hipThreadIdx_x;
    unsigned int bs    = hipBlockDim_x;

    unsigned int outer_idx = tg_id / inner_size;
    unsigned int inner_idx = tg_id % inner_size;
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    // Each thread accumulates over its strided slice of the reduction axis.
    float acc = {identity};
    for (unsigned int r = lid; r < reduce_size; r += bs) {{
        unsigned int src_idx = outer_idx * (reduce_size * inner_size)
                             + r * inner_size
                             + inner_idx;
        acc = reduce_op(acc, input[src_idx]);
    }}
    sdata[lid] = acc;

    __syncthreads();

    // Tree reduction in shared memory.
    for (unsigned int stride = bs / 2; stride > 0; stride >>= 1) {{
        if (lid < stride) {{
            sdata[lid] = reduce_op(sdata[lid], sdata[lid + stride]);
        }}
        __syncthreads();
    }}

    // Thread 0 writes the final result.
    if (lid == 0) {{
        unsigned int out_idx = outer_idx * inner_size + inner_idx;
        output[out_idx] = {final_expr};
    }}
}}
"#,
        combine = combine,
        kernel_name = kernel_name,
        identity = identity,
        final_expr = final_expr,
    )
}

/// Return the HIP kernel function name for a given reduction op.
pub fn reduction_function_name(op: &str) -> &'static str {
    match op {
        "sum" => "reduce_sum_f32",
        "max" => "reduce_max_f32",
        "min" => "reduce_min_f32",
        "mean" => "reduce_mean_f32",
        _ => "reduce_sum_f32",
    }
}

// ─── Attention ────────────────────────────────────────────────────────────────

/// HIP C++ source for a naive scaled dot-product attention kernel.
///
/// Q, K, V are `[batch * heads, seq, head_dim]` row-major.
/// Output O is `[batch * heads, seq_q, head_dim]`.
///
/// Grid: `(seq_q, batch_heads)`, Block: `(head_dim_clamped)`.
pub fn attention_hip() -> &'static str {
    r#"
#include <math.h>

extern "C" __global__ void attention_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float*       __restrict__ o,
    unsigned int seq_q,
    unsigned int seq_kv,
    unsigned int head_dim,
    float scale,
    unsigned int causal
) {
    unsigned int bh = hipBlockIdx_y;
    unsigned int sq = hipBlockIdx_x;
    unsigned int d  = hipThreadIdx_x;
    if (sq >= seq_q || d >= head_dim) return;

    unsigned int q_offset = bh * seq_q * head_dim;
    unsigned int k_offset = bh * seq_kv * head_dim;
    unsigned int v_offset = k_offset;

    // Compute softmax denominator and weighted value accumulation.
    // Naive two-pass: first pass finds max, second pass computes exp-sum.
    float max_score = -HUGE_VALF;
    for (unsigned int sk = 0; sk < seq_kv; ++sk) {
        if (causal && sk > sq) break;
        float dot = 0.0f;
        for (unsigned int dd = 0; dd < head_dim; ++dd) {
            dot += q[q_offset + sq * head_dim + dd]
                 * k[k_offset + sk * head_dim + dd];
        }
        dot *= scale;
        if (dot > max_score) max_score = dot;
    }

    float sum_exp = 0.0f;
    float acc = 0.0f;
    for (unsigned int sk = 0; sk < seq_kv; ++sk) {
        if (causal && sk > sq) break;
        float dot = 0.0f;
        for (unsigned int dd = 0; dd < head_dim; ++dd) {
            dot += q[q_offset + sq * head_dim + dd]
                 * k[k_offset + sk * head_dim + dd];
        }
        dot *= scale;
        float w = expf(dot - max_score);
        sum_exp += w;
        acc += w * v[v_offset + sk * head_dim + d];
    }

    if (sum_exp > 0.0f) {
        o[q_offset + sq * head_dim + d] = acc / sum_exp;
    } else {
        o[q_offset + sq * head_dim + d] = 0.0f;
    }
}
"#
}

// ─── Conv2d ──────────────────────────────────────────────────────────────────

/// HIP C++ source for a naive im2col-free 2D convolution (forward pass).
///
/// Input:  `[N, C, H, W]`, Filter: `[K, C, FH, FW]`, Output: `[N, K, OH, OW]`.
/// Grid: `(OW, OH, N * K)`, Block: `(min(OW, 16), min(OH, 16))`.
pub fn conv2d_forward_hip() -> &'static str {
    r#"
extern "C" __global__ void conv2d_forward_f32(
    const float* __restrict__ input,
    const float* __restrict__ filter,
    float*       __restrict__ output,
    unsigned int n,
    unsigned int c,
    unsigned int h,
    unsigned int w,
    unsigned int k_out,
    unsigned int fh,
    unsigned int fw,
    unsigned int oh,
    unsigned int ow,
    unsigned int stride_h,
    unsigned int stride_w,
    unsigned int pad_h,
    unsigned int pad_w
) {
    unsigned int ox = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    unsigned int oy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    unsigned int nk = hipBlockIdx_z;
    unsigned int batch = nk / k_out;
    unsigned int kf    = nk % k_out;
    if (ox >= ow || oy >= oh || batch >= n) return;

    float acc = 0.0f;
    for (unsigned int ci = 0; ci < c; ++ci) {
        for (unsigned int fy = 0; fy < fh; ++fy) {
            for (unsigned int fx = 0; fx < fw; ++fx) {
                int iy = (int)(oy * stride_h + fy) - (int)pad_h;
                int ix = (int)(ox * stride_w + fx) - (int)pad_w;
                if (iy >= 0 && iy < (int)h && ix >= 0 && ix < (int)w) {
                    unsigned int in_idx = ((batch * c + ci) * h + (unsigned int)iy) * w
                                        + (unsigned int)ix;
                    unsigned int f_idx  = ((kf * c + ci) * fh + fy) * fw + fx;
                    acc += input[in_idx] * filter[f_idx];
                }
            }
        }
    }

    unsigned int out_idx = ((batch * k_out + kf) * oh + oy) * ow + ox;
    output[out_idx] = acc;
}
"#
}

// ─── FP16 GEMM ──────────────────────────────────────────────────────────────

/// HIP C++ source for a half-precision GEMM kernel: `C = alpha * A * B + beta * C`.
///
/// Uses `__half` (HIP FP16) for input/output buffers and `float` for
/// accumulation (mixed precision).  Conversions use `__half2float()` /
/// `__float2half()`.
///
/// Grid: `dim3((n+ts-1)/ts, (m+ts-1)/ts)`, Block: `dim3(ts, ts)`.
pub fn gemm_hip_f16(tile_size: u32) -> String {
    format!(
        r#"
#include <hip/hip_fp16.h>

extern "C" __global__ void gemm_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half*       __restrict__ c,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    float alpha,
    float beta
) {{
    unsigned int row = hipBlockIdx_y * {ts} + hipThreadIdx_y;
    unsigned int col = hipBlockIdx_x * {ts} + hipThreadIdx_x;
    if (row >= m || col >= n) return;

    float acc = 0.0f;
    for (unsigned int i = 0; i < k; ++i) {{
        acc += __half2float(a[row * k + i]) * __half2float(b[i * n + col]);
    }}

    unsigned int idx = row * n + col;
    float c_val = __half2float(c[idx]);
    c[idx] = __float2half(alpha * acc + beta * c_val);
}}
"#,
        ts = tile_size
    )
}

// ─── BF16 GEMM ──────────────────────────────────────────────────────────────

/// HIP C++ source for a BFloat16 GEMM kernel: `C = alpha * A * B + beta * C`.
///
/// Uses `hip_bfloat16` for input/output buffers and `float` for accumulation.
///
/// Grid: `dim3((n+ts-1)/ts, (m+ts-1)/ts)`, Block: `dim3(ts, ts)`.
pub fn gemm_hip_bf16(tile_size: u32) -> String {
    format!(
        r#"
#include <hip/hip_bfloat16.h>

extern "C" __global__ void gemm_bf16(
    const hip_bfloat16* __restrict__ a,
    const hip_bfloat16* __restrict__ b,
    hip_bfloat16*       __restrict__ c,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    float alpha,
    float beta
) {{
    unsigned int row = hipBlockIdx_y * {ts} + hipThreadIdx_y;
    unsigned int col = hipBlockIdx_x * {ts} + hipThreadIdx_x;
    if (row >= m || col >= n) return;

    float acc = 0.0f;
    for (unsigned int i = 0; i < k; ++i) {{
        acc += float(a[row * k + i]) * float(b[i * n + col]);
    }}

    unsigned int idx = row * n + col;
    float c_val = float(c[idx]);
    c[idx] = hip_bfloat16(alpha * acc + beta * c_val);
}}
"#,
        ts = tile_size
    )
}

// ─── FP16 Elementwise unary ─────────────────────────────────────────────────

/// HIP C++ source for an element-wise unary kernel operating on `__half` data.
///
/// Grid: `((n + 255) / 256)`, Block: `256`.
///
/// Supported `op` values: `"relu"`, `"sigmoid"`, `"tanh"`, `"exp"`, `"log"`,
/// `"sqrt"`, `"abs"`, `"neg"`.  Unknown ops become identity.
pub fn elementwise_hip_f16(op: &str) -> String {
    let op_expr = match op {
        "relu" => "fmaxf(x, 0.0f)",
        "sigmoid" => "1.0f / (1.0f + expf(-x))",
        "tanh" => "tanhf(x)",
        "exp" => "expf(x)",
        "log" => "logf(x)",
        "sqrt" => "sqrtf(x)",
        "abs" => "fabsf(x)",
        "neg" => "-x",
        _ => "x", // identity fallback
    };
    format!(
        r#"
#include <hip/hip_fp16.h>

extern "C" __global__ void elementwise_f16(
    const __half* __restrict__ input,
    __half*       __restrict__ output,
    unsigned int n
) {{
    unsigned int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (gid >= n) return;
    float x = __half2float(input[gid]);
    output[gid] = __float2half({op});
}}
"#,
        op = op_expr
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GEMM ──────────────────────────────────────────────────────────────

    #[test]
    fn hip_gemm_contains_global() {
        let src = gemm_hip(16);
        assert!(src.contains("__global__"));
        assert!(src.contains("gemm_f32"));
        assert!(src.contains("alpha"));
        assert!(src.contains("beta"));
    }

    #[test]
    fn hip_gemm_tile_size_embedded() {
        let src8 = gemm_hip(8);
        assert!(src8.contains("hipBlockIdx_y * 8"));
        let src32 = gemm_hip(32);
        assert!(src32.contains("hipBlockIdx_y * 32"));
    }

    // ── Batched GEMM ─────────────────────────────────────────────────────

    #[test]
    fn hip_batched_gemm_contains_global() {
        let src = batched_gemm_hip(16);
        assert!(src.contains("__global__"));
        assert!(src.contains("batched_gemm_f32"));
        assert!(src.contains("alpha"));
        assert!(src.contains("beta"));
    }

    #[test]
    fn hip_batched_gemm_batch_params() {
        let src = batched_gemm_hip(16);
        assert!(src.contains("hipBlockIdx_z"));
        assert!(src.contains("batch_count"));
        assert!(src.contains("batch_index"));
        assert!(src.contains("stride_a"));
        assert!(src.contains("stride_b"));
        assert!(src.contains("stride_c"));
    }

    #[test]
    fn hip_batched_gemm_tile_size_embedded() {
        let src8 = batched_gemm_hip(8);
        assert!(src8.contains("hipBlockIdx_y * 8"));
        let src32 = batched_gemm_hip(32);
        assert!(src32.contains("hipBlockIdx_y * 32"));
    }

    // ── Elementwise ───────────────────────────────────────────────────────

    #[test]
    fn hip_elementwise_relu() {
        let src = elementwise_hip("relu");
        assert!(src.contains("fmaxf(x, 0.0f)"));
        assert!(src.contains("__global__"));
        assert!(src.contains("elementwise_f32"));
    }

    #[test]
    fn hip_elementwise_all_ops() {
        assert!(elementwise_hip("sigmoid").contains("expf(-x)"));
        assert!(elementwise_hip("tanh").contains("tanhf(x)"));
        assert!(elementwise_hip("exp").contains("expf(x)"));
        assert!(elementwise_hip("log").contains("logf(x)"));
        assert!(elementwise_hip("sqrt").contains("sqrtf(x)"));
        assert!(elementwise_hip("abs").contains("fabsf(x)"));
        assert!(elementwise_hip("neg").contains("-x"));
        // Unknown op is identity.
        let unknown = elementwise_hip("unknown_op");
        assert!(unknown.contains("output[gid] = x;"));
    }

    // ── Binary ────────────────────────────────────────────────────────────

    #[test]
    fn hip_binary_add() {
        let src = binary_hip("add");
        assert!(src.contains("a[tid] + b[tid]"));
        assert!(src.contains("__global__"));
        assert!(src.contains("binary_f32"));
    }

    #[test]
    fn hip_binary_all_ops() {
        assert!(binary_hip("sub").contains("a[tid] - b[tid]"));
        assert!(binary_hip("mul").contains("a[tid] * b[tid]"));
        assert!(binary_hip("div").contains("a[tid] / b[tid]"));
        assert!(binary_hip("max").contains("fmaxf(a[tid], b[tid])"));
        assert!(binary_hip("min").contains("fminf(a[tid], b[tid])"));
        // Unknown op is identity on a.
        let unknown = binary_hip("unknown_op");
        assert!(unknown.contains("out[tid] = a[tid];"));
    }

    // ── Reduction ─────────────────────────────────────────────────────────

    #[test]
    fn hip_reduction_sum() {
        let src = reduction_hip("sum");
        assert!(src.contains("reduce_sum_f32"));
        assert!(src.contains("__syncthreads"));
        assert!(src.contains("a + b"));
    }

    #[test]
    fn hip_reduction_max() {
        let src = reduction_hip("max");
        assert!(src.contains("reduce_max_f32"));
        assert!(src.contains("fmaxf(a, b)"));
        assert!(src.contains("-HUGE_VALF"));
    }

    #[test]
    fn hip_reduction_min() {
        let src = reduction_hip("min");
        assert!(src.contains("reduce_min_f32"));
        assert!(src.contains("fminf(a, b)"));
        assert!(src.contains("HUGE_VALF"));
    }

    #[test]
    fn hip_reduction_mean() {
        let src = reduction_hip("mean");
        assert!(src.contains("reduce_mean_f32"));
        assert!(src.contains("(float)reduce_size"));
    }

    #[test]
    fn hip_reduction_unknown_empty() {
        assert!(reduction_hip("unknown_op").is_empty());
    }

    #[test]
    fn hip_reduction_function_names() {
        assert_eq!(reduction_function_name("sum"), "reduce_sum_f32");
        assert_eq!(reduction_function_name("max"), "reduce_max_f32");
        assert_eq!(reduction_function_name("min"), "reduce_min_f32");
        assert_eq!(reduction_function_name("mean"), "reduce_mean_f32");
        assert_eq!(reduction_function_name("other"), "reduce_sum_f32");
    }

    // ── Attention ─────────────────────────────────────────────────────────

    #[test]
    fn hip_attention_source() {
        let src = attention_hip();
        assert!(src.contains("attention_f32"));
        assert!(src.contains("__global__"));
        assert!(src.contains("scale"));
        assert!(src.contains("causal"));
        assert!(src.contains("expf"));
    }

    // ── Conv2d ────────────────────────────────────────────────────────────

    #[test]
    fn hip_conv2d_source() {
        let src = conv2d_forward_hip();
        assert!(src.contains("conv2d_forward_f32"));
        assert!(src.contains("__global__"));
        assert!(src.contains("stride_h"));
        assert!(src.contains("pad_h"));
    }

    // ── FP16 GEMM ────────────────────────────────────────────────────────

    #[test]
    fn hip_gemm_f16_contains_half() {
        let src = gemm_hip_f16(16);
        assert!(src.contains("__half"));
        assert!(src.contains("__half2float"));
        assert!(src.contains("__float2half"));
        assert!(src.contains("__global__"));
        assert!(src.contains("gemm_f16"));
    }

    #[test]
    fn hip_gemm_f16_tile_size_embedded() {
        let src8 = gemm_hip_f16(8);
        assert!(src8.contains("hipBlockIdx_y * 8"));
        let src32 = gemm_hip_f16(32);
        assert!(src32.contains("hipBlockIdx_y * 32"));
    }

    #[test]
    fn hip_gemm_f16_float_accumulation() {
        let src = gemm_hip_f16(16);
        assert!(src.contains("float acc = 0.0f"));
        assert!(src.contains("float alpha"));
        assert!(src.contains("float beta"));
    }

    // ── BF16 GEMM ────────────────────────────────────────────────────────

    #[test]
    fn hip_gemm_bf16_contains_bfloat16() {
        let src = gemm_hip_bf16(16);
        assert!(src.contains("hip_bfloat16"));
        assert!(src.contains("__global__"));
        assert!(src.contains("gemm_bf16"));
    }

    #[test]
    fn hip_gemm_bf16_tile_size_embedded() {
        let src8 = gemm_hip_bf16(8);
        assert!(src8.contains("hipBlockIdx_y * 8"));
        let src32 = gemm_hip_bf16(32);
        assert!(src32.contains("hipBlockIdx_y * 32"));
    }

    #[test]
    fn hip_gemm_bf16_float_accumulation() {
        let src = gemm_hip_bf16(16);
        assert!(src.contains("float acc = 0.0f"));
        assert!(src.contains("float alpha"));
        assert!(src.contains("float beta"));
    }

    // ── FP16 Elementwise ─────────────────────────────────────────────────

    #[test]
    fn hip_elementwise_f16_relu() {
        let src = elementwise_hip_f16("relu");
        assert!(src.contains("fmaxf(x, 0.0f)"));
        assert!(src.contains("__global__"));
        assert!(src.contains("elementwise_f16"));
        assert!(src.contains("__half"));
        assert!(src.contains("__half2float"));
        assert!(src.contains("__float2half"));
    }

    #[test]
    fn hip_elementwise_f16_all_ops() {
        assert!(elementwise_hip_f16("sigmoid").contains("expf(-x)"));
        assert!(elementwise_hip_f16("tanh").contains("tanhf(x)"));
        assert!(elementwise_hip_f16("exp").contains("expf(x)"));
        assert!(elementwise_hip_f16("log").contains("logf(x)"));
        assert!(elementwise_hip_f16("sqrt").contains("sqrtf(x)"));
        assert!(elementwise_hip_f16("abs").contains("fabsf(x)"));
        assert!(elementwise_hip_f16("neg").contains("-x"));
        // Unknown op is identity.
        let unknown = elementwise_hip_f16("unknown_op");
        assert!(unknown.contains("__float2half(x)"));
    }
}
