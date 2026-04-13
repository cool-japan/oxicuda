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
}
