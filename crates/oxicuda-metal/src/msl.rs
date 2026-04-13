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

// ─── Reduction ────────────────────────────────────────────────────────────────

/// MSL source for a simple parallel reduction kernel.
///
/// The `"sum"` op uses `atomic_float` (Metal 3.0+, Apple Silicon required).
/// For other ops an empty string is returned — the caller should fall back.
///
/// # Note on `atomic_float`
/// `atomic_float` with `atomic_fetch_add_explicit` requires Metal 3.0+.
/// On Intel Macs (Metal 2) this kernel will fail to compile; callers should
/// check `device.supports_family(MTLGPUFamily::Metal3)` before use.
pub fn reduction_msl(op: &str) -> &'static str {
    match op {
        "sum" => {
            r#"
#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum_f32(
    device const float* input  [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    constant uint&       count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    atomic_fetch_add_explicit(output, input[gid], memory_order_relaxed);
}
"#
        }
        _ => "",
    }
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
    fn msl_reduction_sum_contains_atomic() {
        let src = reduction_msl("sum");
        assert!(src.contains("atomic_fetch_add_explicit"));
        assert!(src.contains("reduce_sum_f32"));
    }

    #[test]
    fn msl_reduction_unknown_empty() {
        assert!(reduction_msl("max").is_empty());
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
}
