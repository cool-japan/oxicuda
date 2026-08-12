//! MSL (Metal Shading Language) kernel source generation.
//!
//! Each function in this module returns a `&'static str` or `String` containing
//! a complete MSL translation unit.
//!
//! # Two generations of generator
//!
//! * **v1** — [`gemm_msl`], [`batched_gemm_msl`], [`gemm_msl_f16`],
//!   [`elementwise_msl`], [`binary_msl`], [`reduction_msl`], [`conv2d_msl`],
//!   [`attention_msl`].  Naive one-output-per-thread reference kernels whose
//!   shapes are either pinned to the natural contiguous layout or baked into
//!   the source as `constant` globals.  Their kernel names, buffer bindings and
//!   parameter-struct layouts are **frozen** because the backend dispatchers
//!   bind them by name and by byte layout.
//! * **v2** — [`gemm_msl_v2`], [`batched_gemm_msl_v2`], [`elementwise_msl_v2`],
//!   [`binary_msl_v2`], plus [`crate::msl_nn::attention_msl_v2`] and
//!   [`crate::msl_nn::simdgroup_gemm_msl_v2`].  Every shape, stride and
//!   transpose flag travels in a runtime parameter buffer, so a **single**
//!   compiled pipeline serves every shape (no per-shape pipeline-cache
//!   explosion), padded leading dimensions and transposes are honoured instead
//!   of being silently ignored, and the GEMM inner loop is staged through
//!   threadgroup memory with register blocking.
//!
//! The v2 generators are the ones a new dispatcher should use; the v1 ones are
//! retained unchanged for ABI stability.

mod gemm_v2;

pub use gemm_v2::{
    BATCHED_GEMM_PARAMS_V2_BYTES, GEMM_PARAMS_V2_BYTES, GEMM_V2_THREADS_X, GEMM_V2_THREADS_Y,
    GEMM_V2_TILE_K, GEMM_V2_TILE_M, GEMM_V2_TILE_N, GemmDtype, batched_gemm_msl_v2,
    batched_gemm_v2_function_name, gemm_msl_v2, gemm_v2_function_name, validate_gemm_v2_dispatch,
};

use crate::error::{MetalError, MetalResult};

// ─── Floating-point math mode ─────────────────────────────────────────────────

/// The source prelude emitted for [`MslMathMode::Precise`].
const PRECISE_MATH_PRELUDE: &str = r#"// oxicuda: strict-IEEE kernel — must NOT be compiled with fast math.
//
// `#pragma METAL fp math_mode(safe)` is honoured by the Metal 3.2 compiler
// (macOS 15 / iOS 18) and newer.  Older Metal toolchains do not know the pragma
// and merely emit a `-Wunknown-pragmas` warning, so on those stacks the host
// must additionally clear `MTLCompileOptions.fastMathEnabled`
// (`metal::CompileOptions::set_fast_math_enabled(false)`), and kernels
// generated with a `MslMathMode` argument additionally call the
// `metal::precise::` intrinsics, which are honoured all the way back to MSL 1.0.
#pragma METAL fp math_mode(safe)
"#;

/// Floating-point math mode requested from the MSL compiler.
///
/// # Default behaviour, stated honestly
///
/// Every kernel in this crate is compiled through `metal::CompileOptions::new()`
/// with no math configuration, and Metal's default is **fast** math: the
/// compiler may reassociate floating-point expressions, contract `a*b+c` into an
/// FMA, assume all operands are finite, and substitute low-precision
/// approximations for `exp` / `log` / `sqrt` / division.  For bandwidth-bound
/// element-wise work that is exactly what you want, and it stays the default —
/// [`MslMathMode::Fast`] emits no prelude at all, so v1 sources are unchanged
/// byte-for-byte.
///
/// It is *not* what you want for:
///
/// * **Compensated (Dekker/Knuth) arithmetic** — [`crate::msl_nn::gemm_msl_f64_ds`]
///   computes `err = (a - (s - bb)) + (b - bb)`, which is algebraically zero and
///   is exactly what a reassociating compiler folds away, silently collapsing
///   the `df64` emulation back to plain `float`.  `two_prod`'s
///   `p = a*b; fma(a, b, -p)` likewise depends on `p` not being contracted.
/// * **`±INFINITY` reduction identities** — [`reduction_msl`] seeds `max` with
///   `-INFINITY` and `min` with `+INFINITY`, and
///   [`crate::msl_nn::softmax_msl`] seeds its row maximum with `-INFINITY`.
///   Under a finite-math-only assumption neither those constants nor
///   comparisons against them are guaranteed.
/// * **Stable softmax / layer normalisation / attention** — these rely on
///   `exp(x - max) <= 1` and on the mean/variance sums not being reassociated.
///
/// # This is measured, not assumed
///
/// `precise_math_mode_preserves_compensated_arithmetic_on_device` runs the
/// `two_sum` error term and the reassociation canary `(x + y - x) - y` on real
/// hardware with runtime operands. Under `CompileOptions::new()` — the options
/// every path in this crate uses — both collapse to `0.0`, confirming that the
/// `df64` emulation really is silently defeated today. Adding the
/// [`MslMathMode::Precise`] prelude restores both **without** the host having to
/// touch `fastMathEnabled`, so the pragma is doing real work on this toolchain.
/// `df64_gemm_keeps_its_low_limb_under_precise_math` confirms the same end to
/// end on the actual `gemm_f64_ds` kernel.
///
/// # Usage
///
/// ```
/// use oxicuda_metal::msl::{reduction_msl, with_math_mode, MslMathMode};
///
/// let strict = with_math_mode(&reduction_msl("max"), MslMathMode::Precise);
/// assert!(strict.contains("math_mode(safe)"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MslMathMode {
    /// Metal's default: fast math is permitted (reassociation, contraction,
    /// finite-math-only, approximate transcendentals).
    #[default]
    Fast,
    /// Strict IEEE-754 semantics are requested for the whole translation unit.
    Precise,
}

impl MslMathMode {
    /// Source prelude that requests this mode, prepended ahead of `#include`.
    ///
    /// Empty for [`MslMathMode::Fast`].
    pub fn prelude(self) -> &'static str {
        match self {
            Self::Fast => "",
            Self::Precise => PRECISE_MATH_PRELUDE,
        }
    }

    /// Whether this mode asks for strict IEEE semantics.
    pub fn is_precise(self) -> bool {
        matches!(self, Self::Precise)
    }

    /// Namespace qualifier for transcendental intrinsics: `""` or `"precise::"`.
    ///
    /// Mode-aware generators splice this in front of `exp`, `sqrt`, `rsqrt` and
    /// friends so that strictness survives even on a toolchain that ignores the
    /// [`prelude`](Self::prelude) pragma.
    pub fn intrinsic_prefix(self) -> &'static str {
        match self {
            Self::Fast => "",
            Self::Precise => "precise::",
        }
    }
}

/// Prepend the [`MslMathMode`] prelude to an already-generated MSL source.
///
/// This is the retrofit path for the frozen v1 generators, whose bodies cannot
/// change: `with_math_mode(&reduction_msl("max"), MslMathMode::Precise)` yields
/// the same kernel compiled under strict IEEE rules.  The returned string is
/// still a complete, self-contained translation unit.
pub fn with_math_mode(source: &str, mode: MslMathMode) -> String {
    format!("{}{source}", mode.prelude())
}

// ─── Dispatch-side validation helpers ─────────────────────────────────────────

/// Maximum threadgroup width any Apple GPU accepts.
const MAX_THREADGROUP_THREADS: usize = 1024;

/// Validate the threadgroup width a dispatcher intends to use for the
/// tree-reduction kernels ([`reduction_msl`], [`crate::msl_nn::softmax_msl`],
/// [`crate::msl_nn::layernorm_msl`]).
///
/// The tree reductions in those kernels were hardened to be correct for **any**
/// width (they walk `n_active -> ceil(n_active/2)` instead of halving a
/// power-of-two count), so a ragged width is no longer silently wrong.  This
/// validator still rejects one, because a non-power-of-two width wastes SIMD
/// lanes and is never what a dispatcher actually wants; it exists so a
/// dispatcher can enforce the documented contract before encoding.
///
/// Accepts `1..=1024` powers of two.
pub fn validate_threadgroup_size(threads_per_threadgroup: usize) -> MetalResult<()> {
    if threads_per_threadgroup == 0 {
        return Err(MetalError::InvalidArgument(
            "threadgroup width must be non-zero".into(),
        ));
    }
    if threads_per_threadgroup > MAX_THREADGROUP_THREADS {
        return Err(MetalError::InvalidArgument(format!(
            "threadgroup width {threads_per_threadgroup} exceeds the Metal maximum of {MAX_THREADGROUP_THREADS}"
        )));
    }
    if !threads_per_threadgroup.is_power_of_two() {
        return Err(MetalError::InvalidArgument(format!(
            "threadgroup width {threads_per_threadgroup} must be a power of two"
        )));
    }
    Ok(())
}

/// Format an `f32` as an MSL `float` literal, rejecting values that have no
/// literal spelling.
///
/// Rust's `Display` renders `f32::NAN` as `NaN` and `f32::INFINITY` as `inf`,
/// neither of which is valid MSL — a generator that embeds them produces source
/// that fails at *shader-compile* time with an opaque message instead of
/// failing fast with a clear argument error.  This helper rejects them up front
/// and emits a full round-trip-precision literal with an explicit `f` suffix
/// for everything else.
pub fn msl_float_literal(value: f32) -> MetalResult<String> {
    if !value.is_finite() {
        return Err(MetalError::InvalidArgument(format!(
            "cannot embed the non-finite value {value} as an MSL float literal"
        )));
    }
    Ok(format!("{value:?}f"))
}

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
    // BLAS contract: when beta==0 the C operand is not referenced, so a fresh
    // (possibly NaN/Inf) output buffer never poisons the result via 0*NaN.
    float prev = (params.beta == 0.0f) ? 0.0f : params.beta * c[out_idx];
    c[out_idx] = params.alpha * acc + prev;
}
"#
}

// ─── Batched GEMM ────────────────────────────────────────────────────────────

/// MSL source for a single-precision batched GEMM kernel.
///
/// For each batch `b` in `0..batch_count`:
///   `C_b = alpha * A_b * B_b + beta * C_b`
///
/// where `A_b` starts at offset `b * stride_a`, etc.
/// The batch index is derived from `threadgroup_position_in_grid.z`.
pub fn batched_gemm_msl() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

struct BatchedGemmParams {
    uint m;
    uint n;
    uint k;
    float alpha;
    float beta;
    uint batch_count;
    uint stride_a;
    uint stride_b;
    uint stride_c;
};

kernel void batched_gemm_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    constant BatchedGemmParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint batch = tgid.z;
    if (row >= params.m || col >= params.n || batch >= params.batch_count) return;

    uint a_off = batch * params.stride_a;
    uint b_off = batch * params.stride_b;
    uint c_off = batch * params.stride_c;

    float acc = 0.0f;
    for (uint i = 0; i < params.k; i++) {
        acc += a[a_off + row * params.k + i] * b[b_off + i * params.n + col];
    }
    uint out_idx = c_off + row * params.n + col;
    // BLAS contract: beta==0 ⇒ C not referenced (avoids 0*NaN poisoning).
    float prev = (params.beta == 0.0f) ? 0.0f : params.beta * c[out_idx];
    c[out_idx] = params.alpha * acc + prev;
}
"#
}

// ─── FP16 GEMM ──────────────────────────────────────────────────────────────

/// MSL source for a half-precision GEMM kernel (`C = alpha*A*B + beta*C`).
///
/// Uses Metal `half` type for storage buffers but keeps accumulation in `float`
/// for numerical precision.
pub fn gemm_msl_f16() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

struct GemmParamsF16 {
    uint m;
    uint n;
    uint k;
    float alpha;
    float beta;
};

kernel void gemm_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c       [[buffer(2)]],
    constant GemmParamsF16& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.m || col >= params.n) return;

    float acc = 0.0f;
    for (uint i = 0; i < params.k; i++) {
        acc += float(a[row * params.k + i]) * float(b[i * params.n + col]);
    }
    uint out_idx = row * params.n + col;
    // BLAS contract: beta==0 ⇒ C not referenced (avoids 0*NaN poisoning).
    float prev = (params.beta == 0.0f) ? 0.0f : params.beta * float(c[out_idx]);
    c[out_idx] = half(params.alpha * acc + prev);
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
/// `"sqrt"`, `"abs"`, `"neg"`, `"gelu"`, `"silu"` — see [`UNARY_OPS`].
///
/// # Unknown ops
///
/// An unrecognised `op` falls through to the **identity** kernel. That is a
/// silent failure mode (a typo becomes a no-op copy rather than an error) and it
/// is retained here only because the behaviour is frozen; use
/// [`elementwise_msl_v2`], which returns
/// [`MetalError::Unsupported`](crate::error::MetalError::Unsupported) instead.
pub fn elementwise_msl(op: &str) -> String {
    let op_expr = unary_op_expr(op).unwrap_or("x"); // identity fallback (frozen)
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
/// `"min"`, `"pow"` — see [`BINARY_OPS`].
///
/// # Unknown ops
///
/// An unrecognised `op` falls back to copying `a`, which is a silent failure
/// mode retained for ABI stability; use [`binary_msl_v2`], which errors.
pub fn binary_msl(op: &str) -> String {
    let op_expr = binary_op_expr(op).unwrap_or("a[tid]"); // identity (frozen)
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

// ─── Element-wise v2 (erroring on unknown ops) ────────────────────────────────

/// Every unary op [`elementwise_msl_v2`] accepts.
pub const UNARY_OPS: &[&str] = &[
    "relu", "sigmoid", "tanh", "exp", "log", "sqrt", "abs", "neg", "gelu", "silu",
];

/// Every binary op [`binary_msl_v2`] accepts.
pub const BINARY_OPS: &[&str] = &["add", "sub", "mul", "div", "max", "min", "pow"];

/// MSL expression computing the unary `op` on the local `float x`.
fn unary_op_expr(op: &str) -> Option<&'static str> {
    Some(match op {
        "relu" => "max(x, 0.0f)",
        "sigmoid" => "1.0f / (1.0f + exp(-x))",
        "tanh" => "tanh(x)",
        "exp" => "exp(x)",
        "log" => "log(x)",
        "sqrt" => "sqrt(x)",
        "abs" => "abs(x)",
        "neg" => "-x",
        // Hendrycks-Gimpel tanh approximation, the formulation PyTorch uses for
        // `gelu(approximate="tanh")`: 0.7978845608 = sqrt(2/pi).
        "gelu" => "0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)))",
        // SiLU / swish-1: x * sigmoid(x).
        "silu" => "x / (1.0f + exp(-x))",
        _ => return None,
    })
}

/// MSL expression computing the binary `op` on `a[tid]` and `b[tid]`.
fn binary_op_expr(op: &str) -> Option<&'static str> {
    Some(match op {
        "add" => "a[tid] + b[tid]",
        "sub" => "a[tid] - b[tid]",
        "mul" => "a[tid] * b[tid]",
        "div" => "a[tid] / b[tid]",
        "max" => "max(a[tid], b[tid])",
        "min" => "min(a[tid], b[tid])",
        "pow" => "pow(a[tid], b[tid])",
        _ => return None,
    })
}

/// Element-wise unary kernel source, erroring on an unrecognised op.
///
/// Identical output to [`elementwise_msl`] for every op in [`UNARY_OPS`]; the
/// difference is the failure mode. [`elementwise_msl`] emits an identity kernel
/// for an unknown op, so a typo — or a newly added `UnaryOp` variant that nobody
/// wired into the string table — compiles, runs, and silently copies the input.
/// This variant returns
/// [`MetalError::Unsupported`](crate::error::MetalError::Unsupported) instead,
/// matching how [`reduction_msl`] already signals an unknown op.
///
/// `gelu` and `silu`, previously advertised but absent, are implemented here and
/// in [`elementwise_msl`].
///
/// # Callers
///
/// `MetalBackend::dispatch_unary` still calls [`elementwise_msl`]; switching it
/// over is a wave-2 change (it must map the `Err` onto
/// `BackendError::Unsupported`, exactly as `dispatch_reduce` already maps the
/// empty string returned by [`reduction_msl`]).
pub fn elementwise_msl_v2(op: &str) -> MetalResult<String> {
    if unary_op_expr(op).is_none() {
        return Err(MetalError::Unsupported(format!(
            "unary op `{op}` has no MSL implementation (supported: {})",
            UNARY_OPS.join(", ")
        )));
    }
    Ok(elementwise_msl(op))
}

/// Binary element-wise kernel source, erroring on an unrecognised op.
///
/// See [`elementwise_msl_v2`] for the rationale; this is the binary counterpart
/// of the same fix.
pub fn binary_msl_v2(op: &str) -> MetalResult<String> {
    if binary_op_expr(op).is_none() {
        return Err(MetalError::Unsupported(format!(
            "binary op `{op}` has no MSL implementation (supported: {})",
            BINARY_OPS.join(", ")
        )));
    }
    Ok(binary_msl(op))
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
///
/// # Dispatch contract
///
/// * `[[threadgroup(0)]]` must be given `threads_per_threadgroup * 4` bytes via
///   `set_threadgroup_memory_length(0, ..)`.
/// * One threadgroup per `(outer, inner)` slice:
///   `threadgroups = outer_size * inner_size`.
/// * `threads_per_threadgroup` should be a power of two in `1..=1024` — check it
///   with [`validate_threadgroup_size`]. The tree reduction below no longer
///   *requires* it: it walks `n_active -> ceil(n_active / 2)` with an
///   `lid + half < n_active` guard, which is correct for any width. (The
///   previous `for (s = tg_size/2; s > 0; s >>= 1)` form silently dropped
///   partials whenever `tg_size` was not a power of two — e.g. at `tg_size=100`
///   lane 24's partial was never folded in.)
/// * `reduce_size == 0` is rejected by the dispatcher; the `mean` kernel
///   additionally guards the division defensively and writes `0` rather than
///   `NaN` if it is ever reached with a zero extent.
///
/// # Math mode
///
/// `max`/`min` seed the tree with `-INFINITY` / `+INFINITY`, which Metal's
/// default fast math is not obliged to honour. Wrap the source with
/// [`with_math_mode`] (or use [`reduction_msl_with_mode`]) when the input may
/// contain infinities or when strict IEEE behaviour is required.
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

    // Defensive zero guard: a zero-length reduction axis must not write NaN.
    // The dispatcher is expected to reject `reduce_size == 0` up front; this is
    // the second line of defence so the kernel is safe on its own terms.
    let final_expr = if is_mean {
        "(float(reduce_size) > 0.0f) ? (sdata[0] / float(reduce_size)) : 0.0f"
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

    // Tree reduction that is correct for ANY threadgroup width, not just powers
    // of two: fold the upper ceil(n/2) lanes into the lower half each round.
    // The loop condition is uniform across the threadgroup, so the barrier stays
    // uniformly executed.
    for (uint n_active = tg_size; n_active > 1u; ) {{
        uint half_n = (n_active + 1u) / 2u;
        if (lid + half_n < n_active) {{
            sdata[lid] = reduce_fn(sdata[lid], sdata[lid + half_n]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        n_active = half_n;
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

/// [`reduction_msl`] with an explicit [`MslMathMode`] and a real error for an
/// unknown op.
///
/// `max`/`min` carry `±INFINITY` identities and `mean` performs a division, all
/// of which Metal's default fast math may reinterpret;
/// [`MslMathMode::Precise`] pins them down.
pub fn reduction_msl_with_mode(op: &str, mode: MslMathMode) -> MetalResult<String> {
    let src = reduction_msl(op);
    if src.is_empty() {
        return Err(MetalError::Unsupported(format!(
            "reduction op `{op}` has no MSL implementation (supported: sum, max, min, mean)"
        )));
    }
    Ok(with_math_mode(&src, mode))
}

/// [`reduction_msl`] generated for a specific threadgroup width, with the width
/// **validated at generation time**.
///
/// This is the entry point that enforces the dispatch contract rather than only
/// documenting it: it rejects an unknown op and any `threads_per_threadgroup`
/// that is not a power of two in `1..=1024` (see
/// [`validate_threadgroup_size`]), so a dispatcher cannot reach the kernel with
/// a width the contract does not cover.
///
/// The caller still has to bind `threads_per_threadgroup * 4` bytes of
/// threadgroup memory at slot 0 and launch exactly that width.
pub fn reduction_msl_for_threadgroup(
    op: &str,
    threads_per_threadgroup: usize,
    mode: MslMathMode,
) -> MetalResult<String> {
    validate_threadgroup_size(threads_per_threadgroup)?;
    reduction_msl_with_mode(op, mode)
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

/// Size in bytes of the `ConvParamsV2` constant buffer at `[[buffer(3)]]`.
pub const CONV_PARAMS_V2_BYTES: usize = 52;

/// MSL function name generated by [`conv2d_msl_v2`].
pub fn conv2d_v2_function_name() -> &'static str {
    "conv2d_forward_v2_f32"
}

/// Conv2D forward (NCHW) driven by a **runtime** parameter buffer.
///
/// Numerically identical to [`conv2d_msl`] — same accumulation order, same
/// signed padding guard — but every shape arrives at dispatch time instead of
/// being baked in as a `constant uint`. That is what makes it usable from a
/// bounded pipeline cache: [`conv2d_msl`] emits a distinct source (hence a
/// distinct `MTLLibrary` + PSO + cache entry) for **every** convolution shape,
/// so a model with varying batch or resolution evicts the cache continuously,
/// while this kernel compiles once per element type and serves all shapes.
///
/// # Parameter buffer `[[buffer(3)]]` — exact byte layout
///
/// Total 52 bytes ([`CONV_PARAMS_V2_BYTES`]), alignment 4, no padding — a
/// `#[repr(C)]` Rust struct of thirteen `u32` matches it exactly:
///
/// | offset | size | MSL              | meaning |
/// |-------:|-----:|------------------|---------|
/// | 0      | 4    | `uint n_batch`   | `N` of the `[N, C, H, W]` input |
/// | 4      | 4    | `uint c_in`      | input channels |
/// | 8      | 4    | `uint h_in`      | input height |
/// | 12     | 4    | `uint w_in`      | input width |
/// | 16     | 4    | `uint k_out`     | output channels (`K` of `[K, C, Fh, Fw]`) |
/// | 20     | 4    | `uint fh`        | filter height |
/// | 24     | 4    | `uint fw`        | filter width |
/// | 28     | 4    | `uint oh`        | output height |
/// | 32     | 4    | `uint ow`        | output width |
/// | 36     | 4    | `uint stride_h`  | vertical stride (must be non-zero) |
/// | 40     | 4    | `uint stride_w`  | horizontal stride (must be non-zero) |
/// | 44     | 4    | `uint pad_h`     | zero-padding rows on each side |
/// | 48     | 4    | `uint pad_w`     | zero-padding columns on each side |
///
/// # Buffer bindings
///
/// `[[buffer(0)]]` input (`N×C×H×W`), `[[buffer(1)]]` filter (`K×C×Fh×Fw`),
/// `[[buffer(2)]]` output (`N×K×Oh×Ow`) — all `float`, row-major NCHW/KCHW.
///
/// # Dispatch contract
///
/// One thread per output element: `total = n_batch * k_out * oh * ow`. The grid
/// is rounded up to whole threadgroups and the kernel bounds-checks `gid`, so
/// any threadgroup width in `1..=max_total_threads_per_threadgroup` is valid —
/// there is no tree reduction here, so the width need not be a power of two.
/// No threadgroup memory is used, so `set_threadgroup_memory_length` must not be
/// called.
///
/// # Preconditions the kernel cannot check
///
/// `stride_h`/`stride_w` must be non-zero (a zero stride makes every output
/// element read the same window and compiles cleanly), and `oh`/`ow` must be the
/// output extents the caller's own shape arithmetic produced — the kernel writes
/// exactly `n_batch * k_out * oh * ow` elements and reads whatever `oh`/`ow`
/// imply. Validate both on the host before dispatching.
///
/// # Not covered
///
/// No dilation, no grouped/depthwise convolution and no bias term, matching
/// [`conv2d_msl`] and the `ComputeBackend::conv2d_forward` signature, which
/// cannot express any of the three.
pub fn conv2d_msl_v2() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

struct ConvParamsV2 {
    uint n_batch;
    uint c_in;
    uint h_in;
    uint w_in;
    uint k_out;
    uint fh;
    uint fw;
    uint oh;
    uint ow;
    uint stride_h;
    uint stride_w;
    uint pad_h;
    uint pad_w;
};

kernel void conv2d_forward_v2_f32(
    device const float* input  [[buffer(0)]],
    device const float* filter [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant ConvParamsV2& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.n_batch * params.k_out * params.oh * params.ow;
    if (gid >= total) return;

    uint ox  = gid % params.ow;
    uint tmp = gid / params.ow;
    uint oy  = tmp % params.oh;
    tmp      = tmp / params.oh;
    uint kf  = tmp % params.k_out;
    uint b   = tmp / params.k_out;

    float acc = 0.0f;
    for (uint ci = 0u; ci < params.c_in; ci++) {
        for (uint fy = 0u; fy < params.fh; fy++) {
            for (uint fx = 0u; fx < params.fw; fx++) {
                int iy = int(oy * params.stride_h + fy) - int(params.pad_h);
                int ix = int(ox * params.stride_w + fx) - int(params.pad_w);
                if (iy >= 0 && uint(iy) < params.h_in && ix >= 0 && uint(ix) < params.w_in) {
                    uint in_idx = ((b * params.c_in + ci) * params.h_in + uint(iy)) * params.w_in + uint(ix);
                    uint f_idx  = ((kf * params.c_in + ci) * params.fh + fy) * params.fw + fx;
                    acc += input[in_idx] * filter[f_idx];
                }
            }
        }
    }
    output[gid] = acc;
}
"#
}

// ─── Attention ───────────────────────────────────────────────────────────────

/// MSL source for a single-precision scaled dot-product attention kernel.
///
/// Each thread handles one (batch_head, query_position) pair.
/// Uses numerically stable softmax (subtract max before exp).
/// Optional causal masking skips positions where `sk > sq`.
///
/// # Causal-mask alignment
///
/// The mask is **top-left** aligned (`masked ⟺ sk > sq`), matching
/// `oxicuda_backend`'s CPU oracle and `MetalBackend::attention`. See
/// [`crate::msl_nn::attention_msl_v2`] for the KV-cache caveat this implies.
///
/// # Validation
///
/// This entry point performs none: `scale` is embedded by value, so a NaN or
/// infinite `scale` used to emit source that could not compile. It now emits the
/// MSL `NAN` / `INFINITY` macros so the source is at least well-formed, but
/// [`attention_msl_checked`] rejects such arguments up front and is what a
/// dispatcher should call.
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
    // Never emit Rust's `NaN` / `inf` spellings — they are not valid MSL.
    let scale = msl_float_literal(scale).unwrap_or_else(|_| {
        if scale.is_nan() {
            "NAN".to_string()
        } else if scale.is_sign_positive() {
            "INFINITY".to_string()
        } else {
            "(-INFINITY)".to_string()
        }
    });
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

/// [`attention_msl`] with the argument validation the generator has always
/// lacked.
///
/// Rejects a non-finite or non-positive `scale` (`1/sqrt(head_dim)` is the usual
/// value, and a zero/negative scale is never meaningful) and any zero shape,
/// with [`MetalError::InvalidArgument`](crate::error::MetalError::InvalidArgument),
/// instead of deferring the failure to an opaque shader-compile error or
/// producing a kernel that returns immediately.
#[allow(clippy::too_many_arguments)]
pub fn attention_msl_checked(
    batch_heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> MetalResult<String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(MetalError::InvalidArgument(format!(
            "attention scale must be finite and positive, got {scale}"
        )));
    }
    for (name, value) in [
        ("batch_heads", batch_heads),
        ("seq_q", seq_q),
        ("seq_kv", seq_kv),
        ("head_dim", head_dim),
    ] {
        if value == 0 {
            return Err(MetalError::InvalidArgument(format!(
                "attention {name} must be non-zero"
            )));
        }
    }
    Ok(attention_msl(
        batch_heads,
        seq_q,
        seq_kv,
        head_dim,
        scale,
        causal,
    ))
}

/// [`conv2d_msl`] with argument validation.
///
/// Rejects any zero shape and a zero stride. A zero stride makes every output
/// element read the same input window, which compiles cleanly and produces
/// silently wrong results.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_msl_checked(
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
) -> MetalResult<String> {
    for (name, value) in [
        ("n", n),
        ("c_in", c_in),
        ("h_in", h_in),
        ("w_in", w_in),
        ("k_out", k_out),
        ("fh", fh),
        ("fw", fw),
        ("oh", oh),
        ("ow", ow),
        ("stride_h", stride_h),
        ("stride_w", stride_w),
    ] {
        if value == 0 {
            return Err(MetalError::InvalidArgument(format!(
                "conv2d {name} must be non-zero"
            )));
        }
    }
    Ok(conv2d_msl(
        n, c_in, h_in, w_in, k_out, fh, fw, oh, ow, stride_h, stride_w, pad_h, pad_w,
    ))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Math mode ─────────────────────────────────────────────────────────────

    #[test]
    fn math_mode_fast_is_the_default_and_emits_nothing() {
        assert_eq!(MslMathMode::default(), MslMathMode::Fast);
        assert_eq!(MslMathMode::Fast.prelude(), "");
        assert_eq!(MslMathMode::Fast.intrinsic_prefix(), "");
        assert!(!MslMathMode::Fast.is_precise());
        // Wrapping in Fast mode must not perturb a frozen v1 source.
        let src = reduction_msl("sum");
        assert_eq!(with_math_mode(&src, MslMathMode::Fast), src);
    }

    #[test]
    fn math_mode_precise_emits_the_pragma_ahead_of_the_include() {
        let prelude = MslMathMode::Precise.prelude();
        assert!(prelude.contains("#pragma METAL fp math_mode(safe)"));
        assert!(MslMathMode::Precise.is_precise());
        assert_eq!(MslMathMode::Precise.intrinsic_prefix(), "precise::");

        let strict = with_math_mode(&reduction_msl("max"), MslMathMode::Precise);
        let pragma = strict
            .find("#pragma METAL fp math_mode(safe)")
            .expect("pragma present");
        let include = strict
            .find("#include <metal_stdlib>")
            .expect("include present");
        assert!(pragma < include, "the pragma must precede the include");
    }

    #[test]
    fn reduction_with_mode_wraps_and_rejects_unknown_ops() {
        let strict = reduction_msl_with_mode("mean", MslMathMode::Precise).expect("mean");
        assert!(strict.contains("math_mode(safe)"));
        assert!(strict.contains("reduce_mean_f32"));
        assert_eq!(
            reduction_msl_with_mode("sum", MslMathMode::Fast).expect("sum"),
            reduction_msl("sum")
        );
        assert!(reduction_msl_with_mode("nope", MslMathMode::Fast).is_err());
    }

    /// Prove on real hardware that [`MslMathMode::Precise`] actually changes
    /// codegen, rather than merely being an accepted-but-ignored pragma.
    ///
    /// The kernel evaluates Knuth's `two_sum` error term and the classic
    /// reassociation canary `(x + y - x) - y` on **runtime** operands (compile-
    /// time constants would be folded by the host compiler under strict rules
    /// and would prove nothing). Under Metal's default fast math both collapse
    /// to zero; under the strict-IEEE pragma both survive.
    #[cfg(target_os = "macos")]
    #[test]
    fn precise_math_mode_preserves_compensated_arithmetic_on_device() {
        use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
        let Some(device) = Device::system_default() else {
            return;
        };
        let queue = device.new_command_queue();
        let body = r#"
#include <metal_stdlib>
using namespace metal;
kernel void math_mode_canary(
    device float*       out [[buffer(0)]],
    device const float* in  [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = in[0];
    float y = in[1];
    float s = x + y;
    out[0] = (s - x) - y;               // strict: -1.0, reassociated: 0.0
    float a = in[1];
    float b = in[2];
    float t = a + b;
    float bb = t - a;
    out[1] = (a - (t - bb)) + (b - bb); // Knuth two_sum error term
}
"#;
        let run = |src: &str| -> Option<[f32; 2]> {
            let lib = device
                .new_library_with_source(src, &CompileOptions::new())
                .ok()?;
            let func = lib.get_function("math_mode_canary", None).ok()?;
            let pso = device
                .new_compute_pipeline_state_with_function(&func)
                .ok()?;
            let input: [f32; 3] = [1.0e8, 1.0, 1.0e-9];
            let buf_out = device.new_buffer(8, MTLResourceOptions::StorageModeShared);
            let buf_in = device.new_buffer_with_data(
                input.as_ptr() as *const std::ffi::c_void,
                12,
                MTLResourceOptions::StorageModeShared,
            );
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&buf_out), 0);
            enc.set_buffer(1, Some(&buf_in), 0);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            // SAFETY: shared storage, GPU work complete, two floats allocated.
            let v = unsafe { std::slice::from_raw_parts(buf_out.contents() as *const f32, 2) };
            Some([v[0], v[1]])
        };

        let Some(precise) = run(&with_math_mode(body, MslMathMode::Precise)) else {
            return;
        };
        // Strict IEEE in f32: 1e8 + 1 rounds back to 1e8 (ulp there is 8), so
        // (s - x) - y == -1.0; and two_sum's error term is exactly 1e-9.
        assert_eq!(
            precise[0], -1.0,
            "MslMathMode::Precise did not stop reassociation"
        );
        assert!(
            precise[1] != 0.0,
            "MslMathMode::Precise did not preserve the two_sum error term"
        );

        // Informational counterpart: on the toolchain this was written against,
        // the same source compiled without the pragma yields [0.0, 0.0] — the
        // df64 emulation really is silently collapsed by default. Not asserted,
        // because a future toolchain tightening its default would be an
        // improvement, not a regression.
        if let Some(fast) = run(body) {
            assert!(
                fast[0] == 0.0 || fast[0] == -1.0,
                "unexpected fast-math canary value {}",
                fast[0]
            );
        }
    }

    // ── Validation helpers ────────────────────────────────────────────────────

    #[test]
    fn threadgroup_size_validator_enforces_the_dispatch_contract() {
        for good in [1usize, 2, 32, 256, 1024] {
            assert!(validate_threadgroup_size(good).is_ok(), "{good} rejected");
        }
        for bad in [0usize, 3, 100, 255, 2048] {
            assert!(validate_threadgroup_size(bad).is_err(), "{bad} accepted");
        }
    }

    #[test]
    fn reduction_generator_rejects_a_bad_threadgroup_width() {
        assert!(reduction_msl_for_threadgroup("sum", 256, MslMathMode::Fast).is_ok());
        assert!(reduction_msl_for_threadgroup("sum", 100, MslMathMode::Fast).is_err());
        assert!(reduction_msl_for_threadgroup("sum", 0, MslMathMode::Fast).is_err());
        assert!(reduction_msl_for_threadgroup("sum", 2048, MslMathMode::Fast).is_err());
        // The op is validated too.
        assert!(reduction_msl_for_threadgroup("nope", 256, MslMathMode::Fast).is_err());
        // And the math mode is threaded through.
        let strict =
            reduction_msl_for_threadgroup("max", 64, MslMathMode::Precise).expect("valid request");
        assert!(strict.contains("math_mode(safe)"));
    }

    #[test]
    fn float_literal_rejects_non_finite_and_keeps_the_f_suffix() {
        assert_eq!(msl_float_literal(1.0).expect("finite"), "1.0f");
        assert_eq!(msl_float_literal(0.125).expect("finite"), "0.125f");
        assert_eq!(msl_float_literal(-0.5).expect("finite"), "-0.5f");
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(msl_float_literal(bad).is_err(), "{bad} accepted");
        }
    }

    // ── Element-wise v2 ───────────────────────────────────────────────────────

    #[test]
    fn elementwise_v2_errors_on_unknown_ops_instead_of_emitting_identity() {
        for op in UNARY_OPS {
            let src = elementwise_msl_v2(op).unwrap_or_else(|e| panic!("{op}: {e}"));
            assert!(src.contains("elementwise_f32"));
            assert!(
                !src.contains("output[gid] = x;"),
                "{op} degraded to identity"
            );
        }
        let err = elementwise_msl_v2("wat").expect_err("unknown op must error");
        assert!(err.to_string().contains("wat"));
        // The frozen v1 generator still emits identity, on purpose.
        assert!(elementwise_msl("wat").contains("output[gid] = x;"));
    }

    #[test]
    fn elementwise_gains_gelu_and_silu() {
        let gelu = elementwise_msl("gelu");
        assert!(gelu.contains("0.7978845608028654f"));
        assert!(gelu.contains("0.044715f"));
        let silu = elementwise_msl("silu");
        assert!(silu.contains("x / (1.0f + exp(-x))"));
        assert!(UNARY_OPS.contains(&"gelu") && UNARY_OPS.contains(&"silu"));
    }

    #[test]
    fn binary_v2_errors_on_unknown_ops() {
        for op in BINARY_OPS {
            let src = binary_msl_v2(op).unwrap_or_else(|e| panic!("{op}: {e}"));
            assert!(src.contains("binary_f32"));
            assert!(
                !src.contains("out[tid] = a[tid];"),
                "{op} degraded to identity"
            );
        }
        assert!(binary_msl_v2("nope").is_err());
        assert!(binary_msl("nope").contains("out[tid] = a[tid];"));
    }

    // ── Reduction hardening ───────────────────────────────────────────────────

    #[test]
    fn reduction_tree_is_correct_for_non_power_of_two_widths() {
        for op in ["sum", "max", "min", "mean"] {
            let src = reduction_msl(op);
            assert!(
                src.contains("for (uint n_active = tg_size; n_active > 1u; )"),
                "{op} still uses the power-of-two-only halving loop"
            );
            assert!(src.contains("if (lid + half_n < n_active)"));
            assert!(
                !src.contains("s >>= 1"),
                "{op} still contains the lossy halving loop"
            );
        }
    }

    #[test]
    fn reduction_mean_guards_a_zero_length_axis() {
        let src = reduction_msl("mean");
        assert!(src.contains("(float(reduce_size) > 0.0f) ?"));
        assert!(src.contains(": 0.0f"));
    }

    // ── Validated generators ──────────────────────────────────────────────────

    #[test]
    fn attention_checked_rejects_bad_scale_and_zero_shapes() {
        assert!(attention_msl_checked(1, 4, 4, 8, 0.35, false).is_ok());
        for bad_scale in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
            assert!(
                attention_msl_checked(1, 4, 4, 8, bad_scale, false).is_err(),
                "scale {bad_scale} accepted"
            );
        }
        assert!(attention_msl_checked(0, 4, 4, 8, 0.5, false).is_err());
        assert!(attention_msl_checked(1, 0, 4, 8, 0.5, false).is_err());
        assert!(attention_msl_checked(1, 4, 0, 8, 0.5, false).is_err());
        assert!(attention_msl_checked(1, 4, 4, 0, 0.5, false).is_err());
    }

    #[test]
    fn attention_never_emits_rust_nan_or_inf_spellings() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let src = attention_msl(1, 2, 2, 4, bad, false);
            assert!(!src.contains("= NaN;"), "Rust NaN spelling leaked: {src}");
            assert!(!src.contains("= inf;"), "Rust inf spelling leaked: {src}");
            assert!(!src.contains("= -inf;"));
        }
        // Finite scales keep an explicit float suffix.
        assert!(attention_msl(1, 2, 2, 4, 0.125, false).contains("SCALE      = 0.125f;"));
    }

    #[test]
    fn conv2d_checked_rejects_zero_shapes_and_zero_stride() {
        assert!(conv2d_msl_checked(1, 3, 8, 8, 16, 3, 3, 6, 6, 1, 1, 0, 0).is_ok());
        assert!(conv2d_msl_checked(0, 3, 8, 8, 16, 3, 3, 6, 6, 1, 1, 0, 0).is_err());
        assert!(conv2d_msl_checked(1, 3, 8, 8, 16, 3, 3, 6, 6, 0, 1, 0, 0).is_err());
        assert!(conv2d_msl_checked(1, 3, 8, 8, 16, 3, 3, 6, 6, 1, 0, 0, 0).is_err());
        // Zero padding is legitimate.
        assert!(conv2d_msl_checked(1, 1, 4, 4, 1, 3, 3, 4, 4, 1, 1, 1, 1).is_ok());
    }

    #[test]
    fn msl_gemm_contains_kernel_name() {
        let src = gemm_msl();
        assert!(src.contains("gemm_f32"));
        assert!(src.contains("GemmParams"));
        assert!(src.contains("metal_stdlib"));
    }

    #[test]
    fn msl_gemm_guards_c_read_on_beta_zero() {
        // All three GEMM kernels must guard the C read on beta==0 so a fresh
        // (uninitialised, possibly NaN) output buffer is not poisoned by 0*NaN.
        for src in [gemm_msl(), batched_gemm_msl(), gemm_msl_f16()] {
            assert!(
                src.contains("params.beta == 0.0f"),
                "GEMM kernel must guard the C read on beta==0:\n{src}"
            );
        }
    }

    #[test]
    fn msl_batched_gemm_contains_kernel_name() {
        let src = batched_gemm_msl();
        assert!(src.contains("batched_gemm_f32"));
        assert!(src.contains("BatchedGemmParams"));
        assert!(src.contains("metal_stdlib"));
        assert!(src.contains("batch_count"));
        assert!(src.contains("stride_a"));
        assert!(src.contains("stride_b"));
        assert!(src.contains("stride_c"));
    }

    #[test]
    fn msl_batched_gemm_uses_3d_grid() {
        let src = batched_gemm_msl();
        assert!(src.contains("uint3 gid"));
        assert!(src.contains("uint3 tgid"));
        assert!(src.contains("tgid.z"));
    }

    #[test]
    fn msl_gemm_f16_contains_kernel_name() {
        let src = gemm_msl_f16();
        assert!(src.contains("gemm_f16"));
        assert!(src.contains("GemmParamsF16"));
        assert!(src.contains("metal_stdlib"));
    }

    #[test]
    fn msl_gemm_f16_uses_half_type() {
        let src = gemm_msl_f16();
        assert!(src.contains("half"));
        assert!(src.contains("device const half*"));
        assert!(src.contains("device half*"));
        // Accumulation should be in float for precision
        assert!(src.contains("float acc"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn msl_batched_gemm_compiles_on_macos() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        match device.new_library_with_source(batched_gemm_msl(), &opts) {
            Ok(_) => {}
            Err(e) => panic!("Batched GEMM MSL failed to compile: {e}"),
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn msl_gemm_f16_compiles_on_macos() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        match device.new_library_with_source(gemm_msl_f16(), &opts) {
            Ok(_) => {}
            Err(e) => panic!("FP16 GEMM MSL failed to compile: {e}"),
        }
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

    // ── v2 device compilation ─────────────────────────────────────────────────

    #[cfg(target_os = "macos")]
    /// The v2 conv kernel must carry **no** baked shape constants: that is the
    /// entire reason it exists, and one leftover `constant uint` would silently
    /// restore the per-shape pipeline explosion the parameter buffer removes.
    #[test]
    fn conv2d_v2_bakes_no_shape_constants() {
        let src = conv2d_msl_v2();
        assert!(src.contains("struct ConvParamsV2"));
        assert!(src.contains("constant ConvParamsV2& params [[buffer(3)]]"));
        assert!(
            !src.contains("constant uint"),
            "every shape must travel in the parameter buffer, not as a baked constant"
        );
        // The signed padding guard is what makes the kernel agree with the CPU
        // oracle at the borders; keep it verbatim from the v1 kernel.
        assert!(src.contains("int(oy * params.stride_h + fy) - int(params.pad_h)"));
        assert!(src.contains("iy >= 0 && uint(iy) < params.h_in"));
        assert!(src.contains(&format!("kernel void {}(", conv2d_v2_function_name())));
    }

    /// Thirteen `uint` fields, in the order the doc table promises.
    #[test]
    fn conv2d_v2_param_struct_has_the_documented_layout() {
        let src = conv2d_msl_v2();
        let start = src.find("struct ConvParamsV2 {").expect("params struct");
        let end = start + src[start..].find("};").expect("struct end");
        let body = &src[start..end];
        let fields: Vec<&str> = body
            .lines()
            .filter_map(|l| l.trim().strip_prefix("uint "))
            .filter_map(|l| l.strip_suffix(';'))
            .collect();
        assert_eq!(
            fields,
            [
                "n_batch", "c_in", "h_in", "w_in", "k_out", "fh", "fw", "oh", "ow", "stride_h",
                "stride_w", "pad_h", "pad_w",
            ]
        );
        assert_eq!(fields.len() * 4, CONV_PARAMS_V2_BYTES);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn msl_v2_sources_compile_on_macos() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        let mut sources = vec![
            ("gemm_v2_f32".to_string(), gemm_msl_v2(GemmDtype::F32)),
            ("gemm_v2_f16".to_string(), gemm_msl_v2(GemmDtype::F16)),
            (
                "batched_gemm_v2_f32".to_string(),
                batched_gemm_msl_v2(GemmDtype::F32),
            ),
            (
                "batched_gemm_v2_f16".to_string(),
                batched_gemm_msl_v2(GemmDtype::F16),
            ),
            (
                conv2d_v2_function_name().to_string(),
                conv2d_msl_v2().to_string(),
            ),
        ];
        for op in UNARY_OPS {
            sources.push((
                format!("unary {op}"),
                elementwise_msl_v2(op).unwrap_or_else(|e| panic!("{op}: {e}")),
            ));
        }
        for op in BINARY_OPS {
            sources.push((
                format!("binary {op}"),
                binary_msl_v2(op).unwrap_or_else(|e| panic!("{op}: {e}")),
            ));
        }
        // Strict-IEEE variants of every kernel that depends on exact semantics.
        for op in ["sum", "max", "min", "mean"] {
            sources.push((
                format!("precise reduce {op}"),
                reduction_msl_with_mode(op, MslMathMode::Precise).unwrap_or_else(|e| panic!("{e}")),
            ));
        }
        for (label, src) in &sources {
            if let Err(e) = device.new_library_with_source(src, &opts) {
                panic!("{label} MSL failed to compile: {e}\n--- source ---\n{src}");
            }
        }
    }

    /// Regression for the power-of-two-only tree reduction: run the *live*
    /// reduction kernel with a 100-wide threadgroup, which used to drop lane 24's
    /// partial silently.
    #[cfg(target_os = "macos")]
    #[test]
    fn reduction_is_exact_with_a_non_power_of_two_threadgroup() {
        use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
        let Some(device) = Device::system_default() else {
            return;
        };
        let queue = device.new_command_queue();
        let reduce_size = 250u32;
        let input: Vec<f32> = (0..reduce_size).map(|i| (i % 17) as f32 + 0.5).collect();
        let expect: f32 = input.iter().sum();

        for tg in [100u64, 128, 250] {
            let src = reduction_msl("sum");
            let Ok(lib) = device.new_library_with_source(&src, &CompileOptions::new()) else {
                return;
            };
            let Ok(func) = lib.get_function("reduce_sum_f32", None) else {
                return;
            };
            let Ok(pso) = device.new_compute_pipeline_state_with_function(&func) else {
                return;
            };
            let buf_in = device.new_buffer_with_data(
                input.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(input.as_slice()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let buf_out = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
            let (outer, inner) = (1u32, 1u32);
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&buf_in), 0);
            enc.set_buffer(1, Some(&buf_out), 0);
            enc.set_bytes(2, 4, &outer as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(3, 4, &reduce_size as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &inner as *const u32 as *const std::ffi::c_void);
            enc.set_threadgroup_memory_length(0, tg * 4);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            // SAFETY: shared storage, GPU work complete, one f32 allocated.
            let got = unsafe { *(buf_out.contents() as *const f32) };
            assert!(
                (got - expect).abs() <= 1e-3,
                "tg={tg}: got {got} want {expect}"
            );
        }
    }
}
