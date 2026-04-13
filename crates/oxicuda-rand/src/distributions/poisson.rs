//! Poisson distribution PTX helpers.
//!
//! Two algorithms are provided:
//!
//! - **Small lambda (< 30)**: Knuth's algorithm, which generates uniforms
//!   and multiplies them until the product drops below `exp(-lambda)`.
//! - **Large lambda (>= 30)**: Normal approximation where the Poisson
//!   variate is approximated as `round(lambda + sqrt(lambda) * z)` where
//!   `z` is a standard normal.

use oxicuda_ptx::builder::BodyBuilder;
use oxicuda_ptx::ir::{PtxType, Register};

/// Threshold below which we use Knuth's exact algorithm. Above this
/// we switch to the normal approximation for efficiency.
#[allow(dead_code)]
pub const POISSON_LAMBDA_THRESHOLD: f32 = 30.0;

/// Emits PTX for Poisson sampling with small lambda using Knuth's algorithm.
///
/// The algorithm:
/// 1. Let `L = exp(-lambda)`, `k = 0`, `p = 1.0`
/// 2. Loop: `k += 1`, generate uniform `u`, `p *= u`
/// 3. While `p > L`, continue
/// 4. Return `k - 1`
///
/// The `uniform_gen` closure is called each iteration to emit PTX that
/// produces one fresh uniform f32 in (0,1\] from the PRNG state.
///
/// Returns a u32 register containing the Poisson sample.
#[allow(dead_code)]
pub fn emit_poisson_small_f32<F>(
    b: &mut BodyBuilder<'_>,
    lambda_reg: Register,
    uniform_gen: F,
) -> Register
where
    F: Fn(&mut BodyBuilder<'_>) -> Register,
{
    // L = exp(-lambda) = 2^(-lambda * log2(e))
    let log2e = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {log2e}, 0f3FB8AA3B;")); // log2(e)
    let neg_lambda = b.neg_f32(lambda_reg);
    let scaled = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mul.rn.f32 {scaled}, {neg_lambda}, {log2e};"));
    let limit = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("ex2.approx.f32 {limit}, {scaled};")); // exp(-lambda)

    // k = 0, p = 1.0
    let k = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {k}, 0;"));
    let p = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {p}, 0f3F800000;")); // 1.0

    // Loop label
    let loop_label = b.fresh_label("poisson_loop");
    let end_label = b.fresh_label("poisson_end");

    b.label(&loop_label);

    // k += 1
    b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));

    // p *= uniform()
    let u = uniform_gen(b);
    b.raw_ptx(&format!("mul.rn.f32 {p}, {p}, {u};"));

    // If p > L, continue loop
    let pred_continue = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.gt.f32 {pred_continue}, {p}, {limit};"));
    b.branch_if(pred_continue, &loop_label);

    b.label(&end_label);

    // result = k - 1
    let result = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("sub.u32 {result}, {k}, 1;"));
    result
}

/// Emits PTX for Poisson sampling with large lambda using normal approximation.
///
/// For large lambda, `Poisson(lambda) ~ Normal(lambda, sqrt(lambda))`, so:
/// ```text
/// result = max(0, round(lambda + sqrt(lambda) * z))
/// ```
/// where `z` is a standard normal variate.
///
/// Returns a u32 register containing the Poisson sample.
#[allow(dead_code)]
pub fn emit_poisson_large_f32(
    b: &mut BodyBuilder<'_>,
    lambda_reg: Register,
    normal_reg: Register,
) -> Register {
    // sqrt(lambda)
    let sqrt_lambda = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("sqrt.approx.f32 {sqrt_lambda}, {lambda_reg};"));

    // lambda + sqrt(lambda) * z
    let approx = b.fma_f32(sqrt_lambda, normal_reg, lambda_reg);

    // Clamp to >= 0
    let zero = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {zero}, 0f00000000;")); // 0.0
    let clamped = b.max_f32(approx, zero);

    // Round to nearest integer, then truncate to u32
    let rounded = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("cvt.rni.f32.f32 {rounded}, {clamped};"));
    let result = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("cvt.rzi.u32.f32 {result}, {rounded};"));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;
    use oxicuda_ptx::builder::KernelBuilder;

    #[test]
    fn poisson_small_compiles() {
        let ptx = KernelBuilder::new("test_poisson_small")
            .target(SmVersion::Sm80)
            .param("lambda", PtxType::F32)
            .body(|b| {
                let lambda = b.load_param_f32("lambda");
                // Simple uniform generator that just returns a constant for testing
                let _result = emit_poisson_small_f32(b, lambda, |b| {
                    let half = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {half}, 0f3F000000;")); // 0.5
                    half
                });
                b.ret();
            })
            .build();
        let ptx = ptx.expect("should compile");
        assert!(ptx.contains("ex2.approx.f32")); // exp(-lambda)
        assert!(ptx.contains("setp.gt.f32")); // loop condition
    }

    #[test]
    fn poisson_large_compiles() {
        let ptx = KernelBuilder::new("test_poisson_large")
            .target(SmVersion::Sm80)
            .param("lambda", PtxType::F32)
            .param("z", PtxType::F32)
            .body(|b| {
                let lambda = b.load_param_f32("lambda");
                let z = b.load_param_f32("z");
                let _result = emit_poisson_large_f32(b, lambda, z);
                b.ret();
            })
            .build();
        let ptx = ptx.expect("should compile");
        assert!(ptx.contains("sqrt.approx.f32"));
        assert!(ptx.contains("cvt.rni.f32.f32")); // round
    }
}
