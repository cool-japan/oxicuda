//! Log-normal distribution transform.
//!
//! A log-normal random variable is `exp(X)` where `X ~ Normal(mean, stddev)`.
//! The PTX helpers here compute `exp(z)` using the identity
//! `exp(x) = 2^(x * log2(e))` and the PTX `ex2.approx.f32` instruction.

use oxicuda_ptx::builder::BodyBuilder;
use oxicuda_ptx::ir::{PtxType, Register};

/// Emits PTX to compute `exp(z)` in f32.
///
/// Uses `ex2.approx.f32` with the identity `exp(x) = 2^(x * log2(e))`.
/// `log2(e) = 1.4426950408...` which is `0x3FB8AA3B` in IEEE 754 hex.
///
/// Returns the f32 register containing `exp(z)`.
#[allow(dead_code)]
pub fn emit_log_normal_transform_f32(b: &mut BodyBuilder<'_>, normal_val: Register) -> Register {
    // log2(e) constant
    let log2e = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {log2e}, 0f3FB8AA3B;")); // log2(e) = 1.4426950408

    // x * log2(e)
    let scaled = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mul.rn.f32 {scaled}, {normal_val}, {log2e};"));

    // 2^(x * log2(e)) = exp(x)
    let result = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("ex2.approx.f32 {result}, {scaled};"));
    result
}

/// Emits PTX to compute `exp(z)` targeting f64 output.
///
/// Since PTX only provides `ex2.approx.f32`, the exponentiation is done
/// in f32 and the result is widened to f64. For applications needing higher
/// precision, a software exp implementation would be needed.
///
/// Returns the f64 register containing `exp(z)` (widened from f32).
#[allow(dead_code)]
pub fn emit_log_normal_transform_f64(b: &mut BodyBuilder<'_>, normal_val: Register) -> Register {
    // Narrow f64 input to f32 for the approximate transcendental
    let narrow = b.cvt_f64_to_f32(normal_val);

    // Compute exp in f32
    let exp_f32 = emit_log_normal_transform_f32(b, narrow);

    // Widen back to f64
    b.cvt_f32_to_f64(exp_f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;
    use oxicuda_ptx::builder::KernelBuilder;

    #[test]
    fn log_normal_f32_compiles() {
        let ptx = KernelBuilder::new("test_lognormal_f32")
            .target(SmVersion::Sm80)
            .param("z", PtxType::F32)
            .body(|b| {
                let z = b.load_param_f32("z");
                let _result = emit_log_normal_transform_f32(b, z);
                b.ret();
            })
            .build();
        let ptx = ptx.expect("should compile");
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("0f3FB8AA3B")); // log2(e)
    }

    #[test]
    fn log_normal_f64_compiles() {
        let ptx = KernelBuilder::new("test_lognormal_f64")
            .target(SmVersion::Sm80)
            .param("z", PtxType::F64)
            .body(|b| {
                let z = b.load_param_f64("z");
                let _result = emit_log_normal_transform_f64(b, z);
                b.ret();
            })
            .build();
        let ptx = ptx.expect("should compile");
        assert!(ptx.contains("cvt.rn.f32.f64")); // narrow
        assert!(ptx.contains("cvt.f64.f32")); // widen
    }
}
