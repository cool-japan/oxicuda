//! Uniform distribution PTX helpers.
//!
//! Provides helper functions to emit PTX instructions that convert raw
//! integer PRNG output to uniformly distributed floating-point values.

use oxicuda_ptx::builder::BodyBuilder;
use oxicuda_ptx::ir::{PtxType, Register};

/// Emits PTX to convert a raw u32 register to a uniform f32 in \[0, 1).
///
/// The conversion uses multiplication by 2^-32 (hex float `0x2F800000`).
///
/// Returns the f32 register containing the result.
#[allow(dead_code)]
pub fn emit_u32_to_uniform_f32(b: &mut BodyBuilder<'_>, u32_reg: Register) -> Register {
    let fval = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("cvt.rn.f32.u32 {fval}, {u32_reg};"));
    let scale = b.alloc_reg(PtxType::F32);
    // 2^-32 = 2.3283064e-10 in IEEE 754 hex
    b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;"));
    let result = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mul.rn.f32 {result}, {fval}, {scale};"));
    result
}

/// Emits PTX to convert two raw u32 registers to a uniform f64 in \[0, 1).
///
/// Combines `hi` and `lo` words: `(hi * 2^-32 + lo * 2^-64)` for 53-bit
/// precision in the mantissa.
///
/// Returns the f64 register containing the result.
#[allow(dead_code)]
pub fn emit_u32_to_uniform_f64(b: &mut BodyBuilder<'_>, hi: Register, lo: Register) -> Register {
    let fval_lo = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("cvt.rn.f64.u32 {fval_lo}, {lo};"));
    let fval_hi = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("cvt.rn.f64.u32 {fval_hi}, {hi};"));
    // 2^-32 in f64 hex: 0x3DF0000000000000
    let scale_hi = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("mov.f64 {scale_hi}, 0d3DF0000000000000;"));
    // 2^-64 in f64 hex: 0x3BF0000000000000
    let scale_lo = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("mov.f64 {scale_lo}, 0d3BF0000000000000;"));
    let part_hi = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("mul.rn.f64 {part_hi}, {fval_hi}, {scale_hi};"));
    let result = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!(
        "fma.rn.f64 {result}, {fval_lo}, {scale_lo}, {part_hi};"
    ));
    result
}

/// Emits PTX to scale a uniform value from \[0,1) to \[lo, hi).
///
/// Computes `val * (hi - lo) + lo` (i.e. an affine transform).
///
/// The `precision` parameter controls whether f32 or f64 arithmetic is used.
///
/// Returns the register containing the scaled result.
#[allow(dead_code)]
pub fn emit_uniform_scale(
    b: &mut BodyBuilder<'_>,
    val: Register,
    lo_reg: Register,
    hi_reg: Register,
    precision: PtxType,
) -> Register {
    match precision {
        PtxType::F32 => {
            let range = b.sub_f32(hi_reg, lo_reg.clone());
            let scaled = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("mul.rn.f32 {scaled}, {val}, {range};"));
            b.add_f32(scaled, lo_reg)
        }
        PtxType::F64 => {
            let range = b.sub_f64(hi_reg, lo_reg.clone());
            let scaled = b.alloc_reg(PtxType::F64);
            b.raw_ptx(&format!("mul.rn.f64 {scaled}, {val}, {range};"));
            b.add_f64(scaled, lo_reg)
        }
        _ => {
            // Fallback: return val unchanged for unsupported types.
            val
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;
    use oxicuda_ptx::builder::KernelBuilder;

    #[test]
    fn emit_u32_to_f32_compiles() {
        let ptx = KernelBuilder::new("test_u32_to_f32")
            .target(SmVersion::Sm80)
            .param("input", PtxType::U32)
            .param("out", PtxType::U64)
            .body(|b| {
                let inp = b.load_param_u32("input");
                let _result = emit_u32_to_uniform_f32(b, inp);
                b.ret();
            })
            .build();
        assert!(ptx.is_ok());
    }

    #[test]
    fn emit_u32_pair_to_f64_compiles() {
        let ptx = KernelBuilder::new("test_u32_to_f64")
            .target(SmVersion::Sm80)
            .param("hi", PtxType::U32)
            .param("lo", PtxType::U32)
            .param("out", PtxType::U64)
            .body(|b| {
                let hi = b.load_param_u32("hi");
                let lo = b.load_param_u32("lo");
                let _result = emit_u32_to_uniform_f64(b, hi, lo);
                b.ret();
            })
            .build();
        assert!(ptx.is_ok());
    }
}
