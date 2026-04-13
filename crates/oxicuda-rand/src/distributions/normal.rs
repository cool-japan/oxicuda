//! Normal (Gaussian) distribution via Box-Muller transform.
//!
//! Provides PTX emission helpers that transform pairs of uniform random
//! values into standard normal samples using the Box-Muller method:
//!
//! ```text
//! z0 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
//! z1 = sqrt(-2 * ln(u1)) * sin(2 * pi * u2)
//! ```

use oxicuda_ptx::builder::BodyBuilder;
use oxicuda_ptx::ir::{PtxType, Register};

/// Emits the Box-Muller transform in f32 PTX, producing two standard
/// normal values (z0, z1) from two uniform inputs u1, u2 in (0,1\].
///
/// Uses PTX approximate transcendentals (`lg2.approx.f32`, `sqrt.approx.f32`,
/// `cos.approx.f32`, `sin.approx.f32`).
///
/// Returns `(z0, z1)` register pair.
#[allow(dead_code)]
pub fn emit_box_muller_f32(
    b: &mut BodyBuilder<'_>,
    u1: Register,
    u2: Register,
) -> (Register, Register) {
    // Clamp u1 away from zero to avoid ln(0)
    let eps = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {eps}, 0f33800000;")); // ~5.96e-8
    let u1_safe = b.max_f32(u1, eps);

    // ln(u1) = lg2(u1) * ln(2)
    let lg2_u1 = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("lg2.approx.f32 {lg2_u1}, {u1_safe};"));
    let ln2 = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {ln2}, 0f3F317218;")); // ln(2) = 0.6931471805
    let ln_u1 = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mul.rn.f32 {ln_u1}, {lg2_u1}, {ln2};"));

    // -2 * ln(u1)
    let neg2 = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {neg2}, 0fC0000000;")); // -2.0
    let neg2ln = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mul.rn.f32 {neg2ln}, {neg2}, {ln_u1};"));

    // radius = sqrt(-2 * ln(u1))
    let radius = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("sqrt.approx.f32 {radius}, {neg2ln};"));

    // angle = 2 * pi * u2
    let two_pi = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {two_pi}, 0f40C90FDB;")); // 2*pi = 6.2831853
    let angle = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mul.rn.f32 {angle}, {two_pi}, {u2};"));

    // z0 = radius * cos(angle)
    let cos_val = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("cos.approx.f32 {cos_val}, {angle};"));
    let z0 = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mul.rn.f32 {z0}, {radius}, {cos_val};"));

    // z1 = radius * sin(angle)
    let sin_val = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("sin.approx.f32 {sin_val}, {angle};"));
    let z1 = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mul.rn.f32 {z1}, {radius}, {sin_val};"));

    (z0, z1)
}

/// Emits the Box-Muller transform targeting f64 output.
///
/// Since PTX only provides f32 approximate transcendentals, the transform
/// is computed in f32 and the results are widened to f64. This matches
/// cuRAND's approach for double-precision normals.
///
/// Returns `(z0, z1)` register pair in f64.
#[allow(dead_code)]
pub fn emit_box_muller_f64(
    b: &mut BodyBuilder<'_>,
    u1: Register,
    u2: Register,
) -> (Register, Register) {
    // Compute Box-Muller in f32 using approximate transcendentals
    let (z0_f32, z1_f32) = emit_box_muller_f32(b, u1, u2);

    // Widen to f64
    let z0 = b.cvt_f32_to_f64(z0_f32);
    let z1 = b.cvt_f32_to_f64(z1_f32);

    (z0, z1)
}

/// Emits PTX to scale a standard normal value to `mean + stddev * z`.
///
/// Works for both f32 and f64 precisions via the `precision` parameter.
///
/// Returns the register containing the scaled result.
#[allow(dead_code)]
pub fn emit_normal_scale(
    b: &mut BodyBuilder<'_>,
    z: Register,
    mean: Register,
    stddev: Register,
    precision: PtxType,
) -> Register {
    match precision {
        PtxType::F32 => b.fma_f32(stddev, z, mean),
        PtxType::F64 => b.fma_f64(stddev, z, mean),
        _ => z, // Unsupported: return z unchanged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;
    use oxicuda_ptx::builder::KernelBuilder;

    #[test]
    fn box_muller_f32_compiles() {
        let ptx = KernelBuilder::new("test_bm_f32")
            .target(SmVersion::Sm80)
            .param("u1", PtxType::U32)
            .param("u2", PtxType::U32)
            .body(|b| {
                let u1_raw = b.load_param_u32("u1");
                let u2_raw = b.load_param_u32("u2");
                // Convert to f32 first
                let u1_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {u1_f}, {u1_raw};"));
                let u2_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {u2_f}, {u2_raw};"));
                let (_z0, _z1) = emit_box_muller_f32(b, u1_f, u2_f);
                b.ret();
            })
            .build();
        let ptx = ptx.expect("should compile");
        assert!(ptx.contains("lg2.approx.f32"));
        assert!(ptx.contains("cos.approx.f32"));
        assert!(ptx.contains("sin.approx.f32"));
        assert!(ptx.contains("sqrt.approx.f32"));
    }

    #[test]
    fn box_muller_f64_compiles() {
        let ptx = KernelBuilder::new("test_bm_f64")
            .target(SmVersion::Sm80)
            .param("u1", PtxType::U32)
            .param("u2", PtxType::U32)
            .body(|b| {
                let u1_raw = b.load_param_u32("u1");
                let u2_raw = b.load_param_u32("u2");
                let u1_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {u1_f}, {u1_raw};"));
                let u2_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {u2_f}, {u2_raw};"));
                let (_z0, _z1) = emit_box_muller_f64(b, u1_f, u2_f);
                b.ret();
            })
            .build();
        let ptx = ptx.expect("should compile");
        assert!(ptx.contains("cvt.f64.f32"));
    }

    #[test]
    fn normal_scale_f32_compiles() {
        let ptx = KernelBuilder::new("test_nscale_f32")
            .target(SmVersion::Sm80)
            .param("z", PtxType::F32)
            .param("mean", PtxType::F32)
            .param("stddev", PtxType::F32)
            .body(|b| {
                let z = b.load_param_f32("z");
                let mean = b.load_param_f32("mean");
                let stddev = b.load_param_f32("stddev");
                let _result = emit_normal_scale(b, z, mean, stddev, PtxType::F32);
                b.ret();
            })
            .build();
        assert!(ptx.is_ok());
    }
}
