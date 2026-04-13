//! XORWOW generator engine.
//!
//! XORWOW combines a 5-register XORshift generator with a Weyl sequence
//! counter. It is fast and suitable for most Monte Carlo applications.
//!
//! State: 5 x u32 (XORshift) + 1 x u32 (Weyl counter `d`).
//! Each step produces one u32, advancing the state.
//!
//! Reference: Marsaglia, "Xorshift RNGs" (2003), with Weyl addition.

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::error::PtxGenError;
use oxicuda_ptx::ir::PtxType;

/// Weyl sequence increment constant for XORWOW.
const XORWOW_WEYL_INC: u32 = 362_437;

// ---------------------------------------------------------------------------
// Uniform distribution PTX generators
// ---------------------------------------------------------------------------

/// Generates PTX for a XORWOW uniform distribution kernel.
///
/// Each thread initializes its own state from `(seed, thread_id)`, then
/// advances the state to produce one float value.
///
/// Parameters: `(out_ptr, n, seed, offset_lo, offset_hi)`
///
/// # Errors
///
/// Returns `PtxGenError` if kernel construction fails.
pub fn generate_xorwow_uniform_ptx(
    precision: PtxType,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    let kernel_name = match precision {
        PtxType::F32 => "xorwow_uniform_f32",
        PtxType::F64 => "xorwow_uniform_f64",
        _ => return Err(PtxGenError::InvalidType(format!("{precision:?}"))),
    };

    let stride_bytes: u32 = precision.size_bytes() as u32;

    KernelBuilder::new(kernel_name)
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("seed", PtxType::U32)
        .param("offset_lo", PtxType::U32)
        .param("offset_hi", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let seed = b.load_param_u32("seed");

                // Initialize state from seed ^ (gid * scramble constants)
                // s0..s4 = seed XOR gid-derived values, d = 0
                let s0 = b.alloc_reg(PtxType::U32);
                let s1 = b.alloc_reg(PtxType::U32);
                let s2 = b.alloc_reg(PtxType::U32);
                let s3 = b.alloc_reg(PtxType::U32);
                let s4 = b.alloc_reg(PtxType::U32);
                let d = b.alloc_reg(PtxType::U32);

                // Scramble seed with thread ID using different constants
                b.raw_ptx(&format!("xor.b32 {s0}, {seed}, {gid};"));
                let scr1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr1}, {gid}, 1812433253;"));
                b.raw_ptx(&format!("xor.b32 {s1}, {seed}, {scr1};"));
                let scr2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr2}, {gid}, 1566083941;"));
                b.raw_ptx(&format!("xor.b32 {s2}, {seed}, {scr2};"));
                let scr3 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr3}, {gid}, 1103515245;"));
                b.raw_ptx(&format!("xor.b32 {s3}, {seed}, {scr3};"));
                let scr4 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr4}, {gid}, 214013;"));
                b.raw_ptx(&format!("xor.b32 {s4}, {seed}, {scr4};"));
                b.raw_ptx(&format!("mov.u32 {d}, 0;"));

                // Ensure at least one nonzero state word
                let any_zero_pred = b.alloc_reg(PtxType::Pred);
                let or_s = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("or.b32 {or_s}, {s0}, {s1};"));
                b.raw_ptx(&format!("or.b32 {or_s}, {or_s}, {s2};"));
                b.raw_ptx(&format!("or.b32 {or_s}, {or_s}, {s3};"));
                b.raw_ptx(&format!("or.b32 {or_s}, {or_s}, {s4};"));
                b.raw_ptx(&format!("setp.eq.u32 {any_zero_pred}, {or_s}, 0;"));
                b.raw_ptx(&format!("@{any_zero_pred} mov.u32 {s0}, 1;"));

                // One XORWOW step
                emit_xorwow_step(b, &s0, &s1, &s2, &s3, &s4, &d);

                // Output = s4 + d (the XORWOW combined output)
                let raw_val = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {raw_val}, {s4}, {d};"));

                match precision {
                    PtxType::F32 => {
                        let fval = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {fval}, {raw_val};"));
                        let scale = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;"));
                        let result = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {result}, {fval}, {scale};"));
                        let addr = b.byte_offset_addr(out_ptr, gid.clone(), stride_bytes);
                        b.store_global_f32(addr, result);
                    }
                    PtxType::F64 => {
                        let fval = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("cvt.rn.f64.u32 {fval}, {raw_val};"));
                        let scale = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mov.f64 {scale}, 0d3DF0000000000000;"));
                        let result = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {result}, {fval}, {scale};"));
                        let addr = b.byte_offset_addr(out_ptr, gid.clone(), stride_bytes);
                        b.store_global_f64(addr, result);
                    }
                    _ => {}
                }
            });

            b.ret();
        })
        .build()
}

/// Generates PTX for a XORWOW normal distribution kernel (Box-Muller).
///
/// Parameters: `(out_ptr, n, seed, offset_lo, offset_hi, mean, stddev)`
///
/// # Errors
///
/// Returns `PtxGenError` if kernel construction fails.
pub fn generate_xorwow_normal_ptx(
    precision: PtxType,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    let kernel_name = match precision {
        PtxType::F32 => "xorwow_normal_f32",
        PtxType::F64 => "xorwow_normal_f64",
        _ => return Err(PtxGenError::InvalidType(format!("{precision:?}"))),
    };

    let stride_bytes: u32 = precision.size_bytes() as u32;
    let mean_ty = precision;
    let stddev_ty = precision;

    KernelBuilder::new(kernel_name)
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("seed", PtxType::U32)
        .param("offset_lo", PtxType::U32)
        .param("offset_hi", PtxType::U32)
        .param("mean", mean_ty)
        .param("stddev", stddev_ty)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let seed = b.load_param_u32("seed");

                // Initialize XORWOW state
                let s0 = b.alloc_reg(PtxType::U32);
                let s1 = b.alloc_reg(PtxType::U32);
                let s2 = b.alloc_reg(PtxType::U32);
                let s3 = b.alloc_reg(PtxType::U32);
                let s4 = b.alloc_reg(PtxType::U32);
                let d = b.alloc_reg(PtxType::U32);

                b.raw_ptx(&format!("xor.b32 {s0}, {seed}, {gid};"));
                let scr1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr1}, {gid}, 1812433253;"));
                b.raw_ptx(&format!("xor.b32 {s1}, {seed}, {scr1};"));
                let scr2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr2}, {gid}, 1566083941;"));
                b.raw_ptx(&format!("xor.b32 {s2}, {seed}, {scr2};"));
                let scr3 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr3}, {gid}, 1103515245;"));
                b.raw_ptx(&format!("xor.b32 {s3}, {seed}, {scr3};"));
                let scr4 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr4}, {gid}, 214013;"));
                b.raw_ptx(&format!("xor.b32 {s4}, {seed}, {scr4};"));
                b.raw_ptx(&format!("mov.u32 {d}, 0;"));

                // Ensure nonzero
                let or_s = b.alloc_reg(PtxType::U32);
                let any_zero_pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("or.b32 {or_s}, {s0}, {s1};"));
                b.raw_ptx(&format!("or.b32 {or_s}, {or_s}, {s2};"));
                b.raw_ptx(&format!("or.b32 {or_s}, {or_s}, {s3};"));
                b.raw_ptx(&format!("or.b32 {or_s}, {or_s}, {s4};"));
                b.raw_ptx(&format!("setp.eq.u32 {any_zero_pred}, {or_s}, 0;"));
                b.raw_ptx(&format!("@{any_zero_pred} mov.u32 {s0}, 1;"));

                // Generate two u32 values for Box-Muller
                emit_xorwow_step(b, &s0, &s1, &s2, &s3, &s4, &d);
                let u1_raw = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {u1_raw}, {s4}, {d};"));

                emit_xorwow_step(b, &s0, &s1, &s2, &s3, &s4, &d);
                let u2_raw = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {u2_raw}, {s4}, {d};"));

                match precision {
                    PtxType::F32 => {
                        let mean_reg = b.load_param_f32("mean");
                        let stddev_reg = b.load_param_f32("stddev");

                        let scale = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;"));

                        // u1 in (0,1]
                        let u1_f = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {u1_f}, {u1_raw};"));
                        let u1 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {u1}, {u1_f}, {scale};"));
                        let eps = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {eps}, 0f33800000;"));
                        let u1_safe = b.max_f32(u1, eps);

                        // u2 in [0,1)
                        let u2_f = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {u2_f}, {u2_raw};"));
                        let u2 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {u2}, {u2_f}, {scale};"));

                        // Box-Muller
                        let lg2_u1 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("lg2.approx.f32 {lg2_u1}, {u1_safe};"));
                        let ln2 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {ln2}, 0f3F317218;"));
                        let ln_u1 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {ln_u1}, {lg2_u1}, {ln2};"));
                        let neg2 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {neg2}, 0fC0000000;"));
                        let neg2ln = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {neg2ln}, {neg2}, {ln_u1};"));
                        let radius = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("sqrt.approx.f32 {radius}, {neg2ln};"));
                        let two_pi = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {two_pi}, 0f40C90FDB;"));
                        let angle = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {angle}, {two_pi}, {u2};"));
                        let cos_val = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cos.approx.f32 {cos_val}, {angle};"));
                        let z = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {z}, {radius}, {cos_val};"));
                        let result = b.fma_f32(stddev_reg, z, mean_reg);
                        let addr = b.byte_offset_addr(out_ptr, gid.clone(), stride_bytes);
                        b.store_global_f32(addr, result);
                    }
                    PtxType::F64 => {
                        let mean_reg = b.load_param_f64("mean");
                        let stddev_reg = b.load_param_f64("stddev");

                        let scale = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;"));

                        let u1_f = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {u1_f}, {u1_raw};"));
                        let u1 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {u1}, {u1_f}, {scale};"));
                        let eps = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {eps}, 0f33800000;"));
                        let u1_safe = b.max_f32(u1, eps);

                        let u2_f = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {u2_f}, {u2_raw};"));
                        let u2 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {u2}, {u2_f}, {scale};"));

                        let lg2_u1 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("lg2.approx.f32 {lg2_u1}, {u1_safe};"));
                        let ln2 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {ln2}, 0f3F317218;"));
                        let ln_u1 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {ln_u1}, {lg2_u1}, {ln2};"));
                        let neg2 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {neg2}, 0fC0000000;"));
                        let neg2ln = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {neg2ln}, {neg2}, {ln_u1};"));
                        let radius = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("sqrt.approx.f32 {radius}, {neg2ln};"));
                        let two_pi = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {two_pi}, 0f40C90FDB;"));
                        let angle = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {angle}, {two_pi}, {u2};"));
                        let cos_val = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cos.approx.f32 {cos_val}, {angle};"));
                        let z32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {z32}, {radius}, {cos_val};"));

                        let z64 = b.cvt_f32_to_f64(z32);
                        let result = b.fma_f64(stddev_reg, z64, mean_reg);
                        let addr = b.byte_offset_addr(out_ptr, gid.clone(), stride_bytes);
                        b.store_global_f64(addr, result);
                    }
                    _ => {}
                }
            });

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// XORWOW step helper
// ---------------------------------------------------------------------------

/// Emits one step of the XORWOW generator.
///
/// ```text
/// t = s0 ^ (s0 >> 2)
/// s0 = s1; s1 = s2; s2 = s3; s3 = s4
/// s4 = (s4 ^ (s4 << 4)) ^ (t ^ (t << 1))
/// d += 362437
/// ```
fn emit_xorwow_step(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    s0: &oxicuda_ptx::ir::Register,
    s1: &oxicuda_ptx::ir::Register,
    s2: &oxicuda_ptx::ir::Register,
    s3: &oxicuda_ptx::ir::Register,
    s4: &oxicuda_ptx::ir::Register,
    d: &oxicuda_ptx::ir::Register,
) {
    b.comment("XORWOW step");
    let t = b.alloc_reg(PtxType::U32);
    let tmp = b.alloc_reg(PtxType::U32);

    // t = s0 ^ (s0 >> 2)
    b.raw_ptx(&format!("shr.u32 {t}, {s0}, 2;"));
    b.raw_ptx(&format!("xor.b32 {t}, {s0}, {t};"));

    // Shift state: s0=s1, s1=s2, s2=s3, s3=s4
    b.raw_ptx(&format!("mov.u32 {s0}, {s1};"));
    b.raw_ptx(&format!("mov.u32 {s1}, {s2};"));
    b.raw_ptx(&format!("mov.u32 {s2}, {s3};"));
    b.raw_ptx(&format!("mov.u32 {s3}, {s4};"));

    // s4 = (s4 ^ (s4 << 4)) ^ (t ^ (t << 1))
    b.raw_ptx(&format!("shl.b32 {tmp}, {s4}, 4;"));
    b.raw_ptx(&format!("xor.b32 {s4}, {s4}, {tmp};"));
    b.raw_ptx(&format!("shl.b32 {tmp}, {t}, 1;"));
    b.raw_ptx(&format!("xor.b32 {t}, {t}, {tmp};"));
    b.raw_ptx(&format!("xor.b32 {s4}, {s4}, {t};"));

    // d += WEYL_INC
    b.raw_ptx(&format!("add.u32 {d}, {d}, {XORWOW_WEYL_INC};"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;

    #[test]
    fn generate_xorwow_uniform_f32() {
        let ptx = generate_xorwow_uniform_ptx(PtxType::F32, SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry xorwow_uniform_f32"));
        assert!(ptx.contains("xor.b32"));
        assert!(ptx.contains("shr.u32"));
    }

    #[test]
    fn generate_xorwow_normal_f32() {
        let ptx = generate_xorwow_normal_ptx(PtxType::F32, SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry xorwow_normal_f32"));
        assert!(ptx.contains("lg2.approx"));
    }

    #[test]
    fn invalid_precision_returns_error() {
        let result = generate_xorwow_uniform_ptx(PtxType::U32, SmVersion::Sm80);
        assert!(result.is_err());
    }
}
