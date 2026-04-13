//! MRG32k3a combined multiple recursive generator engine.
//!
//! MRG32k3a is a combined MRG with two components of order 3, producing
//! a period of approximately 2^191. It has the highest statistical quality
//! among the engines provided by cuRAND.
//!
//! State: two sets of three u32 values (s10, s11, s12) and (s20, s21, s22).
//!
//! Reference: L'Ecuyer, "Good Parameters and Implementations for Combined
//! Multiple Recursive Random Number Generators" (1999).

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::error::PtxGenError;
use oxicuda_ptx::ir::PtxType;

// ---------------------------------------------------------------------------
// MRG32k3a constants
// ---------------------------------------------------------------------------

/// Modulus for component 1: m1 = 2^32 - 209 = 4294967087.
#[allow(dead_code)]
const M1: u64 = 4_294_967_087;

/// Modulus for component 2: m2 = 2^32 - 22853 = 4294944443.
#[allow(dead_code)]
const M2: u64 = 4_294_944_443;

/// Multiplier a12 for component 1.
#[allow(dead_code)]
const A12: u64 = 1_403_580;

/// Negative multiplier a13 for component 1 (used as subtraction).
#[allow(dead_code)]
const A13N: u64 = 810_728;

/// Multiplier a21 for component 2.
#[allow(dead_code)]
const A21: u64 = 527_612;

/// Negative multiplier a23 for component 2 (used as subtraction).
#[allow(dead_code)]
const A23N: u64 = 1_370_589;

// ---------------------------------------------------------------------------
// Uniform distribution PTX generator
// ---------------------------------------------------------------------------

/// Generates PTX for a MRG32k3a uniform distribution kernel.
///
/// Each thread initializes its own state from `(seed, thread_id)`, performs
/// one MRG32k3a step, and produces a float value in \[0,1).
///
/// Parameters: `(out_ptr, n, seed, offset_lo, offset_hi)`
///
/// # Errors
///
/// Returns `PtxGenError` if kernel construction fails.
pub fn generate_mrg32k3a_uniform_ptx(
    precision: PtxType,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    let kernel_name = match precision {
        PtxType::F32 => "mrg32k3a_uniform_f32",
        PtxType::F64 => "mrg32k3a_uniform_f64",
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

                // Initialize 6 state registers from seed ^ gid-derived values.
                // We use different scramble constants per register.
                let s10 = b.alloc_reg(PtxType::U32);
                let s11 = b.alloc_reg(PtxType::U32);
                let s12 = b.alloc_reg(PtxType::U32);
                let s20 = b.alloc_reg(PtxType::U32);
                let s21 = b.alloc_reg(PtxType::U32);
                let s22 = b.alloc_reg(PtxType::U32);

                // Scramble seed with thread id
                b.raw_ptx(&format!("xor.b32 {s10}, {seed}, {gid};"));
                let scr1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr1}, {gid}, 1812433253;"));
                b.raw_ptx(&format!("xor.b32 {s11}, {seed}, {scr1};"));
                let scr2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr2}, {gid}, 1566083941;"));
                b.raw_ptx(&format!("xor.b32 {s12}, {seed}, {scr2};"));
                let scr3 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr3}, {gid}, 1103515245;"));
                b.raw_ptx(&format!("xor.b32 {s20}, {seed}, {scr3};"));
                let scr4 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr4}, {gid}, 214013;"));
                b.raw_ptx(&format!("xor.b32 {s21}, {seed}, {scr4};"));
                let scr5 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr5}, {gid}, 2531011;"));
                b.raw_ptx(&format!("xor.b32 {s22}, {seed}, {scr5};"));

                // Ensure state values are nonzero and within modulus bounds
                // by OR'ing 1 if zero
                emit_clamp_nonzero(b, &s10);
                emit_clamp_nonzero(b, &s11);
                emit_clamp_nonzero(b, &s12);
                emit_clamp_nonzero(b, &s20);
                emit_clamp_nonzero(b, &s21);
                emit_clamp_nonzero(b, &s22);

                // One MRG32k3a step
                emit_mrg32k3a_step(b, &s10, &s11, &s12, &s20, &s21, &s22);

                // Output = (s10 - s20) mod m1, converted to float
                // Since we work in u32, compute: if s10 >= s20: s10 - s20, else m1 + s10 - s20
                let diff = b.alloc_reg(PtxType::U32);
                let m1_const = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {m1_const}, {};", M1 as u32));
                let pred_ge = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {pred_ge}, {s10}, {s20};"));
                // diff = s10 - s20
                let raw_diff = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("sub.u32 {raw_diff}, {s10}, {s20};"));
                // wrapped = m1 + s10 - s20 = m1 - (s20 - s10)
                let rev_diff = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("sub.u32 {rev_diff}, {s20}, {s10};"));
                let wrapped = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("sub.u32 {wrapped}, {m1_const}, {rev_diff};"));
                b.raw_ptx(&format!(
                    "selp.u32 {diff}, {raw_diff}, {wrapped}, {pred_ge};"
                ));

                match precision {
                    PtxType::F32 => {
                        let fval = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {fval}, {diff};"));
                        // Scale by 1/m1 ~ 2.328306e-10 * m1 is close to 2^32,
                        // use 1.0 / 4294967087.0
                        // In hex: approximately 0x2F800000 (same as 2^-32)
                        let scale = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;"));
                        let result = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {result}, {fval}, {scale};"));
                        let addr = b.byte_offset_addr(out_ptr, gid.clone(), stride_bytes);
                        b.store_global_f32(addr, result);
                    }
                    PtxType::F64 => {
                        let fval = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("cvt.rn.f64.u32 {fval}, {diff};"));
                        // 1/m1 in f64: use 2^-32 approximation
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

// ---------------------------------------------------------------------------
// Normal distribution PTX generator
// ---------------------------------------------------------------------------

/// Generates PTX for a MRG32k3a normal distribution kernel (Box-Muller).
///
/// Each thread generates two MRG32k3a outputs and applies the Box-Muller
/// transform to produce one normal value.
///
/// Parameters: `(out_ptr, n, seed, offset_lo, offset_hi, mean, stddev)`
///
/// # Errors
///
/// Returns `PtxGenError` if kernel construction fails.
pub fn generate_mrg32k3a_normal_ptx(
    precision: PtxType,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    let kernel_name = match precision {
        PtxType::F32 => "mrg32k3a_normal_f32",
        PtxType::F64 => "mrg32k3a_normal_f64",
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

                // Initialize state
                let s10 = b.alloc_reg(PtxType::U32);
                let s11 = b.alloc_reg(PtxType::U32);
                let s12 = b.alloc_reg(PtxType::U32);
                let s20 = b.alloc_reg(PtxType::U32);
                let s21 = b.alloc_reg(PtxType::U32);
                let s22 = b.alloc_reg(PtxType::U32);

                b.raw_ptx(&format!("xor.b32 {s10}, {seed}, {gid};"));
                let scr1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr1}, {gid}, 1812433253;"));
                b.raw_ptx(&format!("xor.b32 {s11}, {seed}, {scr1};"));
                let scr2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr2}, {gid}, 1566083941;"));
                b.raw_ptx(&format!("xor.b32 {s12}, {seed}, {scr2};"));
                let scr3 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr3}, {gid}, 1103515245;"));
                b.raw_ptx(&format!("xor.b32 {s20}, {seed}, {scr3};"));
                let scr4 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr4}, {gid}, 214013;"));
                b.raw_ptx(&format!("xor.b32 {s21}, {seed}, {scr4};"));
                let scr5 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr5}, {gid}, 2531011;"));
                b.raw_ptx(&format!("xor.b32 {s22}, {seed}, {scr5};"));

                emit_clamp_nonzero(b, &s10);
                emit_clamp_nonzero(b, &s11);
                emit_clamp_nonzero(b, &s12);
                emit_clamp_nonzero(b, &s20);
                emit_clamp_nonzero(b, &s21);
                emit_clamp_nonzero(b, &s22);

                // First step -> u1 source
                emit_mrg32k3a_step(b, &s10, &s11, &s12, &s20, &s21, &s22);
                let u1_raw = emit_mrg32k3a_output(b, &s10, &s20);

                // Second step -> u2 source
                emit_mrg32k3a_step(b, &s10, &s11, &s12, &s20, &s21, &s22);
                let u2_raw = emit_mrg32k3a_output(b, &s10, &s20);

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

                        // Box-Muller transform
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

                        // Box-Muller in f32 then widen
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
// U32 PTX generator
// ---------------------------------------------------------------------------

/// Generates PTX for a MRG32k3a raw u32 output kernel.
///
/// Each thread produces one u32 from the MRG32k3a combined output.
///
/// # Errors
///
/// Returns `PtxGenError` if kernel construction fails.
pub fn generate_mrg32k3a_u32_ptx(sm: SmVersion) -> Result<String, PtxGenError> {
    KernelBuilder::new("mrg32k3a_u32")
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

                let s10 = b.alloc_reg(PtxType::U32);
                let s11 = b.alloc_reg(PtxType::U32);
                let s12 = b.alloc_reg(PtxType::U32);
                let s20 = b.alloc_reg(PtxType::U32);
                let s21 = b.alloc_reg(PtxType::U32);
                let s22 = b.alloc_reg(PtxType::U32);

                b.raw_ptx(&format!("xor.b32 {s10}, {seed}, {gid};"));
                let scr1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr1}, {gid}, 1812433253;"));
                b.raw_ptx(&format!("xor.b32 {s11}, {seed}, {scr1};"));
                let scr2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr2}, {gid}, 1566083941;"));
                b.raw_ptx(&format!("xor.b32 {s12}, {seed}, {scr2};"));
                let scr3 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr3}, {gid}, 1103515245;"));
                b.raw_ptx(&format!("xor.b32 {s20}, {seed}, {scr3};"));
                let scr4 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr4}, {gid}, 214013;"));
                b.raw_ptx(&format!("xor.b32 {s21}, {seed}, {scr4};"));
                let scr5 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {scr5}, {gid}, 2531011;"));
                b.raw_ptx(&format!("xor.b32 {s22}, {seed}, {scr5};"));

                emit_clamp_nonzero(b, &s10);
                emit_clamp_nonzero(b, &s11);
                emit_clamp_nonzero(b, &s12);
                emit_clamp_nonzero(b, &s20);
                emit_clamp_nonzero(b, &s21);
                emit_clamp_nonzero(b, &s22);

                emit_mrg32k3a_step(b, &s10, &s11, &s12, &s20, &s21, &s22);

                let output = emit_mrg32k3a_output(b, &s10, &s20);

                let addr = b.byte_offset_addr(out_ptr, gid.clone(), 4);
                b.raw_ptx(&format!("st.global.u32 [{addr}], {output};"));
            });

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// Internal PTX helpers
// ---------------------------------------------------------------------------

/// Emits one step of the MRG32k3a generator.
///
/// Component 1: p1 = (a12 * s11 - a13n * s10) mod m1; shift (s10, s11, s12).
/// Component 2: p2 = (a21 * s22 - a23n * s20) mod m2; shift (s20, s21, s22).
///
/// We perform the computation in 64-bit to avoid overflow, then reduce mod m.
/// PTX has `mul.wide.u32` which produces a 64-bit product from two 32-bit inputs.
fn emit_mrg32k3a_step(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    s10: &oxicuda_ptx::ir::Register,
    s11: &oxicuda_ptx::ir::Register,
    s12: &oxicuda_ptx::ir::Register,
    s20: &oxicuda_ptx::ir::Register,
    s21: &oxicuda_ptx::ir::Register,
    s22: &oxicuda_ptx::ir::Register,
) {
    b.comment("MRG32k3a step - component 1");

    // Component 1: p1 = (a12 * s11 - a13n * s10) mod m1
    // Use 64-bit arithmetic to avoid overflow.
    // mul.wide.u32 gives u64 result from two u32 operands.

    let a12_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {a12_reg}, {};", A12));
    let a13n_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {a13n_reg}, {};", A13N));

    // prod1 = a12 * s11 (as u64)
    let prod1 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mul.wide.u32 {prod1}, {a12_reg}, {s11};"));

    // prod2 = a13n * s10 (as u64)
    let prod2 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mul.wide.u32 {prod2}, {a13n_reg}, {s10};"));

    // diff1 = prod1 - prod2 (could be negative, handle with modular arithmetic)
    // If prod1 >= prod2: result = (prod1 - prod2) mod m1
    // If prod1 < prod2:  result = m1 - ((prod2 - prod1) mod m1)
    let m1_64 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {m1_64}, {};", M1));

    let pred_ge1 = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ge.u64 {pred_ge1}, {prod1}, {prod2};"));

    let abs_diff1 = b.alloc_reg(PtxType::U64);
    let neg_diff1 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("sub.u64 {abs_diff1}, {prod1}, {prod2};"));
    b.raw_ptx(&format!("sub.u64 {neg_diff1}, {prod2}, {prod1};"));

    // Modular reduction: rem.u64 is not available in PTX, so we use
    // a simpler approach -- since a*s < m^2 < 2^64, we can do
    // iterative subtraction (the values are bounded).
    // Actually for PTX, we use div+mul+sub pattern.
    let q1 = b.alloc_reg(PtxType::U64);
    let r1 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("div.u64 {q1}, {abs_diff1}, {m1_64};"));
    b.raw_ptx(&format!("mul.lo.u64 {r1}, {q1}, {m1_64};"));
    b.raw_ptx(&format!("sub.u64 {r1}, {abs_diff1}, {r1};"));

    let q1n = b.alloc_reg(PtxType::U64);
    let r1n = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("div.u64 {q1n}, {neg_diff1}, {m1_64};"));
    b.raw_ptx(&format!("mul.lo.u64 {r1n}, {q1n}, {m1_64};"));
    b.raw_ptx(&format!("sub.u64 {r1n}, {neg_diff1}, {r1n};"));
    // If r1n != 0, result = m1 - r1n, else result = 0
    let adj1 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("sub.u64 {adj1}, {m1_64}, {r1n};"));
    let pred_r1n_zero = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.eq.u64 {pred_r1n_zero}, {r1n}, 0;"));
    let neg_result1 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!(
        "selp.u64 {neg_result1}, 0, {adj1}, {pred_r1n_zero};"
    ));

    let p1_64 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!(
        "selp.u64 {p1_64}, {r1}, {neg_result1}, {pred_ge1};"
    ));

    // Truncate to u32 (result is < m1 < 2^32)
    let p1 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("cvt.u32.u64 {p1}, {p1_64};"));

    // Shift component 1 state: s10 = s11, s11 = s12, s12 = p1
    b.raw_ptx(&format!("mov.u32 {s10}, {s11};"));
    b.raw_ptx(&format!("mov.u32 {s11}, {s12};"));
    b.raw_ptx(&format!("mov.u32 {s12}, {p1};"));

    b.comment("MRG32k3a step - component 2");

    // Component 2: p2 = (a21 * s22 - a23n * s20) mod m2
    let a21_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {a21_reg}, {};", A21));
    let a23n_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {a23n_reg}, {};", A23N));

    let prod3 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mul.wide.u32 {prod3}, {a21_reg}, {s22};"));

    let prod4 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mul.wide.u32 {prod4}, {a23n_reg}, {s20};"));

    let m2_64 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {m2_64}, {};", M2));

    let pred_ge2 = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ge.u64 {pred_ge2}, {prod3}, {prod4};"));

    let abs_diff2 = b.alloc_reg(PtxType::U64);
    let neg_diff2 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("sub.u64 {abs_diff2}, {prod3}, {prod4};"));
    b.raw_ptx(&format!("sub.u64 {neg_diff2}, {prod4}, {prod3};"));

    let q2 = b.alloc_reg(PtxType::U64);
    let r2 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("div.u64 {q2}, {abs_diff2}, {m2_64};"));
    b.raw_ptx(&format!("mul.lo.u64 {r2}, {q2}, {m2_64};"));
    b.raw_ptx(&format!("sub.u64 {r2}, {abs_diff2}, {r2};"));

    let q2n = b.alloc_reg(PtxType::U64);
    let r2n = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("div.u64 {q2n}, {neg_diff2}, {m2_64};"));
    b.raw_ptx(&format!("mul.lo.u64 {r2n}, {q2n}, {m2_64};"));
    b.raw_ptx(&format!("sub.u64 {r2n}, {neg_diff2}, {r2n};"));
    let adj2 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("sub.u64 {adj2}, {m2_64}, {r2n};"));
    let pred_r2n_zero = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.eq.u64 {pred_r2n_zero}, {r2n}, 0;"));
    let neg_result2 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!(
        "selp.u64 {neg_result2}, 0, {adj2}, {pred_r2n_zero};"
    ));

    let p2_64 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!(
        "selp.u64 {p2_64}, {r2}, {neg_result2}, {pred_ge2};"
    ));

    let p2 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("cvt.u32.u64 {p2}, {p2_64};"));

    // Shift component 2 state: s20 = s21, s21 = s22, s22 = p2
    b.raw_ptx(&format!("mov.u32 {s20}, {s21};"));
    b.raw_ptx(&format!("mov.u32 {s21}, {s22};"));
    b.raw_ptx(&format!("mov.u32 {s22}, {p2};"));
}

/// Emits the combined MRG32k3a output: (s12 - s22) mod m1.
///
/// Returns a u32 register containing the output value.
fn emit_mrg32k3a_output(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    s12: &oxicuda_ptx::ir::Register,
    s22: &oxicuda_ptx::ir::Register,
) -> oxicuda_ptx::ir::Register {
    let m1_const = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {m1_const}, {};", M1 as u32));

    let pred_ge = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ge.u32 {pred_ge}, {s12}, {s22};"));

    let raw_diff = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("sub.u32 {raw_diff}, {s12}, {s22};"));

    let rev_diff = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("sub.u32 {rev_diff}, {s22}, {s12};"));
    let wrapped = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("sub.u32 {wrapped}, {m1_const}, {rev_diff};"));

    let result = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "selp.u32 {result}, {raw_diff}, {wrapped}, {pred_ge};"
    ));
    result
}

// ---------------------------------------------------------------------------
// ModMatrix3x3 -- 3x3 matrix for modular arithmetic (skip-ahead)
// ---------------------------------------------------------------------------

/// A 3x3 matrix used for modular arithmetic in MRG32k3a skip-ahead.
///
/// All entries are stored as `u64` to avoid overflow during intermediate
/// multiply-accumulate steps. The final results are always reduced modulo
/// the relevant component modulus (m1 or m2).
#[derive(Debug, Clone, PartialEq, Eq)]
struct ModMatrix3x3 {
    data: [[u64; 3]; 3],
}

impl ModMatrix3x3 {
    /// Returns the 3x3 identity matrix.
    fn identity() -> Self {
        Self {
            data: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
    }

    /// Returns the zero matrix.
    fn zero() -> Self {
        Self {
            data: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }
    }

    /// Multiplies two matrices modulo `modulus`.
    ///
    /// Uses u128 intermediate products to prevent overflow since each entry
    /// can be up to `modulus - 1` (~2^32) and we accumulate 3 products.
    fn mul_mod(&self, other: &Self, modulus: u64) -> Self {
        let mut result = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                let mut acc: u128 = 0;
                for k in 0..3 {
                    acc += (self.data[i][k] as u128) * (other.data[k][j] as u128);
                }
                result.data[i][j] = (acc % (modulus as u128)) as u64;
            }
        }
        result
    }

    /// Computes `self^exp mod modulus` via repeated squaring (O(log exp) time).
    fn pow_mod(&self, mut exp: u64, modulus: u64) -> Self {
        let mut result = Self::identity();
        let mut base = self.clone();

        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul_mod(&base, modulus);
            }
            base = base.mul_mod(&base, modulus);
            exp >>= 1;
        }

        result
    }

    /// Multiplies this matrix by a column vector, reducing modulo `modulus`.
    fn mul_vec_mod(&self, vec: &[u64; 3], modulus: u64) -> [u64; 3] {
        let mut result = [0u64; 3];
        for (i, row) in self.data.iter().enumerate() {
            let mut acc: u128 = 0;
            for (col_val, vec_val) in row.iter().zip(vec.iter()) {
                acc += (*col_val as u128) * (*vec_val as u128);
            }
            result[i] = (acc % (modulus as u128)) as u64;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Mrg32k3aState -- CPU-side state for skip-ahead / stream splitting
// ---------------------------------------------------------------------------

/// CPU-side MRG32k3a state for skip-ahead and stream splitting.
///
/// This is a host-side representation of the MRG32k3a state. It allows
/// computing skip-ahead operations in O(log n) time using matrix
/// exponentiation, which is essential for parallel Monte Carlo simulations
/// where each GPU thread needs a non-overlapping subsequence.
///
/// # Example
///
/// ```rust
/// use oxicuda_rand::engines::mrg32k3a::Mrg32k3aState;
///
/// // Create a state from seed
/// let base = Mrg32k3aState::from_seed(42);
///
/// // Create independent streams for parallel MC
/// let stream0 = Mrg32k3aState::stream(42, 0);
/// let stream1 = Mrg32k3aState::stream(42, 1);
/// let stream2 = Mrg32k3aState::stream(42, 2);
///
/// // Each stream starts at position stream_id * 2^76
/// // ensuring non-overlapping subsequences.
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mrg32k3aState {
    /// Component 1 state: (s10, s11, s12).
    pub comp1: [u64; 3],
    /// Component 2 state: (s20, s21, s22).
    pub comp2: [u64; 3],
}

impl Mrg32k3aState {
    /// Creates a new state with the given component values.
    ///
    /// Values are automatically reduced modulo the respective moduli.
    /// Zero values are replaced with 1 to maintain a valid state.
    pub fn new(comp1: [u64; 3], comp2: [u64; 3]) -> Self {
        let mut s = Self { comp1, comp2 };
        // Reduce and ensure nonzero
        for v in &mut s.comp1 {
            *v %= M1;
            if *v == 0 {
                *v = 1;
            }
        }
        for v in &mut s.comp2 {
            *v %= M2;
            if *v == 0 {
                *v = 1;
            }
        }
        s
    }

    /// Creates a state from a seed value using the same scrambling as the
    /// PTX kernel (thread_id = 0).
    pub fn from_seed(seed: u64) -> Self {
        let s = seed as u32;
        // Match the PTX initialization with gid=0:
        // s10 = seed ^ 0 = seed
        // s11 = seed ^ (0 * 1812433253) = seed
        // etc.
        // For gid=0, all scrambles produce seed itself.
        let s10 = s as u64;
        let s11 = s as u64;
        let s12 = s as u64;
        let s20 = s as u64;
        let s21 = s as u64;
        let s22 = s as u64;
        Self::new([s10, s11, s12], [s20, s21, s22])
    }

    /// Performs one step of the MRG32k3a generator, advancing the state.
    ///
    /// Component 1: `p1 = (a12 * s11 - a13n * s10) mod m1`
    /// Component 2: `p2 = (a21 * s22 - a23n * s20) mod m2`
    pub fn step(&mut self) {
        // Component 1
        let prod_pos = (A12 as u128) * (self.comp1[1] as u128);
        let prod_neg = (A13N as u128) * (self.comp1[0] as u128);
        let p1 = if prod_pos >= prod_neg {
            ((prod_pos - prod_neg) % (M1 as u128)) as u64
        } else {
            let diff = ((prod_neg - prod_pos) % (M1 as u128)) as u64;
            if diff == 0 { 0 } else { M1 - diff }
        };
        self.comp1[0] = self.comp1[1];
        self.comp1[1] = self.comp1[2];
        self.comp1[2] = p1;

        // Component 2
        let prod_pos2 = (A21 as u128) * (self.comp2[2] as u128);
        let prod_neg2 = (A23N as u128) * (self.comp2[0] as u128);
        let p2 = if prod_pos2 >= prod_neg2 {
            ((prod_pos2 - prod_neg2) % (M2 as u128)) as u64
        } else {
            let diff = ((prod_neg2 - prod_pos2) % (M2 as u128)) as u64;
            if diff == 0 { 0 } else { M2 - diff }
        };
        self.comp2[0] = self.comp2[1];
        self.comp2[1] = self.comp2[2];
        self.comp2[2] = p2;
    }

    /// Returns the combined output value: `(s12 - s22) mod m1`.
    pub fn output(&self) -> u64 {
        if self.comp1[2] >= self.comp2[2] {
            self.comp1[2] - self.comp2[2]
        } else {
            M1 - (self.comp2[2] - self.comp1[2])
        }
    }

    /// Gets the transition matrix A1 for component 1.
    ///
    /// ```text
    /// A1 = | 0      1  0 |
    ///      | 0      0  1 |
    ///      | -a13n  a12 0 |
    /// ```
    ///
    /// Since we work modulo m1, `-a13n mod m1 = m1 - a13n`.
    fn transition_matrix_1() -> ModMatrix3x3 {
        ModMatrix3x3 {
            data: [[0, 1, 0], [0, 0, 1], [M1 - A13N, A12, 0]],
        }
    }

    /// Gets the transition matrix A2 for component 2.
    ///
    /// ```text
    /// A2 = | 0      1  0 |
    ///      | 0      0  1 |
    ///      | -a23n  0  a21 |
    /// ```
    ///
    /// Since we work modulo m2, `-a23n mod m2 = m2 - a23n`.
    fn transition_matrix_2() -> ModMatrix3x3 {
        ModMatrix3x3 {
            data: [[0, 1, 0], [0, 0, 1], [M2 - A23N, 0, A21]],
        }
    }

    /// Skips ahead by `n` steps in O(log n) time using matrix exponentiation.
    ///
    /// The MRG32k3a state evolves as `s_{n+1} = A * s_n mod m`. Skipping
    /// ahead by `n` steps is equivalent to computing `A^n * s_0 mod m`.
    /// This uses repeated squaring to compute `A^n` efficiently.
    pub fn skip_ahead(&mut self, n: u64) {
        if n == 0 {
            return;
        }

        // Compute A1^n mod m1 and apply to component 1
        let a1 = Self::transition_matrix_1();
        let a1n = a1.pow_mod(n, M1);
        self.comp1 = a1n.mul_vec_mod(&self.comp1, M1);

        // Compute A2^n mod m2 and apply to component 2
        let a2 = Self::transition_matrix_2();
        let a2n = a2.pow_mod(n, M2);
        self.comp2 = a2n.mul_vec_mod(&self.comp2, M2);
    }

    /// Skips ahead by `2^e` steps.
    ///
    /// This is a convenience for stream splitting where thread `i` skips
    /// by `i * 2^e` to reach its non-overlapping subsequence. The matrix
    /// power `A^(2^e)` is computed by squaring `A` exactly `e` times.
    pub fn skip_ahead_pow2(&mut self, e: u32) {
        if e == 0 {
            // Skip ahead by 2^0 = 1 step
            self.step();
            return;
        }

        // A^(2^e) = ((A^2)^2)...^2  (e squarings)
        let mut a1 = Self::transition_matrix_1();
        let mut a2 = Self::transition_matrix_2();

        for _ in 0..e {
            a1 = a1.mul_mod(&a1, M1);
            a2 = a2.mul_mod(&a2, M2);
        }

        self.comp1 = a1.mul_vec_mod(&self.comp1, M1);
        self.comp2 = a2.mul_vec_mod(&self.comp2, M2);
    }

    /// Creates a new generator state at position `stream_id * 2^76`.
    ///
    /// This is the standard stream spacing for MRG32k3a, providing each
    /// stream with a non-overlapping subsequence of length 2^76 (enough
    /// for trillions of random numbers per stream). With a total period
    /// of ~2^191, this supports up to 2^115 independent streams.
    ///
    /// # Arguments
    ///
    /// * `seed` - Base seed for the generator.
    /// * `stream_id` - Stream identifier (0-based).
    pub fn stream(seed: u64, stream_id: u64) -> Self {
        let mut state = Self::from_seed(seed);

        if stream_id == 0 {
            return state;
        }

        // Skip ahead by stream_id * 2^76.
        // Since 2^76 is too large for u64, we compute A^(2^76) first,
        // then raise that to stream_id.
        //
        // A^(stream_id * 2^76) = (A^(2^76))^stream_id
        let mut a1_base = Self::transition_matrix_1();
        let mut a2_base = Self::transition_matrix_2();

        // Compute A^(2^76) by squaring 76 times
        for _ in 0..76 {
            a1_base = a1_base.mul_mod(&a1_base, M1);
            a2_base = a2_base.mul_mod(&a2_base, M2);
        }

        // Now raise to stream_id power
        let a1_skip = a1_base.pow_mod(stream_id, M1);
        let a2_skip = a2_base.pow_mod(stream_id, M2);

        state.comp1 = a1_skip.mul_vec_mod(&state.comp1, M1);
        state.comp2 = a2_skip.mul_vec_mod(&state.comp2, M2);

        state
    }
}

// ---------------------------------------------------------------------------
// Internal PTX helpers
// ---------------------------------------------------------------------------

/// Ensures a u32 register is nonzero by OR'ing 1 when it is zero.
fn emit_clamp_nonzero(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    reg: &oxicuda_ptx::ir::Register,
) {
    let pred = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.eq.u32 {pred}, {reg}, 0;"));
    b.raw_ptx(&format!("@{pred} mov.u32 {reg}, 1;"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;

    // -----------------------------------------------------------------------
    // PTX generation tests (existing)
    // -----------------------------------------------------------------------

    #[test]
    fn generate_uniform_f32_ptx() {
        let ptx = generate_mrg32k3a_uniform_ptx(PtxType::F32, SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry mrg32k3a_uniform_f32"));
        assert!(ptx.contains("mul.wide.u32"));
    }

    #[test]
    fn generate_uniform_f64_ptx() {
        let ptx = generate_mrg32k3a_uniform_ptx(PtxType::F64, SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry mrg32k3a_uniform_f64"));
    }

    #[test]
    fn generate_normal_f32_ptx() {
        let ptx = generate_mrg32k3a_normal_ptx(PtxType::F32, SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry mrg32k3a_normal_f32"));
        assert!(ptx.contains("lg2.approx"));
        assert!(ptx.contains("cos.approx"));
    }

    #[test]
    fn generate_normal_f64_ptx() {
        let ptx = generate_mrg32k3a_normal_ptx(PtxType::F64, SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry mrg32k3a_normal_f64"));
    }

    #[test]
    fn generate_u32_ptx() {
        let ptx = generate_mrg32k3a_u32_ptx(SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry mrg32k3a_u32"));
        assert!(ptx.contains("st.global.u32"));
    }

    #[test]
    fn invalid_precision_returns_error() {
        let result = generate_mrg32k3a_uniform_ptx(PtxType::U32, SmVersion::Sm80);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // ModMatrix3x3 tests
    // -----------------------------------------------------------------------

    #[test]
    fn matrix_identity_is_identity() {
        let id = ModMatrix3x3::identity();
        assert_eq!(id.data[0], [1, 0, 0]);
        assert_eq!(id.data[1], [0, 1, 0]);
        assert_eq!(id.data[2], [0, 0, 1]);
    }

    #[test]
    fn matrix_mul_identity_left() {
        let id = ModMatrix3x3::identity();
        let a = ModMatrix3x3 {
            data: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        };
        let result = id.mul_mod(&a, M1);
        assert_eq!(result, a);
    }

    #[test]
    fn matrix_mul_identity_right() {
        let id = ModMatrix3x3::identity();
        let a = ModMatrix3x3 {
            data: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        };
        let result = a.mul_mod(&id, M1);
        assert_eq!(result, a);
    }

    #[test]
    fn matrix_mul_known_result() {
        // [[1,2,0],[0,1,0],[0,0,1]] * [[1,0,0],[3,1,0],[0,0,1]] mod large prime
        let a = ModMatrix3x3 {
            data: [[1, 2, 0], [0, 1, 0], [0, 0, 1]],
        };
        let b = ModMatrix3x3 {
            data: [[1, 0, 0], [3, 1, 0], [0, 0, 1]],
        };
        let c = a.mul_mod(&b, M1);
        // Row 0: (1*1+2*3+0, 1*0+2*1+0, 0) = (7, 2, 0)
        // Row 1: (0+3+0, 0+1+0, 0) = (3, 1, 0)
        // Row 2: (0, 0, 1)
        assert_eq!(c.data[0], [7, 2, 0]);
        assert_eq!(c.data[1], [3, 1, 0]);
        assert_eq!(c.data[2], [0, 0, 1]);
    }

    #[test]
    fn matrix_mul_mod_reduces() {
        // Check that large products are properly reduced
        let a = ModMatrix3x3 {
            data: [[M1 - 1, 0, 0], [0, 1, 0], [0, 0, 1]],
        };
        let b = ModMatrix3x3 {
            data: [[M1 - 1, 0, 0], [0, 1, 0], [0, 0, 1]],
        };
        let c = a.mul_mod(&b, M1);
        // (M1-1)^2 mod M1 = 1 (since (M1-1) = -1 mod M1, and (-1)^2 = 1)
        assert_eq!(c.data[0][0], 1);
    }

    #[test]
    fn matrix_pow_zero_is_identity() {
        let a = Mrg32k3aState::transition_matrix_1();
        let result = a.pow_mod(0, M1);
        assert_eq!(result, ModMatrix3x3::identity());
    }

    #[test]
    fn matrix_pow_one_is_self() {
        let a = Mrg32k3aState::transition_matrix_1();
        let result = a.pow_mod(1, M1);
        assert_eq!(result, a);
    }

    #[test]
    fn matrix_pow_two_equals_mul_self() {
        let a = Mrg32k3aState::transition_matrix_1();
        let squared = a.mul_mod(&a, M1);
        let pow2 = a.pow_mod(2, M1);
        assert_eq!(pow2, squared);
    }

    #[test]
    fn matrix_mul_vec_identity() {
        let id = ModMatrix3x3::identity();
        let v = [100, 200, 300];
        let result = id.mul_vec_mod(&v, M1);
        assert_eq!(result, v);
    }

    #[test]
    fn matrix_mul_vec_known() {
        let a = ModMatrix3x3 {
            data: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        };
        let v = [1, 2, 3];
        let result = a.mul_vec_mod(&v, M1);
        // Row 0: 1+4+9 = 14
        // Row 1: 4+10+18 = 32
        // Row 2: 7+16+27 = 50
        assert_eq!(result, [14, 32, 50]);
    }

    // -----------------------------------------------------------------------
    // Mrg32k3aState tests
    // -----------------------------------------------------------------------

    #[test]
    fn state_from_seed_nonzero() {
        let state = Mrg32k3aState::from_seed(0);
        // Seed 0 should produce nonzero state (clamped to 1)
        for v in &state.comp1 {
            assert!(*v > 0);
        }
        for v in &state.comp2 {
            assert!(*v > 0);
        }
    }

    #[test]
    fn state_new_clamps_modulus() {
        // Values exceeding modulus should be reduced
        let state = Mrg32k3aState::new([M1 + 5, M1 + 10, M1 + 15], [M2 + 3, M2 + 7, M2 + 11]);
        assert_eq!(state.comp1[0], 5);
        assert_eq!(state.comp1[1], 10);
        assert_eq!(state.comp1[2], 15);
        assert_eq!(state.comp2[0], 3);
        assert_eq!(state.comp2[1], 7);
        assert_eq!(state.comp2[2], 11);
    }

    #[test]
    fn skip_ahead_zero_is_identity() {
        let original = Mrg32k3aState::from_seed(42);
        let mut skipped = original.clone();
        skipped.skip_ahead(0);
        assert_eq!(original, skipped);
    }

    #[test]
    fn skip_ahead_one_matches_single_step() {
        let mut stepped = Mrg32k3aState::from_seed(42);
        stepped.step();

        let mut skipped = Mrg32k3aState::from_seed(42);
        skipped.skip_ahead(1);

        assert_eq!(stepped, skipped);
    }

    #[test]
    fn skip_ahead_n_matches_n_steps() {
        let n = 100;
        let mut stepped = Mrg32k3aState::from_seed(42);
        for _ in 0..n {
            stepped.step();
        }

        let mut skipped = Mrg32k3aState::from_seed(42);
        skipped.skip_ahead(n);

        assert_eq!(stepped, skipped);
    }

    #[test]
    fn skip_ahead_determinism() {
        let mut s1 = Mrg32k3aState::from_seed(123);
        let mut s2 = Mrg32k3aState::from_seed(123);
        s1.skip_ahead(1000);
        s2.skip_ahead(1000);
        assert_eq!(s1, s2);
    }

    #[test]
    fn skip_ahead_composable() {
        // skip_ahead(a + b) == skip_ahead(a) then skip_ahead(b)
        let mut combined = Mrg32k3aState::from_seed(42);
        combined.skip_ahead(150);

        let mut sequential = Mrg32k3aState::from_seed(42);
        sequential.skip_ahead(100);
        sequential.skip_ahead(50);

        assert_eq!(combined, sequential);
    }

    #[test]
    fn skip_ahead_large_exponent() {
        // Ensure large skip doesn't panic or produce degenerate state
        let mut state = Mrg32k3aState::from_seed(42);
        state.skip_ahead(1_000_000);

        // State should be valid (within modulus bounds, nonzero)
        for v in &state.comp1 {
            assert!(*v > 0);
            assert!(*v < M1);
        }
        for v in &state.comp2 {
            assert!(*v > 0);
            assert!(*v < M2);
        }
    }

    #[test]
    fn skip_ahead_pow2_matches_skip_ahead() {
        // skip_ahead_pow2(e) should equal skip_ahead(2^e)
        for e in 0..10 {
            let n = 1u64 << e;
            let mut via_pow2 = Mrg32k3aState::from_seed(42);
            via_pow2.skip_ahead_pow2(e);

            let mut via_skip = Mrg32k3aState::from_seed(42);
            via_skip.skip_ahead(n);

            assert_eq!(
                via_pow2, via_skip,
                "skip_ahead_pow2({e}) != skip_ahead({n})"
            );
        }
    }

    #[test]
    fn stream_zero_equals_from_seed() {
        let from_seed = Mrg32k3aState::from_seed(42);
        let stream0 = Mrg32k3aState::stream(42, 0);
        assert_eq!(from_seed, stream0);
    }

    #[test]
    fn stream_different_ids_produce_different_states() {
        let s0 = Mrg32k3aState::stream(42, 0);
        let s1 = Mrg32k3aState::stream(42, 1);
        let s2 = Mrg32k3aState::stream(42, 2);
        let s3 = Mrg32k3aState::stream(42, 3);

        assert_ne!(s0, s1);
        assert_ne!(s0, s2);
        assert_ne!(s0, s3);
        assert_ne!(s1, s2);
        assert_ne!(s1, s3);
        assert_ne!(s2, s3);
    }

    #[test]
    fn stream_different_seeds_produce_different_states() {
        let a = Mrg32k3aState::stream(1, 1);
        let b = Mrg32k3aState::stream(2, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn stream_outputs_differ() {
        // Generate one output from each of several streams and verify they differ
        let mut outputs = Vec::new();
        for id in 0..5 {
            let mut s = Mrg32k3aState::stream(42, id);
            s.step();
            outputs.push(s.output());
        }
        // All outputs should be distinct (overwhelmingly likely for a good RNG)
        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                assert_ne!(
                    outputs[i], outputs[j],
                    "stream {i} and {j} produced same output"
                );
            }
        }
    }

    #[test]
    fn transition_matrix_1_structure() {
        let a1 = Mrg32k3aState::transition_matrix_1();
        // Row 0: [0, 1, 0]
        assert_eq!(a1.data[0], [0, 1, 0]);
        // Row 1: [0, 0, 1]
        assert_eq!(a1.data[1], [0, 0, 1]);
        // Row 2: [m1 - a13n, a12, 0]
        assert_eq!(a1.data[2][0], M1 - A13N);
        assert_eq!(a1.data[2][1], A12);
        assert_eq!(a1.data[2][2], 0);
    }

    #[test]
    fn transition_matrix_2_structure() {
        let a2 = Mrg32k3aState::transition_matrix_2();
        // Row 0: [0, 1, 0]
        assert_eq!(a2.data[0], [0, 1, 0]);
        // Row 1: [0, 0, 1]
        assert_eq!(a2.data[1], [0, 0, 1]);
        // Row 2: [m2 - a23n, 0, a21]
        assert_eq!(a2.data[2][0], M2 - A23N);
        assert_eq!(a2.data[2][1], 0);
        assert_eq!(a2.data[2][2], A21);
    }

    #[test]
    fn transition_matrix_1_step_matches_manual_step() {
        // Verify that multiplying A1 by state vector gives the same result as step()
        let state = Mrg32k3aState::new([100, 200, 300], [400, 500, 600]);

        // Manual step for component 1
        let mut stepped = state.clone();
        stepped.step();

        // Matrix multiplication
        let a1 = Mrg32k3aState::transition_matrix_1();
        let new_comp1 = a1.mul_vec_mod(&state.comp1, M1);

        assert_eq!(new_comp1, stepped.comp1);
    }

    #[test]
    fn transition_matrix_2_step_matches_manual_step() {
        let state = Mrg32k3aState::new([100, 200, 300], [400, 500, 600]);

        let mut stepped = state.clone();
        stepped.step();

        let a2 = Mrg32k3aState::transition_matrix_2();
        let new_comp2 = a2.mul_vec_mod(&state.comp2, M2);

        assert_eq!(new_comp2, stepped.comp2);
    }

    #[test]
    fn output_within_valid_range() {
        let mut state = Mrg32k3aState::from_seed(42);
        for _ in 0..100 {
            state.step();
            let out = state.output();
            assert!(out < M1, "output {out} exceeds m1");
        }
    }

    #[test]
    fn step_changes_state() {
        let mut state = Mrg32k3aState::from_seed(42);
        let before = state.clone();
        state.step();
        assert_ne!(before, state);
    }

    // -----------------------------------------------------------------------
    // Quality gate: skip-ahead verified correct for parallel reproducible simulation
    // -----------------------------------------------------------------------

    /// Generate 110 values sequentially, then reset and skip ahead 100 steps.
    /// The 101st sequential output value must equal the value generated after skip.
    #[test]
    fn test_mrg32k3a_skip_ahead_output_reproducible() {
        // Generate 110 values sequentially from seed 12345
        let mut rng1 = Mrg32k3aState::from_seed(12345);
        let mut sequential_outputs: Vec<u64> = Vec::with_capacity(110);
        for _ in 0..110 {
            rng1.step();
            sequential_outputs.push(rng1.output());
        }

        // Reset to initial seed and skip ahead by 100 steps
        let mut rng2 = Mrg32k3aState::from_seed(12345);
        rng2.skip_ahead(100);
        // Now take one step and get output — should match sequential_outputs[100]
        rng2.step();
        let value_after_skip = rng2.output();

        assert_eq!(
            value_after_skip, sequential_outputs[100],
            "Skip-ahead by 100 then step should give same output as sequential[100]: \
             skip={value_after_skip}, seq={}",
            sequential_outputs[100],
        );
    }

    /// Two streams with different subsequence offsets must produce statistically
    /// independent (non-identical) sequences.
    #[test]
    fn test_mrg32k3a_parallel_streams_independent() {
        // Stream 0: starts at seed position 0
        // Stream 1: starts at seed position 0, then skipped 1000 steps
        let mut rng1 = Mrg32k3aState::from_seed(42);
        let mut rng2 = Mrg32k3aState::stream(42, 1);

        // Collect first 100 output values from each stream
        let v1: Vec<u64> = (0..100)
            .map(|_| {
                rng1.step();
                rng1.output()
            })
            .collect();
        let v2: Vec<u64> = (0..100)
            .map(|_| {
                rng2.step();
                rng2.output()
            })
            .collect();

        // Streams should NOT produce identical values (very unlikely for a correct RNG)
        let same_count = v1.iter().zip(&v2).filter(|(a, b)| a == b).count();
        assert!(
            same_count < 10,
            "Parallel streams appear correlated: {same_count}/100 values identical"
        );
    }

    /// Verify that skip-ahead(N) followed by skip-ahead(M) equals skip-ahead(N+M).
    #[test]
    fn test_mrg32k3a_skip_ahead_composability_output() {
        let seed = 99_u64;
        let n = 50_u64;
        let m = 75_u64;

        // Combined skip
        let mut combined = Mrg32k3aState::from_seed(seed);
        combined.skip_ahead(n + m);
        combined.step();
        let combined_output = combined.output();

        // Sequential skips
        let mut sequential = Mrg32k3aState::from_seed(seed);
        sequential.skip_ahead(n);
        sequential.skip_ahead(m);
        sequential.step();
        let sequential_output = sequential.output();

        assert_eq!(
            combined_output,
            sequential_output,
            "skip_ahead({}) + skip_ahead({}) should equal skip_ahead({}): \
             combined={combined_output}, sequential={sequential_output}",
            n,
            m,
            n + m,
        );
    }

    /// Determinism: same seed always produces the same skip-ahead result.
    #[test]
    fn test_mrg32k3a_skip_ahead_determinism_output() {
        let mut s1 = Mrg32k3aState::from_seed(777);
        let mut s2 = Mrg32k3aState::from_seed(777);
        s1.skip_ahead(500);
        s2.skip_ahead(500);
        s1.step();
        s2.step();
        assert_eq!(
            s1.output(),
            s2.output(),
            "Same seed and skip must give identical output"
        );
    }

    // -----------------------------------------------------------------------
    // Quality gate: skip-ahead with 10 sequential values
    // -----------------------------------------------------------------------

    /// Generate 10 values after skip_ahead(100) must match the same values
    /// generated by stepping sequentially 100 + 10 times from the same seed.
    ///
    /// This verifies that skip_ahead does not corrupt state beyond the skip
    /// point — the generator continues correctly for subsequent steps.
    #[test]
    fn test_mrg32k3a_skip_ahead_generate_10_values() {
        // Sequential path: step 110 times, record last 10
        let mut seq = Mrg32k3aState::from_seed(31415);
        let mut sequential_values = [0_u64; 10];
        for _ in 0..100 {
            seq.step();
        }
        for slot in &mut sequential_values {
            seq.step();
            *slot = seq.output();
        }

        // Skip-ahead path: skip 100, then step 10 times
        let mut skip = Mrg32k3aState::from_seed(31415);
        skip.skip_ahead(100);
        let mut skip_values = [0_u64; 10];
        for slot in &mut skip_values {
            skip.step();
            *slot = skip.output();
        }

        for (i, (s, q)) in sequential_values.iter().zip(skip_values.iter()).enumerate() {
            assert_eq!(
                s, q,
                "Value {i} after skip_ahead(100): sequential={s} != skip={q}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Quality gate: four-stream pairwise independence
    // -----------------------------------------------------------------------

    /// Four streams created with stream(seed, 0..3) must produce statistically
    /// independent sequences.
    ///
    /// Pairwise correlation (fraction of identical values) across 100 samples
    /// must be < 10% for any pair.  This is a necessary (but not sufficient)
    /// condition for statistical independence.
    #[test]
    fn test_mrg32k3a_four_stream_independence() {
        let seed = 12_345_678_u64;
        let mut streams: Vec<Mrg32k3aState> =
            (0..4).map(|id| Mrg32k3aState::stream(seed, id)).collect();

        let samples = 100_usize;
        let values: Vec<Vec<u64>> = streams
            .iter_mut()
            .map(|s| {
                (0..samples)
                    .map(|_| {
                        s.step();
                        s.output()
                    })
                    .collect()
            })
            .collect();

        // Check pairwise: streams i and j must share < 10 identical values
        for i in 0..4 {
            for j in (i + 1)..4 {
                let same_count = values[i]
                    .iter()
                    .zip(&values[j])
                    .filter(|(a, b)| a == b)
                    .count();
                assert!(
                    same_count < 10,
                    "Streams {i} and {j} share {same_count}/{samples} identical values — \
                     streams appear correlated"
                );
            }
        }

        // Also verify all four streams start at different initial outputs
        let first_outputs: Vec<u64> = values.iter().map(|v| v[0]).collect();
        let unique_first: std::collections::HashSet<u64> = first_outputs.iter().cloned().collect();
        assert_eq!(
            unique_first.len(),
            4,
            "All four stream initial outputs must be distinct: {first_outputs:?}"
        );
    }
}
