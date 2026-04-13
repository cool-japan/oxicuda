//! Optimized Philox-4x32-10 engine that generates 4 values per thread.
//!
//! The standard Philox engine uses one thread per output value, leaving 3 of
//! the 4 counter outputs unused. This optimized variant exploits the full
//! 128-bit Philox output:
//!
//! 1. Each thread produces 4 random `u32` values per Philox round.
//! 2. Output stored as 4 consecutive `f32` values for coalesced writes.
//! 3. Grid-stride loop for handling arbitrary output sizes.
//! 4. Box-Muller applied pair-wise for normal distribution (2 pairs = 4 normals).
//!
//! Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (SC 2011)

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::error::PtxGenError;
use oxicuda_ptx::ir::PtxType;

// ---------------------------------------------------------------------------
// Philox constants (same as base engine)
// ---------------------------------------------------------------------------

/// Philox multiplier constant 0.
const PHILOX_M4X32_0: u32 = 0xD251_1F53;

/// Philox multiplier constant 1.
const PHILOX_M4X32_1: u32 = 0xCD9E_8D57;

/// Philox Weyl sequence constant 0.
const PHILOX_W32_0: u32 = 0x9E37_79B9;

/// Philox Weyl sequence constant 1.
const PHILOX_W32_1: u32 = 0xBB67_AE85;

/// Threshold for using the optimized 4-per-thread engine.
/// Below this count, the standard engine is more efficient.
pub const OPTIMIZED_THRESHOLD: usize = 1024;

// ---------------------------------------------------------------------------
// Optimized uniform f32 kernel
// ---------------------------------------------------------------------------

/// Generates PTX for an optimized Philox-4x32-10 uniform f32 kernel.
///
/// Each thread computes 4 float values using the full Philox-4x32-10 output.
/// A grid-stride loop handles arbitrary output sizes. The kernel takes:
/// `(out_ptr: u64, n: u32, seed_lo: u32, seed_hi: u32, offset_lo: u32, offset_hi: u32)`.
///
/// # Errors
///
/// Returns `PtxGenError` if kernel construction fails.
pub fn generate_philox_optimized_uniform_f32_ptx(sm: SmVersion) -> Result<String, PtxGenError> {
    KernelBuilder::new("philox_optimized_uniform_f32")
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("seed_lo", PtxType::U32)
        .param("seed_hi", PtxType::U32)
        .param("offset_lo", PtxType::U32)
        .param("offset_hi", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            // Compute global thread id and grid stride
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");
            let out_ptr = b.load_param_u64("out_ptr");
            let seed_lo = b.load_param_u32("seed_lo");
            let seed_hi = b.load_param_u32("seed_hi");
            let offset_lo = b.load_param_u32("offset_lo");
            let offset_hi = b.load_param_u32("offset_hi");

            // Compute grid stride = gridDim.x * blockDim.x
            let ntid = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {ntid}, %ntid.x;"));
            let nctaid = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {nctaid}, %nctaid.x;"));
            let grid_stride = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mul.lo.u32 {grid_stride}, {ntid}, {nctaid};"));

            // n_div4 = (n + 3) / 4  -- number of Philox invocations needed
            let n_plus3 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("add.u32 {n_plus3}, {n_reg}, 3;"));
            let n_div4 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {n_div4}, {n_plus3}, 2;"));

            // Grid-stride loop: for idx = gid; idx < n_div4; idx += grid_stride
            let idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {idx}, {gid};"));

            let loop_label = b.fresh_label("opt_loop");
            let done_label = b.fresh_label("opt_done");

            b.label(&loop_label);

            // Check loop condition: idx < n_div4
            let pred_loop = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lo.u32 {pred_loop}, {idx}, {n_div4};"));
            b.raw_ptx(&format!("@!{pred_loop} bra {done_label};"));

            // counter = idx + offset (as 64-bit)
            b.comment("Compute counter = idx + offset");
            let idx_hi = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {idx_hi}, 0;"));
            let ctr_lo = b.alloc_reg(PtxType::U32);
            let ctr_hi = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("add.cc.u32 {ctr_lo}, {idx}, {offset_lo};"));
            b.raw_ptx(&format!("addc.u32 {ctr_hi}, {idx_hi}, {offset_hi};"));

            // Philox state: c0 = ctr_lo, c1 = ctr_hi, c2 = 0, c3 = 0
            let c0 = b.alloc_reg(PtxType::U32);
            let c1 = b.alloc_reg(PtxType::U32);
            let c2 = b.alloc_reg(PtxType::U32);
            let c3 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {c0}, {ctr_lo};"));
            b.raw_ptx(&format!("mov.u32 {c1}, {ctr_hi};"));
            b.raw_ptx(&format!("mov.u32 {c2}, 0;"));
            b.raw_ptx(&format!("mov.u32 {c3}, 0;"));

            // Key from seed
            let k0 = b.alloc_reg(PtxType::U32);
            let k1 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {k0}, {seed_lo};"));
            b.raw_ptx(&format!("mov.u32 {k1}, {seed_hi};"));

            // 10 rounds of Philox mixing
            emit_philox_10rounds_inline(b, &c0, &c1, &c2, &c3, &k0, &k1);

            // Convert all 4 outputs to f32 in [0, 1)
            b.comment("Convert 4 u32 outputs to f32 in [0, 1)");
            let scale = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;")); // 2^-32

            let f0 = b.alloc_reg(PtxType::F32);
            let f0_raw = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("cvt.rn.f32.u32 {f0_raw}, {c0};"));
            b.raw_ptx(&format!("mul.rn.f32 {f0}, {f0_raw}, {scale};"));

            let f1 = b.alloc_reg(PtxType::F32);
            let f1_raw = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("cvt.rn.f32.u32 {f1_raw}, {c1};"));
            b.raw_ptx(&format!("mul.rn.f32 {f1}, {f1_raw}, {scale};"));

            let f2 = b.alloc_reg(PtxType::F32);
            let f2_raw = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("cvt.rn.f32.u32 {f2_raw}, {c2};"));
            b.raw_ptx(&format!("mul.rn.f32 {f2}, {f2_raw}, {scale};"));

            let f3 = b.alloc_reg(PtxType::F32);
            let f3_raw = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("cvt.rn.f32.u32 {f3_raw}, {c3};"));
            b.raw_ptx(&format!("mul.rn.f32 {f3}, {f3_raw}, {scale};"));

            // Compute base output index = idx * 4
            let out_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shl.b32 {out_idx}, {idx}, 2;"));

            // Store up to 4 values, checking bounds for each
            b.comment("Store 4 values with bounds checking");
            emit_bounded_store_f32(b, &out_ptr, &out_idx, &n_reg, &f0, 0);
            emit_bounded_store_f32(b, &out_ptr, &out_idx, &n_reg, &f1, 1);
            emit_bounded_store_f32(b, &out_ptr, &out_idx, &n_reg, &f2, 2);
            emit_bounded_store_f32(b, &out_ptr, &out_idx, &n_reg, &f3, 3);

            // Advance loop index
            b.raw_ptx(&format!("add.u32 {idx}, {idx}, {grid_stride};"));
            b.raw_ptx(&format!("bra {loop_label};"));
            b.label(&done_label);

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// Optimized normal f32 kernel (Box-Muller, 4 values per thread)
// ---------------------------------------------------------------------------

/// Generates PTX for an optimized Philox-4x32-10 normal f32 kernel.
///
/// Each thread generates 4 normal values using two Box-Muller transforms:
/// - Pair (c0, c1) -> (n0, n1) via Box-Muller
/// - Pair (c2, c3) -> (n2, n3) via Box-Muller
///
/// The kernel takes:
/// `(out_ptr: u64, n: u32, seed_lo: u32, seed_hi: u32, offset_lo: u32, offset_hi: u32, mean: f32, stddev: f32)`.
///
/// # Errors
///
/// Returns `PtxGenError` if kernel construction fails.
pub fn generate_philox_optimized_normal_f32_ptx(sm: SmVersion) -> Result<String, PtxGenError> {
    KernelBuilder::new("philox_optimized_normal_f32")
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("seed_lo", PtxType::U32)
        .param("seed_hi", PtxType::U32)
        .param("offset_lo", PtxType::U32)
        .param("offset_hi", PtxType::U32)
        .param("mean", PtxType::F32)
        .param("stddev", PtxType::F32)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");
            let out_ptr = b.load_param_u64("out_ptr");
            let seed_lo = b.load_param_u32("seed_lo");
            let seed_hi = b.load_param_u32("seed_hi");
            let offset_lo = b.load_param_u32("offset_lo");
            let offset_hi = b.load_param_u32("offset_hi");
            let mean_reg = b.load_param_f32("mean");
            let stddev_reg = b.load_param_f32("stddev");

            // Grid stride
            let ntid = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {ntid}, %ntid.x;"));
            let nctaid = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {nctaid}, %nctaid.x;"));
            let grid_stride = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mul.lo.u32 {grid_stride}, {ntid}, {nctaid};"));

            // n_div4
            let n_plus3 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("add.u32 {n_plus3}, {n_reg}, 3;"));
            let n_div4 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {n_div4}, {n_plus3}, 2;"));

            let idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {idx}, {gid};"));

            let loop_label = b.fresh_label("nopt_loop");
            let done_label = b.fresh_label("nopt_done");

            b.label(&loop_label);
            let pred_loop = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lo.u32 {pred_loop}, {idx}, {n_div4};"));
            b.raw_ptx(&format!("@!{pred_loop} bra {done_label};"));

            // Philox counter computation
            let idx_hi = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {idx_hi}, 0;"));
            let ctr_lo = b.alloc_reg(PtxType::U32);
            let ctr_hi = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("add.cc.u32 {ctr_lo}, {idx}, {offset_lo};"));
            b.raw_ptx(&format!("addc.u32 {ctr_hi}, {idx_hi}, {offset_hi};"));

            let c0 = b.alloc_reg(PtxType::U32);
            let c1 = b.alloc_reg(PtxType::U32);
            let c2 = b.alloc_reg(PtxType::U32);
            let c3 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {c0}, {ctr_lo};"));
            b.raw_ptx(&format!("mov.u32 {c1}, {ctr_hi};"));
            b.raw_ptx(&format!("mov.u32 {c2}, 0;"));
            b.raw_ptx(&format!("mov.u32 {c3}, 0;"));

            let k0 = b.alloc_reg(PtxType::U32);
            let k1 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {k0}, {seed_lo};"));
            b.raw_ptx(&format!("mov.u32 {k1}, {seed_hi};"));

            emit_philox_10rounds_inline(b, &c0, &c1, &c2, &c3, &k0, &k1);

            // Convert to uniform f32 in [0, 1)
            let scale = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;"));

            let u0 = emit_u32_to_f32(b, &c0, &scale);
            let u1 = emit_u32_to_f32(b, &c1, &scale);
            let u2 = emit_u32_to_f32(b, &c2, &scale);
            let u3 = emit_u32_to_f32(b, &c3, &scale);

            // Box-Muller pair 1: (u0, u1) -> (n0, n1)
            b.comment("Box-Muller transform pair 1: (u0, u1) -> (n0, n1)");
            let (n0, n1) = emit_box_muller_f32_inline(b, &u0, &u1);

            // Box-Muller pair 2: (u2, u3) -> (n2, n3)
            b.comment("Box-Muller transform pair 2: (u2, u3) -> (n2, n3)");
            let (n2, n3) = emit_box_muller_f32_inline(b, &u2, &u3);

            // Scale: result = mean + stddev * z
            let r0 = b.fma_f32(stddev_reg.clone(), n0, mean_reg.clone());
            let r1 = b.fma_f32(stddev_reg.clone(), n1, mean_reg.clone());
            let r2 = b.fma_f32(stddev_reg.clone(), n2, mean_reg.clone());
            let r3 = b.fma_f32(stddev_reg, n3, mean_reg);

            // Store with bounds checking
            let out_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shl.b32 {out_idx}, {idx}, 2;"));

            emit_bounded_store_f32(b, &out_ptr, &out_idx, &n_reg, &r0, 0);
            emit_bounded_store_f32(b, &out_ptr, &out_idx, &n_reg, &r1, 1);
            emit_bounded_store_f32(b, &out_ptr, &out_idx, &n_reg, &r2, 2);
            emit_bounded_store_f32(b, &out_ptr, &out_idx, &n_reg, &r3, 3);

            // Advance loop
            b.raw_ptx(&format!("add.u32 {idx}, {idx}, {grid_stride};"));
            b.raw_ptx(&format!("bra {loop_label};"));
            b.label(&done_label);

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// Internal PTX helpers
// ---------------------------------------------------------------------------

/// Emits 10 rounds of Philox-4x32 mixing (inline, for the optimized engine).
///
/// Modifies `c0..c3` in place and bumps key `k0, k1` after each round.
fn emit_philox_10rounds_inline(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    c0: &oxicuda_ptx::ir::Register,
    c1: &oxicuda_ptx::ir::Register,
    c2: &oxicuda_ptx::ir::Register,
    c3: &oxicuda_ptx::ir::Register,
    k0: &oxicuda_ptx::ir::Register,
    k1: &oxicuda_ptx::ir::Register,
) {
    let hi0 = b.alloc_reg(PtxType::U32);
    let lo0 = b.alloc_reg(PtxType::U32);
    let hi1 = b.alloc_reg(PtxType::U32);
    let lo1 = b.alloc_reg(PtxType::U32);
    let t0 = b.alloc_reg(PtxType::U32);
    let t1 = b.alloc_reg(PtxType::U32);
    let t2 = b.alloc_reg(PtxType::U32);
    let t3 = b.alloc_reg(PtxType::U32);

    b.unroll(10, |b, round| {
        b.comment(&format!("Philox round {round}"));

        // hi0, lo0 = mulhilo(M4x32_0, c0)
        b.raw_ptx(&format!("mul.hi.u32 {hi0}, {c0}, {PHILOX_M4X32_0};"));
        b.raw_ptx(&format!("mul.lo.u32 {lo0}, {c0}, {PHILOX_M4X32_0};"));

        // hi1, lo1 = mulhilo(M4x32_1, c2)
        b.raw_ptx(&format!("mul.hi.u32 {hi1}, {c2}, {PHILOX_M4X32_1};"));
        b.raw_ptx(&format!("mul.lo.u32 {lo1}, {c2}, {PHILOX_M4X32_1};"));

        // New counter values:
        // t0 = hi1 ^ c1 ^ k0
        b.raw_ptx(&format!("xor.b32 {t0}, {hi1}, {c1};"));
        b.raw_ptx(&format!("xor.b32 {t0}, {t0}, {k0};"));
        // t1 = lo1
        b.raw_ptx(&format!("mov.u32 {t1}, {lo1};"));
        // t2 = hi0 ^ c3 ^ k1
        b.raw_ptx(&format!("xor.b32 {t2}, {hi0}, {c3};"));
        b.raw_ptx(&format!("xor.b32 {t2}, {t2}, {k1};"));
        // t3 = lo0
        b.raw_ptx(&format!("mov.u32 {t3}, {lo0};"));

        // Write back
        b.raw_ptx(&format!("mov.u32 {c0}, {t0};"));
        b.raw_ptx(&format!("mov.u32 {c1}, {t1};"));
        b.raw_ptx(&format!("mov.u32 {c2}, {t2};"));
        b.raw_ptx(&format!("mov.u32 {c3}, {t3};"));

        // Bump key
        b.raw_ptx(&format!("add.u32 {k0}, {k0}, {PHILOX_W32_0};"));
        b.raw_ptx(&format!("add.u32 {k1}, {k1}, {PHILOX_W32_1};"));
    });
}

/// Converts a u32 register to f32 in [0, 1) using the pre-loaded scale factor.
fn emit_u32_to_f32(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    val: &oxicuda_ptx::ir::Register,
    scale: &oxicuda_ptx::ir::Register,
) -> oxicuda_ptx::ir::Register {
    let raw = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("cvt.rn.f32.u32 {raw}, {val};"));
    let result = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mul.rn.f32 {result}, {raw}, {scale};"));
    result
}

/// Emits an inline Box-Muller transform from two uniform f32 values.
///
/// Returns `(z0, z1)` — two standard normal samples.
fn emit_box_muller_f32_inline(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    u1: &oxicuda_ptx::ir::Register,
    u2: &oxicuda_ptx::ir::Register,
) -> (oxicuda_ptx::ir::Register, oxicuda_ptx::ir::Register) {
    // Clamp u1 away from zero
    let eps = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {eps}, 0f33800000;")); // ~5.96e-8
    let u1_safe = b.max_f32(u1.clone(), eps);

    // ln(u1) = lg2(u1) * ln(2)
    let lg2_u1 = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("lg2.approx.f32 {lg2_u1}, {u1_safe};"));
    let ln2 = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {ln2}, 0f3F317218;")); // ln(2)
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
    b.raw_ptx(&format!("mov.f32 {two_pi}, 0f40C90FDB;")); // 2*pi
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

/// Emits a bounds-checked store of a single f32 value.
///
/// Stores `val` at `out_ptr + (base_idx + element_offset) * 4` if the
/// computed index is less than `n`.
fn emit_bounded_store_f32(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    out_ptr: &oxicuda_ptx::ir::Register,
    base_idx: &oxicuda_ptx::ir::Register,
    n: &oxicuda_ptx::ir::Register,
    val: &oxicuda_ptx::ir::Register,
    element_offset: u32,
) {
    let store_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "add.u32 {store_idx}, {base_idx}, {element_offset};"
    ));
    let pred = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred}, {store_idx}, {n};"));
    // Compute address: out_ptr + store_idx * 4
    let idx64 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {store_idx};"));
    let byte_off = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mul.lo.u64 {byte_off}, {idx64}, 4;"));
    let addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("add.u64 {addr}, {out_ptr}, {byte_off};"));
    b.raw_ptx(&format!("@{pred} st.global.f32 [{addr}], {val};"));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;

    #[test]
    fn generate_optimized_uniform_f32_ptx_valid() {
        let ptx = generate_philox_optimized_uniform_f32_ptx(SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry philox_optimized_uniform_f32"));
        assert!(ptx.contains("mul.hi.u32"));
        assert!(ptx.contains("xor.b32"));
        assert!(ptx.contains("st.global.f32"));
    }

    #[test]
    fn generate_optimized_uniform_contains_grid_stride() {
        let ptx = generate_philox_optimized_uniform_f32_ptx(SmVersion::Sm80)
            .expect("should generate PTX");
        // Grid-stride loop reads nctaid.x
        assert!(ptx.contains("%nctaid.x"));
        // Has loop structure with branch
        assert!(ptx.contains("bra"));
    }

    #[test]
    fn generate_optimized_uniform_has_bounds_checking() {
        let ptx = generate_philox_optimized_uniform_f32_ptx(SmVersion::Sm80)
            .expect("should generate PTX");
        // Bounds checking uses setp.lo.u32 for each element
        let setp_count = ptx.matches("setp.lo.u32").count();
        // At least 5: 1 for loop condition + 4 for stores
        assert!(
            setp_count >= 5,
            "expected >= 5 setp.lo.u32 instructions, found {setp_count}"
        );
    }

    #[test]
    fn generate_optimized_normal_f32_ptx_valid() {
        let ptx = generate_philox_optimized_normal_f32_ptx(SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry philox_optimized_normal_f32"));
        assert!(ptx.contains("lg2.approx.f32"));
        assert!(ptx.contains("cos.approx.f32"));
        assert!(ptx.contains("sin.approx.f32"));
        assert!(ptx.contains("sqrt.approx.f32"));
    }

    #[test]
    fn generate_optimized_normal_has_two_box_muller_pairs() {
        let ptx =
            generate_philox_optimized_normal_f32_ptx(SmVersion::Sm80).expect("should generate PTX");
        // Two Box-Muller transforms: 2 cos + 2 sin
        let cos_count = ptx.matches("cos.approx.f32").count();
        let sin_count = ptx.matches("sin.approx.f32").count();
        assert_eq!(cos_count, 2, "expected 2 cos.approx.f32, found {cos_count}");
        assert_eq!(sin_count, 2, "expected 2 sin.approx.f32, found {sin_count}");
    }

    #[test]
    fn generate_optimized_normal_has_fma_for_scaling() {
        let ptx =
            generate_philox_optimized_normal_f32_ptx(SmVersion::Sm80).expect("should generate PTX");
        // 4 fma instructions for mean + stddev * z
        let fma_count = ptx.matches("fma.rn.f32").count();
        assert!(
            fma_count >= 4,
            "expected >= 4 fma.rn.f32 instructions, found {fma_count}"
        );
    }

    #[test]
    fn generate_optimized_uniform_sm75() {
        let ptx = generate_philox_optimized_uniform_f32_ptx(SmVersion::Sm75);
        let ptx = ptx.expect("should generate for sm_75");
        assert!(ptx.contains(".target sm_75"));
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn optimized_threshold_is_reasonable() {
        assert!(
            OPTIMIZED_THRESHOLD >= 256,
            "threshold should be at least one block"
        );
        assert!(
            OPTIMIZED_THRESHOLD <= 65536,
            "threshold should not be too large"
        );
    }
}
