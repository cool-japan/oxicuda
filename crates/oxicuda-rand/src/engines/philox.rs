//! Philox-4x32-10 counter-based PRNG engine.
//!
//! Philox is the default RNG engine in cuRAND. It is a counter-based PRNG that
//! produces 4 u32 values (128 bits) per call. The algorithm performs 10 rounds
//! of a substitution-permutation network using multiply-high as the bijection.
//!
//! Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (SC 2011)

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::error::PtxGenError;
use oxicuda_ptx::ir::PtxType;

// ---------------------------------------------------------------------------
// Philox constants
// ---------------------------------------------------------------------------

/// Philox multiplier constant 0 (low word).
const PHILOX_M4X32_0: u32 = 0xD251_1F53;

/// Philox multiplier constant 1 (high word).
const PHILOX_M4X32_1: u32 = 0xCD9E_8D57;

/// Philox Weyl sequence constant 0.
const PHILOX_W32_0: u32 = 0x9E37_79B9;

/// Philox Weyl sequence constant 1.
const PHILOX_W32_1: u32 = 0xBB67_AE85;

// ---------------------------------------------------------------------------
// Uniform distribution PTX generators
// ---------------------------------------------------------------------------

/// Generates PTX for a Philox-4x32-10 uniform distribution kernel.
///
/// Each thread computes one float value from the Philox output.
/// The kernel takes: `(out_ptr: u64, n: u32, seed_lo: u32, seed_hi: u32, offset: u64)`.
///
/// # Errors
///
/// Returns `PtxGenError` if kernel construction fails.
pub fn generate_philox_uniform_ptx(
    precision: PtxType,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    let kernel_name = match precision {
        PtxType::F32 => "philox_uniform_f32",
        PtxType::F64 => "philox_uniform_f64",
        _ => return Err(PtxGenError::InvalidType(format!("{precision:?}"))),
    };

    let stride_bytes: u32 = precision.size_bytes() as u32;

    KernelBuilder::new(kernel_name)
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("seed_lo", PtxType::U32)
        .param("seed_hi", PtxType::U32)
        .param("offset_lo", PtxType::U32)
        .param("offset_hi", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let seed_lo = b.load_param_u32("seed_lo");
                let seed_hi = b.load_param_u32("seed_hi");
                let offset_lo = b.load_param_u32("offset_lo");
                let offset_hi = b.load_param_u32("offset_hi");

                // counter = gid + offset (as u64, split into lo/hi)
                b.comment("Compute counter = gid + offset");
                let gid_lo = gid;
                let gid_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {gid_hi}, 0;"));

                // 64-bit add: (gid_lo, gid_hi) + (offset_lo, offset_hi)
                let ctr_lo = b.alloc_reg(PtxType::U32);
                let ctr_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.cc.u32 {ctr_lo}, {gid_lo}, {offset_lo};"));
                b.raw_ptx(&format!("addc.u32 {ctr_hi}, {gid_hi}, {offset_hi};"));

                // Philox state: counter words c0,c1,c2,c3 and key words k0,k1
                // c0 = ctr_lo, c1 = ctr_hi, c2 = 0, c3 = 0
                let c0 = ctr_lo;
                let c1 = ctr_hi;
                let c2 = b.alloc_reg(PtxType::U32);
                let c3 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {c2}, 0;"));
                b.raw_ptx(&format!("mov.u32 {c3}, 0;"));

                let k0 = seed_lo;
                let k1 = seed_hi;

                // 10 rounds of Philox mixing
                emit_philox_10rounds(b, &c0, &c1, &c2, &c3, &k0, &k1);

                // Convert c0 to float [0,1)
                match precision {
                    PtxType::F32 => {
                        let fval = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {fval}, {c0};"));
                        let scale = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;")); // 2^-32 as hex
                        // Actually: 1.0 / 4294967296.0 = 2.328306e-10
                        // Hex float for 2^-32 = 0x2F800000
                        let result = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {result}, {fval}, {scale};"));
                        let addr = b.byte_offset_addr(out_ptr, gid_lo.clone(), stride_bytes);
                        b.store_global_f32(addr, result);
                    }
                    PtxType::F64 => {
                        // Combine c0 and c1 for 53-bit precision
                        let fval_lo = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("cvt.rn.f64.u32 {fval_lo}, {c0};"));
                        let fval_hi = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("cvt.rn.f64.u32 {fval_hi}, {c1};"));
                        // result = (c1 * 2^32 + c0) * 2^-64
                        let scale_hi = b.alloc_reg(PtxType::F64);
                        // 2^-32 in f64 hex: 0x3DF0000000000000
                        b.raw_ptx(&format!("mov.f64 {scale_hi}, 0d3DF0000000000000;"));
                        let scale_lo = b.alloc_reg(PtxType::F64);
                        // 2^-64 in f64 hex: 0x3BF0000000000000
                        b.raw_ptx(&format!("mov.f64 {scale_lo}, 0d3BF0000000000000;"));
                        let part_hi = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {part_hi}, {fval_hi}, {scale_hi};"));
                        let result = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!(
                            "fma.rn.f64 {result}, {fval_lo}, {scale_lo}, {part_hi};"
                        ));
                        let addr = b.byte_offset_addr(out_ptr, gid_lo.clone(), stride_bytes);
                        b.store_global_f64(addr, result);
                    }
                    _ => {}
                }
            });

            b.ret();
        })
        .build()
}

/// Generates PTX for a Philox-4x32-10 normal distribution kernel (Box-Muller).
///
/// Each thread generates a pair of uniform values and applies the Box-Muller
/// transform to produce one normal value. The kernel takes:
/// `(out_ptr: u64, n: u32, seed_lo: u32, seed_hi: u32, offset_lo: u32, offset_hi: u32, mean: f32/f64, stddev: f32/f64)`.
///
/// # Errors
///
/// Returns `PtxGenError` if kernel construction fails.
pub fn generate_philox_normal_ptx(
    precision: PtxType,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    let kernel_name = match precision {
        PtxType::F32 => "philox_normal_f32",
        PtxType::F64 => "philox_normal_f64",
        _ => return Err(PtxGenError::InvalidType(format!("{precision:?}"))),
    };

    let stride_bytes: u32 = precision.size_bytes() as u32;
    let mean_ty = precision;
    let stddev_ty = precision;

    KernelBuilder::new(kernel_name)
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("seed_lo", PtxType::U32)
        .param("seed_hi", PtxType::U32)
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
                let seed_lo = b.load_param_u32("seed_lo");
                let seed_hi = b.load_param_u32("seed_hi");
                let offset_lo = b.load_param_u32("offset_lo");
                let offset_hi = b.load_param_u32("offset_hi");

                // counter = gid + offset
                let gid_lo = gid;
                let gid_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {gid_hi}, 0;"));
                let ctr_lo = b.alloc_reg(PtxType::U32);
                let ctr_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.cc.u32 {ctr_lo}, {gid_lo}, {offset_lo};"));
                b.raw_ptx(&format!("addc.u32 {ctr_hi}, {gid_hi}, {offset_hi};"));

                let c0 = ctr_lo;
                let c1 = ctr_hi;
                let c2 = b.alloc_reg(PtxType::U32);
                let c3 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {c2}, 0;"));
                b.raw_ptx(&format!("mov.u32 {c3}, 0;"));

                emit_philox_10rounds(b, &c0, &c1, &c2, &c3, &seed_lo, &seed_hi);

                // Box-Muller: u1 from c0, u2 from c1
                match precision {
                    PtxType::F32 => {
                        let mean_reg = b.load_param_f32("mean");
                        let stddev_reg = b.load_param_f32("stddev");

                        // Convert c0 -> u1 in (0,1]
                        let u1_raw = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {u1_raw}, {c0};"));
                        let scale = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;"));
                        let u1 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {u1}, {u1_raw}, {scale};"));
                        // Clamp to avoid log(0)
                        let eps = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {eps}, 0f33800000;")); // ~5.96e-8
                        let u1_safe = b.max_f32(u1, eps);

                        // Convert c1 -> u2 in [0,1)
                        let u2_raw = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {u2_raw}, {c1};"));
                        let u2 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {u2}, {u2_raw}, {scale};"));

                        // Box-Muller: z = sqrt(-2 * ln(u1)) * cos(2*pi*u2)
                        // ln(x) = lg2(x) * ln(2), where ln(2) = 0.693147...
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

                        // sqrt
                        let radius = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("sqrt.approx.f32 {radius}, {neg2ln};"));

                        // 2*pi*u2
                        let two_pi = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {two_pi}, 0f40C90FDB;")); // 2*pi
                        let angle = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {angle}, {two_pi}, {u2};"));

                        let cos_val = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cos.approx.f32 {cos_val}, {angle};"));

                        let z = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {z}, {radius}, {cos_val};"));

                        // result = mean + stddev * z
                        let result = b.fma_f32(stddev_reg, z, mean_reg);

                        let addr = b.byte_offset_addr(out_ptr, gid_lo.clone(), stride_bytes);
                        b.store_global_f32(addr, result);
                    }
                    PtxType::F64 => {
                        let mean_reg = b.load_param_f64("mean");
                        let stddev_reg = b.load_param_f64("stddev");

                        // Convert c0 -> u1 (use single precision approx for
                        // transcendentals, then widen)
                        let u1_f32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {u1_f32}, {c0};"));
                        let scale32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {scale32}, 0f2F800000;"));
                        let u1_32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {u1_32}, {u1_f32}, {scale32};"));
                        let eps32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {eps32}, 0f33800000;"));
                        let u1_safe_32 = b.max_f32(u1_32, eps32);

                        let u2_f32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {u2_f32}, {c1};"));
                        let u2_32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {u2_32}, {u2_f32}, {scale32};"));

                        // Box-Muller in f32 then convert to f64
                        let lg2_u1 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("lg2.approx.f32 {lg2_u1}, {u1_safe_32};"));
                        let ln2 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {ln2}, 0f3F317218;"));
                        let ln_u1 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {ln_u1}, {lg2_u1}, {ln2};"));
                        let neg2 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {neg2}, 0fC0000000;"));
                        let neg2ln = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {neg2ln}, {neg2}, {ln_u1};"));
                        let radius32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("sqrt.approx.f32 {radius32}, {neg2ln};"));
                        let two_pi = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {two_pi}, 0f40C90FDB;"));
                        let angle = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {angle}, {two_pi}, {u2_32};"));
                        let cos_val = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cos.approx.f32 {cos_val}, {angle};"));
                        let z32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {z32}, {radius32}, {cos_val};"));

                        // Widen to f64
                        let z64 = b.cvt_f32_to_f64(z32);

                        // result = mean + stddev * z
                        let result = b.fma_f64(stddev_reg, z64, mean_reg);

                        let addr = b.byte_offset_addr(out_ptr, gid_lo.clone(), stride_bytes);
                        b.store_global_f64(addr, result);
                    }
                    _ => {}
                }
            });

            b.ret();
        })
        .build()
}

/// Generates PTX for a Philox-4x32-10 raw u32 output kernel.
///
/// Each thread produces one u32 from the Philox counter output.
///
/// # Errors
///
/// Returns `PtxGenError` if kernel construction fails.
pub fn generate_philox_u32_ptx(sm: SmVersion) -> Result<String, PtxGenError> {
    KernelBuilder::new("philox_u32")
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("seed_lo", PtxType::U32)
        .param("seed_hi", PtxType::U32)
        .param("offset_lo", PtxType::U32)
        .param("offset_hi", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let seed_lo = b.load_param_u32("seed_lo");
                let seed_hi = b.load_param_u32("seed_hi");
                let offset_lo = b.load_param_u32("offset_lo");
                let offset_hi = b.load_param_u32("offset_hi");

                let gid_lo = gid;
                let gid_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {gid_hi}, 0;"));
                let ctr_lo = b.alloc_reg(PtxType::U32);
                let ctr_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.cc.u32 {ctr_lo}, {gid_lo}, {offset_lo};"));
                b.raw_ptx(&format!("addc.u32 {ctr_hi}, {gid_hi}, {offset_hi};"));

                let c0 = ctr_lo;
                let c1 = ctr_hi;
                let c2 = b.alloc_reg(PtxType::U32);
                let c3 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {c2}, 0;"));
                b.raw_ptx(&format!("mov.u32 {c3}, 0;"));

                emit_philox_10rounds(b, &c0, &c1, &c2, &c3, &seed_lo, &seed_hi);

                // Store c0 as raw u32
                let addr = b.byte_offset_addr(out_ptr, gid_lo.clone(), 4);
                b.raw_ptx(&format!("st.global.u32 [{addr}], {c0};"));
            });

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// CPU-side Philox-4x32-10 reference implementation
// ---------------------------------------------------------------------------

/// Multiply two u32 values and return the high 32 bits of the 64-bit product.
#[inline(always)]
fn mulhi32(a: u32, b: u32) -> u32 {
    (((a as u64).wrapping_mul(b as u64)) >> 32) as u32
}

/// Multiply two u32 values and return the low 32 bits of the 64-bit product.
#[inline(always)]
fn mullo32(a: u32, b: u32) -> u32 {
    a.wrapping_mul(b)
}

/// CPU implementation of one Philox-4x32-10 round.
///
/// Applies the bijection on the 4-word counter using the 2-word key.
/// Returns the new counter state after one round.
#[inline(always)]
fn philox4x32_round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let hi0 = mulhi32(PHILOX_M4X32_0, counter[0]);
    let lo0 = mullo32(PHILOX_M4X32_0, counter[0]);
    let hi1 = mulhi32(PHILOX_M4X32_1, counter[2]);
    let lo1 = mullo32(PHILOX_M4X32_1, counter[2]);

    [
        hi1 ^ counter[1] ^ key[0],
        lo1,
        hi0 ^ counter[3] ^ key[1],
        lo0,
    ]
}

/// Bump the Philox key by the Weyl sequence constants.
#[inline(always)]
fn philox4x32_bumpkey(key: [u32; 2]) -> [u32; 2] {
    [
        key[0].wrapping_add(PHILOX_W32_0),
        key[1].wrapping_add(PHILOX_W32_1),
    ]
}

/// CPU implementation of Philox-4x32-10.
///
/// Performs 10 rounds of the Philox bijection on the given counter with the
/// given key. Matches the PTX implementation exactly (key is bumped *after*
/// each round's bijection, so round 0 uses the original key).
///
/// This is the reference CPU implementation used for testing and for the
/// statistical quality tests. It produces the same output as the PTX kernels
/// when given equivalent inputs.
pub(crate) fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut ctr = counter;
    let mut k = key;
    for _ in 0..10 {
        ctr = philox4x32_round(ctr, k);
        k = philox4x32_bumpkey(k);
    }
    ctr
}

/// Generate a sequence of u32 values from Philox-4x32-10 in counter mode.
///
/// Each call to `philox4x32_10` produces 4 u32 values. This function
/// generates enough blocks to fill `n` outputs, starting at `offset`.
/// The counter is `[offset + i/4, offset_hi, 0, 0]` for block `i/4`.
pub(crate) fn philox_generate_u32s(seed: u64, n: usize, start_offset: u64) -> Vec<u32> {
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    let key = [seed_lo, seed_hi];

    let num_blocks = n.div_ceil(4);
    let mut output = Vec::with_capacity(num_blocks * 4);

    for block in 0..num_blocks as u64 {
        let ctr_val = start_offset.wrapping_add(block);
        let counter = [ctr_val as u32, (ctr_val >> 32) as u32, 0u32, 0u32];
        let result = philox4x32_10(counter, key);
        output.extend_from_slice(&result);
    }

    output.truncate(n);
    output
}

// ---------------------------------------------------------------------------
// Internal PTX helpers
// ---------------------------------------------------------------------------

/// Emits 10 rounds of Philox-4x32 mixing into the body builder.
///
/// Modifies `c0..c3` in place (via raw PTX) and bumps the key `k0, k1`
/// after each round.
fn emit_philox_10rounds(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    c0: &oxicuda_ptx::ir::Register,
    c1: &oxicuda_ptx::ir::Register,
    c2: &oxicuda_ptx::ir::Register,
    c3: &oxicuda_ptx::ir::Register,
    k0: &oxicuda_ptx::ir::Register,
    k1: &oxicuda_ptx::ir::Register,
) {
    // Temporaries for the round function
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

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;

    #[test]
    fn generate_uniform_f32_ptx() {
        let ptx = generate_philox_uniform_ptx(PtxType::F32, SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry philox_uniform_f32"));
        assert!(ptx.contains("mul.hi.u32"));
        assert!(ptx.contains("xor.b32"));
    }

    #[test]
    fn generate_uniform_f64_ptx() {
        let ptx = generate_philox_uniform_ptx(PtxType::F64, SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry philox_uniform_f64"));
    }

    #[test]
    fn generate_normal_f32_ptx() {
        let ptx = generate_philox_normal_ptx(PtxType::F32, SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry philox_normal_f32"));
        assert!(ptx.contains("lg2.approx"));
        assert!(ptx.contains("cos.approx"));
        assert!(ptx.contains("sqrt.approx"));
    }

    #[test]
    fn generate_u32_ptx() {
        let ptx = generate_philox_u32_ptx(SmVersion::Sm80);
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry philox_u32"));
        assert!(ptx.contains("st.global.u32"));
    }

    #[test]
    fn invalid_precision_returns_error() {
        let result = generate_philox_uniform_ptx(PtxType::U32, SmVersion::Sm80);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Task S15: Philox bit-for-bit reference vector tests
    // -----------------------------------------------------------------------

    /// Reference vectors from the Random123 test suite (Salmon et al., SC 2011).
    ///
    /// Source: https://github.com/DEShawResearch/random123/blob/main/tests/known_answers.h
    /// Philox-4x32-10, all-zeros key and counter.
    const PHILOX_REF_ZERO_KEY_ZERO_CTR: ([u32; 4], [u32; 2], [u32; 4]) = (
        [0, 0, 0, 0],                                         // counter
        [0, 0],                                               // key
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8], // expected output
    );

    /// Reference vector: all-ones counter and key.
    ///
    /// Verified against our implementation and cross-checked with the Random123
    /// canonical algorithm (mulhi/mullo with the Philox constants).
    const PHILOX_REF_ALL_ONES: ([u32; 4], [u32; 2], [u32; 4]) = (
        [0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff],
        [0xffff_ffff, 0xffff_ffff],
        [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
    );

    /// Reference vector: counter=[1,0,0,0], key=[0,0].
    ///
    /// Verified against the Philox-4x32-10 algorithm specification.
    const PHILOX_REF_CTR1: ([u32; 4], [u32; 2], [u32; 4]) = (
        [1, 0, 0, 0],
        [0, 0],
        [0xf8e4_cca4, 0x5cb2_00db, 0xb1a5_74eb, 0x097e_ff67],
    );

    /// Verify that the CPU Philox-4x32-10 implementation matches Random123
    /// known-answer test vectors. These vectors pin the algorithm to the
    /// canonical Salmon et al. specification.
    #[test]
    fn test_philox_reference_vectors() {
        let (ctr, key, expected) = PHILOX_REF_ZERO_KEY_ZERO_CTR;
        let got = philox4x32_10(ctr, key);
        assert_eq!(
            got, expected,
            "Philox zero/zero KAT failed: got {got:08x?}, expected {expected:08x?}"
        );

        let (ctr, key, expected) = PHILOX_REF_ALL_ONES;
        let got = philox4x32_10(ctr, key);
        assert_eq!(
            got, expected,
            "Philox all-ones KAT failed: got {got:08x?}, expected {expected:08x?}"
        );

        let (ctr, key, expected) = PHILOX_REF_CTR1;
        let got = philox4x32_10(ctr, key);
        assert_eq!(
            got, expected,
            "Philox ctr=1 KAT failed: got {got:08x?}, expected {expected:08x?}"
        );
    }

    /// Verify the counter-mode property: different counter values must
    /// produce different (non-colliding) outputs for the same key.
    #[test]
    fn test_philox_counter_mode_distinct_outputs() {
        let key = [0xDEAD_BEEF_u32, 0xCAFE_BABE];
        let mut seen = std::collections::HashSet::new();

        for i in 0u32..256 {
            let counter = [i, 0, 0, 0];
            let out = philox4x32_10(counter, key);
            // We check the full 128-bit output tuple for uniqueness.
            let fingerprint = (out[0], out[1], out[2], out[3]);
            assert!(
                seen.insert(fingerprint),
                "Counter-mode collision at counter={i}: {fingerprint:?}"
            );
        }
    }

    /// Verify that different keys produce different outputs for the same counter.
    #[test]
    fn test_philox_different_keys_different_outputs() {
        let counter = [0u32, 0, 0, 0];
        let mut seen = std::collections::HashSet::new();

        for k in 0u32..256 {
            let key = [k, 0];
            let out = philox4x32_10(counter, key);
            let fingerprint = (out[0], out[1], out[2], out[3]);
            assert!(
                seen.insert(fingerprint),
                "Key-sensitivity collision at key[0]={k}: {fingerprint:?}"
            );
        }
    }

    /// Verify that advancing the counter by 4 gives the same u32 stream as
    /// consuming the outputs sequentially. This validates `philox_generate_u32s`.
    #[test]
    fn test_philox_counter_advance_consistency() {
        let seed: u64 = 0x0123_4567_89AB_CDEF;
        let key = [seed as u32, (seed >> 32) as u32];

        // Generate 8 values via the helper.
        let generated = philox_generate_u32s(seed, 8, 0);

        // Verify manually: block 0 is counter [0,0,0,0], block 1 is [1,0,0,0].
        let block0 = philox4x32_10([0, 0, 0, 0], key);
        let block1 = philox4x32_10([1, 0, 0, 0], key);

        assert_eq!(&generated[0..4], &block0, "Block 0 mismatch");
        assert_eq!(&generated[4..8], &block1, "Block 1 mismatch");
    }

    /// Verify that the same seed always produces the same sequence
    /// (deterministic counter-mode property).
    #[test]
    fn test_philox_same_seed_same_sequence() {
        let seq_a = philox_generate_u32s(42, 100, 0);
        let seq_b = philox_generate_u32s(42, 100, 0);
        assert_eq!(seq_a, seq_b, "Same seed must give same sequence");
    }

    // -----------------------------------------------------------------------
    // Quality gate: Philox bit-for-bit reference vs known seed values
    // -----------------------------------------------------------------------

    /// Verify first output is non-zero for key=(0,0) — confirms the bijection
    /// actually scrambles the all-zeros input.
    #[test]
    fn test_philox_4x32_10_known_seed_0_nonzero() {
        // Philox-4x32-10 with key=(0,0), counter=(0,0,0,0)
        // Reference from Random123: {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}
        let counter = [0u32, 0, 0, 0];
        let key = [0u32, 0];
        let v = philox4x32_10(counter, key);
        // First output should not be zero (validated against reference vector)
        assert_ne!(
            v[0], 0,
            "First Philox output for zero key/counter must be non-zero"
        );
        // All 4 outputs should be distinct (the bijection spreads entropy)
        let unique: std::collections::HashSet<u32> = v.into_iter().collect();
        assert_eq!(
            unique.len(),
            4,
            "All 4 Philox-4x32 outputs should be distinct"
        );
    }

    /// Crude bucket uniformity test: generate 4096 u32 values and verify
    /// that the top 4 bits distribute roughly uniformly across 16 buckets.
    #[test]
    fn test_philox_uniformity_bucket_distribution() {
        let values = philox_generate_u32s(99, 4096, 0);
        let mut buckets = [0u32; 16];
        for v in &values {
            buckets[(v >> 28) as usize] += 1;
        }
        // Each bucket should have ~256 ± 3σ ≈ 256 ± 48 values
        // (std dev = sqrt(4096 * 1/16 * 15/16) ≈ 15.5, 3σ ≈ 46.5)
        for (i, &count) in buckets.iter().enumerate() {
            assert!(
                count > 128 && count < 384,
                "Bucket {i} has {count} values (expected ~256, range 128..384)"
            );
        }
    }

    /// Verify that the reference known-answer test vectors from Random123 pass.
    /// This pins the implementation to the canonical Salmon et al. specification.
    #[test]
    fn test_philox_kat_zero_key_counter() {
        // Reference: counter=[0,0,0,0], key=[0,0]
        // Expected: {0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8}
        let counter = [0u32, 0, 0, 0];
        let key = [0u32, 0];
        let got = philox4x32_10(counter, key);
        let expected = [0x6627_e8d5u32, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8];
        assert_eq!(
            got, expected,
            "Philox KAT (zero/zero) failed: got {got:08x?}, expected {expected:08x?}"
        );
    }
}
