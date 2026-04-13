//! Truncated normal distribution on an interval [a, b].
//!
//! Uses an accept-reject method: generate standard normal samples, transform
//! to `mean + stddev * z`, and reject any that fall outside [lower, upper].
//!
//! For narrow truncation ranges far from the mean this can be inefficient;
//! in such cases a specialised algorithm (e.g. Robert's method) would be
//! preferable.  However, for the common case where the truncation window
//! covers a significant fraction of the probability mass, accept-reject
//! is simple and effective.
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{RandError, RandResult};

// ---------------------------------------------------------------------------
// Simple xorshift64 PRNG
// ---------------------------------------------------------------------------

fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

fn state_to_uniform(state: u64) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    let upper = (state >> 32) as u32;
    (upper as f32 + 1.0) / (u32::MAX as f32 + 1.0)
}

// ---------------------------------------------------------------------------
// Box-Muller transform (CPU, f32)
// ---------------------------------------------------------------------------

/// Generates a pair of standard normal values from two uniform values.
fn box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let u1_safe = u1.max(f32::MIN_POSITIVE);
    let radius = (-2.0_f32 * u1_safe.ln()).sqrt();
    let angle = 2.0 * std::f32::consts::PI * u2;
    (radius * angle.cos(), radius * angle.sin())
}

// ---------------------------------------------------------------------------
// Parameter validation
// ---------------------------------------------------------------------------

fn validate_params(mean: f32, stddev: f32, lower: f32, upper: f32) -> RandResult<()> {
    if stddev <= 0.0 {
        return Err(RandError::InvalidSize(format!(
            "stddev must be > 0, got {stddev}"
        )));
    }
    if lower >= upper {
        return Err(RandError::InvalidSize(format!(
            "lower ({lower}) must be < upper ({upper})"
        )));
    }
    if !mean.is_finite() || !stddev.is_finite() || !lower.is_finite() || !upper.is_finite() {
        return Err(RandError::InvalidSize(
            "all parameters must be finite".to_string(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU implementation
// ---------------------------------------------------------------------------

/// Maximum number of accept-reject iterations before giving up.
const MAX_REJECT_ITERS: usize = 10_000;

/// Generates truncated normal samples on the CPU.
///
/// Samples are drawn from `N(mean, stddev^2)` and rejected if they lie
/// outside `[lower, upper]`.
///
/// # Errors
///
/// Returns [`RandError::InvalidSize`] for invalid parameters or if the
/// accept-reject loop fails to produce enough samples within
/// `MAX_REJECT_ITERS` per output element.
pub fn generate_truncated_normal_cpu(
    output: &mut [f32],
    count: usize,
    mean: f32,
    stddev: f32,
    lower: f32,
    upper: f32,
    seed: u64,
) -> RandResult<()> {
    validate_params(mean, stddev, lower, upper)?;

    if count == 0 {
        return Err(RandError::InvalidSize("count must be > 0".to_string()));
    }
    if output.len() < count {
        return Err(RandError::InvalidSize(format!(
            "output buffer has {} elements but {} required",
            output.len(),
            count
        )));
    }

    let mut rng_state = seed;
    let mut filled = 0;

    while filled < count {
        let mut attempts = 0;
        loop {
            // Generate two uniform values
            rng_state = xorshift64(rng_state);
            let u1 = state_to_uniform(rng_state);
            rng_state = xorshift64(rng_state);
            let u2 = state_to_uniform(rng_state);

            let (z0, z1) = box_muller(u1, u2);

            // Transform and check both candidates
            let x0 = mean + stddev * z0;
            if x0 >= lower && x0 <= upper && filled < count {
                output[filled] = x0;
                filled += 1;
                if filled >= count {
                    break;
                }
            }

            let x1 = mean + stddev * z1;
            if x1 >= lower && x1 <= upper && filled < count {
                output[filled] = x1;
                filled += 1;
                if filled >= count {
                    break;
                }
            }

            attempts += 1;
            if attempts >= MAX_REJECT_ITERS {
                return Err(RandError::InternalError(format!(
                    "truncated normal: accept-reject failed after {MAX_REJECT_ITERS} iterations \
                     (mean={mean}, stddev={stddev}, [{lower}, {upper}])"
                )));
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

/// Generates PTX for a GPU truncated normal kernel.
///
/// Each thread generates samples using accept-reject with inline Box-Muller.
/// The truncation bounds and distribution parameters are baked into the
/// kernel as immediates.
///
/// Parameters: `(out_ptr, count, seed_lo, seed_hi)`
///
/// # Errors
///
/// Returns [`RandError::PtxGeneration`] on PTX builder failure.
pub fn generate_truncated_normal_ptx(sm: SmVersion) -> RandResult<String> {
    let ptx = KernelBuilder::new("truncated_normal_generate")
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("count", PtxType::U32)
        .param("mean", PtxType::F32)
        .param("stddev", PtxType::F32)
        .param("lower", PtxType::F32)
        .param("upper", PtxType::F32)
        .param("seed_lo", PtxType::U32)
        .param("seed_hi", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let count_reg = b.load_param_u32("count");

            b.if_lt_u32(gid.clone(), count_reg, move |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let mean_reg = b.load_param_f32("mean");
                let stddev_reg = b.load_param_f32("stddev");
                let lower_reg = b.load_param_f32("lower");
                let upper_reg = b.load_param_f32("upper");
                let seed_lo = b.load_param_u32("seed_lo");
                let seed_hi = b.load_param_u32("seed_hi");

                // Per-thread state
                let state = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("xor.b32 {state}, {seed_lo}, {gid};"));

                // Generate candidate normal via simplified hash + transform
                // Try multiple candidates (unrolled attempts)
                let result = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.f32 {result}, {mean_reg};"));

                let found = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {found}, 0, 1;")); // false

                b.unroll(16, |b, attempt| {
                    // Only process if not yet found
                    let not_found = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("not.pred {not_found}, {found};"));

                    // Generate uniform via hash
                    let mix1 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "xor.b32 {mix1}, {state}, {};",
                        (attempt * 2).wrapping_mul(0x9E3779B9)
                    ));
                    let h1 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {h1}, {mix1}, {};", 0x45D9F3B_u32));
                    let mix2 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "xor.b32 {mix2}, {state}, {};",
                        (attempt * 2 + 1).wrapping_mul(0x9E3779B9)
                    ));
                    let h2 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {h2}, {mix2}, {};", 0x45D9F3B_u32));

                    // Combine with seed_hi for more entropy
                    let h1b = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("xor.b32 {h1b}, {h1}, {seed_hi};"));

                    // Convert to uniform f32
                    let u1_f = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("cvt.rn.f32.u32 {u1_f}, {h1b};"));
                    let u2_f = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("cvt.rn.f32.u32 {u2_f}, {h2};"));
                    let scale = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;")); // 2^-32
                    let u1 = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {u1}, {u1_f}, {scale};"));
                    let u2 = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {u2}, {u2_f}, {scale};"));

                    // Clamp u1 away from zero
                    let eps = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {eps}, 0f33800000;")); // ~5.96e-8
                    let u1_safe = b.max_f32(u1, eps);

                    // Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
                    let lg2_u1 = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("lg2.approx.f32 {lg2_u1}, {u1_safe};"));
                    let ln2 = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {ln2}, 0f3F317218;")); // ln(2)
                    let ln_u1 = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {ln_u1}, {lg2_u1}, {ln2};"));
                    let neg2 = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {neg2}, 0fC0000000;")); // -2.0
                    let neg2ln = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {neg2ln}, {neg2}, {ln_u1};"));
                    let radius = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("sqrt.approx.f32 {radius}, {neg2ln};"));
                    let two_pi = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {two_pi}, 0f40C90FDB;")); // 2*pi
                    let angle = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {angle}, {two_pi}, {u2};"));
                    let cos_val = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("cos.approx.f32 {cos_val}, {angle};"));
                    let z = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {z}, {radius}, {cos_val};"));

                    // x = mean + stddev * z
                    let x = b.fma_f32(stddev_reg.clone(), z, mean_reg.clone());

                    // Check bounds: lower <= x <= upper
                    let ge_lower = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.ge.f32 {ge_lower}, {x}, {lower_reg};"));
                    let le_upper = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.le.f32 {le_upper}, {x}, {upper_reg};"));
                    let in_bounds = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("and.pred {in_bounds}, {ge_lower}, {le_upper};"));

                    // Accept if in bounds and not yet found
                    let accept = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("and.pred {accept}, {in_bounds}, {not_found};"));

                    // Conditionally store the result
                    b.raw_ptx(&format!("@{accept} mov.f32 {result}, {x};"));
                    b.raw_ptx(&format!("@{accept} setp.eq.u32 {found}, 1, 1;"));

                    // Advance state
                    b.raw_ptx(&format!("xor.b32 {state}, {state}, {h1};"));
                });

                // Store final result
                let addr = b.byte_offset_addr(out_ptr, gid.clone(), 4);
                b.store_global_f32(addr, result);
            });

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_zero_stddev() {
        assert!(validate_params(0.0, 0.0, -1.0, 1.0).is_err());
    }

    #[test]
    fn validate_rejects_inverted_bounds() {
        assert!(validate_params(0.0, 1.0, 2.0, 1.0).is_err());
    }

    #[test]
    fn validate_rejects_nan() {
        assert!(validate_params(f32::NAN, 1.0, -1.0, 1.0).is_err());
    }

    #[test]
    fn validate_accepts_valid() {
        assert!(validate_params(0.0, 1.0, -2.0, 2.0).is_ok());
    }

    #[test]
    fn cpu_truncated_normal_in_bounds() {
        let count = 100;
        let mut output = vec![0.0_f32; count];
        let res = generate_truncated_normal_cpu(&mut output, count, 0.0, 1.0, -2.0, 2.0, 42);
        assert!(res.is_ok());
        for &v in &output {
            assert!((-2.0..=2.0).contains(&v), "value {v} out of bounds [-2, 2]");
        }
    }

    #[test]
    fn cpu_truncated_normal_narrow_window() {
        let count = 50;
        let mut output = vec![0.0_f32; count];
        let res = generate_truncated_normal_cpu(&mut output, count, 0.0, 1.0, -0.5, 0.5, 42);
        assert!(res.is_ok());
        for &v in &output {
            assert!(
                (-0.5..=0.5).contains(&v),
                "value {v} out of bounds [-0.5, 0.5]"
            );
        }
    }

    #[test]
    fn cpu_truncated_normal_rejects_zero_count() {
        let mut output = vec![0.0_f32; 10];
        let res = generate_truncated_normal_cpu(&mut output, 0, 0.0, 1.0, -1.0, 1.0, 42);
        assert!(res.is_err());
    }

    #[test]
    fn cpu_truncated_normal_rejects_small_buffer() {
        let mut output = vec![0.0_f32; 5];
        let res = generate_truncated_normal_cpu(&mut output, 10, 0.0, 1.0, -1.0, 1.0, 42);
        assert!(res.is_err());
    }

    #[test]
    fn cpu_truncated_normal_deterministic() {
        let count = 20;
        let mut out1 = vec![0.0_f32; count];
        let mut out2 = vec![0.0_f32; count];
        let _ = generate_truncated_normal_cpu(&mut out1, count, 0.0, 1.0, -2.0, 2.0, 42);
        let _ = generate_truncated_normal_cpu(&mut out2, count, 0.0, 1.0, -2.0, 2.0, 42);
        assert_eq!(out1, out2);
    }

    #[test]
    fn ptx_truncated_normal_compiles() {
        let ptx = generate_truncated_normal_ptx(SmVersion::Sm80);
        assert!(ptx.is_ok());
        if let Ok(ptx_str) = ptx {
            assert!(ptx_str.contains(".entry truncated_normal_generate"));
            assert!(ptx_str.contains("cos.approx.f32")); // Box-Muller
        }
    }

    #[test]
    fn cpu_truncated_normal_shifted_mean() {
        let count = 50;
        let mut output = vec![0.0_f32; count];
        let res = generate_truncated_normal_cpu(&mut output, count, 5.0, 0.5, 4.0, 6.0, 42);
        assert!(res.is_ok());
        for &v in &output {
            assert!((4.0..=6.0).contains(&v), "value {v} out of bounds [4, 6]");
        }
    }
}
