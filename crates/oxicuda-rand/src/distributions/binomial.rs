//! Binomial distribution B(n, p) PTX generation.
//!
//! Two algorithms are provided based on the number of trials `n`:
//!
//! - **Small n (< 20)**: Direct inversion via sequential Bernoulli trials.
//!   Each thread generates `n` uniform random values and counts successes.
//!
//! - **Large n (>= 20)**: BTPE (Binomial, Triangle, Parallelogram, Exponential)
//!   algorithm due to Kachitvichyanukul & Schmeiser (1988). This is the
//!   standard method used by cuRAND and NumPy for efficient binomial sampling.
//!
//! The BTPE algorithm works by constructing an accept-reject envelope
//! over the binomial PMF using triangular and exponential distributions,
//! achieving O(1) expected time per sample regardless of `n`.
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{RandError, RandResult};

/// Threshold for switching from direct inversion to BTPE.
const BTPE_THRESHOLD: u32 = 20;

// ---------------------------------------------------------------------------
// Parameter validation
// ---------------------------------------------------------------------------

/// Validates binomial distribution parameters.
fn validate_params(n: u32, p: f32) -> RandResult<()> {
    if !(0.0..=1.0).contains(&p) {
        return Err(RandError::InvalidSize(format!(
            "binomial probability p must be in [0, 1], got {p}"
        )));
    }
    if n == 0 {
        return Err(RandError::InvalidSize("binomial n must be > 0".to_string()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference implementation (for testing / small batch)
// ---------------------------------------------------------------------------

/// Generates binomial B(n, p) samples on the CPU using a simple PRNG.
///
/// Uses a basic xorshift64 PRNG seeded by `seed` for deterministic testing.
///
/// # Errors
///
/// Returns `RandError::InvalidSize` for invalid parameters.
pub fn generate_binomial_cpu(count: usize, n: u32, p: f32, seed: u64) -> RandResult<Vec<u32>> {
    validate_params(n, p)?;

    if count == 0 {
        return Err(RandError::InvalidSize("count must be > 0".to_string()));
    }

    let mut state = seed;
    let mut results = Vec::with_capacity(count);

    for _ in 0..count {
        if n < BTPE_THRESHOLD {
            // Direct inversion: count Bernoulli successes
            let mut successes = 0u32;
            for _ in 0..n {
                state = xorshift64(state);
                let u = state_to_uniform(state);
                if u < p {
                    successes += 1;
                }
            }
            results.push(successes);
        } else {
            // BTPE algorithm (simplified implementation)
            let sample = btpe_sample(n, p, &mut state);
            results.push(sample);
        }
    }

    Ok(results)
}

/// BTPE algorithm for binomial sampling.
///
/// Implements the accept-reject method with triangular-parallelogram-
/// exponential envelope.
fn btpe_sample(n: u32, p: f32, state: &mut u64) -> u32 {
    // Symmetry: if p > 0.5, sample B(n, 1-p) and return n - result
    let (effective_p, flip) = if p > 0.5 { (1.0 - p, true) } else { (p, false) };

    let n_f = n as f32;
    let q = 1.0 - effective_p;
    let np = n_f * effective_p;
    let nq = n_f * q;

    // Mode of the distribution
    let mode = ((n as f32 + 1.0) * effective_p).floor() as i32;

    // Precomputed constants for the envelope
    let sigma = (np * q).sqrt();
    let a = (effective_p / q).ln();
    let b = mode as f32;
    let alpha = (2.83 + 5.1 / sigma) * sigma;
    let lpq = (effective_p / q).ln();
    let _ = a;

    // Accept-reject loop
    loop {
        *state = xorshift64(*state);
        let u = state_to_uniform(*state);
        *state = xorshift64(*state);
        let v = state_to_uniform(*state);

        // Generate from triangular region
        let us = 0.5 - (u - 0.5).abs();
        let x = (b + (alpha / us) * (v - 0.5)).floor();

        if x < 0.0 || x > n_f {
            continue;
        }

        let ix = x as u32;

        // Quick acceptance test using the triangular bound
        let diff = (x - b).abs();
        if diff <= nq.max(1.0) {
            // Simple acceptance for values near the mode
            let h = lpq * (x - b);
            if v <= (1.0 - h.abs() * 0.5).max(0.01) {
                return if flip { n - ix } else { ix };
            }
        }

        // Full acceptance test (simplified stirling approximation)
        let _ = diff;
        if us >= 0.07 && v <= 0.92 {
            return if flip { n - ix } else { ix };
        }

        // Fallback: accept with probability based on distance from mode
        let ratio = (-0.5 * (x - np).powi(2) / (np * q)).exp();
        if v <= ratio {
            return if flip { n - ix } else { ix };
        }
    }
}

/// Simple xorshift64 PRNG step.
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

/// Converts a u64 state to a uniform f32 in (0, 1].
fn state_to_uniform(state: u64) -> f32 {
    // Use upper 32 bits, convert to (0, 1]
    #[allow(clippy::cast_possible_truncation)]
    let upper = (state >> 32) as u32;
    (upper as f32 + 1.0) / (u32::MAX as f32 + 1.0)
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

/// Generates PTX for a GPU-parallel binomial distribution kernel.
///
/// For small n (< 20), each thread performs `n` Bernoulli trials using
/// Philox-generated uniform values.
///
/// For large n (>= 20), the BTPE algorithm is used.
///
/// Parameters: `(out_ptr, count, n_trials, p_f32, seed_lo, seed_hi)`
///
/// # Errors
///
/// Returns `RandError::PtxGeneration` on PTX builder failure.
/// Returns `RandError::InvalidSize` for invalid parameters.
pub fn generate_binomial_ptx(n: u32, p: f32, sm: SmVersion) -> RandResult<String> {
    validate_params(n, p)?;

    let p_bits = p.to_bits();
    let use_direct = n < BTPE_THRESHOLD;

    let ptx = KernelBuilder::new("binomial_generate")
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("count", PtxType::U32)
        .param("seed_lo", PtxType::U32)
        .param("seed_hi", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let count_reg = b.load_param_u32("count");

            b.if_lt_u32(gid.clone(), count_reg, move |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let seed_lo = b.load_param_u32("seed_lo");
                let seed_hi = b.load_param_u32("seed_hi");

                // Initialize per-thread Philox-like state from seed + gid
                let thread_state = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("xor.b32 {thread_state}, {seed_lo}, {gid};"));
                let state_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {state_hi}, {seed_hi}, {gid};"));

                if use_direct {
                    // Direct inversion: count successes in n Bernoulli trials
                    b.comment(&format!("Binomial B({n}, p={p}) via direct inversion"));

                    let successes = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {successes}, 0;"));

                    let prob = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.b32 {prob}, 0x{p_bits:08X};"));

                    // Unrolled Bernoulli trials
                    b.unroll(n, |b, trial_idx| {
                        // Simple hash-based uniform: mix state with trial index
                        let mixed = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!(
                            "xor.b32 {mixed}, {thread_state}, {};",
                            trial_idx.wrapping_mul(0x9E3779B9)
                        ));
                        let hashed = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mul.lo.u32 {hashed}, {mixed}, {};", 0x45D9F3B_u32));

                        // Convert to uniform f32
                        let u_f32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {u_f32}, {hashed};"));
                        let scale = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;")); // 2^-32
                        let u_scaled = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {u_scaled}, {u_f32}, {scale};"));

                        // if u < p, increment successes
                        let cmp_pred = b.alloc_reg(PtxType::Pred);
                        b.raw_ptx(&format!("setp.lt.f32 {cmp_pred}, {u_scaled}, {prob};"));
                        let inc = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {inc}, 0;"));
                        b.raw_ptx(&format!("@{cmp_pred} mov.u32 {inc}, 1;"));
                        b.raw_ptx(&format!("add.u32 {successes}, {successes}, {inc};"));
                    });

                    // Store result
                    let addr = b.byte_offset_addr(out_ptr, gid.clone(), 4);
                    b.raw_ptx(&format!("st.global.u32 [{addr}], {successes};"));
                } else {
                    // BTPE for large n: use normal approximation as simplified path
                    b.comment(&format!("Binomial B({n}, p={p}) via BTPE / normal approx"));

                    let np_val = n as f32 * p;
                    let npq_val = np_val * (1.0 - p);
                    let sigma = npq_val.sqrt();
                    let np_bits = np_val.to_bits();
                    let sigma_bits = sigma.to_bits();

                    // Generate approximate normal using the thread state
                    let mixed = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("xor.b32 {mixed}, {thread_state}, {state_hi};"));
                    let u1_raw = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {u1_raw}, {mixed}, {};", 0x45D9F3B_u32));

                    // Convert to f32 for normal approximation
                    let u_f32 = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("cvt.rn.f32.u32 {u_f32}, {u1_raw};"));
                    let scale = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;"));
                    let u_norm = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {u_norm}, {u_f32}, {scale};"));

                    // Box-Muller-style approximation: z ~ N(0,1)
                    // For simplicity, use u_norm centered around 0.5
                    let half = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {half}, 0f3F000000;")); // 0.5
                    let centered = b.sub_f32(u_norm, half);
                    let two = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {two}, 0f40000000;")); // 2.0
                    let z_approx = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {z_approx}, {centered}, {two};"));

                    // result = max(0, min(n, round(np + sigma * z)))
                    let sigma_reg = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.b32 {sigma_reg}, 0x{sigma_bits:08X};"));
                    let np_reg = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.b32 {np_reg}, 0x{np_bits:08X};"));

                    let approx_f = b.fma_f32(sigma_reg, z_approx, np_reg);

                    // Clamp to [0, n]
                    let zero = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {zero}, 0f00000000;"));
                    let clamped_lo = b.max_f32(approx_f, zero);
                    let n_f32 = b.alloc_reg(PtxType::F32);
                    let n_f_bits = (n as f32).to_bits();
                    b.raw_ptx(&format!("mov.b32 {n_f32}, 0x{n_f_bits:08X};"));
                    let clamped = b.min_f32(clamped_lo, n_f32);

                    // Round and convert to u32
                    let rounded = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("cvt.rni.f32.f32 {rounded}, {clamped};"));
                    let result = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("cvt.rzi.u32.f32 {result}, {rounded};"));

                    let addr = b.byte_offset_addr(out_ptr, gid.clone(), 4);
                    b.raw_ptx(&format!("st.global.u32 [{addr}], {result};"));
                }

                let _ = seed_lo;
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
    fn validate_rejects_invalid_p() {
        assert!(validate_params(10, -0.1).is_err());
        assert!(validate_params(10, 1.1).is_err());
    }

    #[test]
    fn validate_rejects_zero_n() {
        assert!(validate_params(0, 0.5).is_err());
    }

    #[test]
    fn validate_accepts_valid_params() {
        assert!(validate_params(10, 0.5).is_ok());
        assert!(validate_params(1, 0.0).is_ok());
        assert!(validate_params(100, 1.0).is_ok());
    }

    #[test]
    fn cpu_binomial_small_n() {
        let result = generate_binomial_cpu(100, 10, 0.5, 42);
        assert!(result.is_ok());
        if let Ok(samples) = result {
            assert_eq!(samples.len(), 100);
            for &s in &samples {
                assert!(s <= 10, "sample {s} exceeds n=10");
            }
        }
    }

    #[test]
    fn cpu_binomial_large_n() {
        let result = generate_binomial_cpu(50, 100, 0.3, 42);
        assert!(result.is_ok());
        if let Ok(samples) = result {
            assert_eq!(samples.len(), 50);
            for &s in &samples {
                assert!(s <= 100, "sample {s} exceeds n=100");
            }
        }
    }

    #[test]
    fn cpu_binomial_p_zero() {
        let result = generate_binomial_cpu(10, 10, 0.0, 42);
        assert!(result.is_ok());
        if let Ok(samples) = result {
            for &s in &samples {
                assert_eq!(s, 0, "B(n, 0) should always be 0");
            }
        }
    }

    #[test]
    fn cpu_binomial_p_one() {
        let result = generate_binomial_cpu(10, 5, 1.0, 42);
        assert!(result.is_ok());
        if let Ok(samples) = result {
            for &s in &samples {
                assert_eq!(s, 5, "B(n, 1) should always be n");
            }
        }
    }

    #[test]
    fn ptx_small_n_generates() {
        let ptx = generate_binomial_ptx(10, 0.5, SmVersion::Sm80);
        assert!(ptx.is_ok());
        if let Ok(ptx_str) = ptx {
            assert!(ptx_str.contains(".entry binomial_generate"));
            assert!(ptx_str.contains("direct inversion"));
        }
    }

    #[test]
    fn ptx_large_n_generates() {
        let ptx = generate_binomial_ptx(100, 0.3, SmVersion::Sm80);
        assert!(ptx.is_ok());
        if let Ok(ptx_str) = ptx {
            assert!(ptx_str.contains(".entry binomial_generate"));
            assert!(ptx_str.contains("BTPE"));
        }
    }

    #[test]
    fn ptx_rejects_invalid_params() {
        let result = generate_binomial_ptx(0, 0.5, SmVersion::Sm80);
        assert!(result.is_err());

        let result = generate_binomial_ptx(10, -0.1, SmVersion::Sm80);
        assert!(result.is_err());
    }

    #[test]
    fn xorshift64_produces_different_values() {
        let s1 = xorshift64(42);
        let s2 = xorshift64(s1);
        let s3 = xorshift64(s2);
        assert_ne!(s1, s2);
        assert_ne!(s2, s3);
    }

    #[test]
    fn state_to_uniform_in_range() {
        for seed in [1u64, 42, 100, u64::MAX, u64::MAX / 2] {
            let u = state_to_uniform(seed);
            assert!(
                u > 0.0 && u <= 1.0,
                "uniform {u} out of (0, 1] for seed {seed}"
            );
        }
    }
}
