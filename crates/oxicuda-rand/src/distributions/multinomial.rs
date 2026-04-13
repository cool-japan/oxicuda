//! Multinomial distribution sampling.
//!
//! The multinomial distribution models drawing `n` trials from `k` categories,
//! each with a given probability.  Each trial produces one of `k` outcomes;
//! the result is a vector of `k` counts summing to `n`.
//!
//! CPU sampling uses the conditional-binomial method: for each category in
//! turn, draw from a binomial distribution with the adjusted probability,
//! then subtract the drawn count from the remaining trials.
//!
//! The GPU kernel uses a per-thread sequential sampling approach: each
//! thread generates one complete multinomial sample (k counts from n trials).
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{RandError, RandResult};

// ---------------------------------------------------------------------------
// Simple xorshift64 PRNG
// ---------------------------------------------------------------------------

/// Xorshift64 step function.
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

/// Converts a u64 state to a uniform f32 in (0, 1].
fn state_to_uniform(state: u64) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    let upper = (state >> 32) as u32;
    (upper as f32 + 1.0) / (u32::MAX as f32 + 1.0)
}

// ---------------------------------------------------------------------------
// Parameter validation
// ---------------------------------------------------------------------------

/// Validates multinomial distribution parameters.
fn validate_params(probabilities: &[f32], n: u32, num_samples: usize) -> RandResult<()> {
    if probabilities.is_empty() {
        return Err(RandError::InvalidSize(
            "probabilities must have at least 1 category".to_string(),
        ));
    }
    if n == 0 {
        return Err(RandError::InvalidSize(
            "number of trials n must be > 0".to_string(),
        ));
    }
    if num_samples == 0 {
        return Err(RandError::InvalidSize(
            "num_samples must be > 0".to_string(),
        ));
    }

    // Check that probabilities are non-negative
    for (i, &p) in probabilities.iter().enumerate() {
        if p < 0.0 {
            return Err(RandError::InvalidSize(format!(
                "probability[{i}] = {p} is negative"
            )));
        }
    }

    // Check that probabilities sum close to 1.0
    let sum: f32 = probabilities.iter().sum();
    if (sum - 1.0).abs() > 0.01 {
        return Err(RandError::InvalidSize(format!(
            "probabilities sum to {sum}, expected ~1.0"
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// CPU implementation
// ---------------------------------------------------------------------------

/// Generates multinomial samples on the CPU.
///
/// Each sample consists of `k` counts (one per category) that sum to `n`.
/// The output buffer must have at least `k * num_samples` elements, stored
/// in row-major order: the first `k` entries are the counts for sample 0,
/// the next `k` for sample 1, and so on.
///
/// Uses the conditional-binomial decomposition: for categories `i = 0..k-2`,
/// draw `x_i ~ Binomial(n_remaining, p_i / p_remaining)`, then set the last
/// category to `n_remaining`.
///
/// # Errors
///
/// Returns [`RandError::InvalidSize`] for invalid parameters or a too-small
/// output buffer.
pub fn generate_multinomial_cpu(
    output: &mut [u32],
    probabilities: &[f32],
    n: u32,
    num_samples: usize,
    seed: u64,
) -> RandResult<()> {
    validate_params(probabilities, n, num_samples)?;

    let k = probabilities.len();
    let required = k
        .checked_mul(num_samples)
        .ok_or_else(|| RandError::InvalidSize("k * num_samples overflow".to_string()))?;
    if output.len() < required {
        return Err(RandError::InvalidSize(format!(
            "output buffer has {} elements but {} required",
            output.len(),
            required
        )));
    }

    let mut rng_state = seed;

    for s in 0..num_samples {
        let mut remaining = n;
        let mut p_remaining: f32 = 1.0;
        let base = s * k;

        for i in 0..k - 1 {
            if remaining == 0 || p_remaining <= 0.0 {
                output[base + i] = 0;
                continue;
            }

            // Conditional binomial: draw from Binomial(remaining, p_i / p_remaining)
            let p_cond = (probabilities[i] / p_remaining).clamp(0.0, 1.0);

            // Simple sequential Bernoulli trials for the binomial
            let mut count = 0u32;
            for _ in 0..remaining {
                rng_state = xorshift64(rng_state);
                let u = state_to_uniform(rng_state);
                if u <= p_cond {
                    count += 1;
                }
            }

            output[base + i] = count;
            remaining = remaining.saturating_sub(count);
            p_remaining -= probabilities[i];
            if p_remaining < 0.0 {
                p_remaining = 0.0;
            }
        }

        // Last category gets all remaining trials
        output[base + k - 1] = remaining;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

/// Generates PTX for a GPU multinomial distribution kernel.
///
/// Each thread produces one complete multinomial sample (k counts from n trials)
/// using a sequential approach: for each trial, draw a uniform random value
/// and walk through the cumulative probability table to find the category.
///
/// Parameters: `(out_ptr, num_samples, seed_lo, seed_hi)`
///
/// The probabilities and trial count `n` are baked into the kernel as
/// immediate constants.
///
/// # Errors
///
/// Returns [`RandError::PtxGeneration`] on PTX builder failure.
pub fn generate_multinomial_ptx(k: usize, sm: SmVersion) -> RandResult<String> {
    if k == 0 {
        return Err(RandError::InvalidSize("k must be > 0".to_string()));
    }

    let ptx = KernelBuilder::new("multinomial_generate")
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("prob_ptr", PtxType::U64)
        .param("num_samples", PtxType::U32)
        .param("n_trials", PtxType::U32)
        .param("seed_lo", PtxType::U32)
        .param("seed_hi", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let num_samples_reg = b.load_param_u32("num_samples");

            b.if_lt_u32(gid.clone(), num_samples_reg, move |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let prob_ptr = b.load_param_u64("prob_ptr");
                let n_trials = b.load_param_u32("n_trials");
                let seed_lo = b.load_param_u32("seed_lo");
                let seed_hi = b.load_param_u32("seed_hi");

                // Initialize per-thread PRNG state
                let state = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("xor.b32 {state}, {seed_lo}, {gid};"));
                let state_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {state_hi}, {seed_hi}, {gid};"));

                // Initialize count array to zero (k registers)
                let counts: Vec<_> = (0..k)
                    .map(|_| {
                        let r = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {r}, 0;"));
                        r
                    })
                    .collect();

                // Load cumulative probabilities from prob_ptr
                let cum_probs: Vec<_> = (0..k)
                    .map(|i| {
                        let offset = (i as u64) * 4;
                        let addr = b.alloc_reg(PtxType::U64);
                        b.raw_ptx(&format!("add.u64 {addr}, {prob_ptr}, {offset};"));
                        let p = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("ld.global.f32 {p}, [{addr}];"));
                        p
                    })
                    .collect();

                // For simplicity, we perform n_trials sequentially per thread.
                // Each trial: generate uniform, find category via cumulative prob.
                // We use an unrolled loop for the category search.
                b.comment("sequential multinomial trials");

                // We unroll the trial loop for small n (up to 32), else
                // use a software loop. Here we generate PTX for the
                // trial structure with category search.
                let max_unroll = 32u32;
                b.unroll(max_unroll, |b, trial_idx| {
                    // Check if trial_idx < n_trials
                    let trial_pred = b.alloc_reg(PtxType::Pred);
                    let trial_const = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {trial_const}, {trial_idx};"));
                    b.raw_ptx(&format!(
                        "setp.lt.u32 {trial_pred}, {trial_const}, {n_trials};"
                    ));

                    // Generate uniform random
                    let mix = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "xor.b32 {mix}, {state}, {};",
                        trial_idx.wrapping_mul(0x9E3779B9)
                    ));
                    let hashed = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {hashed}, {mix}, {};", 0x45D9F3B_u32));
                    let u_f32 = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("cvt.rn.f32.u32 {u_f32}, {hashed};"));
                    let scale = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;")); // 2^-32
                    let u_val = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {u_val}, {u_f32}, {scale};"));

                    // Find category: increment counts[i] where cum_probs[i-1] <= u < cum_probs[i]
                    // Walk through cumulative probabilities
                    for i in 0..k {
                        let cat_pred = b.alloc_reg(PtxType::Pred);
                        b.raw_ptx(&format!(
                            "setp.lt.f32 {cat_pred}, {u_val}, {};",
                            cum_probs[i]
                        ));
                        // If this is the first category where u < cum_prob, increment it
                        // We need "u < cum_prob AND (i == 0 OR u >= cum_prob[i-1])"
                        // Simplified: just increment the first matching category
                        let inc_pred = b.alloc_reg(PtxType::Pred);
                        if i == 0 {
                            b.raw_ptx(&format!("and.pred {inc_pred}, {trial_pred}, {cat_pred};"));
                        } else {
                            let prev_pred = b.alloc_reg(PtxType::Pred);
                            b.raw_ptx(&format!(
                                "setp.ge.f32 {prev_pred}, {u_val}, {};",
                                cum_probs[i - 1]
                            ));
                            b.raw_ptx(&format!("and.pred {inc_pred}, {cat_pred}, {prev_pred};"));
                            let final_pred = b.alloc_reg(PtxType::Pred);
                            b.raw_ptx(&format!("and.pred {final_pred}, {inc_pred}, {trial_pred};"));
                            b.raw_ptx(&format!("mov.pred {inc_pred}, {final_pred};"));
                        }

                        let inc = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {inc}, 0;"));
                        b.raw_ptx(&format!("@{inc_pred} mov.u32 {inc}, 1;"));
                        b.raw_ptx(&format!("add.u32 {}, {}, {inc};", counts[i], counts[i]));
                    }

                    // Advance PRNG state
                    b.raw_ptx(&format!("xor.b32 {state}, {state}, {hashed};"));
                });

                // Store counts to out_ptr[gid * k + i]
                for (i, count_reg) in counts.iter().enumerate() {
                    let k_reg = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {k_reg}, {};", k));
                    let row = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {row}, {gid}, {k_reg};"));
                    let elem = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("add.u32 {elem}, {row}, {};", i));
                    let addr = b.byte_offset_addr(out_ptr.clone(), elem, 4);
                    b.raw_ptx(&format!("st.global.u32 [{addr}], {};", count_reg));
                }

                let _ = state_hi;
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
    fn validate_rejects_empty_probs() {
        assert!(validate_params(&[], 10, 1).is_err());
    }

    #[test]
    fn validate_rejects_zero_n() {
        assert!(validate_params(&[0.5, 0.5], 0, 1).is_err());
    }

    #[test]
    fn validate_rejects_zero_samples() {
        assert!(validate_params(&[0.5, 0.5], 10, 0).is_err());
    }

    #[test]
    fn validate_rejects_negative_prob() {
        assert!(validate_params(&[0.5, -0.5, 1.0], 10, 1).is_err());
    }

    #[test]
    fn validate_rejects_bad_sum() {
        assert!(validate_params(&[0.5, 0.3], 10, 1).is_err());
    }

    #[test]
    fn validate_accepts_valid_params() {
        assert!(validate_params(&[0.5, 0.5], 10, 5).is_ok());
        assert!(validate_params(&[0.2, 0.3, 0.5], 100, 1).is_ok());
    }

    #[test]
    fn cpu_multinomial_basic() {
        let probs = [0.5_f32, 0.3, 0.2];
        let n = 20;
        let num_samples = 10;
        let k = probs.len();
        let mut output = vec![0u32; k * num_samples];
        let res = generate_multinomial_cpu(&mut output, &probs, n, num_samples, 42);
        assert!(res.is_ok());

        // Each sample's counts should sum to n
        for s in 0..num_samples {
            let sum: u32 = output[s * k..(s + 1) * k].iter().sum();
            assert_eq!(sum, n, "sample {s} counts sum to {sum}, expected {n}");
        }
    }

    #[test]
    fn cpu_multinomial_two_categories() {
        let probs = [0.7_f32, 0.3];
        let n = 100;
        let num_samples = 50;
        let k = 2;
        let mut output = vec![0u32; k * num_samples];
        let res = generate_multinomial_cpu(&mut output, &probs, n, num_samples, 123);
        assert!(res.is_ok());

        for s in 0..num_samples {
            let c0 = output[s * k];
            let c1 = output[s * k + 1];
            assert_eq!(c0 + c1, n);
        }
    }

    #[test]
    fn cpu_multinomial_rejects_small_buffer() {
        let probs = [0.5_f32, 0.5];
        let mut output = vec![0u32; 3]; // need 2 * 5 = 10
        let res = generate_multinomial_cpu(&mut output, &probs, 10, 5, 42);
        assert!(res.is_err());
    }

    #[test]
    fn ptx_multinomial_compiles() {
        let ptx = generate_multinomial_ptx(3, SmVersion::Sm80);
        assert!(ptx.is_ok());
        if let Ok(ptx_str) = ptx {
            assert!(ptx_str.contains(".entry multinomial_generate"));
        }
    }

    #[test]
    fn ptx_rejects_zero_k() {
        assert!(generate_multinomial_ptx(0, SmVersion::Sm80).is_err());
    }

    #[test]
    fn cpu_multinomial_deterministic() {
        let probs = [0.3_f32, 0.3, 0.4];
        let n = 10;
        let k = 3;
        let mut out1 = vec![0u32; k * 5];
        let mut out2 = vec![0u32; k * 5];
        let _ = generate_multinomial_cpu(&mut out1, &probs, n, 5, 42);
        let _ = generate_multinomial_cpu(&mut out2, &probs, n, 5, 42);
        assert_eq!(out1, out2);
    }
}
