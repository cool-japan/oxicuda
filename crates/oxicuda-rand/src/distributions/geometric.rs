//! Geometric distribution via inverse CDF method.
//!
//! The geometric distribution models the number of trials until the first
//! success, where each trial has success probability `p`.
//!
//! ```text
//! P(X = k) = (1-p)^{k-1} * p,   k = 1, 2, 3, ...
//! ```
//!
//! The inverse CDF method computes:
//! ```text
//! k = ceil(log(U) / log(1-p))
//! ```
//! where `U` is a uniform random value in (0, 1].
//!
//! This is efficient for GPU execution since it requires only one uniform
//! value per sample and uses fast approximate logarithm instructions.
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{RandError, RandResult};

// ---------------------------------------------------------------------------
// Parameter validation
// ---------------------------------------------------------------------------

/// Validates geometric distribution parameter p.
fn validate_p(p: f32) -> RandResult<()> {
    if p <= 0.0 || p > 1.0 {
        return Err(RandError::InvalidSize(format!(
            "geometric probability p must be in (0, 1], got {p}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// Generates geometric distribution samples on the CPU.
///
/// Uses the inverse CDF method: `k = ceil(log(U) / log(1 - p))`.
///
/// The minimum value returned is 1 (one trial always needed).
///
/// # Errors
///
/// Returns `RandError::InvalidSize` for invalid parameters.
pub fn generate_geometric_cpu(count: usize, p: f32, seed: u64) -> RandResult<Vec<u32>> {
    validate_p(p)?;

    if count == 0 {
        return Err(RandError::InvalidSize("count must be > 0".to_string()));
    }

    // Special case: p = 1.0 means success on first trial
    if (p - 1.0).abs() < f32::EPSILON {
        return Ok(vec![1u32; count]);
    }

    let log_1mp = (1.0_f64 - p as f64).ln();
    let mut state = seed;
    let mut results = Vec::with_capacity(count);

    for _ in 0..count {
        state = xorshift64(state);
        let u = state_to_uniform_open(state);

        // k = ceil(ln(u) / ln(1-p))
        let k = (u.ln() / log_1mp).ceil();

        // Clamp to valid range [1, u32::MAX]
        let k_clamped = if k < 1.0 {
            1u32
        } else if k > u32::MAX as f64 {
            u32::MAX
        } else {
            k as u32
        };

        results.push(k_clamped);
    }

    Ok(results)
}

/// Simple xorshift64 PRNG step.
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

/// Converts a u64 state to a uniform f64 in (0, 1].
fn state_to_uniform_open(state: u64) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    let upper = (state >> 32) as u32;
    (upper as f64 + 1.0) / (u32::MAX as f64 + 1.0)
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

/// Generates PTX for a GPU-parallel geometric distribution kernel.
///
/// Each thread generates one geometric sample using the inverse CDF:
/// `k = ceil(log(U) / log(1-p))` where U is derived from a Philox-like hash.
///
/// Uses PTX approximate transcendentals for fast logarithm computation.
///
/// Parameters: `(out_ptr, count, p_f32, seed_lo, seed_hi)`
///
/// # Errors
///
/// Returns `RandError::PtxGeneration` on PTX builder failure.
/// Returns `RandError::InvalidSize` for invalid parameters.
pub fn generate_geometric_ptx(p: f32, sm: SmVersion) -> RandResult<String> {
    validate_p(p)?;

    let log_1mp = (1.0_f64 - p as f64).ln() as f32;
    let log_1mp_bits = log_1mp.to_bits();

    let ptx = KernelBuilder::new("geometric_generate")
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

                b.comment("Geometric distribution via inverse CDF");

                // Generate per-thread uniform using hash of seed + gid
                let mixed = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("xor.b32 {mixed}, {seed_lo}, {gid};"));
                let hashed = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {hashed}, {mixed}, {};", 0x45D9F3B_u32));
                let mixed2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("xor.b32 {mixed2}, {hashed}, {seed_hi};"));
                let hashed2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mul.lo.u32 {hashed2}, {mixed2}, {};",
                    0x27D4EB2D_u32
                ));

                // Convert to f32 uniform in (0, 1]
                let u_f32 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {u_f32}, {hashed2};"));
                let scale = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;")); // 2^-32
                let u_scaled = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {u_scaled}, {u_f32}, {scale};"));

                // Clamp away from zero to avoid log(0)
                let eps = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.f32 {eps}, 0f33800000;")); // ~5.96e-8
                let u_safe = b.max_f32(u_scaled, eps);

                // log(u) = lg2(u) * ln(2)
                let lg2_u = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("lg2.approx.f32 {lg2_u}, {u_safe};"));
                let ln2 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.f32 {ln2}, 0f3F317218;")); // ln(2)
                let ln_u = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {ln_u}, {lg2_u}, {ln2};"));

                // k = ceil(ln(u) / ln(1-p))
                let log_1mp_reg = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.b32 {log_1mp_reg}, 0x{log_1mp_bits:08X};"));

                // Division: ln_u / log_1mp
                let ratio = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("div.approx.f32 {ratio}, {ln_u}, {log_1mp_reg};"));

                // ceil
                let ceiled = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rpi.f32.f32 {ceiled}, {ratio};"));

                // Clamp to >= 1
                let one_f = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.f32 {one_f}, 0f3F800000;")); // 1.0
                let clamped = b.max_f32(ceiled, one_f);

                // Convert to u32
                let result = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("cvt.rzi.u32.f32 {result}, {clamped};"));

                // Store
                let addr = b.byte_offset_addr(out_ptr, gid.clone(), 4);
                b.raw_ptx(&format!("st.global.u32 [{addr}], {result};"));

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
    fn validate_rejects_zero_p() {
        assert!(validate_p(0.0).is_err());
    }

    #[test]
    fn validate_rejects_negative_p() {
        assert!(validate_p(-0.5).is_err());
    }

    #[test]
    fn validate_rejects_p_above_one() {
        assert!(validate_p(1.1).is_err());
    }

    #[test]
    fn validate_accepts_valid_p() {
        assert!(validate_p(0.5).is_ok());
        assert!(validate_p(1.0).is_ok());
        assert!(validate_p(0.001).is_ok());
    }

    #[test]
    fn cpu_geometric_p_one() {
        let result = generate_geometric_cpu(10, 1.0, 42);
        assert!(result.is_ok());
        if let Ok(samples) = result {
            assert_eq!(samples.len(), 10);
            for &s in &samples {
                assert_eq!(s, 1, "Geom(1.0) should always be 1");
            }
        }
    }

    #[test]
    fn cpu_geometric_all_positive() {
        let result = generate_geometric_cpu(100, 0.5, 42);
        assert!(result.is_ok());
        if let Ok(samples) = result {
            assert_eq!(samples.len(), 100);
            for &s in &samples {
                assert!(s >= 1, "geometric samples must be >= 1, got {s}");
            }
        }
    }

    #[test]
    fn cpu_geometric_small_p_larger_values() {
        let result = generate_geometric_cpu(100, 0.01, 42);
        assert!(result.is_ok());
        if let Ok(samples) = result {
            // With p=0.01, mean is 100, so some values should be > 1
            let has_large = samples.iter().any(|&s| s > 1);
            assert!(has_large, "Geom(0.01) should produce values > 1");
        }
    }

    #[test]
    fn cpu_geometric_rejects_zero_count() {
        let result = generate_geometric_cpu(0, 0.5, 42);
        assert!(result.is_err());
    }

    #[test]
    fn ptx_generates_f32() {
        let ptx = generate_geometric_ptx(0.5, SmVersion::Sm80);
        assert!(ptx.is_ok());
        if let Ok(ptx_str) = ptx {
            assert!(ptx_str.contains(".entry geometric_generate"));
            assert!(ptx_str.contains("lg2.approx.f32")); // logarithm
            assert!(ptx_str.contains("div.approx.f32")); // division
            assert!(ptx_str.contains("cvt.rpi.f32.f32")); // ceil
        }
    }

    #[test]
    fn ptx_rejects_invalid_p() {
        let result = generate_geometric_ptx(0.0, SmVersion::Sm80);
        assert!(result.is_err());
        let result = generate_geometric_ptx(-0.1, SmVersion::Sm80);
        assert!(result.is_err());
    }

    #[test]
    fn xorshift64_produces_different() {
        let s1 = xorshift64(42);
        let s2 = xorshift64(s1);
        assert_ne!(s1, s2);
    }
}
