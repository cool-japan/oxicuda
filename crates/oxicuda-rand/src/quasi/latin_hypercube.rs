//! Latin Hypercube Sampling (LHS) for space-filling designs.
//!
//! LHS divides each dimension into `n` equal intervals and places exactly
//! one sample in each interval.  This ensures better coverage of the
//! parameter space than simple random sampling, making it particularly
//! useful for design of experiments and uncertainty quantification.
//!
//! The implementation uses Fisher--Yates shuffles to generate independent
//! random permutations for each dimension, seeded deterministically from
//! the user-provided seed.
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{RandError, RandResult};

// ---------------------------------------------------------------------------
// Simple xorshift64 PRNG (matches the one in distributions/binomial.rs)
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
// LatinHypercubeSampler
// ---------------------------------------------------------------------------

/// Latin Hypercube Sampling (LHS) generator.
///
/// Generates space-filling designs where each dimension is stratified into
/// `n` equal-probability intervals and each interval is sampled exactly once.
///
/// The per-dimension permutations are generated using a Fisher--Yates shuffle
/// seeded deterministically from the master `seed`.
#[derive(Debug, Clone)]
pub struct LatinHypercubeSampler {
    /// Number of dimensions.
    dimensions: usize,
    /// Master seed for reproducibility.
    seed: u64,
}

impl LatinHypercubeSampler {
    /// Creates a new LHS sampler.
    ///
    /// # Arguments
    ///
    /// * `dimensions` -- Number of dimensions (must be >= 1).
    /// * `seed` -- Master seed for the Fisher--Yates shuffle.
    pub fn new(dimensions: usize, seed: u64) -> Self {
        Self { dimensions, seed }
    }

    /// Returns the number of dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Generates `n` LHS samples on the CPU.
    ///
    /// The output buffer must have at least `n * dimensions` elements.
    /// Values are written in row-major order: the first `dimensions` entries
    /// correspond to sample 0, the next `dimensions` to sample 1, and so on.
    /// Each coordinate lies in [0, 1).
    ///
    /// # Errors
    ///
    /// Returns [`RandError::InvalidSize`] if the output buffer is too small,
    /// `n` is zero, or `dimensions` is zero.
    pub fn generate_cpu(&self, output: &mut [f32], n: usize) -> RandResult<()> {
        if n == 0 {
            return Err(RandError::InvalidSize("n must be > 0".to_string()));
        }
        if self.dimensions == 0 {
            return Err(RandError::InvalidSize("dimensions must be > 0".to_string()));
        }
        let required = n
            .checked_mul(self.dimensions)
            .ok_or_else(|| RandError::InvalidSize("n * dimensions overflow".to_string()))?;
        if output.len() < required {
            return Err(RandError::InvalidSize(format!(
                "output buffer has {} elements but {} required",
                output.len(),
                required
            )));
        }

        let inv_n = 1.0_f32 / n as f32;
        let mut rng_state = self.seed;

        for d in 0..self.dimensions {
            // Create identity permutation [0, 1, ..., n-1]
            let mut perm: Vec<usize> = (0..n).collect();

            // Fisher--Yates shuffle
            for i in (1..n).rev() {
                rng_state = xorshift64(rng_state);
                #[allow(clippy::cast_possible_truncation)]
                let j = (rng_state % (i as u64 + 1)) as usize;
                perm.swap(i, j);
            }

            // Generate stratified samples
            for i in 0..n {
                rng_state = xorshift64(rng_state);
                let u = state_to_uniform(rng_state);
                // sample = (perm[i] + u) / n, giving a value in [0, 1)
                let val = (perm[i] as f32 + u) * inv_n;
                // Clamp to [0, 1) for safety
                output[i * self.dimensions + d] = val.min(1.0 - f32::EPSILON);
            }
        }

        Ok(())
    }

    /// Generates PTX for a GPU LHS kernel.
    ///
    /// Due to the sequential nature of Fisher--Yates shuffles, the GPU kernel
    /// uses a hash-based permutation approximation: each thread computes a
    /// deterministic pseudo-random mapping from its index to a stratum using
    /// a bijective hash.  This is an approximation of true LHS but avoids
    /// the need for global synchronisation.
    ///
    /// Parameters: `(out_ptr, n_points, seed_lo, seed_hi)`
    ///
    /// # Errors
    ///
    /// Returns [`RandError::PtxGeneration`] on PTX builder failure.
    pub fn generate_ptx(&self, n: usize, sm: SmVersion) -> RandResult<String> {
        if n == 0 {
            return Err(RandError::InvalidSize("n must be > 0".to_string()));
        }

        let dims = self.dimensions;
        let seed_lo = self.seed as u32;
        #[allow(clippy::cast_possible_truncation)]
        let seed_hi = (self.seed >> 32) as u32;

        let ptx = KernelBuilder::new("latin_hypercube_generate")
            .target(sm)
            .param("out_ptr", PtxType::U64)
            .param("n_points", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n_points");

                b.if_lt_u32(gid.clone(), n_reg, move |b| {
                    let out_ptr = b.load_param_u64("out_ptr");

                    // inv_n = 1.0 / n
                    let inv_n_f = 1.0_f32 / n as f32;
                    let inv_n_bits = inv_n_f.to_bits();
                    let inv_n_reg = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.b32 {inv_n_reg}, 0x{inv_n_bits:08X};"));

                    for d in 0..dims {
                        b.comment(&format!("dimension {d}"));

                        // Hash-based stratum: mix gid with dimension and seed
                        let mix1 = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!(
                            "xor.b32 {mix1}, {gid}, {};",
                            seed_lo.wrapping_add(d as u32 * 0x9E3779B9)
                        ));
                        let mix2 = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mul.lo.u32 {mix2}, {mix1}, {};", 0x45D9F3B_u32));
                        let mix3 = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!(
                            "xor.b32 {mix3}, {mix2}, {};",
                            seed_hi.wrapping_add(d as u32 * 0x85EBCA6B)
                        ));
                        let mix4 = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mul.lo.u32 {mix4}, {mix3}, {};", 0xC2B2AE35_u32));

                        // stratum = hash % n
                        let n_const = b.alloc_reg(PtxType::U32);
                        #[allow(clippy::cast_possible_truncation)]
                        let n_u32 = n as u32;
                        b.raw_ptx(&format!("mov.u32 {n_const}, {n_u32};"));
                        let quotient = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("div.u32 {quotient}, {mix4}, {n_const};"));
                        let prod = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mul.lo.u32 {prod}, {quotient}, {n_const};"));
                        let stratum = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("sub.u32 {stratum}, {mix4}, {prod};"));

                        // Uniform jitter within stratum
                        let jitter_hash = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!(
                            "xor.b32 {jitter_hash}, {mix4}, {};",
                            0x27D4EB2F_u32
                        ));
                        let jitter_f = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {jitter_f}, {jitter_hash};"));
                        let scale_2m32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {scale_2m32}, 0f2F800000;"));
                        let u_jitter = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {u_jitter}, {jitter_f}, {scale_2m32};"));

                        // value = (stratum + u_jitter) * inv_n
                        let stratum_f = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {stratum_f}, {stratum};"));
                        let sum = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("add.rn.f32 {sum}, {stratum_f}, {u_jitter};"));
                        let val = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {val}, {sum}, {inv_n_reg};"));

                        // Store at out_ptr[gid * dims + d]
                        let dims_const = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {dims_const}, {};", dims));
                        let row = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mul.lo.u32 {row}, {gid}, {dims_const};"));
                        let elem_idx = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("add.u32 {elem_idx}, {row}, {};", d));
                        let addr = b.byte_offset_addr(out_ptr.clone(), elem_idx, 4);
                        b.store_global_f32(addr, val);
                    }
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_creation() {
        let sampler = LatinHypercubeSampler::new(3, 42);
        assert_eq!(sampler.dimensions(), 3);
    }

    #[test]
    fn generate_cpu_values_in_range() {
        let sampler = LatinHypercubeSampler::new(2, 42);
        let mut buf = vec![0.0_f32; 20]; // 10 * 2
        let res = sampler.generate_cpu(&mut buf, 10);
        assert!(res.is_ok());
        for &v in &buf {
            assert!((0.0..1.0).contains(&v), "value {v} out of range [0, 1)");
        }
    }

    #[test]
    fn generate_cpu_stratification() {
        // With n=10 and 1 dimension, each stratum [k/10, (k+1)/10) should
        // have exactly one sample.
        let sampler = LatinHypercubeSampler::new(1, 42);
        let n = 10;
        let mut buf = vec![0.0_f32; n];
        let res = sampler.generate_cpu(&mut buf, n);
        assert!(res.is_ok());

        let mut strata = vec![false; n];
        for &v in &buf {
            let stratum = (v * n as f32).floor() as usize;
            assert!(stratum < n, "stratum {stratum} out of range for value {v}");
            strata[stratum] = true;
        }
        for (i, &occupied) in strata.iter().enumerate() {
            assert!(occupied, "stratum {i} was not filled");
        }
    }

    #[test]
    fn generate_cpu_rejects_zero_n() {
        let sampler = LatinHypercubeSampler::new(1, 42);
        let mut buf = vec![0.0_f32; 10];
        assert!(sampler.generate_cpu(&mut buf, 0).is_err());
    }

    #[test]
    fn generate_cpu_rejects_small_buffer() {
        let sampler = LatinHypercubeSampler::new(3, 42);
        let mut buf = vec![0.0_f32; 5]; // need 3 * 3 = 9
        assert!(sampler.generate_cpu(&mut buf, 3).is_err());
    }

    #[test]
    fn generate_cpu_deterministic() {
        let sampler = LatinHypercubeSampler::new(2, 42);
        let mut buf1 = vec![0.0_f32; 10];
        let mut buf2 = vec![0.0_f32; 10];
        let _ = sampler.generate_cpu(&mut buf1, 5);
        let _ = sampler.generate_cpu(&mut buf2, 5);
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn generate_ptx_compiles() {
        let sampler = LatinHypercubeSampler::new(2, 42);
        let ptx = sampler.generate_ptx(10, SmVersion::Sm80);
        assert!(ptx.is_ok());
        if let Ok(ptx_str) = ptx {
            assert!(ptx_str.contains(".entry latin_hypercube_generate"));
        }
    }

    #[test]
    fn generate_ptx_rejects_zero_n() {
        let sampler = LatinHypercubeSampler::new(1, 42);
        assert!(sampler.generate_ptx(0, SmVersion::Sm80).is_err());
    }

    #[test]
    fn zero_dimensions_rejected() {
        let sampler = LatinHypercubeSampler::new(0, 42);
        let mut buf = vec![0.0_f32; 10];
        assert!(sampler.generate_cpu(&mut buf, 5).is_err());
    }
}
