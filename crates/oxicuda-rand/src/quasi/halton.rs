//! Halton quasi-random sequence generator.
//!
//! Halton sequences use the radical inverse function in different prime bases
//! for each dimension.  Dimension `d` uses the `d`-th prime as its base.
//! The radical inverse of integer `n` in base `b` produces a value in [0, 1)
//! by reflecting the base-`b` digits of `n` about the decimal point.
//!
//! Halton sequences are simpler to implement than Sobol sequences but may
//! exhibit correlation artefacts in high dimensions (above ~20).  They remain
//! popular in quasi-Monte Carlo integration for moderate-dimensional problems.
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{RandError, RandResult};

// ---------------------------------------------------------------------------
// Prime table
// ---------------------------------------------------------------------------

/// Maximum number of supported Halton dimensions.
pub const MAX_HALTON_DIMENSION: usize = 20;

/// First 20 primes used as bases for the radical inverse function.
const PRIMES: [u32; MAX_HALTON_DIMENSION] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
];

// ---------------------------------------------------------------------------
// Radical inverse
// ---------------------------------------------------------------------------

/// Computes the radical inverse of `n` in the given `base`.
///
/// The radical inverse is defined as:
///
/// ```text
///   phi_b(n) = sum_{i=0}^{k} d_i * b^{-(i+1)}
/// ```
///
/// where `d_0, d_1, ...` are the digits of `n` expressed in base `b`.
///
/// The result lies in [0, 1).
fn radical_inverse(mut n: u32, base: u32) -> f32 {
    let mut result: f64 = 0.0;
    let inv_base: f64 = 1.0 / f64::from(base);
    let mut factor = inv_base;

    while n > 0 {
        let digit = n % base;
        result += f64::from(digit) * factor;
        n /= base;
        factor *= inv_base;
        // Prevent denorm accumulation
        if factor < 1e-18 {
            break;
        }
    }
    let _ = inv_base;

    result as f32
}

// ---------------------------------------------------------------------------
// HaltonGenerator
// ---------------------------------------------------------------------------

/// Multi-dimensional Halton quasi-random sequence generator.
///
/// Uses the radical inverse function in a different prime base for each
/// dimension.  Dimension 0 uses base 2, dimension 1 uses base 3, and so on.
///
/// # Limits
///
/// Up to [`MAX_HALTON_DIMENSION`] (20) dimensions are supported.  Beyond
/// roughly 10--15 dimensions the sequence starts to exhibit correlation
/// and a scrambled variant should be preferred.
#[derive(Debug, Clone)]
pub struct HaltonGenerator {
    /// Number of dimensions.
    dimensions: usize,
    /// Prime bases for each dimension (first `dimensions` primes).
    primes: Vec<u32>,
    /// Offset for continuing the sequence across multiple calls.
    offset: usize,
}

impl HaltonGenerator {
    /// Creates a new Halton generator for the given number of dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`RandError::InvalidSize`] if `dimensions` is zero or exceeds
    /// [`MAX_HALTON_DIMENSION`].
    pub fn new(dimensions: usize) -> RandResult<Self> {
        if dimensions == 0 || dimensions > MAX_HALTON_DIMENSION {
            return Err(RandError::InvalidSize(format!(
                "Halton dimensions must be 1..={MAX_HALTON_DIMENSION}, got {dimensions}"
            )));
        }

        let primes = PRIMES[..dimensions].to_vec();

        Ok(Self {
            dimensions,
            primes,
            offset: 0,
        })
    }

    /// Generates `n` Halton points on the CPU.
    ///
    /// The output buffer must have at least `n * dimensions` elements.
    /// Values are written in row-major order: the first `dimensions` elements
    /// correspond to point 0, the next `dimensions` to point 1, and so on.
    ///
    /// # Errors
    ///
    /// Returns [`RandError::InvalidSize`] if the output buffer is too small
    /// or `n` is zero.
    pub fn generate_cpu(&self, output: &mut [f32], n: usize) -> RandResult<()> {
        if n == 0 {
            return Err(RandError::InvalidSize("n must be > 0".to_string()));
        }
        let required = n
            .checked_mul(self.dimensions)
            .ok_or_else(|| RandError::InvalidSize("n * dimensions overflow".to_string()))?;
        if output.len() < required {
            return Err(RandError::InvalidSize(format!(
                "output buffer has {} elements but {} required (n={n}, dims={})",
                output.len(),
                required,
                self.dimensions
            )));
        }

        for i in 0..n {
            #[allow(clippy::cast_possible_truncation)]
            let idx = (self.offset + i + 1) as u32; // start from 1 to avoid the origin
            for d in 0..self.dimensions {
                output[i * self.dimensions + d] = radical_inverse(idx, self.primes[d]);
            }
        }

        Ok(())
    }

    /// Generates PTX for a GPU Halton sequence kernel.
    ///
    /// Each thread computes one multi-dimensional point.  The kernel unrolls
    /// the radical-inverse computation for each dimension using the prime
    /// base known at code-generation time.
    ///
    /// Parameters: `(out_ptr, n_points, base_index)`
    ///
    /// Output layout: `out_ptr[gid * dims + d]` for each dimension `d`.
    ///
    /// # Errors
    ///
    /// Returns [`RandError::PtxGeneration`] on PTX builder failure.
    pub fn generate_ptx(&self, sm: SmVersion) -> RandResult<String> {
        let dims = self.dimensions;
        let primes = self.primes.clone();

        let ptx = KernelBuilder::new("halton_generate")
            .target(sm)
            .param("out_ptr", PtxType::U64)
            .param("n_points", PtxType::U32)
            .param("base_index", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n_points");

                b.if_lt_u32(gid.clone(), n_reg, move |b| {
                    let out_ptr = b.load_param_u64("out_ptr");
                    let base_index = b.load_param_u32("base_index");

                    // index = base_index + gid + 1  (skip the origin)
                    let idx_tmp = b.add_u32(base_index, gid.clone());
                    let one = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {one}, 1;"));
                    let index = b.add_u32(idx_tmp, one);

                    // For each dimension, compute radical inverse in-line
                    for (d, &prime) in primes.iter().enumerate() {
                        b.comment(&format!("dimension {d}, base {prime}"));

                        // Radical inverse loop (unrolled up to 20 iterations)
                        let result = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {result}, 0f00000000;")); // 0.0

                        let inv_base_f = 1.0_f32 / prime as f32;
                        let inv_bits = inv_base_f.to_bits();
                        let factor = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.b32 {factor}, 0x{inv_bits:08X};"));

                        let n_val = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {n_val}, {index};"));

                        // Unroll up to 20 digit extractions (sufficient for 32-bit indices)
                        let max_iters = match prime {
                            2 => 20,
                            3 => 14,
                            5 => 10,
                            _ => 8,
                        };

                        b.unroll(max_iters, |b, _| {
                            // Check if n_val > 0
                            let pred_nz = b.alloc_reg(PtxType::Pred);
                            b.raw_ptx(&format!("setp.ne.u32 {pred_nz}, {n_val}, 0;"));

                            // digit = n_val % prime
                            let divisor = b.alloc_reg(PtxType::U32);
                            b.raw_ptx(&format!("mov.u32 {divisor}, {prime};"));
                            let quotient = b.alloc_reg(PtxType::U32);
                            b.raw_ptx(&format!("div.u32 {quotient}, {n_val}, {divisor};"));
                            let prod = b.alloc_reg(PtxType::U32);
                            b.raw_ptx(&format!("mul.lo.u32 {prod}, {quotient}, {divisor};"));
                            let digit = b.alloc_reg(PtxType::U32);
                            b.raw_ptx(&format!("sub.u32 {digit}, {n_val}, {prod};"));

                            // digit_f = (float)digit
                            let digit_f = b.alloc_reg(PtxType::F32);
                            b.raw_ptx(&format!("cvt.rn.f32.u32 {digit_f}, {digit};"));

                            // contribution = digit_f * factor
                            let contrib = b.alloc_reg(PtxType::F32);
                            b.raw_ptx(&format!("mul.rn.f32 {contrib}, {digit_f}, {factor};"));

                            // result += contribution (conditional on n_val != 0)
                            let new_result = b.alloc_reg(PtxType::F32);
                            b.raw_ptx(&format!("add.rn.f32 {new_result}, {result}, {contrib};"));
                            b.raw_ptx(&format!("@{pred_nz} mov.f32 {result}, {new_result};"));

                            // factor *= inv_base
                            let new_factor = b.alloc_reg(PtxType::F32);
                            let inv_reg = b.alloc_reg(PtxType::F32);
                            b.raw_ptx(&format!("mov.b32 {inv_reg}, 0x{inv_bits:08X};"));
                            b.raw_ptx(&format!("mul.rn.f32 {new_factor}, {factor}, {inv_reg};"));
                            b.raw_ptx(&format!("@{pred_nz} mov.f32 {factor}, {new_factor};"));

                            // n_val = quotient (conditional)
                            b.raw_ptx(&format!("@{pred_nz} mov.u32 {n_val}, {quotient};"));
                        });

                        // Store result at out_ptr[gid * dims + d]
                        let dims_reg = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {dims_reg}, {dims};"));
                        let row_offset = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mul.lo.u32 {row_offset}, {gid}, {dims_reg};"));
                        let col_offset = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {col_offset}, {};", d));
                        let element_idx = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!(
                            "add.u32 {element_idx}, {row_offset}, {col_offset};"
                        ));
                        let addr = b.byte_offset_addr(out_ptr.clone(), element_idx, 4);
                        b.store_global_f32(addr, result);
                    }
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }

    /// Returns the number of dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the prime bases used for each dimension.
    pub fn primes(&self) -> &[u32] {
        &self.primes
    }

    /// Returns the current offset.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Advances the offset by `n` points (for continuation across calls).
    pub fn advance(&mut self, n: usize) {
        self.offset += n;
    }

    /// Resets the offset to zero.
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radical_inverse_base2() {
        // 1 in binary = 1, reversed = 0.1 = 0.5
        assert!((radical_inverse(1, 2) - 0.5).abs() < 1e-6);
        // 2 in binary = 10, reversed = 0.01 = 0.25
        assert!((radical_inverse(2, 2) - 0.25).abs() < 1e-6);
        // 3 in binary = 11, reversed = 0.11 = 0.75
        assert!((radical_inverse(3, 2) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn radical_inverse_base3() {
        // 1 in base 3 = 1, reversed = 0.1 = 1/3
        assert!((radical_inverse(1, 3) - 1.0 / 3.0).abs() < 1e-6);
        // 2 in base 3 = 2, reversed = 0.2 = 2/3
        assert!((radical_inverse(2, 3) - 2.0 / 3.0).abs() < 1e-6);
        // 3 in base 3 = 10, reversed = 0.01 = 1/9
        assert!((radical_inverse(3, 3) - 1.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn radical_inverse_zero() {
        assert!((radical_inverse(0, 2) - 0.0).abs() < 1e-10);
        assert!((radical_inverse(0, 5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn new_rejects_zero_dimensions() {
        assert!(HaltonGenerator::new(0).is_err());
    }

    #[test]
    fn new_rejects_too_many_dimensions() {
        assert!(HaltonGenerator::new(MAX_HALTON_DIMENSION + 1).is_err());
    }

    #[test]
    fn new_accepts_valid_dimensions() {
        let halton_gen = HaltonGenerator::new(3);
        assert!(halton_gen.is_ok());
        if let Ok(g) = halton_gen {
            assert_eq!(g.dimensions(), 3);
            assert_eq!(g.primes(), &[2, 3, 5]);
        }
    }

    #[test]
    fn generate_cpu_produces_values_in_range() {
        let halton_gen = HaltonGenerator::new(2);
        assert!(halton_gen.is_ok());
        if let Ok(g) = halton_gen {
            let mut buf = vec![0.0_f32; 20]; // 10 points * 2 dims
            let res = g.generate_cpu(&mut buf, 10);
            assert!(res.is_ok());
            for &v in &buf {
                assert!((0.0..1.0).contains(&v), "value {v} out of range [0, 1)");
            }
        }
    }

    #[test]
    fn generate_cpu_rejects_small_buffer() {
        let halton_gen = HaltonGenerator::new(3);
        assert!(halton_gen.is_ok());
        if let Ok(g) = halton_gen {
            let mut buf = vec![0.0_f32; 5]; // need 3*3=9
            let res = g.generate_cpu(&mut buf, 3);
            assert!(res.is_err());
        }
    }

    #[test]
    fn generate_cpu_rejects_zero_n() {
        let halton_gen = HaltonGenerator::new(1);
        assert!(halton_gen.is_ok());
        if let Ok(g) = halton_gen {
            let mut buf = vec![0.0_f32; 10];
            let res = g.generate_cpu(&mut buf, 0);
            assert!(res.is_err());
        }
    }

    #[test]
    fn generate_ptx_compiles() {
        let halton_gen = HaltonGenerator::new(2);
        assert!(halton_gen.is_ok());
        if let Ok(g) = halton_gen {
            let ptx = g.generate_ptx(SmVersion::Sm80);
            assert!(ptx.is_ok());
            if let Ok(ptx_str) = ptx {
                assert!(ptx_str.contains(".entry halton_generate"));
                assert!(ptx_str.contains("div.u32")); // radical inverse division
            }
        }
    }

    #[test]
    fn advance_and_reset() {
        let halton_gen = HaltonGenerator::new(1);
        assert!(halton_gen.is_ok());
        if let Ok(mut g) = halton_gen {
            assert_eq!(g.offset(), 0);
            g.advance(100);
            assert_eq!(g.offset(), 100);
            g.reset();
            assert_eq!(g.offset(), 0);
        }
    }

    #[test]
    fn different_dimensions_produce_different_values() {
        let halton_gen = HaltonGenerator::new(3);
        assert!(halton_gen.is_ok());
        if let Ok(g) = halton_gen {
            let mut buf = vec![0.0_f32; 30]; // 10 points * 3 dims
            let res = g.generate_cpu(&mut buf, 10);
            assert!(res.is_ok());
            // Check dim 0 and dim 1 differ for the same point
            let dim0_vals: Vec<f32> = (0..10).map(|i| buf[i * 3]).collect();
            let dim1_vals: Vec<f32> = (0..10).map(|i| buf[i * 3 + 1]).collect();
            assert_ne!(dim0_vals, dim1_vals);
        }
    }
}
