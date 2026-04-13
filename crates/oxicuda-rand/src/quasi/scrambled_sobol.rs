//! Scrambled Sobol sequences with Owen's scrambling.
//!
//! Applies random digital shifts and bit-reversal permutations to the base
//! Sobol sequence to improve equidistribution while maintaining the
//! low-discrepancy property.
//!
//! Owen's scrambling:
//! 1. XOR each Sobol value with a dimension-specific random mask
//!    (random digital shift).
//! 2. Apply a bit-reversal permutation to further decorrelate dimensions.
//!
//! The scrambled sequence retains the convergence rate O(N^{-1} * (log N)^d)
//! while eliminating the artifacts that plain Sobol sequences can exhibit
//! in certain integrands.
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{RandError, RandResult};
use crate::quasi::sobol::{self, MAX_SOBOL_DIMENSION};

// ---------------------------------------------------------------------------
// Scramble seed generation (deterministic from user seed)
// ---------------------------------------------------------------------------

/// Number of direction bits (32-bit sequences).
const DIRECTION_BITS: usize = 32;

/// Generates per-dimension scramble seeds from a master seed using a
/// simple hash function (SplitMix64-style).
fn generate_scramble_seeds(dimensions: usize, seed: u64) -> Vec<u32> {
    let mut seeds = Vec::with_capacity(dimensions);
    let mut state = seed;
    for _ in 0..dimensions {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        #[allow(clippy::cast_possible_truncation)]
        seeds.push(z as u32);
    }
    seeds
}

/// Applies Owen's scrambling to a single Sobol value.
///
/// 1. XOR with the dimension-specific random mask (digital shift).
/// 2. Bit-reverse the result for decorrelation.
/// 3. XOR again for additional mixing.
fn scramble_value(sobol_val: u32, scramble_seed: u32) -> u32 {
    // Step 1: Random digital shift (XOR)
    let shifted = sobol_val ^ scramble_seed;

    // Step 2: Bit-reversal permutation
    let reversed = shifted.reverse_bits();

    // Step 3: Final XOR mixing with rotated seed
    reversed ^ scramble_seed.rotate_left(16)
}

// ---------------------------------------------------------------------------
// Scrambled Sobol generator
// ---------------------------------------------------------------------------

/// Scrambled Sobol quasi-random sequence generator.
///
/// Wraps a base Sobol sequence and applies Owen's scrambling to each
/// dimension independently. The scrambling preserves the low-discrepancy
/// property while improving uniformity for integrands with certain
/// structure.
///
/// # Scrambling method
///
/// For each dimension `d`:
/// 1. Generate the base Sobol value `v_d` using direction numbers.
/// 2. XOR `v_d` with `scramble_seeds[d]` (random digital shift).
/// 3. Apply bit-reversal permutation for decorrelation.
/// 4. XOR again with a rotated seed for mixing.
///
/// The scramble seeds are deterministically derived from the user-provided
/// master seed, so results are reproducible.
#[derive(Debug)]
pub struct ScrambledSobolGenerator {
    /// Number of dimensions.
    dimensions: usize,
    /// Per-dimension scramble masks.
    scramble_seeds: Vec<u32>,
    /// Pre-computed direction numbers for each dimension.
    direction_numbers: Vec<[u32; DIRECTION_BITS]>,
    /// Number of points generated so far.
    n_generated: u64,
}

impl ScrambledSobolGenerator {
    /// Creates a new scrambled Sobol generator.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Number of dimensions (1..=MAX_SOBOL_DIMENSION)
    /// * `seed` - Master seed for generating per-dimension scramble masks
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if dimensions is out of range.
    pub fn new(dimensions: usize, seed: u64) -> RandResult<Self> {
        if dimensions == 0 || dimensions > MAX_SOBOL_DIMENSION as usize {
            return Err(RandError::InvalidSize(format!(
                "scrambled Sobol dimensions must be 1..={MAX_SOBOL_DIMENSION}, got {dimensions}"
            )));
        }

        let scramble_seeds = generate_scramble_seeds(dimensions, seed);

        let mut direction_numbers = Vec::with_capacity(dimensions);
        for d in 1..=dimensions {
            #[allow(clippy::cast_possible_truncation)]
            let dirs = sobol::compute_direction_numbers(d as u32)?;
            direction_numbers.push(dirs);
        }

        Ok(Self {
            dimensions,
            scramble_seeds,
            direction_numbers,
            n_generated: 0,
        })
    }

    /// Returns the number of dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns how many points have been generated.
    pub fn points_generated(&self) -> u64 {
        self.n_generated
    }

    /// Returns the scramble seeds (one per dimension).
    pub fn scramble_seeds(&self) -> &[u32] {
        &self.scramble_seeds
    }

    /// Resets the generator to restart from the beginning.
    pub fn reset(&mut self) {
        self.n_generated = 0;
    }

    /// Generates scrambled Sobol values on the CPU for the given dimension.
    ///
    /// Produces `n_points` scrambled quasi-random f32 values in [0, 1)
    /// for dimension `dim` (0-indexed).
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `dim >= dimensions` or
    /// `n_points` is zero.
    pub fn generate_cpu(&self, dim: usize, n_points: usize) -> RandResult<Vec<f32>> {
        if dim >= self.dimensions {
            return Err(RandError::InvalidSize(format!(
                "dimension {dim} out of range (max {})",
                self.dimensions - 1
            )));
        }
        if n_points == 0 {
            return Err(RandError::InvalidSize("n_points must be > 0".to_string()));
        }

        let dirs = &self.direction_numbers[dim];
        let seed = self.scramble_seeds[dim];
        let scale = 1.0_f32 / (1u64 << 32) as f32;

        let mut result = Vec::with_capacity(n_points);
        let mut sobol_val: u32 = 0;

        let base = self.n_generated;
        for i in 0..n_points {
            let idx = base + i as u64;
            if idx == 0 {
                sobol_val = 0;
            } else {
                // Gray code: find rightmost zero bit of (idx - 1)
                #[allow(clippy::cast_possible_truncation)]
                let rank = sobol::gray_code_rank((idx - 1) as u32);
                let dir_idx = (rank as usize).min(DIRECTION_BITS - 1);
                sobol_val ^= dirs[dir_idx];
            }

            let scrambled = scramble_value(sobol_val, seed);
            result.push(scrambled as f32 * scale);
        }

        Ok(result)
    }

    /// Generates PTX for the scrambled Sobol kernel.
    ///
    /// Each thread computes one scrambled point using Gray code evaluation
    /// with the scrambling applied inline.
    ///
    /// Parameters: `(out_ptr, dir_ptr, n_points, base_index, scramble_seed)`
    ///
    /// # Errors
    ///
    /// Returns `RandError::PtxGeneration` on PTX builder failure.
    pub fn generate_ptx(&self, sm: SmVersion) -> RandResult<String> {
        let ptx = KernelBuilder::new("scrambled_sobol_generate")
            .target(sm)
            .param("out_ptr", PtxType::U64)
            .param("dir_ptr", PtxType::U64)
            .param("n_points", PtxType::U32)
            .param("base_index", PtxType::U32)
            .param("scramble_seed", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n_points");

                b.if_lt_u32(gid.clone(), n_reg, move |b| {
                    let out_ptr = b.load_param_u64("out_ptr");
                    let dir_ptr = b.load_param_u64("dir_ptr");
                    let base_index = b.load_param_u32("base_index");
                    let seed = b.load_param_u32("scramble_seed");

                    // index = base_index + gid
                    let index = b.add_u32(base_index, gid.clone());

                    // Gray code: g = index ^ (index >> 1)
                    let shifted = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("shr.u32 {shifted}, {index}, 1;"));
                    let gray = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("xor.b32 {gray}, {index}, {shifted};"));

                    // Accumulate Sobol value
                    let sobol_val = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {sobol_val}, 0;"));

                    b.unroll(DIRECTION_BITS as u32, |b, bit_idx| {
                        let bit_pred = b.alloc_reg(PtxType::Pred);
                        let mask = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {mask}, {};", 1u32 << bit_idx));
                        let masked = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("and.b32 {masked}, {gray}, {mask};"));
                        b.raw_ptx(&format!("setp.ne.u32 {bit_pred}, {masked}, 0;"));

                        let dir_offset = (bit_idx as u64) * 4;
                        let dir_addr = b.alloc_reg(PtxType::U64);
                        b.raw_ptx(&format!("add.u64 {dir_addr}, {dir_ptr}, {dir_offset};"));
                        let dir_val = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("ld.global.u32 {dir_val}, [{dir_addr}];"));

                        let xored = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("xor.b32 {xored}, {sobol_val}, {dir_val};"));
                        b.raw_ptx(&format!("@{bit_pred} mov.u32 {sobol_val}, {xored};"));
                    });

                    // Owen's scrambling step 1: XOR with seed
                    let scrambled1 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("xor.b32 {scrambled1}, {sobol_val}, {seed};"));

                    // Owen's scrambling step 2: bit reversal
                    let reversed = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("brev.b32 {reversed}, {scrambled1};"));

                    // Owen's scrambling step 3: XOR with rotated seed
                    let rotated_seed = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {rotated_seed}, {seed};"));
                    // Rotate left by 16: (seed << 16) | (seed >> 16)
                    let shl16 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("shl.b32 {shl16}, {seed}, 16;"));
                    let shr16 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("shr.b32 {shr16}, {seed}, 16;"));
                    b.raw_ptx(&format!("or.b32 {rotated_seed}, {shl16}, {shr16};"));

                    let scrambled_final = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "xor.b32 {scrambled_final}, {reversed}, {rotated_seed};"
                    ));

                    // Convert to f32 in [0, 1)
                    let fval = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("cvt.rn.f32.u32 {fval}, {scrambled_final};"));
                    let scale = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;")); // 2^-32
                    let fresult = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {fresult}, {fval}, {scale};"));

                    let addr = b.byte_offset_addr(out_ptr, gid.clone(), 4);
                    b.store_global_f32(addr, fresult);
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
    fn new_rejects_zero_dimensions() {
        let result = ScrambledSobolGenerator::new(0, 42);
        assert!(result.is_err());
    }

    #[test]
    fn new_rejects_too_many_dimensions() {
        let result = ScrambledSobolGenerator::new(MAX_SOBOL_DIMENSION as usize + 1, 42);
        assert!(result.is_err());
    }

    #[test]
    fn new_accepts_valid_dimensions() {
        let result = ScrambledSobolGenerator::new(5, 12345);
        assert!(result.is_ok());
        if let Ok(g) = result {
            assert_eq!(g.dimensions(), 5);
            assert_eq!(g.points_generated(), 0);
            assert_eq!(g.scramble_seeds().len(), 5);
        }
    }

    #[test]
    fn scramble_seeds_are_deterministic() {
        let seeds1 = generate_scramble_seeds(3, 42);
        let seeds2 = generate_scramble_seeds(3, 42);
        assert_eq!(seeds1, seeds2);
    }

    #[test]
    fn scramble_seeds_differ_across_seeds() {
        let seeds1 = generate_scramble_seeds(3, 42);
        let seeds2 = generate_scramble_seeds(3, 99);
        assert_ne!(seeds1, seeds2);
    }

    #[test]
    fn scramble_seeds_differ_across_dimensions() {
        let seeds = generate_scramble_seeds(4, 42);
        // All 4 seeds should be distinct (with overwhelming probability)
        for i in 0..seeds.len() {
            for j in (i + 1)..seeds.len() {
                assert_ne!(seeds[i], seeds[j], "seeds [{i}] and [{j}] collided");
            }
        }
    }

    #[test]
    fn scramble_value_changes_input() {
        let original = 0x12345678_u32;
        let scrambled = scramble_value(original, 0xABCDEF01);
        assert_ne!(original, scrambled);
    }

    #[test]
    fn cpu_generate_produces_values_in_range() {
        let generator = ScrambledSobolGenerator::new(1, 42);
        assert!(generator.is_ok());
        if let Ok(g) = generator {
            let values = g.generate_cpu(0, 100);
            assert!(values.is_ok());
            if let Ok(vals) = values {
                assert_eq!(vals.len(), 100);
                for &v in &vals {
                    assert!((0.0..1.0).contains(&v), "value {v} out of range [0, 1)");
                }
            }
        }
    }

    #[test]
    fn cpu_generate_rejects_invalid_dimension() {
        let generator = ScrambledSobolGenerator::new(3, 42);
        assert!(generator.is_ok());
        if let Ok(g) = generator {
            let result = g.generate_cpu(3, 10);
            assert!(result.is_err());
        }
    }

    #[test]
    fn cpu_generate_rejects_zero_points() {
        let generator = ScrambledSobolGenerator::new(1, 42);
        assert!(generator.is_ok());
        if let Ok(g) = generator {
            let result = g.generate_cpu(0, 0);
            assert!(result.is_err());
        }
    }

    #[test]
    fn ptx_generates_successfully() {
        let generator = ScrambledSobolGenerator::new(1, 42);
        assert!(generator.is_ok());
        if let Ok(g) = generator {
            let ptx = g.generate_ptx(SmVersion::Sm80);
            assert!(ptx.is_ok());
            if let Ok(ptx_str) = ptx {
                assert!(ptx_str.contains(".entry scrambled_sobol_generate"));
                assert!(ptx_str.contains("xor.b32")); // scrambling
                assert!(ptx_str.contains("brev.b32")); // bit reversal
            }
        }
    }

    #[test]
    fn reset_clears_state() {
        let generator = ScrambledSobolGenerator::new(2, 42);
        assert!(generator.is_ok());
        if let Ok(mut g) = generator {
            g.n_generated = 100;
            assert_eq!(g.points_generated(), 100);
            g.reset();
            assert_eq!(g.points_generated(), 0);
        }
    }
}
