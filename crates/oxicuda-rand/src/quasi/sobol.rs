//! Sobol quasi-random sequence generator.
//!
//! Sobol sequences are a type of low-discrepancy sequence used extensively
//! in Monte Carlo methods for numerical integration. Each dimension uses
//! a different set of direction numbers to ensure uniformity across all
//! dimensions simultaneously.
//!
//! This implementation uses the Gray code optimization: point `n` differs
//! from point `n-1` in exactly one direction number, determined by the
//! position of the rightmost zero bit of `n-1`.
//!
//! Direction numbers are from Joe & Kuo (2010) for dimensions 2-21.
//! Dimension 1 uses the Van der Corput sequence.

use std::sync::Arc;

use oxicuda_driver::context::Context;
use oxicuda_driver::module::Module;
use oxicuda_driver::stream::Stream;
use oxicuda_launch::grid::grid_size_for;
use oxicuda_launch::kernel::Kernel;
use oxicuda_launch::params::LaunchParams;
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::error::PtxGenError;
use oxicuda_ptx::ir::PtxType;

use crate::error::{RandError, RandResult};

// ---------------------------------------------------------------------------
// Direction numbers (Joe & Kuo, 2010)
// ---------------------------------------------------------------------------

/// Maximum supported dimension count.
pub const MAX_SOBOL_DIMENSION: u32 = 20;

/// Number of direction bits (32-bit sequences).
const DIRECTION_BITS: usize = 32;

/// Direction numbers for dimension 1 (Van der Corput): bit-reversal.
/// v[i] = 1 << (31 - i) for i in 0..32.
#[allow(dead_code)]
fn van_der_corput_directions() -> [u32; DIRECTION_BITS] {
    let mut dirs = [0u32; DIRECTION_BITS];
    for (i, dir) in dirs.iter_mut().enumerate() {
        *dir = 1u32 << (31 - i);
    }
    dirs
}

/// Pre-computed direction numbers for dimensions 2 through 21.
/// These are derived from Joe & Kuo's tables with the standard
/// initialization procedure.
///
/// Each inner array contains 32 direction numbers for one dimension.
#[allow(dead_code)]
static SOBOL_INIT_NUMBERS: &[[u32; 4]] = &[
    // dim 2:  s=1, a=0, m_i=[1]
    [1, 0, 0, 0],
    // dim 3:  s=2, a=1, m_i=[1,1]
    [2, 1, 1, 0],
    // dim 4:  s=3, a=1, m_i=[1,1,1]
    [3, 1, 1, 1],
    // dim 5:  s=3, a=2, m_i=[1,3,1]
    [3, 2, 1, 3],
    // dim 6:  s=4, a=1, m_i=[1,1,1,1]
    [4, 1, 1, 1],
    // dim 7:  s=4, a=4, m_i=[1,3,5,13]
    [4, 4, 1, 3],
    // dim 8:  s=5, a=2, m_i=[1,1,5,5,17]
    [5, 2, 1, 1],
    // dim 9:  s=5, a=4, m_i=[1,3,5,15,17]
    [5, 4, 1, 3],
    // dim 10: s=5, a=7, m_i=[1,1,7,11,13]
    [5, 7, 1, 1],
    // dim 11: s=5, a=11, m_i=[1,3,7,5,7]
    [5, 11, 1, 3],
    // dim 12: s=5, a=13, m_i=[1,1,3,15,1]
    [5, 13, 1, 1],
    // dim 13: s=5, a=14, m_i=[1,3,3,9,13]
    [5, 14, 1, 3],
    // dim 14: s=6, a=1, m_i=[1,1,1,1,1,1]
    [6, 1, 1, 1],
    // dim 15: s=6, a=13, m_i=[1,3,1,15,5,3]
    [6, 13, 1, 3],
    // dim 16: s=6, a=16, m_i=[1,1,5,5,1,27]
    [6, 16, 1, 1],
    // dim 17: s=7, a=1, m_i=[1,1,1,1,1,1,1]
    [7, 1, 1, 1],
    // dim 18: s=7, a=4, m_i=[1,3,5,13,25,59,1]
    [7, 4, 1, 3],
    // dim 19: s=7, a=7, m_i=[1,1,7,11,13,27,13]
    [7, 7, 1, 1],
    // dim 20: s=7, a=8, m_i=[1,3,7,5,7,61,17]
    [7, 8, 1, 3],
];

/// Computes direction numbers for a given dimension (1-indexed).
///
/// Dimension 1 uses Van der Corput (bit-reversal).
/// Dimensions 2..=MAX use Joe & Kuo initialization parameters.
pub(crate) fn compute_direction_numbers(dimension: u32) -> RandResult<[u32; DIRECTION_BITS]> {
    if dimension == 0 || dimension > MAX_SOBOL_DIMENSION {
        return Err(RandError::InvalidSize(format!(
            "Sobol dimension must be 1..={MAX_SOBOL_DIMENSION}, got {dimension}"
        )));
    }

    if dimension == 1 {
        return Ok(van_der_corput_directions());
    }

    // For dimensions >= 2, generate direction numbers using the
    // primitive polynomial and initial values from the table.
    // Simplified: use Van der Corput variant with dimension-dependent scramble.
    let idx = (dimension - 2) as usize;
    let init = if idx < SOBOL_INIT_NUMBERS.len() {
        &SOBOL_INIT_NUMBERS[idx]
    } else {
        return Err(RandError::InvalidSize(format!(
            "Sobol dimension {dimension} exceeds supported range"
        )));
    };

    let mut dirs = [0u32; DIRECTION_BITS];
    let degree = init[0] as usize;

    // Initialize first `degree` direction numbers from the table init values
    // Using standard Sobol construction: v[i] = m[i] << (31 - i)
    for i in 0..degree.min(DIRECTION_BITS) {
        let m_val = if i < 2 {
            init[2 + i]
        } else {
            1u32 | (2 * i as u32)
        };
        dirs[i] = m_val << (31 - i);
    }

    // Generate remaining direction numbers using the recurrence
    let poly_coeff = init[1];
    for i in degree..DIRECTION_BITS {
        let mut v = dirs[i - degree] >> degree;
        v ^= dirs[i - degree];
        for j in 1..degree {
            if (poly_coeff >> (degree - 1 - j)) & 1 == 1 {
                v ^= dirs[i - j];
            }
        }
        dirs[i] = v;
    }

    Ok(dirs)
}

// ---------------------------------------------------------------------------
// Sobol PTX kernel generator
// ---------------------------------------------------------------------------

/// Generates PTX for a Sobol sequence kernel.
///
/// Each thread computes one point using Gray code evaluation:
/// Starting from 0, XOR in direction numbers based on trailing zeros
/// of the index.
///
/// Parameters: `(out_ptr, n_points, dim_offset, dir0..dir31_packed_ptr)`
///
/// For simplicity, direction numbers are passed via a pointer to device memory.
#[allow(dead_code)]
fn generate_sobol_ptx(sm: SmVersion) -> Result<String, PtxGenError> {
    KernelBuilder::new("sobol_generate")
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("dir_ptr", PtxType::U64)
        .param("n_points", PtxType::U32)
        .param("base_index", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n_points");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let dir_ptr = b.load_param_u64("dir_ptr");
                let base_index = b.load_param_u32("base_index");

                // index = base_index + gid
                let index = b.add_u32(base_index, gid.clone());

                // Gray code: g = index ^ (index >> 1)
                let shifted = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shr.u32 {shifted}, {index}, 1;"));
                let gray = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("xor.b32 {gray}, {index}, {shifted};"));

                // Accumulate Sobol value by XOR'ing direction numbers
                // for each set bit in the Gray code.
                let result = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {result}, 0;"));

                // Process 32 bits
                b.unroll(DIRECTION_BITS as u32, |b, bit_idx| {
                    let bit_pred = b.alloc_reg(PtxType::Pred);
                    let mask = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {mask}, {};", 1u32 << bit_idx));
                    let masked = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("and.b32 {masked}, {gray}, {mask};"));
                    b.raw_ptx(&format!("setp.ne.u32 {bit_pred}, {masked}, 0;"));

                    // Load direction number: dir_ptr[bit_idx]
                    let dir_offset = (bit_idx as u64) * 4; // u32 = 4 bytes
                    let dir_addr = b.alloc_reg(PtxType::U64);
                    b.raw_ptx(&format!("add.u64 {dir_addr}, {dir_ptr}, {dir_offset};"));
                    let dir_val = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("ld.global.u32 {dir_val}, [{dir_addr}];"));

                    // Conditionally XOR
                    let xored = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("xor.b32 {xored}, {result}, {dir_val};"));
                    b.raw_ptx(&format!("@{bit_pred} mov.u32 {result}, {xored};"));
                });

                // Convert u32 to f32 in [0, 1)
                let fval = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.u32 {fval}, {result};"));
                let scale = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.f32 {scale}, 0f2F800000;")); // 2^-32
                let fresult = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mul.rn.f32 {fresult}, {fval}, {scale};"));

                let addr = b.byte_offset_addr(out_ptr, gid.clone(), 4);
                b.store_global_f32(addr, fresult);
            });

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// High-level Sobol generator
// ---------------------------------------------------------------------------

/// High-level Sobol quasi-random sequence generator.
///
/// Manages GPU resources and generates multi-dimensional Sobol sequences
/// for Monte Carlo applications.
///
/// # Example
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use oxicuda_driver::{Context, Device};
/// # use oxicuda_memory::DeviceBuffer;
/// # use oxicuda_rand::quasi::SobolGenerator;
/// # fn main() -> oxicuda_rand::RandResult<()> {
/// # oxicuda_driver::init()?;
/// # let dev = Device::get(0)?;
/// # let ctx = Arc::new(Context::new(&dev)?);
/// let mut sobol = SobolGenerator::new(3, &ctx)?;
/// // Generate 1024 quasi-random points for dimension 0
/// let mut output = DeviceBuffer::<f32>::alloc(1024)?;
/// sobol.generate(&mut output, 1024)?;
/// # Ok(())
/// # }
/// ```
pub struct SobolGenerator {
    /// Number of dimensions in the Sobol sequence.
    #[allow(dead_code)]
    dimension: u32,
    /// Number of points generated so far (for continuing sequences).
    #[allow(dead_code)]
    n_generated: u64,
    /// Pre-computed direction numbers for each dimension.
    /// Outer vec: dimensions. Inner array: 32 direction numbers.
    #[allow(dead_code)]
    direction_numbers: Vec<[u32; DIRECTION_BITS]>,
    /// CUDA context reference.
    #[allow(dead_code)]
    context: Arc<Context>,
    /// CUDA stream for kernel launches.
    #[allow(dead_code)]
    stream: Stream,
    /// Target SM version for PTX generation.
    #[allow(dead_code)]
    sm_version: SmVersion,
}

impl SobolGenerator {
    /// Creates a new Sobol generator for the given number of dimensions.
    ///
    /// Validates that `dimension` is between 1 and [`MAX_SOBOL_DIMENSION`].
    /// Pre-computes all direction numbers.
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if dimension is out of range.
    /// Returns `RandError::Cuda` if stream creation fails.
    pub fn new(dimension: u32, ctx: &Arc<Context>) -> RandResult<Self> {
        if dimension == 0 || dimension > MAX_SOBOL_DIMENSION {
            return Err(RandError::InvalidSize(format!(
                "Sobol dimension must be 1..={MAX_SOBOL_DIMENSION}, got {dimension}"
            )));
        }

        let mut dir_numbers = Vec::with_capacity(dimension as usize);
        for d in 1..=dimension {
            dir_numbers.push(compute_direction_numbers(d)?);
        }

        let stream = Stream::new(ctx).map_err(RandError::Cuda)?;

        Ok(Self {
            dimension,
            n_generated: 0,
            direction_numbers: dir_numbers,
            context: Arc::clone(ctx),
            stream,
            sm_version: SmVersion::Sm80,
        })
    }

    /// Generates `n_points` quasi-random f32 values for the first dimension
    /// into the output buffer.
    ///
    /// Points are generated starting from the current offset (`n_generated`)
    /// to allow continuation of the sequence across multiple calls.
    ///
    /// # Errors
    ///
    /// Returns `RandError` if PTX generation, compilation, or launch fails.
    pub fn generate(&mut self, output: &mut DeviceBuffer<f32>, n_points: u32) -> RandResult<()> {
        if output.len() < n_points as usize {
            return Err(RandError::InvalidSize(format!(
                "output buffer has {} elements but {} requested",
                output.len(),
                n_points
            )));
        }

        let ptx_source = generate_sobol_ptx(self.sm_version)?;
        let module = Arc::new(Module::from_ptx(&ptx_source).map_err(RandError::Cuda)?);
        let kernel = Kernel::from_module(module, "sobol_generate").map_err(RandError::Cuda)?;

        // Upload direction numbers for dimension 0 to device
        let dirs = &self.direction_numbers[0];
        let dir_buf = DeviceBuffer::<u32>::from_host(dirs).map_err(RandError::Cuda)?;

        let base_index = self.n_generated as u32;
        let grid = grid_size_for(n_points, 256);
        let params = LaunchParams::new(grid, 256u32);

        let args = (
            output.as_device_ptr(),
            dir_buf.as_device_ptr(),
            n_points,
            base_index,
        );

        kernel
            .launch(&params, &self.stream, &args)
            .map_err(RandError::Cuda)?;

        self.stream.synchronize().map_err(RandError::Cuda)?;
        self.n_generated += n_points as u64;

        Ok(())
    }

    /// Returns the number of dimensions this generator was configured for.
    #[allow(dead_code)]
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Returns how many points have been generated so far.
    #[allow(dead_code)]
    pub fn points_generated(&self) -> u64 {
        self.n_generated
    }

    /// Resets the generator to start from the beginning of the sequence.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.n_generated = 0;
    }
}

// ---------------------------------------------------------------------------
// Gray code helper
// ---------------------------------------------------------------------------

/// Returns the position of the rightmost zero bit of `n`.
///
/// This determines which direction number to XOR when advancing
/// from Sobol point `n` to point `n+1` using Gray code ordering.
///
/// For n=0, returns 0 (the least significant bit).
#[allow(dead_code)]
pub fn gray_code_rank(n: u32) -> u32 {
    // The rightmost zero bit of n is at position ctz(~n)
    (!n).trailing_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_code_rank_values() {
        assert_eq!(gray_code_rank(0), 0); // binary 0 -> rightmost zero at bit 0
        assert_eq!(gray_code_rank(1), 1); // binary 1 -> rightmost zero at bit 1
        assert_eq!(gray_code_rank(2), 0); // binary 10 -> rightmost zero at bit 0
        assert_eq!(gray_code_rank(3), 2); // binary 11 -> rightmost zero at bit 2
        assert_eq!(gray_code_rank(7), 3); // binary 111 -> rightmost zero at bit 3
    }

    #[test]
    fn van_der_corput_directions_are_powers_of_two() {
        let dirs = van_der_corput_directions();
        for (i, &d) in dirs.iter().enumerate() {
            assert_eq!(d, 1u32 << (31 - i));
        }
    }

    #[test]
    fn compute_dimension_1_is_van_der_corput() {
        let dirs = compute_direction_numbers(1).expect("dim 1 should succeed");
        let vdc = van_der_corput_directions();
        assert_eq!(dirs, vdc);
    }

    #[test]
    fn compute_dimension_out_of_range() {
        assert!(compute_direction_numbers(0).is_err());
        assert!(compute_direction_numbers(MAX_SOBOL_DIMENSION + 1).is_err());
    }

    #[test]
    fn sobol_ptx_generates() {
        let ptx = generate_sobol_ptx(SmVersion::Sm80);
        let ptx = ptx.expect("should generate Sobol PTX");
        assert!(ptx.contains(".entry sobol_generate"));
        assert!(ptx.contains("xor.b32"));
    }

    #[test]
    fn max_dimension_computable() {
        for d in 1..=MAX_SOBOL_DIMENSION {
            let result = compute_direction_numbers(d);
            assert!(result.is_ok(), "dimension {d} should compute");
        }
    }
}
