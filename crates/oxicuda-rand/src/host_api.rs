//! cuRAND-compatible host-side API for random number generation.
//!
//! This module provides [`CurandGenerator`], a host-side random number generator
//! matching the semantics of NVIDIA's `curandCreateGenerator` interface. On
//! platforms without GPU access (e.g., macOS), the generator fills buffers
//! directly using pure Rust implementations of the same algorithms used by the
//! GPU PTX kernels.
//!
//! # Supported generator types
//!
//! - [`CurandRngType::PseudoDefault`] / [`CurandRngType::PseudoXorwow`] --
//!   XORWOW with Weyl sequence
//! - [`CurandRngType::PseudoPhilox4_32_10`] -- Philox-4x32-10 counter-based
//! - [`CurandRngType::PseudoMrg32k3a`] -- Combined MRG with highest quality
//! - [`CurandRngType::QuasiDefault`] / [`CurandRngType::QuasiSobol32`] -- Sobol
//! - [`CurandRngType::QuasiScrambledSobol32`] -- Scrambled Sobol
//!
//! # Example
//!
//! ```rust
//! use oxicuda_rand::host_api::{CurandGenerator, CurandRngType};
//!
//! let mut rng = CurandGenerator::new(CurandRngType::PseudoDefault).unwrap();
//! rng.set_seed(42);
//! let values = rng.generate_uniform_f32(1000).unwrap();
//! assert_eq!(values.len(), 1000);
//! ```

use crate::error::{RandError, RandResult};

// ---------------------------------------------------------------------------
// cuRAND RNG type enum
// ---------------------------------------------------------------------------

/// Generator type selection, matching cuRAND's `curandRngType_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CurandRngType {
    /// Default pseudo-random generator (XORWOW).
    PseudoDefault,
    /// XORWOW generator with Weyl sequence.
    PseudoXorwow,
    /// MRG32k3a combined multiple recursive generator.
    PseudoMrg32k3a,
    /// Philox-4x32-10 counter-based generator.
    PseudoPhilox4_32_10,
    /// Default quasi-random generator (Sobol 32-bit).
    QuasiDefault,
    /// Sobol 32-bit quasi-random generator.
    QuasiSobol32,
    /// Scrambled Sobol 32-bit quasi-random generator.
    QuasiScrambledSobol32,
}

impl CurandRngType {
    /// Returns `true` if this is a pseudo-random generator type.
    pub fn is_pseudo(&self) -> bool {
        matches!(
            self,
            Self::PseudoDefault
                | Self::PseudoXorwow
                | Self::PseudoMrg32k3a
                | Self::PseudoPhilox4_32_10
        )
    }

    /// Returns `true` if this is a quasi-random generator type.
    pub fn is_quasi(&self) -> bool {
        !self.is_pseudo()
    }
}

impl std::fmt::Display for CurandRngType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PseudoDefault => write!(f, "CURAND_RNG_PSEUDO_DEFAULT"),
            Self::PseudoXorwow => write!(f, "CURAND_RNG_PSEUDO_XORWOW"),
            Self::PseudoMrg32k3a => write!(f, "CURAND_RNG_PSEUDO_MRG32K3A"),
            Self::PseudoPhilox4_32_10 => write!(f, "CURAND_RNG_PSEUDO_PHILOX4_32_10"),
            Self::QuasiDefault => write!(f, "CURAND_RNG_QUASI_DEFAULT"),
            Self::QuasiSobol32 => write!(f, "CURAND_RNG_QUASI_SOBOL32"),
            Self::QuasiScrambledSobol32 => write!(f, "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32"),
        }
    }
}

// ---------------------------------------------------------------------------
// cuRAND ordering enum
// ---------------------------------------------------------------------------

/// Output ordering mode, matching cuRAND's `curandOrdering_t`.
///
/// Controls the trade-off between determinism and performance in output
/// ordering across parallel threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CurandOrdering {
    /// Default ordering -- best balance of determinism and performance.
    Default,
    /// Best performance ordering -- may sacrifice strict reproducibility
    /// for throughput on multi-SM launches.
    Best,
    /// Fully seeded ordering -- every output element is determined solely
    /// by the seed and its position index, ensuring reproducibility across
    /// different GPU configurations.
    Seeded,
}

impl std::fmt::Display for CurandOrdering {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "CURAND_ORDERING_PSEUDO_DEFAULT"),
            Self::Best => write!(f, "CURAND_ORDERING_PSEUDO_BEST"),
            Self::Seeded => write!(f, "CURAND_ORDERING_PSEUDO_SEEDED"),
        }
    }
}

// ---------------------------------------------------------------------------
// cuRAND status enum
// ---------------------------------------------------------------------------

/// Status codes matching cuRAND's `curandStatus_t` for compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CurandStatus {
    /// Operation completed successfully.
    Success,
    /// cuRAND library version mismatch.
    VersionMismatch,
    /// Generator not initialized.
    NotInitialized,
    /// Memory allocation failed.
    AllocationFailed,
    /// Incorrect type for this operation.
    TypeError,
    /// Argument out of valid range.
    OutOfRange,
    /// Requested feature requires too many threads.
    LengthNotMultiple,
    /// GPU does not have double-precision capability.
    DoublePrecisionRequired,
    /// Kernel launch failed.
    LaunchFailure,
    /// Pre-existing conditions caused failure.
    PreexistingFailure,
    /// Internal library error.
    InternalError,
}

impl CurandStatus {
    /// Returns `true` if the status indicates success.
    pub fn is_success(self) -> bool {
        self == Self::Success
    }
}

impl std::fmt::Display for CurandStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success => write!(f, "CURAND_STATUS_SUCCESS"),
            Self::VersionMismatch => write!(f, "CURAND_STATUS_VERSION_MISMATCH"),
            Self::NotInitialized => write!(f, "CURAND_STATUS_NOT_INITIALIZED"),
            Self::AllocationFailed => write!(f, "CURAND_STATUS_ALLOCATION_FAILED"),
            Self::TypeError => write!(f, "CURAND_STATUS_TYPE_ERROR"),
            Self::OutOfRange => write!(f, "CURAND_STATUS_OUT_OF_RANGE"),
            Self::LengthNotMultiple => write!(f, "CURAND_STATUS_LENGTH_NOT_MULTIPLE"),
            Self::DoublePrecisionRequired => {
                write!(f, "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED")
            }
            Self::LaunchFailure => write!(f, "CURAND_STATUS_LAUNCH_FAILURE"),
            Self::PreexistingFailure => write!(f, "CURAND_STATUS_PREEXISTING_FAILURE"),
            Self::InternalError => write!(f, "CURAND_STATUS_INTERNAL_ERROR"),
        }
    }
}

// ---------------------------------------------------------------------------
// cuRAND direction enum
// ---------------------------------------------------------------------------

/// Direction vector type for Sobol sequences, for forward compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CurandDirection {
    /// Standard direction vectors (Joe & Kuo).
    JoeKuo,
    /// Custom user-supplied direction vectors.
    Custom,
}

impl std::fmt::Display for CurandDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JoeKuo => write!(f, "CURAND_DIRECTION_VECTORS_32_JOEKUO6"),
            Self::Custom => write!(f, "CURAND_DIRECTION_VECTORS_CUSTOM"),
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side Philox-4x32-10 implementation
// ---------------------------------------------------------------------------

/// Philox multiplier constant 0.
const PHILOX_M4X32_0: u32 = 0xD251_1F53;
/// Philox multiplier constant 1.
const PHILOX_M4X32_1: u32 = 0xCD9E_8D57;
/// Philox Weyl sequence constant 0.
const PHILOX_W32_0: u32 = 0x9E37_79B9;
/// Philox Weyl sequence constant 1.
const PHILOX_W32_1: u32 = 0xBB67_AE85;

/// Performs one Philox-4x32 round, returning updated counter words.
#[inline]
fn philox_round(c: [u32; 4], k: [u32; 2]) -> [u32; 4] {
    let hi0 = ((c[0] as u64).wrapping_mul(PHILOX_M4X32_0 as u64) >> 32) as u32;
    let lo0 = c[0].wrapping_mul(PHILOX_M4X32_0);
    let hi1 = ((c[2] as u64).wrapping_mul(PHILOX_M4X32_1 as u64) >> 32) as u32;
    let lo1 = c[2].wrapping_mul(PHILOX_M4X32_1);

    [
        hi1 ^ c[1] ^ k[0],
        lo1,
        hi0 ^ c[3] ^ k[1],
        lo0,
    ]
}

/// Runs 10 rounds of Philox-4x32 and returns the 4-word output.
fn philox_4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = counter;
    let mut k = key;
    for _ in 0..10 {
        c = philox_round(c, k);
        k[0] = k[0].wrapping_add(PHILOX_W32_0);
        k[1] = k[1].wrapping_add(PHILOX_W32_1);
    }
    c
}

// ---------------------------------------------------------------------------
// Host-side XORWOW implementation
// ---------------------------------------------------------------------------

/// XORWOW Weyl increment constant.
const XORWOW_WEYL_INC: u32 = 362_437;

/// XORWOW generator state.
#[derive(Clone)]
struct XorwowState {
    x: [u32; 5],
    d: u32,
}

impl XorwowState {
    /// Initializes XORWOW state from seed and thread index.
    fn new(seed: u64, idx: u64) -> Self {
        let seed_lo = seed as u32;
        let mixed = (idx as u32).wrapping_mul(1_812_433_253);
        Self {
            x: [
                seed_lo ^ (idx as u32),
                seed_lo ^ mixed,
                seed_lo ^ mixed.wrapping_mul(1_812_433_253),
                seed_lo ^ mixed.wrapping_mul(2_654_435_761),
                seed_lo ^ mixed.wrapping_mul(3_266_489_917),
            ],
            d: 0,
        }
    }

    /// Advances state and returns next u32.
    fn next_u32(&mut self) -> u32 {
        let mut t = self.x[4];
        let s = self.x[0];
        self.x[4] = self.x[3];
        self.x[3] = self.x[2];
        self.x[2] = self.x[1];
        self.x[1] = s;
        t ^= t >> 2;
        t ^= t << 1;
        t ^= s ^ (s << 4);
        self.x[0] = t;
        self.d = self.d.wrapping_add(XORWOW_WEYL_INC);
        t.wrapping_add(self.d)
    }
}

// ---------------------------------------------------------------------------
// Host-side MRG32k3a implementation
// ---------------------------------------------------------------------------

/// MRG32k3a modulus for component 1.
const MRG_M1: u64 = 4_294_967_087;
/// MRG32k3a modulus for component 2.
const MRG_M2: u64 = 4_294_944_443;
/// MRG32k3a multiplier a12.
const MRG_A12: u64 = 1_403_580;
/// MRG32k3a negative multiplier a13.
const MRG_A13N: u64 = 810_728;
/// MRG32k3a multiplier a21.
const MRG_A21: u64 = 527_612;
/// MRG32k3a negative multiplier a23.
const MRG_A23N: u64 = 1_370_589;

/// MRG32k3a generator state.
#[derive(Clone)]
struct Mrg32k3aState {
    s1: [u64; 3],
    s2: [u64; 3],
}

impl Mrg32k3aState {
    /// Initializes MRG32k3a state from seed and thread index.
    fn new(seed: u64, idx: u64) -> Self {
        let base = seed.wrapping_add(idx.wrapping_mul(1_812_433_253));
        let s10 = (base % (MRG_M1 - 1)) + 1;
        let s11 = (base.wrapping_mul(2_654_435_761) % (MRG_M1 - 1)) + 1;
        let s12 = (base.wrapping_mul(3_266_489_917) % (MRG_M1 - 1)) + 1;
        let s20 = (base.wrapping_mul(668_265_263) % (MRG_M2 - 1)) + 1;
        let s21 = (base.wrapping_mul(1_103_515_245) % (MRG_M2 - 1)) + 1;
        let s22 = (base.wrapping_mul(214_013) % (MRG_M2 - 1)) + 1;
        Self {
            s1: [s10, s11, s12],
            s2: [s20, s21, s22],
        }
    }

    /// Advances state and returns next value in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // Component 1: p1 = a12*s11 - a13n*s10  mod m1
        let p1 = (MRG_A12.wrapping_mul(self.s1[1]) + MRG_M1.wrapping_mul(2))
            .wrapping_sub(MRG_A13N.wrapping_mul(self.s1[0]))
            % MRG_M1;
        self.s1[0] = self.s1[1];
        self.s1[1] = self.s1[2];
        self.s1[2] = p1;

        // Component 2: p2 = a21*s22 - a23n*s20  mod m2
        let p2 = (MRG_A21.wrapping_mul(self.s2[2]) + MRG_M2.wrapping_mul(2))
            .wrapping_sub(MRG_A23N.wrapping_mul(self.s2[0]))
            % MRG_M2;
        self.s2[0] = self.s2[1];
        self.s2[1] = self.s2[2];
        self.s2[2] = p2;

        // Combine
        let result = if p1 > p2 {
            (p1 - p2) as f64 / MRG_M1 as f64
        } else {
            (p1 + MRG_M1 - p2) as f64 / MRG_M1 as f64
        };
        result
    }
}

// ---------------------------------------------------------------------------
// Host-side Sobol implementation (simple Gray-code)
// ---------------------------------------------------------------------------

/// Generates Sobol quasi-random sequence values using Gray code.
fn sobol_generate(n: usize, offset: u64) -> Vec<f32> {
    let scale = 1.0_f32 / (1u64 << 32) as f32;
    let mut result = Vec::with_capacity(n);
    let mut value: u32 = 0;
    let start = offset as u32;

    // Initialize value for `start` using standard bit-reversal directions
    if start > 0 {
        let mut idx = start;
        let mut bit = 0u32;
        while idx > 0 {
            if idx & 1 != 0 {
                value ^= 1u32 << (31 - bit);
            }
            idx >>= 1;
            bit += 1;
        }
    }

    for i in 0..n {
        let idx = start.wrapping_add(i as u32);
        if idx == 0 {
            result.push(0.0);
        } else {
            // Gray code: find rightmost zero bit of (idx - 1)
            let c = (idx - 1).trailing_ones();
            let direction = 1u32 << (31u32.saturating_sub(c));
            value ^= direction;
            result.push(value as f32 * scale);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// CurandGenerator
// ---------------------------------------------------------------------------

/// Host-side random number generator matching cuRAND's `curandGenerator_t`.
///
/// Provides the same API semantics as NVIDIA's cuRAND library, generating
/// random numbers using pure Rust implementations of Philox, XORWOW,
/// MRG32k3a, and Sobol algorithms.
///
/// # Example
///
/// ```rust
/// use oxicuda_rand::host_api::{CurandGenerator, CurandRngType, CurandOrdering};
///
/// let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10).unwrap();
/// rng.set_seed(12345);
/// rng.set_ordering(CurandOrdering::Seeded);
///
/// let uniform = rng.generate_uniform_f32(100).unwrap();
/// let normal = rng.generate_normal_f32(100, 0.0, 1.0).unwrap();
/// ```
pub struct CurandGenerator {
    rng_type: CurandRngType,
    seed: u64,
    offset: u64,
    stream_id: u64,
    ordering: CurandOrdering,
}

impl CurandGenerator {
    /// Creates a new generator of the specified type.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `RandResult` for forward
    /// compatibility with GPU-backed generators that may fail on
    /// initialization.
    pub fn new(rng_type: CurandRngType) -> RandResult<Self> {
        Ok(Self {
            rng_type,
            seed: 0,
            offset: 0,
            stream_id: 0,
            ordering: CurandOrdering::Default,
        })
    }

    /// Returns the generator type.
    pub fn rng_type(&self) -> CurandRngType {
        self.rng_type
    }

    /// Sets the seed for pseudo-random generators.
    ///
    /// Has no effect on quasi-random generators.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Returns the current seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Sets the offset for skipping ahead in the sequence.
    pub fn set_offset(&mut self, offset: u64) {
        self.offset = offset;
    }

    /// Returns the current offset.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Associates this generator with a CUDA stream.
    ///
    /// On the host-side path this is stored for API compatibility but does
    /// not affect generation.
    pub fn set_stream(&mut self, stream_id: u64) {
        self.stream_id = stream_id;
    }

    /// Returns the associated stream identifier.
    pub fn stream_id(&self) -> u64 {
        self.stream_id
    }

    /// Sets the output ordering mode.
    pub fn set_ordering(&mut self, ordering: CurandOrdering) {
        self.ordering = ordering;
    }

    /// Returns the current ordering mode.
    pub fn ordering(&self) -> CurandOrdering {
        self.ordering
    }

    // -----------------------------------------------------------------------
    // Uniform generation
    // -----------------------------------------------------------------------

    /// Generates `n` uniformly distributed f32 values in \[0, 1).
    ///
    /// # Errors
    ///
    /// Returns `RandError::UnsupportedDistribution` for quasi-random types
    /// when called with distributions other than uniform (this method is fine).
    pub fn generate_uniform_f32(&mut self, n: usize) -> RandResult<Vec<f32>> {
        if n == 0 {
            return Ok(Vec::new());
        }

        let result = match self.rng_type {
            CurandRngType::PseudoDefault | CurandRngType::PseudoXorwow => {
                self.generate_xorwow_uniform_f32(n)
            }
            CurandRngType::PseudoPhilox4_32_10 => self.generate_philox_uniform_f32(n),
            CurandRngType::PseudoMrg32k3a => self.generate_mrg_uniform_f32(n),
            CurandRngType::QuasiDefault | CurandRngType::QuasiSobol32 => {
                let values = sobol_generate(n, self.offset);
                self.offset += n as u64;
                values
            }
            CurandRngType::QuasiScrambledSobol32 => {
                // Use sobol with seed-based scrambling
                let mut values = sobol_generate(n, self.offset);
                let scramble = (self.seed & 0xFFFF_FFFF) as u32;
                for v in &mut values {
                    let bits = v.to_bits() ^ scramble;
                    *v = f32::from_bits(bits & 0x7FFF_FFFF) / (1u64 << 31) as f32;
                    // Ensure [0, 1)
                    if *v >= 1.0 {
                        *v = 1.0 - f32::EPSILON;
                    }
                }
                self.offset += n as u64;
                values
            }
        };

        Ok(result)
    }

    /// Generates `n` uniformly distributed f64 values in \[0, 1).
    pub fn generate_uniform_f64(&mut self, n: usize) -> RandResult<Vec<f64>> {
        if n == 0 {
            return Ok(Vec::new());
        }

        let result = match self.rng_type {
            CurandRngType::PseudoDefault | CurandRngType::PseudoXorwow => {
                self.generate_xorwow_uniform_f64(n)
            }
            CurandRngType::PseudoPhilox4_32_10 => self.generate_philox_uniform_f64(n),
            CurandRngType::PseudoMrg32k3a => self.generate_mrg_uniform_f64(n),
            CurandRngType::QuasiDefault
            | CurandRngType::QuasiSobol32
            | CurandRngType::QuasiScrambledSobol32 => {
                let f32_vals = self.generate_uniform_f32(n)?;
                return Ok(f32_vals.into_iter().map(|v| v as f64).collect());
            }
        };

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Normal generation
    // -----------------------------------------------------------------------

    /// Generates `n` normally distributed f32 values with given mean and stddev.
    ///
    /// Uses the Box-Muller transform on uniform values.
    ///
    /// # Errors
    ///
    /// Returns `RandError::UnsupportedDistribution` for quasi-random types.
    pub fn generate_normal_f32(
        &mut self,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> RandResult<Vec<f32>> {
        self.require_pseudo("normal")?;
        if n == 0 {
            return Ok(Vec::new());
        }

        // Generate pairs of uniforms for Box-Muller
        let n_pairs = (n + 1) / 2;
        let uniforms = self.generate_uniform_f32(n_pairs * 2)?;
        let mut result = Vec::with_capacity(n);

        for i in 0..n_pairs {
            let u1 = uniforms[2 * i].max(f32::EPSILON);
            let u2 = uniforms[2 * i + 1];
            let radius = (-2.0_f32 * u1.ln()).sqrt();
            let angle = std::f32::consts::TAU * u2;
            let z0 = radius * angle.cos();
            let z1 = radius * angle.sin();
            result.push(mean + stddev * z0);
            if result.len() < n {
                result.push(mean + stddev * z1);
            }
        }

        Ok(result)
    }

    /// Generates `n` normally distributed f64 values with given mean and stddev.
    ///
    /// # Errors
    ///
    /// Returns `RandError::UnsupportedDistribution` for quasi-random types.
    pub fn generate_normal_f64(
        &mut self,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> RandResult<Vec<f64>> {
        self.require_pseudo("normal")?;
        if n == 0 {
            return Ok(Vec::new());
        }

        let n_pairs = (n + 1) / 2;
        let uniforms = self.generate_uniform_f64(n_pairs * 2)?;
        let mut result = Vec::with_capacity(n);

        for i in 0..n_pairs {
            let u1 = uniforms[2 * i].max(f64::EPSILON);
            let u2 = uniforms[2 * i + 1];
            let radius = (-2.0_f64 * u1.ln()).sqrt();
            let angle = std::f64::consts::TAU * u2;
            let z0 = radius * angle.cos();
            let z1 = radius * angle.sin();
            result.push(mean + stddev * z0);
            if result.len() < n {
                result.push(mean + stddev * z1);
            }
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Log-normal generation
    // -----------------------------------------------------------------------

    /// Generates `n` log-normally distributed f32 values.
    ///
    /// A log-normal variate is `exp(Normal(mean, stddev))`.
    ///
    /// # Errors
    ///
    /// Returns `RandError::UnsupportedDistribution` for quasi-random types.
    pub fn generate_log_normal_f32(
        &mut self,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> RandResult<Vec<f32>> {
        let normals = self.generate_normal_f32(n, mean, stddev)?;
        Ok(normals.into_iter().map(|x| x.exp()).collect())
    }

    /// Generates `n` log-normally distributed f64 values.
    ///
    /// # Errors
    ///
    /// Returns `RandError::UnsupportedDistribution` for quasi-random types.
    pub fn generate_log_normal_f64(
        &mut self,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> RandResult<Vec<f64>> {
        let normals = self.generate_normal_f64(n, mean, stddev)?;
        Ok(normals.into_iter().map(|x| x.exp()).collect())
    }

    // -----------------------------------------------------------------------
    // Poisson generation
    // -----------------------------------------------------------------------

    /// Generates `n` Poisson-distributed u32 values with parameter `lambda`.
    ///
    /// For lambda < 30, uses Knuth's algorithm. For lambda >= 30, uses
    /// normal approximation `round(Normal(lambda, sqrt(lambda)))`.
    ///
    /// # Errors
    ///
    /// Returns `RandError::UnsupportedDistribution` for quasi-random types.
    /// Returns `RandError::InvalidSize` if lambda is not positive.
    pub fn generate_poisson(&mut self, n: usize, lambda: f64) -> RandResult<Vec<u32>> {
        self.require_pseudo("poisson")?;
        if lambda <= 0.0 {
            return Err(RandError::InvalidSize(
                "Poisson lambda must be positive".to_string(),
            ));
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        if lambda < 30.0 {
            self.generate_poisson_knuth(n, lambda)
        } else {
            self.generate_poisson_normal_approx(n, lambda)
        }
    }

    // -----------------------------------------------------------------------
    // Raw u32 generation
    // -----------------------------------------------------------------------

    /// Generates `n` raw u32 random values.
    ///
    /// # Errors
    ///
    /// Returns `RandError::UnsupportedDistribution` for quasi-random types.
    pub fn generate_u32(&mut self, n: usize) -> RandResult<Vec<u32>> {
        self.require_pseudo("u32")?;
        if n == 0 {
            return Ok(Vec::new());
        }

        let result = match self.rng_type {
            CurandRngType::PseudoDefault | CurandRngType::PseudoXorwow => {
                let mut state = XorwowState::new(self.seed, self.offset);
                let values: Vec<u32> = (0..n).map(|_| state.next_u32()).collect();
                self.offset += n as u64;
                values
            }
            CurandRngType::PseudoPhilox4_32_10 => {
                let key = [self.seed as u32, (self.seed >> 32) as u32];
                let mut values = Vec::with_capacity(n);
                let base_offset = self.offset;
                let mut idx = 0u64;
                while values.len() < n {
                    let counter_val = base_offset.wrapping_add(idx);
                    let counter = [
                        counter_val as u32,
                        (counter_val >> 32) as u32,
                        0,
                        0,
                    ];
                    let output = philox_4x32_10(counter, key);
                    for &w in &output {
                        if values.len() < n {
                            values.push(w);
                        }
                    }
                    idx += 1;
                }
                self.offset += ((n + 3) / 4) as u64;
                values
            }
            CurandRngType::PseudoMrg32k3a => {
                let mut state = Mrg32k3aState::new(self.seed, self.offset);
                let values: Vec<u32> = (0..n)
                    .map(|_| {
                        let f = state.next_f64();
                        (f * u32::MAX as f64) as u32
                    })
                    .collect();
                self.offset += n as u64;
                values
            }
            _ => {
                return Err(RandError::UnsupportedDistribution(
                    "u32 generation not supported for quasi-random types".to_string(),
                ));
            }
        };

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Internal: engine-specific uniform generation
    // -----------------------------------------------------------------------

    /// Generates uniform f32 values using the Philox engine.
    fn generate_philox_uniform_f32(&mut self, n: usize) -> Vec<f32> {
        let key = [self.seed as u32, (self.seed >> 32) as u32];
        let scale = 1.0_f32 / (1u64 << 32) as f32;
        let mut result = Vec::with_capacity(n);
        let base = self.offset;
        let mut counter_idx = 0u64;

        while result.len() < n {
            let counter_val = base.wrapping_add(counter_idx);
            let counter = [counter_val as u32, (counter_val >> 32) as u32, 0, 0];
            let output = philox_4x32_10(counter, key);
            for &w in &output {
                if result.len() < n {
                    result.push(w as f32 * scale);
                }
            }
            counter_idx += 1;
        }

        self.offset += ((n + 3) / 4) as u64;
        result
    }

    /// Generates uniform f64 values using the Philox engine.
    fn generate_philox_uniform_f64(&mut self, n: usize) -> Vec<f64> {
        let key = [self.seed as u32, (self.seed >> 32) as u32];
        let scale_lo = 1.0_f64 / (1u64 << 32) as f64;
        let scale_hi = scale_lo / (1u64 << 32) as f64;
        let mut result = Vec::with_capacity(n);
        let base = self.offset;
        let mut counter_idx = 0u64;

        while result.len() < n {
            let counter_val = base.wrapping_add(counter_idx);
            let counter = [counter_val as u32, (counter_val >> 32) as u32, 0, 0];
            let output = philox_4x32_10(counter, key);
            // Use pairs of u32 for f64
            let v0 = output[0] as f64 * scale_lo + output[1] as f64 * scale_hi;
            if result.len() < n {
                result.push(v0);
            }
            let v1 = output[2] as f64 * scale_lo + output[3] as f64 * scale_hi;
            if result.len() < n {
                result.push(v1);
            }
            counter_idx += 1;
        }

        self.offset += ((n + 1) / 2) as u64;
        result
    }

    /// Generates uniform f32 values using the XORWOW engine.
    fn generate_xorwow_uniform_f32(&mut self, n: usize) -> Vec<f32> {
        let scale = 1.0_f32 / (1u64 << 32) as f32;
        let mut state = XorwowState::new(self.seed, self.offset);
        let result: Vec<f32> = (0..n).map(|_| state.next_u32() as f32 * scale).collect();
        self.offset += n as u64;
        result
    }

    /// Generates uniform f64 values using the XORWOW engine.
    fn generate_xorwow_uniform_f64(&mut self, n: usize) -> Vec<f64> {
        let scale = 1.0_f64 / (1u64 << 32) as f64;
        let mut state = XorwowState::new(self.seed, self.offset);
        let result: Vec<f64> = (0..n).map(|_| state.next_u32() as f64 * scale).collect();
        self.offset += n as u64;
        result
    }

    /// Generates uniform f32 values using the MRG32k3a engine.
    fn generate_mrg_uniform_f32(&mut self, n: usize) -> Vec<f32> {
        let mut state = Mrg32k3aState::new(self.seed, self.offset);
        let result: Vec<f32> = (0..n).map(|_| state.next_f64() as f32).collect();
        self.offset += n as u64;
        result
    }

    /// Generates uniform f64 values using the MRG32k3a engine.
    fn generate_mrg_uniform_f64(&mut self, n: usize) -> Vec<f64> {
        let mut state = Mrg32k3aState::new(self.seed, self.offset);
        let result: Vec<f64> = (0..n).map(|_| state.next_f64()).collect();
        self.offset += n as u64;
        result
    }

    // -----------------------------------------------------------------------
    // Internal: Poisson helpers
    // -----------------------------------------------------------------------

    /// Poisson generation using Knuth's algorithm for small lambda.
    fn generate_poisson_knuth(&mut self, n: usize, lambda: f64) -> RandResult<Vec<u32>> {
        let exp_neg_lambda = (-lambda).exp();
        let uniforms = self.generate_uniform_f64(n * 40)?; // generous pool
        let mut result = Vec::with_capacity(n);
        let mut uni_idx = 0;

        for _ in 0..n {
            let mut k = 0u32;
            let mut p = 1.0_f64;
            loop {
                if uni_idx >= uniforms.len() {
                    // Extend with fallback deterministic values
                    p *= 0.5;
                } else {
                    p *= uniforms[uni_idx];
                    uni_idx += 1;
                }
                if p <= exp_neg_lambda {
                    break;
                }
                k = k.saturating_add(1);
                // Safety bound to avoid infinite loops with edge-case floats
                if k >= 10_000 {
                    break;
                }
            }
            result.push(k);
        }

        Ok(result)
    }

    /// Poisson generation using normal approximation for large lambda.
    fn generate_poisson_normal_approx(
        &mut self,
        n: usize,
        lambda: f64,
    ) -> RandResult<Vec<u32>> {
        let mean = lambda;
        let stddev = lambda.sqrt();
        let normals = self.generate_normal_f64(n, mean, stddev)?;
        Ok(normals
            .into_iter()
            .map(|x| {
                if x < 0.0 {
                    0u32
                } else {
                    let rounded = x.round();
                    if rounded > u32::MAX as f64 {
                        u32::MAX
                    } else {
                        rounded as u32
                    }
                }
            })
            .collect())
    }

    // -----------------------------------------------------------------------
    // Internal: validation
    // -----------------------------------------------------------------------

    /// Ensures the generator is a pseudo-random type; returns an error for quasi types.
    fn require_pseudo(&self, distribution: &str) -> RandResult<()> {
        if self.rng_type.is_quasi() {
            return Err(RandError::UnsupportedDistribution(format!(
                "{distribution} distribution is not supported for quasi-random generators"
            )));
        }
        Ok(())
    }
}

impl std::fmt::Debug for CurandGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CurandGenerator")
            .field("rng_type", &self.rng_type)
            .field("seed", &self.seed)
            .field("offset", &self.offset)
            .field("stream_id", &self.stream_id)
            .field("ordering", &self.ordering)
            .finish()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn rng_type_is_pseudo() {
        assert!(CurandRngType::PseudoDefault.is_pseudo());
        assert!(CurandRngType::PseudoXorwow.is_pseudo());
        assert!(CurandRngType::PseudoMrg32k3a.is_pseudo());
        assert!(CurandRngType::PseudoPhilox4_32_10.is_pseudo());
        assert!(!CurandRngType::QuasiDefault.is_pseudo());
        assert!(!CurandRngType::QuasiSobol32.is_pseudo());
        assert!(!CurandRngType::QuasiScrambledSobol32.is_pseudo());
    }

    #[test]
    fn rng_type_is_quasi() {
        assert!(!CurandRngType::PseudoDefault.is_quasi());
        assert!(CurandRngType::QuasiDefault.is_quasi());
        assert!(CurandRngType::QuasiSobol32.is_quasi());
        assert!(CurandRngType::QuasiScrambledSobol32.is_quasi());
    }

    #[test]
    fn rng_type_display() {
        assert_eq!(
            format!("{}", CurandRngType::PseudoPhilox4_32_10),
            "CURAND_RNG_PSEUDO_PHILOX4_32_10"
        );
        assert_eq!(
            format!("{}", CurandRngType::QuasiSobol32),
            "CURAND_RNG_QUASI_SOBOL32"
        );
    }

    #[test]
    fn ordering_display() {
        assert_eq!(
            format!("{}", CurandOrdering::Default),
            "CURAND_ORDERING_PSEUDO_DEFAULT"
        );
        assert_eq!(
            format!("{}", CurandOrdering::Seeded),
            "CURAND_ORDERING_PSEUDO_SEEDED"
        );
    }

    #[test]
    fn status_is_success() {
        assert!(CurandStatus::Success.is_success());
        assert!(!CurandStatus::NotInitialized.is_success());
        assert!(!CurandStatus::InternalError.is_success());
    }

    #[test]
    fn direction_display() {
        assert_eq!(
            format!("{}", CurandDirection::JoeKuo),
            "CURAND_DIRECTION_VECTORS_32_JOEKUO6"
        );
    }

    // -----------------------------------------------------------------------
    // Generator construction and configuration
    // -----------------------------------------------------------------------

    #[test]
    fn generator_new_and_accessors() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        assert_eq!(rng.rng_type(), CurandRngType::PseudoPhilox4_32_10);
        assert_eq!(rng.seed(), 0);
        assert_eq!(rng.offset(), 0);
        assert_eq!(rng.stream_id(), 0);
        assert_eq!(rng.ordering(), CurandOrdering::Default);

        rng.set_seed(42);
        rng.set_offset(100);
        rng.set_stream(7);
        rng.set_ordering(CurandOrdering::Seeded);

        assert_eq!(rng.seed(), 42);
        assert_eq!(rng.offset(), 100);
        assert_eq!(rng.stream_id(), 7);
        assert_eq!(rng.ordering(), CurandOrdering::Seeded);
    }

    #[test]
    fn generator_debug_display() {
        let rng = CurandGenerator::new(CurandRngType::PseudoDefault)
            .expect("should create generator");
        let debug = format!("{rng:?}");
        assert!(debug.contains("CurandGenerator"));
        assert!(debug.contains("PseudoDefault"));
    }

    // -----------------------------------------------------------------------
    // Philox uniform generation
    // -----------------------------------------------------------------------

    #[test]
    fn philox_uniform_f32_range() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        rng.set_seed(42);
        let values = rng.generate_uniform_f32(10_000).expect("should generate");
        assert_eq!(values.len(), 10_000);
        for &v in &values {
            assert!(v >= 0.0, "value {v} should be >= 0.0");
            assert!(v < 1.0, "value {v} should be < 1.0");
        }
    }

    #[test]
    fn philox_uniform_f64_range() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        rng.set_seed(42);
        let values = rng.generate_uniform_f64(5_000).expect("should generate");
        assert_eq!(values.len(), 5_000);
        for &v in &values {
            assert!(v >= 0.0, "value {v} should be >= 0.0");
            assert!(v < 1.0, "value {v} should be < 1.0");
        }
    }

    // -----------------------------------------------------------------------
    // XORWOW uniform generation
    // -----------------------------------------------------------------------

    #[test]
    fn xorwow_uniform_f32_range() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoXorwow)
            .expect("should create generator");
        rng.set_seed(123);
        let values = rng.generate_uniform_f32(5_000).expect("should generate");
        assert_eq!(values.len(), 5_000);
        for &v in &values {
            assert!(v >= 0.0 && v < 1.0, "value {v} out of range");
        }
    }

    // -----------------------------------------------------------------------
    // MRG32k3a uniform generation
    // -----------------------------------------------------------------------

    #[test]
    fn mrg_uniform_f64_range() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoMrg32k3a)
            .expect("should create generator");
        rng.set_seed(999);
        let values = rng.generate_uniform_f64(5_000).expect("should generate");
        assert_eq!(values.len(), 5_000);
        for &v in &values {
            assert!(v >= 0.0 && v < 1.0, "value {v} out of range");
        }
    }

    // -----------------------------------------------------------------------
    // Normal generation
    // -----------------------------------------------------------------------

    #[test]
    fn normal_f32_statistics() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        rng.set_seed(42);
        let n = 50_000;
        let values = rng
            .generate_normal_f32(n, 0.0, 1.0)
            .expect("should generate");
        assert_eq!(values.len(), n);

        let mean: f32 = values.iter().sum::<f32>() / n as f32;
        let variance: f32 =
            values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;

        // Loose bounds for statistical tests
        assert!(
            mean.abs() < 0.1,
            "mean {mean} should be close to 0.0"
        );
        assert!(
            (variance - 1.0).abs() < 0.2,
            "variance {variance} should be close to 1.0"
        );
    }

    #[test]
    fn normal_f64_with_custom_mean_stddev() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        rng.set_seed(7);
        let values = rng
            .generate_normal_f64(10_000, 5.0, 2.0)
            .expect("should generate");
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        assert!(
            (mean - 5.0).abs() < 0.2,
            "mean {mean} should be close to 5.0"
        );
    }

    // -----------------------------------------------------------------------
    // Log-normal generation
    // -----------------------------------------------------------------------

    #[test]
    fn log_normal_f32_positive() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        rng.set_seed(42);
        let values = rng
            .generate_log_normal_f32(5_000, 0.0, 1.0)
            .expect("should generate");
        for &v in &values {
            assert!(v > 0.0, "log-normal value {v} must be positive");
        }
    }

    // -----------------------------------------------------------------------
    // Poisson generation
    // -----------------------------------------------------------------------

    #[test]
    fn poisson_small_lambda() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        rng.set_seed(42);
        let values = rng.generate_poisson(10_000, 5.0).expect("should generate");
        let mean: f64 = values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64;
        assert!(
            (mean - 5.0).abs() < 0.5,
            "Poisson mean {mean} should be close to lambda=5.0"
        );
    }

    #[test]
    fn poisson_large_lambda() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        rng.set_seed(42);
        let values = rng
            .generate_poisson(10_000, 100.0)
            .expect("should generate");
        let mean: f64 = values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64;
        assert!(
            (mean - 100.0).abs() < 5.0,
            "Poisson mean {mean} should be close to lambda=100.0"
        );
    }

    #[test]
    fn poisson_invalid_lambda() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        assert!(rng.generate_poisson(100, 0.0).is_err());
        assert!(rng.generate_poisson(100, -1.0).is_err());
    }

    // -----------------------------------------------------------------------
    // Raw u32 generation
    // -----------------------------------------------------------------------

    #[test]
    fn u32_generation() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        rng.set_seed(42);
        let values = rng.generate_u32(1_000).expect("should generate");
        assert_eq!(values.len(), 1_000);
        // Verify not all zeros (extremely unlikely with good RNG)
        assert!(values.iter().any(|&v| v != 0));
    }

    // -----------------------------------------------------------------------
    // Quasi-random generation
    // -----------------------------------------------------------------------

    #[test]
    fn sobol_uniform_range() {
        let mut rng = CurandGenerator::new(CurandRngType::QuasiSobol32)
            .expect("should create generator");
        let values = rng.generate_uniform_f32(1_000).expect("should generate");
        assert_eq!(values.len(), 1_000);
        for &v in &values {
            assert!(v >= 0.0 && v < 1.0, "Sobol value {v} out of range");
        }
    }

    #[test]
    fn quasi_normal_rejected() {
        let mut rng = CurandGenerator::new(CurandRngType::QuasiSobol32)
            .expect("should create generator");
        assert!(rng.generate_normal_f32(100, 0.0, 1.0).is_err());
    }

    // -----------------------------------------------------------------------
    // Empty generation
    // -----------------------------------------------------------------------

    #[test]
    fn empty_generation() {
        let mut rng = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        assert_eq!(
            rng.generate_uniform_f32(0).expect("should succeed").len(),
            0
        );
        assert_eq!(
            rng.generate_normal_f32(0, 0.0, 1.0)
                .expect("should succeed")
                .len(),
            0
        );
        assert_eq!(
            rng.generate_poisson(0, 5.0).expect("should succeed").len(),
            0
        );
        assert_eq!(rng.generate_u32(0).expect("should succeed").len(), 0);
    }

    // -----------------------------------------------------------------------
    // Reproducibility
    // -----------------------------------------------------------------------

    #[test]
    fn philox_reproducible_with_same_seed() {
        let mut rng1 = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        rng1.set_seed(42);
        let values1 = rng1.generate_uniform_f32(100).expect("should generate");

        let mut rng2 = CurandGenerator::new(CurandRngType::PseudoPhilox4_32_10)
            .expect("should create generator");
        rng2.set_seed(42);
        let values2 = rng2.generate_uniform_f32(100).expect("should generate");

        assert_eq!(values1, values2);
    }
}
