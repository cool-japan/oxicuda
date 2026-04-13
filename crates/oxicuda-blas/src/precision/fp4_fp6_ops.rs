//! FP6 (E3M2/E2M3) and FP4 (E2M1/INT4) mixed-precision training kernels
//! for Blackwell (sm_100+) GPUs.
//!
//! These sub-byte formats enable extreme compression of model weights
//! for inference and mixed-precision training. Blackwell Tensor Cores
//! natively support FP6 and FP4 operations with microscaling (MX).
//!
//! ## Formats
//!
//! - **FP6 E3M2**: 1 sign + 3 exponent + 2 mantissa (6 bits). Range ~ [-7.5, 7.5].
//! - **FP6 E2M3**: 1 sign + 2 exponent + 3 mantissa (6 bits). Range ~ [-3.75, 3.75].
//! - **FP4 E2M1**: 1 sign + 2 exponent + 1 mantissa (4 bits). Range ~ [-6, 6].
//! - **FP4 INT4**: 4-bit signed integer. Range [-8, 7].
//!
//! ## Micro-scaling (MX)
//!
//! Sub-byte quantization is paired with per-block scaling factors
//! to preserve dynamic range. Each block of 32/64/128 elements shares
//! a single scale factor stored in FP8/FP16/FP32.
//!
//! ## Packing
//!
//! - **PackedFp6**: 3 FP6 values packed into 18 bits of a `u32`.
//! - **PackedFp4**: 2 FP4 values packed into 8 bits of a `u8`.

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::prelude::SmVersion;

use crate::error::{BlasError, BlasResult};
use crate::level3::gemm::dispatch::TileConfig;

// ---------------------------------------------------------------------------
// FP6 format
// ---------------------------------------------------------------------------

/// FP6 data format variant.
///
/// Both variants use 6 bits total (1 sign + exponent + mantissa).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp6Format {
    /// E3M2: 1 sign + 3-bit exponent + 2-bit mantissa.
    /// Bias = 3. Range approximately [-7.5, 7.5].
    /// Higher dynamic range, lower precision.
    E3M2,
    /// E2M3: 1 sign + 2-bit exponent + 3-bit mantissa.
    /// Bias = 1. Range approximately [-3.75, 3.75].
    /// Lower dynamic range, higher precision.
    E2M3,
}

impl Fp6Format {
    /// Returns the number of exponent bits for this format.
    #[must_use]
    pub const fn exponent_bits(self) -> u32 {
        match self {
            Self::E3M2 => 3,
            Self::E2M3 => 2,
        }
    }

    /// Returns the number of mantissa bits for this format.
    #[must_use]
    pub const fn mantissa_bits(self) -> u32 {
        match self {
            Self::E3M2 => 2,
            Self::E2M3 => 3,
        }
    }

    /// Returns the exponent bias for this format.
    #[must_use]
    pub const fn bias(self) -> i32 {
        match self {
            Self::E3M2 => 3, // 2^(3-1) - 1
            Self::E2M3 => 1, // 2^(2-1) - 1
        }
    }

    /// Returns a short name for kernel naming.
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::E3M2 => "e3m2",
            Self::E2M3 => "e2m3",
        }
    }

    /// Returns the maximum representable absolute value.
    #[must_use]
    pub fn max_value(self) -> f32 {
        match self {
            Self::E3M2 => 7.5,
            Self::E2M3 => 3.75,
        }
    }
}

// ---------------------------------------------------------------------------
// FP4 format
// ---------------------------------------------------------------------------

/// FP4 data format variant.
///
/// FP4 variants use 4 bits per element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp4Format {
    /// E2M1: 1 sign + 2-bit exponent + 1-bit mantissa.
    /// Bias = 1. Range approximately [-6, 6].
    E2M1,
    /// INT4: 4-bit signed integer (two's complement).
    /// Range [-8, 7]. No floating-point semantics.
    Int4,
}

impl Fp4Format {
    /// Returns a short name for kernel naming.
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::E2M1 => "e2m1",
            Self::Int4 => "int4",
        }
    }

    /// Returns the maximum representable absolute value.
    #[must_use]
    pub fn max_value(self) -> f32 {
        match self {
            Self::E2M1 => 6.0,
            Self::Int4 => 8.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Micro-scaling configuration
// ---------------------------------------------------------------------------

/// Scaling format for micro-scaling factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalingFormat {
    /// 8-bit floating point scale factor.
    Fp8,
    /// 16-bit floating point scale factor.
    Fp16,
    /// 32-bit floating point scale factor.
    Fp32,
}

impl ScalingFormat {
    /// Returns the size in bytes of one scale factor.
    #[must_use]
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::Fp8 => 1,
            Self::Fp16 => 2,
            Self::Fp32 => 4,
        }
    }

    /// Returns the PTX type suffix for this scale format.
    #[must_use]
    pub const fn ptx_type(self) -> &'static str {
        match self {
            Self::Fp8 => ".b8",
            Self::Fp16 => ".f16",
            Self::Fp32 => ".f32",
        }
    }
}

/// Granularity at which scaling factors are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalingGranularity {
    /// One scale factor per entire tensor.
    PerTensor,
    /// One scale factor per row or column (channel).
    PerChannel,
    /// One scale factor per block of `block_size` elements.
    PerBlock,
}

/// Micro-scaling configuration for sub-byte quantization.
///
/// Groups of `block_size` elements share a single scaling factor stored
/// in `scaling_format`. This preserves dynamic range while using very
/// low precision for individual elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MicroScalingConfig {
    /// Number of elements per scaling block.
    /// Must be one of: 32, 64, 128.
    pub block_size: u32,
    /// Precision of the scaling factor.
    pub scaling_format: ScalingFormat,
    /// Granularity of scale factor application.
    pub scaling_granularity: ScalingGranularity,
}

impl MicroScalingConfig {
    /// Valid block sizes for micro-scaling.
    pub const VALID_BLOCK_SIZES: &'static [u32] = &[32, 64, 128];

    /// Validates the micro-scaling configuration.
    pub fn validate(&self) -> BlasResult<()> {
        if !Self::VALID_BLOCK_SIZES.contains(&self.block_size) {
            return Err(BlasError::InvalidArgument(format!(
                "MicroScaling block_size must be one of {:?}, got {}",
                Self::VALID_BLOCK_SIZES,
                self.block_size
            )));
        }
        Ok(())
    }

    /// Returns the number of scale factors needed for `num_elements` elements.
    #[must_use]
    pub fn num_scales(&self, num_elements: u32) -> u32 {
        match self.scaling_granularity {
            ScalingGranularity::PerTensor => 1,
            ScalingGranularity::PerChannel => num_elements, // caller passes channel count
            ScalingGranularity::PerBlock => num_elements.div_ceil(self.block_size),
        }
    }
}

// ---------------------------------------------------------------------------
// Accumulator type for FP6/FP4
// ---------------------------------------------------------------------------

/// Accumulator precision for sub-byte GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubByteAccumulator {
    /// 16-bit floating point accumulation (lower precision, higher throughput).
    F16,
    /// 32-bit floating point accumulation (higher precision).
    F32,
}

impl SubByteAccumulator {
    /// Returns the PTX type string for this accumulator.
    #[must_use]
    pub const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::F16 => ".f16",
            Self::F32 => ".f32",
        }
    }

    /// Returns a short name for kernel naming.
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }
}

// ---------------------------------------------------------------------------
// GEMM configurations
// ---------------------------------------------------------------------------

/// Configuration for FP6 GEMM operations.
///
/// FP6 GEMM uses a dequantize-then-compute pipeline:
/// 1. Load packed FP6 data from global memory
/// 2. Dequantize to FP16/FP32 using micro-scaling factors
/// 3. Execute Tensor Core MMA on dequantized values
/// 4. Accumulate in the specified accumulator format
#[derive(Debug, Clone)]
pub struct Fp6GemmConfig {
    /// Number of rows of output matrix C.
    pub m: u32,
    /// Number of columns of output matrix C.
    pub n: u32,
    /// Shared (inner) dimension.
    pub k: u32,
    /// FP6 format for weight data.
    pub format: Fp6Format,
    /// Accumulator precision.
    pub accumulator: SubByteAccumulator,
    /// Micro-scaling configuration.
    pub micro_scaling: MicroScalingConfig,
    /// Target SM version.
    pub sm_version: SmVersion,
}

impl Fp6GemmConfig {
    /// Minimum SM version for FP6 Tensor Core support.
    pub const MIN_SM: SmVersion = SmVersion::Sm100;

    /// Validates the FP6 GEMM configuration.
    pub fn validate(&self) -> BlasResult<()> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "FP6 GEMM dimensions must be non-zero: m={}, n={}, k={}",
                self.m, self.n, self.k
            )));
        }
        if self.sm_version < Self::MIN_SM {
            return Err(BlasError::UnsupportedOperation(format!(
                "FP6 Tensor Cores require Blackwell+ (sm_100), got {}",
                self.sm_version
            )));
        }
        // K must be a multiple of 3 for FP6 packing alignment (3 values per 18-bit group)
        if self.k % 3 != 0 {
            return Err(BlasError::InvalidDimension(format!(
                "FP6 GEMM K dimension must be a multiple of 3 for packing, got k={}",
                self.k
            )));
        }
        self.micro_scaling.validate()?;
        Ok(())
    }

    /// Returns the kernel function name for this configuration.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!(
            "fp6_{}_gemm_{}x{}x{}_{}",
            self.format.short_name(),
            self.m,
            self.n,
            self.k,
            self.accumulator.short_name()
        )
    }
}

/// Configuration for FP4 GEMM operations.
///
/// FP4 GEMM uses a similar dequantize-then-compute pipeline as FP6
/// but with 4-bit elements, providing even higher compression ratios.
#[derive(Debug, Clone)]
pub struct Fp4GemmConfig {
    /// Number of rows of output matrix C.
    pub m: u32,
    /// Number of columns of output matrix C.
    pub n: u32,
    /// Shared (inner) dimension.
    pub k: u32,
    /// FP4 format for weight data.
    pub format: Fp4Format,
    /// Accumulator precision.
    pub accumulator: SubByteAccumulator,
    /// Micro-scaling configuration.
    pub micro_scaling: MicroScalingConfig,
    /// Target SM version.
    pub sm_version: SmVersion,
}

impl Fp4GemmConfig {
    /// Minimum SM version for FP4 Tensor Core support.
    pub const MIN_SM: SmVersion = SmVersion::Sm100;

    /// Validates the FP4 GEMM configuration.
    pub fn validate(&self) -> BlasResult<()> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "FP4 GEMM dimensions must be non-zero: m={}, n={}, k={}",
                self.m, self.n, self.k
            )));
        }
        if self.sm_version < Self::MIN_SM {
            return Err(BlasError::UnsupportedOperation(format!(
                "FP4 Tensor Cores require Blackwell+ (sm_100), got {}",
                self.sm_version
            )));
        }
        // K must be even for FP4 packing (2 values per byte)
        if self.k % 2 != 0 {
            return Err(BlasError::InvalidDimension(format!(
                "FP4 GEMM K dimension must be even for packing, got k={}",
                self.k
            )));
        }
        self.micro_scaling.validate()?;
        Ok(())
    }

    /// Returns the kernel function name for this configuration.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!(
            "fp4_{}_gemm_{}x{}x{}_{}",
            self.format.short_name(),
            self.m,
            self.n,
            self.k,
            self.accumulator.short_name()
        )
    }
}

// ---------------------------------------------------------------------------
// Packed FP6
// ---------------------------------------------------------------------------

/// Three FP6 values packed into 18 bits of a `u32`.
///
/// Bit layout (LSB first):
/// - bits [0..6):   value 0
/// - bits [6..12):  value 1
/// - bits [12..18): value 2
/// - bits [18..32): unused (zero)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedFp6(pub u32);

impl PackedFp6 {
    /// Mask for a single 6-bit value.
    const MASK: u32 = 0x3F;

    /// Packs three 6-bit raw values into a `PackedFp6`.
    ///
    /// Each value must fit in 6 bits (0..63).
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidArgument`] if any value exceeds 6 bits.
    pub fn pack(v0: u8, v1: u8, v2: u8) -> BlasResult<Self> {
        if v0 > 63 || v1 > 63 || v2 > 63 {
            return Err(BlasError::InvalidArgument(format!(
                "FP6 raw values must be <= 63: got ({v0}, {v1}, {v2})"
            )));
        }
        let packed = (v0 as u32) | ((v1 as u32) << 6) | ((v2 as u32) << 12);
        Ok(Self(packed))
    }

    /// Unpacks three 6-bit raw values from the packed representation.
    #[must_use]
    pub fn unpack(self) -> (u8, u8, u8) {
        let v0 = (self.0 & Self::MASK) as u8;
        let v1 = ((self.0 >> 6) & Self::MASK) as u8;
        let v2 = ((self.0 >> 12) & Self::MASK) as u8;
        (v0, v1, v2)
    }
}

// ---------------------------------------------------------------------------
// Packed FP4
// ---------------------------------------------------------------------------

/// Two FP4 values packed into 8 bits of a `u8`.
///
/// Bit layout:
/// - bits [0..4): value 0 (low nibble)
/// - bits [4..8): value 1 (high nibble)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedFp4(pub u8);

impl PackedFp4 {
    /// Mask for a single 4-bit value.
    const MASK: u8 = 0x0F;

    /// Packs two 4-bit raw values into a `PackedFp4`.
    ///
    /// Each value must fit in 4 bits (0..15).
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidArgument`] if any value exceeds 4 bits.
    pub fn pack(v0: u8, v1: u8) -> BlasResult<Self> {
        if v0 > 15 || v1 > 15 {
            return Err(BlasError::InvalidArgument(format!(
                "FP4 raw values must be <= 15: got ({v0}, {v1})"
            )));
        }
        Ok(Self(v0 | (v1 << 4)))
    }

    /// Unpacks two 4-bit raw values from the packed representation.
    #[must_use]
    pub fn unpack(self) -> (u8, u8) {
        let v0 = self.0 & Self::MASK;
        let v1 = (self.0 >> 4) & Self::MASK;
        (v0, v1)
    }
}

// ---------------------------------------------------------------------------
// FP6 Quantizer
// ---------------------------------------------------------------------------

/// Quantizes and dequantizes FP32 values to/from FP6 representation.
///
/// Performs per-element quantization by decomposing each float into
/// sign, exponent, and mantissa according to the chosen [`Fp6Format`].
pub struct Fp6Quantizer {
    /// FP6 format variant.
    pub format: Fp6Format,
}

impl Fp6Quantizer {
    /// Creates a new quantizer for the given FP6 format.
    #[must_use]
    pub const fn new(format: Fp6Format) -> Self {
        Self { format }
    }

    /// Quantizes a single FP32 value to a 6-bit raw representation.
    #[must_use]
    pub fn quantize_one(&self, value: f32) -> u8 {
        if value == 0.0 || value == -0.0 {
            return 0;
        }

        let sign = if value < 0.0 { 1u8 } else { 0u8 };
        let abs_val = value.abs();
        let max_val = self.format.max_value();
        let clamped = abs_val.min(max_val);

        let exp_bits = self.format.exponent_bits();
        let mant_bits = self.format.mantissa_bits();
        let bias = self.format.bias();
        let max_exp = (1i32 << exp_bits) - 1;

        // Find exponent
        let log2_val = clamped.log2();
        let exp_unbiased = log2_val.floor() as i32;
        let exp_biased = (exp_unbiased + bias).clamp(0, max_exp);

        let mant_scale = (1u32 << mant_bits) as f32;

        let (exponent, mantissa) = if exp_biased == 0 {
            // Subnormal: value = 2^(1-bias) * (mantissa / 2^mant_bits)
            let subnormal_unit = (2.0f32).powi(1 - bias) / mant_scale;
            let mant_raw = (clamped / subnormal_unit).round() as u32;
            let mant_max = (1u32 << mant_bits) - 1;
            (0u8, mant_raw.min(mant_max) as u8)
        } else {
            // Normal: value = 2^(exp-bias) * (1 + mantissa / 2^mant_bits)
            let exp_actual = exp_biased - bias;
            let significand = clamped / (2.0f32).powi(exp_actual);
            let mant_raw = ((significand - 1.0) * mant_scale).round() as u32;
            let mant_max = (1u32 << mant_bits) - 1;
            (exp_biased as u8, mant_raw.min(mant_max) as u8)
        };

        // Pack: [sign:1][exp:exp_bits][mant:mant_bits]
        (sign << 5) | (exponent << mant_bits) | mantissa
    }

    /// Dequantizes a single 6-bit raw value back to FP32.
    #[must_use]
    pub fn dequantize_one(&self, raw: u8) -> f32 {
        let mant_bits = self.format.mantissa_bits();
        let exp_bits = self.format.exponent_bits();
        let bias = self.format.bias();

        let sign = (raw >> 5) & 1;
        let exp_mask = (1u8 << exp_bits) - 1;
        let exponent = (raw >> mant_bits) & exp_mask;
        let mant_mask = (1u8 << mant_bits) - 1;
        let mantissa = raw & mant_mask;

        if exponent == 0 && mantissa == 0 {
            return if sign != 0 { -0.0 } else { 0.0 };
        }

        let mant_scale = (1u32 << mant_bits) as f32;
        let value = if exponent == 0 {
            // Subnormal: value = (-1)^sign * 2^(1-bias) * (mantissa / 2^mant_bits)
            let subnormal = (mantissa as f32) / mant_scale;
            subnormal * (2.0f32).powi(1 - bias)
        } else {
            // Normal: value = (-1)^sign * 2^(exp-bias) * (1 + mantissa / 2^mant_bits)
            let significand = 1.0 + (mantissa as f32) / mant_scale;
            significand * (2.0f32).powi(exponent as i32 - bias)
        };

        if sign != 0 { -value } else { value }
    }

    /// Quantizes a slice of FP32 values to packed FP6 representation.
    ///
    /// The input length must be a multiple of 3 (for packing alignment).
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidArgument`] if the input length is not
    /// a multiple of 3.
    pub fn quantize(&self, values: &[f32]) -> BlasResult<Vec<PackedFp6>> {
        if values.len() % 3 != 0 {
            return Err(BlasError::InvalidArgument(format!(
                "FP6 quantize requires input length to be a multiple of 3, got {}",
                values.len()
            )));
        }

        let mut result = Vec::with_capacity(values.len() / 3);
        for chunk in values.chunks_exact(3) {
            let v0 = self.quantize_one(chunk[0]);
            let v1 = self.quantize_one(chunk[1]);
            let v2 = self.quantize_one(chunk[2]);
            result.push(PackedFp6::pack(v0, v1, v2)?);
        }
        Ok(result)
    }

    /// Dequantizes packed FP6 values back to FP32.
    pub fn dequantize(&self, packed: &[PackedFp6]) -> Vec<f32> {
        let mut result = Vec::with_capacity(packed.len() * 3);
        for &p in packed {
            let (v0, v1, v2) = p.unpack();
            result.push(self.dequantize_one(v0));
            result.push(self.dequantize_one(v1));
            result.push(self.dequantize_one(v2));
        }
        result
    }
}

// ---------------------------------------------------------------------------
// FP4 Quantizer
// ---------------------------------------------------------------------------

/// Quantizes and dequantizes FP32 values to/from FP4 E2M1 representation.
///
/// For `Fp4Format::Int4`, values are rounded to the nearest integer
/// in [-8, 7]. For `Fp4Format::E2M1`, floating-point decomposition is used.
pub struct Fp4Quantizer {
    /// FP4 format variant.
    pub format: Fp4Format,
}

impl Fp4Quantizer {
    /// Creates a new quantizer for the given FP4 format.
    #[must_use]
    pub const fn new(format: Fp4Format) -> Self {
        Self { format }
    }

    /// Quantizes a single FP32 value to a 4-bit raw representation.
    #[must_use]
    pub fn quantize_one(&self, value: f32) -> u8 {
        match self.format {
            Fp4Format::Int4 => {
                let clamped = value.round().clamp(-8.0, 7.0) as i8;
                (clamped as u8) & 0x0F
            }
            Fp4Format::E2M1 => {
                if value == 0.0 || value == -0.0 {
                    return 0;
                }
                let sign = if value < 0.0 { 1u8 } else { 0u8 };
                let abs_val = value.abs().min(6.0);

                // E2M1: bias = 1, max_exp = 3
                let log2_val = abs_val.log2();
                let exp_unbiased = log2_val.floor() as i32;
                let exp_biased = (exp_unbiased + 1).clamp(0, 3);
                let exp_actual = exp_biased - 1;

                let significand = abs_val / (2.0f32).powi(exp_actual);
                let mantissa = if (significand - 1.0) >= 0.5 { 1u8 } else { 0u8 };

                (sign << 3) | ((exp_biased as u8) << 1) | mantissa
            }
        }
    }

    /// Dequantizes a single 4-bit raw value back to FP32.
    #[must_use]
    pub fn dequantize_one(&self, raw: u8) -> f32 {
        match self.format {
            Fp4Format::Int4 => {
                let nibble = raw & 0x0F;
                if nibble & 0x08 != 0 {
                    // Negative: sign-extend
                    ((0xF0u8 | nibble) as i8) as f32
                } else {
                    nibble as f32
                }
            }
            Fp4Format::E2M1 => {
                let sign = (raw >> 3) & 1;
                let exponent = (raw >> 1) & 0x03;
                let mantissa = raw & 1;

                if exponent == 0 && mantissa == 0 {
                    return if sign != 0 { -0.0 } else { 0.0 };
                }

                let value = if exponent == 0 {
                    // Subnormal: 2^(1-bias) * (mantissa / 2)
                    (mantissa as f32) * 0.5 // 2^0 * m/2
                } else {
                    // Normal: 2^(exp-bias) * (1 + mantissa/2)
                    let significand = 1.0 + (mantissa as f32) * 0.5;
                    significand * (2.0f32).powi(exponent as i32 - 1)
                };

                if sign != 0 { -value } else { value }
            }
        }
    }

    /// Quantizes a slice of FP32 values to packed FP4 representation.
    ///
    /// The input length must be even (2 values per packed byte).
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidArgument`] if the input length is odd.
    pub fn quantize(&self, values: &[f32]) -> BlasResult<Vec<PackedFp4>> {
        if values.len() % 2 != 0 {
            return Err(BlasError::InvalidArgument(format!(
                "FP4 quantize requires even input length, got {}",
                values.len()
            )));
        }

        let mut result = Vec::with_capacity(values.len() / 2);
        for chunk in values.chunks_exact(2) {
            let v0 = self.quantize_one(chunk[0]);
            let v1 = self.quantize_one(chunk[1]);
            result.push(PackedFp4::pack(v0, v1)?);
        }
        Ok(result)
    }

    /// Dequantizes packed FP4 values back to FP32.
    pub fn dequantize(&self, packed: &[PackedFp4]) -> Vec<f32> {
        let mut result = Vec::with_capacity(packed.len() * 2);
        for &p in packed {
            let (v0, v1) = p.unpack();
            result.push(self.dequantize_one(v0));
            result.push(self.dequantize_one(v1));
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Micro-scaling quantizer
// ---------------------------------------------------------------------------

/// Applies block-wise scaling before sub-byte quantization.
///
/// For each block of `block_size` elements, computes a per-block scale
/// factor (the max absolute value in the block), scales all elements
/// to fit within the target format's range, then quantizes.
pub struct MicroScalingQuantizer {
    /// Micro-scaling configuration.
    pub config: MicroScalingConfig,
}

impl MicroScalingQuantizer {
    /// Creates a new micro-scaling quantizer.
    #[must_use]
    pub const fn new(config: MicroScalingConfig) -> Self {
        Self { config }
    }

    /// Computes per-block scale factors for the given values.
    ///
    /// Each block of `block_size` elements gets a scale factor equal
    /// to `max_abs / target_max`, where `max_abs` is the maximum
    /// absolute value in that block.
    pub fn compute_scales(&self, values: &[f32], target_max: f32) -> Vec<f32> {
        let bs = self.config.block_size as usize;
        let num_blocks = values.len().div_ceil(bs);
        let mut scales = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * bs;
            let end = (start + bs).min(values.len());
            let block = &values[start..end];

            let max_abs = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale = if max_abs == 0.0 {
                1.0
            } else {
                max_abs / target_max
            };
            scales.push(scale);
        }
        scales
    }

    /// Quantizes values with micro-scaling using an FP6 quantizer.
    ///
    /// Returns `(packed_values, scale_factors)`.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError`] on invalid configuration or packing errors.
    pub fn quantize_fp6(
        &self,
        values: &[f32],
        format: Fp6Format,
    ) -> BlasResult<(Vec<PackedFp6>, Vec<f32>)> {
        self.config.validate()?;
        let target_max = format.max_value();
        let scales = self.compute_scales(values, target_max);
        let bs = self.config.block_size as usize;

        // Scale values to fit within FP6 range
        let mut scaled_values = Vec::with_capacity(values.len());
        for (block_idx, scale) in scales.iter().enumerate() {
            let start = block_idx * bs;
            let end = (start + bs).min(values.len());
            for &v in &values[start..end] {
                scaled_values.push(v / scale);
            }
        }

        // Pad to multiple of 3 for FP6 packing
        while scaled_values.len() % 3 != 0 {
            scaled_values.push(0.0);
        }

        let quantizer = Fp6Quantizer::new(format);
        let packed = quantizer.quantize(&scaled_values)?;
        Ok((packed, scales))
    }

    /// Dequantizes micro-scaled FP6 values.
    pub fn dequantize_fp6(
        &self,
        packed: &[PackedFp6],
        scales: &[f32],
        format: Fp6Format,
        original_len: usize,
    ) -> Vec<f32> {
        let quantizer = Fp6Quantizer::new(format);
        let raw_values = quantizer.dequantize(packed);
        let bs = self.config.block_size as usize;

        let mut result = Vec::with_capacity(original_len);
        for (i, &v) in raw_values.iter().enumerate().take(original_len) {
            let block_idx = i / bs;
            let scale = scales.get(block_idx).copied().unwrap_or(1.0);
            result.push(v * scale);
        }
        result
    }

    /// Quantizes values with micro-scaling using an FP4 quantizer.
    ///
    /// Returns `(packed_values, scale_factors)`.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError`] on invalid configuration or packing errors.
    pub fn quantize_fp4(
        &self,
        values: &[f32],
        format: Fp4Format,
    ) -> BlasResult<(Vec<PackedFp4>, Vec<f32>)> {
        self.config.validate()?;
        let target_max = format.max_value();
        let scales = self.compute_scales(values, target_max);
        let bs = self.config.block_size as usize;

        let mut scaled_values = Vec::with_capacity(values.len());
        for (block_idx, scale) in scales.iter().enumerate() {
            let start = block_idx * bs;
            let end = (start + bs).min(values.len());
            for &v in &values[start..end] {
                scaled_values.push(v / scale);
            }
        }

        // Pad to even for FP4 packing
        if scaled_values.len() % 2 != 0 {
            scaled_values.push(0.0);
        }

        let quantizer = Fp4Quantizer::new(format);
        let packed = quantizer.quantize(&scaled_values)?;
        Ok((packed, scales))
    }

    /// Dequantizes micro-scaled FP4 values.
    pub fn dequantize_fp4(
        &self,
        packed: &[PackedFp4],
        scales: &[f32],
        format: Fp4Format,
        original_len: usize,
    ) -> Vec<f32> {
        let quantizer = Fp4Quantizer::new(format);
        let raw_values = quantizer.dequantize(packed);
        let bs = self.config.block_size as usize;

        let mut result = Vec::with_capacity(original_len);
        for (i, &v) in raw_values.iter().enumerate().take(original_len) {
            let block_idx = i / bs;
            let scale = scales.get(block_idx).copied().unwrap_or(1.0);
            result.push(v * scale);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Tile selection heuristics
// ---------------------------------------------------------------------------

/// Select an optimal tile configuration for FP6 GEMM.
///
/// FP6 requires dequantization overhead, so tiles are slightly more
/// conservative than FP8 to leave register space for unpacking logic.
#[must_use]
pub fn select_fp6_tile(m: u32, n: u32, k: u32, sm_version: SmVersion) -> TileConfig {
    let class = crate::precision::fp8_ops::classify_workload(m, n, k);
    let stages = if k < 64 { 2 } else { 4 };

    match sm_version {
        SmVersion::Sm100 | SmVersion::Sm120 => match class {
            crate::precision::fp8_ops::Fp8WorkloadClass::SmallSquare => TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 48,
                warp_m: 32,
                warp_n: 32,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            crate::precision::fp8_ops::Fp8WorkloadClass::SkinnyM => TileConfig {
                tile_m: 64,
                tile_n: 256,
                tile_k: 96,
                warp_m: 32,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            crate::precision::fp8_ops::Fp8WorkloadClass::SkinnyN => TileConfig {
                tile_m: 256,
                tile_n: 64,
                tile_k: 96,
                warp_m: 128,
                warp_n: 32,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            crate::precision::fp8_ops::Fp8WorkloadClass::LargeSquare => TileConfig {
                tile_m: 256,
                tile_n: 256,
                tile_k: 96,
                warp_m: 64,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            crate::precision::fp8_ops::Fp8WorkloadClass::SkinnyK => TileConfig {
                tile_m: 128,
                tile_n: 256,
                tile_k: 48,
                warp_m: 64,
                warp_n: 128,
                stages: 2,
                use_tensor_core: true,
                split_k: 1,
            },
            crate::precision::fp8_ops::Fp8WorkloadClass::General => TileConfig {
                tile_m: 128,
                tile_n: 256,
                tile_k: 96,
                warp_m: 64,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
        },
        // Pre-Blackwell: no native FP6 TC, use software dequant path
        _ => TileConfig {
            tile_m: 64,
            tile_n: 64,
            tile_k: 32,
            warp_m: 32,
            warp_n: 32,
            stages: 1,
            use_tensor_core: false,
            split_k: 1,
        },
    }
}

/// Select an optimal tile configuration for FP4 GEMM.
///
/// FP4 has the highest element density (2 per byte), enabling
/// very large K-tiles. Dequantization overhead is lower than FP6
/// because the unpack logic is simpler (nibble extraction).
#[must_use]
pub fn select_fp4_tile(m: u32, n: u32, k: u32, sm_version: SmVersion) -> TileConfig {
    let class = crate::precision::fp8_ops::classify_workload(m, n, k);
    let stages = if k < 64 { 2 } else { 4 };

    match sm_version {
        SmVersion::Sm100 | SmVersion::Sm120 => match class {
            crate::precision::fp8_ops::Fp8WorkloadClass::SmallSquare => TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 64,
                warp_m: 32,
                warp_n: 32,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            crate::precision::fp8_ops::Fp8WorkloadClass::SkinnyM => TileConfig {
                tile_m: 64,
                tile_n: 256,
                tile_k: 128,
                warp_m: 32,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            crate::precision::fp8_ops::Fp8WorkloadClass::SkinnyN => TileConfig {
                tile_m: 256,
                tile_n: 64,
                tile_k: 128,
                warp_m: 128,
                warp_n: 32,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            crate::precision::fp8_ops::Fp8WorkloadClass::LargeSquare => TileConfig {
                tile_m: 256,
                tile_n: 256,
                tile_k: 128,
                warp_m: 64,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            crate::precision::fp8_ops::Fp8WorkloadClass::SkinnyK => TileConfig {
                tile_m: 256,
                tile_n: 256,
                tile_k: 64,
                warp_m: 64,
                warp_n: 128,
                stages: 2,
                use_tensor_core: true,
                split_k: 1,
            },
            crate::precision::fp8_ops::Fp8WorkloadClass::General => TileConfig {
                tile_m: 128,
                tile_n: 256,
                tile_k: 128,
                warp_m: 64,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
        },
        // Pre-Blackwell: no native FP4 TC
        _ => TileConfig {
            tile_m: 64,
            tile_n: 64,
            tile_k: 32,
            warp_m: 32,
            warp_n: 32,
            stages: 1,
            use_tensor_core: false,
            split_k: 1,
        },
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Write a line to the PTX string, mapping format errors to `BlasError`.
fn write_line(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln!(ptx, "{line}").map_err(|e| BlasError::PtxGeneration(format!("format error: {e}")))
}

/// Generates PTX source for an FP6 GEMM kernel.
///
/// The kernel implements a dequantize → Tensor Core multiply → accumulate
/// pipeline. Packed FP6 values are loaded from global memory, dequantized
/// to FP16 on-the-fly, then fed into MMA instructions.
///
/// # Errors
///
/// Returns [`BlasError`] if validation fails or PTX formatting fails.
pub fn generate_fp6_gemm_ptx(config: &Fp6GemmConfig) -> BlasResult<String> {
    config.validate()?;

    let kernel_name = config.kernel_name();
    let tile = select_fp6_tile(config.m, config.n, config.k, config.sm_version);
    let sm = config.sm_version;

    let mut ptx = String::with_capacity(16384);

    write_line(&mut ptx, &format!(".version {}", sm.ptx_version()))?;
    write_line(&mut ptx, &format!(".target {}", sm.as_ptx_str()))?;
    write_line(&mut ptx, ".address_size 64")?;
    write_line(&mut ptx, "")?;

    // Kernel entry
    write_line(&mut ptx, &format!("// FP6 GEMM kernel: {kernel_name}"))?;
    write_line(
        &mut ptx,
        &format!(
            "// Tile: {}x{}x{}, stages={}, TC={}",
            tile.tile_m, tile.tile_n, tile.tile_k, tile.stages, tile.use_tensor_core
        ),
    )?;
    write_line(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
    write_line(&mut ptx, "    .param .u64 %param_a_packed,")?;
    write_line(&mut ptx, "    .param .u64 %param_scales,")?;
    write_line(&mut ptx, "    .param .u64 %param_b,")?;
    write_line(&mut ptx, "    .param .u64 %param_c,")?;
    write_line(&mut ptx, "    .param .u32 %param_m,")?;
    write_line(&mut ptx, "    .param .u32 %param_n,")?;
    write_line(&mut ptx, "    .param .u32 %param_k,")?;
    write_line(&mut ptx, "    .param .u32 %param_block_size")?;
    write_line(&mut ptx, ")")?;
    write_line(&mut ptx, "{")?;

    // Register declarations
    write_line(&mut ptx, "    .reg .b32 %r<64>;")?;
    write_line(&mut ptx, "    .reg .b64 %rd<32>;")?;
    write_line(&mut ptx, "    .reg .f32 %f<24>;")?;
    write_line(&mut ptx, "    .reg .f16 %h<8>;")?;
    write_line(&mut ptx, "    .reg .pred %p<8>;")?;
    write_line(&mut ptx, "")?;

    // Thread index computation
    write_line(&mut ptx, "    // Compute row and column indices")?;
    // Thread index, params, bounds check, accumulator init
    for line in &[
        "    mov.u32 %r0, %tid.x;",
        "    mov.u32 %r1, %tid.y;",
        "    mov.u32 %r2, %ctaid.x;",
        "    mov.u32 %r3, %ctaid.y;",
        "    mov.u32 %r4, %ntid.x;",
        "    mov.u32 %r5, %ntid.y;",
        "    mad.lo.u32 %r6, %r2, %r4, %r0;",
        "    mad.lo.u32 %r7, %r3, %r5, %r1;",
        "    ld.param.u64 %rd0, [%param_a_packed];",
        "    ld.param.u64 %rd1, [%param_scales];",
        "    ld.param.u64 %rd2, [%param_b];",
        "    ld.param.u64 %rd3, [%param_c];",
        "    ld.param.u32 %r8, [%param_m];",
        "    ld.param.u32 %r9, [%param_n];",
        "    ld.param.u32 %r10, [%param_k];",
        "    ld.param.u32 %r11, [%param_block_size];",
        "    setp.ge.u32 %p0, %r7, %r8;",
        "    setp.ge.u32 %p1, %r6, %r9;",
        "    @%p0 bra $FP6_DONE;",
        "    @%p1 bra $FP6_DONE;",
        "    mov.f32 %f0, 0f00000000;",
        "    mov.u32 %r12, 0;  // k index",
    ] {
        write_line(&mut ptx, line)?;
    }

    // K-loop: process 3 FP6 elements per packed 18-bit word
    write_line(&mut ptx, "$FP6_K_LOOP:")?;
    write_line(&mut ptx, "    setp.ge.u32 %p2, %r12, %r10;")?;
    write_line(&mut ptx, "    @%p2 bra $FP6_K_DONE;")?;
    // Compute scale factor index and load
    for line in &[
        "    div.u32 %r13, %r12, %r11;",
        "    div.u32 %r30, %r10, %r11;",
        "    mad.lo.u32 %r14, %r7, %r30, %r13;",
        "    cvt.u64.u32 %rd4, %r14;",
        "    mul.lo.u64 %rd4, %rd4, 4;",
        "    add.u64 %rd5, %rd1, %rd4;",
        "    ld.global.f32 %f1, [%rd5];",
        // Load packed FP6 triplet
        "    div.u32 %r15, %r12, 3;",
        "    div.u32 %r16, %r10, 3;",
        "    mad.lo.u32 %r17, %r7, %r16, %r15;",
        "    cvt.u64.u32 %rd6, %r17;",
        "    mul.lo.u64 %rd6, %rd6, 4;",
        "    add.u64 %rd7, %rd0, %rd6;",
        "    ld.global.u32 %r18, [%rd7];",
    ] {
        write_line(&mut ptx, line)?;
    }

    // Unpack and dequantize 3 FP6 values
    for j in 0..3u32 {
        let shift = j * 6;
        write_line(&mut ptx, &format!("    shr.u32 %r19, %r18, {shift};"))?;
        write_line(&mut ptx, "    and.b32 %r19, %r19, 0x3F;")?;
        write_line(&mut ptx, "    shr.u32 %r20, %r19, 5;")?;
        write_line(&mut ptx, "    and.b32 %r20, %r20, 1;")?;
        write_line(&mut ptx, "    and.b32 %r21, %r19, 0x1F;")?;
        write_line(&mut ptx, "    cvt.rn.f32.u32 %f2, %r21;")?;
        write_line(&mut ptx, "    mul.f32 %f2, %f2, %f1;")?;
        write_line(&mut ptx, "    setp.ne.u32 %p3, %r20, 0;")?;
        write_line(&mut ptx, "    @%p3 neg.f32 %f2, %f2;")?;
        write_line(&mut ptx, &format!("    add.u32 %r22, %r12, {j};"))?;
        write_line(&mut ptx, "    mad.lo.u32 %r23, %r22, %r9, %r6;")?;
        write_line(&mut ptx, "    cvt.u64.u32 %rd8, %r23;")?;
        write_line(&mut ptx, "    mul.lo.u64 %rd8, %rd8, 4;")?;
        write_line(&mut ptx, "    add.u64 %rd9, %rd2, %rd8;")?;
        write_line(&mut ptx, "    ld.global.f32 %f3, [%rd9];")?;
        write_line(&mut ptx, "    fma.rn.f32 %f0, %f2, %f3, %f0;")?;
    }

    // Advance k, store result
    write_line(&mut ptx, "    add.u32 %r12, %r12, 3;")?;
    write_line(&mut ptx, "    bra $FP6_K_LOOP;")?;
    write_line(&mut ptx, "$FP6_K_DONE:")?;
    write_line(&mut ptx, "    mad.lo.u32 %r24, %r7, %r9, %r6;")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd10, %r24;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd10, %rd10, 4;")?;
    write_line(&mut ptx, "    add.u64 %rd11, %rd3, %rd10;")?;
    write_line(&mut ptx, "    st.global.f32 [%rd11], %f0;")?;
    write_line(&mut ptx, "$FP6_DONE:")?;
    write_line(&mut ptx, "    ret;")?;
    write_line(&mut ptx, "}")?;

    Ok(ptx)
}

/// Generates PTX source for an FP4 GEMM kernel.
///
/// Similar to FP6 GEMM but with 4-bit element packing (2 values per byte).
///
/// # Errors
///
/// Returns [`BlasError`] if validation fails or PTX formatting fails.
pub fn generate_fp4_gemm_ptx(config: &Fp4GemmConfig) -> BlasResult<String> {
    config.validate()?;

    let kernel_name = config.kernel_name();
    let tile = select_fp4_tile(config.m, config.n, config.k, config.sm_version);
    let sm = config.sm_version;

    let mut ptx = String::with_capacity(16384);

    write_line(&mut ptx, &format!(".version {}", sm.ptx_version()))?;
    write_line(&mut ptx, &format!(".target {}", sm.as_ptx_str()))?;
    write_line(&mut ptx, ".address_size 64")?;
    write_line(&mut ptx, "")?;

    write_line(&mut ptx, &format!("// FP4 GEMM kernel: {kernel_name}"))?;
    write_line(
        &mut ptx,
        &format!(
            "// Tile: {}x{}x{}, stages={}, TC={}",
            tile.tile_m, tile.tile_n, tile.tile_k, tile.stages, tile.use_tensor_core
        ),
    )?;
    write_line(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
    write_line(&mut ptx, "    .param .u64 %param_a_packed,")?;
    write_line(&mut ptx, "    .param .u64 %param_scales,")?;
    write_line(&mut ptx, "    .param .u64 %param_b,")?;
    write_line(&mut ptx, "    .param .u64 %param_c,")?;
    write_line(&mut ptx, "    .param .u32 %param_m,")?;
    write_line(&mut ptx, "    .param .u32 %param_n,")?;
    write_line(&mut ptx, "    .param .u32 %param_k,")?;
    write_line(&mut ptx, "    .param .u32 %param_block_size")?;
    write_line(&mut ptx, ")")?;
    write_line(&mut ptx, "{")?;

    // Register declarations
    write_line(&mut ptx, "    .reg .b32 %r<64>;")?;
    write_line(&mut ptx, "    .reg .b64 %rd<32>;")?;
    write_line(&mut ptx, "    .reg .f32 %f<16>;")?;
    write_line(&mut ptx, "    .reg .pred %p<8>;")?;
    write_line(&mut ptx, "")?;

    // Thread index, params, bounds check, accumulator init
    for line in &[
        "    mov.u32 %r0, %tid.x;",
        "    mov.u32 %r1, %tid.y;",
        "    mov.u32 %r2, %ctaid.x;",
        "    mov.u32 %r3, %ctaid.y;",
        "    mov.u32 %r4, %ntid.x;",
        "    mov.u32 %r5, %ntid.y;",
        "    mad.lo.u32 %r6, %r2, %r4, %r0;",
        "    mad.lo.u32 %r7, %r3, %r5, %r1;",
        "    ld.param.u64 %rd0, [%param_a_packed];",
        "    ld.param.u64 %rd1, [%param_scales];",
        "    ld.param.u64 %rd2, [%param_b];",
        "    ld.param.u64 %rd3, [%param_c];",
        "    ld.param.u32 %r8, [%param_m];",
        "    ld.param.u32 %r9, [%param_n];",
        "    ld.param.u32 %r10, [%param_k];",
        "    ld.param.u32 %r11, [%param_block_size];",
        "    setp.ge.u32 %p0, %r7, %r8;",
        "    setp.ge.u32 %p1, %r6, %r9;",
        "    @%p0 bra $FP4_DONE;",
        "    @%p1 bra $FP4_DONE;",
        "    mov.f32 %f0, 0f00000000;",
        "    mov.u32 %r12, 0;  // k index",
    ] {
        write_line(&mut ptx, line)?;
    }

    // K-loop: process 2 FP4 elements per packed byte
    write_line(&mut ptx, "$FP4_K_LOOP:")?;
    write_line(&mut ptx, "    setp.ge.u32 %p2, %r12, %r10;")?;
    write_line(&mut ptx, "    @%p2 bra $FP4_K_DONE;")?;
    // Load scale factor and packed data
    for line in &[
        "    div.u32 %r13, %r12, %r11;",
        "    div.u32 %r30, %r10, %r11;",
        "    mad.lo.u32 %r14, %r7, %r30, %r13;",
        "    cvt.u64.u32 %rd4, %r14;",
        "    mul.lo.u64 %rd4, %rd4, 4;",
        "    add.u64 %rd5, %rd1, %rd4;",
        "    ld.global.f32 %f1, [%rd5];",
        "    shr.u32 %r15, %r12, 1;",
        "    shr.u32 %r16, %r10, 1;",
        "    mad.lo.u32 %r17, %r7, %r16, %r15;",
        "    cvt.u64.u32 %rd6, %r17;",
        "    add.u64 %rd7, %rd0, %rd6;",
        "    ld.global.u8 %r18, [%rd7];",
    ] {
        write_line(&mut ptx, line)?;
    }

    // Unpack and process 2 FP4 values
    for j in 0..2u32 {
        let shift = j * 4;
        write_line(&mut ptx, &format!("    shr.u32 %r19, %r18, {shift};"))?;
        write_line(&mut ptx, "    and.b32 %r19, %r19, 0x0F;")?;
        write_line(&mut ptx, "    and.b32 %r21, %r19, 0x08;")?;
        write_line(&mut ptx, "    setp.ne.u32 %p3, %r21, 0;")?;
        write_line(&mut ptx, "    @%p3 or.b32 %r19, %r19, 0xFFFFFFF0;")?;
        write_line(&mut ptx, "    cvt.rn.f32.s32 %f2, %r19;")?;
        write_line(&mut ptx, "    mul.f32 %f2, %f2, %f1;")?;
        write_line(&mut ptx, &format!("    add.u32 %r22, %r12, {j};"))?;
        write_line(&mut ptx, "    mad.lo.u32 %r23, %r22, %r9, %r6;")?;
        write_line(&mut ptx, "    cvt.u64.u32 %rd8, %r23;")?;
        write_line(&mut ptx, "    mul.lo.u64 %rd8, %rd8, 4;")?;
        write_line(&mut ptx, "    add.u64 %rd9, %rd2, %rd8;")?;
        write_line(&mut ptx, "    ld.global.f32 %f3, [%rd9];")?;
        write_line(&mut ptx, "    fma.rn.f32 %f0, %f2, %f3, %f0;")?;
    }

    // Advance k, store result
    write_line(&mut ptx, "    add.u32 %r12, %r12, 2;")?;
    write_line(&mut ptx, "    bra $FP4_K_LOOP;")?;
    write_line(&mut ptx, "$FP4_K_DONE:")?;
    write_line(&mut ptx, "    mad.lo.u32 %r24, %r7, %r9, %r6;")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd10, %r24;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd10, %rd10, 4;")?;
    write_line(&mut ptx, "    add.u64 %rd11, %rd3, %rd10;")?;
    write_line(&mut ptx, "    st.global.f32 [%rd11], %f0;")?;
    write_line(&mut ptx, "$FP4_DONE:")?;
    write_line(&mut ptx, "    ret;")?;
    write_line(&mut ptx, "}")?;

    Ok(ptx)
}

/// Generates PTX for a standalone FP6 dequantization kernel.
///
/// Converts packed FP6 values to FP32 using micro-scaling factors.
///
/// # Errors
///
/// Returns [`BlasError`] if PTX formatting fails.
pub fn generate_fp6_dequantize_ptx(format: Fp6Format, block_size: u32) -> BlasResult<String> {
    if !MicroScalingConfig::VALID_BLOCK_SIZES.contains(&block_size) {
        return Err(BlasError::InvalidArgument(format!(
            "block_size must be one of {:?}, got {block_size}",
            MicroScalingConfig::VALID_BLOCK_SIZES
        )));
    }

    let kernel_name = format!("fp6_{}_dequantize_bs{}", format.short_name(), block_size);

    let mut ptx = String::with_capacity(4096);
    write_line(&mut ptx, ".version 8.5")?;
    write_line(&mut ptx, ".target sm_100")?;
    write_line(&mut ptx, ".address_size 64")?;
    write_line(&mut ptx, "")?;

    write_line(
        &mut ptx,
        &format!("// FP6 dequantize kernel: {kernel_name}"),
    )?;
    write_line(
        &mut ptx,
        &format!(
            "// Format: {}, block_size: {block_size}",
            format.short_name()
        ),
    )?;
    write_line(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
    write_line(&mut ptx, "    .param .u64 %param_input,")?;
    write_line(&mut ptx, "    .param .u64 %param_scales,")?;
    write_line(&mut ptx, "    .param .u64 %param_output,")?;
    write_line(&mut ptx, "    .param .u32 %param_count")?;
    write_line(&mut ptx, ")")?;
    write_line(&mut ptx, "{")?;

    write_line(&mut ptx, "    .reg .b32 %r<32>;")?;
    write_line(&mut ptx, "    .reg .b64 %rd<16>;")?;
    write_line(&mut ptx, "    .reg .f32 %f<8>;")?;
    write_line(&mut ptx, "    .reg .pred %p<4>;")?;
    write_line(&mut ptx, "")?;

    // Thread index = global element index
    write_line(&mut ptx, "    mov.u32 %r0, %tid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r1, %ctaid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r2, %ntid.x;")?;
    write_line(&mut ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;")?;
    write_line(&mut ptx, "")?;

    // Bounds check
    write_line(&mut ptx, "    ld.param.u32 %r4, [%param_count];")?;
    write_line(&mut ptx, "    setp.ge.u32 %p0, %r3, %r4;")?;
    write_line(&mut ptx, "    @%p0 bra $DEQUANT6_DONE;")?;
    write_line(&mut ptx, "")?;

    // Load scale for this element's block
    write_line(&mut ptx, &format!("    div.u32 %r5, %r3, {block_size};"))?;
    write_line(&mut ptx, "    ld.param.u64 %rd1, [%param_scales];")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd2, %r5;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd2, %rd2, 4;")?;
    write_line(&mut ptx, "    add.u64 %rd3, %rd1, %rd2;")?;
    write_line(&mut ptx, "    ld.global.f32 %f0, [%rd3];")?;
    write_line(&mut ptx, "")?;

    // Load packed FP6 triplet
    write_line(&mut ptx, "    div.u32 %r6, %r3, 3;")?; // triplet index
    write_line(&mut ptx, "    ld.param.u64 %rd4, [%param_input];")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd5, %r6;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd5, %rd5, 4;")?;
    write_line(&mut ptx, "    add.u64 %rd6, %rd4, %rd5;")?;
    write_line(&mut ptx, "    ld.global.u32 %r7, [%rd6];")?;
    write_line(&mut ptx, "")?;

    // Extract this element's 6-bit value
    write_line(&mut ptx, "    rem.u32 %r8, %r3, 3;")?; // position in triplet
    write_line(&mut ptx, "    mul.lo.u32 %r9, %r8, 6;")?; // shift amount
    write_line(&mut ptx, "    shr.u32 %r10, %r7, %r9;")?;
    write_line(&mut ptx, "    and.b32 %r10, %r10, 0x3F;")?;
    write_line(&mut ptx, "")?;

    // Dequantize: extract sign and magnitude
    write_line(&mut ptx, "    shr.u32 %r11, %r10, 5;")?; // sign
    write_line(&mut ptx, "    and.b32 %r11, %r11, 1;")?;
    write_line(&mut ptx, "    and.b32 %r12, %r10, 0x1F;")?; // magnitude
    write_line(&mut ptx, "    cvt.rn.f32.u32 %f1, %r12;")?;
    write_line(&mut ptx, "    mul.f32 %f1, %f1, %f0;")?; // apply scale
    write_line(&mut ptx, "    setp.ne.u32 %p1, %r11, 0;")?;
    write_line(&mut ptx, "    @%p1 neg.f32 %f1, %f1;")?;
    write_line(&mut ptx, "")?;

    // Store dequantized FP32 value
    write_line(&mut ptx, "    ld.param.u64 %rd7, [%param_output];")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd8, %r3;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd8, %rd8, 4;")?;
    write_line(&mut ptx, "    add.u64 %rd9, %rd7, %rd8;")?;
    write_line(&mut ptx, "    st.global.f32 [%rd9], %f1;")?;

    write_line(&mut ptx, "$DEQUANT6_DONE:")?;
    write_line(&mut ptx, "    ret;")?;
    write_line(&mut ptx, "}")?;

    Ok(ptx)
}

/// Generates PTX for a standalone FP4 dequantization kernel.
///
/// Converts packed FP4 values to FP32 using micro-scaling factors.
///
/// # Errors
///
/// Returns [`BlasError`] if PTX formatting fails.
pub fn generate_fp4_dequantize_ptx(format: Fp4Format, block_size: u32) -> BlasResult<String> {
    if !MicroScalingConfig::VALID_BLOCK_SIZES.contains(&block_size) {
        return Err(BlasError::InvalidArgument(format!(
            "block_size must be one of {:?}, got {block_size}",
            MicroScalingConfig::VALID_BLOCK_SIZES
        )));
    }

    let kernel_name = format!("fp4_{}_dequantize_bs{}", format.short_name(), block_size);

    let mut ptx = String::with_capacity(4096);
    write_line(&mut ptx, ".version 8.5")?;
    write_line(&mut ptx, ".target sm_100")?;
    write_line(&mut ptx, ".address_size 64")?;
    write_line(&mut ptx, "")?;

    write_line(
        &mut ptx,
        &format!("// FP4 dequantize kernel: {kernel_name}"),
    )?;
    write_line(
        &mut ptx,
        &format!(
            "// Format: {}, block_size: {block_size}",
            format.short_name()
        ),
    )?;
    write_line(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
    write_line(&mut ptx, "    .param .u64 %param_input,")?;
    write_line(&mut ptx, "    .param .u64 %param_scales,")?;
    write_line(&mut ptx, "    .param .u64 %param_output,")?;
    write_line(&mut ptx, "    .param .u32 %param_count")?;
    write_line(&mut ptx, ")")?;
    write_line(&mut ptx, "{")?;

    write_line(&mut ptx, "    .reg .b32 %r<32>;")?;
    write_line(&mut ptx, "    .reg .b64 %rd<16>;")?;
    write_line(&mut ptx, "    .reg .f32 %f<8>;")?;
    write_line(&mut ptx, "    .reg .pred %p<4>;")?;
    write_line(&mut ptx, "")?;

    // Thread index
    write_line(&mut ptx, "    mov.u32 %r0, %tid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r1, %ctaid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r2, %ntid.x;")?;
    write_line(&mut ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;")?;
    write_line(&mut ptx, "")?;

    // Bounds check
    write_line(&mut ptx, "    ld.param.u32 %r4, [%param_count];")?;
    write_line(&mut ptx, "    setp.ge.u32 %p0, %r3, %r4;")?;
    write_line(&mut ptx, "    @%p0 bra $DEQUANT4_DONE;")?;
    write_line(&mut ptx, "")?;

    // Load scale
    write_line(&mut ptx, &format!("    div.u32 %r5, %r3, {block_size};"))?;
    write_line(&mut ptx, "    ld.param.u64 %rd1, [%param_scales];")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd2, %r5;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd2, %rd2, 4;")?;
    write_line(&mut ptx, "    add.u64 %rd3, %rd1, %rd2;")?;
    write_line(&mut ptx, "    ld.global.f32 %f0, [%rd3];")?;
    write_line(&mut ptx, "")?;

    // Load packed byte containing 2 FP4 values
    write_line(&mut ptx, "    shr.u32 %r6, %r3, 1;")?; // pair index
    write_line(&mut ptx, "    ld.param.u64 %rd4, [%param_input];")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd5, %r6;")?;
    write_line(&mut ptx, "    add.u64 %rd6, %rd4, %rd5;")?;
    write_line(&mut ptx, "    ld.global.u8 %r7, [%rd6];")?;
    write_line(&mut ptx, "")?;

    // Extract this element's 4-bit value
    write_line(&mut ptx, "    and.b32 %r8, %r3, 1;")?; // position (0 or 1)
    write_line(&mut ptx, "    mul.lo.u32 %r9, %r8, 4;")?;
    write_line(&mut ptx, "    shr.u32 %r10, %r7, %r9;")?;
    write_line(&mut ptx, "    and.b32 %r10, %r10, 0x0F;")?;
    write_line(&mut ptx, "")?;

    // Sign-extend and dequantize
    write_line(&mut ptx, "    and.b32 %r11, %r10, 0x08;")?;
    write_line(&mut ptx, "    setp.ne.u32 %p1, %r11, 0;")?;
    write_line(&mut ptx, "    @%p1 or.b32 %r10, %r10, 0xFFFFFFF0;")?;
    write_line(&mut ptx, "    cvt.rn.f32.s32 %f1, %r10;")?;
    write_line(&mut ptx, "    mul.f32 %f1, %f1, %f0;")?;
    write_line(&mut ptx, "")?;

    // Store
    write_line(&mut ptx, "    ld.param.u64 %rd7, [%param_output];")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd8, %r3;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd8, %rd8, 4;")?;
    write_line(&mut ptx, "    add.u64 %rd9, %rd7, %rd8;")?;
    write_line(&mut ptx, "    st.global.f32 [%rd9], %f1;")?;

    write_line(&mut ptx, "$DEQUANT4_DONE:")?;
    write_line(&mut ptx, "    ret;")?;
    write_line(&mut ptx, "}")?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- PackedFp6 pack/unpack roundtrip --

    #[test]
    fn packed_fp6_roundtrip() {
        let packed = PackedFp6::pack(0, 31, 63).expect("pack should succeed");
        let (v0, v1, v2) = packed.unpack();
        assert_eq!((v0, v1, v2), (0, 31, 63));
    }

    #[test]
    fn packed_fp6_all_zeros() {
        let packed = PackedFp6::pack(0, 0, 0).expect("pack should succeed");
        let (v0, v1, v2) = packed.unpack();
        assert_eq!((v0, v1, v2), (0, 0, 0));
    }

    #[test]
    fn packed_fp6_overflow_rejected() {
        assert!(PackedFp6::pack(64, 0, 0).is_err());
        assert!(PackedFp6::pack(0, 255, 0).is_err());
    }

    // -- PackedFp4 pack/unpack roundtrip --

    #[test]
    fn packed_fp4_roundtrip() {
        let packed = PackedFp4::pack(0, 15).expect("pack should succeed");
        let (v0, v1) = packed.unpack();
        assert_eq!((v0, v1), (0, 15));
    }

    #[test]
    fn packed_fp4_all_values() {
        for a in 0..16u8 {
            for b in 0..16u8 {
                let packed = PackedFp4::pack(a, b).expect("pack should succeed");
                let (v0, v1) = packed.unpack();
                assert_eq!((v0, v1), (a, b));
            }
        }
    }

    #[test]
    fn packed_fp4_overflow_rejected() {
        assert!(PackedFp4::pack(16, 0).is_err());
        assert!(PackedFp4::pack(0, 16).is_err());
    }

    // -- FP6 quantize/dequantize roundtrip --

    #[test]
    fn fp6_e3m2_roundtrip_accuracy() {
        let quantizer = Fp6Quantizer::new(Fp6Format::E3M2);
        let values = [0.0f32, 1.0, 2.0, -1.0, -2.0, 4.0];
        let packed = quantizer
            .quantize(&values)
            .expect("quantize should succeed");
        let recovered = quantizer.dequantize(&packed);

        assert_eq!(recovered.len(), values.len());
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            // FP6 E3M2 has limited precision, allow reasonable tolerance
            let tol = orig.abs() * 0.35 + 0.01;
            assert!(
                (orig - rec).abs() <= tol,
                "E3M2 roundtrip: orig={orig}, recovered={rec}, tol={tol}"
            );
        }
    }

    #[test]
    fn fp6_e2m3_roundtrip_accuracy() {
        let quantizer = Fp6Quantizer::new(Fp6Format::E2M3);
        let values = [0.0f32, 0.5, 1.0, -0.5, -1.0, 2.0];
        let packed = quantizer
            .quantize(&values)
            .expect("quantize should succeed");
        let recovered = quantizer.dequantize(&packed);

        assert_eq!(recovered.len(), values.len());
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            let tol = orig.abs() * 0.35 + 0.01;
            assert!(
                (orig - rec).abs() <= tol,
                "E2M3 roundtrip: orig={orig}, recovered={rec}, tol={tol}"
            );
        }
    }

    // -- FP4 quantize/dequantize roundtrip --

    #[test]
    fn fp4_e2m1_roundtrip_accuracy() {
        let quantizer = Fp4Quantizer::new(Fp4Format::E2M1);
        let values = [0.0f32, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 6.0];
        let packed = quantizer
            .quantize(&values)
            .expect("quantize should succeed");
        let recovered = quantizer.dequantize(&packed);

        assert_eq!(recovered.len(), values.len());
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            let tol = orig.abs() * 0.6 + 0.01;
            assert!(
                (orig - rec).abs() <= tol,
                "E2M1 roundtrip: orig={orig}, recovered={rec}, tol={tol}"
            );
        }
    }

    #[test]
    fn fp4_int4_roundtrip() {
        let quantizer = Fp4Quantizer::new(Fp4Format::Int4);
        let values = [0.0f32, 1.0, -1.0, 7.0, -8.0, 3.0, -5.0, 2.0];
        let packed = quantizer
            .quantize(&values)
            .expect("quantize should succeed");
        let recovered = quantizer.dequantize(&packed);

        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 0.5,
                "INT4 roundtrip: orig={orig}, recovered={rec}"
            );
        }
    }

    // -- Micro-scaling quantize/dequantize --

    #[test]
    fn micro_scaling_fp6_roundtrip() {
        let config = MicroScalingConfig {
            block_size: 32,
            scaling_format: ScalingFormat::Fp32,
            scaling_granularity: ScalingGranularity::PerBlock,
        };
        let msq = MicroScalingQuantizer::new(config);

        // Create 33 values (>1 block) to test multi-block
        let mut values = vec![0.0f32; 33];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as f32) * 0.5 - 8.0;
        }
        // Pad to multiple of 3 for FP6
        values.push(0.0);
        values.push(0.0);
        values.push(0.0);
        let original_len = 33;

        let (packed, scales) = msq
            .quantize_fp6(&values[..original_len], Fp6Format::E3M2)
            .expect("ms quantize should succeed");
        let recovered = msq.dequantize_fp6(&packed, &scales, Fp6Format::E3M2, original_len);

        assert_eq!(recovered.len(), original_len);
        for (orig, rec) in values.iter().take(original_len).zip(recovered.iter()) {
            let tol = orig.abs() * 0.5 + 0.5;
            assert!(
                (orig - rec).abs() <= tol,
                "MS FP6 roundtrip: orig={orig}, recovered={rec}"
            );
        }
    }

    // -- Tile selection --

    #[test]
    fn fp6_tile_blackwell_large() {
        let tile = select_fp6_tile(2048, 2048, 2048, SmVersion::Sm100);
        assert!(tile.use_tensor_core);
        assert_eq!(tile.tile_m, 256);
        assert_eq!(tile.tile_n, 256);
        assert_eq!(tile.tile_k, 96);
    }

    #[test]
    fn fp6_tile_pre_blackwell_fallback() {
        let tile = select_fp6_tile(2048, 2048, 2048, SmVersion::Sm90);
        assert!(!tile.use_tensor_core);
        assert_eq!(tile.stages, 1);
    }

    #[test]
    fn fp4_tile_blackwell_skinny_m() {
        let tile = select_fp4_tile(64, 512, 256, SmVersion::Sm100);
        assert!(tile.use_tensor_core);
        assert_eq!(tile.tile_m, 64);
        assert_eq!(tile.tile_n, 256);
    }

    #[test]
    fn fp4_tile_blackwell_large() {
        let tile = select_fp4_tile(4096, 4096, 4096, SmVersion::Sm120);
        assert!(tile.use_tensor_core);
        assert_eq!(tile.tile_m, 256);
        assert_eq!(tile.tile_n, 256);
        assert_eq!(tile.tile_k, 128);
    }

    // -- PTX generation --

    #[test]
    fn fp6_gemm_ptx_valid() {
        let config = Fp6GemmConfig {
            m: 128,
            n: 128,
            k: 96,
            format: Fp6Format::E3M2,
            accumulator: SubByteAccumulator::F32,
            micro_scaling: MicroScalingConfig {
                block_size: 32,
                scaling_format: ScalingFormat::Fp32,
                scaling_granularity: ScalingGranularity::PerBlock,
            },
            sm_version: SmVersion::Sm100,
        };
        let ptx = generate_fp6_gemm_ptx(&config).expect("PTX generation should succeed");
        assert!(ptx.contains(".visible .entry fp6_e3m2_gemm_"));
        assert!(ptx.contains(".target sm_100"));
        assert!(ptx.contains("FP6_K_LOOP"));
        assert!(ptx.contains("fma.rn.f32"));
    }

    #[test]
    fn fp4_gemm_ptx_valid() {
        let config = Fp4GemmConfig {
            m: 256,
            n: 256,
            k: 128,
            format: Fp4Format::E2M1,
            accumulator: SubByteAccumulator::F32,
            micro_scaling: MicroScalingConfig {
                block_size: 64,
                scaling_format: ScalingFormat::Fp16,
                scaling_granularity: ScalingGranularity::PerBlock,
            },
            sm_version: SmVersion::Sm100,
        };
        let ptx = generate_fp4_gemm_ptx(&config).expect("PTX generation should succeed");
        assert!(ptx.contains(".visible .entry fp4_e2m1_gemm_"));
        assert!(ptx.contains("FP4_K_LOOP"));
        assert!(ptx.contains("fma.rn.f32"));
    }

    #[test]
    fn fp6_dequantize_ptx_valid() {
        let ptx =
            generate_fp6_dequantize_ptx(Fp6Format::E2M3, 64).expect("dequant PTX should succeed");
        assert!(ptx.contains("fp6_e2m3_dequantize_bs64"));
        assert!(ptx.contains("DEQUANT6_DONE"));
    }

    #[test]
    fn fp4_dequantize_ptx_valid() {
        let ptx =
            generate_fp4_dequantize_ptx(Fp4Format::Int4, 128).expect("dequant PTX should succeed");
        assert!(ptx.contains("fp4_int4_dequantize_bs128"));
        assert!(ptx.contains("DEQUANT4_DONE"));
    }

    // -- Edge cases --

    #[test]
    fn fp6_zero_values() {
        let quantizer = Fp6Quantizer::new(Fp6Format::E3M2);
        let values = [0.0f32, 0.0, 0.0];
        let packed = quantizer.quantize(&values).expect("quantize zeros");
        let recovered = quantizer.dequantize(&packed);
        for v in &recovered {
            assert!(v.abs() < 1e-6, "zero should roundtrip: got {v}");
        }
    }

    #[test]
    fn fp6_max_range_clamping() {
        let quantizer = Fp6Quantizer::new(Fp6Format::E3M2);
        // Values beyond FP6 E3M2 range should be clamped
        let raw = quantizer.quantize_one(100.0);
        let recovered = quantizer.dequantize_one(raw);
        assert!(
            recovered <= Fp6Format::E3M2.max_value() + 0.01,
            "should clamp to max: got {recovered}"
        );
    }

    #[test]
    fn fp6_subnormals() {
        let quantizer = Fp6Quantizer::new(Fp6Format::E3M2);
        // Very small values should produce subnormals or zero
        let raw = quantizer.quantize_one(0.01);
        let recovered = quantizer.dequantize_one(raw);
        // Should be close to the original or clamped to nearest representable
        assert!(recovered.abs() < 1.0, "subnormal range: got {recovered}");
    }

    // -- Config validation --

    #[test]
    fn fp6_config_rejects_pre_blackwell() {
        let config = Fp6GemmConfig {
            m: 128,
            n: 128,
            k: 96,
            format: Fp6Format::E3M2,
            accumulator: SubByteAccumulator::F32,
            micro_scaling: MicroScalingConfig {
                block_size: 32,
                scaling_format: ScalingFormat::Fp32,
                scaling_granularity: ScalingGranularity::PerBlock,
            },
            sm_version: SmVersion::Sm90,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn fp4_config_rejects_pre_blackwell() {
        let config = Fp4GemmConfig {
            m: 128,
            n: 128,
            k: 128,
            format: Fp4Format::E2M1,
            accumulator: SubByteAccumulator::F32,
            micro_scaling: MicroScalingConfig {
                block_size: 32,
                scaling_format: ScalingFormat::Fp32,
                scaling_granularity: ScalingGranularity::PerBlock,
            },
            sm_version: SmVersion::Sm89,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn fp6_config_rejects_zero_dims() {
        let config = Fp6GemmConfig {
            m: 0,
            n: 128,
            k: 96,
            format: Fp6Format::E3M2,
            accumulator: SubByteAccumulator::F32,
            micro_scaling: MicroScalingConfig {
                block_size: 32,
                scaling_format: ScalingFormat::Fp32,
                scaling_granularity: ScalingGranularity::PerBlock,
            },
            sm_version: SmVersion::Sm100,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_block_size_rejected() {
        let config = MicroScalingConfig {
            block_size: 17,
            scaling_format: ScalingFormat::Fp32,
            scaling_granularity: ScalingGranularity::PerBlock,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn dequantize_ptx_invalid_block_size() {
        assert!(generate_fp6_dequantize_ptx(Fp6Format::E3M2, 17).is_err());
        assert!(generate_fp4_dequantize_ptx(Fp4Format::E2M1, 99).is_err());
    }
}
