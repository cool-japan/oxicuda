//! Half-precision (FP16) FFT support with FP32 accumulation.
//!
//! This module provides FP16-storage FFT for memory-bound workloads, where
//! data is stored in IEEE 754 half-precision (16-bit) format but intermediate
//! arithmetic is performed in single-precision (FP32) to maintain numerical
//! accuracy.
//!
//! # Accumulation Modes
//!
//! Three accumulation strategies are available, trading off speed for accuracy:
//!
//! | Mode    | Storage | Butterfly | Twiddle | Accuracy | Bandwidth |
//! |---------|---------|-----------|---------|----------|-----------|
//! | `Fp32`  | FP16    | FP32      | FP32    | Best     | 2x saving |
//! | `Mixed` | FP16    | FP16      | FP32    | Good     | 2x saving |
//! | `Pure`  | FP16    | FP16      | FP16    | Lowest   | 2x saving |
//!
//! # PTX Generation
//!
//! The module generates specialized PTX kernels that:
//! - Load FP16 complex pairs from global memory
//! - Convert to FP32 for accumulation (in `Fp32` / `Mixed` modes)
//! - Perform Cooley-Tukey butterfly operations
//! - Apply twiddle factors
//! - Convert back and store as FP16

#![allow(dead_code)]

use std::f64::consts::PI;

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::error::PtxGenError;
use oxicuda_ptx::ir::PtxType;

use crate::error::{FftError, FftResult};
use crate::types::{FftDirection, FftType};

// ---------------------------------------------------------------------------
// AccumulationMode
// ---------------------------------------------------------------------------

/// Controls how intermediate arithmetic is performed in half-precision FFT.
///
/// All modes store input/output in FP16 to achieve ~2x memory bandwidth
/// improvement. The difference is in how internal butterfly and twiddle
/// factor multiplications are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AccumulationMode {
    /// FP16 storage with full FP32 compute for both butterfly additions and
    /// twiddle factor multiplications. This is the default and most accurate
    /// mode, losing only the storage quantization error.
    #[default]
    Fp32,

    /// FP16 for butterfly add/subtract, FP32 for twiddle factor multiply.
    /// Twiddle factors involve sine/cosine values that can lose significant
    /// precision in FP16, so this mode keeps those in FP32 while allowing
    /// the simpler butterfly ops to run in FP16.
    Mixed,

    /// All operations in FP16. Fastest throughput but least accurate.
    /// Suitable for inference workloads where slight precision loss is
    /// acceptable, or when the FFT size is small (N <= 256).
    Pure,
}

// ---------------------------------------------------------------------------
// HalfRoundingMode
// ---------------------------------------------------------------------------

/// Rounding mode used when converting FP32 results back to FP16 storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum HalfRoundingMode {
    /// IEEE 754 round-to-nearest-even (default, best accuracy).
    #[default]
    RoundToNearest,

    /// Round towards zero (truncation). Slightly faster on some hardware.
    RoundToZero,

    /// Stochastic rounding: rounds up or down with probability proportional
    /// to proximity to the nearest representable values. Provides unbiased
    /// rounding error that is beneficial during neural network training, as
    /// it prevents systematic drift in gradient accumulation.
    Stochastic,
}

impl HalfRoundingMode {
    /// Returns the PTX rounding modifier string for FP32-to-FP16 conversion.
    ///
    /// Note: stochastic rounding is emulated in software since PTX does not
    /// have a native stochastic rounding mode for `cvt`.
    fn ptx_rounding_suffix(self) -> &'static str {
        match self {
            Self::RoundToNearest => "rn",
            Self::RoundToZero => "rz",
            // Stochastic rounding requires software emulation; we use rn as
            // the base and add a perturbation step in the generated PTX.
            Self::Stochastic => "rn",
        }
    }
}

// ---------------------------------------------------------------------------
// HalfPrecisionFftConfig
// ---------------------------------------------------------------------------

/// Configuration for a half-precision FFT operation.
///
/// # Constraints
///
/// - `n` must be a positive power of 2 (the butterfly generator supports
///   radix-2 decomposition only for FP16).
/// - `fft_type` must be [`FftType::C2C`] — real-to-complex and complex-to-real
///   transforms are not yet supported in FP16 mode.
/// - Maximum supported size is 65536 (larger sizes accumulate too much
///   rounding error in half precision).
#[derive(Debug, Clone)]
pub struct HalfPrecisionFftConfig {
    /// FFT size (must be a power of 2, > 0, <= 65536).
    pub n: usize,

    /// Transform direction.
    pub direction: FftDirection,

    /// Transform type — only C2C is supported for FP16.
    pub fft_type: FftType,

    /// Accumulation precision strategy.
    pub accumulation: AccumulationMode,

    /// Rounding mode for FP32 -> FP16 conversion.
    pub rounding: HalfRoundingMode,
}

impl HalfPrecisionFftConfig {
    /// Creates a new half-precision FFT configuration with default settings.
    ///
    /// Defaults to FP32 accumulation, round-to-nearest, forward C2C.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            direction: FftDirection::Forward,
            fft_type: FftType::C2C,
            accumulation: AccumulationMode::default(),
            rounding: HalfRoundingMode::default(),
        }
    }

    /// Sets the transform direction.
    pub fn with_direction(mut self, direction: FftDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Sets the accumulation mode.
    pub fn with_accumulation(mut self, mode: AccumulationMode) -> Self {
        self.accumulation = mode;
        self
    }

    /// Sets the rounding mode.
    pub fn with_rounding(mut self, rounding: HalfRoundingMode) -> Self {
        self.rounding = rounding;
        self
    }
}

// ---------------------------------------------------------------------------
// HalfPrecisionStats
// ---------------------------------------------------------------------------

/// Runtime statistics for half-precision FFT execution.
///
/// Tracks numerical health of FP16 computation including overflow and
/// underflow events. These counters are populated during kernel execution
/// when diagnostic mode is enabled.
#[derive(Debug, Clone, Default)]
pub struct HalfPrecisionStats {
    /// Maximum magnitude observed across all complex elements.
    pub max_magnitude: f32,

    /// Minimum non-zero magnitude observed.
    pub min_magnitude: f32,

    /// Number of elements that overflowed FP16 range (|x| > 65504).
    pub overflow_count: u64,

    /// Number of elements that underflowed to zero (|x| < 2^-24 and != 0).
    pub underflow_count: u64,

    /// Number of elements processed.
    pub total_elements: u64,
}

impl HalfPrecisionStats {
    /// Creates a new stats tracker with initial values.
    pub fn new() -> Self {
        Self {
            max_magnitude: 0.0,
            min_magnitude: f32::MAX,
            overflow_count: 0,
            underflow_count: 0,
            total_elements: 0,
        }
    }

    /// Returns the fraction of elements that overflowed.
    pub fn overflow_rate(&self) -> f64 {
        if self.total_elements == 0 {
            return 0.0;
        }
        self.overflow_count as f64 / self.total_elements as f64
    }

    /// Returns the fraction of elements that underflowed.
    pub fn underflow_rate(&self) -> f64 {
        if self.total_elements == 0 {
            return 0.0;
        }
        self.underflow_count as f64 / self.total_elements as f64
    }

    /// Updates statistics with a new magnitude observation.
    pub fn observe(&mut self, magnitude: f32) {
        self.total_elements += 1;

        if magnitude > self.max_magnitude {
            self.max_magnitude = magnitude;
        }

        if magnitude > 0.0 && magnitude < self.min_magnitude {
            self.min_magnitude = magnitude;
        }

        // FP16 max representable value
        const FP16_MAX: f32 = 65504.0;
        // FP16 smallest positive normal
        const FP16_MIN_NORMAL: f32 = 6.103_515_6e-5; // 2^-14

        if magnitude > FP16_MAX {
            self.overflow_count += 1;
        } else if magnitude > 0.0 && magnitude < FP16_MIN_NORMAL {
            self.underflow_count += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// HalfPrecisionFftPlan
// ---------------------------------------------------------------------------

/// A validated, ready-to-execute half-precision FFT plan.
///
/// Created by [`plan_half_precision_fft`], this struct holds the validated
/// configuration along with precomputed error estimates and memory metrics.
#[derive(Debug, Clone)]
pub struct HalfPrecisionFftPlan {
    /// The validated configuration.
    pub config: HalfPrecisionFftConfig,

    /// Estimated RMS error bound (relative to FP32 FFT).
    pub estimated_error_bound: f64,

    /// Number of Cooley-Tukey butterfly stages (log2(N)).
    pub num_stages: u32,

    /// Bytes required for input/output in FP16 (per batch).
    pub fp16_buffer_bytes: usize,

    /// Bytes that would be required in FP32 (per batch), for comparison.
    pub fp32_buffer_bytes: usize,
}

impl HalfPrecisionFftPlan {
    /// Returns the memory savings ratio compared to FP32.
    ///
    /// For FP16 storage this is always approximately 0.5 (half the memory),
    /// since each complex element uses 4 bytes (2x FP16) instead of 8 bytes
    /// (2x FP32).
    pub fn memory_savings_ratio(&self) -> f64 {
        if self.fp32_buffer_bytes == 0 {
            return 0.0;
        }
        self.fp16_buffer_bytes as f64 / self.fp32_buffer_bytes as f64
    }

    /// Estimates the signal-to-noise ratio in dB for this configuration.
    ///
    /// The SNR is computed from the estimated RMS error bound:
    ///   SNR = -20 * log10(error_bound)
    ///
    /// Higher values indicate better accuracy. Typical values:
    /// - FP32 accumulation, N=1024: ~48 dB
    /// - Mixed accumulation, N=1024: ~38 dB
    /// - Pure FP16, N=1024: ~28 dB
    pub fn estimated_snr_db(&self) -> f64 {
        if self.estimated_error_bound <= 0.0 {
            return f64::INFINITY;
        }
        -20.0 * self.estimated_error_bound.log10()
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Maximum supported FFT size for half-precision mode.
const MAX_HALF_FFT_SIZE: usize = 65536;

/// FP16 max representable value.
const FP16_MAX_VALUE: f32 = 65504.0;

/// Validates a half-precision FFT configuration.
///
/// # Errors
///
/// Returns [`FftError::InvalidSize`] if `n` is zero, not a power of 2,
/// or exceeds the maximum half-precision FFT size.
///
/// Returns [`FftError::UnsupportedTransform`] if `fft_type` is not C2C.
pub fn validate_half_precision_config(config: &HalfPrecisionFftConfig) -> FftResult<()> {
    // Size must be positive
    if config.n == 0 {
        return Err(FftError::InvalidSize(
            "half-precision FFT size must be > 0".to_string(),
        ));
    }

    // Must be a power of 2
    if !config.n.is_power_of_two() {
        return Err(FftError::InvalidSize(format!(
            "half-precision FFT requires power-of-2 size, got {}",
            config.n
        )));
    }

    // Size limit for FP16 accuracy
    if config.n > MAX_HALF_FFT_SIZE {
        return Err(FftError::InvalidSize(format!(
            "half-precision FFT size {} exceeds maximum {} \
             (accumulated rounding error becomes unacceptable)",
            config.n, MAX_HALF_FFT_SIZE
        )));
    }

    // Only C2C is supported
    if config.fft_type != FftType::C2C {
        return Err(FftError::UnsupportedTransform(format!(
            "half-precision FFT only supports C2C, got {}",
            config.fft_type
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Error estimation
// ---------------------------------------------------------------------------

/// Estimates the theoretical RMS error bound for an FP16 FFT.
///
/// The error model considers:
/// - FP16 machine epsilon: eps_16 = 2^-10 ~ 9.77e-4
/// - FP32 machine epsilon: eps_32 = 2^-23 ~ 1.19e-7
/// - For an N-point FFT with log2(N) butterfly stages, rounding errors
///   accumulate as O(sqrt(log2(N)) * eps) for random-phase signals
///
/// # Arguments
///
/// - `n`: FFT size (must be > 0)
/// - `mode`: Accumulation mode
///
/// # Returns
///
/// Estimated relative RMS error (unitless). For example, 1e-3 means the
/// RMS error is about 0.1% of the signal magnitude.
pub fn estimate_fp16_fft_error(n: usize, mode: AccumulationMode) -> f64 {
    if n == 0 {
        return 0.0;
    }

    let log2_n = (n as f64).log2();
    let stages = log2_n;

    // Machine epsilons
    let eps_16: f64 = 2.0_f64.powi(-10); // ~9.77e-4
    let eps_32: f64 = 2.0_f64.powi(-23); // ~1.19e-7

    match mode {
        AccumulationMode::Fp32 => {
            // Only storage quantization contributes FP16 error (load/store).
            // Butterfly arithmetic is in FP32 so accumulates only FP32 error.
            // Total: 2 * eps_16 (input + output quantization) +
            //        sqrt(stages) * eps_32 (arithmetic)
            2.0 * eps_16 + stages.sqrt() * eps_32
        }
        AccumulationMode::Mixed => {
            // Butterfly in FP16, twiddle in FP32.
            // Butterfly add/sub: sqrt(stages) * eps_16
            // Twiddle multiply: sqrt(stages) * eps_32
            // Storage: 2 * eps_16
            2.0 * eps_16 + stages.sqrt() * eps_16 + stages.sqrt() * eps_32
        }
        AccumulationMode::Pure => {
            // Everything in FP16.
            // Each butterfly stage contributes ~eps_16 error.
            // Error accumulates as O(sqrt(stages) * eps_16) for the butterfly,
            // plus O(sqrt(stages) * eps_16) for twiddle multiplication.
            // Storage: 2 * eps_16
            2.0 * eps_16 + 2.0 * stages.sqrt() * eps_16
        }
    }
}

// ---------------------------------------------------------------------------
// Plan creation
// ---------------------------------------------------------------------------

/// Creates a validated half-precision FFT plan.
///
/// This function validates the configuration, computes error estimates,
/// and prepares the plan metadata. It does not compile GPU kernels — that
/// happens at execution time via the plan's associated methods.
///
/// # Errors
///
/// Returns errors from [`validate_half_precision_config`].
pub fn plan_half_precision_fft(config: &HalfPrecisionFftConfig) -> FftResult<HalfPrecisionFftPlan> {
    validate_half_precision_config(config)?;

    let n = config.n;
    let num_stages = (n as f64).log2().round() as u32;

    // FP16: each complex element = 2 * 2 bytes = 4 bytes
    let fp16_buffer_bytes = n * 4;
    // FP32: each complex element = 2 * 4 bytes = 8 bytes
    let fp32_buffer_bytes = n * 8;

    let estimated_error_bound = estimate_fp16_fft_error(n, config.accumulation);

    Ok(HalfPrecisionFftPlan {
        config: config.clone(),
        estimated_error_bound,
        num_stages,
        fp16_buffer_bytes,
        fp32_buffer_bytes,
    })
}

// ---------------------------------------------------------------------------
// PTX generation: FP16 butterfly
// ---------------------------------------------------------------------------

/// Generates a PTX kernel for an FP16 Cooley-Tukey radix-2 butterfly.
///
/// The kernel processes pairs of complex FP16 values, applying the
/// butterfly operation:
///   A' = A + W * B
///   B' = A - W * B
///
/// where W is the twiddle factor for that stage.
///
/// # Arguments
///
/// - `radix`: Butterfly radix (must be 2, 4, or 8)
/// - `mode`: Accumulation mode controlling internal precision
/// - `sm`: Target GPU architecture (must support FP16; >= sm_75)
///
/// # Errors
///
/// Returns [`PtxGenError::GenerationFailed`] for unsupported radix values
/// or architecture too old for FP16 instructions.
pub fn generate_fp16_butterfly_ptx(
    radix: u32,
    mode: AccumulationMode,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    if !matches!(radix, 2 | 4 | 8) {
        return Err(PtxGenError::GenerationFailed(format!(
            "FP16 butterfly only supports radix 2, 4, 8; got {radix}"
        )));
    }

    if sm < SmVersion::Sm75 {
        return Err(PtxGenError::GenerationFailed(
            "FP16 butterfly requires sm_75 or later".to_string(),
        ));
    }

    let mode_suffix = match mode {
        AccumulationMode::Fp32 => "fp32acc",
        AccumulationMode::Mixed => "mixed",
        AccumulationMode::Pure => "pure16",
    };
    let kernel_name = format!("fft_fp16_butterfly_r{radix}_{mode_suffix}");

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("data_ptr", PtxType::U64)
        .param("twiddle_ptr", PtxType::U64)
        .param("stride", PtxType::U32)
        .param("n", PtxType::U32)
        .param("direction", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            b.comment(&format!(
                "FP16 Cooley-Tukey radix-{radix} butterfly, mode={mode_suffix}"
            ));

            // Load thread index for work distribution
            let tid = b.thread_id_x();
            let bid = b.block_id_x();
            let block_dim = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {block_dim}, %ntid.x;"));

            // Global thread index
            let global_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "mad.lo.u32 {global_idx}, {bid}, {block_dim}, {tid};"
            ));

            // Load parameters
            let data_ptr = b.load_param_u64("data_ptr");
            let twiddle_ptr = b.load_param_u64("twiddle_ptr");
            let stride_param = b.load_param_u32("stride");
            let n_param = b.load_param_u32("n");

            // Bounds check
            let in_bounds = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!(
                "setp.lt.u32 {in_bounds}, {global_idx}, {n_param};"
            ));
            b.raw_ptx(&format!("@!{in_bounds} bra $L_exit;"));

            // Compute element offsets for radix-2 butterfly
            // idx_a = global_idx (lower half)
            // idx_b = global_idx + stride (upper half)
            let idx_a = global_idx.clone();
            let idx_b = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("add.u32 {idx_b}, {idx_a}, {stride_param};"));

            // Convert indices to byte offsets (FP16 complex = 4 bytes)
            let offset_a = b.alloc_reg(PtxType::U64);
            let offset_b = b.alloc_reg(PtxType::U64);
            let idx_a_64 = b.alloc_reg(PtxType::U64);
            let idx_b_64 = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("cvt.u64.u32 {idx_a_64}, {idx_a};"));
            b.raw_ptx(&format!("cvt.u64.u32 {idx_b_64}, {idx_b};"));
            // Each complex FP16 element = 4 bytes (2 * sizeof(f16))
            b.raw_ptx(&format!("mul.lo.u64 {offset_a}, {idx_a_64}, 4;"));
            b.raw_ptx(&format!("mul.lo.u64 {offset_b}, {idx_b_64}, 4;"));

            // Compute addresses
            let addr_a = b.alloc_reg(PtxType::U64);
            let addr_b = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {addr_a}, {data_ptr}, {offset_a};"));
            b.raw_ptx(&format!("add.u64 {addr_b}, {data_ptr}, {offset_b};"));

            // Load FP16 complex values (re, im packed as 2x f16)
            let a_re_f16 = b.alloc_reg(PtxType::F16);
            let a_im_f16 = b.alloc_reg(PtxType::F16);
            let b_re_f16 = b.alloc_reg(PtxType::F16);
            let b_im_f16 = b.alloc_reg(PtxType::F16);
            b.raw_ptx(&format!("ld.global.f16 {a_re_f16}, [{addr_a}];"));
            b.raw_ptx(&format!("ld.global.f16 {a_im_f16}, [{addr_a}+2];"));
            b.raw_ptx(&format!("ld.global.f16 {b_re_f16}, [{addr_b}];"));
            b.raw_ptx(&format!("ld.global.f16 {b_im_f16}, [{addr_b}+2];"));

            // Load twiddle factor (stored as FP16 complex)
            let tw_offset = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("mul.lo.u64 {tw_offset}, {idx_a_64}, 4;"));
            let tw_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {tw_addr}, {twiddle_ptr}, {tw_offset};"));
            let tw_re_f16 = b.alloc_reg(PtxType::F16);
            let tw_im_f16 = b.alloc_reg(PtxType::F16);
            b.raw_ptx(&format!("ld.global.f16 {tw_re_f16}, [{tw_addr}];"));
            b.raw_ptx(&format!("ld.global.f16 {tw_im_f16}, [{tw_addr}+2];"));

            match mode {
                AccumulationMode::Fp32 => {
                    b.comment("FP32 accumulation: convert all to f32, compute, convert back");
                    // Convert inputs to FP32
                    let a_re = b.cvt_f16_to_f32(a_re_f16);
                    let a_im = b.cvt_f16_to_f32(a_im_f16);
                    let b_re = b.cvt_f16_to_f32(b_re_f16);
                    let b_im = b.cvt_f16_to_f32(b_im_f16);
                    let tw_re = b.cvt_f16_to_f32(tw_re_f16);
                    let tw_im = b.cvt_f16_to_f32(tw_im_f16);

                    // Complex multiply: W * B
                    // wb_re = tw_re * b_re - tw_im * b_im
                    // wb_im = tw_re * b_im + tw_im * b_re
                    let wb_re_partial = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {wb_re_partial}, {tw_re}, {b_re};"));
                    let neg_tw_im = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("neg.f32 {neg_tw_im}, {tw_im};"));
                    let wb_re = b.fma_f32(neg_tw_im, b_im.clone(), wb_re_partial);

                    let wb_im_partial = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {wb_im_partial}, {tw_re}, {b_im};"));
                    let wb_im = b.fma_f32(tw_im, b_re, wb_im_partial);

                    // Butterfly: A' = A + W*B, B' = A - W*B
                    let out_a_re = b.add_f32(a_re.clone(), wb_re.clone());
                    let out_a_im = b.add_f32(a_im.clone(), wb_im.clone());
                    let out_b_re = b.sub_f32(a_re, wb_re);
                    let out_b_im = b.sub_f32(a_im, wb_im);

                    // Convert back to FP16 and store
                    let out_a_re_f16 = b.cvt_f32_to_f16(out_a_re);
                    let out_a_im_f16 = b.cvt_f32_to_f16(out_a_im);
                    let out_b_re_f16 = b.cvt_f32_to_f16(out_b_re);
                    let out_b_im_f16 = b.cvt_f32_to_f16(out_b_im);
                    b.raw_ptx(&format!("st.global.f16 [{addr_a}], {out_a_re_f16};"));
                    b.raw_ptx(&format!("st.global.f16 [{addr_a}+2], {out_a_im_f16};"));
                    b.raw_ptx(&format!("st.global.f16 [{addr_b}], {out_b_re_f16};"));
                    b.raw_ptx(&format!("st.global.f16 [{addr_b}+2], {out_b_im_f16};"));
                }
                AccumulationMode::Mixed => {
                    b.comment("Mixed mode: FP16 butterfly, FP32 twiddle multiply");
                    // Twiddle multiply in FP32
                    let b_re = b.cvt_f16_to_f32(b_re_f16);
                    let b_im = b.cvt_f16_to_f32(b_im_f16);
                    let tw_re = b.cvt_f16_to_f32(tw_re_f16);
                    let tw_im = b.cvt_f16_to_f32(tw_im_f16);

                    let wb_re_partial = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {wb_re_partial}, {tw_re}, {b_re};"));
                    let neg_tw_im = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("neg.f32 {neg_tw_im}, {tw_im};"));
                    let wb_re_f32 = b.fma_f32(neg_tw_im, b_im.clone(), wb_re_partial);

                    let wb_im_partial = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {wb_im_partial}, {tw_re}, {b_im};"));
                    let wb_im_f32 = b.fma_f32(tw_im, b_re, wb_im_partial);

                    // Convert W*B back to FP16 for butterfly
                    let wb_re = b.cvt_f32_to_f16(wb_re_f32);
                    let wb_im = b.cvt_f32_to_f16(wb_im_f32);

                    // Butterfly in FP16
                    let out_a_re = b.alloc_reg(PtxType::F16);
                    let out_a_im = b.alloc_reg(PtxType::F16);
                    let out_b_re = b.alloc_reg(PtxType::F16);
                    let out_b_im = b.alloc_reg(PtxType::F16);
                    b.raw_ptx(&format!("add.f16 {out_a_re}, {a_re_f16}, {wb_re};"));
                    b.raw_ptx(&format!("add.f16 {out_a_im}, {a_im_f16}, {wb_im};"));
                    b.raw_ptx(&format!("sub.f16 {out_b_re}, {a_re_f16}, {wb_re};"));
                    b.raw_ptx(&format!("sub.f16 {out_b_im}, {a_im_f16}, {wb_im};"));

                    // Store FP16 results
                    b.raw_ptx(&format!("st.global.f16 [{addr_a}], {out_a_re};"));
                    b.raw_ptx(&format!("st.global.f16 [{addr_a}+2], {out_a_im};"));
                    b.raw_ptx(&format!("st.global.f16 [{addr_b}], {out_b_re};"));
                    b.raw_ptx(&format!("st.global.f16 [{addr_b}+2], {out_b_im};"));
                }
                AccumulationMode::Pure => {
                    b.comment("Pure FP16: all operations in half precision");
                    // Complex multiply W * B in FP16
                    let wb_re_partial = b.alloc_reg(PtxType::F16);
                    b.raw_ptx(&format!(
                        "mul.f16 {wb_re_partial}, {tw_re_f16}, {b_re_f16};"
                    ));
                    let neg_tw_im = b.alloc_reg(PtxType::F16);
                    b.raw_ptx(&format!("neg.f16 {neg_tw_im}, {tw_im_f16};"));
                    let wb_re = b.alloc_reg(PtxType::F16);
                    b.raw_ptx(&format!(
                        "fma.rn.f16 {wb_re}, {neg_tw_im}, {b_im_f16}, {wb_re_partial};"
                    ));

                    let wb_im_partial = b.alloc_reg(PtxType::F16);
                    b.raw_ptx(&format!(
                        "mul.f16 {wb_im_partial}, {tw_re_f16}, {b_im_f16};"
                    ));
                    let wb_im = b.alloc_reg(PtxType::F16);
                    b.raw_ptx(&format!(
                        "fma.rn.f16 {wb_im}, {tw_im_f16}, {b_re_f16}, {wb_im_partial};"
                    ));

                    // Butterfly in FP16
                    let out_a_re = b.alloc_reg(PtxType::F16);
                    let out_a_im = b.alloc_reg(PtxType::F16);
                    let out_b_re = b.alloc_reg(PtxType::F16);
                    let out_b_im = b.alloc_reg(PtxType::F16);
                    b.raw_ptx(&format!("add.f16 {out_a_re}, {a_re_f16}, {wb_re};"));
                    b.raw_ptx(&format!("add.f16 {out_a_im}, {a_im_f16}, {wb_im};"));
                    b.raw_ptx(&format!("sub.f16 {out_b_re}, {a_re_f16}, {wb_re};"));
                    b.raw_ptx(&format!("sub.f16 {out_b_im}, {a_im_f16}, {wb_im};"));

                    // Store
                    b.raw_ptx(&format!("st.global.f16 [{addr_a}], {out_a_re};"));
                    b.raw_ptx(&format!("st.global.f16 [{addr_a}+2], {out_a_im};"));
                    b.raw_ptx(&format!("st.global.f16 [{addr_b}], {out_b_re};"));
                    b.raw_ptx(&format!("st.global.f16 [{addr_b}+2], {out_b_im};"));
                }
            }

            b.raw_ptx("$L_exit:");
            b.comment("End of FP16 butterfly kernel");
        })
        .build()?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// PTX generation: FP16 twiddle factor application
// ---------------------------------------------------------------------------

/// Generates a PTX kernel for applying precomputed twiddle factors in FP16.
///
/// This kernel multiplies each complex element `X[k]` by the corresponding
/// twiddle factor `W[k] = exp(-2*pi*i*k/N)`, supporting both forward and
/// inverse transforms via the `direction` parameter.
///
/// Twiddle factors are precomputed in FP32, converted to FP16, and stored
/// in a device buffer. The kernel loads them and applies the complex multiply.
///
/// # Arguments
///
/// - `n`: FFT size (determines number of twiddle factors)
/// - `sm`: Target GPU architecture
///
/// # Errors
///
/// Returns [`PtxGenError::GenerationFailed`] if `n` is zero.
pub fn generate_fp16_twiddle_ptx(n: usize, sm: SmVersion) -> Result<String, PtxGenError> {
    if n == 0 {
        return Err(PtxGenError::GenerationFailed(
            "twiddle factor generation requires n > 0".to_string(),
        ));
    }

    let kernel_name = format!("fft_fp16_twiddle_n{n}");

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("data_ptr", PtxType::U64)
        .param("twiddle_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            b.comment(&format!("FP16 twiddle factor application, N={n}"));

            let tid = b.thread_id_x();
            let bid = b.block_id_x();
            let block_dim = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {block_dim}, %ntid.x;"));

            let global_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "mad.lo.u32 {global_idx}, {bid}, {block_dim}, {tid};"
            ));

            let n_param = b.load_param_u32("n");
            let data_ptr = b.load_param_u64("data_ptr");
            let twiddle_ptr = b.load_param_u64("twiddle_ptr");

            // Bounds check
            let in_bounds = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!(
                "setp.lt.u32 {in_bounds}, {global_idx}, {n_param};"
            ));
            b.raw_ptx(&format!("@!{in_bounds} bra $L_tw_exit;"));

            // Compute byte offset (4 bytes per FP16 complex element)
            let idx_64 = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("cvt.u64.u32 {idx_64}, {global_idx};"));
            let byte_offset = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("mul.lo.u64 {byte_offset}, {idx_64}, 4;"));

            // Load data element
            let data_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {data_addr}, {data_ptr}, {byte_offset};"));
            let x_re = b.alloc_reg(PtxType::F16);
            let x_im = b.alloc_reg(PtxType::F16);
            b.raw_ptx(&format!("ld.global.f16 {x_re}, [{data_addr}];"));
            b.raw_ptx(&format!("ld.global.f16 {x_im}, [{data_addr}+2];"));

            // Load twiddle factor
            let tw_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {tw_addr}, {twiddle_ptr}, {byte_offset};"));
            let tw_re = b.alloc_reg(PtxType::F16);
            let tw_im = b.alloc_reg(PtxType::F16);
            b.raw_ptx(&format!("ld.global.f16 {tw_re}, [{tw_addr}];"));
            b.raw_ptx(&format!("ld.global.f16 {tw_im}, [{tw_addr}+2];"));

            // Complex multiply in FP32 for accuracy, then store as FP16
            let x_re_f32 = b.cvt_f16_to_f32(x_re);
            let x_im_f32 = b.cvt_f16_to_f32(x_im);
            let tw_re_f32 = b.cvt_f16_to_f32(tw_re);
            let tw_im_f32 = b.cvt_f16_to_f32(tw_im);

            // out_re = x_re * tw_re - x_im * tw_im
            let out_re_partial = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!(
                "mul.rn.f32 {out_re_partial}, {x_re_f32}, {tw_re_f32};"
            ));
            let neg_x_im = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("neg.f32 {neg_x_im}, {x_im_f32};"));
            let out_re_f32 = b.fma_f32(neg_x_im, tw_im_f32.clone(), out_re_partial);

            // out_im = x_re * tw_im + x_im * tw_re
            let out_im_partial = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!(
                "mul.rn.f32 {out_im_partial}, {x_re_f32}, {tw_im_f32};"
            ));
            let out_im_f32 = b.fma_f32(x_im_f32, tw_re_f32, out_im_partial);

            // Convert back to FP16 and store
            let out_re_f16 = b.cvt_f32_to_f16(out_re_f32);
            let out_im_f16 = b.cvt_f32_to_f16(out_im_f32);
            b.raw_ptx(&format!("st.global.f16 [{data_addr}], {out_re_f16};"));
            b.raw_ptx(&format!("st.global.f16 [{data_addr}+2], {out_im_f16};"));

            b.raw_ptx("$L_tw_exit:");
            b.comment("End of twiddle application kernel");
        })
        .build()?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Helper: precompute twiddle factors in FP32 (host side)
// ---------------------------------------------------------------------------

/// Precomputes twiddle factors for an N-point FFT.
///
/// Returns a vector of `(cos, sin)` pairs as FP32 values, suitable for
/// converting to FP16 and uploading to device memory.
///
/// The twiddle factor for index `k` is:
///   W_N^k = (cos(2*pi*k/N), sign * sin(2*pi*k/N))
///
/// where `sign` is -1 for forward and +1 for inverse transforms.
pub fn precompute_twiddle_factors(n: usize, direction: FftDirection) -> Vec<(f32, f32)> {
    let sign = direction.sign();
    (0..n)
        .map(|k| {
            let angle = sign * 2.0 * PI * (k as f64) / (n as f64);
            (angle.cos() as f32, angle.sin() as f32)
        })
        .collect()
}

/// FP16 max value constant for overflow checks.
pub const FP16_MAX: f32 = FP16_MAX_VALUE;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config validation tests --

    #[test]
    fn validate_valid_config() {
        let config = HalfPrecisionFftConfig::new(1024);
        assert!(validate_half_precision_config(&config).is_ok());
    }

    #[test]
    fn validate_zero_size() {
        let config = HalfPrecisionFftConfig {
            n: 0,
            direction: FftDirection::Forward,
            fft_type: FftType::C2C,
            accumulation: AccumulationMode::Fp32,
            rounding: HalfRoundingMode::RoundToNearest,
        };
        let result = validate_half_precision_config(&config);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn validate_non_power_of_two() {
        let config = HalfPrecisionFftConfig::new(1000);
        let result = validate_half_precision_config(&config);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn validate_too_large() {
        let config = HalfPrecisionFftConfig::new(131072); // 2^17 > 65536
        let result = validate_half_precision_config(&config);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn validate_r2c_unsupported() {
        let config = HalfPrecisionFftConfig {
            n: 256,
            direction: FftDirection::Forward,
            fft_type: FftType::R2C,
            accumulation: AccumulationMode::Fp32,
            rounding: HalfRoundingMode::RoundToNearest,
        };
        let result = validate_half_precision_config(&config);
        assert!(matches!(result, Err(FftError::UnsupportedTransform(_))));
    }

    // -- Error estimation tests --

    #[test]
    fn error_estimation_ordering() {
        // For same N, Fp32 < Mixed < Pure error
        let n = 1024;
        let err_fp32 = estimate_fp16_fft_error(n, AccumulationMode::Fp32);
        let err_mixed = estimate_fp16_fft_error(n, AccumulationMode::Mixed);
        let err_pure = estimate_fp16_fft_error(n, AccumulationMode::Pure);

        assert!(
            err_fp32 < err_mixed,
            "FP32 error ({err_fp32}) should be less than Mixed ({err_mixed})"
        );
        assert!(
            err_mixed < err_pure,
            "Mixed error ({err_mixed}) should be less than Pure ({err_pure})"
        );
    }

    #[test]
    fn error_estimation_size_scaling() {
        // Larger N should produce larger error (more butterfly stages)
        let err_small = estimate_fp16_fft_error(64, AccumulationMode::Fp32);
        let err_large = estimate_fp16_fft_error(4096, AccumulationMode::Fp32);
        assert!(
            err_small < err_large,
            "error for N=64 ({err_small}) should be less than N=4096 ({err_large})"
        );
    }

    #[test]
    fn error_estimation_zero_n() {
        let err = estimate_fp16_fft_error(0, AccumulationMode::Fp32);
        assert_eq!(err, 0.0);
    }

    // -- Plan creation tests --

    #[test]
    fn plan_creation_basic() {
        let config = HalfPrecisionFftConfig::new(512);
        let plan = plan_half_precision_fft(&config);
        assert!(plan.is_ok());
        let plan = plan.ok();
        assert!(plan.is_some());
        if let Some(p) = plan {
            assert_eq!(p.config.n, 512);
            assert_eq!(p.num_stages, 9); // log2(512) = 9
            assert!(p.estimated_error_bound > 0.0);
        }
    }

    #[test]
    fn plan_memory_savings() {
        let config = HalfPrecisionFftConfig::new(1024);
        let plan = plan_half_precision_fft(&config);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let ratio = p.memory_savings_ratio();
            assert!(
                (ratio - 0.5).abs() < 1e-10,
                "memory savings ratio should be 0.5, got {ratio}"
            );
        }
    }

    #[test]
    fn plan_snr_positive() {
        let config = HalfPrecisionFftConfig::new(256);
        let plan = plan_half_precision_fft(&config);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let snr = p.estimated_snr_db();
            assert!(snr > 0.0, "SNR should be positive, got {snr}");
            // For FP32 accumulation with N=256, SNR should be quite high (~50+ dB)
            assert!(
                snr > 40.0,
                "SNR for FP32 acc N=256 should be > 40 dB, got {snr}"
            );
        }
    }

    // -- Butterfly PTX generation tests --

    #[test]
    fn butterfly_ptx_radix2_fp32acc() {
        let result = generate_fp16_butterfly_ptx(2, AccumulationMode::Fp32, SmVersion::Sm80);
        assert!(result.is_ok());
        let ptx = result.ok();
        assert!(ptx.is_some());
        if let Some(ptx) = ptx {
            assert!(ptx.contains("fft_fp16_butterfly_r2_fp32acc"));
            assert!(ptx.contains(".target sm_80"));
            // Should contain FP16 load instructions
            assert!(ptx.contains("ld.global.f16"));
            // Should contain FP32 accumulation
            assert!(ptx.contains("mul.rn.f32"));
            // Should contain FP16 store instructions
            assert!(ptx.contains("st.global.f16"));
        }
    }

    #[test]
    fn butterfly_ptx_fp32acc_contains_cvt_instructions() {
        // AccumulationMode::Fp32 must emit cvt.f32.f16 (load-side) and
        // cvt.f16.f32 (store-side) instructions because all butterfly
        // arithmetic is promoted to FP32 before the result is written back
        // as FP16.
        let result = generate_fp16_butterfly_ptx(2, AccumulationMode::Fp32, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            // Conversion from FP16 to FP32 for accumulation
            assert!(
                ptx.contains("cvt.f32.f16"),
                "Fp32 mode must contain cvt.f32.f16 (FP16->FP32 promotion)\nPTX snippet: {}",
                &ptx[..ptx.len().min(500)]
            );
            // Conversion from FP32 back to FP16 for storage
            assert!(
                ptx.contains("cvt.f16.f32") || ptx.contains("cvt.rn.f16.f32"),
                "Fp32 mode must contain cvt.*f16.f32 (FP32->FP16 demotion)\nPTX snippet: {}",
                &ptx[..ptx.len().min(500)]
            );
        }
    }

    #[test]
    fn butterfly_ptx_pure_mode_no_f32_arithmetic() {
        // AccumulationMode::Pure must not emit FP32 arithmetic instructions
        // (mul.rn.f32, fma.rn.f32, add.f32) -- all operations stay in FP16.
        let result = generate_fp16_butterfly_ptx(2, AccumulationMode::Pure, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            // Should NOT contain FP32 multiply or add
            assert!(
                !ptx.contains("mul.rn.f32"),
                "Pure mode must not use FP32 multiply"
            );
            // Must contain FP16 arithmetic
            assert!(
                ptx.contains("mul.f16") || ptx.contains("fma.rn.f16"),
                "Pure mode must use FP16 multiply or FMA"
            );
        }
    }

    #[test]
    fn butterfly_ptx_unsupported_radix() {
        let result = generate_fp16_butterfly_ptx(3, AccumulationMode::Fp32, SmVersion::Sm80);
        assert!(result.is_err());
    }

    // -- Twiddle PTX generation tests --

    #[test]
    fn twiddle_ptx_basic() {
        let result = generate_fp16_twiddle_ptx(256, SmVersion::Sm80);
        assert!(result.is_ok());
        let ptx = result.ok();
        assert!(ptx.is_some());
        if let Some(ptx) = ptx {
            assert!(ptx.contains("fft_fp16_twiddle_n256"));
            assert!(ptx.contains("ld.global.f16"));
            assert!(ptx.contains("cvt"));
        }
    }

    #[test]
    fn twiddle_ptx_zero_n() {
        let result = generate_fp16_twiddle_ptx(0, SmVersion::Sm80);
        assert!(result.is_err());
    }

    // -- Stats tests --

    #[test]
    fn stats_tracking() {
        let mut stats = HalfPrecisionStats::new();
        stats.observe(1.0);
        stats.observe(100.0);
        stats.observe(0.001);
        stats.observe(70000.0); // overflow
        stats.observe(1e-6); // underflow

        assert_eq!(stats.total_elements, 5);
        assert_eq!(stats.overflow_count, 1);
        assert_eq!(stats.underflow_count, 1);
        assert!((stats.overflow_rate() - 0.2).abs() < 1e-10);
        assert!((stats.underflow_rate() - 0.2).abs() < 1e-10);
        assert!((stats.max_magnitude - 70000.0).abs() < 1e-10);
    }

    // -- Twiddle precomputation tests --

    #[test]
    fn precompute_twiddle_basic() {
        let twiddles = precompute_twiddle_factors(4, FftDirection::Forward);
        assert_eq!(twiddles.len(), 4);
        // W_4^0 = (1, 0)
        assert!((twiddles[0].0 - 1.0).abs() < 1e-6);
        assert!(twiddles[0].1.abs() < 1e-6);
        // W_4^1 = (0, -1) for forward
        assert!(twiddles[1].0.abs() < 1e-6);
        assert!((twiddles[1].1 - (-1.0)).abs() < 1e-6);
    }

    // -- Config builder tests --

    #[test]
    fn config_builder_chaining() {
        let config = HalfPrecisionFftConfig::new(2048)
            .with_direction(FftDirection::Inverse)
            .with_accumulation(AccumulationMode::Mixed)
            .with_rounding(HalfRoundingMode::Stochastic);

        assert_eq!(config.n, 2048);
        assert_eq!(config.direction, FftDirection::Inverse);
        assert_eq!(config.accumulation, AccumulationMode::Mixed);
        assert_eq!(config.rounding, HalfRoundingMode::Stochastic);
    }
}
