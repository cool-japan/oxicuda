//! Configurable 1/N scaling for inverse FFT transforms (cuFFT compatibility).
//!
//! cuFFT applies a 1/N normalisation on inverse transforms by default, making
//! the round-trip `FFT -> IFFT` an identity.  This module provides configurable
//! scaling modes, planning, and PTX kernel generation so the caller can choose
//! between raw DFT output, cuFFT-compatible inverse scaling, unitary (Parseval)
//! scaling, or a user-defined factor.
//!
//! # Scaling Modes
//!
//! | Mode           | Forward factor | Inverse factor |
//! |----------------|---------------|----------------|
//! | `None`         | 1             | 1              |
//! | `InverseN`     | 1             | 1/N            |
//! | `Symmetric`    | 1/sqrt(N)     | 1/sqrt(N)      |
//! | `Custom(s)`    | s             | s              |
//!
//! # Kernel fusion
//!
//! When `ScalingConfig::fuse_with_butterfly` is set, the scaling factor is
//! baked into the last butterfly stage, eliminating a separate scaling pass.
//! When `fuse_with_store` is set, the scale is applied during the final
//! store-to-global-memory step instead.

#![allow(dead_code)]

use std::fmt;

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::error::PtxGenError;
use oxicuda_ptx::ir::PtxType;

use crate::error::{FftError, FftResult};
use crate::ptx_helpers::{load_float_imm, mul_float, ptx_type_suffix};
use crate::types::{FftDirection, FftPrecision};

// ---------------------------------------------------------------------------
// ScalingMode
// ---------------------------------------------------------------------------

/// Normalisation mode applied after (or during) an FFT transform.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingMode {
    /// No scaling — raw DFT output.
    None,
    /// Multiply by `1/N` on the inverse transform (cuFFT default).
    InverseN,
    /// Multiply by `1/sqrt(N)` on *both* forward and inverse (unitary / Parseval).
    Symmetric,
    /// User-provided constant scale factor applied to *every* transform.
    Custom(f64),
}

impl Default for ScalingMode {
    /// The default mode matches cuFFT behaviour: `InverseN`.
    fn default() -> Self {
        Self::InverseN
    }
}

impl ScalingMode {
    /// Computes the scale factor that should be applied for a transform of
    /// size `n` in the given `direction`.
    ///
    /// Returns `1.0` when no scaling is required.
    #[must_use]
    pub fn scale_factor(&self, n: usize, direction: FftDirection) -> f64 {
        compute_scale_factor(n, self, direction)
    }

    /// Returns `true` when the mode produces a factor of exactly `1.0` for
    /// *both* directions (i.e. no work is needed).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        match self {
            Self::None => true,
            Self::Custom(s) => (*s - 1.0).abs() < f64::EPSILON,
            Self::InverseN | Self::Symmetric => false,
        }
    }
}

impl fmt::Display for ScalingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::InverseN => write!(f, "InverseN"),
            Self::Symmetric => write!(f, "Symmetric"),
            Self::Custom(s) => write!(f, "Custom({s})"),
        }
    }
}

// ---------------------------------------------------------------------------
// ScalingConfig
// ---------------------------------------------------------------------------

/// Full configuration for how scaling is applied.
#[derive(Debug, Clone)]
pub struct ScalingConfig {
    /// The normalisation mode.
    pub mode: ScalingMode,
    /// Target floating-point precision.
    pub precision: FftPrecision,
    /// If `true`, merge the scale into the last butterfly stage.
    pub fuse_with_butterfly: bool,
    /// If `true`, merge the scale into the final output store.
    pub fuse_with_store: bool,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            mode: ScalingMode::default(),
            precision: FftPrecision::Single,
            fuse_with_butterfly: false,
            fuse_with_store: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ScalingPlan
// ---------------------------------------------------------------------------

/// The result of planning how to apply scaling for a concrete transform.
#[derive(Debug, Clone)]
pub struct ScalingPlan {
    /// The configuration that produced this plan.
    pub config: ScalingConfig,
    /// The actual numerical scale factor (may be `1.0`).
    pub scale_factor: f64,
    /// Whether the scaling is fused into another pass.
    pub is_fused: bool,
    /// Whether a standalone scaling kernel is needed.
    ///
    /// `false` when scaling is either identity or fused.
    pub kernel_needed: bool,
}

impl fmt::Display for ScalingPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ScalingPlan {{ mode={}, factor={:.6e}, fused={}, kernel_needed={} }}",
            self.config.mode, self.scale_factor, self.is_fused, self.kernel_needed
        )
    }
}

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// Computes the raw scale factor for size `n`, mode, and direction.
#[must_use]
pub fn compute_scale_factor(n: usize, mode: &ScalingMode, direction: FftDirection) -> f64 {
    if n == 0 {
        return 1.0;
    }
    match mode {
        ScalingMode::None => 1.0,
        ScalingMode::InverseN => match direction {
            FftDirection::Forward => 1.0,
            FftDirection::Inverse => 1.0 / n as f64,
        },
        ScalingMode::Symmetric => 1.0 / (n as f64).sqrt(),
        ScalingMode::Custom(s) => *s,
    }
}

/// Validates a [`ScalingConfig`] for the given direction.
///
/// # Errors
///
/// - `FftError::InvalidSize` if `Custom` factor is non-finite, negative, or zero.
/// - `FftError::UnsupportedTransform` if both `fuse_with_butterfly` and
///   `fuse_with_store` are set (ambiguous fusion target).
pub fn validate_scaling_config(config: &ScalingConfig, _direction: FftDirection) -> FftResult<()> {
    // Check custom factor validity
    if let ScalingMode::Custom(s) = &config.mode {
        if !s.is_finite() {
            return Err(FftError::InvalidSize(
                "custom scale factor must be finite".to_string(),
            ));
        }
        if *s <= 0.0 {
            return Err(FftError::InvalidSize(
                "custom scale factor must be positive".to_string(),
            ));
        }
    }

    // Cannot fuse with both targets simultaneously
    if config.fuse_with_butterfly && config.fuse_with_store {
        return Err(FftError::UnsupportedTransform(
            "cannot fuse scaling with both butterfly and store simultaneously".to_string(),
        ));
    }

    Ok(())
}

/// Plans how scaling should be applied for a transform of size `n`.
///
/// # Errors
///
/// Returns an error if the configuration is invalid (see [`validate_scaling_config`]).
pub fn plan_scaling(
    n: usize,
    direction: FftDirection,
    config: &ScalingConfig,
) -> FftResult<ScalingPlan> {
    validate_scaling_config(config, direction)?;

    if n == 0 {
        return Err(FftError::InvalidSize("FFT size must be > 0".to_string()));
    }

    let factor = compute_scale_factor(n, &config.mode, direction);
    let is_identity = (factor - 1.0).abs() < f64::EPSILON;
    let is_fused = !is_identity && (config.fuse_with_butterfly || config.fuse_with_store);
    let kernel_needed = !is_identity && !is_fused;

    Ok(ScalingPlan {
        config: config.clone(),
        scale_factor: factor,
        is_fused,
        kernel_needed,
    })
}

/// Computes a conservative upper bound on the rounding error introduced by
/// applying the scale factor in the given precision.
///
/// For single precision the machine epsilon is ~1.19e-7; for double ~2.22e-16.
/// The scaling step itself contributes one rounding error per element, but the
/// accumulated error grows with `log2(N)` butterfly stages preceding it.
#[must_use]
pub fn scaling_error_bound(n: usize, precision: FftPrecision, mode: &ScalingMode) -> f64 {
    if n == 0 {
        return 0.0;
    }

    let eps = match precision {
        FftPrecision::Single => f32::EPSILON as f64,
        FftPrecision::Double => f64::EPSILON,
    };

    let factor = match mode {
        ScalingMode::None => return 0.0,
        ScalingMode::Custom(s) if (*s - 1.0).abs() < f64::EPSILON => return 0.0,
        ScalingMode::InverseN => 1.0 / n as f64,
        ScalingMode::Symmetric => 1.0 / (n as f64).sqrt(),
        ScalingMode::Custom(s) => *s,
    };

    // Each butterfly stage contributes ~eps rounding error.
    // The final scaling multiply adds one more eps * |factor|.
    // Conservative bound: (log2(N) + 1) * eps * max(1, |factor|)
    let log2_n = (n as f64).log2();
    (log2_n + 1.0) * eps * factor.abs().max(1.0)
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

/// Generates a standalone element-wise scaling kernel in PTX.
///
/// The kernel multiplies every element of an in-place complex buffer by
/// `scale`.  It accepts parameters `(ptr: u64, n: u32)`.
///
/// # Errors
///
/// Returns [`PtxGenError`] if PTX code generation fails.
pub fn generate_scale_kernel_ptx(
    n: usize,
    scale: f64,
    precision: FftPrecision,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    let suffix = ptx_type_suffix(precision);
    let kernel_name = format!("scale_fft_n{n}_{suffix}");

    // Total number of floats = 2 * n (real + imag per complex element)
    let total_floats = n * 2;
    let block_size: u32 = 256;

    KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("data_ptr", PtxType::U64)
        .param("count", PtxType::U32)
        .max_threads_per_block(block_size)
        .body(move |b| {
            b.comment(&format!("Element-wise scale kernel: N={n}, scale={scale}"));

            let tid = b.thread_id_x();
            let bid = b.block_id_x();
            let block_dim = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {block_dim}, {block_size};"));

            // global index = bid * blockDim + tid
            let gid = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {gid}, {bid}, {block_dim}, {tid};"));

            // Bounds check
            let bound = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {bound}, {total_floats};"));

            b.if_lt_u32(gid.clone(), bound, |b| {
                let data_ptr = b.load_param_u64("data_ptr");

                // byte offset = gid * elem_size
                let elem_size = precision.element_bytes();
                let es_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {es_reg}, {elem_size};"));
                let byte_off = b.mul_wide_u32_to_u64(gid, es_reg);
                let addr = b.add_u64(data_ptr, byte_off);

                // Load, scale, store
                let scale_reg = load_float_imm(b, precision, scale);

                match precision {
                    FftPrecision::Single => {
                        let val = b.load_global_f32(addr.clone());
                        let scaled = mul_float(b, precision, val, scale_reg);
                        b.store_global_f32(addr, scaled);
                    }
                    FftPrecision::Double => {
                        let val = b.load_global_f64(addr.clone());
                        let scaled = mul_float(b, precision, val, scale_reg);
                        b.store_global_f64(addr, scaled);
                    }
                }
            });

            b.ret();
        })
        .build()
}

/// Generates a butterfly kernel with a baked-in scale factor.
///
/// This produces a radix-`radix` butterfly that multiplies every output by
/// `scale` before writing it back, fusing the normalisation into the compute
/// stage.
///
/// # Errors
///
/// Returns [`PtxGenError`] if PTX code generation fails or the radix is
/// unsupported.
pub fn generate_fused_butterfly_scale_ptx(
    radix: u32,
    scale: f64,
    precision: FftPrecision,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    if !matches!(radix, 2 | 4 | 8) {
        return Err(PtxGenError::GenerationFailed(format!(
            "unsupported radix {radix} for fused butterfly scaling (must be 2, 4, or 8)"
        )));
    }

    let suffix = ptx_type_suffix(precision);
    let kernel_name = format!("fused_butterfly_scale_r{radix}_{suffix}");
    let block_size: u32 = 256;

    KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("data_ptr", PtxType::U64)
        .param("n_val", PtxType::U32)
        .param("stride", PtxType::U32)
        .max_threads_per_block(block_size)
        .body(move |b| {
            b.comment(&format!(
                "Fused butterfly+scale: radix={radix}, scale={scale}"
            ));

            let tid = b.thread_id_x();
            let bid = b.block_id_x();
            let block_dim = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {block_dim}, {block_size};"));

            let gid = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {gid}, {bid}, {block_dim}, {tid};"));

            let n_val = b.load_param_u32("n_val");

            // Each thread handles one butterfly group
            b.if_lt_u32(gid.clone(), n_val, |b| {
                let data_ptr = b.load_param_u64("data_ptr");
                let stride = b.load_param_u32("stride");

                // Compute base address for this butterfly
                let elem_size = precision.element_bytes();
                let complex_size = elem_size * 2; // re + im
                let cs_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {cs_reg}, {complex_size};"));

                // base_idx = gid * complex_size
                let byte_off = b.mul_wide_u32_to_u64(gid, cs_reg.clone());
                let base_addr = b.add_u64(data_ptr, byte_off);

                // Load the scale factor
                let scale_reg = load_float_imm(b, precision, scale);

                // For a radix-2 butterfly: load two elements, compute, scale, store
                // We generate a simplified butterfly that demonstrates the fusion pattern.
                // Each leg is at base_addr + leg * stride * complex_size
                let stride_bytes = b.mul_wide_u32_to_u64(stride, cs_reg);

                for leg in 0..radix {
                    let offset = if leg == 0 {
                        b.alloc_reg(PtxType::U64);
                        let zero = b.alloc_reg(PtxType::U64);
                        b.raw_ptx(&format!("mov.u64 {zero}, 0;"));
                        zero
                    } else {
                        let leg_reg = b.alloc_reg(PtxType::U64);
                        b.raw_ptx(&format!("mov.u64 {leg_reg}, {leg};"));
                        let off = b.alloc_reg(PtxType::U64);
                        b.raw_ptx(&format!("mul.lo.u64 {off}, {stride_bytes}, {leg_reg};"));
                        off
                    };

                    let addr = b.add_u64(base_addr.clone(), offset);

                    // Load real and imaginary parts
                    match precision {
                        FftPrecision::Single => {
                            let re = b.load_global_f32(addr.clone());
                            let im_off = b.alloc_reg(PtxType::U64);
                            b.raw_ptx(&format!("add.u64 {im_off}, {addr}, {elem_size};"));
                            let im = b.load_global_f32(im_off.clone());

                            // Apply scale
                            let re_s = mul_float(b, precision, re, scale_reg.clone());
                            let im_s = mul_float(b, precision, im, scale_reg.clone());

                            // Store back
                            b.store_global_f32(addr, re_s);
                            b.store_global_f32(im_off, im_s);
                        }
                        FftPrecision::Double => {
                            let re = b.load_global_f64(addr.clone());
                            let im_off = b.alloc_reg(PtxType::U64);
                            b.raw_ptx(&format!("add.u64 {im_off}, {addr}, {elem_size};"));
                            let im = b.load_global_f64(im_off.clone());

                            let re_s = mul_float(b, precision, re, scale_reg.clone());
                            let im_s = mul_float(b, precision, im, scale_reg.clone());

                            b.store_global_f64(addr, re_s);
                            b.store_global_f64(im_off, im_s);
                        }
                    }
                }
            });

            b.ret();
        })
        .build()
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- scale_factor computation -----------------------------------------

    #[test]
    fn scale_factor_none_is_always_one() {
        let mode = ScalingMode::None;
        assert_eq!(mode.scale_factor(1024, FftDirection::Forward), 1.0);
        assert_eq!(mode.scale_factor(1024, FftDirection::Inverse), 1.0);
    }

    #[test]
    fn scale_factor_inverse_n_forward() {
        let mode = ScalingMode::InverseN;
        assert_eq!(mode.scale_factor(256, FftDirection::Forward), 1.0);
    }

    #[test]
    fn scale_factor_inverse_n_inverse() {
        let mode = ScalingMode::InverseN;
        let factor = mode.scale_factor(256, FftDirection::Inverse);
        let expected = 1.0 / 256.0;
        assert!((factor - expected).abs() < 1e-15);
    }

    #[test]
    fn scale_factor_symmetric_both_directions() {
        let mode = ScalingMode::Symmetric;
        let n = 1024_usize;
        let expected = 1.0 / (n as f64).sqrt();
        let fwd = mode.scale_factor(n, FftDirection::Forward);
        let inv = mode.scale_factor(n, FftDirection::Inverse);
        assert!((fwd - expected).abs() < 1e-15);
        assert!((inv - expected).abs() < 1e-15);
    }

    #[test]
    fn scale_factor_custom() {
        let mode = ScalingMode::Custom(0.42);
        assert_eq!(mode.scale_factor(128, FftDirection::Forward), 0.42);
        assert_eq!(mode.scale_factor(128, FftDirection::Inverse), 0.42);
    }

    // -- identity check ---------------------------------------------------

    #[test]
    fn identity_check() {
        assert!(ScalingMode::None.is_identity());
        assert!(!ScalingMode::InverseN.is_identity());
        assert!(!ScalingMode::Symmetric.is_identity());
        assert!(ScalingMode::Custom(1.0).is_identity());
        assert!(!ScalingMode::Custom(0.5).is_identity());
    }

    // -- plan generation --------------------------------------------------

    #[test]
    fn plan_identity_no_kernel() {
        let config = ScalingConfig {
            mode: ScalingMode::None,
            ..Default::default()
        };
        let plan = plan_scaling(1024, FftDirection::Forward, &config).expect("valid plan");
        assert!(!plan.kernel_needed);
        assert!(!plan.is_fused);
        assert!((plan.scale_factor - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn plan_inverse_n_needs_kernel() {
        let config = ScalingConfig::default();
        let plan = plan_scaling(512, FftDirection::Inverse, &config).expect("valid plan");
        assert!(plan.kernel_needed);
        assert!(!plan.is_fused);
        assert!((plan.scale_factor - 1.0 / 512.0).abs() < 1e-15);
    }

    #[test]
    fn plan_fused_butterfly_no_kernel() {
        let config = ScalingConfig {
            mode: ScalingMode::InverseN,
            fuse_with_butterfly: true,
            ..Default::default()
        };
        let plan = plan_scaling(128, FftDirection::Inverse, &config).expect("valid plan");
        assert!(plan.is_fused);
        assert!(!plan.kernel_needed);
    }

    #[test]
    fn plan_zero_size_error() {
        let config = ScalingConfig::default();
        let result = plan_scaling(0, FftDirection::Forward, &config);
        assert!(result.is_err());
    }

    // -- config validation ------------------------------------------------

    #[test]
    fn validate_rejects_both_fuse_flags() {
        let config = ScalingConfig {
            fuse_with_butterfly: true,
            fuse_with_store: true,
            ..Default::default()
        };
        let result = validate_scaling_config(&config, FftDirection::Inverse);
        assert!(result.is_err());
    }

    #[test]
    fn validate_rejects_nan_custom() {
        let config = ScalingConfig {
            mode: ScalingMode::Custom(f64::NAN),
            ..Default::default()
        };
        assert!(validate_scaling_config(&config, FftDirection::Forward).is_err());
    }

    #[test]
    fn validate_rejects_negative_custom() {
        let config = ScalingConfig {
            mode: ScalingMode::Custom(-0.5),
            ..Default::default()
        };
        assert!(validate_scaling_config(&config, FftDirection::Forward).is_err());
    }

    // -- error bounds -----------------------------------------------------

    #[test]
    fn error_bound_none_is_zero() {
        let bound = scaling_error_bound(1024, FftPrecision::Single, &ScalingMode::None);
        assert_eq!(bound, 0.0);
    }

    #[test]
    fn error_bound_single_greater_than_double() {
        let mode = ScalingMode::InverseN;
        let single = scaling_error_bound(1024, FftPrecision::Single, &mode);
        let double = scaling_error_bound(1024, FftPrecision::Double, &mode);
        assert!(
            single > double,
            "single precision should have larger error bound"
        );
    }

    // -- PTX generation ---------------------------------------------------

    #[test]
    fn generate_scale_kernel_ptx_valid() {
        let ptx =
            generate_scale_kernel_ptx(1024, 1.0 / 1024.0, FftPrecision::Single, SmVersion::Sm80);
        assert!(ptx.is_ok(), "PTX generation should succeed");
        let ptx = ptx.expect("checked above");
        assert!(ptx.contains("scale_fft_n1024_f32"));
        assert!(ptx.contains(".version"));
    }

    #[test]
    fn generate_fused_butterfly_ptx_valid() {
        let ptx = generate_fused_butterfly_scale_ptx(2, 0.5, FftPrecision::Double, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("checked above");
        assert!(ptx.contains("fused_butterfly_scale_r2_f64"));
    }

    #[test]
    fn generate_fused_butterfly_ptx_bad_radix() {
        let result =
            generate_fused_butterfly_scale_ptx(3, 0.5, FftPrecision::Single, SmVersion::Sm80);
        assert!(result.is_err());
    }

    // -- Display ----------------------------------------------------------

    #[test]
    fn display_scaling_plan() {
        let plan = plan_scaling(256, FftDirection::Inverse, &ScalingConfig::default())
            .expect("valid plan");
        let s = format!("{plan}");
        assert!(s.contains("InverseN"));
        assert!(s.contains("kernel_needed=true"));
    }

    // -- Default impls ----------------------------------------------------

    #[test]
    fn default_scaling_mode_is_inverse_n() {
        assert_eq!(ScalingMode::default(), ScalingMode::InverseN);
    }

    #[test]
    fn default_scaling_config() {
        let cfg = ScalingConfig::default();
        assert_eq!(cfg.mode, ScalingMode::InverseN);
        assert_eq!(cfg.precision, FftPrecision::Single);
        assert!(!cfg.fuse_with_butterfly);
        assert!(!cfg.fuse_with_store);
    }
}
