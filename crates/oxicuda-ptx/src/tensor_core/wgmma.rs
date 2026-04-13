//! WGMMA (Warp Group Matrix Multiply-Accumulate) helpers for Hopper+ (`sm_90`+).
//!
//! WGMMA instructions operate at the warp-group level (4 warps = 128 threads)
//! and provide higher throughput than per-warp MMA instructions. They use
//! descriptor-based operand addressing and support asynchronous execution
//! overlapped with data movement via TMA.
//!
//! This module provides [`WgmmaConfig`] for validating type combinations and
//! computing register requirements for WGMMA instructions.
//!
//! # Supported shapes
//!
//! All shapes have M=64 and K=16 (for F16/BF16), with N varying:
//!
//! | Shape       | A/B types       | Accumulator | Notes          |
//! |-------------|-----------------|-------------|----------------|
//! | 64xNx16     | F16, BF16       | F32         | N = 8..256     |

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::{PtxType, WgmmaShape};

/// WGMMA instruction configuration.
///
/// Defines the shape and type parameters for a `wgmma.mma_async.sync.aligned`
/// instruction. WGMMA requires Hopper+ (`sm_90`) architecture.
#[derive(Debug, Clone)]
pub struct WgmmaConfig {
    /// Matrix tile shape (M=64, variable N, K=16).
    pub shape: WgmmaShape,
    /// Element type of matrix A (loaded via descriptor).
    pub a_type: PtxType,
    /// Element type of matrix B (loaded via descriptor).
    pub b_type: PtxType,
    /// Accumulator (C/D) element type.
    pub accumulator: PtxType,
}

impl WgmmaConfig {
    /// Creates a new WGMMA configuration.
    #[must_use]
    pub const fn new(
        shape: WgmmaShape,
        a_type: PtxType,
        b_type: PtxType,
        accumulator: PtxType,
    ) -> Self {
        Self {
            shape,
            a_type,
            b_type,
            accumulator,
        }
    }

    /// Validates that the type combination is supported for WGMMA.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::InvalidType`] if:
    /// - A/B types are not F16 or BF16
    /// - A and B types differ
    /// - Accumulator is not F32
    pub fn validate(&self) -> Result<(), PtxGenError> {
        if self.a_type != self.b_type {
            return Err(PtxGenError::InvalidType(format!(
                "WGMMA requires matching A/B types, got A={}, B={}",
                self.a_type.as_ptx_str(),
                self.b_type.as_ptx_str()
            )));
        }

        if !matches!(
            self.a_type,
            PtxType::F16 | PtxType::BF16 | PtxType::E4M3 | PtxType::E5M2
        ) {
            return Err(PtxGenError::InvalidType(format!(
                "WGMMA A/B type must be F16, BF16, E4M3, or E5M2, got {}",
                self.a_type.as_ptx_str()
            )));
        }

        if self.accumulator != PtxType::F32 {
            return Err(PtxGenError::InvalidType(format!(
                "WGMMA accumulator must be F32, got {}",
                self.accumulator.as_ptx_str()
            )));
        }

        Ok(())
    }

    /// Checks whether WGMMA is supported on the given architecture.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::UnsupportedFeature`] if the target does not
    /// support WGMMA instructions (requires `sm_90`+).
    pub fn check_arch_support(&self, sm: SmVersion) -> Result<(), PtxGenError> {
        if !sm.capabilities().has_wgmma {
            return Err(PtxGenError::UnsupportedFeature {
                arch: sm.as_ptx_str().to_string(),
                feature: "WGMMA (warp group MMA, Hopper+)".to_string(),
            });
        }
        Ok(())
    }

    /// Returns the (M, N, K) dimensions for this WGMMA shape.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32, u32) {
        match self.shape {
            WgmmaShape::M64N8K16 => (64, 8, 16),
            WgmmaShape::M64N16K16 => (64, 16, 16),
            WgmmaShape::M64N32K16 => (64, 32, 16),
            WgmmaShape::M64N64K16 => (64, 64, 16),
            WgmmaShape::M64N128K16 => (64, 128, 16),
            WgmmaShape::M64N256K16 => (64, 256, 16),
        }
    }

    /// Returns the number of accumulator registers per thread for this WGMMA shape.
    ///
    /// A warp group contains 128 threads (4 warps). The accumulator fragment
    /// is distributed across all threads. For F32 accumulators, the number of
    /// registers scales with N.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn regs_per_thread_accumulator(&self) -> Result<u32, PtxGenError> {
        self.validate()?;
        let (m, n, _) = self.dimensions();
        // Total accumulator elements = M * N
        // Distributed across 128 threads (warp group)
        // Each element is F32 = 1 register
        let total_elements = m * n;
        let threads_per_warp_group = 128u32;
        Ok(total_elements / threads_per_warp_group)
    }

    /// Returns the number of descriptor registers needed for the A operand.
    ///
    /// WGMMA uses descriptor-based addressing for matrix A. A single 64-bit
    /// descriptor register is used.
    #[must_use]
    pub const fn descriptor_regs_a(&self) -> u32 {
        1 // One 64-bit descriptor register
    }

    /// Returns the number of descriptor registers needed for the B operand.
    ///
    /// WGMMA uses descriptor-based addressing for matrix B. A single 64-bit
    /// descriptor register is used.
    #[must_use]
    pub const fn descriptor_regs_b(&self) -> u32 {
        1 // One 64-bit descriptor register
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_f16_config() {
        let cfg = WgmmaConfig::new(
            WgmmaShape::M64N128K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn valid_bf16_config() {
        let cfg = WgmmaConfig::new(
            WgmmaShape::M64N64K16,
            PtxType::BF16,
            PtxType::BF16,
            PtxType::F32,
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn mismatched_types() {
        let cfg = WgmmaConfig::new(
            WgmmaShape::M64N128K16,
            PtxType::F16,
            PtxType::BF16,
            PtxType::F32,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_accumulator() {
        let cfg = WgmmaConfig::new(
            WgmmaShape::M64N128K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F16,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn arch_support() {
        let cfg = WgmmaConfig::new(
            WgmmaShape::M64N128K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(cfg.check_arch_support(SmVersion::Sm80).is_err());
        assert!(cfg.check_arch_support(SmVersion::Sm90).is_ok());
        assert!(cfg.check_arch_support(SmVersion::Sm90a).is_ok());
        assert!(cfg.check_arch_support(SmVersion::Sm100).is_ok());
    }

    #[test]
    fn dimensions() {
        let cfg = WgmmaConfig::new(
            WgmmaShape::M64N128K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        assert_eq!(cfg.dimensions(), (64, 128, 16));
    }

    #[test]
    fn accumulator_regs() {
        let cfg = WgmmaConfig::new(
            WgmmaShape::M64N128K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        // 64 * 128 = 8192 elements / 128 threads = 64 regs per thread
        assert_eq!(cfg.regs_per_thread_accumulator().expect("valid"), 64);

        let cfg2 = WgmmaConfig::new(
            WgmmaShape::M64N8K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        // 64 * 8 = 512 / 128 = 4 regs per thread
        assert_eq!(cfg2.regs_per_thread_accumulator().expect("valid"), 4);
    }

    // ── wgmma.mma_async instruction format and arch verification ─────────────

    /// Verify accumulator register counts for all standard WGMMA shapes.
    /// Each element is F32 (1 register), distributed across 128 threads.
    #[test]
    fn test_wgmma_mma_async_accumulator_register_distribution() {
        // Shape m64n128k16: 64*128 = 8192 elements / 128 threads = 64 regs
        let cfg = WgmmaConfig::new(
            WgmmaShape::M64N128K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        assert_eq!(
            cfg.regs_per_thread_accumulator().expect("valid"),
            64,
            "m64n128k16 must have 64 accumulator registers per thread"
        );

        // Shape m64n256k16: 64*256 = 16384 elements / 128 threads = 128 regs
        let cfg256 = WgmmaConfig::new(
            WgmmaShape::M64N256K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        assert_eq!(
            cfg256.regs_per_thread_accumulator().expect("valid"),
            128,
            "m64n256k16 must have 128 accumulator registers per thread"
        );

        // Shape m64n16k16: 64*16 = 1024 elements / 128 threads = 8 regs
        let cfg16 = WgmmaConfig::new(
            WgmmaShape::M64N16K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        assert_eq!(
            cfg16.regs_per_thread_accumulator().expect("valid"),
            8,
            "m64n16k16 must have 8 accumulator registers per thread"
        );
    }

    /// Verify that descriptor register counts are always 1 (one 64-bit descriptor each).
    #[test]
    fn test_wgmma_descriptor_register_counts() {
        let cfg = WgmmaConfig::new(
            WgmmaShape::M64N128K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        assert_eq!(
            cfg.descriptor_regs_a(),
            1,
            "WGMMA A operand always uses 1 descriptor register"
        );
        assert_eq!(
            cfg.descriptor_regs_b(),
            1,
            "WGMMA B operand always uses 1 descriptor register"
        );
    }

    /// Verify WGMMA arch check correctly rejects SM < 90.
    #[test]
    fn test_wgmma_requires_sm90_rejects_all_pre_hopper() {
        let cfg = WgmmaConfig::new(
            WgmmaShape::M64N128K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        // All pre-Hopper architectures must fail
        for sm in [
            SmVersion::Sm75,
            SmVersion::Sm80,
            SmVersion::Sm86,
            SmVersion::Sm89,
        ] {
            assert!(
                cfg.check_arch_support(sm).is_err(),
                "WGMMA should be rejected on {sm:?}"
            );
        }
        // Hopper and above must succeed
        for sm in [
            SmVersion::Sm90,
            SmVersion::Sm90a,
            SmVersion::Sm100,
            SmVersion::Sm120,
        ] {
            assert!(
                cfg.check_arch_support(sm).is_ok(),
                "WGMMA should be supported on {sm:?}"
            );
        }
    }

    /// Verify E4M3 and E5M2 FP8 inputs are valid for WGMMA.
    #[test]
    fn test_wgmma_fp8_e4m3_e5m2_inputs_valid() {
        let cfg_e4m3 = WgmmaConfig::new(
            WgmmaShape::M64N128K16,
            PtxType::E4M3,
            PtxType::E4M3,
            PtxType::F32,
        );
        assert!(
            cfg_e4m3.validate().is_ok(),
            "WGMMA with E4M3 inputs should be valid"
        );

        let cfg_e5m2 = WgmmaConfig::new(
            WgmmaShape::M64N64K16,
            PtxType::E5M2,
            PtxType::E5M2,
            PtxType::F32,
        );
        assert!(
            cfg_e5m2.validate().is_ok(),
            "WGMMA with E5M2 inputs should be valid"
        );
    }

    /// Verify the WGMMA shape naming covers all N variants (8 to 256).
    #[test]
    fn test_wgmma_all_shapes_have_m64_k16() {
        let shapes = [
            (WgmmaShape::M64N8K16, (64u32, 8u32, 16u32)),
            (WgmmaShape::M64N16K16, (64, 16, 16)),
            (WgmmaShape::M64N32K16, (64, 32, 16)),
            (WgmmaShape::M64N64K16, (64, 64, 16)),
            (WgmmaShape::M64N128K16, (64, 128, 16)),
            (WgmmaShape::M64N256K16, (64, 256, 16)),
        ];
        for (shape, expected) in shapes {
            let cfg = WgmmaConfig::new(shape, PtxType::F16, PtxType::F16, PtxType::F32);
            assert_eq!(
                cfg.dimensions(),
                expected,
                "shape {shape:?} must have dimensions {expected:?}"
            );
            let (m, _, k) = cfg.dimensions();
            assert_eq!(m, 64, "WGMMA M must always be 64");
            assert_eq!(k, 16, "WGMMA K must always be 16");
        }
    }
}
