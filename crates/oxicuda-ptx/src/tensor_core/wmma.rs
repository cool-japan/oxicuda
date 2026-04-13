//! WMMA (Warp Matrix Multiply-Accumulate) instruction generation helpers.
//!
//! WMMA instructions were introduced in Volta (`sm_70`) and provide warp-level
//! matrix multiply-accumulate operations. Each warp cooperatively loads, computes,
//! and stores matrix fragments using the `wmma.load`, `wmma.mma`, and `wmma.store`
//! instruction families.
//!
//! This module provides [`WmmaConfig`] for computing fragment register counts
//! and validating type combinations, and helper functions for determining the
//! number of registers needed per thread for each fragment.
//!
//! # Supported shapes
//!
//! | Shape     | A/B types      | C/D types       |
//! |-----------|----------------|-----------------|
//! | 16x16x16  | F16            | F16, F32        |
//! | 8x32x16   | F16            | F16, F32        |
//! | 32x8x16   | F16            | F16, F32        |

use crate::error::PtxGenError;
use crate::ir::{PtxType, WmmaShape};

/// WMMA fragment configuration.
///
/// Encapsulates the shape and type parameters for a WMMA operation. Use
/// [`validate`](WmmaConfig::validate) to check that the type combination is
/// supported, and [`fragment_size_a`](WmmaConfig::fragment_size_a) to determine
/// the number of registers each thread needs per fragment.
#[derive(Debug, Clone)]
pub struct WmmaConfig {
    /// Matrix tile shape (M x N x K).
    pub shape: WmmaShape,
    /// Element type of matrix A.
    pub a_type: PtxType,
    /// Element type of matrix B.
    pub b_type: PtxType,
    /// Element type of matrices C and D (accumulator).
    pub c_type: PtxType,
}

impl WmmaConfig {
    /// Creates a new WMMA configuration.
    #[must_use]
    pub const fn new(shape: WmmaShape, a_type: PtxType, b_type: PtxType, c_type: PtxType) -> Self {
        Self {
            shape,
            a_type,
            b_type,
            c_type,
        }
    }

    /// Validates that the type combination is supported for WMMA instructions.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::InvalidType`] if:
    /// - A and B types differ
    /// - A/B type is not F16
    /// - C type is not F16 or F32
    pub fn validate(&self) -> Result<(), PtxGenError> {
        if self.a_type != self.b_type {
            return Err(PtxGenError::InvalidType(format!(
                "WMMA requires matching A/B types, got A={}, B={}",
                self.a_type.as_ptx_str(),
                self.b_type.as_ptx_str()
            )));
        }

        if !matches!(self.a_type, PtxType::F16) {
            return Err(PtxGenError::InvalidType(format!(
                "WMMA A/B type must be F16, got {}",
                self.a_type.as_ptx_str()
            )));
        }

        if !matches!(self.c_type, PtxType::F16 | PtxType::F32) {
            return Err(PtxGenError::InvalidType(format!(
                "WMMA C/D type must be F16 or F32, got {}",
                self.c_type.as_ptx_str()
            )));
        }

        Ok(())
    }

    /// Returns the number of registers per thread for a matrix A fragment.
    ///
    /// For WMMA with F16 inputs, each thread in a warp holds a portion of the
    /// fragment. The number of registers depends on the shape and element size.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn fragment_size_a(&self) -> Result<u32, PtxGenError> {
        self.validate()?;
        // For F16 16x16x16: each thread holds 16 F16 values = 8 x B32 registers
        // For F16 8x32x16 and 32x8x16: same total elements, different layout
        Ok(match self.shape {
            WmmaShape::M16N16K16 | WmmaShape::M8N32K16 | WmmaShape::M32N8K16 => 8,
        })
    }

    /// Returns the number of registers per thread for a matrix B fragment.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn fragment_size_b(&self) -> Result<u32, PtxGenError> {
        self.validate()?;
        Ok(match self.shape {
            WmmaShape::M16N16K16 | WmmaShape::M8N32K16 | WmmaShape::M32N8K16 => 8,
        })
    }

    /// Returns the number of registers per thread for accumulator (C/D) fragments.
    ///
    /// F16 accumulator: 4 registers (8 F16 values packed into 4 B32 regs)
    /// F32 accumulator: 8 registers
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn fragment_size_c(&self) -> Result<u32, PtxGenError> {
        self.validate()?;
        Ok(match self.c_type {
            PtxType::F16 => 4,
            PtxType::F32 => 8,
            _ => {
                return Err(PtxGenError::InvalidType(format!(
                    "unsupported accumulator type: {}",
                    self.c_type.as_ptx_str()
                )));
            }
        })
    }

    /// Returns the (M, N, K) dimensions for this WMMA shape.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32, u32) {
        match self.shape {
            WmmaShape::M16N16K16 => (16, 16, 16),
            WmmaShape::M8N32K16 => (8, 32, 16),
            WmmaShape::M32N8K16 => (32, 8, 16),
        }
    }

    /// Returns the total number of elements in the A fragment across all threads.
    #[must_use]
    pub const fn total_elements_a(&self) -> u32 {
        let (m, _, k) = self.dimensions();
        m * k
    }

    /// Returns the total number of elements in the B fragment across all threads.
    #[must_use]
    pub const fn total_elements_b(&self) -> u32 {
        let (_, n, k) = self.dimensions();
        k * n
    }

    /// Returns the total number of elements in the C/D fragment across all threads.
    #[must_use]
    pub const fn total_elements_c(&self) -> u32 {
        let (m, n, _) = self.dimensions();
        m * n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_f16_f32_config() {
        let cfg = WmmaConfig::new(
            WmmaShape::M16N16K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.fragment_size_a().expect("valid"), 8);
        assert_eq!(cfg.fragment_size_b().expect("valid"), 8);
        assert_eq!(cfg.fragment_size_c().expect("valid"), 8);
    }

    #[test]
    fn valid_f16_f16_config() {
        let cfg = WmmaConfig::new(
            WmmaShape::M16N16K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F16,
        );
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.fragment_size_c().expect("valid"), 4);
    }

    #[test]
    fn mismatched_ab_types() {
        let cfg = WmmaConfig::new(
            WmmaShape::M16N16K16,
            PtxType::F16,
            PtxType::F32,
            PtxType::F32,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_a_type() {
        let cfg = WmmaConfig::new(
            WmmaShape::M16N16K16,
            PtxType::F32,
            PtxType::F32,
            PtxType::F32,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn dimensions() {
        let cfg = WmmaConfig::new(
            WmmaShape::M16N16K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        assert_eq!(cfg.dimensions(), (16, 16, 16));
        assert_eq!(cfg.total_elements_a(), 256);
        assert_eq!(cfg.total_elements_b(), 256);
        assert_eq!(cfg.total_elements_c(), 256);
    }

    #[test]
    fn m8n32k16_dimensions() {
        let cfg = WmmaConfig::new(
            WmmaShape::M8N32K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        assert_eq!(cfg.dimensions(), (8, 32, 16));
        assert_eq!(cfg.total_elements_a(), 128);
        assert_eq!(cfg.total_elements_b(), 512);
        assert_eq!(cfg.total_elements_c(), 256);
    }

    // ── WMMA m16n16k16 fragment layout verification (PTX ISA spec) ──────────

    /// Verify that the m16n16k16 shape fragment sizes match the PTX ISA 7.x spec:
    /// - A fragment: 8 registers per thread (16 F16 values packed into 8 B32 regs)
    /// - B fragment: 8 registers per thread
    /// - C/D fragment (F32 accumulator): 8 registers per thread
    /// - C/D fragment (F16 accumulator): 4 registers per thread
    #[test]
    fn test_wmma_m16n16k16_f16_accumulator_fragment_layout() {
        let cfg_f32 = WmmaConfig::new(
            WmmaShape::M16N16K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        // PTX ISA: m16n16k16 F16/F32 — 8 B32 regs per thread for A, B, and C(f32)
        assert_eq!(
            cfg_f32.fragment_size_a().expect("valid config"),
            8,
            "m16n16k16 A fragment must be 8 registers/thread"
        );
        assert_eq!(
            cfg_f32.fragment_size_b().expect("valid config"),
            8,
            "m16n16k16 B fragment must be 8 registers/thread"
        );
        assert_eq!(
            cfg_f32.fragment_size_c().expect("valid config"),
            8,
            "m16n16k16 C fragment (F32) must be 8 registers/thread"
        );

        let cfg_f16 = WmmaConfig::new(
            WmmaShape::M16N16K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F16,
        );
        // F16 accumulator packs 2 values per B32 register → 4 regs for 8 elements
        assert_eq!(
            cfg_f16.fragment_size_c().expect("valid config"),
            4,
            "m16n16k16 C fragment (F16) must be 4 registers/thread (2 values packed per reg)"
        );
    }

    /// Verify that the total element counts for m16n16k16 match the 16×16×16 geometry.
    #[test]
    fn test_wmma_m16n16k16_total_element_counts() {
        let cfg = WmmaConfig::new(
            WmmaShape::M16N16K16,
            PtxType::F16,
            PtxType::F16,
            PtxType::F32,
        );
        // A is 16×16 = 256 elements total across the warp
        assert_eq!(
            cfg.total_elements_a(),
            256,
            "m16n16k16 total A elements = M*K = 16*16 = 256"
        );
        // B is 16×16 = 256 elements total
        assert_eq!(
            cfg.total_elements_b(),
            256,
            "m16n16k16 total B elements = K*N = 16*16 = 256"
        );
        // C/D is 16×16 = 256 elements total
        assert_eq!(
            cfg.total_elements_c(),
            256,
            "m16n16k16 total C elements = M*N = 16*16 = 256"
        );
    }

    /// Verify that the three WMMA shapes all have the same fragment register count
    /// for A/B (they differ in layout, not count, for F16 inputs).
    #[test]
    fn test_wmma_all_shapes_same_ab_fragment_register_count() {
        for shape in [
            WmmaShape::M16N16K16,
            WmmaShape::M8N32K16,
            WmmaShape::M32N8K16,
        ] {
            let cfg = WmmaConfig::new(shape, PtxType::F16, PtxType::F16, PtxType::F32);
            assert_eq!(
                cfg.fragment_size_a().expect("valid"),
                8,
                "{shape:?} A fragment must be 8 regs"
            );
            assert_eq!(
                cfg.fragment_size_b().expect("valid"),
                8,
                "{shape:?} B fragment must be 8 regs"
            );
        }
    }
}
