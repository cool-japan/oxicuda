//! MMA (Matrix Multiply-Accumulate) instruction configuration helpers.
//!
//! The `mma.sync.aligned` instruction family covers a range of shapes and data
//! types spanning from Turing (`sm_75`) through Hopper (`sm_90`):
//!
//! | Shape    | A/B types                          | Accum.     | Architecture |
//! |----------|------------------------------------|------------|--------------|
//! | 16x8x8   | F16; **TF32** (Ampere+)            | F16, F32   | Volta/Turing |
//! | 16x8x16  | F16, BF16; **S8/U8** (Ampere+)     | F32, S32   | Ampere+      |
//! | 16x8x32  | F16, BF16, E4M3, E5M2; **S8/U8**  | F32, S32   | Hopper+/A+   |
//! | 8x8x16   | **S8, U8** (Turing+)               | S32        | Turing+      |
//! | 8x8x32   | **S4, U4** (Turing+)               | S32        | Turing+      |

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::{MmaShape, PtxType};

// ─── MmaConfig ───────────────────────────────────────────────────────────────

/// MMA instruction configuration.
///
/// Defines the shape and type parameters for an `mma.sync.aligned` instruction.
/// Use [`validate`](MmaConfig::validate) to check type compatibility, and
/// [`check_arch_support`](MmaConfig::check_arch_support) for architecture gating.
#[derive(Debug, Clone)]
pub struct MmaConfig {
    /// Matrix tile shape.
    pub shape: MmaShape,
    /// Element type of matrix A.
    pub a_type: PtxType,
    /// Element type of matrix B.
    pub b_type: PtxType,
    /// Accumulator (C/D) element type.
    pub accumulator: PtxType,
}

impl MmaConfig {
    /// Creates a new MMA configuration.
    #[must_use]
    pub const fn new(
        shape: MmaShape,
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

    /// Validates type constraints for m16n8k8 (F16/TF32).
    fn validate_m16n8k8(&self) -> Result<(), PtxGenError> {
        match self.a_type {
            PtxType::F16 => {
                if !matches!(self.accumulator, PtxType::F16 | PtxType::F32) {
                    return Err(PtxGenError::InvalidType(format!(
                        "m16n8k8 F16: accumulator must be F16 or F32, got {}",
                        self.accumulator.as_ptx_str()
                    )));
                }
            }
            PtxType::TF32 => {
                if self.accumulator != PtxType::F32 {
                    return Err(PtxGenError::InvalidType(format!(
                        "m16n8k8 TF32: accumulator must be F32, got {}",
                        self.accumulator.as_ptx_str()
                    )));
                }
            }
            other => {
                return Err(PtxGenError::InvalidType(format!(
                    "m16n8k8 requires F16 or TF32 A/B, got {}",
                    other.as_ptx_str()
                )));
            }
        }
        Ok(())
    }

    /// Validates type constraints for a specific shape.
    ///
    /// Called by [`validate`](MmaConfig::validate) after the A/B type match check.
    fn validate_shape_types(&self) -> Result<(), PtxGenError> {
        match self.shape {
            // ── m16n8k8: F16 (Volta/Turing) or TF32 (Ampere+) ──────────────
            MmaShape::M16N8K8 => self.validate_m16n8k8()?,
            // ── m16n8k16: F16/BF16 (Ampere+) or S8/U8 INT8 (Ampere+) ───────
            MmaShape::M16N8K16 => match self.a_type {
                PtxType::F16 | PtxType::BF16 => {
                    if self.accumulator != PtxType::F32 {
                        return Err(PtxGenError::InvalidType(format!(
                            "m16n8k16 F16/BF16: accumulator must be F32, got {}",
                            self.accumulator.as_ptx_str()
                        )));
                    }
                }
                PtxType::S8 | PtxType::U8 => {
                    if self.accumulator != PtxType::S32 {
                        return Err(PtxGenError::InvalidType(format!(
                            "m16n8k16 S8/U8: accumulator must be S32, got {}",
                            self.accumulator.as_ptx_str()
                        )));
                    }
                }
                other => {
                    return Err(PtxGenError::InvalidType(format!(
                        "m16n8k16 requires F16, BF16, S8, or U8 A/B, got {}",
                        other.as_ptx_str()
                    )));
                }
            },
            // ── m16n8k32: FP/FP8 (Hopper+) or S8/U8 INT8 (Ampere+) ─────────
            MmaShape::M16N8K32 => match self.a_type {
                PtxType::F16 | PtxType::BF16 | PtxType::E4M3 | PtxType::E5M2 => {
                    if self.accumulator != PtxType::F32 {
                        return Err(PtxGenError::InvalidType(format!(
                            "m16n8k32 FP: accumulator must be F32, got {}",
                            self.accumulator.as_ptx_str()
                        )));
                    }
                }
                PtxType::S8 | PtxType::U8 => {
                    if self.accumulator != PtxType::S32 {
                        return Err(PtxGenError::InvalidType(format!(
                            "m16n8k32 S8/U8: accumulator must be S32, got {}",
                            self.accumulator.as_ptx_str()
                        )));
                    }
                }
                other => {
                    return Err(PtxGenError::InvalidType(format!(
                        "m16n8k32 requires F16, BF16, E4M3, E5M2, S8, or U8 A/B, got {}",
                        other.as_ptx_str()
                    )));
                }
            },
            // ── m8n8k16/m8n8k32: INT8/INT4 IMMA — Turing+ (`sm_75`) ─────────
            // S4/U4 are not first-class PtxType variants (sub-byte types packed
            // into B32 registers); S8/U8 serve as the register carrier type.
            MmaShape::M8N8K16 | MmaShape::M8N8K32 => {
                if !matches!(self.a_type, PtxType::S8 | PtxType::U8) {
                    return Err(PtxGenError::InvalidType(format!(
                        "{:?} requires S8 or U8 A/B (IMMA), got {}",
                        self.shape,
                        self.a_type.as_ptx_str()
                    )));
                }
                if self.accumulator != PtxType::S32 {
                    return Err(PtxGenError::InvalidType(format!(
                        "{:?} accumulator must be S32 (IMMA), got {}",
                        self.shape,
                        self.accumulator.as_ptx_str()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validates that the type/shape combination is supported by the PTX ISA.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if:
    /// - A and B types differ (all MMA variants require matching A/B types).
    /// - The type/shape combination is not defined in the PTX ISA.
    pub fn validate(&self) -> Result<(), PtxGenError> {
        // Rule 1 — A and B types must match.
        if self.a_type != self.b_type {
            return Err(PtxGenError::InvalidType(format!(
                "MMA requires matching A/B types, got A={}, B={}",
                self.a_type.as_ptx_str(),
                self.b_type.as_ptx_str()
            )));
        }
        self.validate_shape_types()
    }

    /// Checks whether this MMA configuration is supported on the given architecture.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::UnsupportedFeature`] if the target arch does not
    /// support the required MMA shape/type combination.
    pub fn check_arch_support(&self, sm: SmVersion) -> Result<(), PtxGenError> {
        let caps = sm.capabilities();

        match self.shape {
            MmaShape::M16N8K8 => {
                if !caps.has_tensor_cores {
                    return Err(PtxGenError::UnsupportedFeature {
                        arch: sm.as_ptx_str().to_string(),
                        feature: "mma.sync m16n8k8 (tensor cores)".to_string(),
                    });
                }
                // TF32 variant requires Ampere (SM 80+).
                if self.a_type == PtxType::TF32 && !caps.has_ampere_mma {
                    return Err(PtxGenError::UnsupportedFeature {
                        arch: sm.as_ptx_str().to_string(),
                        feature: "mma.sync m16n8k8.tf32 (Ampere+)".to_string(),
                    });
                }
            }

            MmaShape::M16N8K16 => {
                if !caps.has_ampere_mma {
                    return Err(PtxGenError::UnsupportedFeature {
                        arch: sm.as_ptx_str().to_string(),
                        feature: "mma.sync m16n8k16 (Ampere+)".to_string(),
                    });
                }
            }

            MmaShape::M16N8K32 => {
                // FP types (F16/BF16/E4M3/E5M2) need Hopper; INT8 only Ampere.
                let is_int8 = matches!(self.a_type, PtxType::S8 | PtxType::U8);
                if is_int8 {
                    if !caps.has_ampere_mma {
                        return Err(PtxGenError::UnsupportedFeature {
                            arch: sm.as_ptx_str().to_string(),
                            feature: "mma.sync m16n8k32.s8 (Ampere+)".to_string(),
                        });
                    }
                } else if sm < SmVersion::Sm90 {
                    return Err(PtxGenError::UnsupportedFeature {
                        arch: sm.as_ptx_str().to_string(),
                        feature: "mma.sync m16n8k32 FP/FP8 (Hopper+)".to_string(),
                    });
                }
            }

            MmaShape::M8N8K16 | MmaShape::M8N8K32 => {
                // INT8/INT4 IMMA require Turing (SM 75+).
                if !caps.has_tensor_cores {
                    return Err(PtxGenError::UnsupportedFeature {
                        arch: sm.as_ptx_str().to_string(),
                        feature: "mma.sync m8n8k16/m8n8k32 INT8/INT4 IMMA (Turing+)".to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Returns the number of registers per thread for the A matrix operand.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn regs_per_thread_a(&self) -> Result<u32, PtxGenError> {
        self.validate()?;
        Ok(match self.shape {
            MmaShape::M16N8K8 => 2,  // 2 × B32/thread (F16/TF32)
            MmaShape::M16N8K16 => 4, // 4 × B32/thread
            MmaShape::M16N8K32 => 8, // 8 × B32/thread
            // 1 × B32/thread: INT8 packed ×4 or INT4 packed ×8
            MmaShape::M8N8K16 | MmaShape::M8N8K32 => 1,
        })
    }

    /// Returns the number of registers per thread for the B matrix operand.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn regs_per_thread_b(&self) -> Result<u32, PtxGenError> {
        self.validate()?;
        Ok(match self.shape {
            MmaShape::M16N8K16 => 2,
            MmaShape::M16N8K32 => 4,
            // M16N8K8, M8N8K16, M8N8K32 all use 1 register per thread for B
            MmaShape::M16N8K8 | MmaShape::M8N8K16 | MmaShape::M8N8K32 => 1,
        })
    }

    /// Returns the number of registers per thread for the accumulator (C/D) operand.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn regs_per_thread_c(&self) -> Result<u32, PtxGenError> {
        self.validate()?;
        Ok(match (self.shape, self.accumulator) {
            // 4 registers: F32 accumulators (m16n8k*) and S32 for wide INT8
            (MmaShape::M16N8K8 | MmaShape::M16N8K16 | MmaShape::M16N8K32, PtxType::F32)
            | (MmaShape::M16N8K16 | MmaShape::M16N8K32, PtxType::S32) => 4,
            // 2 registers: F16 accumulator or S32 for 8-wide IMMA shapes
            (_, PtxType::F16) | (MmaShape::M8N8K16 | MmaShape::M8N8K32, PtxType::S32) => 2,
            (shape, acc) => {
                return Err(PtxGenError::InvalidType(format!(
                    "unsupported (shape={:?}, accumulator={})",
                    shape,
                    acc.as_ptx_str()
                )));
            }
        })
    }

    /// Returns the `(M, N, K)` tile dimensions for this MMA shape.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32, u32) {
        match self.shape {
            MmaShape::M16N8K8 => (16, 8, 8),
            MmaShape::M16N8K16 => (16, 8, 16),
            MmaShape::M16N8K32 => (16, 8, 32),
            MmaShape::M8N8K16 => (8, 8, 16),
            MmaShape::M8N8K32 => (8, 8, 32),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_m16n8k16_f16() {
        let cfg = MmaConfig::new(MmaShape::M16N8K16, PtxType::F16, PtxType::F16, PtxType::F32);
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.regs_per_thread_a().unwrap(), 4);
        assert_eq!(cfg.regs_per_thread_b().unwrap(), 2);
        assert_eq!(cfg.regs_per_thread_c().unwrap(), 4);
    }

    #[test]
    fn valid_m16n8k8_f16() {
        let cfg = MmaConfig::new(MmaShape::M16N8K8, PtxType::F16, PtxType::F16, PtxType::F16);
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.regs_per_thread_c().unwrap(), 2);
    }

    #[test]
    fn valid_m16n8k8_tf32() {
        let cfg = MmaConfig::new(
            MmaShape::M16N8K8,
            PtxType::TF32,
            PtxType::TF32,
            PtxType::F32,
        );
        assert!(cfg.validate().is_ok(), "TF32 m16n8k8 must be valid");
        assert_eq!(cfg.regs_per_thread_a().unwrap(), 2);
        assert_eq!(cfg.regs_per_thread_b().unwrap(), 1);
        assert_eq!(cfg.regs_per_thread_c().unwrap(), 4);
    }

    #[test]
    fn tf32_requires_ampere() {
        let cfg = MmaConfig::new(
            MmaShape::M16N8K8,
            PtxType::TF32,
            PtxType::TF32,
            PtxType::F32,
        );
        assert!(
            cfg.check_arch_support(SmVersion::Sm75).is_err(),
            "TF32 rejected on Turing"
        );
        assert!(
            cfg.check_arch_support(SmVersion::Sm80).is_ok(),
            "TF32 accepted on Ampere"
        );
    }

    #[test]
    fn valid_m16n8k16_bf16() {
        let cfg = MmaConfig::new(
            MmaShape::M16N8K16,
            PtxType::BF16,
            PtxType::BF16,
            PtxType::F32,
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn valid_m8n8k16_s8() {
        let cfg = MmaConfig::new(MmaShape::M8N8K16, PtxType::S8, PtxType::S8, PtxType::S32);
        assert!(cfg.validate().is_ok(), "S8 m8n8k16 INT8 IMMA must be valid");
        assert_eq!(
            cfg.regs_per_thread_a().unwrap(),
            1,
            "A must use 1 reg/thread"
        );
        assert_eq!(
            cfg.regs_per_thread_b().unwrap(),
            1,
            "B must use 1 reg/thread"
        );
        assert_eq!(
            cfg.regs_per_thread_c().unwrap(),
            2,
            "C/D must use 2 S32 regs/thread"
        );
        assert_eq!(cfg.dimensions(), (8, 8, 16));
    }

    #[test]
    fn valid_m8n8k16_u8() {
        let cfg = MmaConfig::new(MmaShape::M8N8K16, PtxType::U8, PtxType::U8, PtxType::S32);
        assert!(cfg.validate().is_ok(), "U8 m8n8k16 INT8 IMMA must be valid");
    }

    #[test]
    fn valid_m8n8k32_s8_int4() {
        let cfg = MmaConfig::new(MmaShape::M8N8K32, PtxType::S8, PtxType::S8, PtxType::S32);
        assert!(
            cfg.validate().is_ok(),
            "m8n8k32 INT4 IMMA (S8 packed) must be valid"
        );
        assert_eq!(cfg.dimensions(), (8, 8, 32));
    }

    #[test]
    fn valid_m16n8k16_s8() {
        let cfg = MmaConfig::new(MmaShape::M16N8K16, PtxType::S8, PtxType::S8, PtxType::S32);
        assert!(
            cfg.validate().is_ok(),
            "S8 m16n8k16 INT8 IMMA must be valid"
        );
        assert_eq!(cfg.regs_per_thread_c().unwrap(), 4);
    }

    #[test]
    fn valid_m16n8k32_s8() {
        let cfg = MmaConfig::new(MmaShape::M16N8K32, PtxType::S8, PtxType::S8, PtxType::S32);
        assert!(cfg.validate().is_ok(), "S8 m16n8k32 INT8 must be valid");
        assert_eq!(cfg.regs_per_thread_c().unwrap(), 4);
    }

    #[test]
    fn int8_m8n8k16_requires_turing() {
        let cfg = MmaConfig::new(MmaShape::M8N8K16, PtxType::S8, PtxType::S8, PtxType::S32);
        // Sm75 is the lowest available variant; use it to represent pre-Turing by
        // checking that M8N8K16 (Turing+) is also rejected on Sm75 is not the test —
        // the real check is that it IS accepted on Sm75. Use Sm75 for "accepted" only:
        // there is no pre-Turing SmVersion, so we skip the "rejected" arm and only
        // verify the positive case below.
        let _ = cfg.check_arch_support(SmVersion::Sm75); // may succeed; that's fine
        assert!(
            cfg.check_arch_support(SmVersion::Sm75).is_ok(),
            "INT8 IMMA accepted on Turing"
        );
    }

    #[test]
    fn int8_m16n8k16_requires_ampere() {
        let cfg = MmaConfig::new(MmaShape::M16N8K16, PtxType::S8, PtxType::S8, PtxType::S32);
        assert!(
            cfg.check_arch_support(SmVersion::Sm75).is_err(),
            "INT8 m16n8k16 rejected on Turing"
        );
        assert!(
            cfg.check_arch_support(SmVersion::Sm80).is_ok(),
            "INT8 m16n8k16 accepted on Ampere"
        );
    }

    #[test]
    fn mismatched_types() {
        let cfg = MmaConfig::new(
            MmaShape::M16N8K16,
            PtxType::F16,
            PtxType::BF16,
            PtxType::F32,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_accumulator_for_m16n8k16_fp() {
        let cfg = MmaConfig::new(MmaShape::M16N8K16, PtxType::F16, PtxType::F16, PtxType::F16);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn arch_support_turing() {
        let cfg = MmaConfig::new(MmaShape::M16N8K8, PtxType::F16, PtxType::F16, PtxType::F32);
        assert!(cfg.check_arch_support(SmVersion::Sm75).is_ok());

        let cfg2 = MmaConfig::new(MmaShape::M16N8K16, PtxType::F16, PtxType::F16, PtxType::F32);
        assert!(cfg2.check_arch_support(SmVersion::Sm75).is_err());
    }

    #[test]
    fn arch_support_ampere() {
        let cfg = MmaConfig::new(MmaShape::M16N8K16, PtxType::F16, PtxType::F16, PtxType::F32);
        assert!(cfg.check_arch_support(SmVersion::Sm80).is_ok());

        // FP variant of m16n8k32 requires Hopper.
        let cfg2 = MmaConfig::new(MmaShape::M16N8K32, PtxType::F16, PtxType::F16, PtxType::F32);
        assert!(cfg2.check_arch_support(SmVersion::Sm80).is_err());
    }

    #[test]
    fn fp8_e4m3_valid_and_requires_sm90() {
        let cfg = MmaConfig::new(
            MmaShape::M16N8K32,
            PtxType::E4M3,
            PtxType::E4M3,
            PtxType::F32,
        );
        assert!(cfg.validate().is_ok());
        assert!(cfg.check_arch_support(SmVersion::Sm80).is_err());
        assert!(cfg.check_arch_support(SmVersion::Sm90).is_ok());
    }

    #[test]
    fn fp8_e5m2_valid() {
        let cfg = MmaConfig::new(
            MmaShape::M16N8K32,
            PtxType::E5M2,
            PtxType::E5M2,
            PtxType::F32,
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn fp8_invalid_for_smaller_shapes() {
        let cfg = MmaConfig::new(
            MmaShape::M16N8K8,
            PtxType::E4M3,
            PtxType::E4M3,
            PtxType::F32,
        );
        assert!(cfg.validate().is_err());

        let cfg = MmaConfig::new(
            MmaShape::M16N8K16,
            PtxType::E4M3,
            PtxType::E4M3,
            PtxType::F32,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn dimensions_all_shapes() {
        assert_eq!(
            MmaConfig::new(MmaShape::M16N8K8, PtxType::F16, PtxType::F16, PtxType::F32)
                .dimensions(),
            (16, 8, 8)
        );
        assert_eq!(
            MmaConfig::new(MmaShape::M16N8K16, PtxType::F16, PtxType::F16, PtxType::F32)
                .dimensions(),
            (16, 8, 16)
        );
        assert_eq!(
            MmaConfig::new(MmaShape::M16N8K32, PtxType::F16, PtxType::F16, PtxType::F32)
                .dimensions(),
            (16, 8, 32)
        );
        assert_eq!(
            MmaConfig::new(MmaShape::M8N8K16, PtxType::S8, PtxType::S8, PtxType::S32).dimensions(),
            (8, 8, 16)
        );
        assert_eq!(
            MmaConfig::new(MmaShape::M8N8K32, PtxType::S8, PtxType::S8, PtxType::S32).dimensions(),
            (8, 8, 32)
        );
    }
}
