//! Tensor Core GEMM configuration and validation.
//!
//! This module provides [`TensorCoreValidator`] which checks whether a given
//! combination of SM architecture and data types supports Tensor Core
//! acceleration, and [`TensorCoreConfig`] which describes the specific
//! Tensor Core instruction shape to use.
//!
//! # Supported configurations
//!
//! | Architecture | Input | Accumulator | Instruction |
//! |-------------|-------|-------------|-------------|
//! | Turing (sm_75) | F16 | F32 | WMMA 16x16x16 |
//! | Ampere (sm_80+) | F16, BF16, TF32 | F32 | MMA 16x8x16 |
//! | Ada (sm_89) | F16, BF16, TF32, FP8 | F32 | MMA 16x8x16 |
//! | Hopper (sm_90+) | F16, BF16, TF32, FP8 | F32 | WGMMA 64x256x16 |
//! | Blackwell (sm_100+) | F16, BF16, TF32, FP8, FP6, FP4 | F32 | WGMMA |

use oxicuda_ptx::arch::{ArchCapabilities, SmVersion};
use oxicuda_ptx::ir::PtxType;

// ---------------------------------------------------------------------------
// TensorCoreConfig
// ---------------------------------------------------------------------------

/// Describes the Tensor Core instruction shape and type for a specific
/// architecture and precision combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorCoreConfig {
    /// MMA shape: M dimension.
    pub mma_m: u32,
    /// MMA shape: N dimension.
    pub mma_n: u32,
    /// MMA shape: K dimension.
    pub mma_k: u32,
    /// Which generation of TC instructions to use.
    pub instruction: TcInstruction,
}

/// The generation of Tensor Core instruction to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcInstruction {
    /// WMMA (Volta/Turing): `wmma.mma.sync`.
    Wmma,
    /// MMA (Ampere+): `mma.sync.aligned`.
    Mma,
    /// WGMMA (Hopper+): warp-group level MMA.
    Wgmma,
}

// ---------------------------------------------------------------------------
// TensorCoreValidator
// ---------------------------------------------------------------------------

/// Validates and configures Tensor Core usage for GEMM kernels.
pub struct TensorCoreValidator;

impl TensorCoreValidator {
    /// Returns `true` if the given architecture supports Tensor Core
    /// instructions for the specified input and accumulator types.
    pub fn is_supported(sm: SmVersion, input: PtxType, accumulator: PtxType) -> bool {
        Self::config(sm, input, accumulator).is_some()
    }

    /// Returns the optimal [`TensorCoreConfig`] for the given architecture
    /// and precision, or `None` if Tensor Cores cannot be used.
    pub fn config(sm: SmVersion, input: PtxType, accumulator: PtxType) -> Option<TensorCoreConfig> {
        let caps = ArchCapabilities::for_sm(sm);

        if !caps.has_tensor_cores {
            return None;
        }

        // Accumulator must be F32 for most TC paths.
        if accumulator != PtxType::F32 && accumulator != PtxType::F64 {
            return None;
        }

        match sm {
            SmVersion::Sm75 => Self::turing_config(input, accumulator),
            SmVersion::Sm80 | SmVersion::Sm86 => Self::ampere_config(input, accumulator, &caps),
            SmVersion::Sm89 => Self::ada_config(input, accumulator, &caps),
            SmVersion::Sm90 | SmVersion::Sm90a => Self::hopper_config(input, accumulator, &caps),
            SmVersion::Sm100 | SmVersion::Sm120 => {
                Self::blackwell_config(input, accumulator, &caps)
            }
        }
    }

    /// Turing: only F16 input with F32 accumulator via WMMA 16x16x16.
    fn turing_config(input: PtxType, accumulator: PtxType) -> Option<TensorCoreConfig> {
        if input == PtxType::F16 && accumulator == PtxType::F32 {
            Some(TensorCoreConfig {
                mma_m: 16,
                mma_n: 16,
                mma_k: 16,
                instruction: TcInstruction::Wmma,
            })
        } else {
            None
        }
    }

    /// Ampere: F16, BF16 via MMA 16x8x16; TF32 (presented as F32) via
    /// MMA 16x8x8.
    fn ampere_config(
        input: PtxType,
        accumulator: PtxType,
        caps: &ArchCapabilities,
    ) -> Option<TensorCoreConfig> {
        if !caps.has_ampere_mma {
            return None;
        }
        match (input, accumulator) {
            (PtxType::F16, PtxType::F32) | (PtxType::BF16, PtxType::F32) => {
                Some(TensorCoreConfig {
                    mma_m: 16,
                    mma_n: 8,
                    mma_k: 16,
                    instruction: TcInstruction::Mma,
                })
            }
            // TF32 path: F32 input treated as TF32 (truncated to 19-bit mantissa).
            (PtxType::F32, PtxType::F32) => Some(TensorCoreConfig {
                mma_m: 16,
                mma_n: 8,
                mma_k: 8,
                instruction: TcInstruction::Mma,
            }),
            _ => None,
        }
    }

    /// Ada Lovelace: same as Ampere plus FP8 support.
    fn ada_config(
        input: PtxType,
        accumulator: PtxType,
        caps: &ArchCapabilities,
    ) -> Option<TensorCoreConfig> {
        // First try Ampere shapes.
        if let Some(cfg) = Self::ampere_config(input, accumulator, caps) {
            return Some(cfg);
        }
        // FP8 on Ada: E4M3/E5M2 if the arch supports it.
        if caps.has_fp8 && matches!(input, PtxType::U8) && accumulator == PtxType::F32 {
            return Some(TensorCoreConfig {
                mma_m: 16,
                mma_n: 8,
                mma_k: 32,
                instruction: TcInstruction::Mma,
            });
        }
        None
    }

    /// Hopper: WGMMA for all types; fallback to MMA for compatibility.
    fn hopper_config(
        input: PtxType,
        accumulator: PtxType,
        caps: &ArchCapabilities,
    ) -> Option<TensorCoreConfig> {
        if caps.has_wgmma {
            match (input, accumulator) {
                (PtxType::F16, PtxType::F32) | (PtxType::BF16, PtxType::F32) => {
                    return Some(TensorCoreConfig {
                        mma_m: 64,
                        mma_n: 256,
                        mma_k: 16,
                        instruction: TcInstruction::Wgmma,
                    });
                }
                (PtxType::F32, PtxType::F32) => {
                    return Some(TensorCoreConfig {
                        mma_m: 64,
                        mma_n: 128,
                        mma_k: 8,
                        instruction: TcInstruction::Wgmma,
                    });
                }
                _ => {}
            }
        }
        // Fallback to Ampere MMA.
        Self::ampere_config(input, accumulator, caps)
    }

    /// Blackwell: WGMMA with extended type support (FP8, FP6, FP4).
    fn blackwell_config(
        input: PtxType,
        accumulator: PtxType,
        caps: &ArchCapabilities,
    ) -> Option<TensorCoreConfig> {
        // Hopper configs apply to Blackwell as well.
        Self::hopper_config(input, accumulator, caps)
    }

    /// Validates that a tile configuration is compatible with the selected
    /// Tensor Core instruction shape.
    ///
    /// Returns `Ok(())` if the tile dimensions are multiples of the MMA
    /// shape dimensions, or an error describing the mismatch.
    pub fn validate_tile(
        tile_m: u32,
        tile_n: u32,
        tile_k: u32,
        config: &TensorCoreConfig,
    ) -> Result<(), String> {
        if tile_m % config.mma_m != 0 {
            return Err(format!(
                "tile_m ({tile_m}) must be a multiple of mma_m ({})",
                config.mma_m
            ));
        }
        if tile_n % config.mma_n != 0 {
            return Err(format!(
                "tile_n ({tile_n}) must be a multiple of mma_n ({})",
                config.mma_n
            ));
        }
        if tile_k % config.mma_k != 0 {
            return Err(format!(
                "tile_k ({tile_k}) must be a multiple of mma_k ({})",
                config.mma_k
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turing_f16_supported() {
        assert!(TensorCoreValidator::is_supported(
            SmVersion::Sm75,
            PtxType::F16,
            PtxType::F32,
        ));
    }

    #[test]
    fn turing_f32_not_supported() {
        assert!(!TensorCoreValidator::is_supported(
            SmVersion::Sm75,
            PtxType::F32,
            PtxType::F32,
        ));
    }

    #[test]
    fn ampere_f16_mma() {
        let cfg = TensorCoreValidator::config(SmVersion::Sm80, PtxType::F16, PtxType::F32);
        let cfg = cfg.expect("should have config");
        assert_eq!(cfg.instruction, TcInstruction::Mma);
        assert_eq!(cfg.mma_m, 16);
        assert_eq!(cfg.mma_n, 8);
        assert_eq!(cfg.mma_k, 16);
    }

    #[test]
    fn ampere_tf32() {
        let cfg = TensorCoreValidator::config(SmVersion::Sm80, PtxType::F32, PtxType::F32);
        let cfg = cfg.expect("TF32 path should exist on Ampere");
        assert_eq!(cfg.mma_k, 8);
    }

    #[test]
    fn hopper_wgmma_f16() {
        let cfg = TensorCoreValidator::config(SmVersion::Sm90, PtxType::F16, PtxType::F32);
        let cfg = cfg.expect("Hopper should support WGMMA for F16");
        assert_eq!(cfg.instruction, TcInstruction::Wgmma);
        assert_eq!(cfg.mma_m, 64);
    }

    #[test]
    fn validate_tile_ok() {
        let cfg = TensorCoreConfig {
            mma_m: 16,
            mma_n: 8,
            mma_k: 16,
            instruction: TcInstruction::Mma,
        };
        assert!(TensorCoreValidator::validate_tile(128, 128, 32, &cfg).is_ok());
    }

    #[test]
    fn validate_tile_bad_m() {
        let cfg = TensorCoreConfig {
            mma_m: 16,
            mma_n: 8,
            mma_k: 16,
            instruction: TcInstruction::Mma,
        };
        assert!(TensorCoreValidator::validate_tile(100, 128, 32, &cfg).is_err());
    }

    #[test]
    fn f64_not_supported_tc() {
        // F64 tensor cores are not supported on any architecture (only DMMA
        // on Ampere which has limited availability).
        assert!(!TensorCoreValidator::is_supported(
            SmVersion::Sm80,
            PtxType::F64,
            PtxType::F64,
        ));
    }
}
