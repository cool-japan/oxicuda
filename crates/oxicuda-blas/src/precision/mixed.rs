//! Mixed precision GEMM configurations.
//!
//! Mixed precision GEMM uses one data type for inputs (A, B matrices) and
//! a wider type for the accumulator (C matrix). This is the dominant
//! pattern in deep learning:
//!
//! - **FP16 input, FP32 accumulator** — standard mixed-precision training
//! - **BF16 input, FP32 accumulator** — preferred for large language models
//! - **FP8 input, FP32 accumulator** — inference on Hopper+
//!
//! The [`MixedPrecisionConfig`] struct provides validation and tile
//! selection for all supported input/accumulator combinations.

use oxicuda_ptx::prelude::{PtxType, SmVersion};

use crate::error::{BlasError, BlasResult};
use crate::level3::gemm::dispatch::TileConfig;

/// Mixed precision GEMM configuration.
///
/// Maps input precision + accumulator precision to optimal kernel config.
/// The accumulator precision determines the output (C) matrix type.
pub struct MixedPrecisionConfig;

impl MixedPrecisionConfig {
    /// Check if a given input/accumulator combination is supported.
    ///
    /// Support depends on both the combination and the target architecture.
    #[must_use]
    pub fn is_supported(input: PtxType, acc: PtxType, sm: SmVersion) -> bool {
        match (input, acc) {
            // FP16 input, FP32 accumulator — most common, all Tensor Core arches
            (PtxType::F16, PtxType::F32) => sm >= SmVersion::Sm75,
            // FP16 input, FP16 accumulator — pure half-precision
            (PtxType::F16, PtxType::F16) => sm >= SmVersion::Sm75,
            // BF16 input, FP32 accumulator — Ampere+
            (PtxType::BF16, PtxType::F32) => sm >= SmVersion::Sm80,
            // Pure FP32 — always supported
            (PtxType::F32, PtxType::F32) => true,
            // TF32 input (FP32 storage), FP32 accumulator — Ampere+
            (PtxType::TF32, PtxType::F32) => sm >= SmVersion::Sm80,
            // Pure FP64 — always supported
            (PtxType::F64, PtxType::F64) => true,
            // FP8 E4M3 input, FP32 accumulator — Ada Lovelace+
            (PtxType::E4M3, PtxType::F32) => sm >= SmVersion::Sm89,
            // FP8 E5M2 input, FP32 accumulator — Ada Lovelace+
            (PtxType::E5M2, PtxType::F32) => sm >= SmVersion::Sm89,
            // FP8 E4M3 input, FP16 accumulator — Hopper+
            (PtxType::E4M3, PtxType::F16) => sm >= SmVersion::Sm90,
            // FP8 E5M2 input, FP16 accumulator — Hopper+
            (PtxType::E5M2, PtxType::F16) => sm >= SmVersion::Sm90,
            _ => false,
        }
    }

    /// Get optimal tile configuration for a mixed-precision combination.
    ///
    /// Returns `None` if the combination is not supported on the given
    /// architecture.
    #[must_use]
    pub fn optimal_config(input: PtxType, acc: PtxType, sm: SmVersion) -> Option<TileConfig> {
        if !Self::is_supported(input, acc, sm) {
            return None;
        }

        Some(match (input, acc) {
            (PtxType::F16, PtxType::F32) | (PtxType::F16, PtxType::F16) => Self::f16_config(sm),
            (PtxType::BF16, PtxType::F32) => Self::bf16_config(sm),
            (PtxType::TF32, PtxType::F32) => Self::tf32_config(sm),
            (PtxType::F32, PtxType::F32) => Self::f32_config(sm),
            (PtxType::F64, PtxType::F64) => Self::f64_config(sm),
            (PtxType::E4M3, PtxType::F32)
            | (PtxType::E5M2, PtxType::F32)
            | (PtxType::E4M3, PtxType::F16)
            | (PtxType::E5M2, PtxType::F16) => Self::fp8_config(sm),
            _ => return None,
        })
    }

    /// Validate a mixed-precision configuration for the given architecture
    /// and dimensions.
    pub fn validate(
        input: PtxType,
        acc: PtxType,
        sm: SmVersion,
        m: u32,
        n: u32,
        k: u32,
    ) -> BlasResult<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "mixed-precision GEMM dimensions must be non-zero: m={m}, n={n}, k={k}"
            )));
        }
        if !Self::is_supported(input, acc, sm) {
            return Err(BlasError::UnsupportedOperation(format!(
                "mixed precision {input:?} -> {acc:?} not supported on {sm}"
            )));
        }
        Ok(())
    }

    /// Return the element size in bytes for the input type.
    #[must_use]
    pub fn input_element_bytes(input: PtxType) -> Option<u32> {
        match input {
            PtxType::E4M3 | PtxType::E5M2 => Some(1),
            PtxType::F16 | PtxType::BF16 => Some(2),
            PtxType::F32 | PtxType::TF32 => Some(4),
            PtxType::F64 => Some(8),
            _ => None,
        }
    }

    /// Return the element size in bytes for the accumulator type.
    #[must_use]
    pub fn accumulator_element_bytes(acc: PtxType) -> Option<u32> {
        match acc {
            PtxType::F16 => Some(2),
            PtxType::F32 => Some(4),
            PtxType::F64 => Some(8),
            _ => None,
        }
    }

    // -- Internal tile-config helpers ------------------------------------------

    fn f16_config(sm: SmVersion) -> TileConfig {
        match sm {
            SmVersion::Sm90 | SmVersion::Sm90a | SmVersion::Sm100 | SmVersion::Sm120 => {
                TileConfig {
                    tile_m: 128,
                    tile_n: 256,
                    tile_k: 64,
                    warp_m: 64,
                    warp_n: 128,
                    stages: 4,
                    use_tensor_core: true,
                    split_k: 1,
                }
            }
            SmVersion::Sm80 | SmVersion::Sm86 | SmVersion::Sm89 => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
            _ => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 32,
                warp_n: 32,
                stages: 2,
                use_tensor_core: true,
                split_k: 1,
            },
        }
    }

    fn bf16_config(sm: SmVersion) -> TileConfig {
        match sm {
            SmVersion::Sm90 | SmVersion::Sm90a | SmVersion::Sm100 | SmVersion::Sm120 => {
                TileConfig {
                    tile_m: 128,
                    tile_n: 256,
                    tile_k: 64,
                    warp_m: 64,
                    warp_n: 128,
                    stages: 4,
                    use_tensor_core: true,
                    split_k: 1,
                }
            }
            _ => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
        }
    }

    fn tf32_config(sm: SmVersion) -> TileConfig {
        match sm {
            SmVersion::Sm90 | SmVersion::Sm90a | SmVersion::Sm100 | SmVersion::Sm120 => {
                TileConfig {
                    tile_m: 128,
                    tile_n: 128,
                    tile_k: 32,
                    warp_m: 64,
                    warp_n: 64,
                    stages: 4,
                    use_tensor_core: true,
                    split_k: 1,
                }
            }
            _ => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 16,
                warp_m: 64,
                warp_n: 64,
                stages: 3,
                use_tensor_core: true,
                split_k: 1,
            },
        }
    }

    fn f32_config(sm: SmVersion) -> TileConfig {
        match sm {
            SmVersion::Sm90 | SmVersion::Sm90a | SmVersion::Sm100 | SmVersion::Sm120 => {
                TileConfig {
                    tile_m: 128,
                    tile_n: 128,
                    tile_k: 32,
                    warp_m: 64,
                    warp_n: 64,
                    stages: 4,
                    use_tensor_core: false,
                    split_k: 1,
                }
            }
            _ => TileConfig {
                tile_m: 128,
                tile_n: 64,
                tile_k: 16,
                warp_m: 32,
                warp_n: 32,
                stages: 2,
                use_tensor_core: false,
                split_k: 1,
            },
        }
    }

    fn f64_config(sm: SmVersion) -> TileConfig {
        match sm {
            SmVersion::Sm90 | SmVersion::Sm90a | SmVersion::Sm100 | SmVersion::Sm120 => {
                TileConfig {
                    tile_m: 64,
                    tile_n: 64,
                    tile_k: 16,
                    warp_m: 32,
                    warp_n: 32,
                    stages: 4,
                    use_tensor_core: true,
                    split_k: 1,
                }
            }
            SmVersion::Sm80 | SmVersion::Sm86 => TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 16,
                warp_m: 32,
                warp_n: 32,
                stages: 3,
                use_tensor_core: true,
                split_k: 1,
            },
            _ => TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 8,
                warp_m: 32,
                warp_n: 32,
                stages: 2,
                use_tensor_core: false,
                split_k: 1,
            },
        }
    }

    fn fp8_config(sm: SmVersion) -> TileConfig {
        match sm {
            SmVersion::Sm90 | SmVersion::Sm90a => TileConfig {
                tile_m: 128,
                tile_n: 256,
                tile_k: 128,
                warp_m: 64,
                warp_n: 128,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
            SmVersion::Sm100 | SmVersion::Sm120 => TileConfig {
                tile_m: 256,
                tile_n: 256,
                tile_k: 128,
                warp_m: 64,
                warp_n: 128,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
            _ => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 64,
                warp_m: 64,
                warp_n: 64,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_f32_supported_everywhere() {
        assert!(MixedPrecisionConfig::is_supported(
            PtxType::F16,
            PtxType::F32,
            SmVersion::Sm75
        ));
        assert!(MixedPrecisionConfig::is_supported(
            PtxType::F16,
            PtxType::F32,
            SmVersion::Sm90
        ));
    }

    #[test]
    fn bf16_requires_ampere() {
        assert!(!MixedPrecisionConfig::is_supported(
            PtxType::BF16,
            PtxType::F32,
            SmVersion::Sm75
        ));
        assert!(MixedPrecisionConfig::is_supported(
            PtxType::BF16,
            PtxType::F32,
            SmVersion::Sm80
        ));
    }

    #[test]
    fn fp8_requires_ada() {
        assert!(!MixedPrecisionConfig::is_supported(
            PtxType::E4M3,
            PtxType::F32,
            SmVersion::Sm86
        ));
        assert!(MixedPrecisionConfig::is_supported(
            PtxType::E4M3,
            PtxType::F32,
            SmVersion::Sm89
        ));
    }

    #[test]
    fn fp8_f16_acc_requires_hopper() {
        assert!(!MixedPrecisionConfig::is_supported(
            PtxType::E4M3,
            PtxType::F16,
            SmVersion::Sm89
        ));
        assert!(MixedPrecisionConfig::is_supported(
            PtxType::E4M3,
            PtxType::F16,
            SmVersion::Sm90
        ));
    }

    #[test]
    fn unsupported_combo_returns_none() {
        assert!(
            MixedPrecisionConfig::optimal_config(PtxType::BF16, PtxType::F64, SmVersion::Sm90)
                .is_none()
        );
    }

    #[test]
    fn optimal_config_f16_f32_hopper() {
        let cfg = MixedPrecisionConfig::optimal_config(PtxType::F16, PtxType::F32, SmVersion::Sm90);
        assert!(cfg.is_some());
        let cfg = cfg.unwrap_or_else(|| unreachable!());
        assert!(cfg.use_tensor_core);
        assert_eq!(cfg.tile_n, 256);
    }

    #[test]
    fn validate_ok() {
        assert!(
            MixedPrecisionConfig::validate(
                PtxType::F16,
                PtxType::F32,
                SmVersion::Sm80,
                128,
                128,
                128
            )
            .is_ok()
        );
    }

    #[test]
    fn validate_zero_dim() {
        assert!(
            MixedPrecisionConfig::validate(
                PtxType::F16,
                PtxType::F32,
                SmVersion::Sm80,
                0,
                128,
                128
            )
            .is_err()
        );
    }

    #[test]
    fn validate_unsupported_combo() {
        assert!(
            MixedPrecisionConfig::validate(
                PtxType::BF16,
                PtxType::F32,
                SmVersion::Sm75,
                128,
                128,
                128
            )
            .is_err()
        );
    }

    #[test]
    fn input_element_bytes() {
        assert_eq!(
            MixedPrecisionConfig::input_element_bytes(PtxType::E4M3),
            Some(1)
        );
        assert_eq!(
            MixedPrecisionConfig::input_element_bytes(PtxType::F16),
            Some(2)
        );
        assert_eq!(
            MixedPrecisionConfig::input_element_bytes(PtxType::F32),
            Some(4)
        );
        assert_eq!(
            MixedPrecisionConfig::input_element_bytes(PtxType::F64),
            Some(8)
        );
    }

    #[test]
    fn accumulator_element_bytes() {
        assert_eq!(
            MixedPrecisionConfig::accumulator_element_bytes(PtxType::F16),
            Some(2)
        );
        assert_eq!(
            MixedPrecisionConfig::accumulator_element_bytes(PtxType::F32),
            Some(4)
        );
        assert_eq!(
            MixedPrecisionConfig::accumulator_element_bytes(PtxType::F64),
            Some(8)
        );
        assert_eq!(
            MixedPrecisionConfig::accumulator_element_bytes(PtxType::U32),
            None
        );
    }

    #[test]
    fn f32_f32_always_supported() {
        assert!(MixedPrecisionConfig::is_supported(
            PtxType::F32,
            PtxType::F32,
            SmVersion::Sm75
        ));
        let cfg = MixedPrecisionConfig::optimal_config(PtxType::F32, PtxType::F32, SmVersion::Sm75);
        assert!(cfg.is_some());
    }
}
