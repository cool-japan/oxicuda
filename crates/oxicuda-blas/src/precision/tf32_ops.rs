//! TF32 mode: FP32 inputs truncated to TF32 for Ampere+ Tensor Cores.
//!
//! TF32 (TensorFloat-32) is not a distinct storage format but a Tensor
//! Core execution mode where FP32 inputs are internally truncated to
//! 19 bits (8-bit exponent + 10-bit mantissa) before being processed by
//! MMA hardware. The accumulator remains full FP32.
//!
//! This gives roughly 8x the throughput of standard FP32 FMA on Ampere
//! with minimal precision loss for most deep-learning and HPC workloads.

use oxicuda_ptx::prelude::SmVersion;

use crate::error::{BlasError, BlasResult};
use crate::level3::gemm::dispatch::TileConfig;

/// TF32 GEMM configuration.
///
/// TF32 is available from Ampere (sm_80). The input and output are standard
/// FP32 buffers; only the internal compute path uses TF32 truncation.
pub struct Tf32Config;

impl Tf32Config {
    /// Element size in bytes (same as FP32 since TF32 is a compute mode).
    pub const ELEMENT_BYTES: u32 = 4;

    /// Get optimal tile config for TF32 GEMM.
    ///
    /// TF32 uses Tensor Cores, so the tile sizes are larger than pure FP32
    /// and the pipeline depths are deeper.
    #[must_use]
    pub fn tile_config(sm: SmVersion, _m: u32, _n: u32, _k: u32) -> TileConfig {
        match sm {
            SmVersion::Sm80 | SmVersion::Sm86 => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 16,
                warp_m: 64,
                warp_n: 64,
                stages: 3,
                use_tensor_core: true,
                split_k: 1,
            },
            SmVersion::Sm89 => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 16,
                warp_m: 64,
                warp_n: 64,
                stages: 3,
                use_tensor_core: true,
                split_k: 1,
            },
            SmVersion::Sm90 | SmVersion::Sm90a => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
            SmVersion::Sm100 | SmVersion::Sm120 => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
            _ => {
                // TF32 is not available on pre-Ampere; return a pure FP32 config.
                TileConfig {
                    tile_m: 128,
                    tile_n: 64,
                    tile_k: 16,
                    warp_m: 32,
                    warp_n: 32,
                    stages: 2,
                    use_tensor_core: false,
                    split_k: 1,
                }
            }
        }
    }

    /// Check if TF32 Tensor Core mode is available.
    #[must_use]
    pub fn is_available(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm80
    }

    /// Validate dimensions and architecture for TF32 GEMM.
    ///
    /// Returns an error if dimensions are zero or if the target architecture
    /// does not support TF32.
    pub fn validate(sm: SmVersion, m: u32, n: u32, k: u32) -> BlasResult<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "TF32 GEMM dimensions must be non-zero: m={m}, n={n}, k={k}"
            )));
        }
        if !Self::is_available(sm) {
            return Err(BlasError::UnsupportedOperation(format!(
                "TF32 Tensor Core mode requires Ampere+ (sm_80), got {sm}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tf32_not_available_on_turing() {
        assert!(!Tf32Config::is_available(SmVersion::Sm75));
    }

    #[test]
    fn tf32_available_on_ampere() {
        assert!(Tf32Config::is_available(SmVersion::Sm80));
        assert!(Tf32Config::is_available(SmVersion::Sm86));
    }

    #[test]
    fn tile_config_ampere() {
        let cfg = Tf32Config::tile_config(SmVersion::Sm80, 1024, 1024, 1024);
        assert!(cfg.use_tensor_core);
        assert_eq!(cfg.tile_k, 16);
    }

    #[test]
    fn tile_config_turing_fallback() {
        let cfg = Tf32Config::tile_config(SmVersion::Sm75, 512, 512, 512);
        assert!(!cfg.use_tensor_core);
    }

    #[test]
    fn validate_ok() {
        assert!(Tf32Config::validate(SmVersion::Sm80, 128, 256, 64).is_ok());
    }

    #[test]
    fn validate_unsupported_arch() {
        let err = Tf32Config::validate(SmVersion::Sm75, 128, 128, 128);
        assert!(err.is_err());
    }

    #[test]
    fn validate_zero_dim() {
        assert!(Tf32Config::validate(SmVersion::Sm80, 0, 128, 128).is_err());
    }
}
