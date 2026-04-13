//! FP16 HGEMM configuration and helpers.
//!
//! Half-precision GEMM is the most common Tensor Core workload for deep
//! learning inference and mixed-precision training. On Turing and later,
//! Tensor Core HMMA instructions provide 2-8x the throughput of FP32 FMA
//! depending on the architecture generation.
//!
//! This module is gated behind `#[cfg(feature = "f16")]` because it
//! depends on the `half` crate for the [`half::f16`] type.

use oxicuda_ptx::prelude::SmVersion;

use crate::error::{BlasError, BlasResult};
use crate::level3::gemm::dispatch::TileConfig;

/// FP16 specific GEMM configuration.
///
/// Provides architecture-aware tile selection for half-precision GEMM
/// (HGEMM). All supported architectures (Turing+) have FP16 Tensor Cores.
pub struct F16Config;

impl F16Config {
    /// Element size in bytes for FP16.
    pub const ELEMENT_BYTES: u32 = 2;

    /// Get optimal tile config for FP16 GEMM.
    ///
    /// FP16 benefits from large tiles because each Tensor Core instruction
    /// processes a larger fragment relative to memory traffic. Hopper and
    /// later use WGMMA for even larger effective tiles.
    #[must_use]
    pub fn tile_config(sm: SmVersion, _m: u32, _n: u32, _k: u32) -> TileConfig {
        match sm {
            SmVersion::Sm75 => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 32,
                warp_n: 32,
                stages: 2,
                use_tensor_core: true,
                split_k: 1,
            },
            SmVersion::Sm80 | SmVersion::Sm86 => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
            SmVersion::Sm89 => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
            SmVersion::Sm90 | SmVersion::Sm90a => TileConfig {
                tile_m: 128,
                tile_n: 256,
                tile_k: 64,
                warp_m: 64,
                warp_n: 128,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
            SmVersion::Sm100 | SmVersion::Sm120 => TileConfig {
                tile_m: 128,
                tile_n: 256,
                tile_k: 64,
                warp_m: 64,
                warp_n: 128,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
        }
    }

    /// FP16 Tensor Cores are available on all supported architectures.
    #[must_use]
    pub fn has_tensor_core(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm75
    }

    /// Check if WGMMA (warp-group MMA) is available for FP16.
    ///
    /// WGMMA provides 2x the throughput of standard HMMA by fusing four
    /// warps into a single cooperative group.
    #[must_use]
    pub fn has_wgmma(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm90
    }

    /// Validate dimensions for FP16 GEMM.
    pub fn validate_dimensions(m: u32, n: u32, k: u32) -> BlasResult<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "FP16 GEMM dimensions must be non-zero: m={m}, n={n}, k={k}"
            )));
        }
        Ok(())
    }

    /// Compute the split-K factor for FP16 problems.
    ///
    /// FP16 has higher compute throughput so the threshold for split-K is
    /// proportionally higher.
    #[must_use]
    pub fn compute_split_k(m: u32, n: u32, k: u32) -> u32 {
        if m * n < 4096 && k > 16384 {
            let factor = k / 4096;
            factor.clamp(2, 32)
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_core_all_arches() {
        assert!(F16Config::has_tensor_core(SmVersion::Sm75));
        assert!(F16Config::has_tensor_core(SmVersion::Sm80));
        assert!(F16Config::has_tensor_core(SmVersion::Sm90a));
    }

    #[test]
    fn wgmma_hopper_only() {
        assert!(!F16Config::has_wgmma(SmVersion::Sm80));
        assert!(!F16Config::has_wgmma(SmVersion::Sm89));
        assert!(F16Config::has_wgmma(SmVersion::Sm90));
        assert!(F16Config::has_wgmma(SmVersion::Sm90a));
    }

    #[test]
    fn tile_config_hopper_large_tiles() {
        let cfg = F16Config::tile_config(SmVersion::Sm90a, 4096, 4096, 4096);
        assert_eq!(cfg.tile_n, 256);
        assert_eq!(cfg.tile_k, 64);
        assert!(cfg.use_tensor_core);
    }

    #[test]
    fn validate_zero() {
        assert!(F16Config::validate_dimensions(0, 128, 128).is_err());
    }

    #[test]
    fn validate_ok() {
        assert!(F16Config::validate_dimensions(64, 64, 64).is_ok());
    }

    #[test]
    fn split_k_reduction_heavy() {
        let sk = F16Config::compute_split_k(32, 32, 32768);
        assert!(sk >= 2);
    }
}
