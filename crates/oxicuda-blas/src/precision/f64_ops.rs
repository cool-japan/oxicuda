//! FP64 DGEMM configuration and helpers.
//!
//! FP64 has lower throughput than FP32 on consumer GPUs (typically 1/32
//! or 1/64 of FP32 rate). On data-centre GPUs (A100, H100) Tensor Core
//! DMMA provides significantly better FP64 performance, approaching
//! half the FP32 peak.
//!
//! This module selects optimal tile sizes and pipeline depths for each
//! architecture generation.

use oxicuda_ptx::prelude::SmVersion;

use crate::error::{BlasError, BlasResult};
use crate::level3::gemm::dispatch::TileConfig;

/// FP64 specific GEMM configuration.
///
/// Provides architecture-aware tile selection and capability queries for
/// double-precision GEMM (DGEMM).
pub struct F64Config;

impl F64Config {
    /// Element size in bytes for FP64.
    pub const ELEMENT_BYTES: u32 = 8;

    /// Get optimal tile config for FP64 GEMM.
    ///
    /// Ampere and Hopper use Tensor Core DMMA with larger pipeline depths.
    /// Older architectures fall back to smaller tiles without tensor cores.
    #[must_use]
    pub fn tile_config(sm: SmVersion, _m: u32, _n: u32, _k: u32) -> TileConfig {
        match sm {
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
            SmVersion::Sm90 | SmVersion::Sm90a => TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 16,
                warp_m: 32,
                warp_n: 32,
                stages: 4,
                use_tensor_core: true,
                split_k: 1,
            },
            SmVersion::Sm100 | SmVersion::Sm120 => TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 16,
                warp_m: 32,
                warp_n: 32,
                stages: 4,
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

    /// Check if Tensor Core DMMA is available for FP64.
    ///
    /// DMMA (double-precision MMA) is available on Ampere (sm_80) and later
    /// data-centre GPUs.
    #[must_use]
    pub fn has_tensor_core(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm80
    }

    /// Validate that the given dimensions are suitable for FP64 GEMM.
    ///
    /// Returns an error if any dimension is zero.
    pub fn validate_dimensions(m: u32, n: u32, k: u32) -> BlasResult<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "FP64 GEMM dimensions must be non-zero: m={m}, n={n}, k={k}"
            )));
        }
        Ok(())
    }

    /// Compute the split-K factor for tall-skinny or reduction-heavy problems.
    ///
    /// When K is much larger than M and N, splitting the reduction across
    /// multiple thread blocks can improve occupancy.
    #[must_use]
    pub fn compute_split_k(m: u32, n: u32, k: u32) -> u32 {
        if m * n < 4096 && k > 4096 {
            // For very tall-skinny problems, split K
            let factor = k / 1024;
            factor.clamp(2, 16)
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_core_availability() {
        assert!(!F64Config::has_tensor_core(SmVersion::Sm75));
        assert!(F64Config::has_tensor_core(SmVersion::Sm80));
        assert!(F64Config::has_tensor_core(SmVersion::Sm90));
    }

    #[test]
    fn tile_config_turing_no_tensor_core() {
        let cfg = F64Config::tile_config(SmVersion::Sm75, 1024, 1024, 1024);
        assert!(!cfg.use_tensor_core);
        assert_eq!(cfg.tile_k, 8);
    }

    #[test]
    fn tile_config_ampere_uses_tensor_core() {
        let cfg = F64Config::tile_config(SmVersion::Sm80, 1024, 1024, 1024);
        assert!(cfg.use_tensor_core);
        assert_eq!(cfg.stages, 3);
    }

    #[test]
    fn validate_zero_dimension() {
        assert!(F64Config::validate_dimensions(0, 1024, 1024).is_err());
        assert!(F64Config::validate_dimensions(1024, 0, 1024).is_err());
        assert!(F64Config::validate_dimensions(1024, 1024, 0).is_err());
    }

    #[test]
    fn validate_valid_dimensions() {
        assert!(F64Config::validate_dimensions(128, 256, 512).is_ok());
    }

    #[test]
    fn split_k_small_mn_large_k() {
        let sk = F64Config::compute_split_k(32, 32, 8192);
        assert!(sk >= 2);
        assert!(sk <= 16);
    }

    #[test]
    fn split_k_balanced() {
        let sk = F64Config::compute_split_k(1024, 1024, 1024);
        assert_eq!(sk, 1);
    }
}
