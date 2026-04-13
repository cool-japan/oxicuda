//! FP32 SGEMM configuration and helpers.
//!
//! FP32 is the most common precision for general-purpose GPU computing.
//! On Ampere+ architectures the TF32 path (transparent FP32-to-TF32
//! truncation) can double throughput at the cost of a few mantissa bits;
//! see [`crate::precision::tf32_ops`] for the dedicated TF32 module.
//!
//! This module provides the pure FP32 (non-TF32) tiling strategy.

use oxicuda_ptx::prelude::SmVersion;

use crate::error::{BlasError, BlasResult};
use crate::level3::gemm::dispatch::TileConfig;

/// FP32 specific GEMM configuration.
///
/// Provides architecture-aware tile selection for single-precision GEMM
/// (SGEMM) without TF32 truncation. For the TF32 path, see
/// [`Tf32Config`](super::Tf32Config).
pub struct F32Config;

impl F32Config {
    /// Element size in bytes for FP32.
    pub const ELEMENT_BYTES: u32 = 4;

    /// Get optimal tile config for FP32 GEMM.
    ///
    /// FP32 GEMM uses CUDA cores (FMA pipeline). On Ampere+, larger tiles
    /// and deeper pipelines improve throughput through better latency hiding
    /// via `cp.async`.
    #[must_use]
    pub fn tile_config(sm: SmVersion, _m: u32, _n: u32, _k: u32) -> TileConfig {
        match sm {
            SmVersion::Sm80 | SmVersion::Sm86 => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 32,
                warp_n: 64,
                stages: 3,
                use_tensor_core: false,
                split_k: 1,
            },
            SmVersion::Sm89 => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 32,
                warp_n: 64,
                stages: 3,
                use_tensor_core: false,
                split_k: 1,
            },
            SmVersion::Sm90 | SmVersion::Sm90a => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: 4,
                use_tensor_core: false,
                split_k: 1,
            },
            SmVersion::Sm100 | SmVersion::Sm120 => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: 4,
                use_tensor_core: false,
                split_k: 1,
            },
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

    /// Whether TF32 Tensor Core acceleration is available.
    ///
    /// TF32 uses Tensor Cores with FP32 inputs truncated to 19-bit mantissa.
    /// This is exposed here for convenience; the dedicated TF32 config lives
    /// in [`Tf32Config`](super::Tf32Config).
    #[must_use]
    pub fn has_tf32_support(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm80
    }

    /// Validate that the given dimensions are suitable for FP32 GEMM.
    pub fn validate_dimensions(m: u32, n: u32, k: u32) -> BlasResult<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "FP32 GEMM dimensions must be non-zero: m={m}, n={n}, k={k}"
            )));
        }
        Ok(())
    }

    /// Compute the split-K factor for reduction-heavy FP32 problems.
    #[must_use]
    pub fn compute_split_k(m: u32, n: u32, k: u32) -> u32 {
        if m * n < 8192 && k > 8192 {
            let factor = k / 2048;
            factor.clamp(2, 16)
        } else {
            1
        }
    }

    /// Estimate the number of FLOPs for a GEMM of the given dimensions.
    ///
    /// Standard GEMM requires 2*M*N*K FLOPs (multiply + add per element).
    #[must_use]
    pub fn estimate_flops(m: u32, n: u32, k: u32) -> u64 {
        2 * u64::from(m) * u64::from(n) * u64::from(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_config_turing() {
        let cfg = F32Config::tile_config(SmVersion::Sm75, 512, 512, 512);
        assert!(!cfg.use_tensor_core);
        assert_eq!(cfg.stages, 2);
    }

    #[test]
    fn tile_config_ampere() {
        let cfg = F32Config::tile_config(SmVersion::Sm80, 1024, 1024, 1024);
        assert_eq!(cfg.tile_m, 128);
        assert_eq!(cfg.tile_n, 128);
        assert_eq!(cfg.stages, 3);
    }

    #[test]
    fn tf32_support() {
        assert!(!F32Config::has_tf32_support(SmVersion::Sm75));
        assert!(F32Config::has_tf32_support(SmVersion::Sm80));
        assert!(F32Config::has_tf32_support(SmVersion::Sm90a));
    }

    #[test]
    fn validate_zero_dims() {
        assert!(F32Config::validate_dimensions(0, 512, 512).is_err());
    }

    #[test]
    fn validate_ok() {
        assert!(F32Config::validate_dimensions(128, 256, 64).is_ok());
    }

    #[test]
    fn flops_estimation() {
        assert_eq!(F32Config::estimate_flops(100, 100, 100), 2_000_000);
    }

    #[test]
    fn split_k_tall_skinny() {
        let sk = F32Config::compute_split_k(16, 16, 16384);
        assert!(sk >= 2);
    }
}
