//! BF16 GEMM configuration for training workloads.
//!
//! Brain floating-point (BF16) uses the same 8-bit exponent as FP32 but
//! only 7 mantissa bits. This gives it the same dynamic range as FP32,
//! making it popular for deep learning training where gradient magnitudes
//! vary widely. BF16 Tensor Cores are available from Ampere (sm_80).
//!
//! This module is gated behind `#[cfg(feature = "f16")]` because it
//! depends on the `half` crate for the [`half::bf16`] type.

use oxicuda_ptx::prelude::SmVersion;

use crate::error::{BlasError, BlasResult};
use crate::level3::gemm::dispatch::TileConfig;

/// BF16 specific GEMM configuration.
///
/// Provides architecture-aware tile selection for BF16 GEMM. BF16 Tensor
/// Cores are only available on Ampere and later; Turing has no BF16
/// hardware support.
pub struct Bf16Config;

impl Bf16Config {
    /// Element size in bytes for BF16.
    pub const ELEMENT_BYTES: u32 = 2;

    /// Get optimal tile config for BF16 GEMM.
    ///
    /// BF16 shares the same Tensor Core throughput as FP16 on Ampere+,
    /// so tile sizes are similar to the FP16 configs.
    #[must_use]
    pub fn tile_config(sm: SmVersion, _m: u32, _n: u32, _k: u32) -> TileConfig {
        match sm {
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
            _ => {
                // Turing (sm_75) has no BF16 Tensor Core support;
                // fall back to CUDA core emulation with smaller tiles.
                TileConfig {
                    tile_m: 64,
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

    /// Check if BF16 Tensor Cores are available.
    ///
    /// BF16 hardware support starts with Ampere (sm_80). Turing does not
    /// have BF16 Tensor Core instructions.
    #[must_use]
    pub fn has_tensor_core(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm80
    }

    /// Check if WGMMA is available for BF16.
    #[must_use]
    pub fn has_wgmma(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm90
    }

    /// Validate dimensions for BF16 GEMM.
    pub fn validate_dimensions(m: u32, n: u32, k: u32) -> BlasResult<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "BF16 GEMM dimensions must be non-zero: m={m}, n={n}, k={k}"
            )));
        }
        Ok(())
    }

    /// Check whether this architecture supports BF16 at all (even without
    /// Tensor Cores, via software emulation).
    ///
    /// On Turing, BF16 operations would need to be emulated through FP32
    /// conversion, which negates the throughput benefit. Returns `false`
    /// for architectures where BF16 GEMM would be slower than FP32.
    #[must_use]
    pub fn is_efficient(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm80
    }

    /// Compute the split-K factor for BF16 problems.
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
    fn no_bf16_tensor_core_on_turing() {
        assert!(!Bf16Config::has_tensor_core(SmVersion::Sm75));
        let cfg = Bf16Config::tile_config(SmVersion::Sm75, 512, 512, 512);
        assert!(!cfg.use_tensor_core);
    }

    #[test]
    fn bf16_tensor_core_on_ampere() {
        assert!(Bf16Config::has_tensor_core(SmVersion::Sm80));
        let cfg = Bf16Config::tile_config(SmVersion::Sm80, 1024, 1024, 1024);
        assert!(cfg.use_tensor_core);
    }

    #[test]
    fn wgmma_support() {
        assert!(!Bf16Config::has_wgmma(SmVersion::Sm89));
        assert!(Bf16Config::has_wgmma(SmVersion::Sm90));
    }

    #[test]
    fn efficiency_check() {
        assert!(!Bf16Config::is_efficient(SmVersion::Sm75));
        assert!(Bf16Config::is_efficient(SmVersion::Sm80));
    }

    #[test]
    fn validate_ok() {
        assert!(Bf16Config::validate_dimensions(256, 256, 128).is_ok());
    }

    #[test]
    fn validate_zero() {
        assert!(Bf16Config::validate_dimensions(256, 0, 128).is_err());
    }
}
