//! FP8 (E4M3/E5M2) GEMM configuration for Hopper+ inference.
//!
//! FP8 provides two formats:
//!
//! - **E4M3**: 4-bit exponent, 3-bit mantissa — higher precision, used for
//!   weights and activations during inference.
//! - **E5M2**: 5-bit exponent, 2-bit mantissa — larger dynamic range, used
//!   for gradients during FP8 training.
//!
//! FP8 Tensor Cores are available from Ada Lovelace (sm_89) for inference
//! and Hopper (sm_90) for WGMMA-based high-throughput FP8 GEMM.
//!
//! ## Dynamic Tile Selection
//!
//! The tile heuristic classifies each GEMM problem into a workload class
//! (small-square, large-square, skinny-M/N/K, or general) and selects tile
//! dimensions accordingly. This avoids the one-size-fits-all approach and
//! improves occupancy for non-square shapes common in transformer inference
//! (e.g. batch=1 token generation with skinny M).

use oxicuda_ptx::prelude::SmVersion;

use crate::error::{BlasError, BlasResult};
use crate::level3::gemm::dispatch::TileConfig;

/// FP8 data format variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp8Format {
    /// E4M3: 1 sign + 4 exponent + 3 mantissa bits.
    /// Range ~[-448, 448], higher precision than E5M2.
    E4M3,
    /// E5M2: 1 sign + 5 exponent + 2 mantissa bits.
    /// Range ~[-57344, 57344], larger dynamic range for gradients.
    E5M2,
}

/// Workload classification for FP8 GEMM shape-dependent tile selection.
///
/// Each variant captures a distinct regime where different tile shapes
/// yield better utilization of the streaming multiprocessors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp8WorkloadClass {
    /// Both M and N are small (m*n < 8192). Use compact tiles to avoid
    /// under-filling the SM with partially-occupied CTAs.
    SmallSquare,
    /// Both M and N are large (>= 1024). Use the largest tile the
    /// architecture supports to maximise arithmetic intensity.
    LargeSquare,
    /// M is narrow (< 128) relative to N. Shrink tile_m and widen tile_n
    /// so that the CTA covers more columns per launch.
    SkinnyM,
    /// N is narrow (< 128) relative to M. Widen tile_m and shrink tile_n.
    SkinnyN,
    /// K is very short (< 64). Deep pipelines waste registers on stages
    /// that will never be filled, so we reduce pipeline depth.
    SkinnyK,
    /// None of the above special cases apply — use architecture defaults.
    General,
}

/// Classify a GEMM problem (m, n, k) into a workload category.
///
/// The classification order is deliberate: `SkinnyK` is checked last
/// because it modifies *stages* rather than tile shape, and the tile
/// shape should be chosen first based on M/N geometry.
#[must_use]
pub fn classify_workload(m: u32, n: u32, k: u32) -> Fp8WorkloadClass {
    // Use u64 to avoid overflow on large dimensions.
    let mn = (m as u64) * (n as u64);

    if mn < 8192 {
        return Fp8WorkloadClass::SmallSquare;
    }
    if m < 128 && n >= 128 {
        return Fp8WorkloadClass::SkinnyM;
    }
    if n < 128 && m >= 128 {
        return Fp8WorkloadClass::SkinnyN;
    }
    if k < 64 {
        return Fp8WorkloadClass::SkinnyK;
    }
    if m >= 1024 && n >= 1024 {
        return Fp8WorkloadClass::LargeSquare;
    }
    Fp8WorkloadClass::General
}

/// Shape-dependent tile heuristic for FP8 GEMM.
///
/// Combines workload classification with architecture capabilities to
/// produce a `TileConfig` tuned for the specific problem shape.
pub struct Fp8TileHeuristic;

impl Fp8TileHeuristic {
    /// Select an optimal `TileConfig` for the given problem shape and SM version.
    #[must_use]
    pub fn select(m: u32, n: u32, k: u32, sm: SmVersion) -> TileConfig {
        let class = classify_workload(m, n, k);

        match sm {
            SmVersion::Sm89 => Self::select_ada(class, k),
            SmVersion::Sm90 | SmVersion::Sm90a => Self::select_hopper(class, k),
            SmVersion::Sm100 | SmVersion::Sm120 => Self::select_blackwell(class, k),
            _ => Self::select_fallback(),
        }
    }

    // ── Ada Lovelace (sm_89) ────────────────────────────────────────────

    fn select_ada(class: Fp8WorkloadClass, k: u32) -> TileConfig {
        let stages = if k < 64 { 2 } else { 4 };

        match class {
            Fp8WorkloadClass::SmallSquare => TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 32,
                warp_m: 32,
                warp_n: 32,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::SkinnyM => TileConfig {
                tile_m: 64,
                tile_n: 128,
                tile_k: 64,
                warp_m: 32,
                warp_n: 64,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::SkinnyN => TileConfig {
                tile_m: 128,
                tile_n: 64,
                tile_k: 64,
                warp_m: 64,
                warp_n: 32,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::LargeSquare => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 64,
                warp_m: 64,
                warp_n: 64,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::SkinnyK | Fp8WorkloadClass::General => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 64,
                warp_m: 64,
                warp_n: 64,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
        }
    }

    // ── Hopper (sm_90 / sm_90a) ─────────────────────────────────────────

    fn select_hopper(class: Fp8WorkloadClass, k: u32) -> TileConfig {
        let stages = if k < 64 { 2 } else { 4 };

        match class {
            Fp8WorkloadClass::SmallSquare => TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 64,
                warp_m: 32,
                warp_n: 32,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::SkinnyM => TileConfig {
                tile_m: 64,
                tile_n: 256,
                tile_k: 128,
                warp_m: 32,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::SkinnyN => TileConfig {
                tile_m: 256,
                tile_n: 64,
                tile_k: 128,
                warp_m: 128,
                warp_n: 32,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::LargeSquare => TileConfig {
                tile_m: 256,
                tile_n: 256,
                tile_k: 128,
                warp_m: 64,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::SkinnyK => TileConfig {
                tile_m: 128,
                tile_n: 256,
                tile_k: 128,
                warp_m: 64,
                warp_n: 128,
                stages: 2,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::General => TileConfig {
                tile_m: 128,
                tile_n: 256,
                tile_k: 128,
                warp_m: 64,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
        }
    }

    // ── Blackwell (sm_100 / sm_120) ─────────────────────────────────────

    fn select_blackwell(class: Fp8WorkloadClass, k: u32) -> TileConfig {
        let stages = if k < 64 { 2 } else { 4 };

        match class {
            Fp8WorkloadClass::SmallSquare => TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 64,
                warp_m: 32,
                warp_n: 32,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::SkinnyM => TileConfig {
                tile_m: 64,
                tile_n: 256,
                tile_k: 128,
                warp_m: 32,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::SkinnyN => TileConfig {
                tile_m: 256,
                tile_n: 64,
                tile_k: 128,
                warp_m: 128,
                warp_n: 32,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::LargeSquare => TileConfig {
                tile_m: 256,
                tile_n: 256,
                tile_k: 128,
                warp_m: 64,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::SkinnyK => TileConfig {
                tile_m: 256,
                tile_n: 256,
                tile_k: 128,
                warp_m: 64,
                warp_n: 128,
                stages: 2,
                use_tensor_core: true,
                split_k: 1,
            },
            Fp8WorkloadClass::General => TileConfig {
                tile_m: 256,
                tile_n: 256,
                tile_k: 128,
                warp_m: 64,
                warp_n: 128,
                stages,
                use_tensor_core: true,
                split_k: 1,
            },
        }
    }

    // ── Fallback (pre-Ada, no FP8 TC) ──────────────────────────────────

    fn select_fallback() -> TileConfig {
        TileConfig {
            tile_m: 64,
            tile_n: 64,
            tile_k: 32,
            warp_m: 32,
            warp_n: 32,
            stages: 1,
            use_tensor_core: false,
            split_k: 1,
        }
    }
}

/// FP8 specific GEMM configuration.
///
/// Provides architecture-aware tile selection for FP8 GEMM. The accumulator
/// is always FP32 to preserve precision during the reduction.
pub struct Fp8Config;

impl Fp8Config {
    /// Element size in bytes for FP8.
    pub const ELEMENT_BYTES: u32 = 1;

    /// Get optimal tile config for FP8 GEMM.
    ///
    /// FP8 has the highest arithmetic throughput per byte of data moved,
    /// so very large tiles and deep pipelines are optimal to keep the
    /// Tensor Cores saturated. The tile selection is now shape-dependent:
    /// see `Fp8TileHeuristic` and `classify_workload` for details.
    #[must_use]
    pub fn tile_config(sm: SmVersion, m: u32, n: u32, k: u32) -> TileConfig {
        Fp8TileHeuristic::select(m, n, k, sm)
    }

    /// Check if FP8 Tensor Cores are available.
    #[must_use]
    pub fn is_available(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm89
    }

    /// Check if a specific FP8 format is supported on the given architecture.
    #[must_use]
    pub fn is_format_supported(format: Fp8Format, sm: SmVersion) -> bool {
        match format {
            Fp8Format::E4M3 => sm >= SmVersion::Sm89,
            Fp8Format::E5M2 => sm >= SmVersion::Sm89,
        }
    }

    /// Check if WGMMA-based FP8 is available (Hopper+).
    #[must_use]
    pub fn has_wgmma(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm90
    }

    /// Validate dimensions and architecture for FP8 GEMM.
    pub fn validate(sm: SmVersion, format: Fp8Format, m: u32, n: u32, k: u32) -> BlasResult<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "FP8 GEMM dimensions must be non-zero: m={m}, n={n}, k={k}"
            )));
        }
        if !Self::is_available(sm) {
            return Err(BlasError::UnsupportedOperation(format!(
                "FP8 Tensor Cores require Ada Lovelace+ (sm_89), got {sm}"
            )));
        }
        if !Self::is_format_supported(format, sm) {
            return Err(BlasError::UnsupportedOperation(format!(
                "FP8 format {format:?} not supported on {sm}"
            )));
        }
        Ok(())
    }

    /// Compute the split-K factor for FP8 problems.
    ///
    /// FP8 has extremely high compute throughput, so split-K is rarely
    /// needed except for very skinny problems.
    #[must_use]
    pub fn compute_split_k(m: u32, n: u32, k: u32) -> u32 {
        if m * n < 2048 && k > 32768 {
            let factor = k / 8192;
            factor.clamp(2, 64)
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Existing tests ──────────────────────────────────────────────────

    #[test]
    fn fp8_not_on_ampere() {
        assert!(!Fp8Config::is_available(SmVersion::Sm80));
        assert!(!Fp8Config::is_available(SmVersion::Sm86));
    }

    #[test]
    fn fp8_on_ada() {
        assert!(Fp8Config::is_available(SmVersion::Sm89));
    }

    #[test]
    fn fp8_on_hopper() {
        assert!(Fp8Config::is_available(SmVersion::Sm90));
        assert!(Fp8Config::has_wgmma(SmVersion::Sm90a));
    }

    #[test]
    fn tile_config_hopper() {
        let cfg = Fp8Config::tile_config(SmVersion::Sm90, 2048, 2048, 2048);
        assert!(cfg.use_tensor_core);
        assert_eq!(cfg.tile_k, 128);
    }

    #[test]
    fn validate_ok() {
        assert!(Fp8Config::validate(SmVersion::Sm89, Fp8Format::E4M3, 128, 128, 128).is_ok());
    }

    #[test]
    fn validate_unsupported() {
        assert!(Fp8Config::validate(SmVersion::Sm80, Fp8Format::E4M3, 128, 128, 128).is_err());
    }

    #[test]
    fn validate_zero_dim() {
        assert!(Fp8Config::validate(SmVersion::Sm90, Fp8Format::E5M2, 0, 128, 128).is_err());
    }

    #[test]
    fn format_support() {
        assert!(Fp8Config::is_format_supported(
            Fp8Format::E4M3,
            SmVersion::Sm89
        ));
        assert!(Fp8Config::is_format_supported(
            Fp8Format::E5M2,
            SmVersion::Sm90
        ));
        assert!(!Fp8Config::is_format_supported(
            Fp8Format::E4M3,
            SmVersion::Sm80
        ));
    }

    // ── Workload classification tests ───────────────────────────────────

    #[test]
    fn classify_small_square() {
        // 64*64 = 4096 < 8192
        assert_eq!(
            classify_workload(64, 64, 256),
            Fp8WorkloadClass::SmallSquare
        );
        // 32*128 = 4096 < 8192
        assert_eq!(
            classify_workload(32, 128, 512),
            Fp8WorkloadClass::SmallSquare
        );
    }

    #[test]
    fn classify_large_square() {
        assert_eq!(
            classify_workload(1024, 1024, 512),
            Fp8WorkloadClass::LargeSquare
        );
        assert_eq!(
            classify_workload(4096, 4096, 4096),
            Fp8WorkloadClass::LargeSquare
        );
    }

    #[test]
    fn classify_skinny_m() {
        // m=64 < 128, n=512 >= 128, and m*n = 32768 >= 8192
        assert_eq!(classify_workload(64, 512, 256), Fp8WorkloadClass::SkinnyM);
    }

    #[test]
    fn classify_skinny_n() {
        // n=64 < 128, m=512 >= 128, and m*n = 32768 >= 8192
        assert_eq!(classify_workload(512, 64, 256), Fp8WorkloadClass::SkinnyN);
    }

    #[test]
    fn classify_skinny_k() {
        // k=32 < 64, m and n medium, m*n >= 8192, neither skinny M nor N
        assert_eq!(classify_workload(256, 256, 32), Fp8WorkloadClass::SkinnyK);
    }

    #[test]
    fn classify_general() {
        // m=256, n=512 — neither small, large, skinny M/N, nor skinny K
        assert_eq!(classify_workload(256, 512, 128), Fp8WorkloadClass::General);
    }

    #[test]
    fn classify_edge_k_equals_1() {
        // k=1 is both < 64 (skinnyK candidate) but m*n matters first
        // m=1, n=1 => m*n = 1 < 8192 => SmallSquare takes priority
        assert_eq!(classify_workload(1, 1, 1), Fp8WorkloadClass::SmallSquare);
    }

    #[test]
    fn classify_edge_m_equals_1() {
        // m=1, n=16384 => m*n = 16384 >= 8192, m < 128, n >= 128 => SkinnyM
        assert_eq!(classify_workload(1, 16384, 256), Fp8WorkloadClass::SkinnyM);
    }

    #[test]
    fn classify_edge_n_equals_1() {
        // n=1, m=16384 => m*n = 16384 >= 8192, n < 128, m >= 128 => SkinnyN
        assert_eq!(classify_workload(16384, 1, 256), Fp8WorkloadClass::SkinnyN);
    }

    // ── Heuristic tile selection tests ──────────────────────────────────

    #[test]
    fn heuristic_small_square_ada() {
        let cfg = Fp8TileHeuristic::select(64, 64, 256, SmVersion::Sm89);
        assert_eq!(cfg.tile_m, 64);
        assert_eq!(cfg.tile_n, 64);
        assert!(cfg.use_tensor_core);
    }

    #[test]
    fn heuristic_skinny_m_hopper() {
        let cfg = Fp8TileHeuristic::select(64, 512, 256, SmVersion::Sm90);
        assert_eq!(cfg.tile_m, 64);
        assert_eq!(cfg.tile_n, 256);
        assert_eq!(cfg.tile_k, 128);
    }

    #[test]
    fn heuristic_skinny_n_hopper() {
        let cfg = Fp8TileHeuristic::select(512, 64, 256, SmVersion::Sm90);
        assert_eq!(cfg.tile_m, 256);
        assert_eq!(cfg.tile_n, 64);
    }

    #[test]
    fn heuristic_skinny_k_reduces_stages() {
        let cfg = Fp8TileHeuristic::select(256, 256, 32, SmVersion::Sm90);
        assert_eq!(cfg.stages, 2);
    }

    #[test]
    fn heuristic_large_square_blackwell() {
        let cfg = Fp8TileHeuristic::select(2048, 2048, 2048, SmVersion::Sm100);
        assert_eq!(cfg.tile_m, 256);
        assert_eq!(cfg.tile_n, 256);
        assert_eq!(cfg.stages, 4);
    }

    #[test]
    fn heuristic_general_hopper() {
        let cfg = Fp8TileHeuristic::select(256, 512, 128, SmVersion::Sm90);
        assert_eq!(cfg.tile_m, 128);
        assert_eq!(cfg.tile_n, 256);
        assert_eq!(cfg.stages, 4);
    }

    #[test]
    fn heuristic_fallback_pre_ada() {
        let cfg = Fp8TileHeuristic::select(256, 256, 256, SmVersion::Sm80);
        assert!(!cfg.use_tensor_core);
        assert_eq!(cfg.stages, 1);
    }

    #[test]
    fn tile_config_delegates_to_heuristic() {
        // Verify that Fp8Config::tile_config now produces shape-dependent results.
        let general = Fp8Config::tile_config(SmVersion::Sm90, 256, 512, 128);
        let skinny_m = Fp8Config::tile_config(SmVersion::Sm90, 64, 512, 256);
        // General should have larger tile_m than skinny-M case.
        assert!(general.tile_m > skinny_m.tile_m);
    }

    #[test]
    fn heuristic_skinny_m_blackwell() {
        let cfg = Fp8TileHeuristic::select(1, 16384, 256, SmVersion::Sm100);
        assert_eq!(cfg.tile_m, 64);
        assert_eq!(cfg.tile_n, 256);
    }

    #[test]
    fn heuristic_skinny_k_ada() {
        // On Ada, skinny K still reduces stages to 2.
        let cfg = Fp8TileHeuristic::select(256, 256, 16, SmVersion::Sm89);
        assert_eq!(cfg.stages, 2);
    }

    #[test]
    fn heuristic_large_square_sm120() {
        let cfg = Fp8TileHeuristic::select(4096, 4096, 4096, SmVersion::Sm120);
        assert_eq!(cfg.tile_m, 256);
        assert_eq!(cfg.tile_n, 256);
        assert_eq!(cfg.tile_k, 128);
        assert_eq!(cfg.stages, 4);
        assert!(cfg.use_tensor_core);
    }

    // =========================================================================
    // Quality gate: Ada FP8 GEMM with E4M3 / E5M2 inputs
    // =========================================================================

    /// classify_workload(256, 256, 256) should return General:
    ///   m*n = 65536 >= 8192, neither m nor n < 128 vs the other,
    ///   k = 256 >= 64, and neither m nor n >= 1024.
    #[test]
    fn fp8_workload_class_256x256x256_is_general() {
        let wl = classify_workload(256, 256, 256);
        assert_eq!(
            wl,
            Fp8WorkloadClass::General,
            "256×256×256 should classify as General"
        );
    }

    /// Large matrix (4096 × 4096 × 4096) should be LargeSquare.
    #[test]
    fn fp8_workload_class_large_matrix_is_large_square() {
        let wl = classify_workload(4096, 4096, 4096);
        assert_eq!(
            wl,
            Fp8WorkloadClass::LargeSquare,
            "4096×4096×4096 should classify as LargeSquare"
        );
    }

    /// Ada (SM89) FP8 E4M3 tile config for 128×128×128 must use tensor cores.
    #[test]
    fn ada_fp8_e4m3_tile_config_uses_tensor_core() {
        let cfg = Fp8TileHeuristic::select(128, 128, 128, SmVersion::Sm89);
        assert!(
            cfg.use_tensor_core,
            "Ada FP8 GEMM must use Tensor Core path (mma.sync)"
        );
    }

    /// Ada (SM89) FP8 E4M3 format is supported (Ada introduced FP8 TCs).
    #[test]
    fn ada_fp8_e4m3_format_is_supported() {
        assert!(
            Fp8Config::is_format_supported(Fp8Format::E4M3, SmVersion::Sm89),
            "Ada Lovelace (SM89) must support FP8 E4M3"
        );
        assert!(
            Fp8Config::is_available(SmVersion::Sm89),
            "Fp8Config::is_available must return true for SM89"
        );
    }

    /// Ada (SM89) FP8 E5M2 format is supported.
    #[test]
    fn ada_fp8_e5m2_format_is_supported() {
        assert!(
            Fp8Config::is_format_supported(Fp8Format::E5M2, SmVersion::Sm89),
            "Ada Lovelace (SM89) must support FP8 E5M2"
        );
    }

    /// Mixed E4M3 × E5M2: both formats are individually valid on SM89.
    /// FP8 GEMM can accept A in E4M3 and B in E5M2 (mixed precision inference).
    #[test]
    fn ada_fp8_mixed_e4m3_e5m2_both_valid_on_sm89() {
        // Validate each format independently — FP8 GEMM accepts either
        // format for each operand separately.
        let e4m3_ok = Fp8Config::validate(SmVersion::Sm89, Fp8Format::E4M3, 128, 128, 128);
        let e5m2_ok = Fp8Config::validate(SmVersion::Sm89, Fp8Format::E5M2, 128, 128, 128);

        assert!(
            e4m3_ok.is_ok(),
            "Ada E4M3 validation must pass: {:?}",
            e4m3_ok.err()
        );
        assert!(
            e5m2_ok.is_ok(),
            "Ada E5M2 validation must pass: {:?}",
            e5m2_ok.err()
        );
    }

    /// Ada FP8 is not available on Ampere (SM80) — pre-Ada architectures
    /// lack FP8 Tensor Core support.
    #[test]
    fn fp8_not_supported_on_pre_ada_sm80() {
        assert!(
            !Fp8Config::is_available(SmVersion::Sm80),
            "Ampere (SM80) must NOT support FP8 GEMM"
        );
        assert!(
            !Fp8Config::is_format_supported(Fp8Format::E4M3, SmVersion::Sm80),
            "Ampere must NOT support E4M3"
        );
        assert!(
            !Fp8Config::is_format_supported(Fp8Format::E5M2, SmVersion::Sm80),
            "Ampere must NOT support E5M2"
        );
    }

    /// Hopper (SM90) has WGMMA-based FP8 (wider tile_k=128).
    #[test]
    fn hopper_fp8_wgmma_larger_tile_k_than_ada() {
        let ada_cfg = Fp8TileHeuristic::select(4096, 4096, 4096, SmVersion::Sm89);
        let hopper_cfg = Fp8TileHeuristic::select(4096, 4096, 4096, SmVersion::Sm90);

        // Hopper WGMMA uses tile_k=128 vs Ada tile_k=64 for LargeSquare.
        assert!(
            hopper_cfg.tile_k >= ada_cfg.tile_k,
            "Hopper FP8 tile_k ({}) must be >= Ada FP8 tile_k ({})",
            hopper_cfg.tile_k,
            ada_cfg.tile_k
        );
        assert!(
            Fp8Config::has_wgmma(SmVersion::Sm90),
            "Hopper must have WGMMA for high-throughput FP8"
        );
    }
}
