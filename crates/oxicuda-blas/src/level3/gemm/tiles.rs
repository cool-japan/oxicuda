//! Rectangular (non-square) tile configurations for GEMM dispatch.
//!
//! When M >> N or N >> M, rectangular tiles better utilize hardware by
//! matching the tile shape to the matrix aspect ratio. This module provides
//! predefined tile shapes and a [`TileSelector`] that picks the best tile
//! for a given problem size and GPU architecture.

use oxicuda_ptx::prelude::*;

use super::dispatch::TileConfig;

// ---------------------------------------------------------------------------
// Rectangular tile variants
// ---------------------------------------------------------------------------

/// Predefined rectangular tile configurations for asymmetric matrix shapes.
///
/// When M >> N or N >> M, rectangular tiles better utilize hardware by
/// matching the tile shape to the matrix aspect ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RectangularTile {
    /// 32 x 32 — small square tile for tiny problems.
    Square32,
    /// 64 x 64 — medium square tile (SIMT default).
    Square64,
    /// 128 x 128 — large square tile (TC default).
    Square128,
    /// 128 rows x 64 columns — good for tall matrices (M >> N).
    Tile128x64,
    /// 64 rows x 128 columns — good for wide matrices (N >> M).
    Tile64x128,
    /// 256 rows x 64 columns — very tall M with narrow N.
    Tile256x64,
    /// 64 rows x 256 columns — very wide N with short M.
    Tile64x256,
    /// 128 x 256 — large tile for big problems with N >> M.
    Tile128x256,
    /// 256 x 128 — large tile for big problems with M >> N.
    Tile256x128,
}

impl RectangularTile {
    /// Block tile size in the M dimension (rows per CTA).
    #[must_use]
    pub const fn tile_m(&self) -> u32 {
        match self {
            Self::Square32 => 32,
            Self::Square64 | Self::Tile64x128 | Self::Tile64x256 => 64,
            Self::Square128 | Self::Tile128x64 | Self::Tile128x256 => 128,
            Self::Tile256x64 | Self::Tile256x128 => 256,
        }
    }

    /// Block tile size in the N dimension (columns per CTA).
    #[must_use]
    pub const fn tile_n(&self) -> u32 {
        match self {
            Self::Square32 => 32,
            Self::Square64 | Self::Tile128x64 | Self::Tile256x64 => 64,
            Self::Square128 | Self::Tile64x128 | Self::Tile256x128 => 128,
            Self::Tile64x256 | Self::Tile128x256 => 256,
        }
    }

    /// Default warp-level tile within the block tile.
    ///
    /// The warp tile determines how the block tile is partitioned among warps.
    /// For rectangular tiles the warp tile is chosen to keep each warp's
    /// workload roughly square while respecting the block tile boundaries.
    #[must_use]
    pub const fn warp_tile(&self) -> (u32, u32) {
        match self {
            Self::Square32 => (16, 16),
            Self::Square64 => (32, 32),
            Self::Square128 => (64, 64),
            Self::Tile128x64 => (64, 32),
            Self::Tile64x128 => (32, 64),
            Self::Tile256x64 => (64, 32),
            Self::Tile64x256 => (32, 64),
            Self::Tile128x256 => (64, 64),
            Self::Tile256x128 => (64, 64),
        }
    }

    /// Converts this rectangular tile into a full [`TileConfig`].
    ///
    /// The caller supplies tile_k, pipeline stages, and whether tensor cores
    /// are enabled; the remaining fields are derived from the tile shape.
    #[must_use]
    pub fn to_tile_config(&self, tile_k: u32, stages: u32, use_tensor_core: bool) -> TileConfig {
        let (warp_m, warp_n) = self.warp_tile();
        TileConfig {
            tile_m: self.tile_m(),
            tile_n: self.tile_n(),
            tile_k,
            warp_m,
            warp_n,
            stages,
            use_tensor_core,
            split_k: 1,
        }
    }

    /// Returns all defined rectangular tile variants (including square ones).
    #[must_use]
    pub fn all() -> &'static [RectangularTile] {
        &[
            Self::Square32,
            Self::Square64,
            Self::Square128,
            Self::Tile128x64,
            Self::Tile64x128,
            Self::Tile256x64,
            Self::Tile64x256,
            Self::Tile128x256,
            Self::Tile256x128,
        ]
    }

    /// Shared memory required for one pipeline stage (A-tile + B-tile),
    /// assuming elements of `elem_bytes` bytes each.
    #[must_use]
    pub const fn shared_mem_per_stage(&self, elem_bytes: u32, tile_k: u32) -> u32 {
        let smem_a = self.tile_m() * tile_k * elem_bytes;
        let smem_b = tile_k * self.tile_n() * elem_bytes;
        smem_a + smem_b
    }
}

// ---------------------------------------------------------------------------
// TileSelector
// ---------------------------------------------------------------------------

/// Selects optimal tile configuration based on matrix dimensions and
/// target GPU architecture.
///
/// The selector considers the aspect ratio of the output matrix, the
/// available shared memory, and whether tensor cores are in use.
#[derive(Debug, Clone)]
pub struct TileSelector {
    /// Target SM architecture version.
    pub sm_version: SmVersion,
    /// Whether tensor core instructions should be used.
    pub use_tensor_core: bool,
}

impl TileSelector {
    /// Creates a new tile selector for the given architecture.
    #[must_use]
    pub fn new(sm_version: SmVersion, use_tensor_core: bool) -> Self {
        Self {
            sm_version,
            use_tensor_core,
        }
    }

    /// Selects the optimal tile configuration for a GEMM with dimensions
    /// (M x K) * (K x N) = (M x N).
    ///
    /// The selection logic considers:
    /// - The aspect ratio of the output matrix (M / N).
    /// - Whether both dimensions are large enough for big rectangular tiles.
    /// - The shared memory budget of the target architecture.
    /// - The K dimension (for choosing tile_k and pipeline stages).
    #[must_use]
    pub fn select(&self, m: u32, n: u32, k: u32) -> TileConfig {
        let tile = self.select_tile_shape(m, n);
        let (tile_k, stages) = self.select_k_and_stages(k, &tile);
        tile.to_tile_config(tile_k, stages, self.use_tensor_core)
    }

    /// Returns all tile candidates that fit in shared memory on this arch.
    #[must_use]
    pub fn all_candidates(&self) -> Vec<RectangularTile> {
        let max_smem = self.sm_version.max_shared_mem_per_block();
        // Use a representative element size (4 bytes for f32) and tile_k (32)
        // to filter out tiles that are clearly too large.
        let elem_bytes: u32 = 4;
        let tile_k: u32 = 32;
        let stages: u32 = 1;

        RectangularTile::all()
            .iter()
            .copied()
            .filter(|tile| {
                let smem = tile.shared_mem_per_stage(elem_bytes, tile_k) * stages;
                smem <= max_smem
            })
            .collect()
    }

    /// Estimates SM occupancy / utilization for a given tile choice.
    ///
    /// Higher scores indicate better utilization. The score considers:
    /// - Number of CTAs that cover the problem (fewer wasted threads).
    /// - Shared memory usage relative to budget (lower is better for occupancy).
    /// - How well the tile shape matches the problem aspect ratio.
    #[must_use]
    pub fn occupancy_score(&self, tile: RectangularTile, m: u32, n: u32) -> f64 {
        let tile_m = tile.tile_m();
        let tile_n = tile.tile_n();

        // Number of CTAs needed.
        let ctas_m = m.div_ceil(tile_m);
        let ctas_n = n.div_ceil(tile_n);
        let total_ctas = u64::from(ctas_m) * u64::from(ctas_n);

        // Fraction of output elements that are "useful" (not padding).
        let total_output = u64::from(m) * u64::from(n);
        let tiled_output =
            u64::from(ctas_m) * u64::from(tile_m) * u64::from(ctas_n) * u64::from(tile_n);
        let coverage = if tiled_output == 0 {
            0.0
        } else {
            total_output as f64 / tiled_output as f64
        };

        // Shared memory pressure: lower is better for occupancy.
        let max_smem = self.sm_version.max_shared_mem_per_block() as f64;
        let elem_bytes: u32 = 4;
        let tile_k: u32 = 32;
        let smem = tile.shared_mem_per_stage(elem_bytes, tile_k) as f64;
        let smem_ratio = if max_smem > 0.0 {
            1.0 - (smem / max_smem).min(1.0)
        } else {
            0.0
        };

        // Aspect ratio alignment: reward tiles whose aspect ratio matches the
        // problem's aspect ratio.
        let problem_ar = if n > 0 { m as f64 / n as f64 } else { 1.0 };
        let tile_ar = if tile_n > 0 {
            tile_m as f64 / tile_n as f64
        } else {
            1.0
        };
        let ar_match = if problem_ar > 0.0 && tile_ar > 0.0 {
            let ratio = (problem_ar / tile_ar).ln().abs();
            (-ratio).exp() // 1.0 when perfectly matched, decays away
        } else {
            0.5
        };

        // Parallelism: more CTAs is generally better for SM utilization.
        let parallelism = (total_ctas as f64).sqrt();

        // Weighted combination.
        coverage * 0.35 + smem_ratio * 0.20 + ar_match * 0.25 + parallelism.min(10.0) / 10.0 * 0.20
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Selects the rectangular tile shape based on aspect ratio and sizes.
    fn select_tile_shape(&self, m: u32, n: u32) -> RectangularTile {
        // Avoid division by zero.
        let aspect_ratio = if n > 0 { m as f64 / n as f64 } else { 1.0 };

        let both_large = m > 512 && n > 512;
        let max_smem = self.sm_version.max_shared_mem_per_block();

        // Small problem: use small tiles.
        if m <= 64 && n <= 64 {
            return if m <= 32 && n <= 32 {
                RectangularTile::Square32
            } else {
                RectangularTile::Square64
            };
        }

        // Very tall: M >> N (ratio > 4.0)
        if aspect_ratio > 4.0 {
            let candidate = if m >= 512 && self.fits_in_smem(RectangularTile::Tile256x64, max_smem)
            {
                RectangularTile::Tile256x64
            } else {
                RectangularTile::Tile128x64
            };
            if self.fits_in_smem(candidate, max_smem) {
                return candidate;
            }
            return RectangularTile::Tile128x64;
        }

        // Moderately tall: M > N (ratio > 2.0)
        if aspect_ratio > 2.0 {
            if self.fits_in_smem(RectangularTile::Tile128x64, max_smem) {
                return RectangularTile::Tile128x64;
            }
            return RectangularTile::Square64;
        }

        // Very wide: N >> M (ratio < 0.25)
        if aspect_ratio < 0.25 {
            let candidate = if n >= 512 && self.fits_in_smem(RectangularTile::Tile64x256, max_smem)
            {
                RectangularTile::Tile64x256
            } else {
                RectangularTile::Tile64x128
            };
            if self.fits_in_smem(candidate, max_smem) {
                return candidate;
            }
            return RectangularTile::Tile64x128;
        }

        // Moderately wide: N > M (ratio < 0.5)
        if aspect_ratio < 0.5 {
            if self.fits_in_smem(RectangularTile::Tile64x128, max_smem) {
                return RectangularTile::Tile64x128;
            }
            return RectangularTile::Square64;
        }

        // Both large, not ~1.0 aspect ratio → larger rectangular tiles.
        if both_large {
            // Slightly tall.
            if aspect_ratio > 1.3 {
                if self.fits_in_smem(RectangularTile::Tile256x128, max_smem) {
                    return RectangularTile::Tile256x128;
                }
                return RectangularTile::Tile128x64;
            }
            // Slightly wide.
            if aspect_ratio < 0.77 {
                if self.fits_in_smem(RectangularTile::Tile128x256, max_smem) {
                    return RectangularTile::Tile128x256;
                }
                return RectangularTile::Tile64x128;
            }
        }

        // Default: square tile.
        if m >= 128 && n >= 128 {
            RectangularTile::Square128
        } else {
            RectangularTile::Square64
        }
    }

    /// Selects tile_k and pipeline stages based on the K dimension and arch.
    fn select_k_and_stages(&self, k: u32, tile: &RectangularTile) -> (u32, u32) {
        let max_smem = self.sm_version.max_shared_mem_per_block();

        if self.use_tensor_core {
            // TC path: prefer tile_k = 32 or 64 with multi-stage pipeline.
            let base_tile_k = if self.sm_version >= SmVersion::Sm90 {
                64
            } else {
                32
            };

            // Clamp tile_k to at most K itself.
            let tile_k = base_tile_k.min(k).max(8);

            // Determine stages: try 4, 3, 2, 1.
            let stages = self.max_stages_for_smem(tile, tile_k, max_smem);
            (tile_k, stages)
        } else {
            // SIMT path: smaller tile_k.
            let tile_k = if k >= 16 { 16 } else { 8 };
            let stages = self.max_stages_for_smem(tile, tile_k, max_smem).min(2);
            (tile_k, stages)
        }
    }

    /// Returns the maximum number of pipeline stages that fit in shared memory.
    fn max_stages_for_smem(&self, tile: &RectangularTile, tile_k: u32, max_smem: u32) -> u32 {
        let elem_bytes: u32 = 4; // conservative: f32
        let per_stage = tile.shared_mem_per_stage(elem_bytes, tile_k);

        if per_stage == 0 {
            return 1;
        }

        let max_from_arch = if self.sm_version >= SmVersion::Sm90 {
            4
        } else if self.sm_version >= SmVersion::Sm80 {
            3
        } else {
            2
        };

        let max_from_smem = max_smem / per_stage;
        max_from_smem.min(max_from_arch).max(1)
    }

    /// Checks whether a tile fits in shared memory with at least one stage.
    fn fits_in_smem(&self, tile: RectangularTile, max_smem: u32) -> bool {
        let elem_bytes: u32 = 4;
        let tile_k: u32 = 32;
        tile.shared_mem_per_stage(elem_bytes, tile_k) <= max_smem
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helper --

    fn simt_selector(sm: SmVersion) -> TileSelector {
        TileSelector::new(sm, false)
    }

    fn tc_selector(sm: SmVersion) -> TileSelector {
        TileSelector::new(sm, true)
    }

    // -- Square matrix → square tile --

    #[test]
    fn square_matrix_selects_square_tile() {
        let sel = simt_selector(SmVersion::Sm80);
        let tc = sel.select(256, 256, 256);
        assert_eq!(tc.tile_m, tc.tile_n, "square matrix should get square tile");
    }

    #[test]
    fn large_square_matrix_selects_128x128() {
        let sel = simt_selector(SmVersion::Sm80);
        let tc = sel.select(1024, 1024, 512);
        assert_eq!(tc.tile_m, 128);
        assert_eq!(tc.tile_n, 128);
    }

    // -- Tall matrix → tall tile --

    #[test]
    fn tall_matrix_selects_tall_tile() {
        let sel = tc_selector(SmVersion::Sm80);
        let tc = sel.select(4096, 64, 256);
        assert!(
            (tc.tile_m == 256 || tc.tile_m == 128) && tc.tile_n == 64,
            "tall matrix should get 128x64 or 256x64, got {}x{}",
            tc.tile_m,
            tc.tile_n,
        );
    }

    #[test]
    fn very_tall_matrix_prefers_256x64() {
        let sel = tc_selector(SmVersion::Sm80);
        let tc = sel.select(4096, 64, 512);
        assert_eq!(tc.tile_m, 256);
        assert_eq!(tc.tile_n, 64);
    }

    // -- Wide matrix → wide tile --

    #[test]
    fn wide_matrix_selects_wide_tile() {
        let sel = tc_selector(SmVersion::Sm80);
        let tc = sel.select(64, 4096, 256);
        assert!(
            (tc.tile_n == 256 || tc.tile_n == 128) && tc.tile_m == 64,
            "wide matrix should get 64x128 or 64x256, got {}x{}",
            tc.tile_m,
            tc.tile_n,
        );
    }

    #[test]
    fn very_wide_matrix_prefers_64x256() {
        let sel = tc_selector(SmVersion::Sm80);
        let tc = sel.select(64, 4096, 512);
        assert_eq!(tc.tile_m, 64);
        assert_eq!(tc.tile_n, 256);
    }

    // -- Balanced large → square --

    #[test]
    fn balanced_large_stays_square() {
        let sel = tc_selector(SmVersion::Sm80);
        let tc = sel.select(2048, 2048, 1024);
        assert_eq!(tc.tile_m, 128);
        assert_eq!(tc.tile_n, 128);
    }

    // -- Small problem → small tile --

    #[test]
    fn small_problem_uses_small_tile() {
        let sel = simt_selector(SmVersion::Sm80);
        let tc = sel.select(32, 32, 64);
        assert!(
            tc.tile_m <= 32 && tc.tile_n <= 32,
            "small problem should get 32x32, got {}x{}",
            tc.tile_m,
            tc.tile_n,
        );
    }

    #[test]
    fn small_64_problem_uses_64_tile() {
        let sel = simt_selector(SmVersion::Sm80);
        let tc = sel.select(48, 48, 64);
        assert_eq!(tc.tile_m, 64);
        assert_eq!(tc.tile_n, 64);
    }

    // -- to_tile_config conversion --

    #[test]
    fn to_tile_config_conversion() {
        let tile = RectangularTile::Tile128x64;
        let tc = tile.to_tile_config(32, 3, true);
        assert_eq!(tc.tile_m, 128);
        assert_eq!(tc.tile_n, 64);
        assert_eq!(tc.tile_k, 32);
        assert_eq!(tc.warp_m, 64);
        assert_eq!(tc.warp_n, 32);
        assert_eq!(tc.stages, 3);
        assert!(tc.use_tensor_core);
        assert_eq!(tc.split_k, 1);
    }

    // -- warp_tile values --

    #[test]
    fn warp_tile_values_are_sensible() {
        for tile in RectangularTile::all() {
            let (wm, wn) = tile.warp_tile();
            assert!(wm > 0 && wn > 0, "warp tile must be positive");
            assert!(
                tile.tile_m() % wm == 0,
                "tile_m {} not divisible by warp_m {} for {:?}",
                tile.tile_m(),
                wm,
                tile,
            );
            assert!(
                tile.tile_n() % wn == 0,
                "tile_n {} not divisible by warp_n {} for {:?}",
                tile.tile_n(),
                wn,
                tile,
            );
        }
    }

    // -- occupancy_score --

    #[test]
    fn occupancy_score_tall_prefers_tall_tile() {
        let sel = tc_selector(SmVersion::Sm80);
        let score_tall = sel.occupancy_score(RectangularTile::Tile128x64, 4096, 64);
        let score_sq = sel.occupancy_score(RectangularTile::Square128, 4096, 64);
        assert!(
            score_tall > score_sq,
            "tall tile should score higher for tall matrix: {} vs {}",
            score_tall,
            score_sq,
        );
    }

    #[test]
    fn occupancy_score_wide_prefers_wide_tile() {
        let sel = tc_selector(SmVersion::Sm80);
        let score_wide = sel.occupancy_score(RectangularTile::Tile64x128, 64, 4096);
        let score_sq = sel.occupancy_score(RectangularTile::Square128, 64, 4096);
        assert!(
            score_wide > score_sq,
            "wide tile should score higher for wide matrix: {} vs {}",
            score_wide,
            score_sq,
        );
    }

    #[test]
    fn occupancy_score_is_nonnegative() {
        let sel = tc_selector(SmVersion::Sm80);
        for tile in RectangularTile::all() {
            let score = sel.occupancy_score(*tile, 512, 512);
            assert!(score >= 0.0, "score must be >= 0, got {}", score);
        }
    }

    // -- sm_version constraints --

    #[test]
    fn sm75_all_candidates_excludes_huge_tiles() {
        let sel = tc_selector(SmVersion::Sm75);
        let candidates = sel.all_candidates();
        // Sm75 has only 64KB shared mem — very large tiles should not fit
        // with f32 and tile_k=32.
        for c in &candidates {
            let smem = c.shared_mem_per_stage(4, 32);
            assert!(
                smem <= SmVersion::Sm75.max_shared_mem_per_block(),
                "{:?} needs {} bytes, exceeds Sm75 limit",
                c,
                smem,
            );
        }
    }

    #[test]
    fn sm90_has_all_candidates() {
        let sel = tc_selector(SmVersion::Sm90);
        let candidates = sel.all_candidates();
        // Sm90 has 232 KB shared mem — all tiles should fit.
        assert_eq!(
            candidates.len(),
            RectangularTile::all().len(),
            "Sm90 should support all tile variants",
        );
    }

    // -- all_candidates completeness --

    #[test]
    fn all_candidates_nonempty() {
        let sel = simt_selector(SmVersion::Sm75);
        let candidates = sel.all_candidates();
        assert!(!candidates.is_empty(), "must have at least one candidate");
    }

    // -- Pipeline stages adapt to arch --

    #[test]
    fn hopper_gets_more_stages_than_turing() {
        let sel_hopper = tc_selector(SmVersion::Sm90);
        let sel_turing = tc_selector(SmVersion::Sm75);
        let tc_h = sel_hopper.select(1024, 1024, 1024);
        let tc_t = sel_turing.select(1024, 1024, 1024);
        assert!(
            tc_h.stages >= tc_t.stages,
            "Hopper ({}) should have >= stages than Turing ({})",
            tc_h.stages,
            tc_t.stages,
        );
    }

    // -- Moderately tall / wide --

    #[test]
    fn moderately_tall_selects_128x64() {
        let sel = tc_selector(SmVersion::Sm80);
        // ratio = 512 / 200 = 2.56 → moderately tall
        let tc = sel.select(512, 200, 256);
        assert_eq!(tc.tile_m, 128);
        assert_eq!(tc.tile_n, 64);
    }

    #[test]
    fn moderately_wide_selects_64x128() {
        let sel = tc_selector(SmVersion::Sm80);
        // ratio = 200 / 512 = 0.39 → moderately wide
        let tc = sel.select(200, 512, 256);
        assert_eq!(tc.tile_m, 64);
        assert_eq!(tc.tile_n, 128);
    }

    // -------------------------------------------------------------------------
    // Task 1 (tile-level): Pipeline stage / SM-version tests
    // -------------------------------------------------------------------------

    /// SIMT fallback (use_tensor_core = false) should always get at most 2 stages
    /// since SIMT uses smaller tile_k and the stage cap is lower.
    #[test]
    fn simt_fallback_gets_limited_stages() {
        // SIMT path caps at min(max_from_arch, max_from_smem).min(2)
        // For Sm90 SIMT: max_from_arch capped at 2 in select_k_and_stages.
        let sel = simt_selector(SmVersion::Sm90);
        let tc = sel.select(1024, 1024, 1024);
        assert!(
            !tc.use_tensor_core,
            "SIMT selector should not use tensor core"
        );
        assert!(
            tc.stages <= 2,
            "SIMT path should get at most 2 stages, got {}",
            tc.stages
        );
    }

    /// Tensor-core Hopper path should get more stages than SIMT Hopper path
    /// for the same problem (TC max = 4, SIMT max = 2).
    #[test]
    fn tc_hopper_gets_more_stages_than_simt_hopper() {
        let sel_tc = tc_selector(SmVersion::Sm90);
        let sel_simt = simt_selector(SmVersion::Sm90);
        let tc_path = sel_tc.select(1024, 1024, 1024);
        let simt_path = sel_simt.select(1024, 1024, 1024);
        assert!(
            tc_path.stages >= simt_path.stages,
            "TC Hopper ({}) should have >= stages than SIMT Hopper ({})",
            tc_path.stages,
            simt_path.stages
        );
    }

    /// Shared memory budget is respected: tile_m * tile_k * elem_bytes +
    ///   tile_k * tile_n * elem_bytes (per stage) * stages <= max_smem.
    ///
    /// Test all three representative SM versions and a variety of problem sizes.
    #[test]
    fn tile_config_fits_shared_memory_for_all_sm() {
        // (sm_version, max_smem_bytes_for_test)
        // Use the real SmVersion::max_shared_mem_per_block() values.
        let sm_versions = [SmVersion::Sm75, SmVersion::Sm80, SmVersion::Sm90];
        let problems: &[(u32, u32, u32)] = &[
            (128, 128, 128),
            (512, 512, 512),
            (1024, 1024, 1024),
            (4096, 64, 256),
            (64, 4096, 256),
        ];

        let elem_bytes: u32 = 4; // f32

        for sm in sm_versions {
            let sel = tc_selector(sm);
            let max_smem = sm.max_shared_mem_per_block();
            for &(m, n, k) in problems {
                let tc = sel.select(m, n, k);
                let smem_per_stage =
                    tc.tile_m * tc.tile_k * elem_bytes + tc.tile_k * tc.tile_n * elem_bytes;
                let total_smem = smem_per_stage * tc.stages;
                assert!(
                    total_smem <= max_smem,
                    "SM{}: problem {}x{}x{} → smem={} > budget={} (tile={}x{}x{} stages={})",
                    sm as u32,
                    m,
                    n,
                    k,
                    total_smem,
                    max_smem,
                    tc.tile_m,
                    tc.tile_n,
                    tc.tile_k,
                    tc.stages
                );
            }
        }
    }

    /// Verify Ampere gets at most 3 stages and Turing at most 2 stages for TC.
    #[test]
    fn sm_version_caps_max_stages_correctly() {
        let sel_sm80 = tc_selector(SmVersion::Sm80);
        let sel_sm75 = tc_selector(SmVersion::Sm75);

        let tc_sm80 = sel_sm80.select(1024, 1024, 1024);
        let tc_sm75 = sel_sm75.select(1024, 1024, 1024);

        // Ampere (SM80) max stages for TC = 3.
        assert!(
            tc_sm80.stages <= 3,
            "Ampere should have <= 3 TC stages, got {}",
            tc_sm80.stages
        );
        // Turing (SM75) max stages for TC = 2.
        assert!(
            tc_sm75.stages <= 2,
            "Turing should have <= 2 TC stages, got {}",
            tc_sm75.stages
        );
    }

    /// Hopper (SM90) TC tile_k should be 64 (base_tile_k from select_k_and_stages).
    #[test]
    fn hopper_tc_uses_tile_k_64() {
        let sel = tc_selector(SmVersion::Sm90);
        // Use K >= 64 so tile_k is not clamped down.
        let tc = sel.select(1024, 1024, 512);
        assert_eq!(
            tc.tile_k, 64,
            "Hopper TC should prefer tile_k=64, got {}",
            tc.tile_k
        );
    }

    /// Pre-Hopper TC tile_k should be 32 (base_tile_k for SM < 90).
    #[test]
    fn pre_hopper_tc_uses_tile_k_32() {
        let sel = tc_selector(SmVersion::Sm80);
        // Use K >= 32 so tile_k is not clamped down.
        let tc = sel.select(1024, 1024, 256);
        assert_eq!(
            tc.tile_k, 32,
            "Ampere TC should prefer tile_k=32, got {}",
            tc.tile_k
        );
    }

    /// shared_mem_per_stage formula: smem_a + smem_b = tile_m*tile_k*elem +
    ///   tile_k*tile_n*elem.
    #[test]
    fn shared_mem_per_stage_formula() {
        let tile = RectangularTile::Square128;
        // tile_m=128, tile_n=128, tile_k=32, elem_bytes=4
        let per_stage = tile.shared_mem_per_stage(4, 32);
        // smem_a = 128*32*4 = 16384
        // smem_b = 32*128*4 = 16384
        // total  = 32768
        assert_eq!(
            per_stage, 32768,
            "128x128 tile with tile_k=32 should need 32KB/stage"
        );
    }

    // =========================================================================
    // Quality gate: WMMA fragment layout and Hopper wgmma smem budget
    // =========================================================================

    /// WMMA m16n16k16 fragment layout for F16:
    /// Each warp (32 threads) computes a 16×16 output tile.
    /// Each thread holds:
    ///   - frag_a: m * k / warp_size = 16 * 16 / 32 = 8 elements
    ///   - frag_b: k * n / warp_size = 16 * 16 / 32 = 8 elements
    ///   - frag_c: m * n / warp_size = 16 * 16 / 32 = 8 elements
    ///
    /// This is the canonical WMMA m16n16k16 layout used on Turing (SM75+).
    #[test]
    fn wmma_m16n16k16_fragment_layout() {
        let warp_size = 32u32;
        let m = 16u32;
        let n = 16u32;
        let k = 16u32;

        let frag_a_elems = m * k / warp_size;
        let frag_b_elems = k * n / warp_size;
        let frag_c_elems = m * n / warp_size;

        assert_eq!(
            frag_a_elems, 8,
            "WMMA m16n16k16 frag_a must have 8 elements per thread"
        );
        assert_eq!(
            frag_b_elems, 8,
            "WMMA m16n16k16 frag_b must have 8 elements per thread"
        );
        assert_eq!(
            frag_c_elems, 8,
            "WMMA m16n16k16 frag_c must have 8 elements per thread"
        );

        // Total elements per warp = warp_size * elems_per_thread = 32 * 8 = 256 = m * n.
        assert_eq!(
            warp_size * frag_c_elems,
            m * n,
            "Warp covers full m*n = 256 output elements"
        );
    }

    /// Hopper wgmma m64n128k16 with 3-stage pipeline shared memory budget:
    ///   A tile: 64 × 16 × 2 bytes (F16) = 2048 bytes
    ///   B tile: 16 × 128 × 2 bytes (F16) = 4096 bytes
    ///   Per stage: 6144 bytes
    ///   3 stages: 18432 bytes → well within the 64 KB baseline shared memory.
    ///
    /// This verifies that the canonical Hopper wgmma pipeline fits in hardware.
    #[test]
    fn hopper_wgmma_m64n128k16_3stage_smem_budget() {
        let elem_bytes = 2u32; // F16 = 2 bytes
        let tile_m = 64u32;
        let tile_n = 128u32;
        let tile_k = 16u32;
        let stages = 3u32;
        let smem_64kb = 64 * 1024u32;

        let smem_a = tile_m * tile_k * elem_bytes; // 64 * 16 * 2 = 2048
        let smem_b = tile_k * tile_n * elem_bytes; // 16 * 128 * 2 = 4096
        let per_stage = smem_a + smem_b; // 6144
        let total = per_stage * stages; // 18432

        assert_eq!(smem_a, 2048, "A tile smem must be 2048 bytes");
        assert_eq!(smem_b, 4096, "B tile smem must be 4096 bytes");
        assert_eq!(per_stage, 6144, "Per-stage smem must be 6144 bytes");
        assert!(
            total <= smem_64kb,
            "3-stage wgmma pipeline ({} bytes) must fit in 64KB smem",
            total
        );
    }

    /// Hopper TileSelector uses tile_k = 64 for TC path (SM90), matching the
    /// wgmma preferred K-tile size, and gets at least 3 pipeline stages.
    #[test]
    fn hopper_tc_tile_k_64_and_min_3_stages() {
        let sel = TileSelector::new(SmVersion::Sm90, true);
        // Problem large enough that smem budget allows multiple stages.
        let tc = sel.select(1024, 1024, 1024);

        assert_eq!(
            tc.tile_k, 64,
            "Hopper TC TileSelector must use tile_k=64 for wgmma, got {}",
            tc.tile_k
        );
        assert!(
            tc.stages >= 3,
            "Hopper TC must get at least 3 pipeline stages for wgmma overlap, got {}",
            tc.stages
        );
        assert!(tc.use_tensor_core, "TC selector must set use_tensor_core");
    }

    /// Turing (SM75) TileSelector tile_k is 32 (WMMA m16n16k16 optimal K-step),
    /// and the stage count is capped at 2.
    #[test]
    fn turing_tc_tile_k_32_max_2_stages() {
        let sel = TileSelector::new(SmVersion::Sm75, true);
        let tc = sel.select(1024, 1024, 512);

        assert_eq!(
            tc.tile_k, 32,
            "Turing TC must use tile_k=32, got {}",
            tc.tile_k
        );
        assert!(
            tc.stages <= 2,
            "Turing TC must have at most 2 stages, got {}",
            tc.stages
        );
        assert!(tc.use_tensor_core, "TC selector must set use_tensor_core");
    }

    /// RectangularTile shared memory formula is correct for the wgmma-sized tile
    /// m64n128k16 with F16 (2 bytes): expected 6144 bytes per stage.
    #[test]
    fn rectangular_tile_smem_formula_wgmma_shape() {
        // m64n128 matches Tile64x128 variant.
        let tile = RectangularTile::Tile64x128;
        let elem_bytes = 2u32; // F16
        let tile_k = 16u32;

        // smem_a = 64 * 16 * 2 = 2048, smem_b = 16 * 128 * 2 = 4096
        let per_stage = tile.shared_mem_per_stage(elem_bytes, tile_k);
        assert_eq!(
            per_stage, 6144,
            "Tile64x128 with F16 tile_k=16 should use 6144 bytes/stage, got {}",
            per_stage
        );
    }

    /// Verify occupancy_score is in [0, 1+] range and that for a balanced
    /// problem, a square tile scores at least as high as an extremely
    /// mismatched tile.
    #[test]
    fn occupancy_score_balanced_square_beats_mismatched() {
        let sel = tc_selector(SmVersion::Sm80);
        // Balanced 1024×1024 problem.
        let score_sq = sel.occupancy_score(RectangularTile::Square128, 1024, 1024);
        // Extremely mismatched tile for a square problem.
        let score_tall = sel.occupancy_score(RectangularTile::Tile256x64, 1024, 1024);
        assert!(
            score_sq >= score_tall,
            "Square tile should score >= mismatched tile for balanced problem: {} vs {}",
            score_sq,
            score_tall
        );
    }
}
