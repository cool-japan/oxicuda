//! Runtime kernel dispatch with tiered fallback.
//!
//! The [`Dispatcher`] selects the best kernel configuration for a given
//! problem at runtime, using a 3-tier strategy:
//!
//! 1. **Exact match** — look up the (GPU, kernel, problem) triple in
//!    the persistent result database.
//! 2. **Nearest neighbor** — find the closest matching problem key
//!    (useful when the exact matrix size has not been benchmarked).
//! 3. **Default config** — fall back to a sensible default
//!    configuration.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_autotune::{Dispatcher, Config};
//!
//! # fn example() -> Result<(), oxicuda_autotune::AutotuneError> {
//! let dispatcher = Dispatcher::new("NVIDIA RTX 4090".to_string())?;
//!
//! let config = dispatcher.select_config("sgemm", "1024x1024x1024");
//! println!("Selected tile_m={}, tile_n={}", config.tile_m, config.tile_n);
//! # Ok(())
//! # }
//! ```

use crate::config::Config;
use crate::error::AutotuneError;
use crate::result_db::ResultDb;

/// Runtime dispatcher for optimal kernel selection.
///
/// Implements a 3-tier fallback strategy for selecting the best kernel
/// configuration without requiring a GPU benchmark at every program
/// startup.  See the [module-level documentation](self) for details.
pub struct Dispatcher {
    /// Persistent result database.
    result_db: ResultDb,
    /// GPU name used as the top-level lookup key.
    gpu_name: String,
    /// Fallback configuration when no database match is found.
    default_config: Config,
}

impl Dispatcher {
    /// Creates a new dispatcher for the given GPU.
    ///
    /// Opens (or creates) the default result database.  The
    /// `gpu_name` is used as the top-level key for all lookups.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::IoError`] if the result database
    /// cannot be opened or created.
    pub fn new(gpu_name: String) -> Result<Self, AutotuneError> {
        let result_db = ResultDb::open()?;
        Ok(Self {
            result_db,
            gpu_name,
            default_config: Config::default(),
        })
    }

    /// Creates a new dispatcher with a custom result database and
    /// default configuration.
    ///
    /// This is useful for testing or when the database path is
    /// explicitly controlled.
    pub fn with_db_and_default(
        result_db: ResultDb,
        gpu_name: String,
        default_config: Config,
    ) -> Self {
        Self {
            result_db,
            gpu_name,
            default_config,
        }
    }

    /// Selects the best configuration for a problem using the 3-tier
    /// fallback strategy.
    ///
    /// 1. **Tier 1** — exact match in the result database.
    /// 2. **Tier 2** — nearest neighbor interpolation.
    /// 3. **Tier 3** — the default configuration.
    ///
    /// This method never fails; it always returns at least the default.
    #[must_use]
    pub fn select_config(&self, kernel_name: &str, problem_key: &str) -> Config {
        // Tier 1: exact match
        if let Some(result) = self
            .result_db
            .lookup(&self.gpu_name, kernel_name, problem_key)
        {
            tracing::debug!(
                gpu = %self.gpu_name,
                kernel = kernel_name,
                problem = problem_key,
                "autotune: exact match found (median={:.1} us)",
                result.median_us,
            );
            return result.config.clone();
        }

        // Tier 2: nearest neighbor
        if let Some(result) =
            self.result_db
                .lookup_nearest(&self.gpu_name, kernel_name, problem_key)
        {
            tracing::debug!(
                gpu = %self.gpu_name,
                kernel = kernel_name,
                problem = problem_key,
                "autotune: nearest neighbor match (median={:.1} us)",
                result.median_us,
            );
            return result.config.clone();
        }

        // Tier 3: default
        tracing::debug!(
            gpu = %self.gpu_name,
            kernel = kernel_name,
            problem = problem_key,
            "autotune: using default config",
        );
        self.default_config.clone()
    }

    /// Selects the best configuration, returning the tier that was used.
    ///
    /// The returned `DispatchTier` indicates which fallback level
    /// produced the result, which can be useful for logging or
    /// deciding whether to trigger an on-demand tuning run.
    #[must_use]
    pub fn select_config_with_tier(
        &self,
        kernel_name: &str,
        problem_key: &str,
    ) -> (Config, DispatchTier) {
        // Tier 1
        if let Some(result) = self
            .result_db
            .lookup(&self.gpu_name, kernel_name, problem_key)
        {
            return (result.config.clone(), DispatchTier::ExactMatch);
        }

        // Tier 2
        if let Some(result) =
            self.result_db
                .lookup_nearest(&self.gpu_name, kernel_name, problem_key)
        {
            return (result.config.clone(), DispatchTier::NearestNeighbor);
        }

        // Tier 3
        (self.default_config.clone(), DispatchTier::Default)
    }

    /// Returns a reference to the GPU name this dispatcher targets.
    #[must_use]
    pub fn gpu_name(&self) -> &str {
        &self.gpu_name
    }

    /// Returns a reference to the default configuration.
    #[must_use]
    pub fn default_config(&self) -> &Config {
        &self.default_config
    }

    /// Sets a new default configuration for Tier 3 fallback.
    pub fn set_default_config(&mut self, config: Config) {
        self.default_config = config;
    }

    /// Returns a mutable reference to the underlying result database.
    ///
    /// This allows saving new benchmark results through the dispatcher.
    pub fn result_db_mut(&mut self) -> &mut ResultDb {
        &mut self.result_db
    }

    /// Returns a reference to the underlying result database.
    #[must_use]
    pub fn result_db(&self) -> &ResultDb {
        &self.result_db
    }
}

/// Indicates which tier of the dispatch fallback strategy produced
/// the selected configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchTier {
    /// Tier 1: exact (GPU, kernel, problem) match in the database.
    ExactMatch,
    /// Tier 2: nearest-neighbor approximation from the database.
    NearestNeighbor,
    /// Tier 3: the default configuration (no database match).
    Default,
}

impl std::fmt::Display for DispatchTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExactMatch => write!(f, "exact-match"),
            Self::NearestNeighbor => write!(f, "nearest-neighbor"),
            Self::Default => write!(f, "default"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkResult;

    #[test]
    fn dispatcher_tier3_when_empty() {
        let dir = std::env::temp_dir().join("oxicuda_test_dispatch_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("results.json");

        let db = ResultDb::open_at(path).expect("open");
        let dispatcher =
            Dispatcher::with_db_and_default(db, "GPU0".to_string(), Config::new().with_tile_m(99));

        let (cfg, tier) = dispatcher.select_config_with_tier("sgemm", "1024x1024");
        assert_eq!(tier, DispatchTier::Default);
        assert_eq!(cfg.tile_m, 99);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dispatcher_tier1_exact_match() {
        let dir = std::env::temp_dir().join("oxicuda_test_dispatch_exact");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("results.json");

        let mut db = ResultDb::open_at(path).expect("open");
        let result = BenchmarkResult {
            config: Config::new().with_tile_m(64),
            median_us: 10.0,
            min_us: 9.0,
            max_us: 11.0,
            stddev_us: 0.5,
            gflops: None,
            efficiency: None,
        };
        db.save("GPU0", "sgemm", "1024x1024", result).expect("save");

        let dispatcher = Dispatcher::with_db_and_default(db, "GPU0".to_string(), Config::default());

        let (cfg, tier) = dispatcher.select_config_with_tier("sgemm", "1024x1024");
        assert_eq!(tier, DispatchTier::ExactMatch);
        assert_eq!(cfg.tile_m, 64);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dispatch_tier_display() {
        assert_eq!(format!("{}", DispatchTier::ExactMatch), "exact-match");
        assert_eq!(
            format!("{}", DispatchTier::NearestNeighbor),
            "nearest-neighbor"
        );
        assert_eq!(format!("{}", DispatchTier::Default), "default");
    }

    // -----------------------------------------------------------------------
    // RAII cleanup guard — keeps temp dirs tidy even when tests panic.
    // -----------------------------------------------------------------------
    struct CleanupGuard(std::path::PathBuf);

    impl Drop for CleanupGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tmp_db(suffix: &str) -> (ResultDb, CleanupGuard) {
        let dir = std::env::temp_dir().join(format!("oxicuda_disp_{suffix}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let guard = CleanupGuard(dir.clone());
        let db = ResultDb::open_at(dir.join("results.json")).expect("open db");
        (db, guard)
    }

    fn make_result_with_config(config: Config, median_us: f64) -> BenchmarkResult {
        BenchmarkResult {
            config,
            median_us,
            min_us: median_us - 1.0,
            max_us: median_us + 1.0,
            stddev_us: 0.5,
            gflops: None,
            efficiency: None,
        }
    }

    // -----------------------------------------------------------------------
    // Tier 1: when a result exists for the exact (gpu, kernel, problem_size),
    //         it is returned directly and the tier is ExactMatch.
    // -----------------------------------------------------------------------
    #[test]
    fn test_dispatcher_exact_match() {
        let (mut db, _guard) = tmp_db("exact_match");

        let expected_config = Config::new().with_tile_m(64).with_tile_n(64).with_stages(3);
        db.save(
            "RTX4090",
            "sgemm",
            "1024x1024x1024",
            make_result_with_config(expected_config.clone(), 20.0),
        )
        .expect("save");

        let dispatcher =
            Dispatcher::with_db_and_default(db, "RTX4090".to_string(), Config::default());

        let (cfg, tier) = dispatcher.select_config_with_tier("sgemm", "1024x1024x1024");

        assert_eq!(
            tier,
            DispatchTier::ExactMatch,
            "expected Tier 1 exact match"
        );
        assert_eq!(
            cfg, expected_config,
            "config from exact match must equal stored config"
        );
    }

    // -----------------------------------------------------------------------
    // Tier 2: when no exact match exists but a nearby problem key does,
    //         lookup_nearest() returns the closest match.
    // -----------------------------------------------------------------------
    #[test]
    fn test_dispatcher_nearest_match() {
        let (mut db, _guard) = tmp_db("nearest_match");

        // Store a result for "1024x1024x1024".
        let stored_config = Config::new().with_tile_m(128).with_tile_n(128);
        db.save(
            "A100",
            "sgemm",
            "1024x1024x1024",
            make_result_with_config(stored_config.clone(), 15.0),
        )
        .expect("save");

        let dispatcher = Dispatcher::with_db_and_default(
            db,
            "A100".to_string(),
            Config::new().with_tile_m(32), // default is clearly different
        );

        // Query with a problem key that has no exact match but is lexicographically
        // close to the stored key.  The nearest-neighbor lookup should find it.
        let (cfg, tier) = dispatcher.select_config_with_tier("sgemm", "1024x1024x512");

        assert_eq!(
            tier,
            DispatchTier::NearestNeighbor,
            "expected Tier 2 nearest-neighbor match",
        );
        assert_eq!(
            cfg, stored_config,
            "nearest match must return the stored config",
        );
    }

    // -----------------------------------------------------------------------
    // Tier 3: when no results exist at all, the default config is returned.
    // -----------------------------------------------------------------------
    #[test]
    fn test_dispatcher_default_config_fallback() {
        let (db, _guard) = tmp_db("default_fallback");

        let default_cfg = Config::new().with_tile_m(77).with_tile_n(77);
        let dispatcher =
            Dispatcher::with_db_and_default(db, "V100".to_string(), default_cfg.clone());

        let (cfg, tier) = dispatcher.select_config_with_tier("sgemm", "512x512x512");

        assert_eq!(
            tier,
            DispatchTier::Default,
            "expected Tier 3 default fallback"
        );
        assert_eq!(
            cfg, default_cfg,
            "default fallback must return the stored default config"
        );
    }

    // -----------------------------------------------------------------------
    // Validate all three tiers in sequence against the same dispatcher:
    //   1. Exact match for a known problem key.
    //   2. Nearest match for a similar but unregistered problem key.
    //   3. Default for a kernel not in the database at all.
    // -----------------------------------------------------------------------
    #[test]
    fn test_dispatcher_all_three_tiers() {
        let (mut db, _guard) = tmp_db("all_three_tiers");

        let exact_config = Config::new().with_tile_m(64);
        let default_config = Config::new().with_tile_m(16);

        // Register only one result: GPU="H100", kernel="sgemm", problem="2048x2048".
        db.save(
            "H100",
            "sgemm",
            "2048x2048",
            make_result_with_config(exact_config.clone(), 8.0),
        )
        .expect("save");

        let dispatcher =
            Dispatcher::with_db_and_default(db, "H100".to_string(), default_config.clone());

        // Tier 1 — exact match.
        let (cfg1, tier1) = dispatcher.select_config_with_tier("sgemm", "2048x2048");
        assert_eq!(tier1, DispatchTier::ExactMatch);
        assert_eq!(cfg1, exact_config);

        // Tier 2 — nearest match (problem key differs from stored one).
        let (cfg2, tier2) = dispatcher.select_config_with_tier("sgemm", "2048x1024");
        assert_eq!(tier2, DispatchTier::NearestNeighbor);
        assert_eq!(
            cfg2, exact_config,
            "nearest match should return the only stored config"
        );

        // Tier 3 — no data for "dgemm" at all, falls back to default.
        let (cfg3, tier3) = dispatcher.select_config_with_tier("dgemm", "2048x2048");
        assert_eq!(tier3, DispatchTier::Default);
        assert_eq!(cfg3, default_config);
    }

    // -----------------------------------------------------------------------
    // `lookup_nearest()` distance metric: nearest neighbor returns the
    // lexicographically closest problem key measured by Levenshtein distance.
    //
    // We use zero-padded 3-digit keys of the form `"m=064"`, `"m=128"`,
    // `"m=256"` to ensure the edit-distance winners are unambiguous:
    //   * query `"m=100"` → distance 2 to `"m=128"`, 3 to others.
    //   * query `"m=070"` → distance 2 to `"m=064"`, 3 to others.
    // -----------------------------------------------------------------------

    #[test]
    fn dispatcher_lookup_nearest_distance_metric() {
        let (mut db, _guard) = tmp_db("nearest_distance");

        // Three configs stored under problem keys that encode the M size.
        let cfg_64 = Config::new().with_tile_m(64).with_tile_n(32);
        let cfg_128 = Config::new().with_tile_m(128).with_tile_n(32);
        let cfg_256 = Config::new().with_tile_m(256).with_tile_n(32);

        db.save(
            "A100",
            "sgemm",
            "m=064",
            make_result_with_config(cfg_64.clone(), 30.0),
        )
        .expect("save m=064");
        db.save(
            "A100",
            "sgemm",
            "m=128",
            make_result_with_config(cfg_128.clone(), 20.0),
        )
        .expect("save m=128");
        db.save(
            "A100",
            "sgemm",
            "m=256",
            make_result_with_config(cfg_256.clone(), 25.0),
        )
        .expect("save m=256");

        let dispatcher = Dispatcher::with_db_and_default(db, "A100".to_string(), Config::default());

        // Query "m=100" is closest to "m=128" (edit distance 2 vs 3 for others).
        let (cfg_near_128, tier_128) = dispatcher.select_config_with_tier("sgemm", "m=100");
        assert_eq!(
            tier_128,
            DispatchTier::NearestNeighbor,
            "m=100 should resolve via nearest-neighbor"
        );
        assert_eq!(
            cfg_near_128.tile_m, 128,
            "m=100 should map to the m=128 config (nearest by edit distance), \
             got tile_m={}",
            cfg_near_128.tile_m
        );

        // Query "m=070" is closest to "m=064" (edit distance 2 vs 3 for others).
        let (cfg_near_64, tier_64) = dispatcher.select_config_with_tier("sgemm", "m=070");
        assert_eq!(
            tier_64,
            DispatchTier::NearestNeighbor,
            "m=070 should resolve via nearest-neighbor"
        );
        assert_eq!(
            cfg_near_64.tile_m, 64,
            "m=070 should map to the m=064 config (nearest by edit distance), \
             got tile_m={}",
            cfg_near_64.tile_m
        );
    }
}
