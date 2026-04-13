//! Autotune-guided PTX generation.
//!
//! Closes the feedback loop between autotuning results and PTX code
//! generation.  Given a populated [`ResultDb`], the [`GuidedPtxGenerator`]
//! extracts optimal kernel parameters and produces [`PtxGenerationHint`]s
//! that PTX template instantiation can consume directly.
//!
//! # Strategies
//!
//! | Strategy        | Description                                             |
//! |-----------------|---------------------------------------------------------|
//! | `BestKnown`     | Single best config from the database                    |
//! | `TopK { k }`    | Top-K configs for runtime selection                     |
//! | `Interpolated`  | Interpolate from nearby results for exact problem size  |
//! | `Adaptive`      | Switch strategy when performance drops below threshold  |
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::guided_ptx::*;
//! use oxicuda_autotune::{Config, BenchmarkResult, ResultDb};
//!
//! let hint = PtxGenerationHint::from_config(&Config::new());
//! assert_eq!(hint.tile_m, 128);
//! assert_eq!(hint.tile_n, 128);
//! ```

use std::collections::HashMap;

use crate::benchmark::BenchmarkResult;
use crate::config::Config;
use crate::error::AutotuneError;
use crate::result_db::ResultDb;

// ---------------------------------------------------------------------------
// PtxGenerationHint
// ---------------------------------------------------------------------------

/// Hints derived from autotuning results that guide PTX code generation.
///
/// Each field corresponds to a tunable parameter in a GPU kernel template.
/// Optional fields (`unroll_factor`, `vectorize_width`, `prefetch_distance`)
/// are populated only when the autotuner has enough data to make a
/// confident recommendation.
#[derive(Debug, Clone, PartialEq)]
pub struct PtxGenerationHint {
    /// Block tile size in the M (row) dimension.
    pub tile_m: u32,
    /// Block tile size in the N (column) dimension.
    pub tile_n: u32,
    /// Block tile size in the K (reduction) dimension.
    pub tile_k: u32,
    /// Number of software pipeline stages.
    pub pipeline_depth: u32,
    /// Threads per thread block.
    pub threads: u32,
    /// Whether to emit Tensor Core (WMMA/MMA) instructions.
    pub use_tensor_cores: bool,
    /// Warp tile size in the M dimension.
    pub warp_tile_m: u32,
    /// Warp tile size in the N dimension.
    pub warp_tile_n: u32,
    /// Estimated shared memory usage in bytes.
    pub shared_memory_bytes: u32,
    /// Suggested loop unroll factor (e.g. 2, 4, 8).
    pub unroll_factor: Option<u32>,
    /// Suggested vector load width (1, 2, or 4 elements).
    pub vectorize_width: Option<u32>,
    /// Prefetch lookahead distance in pipeline stages.
    pub prefetch_distance: Option<u32>,
}

impl PtxGenerationHint {
    /// Construct a hint from an autotuning [`Config`].
    ///
    /// Maps each `Config` field to the corresponding hint field.
    /// Optional hint fields (`unroll_factor`, `vectorize_width`,
    /// `prefetch_distance`) are extracted from `Config::extra` when
    /// the keys `"unroll_factor"`, `"vector_width"`, and
    /// `"prefetch_distance"` are present.
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        let smem = config.estimated_shared_mem(4) as u32;
        Self {
            tile_m: config.tile_m,
            tile_n: config.tile_n,
            tile_k: config.tile_k,
            pipeline_depth: config.stages,
            threads: config.block_size,
            use_tensor_cores: config.use_tensor_core,
            warp_tile_m: config.warp_m,
            warp_tile_n: config.warp_n,
            shared_memory_bytes: smem,
            unroll_factor: config.extra.get("unroll_factor").copied(),
            vectorize_width: config.extra.get("vector_width").copied(),
            prefetch_distance: config.extra.get("prefetch_distance").copied(),
        }
    }

    /// Construct a hint from a [`BenchmarkResult`].
    ///
    /// Delegates to [`from_config`](Self::from_config) using the
    /// configuration embedded in the result.
    #[must_use]
    pub fn from_benchmark_result(result: &BenchmarkResult) -> Self {
        Self::from_config(&result.config)
    }
}

// ---------------------------------------------------------------------------
// GuidedPtxStrategy
// ---------------------------------------------------------------------------

/// Strategy for selecting which autotuning results feed PTX generation.
#[derive(Debug, Clone, PartialEq)]
pub enum GuidedPtxStrategy {
    /// Use the single best configuration from the database.
    BestKnown,
    /// Generate PTX for the top-K configurations for runtime dispatch.
    TopK {
        /// Number of top configurations to consider.
        k: usize,
    },
    /// Interpolate parameters for the exact problem size from nearby results.
    Interpolated,
    /// Switch strategy dynamically if performance drops below a threshold.
    Adaptive {
        /// Minimum acceptable throughput in GFLOPS.
        threshold_gflops: f64,
    },
}

// ---------------------------------------------------------------------------
// PtxTemplate trait (feature-gated)
// ---------------------------------------------------------------------------

/// Abstract interface for PTX template instantiation.
///
/// Implementors generate PTX assembly strings from a set of
/// [`PtxGenerationHint`] parameters.  This trait is feature-gated
/// behind the `ptx` feature.
#[cfg(feature = "ptx")]
pub trait PtxTemplate {
    /// Generate a PTX assembly string from the given hints.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::PtxError`] if the hints are incompatible
    /// with the template constraints.
    fn instantiate(&self, hints: &PtxGenerationHint) -> Result<String, AutotuneError>;

    /// Return the human-readable template name (e.g. `"gemm_f32"`).
    fn template_name(&self) -> &str;

    /// List the hint fields that this template actually uses.
    ///
    /// Useful for diagnostics — callers can warn when a hint field is
    /// set but the template ignores it.
    fn supported_hints(&self) -> Vec<String>;
}

// ---------------------------------------------------------------------------
// PtxSpecialization
// ---------------------------------------------------------------------------

/// A grouped specialization: one PTX variant covering multiple problem sizes.
///
/// The [`GuidedPtxGenerator`] groups problems that share the same optimal
/// configuration into a single specialization, reducing the number of
/// distinct PTX variants that need to be compiled.
#[derive(Debug, Clone)]
pub struct PtxSpecialization {
    /// The generation hint for this specialization.
    pub hint: PtxGenerationHint,
    /// Problem sizes covered by this specialization.
    pub problem_sizes: Vec<String>,
    /// Average expected throughput across covered problems (GFLOPS).
    pub expected_gflops: f64,
    /// Fraction of all known problems covered by this specialization (0.0-1.0).
    pub coverage: f64,
}

// ---------------------------------------------------------------------------
// GuidedPtxGenerator
// ---------------------------------------------------------------------------

/// Generates PTX hints from autotuning results stored in a [`ResultDb`].
///
/// The generator reads the result database for a specific GPU and applies
/// the chosen [`GuidedPtxStrategy`] to produce [`PtxGenerationHint`]s
/// that can be fed directly into PTX template instantiation.
pub struct GuidedPtxGenerator<'db> {
    /// Reference to the result database.
    db: &'db ResultDb,
    /// GPU name used for database lookups.
    gpu_name: String,
    /// Strategy for selecting configurations.
    strategy: GuidedPtxStrategy,
}

impl<'db> GuidedPtxGenerator<'db> {
    /// Create a new generator for the given GPU and strategy.
    #[must_use]
    pub fn new(db: &'db ResultDb, gpu_name: String, strategy: GuidedPtxStrategy) -> Self {
        Self {
            db,
            gpu_name,
            strategy,
        }
    }

    /// Generate a [`PtxGenerationHint`] for a single kernel + problem pair.
    ///
    /// The strategy determines how the hint is derived:
    ///
    /// - **BestKnown**: exact or nearest-neighbor lookup.
    /// - **TopK**: picks the best from top-K results (same as BestKnown
    ///   for a single problem).
    /// - **Interpolated**: nearest-neighbor interpolation.
    /// - **Adaptive**: falls back to default if stored performance is
    ///   below the threshold.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::NoViableConfig`] when no result is found
    /// and no fallback is possible.
    pub fn generate_hints(
        &self,
        kernel: &str,
        problem: &str,
    ) -> Result<PtxGenerationHint, AutotuneError> {
        match &self.strategy {
            GuidedPtxStrategy::BestKnown | GuidedPtxStrategy::TopK { .. } => {
                self.hint_from_best(kernel, problem)
            }
            GuidedPtxStrategy::Interpolated => self.hint_interpolated(kernel, problem),
            GuidedPtxStrategy::Adaptive { threshold_gflops } => {
                let threshold = *threshold_gflops;
                self.hint_adaptive(kernel, problem, threshold)
            }
        }
    }

    /// Generate hints for multiple problems at once.
    ///
    /// Returns a vec of `(problem_key, hint)` pairs.  Problems that
    /// cannot be resolved are silently skipped.
    ///
    /// # Errors
    ///
    /// Returns an error only if *all* problems fail to resolve.
    pub fn generate_hints_multi(
        &self,
        kernel: &str,
        problems: &[&str],
    ) -> Result<Vec<(String, PtxGenerationHint)>, AutotuneError> {
        let mut results = Vec::new();
        for &p in problems {
            if let Ok(hint) = self.generate_hints(kernel, p) {
                results.push((p.to_string(), hint));
            }
        }
        if results.is_empty() {
            return Err(AutotuneError::NoViableConfig);
        }
        Ok(results)
    }

    /// Group problems by their optimal configuration to minimise PTX variants.
    ///
    /// Problems that share the same best config are merged into a single
    /// [`PtxSpecialization`].  Each specialization reports the average
    /// expected GFLOPS and the fraction of total problems it covers.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::NoViableConfig`] if the database has no
    /// results for this kernel on the target GPU.
    pub fn suggest_specializations(
        &self,
        kernel: &str,
    ) -> Result<Vec<PtxSpecialization>, AutotuneError> {
        let entries = self.db.list_gpu(&self.gpu_name);
        let kernel_entries: Vec<_> = entries
            .into_iter()
            .filter(|(k, _, _)| *k == kernel)
            .collect();

        if kernel_entries.is_empty() {
            return Err(AutotuneError::NoViableConfig);
        }

        let total = kernel_entries.len() as f64;

        // Group by config identity (tile_m, tile_n, tile_k, stages, block_size, tc).
        let mut groups: HashMap<ConfigKey, Vec<(&str, f64)>> = HashMap::new();
        for &(_, problem, result) in &kernel_entries {
            let key = ConfigKey::from_config(&result.config);
            let gflops = result.gflops.unwrap_or(0.0);
            groups.entry(key).or_default().push((problem, gflops));
        }

        let mut specs: Vec<PtxSpecialization> = Vec::new();
        for (key, members) in &groups {
            let avg_gflops = if members.is_empty() {
                0.0
            } else {
                members.iter().map(|(_, g)| g).sum::<f64>() / members.len() as f64
            };
            let coverage = members.len() as f64 / total;

            // Reconstruct config from key to build hint.
            let config = key.to_config();
            let hint = PtxGenerationHint::from_config(&config);

            specs.push(PtxSpecialization {
                hint,
                problem_sizes: members.iter().map(|(p, _)| (*p).to_string()).collect(),
                expected_gflops: avg_gflops,
                coverage,
            });
        }

        // Sort by coverage descending.
        specs.sort_by(|a, b| {
            b.coverage
                .partial_cmp(&a.coverage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(specs)
    }

    // -- Private helpers -------------------------------------------------------

    /// Derive a hint from the best (or nearest) DB entry.
    fn hint_from_best(
        &self,
        kernel: &str,
        problem: &str,
    ) -> Result<PtxGenerationHint, AutotuneError> {
        let result = self
            .db
            .lookup_nearest(&self.gpu_name, kernel, problem)
            .ok_or(AutotuneError::NoViableConfig)?;
        Ok(PtxGenerationHint::from_benchmark_result(result))
    }

    /// Derive a hint via nearest-neighbor interpolation (same as best
    /// for now — delegates to `lookup_nearest`).
    fn hint_interpolated(
        &self,
        kernel: &str,
        problem: &str,
    ) -> Result<PtxGenerationHint, AutotuneError> {
        self.hint_from_best(kernel, problem)
    }

    /// Adaptive: use DB result if it meets the threshold, otherwise
    /// fall back to a default hint.
    fn hint_adaptive(
        &self,
        kernel: &str,
        problem: &str,
        threshold: f64,
    ) -> Result<PtxGenerationHint, AutotuneError> {
        if let Some(result) = self.db.lookup_nearest(&self.gpu_name, kernel, problem) {
            if result.gflops.unwrap_or(0.0) >= threshold {
                return Ok(PtxGenerationHint::from_benchmark_result(result));
            }
        }
        // Fall back to default config.
        Ok(PtxGenerationHint::from_config(&Config::default()))
    }
}

// ---------------------------------------------------------------------------
// HintMerger
// ---------------------------------------------------------------------------

/// Utilities for merging multiple [`PtxGenerationHint`]s into a consensus.
pub struct HintMerger;

impl HintMerger {
    /// Merge multiple hints into a single consensus hint.
    ///
    /// Discrete fields (tile sizes, threads, booleans) use the **mode**
    /// (most frequent value).  Continuous/optional fields use the **mean**
    /// (rounded to nearest integer).  If `hints` is empty, returns the
    /// hint derived from [`Config::default()`].
    #[must_use]
    pub fn merge(hints: &[PtxGenerationHint]) -> PtxGenerationHint {
        if hints.is_empty() {
            return PtxGenerationHint::from_config(&Config::default());
        }
        if hints.len() == 1 {
            return hints[0].clone();
        }

        PtxGenerationHint {
            tile_m: mode_u32(hints.iter().map(|h| h.tile_m)),
            tile_n: mode_u32(hints.iter().map(|h| h.tile_n)),
            tile_k: mode_u32(hints.iter().map(|h| h.tile_k)),
            pipeline_depth: mode_u32(hints.iter().map(|h| h.pipeline_depth)),
            threads: mode_u32(hints.iter().map(|h| h.threads)),
            use_tensor_cores: mode_bool(hints.iter().map(|h| h.use_tensor_cores)),
            warp_tile_m: mode_u32(hints.iter().map(|h| h.warp_tile_m)),
            warp_tile_n: mode_u32(hints.iter().map(|h| h.warp_tile_n)),
            shared_memory_bytes: mode_u32(hints.iter().map(|h| h.shared_memory_bytes)),
            unroll_factor: mean_option(hints.iter().filter_map(|h| h.unroll_factor)),
            vectorize_width: mean_option(hints.iter().filter_map(|h| h.vectorize_width)),
            prefetch_distance: mean_option(hints.iter().filter_map(|h| h.prefetch_distance)),
        }
    }

    /// Compute a conflict score measuring how much the hints disagree.
    ///
    /// Returns 0.0 when all hints are identical, and approaches 1.0 as
    /// disagreement increases.  The score is the fraction of discrete
    /// fields where the mode does not account for all values.
    #[must_use]
    pub fn conflict_score(hints: &[PtxGenerationHint]) -> f64 {
        if hints.len() <= 1 {
            return 0.0;
        }

        let fields: &[Vec<u32>] = &[
            hints.iter().map(|h| h.tile_m).collect(),
            hints.iter().map(|h| h.tile_n).collect(),
            hints.iter().map(|h| h.tile_k).collect(),
            hints.iter().map(|h| h.pipeline_depth).collect(),
            hints.iter().map(|h| h.threads).collect(),
            hints.iter().map(|h| h.warp_tile_m).collect(),
            hints.iter().map(|h| h.warp_tile_n).collect(),
            hints.iter().map(|h| h.shared_memory_bytes).collect(),
        ];

        let total_fields = fields.len() as f64;
        let disagreeing = fields
            .iter()
            .filter(|vals| {
                let first = vals.first();
                !vals.iter().all(|v| Some(v) == first)
            })
            .count() as f64;

        disagreeing / total_fields
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Hashable key for grouping configs by their core parameters.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ConfigKey {
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    stages: u32,
    block_size: u32,
    use_tensor_core: bool,
    warp_m: u32,
    warp_n: u32,
}

impl ConfigKey {
    fn from_config(c: &Config) -> Self {
        Self {
            tile_m: c.tile_m,
            tile_n: c.tile_n,
            tile_k: c.tile_k,
            stages: c.stages,
            block_size: c.block_size,
            use_tensor_core: c.use_tensor_core,
            warp_m: c.warp_m,
            warp_n: c.warp_n,
        }
    }

    fn to_config(&self) -> Config {
        Config::new()
            .with_tile_m(self.tile_m)
            .with_tile_n(self.tile_n)
            .with_tile_k(self.tile_k)
            .with_stages(self.stages)
            .with_block_size(self.block_size)
            .with_use_tensor_core(self.use_tensor_core)
            .with_warp_m(self.warp_m)
            .with_warp_n(self.warp_n)
    }
}

/// Compute the mode (most frequent value) of an iterator of u32.
fn mode_u32(iter: impl Iterator<Item = u32>) -> u32 {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for v in iter {
        *counts.entry(v).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(val, _)| val)
        .unwrap_or(0)
}

/// Compute the mode of an iterator of booleans.
fn mode_bool(iter: impl Iterator<Item = bool>) -> bool {
    let mut true_count = 0usize;
    let mut false_count = 0usize;
    for v in iter {
        if v {
            true_count += 1;
        } else {
            false_count += 1;
        }
    }
    true_count > false_count
}

/// Compute the mean of optional u32 values, returning `None` if empty.
fn mean_option(iter: impl Iterator<Item = u32>) -> Option<u32> {
    let vals: Vec<u32> = iter.collect();
    if vals.is_empty() {
        return None;
    }
    let sum: u64 = vals.iter().map(|&v| u64::from(v)).sum();
    Some((sum / vals.len() as u64) as u32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(tile_m: u32, tile_n: u32, gflops: Option<f64>) -> BenchmarkResult {
        BenchmarkResult {
            config: Config::new().with_tile_m(tile_m).with_tile_n(tile_n),
            median_us: 100.0,
            min_us: 95.0,
            max_us: 110.0,
            stddev_us: 3.0,
            gflops,
            efficiency: None,
        }
    }

    fn make_db_with_entries() -> (ResultDb, std::path::PathBuf) {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tid = std::thread::current().id();
        let dir = std::env::temp_dir().join(format!("oxicuda_guided_ptx_{id}_{tid:?}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("results.json");
        let mut db = ResultDb::open_at(path).expect("open");
        db.save(
            "RTX4090",
            "sgemm",
            "1024x1024x1024",
            make_config(128, 128, Some(1500.0)),
        )
        .expect("save");
        db.save(
            "RTX4090",
            "sgemm",
            "2048x2048x2048",
            make_config(256, 128, Some(1800.0)),
        )
        .expect("save");
        db.save(
            "RTX4090",
            "sgemm",
            "512x512x512",
            make_config(64, 64, Some(900.0)),
        )
        .expect("save");
        (db, dir)
    }

    // -- PtxGenerationHint from Config -----------------------------------------

    #[test]
    fn hint_from_config_maps_fields() {
        let cfg = Config::new()
            .with_tile_m(64)
            .with_tile_n(128)
            .with_tile_k(16)
            .with_stages(3)
            .with_block_size(256)
            .with_use_tensor_core(true)
            .with_warp_m(32)
            .with_warp_n(64)
            .with_extra("unroll_factor", 4)
            .with_extra("vector_width", 2);
        let hint = PtxGenerationHint::from_config(&cfg);
        assert_eq!(hint.tile_m, 64);
        assert_eq!(hint.tile_n, 128);
        assert_eq!(hint.tile_k, 16);
        assert_eq!(hint.pipeline_depth, 3);
        assert_eq!(hint.threads, 256);
        assert!(hint.use_tensor_cores);
        assert_eq!(hint.warp_tile_m, 32);
        assert_eq!(hint.warp_tile_n, 64);
        assert_eq!(hint.unroll_factor, Some(4));
        assert_eq!(hint.vectorize_width, Some(2));
        assert_eq!(hint.prefetch_distance, None);
    }

    // -- PtxGenerationHint from BenchmarkResult --------------------------------

    #[test]
    fn hint_from_benchmark_result() {
        let result = make_config(128, 128, Some(1500.0));
        let hint = PtxGenerationHint::from_benchmark_result(&result);
        assert_eq!(hint.tile_m, 128);
        assert_eq!(hint.tile_n, 128);
        assert_eq!(hint.threads, 256); // default block_size
    }

    // -- GuidedPtxGenerator BestKnown ------------------------------------------

    #[test]
    fn generator_best_known_exact_match() {
        let (db, dir) = make_db_with_entries();
        let generator =
            GuidedPtxGenerator::new(&db, "RTX4090".to_string(), GuidedPtxStrategy::BestKnown);
        let hint = generator
            .generate_hints("sgemm", "1024x1024x1024")
            .expect("hint");
        assert_eq!(hint.tile_m, 128);
        assert_eq!(hint.tile_n, 128);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- GuidedPtxGenerator TopK -----------------------------------------------

    #[test]
    fn generator_top_k_returns_best() {
        let (db, dir) = make_db_with_entries();
        let generator =
            GuidedPtxGenerator::new(&db, "RTX4090".to_string(), GuidedPtxStrategy::TopK { k: 3 });
        let hint = generator
            .generate_hints("sgemm", "2048x2048x2048")
            .expect("hint");
        assert_eq!(hint.tile_m, 256);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- GuidedPtxGenerator Interpolated ---------------------------------------

    #[test]
    fn generator_interpolated_nearest() {
        let (db, dir) = make_db_with_entries();
        let generator =
            GuidedPtxGenerator::new(&db, "RTX4090".to_string(), GuidedPtxStrategy::Interpolated);
        // 768x768 not in DB — should find nearest match.
        let hint = generator
            .generate_hints("sgemm", "768x768x768")
            .expect("hint");
        // Should return one of the stored configs.
        assert!(hint.tile_m == 64 || hint.tile_m == 128 || hint.tile_m == 256);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- GuidedPtxGenerator Adaptive -------------------------------------------

    #[test]
    fn generator_adaptive_above_threshold() {
        let (db, dir) = make_db_with_entries();
        let generator = GuidedPtxGenerator::new(
            &db,
            "RTX4090".to_string(),
            GuidedPtxStrategy::Adaptive {
                threshold_gflops: 1000.0,
            },
        );
        // 1024x1024 has 1500 GFLOPS > 1000 threshold.
        let hint = generator
            .generate_hints("sgemm", "1024x1024x1024")
            .expect("hint");
        assert_eq!(hint.tile_m, 128);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generator_adaptive_below_threshold_falls_back() {
        let (db, dir) = make_db_with_entries();
        let generator = GuidedPtxGenerator::new(
            &db,
            "RTX4090".to_string(),
            GuidedPtxStrategy::Adaptive {
                threshold_gflops: 2000.0,
            },
        );
        // All entries are below 2000 GFLOPS — should fall back to default.
        let hint = generator
            .generate_hints("sgemm", "1024x1024x1024")
            .expect("hint");
        let default_hint = PtxGenerationHint::from_config(&Config::default());
        assert_eq!(hint.tile_m, default_hint.tile_m);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- Missing DB entries (graceful fallback) --------------------------------

    #[test]
    fn generator_missing_kernel_returns_error() {
        let (db, dir) = make_db_with_entries();
        let generator =
            GuidedPtxGenerator::new(&db, "RTX4090".to_string(), GuidedPtxStrategy::BestKnown);
        let result = generator.generate_hints("nonexistent_kernel", "1024x1024");
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generator_missing_gpu_returns_error() {
        let (db, dir) = make_db_with_entries();
        let generator =
            GuidedPtxGenerator::new(&db, "UNKNOWN_GPU".to_string(), GuidedPtxStrategy::BestKnown);
        let result = generator.generate_hints("sgemm", "1024x1024");
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- Multi-problem hint generation -----------------------------------------

    #[test]
    fn generator_multi_problem_hints() {
        let (db, dir) = make_db_with_entries();
        let generator =
            GuidedPtxGenerator::new(&db, "RTX4090".to_string(), GuidedPtxStrategy::BestKnown);
        let problems = &["1024x1024x1024", "2048x2048x2048", "512x512x512"];
        let hints = generator
            .generate_hints_multi("sgemm", problems)
            .expect("hints");
        assert_eq!(hints.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generator_multi_problem_all_missing() {
        let (db, dir) = make_db_with_entries();
        let generator =
            GuidedPtxGenerator::new(&db, "UNKNOWN_GPU".to_string(), GuidedPtxStrategy::BestKnown);
        let result = generator.generate_hints_multi("sgemm", &["1024x1024"]);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- HintMerger ------------------------------------------------------------

    #[test]
    fn merge_unanimous_hints() {
        let hint = PtxGenerationHint::from_config(&Config::new());
        let merged = HintMerger::merge(&[hint.clone(), hint.clone(), hint.clone()]);
        assert_eq!(merged, hint);
    }

    #[test]
    fn merge_empty_returns_default() {
        let merged = HintMerger::merge(&[]);
        let default = PtxGenerationHint::from_config(&Config::default());
        assert_eq!(merged.tile_m, default.tile_m);
    }

    #[test]
    fn conflict_score_unanimous_is_zero() {
        let hint = PtxGenerationHint::from_config(&Config::new());
        let score = HintMerger::conflict_score(&[hint.clone(), hint.clone()]);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn conflict_score_disagreement_is_positive() {
        let h1 = PtxGenerationHint::from_config(&Config::new().with_tile_m(64));
        let h2 = PtxGenerationHint::from_config(&Config::new().with_tile_m(128));
        let score = HintMerger::conflict_score(&[h1, h2]);
        assert!(score > 0.0);
    }

    // -- Specialization grouping -----------------------------------------------

    #[test]
    fn suggest_specializations_groups_by_config() {
        let (db, dir) = make_db_with_entries();
        let generator =
            GuidedPtxGenerator::new(&db, "RTX4090".to_string(), GuidedPtxStrategy::BestKnown);
        let specs = generator.suggest_specializations("sgemm").expect("specs");
        // We stored 3 entries with different configs, so we expect up to 3 groups.
        assert!(!specs.is_empty());
        // Coverage should sum to 1.0.
        let total_coverage: f64 = specs.iter().map(|s| s.coverage).sum();
        assert!((total_coverage - 1.0).abs() < 1e-9);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
