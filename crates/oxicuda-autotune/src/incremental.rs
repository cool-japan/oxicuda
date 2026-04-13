//! Incremental re-tuning for hardware/driver change detection.
//!
//! This module detects when hardware or driver changes invalidate stored
//! autotuning results and selectively re-benchmarks affected configurations.
//! Rather than re-tuning everything from scratch, it compares a saved
//! [`HardwareFingerprint`] against the current system and produces a
//! [`RetunePlan`] that targets only the configurations likely to be affected.
//!
//! # Workflow
//!
//! 1. On startup, call [`IncrementalTuner::check_and_plan()`] to detect changes.
//! 2. If a [`RetunePlan`] is returned, inspect its priority and scope.
//! 3. Call [`IncrementalTuner::execute_plan()`] with a benchmark closure to
//!    re-measure affected configurations.
//! 4. Call [`IncrementalTuner::save_fingerprint()`] to persist the current
//!    hardware state for future comparisons.
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::benchmark::BenchmarkResult;
use crate::config::Config;
use crate::error::AutotuneError;
use crate::result_db::ResultDb;

// ---------------------------------------------------------------------------
// HardwareFingerprint
// ---------------------------------------------------------------------------

/// A snapshot of the GPU hardware and driver state.
///
/// Used to detect when the system configuration has changed in a way
/// that may invalidate previously stored autotuning results.  Each
/// field captures one dimension of the hardware identity; the
/// [`fingerprint_hash`](Self::fingerprint_hash) provides a quick
/// equality check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareFingerprint {
    /// GPU model name (e.g. `"NVIDIA RTX 4090"`).
    pub gpu_name: String,
    /// Driver version string (e.g. `"535.104.05"`).
    pub driver_version: String,
    /// Compute capability (e.g. `"8.9"`).
    pub compute_capability: String,
    /// Total GPU memory in megabytes.
    pub total_memory_mb: u64,
    /// GPU clock speed in MHz.
    pub clock_mhz: u32,
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// Precomputed hash of all fields for quick comparison.
    pub fingerprint_hash: u64,
}

impl HardwareFingerprint {
    /// Queries the current GPU hardware and builds a fingerprint.
    ///
    /// On macOS (or when no GPU is available), returns a synthetic
    /// fingerprint suitable for testing and development.
    pub fn current() -> Self {
        Self::try_from_device().unwrap_or_else(|_| Self::synthetic())
    }

    /// Attempts to build a fingerprint by querying the GPU driver.
    fn try_from_device() -> Result<Self, AutotuneError> {
        #[cfg(not(target_os = "macos"))]
        {
            // Attempt to initialize the driver and query device 0.
            oxicuda_driver::init()?;
            let device = oxicuda_driver::device::Device::get(0)?;

            let gpu_name = device.name()?;
            let (major, minor) = device.compute_capability()?;
            let compute_capability = format!("{major}.{minor}");
            let total_memory_mb = (device.total_memory()? / (1024 * 1024)) as u64;
            let clock_mhz = (device.clock_rate_khz()? / 1000) as u32;
            let sm_count = device.multiprocessor_count()? as u32;

            let driver_ver = oxicuda_driver::device::driver_version()?;
            let driver_major = driver_ver / 1000;
            let driver_minor = (driver_ver % 1000) / 10;
            let driver_version = format!("{driver_major}.{driver_minor}");

            let mut fp = Self {
                gpu_name,
                driver_version,
                compute_capability,
                total_memory_mb,
                clock_mhz,
                sm_count,
                fingerprint_hash: 0,
            };
            fp.fingerprint_hash = fp.compute_hash();
            Ok(fp)
        }

        #[cfg(target_os = "macos")]
        {
            Err(AutotuneError::BenchmarkFailed(
                "GPU not available on macOS".to_string(),
            ))
        }
    }

    /// Returns a synthetic fingerprint for testing or non-GPU environments.
    #[must_use]
    pub fn synthetic() -> Self {
        let mut fp = Self {
            gpu_name: "Synthetic GPU (no driver)".to_string(),
            driver_version: "0.0".to_string(),
            compute_capability: "0.0".to_string(),
            total_memory_mb: 0,
            clock_mhz: 0,
            sm_count: 0,
            fingerprint_hash: 0,
        };
        fp.fingerprint_hash = fp.compute_hash();
        fp
    }

    /// Computes a 64-bit hash from all fingerprint fields.
    ///
    /// This hash is used for quick equality checks without comparing
    /// every field individually.
    #[must_use]
    pub fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.gpu_name.hash(&mut hasher);
        self.driver_version.hash(&mut hasher);
        self.compute_capability.hash(&mut hasher);
        self.total_memory_mb.hash(&mut hasher);
        self.clock_mhz.hash(&mut hasher);
        self.sm_count.hash(&mut hasher);
        hasher.finish()
    }

    /// Checks whether two fingerprints represent compatible hardware.
    ///
    /// Two fingerprints are compatible if they refer to the same GPU model.
    /// Driver version and clock speed differences do not break compatibility,
    /// but may warrant re-tuning at a lower priority.
    #[must_use]
    pub fn is_compatible(&self, other: &HardwareFingerprint) -> bool {
        self.gpu_name == other.gpu_name
    }
}

impl Hash for HardwareFingerprint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.gpu_name.hash(state);
        self.driver_version.hash(state);
        self.compute_capability.hash(state);
        self.total_memory_mb.hash(state);
        self.clock_mhz.hash(state);
        self.sm_count.hash(state);
    }
}

// ---------------------------------------------------------------------------
// HardwareChange
// ---------------------------------------------------------------------------

/// Describes a specific change detected between two hardware fingerprints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HardwareChange {
    /// The GPU model has changed entirely.
    GpuChanged {
        /// Previous GPU name.
        old: String,
        /// Current GPU name.
        new: String,
    },
    /// The driver version has changed.
    DriverVersionChanged {
        /// Previous driver version.
        old: String,
        /// Current driver version.
        new: String,
    },
    /// The GPU clock speed has changed.
    ClockSpeedChanged {
        /// Previous clock speed in MHz.
        old: u32,
        /// Current clock speed in MHz.
        new: u32,
    },
    /// The available GPU memory has changed.
    MemoryChanged {
        /// Previous memory in MB.
        old: u64,
        /// Current memory in MB.
        new: u64,
    },
    /// No change detected.
    NoChange,
}

// ---------------------------------------------------------------------------
// RetunePriority
// ---------------------------------------------------------------------------

/// Priority level for a re-tuning operation.
///
/// Higher-priority levels indicate that stored results are more likely
/// to be invalid and re-tuning is more urgent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RetunePriority {
    /// Minor changes unlikely to affect kernel performance.
    Low,
    /// Clock speed or minor memory changes.
    Medium,
    /// Driver version changed — may affect register allocation or scheduling.
    High,
    /// GPU model changed — all stored results are likely invalid.
    Critical,
}

impl RetunePriority {
    /// Determines the priority level from a single hardware change.
    #[must_use]
    pub fn from_change(change: &HardwareChange) -> Self {
        match change {
            HardwareChange::GpuChanged { .. } => Self::Critical,
            HardwareChange::DriverVersionChanged { .. } => Self::High,
            HardwareChange::ClockSpeedChanged { .. } => Self::Medium,
            HardwareChange::MemoryChanged { .. } => Self::Low,
            HardwareChange::NoChange => Self::Low,
        }
    }

    /// Returns the highest priority from a set of changes.
    #[must_use]
    pub fn from_changes(changes: &[HardwareChange]) -> Self {
        changes
            .iter()
            .map(Self::from_change)
            .max()
            .unwrap_or(Self::Low)
    }
}

// ---------------------------------------------------------------------------
// ChangeDetector
// ---------------------------------------------------------------------------

/// Compares previous and current hardware fingerprints to detect changes.
pub struct ChangeDetector {
    /// The previously saved fingerprint (if any).
    previous: Option<HardwareFingerprint>,
    /// The current system fingerprint.
    current: HardwareFingerprint,
}

impl ChangeDetector {
    /// Creates a new change detector.
    #[must_use]
    pub fn new(previous: Option<HardwareFingerprint>, current: HardwareFingerprint) -> Self {
        Self { previous, current }
    }

    /// Detects all hardware changes between the previous and current state.
    ///
    /// If there is no previous fingerprint, returns a single `GpuChanged`
    /// entry to trigger a full re-tune.
    #[must_use]
    pub fn detect_changes(&self) -> Vec<HardwareChange> {
        let prev = match &self.previous {
            Some(p) => p,
            None => {
                return vec![HardwareChange::GpuChanged {
                    old: String::new(),
                    new: self.current.gpu_name.clone(),
                }];
            }
        };

        // Fast path: identical fingerprints.
        if prev.fingerprint_hash == self.current.fingerprint_hash {
            return vec![HardwareChange::NoChange];
        }

        let mut changes = Vec::new();

        if prev.gpu_name != self.current.gpu_name {
            changes.push(HardwareChange::GpuChanged {
                old: prev.gpu_name.clone(),
                new: self.current.gpu_name.clone(),
            });
        }

        if prev.driver_version != self.current.driver_version {
            changes.push(HardwareChange::DriverVersionChanged {
                old: prev.driver_version.clone(),
                new: self.current.driver_version.clone(),
            });
        }

        if prev.clock_mhz != self.current.clock_mhz {
            changes.push(HardwareChange::ClockSpeedChanged {
                old: prev.clock_mhz,
                new: self.current.clock_mhz,
            });
        }

        if prev.total_memory_mb != self.current.total_memory_mb {
            changes.push(HardwareChange::MemoryChanged {
                old: prev.total_memory_mb,
                new: self.current.total_memory_mb,
            });
        }

        if changes.is_empty() {
            changes.push(HardwareChange::NoChange);
        }

        changes
    }

    /// Returns `true` if any detected change affects kernel performance.
    #[must_use]
    pub fn needs_retune(&self) -> bool {
        let changes = self.detect_changes();
        changes
            .iter()
            .any(|c| !matches!(c, HardwareChange::NoChange))
    }

    /// Returns a reference to the current fingerprint.
    #[must_use]
    pub fn current_fingerprint(&self) -> &HardwareFingerprint {
        &self.current
    }

    /// Returns a reference to the previous fingerprint (if any).
    #[must_use]
    pub fn previous_fingerprint(&self) -> Option<&HardwareFingerprint> {
        self.previous.as_ref()
    }
}

// ---------------------------------------------------------------------------
// RetunePlan
// ---------------------------------------------------------------------------

/// A plan describing which configurations need re-benchmarking.
///
/// Created by analyzing detected hardware changes against the stored
/// result database.  The plan identifies which kernel configurations
/// are affected and estimates the time required for re-tuning.
#[derive(Debug, Clone)]
pub struct RetunePlan {
    /// The hardware changes that triggered this plan.
    pub changes: Vec<HardwareChange>,
    /// Overall priority of the re-tune operation.
    pub priority: RetunePriority,
    /// Kernel names whose stored results are affected.
    pub affected_kernels: Vec<String>,
    /// Estimated time in seconds to complete re-tuning.
    pub estimated_time_seconds: f64,
}

impl RetunePlan {
    /// Builds a retune plan from detected changes and the current database.
    ///
    /// Inspects the result database to determine which kernels have stored
    /// results that may be invalidated by the detected changes.  The
    /// estimated time is based on a heuristic of ~0.5 seconds per kernel
    /// configuration.
    #[must_use]
    pub fn from_changes(changes: &[HardwareChange], db: &ResultDb) -> Self {
        let priority = RetunePriority::from_changes(changes);

        // Collect all kernel names from all GPUs in the database.
        let mut affected_kernels = Vec::new();
        for gpu in db.list_gpus() {
            for (kernel_name, _problem, _result) in db.list_gpu(gpu) {
                let name = kernel_name.to_string();
                if !affected_kernels.contains(&name) {
                    affected_kernels.push(name);
                }
            }
        }

        // For lower-priority changes, only a subset needs re-tuning.
        // Critical/High: re-tune everything. Medium/Low: subset.
        let fraction = match priority {
            RetunePriority::Critical => 1.0,
            RetunePriority::High => 0.8,
            RetunePriority::Medium => 0.3,
            RetunePriority::Low => 0.1,
        };

        let total_entries = db.total_entries();
        let affected_count = ((total_entries as f64) * fraction)
            .ceil()
            .max(affected_kernels.len() as f64);

        // Heuristic: ~0.5 seconds per configuration to re-benchmark.
        let estimated_time_seconds = affected_count * 0.5;

        Self {
            changes: changes.to_vec(),
            priority,
            affected_kernels,
            estimated_time_seconds,
        }
    }

    /// Returns `true` if this plan has no work to do.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.affected_kernels.is_empty()
    }
}

// ---------------------------------------------------------------------------
// RetuneReport
// ---------------------------------------------------------------------------

/// Summary of a completed re-tuning operation.
#[derive(Debug, Clone)]
pub struct RetuneReport {
    /// Number of configurations that were re-benchmarked.
    pub configs_tested: usize,
    /// Number of configurations where the new result was faster.
    pub configs_improved: usize,
    /// Number of configurations where performance was unchanged or slower.
    pub configs_unchanged: usize,
    /// Total wall-clock time for the re-tune operation in milliseconds.
    pub total_time_ms: u64,
    /// Detailed improvements: (kernel_name, old_median_us, new_median_us).
    pub improvements: Vec<(String, f64, f64)>,
}

impl RetuneReport {
    /// Creates an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            configs_tested: 0,
            configs_improved: 0,
            configs_unchanged: 0,
            total_time_ms: 0,
            improvements: Vec::new(),
        }
    }

    /// Returns the improvement ratio (fraction of tested configs that improved).
    #[must_use]
    pub fn improvement_ratio(&self) -> f64 {
        if self.configs_tested == 0 {
            return 0.0;
        }
        self.configs_improved as f64 / self.configs_tested as f64
    }
}

impl Default for RetuneReport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// IncrementalTuner
// ---------------------------------------------------------------------------

/// Orchestrates incremental re-tuning when hardware changes are detected.
///
/// The tuner loads a previously saved [`HardwareFingerprint`], compares
/// it with the current system, and produces a [`RetunePlan`] if any
/// changes warrant re-benchmarking.  The plan can then be executed to
/// selectively re-test affected configurations.
pub struct IncrementalTuner {
    /// Path to the saved fingerprint file.
    fingerprint_path: PathBuf,
    /// The result database to check and update.
    db: ResultDb,
}

impl IncrementalTuner {
    /// Creates a new incremental tuner.
    ///
    /// # Arguments
    ///
    /// * `fingerprint_path` — Path where the hardware fingerprint is
    ///   persisted between runs.
    /// * `db` — The autotuning result database.
    #[must_use]
    pub fn new(fingerprint_path: PathBuf, db: ResultDb) -> Self {
        Self {
            fingerprint_path,
            db,
        }
    }

    /// Checks the current hardware against the saved fingerprint and
    /// returns a retune plan if changes are detected.
    ///
    /// Returns `Ok(None)` if the hardware is unchanged and no re-tuning
    /// is needed.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::IoError`] if the fingerprint file cannot
    /// be read, or [`AutotuneError::SerdeError`] if it cannot be parsed.
    pub fn check_and_plan(&self) -> Result<Option<RetunePlan>, AutotuneError> {
        let previous = self.load_fingerprint()?;
        let current = HardwareFingerprint::current();
        let detector = ChangeDetector::new(previous, current);

        if !detector.needs_retune() {
            return Ok(None);
        }

        let changes = detector.detect_changes();
        let plan = RetunePlan::from_changes(&changes, &self.db);

        if plan.is_empty() {
            return Ok(None);
        }

        Ok(Some(plan))
    }

    /// Executes a retune plan by re-benchmarking affected configurations.
    ///
    /// The `benchmark_fn` closure receives a `Config` and a kernel name,
    /// and must return a `BenchmarkResult` for that configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any benchmark or database operation fails.
    pub fn execute_plan<F>(
        &mut self,
        plan: &RetunePlan,
        benchmark_fn: F,
    ) -> Result<RetuneReport, AutotuneError>
    where
        F: Fn(&Config, &str) -> Result<BenchmarkResult, AutotuneError>,
    {
        let start = std::time::Instant::now();
        let mut report = RetuneReport::new();

        // Collect entries to re-benchmark.
        let mut work_items: Vec<(String, String, String, Config, f64)> = Vec::new();
        for gpu in self.db.list_gpus() {
            for (kernel, problem, result) in self.db.list_gpu(gpu) {
                if plan.affected_kernels.contains(&kernel.to_string()) {
                    work_items.push((
                        gpu.to_string(),
                        kernel.to_string(),
                        problem.to_string(),
                        result.config.clone(),
                        result.median_us,
                    ));
                }
            }
        }

        for (gpu, kernel, problem, config, old_median) in &work_items {
            let new_result = benchmark_fn(config, kernel)?;
            report.configs_tested += 1;

            if new_result.median_us < *old_median {
                report.configs_improved += 1;
                report
                    .improvements
                    .push((kernel.clone(), *old_median, new_result.median_us));
            } else {
                report.configs_unchanged += 1;
            }

            // Update the database with the new result regardless.
            // ResultDb::save only replaces if the new result is faster,
            // so we force-save by checking ourselves.
            self.db.save(gpu, kernel, problem, new_result)?;
        }

        report.total_time_ms = start.elapsed().as_millis() as u64;
        Ok(report)
    }

    /// Persists the current hardware fingerprint to disk.
    ///
    /// Call this after a successful re-tune to record the current
    /// hardware state for future comparisons.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::IoError`] if the file cannot be written.
    pub fn save_fingerprint(&self) -> Result<(), AutotuneError> {
        let fp = HardwareFingerprint::current();
        let json = serde_json::to_string_pretty(&fp)?;

        if let Some(parent) = self.fingerprint_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(&self.fingerprint_path, json)?;
        Ok(())
    }

    /// Returns a reference to the result database.
    #[must_use]
    pub fn db(&self) -> &ResultDb {
        &self.db
    }

    /// Returns a mutable reference to the result database.
    pub fn db_mut(&mut self) -> &mut ResultDb {
        &mut self.db
    }

    /// Loads the previously saved fingerprint from disk.
    ///
    /// Returns `Ok(None)` if the file does not exist.
    fn load_fingerprint(&self) -> Result<Option<HardwareFingerprint>, AutotuneError> {
        if !self.fingerprint_path.exists() {
            return Ok(None);
        }

        let contents = std::fs::read_to_string(&self.fingerprint_path)?;
        if contents.trim().is_empty() {
            return Ok(None);
        }

        let fp: HardwareFingerprint = serde_json::from_str(&contents)?;
        Ok(Some(fp))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkResult;
    use crate::config::Config;

    /// Helper: create a test fingerprint with specific values.
    fn test_fingerprint(
        gpu: &str,
        driver: &str,
        cc: &str,
        mem_mb: u64,
        clock: u32,
        sms: u32,
    ) -> HardwareFingerprint {
        let mut fp = HardwareFingerprint {
            gpu_name: gpu.to_string(),
            driver_version: driver.to_string(),
            compute_capability: cc.to_string(),
            total_memory_mb: mem_mb,
            clock_mhz: clock,
            sm_count: sms,
            fingerprint_hash: 0,
        };
        fp.fingerprint_hash = fp.compute_hash();
        fp
    }

    /// Helper: create a mock BenchmarkResult.
    fn mock_result(median_us: f64) -> BenchmarkResult {
        BenchmarkResult {
            config: Config::new(),
            median_us,
            min_us: median_us - 1.0,
            max_us: median_us + 1.0,
            stddev_us: 0.5,
            gflops: None,
            efficiency: None,
        }
    }

    /// Helper: create a temporary ResultDb.
    fn temp_db(name: &str) -> ResultDb {
        let dir = std::env::temp_dir().join(format!("oxicuda_incr_test_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("results.json");
        ResultDb::open_at(path).expect("open temp db")
    }

    #[test]
    fn fingerprint_creation_synthetic() {
        let fp = HardwareFingerprint::synthetic();
        assert_eq!(fp.gpu_name, "Synthetic GPU (no driver)");
        assert_eq!(fp.driver_version, "0.0");
        assert_eq!(fp.compute_capability, "0.0");
        assert_eq!(fp.total_memory_mb, 0);
        assert_eq!(fp.clock_mhz, 0);
        assert_eq!(fp.sm_count, 0);
        assert_ne!(fp.fingerprint_hash, 0);
    }

    #[test]
    fn fingerprint_hash_consistency() {
        let fp1 = test_fingerprint("RTX 4090", "535.104", "8.9", 24576, 2520, 128);
        let fp2 = test_fingerprint("RTX 4090", "535.104", "8.9", 24576, 2520, 128);
        assert_eq!(fp1.fingerprint_hash, fp2.fingerprint_hash);
        assert_eq!(fp1.compute_hash(), fp2.compute_hash());

        // Different GPU should yield different hash.
        let fp3 = test_fingerprint("RTX 3090", "535.104", "8.6", 24576, 1695, 82);
        assert_ne!(fp1.fingerprint_hash, fp3.fingerprint_hash);
    }

    #[test]
    fn change_detection_no_change() {
        let fp = test_fingerprint("RTX 4090", "535.104", "8.9", 24576, 2520, 128);
        let detector = ChangeDetector::new(Some(fp.clone()), fp);
        let changes = detector.detect_changes();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0], HardwareChange::NoChange);
        assert!(!detector.needs_retune());
    }

    #[test]
    fn change_detection_gpu_changed() {
        let old = test_fingerprint("RTX 3090", "535.104", "8.6", 24576, 1695, 82);
        let new = test_fingerprint("RTX 4090", "535.104", "8.9", 24576, 2520, 128);
        let detector = ChangeDetector::new(Some(old), new);
        let changes = detector.detect_changes();

        let has_gpu_change = changes
            .iter()
            .any(|c| matches!(c, HardwareChange::GpuChanged { .. }));
        assert!(has_gpu_change);
        assert!(detector.needs_retune());
    }

    #[test]
    fn change_detection_driver_changed() {
        let old = test_fingerprint("RTX 4090", "525.60", "8.9", 24576, 2520, 128);
        let new = test_fingerprint("RTX 4090", "535.104", "8.9", 24576, 2520, 128);
        let detector = ChangeDetector::new(Some(old), new);
        let changes = detector.detect_changes();

        let has_driver_change = changes
            .iter()
            .any(|c| matches!(c, HardwareChange::DriverVersionChanged { .. }));
        assert!(has_driver_change);
        assert!(detector.needs_retune());
    }

    #[test]
    fn retune_priority_levels() {
        assert_eq!(
            RetunePriority::from_change(&HardwareChange::GpuChanged {
                old: "A".into(),
                new: "B".into(),
            }),
            RetunePriority::Critical
        );
        assert_eq!(
            RetunePriority::from_change(&HardwareChange::DriverVersionChanged {
                old: "1.0".into(),
                new: "2.0".into(),
            }),
            RetunePriority::High
        );
        assert_eq!(
            RetunePriority::from_change(&HardwareChange::ClockSpeedChanged {
                old: 1000,
                new: 2000,
            }),
            RetunePriority::Medium
        );
        assert_eq!(
            RetunePriority::from_change(&HardwareChange::MemoryChanged {
                old: 8192,
                new: 16384,
            }),
            RetunePriority::Low
        );
        assert_eq!(
            RetunePriority::from_change(&HardwareChange::NoChange),
            RetunePriority::Low
        );

        // from_changes picks the highest priority.
        let changes = vec![
            HardwareChange::MemoryChanged {
                old: 8192,
                new: 16384,
            },
            HardwareChange::DriverVersionChanged {
                old: "1.0".into(),
                new: "2.0".into(),
            },
        ];
        assert_eq!(RetunePriority::from_changes(&changes), RetunePriority::High);
    }

    #[test]
    fn retune_plan_creation() {
        let mut db = temp_db("plan_creation");
        db.save("RTX 4090", "sgemm", "1024x1024", mock_result(42.0))
            .expect("save");
        db.save("RTX 4090", "dgemm", "512x512", mock_result(80.0))
            .expect("save");

        let changes = vec![HardwareChange::DriverVersionChanged {
            old: "525.0".into(),
            new: "535.0".into(),
        }];

        let plan = RetunePlan::from_changes(&changes, &db);
        assert_eq!(plan.priority, RetunePriority::High);
        assert!(plan.affected_kernels.contains(&"sgemm".to_string()));
        assert!(plan.affected_kernels.contains(&"dgemm".to_string()));
        assert!(plan.estimated_time_seconds > 0.0);

        // Cleanup
        let dir = std::env::temp_dir().join("oxicuda_incr_test_plan_creation");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compatible_fingerprints() {
        let fp1 = test_fingerprint("RTX 4090", "535.104", "8.9", 24576, 2520, 128);
        let fp2 = test_fingerprint("RTX 4090", "525.60", "8.9", 24576, 2100, 128);
        assert!(fp1.is_compatible(&fp2));
    }

    #[test]
    fn incompatible_fingerprints() {
        let fp1 = test_fingerprint("RTX 4090", "535.104", "8.9", 24576, 2520, 128);
        let fp2 = test_fingerprint("RTX 3090", "535.104", "8.6", 24576, 1695, 82);
        assert!(!fp1.is_compatible(&fp2));
    }

    #[test]
    fn fingerprint_serialization_roundtrip() {
        let fp = test_fingerprint("A100-SXM4-80GB", "535.104", "8.0", 81920, 1410, 108);
        let json = serde_json::to_string_pretty(&fp).expect("serialize");
        let restored: HardwareFingerprint = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(fp, restored);
        assert_eq!(fp.fingerprint_hash, restored.fingerprint_hash);
    }

    #[test]
    fn empty_database_plan() {
        let db = temp_db("empty_db_plan");
        let changes = vec![HardwareChange::GpuChanged {
            old: "RTX 3090".into(),
            new: "RTX 4090".into(),
        }];

        let plan = RetunePlan::from_changes(&changes, &db);
        assert_eq!(plan.priority, RetunePriority::Critical);
        assert!(plan.affected_kernels.is_empty());
        assert!(plan.is_empty());

        // Cleanup
        let dir = std::env::temp_dir().join("oxicuda_incr_test_empty_db_plan");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn retune_report_statistics() {
        let mut report = RetuneReport::new();
        assert_eq!(report.configs_tested, 0);
        assert_eq!(report.configs_improved, 0);
        assert_eq!(report.configs_unchanged, 0);
        assert!((report.improvement_ratio() - 0.0).abs() < f64::EPSILON);

        report.configs_tested = 10;
        report.configs_improved = 3;
        report.configs_unchanged = 7;
        report.total_time_ms = 5000;
        report.improvements = vec![
            ("sgemm".to_string(), 42.0, 38.0),
            ("dgemm".to_string(), 80.0, 72.0),
            ("hgemm".to_string(), 25.0, 22.0),
        ];

        assert!((report.improvement_ratio() - 0.3).abs() < f64::EPSILON);
        assert_eq!(report.improvements.len(), 3);
    }

    #[test]
    fn incremental_tuner_check_and_plan_no_previous() {
        let dir = std::env::temp_dir().join("oxicuda_incr_test_no_prev");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");

        let fp_path = dir.join("fingerprint.json");
        let db_path = dir.join("results.json");
        let mut db = ResultDb::open_at(db_path).expect("open db");
        db.save("GPU0", "sgemm", "1024", mock_result(42.0))
            .expect("save");

        let tuner = IncrementalTuner::new(fp_path, db);
        // No previous fingerprint means it should recommend re-tuning.
        let plan = tuner.check_and_plan().expect("check_and_plan");
        assert!(plan.is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn incremental_tuner_save_and_load_fingerprint() {
        let dir = std::env::temp_dir().join("oxicuda_incr_test_save_load");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");

        let fp_path = dir.join("fingerprint.json");
        let db_path = dir.join("results.json");
        let db = ResultDb::open_at(db_path).expect("open db");

        let tuner = IncrementalTuner::new(fp_path.clone(), db);
        tuner.save_fingerprint().expect("save fingerprint");

        // Verify the file exists and can be loaded.
        assert!(fp_path.exists());
        let contents = std::fs::read_to_string(&fp_path).expect("read fp");
        let fp: HardwareFingerprint = serde_json::from_str(&contents).expect("parse fp");
        assert!(!fp.gpu_name.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
