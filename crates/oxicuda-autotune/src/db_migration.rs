//! Versioned schema migration for the autotune result database.
//!
//! As the [`ResultDb`](crate::ResultDb) schema evolves, this module provides
//! forward-compatible migration from older on-disk formats to the current
//! schema version.  Each migration step is represented by a [`MigrationStep`]
//! trait implementation, and the [`MigrationRunner`] chains them together
//! to bring any historical database file up to date.
//!
//! # Schema versions
//!
//! | Version | Description                                    |
//! |---------|------------------------------------------------|
//! | v1.0.0  | Original bare JSON (no metadata envelope)      |
//! | v2.0.0  | Wrapped in [`VersionedDb`] with [`DbMetadata`] |
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_autotune::db_migration::{MigrationRunner, SchemaVersion};
//!
//! # fn example() -> Result<(), oxicuda_autotune::AutotuneError> {
//! let runner = MigrationRunner::new();
//! let raw = std::fs::read_to_string("results.json")?;
//! let mut data: serde_json::Value = serde_json::from_str(&raw)?;
//!
//! if MigrationRunner::needs_migration(&data) {
//!     let report = runner.migrate_to_latest(&mut data)?;
//!     println!("Migrated {} -> {}", report.source_version, report.target_version);
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::AutotuneError;

// ---------------------------------------------------------------------------
// SchemaVersion
// ---------------------------------------------------------------------------

/// Semantic version of the result-database schema.
///
/// Compatibility is determined by major version: two versions are
/// compatible if and only if they share the same major number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Major version -- incompatible schema changes.
    pub major: u32,
    /// Minor version -- backwards-compatible additions.
    pub minor: u32,
    /// Patch version -- bug fixes in migration logic.
    pub patch: u32,
}

impl SchemaVersion {
    /// The original schema version (bare JSON, no metadata envelope).
    pub const V1_0_0: Self = Self {
        major: 1,
        minor: 0,
        patch: 0,
    };

    /// Current schema version (with [`VersionedDb`] envelope).
    pub const CURRENT: Self = Self {
        major: 2,
        minor: 0,
        patch: 0,
    };

    /// Returns `true` if `self` is compatible with `other`.
    ///
    /// Two versions are compatible when they share the same major number.
    #[must_use]
    pub fn is_compatible_with(&self, other: &SchemaVersion) -> bool {
        self.major == other.major
    }
}

impl fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ---------------------------------------------------------------------------
// DbMetadata
// ---------------------------------------------------------------------------

/// Metadata stored alongside the autotune result data.
///
/// Captures the schema version, timestamps, and entry count so that
/// tooling can quickly inspect a database file without parsing the
/// full data payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbMetadata {
    /// Schema version of this database file.
    pub schema_version: SchemaVersion,
    /// ISO 8601 timestamp of initial creation.
    pub created_at: String,
    /// ISO 8601 timestamp of the last modification.
    pub last_modified: String,
    /// Number of (GPU, kernel, problem) entries in the database.
    pub entry_count: usize,
    /// Tool version string (e.g. `"oxicuda-autotune 0.1.0"`).
    pub tool_version: String,
}

// ---------------------------------------------------------------------------
// VersionedDb
// ---------------------------------------------------------------------------

/// A result database wrapped with a metadata envelope.
///
/// This is the v2 on-disk format: the original flat
/// `HashMap<GPU, HashMap<kernel, HashMap<problem, result>>>` is stored
/// under the `data` key, with a `metadata` sibling for schema tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedDb {
    /// Schema and bookkeeping metadata.
    pub metadata: DbMetadata,
    /// The tuning results, keyed by GPU -> kernel -> problem.
    pub data: HashMap<String, HashMap<String, HashMap<String, serde_json::Value>>>,
}

// ---------------------------------------------------------------------------
// MigrationStep trait
// ---------------------------------------------------------------------------

/// A single schema migration step.
///
/// Implementations transform a [`serde_json::Value`] from
/// [`source_version`](MigrationStep::source_version) to
/// [`target_version`](MigrationStep::target_version).
pub trait MigrationStep: Send + Sync {
    /// The version this step migrates **from**.
    fn source_version(&self) -> SchemaVersion;

    /// The version this step migrates **to**.
    fn target_version(&self) -> SchemaVersion;

    /// Apply the migration in place.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError`] if the data cannot be transformed.
    fn migrate(&self, data: &mut serde_json::Value) -> Result<(), AutotuneError>;

    /// Human-readable description of this migration step.
    fn description(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Built-in migration: v1 -> v2
// ---------------------------------------------------------------------------

/// Migrates from bare JSON (v1) to the [`VersionedDb`] envelope (v2).
///
/// Detection: a v1 database is a JSON object that does **not** contain
/// a `"metadata"` key at the top level.
struct MigrateV1ToV2;

impl MigrationStep for MigrateV1ToV2 {
    fn source_version(&self) -> SchemaVersion {
        SchemaVersion::V1_0_0
    }

    fn target_version(&self) -> SchemaVersion {
        SchemaVersion::CURRENT
    }

    fn migrate(&self, data: &mut serde_json::Value) -> Result<(), AutotuneError> {
        // The v1 format is `{ "entries": { GPU -> kernel -> problem -> result } }`
        // or possibly the bare triple-nested map without the "entries" wrapper.
        let entries_value = if let Some(entries) = data.get("entries") {
            entries.clone()
        } else if data.is_object() {
            data.clone()
        } else {
            return Err(AutotuneError::BenchmarkFailed(
                "v1 database is not a JSON object".to_string(),
            ));
        };

        // Count entries for metadata.
        let entry_count = count_entries(&entries_value);

        let now = iso8601_now();
        let metadata = DbMetadata {
            schema_version: SchemaVersion::CURRENT,
            created_at: now.clone(),
            last_modified: now,
            entry_count,
            tool_version: format!("oxicuda-autotune {}", env!("CARGO_PKG_VERSION")),
        };

        let metadata_value = serde_json::to_value(&metadata).map_err(AutotuneError::from)?;

        // Rebuild the top-level object as a VersionedDb.
        let mut new_obj = serde_json::Map::new();
        new_obj.insert("metadata".to_string(), metadata_value);
        new_obj.insert("data".to_string(), entries_value);
        *data = serde_json::Value::Object(new_obj);

        Ok(())
    }

    fn description(&self) -> &str {
        "Wrap bare JSON in VersionedDb envelope with metadata (v1 -> v2)"
    }
}

// ---------------------------------------------------------------------------
// MigrationReport
// ---------------------------------------------------------------------------

/// Summary of a completed migration run.
#[derive(Debug, Clone)]
pub struct MigrationReport {
    /// Human-readable descriptions of applied steps.
    pub steps_applied: Vec<String>,
    /// Version before migration.
    pub source_version: SchemaVersion,
    /// Version after migration.
    pub target_version: SchemaVersion,
    /// Path to the backup file, if one was created.
    pub backup_path: Option<PathBuf>,
    /// Total number of (GPU, kernel, problem) entries that were migrated.
    pub entries_migrated: usize,
}

// ---------------------------------------------------------------------------
// MigrationRunner
// ---------------------------------------------------------------------------

/// Orchestrates schema migrations from any historical version to the
/// current schema.
///
/// Built-in migration steps are registered at construction time.
/// Additional steps can be registered via [`register`](Self::register).
pub struct MigrationRunner {
    steps: Vec<Box<dyn MigrationStep>>,
}

impl MigrationRunner {
    /// Creates a new runner with all built-in migration steps registered.
    #[must_use]
    pub fn new() -> Self {
        let mut runner = Self { steps: Vec::new() };
        runner.steps.push(Box::new(MigrateV1ToV2));
        runner
    }

    /// Registers an additional migration step.
    pub fn register(&mut self, step: Box<dyn MigrationStep>) {
        self.steps.push(step);
    }

    /// Detects the schema version of the given JSON value.
    ///
    /// Returns [`SchemaVersion::CURRENT`] if the value has a valid
    /// `metadata.schema_version`, or [`SchemaVersion::V1_0_0`] for
    /// legacy data without a metadata envelope.
    #[must_use]
    pub fn detect_version(data: &serde_json::Value) -> SchemaVersion {
        if let Some(metadata) = data.get("metadata") {
            if let Some(sv) = metadata.get("schema_version") {
                if let Ok(version) = serde_json::from_value::<SchemaVersion>(sv.clone()) {
                    return version;
                }
            }
        }
        SchemaVersion::V1_0_0
    }

    /// Returns `true` if the data requires migration to reach the current
    /// schema version.
    #[must_use]
    pub fn needs_migration(data: &serde_json::Value) -> bool {
        Self::detect_version(data) != SchemaVersion::CURRENT
    }

    /// Migrates the data to the latest schema version.
    ///
    /// Steps are applied in order, matching each step whose
    /// `source_version` equals the current detected version, until
    /// the data reaches [`SchemaVersion::CURRENT`].
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError`] if any migration step fails.
    pub fn migrate_to_latest(
        &self,
        data: &mut serde_json::Value,
    ) -> Result<MigrationReport, AutotuneError> {
        let source_version = Self::detect_version(data);
        let mut current = source_version;
        let mut steps_applied = Vec::new();
        let mut entries_migrated = 0;

        // Apply steps in order until we reach CURRENT.
        let max_iterations = self.steps.len() + 1;
        for _ in 0..max_iterations {
            if current == SchemaVersion::CURRENT {
                break;
            }

            let step = self.steps.iter().find(|s| s.source_version() == current);

            match step {
                Some(s) => {
                    s.migrate(data)?;
                    steps_applied.push(s.description().to_string());
                    entries_migrated = count_entries_versioned(data);
                    current = s.target_version();
                }
                None => {
                    return Err(AutotuneError::BenchmarkFailed(format!(
                        "no migration step found for version {current}"
                    )));
                }
            }
        }

        Ok(MigrationReport {
            steps_applied,
            source_version,
            target_version: current,
            backup_path: None,
            entries_migrated,
        })
    }

    /// Creates a timestamped backup of the database file.
    ///
    /// The backup is written to `{path}.backup.{timestamp}` where
    /// timestamp is seconds since the Unix epoch.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::IoError`] if the file cannot be copied.
    pub fn create_backup(path: &Path) -> Result<PathBuf, AutotuneError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let backup_name = format!(
            "{}.backup.{timestamp}",
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("results.json")
        );

        let backup_path = path
            .parent()
            .map(|p| p.join(&backup_name))
            .unwrap_or_else(|| PathBuf::from(&backup_name));

        std::fs::copy(path, &backup_path)?;
        Ok(backup_path)
    }
}

impl Default for MigrationRunner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Counts entries in a bare triple-nested map (v1 format).
fn count_entries(value: &serde_json::Value) -> usize {
    let mut count = 0;
    if let Some(obj) = value.as_object() {
        for (_gpu, kernels) in obj {
            if let Some(k_obj) = kernels.as_object() {
                for (_kernel, problems) in k_obj {
                    if let Some(p_obj) = problems.as_object() {
                        count += p_obj.len();
                    }
                }
            }
        }
    }
    count
}

/// Counts entries in a versioned database (v2 format).
fn count_entries_versioned(value: &serde_json::Value) -> usize {
    if let Some(data) = value.get("data") {
        count_entries(data)
    } else {
        count_entries(value)
    }
}

/// Returns the current time as an ISO 8601 string (UTC, second precision).
fn iso8601_now() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Simple UTC formatting without external deps.
    // seconds -> (year, month, day, hour, minute, second)
    let (y, mo, d, h, mi, s) = seconds_to_datetime(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Converts Unix timestamp to (year, month, day, hour, minute, second) in UTC.
fn seconds_to_datetime(secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let s = secs % 60;
    let total_min = secs / 60;
    let mi = total_min % 60;
    let total_hours = total_min / 60;
    let h = total_hours % 24;
    let mut days = total_hours / 24;

    // Compute year
    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    // Compute month/day
    let month_days: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1u64;
    for &md in &month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    let day = days + 1;

    (year, month, day, h, mi, s)
}

/// Returns `true` if the given year is a leap year.
const fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- SchemaVersion tests ------------------------------------------------

    #[test]
    fn schema_version_display() {
        assert_eq!(SchemaVersion::V1_0_0.to_string(), "v1.0.0");
        assert_eq!(SchemaVersion::CURRENT.to_string(), "v2.0.0");
    }

    #[test]
    fn schema_version_compatibility_same_major() {
        let a = SchemaVersion {
            major: 2,
            minor: 0,
            patch: 0,
        };
        let b = SchemaVersion {
            major: 2,
            minor: 1,
            patch: 3,
        };
        assert!(a.is_compatible_with(&b));
        assert!(b.is_compatible_with(&a));
    }

    #[test]
    fn schema_version_incompatible_different_major() {
        assert!(!SchemaVersion::V1_0_0.is_compatible_with(&SchemaVersion::CURRENT));
    }

    #[test]
    fn schema_version_serialize_roundtrip() {
        let v = SchemaVersion::CURRENT;
        let json = serde_json::to_string(&v).expect("serialize");
        let v2: SchemaVersion = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(v, v2);
    }

    // -- Version detection --------------------------------------------------

    #[test]
    fn detect_v1_from_bare_json() {
        let v1_data = serde_json::json!({
            "entries": {
                "RTX 4090": {
                    "sgemm": {
                        "1024x1024": {
                            "config": {},
                            "median_us": 42.0
                        }
                    }
                }
            }
        });
        assert_eq!(
            MigrationRunner::detect_version(&v1_data),
            SchemaVersion::V1_0_0
        );
    }

    #[test]
    fn detect_v2_from_versioned_json() {
        let v2_data = serde_json::json!({
            "metadata": {
                "schema_version": { "major": 2, "minor": 0, "patch": 0 },
                "created_at": "2026-01-01T00:00:00Z",
                "last_modified": "2026-01-01T00:00:00Z",
                "entry_count": 1,
                "tool_version": "oxicuda-autotune 0.1.0"
            },
            "data": {}
        });
        assert_eq!(
            MigrationRunner::detect_version(&v2_data),
            SchemaVersion::CURRENT
        );
    }

    #[test]
    fn needs_migration_for_v1() {
        let v1 = serde_json::json!({ "entries": {} });
        assert!(MigrationRunner::needs_migration(&v1));
    }

    #[test]
    fn no_migration_needed_for_current() {
        let v2 = serde_json::json!({
            "metadata": {
                "schema_version": { "major": 2, "minor": 0, "patch": 0 },
                "created_at": "2026-01-01T00:00:00Z",
                "last_modified": "2026-01-01T00:00:00Z",
                "entry_count": 0,
                "tool_version": "test"
            },
            "data": {}
        });
        assert!(!MigrationRunner::needs_migration(&v2));
    }

    // -- v1 -> v2 migration -------------------------------------------------

    #[test]
    fn migrate_v1_to_v2_with_entries_wrapper() {
        let mut data = serde_json::json!({
            "entries": {
                "GPU0": {
                    "sgemm": {
                        "1024x1024": { "median_us": 42.0 }
                    }
                }
            }
        });

        let runner = MigrationRunner::new();
        let report = runner.migrate_to_latest(&mut data).expect("migrate");

        assert_eq!(report.source_version, SchemaVersion::V1_0_0);
        assert_eq!(report.target_version, SchemaVersion::CURRENT);
        assert_eq!(report.steps_applied.len(), 1);
        assert_eq!(report.entries_migrated, 1);

        // Should now have metadata at top level.
        assert!(data.get("metadata").is_some());
        assert!(data.get("data").is_some());

        // Detected version should be CURRENT.
        assert_eq!(
            MigrationRunner::detect_version(&data),
            SchemaVersion::CURRENT
        );
    }

    #[test]
    fn migrate_v1_bare_map() {
        // v1 without "entries" wrapper (raw triple-nested map).
        let mut data = serde_json::json!({
            "GPU0": {
                "kernel_a": {
                    "prob1": { "median_us": 10.0 },
                    "prob2": { "median_us": 20.0 }
                }
            },
            "GPU1": {
                "kernel_b": {
                    "prob3": { "median_us": 30.0 }
                }
            }
        });

        let runner = MigrationRunner::new();
        let report = runner.migrate_to_latest(&mut data).expect("migrate");

        assert_eq!(report.entries_migrated, 3);
        assert_eq!(report.target_version, SchemaVersion::CURRENT);
    }

    #[test]
    fn migrate_already_current_is_noop() {
        let mut data = serde_json::json!({
            "metadata": {
                "schema_version": { "major": 2, "minor": 0, "patch": 0 },
                "created_at": "2026-01-01T00:00:00Z",
                "last_modified": "2026-01-01T00:00:00Z",
                "entry_count": 0,
                "tool_version": "test"
            },
            "data": {}
        });

        let runner = MigrationRunner::new();
        let report = runner.migrate_to_latest(&mut data).expect("migrate");

        assert!(report.steps_applied.is_empty());
        assert_eq!(report.source_version, SchemaVersion::CURRENT);
        assert_eq!(report.target_version, SchemaVersion::CURRENT);
    }

    // -- Backup creation ----------------------------------------------------

    #[test]
    fn create_backup_copies_file() {
        let dir = std::env::temp_dir().join("oxicuda_test_db_migration_backup");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("results.json");
        std::fs::write(&path, r#"{"entries":{}}"#).expect("write");

        let backup = MigrationRunner::create_backup(&path).expect("backup");
        assert!(backup.exists());
        assert!(
            backup
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("results.json.backup."))
        );

        let original = std::fs::read_to_string(&path).expect("read original");
        let backed_up = std::fs::read_to_string(&backup).expect("read backup");
        assert_eq!(original, backed_up);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- Round-trip: migrate then deserialize VersionedDb -------------------

    #[test]
    fn roundtrip_migrate_then_parse_versioned_db() {
        let mut data = serde_json::json!({
            "entries": {
                "GPU0": {
                    "sgemm": {
                        "1024": { "median_us": 5.0 }
                    }
                }
            }
        });

        let runner = MigrationRunner::new();
        let _report = runner.migrate_to_latest(&mut data).expect("migrate");

        // Should be parseable as VersionedDb.
        let vdb: VersionedDb = serde_json::from_value(data).expect("parse VersionedDb");
        assert_eq!(vdb.metadata.schema_version, SchemaVersion::CURRENT);
        assert!(vdb.data.contains_key("GPU0"));
    }

    // -- MigrationReport contents -------------------------------------------

    #[test]
    fn migration_report_details() {
        let mut data = serde_json::json!({
            "entries": {
                "GPU0": {
                    "k1": { "p1": {} },
                    "k2": { "p2": {}, "p3": {} }
                }
            }
        });

        let runner = MigrationRunner::new();
        let report = runner.migrate_to_latest(&mut data).expect("migrate");

        assert_eq!(report.source_version, SchemaVersion::V1_0_0);
        assert_eq!(report.target_version, SchemaVersion::CURRENT);
        assert_eq!(report.steps_applied.len(), 1);
        assert!(report.steps_applied[0].contains("v1 -> v2"));
        assert_eq!(report.entries_migrated, 3);
        assert!(report.backup_path.is_none());
    }

    // -- DbMetadata serde ---------------------------------------------------

    #[test]
    fn db_metadata_serialize_deserialize() {
        let meta = DbMetadata {
            schema_version: SchemaVersion::CURRENT,
            created_at: "2026-04-11T12:00:00Z".to_string(),
            last_modified: "2026-04-11T12:00:00Z".to_string(),
            entry_count: 42,
            tool_version: "oxicuda-autotune 0.1.0".to_string(),
        };
        let json = serde_json::to_string(&meta).expect("serialize");
        let meta2: DbMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(meta2.entry_count, 42);
        assert_eq!(meta2.schema_version, SchemaVersion::CURRENT);
    }

    // -- iso8601_now helper -------------------------------------------------

    #[test]
    fn iso8601_now_format() {
        let ts = iso8601_now();
        // Should look like "2026-04-11T12:34:56Z"
        assert_eq!(ts.len(), 20);
        assert!(ts.ends_with('Z'));
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
    }

    // -- Edge: empty database -----------------------------------------------

    #[test]
    fn migrate_empty_v1_database() {
        let mut data = serde_json::json!({ "entries": {} });

        let runner = MigrationRunner::new();
        let report = runner.migrate_to_latest(&mut data).expect("migrate");

        assert_eq!(report.entries_migrated, 0);
        assert_eq!(report.target_version, SchemaVersion::CURRENT);
    }

    // -- Custom migration step registration ----------------------------------

    #[test]
    fn register_custom_migration_step() {
        struct NoopStep;
        impl MigrationStep for NoopStep {
            fn source_version(&self) -> SchemaVersion {
                SchemaVersion {
                    major: 99,
                    minor: 0,
                    patch: 0,
                }
            }
            fn target_version(&self) -> SchemaVersion {
                SchemaVersion {
                    major: 100,
                    minor: 0,
                    patch: 0,
                }
            }
            fn migrate(&self, _data: &mut serde_json::Value) -> Result<(), AutotuneError> {
                Ok(())
            }
            fn description(&self) -> &str {
                "noop v99 -> v100"
            }
        }

        let mut runner = MigrationRunner::new();
        runner.register(Box::new(NoopStep));
        // Verify it was registered (2 steps total).
        assert_eq!(runner.steps.len(), 2);
    }
}
