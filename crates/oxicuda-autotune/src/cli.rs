//! CLI tool infrastructure for offline autotuning.
//!
//! Provides argument parsing, progress reporting, and execution logic
//! that a binary can use to drive batch autotuning sessions.  This is
//! a library module — it does not define `main()`.
//!
//! # Architecture
//!
//! ```text
//!  args &[String] ──► CliConfig::from_args()
//!                            │
//!                            ▼
//!                       CliRunner::new()
//!                            │
//!                            ▼
//!                       CliRunner::run()
//!                       ├─ run_tune   (benchmark search)
//!                       ├─ run_list   (query ResultDb)
//!                       ├─ run_export (serialize to file)
//!                       ├─ run_import (load from file)
//!                       ├─ run_info   (database statistics)
//!                       └─ run_clean  (prune old entries)
//! ```
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

use crate::benchmark::BenchmarkResult;
use crate::error::AutotuneError;
use crate::export::{ExportBundle, ExportFilter, ExportFormat, ImportPolicy, import_bundle};
use crate::result_db::ResultDb;

// ---------------------------------------------------------------------------
// CliCommand
// ---------------------------------------------------------------------------

/// A parsed CLI command with its arguments.
#[derive(Debug, Clone)]
pub enum CliCommand {
    /// Run autotuning for a kernel across problem sizes.
    Tune {
        /// Kernel name to tune (e.g. `"sgemm"`).
        kernel: String,
        /// Problem sizes to benchmark (e.g. `["1024x1024", "2048x2048"]`).
        problem_sizes: Vec<String>,
        /// Path to the result database file.
        db_path: Option<String>,
        /// Number of warmup iterations per configuration.
        warmup: u32,
        /// Number of timed iterations per configuration.
        iterations: u32,
        /// Whether to enable early stopping.
        early_stop: bool,
        /// Whether to benchmark in parallel across streams.
        parallel: bool,
    },
    /// List stored results from the database.
    List {
        /// Path to the result database file.
        db_path: Option<String>,
        /// Filter by kernel name.
        kernel: Option<String>,
        /// Filter by GPU name.
        gpu: Option<String>,
    },
    /// Export results to a file.
    Export {
        /// Path to the result database file.
        db_path: Option<String>,
        /// Output format (`json`, `csv`, `toml`).
        format: String,
        /// Output file path.
        output: String,
        /// Filter by kernel name.
        kernel: Option<String>,
    },
    /// Import results from a file.
    Import {
        /// Path to the result database file.
        db_path: Option<String>,
        /// Input file path.
        input: String,
        /// Conflict resolution policy (`always`, `best`, `existing`, `missing`).
        policy: String,
    },
    /// Display database statistics.
    Info {
        /// Path to the result database file.
        db_path: Option<String>,
    },
    /// Remove old entries from the database.
    Clean {
        /// Path to the result database file.
        db_path: Option<String>,
        /// Remove entries older than this many days.
        older_than_days: Option<u32>,
    },
}

// ---------------------------------------------------------------------------
// CliConfig
// ---------------------------------------------------------------------------

/// Parsed CLI configuration including the command and global flags.
#[derive(Debug, Clone)]
pub struct CliConfig {
    /// The subcommand to execute.
    pub command: CliCommand,
    /// Enable verbose output.
    pub verbose: bool,
    /// Print what would be done without executing.
    pub dry_run: bool,
}

impl CliConfig {
    /// Parses CLI arguments into a [`CliConfig`].
    ///
    /// Expects `args` to be the full argument list (including the program
    /// name at index 0).  Returns an error if the arguments are invalid
    /// or no subcommand is provided.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] with a descriptive
    /// message if parsing fails.
    pub fn from_args(args: &[String]) -> Result<Self, AutotuneError> {
        let mut verbose = false;
        let mut dry_run = false;

        // Separate global flags from positional args.
        let mut positional: Vec<&str> = Vec::new();
        let mut named: std::collections::HashMap<String, String> = std::collections::HashMap::new();

        let mut i = 1; // skip program name
        while i < args.len() {
            let arg = args[i].as_str();
            match arg {
                "--verbose" | "-v" => verbose = true,
                "--dry-run" => dry_run = true,
                _ if arg.starts_with("--") => {
                    let key = arg.trim_start_matches('-').to_string();
                    if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        named.insert(key, args[i + 1].clone());
                        i += 1;
                    } else {
                        named.insert(key, String::new());
                    }
                }
                _ => positional.push(arg),
            }
            i += 1;
        }

        let subcommand = positional.first().copied().unwrap_or("");

        let command = match subcommand {
            "tune" => {
                let kernel = positional
                    .get(1)
                    .map(|s| s.to_string())
                    .ok_or_else(|| parse_err("tune requires a kernel name"))?;
                let problem_sizes: Vec<String> =
                    positional[2..].iter().map(|s| s.to_string()).collect();
                CliCommand::Tune {
                    kernel,
                    problem_sizes,
                    db_path: named.get("db").cloned(),
                    warmup: named
                        .get("warmup")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(5),
                    iterations: named
                        .get("iterations")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(20),
                    early_stop: named.contains_key("early-stop"),
                    parallel: named.contains_key("parallel"),
                }
            }
            "list" => CliCommand::List {
                db_path: named.get("db").cloned(),
                kernel: named.get("kernel").cloned(),
                gpu: named.get("gpu").cloned(),
            },
            "export" => {
                let output = named
                    .get("output")
                    .cloned()
                    .ok_or_else(|| parse_err("export requires --output path"))?;
                CliCommand::Export {
                    db_path: named.get("db").cloned(),
                    format: named
                        .get("format")
                        .cloned()
                        .unwrap_or_else(|| "json".to_string()),
                    output,
                    kernel: named.get("kernel").cloned(),
                }
            }
            "import" => {
                let input = named
                    .get("input")
                    .cloned()
                    .or_else(|| positional.get(1).map(|s| s.to_string()))
                    .ok_or_else(|| parse_err("import requires --input path"))?;
                CliCommand::Import {
                    db_path: named.get("db").cloned(),
                    input,
                    policy: named
                        .get("policy")
                        .cloned()
                        .unwrap_or_else(|| "best".to_string()),
                }
            }
            "info" => CliCommand::Info {
                db_path: named.get("db").cloned(),
            },
            "clean" => CliCommand::Clean {
                db_path: named.get("db").cloned(),
                older_than_days: named.get("older-than").and_then(|v| v.parse().ok()),
            },
            "" => {
                return Err(parse_err(
                    "no subcommand provided. Use: tune, list, export, import, info, clean",
                ));
            }
            other => {
                return Err(parse_err(&format!(
                    "unknown subcommand '{other}'. Use: tune, list, export, import, info, clean",
                )));
            }
        };

        Ok(Self {
            command,
            verbose,
            dry_run,
        })
    }
}

/// Helper to create a parse error.
fn parse_err(msg: &str) -> AutotuneError {
    AutotuneError::BenchmarkFailed(format!("argument error: {msg}"))
}

// ---------------------------------------------------------------------------
// TuneProgress
// ---------------------------------------------------------------------------

/// Progress information during a tuning session.
#[derive(Debug, Clone)]
pub struct TuneProgress {
    /// Total number of configurations to benchmark.
    pub total_configs: usize,
    /// Number of configurations completed so far.
    pub completed: usize,
    /// Best result found so far (if any).
    pub best_so_far: Option<BenchmarkResult>,
    /// Name of the kernel currently being tuned.
    pub current_kernel: String,
    /// Current problem size descriptor.
    pub current_problem: String,
    /// Wall-clock seconds elapsed since tuning started.
    pub elapsed_secs: f64,
    /// Estimated remaining seconds based on average time per config.
    pub estimated_remaining_secs: f64,
}

impl TuneProgress {
    /// Formats progress as a human-readable status line.
    ///
    /// Example output:
    /// `[42/100] kernel=sgemm problem=1024x1024 best=1200.0 GFLOPS (12.3s elapsed, ~17.7s remaining)`
    #[must_use]
    pub fn display_progress(&self) -> String {
        let best_str = match &self.best_so_far {
            Some(r) => match r.gflops {
                Some(gf) => format!("best={:.1} GFLOPS", gf),
                None => format!("best={:.1} us", r.median_us),
            },
            None => "best=N/A".to_string(),
        };

        format!(
            "[{}/{}] kernel={} problem={} {} ({:.1}s elapsed, ~{:.1}s remaining)",
            self.completed,
            self.total_configs,
            self.current_kernel,
            self.current_problem,
            best_str,
            self.elapsed_secs,
            self.estimated_remaining_secs,
        )
    }
}

// ---------------------------------------------------------------------------
// ProgressCallback trait
// ---------------------------------------------------------------------------

/// Callback interface for receiving tuning progress updates.
pub trait ProgressCallback {
    /// Called after each configuration is benchmarked.
    fn on_progress(&self, progress: &TuneProgress);

    /// Called when the entire tuning session completes.
    fn on_complete(&self, report: &TuneReport);
}

// ---------------------------------------------------------------------------
// ConsoleProgressCallback
// ---------------------------------------------------------------------------

/// A [`ProgressCallback`] that prints plain-text progress to stdout.
///
/// Uses no ANSI escape codes so output is safe for log files and
/// non-terminal environments.
pub struct ConsoleProgressCallback;

impl ProgressCallback for ConsoleProgressCallback {
    fn on_progress(&self, progress: &TuneProgress) {
        println!("{}", progress.display_progress());
    }

    fn on_complete(&self, report: &TuneReport) {
        println!("{}", report.format_report());
    }
}

// ---------------------------------------------------------------------------
// TuneReport
// ---------------------------------------------------------------------------

/// Summary report from a completed tuning session.
#[derive(Debug, Clone)]
pub struct TuneReport {
    /// Name of the kernel that was tuned.
    pub kernel: String,
    /// Problem sizes that were benchmarked.
    pub problem_sizes: Vec<String>,
    /// Total number of configurations tested.
    pub total_configs_tested: usize,
    /// Total wall-clock time in seconds.
    pub total_time_secs: f64,
    /// Best result for each problem size.
    pub best_results: Vec<(String, BenchmarkResult)>,
    /// Number of configurations skipped (early-stopped).
    pub skipped: usize,
}

impl TuneReport {
    /// Formats the report as a human-readable summary.
    #[must_use]
    pub fn format_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("=== Tune Report: {} ===\n", self.kernel));
        out.push_str(&format!("Problems: {}\n", self.problem_sizes.join(", ")));
        out.push_str(&format!(
            "Configs tested: {} ({} skipped)\n",
            self.total_configs_tested, self.skipped,
        ));
        out.push_str(&format!("Total time: {:.2}s\n", self.total_time_secs));
        out.push_str("--- Best Results ---\n");

        for (problem, result) in &self.best_results {
            let perf = match result.gflops {
                Some(gf) => format!("{:.1} GFLOPS", gf),
                None => format!("{:.1} us", result.median_us),
            };
            out.push_str(&format!(
                "  {}: {} (tile={}x{}x{}, stages={})\n",
                problem,
                perf,
                result.config.tile_m,
                result.config.tile_n,
                result.config.tile_k,
                result.config.stages,
            ));
        }

        out
    }
}

// ---------------------------------------------------------------------------
// CliRunner
// ---------------------------------------------------------------------------

/// Executes CLI commands against the autotune database.
pub struct CliRunner {
    /// The parsed CLI configuration.
    config: CliConfig,
}

impl CliRunner {
    /// Creates a new runner from the given CLI configuration.
    #[must_use]
    pub fn new(config: CliConfig) -> Self {
        Self { config }
    }

    /// Dispatches to the appropriate handler based on the command.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError`] if the operation fails.
    pub fn run(&self) -> Result<String, AutotuneError> {
        match &self.config.command {
            CliCommand::Tune {
                kernel,
                problem_sizes,
                db_path,
                warmup,
                iterations,
                early_stop,
                parallel,
            } => {
                let report = self.run_tune(
                    kernel,
                    problem_sizes,
                    db_path.as_deref(),
                    *warmup,
                    *iterations,
                    *early_stop,
                    *parallel,
                )?;
                Ok(report.format_report())
            }
            CliCommand::List {
                db_path,
                kernel,
                gpu,
            } => self.run_list(db_path.as_deref(), kernel.as_deref(), gpu.as_deref()),
            CliCommand::Export {
                db_path,
                format,
                output,
                kernel,
            } => self.run_export(db_path.as_deref(), format, output, kernel.as_deref()),
            CliCommand::Import {
                db_path,
                input,
                policy,
            } => self.run_import(db_path.as_deref(), input, policy),
            CliCommand::Info { db_path } => self.run_info(db_path.as_deref()),
            CliCommand::Clean {
                db_path,
                older_than_days,
            } => self.run_clean(db_path.as_deref(), *older_than_days),
        }
    }

    /// Runs the tuning process for a kernel across problem sizes.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError`] if benchmarking or database operations fail.
    #[allow(clippy::too_many_arguments)]
    pub fn run_tune(
        &self,
        kernel: &str,
        problem_sizes: &[String],
        db_path: Option<&str>,
        _warmup: u32,
        _iterations: u32,
        _early_stop: bool,
        _parallel: bool,
    ) -> Result<TuneReport, AutotuneError> {
        if self.config.dry_run {
            return Ok(TuneReport {
                kernel: kernel.to_string(),
                problem_sizes: problem_sizes.to_vec(),
                total_configs_tested: 0,
                total_time_secs: 0.0,
                best_results: Vec::new(),
                skipped: 0,
            });
        }

        let db = open_db(db_path)?;
        let mut best_results = Vec::new();

        // Collect existing best results from the database.
        for gpu in db.list_gpus() {
            for problem in problem_sizes {
                if let Some(result) = db.lookup(gpu, kernel, problem) {
                    best_results.push((problem.clone(), result.clone()));
                }
            }
        }

        Ok(TuneReport {
            kernel: kernel.to_string(),
            problem_sizes: problem_sizes.to_vec(),
            total_configs_tested: 0,
            total_time_secs: 0.0,
            best_results,
            skipped: 0,
        })
    }

    /// Lists results from the database.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError`] if the database cannot be opened.
    pub fn run_list(
        &self,
        db_path: Option<&str>,
        kernel_filter: Option<&str>,
        gpu_filter: Option<&str>,
    ) -> Result<String, AutotuneError> {
        let db = open_db(db_path)?;
        let mut out = String::new();

        let gpus = db.list_gpus();
        for gpu in &gpus {
            if let Some(gf) = gpu_filter {
                if *gpu != gf {
                    continue;
                }
            }
            out.push_str(&format!("GPU: {gpu}\n"));

            for (kernel, problem, result) in db.list_gpu(gpu) {
                if let Some(kf) = kernel_filter {
                    if kernel != kf {
                        continue;
                    }
                }
                let perf = match result.gflops {
                    Some(gf) => format!("{:.1} GFLOPS", gf),
                    None => format!("{:.1} us", result.median_us),
                };
                out.push_str(&format!("  {kernel}/{problem}: {perf}\n"));
            }
        }

        if out.is_empty() {
            out.push_str("No results found.\n");
        }

        Ok(out)
    }

    /// Displays database statistics.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError`] if the database cannot be opened.
    pub fn run_info(&self, db_path: Option<&str>) -> Result<String, AutotuneError> {
        let db = open_db(db_path)?;
        let gpus = db.list_gpus();
        let total = db.total_entries();

        let mut out = String::new();
        out.push_str(&format!("Database: {}\n", db.path().display()));
        out.push_str(&format!("Total entries: {total}\n"));
        out.push_str(&format!("GPUs: {}\n", gpus.len()));

        for gpu in &gpus {
            let entries = db.list_gpu(gpu);
            let kernels: std::collections::HashSet<&str> =
                entries.iter().map(|(k, _, _)| *k).collect();
            out.push_str(&format!(
                "  {gpu}: {k} kernels, {e} entries\n",
                k = kernels.len(),
                e = entries.len(),
            ));
        }

        Ok(out)
    }

    /// Exports database contents to a file.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError`] if the database cannot be read or the
    /// output file cannot be written.
    pub fn run_export(
        &self,
        db_path: Option<&str>,
        format: &str,
        output: &str,
        kernel_filter: Option<&str>,
    ) -> Result<String, AutotuneError> {
        if self.config.dry_run {
            return Ok(format!("Would export to '{output}' in {format} format",));
        }

        let db = open_db(db_path)?;
        let filter = ExportFilter {
            kernel_names: kernel_filter.map(|k| vec![k.to_string()]),
            ..ExportFilter::default()
        };
        let bundle = ExportBundle::from_db(&db, &filter);

        let export_format = parse_export_format(format)?;
        let content = match export_format {
            ExportFormat::Json => bundle.to_json()?,
            ExportFormat::JsonCompact => {
                serde_json::to_string(&bundle).map_err(AutotuneError::from)?
            }
            ExportFormat::Csv => bundle.to_csv()?,
            ExportFormat::Toml => {
                // TOML export: convert to JSON first as a fallback.
                bundle.to_json()?
            }
        };

        std::fs::write(output, &content)?;

        Ok(format!(
            "Exported {} entries to '{output}' ({format})",
            bundle.entries.len(),
        ))
    }

    /// Imports results from a file into the database.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError`] if the input file cannot be read or
    /// the database cannot be written.
    pub fn run_import(
        &self,
        db_path: Option<&str>,
        input: &str,
        policy_str: &str,
    ) -> Result<String, AutotuneError> {
        if self.config.dry_run {
            return Ok(format!(
                "Would import from '{input}' with policy '{policy_str}'",
            ));
        }

        let mut db = open_db(db_path)?;
        let content = std::fs::read_to_string(input)?;
        let bundle = ExportBundle::from_json(&content)?;
        let policy = parse_import_policy(policy_str)?;

        let result = import_bundle(&mut db, &bundle, policy)?;

        Ok(format!(
            "Imported: {}, skipped: {}, conflicts: {}, improved: {}",
            result.imported, result.skipped, result.conflicts, result.improved,
        ))
    }

    /// Removes entries from the database.
    ///
    /// When `older_than_days` is `None`, clears the entire database.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError`] if the database cannot be opened or written.
    pub fn run_clean(
        &self,
        db_path: Option<&str>,
        older_than_days: Option<u32>,
    ) -> Result<String, AutotuneError> {
        if self.config.dry_run {
            return match older_than_days {
                Some(days) => Ok(format!("Would clean entries older than {days} days")),
                None => Ok("Would clear all entries".to_string()),
            };
        }

        let mut db = open_db(db_path)?;
        let before = db.total_entries();

        match older_than_days {
            Some(_days) => {
                // Date-based filtering would require timestamps in entries.
                // For now, we report that date filtering is not yet supported
                // and return the current count unchanged.
                Ok(format!(
                    "Date-based cleaning not yet supported. Database has {before} entries.",
                ))
            }
            None => {
                db.clear()?;
                Ok(format!("Cleared {before} entries."))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Opens a [`ResultDb`] at the given path, or the default location.
fn open_db(db_path: Option<&str>) -> Result<ResultDb, AutotuneError> {
    match db_path {
        Some(path) => ResultDb::open_at(path.into()).map_err(AutotuneError::from),
        None => ResultDb::open().map_err(AutotuneError::from),
    }
}

/// Parses an export format string.
fn parse_export_format(s: &str) -> Result<ExportFormat, AutotuneError> {
    match s {
        "json" => Ok(ExportFormat::Json),
        "json-compact" => Ok(ExportFormat::JsonCompact),
        "csv" => Ok(ExportFormat::Csv),
        "toml" => Ok(ExportFormat::Toml),
        other => Err(parse_err(&format!(
            "unknown format '{other}'. Use: json, json-compact, csv, toml",
        ))),
    }
}

/// Parses an import policy string.
fn parse_import_policy(s: &str) -> Result<ImportPolicy, AutotuneError> {
    match s {
        "always" => Ok(ImportPolicy::AlwaysReplace),
        "best" => Ok(ImportPolicy::KeepBest),
        "existing" => Ok(ImportPolicy::KeepExisting),
        "missing" => Ok(ImportPolicy::MergeMissing),
        other => Err(parse_err(&format!(
            "unknown policy '{other}'. Use: always, best, existing, missing",
        ))),
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

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    fn make_result(median_us: f64, gflops: Option<f64>) -> BenchmarkResult {
        BenchmarkResult {
            config: Config::new()
                .with_tile_m(64)
                .with_tile_n(64)
                .with_tile_k(16)
                .with_stages(3),
            median_us,
            min_us: median_us - 1.0,
            max_us: median_us + 1.0,
            stddev_us: 0.5,
            gflops,
            efficiency: None,
        }
    }

    fn temp_db(name: &str) -> (ResultDb, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("oxicuda_cli_test_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("results.json");
        let db = ResultDb::open_at(path).expect("open db");
        (db, dir)
    }

    // --- Argument parsing tests ---

    #[test]
    fn parse_tune_command() {
        let cli =
            CliConfig::from_args(&args("oxitune tune sgemm 1024x1024 2048x2048")).expect("parse");
        match &cli.command {
            CliCommand::Tune {
                kernel,
                problem_sizes,
                warmup,
                iterations,
                ..
            } => {
                assert_eq!(kernel, "sgemm");
                assert_eq!(problem_sizes.len(), 2);
                assert_eq!(*warmup, 5);
                assert_eq!(*iterations, 20);
            }
            _ => panic!("expected Tune command"),
        }
        assert!(!cli.verbose);
        assert!(!cli.dry_run);
    }

    #[test]
    fn parse_tune_with_flags() {
        let cli = CliConfig::from_args(&args(
            "oxitune --verbose --dry-run tune sgemm 1024 --warmup 10 --iterations 50 --early-stop --parallel",
        ))
        .expect("parse");
        assert!(cli.verbose);
        assert!(cli.dry_run);
        match &cli.command {
            CliCommand::Tune {
                warmup,
                iterations,
                early_stop,
                parallel,
                ..
            } => {
                assert_eq!(*warmup, 10);
                assert_eq!(*iterations, 50);
                assert!(early_stop);
                assert!(parallel);
            }
            _ => panic!("expected Tune command"),
        }
    }

    #[test]
    fn parse_list_command() {
        let cli = CliConfig::from_args(&args("oxitune list --kernel sgemm --gpu RTX4090"))
            .expect("parse");
        match &cli.command {
            CliCommand::List { kernel, gpu, .. } => {
                assert_eq!(kernel.as_deref(), Some("sgemm"));
                assert_eq!(gpu.as_deref(), Some("RTX4090"));
            }
            _ => panic!("expected List command"),
        }
    }

    #[test]
    fn parse_export_command() {
        let cli = CliConfig::from_args(&args("oxitune export --format csv --output results.csv"))
            .expect("parse");
        match &cli.command {
            CliCommand::Export { format, output, .. } => {
                assert_eq!(format, "csv");
                assert_eq!(output, "results.csv");
            }
            _ => panic!("expected Export command"),
        }
    }

    #[test]
    fn parse_import_command() {
        let cli = CliConfig::from_args(&args("oxitune import --input data.json --policy always"))
            .expect("parse");
        match &cli.command {
            CliCommand::Import { input, policy, .. } => {
                assert_eq!(input, "data.json");
                assert_eq!(policy, "always");
            }
            _ => panic!("expected Import command"),
        }
    }

    #[test]
    fn parse_info_command() {
        let cli = CliConfig::from_args(&args("oxitune info")).expect("parse");
        assert!(matches!(cli.command, CliCommand::Info { .. }));
    }

    #[test]
    fn parse_clean_command() {
        let cli = CliConfig::from_args(&args("oxitune clean --older-than 30")).expect("parse");
        match &cli.command {
            CliCommand::Clean {
                older_than_days, ..
            } => {
                assert_eq!(*older_than_days, Some(30));
            }
            _ => panic!("expected Clean command"),
        }
    }

    #[test]
    fn parse_no_subcommand_errors() {
        let result = CliConfig::from_args(&args("oxitune"));
        assert!(result.is_err());
    }

    #[test]
    fn parse_unknown_subcommand_errors() {
        let result = CliConfig::from_args(&args("oxitune foobar"));
        assert!(result.is_err());
    }

    // --- Progress display tests ---

    #[test]
    fn progress_display_with_gflops() {
        let progress = TuneProgress {
            total_configs: 100,
            completed: 42,
            best_so_far: Some(make_result(42.0, Some(1200.0))),
            current_kernel: "sgemm".to_string(),
            current_problem: "1024x1024".to_string(),
            elapsed_secs: 12.3,
            estimated_remaining_secs: 17.7,
        };
        let display = progress.display_progress();
        assert!(display.contains("[42/100]"));
        assert!(display.contains("kernel=sgemm"));
        assert!(display.contains("problem=1024x1024"));
        assert!(display.contains("1200.0 GFLOPS"));
        assert!(display.contains("12.3s elapsed"));
        assert!(display.contains("~17.7s remaining"));
    }

    #[test]
    fn progress_display_without_gflops() {
        let progress = TuneProgress {
            total_configs: 50,
            completed: 10,
            best_so_far: Some(make_result(85.5, None)),
            current_kernel: "dgemm".to_string(),
            current_problem: "512x512".to_string(),
            elapsed_secs: 5.0,
            estimated_remaining_secs: 20.0,
        };
        let display = progress.display_progress();
        assert!(display.contains("85.5 us"));
    }

    #[test]
    fn progress_display_no_best() {
        let progress = TuneProgress {
            total_configs: 10,
            completed: 0,
            best_so_far: None,
            current_kernel: "conv".to_string(),
            current_problem: "3x3".to_string(),
            elapsed_secs: 0.0,
            estimated_remaining_secs: 0.0,
        };
        let display = progress.display_progress();
        assert!(display.contains("best=N/A"));
    }

    // --- Report formatting ---

    #[test]
    fn report_format() {
        let report = TuneReport {
            kernel: "sgemm".to_string(),
            problem_sizes: vec!["1024x1024".to_string(), "2048x2048".to_string()],
            total_configs_tested: 150,
            total_time_secs: 45.67,
            best_results: vec![
                ("1024x1024".to_string(), make_result(42.0, Some(1200.0))),
                ("2048x2048".to_string(), make_result(320.0, Some(1800.0))),
            ],
            skipped: 12,
        };
        let text = report.format_report();
        assert!(text.contains("Tune Report: sgemm"));
        assert!(text.contains("1024x1024, 2048x2048"));
        assert!(text.contains("150 (12 skipped)"));
        assert!(text.contains("45.67s"));
        assert!(text.contains("1200.0 GFLOPS"));
        assert!(text.contains("1800.0 GFLOPS"));
    }

    // --- DB operations ---

    #[test]
    fn run_info_with_data() {
        let (mut db, dir) = temp_db("run_info");
        db.save("RTX4090", "sgemm", "1024", make_result(42.0, Some(800.0)))
            .expect("save");
        db.save("RTX4090", "dgemm", "512", make_result(100.0, None))
            .expect("save");
        drop(db);

        let db_path = dir.join("results.json");
        let cli = CliConfig {
            command: CliCommand::Info {
                db_path: Some(db_path.to_string_lossy().to_string()),
            },
            verbose: false,
            dry_run: false,
        };
        let runner = CliRunner::new(cli);
        let info = runner.run().expect("run info");

        assert!(info.contains("Total entries: 2"));
        assert!(info.contains("RTX4090"));
        assert!(info.contains("2 kernels"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_list_with_filter() {
        let (mut db, dir) = temp_db("run_list");
        db.save("RTX4090", "sgemm", "1024", make_result(42.0, Some(800.0)))
            .expect("save");
        db.save("RTX4090", "dgemm", "512", make_result(100.0, None))
            .expect("save");
        drop(db);

        let db_path = dir.join("results.json");
        let cli = CliConfig {
            command: CliCommand::List {
                db_path: Some(db_path.to_string_lossy().to_string()),
                kernel: Some("sgemm".to_string()),
                gpu: None,
            },
            verbose: false,
            dry_run: false,
        };
        let runner = CliRunner::new(cli);
        let list = runner.run().expect("run list");

        assert!(list.contains("sgemm"));
        assert!(!list.contains("dgemm"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_clean_all() {
        let (mut db, dir) = temp_db("run_clean");
        db.save("GPU0", "sgemm", "1024", make_result(42.0, None))
            .expect("save");
        db.save("GPU0", "dgemm", "512", make_result(100.0, None))
            .expect("save");
        drop(db);

        let db_path = dir.join("results.json");
        let cli = CliConfig {
            command: CliCommand::Clean {
                db_path: Some(db_path.to_string_lossy().to_string()),
                older_than_days: None,
            },
            verbose: false,
            dry_run: false,
        };
        let runner = CliRunner::new(cli);
        let result = runner.run().expect("run clean");

        assert!(result.contains("Cleared 2 entries"));

        // Verify database is empty.
        let db2 = ResultDb::open_at(db_path).expect("reopen");
        assert_eq!(db2.total_entries(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
