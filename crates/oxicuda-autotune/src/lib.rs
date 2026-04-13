//! # OxiCUDA Autotune — Automatic GPU Kernel Parameter Optimization
//!
//! This crate provides a measurement-based autotuning engine for GPU kernel
//! parameters.  Given a search space of candidate configurations (tile sizes,
//! pipeline depths, thread counts, etc.), the engine benchmarks each viable
//! configuration and persists the best result for future lookups.
//!
//! ## Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────┐
//!  │                     Autotuner                           │
//!  │                                                         │
//!  │  SearchSpace ──► prune(arch) ──► [Config₁..Configₙ]    │
//!  │                                       │                 │
//!  │                                       ▼                 │
//!  │                              BenchmarkEngine            │
//!  │                     (warmup + event-timed runs)         │
//!  │                                       │                 │
//!  │                                       ▼                 │
//!  │                             BenchmarkResult             │
//!  │                        (median, min, max, GFLOPS)       │
//!  │                                       │                 │
//!  │                                       ▼                 │
//!  │                               ResultDb                  │
//!  │                     (GPU → kernel → problem → best)     │
//!  └─────────────────────────────────────────────────────────┘
//!
//!  At runtime:
//!
//!  ┌──────────────────────────────────────────────────┐
//!  │                   Dispatcher                     │
//!  │                                                  │
//!  │  Tier 1: exact match in ResultDb                 │
//!  │  Tier 2: nearest-neighbor interpolation           │
//!  │  Tier 3: default Config                          │
//!  └──────────────────────────────────────────────────┘
//! ```
//!
//! ## Workflow
//!
//! 1. **Define the search space** — use [`SearchSpace::gemm_default()`] or
//!    build a custom space with [`SearchSpaceBuilder`].
//!
//! 2. **Prune** — call [`SearchSpace::prune()`] with the target GPU's
//!    shared memory and register limits to eliminate infeasible configs.
//!
//! 3. **Benchmark** — for each surviving [`Config`], use
//!    [`BenchmarkEngine::benchmark()`] to measure GPU execution time
//!    via CUDA events.
//!
//! 4. **Persist** — store the best [`BenchmarkResult`] in the
//!    [`ResultDb`] for this (GPU, kernel, problem) triple.
//!
//! 5. **Dispatch** — at runtime, use [`Dispatcher::select_config()`]
//!    to retrieve the optimal config with 3-tier fallback.
//!
//! ## Key types
//!
//! | Type                | Purpose                                     |
//! |---------------------|---------------------------------------------|
//! | [`Config`]          | One point in the tuning search space         |
//! | [`SearchSpace`]     | Defines candidate values per dimension       |
//! | [`BenchmarkEngine`] | Measures kernel timing via CUDA events       |
//! | [`BenchmarkResult`] | Timing statistics for one configuration      |
//! | [`ResultDb`]        | Persistent JSON-backed result storage        |
//! | [`Dispatcher`]      | Runtime config selection with 3-tier fallback|
//! | [`TunableKernel`]   | Trait for kernels that can be autotuned      |
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxicuda_autotune::prelude::*;
//!
//! # fn example() -> Result<(), oxicuda_autotune::AutotuneError> {
//! // 1. Define and prune the search space.
//! let space = SearchSpace::gemm_default();
//! let configs = space.prune(48 * 1024, 255, 4); // 48 KiB smem, 255 regs, f32
//!
//! // 2. At runtime, select the best config.
//! let dispatcher = Dispatcher::new("NVIDIA RTX 4090".to_string())?;
//! let config = dispatcher.select_config("sgemm", "1024x1024x1024");
//! println!("Using tile_m={}, tile_n={}", config.tile_m, config.tile_n);
//! # Ok(())
//! # }
//! ```

#![warn(clippy::all)]
#![warn(missing_docs)]

// ---------------------------------------------------------------------------
// Module declarations
// ---------------------------------------------------------------------------

pub mod adaptive;
pub mod bayesian;
pub mod benchmark;
pub mod cli;
pub mod config;
pub mod db_migration;
pub mod dispatch;
pub mod distributed;
pub mod early_stopping;
pub mod error;
pub mod export;
pub mod genetic;
pub mod guided_ptx;
pub mod incremental;
pub mod interpolation;
pub mod kernel_similarity;
pub mod memory_constrained;
pub mod multi_objective;
pub mod parallel_bench;
pub mod power_aware;

pub mod ptx_integration;
pub mod result_db;
pub mod search_space;
pub mod simulated_annealing;
pub mod transfer_learning;
pub mod tunable;
pub mod visualization;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use adaptive::{
    AdaptivePolicy, AdaptiveSelector, AdaptiveTuneReport, ExplorationScheduler, MetricsWindow,
    PerformanceRegression, PerformanceTrend, RuntimeMetric, SwitchDecision,
};
pub use bayesian::{AcquisitionFunction, BayesianOptimizer, GaussianProcess, GpPrediction};
pub use benchmark::{BenchmarkConfig, BenchmarkEngine, BenchmarkResult};
pub use cli::{
    CliCommand, CliConfig, CliRunner, ConsoleProgressCallback, ProgressCallback, TuneProgress,
    TuneReport,
};
pub use config::Config;
pub use dispatch::{DispatchTier, Dispatcher};
pub use distributed::{
    DistributedCoordinator, DistributedTuneConfig, DistributedTuneJob, InMemoryTransport, NodeId,
    Transport, TuneMessage, TuneTask, TuneTaskResult, WorkDistributor,
};
pub use early_stopping::{
    EarlyStoppingConfig, EarlyStoppingSummary, EarlyStoppingTracker, StopReason,
};
pub use error::{AutotuneError, AutotuneResult};
pub use export::{
    ExportBundle, ExportEntry, ExportFilter, ExportFormat, ExportManifest, ImportPolicy,
    ImportResult, ValidationWarning, import_bundle, validate_bundle,
};
pub use genetic::{GeneticAlgorithm, GeneticConfig};
pub use guided_ptx::{
    GuidedPtxGenerator, GuidedPtxStrategy, HintMerger, PtxGenerationHint, PtxSpecialization,
};
pub use incremental::{
    ChangeDetector, HardwareChange, HardwareFingerprint, IncrementalTuner, RetunePlan,
    RetunePriority, RetuneReport,
};
pub use interpolation::{ProblemSize, SizeInterpolator};
pub use kernel_similarity::{
    AccessPattern, ArithmeticIntensity, ConfigAdapter, KernelSignature, KernelSimilarityIndex,
    ReductionType, SimilarityMatch, SimilarityScore,
};
pub use memory_constrained::{
    BudgetSuggestion, BudgetUtilization, ConstrainedSearchSpace, ConstrainedTuner,
    GemmMemoryEstimator, MemoryBudget, MemoryEstimate, MemoryEstimator, ProblemDims,
};
pub use multi_objective::{
    MultiObjectiveOptimizer, MultiObjectiveResult, Objective, ObjectiveDirection, ObjectiveSpec,
    ObjectiveValue, ParetoFront, WeightedScalarization,
};
pub use parallel_bench::{
    BenchProgress, BenchProgressCallback, ParallelBenchConfig, ParallelBenchResult,
    ParallelBenchmarkEngine, ParallelStrategy, StreamPool,
};
pub use power_aware::{
    NvidiaSmiMonitor, PowerAwareBenchmarkEngine, PowerAwareBenchmarkResult, PowerAwareSelector,
    PowerConstraint, PowerMonitor, PowerProfile, PowerReading, SyntheticPowerMonitor,
};
pub use ptx_integration::{
    TemplateAutotuner, elementwise_search_space, gemm_search_space, reduction_search_space,
    scan_search_space,
};
pub use result_db::{ProblemKey, ResultDb};
pub use search_space::{SearchSpace, SearchSpaceBuilder};
pub use simulated_annealing::{SimulatedAnnealing, SimulatedAnnealingConfig};
pub use transfer_learning::{
    ArchitectureProfile, ArchitectureSimilarity, ConfigTransferRule, ConfigTransformer,
    TransferLearningEngine, TransferStrategy, TransferredConfig,
};
pub use tunable::TunableKernel;
pub use visualization::{
    AsciiChart, ChartData, CsvVisualizationExporter, DataPoint, DataSeries, GnuplotExporter,
    SeriesStyle, VisualizationBuilder,
};

// ---------------------------------------------------------------------------
// Prelude — convenient glob import
// ---------------------------------------------------------------------------

/// Convenient glob import for common OxiCUDA Autotune types.
///
/// ```rust
/// use oxicuda_autotune::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        AccessPattern, AcquisitionFunction, AdaptivePolicy, AdaptiveSelector, AdaptiveTuneReport,
        ArchitectureProfile, ArchitectureSimilarity, ArithmeticIntensity, AutotuneError,
        AutotuneResult, BayesianOptimizer, BenchmarkConfig, BenchmarkEngine, BenchmarkResult,
        BudgetSuggestion, BudgetUtilization, Config, ConfigAdapter, ConfigTransferRule,
        ConfigTransformer, ConstrainedSearchSpace, ConstrainedTuner, DispatchTier, Dispatcher,
        EarlyStoppingConfig, EarlyStoppingSummary, EarlyStoppingTracker, ExplorationScheduler,
        GaussianProcess, GemmMemoryEstimator, GeneticAlgorithm, GeneticConfig, GpPrediction,
        GuidedPtxGenerator, GuidedPtxStrategy, HintMerger, KernelSignature, KernelSimilarityIndex,
        MemoryBudget, MemoryEstimate, MemoryEstimator, MetricsWindow, MultiObjectiveOptimizer,
        MultiObjectiveResult, Objective, ObjectiveDirection, ObjectiveSpec, ObjectiveValue,
        ParetoFront, PerformanceRegression, PerformanceTrend, PowerAwareBenchmarkEngine,
        PowerAwareBenchmarkResult, PowerAwareSelector, PowerConstraint, PowerMonitor, PowerProfile,
        PowerReading, ProblemDims, ProblemKey, ProblemSize, PtxGenerationHint, PtxSpecialization,
        ReductionType, ResultDb, RuntimeMetric, SearchSpace, SearchSpaceBuilder, SimilarityMatch,
        SimilarityScore, SimulatedAnnealing, SimulatedAnnealingConfig, SizeInterpolator,
        StopReason, SwitchDecision, TemplateAutotuner, TransferLearningEngine, TransferStrategy,
        TransferredConfig, TunableKernel, WeightedScalarization,
    };
}
