//! Memory-constrained autotuning.
//!
//! This module provides types and algorithms for maximising kernel
//! throughput within a fixed VRAM budget.  The key workflow is:
//!
//! 1. Define a [`MemoryBudget`] describing the hardware limits.
//! 2. Implement [`MemoryEstimator`] (or use [`GemmMemoryEstimator`]).
//! 3. Wrap a [`SearchSpace`] in a [`ConstrainedSearchSpace`] to filter
//!    and rank configurations by memory efficiency.
//! 4. Use [`ConstrainedTuner`] to benchmark only feasible configs and
//!    select the best one.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::memory_constrained::*;
//! use oxicuda_autotune::{SearchSpace, Config};
//!
//! let budget = MemoryBudget::new()
//!     .with_max_shared_mem_per_block(48 * 1024)
//!     .with_max_global_mem(8 * 1024 * 1024 * 1024)
//!     .with_max_registers_per_thread(255)
//!     .with_reserved_mem(64 * 1024 * 1024);
//!
//! let dims = ProblemDims::new(1024, 1024, 1024, 1, 4);
//! let estimator = GemmMemoryEstimator;
//!
//! let space = SearchSpace::minimal();
//! let constrained = ConstrainedSearchSpace::new(space, budget, dims);
//! let viable = constrained.prune_by_memory(&estimator);
//! assert!(!viable.is_empty());
//! ```

use serde::{Deserialize, Serialize};

use crate::benchmark::{BenchmarkConfig, BenchmarkEngine, BenchmarkResult};
use crate::config::Config;
use crate::error::{AutotuneError, AutotuneResult};
use crate::search_space::SearchSpace;

// ---------------------------------------------------------------------------
// MemoryBudget
// ---------------------------------------------------------------------------

/// Describes the memory constraints of a target GPU.
///
/// All sizes are in bytes unless otherwise noted.  Use the builder
/// methods to configure each limit individually.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBudget {
    /// Maximum shared memory available per thread block (bytes).
    pub max_shared_mem_per_block: u64,
    /// Total global (VRAM) memory available on the device (bytes).
    pub max_global_mem: u64,
    /// Maximum number of registers per thread.
    pub max_registers_per_thread: u32,
    /// Memory reserved for non-kernel uses such as workspace
    /// allocations, cuDNN scratch buffers, etc. (bytes).
    pub reserved_mem: u64,
}

impl MemoryBudget {
    /// Creates a budget with generous defaults suitable for modern GPUs.
    ///
    /// | Field                      | Default           |
    /// |----------------------------|-------------------|
    /// | `max_shared_mem_per_block`  | 48 KiB            |
    /// | `max_global_mem`            | 8 GiB             |
    /// | `max_registers_per_thread`  | 255               |
    /// | `reserved_mem`              | 0                 |
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the per-block shared memory limit.
    #[must_use]
    pub fn with_max_shared_mem_per_block(mut self, bytes: u64) -> Self {
        self.max_shared_mem_per_block = bytes;
        self
    }

    /// Sets the total global memory limit.
    #[must_use]
    pub fn with_max_global_mem(mut self, bytes: u64) -> Self {
        self.max_global_mem = bytes;
        self
    }

    /// Sets the per-thread register limit.
    #[must_use]
    pub fn with_max_registers_per_thread(mut self, regs: u32) -> Self {
        self.max_registers_per_thread = regs;
        self
    }

    /// Sets the amount of memory reserved for other uses.
    #[must_use]
    pub fn with_reserved_mem(mut self, bytes: u64) -> Self {
        self.reserved_mem = bytes;
        self
    }

    /// Returns the effective global memory available for kernel data
    /// (total minus reserved).
    #[must_use]
    pub fn available_global_mem(&self) -> u64 {
        self.max_global_mem.saturating_sub(self.reserved_mem)
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            max_shared_mem_per_block: 48 * 1024,
            max_global_mem: 8 * 1024 * 1024 * 1024,
            max_registers_per_thread: 255,
            reserved_mem: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryEstimate
// ---------------------------------------------------------------------------

/// Estimated memory footprint of a kernel configuration.
///
/// Each field describes one class of GPU memory consumed by running a
/// kernel with a given [`Config`] and [`ProblemDims`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    /// Shared memory consumed per thread block (bytes).
    pub shared_mem_bytes: u64,
    /// Global memory consumed by input/output tensors (bytes).
    pub global_mem_bytes: u64,
    /// Registers consumed per thread.
    pub registers_per_thread: u32,
    /// Extra global memory needed for workspace buffers such as
    /// split-K partial sums (bytes).
    pub workspace_bytes: u64,
}

impl MemoryEstimate {
    /// Total global memory consumed (data + workspace).
    #[must_use]
    pub fn total_global_bytes(&self) -> u64 {
        self.global_mem_bytes.saturating_add(self.workspace_bytes)
    }

    /// Returns `true` if every resource fits within `budget`.
    #[must_use]
    pub fn fits_budget(&self, budget: &MemoryBudget) -> bool {
        self.shared_mem_bytes <= budget.max_shared_mem_per_block
            && self.registers_per_thread <= budget.max_registers_per_thread
            && self.total_global_bytes() <= budget.available_global_mem()
    }
}

// ---------------------------------------------------------------------------
// ProblemDims
// ---------------------------------------------------------------------------

/// Dimensions of a GEMM (or GEMM-like) problem.
///
/// Used by [`MemoryEstimator`] implementations to compute the
/// memory footprint of input/output tensors and workspace buffers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemDims {
    /// Number of rows in the output matrix (M dimension).
    pub m: u64,
    /// Number of columns in the output matrix (N dimension).
    pub n: u64,
    /// Shared/reduction dimension (K dimension).
    pub k: u64,
    /// Batch count (1 for non-batched GEMM).
    pub batch_count: u64,
    /// Size of one matrix element in bytes (e.g. 4 for `f32`).
    pub element_bytes: u32,
}

impl ProblemDims {
    /// Creates a new `ProblemDims`.
    #[must_use]
    pub fn new(m: u64, n: u64, k: u64, batch_count: u64, element_bytes: u32) -> Self {
        Self {
            m,
            n,
            k,
            batch_count,
            element_bytes,
        }
    }

    /// Total floating-point operations for a standard GEMM
    /// (`C = alpha * A * B + beta * C`).
    ///
    /// This counts `2 * M * N * K * batch_count` FLOPs (one multiply
    /// and one add per output element per K iteration).
    #[must_use]
    pub fn gemm_flops(&self) -> f64 {
        2.0 * self.m as f64 * self.n as f64 * self.k as f64 * self.batch_count as f64
    }
}

// ---------------------------------------------------------------------------
// MemoryEstimator trait
// ---------------------------------------------------------------------------

/// Trait for computing the memory footprint of a kernel configuration.
///
/// Implementations are kernel-specific: a GEMM estimator differs from
/// a convolution estimator, etc.
pub trait MemoryEstimator {
    /// Estimates the memory consumption of `config` for the given
    /// `problem_size`.
    fn estimate(&self, config: &Config, problem_size: &ProblemDims) -> MemoryEstimate;
}

// ---------------------------------------------------------------------------
// GemmMemoryEstimator
// ---------------------------------------------------------------------------

/// [`MemoryEstimator`] for standard dense GEMM kernels.
///
/// # Memory model
///
/// | Resource         | Formula                                                     |
/// |------------------|-------------------------------------------------------------|
/// | Shared memory    | `(tile_m*tile_k + tile_k*tile_n) * stages * element_bytes`  |
/// | Global memory    | `(m*k + k*n + m*n) * batch_count * element_bytes`           |
/// | Registers/thread | heuristic from warp tile and tensor-core usage               |
/// | Workspace        | `split_k * m * n * element_bytes * batch_count` (if split-K) |
#[derive(Debug, Clone, Copy)]
pub struct GemmMemoryEstimator;

impl MemoryEstimator for GemmMemoryEstimator {
    fn estimate(&self, config: &Config, problem_size: &ProblemDims) -> MemoryEstimate {
        let eb = u64::from(problem_size.element_bytes);

        // Shared memory: double-buffered A and B tiles
        let a_tile = u64::from(config.tile_m) * u64::from(config.tile_k);
        let b_tile = u64::from(config.tile_k) * u64::from(config.tile_n);
        let shared_mem_bytes = (a_tile + b_tile) * u64::from(config.stages) * eb;

        // Global memory: A(m*k) + B(k*n) + C(m*n), batched
        let a_global = problem_size.m * problem_size.k;
        let b_global = problem_size.k * problem_size.n;
        let c_global = problem_size.m * problem_size.n;
        let global_mem_bytes = (a_global + b_global + c_global) * problem_size.batch_count * eb;

        // Registers per thread (heuristic)
        let registers_per_thread = config.estimated_registers_per_thread();

        // Workspace for split-K
        let split_k = u64::from(config.extra.get("split_k").copied().unwrap_or(1));
        let workspace_bytes = if split_k > 1 {
            split_k * problem_size.m * problem_size.n * eb * problem_size.batch_count
        } else {
            0
        };

        MemoryEstimate {
            shared_mem_bytes,
            global_mem_bytes,
            registers_per_thread,
            workspace_bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// ConstrainedSearchSpace
// ---------------------------------------------------------------------------

/// A search space wrapped with memory budget constraints.
///
/// Provides methods to filter out configurations that exceed the budget
/// and to rank the survivors by memory efficiency.
#[derive(Debug, Clone)]
pub struct ConstrainedSearchSpace {
    /// The underlying search space.
    pub space: SearchSpace,
    /// The memory budget to enforce.
    pub budget: MemoryBudget,
    /// Problem dimensions used for estimation.
    pub dims: ProblemDims,
}

impl ConstrainedSearchSpace {
    /// Creates a new constrained search space.
    #[must_use]
    pub fn new(space: SearchSpace, budget: MemoryBudget, dims: ProblemDims) -> Self {
        Self {
            space,
            budget,
            dims,
        }
    }

    /// Prunes configurations that exceed the memory budget.
    ///
    /// First applies the architectural pruning from [`SearchSpace::prune`],
    /// then removes any configuration whose estimated memory footprint
    /// exceeds the budget.
    #[must_use]
    pub fn prune_by_memory(&self, estimator: &dyn MemoryEstimator) -> Vec<Config> {
        // First do architectural pruning
        let arch_pruned = self.space.prune(
            self.budget.max_shared_mem_per_block as usize,
            self.budget.max_registers_per_thread,
            self.dims.element_bytes,
        );

        // Then apply memory budget filtering
        arch_pruned
            .into_iter()
            .filter(|cfg| {
                let est = estimator.estimate(cfg, &self.dims);
                est.fits_budget(&self.budget)
            })
            .collect()
    }

    /// Prunes by memory then sorts survivors by memory efficiency
    /// (ascending total memory usage, so the most memory-efficient
    /// configs come first).
    #[must_use]
    pub fn rank_by_efficiency(
        &self,
        estimator: &dyn MemoryEstimator,
    ) -> Vec<(Config, MemoryEstimate)> {
        let viable = self.prune_by_memory(estimator);

        let mut ranked: Vec<(Config, MemoryEstimate)> = viable
            .into_iter()
            .map(|cfg| {
                let est = estimator.estimate(&cfg, &self.dims);
                (cfg, est)
            })
            .collect();

        // Sort by total global bytes ascending — configs that use
        // less memory rank higher (more room for other work).
        ranked.sort_by(|a, b| {
            let a_total = a.1.total_global_bytes() + a.1.shared_mem_bytes;
            let b_total = b.1.total_global_bytes() + b.1.shared_mem_bytes;
            a_total.cmp(&b_total)
        });

        ranked
    }

    /// Returns the minimum shared memory required by any configuration
    /// in the search space.
    ///
    /// Useful for [`ConstrainedTuner::suggest_budget_increase`].
    #[must_use]
    pub fn min_shared_mem(&self, estimator: &dyn MemoryEstimator) -> u64 {
        self.space
            .enumerate()
            .iter()
            .map(|cfg| estimator.estimate(cfg, &self.dims).shared_mem_bytes)
            .min()
            .unwrap_or(0)
    }

    /// Returns the minimum global memory required by any configuration
    /// in the search space.
    #[must_use]
    pub fn min_global_mem(&self, estimator: &dyn MemoryEstimator) -> u64 {
        self.space
            .enumerate()
            .iter()
            .map(|cfg| estimator.estimate(cfg, &self.dims).total_global_bytes())
            .min()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// ConstrainedTuner
// ---------------------------------------------------------------------------

/// Orchestrates memory-aware autotuning.
///
/// The tuner benchmarks only configurations that fit within the memory
/// budget and returns the best-performing one.  If no configuration
/// fits, it can suggest a minimum budget increase.
pub struct ConstrainedTuner {
    /// The constrained search space.
    constrained_space: ConstrainedSearchSpace,
    /// Benchmark engine for timing.
    bench_engine: BenchmarkEngine,
}

impl ConstrainedTuner {
    /// Creates a new constrained tuner.
    #[must_use]
    pub fn new(constrained_space: ConstrainedSearchSpace) -> Self {
        Self {
            constrained_space,
            bench_engine: BenchmarkEngine::new(),
        }
    }

    /// Creates a tuner with a custom benchmark configuration.
    #[must_use]
    pub fn with_bench_config(
        constrained_space: ConstrainedSearchSpace,
        bench_config: BenchmarkConfig,
    ) -> Self {
        Self {
            constrained_space,
            bench_engine: BenchmarkEngine::with_config(bench_config),
        }
    }

    /// Returns the constrained search space.
    #[must_use]
    pub fn constrained_space(&self) -> &ConstrainedSearchSpace {
        &self.constrained_space
    }

    /// Benchmarks all memory-feasible configurations using wall-clock
    /// timing and returns the best result.
    ///
    /// The `run_fn` closure receives a [`Config`] reference and should
    /// execute the kernel (or a representative workload) once.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::NoViableConfig`] if no configuration
    /// fits the budget.
    pub fn tune<F>(
        &self,
        estimator: &dyn MemoryEstimator,
        run_fn: F,
    ) -> AutotuneResult<BenchmarkResult>
    where
        F: Fn(&Config) -> Result<(), AutotuneError>,
    {
        let viable = self.constrained_space.prune_by_memory(estimator);

        if viable.is_empty() {
            return Err(AutotuneError::NoViableConfig);
        }

        let flops = self.constrained_space.dims.gemm_flops();

        let mut best: Option<BenchmarkResult> = None;

        for cfg in &viable {
            let cfg_clone = cfg.clone();
            let result = self
                .bench_engine
                .benchmark_wallclock(&cfg_clone, Some(flops), || run_fn(cfg));

            match result {
                Ok(res) => {
                    let dominated = best
                        .as_ref()
                        .map(|b| res.median_us < b.median_us)
                        .unwrap_or(true);
                    if dominated {
                        best = Some(res);
                    }
                }
                Err(_) => {
                    // Skip configs that fail to benchmark — they may
                    // hit runtime limits not captured by the estimator.
                    continue;
                }
            }
        }

        best.ok_or(AutotuneError::NoViableConfig)
    }

    /// Computes what fraction of the memory budget a configuration uses.
    ///
    /// Returns a value in `[0.0, ∞)` where 1.0 means the config exactly
    /// fills the budget.  Values above 1.0 mean the config exceeds the
    /// budget.
    #[must_use]
    pub fn budget_utilization(
        &self,
        config: &Config,
        estimator: &dyn MemoryEstimator,
    ) -> BudgetUtilization {
        let est = estimator.estimate(config, &self.constrained_space.dims);
        let budget = &self.constrained_space.budget;

        let shared_util = if budget.max_shared_mem_per_block > 0 {
            est.shared_mem_bytes as f64 / budget.max_shared_mem_per_block as f64
        } else {
            0.0
        };

        let global_util = if budget.available_global_mem() > 0 {
            est.total_global_bytes() as f64 / budget.available_global_mem() as f64
        } else {
            0.0
        };

        let register_util = if budget.max_registers_per_thread > 0 {
            est.registers_per_thread as f64 / budget.max_registers_per_thread as f64
        } else {
            0.0
        };

        BudgetUtilization {
            shared_mem_fraction: shared_util,
            global_mem_fraction: global_util,
            register_fraction: register_util,
            overall: shared_util.max(global_util).max(register_util),
        }
    }

    /// Suggests the minimum budget increase needed to admit at least
    /// one configuration.
    ///
    /// Returns `None` if there are already viable configurations.
    /// Otherwise, returns a [`BudgetSuggestion`] with the minimum
    /// shared memory and global memory needed.
    #[must_use]
    pub fn suggest_budget_increase(
        &self,
        estimator: &dyn MemoryEstimator,
    ) -> Option<BudgetSuggestion> {
        let viable = self.constrained_space.prune_by_memory(estimator);

        if !viable.is_empty() {
            return None;
        }

        let min_shared = self.constrained_space.min_shared_mem(estimator);
        let min_global = self.constrained_space.min_global_mem(estimator);
        let budget = &self.constrained_space.budget;

        Some(BudgetSuggestion {
            min_shared_mem_per_block: min_shared,
            min_global_mem: min_global.saturating_add(budget.reserved_mem),
            current_shared_mem_per_block: budget.max_shared_mem_per_block,
            current_global_mem: budget.max_global_mem,
        })
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Fraction of the memory budget consumed by a configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetUtilization {
    /// Fraction of shared memory budget used (0.0–∞).
    pub shared_mem_fraction: f64,
    /// Fraction of global memory budget used (0.0–∞).
    pub global_mem_fraction: f64,
    /// Fraction of register budget used (0.0–∞).
    pub register_fraction: f64,
    /// Overall utilization — the maximum of all individual fractions.
    ///
    /// A value ≤ 1.0 means the config fits; > 1.0 means it exceeds
    /// the budget in at least one resource.
    pub overall: f64,
}

/// Suggestion for the minimum budget increase to admit at least one
/// configuration from the search space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetSuggestion {
    /// Minimum shared memory per block needed (bytes).
    pub min_shared_mem_per_block: u64,
    /// Minimum total global memory needed (bytes, including reserved).
    pub min_global_mem: u64,
    /// Current shared memory budget (for comparison).
    pub current_shared_mem_per_block: u64,
    /// Current global memory budget (for comparison).
    pub current_global_mem: u64,
}

impl BudgetSuggestion {
    /// Returns `true` if the shared memory budget needs to increase.
    #[must_use]
    pub fn needs_shared_increase(&self) -> bool {
        self.min_shared_mem_per_block > self.current_shared_mem_per_block
    }

    /// Returns `true` if the global memory budget needs to increase.
    #[must_use]
    pub fn needs_global_increase(&self) -> bool {
        self.min_global_mem > self.current_global_mem
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_space::SearchSpaceBuilder;

    fn small_space() -> SearchSpace {
        SearchSpaceBuilder::new()
            .tile_m(vec![64, 128])
            .tile_n(vec![64, 128])
            .tile_k(vec![16, 32])
            .warp_m(vec![32, 64])
            .warp_n(vec![32, 64])
            .stages(vec![2])
            .use_tensor_core(vec![false])
            .block_size(vec![128, 256])
            .build()
    }

    fn default_dims() -> ProblemDims {
        ProblemDims::new(1024, 1024, 1024, 1, 4)
    }

    fn generous_budget() -> MemoryBudget {
        MemoryBudget::new()
            .with_max_shared_mem_per_block(96 * 1024)
            .with_max_global_mem(16u64 * 1024 * 1024 * 1024)
            .with_max_registers_per_thread(255)
    }

    // -- MemoryBudget tests --

    #[test]
    fn budget_available_global_mem_subtracts_reserved() {
        let b = MemoryBudget::new()
            .with_max_global_mem(8 * 1024 * 1024 * 1024)
            .with_reserved_mem(1024 * 1024 * 1024);
        assert_eq!(b.available_global_mem(), 7 * 1024 * 1024 * 1024);
    }

    #[test]
    fn budget_reserved_exceeds_total_saturates_to_zero() {
        let b = MemoryBudget::new()
            .with_max_global_mem(100)
            .with_reserved_mem(200);
        assert_eq!(b.available_global_mem(), 0);
    }

    // -- MemoryEstimate tests --

    #[test]
    fn estimate_fits_budget_within_limits() {
        let est = MemoryEstimate {
            shared_mem_bytes: 32768,
            global_mem_bytes: 1024 * 1024,
            registers_per_thread: 64,
            workspace_bytes: 0,
        };
        let budget = generous_budget();
        assert!(est.fits_budget(&budget));
    }

    #[test]
    fn estimate_exceeds_shared_mem() {
        let est = MemoryEstimate {
            shared_mem_bytes: 200 * 1024,
            global_mem_bytes: 1024,
            registers_per_thread: 32,
            workspace_bytes: 0,
        };
        let budget = MemoryBudget::new().with_max_shared_mem_per_block(48 * 1024);
        assert!(!est.fits_budget(&budget));
    }

    #[test]
    fn estimate_total_global_includes_workspace() {
        let est = MemoryEstimate {
            shared_mem_bytes: 0,
            global_mem_bytes: 1000,
            registers_per_thread: 0,
            workspace_bytes: 500,
        };
        assert_eq!(est.total_global_bytes(), 1500);
    }

    // -- ProblemDims tests --

    #[test]
    fn gemm_flops_calculation() {
        let dims = ProblemDims::new(1024, 1024, 1024, 1, 4);
        let expected = 2.0 * 1024.0 * 1024.0 * 1024.0;
        assert!((dims.gemm_flops() - expected).abs() < 1.0);
    }

    #[test]
    fn gemm_flops_batched() {
        let dims = ProblemDims::new(256, 256, 256, 8, 4);
        let expected = 2.0 * 256.0 * 256.0 * 256.0 * 8.0;
        assert!((dims.gemm_flops() - expected).abs() < 1.0);
    }

    // -- GemmMemoryEstimator tests --

    #[test]
    fn gemm_estimator_shared_mem() {
        let cfg = Config::new()
            .with_tile_m(128)
            .with_tile_n(128)
            .with_tile_k(32)
            .with_stages(2);
        let dims = ProblemDims::new(1024, 1024, 1024, 1, 4);
        let est = GemmMemoryEstimator.estimate(&cfg, &dims);
        // (128*32 + 32*128) * 2 * 4 = 65536
        assert_eq!(est.shared_mem_bytes, 65536);
    }

    #[test]
    fn gemm_estimator_global_mem() {
        let cfg = Config::new();
        let dims = ProblemDims::new(1024, 1024, 512, 1, 4);
        let est = GemmMemoryEstimator.estimate(&cfg, &dims);
        // A=1024*512 + B=512*1024 + C=1024*1024 = 524288 + 524288 + 1048576 = 2097152
        // * 4 bytes = 8388608
        assert_eq!(est.global_mem_bytes, 8_388_608);
    }

    #[test]
    fn gemm_estimator_workspace_split_k() {
        let cfg = Config::new().with_extra("split_k", 4);
        let dims = ProblemDims::new(128, 128, 512, 1, 4);
        let est = GemmMemoryEstimator.estimate(&cfg, &dims);
        // workspace = 4 * 128 * 128 * 4 * 1 = 262144
        assert_eq!(est.workspace_bytes, 262_144);
    }

    #[test]
    fn gemm_estimator_no_split_k_zero_workspace() {
        let cfg = Config::new();
        let dims = default_dims();
        let est = GemmMemoryEstimator.estimate(&cfg, &dims);
        assert_eq!(est.workspace_bytes, 0);
    }

    // -- ConstrainedSearchSpace tests --

    #[test]
    fn prune_by_memory_returns_subset() {
        let space = small_space();
        let all_count = space.enumerate().len();
        let constrained = ConstrainedSearchSpace::new(space, generous_budget(), default_dims());
        let viable = constrained.prune_by_memory(&GemmMemoryEstimator);
        assert!(!viable.is_empty());
        assert!(viable.len() <= all_count);
    }

    #[test]
    fn prune_by_memory_tight_budget_filters_more() {
        let space = small_space();
        let generous =
            ConstrainedSearchSpace::new(space.clone(), generous_budget(), default_dims());
        let tight = ConstrainedSearchSpace::new(
            space,
            MemoryBudget::new()
                .with_max_shared_mem_per_block(16 * 1024)
                .with_max_global_mem(16u64 * 1024 * 1024 * 1024),
            default_dims(),
        );
        let generous_count = generous.prune_by_memory(&GemmMemoryEstimator).len();
        let tight_count = tight.prune_by_memory(&GemmMemoryEstimator).len();
        assert!(tight_count <= generous_count);
    }

    #[test]
    fn rank_by_efficiency_sorted_ascending() {
        let space = small_space();
        let constrained = ConstrainedSearchSpace::new(space, generous_budget(), default_dims());
        let ranked = constrained.rank_by_efficiency(&GemmMemoryEstimator);
        if ranked.len() >= 2 {
            for pair in ranked.windows(2) {
                let a_total = pair[0].1.total_global_bytes() + pair[0].1.shared_mem_bytes;
                let b_total = pair[1].1.total_global_bytes() + pair[1].1.shared_mem_bytes;
                assert!(a_total <= b_total);
            }
        }
    }

    // -- ConstrainedTuner tests --

    #[test]
    fn budget_utilization_within_budget() {
        let space = small_space();
        let constrained = ConstrainedSearchSpace::new(space, generous_budget(), default_dims());
        let tuner = ConstrainedTuner::new(constrained);
        let cfg = Config::new()
            .with_tile_m(64)
            .with_tile_n(64)
            .with_tile_k(16)
            .with_stages(2);
        let util = tuner.budget_utilization(&cfg, &GemmMemoryEstimator);
        assert!(util.overall <= 1.0, "should fit generous budget");
        assert!(util.shared_mem_fraction > 0.0);
        assert!(util.global_mem_fraction > 0.0);
    }

    #[test]
    fn suggest_budget_increase_returns_none_when_viable() {
        let space = small_space();
        let constrained = ConstrainedSearchSpace::new(space, generous_budget(), default_dims());
        let tuner = ConstrainedTuner::new(constrained);
        assert!(
            tuner
                .suggest_budget_increase(&GemmMemoryEstimator)
                .is_none()
        );
    }

    #[test]
    fn suggest_budget_increase_returns_suggestion_when_all_filtered() {
        let space = small_space();
        let tiny_budget = MemoryBudget::new()
            .with_max_shared_mem_per_block(64) // impossibly small
            .with_max_global_mem(1024);
        let constrained = ConstrainedSearchSpace::new(space, tiny_budget, default_dims());
        let tuner = ConstrainedTuner::new(constrained);
        let suggestion = tuner.suggest_budget_increase(&GemmMemoryEstimator);
        assert!(suggestion.is_some());
        let s = suggestion.expect("checked above");
        assert!(s.needs_shared_increase());
    }
}
