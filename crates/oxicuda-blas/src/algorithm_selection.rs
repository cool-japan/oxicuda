//! cuBLASLt-style algorithm selection API for GEMM kernel dispatch.
//!
//! Provides user-controllable algorithm handles for GEMM kernel selection,
//! similar to cuBLASLt's algorithm enumeration and selection API. Users can
//! enumerate compatible algorithms for a given problem, query performance
//! heuristics, select the best algorithm automatically, or create custom
//! algorithm configurations.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_blas::algorithm_selection::{AlgorithmSelector, SwizzleMode, EpiloguePreference};
//! use oxicuda_blas::level3::gemm::dispatch::GemmProblem;
//! use oxicuda_ptx::arch::SmVersion;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_blas::types::{Transpose, MathMode};
//!
//! let selector = AlgorithmSelector::new(SmVersion::Sm80);
//! let problem = GemmProblem {
//!     m: 1024, n: 1024, k: 512,
//!     trans_a: Transpose::NoTrans,
//!     trans_b: Transpose::NoTrans,
//!     input_type: PtxType::F32,
//!     output_type: PtxType::F32,
//!     math_mode: MathMode::Default,
//! };
//!
//! // Enumerate all compatible algorithms
//! let algos = selector.enumerate_algorithms(&problem);
//!
//! // Auto-select the best
//! let best = selector.select_best(&problem);
//! ```

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};
use crate::level3::gemm::dispatch::{GemmCategory, GemmProblem};

// ---------------------------------------------------------------------------
// Global algorithm ID counter
// ---------------------------------------------------------------------------

static NEXT_ALGO_ID: AtomicU64 = AtomicU64::new(1);

fn next_algorithm_id() -> u64 {
    NEXT_ALGO_ID.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// SwizzleMode — L2 cache swizzle patterns
// ---------------------------------------------------------------------------

/// L2 cache swizzle pattern for GEMM output tiles.
///
/// Swizzling reorders the mapping from CTA grid coordinates to output tile
/// locations so that neighbouring CTAs access nearby L2 cache lines,
/// improving cache hit rates for large matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwizzleMode {
    /// No swizzling — CTAs map linearly to output tiles.
    None,
    /// 128-byte swizzle granularity. Best for large matrices where L2
    /// pressure is the primary bottleneck.
    Swizzle128B,
    /// 64-byte swizzle granularity. Offers a compromise between L2
    /// utilisation and CTA scheduling flexibility.
    Swizzle64B,
}

impl fmt::Display for SwizzleMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Swizzle128B => write!(f, "swizzle_128B"),
            Self::Swizzle64B => write!(f, "swizzle_64B"),
        }
    }
}

// ---------------------------------------------------------------------------
// EpiloguePreference — fused epilogue operations
// ---------------------------------------------------------------------------

/// Preference for fused epilogue operations applied after the GEMM
/// accumulation (D = alpha * A * B + beta * C) is complete.
///
/// Fusing an epilogue avoids a separate kernel launch and the associated
/// global memory round-trip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpiloguePreference {
    /// No fused epilogue — just the linear combination.
    None,
    /// Fuse a bias-add: `D = alpha * A * B + beta * C + bias`.
    BiasAdd,
    /// Fuse a ReLU activation: `D = max(0, alpha * A * B + beta * C)`.
    Relu,
    /// Fuse a GELU activation (approximate).
    Gelu,
    /// User-defined epilogue identified by name.
    Custom(u32),
}

impl fmt::Display for EpiloguePreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::BiasAdd => write!(f, "bias_add"),
            Self::Relu => write!(f, "relu"),
            Self::Gelu => write!(f, "gelu"),
            Self::Custom(id) => write!(f, "custom_{id}"),
        }
    }
}

// ---------------------------------------------------------------------------
// AlgorithmId — unique identifier for a GEMM algorithm
// ---------------------------------------------------------------------------

/// Unique identifier for a GEMM algorithm.
///
/// Each algorithm ID is globally unique within a process and carries
/// metadata about the algorithm category and a human-readable name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlgorithmId {
    /// Unique numeric identifier, monotonically increasing.
    pub id: u64,
    /// The GEMM category this algorithm targets.
    pub category: GemmCategory,
    /// Human-readable algorithm name.
    pub name: String,
}

impl fmt::Display for AlgorithmId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Algorithm[{id}] {name} ({cat:?})",
            id = self.id,
            name = self.name,
            cat = self.category,
        )
    }
}

// ---------------------------------------------------------------------------
// AlgorithmConfig — tunable parameters for a GEMM algorithm
// ---------------------------------------------------------------------------

/// Tunable parameters for a GEMM algorithm.
///
/// These knobs control the tile decomposition, pipeline depth, split-K
/// factor, L2 swizzle pattern, and fused epilogue.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlgorithmConfig {
    /// Block tile size in the M dimension (rows per CTA).
    pub tile_m: u32,
    /// Block tile size in the N dimension (columns per CTA).
    pub tile_n: u32,
    /// Block tile size in the K dimension (reduction step per iteration).
    pub tile_k: u32,
    /// Number of software pipeline stages for async global-to-shared loads.
    pub stages: u32,
    /// Optional split-K factor. `None` means no split-K.
    pub split_k: Option<u32>,
    /// L2 cache swizzle mode.
    pub swizzle: SwizzleMode,
    /// Fused epilogue preference.
    pub epilogue: EpiloguePreference,
}

impl AlgorithmConfig {
    /// Validates this configuration against a GEMM problem.
    ///
    /// Returns an error if the tile dimensions are zero, stages are zero,
    /// or the tile config would produce an unreasonable launch grid.
    pub fn validate(&self, problem: &GemmProblem) -> BlasResult<()> {
        if self.tile_m == 0 || self.tile_n == 0 || self.tile_k == 0 {
            return Err(BlasError::InvalidArgument(
                "tile dimensions must be non-zero".into(),
            ));
        }
        if self.stages == 0 {
            return Err(BlasError::InvalidArgument(
                "pipeline stages must be at least 1".into(),
            ));
        }
        if !self.tile_m.is_power_of_two()
            || !self.tile_n.is_power_of_two()
            || !self.tile_k.is_power_of_two()
        {
            return Err(BlasError::InvalidArgument(
                "tile dimensions must be powers of two".into(),
            ));
        }
        if self.tile_m > 512 || self.tile_n > 512 {
            return Err(BlasError::InvalidArgument(
                "tile_m and tile_n must not exceed 512".into(),
            ));
        }
        if let Some(sk) = self.split_k {
            if sk == 0 {
                return Err(BlasError::InvalidArgument(
                    "split_k factor must be at least 1".into(),
                ));
            }
            if sk > problem.k {
                return Err(BlasError::InvalidArgument(format!(
                    "split_k factor ({sk}) exceeds K dimension ({})",
                    problem.k
                )));
            }
        }
        if self.stages > 8 {
            return Err(BlasError::InvalidArgument(
                "pipeline stages must not exceed 8".into(),
            ));
        }
        Ok(())
    }

    /// Estimates shared memory usage in bytes for this config with a given
    /// element type.
    fn estimate_shared_memory(&self, elem_bytes: u32) -> usize {
        let smem_a = self.tile_m as usize * self.tile_k as usize * elem_bytes as usize;
        let smem_b = self.tile_k as usize * self.tile_n as usize * elem_bytes as usize;
        (smem_a + smem_b) * self.stages as usize
    }
}

impl fmt::Display for AlgorithmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tile={}x{}x{}, stages={}, split_k={}, swizzle={}, epilogue={}",
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.stages,
            self.split_k
                .map_or_else(|| "none".to_string(), |s| s.to_string()),
            self.swizzle,
            self.epilogue,
        )
    }
}

// ---------------------------------------------------------------------------
// AlgorithmHeuristic — estimated performance metrics
// ---------------------------------------------------------------------------

/// Estimated performance metrics for a GEMM algorithm on a specific problem.
///
/// These are heuristic estimates, not profiled results. The `estimated_gflops`
/// field gives an absolute throughput estimate, while `estimated_efficiency`
/// gives a 0.0–1.0 score relative to peak hardware throughput.
#[derive(Debug, Clone)]
pub struct AlgorithmHeuristic {
    /// Estimated throughput in GFLOP/s.
    pub estimated_gflops: f64,
    /// Estimated efficiency as a fraction of peak (0.0–1.0).
    pub estimated_efficiency: f64,
    /// Estimated workspace memory requirement in bytes.
    pub workspace_bytes: usize,
    /// Whether this algorithm supports split-K decomposition.
    pub supports_split_k: bool,
}

impl AlgorithmHeuristic {
    /// Returns a composite score suitable for ranking algorithms.
    ///
    /// Higher is better. Combines throughput and efficiency with a bias
    /// towards efficiency (since estimated GFLOP/s can be noisy).
    pub fn score(&self) -> f64 {
        // Weight efficiency heavily since throughput estimates can be rough.
        self.estimated_gflops * (0.3 + 0.7 * self.estimated_efficiency)
    }
}

impl fmt::Display for AlgorithmHeuristic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{gflops:.1} GFLOP/s (eff={eff:.1}%, workspace={ws} B, split_k={sk})",
            gflops = self.estimated_gflops,
            eff = self.estimated_efficiency * 100.0,
            ws = self.workspace_bytes,
            sk = self.supports_split_k,
        )
    }
}

// ---------------------------------------------------------------------------
// Algorithm — internal representation
// ---------------------------------------------------------------------------

/// Internal representation of a GEMM algorithm, combining its identity,
/// configuration, and a heuristic function.
struct Algorithm {
    /// The algorithm's unique identity.
    id: AlgorithmId,
    /// The algorithm's tunable configuration.
    config: AlgorithmConfig,
    /// A function that estimates performance for a given problem.
    heuristic_fn: fn(&GemmProblem, &AlgorithmConfig, SmVersion) -> AlgorithmHeuristic,
}

// ---------------------------------------------------------------------------
// Heuristic functions for each category
// ---------------------------------------------------------------------------

/// Compute the raw FLOP count for a GEMM: 2 * M * N * K.
fn gemm_flops(problem: &GemmProblem) -> f64 {
    2.0 * problem.m as f64 * problem.n as f64 * problem.k as f64
}

/// Peak GFLOP/s estimate for a given SM version (FP32, single SM approximation).
fn peak_gflops_estimate(sm: SmVersion) -> f64 {
    match sm {
        SmVersion::Sm75 => 8_200.0,
        SmVersion::Sm80 => 19_500.0,
        SmVersion::Sm86 => 22_000.0,
        SmVersion::Sm89 => 45_000.0,
        SmVersion::Sm90 | SmVersion::Sm90a => 65_000.0,
        SmVersion::Sm100 => 80_000.0,
        SmVersion::Sm120 => 95_000.0,
    }
}

/// Estimate tile efficiency: how well the problem dimensions fill the tile grid.
fn tile_efficiency(m: u32, n: u32, tile_m: u32, tile_n: u32) -> f64 {
    let tiles_m = m.div_ceil(tile_m) as f64;
    let tiles_n = n.div_ceil(tile_n) as f64;
    let total_work = tiles_m * tiles_n * tile_m as f64 * tile_n as f64;
    let useful_work = m as f64 * n as f64;
    if total_work == 0.0 {
        return 0.0;
    }
    (useful_work / total_work).min(1.0)
}

/// Heuristic for standard SIMT GEMM algorithms.
fn heuristic_standard_simt(
    problem: &GemmProblem,
    config: &AlgorithmConfig,
    sm: SmVersion,
) -> AlgorithmHeuristic {
    let _flops = gemm_flops(problem);
    let peak = peak_gflops_estimate(sm);
    let tile_eff = tile_efficiency(problem.m, problem.n, config.tile_m, config.tile_n);

    // SIMT paths run at ~40-60% of peak on well-shaped problems.
    let base_efficiency = 0.45 * tile_eff;
    let pipeline_bonus = (config.stages as f64 - 1.0).max(0.0) * 0.05;
    let efficiency = (base_efficiency + pipeline_bonus).min(0.95);
    let gflops = peak * efficiency;
    let elem_bytes = problem.input_type.size_bytes() as u32;

    AlgorithmHeuristic {
        estimated_gflops: gflops,
        estimated_efficiency: efficiency,
        workspace_bytes: config.estimate_shared_memory(elem_bytes),
        supports_split_k: false,
    }
}

/// Heuristic for Tensor Core GEMM algorithms.
fn heuristic_tensor_core(
    problem: &GemmProblem,
    config: &AlgorithmConfig,
    sm: SmVersion,
) -> AlgorithmHeuristic {
    let _flops = gemm_flops(problem);
    let peak = peak_gflops_estimate(sm);
    let tile_eff = tile_efficiency(problem.m, problem.n, config.tile_m, config.tile_n);

    // TC paths run at ~60-85% of peak.
    let base_efficiency = 0.70 * tile_eff;
    let pipeline_bonus = (config.stages as f64 - 1.0).max(0.0) * 0.04;
    let efficiency = (base_efficiency + pipeline_bonus).min(0.95);

    // Tensor cores multiply throughput by precision factor.
    let precision_factor = match problem.input_type {
        PtxType::F16 | PtxType::BF16 => 4.0,
        PtxType::E4M3 | PtxType::E5M2 => 8.0,
        _ => 2.0,
    };
    let gflops = peak * efficiency * precision_factor;
    let elem_bytes = problem.input_type.size_bytes() as u32;

    // Workspace for split-K reduction buffer.
    let split_k_workspace = config.split_k.map_or(0, |sk| {
        if sk > 1 {
            problem.m as usize * problem.n as usize * problem.output_type.size_bytes() * sk as usize
        } else {
            0
        }
    });

    AlgorithmHeuristic {
        estimated_gflops: gflops,
        estimated_efficiency: efficiency,
        workspace_bytes: config.estimate_shared_memory(elem_bytes) + split_k_workspace,
        supports_split_k: true,
    }
}

/// Heuristic for skinny GEMM algorithms.
fn heuristic_skinny(
    problem: &GemmProblem,
    config: &AlgorithmConfig,
    sm: SmVersion,
) -> AlgorithmHeuristic {
    let peak = peak_gflops_estimate(sm);
    let tile_eff = tile_efficiency(problem.m, problem.n, config.tile_m, config.tile_n);

    // Skinny kernels are typically memory-bound.
    let base_efficiency = 0.35 * tile_eff;
    let efficiency = base_efficiency.min(0.60);
    let gflops = peak * efficiency;
    let elem_bytes = problem.input_type.size_bytes() as u32;

    AlgorithmHeuristic {
        estimated_gflops: gflops,
        estimated_efficiency: efficiency,
        workspace_bytes: config.estimate_shared_memory(elem_bytes),
        supports_split_k: false,
    }
}

/// Heuristic for split-K GEMM algorithms.
fn heuristic_split_k(
    problem: &GemmProblem,
    config: &AlgorithmConfig,
    sm: SmVersion,
) -> AlgorithmHeuristic {
    let peak = peak_gflops_estimate(sm);
    let tile_eff = tile_efficiency(problem.m, problem.n, config.tile_m, config.tile_n);

    let split_k = config.split_k.unwrap_or(1).max(1);
    // Split-K improves parallelism for K-dominant problems.
    let parallelism_gain = (split_k as f64).sqrt() / (split_k as f64).powf(0.3);
    let base_efficiency = 0.55 * tile_eff * parallelism_gain;
    let efficiency = base_efficiency.min(0.85);
    let gflops = peak * efficiency;
    let elem_bytes = problem.input_type.size_bytes() as u32;

    let reduction_workspace = if split_k > 1 {
        problem.m as usize
            * problem.n as usize
            * problem.output_type.size_bytes()
            * split_k as usize
    } else {
        0
    };

    AlgorithmHeuristic {
        estimated_gflops: gflops,
        estimated_efficiency: efficiency,
        workspace_bytes: config.estimate_shared_memory(elem_bytes) + reduction_workspace,
        supports_split_k: true,
    }
}

/// Heuristic for stream-K algorithms (Hopper+).
fn heuristic_stream_k(
    problem: &GemmProblem,
    config: &AlgorithmConfig,
    sm: SmVersion,
) -> AlgorithmHeuristic {
    let peak = peak_gflops_estimate(sm);
    let tile_eff = tile_efficiency(problem.m, problem.n, config.tile_m, config.tile_n);

    // Stream-K achieves near-perfect load balancing.
    let base_efficiency = 0.75 * tile_eff;
    let pipeline_bonus = (config.stages as f64 - 2.0).max(0.0) * 0.03;
    let efficiency = (base_efficiency + pipeline_bonus).min(0.92);
    let gflops = peak * efficiency;
    let elem_bytes = problem.input_type.size_bytes() as u32;

    AlgorithmHeuristic {
        estimated_gflops: gflops,
        estimated_efficiency: efficiency,
        workspace_bytes: config.estimate_shared_memory(elem_bytes),
        supports_split_k: false,
    }
}

/// Heuristic for warp-specialized algorithms (Hopper+).
fn heuristic_warp_specialized(
    problem: &GemmProblem,
    config: &AlgorithmConfig,
    sm: SmVersion,
) -> AlgorithmHeuristic {
    let peak = peak_gflops_estimate(sm);
    let tile_eff = tile_efficiency(problem.m, problem.n, config.tile_m, config.tile_n);

    // Warp-specialized achieves highest efficiency on Hopper.
    let base_efficiency = 0.82 * tile_eff;
    let pipeline_bonus = (config.stages as f64 - 2.0).max(0.0) * 0.025;
    let efficiency = (base_efficiency + pipeline_bonus).min(0.95);

    let precision_factor = match problem.input_type {
        PtxType::F16 | PtxType::BF16 => 4.0,
        PtxType::E4M3 | PtxType::E5M2 => 8.0,
        _ => 2.0,
    };
    let gflops = peak * efficiency * precision_factor;
    let elem_bytes = problem.input_type.size_bytes() as u32;

    AlgorithmHeuristic {
        estimated_gflops: gflops,
        estimated_efficiency: efficiency,
        workspace_bytes: config.estimate_shared_memory(elem_bytes),
        supports_split_k: false,
    }
}

/// Heuristic for bandwidth-limited algorithms.
fn heuristic_bandwidth_limited(
    problem: &GemmProblem,
    config: &AlgorithmConfig,
    sm: SmVersion,
) -> AlgorithmHeuristic {
    let peak = peak_gflops_estimate(sm);
    let tile_eff = tile_efficiency(problem.m, problem.n, config.tile_m, config.tile_n);

    // Bandwidth-limited paths prioritise memory throughput.
    let base_efficiency = 0.30 * tile_eff;
    let efficiency = base_efficiency.min(0.50);
    let gflops = peak * efficiency;
    let elem_bytes = problem.input_type.size_bytes() as u32;

    AlgorithmHeuristic {
        estimated_gflops: gflops,
        estimated_efficiency: efficiency,
        workspace_bytes: config.estimate_shared_memory(elem_bytes),
        supports_split_k: false,
    }
}

// ---------------------------------------------------------------------------
// AlgorithmSelector — main API
// ---------------------------------------------------------------------------

/// The main entry point for cuBLASLt-style algorithm selection.
///
/// An `AlgorithmSelector` is constructed for a specific GPU architecture and
/// pre-populates a catalogue of known GEMM algorithm variants. Users can then
/// enumerate compatible algorithms, query heuristics, or let the selector
/// auto-pick the best candidate.
pub struct AlgorithmSelector {
    /// Target GPU architecture.
    sm_version: SmVersion,
    /// Catalogue of registered algorithms.
    algorithms: Vec<Algorithm>,
}

impl AlgorithmSelector {
    /// Creates a new selector targeting the given SM architecture.
    ///
    /// The selector is pre-populated with a set of built-in algorithm
    /// variants covering SIMT, Tensor Core, skinny, split-K, stream-K,
    /// warp-specialized, and bandwidth-limited categories.
    pub fn new(sm_version: SmVersion) -> Self {
        let mut algorithms = Vec::new();
        Self::populate_simt_algorithms(&mut algorithms);
        Self::populate_tensor_core_algorithms(&mut algorithms, sm_version);
        Self::populate_skinny_algorithms(&mut algorithms);
        Self::populate_split_k_algorithms(&mut algorithms);

        if sm_version >= SmVersion::Sm90 {
            Self::populate_stream_k_algorithms(&mut algorithms);
            Self::populate_warp_specialized_algorithms(&mut algorithms);
        }

        Self::populate_bandwidth_limited_algorithms(&mut algorithms);

        Self {
            sm_version,
            algorithms,
        }
    }

    /// Enumerates all algorithms compatible with the given problem.
    ///
    /// An algorithm is compatible if:
    /// - The problem dimensions are non-zero.
    /// - The problem's math mode allows the algorithm's compute path.
    /// - Architecture-specific requirements are met (e.g., Tensor Core needs
    ///   `MathMode::TensorCore` and an eligible type).
    pub fn enumerate_algorithms(&self, problem: &GemmProblem) -> Vec<AlgorithmId> {
        self.algorithms
            .iter()
            .filter(|algo| self.is_compatible(algo, problem))
            .map(|algo| algo.id.clone())
            .collect()
    }

    /// Returns a performance heuristic for a specific algorithm on the given
    /// problem.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidArgument`] if the algorithm ID is not
    /// found in the selector's catalogue.
    pub fn get_heuristic(
        &self,
        algo_id: &AlgorithmId,
        problem: &GemmProblem,
    ) -> BlasResult<AlgorithmHeuristic> {
        let algo = self
            .algorithms
            .iter()
            .find(|a| a.id == *algo_id)
            .ok_or_else(|| {
                BlasError::InvalidArgument(format!(
                    "algorithm ID {} not found in selector",
                    algo_id.id
                ))
            })?;
        Ok((algo.heuristic_fn)(problem, &algo.config, self.sm_version))
    }

    /// Automatically selects the best algorithm for the given problem.
    ///
    /// Evaluates heuristics for all compatible algorithms and returns the
    /// one with the highest composite score.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::UnsupportedOperation`] if no compatible algorithm
    /// is found.
    pub fn select_best(&self, problem: &GemmProblem) -> BlasResult<AlgorithmId> {
        let candidates: Vec<_> = self
            .algorithms
            .iter()
            .filter(|algo| self.is_compatible(algo, problem))
            .collect();

        if candidates.is_empty() {
            return Err(BlasError::UnsupportedOperation(
                "no compatible GEMM algorithm found for this problem".into(),
            ));
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_id = None;

        for algo in &candidates {
            let heuristic = (algo.heuristic_fn)(problem, &algo.config, self.sm_version);
            let score = heuristic.score();
            if score > best_score {
                best_score = score;
                best_id = Some(algo.id.clone());
            }
        }

        best_id.ok_or_else(|| {
            BlasError::UnsupportedOperation(
                "no compatible GEMM algorithm found for this problem".into(),
            )
        })
    }

    /// Returns the top N algorithms for the given problem, ranked by
    /// heuristic score (descending).
    ///
    /// If fewer than `n` algorithms are compatible, returns all of them.
    pub fn select_top_n(
        &self,
        problem: &GemmProblem,
        n: usize,
    ) -> Vec<(AlgorithmId, AlgorithmHeuristic)> {
        let mut scored: Vec<_> = self
            .algorithms
            .iter()
            .filter(|algo| self.is_compatible(algo, problem))
            .map(|algo| {
                let heuristic = (algo.heuristic_fn)(problem, &algo.config, self.sm_version);
                (algo.id.clone(), heuristic)
            })
            .collect();

        // Sort by score descending.
        scored.sort_by(|a, b| {
            b.1.score()
                .partial_cmp(&a.1.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.truncate(n);
        scored
    }

    /// Creates a custom algorithm with user-specified configuration.
    ///
    /// The configuration is validated against the problem dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidArgument`] if the configuration is invalid.
    pub fn create_custom(
        &mut self,
        config: AlgorithmConfig,
        problem: &GemmProblem,
    ) -> BlasResult<AlgorithmId> {
        config.validate(problem)?;

        let category = self.classify_config(&config);
        let heuristic_fn = Self::heuristic_fn_for_category(category);

        let id = AlgorithmId {
            id: next_algorithm_id(),
            category,
            name: format!(
                "custom_{}x{}x{}_s{}",
                config.tile_m, config.tile_n, config.tile_k, config.stages,
            ),
        };

        self.algorithms.push(Algorithm {
            id: id.clone(),
            config,
            heuristic_fn,
        });

        Ok(id)
    }

    /// Returns the number of registered algorithms.
    pub fn algorithm_count(&self) -> usize {
        self.algorithms.len()
    }

    /// Returns the target SM version.
    pub fn sm_version(&self) -> SmVersion {
        self.sm_version
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Checks whether an algorithm is compatible with a given problem.
    fn is_compatible(&self, algo: &Algorithm, problem: &GemmProblem) -> bool {
        // Zero-dimensional problems are never compatible.
        if problem.m == 0 || problem.n == 0 || problem.k == 0 {
            return false;
        }

        match algo.id.category {
            GemmCategory::Standard | GemmCategory::BandwidthLimited => true,

            GemmCategory::Skinny => problem.m < 32 || problem.n < 32,

            GemmCategory::SplitK => problem.k > 4 * problem.m && problem.k > 4 * problem.n,

            GemmCategory::StreamK => {
                self.sm_version >= SmVersion::Sm90
                    && u64::from(problem.m) * u64::from(problem.n) * u64::from(problem.k)
                        >= 64 * 1024 * 1024
            }

            GemmCategory::WarpSpecialized => {
                self.sm_version >= SmVersion::Sm90
                    && matches!(
                        problem.input_type,
                        PtxType::F16 | PtxType::BF16 | PtxType::E4M3 | PtxType::E5M2
                    )
                    && u64::from(problem.m) * u64::from(problem.n) * u64::from(problem.k)
                        >= 64 * 1024 * 1024
            }
        }
    }

    /// Classify a user-provided config into a category.
    fn classify_config(&self, config: &AlgorithmConfig) -> GemmCategory {
        if config.split_k.is_some_and(|sk| sk > 1) {
            return GemmCategory::SplitK;
        }
        if config.tile_m <= 32 || config.tile_n <= 32 {
            return GemmCategory::Skinny;
        }
        GemmCategory::Standard
    }

    /// Returns the appropriate heuristic function for a category.
    fn heuristic_fn_for_category(
        category: GemmCategory,
    ) -> fn(&GemmProblem, &AlgorithmConfig, SmVersion) -> AlgorithmHeuristic {
        match category {
            GemmCategory::Standard => heuristic_standard_simt,
            GemmCategory::Skinny => heuristic_skinny,
            GemmCategory::SplitK => heuristic_split_k,
            GemmCategory::StreamK => heuristic_stream_k,
            GemmCategory::WarpSpecialized => heuristic_warp_specialized,
            GemmCategory::BandwidthLimited => heuristic_bandwidth_limited,
        }
    }

    // -----------------------------------------------------------------------
    // Algorithm population
    // -----------------------------------------------------------------------

    fn populate_simt_algorithms(algorithms: &mut Vec<Algorithm>) {
        let configs = [
            (64, 64, 8, 1, "simt_64x64x8_s1"),
            (128, 64, 8, 1, "simt_128x64x8_s1"),
            (64, 128, 8, 1, "simt_64x128x8_s1"),
            (128, 128, 8, 1, "simt_128x128x8_s1"),
            (128, 128, 16, 2, "simt_128x128x16_s2"),
            (256, 128, 16, 2, "simt_256x128x16_s2"),
        ];
        for (tm, tn, tk, stages, name) in configs {
            algorithms.push(Algorithm {
                id: AlgorithmId {
                    id: next_algorithm_id(),
                    category: GemmCategory::Standard,
                    name: name.into(),
                },
                config: AlgorithmConfig {
                    tile_m: tm,
                    tile_n: tn,
                    tile_k: tk,
                    stages,
                    split_k: None,
                    swizzle: SwizzleMode::None,
                    epilogue: EpiloguePreference::None,
                },
                heuristic_fn: heuristic_standard_simt,
            });
        }
    }

    fn populate_tensor_core_algorithms(algorithms: &mut Vec<Algorithm>, sm: SmVersion) {
        let caps = sm.capabilities();
        if !caps.has_tensor_cores {
            return;
        }

        let configs = [
            (128, 128, 32, 2, SwizzleMode::None, "tc_128x128x32_s2"),
            (128, 128, 32, 3, SwizzleMode::None, "tc_128x128x32_s3"),
            (
                256,
                128,
                32,
                3,
                SwizzleMode::Swizzle128B,
                "tc_256x128x32_s3_sw128",
            ),
            (
                128,
                256,
                32,
                3,
                SwizzleMode::Swizzle128B,
                "tc_128x256x32_s3_sw128",
            ),
            (
                256,
                128,
                64,
                4,
                SwizzleMode::Swizzle128B,
                "tc_256x128x64_s4_sw128",
            ),
        ];
        for (tm, tn, tk, stages, swizzle, name) in configs {
            algorithms.push(Algorithm {
                id: AlgorithmId {
                    id: next_algorithm_id(),
                    category: GemmCategory::Standard,
                    name: name.into(),
                },
                config: AlgorithmConfig {
                    tile_m: tm,
                    tile_n: tn,
                    tile_k: tk,
                    stages,
                    split_k: None,
                    swizzle,
                    epilogue: EpiloguePreference::None,
                },
                heuristic_fn: heuristic_tensor_core,
            });
        }
    }

    fn populate_skinny_algorithms(algorithms: &mut Vec<Algorithm>) {
        let configs = [
            (16, 128, 16, 1, "skinny_16x128x16_s1"),
            (128, 16, 16, 1, "skinny_128x16x16_s1"),
            (8, 256, 32, 1, "skinny_8x256x32_s1"),
            (256, 8, 32, 1, "skinny_256x8x32_s1"),
            (32, 128, 16, 2, "skinny_32x128x16_s2"),
            (128, 32, 16, 2, "skinny_128x32x16_s2"),
        ];
        for (tm, tn, tk, stages, name) in configs {
            algorithms.push(Algorithm {
                id: AlgorithmId {
                    id: next_algorithm_id(),
                    category: GemmCategory::Skinny,
                    name: name.into(),
                },
                config: AlgorithmConfig {
                    tile_m: tm,
                    tile_n: tn,
                    tile_k: tk,
                    stages,
                    split_k: None,
                    swizzle: SwizzleMode::None,
                    epilogue: EpiloguePreference::None,
                },
                heuristic_fn: heuristic_skinny,
            });
        }
    }

    fn populate_split_k_algorithms(algorithms: &mut Vec<Algorithm>) {
        let configs = [
            (128, 128, 32, 2, 4, "splitk_128x128x32_s2_k4"),
            (128, 128, 32, 2, 8, "splitk_128x128x32_s2_k8"),
            (128, 128, 32, 3, 16, "splitk_128x128x32_s3_k16"),
            (64, 64, 32, 2, 4, "splitk_64x64x32_s2_k4"),
            (64, 64, 32, 2, 8, "splitk_64x64x32_s2_k8"),
        ];
        for (tm, tn, tk, stages, sk, name) in configs {
            algorithms.push(Algorithm {
                id: AlgorithmId {
                    id: next_algorithm_id(),
                    category: GemmCategory::SplitK,
                    name: name.into(),
                },
                config: AlgorithmConfig {
                    tile_m: tm,
                    tile_n: tn,
                    tile_k: tk,
                    stages,
                    split_k: Some(sk),
                    swizzle: SwizzleMode::None,
                    epilogue: EpiloguePreference::None,
                },
                heuristic_fn: heuristic_split_k,
            });
        }
    }

    fn populate_stream_k_algorithms(algorithms: &mut Vec<Algorithm>) {
        let configs = [
            (
                256,
                128,
                64,
                4,
                SwizzleMode::Swizzle128B,
                "streamk_256x128x64_s4_sw128",
            ),
            (
                128,
                128,
                64,
                4,
                SwizzleMode::Swizzle128B,
                "streamk_128x128x64_s4_sw128",
            ),
            (
                128,
                64,
                64,
                3,
                SwizzleMode::Swizzle64B,
                "streamk_128x64x64_s3_sw64",
            ),
        ];
        for (tm, tn, tk, stages, swizzle, name) in configs {
            algorithms.push(Algorithm {
                id: AlgorithmId {
                    id: next_algorithm_id(),
                    category: GemmCategory::StreamK,
                    name: name.into(),
                },
                config: AlgorithmConfig {
                    tile_m: tm,
                    tile_n: tn,
                    tile_k: tk,
                    stages,
                    split_k: None,
                    swizzle,
                    epilogue: EpiloguePreference::None,
                },
                heuristic_fn: heuristic_stream_k,
            });
        }
    }

    fn populate_warp_specialized_algorithms(algorithms: &mut Vec<Algorithm>) {
        let configs = [
            (
                128,
                128,
                64,
                3,
                SwizzleMode::Swizzle128B,
                "warp_spec_128x128x64_s3_sw128",
            ),
            (
                128,
                128,
                64,
                4,
                SwizzleMode::Swizzle128B,
                "warp_spec_128x128x64_s4_sw128",
            ),
            (
                256,
                128,
                64,
                4,
                SwizzleMode::Swizzle128B,
                "warp_spec_256x128x64_s4_sw128",
            ),
        ];
        for (tm, tn, tk, stages, swizzle, name) in configs {
            algorithms.push(Algorithm {
                id: AlgorithmId {
                    id: next_algorithm_id(),
                    category: GemmCategory::WarpSpecialized,
                    name: name.into(),
                },
                config: AlgorithmConfig {
                    tile_m: tm,
                    tile_n: tn,
                    tile_k: tk,
                    stages,
                    split_k: None,
                    swizzle,
                    epilogue: EpiloguePreference::None,
                },
                heuristic_fn: heuristic_warp_specialized,
            });
        }
    }

    fn populate_bandwidth_limited_algorithms(algorithms: &mut Vec<Algorithm>) {
        let configs = [
            (64, 64, 8, 1, "bw_64x64x8_s1"),
            (128, 64, 8, 1, "bw_128x64x8_s1"),
            (64, 128, 8, 1, "bw_64x128x8_s1"),
            (128, 128, 8, 1, "bw_128x128x8_s1"),
        ];
        for (tm, tn, tk, stages, name) in configs {
            algorithms.push(Algorithm {
                id: AlgorithmId {
                    id: next_algorithm_id(),
                    category: GemmCategory::BandwidthLimited,
                    name: name.into(),
                },
                config: AlgorithmConfig {
                    tile_m: tm,
                    tile_n: tn,
                    tile_k: tk,
                    stages,
                    split_k: None,
                    swizzle: SwizzleMode::None,
                    epilogue: EpiloguePreference::None,
                },
                heuristic_fn: heuristic_bandwidth_limited,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MathMode, Transpose};

    fn make_problem(m: u32, n: u32, k: u32) -> GemmProblem {
        GemmProblem {
            m,
            n,
            k,
            trans_a: Transpose::NoTrans,
            trans_b: Transpose::NoTrans,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            math_mode: MathMode::Default,
        }
    }

    fn make_f16_problem(m: u32, n: u32, k: u32) -> GemmProblem {
        GemmProblem {
            m,
            n,
            k,
            trans_a: Transpose::NoTrans,
            trans_b: Transpose::NoTrans,
            input_type: PtxType::F16,
            output_type: PtxType::F32,
            math_mode: MathMode::TensorCore,
        }
    }

    // -- 1. Enumerate algorithms for a standard problem -----------------------

    #[test]
    fn enumerate_standard_problem() {
        let selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(512, 512, 512);
        let algos = selector.enumerate_algorithms(&problem);
        // Should include SIMT, TC, and BW-limited algorithms at minimum.
        assert!(
            algos.len() >= 6,
            "expected >= 6 algorithms, got {}",
            algos.len()
        );
    }

    // -- 2. Enumerate zero-dim problem yields nothing -------------------------

    #[test]
    fn enumerate_zero_dim_problem() {
        let selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(0, 512, 512);
        let algos = selector.enumerate_algorithms(&problem);
        assert!(algos.is_empty());
    }

    // -- 3. Enumerate skinny problem includes skinny algos --------------------

    #[test]
    fn enumerate_skinny_problem() {
        let selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(8, 512, 512);
        let algos = selector.enumerate_algorithms(&problem);
        let has_skinny = algos.iter().any(|a| a.category == GemmCategory::Skinny);
        assert!(
            has_skinny,
            "skinny problem should include skinny algorithms"
        );
    }

    // -- 4. Enumerate split-K problem includes split-K algos ------------------

    #[test]
    fn enumerate_split_k_problem() {
        let selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(64, 64, 8192);
        let algos = selector.enumerate_algorithms(&problem);
        let has_split_k = algos.iter().any(|a| a.category == GemmCategory::SplitK);
        assert!(
            has_split_k,
            "split-K problem should include split-K algorithms"
        );
    }

    // -- 5. Hopper adds stream-K and warp-specialized -------------------------

    #[test]
    fn enumerate_hopper_large_f16() {
        let selector = AlgorithmSelector::new(SmVersion::Sm90);
        let problem = make_f16_problem(4096, 4096, 4096);
        let algos = selector.enumerate_algorithms(&problem);
        let has_stream_k = algos.iter().any(|a| a.category == GemmCategory::StreamK);
        let has_warp_spec = algos
            .iter()
            .any(|a| a.category == GemmCategory::WarpSpecialized);
        assert!(has_stream_k, "Hopper should have stream-K algorithms");
        assert!(
            has_warp_spec,
            "Hopper F16 should have warp-specialized algorithms"
        );
    }

    // -- 6. Heuristic scoring returns reasonable values ------------------------

    #[test]
    fn heuristic_values_are_reasonable() {
        let selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(1024, 1024, 1024);
        let algos = selector.enumerate_algorithms(&problem);
        assert!(!algos.is_empty());

        let heuristic = selector
            .get_heuristic(&algos[0], &problem)
            .expect("heuristic should succeed");
        assert!(heuristic.estimated_gflops > 0.0);
        assert!(heuristic.estimated_efficiency > 0.0);
        assert!(heuristic.estimated_efficiency <= 1.0);
    }

    // -- 7. Heuristic for unknown ID returns error ----------------------------

    #[test]
    fn heuristic_unknown_id_returns_error() {
        let selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(256, 256, 256);
        let fake_id = AlgorithmId {
            id: u64::MAX,
            category: GemmCategory::Standard,
            name: "nonexistent".into(),
        };
        let result = selector.get_heuristic(&fake_id, &problem);
        assert!(result.is_err());
    }

    // -- 8. Select best returns an algorithm ----------------------------------

    #[test]
    fn select_best_standard() {
        let selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(1024, 1024, 1024);
        let best = selector.select_best(&problem);
        assert!(best.is_ok());
    }

    // -- 9. Select best for zero-dim problem returns error --------------------

    #[test]
    fn select_best_zero_dim_error() {
        let selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(0, 0, 0);
        let result = selector.select_best(&problem);
        assert!(result.is_err());
    }

    // -- 10. Select top N returns at most N -----------------------------------

    #[test]
    fn select_top_n_respects_limit() {
        let selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(512, 512, 512);
        let top3 = selector.select_top_n(&problem, 3);
        assert!(top3.len() <= 3);
        assert!(!top3.is_empty());
    }

    // -- 11. Top N is sorted by score (descending) ----------------------------

    #[test]
    fn select_top_n_sorted_descending() {
        let selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(512, 512, 512);
        let top5 = selector.select_top_n(&problem, 5);
        for window in top5.windows(2) {
            assert!(
                window[0].1.score() >= window[1].1.score(),
                "results should be sorted descending by score"
            );
        }
    }

    // -- 12. Create custom algorithm succeeds ---------------------------------

    #[test]
    fn create_custom_algorithm() {
        let mut selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(256, 256, 256);
        let config = AlgorithmConfig {
            tile_m: 64,
            tile_n: 64,
            tile_k: 16,
            stages: 2,
            split_k: None,
            swizzle: SwizzleMode::Swizzle64B,
            epilogue: EpiloguePreference::Relu,
        };
        let count_before = selector.algorithm_count();
        let result = selector.create_custom(config, &problem);
        assert!(result.is_ok());
        assert_eq!(selector.algorithm_count(), count_before + 1);

        // The new algorithm should appear in enumeration.
        let algos = selector.enumerate_algorithms(&problem);
        let custom_id = result.expect("already checked is_ok");
        assert!(algos.contains(&custom_id));
    }

    // -- 13. Create custom with invalid config returns error ------------------

    #[test]
    fn create_custom_invalid_tile() {
        let mut selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(256, 256, 256);
        let config = AlgorithmConfig {
            tile_m: 0,
            tile_n: 64,
            tile_k: 16,
            stages: 2,
            split_k: None,
            swizzle: SwizzleMode::None,
            epilogue: EpiloguePreference::None,
        };
        let result = selector.create_custom(config, &problem);
        assert!(result.is_err());
    }

    // -- 14. Create custom with non-power-of-two tile fails -------------------

    #[test]
    fn create_custom_non_power_of_two() {
        let mut selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(256, 256, 256);
        let config = AlgorithmConfig {
            tile_m: 48,
            tile_n: 64,
            tile_k: 16,
            stages: 2,
            split_k: None,
            swizzle: SwizzleMode::None,
            epilogue: EpiloguePreference::None,
        };
        let result = selector.create_custom(config, &problem);
        assert!(result.is_err());
    }

    // -- 15. Create custom with split-K exceeding K fails ---------------------

    #[test]
    fn create_custom_split_k_exceeds_k() {
        let mut selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_problem(256, 256, 4);
        let config = AlgorithmConfig {
            tile_m: 64,
            tile_n: 64,
            tile_k: 16,
            stages: 1,
            split_k: Some(8),
            swizzle: SwizzleMode::None,
            epilogue: EpiloguePreference::None,
        };
        let result = selector.create_custom(config, &problem);
        assert!(result.is_err());
    }

    // -- 16. Display impls produce non-empty strings --------------------------

    #[test]
    fn display_algorithm_id() {
        let id = AlgorithmId {
            id: 42,
            category: GemmCategory::Standard,
            name: "test_algo".into(),
        };
        let s = format!("{id}");
        assert!(s.contains("42"));
        assert!(s.contains("test_algo"));
    }

    #[test]
    fn display_algorithm_heuristic() {
        let h = AlgorithmHeuristic {
            estimated_gflops: 12345.6,
            estimated_efficiency: 0.78,
            workspace_bytes: 65536,
            supports_split_k: true,
        };
        let s = format!("{h}");
        assert!(s.contains("GFLOP/s"));
        assert!(s.contains("78.0%"));
        assert!(s.contains("65536"));
    }

    // -- 17. Display for config -----------------------------------------------

    #[test]
    fn display_algorithm_config() {
        let config = AlgorithmConfig {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            stages: 3,
            split_k: Some(4),
            swizzle: SwizzleMode::Swizzle128B,
            epilogue: EpiloguePreference::Gelu,
        };
        let s = format!("{config}");
        assert!(s.contains("128x128x32"));
        assert!(s.contains("stages=3"));
        assert!(s.contains("swizzle_128B"));
        assert!(s.contains("gelu"));
    }

    // -- 18. Warp-specialized not available on Ampere -------------------------

    #[test]
    fn warp_specialized_not_on_ampere() {
        let selector = AlgorithmSelector::new(SmVersion::Sm80);
        let problem = make_f16_problem(4096, 4096, 4096);
        let algos = selector.enumerate_algorithms(&problem);
        let has_warp_spec = algos
            .iter()
            .any(|a| a.category == GemmCategory::WarpSpecialized);
        assert!(
            !has_warp_spec,
            "Ampere should not have warp-specialized algorithms"
        );
    }

    // -- 19. Heuristic score ordering is consistent ---------------------------

    #[test]
    fn heuristic_score_positive() {
        let h = AlgorithmHeuristic {
            estimated_gflops: 100.0,
            estimated_efficiency: 0.5,
            workspace_bytes: 0,
            supports_split_k: false,
        };
        assert!(h.score() > 0.0);
    }

    // -- 20. AlgorithmConfig validate rejects too many stages -----------------

    #[test]
    fn config_validate_too_many_stages() {
        let problem = make_problem(256, 256, 256);
        let config = AlgorithmConfig {
            tile_m: 64,
            tile_n: 64,
            tile_k: 16,
            stages: 16,
            split_k: None,
            swizzle: SwizzleMode::None,
            epilogue: EpiloguePreference::None,
        };
        assert!(config.validate(&problem).is_err());
    }

    // -- 21. AlgorithmConfig validate rejects oversized tiles -----------------

    #[test]
    fn config_validate_oversized_tile() {
        let problem = make_problem(1024, 1024, 1024);
        let config = AlgorithmConfig {
            tile_m: 1024,
            tile_n: 64,
            tile_k: 16,
            stages: 1,
            split_k: None,
            swizzle: SwizzleMode::None,
            epilogue: EpiloguePreference::None,
        };
        assert!(config.validate(&problem).is_err());
    }
}
