//! Multi-objective optimization with Pareto front analysis.
//!
//! This module provides tools for optimizing GPU kernel configurations
//! across multiple competing objectives simultaneously (e.g. throughput
//! vs latency vs power consumption). Instead of reducing everything to
//! a single score, it maintains the **Pareto front** — the set of
//! non-dominated solutions where no single objective can be improved
//! without degrading another.
//!
//! # Key concepts
//!
//! - **Pareto dominance**: Solution A *dominates* B if A is at least as
//!   good on every objective and strictly better on at least one.
//! - **Pareto front**: The set of all non-dominated solutions.
//! - **Crowding distance** (NSGA-II): A diversity metric that favours
//!   solutions in sparse regions of the front.
//! - **Knee point**: The solution offering the best overall tradeoff,
//!   defined as the point with maximum perpendicular distance from
//!   the line connecting the extremes of the front.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::multi_objective::*;
//! use oxicuda_autotune::{BenchmarkResult, Config};
//!
//! // Define objectives
//! let specs = vec![
//!     ObjectiveSpec::new(Objective::Throughput, ObjectiveDirection::Maximize),
//!     ObjectiveSpec::new(Objective::Latency, ObjectiveDirection::Minimize),
//! ];
//!
//! let mut front = ParetoFront::new(specs);
//!
//! let result = MultiObjectiveResult {
//!     benchmark: BenchmarkResult {
//!         config: Config::new(),
//!         median_us: 100.0,
//!         min_us: 90.0,
//!         max_us: 110.0,
//!         stddev_us: 5.0,
//!         gflops: Some(500.0),
//!         efficiency: Some(0.8),
//!     },
//!     scores: vec![500.0, 100.0], // throughput=500, latency=100
//! };
//!
//! let was_inserted = front.insert(result);
//! assert!(was_inserted);
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::benchmark::BenchmarkResult;
use crate::config::Config;
use crate::error::{AutotuneError, AutotuneResult};

// ---------------------------------------------------------------------------
// Objective direction
// ---------------------------------------------------------------------------

/// Whether an objective should be minimized or maximized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectiveDirection {
    /// Lower values are better (e.g. latency, power).
    Minimize,
    /// Higher values are better (e.g. throughput, GFLOPS).
    Maximize,
}

impl ObjectiveDirection {
    /// Returns `true` if `a` is strictly better than `b` under this direction.
    #[must_use]
    pub fn is_better(self, a: f64, b: f64) -> bool {
        match self {
            Self::Minimize => a < b,
            Self::Maximize => a > b,
        }
    }

    /// Returns `true` if `a` is at least as good as `b` under this direction.
    #[must_use]
    pub fn is_at_least_as_good(self, a: f64, b: f64) -> bool {
        match self {
            Self::Minimize => a <= b,
            Self::Maximize => a >= b,
        }
    }
}

impl fmt::Display for ObjectiveDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minimize => write!(f, "minimize"),
            Self::Maximize => write!(f, "maximize"),
        }
    }
}

// ---------------------------------------------------------------------------
// Objective
// ---------------------------------------------------------------------------

/// An optimization objective for GPU kernel tuning.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Objective {
    /// Computational throughput (GFLOPS, GB/s, etc.).
    Throughput,
    /// Execution latency (time per kernel invocation).
    Latency,
    /// Power consumption (watts).
    Power,
    /// Memory usage (bytes).
    Memory,
    /// A user-defined objective with a descriptive name.
    Custom(String),
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Throughput => write!(f, "throughput"),
            Self::Latency => write!(f, "latency"),
            Self::Power => write!(f, "power"),
            Self::Memory => write!(f, "memory"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ObjectiveSpec
// ---------------------------------------------------------------------------

/// Pairs an [`Objective`] with its optimization [`ObjectiveDirection`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    /// The objective being optimized.
    pub objective: Objective,
    /// Whether to minimize or maximize this objective.
    pub direction: ObjectiveDirection,
}

impl ObjectiveSpec {
    /// Creates a new objective specification.
    #[must_use]
    pub fn new(objective: Objective, direction: ObjectiveDirection) -> Self {
        Self {
            objective,
            direction,
        }
    }
}

impl fmt::Display for ObjectiveSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.objective, self.direction)
    }
}

// ---------------------------------------------------------------------------
// ObjectiveValue
// ---------------------------------------------------------------------------

/// A scored value for a single objective, aware of its optimization direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveValue {
    /// The objective this value corresponds to.
    pub objective: Objective,
    /// The raw numeric score.
    pub value: f64,
    /// The optimization direction.
    pub direction: ObjectiveDirection,
}

impl ObjectiveValue {
    /// Creates a new objective value.
    #[must_use]
    pub fn new(objective: Objective, value: f64, direction: ObjectiveDirection) -> Self {
        Self {
            objective,
            value,
            direction,
        }
    }

    /// Returns `true` if this value is strictly better than `other`.
    ///
    /// Respects the optimization direction: for [`ObjectiveDirection::Minimize`],
    /// a lower value is better; for [`ObjectiveDirection::Maximize`], higher
    /// is better.
    #[must_use]
    pub fn is_better_than(&self, other: &Self) -> bool {
        self.direction.is_better(self.value, other.value)
    }

    /// Returns `true` if this value is at least as good as `other`.
    #[must_use]
    pub fn is_at_least_as_good_as(&self, other: &Self) -> bool {
        self.direction.is_at_least_as_good(self.value, other.value)
    }
}

// ---------------------------------------------------------------------------
// MultiObjectiveResult
// ---------------------------------------------------------------------------

/// A benchmark result augmented with multiple objective scores.
///
/// The `scores` vector must have the same length as the objective
/// specifications used to construct the [`ParetoFront`]. Each element
/// corresponds positionally to the objective at the same index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveResult {
    /// The underlying benchmark result with timing statistics.
    pub benchmark: BenchmarkResult,
    /// Objective scores, one per objective in positional order.
    pub scores: Vec<f64>,
}

impl MultiObjectiveResult {
    /// Creates a new multi-objective result.
    #[must_use]
    pub fn new(benchmark: BenchmarkResult, scores: Vec<f64>) -> Self {
        Self { benchmark, scores }
    }

    /// Returns the configuration from the underlying benchmark result.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.benchmark.config
    }

    /// Converts scores to [`ObjectiveValue`]s using the given specs.
    ///
    /// Returns `None` if the number of scores does not match the number
    /// of specs.
    #[must_use]
    pub fn to_objective_values(&self, specs: &[ObjectiveSpec]) -> Option<Vec<ObjectiveValue>> {
        if self.scores.len() != specs.len() {
            return None;
        }
        Some(
            specs
                .iter()
                .zip(self.scores.iter())
                .map(|(spec, &val)| {
                    ObjectiveValue::new(spec.objective.clone(), val, spec.direction)
                })
                .collect(),
        )
    }
}

// ---------------------------------------------------------------------------
// Pareto dominance
// ---------------------------------------------------------------------------

/// Checks whether solution `a` dominates solution `b` on all objectives.
///
/// Dominance requires that `a` is at least as good as `b` on every
/// objective AND strictly better on at least one.
///
/// # Errors
///
/// Returns [`AutotuneError::BenchmarkFailed`] if the scores and specs
/// have mismatched lengths.
pub fn dominates(
    a_scores: &[f64],
    b_scores: &[f64],
    specs: &[ObjectiveSpec],
) -> AutotuneResult<bool> {
    if a_scores.len() != specs.len() || b_scores.len() != specs.len() {
        return Err(AutotuneError::BenchmarkFailed(
            "score length does not match objective count".to_string(),
        ));
    }

    let mut all_at_least_as_good = true;
    let mut any_strictly_better = false;

    for (i, spec) in specs.iter().enumerate() {
        let a_val = a_scores.get(i).copied().unwrap_or(0.0);
        let b_val = b_scores.get(i).copied().unwrap_or(0.0);

        if !spec.direction.is_at_least_as_good(a_val, b_val) {
            all_at_least_as_good = false;
            break;
        }
        if spec.direction.is_better(a_val, b_val) {
            any_strictly_better = true;
        }
    }

    Ok(all_at_least_as_good && any_strictly_better)
}

// ---------------------------------------------------------------------------
// ParetoFront
// ---------------------------------------------------------------------------

/// Maintains the non-dominated set of solutions (the Pareto front).
///
/// Solutions are stored in objective-score order. When a new solution
/// is inserted, any existing solutions it dominates are removed, and
/// the new solution is only kept if it is not dominated by any
/// existing solution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFront {
    /// The objective specifications defining this front.
    specs: Vec<ObjectiveSpec>,
    /// The current set of non-dominated solutions.
    solutions: Vec<MultiObjectiveResult>,
}

impl ParetoFront {
    /// Creates a new, empty Pareto front with the given objective specs.
    #[must_use]
    pub fn new(specs: Vec<ObjectiveSpec>) -> Self {
        Self {
            specs,
            solutions: Vec::new(),
        }
    }

    /// Returns the objective specifications.
    #[must_use]
    pub fn specs(&self) -> &[ObjectiveSpec] {
        &self.specs
    }

    /// Returns the current Pareto front (non-dominated solutions).
    #[must_use]
    pub fn solutions(&self) -> &[MultiObjectiveResult] {
        &self.solutions
    }

    /// Returns the number of solutions on the front.
    #[must_use]
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Returns `true` if the front contains no solutions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Attempts to insert a result into the Pareto front.
    ///
    /// Returns `true` if the result was non-dominated and was added.
    /// Any existing solutions dominated by the new result are removed.
    /// Returns `false` if the result is dominated by an existing solution.
    ///
    /// # Panics
    ///
    /// This method does not panic. If score lengths mismatch, the
    /// result is silently rejected (returns `false`).
    pub fn insert(&mut self, result: MultiObjectiveResult) -> bool {
        if result.scores.len() != self.specs.len() {
            return false;
        }

        // Check if any existing solution dominates the new one.
        for existing in &self.solutions {
            if let Ok(true) = dominates(&existing.scores, &result.scores, &self.specs) {
                return false; // New result is dominated.
            }
        }

        // Remove any existing solutions dominated by the new one.
        self.solutions.retain(|existing| {
            dominates(&result.scores, &existing.scores, &self.specs)
                .map(|d| !d)
                .unwrap_or(true)
        });

        self.solutions.push(result);
        true
    }

    /// Computes the NSGA-II crowding distance for each solution on the front.
    ///
    /// Returns a vector of distances in the same order as [`Self::solutions`].
    /// Boundary solutions (best/worst on any objective) receive
    /// [`f64::INFINITY`]. An empty front returns an empty vector.
    #[must_use]
    pub fn crowding_distance(&self) -> Vec<f64> {
        let n = self.solutions.len();
        if n == 0 {
            return Vec::new();
        }
        if n <= 2 {
            return vec![f64::INFINITY; n];
        }

        let num_objectives = self.specs.len();
        let mut distances = vec![0.0_f64; n];

        for obj_idx in 0..num_objectives {
            // Create index-score pairs and sort by this objective.
            let mut indexed: Vec<(usize, f64)> = self
                .solutions
                .iter()
                .enumerate()
                .map(|(i, r)| (i, r.scores.get(obj_idx).copied().unwrap_or(0.0)))
                .collect();

            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Boundary points get infinite distance.
            if let Some(&(first_idx, _)) = indexed.first() {
                distances[first_idx] = f64::INFINITY;
            }
            if let Some(&(last_idx, _)) = indexed.last() {
                distances[last_idx] = f64::INFINITY;
            }

            // Objective range for normalization.
            let f_min = indexed.first().map(|x| x.1).unwrap_or(0.0);
            let f_max = indexed.last().map(|x| x.1).unwrap_or(0.0);
            let range = f_max - f_min;

            if range.abs() < f64::EPSILON {
                continue; // All values identical — no contribution.
            }

            // Interior points: normalized distance between neighbours.
            for k in 1..(indexed.len() - 1) {
                let (idx, _) = indexed[k];
                let prev_val = indexed[k - 1].1;
                let next_val = indexed[k + 1].1;
                distances[idx] += (next_val - prev_val) / range;
            }
        }

        distances
    }

    /// Selects the knee point — the solution with the best overall
    /// tradeoff on the Pareto front.
    ///
    /// The knee is defined as the point with maximum perpendicular
    /// distance from the hyperplane connecting the extreme solutions
    /// in normalized objective space.
    ///
    /// Returns `None` if the front is empty.
    #[must_use]
    pub fn select_knee_point(&self) -> Option<&MultiObjectiveResult> {
        if self.solutions.is_empty() {
            return None;
        }
        if self.solutions.len() <= 2 {
            return self.solutions.first();
        }

        let num_obj = self.specs.len();
        if num_obj == 0 {
            return self.solutions.first();
        }

        // Normalize all scores to [0, 1] (direction-adjusted so that
        // higher normalized value = better).
        let (mins, maxs) = self.compute_score_ranges();

        let normalized: Vec<Vec<f64>> = self
            .solutions
            .iter()
            .map(|r| self.normalize_scores(&r.scores, &mins, &maxs))
            .collect();

        // For 2-objective case, use distance from line between extremes.
        // For N-objective case, use distance from the hyperplane.
        if num_obj == 2 {
            self.knee_point_2d(&normalized)
        } else {
            self.knee_point_nd(&normalized)
        }
    }

    /// Computes (min, max) ranges for each objective.
    fn compute_score_ranges(&self) -> (Vec<f64>, Vec<f64>) {
        let num_obj = self.specs.len();
        let mut mins = vec![f64::INFINITY; num_obj];
        let mut maxs = vec![f64::NEG_INFINITY; num_obj];

        for sol in &self.solutions {
            for (i, &val) in sol.scores.iter().enumerate() {
                if i < num_obj {
                    if val < mins[i] {
                        mins[i] = val;
                    }
                    if val > maxs[i] {
                        maxs[i] = val;
                    }
                }
            }
        }
        (mins, maxs)
    }

    /// Normalizes scores to [0, 1] with direction adjustment.
    /// After normalization, higher = better for all objectives.
    fn normalize_scores(&self, scores: &[f64], mins: &[f64], maxs: &[f64]) -> Vec<f64> {
        scores
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let range =
                    maxs.get(i).copied().unwrap_or(1.0) - mins.get(i).copied().unwrap_or(0.0);
                if range.abs() < f64::EPSILON {
                    return 0.5;
                }
                let normalized = (val - mins.get(i).copied().unwrap_or(0.0)) / range;
                // Flip for minimize objectives so higher = better.
                let spec = self.specs.get(i);
                match spec.map(|s| s.direction) {
                    Some(ObjectiveDirection::Minimize) => 1.0 - normalized,
                    _ => normalized,
                }
            })
            .collect()
    }

    /// 2D knee point: maximum distance from line between extreme points.
    fn knee_point_2d(&self, normalized: &[Vec<f64>]) -> Option<&MultiObjectiveResult> {
        // Find indices of extreme points (best on each objective).
        let mut best_0_idx = 0;
        let mut best_1_idx = 0;
        let mut best_0_val = f64::NEG_INFINITY;
        let mut best_1_val = f64::NEG_INFINITY;

        for (i, norms) in normalized.iter().enumerate() {
            let v0 = norms.first().copied().unwrap_or(0.0);
            let v1 = norms.get(1).copied().unwrap_or(0.0);
            if v0 > best_0_val {
                best_0_val = v0;
                best_0_idx = i;
            }
            if v1 > best_1_val {
                best_1_val = v1;
                best_1_idx = i;
            }
        }

        let p1 = &normalized[best_0_idx];
        let p2 = &normalized[best_1_idx];

        let ax = p1.first().copied().unwrap_or(0.0);
        let ay = p1.get(1).copied().unwrap_or(0.0);
        let bx = p2.first().copied().unwrap_or(0.0);
        let by = p2.get(1).copied().unwrap_or(0.0);

        // Line direction vector.
        let dx = bx - ax;
        let dy = by - ay;
        let line_len = (dx * dx + dy * dy).sqrt();

        if line_len < f64::EPSILON {
            return self.solutions.first();
        }

        // Find point with maximum perpendicular distance from line.
        let mut max_dist = f64::NEG_INFINITY;
        let mut knee_idx = 0;

        for (i, norms) in normalized.iter().enumerate() {
            let px = norms.first().copied().unwrap_or(0.0);
            let py = norms.get(1).copied().unwrap_or(0.0);

            // Perpendicular distance = |cross product| / |line_len|
            let cross = ((px - ax) * dy - (py - ay) * dx).abs();
            let dist = cross / line_len;

            if dist > max_dist {
                max_dist = dist;
                knee_idx = i;
            }
        }

        self.solutions.get(knee_idx)
    }

    /// N-dimensional knee point: maximum distance from hyperplane through extremes.
    fn knee_point_nd(&self, normalized: &[Vec<f64>]) -> Option<&MultiObjectiveResult> {
        // Use sum-of-normalized-scores as a proxy for distance from
        // the utopia-nadir diagonal. The point with the highest sum
        // of normalized scores (all direction-adjusted) is the best
        // overall tradeoff.
        let mut best_idx = 0;
        let mut best_sum = f64::NEG_INFINITY;

        for (i, norms) in normalized.iter().enumerate() {
            let sum: f64 = norms.iter().sum();
            if sum > best_sum {
                best_sum = sum;
                best_idx = i;
            }
        }

        self.solutions.get(best_idx)
    }
}

// ---------------------------------------------------------------------------
// WeightedScalarization
// ---------------------------------------------------------------------------

/// Converts multi-objective scores to a single scalar using weighted sum.
///
/// Each objective score is normalized to [0, 1] and multiplied by its
/// weight. The final score is the weighted sum. Direction is accounted
/// for: minimize objectives are flipped so that higher normalized
/// values always mean "better".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedScalarization {
    /// Weights for each objective (same positional order as specs).
    weights: Vec<f64>,
    /// The objective specifications.
    specs: Vec<ObjectiveSpec>,
}

impl WeightedScalarization {
    /// Creates a new weighted scalarization.
    ///
    /// # Errors
    ///
    /// Returns an error if the weights and specs have different lengths,
    /// or if any weight is negative.
    pub fn new(weights: Vec<f64>, specs: Vec<ObjectiveSpec>) -> AutotuneResult<Self> {
        if weights.len() != specs.len() {
            return Err(AutotuneError::BenchmarkFailed(
                "weights and specs must have the same length".to_string(),
            ));
        }
        if weights.iter().any(|&w| w < 0.0) {
            return Err(AutotuneError::BenchmarkFailed(
                "weights must be non-negative".to_string(),
            ));
        }
        Ok(Self { weights, specs })
    }

    /// Computes the scalarized score for a set of objective values.
    ///
    /// Scores are normalized using the provided min/max ranges. Higher
    /// return values are always better.
    ///
    /// Returns `None` if the scores length does not match.
    #[must_use]
    pub fn scalarize(&self, scores: &[f64], mins: &[f64], maxs: &[f64]) -> Option<f64> {
        if scores.len() != self.specs.len() {
            return None;
        }

        let mut total = 0.0;
        let mut weight_sum = 0.0;

        for (i, (spec, &weight)) in self.specs.iter().zip(self.weights.iter()).enumerate() {
            let val = scores.get(i).copied().unwrap_or(0.0);
            let min_val = mins.get(i).copied().unwrap_or(0.0);
            let max_val = maxs.get(i).copied().unwrap_or(1.0);
            let range = max_val - min_val;

            let normalized = if range.abs() < f64::EPSILON {
                0.5
            } else {
                (val - min_val) / range
            };

            let adjusted = match spec.direction {
                ObjectiveDirection::Minimize => 1.0 - normalized,
                ObjectiveDirection::Maximize => normalized,
            };

            total += adjusted * weight;
            weight_sum += weight;
        }

        if weight_sum.abs() < f64::EPSILON {
            Some(0.0)
        } else {
            Some(total / weight_sum)
        }
    }

    /// Returns the weights.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Returns the objective specs.
    #[must_use]
    pub fn specs(&self) -> &[ObjectiveSpec] {
        &self.specs
    }
}

// ---------------------------------------------------------------------------
// MultiObjectiveOptimizer
// ---------------------------------------------------------------------------

/// Drives multi-objective optimization over a set of kernel configurations.
///
/// The optimizer evaluates candidate configurations using a user-provided
/// evaluation function and builds a Pareto front of non-dominated
/// solutions.
#[derive(Debug, Clone)]
pub struct MultiObjectiveOptimizer {
    /// The objective specifications.
    specs: Vec<ObjectiveSpec>,
    /// Maximum number of evaluations to perform.
    max_evaluations: usize,
}

impl MultiObjectiveOptimizer {
    /// Creates a new multi-objective optimizer.
    ///
    /// # Arguments
    ///
    /// * `specs` — The objectives to optimize with their directions.
    /// * `max_evaluations` — Upper bound on the number of configurations
    ///   to evaluate.
    #[must_use]
    pub fn new(specs: Vec<ObjectiveSpec>, max_evaluations: usize) -> Self {
        Self {
            specs,
            max_evaluations,
        }
    }

    /// Returns the objective specifications.
    #[must_use]
    pub fn specs(&self) -> &[ObjectiveSpec] {
        &self.specs
    }

    /// Returns the maximum number of evaluations.
    #[must_use]
    pub fn max_evaluations(&self) -> usize {
        self.max_evaluations
    }

    /// Runs the optimization loop.
    ///
    /// Evaluates each configuration in `candidates` using the provided
    /// `evaluate` closure, which must return a vector of objective scores
    /// in the same order as the specs. Builds and returns the Pareto
    /// front of non-dominated solutions.
    ///
    /// The evaluation function receives a `&Config` and a
    /// `&BenchmarkResult` and must produce `Vec<f64>` scores.
    ///
    /// # Errors
    ///
    /// Returns an error if the evaluation function fails or if no valid
    /// results are produced.
    pub fn optimize<F>(
        &self,
        candidates: &[Config],
        benchmark_fn: impl Fn(&Config) -> AutotuneResult<BenchmarkResult>,
        score_fn: F,
    ) -> AutotuneResult<ParetoFront>
    where
        F: Fn(&Config, &BenchmarkResult) -> AutotuneResult<Vec<f64>>,
    {
        let mut front = ParetoFront::new(self.specs.clone());
        let limit = self.max_evaluations.min(candidates.len());

        for config in candidates.iter().take(limit) {
            let bench_result = match benchmark_fn(config) {
                Ok(r) => r,
                Err(_) => continue, // Skip failed benchmarks.
            };

            let scores = match score_fn(config, &bench_result) {
                Ok(s) if s.len() == self.specs.len() => s,
                _ => continue, // Skip invalid scores.
            };

            let mo_result = MultiObjectiveResult::new(bench_result, scores);
            front.insert(mo_result);
        }

        if front.is_empty() {
            return Err(AutotuneError::NoViableConfig);
        }

        Ok(front)
    }

    /// Selects the knee point from the Pareto front.
    ///
    /// Convenience method that runs optimization and returns the single
    /// best-tradeoff solution.
    ///
    /// # Errors
    ///
    /// Returns an error if optimization fails or the front is empty.
    pub fn select_best<F>(
        &self,
        candidates: &[Config],
        benchmark_fn: impl Fn(&Config) -> AutotuneResult<BenchmarkResult>,
        score_fn: F,
    ) -> AutotuneResult<MultiObjectiveResult>
    where
        F: Fn(&Config, &BenchmarkResult) -> AutotuneResult<Vec<f64>>,
    {
        let front = self.optimize(candidates, benchmark_fn, score_fn)?;
        front
            .select_knee_point()
            .cloned()
            .ok_or(AutotuneError::NoViableConfig)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a simple BenchmarkResult for testing.
    fn make_bench(median_us: f64, gflops: Option<f64>) -> BenchmarkResult {
        BenchmarkResult {
            config: Config::new(),
            median_us,
            min_us: median_us * 0.9,
            max_us: median_us * 1.1,
            stddev_us: median_us * 0.05,
            gflops,
            efficiency: None,
        }
    }

    /// Helper to create a MultiObjectiveResult.
    fn make_mo(median_us: f64, scores: Vec<f64>) -> MultiObjectiveResult {
        MultiObjectiveResult::new(make_bench(median_us, None), scores)
    }

    fn throughput_latency_specs() -> Vec<ObjectiveSpec> {
        vec![
            ObjectiveSpec::new(Objective::Throughput, ObjectiveDirection::Maximize),
            ObjectiveSpec::new(Objective::Latency, ObjectiveDirection::Minimize),
        ]
    }

    // -- ObjectiveDirection tests --

    #[test]
    fn direction_minimize_better() {
        assert!(ObjectiveDirection::Minimize.is_better(10.0, 20.0));
        assert!(!ObjectiveDirection::Minimize.is_better(20.0, 10.0));
        assert!(!ObjectiveDirection::Minimize.is_better(10.0, 10.0));
    }

    #[test]
    fn direction_maximize_better() {
        assert!(ObjectiveDirection::Maximize.is_better(20.0, 10.0));
        assert!(!ObjectiveDirection::Maximize.is_better(10.0, 20.0));
        assert!(!ObjectiveDirection::Maximize.is_better(10.0, 10.0));
    }

    #[test]
    fn direction_at_least_as_good() {
        assert!(ObjectiveDirection::Minimize.is_at_least_as_good(10.0, 10.0));
        assert!(ObjectiveDirection::Minimize.is_at_least_as_good(5.0, 10.0));
        assert!(!ObjectiveDirection::Minimize.is_at_least_as_good(15.0, 10.0));

        assert!(ObjectiveDirection::Maximize.is_at_least_as_good(10.0, 10.0));
        assert!(ObjectiveDirection::Maximize.is_at_least_as_good(15.0, 10.0));
        assert!(!ObjectiveDirection::Maximize.is_at_least_as_good(5.0, 10.0));
    }

    // -- Objective display --

    #[test]
    fn objective_display() {
        assert_eq!(format!("{}", Objective::Throughput), "throughput");
        assert_eq!(format!("{}", Objective::Latency), "latency");
        assert_eq!(format!("{}", Objective::Power), "power");
        assert_eq!(format!("{}", Objective::Memory), "memory");
        assert_eq!(
            format!("{}", Objective::Custom("energy".to_string())),
            "energy"
        );
    }

    // -- Dominance tests --

    #[test]
    fn dominance_basic() {
        let specs = throughput_latency_specs();

        // a: throughput=500 (max), latency=50 (min) — better on both
        // b: throughput=400, latency=60
        let a = [500.0, 50.0];
        let b = [400.0, 60.0];
        assert!(dominates(&a, &b, &specs).unwrap_or(false));
        assert!(!dominates(&b, &a, &specs).unwrap_or(true));
    }

    #[test]
    fn dominance_equal_is_not_domination() {
        let specs = throughput_latency_specs();
        let a = [500.0, 50.0];
        assert!(!dominates(&a, &a, &specs).unwrap_or(true));
    }

    #[test]
    fn dominance_tradeoff_no_domination() {
        let specs = throughput_latency_specs();
        // a: higher throughput, higher latency
        // b: lower throughput, lower latency
        let a = [600.0, 80.0];
        let b = [400.0, 40.0];
        assert!(!dominates(&a, &b, &specs).unwrap_or(true));
        assert!(!dominates(&b, &a, &specs).unwrap_or(true));
    }

    #[test]
    fn dominance_mismatched_lengths() {
        let specs = throughput_latency_specs();
        let result = dominates(&[1.0], &[1.0, 2.0], &specs);
        assert!(result.is_err());
    }

    // -- ParetoFront tests --

    #[test]
    fn pareto_front_insert_single() {
        let mut front = ParetoFront::new(throughput_latency_specs());
        assert!(front.insert(make_mo(100.0, vec![500.0, 100.0])));
        assert_eq!(front.len(), 1);
    }

    #[test]
    fn pareto_front_dominated_rejected() {
        let mut front = ParetoFront::new(throughput_latency_specs());
        // Good: high throughput, low latency.
        assert!(front.insert(make_mo(50.0, vec![600.0, 50.0])));
        // Dominated: lower throughput AND higher latency.
        assert!(!front.insert(make_mo(100.0, vec![400.0, 80.0])));
        assert_eq!(front.len(), 1);
    }

    #[test]
    fn pareto_front_dominator_removes_existing() {
        let mut front = ParetoFront::new(throughput_latency_specs());
        assert!(front.insert(make_mo(100.0, vec![400.0, 80.0])));
        assert_eq!(front.len(), 1);
        // Better on both objectives.
        assert!(front.insert(make_mo(50.0, vec![600.0, 50.0])));
        assert_eq!(front.len(), 1);
        // The surviving solution should be the dominator.
        assert!((front.solutions()[0].scores[0] - 600.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pareto_front_tradeoff_both_kept() {
        let mut front = ParetoFront::new(throughput_latency_specs());
        // High throughput, high latency.
        assert!(front.insert(make_mo(80.0, vec![700.0, 80.0])));
        // Low throughput, low latency — tradeoff, not dominated.
        assert!(front.insert(make_mo(30.0, vec![300.0, 30.0])));
        assert_eq!(front.len(), 2);
    }

    #[test]
    fn pareto_front_mismatched_scores_rejected() {
        let mut front = ParetoFront::new(throughput_latency_specs());
        // Wrong number of scores.
        assert!(!front.insert(make_mo(100.0, vec![500.0])));
        assert!(front.is_empty());
    }

    // -- Crowding distance tests --

    #[test]
    fn crowding_distance_empty() {
        let front = ParetoFront::new(throughput_latency_specs());
        assert!(front.crowding_distance().is_empty());
    }

    #[test]
    fn crowding_distance_two_points() {
        let mut front = ParetoFront::new(throughput_latency_specs());
        front.insert(make_mo(50.0, vec![600.0, 50.0]));
        front.insert(make_mo(30.0, vec![300.0, 30.0]));
        let dists = front.crowding_distance();
        assert_eq!(dists.len(), 2);
        assert!(dists[0].is_infinite());
        assert!(dists[1].is_infinite());
    }

    #[test]
    fn crowding_distance_three_points() {
        let mut front = ParetoFront::new(throughput_latency_specs());
        front.insert(make_mo(80.0, vec![700.0, 80.0]));
        front.insert(make_mo(50.0, vec![500.0, 50.0]));
        front.insert(make_mo(30.0, vec![300.0, 30.0]));
        let dists = front.crowding_distance();
        assert_eq!(dists.len(), 3);
        // The two boundary points on each objective should be infinite.
        // The middle point should have a finite distance.
        let finite_count = dists.iter().filter(|d| d.is_finite()).count();
        // At least one point should have finite distance.
        assert!(finite_count >= 1);
    }

    // -- Knee point tests --

    #[test]
    fn knee_point_empty_returns_none() {
        let front = ParetoFront::new(throughput_latency_specs());
        assert!(front.select_knee_point().is_none());
    }

    #[test]
    fn knee_point_single_returns_it() {
        let mut front = ParetoFront::new(throughput_latency_specs());
        front.insert(make_mo(100.0, vec![500.0, 100.0]));
        let knee = front.select_knee_point();
        assert!(knee.is_some());
    }

    #[test]
    fn knee_point_selects_balanced_solution() {
        let mut front = ParetoFront::new(throughput_latency_specs());
        // Extreme 1: very high throughput, very high latency.
        front.insert(make_mo(200.0, vec![1000.0, 200.0]));
        // Middle: balanced.
        front.insert(make_mo(80.0, vec![600.0, 80.0]));
        // Extreme 2: low throughput, very low latency.
        front.insert(make_mo(10.0, vec![200.0, 10.0]));

        let knee = front.select_knee_point();
        assert!(knee.is_some());
        let knee = knee.unwrap_or_else(|| &front.solutions()[0]);
        // The balanced solution should be the knee.
        assert!(
            (knee.scores[0] - 600.0).abs() < f64::EPSILON,
            "expected knee at throughput=600, got {}",
            knee.scores[0]
        );
    }

    // -- WeightedScalarization tests --

    #[test]
    fn scalarization_basic() {
        let specs = throughput_latency_specs();
        let scalar =
            WeightedScalarization::new(vec![1.0, 1.0], specs).expect("valid scalarization");

        let scores = [500.0, 50.0];
        let mins = [200.0, 20.0];
        let maxs = [800.0, 80.0];

        let result = scalar.scalarize(&scores, &mins, &maxs);
        assert!(result.is_some());
        let val = result.unwrap_or(0.0);
        // throughput: (500-200)/(800-200) = 0.5 (maximize, keep as is)
        // latency: (50-20)/(80-20) = 0.5, flipped = 0.5
        // average = 0.5
        assert!((val - 0.5).abs() < 1e-9);
    }

    #[test]
    fn scalarization_mismatched_lengths() {
        let specs = throughput_latency_specs();
        let result = WeightedScalarization::new(vec![1.0], specs);
        assert!(result.is_err());
    }

    #[test]
    fn scalarization_negative_weight_rejected() {
        let specs = throughput_latency_specs();
        let result = WeightedScalarization::new(vec![1.0, -0.5], specs);
        assert!(result.is_err());
    }

    // -- MultiObjectiveOptimizer tests --

    #[test]
    fn optimizer_basic() {
        let specs = throughput_latency_specs();
        let optimizer = MultiObjectiveOptimizer::new(specs, 10);

        let candidates = vec![
            Config::new().with_tile_m(64),
            Config::new().with_tile_m(128),
            Config::new().with_tile_m(256),
        ];

        // Synthetic benchmark: latency proportional to tile_m,
        // throughput inversely proportional.
        let bench_fn = |config: &Config| -> AutotuneResult<BenchmarkResult> {
            let median = f64::from(config.tile_m);
            Ok(BenchmarkResult {
                config: config.clone(),
                median_us: median,
                min_us: median * 0.9,
                max_us: median * 1.1,
                stddev_us: median * 0.05,
                gflops: Some(10000.0 / median),
                efficiency: None,
            })
        };

        let score_fn = |config: &Config, bench: &BenchmarkResult| -> AutotuneResult<Vec<f64>> {
            let throughput = bench.gflops.unwrap_or(0.0);
            let latency = f64::from(config.tile_m);
            Ok(vec![throughput, latency])
        };

        let front = optimizer.optimize(&candidates, bench_fn, score_fn);
        assert!(front.is_ok());
        let front = front.unwrap_or_else(|_| ParetoFront::new(Vec::new()));
        // All three are tradeoffs (higher tile = more throughput but more latency).
        // Actually: throughput = 10000/tile_m, so smaller tile = higher throughput
        // AND lower latency. So tile_m=64 dominates the others.
        assert!(!front.is_empty());
    }

    #[test]
    fn optimizer_select_best() {
        let specs = throughput_latency_specs();
        let optimizer = MultiObjectiveOptimizer::new(specs, 10);

        let candidates = vec![
            Config::new().with_tile_m(64),
            Config::new().with_tile_m(128),
        ];

        let bench_fn = |config: &Config| -> AutotuneResult<BenchmarkResult> {
            let median = f64::from(config.tile_m);
            Ok(BenchmarkResult {
                config: config.clone(),
                median_us: median,
                min_us: median * 0.9,
                max_us: median * 1.1,
                stddev_us: median * 0.05,
                gflops: None,
                efficiency: None,
            })
        };

        // Create a genuine tradeoff so both survive.
        let score_fn = |config: &Config, _bench: &BenchmarkResult| -> AutotuneResult<Vec<f64>> {
            let tile = f64::from(config.tile_m);
            // Throughput increases with tile, latency also increases.
            Ok(vec![tile * 2.0, tile])
        };

        let best = optimizer.select_best(&candidates, bench_fn, score_fn);
        assert!(best.is_ok());
    }

    #[test]
    fn optimizer_empty_candidates() {
        let specs = throughput_latency_specs();
        let optimizer = MultiObjectiveOptimizer::new(specs, 10);
        let candidates: Vec<Config> = Vec::new();

        let bench_fn =
            |_: &Config| -> AutotuneResult<BenchmarkResult> { Err(AutotuneError::NoViableConfig) };
        let score_fn = |_: &Config, _: &BenchmarkResult| -> AutotuneResult<Vec<f64>> { Ok(vec![]) };

        let result = optimizer.optimize(&candidates, bench_fn, score_fn);
        assert!(result.is_err());
    }

    // -- ObjectiveValue tests --

    #[test]
    fn objective_value_comparison() {
        let a = ObjectiveValue::new(Objective::Latency, 10.0, ObjectiveDirection::Minimize);
        let b = ObjectiveValue::new(Objective::Latency, 20.0, ObjectiveDirection::Minimize);
        assert!(a.is_better_than(&b));
        assert!(!b.is_better_than(&a));
        assert!(a.is_at_least_as_good_as(&a));
    }

    // -- MultiObjectiveResult tests --

    #[test]
    fn mo_result_to_objective_values() {
        let specs = throughput_latency_specs();
        let result = make_mo(100.0, vec![500.0, 100.0]);
        let vals = result.to_objective_values(&specs);
        assert!(vals.is_some());
        let vals = vals.unwrap_or_default();
        assert_eq!(vals.len(), 2);
        assert!((vals[0].value - 500.0).abs() < f64::EPSILON);
        assert!((vals[1].value - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mo_result_to_objective_values_mismatch() {
        let specs = throughput_latency_specs();
        let result = make_mo(100.0, vec![500.0]); // Wrong length.
        assert!(result.to_objective_values(&specs).is_none());
    }

    // -- Serde roundtrip --

    #[test]
    fn serde_roundtrip_pareto_front() {
        let mut front = ParetoFront::new(throughput_latency_specs());
        front.insert(make_mo(100.0, vec![500.0, 100.0]));
        front.insert(make_mo(50.0, vec![300.0, 50.0]));

        let json = serde_json::to_string(&front).expect("serialize");
        let restored: ParetoFront = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.len(), front.len());
        assert_eq!(restored.specs().len(), front.specs().len());
    }
}
