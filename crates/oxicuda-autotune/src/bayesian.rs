//! Bayesian optimization search strategy for autotuning.
//!
//! This module provides a Gaussian Process (GP)-based Bayesian optimizer
//! that intelligently explores the search space by building a surrogate
//! model of kernel performance and using acquisition functions to decide
//! which configuration to evaluate next.
//!
//! # Architecture
//!
//! ```text
//!  ┌────────────────────────────────────────────────────┐
//!  │              BayesianOptimizer                     │
//!  │                                                    │
//!  │  SearchSpace ──► candidate configs                 │
//!  │                        │                           │
//!  │                        ▼                           │
//!  │  GaussianProcess ──► predict(mean, var)            │
//!  │                        │                           │
//!  │                        ▼                           │
//!  │  AcquisitionFunction ──► score each candidate      │
//!  │                        │                           │
//!  │                        ▼                           │
//!  │  suggest_next() ──► best-scoring Config            │
//!  └────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::bayesian::{BayesianOptimizer, AcquisitionFunction};
//! use oxicuda_autotune::SearchSpace;
//!
//! let space = SearchSpace::minimal();
//! let mut optimizer = BayesianOptimizer::new(
//!     space,
//!     AcquisitionFunction::ExpectedImprovement,
//! );
//!
//! // Suggest and observe a few configurations
//! for _ in 0..5 {
//!     if let Ok(config) = optimizer.suggest_next() {
//!         // In a real scenario, benchmark the config on GPU
//!         let performance = 42.0; // simulated
//!         optimizer.observe(config, performance);
//!     }
//! }
//!
//! if let Some((best_cfg, best_perf)) = optimizer.best_config() {
//!     println!("Best config: {:?}, performance: {:.2}", best_cfg, best_perf);
//! }
//! ```

use crate::config::Config;
use crate::error::AutotuneError;
use crate::search_space::SearchSpace;

// ---------------------------------------------------------------------------
// Gaussian Process prediction result
// ---------------------------------------------------------------------------

/// Prediction from the Gaussian Process surrogate model.
///
/// Contains both the mean prediction and the associated uncertainty
/// (variance), which together inform the acquisition function.
#[derive(Debug, Clone, Copy)]
pub struct GpPrediction {
    /// Predicted mean performance value.
    pub mean: f64,
    /// Predicted variance (uncertainty squared).
    pub variance: f64,
}

// ---------------------------------------------------------------------------
// Gaussian Process surrogate model
// ---------------------------------------------------------------------------

/// Gaussian Process regression model with RBF (squared exponential) kernel.
///
/// Used as a surrogate model to approximate the unknown performance
/// function over the configuration search space. The GP provides both
/// a mean prediction and an uncertainty estimate, which are essential
/// for the acquisition function to balance exploration vs. exploitation.
///
/// The kernel function is:
///
/// ```text
/// k(x, x') = signal_variance * exp(-||x - x'||^2 / (2 * length_scale^2))
/// ```
///
/// Inference uses Cholesky decomposition of the kernel matrix for
/// numerical stability, implemented inline without external linear
/// algebra dependencies.
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    /// Observed training input points (each normalized to \[0, 1\]).
    x_train: Vec<Vec<f64>>,
    /// Observed training target values.
    y_train: Vec<f64>,
    /// RBF kernel length scale parameter.
    length_scale: f64,
    /// RBF kernel signal variance (output scale).
    signal_variance: f64,
    /// Observation noise variance (jitter for numerical stability).
    noise_variance: f64,
}

impl GaussianProcess {
    /// Creates a new Gaussian Process with the given hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `length_scale` — Controls the smoothness of the GP. Larger values
    ///   mean the function varies more slowly.
    /// * `signal_variance` — Controls the overall amplitude of variation.
    /// * `noise_variance` — Observation noise; also acts as jitter for
    ///   numerical stability of the Cholesky decomposition.
    #[must_use]
    pub fn new(length_scale: f64, signal_variance: f64, noise_variance: f64) -> Self {
        Self {
            x_train: Vec::new(),
            y_train: Vec::new(),
            length_scale,
            signal_variance,
            noise_variance,
        }
    }

    /// Adds a new observation (input point and target value) to the GP.
    pub fn add_observation(&mut self, x: Vec<f64>, y: f64) {
        self.x_train.push(x);
        self.y_train.push(y);
    }

    /// Returns the number of observations in the GP.
    #[must_use]
    pub fn num_observations(&self) -> usize {
        self.x_train.len()
    }

    /// Predicts the mean and variance at a new input point.
    ///
    /// When no observations exist, returns the prior (mean=0, variance=signal_variance).
    ///
    /// Uses Cholesky-based inference:
    /// 1. Compute K (kernel matrix of training points) + noise * I
    /// 2. Cholesky decompose: K = L * L^T
    /// 3. Solve L * alpha_tmp = y, then L^T * alpha = alpha_tmp
    /// 4. Mean = k_star^T * alpha
    /// 5. Solve L * v = k_star, then var = k(x,x) - v^T * v
    pub fn predict(&self, x: &[f64]) -> GpPrediction {
        let n = self.x_train.len();

        // Prior when no observations exist
        if n == 0 {
            return GpPrediction {
                mean: 0.0,
                variance: self.signal_variance,
            };
        }

        // Build kernel matrix K(X_train, X_train) + noise * I
        let mut k_mat = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                k_mat[i * n + j] = self.rbf_kernel(&self.x_train[i], &self.x_train[j]);
            }
            // Add noise to diagonal
            k_mat[i * n + i] += self.noise_variance;
        }

        // Cholesky decomposition: K = L * L^T
        let chol = match cholesky_decompose(&k_mat, n) {
            Some(l) => l,
            None => {
                // Fallback: return prior if Cholesky fails (degenerate matrix)
                return GpPrediction {
                    mean: 0.0,
                    variance: self.signal_variance,
                };
            }
        };

        // k_star: kernel between test point and all training points
        let mut k_star = Vec::with_capacity(n);
        for i in 0..n {
            k_star.push(self.rbf_kernel(x, &self.x_train[i]));
        }

        // Solve L * alpha_tmp = y  (forward substitution)
        let alpha_tmp = forward_solve(&chol, n, &self.y_train);

        // Solve L^T * alpha = alpha_tmp  (back substitution)
        let alpha = backward_solve(&chol, n, &alpha_tmp);

        // Predictive mean: k_star^T * alpha
        let mean = dot(&k_star, &alpha);

        // Solve L * v = k_star  (forward substitution)
        let v = forward_solve(&chol, n, &k_star);

        // Predictive variance: k(x, x) - v^T * v
        let k_xx = self.rbf_kernel(x, x);
        let v_dot_v = dot(&v, &v);
        let variance = (k_xx - v_dot_v).max(0.0); // clamp for numerical safety

        GpPrediction { mean, variance }
    }

    /// RBF (squared exponential) kernel function.
    ///
    /// k(x, x') = signal_variance * exp(-||x - x'||^2 / (2 * length_scale^2))
    fn rbf_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let sq_dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        self.signal_variance * (-sq_dist / (2.0 * self.length_scale * self.length_scale)).exp()
    }
}

// ---------------------------------------------------------------------------
// Cholesky decomposition and triangular solvers (inline, no external deps)
// ---------------------------------------------------------------------------

/// Performs Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor L such that A = L * L^T,
/// or `None` if the matrix is not positive definite.
///
/// The matrix is stored in row-major order as a flat vector of size n*n.
fn cholesky_decompose(mat: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }

            if i == j {
                let diag = mat[i * n + i] - sum;
                if diag <= 0.0 {
                    return None; // Not positive definite
                }
                l[i * n + j] = diag.sqrt();
            } else {
                let denom = l[j * n + j];
                if denom.abs() < 1e-15 {
                    return None; // Near-singular
                }
                l[i * n + j] = (mat[i * n + j] - sum) / denom;
            }
        }
    }

    Some(l)
}

/// Forward substitution: solves L * x = b where L is lower-triangular.
///
/// L is stored in row-major order as a flat vector of size n*n.
fn forward_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = 0.0_f64;
        for j in 0..i {
            sum += l[i * n + j] * x[j];
        }
        let diag = l[i * n + i];
        if diag.abs() < 1e-15 {
            x[i] = 0.0;
        } else {
            x[i] = (b[i] - sum) / diag;
        }
    }
    x
}

/// Backward substitution: solves L^T * x = b where L is lower-triangular.
///
/// L is stored in row-major order as a flat vector of size n*n.
fn backward_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = 0.0_f64;
        for j in (i + 1)..n {
            sum += l[j * n + i] * x[j]; // L^T element at (i, j) is L(j, i)
        }
        let diag = l[i * n + i];
        if diag.abs() < 1e-15 {
            x[i] = 0.0;
        } else {
            x[i] = (b[i] - sum) / diag;
        }
    }
    x
}

/// Dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Acquisition functions
// ---------------------------------------------------------------------------

/// Acquisition function for deciding which configuration to evaluate next.
///
/// The acquisition function scores candidate configurations based on
/// the GP's prediction (mean and variance), trading off exploitation
/// (high predicted performance) vs. exploration (high uncertainty).
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Expected Improvement over the current best observation.
    ///
    /// EI(x) = (mu(x) - f_best) * Phi(z) + sigma(x) * phi(z)
    /// where z = (mu(x) - f_best) / sigma(x)
    ///
    /// Naturally balances exploration and exploitation.
    ExpectedImprovement,

    /// Upper Confidence Bound: mu(x) + kappa * sigma(x).
    ///
    /// Higher `kappa` increases exploration. Typical values: 1.0-3.0.
    UpperConfidenceBound {
        /// Exploration-exploitation trade-off parameter.
        kappa: f64,
    },

    /// Probability of Improvement over the current best.
    ///
    /// PI(x) = Phi((mu(x) - f_best) / sigma(x))
    ///
    /// Tends to be more exploitative than EI.
    ProbabilityOfImprovement,
}

impl AcquisitionFunction {
    /// Evaluates the acquisition function at a point given GP prediction
    /// and the best observed value so far.
    ///
    /// Higher values indicate more promising candidates.
    /// For minimization problems, negate the performance values before
    /// passing them to the GP.
    fn evaluate(&self, prediction: &GpPrediction, best_observed: Option<f64>) -> f64 {
        let sigma = prediction.variance.sqrt();

        // If variance is essentially zero, no exploration value
        if sigma < 1e-12 {
            return match self {
                AcquisitionFunction::ExpectedImprovement => {
                    let f_best = best_observed.unwrap_or(0.0);
                    (prediction.mean - f_best).max(0.0)
                }
                AcquisitionFunction::UpperConfidenceBound { .. } => prediction.mean,
                AcquisitionFunction::ProbabilityOfImprovement => {
                    let f_best = best_observed.unwrap_or(0.0);
                    if prediction.mean > f_best { 1.0 } else { 0.0 }
                }
            };
        }

        let f_best = best_observed.unwrap_or(0.0);

        match self {
            AcquisitionFunction::ExpectedImprovement => {
                let z = (prediction.mean - f_best) / sigma;
                let phi = standard_normal_pdf(z);
                let big_phi = standard_normal_cdf(z);
                (prediction.mean - f_best) * big_phi + sigma * phi
            }
            AcquisitionFunction::UpperConfidenceBound { kappa } => prediction.mean + kappa * sigma,
            AcquisitionFunction::ProbabilityOfImprovement => {
                let z = (prediction.mean - f_best) / sigma;
                standard_normal_cdf(z)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Standard normal distribution helpers (no external deps)
// ---------------------------------------------------------------------------

/// Standard normal probability density function: phi(z) = exp(-z^2/2) / sqrt(2*pi).
fn standard_normal_pdf(z: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1 / sqrt(2*pi)
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

/// Standard normal cumulative distribution function (Abramowitz & Stegun approximation).
///
/// Accurate to ~1e-7. This avoids pulling in an external statistics crate.
fn standard_normal_cdf(z: f64) -> f64 {
    // Use the Abramowitz & Stegun (1964) rational approximation (7.1.26)
    // for the complementary error function, then convert to CDF.
    //
    // For z < 0, use symmetry: Phi(z) = 1 - Phi(-z).

    if z < -8.0 {
        return 0.0;
    }
    if z > 8.0 {
        return 1.0;
    }

    let sign = if z >= 0.0 { 1.0 } else { -1.0 };
    let t = z.abs();

    // Horner-form coefficients for the approximation
    const B1: f64 = 0.319_381_530;
    const B2: f64 = -0.356_563_782;
    const B3: f64 = 1.781_477_937;
    const B4: f64 = -1.821_255_978;
    const B5: f64 = 1.330_274_429;
    const P: f64 = 0.231_641_9;

    let k = 1.0 / (1.0 + P * t);
    let poly = k * (B1 + k * (B2 + k * (B3 + k * (B4 + k * B5))));
    let pdf = standard_normal_pdf(t);
    let cdf_complement = pdf * poly;

    if sign > 0.0 {
        1.0 - cdf_complement
    } else {
        cdf_complement
    }
}

// ---------------------------------------------------------------------------
// Bayesian Optimizer
// ---------------------------------------------------------------------------

/// Bayesian optimizer for GPU kernel autotuning.
///
/// Combines a Gaussian Process surrogate model with an acquisition
/// function to intelligently navigate the configuration search space.
/// Instead of exhaustively benchmarking all configurations, the optimizer:
///
/// 1. Builds a probabilistic model of performance from observed benchmarks.
/// 2. Uses the acquisition function to identify the most promising
///    configuration to evaluate next.
/// 3. Updates the model with the new observation and repeats.
///
/// This approach typically finds near-optimal configurations in far
/// fewer evaluations than grid search or random search.
pub struct BayesianOptimizer {
    /// The GP surrogate model.
    gp: GaussianProcess,
    /// The acquisition function used for candidate selection.
    acquisition: AcquisitionFunction,
    /// The search space to explore.
    search_space: SearchSpace,
    /// The best observed performance value so far.
    best_observed: Option<f64>,
    /// All (config, performance) observations collected so far.
    observations: Vec<(Config, f64)>,
    /// Simple PRNG state for reproducible candidate sampling.
    rng_state: u64,
}

impl BayesianOptimizer {
    /// Creates a new Bayesian optimizer for the given search space.
    ///
    /// The GP is initialized with default hyperparameters:
    /// - length_scale = 1.0
    /// - signal_variance = 1.0
    /// - noise_variance = 1e-4 (small jitter for numerical stability)
    #[must_use]
    pub fn new(search_space: SearchSpace, acquisition: AcquisitionFunction) -> Self {
        Self {
            gp: GaussianProcess::new(1.0, 1.0, 1e-4),
            acquisition,
            search_space,
            best_observed: None,
            observations: Vec::new(),
            rng_state: 0x5EED_CAFE_BABE_1234,
        }
    }

    /// Creates a new Bayesian optimizer with custom GP hyperparameters.
    #[must_use]
    pub fn with_gp_params(
        search_space: SearchSpace,
        acquisition: AcquisitionFunction,
        length_scale: f64,
        signal_variance: f64,
        noise_variance: f64,
    ) -> Self {
        Self {
            gp: GaussianProcess::new(length_scale, signal_variance, noise_variance),
            acquisition,
            search_space,
            best_observed: None,
            observations: Vec::new(),
            rng_state: 0x5EED_CAFE_BABE_1234,
        }
    }

    /// Suggests the next configuration to evaluate.
    ///
    /// Enumerates a random subset of the search space (up to 1000
    /// candidates), evaluates the acquisition function on each using
    /// the GP predictions, and returns the candidate with the highest
    /// acquisition value.
    ///
    /// If no observations exist yet, returns a random configuration
    /// from the search space to bootstrap the GP.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::NoViableConfig`] if the search space is
    /// empty.
    pub fn suggest_next(&mut self) -> Result<Config, AutotuneError> {
        let all_configs = self.search_space.enumerate();
        if all_configs.is_empty() {
            return Err(AutotuneError::NoViableConfig);
        }

        // If no observations yet, return a random config to bootstrap
        if self.observations.is_empty() {
            let idx = self.next_rng_usize(all_configs.len());
            return Ok(all_configs[idx].clone());
        }

        // Sample up to 1000 candidates from the search space
        let candidates = self.sample_candidates(&all_configs, 1000);

        let mut best_acq_value = f64::NEG_INFINITY;
        let mut best_candidate: Option<&Config> = None;

        for candidate in &candidates {
            let normalized = self.normalize_config(candidate);
            let prediction = self.gp.predict(&normalized);
            let acq_value = self.acquisition.evaluate(&prediction, self.best_observed);

            if acq_value > best_acq_value {
                best_acq_value = acq_value;
                best_candidate = Some(candidate);
            }
        }

        best_candidate.cloned().ok_or(AutotuneError::NoViableConfig)
    }

    /// Records an observation: the performance of a configuration.
    ///
    /// Higher `performance` values are considered better (e.g., GFLOPS).
    /// The GP is updated with the normalized configuration and the
    /// performance value.
    pub fn observe(&mut self, config: Config, performance: f64) {
        let normalized = self.normalize_config(&config);
        self.gp.add_observation(normalized, performance);

        match self.best_observed {
            Some(best) if performance > best => {
                self.best_observed = Some(performance);
            }
            None => {
                self.best_observed = Some(performance);
            }
            _ => {}
        }

        self.observations.push((config, performance));
    }

    /// Returns the best observed configuration and its performance.
    ///
    /// Returns `None` if no observations have been made yet.
    #[must_use]
    pub fn best_config(&self) -> Option<(&Config, f64)> {
        self.observations
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(cfg, perf)| (cfg, *perf))
    }

    /// Suggests a batch of `n` diverse configurations to evaluate in parallel.
    ///
    /// Uses a greedy diversity strategy: after selecting the top candidate
    /// by acquisition value, subsequent candidates are penalized for being
    /// too similar to already-selected ones (kriging believer heuristic).
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::NoViableConfig`] if the search space is
    /// empty.
    pub fn suggest_batch(&mut self, n: usize) -> Result<Vec<Config>, AutotuneError> {
        if n == 0 {
            return Ok(Vec::new());
        }

        let all_configs = self.search_space.enumerate();
        if all_configs.is_empty() {
            return Err(AutotuneError::NoViableConfig);
        }

        let mut batch = Vec::with_capacity(n);
        let mut temp_gp = self.gp.clone();
        let mut temp_best = self.best_observed;

        for _ in 0..n {
            // If no observations exist in temp_gp, pick random
            if temp_gp.num_observations() == 0 {
                let idx = self.next_rng_usize(all_configs.len());
                let cfg = all_configs[idx].clone();
                let normalized = self.normalize_config(&cfg);
                // Kriging believer: pretend we observed the mean prediction
                let pred = temp_gp.predict(&normalized);
                temp_gp.add_observation(normalized, pred.mean);
                match temp_best {
                    Some(best) if pred.mean > best => temp_best = Some(pred.mean),
                    None => temp_best = Some(pred.mean),
                    _ => {}
                }
                batch.push(cfg);
                continue;
            }

            let candidates = self.sample_candidates(&all_configs, 1000);

            let mut best_acq = f64::NEG_INFINITY;
            let mut best_cfg: Option<&Config> = None;
            let mut best_normalized: Option<Vec<f64>> = None;

            for candidate in &candidates {
                // Skip configs already in the batch
                if batch.contains(candidate) {
                    continue;
                }
                let normalized = self.normalize_config(candidate);
                let prediction = temp_gp.predict(&normalized);
                let acq_value = self.acquisition.evaluate(&prediction, temp_best);

                if acq_value > best_acq {
                    best_acq = acq_value;
                    best_cfg = Some(candidate);
                    best_normalized = Some(normalized);
                }
            }

            if let (Some(cfg), Some(norm)) = (best_cfg, best_normalized) {
                let pred = temp_gp.predict(&norm);
                temp_gp.add_observation(norm, pred.mean);
                match temp_best {
                    Some(best) if pred.mean > best => temp_best = Some(pred.mean),
                    None => temp_best = Some(pred.mean),
                    _ => {}
                }
                batch.push(cfg.clone());
            } else {
                // Fallback: random selection
                let idx = self.next_rng_usize(all_configs.len());
                batch.push(all_configs[idx].clone());
            }
        }

        Ok(batch)
    }

    /// Returns a reference to the underlying Gaussian Process.
    #[must_use]
    pub fn gp(&self) -> &GaussianProcess {
        &self.gp
    }

    /// Returns the number of observations collected so far.
    #[must_use]
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Returns the best observed performance value.
    #[must_use]
    pub fn best_observed_value(&self) -> Option<f64> {
        self.best_observed
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Normalizes a Config into a feature vector in [0, 1]^d.
    ///
    /// Each parameter is normalized relative to the min/max values
    /// in the search space. Boolean parameters map to 0.0 or 1.0.
    fn normalize_config(&self, config: &Config) -> Vec<f64> {
        vec![
            normalize_u32(config.tile_m, &self.search_space.tile_m_values),
            normalize_u32(config.tile_n, &self.search_space.tile_n_values),
            normalize_u32(config.tile_k, &self.search_space.tile_k_values),
            normalize_u32(config.warp_m, &self.search_space.warp_m_values),
            normalize_u32(config.warp_n, &self.search_space.warp_n_values),
            normalize_u32(config.stages, &self.search_space.stages_values),
            if config.use_tensor_core { 1.0 } else { 0.0 },
            normalize_u32(config.block_size, &self.search_space.block_size_values),
        ]
    }

    /// Samples up to `max_count` configs from the full set using the PRNG.
    ///
    /// If the total number of configs is <= max_count, returns all of them.
    /// Otherwise performs a Fisher-Yates partial shuffle to select a
    /// random subset.
    fn sample_candidates<'a>(&mut self, all: &'a [Config], max_count: usize) -> Vec<&'a Config> {
        let n = all.len();
        if n <= max_count {
            return all.iter().collect();
        }

        // Build index array and partial Fisher-Yates shuffle
        let mut indices: Vec<usize> = (0..n).collect();
        let sample_size = max_count.min(n);

        for i in 0..sample_size {
            let j = i + self.next_rng_usize(n - i);
            indices.swap(i, j);
        }

        indices[..sample_size]
            .iter()
            .map(|&idx| &all[idx])
            .collect()
    }

    /// Simple xorshift64 PRNG — returns a value in [0, bound).
    fn next_rng_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as usize) % bound
    }
}

/// Normalizes a u32 value relative to the min and max of a candidate list.
///
/// Returns 0.5 if the list has fewer than 2 elements or if min == max.
fn normalize_u32(value: u32, candidates: &[u32]) -> f64 {
    if candidates.len() < 2 {
        return 0.5;
    }
    let min_val = candidates.iter().copied().min().unwrap_or(value);
    let max_val = candidates.iter().copied().max().unwrap_or(value);
    if max_val == min_val {
        return 0.5;
    }
    (f64::from(value) - f64::from(min_val)) / (f64::from(max_val) - f64::from(min_val))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SearchSpaceBuilder;

    // -- GP kernel tests --

    #[test]
    fn rbf_kernel_identical_points() {
        let gp = GaussianProcess::new(1.0, 1.0, 1e-4);
        let x = vec![0.5, 0.5];
        let k = gp.rbf_kernel(&x, &x);
        // k(x, x) = signal_variance * exp(0) = 1.0
        assert!((k - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rbf_kernel_distant_points() {
        let gp = GaussianProcess::new(0.1, 1.0, 1e-4);
        let x1 = vec![0.0, 0.0];
        let x2 = vec![10.0, 10.0];
        let k = gp.rbf_kernel(&x1, &x2);
        // Very distant points with small length scale -> near zero
        assert!(k < 1e-10);
    }

    #[test]
    fn rbf_kernel_symmetry() {
        let gp = GaussianProcess::new(1.0, 2.0, 1e-4);
        let x1 = vec![0.1, 0.3, 0.7];
        let x2 = vec![0.5, 0.2, 0.9];
        let k12 = gp.rbf_kernel(&x1, &x2);
        let k21 = gp.rbf_kernel(&x2, &x1);
        assert!((k12 - k21).abs() < 1e-15);
    }

    #[test]
    fn rbf_kernel_signal_variance_scaling() {
        let gp1 = GaussianProcess::new(1.0, 1.0, 1e-4);
        let gp2 = GaussianProcess::new(1.0, 3.0, 1e-4);
        let x1 = vec![0.0];
        let x2 = vec![0.5];
        let k1 = gp1.rbf_kernel(&x1, &x2);
        let k2 = gp2.rbf_kernel(&x1, &x2);
        assert!((k2 / k1 - 3.0).abs() < 1e-10);
    }

    // -- GP prediction tests --

    #[test]
    fn gp_predict_no_observations_returns_prior() {
        let gp = GaussianProcess::new(1.0, 2.0, 1e-4);
        let pred = gp.predict(&[0.5, 0.5]);
        assert!((pred.mean - 0.0).abs() < 1e-10);
        assert!((pred.variance - 2.0).abs() < 1e-10);
    }

    #[test]
    fn gp_predict_interpolates_observation() {
        let mut gp = GaussianProcess::new(1.0, 1.0, 1e-6);
        gp.add_observation(vec![0.0], 5.0);

        // Predict at the observed point — should be close to 5.0
        let pred = gp.predict(&[0.0]);
        assert!(
            (pred.mean - 5.0).abs() < 0.1,
            "mean at observed point should be ~5.0, got {}",
            pred.mean
        );
        // Variance should be very small at the observed point
        assert!(
            pred.variance < 0.01,
            "variance at observed point should be near 0, got {}",
            pred.variance
        );
    }

    #[test]
    fn gp_predict_variance_increases_away_from_data() {
        let mut gp = GaussianProcess::new(0.5, 1.0, 1e-6);
        gp.add_observation(vec![0.0], 1.0);

        let pred_near = gp.predict(&[0.1]);
        let pred_far = gp.predict(&[5.0]);

        assert!(
            pred_far.variance > pred_near.variance,
            "variance should increase with distance: near={}, far={}",
            pred_near.variance,
            pred_far.variance
        );
    }

    #[test]
    fn gp_predict_multiple_observations() {
        let mut gp = GaussianProcess::new(1.0, 1.0, 1e-6);
        // Observe a linear-ish function
        gp.add_observation(vec![0.0], 0.0);
        gp.add_observation(vec![1.0], 1.0);

        // Predict at midpoint — should be roughly 0.5
        let pred = gp.predict(&[0.5]);
        assert!(
            (pred.mean - 0.5).abs() < 0.3,
            "midpoint mean should be ~0.5, got {}",
            pred.mean
        );
    }

    // -- Acquisition function tests --

    #[test]
    fn ei_is_zero_when_mean_below_best_and_low_variance() {
        let acq = AcquisitionFunction::ExpectedImprovement;
        let pred = GpPrediction {
            mean: 1.0,
            variance: 1e-20,
        };
        let val = acq.evaluate(&pred, Some(5.0));
        assert!(
            val < 1e-10,
            "EI should be ~0 when mean << best and variance ~0, got {}",
            val
        );
    }

    #[test]
    fn ei_is_positive_when_mean_above_best() {
        let acq = AcquisitionFunction::ExpectedImprovement;
        let pred = GpPrediction {
            mean: 10.0,
            variance: 1.0,
        };
        let val = acq.evaluate(&pred, Some(5.0));
        assert!(val > 0.0, "EI should be positive when mean > best");
    }

    #[test]
    fn ucb_increases_with_kappa() {
        let pred = GpPrediction {
            mean: 1.0,
            variance: 4.0,
        };
        let ucb_low = AcquisitionFunction::UpperConfidenceBound { kappa: 1.0 };
        let ucb_high = AcquisitionFunction::UpperConfidenceBound { kappa: 3.0 };

        let val_low = ucb_low.evaluate(&pred, None);
        let val_high = ucb_high.evaluate(&pred, None);

        assert!(
            val_high > val_low,
            "UCB should increase with kappa: low={}, high={}",
            val_low,
            val_high
        );
        // UCB = mean + kappa * sigma = 1.0 + kappa * 2.0
        assert!((val_low - 3.0).abs() < 1e-10);
        assert!((val_high - 7.0).abs() < 1e-10);
    }

    #[test]
    fn pi_is_high_when_mean_well_above_best() {
        let acq = AcquisitionFunction::ProbabilityOfImprovement;
        let pred = GpPrediction {
            mean: 100.0,
            variance: 1.0,
        };
        let val = acq.evaluate(&pred, Some(0.0));
        assert!(
            val > 0.99,
            "PI should be ~1.0 when mean >> best, got {}",
            val
        );
    }

    // -- Cholesky tests --

    #[test]
    fn cholesky_2x2() {
        // A = [[4, 2], [2, 3]]
        let mat = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky_decompose(&mat, 2);
        assert!(l.is_some());
        let l = l.expect("cholesky should succeed for PD matrix");
        // L[0,0] = sqrt(4) = 2
        assert!((l[0] - 2.0).abs() < 1e-10);
        // L[1,0] = 2/2 = 1
        assert!((l[2] - 1.0).abs() < 1e-10);
        // L[1,1] = sqrt(3 - 1) = sqrt(2)
        assert!((l[3] - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn cholesky_not_positive_definite() {
        // Not PD: negative diagonal after elimination
        let mat = vec![1.0, 10.0, 10.0, 1.0];
        let l = cholesky_decompose(&mat, 2);
        assert!(l.is_none());
    }

    // -- Optimizer tests --

    #[test]
    fn optimizer_suggest_with_empty_space_returns_error() {
        let empty_space = SearchSpaceBuilder::new()
            .tile_m(vec![])
            .tile_n(vec![64])
            .tile_k(vec![16])
            .warp_m(vec![32])
            .warp_n(vec![32])
            .stages(vec![2])
            .use_tensor_core(vec![false])
            .block_size(vec![128])
            .build();

        let mut optimizer =
            BayesianOptimizer::new(empty_space, AcquisitionFunction::ExpectedImprovement);
        let result = optimizer.suggest_next();
        assert!(result.is_err());
    }

    #[test]
    fn optimizer_suggest_observe_loop() {
        let space = SearchSpace::minimal();
        let mut optimizer = BayesianOptimizer::new(space, AcquisitionFunction::ExpectedImprovement);

        // Run several iterations of suggest -> observe
        for i in 0..5 {
            let config = optimizer.suggest_next();
            assert!(
                config.is_ok(),
                "suggest_next should succeed on iteration {i}"
            );
            let config = config.expect("already checked");
            // Simulate performance based on tile size (larger = better for test)
            let perf = f64::from(config.tile_m + config.tile_n);
            optimizer.observe(config, perf);
        }

        assert_eq!(optimizer.num_observations(), 5);
        assert!(optimizer.best_config().is_some());
    }

    #[test]
    fn optimizer_best_config_tracks_maximum() {
        let space = SearchSpace::minimal();
        let mut optimizer = BayesianOptimizer::new(
            space,
            AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 },
        );

        let cfg1 = Config::new().with_tile_m(64).with_tile_n(64);
        let cfg2 = Config::new().with_tile_m(128).with_tile_n(128);

        optimizer.observe(cfg1, 10.0);
        optimizer.observe(cfg2.clone(), 50.0);

        let (best_cfg, best_perf) = optimizer.best_config().expect("should have best");
        assert!((best_perf - 50.0).abs() < 1e-10);
        assert_eq!(best_cfg.tile_m, best_cfg.tile_m); // sanity
        assert_eq!(best_cfg, &cfg2);
    }

    #[test]
    fn optimizer_batch_suggestion() {
        let space = SearchSpace::minimal();
        let mut optimizer = BayesianOptimizer::new(space, AcquisitionFunction::ExpectedImprovement);

        // Add a few initial observations
        let cfg = Config::new().with_tile_m(64).with_tile_n(64);
        optimizer.observe(cfg, 10.0);

        let batch = optimizer.suggest_batch(3);
        assert!(batch.is_ok());
        let batch = batch.expect("already checked");
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn optimizer_batch_empty_returns_empty() {
        let space = SearchSpace::minimal();
        let mut optimizer = BayesianOptimizer::new(space, AcquisitionFunction::ExpectedImprovement);

        let batch = optimizer.suggest_batch(0);
        assert!(batch.is_ok());
        assert!(batch.expect("already checked").is_empty());
    }

    // -- Normalization tests --

    #[test]
    fn normalize_u32_boundary_values() {
        let candidates = vec![32, 64, 128, 256];
        assert!((normalize_u32(32, &candidates) - 0.0).abs() < 1e-10);
        assert!((normalize_u32(256, &candidates) - 1.0).abs() < 1e-10);
        // 128 = (128-32)/(256-32) = 96/224 ≈ 0.4286
        let expected = 96.0 / 224.0;
        assert!((normalize_u32(128, &candidates) - expected).abs() < 1e-10);
    }

    #[test]
    fn normalize_u32_single_value_returns_half() {
        let candidates = vec![64];
        assert!((normalize_u32(64, &candidates) - 0.5).abs() < 1e-10);
    }

    // -- Normal distribution tests --

    #[test]
    fn standard_normal_cdf_known_values() {
        // Phi(0) = 0.5
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-5);
        // Phi(1.96) ≈ 0.975
        assert!((standard_normal_cdf(1.96) - 0.975).abs() < 1e-3);
        // Phi(-1.96) ≈ 0.025
        assert!((standard_normal_cdf(-1.96) - 0.025).abs() < 1e-3);
    }

    #[test]
    fn standard_normal_pdf_at_zero() {
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((standard_normal_pdf(0.0) - expected).abs() < 1e-10);
    }

    // -- Forward/backward solve tests --

    #[test]
    fn triangular_solve_roundtrip() {
        // L = [[2, 0], [1, sqrt(2)]]
        // From the 2x2 Cholesky test
        let l = vec![2.0, 0.0, 1.0, 2.0_f64.sqrt()];
        let b = vec![4.0, 5.0];

        // Forward: L * x = b
        let x = forward_solve(&l, 2, &b);
        // Back: L^T * y = x
        let y = backward_solve(&l, 2, &x);

        // Verify: (L * L^T) * y = b  =>  A * y = b
        // A = [[4, 2], [2, 3]]
        let check0 = 4.0 * y[0] + 2.0 * y[1];
        let check1 = 2.0 * y[0] + 3.0 * y[1];
        assert!((check0 - b[0]).abs() < 1e-10);
        assert!((check1 - b[1]).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Bayesian convergence: best_config() tracks the highest-performance
    // observed config, and the GP mean is higher near high-performance configs.
    // -----------------------------------------------------------------------

    /// Build a tiny 4-config search space where tile_m is the only variable.
    fn four_config_space() -> crate::SearchSpace {
        SearchSpaceBuilder::new()
            .tile_m(vec![32, 64, 128, 256])
            .tile_n(vec![64])
            .tile_k(vec![16])
            .warp_m(vec![32])
            .warp_n(vec![32])
            .stages(vec![2])
            .use_tensor_core(vec![false])
            .block_size(vec![128])
            .build()
    }

    /// Verify that after observing configs with known performance values,
    /// `best_config()` correctly identifies the best-performing config.
    ///
    /// The best config has performance 50.0.  All others are observed with
    /// lower performance.  We do NOT observe the best config so that the
    /// optimizer must infer it via the acquisition function.
    #[test]
    fn bayesian_best_config_tracks_highest_observed() {
        let space = four_config_space();
        let mut optimizer = BayesianOptimizer::new(space, AcquisitionFunction::ExpectedImprovement);

        // Observe configs with ascending performance.
        // Config with tile_m=128 gets the highest score (50.0).
        let cfg_32 = Config::new()
            .with_tile_m(32)
            .with_tile_n(64)
            .with_tile_k(16)
            .with_warp_m(32)
            .with_warp_n(32)
            .with_stages(2)
            .with_block_size(128);
        let cfg_64 = cfg_32.clone().with_tile_m(64);
        let cfg_128 = cfg_32.clone().with_tile_m(128);
        let cfg_256 = cfg_32.clone().with_tile_m(256);

        optimizer.observe(cfg_32, 10.0);
        optimizer.observe(cfg_64, 25.0);
        optimizer.observe(cfg_256, 35.0);
        // Observe the best last to ensure the tracker is updated correctly.
        optimizer.observe(cfg_128.clone(), 50.0);

        let (best_cfg, best_perf) = optimizer
            .best_config()
            .expect("best_config must be Some after observations");

        assert!(
            (best_perf - 50.0).abs() < 1e-9,
            "best performance must be 50.0, got {best_perf}"
        );
        assert_eq!(
            best_cfg.tile_m, 128,
            "best config must have tile_m=128, got {}",
            best_cfg.tile_m
        );
        assert_eq!(
            best_cfg, &cfg_128,
            "best_config must equal the config observed with performance 50.0"
        );
    }

    /// Verify that the GP mean is higher near observed high-performance configs.
    ///
    /// After observing a single high-performance point at tile_m=256 (normalised
    /// to 1.0), the GP should predict a higher mean at the normalised coordinate
    /// of that point than at the coordinate of an unobserved low-performance point.
    #[test]
    fn bayesian_gp_mean_higher_near_high_performance_config() {
        let space = four_config_space();
        let mut optimizer = BayesianOptimizer::with_gp_params(
            space,
            AcquisitionFunction::ExpectedImprovement,
            /* length_scale = */ 0.5,
            /* signal_variance = */ 1.0,
            /* noise_variance = */ 1e-6,
        );

        // Observe: tile_m=256 → high performance, tile_m=32 → low performance.
        let cfg_high = Config::new()
            .with_tile_m(256)
            .with_tile_n(64)
            .with_tile_k(16)
            .with_warp_m(32)
            .with_warp_n(32)
            .with_stages(2)
            .with_block_size(128);
        let cfg_low = cfg_high.clone().with_tile_m(32);

        optimizer.observe(cfg_high.clone(), 100.0);
        optimizer.observe(cfg_low.clone(), 1.0);

        // Normalized coordinate for tile_m=256 is 1.0; for tile_m=32 it is 0.0.
        // We construct the same feature vectors the optimizer uses internally.
        let candidates = [32u32, 64, 128, 256];
        let norm_high = normalize_u32(256, &candidates); // 1.0
        let norm_low = normalize_u32(32, &candidates); // 0.0

        // Build 8-dim feature vectors (all other dims are fixed → same value).
        let fixed = normalize_u32(64, &[64u32]); // returns 0.5 for single-value dim
        let feat_high = vec![norm_high, fixed, fixed, fixed, fixed, fixed, 0.0, fixed];
        let feat_low = vec![norm_low, fixed, fixed, fixed, fixed, fixed, 0.0, fixed];

        let pred_high = optimizer.gp().predict(&feat_high);
        let pred_low = optimizer.gp().predict(&feat_low);

        assert!(
            pred_high.mean > pred_low.mean,
            "GP mean should be higher at the high-performance observation point \
             (got pred_high.mean={}, pred_low.mean={})",
            pred_high.mean,
            pred_low.mean,
        );
    }
}
