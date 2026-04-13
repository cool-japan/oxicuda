//! Simulated annealing search strategy for autotuning.
//!
//! Simulated annealing explores the search space by accepting worse
//! configurations with a probability that decreases over time (as the
//! "temperature" cools).  This allows escaping local minima early in
//! the search while converging toward the global optimum later.
//!
//! The implementation uses a simple xorshift64 PRNG to avoid external
//! dependencies while providing deterministic behavior for a given seed.
//!
//! # Algorithm
//!
//! 1. Start at a random configuration with high temperature.
//! 2. At each step, generate a **neighbor** by mutating one random
//!    dimension of the current configuration.
//! 3. If the neighbor is better, always accept it.
//! 4. If the neighbor is worse, accept it with probability
//!    `exp((new - old) / temperature)` (Metropolis criterion).
//! 5. After `iterations_per_temperature` steps, cool:
//!    `temperature *= cooling_rate`.
//! 6. Stop when `temperature < min_temperature`.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::simulated_annealing::{SimulatedAnnealing, SimulatedAnnealingConfig};
//! use oxicuda_autotune::SearchSpace;
//!
//! let sa_cfg = SimulatedAnnealingConfig {
//!     initial_temperature: 100.0,
//!     cooling_rate: 0.95,
//!     min_temperature: 0.01,
//!     iterations_per_temperature: 10,
//! };
//! let space = SearchSpace::minimal();
//! let mut sa = SimulatedAnnealing::new(space, sa_cfg);
//!
//! // Suggest-observe loop
//! let config = sa.suggest_next();
//! sa.observe(config, 42.0);
//! sa.step(); // cool down
//! ```

use crate::config::Config;
use crate::search_space::SearchSpace;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Xorshift64 — minimal deterministic PRNG
// ---------------------------------------------------------------------------

/// Advances a xorshift64 state and returns the new value.
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// Returns a random `f64` in `[0, 1)`.
fn rand_f64(state: &mut u64) -> f64 {
    let v = xorshift64(state);
    // Use 53 bits for double precision mantissa
    (v >> 11) as f64 / (1u64 << 53) as f64
}

/// Returns a random index in `[0, len)`.  Returns 0 if `len == 0`.
fn rand_index(state: &mut u64, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    (xorshift64(state) as usize) % len
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for the simulated annealing search.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingConfig {
    /// Starting temperature.  Higher values make the search more
    /// exploratory at the beginning.
    pub initial_temperature: f64,
    /// Multiplicative cooling factor applied after each temperature
    /// level: `T_new = T_old * cooling_rate`.  Must be in `(0, 1)`.
    pub cooling_rate: f64,
    /// The search terminates when the temperature drops below this
    /// threshold.
    pub min_temperature: f64,
    /// Number of suggest/observe iterations to perform at each
    /// temperature level before cooling.
    pub iterations_per_temperature: usize,
}

impl Default for SimulatedAnnealingConfig {
    fn default() -> Self {
        Self {
            initial_temperature: 100.0,
            cooling_rate: 0.95,
            min_temperature: 0.01,
            iterations_per_temperature: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Simulated Annealing engine
// ---------------------------------------------------------------------------

/// Simulated annealing optimizer for GPU kernel autotuning.
///
/// Maintains the current and best configurations along with their
/// measured performance values.  Higher performance values are
/// considered better (e.g., GFLOPS, throughput).
pub struct SimulatedAnnealing {
    config: SimulatedAnnealingConfig,
    search_space: SearchSpace,
    current: Option<(Config, f64)>,
    best: Option<(Config, f64)>,
    temperature: f64,
    rng_state: u64,
}

/// The number of mutable dimensions in a `Config` (excluding `extra`).
const NUM_DIMENSIONS: usize = 8;

impl SimulatedAnnealing {
    /// Creates a new simulated annealing optimizer.
    ///
    /// The initial temperature is taken from `config.initial_temperature`.
    /// The RNG is seeded deterministically so that runs are reproducible.
    #[must_use]
    pub fn new(search_space: SearchSpace, config: SimulatedAnnealingConfig) -> Self {
        let temperature = config.initial_temperature;
        Self {
            config,
            search_space,
            current: None,
            best: None,
            temperature,
            rng_state: 0xDEAD_BEEF_CAFE_1234,
        }
    }

    /// Creates a new optimizer with a specific RNG seed for
    /// reproducibility.
    #[must_use]
    pub fn with_seed(
        search_space: SearchSpace,
        config: SimulatedAnnealingConfig,
        seed: u64,
    ) -> Self {
        let temperature = config.initial_temperature;
        // Ensure seed is never 0 (xorshift64 fixed point)
        let rng_state = if seed == 0 { 1 } else { seed };
        Self {
            config,
            search_space,
            current: None,
            best: None,
            temperature,
            rng_state,
        }
    }

    /// Suggests the next configuration to evaluate.
    ///
    /// If no configuration has been observed yet, a random configuration
    /// from the search space is returned.  Otherwise, a neighbor of the
    /// current configuration is generated by mutating one random
    /// dimension to another valid value from the search space.
    pub fn suggest_next(&mut self) -> Config {
        match &self.current {
            None => self.random_config(),
            Some((current, _)) => {
                let base = current.clone();
                self.neighbor(&base)
            }
        }
    }

    /// Records the measured performance of a configuration and applies
    /// the Metropolis acceptance criterion.
    ///
    /// `performance` should be a higher-is-better metric (e.g., GFLOPS).
    /// The configuration is always accepted if it improves on the current
    /// solution.  Otherwise it is accepted with probability
    /// `exp((new_perf - current_perf) / temperature)`.
    pub fn observe(&mut self, config: Config, performance: f64) {
        // Update best if this is the best we have seen
        let is_new_best = match &self.best {
            None => true,
            Some((_, best_perf)) => performance > *best_perf,
        };
        if is_new_best {
            self.best = Some((config.clone(), performance));
        }

        // Metropolis acceptance criterion
        let accept = match &self.current {
            None => true,
            Some((_, current_perf)) => {
                if performance >= *current_perf {
                    true
                } else {
                    let delta = performance - current_perf;
                    if self.temperature <= 0.0 {
                        false
                    } else {
                        let prob = (delta / self.temperature).exp();
                        rand_f64(&mut self.rng_state) < prob
                    }
                }
            }
        };

        if accept {
            self.current = Some((config, performance));
        }
    }

    /// Returns the best configuration found so far and its performance.
    #[must_use]
    pub fn best_config(&self) -> Option<(&Config, f64)> {
        self.best.as_ref().map(|(c, p)| (c, *p))
    }

    /// Returns the current temperature.
    #[must_use]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Cools the temperature by one step: `T *= cooling_rate`.
    pub fn step(&mut self) {
        self.temperature *= self.config.cooling_rate;
    }

    /// Returns `true` if the temperature has dropped below the minimum.
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.temperature < self.config.min_temperature
    }

    /// Returns the acceptance probability for a given performance delta
    /// at the current temperature.
    ///
    /// This is exposed for testing and diagnostics.
    #[must_use]
    pub fn acceptance_probability(&self, delta: f64) -> f64 {
        if delta >= 0.0 {
            1.0
        } else if self.temperature <= 0.0 {
            0.0
        } else {
            (delta / self.temperature).exp()
        }
    }

    // -- private helpers ----------------------------------------------------

    /// Generates a random configuration from the search space.
    fn random_config(&mut self) -> Config {
        let tile_m = self.random_from_slice(&self.search_space.tile_m_values.clone());
        let tile_n = self.random_from_slice(&self.search_space.tile_n_values.clone());
        let tile_k = self.random_from_slice(&self.search_space.tile_k_values.clone());
        let warp_m = self.random_from_slice(&self.search_space.warp_m_values.clone());
        let warp_n = self.random_from_slice(&self.search_space.warp_n_values.clone());
        let stages = self.random_from_slice(&self.search_space.stages_values.clone());
        let use_tc_vals = self.search_space.use_tensor_core_values.clone();
        let use_tensor_core = self.random_from_bool_slice(&use_tc_vals);
        let block_size = self.random_from_slice(&self.search_space.block_size_values.clone());

        Config {
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            stages,
            use_tensor_core,
            block_size,
            extra: HashMap::new(),
        }
    }

    /// Generates a neighbor of `base` by mutating exactly one random
    /// dimension to another valid value from the search space.
    fn neighbor(&mut self, base: &Config) -> Config {
        let mut cfg = base.clone();
        let dim = rand_index(&mut self.rng_state, NUM_DIMENSIONS);

        match dim {
            0 => cfg.tile_m = self.random_from_slice(&self.search_space.tile_m_values.clone()),
            1 => cfg.tile_n = self.random_from_slice(&self.search_space.tile_n_values.clone()),
            2 => cfg.tile_k = self.random_from_slice(&self.search_space.tile_k_values.clone()),
            3 => cfg.warp_m = self.random_from_slice(&self.search_space.warp_m_values.clone()),
            4 => cfg.warp_n = self.random_from_slice(&self.search_space.warp_n_values.clone()),
            5 => cfg.stages = self.random_from_slice(&self.search_space.stages_values.clone()),
            6 => {
                let vals = self.search_space.use_tensor_core_values.clone();
                cfg.use_tensor_core = self.random_from_bool_slice(&vals);
            }
            7 => {
                cfg.block_size =
                    self.random_from_slice(&self.search_space.block_size_values.clone())
            }
            _ => {} // unreachable with NUM_DIMENSIONS = 8
        }
        cfg
    }

    /// Picks a random value from a `u32` slice.  Returns 0 if empty.
    fn random_from_slice(&mut self, values: &[u32]) -> u32 {
        if values.is_empty() {
            return 0;
        }
        let idx = rand_index(&mut self.rng_state, values.len());
        values[idx]
    }

    /// Picks a random value from a `bool` slice.  Returns `false` if empty.
    fn random_from_bool_slice(&mut self, values: &[bool]) -> bool {
        if values.is_empty() {
            return false;
        }
        let idx = rand_index(&mut self.rng_state, values.len());
        values[idx]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SearchSpace;

    fn test_space() -> SearchSpace {
        SearchSpace::minimal()
    }

    fn test_sa_config() -> SimulatedAnnealingConfig {
        SimulatedAnnealingConfig {
            initial_temperature: 100.0,
            cooling_rate: 0.9,
            min_temperature: 0.01,
            iterations_per_temperature: 5,
        }
    }

    #[test]
    fn config_creation_default() {
        let cfg = SimulatedAnnealingConfig::default();
        assert!((cfg.initial_temperature - 100.0).abs() < f64::EPSILON);
        assert!((cfg.cooling_rate - 0.95).abs() < f64::EPSILON);
        assert!((cfg.min_temperature - 0.01).abs() < f64::EPSILON);
        assert_eq!(cfg.iterations_per_temperature, 10);
    }

    #[test]
    fn new_sets_initial_temperature() {
        let sa = SimulatedAnnealing::new(test_space(), test_sa_config());
        assert!((sa.temperature() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn suggest_returns_valid_config() {
        let mut sa = SimulatedAnnealing::new(test_space(), test_sa_config());
        let cfg = sa.suggest_next();
        let space = test_space();
        assert!(space.tile_m_values.contains(&cfg.tile_m));
        assert!(space.tile_n_values.contains(&cfg.tile_n));
        assert!(space.tile_k_values.contains(&cfg.tile_k));
        assert!(space.block_size_values.contains(&cfg.block_size));
    }

    #[test]
    fn observe_sets_best() {
        let mut sa = SimulatedAnnealing::new(test_space(), test_sa_config());
        let cfg = sa.suggest_next();
        sa.observe(cfg.clone(), 50.0);
        let (best, perf) = sa.best_config().expect("should have best");
        assert_eq!(best, &cfg);
        assert!((perf - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn better_config_always_accepted() {
        let mut sa = SimulatedAnnealing::new(test_space(), test_sa_config());
        let c1 = sa.suggest_next();
        sa.observe(c1, 10.0);

        let c2 = sa.suggest_next();
        sa.observe(c2.clone(), 100.0);

        // The best should be the second config
        let (best, perf) = sa.best_config().expect("should have best");
        assert_eq!(best, &c2);
        assert!((perf - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn temperature_cooling() {
        let mut sa = SimulatedAnnealing::new(test_space(), test_sa_config());
        let initial = sa.temperature();
        sa.step();
        let after = sa.temperature();
        assert!((after - initial * 0.9).abs() < 1e-10);
        assert!(after < initial);
    }

    #[test]
    fn temperature_cools_repeatedly() {
        let mut sa = SimulatedAnnealing::new(test_space(), test_sa_config());
        for _ in 0..100 {
            sa.step();
        }
        assert!(sa.temperature() < 0.01);
        assert!(sa.is_frozen());
    }

    #[test]
    fn acceptance_probability_better_is_one() {
        let sa = SimulatedAnnealing::new(test_space(), test_sa_config());
        assert!((sa.acceptance_probability(10.0) - 1.0).abs() < f64::EPSILON);
        assert!((sa.acceptance_probability(0.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn acceptance_probability_worse_decreases_with_temperature() {
        let cfg = test_sa_config();
        let sa_hot = SimulatedAnnealing::new(test_space(), cfg.clone());

        let mut cold_cfg = cfg;
        cold_cfg.initial_temperature = 1.0;
        let sa_cold = SimulatedAnnealing::new(test_space(), cold_cfg);

        let delta = -10.0;
        let prob_hot = sa_hot.acceptance_probability(delta);
        let prob_cold = sa_cold.acceptance_probability(delta);
        assert!(prob_hot > prob_cold);
    }

    #[test]
    fn acceptance_probability_zero_temp() {
        let mut cfg = test_sa_config();
        cfg.initial_temperature = 0.0;
        let sa = SimulatedAnnealing::new(test_space(), cfg);
        assert!((sa.acceptance_probability(-5.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn neighbor_differs_by_at_most_one_dimension() {
        let mut sa = SimulatedAnnealing::with_seed(test_space(), test_sa_config(), 42);
        let c1 = sa.suggest_next();
        sa.observe(c1.clone(), 50.0);

        // Generate many neighbors and check each differs in at most one
        // core dimension from c1
        for _ in 0..20 {
            let n = sa.suggest_next();
            let mut diffs = 0u32;
            if n.tile_m != c1.tile_m {
                diffs += 1;
            }
            if n.tile_n != c1.tile_n {
                diffs += 1;
            }
            if n.tile_k != c1.tile_k {
                diffs += 1;
            }
            if n.warp_m != c1.warp_m {
                diffs += 1;
            }
            if n.warp_n != c1.warp_n {
                diffs += 1;
            }
            if n.stages != c1.stages {
                diffs += 1;
            }
            if n.use_tensor_core != c1.use_tensor_core {
                diffs += 1;
            }
            if n.block_size != c1.block_size {
                diffs += 1;
            }
            assert!(
                diffs <= 1,
                "neighbor should differ in at most 1 dimension, got {diffs}"
            );
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut sa1 = SimulatedAnnealing::with_seed(test_space(), test_sa_config(), 42);
        let mut sa2 = SimulatedAnnealing::with_seed(test_space(), test_sa_config(), 42);

        for _ in 0..10 {
            let c1 = sa1.suggest_next();
            let c2 = sa2.suggest_next();
            assert_eq!(c1, c2);
            sa1.observe(c1, 50.0);
            sa2.observe(c2, 50.0);
        }
    }

    #[test]
    fn suggest_observe_cycle_many_iterations() {
        let mut sa = SimulatedAnnealing::new(test_space(), test_sa_config());
        // Run many iterations — should not panic
        for i in 0..100 {
            let cfg = sa.suggest_next();
            // Simulate a simple performance function
            let perf = 100.0 - (cfg.tile_m as f64 - 96.0).abs();
            sa.observe(cfg, perf);
            if i % 5 == 0 {
                sa.step();
            }
        }
        assert!(sa.best_config().is_some());
    }

    #[test]
    fn is_frozen_works() {
        let mut cfg = test_sa_config();
        cfg.initial_temperature = 0.001;
        cfg.min_temperature = 0.01;
        let sa = SimulatedAnnealing::new(test_space(), cfg);
        assert!(sa.is_frozen());
    }

    // -----------------------------------------------------------------------
    // SA convergence on a synthetic 2-D search space.
    //
    // Performance function: f(tile_m, tile_n) = -(|tile_m - 128| + |tile_n - 128|)
    // Peak at (128, 128) where f = 0.0.
    // -----------------------------------------------------------------------

    /// Build the 2-D synthetic search space.
    fn synthetic_2d_space() -> SearchSpace {
        SearchSpace {
            tile_m_values: vec![32, 64, 128, 256],
            tile_n_values: vec![32, 64, 128, 256],
            tile_k_values: vec![16],
            warp_m_values: vec![32],
            warp_n_values: vec![32],
            stages_values: vec![2],
            use_tensor_core_values: vec![false],
            block_size_values: vec![128],
        }
    }

    /// Evaluate the synthetic performance function (higher = better).
    fn synthetic_perf(tile_m: u32, tile_n: u32) -> f64 {
        -(((tile_m as i32 - 128).abs() + (tile_n as i32 - 128).abs()) as f64)
    }

    /// SA must converge to (128, 128) on the synthetic 2-D function within
    /// 200 steps.  We use a fixed seed so the test is deterministic.
    #[test]
    fn sa_convergence_finds_optimum_on_synthetic_function() {
        let space = synthetic_2d_space();
        let sa_cfg = SimulatedAnnealingConfig {
            initial_temperature: 200.0,
            cooling_rate: 0.97,
            min_temperature: 0.1,
            iterations_per_temperature: 10,
        };

        let mut sa = SimulatedAnnealing::with_seed(space, sa_cfg, 42);

        // Run 200 suggest–observe steps with manual cooling every 10 steps.
        for step in 0..200 {
            let cfg = sa.suggest_next();
            let perf = synthetic_perf(cfg.tile_m, cfg.tile_n);
            sa.observe(cfg, perf);
            if step % 10 == 9 {
                sa.step();
            }
        }

        let (best, best_perf) = sa
            .best_config()
            .expect("SA should have found at least one config");

        assert_eq!(
            best.tile_m, 128,
            "SA should converge to tile_m=128, got {}",
            best.tile_m
        );
        assert_eq!(
            best.tile_n, 128,
            "SA should converge to tile_n=128, got {}",
            best.tile_n
        );
        assert!(
            (best_perf - 0.0).abs() < 1e-9,
            "Peak performance at (128,128) should be 0.0, got {best_perf}"
        );
    }

    /// The default initial temperature (100.0) is reasonable for a GEMM
    /// search space — assert it is >= 100.0.
    #[test]
    fn sa_gemm_space_t0_reasonable() {
        let space = SearchSpace::gemm_default();
        let cfg = SimulatedAnnealingConfig::default();
        let sa = SimulatedAnnealing::new(space, cfg);
        assert!(
            sa.temperature() >= 100.0,
            "Initial temperature for GEMM space should be >= 100.0, got {}",
            sa.temperature()
        );
    }
}
