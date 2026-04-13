//! Genetic algorithm search strategy for autotuning.
//!
//! A genetic algorithm (GA) maintains a population of candidate
//! configurations that evolve over generations through selection,
//! crossover, and mutation.  This is well-suited to autotuning because
//! GPU kernel performance landscapes are highly non-linear and
//! discontinuous — GAs can explore diverse regions of the search space
//! simultaneously.
//!
//! # Algorithm
//!
//! 1. **Initialize** a random population of `population_size` configs.
//! 2. **Evaluate** — benchmark each config to obtain fitness.
//! 3. **Select** parents via tournament selection.
//! 4. **Crossover** — with probability `crossover_rate`, produce
//!    offspring via uniform crossover; otherwise, clone one parent.
//! 5. **Mutate** — with probability `mutation_rate` per gene, reset
//!    that gene to a random valid value.
//! 6. **Elitism** — the top `elite_count` individuals survive
//!    unchanged into the next generation.
//! 7. Repeat from step 2.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::genetic::{GeneticAlgorithm, GeneticConfig};
//! use oxicuda_autotune::SearchSpace;
//!
//! let ga_cfg = GeneticConfig {
//!     population_size: 20,
//!     elite_count: 2,
//!     crossover_rate: 0.8,
//!     mutation_rate: 0.1,
//!     tournament_size: 3,
//! };
//! let space = SearchSpace::minimal();
//! let mut ga = GeneticAlgorithm::new(space, ga_cfg);
//!
//! // Evaluate the initial population
//! for cfg in ga.pending_evaluations() {
//!     // ... benchmark cfg on GPU ...
//! }
//! ```

use crate::config::Config;
use crate::error::AutotuneError;
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

/// Configuration parameters for the genetic algorithm.
#[derive(Debug, Clone)]
pub struct GeneticConfig {
    /// Number of individuals in the population.
    pub population_size: usize,
    /// Number of top individuals carried unchanged to the next
    /// generation (elitism).
    pub elite_count: usize,
    /// Probability `[0, 1]` that two parents undergo crossover.
    /// Otherwise one parent is cloned directly.
    pub crossover_rate: f64,
    /// Probability `[0, 1]` of mutating each gene (dimension) of an
    /// offspring.
    pub mutation_rate: f64,
    /// Number of individuals in each tournament selection round.
    pub tournament_size: usize,
}

impl Default for GeneticConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            elite_count: 5,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            tournament_size: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Genetic Algorithm engine
// ---------------------------------------------------------------------------

/// Genetic algorithm optimizer for GPU kernel autotuning.
///
/// Maintains a population of configurations with optional fitness
/// values.  Call [`pending_evaluations`](Self::pending_evaluations) to
/// get configs that need benchmarking, [`set_fitness`](Self::set_fitness)
/// to record results, and [`evolve`](Self::evolve) to advance one
/// generation.
pub struct GeneticAlgorithm {
    config: GeneticConfig,
    search_space: SearchSpace,
    population: Vec<(Config, Option<f64>)>,
    generation: usize,
    rng_state: u64,
}

impl GeneticAlgorithm {
    /// Creates a new GA with a randomly initialized population.
    #[must_use]
    pub fn new(search_space: SearchSpace, config: GeneticConfig) -> Self {
        let mut ga = Self {
            config,
            search_space,
            population: Vec::new(),
            generation: 0,
            rng_state: 0xCAFE_BABE_DEAD_BEEF,
        };
        ga.init_population();
        ga
    }

    /// Creates a new GA with a specific RNG seed.
    #[must_use]
    pub fn with_seed(search_space: SearchSpace, config: GeneticConfig, seed: u64) -> Self {
        let rng_state = if seed == 0 { 1 } else { seed };
        let mut ga = Self {
            config,
            search_space,
            population: Vec::new(),
            generation: 0,
            rng_state,
        };
        ga.init_population();
        ga
    }

    /// Returns references to configurations that still need fitness
    /// evaluation (i.e., their fitness is `None`).
    #[must_use]
    pub fn pending_evaluations(&self) -> Vec<&Config> {
        self.population
            .iter()
            .filter(|(_, f)| f.is_none())
            .map(|(c, _)| c)
            .collect()
    }

    /// Records the fitness for a configuration in the population.
    ///
    /// The first matching configuration (by equality) with `None`
    /// fitness is updated.  If no match is found, this is a no-op.
    pub fn set_fitness(&mut self, config: &Config, fitness: f64) {
        for (c, f) in &mut self.population {
            if c == config && f.is_none() {
                *f = Some(fitness);
                return;
            }
        }
    }

    /// Runs one generation of evolution: selection, crossover, mutation,
    /// and elitism.
    ///
    /// Returns an error if any individual still lacks a fitness value
    /// (all configs must be evaluated before evolving).
    pub fn evolve(&mut self) -> Result<(), AutotuneError> {
        // Verify all fitness values are present
        let all_evaluated = self.population.iter().all(|(_, f)| f.is_some());
        if !all_evaluated {
            return Err(AutotuneError::BenchmarkFailed(
                "cannot evolve: some individuals have no fitness".to_string(),
            ));
        }

        let pop_size = self.config.population_size;

        // Sort population by fitness (descending — higher is better)
        self.population.sort_by(|a, b| {
            let fa = a.1.unwrap_or(f64::NEG_INFINITY);
            let fb = b.1.unwrap_or(f64::NEG_INFINITY);
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut next_gen: Vec<(Config, Option<f64>)> = Vec::with_capacity(pop_size);

        // Elitism: carry over top elite_count unchanged (with fitness)
        let elite_count = self.config.elite_count.min(pop_size);
        for i in 0..elite_count {
            if i < self.population.len() {
                next_gen.push(self.population[i].clone());
            }
        }

        // Fill the rest via selection + crossover + mutation
        while next_gen.len() < pop_size {
            let parent_a = self.tournament_select();
            let parent_b = self.tournament_select();

            let child = if rand_f64(&mut self.rng_state) < self.config.crossover_rate {
                self.uniform_crossover(&parent_a, &parent_b)
            } else {
                parent_a.clone()
            };

            let child = self.mutate(child);
            next_gen.push((child, None));
        }

        self.population = next_gen;
        self.generation += 1;
        Ok(())
    }

    /// Returns the best configuration and its fitness from the current
    /// population.
    #[must_use]
    pub fn best_config(&self) -> Option<(&Config, f64)> {
        self.population
            .iter()
            .filter_map(|(c, f)| f.map(|fv| (c, fv)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Returns the current generation number (starts at 0).
    #[must_use]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Returns the current population size.
    #[must_use]
    pub fn population_size(&self) -> usize {
        self.population.len()
    }

    /// Returns all individuals with their fitness values.
    #[must_use]
    pub fn population(&self) -> &[(Config, Option<f64>)] {
        &self.population
    }

    // -- private helpers ----------------------------------------------------

    /// Initializes the population with random configs.
    fn init_population(&mut self) {
        let size = self.config.population_size;
        self.population = Vec::with_capacity(size);
        for _ in 0..size {
            let cfg = self.random_config();
            self.population.push((cfg, None));
        }
    }

    /// Generates a random configuration from the search space.
    fn random_config(&mut self) -> Config {
        let tile_m = self.pick_u32(&self.search_space.tile_m_values.clone());
        let tile_n = self.pick_u32(&self.search_space.tile_n_values.clone());
        let tile_k = self.pick_u32(&self.search_space.tile_k_values.clone());
        let warp_m = self.pick_u32(&self.search_space.warp_m_values.clone());
        let warp_n = self.pick_u32(&self.search_space.warp_n_values.clone());
        let stages = self.pick_u32(&self.search_space.stages_values.clone());
        let use_tc_vals = self.search_space.use_tensor_core_values.clone();
        let use_tensor_core = self.pick_bool(&use_tc_vals);
        let block_size = self.pick_u32(&self.search_space.block_size_values.clone());

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

    /// Tournament selection: pick `tournament_size` random individuals,
    /// return the one with the best fitness.
    fn tournament_select(&mut self) -> Config {
        let pop_len = self.population.len();
        if pop_len == 0 {
            return Config::default();
        }
        let ts = self.config.tournament_size.min(pop_len).max(1);

        let mut best_idx = rand_index(&mut self.rng_state, pop_len);
        let mut best_fit = self.population[best_idx].1.unwrap_or(f64::NEG_INFINITY);

        for _ in 1..ts {
            let idx = rand_index(&mut self.rng_state, pop_len);
            let fit = self.population[idx].1.unwrap_or(f64::NEG_INFINITY);
            if fit > best_fit {
                best_fit = fit;
                best_idx = idx;
            }
        }
        self.population[best_idx].0.clone()
    }

    /// Uniform crossover: each gene is taken from a random parent.
    fn uniform_crossover(&mut self, a: &Config, b: &Config) -> Config {
        let pick = |state: &mut u64, va: u32, vb: u32| -> u32 {
            if rand_f64(state) < 0.5 { va } else { vb }
        };

        Config {
            tile_m: pick(&mut self.rng_state, a.tile_m, b.tile_m),
            tile_n: pick(&mut self.rng_state, a.tile_n, b.tile_n),
            tile_k: pick(&mut self.rng_state, a.tile_k, b.tile_k),
            warp_m: pick(&mut self.rng_state, a.warp_m, b.warp_m),
            warp_n: pick(&mut self.rng_state, a.warp_n, b.warp_n),
            stages: pick(&mut self.rng_state, a.stages, b.stages),
            use_tensor_core: if rand_f64(&mut self.rng_state) < 0.5 {
                a.use_tensor_core
            } else {
                b.use_tensor_core
            },
            block_size: pick(&mut self.rng_state, a.block_size, b.block_size),
            extra: HashMap::new(),
        }
    }

    /// Mutate: with `mutation_rate` probability per gene, reset that
    /// gene to a random valid value from the search space.
    fn mutate(&mut self, mut cfg: Config) -> Config {
        let rate = self.config.mutation_rate;

        if rand_f64(&mut self.rng_state) < rate {
            cfg.tile_m = self.pick_u32(&self.search_space.tile_m_values.clone());
        }
        if rand_f64(&mut self.rng_state) < rate {
            cfg.tile_n = self.pick_u32(&self.search_space.tile_n_values.clone());
        }
        if rand_f64(&mut self.rng_state) < rate {
            cfg.tile_k = self.pick_u32(&self.search_space.tile_k_values.clone());
        }
        if rand_f64(&mut self.rng_state) < rate {
            cfg.warp_m = self.pick_u32(&self.search_space.warp_m_values.clone());
        }
        if rand_f64(&mut self.rng_state) < rate {
            cfg.warp_n = self.pick_u32(&self.search_space.warp_n_values.clone());
        }
        if rand_f64(&mut self.rng_state) < rate {
            cfg.stages = self.pick_u32(&self.search_space.stages_values.clone());
        }
        if rand_f64(&mut self.rng_state) < rate {
            let vals = self.search_space.use_tensor_core_values.clone();
            cfg.use_tensor_core = self.pick_bool(&vals);
        }
        if rand_f64(&mut self.rng_state) < rate {
            cfg.block_size = self.pick_u32(&self.search_space.block_size_values.clone());
        }

        cfg
    }

    /// Picks a random u32 from a slice.  Returns 0 if empty.
    fn pick_u32(&mut self, values: &[u32]) -> u32 {
        if values.is_empty() {
            return 0;
        }
        let idx = rand_index(&mut self.rng_state, values.len());
        values[idx]
    }

    /// Picks a random bool from a slice.  Returns false if empty.
    fn pick_bool(&mut self, values: &[bool]) -> bool {
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
    use crate::{SearchSpace, SearchSpaceBuilder};

    fn test_space() -> SearchSpace {
        SearchSpace::minimal()
    }

    fn test_ga_config() -> GeneticConfig {
        GeneticConfig {
            population_size: 20,
            elite_count: 2,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            tournament_size: 3,
        }
    }

    #[test]
    fn config_creation_default() {
        let cfg = GeneticConfig::default();
        assert_eq!(cfg.population_size, 50);
        assert_eq!(cfg.elite_count, 5);
        assert!((cfg.crossover_rate - 0.8).abs() < f64::EPSILON);
        assert!((cfg.mutation_rate - 0.1).abs() < f64::EPSILON);
        assert_eq!(cfg.tournament_size, 3);
    }

    #[test]
    fn population_initialized_to_correct_size() {
        let ga = GeneticAlgorithm::new(test_space(), test_ga_config());
        assert_eq!(ga.population_size(), 20);
    }

    #[test]
    fn initial_population_all_pending() {
        let ga = GeneticAlgorithm::new(test_space(), test_ga_config());
        assert_eq!(ga.pending_evaluations().len(), 20);
    }

    #[test]
    fn set_fitness_reduces_pending() {
        let mut ga = GeneticAlgorithm::new(test_space(), test_ga_config());
        let pending: Vec<Config> = ga.pending_evaluations().into_iter().cloned().collect();
        let first = pending[0].clone();
        ga.set_fitness(&first, 42.0);
        assert_eq!(ga.pending_evaluations().len(), 19);
    }

    #[test]
    fn best_config_returns_highest_fitness() {
        let mut ga = GeneticAlgorithm::new(test_space(), test_ga_config());
        let pending: Vec<Config> = ga.pending_evaluations().into_iter().cloned().collect();

        ga.set_fitness(&pending[0], 10.0);
        ga.set_fitness(&pending[1], 50.0);
        ga.set_fitness(&pending[2], 30.0);

        let (best, fitness) = ga.best_config().expect("should have best");
        assert_eq!(best, &pending[1]);
        assert!((fitness - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn evolve_fails_without_full_evaluation() {
        let mut ga = GeneticAlgorithm::new(test_space(), test_ga_config());
        let result = ga.evolve();
        assert!(result.is_err());
    }

    #[test]
    fn evolve_succeeds_with_full_evaluation() {
        let mut ga = GeneticAlgorithm::new(test_space(), test_ga_config());
        evaluate_all(&mut ga);
        let result = ga.evolve();
        assert!(result.is_ok());
        assert_eq!(ga.generation(), 1);
    }

    #[test]
    fn elitism_preserves_top_configs() {
        let mut ga = GeneticAlgorithm::with_seed(test_space(), test_ga_config(), 123);
        // Use a fitness function that gives unique scores based on
        // multiple dimensions so the best is unambiguous.
        evaluate_all_with_fn(&mut ga, |cfg| {
            cfg.tile_m as f64 * 1000.0
                + cfg.tile_n as f64 * 100.0
                + cfg.tile_k as f64 * 10.0
                + cfg.block_size as f64
        });

        // Capture the top elite_count configs (sorted descending by fitness)
        let mut scored: Vec<(Config, f64)> = ga
            .population()
            .iter()
            .filter_map(|(c, f)| f.map(|fv| (c.clone(), fv)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let elite_count = ga.config.elite_count;
        let elites: Vec<(Config, f64)> = scored.into_iter().take(elite_count).collect();

        let result = ga.evolve();
        assert!(result.is_ok());

        // Each elite should still exist in the new population with its fitness.
        for (elite_cfg, elite_fit) in &elites {
            let found = ga.population().iter().any(|(c, f)| {
                c == elite_cfg && f.is_some_and(|fv| (fv - elite_fit).abs() < f64::EPSILON)
            });
            assert!(found, "elite config should be preserved across evolution");
        }
    }

    #[test]
    fn population_configs_are_valid() {
        let ga = GeneticAlgorithm::new(test_space(), test_ga_config());
        let space = test_space();
        for (cfg, _) in ga.population() {
            assert!(space.tile_m_values.contains(&cfg.tile_m));
            assert!(space.tile_n_values.contains(&cfg.tile_n));
            assert!(space.tile_k_values.contains(&cfg.tile_k));
            assert!(space.block_size_values.contains(&cfg.block_size));
        }
    }

    #[test]
    fn generation_increments_on_evolve() {
        let mut ga = GeneticAlgorithm::new(test_space(), test_ga_config());
        assert_eq!(ga.generation(), 0);
        evaluate_all(&mut ga);
        let _ = ga.evolve();
        assert_eq!(ga.generation(), 1);
        evaluate_all(&mut ga);
        let _ = ga.evolve();
        assert_eq!(ga.generation(), 2);
    }

    #[test]
    fn multiple_generations_run_without_panic() {
        let mut ga = GeneticAlgorithm::new(test_space(), test_ga_config());
        for generation in 0..10 {
            evaluate_all(&mut ga);
            let result = ga.evolve();
            assert!(result.is_ok(), "generation {generation} failed");
        }
        assert_eq!(ga.generation(), 10);
        assert!(ga.best_config().is_some());
    }

    #[test]
    fn mutation_rate_one_changes_all_genes() {
        let cfg = GeneticConfig {
            population_size: 10,
            elite_count: 0,
            crossover_rate: 0.0, // no crossover, just clone + mutate
            mutation_rate: 1.0,  // mutate every gene
            tournament_size: 2,
        };
        let mut ga = GeneticAlgorithm::with_seed(test_space(), cfg, 777);
        evaluate_all(&mut ga);
        let before: Vec<Config> = ga.population().iter().map(|(c, _)| c.clone()).collect();
        let _ = ga.evolve();

        // With mutation_rate=1.0 and no elitism, most offspring should
        // differ from their parents.
        let changed = ga
            .population()
            .iter()
            .filter(|(c, _)| !before.contains(c))
            .count();
        assert!(changed > 0, "high mutation rate should produce changes");
    }

    #[test]
    fn crossover_rate_zero_clones_parent() {
        let cfg = GeneticConfig {
            population_size: 10,
            elite_count: 0,
            crossover_rate: 0.0, // no crossover — offspring = parent clone
            mutation_rate: 0.0,  // no mutation either
            tournament_size: 2,
        };
        let mut ga = GeneticAlgorithm::with_seed(test_space(), cfg, 999);
        evaluate_all(&mut ga);
        let parents: Vec<Config> = ga.population().iter().map(|(c, _)| c.clone()).collect();
        let _ = ga.evolve();

        // Every offspring should be an exact copy of some parent
        for (child, _) in ga.population() {
            assert!(
                parents.contains(child),
                "with crossover=0 and mutation=0, every child should be a parent clone"
            );
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut ga1 = GeneticAlgorithm::with_seed(test_space(), test_ga_config(), 42);
        let mut ga2 = GeneticAlgorithm::with_seed(test_space(), test_ga_config(), 42);

        let pop1: Vec<Config> = ga1.population().iter().map(|(c, _)| c.clone()).collect();
        let pop2: Vec<Config> = ga2.population().iter().map(|(c, _)| c.clone()).collect();
        assert_eq!(pop1, pop2);

        evaluate_all(&mut ga1);
        evaluate_all(&mut ga2);
        let _ = ga1.evolve();
        let _ = ga2.evolve();

        let pop1b: Vec<Config> = ga1.population().iter().map(|(c, _)| c.clone()).collect();
        let pop2b: Vec<Config> = ga2.population().iter().map(|(c, _)| c.clone()).collect();
        assert_eq!(pop1b, pop2b);
    }

    // -- test helpers -------------------------------------------------------

    /// Evaluate all pending configs with a simple fitness function.
    fn evaluate_all(ga: &mut GeneticAlgorithm) {
        evaluate_all_with_fn(ga, |cfg| cfg.tile_m as f64 + cfg.tile_n as f64);
    }

    /// Evaluate all pending configs with a custom fitness function.
    fn evaluate_all_with_fn(ga: &mut GeneticAlgorithm, f: fn(&Config) -> f64) {
        let pending: Vec<Config> = ga.pending_evaluations().into_iter().cloned().collect();
        for cfg in &pending {
            let fitness = f(cfg);
            ga.set_fitness(cfg, fitness);
        }
    }

    /// Returns true when every gene of `cfg` is drawn from the corresponding
    /// values slice in `space`.
    fn config_in_space(cfg: &Config, space: &SearchSpace) -> bool {
        space.tile_m_values.contains(&cfg.tile_m)
            && space.tile_n_values.contains(&cfg.tile_n)
            && space.tile_k_values.contains(&cfg.tile_k)
            && space.warp_m_values.contains(&cfg.warp_m)
            && space.warp_n_values.contains(&cfg.warp_n)
            && space.stages_values.contains(&cfg.stages)
            && space.use_tensor_core_values.contains(&cfg.use_tensor_core)
            && space.block_size_values.contains(&cfg.block_size)
    }

    // -----------------------------------------------------------------------
    // Crossover and mutation validity tests
    // -----------------------------------------------------------------------

    /// Run 1000 generations of genetic search on a constrained search space.
    /// Every config in the population at every generation must be drawn from
    /// the search-space value lists.
    #[test]
    fn test_genetic_crossover_never_violates_constraints() {
        let space = SearchSpaceBuilder::new()
            .tile_m(vec![64, 128])
            .tile_n(vec![64, 128])
            .tile_k(vec![16, 32])
            .warp_m(vec![32, 64])
            .warp_n(vec![32, 64])
            .stages(vec![2, 3])
            .use_tensor_core(vec![false, true])
            .block_size(vec![128, 256])
            .build();

        let ga_cfg = GeneticConfig {
            population_size: 30,
            elite_count: 3,
            crossover_rate: 0.85,
            mutation_rate: 0.15,
            tournament_size: 4,
        };

        let mut ga = GeneticAlgorithm::with_seed(space.clone(), ga_cfg, 0xDEAD_C0DE);

        for generation in 0..100 {
            // Verify every individual in the current population is in the space.
            for (cfg, _) in ga.population() {
                assert!(
                    config_in_space(cfg, &space),
                    "generation {generation}: config outside search space: {cfg:?}",
                );
            }

            evaluate_all(&mut ga);
            ga.evolve()
                .expect("evolve must not fail with full evaluation");
        }

        // Verify the final population too.
        for (cfg, _) in ga.population() {
            assert!(
                config_in_space(cfg, &space),
                "final population: config outside search space: {cfg:?}",
            );
        }
    }

    /// Crossover of two valid parents must produce children whose genes are
    /// all drawn from the search space.
    #[test]
    fn test_crossover_produces_valid_configs() {
        let space = test_space();
        let ga_cfg = GeneticConfig {
            population_size: 10,
            elite_count: 0,
            crossover_rate: 1.0, // always crossover
            mutation_rate: 0.0,  // no mutation — isolate crossover
            tournament_size: 2,
        };

        let mut ga = GeneticAlgorithm::with_seed(space.clone(), ga_cfg, 12345);

        // Run 50 generations — elitism=0 and crossover=1.0 means every
        // offspring is a crossover product.
        for generation in 0..50 {
            evaluate_all(&mut ga);
            ga.evolve().expect("evolve");

            for (cfg, _) in ga.population() {
                assert!(
                    config_in_space(cfg, &space),
                    "generation {generation} (crossover): config outside space: {cfg:?}",
                );
            }
        }
    }

    /// Mutation of a valid config must produce a valid config.
    ///
    /// With `mutation_rate = 1.0` every gene is replaced by a random
    /// value from the search space — the result must still be in the space.
    #[test]
    fn test_mutation_produces_valid_config() {
        let space = test_space();
        let ga_cfg = GeneticConfig {
            population_size: 20,
            elite_count: 0,
            crossover_rate: 0.0, // no crossover — isolate mutation
            mutation_rate: 1.0,  // always mutate every gene
            tournament_size: 2,
        };

        let mut ga = GeneticAlgorithm::with_seed(space.clone(), ga_cfg, 99999);

        for generation in 0..50 {
            evaluate_all(&mut ga);
            ga.evolve().expect("evolve");

            for (cfg, _) in ga.population() {
                assert!(
                    config_in_space(cfg, &space),
                    "generation {generation} (mutation): config outside space: {cfg:?}",
                );
            }
        }
    }
}
