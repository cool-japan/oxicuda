//! Search space definition for autotuning.
//!
//! A [`SearchSpace`] defines the candidate values for each tunable
//! parameter.  The full search space is the Cartesian product of all
//! dimensions.  Before benchmarking, the search space can be pruned
//! with architecture constraints (shared memory limit, register limit)
//! to eliminate configurations that would fail at launch time.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::SearchSpace;
//!
//! let space = SearchSpace::gemm_default();
//! println!("Total configs: {}", space.total_configs());
//!
//! // Prune for a GPU with 48 KiB shared memory and 255 max registers.
//! let viable = space.prune(48 * 1024, 255, 4);
//! println!("Viable configs: {}", viable.len());
//! ```

use crate::config::Config;

/// Defines the search space for autotuning.
///
/// Each field is a list of candidate values for one tunable dimension.
/// The total search space is the Cartesian product of all dimensions.
/// Use [`SearchSpace::prune`] to eliminate architecturally infeasible
/// configurations before running expensive benchmarks.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Candidate values for block tile M.
    pub tile_m_values: Vec<u32>,
    /// Candidate values for block tile N.
    pub tile_n_values: Vec<u32>,
    /// Candidate values for block tile K.
    pub tile_k_values: Vec<u32>,
    /// Candidate values for warp tile M.
    pub warp_m_values: Vec<u32>,
    /// Candidate values for warp tile N.
    pub warp_n_values: Vec<u32>,
    /// Candidate values for pipeline stages.
    pub stages_values: Vec<u32>,
    /// Candidate values for Tensor Core usage.
    pub use_tensor_core_values: Vec<bool>,
    /// Candidate values for block size (threads per block).
    pub block_size_values: Vec<u32>,
}

impl SearchSpace {
    /// Creates a default GEMM search space covering common tile sizes.
    ///
    /// This space is designed to cover a wide range of architectures
    /// from Volta through Hopper.  The total number of raw combinations
    /// is large, so [`prune`](Self::prune) should be applied before
    /// benchmarking.
    #[must_use]
    pub fn gemm_default() -> Self {
        Self {
            tile_m_values: vec![32, 64, 128, 256],
            tile_n_values: vec![32, 64, 128, 256],
            tile_k_values: vec![8, 16, 32, 64],
            warp_m_values: vec![16, 32, 64],
            warp_n_values: vec![16, 32, 64],
            stages_values: vec![2, 3, 4],
            use_tensor_core_values: vec![false, true],
            block_size_values: vec![64, 128, 256],
        }
    }

    /// Creates a minimal search space for quick testing or CI.
    ///
    /// Contains only a handful of configurations to keep benchmark
    /// time short.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            tile_m_values: vec![64, 128],
            tile_n_values: vec![64, 128],
            tile_k_values: vec![16, 32],
            warp_m_values: vec![32, 64],
            warp_n_values: vec![32, 64],
            stages_values: vec![2],
            use_tensor_core_values: vec![false],
            block_size_values: vec![128, 256],
        }
    }

    /// Returns the total number of configurations (Cartesian product).
    ///
    /// This may be very large; always prune before benchmarking.
    #[must_use]
    pub fn total_configs(&self) -> usize {
        self.tile_m_values.len()
            * self.tile_n_values.len()
            * self.tile_k_values.len()
            * self.warp_m_values.len()
            * self.warp_n_values.len()
            * self.stages_values.len()
            * self.use_tensor_core_values.len()
            * self.block_size_values.len()
    }

    /// Generates all configurations in the search space (no filtering).
    ///
    /// **Warning:** this can produce a very large vector.  Prefer
    /// [`prune`](Self::prune) for real workloads.
    #[must_use]
    pub fn enumerate(&self) -> Vec<Config> {
        let mut configs = Vec::with_capacity(self.total_configs());
        for &tile_m in &self.tile_m_values {
            for &tile_n in &self.tile_n_values {
                for &tile_k in &self.tile_k_values {
                    for &warp_m in &self.warp_m_values {
                        for &warp_n in &self.warp_n_values {
                            for &stages in &self.stages_values {
                                for &use_tc in &self.use_tensor_core_values {
                                    for &block_size in &self.block_size_values {
                                        configs.push(Config {
                                            tile_m,
                                            tile_n,
                                            tile_k,
                                            warp_m,
                                            warp_n,
                                            stages,
                                            use_tensor_core: use_tc,
                                            block_size,
                                            extra: std::collections::HashMap::new(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        configs
    }

    /// Applies architecture constraints and heuristics to eliminate
    /// invalid configurations.
    ///
    /// A configuration is pruned if any of the following hold:
    ///
    /// 1. **Shared memory exceeded** — the estimated shared memory usage
    ///    (for double-buffered A and B tiles) exceeds `max_shared_mem`.
    /// 2. **Register pressure too high** — the estimated per-thread
    ///    register count exceeds `max_regs_per_thread`.
    /// 3. **Warp tile exceeds block tile** — `warp_m > tile_m` or
    ///    `warp_n > tile_n`, which makes no geometric sense.
    /// 4. **Insufficient threads** — the number of warps needed to
    ///    cover the block tile (based on warp tiles) exceeds the warps
    ///    available at the given `block_size`.
    ///
    /// `element_bytes` is the size of one matrix element in bytes
    /// (e.g. 4 for `f32`, 2 for `f16`).
    #[must_use]
    pub fn prune(
        &self,
        max_shared_mem: usize,
        max_regs_per_thread: u32,
        element_bytes: u32,
    ) -> Vec<Config> {
        self.enumerate()
            .into_iter()
            .filter(|cfg| {
                // Constraint 1: shared memory
                let shared = cfg.estimated_shared_mem(element_bytes);
                if shared as usize > max_shared_mem {
                    return false;
                }

                // Constraint 2: register pressure
                let regs = cfg.estimated_registers_per_thread();
                if regs > max_regs_per_thread {
                    return false;
                }

                // Constraint 3: warp tile must fit within block tile
                if cfg.warp_m > cfg.tile_m || cfg.warp_n > cfg.tile_n {
                    return false;
                }

                // Constraint 4: enough warps to cover the block tile
                // Number of warp-tiles that tile the block:
                //   (tile_m / warp_m) * (tile_n / warp_n) warps needed
                // Available warps: block_size / 32
                if cfg.warp_m == 0 || cfg.warp_n == 0 {
                    return false;
                }
                let warps_m = cfg.tile_m / cfg.warp_m;
                let warps_n = cfg.tile_n / cfg.warp_n;
                let warps_needed = warps_m * warps_n;
                let warps_available = cfg.block_size / 32;
                if warps_needed > warps_available || warps_available == 0 {
                    return false;
                }

                true
            })
            .collect()
    }
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::gemm_default()
    }
}

/// Builder for constructing custom search spaces one dimension at a time.
///
/// # Example
///
/// ```rust
/// use oxicuda_autotune::SearchSpaceBuilder;
///
/// let space = SearchSpaceBuilder::new()
///     .tile_m(vec![64, 128])
///     .tile_n(vec![64, 128])
///     .tile_k(vec![32])
///     .stages(vec![2, 3, 4])
///     .build();
///
/// assert!(space.total_configs() > 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SearchSpaceBuilder {
    tile_m: Option<Vec<u32>>,
    tile_n: Option<Vec<u32>>,
    tile_k: Option<Vec<u32>>,
    warp_m: Option<Vec<u32>>,
    warp_n: Option<Vec<u32>>,
    stages: Option<Vec<u32>>,
    use_tensor_core: Option<Vec<bool>>,
    block_size: Option<Vec<u32>>,
}

impl SearchSpaceBuilder {
    /// Creates a new builder with no dimensions set.
    ///
    /// Any dimension left unset will use the [`SearchSpace::gemm_default`]
    /// values when [`build`](Self::build) is called.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the candidate values for block tile M.
    #[must_use]
    pub fn tile_m(mut self, values: Vec<u32>) -> Self {
        self.tile_m = Some(values);
        self
    }

    /// Sets the candidate values for block tile N.
    #[must_use]
    pub fn tile_n(mut self, values: Vec<u32>) -> Self {
        self.tile_n = Some(values);
        self
    }

    /// Sets the candidate values for block tile K.
    #[must_use]
    pub fn tile_k(mut self, values: Vec<u32>) -> Self {
        self.tile_k = Some(values);
        self
    }

    /// Sets the candidate values for warp tile M.
    #[must_use]
    pub fn warp_m(mut self, values: Vec<u32>) -> Self {
        self.warp_m = Some(values);
        self
    }

    /// Sets the candidate values for warp tile N.
    #[must_use]
    pub fn warp_n(mut self, values: Vec<u32>) -> Self {
        self.warp_n = Some(values);
        self
    }

    /// Sets the candidate values for pipeline stages.
    #[must_use]
    pub fn stages(mut self, values: Vec<u32>) -> Self {
        self.stages = Some(values);
        self
    }

    /// Sets the candidate values for Tensor Core usage.
    #[must_use]
    pub fn use_tensor_core(mut self, values: Vec<bool>) -> Self {
        self.use_tensor_core = Some(values);
        self
    }

    /// Sets the candidate values for block size.
    #[must_use]
    pub fn block_size(mut self, values: Vec<u32>) -> Self {
        self.block_size = Some(values);
        self
    }

    /// Builds the search space, using [`SearchSpace::gemm_default`]
    /// values for any dimension not explicitly set.
    #[must_use]
    pub fn build(self) -> SearchSpace {
        let defaults = SearchSpace::gemm_default();
        SearchSpace {
            tile_m_values: self.tile_m.unwrap_or(defaults.tile_m_values),
            tile_n_values: self.tile_n.unwrap_or(defaults.tile_n_values),
            tile_k_values: self.tile_k.unwrap_or(defaults.tile_k_values),
            warp_m_values: self.warp_m.unwrap_or(defaults.warp_m_values),
            warp_n_values: self.warp_n.unwrap_or(defaults.warp_n_values),
            stages_values: self.stages.unwrap_or(defaults.stages_values),
            use_tensor_core_values: self
                .use_tensor_core
                .unwrap_or(defaults.use_tensor_core_values),
            block_size_values: self.block_size.unwrap_or(defaults.block_size_values),
        }
    }
}

/// Returns true when a config satisfies the shared-memory constraint for
/// the given limit and element size.
///
/// This mirrors the logic inside [`SearchSpace::prune`] and is exposed
/// so that the correctness tests below can use it independently.
#[cfg(test)]
fn satisfies_prune_constraints(
    cfg: &Config,
    max_shared: usize,
    max_regs: u32,
    elem_bytes: u32,
) -> bool {
    // Constraint 1: shared memory
    if cfg.estimated_shared_mem(elem_bytes) as usize > max_shared {
        return false;
    }
    // Constraint 2: register pressure
    if cfg.estimated_registers_per_thread() > max_regs {
        return false;
    }
    // Constraint 3: warp tile must not exceed block tile
    if cfg.warp_m > cfg.tile_m || cfg.warp_n > cfg.tile_n {
        return false;
    }
    // Constraint 4: enough warps to cover the block tile
    if cfg.warp_m == 0 || cfg.warp_n == 0 {
        return false;
    }
    let warps_m = cfg.tile_m / cfg.warp_m;
    let warps_n = cfg.tile_n / cfg.warp_n;
    let warps_needed = warps_m * warps_n;
    let warps_available = cfg.block_size / 32;
    if warps_needed > warps_available || warps_available == 0 {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_default_total_configs() {
        let space = SearchSpace::gemm_default();
        // 4 * 4 * 4 * 3 * 3 * 3 * 2 * 3 = 10368
        assert_eq!(space.total_configs(), 10368);
    }

    #[test]
    fn minimal_space_is_small() {
        let space = SearchSpace::minimal();
        // 2 * 2 * 2 * 2 * 2 * 1 * 1 * 2 = 64
        assert_eq!(space.total_configs(), 64);
    }

    #[test]
    fn enumerate_produces_correct_count() {
        let space = SearchSpace::minimal();
        let configs = space.enumerate();
        assert_eq!(configs.len(), space.total_configs());
    }

    #[test]
    fn prune_removes_invalid_configs() {
        let space = SearchSpace::minimal();
        let all = space.enumerate();
        // 16 KiB shared memory — should reject large tile configs
        let pruned = space.prune(16 * 1024, 255, 4);
        assert!(pruned.len() <= all.len());
        // Every surviving config must satisfy shared mem constraint
        for cfg in &pruned {
            assert!(cfg.estimated_shared_mem(4) <= 16 * 1024);
        }
    }

    #[test]
    fn prune_rejects_warp_larger_than_block_tile() {
        let space = SearchSpaceBuilder::new()
            .tile_m(vec![32])
            .tile_n(vec![32])
            .tile_k(vec![16])
            .warp_m(vec![64]) // warp_m > tile_m — invalid
            .warp_n(vec![32])
            .stages(vec![2])
            .use_tensor_core(vec![false])
            .block_size(vec![256])
            .build();

        let pruned = space.prune(256 * 1024, 255, 4);
        assert!(pruned.is_empty());
    }

    #[test]
    fn builder_defaults_fill_in() {
        let space = SearchSpaceBuilder::new().tile_m(vec![64]).build();
        assert_eq!(space.tile_m_values, vec![64]);
        // Other dimensions should use gemm_default values
        assert_eq!(
            space.tile_n_values,
            SearchSpace::gemm_default().tile_n_values
        );
    }
}

// ---------------------------------------------------------------------------
// Pruning correctness tests
// ---------------------------------------------------------------------------
//
// These tests use a fully hand-crafted search space with a small number of
// dimensions so that every valid/invalid combination can be enumerated and
// verified manually.
//
// Search space (48 total combos):
//   tile_m in [64, 128, 256]
//   tile_n in [64, 128]
//   tile_k = [16]          (fixed — keeps the shared-mem formula simple)
//   warp_m = [64]          (fixed)
//   warp_n = [64]          (fixed)
//   stages in [2, 3]
//   use_tensor_core in [false, true]
//   block_size in [128, 256]
//
// Parameters passed to `prune`:
//   max_shared_mem  = 164 * 1024  (168,960 bytes — Ampere shared memory)
//   max_regs        = 64
//   element_bytes   = 2  (f16)
//
// With these parameters the `estimated_shared_mem(2)` formula expands to:
//   (tile_m * 16 + 16 * tile_n) * stages * 2 = 32 * (tile_m + tile_n) * stages
//
// All (tile_m, tile_n, stages) combinations produce values well under 164 KiB,
// so constraint 1 never rejects any config in this space.
//
// `estimated_registers_per_thread()` = 32 + (warp_m * warp_n) / block_size + tc_overhead
//   where warp_m=64, warp_n=64 → warp product = 4096
//   tc_overhead = 16 when use_tensor_core=true, else 0
//
// Block 128:  regs_no_tc = 32 + 4096/128       = 64  (≤64 → OK)
//             regs_tc    = 32 + 4096/128 + 16  = 80  (>64 → PRUNED)
// Block 256:  regs_no_tc = 32 + 4096/256       = 48  (≤64 → OK)
//             regs_tc    = 32 + 4096/256 + 16  = 64  (≤64 → OK)
//
// Register pruning removes all combos with use_tc=true AND block_size=128:
//   3 tile_m × 2 tile_n × 2 stages × 1 use_tc × 1 block_size = 12 pruned
//
// Constraint 3 (warp ≤ tile): warp_m=64, warp_n=64 — all tile_m ≥ 64 and
//   all tile_n ≥ 64, so no additional pruning from constraint 3.
//
// Constraint 4 (sufficient warps):
//   warps_needed = (tile_m / warp_m) * (tile_n / warp_n)
//   warps_available = block_size / 32
//
//   block_size=128 → warps_available = 4
//   block_size=256 → warps_available = 8
//
//   For (tile_m=256, tile_n=128):
//     warps_needed = (256/64) * (128/64) = 4 * 2 = 8
//     block_size=128 → 8 > 4 → PRUNED
//     block_size=256 → 8 ≤ 8 → OK
//
//   All other (tile_m, tile_n) combinations give warps_needed ≤ 4, so they
//   pass for both block sizes.
//
// Constraint 4 removes (tile_m=256, tile_n=128, block_size=128, use_tc=false):
//   2 stages × 1 = 2 combos   (use_tc=true+block_size=128 already counted above)
//
// Total pruned = 12 (register) + 2 (warp coverage) = 14
// Remaining = 48 − 14 = 34

#[cfg(test)]
mod pruning_tests {
    use super::*;

    /// Returns the hand-crafted search space used by all tests in this module.
    fn test_space() -> SearchSpace {
        SearchSpaceBuilder::new()
            .tile_m(vec![64, 128, 256])
            .tile_n(vec![64, 128])
            .tile_k(vec![16])
            .warp_m(vec![64])
            .warp_n(vec![64])
            .stages(vec![2, 3])
            .use_tensor_core(vec![false, true])
            .block_size(vec![128, 256])
            .build()
    }

    const MAX_SHARED: usize = 164 * 1024; // 168,960 bytes
    const MAX_REGS: u32 = 64;
    const ELEM_BYTES: u32 = 2; // f16

    // -----------------------------------------------------------------------
    // Test that total_combinations() before pruning equals the Cartesian
    // product count (3 * 2 * 2 * 2 * 2 = 48).
    // -----------------------------------------------------------------------
    #[test]
    fn test_total_combinations_is_cartesian_product() {
        let space = test_space();
        // 3 × 2 × 2 × 2 × 2
        let expected = 3_usize * 2 * 2 * 2 * 2;
        assert_eq!(space.total_configs(), expected);
        assert_eq!(space.enumerate().len(), expected);
    }

    // -----------------------------------------------------------------------
    // Test that pruning never produces invalid configs — every surviving
    // config must pass all four constraints.
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_pruned_configs_satisfy_constraints() {
        let space = test_space();
        let pruned = space.prune(MAX_SHARED, MAX_REGS, ELEM_BYTES);

        for cfg in &pruned {
            assert!(
                satisfies_prune_constraints(cfg, MAX_SHARED, MAX_REGS, ELEM_BYTES),
                "pruned config violates at least one constraint: {cfg:?}",
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test that pruning matches the manually calculated valid set.
    //
    // Expected count: 48 total − 12 (register) − 2 (warp coverage) = 34.
    // -----------------------------------------------------------------------
    #[test]
    fn test_pruning_matches_manual_calculation() {
        let space = test_space();
        let pruned = space.prune(MAX_SHARED, MAX_REGS, ELEM_BYTES);

        // Manually enumerate every combination and apply the constraints.
        let tile_ms = [64_u32, 128, 256];
        let tile_ns = [64_u32, 128];
        let tile_k = 16_u32;
        let warp_m = 64_u32;
        let warp_n = 64_u32;
        let stages_vals = [2_u32, 3];
        let use_tcs = [false, true];
        let block_sizes = [128_u32, 256];

        let mut manual_valid: Vec<Config> = Vec::new();
        for &tm in &tile_ms {
            for &tn in &tile_ns {
                for &st in &stages_vals {
                    for &utc in &use_tcs {
                        for &bs in &block_sizes {
                            let cfg = Config {
                                tile_m: tm,
                                tile_n: tn,
                                tile_k,
                                warp_m,
                                warp_n,
                                stages: st,
                                use_tensor_core: utc,
                                block_size: bs,
                                extra: std::collections::HashMap::new(),
                            };
                            if satisfies_prune_constraints(&cfg, MAX_SHARED, MAX_REGS, ELEM_BYTES) {
                                manual_valid.push(cfg);
                            }
                        }
                    }
                }
            }
        }

        // The two sets must have the same size.
        assert_eq!(
            pruned.len(),
            manual_valid.len(),
            "prune() returned {} configs but manual calculation yields {}",
            pruned.len(),
            manual_valid.len(),
        );

        // Verify the expected count matches our analysis: 48 − 14 = 34.
        assert_eq!(
            pruned.len(),
            34,
            "expected 34 valid configs after pruning, got {}",
            pruned.len(),
        );

        // Every config returned by prune() must appear in the manual set.
        for cfg in &pruned {
            assert!(
                manual_valid.contains(cfg),
                "prune() returned a config not in the manual valid set: {cfg:?}",
            );
        }
        // Every manual valid config must appear in the pruned set.
        for cfg in &manual_valid {
            assert!(
                pruned.contains(cfg),
                "manually valid config was missing from prune(): {cfg:?}",
            );
        }
    }

    // -----------------------------------------------------------------------
    // Regression: make sure the previously-tested "warp > tile" rejection
    // still works with the pruning_tests space.
    // -----------------------------------------------------------------------
    #[test]
    fn test_pruning_rejects_oversized_warp_tile() {
        let space = SearchSpaceBuilder::new()
            .tile_m(vec![32])
            .tile_n(vec![32])
            .tile_k(vec![16])
            .warp_m(vec![64]) // warp_m > tile_m — always invalid
            .warp_n(vec![32])
            .stages(vec![2])
            .use_tensor_core(vec![false])
            .block_size(vec![256])
            .build();

        let pruned = space.prune(256 * 1024, 255, 4);
        assert!(pruned.is_empty(), "expected empty after warp-tile pruning");
    }
}
