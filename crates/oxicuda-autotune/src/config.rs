//! Tunable kernel configuration.
//!
//! A [`Config`] represents one point in the autotuning search space —
//! a specific combination of tile sizes, pipeline depths, thread-block
//! dimensions, and optional custom parameters.  The autotune engine
//! benchmarks many `Config` instances to find the fastest one for a
//! given (GPU, kernel, problem-size) triple.
//!
//! # Builder pattern
//!
//! ```rust
//! use oxicuda_autotune::Config;
//!
//! let cfg = Config::new()
//!     .with_tile_m(64)
//!     .with_tile_n(64)
//!     .with_tile_k(16)
//!     .with_stages(3)
//!     .with_use_tensor_core(true)
//!     .with_block_size(128);
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

/// A tunable kernel configuration.
///
/// Represents one point in the search space — a specific combination
/// of tile sizes, pipeline depths, and other parameters that control
/// how a GPU kernel is launched and how it partitions work across
/// thread blocks and warps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// Block tile size in the M (row) dimension.
    pub tile_m: u32,
    /// Block tile size in the N (column) dimension.
    pub tile_n: u32,
    /// Block tile size in the K (reduction / inner) dimension.
    pub tile_k: u32,
    /// Warp tile size in the M dimension.
    pub warp_m: u32,
    /// Warp tile size in the N dimension.
    pub warp_n: u32,
    /// Number of pipeline stages (software pipelining / multi-buffering).
    pub stages: u32,
    /// Whether to use Tensor Core (WMMA / MMA) instructions.
    pub use_tensor_core: bool,
    /// Block size — total number of threads per thread block.
    pub block_size: u32,
    /// Additional key-value parameters for custom or non-GEMM kernels.
    ///
    /// This map allows extending the configuration without changing
    /// the struct definition.  Example keys: `"unroll_factor"`,
    /// `"vector_width"`, `"shared_pad"`.
    pub extra: HashMap<String, u32>,
}

impl Hash for Config {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tile_m.hash(state);
        self.tile_n.hash(state);
        self.tile_k.hash(state);
        self.warp_m.hash(state);
        self.warp_n.hash(state);
        self.stages.hash(state);
        self.use_tensor_core.hash(state);
        self.block_size.hash(state);
        // Hash extra map in a deterministic order (sorted by key).
        let mut pairs: Vec<_> = self.extra.iter().collect();
        pairs.sort_by_key(|(k, _)| *k);
        for (k, v) in pairs {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl Config {
    /// Creates a new configuration with sensible defaults.
    ///
    /// The defaults are suitable as a starting point for medium-sized
    /// GEMM problems on Ampere-class GPUs:
    ///
    /// | Parameter       | Default |
    /// |-----------------|---------|
    /// | `tile_m`        | 128     |
    /// | `tile_n`        | 128     |
    /// | `tile_k`        | 32      |
    /// | `warp_m`        | 64      |
    /// | `warp_n`        | 64      |
    /// | `stages`        | 2       |
    /// | `use_tensor_core`| false  |
    /// | `block_size`    | 256     |
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the block tile size in the M dimension.
    #[must_use]
    pub fn with_tile_m(mut self, v: u32) -> Self {
        self.tile_m = v;
        self
    }

    /// Sets the block tile size in the N dimension.
    #[must_use]
    pub fn with_tile_n(mut self, v: u32) -> Self {
        self.tile_n = v;
        self
    }

    /// Sets the block tile size in the K dimension.
    #[must_use]
    pub fn with_tile_k(mut self, v: u32) -> Self {
        self.tile_k = v;
        self
    }

    /// Sets the warp tile size in the M dimension.
    #[must_use]
    pub fn with_warp_m(mut self, v: u32) -> Self {
        self.warp_m = v;
        self
    }

    /// Sets the warp tile size in the N dimension.
    #[must_use]
    pub fn with_warp_n(mut self, v: u32) -> Self {
        self.warp_n = v;
        self
    }

    /// Sets the number of software pipeline stages.
    #[must_use]
    pub fn with_stages(mut self, v: u32) -> Self {
        self.stages = v;
        self
    }

    /// Enables or disables Tensor Core usage.
    #[must_use]
    pub fn with_use_tensor_core(mut self, v: bool) -> Self {
        self.use_tensor_core = v;
        self
    }

    /// Sets the thread-block size (threads per block).
    #[must_use]
    pub fn with_block_size(mut self, v: u32) -> Self {
        self.block_size = v;
        self
    }

    /// Inserts a custom key-value parameter into the `extra` map.
    #[must_use]
    pub fn with_extra(mut self, key: impl Into<String>, value: u32) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// Returns the number of warps this configuration requires per block.
    ///
    /// Assumes a warp size of 32 threads.
    #[must_use]
    pub fn warps_per_block(&self) -> u32 {
        self.block_size / 32
    }

    /// Estimates the shared memory usage (in bytes) for a GEMM kernel
    /// at the given element precision.
    ///
    /// The estimate covers double-buffered tiles for matrices A and B:
    ///
    /// ```text
    /// shared_bytes = (tile_m * tile_k + tile_k * tile_n) * stages * element_bytes
    /// ```
    #[must_use]
    pub fn estimated_shared_mem(&self, element_bytes: u32) -> u64 {
        let a_tile = u64::from(self.tile_m) * u64::from(self.tile_k);
        let b_tile = u64::from(self.tile_k) * u64::from(self.tile_n);
        (a_tile + b_tile) * u64::from(self.stages) * u64::from(element_bytes)
    }

    /// Heuristic estimate of register pressure per thread.
    ///
    /// This is a rough approximation based on the warp-level tile
    /// dimensions and whether Tensor Cores are used.  Real register
    /// usage depends on the compiler, but this is useful for pruning
    /// clearly infeasible configurations.
    #[must_use]
    pub fn estimated_registers_per_thread(&self) -> u32 {
        let base_regs: u32 = 32;
        let fragment_regs = (self.warp_m * self.warp_n) / self.block_size;
        let tc_overhead = if self.use_tensor_core { 16 } else { 0 };
        base_regs + fragment_regs + tc_overhead
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            warp_m: 64,
            warp_n: 64,
            stages: 2,
            use_tensor_core: false,
            block_size: 256,
            extra: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let cfg = Config::default();
        assert_eq!(cfg.tile_m, 128);
        assert_eq!(cfg.tile_n, 128);
        assert_eq!(cfg.tile_k, 32);
        assert_eq!(cfg.block_size, 256);
        assert_eq!(cfg.stages, 2);
        assert!(!cfg.use_tensor_core);
    }

    #[test]
    fn builder_chain_works() {
        let cfg = Config::new()
            .with_tile_m(64)
            .with_tile_n(64)
            .with_tile_k(16)
            .with_warp_m(32)
            .with_warp_n(32)
            .with_stages(3)
            .with_use_tensor_core(true)
            .with_block_size(128)
            .with_extra("unroll", 4);

        assert_eq!(cfg.tile_m, 64);
        assert_eq!(cfg.tile_n, 64);
        assert_eq!(cfg.tile_k, 16);
        assert_eq!(cfg.warp_m, 32);
        assert_eq!(cfg.warp_n, 32);
        assert_eq!(cfg.stages, 3);
        assert!(cfg.use_tensor_core);
        assert_eq!(cfg.block_size, 128);
        assert_eq!(cfg.extra.get("unroll"), Some(&4));
    }

    #[test]
    fn estimated_shared_mem_calculation() {
        let cfg = Config::new()
            .with_tile_m(128)
            .with_tile_n(128)
            .with_tile_k(32)
            .with_stages(2);
        // (128*32 + 32*128) * 2 * 4 = (4096 + 4096) * 8 = 65536
        assert_eq!(cfg.estimated_shared_mem(4), 65536);
    }

    #[test]
    fn warps_per_block() {
        let cfg = Config::new().with_block_size(256);
        assert_eq!(cfg.warps_per_block(), 8);
    }

    #[test]
    fn serde_roundtrip() {
        let cfg = Config::new().with_tile_m(64).with_extra("foo", 42);
        let json = serde_json::to_string(&cfg).expect("serialize");
        let restored: Config = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, restored);
    }
}
