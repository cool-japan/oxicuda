//! The [`TunableKernel`] trait for autotunable GPU kernels.
//!
//! Any kernel that participates in automatic tuning must implement
//! this trait.  It provides the autotune engine with the information
//! needed to generate problem keys, estimate resource usage, and
//! validate configurations against hardware constraints.
//!
//! # Implementing `TunableKernel`
//!
//! ```rust
//! use oxicuda_autotune::{TunableKernel, Config};
//!
//! /// A GEMM problem description.
//! struct GemmProblem {
//!     m: u32,
//!     n: u32,
//!     k: u32,
//! }
//!
//! struct GemmKernel;
//!
//! impl TunableKernel for GemmKernel {
//!     type Problem = GemmProblem;
//!
//!     fn problem_key(&self, problem: &GemmProblem) -> String {
//!         format!("{}x{}x{}", problem.m, problem.n, problem.k)
//!     }
//!
//!     fn kernel_name(&self) -> &str {
//!         "sgemm"
//!     }
//!
//!     fn compute_flops(&self, problem: &GemmProblem) -> f64 {
//!         2.0 * f64::from(problem.m) * f64::from(problem.n) * f64::from(problem.k)
//!     }
//!
//!     fn shared_mem_bytes(&self, config: &Config) -> u32 {
//!         config.estimated_shared_mem(4) as u32
//!     }
//! }
//! ```

use crate::config::Config;

/// Trait for kernels that can be autotuned.
///
/// Implementations provide problem description, resource estimation,
/// and configuration validation for a specific kernel type (GEMM,
/// convolution, FFT, etc.).
///
/// The engine uses this trait to:
/// - Generate database keys via [`problem_key`](Self::problem_key).
/// - Compute theoretical FLOP counts for efficiency metrics.
/// - Estimate shared memory usage to prune infeasible configs.
/// - Validate configurations against architecture limits.
pub trait TunableKernel: Send + Sync {
    /// The problem description type for this kernel.
    ///
    /// For GEMM this might contain (M, N, K, data type); for
    /// convolution it might contain (batch, channels, height, width,
    /// filter size).
    type Problem;

    /// Generates a string key for the result database.
    ///
    /// The key should uniquely identify the problem dimensions that
    /// affect performance.  Example: `"1024x1024x1024"` for GEMM.
    fn problem_key(&self, problem: &Self::Problem) -> String;

    /// Returns the kernel name (e.g. `"sgemm"`, `"conv2d_fwd"`).
    ///
    /// This is used as the second-level key in the result database.
    fn kernel_name(&self) -> &str;

    /// Computes the theoretical floating-point operation count.
    ///
    /// For GEMM: `2 * M * N * K`.  Used to calculate GFLOPS from
    /// the measured execution time.
    fn compute_flops(&self, problem: &Self::Problem) -> f64;

    /// Estimates shared memory usage (in bytes) for a configuration.
    ///
    /// The autotune engine uses this to prune configurations that
    /// would exceed the GPU's shared memory limit.
    fn shared_mem_bytes(&self, config: &Config) -> u32;

    /// Checks whether a configuration is valid for the given
    /// shared memory limit.
    ///
    /// The default implementation compares [`shared_mem_bytes`](Self::shared_mem_bytes)
    /// against `max_shared_mem`.  Override this to add custom
    /// validation logic (e.g. alignment requirements, register
    /// limits).
    fn is_valid_config(&self, config: &Config, max_shared_mem: usize) -> bool {
        (self.shared_mem_bytes(config) as usize) <= max_shared_mem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyProblem {
        m: u32,
        n: u32,
        k: u32,
    }

    struct DummyKernel;

    impl TunableKernel for DummyKernel {
        type Problem = DummyProblem;

        fn problem_key(&self, problem: &DummyProblem) -> String {
            format!("{}x{}x{}", problem.m, problem.n, problem.k)
        }

        fn kernel_name(&self) -> &str {
            "dummy_gemm"
        }

        fn compute_flops(&self, problem: &DummyProblem) -> f64 {
            2.0 * f64::from(problem.m) * f64::from(problem.n) * f64::from(problem.k)
        }

        fn shared_mem_bytes(&self, config: &Config) -> u32 {
            config.estimated_shared_mem(4) as u32
        }
    }

    #[test]
    fn problem_key_format() {
        let kernel = DummyKernel;
        let problem = DummyProblem {
            m: 1024,
            n: 512,
            k: 256,
        };
        assert_eq!(kernel.problem_key(&problem), "1024x512x256");
    }

    #[test]
    fn compute_flops_gemm() {
        let kernel = DummyKernel;
        let problem = DummyProblem {
            m: 1024,
            n: 1024,
            k: 1024,
        };
        let expected = 2.0 * 1024.0 * 1024.0 * 1024.0;
        assert!((kernel.compute_flops(&problem) - expected).abs() < 1.0);
    }

    #[test]
    fn is_valid_config_checks_shared_mem() {
        let kernel = DummyKernel;
        let cfg = Config::new()
            .with_tile_m(128)
            .with_tile_n(128)
            .with_tile_k(32)
            .with_stages(2);
        let shared = kernel.shared_mem_bytes(&cfg) as usize;

        assert!(kernel.is_valid_config(&cfg, shared));
        assert!(kernel.is_valid_config(&cfg, shared + 1));
        assert!(!kernel.is_valid_config(&cfg, shared - 1));
    }
}
