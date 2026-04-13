//! # OxiCUDA Rand -- GPU-Accelerated Random Number Generation
//!
//! This crate provides GPU-accelerated random number generation, serving as
//! a pure Rust equivalent to NVIDIA's cuRAND library. It supports multiple
//! RNG engines (Philox, XORWOW, MRG32k3a), various distributions (uniform,
//! normal, log-normal, Poisson), and quasi-random sequences (Sobol).
//!
//! ## Engines
//!
//! - **Philox-4x32-10** -- Counter-based PRNG, the cuRAND default.
//! - **XORWOW** -- XORshift with Weyl sequence, fast and suitable for
//!   most Monte Carlo workloads.
//! - **MRG32k3a** -- Combined multiple recursive generator with the
//!   highest statistical quality.
//!
//! ## Distributions
//!
//! - Uniform \[0, 1) in f32/f64
//! - Normal (Gaussian) via Box-Muller transform
//! - Log-normal via exponentiation of normals
//! - Poisson (Knuth for small lambda, normal approximation for large lambda)
//!
//! ## Quasi-random
//!
//! - Sobol sequences for low-discrepancy sampling in Monte Carlo integration

#![warn(clippy::all)]
#![warn(missing_docs)]

pub mod distributions;
pub mod engines;
pub mod error;
pub mod generator;
pub mod graph_gen;
pub mod matrix_gen;
pub mod monte_carlo;
pub mod quasi;
pub mod statistical_tests;

pub use error::{RandError, RandResult};
pub use generator::{RngEngine, RngGenerator};
pub use graph_gen::{
    AdjacencyList, BarabasiAlbertGenerator, ErdosRenyiGenerator, GraphStats, GraphType,
    RandomRegularGenerator, StochasticBlockModelGenerator, WattsStrogatzGenerator,
};
pub use matrix_gen::{
    CorrelationMatrixGenerator, GaussianMatrixGenerator, MatrixLayout, OrthogonalMatrixGenerator,
    RandomMatrix, SymmetricPositiveDefiniteGenerator, WishartGenerator,
};
pub use monte_carlo::{
    BlackScholesParams, HamiltonianMC, McmcResult, MetropolisHastings, MonteCarloConfig,
    MonteCarloResult, SamplerState,
};
pub use quasi::{HaltonGenerator, LatinHypercubeSampler, ScrambledSobolGenerator, SobolGenerator};

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::error::{RandError, RandResult};
    pub use crate::generator::{RngEngine, RngGenerator};
    pub use crate::graph_gen::{
        AdjacencyList, BarabasiAlbertGenerator, ErdosRenyiGenerator, GraphStats, GraphType,
        RandomRegularGenerator, StochasticBlockModelGenerator, WattsStrogatzGenerator,
    };
    pub use crate::matrix_gen::{
        CorrelationMatrixGenerator, GaussianMatrixGenerator, MatrixLayout,
        OrthogonalMatrixGenerator, RandomMatrix, SymmetricPositiveDefiniteGenerator,
        WishartGenerator,
    };
    pub use crate::monte_carlo::{
        BlackScholesParams, HamiltonianMC, McmcResult, MetropolisHastings, MonteCarloConfig,
        MonteCarloResult, SamplerState,
    };
    pub use crate::quasi::{
        HaltonGenerator, LatinHypercubeSampler, ScrambledSobolGenerator, SobolGenerator,
    };
}
