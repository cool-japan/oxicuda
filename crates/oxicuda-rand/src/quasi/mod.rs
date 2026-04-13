//! Quasi-random number generation for Monte Carlo methods.
//!
//! Quasi-random (low-discrepancy) sequences fill the sample space more
//! uniformly than pseudorandom sequences, leading to faster convergence
//! in Monte Carlo integration.
//!
//! - [`sobol`] -- Sobol sequences using direction numbers from Joe & Kuo.

pub mod halton;
pub mod latin_hypercube;
pub mod scrambled_sobol;
pub mod sobol;

pub use halton::HaltonGenerator;
pub use latin_hypercube::LatinHypercubeSampler;
pub use scrambled_sobol::ScrambledSobolGenerator;
pub use sobol::SobolGenerator;
