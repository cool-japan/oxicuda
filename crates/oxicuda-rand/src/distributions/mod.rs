//! Distribution transform utilities for GPU random number generation.
//!
//! These modules provide PTX emission helpers that transform raw uniform
//! random values into samples from various probability distributions.
//!
//! - [`uniform`] -- Convert raw u32 to uniform float in \[0,1)
//! - [`normal`] -- Box-Muller transform for Gaussian samples
//! - [`log_normal`] -- Exponentiation of normal samples
//! - [`poisson`] -- Poisson distribution via Knuth and normal approximation

pub mod binomial;
pub mod geometric;
pub mod log_normal;
pub mod multinomial;
pub mod normal;
pub mod poisson;
pub mod truncated_normal;
pub mod uniform;
