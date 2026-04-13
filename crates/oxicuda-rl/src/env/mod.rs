//! Environment abstractions for OxiCUDA-RL.
//!
//! # Modules
//!
//! * `env` — [`crate::env::env::Env`] trait, [`crate::env::env::EnvInfo`],
//!   [`crate::env::env::StepResult`], and [`crate::env::env::LinearQuadraticEnv`]
//!   (the reference test environment).
//! * `vectorized` — [`crate::env::vectorized::VecEnv`] synchronous vectorized
//!   wrapper and [`crate::env::vectorized::VecStepResult`].
//!
//! # Quick start
//!
//! ```rust
//! use oxicuda_rl::env::env::{Env, LinearQuadraticEnv};
//! use oxicuda_rl::env::vectorized::VecEnv;
//!
//! // Single environment.
//! let mut single = LinearQuadraticEnv::new(4, 200);
//! let obs = single.reset().unwrap();
//! assert_eq!(obs.len(), 4);
//!
//! // Vectorized.
//! let envs: Vec<_> = (0..4).map(|_| LinearQuadraticEnv::new(4, 200)).collect();
//! let mut ve = VecEnv::new(envs);
//! let flat = ve.reset_all().unwrap();
//! assert_eq!(flat.len(), 4 * 4);
//! ```

pub mod env;
pub mod vectorized;

// Re-export the most-used types for ergonomic access via `crate::env::*`.
pub use env::{Env, EnvInfo, LinearQuadraticEnv, StepResult};
pub use vectorized::{VecEnv, VecStepResult};
