//! [`MetalBackend`] — the main entry point for the oxicuda-metal crate.
//!
//! Implements the [`oxicuda_backend::ComputeBackend`] trait using Apple's
//! Metal API for GPU compute on macOS.
//!
//! Submodule layout:
//! - [`types`]: the [`MetalBackend`] struct, init helpers, and Metal/stub
//!   dispatch implementations split by `target_os`.
//! - [`nn`]: GPU dispatch for the neural-network ops (conv2d, attention,
//!   softmax, layer norm, scan) plus the host attention fallback.
//! - [`trait_impls`]: the public `Default` and `ComputeBackend` impls.
//! - [`functions`]: small utility helpers and the `mod tests` integration
//!   suite.

pub mod functions;
pub mod nn;
pub mod trait_impls;
pub mod types;

/// On-device numeric tests for the GPU paths (kept out of [`functions`] so that
/// file stays well under the 2000-line refactoring limit).
#[cfg(test)]
mod gpu_tests;

pub use types::{MetalBackend, MetalExternalBuffer};
