//! High-level PTX kernel builder DSL.
//!
//! This module provides the ergonomic API for constructing PTX kernels
//! from Rust code. The main entry point is [`KernelBuilder`], which handles
//! the overall kernel structure (parameters, target, directives), while
//! [`BodyBuilder`] provides the instruction-level API used inside the
//! kernel body closure.
//!
//! See [`KernelBuilder`] for usage examples.

mod body_builder;
mod kernel_builder;

// P1 / P8 quality-gate tests (test-only module, compiled only in test mode).
#[cfg(test)]
mod quality_gates;

pub use body_builder::BodyBuilder;
// Warp shuffle/vote emission helpers and the SIMD-flavored warp-vector layer.
pub use body_builder::{warp_ops, warp_vec};
pub use kernel_builder::KernelBuilder;
