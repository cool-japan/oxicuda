//! Reduction operations for OxiCUDA BLAS.
//!
//! This module provides parallel reduction primitives over device buffers:
//! sum, max, min, mean, variance, and softmax. Each operation generates PTX
//! via templates from `oxicuda-ptx`, performs a two-phase block-level
//! reduction when needed, and writes the scalar (or vector) result to device
//! memory.

pub mod axis;
mod max;
mod mean;
mod min;
mod ops;
mod softmax;
mod sum;
mod variance;

pub use axis::reduce_axis;
pub use max::max;
pub use mean::mean;
pub use min::min;
pub use ops::ReductionOp;
pub use softmax::softmax;
pub use sum::sum;
pub use variance::variance;
