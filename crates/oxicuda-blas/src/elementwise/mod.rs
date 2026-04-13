//! Elementwise GPU operations for OxiCUDA BLAS.
//!
//! This module provides unary and binary elementwise operations over device
//! buffers, including activation functions (ReLU, GELU, sigmoid, SiLU, tanh),
//! scaling, and fused operations (add+relu, scale+add).
//!
//! Each function generates PTX on the fly via [`oxicuda_ptx::templates::elementwise::ElementwiseTemplate`] from
//! `oxicuda-ptx`, loads the resulting module, and launches the kernel on the
//! handle's stream.

mod binary;
mod ops;
mod unary;

pub use binary::{add, fused_add_relu, fused_scale_add, mul};
pub use ops::ElementwiseOp;
pub use unary::{gelu, relu, scale, sigmoid, silu, tanh_activation};
