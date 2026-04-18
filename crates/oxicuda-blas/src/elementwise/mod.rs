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
mod broadcast;
mod fill;
mod ops;
mod unary;

pub use binary::{
    add, cmp_eq, cmp_ge, cmp_gt, cmp_le, cmp_lt, cmp_ne, div, fused_add_relu, fused_scale_add, max,
    min, mul, nand, nor, or_max, or_prob_sum, pow, sub, xor,
};
pub use broadcast::broadcast_axes;
pub use fill::fill;
pub use ops::ElementwiseOp;
pub use unary::{
    abs_val, ceil, exp, floor, gelu, hard_sigmoid, hard_swish, leaky_relu, log, neg, one_minus,
    relu, rsqrt, scale, sigmoid, silu, softplus, sqrt, tanh_activation,
};
