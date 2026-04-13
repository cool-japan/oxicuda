//! Backward data gradient (dgrad) implementations.
//!
//! Computes the gradient of the loss with respect to the input tensor.

pub mod implicit_gemm;
pub mod winograd;
