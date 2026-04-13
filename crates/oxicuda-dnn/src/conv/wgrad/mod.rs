//! Backward filter gradient (wgrad) implementations.
//!
//! Computes the gradient of the loss with respect to the filter weights.

pub mod implicit_gemm;
pub mod winograd;
