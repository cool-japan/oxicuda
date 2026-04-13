//! Forward convolution (fprop) implementations.
//!
//! Each sub-module provides a different algorithm for computing the
//! forward pass of a convolution operation.

pub mod direct;
pub mod im2col_gemm;
pub mod implicit_gemm;
pub mod winograd;
