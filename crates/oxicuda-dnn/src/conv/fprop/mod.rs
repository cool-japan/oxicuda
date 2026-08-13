//! Forward convolution (fprop) implementations.
//!
//! Each sub-module provides a different algorithm for computing the
//! forward pass of a convolution operation.
//!
//! [`implicit_gemm`] and [`tiled_implicit_gemm`] compute the *same* thing by
//! the same index mapping: the former one thread per output element (the
//! numeric oracle, and the fallback for every configuration the tiling
//! declines), the latter a CTA-tiled, shared-memory-staged GEMM mainloop.
//! [`ImplicitGemmConv::execute`](implicit_gemm::ImplicitGemmConv::execute)
//! routes between them, so callers never choose.

pub mod direct;
pub mod im2col_gemm;
pub mod implicit_gemm;
pub(crate) mod standard_conv;
pub mod tiled_implicit_gemm;
pub mod winograd;
