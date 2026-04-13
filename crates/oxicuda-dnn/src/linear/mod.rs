//! Fused linear (fully-connected) layer operations.
//!
//! Provides GPU-accelerated fused GEMM + bias + activation kernels for
//! dense layers in neural networks. By fusing these operations into a
//! single kernel pass, we eliminate intermediate memory round-trips for
//! the bias addition and activation function.
//!
//! | Sub-module       | Description                                     |
//! |------------------|-------------------------------------------------|
//! | [`mod@fused_linear`] | Fused `Y = activation(X @ W^T + bias)` kernel  |

pub mod fused_linear;

pub use fused_linear::{FusedLinearConfig, fused_linear};
