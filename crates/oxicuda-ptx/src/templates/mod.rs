//! High-level PTX kernel templates for common GPU operations.
//!
//! This module provides parameterized templates that generate complete PTX
//! kernels for standard GPU workloads: elementwise operations, reductions,
//! GEMM, softmax, attention, and batch normalization. Each template
//! encapsulates architecture-specific optimizations and produces correct
//! PTX text via the builder infrastructure.
//!
//! - [`crate::templates::elementwise`]: Unary and binary elementwise operations (add, relu, sigmoid, etc.)
//! - [`crate::templates::bias_add`]: Broadcast bias-add `out[i, j] = in[i, j] + bias[j]`
//! - [`crate::templates::channel_broadcast`]: `[1,C,1,1]`-vs-`[1,C,H,W]` /
//!   scalar-vs-tensor broadcast binary arithmetic, and per-channel `PRelu`
//! - [`crate::templates::cp_async_gen`]: Standalone `cp.async` global-to-shared copy and multi-stage pipeline generator
//! - [`crate::templates::reduction`]: Parallel block-level reductions (sum, max, min, etc.)
//! - [`crate::templates::gemm`]: Matrix multiplication templates (stub for Vol.3)
//! - [`crate::templates::softmax`]: Numerically stable row-wise softmax
//! - [`crate::templates::causal_softmax`]: Causal (masked) row-wise softmax for autoregressive attention
//! - [`crate::templates::attention`]: Simplified FlashAttention-style scaled dot-product attention
//! - [`crate::templates::batch_norm`]: Batch normalization (training and inference modes)
//! - [`crate::templates::tiled_mainloop`]: Reusable CTA-tiled, shared-memory-staged
//!   f32 GEMM mainloop, driven by caller-supplied operand address callbacks so
//!   one emitter serves both a plain GEMM and a convolution's implicit im2col

pub mod attention;
pub mod batch_norm;
pub mod bias_add;
pub mod broadcast;
pub mod causal_softmax;
pub mod channel_broadcast;
pub mod convolution;
pub mod cp_async_gen;
pub mod elementwise;
pub mod gemm;
pub mod moe;
pub mod reduction;
pub mod scan;
pub mod softmax;
pub mod tiled_mainloop;
pub mod transpose;

pub use bias_add::BiasAddTemplate;
pub use broadcast::{BroadcastTemplate, MAX_BROADCAST_RANK};
pub use channel_broadcast::{ChannelBroadcastOp, ChannelBroadcastTemplate, PReluTemplate};
pub use cp_async_gen::{CP_ASYNC_MIN_SM, CpAsyncCachePolicy, CpAsyncGenerator};
