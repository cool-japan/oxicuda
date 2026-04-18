//! High-level PTX kernel templates for common GPU operations.
//!
//! This module provides parameterized templates that generate complete PTX
//! kernels for standard GPU workloads: elementwise operations, reductions,
//! GEMM, softmax, attention, and batch normalization. Each template
//! encapsulates architecture-specific optimizations and produces correct
//! PTX text via the builder infrastructure.
//!
//! - [`crate::templates::elementwise`]: Unary and binary elementwise operations (add, relu, sigmoid, etc.)
//! - [`crate::templates::reduction`]: Parallel block-level reductions (sum, max, min, etc.)
//! - [`crate::templates::gemm`]: Matrix multiplication templates (stub for Vol.3)
//! - [`crate::templates::softmax`]: Numerically stable row-wise softmax
//! - [`crate::templates::attention`]: Simplified FlashAttention-style scaled dot-product attention
//! - [`crate::templates::batch_norm`]: Batch normalization (training and inference modes)

pub mod attention;
pub mod batch_norm;
pub mod broadcast;
pub mod convolution;
pub mod elementwise;
pub mod gemm;
pub mod moe;
pub mod reduction;
pub mod scan;
pub mod softmax;
pub mod transpose;

pub use broadcast::{BroadcastTemplate, MAX_BROADCAST_RANK};
