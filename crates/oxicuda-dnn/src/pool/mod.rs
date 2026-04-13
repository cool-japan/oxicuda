//! Pooling operations for DNN.
//!
//! This module provides GPU-accelerated 2D pooling primitives:
//!
//! - [`max_pool`] — Max pooling with optional index tracking for backpropagation.
//! - [`avg_pool`] — Average pooling with configurable padding inclusion.
//! - [`adaptive_pool`] — Adaptive pooling that adjusts kernel/stride automatically.
//! - [`global_pool`] — Global pooling (reduces spatial dims to 1x1).

pub mod adaptive_pool;
pub mod avg_pool;
pub mod global_pool;
pub mod max_pool;

pub use adaptive_pool::{adaptive_avg_pool2d, adaptive_max_pool2d};
pub use avg_pool::avg_pool2d;
pub use global_pool::{global_avg_pool2d, global_max_pool2d};
pub use max_pool::{max_pool2d, max_pool2d_backward};
