//! # OxiCUDA DNN -- GPU-Accelerated Deep Learning Primitives
//!
//! This crate provides GPU-accelerated deep learning primitives,
//! serving as a pure Rust equivalent to cuDNN.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`error`] | Error types and `DnnResult<T>` alias |
//! | [`types`] | Tensor descriptors, layouts, activations, conv descriptors |
//! | [`handle`] | `DnnHandle` -- central entry point for all operations |
//! | [`conv`] | Convolution forward / backward / fused operations |

#![warn(clippy::all)]
#![warn(missing_docs)]

pub mod attn;
pub mod conv;
pub mod dynamic_batch;
pub mod error;
pub mod handle;
pub mod linear;
pub mod moe;
pub mod norm;
pub mod pool;
pub mod quantize;
pub mod resize;
pub mod rnn;
pub mod types;

pub(crate) mod ptx_helpers;
pub(crate) mod tensor_util;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use dynamic_batch::{
    BatchConfig, BatchDecision, BatchMetrics, BatchSlot, ContinuousBatcher, InferenceRequest,
    PagedKvManager, PreemptionPolicy, Priority, RequestId, SchedulingPolicy, SpeculativeDecoder,
    TokenBudgetAllocator,
};
pub use error::{DnnError, DnnResult};
pub use handle::DnnHandle;
pub use types::{
    Activation, ConvAlgorithm, ConvolutionDescriptor, TensorDesc, TensorDescMut, TensorLayout,
    pool_output_size,
};

/// Prelude module for convenient glob imports.
///
/// ```rust,no_run
/// use oxicuda_dnn::prelude::*;
/// ```
pub mod prelude {
    pub use crate::error::{DnnError, DnnResult};
    pub use crate::handle::DnnHandle;
    pub use crate::types::{
        Activation, ConvAlgorithm, ConvolutionDescriptor, TensorDesc, TensorDescMut, TensorLayout,
        pool_output_size,
    };
}
