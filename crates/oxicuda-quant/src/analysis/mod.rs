//! # Quantization Analysis Tools
//!
//! Utilities for understanding and optimising model compression.
//!
//! | Module        | Contents                                               |
//! |---------------|--------------------------------------------------------|
//! | `sensitivity` | [`LayerSensitivity`], [`SensitivityAnalyzer`]          |
//! | `metrics`     | [`CompressionMetrics`], [`ModelCompressionMetrics`]    |
//! | `policy`      | [`MixedPrecisionPolicy`] — greedy bit-width assignment |

pub mod metrics;
pub mod policy;
pub mod sensitivity;

pub use metrics::{CompressionMetrics, ModelCompressionMetrics};
pub use policy::MixedPrecisionPolicy;
pub use sensitivity::{LayerSensitivity, SensitivityAnalyzer};
