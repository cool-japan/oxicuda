//! Device-wide parallel primitives operating across all thread blocks.
//!
//! This module provides CUB-equivalent algorithms that aggregate or transform
//! entire device arrays:
//!
//! * [`reduce`] — compute a single aggregate value from all elements
//! * [`scan`]   — compute a prefix sum / prefix op across all elements
//! * [`select`] — stream compaction: keep only elements satisfying a predicate
//! * [`histogram`] — count elements in each equal-width bin

pub mod histogram;
pub mod reduce;
pub mod scan;
pub mod select;

pub use histogram::{DeviceHistogramConfig, DeviceHistogramMode, DeviceHistogramTemplate};
pub use reduce::{DEFAULT_BLOCK_SIZE, DeviceReduceConfig, DeviceReduceTemplate};
pub use scan::{DeviceScanConfig, DeviceScanTemplate};
pub use select::{DeviceSelectConfig, DeviceSelectTemplate, SelectPredicate};
