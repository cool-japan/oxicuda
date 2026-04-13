//! # Pruning
//!
//! Weight-pruning strategies for model compression.
//!
//! | Module      | Strategy                                               |
//! |-------------|--------------------------------------------------------|
//! | `mask`      | [`SparseMask`] — boolean weight mask primitives        |
//! | `magnitude` | [`MagnitudePruner`] — unstructured L1/L2 pruning       |
//! | `structured`| [`StructuredPruner`] — channel / filter / head pruning |

pub mod magnitude;
pub mod mask;
pub mod structured;

pub use magnitude::{MagnitudeNorm, MagnitudePruner};
pub use mask::SparseMask;
pub use structured::{PruneGranularity, StructuredPruner};
