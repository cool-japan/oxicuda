//! Block-level parallel primitives using shared memory and warp shuffles.
//!
//! Block primitives aggregate values across all threads in a single thread
//! block using shared memory as the communication medium and warp-level
//! shuffle operations for the final intra-warp reduction.
//!
//! # Modules
//!
//! * [`reduce`] — block-wide reduction (two-stage: per-warp via shfl, then warp 0 reduces partials)
//! * [`scan`] — block-wide prefix scan (Blelloch work-efficient algorithm)

pub mod reduce;
pub mod scan;

pub use reduce::{BlockReduceConfig, BlockReduceTemplate, MAX_BLOCK_SIZE};
pub use scan::{BlockScanConfig, BlockScanTemplate};
