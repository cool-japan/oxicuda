//! Optimisation passes for the OxiCUDA computation graph.
//!
//! Three complementary passes transform a validated [`ComputeGraph`] into
//! a form ready for execution:
//!
//! * [`fusion`] — Identifies chains of element-wise kernels that can be
//!   merged into a single PTX kernel, reducing kernel-launch overhead and
//!   improving cache behaviour.
//!
//! * [`memory`] — Assigns device memory slots to logical buffers using
//!   live-interval graph colouring, maximising buffer reuse and minimising
//!   peak device memory usage.
//!
//! * [`stream`] — Partitions independent nodes onto separate CUDA streams
//!   for concurrent GPU execution, and identifies the cross-stream event
//!   synchronisation points needed to enforce dependencies.
//!
//! [`ComputeGraph`]: crate::graph::ComputeGraph

pub mod fusion;
pub mod memory;
pub mod stream;

pub use fusion::{FusionGroup, FusionPlan, analyse as fusion_analyse};
pub use memory::{MemoryPlan, SlotAssignment, analyse as memory_analyse};
pub use stream::{StreamPlan, SyncPoint, analyse as stream_analyse};
