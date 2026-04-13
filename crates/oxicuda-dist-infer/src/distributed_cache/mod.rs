//! Distributed PagedKV cache management.
//!
//! Extends `oxicuda_infer`'s single-GPU PagedKV cache to a multi-GPU setting.
//! Each rank maintains a local block pool; sequence ownership is partitioned
//! across ranks so memory pressure is distributed evenly.
//!
//! # Modules
//!
//! * `partition` — `CachePartition`: assigns sequences to ranks and
//!   rebalances ownership when ranks are overloaded.
//! * `migration` — `BlockMigrator`: handles cross-rank block migration
//!   for load balancing (simulated, no real P2P GPU copy).

pub mod migration;
pub mod partition;

pub use migration::BlockMigrator;
pub use partition::{CachePartition, SeqOwnership};
