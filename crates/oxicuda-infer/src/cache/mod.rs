//! KV cache subsystem — PagedAttention block management.
//!
//! Modules:
//! * [`kv_cache`]       — physical block pool and per-block K/V storage.
//! * [`cache_manager`]  — per-sequence block table management.
//! * [`prefix_cache`]   — LRU prefix-sharing cache for prompt reuse.

pub mod cache_manager;
pub mod kv_cache;
pub mod prefix_cache;

pub use cache_manager::CacheManager;
pub use kv_cache::{BlockId, KvBlock, PagedKvCache};
pub use prefix_cache::{PrefixCache, PrefixEntry};
