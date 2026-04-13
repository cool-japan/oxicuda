//! # Prefix Cache
//!
//! Implements prompt/prefix caching: when multiple requests share a common
//! token prefix (e.g., a system prompt), the KV blocks computed for that
//! prefix can be reused across requests, avoiding redundant computation.
//!
//! ## Algorithm
//!
//! Prefixes are keyed by the FNV-1a hash of their token sequence.  On a
//! cache hit the associated physical [`BlockId`]s are returned and their
//! reference counts incremented in the [`crate::cache::kv_cache::PagedKvCache`].  An LRU eviction
//! policy removes entries when the cache exceeds `max_entries`.
//!
//! ## Limitations
//!
//! * Prefix matching is prefix-complete: the full prefix must match.
//!   Sub-prefix sharing (e.g., first 50 tokens of a 100-token prefix) can
//!   be layered on top if needed by the caller.
//! * The cache does **not** call into the KV cache directly; block lifecycle
//!   management (inc_ref / dec_ref) is the responsibility of the caller.

use std::collections::HashMap;

use crate::cache::kv_cache::BlockId;
use crate::error::{InferError, InferResult};

// ─── PrefixEntry ─────────────────────────────────────────────────────────────

/// A cached prefix: an ordered list of physical blocks covering `prefix_len`
/// tokens for **all** transformer layers.
#[derive(Debug, Clone)]
pub struct PrefixEntry {
    /// Physical block IDs covering the prefix (ordered).
    pub block_ids: Vec<BlockId>,
    /// Number of tokens in the prefix.
    pub prefix_len: usize,
    /// Total number of successful lookups for this entry.
    pub hit_count: u64,
    /// Logical clock at last access (for LRU eviction).
    pub last_access: u64,
}

// ─── PrefixCache ─────────────────────────────────────────────────────────────

/// LRU prefix cache keyed by the FNV-1a hash of the token sequence.
pub struct PrefixCache {
    /// Hash → cached entry.
    entries: HashMap<u64, PrefixEntry>,
    /// Maximum number of cached prefixes.
    max_entries: usize,
    /// Monotonically increasing logical clock.
    clock: u64,
    /// Total lookup attempts.
    total_queries: u64,
    /// Total successful lookups.
    total_hits: u64,
}

impl PrefixCache {
    /// Create a new prefix cache with the given capacity.
    ///
    /// # Panics
    /// Panics if `max_entries == 0`.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        assert!(max_entries > 0, "max_entries must be > 0");
        Self {
            entries: HashMap::new(),
            max_entries,
            clock: 0,
            total_queries: 0,
            total_hits: 0,
        }
    }

    // ── Hashing ──────────────────────────────────────────────────────────────

    /// FNV-1a hash of a token sequence.
    ///
    /// 64-bit FNV-1a: basis = 14695981039346656037, prime = 1099511628211.
    #[must_use]
    pub fn hash_tokens(tokens: &[u32]) -> u64 {
        const FNV_BASIS: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;
        let mut h = FNV_BASIS;
        for &tok in tokens {
            let bytes = tok.to_le_bytes();
            for b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
        h
    }

    // ── Lookup ───────────────────────────────────────────────────────────────

    /// Look up a prefix by its token sequence.
    ///
    /// On a hit, updates the access clock and hit counters, and returns a
    /// reference to the [`PrefixEntry`].
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<&PrefixEntry> {
        self.total_queries += 1;
        self.clock += 1;
        let key = Self::hash_tokens(tokens);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.hit_count += 1;
            entry.last_access = self.clock;
            self.total_hits += 1;
            // Re-borrow immutably to return a shared reference.
            return self.entries.get(&key);
        }
        None
    }

    // ── Insertion ────────────────────────────────────────────────────────────

    /// Insert a new prefix entry.
    ///
    /// If the cache is at capacity, evicts the least-recently-used entry.
    ///
    /// Returns `false` if insertion was skipped (e.g., entry already present).
    pub fn insert(&mut self, tokens: &[u32], block_ids: Vec<BlockId>) -> bool {
        let key = Self::hash_tokens(tokens);
        if self.entries.contains_key(&key) {
            return false; // Already cached; don't overwrite.
        }
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }
        self.clock += 1;
        self.entries.insert(
            key,
            PrefixEntry {
                block_ids,
                prefix_len: tokens.len(),
                hit_count: 0,
                last_access: self.clock,
            },
        );
        true
    }

    // ── Eviction ─────────────────────────────────────────────────────────────

    /// Evict the least-recently-used prefix entry.
    ///
    /// Returns the evicted entry's block IDs so the caller can decrement
    /// reference counts in the KV cache, or `None` if the cache is empty.
    pub fn evict_lru(&mut self) -> Option<Vec<BlockId>> {
        if self.entries.is_empty() {
            return None;
        }
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, v)| v.last_access)
            .map(|(&k, _)| k)
            .expect("entries non-empty");
        self.entries.remove(&lru_key).map(|e| e.block_ids)
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    /// Cache hit rate (0.0 if no queries have been made).
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.total_hits as f64 / self.total_queries as f64
        }
    }

    /// Number of entries currently cached.
    #[must_use]
    pub fn n_entries(&self) -> usize {
        self.entries.len()
    }

    /// Remove all cached entries.  Returns the evicted block ID lists.
    pub fn clear(&mut self) -> Vec<Vec<BlockId>> {
        self.entries.drain().map(|(_, e)| e.block_ids).collect()
    }

    /// Look up a cached entry by hash key (for testing / introspection).
    pub fn lookup_by_hash(&self, hash: u64) -> Option<&PrefixEntry> {
        self.entries.get(&hash)
    }

    /// Remove a specific entry by its token sequence; return the block IDs.
    pub fn remove(&mut self, tokens: &[u32]) -> InferResult<Vec<BlockId>> {
        let key = Self::hash_tokens(tokens);
        self.entries
            .remove(&key)
            .map(|e| e.block_ids)
            .ok_or_else(|| {
                InferError::PrefixCacheError(format!("prefix not found (hash {key:#018x})"))
            })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_blocks(n: usize) -> Vec<BlockId> {
        (0..n as u32).map(BlockId).collect()
    }

    #[test]
    fn insert_and_lookup() {
        let mut cache = PrefixCache::new(16);
        let tokens = vec![1_u32, 2, 3, 4];
        cache.insert(&tokens, fake_blocks(2));
        let entry = cache.lookup(&tokens).expect("should hit");
        assert_eq!(entry.prefix_len, 4);
        assert_eq!(entry.block_ids.len(), 2);
    }

    #[test]
    fn miss_returns_none() {
        let mut cache = PrefixCache::new(16);
        assert!(cache.lookup(&[9, 8, 7]).is_none());
    }

    #[test]
    fn hit_rate_tracking() {
        let mut cache = PrefixCache::new(16);
        let t = vec![1_u32, 2, 3];
        cache.insert(&t, fake_blocks(1));
        cache.lookup(&t); // hit
        cache.lookup(&[99_u32]); // miss
        // 1 hit out of 2 queries
        assert!((cache.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn lru_eviction_on_capacity() {
        let mut cache = PrefixCache::new(2);
        cache.insert(&[1], fake_blocks(1));
        cache.lookup(&[1]); // access entry 1 → recent
        cache.insert(&[2], fake_blocks(1));
        // Cache is now full; inserting a third entry should evict LRU.
        cache.insert(&[3], fake_blocks(1));
        assert_eq!(cache.n_entries(), 2);
    }

    #[test]
    fn duplicate_insert_ignored() {
        let mut cache = PrefixCache::new(16);
        let t = vec![5_u32, 6];
        let inserted = cache.insert(&t, fake_blocks(1));
        assert!(inserted);
        let inserted2 = cache.insert(&t, fake_blocks(2)); // same key
        assert!(!inserted2);
        assert_eq!(cache.n_entries(), 1);
    }

    #[test]
    fn fnv_hash_deterministic() {
        let t = vec![10_u32, 20, 30];
        let h1 = PrefixCache::hash_tokens(&t);
        let h2 = PrefixCache::hash_tokens(&t);
        assert_eq!(h1, h2);
    }

    #[test]
    fn fnv_hash_different_sequences() {
        let h1 = PrefixCache::hash_tokens(&[1, 2, 3]);
        let h2 = PrefixCache::hash_tokens(&[3, 2, 1]);
        assert_ne!(h1, h2, "different orderings should hash differently");
    }

    #[test]
    fn remove_existing_entry() {
        let mut cache = PrefixCache::new(16);
        let t = vec![7_u32, 8];
        cache.insert(&t, fake_blocks(2));
        let blocks = cache.remove(&t).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(cache.n_entries(), 0);
    }

    #[test]
    fn remove_missing_entry_error() {
        let mut cache = PrefixCache::new(16);
        assert!(cache.remove(&[99_u32]).is_err());
    }

    #[test]
    fn clear_returns_all_blocks() {
        let mut cache = PrefixCache::new(16);
        cache.insert(&[1], fake_blocks(2));
        cache.insert(&[2], fake_blocks(3));
        let all_blocks = cache.clear();
        let total: usize = all_blocks.iter().map(Vec::len).sum();
        assert_eq!(total, 5);
        assert_eq!(cache.n_entries(), 0);
    }
}
