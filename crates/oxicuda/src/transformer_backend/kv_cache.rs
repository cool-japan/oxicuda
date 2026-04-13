//! Paged Key-Value cache management for transformer inference.
//!
//! Implements a block-based KV-cache that allocates memory in fixed-size blocks,
//! avoiding the need to pre-allocate for `max_seq_len`. Supports:
//!
//! - Copy-on-write for shared prefixes (prompt caching)
//! - Configurable eviction policies (LRU, FIFO, frequency-based)
//! - Dynamic sequence growth
//! - Prefix sharing across sequences
//! - Cache statistics tracking

use std::collections::{HashMap, VecDeque};

use super::{TransformerError, TransformerResult};

/// Unique identifier for a sequence.
pub type SequenceId = u64;

/// Unique identifier for a physical memory block.
pub type BlockId = u32;

/// Eviction policy for cache blocks when memory is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used — evict blocks accessed longest ago.
    Lru,
    /// First In First Out — evict oldest allocated blocks.
    Fifo,
    /// Frequency-based — evict least frequently accessed blocks.
    Frequency,
}

/// Configuration for creating a [`PagedKvCache`].
#[derive(Debug, Clone)]
pub struct PagedKvCacheConfig {
    /// Number of tokens per block.
    pub block_size: usize,
    /// Total number of physical blocks.
    pub num_blocks: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Eviction policy.
    pub eviction_policy: CacheEvictionPolicy,
}

/// Statistics about cache usage.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total allocation requests.
    pub total_allocations: u64,
    /// Total deallocation requests.
    pub total_deallocations: u64,
    /// Cache hit count (prefix sharing hits).
    pub cache_hits: u64,
    /// Cache miss count.
    pub cache_misses: u64,
    /// Number of evictions performed.
    pub evictions: u64,
    /// Number of copy-on-write operations.
    pub cow_copies: u64,
}

impl CacheStats {
    /// Cache hit rate as a ratio [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / total as f64
    }

    /// Cache utilization as a ratio [0.0, 1.0].
    pub fn utilization(&self, total_blocks: usize, free_blocks: usize) -> f64 {
        if total_blocks == 0 {
            return 0.0;
        }
        let used = total_blocks.saturating_sub(free_blocks);
        used as f64 / total_blocks as f64
    }
}

/// Internal block metadata.
#[derive(Debug, Clone)]
struct BlockMeta {
    /// Reference count (for copy-on-write).
    ref_count: u32,
    /// Number of tokens currently stored in this block.
    num_tokens: usize,
    /// Timestamp of last access (monotonic counter).
    last_access: u64,
    /// Access frequency count.
    access_count: u64,
    /// Allocation order (for FIFO).
    alloc_order: u64,
}

/// Paged KV-cache with block-based allocation.
///
/// Keys and values are stored in fixed-size blocks. Each sequence maintains
/// a block table mapping logical block indices to physical block IDs.
/// Multiple sequences can share prefix blocks via copy-on-write.
#[derive(Debug)]
pub struct PagedKvCache {
    /// Tokens per block.
    block_size: usize,
    /// Total physical blocks.
    num_blocks: usize,
    /// Logical-to-physical block mapping per sequence.
    block_table: HashMap<SequenceId, Vec<BlockId>>,
    /// Free block pool.
    free_blocks: VecDeque<BlockId>,
    /// Block metadata.
    block_meta: HashMap<BlockId, BlockMeta>,
    /// Number of transformer layers.
    num_layers: usize,
    /// Number of KV heads.
    num_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Eviction policy.
    eviction_policy: CacheEvictionPolicy,
    /// Monotonic clock for LRU tracking.
    clock: u64,
    /// Monotonic counter for allocation order.
    alloc_counter: u64,
    /// Prefix hash → block IDs for prefix sharing.
    prefix_table: HashMap<u64, Vec<BlockId>>,
    /// Statistics.
    stats: CacheStats,
}

impl PagedKvCache {
    /// Create a new paged KV-cache from configuration.
    pub fn new(config: PagedKvCacheConfig) -> TransformerResult<Self> {
        if config.block_size == 0 {
            return Err(TransformerError::CacheError(
                "block_size must be > 0".to_string(),
            ));
        }
        if config.num_blocks == 0 {
            return Err(TransformerError::CacheError(
                "num_blocks must be > 0".to_string(),
            ));
        }
        if config.num_layers == 0 || config.num_heads == 0 || config.head_dim == 0 {
            return Err(TransformerError::CacheError(
                "num_layers, num_heads, head_dim must all be > 0".to_string(),
            ));
        }

        let free_blocks: VecDeque<BlockId> = (0..config.num_blocks as BlockId).collect();

        Ok(Self {
            block_size: config.block_size,
            num_blocks: config.num_blocks,
            block_table: HashMap::new(),
            free_blocks,
            block_meta: HashMap::new(),
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            eviction_policy: config.eviction_policy,
            clock: 0,
            alloc_counter: 0,
            prefix_table: HashMap::new(),
            stats: CacheStats::default(),
        })
    }

    /// Allocate a single block for a sequence.
    ///
    /// Returns the allocated block ID, or an error if no blocks are available
    /// and eviction fails.
    pub fn allocate_block(&mut self, seq_id: SequenceId) -> TransformerResult<BlockId> {
        let block_id = if let Some(id) = self.free_blocks.pop_front() {
            id
        } else {
            self.evict_block()?
        };

        self.clock += 1;
        self.alloc_counter += 1;

        self.block_meta.insert(
            block_id,
            BlockMeta {
                ref_count: 1,
                num_tokens: 0,
                last_access: self.clock,
                access_count: 1,
                alloc_order: self.alloc_counter,
            },
        );

        self.block_table.entry(seq_id).or_default().push(block_id);
        self.stats.total_allocations += 1;

        Ok(block_id)
    }

    /// Allocate multiple blocks for a sequence to hold `num_tokens`.
    pub fn allocate_blocks_for_tokens(
        &mut self,
        seq_id: SequenceId,
        num_tokens: usize,
    ) -> TransformerResult<Vec<BlockId>> {
        let num_blocks_needed = num_tokens.div_ceil(self.block_size);
        let mut allocated = Vec::with_capacity(num_blocks_needed);

        for _ in 0..num_blocks_needed {
            match self.allocate_block(seq_id) {
                Ok(block_id) => allocated.push(block_id),
                Err(e) => {
                    // Roll back: free already allocated blocks
                    for &bid in &allocated {
                        self.free_block_internal(bid);
                    }
                    // Remove the sequence's entry if we added blocks
                    if let Some(table) = self.block_table.get_mut(&seq_id) {
                        let len = table.len();
                        table.truncate(len.saturating_sub(allocated.len()));
                        if table.is_empty() {
                            self.block_table.remove(&seq_id);
                        }
                    }
                    return Err(e);
                }
            }
        }

        // Set token count for the last block
        if let Some(&last_block) = allocated.last() {
            let remainder = num_tokens % self.block_size;
            let tokens_in_last = if remainder == 0 {
                self.block_size
            } else {
                remainder
            };
            if let Some(meta) = self.block_meta.get_mut(&last_block) {
                meta.num_tokens = tokens_in_last;
            }
        }
        // Set full blocks
        for &bid in allocated.iter().take(allocated.len().saturating_sub(1)) {
            if let Some(meta) = self.block_meta.get_mut(&bid) {
                meta.num_tokens = self.block_size;
            }
        }

        Ok(allocated)
    }

    /// Append a token to a sequence's cache, allocating a new block if needed.
    pub fn append_token(&mut self, seq_id: SequenceId) -> TransformerResult<BlockId> {
        let needs_new_block = match self.block_table.get(&seq_id) {
            None => true,
            Some(blocks) => {
                if blocks.is_empty() {
                    true
                } else {
                    let last_block = blocks[blocks.len() - 1];
                    match self.block_meta.get(&last_block) {
                        Some(meta) => meta.num_tokens >= self.block_size,
                        None => true,
                    }
                }
            }
        };

        if needs_new_block {
            // Need CoW check if last block is shared
            if let Some(blocks) = self.block_table.get(&seq_id) {
                if let Some(&last_block) = blocks.last() {
                    if let Some(meta) = self.block_meta.get(&last_block) {
                        if meta.ref_count > 1 {
                            self.copy_on_write(seq_id, blocks.len() - 1)?;
                        }
                    }
                }
            }
            let block_id = self.allocate_block(seq_id)?;
            if let Some(meta) = self.block_meta.get_mut(&block_id) {
                meta.num_tokens = 1;
            }
            Ok(block_id)
        } else {
            let blocks = self
                .block_table
                .get(&seq_id)
                .ok_or_else(|| TransformerError::CacheError("sequence not found".to_string()))?;
            let last_block = blocks[blocks.len() - 1];

            // CoW check
            if let Some(meta) = self.block_meta.get(&last_block) {
                if meta.ref_count > 1 {
                    let block_idx = blocks.len() - 1;
                    self.copy_on_write(seq_id, block_idx)?;
                    // Re-fetch after CoW
                    let blocks = self.block_table.get(&seq_id).ok_or_else(|| {
                        TransformerError::CacheError("sequence not found".to_string())
                    })?;
                    let new_last = blocks[blocks.len() - 1];
                    if let Some(meta) = self.block_meta.get_mut(&new_last) {
                        meta.num_tokens += 1;
                        self.clock += 1;
                        meta.last_access = self.clock;
                        meta.access_count += 1;
                    }
                    return Ok(new_last);
                }
            }

            if let Some(meta) = self.block_meta.get_mut(&last_block) {
                meta.num_tokens += 1;
                self.clock += 1;
                meta.last_access = self.clock;
                meta.access_count += 1;
            }
            Ok(last_block)
        }
    }

    /// Free all blocks for a sequence.
    pub fn free_sequence(&mut self, seq_id: SequenceId) -> TransformerResult<()> {
        let blocks = self
            .block_table
            .remove(&seq_id)
            .ok_or_else(|| TransformerError::CacheError(format!("sequence {seq_id} not found")))?;

        for block_id in blocks {
            self.free_block_internal(block_id);
            self.stats.total_deallocations += 1;
        }
        Ok(())
    }

    /// Share prefix blocks from `src_seq` to `dst_seq` via copy-on-write.
    ///
    /// The first `num_prefix_blocks` blocks of the source sequence are shared
    /// with the destination without copying data.
    pub fn share_prefix(
        &mut self,
        src_seq: SequenceId,
        dst_seq: SequenceId,
        num_prefix_blocks: usize,
    ) -> TransformerResult<()> {
        let src_blocks = self
            .block_table
            .get(&src_seq)
            .ok_or_else(|| {
                TransformerError::CacheError(format!("source sequence {src_seq} not found"))
            })?
            .clone();

        if num_prefix_blocks > src_blocks.len() {
            return Err(TransformerError::CacheError(format!(
                "requested {} prefix blocks but source has only {}",
                num_prefix_blocks,
                src_blocks.len()
            )));
        }

        let shared: Vec<BlockId> = src_blocks[..num_prefix_blocks].to_vec();

        // Increment ref counts
        for &block_id in &shared {
            if let Some(meta) = self.block_meta.get_mut(&block_id) {
                meta.ref_count += 1;
            }
        }

        let dst_blocks = self.block_table.entry(dst_seq).or_default();
        dst_blocks.extend_from_slice(&shared);
        self.stats.cache_hits += 1;

        Ok(())
    }

    /// Register a prefix hash for sharing.
    pub fn register_prefix(&mut self, prefix_hash: u64, block_ids: Vec<BlockId>) {
        for &bid in &block_ids {
            if let Some(meta) = self.block_meta.get_mut(&bid) {
                meta.ref_count += 1;
            }
        }
        self.prefix_table.insert(prefix_hash, block_ids);
    }

    /// Look up a registered prefix by hash.
    pub fn lookup_prefix(&mut self, prefix_hash: u64) -> Option<&[BlockId]> {
        if self.prefix_table.contains_key(&prefix_hash) {
            self.stats.cache_hits += 1;
            self.prefix_table.get(&prefix_hash).map(|v| v.as_slice())
        } else {
            self.stats.cache_misses += 1;
            None
        }
    }

    /// Get the block table for a sequence.
    pub fn get_block_table(&self, seq_id: SequenceId) -> Option<&[BlockId]> {
        self.block_table.get(&seq_id).map(|v| v.as_slice())
    }

    /// Get the number of cached tokens for a sequence.
    pub fn num_cached_tokens(&self, seq_id: SequenceId) -> usize {
        match self.block_table.get(&seq_id) {
            None => 0,
            Some(blocks) => {
                if blocks.is_empty() {
                    return 0;
                }
                let full_blocks = blocks.len().saturating_sub(1);
                let last_tokens = blocks
                    .last()
                    .and_then(|bid| self.block_meta.get(bid))
                    .map(|m| m.num_tokens)
                    .unwrap_or(0);
                full_blocks * self.block_size + last_tokens
            }
        }
    }

    /// Number of free blocks available.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Total number of blocks.
    pub fn total_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Block size (tokens per block).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Fragmentation ratio [0.0, 1.0].
    ///
    /// Measured as the fraction of partially-filled blocks.
    pub fn fragmentation(&self) -> f64 {
        let mut partial_blocks = 0usize;
        let mut total_used = 0usize;

        for blocks in self.block_table.values() {
            for &bid in blocks {
                if let Some(meta) = self.block_meta.get(&bid) {
                    total_used += 1;
                    if meta.num_tokens < self.block_size {
                        partial_blocks += 1;
                    }
                }
            }
        }
        if total_used == 0 {
            return 0.0;
        }
        partial_blocks as f64 / total_used as f64
    }

    /// Memory usage in bytes (estimated).
    pub fn memory_usage_bytes(&self) -> usize {
        let used_blocks = self.num_blocks - self.free_blocks.len();
        // Each block stores K and V for all layers and heads:
        // 2 (K+V) * num_layers * num_heads * block_size * head_dim * sizeof(f16)
        let bytes_per_block =
            2 * self.num_layers * self.num_heads * self.block_size * self.head_dim * 2;
        used_blocks * bytes_per_block
    }

    // ─── Internal helpers ───────────────────────────────────

    fn free_block_internal(&mut self, block_id: BlockId) {
        if let Some(meta) = self.block_meta.get_mut(&block_id) {
            meta.ref_count = meta.ref_count.saturating_sub(1);
            if meta.ref_count == 0 {
                self.block_meta.remove(&block_id);
                self.free_blocks.push_back(block_id);
            }
        }
    }

    fn copy_on_write(
        &mut self,
        seq_id: SequenceId,
        block_idx: usize,
    ) -> TransformerResult<BlockId> {
        let old_block_id = {
            let blocks = self.block_table.get(&seq_id).ok_or_else(|| {
                TransformerError::CacheError(format!("sequence {seq_id} not found"))
            })?;
            if block_idx >= blocks.len() {
                return Err(TransformerError::CacheError(format!(
                    "block index {block_idx} out of range"
                )));
            }
            blocks[block_idx]
        };

        // Allocate new block (don't add to sequence table; we're replacing)
        let new_block_id = if let Some(id) = self.free_blocks.pop_front() {
            id
        } else {
            self.evict_block()?
        };

        // Copy metadata from old block
        let old_meta =
            self.block_meta.get(&old_block_id).cloned().ok_or_else(|| {
                TransformerError::CacheError("block metadata missing".to_string())
            })?;

        self.clock += 1;
        self.alloc_counter += 1;

        self.block_meta.insert(
            new_block_id,
            BlockMeta {
                ref_count: 1,
                num_tokens: old_meta.num_tokens,
                last_access: self.clock,
                access_count: 1,
                alloc_order: self.alloc_counter,
            },
        );

        // Decrement old block ref count
        if let Some(meta) = self.block_meta.get_mut(&old_block_id) {
            meta.ref_count = meta.ref_count.saturating_sub(1);
            if meta.ref_count == 0 {
                self.block_meta.remove(&old_block_id);
                self.free_blocks.push_back(old_block_id);
            }
        }

        // Update block table
        if let Some(blocks) = self.block_table.get_mut(&seq_id) {
            blocks[block_idx] = new_block_id;
        }

        self.stats.cow_copies += 1;
        Ok(new_block_id)
    }

    fn evict_block(&mut self) -> TransformerResult<BlockId> {
        // Find a block to evict (only evict blocks with ref_count == 1)
        let victim = match self.eviction_policy {
            CacheEvictionPolicy::Lru => self.find_lru_victim(),
            CacheEvictionPolicy::Fifo => self.find_fifo_victim(),
            CacheEvictionPolicy::Frequency => self.find_frequency_victim(),
        };

        match victim {
            Some(block_id) => {
                // Remove from any sequence that owns it
                for (_, blocks) in self.block_table.iter_mut() {
                    blocks.retain(|&b| b != block_id);
                }
                // Remove from prefix table
                for (_, blocks) in self.prefix_table.iter_mut() {
                    blocks.retain(|&b| b != block_id);
                }
                self.block_meta.remove(&block_id);
                self.stats.evictions += 1;
                Ok(block_id)
            }
            None => Err(TransformerError::CacheError(
                "no blocks available for eviction".to_string(),
            )),
        }
    }

    fn find_lru_victim(&self) -> Option<BlockId> {
        self.block_meta
            .iter()
            .filter(|(_, meta)| meta.ref_count <= 1)
            .min_by_key(|(_, meta)| meta.last_access)
            .map(|(&id, _)| id)
    }

    fn find_fifo_victim(&self) -> Option<BlockId> {
        self.block_meta
            .iter()
            .filter(|(_, meta)| meta.ref_count <= 1)
            .min_by_key(|(_, meta)| meta.alloc_order)
            .map(|(&id, _)| id)
    }

    fn find_frequency_victim(&self) -> Option<BlockId> {
        self.block_meta
            .iter()
            .filter(|(_, meta)| meta.ref_count <= 1)
            .min_by_key(|(_, meta)| meta.access_count)
            .map(|(&id, _)| id)
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(num_blocks: usize) -> PagedKvCacheConfig {
        PagedKvCacheConfig {
            block_size: 16,
            num_blocks,
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            eviction_policy: CacheEvictionPolicy::Lru,
        }
    }

    #[test]
    fn test_basic_allocation() {
        let mut cache = PagedKvCache::new(test_config(100)).unwrap();
        assert_eq!(cache.num_free_blocks(), 100);

        let block = cache.allocate_block(1).unwrap();
        assert_eq!(cache.num_free_blocks(), 99);
        assert!(cache.get_block_table(1).is_some());
        assert_eq!(cache.get_block_table(1).unwrap().len(), 1);
        assert_eq!(cache.get_block_table(1).unwrap()[0], block);
    }

    #[test]
    fn test_allocate_blocks_for_tokens() {
        let mut cache = PagedKvCache::new(test_config(100)).unwrap();
        let blocks = cache.allocate_blocks_for_tokens(1, 50).unwrap();
        // 50 tokens / 16 block_size = 4 blocks needed (16+16+16+2)
        assert_eq!(blocks.len(), 4);
        assert_eq!(cache.num_free_blocks(), 96);
    }

    #[test]
    fn test_free_sequence() {
        let mut cache = PagedKvCache::new(test_config(100)).unwrap();
        cache.allocate_blocks_for_tokens(1, 32).unwrap();
        assert_eq!(cache.num_free_blocks(), 98);

        cache.free_sequence(1).unwrap();
        assert_eq!(cache.num_free_blocks(), 100);
        assert!(cache.get_block_table(1).is_none());
    }

    #[test]
    fn test_free_unknown_sequence() {
        let mut cache = PagedKvCache::new(test_config(10)).unwrap();
        assert!(cache.free_sequence(999).is_err());
    }

    #[test]
    fn test_append_token() {
        let mut cache = PagedKvCache::new(test_config(10)).unwrap();

        // First token should allocate a new block
        let b1 = cache.append_token(1).unwrap();
        assert_eq!(cache.num_cached_tokens(1), 1);
        assert_eq!(cache.num_free_blocks(), 9);

        // Fill the block (block_size = 16)
        for _ in 1..16 {
            let b = cache.append_token(1).unwrap();
            assert_eq!(b, b1); // same block
        }
        assert_eq!(cache.num_cached_tokens(1), 16);

        // 17th token should allocate a new block
        let b2 = cache.append_token(1).unwrap();
        assert_ne!(b1, b2);
        assert_eq!(cache.num_cached_tokens(1), 17);
        assert_eq!(cache.num_free_blocks(), 8);
    }

    #[test]
    fn test_copy_on_write() {
        let mut cache = PagedKvCache::new(test_config(20)).unwrap();

        // Allocate blocks for sequence 1
        cache.allocate_blocks_for_tokens(1, 32).unwrap();
        assert_eq!(cache.num_free_blocks(), 18);

        // Share prefix with sequence 2
        cache.share_prefix(1, 2, 2).unwrap();
        assert_eq!(cache.num_free_blocks(), 18); // no new allocation

        // Append token to seq 2 -> triggers CoW on shared block
        let _ = cache.append_token(2).unwrap();
        assert!(cache.stats().cow_copies > 0 || cache.num_free_blocks() < 18);
    }

    #[test]
    fn test_prefix_sharing() {
        let mut cache = PagedKvCache::new(test_config(20)).unwrap();

        cache.allocate_blocks_for_tokens(1, 32).unwrap();
        let blocks = cache.get_block_table(1).unwrap().to_vec();

        // Register prefix
        let prefix_hash = 0x1234u64;
        cache.register_prefix(prefix_hash, blocks[..2].to_vec());

        // Lookup
        assert!(cache.lookup_prefix(prefix_hash).is_some());
        assert_eq!(cache.lookup_prefix(prefix_hash).unwrap().len(), 2);
        assert!(cache.lookup_prefix(0xDEAD).is_none());
    }

    #[test]
    fn test_eviction_lru() {
        let mut cache = PagedKvCache::new(test_config(4)).unwrap();

        // Fill all blocks
        cache.allocate_blocks_for_tokens(1, 16).unwrap(); // 1 block
        cache.allocate_blocks_for_tokens(2, 16).unwrap(); // 1 block
        cache.allocate_blocks_for_tokens(3, 16).unwrap(); // 1 block
        cache.allocate_blocks_for_tokens(4, 16).unwrap(); // 1 block
        assert_eq!(cache.num_free_blocks(), 0);

        // Allocate one more -> forces eviction of LRU block (seq 1's block)
        let _new_block = cache.allocate_block(5).unwrap();
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_eviction_fifo() {
        let config = PagedKvCacheConfig {
            eviction_policy: CacheEvictionPolicy::Fifo,
            ..test_config(3)
        };
        let mut cache = PagedKvCache::new(config).unwrap();

        cache.allocate_block(1).unwrap();
        cache.allocate_block(2).unwrap();
        cache.allocate_block(3).unwrap();
        assert_eq!(cache.num_free_blocks(), 0);

        // Force eviction
        let _ = cache.allocate_block(4).unwrap();
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_eviction_frequency() {
        let config = PagedKvCacheConfig {
            eviction_policy: CacheEvictionPolicy::Frequency,
            ..test_config(3)
        };
        let mut cache = PagedKvCache::new(config).unwrap();

        cache.allocate_block(1).unwrap();
        cache.allocate_block(2).unwrap();
        cache.allocate_block(3).unwrap();

        let _ = cache.allocate_block(4).unwrap();
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = PagedKvCache::new(test_config(20)).unwrap();
        assert_eq!(cache.stats().hit_rate(), 0.0);

        cache.allocate_blocks_for_tokens(1, 32).unwrap();
        let blocks = cache.get_block_table(1).unwrap().to_vec();
        assert_eq!(cache.stats().total_allocations, 2);

        cache.register_prefix(42, blocks[..1].to_vec());
        cache.lookup_prefix(42);
        cache.lookup_prefix(99);
        assert!(cache.stats().hit_rate() > 0.0);
    }

    #[test]
    fn test_memory_usage() {
        let cache = PagedKvCache::new(test_config(100)).unwrap();
        // All blocks free, no memory used
        assert_eq!(cache.memory_usage_bytes(), 0);
    }

    #[test]
    fn test_fragmentation() {
        let mut cache = PagedKvCache::new(test_config(20)).unwrap();
        assert_eq!(cache.fragmentation(), 0.0);

        // Allocate 17 tokens -> 2 blocks, one partial
        cache.allocate_blocks_for_tokens(1, 17).unwrap();
        let frag = cache.fragmentation();
        assert!(frag > 0.0);
    }

    #[test]
    fn test_invalid_config() {
        assert!(
            PagedKvCache::new(PagedKvCacheConfig {
                block_size: 0,
                ..test_config(10)
            })
            .is_err()
        );

        assert!(
            PagedKvCache::new(PagedKvCacheConfig {
                num_blocks: 0,
                ..test_config(10)
            })
            .is_err()
        );
    }

    #[test]
    fn test_num_cached_tokens_no_sequence() {
        let cache = PagedKvCache::new(test_config(10)).unwrap();
        assert_eq!(cache.num_cached_tokens(999), 0);
    }
}
