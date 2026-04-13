//! # Cache Manager
//!
//! Manages the per-sequence block tables that map logical token positions
//! to physical blocks in the [`PagedKvCache`].
//!
//! Each sequence owns a list of physical [`BlockId`]s; the last block in
//! the list is the "current" block receiving new tokens.  When the current
//! block fills up, a new block is allocated and appended to the table.
//!
//! ## Responsibilities
//!
//! * Allocate / free block tables on sequence admission / completion.
//! * Append K/V for a new token to the correct block+slot for every layer.
//! * Provide the block table (as `&[BlockId]`) to the attention kernel.
//! * Track memory pressure so the scheduler can trigger preemption.

use std::collections::HashMap;

use crate::cache::kv_cache::{BlockId, PagedKvCache};
use crate::error::{InferError, InferResult};

// ─── CacheManager ─────────────────────────────────────────────────────────────

/// Manages per-sequence block tables over a shared [`PagedKvCache`].
pub struct CacheManager {
    kv_cache: PagedKvCache,
    /// sequence_id → ordered list of physical blocks.
    block_tables: HashMap<u64, Vec<BlockId>>,
    /// sequence_id → number of tokens currently stored.
    seq_lengths: HashMap<u64, usize>,
}

impl CacheManager {
    /// Create a cache manager wrapping the given paged KV cache.
    #[must_use]
    pub fn new(kv_cache: PagedKvCache) -> Self {
        Self {
            kv_cache,
            block_tables: HashMap::new(),
            seq_lengths: HashMap::new(),
        }
    }

    /// Allocate a block table for a new sequence.
    ///
    /// Pre-allocates enough blocks to hold `initial_tokens`.
    /// Returns an error if the KV cache is exhausted.
    pub fn allocate_sequence(&mut self, seq_id: u64, initial_tokens: usize) -> InferResult<()> {
        if self.block_tables.contains_key(&seq_id) {
            return Err(InferError::CacheManagerError(format!(
                "sequence {seq_id} already has a block table"
            )));
        }
        let n_blocks_needed = initial_tokens.div_ceil(self.kv_cache.block_size);
        let n_blocks_needed = n_blocks_needed.max(1);
        let ids = self.kv_cache.alloc_blocks(n_blocks_needed)?;
        self.block_tables.insert(seq_id, ids);
        self.seq_lengths.insert(seq_id, 0);
        Ok(())
    }

    /// Free all KV cache blocks belonging to a sequence.
    pub fn free_sequence(&mut self, seq_id: u64) {
        if let Some(blocks) = self.block_tables.remove(&seq_id) {
            for id in blocks {
                self.kv_cache.dec_ref(id);
            }
        }
        self.seq_lengths.remove(&seq_id);
    }

    /// Append one new token's K/V (for **every** layer) to the sequence's cache.
    ///
    /// Automatically allocates a new block if the current tail block is full.
    pub fn append_token(
        &mut self,
        seq_id: u64,
        kvs: &[(Vec<f32>, Vec<f32>)], // kvs[layer] = (k_vec, v_vec)
    ) -> InferResult<()> {
        // Ensure the sequence has a current block with free space.
        self.ensure_space(seq_id)?;

        let table = self
            .block_tables
            .get(&seq_id)
            .ok_or(InferError::InvalidSequenceId(seq_id))?;
        let tail_id = *table.last().expect("table guaranteed non-empty");
        let slot = self.kv_cache.block_filled(tail_id);

        if kvs.len() != self.kv_cache.n_layers {
            return Err(InferError::DimensionMismatch {
                expected: self.kv_cache.n_layers,
                got: kvs.len(),
            });
        }
        for (layer, (k, v)) in kvs.iter().enumerate() {
            self.kv_cache.append_token(tail_id, layer, k, v)?;
            // Only the first layer controls `filled`; other layers follow.
            // (In the real impl every layer has an identical block-layout;
            //  we advance the "tail" pointer via layer-0.)
            let _ = slot;
        }
        *self.seq_lengths.get_mut(&seq_id).expect("seq present") += 1;
        Ok(())
    }

    /// Return an immutable view of the block table for `seq_id`.
    pub fn block_table(&self, seq_id: u64) -> InferResult<&[BlockId]> {
        self.block_tables
            .get(&seq_id)
            .map(Vec::as_slice)
            .ok_or(InferError::InvalidSequenceId(seq_id))
    }

    /// Number of tokens stored for `seq_id`.
    pub fn seq_length(&self, seq_id: u64) -> InferResult<usize> {
        self.seq_lengths
            .get(&seq_id)
            .copied()
            .ok_or(InferError::InvalidSequenceId(seq_id))
    }

    /// Free blocks available in the underlying cache.
    #[must_use]
    pub fn n_free_blocks(&self) -> usize {
        self.kv_cache.n_free_blocks()
    }

    /// Check whether at least `n` more blocks can be allocated.
    #[must_use]
    pub fn can_allocate(&self, n: usize) -> bool {
        self.kv_cache.n_free_blocks() >= n
    }

    /// Number of active (allocated) sequences.
    #[must_use]
    pub fn n_active_sequences(&self) -> usize {
        self.block_tables.len()
    }

    // Ensure the tail block of `seq_id` has at least one free slot, allocating
    // a new block if necessary.
    fn ensure_space(&mut self, seq_id: u64) -> InferResult<()> {
        let table = self
            .block_tables
            .get_mut(&seq_id)
            .ok_or(InferError::InvalidSequenceId(seq_id))?;
        let tail_id = *table
            .last()
            .expect("table guaranteed non-empty after allocate");
        if self.kv_cache.block_filled(tail_id) < self.kv_cache.block_size {
            return Ok(()); // Current tail block still has room.
        }
        // Current block is full — allocate a new one.
        let new_id = self.kv_cache.alloc_block()?;
        self.block_tables
            .get_mut(&seq_id)
            .expect("still present")
            .push(new_id);
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::kv_cache::PagedKvCache;

    fn make_manager(n_blocks: usize) -> CacheManager {
        // 2 layers, 2 kv-heads, head_dim=4, block_size=4
        CacheManager::new(PagedKvCache::new(2, 2, 4, 4, n_blocks))
    }

    fn zero_kvs(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        (0..n_layers)
            .map(|_| {
                (
                    vec![0.0_f32; n_kv_heads * head_dim],
                    vec![0.0_f32; n_kv_heads * head_dim],
                )
            })
            .collect()
    }

    #[test]
    fn allocate_and_free() {
        let mut m = make_manager(8);
        m.allocate_sequence(1, 1).unwrap();
        assert_eq!(m.n_active_sequences(), 1);
        m.free_sequence(1);
        assert_eq!(m.n_active_sequences(), 0);
    }

    #[test]
    fn duplicate_allocate_error() {
        let mut m = make_manager(8);
        m.allocate_sequence(1, 1).unwrap();
        assert!(m.allocate_sequence(1, 1).is_err());
    }

    #[test]
    fn append_increments_length() {
        let mut m = make_manager(8);
        m.allocate_sequence(42, 4).unwrap();
        let kvs = zero_kvs(2, 2, 4);
        m.append_token(42, &kvs).unwrap();
        assert_eq!(m.seq_length(42).unwrap(), 1);
        m.append_token(42, &kvs).unwrap();
        assert_eq!(m.seq_length(42).unwrap(), 2);
    }

    #[test]
    fn block_overflow_triggers_new_block() {
        let mut m = make_manager(16);
        m.allocate_sequence(7, 4).unwrap(); // block_size=4, pre-alloc 1 block
        let kvs = zero_kvs(2, 2, 4);
        let free_before = m.n_free_blocks();
        // Fill the first block (4 tokens)
        for _ in 0..4 {
            m.append_token(7, &kvs).unwrap();
        }
        // 5th token should trigger a new block alloc
        m.append_token(7, &kvs).unwrap();
        assert!(
            m.n_free_blocks() < free_before,
            "new block should be allocated"
        );
        assert_eq!(m.seq_length(7).unwrap(), 5);
    }

    #[test]
    fn block_table_length_matches() {
        let mut m = make_manager(16);
        m.allocate_sequence(3, 4).unwrap();
        let kvs = zero_kvs(2, 2, 4);
        for _ in 0..4 {
            m.append_token(3, &kvs).unwrap();
        }
        assert!(!m.block_table(3).unwrap().is_empty());
    }

    #[test]
    fn invalid_seq_id_error() {
        let m = make_manager(8);
        assert!(matches!(
            m.seq_length(99),
            Err(InferError::InvalidSequenceId(99))
        ));
        assert!(matches!(
            m.block_table(99),
            Err(InferError::InvalidSequenceId(99))
        ));
    }

    #[test]
    fn free_returns_blocks_to_pool() {
        let mut m = make_manager(8);
        let free_start = m.n_free_blocks();
        m.allocate_sequence(5, 4).unwrap();
        let free_mid = m.n_free_blocks();
        assert!(free_mid < free_start);
        m.free_sequence(5);
        assert_eq!(m.n_free_blocks(), free_start);
    }
}
