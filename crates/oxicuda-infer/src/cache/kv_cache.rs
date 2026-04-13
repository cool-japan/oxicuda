//! # Paged KV Cache
//!
//! Implements the PagedAttention memory model (Kwon et al., 2023):
//! key-value tensors are stored in fixed-size physical blocks rather than
//! contiguous per-sequence tensors.  A block table maps logical token
//! positions to physical block IDs for each sequence.
//!
//! ## Layout
//!
//! Each physical block holds `block_size` consecutive token slots for
//! **one transformer layer**:
//! ```text
//! Block[phys_id, layer]:
//!   keys   [block_size, n_kv_heads, head_dim]  f32
//!   values [block_size, n_kv_heads, head_dim]  f32
//! ```
//! The block pool is preallocated at construction time and reclaimed via a
//! free-list rather than dynamic allocation, making allocation O(1).

use crate::error::{InferError, InferResult};

// ─── BlockId ─────────────────────────────────────────────────────────────────

/// Opaque physical block identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

// ─── KvBlock ─────────────────────────────────────────────────────────────────

/// One physical KV block for a single transformer layer.
///
/// Stores keys and values as two contiguous arrays each of shape
/// `[block_size × n_kv_heads × head_dim]`.
#[derive(Debug, Clone)]
pub struct KvBlock {
    /// Key data: `[block_size, n_kv_heads, head_dim]` flattened.
    pub keys: Vec<f32>,
    /// Value data: `[block_size, n_kv_heads, head_dim]` flattened.
    pub values: Vec<f32>,
    /// Number of slots that have been written (0..=block_size).
    pub filled: usize,
    /// Tokens-per-block capacity (= block_size).
    pub capacity: usize,
}

impl KvBlock {
    /// Allocate an empty block for `block_size` token slots.
    #[must_use]
    pub fn new(block_size: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let elems = block_size * n_kv_heads * head_dim;
        Self {
            keys: vec![0.0_f32; elems],
            values: vec![0.0_f32; elems],
            filled: 0,
            capacity: block_size,
        }
    }

    /// Is this block completely full?
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.filled >= self.capacity
    }

    /// Remaining free slots.
    #[must_use]
    pub fn free_slots(&self) -> usize {
        self.capacity.saturating_sub(self.filled)
    }

    /// Append one token's key and value into the next free slot.
    ///
    /// `k` and `v` must each have exactly `n_kv_heads * head_dim` elements.
    /// Returns `false` if the block is already full.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> bool {
        if self.filled >= self.capacity {
            return false;
        }
        let stride = self.keys.len() / self.capacity;
        let base = self.filled * stride;
        self.keys[base..base + stride].copy_from_slice(k);
        self.values[base..base + stride].copy_from_slice(v);
        self.filled += 1;
        true
    }

    /// Read the key slice for slot `slot_idx`.
    ///
    /// # Panics
    /// Panics in debug mode if `slot_idx >= filled`.
    #[must_use]
    pub fn key_slice(&self, slot_idx: usize) -> &[f32] {
        let stride = self.keys.len() / self.capacity;
        let base = slot_idx * stride;
        &self.keys[base..base + stride]
    }

    /// Read the value slice for slot `slot_idx`.
    #[must_use]
    pub fn value_slice(&self, slot_idx: usize) -> &[f32] {
        let stride = self.values.len() / self.capacity;
        let base = slot_idx * stride;
        &self.values[base..base + stride]
    }

    /// Reset the block to empty (overwrite data with zeros).
    pub fn reset(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.filled = 0;
    }
}

// ─── PagedKvCache ─────────────────────────────────────────────────────────────

/// The global paged KV cache managing all physical blocks across all layers.
///
/// Each physical block exists independently for each layer:
/// `blocks[layer][block_id]`.  Block allocation is O(1) via a free list.
/// Reference counts support copy-on-write sharing for prefix caching.
pub struct PagedKvCache {
    /// Number of transformer layers.
    pub n_layers: usize,
    /// KV head count.
    pub n_kv_heads: usize,
    /// Attention head dimension.
    pub head_dim: usize,
    /// Tokens per block.
    pub block_size: usize,
    /// Total number of physical blocks preallocated.
    pub n_blocks: usize,
    /// Block storage: `[layer][block_id]`.
    blocks: Vec<Vec<KvBlock>>,
    /// Free block list (IDs available for allocation).
    free_list: Vec<BlockId>,
    /// Reference counts per block (shared prefix caching).
    ref_counts: Vec<u32>,
}

impl PagedKvCache {
    /// Preallocate a KV cache with `n_blocks` physical blocks.
    ///
    /// Memory consumption: `2 × n_blocks × n_layers × block_size × n_kv_heads × head_dim × 4` bytes.
    #[must_use]
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
    ) -> Self {
        let blocks: Vec<Vec<KvBlock>> = (0..n_layers)
            .map(|_| {
                (0..n_blocks)
                    .map(|_| KvBlock::new(block_size, n_kv_heads, head_dim))
                    .collect()
            })
            .collect();
        let free_list: Vec<BlockId> = (0..n_blocks as u32).rev().map(BlockId).collect();
        let ref_counts = vec![0_u32; n_blocks];
        Self {
            n_layers,
            n_kv_heads,
            head_dim,
            block_size,
            n_blocks,
            blocks,
            free_list,
            ref_counts,
        }
    }

    /// Number of currently free blocks.
    #[must_use]
    pub fn n_free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Number of currently allocated blocks.
    #[must_use]
    pub fn n_used_blocks(&self) -> usize {
        self.n_blocks - self.free_list.len()
    }

    /// Allocate one physical block.
    ///
    /// Returns `Err(BlockAllocFailed)` if the cache is exhausted.
    pub fn alloc_block(&mut self) -> InferResult<BlockId> {
        self.free_list
            .pop()
            .inspect(|id| {
                self.ref_counts[id.0 as usize] = 1;
            })
            .ok_or(InferError::BlockAllocFailed {
                needed: 1,
                available: 0,
            })
    }

    /// Allocate `n` physical blocks at once.
    ///
    /// Fails atomically if fewer than `n` blocks are available.
    pub fn alloc_blocks(&mut self, n: usize) -> InferResult<Vec<BlockId>> {
        if self.free_list.len() < n {
            return Err(InferError::BlockAllocFailed {
                needed: n,
                available: self.free_list.len(),
            });
        }
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.free_list.pop().expect("checked above");
            self.ref_counts[id.0 as usize] = 1;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Increment the reference count for a block (prefix sharing).
    pub fn inc_ref(&mut self, id: BlockId) {
        self.ref_counts[id.0 as usize] += 1;
    }

    /// Decrement the reference count; return the block to the free list when
    /// it reaches zero.
    pub fn dec_ref(&mut self, id: BlockId) {
        let rc = &mut self.ref_counts[id.0 as usize];
        if *rc > 0 {
            *rc -= 1;
        }
        if *rc == 0 {
            // Reset block data and return to free list.
            for layer in &mut self.blocks {
                layer[id.0 as usize].reset();
            }
            self.free_list.push(id);
        }
    }

    /// Append one token's K/V to block `id`, layer `layer`.
    ///
    /// `k` and `v` must each be `n_kv_heads × head_dim` elements.
    pub fn append_token(
        &mut self,
        id: BlockId,
        layer: usize,
        k: &[f32],
        v: &[f32],
    ) -> InferResult<()> {
        if layer >= self.n_layers {
            return Err(InferError::DimensionMismatch {
                expected: self.n_layers,
                got: layer + 1,
            });
        }
        let block = &mut self.blocks[layer][id.0 as usize];
        if !block.append(k, v) {
            return Err(InferError::BlockAllocFailed {
                needed: 1,
                available: 0,
            });
        }
        Ok(())
    }

    /// Access a block (immutable) for a given layer.
    pub fn block(&self, id: BlockId, layer: usize) -> InferResult<&KvBlock> {
        if layer >= self.n_layers {
            return Err(InferError::DimensionMismatch {
                expected: self.n_layers,
                got: layer + 1,
            });
        }
        Ok(&self.blocks[layer][id.0 as usize])
    }

    /// Number of filled slots in block `id` (same across layers after sync).
    ///
    /// Returns the filled count from layer 0 as a representative value.
    pub fn block_filled(&self, id: BlockId) -> usize {
        if self.n_layers == 0 {
            return 0;
        }
        self.blocks[0][id.0 as usize].filled
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cache() -> PagedKvCache {
        // 2 layers, 2 kv-heads, head_dim=4, block_size=4, 8 blocks
        PagedKvCache::new(2, 2, 4, 4, 8)
    }

    #[test]
    fn alloc_and_free() {
        let mut cache = tiny_cache();
        assert_eq!(cache.n_free_blocks(), 8);
        let id = cache.alloc_block().unwrap();
        assert_eq!(cache.n_free_blocks(), 7);
        cache.dec_ref(id);
        assert_eq!(cache.n_free_blocks(), 8);
    }

    #[test]
    fn alloc_exhaustion_error() {
        let mut cache = PagedKvCache::new(1, 1, 4, 4, 2);
        cache.alloc_block().unwrap();
        cache.alloc_block().unwrap();
        let res = cache.alloc_block();
        assert!(matches!(res, Err(InferError::BlockAllocFailed { .. })));
    }

    #[test]
    fn alloc_blocks_batch() {
        let mut cache = tiny_cache();
        let ids = cache.alloc_blocks(3).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(cache.n_free_blocks(), 5);
    }

    #[test]
    fn alloc_blocks_too_many_fails() {
        let mut cache = PagedKvCache::new(1, 1, 4, 4, 2);
        assert!(cache.alloc_blocks(3).is_err());
    }

    #[test]
    fn append_and_read_token() {
        let mut cache = tiny_cache();
        let id = cache.alloc_block().unwrap();
        let k = vec![1.0_f32; 2 * 4]; // n_kv_heads=2, head_dim=4
        let v = vec![2.0_f32; 2 * 4];
        cache.append_token(id, 0, &k, &v).unwrap();
        let blk = cache.block(id, 0).unwrap();
        assert_eq!(blk.filled, 1);
        assert!(blk.key_slice(0).iter().all(|&x| (x - 1.0).abs() < 1e-7));
        assert!(blk.value_slice(0).iter().all(|&x| (x - 2.0).abs() < 1e-7));
    }

    #[test]
    fn block_full_error() {
        let mut cache = PagedKvCache::new(1, 2, 4, 2, 4); // block_size=2
        let id = cache.alloc_block().unwrap();
        let kv = vec![0.5_f32; 2 * 4];
        cache.append_token(id, 0, &kv, &kv).unwrap();
        cache.append_token(id, 0, &kv, &kv).unwrap();
        // Block now full
        assert!(cache.append_token(id, 0, &kv, &kv).is_err());
    }

    #[test]
    fn ref_count_sharing() {
        let mut cache = tiny_cache();
        let id = cache.alloc_block().unwrap();
        // Block should be allocated (refcount = 1)
        assert_eq!(cache.n_used_blocks(), 1);
        cache.inc_ref(id);
        // refcount = 2; one dec should NOT free it
        cache.dec_ref(id);
        assert_eq!(cache.n_free_blocks(), 7);
        // second dec should free it
        cache.dec_ref(id);
        assert_eq!(cache.n_free_blocks(), 8);
    }

    #[test]
    fn block_reset_on_free() {
        let mut cache = tiny_cache();
        let id = cache.alloc_block().unwrap();
        let kv = vec![9.0_f32; 2 * 4];
        cache.append_token(id, 0, &kv, &kv).unwrap();
        // Free and reallocate
        cache.dec_ref(id);
        let id2 = cache.alloc_block().unwrap();
        let blk = cache.block(id2, 0).unwrap();
        assert_eq!(blk.filled, 0);
        assert!(blk.keys.iter().all(|&x| x == 0.0));
    }
}
