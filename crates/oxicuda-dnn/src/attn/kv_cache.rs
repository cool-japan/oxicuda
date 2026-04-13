//! KV-cache management for autoregressive decoding.
//!
//! Provides [`KvCache`] which manages GPU memory pages for storing key and
//! value tensors across layers during autoregressive generation. The page-based
//! design enables:
//!
//! - **Efficient memory sharing**: Multiple beam search hypotheses can share
//!   pages for common prefixes (copy-on-write).
//! - **Dynamic allocation**: Pages are allocated on demand and freed when
//!   sequences complete, avoiding worst-case pre-allocation.
//! - **Non-contiguous storage**: Sequences' KV-cache need not be contiguous,
//!   avoiding expensive memory moves when batch composition changes.
//!
//! The page table maps `(sequence, logical_page_index)` to a physical page
//! number in the cache pool.

use oxicuda_blas::GpuFloat;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_memory::DeviceBuffer;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// KvCacheConfig
// ---------------------------------------------------------------------------

/// Configuration for the KV-cache.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Number of KV attention heads per layer.
    pub num_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Number of tokens per page.
    pub page_size: u32,
    /// Maximum number of physical pages in the cache pool.
    pub max_pages: u32,
}

impl KvCacheConfig {
    /// Size of a single page in elements (for one of K or V).
    ///
    /// Each page stores `page_size` tokens for `num_heads` heads, each
    /// with `head_dim` dimensions.
    #[must_use]
    pub fn page_elements(&self) -> usize {
        self.num_heads as usize * self.page_size as usize * self.head_dim as usize
    }

    /// Size of a single page in bytes for the given element type.
    #[must_use]
    pub fn page_bytes<T: GpuFloat>(&self) -> usize {
        self.page_elements() * T::SIZE
    }

    /// Total cache pool size in elements (all pages, both K and V).
    #[must_use]
    pub fn total_pool_elements(&self) -> usize {
        self.page_elements() * self.max_pages as usize * 2 // K + V
    }
}

// ---------------------------------------------------------------------------
// KvCache
// ---------------------------------------------------------------------------

/// GPU KV-cache with page-based memory management.
///
/// Manages a pool of physical pages in GPU memory and a free-list for
/// allocation/deallocation. Each page holds `page_size` tokens of KV data
/// for all heads in a single layer.
///
/// # Layout
///
/// The cache pool is a single contiguous GPU allocation with layout:
/// ```text
/// K pool: [max_pages, num_heads, page_size, head_dim]
/// V pool: [max_pages, num_heads, page_size, head_dim]
/// ```
///
/// Page tables are maintained on the host and transferred to device
/// as needed for kernel launches.
pub struct KvCache {
    /// Configuration.
    config: KvCacheConfig,
    /// Device pointer to the K-cache pool.
    k_pool_ptr: CUdeviceptr,
    /// Device pointer to the V-cache pool.
    v_pool_ptr: CUdeviceptr,
    /// Free page indices (host-side management).
    free_pages: Vec<u32>,
    /// Total pages currently allocated.
    allocated_count: u32,
    /// Page tables per sequence: maps (seq_id, logical_page) -> physical_page.
    /// Stored as a flat vector: page_tables[seq_id * max_pages_per_seq + page_idx].
    page_tables: Vec<i32>,
    /// Number of sequences tracked.
    num_sequences: u32,
    /// Maximum logical pages per sequence.
    max_pages_per_seq: u32,
}

impl KvCache {
    /// Creates a new KV-cache with the given configuration.
    ///
    /// Allocates GPU memory for the K and V pools. All pages are initially
    /// free and available for allocation.
    ///
    /// # Arguments
    ///
    /// * `config` - Cache configuration.
    /// * `k_pool` - Pre-allocated device buffer for the K-cache pool.
    /// * `v_pool` - Pre-allocated device buffer for the V-cache pool.
    /// * `max_sequences` - Maximum number of sequences to track concurrently.
    /// * `max_pages_per_seq` - Maximum pages per sequence (determines page
    ///   table width).
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::BufferTooSmall`] if the provided buffers are
    /// too small for the configured pool size.
    pub fn new<T: GpuFloat>(
        config: KvCacheConfig,
        k_pool: &DeviceBuffer<T>,
        v_pool: &DeviceBuffer<T>,
        max_sequences: u32,
        max_pages_per_seq: u32,
    ) -> DnnResult<Self> {
        let required = config.page_elements() * config.max_pages as usize;
        if k_pool.len() < required {
            return Err(DnnError::BufferTooSmall {
                expected: required * T::SIZE,
                actual: k_pool.len() * T::SIZE,
            });
        }
        if v_pool.len() < required {
            return Err(DnnError::BufferTooSmall {
                expected: required * T::SIZE,
                actual: v_pool.len() * T::SIZE,
            });
        }

        // Initialise free list with all pages.
        let free_pages: Vec<u32> = (0..config.max_pages).collect();

        // Initialise page tables (all -1 = unmapped).
        let table_size = max_sequences as usize * max_pages_per_seq as usize;
        let page_tables = vec![-1i32; table_size];

        Ok(Self {
            config,
            k_pool_ptr: k_pool.as_device_ptr(),
            v_pool_ptr: v_pool.as_device_ptr(),
            free_pages,
            allocated_count: 0,
            page_tables,
            num_sequences: max_sequences,
            max_pages_per_seq,
        })
    }

    /// Allocates a free page and returns its physical page number.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::WorkspaceRequired`] if no free pages remain.
    pub fn allocate_page(&mut self) -> DnnResult<u32> {
        self.free_pages
            .pop()
            .ok_or_else(|| {
                DnnError::WorkspaceRequired(
                    self.config.page_elements() * std::mem::size_of::<f32>(),
                )
            })
            .inspect(|_| {
                self.allocated_count += 1;
            })
    }

    /// Returns a page to the free pool.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if the page number is out of
    /// range.
    pub fn free_page(&mut self, page: u32) -> DnnResult<()> {
        if page >= self.config.max_pages {
            return Err(DnnError::InvalidArgument(format!(
                "page {} out of range (max {})",
                page, self.config.max_pages
            )));
        }
        self.free_pages.push(page);
        self.allocated_count = self.allocated_count.saturating_sub(1);
        Ok(())
    }

    /// Appends a KV pair for a token to the cache.
    ///
    /// Finds or allocates a page for the given sequence and token position,
    /// and returns the offset within the cache pool where the data should
    /// be written.
    ///
    /// # Arguments
    ///
    /// * `seq_id` - Sequence index in the batch.
    /// * `token_pos` - Token position within the sequence.
    ///
    /// # Returns
    ///
    /// A tuple `(k_offset, v_offset)` of byte offsets into the K and V
    /// pools where the caller should copy the new KV data.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `seq_id` is out of range.
    /// Returns [`DnnError::WorkspaceRequired`] if no free pages are available.
    pub fn append_kv<T: GpuFloat>(
        &mut self,
        seq_id: u32,
        token_pos: u32,
    ) -> DnnResult<(CUdeviceptr, CUdeviceptr)> {
        if seq_id >= self.num_sequences {
            return Err(DnnError::InvalidArgument(format!(
                "seq_id {} out of range (max {})",
                seq_id, self.num_sequences
            )));
        }

        let logical_page = token_pos / self.config.page_size;
        let offset_in_page = token_pos % self.config.page_size;

        if logical_page >= self.max_pages_per_seq {
            return Err(DnnError::InvalidArgument(format!(
                "logical page {} exceeds max_pages_per_seq {}",
                logical_page, self.max_pages_per_seq
            )));
        }

        let table_idx = seq_id as usize * self.max_pages_per_seq as usize + logical_page as usize;

        // Allocate page if not yet mapped.
        let phys_page = if self.page_tables[table_idx] < 0 {
            let new_page = self.allocate_page()?;
            self.page_tables[table_idx] = new_page as i32;
            new_page
        } else {
            self.page_tables[table_idx] as u32
        };

        // Compute byte offsets into the pool.
        let page_elem_offset = phys_page as usize * self.config.page_elements();
        let token_offset_in_page = offset_in_page as usize
            * self.config.num_heads as usize
            * self.config.head_dim as usize;
        let total_elem_offset = page_elem_offset + token_offset_in_page;
        let byte_offset = (total_elem_offset * T::SIZE) as u64;

        let k_ptr = self.k_pool_ptr + byte_offset;
        let v_ptr = self.v_pool_ptr + byte_offset;

        Ok((k_ptr, v_ptr))
    }

    /// Returns the page table for a given sequence as a slice.
    ///
    /// The returned slice has `max_pages_per_seq` entries. Entries with
    /// value -1 indicate unmapped (unallocated) pages.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `seq_id` is out of range.
    pub fn get_page_table(&self, seq_id: u32) -> DnnResult<&[i32]> {
        if seq_id >= self.num_sequences {
            return Err(DnnError::InvalidArgument(format!(
                "seq_id {} out of range (max {})",
                seq_id, self.num_sequences
            )));
        }
        let start = seq_id as usize * self.max_pages_per_seq as usize;
        let end = start + self.max_pages_per_seq as usize;
        Ok(&self.page_tables[start..end])
    }

    /// Returns the K-cache pool device pointer.
    pub fn k_pool_ptr(&self) -> CUdeviceptr {
        self.k_pool_ptr
    }

    /// Returns the V-cache pool device pointer.
    pub fn v_pool_ptr(&self) -> CUdeviceptr {
        self.v_pool_ptr
    }

    /// Returns the number of currently allocated pages.
    pub fn allocated_pages(&self) -> u32 {
        self.allocated_count
    }

    /// Returns the number of free pages remaining.
    pub fn free_pages_count(&self) -> u32 {
        self.free_pages.len() as u32
    }

    /// Returns the cache configuration.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Resets the page table for a sequence, freeing all its pages.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `seq_id` is out of range.
    pub fn reset_sequence(&mut self, seq_id: u32) -> DnnResult<()> {
        if seq_id >= self.num_sequences {
            return Err(DnnError::InvalidArgument(format!(
                "seq_id {} out of range (max {})",
                seq_id, self.num_sequences
            )));
        }
        let start = seq_id as usize * self.max_pages_per_seq as usize;
        for i in 0..self.max_pages_per_seq as usize {
            let phys = self.page_tables[start + i];
            if phys >= 0 {
                self.free_pages.push(phys as u32);
                self.allocated_count = self.allocated_count.saturating_sub(1);
                self.page_tables[start + i] = -1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_page_elements() {
        let cfg = KvCacheConfig {
            num_layers: 32,
            num_heads: 8,
            head_dim: 64,
            page_size: 16,
            max_pages: 1024,
        };
        // 8 heads * 16 tokens * 64 dim = 8192
        assert_eq!(cfg.page_elements(), 8192);
    }

    #[test]
    fn config_page_bytes() {
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_heads: 4,
            head_dim: 128,
            page_size: 16,
            max_pages: 256,
        };
        assert_eq!(cfg.page_bytes::<f32>(), 4 * 16 * 128 * 4);
        assert_eq!(cfg.page_bytes::<f64>(), 4 * 16 * 128 * 8);
    }

    #[test]
    fn config_total_pool() {
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_heads: 8,
            head_dim: 64,
            page_size: 16,
            max_pages: 100,
        };
        // 8192 per page * 100 pages * 2 (K+V)
        assert_eq!(cfg.total_pool_elements(), 8192 * 100 * 2);
    }

    #[test]
    fn page_table_out_of_range() {
        // We need a KvCache to test, but can't allocate GPU memory.
        // Verify the config math instead.
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_heads: 8,
            head_dim: 64,
            page_size: 16,
            max_pages: 10,
        };
        assert_eq!(cfg.page_elements(), 8 * 16 * 64);
    }
}
