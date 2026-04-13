//! Standalone paged KV-cache allocator for speculative decoding.
//!
//! [`KvManager`] manages a flat pool of fixed-size pages, assigning pages to
//! sequences on demand.  Pages are tracked in a per-sequence page table and
//! returned to the free pool when a sequence is freed or when a
//! [`KvCheckpoint`] is restored.
//!
//! This is a *host-side* data structure: it tracks which physical page indices
//! belong to which sequence, but does not hold GPU memory itself.

use std::collections::HashMap;

use crate::error::DnnError;

// ---------------------------------------------------------------------------
// KvManager
// ---------------------------------------------------------------------------

/// Paged KV-cache allocator.
///
/// Manages a pool of `total_pages` fixed-size pages (each holding
/// `page_size` tokens).  Sequences are identified by `u64` sequence IDs.
/// Pages are lazily allocated per sequence and eagerly returned to the free
/// pool when a sequence is freed or a checkpoint is restored.
///
/// # Example
/// ```no_run
/// use oxicuda_dnn::attn::speculative_decode::KvManager;
///
/// let mut mgr = KvManager::new(128, 16, 64);
/// let page = mgr.allocate_page(42).unwrap();
/// assert_eq!(mgr.free_page_count(), 127);
/// mgr.free_sequence(42);
/// assert_eq!(mgr.free_page_count(), 128);
/// ```
#[derive(Debug, Clone)]
pub struct KvManager {
    /// Total physical pages in the pool.
    total_pages: usize,
    /// Tokens per page.
    page_size: usize,
    /// Per-head dimension (informational; used for capacity queries).
    head_dim: usize,
    /// Stack of free physical page indices.
    free_pages: Vec<usize>,
    /// Per-sequence page table: seq_id → ordered list of physical page indices.
    page_tables: HashMap<u64, Vec<usize>>,
}

impl KvManager {
    /// Creates a new `KvManager` with `total_pages` free pages.
    ///
    /// * `total_pages` — physical pages available.
    /// * `page_size`   — tokens per page.
    /// * `head_dim`    — per-head dimension (used in capacity queries).
    #[must_use]
    pub fn new(total_pages: usize, page_size: usize, head_dim: usize) -> Self {
        // Free list is a stack; indices from 0..total_pages.
        let free_pages: Vec<usize> = (0..total_pages).rev().collect();
        Self {
            total_pages,
            page_size,
            head_dim,
            free_pages,
            page_tables: HashMap::new(),
        }
    }

    /// Allocates one free page and appends it to `seq_id`'s page table.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::WorkspaceRequired`] when no free pages remain.
    pub fn allocate_page(&mut self, seq_id: u64) -> Result<usize, DnnError> {
        let page = self.free_pages.pop().ok_or_else(|| {
            DnnError::WorkspaceRequired(self.page_size * self.head_dim * std::mem::size_of::<f32>())
        })?;
        self.page_tables.entry(seq_id).or_default().push(page);
        Ok(page)
    }

    /// Frees all pages allocated to `seq_id` and removes it from the table.
    ///
    /// If `seq_id` is unknown this is a no-op.
    pub fn free_sequence(&mut self, seq_id: u64) {
        if let Some(pages) = self.page_tables.remove(&seq_id) {
            self.free_pages.extend(pages);
        }
    }

    /// Returns the number of pages currently allocated to `seq_id`.
    #[must_use]
    pub fn page_count(&self, seq_id: u64) -> usize {
        self.page_tables.get(&seq_id).map_or(0, |v| v.len())
    }

    /// Returns the maximum token capacity of the pages allocated to `seq_id`.
    ///
    /// `capacity = page_count × page_size`
    #[must_use]
    pub fn token_capacity(&self, seq_id: u64) -> usize {
        self.page_count(seq_id) * self.page_size
    }

    /// Number of pages still in the free pool.
    #[must_use]
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }

    /// Total physical pages in the pool (constant after construction).
    #[must_use]
    pub fn total_pages(&self) -> usize {
        self.total_pages
    }

    /// Page size (tokens per page).
    #[must_use]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Per-head dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    // -----------------------------------------------------------------------
    // Checkpoint / restore
    // -----------------------------------------------------------------------

    /// Creates a [`KvCheckpoint`] capturing the current page table for
    /// `seq_id`.
    ///
    /// Returns `None` if `seq_id` has never been seen (no pages allocated).
    #[must_use]
    pub fn checkpoint(&self, seq_id: u64) -> Option<KvCheckpoint> {
        let pages = self.page_tables.get(&seq_id)?;
        Some(KvCheckpoint {
            seq_id,
            page_snapshot: pages.clone(),
            token_count: pages.len() * self.page_size,
        })
    }

    /// Restores the allocator to the state captured in `checkpoint`.
    ///
    /// Pages that were allocated *after* the checkpoint are returned to the
    /// free pool.  Pages that were present at checkpoint time are re-installed
    /// without any GPU-side side effects.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if the checkpoint's page snapshot
    /// is inconsistent (contains page indices outside the pool bounds).
    pub fn restore(&mut self, checkpoint: &KvCheckpoint) -> Result<(), DnnError> {
        // Validate snapshot indices.
        for &p in &checkpoint.page_snapshot {
            if p >= self.total_pages {
                return Err(DnnError::InvalidArgument(format!(
                    "checkpoint contains page {} which is out of pool bounds ({})",
                    p, self.total_pages
                )));
            }
        }

        let seq_id = checkpoint.seq_id;

        // Determine which pages were added after the checkpoint.
        let current: Vec<usize> = self.page_tables.get(&seq_id).cloned().unwrap_or_default();

        let snap_len = checkpoint.page_snapshot.len();
        if current.len() > snap_len {
            // Return excess pages to the free pool.
            let excess = &current[snap_len..];
            self.free_pages.extend_from_slice(excess);
        }

        // Restore snapshot (or remove entry if snapshot is empty).
        if checkpoint.page_snapshot.is_empty() {
            self.page_tables.remove(&seq_id);
        } else {
            self.page_tables
                .insert(seq_id, checkpoint.page_snapshot.clone());
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// KvCheckpoint
// ---------------------------------------------------------------------------

/// Snapshot of a single sequence's page allocation in a [`KvManager`].
///
/// Created by [`KvManager::checkpoint`]; applied via [`KvManager::restore`].
#[derive(Debug, Clone)]
pub struct KvCheckpoint {
    /// The sequence this checkpoint belongs to.
    pub seq_id: u64,
    /// Copy of the page table at checkpoint time.
    pub page_snapshot: Vec<usize>,
    /// Approximate token count at checkpoint time (`pages × page_size`).
    pub token_count: usize,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mgr(pages: usize) -> KvManager {
        KvManager::new(pages, 16, 64)
    }

    // -----------------------------------------------------------------------
    // Basic allocator operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_kvmanager_allocates_pages() {
        let mut mgr = make_mgr(4);
        let p0 = mgr.allocate_page(1).expect("allocate page 0");
        let p1 = mgr.allocate_page(1).expect("allocate page 1");
        assert_ne!(p0, p1, "each allocated page must be distinct");
        assert_eq!(mgr.page_count(1), 2);
        assert_eq!(mgr.free_page_count(), 2);
    }

    #[test]
    fn test_kvmanager_free_sequence_returns_pages() {
        let mut mgr = make_mgr(4);
        mgr.allocate_page(7).expect("alloc");
        mgr.allocate_page(7).expect("alloc");
        assert_eq!(mgr.free_page_count(), 2);
        mgr.free_sequence(7);
        assert_eq!(mgr.free_page_count(), 4);
        assert_eq!(mgr.page_count(7), 0);
    }

    #[test]
    fn test_kvmanager_out_of_pages_error() {
        let mut mgr = make_mgr(2);
        mgr.allocate_page(1).expect("page 0");
        mgr.allocate_page(1).expect("page 1");
        let err = mgr.allocate_page(1);
        assert!(err.is_err());
        assert!(matches!(err.unwrap_err(), DnnError::WorkspaceRequired(_)));
    }

    #[test]
    fn test_kvmanager_token_capacity() {
        let mut mgr = KvManager::new(8, 16, 64); // page_size = 16
        mgr.allocate_page(3).expect("p0");
        mgr.allocate_page(3).expect("p1");
        // 2 pages × 16 tokens/page = 32
        assert_eq!(mgr.token_capacity(3), 32);
    }

    #[test]
    fn test_kvmanager_free_unknown_seq_is_noop() {
        let mut mgr = make_mgr(4);
        // No allocation for seq 99.
        mgr.free_sequence(99); // must not panic or error
        assert_eq!(mgr.free_page_count(), 4);
    }

    // -----------------------------------------------------------------------
    // Checkpoint / restore
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_restore_roundtrip() {
        let mut mgr = make_mgr(8);
        mgr.allocate_page(5).expect("p0");
        let cp = mgr.checkpoint(5).expect("checkpoint should exist");
        assert_eq!(cp.seq_id, 5);
        assert_eq!(cp.page_snapshot.len(), 1);

        // Allocate more.
        mgr.allocate_page(5).expect("p1");
        mgr.allocate_page(5).expect("p2");
        assert_eq!(mgr.page_count(5), 3);

        // Restore: should drop back to 1 page, return 2 to free pool.
        mgr.restore(&cp).expect("restore");
        assert_eq!(mgr.page_count(5), 1);
        assert_eq!(mgr.free_page_count(), 7); // 8 total - 1 still in use
    }

    #[test]
    fn test_restore_frees_pages_allocated_after_checkpoint() {
        let mut mgr = make_mgr(4);
        // Before checkpoint: use up 2 pages for seq 10.
        mgr.allocate_page(10).expect("pre-cp page 0");
        mgr.allocate_page(10).expect("pre-cp page 1");
        let cp = mgr.checkpoint(10).expect("cp");

        // After checkpoint: 2 more pages.
        mgr.allocate_page(10).expect("post-cp page 0");
        mgr.allocate_page(10).expect("post-cp page 1");
        assert_eq!(mgr.free_page_count(), 0); // pool exhausted

        mgr.restore(&cp).expect("restore");
        // 2 pages restored to free pool.
        assert_eq!(mgr.free_page_count(), 2);
        assert_eq!(mgr.page_count(10), 2);
    }

    #[test]
    fn test_checkpoint_none_for_unknown_seq() {
        let mgr = make_mgr(4);
        assert!(mgr.checkpoint(999).is_none());
    }
}
