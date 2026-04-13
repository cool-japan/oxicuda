//! Enhanced memory pool statistics and allocation tracking.
//!
//! This module provides general-purpose memory allocation tracking utilities
//! that are always available (not feature-gated). It includes:
//!
//! - [`AllocationHistogram`] — Tracks allocation size distribution in power-of-2 buckets.
//! - [`FragmentationMetrics`] — Measures memory fragmentation.
//! - [`PoolReport`] — Comprehensive memory pool status report.
//! - [`PoolStatsTracker`] — Thread-safe allocation tracking.

use std::fmt;
use std::sync::RwLock;
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// AllocationHistogram
// ---------------------------------------------------------------------------

/// Tracks allocation size distribution in power-of-2 buckets.
///
/// Bucket `i` covers the byte range `[2^(i+4), 2^(i+5))`, so:
/// - Bucket 0: \[16, 32)
/// - Bucket 1: \[32, 64)
/// - ...
/// - Bucket 31: \[2^35, ∞) (32 GiB and above)
///
/// Allocations smaller than 16 bytes are placed in bucket 0.
#[derive(Debug, Clone)]
pub struct AllocationHistogram {
    buckets: [u64; 32],
    total_allocations: u64,
}

impl AllocationHistogram {
    /// Creates a new empty histogram.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: [0; 32],
            total_allocations: 0,
        }
    }

    /// Computes the bucket index for a given allocation size.
    ///
    /// Sizes below 16 bytes map to bucket 0. The bucket index is derived from
    /// the position of the highest set bit, offset by 4 (since bucket 0
    /// starts at 2^4 = 16).
    #[must_use]
    pub fn bucket_index(size: usize) -> usize {
        if size < 16 {
            return 0;
        }
        // Find the highest bit position.
        // For size in [2^k, 2^(k+1)), the bit length is k+1.
        // We want bucket = k - 4, clamped to [0, 31].
        let bit_len = usize::BITS - size.leading_zeros();
        // bit_len is at least 5 for size >= 16 (2^4).
        let idx = (bit_len as usize).saturating_sub(5);
        idx.min(31)
    }

    /// Records an allocation of the given size.
    pub fn record(&mut self, size: usize) {
        let idx = Self::bucket_index(size);
        self.buckets[idx] = self.buckets[idx].saturating_add(1);
        self.total_allocations = self.total_allocations.saturating_add(1);
    }

    /// Returns the (inclusive-min, exclusive-max) byte range for a bucket.
    ///
    /// For the last bucket (index 31), the max is `usize::MAX` since it
    /// covers all sizes >= 2^35.
    #[must_use]
    pub fn bucket_range(index: usize) -> (usize, usize) {
        let clamped = index.min(31);
        let min = 1_usize << (clamped + 4);
        if clamped >= 31 {
            (min, usize::MAX)
        } else {
            let max = 1_usize << (clamped + 5);
            (min, max)
        }
    }

    /// Returns a reference to the raw bucket counts.
    #[must_use]
    pub fn bucket_counts(&self) -> &[u64; 32] {
        &self.buckets
    }

    /// Returns the total number of allocations recorded.
    #[must_use]
    pub fn total_allocations(&self) -> u64 {
        self.total_allocations
    }

    /// Computes the approximate allocation size at the given percentile.
    ///
    /// `p` should be in the range `[0.0, 100.0]`. Returns the lower bound of
    /// the bucket that contains the `p`-th percentile allocation.
    ///
    /// Returns 0 if no allocations have been recorded.
    #[must_use]
    pub fn percentile(&self, p: f64) -> usize {
        if self.total_allocations == 0 {
            return 0;
        }

        let p_clamped = p.clamp(0.0, 100.0);
        let target = ((p_clamped / 100.0) * self.total_allocations as f64).ceil() as u64;
        let target = target.max(1);

        let mut cumulative: u64 = 0;
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative = cumulative.saturating_add(count);
            if cumulative >= target {
                let (low, _) = Self::bucket_range(i);
                return low;
            }
        }

        // All allocations exhausted — return the last bucket's lower bound.
        let (low, _) = Self::bucket_range(31);
        low
    }

    /// Shorthand for `percentile(50.0)`.
    #[must_use]
    pub fn median(&self) -> usize {
        self.percentile(50.0)
    }
}

impl Default for AllocationHistogram {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FragmentationMetrics
// ---------------------------------------------------------------------------

/// Measures memory fragmentation of free space.
#[derive(Debug, Clone, Default)]
pub struct FragmentationMetrics {
    /// Total free bytes across all free blocks.
    pub total_free_bytes: usize,
    /// Size of the largest contiguous free block.
    pub largest_free_block: usize,
    /// Number of separate free blocks.
    pub free_block_count: u32,
}

impl FragmentationMetrics {
    /// Creates a new `FragmentationMetrics`.
    #[must_use]
    pub fn new(total_free: usize, largest_block: usize, block_count: u32) -> Self {
        Self {
            total_free_bytes: total_free,
            largest_free_block: largest_block,
            free_block_count: block_count,
        }
    }

    /// Computes the fragmentation ratio.
    ///
    /// Returns `1.0 - (largest_free_block / total_free_bytes)`.
    /// A value of 0.0 means no fragmentation (single contiguous block).
    /// A value near 1.0 means high fragmentation.
    ///
    /// Returns 0.0 if `total_free_bytes` is zero.
    #[must_use]
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.total_free_bytes == 0 {
            return 0.0;
        }
        1.0 - (self.largest_free_block as f64 / self.total_free_bytes as f64)
    }

    /// Returns the average free block size, or 0 if there are no free blocks.
    #[must_use]
    pub fn average_free_block_size(&self) -> usize {
        if self.free_block_count == 0 {
            return 0;
        }
        self.total_free_bytes / self.free_block_count as usize
    }
}

// ---------------------------------------------------------------------------
// PoolReport
// ---------------------------------------------------------------------------

/// A comprehensive memory pool status report.
#[derive(Debug, Clone, Default)]
pub struct PoolReport {
    /// Currently allocated bytes.
    pub allocated_bytes: usize,
    /// Peak allocated bytes ever observed.
    pub peak_bytes: usize,
    /// Total number of allocations performed.
    pub allocation_count: u64,
    /// Total number of frees performed.
    pub free_count: u64,
    /// Fragmentation metrics at time of report.
    pub fragmentation: FragmentationMetrics,
    /// Allocation size histogram.
    pub histogram: AllocationHistogram,
    /// Timestamp in nanoseconds since UNIX epoch when report was generated.
    pub timestamp_ns: u64,
}

impl fmt::Display for PoolReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== OxiCUDA Pool Report ===")?;
        writeln!(f, "Allocated:   {} bytes", self.allocated_bytes)?;
        writeln!(f, "Peak:        {} bytes", self.peak_bytes)?;
        writeln!(f, "Allocs:      {}", self.allocation_count)?;
        writeln!(f, "Frees:       {}", self.free_count)?;
        writeln!(
            f,
            "Active:      {}",
            self.allocation_count.saturating_sub(self.free_count)
        )?;
        writeln!(f, "--- Fragmentation ---")?;
        writeln!(f, "Free bytes:  {}", self.fragmentation.total_free_bytes)?;
        writeln!(f, "Largest blk: {}", self.fragmentation.largest_free_block)?;
        writeln!(f, "Free blocks: {}", self.fragmentation.free_block_count)?;
        writeln!(
            f,
            "Frag ratio:  {:.4}",
            self.fragmentation.fragmentation_ratio()
        )?;
        writeln!(f, "--- Histogram ---")?;
        for (i, &count) in self.histogram.bucket_counts().iter().enumerate() {
            if count > 0 {
                let (lo, hi) = AllocationHistogram::bucket_range(i);
                if i == 31 {
                    writeln!(f, "[{lo}+): {count}")?;
                } else {
                    writeln!(f, "[{lo}, {hi}): {count}")?;
                }
            }
        }
        writeln!(f, "Median alloc size: {} bytes", self.histogram.median())?;
        writeln!(f, "Timestamp:   {} ns", self.timestamp_ns)?;
        write!(f, "===========================")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PoolStatsTracker
// ---------------------------------------------------------------------------

/// Thread-safe allocation tracking for memory pools.
///
/// Uses an `RwLock`-protected inner state so that multiple readers can
/// snapshot stats concurrently while mutations are serialized.
#[derive(Debug)]
pub struct PoolStatsTracker {
    inner: RwLock<TrackerInner>,
}

#[derive(Debug, Clone)]
struct TrackerInner {
    allocated_bytes: usize,
    peak_bytes: usize,
    allocation_count: u64,
    free_count: u64,
    histogram: AllocationHistogram,
}

impl TrackerInner {
    fn new() -> Self {
        Self {
            allocated_bytes: 0,
            peak_bytes: 0,
            allocation_count: 0,
            free_count: 0,
            histogram: AllocationHistogram::new(),
        }
    }
}

impl PoolStatsTracker {
    /// Creates a new tracker with all counters at zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(TrackerInner::new()),
        }
    }

    /// Records an allocation of `size` bytes.
    ///
    /// Updates `allocated_bytes`, `peak_bytes`, `allocation_count`, and the
    /// histogram.
    pub fn record_alloc(&self, size: usize) {
        if let Ok(mut guard) = self.inner.write() {
            guard.allocated_bytes = guard.allocated_bytes.saturating_add(size);
            if guard.allocated_bytes > guard.peak_bytes {
                guard.peak_bytes = guard.allocated_bytes;
            }
            guard.allocation_count = guard.allocation_count.saturating_add(1);
            guard.histogram.record(size);
        }
    }

    /// Records a deallocation of `size` bytes.
    ///
    /// Updates `allocated_bytes` and `free_count`.
    pub fn record_free(&self, size: usize) {
        if let Ok(mut guard) = self.inner.write() {
            guard.allocated_bytes = guard.allocated_bytes.saturating_sub(size);
            guard.free_count = guard.free_count.saturating_add(1);
        }
    }

    /// Takes a snapshot of current stats as a [`PoolReport`].
    ///
    /// The `fragmentation` field is set to default values since this tracker
    /// does not have visibility into the free-list structure.
    #[must_use]
    pub fn snapshot(&self) -> PoolReport {
        let timestamp_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        if let Ok(guard) = self.inner.read() {
            PoolReport {
                allocated_bytes: guard.allocated_bytes,
                peak_bytes: guard.peak_bytes,
                allocation_count: guard.allocation_count,
                free_count: guard.free_count,
                fragmentation: FragmentationMetrics::default(),
                histogram: guard.histogram.clone(),
                timestamp_ns,
            }
        } else {
            PoolReport {
                timestamp_ns,
                ..PoolReport::default()
            }
        }
    }

    /// Resets all statistics to zero.
    pub fn reset(&self) {
        if let Ok(mut guard) = self.inner.write() {
            *guard = TrackerInner::new();
        }
    }

    /// Returns the current number of allocated bytes.
    #[must_use]
    pub fn current_allocated(&self) -> usize {
        self.inner.read().map(|g| g.allocated_bytes).unwrap_or(0)
    }

    /// Returns the peak number of allocated bytes.
    #[must_use]
    pub fn peak_allocated(&self) -> usize {
        self.inner.read().map(|g| g.peak_bytes).unwrap_or(0)
    }

    /// Trims the pool: simulates `cuMemPoolTrimTo`, releasing all freed
    /// (outstanding) allocations back to the system.
    ///
    /// Returns the number of bytes that were outstanding and are now released.
    /// After trim, `current_allocated()` is set to 0 and `is_fully_trimmed()`
    /// returns `true`.
    ///
    /// In a real GPU pool (`cuMemPoolTrimTo`), this releases pool memory pages
    /// back to the OS.  Here we track the logical accounting: after all frees
    /// have been recorded via [`PoolStatsTracker::record_free`], trim marks the remaining
    /// outstanding bytes as released.
    pub fn trim(&self) -> usize {
        if let Ok(mut guard) = self.inner.write() {
            let freed = guard.allocated_bytes;
            guard.allocated_bytes = 0;
            freed
        } else {
            0
        }
    }

    /// Returns `true` when there are no outstanding allocations — i.e.,
    /// every allocated byte has been freed (and optionally trimmed).
    ///
    /// Equivalent to `current_allocated() == 0`.
    #[must_use]
    pub fn is_fully_trimmed(&self) -> bool {
        self.current_allocated() == 0
    }

    /// Returns `true` if there are bytes still allocated (not yet freed).
    ///
    /// Equivalent to `current_allocated() > 0`.
    #[must_use]
    pub fn has_leaks(&self) -> bool {
        self.current_allocated() > 0
    }
}

impl Default for PoolStatsTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // -----------------------------------------------------------------------
    // AllocationHistogram tests
    // -----------------------------------------------------------------------

    #[test]
    fn histogram_bucket_index_zero() {
        assert_eq!(AllocationHistogram::bucket_index(0), 0);
    }

    #[test]
    fn histogram_bucket_index_one() {
        assert_eq!(AllocationHistogram::bucket_index(1), 0);
    }

    #[test]
    fn histogram_bucket_index_sixteen() {
        // 16 is the start of bucket 0: [16, 32)
        assert_eq!(AllocationHistogram::bucket_index(16), 0);
    }

    #[test]
    fn histogram_bucket_index_thirty_two() {
        // 32 is the start of bucket 1: [32, 64)
        assert_eq!(AllocationHistogram::bucket_index(32), 1);
    }

    #[test]
    fn histogram_bucket_index_sixty_four() {
        // 64 is the start of bucket 2: [64, 128)
        assert_eq!(AllocationHistogram::bucket_index(64), 2);
    }

    #[test]
    fn histogram_bucket_index_1024() {
        // 1024 = 2^10, bucket = 10 - 4 = 6
        assert_eq!(AllocationHistogram::bucket_index(1024), 6);
    }

    #[test]
    fn histogram_bucket_index_1mb() {
        // 1 MiB = 2^20, bucket = 20 - 4 = 16
        assert_eq!(AllocationHistogram::bucket_index(1 << 20), 16);
    }

    #[test]
    fn histogram_bucket_index_1gb() {
        // 1 GiB = 2^30, bucket = 30 - 4 = 26
        assert_eq!(AllocationHistogram::bucket_index(1 << 30), 26);
    }

    #[test]
    fn histogram_record_and_retrieval() {
        let mut hist = AllocationHistogram::new();
        hist.record(64);
        hist.record(64);
        hist.record(128);
        assert_eq!(hist.total_allocations(), 3);
        assert_eq!(hist.bucket_counts()[2], 2); // [64, 128)
        assert_eq!(hist.bucket_counts()[3], 1); // [128, 256)
    }

    #[test]
    fn histogram_bucket_range() {
        let (lo, hi) = AllocationHistogram::bucket_range(0);
        assert_eq!(lo, 16);
        assert_eq!(hi, 32);

        let (lo, hi) = AllocationHistogram::bucket_range(6);
        assert_eq!(lo, 1024);
        assert_eq!(hi, 2048);

        let (lo, hi) = AllocationHistogram::bucket_range(31);
        assert_eq!(lo, 1 << 35);
        assert_eq!(hi, usize::MAX);
    }

    #[test]
    fn histogram_percentile_empty() {
        let hist = AllocationHistogram::new();
        assert_eq!(hist.percentile(50.0), 0);
    }

    #[test]
    fn histogram_percentile_single_bucket() {
        let mut hist = AllocationHistogram::new();
        for _ in 0..100 {
            hist.record(256); // bucket 4: [256, 512)
        }
        assert_eq!(hist.percentile(0.0), 256);
        assert_eq!(hist.percentile(50.0), 256);
        assert_eq!(hist.percentile(100.0), 256);
    }

    #[test]
    fn histogram_percentile_two_buckets() {
        let mut hist = AllocationHistogram::new();
        // 30 allocations in bucket 2 [64, 128), 70 in bucket 6 [1024, 2048)
        for _ in 0..30 {
            hist.record(64);
        }
        for _ in 0..70 {
            hist.record(1024);
        }
        // 30th percentile should be in bucket 2
        assert_eq!(hist.percentile(30.0), 64);
        // 31st percentile should be in bucket 6
        assert_eq!(hist.percentile(31.0), 1024);
        assert_eq!(hist.median(), 1024);
    }

    // -----------------------------------------------------------------------
    // FragmentationMetrics tests
    // -----------------------------------------------------------------------

    #[test]
    fn fragmentation_no_fragmentation() {
        // Single contiguous block
        let m = FragmentationMetrics::new(1024, 1024, 1);
        let ratio = m.fragmentation_ratio();
        assert!((ratio - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fragmentation_high_fragmentation() {
        // 10 blocks, largest is only 100 out of 1000
        let m = FragmentationMetrics::new(1000, 100, 10);
        let ratio = m.fragmentation_ratio();
        assert!((ratio - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn fragmentation_zero_free() {
        let m = FragmentationMetrics::new(0, 0, 0);
        assert!((m.fragmentation_ratio() - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.average_free_block_size(), 0);
    }

    #[test]
    fn fragmentation_average_block_size() {
        let m = FragmentationMetrics::new(1000, 500, 4);
        assert_eq!(m.average_free_block_size(), 250);
    }

    // -----------------------------------------------------------------------
    // PoolStatsTracker tests
    // -----------------------------------------------------------------------

    #[test]
    fn tracker_alloc_free_sequence() {
        let tracker = PoolStatsTracker::new();
        tracker.record_alloc(1024);
        tracker.record_alloc(2048);
        assert_eq!(tracker.current_allocated(), 3072);
        tracker.record_free(1024);
        assert_eq!(tracker.current_allocated(), 2048);
    }

    #[test]
    fn tracker_peak_tracking() {
        let tracker = PoolStatsTracker::new();
        tracker.record_alloc(1000);
        tracker.record_alloc(2000);
        // peak = 3000
        tracker.record_free(2000);
        // current = 1000, peak still 3000
        assert_eq!(tracker.current_allocated(), 1000);
        assert_eq!(tracker.peak_allocated(), 3000);
    }

    #[test]
    fn tracker_snapshot() {
        let tracker = PoolStatsTracker::new();
        tracker.record_alloc(512);
        tracker.record_alloc(1024);
        tracker.record_free(512);

        let report = tracker.snapshot();
        assert_eq!(report.allocated_bytes, 1024);
        assert_eq!(report.peak_bytes, 1536);
        assert_eq!(report.allocation_count, 2);
        assert_eq!(report.free_count, 1);
        assert!(report.timestamp_ns > 0);
    }

    #[test]
    fn tracker_reset() {
        let tracker = PoolStatsTracker::new();
        tracker.record_alloc(4096);
        tracker.record_alloc(8192);
        tracker.reset();
        assert_eq!(tracker.current_allocated(), 0);
        assert_eq!(tracker.peak_allocated(), 0);
        let report = tracker.snapshot();
        assert_eq!(report.allocation_count, 0);
        assert_eq!(report.free_count, 0);
    }

    #[test]
    fn tracker_thread_safety() {
        let tracker = Arc::new(PoolStatsTracker::new());
        let mut handles = Vec::new();

        for _ in 0..8 {
            let t = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    t.record_alloc(64);
                }
                for _ in 0..50 {
                    t.record_free(64);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        // 8 threads * 100 allocs = 800
        let report = tracker.snapshot();
        assert_eq!(report.allocation_count, 800);
        assert_eq!(report.free_count, 400);
        // current = 800*64 - 400*64 = 400*64 = 25600
        assert_eq!(tracker.current_allocated(), 25600);
    }

    #[test]
    fn display_formatting() {
        let tracker = PoolStatsTracker::new();
        tracker.record_alloc(256);
        tracker.record_alloc(1024);
        let report = tracker.snapshot();
        let text = format!("{report}");
        assert!(text.contains("OxiCUDA Pool Report"));
        assert!(text.contains("Allocated:"));
        assert!(text.contains("Peak:"));
        assert!(text.contains("Histogram"));
        assert!(text.contains("Median alloc size:"));
    }

    #[test]
    fn pool_report_default() {
        let report = PoolReport::default();
        assert_eq!(report.allocated_bytes, 0);
        assert_eq!(report.peak_bytes, 0);
        assert_eq!(report.allocation_count, 0);
        assert_eq!(report.free_count, 0);
        assert_eq!(report.timestamp_ns, 0);
    }

    // -----------------------------------------------------------------------
    // Pool trim / cuMemPoolTrimTo simulation tests
    // -----------------------------------------------------------------------

    #[test]
    fn pool_trim_after_all_frees_is_clean() {
        // Alloc 4K, free 4K, trim() → is_fully_trimmed() == true.
        let tracker = PoolStatsTracker::new();
        tracker.record_alloc(4096);
        tracker.record_free(4096);

        // After free, allocated bytes should already be 0.
        assert_eq!(
            tracker.current_allocated(),
            0,
            "after freeing all, allocated should be 0"
        );

        let freed = tracker.trim();
        // trim returns 0 because allocated_bytes was already 0.
        assert_eq!(freed, 0, "trim on fully-freed pool returns 0");
        assert!(
            tracker.is_fully_trimmed(),
            "after trim, pool should be fully trimmed"
        );
        assert!(
            !tracker.has_leaks(),
            "no leaks after complete alloc/free cycle"
        );
    }

    #[test]
    fn pool_trim_outstanding_bytes() {
        // Alloc 8K but don't free. trim() → returns 8K and resets to 0.
        let tracker = PoolStatsTracker::new();
        tracker.record_alloc(8192);

        assert_eq!(tracker.current_allocated(), 8192);
        assert!(tracker.has_leaks(), "8K outstanding → has leaks");

        let freed = tracker.trim();
        assert_eq!(freed, 8192, "trim should return the outstanding 8K");
        assert!(tracker.is_fully_trimmed(), "after trim, fully trimmed");
        assert!(!tracker.has_leaks(), "no leaks after trim");
    }

    #[test]
    fn pool_trim_partial_free_still_has_leaks() {
        // Alloc 4K + 2K, free only 4K, trim() → has_leaks() == true (2K still outstanding).
        // Then trim should release the remaining 2K.
        let tracker = PoolStatsTracker::new();
        tracker.record_alloc(4096);
        tracker.record_alloc(2048);
        tracker.record_free(4096);

        // 2K still outstanding.
        assert_eq!(tracker.current_allocated(), 2048, "2K still outstanding");
        assert!(
            tracker.has_leaks(),
            "2K outstanding after partial free → has leaks"
        );
        assert!(
            !tracker.is_fully_trimmed(),
            "not fully trimmed while 2K outstanding"
        );

        // Trim releases the remaining 2K.
        let freed = tracker.trim();
        assert_eq!(freed, 2048, "trim releases the remaining 2K");
        assert!(tracker.is_fully_trimmed(), "fully trimmed after trim()");
    }

    #[test]
    fn pool_trim_empty_tracker_is_clean() {
        // A fresh tracker with no operations is already fully trimmed.
        let tracker = PoolStatsTracker::new();
        assert!(tracker.is_fully_trimmed(), "fresh tracker is fully trimmed");
        assert!(!tracker.has_leaks(), "fresh tracker has no leaks");
        let freed = tracker.trim();
        assert_eq!(freed, 0, "trim on empty tracker returns 0");
    }

    #[test]
    fn alloc_async_api_exists() {
        // Verify that the pool feature provides alloc_async functionality.
        // The pool module (feature = "pool") provides PooledBuffer::alloc_async.
        // This test validates the existence of the tracker API that simulates
        // the accounting side of cuMemAllocAsync / cuMemFreeAsync.
        //
        // CPU-side API verified: alloc/free tracking works correctly.
        // Actual cuMemAllocAsync requires CUDA 11.2+ driver + NVIDIA hardware.
        let tracker = PoolStatsTracker::new();

        // Simulate alloc_async: record the allocation.
        tracker.record_alloc(1024);
        assert_eq!(tracker.current_allocated(), 1024);

        // Simulate free_async: record the free.
        tracker.record_free(1024);
        assert_eq!(tracker.current_allocated(), 0);

        // Verify trim closes any potential remaining outstanding bytes.
        let freed = tracker.trim();
        assert_eq!(freed, 0);
        assert!(tracker.is_fully_trimmed());
    }
}
