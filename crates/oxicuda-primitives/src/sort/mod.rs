//! Device-wide sort algorithms.
//!
//! * [`radix_sort`] — 4-bit LSD radix sort; fastest for integer keys.
//! * [`merge_sort`] — stable bitonic-block + binary-search merge sort.

pub mod merge_sort;
pub mod radix_sort;

pub use merge_sort::{MergeSortConfig, MergeSortTemplate};
pub use radix_sort::{RadixSortConfig, RadixSortTemplate};
