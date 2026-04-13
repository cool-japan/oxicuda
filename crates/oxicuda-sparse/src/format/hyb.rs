//! HYB (Hybrid ELL+COO) sparse matrix format.
//!
//! The HYB format splits a sparse matrix into two portions:
//! - An **ELL** (ELLPACK) portion storing the regular part: up to `ell_width`
//!   entries per row, padded with sentinel values (-1) for unused slots.
//! - A **COO** (Coordinate) overflow portion storing entries that exceed the
//!   ELL width.
//!
//! This hybrid approach combines the coalesced memory access benefits of ELL
//! (for rows with typical density) with the flexibility of COO (for rows
//! with atypical density), yielding better performance than either format
//! alone on matrices with irregular sparsity patterns.
//!
//! The [`HybPartition`] enum controls how the ELL width is chosen:
//! - `Auto` uses the median row nnz (good default).
//! - `Max` uses the maximum row nnz (pure ELL, no COO overflow).
//! - `Threshold(f64)` uses a percentile of the row nnz distribution.
//! - `Fixed(usize)` lets the user specify the ELL width directly.

use std::mem;

use oxicuda_blas::GpuFloat;

use crate::error::{SparseError, SparseResult};
use crate::format::CsrMatrix;

/// Sentinel value for unused ELL entries in the HYB format.
pub const HYB_ELL_SENTINEL: i32 = -1;

/// Strategy for partitioning non-zeros between ELL and COO portions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HybPartition {
    /// Automatic: use the median nnz per row as the ELL width.
    Auto,
    /// Maximum: use the max nnz per row (pure ELL, zero COO overflow).
    Max,
    /// Percentile threshold: use the given percentile (0.0--1.0) of the
    /// per-row nnz distribution as the ELL width. For example, 0.9 means
    /// the 90th percentile.
    Threshold(f64),
    /// Fixed ELL width chosen by the user.
    Fixed(usize),
}

/// A sparse matrix in HYB (Hybrid ELL+COO) format, stored on host.
///
/// The ELL portion uses column-major layout: element `(row, k)` is at index
/// `k * rows + row`. Unused ELL slots have column index [`HYB_ELL_SENTINEL`]
/// and a zero value.
///
/// The COO portion stores overflow entries as unsorted `(row, col, value)`
/// triplets.
#[derive(Debug, Clone)]
pub struct HybMatrix<T: GpuFloat> {
    /// Number of rows in the matrix.
    rows: usize,
    /// Number of columns in the matrix.
    cols: usize,
    /// ELL width: maximum entries stored per row in the ELL portion.
    ell_width: usize,
    /// ELL column indices, length `rows * ell_width`, column-major.
    /// Unused entries are [`HYB_ELL_SENTINEL`] (-1).
    ell_col_indices: Vec<i32>,
    /// ELL values, length `rows * ell_width`, column-major.
    /// Unused entries are zero.
    ell_values: Vec<T>,
    /// Number of overflow entries stored in the COO portion.
    coo_nnz: usize,
    /// COO overflow row indices, length `coo_nnz`.
    coo_row_indices: Vec<i32>,
    /// COO overflow column indices, length `coo_nnz`.
    coo_col_indices: Vec<i32>,
    /// COO overflow values, length `coo_nnz`.
    coo_values: Vec<T>,
}

/// Statistics describing the quality of the HYB partition.
#[derive(Debug, Clone, Copy)]
pub struct HybStatistics {
    /// Fraction of total nnz stored in the ELL portion (0.0--1.0).
    pub ell_fraction: f64,
    /// Fraction of total nnz stored in the COO portion (0.0--1.0).
    pub coo_fraction: f64,
    /// Ratio of wasted (padding) slots in the ELL portion.
    /// Computed as `(ell_total_slots - ell_nnz) / ell_total_slots`.
    pub ell_padding_ratio: f64,
    /// Total memory consumption of the HYB format in bytes.
    pub memory_bytes: usize,
    /// Memory consumption of an equivalent CSR representation in bytes,
    /// for comparison purposes.
    pub csr_memory_bytes: usize,
}

/// Compute the optimal ELL width from a distribution of per-row nnz counts.
///
/// The heuristic balances two competing goals:
/// 1. Minimise ELL padding waste (favours small width).
/// 2. Minimise COO overflow volume (favours large width).
///
/// It selects the width that minimises estimated total storage cost,
/// modelling ELL cost as `rows * width * (sizeof(i32) + sizeof(T))` and COO
/// cost as `overflow_nnz * (2 * sizeof(i32) + sizeof(T))`.
///
/// Returns 1 if `row_nnz` is empty.
pub fn optimal_ell_width<T: GpuFloat>(row_nnz: &[usize]) -> usize {
    if row_nnz.is_empty() {
        return 1;
    }

    let rows = row_nnz.len();
    let max_nnz = row_nnz.iter().copied().max().unwrap_or(1);
    if max_nnz == 0 {
        return 1;
    }

    let ell_entry_bytes = mem::size_of::<i32>() + mem::size_of::<T>();
    let coo_entry_bytes = 2 * mem::size_of::<i32>() + mem::size_of::<T>();

    let mut best_width = 1;
    let mut best_cost = usize::MAX;

    // Evaluate each candidate width from 1..=max_nnz
    for w in 1..=max_nnz {
        let ell_cost = rows * w * ell_entry_bytes;
        let overflow: usize = row_nnz.iter().map(|&r| r.saturating_sub(w)).sum();
        let coo_cost = overflow * coo_entry_bytes;
        let total = ell_cost + coo_cost;
        if total < best_cost {
            best_cost = total;
            best_width = w;
        }
    }

    best_width
}

/// Compute the ELL width from a per-row nnz distribution and a partition
/// strategy.
fn compute_ell_width(row_nnz: &[usize], partition: HybPartition) -> usize {
    if row_nnz.is_empty() {
        return 1;
    }
    match partition {
        HybPartition::Auto => {
            // Median of the per-row nnz distribution
            let mut sorted = row_nnz.to_vec();
            sorted.sort_unstable();
            let mid = sorted.len() / 2;
            let median = if sorted.len() % 2 == 0 {
                (sorted[mid.saturating_sub(1)] + sorted[mid]) / 2
            } else {
                sorted[mid]
            };
            median.max(1)
        }
        HybPartition::Max => row_nnz.iter().copied().max().unwrap_or(1).max(1),
        HybPartition::Threshold(pct) => {
            let pct = pct.clamp(0.0, 1.0);
            let mut sorted = row_nnz.to_vec();
            sorted.sort_unstable();
            let idx = ((sorted.len() as f64 * pct).ceil() as usize)
                .min(sorted.len())
                .saturating_sub(1);
            sorted[idx].max(1)
        }
        HybPartition::Fixed(w) => w.max(1),
    }
}

impl<T: GpuFloat> HybMatrix<T> {
    /// Creates a new HYB matrix from raw host-side arrays.
    ///
    /// # Arguments
    ///
    /// * `rows` -- Number of rows.
    /// * `cols` -- Number of columns.
    /// * `ell_width` -- Maximum entries per row in the ELL portion.
    /// * `ell_col_indices` -- ELL column indices, length `rows * ell_width`,
    ///   column-major. Unused entries must be [`HYB_ELL_SENTINEL`].
    /// * `ell_values` -- ELL values, length `rows * ell_width`, column-major.
    /// * `coo_row_indices` -- COO overflow row indices.
    /// * `coo_col_indices` -- COO overflow column indices.
    /// * `coo_values` -- COO overflow values.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidFormat`] if array lengths are inconsistent.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: usize,
        cols: usize,
        ell_width: usize,
        ell_col_indices: Vec<i32>,
        ell_values: Vec<T>,
        coo_row_indices: Vec<i32>,
        coo_col_indices: Vec<i32>,
        coo_values: Vec<T>,
    ) -> SparseResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(SparseError::InvalidFormat(
                "rows and cols must be non-zero".to_string(),
            ));
        }
        if ell_width == 0 {
            return Err(SparseError::InvalidFormat(
                "ell_width must be non-zero".to_string(),
            ));
        }
        let ell_total = rows * ell_width;
        if ell_col_indices.len() != ell_total {
            return Err(SparseError::InvalidFormat(format!(
                "ell_col_indices length ({}) must be rows * ell_width ({})",
                ell_col_indices.len(),
                ell_total
            )));
        }
        if ell_values.len() != ell_total {
            return Err(SparseError::InvalidFormat(format!(
                "ell_values length ({}) must be rows * ell_width ({})",
                ell_values.len(),
                ell_total
            )));
        }
        let coo_nnz = coo_values.len();
        if coo_row_indices.len() != coo_nnz || coo_col_indices.len() != coo_nnz {
            return Err(SparseError::InvalidFormat(format!(
                "COO arrays must have equal length: row_indices={}, col_indices={}, values={}",
                coo_row_indices.len(),
                coo_col_indices.len(),
                coo_nnz
            )));
        }

        Ok(Self {
            rows,
            cols,
            ell_width,
            ell_col_indices,
            ell_values,
            coo_nnz,
            coo_row_indices,
            coo_col_indices,
            coo_values,
        })
    }

    /// Constructs a HYB matrix from a CSR matrix using the given partition
    /// strategy.
    ///
    /// Downloads the CSR data from GPU, computes per-row nnz, determines
    /// the ELL width according to `partition`, then fills ELL up to that
    /// width and overflows the remainder into COO.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on GPU transfer failure.
    pub fn from_csr(csr: &CsrMatrix<T>, partition: HybPartition) -> SparseResult<Self> {
        let (h_row_ptr, h_col_idx, h_values) = csr.to_host()?;
        let rows = csr.rows() as usize;
        let cols = csr.cols() as usize;

        // Compute per-row nnz
        let mut row_nnz = Vec::with_capacity(rows);
        for i in 0..rows {
            row_nnz.push((h_row_ptr[i + 1] - h_row_ptr[i]) as usize);
        }

        let ell_width = compute_ell_width(&row_nnz, partition);
        Self::build_from_csr_host(rows, cols, ell_width, &h_row_ptr, &h_col_idx, &h_values)
    }

    /// Constructs a HYB matrix from a COO matrix (via CSR intermediate)
    /// using the given partition strategy.
    ///
    /// Converts COO to CSR first, then builds HYB from the CSR data.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on GPU transfer failure.
    pub fn from_coo(coo: &super::CooMatrix<T>, partition: HybPartition) -> SparseResult<Self> {
        let csr = coo.to_csr()?;
        Self::from_csr(&csr, partition)
    }

    /// Converts this HYB matrix back to CSR format, uploading to GPU.
    ///
    /// Merges the ELL and COO portions into a single CSR representation.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on GPU allocation failure.
    /// Returns [`SparseError::ZeroNnz`] if the matrix has no non-zeros.
    pub fn to_csr(&self) -> SparseResult<CsrMatrix<T>> {
        let total_nnz = self.total_nnz();
        if total_nnz == 0 {
            return Err(SparseError::ZeroNnz);
        }

        // Count per-row nnz from both ELL and COO
        let mut row_counts = vec![0usize; self.rows];

        // ELL contribution
        for (i, count) in row_counts.iter_mut().enumerate() {
            for k in 0..self.ell_width {
                let idx = k * self.rows + i;
                if self.ell_col_indices[idx] != HYB_ELL_SENTINEL {
                    *count += 1;
                }
            }
        }

        // COO contribution
        for &r in &self.coo_row_indices {
            let row = r as usize;
            if row < self.rows {
                row_counts[row] += 1;
            }
        }

        // Build row_ptr
        let mut h_row_ptr = vec![0i32; self.rows + 1];
        for i in 0..self.rows {
            h_row_ptr[i + 1] = h_row_ptr[i] + row_counts[i] as i32;
        }

        // Place entries
        let mut h_col_idx = vec![0i32; total_nnz];
        let mut h_values = vec![T::gpu_zero(); total_nnz];
        let mut write_pos: Vec<i32> = h_row_ptr.clone();

        // ELL entries first
        for (i, pos) in write_pos.iter_mut().enumerate().take(self.rows) {
            for k in 0..self.ell_width {
                let idx = k * self.rows + i;
                let col = self.ell_col_indices[idx];
                if col != HYB_ELL_SENTINEL {
                    let dest = *pos as usize;
                    h_col_idx[dest] = col;
                    h_values[dest] = self.ell_values[idx];
                    *pos += 1;
                }
            }
        }

        // COO entries
        for j in 0..self.coo_nnz {
            let row = self.coo_row_indices[j] as usize;
            if row < self.rows {
                let dest = write_pos[row] as usize;
                h_col_idx[dest] = self.coo_col_indices[j];
                h_values[dest] = self.coo_values[j];
                write_pos[row] += 1;
            }
        }

        CsrMatrix::from_host(
            self.rows as u32,
            self.cols as u32,
            &h_row_ptr,
            &h_col_idx,
            &h_values,
        )
    }

    /// Returns the total number of non-zeros (ELL + COO).
    pub fn total_nnz(&self) -> usize {
        self.ell_nnz() + self.coo_nnz
    }

    /// Returns the number of non-zeros (alias for [`total_nnz`](Self::total_nnz)).
    #[inline]
    pub fn nnz(&self) -> usize {
        self.total_nnz()
    }

    /// Returns `true` if the matrix contains no non-zero entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_nnz() == 0
    }

    /// Returns the number of non-zeros stored in the ELL portion.
    pub fn ell_nnz(&self) -> usize {
        self.ell_col_indices
            .iter()
            .filter(|&&c| c != HYB_ELL_SENTINEL)
            .count()
    }

    /// Returns the number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the ELL width (max entries per row in ELL portion).
    #[inline]
    pub fn ell_width(&self) -> usize {
        self.ell_width
    }

    /// Returns a reference to the ELL column indices (column-major).
    #[inline]
    pub fn ell_col_indices(&self) -> &[i32] {
        &self.ell_col_indices
    }

    /// Returns a reference to the ELL values (column-major).
    #[inline]
    pub fn ell_values(&self) -> &[T] {
        &self.ell_values
    }

    /// Returns the number of overflow entries in the COO portion.
    #[inline]
    pub fn coo_nnz(&self) -> usize {
        self.coo_nnz
    }

    /// Returns a reference to the COO overflow row indices.
    #[inline]
    pub fn coo_row_indices(&self) -> &[i32] {
        &self.coo_row_indices
    }

    /// Returns a reference to the COO overflow column indices.
    #[inline]
    pub fn coo_col_indices(&self) -> &[i32] {
        &self.coo_col_indices
    }

    /// Returns a reference to the COO overflow values.
    #[inline]
    pub fn coo_values(&self) -> &[T] {
        &self.coo_values
    }

    /// Computes statistics describing the quality of the HYB partition.
    ///
    /// The returned [`HybStatistics`] includes the fraction of nnz in each
    /// portion, the ELL padding ratio, and memory consumption compared to
    /// an equivalent CSR matrix.
    pub fn statistics(&self) -> HybStatistics {
        let ell_nnz = self.ell_nnz();
        let total_nnz = ell_nnz + self.coo_nnz;

        let (ell_fraction, coo_fraction) = if total_nnz == 0 {
            (0.0, 0.0)
        } else {
            (
                ell_nnz as f64 / total_nnz as f64,
                self.coo_nnz as f64 / total_nnz as f64,
            )
        };

        let ell_total_slots = self.rows * self.ell_width;
        let ell_padding_ratio = if ell_total_slots == 0 {
            0.0
        } else {
            (ell_total_slots - ell_nnz) as f64 / ell_total_slots as f64
        };

        // HYB memory: ELL (col_indices + values) + COO (row + col + values)
        let ell_mem = ell_total_slots * (mem::size_of::<i32>() + mem::size_of::<T>());
        let coo_mem = self.coo_nnz * (2 * mem::size_of::<i32>() + mem::size_of::<T>());
        let memory_bytes = ell_mem + coo_mem;

        // CSR memory: row_ptr + col_idx + values
        let csr_memory_bytes = (self.rows + 1) * mem::size_of::<i32>()
            + total_nnz * mem::size_of::<i32>()
            + total_nnz * mem::size_of::<T>();

        HybStatistics {
            ell_fraction,
            coo_fraction,
            ell_padding_ratio,
            memory_bytes,
            csr_memory_bytes,
        }
    }

    /// Internal helper: build HYB from host-side CSR arrays.
    fn build_from_csr_host(
        rows: usize,
        cols: usize,
        ell_width: usize,
        h_row_ptr: &[i32],
        h_col_idx: &[i32],
        h_values: &[T],
    ) -> SparseResult<Self> {
        let ell_total = rows * ell_width;
        let mut ell_col_indices = vec![HYB_ELL_SENTINEL; ell_total];
        let mut ell_values = vec![T::gpu_zero(); ell_total];

        let mut coo_row_indices = Vec::new();
        let mut coo_col_indices = Vec::new();
        let mut coo_values = Vec::new();

        for i in 0..rows {
            let start = h_row_ptr[i] as usize;
            let end = h_row_ptr[i + 1] as usize;
            let row_entries = end - start;

            // Fill ELL portion (up to ell_width entries)
            let ell_count = row_entries.min(ell_width);
            for k in 0..ell_count {
                let ell_idx = k * rows + i; // column-major
                ell_col_indices[ell_idx] = h_col_idx[start + k];
                ell_values[ell_idx] = h_values[start + k];
            }

            // Overflow goes to COO
            if row_entries > ell_width {
                for j in (start + ell_width)..end {
                    coo_row_indices.push(i as i32);
                    coo_col_indices.push(h_col_idx[j]);
                    coo_values.push(h_values[j]);
                }
            }
        }

        let coo_nnz = coo_values.len();

        Ok(Self {
            rows,
            cols,
            ell_width,
            ell_col_indices,
            ell_values,
            coo_nnz,
            coo_row_indices,
            coo_col_indices,
            coo_values,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a HYB matrix from host CSR arrays directly (no GPU).
    fn hyb_from_csr_host(
        rows: usize,
        cols: usize,
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[f64],
        partition: HybPartition,
    ) -> SparseResult<HybMatrix<f64>> {
        let mut row_nnz = Vec::with_capacity(rows);
        for i in 0..rows {
            row_nnz.push((row_ptr[i + 1] - row_ptr[i]) as usize);
        }
        let ell_width = compute_ell_width(&row_nnz, partition);
        HybMatrix::build_from_csr_host(rows, cols, ell_width, row_ptr, col_idx, values)
    }

    /// 4x4 identity-like test matrix:
    /// Row 0: (0,0)=1.0
    /// Row 1: (1,1)=2.0
    /// Row 2: (2,0)=3.0, (2,2)=4.0
    /// Row 3: (3,0)=5.0, (3,1)=6.0, (3,3)=7.0
    fn test_csr_data() -> (usize, usize, Vec<i32>, Vec<i32>, Vec<f64>) {
        let rows = 4;
        let cols = 4;
        let row_ptr = vec![0, 1, 2, 4, 7];
        let col_idx = vec![0, 1, 0, 2, 0, 1, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        (rows, cols, row_ptr, col_idx, values)
    }

    #[test]
    fn hyb_from_csr_auto_partition() {
        let (rows, cols, row_ptr, col_idx, values) = test_csr_data();
        // Row nnz: [1, 1, 2, 3] => median = (1+2)/2 = 1 (integer division)
        // Actually median of sorted [1,1,2,3]: even len => (sorted[1]+sorted[2])/2 = (1+2)/2 = 1
        let hyb = hyb_from_csr_host(rows, cols, &row_ptr, &col_idx, &values, HybPartition::Auto);
        assert!(hyb.is_ok());
        let hyb = hyb.expect("test helper");
        assert_eq!(hyb.rows(), 4);
        assert_eq!(hyb.cols(), 4);
        assert_eq!(hyb.ell_width(), 1);
        // ELL stores first entry per row: 4 entries
        assert_eq!(hyb.ell_nnz(), 4);
        // COO stores overflow: row 2 has 1 extra, row 3 has 2 extra => 3
        assert_eq!(hyb.coo_nnz(), 3);
        assert_eq!(hyb.total_nnz(), 7);
    }

    #[test]
    fn hyb_from_csr_max_partition() {
        let (rows, cols, row_ptr, col_idx, values) = test_csr_data();
        // Max nnz = 3, so pure ELL, no COO
        let hyb = hyb_from_csr_host(rows, cols, &row_ptr, &col_idx, &values, HybPartition::Max);
        let hyb = hyb.expect("test helper");
        assert_eq!(hyb.ell_width(), 3);
        assert_eq!(hyb.coo_nnz(), 0);
        assert_eq!(hyb.total_nnz(), 7);
    }

    #[test]
    fn hyb_from_csr_fixed_partition() {
        let (rows, cols, row_ptr, col_idx, values) = test_csr_data();
        let hyb = hyb_from_csr_host(
            rows,
            cols,
            &row_ptr,
            &col_idx,
            &values,
            HybPartition::Fixed(2),
        );
        let hyb = hyb.expect("test helper");
        assert_eq!(hyb.ell_width(), 2);
        // Rows: 1,1,2,3 => ELL stores min(nnz,2) each: 1+1+2+2 = 6
        assert_eq!(hyb.ell_nnz(), 6);
        // COO: row 3 has 1 overflow
        assert_eq!(hyb.coo_nnz(), 1);
        assert_eq!(hyb.total_nnz(), 7);
    }

    #[test]
    fn hyb_from_csr_threshold_partition() {
        let (rows, cols, row_ptr, col_idx, values) = test_csr_data();
        // Sorted row nnz: [1, 1, 2, 3], 90th percentile => index ceil(4*0.9)-1=3 => 3
        let hyb = hyb_from_csr_host(
            rows,
            cols,
            &row_ptr,
            &col_idx,
            &values,
            HybPartition::Threshold(0.9),
        );
        let hyb = hyb.expect("test helper");
        assert_eq!(hyb.ell_width(), 3);
        assert_eq!(hyb.coo_nnz(), 0);
    }

    #[test]
    fn hyb_to_csr_roundtrip() {
        let (rows, cols, row_ptr, col_idx, values) = test_csr_data();
        let hyb = hyb_from_csr_host(
            rows,
            cols,
            &row_ptr,
            &col_idx,
            &values,
            HybPartition::Fixed(2),
        );
        let hyb = hyb.expect("test helper");

        // to_csr requires GPU; test the internal logic by reconstructing manually
        // We verify that total_nnz is preserved and data is consistent
        assert_eq!(hyb.total_nnz(), 7);

        // Verify ELL col-major layout: element (row, k) at k*rows+row
        // Row 0, k=0: idx=0*4+0=0 => col 0
        assert_eq!(hyb.ell_col_indices()[0], 0);
        // Row 1, k=0: idx=0*4+1=1 => col 1
        assert_eq!(hyb.ell_col_indices()[1], 1);
        // Row 2, k=0: idx=0*4+2=2 => col 0
        assert_eq!(hyb.ell_col_indices()[2], 0);
        // Row 3, k=0: idx=0*4+3=3 => col 0
        assert_eq!(hyb.ell_col_indices()[3], 0);

        // Row 2, k=1: idx=1*4+2=6 => col 2
        assert_eq!(hyb.ell_col_indices()[6], 2);
        // Row 3, k=1: idx=1*4+3=7 => col 1
        assert_eq!(hyb.ell_col_indices()[7], 1);

        // COO overflow: row 3, entry (3,3)=7.0
        assert_eq!(hyb.coo_row_indices(), &[3]);
        assert_eq!(hyb.coo_col_indices(), &[3]);
        assert!((hyb.coo_values()[0] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn hyb_is_empty_all_padding() {
        // A matrix where all ELL entries are sentinel (but that's artificial)
        let ell_col = vec![HYB_ELL_SENTINEL; 4];
        let ell_val = vec![0.0f64; 4];
        let hyb = HybMatrix::new(2, 2, 2, ell_col, ell_val, vec![], vec![], vec![]);
        let hyb = hyb.expect("test helper");
        assert!(hyb.is_empty());
        assert_eq!(hyb.nnz(), 0);
    }

    #[test]
    fn hyb_statistics_pure_ell() {
        let (rows, cols, row_ptr, col_idx, values) = test_csr_data();
        let hyb = hyb_from_csr_host(rows, cols, &row_ptr, &col_idx, &values, HybPartition::Max);
        let hyb = hyb.expect("test helper");
        let stats = hyb.statistics();

        assert!((stats.ell_fraction - 1.0).abs() < 1e-12);
        assert!((stats.coo_fraction - 0.0).abs() < 1e-12);
        // ELL total slots = 4*3 = 12, nnz = 7, padding = 5/12
        assert!((stats.ell_padding_ratio - 5.0 / 12.0).abs() < 1e-12);
        assert!(stats.memory_bytes > 0);
        assert!(stats.csr_memory_bytes > 0);
    }

    #[test]
    fn hyb_statistics_mixed() {
        let (rows, cols, row_ptr, col_idx, values) = test_csr_data();
        let hyb = hyb_from_csr_host(
            rows,
            cols,
            &row_ptr,
            &col_idx,
            &values,
            HybPartition::Fixed(1),
        );
        let hyb = hyb.expect("test helper");
        let stats = hyb.statistics();

        // ELL nnz = 4 (one per row), COO nnz = 3
        assert!((stats.ell_fraction - 4.0 / 7.0).abs() < 1e-12);
        assert!((stats.coo_fraction - 3.0 / 7.0).abs() < 1e-12);
        // ELL padding = 0/4 = 0 (all slots used)
        assert!((stats.ell_padding_ratio - 0.0).abs() < 1e-12);
    }

    #[test]
    fn hyb_new_validation_bad_ell_lengths() {
        let result = HybMatrix::<f64>::new(
            2,
            2,
            2,
            vec![0; 3], // wrong length, should be 4
            vec![1.0; 4],
            vec![],
            vec![],
            vec![],
        );
        assert!(result.is_err());
    }

    #[test]
    fn hyb_new_validation_bad_coo_lengths() {
        let result = HybMatrix::<f64>::new(
            2,
            2,
            1,
            vec![HYB_ELL_SENTINEL; 2],
            vec![0.0; 2],
            vec![0],    // 1 entry
            vec![0, 1], // 2 entries -- mismatch
            vec![1.0],  // 1 entry
        );
        assert!(result.is_err());
    }

    #[test]
    fn hyb_new_validation_zero_rows() {
        let result = HybMatrix::<f64>::new(0, 2, 1, vec![], vec![], vec![], vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn hyb_new_validation_zero_ell_width() {
        let result = HybMatrix::<f64>::new(2, 2, 0, vec![], vec![], vec![], vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn hyb_ell_values_column_major_layout() {
        // 3x3 matrix:
        // Row 0: (0,0)=1, (0,1)=2
        // Row 1: (1,2)=3
        // Row 2: (2,0)=4, (2,1)=5, (2,2)=6
        let row_ptr = vec![0, 2, 3, 6];
        let col_idx = vec![0, 1, 2, 0, 1, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let hyb = hyb_from_csr_host(3, 3, &row_ptr, &col_idx, &values, HybPartition::Fixed(2));
        let hyb = hyb.expect("test helper");

        // ELL width=2, rows=3
        // (0,0): k=0, row=0 => idx=0*3+0=0 => col 0, val 1.0
        assert_eq!(hyb.ell_col_indices()[0], 0);
        assert!((hyb.ell_values()[0] - 1.0).abs() < 1e-12);

        // (1,0): k=0, row=1 => idx=0*3+1=1 => col 2, val 3.0
        assert_eq!(hyb.ell_col_indices()[1], 2);
        assert!((hyb.ell_values()[1] - 3.0).abs() < 1e-12);

        // (2,0): k=0, row=2 => idx=0*3+2=2 => col 0, val 4.0
        assert_eq!(hyb.ell_col_indices()[2], 0);

        // (0,1): k=1, row=0 => idx=1*3+0=3 => col 1, val 2.0
        assert_eq!(hyb.ell_col_indices()[3], 1);
        assert!((hyb.ell_values()[3] - 2.0).abs() < 1e-12);

        // (1,1): k=1, row=1 => idx=1*3+1=4 => sentinel (row 1 only has 1 entry)
        assert_eq!(hyb.ell_col_indices()[4], HYB_ELL_SENTINEL);

        // (2,1): k=1, row=2 => idx=1*3+2=5 => col 1, val 5.0
        assert_eq!(hyb.ell_col_indices()[5], 1);

        // COO overflow: row 2 has 1 overflow entry (2,2)=6.0
        assert_eq!(hyb.coo_nnz(), 1);
        assert_eq!(hyb.coo_row_indices(), &[2]);
        assert_eq!(hyb.coo_col_indices(), &[2]);
        assert!((hyb.coo_values()[0] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn optimal_ell_width_basic() {
        // All rows have 2 nnz => optimal width is 2
        let row_nnz = vec![2, 2, 2, 2];
        let w = optimal_ell_width::<f64>(&row_nnz);
        assert_eq!(w, 2);
    }

    #[test]
    fn optimal_ell_width_skewed() {
        // Most rows have 1 nnz, one outlier with 100
        let mut row_nnz = vec![1; 99];
        row_nnz.push(100);
        let w = optimal_ell_width::<f64>(&row_nnz);
        // Should pick a small width since padding 99 rows to 100 is wasteful
        assert!(w < 10, "expected small ELL width, got {w}");
    }

    #[test]
    fn optimal_ell_width_empty() {
        let w = optimal_ell_width::<f64>(&[]);
        assert_eq!(w, 1);
    }

    #[test]
    fn optimal_ell_width_all_zero() {
        let w = optimal_ell_width::<f64>(&[0, 0, 0]);
        assert_eq!(w, 1);
    }

    #[test]
    fn hyb_partition_threshold_boundary() {
        // Threshold(1.0) should behave like Max
        let (rows, cols, row_ptr, col_idx, values) = test_csr_data();
        let hyb = hyb_from_csr_host(
            rows,
            cols,
            &row_ptr,
            &col_idx,
            &values,
            HybPartition::Threshold(1.0),
        );
        let hyb = hyb.expect("test helper");
        assert_eq!(hyb.ell_width(), 3);
        assert_eq!(hyb.coo_nnz(), 0);
    }

    #[test]
    fn hyb_partition_threshold_zero() {
        // Threshold(0.0) should give width = 1 (minimum nnz)
        let (rows, cols, row_ptr, col_idx, values) = test_csr_data();
        let hyb = hyb_from_csr_host(
            rows,
            cols,
            &row_ptr,
            &col_idx,
            &values,
            HybPartition::Threshold(0.0),
        );
        let hyb = hyb.expect("test helper");
        assert_eq!(hyb.ell_width(), 1);
    }

    #[test]
    fn hyb_statistics_memory_comparison() {
        // For a well-partitioned matrix, HYB memory should be reasonable
        let (rows, cols, row_ptr, col_idx, values) = test_csr_data();
        let hyb = hyb_from_csr_host(
            rows,
            cols,
            &row_ptr,
            &col_idx,
            &values,
            HybPartition::Fixed(2),
        );
        let hyb = hyb.expect("test helper");
        let stats = hyb.statistics();

        // ELL: 4*2*(4+8) = 96 bytes
        // COO: 1*(8+8) = 20 bytes (2*4 + 8)
        // Total: 116 bytes
        let expected_ell = 4 * 2 * (std::mem::size_of::<i32>() + std::mem::size_of::<f64>());
        let expected_coo = 2 * std::mem::size_of::<i32>() + std::mem::size_of::<f64>();
        assert_eq!(stats.memory_bytes, expected_ell + expected_coo);

        // CSR: row_ptr(5*4) + col_idx(7*4) + values(7*8) = 20+28+56 = 104
        let expected_csr = 5 * std::mem::size_of::<i32>()
            + 7 * std::mem::size_of::<i32>()
            + 7 * std::mem::size_of::<f64>();
        assert_eq!(stats.csr_memory_bytes, expected_csr);
    }

    // --- New targeted tests ---

    #[test]
    fn hyb_identity_4x4_no_coo_overflow() {
        // 4x4 identity: each row has exactly 1 nnz
        // row_ptr = [0,1,2,3,4], col_idx = [0,1,2,3], values = [1,1,1,1]
        // Auto uses median([1,1,1,1]) = 1 => ell_width = 1, zero COO overflow
        let rows = 4usize;
        let cols = 4usize;
        let row_ptr = vec![0i32, 1, 2, 3, 4];
        let col_idx = vec![0i32, 1, 2, 3];
        let values = vec![1.0f64; 4];
        let hyb = hyb_from_csr_host(rows, cols, &row_ptr, &col_idx, &values, HybPartition::Auto)
            .expect("identity 4x4 hyb construction");
        assert_eq!(hyb.ell_width(), 1, "ell_width should be 1 for identity");
        assert_eq!(
            hyb.coo_nnz(),
            0,
            "no COO overflow for uniform-density matrix"
        );
        assert_eq!(hyb.total_nnz(), 4);
    }

    #[test]
    fn hyb_irregular_matrix_has_coo_entries() {
        // 4x4 matrix: row 0 has 5 entries (but max col = 3, so wrap), others have 1
        // Use 5 columns to allow 5 entries in row 0
        // row 0: cols 0,1,2,3,4 (5 entries)
        // row 1: col 0 (1 entry)
        // row 2: col 1 (1 entry)
        // row 3: col 2 (1 entry)
        // nnz: [5,1,1,1], median of sorted [1,1,1,5] => (1+1)/2 = 1 => ell_width=1
        // COO overflow: row 0 has 4 extra entries
        let rows = 4usize;
        let cols = 5usize;
        let row_ptr = vec![0i32, 5, 6, 7, 8];
        let col_idx = vec![0i32, 1, 2, 3, 4, 0, 1, 2];
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let hyb = hyb_from_csr_host(rows, cols, &row_ptr, &col_idx, &values, HybPartition::Auto)
            .expect("irregular hyb construction");
        assert_eq!(hyb.ell_width(), 1);
        assert_eq!(
            hyb.coo_nnz(),
            4,
            "row 0 overflows 4 entries into COO (5 nnz with ell_width=1)"
        );
    }

    #[test]
    fn hyb_spmv_matches_csr() {
        // 4x4 banded matrix: row i has entries at col i and col (i+1)%4
        // Use Fixed(1) => ell_width=1, COO gets the second entry per row
        // Verify y = A*x for x=[1,2,3,4] matches naive CSR multiply
        let rows = 4usize;
        let cols = 4usize;
        // Each row i: entries at col i (value=2.0) and col (i+1)%4 (value=1.0)
        let row_ptr = vec![0i32, 2, 4, 6, 8];
        let col_idx = vec![0i32, 1, 1, 2, 2, 3, 3, 0];
        let values = vec![2.0f64, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0];
        let x = [1.0f64, 2.0, 3.0, 4.0];

        // Compute naive CSR SpMV
        let mut y_csr = vec![0.0f64; rows];
        for i in 0..rows {
            let start = row_ptr[i] as usize;
            let end = row_ptr[i + 1] as usize;
            for j in start..end {
                y_csr[i] += values[j] * x[col_idx[j] as usize];
            }
        }

        // Build HYB with ell_width=1 (COO gets the second column per row)
        let hyb = hyb_from_csr_host(
            rows,
            cols,
            &row_ptr,
            &col_idx,
            &values,
            HybPartition::Fixed(1),
        )
        .expect("banded hyb construction");
        assert_eq!(hyb.coo_nnz(), 4, "each row overflows 1 entry to COO");

        // Compute HYB SpMV manually:
        // ELL part (column-major): slot k of row i is at index k*rows+i
        let mut y_hyb = vec![0.0f64; rows];
        for k in 0..hyb.ell_width() {
            for (i, y_val) in y_hyb.iter_mut().enumerate() {
                let idx = k * rows + i;
                let c = hyb.ell_col_indices()[idx];
                if c >= 0 {
                    *y_val += hyb.ell_values()[idx] * x[c as usize];
                }
            }
        }
        // COO part
        for idx in 0..hyb.coo_nnz() {
            let r = hyb.coo_row_indices()[idx] as usize;
            let c = hyb.coo_col_indices()[idx] as usize;
            y_hyb[r] += hyb.coo_values()[idx] * x[c];
        }

        for i in 0..rows {
            assert!(
                (y_hyb[i] - y_csr[i]).abs() < 1e-10,
                "HYB SpMV mismatch at row {}: hyb={}, csr={}",
                i,
                y_hyb[i],
                y_csr[i]
            );
        }
    }

    #[test]
    fn hyb_ell_width_is_avg_nnz() {
        // 4x4 matrix with 8 total nnz: each row has exactly 2 nnz
        // Use Fixed(2) to explicitly set ell_width=2 (avg = 8/4 = 2)
        let rows = 4usize;
        let cols = 4usize;
        let row_ptr = vec![0i32, 2, 4, 6, 8];
        let col_idx = vec![0i32, 1, 1, 2, 2, 3, 3, 0];
        let values = vec![1.0f64; 8];
        let hyb = hyb_from_csr_host(
            rows,
            cols,
            &row_ptr,
            &col_idx,
            &values,
            HybPartition::Fixed(2),
        )
        .expect("uniform 2-per-row hyb construction");
        assert_eq!(
            hyb.ell_width(),
            2,
            "ell_width should be 2 (= avg nnz per row)"
        );
        assert_eq!(
            hyb.coo_nnz(),
            0,
            "no overflow when ell_width == max nnz per row"
        );
    }

    #[test]
    fn hyb_coo_stores_overflow() {
        // Matrix where row 0 has 3 entries, rows 1-3 have 1 entry each
        // ell_width = Fixed(1) => row 0 overflows 2 entries to COO
        // Verify coo_row_indices contains [0, 0] for both overflow entries
        let rows = 4usize;
        let cols = 4usize;
        let row_ptr = vec![0i32, 3, 4, 5, 6];
        let col_idx = vec![0i32, 1, 2, 1, 2, 3];
        let values = vec![10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0];
        let hyb = hyb_from_csr_host(
            rows,
            cols,
            &row_ptr,
            &col_idx,
            &values,
            HybPartition::Fixed(1),
        )
        .expect("overflow test hyb construction");
        assert_eq!(hyb.coo_nnz(), 2, "row 0 overflows 2 entries to COO");
        // Both COO entries must belong to row 0
        for &r in hyb.coo_row_indices() {
            assert_eq!(r, 0i32, "all COO overflow entries should be from row 0");
        }
    }
}
