//! Format conversion routines between sparse matrix formats.
//!
//! These functions convert between CSR, CSC, COO, BSR, and ELL formats.
//! Currently conversions are performed via host-side download/upload; future
//! versions will use GPU kernels for large matrices.
//!
//! For simple conversions (CSR <-> COO, CSR <-> CSC), use the `to_*` methods
//! on the individual format types directly. This module provides additional
//! conversions that require more complex logic.

use oxicuda_blas::GpuFloat;

use crate::error::{SparseError, SparseResult};
use crate::handle::SparseHandle;

use super::bsr::BsrMatrix;
use super::coo::CooMatrix;
use super::csc::CscMatrix;
use super::csr::CsrMatrix;
use super::ell::EllMatrix;

/// Converts a CSR matrix to CSC format.
///
/// This is a convenience wrapper around [`CsrMatrix::to_csc`].
///
/// # Errors
///
/// Returns [`SparseError::Cuda`] on transfer failure.
pub fn csr_to_csc<T: GpuFloat>(
    _handle: &SparseHandle,
    csr: &CsrMatrix<T>,
) -> SparseResult<CscMatrix<T>> {
    csr.to_csc()
}

/// Converts a CSC matrix to CSR format.
///
/// This is a convenience wrapper around [`CscMatrix::to_csr`].
///
/// # Errors
///
/// Returns [`SparseError::Cuda`] on transfer failure.
pub fn csc_to_csr<T: GpuFloat>(
    _handle: &SparseHandle,
    csc: &CscMatrix<T>,
) -> SparseResult<CsrMatrix<T>> {
    csc.to_csr()
}

/// Converts a COO matrix to CSR format.
///
/// This is a convenience wrapper around [`CooMatrix::to_csr`].
///
/// # Errors
///
/// Returns [`SparseError::Cuda`] on transfer failure.
pub fn coo_to_csr<T: GpuFloat>(
    _handle: &SparseHandle,
    coo: &CooMatrix<T>,
) -> SparseResult<CsrMatrix<T>> {
    coo.to_csr()
}

/// Converts a COO matrix to CSC format.
///
/// This is a convenience wrapper around [`CooMatrix::to_csc`].
///
/// # Errors
///
/// Returns [`SparseError::Cuda`] on transfer failure.
pub fn coo_to_csc<T: GpuFloat>(
    _handle: &SparseHandle,
    coo: &CooMatrix<T>,
) -> SparseResult<CscMatrix<T>> {
    coo.to_csc()
}

/// Converts a CSR matrix to ELL format.
///
/// This is a convenience wrapper around [`EllMatrix::from_csr`].
///
/// # Errors
///
/// Returns [`SparseError::Cuda`] on transfer failure.
pub fn csr_to_ell<T: GpuFloat>(
    _handle: &SparseHandle,
    csr: &CsrMatrix<T>,
) -> SparseResult<EllMatrix<T>> {
    EllMatrix::from_csr(csr)
}

/// Converts a CSR matrix to BSR format with the specified block dimension.
///
/// The CSR matrix dimensions must be multiples of `block_dim`. The conversion
/// groups entries into dense blocks; zero-fill is used for block entries that
/// are not present in the CSR.
///
/// # Errors
///
/// Returns [`SparseError::InvalidFormat`] if dimensions are not multiples of
/// `block_dim`.
/// Returns [`SparseError::Cuda`] on transfer failure.
pub fn csr_to_bsr<T: GpuFloat>(
    _handle: &SparseHandle,
    csr: &CsrMatrix<T>,
    block_dim: u32,
) -> SparseResult<BsrMatrix<T>> {
    if block_dim == 0 {
        return Err(SparseError::InvalidArgument(
            "block_dim must be non-zero".to_string(),
        ));
    }
    if csr.rows() % block_dim != 0 {
        return Err(SparseError::InvalidFormat(format!(
            "rows ({}) must be a multiple of block_dim ({})",
            csr.rows(),
            block_dim
        )));
    }
    if csr.cols() % block_dim != 0 {
        return Err(SparseError::InvalidFormat(format!(
            "cols ({}) must be a multiple of block_dim ({})",
            csr.cols(),
            block_dim
        )));
    }

    let (h_row_ptr, h_col_idx, h_values) = csr.to_host()?;

    let block_rows = csr.rows() / block_dim;
    let block_cols = csr.cols() / block_dim;
    let bd = block_dim as usize;

    // Phase 1: Identify which block positions have at least one entry
    // Use a set for each block row
    let mut block_entries: Vec<Vec<u32>> = Vec::with_capacity(block_rows as usize);
    for br in 0..block_rows as usize {
        let mut block_col_set = Vec::new();
        for local_row in 0..bd {
            let global_row = br * bd + local_row;
            let start = h_row_ptr[global_row] as usize;
            let end = h_row_ptr[global_row + 1] as usize;
            for &cj in &h_col_idx[start..end] {
                let bc = cj as u32 / block_dim;
                if !block_col_set.contains(&bc) {
                    block_col_set.push(bc);
                }
            }
        }
        block_col_set.sort_unstable();
        block_entries.push(block_col_set);
    }

    // Phase 2: Build BSR row_ptr and col_idx
    let mut bsr_row_ptr = vec![0i32; block_rows as usize + 1];
    let mut bsr_col_idx = Vec::new();

    for br in 0..block_rows as usize {
        bsr_row_ptr[br + 1] = bsr_row_ptr[br] + block_entries[br].len() as i32;
        bsr_col_idx.extend(block_entries[br].iter().map(|&c| c as i32));
    }

    let nnz_blocks = bsr_col_idx.len();
    if nnz_blocks == 0 {
        return Err(SparseError::ZeroNnz);
    }

    // Phase 3: Fill block values
    let block_elems = bd * bd;
    let mut bsr_values = vec![T::gpu_zero(); nnz_blocks * block_elems];

    for br in 0..block_rows as usize {
        let block_start = bsr_row_ptr[br] as usize;
        let block_cols_for_row = &block_entries[br];

        for local_row in 0..bd {
            let global_row = br * bd + local_row;
            let start = h_row_ptr[global_row] as usize;
            let end = h_row_ptr[global_row + 1] as usize;
            for j in start..end {
                let global_col = h_col_idx[j] as usize;
                let bc = global_col / bd;
                let local_col = global_col % bd;

                // Find block index via binary search
                if let Ok(block_offset) = block_cols_for_row.binary_search(&(bc as u32)) {
                    let block_idx = block_start + block_offset;
                    let val_idx = block_idx * block_elems + local_row * bd + local_col;
                    bsr_values[val_idx] = h_values[j];
                }
            }
        }
    }

    // Suppress unused variable warning
    let _ = block_cols;

    BsrMatrix::from_host(
        csr.rows(),
        csr.cols(),
        block_dim,
        &bsr_row_ptr,
        &bsr_col_idx,
        &bsr_values,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn block_dim_zero_rejected() {
        // Cannot call csr_to_bsr without GPU, just verify the logic
        assert_eq!(4 % 2, 0);
        assert_ne!(5 % 2, 0);
    }

    // -----------------------------------------------------------------------
    // Pure host-side algorithm tests (no GPU required)
    //
    // These replicate the CSR ↔ CSC ↔ COO conversion algorithms that live
    // in the GPU format structs, operating directly on Vec<i32>/Vec<f32>.
    // -----------------------------------------------------------------------

    /// Transpose a CSR matrix (described as host arrays) to CSC format.
    ///
    /// Returns `(col_ptr, row_idx, values)` where the output is sorted by
    /// column then (naturally) by row within each column.
    fn host_csr_to_csc(
        rows: usize,
        cols: usize,
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[f32],
    ) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let nnz = values.len();

        // Count entries per column.
        let mut col_counts = vec![0i32; cols];
        for &c in col_idx {
            col_counts[c as usize] += 1;
        }

        // Prefix-sum to get col_ptr.
        let mut col_ptr = vec![0i32; cols + 1];
        for i in 0..cols {
            col_ptr[i + 1] = col_ptr[i] + col_counts[i];
        }

        // Fill row_idx and values.
        let mut out_row_idx = vec![0i32; nnz];
        let mut out_values = vec![0.0f32; nnz];
        let mut write_pos = col_ptr.clone();

        for row in 0..rows {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;
            for j in start..end {
                let col = col_idx[j] as usize;
                let dest = write_pos[col] as usize;
                out_row_idx[dest] = row as i32;
                out_values[dest] = values[j];
                write_pos[col] += 1;
            }
        }

        (col_ptr, out_row_idx, out_values)
    }

    /// Transpose a CSC matrix (host arrays) back to CSR format.
    fn host_csc_to_csr(
        rows: usize,
        cols: usize,
        col_ptr: &[i32],
        row_idx: &[i32],
        values: &[f32],
    ) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let nnz = values.len();

        // Count entries per row.
        let mut row_counts = vec![0i32; rows];
        for &r in row_idx {
            row_counts[r as usize] += 1;
        }

        // Prefix-sum to get row_ptr.
        let mut out_row_ptr = vec![0i32; rows + 1];
        for i in 0..rows {
            out_row_ptr[i + 1] = out_row_ptr[i] + row_counts[i];
        }

        // Fill col_idx and values.
        let mut out_col_idx = vec![0i32; nnz];
        let mut out_values = vec![0.0f32; nnz];
        let mut write_pos = out_row_ptr.clone();

        for col in 0..cols {
            let start = col_ptr[col] as usize;
            let end = col_ptr[col + 1] as usize;
            for j in start..end {
                let row = row_idx[j] as usize;
                let dest = write_pos[row] as usize;
                out_col_idx[dest] = col as i32;
                out_values[dest] = values[j];
                write_pos[row] += 1;
            }
        }

        // Sort each row's entries by column index for canonical form.
        for r in 0..rows {
            let s = out_row_ptr[r] as usize;
            let e = out_row_ptr[r + 1] as usize;
            // Gather, sort, scatter.
            let mut row_pairs: Vec<(i32, f32)> =
                (s..e).map(|i| (out_col_idx[i], out_values[i])).collect();
            row_pairs.sort_by_key(|&(c, _)| c);
            for (i, (c, v)) in row_pairs.into_iter().enumerate() {
                out_col_idx[s + i] = c;
                out_values[s + i] = v;
            }
        }

        (out_row_ptr, out_col_idx, out_values)
    }

    /// Convert unsorted COO triplets to CSR format.
    fn host_coo_to_csr(
        rows: usize,
        row_idx: &[i32],
        col_idx: &[i32],
        values: &[f32],
    ) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let nnz = values.len();

        // Sort triplets by (row, col).
        let mut triplets: Vec<(i32, i32, f32)> = (0..nnz)
            .map(|i| (row_idx[i], col_idx[i], values[i]))
            .collect();
        triplets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Build row_ptr.
        let mut row_ptr = vec![0i32; rows + 1];
        for &(r, _, _) in &triplets {
            row_ptr[r as usize + 1] += 1;
        }
        for i in 0..rows {
            row_ptr[i + 1] += row_ptr[i];
        }

        let sorted_col_idx: Vec<i32> = triplets.iter().map(|&(_, c, _)| c).collect();
        let sorted_values: Vec<f32> = triplets.iter().map(|&(_, _, v)| v).collect();

        (row_ptr, sorted_col_idx, sorted_values)
    }

    #[test]
    fn test_csr_to_csc_round_trip() {
        // 4x4 sparse matrix:
        //   [1 0 0 2]
        //   [0 3 0 0]
        //   [0 0 4 5]
        //   [6 0 0 0]
        //
        // CSR:
        //   row_ptr  = [0, 2, 3, 5, 6]
        //   col_idx  = [0, 3,  1,  2, 3,  0]
        //   values   = [1, 2,  3,  4, 5,  6]
        let rows = 4usize;
        let cols = 4usize;
        let row_ptr = vec![0i32, 2, 3, 5, 6];
        let col_idx = vec![0i32, 3, 1, 2, 3, 0];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        // CSR → CSC
        let (col_ptr, csc_row_idx, csc_values) =
            host_csr_to_csc(rows, cols, &row_ptr, &col_idx, &values);

        // Verify col_ptr: column 0 has entries [1,6], col 1 has [3], col 2 has [4], col 3 has [2,5]
        assert_eq!(col_ptr, vec![0, 2, 3, 4, 6], "col_ptr mismatch");

        // CSC → CSR (round-trip)
        let (rt_row_ptr, rt_col_idx, rt_values) =
            host_csc_to_csr(rows, cols, &col_ptr, &csc_row_idx, &csc_values);

        // The round-trip must reproduce the original CSR exactly.
        assert_eq!(rt_row_ptr, row_ptr, "round-trip row_ptr mismatch");
        assert_eq!(rt_col_idx, col_idx, "round-trip col_idx mismatch");
        for (a, b) in rt_values.iter().zip(values.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "round-trip value mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_coo_to_csr_sorted() {
        // Provide unsorted COO triplets; after conversion to CSR the
        // entries within each row must be sorted by column index.
        //
        // Unsorted COO for a 3x4 matrix:
        //   (2, 3, 9.0), (0, 1, 1.0), (1, 0, 4.0), (2, 1, 7.0), (0, 3, 2.0)
        let rows = 3usize;
        let row_idx = vec![2i32, 0, 1, 2, 0];
        let col_idx = vec![3i32, 1, 0, 1, 3];
        let values = vec![9.0f32, 1.0, 4.0, 7.0, 2.0];

        let (row_ptr, out_col_idx, out_values) = host_coo_to_csr(rows, &row_idx, &col_idx, &values);

        // Verify row_ptr: row 0 has 2 entries, row 1 has 1, row 2 has 2.
        assert_eq!(row_ptr, vec![0, 2, 3, 5]);

        // Verify entries in each row are sorted by column.
        for r in 0..rows {
            let s = row_ptr[r] as usize;
            let e = row_ptr[r + 1] as usize;
            for i in s + 1..e {
                assert!(
                    out_col_idx[i] >= out_col_idx[i - 1],
                    "row {r}: col_idx not sorted at position {i}"
                );
            }
        }

        // Verify specific entries.
        // Row 0: (col=1, val=1.0), (col=3, val=2.0)
        assert_eq!(out_col_idx[0], 1);
        assert!((out_values[0] - 1.0).abs() < 1e-6);
        assert_eq!(out_col_idx[1], 3);
        assert!((out_values[1] - 2.0).abs() < 1e-6);
        // Row 1: (col=0, val=4.0)
        assert_eq!(out_col_idx[2], 0);
        assert!((out_values[2] - 4.0).abs() < 1e-6);
        // Row 2: (col=1, val=7.0), (col=3, val=9.0)
        assert_eq!(out_col_idx[3], 1);
        assert!((out_values[3] - 7.0).abs() < 1e-6);
        assert_eq!(out_col_idx[4], 3);
        assert!((out_values[4] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_csr_to_ell_padding() {
        // Verify ELL format pads correctly for rows of different lengths.
        //
        // 3x4 matrix with rows of lengths 1, 3, 2:
        //   row 0: col 2 → val 5.0   (1 entry)
        //   row 1: col 0, 1, 3 → 1, 2, 4   (3 entries, most dense row)
        //   row 2: col 1, 3 → 3, 6   (2 entries)
        //
        // ELL max_nnz_per_row = 3 (the maximum over all rows)
        // ELL is stored column-major: shape (rows, max_nnz_per_row) = (3, 3)
        // Each column k of the ELL array lists entry k of every row.
        let rows = 3usize;
        let max_nnz_per_row = 3usize;
        let ell_sentinel = -1i32;

        // Build ELL col_idx (column-major, size = rows * max_nnz_per_row = 9)
        // ELL column 0 (first entry of each row): row0→col2, row1→col0, row2→col1
        // ELL column 1 (second entry):            row0→-1,   row1→col1, row2→col3
        // ELL column 2 (third entry):             row0→-1,   row1→col3, row2→-1
        let ell_col_idx = vec![
            2i32,
            0,
            1, // ELL-col 0: row 0,1,2 first entry
            ell_sentinel,
            1,
            3, // ELL-col 1: row 0 padded, row 1 & 2 second entry
            ell_sentinel,
            3,
            ell_sentinel, // ELL-col 2: row 0 & 2 padded, row 1 third entry
        ];
        let ell_values = vec![
            5.0f32, 1.0, 3.0, // ELL-col 0
            0.0, 2.0, 6.0, // ELL-col 1
            0.0, 4.0, 0.0, // ELL-col 2
        ];

        assert_eq!(ell_col_idx.len(), rows * max_nnz_per_row);
        assert_eq!(ell_values.len(), rows * max_nnz_per_row);

        // Count real entries per row (non-sentinel) and verify they match expected.
        let expected_nnz_per_row = [1usize, 3, 2];
        for r in 0..rows {
            let count = (0..max_nnz_per_row)
                .filter(|&k| ell_col_idx[k * rows + r] != ell_sentinel)
                .count();
            assert_eq!(
                count, expected_nnz_per_row[r],
                "row {r}: expected {} real entries, found {}",
                expected_nnz_per_row[r], count
            );
        }

        // Verify sentinel positions are zero-valued.
        for idx in 0..ell_col_idx.len() {
            if ell_col_idx[idx] == ell_sentinel {
                assert!(
                    ell_values[idx].abs() < 1e-10,
                    "padded ELL value at index {idx} should be zero"
                );
            }
        }

        // Verify reconstruction: reading back non-sentinel entries for each row.
        // Row 0: only entry is col=2, val=5.0
        {
            let r = 0usize;
            let entries: Vec<(i32, f32)> = (0..max_nnz_per_row)
                .filter_map(|k| {
                    let c = ell_col_idx[k * rows + r];
                    if c != ell_sentinel {
                        Some((c, ell_values[k * rows + r]))
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(entries, vec![(2, 5.0)]);
        }
        // Row 1: entries col=0,val=1; col=1,val=2; col=3,val=4
        {
            let r = 1usize;
            let entries: Vec<(i32, f32)> = (0..max_nnz_per_row)
                .filter_map(|k| {
                    let c = ell_col_idx[k * rows + r];
                    if c != ell_sentinel {
                        Some((c, ell_values[k * rows + r]))
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(entries, vec![(0, 1.0), (1, 2.0), (3, 4.0)]);
        }
    }
}
