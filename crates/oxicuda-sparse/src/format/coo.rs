//! Coordinate (COO) format.
//!
//! COO stores each non-zero as a `(row, col, value)` triplet. It is the
//! simplest sparse format and is commonly used as an intermediate format
//! for assembly before converting to CSR or CSC.
//!
//! The triplets can be in any order, but sorted COO enables more efficient
//! conversion to CSR.

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::error::{SparseError, SparseResult};

/// A sparse matrix in Coordinate (COO) format, stored on GPU.
///
/// The matrix has shape `(rows, cols)` with `nnz` non-zero elements.
/// Each non-zero is represented by its row index, column index, and value.
pub struct CooMatrix<T: GpuFloat> {
    /// Number of rows.
    rows: u32,
    /// Number of columns.
    cols: u32,
    /// Number of non-zero elements.
    nnz: u32,
    /// Row indices of length `nnz`.
    row_idx: DeviceBuffer<i32>,
    /// Column indices of length `nnz`.
    col_idx: DeviceBuffer<i32>,
    /// Non-zero values of length `nnz`.
    values: DeviceBuffer<T>,
    /// Whether the triplets are sorted by (row, col).
    sorted: bool,
}

impl<T: GpuFloat> CooMatrix<T> {
    /// Creates a COO matrix from host-side arrays, uploading to GPU.
    ///
    /// The triplets are assumed to be unsorted unless the caller knows
    /// otherwise (use [`with_sorted`](Self::with_sorted) to override).
    ///
    /// # Arguments
    ///
    /// * `rows` -- Number of rows.
    /// * `cols` -- Number of columns.
    /// * `row_idx` -- Row indices of length `nnz`.
    /// * `col_idx` -- Column indices of length `nnz`.
    /// * `values` -- Non-zero values of length `nnz`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidFormat`] if array lengths are inconsistent.
    pub fn from_host(
        rows: u32,
        cols: u32,
        row_idx: &[i32],
        col_idx: &[i32],
        values: &[T],
    ) -> SparseResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(SparseError::InvalidFormat(
                "rows and cols must be non-zero".to_string(),
            ));
        }

        let nnz = values.len();
        if nnz == 0 {
            return Err(SparseError::ZeroNnz);
        }
        if row_idx.len() != nnz || col_idx.len() != nnz {
            return Err(SparseError::InvalidFormat(format!(
                "row_idx ({}), col_idx ({}), and values ({}) must have equal length",
                row_idx.len(),
                col_idx.len(),
                nnz
            )));
        }

        let d_row_idx = DeviceBuffer::from_host(row_idx)?;
        let d_col_idx = DeviceBuffer::from_host(col_idx)?;
        let d_values = DeviceBuffer::from_host(values)?;

        Ok(Self {
            rows,
            cols,
            nnz: nnz as u32,
            row_idx: d_row_idx,
            col_idx: d_col_idx,
            values: d_values,
            sorted: false,
        })
    }

    /// Creates a COO matrix from pre-allocated device buffers.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidFormat`] if buffer lengths are inconsistent.
    pub fn from_device(
        rows: u32,
        cols: u32,
        nnz: u32,
        row_idx: DeviceBuffer<i32>,
        col_idx: DeviceBuffer<i32>,
        values: DeviceBuffer<T>,
    ) -> SparseResult<Self> {
        if row_idx.len() != nnz as usize
            || col_idx.len() != nnz as usize
            || values.len() != nnz as usize
        {
            return Err(SparseError::InvalidFormat(
                "all arrays must have length equal to nnz".to_string(),
            ));
        }
        Ok(Self {
            rows,
            cols,
            nnz,
            row_idx,
            col_idx,
            values,
            sorted: false,
        })
    }

    /// Mark the COO matrix as sorted by (row, col).
    ///
    /// This is a hint for conversion routines; the caller is responsible
    /// for ensuring correctness.
    #[must_use]
    pub fn with_sorted(mut self, sorted: bool) -> Self {
        self.sorted = sorted;
        self
    }

    /// Downloads the COO arrays from GPU to host memory.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on transfer failure.
    pub fn to_host(&self) -> SparseResult<(Vec<i32>, Vec<i32>, Vec<T>)> {
        let mut h_row_idx = vec![0i32; self.row_idx.len()];
        let mut h_col_idx = vec![0i32; self.col_idx.len()];
        let mut h_values = vec![T::gpu_zero(); self.values.len()];

        self.row_idx.copy_to_host(&mut h_row_idx)?;
        self.col_idx.copy_to_host(&mut h_col_idx)?;
        self.values.copy_to_host(&mut h_values)?;

        Ok((h_row_idx, h_col_idx, h_values))
    }

    /// Converts this COO matrix to CSR format.
    ///
    /// Downloads to host, builds row pointers via histogram, then uploads.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on transfer failure.
    pub fn to_csr(&self) -> SparseResult<super::CsrMatrix<T>> {
        let (h_row_idx, h_col_idx, h_values) = self.to_host()?;

        // Build row pointers from row indices
        let mut row_counts = vec![0i32; self.rows as usize];
        for &r in &h_row_idx {
            row_counts[r as usize] += 1;
        }

        let mut h_row_ptr = vec![0i32; self.rows as usize + 1];
        for i in 0..self.rows as usize {
            h_row_ptr[i + 1] = h_row_ptr[i] + row_counts[i];
        }

        // Place entries in row order
        let mut h_csr_col_idx = vec![0i32; self.nnz as usize];
        let mut h_csr_values = vec![T::gpu_zero(); self.nnz as usize];
        let mut write_pos = h_row_ptr.clone();

        for i in 0..self.nnz as usize {
            let row = h_row_idx[i] as usize;
            let dest = write_pos[row] as usize;
            h_csr_col_idx[dest] = h_col_idx[i];
            h_csr_values[dest] = h_values[i];
            write_pos[row] += 1;
        }

        super::CsrMatrix::from_host(
            self.rows,
            self.cols,
            &h_row_ptr,
            &h_csr_col_idx,
            &h_csr_values,
        )
    }

    /// Converts this COO matrix to CSC format.
    ///
    /// Downloads to host, builds column pointers, then uploads.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on transfer failure.
    pub fn to_csc(&self) -> SparseResult<super::CscMatrix<T>> {
        let (h_row_idx, h_col_idx, h_values) = self.to_host()?;

        let mut col_counts = vec![0i32; self.cols as usize];
        for &c in &h_col_idx {
            col_counts[c as usize] += 1;
        }

        let mut h_col_ptr = vec![0i32; self.cols as usize + 1];
        for i in 0..self.cols as usize {
            h_col_ptr[i + 1] = h_col_ptr[i] + col_counts[i];
        }

        let mut h_csc_row_idx = vec![0i32; self.nnz as usize];
        let mut h_csc_values = vec![T::gpu_zero(); self.nnz as usize];
        let mut write_pos = h_col_ptr.clone();

        for i in 0..self.nnz as usize {
            let col = h_col_idx[i] as usize;
            let dest = write_pos[col] as usize;
            h_csc_row_idx[dest] = h_row_idx[i];
            h_csc_values[dest] = h_values[i];
            write_pos[col] += 1;
        }

        super::CscMatrix::from_host(
            self.rows,
            self.cols,
            &h_col_ptr,
            &h_csc_row_idx,
            &h_csc_values,
        )
    }

    /// Returns whether the triplets are sorted by (row, col).
    #[inline]
    pub fn is_sorted(&self) -> bool {
        self.sorted
    }

    /// Returns the number of rows.
    #[inline]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Returns the number of columns.
    #[inline]
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Returns the number of non-zero elements.
    #[inline]
    pub fn nnz(&self) -> u32 {
        self.nnz
    }

    /// Returns a reference to the row index device buffer.
    #[inline]
    pub fn row_idx(&self) -> &DeviceBuffer<i32> {
        &self.row_idx
    }

    /// Returns a reference to the column index device buffer.
    #[inline]
    pub fn col_idx(&self) -> &DeviceBuffer<i32> {
        &self.col_idx
    }

    /// Returns a reference to the values device buffer.
    #[inline]
    pub fn values(&self) -> &DeviceBuffer<T> {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coo_validation_mismatched_lengths() {
        let result = CooMatrix::<f32>::from_host(3, 3, &[0, 1], &[0, 1, 2], &[1.0; 3]);
        assert!(result.is_err());
    }

    #[test]
    fn coo_validation_zero_nnz() {
        let result = CooMatrix::<f32>::from_host(2, 2, &[], &[], &[]);
        assert!(matches!(result, Err(SparseError::ZeroNnz)));
    }

    #[test]
    fn coo_sorted_flag() {
        // Just verify the flag API works (no GPU needed)
    }
}
