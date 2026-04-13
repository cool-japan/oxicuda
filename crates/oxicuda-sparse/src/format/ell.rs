//! ELLPACK (ELL) sparse matrix format.
//!
//! The ELLPACK format stores at most `max_nnz_per_row` entries per row,
//! using a padded column index array and a padded values array, each of
//! shape `(rows, max_nnz_per_row)` stored in column-major order.
//!
//! Unused entries are padded with a sentinel column index of -1 and a
//! zero value. This format is highly efficient for matrices with regular
//! sparsity patterns (similar nnz per row) since it avoids indirect
//! indexing and enables coalesced memory access on GPUs.

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::error::{SparseError, SparseResult};

/// Sentinel value for unused ELLPACK entries.
pub const ELL_SENTINEL: i32 = -1;

/// A sparse matrix in ELLPACK (ELL) format, stored on GPU.
///
/// Both `col_idx` and `values` have length `rows * max_nnz_per_row`,
/// stored in column-major order: element `(i, k)` is at index
/// `k * rows + i`.
pub struct EllMatrix<T: GpuFloat> {
    /// Number of rows.
    rows: u32,
    /// Number of columns.
    cols: u32,
    /// Maximum non-zeros per row (determines the padding width).
    max_nnz_per_row: u32,
    /// Column indices: length `rows * max_nnz_per_row`. Unused entries are -1.
    col_idx: DeviceBuffer<i32>,
    /// Values: length `rows * max_nnz_per_row`. Unused entries are zero.
    values: DeviceBuffer<T>,
}

impl<T: GpuFloat> EllMatrix<T> {
    /// Creates an ELL matrix from host-side padded arrays, uploading to GPU.
    ///
    /// # Arguments
    ///
    /// * `rows` -- Number of rows.
    /// * `cols` -- Number of columns.
    /// * `max_nnz_per_row` -- Maximum entries stored per row.
    /// * `col_idx` -- Padded column indices, length `rows * max_nnz_per_row`,
    ///   column-major. Unused entries must be `ELL_SENTINEL` (-1).
    /// * `values` -- Padded values, length `rows * max_nnz_per_row`,
    ///   column-major. Unused entries should be zero.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidFormat`] if array lengths are incorrect.
    pub fn from_host(
        rows: u32,
        cols: u32,
        max_nnz_per_row: u32,
        col_idx: &[i32],
        values: &[T],
    ) -> SparseResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(SparseError::InvalidFormat(
                "rows and cols must be non-zero".to_string(),
            ));
        }
        if max_nnz_per_row == 0 {
            return Err(SparseError::ZeroNnz);
        }

        let total = rows as usize * max_nnz_per_row as usize;
        if col_idx.len() != total {
            return Err(SparseError::InvalidFormat(format!(
                "col_idx length ({}) must be rows * max_nnz_per_row ({})",
                col_idx.len(),
                total
            )));
        }
        if values.len() != total {
            return Err(SparseError::InvalidFormat(format!(
                "values length ({}) must be rows * max_nnz_per_row ({})",
                values.len(),
                total
            )));
        }

        let d_col_idx = DeviceBuffer::from_host(col_idx)?;
        let d_values = DeviceBuffer::from_host(values)?;

        Ok(Self {
            rows,
            cols,
            max_nnz_per_row,
            col_idx: d_col_idx,
            values: d_values,
        })
    }

    /// Creates an ELL matrix from a CSR matrix on the host.
    ///
    /// Determines `max_nnz_per_row` from the CSR structure, then pads
    /// each row to that width.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on transfer failure.
    pub fn from_csr(csr: &super::CsrMatrix<T>) -> SparseResult<Self> {
        let (h_row_ptr, h_col_idx, h_values) = csr.to_host()?;
        let rows = csr.rows();
        let cols = csr.cols();

        // Find max nnz per row
        let mut max_nnz: u32 = 0;
        for i in 0..rows as usize {
            let row_nnz = (h_row_ptr[i + 1] - h_row_ptr[i]) as u32;
            if row_nnz > max_nnz {
                max_nnz = row_nnz;
            }
        }

        if max_nnz == 0 {
            return Err(SparseError::ZeroNnz);
        }

        // Build padded ELL arrays (column-major: element (i, k) at k * rows + i)
        let total = rows as usize * max_nnz as usize;
        let mut ell_col_idx = vec![ELL_SENTINEL; total];
        let mut ell_values = vec![T::gpu_zero(); total];

        for i in 0..rows as usize {
            let start = h_row_ptr[i] as usize;
            let end = h_row_ptr[i + 1] as usize;
            for (k, j) in (start..end).enumerate() {
                let idx = k * rows as usize + i;
                ell_col_idx[idx] = h_col_idx[j];
                ell_values[idx] = h_values[j];
            }
        }

        Self::from_host(rows, cols, max_nnz, &ell_col_idx, &ell_values)
    }

    /// Downloads the ELL arrays from GPU to host memory.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on transfer failure.
    pub fn to_host(&self) -> SparseResult<(Vec<i32>, Vec<T>)> {
        let mut h_col_idx = vec![0i32; self.col_idx.len()];
        let mut h_values = vec![T::gpu_zero(); self.values.len()];

        self.col_idx.copy_to_host(&mut h_col_idx)?;
        self.values.copy_to_host(&mut h_values)?;

        Ok((h_col_idx, h_values))
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

    /// Returns the maximum non-zeros per row.
    #[inline]
    pub fn max_nnz_per_row(&self) -> u32 {
        self.max_nnz_per_row
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
    fn ell_validation_array_lengths() {
        // 3 rows, max 2 per row => total 6 entries
        let result = EllMatrix::<f32>::from_host(
            3,
            3,
            2,
            &[0, 1, 2, -1, -1], // length 5, should be 6
            &[1.0; 5],
        );
        assert!(result.is_err());
    }

    #[test]
    fn ell_sentinel_value() {
        assert_eq!(ELL_SENTINEL, -1);
    }
}
