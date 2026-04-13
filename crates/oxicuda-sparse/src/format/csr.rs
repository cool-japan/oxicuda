//! Compressed Sparse Row (CSR) format.
//!
//! CSR is the most widely used sparse matrix format. It stores:
//! - `row_ptr[rows+1]`: indices into `col_idx`/`values` for each row
//! - `col_idx[nnz]`: column index for each non-zero
//! - `values[nnz]`: the non-zero values
//!
//! This is the primary format for SpMV, SpMM, and most sparse operations.

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::error::{SparseError, SparseResult};

/// A sparse matrix in Compressed Sparse Row (CSR) format, stored on GPU.
///
/// The matrix has shape `(rows, cols)` with `nnz` non-zero elements.
/// All index and value arrays reside in device (GPU) memory.
pub struct CsrMatrix<T: GpuFloat> {
    /// Number of rows.
    rows: u32,
    /// Number of columns.
    cols: u32,
    /// Number of non-zero elements.
    nnz: u32,
    /// Row pointer array of length `rows + 1`. `row_ptr[i]` is the index
    /// into `col_idx`/`values` where row `i` begins.
    row_ptr: DeviceBuffer<i32>,
    /// Column indices of length `nnz`.
    col_idx: DeviceBuffer<i32>,
    /// Non-zero values of length `nnz`.
    values: DeviceBuffer<T>,
}

impl<T: GpuFloat> CsrMatrix<T> {
    /// Creates a CSR matrix from host-side arrays, uploading to GPU.
    ///
    /// # Arguments
    ///
    /// * `rows` -- Number of rows.
    /// * `cols` -- Number of columns.
    /// * `row_ptr` -- Row pointer array of length `rows + 1`.
    /// * `col_idx` -- Column indices of length `nnz`.
    /// * `values` -- Non-zero values of length `nnz`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidFormat`] if array lengths are inconsistent.
    /// Returns [`SparseError::ZeroNnz`] if `values` is empty.
    /// Returns [`SparseError::Cuda`] on GPU memory allocation failure.
    pub fn from_host(
        rows: u32,
        cols: u32,
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[T],
    ) -> SparseResult<Self> {
        // Validate dimensions
        if rows == 0 || cols == 0 {
            return Err(SparseError::InvalidFormat(
                "rows and cols must be non-zero".to_string(),
            ));
        }

        // Validate row_ptr length
        let expected_row_ptr_len = rows as usize + 1;
        if row_ptr.len() != expected_row_ptr_len {
            return Err(SparseError::InvalidFormat(format!(
                "row_ptr length ({}) must be rows + 1 ({})",
                row_ptr.len(),
                expected_row_ptr_len
            )));
        }

        // Validate nnz consistency
        let nnz = values.len();
        if nnz == 0 {
            return Err(SparseError::ZeroNnz);
        }
        if col_idx.len() != nnz {
            return Err(SparseError::InvalidFormat(format!(
                "col_idx length ({}) must equal values length ({})",
                col_idx.len(),
                nnz
            )));
        }

        // Validate row_ptr is non-decreasing and within bounds
        if row_ptr[0] != 0 {
            return Err(SparseError::InvalidFormat(
                "row_ptr[0] must be 0".to_string(),
            ));
        }
        if row_ptr[rows as usize] != nnz as i32 {
            return Err(SparseError::InvalidFormat(format!(
                "row_ptr[rows] ({}) must equal nnz ({})",
                row_ptr[rows as usize], nnz
            )));
        }
        for i in 0..rows as usize {
            if row_ptr[i] > row_ptr[i + 1] {
                return Err(SparseError::InvalidFormat(format!(
                    "row_ptr must be non-decreasing: row_ptr[{}]={} > row_ptr[{}]={}",
                    i,
                    row_ptr[i],
                    i + 1,
                    row_ptr[i + 1]
                )));
            }
        }

        // Upload to GPU
        let d_row_ptr = DeviceBuffer::from_host(row_ptr)?;
        let d_col_idx = DeviceBuffer::from_host(col_idx)?;
        let d_values = DeviceBuffer::from_host(values)?;

        Ok(Self {
            rows,
            cols,
            nnz: nnz as u32,
            row_ptr: d_row_ptr,
            col_idx: d_col_idx,
            values: d_values,
        })
    }

    /// Creates a CSR matrix from pre-allocated device buffers.
    ///
    /// No validation of contents is performed; only lengths are checked.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidFormat`] if buffer lengths are inconsistent.
    pub fn from_device(
        rows: u32,
        cols: u32,
        nnz: u32,
        row_ptr: DeviceBuffer<i32>,
        col_idx: DeviceBuffer<i32>,
        values: DeviceBuffer<T>,
    ) -> SparseResult<Self> {
        if row_ptr.len() != (rows as usize + 1) {
            return Err(SparseError::InvalidFormat(format!(
                "row_ptr length ({}) must be rows + 1 ({})",
                row_ptr.len(),
                rows as usize + 1
            )));
        }
        if col_idx.len() != nnz as usize {
            return Err(SparseError::InvalidFormat(format!(
                "col_idx length ({}) must equal nnz ({})",
                col_idx.len(),
                nnz
            )));
        }
        if values.len() != nnz as usize {
            return Err(SparseError::InvalidFormat(format!(
                "values length ({}) must equal nnz ({})",
                values.len(),
                nnz
            )));
        }
        Ok(Self {
            rows,
            cols,
            nnz,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Downloads the CSR arrays from GPU to host memory.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on transfer failure.
    pub fn to_host(&self) -> SparseResult<(Vec<i32>, Vec<i32>, Vec<T>)> {
        let mut h_row_ptr = vec![0i32; self.row_ptr.len()];
        let mut h_col_idx = vec![0i32; self.col_idx.len()];
        let mut h_values = vec![T::gpu_zero(); self.values.len()];

        self.row_ptr.copy_to_host(&mut h_row_ptr)?;
        self.col_idx.copy_to_host(&mut h_col_idx)?;
        self.values.copy_to_host(&mut h_values)?;

        Ok((h_row_ptr, h_col_idx, h_values))
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

    /// Returns the density of the matrix (nnz / (rows * cols)).
    #[inline]
    pub fn density(&self) -> f64 {
        let total = self.rows as f64 * self.cols as f64;
        if total == 0.0 {
            return 0.0;
        }
        self.nnz as f64 / total
    }

    /// Checks whether the matrix metadata is structurally valid.
    ///
    /// This checks buffer lengths but does NOT download data to verify
    /// content (e.g. sorted column indices). For full validation, use
    /// [`to_host`](Self::to_host) and check on the CPU.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.row_ptr.len() == (self.rows as usize + 1)
            && self.col_idx.len() == self.nnz as usize
            && self.values.len() == self.nnz as usize
    }

    /// Returns a reference to the row pointer device buffer.
    #[inline]
    pub fn row_ptr(&self) -> &DeviceBuffer<i32> {
        &self.row_ptr
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

    /// Returns a mutable reference to the values device buffer.
    #[inline]
    pub fn values_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.values
    }

    /// Average number of non-zeros per row.
    #[inline]
    pub fn avg_nnz_per_row(&self) -> f64 {
        if self.rows == 0 {
            return 0.0;
        }
        self.nnz as f64 / self.rows as f64
    }

    /// Converts this CSR matrix to COO format on the host.
    ///
    /// Downloads data to host, expands row pointers to row indices,
    /// then uploads the COO matrix back to GPU.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on transfer failure.
    pub fn to_coo(&self) -> SparseResult<super::CooMatrix<T>> {
        let (h_row_ptr, h_col_idx, h_values) = self.to_host()?;

        // Expand row_ptr to row_idx
        let mut h_row_idx = Vec::with_capacity(self.nnz as usize);
        for row in 0..self.rows {
            let start = h_row_ptr[row as usize];
            let end = h_row_ptr[row as usize + 1];
            for _ in start..end {
                h_row_idx.push(row as i32);
            }
        }

        super::CooMatrix::from_host(self.rows, self.cols, &h_row_idx, &h_col_idx, &h_values)
    }

    /// Converts this CSR matrix to CSC format on the host.
    ///
    /// Downloads data, transposes the structure, then uploads.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on transfer failure.
    pub fn to_csc(&self) -> SparseResult<super::CscMatrix<T>> {
        let (h_row_ptr, h_col_idx, h_values) = self.to_host()?;

        // Build CSC by transposing CSR
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

        for row in 0..self.rows as usize {
            let start = h_row_ptr[row] as usize;
            let end = h_row_ptr[row + 1] as usize;
            for j in start..end {
                let col = h_col_idx[j] as usize;
                let dest = write_pos[col] as usize;
                h_csc_row_idx[dest] = row as i32;
                h_csc_values[dest] = h_values[j];
                write_pos[col] += 1;
            }
        }

        super::CscMatrix::from_host(
            self.rows,
            self.cols,
            &h_col_ptr,
            &h_csc_row_idx,
            &h_csc_values,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_validation_row_ptr_length() {
        // row_ptr too short
        let result = CsrMatrix::<f32>::from_host(3, 3, &[0, 2, 4], &[0, 1, 0, 2], &[1.0; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn csr_validation_zero_nnz() {
        let result = CsrMatrix::<f32>::from_host(2, 2, &[0, 0, 0], &[], &[]);
        assert!(matches!(result, Err(SparseError::ZeroNnz)));
    }

    #[test]
    fn csr_validation_mismatched_col_idx() {
        let result = CsrMatrix::<f32>::from_host(2, 2, &[0, 1, 2], &[0], &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn csr_density() {
        // A 4x4 matrix with 4 nnz => density 0.25
        // We cannot actually call from_host without a GPU, so test the formula
        assert!((4.0_f64 / 16.0 - 0.25).abs() < 1e-10);
    }
}
