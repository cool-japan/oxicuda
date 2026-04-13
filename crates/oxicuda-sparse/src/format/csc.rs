//! Compressed Sparse Column (CSC) format.
//!
//! CSC is the column-compressed counterpart of CSR. It stores:
//! - `col_ptr[cols+1]`: indices into `row_idx`/`values` for each column
//! - `row_idx[nnz]`: row index for each non-zero
//! - `values[nnz]`: the non-zero values
//!
//! CSC is efficient for column-oriented operations and is the natural
//! format for the transpose of a CSR matrix.

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::error::{SparseError, SparseResult};

/// A sparse matrix in Compressed Sparse Column (CSC) format, stored on GPU.
///
/// The matrix has shape `(rows, cols)` with `nnz` non-zero elements.
pub struct CscMatrix<T: GpuFloat> {
    /// Number of rows.
    rows: u32,
    /// Number of columns.
    cols: u32,
    /// Number of non-zero elements.
    nnz: u32,
    /// Column pointer array of length `cols + 1`.
    col_ptr: DeviceBuffer<i32>,
    /// Row indices of length `nnz`.
    row_idx: DeviceBuffer<i32>,
    /// Non-zero values of length `nnz`.
    values: DeviceBuffer<T>,
}

impl<T: GpuFloat> CscMatrix<T> {
    /// Creates a CSC matrix from host-side arrays, uploading to GPU.
    ///
    /// # Arguments
    ///
    /// * `rows` -- Number of rows.
    /// * `cols` -- Number of columns.
    /// * `col_ptr` -- Column pointer array of length `cols + 1`.
    /// * `row_idx` -- Row indices of length `nnz`.
    /// * `values` -- Non-zero values of length `nnz`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidFormat`] if array lengths are inconsistent.
    pub fn from_host(
        rows: u32,
        cols: u32,
        col_ptr: &[i32],
        row_idx: &[i32],
        values: &[T],
    ) -> SparseResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(SparseError::InvalidFormat(
                "rows and cols must be non-zero".to_string(),
            ));
        }

        let expected_col_ptr_len = cols as usize + 1;
        if col_ptr.len() != expected_col_ptr_len {
            return Err(SparseError::InvalidFormat(format!(
                "col_ptr length ({}) must be cols + 1 ({})",
                col_ptr.len(),
                expected_col_ptr_len
            )));
        }

        let nnz = values.len();
        if nnz == 0 {
            return Err(SparseError::ZeroNnz);
        }
        if row_idx.len() != nnz {
            return Err(SparseError::InvalidFormat(format!(
                "row_idx length ({}) must equal values length ({})",
                row_idx.len(),
                nnz
            )));
        }

        if col_ptr[0] != 0 {
            return Err(SparseError::InvalidFormat(
                "col_ptr[0] must be 0".to_string(),
            ));
        }
        if col_ptr[cols as usize] != nnz as i32 {
            return Err(SparseError::InvalidFormat(format!(
                "col_ptr[cols] ({}) must equal nnz ({})",
                col_ptr[cols as usize], nnz
            )));
        }
        for i in 0..cols as usize {
            if col_ptr[i] > col_ptr[i + 1] {
                return Err(SparseError::InvalidFormat(format!(
                    "col_ptr must be non-decreasing: col_ptr[{}]={} > col_ptr[{}]={}",
                    i,
                    col_ptr[i],
                    i + 1,
                    col_ptr[i + 1]
                )));
            }
        }

        let d_col_ptr = DeviceBuffer::from_host(col_ptr)?;
        let d_row_idx = DeviceBuffer::from_host(row_idx)?;
        let d_values = DeviceBuffer::from_host(values)?;

        Ok(Self {
            rows,
            cols,
            nnz: nnz as u32,
            col_ptr: d_col_ptr,
            row_idx: d_row_idx,
            values: d_values,
        })
    }

    /// Creates a CSC matrix from pre-allocated device buffers.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidFormat`] if buffer lengths are inconsistent.
    pub fn from_device(
        rows: u32,
        cols: u32,
        nnz: u32,
        col_ptr: DeviceBuffer<i32>,
        row_idx: DeviceBuffer<i32>,
        values: DeviceBuffer<T>,
    ) -> SparseResult<Self> {
        if col_ptr.len() != (cols as usize + 1) {
            return Err(SparseError::InvalidFormat(format!(
                "col_ptr length ({}) must be cols + 1 ({})",
                col_ptr.len(),
                cols as usize + 1
            )));
        }
        if row_idx.len() != nnz as usize || values.len() != nnz as usize {
            return Err(SparseError::InvalidFormat(
                "row_idx and values lengths must equal nnz".to_string(),
            ));
        }
        Ok(Self {
            rows,
            cols,
            nnz,
            col_ptr,
            row_idx,
            values,
        })
    }

    /// Downloads the CSC arrays from GPU to host memory.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on transfer failure.
    pub fn to_host(&self) -> SparseResult<(Vec<i32>, Vec<i32>, Vec<T>)> {
        let mut h_col_ptr = vec![0i32; self.col_ptr.len()];
        let mut h_row_idx = vec![0i32; self.row_idx.len()];
        let mut h_values = vec![T::gpu_zero(); self.values.len()];

        self.col_ptr.copy_to_host(&mut h_col_ptr)?;
        self.row_idx.copy_to_host(&mut h_row_idx)?;
        self.values.copy_to_host(&mut h_values)?;

        Ok((h_col_ptr, h_row_idx, h_values))
    }

    /// Converts this CSC matrix to CSR format on the host.
    ///
    /// Downloads data, transposes the structure (CSC -> CSR is analogous
    /// to transposing the matrix), then uploads.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on transfer failure.
    pub fn to_csr(&self) -> SparseResult<super::CsrMatrix<T>> {
        let (h_col_ptr, h_row_idx, h_values) = self.to_host()?;

        // CSC -> CSR is the same as transposing the CSR->CSC algorithm
        let mut row_counts = vec![0i32; self.rows as usize];
        for &r in &h_row_idx {
            row_counts[r as usize] += 1;
        }

        let mut h_row_ptr = vec![0i32; self.rows as usize + 1];
        for i in 0..self.rows as usize {
            h_row_ptr[i + 1] = h_row_ptr[i] + row_counts[i];
        }

        let mut h_csr_col_idx = vec![0i32; self.nnz as usize];
        let mut h_csr_values = vec![T::gpu_zero(); self.nnz as usize];
        let mut write_pos = h_row_ptr.clone();

        for col in 0..self.cols as usize {
            let start = h_col_ptr[col] as usize;
            let end = h_col_ptr[col + 1] as usize;
            for j in start..end {
                let row = h_row_idx[j] as usize;
                let dest = write_pos[row] as usize;
                h_csr_col_idx[dest] = col as i32;
                h_csr_values[dest] = h_values[j];
                write_pos[row] += 1;
            }
        }

        super::CsrMatrix::from_host(
            self.rows,
            self.cols,
            &h_row_ptr,
            &h_csr_col_idx,
            &h_csr_values,
        )
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

    /// Returns a reference to the column pointer device buffer.
    #[inline]
    pub fn col_ptr(&self) -> &DeviceBuffer<i32> {
        &self.col_ptr
    }

    /// Returns a reference to the row index device buffer.
    #[inline]
    pub fn row_idx(&self) -> &DeviceBuffer<i32> {
        &self.row_idx
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
    fn csc_validation_col_ptr_length() {
        let result = CscMatrix::<f32>::from_host(3, 3, &[0, 2, 4], &[0, 1, 0, 2], &[1.0; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn csc_validation_zero_nnz() {
        let result = CscMatrix::<f32>::from_host(2, 2, &[0, 0, 0], &[], &[]);
        assert!(matches!(result, Err(SparseError::ZeroNnz)));
    }
}
