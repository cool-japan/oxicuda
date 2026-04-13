//! Block Sparse Row (BSR) format.
//!
//! BSR stores sparse matrices as a collection of dense blocks of size
//! `block_dim x block_dim`. This is efficient for matrices with structured
//! sparsity patterns, such as those arising from finite-element
//! discretizations.
//!
//! The storage is organized as:
//! - `row_ptr[block_rows+1]`: indices into `col_idx`/`values` for each block row
//! - `col_idx[nnz_blocks]`: block column index for each non-zero block
//! - `values[nnz_blocks * block_dim * block_dim]`: dense block data (row-major)

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::error::{SparseError, SparseResult};

/// A sparse matrix in Block Sparse Row (BSR) format, stored on GPU.
///
/// The matrix has shape `(rows, cols)` where `rows` and `cols` are multiples
/// of `block_dim`. It contains `nnz_blocks` non-zero dense blocks.
pub struct BsrMatrix<T: GpuFloat> {
    /// Number of rows (must be a multiple of `block_dim`).
    rows: u32,
    /// Number of columns (must be a multiple of `block_dim`).
    cols: u32,
    /// Number of non-zero blocks.
    nnz_blocks: u32,
    /// Size of each dense block (block_dim x block_dim).
    block_dim: u32,
    /// Block row pointer array of length `block_rows + 1`.
    row_ptr: DeviceBuffer<i32>,
    /// Block column indices of length `nnz_blocks`.
    col_idx: DeviceBuffer<i32>,
    /// Dense block values of length `nnz_blocks * block_dim * block_dim`.
    values: DeviceBuffer<T>,
}

impl<T: GpuFloat> BsrMatrix<T> {
    /// Creates a BSR matrix from host-side arrays, uploading to GPU.
    ///
    /// # Arguments
    ///
    /// * `rows` -- Number of rows (must be a multiple of `block_dim`).
    /// * `cols` -- Number of columns (must be a multiple of `block_dim`).
    /// * `block_dim` -- Size of each dense block.
    /// * `row_ptr` -- Block row pointer array of length `block_rows + 1`.
    /// * `col_idx` -- Block column indices of length `nnz_blocks`.
    /// * `values` -- Dense block values (row-major within each block).
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidFormat`] if dimensions or array lengths
    /// are inconsistent.
    #[allow(clippy::too_many_arguments)]
    pub fn from_host(
        rows: u32,
        cols: u32,
        block_dim: u32,
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[T],
    ) -> SparseResult<Self> {
        if rows == 0 || cols == 0 || block_dim == 0 {
            return Err(SparseError::InvalidFormat(
                "rows, cols, and block_dim must be non-zero".to_string(),
            ));
        }
        if rows % block_dim != 0 {
            return Err(SparseError::InvalidFormat(format!(
                "rows ({rows}) must be a multiple of block_dim ({block_dim})"
            )));
        }
        if cols % block_dim != 0 {
            return Err(SparseError::InvalidFormat(format!(
                "cols ({cols}) must be a multiple of block_dim ({block_dim})"
            )));
        }

        let block_rows = rows / block_dim;
        let expected_row_ptr_len = block_rows as usize + 1;
        if row_ptr.len() != expected_row_ptr_len {
            return Err(SparseError::InvalidFormat(format!(
                "row_ptr length ({}) must be block_rows + 1 ({})",
                row_ptr.len(),
                expected_row_ptr_len
            )));
        }

        let nnz_blocks = col_idx.len() as u32;
        if nnz_blocks == 0 {
            return Err(SparseError::ZeroNnz);
        }

        let block_elems = block_dim as usize * block_dim as usize;
        let expected_values_len = nnz_blocks as usize * block_elems;
        if values.len() != expected_values_len {
            return Err(SparseError::InvalidFormat(format!(
                "values length ({}) must be nnz_blocks * block_dim^2 ({})",
                values.len(),
                expected_values_len
            )));
        }

        // Validate row_ptr
        if row_ptr[0] != 0 {
            return Err(SparseError::InvalidFormat(
                "row_ptr[0] must be 0".to_string(),
            ));
        }
        if row_ptr[block_rows as usize] != nnz_blocks as i32 {
            return Err(SparseError::InvalidFormat(format!(
                "row_ptr[block_rows] ({}) must equal nnz_blocks ({})",
                row_ptr[block_rows as usize], nnz_blocks
            )));
        }

        let d_row_ptr = DeviceBuffer::from_host(row_ptr)?;
        let d_col_idx = DeviceBuffer::from_host(col_idx)?;
        let d_values = DeviceBuffer::from_host(values)?;

        Ok(Self {
            rows,
            cols,
            nnz_blocks,
            block_dim,
            row_ptr: d_row_ptr,
            col_idx: d_col_idx,
            values: d_values,
        })
    }

    /// Downloads the BSR arrays from GPU to host memory.
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

    /// Returns the number of non-zero blocks.
    #[inline]
    pub fn nnz_blocks(&self) -> u32 {
        self.nnz_blocks
    }

    /// Returns the block dimension.
    #[inline]
    pub fn block_dim(&self) -> u32 {
        self.block_dim
    }

    /// Returns the number of block rows.
    #[inline]
    pub fn block_rows(&self) -> u32 {
        self.rows / self.block_dim
    }

    /// Returns the number of block columns.
    #[inline]
    pub fn block_cols(&self) -> u32 {
        self.cols / self.block_dim
    }

    /// Returns the total scalar non-zero count.
    #[inline]
    pub fn scalar_nnz(&self) -> u32 {
        self.nnz_blocks * self.block_dim * self.block_dim
    }

    /// Returns a reference to the block row pointer device buffer.
    #[inline]
    pub fn row_ptr(&self) -> &DeviceBuffer<i32> {
        &self.row_ptr
    }

    /// Returns a reference to the block column index device buffer.
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
    fn bsr_validation_block_alignment() {
        // rows not a multiple of block_dim
        let result = BsrMatrix::<f32>::from_host(
            5,
            4,
            2,
            &[0, 1, 2, 3], // would be wrong length too
            &[0, 1, 0],
            &[1.0; 12],
        );
        assert!(result.is_err());
    }

    #[test]
    fn bsr_block_counts() {
        // 4x4 matrix with 2x2 blocks => 2 block rows, 2 block cols
        let br = 4 / 2;
        let bc = 4 / 2;
        assert_eq!(br, 2);
        assert_eq!(bc, 2);
    }
}
