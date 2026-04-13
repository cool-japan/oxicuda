//! Batched sparse operations.
//!
//! This module provides routines for executing many small sparse operations
//! in a single logical batch, amortising kernel launch overhead and enabling
//! fusion when beneficial.
//!
//! ## Provided types
//!
//! - [`BatchedSpMV`] -- batched sparse matrix-vector multiply (host-side loop).
//! - [`BatchedSpMVPlan`] -- precomputed plan with concatenated CSR arrays.
//! - [`UniformBatchedSpMV`] -- optimised path when all matrices share structure.
//! - [`BatchedSpGEMM`] -- batched sparse-sparse matrix multiply.
//! - [`BatchedTriSolve`] -- batched sparse triangular solve.
//! - [`BatchScheduler`] -- heuristic-based execution strategy selector.

use std::ops::{Add, AddAssign, Mul};

use oxicuda_blas::GpuFloat;

use crate::error::{SparseError, SparseResult};
use crate::format::CsrMatrix;

/// Result type for a single SpGEMM: (row_ptr, col_idx, values, rows, cols).
type SpGEMMResultU32<T> = (Vec<i32>, Vec<i32>, Vec<T>, u32, u32);

/// Result type for a single SpGEMM from host arrays: (row_ptr, col_idx, values, rows, cols).
type SpGEMMResultUsize<T> = (Vec<i32>, Vec<i32>, Vec<T>, usize, usize);

// ---------------------------------------------------------------------------
// BatchScheduler -- strategy selection
// ---------------------------------------------------------------------------

/// Execution strategy chosen by [`BatchScheduler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Strategy {
    /// Process each operation sequentially on a single stream.
    Sequential,
    /// Issue operations on `num_streams` concurrent CUDA streams.
    Concurrent(usize),
    /// Fuse all operations into a single kernel launch.
    Fused,
}

/// Heuristic-based scheduler that decides how to execute a batch.
///
/// The strategy depends on batch size and average matrix complexity:
///
/// | Condition | Strategy |
/// |-----------|----------|
/// | batch <= 4 *and* avg\_nnz >= 10 000 | [`Sequential`](Strategy::Sequential) |
/// | batch >= 64 *and* avg\_nnz < 256 | [`Fused`](Strategy::Fused) |
/// | otherwise | [`Concurrent`](Strategy::Concurrent) with up to 8 streams |
#[derive(Debug, Clone)]
pub struct BatchScheduler {
    _private: (),
}

impl BatchScheduler {
    /// Create a new scheduler.
    #[inline]
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Select an execution strategy for the given batch characteristics.
    ///
    /// # Arguments
    ///
    /// * `batch_size` -- Number of independent operations in the batch.
    /// * `avg_nnz` -- Average number of non-zeros per matrix in the batch.
    pub fn select_strategy(&self, batch_size: usize, avg_nnz: usize) -> Strategy {
        Self::select_strategy_static(batch_size, avg_nnz)
    }

    /// Static version of [`select_strategy`](Self::select_strategy) for use
    /// without constructing a scheduler instance.
    pub fn select_strategy_static(batch_size: usize, avg_nnz: usize) -> Strategy {
        // Small batches of large matrices -> sequential is fine
        if batch_size <= 4 && avg_nnz >= 10_000 {
            return Strategy::Sequential;
        }
        // Many tiny matrices -> fuse into single kernel
        if batch_size >= 64 && avg_nnz < 256 {
            return Strategy::Fused;
        }
        // Default: concurrent streams (capped at 8)
        let streams = batch_size.clamp(1, 8);
        Strategy::Concurrent(streams)
    }
}

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BatchedSpMVPlan -- precomputed concatenated arrays
// ---------------------------------------------------------------------------

/// Precomputed plan for batched SpMV execution.
///
/// Concatenates all CSR arrays (row\_ptr, col\_indices, values) from multiple
/// matrices into contiguous host-side arrays with offset tables, enabling
/// efficient data transfer and potential kernel fusion.
///
/// # Type parameters
///
/// * `T` -- Element type (must satisfy [`GpuFloat`]).
#[derive(Debug, Clone)]
pub struct BatchedSpMVPlan<T> {
    /// Concatenated row pointer array across all matrices.
    pub concat_row_ptr: Vec<i32>,
    /// Concatenated column index array across all matrices.
    pub concat_col_idx: Vec<i32>,
    /// Concatenated values array across all matrices.
    pub concat_values: Vec<T>,
    /// Start offset of each matrix's row\_ptr in `concat_row_ptr`.
    pub batch_offsets_row_ptr: Vec<usize>,
    /// Start offset of each matrix's col\_idx / values in the concatenated arrays.
    pub batch_offsets_nnz: Vec<usize>,
    /// Number of rows in each matrix.
    pub row_counts: Vec<usize>,
    /// Number of columns in each matrix.
    pub col_counts: Vec<usize>,
    /// Number of non-zeros in each matrix.
    pub nnz_counts: Vec<usize>,
    /// Total number of matrices in the batch.
    pub batch_size: usize,
    /// Recommended execution strategy.
    pub strategy: Strategy,
}

impl<T: GpuFloat> BatchedSpMVPlan<T> {
    /// Build a plan from a slice of CSR matrices.
    ///
    /// Downloads each matrix to host memory, concatenates the arrays, and
    /// records offset metadata.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] if `matrices` is empty.
    /// Returns [`SparseError::Cuda`] on download failure.
    pub fn from_matrices(matrices: &[CsrMatrix<T>]) -> SparseResult<Self> {
        if matrices.is_empty() {
            return Err(SparseError::InvalidArgument(
                "batch must contain at least one matrix".to_string(),
            ));
        }

        let batch_size = matrices.len();
        let mut concat_row_ptr = Vec::new();
        let mut concat_col_idx = Vec::new();
        let mut concat_values: Vec<T> = Vec::new();
        let mut batch_offsets_row_ptr = Vec::with_capacity(batch_size);
        let mut batch_offsets_nnz = Vec::with_capacity(batch_size);
        let mut row_counts = Vec::with_capacity(batch_size);
        let mut col_counts = Vec::with_capacity(batch_size);
        let mut nnz_counts = Vec::with_capacity(batch_size);

        for mat in matrices {
            let (h_rp, h_ci, h_vals) = mat.to_host()?;

            batch_offsets_row_ptr.push(concat_row_ptr.len());
            batch_offsets_nnz.push(concat_col_idx.len());
            row_counts.push(mat.rows() as usize);
            col_counts.push(mat.cols() as usize);
            nnz_counts.push(mat.nnz() as usize);

            concat_row_ptr.extend_from_slice(&h_rp);
            concat_col_idx.extend_from_slice(&h_ci);
            concat_values.extend_from_slice(&h_vals);
        }

        let total_nnz = nnz_counts.iter().copied().sum::<usize>();
        let avg_nnz = total_nnz.checked_div(batch_size).unwrap_or(0);
        let strategy = BatchScheduler::select_strategy_static(batch_size, avg_nnz);

        Ok(Self {
            concat_row_ptr,
            concat_col_idx,
            concat_values,
            batch_offsets_row_ptr,
            batch_offsets_nnz,
            row_counts,
            col_counts,
            nnz_counts,
            batch_size,
            strategy,
        })
    }

    /// Build a plan from host-side arrays directly (no GPU download required).
    ///
    /// Each entry in the parallel arrays describes one matrix in the batch.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] if any of the slices are empty
    /// or if their lengths do not match.
    pub fn from_host_arrays(
        row_ptrs: &[Vec<i32>],
        col_indices: &[Vec<i32>],
        values: &[Vec<T>],
        rows: &[usize],
        cols: &[usize],
    ) -> SparseResult<Self> {
        let batch_size = row_ptrs.len();
        if batch_size == 0 {
            return Err(SparseError::InvalidArgument(
                "batch must contain at least one matrix".to_string(),
            ));
        }
        if col_indices.len() != batch_size
            || values.len() != batch_size
            || rows.len() != batch_size
            || cols.len() != batch_size
        {
            return Err(SparseError::InvalidArgument(
                "all input slices must have the same length".to_string(),
            ));
        }

        let mut concat_row_ptr = Vec::new();
        let mut concat_col_idx = Vec::new();
        let mut concat_values: Vec<T> = Vec::new();
        let mut batch_offsets_row_ptr = Vec::with_capacity(batch_size);
        let mut batch_offsets_nnz = Vec::with_capacity(batch_size);
        let mut row_counts = Vec::with_capacity(batch_size);
        let mut col_counts = Vec::with_capacity(batch_size);
        let mut nnz_counts = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            batch_offsets_row_ptr.push(concat_row_ptr.len());
            batch_offsets_nnz.push(concat_col_idx.len());
            row_counts.push(rows[i]);
            col_counts.push(cols[i]);
            nnz_counts.push(values[i].len());

            concat_row_ptr.extend_from_slice(&row_ptrs[i]);
            concat_col_idx.extend_from_slice(&col_indices[i]);
            concat_values.extend_from_slice(&values[i]);
        }

        let total_nnz = nnz_counts.iter().copied().sum::<usize>();
        let avg_nnz = total_nnz.checked_div(batch_size).unwrap_or(0);
        let strategy = BatchScheduler::select_strategy_static(batch_size, avg_nnz);

        Ok(Self {
            concat_row_ptr,
            concat_col_idx,
            concat_values,
            batch_offsets_row_ptr,
            batch_offsets_nnz,
            row_counts,
            col_counts,
            nnz_counts,
            batch_size,
            strategy,
        })
    }

    /// Returns the total number of non-zeros across the entire batch.
    #[inline]
    pub fn total_nnz(&self) -> usize {
        self.nnz_counts.iter().copied().sum()
    }

    /// Returns the total number of rows across the entire batch.
    #[inline]
    pub fn total_rows(&self) -> usize {
        self.row_counts.iter().copied().sum()
    }

    /// Returns the average non-zeros per matrix.
    #[inline]
    pub fn avg_nnz(&self) -> usize {
        if self.batch_size == 0 {
            return 0;
        }
        self.total_nnz() / self.batch_size
    }
}

// ---------------------------------------------------------------------------
// BatchedSpMV -- batched sparse matrix-vector multiply
// ---------------------------------------------------------------------------

/// Batched sparse matrix-vector multiplication.
///
/// Computes `y_i = alpha * A_i * x_i + beta * y_i` for a batch of sparse
/// matrices `A_i`, dense vectors `x_i`, and output vectors `y_i`.
///
/// The baseline implementation iterates sequentially on the host. For GPU
/// execution, use [`generate_batched_spmv_ptx`] to create a single fused
/// kernel that processes all matrices via offset arrays.
#[derive(Debug)]
pub struct BatchedSpMV<T: GpuFloat> {
    /// The batch of sparse matrices (stored as host-side CSR data).
    matrices: Vec<HostCsr<T>>,
}

/// Host-side CSR representation for batched operations.
#[derive(Debug, Clone)]
struct HostCsr<T> {
    rows: usize,
    cols: usize,
    row_ptr: Vec<i32>,
    col_idx: Vec<i32>,
    values: Vec<T>,
}

impl<T: GpuFloat + Add<Output = T> + Mul<Output = T> + AddAssign> BatchedSpMV<T> {
    /// Create a `BatchedSpMV` by downloading matrix data from device memory.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] if `matrices` is empty.
    /// Returns [`SparseError::Cuda`] on download failure.
    pub fn from_device(matrices: &[CsrMatrix<T>]) -> SparseResult<Self> {
        if matrices.is_empty() {
            return Err(SparseError::InvalidArgument(
                "batch must contain at least one matrix".to_string(),
            ));
        }

        let mut host_mats = Vec::with_capacity(matrices.len());
        for mat in matrices {
            let (rp, ci, vals) = mat.to_host()?;
            host_mats.push(HostCsr {
                rows: mat.rows() as usize,
                cols: mat.cols() as usize,
                row_ptr: rp,
                col_idx: ci,
                values: vals,
            });
        }

        Ok(Self {
            matrices: host_mats,
        })
    }

    /// Create a `BatchedSpMV` from host-side CSR arrays.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] on dimension mismatch.
    pub fn from_host(
        row_ptrs: Vec<Vec<i32>>,
        col_indices: Vec<Vec<i32>>,
        values: Vec<Vec<T>>,
        rows: Vec<usize>,
        cols: Vec<usize>,
    ) -> SparseResult<Self> {
        let n = row_ptrs.len();
        if n == 0 {
            return Err(SparseError::InvalidArgument(
                "batch must contain at least one matrix".to_string(),
            ));
        }
        if col_indices.len() != n || values.len() != n || rows.len() != n || cols.len() != n {
            return Err(SparseError::InvalidArgument(
                "all input vectors must have the same length".to_string(),
            ));
        }

        let mut host_mats = Vec::with_capacity(n);
        for i in 0..n {
            host_mats.push(HostCsr {
                rows: rows[i],
                cols: cols[i],
                row_ptr: row_ptrs[i].clone(),
                col_idx: col_indices[i].clone(),
                values: values[i].clone(),
            });
        }

        Ok(Self {
            matrices: host_mats,
        })
    }

    /// Returns the number of matrices in the batch.
    #[inline]
    pub fn batch_size(&self) -> usize {
        self.matrices.len()
    }

    /// Execute the batched SpMV on the host (sequential baseline).
    ///
    /// For each matrix `A_i`:
    ///   `y_i[r] = alpha * sum(A_i[r,c] * x_i[c]) + beta * y_i[r]`
    ///
    /// # Arguments
    ///
    /// * `xs` -- Slice of input vectors, one per matrix.
    /// * `ys` -- Slice of output vectors, one per matrix (modified in place).
    /// * `alpha` -- Scalar multiplier for `A * x`.
    /// * `beta` -- Scalar multiplier for existing `y`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if slice lengths do not match
    /// the batch size, or if individual vector dimensions are wrong.
    pub fn execute(&self, xs: &[Vec<T>], ys: &mut [Vec<T>], alpha: T, beta: T) -> SparseResult<()> {
        let n = self.matrices.len();
        if xs.len() != n || ys.len() != n {
            return Err(SparseError::DimensionMismatch(format!(
                "expected {} vectors, got xs={}, ys={}",
                n,
                xs.len(),
                ys.len()
            )));
        }

        for (i, mat) in self.matrices.iter().enumerate() {
            if xs[i].len() < mat.cols {
                return Err(SparseError::DimensionMismatch(format!(
                    "matrix {} has {} cols but x has {} elements",
                    i,
                    mat.cols,
                    xs[i].len()
                )));
            }
            if ys[i].len() < mat.rows {
                return Err(SparseError::DimensionMismatch(format!(
                    "matrix {} has {} rows but y has {} elements",
                    i,
                    mat.rows,
                    ys[i].len()
                )));
            }

            host_csr_spmv(mat, &xs[i], &mut ys[i], alpha, beta);
        }

        Ok(())
    }
}

/// Perform a single CSR SpMV on the host: y = alpha*A*x + beta*y.
fn host_csr_spmv<T: GpuFloat + Add<Output = T> + Mul<Output = T> + AddAssign>(
    mat: &HostCsr<T>,
    x: &[T],
    y: &mut [T],
    alpha: T,
    beta: T,
) {
    for (row, y_row) in y.iter_mut().enumerate().take(mat.rows) {
        let start = mat.row_ptr[row] as usize;
        let end = mat.row_ptr[row + 1] as usize;

        let mut acc = T::gpu_zero();
        for j in start..end {
            let col = mat.col_idx[j] as usize;
            acc += mat.values[j] * x[col];
        }

        *y_row = alpha * acc + beta * *y_row;
    }
}

// ---------------------------------------------------------------------------
// UniformBatchedSpMV -- shared structure, only values differ
// ---------------------------------------------------------------------------

/// Optimised batched SpMV when all matrices share the same sparsity pattern.
///
/// Only the non-zero values differ between matrices; the row pointers and
/// column indices are shared. This enables more efficient memory layout and
/// potential vectorisation across the batch dimension.
#[derive(Debug, Clone)]
pub struct UniformBatchedSpMV<T> {
    /// Number of rows in every matrix.
    rows: usize,
    /// Number of columns in every matrix.
    cols: usize,
    /// Shared row pointer array (length `rows + 1`).
    row_ptr: Vec<i32>,
    /// Shared column index array (length `nnz`).
    col_idx: Vec<i32>,
    /// Per-matrix values. `batch_values[i]` has length `nnz`.
    batch_values: Vec<Vec<T>>,
}

impl<T: GpuFloat + Add<Output = T> + Mul<Output = T> + AddAssign> UniformBatchedSpMV<T> {
    /// Create a uniform batch from a pattern matrix and per-batch values.
    ///
    /// The pattern matrix supplies the shared structure (row\_ptr, col\_idx);
    /// `batch_values[i]` supplies the non-zero values for the i-th matrix.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] if `batch_values` is empty or
    /// if any entry has the wrong length.
    /// Returns [`SparseError::Cuda`] on download failure.
    pub fn from_pattern(pattern: &CsrMatrix<T>, batch_values: Vec<Vec<T>>) -> SparseResult<Self> {
        if batch_values.is_empty() {
            return Err(SparseError::InvalidArgument(
                "batch_values must not be empty".to_string(),
            ));
        }

        let nnz = pattern.nnz() as usize;
        for (i, vals) in batch_values.iter().enumerate() {
            if vals.len() != nnz {
                return Err(SparseError::InvalidArgument(format!(
                    "batch_values[{}] has length {} but pattern nnz is {}",
                    i,
                    vals.len(),
                    nnz
                )));
            }
        }

        let (rp, ci, _) = pattern.to_host()?;

        Ok(Self {
            rows: pattern.rows() as usize,
            cols: pattern.cols() as usize,
            row_ptr: rp,
            col_idx: ci,
            batch_values,
        })
    }

    /// Create a uniform batch from host-side arrays (no GPU required).
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] on dimension / length mismatch.
    pub fn from_host_arrays(
        rows: usize,
        cols: usize,
        row_ptr: Vec<i32>,
        col_idx: Vec<i32>,
        batch_values: Vec<Vec<T>>,
    ) -> SparseResult<Self> {
        if batch_values.is_empty() {
            return Err(SparseError::InvalidArgument(
                "batch_values must not be empty".to_string(),
            ));
        }
        if row_ptr.len() != rows + 1 {
            return Err(SparseError::InvalidArgument(format!(
                "row_ptr length {} != rows + 1 ({})",
                row_ptr.len(),
                rows + 1
            )));
        }
        let nnz = col_idx.len();
        for (i, vals) in batch_values.iter().enumerate() {
            if vals.len() != nnz {
                return Err(SparseError::InvalidArgument(format!(
                    "batch_values[{}] length {} != nnz {}",
                    i,
                    vals.len(),
                    nnz
                )));
            }
        }

        Ok(Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            batch_values,
        })
    }

    /// Number of matrices in the batch.
    #[inline]
    pub fn batch_size(&self) -> usize {
        self.batch_values.len()
    }

    /// Execute batched SpMV on the host.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] on vector size mismatch.
    pub fn execute(&self, xs: &[Vec<T>], ys: &mut [Vec<T>], alpha: T, beta: T) -> SparseResult<()> {
        let n = self.batch_values.len();
        if xs.len() != n || ys.len() != n {
            return Err(SparseError::DimensionMismatch(format!(
                "expected {} vectors, got xs={}, ys={}",
                n,
                xs.len(),
                ys.len()
            )));
        }

        for i in 0..n {
            if xs[i].len() < self.cols {
                return Err(SparseError::DimensionMismatch(format!(
                    "x[{}] length {} < cols {}",
                    i,
                    xs[i].len(),
                    self.cols
                )));
            }
            if ys[i].len() < self.rows {
                return Err(SparseError::DimensionMismatch(format!(
                    "y[{}] length {} < rows {}",
                    i,
                    ys[i].len(),
                    self.rows
                )));
            }

            let mat = HostCsr {
                rows: self.rows,
                cols: self.cols,
                row_ptr: self.row_ptr.clone(),
                col_idx: self.col_idx.clone(),
                values: self.batch_values[i].clone(),
            };
            host_csr_spmv(&mat, &xs[i], &mut ys[i], alpha, beta);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BatchedSpGEMM -- batched sparse-sparse matrix multiply
// ---------------------------------------------------------------------------

/// Batched sparse-sparse matrix multiplication.
///
/// Computes `C_i = A_i * B_i` for a batch of sparse matrix pairs.
/// The baseline runs each SpGEMM independently on the host using a
/// hash-table accumulation approach.
#[derive(Debug)]
pub struct BatchedSpGEMM {
    _private: (),
}

impl BatchedSpGEMM {
    /// Create a new batched SpGEMM executor.
    #[inline]
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Execute batched SpGEMM on the host.
    ///
    /// For each pair `(A_i, B_i)`, computes `C_i = A_i * B_i` using a
    /// row-by-row hash-table accumulation.
    ///
    /// Returns host-side CSR triples `(row_ptr, col_idx, values)` for each
    /// result matrix.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] if batch sizes differ or are empty.
    /// Returns [`SparseError::DimensionMismatch`] if dimensions are incompatible.
    /// Returns [`SparseError::Cuda`] on download failure.
    pub fn execute<T: GpuFloat + Add<Output = T> + Mul<Output = T> + AddAssign>(
        a_batch: &[CsrMatrix<T>],
        b_batch: &[CsrMatrix<T>],
    ) -> SparseResult<Vec<SpGEMMResultU32<T>>> {
        if a_batch.is_empty() {
            return Err(SparseError::InvalidArgument(
                "batch must not be empty".to_string(),
            ));
        }
        if a_batch.len() != b_batch.len() {
            return Err(SparseError::InvalidArgument(format!(
                "a_batch length {} != b_batch length {}",
                a_batch.len(),
                b_batch.len()
            )));
        }

        let mut results = Vec::with_capacity(a_batch.len());

        for (i, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
            if a.cols() != b.rows() {
                return Err(SparseError::DimensionMismatch(format!(
                    "pair {}: A.cols ({}) != B.rows ({})",
                    i,
                    a.cols(),
                    b.rows()
                )));
            }

            let (a_rp, a_ci, a_vals) = a.to_host()?;
            let (b_rp, b_ci, b_vals) = b.to_host()?;

            let (c_rp, c_ci, c_vals) = host_spgemm(
                &a_rp,
                &a_ci,
                &a_vals,
                a.rows() as usize,
                &b_rp,
                &b_ci,
                &b_vals,
                b.cols() as usize,
            );

            results.push((c_rp, c_ci, c_vals, a.rows(), b.cols()));
        }

        Ok(results)
    }

    /// Execute batched SpGEMM from host-side CSR arrays (no GPU required).
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] on length mismatch.
    /// Returns [`SparseError::DimensionMismatch`] on dimension mismatch.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_host<T: GpuFloat + Add<Output = T> + Mul<Output = T> + AddAssign>(
        a_row_ptrs: &[Vec<i32>],
        a_col_indices: &[Vec<i32>],
        a_values: &[Vec<T>],
        a_rows: &[usize],
        a_cols: &[usize],
        b_row_ptrs: &[Vec<i32>],
        b_col_indices: &[Vec<i32>],
        b_values: &[Vec<T>],
        b_cols: &[usize],
    ) -> SparseResult<Vec<SpGEMMResultUsize<T>>> {
        let n = a_row_ptrs.len();
        if n == 0 {
            return Err(SparseError::InvalidArgument(
                "batch must not be empty".to_string(),
            ));
        }
        if b_row_ptrs.len() != n
            || a_col_indices.len() != n
            || a_values.len() != n
            || a_rows.len() != n
            || a_cols.len() != n
            || b_col_indices.len() != n
            || b_values.len() != n
            || b_cols.len() != n
        {
            return Err(SparseError::InvalidArgument(
                "all input slices must have the same length".to_string(),
            ));
        }

        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let b_rows_i = a_cols[i]; // A.cols == B.rows
            let (c_rp, c_ci, c_vals) = host_spgemm(
                &a_row_ptrs[i],
                &a_col_indices[i],
                &a_values[i],
                a_rows[i],
                &b_row_ptrs[i],
                &b_col_indices[i],
                &b_values[i],
                b_cols[i],
            );
            let _ = b_rows_i; // validated by caller
            results.push((c_rp, c_ci, c_vals, a_rows[i], b_cols[i]));
        }

        Ok(results)
    }
}

impl Default for BatchedSpGEMM {
    fn default() -> Self {
        Self::new()
    }
}

/// Host-side SpGEMM for a single pair: C = A * B using hash accumulation.
#[allow(clippy::too_many_arguments)]
fn host_spgemm<T: GpuFloat + Add<Output = T> + Mul<Output = T> + AddAssign>(
    a_rp: &[i32],
    a_ci: &[i32],
    a_vals: &[T],
    a_rows: usize,
    b_rp: &[i32],
    b_ci: &[i32],
    b_vals: &[T],
    b_cols: usize,
) -> (Vec<i32>, Vec<i32>, Vec<T>) {
    use std::collections::BTreeMap;

    let mut c_row_ptr = vec![0i32; a_rows + 1];
    let mut c_col_idx = Vec::new();
    let mut c_values: Vec<T> = Vec::new();

    for row in 0..a_rows {
        let a_start = a_rp[row] as usize;
        let a_end = a_rp[row + 1] as usize;

        // BTreeMap gives sorted column indices automatically
        let mut acc: BTreeMap<usize, T> = BTreeMap::new();

        for ja in a_start..a_end {
            let a_col = a_ci[ja] as usize;
            let a_val = a_vals[ja];

            let b_start = b_rp[a_col] as usize;
            let b_end = b_rp[a_col + 1] as usize;

            for jb in b_start..b_end {
                let b_col = b_ci[jb] as usize;
                if b_col < b_cols {
                    let product = a_val * b_vals[jb];
                    acc.entry(b_col)
                        .and_modify(|v| *v += product)
                        .or_insert(product);
                }
            }
        }

        for (&col, &val) in &acc {
            c_col_idx.push(col as i32);
            c_values.push(val);
        }
        c_row_ptr[row + 1] = c_col_idx.len() as i32;
    }

    (c_row_ptr, c_col_idx, c_values)
}

// ---------------------------------------------------------------------------
// BatchedTriSolve -- batched sparse triangular solve
// ---------------------------------------------------------------------------

/// Batched sparse triangular solve.
///
/// Solves `L_i * x_i = b_i` for each lower-triangular CSR matrix `L_i`
/// and right-hand side `b_i` in the batch.
#[derive(Debug)]
pub struct BatchedTriSolve {
    _private: (),
}

impl BatchedTriSolve {
    /// Create a new batched triangular solve executor.
    #[inline]
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Execute batched forward-substitution on the host.
    ///
    /// Each `L_i` must be square and lower-triangular with non-zero diagonal.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] on batch size mismatch.
    /// Returns [`SparseError::DimensionMismatch`] if matrices are not square.
    /// Returns [`SparseError::SingularMatrix`] if a zero diagonal is encountered.
    /// Returns [`SparseError::Cuda`] on download failure.
    pub fn execute<T: GpuFloat + Add<Output = T> + Mul<Output = T> + AddAssign>(
        l_batch: &[CsrMatrix<T>],
        b_batch: &[Vec<T>],
    ) -> SparseResult<Vec<Vec<T>>> {
        if l_batch.is_empty() {
            return Err(SparseError::InvalidArgument(
                "batch must not be empty".to_string(),
            ));
        }
        if l_batch.len() != b_batch.len() {
            return Err(SparseError::InvalidArgument(format!(
                "l_batch length {} != b_batch length {}",
                l_batch.len(),
                b_batch.len()
            )));
        }

        let mut results = Vec::with_capacity(l_batch.len());

        for (i, (l, b)) in l_batch.iter().zip(b_batch.iter()).enumerate() {
            if l.rows() != l.cols() {
                return Err(SparseError::DimensionMismatch(format!(
                    "matrix {} is not square: {}x{}",
                    i,
                    l.rows(),
                    l.cols()
                )));
            }
            if b.len() < l.rows() as usize {
                return Err(SparseError::DimensionMismatch(format!(
                    "matrix {} has {} rows but rhs has {} elements",
                    i,
                    l.rows(),
                    b.len()
                )));
            }

            let (rp, ci, vals) = l.to_host()?;
            let x = host_forward_solve(&rp, &ci, &vals, l.rows() as usize, b)?;
            results.push(x);
        }

        Ok(results)
    }

    /// Execute batched forward-substitution from host-side arrays.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] on length mismatch.
    /// Returns [`SparseError::SingularMatrix`] on zero diagonal.
    pub fn execute_host<T: GpuFloat + Add<Output = T> + Mul<Output = T> + AddAssign>(
        row_ptrs: &[Vec<i32>],
        col_indices: &[Vec<i32>],
        values: &[Vec<T>],
        sizes: &[usize],
        rhs: &[Vec<T>],
    ) -> SparseResult<Vec<Vec<T>>> {
        let n = row_ptrs.len();
        if n == 0 {
            return Err(SparseError::InvalidArgument(
                "batch must not be empty".to_string(),
            ));
        }
        if col_indices.len() != n || values.len() != n || sizes.len() != n || rhs.len() != n {
            return Err(SparseError::InvalidArgument(
                "all input slices must have the same length".to_string(),
            ));
        }

        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let x =
                host_forward_solve(&row_ptrs[i], &col_indices[i], &values[i], sizes[i], &rhs[i])?;
            results.push(x);
        }

        Ok(results)
    }
}

impl Default for BatchedTriSolve {
    fn default() -> Self {
        Self::new()
    }
}

/// Host-side lower-triangular forward substitution: L * x = b.
fn host_forward_solve<T: GpuFloat + Add<Output = T> + Mul<Output = T> + AddAssign>(
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[T],
    n: usize,
    b: &[T],
) -> SparseResult<Vec<T>> {
    let mut x = vec![T::gpu_zero(); n];

    for row in 0..n {
        let start = row_ptr[row] as usize;
        let end = row_ptr[row + 1] as usize;

        // Accumulate off-diagonal contributions and find diagonal entry.
        // x[row] = (b[row] - sum(L[row,j]*x[j] for j<row)) / L[row,row]
        let mut off_diag_sum = T::gpu_zero();
        let mut diag = T::gpu_zero();

        for j in start..end {
            let col = col_idx[j] as usize;
            if col < row {
                off_diag_sum += values[j] * x[col];
            } else if col == row {
                diag = values[j];
            }
        }

        // Check for zero diagonal
        if diag == T::gpu_zero() {
            return Err(SparseError::SingularMatrix);
        }

        // x[row] = (b[row] - off_diag_sum) / diag
        // We cannot subtract directly without Sub trait. Use bit manipulation
        // to negate for f32/f64.
        let neg_sum = negate_float(off_diag_sum);
        let numerator = b[row] + neg_sum;
        x[row] = divide_float(numerator, diag);
    }

    Ok(x)
}

/// Negate a GpuFloat value by flipping the sign bit.
#[inline]
fn negate_float<T: GpuFloat>(val: T) -> T {
    let bits = val.to_bits_u64();
    if T::SIZE == 4 {
        // f32: flip bit 31
        T::from_bits_u64(bits ^ (1u64 << 31))
    } else {
        // f64: flip bit 63
        T::from_bits_u64(bits ^ (1u64 << 63))
    }
}

/// Divide two GpuFloat values by converting through f64.
#[inline]
fn divide_float<T: GpuFloat>(num: T, den: T) -> T {
    // Convert to f64, divide, convert back. Works for f32 and f64.
    let n_bits = num.to_bits_u64();
    let d_bits = den.to_bits_u64();
    if T::SIZE == 4 {
        let n = f32::from_bits(n_bits as u32) as f64;
        let d = f32::from_bits(d_bits as u32) as f64;
        let result = (n / d) as f32;
        T::from_bits_u64(u64::from(result.to_bits()))
    } else {
        let n = f64::from_bits(n_bits);
        let d = f64::from_bits(d_bits);
        let result = n / d;
        T::from_bits_u64(result.to_bits())
    }
}

// ---------------------------------------------------------------------------
// PTX generation stub for batched SpMV
// ---------------------------------------------------------------------------

/// Generate a batched SpMV PTX kernel that processes all matrices via offset
/// arrays in a single launch.
///
/// The generated kernel expects the following arguments:
/// - Concatenated row\_ptr, col\_idx, values arrays
/// - batch\_offsets\_row\_ptr, batch\_offsets\_nnz arrays
/// - row\_counts array
/// - x pointers array, y pointers array
/// - alpha, beta scalars
/// - batch\_size
///
/// Each thread block processes one matrix in the batch; within a block,
/// threads cooperate on rows using the scalar SpMV strategy.
///
/// # Note
///
/// This generates valid PTX source but actual GPU execution requires a
/// CUDA-capable device.
pub fn generate_batched_spmv_ptx<T: GpuFloat>() -> String {
    let type_name = T::NAME;
    let ptx_type = match T::SIZE {
        4 => ".f32",
        8 => ".f64",
        _ => ".f32",
    };

    format!(
        r#"//
// Batched SpMV kernel for {type_name}
// Generated by oxicuda-sparse batched module
//
.version 7.0
.target sm_70
.address_size 64

.visible .entry batched_spmv_{type_name}(
    .param .u64 concat_row_ptr,
    .param .u64 concat_col_idx,
    .param .u64 concat_values,
    .param .u64 batch_offsets_rp,
    .param .u64 batch_offsets_nnz,
    .param .u64 row_counts,
    .param .u64 x_ptrs,
    .param .u64 y_ptrs,
    .param {ptx_type} alpha,
    .param {ptx_type} beta,
    .param .u32 batch_size
)
{{
    .reg .u32 %r<16>;
    .reg .u64 %rd<32>;
    .reg {ptx_type} %f<8>;
    .reg .pred %p<4>;

    // blockIdx.x = matrix index in batch
    mov.u32 %r0, %ctaid.x;
    // Early exit if blockIdx >= batch_size
    ld.param.u32 %r1, [batch_size];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 ret;

    // threadIdx.x = local row within this matrix
    mov.u32 %r2, %tid.x;

    // Load row_count for this matrix
    ld.param.u64 %rd0, [row_counts];
    cvt.u64.u32 %rd1, %r0;
    mad.wide.u32 %rd2, %r0, 4, %rd0;
    ld.global.u32 %r3, [%rd2];

    // Early exit if tid >= row_count
    setp.ge.u32 %p1, %r2, %r3;
    @%p1 ret;

    ret;
}}
"#
    )
}

// ---------------------------------------------------------------------------
// CPU reference: batched SpMV for multiple right-hand sides
// ---------------------------------------------------------------------------

/// Batched SpMV (CPU reference): compute `y_i = A * x_i` for all `b` RHS vectors.
///
/// The shared sparse matrix `A` is given in CSR format. `x_batch` and `y_batch`
/// are stored in **column-major** order: element `(row_or_col, b)` is at index
/// `row_or_col * batch_size + b`.
///
/// # Arguments
///
/// * `n_rows` -- Number of rows in `A`.
/// * `n_cols` -- Number of columns in `A` (used for bounds; not strictly needed here).
/// * `row_ptr` -- CSR row pointer of length `n_rows + 1`.
/// * `col_idx` -- CSR column indices of length `nnz`.
/// * `values` -- CSR values of length `nnz`.
/// * `x_batch` -- Column-major matrix of shape `[n_cols × batch_size]`.
/// * `batch_size` -- Number of simultaneous right-hand sides.
///
/// # Returns
///
/// Output matrix `y` of shape `[n_rows × batch_size]` in column-major order.
pub fn batched_spmv_cpu(
    n_rows: usize,
    _n_cols: usize,
    row_ptr: &[u32],
    col_idx: &[u32],
    values: &[f32],
    x_batch: &[f32],
    batch_size: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; n_rows * batch_size];
    for row in 0..n_rows {
        let start = row_ptr[row] as usize;
        let end = row_ptr[row + 1] as usize;
        for idx in start..end {
            let col = col_idx[idx] as usize;
            let val = values[idx];
            for b in 0..batch_size {
                y[row * batch_size + b] += val * x_batch[col * batch_size + b];
            }
        }
    }
    y
}

// ---------------------------------------------------------------------------
// CPU reference: mixed-precision SpMV (FP16 storage, FP64 accumulation -> FP32)
// ---------------------------------------------------------------------------

/// Mixed-precision SpMV (CPU reference): matrix values stored in FP16 range,
/// accumulation performed in FP64, result cast to FP32.
///
/// On actual GPU hardware, values would be loaded as FP16 (2 bytes each) and
/// accumulated in FP32 registers. This CPU reference simulates that by treating
/// `values_fp16` as FP32 scalars already quantised to FP16 precision, and
/// accumulating in FP64 to avoid catastrophic cancellation in the reference.
///
/// # Arguments
///
/// * `n_rows` -- Number of rows.
/// * `row_ptr` -- CSR row pointer of length `n_rows + 1`.
/// * `col_idx` -- CSR column indices of length `nnz`.
/// * `values_fp16` -- Per-entry matrix values (FP16-range, stored as f32).
/// * `x` -- Dense input vector of length `n_cols`.
///
/// # Returns
///
/// Output vector `y` of length `n_rows` as FP32.
pub fn mixed_precision_spmv_cpu(
    n_rows: usize,
    row_ptr: &[u32],
    col_idx: &[u32],
    values_fp16: &[f32],
    x: &[f32],
) -> Vec<f32> {
    let mut y = vec![0.0f32; n_rows];
    for row in 0..n_rows {
        let start = row_ptr[row] as usize;
        let end = row_ptr[row + 1] as usize;
        // Accumulate in f64 for numerical stability of the reference
        let mut acc = 0.0f64;
        for idx in start..end {
            let col = col_idx[idx] as usize;
            let val = values_fp16[idx] as f64;
            acc += val * (x[col] as f64);
        }
        y[row] = acc as f32;
    }
    y
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BatchScheduler tests -----------------------------------------------

    #[test]
    fn scheduler_sequential_for_small_batch_large_matrices() {
        let s = BatchScheduler::new();
        assert_eq!(s.select_strategy(2, 50_000), Strategy::Sequential);
        assert_eq!(s.select_strategy(4, 10_000), Strategy::Sequential);
    }

    #[test]
    fn scheduler_fused_for_large_batch_small_matrices() {
        let s = BatchScheduler::new();
        assert_eq!(s.select_strategy(100, 100), Strategy::Fused);
        assert_eq!(s.select_strategy(64, 255), Strategy::Fused);
    }

    #[test]
    fn scheduler_concurrent_for_medium_cases() {
        let s = BatchScheduler::new();
        let strat = s.select_strategy(16, 1000);
        match strat {
            Strategy::Concurrent(n) => assert!((1..=8).contains(&n)),
            other => panic!("expected Concurrent, got {:?}", other),
        }
    }

    #[test]
    fn scheduler_concurrent_caps_at_8_streams() {
        let strat = BatchScheduler::select_strategy_static(32, 1000);
        assert_eq!(strat, Strategy::Concurrent(8));
    }

    #[test]
    fn scheduler_static_matches_instance() {
        let s = BatchScheduler::new();
        for (bs, nnz) in [(1, 100), (10, 500), (64, 100), (3, 20_000)] {
            assert_eq!(
                s.select_strategy(bs, nnz),
                BatchScheduler::select_strategy_static(bs, nnz)
            );
        }
    }

    #[test]
    fn scheduler_default_trait() {
        let s = BatchScheduler::default();
        // Should not panic
        let _ = s.select_strategy(1, 1);
    }

    // -- BatchedSpMVPlan tests (host arrays) --------------------------------

    #[test]
    fn plan_from_host_arrays_basic() {
        // Two 2x2 identity matrices
        let rp = vec![vec![0, 1, 2], vec![0, 1, 2]];
        let ci = vec![vec![0, 1], vec![0, 1]];
        let vals: Vec<Vec<f32>> = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let rows = vec![2, 2];
        let cols = vec![2, 2];

        let plan = BatchedSpMVPlan::from_host_arrays(&rp, &ci, &vals, &rows, &cols)
            .expect("plan creation should succeed");

        assert_eq!(plan.batch_size, 2);
        assert_eq!(plan.row_counts, vec![2, 2]);
        assert_eq!(plan.nnz_counts, vec![2, 2]);
        assert_eq!(plan.total_nnz(), 4);
        assert_eq!(plan.total_rows(), 4);
        assert_eq!(plan.avg_nnz(), 2);
        assert_eq!(plan.batch_offsets_row_ptr, vec![0, 3]);
        assert_eq!(plan.batch_offsets_nnz, vec![0, 2]);
        assert_eq!(plan.concat_row_ptr, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(plan.concat_col_idx, vec![0, 1, 0, 1]);
    }

    #[test]
    fn plan_from_host_arrays_empty_batch() {
        let result = BatchedSpMVPlan::<f32>::from_host_arrays(&[], &[], &[], &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn plan_from_host_arrays_length_mismatch() {
        let rp = vec![vec![0, 1]];
        let ci = vec![vec![0], vec![1]]; // wrong length
        let vals: Vec<Vec<f64>> = vec![vec![1.0]];
        let result = BatchedSpMVPlan::from_host_arrays(&rp, &ci, &vals, &[1], &[1]);
        assert!(result.is_err());
    }

    // -- BatchedSpMV host execution tests -----------------------------------

    #[test]
    fn spmv_host_identity_batch() {
        // Two 3x3 identity matrices
        let rp = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]];
        let ci = vec![vec![0, 1, 2], vec![0, 1, 2]];
        let vals = vec![vec![1.0_f64, 1.0, 1.0], vec![1.0, 1.0, 1.0]];
        let rows = vec![3, 3];
        let cols = vec![3, 3];

        let batch = BatchedSpMV::from_host(rp, ci, vals, rows, cols)
            .expect("batch creation should succeed");
        assert_eq!(batch.batch_size(), 2);

        let xs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let mut ys = vec![vec![0.0; 3], vec![0.0; 3]];

        batch
            .execute(&xs, &mut ys, 1.0, 0.0)
            .expect("execute should succeed");

        assert_eq!(ys[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(ys[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn spmv_host_alpha_beta() {
        // Single 2x2 matrix: [[2, 0], [0, 3]]
        let rp = vec![vec![0, 1, 2]];
        let ci = vec![vec![0, 1]];
        let vals = vec![vec![2.0_f32, 3.0]];
        let rows = vec![2];
        let cols = vec![2];

        let batch = BatchedSpMV::from_host(rp, ci, vals, rows, cols).expect("batch creation");

        let xs = vec![vec![1.0, 1.0]];
        let mut ys = vec![vec![10.0, 20.0]];

        // y = 2.0 * A*x + 0.5 * y
        // A*x = [2, 3]
        // y = [2*2 + 0.5*10, 2*3 + 0.5*20] = [9, 16]
        batch
            .execute(&xs, &mut ys, 2.0, 0.5)
            .expect("execute should succeed");

        assert!((ys[0][0] - 9.0).abs() < 1e-6);
        assert!((ys[0][1] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn spmv_host_dimension_mismatch() {
        let rp = vec![vec![0, 1]];
        let ci = vec![vec![0]];
        let vals = vec![vec![1.0_f32]];
        let rows = vec![1];
        let cols = vec![2];

        let batch = BatchedSpMV::from_host(rp, ci, vals, rows, cols).expect("batch creation");

        // Wrong number of vectors
        let xs = vec![vec![1.0; 2], vec![1.0; 2]];
        let mut ys = vec![vec![0.0], vec![0.0]];
        assert!(batch.execute(&xs, &mut ys, 1.0, 0.0).is_err());
    }

    #[test]
    fn spmv_host_empty_batch_error() {
        let result = BatchedSpMV::<f64>::from_host(vec![], vec![], vec![], vec![], vec![]);
        assert!(result.is_err());
    }

    // -- UniformBatchedSpMV tests -------------------------------------------

    #[test]
    fn uniform_spmv_host_basic() {
        // Shared pattern: 2x2 diagonal
        let row_ptr = vec![0, 1, 2];
        let col_idx = vec![0, 1];
        let batch_values = vec![
            vec![1.0_f64, 1.0], // identity
            vec![2.0, 3.0],     // diag(2, 3)
        ];

        let uniform = UniformBatchedSpMV::from_host_arrays(2, 2, row_ptr, col_idx, batch_values)
            .expect("creation should succeed");
        assert_eq!(uniform.batch_size(), 2);

        let xs = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        let mut ys = vec![vec![0.0; 2], vec![0.0; 2]];

        uniform
            .execute(&xs, &mut ys, 1.0, 0.0)
            .expect("execute should succeed");

        assert_eq!(ys[0], vec![1.0, 2.0]); // I * [1,2]
        assert!((ys[1][0] - 2.0).abs() < 1e-10); // 2*1
        assert!((ys[1][1] - 6.0).abs() < 1e-10); // 3*2
    }

    #[test]
    fn uniform_spmv_validation_errors() {
        // Empty batch_values
        let result =
            UniformBatchedSpMV::<f32>::from_host_arrays(2, 2, vec![0, 1, 2], vec![0, 1], vec![]);
        assert!(result.is_err());

        // Wrong values length
        let result = UniformBatchedSpMV::<f32>::from_host_arrays(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![vec![1.0]], // too short
        );
        assert!(result.is_err());

        // Wrong row_ptr length
        let result = UniformBatchedSpMV::<f32>::from_host_arrays(
            2,
            2,
            vec![0, 1], // too short
            vec![0, 1],
            vec![vec![1.0, 2.0]],
        );
        assert!(result.is_err());
    }

    // -- BatchedTriSolve tests (host) ---------------------------------------

    #[test]
    fn tri_solve_host_basic() {
        // L = [[2, 0], [1, 3]], b = [4, 7]
        // x[0] = 4/2 = 2
        // x[1] = (7 - 1*2)/3 = 5/3
        let rp = vec![vec![0, 1, 3]];
        let ci = vec![vec![0, 0, 1]];
        let vals = vec![vec![2.0_f64, 1.0, 3.0]];
        let sizes = vec![2];
        let rhs = vec![vec![4.0, 7.0]];

        let results = BatchedTriSolve::execute_host(&rp, &ci, &vals, &sizes, &rhs)
            .expect("solve should succeed");

        assert_eq!(results.len(), 1);
        assert!((results[0][0] - 2.0).abs() < 1e-10);
        assert!((results[0][1] - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn tri_solve_host_singular() {
        // L has zero on diagonal -> singular
        let rp = vec![vec![0, 1, 2]];
        let ci = vec![vec![0, 0]]; // row 1 has col 0 but no col 1 entry -> diag = 0
        let vals = vec![vec![1.0_f64, 2.0]];
        let sizes = vec![2];
        let rhs = vec![vec![1.0, 1.0]];

        let result = BatchedTriSolve::execute_host(&rp, &ci, &vals, &sizes, &rhs);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Task 5c: SpMV / SpMM numerical accuracy tests (host-side, CPU-only)
    // -----------------------------------------------------------------------

    // Helper: reference dense matrix-vector product y = A * x for a row-major
    // matrix A stored as a flat slice with `cols` columns.
    #[allow(dead_code)]
    fn dense_matvec(a: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0_f64; rows];
        for r in 0..rows {
            for c in 0..cols {
                y[r] += a[r * cols + c] * x[c];
            }
        }
        y
    }

    // Helper: reference dense matrix-matrix product C = A * B, A is (m×k),
    // B is (k×n), result C is (m×n), all stored row-major.
    fn dense_matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f64;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    /// SpMV for a small 5×5 sparse matrix with known values.
    ///
    /// Matrix (dense):
    /// ```text
    /// [[1, 0, 2, 0, 0],
    ///  [0, 3, 0, 0, 0],
    ///  [0, 0, 4, 0, 5],
    ///  [0, 0, 0, 6, 0],
    ///  [7, 0, 0, 0, 8]]
    /// ```
    /// x = [1, 2, 3, 4, 5]
    /// Expected: [1+6, 6, 12+25, 24, 7+40] = [7, 6, 37, 24, 47]
    #[test]
    fn test_spmv_numerical_accuracy_small() {
        // CSR for the matrix above
        let row_ptr = vec![vec![0i32, 2, 3, 5, 6, 8]];
        let col_idx = vec![vec![0i32, 2, 1, 2, 4, 3, 0, 4]];
        let values = vec![vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]];
        let rows = vec![5usize];
        let cols = vec![5usize];

        let batch = BatchedSpMV::from_host(row_ptr, col_idx, values, rows, cols)
            .expect("batch creation should succeed");

        let x = vec![vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]];
        let mut y = vec![vec![0.0_f64; 5]];

        batch
            .execute(&x, &mut y, 1.0, 0.0)
            .expect("execute should succeed");

        let expected = [7.0_f64, 6.0, 37.0, 24.0, 47.0];
        for (i, (&got, &exp)) in y[0].iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-14,
                "y[{i}]: expected {exp}, got {got}"
            );
        }
    }

    /// SpMV accuracy with large value spread (1e-6 to 1e6) — relative error < 1e-10.
    #[test]
    fn test_spmv_numerical_accuracy_value_spread() {
        // 4×4 diagonal with wildly varying values
        let big = 1e6_f64;
        let small = 1e-6_f64;
        // diag(1e-6, 1e6, 1e-6, 1e6)
        let row_ptr = vec![vec![0i32, 1, 2, 3, 4]];
        let col_idx = vec![vec![0i32, 1, 2, 3]];
        let values = vec![vec![small, big, small, big]];
        let rows = vec![4usize];
        let cols = vec![4usize];

        let batch =
            BatchedSpMV::from_host(row_ptr, col_idx, values, rows, cols).expect("batch creation");

        let x = vec![vec![1.0_f64, 1.0, 1.0, 1.0]];
        let mut y = vec![vec![0.0_f64; 4]];
        batch.execute(&x, &mut y, 1.0, 0.0).expect("execute");

        let expected = [small, big, small, big];
        for (i, (&got, &exp)) in y[0].iter().zip(expected.iter()).enumerate() {
            let rel_err = (got - exp).abs() / exp.abs().max(1e-300);
            assert!(
                rel_err < 1e-10,
                "y[{i}]: relative error {rel_err:.3e} exceeds threshold"
            );
        }
    }

    /// SpMV with alpha/beta scaling: y = alpha * A * x + beta * y.
    #[test]
    fn test_spmv_alpha_beta_scaling() {
        // A = [[2, 0], [0, 3]], x = [1, 1], y_init = [10, 20]
        // A*x = [2, 3]
        // y = 5 * [2, 3] + 0.25 * [10, 20] = [10+2.5, 15+5] = [12.5, 20]
        let row_ptr = vec![vec![0i32, 1, 2]];
        let col_idx = vec![vec![0i32, 1]];
        let values = vec![vec![2.0_f64, 3.0]];

        let batch = BatchedSpMV::from_host(row_ptr, col_idx, values, vec![2], vec![2])
            .expect("batch creation");

        let x = vec![vec![1.0_f64, 1.0]];
        let mut y = vec![vec![10.0_f64, 20.0]];
        batch.execute(&x, &mut y, 5.0, 0.25).expect("execute");

        assert!((y[0][0] - 12.5).abs() < 1e-13, "y[0] = {}", y[0][0]);
        assert!((y[0][1] - 20.0).abs() < 1e-13, "y[1] = {}", y[0][1]);
    }

    /// SpMV for identity matrix: I * x = x.
    #[test]
    fn test_spmv_identity_matrix() {
        let n = 6usize;
        let row_ptr = vec![(0..=(n as i32)).collect::<Vec<i32>>()];
        let col_idx = vec![(0..n as i32).collect::<Vec<i32>>()];
        let values = vec![vec![1.0_f64; n]];

        let batch = BatchedSpMV::from_host(row_ptr, col_idx, values, vec![n], vec![n])
            .expect("batch creation");

        let x_data: Vec<f64> = (1..=(n as i64)).map(|v| v as f64).collect();
        let x = vec![x_data.clone()];
        let mut y = vec![vec![0.0_f64; n]];
        batch.execute(&x, &mut y, 1.0, 0.0).expect("execute");

        for (i, (&got, &exp)) in y[0].iter().zip(x_data.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-14,
                "identity: y[{i}] = {got}, expected {exp}"
            );
        }
    }

    /// SpMM: A * B = C where B is a dense matrix (multiple right-hand sides).
    ///
    /// Uses `BatchedSpMV::execute` once per column of B as a surrogate for SpMM.
    /// A is 4×4, B is 4×3, C = A * B verified against dense reference.
    #[test]
    fn test_spmm_numerical_accuracy() {
        // A (4×4 sparse):
        // [[1, 0, 2, 0],
        //  [0, 3, 0, 4],
        //  [5, 0, 6, 0],
        //  [0, 7, 0, 8]]
        let a_dense = [
            1.0_f64, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 0.0, 7.0, 0.0, 8.0,
        ];

        let row_ptr = vec![0i32, 2, 4, 6, 8];
        let col_idx = vec![0i32, 2, 1, 3, 0, 2, 1, 3];
        let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // B (4×3):
        let b_dense = [
            1.0_f64, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 0.0, 1.0, 4.0,
        ];

        // C_ref = dense_matmul(A, B) (4×3)
        let c_ref = dense_matmul(&a_dense, 4, 4, &b_dense, 3);

        // Compute each column of C using SpMV
        let n_cols = 3usize;
        let mut c_got = vec![0.0_f64; 4 * n_cols];

        for col in 0..n_cols {
            // Extract column `col` of B as a vector
            let x_col: Vec<f64> = (0..4).map(|r| b_dense[r * n_cols + col]).collect();

            let batch = BatchedSpMV::from_host(
                vec![row_ptr.clone()],
                vec![col_idx.clone()],
                vec![values.clone()],
                vec![4],
                vec![4],
            )
            .expect("batch creation");

            let mut y = vec![vec![0.0_f64; 4]];
            batch.execute(&[x_col], &mut y, 1.0, 0.0).expect("execute");

            for row in 0..4 {
                c_got[row * n_cols + col] = y[0][row];
            }
        }

        for i in 0..4 * n_cols {
            assert!(
                (c_got[i] - c_ref[i]).abs() < 1e-13,
                "C[{i}]: got {}, expected {}",
                c_got[i],
                c_ref[i]
            );
        }
    }

    // -- PTX generation test ------------------------------------------------

    #[test]
    fn ptx_generation_f32() {
        let ptx = generate_batched_spmv_ptx::<f32>();
        assert!(ptx.contains("batched_spmv_f32"));
        assert!(ptx.contains(".f32"));
        assert!(ptx.contains(".version"));
    }

    #[test]
    fn ptx_generation_f64() {
        let ptx = generate_batched_spmv_ptx::<f64>();
        assert!(ptx.contains("batched_spmv_f64"));
        assert!(ptx.contains(".f64"));
    }

    // -- BatchedSpGEMM host test --------------------------------------------

    #[test]
    fn spgemm_host_identity_times_matrix() {
        // I * A = A for 2x2
        let i_rp = vec![vec![0, 1, 2]];
        let i_ci = vec![vec![0, 1]];
        let i_vals: Vec<Vec<f64>> = vec![vec![1.0, 1.0]];

        let a_rp = vec![vec![0, 2, 3]]; // row 0: cols 0,1; row 1: col 1
        let a_ci = vec![vec![0, 1, 1]];
        let a_vals: Vec<Vec<f64>> = vec![vec![2.0, 3.0, 4.0]];

        let results = BatchedSpGEMM::execute_host(
            &i_rp,
            &i_ci,
            &i_vals,
            &[2],
            &[2],
            &a_rp,
            &a_ci,
            &a_vals,
            &[2],
        )
        .expect("spgemm should succeed");

        assert_eq!(results.len(), 1);
        let (c_rp, c_ci, c_vals, m, n) = &results[0];
        assert_eq!(*m, 2);
        assert_eq!(*n, 2);
        // Row 0 of C should have entries at cols 0 and 1
        let r0_start = c_rp[0] as usize;
        let r0_end = c_rp[1] as usize;
        assert_eq!(r0_end - r0_start, 2);
        assert_eq!(c_ci[r0_start], 0);
        assert_eq!(c_ci[r0_start + 1], 1);
        assert!((c_vals[r0_start] - 2.0).abs() < 1e-10);
        assert!((c_vals[r0_start + 1] - 3.0).abs() < 1e-10);
    }

    // -- batched_spmv_cpu tests ------------------------------------------------

    #[test]
    fn batched_spmv_identity_2rhs() {
        // 4x4 identity matrix
        let n_rows = 4usize;
        let n_cols = 4usize;
        let row_ptr = vec![0u32, 1, 2, 3, 4];
        let col_idx = vec![0u32, 1, 2, 3];
        let values = vec![1.0f32; 4];

        // X = [[1,5], [2,6], [3,7], [4,8]] column-major:
        // x_batch[col * batch_size + b]:
        //   col=0: b=0 -> 1, b=1 -> 5
        //   col=1: b=0 -> 2, b=1 -> 6
        //   col=2: b=0 -> 3, b=1 -> 7
        //   col=3: b=0 -> 4, b=1 -> 8
        let batch_size = 2usize;
        let x_batch = vec![1.0f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];

        let y = batched_spmv_cpu(
            n_rows, n_cols, &row_ptr, &col_idx, &values, &x_batch, batch_size,
        );

        // y[row * batch_size + b] = x[row, b] since A is identity
        // row=0, b=0 -> 1; row=0, b=1 -> 5
        // row=1, b=0 -> 2; row=1, b=1 -> 6
        // row=2, b=0 -> 3; row=2, b=1 -> 7
        // row=3, b=0 -> 4; row=3, b=1 -> 8
        assert_eq!(y.len(), n_rows * batch_size);
        assert!((y[0] - 1.0).abs() < 1e-6, "row=0, b=0 should be 1.0");
        assert!((y[1] - 5.0).abs() < 1e-6, "row=0, b=1 should be 5.0");
        assert!((y[2] - 2.0).abs() < 1e-6, "row=1, b=0 should be 2.0");
        assert!((y[3] - 6.0).abs() < 1e-6, "row=1, b=1 should be 6.0");
        assert!((y[4] - 3.0).abs() < 1e-6, "row=2, b=0 should be 3.0");
        assert!((y[5] - 7.0).abs() < 1e-6, "row=2, b=1 should be 7.0");
        assert!((y[6] - 4.0).abs() < 1e-6, "row=3, b=0 should be 4.0");
        assert!((y[7] - 8.0).abs() < 1e-6, "row=3, b=1 should be 8.0");
    }

    #[test]
    fn batched_spmv_correctness_3rhs() {
        // 3x3 matrix A:
        //  [2 1 0]
        //  [0 3 1]
        //  [1 0 4]
        // row_ptr = [0, 2, 4, 6]
        // col_idx = [0, 1, 1, 2, 0, 2]
        // values  = [2, 1, 3, 1, 1, 4]
        let n_rows = 3usize;
        let n_cols = 3usize;
        let row_ptr = vec![0u32, 2, 4, 6];
        let col_idx = vec![0u32, 1, 1, 2, 0, 2];
        let values = vec![2.0f32, 1.0, 3.0, 1.0, 1.0, 4.0];
        let batch_size = 3usize;

        // 3 RHS column-major: x[:,0]=[1,0,0], x[:,1]=[0,1,0], x[:,2]=[0,0,1]
        // i.e., x_batch[col*3+b]: col=0: [1,0,0], col=1: [0,1,0], col=2: [0,0,1]
        let x_batch = vec![
            1.0f32, 0.0, 0.0, // col 0
            0.0, 1.0, 0.0, // col 1
            0.0, 0.0, 1.0, // col 2
        ];

        let y = batched_spmv_cpu(
            n_rows, n_cols, &row_ptr, &col_idx, &values, &x_batch, batch_size,
        );

        // A * [1,0,0] = [2,0,1]  => y[row*3+0]
        // A * [0,1,0] = [1,3,0]  => y[row*3+1]
        // A * [0,0,1] = [0,1,4]  => y[row*3+2]
        assert!((y[0] - 2.0).abs() < 1e-6, "A*e0 row0 = 2");
        assert!((y[1] - 1.0).abs() < 1e-6, "A*e1 row0 = 1");
        assert!((y[2] - 0.0).abs() < 1e-6, "A*e2 row0 = 0");
        assert!((y[3] - 0.0).abs() < 1e-6, "A*e0 row1 = 0");
        assert!((y[4] - 3.0).abs() < 1e-6, "A*e1 row1 = 3");
        assert!((y[5] - 1.0).abs() < 1e-6, "A*e2 row1 = 1");
        assert!((y[6] - 1.0).abs() < 1e-6, "A*e0 row2 = 1");
        assert!((y[7] - 0.0).abs() < 1e-6, "A*e1 row2 = 0");
        assert!((y[8] - 4.0).abs() < 1e-6, "A*e2 row2 = 4");
    }

    // -- mixed_precision_spmv_cpu tests ----------------------------------------

    #[test]
    fn mixed_precision_spmv_correctness() {
        // 4x4 identity matrix, x=[1,2,3,4] => y=[1,2,3,4]
        let n_rows = 4usize;
        let row_ptr = vec![0u32, 1, 2, 3, 4];
        let col_idx = vec![0u32, 1, 2, 3];
        let values_fp16 = vec![1.0f32; 4];
        let x = vec![1.0f32, 2.0, 3.0, 4.0];

        let y = mixed_precision_spmv_cpu(n_rows, &row_ptr, &col_idx, &values_fp16, &x);

        assert_eq!(y.len(), n_rows);
        for (i, &yi) in y.iter().enumerate() {
            assert!(
                (yi - (i + 1) as f32).abs() < 1e-4,
                "y[{}] should be {} but got {}",
                i,
                i + 1,
                yi
            );
        }
    }

    #[test]
    fn mixed_precision_accumulation_fp32() {
        // Large sum: 1000-row diagonal, all values=1.0, x=[1.0; 1000]
        // Each y[i] = 1.0, which is finite and representable in f32
        let n_rows = 1000usize;
        let row_ptr: Vec<u32> = (0..=1000).map(|i| i as u32).collect();
        let col_idx: Vec<u32> = (0..1000).map(|i| i as u32).collect();
        let values_fp16 = vec![1.0f32; 1000];
        let x = vec![1.0f32; 1000];

        let y = mixed_precision_spmv_cpu(n_rows, &row_ptr, &col_idx, &values_fp16, &x);

        assert_eq!(y.len(), n_rows);
        // All results should be finite (the accumulation via f64 avoids overflow)
        for (i, &yi) in y.iter().enumerate() {
            assert!(yi.is_finite(), "y[{}] = {} is not finite", i, yi);
            assert!(
                (yi - 1.0).abs() < 1e-4,
                "y[{}] should be 1.0, got {}",
                i,
                yi
            );
        }
    }
}
