//! Merge-based SpGEMM: `C = A * B` using merge-path load balancing.
//!
//! When rows of `A` have widely varying numbers of non-zeros (skewed row
//! lengths), a simple row-based SpGEMM assigns unequal work to each thread.
//! The merge-path approach partitions the total work evenly across threads
//! by treating the SpGEMM as a merge of work intervals.
//!
//! ## Algorithm
//!
//! **Two phases:**
//! 1. **Symbolic**: Determine `C`'s structure. For each row of `A`, count the
//!    total number of intermediate products (sum of `B.nnz_in_row[k]` for each
//!    nonzero `A[row, k]`). Use merge-path partitioning to balance work across
//!    threads. Accumulate unique column indices per row of `C` using hash sets.
//!
//! 2. **Numeric**: Compute values with balanced work distribution. Each thread
//!    processes its assigned merge-path interval, accumulating `A[i,k] * B[k,j]`
//!    products into the correct positions of `C`.
//!
//! This implementation uses a host-side (CPU) fallback for correctness, with
//! the merge-path work distribution to demonstrate the algorithm structure.
#![allow(dead_code)]

use std::collections::HashMap;

use oxicuda_blas::GpuFloat;

use crate::error::{SparseError, SparseResult};
use crate::format::CsrMatrix;
use crate::handle::SparseHandle;

// ---------------------------------------------------------------------------
// GpuFloat <-> f64 conversion helpers
// ---------------------------------------------------------------------------

fn to_f64<T: GpuFloat>(val: T) -> f64 {
    if T::SIZE == 4 {
        f32::from_bits(val.to_bits_u64() as u32) as f64
    } else {
        f64::from_bits(val.to_bits_u64())
    }
}

fn from_f64<T: GpuFloat>(val: f64) -> T {
    if T::SIZE == 4 {
        T::from_bits_u64(u64::from((val as f32).to_bits()))
    } else {
        T::from_bits_u64(val.to_bits())
    }
}

fn add_gpu_float<T: GpuFloat>(a: T, b: T) -> T {
    from_f64::<T>(to_f64(a) + to_f64(b))
}

fn mul_gpu_float<T: GpuFloat>(a: T, b: T) -> T {
    from_f64::<T>(to_f64(a) * to_f64(b))
}

// ---------------------------------------------------------------------------
// Merge-path work distribution
// ---------------------------------------------------------------------------

/// Represents a work interval for a single row of C.
struct RowWork {
    /// Row index in A (and C).
    row: usize,
    /// Total number of intermediate products for this row.
    total_products: usize,
}

/// Compute the work distribution for merge-path SpGEMM.
///
/// Returns the total work per row (number of intermediate products).
fn compute_row_work(
    a_row_ptr: &[i32],
    a_col_idx: &[i32],
    b_row_ptr: &[i32],
    m: usize,
) -> Vec<RowWork> {
    let mut work = Vec::with_capacity(m);
    for row in 0..m {
        let a_start = a_row_ptr[row] as usize;
        let a_end = a_row_ptr[row + 1] as usize;
        let mut total = 0usize;
        for &a_col in &a_col_idx[a_start..a_end] {
            let k = a_col as usize;
            let b_nnz_row_k = (b_row_ptr[k + 1] - b_row_ptr[k]) as usize;
            total += b_nnz_row_k;
        }
        work.push(RowWork {
            row,
            total_products: total,
        });
    }
    work
}

/// Merge-path partition: given cumulative work array, find the row boundaries
/// for `num_partitions` balanced partitions.
///
/// Returns partition boundaries: `partitions[p]` is the first row assigned to
/// partition `p`. `partitions[num_partitions]` = m.
fn merge_path_partition(row_work: &[RowWork], num_partitions: usize) -> Vec<usize> {
    let m = row_work.len();
    if m == 0 || num_partitions == 0 {
        return vec![0; num_partitions + 1];
    }

    let total_work: usize = row_work.iter().map(|w| w.total_products).sum();
    let work_per_partition = total_work.div_ceil(num_partitions);

    let mut partitions = Vec::with_capacity(num_partitions + 1);
    partitions.push(0);

    let mut cumulative = 0usize;
    let mut row_idx = 0;

    for p in 1..num_partitions {
        let target = p * work_per_partition;
        while row_idx < m && cumulative + row_work[row_idx].total_products <= target {
            cumulative += row_work[row_idx].total_products;
            row_idx += 1;
        }
        partitions.push(row_idx);
    }

    partitions.push(m);
    partitions
}

// ---------------------------------------------------------------------------
// SpGEMM Merge: symbolic + numeric combined
// ---------------------------------------------------------------------------

/// Merge-based SpGEMM: `C = A * B` using merge-path load balancing.
///
/// Better than row-based SpGEMM when row lengths vary widely. Uses a two-phase
/// approach:
/// 1. Symbolic: determine C's structure via merge-path partitioning.
/// 2. Numeric: compute values with balanced work distribution.
///
/// # Arguments
///
/// * `_handle` -- Sparse handle (reserved for GPU path).
/// * `a` -- Sparse CSR matrix `A` of shape `(m, k)`.
/// * `b` -- Sparse CSR matrix `B` of shape `(k, n)`.
///
/// # Returns
///
/// The product `C = A * B` as a CSR matrix of shape `(m, n)`.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `A.cols() != B.rows()`.
pub fn spgemm_merge<T: GpuFloat>(
    _handle: &SparseHandle,
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
) -> SparseResult<CsrMatrix<T>> {
    if a.cols() != b.rows() {
        return Err(SparseError::DimensionMismatch(format!(
            "A.cols ({}) != B.rows ({})",
            a.cols(),
            b.rows()
        )));
    }

    let m = a.rows() as usize;
    let n = b.cols();

    if m == 0 {
        // Return empty matrix
        let row_ptr = vec![0i32; 2];
        let col_idx = vec![0i32];
        let values = vec![T::gpu_zero()];
        return CsrMatrix::from_host(1, n, &row_ptr, &col_idx, &values);
    }

    let (a_row_ptr, a_col_idx, a_values) = a.to_host()?;
    let (b_row_ptr, b_col_idx, b_values) = b.to_host()?;

    // Compute work distribution
    let row_work = compute_row_work(&a_row_ptr, &a_col_idx, &b_row_ptr, m);

    // Use merge-path partitioning (for demonstration, we use a reasonable number)
    let num_partitions = m.min(256);
    let partitions = merge_path_partition(&row_work, num_partitions);

    // Process each partition and accumulate results
    // Each partition processes its assigned rows
    let mut c_row_ptr = vec![0i32; m + 1];
    let mut c_col_idx = Vec::new();
    let mut c_values = Vec::new();

    // Process partitions sequentially on host (GPU would parallelize)
    for p in 0..num_partitions {
        let row_start = partitions[p];
        let row_end = partitions[p + 1];

        for row in row_start..row_end {
            let a_start = a_row_ptr[row] as usize;
            let a_end = a_row_ptr[row + 1] as usize;

            // Accumulate: column -> value
            let mut accum: HashMap<i32, T> = HashMap::new();

            for a_nz in a_start..a_end {
                let k = a_col_idx[a_nz] as usize;
                let a_val = a_values[a_nz];

                let b_start = b_row_ptr[k] as usize;
                let b_end = b_row_ptr[k + 1] as usize;

                for b_nz in b_start..b_end {
                    let j = b_col_idx[b_nz];
                    let product = mul_gpu_float(a_val, b_values[b_nz]);
                    let entry = accum.entry(j).or_insert(T::gpu_zero());
                    *entry = add_gpu_float(*entry, product);
                }
            }

            // Sort by column index and append to C
            let mut sorted_entries: Vec<(i32, T)> = accum.into_iter().collect();
            sorted_entries.sort_by_key(|&(col, _)| col);

            for (col, val) in &sorted_entries {
                c_col_idx.push(*col);
                c_values.push(*val);
            }

            c_row_ptr[row + 1] = c_col_idx.len() as i32;
        }
    }

    if c_values.is_empty() {
        return Err(SparseError::ZeroNnz);
    }

    CsrMatrix::from_host(a.rows(), n, &c_row_ptr, &c_col_idx, &c_values)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_work_identity() {
        // Identity 3x3
        let a_row_ptr = vec![0, 1, 2, 3];
        let a_col_idx = vec![0, 1, 2];
        let b_row_ptr = vec![0, 1, 2, 3];

        let work = compute_row_work(&a_row_ptr, &a_col_idx, &b_row_ptr, 3);
        assert_eq!(work.len(), 3);
        for w in &work {
            assert_eq!(w.total_products, 1);
        }
    }

    #[test]
    fn row_work_skewed() {
        // Row 0 has 3 nnz, row 1 has 1 nnz (in A)
        // B is identity 3x3
        let a_row_ptr = vec![0, 3, 4];
        let a_col_idx = vec![0, 1, 2, 0];
        let b_row_ptr = vec![0, 1, 2, 3];

        let work = compute_row_work(&a_row_ptr, &a_col_idx, &b_row_ptr, 2);
        assert_eq!(work[0].total_products, 3);
        assert_eq!(work[1].total_products, 1);
    }

    #[test]
    fn merge_path_partition_balanced() {
        let work = vec![
            RowWork {
                row: 0,
                total_products: 10,
            },
            RowWork {
                row: 1,
                total_products: 10,
            },
            RowWork {
                row: 2,
                total_products: 10,
            },
            RowWork {
                row: 3,
                total_products: 10,
            },
        ];
        let parts = merge_path_partition(&work, 2);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], 0);
        assert_eq!(parts[2], 4);
        // Each partition should get ~20 work units
        let p1_work: usize = work[parts[0]..parts[1]]
            .iter()
            .map(|w| w.total_products)
            .sum();
        let p2_work: usize = work[parts[1]..parts[2]]
            .iter()
            .map(|w| w.total_products)
            .sum();
        assert!(p1_work > 0);
        assert!(p2_work > 0);
    }

    #[test]
    fn merge_path_partition_empty() {
        let work: Vec<RowWork> = Vec::new();
        let parts = merge_path_partition(&work, 4);
        assert_eq!(parts.len(), 5);
        assert!(parts.iter().all(|&p| p == 0));
    }

    #[test]
    fn merge_path_partition_single_row() {
        let work = vec![RowWork {
            row: 0,
            total_products: 100,
        }];
        let parts = merge_path_partition(&work, 4);
        assert_eq!(parts[0], 0);
        assert_eq!(*parts.last().expect("test: non-empty"), 1);
    }

    #[test]
    fn gpu_float_arithmetic_f32() {
        let a = 3.0_f32;
        let b = 4.0_f32;
        let result = mul_gpu_float(a, b);
        assert!((result - 12.0_f32).abs() < 1e-6);

        let result = add_gpu_float(a, b);
        assert!((result - 7.0_f32).abs() < 1e-6);
    }

    #[test]
    fn gpu_float_arithmetic_f64() {
        let a = 3.0_f64;
        let b = 4.0_f64;
        let result = mul_gpu_float(a, b);
        assert!((result - 12.0_f64).abs() < 1e-12);

        let result = add_gpu_float(a, b);
        assert!((result - 7.0_f64).abs() < 1e-12);
    }

    #[test]
    fn merge_path_partition_skewed() {
        // Very skewed: one huge row and many tiny rows
        let mut work = Vec::new();
        work.push(RowWork {
            row: 0,
            total_products: 1000,
        });
        for i in 1..10 {
            work.push(RowWork {
                row: i,
                total_products: 1,
            });
        }
        let parts = merge_path_partition(&work, 4);
        assert_eq!(parts[0], 0);
        assert_eq!(*parts.last().expect("test: non-empty"), 10);
    }
}
