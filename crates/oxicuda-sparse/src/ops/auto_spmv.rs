//! Heuristic-based format selection and auto-dispatching SpMV.
//!
//! Analyzes the sparsity structure of a CSR matrix and recommends the optimal
//! storage format and SpMV algorithm. The auto-dispatching SpMV function
//! selects the best algorithm at runtime based on matrix characteristics.
//!
//! ## Heuristics
//!
//! The format recommendation is based on:
//! - **Coefficient of variation** of nnz/row: measures regularity
//! - **Max/avg ratio** of nnz/row: detects extreme outlier rows
//! - **Block detection**: samples the matrix for dense sub-blocks
//!
//! | Format | Best when... |
//! |--------|-------------|
//! | CSR    | General, irregular sparsity |
//! | ELL    | Regular sparsity, similar nnz/row (low CoV) |
//! | BSR    | Block structure detected (e.g. from FEM) |
//! | CSR5   | Highly irregular row lengths (high max/avg) |
#![allow(dead_code)]

use oxicuda_blas::GpuFloat;
use oxicuda_driver::ffi::CUdeviceptr;

use crate::error::SparseResult;
use crate::format::CsrMatrix;
use crate::handle::SparseHandle;
use crate::ops::spmv::{SpMVAlgo, spmv};

// ---------------------------------------------------------------------------
// Recommended format
// ---------------------------------------------------------------------------

/// Recommended sparse matrix format for SpMV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecommendedFormat {
    /// General CSR: good default for irregular sparsity.
    Csr,
    /// ELLPACK: best for matrices with regular row lengths (similar nnz/row).
    Ell,
    /// Block Sparse Row: best when dense sub-blocks are detected.
    Bsr,
    /// CSR5: best for highly irregular row lengths (extreme max/avg ratio).
    Csr5,
}

impl std::fmt::Display for RecommendedFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Csr => write!(f, "CSR"),
            Self::Ell => write!(f, "ELL"),
            Self::Bsr => write!(f, "BSR"),
            Self::Csr5 => write!(f, "CSR5"),
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix structure analysis
// ---------------------------------------------------------------------------

/// Statistics about the sparsity structure of a CSR matrix.
#[derive(Debug, Clone)]
pub struct SparsityStats {
    /// Number of rows.
    pub rows: usize,
    /// Number of non-zeros.
    pub nnz: usize,
    /// Average nnz per row.
    pub avg_nnz_per_row: f64,
    /// Maximum nnz in any row.
    pub max_nnz_per_row: usize,
    /// Minimum nnz in any row.
    pub min_nnz_per_row: usize,
    /// Standard deviation of nnz per row.
    pub std_nnz_per_row: f64,
    /// Coefficient of variation (std / avg).
    pub cov_nnz_per_row: f64,
    /// Ratio of max nnz to avg nnz.
    pub max_avg_ratio: f64,
    /// Whether block structure was detected.
    pub block_detected: bool,
    /// Detected block size (if any).
    pub block_size: usize,
}

/// Analyze the sparsity structure of a CSR matrix.
///
/// Downloads the row pointer array to compute per-row nnz statistics and
/// performs block detection via sampling.
///
/// # Errors
///
/// Returns [`crate::error::SparseError::Cuda`] on data transfer failure.
pub fn analyze_sparsity<T: GpuFloat>(matrix: &CsrMatrix<T>) -> SparseResult<SparsityStats> {
    let n = matrix.rows() as usize;
    if n == 0 {
        return Ok(SparsityStats {
            rows: 0,
            nnz: 0,
            avg_nnz_per_row: 0.0,
            max_nnz_per_row: 0,
            min_nnz_per_row: 0,
            std_nnz_per_row: 0.0,
            cov_nnz_per_row: 0.0,
            max_avg_ratio: 0.0,
            block_detected: false,
            block_size: 1,
        });
    }

    let (h_row_ptr, h_col_idx, _) = matrix.to_host()?;

    // Compute per-row nnz
    let row_nnz: Vec<usize> = (0..n)
        .map(|i| (h_row_ptr[i + 1] - h_row_ptr[i]) as usize)
        .collect();

    let total_nnz = matrix.nnz() as usize;
    let avg = total_nnz as f64 / n as f64;
    let max_nnz = row_nnz.iter().copied().max().unwrap_or(0);
    let min_nnz = row_nnz.iter().copied().min().unwrap_or(0);

    // Standard deviation
    let variance: f64 = row_nnz
        .iter()
        .map(|&x| {
            let diff = x as f64 - avg;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;
    let std_dev = variance.sqrt();

    let cov = if avg > 0.0 { std_dev / avg } else { 0.0 };
    let max_avg_ratio = if avg > 0.0 { max_nnz as f64 / avg } else { 0.0 };

    // Block detection: sample rows and check for block structure
    let (block_detected, block_size) =
        detect_blocks(&h_row_ptr, &h_col_idx, n, matrix.cols() as usize);

    Ok(SparsityStats {
        rows: n,
        nnz: total_nnz,
        avg_nnz_per_row: avg,
        max_nnz_per_row: max_nnz,
        min_nnz_per_row: min_nnz,
        std_nnz_per_row: std_dev,
        cov_nnz_per_row: cov,
        max_avg_ratio,
        block_detected,
        block_size,
    })
}

/// Detect block structure in a CSR matrix by sampling.
///
/// For BSR with block size `bs`, each row should have nnz that is a multiple
/// of `bs`, and columns should form groups of `bs` consecutive indices starting
/// at multiples of `bs`. All rows within a block-row should have the same
/// column structure.
fn detect_blocks(row_ptr: &[i32], col_idx: &[i32], n: usize, _cols: usize) -> (bool, usize) {
    if n < 4 {
        return (false, 1);
    }

    // Try block sizes 4, 3, 2 (prefer larger blocks)
    for bs in [4usize, 3, 2] {
        if n % bs != 0 {
            continue;
        }

        let num_block_rows = n / bs;
        let sample_count = num_block_rows.min(16);
        let step = if num_block_rows > sample_count {
            num_block_rows / sample_count
        } else {
            1
        };

        let mut aligned_count = 0usize;
        let mut total_checked = 0usize;

        for block_row in (0..num_block_rows).step_by(step) {
            let base_row = block_row * bs;
            if base_row + bs > n {
                break;
            }

            let r0_start = row_ptr[base_row] as usize;
            let r0_end = row_ptr[base_row + 1] as usize;
            let r0_nnz = r0_end - r0_start;

            // Require nnz to be a multiple of bs and >= bs
            if r0_nnz < bs || r0_nnz % bs != 0 {
                total_checked += 1;
                continue;
            }

            // Check that base row columns form groups of bs consecutive indices
            // starting at multiples of bs
            let num_blocks_in_row = r0_nnz / bs;
            let base_cols_grouped = (0..num_blocks_in_row).all(|blk| {
                let group_start = r0_start + blk * bs;
                let first_col = col_idx[group_start] as usize;
                // First column in group must be block-aligned
                if first_col % bs != 0 {
                    return false;
                }
                // Remaining columns in group must be consecutive
                (1..bs).all(|offset| col_idx[group_start + offset] as usize == first_col + offset)
            });

            if !base_cols_grouped {
                total_checked += 1;
                continue;
            }

            // Check that all other rows in the block have identical column structure
            let mut block_aligned = true;
            for sub_row in 1..bs {
                let ri_start = row_ptr[base_row + sub_row] as usize;
                let ri_end = row_ptr[base_row + sub_row + 1] as usize;
                let ri_nnz = ri_end - ri_start;

                if ri_nnz != r0_nnz {
                    block_aligned = false;
                    break;
                }

                // Columns should be identical to the base row
                let cols_match =
                    (0..r0_nnz).all(|k| col_idx[ri_start + k] == col_idx[r0_start + k]);

                if !cols_match {
                    block_aligned = false;
                    break;
                }
            }

            total_checked += 1;
            if block_aligned {
                aligned_count += 1;
            }
        }

        // If > 60% of sampled blocks are aligned, declare block structure
        if total_checked > 0 && aligned_count * 10 > total_checked * 6 {
            return (true, bs);
        }
    }

    (false, 1)
}

// ---------------------------------------------------------------------------
// Format recommendation
// ---------------------------------------------------------------------------

/// Thresholds for format selection.
const COV_ELL_THRESHOLD: f64 = 0.3;
const MAX_AVG_CSR5_THRESHOLD: f64 = 10.0;

/// Analyze matrix structure and recommend the optimal sparse format.
///
/// # Arguments
///
/// * `matrix` -- CSR matrix to analyze.
///
/// # Returns
///
/// The recommended format based on sparsity structure heuristics.
///
/// # Errors
///
/// Returns [`crate::error::SparseError::Cuda`] on data transfer failure.
pub fn recommend_format<T: GpuFloat>(matrix: &CsrMatrix<T>) -> SparseResult<RecommendedFormat> {
    let stats = analyze_sparsity(matrix)?;

    if stats.rows == 0 {
        return Ok(RecommendedFormat::Csr);
    }

    // Priority: block > ELL > CSR5 > CSR
    if stats.block_detected && stats.block_size > 1 {
        return Ok(RecommendedFormat::Bsr);
    }

    if stats.cov_nnz_per_row < COV_ELL_THRESHOLD && stats.avg_nnz_per_row > 1.0 {
        return Ok(RecommendedFormat::Ell);
    }

    if stats.max_avg_ratio > MAX_AVG_CSR5_THRESHOLD {
        return Ok(RecommendedFormat::Csr5);
    }

    Ok(RecommendedFormat::Csr)
}

// ---------------------------------------------------------------------------
// Simple heuristic format selection (pure CPU, no GPU struct required)
// ---------------------------------------------------------------------------

/// Sparse matrix format selected by the simple heuristic.
///
/// This complements [`RecommendedFormat`] (which analyses a live GPU matrix)
/// by providing a lightweight, testable pure-function selector that takes
/// pre-computed statistics as inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpMatFormat {
    /// General Compressed Sparse Row. Best default for very sparse matrices
    /// (avg_nnz_per_row ≤ 2).
    Csr,
    /// ELLPACK. Best for matrices with regular row lengths and moderate fill
    /// (2 < avg_nnz_per_row ≤ 32).
    Ell,
    /// Hybrid ELL+COO. Best for matrices with irregular sparsity patterns
    /// that do not fit CSR5 criteria.
    Hyb,
    /// CSR5 load-balanced format. Best for large matrices with high average
    /// fill (avg_nnz_per_row > 128 and n_rows >= 1024).
    Csr5,
}

/// Select the recommended sparse format based on average nnz per row and
/// matrix size.
///
/// This is a pure heuristic function with no GPU I/O and no format conversion.
/// It is intended to guide decisions about *which format to build*, not to
/// dispatch an existing GPU matrix.
///
/// ## Selection rules (evaluated in order)
///
/// 1. `avg_nnz_per_row <= 2.0` → [`SpMatFormat::Csr`] (scalar kernel territory)
/// 2. `avg_nnz_per_row <= 32.0` → [`SpMatFormat::Ell`] (regular, coalesced ELL)
/// 3. `avg_nnz_per_row > 128.0 && n_rows >= 1024` → [`SpMatFormat::Csr5`] (load-balanced)
/// 4. Otherwise → [`SpMatFormat::Hyb`] (irregular overflow pattern)
#[must_use]
pub fn select_format(avg_nnz_per_row: f64, n_rows: usize) -> SpMatFormat {
    if avg_nnz_per_row <= 2.0 {
        SpMatFormat::Csr
    } else if avg_nnz_per_row <= 32.0 {
        SpMatFormat::Ell
    } else if avg_nnz_per_row > 128.0 && n_rows >= 1024 {
        SpMatFormat::Csr5
    } else {
        SpMatFormat::Hyb
    }
}

// ---------------------------------------------------------------------------
// Auto-dispatching SpMV
// ---------------------------------------------------------------------------

/// Auto-dispatching SpMV: selects the best CSR algorithm based on matrix
/// structure, then executes `y = alpha * A * x + beta * y`.
///
/// This currently dispatches to the CSR-based SpMV (scalar or vector) since
/// format conversion is expensive and should be done once, not per call.
/// The [`recommend_format`] function can guide the user to convert beforehand.
///
/// # Arguments
///
/// * `handle` -- Sparse handle.
/// * `matrix` -- CSR matrix `A`.
/// * `x` -- Device pointer to input vector `x`.
/// * `y` -- Device pointer to output vector `y`.
/// * `alpha` -- Scalar multiplier.
/// * `beta` -- Scalar multiplier for existing `y`.
///
/// # Errors
///
/// Returns errors from the underlying SpMV implementation.
#[allow(clippy::too_many_arguments)]
pub fn auto_spmv<T: GpuFloat>(
    handle: &SparseHandle,
    matrix: &CsrMatrix<T>,
    x: CUdeviceptr,
    y: CUdeviceptr,
    alpha: T,
    beta: T,
) -> SparseResult<()> {
    // Use adaptive algorithm selection from the CSR SpMV
    spmv(handle, SpMVAlgo::Adaptive, alpha, matrix, x, beta, y)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommended_format_display() {
        assert_eq!(format!("{}", RecommendedFormat::Csr), "CSR");
        assert_eq!(format!("{}", RecommendedFormat::Ell), "ELL");
        assert_eq!(format!("{}", RecommendedFormat::Bsr), "BSR");
        assert_eq!(format!("{}", RecommendedFormat::Csr5), "CSR5");
    }

    #[test]
    fn detect_blocks_too_small() {
        // Matrix too small for block detection
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let (detected, size) = detect_blocks(&row_ptr, &col_idx, 3, 3);
        assert!(!detected);
        assert_eq!(size, 1);
    }

    #[test]
    fn detect_blocks_diagonal_4x4() {
        // 4x4 diagonal: columns don't form blocks
        let row_ptr = vec![0, 1, 2, 3, 4];
        let col_idx = vec![0, 1, 2, 3];
        let (detected, _) = detect_blocks(&row_ptr, &col_idx, 4, 4);
        // Diagonal has same nnz but columns differ by 1, which matches bs=1
        // For bs=2: col[0]=0, col[1]=1, diff=1 != sub_row(1), so no block
        assert!(!detected);
    }

    #[test]
    fn detect_blocks_2x2_block_diagonal() {
        // 4x4 with 2x2 block diagonal:
        // [1 1 0 0]
        // [1 1 0 0]
        // [0 0 1 1]
        // [0 0 1 1]
        let row_ptr = vec![0, 2, 4, 6, 8];
        let col_idx = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let (detected, size) = detect_blocks(&row_ptr, &col_idx, 4, 4);
        assert!(detected);
        assert_eq!(size, 2);
    }

    #[test]
    fn cov_threshold_sanity() {
        const { assert!(COV_ELL_THRESHOLD > 0.0) };
        const { assert!(COV_ELL_THRESHOLD < 1.0) };
    }

    #[test]
    fn max_avg_threshold_sanity() {
        const { assert!(MAX_AVG_CSR5_THRESHOLD > 1.0) };
    }

    #[test]
    fn sparsity_stats_empty() {
        let stats = SparsityStats {
            rows: 0,
            nnz: 0,
            avg_nnz_per_row: 0.0,
            max_nnz_per_row: 0,
            min_nnz_per_row: 0,
            std_nnz_per_row: 0.0,
            cov_nnz_per_row: 0.0,
            max_avg_ratio: 0.0,
            block_detected: false,
            block_size: 1,
        };
        assert_eq!(stats.rows, 0);
    }

    #[test]
    fn detect_blocks_non_block() {
        // 4x4 with irregular sparsity
        let row_ptr = vec![0, 3, 5, 6, 8];
        let col_idx = vec![0, 1, 3, 1, 2, 2, 0, 3];
        let (detected, _) = detect_blocks(&row_ptr, &col_idx, 4, 4);
        assert!(!detected);
    }

    // --- select_format tests ---

    #[test]
    fn format_select_very_sparse_uses_csr() {
        // avg=1.0 is <= 2.0 => Csr
        assert_eq!(select_format(1.0, 1000), SpMatFormat::Csr);
    }

    #[test]
    fn format_select_boundary_2_uses_csr() {
        // avg=2.0 exactly => still Csr (boundary condition)
        assert_eq!(select_format(2.0, 1000), SpMatFormat::Csr);
    }

    #[test]
    fn format_select_regular_uses_ell() {
        // avg=8.0, 2.0 < 8.0 <= 32.0 => Ell
        assert_eq!(select_format(8.0, 1000), SpMatFormat::Ell);
    }

    #[test]
    fn format_select_boundary_32_uses_ell() {
        // avg=32.0 exactly => still Ell (boundary condition)
        assert_eq!(select_format(32.0, 1000), SpMatFormat::Ell);
    }

    #[test]
    fn format_select_irregular_uses_hyb() {
        // avg=50.0, n_rows=100: > 32.0 but not > 128.0 => Hyb
        assert_eq!(select_format(50.0, 100), SpMatFormat::Hyb);
    }

    #[test]
    fn format_select_large_but_insufficient_rows_uses_hyb() {
        // avg=200.0, n_rows=500 (< 1024) => Hyb, not Csr5
        assert_eq!(select_format(200.0, 500), SpMatFormat::Hyb);
    }

    #[test]
    fn format_select_large_dense_uses_csr5() {
        // avg=200.0, n_rows=2048: > 128.0 and n_rows >= 1024 => Csr5
        assert_eq!(select_format(200.0, 2048), SpMatFormat::Csr5);
    }

    #[test]
    fn format_select_exactly_128_avg_uses_hyb() {
        // avg=128.0 is NOT > 128.0 => Hyb (boundary)
        assert_eq!(select_format(128.0, 2048), SpMatFormat::Hyb);
    }
}
