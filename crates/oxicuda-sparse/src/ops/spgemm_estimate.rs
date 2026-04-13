//! SpGEMM memory estimation for `C = A * B`.
//!
//! Pre-computes the output nnz (number of non-zeros) to avoid over-allocation
//! in the symbolic phase. Three estimation strategies are provided:
//!
//! 1. **Upper bound** -- O(nnz(A)) time, overestimates due to column collisions.
//! 2. **Hash-based exact** -- O(flops) time, exact count using hash sets / bitsets.
//! 3. **Sampling-based** -- sqrt(nrows) sampled rows, extrapolated with bounds.
//!
//! [`auto_estimate_spgemm`] selects the best strategy based on matrix dimensions.

use std::collections::HashSet;

use oxicuda_blas::GpuFloat;

use crate::error::{SparseError, SparseResult};
use crate::format::CsrMatrix;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The method used for SpGEMM nnz estimation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EstimationMethod {
    /// Per-row upper bound summing nnz of accessed B-rows.
    UpperBound,
    /// Exact count using hash sets (or bitsets for narrow matrices).
    Exact,
    /// Sampling-based estimate with the given number of sampled rows.
    Sampling {
        /// Number of rows actually sampled.
        sample_count: usize,
    },
}

/// Result of a SpGEMM nnz estimation.
#[derive(Debug, Clone)]
pub struct SpGEMMEstimate {
    /// Best estimate of nnz(C).
    pub estimated_nnz: usize,
    /// Lower bound on nnz(C).
    pub lower_bound: usize,
    /// Upper bound on nnz(C).
    pub upper_bound: usize,
    /// Which estimation method was used.
    pub method: EstimationMethod,
}

// ---------------------------------------------------------------------------
// Threshold constants
// ---------------------------------------------------------------------------

/// Matrices with fewer rows than this use exact counting in auto mode.
const SMALL_THRESHOLD: usize = 1_000;

/// Matrices with more rows than this use upper-bound in auto mode.
const LARGE_THRESHOLD: usize = 100_000;

/// Maximum column count for which we use a bitset instead of a hash set.
const BITSET_THRESHOLD: usize = 65_536;

// ---------------------------------------------------------------------------
// Dimension validation helper
// ---------------------------------------------------------------------------

fn validate_dims<T: GpuFloat>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> SparseResult<()> {
    if a.cols() != b.rows() {
        return Err(SparseError::DimensionMismatch(format!(
            "A.cols ({}) != B.rows ({})",
            a.cols(),
            b.rows()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 1. Upper-bound estimation
// ---------------------------------------------------------------------------

/// Computes an upper bound on nnz(C) where `C = A * B`.
///
/// For each row `i` of A, sums `nnz(B[j, :])` for every column `j` where
/// `A[i, j] != 0`. This overestimates because it does not account for column
/// collisions (multiple A-row entries mapping to the same C-column).
///
/// Time complexity: O(nnz(A)).
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `A.cols() != B.rows()`.
/// Returns [`SparseError::Cuda`] on device-to-host transfer failure.
pub fn estimate_nnz_upper_bound<T: GpuFloat>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
) -> SparseResult<SpGEMMEstimate> {
    validate_dims(a, b)?;

    let m = a.rows() as usize;
    if m == 0 {
        return Ok(SpGEMMEstimate {
            estimated_nnz: 0,
            lower_bound: 0,
            upper_bound: 0,
            method: EstimationMethod::UpperBound,
        });
    }

    let (a_row_ptr, a_col_idx, _) = a.to_host()?;
    let (b_row_ptr, _, _) = b.to_host()?;

    let n = b.cols() as usize;
    let mut total_upper: usize = 0;

    for row in 0..m {
        let a_start = a_row_ptr[row] as usize;
        let a_end = a_row_ptr[row + 1] as usize;
        let mut row_upper: usize = 0;

        for &col_val in &a_col_idx[a_start..a_end] {
            let k = col_val as usize;
            let b_nnz_k = (b_row_ptr[k + 1] - b_row_ptr[k]) as usize;
            row_upper += b_nnz_k;
        }

        // Clamp to n (a single row cannot have more than n non-zeros)
        total_upper += row_upper.min(n);
    }

    Ok(SpGEMMEstimate {
        estimated_nnz: total_upper,
        lower_bound: 0,
        upper_bound: total_upper,
        method: EstimationMethod::UpperBound,
    })
}

// ---------------------------------------------------------------------------
// 2. Hash-based exact count
// ---------------------------------------------------------------------------

/// Computes the exact nnz(C) where `C = A * B`.
///
/// For each row of C, uses a hash set (or a bitset for narrow matrices) to
/// track unique column indices produced by the row-wise dot product.
///
/// Time complexity: O(flops) where flops = sum of nnz(A\[i,:\]) * nnz(B\[j,:\]).
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `A.cols() != B.rows()`.
/// Returns [`SparseError::Cuda`] on device-to-host transfer failure.
pub fn count_nnz_exact<T: GpuFloat>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
) -> SparseResult<SpGEMMEstimate> {
    validate_dims(a, b)?;

    let m = a.rows() as usize;
    if m == 0 {
        return Ok(SpGEMMEstimate {
            estimated_nnz: 0,
            lower_bound: 0,
            upper_bound: 0,
            method: EstimationMethod::Exact,
        });
    }

    let (a_row_ptr, a_col_idx, _) = a.to_host()?;
    let (b_row_ptr, b_col_idx, _) = b.to_host()?;

    let n = b.cols() as usize;
    let use_bitset = n <= BITSET_THRESHOLD && n > 0;

    let mut total_nnz: usize = 0;

    if use_bitset {
        // Reuse a single bitset buffer across rows
        let words = n.div_ceil(64);
        let mut bitset = vec![0u64; words];

        for row in 0..m {
            // Clear the bitset
            for w in bitset.iter_mut() {
                *w = 0;
            }

            let a_start = a_row_ptr[row] as usize;
            let a_end = a_row_ptr[row + 1] as usize;

            for &a_col in &a_col_idx[a_start..a_end] {
                let k = a_col as usize;
                let b_start = b_row_ptr[k] as usize;
                let b_end = b_row_ptr[k + 1] as usize;

                for &b_col in &b_col_idx[b_start..b_end] {
                    let col = b_col as usize;
                    let word = col / 64;
                    let bit = col % 64;
                    bitset[word] |= 1u64 << bit;
                }
            }

            // Popcount the bitset
            let row_nnz: u32 = bitset.iter().map(|w| w.count_ones()).sum();
            total_nnz += row_nnz as usize;
        }
    } else {
        // Use hash set for wide matrices
        let mut set = HashSet::new();

        for row in 0..m {
            set.clear();

            let a_start = a_row_ptr[row] as usize;
            let a_end = a_row_ptr[row + 1] as usize;

            for &a_col in &a_col_idx[a_start..a_end] {
                let k = a_col as usize;
                let b_start = b_row_ptr[k] as usize;
                let b_end = b_row_ptr[k + 1] as usize;

                for &b_col in &b_col_idx[b_start..b_end] {
                    set.insert(b_col);
                }
            }

            total_nnz += set.len();
        }
    }

    Ok(SpGEMMEstimate {
        estimated_nnz: total_nnz,
        lower_bound: total_nnz,
        upper_bound: total_nnz,
        method: EstimationMethod::Exact,
    })
}

// ---------------------------------------------------------------------------
// 3. Sampling-based estimation
// ---------------------------------------------------------------------------

/// Estimates nnz(C) by sampling a subset of rows and extrapolating.
///
/// Samples approximately `sqrt(nrows)` rows (at least 1, at most nrows),
/// computes exact nnz for those rows, and extrapolates to the full matrix
/// with conservative lower and upper confidence bounds.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `A.cols() != B.rows()`.
/// Returns [`SparseError::Cuda`] on device-to-host transfer failure.
pub fn estimate_nnz_sampling<T: GpuFloat>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
) -> SparseResult<SpGEMMEstimate> {
    validate_dims(a, b)?;

    let m = a.rows() as usize;
    if m == 0 {
        return Ok(SpGEMMEstimate {
            estimated_nnz: 0,
            lower_bound: 0,
            upper_bound: 0,
            method: EstimationMethod::Sampling { sample_count: 0 },
        });
    }

    let (a_row_ptr, a_col_idx, _) = a.to_host()?;
    let (b_row_ptr, b_col_idx, _) = b.to_host()?;

    let n = b.cols() as usize;

    // Determine sample count: sqrt(m), clamped to [1, m]
    let sample_count = (m as f64).sqrt().ceil() as usize;
    let sample_count = sample_count.max(1).min(m);

    // Deterministic sampling: evenly spaced rows
    let sample_indices = deterministic_sample_indices(m, sample_count);
    let actual_sample_count = sample_indices.len();

    // Compute exact nnz for each sampled row
    let use_bitset = n <= BITSET_THRESHOLD && n > 0;
    let mut row_nnz_samples = Vec::with_capacity(actual_sample_count);

    if use_bitset {
        let words = n.div_ceil(64);
        let mut bitset = vec![0u64; words];

        for &row in &sample_indices {
            for w in bitset.iter_mut() {
                *w = 0;
            }

            let a_start = a_row_ptr[row] as usize;
            let a_end = a_row_ptr[row + 1] as usize;

            for &a_col in &a_col_idx[a_start..a_end] {
                let k = a_col as usize;
                let b_start = b_row_ptr[k] as usize;
                let b_end = b_row_ptr[k + 1] as usize;

                for &b_col in &b_col_idx[b_start..b_end] {
                    let col = b_col as usize;
                    let word = col / 64;
                    let bit = col % 64;
                    bitset[word] |= 1u64 << bit;
                }
            }

            let row_nnz: u32 = bitset.iter().map(|w| w.count_ones()).sum();
            row_nnz_samples.push(row_nnz as f64);
        }
    } else {
        let mut set = HashSet::new();

        for &row in &sample_indices {
            set.clear();

            let a_start = a_row_ptr[row] as usize;
            let a_end = a_row_ptr[row + 1] as usize;

            for &a_col in &a_col_idx[a_start..a_end] {
                let k = a_col as usize;
                let b_start = b_row_ptr[k] as usize;
                let b_end = b_row_ptr[k + 1] as usize;

                for &b_col in &b_col_idx[b_start..b_end] {
                    set.insert(b_col);
                }
            }

            row_nnz_samples.push(set.len() as f64);
        }
    }

    // Compute statistics
    let (mean, std_dev) = compute_mean_stddev(&row_nnz_samples);

    let estimated_nnz = (mean * m as f64).round() as usize;

    // Conservative bounds using 2-sigma (approx 95% confidence)
    let margin = 2.0 * std_dev * (m as f64) / (actual_sample_count as f64).sqrt();
    let lower_raw = (mean * m as f64 - margin).max(0.0);
    let upper_raw = mean * m as f64 + margin;

    // Clamp upper bound to theoretical maximum m * n
    let max_possible = m.saturating_mul(n);
    let lower_bound = lower_raw.round() as usize;
    let upper_bound = (upper_raw.round() as usize).min(max_possible);

    Ok(SpGEMMEstimate {
        estimated_nnz: estimated_nnz.min(max_possible),
        lower_bound,
        upper_bound,
        method: EstimationMethod::Sampling {
            sample_count: actual_sample_count,
        },
    })
}

// ---------------------------------------------------------------------------
// Auto-selection
// ---------------------------------------------------------------------------

/// Estimates workspace memory needed for SpGEMM `C = A * B`.
///
/// This is an alias for [`auto_estimate_spgemm`].
pub fn estimate_spgemm_memory<T: GpuFloat>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
) -> SparseResult<SpGEMMEstimate> {
    auto_estimate_spgemm(a, b)
}

/// Chooses the best estimation strategy based on matrix sizes and computes
/// the nnz estimate for `C = A * B`.
///
/// - Small matrices (< 1000 rows): exact hash-based count.
/// - Medium matrices (1000 -- 100K rows): sampling-based estimate.
/// - Large matrices (> 100K rows): upper-bound estimate.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `A.cols() != B.rows()`.
/// Returns [`SparseError::Cuda`] on device-to-host transfer failure.
pub fn auto_estimate_spgemm<T: GpuFloat>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
) -> SparseResult<SpGEMMEstimate> {
    validate_dims(a, b)?;

    let m = a.rows() as usize;

    if m < SMALL_THRESHOLD {
        count_nnz_exact(a, b)
    } else if m <= LARGE_THRESHOLD {
        estimate_nnz_sampling(a, b)
    } else {
        estimate_nnz_upper_bound(a, b)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Produces `count` deterministic, evenly-spaced sample indices from `[0, total)`.
fn deterministic_sample_indices(total: usize, count: usize) -> Vec<usize> {
    if count >= total {
        return (0..total).collect();
    }
    if count == 0 {
        return Vec::new();
    }

    let mut indices = Vec::with_capacity(count);
    for i in 0..count {
        // Evenly spaced: floor(i * total / count)
        let idx = i * total / count;
        indices.push(idx);
    }
    indices
}

/// Computes mean and sample standard deviation of a slice.
///
/// Returns `(0.0, 0.0)` for empty slices and `(val, 0.0)` for single-element
/// slices.
fn compute_mean_stddev(samples: &[f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }

    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;

    if samples.len() == 1 {
        return (mean, 0.0);
    }

    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, variance.sqrt())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a small CSR matrix on the host for testing.
    /// Returns an error if no GPU context is available, so tests that call
    /// this are effectively integration tests requiring a GPU.
    ///
    /// For unit tests that do NOT need a GPU we test the internal helpers.
    #[cfg(feature = "gpu-tests")]
    fn try_make_csr(
        rows: u32,
        cols: u32,
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[f64],
    ) -> SparseResult<CsrMatrix<f64>> {
        CsrMatrix::from_host(rows, cols, row_ptr, col_idx, values)
    }

    // ------------------------------------------------------------------
    // Internal helper tests (no GPU required)
    // ------------------------------------------------------------------

    #[test]
    fn deterministic_sample_full() {
        let indices = deterministic_sample_indices(5, 10);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn deterministic_sample_subset() {
        let indices = deterministic_sample_indices(10, 3);
        assert_eq!(indices.len(), 3);
        // Evenly spaced: 0, 3, 6
        assert_eq!(indices, vec![0, 3, 6]);
    }

    #[test]
    fn deterministic_sample_zero() {
        let indices = deterministic_sample_indices(10, 0);
        assert!(indices.is_empty());
    }

    #[test]
    fn deterministic_sample_one() {
        let indices = deterministic_sample_indices(100, 1);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn mean_stddev_empty() {
        let (mean, std) = compute_mean_stddev(&[]);
        assert!((mean - 0.0).abs() < f64::EPSILON);
        assert!((std - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mean_stddev_single() {
        let (mean, std) = compute_mean_stddev(&[42.0]);
        assert!((mean - 42.0).abs() < f64::EPSILON);
        assert!((std - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mean_stddev_basic() {
        // [2, 4, 6, 8] => mean = 5, sample std = sqrt(20/3) ~= 2.5820
        let (mean, std) = compute_mean_stddev(&[2.0, 4.0, 6.0, 8.0]);
        assert!((mean - 5.0).abs() < 1e-10);
        let expected_std = (20.0_f64 / 3.0).sqrt();
        assert!((std - expected_std).abs() < 1e-10);
    }

    #[test]
    fn mean_stddev_uniform() {
        // All same => stddev 0
        let (mean, std) = compute_mean_stddev(&[7.0, 7.0, 7.0]);
        assert!((mean - 7.0).abs() < f64::EPSILON);
        assert!((std - 0.0).abs() < 1e-15);
    }

    #[test]
    fn estimation_method_debug() {
        let m = EstimationMethod::UpperBound;
        let s = format!("{:?}", m);
        assert!(s.contains("UpperBound"));

        let m2 = EstimationMethod::Sampling { sample_count: 10 };
        let s2 = format!("{:?}", m2);
        assert!(s2.contains("10"));
    }

    #[test]
    fn estimation_method_eq() {
        assert_eq!(EstimationMethod::Exact, EstimationMethod::Exact);
        assert_ne!(EstimationMethod::Exact, EstimationMethod::UpperBound);
        assert_eq!(
            EstimationMethod::Sampling { sample_count: 5 },
            EstimationMethod::Sampling { sample_count: 5 }
        );
        assert_ne!(
            EstimationMethod::Sampling { sample_count: 5 },
            EstimationMethod::Sampling { sample_count: 6 }
        );
    }

    #[test]
    fn estimate_clone() {
        let est = SpGEMMEstimate {
            estimated_nnz: 100,
            lower_bound: 80,
            upper_bound: 120,
            method: EstimationMethod::Exact,
        };
        let cloned = est.clone();
        assert_eq!(cloned.estimated_nnz, 100);
        assert_eq!(cloned.lower_bound, 80);
        assert_eq!(cloned.upper_bound, 120);
        assert_eq!(cloned.method, EstimationMethod::Exact);
    }

    // ------------------------------------------------------------------
    // GPU integration tests (behind feature gate)
    // ------------------------------------------------------------------

    #[cfg(feature = "gpu-tests")]
    mod gpu {
        use super::*;

        /// Create a live CUDA context for the calling test thread, or return
        /// `None` if no GPU is available.
        ///
        /// The CUDA driver API (`cuMemAlloc`, etc.) requires an *active* context
        /// on the **calling thread** — simply checking device count is not enough.
        /// `Context::new` calls `cuCtxCreate` which both creates and sets the
        /// context as current for the calling thread.
        ///
        /// Callers must hold the returned `Context` alive for the duration of
        /// the test:
        ///
        /// ```ignore
        /// let Some(_ctx) = gpu_context() else { return; };
        /// // GPU operations here ...
        /// ```
        fn gpu_context() -> Option<oxicuda_driver::Context> {
            oxicuda_driver::init().ok()?;
            if oxicuda_driver::Device::count().ok()? == 0 {
                return None;
            }
            let dev = oxicuda_driver::Device::get(0).ok()?;
            oxicuda_driver::Context::new(&dev).ok()
        }

        /// Identity 3x3 matrix: C = I * I = I, nnz = 3.
        fn identity_3x3() -> SparseResult<CsrMatrix<f64>> {
            try_make_csr(3, 3, &[0, 1, 2, 3], &[0, 1, 2], &[1.0, 1.0, 1.0])
        }

        /// Diagonal 4x4 matrix.
        fn diagonal_4x4() -> SparseResult<CsrMatrix<f64>> {
            try_make_csr(4, 4, &[0, 1, 2, 3, 4], &[0, 1, 2, 3], &[2.0, 3.0, 4.0, 5.0])
        }

        /// Dense-ish 2x3 matrix:
        /// [1 2 3]
        /// [4 5 0]
        fn dense_2x3() -> SparseResult<CsrMatrix<f64>> {
            try_make_csr(
                2,
                3,
                &[0, 3, 5],
                &[0, 1, 2, 0, 1],
                &[1.0, 2.0, 3.0, 4.0, 5.0],
            )
        }

        /// 3x2 matrix:
        /// [1 0]
        /// [0 2]
        /// [3 4]
        fn sparse_3x2() -> SparseResult<CsrMatrix<f64>> {
            try_make_csr(3, 2, &[0, 1, 2, 4], &[0, 1, 0, 1], &[1.0, 2.0, 3.0, 4.0])
        }

        /// Single-row matrix: [1 0 2]
        fn single_row() -> SparseResult<CsrMatrix<f64>> {
            try_make_csr(1, 3, &[0, 2], &[0, 2], &[1.0, 2.0])
        }

        /// Single-column matrix:
        /// [1]
        /// [2]
        /// [3]
        fn single_col() -> SparseResult<CsrMatrix<f64>> {
            try_make_csr(3, 1, &[0, 1, 2, 3], &[0, 0, 0], &[1.0, 2.0, 3.0])
        }

        #[test]
        fn upper_bound_identity() {
            let Some(_ctx) = gpu_context() else {
                return;
            };
            let a = identity_3x3().expect("test: GPU context required");
            let est = estimate_nnz_upper_bound(&a, &a)
                .expect("test: upper bound estimation should succeed");
            // I * I = I, exact nnz = 3, upper bound >= 3
            assert!(est.upper_bound >= 3);
            assert_eq!(est.method, EstimationMethod::UpperBound);
        }

        #[test]
        fn exact_identity() {
            let Some(_ctx) = gpu_context() else {
                return;
            };
            let a = identity_3x3().expect("test: GPU context required");
            let est = count_nnz_exact(&a, &a).expect("test: exact counting should succeed");
            assert_eq!(est.estimated_nnz, 3);
            assert_eq!(est.lower_bound, 3);
            assert_eq!(est.upper_bound, 3);
            assert_eq!(est.method, EstimationMethod::Exact);
        }

        #[test]
        fn exact_diagonal() {
            let Some(_ctx) = gpu_context() else {
                return;
            };
            let a = diagonal_4x4().expect("test: GPU context required");
            let est = count_nnz_exact(&a, &a).expect("test: exact counting should succeed");
            // D * D = D^2, still diagonal => nnz = 4
            assert_eq!(est.estimated_nnz, 4);
        }

        #[test]
        fn upper_bound_ge_exact() {
            let Some(_ctx) = gpu_context() else {
                return;
            };
            let a = dense_2x3().expect("test: GPU context required");
            let b = sparse_3x2().expect("test: GPU context required");

            let exact = count_nnz_exact(&a, &b).expect("test: exact counting should succeed");
            let upper = estimate_nnz_upper_bound(&a, &b).expect("test: upper bound should succeed");

            assert!(
                upper.upper_bound >= exact.estimated_nnz,
                "upper bound ({}) must be >= exact ({})",
                upper.upper_bound,
                exact.estimated_nnz
            );
        }

        #[test]
        fn sampling_identity() {
            let Some(_ctx) = gpu_context() else {
                return;
            };
            let a = identity_3x3().expect("test: GPU context required");
            let est =
                estimate_nnz_sampling(&a, &a).expect("test: sampling estimation should succeed");
            // With only 3 rows, sample_count = ceil(sqrt(3)) = 2
            // But for 3 rows sample covers most/all, should be close to 3
            assert!(est.estimated_nnz >= 1);
            assert!(est.upper_bound >= est.lower_bound);
            matches!(est.method, EstimationMethod::Sampling { .. });
        }

        #[test]
        fn auto_small_uses_exact() {
            let Some(_ctx) = gpu_context() else {
                return;
            };
            let a = identity_3x3().expect("test: GPU context required");
            let est = auto_estimate_spgemm(&a, &a).expect("test: auto estimation should succeed");
            // 3 rows < 1000 => should pick exact
            assert_eq!(est.method, EstimationMethod::Exact);
            assert_eq!(est.estimated_nnz, 3);
        }

        #[test]
        fn dimension_mismatch_error() {
            let Some(_ctx) = gpu_context() else {
                return;
            };
            let a = dense_2x3().expect("test: GPU context required");
            let b = dense_2x3().expect("test: GPU context required");
            // A is 2x3, B is 2x3 => A.cols (3) != B.rows (2)
            let err = estimate_nnz_upper_bound(&a, &b);
            assert!(err.is_err());
            let err = count_nnz_exact(&a, &b);
            assert!(err.is_err());
            let err = estimate_nnz_sampling(&a, &b);
            assert!(err.is_err());
            let err = auto_estimate_spgemm(&a, &b);
            assert!(err.is_err());
        }

        #[test]
        fn single_row_matrix() {
            let Some(_ctx) = gpu_context() else {
                return;
            };
            let a = single_row().expect("test: GPU context required");
            // a is 1x3 with nnz=2 at cols 0 and 2
            // Need a 3x? matrix for b
            let b = try_make_csr(3, 2, &[0, 1, 1, 2], &[0, 1], &[1.0, 1.0])
                .expect("test: GPU context required");
            // C = a * b: row 0 picks B rows 0 and 2
            //   B row 0 has col 0, B row 2 has col 1 => C row 0 has cols {0, 1} => nnz = 2
            let exact = count_nnz_exact(&a, &b).expect("test: exact counting should succeed");
            assert_eq!(exact.estimated_nnz, 2);
        }

        #[test]
        fn single_col_times_single_row() {
            let Some(_ctx) = gpu_context() else {
                return;
            };
            let a = single_col().expect("test: GPU context required");
            let b = single_row().expect("test: GPU context required");
            // A is 3x1, B is 1x3 => C is 3x3
            // Each row of A has one entry in col 0, B row 0 has 2 entries
            // => each C row has 2 entries => nnz = 6
            let exact = count_nnz_exact(&a, &b).expect("test: exact counting should succeed");
            assert_eq!(exact.estimated_nnz, 6);

            let upper = estimate_nnz_upper_bound(&a, &b).expect("test: upper bound should succeed");
            assert!(upper.upper_bound >= 6);
        }

        #[test]
        fn estimate_spgemm_memory_alias() {
            let Some(_ctx) = gpu_context() else {
                return;
            };
            let a = identity_3x3().expect("test: GPU context required");
            let est =
                estimate_spgemm_memory(&a, &a).expect("test: memory estimation should succeed");
            // Should behave like auto_estimate
            assert_eq!(est.estimated_nnz, 3);
        }

        #[test]
        fn sampling_bounds_consistency() {
            let Some(_ctx) = gpu_context() else {
                return;
            };
            let a = diagonal_4x4().expect("test: GPU context required");
            let est = estimate_nnz_sampling(&a, &a).expect("test: sampling should succeed");
            assert!(
                est.lower_bound <= est.estimated_nnz,
                "lower_bound ({}) should be <= estimated_nnz ({})",
                est.lower_bound,
                est.estimated_nnz
            );
            assert!(
                est.estimated_nnz <= est.upper_bound,
                "estimated_nnz ({}) should be <= upper_bound ({})",
                est.estimated_nnz,
                est.upper_bound
            );
        }
    }
}
