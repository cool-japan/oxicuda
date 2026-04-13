//! Numerical accuracy test matrix patterns for GEMM correctness verification.
//!
//! This module provides helper functions that generate the canonical test
//! matrix patterns described in Vol.3 Sec 10.2 of the OxiCUDA blueprint:
//!
//! 1. **Random** — uniform values in [-1, 1] from a deterministic LCG seed.
//! 2. **Identity** — the `n×n` identity matrix.
//! 3. **Diagonal** — a diagonal matrix with a fixed non-zero value.
//! 4. **Ill-conditioned** — a diagonal matrix whose entries span many orders of
//!    magnitude, yielding a high condition number.
//! 5. **Zero** — all elements zero.
//! 6. **Ones** — all elements one.
//!
//! These helpers are intentionally `#[cfg(test)]`-only: they produce host-side
//! `Vec<f32>` data for algorithmic verification without touching GPU memory.

// ---------------------------------------------------------------------------
// Matrix generators
// ---------------------------------------------------------------------------

/// Generate a row-major matrix of `rows × cols` with uniform values in [-1, 1].
///
/// The generator uses a minimal 64-bit LCG so results are deterministic and
/// reproducible without introducing any PRNG crate dependency.
///
/// Layout: `mat[i * cols + j]` is element at row `i`, column `j`.
#[must_use]
pub fn random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let n = rows * cols;
    let mut state = seed.wrapping_add(1); // avoid zero state
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // LCG from Knuth: multiplier = 6364136223846793005, addend = 1442695040888963407.
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map upper bits to [0, 1) then scale to [-1, 1].
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        out.push((u * 2.0 - 1.0) as f32);
    }
    out
}

/// Generate the `n × n` identity matrix (row-major).
///
/// `identity[i * n + j] = 1` if `i == j`, else `0`.
#[must_use]
pub fn identity_matrix(n: usize) -> Vec<f32> {
    let mut mat = vec![0.0_f32; n * n];
    for i in 0..n {
        mat[i * n + i] = 1.0;
    }
    mat
}

/// Generate the `n × n` diagonal matrix with `diag_value` on the main diagonal.
///
/// `diag[i * n + j] = diag_value` if `i == j`, else `0`.
#[must_use]
pub fn diagonal_matrix(n: usize, diag_value: f32) -> Vec<f32> {
    let mut mat = vec![0.0_f32; n * n];
    for i in 0..n {
        mat[i * n + i] = diag_value;
    }
    mat
}

/// Generate an ill-conditioned `n × n` diagonal matrix.
///
/// The diagonal entries are spaced logarithmically from `1.0` to `1e6`,
/// so the 2-norm condition number is approximately `1e6` for `n >= 3`.
/// For `n == 1` the condition number is 1 (trivially well-conditioned).
/// For `n == 2` the condition number is `1e6`.
///
/// This pattern exercises numerical stability when solving linear systems or
/// computing matrix products that involve widely varying magnitudes.
#[must_use]
pub fn ill_conditioned_matrix(n: usize) -> Vec<f32> {
    let mut mat = vec![0.0_f32; n * n];
    for i in 0..n {
        let t = if n <= 1 {
            0.0_f64
        } else {
            i as f64 / (n - 1) as f64
        };
        // Exponential spacing from 1 (10^0) to 1e6 (10^6).
        let val = (10.0_f64).powf(t * 6.0);
        mat[i * n + i] = val as f32;
    }
    mat
}

/// Generate a `rows × cols` zero matrix.
#[must_use]
pub fn zero_matrix(rows: usize, cols: usize) -> Vec<f32> {
    vec![0.0_f32; rows * cols]
}

/// Generate a `rows × cols` matrix of all ones.
#[must_use]
pub fn ones_matrix(rows: usize, cols: usize) -> Vec<f32> {
    vec![1.0_f32; rows * cols]
}

// ---------------------------------------------------------------------------
// Utility: small dense matrix-vector product for tests
// ---------------------------------------------------------------------------

/// Compute the product of an `n × n` row-major matrix `mat` with vector `x`.
///
/// Returns `y` where `y[i] = sum_j mat[i*n + j] * x[j]`.
/// Panics if `mat.len() != n*n` or `x.len() != n`.
#[must_use]
pub fn matvec_square(mat: &[f32], x: &[f32]) -> Vec<f32> {
    let n = x.len();
    assert_eq!(
        mat.len(),
        n * n,
        "matvec_square: matrix size mismatch (got {}, expected {}×{}={})",
        mat.len(),
        n,
        n,
        n * n
    );
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| mat[i * n + j] * x[j])
                .fold(0.0_f32, |acc, v| acc + v)
        })
        .collect()
}

/// Compute the Frobenius norm of a matrix.
///
/// `frob(A) = sqrt(sum_ij A[i,j]^2)`.
#[must_use]
pub fn frobenius_norm(mat: &[f32]) -> f32 {
    mat.iter()
        .map(|x| x * x)
        .fold(0.0_f32, |acc, v| acc + v)
        .sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert two f32 slices are element-wise close within epsilon.
    fn assert_close(result: &[f32], expected: &[f32], epsilon: f32, label: &str) {
        assert_eq!(result.len(), expected.len(), "{label}: length mismatch");
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            let diff = (r - e).abs();
            assert!(
                diff <= epsilon,
                "{label}[{i}]: got {r}, expected {e}, diff={diff} > epsilon={epsilon}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // random_matrix
    // -----------------------------------------------------------------------

    #[test]
    fn random_matrix_has_correct_length() {
        let mat = random_matrix(4, 8, 42);
        assert_eq!(mat.len(), 32);
    }

    #[test]
    fn random_matrix_values_in_range() {
        let mat = random_matrix(16, 16, 7);
        for &v in &mat {
            assert!(
                (-1.0_f32..=1.0_f32).contains(&v),
                "random_matrix value {v} outside [-1, 1]"
            );
        }
    }

    #[test]
    fn random_matrix_deterministic() {
        let a = random_matrix(8, 8, 99);
        let b = random_matrix(8, 8, 99);
        assert_eq!(a, b, "same seed must produce identical matrices");
    }

    #[test]
    fn random_matrix_different_seeds_differ() {
        let a = random_matrix(8, 8, 1);
        let b = random_matrix(8, 8, 2);
        // With a good LCG, two seeds should produce different data.
        assert_ne!(a, b, "different seeds should produce different matrices");
    }

    // -----------------------------------------------------------------------
    // identity_matrix
    // -----------------------------------------------------------------------

    #[test]
    fn identity_matrix_diagonal_is_one() {
        let n = 5;
        let id = identity_matrix(n);
        for i in 0..n {
            assert_eq!(id[i * n + i], 1.0, "identity diagonal [{i},{i}] must be 1");
        }
    }

    #[test]
    fn identity_matrix_off_diagonal_is_zero() {
        let n = 4;
        let id = identity_matrix(n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert_eq!(
                        id[i * n + j],
                        0.0,
                        "identity off-diagonal [{i},{j}] must be 0"
                    );
                }
            }
        }
    }

    /// I @ x = x for a small vector.
    #[test]
    fn identity_matrix_times_vector() {
        let n = 4;
        let id = identity_matrix(n);
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let result = matvec_square(&id, &x);
        assert_close(&result, &x, 1e-7, "I @ x");
    }

    /// I @ I = I (matrix times itself via matvec on each column).
    #[test]
    fn identity_matrix_product_is_identity() {
        let n = 3;
        let id = identity_matrix(n);
        for col in 0..n {
            let e_col: Vec<f32> = (0..n).map(|i| if i == col { 1.0 } else { 0.0 }).collect();
            let result = matvec_square(&id, &e_col);
            assert_close(&result, &e_col, 1e-7, "I @ e_col");
        }
    }

    // -----------------------------------------------------------------------
    // diagonal_matrix
    // -----------------------------------------------------------------------

    #[test]
    fn diagonal_matrix_correct_values() {
        let n = 4;
        let d = 7.0_f32;
        let mat = diagonal_matrix(n, d);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { d } else { 0.0 };
                assert_eq!(mat[i * n + j], expected, "diagonal [{i},{j}]");
            }
        }
    }

    /// Frobenius norm of diag(d, d, ..., d) [n×n] = d * sqrt(n).
    #[test]
    fn diagonal_frobenius_norm() {
        let n = 9_usize;
        let d = 3.0_f32;
        let mat = diagonal_matrix(n, d);
        let frob = frobenius_norm(&mat);
        let expected = d * (n as f32).sqrt();
        assert!(
            (frob - expected).abs() <= 1e-5,
            "Frobenius norm of diag({d}) [{n}×{n}]: got {frob}, expected {expected}"
        );
    }

    /// D @ x = d * x for a diagonal matrix D = diag(d, d, ...).
    #[test]
    fn diagonal_matrix_times_vector() {
        let n = 4;
        let d = 2.5_f32;
        let mat = diagonal_matrix(n, d);
        let x = [1.0_f32, 2.0, -3.0, 4.0];
        let result = matvec_square(&mat, &x);
        let expected: Vec<f32> = x.iter().map(|xi| d * xi).collect();
        assert_close(&result, &expected, 1e-6, "D @ x");
    }

    // -----------------------------------------------------------------------
    // ill_conditioned_matrix
    // -----------------------------------------------------------------------

    /// Condition number estimate: max_diag / min_diag should be approximately 1e6.
    #[test]
    fn ill_conditioned_has_high_condition_number() {
        let n = 4;
        let mat = ill_conditioned_matrix(n);
        // Extract diagonal entries.
        let diag: Vec<f32> = (0..n).map(|i| mat[i * n + i]).collect();
        let max_d = diag.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_d = diag.iter().cloned().fold(f32::INFINITY, f32::min);
        let cond = max_d / min_d;
        assert!(
            cond >= 1e5,
            "ill_conditioned condition number={cond}, expected >= 1e5"
        );
    }

    /// Off-diagonal entries of the ill-conditioned matrix are all zero.
    #[test]
    fn ill_conditioned_off_diagonal_zero() {
        let n = 5;
        let mat = ill_conditioned_matrix(n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert_eq!(
                        mat[i * n + j],
                        0.0,
                        "ill_conditioned off-diagonal [{i},{j}] must be 0"
                    );
                }
            }
        }
    }

    /// Diagonal entries are all strictly positive.
    #[test]
    fn ill_conditioned_diagonal_positive() {
        let n = 6;
        let mat = ill_conditioned_matrix(n);
        for i in 0..n {
            let v = mat[i * n + i];
            assert!(
                v > 0.0,
                "ill_conditioned diagonal [{i},{i}]={v} must be > 0"
            );
        }
    }

    // -----------------------------------------------------------------------
    // zero_matrix
    // -----------------------------------------------------------------------

    #[test]
    fn zero_matrix_all_zeros() {
        let mat = zero_matrix(5, 7);
        assert_eq!(mat.len(), 35);
        for (idx, &v) in mat.iter().enumerate() {
            assert_eq!(v, 0.0, "zero_matrix[{idx}]={v}");
        }
    }

    /// Zero matrix times any vector gives zero vector.
    #[test]
    fn zero_matrix_times_vector_is_zero() {
        let n = 4;
        let mat = zero_matrix(n, n);
        let x = [1.0_f32, -2.0, 3.0, -4.0];
        let result = matvec_square(&mat, &x);
        let expected = vec![0.0_f32; n];
        assert_close(&result, &expected, 1e-7, "0 @ x");
    }

    /// Frobenius norm of zero matrix is zero.
    #[test]
    fn zero_matrix_frobenius_norm_is_zero() {
        let mat = zero_matrix(8, 8);
        let frob = frobenius_norm(&mat);
        assert_eq!(frob, 0.0, "Frobenius norm of zero matrix must be 0");
    }

    // -----------------------------------------------------------------------
    // ones_matrix
    // -----------------------------------------------------------------------

    #[test]
    fn ones_matrix_all_ones() {
        let mat = ones_matrix(3, 4);
        assert_eq!(mat.len(), 12);
        for (idx, &v) in mat.iter().enumerate() {
            assert_eq!(v, 1.0, "ones_matrix[{idx}]={v}");
        }
    }

    /// Frobenius norm of a ones matrix: sqrt(rows * cols).
    #[test]
    fn ones_matrix_frobenius_norm() {
        let rows = 4;
        let cols = 9;
        let mat = ones_matrix(rows, cols);
        let frob = frobenius_norm(&mat);
        let expected = ((rows * cols) as f32).sqrt();
        assert!(
            (frob - expected).abs() <= 1e-5,
            "Frobenius norm of ones_matrix({rows}×{cols}): got {frob}, expected {expected}"
        );
    }

    /// ones_matrix @ ones_vector = n * ones_vector (where ones_vector has n elements).
    #[test]
    fn ones_matrix_times_ones_vector() {
        let n = 5;
        let mat = ones_matrix(n, n);
        let x = vec![1.0_f32; n];
        let result = matvec_square(&mat, &x);
        let expected = vec![n as f32; n];
        assert_close(&result, &expected, 1e-6, "J @ 1");
    }

    // -----------------------------------------------------------------------
    // matvec_square utility
    // -----------------------------------------------------------------------

    #[test]
    fn matvec_square_2x2() {
        // A = [[1,2],[3,4]], x = [1,1] → y = [3, 7]
        let mat = [1.0_f32, 2.0, 3.0, 4.0];
        let x = [1.0_f32, 1.0];
        let result = matvec_square(&mat, &x);
        assert_close(&result, &[3.0, 7.0], 1e-7, "2x2 matvec");
    }

    #[test]
    fn matvec_square_3x3_rotation_like() {
        // A = [[0,-1,0],[1,0,0],[0,0,1]], x=[1,0,0] → [0,1,0]  (90° rotation in xy)
        let mat = [0.0_f32, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let x = [1.0_f32, 0.0, 0.0];
        let result = matvec_square(&mat, &x);
        assert_close(&result, &[0.0, 1.0, 0.0], 1e-7, "rotation matvec");
    }

    // -----------------------------------------------------------------------
    // frobenius_norm utility
    // -----------------------------------------------------------------------

    #[test]
    fn frobenius_norm_3_4_5_triangle() {
        // mat = [[3, 0], [0, 4]] → frob = sqrt(9+16) = 5
        let mat = [3.0_f32, 0.0, 0.0, 4.0];
        let frob = frobenius_norm(&mat);
        assert!(
            (frob - 5.0).abs() <= 1e-6,
            "Frobenius norm: got {frob}, expected 5"
        );
    }
}
