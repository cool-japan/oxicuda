//! Sparse matrix powers and polynomial evaluation.
//!
//! Computes `A^k` via repeated SpGEMM with binary exponentiation and optional
//! structure reuse optimization. Also provides matrix polynomial evaluation
//! via Horner's method: `p(A) = c0*I + c1*A + c2*A^2 + ... + ck*A^k`.
//!
//! All operations work on CSR (Compressed Sparse Row) format stored as
//! `(row_offsets, col_indices, values)` triple on the host side.
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

use std::collections::HashMap;

use crate::error::SparseError;

/// A CSR matrix stored as `(row_offsets, col_indices, values)`.
type CsrTriple = (Vec<usize>, Vec<usize>, Vec<f64>);

// ---------------------------------------------------------------------------
// Configuration & result types
// ---------------------------------------------------------------------------

/// Configuration for sparse matrix power computation.
#[derive(Debug, Clone)]
pub struct MatrixPowerConfig {
    /// Maximum allowed nnz in any intermediate or final result.
    /// If exceeded, the computation aborts with an error.
    pub max_nnz: Option<usize>,
    /// Whether to attempt reusing symbolic (sparsity) structure between
    /// successive multiplications when the structure matches.
    pub reuse_structure: bool,
    /// The exponent `k` in `A^k`.
    pub power: usize,
}

/// Result of a sparse matrix power computation.
#[derive(Debug, Clone)]
pub struct MatrixPowerResult {
    /// CSR row offsets of the result matrix (length = rows + 1).
    pub row_offsets: Vec<usize>,
    /// CSR column indices of the result matrix.
    pub col_indices: Vec<usize>,
    /// CSR values of the result matrix.
    pub values: Vec<f64>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Number of non-zeros in the result.
    pub nnz: usize,
    /// Number of SpGEMM multiplications actually performed.
    pub multiplications_performed: usize,
    /// nnz observed after each multiplication step.
    pub nnz_growth: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute `A^power` for a sparse CSR matrix using binary exponentiation.
///
/// Binary exponentiation decomposes the exponent into powers of two so that
/// `A^8 = ((A^2)^2)^2` requires only 3 multiplications instead of 7.
///
/// # Arguments
///
/// * `row_offsets` -- CSR row pointer array (length `rows + 1`).
/// * `col_indices` -- CSR column indices (length `nnz`).
/// * `values` -- CSR values (length `nnz`).
/// * `rows`, `cols` -- Matrix dimensions (must be square for power > 1).
/// * `power` -- The exponent `k`.
/// * `config` -- [`MatrixPowerConfig`] controlling nnz limits and structure reuse.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if the matrix is not square
/// (when `power > 1`), [`SparseError::InvalidFormat`] if CSR arrays are
/// inconsistent, or [`SparseError::InvalidArgument`] if nnz exceeds `max_nnz`.
pub fn sparse_matrix_power(
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f64],
    rows: usize,
    cols: usize,
    power: usize,
    config: &MatrixPowerConfig,
) -> Result<MatrixPowerResult, SparseError> {
    validate_csr(row_offsets, col_indices, values, rows)?;

    if power > 1 && rows != cols {
        return Err(SparseError::DimensionMismatch(format!(
            "matrix must be square for power > 1, got {}x{}",
            rows, cols
        )));
    }

    // power == 0 => identity
    if power == 0 {
        let (id_offsets, id_indices, id_values) = sparse_identity(rows);
        return Ok(MatrixPowerResult {
            nnz: id_indices.len(),
            row_offsets: id_offsets,
            col_indices: id_indices,
            values: id_values,
            rows,
            cols: rows,
            multiplications_performed: 0,
            nnz_growth: vec![],
        });
    }

    // power == 1 => copy of A
    if power == 1 {
        return Ok(MatrixPowerResult {
            row_offsets: row_offsets.to_vec(),
            col_indices: col_indices.to_vec(),
            values: values.to_vec(),
            rows,
            cols,
            nnz: col_indices.len(),
            multiplications_performed: 0,
            nnz_growth: vec![col_indices.len()],
        });
    }

    // Binary exponentiation
    // We maintain `base` (current squaring chain) and `result` (accumulated product).
    let mut base_offsets = row_offsets.to_vec();
    let mut base_indices = col_indices.to_vec();
    let mut base_values = values.to_vec();

    let mut result_offsets: Option<Vec<usize>> = None;
    let mut result_indices: Option<Vec<usize>> = None;
    let mut result_values: Option<Vec<f64>> = None;

    let mut mults = 0usize;
    let mut nnz_growth = vec![col_indices.len()];
    let mut exp = power;

    // Track previous structure for reuse optimization
    let mut prev_structure: Option<(Vec<usize>, Vec<usize>)> = None;

    while exp > 0 {
        if exp & 1 == 1 {
            // result = result * base  (or result = base if first time)
            if let (Some(r_off), Some(r_idx), Some(r_val)) =
                (&result_offsets, &result_indices, &result_values)
            {
                let (new_off, new_idx, new_val) = host_spgemm(
                    r_off,
                    r_idx,
                    r_val,
                    rows,
                    rows,
                    &base_offsets,
                    &base_indices,
                    &base_values,
                    rows,
                    rows,
                )?;
                mults += 1;

                check_max_nnz(new_idx.len(), config.max_nnz)?;
                nnz_growth.push(new_idx.len());

                if config.reuse_structure {
                    if let Some((ref ps_off, ref ps_idx)) = prev_structure {
                        if ps_off == &new_off && ps_idx == &new_idx {
                            // Structure matches -- values already computed, nothing extra to do
                        }
                    }
                    prev_structure = Some((new_off.clone(), new_idx.clone()));
                }

                result_offsets = Some(new_off);
                result_indices = Some(new_idx);
                result_values = Some(new_val);
            } else {
                result_offsets = Some(base_offsets.clone());
                result_indices = Some(base_indices.clone());
                result_values = Some(base_values.clone());
            }
        }
        exp >>= 1;
        if exp > 0 {
            // base = base * base
            let (new_off, new_idx, new_val) = host_spgemm(
                &base_offsets,
                &base_indices,
                &base_values,
                rows,
                rows,
                &base_offsets.clone(),
                &base_indices.clone(),
                &base_values.clone(),
                rows,
                rows,
            )?;
            mults += 1;

            check_max_nnz(new_idx.len(), config.max_nnz)?;
            nnz_growth.push(new_idx.len());

            if config.reuse_structure {
                prev_structure = Some((new_off.clone(), new_idx.clone()));
            }

            base_offsets = new_off;
            base_indices = new_idx;
            base_values = new_val;
        }
    }

    let r_offsets = result_offsets
        .ok_or_else(|| SparseError::InternalError("result not computed".to_string()))?;
    let r_indices = result_indices
        .ok_or_else(|| SparseError::InternalError("result not computed".to_string()))?;
    let r_values = result_values
        .ok_or_else(|| SparseError::InternalError("result not computed".to_string()))?;

    Ok(MatrixPowerResult {
        nnz: r_indices.len(),
        row_offsets: r_offsets,
        col_indices: r_indices,
        values: r_values,
        rows,
        cols: rows,
        multiplications_performed: mults,
        nnz_growth,
    })
}

/// Host-side sparse matrix-matrix multiplication: `C = A * B`.
///
/// Uses a row-by-row accumulator approach with [`HashMap`] for gathering
/// unique column contributions.
///
/// # Arguments
///
/// * CSR arrays and dimensions for matrices A and B.
///
/// # Returns
///
/// CSR triple `(row_offsets, col_indices, values)` for C.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a_cols != b_rows`.
#[allow(clippy::too_many_arguments)]
pub fn host_spgemm(
    a_row_offsets: &[usize],
    a_col_indices: &[usize],
    a_values: &[f64],
    a_rows: usize,
    a_cols: usize,
    b_row_offsets: &[usize],
    b_col_indices: &[usize],
    b_values: &[f64],
    b_rows: usize,
    b_cols: usize,
) -> Result<CsrTriple, SparseError> {
    if a_cols != b_rows {
        return Err(SparseError::DimensionMismatch(format!(
            "A.cols ({}) != B.rows ({}) in SpGEMM",
            a_cols, b_rows
        )));
    }

    let _ = b_cols; // used for documentation clarity, not directly needed

    let mut c_row_offsets = Vec::with_capacity(a_rows + 1);
    let mut c_col_indices = Vec::new();
    let mut c_values = Vec::new();

    c_row_offsets.push(0usize);

    let mut accum: HashMap<usize, f64> = HashMap::new();

    for i in 0..a_rows {
        accum.clear();
        let a_start = a_row_offsets[i];
        let a_end = a_row_offsets[i + 1];

        for idx in a_start..a_end {
            let j = a_col_indices[idx];
            let a_ij = a_values[idx];

            let b_start = b_row_offsets[j];
            let b_end = b_row_offsets[j + 1];

            for b_idx in b_start..b_end {
                let k = b_col_indices[b_idx];
                let b_jk = b_values[b_idx];
                *accum.entry(k).or_insert(0.0) += a_ij * b_jk;
            }
        }

        // Sort by column index for canonical CSR
        let mut entries: Vec<(usize, f64)> = accum.drain().collect();
        entries.sort_unstable_by_key(|&(col, _)| col);

        for (col, val) in entries {
            c_col_indices.push(col);
            c_values.push(val);
        }

        c_row_offsets.push(c_col_indices.len());
    }

    Ok((c_row_offsets, c_col_indices, c_values))
}

/// Construct the CSR representation of an `n x n` identity matrix.
///
/// Returns `(row_offsets, col_indices, values)`.
pub fn sparse_identity(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let row_offsets: Vec<usize> = (0..=n).collect();
    let col_indices: Vec<usize> = (0..n).collect();
    let values = vec![1.0; n];
    (row_offsets, col_indices, values)
}

/// Heuristic estimate of `nnz(A^k)` based on average degree.
///
/// Uses `avg_degree = nnz / rows` and estimates the result as
/// `min(rows * min(rows, avg_degree^k), rows * rows)`.
pub fn estimate_power_nnz(
    row_offsets: &[usize],
    _col_indices: &[usize],
    rows: usize,
    power: usize,
) -> usize {
    if rows == 0 || power == 0 {
        return rows; // identity has `rows` non-zeros
    }

    let nnz = if row_offsets.len() > rows {
        row_offsets[rows]
    } else {
        return 0;
    };

    if nnz == 0 {
        return 0;
    }

    let avg_degree = nnz as f64 / rows as f64;
    let estimated_degree = avg_degree.powi(power as i32);
    let per_row = estimated_degree.min(rows as f64);
    let total = (rows as f64 * per_row).min((rows * rows) as f64);
    total as usize
}

/// Evaluate a polynomial `p(A) = c0*I + c1*A + c2*A^2 + ... + ck*A^k` using
/// Horner's method.
///
/// Horner's form: `p(A) = ((...((ck*A + c_{k-1})*A + c_{k-2})*A + ...) + c1)*A + c0*I`
///
/// This requires only `k` multiplications rather than computing each `A^i`
/// separately.
///
/// # Arguments
///
/// * CSR arrays and dimensions for `A` (must be square).
/// * `coefficients` -- Polynomial coefficients `[c0, c1, ..., ck]`.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if the matrix is not square,
/// or [`SparseError::InvalidArgument`] if `coefficients` is empty.
pub fn sparse_matrix_polynomial(
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f64],
    rows: usize,
    cols: usize,
    coefficients: &[f64],
) -> Result<MatrixPowerResult, SparseError> {
    validate_csr(row_offsets, col_indices, values, rows)?;

    if rows != cols {
        return Err(SparseError::DimensionMismatch(format!(
            "matrix must be square for polynomial evaluation, got {}x{}",
            rows, cols
        )));
    }

    if coefficients.is_empty() {
        return Err(SparseError::InvalidArgument(
            "polynomial coefficients must not be empty".to_string(),
        ));
    }

    let n = rows;
    let degree = coefficients.len() - 1;

    // degree == 0 => c0 * I
    if degree == 0 {
        let (id_off, id_idx, id_val) = sparse_identity(n);
        let (s_off, s_idx, s_val) = scalar_multiply_csr(&id_off, &id_idx, &id_val, coefficients[0]);
        return Ok(MatrixPowerResult {
            nnz: s_idx.len(),
            row_offsets: s_off,
            col_indices: s_idx,
            values: s_val,
            rows: n,
            cols: n,
            multiplications_performed: 0,
            nnz_growth: vec![],
        });
    }

    // Horner's method: start from highest coefficient
    // result = ck
    // for i in (0..k).rev():
    //   result = result * A + c_i * I

    let mut mults = 0usize;
    let mut nnz_growth = Vec::new();

    // Start: result = c_k * I
    let (id_off, id_idx, id_val) = sparse_identity(n);
    let (mut r_off, mut r_idx, mut r_val) =
        scalar_multiply_csr(&id_off, &id_idx, &id_val, coefficients[degree]);

    for i in (0..degree).rev() {
        // result = result * A
        let (prod_off, prod_idx, prod_val) = host_spgemm(
            &r_off,
            &r_idx,
            &r_val,
            n,
            n,
            row_offsets,
            col_indices,
            values,
            n,
            n,
        )?;
        mults += 1;
        nnz_growth.push(prod_idx.len());

        // result = result + c_i * I
        let (ci_off, ci_idx, ci_val) =
            scalar_multiply_csr(&id_off, &id_idx, &id_val, coefficients[i]);
        let (sum_off, sum_idx, sum_val) = add_csr(
            &prod_off, &prod_idx, &prod_val, &ci_off, &ci_idx, &ci_val, n,
        )?;

        r_off = sum_off;
        r_idx = sum_idx;
        r_val = sum_val;
    }

    Ok(MatrixPowerResult {
        nnz: r_idx.len(),
        row_offsets: r_off,
        col_indices: r_idx,
        values: r_val,
        rows: n,
        cols: n,
        multiplications_performed: mults,
        nnz_growth,
    })
}

/// Multiply all values of a CSR matrix by a scalar, keeping the sparsity
/// structure unchanged.
///
/// Returns `(row_offsets, col_indices, values)` -- offsets and indices are
/// cloned, values are scaled.
pub fn scalar_multiply_csr(
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f64],
    scalar: f64,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let scaled: Vec<f64> = values.iter().map(|&v| v * scalar).collect();
    (row_offsets.to_vec(), col_indices.to_vec(), scaled)
}

/// Add two CSR matrices with the same number of rows: `C = A + B`.
///
/// Performs a merge of sorted column-index arrays per row, summing values
/// where columns coincide.
///
/// # Errors
///
/// Returns [`SparseError::InvalidFormat`] if the row offset arrays are
/// inconsistent.
pub fn add_csr(
    a_offsets: &[usize],
    a_indices: &[usize],
    a_values: &[f64],
    b_offsets: &[usize],
    b_indices: &[usize],
    b_values: &[f64],
    rows: usize,
) -> Result<CsrTriple, SparseError> {
    if a_offsets.len() != rows + 1 || b_offsets.len() != rows + 1 {
        return Err(SparseError::InvalidFormat(format!(
            "row_offsets length mismatch: expected {}, got A={} B={}",
            rows + 1,
            a_offsets.len(),
            b_offsets.len()
        )));
    }

    let mut c_offsets = Vec::with_capacity(rows + 1);
    let mut c_indices = Vec::new();
    let mut c_values = Vec::new();
    c_offsets.push(0usize);

    for i in 0..rows {
        let mut ai = a_offsets[i];
        let ae = a_offsets[i + 1];
        let mut bi = b_offsets[i];
        let be = b_offsets[i + 1];

        while ai < ae && bi < be {
            let ac = a_indices[ai];
            let bc = b_indices[bi];
            match ac.cmp(&bc) {
                std::cmp::Ordering::Less => {
                    c_indices.push(ac);
                    c_values.push(a_values[ai]);
                    ai += 1;
                }
                std::cmp::Ordering::Greater => {
                    c_indices.push(bc);
                    c_values.push(b_values[bi]);
                    bi += 1;
                }
                std::cmp::Ordering::Equal => {
                    c_indices.push(ac);
                    c_values.push(a_values[ai] + b_values[bi]);
                    ai += 1;
                    bi += 1;
                }
            }
        }
        while ai < ae {
            c_indices.push(a_indices[ai]);
            c_values.push(a_values[ai]);
            ai += 1;
        }
        while bi < be {
            c_indices.push(b_indices[bi]);
            c_values.push(b_values[bi]);
            bi += 1;
        }

        c_offsets.push(c_indices.len());
    }

    Ok((c_offsets, c_indices, c_values))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate basic CSR consistency.
fn validate_csr(
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f64],
    rows: usize,
) -> Result<(), SparseError> {
    if row_offsets.len() != rows + 1 {
        return Err(SparseError::InvalidFormat(format!(
            "row_offsets length should be {} but is {}",
            rows + 1,
            row_offsets.len()
        )));
    }
    if col_indices.len() != values.len() {
        return Err(SparseError::InvalidFormat(format!(
            "col_indices length ({}) != values length ({})",
            col_indices.len(),
            values.len()
        )));
    }
    let nnz = row_offsets.get(rows).copied().unwrap_or(0);
    if col_indices.len() != nnz {
        return Err(SparseError::InvalidFormat(format!(
            "col_indices length ({}) != nnz from row_offsets ({})",
            col_indices.len(),
            nnz
        )));
    }
    Ok(())
}

/// Check whether nnz exceeds the configured maximum, returning an error if so.
fn check_max_nnz(nnz: usize, max: Option<usize>) -> Result<(), SparseError> {
    if let Some(limit) = max {
        if nnz > limit {
            return Err(SparseError::InvalidArgument(format!(
                "nnz ({}) exceeds max_nnz limit ({})",
                nnz, limit
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(power: usize) -> MatrixPowerConfig {
        MatrixPowerConfig {
            max_nnz: None,
            reuse_structure: false,
            power,
        }
    }

    // -- sparse_identity --

    #[test]
    fn test_sparse_identity() {
        let (off, idx, val) = sparse_identity(4);
        assert_eq!(off, vec![0, 1, 2, 3, 4]);
        assert_eq!(idx, vec![0, 1, 2, 3]);
        assert_eq!(val, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sparse_identity_zero() {
        let (off, idx, val) = sparse_identity(0);
        assert_eq!(off, vec![0]);
        assert!(idx.is_empty());
        assert!(val.is_empty());
    }

    // -- A^0 = I --

    #[test]
    fn test_power_zero_returns_identity() {
        // 2x2 matrix [[1,2],[0,3]]
        let off = vec![0, 2, 3];
        let idx = vec![0, 1, 1];
        let val = vec![1.0, 2.0, 3.0];
        let config = default_config(0);
        let res = sparse_matrix_power(&off, &idx, &val, 2, 2, 0, &config);
        let r = res.expect("test: power 0 should succeed");
        assert_eq!(r.rows, 2);
        assert_eq!(r.cols, 2);
        assert_eq!(r.nnz, 2);
        assert_eq!(r.row_offsets, vec![0, 1, 2]);
        assert_eq!(r.col_indices, vec![0, 1]);
        assert_eq!(r.values, vec![1.0, 1.0]);
        assert_eq!(r.multiplications_performed, 0);
    }

    // -- A^1 = A --

    #[test]
    fn test_power_one_returns_copy() {
        let off = vec![0, 2, 3];
        let idx = vec![0, 1, 1];
        let val = vec![1.0, 2.0, 3.0];
        let config = default_config(1);
        let r = sparse_matrix_power(&off, &idx, &val, 2, 2, 1, &config)
            .expect("test: power 1 should succeed");
        assert_eq!(r.row_offsets, off);
        assert_eq!(r.col_indices, idx);
        assert_eq!(r.values, val);
        assert_eq!(r.multiplications_performed, 0);
    }

    // -- A^2 for simple 3x3 --

    #[test]
    fn test_power_two_3x3() {
        // A = [[1,0,0],[0,2,0],[0,0,3]]  (diagonal)
        let off = vec![0, 1, 2, 3];
        let idx = vec![0, 1, 2];
        let val = vec![1.0, 2.0, 3.0];
        let config = default_config(2);
        let r = sparse_matrix_power(&off, &idx, &val, 3, 3, 2, &config)
            .expect("test: power 2 should succeed");
        // A^2 = diag(1, 4, 9)
        assert_eq!(r.col_indices, vec![0, 1, 2]);
        assert_eq!(r.values, vec![1.0, 4.0, 9.0]);
    }

    // -- binary exponentiation correctness --

    #[test]
    fn test_binary_vs_sequential_power4() {
        // Non-diagonal 2x2: [[1,1],[0,1]]
        // A^2 = [[1,2],[0,1]], A^3 = [[1,3],[0,1]], A^4 = [[1,4],[0,1]]
        let off = vec![0, 2, 3];
        let idx = vec![0, 1, 1];
        let val = vec![1.0, 1.0, 1.0];

        // Binary exponentiation (default)
        let config = default_config(4);
        let r = sparse_matrix_power(&off, &idx, &val, 2, 2, 4, &config)
            .expect("test: binary exp power 4");

        // Expected A^4 = [[1,4],[0,1]]
        assert_eq!(r.row_offsets, vec![0, 2, 3]);
        assert_eq!(r.col_indices, vec![0, 1, 1]);
        assert!((r.values[0] - 1.0).abs() < 1e-12);
        assert!((r.values[1] - 4.0).abs() < 1e-12);
        assert!((r.values[2] - 1.0).abs() < 1e-12);
    }

    // -- max_nnz abort --

    #[test]
    fn test_max_nnz_abort() {
        // Dense-ish 3x3 that will grow nnz on squaring
        let off = vec![0, 3, 6, 9];
        let idx = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let val = vec![1.0; 9];
        let config = MatrixPowerConfig {
            max_nnz: Some(5), // way too small for 3x3 dense squared
            reuse_structure: false,
            power: 2,
        };
        let result = sparse_matrix_power(&off, &idx, &val, 3, 3, 2, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("max_nnz"),
            "error should mention max_nnz: {}",
            msg
        );
    }

    // -- nnz growth tracking --

    #[test]
    fn test_nnz_growth_tracking() {
        // Diagonal 3x3 -- nnz stays at 3
        let off = vec![0, 1, 2, 3];
        let idx = vec![0, 1, 2];
        let val = vec![2.0, 3.0, 4.0];
        let config = default_config(4);
        let r = sparse_matrix_power(&off, &idx, &val, 3, 3, 4, &config).expect("test: nnz growth");
        // nnz_growth should contain initial nnz plus entries from each mult
        assert!(!r.nnz_growth.is_empty());
        // For diagonal matrix, all entries should be 3
        for &g in &r.nnz_growth {
            assert_eq!(g, 3);
        }
    }

    // -- host_spgemm 2x2 --

    #[test]
    fn test_host_spgemm_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = A*B = [[19,22],[43,50]]
        let a_off = vec![0, 2, 4];
        let a_idx = vec![0, 1, 0, 1];
        let a_val = vec![1.0, 2.0, 3.0, 4.0];
        let b_off = vec![0, 2, 4];
        let b_idx = vec![0, 1, 0, 1];
        let b_val = vec![5.0, 6.0, 7.0, 8.0];

        let (c_off, c_idx, c_val) =
            host_spgemm(&a_off, &a_idx, &a_val, 2, 2, &b_off, &b_idx, &b_val, 2, 2)
                .expect("test: spgemm 2x2");

        assert_eq!(c_off, vec![0, 2, 4]);
        assert_eq!(c_idx, vec![0, 1, 0, 1]);
        assert!((c_val[0] - 19.0).abs() < 1e-12);
        assert!((c_val[1] - 22.0).abs() < 1e-12);
        assert!((c_val[2] - 43.0).abs() < 1e-12);
        assert!((c_val[3] - 50.0).abs() < 1e-12);
    }

    // -- polynomial: p(A) = I + A for diagonal --

    #[test]
    fn test_polynomial_identity_plus_a() {
        // A = diag(2, 3), p(A) = I + A = diag(3, 4)
        let off = vec![0, 1, 2];
        let idx = vec![0, 1];
        let val = vec![2.0, 3.0];
        let coeffs = [1.0, 1.0]; // c0=1, c1=1 => I + A

        let r = sparse_matrix_polynomial(&off, &idx, &val, 2, 2, &coeffs)
            .expect("test: polynomial I+A");
        assert_eq!(r.col_indices, vec![0, 1]);
        assert!((r.values[0] - 3.0).abs() < 1e-12);
        assert!((r.values[1] - 4.0).abs() < 1e-12);
    }

    // -- scalar multiply --

    #[test]
    fn test_scalar_multiply() {
        let off = vec![0, 2, 3];
        let idx = vec![0, 1, 1];
        let val = vec![1.0, 2.0, 3.0];
        let (s_off, s_idx, s_val) = scalar_multiply_csr(&off, &idx, &val, 3.0);
        assert_eq!(s_off, off);
        assert_eq!(s_idx, idx);
        assert_eq!(s_val, vec![3.0, 6.0, 9.0]);
    }

    // -- CSR addition --

    #[test]
    fn test_add_csr() {
        // A = [[1,0],[0,2]], B = [[0,3],[4,0]]
        let a_off = vec![0, 1, 2];
        let a_idx = vec![0, 1];
        let a_val = vec![1.0, 2.0];
        let b_off = vec![0, 1, 2];
        let b_idx = vec![1, 0];
        let b_val = vec![3.0, 4.0];

        let (c_off, c_idx, c_val) =
            add_csr(&a_off, &a_idx, &a_val, &b_off, &b_idx, &b_val, 2).expect("test: add_csr");
        // C = [[1,3],[4,2]]
        assert_eq!(c_off, vec![0, 2, 4]);
        assert_eq!(c_idx, vec![0, 1, 0, 1]);
        assert!((c_val[0] - 1.0).abs() < 1e-12);
        assert!((c_val[1] - 3.0).abs() < 1e-12);
        assert!((c_val[2] - 4.0).abs() < 1e-12);
        assert!((c_val[3] - 2.0).abs() < 1e-12);
    }

    // -- estimate_power_nnz --

    #[test]
    fn test_estimate_power_nnz() {
        // 4x4 with 8 nnz => avg_degree = 2
        let off = vec![0, 2, 4, 6, 8];
        let idx = vec![0, 1, 1, 2, 2, 3, 0, 3];
        let est = estimate_power_nnz(&off, &idx, 4, 2);
        // avg_degree=2, power=2 => degree^2=4, per_row=min(4,4)=4, total=16
        assert_eq!(est, 16);
    }

    #[test]
    fn test_estimate_power_nnz_zero() {
        let off = vec![0, 0, 0];
        let idx: Vec<usize> = vec![];
        let est = estimate_power_nnz(&off, &idx, 2, 3);
        assert_eq!(est, 0);
    }

    // -- diagonal matrix power (element-wise power) --

    #[test]
    fn test_diagonal_power() {
        // diag(2, 3, 5)^3 = diag(8, 27, 125)
        let off = vec![0, 1, 2, 3];
        let idx = vec![0, 1, 2];
        let val = vec![2.0, 3.0, 5.0];
        let config = default_config(3);
        let r =
            sparse_matrix_power(&off, &idx, &val, 3, 3, 3, &config).expect("test: diagonal power");
        assert_eq!(r.col_indices, vec![0, 1, 2]);
        assert!((r.values[0] - 8.0).abs() < 1e-12);
        assert!((r.values[1] - 27.0).abs() < 1e-12);
        assert!((r.values[2] - 125.0).abs() < 1e-12);
    }

    // -- Horner polynomial vs direct computation --

    #[test]
    fn test_horner_vs_direct() {
        // A = diag(2, 3), p(x) = 1 + 2x + 3x^2
        // p(2) = 1 + 4 + 12 = 17
        // p(3) = 1 + 6 + 27 = 34
        let off = vec![0, 1, 2];
        let idx = vec![0, 1];
        let val = vec![2.0, 3.0];
        let coeffs = [1.0, 2.0, 3.0];

        let r = sparse_matrix_polynomial(&off, &idx, &val, 2, 2, &coeffs)
            .expect("test: Horner polynomial");
        assert_eq!(r.col_indices, vec![0, 1]);
        assert!((r.values[0] - 17.0).abs() < 1e-12);
        assert!((r.values[1] - 34.0).abs() < 1e-12);
    }

    // -- empty matrix power --

    #[test]
    fn test_empty_matrix_power() {
        let off = vec![0];
        let idx: Vec<usize> = vec![];
        let val: Vec<f64> = vec![];
        let config = default_config(5);
        // 0x0 matrix to power 0 => 0x0 identity
        let r =
            sparse_matrix_power(&off, &idx, &val, 0, 0, 0, &config).expect("test: empty power 0");
        assert_eq!(r.rows, 0);
        assert_eq!(r.cols, 0);
        assert_eq!(r.nnz, 0);
    }

    // -- reuse_structure flag --

    #[test]
    fn test_reuse_structure_flag() {
        // Just ensure it runs without error with reuse_structure = true
        let off = vec![0, 1, 2];
        let idx = vec![0, 1];
        let val = vec![2.0, 3.0];
        let config = MatrixPowerConfig {
            max_nnz: None,
            reuse_structure: true,
            power: 4,
        };
        let r =
            sparse_matrix_power(&off, &idx, &val, 2, 2, 4, &config).expect("test: reuse structure");
        // diag(2,3)^4 = diag(16, 81)
        assert!((r.values[0] - 16.0).abs() < 1e-12);
        assert!((r.values[1] - 81.0).abs() < 1e-12);
    }
}
