//! ILU(k) -- Incomplete LU factorization with k levels of fill-in.
//!
//! ILU(0) keeps only original nonzero positions. ILU(k) allows fill elements
//! up to k steps from an original nonzero, producing a more accurate (but
//! denser) preconditioner.
//!
//! ## Algorithm
//!
//! Two phases:
//! 1. **Symbolic phase**: Determine the fill pattern using level-of-fill arrays.
//!    For each entry `(i, j)` in the original matrix, level is 0. During
//!    elimination, if row `i` has entry at column `k` with level `lev_ik` and
//!    row `k` has entry at column `j` with level `lev_kj`, a fill-in entry
//!    `(i, j)` is generated with level `lev_ik + lev_kj + 1`. Entries with
//!    level `<= fill_level` are kept.
//! 2. **Numeric phase**: Compute the actual ILU factorization values on the
//!    fill pattern determined in step 1.
#![allow(dead_code)]

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

fn div_gpu_float<T: GpuFloat>(a: T, b: T) -> T {
    from_f64::<T>(to_f64(a) / to_f64(b))
}

fn mul_gpu_float<T: GpuFloat>(a: T, b: T) -> T {
    from_f64::<T>(to_f64(a) * to_f64(b))
}

fn sub_gpu_float<T: GpuFloat>(a: T, b: T) -> T {
    from_f64::<T>(to_f64(a) - to_f64(b))
}

// ---------------------------------------------------------------------------
// ILU(k) configuration
// ---------------------------------------------------------------------------

/// ILU(k) preconditioner configuration.
#[derive(Default)]
pub struct IlukConfig {
    /// Level of fill: 0 = ILU(0), 1 = ILU(1), etc.
    pub fill_level: usize,
}

// ---------------------------------------------------------------------------
// ILU(k) factorization result
// ---------------------------------------------------------------------------

/// ILU(k) factorization result stored as combined L\U in CSR.
///
/// `L` is unit lower triangular (diagonal = 1, stored implicitly) and `U` is
/// upper triangular (diagonal stored explicitly). Both share the CSR storage
/// `lu`.
pub struct IlukFactorization<T: GpuFloat> {
    /// Combined L\U factors in CSR format.
    /// Lower triangle contains L (without unit diagonal).
    /// Upper triangle (including diagonal) contains U.
    pub lu: CsrMatrix<T>,
    /// Inverse of the diagonal of U, for efficient forward/backward solve.
    pub diag_inv: Vec<T>,
    /// The fill level used to compute this factorization.
    pub fill_level: usize,
}

// ---------------------------------------------------------------------------
// Symbolic phase: determine fill pattern
// ---------------------------------------------------------------------------

/// Entry in the symbolic ILU(k) structure.
struct SymbolicEntry {
    col: usize,
    level: usize,
}

/// Symbolic ILU(k): determine which entries (including fill-in) to keep.
///
/// Returns `(row_ptr, col_idx, levels)` where `levels[nz]` is the level-of-fill
/// for each nonzero position.
fn iluk_symbolic(
    row_ptr: &[i32],
    col_idx: &[i32],
    n: usize,
    fill_level: usize,
) -> SparseResult<(Vec<i32>, Vec<i32>, Vec<usize>)> {
    // For each row, maintain a sorted list of (column, level) pairs.
    let mut rows: Vec<Vec<SymbolicEntry>> = Vec::with_capacity(n);

    for i in 0..n {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        let mut row_entries: Vec<SymbolicEntry> = Vec::with_capacity(end - start);
        for &cj in &col_idx[start..end] {
            row_entries.push(SymbolicEntry {
                col: cj as usize,
                level: 0,
            });
        }
        // Sort by column
        row_entries.sort_by_key(|e| e.col);
        rows.push(row_entries);
    }

    // Symbolic factorization: process rows in order
    for i in 0..n {
        // Find all lower-triangular entries in row i
        let mut k_idx = 0;
        loop {
            if k_idx >= rows[i].len() {
                break;
            }
            let k = rows[i][k_idx].col;
            if k >= i {
                break;
            }
            let lev_ik = rows[i][k_idx].level;

            // Find diagonal of row k
            let diag_pos = rows[k].iter().position(|e| e.col == k);
            if diag_pos.is_none() {
                k_idx += 1;
                continue;
            }

            // For each entry in row k with col j > k
            let row_k_entries: Vec<(usize, usize)> = rows[k]
                .iter()
                .filter(|e| e.col > k)
                .map(|e| (e.col, e.level))
                .collect();

            for (j, lev_kj) in row_k_entries {
                let new_level = lev_ik + lev_kj + 1;
                if new_level > fill_level {
                    continue;
                }

                // Check if column j already exists in row i
                let existing = rows[i].iter().position(|e| e.col == j);
                match existing {
                    Some(pos) => {
                        // Update level to minimum
                        if new_level < rows[i][pos].level {
                            rows[i][pos].level = new_level;
                        }
                    }
                    None => {
                        // Insert new fill-in entry
                        let insert_pos = rows[i]
                            .iter()
                            .position(|e| e.col > j)
                            .unwrap_or(rows[i].len());
                        rows[i].insert(
                            insert_pos,
                            SymbolicEntry {
                                col: j,
                                level: new_level,
                            },
                        );
                    }
                }
            }

            k_idx += 1;
        }
    }

    // Build CSR arrays from the symbolic structure
    let mut out_row_ptr = vec![0i32; n + 1];
    let mut out_col_idx = Vec::new();
    let mut out_levels = Vec::new();

    for (i, row_entries) in rows.iter().enumerate() {
        for entry in row_entries {
            out_col_idx.push(entry.col as i32);
            out_levels.push(entry.level);
        }
        out_row_ptr[i + 1] = out_col_idx.len() as i32;
    }

    Ok((out_row_ptr, out_col_idx, out_levels))
}

// ---------------------------------------------------------------------------
// Numeric phase
// ---------------------------------------------------------------------------

/// Performs numeric ILU(k) factorization on the given fill pattern.
///
/// `sym_row_ptr`, `sym_col_idx` define the sparsity pattern (from symbolic phase).
/// `orig_row_ptr`, `orig_col_idx`, `orig_values` are the original matrix data.
///
/// Returns the factored values array (same length as `sym_col_idx`).
fn iluk_numeric<T: GpuFloat>(
    sym_row_ptr: &[i32],
    sym_col_idx: &[i32],
    orig_row_ptr: &[i32],
    orig_col_idx: &[i32],
    orig_values: &[T],
    n: usize,
) -> SparseResult<Vec<T>> {
    let nnz = sym_col_idx.len();
    let mut values = vec![T::gpu_zero(); nnz];

    // Copy original values into the new pattern
    for i in 0..n {
        let orig_start = orig_row_ptr[i] as usize;
        let orig_end = orig_row_ptr[i + 1] as usize;
        let sym_start = sym_row_ptr[i] as usize;
        let sym_end = sym_row_ptr[i + 1] as usize;

        let mut sym_k = sym_start;
        for orig_k in orig_start..orig_end {
            let col = orig_col_idx[orig_k];
            // Find col in symbolic row (it must exist since original entries have level 0)
            while sym_k < sym_end && sym_col_idx[sym_k] < col {
                sym_k += 1;
            }
            if sym_k < sym_end && sym_col_idx[sym_k] == col {
                values[sym_k] = orig_values[orig_k];
            }
        }
    }

    // Perform ILU factorization on the filled pattern
    for i in 0..n {
        let row_start = sym_row_ptr[i] as usize;
        let row_end = sym_row_ptr[i + 1] as usize;

        // Process lower-triangular entries in row i
        for nz in row_start..row_end {
            let k = sym_col_idx[nz] as usize;
            if k >= i {
                break;
            }

            // Find diagonal of row k
            let k_start = sym_row_ptr[k] as usize;
            let k_end = sym_row_ptr[k + 1] as usize;
            let diag_pos = find_col_in_row(&sym_col_idx[k_start..k_end], k as i32);
            let diag_pos = match diag_pos {
                Some(pos) => k_start + pos,
                None => return Err(SparseError::SingularMatrix),
            };

            let a_kk = values[diag_pos];
            if a_kk == T::gpu_zero() {
                return Err(SparseError::SingularMatrix);
            }

            // a_ik /= a_kk
            let ratio = div_gpu_float(values[nz], a_kk);
            values[nz] = ratio;

            // For each j in row k with j > k, update a_ij -= ratio * a_kj
            for k_nz in (diag_pos + 1)..k_end {
                let j = sym_col_idx[k_nz];
                if let Some(ij_off) = find_col_in_row(&sym_col_idx[row_start..row_end], j) {
                    let ij_pos = row_start + ij_off;
                    let update = mul_gpu_float(ratio, values[k_nz]);
                    values[ij_pos] = sub_gpu_float(values[ij_pos], update);
                }
            }
        }
    }

    Ok(values)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl<T: GpuFloat> IlukFactorization<T> {
    /// Compute ILU(k) from a CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `_handle` -- Sparse handle (reserved for GPU path).
    /// * `matrix` -- Square CSR matrix to factor.
    /// * `config` -- ILU(k) configuration (fill level).
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if the matrix is not square.
    /// Returns [`SparseError::SingularMatrix`] if a zero pivot is encountered.
    pub fn compute(
        _handle: &SparseHandle,
        matrix: &CsrMatrix<T>,
        config: &IlukConfig,
    ) -> SparseResult<Self> {
        if matrix.rows() != matrix.cols() {
            return Err(SparseError::DimensionMismatch(format!(
                "ILU(k) requires square matrix, got {}x{}",
                matrix.rows(),
                matrix.cols()
            )));
        }

        let n = matrix.rows() as usize;
        if n == 0 {
            return Err(SparseError::InvalidArgument(
                "cannot factor an empty matrix".to_string(),
            ));
        }

        let (h_row_ptr, h_col_idx, h_values) = matrix.to_host()?;

        // Symbolic phase: determine fill pattern
        let (sym_row_ptr, sym_col_idx, _levels) =
            iluk_symbolic(&h_row_ptr, &h_col_idx, n, config.fill_level)?;

        // Numeric phase: compute values
        let factored_values = iluk_numeric::<T>(
            &sym_row_ptr,
            &sym_col_idx,
            &h_row_ptr,
            &h_col_idx,
            &h_values,
            n,
        )?;

        // Extract diagonal inverses
        let mut diag_inv = vec![T::gpu_zero(); n];
        for i in 0..n {
            let start = sym_row_ptr[i] as usize;
            let end = sym_row_ptr[i + 1] as usize;
            let diag_pos = find_col_in_row(&sym_col_idx[start..end], i as i32);
            match diag_pos {
                Some(pos) => {
                    let diag_val = factored_values[start + pos];
                    if diag_val == T::gpu_zero() {
                        return Err(SparseError::SingularMatrix);
                    }
                    diag_inv[i] = div_gpu_float(T::gpu_one(), diag_val);
                }
                None => return Err(SparseError::SingularMatrix),
            }
        }

        let nnz = sym_col_idx.len() as u32;
        if nnz == 0 {
            return Err(SparseError::ZeroNnz);
        }

        let lu = CsrMatrix::from_host(
            matrix.rows(),
            matrix.cols(),
            &sym_row_ptr,
            &sym_col_idx,
            &factored_values,
        )?;

        Ok(Self {
            lu,
            diag_inv,
            fill_level: config.fill_level,
        })
    }

    /// Apply as preconditioner: solve `(L*U)*z = r`.
    ///
    /// Forward solve `L*y = r` then backward solve `U*z = y`.
    ///
    /// # Arguments
    ///
    /// * `r` -- Right-hand side vector (host, length n).
    /// * `z` -- Output vector (host, length n).
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if vector lengths are wrong.
    pub fn apply(&self, r: &[T], z: &mut [T]) -> SparseResult<()> {
        let n = self.lu.rows() as usize;
        if r.len() != n || z.len() != n {
            return Err(SparseError::DimensionMismatch(format!(
                "vector length mismatch: r={}, z={}, expected {}",
                r.len(),
                z.len(),
                n
            )));
        }

        let (h_row_ptr, h_col_idx, h_values) = self.lu.to_host()?;

        // Forward solve: L*y = r
        // L has unit diagonal (implicit), lower triangle stored in lu
        let mut y = vec![T::gpu_zero(); n];
        for i in 0..n {
            let start = h_row_ptr[i] as usize;
            let end = h_row_ptr[i + 1] as usize;
            let mut sum = r[i];

            for nz in start..end {
                let j = h_col_idx[nz] as usize;
                if j >= i {
                    break;
                }
                let update = mul_gpu_float(h_values[nz], y[j]);
                sum = sub_gpu_float(sum, update);
            }
            y[i] = sum;
        }

        // Backward solve: U*z = y
        // U has explicit diagonal, upper triangle stored in lu
        for i in (0..n).rev() {
            let start = h_row_ptr[i] as usize;
            let end = h_row_ptr[i + 1] as usize;
            let mut sum = y[i];

            for nz in start..end {
                let j = h_col_idx[nz] as usize;
                if j <= i {
                    continue;
                }
                let update = mul_gpu_float(h_values[nz], z[j]);
                sum = sub_gpu_float(sum, update);
            }
            z[i] = mul_gpu_float(sum, self.diag_inv[i]);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_col_in_row(col_slice: &[i32], target_col: i32) -> Option<usize> {
    col_slice.iter().position(|&c| c == target_col)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iluk_config_default() {
        let cfg = IlukConfig::default();
        assert_eq!(cfg.fill_level, 0);
    }

    #[test]
    fn symbolic_identity_no_fill() {
        // Identity matrix: no fill at any level
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let (sym_rp, sym_ci, levels) =
            iluk_symbolic(&row_ptr, &col_idx, 3, 5).expect("test: symbolic should succeed");
        assert_eq!(sym_rp, row_ptr);
        assert_eq!(sym_ci, col_idx);
        assert!(levels.iter().all(|&l| l == 0));
    }

    #[test]
    fn symbolic_tridiagonal_fill_level_0() {
        // Tridiagonal 3x3:
        // [1 2 0]
        // [3 4 5]
        // [0 6 7]
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        let (sym_rp, sym_ci, _) =
            iluk_symbolic(&row_ptr, &col_idx, 3, 0).expect("test: symbolic should succeed");
        // ILU(0) keeps the same pattern
        assert_eq!(sym_rp, row_ptr);
        assert_eq!(sym_ci, col_idx);
    }

    #[test]
    fn symbolic_tridiagonal_fill_level_1() {
        // Tridiagonal 4x4, ILU(1) may produce fill-in at (0,2) and (2,0)
        // Row 0: [0,1], Row 1: [0,1,2], Row 2: [1,2,3], Row 3: [2,3]
        let row_ptr = vec![0, 2, 5, 8, 10];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let (sym_rp, sym_ci, _) =
            iluk_symbolic(&row_ptr, &col_idx, 4, 1).expect("test: symbolic should succeed");
        // With level 1, fill is allowed. Check nnz increased or stayed same.
        let orig_nnz = col_idx.len();
        let sym_nnz = sym_ci.len();
        assert!(sym_nnz >= orig_nnz);
        // Verify row_ptr is consistent
        assert_eq!(sym_rp.len(), 5);
        assert_eq!(sym_rp[0], 0);
        assert_eq!(sym_rp[4], sym_nnz as i32);
    }

    #[test]
    fn numeric_identity() {
        // Identity 3x3: ILU(0) should give L=I, U=I
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values: Vec<f64> = vec![2.0, 3.0, 4.0];
        let result = iluk_numeric::<f64>(&row_ptr, &col_idx, &row_ptr, &col_idx, &values, 3);
        assert!(result.is_ok());
        let vals = result.expect("test: numeric should succeed");
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!((vals[1] - 3.0).abs() < 1e-12);
        assert!((vals[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn numeric_singular_detection() {
        // Matrix with zero diagonal
        let row_ptr = vec![0, 2, 4];
        let col_idx = vec![0, 1, 0, 1];
        let values: Vec<f64> = vec![0.0, 1.0, 1.0, 2.0];
        let result = iluk_numeric::<f64>(&row_ptr, &col_idx, &row_ptr, &col_idx, &values, 2);
        assert!(matches!(result, Err(SparseError::SingularMatrix)));
    }

    #[test]
    fn numeric_tridiagonal_f32() {
        // 3x3 tridiagonal: well-conditioned
        // [4 -1  0]
        // [-1 4 -1]
        // [0 -1  4]
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        let values: Vec<f32> = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let result = iluk_numeric::<f32>(&row_ptr, &col_idx, &row_ptr, &col_idx, &values, 3);
        assert!(result.is_ok());
        let vals = result.expect("test: numeric should succeed");
        // Check diagonal is nonzero
        assert!(to_f64(vals[0]).abs() > 1e-6);
        // Check L factor: a_10 / a_00 = -1/4 = -0.25
        assert!((to_f64(vals[2]) - (-0.25)).abs() < 1e-5);
    }

    #[test]
    fn find_col_works() {
        let cols = [0, 2, 5, 7];
        assert_eq!(find_col_in_row(&cols, 2), Some(1));
        assert_eq!(find_col_in_row(&cols, 3), None);
        assert_eq!(find_col_in_row(&cols, 7), Some(3));
    }

    #[test]
    fn symbolic_empty_row() {
        // Matrix with varying row lengths
        // Row 0: [0], Row 1: [0,1], Row 2: [2]
        let row_ptr = vec![0, 1, 3, 4];
        let col_idx = vec![0, 0, 1, 2];
        let (sym_rp, sym_ci, _) =
            iluk_symbolic(&row_ptr, &col_idx, 3, 0).expect("test: symbolic should succeed");
        assert_eq!(sym_rp.len(), 4);
        assert_eq!(sym_ci.len() as i32, sym_rp[3]);
    }
}
