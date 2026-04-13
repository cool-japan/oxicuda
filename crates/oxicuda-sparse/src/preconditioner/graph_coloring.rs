//! Distance-2 graph coloring for parallel ILU/IC.
//!
//! Colors vertices of the adjacency graph of a sparse matrix such that no two
//! vertices within distance 2 share the same color. This enables parallel
//! processing of independent rows during ILU(0) or IC(0) factorization:
//! all rows with the same color can be processed simultaneously without
//! data races.
//!
//! # Algorithm
//!
//! Uses a greedy distance-2 coloring:
//! 1. Process rows in order 0, 1, ..., n-1.
//! 2. For each row `i`, collect colors of all distance-1 neighbors (columns
//!    in row `i`) and distance-2 neighbors (columns of the rows of distance-1
//!    neighbors).
//! 3. Assign to row `i` the smallest color not in the collected set.
//!
//! This yields an upper bound of `Δ^2 + 1` colors where `Δ` is the maximum
//! degree, which is tight for some graphs but typically much better in
//! practice for sparse matrices from discretized PDEs.
//!
//! # Parallel ILU(0)
//!
//! After coloring, ILU(0) proceeds color-by-color: for each color `c`,
//! all rows with that color can be updated in parallel. Within a color,
//! no row depends on another (guaranteed by the distance-2 property).

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

// ---------------------------------------------------------------------------
// Graph coloring result
// ---------------------------------------------------------------------------

/// Result of a distance-2 graph coloring of a sparse matrix.
///
/// Contains the color assignment for each row, plus auxiliary data structures
/// for efficiently iterating over rows of the same color.
pub struct GraphColoring {
    /// Number of distinct colors used.
    pub num_colors: usize,
    /// Color assigned to each row: `colors[i]` is the color of row `i`.
    pub colors: Vec<u32>,
    /// Start index of each color group in `color_order`.
    /// `color_offsets[c]..color_offsets[c+1]` gives the range of row indices
    /// with color `c` in `color_order`.
    pub color_offsets: Vec<usize>,
    /// Row indices sorted by color.
    pub color_order: Vec<u32>,
}

impl GraphColoring {
    /// Compute a distance-2 coloring of the adjacency graph of a CSR matrix.
    ///
    /// The matrix must be square. The coloring considers the structural
    /// (non-zero) pattern, not the numerical values.
    ///
    /// # Arguments
    ///
    /// * `_handle` — sparse handle (reserved for future GPU-accelerated variants).
    /// * `matrix` — a square CSR matrix.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if the matrix is not square.
    pub fn from_csr<T: GpuFloat>(
        _handle: &SparseHandle,
        matrix: &CsrMatrix<T>,
    ) -> SparseResult<Self> {
        if matrix.rows() != matrix.cols() {
            return Err(SparseError::DimensionMismatch(format!(
                "GraphColoring requires square matrix, got {}x{}",
                matrix.rows(),
                matrix.cols()
            )));
        }

        let n = matrix.rows() as usize;
        if n == 0 {
            return Ok(Self {
                num_colors: 0,
                colors: Vec::new(),
                color_offsets: vec![0],
                color_order: Vec::new(),
            });
        }

        // Download structure to host.
        let (h_row_ptr, h_col_idx, _h_values) = matrix.to_host()?;

        Self::from_csr_host(&h_row_ptr, &h_col_idx, n)
    }

    /// Compute distance-2 coloring from host-side CSR structure.
    ///
    /// This is the core algorithm, operating on host data.
    pub fn from_csr_host(row_ptr: &[i32], col_idx: &[i32], n: usize) -> SparseResult<Self> {
        if n == 0 {
            return Ok(Self {
                num_colors: 0,
                colors: Vec::new(),
                color_offsets: vec![0],
                color_order: Vec::new(),
            });
        }

        let mut colors = vec![u32::MAX; n];
        let mut max_color: u32 = 0;

        // Temporary buffer for forbidden colors.
        // Using a Vec<bool> indexed by color to track which colors are forbidden.
        let mut forbidden = Vec::new();

        for i in 0..n {
            let row_start = row_ptr[i] as usize;
            let row_end = row_ptr[i + 1] as usize;

            // Clear forbidden set.
            for f in forbidden.iter_mut() {
                *f = false;
            }

            // Distance-1 neighbors: columns in row i.
            for nz in row_start..row_end {
                let j = col_idx[nz] as usize;
                if j == i || j >= n {
                    continue;
                }
                // If neighbor j is already colored, forbid that color.
                if colors[j] != u32::MAX {
                    let c = colors[j] as usize;
                    if c >= forbidden.len() {
                        forbidden.resize(c + 1, false);
                    }
                    forbidden[c] = true;
                }

                // Distance-2 neighbors: columns in row j.
                let j_start = row_ptr[j] as usize;
                let j_end = row_ptr[j + 1] as usize;
                for &col in &col_idx[j_start..j_end] {
                    let k = col as usize;
                    if k == i || k >= n {
                        continue;
                    }
                    if colors[k] != u32::MAX {
                        let c = colors[k] as usize;
                        if c >= forbidden.len() {
                            forbidden.resize(c + 1, false);
                        }
                        forbidden[c] = true;
                    }
                }
            }

            // Find smallest non-forbidden color.
            let mut chosen = 0u32;
            loop {
                let c = chosen as usize;
                if c >= forbidden.len() || !forbidden[c] {
                    break;
                }
                chosen += 1;
            }

            colors[i] = chosen;
            if chosen > max_color {
                max_color = chosen;
            }
        }

        let num_colors = max_color as usize + 1;

        // Build color_offsets and color_order.
        let mut color_counts = vec![0usize; num_colors];
        for &c in &colors {
            color_counts[c as usize] += 1;
        }

        let mut color_offsets = vec![0usize; num_colors + 1];
        for c in 0..num_colors {
            color_offsets[c + 1] = color_offsets[c] + color_counts[c];
        }

        let mut color_order = vec![0u32; n];
        let mut write_pos = color_offsets[..num_colors].to_vec();
        for (i, &c) in colors.iter().enumerate() {
            let c_usize = c as usize;
            color_order[write_pos[c_usize]] = i as u32;
            write_pos[c_usize] += 1;
        }

        Ok(Self {
            num_colors,
            colors,
            color_offsets,
            color_order,
        })
    }

    /// Returns the row indices for a given color.
    ///
    /// # Panics
    ///
    /// This method does not panic; returns an empty slice if `color >= num_colors`.
    pub fn rows_for_color(&self, color: usize) -> &[u32] {
        if color >= self.num_colors {
            return &[];
        }
        let start = self.color_offsets[color];
        let end = self.color_offsets[color + 1];
        &self.color_order[start..end]
    }

    /// Apply coloring to parallelize ILU(0) factorization.
    ///
    /// Performs ILU(0) on a CSR matrix, processing rows color-by-color.
    /// Within each color, rows are independent and can be processed in parallel.
    ///
    /// Returns `(L, U)` where `L` is unit lower triangular and `U` is upper
    /// triangular (including diagonal).
    ///
    /// # Arguments
    ///
    /// * `_handle` — sparse handle.
    /// * `matrix` — the CSR matrix to factor (square, will be consumed for values).
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::SingularMatrix`] if a zero pivot is encountered.
    pub fn parallel_ilu0<T: GpuFloat>(
        &self,
        _handle: &SparseHandle,
        matrix: &CsrMatrix<T>,
    ) -> SparseResult<(CsrMatrix<T>, CsrMatrix<T>)> {
        let n = matrix.rows() as usize;
        if n == 0 {
            return Err(SparseError::InvalidArgument(
                "cannot factor an empty matrix".to_string(),
            ));
        }

        let (h_row_ptr, h_col_idx, h_values) = matrix.to_host()?;
        let mut work = h_values;

        // Process color by color.
        for color in 0..self.num_colors {
            let rows = self.rows_for_color(color);

            // All rows in this color are independent — can be parallelized.
            for &row_u32 in rows {
                let row = row_u32 as usize;
                let row_start = h_row_ptr[row] as usize;
                let row_end = h_row_ptr[row + 1] as usize;

                for nz in row_start..row_end {
                    let k = h_col_idx[nz] as usize;
                    if k >= row {
                        break;
                    }

                    // Find diagonal of row k.
                    let k_start = h_row_ptr[k] as usize;
                    let k_end = h_row_ptr[k + 1] as usize;
                    let diag_pos = find_col_in_row(&h_col_idx[k_start..k_end], k as i32);
                    let diag_pos = match diag_pos {
                        Some(pos) => k_start + pos,
                        None => return Err(SparseError::SingularMatrix),
                    };

                    let a_kk = work[diag_pos];
                    if a_kk == T::gpu_zero() {
                        return Err(SparseError::SingularMatrix);
                    }

                    // a_ik /= a_kk
                    let ratio = div_gpu_float(work[nz], a_kk);
                    work[nz] = ratio;

                    // Update: a_ij -= ratio * a_kj for j > k
                    for k_nz in (diag_pos + 1)..k_end {
                        let j = h_col_idx[k_nz];
                        if let Some(ij_off) = find_col_in_row(&h_col_idx[row_start..row_end], j) {
                            let ij_pos = row_start + ij_off;
                            let update = mul_gpu_float(ratio, work[k_nz]);
                            work[ij_pos] = sub_gpu_float(work[ij_pos], update);
                        }
                    }
                }
            }
        }

        // Split into L and U.
        split_lu(&h_row_ptr, &h_col_idx, &work, matrix.rows())
    }

    /// Validates the coloring: checks that no two vertices within distance 2
    /// share the same color.
    ///
    /// Returns `true` if the coloring is valid.
    pub fn validate(&self, row_ptr: &[i32], col_idx: &[i32], n: usize) -> bool {
        for i in 0..n {
            let row_start = row_ptr[i] as usize;
            let row_end = row_ptr[i + 1] as usize;
            let ci = self.colors[i];

            // Check distance-1 neighbors.
            for nz in row_start..row_end {
                let j = col_idx[nz] as usize;
                if j == i || j >= n {
                    continue;
                }
                if self.colors[j] == ci {
                    return false;
                }

                // Check distance-2 neighbors.
                let j_start = row_ptr[j] as usize;
                let j_end = row_ptr[j + 1] as usize;
                for &col in &col_idx[j_start..j_end] {
                    let k = col as usize;
                    if k == i || k >= n {
                        continue;
                    }
                    if self.colors[k] == ci {
                        return false;
                    }
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_col_in_row(col_slice: &[i32], target_col: i32) -> Option<usize> {
    col_slice.iter().position(|&c| c == target_col)
}

fn div_gpu_float<T: GpuFloat>(a: T, b: T) -> T {
    let fa = to_f64(a);
    let fb = to_f64(b);
    from_f64::<T>(fa / fb)
}

fn mul_gpu_float<T: GpuFloat>(a: T, b: T) -> T {
    let fa = to_f64(a);
    let fb = to_f64(b);
    from_f64::<T>(fa * fb)
}

fn sub_gpu_float<T: GpuFloat>(a: T, b: T) -> T {
    let fa = to_f64(a);
    let fb = to_f64(b);
    from_f64::<T>(fa - fb)
}

/// Splits factored values into L (unit lower triangular) and U (upper + diagonal).
fn split_lu<T: GpuFloat>(
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[T],
    n: u32,
) -> SparseResult<(CsrMatrix<T>, CsrMatrix<T>)> {
    let n_usize = n as usize;

    let mut l_nnz = 0usize;
    let mut u_nnz = 0usize;
    for i in 0..n_usize {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        for &cj in &col_idx[start..end] {
            let j = cj as usize;
            if j < i {
                l_nnz += 1;
            } else {
                u_nnz += 1;
            }
        }
        l_nnz += 1; // unit diagonal for L
    }

    let mut l_row_ptr = vec![0i32; n_usize + 1];
    let mut l_col_idx = Vec::with_capacity(l_nnz);
    let mut l_values = Vec::with_capacity(l_nnz);

    let mut u_row_ptr = vec![0i32; n_usize + 1];
    let mut u_col_idx = Vec::with_capacity(u_nnz);
    let mut u_values = Vec::with_capacity(u_nnz);

    for i in 0..n_usize {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;

        for idx in start..end {
            let j = col_idx[idx] as usize;
            if j < i {
                l_col_idx.push(col_idx[idx]);
                l_values.push(values[idx]);
            }
        }
        l_col_idx.push(i as i32);
        l_values.push(T::gpu_one());
        l_row_ptr[i + 1] = l_col_idx.len() as i32;

        for idx in start..end {
            let j = col_idx[idx] as usize;
            if j >= i {
                u_col_idx.push(col_idx[idx]);
                u_values.push(values[idx]);
            }
        }
        u_row_ptr[i + 1] = u_col_idx.len() as i32;
    }

    let l_mat = CsrMatrix::from_host(n, n, &l_row_ptr, &l_col_idx, &l_values)?;
    let u_mat = CsrMatrix::from_host(n, n, &u_row_ptr, &u_col_idx, &u_values)?;

    Ok((l_mat, u_mat))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Greedy coloring tests on host-side structures ---

    #[test]
    fn coloring_diagonal_matrix() {
        // Diagonal matrix: no off-diagonal neighbors => all rows get color 0.
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let n = 3;

        let coloring = GraphColoring::from_csr_host(&row_ptr, &col_idx, n);
        assert!(coloring.is_ok());
        let coloring = match coloring {
            Ok(v) => v,
            Err(e) => panic!("test: {e}"),
        };
        assert_eq!(coloring.num_colors, 1);
        for &c in &coloring.colors {
            assert_eq!(c, 0);
        }
    }

    #[test]
    fn coloring_tridiagonal() {
        // Tridiagonal 4x4:
        // Row 0: cols [0, 1]
        // Row 1: cols [0, 1, 2]
        // Row 2: cols [1, 2, 3]
        // Row 3: cols [2, 3]
        let row_ptr = vec![0, 2, 5, 8, 10];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let n = 4;

        let coloring = GraphColoring::from_csr_host(&row_ptr, &col_idx, n);
        assert!(coloring.is_ok());
        let coloring = match coloring {
            Ok(v) => v,
            Err(e) => panic!("test: {e}"),
        };

        // Distance-2 coloring of a path graph needs at least 3 colors.
        assert!(coloring.num_colors >= 3, "need >= 3 colors for path graph");

        // Validate the coloring.
        assert!(coloring.validate(&row_ptr, &col_idx, n));
    }

    #[test]
    fn coloring_dense_small() {
        // Fully connected 3x3 (all entries non-zero):
        // Distance-2 coloring of K_3 needs 3 colors.
        let row_ptr = vec![0, 3, 6, 9];
        let col_idx = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let n = 3;

        let coloring = GraphColoring::from_csr_host(&row_ptr, &col_idx, n);
        assert!(coloring.is_ok());
        let coloring = match coloring {
            Ok(v) => v,
            Err(e) => panic!("test: {e}"),
        };
        assert_eq!(coloring.num_colors, 3);
        assert!(coloring.validate(&row_ptr, &col_idx, n));
    }

    #[test]
    fn coloring_rows_for_color() {
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let n = 3;

        let coloring = match GraphColoring::from_csr_host(&row_ptr, &col_idx, n) {
            Ok(v) => v,
            Err(e) => panic!("test: {e}"),
        };

        // All rows should be in color 0.
        let rows = coloring.rows_for_color(0);
        assert_eq!(rows.len(), 3);

        // No rows in color 1.
        let rows = coloring.rows_for_color(1);
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn coloring_single_node() {
        // Single node with self-loop.
        let row_ptr = vec![0, 1];
        let col_idx = vec![0];
        let n = 1;

        let coloring = GraphColoring::from_csr_host(&row_ptr, &col_idx, n);
        assert!(coloring.is_ok());
        let coloring = match coloring {
            Ok(v) => v,
            Err(e) => panic!("test: {e}"),
        };
        assert_eq!(coloring.num_colors, 1);
        assert_eq!(coloring.colors[0], 0);
        assert_eq!(coloring.color_order.len(), 1);
    }

    #[test]
    fn coloring_color_offsets_consistency() {
        // 5x5 banded matrix
        let row_ptr = vec![0, 2, 5, 8, 11, 13];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let n = 5;

        let coloring = match GraphColoring::from_csr_host(&row_ptr, &col_idx, n) {
            Ok(v) => v,
            Err(e) => panic!("test: {e}"),
        };

        // Check that color_offsets is consistent with color_order.
        let mut total = 0;
        for c in 0..coloring.num_colors {
            let count = coloring.rows_for_color(c).len();
            total += count;
        }
        assert_eq!(total, n);

        // Last offset should equal n.
        assert_eq!(
            coloring.color_offsets[coloring.num_colors], n,
            "last color offset should equal n"
        );

        // Validate coloring.
        assert!(coloring.validate(&row_ptr, &col_idx, n));
    }

    #[test]
    fn coloring_validate_detects_bad_coloring() {
        // Manually create a BAD coloring for a tridiagonal matrix.
        let row_ptr = vec![0, 2, 4, 6];
        let col_idx = vec![0, 1, 1, 2, 0, 2]; // row0=[0,1], row1=[1,2], row2=[0,2]
        let n = 3;

        // All same color => distance-1 conflict.
        let bad_coloring = GraphColoring {
            num_colors: 1,
            colors: vec![0, 0, 0],
            color_offsets: vec![0, 3],
            color_order: vec![0, 1, 2],
        };
        assert!(!bad_coloring.validate(&row_ptr, &col_idx, n));
    }

    // --- Arithmetic helpers ---

    #[test]
    fn host_float_arithmetic_f32() {
        let a = 6.0_f32;
        let b = 2.0_f32;
        let result = div_gpu_float(a, b);
        assert!((result - 3.0_f32).abs() < 1e-6);

        let result = mul_gpu_float(a, b);
        assert!((result - 12.0_f32).abs() < 1e-6);

        let result = sub_gpu_float(a, b);
        assert!((result - 4.0_f32).abs() < 1e-6);
    }

    #[test]
    fn host_float_arithmetic_f64() {
        let a = 6.0_f64;
        let b = 2.0_f64;
        assert!((div_gpu_float(a, b) - 3.0_f64).abs() < 1e-12);
        assert!((mul_gpu_float(a, b) - 12.0_f64).abs() < 1e-12);
        assert!((sub_gpu_float(a, b) - 4.0_f64).abs() < 1e-12);
    }

    #[test]
    fn find_col_in_row_works() {
        let cols = [0, 2, 5, 7];
        assert_eq!(find_col_in_row(&cols, 2), Some(1));
        assert_eq!(find_col_in_row(&cols, 3), None);
        assert_eq!(find_col_in_row(&cols, 7), Some(3));
    }
}
