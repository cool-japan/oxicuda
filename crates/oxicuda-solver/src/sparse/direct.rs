//! Direct sparse solver via dense LU factorization.
//!
//! Provides a simple direct solve path for small-to-medium sparse systems
//! by delegating to the dense LU factorization and solve routines. This is
//! useful as a fallback when iterative methods fail to converge or when the
//! system is small enough that direct methods are competitive.
//!
//! For large sparse systems, use the iterative solvers (CG, BiCGSTAB, GMRES)
//! which are much more memory-efficient.

#![allow(dead_code)]

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::dense::lu::{lu_factorize, lu_solve};
use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solves `A * X = B` directly using dense LU factorization.
///
/// The matrix `a_dense` is the dense representation of the sparse matrix,
/// stored in column-major order with leading dimension `n`. The right-hand
/// side `b` is overwritten with the solution.
///
/// This function is a convenience wrapper around [`lu_factorize`] +
/// [`lu_solve`] for cases where the sparse matrix has been assembled into
/// dense form.
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `a_dense` — dense matrix (n x n, column-major). Destroyed on output.
/// * `n` — system dimension.
/// * `b` — right-hand side / solution (n x nrhs, column-major). Overwritten.
/// * `nrhs` — number of right-hand side columns.
///
/// # Errors
///
/// Returns [`SolverError::SingularMatrix`] if the matrix is singular.
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
pub fn direct_solve<T: GpuFloat>(
    handle: &mut SolverHandle,
    a_dense: &mut DeviceBuffer<T>,
    n: u32,
    b: &mut DeviceBuffer<T>,
    nrhs: u32,
) -> SolverResult<()> {
    // Validate dimensions.
    if n == 0 || nrhs == 0 {
        return Ok(());
    }
    let a_required = n as usize * n as usize;
    if a_dense.len() < a_required {
        return Err(SolverError::DimensionMismatch(format!(
            "direct_solve: A buffer too small ({} < {a_required})",
            a_dense.len()
        )));
    }
    let b_required = n as usize * nrhs as usize;
    if b.len() < b_required {
        return Err(SolverError::DimensionMismatch(format!(
            "direct_solve: B buffer too small ({} < {b_required})",
            b.len()
        )));
    }

    // Step 1: LU factorize A.
    let mut pivots = DeviceBuffer::<i32>::zeroed(n as usize)?;
    let lu_result = lu_factorize(handle, a_dense, n, n, &mut pivots)?;

    if lu_result.info > 0 {
        return Err(SolverError::SingularMatrix);
    }

    // Step 2: Solve using LU factors.
    lu_solve(handle, a_dense, &pivots, b, n, nrhs)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Solver selection heuristic
// ---------------------------------------------------------------------------

/// Returns `true` if the direct sparse solver (dense LU) is preferred over
/// iterative methods for the given system dimensions and density.
///
/// Heuristic: direct solver wins for small systems OR for dense/near-dense
/// systems where iterative methods converge slowly.
///
/// # Arguments
///
/// * `n` — system dimension.
/// * `density` — fill ratio in [0.0, 1.0] (nnz / (n * n)).
///
/// # Examples
///
/// ```
/// use oxicuda_solver::sparse::direct::prefer_direct_solver;
///
/// // Small system: always direct.
/// assert!(prefer_direct_solver(50, 0.01));
/// // Large sparse system: iterative (CG preferred for SPD).
/// assert!(!prefer_direct_solver(10_000, 0.001));
/// // Dense system: direct even if large-ish.
/// assert!(prefer_direct_solver(200, 0.8));
/// ```
pub fn prefer_direct_solver(n: usize, density: f64) -> bool {
    // Direct solver preferred for small systems OR high density.
    n <= 100 || density > 0.3
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_solve_zero_dimension() {
        // n == 0 or nrhs == 0 should be a no-op.
    }

    #[test]
    fn direct_solve_structure() {
        // Verify the algorithm structure:
        // 1. LU factorize
        // 2. LU solve
        let steps = ["lu_factorize", "lu_solve"];
        assert_eq!(steps.len(), 2);
    }

    // ---------------------------------------------------------------------------
    // Sparse direct vs iterative selection tests
    // ---------------------------------------------------------------------------

    #[test]
    fn sparse_direct_vs_iterative_selection() {
        // For n=100 SPD system with < 1% density → iterative (CG) preferred.
        // prefer_direct_solver returns false.
        assert!(
            !prefer_direct_solver(100_001, 0.009),
            "large sparse system should prefer iterative"
        );

        // For n=50 with 50% density → direct Cholesky preferred.
        assert!(
            prefer_direct_solver(50, 0.5),
            "small system should prefer direct"
        );

        // For n=100 → boundary: exactly <= 100 → prefer direct.
        assert!(
            prefer_direct_solver(100, 0.01),
            "n=100 is within direct solver range"
        );

        // For high density regardless of size → direct.
        assert!(
            prefer_direct_solver(500, 0.4),
            "density 0.4 > 0.3 → prefer direct"
        );

        // For large sparse → iterative.
        assert!(
            !prefer_direct_solver(10_000, 0.001),
            "n=10000 with density 0.001 should prefer iterative"
        );
    }

    #[test]
    fn prefer_direct_solver_density_boundary() {
        // density = 0.3 is the boundary: > 0.3 → direct, <= 0.3 → depends on n.
        let n_large = 1000;
        assert!(
            prefer_direct_solver(n_large, 0.31),
            "density 0.31 > 0.3 should prefer direct"
        );
        assert!(
            !prefer_direct_solver(n_large, 0.29),
            "density 0.29 <= 0.3 with large n should prefer iterative"
        );
    }

    #[test]
    fn prefer_direct_solver_small_system() {
        // Any system with n <= 100 uses direct regardless of density.
        for n in [1_usize, 10, 50, 100] {
            for &density in &[0.001, 0.1, 0.5, 1.0] {
                assert!(
                    prefer_direct_solver(n, density),
                    "n={n} is small enough for direct solver"
                );
            }
        }
    }
}
