//! Utility functions for ODE/PDE solvers.

use crate::error::{SolverError, SolverResult};

use super::pde::BoundaryCondition;
use super::types::OdeConfig;

/// Compute the numerical Jacobian of an ODE system by finite differences.
///
/// Returns an `n x n` matrix where `J[i][j] = df_i / dy_j`, approximated
/// by a first-order forward difference with perturbation `eps`.
pub fn numerical_jacobian(
    system: &dyn super::types::OdeSystem,
    t: f64,
    y: &[f64],
    eps: f64,
) -> SolverResult<Vec<Vec<f64>>> {
    let n = system.dim();
    let mut f0 = vec![0.0; n];
    system.rhs(t, y, &mut f0)?;

    let mut jac = vec![vec![0.0; n]; n];
    let mut y_pert = y.to_vec();
    let mut f_pert = vec![0.0; n];

    for j in 0..n {
        let h = eps * y[j].abs().max(1.0);
        y_pert[j] = y[j] + h;
        system.rhs(t, &y_pert, &mut f_pert)?;

        for i in 0..n {
            jac[i][j] = (f_pert[i] - f0[i]) / h;
        }
        y_pert[j] = y[j]; // restore
    }

    Ok(jac)
}

/// Solve a tridiagonal system using the Thomas algorithm.
///
/// `a` is the sub-diagonal (length n-1), `b` the main diagonal (length n),
/// `c` the super-diagonal (length n-1), and `d` the right-hand side (length n).
pub fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> SolverResult<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        if b[0].abs() < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }
        return Ok(vec![d[0] / b[0]]);
    }
    if a.len() < n - 1 || c.len() < n - 1 || d.len() < n {
        return Err(SolverError::DimensionMismatch(
            "solve_tridiagonal: inconsistent array lengths".to_string(),
        ));
    }

    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    if b[0].abs() < 1e-300 {
        return Err(SolverError::SingularMatrix);
    }

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i - 1] * c_prime[i - 1];
        if denom.abs() < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }
        if i < n - 1 {
            c_prime[i] = c[i] / denom;
        }
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom;
    }

    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

// =========================================================================
// Internal helpers
// =========================================================================

/// Validate ODE input dimensions and config.
pub(super) fn validate_ode_inputs(dim: usize, y0: &[f64], config: &OdeConfig) -> SolverResult<()> {
    if dim == 0 {
        return Err(SolverError::DimensionMismatch(
            "ODE system dimension must be > 0".to_string(),
        ));
    }
    if y0.len() != dim {
        return Err(SolverError::DimensionMismatch(format!(
            "y0 length ({}) != system dimension ({dim})",
            y0.len()
        )));
    }
    if config.dt <= 0.0 {
        return Err(SolverError::InternalError(
            "step size dt must be positive".to_string(),
        ));
    }
    if config.t_end <= config.t_start {
        return Err(SolverError::InternalError(
            "t_end must be greater than t_start".to_string(),
        ));
    }
    Ok(())
}

/// Euclidean norm of a vector.
pub(super) fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Apply 1-D boundary conditions in-place.
pub(super) fn apply_bc_1d(
    u: &mut [f64],
    bc_left: &BoundaryCondition,
    bc_right: &BoundaryCondition,
    nx: usize,
) {
    if nx == 0 {
        return;
    }
    match *bc_left {
        BoundaryCondition::Dirichlet(val) => {
            u[0] = val;
        }
        BoundaryCondition::Neumann(val) => {
            // Forward difference: (u[1] - u[0]) / dx = val => u[0] = u[1] - dx*val
            // dx is not available here, so we use a simple ghost-point approach:
            // u[0] = u[1] (zero Neumann approximation when val = 0)
            if nx > 1 {
                u[0] = u[1] - val; // caller must scale val by dx if needed
            }
        }
        BoundaryCondition::Periodic => {
            if nx > 1 {
                u[0] = u[nx - 2];
            }
        }
    }
    match *bc_right {
        BoundaryCondition::Dirichlet(val) => {
            u[nx - 1] = val;
        }
        BoundaryCondition::Neumann(val) => {
            if nx > 1 {
                u[nx - 1] = u[nx - 2] + val;
            }
        }
        BoundaryCondition::Periodic => {
            if nx > 1 {
                u[nx - 1] = u[1];
            }
        }
    }
}

/// Solve a dense linear system A*x = b using Gaussian elimination with
/// partial pivoting. Used internally by the implicit ODE solvers for
/// Newton iteration.
pub(super) fn solve_dense_system(a: &[Vec<f64>], b: &[f64]) -> SolverResult<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build augmented matrix
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n + 1);
        row.extend_from_slice(&a[i]);
        row.push(b[i]);
        aug.push(row);
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for (row, aug_row) in aug.iter().enumerate().skip(col + 1) {
            let val = aug_row[col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }

        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            // Cannot borrow aug mutably and immutably at same time,
            // so copy the pivot row values we need.
            let pivot_row: Vec<f64> = aug[col][col..=n].to_vec();
            for (j, &pv) in (col..=n).zip(pivot_row.iter()) {
                aug[row][j] -= factor * pv;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
}
