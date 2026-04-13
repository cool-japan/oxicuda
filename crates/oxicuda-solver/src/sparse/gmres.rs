//! GMRES(m) (Generalized Minimal Residual with restart) iterative solver.
//!
//! Solves the linear system `A * x = b` for general matrices using the
//! GMRES algorithm with periodic restarts after `m` iterations.
//!
//! # Algorithm
//!
//! GMRES builds an orthonormal basis for the Krylov subspace
//! `K_m = span{r, A*r, A^2*r, ..., A^{m-1}*r}` via the Arnoldi process,
//! then solves a small least squares problem on the resulting upper
//! Hessenberg matrix using Givens rotations.
//!
//! After `m` iterations without convergence, the algorithm restarts with
//! the current best solution as the new initial guess.

#![allow(dead_code)]

use oxicuda_blas::GpuFloat;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;

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

/// Default restart parameter for GMRES.
const DEFAULT_RESTART: u32 = 30;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the GMRES(m) solver.
#[derive(Debug, Clone)]
pub struct GmresConfig {
    /// Maximum total number of iterations (across all restarts).
    pub max_iter: u32,
    /// Convergence tolerance (relative to ||b||).
    pub tol: f64,
    /// Restart parameter: number of Arnoldi steps before restarting.
    pub restart: u32,
}

impl Default for GmresConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
            restart: DEFAULT_RESTART,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solves `A * x = b` using GMRES(m) with restart.
///
/// On entry, `x` should contain an initial guess. On exit, `x` contains
/// the approximate solution.
///
/// # Arguments
///
/// * `_handle` — solver handle (reserved for future GPU-accelerated variants).
/// * `spmv` — closure computing `y = A * x`: `spmv(x, y)`.
/// * `b` — right-hand side vector (length n).
/// * `x` — initial guess / solution vector (length n), modified in-place.
/// * `n` — system dimension.
/// * `config` — solver configuration.
///
/// # Returns
///
/// The total number of matrix-vector products performed.
///
/// # Errors
///
/// Returns [`SolverError::ConvergenceFailure`] if the solver does not converge.
pub fn gmres_solve<T, F>(
    _handle: &SolverHandle,
    spmv: F,
    b: &[T],
    x: &mut [T],
    n: u32,
    config: &GmresConfig,
) -> SolverResult<u32>
where
    T: GpuFloat,
    F: Fn(&[T], &mut [T]) -> SolverResult<()>,
{
    let n_usize = n as usize;

    // Validate dimensions.
    if b.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "gmres_solve: b length ({}) < n ({n})",
            b.len()
        )));
    }
    if x.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "gmres_solve: x length ({}) < n ({n})",
            x.len()
        )));
    }
    if n == 0 {
        return Ok(0);
    }

    let b_norm = vec_norm(b, n_usize);
    let abs_tol = if b_norm > 0.0 {
        config.tol * b_norm
    } else {
        for xi in x.iter_mut().take(n_usize) {
            *xi = T::gpu_zero();
        }
        return Ok(0);
    };

    let m = config.restart.min(n) as usize;
    let mut total_iters = 0_u32;

    // Outer restart loop.
    while total_iters < config.max_iter {
        let iters = gmres_cycle(
            &spmv,
            b,
            x,
            n_usize,
            m,
            abs_tol,
            config.max_iter - total_iters,
        )?;
        total_iters += iters;

        // Check if we converged in this cycle.
        let mut r = vec![T::gpu_zero(); n_usize];
        let mut ax = vec![T::gpu_zero(); n_usize];
        spmv(x, &mut ax)?;
        for i in 0..n_usize {
            r[i] = sub_t(b[i], ax[i]);
        }
        total_iters += 1; // Count the residual check spmv.

        let r_norm = vec_norm(&r, n_usize);
        if r_norm < abs_tol {
            return Ok(total_iters);
        }

        if iters == 0 {
            break; // No progress in this cycle.
        }
    }

    // Compute final residual for error reporting.
    let mut r = vec![T::gpu_zero(); n_usize];
    let mut ax = vec![T::gpu_zero(); n_usize];
    spmv(x, &mut ax)?;
    for i in 0..n_usize {
        r[i] = sub_t(b[i], ax[i]);
    }
    let r_norm = vec_norm(&r, n_usize);

    if r_norm < abs_tol {
        Ok(total_iters)
    } else {
        Err(SolverError::ConvergenceFailure {
            iterations: total_iters,
            residual: r_norm,
        })
    }
}

// ---------------------------------------------------------------------------
// GMRES cycle (one restart)
// ---------------------------------------------------------------------------

/// One GMRES cycle: runs up to `m` Arnoldi steps, solves the Hessenberg
/// least squares problem, and updates `x`.
///
/// Returns the number of matrix-vector products performed in this cycle.
fn gmres_cycle<T, F>(
    spmv: &F,
    b: &[T],
    x: &mut [T],
    n: usize,
    m: usize,
    abs_tol: f64,
    max_iters: u32,
) -> SolverResult<u32>
where
    T: GpuFloat,
    F: Fn(&[T], &mut [T]) -> SolverResult<()>,
{
    // Compute initial residual r = b - A*x.
    let mut r = vec![T::gpu_zero(); n];
    let mut ax = vec![T::gpu_zero(); n];
    spmv(x, &mut ax)?;
    for i in 0..n {
        r[i] = sub_t(b[i], ax[i]);
    }
    let beta = vec_norm(&r, n);

    if beta < abs_tol {
        return Ok(0);
    }

    // Arnoldi basis vectors: V = [v_0, v_1, ..., v_m] where each v_i is length n.
    let mut v_basis: Vec<Vec<T>> = Vec::with_capacity(m + 1);

    // v_0 = r / beta
    let inv_beta = from_f64(1.0 / beta);
    let v0: Vec<T> = r.iter().map(|&ri| mul_t(ri, inv_beta)).collect();
    v_basis.push(v0);

    // Upper Hessenberg matrix H (m+1 x m), stored column-major as Vec<Vec<f64>>.
    let mut h = vec![vec![0.0_f64; m + 1]; m];

    // Givens rotation parameters.
    let mut cs = vec![0.0_f64; m];
    let mut sn = vec![0.0_f64; m];

    // Right-hand side for the Hessenberg least squares: g = beta * e_1.
    let mut g = vec![0.0_f64; m + 1];
    g[0] = beta;

    let mut j = 0;
    let max_j = m.min(max_iters as usize);

    while j < max_j {
        // Arnoldi step: w = A * v_j.
        let mut w = vec![T::gpu_zero(); n];
        spmv(&v_basis[j], &mut w)?;

        // Modified Gram-Schmidt orthogonalization.
        for i in 0..=j {
            h[j][i] = dot_product(&v_basis[i], &w, n);
            let h_ij_t = from_f64(h[j][i]);
            for k in 0..n {
                w[k] = sub_t(w[k], mul_t(h_ij_t, v_basis[i][k]));
            }
        }

        let w_norm = vec_norm(&w, n);
        h[j][j + 1] = w_norm;

        // Normalize w to get v_{j+1}.
        if w_norm > 1e-300 {
            let inv_w = from_f64(1.0 / w_norm);
            let vj1: Vec<T> = w.iter().map(|&wi| mul_t(wi, inv_w)).collect();
            v_basis.push(vj1);
        } else {
            // Lucky breakdown: w is in the span of existing basis.
            let vj1 = vec![T::gpu_zero(); n];
            v_basis.push(vj1);
        }

        // Apply previous Givens rotations to the new column of H.
        for i in 0..j {
            let tmp = cs[i] * h[j][i] + sn[i] * h[j][i + 1];
            h[j][i + 1] = -sn[i] * h[j][i] + cs[i] * h[j][i + 1];
            h[j][i] = tmp;
        }

        // Compute new Givens rotation for row (j, j+1).
        let (c, s) = givens_rotation(h[j][j], h[j][j + 1]);
        cs[j] = c;
        sn[j] = s;

        // Apply to H.
        h[j][j] = c * h[j][j] + s * h[j][j + 1];
        h[j][j + 1] = 0.0;

        // Apply to g.
        let tmp = cs[j] * g[j] + sn[j] * g[j + 1];
        g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
        g[j] = tmp;

        j += 1;

        // Check convergence: |g[j]| is the residual norm.
        if g[j].abs() < abs_tol {
            break;
        }
    }

    // Solve the upper triangular system H[0:j, 0:j] * y = g[0:j].
    let mut y = vec![0.0_f64; j];
    for i in (0..j).rev() {
        y[i] = g[i];
        for k in (i + 1)..j {
            y[i] -= h[k][i] * y[k];
        }
        if h[i][i].abs() > 1e-300 {
            y[i] /= h[i][i];
        }
    }

    // Update x: x += V * y.
    for i in 0..j {
        let yi_t = from_f64(y[i]);
        for k in 0..n {
            x[k] = add_t(x[k], mul_t(yi_t, v_basis[i][k]));
        }
    }

    Ok(j as u32)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-300 {
        return (1.0, 0.0);
    }
    if a.abs() < 1e-300 {
        return (0.0, if b >= 0.0 { 1.0 } else { -1.0 });
    }
    let r = (a * a + b * b).sqrt();
    (a / r, b / r)
}

fn dot_product<T: GpuFloat>(a: &[T], b: &[T], n: usize) -> f64 {
    let mut sum = 0.0_f64;
    for i in 0..n {
        sum += to_f64(a[i]) * to_f64(b[i]);
    }
    sum
}

fn vec_norm<T: GpuFloat>(v: &[T], n: usize) -> f64 {
    dot_product(v, v, n).sqrt()
}

fn add_t<T: GpuFloat>(a: T, b: T) -> T {
    from_f64(to_f64(a) + to_f64(b))
}

fn sub_t<T: GpuFloat>(a: T, b: T) -> T {
    from_f64(to_f64(a) - to_f64(b))
}

fn mul_t<T: GpuFloat>(a: T, b: T) -> T {
    from_f64(to_f64(a) * to_f64(b))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU-only GMRES solver for testing without a GPU handle.
    ///
    /// Mirrors `gmres_solve` but omits the `_handle` parameter, enabling
    /// pure host testing with a closure-based matrix-vector product.
    fn gmres_solve_cpu<T, F>(
        spmv: F,
        b: &[T],
        x: &mut [T],
        n: u32,
        config: &GmresConfig,
    ) -> SolverResult<u32>
    where
        T: GpuFloat,
        F: Fn(&[T], &mut [T]) -> SolverResult<()>,
    {
        let n_usize = n as usize;

        if b.len() < n_usize {
            return Err(SolverError::DimensionMismatch(format!(
                "gmres_solve_cpu: b length ({}) < n ({n})",
                b.len()
            )));
        }
        if x.len() < n_usize {
            return Err(SolverError::DimensionMismatch(format!(
                "gmres_solve_cpu: x length ({}) < n ({n})",
                x.len()
            )));
        }
        if n == 0 {
            return Ok(0);
        }

        let b_norm = vec_norm(b, n_usize);
        let abs_tol = if b_norm > 0.0 {
            config.tol * b_norm
        } else {
            for xi in x.iter_mut().take(n_usize) {
                *xi = T::gpu_zero();
            }
            return Ok(0);
        };

        let m = config.restart.min(n) as usize;
        let mut total_iters = 0_u32;

        while total_iters < config.max_iter {
            let iters = gmres_cycle(
                &spmv,
                b,
                x,
                n_usize,
                m,
                abs_tol,
                config.max_iter - total_iters,
            )?;
            total_iters += iters;

            let mut r = vec![T::gpu_zero(); n_usize];
            let mut ax = vec![T::gpu_zero(); n_usize];
            spmv(x, &mut ax)?;
            for i in 0..n_usize {
                r[i] = sub_t(b[i], ax[i]);
            }
            total_iters += 1;

            let r_norm = vec_norm(&r, n_usize);
            if r_norm < abs_tol {
                return Ok(total_iters);
            }

            if iters == 0 {
                break;
            }
        }

        let mut r = vec![T::gpu_zero(); n_usize];
        let mut ax = vec![T::gpu_zero(); n_usize];
        spmv(x, &mut ax)?;
        for i in 0..n_usize {
            r[i] = sub_t(b[i], ax[i]);
        }
        let r_norm = vec_norm(&r, n_usize);

        if r_norm < abs_tol {
            Ok(total_iters)
        } else {
            Err(SolverError::ConvergenceFailure {
                iterations: total_iters,
                residual: r_norm,
            })
        }
    }

    #[test]
    fn gmres_config_default() {
        let cfg = GmresConfig::default();
        assert_eq!(cfg.max_iter, 1000);
        assert!((cfg.tol - 1e-6).abs() < 1e-15);
        assert_eq!(cfg.restart, DEFAULT_RESTART);
    }

    #[test]
    fn gmres_config_custom() {
        let cfg = GmresConfig {
            max_iter: 500,
            tol: 1e-10,
            restart: 50,
        };
        assert_eq!(cfg.restart, 50);
    }

    #[test]
    fn givens_rotation_basic() {
        let (cs, sn) = givens_rotation(3.0, 4.0);
        let r = cs * 3.0 + sn * 4.0;
        assert!((r - 5.0).abs() < 1e-10);
    }

    #[test]
    fn givens_rotation_zero_b() {
        let (cs, sn) = givens_rotation(5.0, 0.0);
        assert!((cs - 1.0).abs() < 1e-15);
        assert!(sn.abs() < 1e-15);
    }

    #[test]
    fn dot_product_basic() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        assert!((dot_product(&a, &b, 3) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn vec_norm_unit() {
        let v = [1.0_f64, 0.0, 0.0];
        assert!((vec_norm(&v, 3) - 1.0).abs() < 1e-15);
    }

    /// GMRES converges on a 3×3 identity matrix in a single Arnoldi step.
    ///
    /// A = I, b = [3, 7, -2] → exact solution x = [3, 7, -2].
    /// The identity matrix has a single eigenvalue λ=1, so GMRES minimises
    /// the residual in exactly one step (Krylov space = full space).
    #[test]
    fn gmres_converges_identity_3x3() {
        let b = vec![3.0_f64, 7.0, -2.0];
        let mut x = vec![0.0_f64; 3];
        let config = GmresConfig {
            max_iter: 50,
            tol: 1e-10,
            restart: 10,
        };

        // A = I
        let spmv = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
            out.copy_from_slice(v);
            Ok(())
        };

        let _iters = gmres_solve_cpu(spmv, &b, &mut x, 3, &config)
            .expect("GMRES should converge on identity system");

        assert!((x[0] - 3.0).abs() < 1e-8, "x[0] = {} expected 3.0", x[0]);
        assert!((x[1] - 7.0).abs() < 1e-8, "x[1] = {} expected 7.0", x[1]);
        assert!(
            (x[2] - (-2.0)).abs() < 1e-8,
            "x[2] = {} expected -2.0",
            x[2]
        );
    }

    /// GMRES converges on a 4×4 tridiagonal SPD system in ≤ N steps.
    ///
    /// A = tridiag(-1, 2, -1), b = [1, 1, 1, 1], exact x = [2, 3, 3, 2].
    #[test]
    fn gmres_converges_tridiagonal_4x4() {
        let b = vec![1.0_f64, 1.0, 1.0, 1.0];
        let mut x = vec![0.0_f64; 4];
        let config = GmresConfig {
            max_iter: 200,
            tol: 1e-10,
            restart: 10,
        };

        // A = tridiag(-1, 2, -1), 4×4
        let spmv = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
            out[0] = 2.0 * v[0] - v[1];
            out[1] = -v[0] + 2.0 * v[1] - v[2];
            out[2] = -v[1] + 2.0 * v[2] - v[3];
            out[3] = -v[2] + 2.0 * v[3];
            Ok(())
        };

        let _iters = gmres_solve_cpu(spmv, &b, &mut x, 4, &config)
            .expect("GMRES should converge on tridiagonal system");

        assert!((x[0] - 2.0).abs() < 1e-7, "x[0] = {} expected 2.0", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-7, "x[1] = {} expected 3.0", x[1]);
        assert!((x[2] - 3.0).abs() < 1e-7, "x[2] = {} expected 3.0", x[2]);
        assert!((x[3] - 2.0).abs() < 1e-7, "x[3] = {} expected 2.0", x[3]);
    }

    /// GMRES with zero RHS returns immediately without iterating.
    #[test]
    fn gmres_zero_rhs_returns_zero() {
        let b = vec![0.0_f64; 3];
        let mut x = vec![1.0_f64; 3]; // non-zero initial guess
        let config = GmresConfig::default();

        let spmv = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
            out.copy_from_slice(v);
            Ok(())
        };

        let iters = gmres_solve_cpu(spmv, &b, &mut x, 3, &config).expect("zero RHS should succeed");
        assert_eq!(iters, 0);
        for &xi in &x {
            assert!(xi.abs() < 1e-15, "x should be zeroed for zero RHS");
        }
    }

    /// GMRES returns DimensionMismatch when b is shorter than n.
    #[test]
    fn gmres_dimension_mismatch() {
        let b = vec![1.0_f64]; // length 1, n = 3
        let mut x = vec![0.0_f64; 3];
        let config = GmresConfig::default();
        let spmv = |_: &[f64], _: &mut [f64]| -> SolverResult<()> { Ok(()) };
        let result = gmres_solve_cpu(spmv, &b, &mut x, 3, &config);
        assert!(matches!(result, Err(SolverError::DimensionMismatch(_))));
    }

    /// GMRES converges on a diagonal SPD system in at most N Arnoldi steps.
    ///
    /// A = diag(1, 4, 9), b = [1, 4, 9] → exact x = [1, 1, 1].
    #[test]
    fn gmres_converges_diagonal_spd() {
        let b = vec![1.0_f64, 4.0, 9.0];
        let mut x = vec![0.0_f64; 3];
        let config = GmresConfig {
            max_iter: 100,
            tol: 1e-10,
            restart: 10,
        };

        let spmv = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
            out[0] = 1.0 * v[0];
            out[1] = 4.0 * v[1];
            out[2] = 9.0 * v[2];
            Ok(())
        };

        let _iters = gmres_solve_cpu(spmv, &b, &mut x, 3, &config)
            .expect("GMRES should converge on diagonal SPD");

        assert!((x[0] - 1.0).abs() < 1e-8, "x[0] = {} expected 1.0", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-8, "x[1] = {} expected 1.0", x[1]);
        assert!((x[2] - 1.0).abs() < 1e-8, "x[2] = {} expected 1.0", x[2]);
    }
}
