//! Preconditioned iterative solvers: PCG and PGMRES(m).
//!
//! Provides preconditioned variants of the Conjugate Gradient and GMRES(m)
//! solvers. A preconditioner approximates `M^{-1}` to accelerate convergence
//! by reducing the condition number of the effective system.
//!
//! # Algorithm — Preconditioned CG (PCG)
//!
//! Standard CG with a preconditioner M applied to the residual:
//! 1. r = b - A*x; z = M^{-1}*r; p = z; rz_old = r^T * z
//! 2. For each iteration:
//!    a. Ap = A * p
//!    b. alpha = rz_old / (p^T * Ap)
//!    c. x += alpha * p
//!    d. r -= alpha * Ap
//!    e. z = M^{-1} * r
//!    f. rz_new = r^T * z
//!    g. If ||r|| < tol * ||b||: converged
//!    h. beta = rz_new / rz_old
//!    i. p = z + beta * p
//!    j. rz_old = rz_new
//!
//! # Algorithm — Preconditioned GMRES(m) (Left-preconditioned)
//!
//! Left-preconditioned GMRES builds the Krylov space from `M^{-1}*A` instead
//! of `A`, solving the preconditioned system `M^{-1} A x = M^{-1} b`.

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

// ---------------------------------------------------------------------------
// Preconditioner trait
// ---------------------------------------------------------------------------

/// Preconditioner trait for iterative solvers.
///
/// A preconditioner approximates the solution of `M * z = r`, where `M`
/// is some approximation of the system matrix `A`. Applying the preconditioner
/// should be cheaper than solving the original system.
pub trait Preconditioner<T: GpuFloat> {
    /// Apply preconditioner: approximately solve `M * z = r`.
    ///
    /// Reads from `r` and writes the result into `z`.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError`] if the preconditioner application fails.
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()>;
}

/// Identity preconditioner (no preconditioning).
///
/// Simply copies `r` to `z`, equivalent to `M = I`.
pub struct IdentityPreconditioner;

impl<T: GpuFloat> Preconditioner<T> for IdentityPreconditioner {
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        let n = r.len().min(z.len());
        z[..n].copy_from_slice(&r[..n]);
        Ok(())
    }
}

/// Jacobi (diagonal) preconditioner.
///
/// Stores the inverse diagonal of the system matrix. Applying the
/// preconditioner computes `z_i = r_i / a_ii` for each component.
pub struct JacobiPreconditioner<T: GpuFloat> {
    /// Inverse diagonal entries: `inv_diag[i] = 1.0 / a_ii`.
    inv_diag: Vec<T>,
}

impl<T: GpuFloat> JacobiPreconditioner<T> {
    /// Creates a Jacobi preconditioner from the diagonal of the matrix.
    ///
    /// # Arguments
    ///
    /// * `diagonal` — the main diagonal of the system matrix.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::InternalError`] if any diagonal entry is zero.
    pub fn new(diagonal: &[T]) -> SolverResult<Self> {
        let mut inv_diag = Vec::with_capacity(diagonal.len());
        for (i, &d) in diagonal.iter().enumerate() {
            let dv = to_f64(d);
            if dv.abs() < 1e-300 {
                return Err(SolverError::InternalError(format!(
                    "JacobiPreconditioner: zero diagonal at index {i}"
                )));
            }
            inv_diag.push(from_f64(1.0 / dv));
        }
        Ok(Self { inv_diag })
    }
}

impl<T: GpuFloat> Preconditioner<T> for JacobiPreconditioner<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        let n = r.len().min(z.len()).min(self.inv_diag.len());
        for i in 0..n {
            z[i] = mul_t(r[i], self.inv_diag[i]);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Iterative solver result
// ---------------------------------------------------------------------------

/// Result from a preconditioned iterative solver.
#[derive(Debug, Clone)]
pub struct IterativeSolverResult<T> {
    /// Number of iterations performed.
    pub iterations: u32,
    /// Final residual norm (||b - A*x||).
    pub residual: T,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Preconditioned CG
// ---------------------------------------------------------------------------

/// Configuration for the Preconditioned Conjugate Gradient solver.
#[derive(Debug, Clone)]
pub struct PcgConfig {
    /// Maximum number of iterations.
    pub max_iter: u32,
    /// Convergence tolerance (relative to ||b||).
    pub tol: f64,
}

impl Default for PcgConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

/// Solves `A * x = b` using the Preconditioned Conjugate Gradient method.
///
/// The matrix A is not passed directly. Instead, the caller provides a closure
/// `spmv` that computes `y = A * x` given `x` and `y` buffers, plus a
/// preconditioner `P` that approximately solves `M * z = r`.
///
/// On entry, `x` should contain an initial guess (e.g., zeros). On exit, `x`
/// contains the approximate solution.
///
/// # Arguments
///
/// * `_handle` — solver handle (reserved for future GPU-accelerated variants).
/// * `spmv` — closure computing `y = A * x`: `spmv(x, y)`.
/// * `precond` — preconditioner implementing [`Preconditioner`].
/// * `b` — right-hand side vector (length n).
/// * `x` — initial guess / solution vector (length n), modified in-place.
/// * `n` — system dimension.
/// * `config` — solver configuration (tolerance, max iterations).
///
/// # Returns
///
/// An [`IterativeSolverResult`] containing iteration count, residual, and
/// convergence status.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] if vector lengths are invalid.
/// Returns [`SolverError::InternalError`] if a breakdown is detected.
pub fn preconditioned_cg<T, P, F>(
    _handle: &SolverHandle,
    spmv: F,
    precond: &P,
    b: &[T],
    x: &mut [T],
    n: u32,
    config: &PcgConfig,
) -> SolverResult<IterativeSolverResult<T>>
where
    T: GpuFloat,
    P: Preconditioner<T>,
    F: Fn(&[T], &mut [T]) -> SolverResult<()>,
{
    let n_usize = n as usize;

    // Validate dimensions.
    if b.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "preconditioned_cg: b length ({}) < n ({n})",
            b.len()
        )));
    }
    if x.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "preconditioned_cg: x length ({}) < n ({n})",
            x.len()
        )));
    }
    if n == 0 {
        return Ok(IterativeSolverResult {
            iterations: 0,
            residual: T::gpu_zero(),
            converged: true,
        });
    }

    // Compute ||b|| for relative convergence check.
    let b_norm = vec_norm(b, n_usize);
    let abs_tol = if b_norm > 0.0 {
        config.tol * b_norm
    } else {
        for xi in x.iter_mut().take(n_usize) {
            *xi = T::gpu_zero();
        }
        return Ok(IterativeSolverResult {
            iterations: 0,
            residual: T::gpu_zero(),
            converged: true,
        });
    };

    // r = b - A*x
    let mut r = vec![T::gpu_zero(); n_usize];
    let mut ap = vec![T::gpu_zero(); n_usize];
    spmv(x, &mut ap)?;
    for i in 0..n_usize {
        r[i] = sub_t(b[i], ap[i]);
    }

    // z = M^{-1} * r
    let mut z = vec![T::gpu_zero(); n_usize];
    precond.apply(&r, &mut z)?;

    // p = z
    let mut p = z.clone();

    // rz_old = r^T * z
    let mut rz_old = dot_product(&r, &z, n_usize);

    let r_norm = vec_norm(&r, n_usize);
    if r_norm < abs_tol {
        return Ok(IterativeSolverResult {
            iterations: 0,
            residual: from_f64(r_norm),
            converged: true,
        });
    }

    for iter in 0..config.max_iter {
        // Ap = A * p
        spmv(&p, &mut ap)?;

        // alpha = rz_old / (p^T * Ap)
        let pap = dot_product(&p, &ap, n_usize);
        if pap.abs() < 1e-300 {
            return Err(SolverError::InternalError(
                "preconditioned_cg: p^T * A * p is near zero (A may not be SPD)".into(),
            ));
        }
        let alpha = rz_old / pap;
        let alpha_t = from_f64(alpha);

        // x += alpha * p
        for i in 0..n_usize {
            x[i] = add_t(x[i], mul_t(alpha_t, p[i]));
        }

        // r -= alpha * Ap
        for i in 0..n_usize {
            r[i] = sub_t(r[i], mul_t(alpha_t, ap[i]));
        }

        // Check convergence on ||r||.
        let r_norm_now = vec_norm(&r, n_usize);
        if r_norm_now < abs_tol {
            return Ok(IterativeSolverResult {
                iterations: iter + 1,
                residual: from_f64(r_norm_now),
                converged: true,
            });
        }

        // z = M^{-1} * r
        precond.apply(&r, &mut z)?;

        // rz_new = r^T * z
        let rz_new = dot_product(&r, &z, n_usize);

        // beta = rz_new / rz_old
        if rz_old.abs() < 1e-300 {
            return Err(SolverError::InternalError(
                "preconditioned_cg: rz_old is near zero".into(),
            ));
        }
        let beta = rz_new / rz_old;
        let beta_t = from_f64(beta);

        // p = z + beta * p
        for i in 0..n_usize {
            p[i] = add_t(z[i], mul_t(beta_t, p[i]));
        }

        rz_old = rz_new;
    }

    let final_norm = vec_norm(&r, n_usize);
    Ok(IterativeSolverResult {
        iterations: config.max_iter,
        residual: from_f64(final_norm),
        converged: false,
    })
}

// ---------------------------------------------------------------------------
// Preconditioned GMRES(m) with restart
// ---------------------------------------------------------------------------

/// Configuration for the Preconditioned GMRES(m) solver.
#[derive(Debug, Clone)]
pub struct PgmresConfig {
    /// Maximum total number of iterations (across all restarts).
    pub max_iter: u32,
    /// Convergence tolerance (relative to ||b||).
    pub tol: f64,
    /// Restart parameter: number of Arnoldi steps before restarting.
    pub restart: u32,
}

impl Default for PgmresConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
            restart: 30,
        }
    }
}

/// Solves `A * x = b` using left-preconditioned GMRES(m) with restart.
///
/// Left-preconditioned GMRES solves `M^{-1} A x = M^{-1} b` using the
/// Arnoldi process applied to the operator `M^{-1} A`.
///
/// On entry, `x` should contain an initial guess. On exit, `x` contains
/// the approximate solution.
///
/// # Arguments
///
/// * `_handle` — solver handle (reserved for future GPU variants).
/// * `spmv` — closure computing `y = A * x`: `spmv(x, y)`.
/// * `precond` — preconditioner implementing [`Preconditioner`].
/// * `b` — right-hand side vector (length n).
/// * `x` — initial guess / solution vector (length n), modified in-place.
/// * `n` — system dimension.
/// * `config` — solver configuration.
///
/// # Returns
///
/// An [`IterativeSolverResult`] with iteration count, residual, and convergence.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
pub fn preconditioned_gmres<T, P, F>(
    _handle: &SolverHandle,
    spmv: F,
    precond: &P,
    b: &[T],
    x: &mut [T],
    n: u32,
    config: &PgmresConfig,
) -> SolverResult<IterativeSolverResult<T>>
where
    T: GpuFloat,
    P: Preconditioner<T>,
    F: Fn(&[T], &mut [T]) -> SolverResult<()>,
{
    let n_usize = n as usize;

    // Validate dimensions.
    if b.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "preconditioned_gmres: b length ({}) < n ({n})",
            b.len()
        )));
    }
    if x.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "preconditioned_gmres: x length ({}) < n ({n})",
            x.len()
        )));
    }
    if n == 0 {
        return Ok(IterativeSolverResult {
            iterations: 0,
            residual: T::gpu_zero(),
            converged: true,
        });
    }

    let b_norm = vec_norm(b, n_usize);
    let abs_tol = if b_norm > 0.0 {
        config.tol * b_norm
    } else {
        for xi in x.iter_mut().take(n_usize) {
            *xi = T::gpu_zero();
        }
        return Ok(IterativeSolverResult {
            iterations: 0,
            residual: T::gpu_zero(),
            converged: true,
        });
    };

    let m = config.restart.min(n) as usize;
    let mut total_iters = 0_u32;

    // Outer restart loop.
    while total_iters < config.max_iter {
        let (iters, converged, res_norm) = pgmres_cycle(
            &spmv,
            precond,
            b,
            x,
            n_usize,
            m,
            abs_tol,
            config.max_iter.saturating_sub(total_iters),
        )?;
        total_iters += iters;

        if converged {
            return Ok(IterativeSolverResult {
                iterations: total_iters,
                residual: from_f64(res_norm),
                converged: true,
            });
        }

        if iters == 0 {
            break; // No progress in this cycle.
        }
    }

    // Compute final residual.
    let mut r = vec![T::gpu_zero(); n_usize];
    let mut ax = vec![T::gpu_zero(); n_usize];
    spmv(x, &mut ax)?;
    for i in 0..n_usize {
        r[i] = sub_t(b[i], ax[i]);
    }
    let r_norm = vec_norm(&r, n_usize);

    Ok(IterativeSolverResult {
        iterations: total_iters,
        residual: from_f64(r_norm),
        converged: r_norm < abs_tol,
    })
}

/// One left-preconditioned GMRES cycle.
///
/// Returns `(iters, converged, residual_norm)`.
#[allow(clippy::too_many_arguments)]
fn pgmres_cycle<T, P, F>(
    spmv: &F,
    precond: &P,
    b: &[T],
    x: &mut [T],
    n: usize,
    m: usize,
    abs_tol: f64,
    max_iters: u32,
) -> SolverResult<(u32, bool, f64)>
where
    T: GpuFloat,
    P: Preconditioner<T>,
    F: Fn(&[T], &mut [T]) -> SolverResult<()>,
{
    // Compute initial residual r = b - A*x.
    let mut r = vec![T::gpu_zero(); n];
    let mut ax = vec![T::gpu_zero(); n];
    spmv(x, &mut ax)?;
    for i in 0..n {
        r[i] = sub_t(b[i], ax[i]);
    }

    // Apply preconditioner to initial residual: z = M^{-1} * r
    let mut z = vec![T::gpu_zero(); n];
    precond.apply(&r, &mut z)?;

    let beta = vec_norm(&z, n);

    if beta < abs_tol {
        return Ok((0, true, vec_norm(&r, n)));
    }

    // Arnoldi basis vectors.
    let mut v_basis: Vec<Vec<T>> = Vec::with_capacity(m + 1);

    // v_0 = z / beta
    let inv_beta = from_f64(1.0 / beta);
    let v0: Vec<T> = z.iter().map(|&zi| mul_t(zi, inv_beta)).collect();
    v_basis.push(v0);

    // Upper Hessenberg matrix H (m+1 x m), stored column-major.
    let mut h = vec![vec![0.0_f64; m + 1]; m];

    // Givens rotation parameters.
    let mut cs = vec![0.0_f64; m];
    let mut sn = vec![0.0_f64; m];

    // Right-hand side for Hessenberg least squares: g = beta * e_1.
    let mut g = vec![0.0_f64; m + 1];
    g[0] = beta;

    let mut j = 0;
    let max_j = m.min(max_iters as usize);
    let mut converged = false;

    while j < max_j {
        // w = M^{-1} * A * v_j
        let mut av = vec![T::gpu_zero(); n];
        spmv(&v_basis[j], &mut av)?;
        let mut w = vec![T::gpu_zero(); n];
        precond.apply(&av, &mut w)?;

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
            // Lucky breakdown.
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

        // Check convergence.
        if g[j].abs() < abs_tol {
            converged = true;
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

    // Compute actual residual norm for reporting.
    let mut r_final = vec![T::gpu_zero(); n];
    let mut ax_final = vec![T::gpu_zero(); n];
    spmv(x, &mut ax_final)?;
    for i in 0..n {
        r_final[i] = sub_t(b[i], ax_final[i]);
    }
    let r_norm = vec_norm(&r_final, n);

    Ok((j as u32, converged || r_norm < abs_tol, r_norm))
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

    // --- Config tests ---

    #[test]
    fn pcg_config_default() {
        let cfg = PcgConfig::default();
        assert_eq!(cfg.max_iter, 1000);
        assert!((cfg.tol - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn pgmres_config_default() {
        let cfg = PgmresConfig::default();
        assert_eq!(cfg.max_iter, 1000);
        assert!((cfg.tol - 1e-6).abs() < 1e-15);
        assert_eq!(cfg.restart, 30);
    }

    #[test]
    fn pcg_config_custom() {
        let cfg = PcgConfig {
            max_iter: 500,
            tol: 1e-10,
        };
        assert_eq!(cfg.max_iter, 500);
        assert!((cfg.tol - 1e-10).abs() < 1e-20);
    }

    #[test]
    fn pgmres_config_custom() {
        let cfg = PgmresConfig {
            max_iter: 2000,
            tol: 1e-8,
            restart: 50,
        };
        assert_eq!(cfg.max_iter, 2000);
        assert_eq!(cfg.restart, 50);
    }

    // --- Identity preconditioner tests ---

    #[test]
    fn identity_preconditioner_copies() {
        let r = [1.0_f64, 2.0, 3.0];
        let mut z = [0.0_f64; 3];
        let p = IdentityPreconditioner;
        let result = p.apply(&r, &mut z);
        assert!(result.is_ok());
        for i in 0..3 {
            assert!((z[i] - r[i]).abs() < 1e-15);
        }
    }

    // --- Jacobi preconditioner tests ---

    #[test]
    fn jacobi_preconditioner_basic() {
        let diag = [2.0_f64, 4.0, 8.0];
        let jp = JacobiPreconditioner::new(&diag);
        assert!(jp.is_ok());
        let jp = match jp {
            Ok(v) => v,
            Err(e) => panic!("test: {e}"),
        };

        let r = [2.0_f64, 8.0, 16.0];
        let mut z = [0.0_f64; 3];
        let result = jp.apply(&r, &mut z);
        assert!(result.is_ok());
        assert!((z[0] - 1.0).abs() < 1e-12);
        assert!((z[1] - 2.0).abs() < 1e-12);
        assert!((z[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn jacobi_preconditioner_zero_diagonal() {
        let diag = [1.0_f64, 0.0, 3.0];
        let jp = JacobiPreconditioner::new(&diag);
        assert!(jp.is_err());
    }

    // --- IterativeSolverResult tests ---

    #[test]
    fn iterative_result_fields() {
        let result: IterativeSolverResult<f64> = IterativeSolverResult {
            iterations: 42,
            residual: 1e-8,
            converged: true,
        };
        assert_eq!(result.iterations, 42);
        assert!(result.converged);
        assert!((result.residual - 1e-8).abs() < 1e-15);
    }

    // --- Givens rotation tests ---

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

    // --- Dot product / norm tests ---

    #[test]
    fn dot_product_basic() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        assert!((dot_product(&a, &b, 3) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn vec_norm_basic() {
        let v = [3.0_f64, 4.0];
        assert!((vec_norm(&v, 2) - 5.0).abs() < 1e-10);
    }

    // --- Arithmetic helper tests ---

    #[test]
    fn add_sub_mul_helpers() {
        let a = 3.0_f64;
        let b = 4.0_f64;
        assert!((to_f64(add_t(a, b)) - 7.0).abs() < 1e-15);
        assert!((to_f64(sub_t(a, b)) - (-1.0)).abs() < 1e-15);
        assert!((to_f64(mul_t(a, b)) - 12.0).abs() < 1e-15);
    }

    #[test]
    fn f32_conversion_roundtrip() {
        let val = 3.15_f32;
        let as_f64 = to_f64(val);
        let back: f32 = from_f64(as_f64);
        assert!((back - val).abs() < 1e-6);
    }

    #[test]
    fn f64_conversion_roundtrip() {
        let val = std::f64::consts::PI;
        let as_f64 = to_f64(val);
        let back: f64 = from_f64(as_f64);
        assert!((back - val).abs() < 1e-15);
    }

    #[test]
    fn jacobi_preconditioner_f32() {
        let diag = [2.0_f32, 4.0, 8.0];
        let jp = match JacobiPreconditioner::new(&diag) {
            Ok(v) => v,
            Err(e) => panic!("test: {e}"),
        };

        let r = [2.0_f32, 8.0, 16.0];
        let mut z = [0.0_f32; 3];
        let result = jp.apply(&r, &mut z);
        assert!(result.is_ok());
        assert!((z[0] - 1.0_f32).abs() < 1e-5);
        assert!((z[1] - 2.0_f32).abs() < 1e-5);
        assert!((z[2] - 2.0_f32).abs() < 1e-5);
    }

    #[test]
    fn identity_preconditioner_f32() {
        let r = [1.0_f32, 2.0, 3.0];
        let mut z = [0.0_f32; 3];
        let p = IdentityPreconditioner;
        let result = p.apply(&r, &mut z);
        assert!(result.is_ok());
        for i in 0..3 {
            assert!((z[i] - r[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn iterative_result_not_converged() {
        let result: IterativeSolverResult<f64> = IterativeSolverResult {
            iterations: 1000,
            residual: 1e-2,
            converged: false,
        };
        assert_eq!(result.iterations, 1000);
        assert!(!result.converged);
    }
}
