//! Flexible GMRES (FGMRES) iterative solver.
//!
//! Flexible GMRES allows a different (possibly nonlinear or variable)
//! preconditioner at each iteration. Unlike standard right-preconditioned
//! GMRES which stores only the Krylov basis vectors V, FGMRES also stores
//! the preconditioned vectors Z_j = M_j^{-1} * v_j separately. The final
//! solution update uses the Z vectors rather than V.
//!
//! # Algorithm
//!
//! 1. Compute r_0 = b - A*x_0, beta = ||r_0||, v_1 = r_0 / beta.
//! 2. For j = 1, 2, ..., m:
//!    a. z_j = M_j^{-1} * v_j  (preconditioner may vary per iteration)
//!    b. w = A * z_j
//!    c. Modified Gram-Schmidt: orthogonalize w against v_1, ..., v_j
//!    d. h_{j+1,j} = ||w||; v_{j+1} = w / h_{j+1,j}
//!    e. Apply previous Givens rotations; compute new rotation
//!    f. Check convergence
//! 3. Solve the Hessenberg least squares: H * y = g
//! 4. Update: x = x_0 + Z * y  (NOT x_0 + V * y)
//!
//! The key difference from standard GMRES is step 4: Z vectors are used
//! instead of V vectors.

#![allow(dead_code)]

use oxicuda_blas::GpuFloat;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;
use crate::sparse::preconditioned::{IterativeSolverResult, Preconditioner};

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
// Configuration
// ---------------------------------------------------------------------------

/// FGMRES configuration.
#[derive(Debug, Clone)]
pub struct FgmresConfig {
    /// Restart parameter: maximum Arnoldi steps per cycle.
    pub restart: usize,
    /// Maximum total iterations (across all restarts).
    pub max_iter: usize,
    /// Convergence tolerance (relative to ||b||).
    pub tol: f64,
}

impl Default for FgmresConfig {
    fn default() -> Self {
        Self {
            restart: 30,
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solves `A * x = b` using Flexible GMRES with variable preconditioner.
///
/// On entry, `x` should contain an initial guess. On exit, `x` contains
/// the approximate solution.
///
/// # Type Parameters
///
/// * `T` — floating-point type (f32 or f64).
/// * `P` — preconditioner type (may vary internally per iteration).
/// * `F` — closure computing `y = A * x`.
///
/// # Arguments
///
/// * `_handle` — solver handle (reserved for future GPU variants).
/// * `spmv` — closure computing `y = A * x`: `spmv(x, y)`.
/// * `precond` — preconditioner implementing [`Preconditioner`].
/// * `b` — right-hand side vector (length n).
/// * `x` — initial guess / solution vector (length n), modified in-place.
/// * `config` — FGMRES configuration.
///
/// # Returns
///
/// An [`IterativeSolverResult`] with iteration count, residual, and convergence.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
pub fn fgmres<T, P, F>(
    _handle: &SolverHandle,
    spmv: F,
    precond: &P,
    b: &[T],
    x: &mut [T],
    config: &FgmresConfig,
) -> SolverResult<IterativeSolverResult<T>>
where
    T: GpuFloat,
    P: Preconditioner<T>,
    F: Fn(&[T], &mut [T]) -> SolverResult<()>,
{
    let n = b.len();
    if x.len() < n {
        return Err(SolverError::DimensionMismatch(format!(
            "fgmres: x length ({}) < b length ({n})",
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

    let b_norm = vec_norm(b, n);
    let abs_tol = if b_norm > 0.0 {
        config.tol * b_norm
    } else {
        for xi in x.iter_mut().take(n) {
            *xi = T::gpu_zero();
        }
        return Ok(IterativeSolverResult {
            iterations: 0,
            residual: T::gpu_zero(),
            converged: true,
        });
    };

    let m = config.restart.min(n);
    let mut total_iters = 0_u32;

    // Outer restart loop.
    while (total_iters as usize) < config.max_iter {
        let remaining = config.max_iter.saturating_sub(total_iters as usize);
        let (iters, converged, res_norm) =
            fgmres_cycle(&spmv, precond, b, x, n, m, abs_tol, remaining)?;
        total_iters += iters;

        if converged {
            return Ok(IterativeSolverResult {
                iterations: total_iters,
                residual: from_f64(res_norm),
                converged: true,
            });
        }

        if iters == 0 {
            break; // No progress.
        }
    }

    // Compute final residual.
    let mut r = vec![T::gpu_zero(); n];
    let mut ax = vec![T::gpu_zero(); n];
    spmv(x, &mut ax)?;
    for i in 0..n {
        r[i] = sub_t(b[i], ax[i]);
    }
    let r_norm = vec_norm(&r, n);

    Ok(IterativeSolverResult {
        iterations: total_iters,
        residual: from_f64(r_norm),
        converged: r_norm < abs_tol,
    })
}

// ---------------------------------------------------------------------------
// FGMRES cycle
// ---------------------------------------------------------------------------

/// One FGMRES cycle: runs up to `m` Arnoldi steps with flexible preconditioning,
/// solves the Hessenberg least squares, and updates `x` using Z vectors.
///
/// Returns `(iters, converged, residual_norm)`.
#[allow(clippy::too_many_arguments)]
fn fgmres_cycle<T, P, F>(
    spmv: &F,
    precond: &P,
    b: &[T],
    x: &mut [T],
    n: usize,
    m: usize,
    abs_tol: f64,
    max_iters: usize,
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
    let beta = vec_norm(&r, n);

    if beta < abs_tol {
        return Ok((0, true, beta));
    }

    // V basis vectors (Krylov space).
    let mut v_basis: Vec<Vec<T>> = Vec::with_capacity(m + 1);
    // Z vectors (preconditioned basis) — this is the key FGMRES difference.
    let mut z_basis: Vec<Vec<T>> = Vec::with_capacity(m);

    // v_0 = r / beta
    let inv_beta = from_f64(1.0 / beta);
    let v0: Vec<T> = r.iter().map(|&ri| mul_t(ri, inv_beta)).collect();
    v_basis.push(v0);

    // Upper Hessenberg matrix H (m+1 x m), stored column-major as Vec<Vec<f64>>.
    let mut h = vec![vec![0.0_f64; m + 1]; m];

    // Givens rotation parameters.
    let mut cs = vec![0.0_f64; m];
    let mut sn = vec![0.0_f64; m];

    // Right-hand side for Hessenberg least squares: g = beta * e_1.
    let mut g = vec![0.0_f64; m + 1];
    g[0] = beta;

    let mut j = 0;
    let max_j = m.min(max_iters);
    let mut converged = false;

    while j < max_j {
        // FGMRES step a: z_j = M_j^{-1} * v_j
        let mut z_j = vec![T::gpu_zero(); n];
        precond.apply(&v_basis[j], &mut z_j)?;
        z_basis.push(z_j);

        // FGMRES step b: w = A * z_j
        let mut w = vec![T::gpu_zero(); n];
        spmv(&z_basis[j], &mut w)?;

        // FGMRES step c: Modified Gram-Schmidt orthogonalization.
        for i in 0..=j {
            h[j][i] = dot_product(&v_basis[i], &w, n);
            let h_ij_t = from_f64(h[j][i]);
            for k in 0..n {
                w[k] = sub_t(w[k], mul_t(h_ij_t, v_basis[i][k]));
            }
        }

        // FGMRES step d: Normalize w to get v_{j+1}.
        let w_norm = vec_norm(&w, n);
        h[j][j + 1] = w_norm;

        if w_norm > 1e-300 {
            let inv_w = from_f64(1.0 / w_norm);
            let vj1: Vec<T> = w.iter().map(|&wi| mul_t(wi, inv_w)).collect();
            v_basis.push(vj1);
        } else {
            // Lucky breakdown.
            let vj1 = vec![T::gpu_zero(); n];
            v_basis.push(vj1);
        }

        // FGMRES step e: Apply previous Givens rotations to new column of H.
        for i in 0..j {
            let tmp = cs[i] * h[j][i] + sn[i] * h[j][i + 1];
            h[j][i + 1] = -sn[i] * h[j][i] + cs[i] * h[j][i + 1];
            h[j][i] = tmp;
        }

        // Compute new Givens rotation.
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

        // FGMRES step f: Check convergence.
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

    // FGMRES update: x += Z * y  (NOT V * y — this is the key difference).
    for i in 0..j {
        let yi_t = from_f64(y[i]);
        for k in 0..n {
            x[k] = add_t(x[k], mul_t(yi_t, z_basis[i][k]));
        }
    }

    // Compute actual residual for reporting.
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
    use crate::sparse::preconditioned::IdentityPreconditioner;

    #[test]
    fn fgmres_config_default() {
        let cfg = FgmresConfig::default();
        assert_eq!(cfg.restart, 30);
        assert_eq!(cfg.max_iter, 1000);
        assert!((cfg.tol - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn fgmres_config_custom() {
        let cfg = FgmresConfig {
            restart: 50,
            max_iter: 2000,
            tol: 1e-10,
        };
        assert_eq!(cfg.restart, 50);
        assert_eq!(cfg.max_iter, 2000);
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
    fn givens_rotation_zero_a() {
        let (cs, sn) = givens_rotation(0.0, 3.0);
        assert!(cs.abs() < 1e-15);
        assert!((sn - 1.0).abs() < 1e-15);
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

    #[test]
    fn vec_norm_345() {
        let v = [3.0_f64, 4.0];
        assert!((vec_norm(&v, 2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn add_sub_mul_helpers() {
        let a = 3.0_f64;
        let b = 4.0_f64;
        assert!((to_f64(add_t(a, b)) - 7.0).abs() < 1e-15);
        assert!((to_f64(sub_t(a, b)) - (-1.0)).abs() < 1e-15);
        assert!((to_f64(mul_t(a, b)) - 12.0).abs() < 1e-15);
    }

    #[test]
    fn identity_preconditioner_with_fgmres() {
        let _precond = IdentityPreconditioner;
        // Verify that IdentityPreconditioner implements Preconditioner.
        let r = [1.0_f64, 2.0, 3.0];
        let mut z = [0.0_f64; 3];
        let result = _precond.apply(&r, &mut z);
        assert!(result.is_ok());
        assert!((z[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn f64_conversion_roundtrip() {
        let val = std::f64::consts::PI;
        let as_f64 = to_f64(val);
        let back: f64 = from_f64(as_f64);
        assert!((back - val).abs() < 1e-15);
    }

    #[test]
    fn f32_conversion_roundtrip() {
        let val = std::f32::consts::PI;
        let as_f64 = to_f64(val);
        let back: f32 = from_f64(as_f64);
        assert!((back - val).abs() < 1e-5);
    }
}
