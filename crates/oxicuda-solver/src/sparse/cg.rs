//! Conjugate Gradient (CG) iterative solver.
//!
//! Solves the linear system `A * x = b` where A is symmetric positive definite.
//! The solver is matrix-free: it only requires a closure that computes the
//! matrix-vector product `y = A * x`.
//!
//! # Algorithm
//!
//! The standard Conjugate Gradient algorithm (Hestenes & Stiefel, 1952):
//! 1. r = b - A*x; p = r; rsold = r^T * r
//! 2. For each iteration:
//!    a. Ap = A * p
//!    b. alpha = rsold / (p^T * Ap)
//!    c. x += alpha * p
//!    d. r -= alpha * Ap
//!    e. rsnew = r^T * r
//!    f. If sqrt(rsnew) < tol * ||b||: converged
//!    g. p = r + (rsnew / rsold) * p
//!    h. rsold = rsnew
//!
//! The solver operates on host-side vectors. For GPU-accelerated sparse
//! matrix-vector products, the `spmv` closure should internally manage
//! device memory transfers.

#![allow(dead_code)]

use oxicuda_blas::GpuFloat;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;

// ---------------------------------------------------------------------------
// GpuFloat <-> f64 conversion helpers
// ---------------------------------------------------------------------------

/// Converts a `GpuFloat` value to `f64` via bit reinterpretation.
fn to_f64<T: GpuFloat>(val: T) -> f64 {
    if T::SIZE == 4 {
        f32::from_bits(val.to_bits_u64() as u32) as f64
    } else {
        f64::from_bits(val.to_bits_u64())
    }
}

/// Converts an `f64` value to `T: GpuFloat` via bit reinterpretation.
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

/// Configuration for the Conjugate Gradient solver.
#[derive(Debug, Clone)]
pub struct CgConfig {
    /// Maximum number of iterations.
    pub max_iter: u32,
    /// Convergence tolerance (relative to ||b||).
    pub tol: f64,
}

impl Default for CgConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solves `A * x = b` using the Conjugate Gradient method.
///
/// The matrix A is not passed directly. Instead, the caller provides a closure
/// `spmv` that computes `y = A * x` given `x` and `y` buffers. This enables
/// use with any sparse format, preconditioner, or matrix-free operator.
///
/// On entry, `x` should contain an initial guess (e.g., zeros). On exit, `x`
/// contains the approximate solution.
///
/// # Arguments
///
/// * `_handle` — solver handle (reserved for future GPU-accelerated variants).
/// * `spmv` — closure computing `y = A * x`: `spmv(x, y)`.
/// * `b` — right-hand side vector (length n).
/// * `x` — initial guess / solution vector (length n), modified in-place.
/// * `n` — system dimension.
/// * `config` — solver configuration (tolerance, max iterations).
///
/// # Returns
///
/// The number of iterations performed.
///
/// # Errors
///
/// Returns [`SolverError::ConvergenceFailure`] if the solver does not converge
/// within `max_iter` iterations.
/// Returns [`SolverError::DimensionMismatch`] if vector lengths are invalid.
pub fn cg_solve<T, F>(
    _handle: &SolverHandle,
    spmv: F,
    b: &[T],
    x: &mut [T],
    n: u32,
    config: &CgConfig,
) -> SolverResult<u32>
where
    T: GpuFloat,
    F: Fn(&[T], &mut [T]) -> SolverResult<()>,
{
    let n_usize = n as usize;

    // Validate dimensions.
    if b.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "cg_solve: b length ({}) < n ({n})",
            b.len()
        )));
    }
    if x.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "cg_solve: x length ({}) < n ({n})",
            x.len()
        )));
    }
    if n == 0 {
        return Ok(0);
    }

    // Compute ||b|| for relative convergence check.
    let b_norm = vec_norm(b, n_usize);
    let abs_tol = if b_norm > 0.0 {
        config.tol * b_norm
    } else {
        // b = 0 => x = 0 is the exact solution.
        for xi in x.iter_mut().take(n_usize) {
            *xi = T::gpu_zero();
        }
        return Ok(0);
    };

    // r = b - A*x
    let mut r = vec![T::gpu_zero(); n_usize];
    let mut ap = vec![T::gpu_zero(); n_usize];
    spmv(x, &mut ap)?;
    for i in 0..n_usize {
        r[i] = sub_t(b[i], ap[i]);
    }

    // p = r.clone()
    let mut p = r.clone();

    // rsold = r^T * r
    let mut rsold = dot_product(&r, &r, n_usize);

    if rsold.sqrt() < abs_tol {
        return Ok(0);
    }

    for iter in 0..config.max_iter {
        // Ap = A * p
        spmv(&p, &mut ap)?;

        // alpha = rsold / (p^T * Ap)
        let pap = dot_product(&p, &ap, n_usize);
        if pap.abs() < 1e-300 {
            return Err(SolverError::InternalError(
                "cg_solve: p^T * A * p is near zero (A may not be SPD)".into(),
            ));
        }
        let alpha = rsold / pap;
        let alpha_t = from_f64(alpha);

        // x += alpha * p
        for i in 0..n_usize {
            x[i] = add_t(x[i], mul_t(alpha_t, p[i]));
        }

        // r -= alpha * Ap
        for i in 0..n_usize {
            r[i] = sub_t(r[i], mul_t(alpha_t, ap[i]));
        }

        // rsnew = r^T * r
        let rsnew = dot_product(&r, &r, n_usize);

        // Check convergence.
        if rsnew.sqrt() < abs_tol {
            return Ok(iter + 1);
        }

        // beta = rsnew / rsold
        let beta = rsnew / rsold;
        let beta_t = from_f64(beta);

        // p = r + beta * p
        for i in 0..n_usize {
            p[i] = add_t(r[i], mul_t(beta_t, p[i]));
        }

        rsold = rsnew;
    }

    Err(SolverError::ConvergenceFailure {
        iterations: config.max_iter,
        residual: rsold.sqrt(),
    })
}

// ---------------------------------------------------------------------------
// Vector arithmetic helpers (host-side, generic over GpuFloat)
// ---------------------------------------------------------------------------

/// Computes the dot product of two vectors as f64.
fn dot_product<T: GpuFloat>(a: &[T], b: &[T], n: usize) -> f64 {
    let mut sum = 0.0_f64;
    for i in 0..n {
        sum += to_f64(a[i]) * to_f64(b[i]);
    }
    sum
}

/// Computes the 2-norm of a vector as f64.
fn vec_norm<T: GpuFloat>(v: &[T], n: usize) -> f64 {
    dot_product(v, v, n).sqrt()
}

/// Adds two GpuFloat values.
fn add_t<T: GpuFloat>(a: T, b: T) -> T {
    from_f64(to_f64(a) + to_f64(b))
}

/// Subtracts two GpuFloat values.
fn sub_t<T: GpuFloat>(a: T, b: T) -> T {
    from_f64(to_f64(a) - to_f64(b))
}

/// Multiplies two GpuFloat values.
fn mul_t<T: GpuFloat>(a: T, b: T) -> T {
    from_f64(to_f64(a) * to_f64(b))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_config_default() {
        let cfg = CgConfig::default();
        assert_eq!(cfg.max_iter, 1000);
        assert!((cfg.tol - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn dot_product_basic() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        let result = dot_product(&a, &b, 3);
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn vec_norm_basic() {
        let v = [3.0_f64, 4.0];
        let result = vec_norm(&v, 2);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn add_sub_mul() {
        let a = 3.0_f64;
        let b = 4.0_f64;
        assert!((to_f64(add_t(a, b)) - 7.0).abs() < 1e-15);
        assert!((to_f64(sub_t(a, b)) - (-1.0)).abs() < 1e-15);
        assert!((to_f64(mul_t(a, b)) - 12.0).abs() < 1e-15);
    }

    #[test]
    fn cg_config_custom() {
        let cfg = CgConfig {
            max_iter: 500,
            tol: 1e-10,
        };
        assert_eq!(cfg.max_iter, 500);
        assert!((cfg.tol - 1e-10).abs() < 1e-20);
    }

    // -----------------------------------------------------------------------
    // Quality gate: CG convergence on a 2×2 SPD system (CPU simulation)
    // -----------------------------------------------------------------------

    /// CPU-only conjugate gradient implementation for testing purposes.
    ///
    /// Solves A * x = b without requiring a `SolverHandle` (GPU context).
    /// This isolates the algorithmic correctness from the GPU infrastructure.
    fn cpu_cg_f64(
        spmv: impl Fn(&[f64], &mut [f64]),
        b: &[f64],
        x: &mut [f64],
        n: usize,
        max_iter: usize,
        tol: f64,
    ) -> usize {
        let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let abs_tol = tol * b_norm;

        let mut ap = vec![0.0_f64; n];
        spmv(x, &mut ap);
        let mut r: Vec<f64> = (0..n).map(|i| b[i] - ap[i]).collect();
        let mut p = r.clone();
        let mut rsold: f64 = r.iter().map(|v| v * v).sum();

        for iter in 0..max_iter {
            spmv(&p, &mut ap);
            let pap: f64 = p.iter().zip(&ap).map(|(pi, api)| pi * api).sum();
            if pap.abs() < 1e-300 {
                return iter;
            }
            let alpha = rsold / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rsnew: f64 = r.iter().map(|v| v * v).sum();
            if rsnew.sqrt() < abs_tol {
                return iter + 1;
            }
            let beta = rsnew / rsold;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rsold = rsnew;
        }
        max_iter
    }

    /// Quality gate: CG convergence on A = [[4, 1], [1, 3]], b = [1, 2].
    ///
    /// Exact solution: x = A^{-1} b
    ///   det(A) = 4*3 - 1*1 = 11
    ///   A^{-1} = (1/11) * [[3, -1], [-1, 4]]
    ///   x = (1/11) * [3*1 + (-1)*2, (-1)*1 + 4*2] = [1/11, 7/11]
    ///
    /// CG must converge in ≤ 5 iterations (at most n=2 for exact arithmetic).
    #[test]
    fn test_cg_convergence_spd_2x2() {
        // A = [[4, 1], [1, 3]] — symmetric positive definite (eigenvalues 3.27, 3.73)
        let a = [[4.0_f64, 1.0], [1.0, 3.0]];
        let spmv = |x: &[f64], y: &mut [f64]| {
            y[0] = a[0][0] * x[0] + a[0][1] * x[1];
            y[1] = a[1][0] * x[0] + a[1][1] * x[1];
        };

        let b = [1.0_f64, 2.0];
        let mut x = [0.0_f64, 0.0]; // zero initial guess

        let iters = cpu_cg_f64(spmv, &b, &mut x, 2, 100, 1e-12);

        // CG on an n×n SPD system converges in at most n steps in exact arithmetic.
        assert!(
            iters <= 5,
            "CG on 2×2 SPD system must converge in ≤ 5 iterations, took {iters}"
        );

        // Verify solution matches x = [1/11, 7/11]
        let x_exact = [1.0_f64 / 11.0, 7.0 / 11.0];
        assert!(
            (x[0] - x_exact[0]).abs() < 1e-10,
            "CG 2×2: x[0]={} expected {}",
            x[0],
            x_exact[0],
        );
        assert!(
            (x[1] - x_exact[1]).abs() < 1e-10,
            "CG 2×2: x[1]={} expected {}",
            x[1],
            x_exact[1],
        );
    }

    /// Quality gate: CG convergence on a 5×5 diagonal SPD system.
    ///
    /// For D = diag(1, 2, 3, 4, 5) and b = [1, 2, 3, 4, 5],
    /// the exact solution is x = [1, 1, 1, 1, 1].
    /// CG must converge in ≤ 10 iterations.
    #[test]
    fn test_cg_convergence_diagonal_5x5() {
        let diag = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let spmv = |x: &[f64], y: &mut [f64]| {
            for i in 0..5 {
                y[i] = diag[i] * x[i];
            }
        };
        let b = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut x = [0.0_f64; 5];

        let iters = cpu_cg_f64(spmv, &b, &mut x, 5, 100, 1e-12);

        assert!(
            iters <= 10,
            "CG on 5×5 diagonal SPD must converge in ≤ 10 iterations, took {iters}"
        );

        for (i, &xi) in x.iter().enumerate() {
            assert!(
                (xi - 1.0).abs() < 1e-10,
                "CG diagonal 5×5: x[{i}]={xi} expected 1.0",
            );
        }
    }
}
