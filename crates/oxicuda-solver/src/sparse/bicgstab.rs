//! BiCGSTAB (Biconjugate Gradient Stabilized) iterative solver.
//!
//! Solves the linear system `A * x = b` for general (non-symmetric) matrices.
//! The solver is matrix-free: it only requires a closure that computes the
//! matrix-vector product `y = A * x`.
//!
//! # Algorithm
//!
//! The BiCGSTAB algorithm (van der Vorst, 1992):
//! 1. r = b - A*x; r0_hat = r; rho = alpha = omega = 1; v = p = 0
//! 2. For each iteration:
//!    a. rho_new = r0_hat^T * r
//!    b. beta = (rho_new / rho) * (alpha / omega)
//!    c. p = r + beta * (p - omega * v)
//!    d. v = A * p
//!    e. alpha = rho_new / (r0_hat^T * v)
//!    f. s = r - alpha * v
//!    g. t = A * s
//!    h. omega = (t^T * s) / (t^T * t)
//!    i. x += alpha * p + omega * s
//!    j. r = s - omega * t
//!    k. Check convergence: ||r|| < tol * ||b||

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
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the BiCGSTAB solver.
#[derive(Debug, Clone)]
pub struct BiCgStabConfig {
    /// Maximum number of iterations.
    pub max_iter: u32,
    /// Convergence tolerance (relative to ||b||).
    pub tol: f64,
}

impl Default for BiCgStabConfig {
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

/// Solves `A * x = b` using the BiCGSTAB method.
///
/// On entry, `x` should contain an initial guess. On exit, `x` contains the
/// approximate solution.
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
/// The number of iterations performed.
///
/// # Errors
///
/// Returns [`SolverError::ConvergenceFailure`] if the solver does not converge.
/// Returns [`SolverError::InternalError`] if a breakdown is detected (e.g., rho = 0).
pub fn bicgstab_solve<T, F>(
    _handle: &SolverHandle,
    spmv: F,
    b: &[T],
    x: &mut [T],
    n: u32,
    config: &BiCgStabConfig,
) -> SolverResult<u32>
where
    T: GpuFloat,
    F: Fn(&[T], &mut [T]) -> SolverResult<()>,
{
    let n_usize = n as usize;

    // Validate dimensions.
    if b.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "bicgstab_solve: b length ({}) < n ({n})",
            b.len()
        )));
    }
    if x.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "bicgstab_solve: x length ({}) < n ({n})",
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
        for xi in x.iter_mut().take(n_usize) {
            *xi = T::gpu_zero();
        }
        return Ok(0);
    };

    // r = b - A*x
    let mut r = vec![T::gpu_zero(); n_usize];
    let mut tmp = vec![T::gpu_zero(); n_usize];
    spmv(x, &mut tmp)?;
    for i in 0..n_usize {
        r[i] = sub_t(b[i], tmp[i]);
    }

    // r0_hat = r (shadow residual, kept constant)
    let r0_hat = r.clone();

    // Initialize scalars.
    let mut rho = 1.0_f64;
    let mut alpha = 1.0_f64;
    let mut omega = 1.0_f64;

    // Initialize vectors.
    let mut v = vec![T::gpu_zero(); n_usize];
    let mut p = vec![T::gpu_zero(); n_usize];
    let mut s = vec![T::gpu_zero(); n_usize];
    let mut t = vec![T::gpu_zero(); n_usize];

    for iter in 0..config.max_iter {
        // rho_new = r0_hat^T * r
        let rho_new = dot_product(&r0_hat, &r, n_usize);

        if rho_new.abs() < 1e-300 {
            return Err(SolverError::InternalError(
                "bicgstab_solve: rho breakdown (r0_hat^T * r ~ 0)".into(),
            ));
        }

        // beta = (rho_new / rho) * (alpha / omega)
        let beta = if rho.abs() > 1e-300 && omega.abs() > 1e-300 {
            (rho_new / rho) * (alpha / omega)
        } else {
            0.0
        };
        let beta_t = from_f64(beta);
        let omega_t = from_f64(omega);

        // p = r + beta * (p - omega * v)
        for i in 0..n_usize {
            let pv = sub_t(p[i], mul_t(omega_t, v[i]));
            p[i] = add_t(r[i], mul_t(beta_t, pv));
        }

        // v = A * p
        spmv(&p, &mut v)?;

        // alpha = rho_new / (r0_hat^T * v)
        let r0v = dot_product(&r0_hat, &v, n_usize);
        if r0v.abs() < 1e-300 {
            return Err(SolverError::InternalError(
                "bicgstab_solve: alpha breakdown (r0_hat^T * v ~ 0)".into(),
            ));
        }
        alpha = rho_new / r0v;
        let alpha_t = from_f64(alpha);

        // s = r - alpha * v
        for i in 0..n_usize {
            s[i] = sub_t(r[i], mul_t(alpha_t, v[i]));
        }

        // Check if s is small enough (early exit).
        let s_norm = vec_norm(&s, n_usize);
        if s_norm < abs_tol {
            // x += alpha * p
            for i in 0..n_usize {
                x[i] = add_t(x[i], mul_t(alpha_t, p[i]));
            }
            return Ok(iter + 1);
        }

        // t = A * s
        spmv(&s, &mut t)?;

        // omega = (t^T * s) / (t^T * t)
        let tt = dot_product(&t, &t, n_usize);
        omega = if tt.abs() > 1e-300 {
            dot_product(&t, &s, n_usize) / tt
        } else {
            0.0
        };
        let omega_new_t = from_f64(omega);

        // x += alpha * p + omega * s
        for i in 0..n_usize {
            x[i] = add_t(x[i], add_t(mul_t(alpha_t, p[i]), mul_t(omega_new_t, s[i])));
        }

        // r = s - omega * t
        for i in 0..n_usize {
            r[i] = sub_t(s[i], mul_t(omega_new_t, t[i]));
        }

        // Check convergence.
        let r_norm = vec_norm(&r, n_usize);
        if r_norm < abs_tol {
            return Ok(iter + 1);
        }

        if omega.abs() < 1e-300 {
            return Err(SolverError::InternalError(
                "bicgstab_solve: omega breakdown".into(),
            ));
        }

        rho = rho_new;
    }

    Err(SolverError::ConvergenceFailure {
        iterations: config.max_iter,
        residual: vec_norm(&r, n_usize),
    })
}

// ---------------------------------------------------------------------------
// Vector arithmetic helpers
// ---------------------------------------------------------------------------

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

    /// CPU-only BiCGSTAB solver for testing without a GPU handle.
    ///
    /// Mirrors `bicgstab_solve` but omits `_handle`, enabling pure host testing
    /// with a closure-based matrix-vector product.
    fn bicgstab_solve_cpu<T, F>(
        spmv: F,
        b: &[T],
        x: &mut [T],
        n: u32,
        config: &BiCgStabConfig,
    ) -> SolverResult<u32>
    where
        T: GpuFloat,
        F: Fn(&[T], &mut [T]) -> SolverResult<()>,
    {
        let n_usize = n as usize;

        if b.len() < n_usize {
            return Err(SolverError::DimensionMismatch(format!(
                "bicgstab_solve_cpu: b length ({}) < n ({n})",
                b.len()
            )));
        }
        if x.len() < n_usize {
            return Err(SolverError::DimensionMismatch(format!(
                "bicgstab_solve_cpu: x length ({}) < n ({n})",
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

        let mut r = vec![T::gpu_zero(); n_usize];
        let mut tmp = vec![T::gpu_zero(); n_usize];
        spmv(x, &mut tmp)?;
        for i in 0..n_usize {
            r[i] = sub_t(b[i], tmp[i]);
        }

        let r0_hat = r.clone();
        let mut rho = 1.0_f64;
        let mut alpha = 1.0_f64;
        let mut omega = 1.0_f64;
        let mut v = vec![T::gpu_zero(); n_usize];
        let mut p = vec![T::gpu_zero(); n_usize];
        let mut s = vec![T::gpu_zero(); n_usize];
        let mut t = vec![T::gpu_zero(); n_usize];

        for iter in 0..config.max_iter {
            let rho_new = dot_product(&r0_hat, &r, n_usize);
            if rho_new.abs() < 1e-300 {
                return Err(SolverError::InternalError(
                    "bicgstab_solve_cpu: rho breakdown".into(),
                ));
            }

            let beta = if rho.abs() > 1e-300 && omega.abs() > 1e-300 {
                (rho_new / rho) * (alpha / omega)
            } else {
                0.0
            };
            let beta_t = from_f64(beta);
            let omega_t = from_f64(omega);

            for i in 0..n_usize {
                let pv = sub_t(p[i], mul_t(omega_t, v[i]));
                p[i] = add_t(r[i], mul_t(beta_t, pv));
            }

            spmv(&p, &mut v)?;

            let r0v = dot_product(&r0_hat, &v, n_usize);
            if r0v.abs() < 1e-300 {
                return Err(SolverError::InternalError(
                    "bicgstab_solve_cpu: alpha breakdown".into(),
                ));
            }
            alpha = rho_new / r0v;
            let alpha_t = from_f64(alpha);

            for i in 0..n_usize {
                s[i] = sub_t(r[i], mul_t(alpha_t, v[i]));
            }

            let s_norm = vec_norm(&s, n_usize);
            if s_norm < abs_tol {
                for i in 0..n_usize {
                    x[i] = add_t(x[i], mul_t(alpha_t, p[i]));
                }
                return Ok(iter + 1);
            }

            spmv(&s, &mut t)?;

            let tt = dot_product(&t, &t, n_usize);
            omega = if tt.abs() > 1e-300 {
                dot_product(&t, &s, n_usize) / tt
            } else {
                0.0
            };
            let omega_new_t = from_f64(omega);

            for i in 0..n_usize {
                x[i] = add_t(x[i], add_t(mul_t(alpha_t, p[i]), mul_t(omega_new_t, s[i])));
            }

            for i in 0..n_usize {
                r[i] = sub_t(s[i], mul_t(omega_new_t, t[i]));
            }

            let r_norm = vec_norm(&r, n_usize);
            if r_norm < abs_tol {
                return Ok(iter + 1);
            }

            if omega.abs() < 1e-300 {
                return Err(SolverError::InternalError(
                    "bicgstab_solve_cpu: omega breakdown".into(),
                ));
            }

            rho = rho_new;
        }

        Err(SolverError::ConvergenceFailure {
            iterations: config.max_iter,
            residual: vec_norm(&r, n_usize),
        })
    }

    #[test]
    fn bicgstab_config_default() {
        let cfg = BiCgStabConfig::default();
        assert_eq!(cfg.max_iter, 1000);
        assert!((cfg.tol - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn bicgstab_config_custom() {
        let cfg = BiCgStabConfig {
            max_iter: 2000,
            tol: 1e-8,
        };
        assert_eq!(cfg.max_iter, 2000);
        assert!((cfg.tol - 1e-8).abs() < 1e-20);
    }

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

    /// BiCGSTAB converges on a 3×3 symmetric positive definite system.
    ///
    /// A = [[4,-1,0],[-1,4,-1],[0,-1,4]], b = [6, 0, 6].
    /// Exact solution (via numpy): x = [12/7, 6/7, 12/7] ≈ [1.7143, 0.8571, 1.7143].
    #[test]
    fn bicgstab_converges_spd_3x3() {
        let b = vec![6.0_f64, 0.0, 6.0];
        let mut x = vec![0.0_f64; 3];
        let config = BiCgStabConfig {
            max_iter: 200,
            tol: 1e-10,
        };

        // A = [[4,-1,0],[-1,4,-1],[0,-1,4]]
        let spmv = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
            out[0] = 4.0 * v[0] - v[1];
            out[1] = -v[0] + 4.0 * v[1] - v[2];
            out[2] = -v[1] + 4.0 * v[2];
            Ok(())
        };

        let _iters = bicgstab_solve_cpu(spmv, &b, &mut x, 3, &config)
            .expect("BiCGSTAB should converge on SPD system");

        let x0_exact = 12.0_f64 / 7.0; // ≈ 1.714286
        let x1_exact = 6.0_f64 / 7.0; // ≈ 0.857143
        assert!(
            (x[0] - x0_exact).abs() < 1e-7,
            "x[0] = {} expected {x0_exact}",
            x[0]
        );
        assert!(
            (x[1] - x1_exact).abs() < 1e-7,
            "x[1] = {} expected {x1_exact}",
            x[1]
        );
        assert!(
            (x[2] - x0_exact).abs() < 1e-7,
            "x[2] = {} expected {x0_exact}",
            x[2]
        );
    }

    /// BiCGSTAB converges on the identity system in a single iteration.
    #[test]
    fn bicgstab_converges_identity() {
        let b = vec![5.0_f64, -3.0, 2.0];
        let mut x = vec![0.0_f64; 3];
        let config = BiCgStabConfig {
            max_iter: 50,
            tol: 1e-12,
        };

        // A = I
        let spmv = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
            out.copy_from_slice(v);
            Ok(())
        };

        let _iters = bicgstab_solve_cpu(spmv, &b, &mut x, 3, &config)
            .expect("BiCGSTAB should converge on identity");

        assert!((x[0] - 5.0).abs() < 1e-9);
        assert!((x[1] - (-3.0)).abs() < 1e-9);
        assert!((x[2] - 2.0).abs() < 1e-9);
    }

    /// BiCGSTAB with zero RHS returns the zero vector immediately.
    #[test]
    fn bicgstab_zero_rhs_returns_zero() {
        let b = vec![0.0_f64; 3];
        let mut x = vec![1.0_f64; 3];
        let config = BiCgStabConfig::default();

        let spmv = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
            out.copy_from_slice(v);
            Ok(())
        };

        let iters =
            bicgstab_solve_cpu(spmv, &b, &mut x, 3, &config).expect("zero RHS should succeed");
        assert_eq!(iters, 0);
        for &xi in &x {
            assert!(xi.abs() < 1e-15);
        }
    }

    /// BiCGSTAB returns DimensionMismatch when b is shorter than n.
    #[test]
    fn bicgstab_dimension_mismatch() {
        let b = vec![1.0_f64]; // length 1, n = 3
        let mut x = vec![0.0_f64; 3];
        let config = BiCgStabConfig::default();
        let spmv = |_: &[f64], _: &mut [f64]| -> SolverResult<()> { Ok(()) };
        let result = bicgstab_solve_cpu(spmv, &b, &mut x, 3, &config);
        assert!(matches!(result, Err(SolverError::DimensionMismatch(_))));
    }

    /// BiCGSTAB converges on a diagonal system with varying eigenvalues.
    ///
    /// A = diag(1, 3, 7), b = [2, 9, 14] → exact x = [2, 3, 2].
    #[test]
    fn bicgstab_converges_diagonal() {
        let b = vec![2.0_f64, 9.0, 14.0];
        let mut x = vec![0.0_f64; 3];
        let config = BiCgStabConfig {
            max_iter: 200,
            tol: 1e-10,
        };

        let spmv = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
            out[0] = 1.0 * v[0];
            out[1] = 3.0 * v[1];
            out[2] = 7.0 * v[2];
            Ok(())
        };

        let _iters = bicgstab_solve_cpu(spmv, &b, &mut x, 3, &config)
            .expect("BiCGSTAB should converge on diagonal system");

        assert!((x[0] - 2.0).abs() < 1e-8, "x[0] = {} expected 2.0", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-8, "x[1] = {} expected 3.0", x[1]);
        assert!((x[2] - 2.0).abs() < 1e-8, "x[2] = {} expected 2.0", x[2]);
    }
}
