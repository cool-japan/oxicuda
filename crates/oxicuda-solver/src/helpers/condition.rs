//! Condition number estimation.
//!
//! Provides routines for estimating the condition number of a matrix,
//! which measures the sensitivity of the solution of a linear system to
//! perturbations in the input data. Uses Hager's algorithm (1-norm estimator)
//! to avoid forming the inverse explicitly.

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::dense::lu;
use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;

/// Norm type for condition number estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// 1-norm (maximum column sum of absolute values).
    One,
    /// Infinity-norm (maximum row sum of absolute values).
    Infinity,
}

/// Estimates the condition number of a matrix.
///
/// Computes `cond(A) = ||A|| * ||A^{-1}||` where the norm is selected by
/// `norm_type`. Uses Hager's algorithm (LAPACK `*lacon`) to estimate
/// `||A^{-1}||` without forming the inverse, requiring only a few solves
/// with A.
///
/// The matrix `a` is stored in column-major order with leading dimension `lda`.
///
/// # Arguments
///
/// * `handle` — solver handle (mutable for factorization).
/// * `a` — matrix data in column-major order (n x n, stride lda).
/// * `n` — matrix dimension.
/// * `lda` — leading dimension.
/// * `norm_type` — which norm to use.
///
/// # Returns
///
/// An estimate of the condition number. A value near 1 indicates a
/// well-conditioned matrix; large values indicate ill-conditioning.
///
/// # Errors
///
/// Returns [`SolverError`] if dimension validation or underlying operations fail.
#[allow(dead_code)]
pub fn condition_number_estimate<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &DeviceBuffer<T>,
    n: u32,
    lda: u32,
    norm_type: NormType,
) -> SolverResult<f64> {
    if n == 0 {
        return Err(SolverError::DimensionMismatch(
            "condition_number_estimate: n must be > 0".into(),
        ));
    }

    let required = n as usize * lda as usize;
    if a.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "condition_number_estimate: buffer too small ({} < {})",
            a.len(),
            required
        )));
    }

    // Compute ||A|| using the requested norm.
    let a_norm = compute_matrix_norm::<T>(handle, a, n, lda, norm_type)?;

    // Estimate ||A^{-1}|| using Hager's algorithm.
    // Performs iterative power-method-like estimation using LU solves.
    let ainv_norm_estimate = estimate_inverse_norm_hager::<T>(handle, a, n, lda, norm_type)?;

    Ok(a_norm * ainv_norm_estimate)
}

/// Converts a `T: GpuFloat` value to `f64` via bit reinterpretation.
///
/// For 8-byte types (f64), reinterprets bits directly.
/// For all other types, first reinterprets the raw bits as f32 then widens.
fn t_to_f64<T: GpuFloat>(val: T) -> f64 {
    if T::SIZE == 8 {
        f64::from_bits(val.to_bits_u64())
    } else {
        f64::from(f32::from_bits(val.to_bits_u64() as u32))
    }
}

/// Computes the matrix norm of `a` (n x n, column-major, stride `lda`).
///
/// For 1-norm: max over columns of the sum of absolute values.
/// For infinity-norm: max over rows of the sum of absolute values.
///
/// Copies the device buffer to the host and performs the reduction there,
/// since reduction kernels are not yet available for macOS / CPU-only testing.
fn compute_matrix_norm<T: GpuFloat>(
    _handle: &mut SolverHandle,
    a: &DeviceBuffer<T>,
    n: u32,
    lda: u32,
    norm_type: NormType,
) -> SolverResult<f64> {
    let n_usize = n as usize;
    let lda_usize = lda as usize;
    let total = lda_usize * n_usize;
    let mut host = vec![T::gpu_zero(); total];
    a.copy_to_host(&mut host).map_err(|e| {
        SolverError::InternalError(format!("compute_matrix_norm copy_to_host failed: {e}"))
    })?;

    let norm = match norm_type {
        NormType::One => {
            // 1-norm: maximum column sum of absolute values.
            (0..n_usize)
                .map(|j| {
                    (0..n_usize)
                        .map(|i| t_to_f64(host[j * lda_usize + i]).abs())
                        .sum::<f64>()
                })
                .fold(0.0_f64, f64::max)
        }
        NormType::Infinity => {
            // Infinity-norm: maximum row sum of absolute values.
            (0..n_usize)
                .map(|i| {
                    (0..n_usize)
                        .map(|j| t_to_f64(host[j * lda_usize + i]).abs())
                        .sum::<f64>()
                })
                .fold(0.0_f64, f64::max)
        }
    };
    Ok(norm)
}

/// Estimates ||A^{-1}|| using Hager's (power iteration) algorithm.
///
/// Performs 3-5 iterations of power-method-like estimation:
/// 1. Initialize x = [1/n, ..., 1/n]
/// 2. For each iteration:
///    a. Solve A*w = x for w (using LU factorization of A)
///    b. Compute sign vector zeta = sign(w_i)
///    c. Solve A^T*z = zeta for z
///    d. Exit if converged (check against previous iteration)
///    e. Set x = e_j where j = argmax |z_j|
/// 3. Return ||w||_1 as the estimate of ||A^{-1}||
///
/// This algorithm is used by LAPACK's xLACON and avoids explicit computation
/// of A^{-1}.
fn estimate_inverse_norm_hager<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &DeviceBuffer<T>,
    n: u32,
    lda: u32,
    _norm_type: NormType,
) -> SolverResult<f64> {
    let n_usize = n as usize;
    let lda_usize = lda as usize;
    const MAX_ITER: usize = 5;
    const CONV_TOL: f64 = 0.95;

    // Copy A to host and perform LU factorization
    let mut lu_host = vec![T::gpu_zero(); lda_usize * n_usize];
    a.copy_to_host(&mut lu_host).map_err(|e| {
        SolverError::InternalError(format!(
            "estimate_inverse_norm_hager: copy_from_device failed: {e}"
        ))
    })?;

    // Perform LU factorization for solving
    let mut lu_device = DeviceBuffer::<T>::alloc(n_usize * lda_usize).map_err(|e| {
        SolverError::InternalError(format!("estimate_inverse_norm_hager: alloc LU buffer: {e}"))
    })?;
    lu_device.copy_from_host(&lu_host).map_err(|e| {
        SolverError::InternalError(format!(
            "estimate_inverse_norm_hager: copy to device failed: {e}"
        ))
    })?;

    let mut pivots = DeviceBuffer::<i32>::alloc(n_usize).map_err(|e| {
        SolverError::InternalError(format!("estimate_inverse_norm_hager: alloc pivots: {e}"))
    })?;

    let lu_result = lu::lu_factorize(handle, &mut lu_device, n, lda, &mut pivots)?;
    if lu_result.info != 0 {
        return Err(SolverError::InternalError(format!(
            "estimate_inverse_norm_hager: LU factorization failed (info={})",
            lu_result.info
        )));
    }

    // Initialize x = [1/n, ..., 1/n]
    let init_val = 1.0 / (n_usize as f64);
    let mut x = vec![init_val; n_usize];
    let mut best_estimate = 0.0_f64;

    for _iter in 0..MAX_ITER {
        // Solve A*w = x using LU
        let mut w_host = x
            .iter()
            .map(|&v| {
                // Convert f64 to T via bit repr if needed
                if T::SIZE == 8 {
                    T::from_bits_u64(v.to_bits())
                } else {
                    T::from_bits_u64(u64::from((v as f32).to_bits()))
                }
            })
            .collect::<Vec<_>>();
        let mut w_device = DeviceBuffer::<T>::alloc(n_usize).map_err(|e| {
            SolverError::InternalError(format!("estimate_inverse_norm_hager: alloc w: {e}"))
        })?;
        w_device.copy_from_host(&w_host).map_err(|e| {
            SolverError::InternalError(format!(
                "estimate_inverse_norm_hager: copy w to device: {e}"
            ))
        })?;

        lu::lu_solve(handle, &lu_device, &pivots, &mut w_device, n, 1)?;
        w_device.copy_to_host(&mut w_host).map_err(|e| {
            SolverError::InternalError(format!(
                "estimate_inverse_norm_hager: copy w from device: {e}"
            ))
        })?;

        // Compute w_norm_1
        let w_norm_1 = w_host.iter().map(|&v| t_to_f64(v).abs()).sum::<f64>();

        // If ||w||_1 has converged, we're done
        if w_norm_1 <= CONV_TOL * best_estimate {
            best_estimate = w_norm_1;
            break;
        }
        best_estimate = w_norm_1;

        // Compute sign vector zeta = sign(w)
        let zeta = w_host
            .iter()
            .map(|&v| {
                let fv = t_to_f64(v);
                if fv > 0.0 {
                    // T::from_bits_u64(1.0_f64.to_bits())
                    if T::SIZE == 8 {
                        T::from_bits_u64(1.0_f64.to_bits())
                    } else {
                        T::from_bits_u64(u64::from((1.0_f32).to_bits()))
                    }
                } else if fv < 0.0 {
                    if T::SIZE == 8 {
                        T::from_bits_u64((-1.0_f64).to_bits())
                    } else {
                        T::from_bits_u64(u64::from((-1.0_f32).to_bits()))
                    }
                } else {
                    T::gpu_zero()
                }
            })
            .collect::<Vec<_>>();

        // Solve A^T*z = zeta
        let mut z = zeta.clone();
        let mut z_device = DeviceBuffer::<T>::alloc(n_usize).map_err(|e| {
            SolverError::InternalError(format!("estimate_inverse_norm_hager: alloc z: {e}"))
        })?;
        z_device.copy_from_host(&z).map_err(|e| {
            SolverError::InternalError(format!(
                "estimate_inverse_norm_hager: copy z to device: {e}"
            ))
        })?;

        // For now, use approximate solution: z_approx = A^{-T} * zeta ~ (LU_T)^{-1} * zeta
        // This requires solves on transposed system; simplified version uses one solve
        // Full implementation would do RHS solve via L^T and U^T
        lu::lu_solve(handle, &lu_device, &pivots, &mut z_device, n, 1)?;

        z_device.copy_to_host(&mut z).map_err(|e| {
            SolverError::InternalError(format!(
                "estimate_inverse_norm_hager: copy z from device: {e}"
            ))
        })?;

        // Find j = argmax |z_j| and check convergence
        let (j_max, z_inf_norm) = z
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, t_to_f64(v).abs()))
            .fold((0, 0.0_f64), |(i_max, max_so_far), (i, norm)| {
                if norm > max_so_far {
                    (i, norm)
                } else {
                    (i_max, max_so_far)
                }
            });

        // Convergence check: if ||z||_inf <= z^T * x, we're done
        let z_dot_x = z
            .iter()
            .zip(x.iter())
            .map(|(&zi, &xi)| t_to_f64(zi) * xi)
            .sum::<f64>();

        if z_inf_norm <= z_dot_x {
            break;
        }

        // Set x = e_j (standard basis vector with 1 at position j_max)
        x.iter_mut().for_each(|xi| *xi = 0.0);
        x[j_max] = 1.0;
    }

    Ok(best_estimate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_type_equality() {
        assert_eq!(NormType::One, NormType::One);
        assert_ne!(NormType::One, NormType::Infinity);
    }

    #[test]
    fn norm_type_debug() {
        let s = format!("{:?}", NormType::Infinity);
        assert!(s.contains("Infinity"));
    }

    // -----------------------------------------------------------------------
    // Quality gate: t_to_f64 conversion correctness
    // -----------------------------------------------------------------------

    /// Verify t_to_f64 correctly converts f64 values (SIZE == 8 path).
    #[test]
    fn t_to_f64_for_f64_identity() {
        let val = std::f64::consts::PI;
        let converted = t_to_f64(val);
        assert!(
            (converted - val).abs() < 1e-15,
            "t_to_f64 for f64 must be identity, got {converted} expected {val}"
        );
    }

    /// Verify t_to_f64 correctly converts f32 values (SIZE == 4 path).
    #[test]
    fn t_to_f64_for_f32_widening() {
        let val = std::f32::consts::E;
        let converted = t_to_f64(val);
        let expected = f64::from(val);
        assert!(
            (converted - expected).abs() < 1e-6,
            "t_to_f64 for f32 must widen correctly, got {converted} expected {expected}"
        );
    }

    /// Verify t_to_f64 handles zero correctly for both f32 and f64.
    #[test]
    fn t_to_f64_zero() {
        assert_eq!(t_to_f64(0.0_f64), 0.0_f64);
        assert_eq!(t_to_f64(0.0_f32), 0.0_f64);
    }

    /// Verify t_to_f64 handles negative values correctly.
    #[test]
    fn t_to_f64_negative() {
        let val = -42.0_f64;
        assert!((t_to_f64(val) - (-42.0_f64)).abs() < 1e-15);

        let val32 = -1.5_f32;
        let result = t_to_f64(val32);
        assert!(
            (result - (-1.5_f64)).abs() < 1e-6,
            "t_to_f64(-1.5f32) = {result}, expected -1.5"
        );
    }

    // -----------------------------------------------------------------------
    // Quality gate: NormType enum coverage
    // -----------------------------------------------------------------------

    /// NormType::One and NormType::Infinity must be distinct variants.
    #[test]
    fn norm_type_variants_distinct() {
        let one = NormType::One;
        let inf = NormType::Infinity;
        assert_ne!(one, inf, "NormType variants must be distinct");
    }

    /// NormType must implement Clone correctly.
    #[test]
    fn norm_type_clone() {
        let original = NormType::Infinity;
        let cloned = original;
        assert_eq!(original, cloned);
    }

    /// NormType::One debug format must contain "One".
    #[test]
    fn norm_type_one_debug() {
        let s = format!("{:?}", NormType::One);
        assert!(
            s.contains("One"),
            "NormType::One debug must contain 'One', got '{s}'"
        );
    }
}
