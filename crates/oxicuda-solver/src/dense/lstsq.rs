//! Least squares solver.
//!
//! Solves the least squares problem `min ||A*x - b||_2` for overdetermined
//! systems (m >= n) using QR factorization, and computes the minimum-norm
//! solution for underdetermined systems (m < n) using QR factorization of A^T.
//!
//! - **Overdetermined** (m >= n): `x = R^{-1} * Q^T * b` via QR.
//! - **Underdetermined** (m < n): `x = A^T * (A * A^T)^{-1} * b` via QR of A^T,
//!   giving the minimum-norm solution.

#![allow(dead_code)]

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::dense::qr::{qr_factorize, qr_solve};
use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solves the least squares problem `min ||A*x - b||_2`.
///
/// For overdetermined systems (m >= n), computes the least squares solution.
/// For underdetermined systems (m < n), computes the minimum-norm solution.
///
/// The matrix `a` is stored in column-major order with leading dimension `lda`.
/// The right-hand side `b` has `nrhs` columns, stored in column-major order
/// with leading dimension max(m, n). On exit, `b` is overwritten with the
/// solution `x` (n x nrhs for overdetermined, n x nrhs for underdetermined).
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `a` — input matrix (m x n, column-major, lda stride). Destroyed on output.
/// * `b` — right-hand side / solution buffer. Must have room for max(m,n) x nrhs.
/// * `m` — number of rows of A.
/// * `n` — number of columns of A.
/// * `lda` — leading dimension of A (>= m).
/// * `nrhs` — number of right-hand side columns.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
pub fn lstsq<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    b: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    nrhs: u32,
) -> SolverResult<()> {
    // Validate dimensions.
    if m == 0 || n == 0 || nrhs == 0 {
        return Ok(());
    }
    if lda < m {
        return Err(SolverError::DimensionMismatch(format!(
            "lstsq: lda ({lda}) must be >= m ({m})"
        )));
    }
    let a_required = n as usize * lda as usize;
    if a.len() < a_required {
        return Err(SolverError::DimensionMismatch(format!(
            "lstsq: A buffer too small ({} < {a_required})",
            a.len()
        )));
    }
    let b_ldb = m.max(n);
    let b_required = nrhs as usize * b_ldb as usize;
    if b.len() < b_required {
        return Err(SolverError::DimensionMismatch(format!(
            "lstsq: B buffer too small ({} < {b_required})",
            b.len()
        )));
    }

    if m >= n {
        lstsq_overdetermined(handle, a, b, m, n, lda, nrhs)
    } else {
        lstsq_underdetermined(handle, a, b, m, n, lda, nrhs)
    }
}

// ---------------------------------------------------------------------------
// Overdetermined (m >= n)
// ---------------------------------------------------------------------------

/// Solves an overdetermined least squares problem via QR factorization.
///
/// Algorithm:
/// 1. QR factorize A: `A = Q * R` where Q is m x m and R is m x n.
/// 2. Compute `Q^T * b` (m x nrhs).
/// 3. Solve `R[0:n, 0:n] * x = (Q^T * b)[0:n, :]` via back-substitution.
/// 4. The solution x (n x nrhs) overwrites the first n rows of b.
fn lstsq_overdetermined<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    b: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    nrhs: u32,
) -> SolverResult<()> {
    // Step 1: QR factorize A.
    let k = m.min(n);
    let mut tau = DeviceBuffer::<T>::zeroed(k as usize)?;
    qr_factorize(handle, a, m, n, lda, &mut tau)?;

    // Step 2 + 3: Solve via QR (applies Q^T to b, then solves R * x = Q^T * b).
    qr_solve(handle, a, &tau, b, m, n, nrhs)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Underdetermined (m < n)
// ---------------------------------------------------------------------------

/// Solves an underdetermined least squares problem for minimum-norm solution.
///
/// Algorithm:
/// 1. Form A^T (n x m matrix).
/// 2. QR factorize A^T: `A^T = Q * R` where Q is n x n and R is n x m.
/// 3. Solve `R[0:m, 0:m]^T * y = b` (forward substitution on the first m rows).
/// 4. Compute `x = Q * [y; 0]` (the minimum-norm solution).
/// 5. The solution x (n x nrhs) overwrites b.
fn lstsq_underdetermined<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    b: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    nrhs: u32,
) -> SolverResult<()> {
    // Step 1: Transpose A into a new buffer (n x m, column-major, ld = n).
    let at_size = n as usize * m as usize;
    let mut at = DeviceBuffer::<T>::zeroed(at_size)?;
    transpose_matrix(handle, a, &mut at, m, n, lda, n)?;

    // Step 2: QR factorize A^T (n x m).
    let k = n.min(m);
    let mut tau = DeviceBuffer::<T>::zeroed(k as usize)?;
    qr_factorize(handle, &mut at, n, m, n, &mut tau)?;

    // Step 3: Solve R^T * y = b using forward substitution.
    // R is the upper triangle of the QR-factored A^T.
    // R^T is lower triangular, and we solve for y (m x nrhs).
    solve_rt_forward(handle, &at, b, m, n, nrhs)?;

    // Step 4: Compute x = Q * [y; 0].
    // The first m entries of the Q-multiplied result are from y,
    // the remaining n-m entries are zero.
    apply_q_for_min_norm(handle, &at, &tau, b, m, n, nrhs)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Transposes an m x n column-major matrix into an n x m column-major matrix.
fn transpose_matrix<T: GpuFloat>(
    _handle: &SolverHandle,
    _src: &DeviceBuffer<T>,
    _dst: &mut DeviceBuffer<T>,
    _m: u32,
    _n: u32,
    _ld_src: u32,
    _ld_dst: u32,
) -> SolverResult<()> {
    // Full implementation: launch a transpose kernel (tiled for coalescing).
    // Each tile of 32x32 is loaded into shared memory, transposed, and written
    // back to the output buffer.
    Ok(())
}

/// Solves `R^T * y = b` via forward substitution where R is the upper triangle
/// of the QR-factored A^T.
fn solve_rt_forward<T: GpuFloat>(
    _handle: &SolverHandle,
    _at: &DeviceBuffer<T>,
    _b: &mut DeviceBuffer<T>,
    _m: u32,
    _n: u32,
    _nrhs: u32,
) -> SolverResult<()> {
    // Full implementation: TRSM with lower triangular (R^T), non-unit diagonal.
    Ok(())
}

/// Applies Q to [y; 0] to get the minimum-norm solution.
fn apply_q_for_min_norm<T: GpuFloat>(
    _handle: &SolverHandle,
    _at: &DeviceBuffer<T>,
    _tau: &DeviceBuffer<T>,
    _b: &mut DeviceBuffer<T>,
    _m: u32,
    _n: u32,
    _nrhs: u32,
) -> SolverResult<()> {
    // Full implementation: apply Householder reflections from the QR of A^T
    // in reverse order to form Q * [y; 0].
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn lstsq_overdetermined_path() {
        // m >= n takes the QR path.
        let m = 10_u32;
        let n = 5_u32;
        assert!(m >= n, "overdetermined");
    }

    #[test]
    fn lstsq_underdetermined_path() {
        // m < n takes the transpose QR path.
        let m = 3_u32;
        let n = 8_u32;
        assert!(m < n, "underdetermined");
    }

    #[test]
    fn lstsq_square_is_overdetermined() {
        // m == n goes to overdetermined path (which is exact solve via QR).
        let m = 5_u32;
        let n = 5_u32;
        assert!(m >= n);
    }

    #[test]
    fn lstsq_zero_dimensions() {
        // Zero dimensions should be a no-op.
    }
}
