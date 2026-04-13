//! Tridiagonal system solver.
//!
//! Solves systems of the form:
//!
//! ```text
//! a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i
//! ```
//!
//! where `a` is the sub-diagonal (length n-1), `b` is the main diagonal
//! (length n), `c` is the super-diagonal (length n-1), and `d` is the
//! right-hand side (length n).
//!
//! Two algorithms are provided:
//!
//! - **Thomas algorithm** (sequential): O(n) time, O(1) extra space. Used
//!   internally for single-system solves when the system is small.
//! - **Cyclic reduction** (parallel-friendly): O(n) work in O(log n) parallel
//!   steps. Forms the basis of the GPU kernel for large systems.
//!
//! For batched solves, each independent system can be solved concurrently.

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
// Public API
// ---------------------------------------------------------------------------

/// Solves a tridiagonal system using the Thomas algorithm (sequential).
///
/// The system is: `lower[i] * x[i-1] + diag[i] * x[i] + upper[i] * x[i+1] = rhs[i]`
/// where `lower[0]` corresponds to `a_1` (the sub-diagonal starting at row 1).
///
/// On exit, `rhs` is overwritten with the solution vector `x`.
///
/// # Arguments
///
/// * `_handle` — solver handle (reserved for future GPU-accelerated variants).
/// * `lower` — sub-diagonal array of length `n - 1`.
/// * `diag` — main diagonal array of length `n`.
/// * `upper` — super-diagonal array of length `n - 1`.
/// * `rhs` — right-hand side of length `n`, overwritten with the solution.
/// * `n` — system dimension.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] if array lengths are invalid.
/// Returns [`SolverError::SingularMatrix`] if a zero pivot is encountered.
pub fn tridiagonal_solve<T: GpuFloat>(
    _handle: &SolverHandle,
    lower: &[T],
    diag: &[T],
    upper: &[T],
    rhs: &mut [T],
    n: usize,
) -> SolverResult<()> {
    validate_tridiagonal_dims(lower, diag, upper, rhs, n)?;

    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        return solve_1x1(diag, rhs);
    }

    // Use cyclic reduction for larger systems (parallel-friendly algorithm)
    // which degenerates gracefully on CPU to O(n log n) but is correct.
    // For small n, Thomas is simpler; threshold at 64.
    if n <= 64 {
        thomas_solve(lower, diag, upper, rhs, n)
    } else {
        cyclic_reduction_solve(lower, diag, upper, rhs, n)
    }
}

/// Solves a batch of independent tridiagonal systems.
///
/// Each system has the same dimension `n`. The arrays are stored contiguously:
/// `lower[k * (n-1) .. (k+1) * (n-1)]` is the sub-diagonal for system `k`, etc.
///
/// # Arguments
///
/// * `_handle` — solver handle.
/// * `lower` — sub-diagonals, total length `batch_count * (n - 1)`.
/// * `diag` — main diagonals, total length `batch_count * n`.
/// * `upper` — super-diagonals, total length `batch_count * (n - 1)`.
/// * `rhs` — right-hand sides, total length `batch_count * n`. Overwritten with solutions.
/// * `n` — dimension of each system.
/// * `batch_count` — number of systems.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] if total array lengths are wrong.
/// Returns [`SolverError::SingularMatrix`] if any system has a zero pivot.
pub fn batched_tridiagonal_solve<T: GpuFloat>(
    _handle: &SolverHandle,
    lower: &[T],
    diag: &[T],
    upper: &[T],
    rhs: &mut [T],
    n: usize,
    batch_count: usize,
) -> SolverResult<()> {
    if batch_count == 0 || n == 0 {
        return Ok(());
    }

    let sub_len = n.saturating_sub(1);
    let expected_lower = batch_count * sub_len;
    let expected_diag = batch_count * n;
    let expected_upper = batch_count * sub_len;
    let expected_rhs = batch_count * n;

    if lower.len() < expected_lower {
        return Err(SolverError::DimensionMismatch(format!(
            "batched_tridiagonal_solve: lower length ({}) < expected ({})",
            lower.len(),
            expected_lower
        )));
    }
    if diag.len() < expected_diag {
        return Err(SolverError::DimensionMismatch(format!(
            "batched_tridiagonal_solve: diag length ({}) < expected ({})",
            diag.len(),
            expected_diag
        )));
    }
    if upper.len() < expected_upper {
        return Err(SolverError::DimensionMismatch(format!(
            "batched_tridiagonal_solve: upper length ({}) < expected ({})",
            upper.len(),
            expected_upper
        )));
    }
    if rhs.len() < expected_rhs {
        return Err(SolverError::DimensionMismatch(format!(
            "batched_tridiagonal_solve: rhs length ({}) < expected ({})",
            rhs.len(),
            expected_rhs
        )));
    }

    // Each system is independent — solve sequentially on CPU.
    // (On GPU each system would be a separate thread block.)
    for k in 0..batch_count {
        let l_start = k * sub_len;
        let d_start = k * n;
        let u_start = k * sub_len;
        let r_start = k * n;

        let l_slice = &lower[l_start..l_start + sub_len];
        let d_slice = &diag[d_start..d_start + n];
        let u_slice = &upper[u_start..u_start + sub_len];
        let r_slice = &mut rhs[r_start..r_start + n];

        if n == 1 {
            solve_1x1(d_slice, r_slice)?;
        } else {
            thomas_solve(l_slice, d_slice, u_slice, r_slice, n)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Thomas algorithm (sequential forward/backward elimination)
// ---------------------------------------------------------------------------

/// Thomas algorithm for tridiagonal systems.
///
/// Forward sweep modifies copies of the diagonal and RHS, then a backward
/// sweep computes the solution.
fn thomas_solve<T: GpuFloat>(
    lower: &[T],
    diag: &[T],
    upper: &[T],
    rhs: &mut [T],
    n: usize,
) -> SolverResult<()> {
    // Work with f64 for numerical stability.
    let mut c_prime = vec![0.0_f64; n];
    let mut d_prime = vec![0.0_f64; n];

    let d0 = to_f64(diag[0]);
    if d0.abs() < 1e-300 {
        return Err(SolverError::SingularMatrix);
    }

    c_prime[0] = to_f64(upper[0]) / d0;
    d_prime[0] = to_f64(rhs[0]) / d0;

    // Forward sweep.
    for i in 1..n {
        let a_i = to_f64(lower[i - 1]);
        let b_i = to_f64(diag[i]);
        let d_i = to_f64(rhs[i]);

        let denom = b_i - a_i * c_prime[i - 1];
        if denom.abs() < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }

        if i < n - 1 {
            c_prime[i] = to_f64(upper[i]) / denom;
        }
        d_prime[i] = (d_i - a_i * d_prime[i - 1]) / denom;
    }

    // Back substitution.
    rhs[n - 1] = from_f64(d_prime[n - 1]);
    for i in (0..n - 1).rev() {
        d_prime[i] -= c_prime[i] * to_f64(rhs[i + 1]);
        rhs[i] = from_f64(d_prime[i]);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Cyclic reduction (parallel-friendly)
// ---------------------------------------------------------------------------

/// Cyclic reduction algorithm for tridiagonal systems.
///
/// This algorithm reduces the system in O(log n) steps by eliminating
/// odd-indexed unknowns, then solving the reduced even-indexed system,
/// and back-substituting. Each reduction step can be parallelized.
///
/// On CPU this is O(n log n), but it maps to O(log n) parallel steps on GPU.
fn cyclic_reduction_solve<T: GpuFloat>(
    lower: &[T],
    diag: &[T],
    upper: &[T],
    rhs: &mut [T],
    n: usize,
) -> SolverResult<()> {
    // Work entirely in f64 for stability.
    let mut a = vec![0.0_f64; n]; // sub-diagonal (a[0] = 0)
    let mut b = vec![0.0_f64; n]; // main diagonal
    let mut c = vec![0.0_f64; n]; // super-diagonal (c[n-1] = 0)
    let mut d = vec![0.0_f64; n]; // RHS

    // Initialize from input arrays.
    b[0] = to_f64(diag[0]);
    d[0] = to_f64(rhs[0]);
    if n > 1 {
        c[0] = to_f64(upper[0]);
    }

    for i in 1..n {
        a[i] = to_f64(lower[i - 1]);
        b[i] = to_f64(diag[i]);
        d[i] = to_f64(rhs[i]);
        if i < n - 1 {
            c[i] = to_f64(upper[i]);
        }
    }

    // Forward reduction phases.
    let mut stride = 1_usize;
    let mut active_n = n;

    // Store reduction history for back-substitution.
    // Each level stores (a, b, c, d) for the reduced system.
    type ReductionLevel = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
    let mut levels: Vec<ReductionLevel> = Vec::new();

    while active_n > 2 {
        levels.push((a.clone(), b.clone(), c.clone(), d.clone()));

        let mut a_new = vec![0.0_f64; n];
        let mut b_new = vec![0.0_f64; n];
        let mut c_new = vec![0.0_f64; n];
        let mut d_new = vec![0.0_f64; n];

        // For each equation i at the current stride level,
        // eliminate dependencies on i-stride and i+stride.
        let mut count = 0;
        let mut i = stride;
        while i < n {
            let left = i.saturating_sub(stride);
            let right = if i + stride < n { i + stride } else { n - 1 };

            let bi = b[i];
            if bi.abs() < 1e-300 {
                return Err(SolverError::SingularMatrix);
            }

            // Elimination factors
            let alpha = if left < i && b[left].abs() > 1e-300 {
                -a[i] / b[left]
            } else {
                0.0
            };
            let gamma = if right > i && b[right].abs() > 1e-300 {
                -c[i] / b[right]
            } else {
                0.0
            };

            // Update equation i
            b_new[i] = bi + alpha * c[left] + gamma * a[right];
            d_new[i] = d[i] + alpha * d[left] + gamma * d[right];
            a_new[i] = alpha * a[left];
            c_new[i] = gamma * c[right];

            count += 1;
            i += 2 * stride;
        }

        a = a_new;
        b = b_new;
        c = c_new;
        d = d_new;

        stride *= 2;
        active_n = count;

        if active_n <= 1 {
            break;
        }
    }

    // Solve the reduced 1-2 equation system directly.
    // Find the active equations (those at positions that are multiples of stride).
    let mut active_indices = Vec::new();
    let mut idx = stride - 1;
    if idx >= n {
        // Fallback: if stride overshoots, use Thomas on original data.
        return thomas_solve(lower, diag, upper, rhs, n);
    }
    while idx < n {
        active_indices.push(idx);
        idx += stride;
    }

    // Solve the small system of active equations.
    match active_indices.len() {
        0 => {}
        1 => {
            let i = active_indices[0];
            if b[i].abs() < 1e-300 {
                return Err(SolverError::SingularMatrix);
            }
            d[i] /= b[i];
        }
        2 => {
            let i0 = active_indices[0];
            let i1 = active_indices[1];
            // 2x2 system: b[i0]*x0 + c[i0]*x1 = d[i0]
            //              a[i1]*x0 + b[i1]*x1 = d[i1]
            let det = b[i0] * b[i1] - c[i0] * a[i1];
            if det.abs() < 1e-300 {
                return Err(SolverError::SingularMatrix);
            }
            let x0 = (d[i0] * b[i1] - c[i0] * d[i1]) / det;
            let x1 = (b[i0] * d[i1] - a[i1] * d[i0]) / det;
            d[i0] = x0;
            d[i1] = x1;
        }
        _ => {
            // More than 2 active: solve with Thomas on the reduced system.
            let k = active_indices.len();
            let mut sub_a = vec![0.0_f64; k];
            let mut sub_b = vec![0.0_f64; k];
            let mut sub_c = vec![0.0_f64; k];
            let mut sub_d = vec![0.0_f64; k];
            for (j, &ai) in active_indices.iter().enumerate() {
                sub_a[j] = a[ai];
                sub_b[j] = b[ai];
                sub_c[j] = c[ai];
                sub_d[j] = d[ai];
            }
            // Thomas on the reduced system
            if sub_b[0].abs() < 1e-300 {
                return Err(SolverError::SingularMatrix);
            }
            let mut cp = vec![0.0_f64; k];
            let mut dp = vec![0.0_f64; k];
            cp[0] = sub_c[0] / sub_b[0];
            dp[0] = sub_d[0] / sub_b[0];
            for j in 1..k {
                let denom = sub_b[j] - sub_a[j] * cp[j - 1];
                if denom.abs() < 1e-300 {
                    return Err(SolverError::SingularMatrix);
                }
                cp[j] = sub_c[j] / denom;
                dp[j] = (sub_d[j] - sub_a[j] * dp[j - 1]) / denom;
            }
            sub_d[k - 1] = dp[k - 1];
            for j in (0..k - 1).rev() {
                sub_d[j] = dp[j] - cp[j] * sub_d[j + 1];
            }
            for (j, &ai) in active_indices.iter().enumerate() {
                d[ai] = sub_d[j];
            }
        }
    }

    // Back-substitution through reduction levels.
    for level_data in levels.iter().rev() {
        let (ref la, ref lb, ref _lc, ref ld) = *level_data;
        // Recover solutions for equations that were eliminated.
        let half_stride = stride / 2;
        let mut i = half_stride.saturating_sub(1);
        while i < n {
            // Check if this index is NOT already solved (not a multiple of stride)
            let is_solved = if stride > 0 {
                (i + 1) % stride == 0
            } else {
                false
            };

            if !is_solved {
                // This equation was eliminated: x[i] = (d[i] - a[i]*x[i-hs] - c[i]*x[i+hs]) / b[i]
                let bi = lb[i];
                if bi.abs() < 1e-300 {
                    return Err(SolverError::SingularMatrix);
                }
                let left_val = if i >= half_stride {
                    d[i - half_stride]
                } else {
                    0.0
                };
                let right_val = if i + half_stride < n {
                    d[i + half_stride]
                } else {
                    0.0
                };
                d[i] = (ld[i] - la[i] * left_val - level_data.2[i] * right_val) / bi;
            }
            i += half_stride;
        }
        stride /= 2;
    }

    // Write solution back.
    for i in 0..n {
        rhs[i] = from_f64(d[i]);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validates dimensions for tridiagonal solver inputs.
fn validate_tridiagonal_dims<T: GpuFloat>(
    lower: &[T],
    diag: &[T],
    upper: &[T],
    rhs: &[T],
    n: usize,
) -> SolverResult<()> {
    if diag.len() < n {
        return Err(SolverError::DimensionMismatch(format!(
            "tridiagonal_solve: diag length ({}) < n ({n})",
            diag.len()
        )));
    }
    if rhs.len() < n {
        return Err(SolverError::DimensionMismatch(format!(
            "tridiagonal_solve: rhs length ({}) < n ({n})",
            rhs.len()
        )));
    }
    if n > 1 {
        if lower.len() < n - 1 {
            return Err(SolverError::DimensionMismatch(format!(
                "tridiagonal_solve: lower length ({}) < n-1 ({})",
                lower.len(),
                n - 1
            )));
        }
        if upper.len() < n - 1 {
            return Err(SolverError::DimensionMismatch(format!(
                "tridiagonal_solve: upper length ({}) < n-1 ({})",
                upper.len(),
                n - 1
            )));
        }
    }
    Ok(())
}

/// Solves a trivial 1x1 system.
fn solve_1x1<T: GpuFloat>(diag: &[T], rhs: &mut [T]) -> SolverResult<()> {
    let d = to_f64(diag[0]);
    if d.abs() < 1e-300 {
        return Err(SolverError::SingularMatrix);
    }
    rhs[0] = from_f64(to_f64(rhs[0]) / d);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Validation tests ---

    #[test]
    fn validate_dims_ok() {
        let lower = [1.0_f64; 2];
        let diag = [2.0_f64; 3];
        let upper = [1.0_f64; 2];
        let rhs = [1.0_f64; 3];
        let result = validate_tridiagonal_dims(&lower, &diag, &upper, &rhs, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_dims_diag_too_short() {
        let lower = [1.0_f64; 2];
        let diag = [2.0_f64; 2];
        let upper = [1.0_f64; 2];
        let rhs = [1.0_f64; 3];
        let result = validate_tridiagonal_dims(&lower, &diag, &upper, &rhs, 3);
        assert!(result.is_err());
    }

    #[test]
    fn validate_dims_lower_too_short() {
        let lower = [1.0_f64; 1];
        let diag = [2.0_f64; 3];
        let upper = [1.0_f64; 2];
        let rhs = [1.0_f64; 3];
        let result = validate_tridiagonal_dims(&lower, &diag, &upper, &rhs, 3);
        assert!(result.is_err());
    }

    // --- Thomas algorithm tests ---

    #[test]
    fn thomas_solve_2x2() {
        // [2 1] [x0]   [5]
        // [1 3] [x1] = [7]
        // x0 = 1.6, x1 = 1.8
        let lower = [1.0_f64];
        let diag = [2.0_f64, 3.0];
        let upper = [1.0_f64];
        let mut rhs = [5.0_f64, 7.0];

        let result = thomas_solve(&lower, &diag, &upper, &mut rhs, 2);
        assert!(result.is_ok());
        assert!((rhs[0] - 1.6).abs() < 1e-10);
        assert!((rhs[1] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn thomas_solve_3x3() {
        // [4 1 0] [x0]   [5]
        // [1 4 1] [x1] = [6]
        // [0 1 4] [x2]   [5]
        // Solution: x0=1, x1=1, x2=1
        let lower = [1.0_f64, 1.0];
        let diag = [4.0_f64, 4.0, 4.0];
        let upper = [1.0_f64, 1.0];
        let mut rhs = [5.0_f64, 6.0, 5.0];

        let result = thomas_solve(&lower, &diag, &upper, &mut rhs, 3);
        assert!(result.is_ok());
        assert!((rhs[0] - 1.0).abs() < 1e-10);
        assert!((rhs[1] - 1.0).abs() < 1e-10);
        assert!((rhs[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn thomas_solve_singular() {
        let lower = [1.0_f64];
        let diag = [0.0_f64, 1.0]; // zero pivot
        let upper = [1.0_f64];
        let mut rhs = [1.0_f64, 1.0];

        let result = thomas_solve(&lower, &diag, &upper, &mut rhs, 2);
        assert!(result.is_err());
    }

    // --- Cyclic reduction tests ---

    #[test]
    fn cyclic_reduction_3x3() {
        let lower = [1.0_f64, 1.0];
        let diag = [4.0_f64, 4.0, 4.0];
        let upper = [1.0_f64, 1.0];
        let mut rhs = [5.0_f64, 6.0, 5.0];

        let result = cyclic_reduction_solve(&lower, &diag, &upper, &mut rhs, 3);
        assert!(result.is_ok());
        assert!((rhs[0] - 1.0).abs() < 1e-8);
        assert!((rhs[1] - 1.0).abs() < 1e-8);
        assert!((rhs[2] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn cyclic_reduction_4x4() {
        // [2 -1  0  0] [x0]   [1]
        // [-1 2 -1  0] [x1] = [0]
        // [0 -1  2 -1] [x2]   [0]
        // [0  0 -1  2] [x3]   [1]
        // Solution: x = [1, 1, 1, 1]
        let lower = [-1.0_f64, -1.0, -1.0];
        let diag = [2.0_f64, 2.0, 2.0, 2.0];
        let upper = [-1.0_f64, -1.0, -1.0];
        let mut rhs = [1.0_f64, 0.0, 0.0, 1.0];

        let result = cyclic_reduction_solve(&lower, &diag, &upper, &mut rhs, 4);
        assert!(result.is_ok());
        for (i, &val) in rhs.iter().enumerate() {
            assert!((val - 1.0).abs() < 1e-8, "x[{i}] = {val} (expected 1.0)",);
        }
    }

    // --- 1x1 solve test ---

    #[test]
    fn solve_1x1_basic() {
        let diag = [5.0_f64];
        let mut rhs = [10.0_f64];
        let result = solve_1x1(&diag, &mut rhs);
        assert!(result.is_ok());
        assert!((rhs[0] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn solve_1x1_zero_diag() {
        let diag = [0.0_f64];
        let mut rhs = [10.0_f64];
        let result = solve_1x1(&diag, &mut rhs);
        assert!(result.is_err());
    }

    // --- f32 tests ---

    #[test]
    fn thomas_solve_f32() {
        let lower = [1.0_f32];
        let diag = [2.0_f32, 3.0];
        let upper = [1.0_f32];
        let mut rhs = [5.0_f32, 7.0];

        let result = thomas_solve(&lower, &diag, &upper, &mut rhs, 2);
        assert!(result.is_ok());
        assert!((rhs[0] - 1.6_f32).abs() < 1e-5);
        assert!((rhs[1] - 1.8_f32).abs() < 1e-5);
    }

    // --- Conversion roundtrip ---

    #[test]
    fn f64_roundtrip() {
        let val = std::f64::consts::PI;
        let back: f64 = from_f64(to_f64(val));
        assert!((back - val).abs() < 1e-15);
    }

    #[test]
    fn f32_roundtrip() {
        let val = 3.15_f32;
        let back: f32 = from_f64(to_f64(val));
        assert!((back - val).abs() < 1e-6);
    }
}
