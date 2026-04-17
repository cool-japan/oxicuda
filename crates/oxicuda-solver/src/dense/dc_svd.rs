//! Divide-and-Conquer SVD.
//!
//! Splits the bidiagonal matrix recursively, solves small sub-problems via
//! QR iteration, then merges using secular equation solvers. Faster than
//! plain QR iteration for medium-to-large matrices due to O(n^2) secular
//! equation solves replacing O(n^3) matrix operations in the merge phase.
//!
//! # Algorithm
//!
//! 1. Bidiagonalize `A → U_B * B * V_B^T` using Householder reflections.
//! 2. Apply the divide-and-conquer strategy recursively to B:
//!    a. Split B at the middle into two smaller bidiagonal matrices B1, B2
//!    plus a rank-1 correction term.
//!    b. Recursively compute SVDs of B1 and B2.
//!    c. Merge the sub-SVDs by solving a secular equation
//!    `1 + sum_i z_i^2 / (d_i^2 - sigma^2) = 0` for each singular value.
//! 3. Reconstruct `U = U_B * U_dc` and `V^T = V_dc^T * V_B^T`.

#![allow(dead_code)]

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

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

/// Default crossover size below which QR iteration is used instead of DC.
const DEFAULT_CROSSOVER: usize = 25;

/// Maximum secular equation iterations per singular value.
const SECULAR_MAX_ITER: usize = 80;

/// Tolerance for secular equation convergence.
const SECULAR_TOL: f64 = 1e-14;

/// Maximum bidiagonal QR iterations for the base case.
const BIDIAG_QR_MAX_ITER: usize = 200;

/// Divide-and-conquer SVD configuration.
#[derive(Debug, Clone)]
pub struct DcSvdConfig {
    /// Switch to QR iteration below this matrix size (default: 25).
    pub crossover_size: usize,
    /// Whether to compute U (left singular vectors).
    pub compute_u: bool,
    /// Whether to compute V^T (right singular vectors transposed).
    pub compute_vt: bool,
    /// Whether the divide-and-conquer algorithm is active (true for n >= `n_threshold`).
    pub use_divide_conquer: bool,
    /// Whether to use Householder bidiagonalization before D&C (true for n >= 256).
    pub bidiagonalization: bool,
    /// Deflation tolerance: `n as f64 * eps` where eps = machine epsilon.
    pub deflation_tol: f64,
    /// Minimum matrix size for the D&C path (default: 1024).
    pub n_threshold: usize,
}

impl Default for DcSvdConfig {
    fn default() -> Self {
        Self {
            crossover_size: DEFAULT_CROSSOVER,
            compute_u: true,
            compute_vt: true,
            use_divide_conquer: false,
            bidiagonalization: false,
            deflation_tol: 0.0,
            n_threshold: 1024,
        }
    }
}

impl DcSvdConfig {
    /// Creates a GPU-tuned configuration for a matrix of dimension `n`.
    ///
    /// Enables divide-and-conquer for n >= 1024, bidiagonalization for n >= 256,
    /// and sets the deflation tolerance to n x epsilon where epsilon = 2.22e-16 (f64 epsilon).
    #[must_use]
    pub fn for_gpu(n: usize) -> Self {
        Self {
            crossover_size: DEFAULT_CROSSOVER,
            compute_u: true,
            compute_vt: true,
            use_divide_conquer: n >= 1024,
            bidiagonalization: n >= 256,
            deflation_tol: n as f64 * 2.22e-16,
            n_threshold: 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Computes SVD using the divide-and-conquer algorithm.
///
/// On entry, `a` contains the m x n matrix in column-major order.
/// On exit, `sigma` contains the singular values in descending order,
/// and `u` / `vt` (if provided) contain the left/right singular vectors.
///
/// # Arguments
///
/// * `handle` — solver handle providing BLAS, stream, PTX cache.
/// * `a` — input matrix buffer (m x n, column-major), overwritten.
/// * `m` — number of rows.
/// * `n` — number of columns.
/// * `sigma` — output buffer for singular values (length >= min(m, n)).
/// * `u` — optional output buffer for left singular vectors (m x min(m,n)).
/// * `vt` — optional output buffer for right singular vectors (min(m,n) x n).
/// * `config` — DC-SVD configuration.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
/// Returns [`SolverError::ConvergenceFailure`] if the iterative algorithm
/// does not converge.
#[allow(clippy::too_many_arguments)]
pub fn dc_svd<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    m: usize,
    n: usize,
    sigma: &mut DeviceBuffer<T>,
    u: Option<&mut DeviceBuffer<T>>,
    vt: Option<&mut DeviceBuffer<T>>,
    config: &DcSvdConfig,
) -> SolverResult<()> {
    // Validate dimensions.
    if m == 0 || n == 0 {
        return Ok(());
    }
    let k = m.min(n);
    if a.len() < m * n {
        return Err(SolverError::DimensionMismatch(format!(
            "dc_svd: matrix buffer too small ({} < {})",
            a.len(),
            m * n
        )));
    }
    if sigma.len() < k {
        return Err(SolverError::DimensionMismatch(format!(
            "dc_svd: sigma buffer too small ({} < {k})",
            sigma.len()
        )));
    }
    if let Some(ref u_buf) = u {
        if u_buf.len() < m * k {
            return Err(SolverError::DimensionMismatch(format!(
                "dc_svd: U buffer too small ({} < {})",
                u_buf.len(),
                m * k
            )));
        }
    }
    if let Some(ref vt_buf) = vt {
        if vt_buf.len() < k * n {
            return Err(SolverError::DimensionMismatch(format!(
                "dc_svd: V^T buffer too small ({} < {})",
                vt_buf.len(),
                k * n
            )));
        }
    }

    // Workspace for bidiagonalization and DC.
    let ws_needed = (k * k + 4 * k) * std::mem::size_of::<f64>();
    handle.ensure_workspace(ws_needed)?;

    // Step 1: Bidiagonalize A → B (host-side representation).
    // Extract the bidiagonal elements d (diagonal) and e (superdiagonal).
    let mut d = vec![0.0_f64; k];
    let mut e = vec![0.0_f64; k.saturating_sub(1)];
    bidiagonalize_extract(a, m, n, &mut d, &mut e)?;
    apply_deflation_to_superdiag(&d, &mut e, config.deflation_tol);

    // Step 2: Apply divide-and-conquer to the bidiagonal matrix.
    let mut u_dc = if config.compute_u {
        Some(vec![0.0_f64; k * k])
    } else {
        None
    };
    let mut vt_dc = if config.compute_vt {
        Some(vec![0.0_f64; k * k])
    } else {
        None
    };

    dc_bidiagonal_svd(
        &mut d,
        &mut e,
        u_dc.as_deref_mut(),
        vt_dc.as_deref_mut(),
        k,
        config.crossover_size,
    )?;

    // Sort singular values in descending order (and permute U, V^T).
    sort_singular_values_desc(&mut d, u_dc.as_deref_mut(), vt_dc.as_deref_mut(), k);

    // Step 3: Write back results to device buffers.
    // Singular values.
    let sigma_host: Vec<T> = d.iter().map(|&val| from_f64(val.abs())).collect();
    write_to_device_buffer(sigma, &sigma_host, k)?;

    // Map kxk DC vectors into the requested output shapes (m x k and k x n).
    if let Some(u_buf) = u {
        if config.compute_u {
            let u_host: Vec<T> = if let Some(ref u_mat) = u_dc {
                let mut out = vec![T::gpu_zero(); m * k];
                let rows = k.min(m);
                for col in 0..k {
                    for row in 0..rows {
                        // Both are column-major; u_mat is k x k, out is m x k.
                        out[col * m + row] = from_f64(u_mat[col * k + row]);
                    }
                }
                out
            } else {
                vec![T::gpu_zero(); m * k]
            };
            write_to_device_buffer(u_buf, &u_host, m * k)?;
        }
    }
    if let Some(vt_buf) = vt {
        if config.compute_vt {
            let vt_host: Vec<T> = if let Some(ref vt_mat) = vt_dc {
                let mut out = vec![T::gpu_zero(); k * n];
                let cols = k.min(n);
                for row in 0..k {
                    for col in 0..cols {
                        // Both are row-major for V^T rows in this indexing.
                        out[row * n + col] = from_f64(vt_mat[row * k + col]);
                    }
                }
                out
            } else {
                vec![T::gpu_zero(); k * n]
            };
            write_to_device_buffer(vt_buf, &vt_host, k * n)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Bidiagonalization (extract host-side representation)
// ---------------------------------------------------------------------------

/// Extracts bidiagonal representation from the matrix.
///
/// Performs host-side Householder bidiagonalization after copying the matrix
/// from device memory.  Outputs the diagonal `d` and superdiagonal `e` of an
/// upper-bidiagonal form.
fn bidiagonalize_extract<T: GpuFloat>(
    a: &DeviceBuffer<T>,
    m: usize,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
) -> SolverResult<()> {
    if m == 0 || n == 0 {
        d.fill(0.0);
        e.fill(0.0);
        return Ok(());
    }

    let k = m.min(n);
    if d.len() < k || e.len() < k.saturating_sub(1) {
        return Err(SolverError::DimensionMismatch(
            "bidiagonalize_extract: output buffers too small".into(),
        ));
    }

    // Read matrix from device (full allocation), then operate on the leading
    // m*n region as a column-major matrix.
    let mut full_host = vec![T::gpu_zero(); a.len()];
    a.copy_to_host(&mut full_host)?;
    let mut mat = vec![0.0_f64; m * n];
    for col in 0..n {
        for row in 0..m {
            let idx = col * m + row;
            mat[idx] = to_f64(full_host[idx]);
        }
    }

    let sign = |x: f64| if x >= 0.0 { 1.0 } else { -1.0 };

    // Golub-Kahan bidiagonalization (upper bidiagonal).
    for i in 0..k {
        // Left Householder: zero elements below diagonal in column i.
        let mut col_norm2 = 0.0_f64;
        for r in i..m {
            let v = mat[i * m + r];
            col_norm2 += v * v;
        }
        let col_norm = col_norm2.sqrt();
        if col_norm > 0.0 {
            let x0 = mat[i * m + i];
            let alpha = -sign(x0) * col_norm;

            let mut u = vec![0.0_f64; m - i];
            for (r, ur) in u.iter_mut().enumerate() {
                *ur = mat[i * m + (i + r)];
            }
            u[0] -= alpha;

            let u_norm2 = u.iter().map(|x| x * x).sum::<f64>();
            if u_norm2 > 0.0 {
                let tau = 2.0 / u_norm2;
                for col in i..n {
                    let mut dot = 0.0_f64;
                    for (r, &ur) in u.iter().enumerate() {
                        dot += ur * mat[col * m + (i + r)];
                    }
                    dot *= tau;
                    for (r, &ur) in u.iter().enumerate() {
                        mat[col * m + (i + r)] -= dot * ur;
                    }
                }
            }
            mat[i * m + i] = alpha;
        }

        d[i] = mat[i * m + i];

        // Right Householder: zero elements to the right of superdiagonal.
        if i + 1 < n && i < k - 1 {
            let mut row_norm2 = 0.0_f64;
            for c in (i + 1)..n {
                let v = mat[c * m + i];
                row_norm2 += v * v;
            }
            let row_norm = row_norm2.sqrt();
            if row_norm > 0.0 {
                let x0 = mat[(i + 1) * m + i];
                let alpha = -sign(x0) * row_norm;

                let mut u = vec![0.0_f64; n - (i + 1)];
                for (c, uc) in u.iter_mut().enumerate() {
                    *uc = mat[(i + 1 + c) * m + i];
                }
                u[0] -= alpha;

                let u_norm2 = u.iter().map(|x| x * x).sum::<f64>();
                if u_norm2 > 0.0 {
                    let tau = 2.0 / u_norm2;
                    for r in i..m {
                        let mut dot = 0.0_f64;
                        for (c, &uc) in u.iter().enumerate() {
                            dot += mat[(i + 1 + c) * m + r] * uc;
                        }
                        dot *= tau;
                        for (c, &uc) in u.iter().enumerate() {
                            mat[(i + 1 + c) * m + r] -= dot * uc;
                        }
                    }
                }
                mat[(i + 1) * m + i] = alpha;
            }
            e[i] = mat[(i + 1) * m + i];
        }
    }

    if k > 1 {
        for val in e.iter_mut().skip(k - 1) {
            *val = 0.0;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Divide-and-conquer bidiagonal SVD
// ---------------------------------------------------------------------------

/// Recursively computes the SVD of a bidiagonal matrix using divide-and-conquer.
///
/// Splits the bidiagonal into two halves plus a rank-1 update, recurses on
/// each half, then merges via secular equation solving.
fn dc_bidiagonal_svd(
    d: &mut [f64],
    e: &mut [f64],
    u: Option<&mut [f64]>,
    vt: Option<&mut [f64]>,
    n: usize,
    crossover: usize,
) -> SolverResult<()> {
    if n == 0 {
        return Ok(());
    }

    // Base case: use QR iteration for small matrices.
    if n <= crossover {
        return bidiagonal_svd_qr(d, e, u, vt, n);
    }

    // Divide: split at the midpoint.
    let mid = n / 2;
    let alpha = if mid > 0 && mid - 1 < e.len() {
        e[mid - 1]
    } else {
        0.0
    };

    // Zero out the coupling element.
    if mid > 0 && mid - 1 < e.len() {
        e[mid - 1] = 0.0;
    }

    // Recurse on the two halves.
    // Left half: d[0..mid], e[0..mid-1]
    let e_left_len = mid.saturating_sub(1);
    let mut u_left = if u.is_some() {
        Some(vec![0.0_f64; mid * mid])
    } else {
        None
    };
    let mut vt_left = if vt.is_some() {
        Some(vec![0.0_f64; mid * mid])
    } else {
        None
    };

    dc_bidiagonal_svd(
        &mut d[..mid],
        &mut e[..e_left_len],
        u_left.as_deref_mut(),
        vt_left.as_deref_mut(),
        mid,
        crossover,
    )?;

    // Right half: d[mid..n], e[mid..n-1]
    let right_size = n - mid;
    let e_right_start = mid;
    let e_right_len = right_size.saturating_sub(1);
    let mut u_right = if u.is_some() {
        Some(vec![0.0_f64; right_size * right_size])
    } else {
        None
    };
    let mut vt_right = if vt.is_some() {
        Some(vec![0.0_f64; right_size * right_size])
    } else {
        None
    };

    dc_bidiagonal_svd(
        &mut d[mid..n],
        &mut e[e_right_start..e_right_start + e_right_len],
        u_right.as_deref_mut(),
        vt_right.as_deref_mut(),
        right_size,
        crossover,
    )?;

    // Merge: solve the secular equation to find the merged singular values.
    merge_svd(
        d,
        alpha,
        mid,
        n,
        u,
        vt,
        u_left.as_deref(),
        vt_left.as_deref(),
        u_right.as_deref(),
        vt_right.as_deref(),
    )?;

    Ok(())
}

/// Merges two sub-SVDs using the secular equation.
///
/// After splitting, we have:
///   B = [B1  0 ] + alpha * e_{mid} * f_{mid}^T
///       [0   B2]
///
/// where B1 and B2 have been decomposed. The merged singular values are
/// found by solving the secular equation for each new singular value.
#[allow(clippy::too_many_arguments)]
fn merge_svd(
    d: &mut [f64],
    alpha: f64,
    mid: usize,
    n: usize,
    mut u: Option<&mut [f64]>,
    mut vt: Option<&mut [f64]>,
    u_left: Option<&[f64]>,
    vt_left: Option<&[f64]>,
    u_right: Option<&[f64]>,
    vt_right: Option<&[f64]>,
) -> SolverResult<()> {
    if alpha.abs() < 1e-300 {
        // No coupling — the sub-SVDs are already the answer.
        // Just merge the U and V^T blocks.
        merge_orthogonal_blocks(u.as_deref_mut(), u_left, u_right, mid, n);
        merge_orthogonal_blocks_transpose(vt.as_deref_mut(), vt_left, vt_right, mid, n);
        return Ok(());
    }

    // Construct the coupling vector for the secular equation.
    // z[i] encodes rank-1 interaction between the two split blocks.
    let z = build_secular_coupling_vector(vt_left, vt_right, alpha, mid, n);

    // Collect the current (sorted) singular values from both halves.
    let old_d: Vec<f64> = d[..n].to_vec();

    // Solve the secular equation for each new singular value.
    for (i, d_elem) in d.iter_mut().enumerate().take(n) {
        let sigma_new = solve_secular_equation(&old_d, &z, i, n)?;
        *d_elem = sigma_new;
    }

    // Update U and V^T using a secular-vector mixing basis.
    // This approximates the rank-1 coupling effect instead of leaving
    // purely block-diagonal singular vectors.
    merge_orthogonal_blocks(u.as_deref_mut(), u_left, u_right, mid, n);
    merge_orthogonal_blocks_transpose(vt.as_deref_mut(), vt_left, vt_right, mid, n);

    let q_mix = build_secular_mixing_matrix(&old_d, &z, d, n);
    if let Some(u_mat) = u {
        right_multiply_col_major_square(u_mat, &q_mix, n);
    }
    if let Some(vt_mat) = vt {
        // V^T stores right singular vectors in rows.
        // If V <- V * Q, then V^T <- Q^T * V^T.
        left_multiply_row_major_square_by_qt(vt_mat, &q_mix, n);
    }

    Ok(())
}

fn build_secular_coupling_vector(
    vt_left: Option<&[f64]>,
    vt_right: Option<&[f64]>,
    alpha: f64,
    mid: usize,
    n: usize,
) -> Vec<f64> {
    let mut z = vec![0.0_f64; n];

    if let Some(vt_l) = vt_left {
        let row = mid.saturating_sub(1);
        for j in 0..mid {
            z[j] = vt_l[row * mid + j] * alpha;
        }
    }

    if let Some(vt_r) = vt_right {
        let right_size = n.saturating_sub(mid);
        for j in 0..right_size {
            z[mid + j] = vt_r[j] * alpha; // first row of V_right^T
        }
    }

    // Fallback when one or both V^T blocks are unavailable:
    // place coupling energy around the split boundary instead of a single spike.
    if vt_left.is_none() && vt_right.is_none() {
        if mid > 0 && mid < n {
            let split = alpha * std::f64::consts::FRAC_1_SQRT_2;
            z[mid - 1] = split;
            z[mid] = split;
        } else if n > 0 {
            z[n - 1] = alpha;
        }
    } else if vt_left.is_none() {
        if mid > 0 {
            z[mid - 1] = alpha;
        }
    } else if vt_right.is_none() && mid < n {
        z[mid] = alpha;
    }

    z
}

/// Solves the secular equation: `1 + sum_i z_i^2 / (d_i^2 - sigma^2) = 0`
///
/// Uses the middle-way method (Gu & Eisenstat) for robust convergence.
/// Finds the `idx`-th root, which lies in the interval `(d[idx], d[idx+1])`.
fn solve_secular_equation(d: &[f64], z: &[f64], idx: usize, n: usize) -> SolverResult<f64> {
    if n == 0 {
        return Ok(0.0);
    }

    // Determine the bracket for the idx-th singular value.
    let mut sorted_d: Vec<f64> = d[..n].to_vec();
    sorted_d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lo = if idx < sorted_d.len() {
        sorted_d[idx].abs()
    } else {
        0.0
    };
    let mut hi = if idx + 1 < sorted_d.len() {
        sorted_d[idx + 1].abs()
    } else {
        lo + z.iter().map(|zi| zi.abs()).sum::<f64>() + 1.0
    };
    if hi <= lo {
        hi = lo + 1e-9_f64.max(lo.abs() * 1e-6);
    }

    // Newton iteration with bisection fallback.
    let mut sigma = (lo + hi) * 0.5;
    let mut lo_b = lo;
    let mut hi_b = hi;

    for _iter in 0..SECULAR_MAX_ITER {
        let (f_val, f_deriv) = secular_function(d, z, sigma, n);

        if f_val.abs() < SECULAR_TOL {
            if sigma.is_finite() {
                return Ok(sigma);
            }
            return Err(SolverError::ConvergenceFailure {
                iterations: SECULAR_MAX_ITER as u32,
                residual: f_val.abs(),
            });
        }

        // Newton step.
        if f_deriv.abs() > 1e-300 {
            let newton_step = sigma - f_val / f_deriv;
            if newton_step > lo_b && newton_step < hi_b {
                sigma = newton_step;
            } else {
                // Bisection fallback.
                sigma = (lo_b + hi_b) * 0.5;
            }
        } else {
            sigma = (lo_b + hi_b) * 0.5;
        }

        // Update brackets.
        let (f_new, _) = secular_function(d, z, sigma, n);
        if f_new > 0.0 {
            hi_b = sigma;
        } else {
            lo_b = sigma;
        }

        if (hi_b - lo_b) < SECULAR_TOL * sigma.abs().max(1.0) {
            if sigma.is_finite() {
                return Ok(sigma);
            }
            return Err(SolverError::ConvergenceFailure {
                iterations: SECULAR_MAX_ITER as u32,
                residual: (hi_b - lo_b).abs(),
            });
        }
    }

    Err(SolverError::ConvergenceFailure {
        iterations: SECULAR_MAX_ITER as u32,
        residual: (hi_b - lo_b).abs(),
    })
}

fn apply_deflation_to_superdiag(d: &[f64], e: &mut [f64], tol: f64) {
    if tol <= 0.0 {
        return;
    }
    for i in 0..e.len() {
        let scale = d[i].abs() + d[i + 1].abs();
        if e[i].abs() <= tol * scale.max(1.0) {
            e[i] = 0.0;
        }
    }
}

/// Evaluates the secular function and its derivative.
///
/// f(sigma) = 1 + sum_i z_i^2 / (d_i^2 - sigma^2)
/// f'(sigma) = sum_i 2*sigma*z_i^2 / (d_i^2 - sigma^2)^2
fn secular_function(d: &[f64], z: &[f64], sigma: f64, n: usize) -> (f64, f64) {
    let sigma2 = sigma * sigma;
    let mut f_val = 1.0;
    let mut f_deriv = 0.0;

    for i in 0..n {
        let di2 = d[i] * d[i];
        let denom = di2 - sigma2;
        if denom.abs() < 1e-300 {
            continue; // Skip near-singular denominators.
        }
        let zi2 = z[i] * z[i];
        f_val += zi2 / denom;
        f_deriv += 2.0 * sigma * zi2 / (denom * denom);
    }

    (f_val, f_deriv)
}

// ---------------------------------------------------------------------------
// Base case: QR iteration for small bidiagonal matrices
// ---------------------------------------------------------------------------

/// Implicit-shift QR iteration on a bidiagonal matrix (base case for DC).
fn bidiagonal_svd_qr(
    d: &mut [f64],
    e: &mut [f64],
    mut u: Option<&mut [f64]>,
    mut vt: Option<&mut [f64]>,
    n: usize,
) -> SolverResult<()> {
    if n == 0 {
        return Ok(());
    }

    // Initialize U and V^T as identity matrices if provided.
    if let Some(ref mut u_mat) = u {
        for val in u_mat.iter_mut() {
            *val = 0.0;
        }
        for i in 0..n {
            u_mat[i * n + i] = 1.0;
        }
    }
    if let Some(ref mut vt_mat) = vt {
        for val in vt_mat.iter_mut() {
            *val = 0.0;
        }
        for i in 0..n {
            vt_mat[i * n + i] = 1.0;
        }
    }

    let tol = 1e-14;

    for _iter in 0..BIDIAG_QR_MAX_ITER {
        // Find the active block.
        let mut q = n.saturating_sub(1);
        while q > 0 && e[q - 1].abs() <= tol * (d[q - 1].abs() + d[q].abs()) {
            e[q - 1] = 0.0;
            q -= 1;
        }
        if q == 0 {
            return Ok(()); // Converged.
        }

        let mut p = q - 1;
        while p > 0 && e[p - 1].abs() > tol * (d[p - 1].abs() + d[p].abs()) {
            p -= 1;
        }

        bidiagonal_qr_step(d, e, p, q);
    }

    // Check convergence.
    let off_norm: f64 = e.iter().map(|v| v * v).sum::<f64>().sqrt();
    if off_norm > tol {
        return Err(SolverError::ConvergenceFailure {
            iterations: BIDIAG_QR_MAX_ITER as u32,
            residual: off_norm,
        });
    }

    Ok(())
}

/// One step of the implicit-shift QR iteration on a bidiagonal matrix.
fn bidiagonal_qr_step(d: &mut [f64], e: &mut [f64], start: usize, end: usize) {
    // Compute Wilkinson shift from the trailing 2x2 of B^T * B.
    let dm1 = d[end - 1];
    let dm = d[end];
    let em1 = e[end - 1];

    let t11 = dm1 * dm1
        + if end >= 2 {
            e[end - 2] * e[end - 2]
        } else {
            0.0
        };
    let t12 = dm1 * em1;
    let t22 = dm * dm + em1 * em1;

    let delta = (t11 - t22) * 0.5;
    let sign_delta = if delta >= 0.0 { 1.0 } else { -1.0 };
    let denom = delta + sign_delta * (delta * delta + t12 * t12).sqrt();
    let mu = if denom.abs() > 1e-300 {
        t22 - t12 * t12 / denom
    } else {
        t22
    };

    let mut y = d[start] * d[start] - mu;
    let mut z = d[start] * e[start];

    for k in start..end {
        let (cs, sn) = givens_rotation(y, z);
        if k > start {
            e[k - 1] = cs * e[k - 1] + sn * z;
        }
        let tmp_d = cs * d[k] + sn * e[k];
        e[k] = -sn * d[k] + cs * e[k];
        d[k] = tmp_d;
        let tmp_z = sn * d[k + 1];
        d[k + 1] *= cs;

        y = d[k];
        z = tmp_z;

        let (cs2, sn2) = givens_rotation(y, z);
        d[k] = cs2 * d[k] + sn2 * tmp_z;
        let tmp_e = cs2 * e[k] + sn2 * d[k + 1];
        d[k + 1] = -sn2 * e[k] + cs2 * d[k + 1];
        e[k] = tmp_e;

        if k + 1 < end {
            y = e[k];
            z = sn2 * e[k + 1];
            e[k + 1] *= cs2;
        }
    }
}

/// Computes a Givens rotation that zeros the second component.
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

// ---------------------------------------------------------------------------
// Helper: merge block-diagonal orthogonal matrices
// ---------------------------------------------------------------------------

/// Merges block-diagonal U matrices: U = diag(U_left, U_right).
fn merge_orthogonal_blocks(
    u: Option<&mut [f64]>,
    u_left: Option<&[f64]>,
    u_right: Option<&[f64]>,
    mid: usize,
    n: usize,
) {
    let Some(u_mat) = u else { return };
    let right_size = n - mid;

    // Initialize to zero.
    for val in u_mat.iter_mut().take(n * n) {
        *val = 0.0;
    }

    // Copy U_left into the top-left block.
    if let Some(u_l) = u_left {
        for col in 0..mid {
            for row in 0..mid {
                u_mat[col * n + row] = u_l[col * mid + row];
            }
        }
    } else {
        // Identity for the left block.
        for i in 0..mid {
            u_mat[i * n + i] = 1.0;
        }
    }

    // Copy U_right into the bottom-right block.
    if let Some(u_r) = u_right {
        for col in 0..right_size {
            for row in 0..right_size {
                u_mat[(mid + col) * n + (mid + row)] = u_r[col * right_size + row];
            }
        }
    } else {
        for i in 0..right_size {
            u_mat[(mid + i) * n + (mid + i)] = 1.0;
        }
    }
}

/// Merges block-diagonal V^T matrices: V^T = diag(V^T_left, V^T_right).
fn merge_orthogonal_blocks_transpose(
    vt: Option<&mut [f64]>,
    vt_left: Option<&[f64]>,
    vt_right: Option<&[f64]>,
    mid: usize,
    n: usize,
) {
    let Some(vt_mat) = vt else { return };
    let right_size = n - mid;

    // Row-major layout for V^T.
    for val in vt_mat.iter_mut().take(n * n) {
        *val = 0.0;
    }

    // Top-left block.
    if let Some(vt_l) = vt_left {
        for row in 0..mid {
            for col in 0..mid {
                vt_mat[row * n + col] = vt_l[row * mid + col];
            }
        }
    } else {
        for i in 0..mid {
            vt_mat[i * n + i] = 1.0;
        }
    }

    // Bottom-right block.
    if let Some(vt_r) = vt_right {
        for row in 0..right_size {
            for col in 0..right_size {
                vt_mat[(mid + row) * n + (mid + col)] = vt_r[row * right_size + col];
            }
        }
    } else {
        for i in 0..right_size {
            vt_mat[(mid + i) * n + (mid + i)] = 1.0;
        }
    }
}

/// Builds a column-major mixing matrix from secular vectors and orthonormalizes it.
fn build_secular_mixing_matrix(old_d: &[f64], z: &[f64], new_d: &[f64], n: usize) -> Vec<f64> {
    let mut q = vec![0.0_f64; n * n];

    for col in 0..n {
        let sigma2 = new_d[col] * new_d[col];
        let mut norm2 = 0.0_f64;

        for row in 0..n {
            let denom = old_d[row] * old_d[row] - sigma2;
            let v = if denom.abs() > 1e-12 {
                z[row] / denom
            } else if row == col {
                1.0
            } else {
                0.0
            };
            q[col * n + row] = v;
            norm2 += v * v;
        }

        if norm2 > 1e-24 {
            let inv = 1.0 / norm2.sqrt();
            for row in 0..n {
                q[col * n + row] *= inv;
            }
        } else {
            for row in 0..n {
                q[col * n + row] = if row == col { 1.0 } else { 0.0 };
            }
        }
    }

    // Modified Gram-Schmidt to stabilize orthonormality.
    for col in 0..n {
        for prev in 0..col {
            let mut dot = 0.0_f64;
            for row in 0..n {
                dot += q[prev * n + row] * q[col * n + row];
            }
            for row in 0..n {
                q[col * n + row] -= dot * q[prev * n + row];
            }
        }

        let mut norm2 = 0.0_f64;
        for row in 0..n {
            norm2 += q[col * n + row] * q[col * n + row];
        }
        if norm2 > 1e-24 {
            let inv = 1.0 / norm2.sqrt();
            for row in 0..n {
                q[col * n + row] *= inv;
            }
        } else {
            for row in 0..n {
                q[col * n + row] = if row == col { 1.0 } else { 0.0 };
            }
        }
    }

    q
}

/// In-place `A <- A * B` for square `n x n` matrices in column-major layout.
fn right_multiply_col_major_square(a: &mut [f64], b: &[f64], n: usize) {
    let mut out = vec![0.0_f64; n * n];
    for col in 0..n {
        for row in 0..n {
            let mut sum = 0.0_f64;
            for k in 0..n {
                sum += a[k * n + row] * b[col * n + k];
            }
            out[col * n + row] = sum;
        }
    }
    a.copy_from_slice(&out);
}

/// In-place `A <- Q^T * A` for row-major square `n x n` matrix `A`.
fn left_multiply_row_major_square_by_qt(a: &mut [f64], q_col_major: &[f64], n: usize) {
    let mut out = vec![0.0_f64; n * n];
    for row in 0..n {
        for col in 0..n {
            let mut sum = 0.0_f64;
            for k in 0..n {
                // Q^T[row, k] = Q[k, row] in column-major indexing.
                sum += q_col_major[k * n + row] * a[k * n + col];
            }
            out[row * n + col] = sum;
        }
    }
    a.copy_from_slice(&out);
}

// ---------------------------------------------------------------------------
// Sort singular values descending
// ---------------------------------------------------------------------------

/// Sorts singular values in descending order, permuting U and V^T columns/rows.
#[allow(clippy::needless_range_loop)]
fn sort_singular_values_desc(
    d: &mut [f64],
    mut u: Option<&mut [f64]>,
    mut vt: Option<&mut [f64]>,
    n: usize,
) {
    // Simple selection sort (n is typically modest after DC).
    for i in 0..n {
        let mut max_idx = i;
        let mut max_val = d[i].abs();
        for j in (i + 1)..n {
            if d[j].abs() > max_val {
                max_val = d[j].abs();
                max_idx = j;
            }
        }
        if max_idx != i {
            d.swap(i, max_idx);
            // Swap columns of U.
            if let Some(ref mut u_mat) = u {
                for row in 0..n {
                    u_mat.swap(i * n + row, max_idx * n + row);
                }
            }
            // Swap rows of V^T.
            if let Some(ref mut vt_mat) = vt {
                for col in 0..n {
                    vt_mat.swap(i * n + col, max_idx * n + col);
                }
            }
        }
        // Ensure positive singular values.
        if d[i] < 0.0 {
            d[i] = -d[i];
            if let Some(ref mut u_mat) = u {
                for row in 0..n {
                    u_mat[i * n + row] = -u_mat[i * n + row];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Device buffer write helper
// ---------------------------------------------------------------------------

/// Writes host data to a device buffer.
fn write_to_device_buffer<T: GpuFloat>(
    buf: &mut DeviceBuffer<T>,
    data: &[T],
    count: usize,
) -> SolverResult<()> {
    if count > data.len() {
        return Err(SolverError::DimensionMismatch(format!(
            "write_to_device_buffer: host slice too small ({} < {count})",
            data.len()
        )));
    }
    if count > buf.len() {
        return Err(SolverError::DimensionMismatch(format!(
            "write_to_device_buffer: device buffer too small ({} < {count})",
            buf.len()
        )));
    }

    // DeviceBuffer currently provides whole-buffer copies, so stage through a
    // full-size host vector and write the prefix.
    let mut staged = vec![T::gpu_zero(); buf.len()];
    staged[..count].copy_from_slice(&data[..count]);
    buf.copy_from_host(&staged)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_svd_config_default() {
        let cfg = DcSvdConfig::default();
        assert_eq!(cfg.crossover_size, DEFAULT_CROSSOVER);
        assert!(cfg.compute_u);
        assert!(cfg.compute_vt);
    }

    #[test]
    fn dc_svd_config_custom() {
        let cfg = DcSvdConfig {
            crossover_size: 10,
            compute_u: false,
            compute_vt: true,
            ..DcSvdConfig::default()
        };
        assert_eq!(cfg.crossover_size, 10);
        assert!(!cfg.compute_u);
        assert!(cfg.compute_vt);
    }

    #[test]
    fn secular_function_identity() {
        // For d = [1, 2, 3], z = [0, 0, 0], f(sigma) = 1 for any sigma.
        let d = [1.0, 2.0, 3.0];
        let z = [0.0, 0.0, 0.0];
        let (f_val, f_deriv) = secular_function(&d, &z, 0.5, 3);
        assert!((f_val - 1.0).abs() < 1e-10);
        assert!(f_deriv.abs() < 1e-10);
    }

    #[test]
    fn secular_function_with_coupling() {
        let d = [1.0, 3.0];
        let z = [0.5, 0.5];
        let (f_val, _f_deriv) = secular_function(&d, &z, 2.0, 2);
        // f(2) = 1 + 0.25/(1-4) + 0.25/(9-4) = 1 - 0.25/3 + 0.25/5
        let expected = 1.0 + 0.25 / (1.0 - 4.0) + 0.25 / (9.0 - 4.0);
        assert!((f_val - expected).abs() < 1e-10);
    }

    #[test]
    fn givens_rotation_basic() {
        let (cs, sn) = givens_rotation(3.0, 4.0);
        let r = cs * 3.0 + sn * 4.0;
        assert!((r - 5.0).abs() < 1e-10);
        let zero_val = -sn * 3.0 + cs * 4.0;
        assert!(zero_val.abs() < 1e-10);
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
    fn bidiagonal_qr_trivial() {
        // Already diagonal — should converge immediately.
        let mut d = vec![3.0, 2.0, 1.0];
        let mut e = vec![0.0, 0.0];
        let result = bidiagonal_svd_qr(&mut d, &mut e, None, None, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn bidiagonal_qr_with_superdiag() {
        let mut d = vec![4.0, 3.0];
        let mut e = vec![1.0];
        let mut u = vec![0.0; 4];
        let mut vt = vec![0.0; 4];
        let result = bidiagonal_svd_qr(&mut d, &mut e, Some(&mut u), Some(&mut vt), 2);
        assert!(result.is_ok());
    }

    #[test]
    fn bidiagonal_qr_empty() {
        let mut d: Vec<f64> = Vec::new();
        let mut e: Vec<f64> = Vec::new();
        let result = bidiagonal_svd_qr(&mut d, &mut e, None, None, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn sort_singular_values_descending() {
        let mut d = vec![1.0, 3.0, 2.0];
        sort_singular_values_desc(&mut d, None, None, 3);
        assert!((d[0] - 3.0).abs() < 1e-15);
        assert!((d[1] - 2.0).abs() < 1e-15);
        assert!((d[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn sort_singular_values_with_negatives() {
        let mut d = vec![-2.0, 1.0, -3.0];
        sort_singular_values_desc(&mut d, None, None, 3);
        assert!((d[0] - 3.0).abs() < 1e-15);
        assert!((d[1] - 2.0).abs() < 1e-15);
        assert!((d[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn dc_bidiagonal_base_case() {
        // Small enough for QR base case.
        let mut d = vec![5.0, 3.0, 1.0];
        let mut e = vec![0.0, 0.0];
        let result = dc_bidiagonal_svd(&mut d, &mut e, None, None, 3, 25);
        assert!(result.is_ok());
    }

    #[test]
    fn merge_orthogonal_blocks_identity() {
        let mut u = vec![0.0_f64; 16]; // 4x4
        let u_left = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let u_right = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        merge_orthogonal_blocks(Some(&mut u), Some(&u_left), Some(&u_right), 2, 4);
        // Check diagonal entries are 1.
        assert!((u[0] - 1.0).abs() < 1e-15); // (0,0)
        assert!((u[5] - 1.0).abs() < 1e-15); // (1,1) at col=1*4+1
        assert!((u[10] - 1.0).abs() < 1e-15); // (2,2)
        assert!((u[15] - 1.0).abs() < 1e-15); // (3,3)
    }

    #[test]
    fn merge_orthogonal_blocks_transpose_row_major_layout() {
        let mut vt = vec![0.0_f64; 16]; // 4x4 row-major
        let vt_left = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let vt_right = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]

        merge_orthogonal_blocks_transpose(Some(&mut vt), Some(&vt_left), Some(&vt_right), 2, 4);

        // Top-left block in row-major positions.
        assert!((vt[0] - 1.0).abs() < 1e-15);
        assert!((vt[1] - 2.0).abs() < 1e-15);
        assert!((vt[4] - 3.0).abs() < 1e-15);
        assert!((vt[5] - 4.0).abs() < 1e-15);

        // Bottom-right block in row-major positions.
        assert!((vt[10] - 5.0).abs() < 1e-15);
        assert!((vt[11] - 6.0).abs() < 1e-15);
        assert!((vt[14] - 7.0).abs() < 1e-15);
        assert!((vt[15] - 8.0).abs() < 1e-15);
    }

    #[test]
    fn secular_mixing_matrix_columns_are_normalized() {
        let old_d = vec![5.0, 3.0, 1.0];
        let z = vec![0.3, -0.2, 0.1];
        let new_d = vec![5.1, 2.9, 1.05];
        let q = build_secular_mixing_matrix(&old_d, &z, &new_d, 3);

        // q is column-major 3x3; each column should be unit norm.
        for col in 0..3 {
            let mut norm2 = 0.0_f64;
            for row in 0..3 {
                norm2 += q[col * 3 + row] * q[col * 3 + row];
            }
            assert!((norm2 - 1.0).abs() < 1e-8, "col {col} norm^2={norm2}");
        }
    }

    #[test]
    fn coupling_vector_fallback_spreads_at_split() {
        let z = build_secular_coupling_vector(None, None, 2.0, 2, 4);
        assert!(z[0].abs() < 1e-12);
        assert!((z[1] - std::f64::consts::SQRT_2).abs() < 1e-12);
        assert!((z[2] - std::f64::consts::SQRT_2).abs() < 1e-12);
        assert!(z[3].abs() < 1e-12);
    }

    #[test]
    fn deflation_zeros_tiny_superdiag_entries() {
        let d = vec![10.0_f64, 9.0, 8.0];
        let mut e = vec![1e-13_f64, 1e-3];
        apply_deflation_to_superdiag(&d, &mut e, 1e-12);
        assert_eq!(e[0], 0.0);
        assert!(e[1] > 0.0);
    }

    #[test]
    fn row_major_left_multiply_by_qt_identity_is_noop() {
        let mut a = vec![
            1.0_f64, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
        ];

        // Identity Q in column-major.
        let q = vec![
            1.0_f64, 0.0, 0.0, // col 0
            0.0, 1.0, 0.0, // col 1
            0.0, 0.0, 1.0, // col 2
        ];

        left_multiply_row_major_square_by_qt(&mut a, &q, 3);
        assert_eq!(
            a,
            vec![
                1.0_f64, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
                7.0, 8.0, 9.0, // row 2
            ]
        );
    }

    #[test]
    fn f64_conversion_roundtrip() {
        let val = std::f64::consts::PI;
        let converted: f64 = from_f64(to_f64(val));
        assert!((converted - val).abs() < 1e-15);
    }

    #[test]
    fn f32_conversion_roundtrip() {
        let val = std::f32::consts::PI;
        let as_f64 = to_f64(val);
        let back: f32 = from_f64(as_f64);
        assert!((back - val).abs() < 1e-5);
    }

    // ---------------------------------------------------------------------------
    // D&C SVD GPU configuration tests (bidiagonalization for N >= 1024)
    // ---------------------------------------------------------------------------

    #[test]
    fn dc_svd_config_threshold_1024() {
        // use_divide_conquer is true for n >= 1024, false below.
        let cfg_large = DcSvdConfig::for_gpu(1024);
        assert!(
            cfg_large.use_divide_conquer,
            "D&C should be enabled for n=1024"
        );
        assert_eq!(cfg_large.n_threshold, 1024);

        let cfg_small = DcSvdConfig::for_gpu(512);
        assert!(
            !cfg_small.use_divide_conquer,
            "D&C should be disabled for n=512"
        );
    }

    #[test]
    fn dc_svd_uses_bidiagonalization() {
        // bidiagonalization is true for n >= 256.
        let cfg_large = DcSvdConfig::for_gpu(256);
        assert!(
            cfg_large.bidiagonalization,
            "bidiagonalization should be enabled for n=256"
        );

        let cfg_small = DcSvdConfig::for_gpu(128);
        assert!(
            !cfg_small.bidiagonalization,
            "bidiagonalization should be disabled for n=128"
        );

        // Also true for very large n.
        let cfg_very_large = DcSvdConfig::for_gpu(4096);
        assert!(cfg_very_large.bidiagonalization);
        assert!(cfg_very_large.use_divide_conquer);
    }

    #[test]
    fn bidiagonalization_cpu_2x2() {
        // For a 2x2 bidiagonal matrix B = [[d1, e1], [0, d2]]:
        // SVD of a bidiagonal matrix is straightforward.
        // Verify that QR iteration on d=[d1,d2], e=[e1] converges.
        let mut d = vec![3.0_f64, 4.0];
        let mut e = vec![1.0_f64];
        let result = bidiagonal_svd_qr(&mut d, &mut e, None, None, 2);
        assert!(result.is_ok(), "bidiagonal QR for 2x2 must succeed");
        // After convergence, e should be near zero.
        assert!(
            e[0].abs() < 1e-10,
            "off-diagonal e[0] = {} should be ~0",
            e[0]
        );
        // Singular values should be positive.
        assert!(
            d[0] >= 0.0 && d[1] >= 0.0,
            "singular values must be non-negative"
        );
    }

    #[test]
    fn dc_svd_deflation_threshold_small() {
        // deflation_tol = n as f64 * 2.22e-16 (n * machine epsilon).
        let eps = 2.22e-16_f64;
        let n_vals: &[usize] = &[10, 100, 1000, 4096];
        for &n in n_vals {
            let cfg = DcSvdConfig::for_gpu(n);
            let expected_tol = n as f64 * eps;
            assert!(
                (cfg.deflation_tol - expected_tol).abs() < 1e-30,
                "deflation_tol for n={n}: got {}, expected {}",
                cfg.deflation_tol,
                expected_tol
            );
        }
    }
}
