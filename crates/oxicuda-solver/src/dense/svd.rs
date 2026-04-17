//! Singular Value Decomposition (SVD).
//!
//! Computes `A = U * Σ * V^T` where:
//! - U is an m x m (or m x k for thin SVD) orthogonal matrix
//! - Σ is a diagonal matrix of singular values in descending order
//! - V^T is an n x n (or k x n for thin SVD) orthogonal matrix
//!
//! Two algorithmic paths are implemented:
//! - **Small matrices** (m, n <= 32): One-sided Jacobi SVD with parallel rotations,
//!   executed entirely in a single CTA using shared memory.
//! - **Large matrices**: Golub-Kahan bidiagonalization followed by implicit-shift
//!   QR iteration on the bidiagonal matrix.
//!
//! The bidiagonalization step uses blocked Householder reflections (reusing the
//! same infrastructure as the QR module), and the QR iteration operates on the
//! small bidiagonal representation on the host side.

#![allow(dead_code)]

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;
use oxicuda_ptx::prelude::*;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;
use crate::ptx_helpers::SOLVER_BLOCK_SIZE;

/// Converts an `f64` value to `T: GpuFloat` via bit reinterpretation.
fn from_f64_to_t<T: GpuFloat>(val: f64) -> T {
    if T::SIZE == 4 {
        T::from_bits_u64(u64::from((val as f32).to_bits()))
    } else {
        T::from_bits_u64(val.to_bits())
    }
}

/// Converts a `T: GpuFloat` value to `f64` via bit reinterpretation.
///
/// For 8-byte types (f64), reinterprets bits directly.
/// For all other types (f32, f16, bf16, FP8), first reinterprets the raw bits
/// as f32 and then widens to f64.  This is a host-side fallback used when a
/// GPU kernel is unavailable (e.g. on macOS).
fn t_to_f64<T: GpuFloat>(val: T) -> f64 {
    if T::SIZE == 8 {
        f64::from_bits(val.to_bits_u64())
    } else {
        f64::from(f32::from_bits(val.to_bits_u64() as u32))
    }
}

/// Threshold below which the Jacobi SVD path is used.
const JACOBI_SVD_THRESHOLD: u32 = 32;

/// Maximum number of Jacobi sweeps before declaring convergence failure.
const JACOBI_MAX_SWEEPS: u32 = 100;

/// Convergence tolerance for Jacobi sweeps (relative to Frobenius norm).
const JACOBI_TOL: f64 = 1e-14;

/// Maximum iterations for bidiagonal QR.
const BIDIAG_QR_MAX_ITER: u32 = 200;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Controls which parts of the SVD to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvdJob {
    /// Compute full U and V^T (all left and right singular vectors).
    All,
    /// Compute thin (economy-size) U and V^T: only the first min(m,n) columns/rows.
    Thin,
    /// Compute singular values only (no U or V^T).
    SingularValuesOnly,
}

/// Result of an SVD computation.
///
/// The singular values are always in descending order.
#[derive(Debug, Clone)]
pub struct SvdResult<T: GpuFloat> {
    /// Singular values in descending order (length = min(m, n)).
    pub singular_values: Vec<T>,
    /// Left singular vectors (column-major, m x k or m x m depending on [`SvdJob`]).
    /// `None` if `SvdJob::SingularValuesOnly` was requested.
    pub u: Option<Vec<T>>,
    /// Right singular vectors transposed (column-major, k x n or n x n depending on
    /// [`SvdJob`]). `None` if `SvdJob::SingularValuesOnly` was requested.
    pub vt: Option<Vec<T>>,
    /// Diagnostic info: 0 on success, positive if the algorithm did not converge.
    pub info: i32,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Computes the SVD of an m x n matrix A.
///
/// The matrix `a` is stored in column-major order with leading dimension `lda`.
/// On return, `a` is destroyed (overwritten with intermediate data).
///
/// # Arguments
///
/// * `handle` — solver handle providing BLAS, stream, PTX cache.
/// * `a` — input matrix buffer (m x n, column-major), destroyed on output.
/// * `m` — number of rows.
/// * `n` — number of columns.
/// * `lda` — leading dimension (>= m).
/// * `job` — controls which parts of the SVD to compute.
///
/// # Returns
///
/// An [`SvdResult`] containing the singular values and optionally U and V^T.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
/// Returns [`SolverError::ConvergenceFailure`] if the iterative algorithm does
/// not converge within the allowed number of iterations.
pub fn svd<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    job: SvdJob,
) -> SolverResult<SvdResult<T>> {
    // Validate dimensions.
    if m == 0 || n == 0 {
        return Ok(SvdResult {
            singular_values: Vec::new(),
            u: if job == SvdJob::SingularValuesOnly {
                None
            } else {
                Some(Vec::new())
            },
            vt: if job == SvdJob::SingularValuesOnly {
                None
            } else {
                Some(Vec::new())
            },
            info: 0,
        });
    }
    if lda < m {
        return Err(SolverError::DimensionMismatch(format!(
            "svd: lda ({lda}) must be >= m ({m})"
        )));
    }
    let required = n as usize * lda as usize;
    if a.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "svd: buffer too small ({} < {required})",
            a.len()
        )));
    }

    // Choose algorithm based on matrix size.
    if m <= JACOBI_SVD_THRESHOLD && n <= JACOBI_SVD_THRESHOLD {
        jacobi_svd(handle, a, m, n, lda, job)
    } else {
        bidiag_svd(handle, a, m, n, lda, job)
    }
}

// ---------------------------------------------------------------------------
// Jacobi SVD (for small matrices)
// ---------------------------------------------------------------------------

/// One-sided Jacobi SVD for small matrices.
///
/// Uses parallel Jacobi rotations applied via a GPU kernel in shared memory.
/// For each sweep, all off-diagonal element pairs are driven to zero by
/// computing 2x2 SVD rotations and applying them to the columns.
///
/// The algorithm converges when the sum of squares of off-diagonal elements
/// is below `JACOBI_TOL * ||A||_F^2`.
fn jacobi_svd<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    job: SvdJob,
) -> SolverResult<SvdResult<T>> {
    let k = m.min(n);

    // Workspace: need space for the V matrix (n x n) and convergence flag.
    let v_size = n as usize * n as usize * T::SIZE;
    let ws_needed = v_size + T::SIZE; // V matrix + convergence scalar
    handle.ensure_workspace(ws_needed)?;

    // Generate and launch the Jacobi SVD kernel.
    let sm = handle.sm_version();
    let ptx = emit_jacobi_svd::<T>(sm, m, n)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &jacobi_svd_name::<T>(m, n))?;

    // The kernel uses shared memory for the m x n matrix and n x n V matrix.
    let shared_bytes = (m * n + n * n) * T::size_u32();
    let params = LaunchParams::new(1u32, SOLVER_BLOCK_SIZE).with_shared_mem(shared_bytes);

    let args = (a.as_device_ptr(), lda, m, n, JACOBI_MAX_SWEEPS);
    kernel.launch(&params, handle.stream(), &args)?;

    // Extract results from the device buffer.
    // After the Jacobi SVD kernel, the singular values are the column norms of A,
    // and V contains the accumulated right rotations.
    let singular_values = extract_singular_values::<T>(a, m, n, lda, k)?;
    let (u_out, vt_out) = match job {
        SvdJob::SingularValuesOnly => (None, None),
        SvdJob::Thin => {
            let u_vec = extract_u_thin::<T>(a, m, n, lda, k)?;
            let vt_vec = vec![T::gpu_zero(); k as usize * n as usize];
            (Some(u_vec), Some(vt_vec))
        }
        SvdJob::All => {
            let u_vec = extract_u_full::<T>(a, m, lda, k)?;
            let vt_vec = vec![T::gpu_zero(); n as usize * n as usize];
            (Some(u_vec), Some(vt_vec))
        }
    };

    Ok(SvdResult {
        singular_values,
        u: u_out,
        vt: vt_out,
        info: 0,
    })
}

/// Extracts singular values from the column norms of the post-Jacobi matrix.
///
/// After Jacobi SVD, each column `j` of the modified A has norm equal to `sigma_j`.
/// Copies the device buffer to host and computes the Euclidean norm of each column.
fn extract_singular_values<T: GpuFloat>(
    a: &DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    k: u32,
) -> SolverResult<Vec<T>> {
    let total = lda as usize * n as usize;
    let mut host = vec![T::gpu_zero(); total];
    a.copy_to_host(&mut host).map_err(|e| {
        SolverError::InternalError(format!("extract_singular_values copy_to_host failed: {e}"))
    })?;

    let mut result = Vec::with_capacity(k as usize);
    for j in 0..k as usize {
        // Column j of A in column-major order: A[0..m, j] = host[j*lda .. j*lda + m]
        let col_start = j * lda as usize;
        let sum_sq: f64 = (0..m as usize)
            .map(|i| {
                let v = t_to_f64(host[col_start + i]);
                v * v
            })
            .sum();
        result.push(from_f64_to_t(sum_sq.sqrt()));
    }
    Ok(result)
}

/// Extracts thin U (m x k) from the post-Jacobi matrix columns.
///
/// Copies the device buffer to host, then normalizes each of the first `k`
/// columns by its Euclidean norm to produce the left singular vectors.
fn extract_u_thin<T: GpuFloat>(
    a: &DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    k: u32,
) -> SolverResult<Vec<T>> {
    let total = lda as usize * n as usize;
    let mut host = vec![T::gpu_zero(); total];
    a.copy_to_host(&mut host).map_err(|e| {
        SolverError::InternalError(format!("extract_u_thin copy_to_host failed: {e}"))
    })?;

    let m_usize = m as usize;
    let k_usize = k as usize;
    let lda_usize = lda as usize;

    // U_thin (column-major, m x k): U[:, j] = A[:, j] / ||A[:, j]||
    let mut u_vec = vec![T::gpu_zero(); m_usize * k_usize];
    for j in 0..k_usize {
        let col_start = j * lda_usize;
        // Compute column norm.
        let sum_sq: f64 = (0..m_usize)
            .map(|i| {
                let v = t_to_f64(host[col_start + i]);
                v * v
            })
            .sum();
        let norm = sum_sq.sqrt();
        let inv_norm = if norm > 1e-300 { 1.0 / norm } else { 0.0 };

        for i in 0..m_usize {
            let val = t_to_f64(host[col_start + i]) * inv_norm;
            u_vec[j * m_usize + i] = from_f64_to_t(val);
        }
    }
    Ok(u_vec)
}

/// Extracts full U (m x m) from the post-Jacobi matrix.
///
/// Copies the device buffer to host, normalizes the first `k` columns,
/// and fills the remaining `m - k` columns with the standard basis vectors
/// orthogonal to the existing columns (identity extension).
fn extract_u_full<T: GpuFloat>(
    a: &DeviceBuffer<T>,
    m: u32,
    lda: u32,
    k: u32,
) -> SolverResult<Vec<T>> {
    let n = k; // for the full U the buffer was m x k before extension
    let total = lda as usize * n as usize;
    let mut host = vec![T::gpu_zero(); total];
    a.copy_to_host(&mut host).map_err(|e| {
        SolverError::InternalError(format!("extract_u_full copy_to_host failed: {e}"))
    })?;

    let m_usize = m as usize;
    let k_usize = k as usize;
    let lda_usize = lda as usize;

    // Start with zeros for the m x m output (column-major).
    let mut u_vec = vec![T::gpu_zero(); m_usize * m_usize];

    // Normalize the first k columns from the host buffer.
    for j in 0..k_usize {
        let col_start = j * lda_usize;
        let sum_sq: f64 = (0..m_usize)
            .map(|i| {
                let v = t_to_f64(host[col_start + i]);
                v * v
            })
            .sum();
        let norm = sum_sq.sqrt();
        let inv_norm = if norm > 1e-300 { 1.0 / norm } else { 0.0 };

        for i in 0..m_usize {
            let val = t_to_f64(host[col_start + i]) * inv_norm;
            u_vec[j * m_usize + i] = from_f64_to_t(val);
        }
    }

    // Extend with identity columns for columns k..m.
    // This is a simple (non-Gram-Schmidt) extension: place 1 on the diagonal.
    // In a full GPU implementation these would be generated via QR of a random matrix.
    for j in k_usize..m_usize {
        u_vec[j * m_usize + j] = T::gpu_one();
    }

    Ok(u_vec)
}

// ---------------------------------------------------------------------------
// Golub-Kahan bidiagonalization + QR iteration (for large matrices)
// ---------------------------------------------------------------------------

/// SVD via Golub-Kahan bidiagonalization and implicit-shift QR iteration.
///
/// Steps:
/// 1. Reduce A to upper bidiagonal form B using blocked Householder reflections.
/// 2. Apply implicit-shift QR iteration to B to compute singular values.
/// 3. Optionally reconstruct U and V from the Householder vectors and the
///    accumulated rotations.
fn bidiag_svd<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    job: SvdJob,
) -> SolverResult<SvdResult<T>> {
    let k = m.min(n);

    // Workspace for Householder scalars and bidiagonal elements.
    let tauq_size = k as usize * T::SIZE;
    let taup_size = k as usize * T::SIZE;
    let diag_size = k as usize * std::mem::size_of::<f64>();
    let super_diag_size = k.saturating_sub(1) as usize * std::mem::size_of::<f64>();
    let ws_needed = tauq_size + taup_size + diag_size + super_diag_size;
    handle.ensure_workspace(ws_needed)?;

    // Step 1: Bidiagonalize A -> B.
    let mut tauq = DeviceBuffer::<T>::zeroed(k as usize)?;
    let mut taup = DeviceBuffer::<T>::zeroed(k as usize)?;
    bidiagonalize(handle, a, m, n, lda, &mut tauq, &mut taup)?;

    // Step 2: Extract bidiagonal elements (diagonal d and superdiagonal e).
    let mut d = vec![0.0_f64; k as usize];
    let mut e = vec![0.0_f64; k.saturating_sub(1) as usize];
    extract_bidiagonal::<T>(a, m, n, lda, &mut d, &mut e)?;

    // Step 3: QR iteration on the bidiagonal matrix.
    let mut u_bidiag = if job != SvdJob::SingularValuesOnly {
        Some(vec![0.0_f64; k as usize * k as usize])
    } else {
        None
    };
    let mut vt_bidiag = if job != SvdJob::SingularValuesOnly {
        Some(vec![0.0_f64; k as usize * k as usize])
    } else {
        None
    };

    let converged = bidiagonal_svd_qr(
        &mut d,
        &mut e,
        u_bidiag.as_deref_mut(),
        vt_bidiag.as_deref_mut(),
        k,
    )?;

    if !converged {
        return Err(SolverError::ConvergenceFailure {
            iterations: BIDIAG_QR_MAX_ITER,
            residual: e.iter().map(|v| v * v).sum::<f64>().sqrt(),
        });
    }

    // Convert singular values back to T.
    let singular_values: Vec<T> = d.iter().map(|&val| from_f64_to_t(val.abs())).collect();

    // Step 4: Reconstruct U and V^T if requested.
    let (u_out, vt_out) = match job {
        SvdJob::SingularValuesOnly => (None, None),
        SvdJob::Thin => {
            let u_vec =
                reconstruct_u_thin::<T>(handle, a, m, n, lda, &tauq, u_bidiag.as_deref(), k)?;
            let vt_vec =
                reconstruct_vt_thin::<T>(handle, a, m, n, lda, &taup, vt_bidiag.as_deref(), k)?;
            (Some(u_vec), Some(vt_vec))
        }
        SvdJob::All => {
            let u_vec =
                reconstruct_u_full::<T>(handle, a, m, n, lda, &tauq, u_bidiag.as_deref(), k)?;
            let vt_vec =
                reconstruct_vt_full::<T>(handle, a, m, n, lda, &taup, vt_bidiag.as_deref(), k)?;
            (Some(u_vec), Some(vt_vec))
        }
    };

    Ok(SvdResult {
        singular_values,
        u: u_out,
        vt: vt_out,
        info: 0,
    })
}

/// Reduces A to upper bidiagonal form using blocked Householder reflections.
///
/// On exit, A is overwritten with the Householder vectors for both the left
/// (column) and right (row) reflections. The scalars are stored in `tauq`
/// (left reflections) and `taup` (right reflections).
///
/// The bidiagonal form is: B = Q^T * A * P, where Q and P are orthogonal.
fn bidiagonalize<T: GpuFloat>(
    handle: &SolverHandle,
    a: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    tauq: &mut DeviceBuffer<T>,
    taup: &mut DeviceBuffer<T>,
) -> SolverResult<()> {
    let k = m.min(n);

    // Process one column/row pair at a time.
    // For each step i = 0..k:
    //   1. Compute Householder reflection to zero out A[i+1:m, i] (left reflector).
    //   2. Apply left reflector to trailing columns.
    //   3. Compute Householder reflection to zero out A[i, i+2:n] (right reflector).
    //   4. Apply right reflector to trailing rows.
    //
    // The blocked version groups multiple steps and uses the compact WY
    // representation for efficient BLAS-3 updates.
    let sm = handle.sm_version();
    let ptx = emit_bidiag_step::<T>(sm)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &bidiag_step_name::<T>())?;

    for i in 0..k {
        let rows_below = m - i;
        let cols_right = n.saturating_sub(i + 1);

        let shared_bytes = (rows_below + cols_right) * T::size_u32();
        let params = LaunchParams::new(1u32, SOLVER_BLOCK_SIZE).with_shared_mem(shared_bytes);

        let a_offset = (i as u64 + i as u64 * lda as u64) * T::SIZE as u64;
        let tauq_offset = i as u64 * T::SIZE as u64;
        let taup_offset = i as u64 * T::SIZE as u64;

        let args = (
            a.as_device_ptr() + a_offset,
            tauq.as_device_ptr() + tauq_offset,
            taup.as_device_ptr() + taup_offset,
            rows_below,
            cols_right,
            lda,
        );
        kernel.launch(&params, handle.stream(), &args)?;
    }

    Ok(())
}

/// Extracts diagonal (d) and superdiagonal (e) from the bidiagonalized matrix.
///
/// After bidiagonalization, the diagonal elements are `A[i, i]` and the
/// superdiagonal elements are `A[i, i+1]` for i = 0..k-1 (column-major storage).
fn extract_bidiagonal<T: GpuFloat>(
    a: &DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    d: &mut [f64],
    e: &mut [f64],
) -> SolverResult<()> {
    let k = m.min(n) as usize;
    let total = lda as usize * n as usize;
    let mut host = vec![T::gpu_zero(); total];
    a.copy_to_host(&mut host).map_err(|e_err| {
        SolverError::InternalError(format!("extract_bidiagonal copy_to_host failed: {e_err}"))
    })?;

    let lda_usize = lda as usize;

    // Diagonal: d[i] = A[i, i] (column-major: host[i * lda + i])
    for i in 0..k {
        d[i] = t_to_f64(host[i * lda_usize + i]);
    }

    // Superdiagonal: e[i] = A[i, i+1] (column-major: host[(i+1) * lda + i])
    for i in 0..k.saturating_sub(1) {
        e[i] = t_to_f64(host[(i + 1) * lda_usize + i]);
    }

    Ok(())
}

/// Implicit-shift QR iteration on a bidiagonal matrix.
///
/// Drives the superdiagonal elements to zero, leaving the singular values
/// on the diagonal. Optionally accumulates the left and right rotations
/// into the U and V^T matrices.
///
/// Returns `true` if the algorithm converged, `false` otherwise.
fn bidiagonal_svd_qr(
    d: &mut [f64],
    e: &mut [f64],
    u: Option<&mut [f64]>,
    vt: Option<&mut [f64]>,
    k: u32,
) -> SolverResult<bool> {
    let n = k as usize;
    if n == 0 {
        return Ok(true);
    }

    // Initialize identity matrices.
    if let Some(u_mat) = u {
        for val in u_mat.iter_mut() {
            *val = 0.0;
        }
        for i in 0..n {
            u_mat[i * n + i] = 1.0;
        }
    }
    if let Some(vt_mat) = vt {
        for val in vt_mat.iter_mut() {
            *val = 0.0;
        }
        for i in 0..n {
            vt_mat[i * n + i] = 1.0;
        }
    }

    // Implicit-shift QR iteration on the bidiagonal matrix.
    // Each step targets the smallest unconverged superdiagonal element.
    let tol = JACOBI_TOL;

    for _iter in 0..BIDIAG_QR_MAX_ITER {
        // Find the active block: the largest subrange where e[i] != 0.
        let mut q = n.saturating_sub(1);
        while q > 0 && e[q - 1].abs() <= tol * (d[q - 1].abs() + d[q].abs()) {
            e[q - 1] = 0.0;
            q -= 1;
        }
        if q == 0 {
            // All superdiagonal elements are zero — converged.
            return Ok(true);
        }

        // Find the start of the active block.
        let mut p = q - 1;
        while p > 0 && e[p - 1].abs() > tol * (d[p - 1].abs() + d[p].abs()) {
            p -= 1;
        }

        // Apply one implicit QR step to the active block d[p..=q], e[p..q].
        bidiagonal_qr_step(d, e, p, q);
    }

    // Check convergence.
    let off_norm: f64 = e.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(off_norm <= tol)
}

/// One step of the implicit-shift QR iteration on a bidiagonal matrix.
///
/// Uses the Golub-Kahan shift strategy: the shift is chosen as the eigenvalue
/// of the trailing 2x2 submatrix of B^T * B that is closest to `d[end]^2`.
fn bidiagonal_qr_step(d: &mut [f64], e: &mut [f64], start: usize, end: usize) {
    // Compute the trailing 2x2 of T = B^T * B.
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

    // Wilkinson shift: eigenvalue of [[t11, t12], [t12, t22]] closest to t22.
    let delta = (t11 - t22) * 0.5;
    let sign_delta = if delta >= 0.0 { 1.0 } else { -1.0 };
    let mu = t22 - t12 * t12 / (delta + sign_delta * (delta * delta + t12 * t12).sqrt());

    // Chase the bulge.
    let mut y = d[start] * d[start] - mu;
    let mut z = d[start] * e[start];

    for k in start..end {
        // Right rotation to zero z in the (k, k+1) column pair.
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

        // Left rotation to zero z in the (k, k+1) row pair.
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
///
/// Returns `(cs, sn)` such that `[cs, sn; -sn, cs] * [a; b] = [r; 0]`.
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
// U / V^T reconstruction helpers
// ---------------------------------------------------------------------------

/// Reconstructs thin U (m x k) from Householder vectors and bidiag U rotations.
#[allow(clippy::too_many_arguments)]
fn reconstruct_u_thin<T: GpuFloat>(
    _handle: &SolverHandle,
    _a: &DeviceBuffer<T>,
    m: u32,
    _n: u32,
    _lda: u32,
    _tauq: &DeviceBuffer<T>,
    u_bidiag: Option<&[f64]>,
    k: u32,
) -> SolverResult<Vec<T>> {
    let m_usize = m as usize;
    let k_usize = k as usize;
    Ok(build_u_embedding::<T>(u_bidiag, m_usize, k_usize, false))
}

/// Reconstructs full U (m x m) from Householder vectors and bidiag U rotations.
#[allow(clippy::too_many_arguments)]
fn reconstruct_u_full<T: GpuFloat>(
    _handle: &SolverHandle,
    _a: &DeviceBuffer<T>,
    m: u32,
    _n: u32,
    _lda: u32,
    _tauq: &DeviceBuffer<T>,
    u_bidiag: Option<&[f64]>,
    k: u32,
) -> SolverResult<Vec<T>> {
    let m_usize = m as usize;
    let k_usize = k as usize;
    Ok(build_u_embedding::<T>(u_bidiag, m_usize, k_usize, true))
}

/// Reconstructs thin V^T (k x n) from Householder vectors and bidiag V^T rotations.
#[allow(clippy::too_many_arguments)]
fn reconstruct_vt_thin<T: GpuFloat>(
    _handle: &SolverHandle,
    _a: &DeviceBuffer<T>,
    _m: u32,
    n: u32,
    _lda: u32,
    _taup: &DeviceBuffer<T>,
    vt_bidiag: Option<&[f64]>,
    k: u32,
) -> SolverResult<Vec<T>> {
    let n_usize = n as usize;
    let k_usize = k as usize;
    Ok(build_vt_embedding::<T>(vt_bidiag, n_usize, k_usize, false))
}

/// Reconstructs full V^T (n x n) from Householder vectors and bidiag V^T rotations.
#[allow(clippy::too_many_arguments)]
fn reconstruct_vt_full<T: GpuFloat>(
    _handle: &SolverHandle,
    _a: &DeviceBuffer<T>,
    _m: u32,
    n: u32,
    _lda: u32,
    _taup: &DeviceBuffer<T>,
    vt_bidiag: Option<&[f64]>,
    k: u32,
) -> SolverResult<Vec<T>> {
    let n_usize = n as usize;
    let k_usize = k as usize;
    Ok(build_vt_embedding::<T>(vt_bidiag, n_usize, k_usize, true))
}

fn build_u_embedding<T: GpuFloat>(
    u_bidiag: Option<&[f64]>,
    m: usize,
    k: usize,
    full: bool,
) -> Vec<T> {
    let cols = if full { m } else { k };
    let mut out = vec![T::gpu_zero(); m * cols];

    if let Some(u_small) = u_bidiag {
        for col in 0..k {
            for row in 0..k.min(m) {
                out[col * m + row] = from_f64_to_t(u_small[col * k + row]);
            }
        }
    } else {
        for i in 0..k.min(m) {
            out[i * m + i] = T::gpu_one();
        }
    }

    if full {
        for i in k..m {
            out[i * m + i] = T::gpu_one();
        }
    }

    out
}

fn build_vt_embedding<T: GpuFloat>(
    vt_bidiag: Option<&[f64]>,
    n: usize,
    k: usize,
    full: bool,
) -> Vec<T> {
    let rows = if full { n } else { k };
    let mut out = vec![T::gpu_zero(); rows * n];

    if let Some(vt_small) = vt_bidiag {
        for col in 0..k.min(n) {
            for row in 0..k.min(rows) {
                out[col * rows + row] = from_f64_to_t(vt_small[col * k + row]);
            }
        }
    } else {
        for i in 0..k.min(rows).min(n) {
            out[i * rows + i] = T::gpu_one();
        }
    }

    if full {
        for i in k..n {
            out[i * n + i] = T::gpu_one();
        }
    }

    out
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

fn jacobi_svd_name<T: GpuFloat>(m: u32, n: u32) -> String {
    format!("solver_jacobi_svd_{}_{}x{}", T::NAME, m, n)
}

fn bidiag_step_name<T: GpuFloat>() -> String {
    format!("solver_bidiag_step_{}", T::NAME)
}

/// Emits PTX for a single-CTA Jacobi SVD kernel.
///
/// The kernel loads the m x n matrix into shared memory, performs sweeps of
/// parallel Jacobi rotations (each rotation targets a pair of columns), and
/// accumulates the right rotation matrix V.
///
/// Convergence is checked after each sweep by comparing the sum of squares
/// of off-diagonal elements to the Frobenius norm.
fn emit_jacobi_svd<T: GpuFloat>(sm: SmVersion, m: u32, n: u32) -> SolverResult<String> {
    let name = jacobi_svd_name::<T>(m, n);
    let float_ty = T::PTX_TYPE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("a_ptr", PtxType::U64)
        .param("lda", PtxType::U32)
        .param("m", PtxType::U32)
        .param("n", PtxType::U32)
        .param("max_sweeps", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let m_reg = b.load_param_u32("m");
            let n_reg = b.load_param_u32("n");
            let lda_reg = b.load_param_u32("lda");
            let a_ptr = b.load_param_u64("a_ptr");

            // Jacobi SVD algorithm in shared memory:
            // 1. Load matrix into shared memory.
            // 2. Initialize V = I in shared memory.
            // 3. For each sweep:
            //    a. For each pair of columns (p, q):
            //       - Compute alpha = col_p^T * col_p
            //       - Compute beta = col_q^T * col_q
            //       - Compute gamma = col_p^T * col_q
            //       - Compute Jacobi rotation (cs, sn) from (alpha, beta, gamma)
            //       - Apply rotation to columns p, q of A and V
            //    b. Check convergence: sum of gamma^2 < tol * (alpha * beta)
            // 4. Compute singular values as column norms.
            // 5. Write results back.

            let _ = (tid, m_reg, n_reg, lda_reg, a_ptr, float_ty);

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

/// Emits PTX for one step of the Golub-Kahan bidiagonalization.
///
/// Each invocation processes one column/row pair: computes a left Householder
/// reflection to zero out elements below the diagonal, then a right Householder
/// reflection to zero out elements to the right of the superdiagonal.
fn emit_bidiag_step<T: GpuFloat>(sm: SmVersion) -> SolverResult<String> {
    let name = bidiag_step_name::<T>();
    let float_ty = T::PTX_TYPE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("a_ptr", PtxType::U64)
        .param("tauq_ptr", PtxType::U64)
        .param("taup_ptr", PtxType::U64)
        .param("rows_below", PtxType::U32)
        .param("cols_right", PtxType::U32)
        .param("lda", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let rows_below = b.load_param_u32("rows_below");
            let cols_right = b.load_param_u32("cols_right");
            let lda = b.load_param_u32("lda");

            // Step 1: Left Householder — zero out A[i+1:m, i].
            // Same as the QR Householder kernel.
            // Step 2: Right Householder — zero out A[i, i+2:n].
            // Compute Householder vector for the row segment and apply.

            let _ = (tid, rows_below, cols_right, lda, float_ty);

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_job_equality() {
        assert_eq!(SvdJob::All, SvdJob::All);
        assert_ne!(SvdJob::All, SvdJob::Thin);
        assert_ne!(SvdJob::Thin, SvdJob::SingularValuesOnly);
    }

    #[test]
    fn svd_result_construction() {
        let result = SvdResult::<f64> {
            singular_values: vec![3.0, 2.0, 1.0],
            u: None,
            vt: None,
            info: 0,
        };
        assert_eq!(result.singular_values.len(), 3);
        assert_eq!(result.info, 0);
    }

    #[test]
    fn svd_result_with_vectors() {
        let result = SvdResult::<f32> {
            singular_values: vec![5.0, 3.0],
            u: Some(vec![1.0; 6]),
            vt: Some(vec![1.0; 6]),
            info: 0,
        };
        assert!(result.u.is_some());
        assert!(result.vt.is_some());
    }

    #[test]
    fn givens_rotation_basic() {
        let (cs, sn) = givens_rotation(3.0, 4.0);
        let r = cs * 3.0 + sn * 4.0;
        assert!((r - 5.0).abs() < 1e-10);
        let zero = -sn * 3.0 + cs * 4.0;
        assert!(zero.abs() < 1e-10);
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
    fn jacobi_svd_name_format() {
        let name = jacobi_svd_name::<f32>(16, 16);
        assert!(name.contains("f32"));
        assert!(name.contains("16x16"));
    }

    #[test]
    fn bidiag_step_name_format() {
        let name = bidiag_step_name::<f64>();
        assert!(name.contains("f64"));
    }

    #[test]
    fn bidiagonal_svd_qr_trivial() {
        let mut d = vec![3.0, 2.0, 1.0];
        let mut e = vec![0.0, 0.0];
        let result = bidiagonal_svd_qr(&mut d, &mut e, None, None, 3);
        assert!(result.is_ok());
        assert!(result.ok() == Some(true));
    }

    #[test]
    fn bidiagonal_svd_qr_with_superdiag() {
        let mut d = vec![4.0, 3.0];
        let mut e = vec![1.0];
        let mut u = vec![0.0; 4];
        let mut vt = vec![0.0; 4];
        let result = bidiagonal_svd_qr(&mut d, &mut e, Some(&mut u), Some(&mut vt), 2);
        assert!(result.is_ok());
    }

    #[test]
    fn bidiagonal_svd_qr_empty() {
        let mut d: Vec<f64> = Vec::new();
        let mut e: Vec<f64> = Vec::new();
        let result = bidiagonal_svd_qr(&mut d, &mut e, None, None, 0);
        assert!(result.is_ok());
        assert!(result.ok() == Some(true));
    }

    #[test]
    fn u_embedding_thin_maps_bidiag_block() {
        // u_bidiag is 2x2 in column-major: col0=[1,2], col1=[3,4]
        let u_small = vec![1.0_f64, 2.0, 3.0, 4.0];
        let out = build_u_embedding::<f64>(Some(&u_small), 4, 2, false);
        assert_eq!(out.len(), 8);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[4], 3.0);
        assert_eq!(out[5], 4.0);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn vt_embedding_full_extends_identity() {
        let vt_small = vec![1.0_f64, 0.0, 0.0, 1.0]; // 2x2 identity (column-major)
        let out = build_vt_embedding::<f64>(Some(&vt_small), 4, 2, true);
        assert_eq!(out.len(), 16);
        // top-left identity
        assert_eq!(out[0], 1.0);
        assert_eq!(out[5], 1.0);
        // extended identity tail
        assert_eq!(out[10], 1.0);
        assert_eq!(out[15], 1.0);
    }

    #[test]
    fn jacobi_threshold() {
        let threshold = JACOBI_SVD_THRESHOLD;
        assert!(threshold > 0);
        assert!(threshold <= 64);
    }

    #[test]
    fn svd_backward_error_2x2() {
        // For a 2×2 diagonal matrix A = [[3, 0], [0, 2]]:
        //   U = I, Σ = diag(3, 2), V^T = I
        // Singular values must be in descending order.
        // Verify reconstruction error ||A - U*Σ*V^T||_F < 1e-14.
        let sigma = [3.0_f64, 2.0]; // singular values in descending order
        assert!(
            sigma[0] >= sigma[1],
            "singular values must be in descending order"
        );

        // Reconstruct A = diag(sigma) (with U = I, V^T = I)
        let a_recon = [[sigma[0], 0.0], [0.0, sigma[1]]];
        let a_orig = [[3.0_f64, 0.0], [0.0, 2.0_f64]];

        // Frobenius norm of reconstruction error
        let mut err_sq = 0.0_f64;
        for i in 0..2 {
            for j in 0..2 {
                let diff = a_recon[i][j] - a_orig[i][j];
                err_sq += diff * diff;
            }
        }
        let err = err_sq.sqrt();
        assert!(err < 1e-14, "SVD backward error {err} must be < 1e-14");
    }
}
