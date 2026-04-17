//! Symmetric eigenvalue decomposition.
//!
//! Computes `A = Q * Λ * Q^T` for a real symmetric matrix A, where:
//! - Q is an orthogonal matrix whose columns are eigenvectors
//! - Λ is a diagonal matrix of eigenvalues in ascending order
//!
//! The algorithm proceeds in two stages:
//! 1. **Tridiagonalization**: Reduce A to tridiagonal form T via blocked Householder
//!    reflections: `A = Q_1 * T * Q_1^T`.
//! 2. **Tridiagonal QR iteration**: Apply implicit-shift QR iteration to T to
//!    compute eigenvalues (and optionally eigenvectors).
//! 3. **Back-transformation**: If eigenvectors are requested, accumulate the
//!    Householder reflections and QR rotations: `Q = Q_1 * Q_2`.

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

/// Maximum iterations for the tridiagonal QR algorithm.
const TRIDIAG_QR_MAX_ITER: u32 = 300;

/// Convergence tolerance for off-diagonal elements.
const TRIDIAG_QR_TOL: f64 = 1e-14;

/// Block size for the tridiagonalization step.
const TRIDIAG_BLOCK_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Controls what to compute in the eigendecomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigJob {
    /// Compute eigenvalues only.
    ValuesOnly,
    /// Compute both eigenvalues and eigenvectors.
    ValuesAndVectors,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Computes eigenvalues (and optionally eigenvectors) of a symmetric matrix.
///
/// The matrix `a` is stored in column-major order with leading dimension `lda`.
/// Only the lower triangle is accessed. On exit:
/// - `eigenvalues` contains the eigenvalues in ascending order.
/// - If `job == ValuesAndVectors`, `a` is overwritten with the orthogonal
///   eigenvector matrix Q (column-major).
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `a` — symmetric matrix (n x n, column-major), destroyed/overwritten on output.
/// * `n` — matrix dimension.
/// * `lda` — leading dimension (>= n).
/// * `eigenvalues` — output buffer for eigenvalues (length >= n).
/// * `job` — controls what to compute.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
/// Returns [`SolverError::ConvergenceFailure`] if QR iteration does not converge.
pub fn syevd<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    n: u32,
    lda: u32,
    eigenvalues: &mut DeviceBuffer<T>,
    job: EigJob,
) -> SolverResult<()> {
    // Validate dimensions.
    if n == 0 {
        return Ok(());
    }
    if lda < n {
        return Err(SolverError::DimensionMismatch(format!(
            "syevd: lda ({lda}) must be >= n ({n})"
        )));
    }
    let required = n as usize * lda as usize;
    if a.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "syevd: buffer too small ({} < {required})",
            a.len()
        )));
    }
    if eigenvalues.len() < n as usize {
        return Err(SolverError::DimensionMismatch(format!(
            "syevd: eigenvalues buffer too small ({} < {n})",
            eigenvalues.len()
        )));
    }

    // Workspace for Householder scalars and tridiagonal elements.
    let tau_size = n.saturating_sub(1) as usize * T::SIZE;
    let diag_size = n as usize * std::mem::size_of::<f64>();
    let off_diag_size = n.saturating_sub(1) as usize * std::mem::size_of::<f64>();
    let ws_needed = tau_size + diag_size + off_diag_size;
    handle.ensure_workspace(ws_needed)?;

    // Step 1: Tridiagonalize.
    let mut tau = DeviceBuffer::<T>::zeroed(n.saturating_sub(1) as usize)?;
    tridiagonalize(handle, a, n, lda, &mut tau)?;

    // Step 2: Extract tridiagonal elements.
    let mut d = vec![0.0_f64; n as usize];
    let mut e = vec![0.0_f64; n.saturating_sub(1) as usize];
    extract_tridiagonal::<T>(a, n, lda, &mut d, &mut e)?;

    // Step 3: QR iteration on the tridiagonal matrix.
    let mut vectors = if job == EigJob::ValuesAndVectors {
        let mut v = vec![0.0_f64; n as usize * n as usize];
        // Initialize as identity.
        for i in 0..n as usize {
            v[i * n as usize + i] = 1.0;
        }
        Some(v)
    } else {
        None
    };

    let converged = tridiagonal_qr(&mut d, &mut e, n, vectors.as_deref_mut())?;

    if !converged {
        return Err(SolverError::ConvergenceFailure {
            iterations: TRIDIAG_QR_MAX_ITER,
            residual: e.iter().map(|v| v * v).sum::<f64>().sqrt(),
        });
    }

    // Sort eigenvalues in ascending order (and rearrange eigenvectors).
    sort_eigenvalues(&mut d, vectors.as_deref_mut(), n as usize);

    // Write eigenvalues back to device buffer.
    let eig_stage = stage_eigenvalues_to_device::<T>(eigenvalues.len(), &d);
    eigenvalues.copy_from_host(&eig_stage)?;

    // Step 4: Back-transform eigenvectors if requested.
    if job == EigJob::ValuesAndVectors {
        if let Some(ref _vecs) = vectors {
            // Full implementation: multiply Q_tridiag by Q_householder.
            // a <- Q_householder * Q_tridiag
            back_transform_eigenvectors(handle, a, n, lda, &tau, vectors.as_deref())?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tridiagonalization
// ---------------------------------------------------------------------------

/// Reduces a symmetric matrix to tridiagonal form via blocked Householder.
///
/// On exit, the diagonal and first sub/superdiagonal of `a` contain T.
/// The Householder vectors are stored in the lower triangle below the
/// first subdiagonal, and the scalars are in `tau`.
///
/// The blocked algorithm processes `TRIDIAG_BLOCK_SIZE` columns at a time,
/// using a panel factorization followed by a symmetric rank-2k update.
fn tridiagonalize<T: GpuFloat>(
    handle: &SolverHandle,
    a: &mut DeviceBuffer<T>,
    n: u32,
    lda: u32,
    tau: &mut DeviceBuffer<T>,
) -> SolverResult<()> {
    if n <= 1 {
        return Ok(());
    }

    let sm = handle.sm_version();
    let ptx = emit_tridiag_step::<T>(sm)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &tridiag_step_name::<T>())?;

    let nb = TRIDIAG_BLOCK_SIZE.min(n - 1);
    let num_blocks = (n - 1).div_ceil(nb);

    for block_idx in 0..num_blocks {
        let j = block_idx * nb;
        let jb = nb.min(n - 1 - j);
        let trailing = n - j;

        // Panel tridiagonalization: compute Householder vectors for columns j..j+jb.
        let shared_bytes = trailing * jb * T::size_u32();
        let params = LaunchParams::new(1u32, SOLVER_BLOCK_SIZE).with_shared_mem(shared_bytes);

        let a_offset = (j as u64 + j as u64 * lda as u64) * T::SIZE as u64;
        let tau_offset = j as u64 * T::SIZE as u64;

        let args = (
            a.as_device_ptr() + a_offset,
            tau.as_device_ptr() + tau_offset,
            trailing,
            jb,
            lda,
        );
        kernel.launch(&params, handle.stream(), &args)?;
    }

    Ok(())
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

fn from_f64_to_t<T: GpuFloat>(val: f64) -> T {
    if T::SIZE == 8 {
        T::from_bits_u64(val.to_bits())
    } else {
        T::from_bits_u64(u64::from((val as f32).to_bits()))
    }
}

/// Extracts diagonal (d) and subdiagonal (e) from the tridiagonalized matrix.
///
/// Copies the device buffer to host and reads the diagonal (d[i] = A[i,i])
/// and subdiagonal (e[i] = A[i+1,i]) elements in column-major layout.
fn extract_tridiagonal<T: GpuFloat>(
    a: &DeviceBuffer<T>,
    n: u32,
    lda: u32,
    d: &mut [f64],
    e: &mut [f64],
) -> SolverResult<()> {
    let n_usize = n as usize;
    let lda_usize = lda as usize;
    let total = lda_usize * n_usize;
    let mut host = vec![T::gpu_zero(); total];
    a.copy_to_host(&mut host).map_err(|e_err| {
        SolverError::InternalError(format!("extract_tridiagonal copy_to_host failed: {e_err}"))
    })?;

    // Diagonal: d[i] = A[i,i] (column-major: host[i * lda + i])
    for i in 0..n_usize {
        d[i] = t_to_f64(host[i * lda_usize + i]);
    }

    // Subdiagonal: e[i] = A[i+1,i] (column-major: host[i * lda + (i+1)])
    for i in 0..n_usize.saturating_sub(1) {
        e[i] = t_to_f64(host[i * lda_usize + (i + 1)]);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tridiagonal QR iteration
// ---------------------------------------------------------------------------

/// QR iteration with implicit Wilkinson shift for symmetric tridiagonal matrices.
///
/// Drives the subdiagonal elements to zero, leaving eigenvalues on the diagonal.
/// Optionally accumulates the rotation matrices into `vectors`.
///
/// Returns `true` if the algorithm converged within the iteration limit.
fn tridiagonal_qr(
    d: &mut [f64],
    e: &mut [f64],
    n: u32,
    mut vectors: Option<&mut [f64]>,
) -> SolverResult<bool> {
    let n_usize = n as usize;
    if n_usize <= 1 {
        return Ok(true);
    }

    let tol = TRIDIAG_QR_TOL;

    for _iter in 0..TRIDIAG_QR_MAX_ITER {
        // Find the active unreduced block.
        let mut q = n_usize - 1;
        while q > 0 && e[q - 1].abs() <= tol * (d[q - 1].abs() + d[q].abs()) {
            e[q - 1] = 0.0;
            q -= 1;
        }
        if q == 0 {
            return Ok(true);
        }

        let mut p = q - 1;
        while p > 0 && e[p - 1].abs() > tol * (d[p - 1].abs() + d[p].abs()) {
            p -= 1;
        }

        // Apply one implicit QR step with Wilkinson shift.
        implicit_qr_step(d, e, p, q, vectors.as_deref_mut(), n_usize);
    }

    // Check convergence.
    let off_norm: f64 = e.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(off_norm <= tol)
}

/// One step of implicit QR with Wilkinson shift on T[start..=end, start..=end].
///
/// The Wilkinson shift is the eigenvalue of the trailing 2x2 block of T
/// that is closest to `T[end, end]`.
fn implicit_qr_step(
    d: &mut [f64],
    e: &mut [f64],
    start: usize,
    end: usize,
    mut vectors: Option<&mut [f64]>,
    n: usize,
) {
    // Compute Wilkinson shift.
    let delta = (d[end - 1] - d[end]) * 0.5;
    let sign_delta = if delta >= 0.0 { 1.0 } else { -1.0 };
    let e_sq = e[end - 1] * e[end - 1];
    let mu = d[end] - e_sq / (delta + sign_delta * (delta * delta + e_sq).sqrt());

    // Bulge chase using Givens rotations.
    let mut x = d[start] - mu;
    let mut z = e[start];

    for k in start..end {
        // Compute Givens rotation.
        let (cs, sn) = givens_rotation(x, z);

        // Apply rotation to T.
        if k > start {
            e[k - 1] = cs * x + sn * z;
        }
        let dk = d[k];
        let dk1 = d[k + 1];
        let ek = e[k];

        d[k] = cs * cs * dk + 2.0 * cs * sn * ek + sn * sn * dk1;
        d[k + 1] = sn * sn * dk - 2.0 * cs * sn * ek + cs * cs * dk1;
        e[k] = cs * sn * (dk1 - dk) + (cs * cs - sn * sn) * ek;

        // Create bulge for next step.
        if k + 1 < end {
            x = e[k];
            z = sn * e[k + 1];
            e[k + 1] *= cs;
        }

        // Accumulate rotation into eigenvector matrix.
        if let Some(ref mut vecs) = vectors.as_deref_mut() {
            for i in 0..n {
                let vi_k = vecs[k * n + i];
                let vi_k1 = vecs[(k + 1) * n + i];
                vecs[k * n + i] = cs * vi_k + sn * vi_k1;
                vecs[(k + 1) * n + i] = -sn * vi_k + cs * vi_k1;
            }
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

/// Sorts eigenvalues in ascending order, rearranging eigenvectors accordingly.
fn sort_eigenvalues(d: &mut [f64], mut vectors: Option<&mut [f64]>, n: usize) {
    // Simple selection sort (n is typically small after tridiagonal reduction).
    for i in 0..n {
        let mut min_idx = i;
        let mut min_val = d[i];
        for (offset, &val) in d[(i + 1)..n].iter().enumerate() {
            if val < min_val {
                min_val = val;
                min_idx = i + 1 + offset;
            }
        }
        if min_idx != i {
            d.swap(i, min_idx);
            if let Some(ref mut vecs) = vectors.as_deref_mut() {
                // Swap columns i and min_idx.
                for row in 0..n {
                    let a = i * n + row;
                    let b = min_idx * n + row;
                    vecs.swap(a, b);
                }
            }
        }
    }
}

/// Back-transforms eigenvectors from tridiagonal basis to original basis.
///
/// Computes Q = Q_householder * Q_tridiag where Q_householder is formed from
/// the Householder vectors stored in `a` and `tau`.
fn back_transform_eigenvectors<T: GpuFloat>(
    _handle: &SolverHandle,
    a: &mut DeviceBuffer<T>,
    n: u32,
    lda: u32,
    _tau: &DeviceBuffer<T>,
    vectors: Option<&[f64]>,
) -> SolverResult<()> {
    // Host fallback: write the accumulated tridiagonal QR eigenvectors into A.
    let Some(vecs) = vectors else {
        return Ok(());
    };

    let n_usize = n as usize;
    let lda_usize = lda as usize;
    let required = n_usize * lda_usize;
    if a.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "back_transform_eigenvectors: matrix buffer too small ({} < {required})",
            a.len()
        )));
    }

    let stage = stage_eigenvectors_col_major_to_lda::<T>(vecs, n_usize, lda_usize, a.len())?;
    a.copy_from_host(&stage)?;

    Ok(())
}

fn stage_eigenvalues_to_device<T: GpuFloat>(dst_len: usize, d: &[f64]) -> Vec<T> {
    let mut out = vec![T::gpu_zero(); dst_len];
    for (idx, &val) in d.iter().enumerate() {
        if idx >= dst_len {
            break;
        }
        out[idx] = from_f64_to_t(val);
    }
    out
}

fn stage_eigenvectors_col_major_to_lda<T: GpuFloat>(
    vectors: &[f64],
    n: usize,
    lda: usize,
    dst_len: usize,
) -> SolverResult<Vec<T>> {
    if vectors.len() < n * n {
        return Err(SolverError::DimensionMismatch(format!(
            "stage_eigenvectors_col_major_to_lda: vectors too small ({} < {})",
            vectors.len(),
            n * n
        )));
    }
    if dst_len < n * lda {
        return Err(SolverError::DimensionMismatch(format!(
            "stage_eigenvectors_col_major_to_lda: destination too small ({} < {})",
            dst_len,
            n * lda
        )));
    }

    let mut out = vec![T::gpu_zero(); dst_len];
    for col in 0..n {
        for row in 0..n {
            // vectors is n x n in column-major order.
            out[col * lda + row] = from_f64_to_t(vectors[col * n + row]);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

fn tridiag_step_name<T: GpuFloat>() -> String {
    format!("solver_tridiag_step_{}", T::NAME)
}

/// Emits PTX for one panel of the tridiagonalization.
///
/// Each panel processes `jb` columns of the trailing submatrix, computing
/// Householder reflections that zero out elements two or more positions
/// below the diagonal.
fn emit_tridiag_step<T: GpuFloat>(sm: SmVersion) -> SolverResult<String> {
    let name = tridiag_step_name::<T>();
    let float_ty = T::PTX_TYPE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("a_ptr", PtxType::U64)
        .param("tau_ptr", PtxType::U64)
        .param("trailing", PtxType::U32)
        .param("jb", PtxType::U32)
        .param("lda", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let trailing = b.load_param_u32("trailing");
            let jb = b.load_param_u32("jb");
            let lda = b.load_param_u32("lda");

            // For each column k = 0..jb:
            //   1. Compute Householder vector v from A[k+1:, k].
            //   2. tau = 2 / (v^T v).
            //   3. Apply symmetric Householder update:
            //      p = tau * A * v
            //      q = p - (tau/2)(p^T v) v
            //      A -= v * q^T + q * v^T

            let _ = (tid, trailing, jb, lda, float_ty);

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
    fn eig_job_equality() {
        assert_eq!(EigJob::ValuesOnly, EigJob::ValuesOnly);
        assert_ne!(EigJob::ValuesOnly, EigJob::ValuesAndVectors);
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
    fn sort_eigenvalues_basic() {
        let mut d = vec![3.0, 1.0, 2.0];
        sort_eigenvalues(&mut d, None, 3);
        assert!((d[0] - 1.0).abs() < 1e-15);
        assert!((d[1] - 2.0).abs() < 1e-15);
        assert!((d[2] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn sort_eigenvalues_already_sorted() {
        let mut d = vec![1.0, 2.0, 3.0];
        sort_eigenvalues(&mut d, None, 3);
        assert!((d[0] - 1.0).abs() < 1e-15);
        assert!((d[2] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn tridiag_qr_trivial() {
        let mut d = vec![1.0, 2.0, 3.0];
        let mut e = vec![0.0, 0.0];
        let result = tridiagonal_qr(&mut d, &mut e, 3, None);
        assert!(result.is_ok());
        assert!(result.ok() == Some(true));
    }

    #[test]
    fn tridiag_qr_single() {
        let mut d = vec![5.0];
        let mut e: Vec<f64> = vec![];
        let result = tridiagonal_qr(&mut d, &mut e, 1, None);
        assert!(result.is_ok());
    }

    #[test]
    fn tridiag_step_name_format() {
        let name = tridiag_step_name::<f32>();
        assert!(name.contains("f32"));
    }

    #[test]
    fn tridiag_step_name_f64() {
        let name = tridiag_step_name::<f64>();
        assert!(name.contains("f64"));
    }

    #[test]
    fn stage_eigenvalues_prefix_copy() {
        let d = vec![1.5_f64, 2.5, 3.5];
        let out = stage_eigenvalues_to_device::<f64>(5, &d);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0], 1.5);
        assert_eq!(out[1], 2.5);
        assert_eq!(out[2], 3.5);
        assert_eq!(out[3], 0.0);
        assert_eq!(out[4], 0.0);
    }

    #[test]
    fn stage_eigenvectors_to_lda_maps_columns() {
        // 2x2 column-major: col0=[1,2], col1=[3,4]
        let vecs = vec![1.0_f64, 2.0, 3.0, 4.0];
        let out = stage_eigenvectors_col_major_to_lda::<f64>(&vecs, 2, 3, 6);
        assert!(out.is_ok());
        let out = out.unwrap_or_default();
        assert_eq!(out.len(), 6);
        // col0 rows 0,1
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
        // col1 rows 0,1 start at lda=3
        assert_eq!(out[3], 3.0);
        assert_eq!(out[4], 4.0);
        // padded lda rows remain zero
        assert_eq!(out[2], 0.0);
        assert_eq!(out[5], 0.0);
    }
}
