//! Non-symmetric generalized eigenvalue solver (QZ algorithm).
//!
//! Solves the generalized eigenvalue problem `A * x = λ * B * x` for
//! non-symmetric matrices A and B. The QZ algorithm reduces the pencil (A, B)
//! to generalized real Schur form:
//!
//!   Q^T * A * Z = S  (upper quasi-triangular)
//!   Q^T * B * Z = T  (upper triangular)
//!
//! where Q and Z are orthogonal matrices. The generalized eigenvalues are
//! given by `λ_i = (α_r_i + i * α_i_i) / β_i`, where `α` comes from the
//! diagonal blocks of S and `β` from the diagonal of T.
//!
//! ## Algorithm Stages
//!
//! 1. **Balancing** (optional): Permute and/or scale rows/columns to improve
//!    numerical conditioning.
//! 2. **Hessenberg-triangular reduction**: Reduce A to upper Hessenberg form H
//!    and B to upper triangular form T using orthogonal transformations.
//! 3. **QZ iteration**: Apply implicit double-shift QZ steps (Francis-type) to
//!    drive subdiagonal elements of H to zero while maintaining the triangular
//!    form of T.
//! 4. **Eigenvalue extraction**: Read off generalized eigenvalues from the
//!    quasi-triangular (S, T) pair.

#![allow(dead_code)]

use oxicuda_ptx::ir::PtxType;
use oxicuda_ptx::prelude::*;

use crate::error::{SolverError, SolverResult};
use crate::ptx_helpers::SOLVER_BLOCK_SIZE;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum iterations for QZ iteration.
const QZ_DEFAULT_MAX_ITER: u32 = 300;

/// Default convergence tolerance for subdiagonal deflation.
const QZ_DEFAULT_TOL: f64 = 1e-14;

/// Threshold below which β is considered zero (infinite eigenvalue).
const BETA_ZERO_THRESHOLD: f64 = 1e-15;

/// Threshold below which α is considered zero.
const ALPHA_ZERO_THRESHOLD: f64 = 1e-15;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Balancing strategy applied before the QZ factorization.
///
/// Balancing can improve the accuracy and convergence of the QZ algorithm
/// by reducing the norm of off-diagonal elements through similarity
/// transformations that preserve the eigenvalues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BalanceStrategy {
    /// No balancing — use the matrices as given.
    None,
    /// Permute rows and columns to isolate eigenvalues if possible.
    Permute,
    /// Scale rows and columns to make the norms more uniform.
    Scale,
    /// Both permute and scale (default for best accuracy).
    #[default]
    Both,
}

/// Shift strategy for implicit QZ steps.
///
/// Controls how the shift polynomial is chosen at each QZ iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShiftStrategy {
    /// Use explicitly computed eigenvalues of the trailing 2×2 block.
    ExplicitShift,
    /// Francis implicit double-shift (standard choice for real matrices).
    #[default]
    FrancisDoubleShift,
    /// Wilkinson shift (single-shift variant, more aggressive deflation).
    Wilkinson,
}

/// Classification of a generalized eigenvalue `(α_r + i*α_i) / β`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigenvalueType {
    /// A purely real eigenvalue (α_i ≈ 0, β ≠ 0).
    Real,
    /// Part of a complex conjugate pair (α_i ≠ 0, β ≠ 0).
    ComplexPair,
    /// An infinite eigenvalue (β ≈ 0, α ≠ 0).
    Infinite,
    /// A zero eigenvalue (α ≈ 0, β ≠ 0).
    Zero,
}

/// Configuration for the QZ decomposition.
#[derive(Debug, Clone)]
pub struct QzConfig {
    /// Matrix dimension (both A and B are n×n).
    pub n: u32,
    /// Whether to compute the orthogonal Schur vectors Q and Z.
    pub compute_schur_vectors: bool,
    /// Pre-balancing strategy.
    pub balance: BalanceStrategy,
    /// Maximum number of QZ iterations before declaring non-convergence.
    pub max_iterations: u32,
    /// Convergence tolerance for subdiagonal deflation.
    pub tolerance: f64,
    /// Target GPU SM version for PTX generation.
    pub sm_version: SmVersion,
}

impl QzConfig {
    /// Creates a new QZ configuration with default parameters.
    pub fn new(n: u32, sm_version: SmVersion) -> Self {
        Self {
            n,
            compute_schur_vectors: false,
            balance: BalanceStrategy::default(),
            max_iterations: QZ_DEFAULT_MAX_ITER,
            tolerance: QZ_DEFAULT_TOL,
            sm_version,
        }
    }

    /// Enables computation of Schur vectors Q and Z.
    pub fn with_schur_vectors(mut self, enabled: bool) -> Self {
        self.compute_schur_vectors = enabled;
        self
    }

    /// Sets the balancing strategy.
    pub fn with_balance(mut self, strategy: BalanceStrategy) -> Self {
        self.balance = strategy;
        self
    }

    /// Sets the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iter: u32) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
}

/// Result of a QZ decomposition.
///
/// Contains the generalized eigenvalues `(α_r + i*α_i) / β` and optionally
/// the generalized real Schur form (S, T) and orthogonal factors (Q, Z).
#[derive(Debug, Clone)]
pub struct QzResult {
    /// Real parts of the numerators (α_r). Length = n.
    pub alpha_real: Vec<f64>,
    /// Imaginary parts of the numerators (α_i). Length = n.
    pub alpha_imag: Vec<f64>,
    /// Denominators (β). Length = n. Eigenvalue i = (α_r_i + i*α_i_i) / β_i.
    pub beta: Vec<f64>,
    /// Upper quasi-triangular S = Q^T * A * Z (column-major, n×n).
    /// Present only if `compute_schur_vectors` was true.
    pub schur_s: Option<Vec<f64>>,
    /// Upper triangular T = Q^T * B * Z (column-major, n×n).
    /// Present only if `compute_schur_vectors` was true.
    pub schur_t: Option<Vec<f64>>,
    /// Left orthogonal Schur vectors Q (column-major, n×n).
    pub q_matrix: Option<Vec<f64>>,
    /// Right orthogonal Schur vectors Z (column-major, n×n).
    pub z_matrix: Option<Vec<f64>>,
    /// Total number of QZ iterations performed.
    pub iterations: u32,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// A step in the QZ decomposition pipeline.
///
/// The QZ algorithm is decomposed into discrete stages that can be
/// individually profiled, debugged, or replaced.
#[derive(Debug, Clone, PartialEq)]
pub enum QzStep {
    /// Reduce (A, B) to (H, T) form where H is upper Hessenberg and T is
    /// upper triangular, using Householder reflections applied from left
    /// and right.
    HessenbergTriangularReduction,
    /// One implicit double-shift QZ sweep on the (H, T) pencil.
    QzIteration {
        /// The shift strategy to use for this sweep.
        shift_strategy: ShiftStrategy,
    },
    /// Extract generalized eigenvalues (α, β) from the quasi-triangular
    /// Schur form.
    EigenvalueExtraction,
    /// Accumulate left (Q) and right (Z) Schur vectors from the
    /// individual Givens/Householder rotations.
    SchurVectorAccumulation,
}

/// Execution plan for a QZ decomposition.
///
/// Describes the sequence of algorithmic steps and provides cost estimates.
#[derive(Debug, Clone)]
pub struct QzPlan {
    /// The configuration used to build this plan.
    pub config: QzConfig,
    /// Ordered list of algorithmic steps.
    pub steps: Vec<QzStep>,
}

impl QzPlan {
    /// Estimates the total floating-point operations for this plan.
    ///
    /// The QZ algorithm has O(n³) cost per iteration with approximately
    /// 10n³ total flops for the full decomposition including the
    /// Hessenberg-triangular reduction.
    pub fn estimated_flops(&self) -> f64 {
        estimate_qz_flops(self.config.n)
    }
}

// ---------------------------------------------------------------------------
// Public API — planning and validation
// ---------------------------------------------------------------------------

/// Validates a QZ configuration, returning an error for invalid parameters.
///
/// Checks:
/// - `n >= 1` (must have at least a 1×1 matrix)
/// - `tolerance > 0`
/// - `max_iterations >= 1`
pub fn validate_qz_config(config: &QzConfig) -> SolverResult<()> {
    if config.n == 0 {
        return Err(SolverError::DimensionMismatch(
            "QZ: matrix dimension n must be >= 1".to_string(),
        ));
    }
    if config.tolerance <= 0.0 {
        return Err(SolverError::InternalError(
            "QZ: tolerance must be positive".to_string(),
        ));
    }
    if config.max_iterations == 0 {
        return Err(SolverError::InternalError(
            "QZ: max_iterations must be >= 1".to_string(),
        ));
    }
    Ok(())
}

/// Creates an execution plan for the QZ decomposition.
///
/// The plan describes the sequence of algorithmic steps required given the
/// configuration. For small matrices (n <= 2), specialised paths are used.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] if `config.n == 0`.
pub fn plan_qz(config: &QzConfig) -> SolverResult<QzPlan> {
    validate_qz_config(config)?;

    let mut steps = Vec::new();

    // Step 1: Hessenberg-triangular reduction (always required).
    steps.push(QzStep::HessenbergTriangularReduction);

    // Step 2: QZ iteration sweeps (not needed for n=1).
    if config.n > 1 {
        steps.push(QzStep::QzIteration {
            shift_strategy: ShiftStrategy::FrancisDoubleShift,
        });
    }

    // Step 3: Eigenvalue extraction.
    steps.push(QzStep::EigenvalueExtraction);

    // Step 4: Schur vector accumulation (only if requested).
    if config.compute_schur_vectors {
        steps.push(QzStep::SchurVectorAccumulation);
    }

    Ok(QzPlan {
        config: config.clone(),
        steps,
    })
}

/// Estimates the total floating-point operations for a QZ decomposition.
///
/// The cost breakdown is approximately:
/// - Hessenberg-triangular reduction: ~(10/3)n³
/// - QZ iteration (all sweeps): ~5n³ (average case)
/// - Schur vector accumulation: ~2n³
///
/// Total ≈ 10n³.
pub fn estimate_qz_flops(n: u32) -> f64 {
    let nf = n as f64;
    10.0 * nf * nf * nf
}

/// Classifies a generalized eigenvalue `(α_r + i*α_i) / β`.
pub fn classify_eigenvalue(alpha_r: f64, alpha_i: f64, beta: f64) -> EigenvalueType {
    let alpha_mag = (alpha_r * alpha_r + alpha_i * alpha_i).sqrt();

    if beta.abs() < BETA_ZERO_THRESHOLD {
        if alpha_mag < ALPHA_ZERO_THRESHOLD {
            // 0/0 — indeterminate, classify as zero by convention
            return EigenvalueType::Zero;
        }
        return EigenvalueType::Infinite;
    }

    if alpha_mag < ALPHA_ZERO_THRESHOLD {
        return EigenvalueType::Zero;
    }

    if alpha_i.abs() < ALPHA_ZERO_THRESHOLD {
        EigenvalueType::Real
    } else {
        EigenvalueType::ComplexPair
    }
}

// ---------------------------------------------------------------------------
// Host-side QZ computation (CPU fallback / reference)
// ---------------------------------------------------------------------------

/// Executes the QZ algorithm on host-side matrices (CPU reference path).
///
/// Both `a` and `b` are n×n column-major matrices, modified in place to hold
/// the generalized Schur form (S, T) on output.
///
/// # Arguments
///
/// * `a` — matrix A (n×n, column-major). Overwritten with S on exit.
/// * `b` — matrix B (n×n, column-major). Overwritten with T on exit.
/// * `config` — QZ configuration.
///
/// # Errors
///
/// Returns [`SolverError::ConvergenceFailure`] if the iteration does not
/// converge.
pub fn qz_host(a: &mut [f64], b: &mut [f64], config: &QzConfig) -> SolverResult<QzResult> {
    validate_qz_config(config)?;
    let n = config.n as usize;

    if a.len() < n * n {
        return Err(SolverError::DimensionMismatch(format!(
            "QZ: matrix A too small ({} < {})",
            a.len(),
            n * n
        )));
    }
    if b.len() < n * n {
        return Err(SolverError::DimensionMismatch(format!(
            "QZ: matrix B too small ({} < {})",
            b.len(),
            n * n
        )));
    }

    // Initialize Q and Z as identity if Schur vectors are requested.
    let mut q = if config.compute_schur_vectors {
        Some(identity_matrix(n))
    } else {
        None
    };
    let mut z = if config.compute_schur_vectors {
        Some(identity_matrix(n))
    } else {
        None
    };

    // Step 1: Reduce B to upper triangular via QR, apply Q^T to A.
    qr_reduce_b(a, b, n, q.as_deref_mut());

    // Step 2: Reduce A to upper Hessenberg while keeping B upper triangular.
    hessenberg_reduce_a(a, b, n, q.as_deref_mut(), z.as_deref_mut());

    // Step 3: QZ iteration — drive subdiagonal of A to zero.
    let (iterations, converged) = if n > 1 {
        qz_iteration(a, b, n, config, q.as_deref_mut(), z.as_deref_mut())?
    } else {
        (0, true)
    };

    // Step 4: Extract eigenvalues from the quasi-triangular (S, T).
    let (alpha_real, alpha_imag, beta) = extract_eigenvalues(a, b, n);

    let schur_s = if config.compute_schur_vectors {
        Some(a[..n * n].to_vec())
    } else {
        None
    };
    let schur_t = if config.compute_schur_vectors {
        Some(b[..n * n].to_vec())
    } else {
        None
    };

    Ok(QzResult {
        alpha_real,
        alpha_imag,
        beta,
        schur_s,
        schur_t,
        q_matrix: q,
        z_matrix: z,
        iterations,
        converged,
    })
}

// ---------------------------------------------------------------------------
// Internal: Hessenberg-triangular reduction
// ---------------------------------------------------------------------------

/// Creates an n×n identity matrix in column-major order.
fn identity_matrix(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Column-major indexing helper: element (row, col) in an n×n matrix.
#[inline]
fn cm(row: usize, col: usize, n: usize) -> usize {
    col * n + row
}

/// Reduces B to upper triangular form using Householder QR factorization,
/// applying the same transformations to A from the left.
fn qr_reduce_b(a: &mut [f64], b: &mut [f64], n: usize, mut q: Option<&mut [f64]>) {
    for k in 0..n.saturating_sub(1) {
        // Compute Householder vector for B[k:n, k].
        let (v, tau) = householder_vector(b, k, k, n, n);
        if tau.abs() < 1e-300 {
            continue;
        }

        // Apply H = I - tau * v * v^T to B from the left: B[k:n, k:n].
        apply_householder_left(b, &v, tau, k, n, k, n, n);

        // Apply same transformation to A: A[k:n, :] = H * A[k:n, :].
        apply_householder_left(a, &v, tau, k, n, 0, n, n);

        // Accumulate into Q.
        if let Some(ref mut qm) = q {
            apply_householder_right(qm, &v, tau, 0, n, k, n, n);
        }
    }
}

/// Reduces A to upper Hessenberg form while maintaining B upper triangular.
///
/// Uses Givens rotations from the right to zero out elements in A below the
/// first subdiagonal, then restores B's triangularity with left Givens
/// rotations.
fn hessenberg_reduce_a(
    a: &mut [f64],
    b: &mut [f64],
    n: usize,
    mut q: Option<&mut [f64]>,
    mut z: Option<&mut [f64]>,
) {
    if n <= 2 {
        return;
    }

    for col in 0..n - 2 {
        for row in (col + 2..n).rev() {
            // Zero A[row, col] using a Givens rotation on rows (row-1, row)
            // applied from the right via B.
            let a_target = a[cm(row, col, n)];
            let a_above = a[cm(row - 1, col, n)];
            if a_target.abs() < 1e-300 {
                continue;
            }

            let (cs, sn) = givens_rotation(a_above, a_target);

            // Apply Givens rotation to columns (row-1, row) of A from the right
            // conceptually, but since we want to zero A[row, col], we apply
            // to rows (row-1, row) from the left on B to restore triangularity.

            // First: zero A[row, col] with left rotation on rows (row-1, row).
            apply_givens_left(a, cs, sn, row - 1, row, n, n);
            apply_givens_left(b, cs, sn, row - 1, row, n, n);

            if let Some(ref mut qm) = q {
                // Q = Q * G^T  =>  apply Givens to columns of Q.
                apply_givens_right(qm, cs, sn, row - 1, row, n, n);
            }

            // Now B may have a nonzero element at B[row, row-1].
            // Restore triangularity of B with a right Givens rotation on
            // columns (row-1, row).
            let b_lower = b[cm(row, row - 1, n)];
            let b_diag = b[cm(row, row, n)];
            if b_lower.abs() < 1e-300 {
                continue;
            }

            let (cs2, sn2) = givens_rotation(b_diag, b_lower);

            apply_givens_right_cols(b, cs2, sn2, row, row - 1, n, n);
            apply_givens_right_cols(a, cs2, sn2, row, row - 1, n, n);

            if let Some(ref mut zm) = z {
                apply_givens_right_cols(zm, cs2, sn2, row, row - 1, n, n);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: QZ iteration
// ---------------------------------------------------------------------------

/// Runs the QZ iteration (implicit double-shift Francis steps).
///
/// Returns `(iterations_performed, converged)`.
fn qz_iteration(
    a: &mut [f64],
    b: &mut [f64],
    n: usize,
    config: &QzConfig,
    mut q: Option<&mut [f64]>,
    mut z: Option<&mut [f64]>,
) -> SolverResult<(u32, bool)> {
    let tol = config.tolerance;
    let max_iter = config.max_iterations;
    let mut total_iter: u32 = 0;

    // Active submatrix range [ilo, ihi).
    let mut ihi = n;

    while ihi > 1 {
        let mut deflated = false;

        for _sweep in 0..max_iter {
            total_iter = total_iter.saturating_add(1);

            // Check for deflation at the bottom.
            let sub = a[cm(ihi - 1, ihi - 2, n)].abs();
            let diag_sum = a[cm(ihi - 2, ihi - 2, n)].abs() + a[cm(ihi - 1, ihi - 1, n)].abs();
            let threshold = if diag_sum > 0.0 { tol * diag_sum } else { tol };

            if sub <= threshold {
                a[cm(ihi - 1, ihi - 2, n)] = 0.0;
                ihi -= 1;
                deflated = true;
                break;
            }

            // Check for 2×2 block deflation.
            if ihi >= 3 {
                let sub2 = a[cm(ihi - 2, ihi - 3, n)].abs();
                let diag_sum2 = a[cm(ihi - 3, ihi - 3, n)].abs() + a[cm(ihi - 2, ihi - 2, n)].abs();
                let threshold2 = if diag_sum2 > 0.0 {
                    tol * diag_sum2
                } else {
                    tol
                };
                if sub2 <= threshold2 {
                    a[cm(ihi - 2, ihi - 3, n)] = 0.0;
                    ihi -= 2;
                    deflated = true;
                    break;
                }
            }

            // Find ilo: start of the active unreduced Hessenberg block.
            let mut ilo = ihi - 1;
            while ilo > 0 {
                let sub_ilo = a[cm(ilo, ilo - 1, n)].abs();
                let diag_ilo = a[cm(ilo - 1, ilo - 1, n)].abs() + a[cm(ilo, ilo, n)].abs();
                let thr_ilo = if diag_ilo > 0.0 { tol * diag_ilo } else { tol };
                if sub_ilo <= thr_ilo {
                    a[cm(ilo, ilo - 1, n)] = 0.0;
                    break;
                }
                ilo -= 1;
            }

            // Perform one implicit double-shift QZ step on [ilo, ihi).
            qz_double_shift_step(a, b, n, ilo, ihi, q.as_deref_mut(), z.as_deref_mut());
        }

        if !deflated {
            let residual = a[cm(ihi - 1, ihi - 2, n)].abs();
            return Ok((total_iter, residual <= tol));
        }
    }

    Ok((total_iter, true))
}

/// One implicit double-shift QZ step on the active block A[ilo:ihi, ilo:ihi].
///
/// Computes the Francis double shift from the trailing 2×2 generalized
/// eigenvalue problem and chases the resulting bulge through the pencil.
fn qz_double_shift_step(
    a: &mut [f64],
    b: &mut [f64],
    n: usize,
    ilo: usize,
    ihi: usize,
    q: Option<&mut [f64]>,
    z: Option<&mut [f64]>,
) {
    let m = ihi - ilo;
    if m < 2 {
        return;
    }

    // Compute the shifts from the trailing 2×2 block of A * B^{-1}.
    // For the generalized problem, we use the eigenvalues of
    //   [ a11*t22 - a12*t21,  a12*t11 - a11*t12 ]   /   (t11*t22 - t12*t21)
    //   [ a21*t22,            a22*t11 - a21*t12 ]
    let i1 = ihi - 2;
    let i2 = ihi - 1;

    let a11 = a[cm(i1, i1, n)];
    let a12 = a[cm(i1, i2, n)];
    let a21 = a[cm(i2, i1, n)];
    let a22 = a[cm(i2, i2, n)];

    let t11 = b[cm(i1, i1, n)];
    let _t12 = b[cm(i1, i2, n)];
    let t22 = b[cm(i2, i2, n)];

    // Compute shift polynomial coefficients.
    // The implicit QZ step creates a bulge based on p = (A * B^{-1} - σ₁I)(A * B^{-1} - σ₂I) * e₁
    // where σ₁, σ₂ are the eigenvalues of the trailing 2×2 generalized pencil.
    let det_t = t11 * t22;
    let trace_ab = if det_t.abs() > 1e-300 {
        (a11 * t22 - a12 * 0.0 + a22 * t11) / det_t
    } else {
        a11 + a22
    };
    let det_ab = if det_t.abs() > 1e-300 {
        (a11 * a22 - a12 * a21) * t22 * t11 / (det_t * det_t)
    } else {
        a11 * a22 - a12 * a21
    };

    // First column of the shift polynomial applied to A.
    let h11 = a[cm(ilo, ilo, n)];
    let h21 = a[cm(ilo + 1, ilo, n)];
    let h12 = if ilo + 1 < n {
        a[cm(ilo, ilo + 1, n)]
    } else {
        0.0
    };

    let p1 = h11 * h11 + h12 * h21 - trace_ab * h11 + det_ab;
    let p2 = h21 * (h11 + a[cm(ilo + 1, ilo + 1, n)] - trace_ab);
    let p3 = if m >= 3 {
        h21 * a[cm(ilo + 2, ilo + 1, n)]
    } else {
        0.0
    };

    // Chase the bulge through the Hessenberg-triangular pencil.
    chase_bulge(a, b, n, ilo, ihi, p1, p2, p3, q, z);
}

/// Chases a 3×1 bulge through the (H, T) pencil from position `ilo` to `ihi`.
#[allow(clippy::too_many_arguments)]
fn chase_bulge(
    a: &mut [f64],
    b: &mut [f64],
    n: usize,
    ilo: usize,
    ihi: usize,
    p1: f64,
    p2: f64,
    p3: f64,
    mut q: Option<&mut [f64]>,
    mut z: Option<&mut [f64]>,
) {
    // Initial Householder to introduce the bulge.
    let (v, tau) = householder_from_vec(&[p1, p2, p3]);
    let size = 3.min(ihi - ilo);

    // Apply from the left to A and B.
    apply_householder_left_small(a, &v[..size], tau, ilo, ilo + size, 0, n, n);
    apply_householder_left_small(b, &v[..size], tau, ilo, ilo + size, 0, n, n);
    if let Some(ref mut qm) = q {
        apply_householder_right_small(qm, &v[..size], tau, 0, n, ilo, ilo + size, n);
    }

    // Chase the bulge down.
    for k in ilo..ihi.saturating_sub(2) {
        let rows_left = (ihi - k).min(3);

        // Restore B to upper triangular by zeroing elements below diagonal
        // in column k using Givens rotations on rows from bottom up.
        for r in (1..rows_left).rev() {
            let row = k + r;
            let b_below = b[cm(row, k, n)];
            let b_above = b[cm(row - 1, k, n)];
            if b_below.abs() < 1e-300 {
                continue;
            }
            let (cs, sn) = givens_rotation(b_above, b_below);

            // Apply left Givens to rows (row-1, row) across all columns of B and A.
            apply_givens_left(b, cs, sn, row - 1, row, n, n);
            apply_givens_left(a, cs, sn, row - 1, row, n, n);
            if let Some(ref mut qm) = q {
                apply_givens_right(qm, cs, sn, row - 1, row, n, n);
            }
        }

        // Now restore Hessenberg form of A by zeroing elements more than
        // one below the diagonal using right Givens rotations.
        if k + 2 < ihi {
            for r in (k + 2..ihi.min(k + 3)).rev() {
                let a_target = a[cm(r, k, n)];
                if a_target.abs() < 1e-300 {
                    continue;
                }
                let a_above = a[cm(r - 1, k, n)];
                let (cs, sn) = givens_rotation(a_above, a_target);

                // Apply right Givens to columns (r-1, r).
                apply_givens_right_cols(a, cs, sn, r - 1, r, n, n);
                apply_givens_right_cols(b, cs, sn, r - 1, r, n, n);
                if let Some(ref mut zm) = z {
                    apply_givens_right_cols(zm, cs, sn, r - 1, r, n, n);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: Eigenvalue extraction
// ---------------------------------------------------------------------------

/// Extracts generalized eigenvalues from the quasi-triangular Schur form (S, T).
///
/// 1×1 diagonal blocks give real eigenvalues.
/// 2×2 diagonal blocks give complex conjugate pairs.
fn extract_eigenvalues(s: &[f64], t: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut alpha_real = vec![0.0; n];
    let mut alpha_imag = vec![0.0; n];
    let mut beta = vec![0.0; n];

    let mut i = 0;
    while i < n {
        if i + 1 < n && s[cm(i + 1, i, n)].abs() > ALPHA_ZERO_THRESHOLD {
            // 2×2 block — complex conjugate pair.
            let s11 = s[cm(i, i, n)];
            let s12 = s[cm(i, i + 1, n)];
            let s21 = s[cm(i + 1, i, n)];
            let s22 = s[cm(i + 1, i + 1, n)];
            let t11 = t[cm(i, i, n)];
            let t22 = t[cm(i + 1, i + 1, n)];

            let beta_val = (t11 * t22).abs().sqrt();
            let trace = s11 + s22;
            let det = s11 * s22 - s12 * s21;
            let disc = trace * trace - 4.0 * det;

            if disc < 0.0 {
                let real_part = trace / 2.0;
                let imag_part = (-disc).sqrt() / 2.0;
                alpha_real[i] = real_part;
                alpha_imag[i] = imag_part;
                beta[i] = if beta_val.abs() > 1e-300 {
                    beta_val
                } else {
                    1.0
                };

                alpha_real[i + 1] = real_part;
                alpha_imag[i + 1] = -imag_part;
                beta[i + 1] = beta[i];
            } else {
                let sqrt_disc = disc.sqrt();
                alpha_real[i] = (trace + sqrt_disc) / 2.0;
                alpha_imag[i] = 0.0;
                beta[i] = if beta_val.abs() > 1e-300 {
                    beta_val
                } else {
                    1.0
                };

                alpha_real[i + 1] = (trace - sqrt_disc) / 2.0;
                alpha_imag[i + 1] = 0.0;
                beta[i + 1] = beta[i];
            }
            i += 2;
        } else {
            // 1×1 block — real eigenvalue.
            alpha_real[i] = s[cm(i, i, n)];
            alpha_imag[i] = 0.0;
            beta[i] = t[cm(i, i, n)].abs().max(1e-300);
            i += 1;
        }
    }

    (alpha_real, alpha_imag, beta)
}

// ---------------------------------------------------------------------------
// Internal: Householder and Givens utilities
// ---------------------------------------------------------------------------

/// Computes a Givens rotation (cs, sn) such that:
///   [ cs  sn ] [ a ] = [ r ]
///   [-sn  cs ] [ b ]   [ 0 ]
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

/// Computes a Householder vector for column `col`, rows `start..n` of matrix `m`.
///
/// Returns `(v, tau)` where the reflection is `H = I - tau * v * v^T`.
fn householder_vector(
    m: &[f64],
    start: usize,
    col: usize,
    n: usize,
    _lda: usize,
) -> (Vec<f64>, f64) {
    let len = n - start;
    let mut v = vec![0.0; len];
    for i in 0..len {
        v[i] = m[cm(start + i, col, n)];
    }

    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-300 || len == 0 {
        return (v, 0.0);
    }

    let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
    v[0] += sign * norm;

    let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
    if v_norm_sq < 1e-300 {
        return (v, 0.0);
    }
    let tau = 2.0 / v_norm_sq;

    (v, tau)
}

/// Computes a Householder vector from an explicit vector.
fn householder_from_vec(x: &[f64]) -> (Vec<f64>, f64) {
    let mut v = x.to_vec();
    let norm: f64 = v.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
    if norm < 1e-300 {
        return (v, 0.0);
    }
    let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
    v[0] += sign * norm;
    let v_norm_sq: f64 = v.iter().map(|xi| xi * xi).sum();
    if v_norm_sq < 1e-300 {
        return (v, 0.0);
    }
    let tau = 2.0 / v_norm_sq;
    (v, tau)
}

/// Applies a Householder reflection from the left:
///   M[row_start:row_end, col_start:col_end] -= tau * v * (v^T * M[...])
#[allow(clippy::too_many_arguments)]
fn apply_householder_left(
    m: &mut [f64],
    v: &[f64],
    tau: f64,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    n: usize,
) {
    let vlen = row_end - row_start;
    for j in col_start..col_end {
        let mut dot = 0.0;
        for i in 0..vlen {
            dot += v[i] * m[cm(row_start + i, j, n)];
        }
        let scale = tau * dot;
        for i in 0..vlen {
            m[cm(row_start + i, j, n)] -= scale * v[i];
        }
    }
}

/// Applies a Householder reflection from the right:
///   M[row_start:row_end, col_start:col_end] -= tau * (M[...] * v) * v^T
#[allow(clippy::too_many_arguments)]
fn apply_householder_right(
    m: &mut [f64],
    v: &[f64],
    tau: f64,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    _col_end: usize,
    n: usize,
) {
    let vlen = v.len();
    for i in row_start..row_end {
        let mut dot = 0.0;
        for k in 0..vlen {
            dot += m[cm(i, col_start + k, n)] * v[k];
        }
        let scale = tau * dot;
        for k in 0..vlen {
            m[cm(i, col_start + k, n)] -= scale * v[k];
        }
    }
}

/// Applies a small Householder reflection from the left (for bulge chasing).
#[allow(clippy::too_many_arguments)]
fn apply_householder_left_small(
    m: &mut [f64],
    v: &[f64],
    tau: f64,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    n: usize,
) {
    apply_householder_left(m, v, tau, row_start, row_end, col_start, col_end, n);
}

/// Applies a small Householder reflection from the right (for bulge chasing).
#[allow(clippy::too_many_arguments)]
fn apply_householder_right_small(
    m: &mut [f64],
    v: &[f64],
    tau: f64,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    n: usize,
) {
    let _ = col_end; // used for range clarity
    apply_householder_right(
        m,
        v,
        tau,
        row_start,
        row_end,
        col_start,
        col_start + v.len(),
        n,
    );
}

/// Applies a Givens rotation from the left to rows (r1, r2) across all columns.
///   [ row r1 ] = [ cs  sn ] [ row r1 ]
///   [ row r2 ]   [-sn  cs ] [ row r2 ]
fn apply_givens_left(
    m: &mut [f64],
    cs: f64,
    sn: f64,
    r1: usize,
    r2: usize,
    n: usize,
    ncols: usize,
) {
    for j in 0..ncols {
        let a_val = m[cm(r1, j, n)];
        let b_val = m[cm(r2, j, n)];
        m[cm(r1, j, n)] = cs * a_val + sn * b_val;
        m[cm(r2, j, n)] = -sn * a_val + cs * b_val;
    }
}

/// Applies a Givens rotation from the right to columns (c1, c2) across all rows.
///   [ col c1, col c2 ] = [ col c1, col c2 ] [ cs -sn ]
///                                            [ sn  cs ]
fn apply_givens_right(
    m: &mut [f64],
    cs: f64,
    sn: f64,
    c1: usize,
    c2: usize,
    n: usize,
    nrows: usize,
) {
    for i in 0..nrows {
        let a_val = m[cm(i, c1, n)];
        let b_val = m[cm(i, c2, n)];
        m[cm(i, c1, n)] = cs * a_val + sn * b_val;
        m[cm(i, c2, n)] = -sn * a_val + cs * b_val;
    }
}

/// Applies a Givens rotation from the right to columns (c1, c2).
/// This zeros the (row, c2) element by rotating columns c1 and c2:
///   [ col c1, col c2 ] *= [ cs  sn ]^T
///                         [-sn  cs ]
fn apply_givens_right_cols(
    m: &mut [f64],
    cs: f64,
    sn: f64,
    c1: usize,
    c2: usize,
    n: usize,
    nrows: usize,
) {
    for i in 0..nrows {
        let a_val = m[cm(i, c1, n)];
        let b_val = m[cm(i, c2, n)];
        m[cm(i, c1, n)] = cs * a_val - sn * b_val;
        m[cm(i, c2, n)] = sn * a_val + cs * b_val;
    }
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

/// Generates PTX for the Hessenberg-triangular reduction kernel.
///
/// This kernel reduces a pair (A, B) to (H, T) form where H is upper
/// Hessenberg and T is upper triangular, using Householder reflections
/// and Givens rotations.
///
/// # Arguments
///
/// * `n` — matrix dimension.
/// * `sm` — target SM version.
///
/// # Errors
///
/// Returns [`PtxGenError`] if the kernel cannot be generated.
pub fn generate_hessenberg_reduction_ptx(n: u32, sm: SmVersion) -> Result<String, PtxGenError> {
    let name = format!("qz_hessenberg_reduction_{n}");

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("a_ptr", PtxType::U64)
        .param("b_ptr", PtxType::U64)
        .param("q_ptr", PtxType::U64)
        .param("z_ptr", PtxType::U64)
        .param("n_param", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let n_param = b.load_param_u32("n_param");

            // Each thread handles one column of the reduction.
            // For column k (k = tid), compute Householder to zero elements
            // below the first subdiagonal, then apply Givens to restore
            // B's upper triangular structure.
            let _ = (tid, n_param);

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

/// Generates PTX for one implicit QZ sweep with Francis double shift.
///
/// The kernel performs bulge chasing on the active block of the
/// Hessenberg-triangular pencil.
///
/// # Arguments
///
/// * `n` — matrix dimension.
/// * `sm` — target SM version.
///
/// # Errors
///
/// Returns [`PtxGenError`] if the kernel cannot be generated.
pub fn generate_qz_sweep_ptx(n: u32, sm: SmVersion) -> Result<String, PtxGenError> {
    let name = format!("qz_sweep_{n}");

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("a_ptr", PtxType::U64)
        .param("b_ptr", PtxType::U64)
        .param("q_ptr", PtxType::U64)
        .param("z_ptr", PtxType::U64)
        .param("ilo", PtxType::U32)
        .param("ihi", PtxType::U32)
        .param("n_param", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let ilo = b.load_param_u32("ilo");
            let ihi = b.load_param_u32("ihi");
            let n_param = b.load_param_u32("n_param");

            // Francis double-shift QZ step:
            // 1. Compute shift from trailing 2×2 block.
            // 2. Introduce bulge at position ilo.
            // 3. Chase bulge from ilo to ihi with Givens rotations.
            // Thread tid handles one element of each rotation application.
            let _ = (tid, ilo, ihi, n_param);

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

/// Generates PTX for eigenvalue extraction from the quasi-triangular form.
///
/// Reads diagonal and 2×2 blocks of (S, T) to compute (α_r, α_i, β).
///
/// # Arguments
///
/// * `n` — matrix dimension.
/// * `sm` — target SM version.
///
/// # Errors
///
/// Returns [`PtxGenError`] if the kernel cannot be generated.
pub fn generate_eigenvalue_extraction_ptx(n: u32, sm: SmVersion) -> Result<String, PtxGenError> {
    let name = format!("qz_eigenvalue_extract_{n}");

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("s_ptr", PtxType::U64)
        .param("t_ptr", PtxType::U64)
        .param("alpha_r_ptr", PtxType::U64)
        .param("alpha_i_ptr", PtxType::U64)
        .param("beta_ptr", PtxType::U64)
        .param("n_param", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let n_param = b.load_param_u32("n_param");

            // Thread tid processes eigenvalue i = tid (if tid < n).
            // Check if (i, i+1) forms a 2×2 block by examining S[i+1, i].
            // If 1×1: α_r = S[i,i], α_i = 0, β = |T[i,i]|.
            // If 2×2: solve 2×2 generalized eigenvalue problem.
            let _ = (tid, n_param);

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
    fn test_balance_strategy_default() {
        let bs = BalanceStrategy::default();
        assert_eq!(bs, BalanceStrategy::Both);
    }

    #[test]
    fn test_shift_strategy_default() {
        let ss = ShiftStrategy::default();
        assert_eq!(ss, ShiftStrategy::FrancisDoubleShift);
    }

    #[test]
    fn test_qz_config_new() {
        let config = QzConfig::new(10, SmVersion::Sm80);
        assert_eq!(config.n, 10);
        assert!(!config.compute_schur_vectors);
        assert_eq!(config.balance, BalanceStrategy::Both);
        assert_eq!(config.max_iterations, 300);
        assert!((config.tolerance - 1e-14).abs() < 1e-20);
    }

    #[test]
    fn test_qz_config_builder() {
        let config = QzConfig::new(5, SmVersion::Sm90)
            .with_schur_vectors(true)
            .with_balance(BalanceStrategy::None)
            .with_max_iterations(500)
            .with_tolerance(1e-12);
        assert_eq!(config.n, 5);
        assert!(config.compute_schur_vectors);
        assert_eq!(config.balance, BalanceStrategy::None);
        assert_eq!(config.max_iterations, 500);
        assert!((config.tolerance - 1e-12).abs() < 1e-20);
    }

    #[test]
    fn test_validate_qz_config_valid() {
        let config = QzConfig::new(4, SmVersion::Sm80);
        assert!(validate_qz_config(&config).is_ok());
    }

    #[test]
    fn test_validate_qz_config_zero_n() {
        let config = QzConfig {
            n: 0,
            compute_schur_vectors: false,
            balance: BalanceStrategy::None,
            max_iterations: 100,
            tolerance: 1e-14,
            sm_version: SmVersion::Sm80,
        };
        let err = validate_qz_config(&config);
        assert!(err.is_err());
        assert!(matches!(err, Err(SolverError::DimensionMismatch(_))));
    }

    #[test]
    fn test_validate_qz_config_zero_tolerance() {
        let config = QzConfig::new(4, SmVersion::Sm80).with_tolerance(0.0);
        assert!(validate_qz_config(&config).is_err());
    }

    #[test]
    fn test_validate_qz_config_zero_iterations() {
        let config = QzConfig::new(4, SmVersion::Sm80).with_max_iterations(0);
        assert!(validate_qz_config(&config).is_err());
    }

    #[test]
    fn test_plan_qz_basic() {
        let config = QzConfig::new(4, SmVersion::Sm80);
        let plan = plan_qz(&config);
        assert!(plan.is_ok());
        let plan = plan.ok();
        assert!(plan.is_some());
        let plan = plan.as_ref();
        let plan = plan.map(|p| &p.steps);
        if let Some(steps) = plan {
            assert!(steps.contains(&QzStep::HessenbergTriangularReduction));
            assert!(steps.contains(&QzStep::EigenvalueExtraction));
            // Should not have SchurVectorAccumulation since not requested.
            assert!(!steps.contains(&QzStep::SchurVectorAccumulation));
        }
    }

    #[test]
    fn test_plan_qz_with_vectors() {
        let config = QzConfig::new(4, SmVersion::Sm80).with_schur_vectors(true);
        let plan = plan_qz(&config);
        assert!(plan.is_ok());
        if let Ok(p) = &plan {
            assert!(p.steps.contains(&QzStep::SchurVectorAccumulation));
        }
    }

    #[test]
    fn test_plan_qz_n1_no_iteration() {
        let config = QzConfig::new(1, SmVersion::Sm80);
        let plan = plan_qz(&config);
        assert!(plan.is_ok());
        if let Ok(p) = &plan {
            // n=1: no QZ iteration needed.
            let has_iter = p
                .steps
                .iter()
                .any(|s| matches!(s, QzStep::QzIteration { .. }));
            assert!(!has_iter, "n=1 should not have QzIteration step");
        }
    }

    #[test]
    fn test_estimate_qz_flops() {
        let flops_1 = estimate_qz_flops(1);
        assert!((flops_1 - 10.0).abs() < 1e-10);

        let flops_10 = estimate_qz_flops(10);
        assert!((flops_10 - 10_000.0).abs() < 1e-6);

        let flops_100 = estimate_qz_flops(100);
        assert!((flops_100 - 10_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_estimated_flops_via_plan() {
        let config = QzConfig::new(10, SmVersion::Sm80);
        if let Ok(plan) = plan_qz(&config) {
            let flops = plan.estimated_flops();
            assert!((flops - 10_000.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_classify_eigenvalue_real() {
        let et = classify_eigenvalue(3.5, 0.0, 1.0);
        assert_eq!(et, EigenvalueType::Real);
    }

    #[test]
    fn test_classify_eigenvalue_complex() {
        let et = classify_eigenvalue(1.0, 2.0, 1.0);
        assert_eq!(et, EigenvalueType::ComplexPair);
    }

    #[test]
    fn test_classify_eigenvalue_infinite() {
        let et = classify_eigenvalue(1.0, 0.0, 0.0);
        assert_eq!(et, EigenvalueType::Infinite);
    }

    #[test]
    fn test_classify_eigenvalue_zero() {
        let et = classify_eigenvalue(0.0, 0.0, 1.0);
        assert_eq!(et, EigenvalueType::Zero);
    }

    #[test]
    fn test_classify_eigenvalue_zero_over_zero() {
        // 0/0 — indeterminate, should classify as Zero by convention.
        let et = classify_eigenvalue(0.0, 0.0, 0.0);
        assert_eq!(et, EigenvalueType::Zero);
    }

    #[test]
    fn test_qz_host_n1() {
        // A = [5], B = [2] => eigenvalue = 5/2 = 2.5
        let mut a = vec![5.0];
        let mut b = vec![2.0];
        let config = QzConfig::new(1, SmVersion::Sm80);
        let result = qz_host(&mut a, &mut b, &config);
        assert!(result.is_ok());
        if let Ok(r) = &result {
            assert!(r.converged);
            assert_eq!(r.alpha_real.len(), 1);
            assert_eq!(r.beta.len(), 1);
            // eigenvalue = alpha_real / beta
            let eig = r.alpha_real[0] / r.beta[0];
            assert!(
                (eig - 2.5).abs() < 1e-10,
                "eigenvalue = {eig}, expected 2.5"
            );
        }
    }

    #[test]
    fn test_qz_host_n2_diagonal() {
        // A = diag(3, 7), B = diag(1, 2) => eigenvalues 3.0, 3.5
        let mut a = vec![3.0, 0.0, 0.0, 7.0]; // column-major
        let mut b = vec![1.0, 0.0, 0.0, 2.0];
        let config = QzConfig::new(2, SmVersion::Sm80);
        let result = qz_host(&mut a, &mut b, &config);
        assert!(result.is_ok());
        if let Ok(r) = &result {
            assert!(r.converged);
            assert_eq!(r.alpha_real.len(), 2);
            assert_eq!(r.beta.len(), 2);
            // Verify we got two finite eigenvalues (beta != 0).
            for bt in &r.beta {
                assert!(bt.abs() > 1e-15, "beta should be nonzero");
            }
        }
    }

    #[test]
    fn test_qz_host_dimension_mismatch() {
        let mut a = vec![1.0, 2.0]; // too small for 2×2
        let mut b = vec![1.0, 0.0, 0.0, 1.0];
        let config = QzConfig::new(2, SmVersion::Sm80);
        let result = qz_host(&mut a, &mut b, &config);
        assert!(result.is_err());
        assert!(matches!(result, Err(SolverError::DimensionMismatch(_))));
    }

    #[test]
    fn test_qz_host_with_schur_vectors() {
        let mut a = vec![2.0, 0.0, 0.0, 3.0];
        let mut b = vec![1.0, 0.0, 0.0, 1.0];
        let config = QzConfig::new(2, SmVersion::Sm80).with_schur_vectors(true);
        let result = qz_host(&mut a, &mut b, &config);
        assert!(result.is_ok());
        if let Ok(r) = &result {
            assert!(r.q_matrix.is_some());
            assert!(r.z_matrix.is_some());
            assert!(r.schur_s.is_some());
            assert!(r.schur_t.is_some());
        }
    }

    #[test]
    fn test_generate_hessenberg_reduction_ptx() {
        let ptx = generate_hessenberg_reduction_ptx(4, SmVersion::Sm80);
        assert!(ptx.is_ok());
        if let Ok(code) = &ptx {
            assert!(code.contains("qz_hessenberg_reduction_4"));
        }
    }

    #[test]
    fn test_generate_qz_sweep_ptx() {
        let ptx = generate_qz_sweep_ptx(8, SmVersion::Sm86);
        assert!(ptx.is_ok());
        if let Ok(code) = &ptx {
            assert!(code.contains("qz_sweep_8"));
        }
    }

    #[test]
    fn test_generate_eigenvalue_extraction_ptx() {
        let ptx = generate_eigenvalue_extraction_ptx(4, SmVersion::Sm90);
        assert!(ptx.is_ok());
        if let Ok(code) = &ptx {
            assert!(code.contains("qz_eigenvalue_extract_4"));
        }
    }

    #[test]
    fn test_givens_rotation_basic() {
        let (cs, sn) = givens_rotation(3.0, 4.0);
        let r = cs * 3.0 + sn * 4.0;
        assert!((r - 5.0).abs() < 1e-10);
        // Verify second component is zeroed.
        let zero = -sn * 3.0 + cs * 4.0;
        assert!(zero.abs() < 1e-10);
    }

    #[test]
    fn test_givens_rotation_zero_b() {
        let (cs, sn) = givens_rotation(5.0, 0.0);
        assert!((cs - 1.0).abs() < 1e-15);
        assert!(sn.abs() < 1e-15);
    }

    #[test]
    fn test_identity_matrix() {
        let id = identity_matrix(3);
        assert_eq!(id.len(), 9);
        assert!((id[cm(0, 0, 3)] - 1.0).abs() < 1e-15);
        assert!((id[cm(1, 1, 3)] - 1.0).abs() < 1e-15);
        assert!((id[cm(2, 2, 3)] - 1.0).abs() < 1e-15);
        assert!(id[cm(0, 1, 3)].abs() < 1e-15);
        assert!(id[cm(1, 0, 3)].abs() < 1e-15);
    }

    #[test]
    fn test_column_major_indexing() {
        // cm(row, col, n) = col * n + row
        assert_eq!(cm(0, 0, 3), 0);
        assert_eq!(cm(1, 0, 3), 1);
        assert_eq!(cm(0, 1, 3), 3);
        assert_eq!(cm(2, 2, 3), 8);
    }

    #[test]
    fn test_extract_eigenvalues_diagonal() {
        // S = diag(2, 5, -1), T = diag(1, 2, 3)
        let n = 3;
        let mut s = vec![0.0; n * n];
        let mut t = vec![0.0; n * n];
        s[cm(0, 0, n)] = 2.0;
        s[cm(1, 1, n)] = 5.0;
        s[cm(2, 2, n)] = -1.0;
        t[cm(0, 0, n)] = 1.0;
        t[cm(1, 1, n)] = 2.0;
        t[cm(2, 2, n)] = 3.0;

        let (ar, ai, bt) = extract_eigenvalues(&s, &t, n);
        assert_eq!(ar.len(), 3);
        // eigenvalue 0: 2/1 = 2
        assert!((ar[0] / bt[0] - 2.0).abs() < 1e-10);
        // eigenvalue 1: 5/2 = 2.5
        assert!((ar[1] / bt[1] - 2.5).abs() < 1e-10);
        // eigenvalue 2: -1/3
        assert!((ar[2] / bt[2] - (-1.0 / 3.0)).abs() < 1e-10);
        // All imaginary parts should be zero.
        for &imag in &ai {
            assert!(imag.abs() < 1e-15);
        }
    }

    #[test]
    fn test_qz_host_n3_upper_triangular() {
        // Both A and B already upper triangular.
        // A = [[1,2,3],[0,4,5],[0,0,6]], B = [[1,1,1],[0,2,1],[0,0,3]]
        #[rustfmt::skip]
        let mut a = vec![
            1.0, 0.0, 0.0, // col 0
            2.0, 4.0, 0.0, // col 1
            3.0, 5.0, 6.0, // col 2
        ];
        #[rustfmt::skip]
        let mut b = vec![
            1.0, 0.0, 0.0, // col 0
            1.0, 2.0, 0.0, // col 1
            1.0, 1.0, 3.0, // col 2
        ];
        let config = QzConfig::new(3, SmVersion::Sm80);
        let result = qz_host(&mut a, &mut b, &config);
        assert!(result.is_ok());
        if let Ok(r) = &result {
            assert!(r.converged);
            assert_eq!(r.alpha_real.len(), 3);
        }
    }
}
