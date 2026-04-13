//! Krylov subspace methods for sparse eigenvalue computation.
//!
//! This module provides GPU-accelerated Lanczos and Arnoldi iteration
//! for computing extreme eigenvalues and eigenvectors of large sparse matrices.
//!
//! - [`LanczosPlan`] -- Lanczos iteration for symmetric matrices, producing a
//!   tridiagonal matrix whose eigenvalues approximate those of the original matrix.
//! - [`ArnoldiPlan`] -- Arnoldi iteration for general (non-symmetric) matrices,
//!   producing an upper Hessenberg matrix.
//!
//! Both methods rely on repeated SpMV (sparse matrix-vector multiplication)
//! as the core computational primitive, combined with vector orthogonalization
//! kernels generated as PTX at runtime.

use oxicuda_ptx::prelude::*;

use crate::error::{SparseError, SparseResult};
use crate::ptx_helpers::{
    emit_warp_reduce_sum, load_float_imm, load_global_float, mul_float, store_global_float,
};

// ---------------------------------------------------------------------------
// Common types
// ---------------------------------------------------------------------------

/// Specifies which eigenvalues to target in Krylov iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EigenTarget {
    /// Eigenvalues with the largest absolute value.
    LargestMagnitude,
    /// Eigenvalues with the smallest absolute value.
    SmallestMagnitude,
    /// Eigenvalues with the largest real part (algebraic maximum for symmetric).
    LargestAlgebraic,
    /// Eigenvalues with the smallest real part (algebraic minimum for symmetric).
    SmallestAlgebraic,
}

/// Default block size for Krylov vector operations.
pub const KRYLOV_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// Lanczos iteration (symmetric matrices)
// ---------------------------------------------------------------------------

/// Configuration for Lanczos iteration on symmetric sparse matrices.
#[derive(Debug, Clone)]
pub struct LanczosConfig {
    /// Maximum Krylov subspace dimension (number of Lanczos steps).
    pub max_iterations: usize,
    /// Convergence tolerance for eigenvalue residuals.
    pub tolerance: f64,
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Which eigenvalues to target.
    pub which: EigenTarget,
}

/// Result of a Lanczos iteration.
#[derive(Debug, Clone)]
pub struct LanczosResult {
    /// Converged eigenvalues (sorted according to the target).
    pub eigenvalues: Vec<f64>,
    /// Diagonal of the tridiagonal matrix T (alpha coefficients).
    pub alpha: Vec<f64>,
    /// Sub-diagonal of the tridiagonal matrix T (beta coefficients).
    pub beta: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the iteration converged within tolerance.
    pub converged: bool,
}

/// Lanczos iteration plan for symmetric sparse eigenvalue problems.
///
/// Generates PTX kernels for each phase of the Lanczos recurrence:
/// 1. SpMV: `w = A * v_j` (delegated to the SpMV module)
/// 2. Dot product: `alpha_j = w . v_j`
/// 3. Orthogonalization: `w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}`
/// 4. Norm computation: `beta_j = ||w||`
/// 5. Normalization: `v_{j+1} = w / beta_j`
///
/// Additionally provides full reorthogonalization to maintain numerical stability.
#[derive(Debug)]
pub struct LanczosPlan {
    config: LanczosConfig,
    /// Dimension of the matrix (n x n).
    n: usize,
}

impl LanczosPlan {
    /// Creates a new Lanczos plan for an n x n symmetric matrix.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] if configuration is invalid:
    /// - `n == 0`
    /// - `num_eigenvalues == 0`
    /// - `max_iterations < num_eigenvalues`
    /// - `max_iterations > n`
    /// - `tolerance <= 0`
    pub fn new(config: LanczosConfig, n: usize) -> SparseResult<Self> {
        if n == 0 {
            return Err(SparseError::InvalidArgument(
                "matrix dimension n must be positive".to_string(),
            ));
        }
        if config.num_eigenvalues == 0 {
            return Err(SparseError::InvalidArgument(
                "num_eigenvalues must be positive".to_string(),
            ));
        }
        if config.max_iterations < config.num_eigenvalues {
            return Err(SparseError::InvalidArgument(format!(
                "max_iterations ({}) must be >= num_eigenvalues ({})",
                config.max_iterations, config.num_eigenvalues
            )));
        }
        if config.max_iterations > n {
            return Err(SparseError::InvalidArgument(format!(
                "max_iterations ({}) must be <= matrix dimension n ({})",
                config.max_iterations, n
            )));
        }
        if config.tolerance <= 0.0 {
            return Err(SparseError::InvalidArgument(
                "tolerance must be positive".to_string(),
            ));
        }

        Ok(Self { config, n })
    }

    /// Returns the configuration for this plan.
    #[inline]
    pub fn config(&self) -> &LanczosConfig {
        &self.config
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Returns the workspace size in bytes needed for f64 Lanczos vectors.
    ///
    /// The workspace must hold:
    /// - `max_iterations + 1` Lanczos vectors of dimension `n` (for reorthogonalization)
    /// - 1 work vector `w` of dimension `n`
    /// - `max_iterations` alpha values
    /// - `max_iterations` beta values
    pub fn workspace_bytes_f64(&self) -> usize {
        let k = self.config.max_iterations;
        let n = self.n;
        let vectors = (k + 2) * n * 8; // (k+1) Lanczos vecs + 1 work vec, f64
        let scalars = (k + k) * 8; // alpha + beta arrays
        vectors + scalars
    }

    /// Returns the workspace size in bytes needed for f32 Lanczos vectors.
    pub fn workspace_bytes_f32(&self) -> usize {
        let k = self.config.max_iterations;
        let n = self.n;
        let vectors = (k + 2) * n * 4;
        let scalars = (k + k) * 4;
        vectors + scalars
    }

    /// Generates PTX for a single Lanczos step kernel (f64).
    ///
    /// The kernel performs the orthogonalization and normalization phases
    /// of one Lanczos iteration:
    ///
    /// ```text
    /// // Input: w = A * v_j (SpMV already done)
    /// alpha_j = dot(w, v_j)                                  // dot product
    /// w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}           // orthogonalize
    /// beta_j = ||w||                                          // norm
    /// v_{j+1} = w / beta_j                                   // normalize
    /// ```
    ///
    /// # Kernel Parameters
    ///
    /// - `w_ptr`: device pointer to work vector w (length n), modified in-place
    /// - `v_j_ptr`: device pointer to current Lanczos vector v_j (length n)
    /// - `v_jm1_ptr`: device pointer to previous Lanczos vector v_{j-1} (length n)
    /// - `v_jp1_ptr`: device pointer to output vector v_{j+1} (length n)
    /// - `alpha_ptr`: device pointer to output scalar alpha_j
    /// - `beta_prev`: the previous beta_{j-1} value (as bits)
    /// - `beta_out_ptr`: device pointer to output scalar beta_j
    /// - `n`: vector length
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
    pub fn generate_lanczos_step_ptx(&self) -> SparseResult<String> {
        emit_lanczos_step_f64(self.n)
    }

    /// Generates PTX for a single Lanczos step kernel (f32).
    ///
    /// Same semantics as [`generate_lanczos_step_ptx`](Self::generate_lanczos_step_ptx)
    /// but for single-precision floating point.
    pub fn generate_lanczos_step_ptx_f32(&self) -> SparseResult<String> {
        emit_lanczos_step_f32(self.n)
    }

    /// Generates PTX for full reorthogonalization against all previous Lanczos vectors.
    ///
    /// This kernel applies modified Gram-Schmidt orthogonalization of `w`
    /// against the first `j` Lanczos vectors stored column-major in `V`.
    ///
    /// ```text
    /// for i in 0..j:
    ///     h = dot(w, V[:, i])
    ///     w = w - h * V[:, i]
    /// ```
    ///
    /// Each thread handles one element of `w` and iterates over all `j` vectors,
    /// using warp reductions for the dot products.
    ///
    /// # Kernel Parameters
    ///
    /// - `w_ptr`: device pointer to vector to orthogonalize (length n), modified in-place
    /// - `v_basis_ptr`: device pointer to basis matrix V stored column-major (n x j)
    /// - `coeffs_ptr`: device pointer to output coefficients h_i (length j)
    /// - `num_vecs`: number of basis vectors j to orthogonalize against
    /// - `n`: vector length
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
    pub fn generate_reorthogonalize_ptx(&self) -> SparseResult<String> {
        emit_reorthogonalize_f64(self.n)
    }

    /// Generates PTX for full reorthogonalization (f32 variant).
    pub fn generate_reorthogonalize_ptx_f32(&self) -> SparseResult<String> {
        emit_reorthogonalize_f32(self.n)
    }

    /// Generates PTX for the dot product reduction kernel (f64).
    ///
    /// Used to compute `alpha_j = dot(w, v_j)` in the Lanczos recurrence.
    pub fn generate_dot_product_ptx(&self) -> SparseResult<String> {
        emit_dot_product_reduce_f64(self.n)
    }

    /// Generates PTX for the dot product reduction kernel (f32).
    pub fn generate_dot_product_ptx_f32(&self) -> SparseResult<String> {
        emit_dot_product_reduce_f32(self.n)
    }

    /// Generates PTX for the vector norm-squared reduction kernel (f64).
    ///
    /// Used to compute `beta_j = ||w||` (caller takes sqrt of the result).
    pub fn generate_norm_sq_ptx(&self) -> SparseResult<String> {
        emit_norm_sq_reduce_f64(self.n)
    }

    /// Generates PTX for the vector norm-squared reduction kernel (f32).
    pub fn generate_norm_sq_ptx_f32(&self) -> SparseResult<String> {
        emit_norm_sq_reduce_f32(self.n)
    }
}

// ---------------------------------------------------------------------------
// Arnoldi iteration (general matrices)
// ---------------------------------------------------------------------------

/// Configuration for Arnoldi iteration on general sparse matrices.
#[derive(Debug, Clone)]
pub struct ArnoldiConfig {
    /// Maximum Krylov subspace dimension (number of Arnoldi steps).
    pub max_iterations: usize,
    /// Convergence tolerance for eigenvalue residuals.
    pub tolerance: f64,
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Which eigenvalues to target.
    pub which: EigenTarget,
}

/// Result of an Arnoldi iteration.
#[derive(Debug, Clone)]
pub struct ArnoldiResult {
    /// Converged eigenvalues as (real, imaginary) pairs.
    pub eigenvalues: Vec<(f64, f64)>,
    /// Upper Hessenberg matrix H (stored as row-major dense Vec\<Vec\<`f64`\>\>).
    pub hessenberg: Vec<Vec<f64>>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the iteration converged within tolerance.
    pub converged: bool,
}

/// Arnoldi iteration plan for general sparse eigenvalue problems.
///
/// Generates PTX kernels for each phase of the Arnoldi recurrence:
/// 1. SpMV: `w = A * v_j` (delegated to the SpMV module)
/// 2. Modified Gram-Schmidt: for i = 0..j: `h_{i,j} = w . v_i`, `w = w - h_{i,j} * v_i`
/// 3. Norm and normalize: `h_{j+1,j} = ||w||`, `v_{j+1} = w / h_{j+1,j}`
///
/// Unlike Lanczos (which only orthogonalizes against 2 vectors), Arnoldi
/// orthogonalizes against all previous basis vectors at every step.
#[derive(Debug)]
pub struct ArnoldiPlan {
    config: ArnoldiConfig,
    /// Dimension of the matrix (n x n).
    n: usize,
}

impl ArnoldiPlan {
    /// Creates a new Arnoldi plan for an n x n general matrix.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidArgument`] if configuration is invalid:
    /// - `n == 0`
    /// - `num_eigenvalues == 0`
    /// - `max_iterations < num_eigenvalues`
    /// - `max_iterations > n`
    /// - `tolerance <= 0`
    pub fn new(config: ArnoldiConfig, n: usize) -> SparseResult<Self> {
        if n == 0 {
            return Err(SparseError::InvalidArgument(
                "matrix dimension n must be positive".to_string(),
            ));
        }
        if config.num_eigenvalues == 0 {
            return Err(SparseError::InvalidArgument(
                "num_eigenvalues must be positive".to_string(),
            ));
        }
        if config.max_iterations < config.num_eigenvalues {
            return Err(SparseError::InvalidArgument(format!(
                "max_iterations ({}) must be >= num_eigenvalues ({})",
                config.max_iterations, config.num_eigenvalues
            )));
        }
        if config.max_iterations > n {
            return Err(SparseError::InvalidArgument(format!(
                "max_iterations ({}) must be <= matrix dimension n ({})",
                config.max_iterations, n
            )));
        }
        if config.tolerance <= 0.0 {
            return Err(SparseError::InvalidArgument(
                "tolerance must be positive".to_string(),
            ));
        }

        Ok(Self { config, n })
    }

    /// Returns the configuration for this plan.
    #[inline]
    pub fn config(&self) -> &ArnoldiConfig {
        &self.config
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Returns the workspace size in bytes needed for f64 Arnoldi vectors.
    ///
    /// The workspace must hold:
    /// - `max_iterations + 1` Arnoldi vectors of dimension `n`
    /// - 1 work vector `w` of dimension `n`
    /// - `(max_iterations + 1) x max_iterations` Hessenberg matrix H
    pub fn workspace_bytes_f64(&self) -> usize {
        let k = self.config.max_iterations;
        let n = self.n;
        let vectors = (k + 2) * n * 8; // (k+1) basis vecs + 1 work vec
        let hessenberg = (k + 1) * k * 8; // H is (k+1) x k
        vectors + hessenberg
    }

    /// Returns the workspace size in bytes needed for f32 Arnoldi vectors.
    pub fn workspace_bytes_f32(&self) -> usize {
        let k = self.config.max_iterations;
        let n = self.n;
        let vectors = (k + 2) * n * 4;
        let hessenberg = (k + 1) * k * 4;
        vectors + hessenberg
    }

    /// Generates PTX for a single Arnoldi step kernel (f64).
    ///
    /// The kernel performs the modified Gram-Schmidt orthogonalization
    /// and normalization of one Arnoldi iteration step:
    ///
    /// ```text
    /// // Input: w = A * v_j (SpMV already done)
    /// for i in 0..j:
    ///     h_{i,j} = dot(w, v_i)
    ///     w = w - h_{i,j} * v_i
    /// h_{j+1,j} = ||w||
    /// v_{j+1} = w / h_{j+1,j}
    /// ```
    ///
    /// # Kernel Parameters
    ///
    /// - `w_ptr`: device pointer to work vector w (length n), modified in-place
    /// - `v_basis_ptr`: device pointer to basis matrix V column-major (n x (j+1))
    /// - `h_col_ptr`: device pointer to Hessenberg column h_{:,j} (length j+1)
    /// - `v_jp1_ptr`: device pointer to output vector v_{j+1} (length n)
    /// - `j`: current iteration index (number of existing basis vectors)
    /// - `n`: vector length
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
    pub fn generate_arnoldi_step_ptx(&self) -> SparseResult<String> {
        emit_arnoldi_step_f64(self.n)
    }

    /// Generates PTX for a single Arnoldi step kernel (f32).
    pub fn generate_arnoldi_step_ptx_f32(&self) -> SparseResult<String> {
        emit_arnoldi_step_f32(self.n)
    }

    /// Generates PTX for modified Gram-Schmidt orthogonalization kernel (f64).
    ///
    /// This is the inner loop of Arnoldi: orthogonalize `w` against all
    /// existing basis vectors using modified Gram-Schmidt. This kernel
    /// uses warp-level reductions for the dot products.
    ///
    /// # Kernel Parameters
    ///
    /// - `w_ptr`: device pointer to vector to orthogonalize (length n)
    /// - `v_basis_ptr`: device pointer to basis matrix V column-major (n x j)
    /// - `coeffs_ptr`: device pointer to output coefficients h_{i,j} (length j)
    /// - `num_vecs`: number of basis vectors j
    /// - `n`: vector length
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
    pub fn generate_gram_schmidt_ptx(&self) -> SparseResult<String> {
        emit_gram_schmidt_f64(self.n)
    }

    /// Generates PTX for modified Gram-Schmidt orthogonalization kernel (f32).
    pub fn generate_gram_schmidt_ptx_f32(&self) -> SparseResult<String> {
        emit_gram_schmidt_f32(self.n)
    }

    /// Generates PTX for the dot product reduction kernel (f64).
    ///
    /// Used to compute `h_{i,j} = dot(w, v_i)` in the Arnoldi recurrence.
    pub fn generate_dot_product_ptx(&self) -> SparseResult<String> {
        emit_dot_product_reduce_f64(self.n)
    }

    /// Generates PTX for the dot product reduction kernel (f32).
    pub fn generate_dot_product_ptx_f32(&self) -> SparseResult<String> {
        emit_dot_product_reduce_f32(self.n)
    }

    /// Generates PTX for the vector norm-squared reduction kernel (f64).
    ///
    /// Used to compute `h_{j+1,j} = ||w||` (caller takes sqrt of the result).
    pub fn generate_norm_sq_ptx(&self) -> SparseResult<String> {
        emit_norm_sq_reduce_f64(self.n)
    }

    /// Generates PTX for the vector norm-squared reduction kernel (f32).
    pub fn generate_norm_sq_ptx_f32(&self) -> SparseResult<String> {
        emit_norm_sq_reduce_f32(self.n)
    }
}

// ---------------------------------------------------------------------------
// PTX emission: Lanczos step
// ---------------------------------------------------------------------------

/// Emits PTX for the Lanczos orthogonalization + normalization step (f64).
///
/// Kernel: `lanczos_step_f64`
///
/// Each thread handles one element of the vectors. The kernel:
/// 1. Computes partial dot product `w[tid] * v_j[tid]` (summed via warp reduce + atomics)
/// 2. Orthogonalizes `w[tid] -= alpha * v_j[tid] + beta_prev * v_jm1[tid]`
/// 3. Computes partial norm `w[tid]^2` (summed via warp reduce + atomics)
/// 4. Normalizes `v_jp1[tid] = w[tid] / beta`
///
/// Because dot product and norm require global synchronization, this kernel
/// is designed for a two-pass approach: pass 1 computes alpha (dot), pass 2
/// computes orthogonalization + beta (norm) + normalization.
fn emit_lanczos_step_f64(n: usize) -> SparseResult<String> {
    emit_lanczos_step_typed::<f64>(n, "lanczos_step_f64")
}

/// Emits PTX for the Lanczos step (f32).
fn emit_lanczos_step_f32(n: usize) -> SparseResult<String> {
    emit_lanczos_step_typed::<f32>(n, "lanczos_step_f32")
}

/// Generic Lanczos step PTX emitter.
///
/// This kernel handles the orthogonalization pass:
/// `w[i] = w[i] - alpha * v_j[i] - beta_prev * v_jm1[i]`
/// and then normalizes: `v_jp1[i] = w[i] / beta_j`
///
/// The dot products (alpha computation) and norm (beta computation) are
/// handled by separate reduction kernels launched from the host.
fn emit_lanczos_step_typed<T: oxicuda_blas::GpuFloat>(
    _n: usize,
    kernel_name: &str,
) -> SparseResult<String> {
    let is_f64 = T::SIZE == 8;
    let elem_bytes = T::size_u32();
    let mov_suffix = if is_f64 { "f64" } else { "f32" };

    KernelBuilder::new(kernel_name)
        .target(SmVersion::Sm80)
        .param("w_ptr", PtxType::U64)
        .param("v_j_ptr", PtxType::U64)
        .param("v_jm1_ptr", PtxType::U64)
        .param("v_jp1_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_prev_bits", PtxType::U64)
        .param("beta_j_bits", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_param = b.load_param_u32("n");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, n_param, move |b| {
                let tid = gid_inner;
                let w_ptr = b.load_param_u64("w_ptr");
                let v_j_ptr = b.load_param_u64("v_j_ptr");
                let v_jm1_ptr = b.load_param_u64("v_jm1_ptr");
                let v_jp1_ptr = b.load_param_u64("v_jp1_ptr");
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_prev_bits = b.load_param_u64("beta_prev_bits");
                let beta_j_bits = b.load_param_u64("beta_j_bits");

                let alpha = reinterpret_bits::<T>(b, alpha_bits);
                let beta_prev = reinterpret_bits::<T>(b, beta_prev_bits);
                let beta_j = reinterpret_bits::<T>(b, beta_j_bits);

                // Load w[tid], v_j[tid], v_jm1[tid]
                let w_addr = b.byte_offset_addr(w_ptr, tid.clone(), elem_bytes);
                let w_val = load_global_float::<T>(b, w_addr.clone());

                let vj_addr = b.byte_offset_addr(v_j_ptr, tid.clone(), elem_bytes);
                let vj_val = load_global_float::<T>(b, vj_addr);

                let vjm1_addr = b.byte_offset_addr(v_jm1_ptr, tid.clone(), elem_bytes);
                let vjm1_val = load_global_float::<T>(b, vjm1_addr);

                // Orthogonalize: w[i] = w[i] - alpha * v_j[i] - beta_prev * v_jm1[i]
                let alpha_vj = mul_float::<T>(b, alpha, vj_val);
                let beta_vjm1 = mul_float::<T>(b, beta_prev, vjm1_val);
                let sub1 = sub_float::<T>(b, w_val, alpha_vj);
                let w_orth = sub_float::<T>(b, sub1, beta_vjm1);

                // Store orthogonalized w
                store_global_float::<T>(b, w_addr, w_orth.clone());

                // Normalize: v_jp1[i] = w_orth[i] / beta_j
                let v_jp1_val = div_float::<T>(b, w_orth, beta_j);
                let vjp1_addr = b.byte_offset_addr(v_jp1_ptr, tid, elem_bytes);
                store_global_float::<T>(b, vjp1_addr, v_jp1_val);
            });

            // Suppress unused variable warning on mov_suffix
            let _ = mov_suffix;

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

// ---------------------------------------------------------------------------
// PTX emission: Reorthogonalization
// ---------------------------------------------------------------------------

/// Emits PTX for full reorthogonalization kernel (f64).
fn emit_reorthogonalize_f64(n: usize) -> SparseResult<String> {
    emit_reorthogonalize_typed::<f64>(n, "reorthogonalize_f64")
}

/// Emits PTX for full reorthogonalization kernel (f32).
fn emit_reorthogonalize_f32(n: usize) -> SparseResult<String> {
    emit_reorthogonalize_typed::<f32>(n, "reorthogonalize_f32")
}

/// Generic reorthogonalization PTX emitter.
///
/// For each basis vector `v_i` (i in 0..num_vecs):
///   1. Compute partial dot: `w[tid] * v_i[tid]` -> warp reduce -> atomic add to coeffs[i]
///   2. Subtract projection: `w[tid] -= coeff * v_i[tid]`
///
/// This requires a grid-wide sync between steps 1 and 2 for each vector,
/// so the kernel processes one basis vector per launch. The host loops
/// over basis vectors, launching this kernel `num_vecs` times.
///
/// Kernel: `reorthogonalize_{f32,f64}`
/// Params: w_ptr, v_i_ptr, dot_result_ptr, n
fn emit_reorthogonalize_typed<T: oxicuda_blas::GpuFloat>(
    _n: usize,
    kernel_name: &str,
) -> SparseResult<String> {
    let elem_bytes = T::size_u32();

    KernelBuilder::new(kernel_name)
        .target(SmVersion::Sm80)
        .param("w_ptr", PtxType::U64)
        .param("v_i_ptr", PtxType::U64)
        .param("coeff_bits", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_param = b.load_param_u32("n");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, n_param, move |b| {
                let tid = gid_inner;
                let w_ptr = b.load_param_u64("w_ptr");
                let v_i_ptr = b.load_param_u64("v_i_ptr");
                let coeff_bits = b.load_param_u64("coeff_bits");

                let coeff = reinterpret_bits::<T>(b, coeff_bits);

                // Load w[tid] and v_i[tid]
                let w_addr = b.byte_offset_addr(w_ptr, tid.clone(), elem_bytes);
                let w_val = load_global_float::<T>(b, w_addr.clone());

                let vi_addr = b.byte_offset_addr(v_i_ptr, tid, elem_bytes);
                let vi_val = load_global_float::<T>(b, vi_addr);

                // w[tid] -= coeff * v_i[tid]
                let proj = mul_float::<T>(b, coeff, vi_val);
                let w_new = sub_float::<T>(b, w_val, proj);

                store_global_float::<T>(b, w_addr, w_new);
            });

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

// ---------------------------------------------------------------------------
// PTX emission: Arnoldi step
// ---------------------------------------------------------------------------

/// Emits PTX for the Arnoldi orthogonalization step (f64).
fn emit_arnoldi_step_f64(n: usize) -> SparseResult<String> {
    emit_arnoldi_step_typed::<f64>(n, "arnoldi_step_f64")
}

/// Emits PTX for the Arnoldi step (f32).
fn emit_arnoldi_step_f32(n: usize) -> SparseResult<String> {
    emit_arnoldi_step_typed::<f32>(n, "arnoldi_step_f32")
}

/// Generic Arnoldi step PTX emitter.
///
/// The Arnoldi step kernel handles the normalization phase:
/// `v_jp1[i] = w[i] / h_{j+1,j}`
///
/// The modified Gram-Schmidt orthogonalization (computing h_{i,j} and
/// subtracting projections) is handled by the separate Gram-Schmidt kernel,
/// which is launched once per basis vector from the host.
///
/// Kernel: `arnoldi_step_{f32,f64}`
/// Params: w_ptr, v_jp1_ptr, h_jp1_j_bits (norm), n
fn emit_arnoldi_step_typed<T: oxicuda_blas::GpuFloat>(
    _n: usize,
    kernel_name: &str,
) -> SparseResult<String> {
    let elem_bytes = T::size_u32();

    KernelBuilder::new(kernel_name)
        .target(SmVersion::Sm80)
        .param("w_ptr", PtxType::U64)
        .param("v_jp1_ptr", PtxType::U64)
        .param("h_jp1_j_bits", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_param = b.load_param_u32("n");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, n_param, move |b| {
                let tid = gid_inner;
                let w_ptr = b.load_param_u64("w_ptr");
                let v_jp1_ptr = b.load_param_u64("v_jp1_ptr");
                let h_bits = b.load_param_u64("h_jp1_j_bits");

                let h_jp1_j = reinterpret_bits::<T>(b, h_bits);

                // Load w[tid]
                let w_addr = b.byte_offset_addr(w_ptr, tid.clone(), elem_bytes);
                let w_val = load_global_float::<T>(b, w_addr);

                // v_jp1[tid] = w[tid] / h_{j+1,j}
                let v_new = div_float::<T>(b, w_val, h_jp1_j);
                let vjp1_addr = b.byte_offset_addr(v_jp1_ptr, tid, elem_bytes);
                store_global_float::<T>(b, vjp1_addr, v_new);
            });

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

// ---------------------------------------------------------------------------
// PTX emission: Modified Gram-Schmidt
// ---------------------------------------------------------------------------

/// Emits PTX for modified Gram-Schmidt projection kernel (f64).
fn emit_gram_schmidt_f64(n: usize) -> SparseResult<String> {
    emit_gram_schmidt_typed::<f64>(n, "gram_schmidt_f64")
}

/// Emits PTX for modified Gram-Schmidt projection kernel (f32).
fn emit_gram_schmidt_f32(n: usize) -> SparseResult<String> {
    emit_gram_schmidt_typed::<f32>(n, "gram_schmidt_f32")
}

/// Generic modified Gram-Schmidt PTX emitter.
///
/// This kernel subtracts the projection of `w` onto a single basis vector `v_i`:
///   `w[tid] -= h_{i,j} * v_i[tid]`
///
/// The dot product `h_{i,j} = dot(w, v_i)` is computed separately via a
/// reduction kernel. The host launches this kernel once per basis vector
/// in the modified Gram-Schmidt loop: for i in 0..j.
///
/// Kernel: `gram_schmidt_{f32,f64}`
/// Params: w_ptr, v_i_ptr, h_ij_bits, n
fn emit_gram_schmidt_typed<T: oxicuda_blas::GpuFloat>(
    _n: usize,
    kernel_name: &str,
) -> SparseResult<String> {
    let elem_bytes = T::size_u32();

    KernelBuilder::new(kernel_name)
        .target(SmVersion::Sm80)
        .param("w_ptr", PtxType::U64)
        .param("v_i_ptr", PtxType::U64)
        .param("h_ij_bits", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_param = b.load_param_u32("n");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, n_param, move |b| {
                let tid = gid_inner;
                let w_ptr = b.load_param_u64("w_ptr");
                let v_i_ptr = b.load_param_u64("v_i_ptr");
                let h_bits = b.load_param_u64("h_ij_bits");

                let h_ij = reinterpret_bits::<T>(b, h_bits);

                // Load w[tid] and v_i[tid]
                let w_addr = b.byte_offset_addr(w_ptr, tid.clone(), elem_bytes);
                let w_val = load_global_float::<T>(b, w_addr.clone());

                let vi_addr = b.byte_offset_addr(v_i_ptr, tid, elem_bytes);
                let vi_val = load_global_float::<T>(b, vi_addr);

                // w[tid] -= h_{i,j} * v_i[tid]
                let proj = mul_float::<T>(b, h_ij, vi_val);
                let w_new = sub_float::<T>(b, w_val, proj);

                store_global_float::<T>(b, w_addr, w_new);
            });

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

// ---------------------------------------------------------------------------
// PTX emission: Dot product reduction
// ---------------------------------------------------------------------------

/// Emits PTX for a warp-level partial dot product kernel.
///
/// Each warp computes a partial dot product of two vectors and the first
/// lane atomically adds the result to a global accumulator. This is used
/// by both Lanczos (alpha computation) and Arnoldi (h_{i,j} computation).
///
/// Kernel: `dot_product_reduce_{f32,f64}`
/// Params: a_ptr, b_ptr, result_ptr, n
fn emit_dot_product_reduce_f64(_n: usize) -> SparseResult<String> {
    emit_dot_product_reduce_typed::<f64>("dot_product_reduce_f64")
}

fn emit_dot_product_reduce_f32(_n: usize) -> SparseResult<String> {
    emit_dot_product_reduce_typed::<f32>("dot_product_reduce_f32")
}

fn emit_dot_product_reduce_typed<T: oxicuda_blas::GpuFloat>(
    kernel_name: &str,
) -> SparseResult<String> {
    let elem_bytes = T::size_u32();

    KernelBuilder::new(kernel_name)
        .target(SmVersion::Sm80)
        .param("a_ptr", PtxType::U64)
        .param("b_ptr", PtxType::U64)
        .param("result_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_param = b.load_param_u32("n");

            // Save gid for lane computation after the closure
            let gid_for_lane = gid.clone();

            // Each thread computes a[gid] * b[gid] if in bounds, else 0
            let prod = load_float_imm::<T>(b, 0.0);

            let gid_inner = gid.clone();
            let prod_inner = prod.clone();
            b.if_lt_u32(gid, n_param, move |b| {
                let tid = gid_inner;
                let a_ptr = b.load_param_u64("a_ptr");
                let b_ptr_reg = b.load_param_u64("b_ptr");

                let a_addr = b.byte_offset_addr(a_ptr, tid.clone(), elem_bytes);
                let a_val = load_global_float::<T>(b, a_addr);

                let b_addr = b.byte_offset_addr(b_ptr_reg, tid, elem_bytes);
                let b_val = load_global_float::<T>(b, b_addr);

                let p = mul_float::<T>(b, a_val, b_val);
                let suffix = if T::SIZE == 8 { "f64" } else { "f32" };
                b.raw_ptx(&format!("mov.{suffix} {prod_inner}, {p};"));
            });

            // Warp reduce
            let reduced = emit_warp_reduce_sum::<T>(b, prod);

            // Lane 0 atomically adds to result
            let lane = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("and.b32 {lane}, {gid_for_lane}, 31;"));

            let is_lane_0 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.eq.u32 {is_lane_0}, {lane}, 0;"));

            let skip_label = b.fresh_label("dot_skip");
            b.raw_ptx(&format!("@!{is_lane_0} bra {skip_label};"));

            let result_ptr = b.load_param_u64("result_ptr");
            crate::ptx_helpers::emit_atomic_add_float::<T>(b, result_ptr, reduced);

            b.label(&skip_label);

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

// ---------------------------------------------------------------------------
// PTX emission: Vector norm (squared) reduction
// ---------------------------------------------------------------------------

/// Emits PTX for warp-level vector norm-squared reduction kernel.
///
/// Each warp computes partial ||v||^2 and lane 0 atomically adds to result.
/// The host takes sqrt of the accumulated result to get the 2-norm.
///
/// Kernel: `norm_sq_reduce_{f32,f64}`
/// Params: v_ptr, result_ptr, n
fn emit_norm_sq_reduce_f64(_n: usize) -> SparseResult<String> {
    emit_norm_sq_reduce_typed::<f64>("norm_sq_reduce_f64")
}

fn emit_norm_sq_reduce_f32(_n: usize) -> SparseResult<String> {
    emit_norm_sq_reduce_typed::<f32>("norm_sq_reduce_f32")
}

fn emit_norm_sq_reduce_typed<T: oxicuda_blas::GpuFloat>(kernel_name: &str) -> SparseResult<String> {
    let elem_bytes = T::size_u32();

    KernelBuilder::new(kernel_name)
        .target(SmVersion::Sm80)
        .param("v_ptr", PtxType::U64)
        .param("result_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_param = b.load_param_u32("n");

            // Save gid for lane computation after the closure
            let gid_for_lane = gid.clone();

            let sq = load_float_imm::<T>(b, 0.0);

            let gid_inner = gid.clone();
            let sq_inner = sq.clone();
            b.if_lt_u32(gid, n_param, move |b| {
                let tid = gid_inner;
                let v_ptr = b.load_param_u64("v_ptr");

                let v_addr = b.byte_offset_addr(v_ptr, tid, elem_bytes);
                let v_val = load_global_float::<T>(b, v_addr);

                let p = mul_float::<T>(b, v_val.clone(), v_val);
                let suffix = if T::SIZE == 8 { "f64" } else { "f32" };
                b.raw_ptx(&format!("mov.{suffix} {sq_inner}, {p};"));
            });

            // Warp reduce
            let reduced = emit_warp_reduce_sum::<T>(b, sq);

            // Lane 0 atomically adds
            let lane = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("and.b32 {lane}, {gid_for_lane}, 31;"));

            let is_lane_0 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.eq.u32 {is_lane_0}, {lane}, 0;"));

            let skip_label = b.fresh_label("norm_skip");
            b.raw_ptx(&format!("@!{is_lane_0} bra {skip_label};"));

            let result_ptr = b.load_param_u64("result_ptr");
            crate::ptx_helpers::emit_atomic_add_float::<T>(b, result_ptr, reduced);

            b.label(&skip_label);

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

// ---------------------------------------------------------------------------
// Helper: float arithmetic
// ---------------------------------------------------------------------------

/// Reinterprets u64 bits as the float type T (like ptx_helpers::reinterpret_bits_to_float).
fn reinterpret_bits<T: oxicuda_blas::GpuFloat>(
    b: &mut BodyBuilder<'_>,
    bits: Register,
) -> Register {
    crate::ptx_helpers::reinterpret_bits_to_float::<T>(b, bits)
}

/// Emits a subtraction: `dst = a - bv`.
fn sub_float<T: oxicuda_blas::GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let dst = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("sub.rn.f32 {dst}, {a}, {bv};"));
        dst
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("sub.rn.f64 {dst}, {a}, {bv};"));
        dst
    }
}

/// Emits a division: `dst = a / bv`.
fn div_float<T: oxicuda_blas::GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let dst = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("div.rn.f32 {dst}, {a}, {bv};"));
        dst
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("div.rn.f64 {dst}, {a}, {bv};"));
        dst
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Lanczos config validation tests --

    #[test]
    fn lanczos_new_valid_config() {
        let config = LanczosConfig {
            max_iterations: 50,
            tolerance: 1e-10,
            num_eigenvalues: 5,
            which: EigenTarget::LargestMagnitude,
        };
        let plan = LanczosPlan::new(config, 100);
        assert!(plan.is_ok());
        let plan = plan.expect("test: valid config should succeed");
        assert_eq!(plan.dimension(), 100);
    }

    #[test]
    fn lanczos_rejects_zero_dimension() {
        let config = LanczosConfig {
            max_iterations: 10,
            tolerance: 1e-6,
            num_eigenvalues: 3,
            which: EigenTarget::SmallestMagnitude,
        };
        let result = LanczosPlan::new(config, 0);
        assert!(result.is_err());
        match result {
            Err(SparseError::InvalidArgument(msg)) => {
                assert!(msg.contains("dimension"));
            }
            other => panic!("expected InvalidArgument, got: {other:?}"),
        }
    }

    #[test]
    fn lanczos_rejects_zero_eigenvalues() {
        let config = LanczosConfig {
            max_iterations: 10,
            tolerance: 1e-6,
            num_eigenvalues: 0,
            which: EigenTarget::LargestAlgebraic,
        };
        let result = LanczosPlan::new(config, 100);
        assert!(result.is_err());
    }

    #[test]
    fn lanczos_rejects_iterations_less_than_eigenvalues() {
        let config = LanczosConfig {
            max_iterations: 3,
            tolerance: 1e-6,
            num_eigenvalues: 10,
            which: EigenTarget::SmallestAlgebraic,
        };
        let result = LanczosPlan::new(config, 100);
        assert!(matches!(result, Err(SparseError::InvalidArgument(_))));
    }

    #[test]
    fn lanczos_rejects_iterations_greater_than_n() {
        let config = LanczosConfig {
            max_iterations: 200,
            tolerance: 1e-6,
            num_eigenvalues: 5,
            which: EigenTarget::LargestMagnitude,
        };
        let result = LanczosPlan::new(config, 100);
        assert!(matches!(result, Err(SparseError::InvalidArgument(_))));
    }

    #[test]
    fn lanczos_rejects_non_positive_tolerance() {
        let config = LanczosConfig {
            max_iterations: 50,
            tolerance: 0.0,
            num_eigenvalues: 5,
            which: EigenTarget::LargestMagnitude,
        };
        let result = LanczosPlan::new(config, 100);
        assert!(matches!(result, Err(SparseError::InvalidArgument(_))));

        let config_neg = LanczosConfig {
            max_iterations: 50,
            tolerance: -1e-6,
            num_eigenvalues: 5,
            which: EigenTarget::LargestMagnitude,
        };
        let result_neg = LanczosPlan::new(config_neg, 100);
        assert!(matches!(result_neg, Err(SparseError::InvalidArgument(_))));
    }

    // -- Lanczos PTX generation tests --

    #[test]
    fn lanczos_step_ptx_f64_generates() {
        let config = LanczosConfig {
            max_iterations: 30,
            tolerance: 1e-10,
            num_eigenvalues: 5,
            which: EigenTarget::LargestMagnitude,
        };
        let plan = LanczosPlan::new(config, 1000).expect("test: valid config");
        let ptx = plan.generate_lanczos_step_ptx();
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry lanczos_step_f64"));
        assert!(ptx_str.contains(".target sm_80"));
        // Should reference the parameter names
        assert!(ptx_str.contains("w_ptr"));
        assert!(ptx_str.contains("v_j_ptr"));
    }

    #[test]
    fn lanczos_step_ptx_f32_generates() {
        let config = LanczosConfig {
            max_iterations: 20,
            tolerance: 1e-6,
            num_eigenvalues: 3,
            which: EigenTarget::SmallestMagnitude,
        };
        let plan = LanczosPlan::new(config, 500).expect("test: valid config");
        let ptx = plan.generate_lanczos_step_ptx_f32();
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry lanczos_step_f32"));
    }

    #[test]
    fn lanczos_reorthogonalize_ptx_generates() {
        let config = LanczosConfig {
            max_iterations: 30,
            tolerance: 1e-10,
            num_eigenvalues: 5,
            which: EigenTarget::LargestAlgebraic,
        };
        let plan = LanczosPlan::new(config, 1000).expect("test: valid config");
        let ptx = plan.generate_reorthogonalize_ptx();
        assert!(ptx.is_ok(), "Reorthogonalize PTX failed: {ptx:?}");
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry reorthogonalize_f64"));
        assert!(ptx_str.contains("w_ptr"));
    }

    // -- Arnoldi config validation tests --

    #[test]
    fn arnoldi_new_valid_config() {
        let config = ArnoldiConfig {
            max_iterations: 50,
            tolerance: 1e-10,
            num_eigenvalues: 5,
            which: EigenTarget::LargestMagnitude,
        };
        let plan = ArnoldiPlan::new(config, 200);
        assert!(plan.is_ok());
        let plan = plan.expect("test: valid config should succeed");
        assert_eq!(plan.dimension(), 200);
    }

    #[test]
    fn arnoldi_rejects_invalid_config() {
        // Zero dimension
        let config = ArnoldiConfig {
            max_iterations: 10,
            tolerance: 1e-6,
            num_eigenvalues: 3,
            which: EigenTarget::LargestMagnitude,
        };
        assert!(ArnoldiPlan::new(config, 0).is_err());

        // max_iterations > n
        let config2 = ArnoldiConfig {
            max_iterations: 500,
            tolerance: 1e-6,
            num_eigenvalues: 3,
            which: EigenTarget::SmallestMagnitude,
        };
        assert!(ArnoldiPlan::new(config2, 100).is_err());

        // num_eigenvalues > max_iterations
        let config3 = ArnoldiConfig {
            max_iterations: 5,
            tolerance: 1e-6,
            num_eigenvalues: 20,
            which: EigenTarget::LargestAlgebraic,
        };
        assert!(ArnoldiPlan::new(config3, 100).is_err());
    }

    // -- Arnoldi PTX generation tests --

    #[test]
    fn arnoldi_step_ptx_f64_generates() {
        let config = ArnoldiConfig {
            max_iterations: 30,
            tolerance: 1e-10,
            num_eigenvalues: 5,
            which: EigenTarget::LargestMagnitude,
        };
        let plan = ArnoldiPlan::new(config, 500).expect("test: valid config");
        let ptx = plan.generate_arnoldi_step_ptx();
        assert!(ptx.is_ok(), "Arnoldi PTX failed: {ptx:?}");
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry arnoldi_step_f64"));
        assert!(ptx_str.contains("w_ptr"));
    }

    #[test]
    fn arnoldi_step_ptx_f32_generates() {
        let config = ArnoldiConfig {
            max_iterations: 20,
            tolerance: 1e-6,
            num_eigenvalues: 3,
            which: EigenTarget::SmallestAlgebraic,
        };
        let plan = ArnoldiPlan::new(config, 300).expect("test: valid config");
        let ptx = plan.generate_arnoldi_step_ptx_f32();
        assert!(ptx.is_ok(), "Arnoldi f32 PTX failed: {ptx:?}");
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry arnoldi_step_f32"));
    }

    #[test]
    fn arnoldi_gram_schmidt_ptx_generates() {
        let config = ArnoldiConfig {
            max_iterations: 30,
            tolerance: 1e-10,
            num_eigenvalues: 5,
            which: EigenTarget::LargestMagnitude,
        };
        let plan = ArnoldiPlan::new(config, 500).expect("test: valid config");
        let ptx = plan.generate_gram_schmidt_ptx();
        assert!(ptx.is_ok(), "Gram-Schmidt PTX failed: {ptx:?}");
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry gram_schmidt_f64"));
    }

    // -- Workspace size tests --

    #[test]
    fn lanczos_workspace_size_f64() {
        let config = LanczosConfig {
            max_iterations: 50,
            tolerance: 1e-10,
            num_eigenvalues: 5,
            which: EigenTarget::LargestMagnitude,
        };
        let plan = LanczosPlan::new(config, 1000).expect("test: valid config");
        let ws = plan.workspace_bytes_f64();
        // (50+2) * 1000 * 8 + (50+50) * 8 = 52*8000 + 800 = 416000 + 800 = 416800
        assert_eq!(ws, 416_800);
    }

    #[test]
    fn lanczos_workspace_size_f32() {
        let config = LanczosConfig {
            max_iterations: 50,
            tolerance: 1e-10,
            num_eigenvalues: 5,
            which: EigenTarget::LargestMagnitude,
        };
        let plan = LanczosPlan::new(config, 1000).expect("test: valid config");
        let ws = plan.workspace_bytes_f32();
        // (50+2) * 1000 * 4 + (50+50) * 4 = 208000 + 400 = 208400
        assert_eq!(ws, 208_400);
    }

    #[test]
    fn arnoldi_workspace_size_f64() {
        let config = ArnoldiConfig {
            max_iterations: 30,
            tolerance: 1e-10,
            num_eigenvalues: 5,
            which: EigenTarget::LargestMagnitude,
        };
        let plan = ArnoldiPlan::new(config, 500).expect("test: valid config");
        let ws = plan.workspace_bytes_f64();
        // vectors: (30+2) * 500 * 8 = 128000
        // hessenberg: (30+1) * 30 * 8 = 7440
        // total = 135440
        assert_eq!(ws, 135_440);
    }

    // -- Tridiagonal structure tests --

    #[test]
    fn lanczos_result_tridiagonal_structure() {
        // Verify that a LanczosResult can represent a proper tridiagonal matrix
        let result = LanczosResult {
            eigenvalues: vec![5.0, 3.0, 1.0],
            alpha: vec![4.0, 3.5, 2.0, 1.5, 1.0], // diagonal
            beta: vec![1.2, 0.8, 0.5, 0.3],       // sub-diagonal (length = alpha.len() - 1)
            iterations: 5,
            converged: true,
        };
        // Tridiagonal matrix T is k x k where k = alpha.len()
        assert_eq!(result.alpha.len(), 5);
        assert_eq!(result.beta.len(), result.alpha.len() - 1);
        assert!(result.converged);
        assert_eq!(result.iterations, 5);
    }

    // -- Hessenberg structure tests --

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn arnoldi_result_hessenberg_structure() {
        // Verify that an ArnoldiResult can represent an upper Hessenberg matrix
        let k = 4;
        let mut h = vec![vec![0.0; k]; k + 1]; // (k+1) x k
        // Fill upper Hessenberg structure: h[i][j] != 0 only if i <= j+1
        for j in 0..k {
            for i in 0..=j + 1 {
                h[i][j] = (i + j + 1) as f64;
            }
        }
        // Verify sub-sub-diagonal is zero (i > j + 1)
        for j in 0..k {
            for i in (j + 2)..(k + 1) {
                assert!(
                    (h[i][j]).abs() < 1e-15,
                    "h[{i}][{j}] should be zero in upper Hessenberg"
                );
            }
        }

        let result = ArnoldiResult {
            eigenvalues: vec![(3.0, 0.5), (3.0, -0.5), (1.0, 0.0)],
            hessenberg: h,
            iterations: k,
            converged: true,
        };
        assert_eq!(result.hessenberg.len(), k + 1);
        assert_eq!(result.hessenberg[0].len(), k);
        assert!(result.converged);
        // Complex eigenvalues come in conjugate pairs
        let (r1, i1) = result.eigenvalues[0];
        let (r2, i2) = result.eigenvalues[1];
        assert!((r1 - r2).abs() < 1e-15, "conjugate pair: same real part");
        assert!(
            (i1 + i2).abs() < 1e-15,
            "conjugate pair: opposite imag part"
        );
    }

    // -- EigenTarget coverage --

    #[test]
    fn eigen_target_variants() {
        // Ensure all variants exist and are distinct
        let targets = [
            EigenTarget::LargestMagnitude,
            EigenTarget::SmallestMagnitude,
            EigenTarget::LargestAlgebraic,
            EigenTarget::SmallestAlgebraic,
        ];
        for i in 0..targets.len() {
            for j in (i + 1)..targets.len() {
                assert_ne!(targets[i], targets[j]);
            }
        }
    }

    // -- Dot product and norm reduction PTX tests --

    #[test]
    fn dot_product_reduce_ptx_f64_generates() {
        let ptx = emit_dot_product_reduce_f64(1000);
        assert!(ptx.is_ok(), "dot product PTX failed: {ptx:?}");
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry dot_product_reduce_f64"));
    }

    #[test]
    fn dot_product_reduce_ptx_f32_generates() {
        let ptx = emit_dot_product_reduce_f32(1000);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry dot_product_reduce_f32"));
    }

    #[test]
    fn norm_sq_reduce_ptx_generates() {
        let ptx_f64 = emit_norm_sq_reduce_f64(1000);
        assert!(ptx_f64.is_ok());
        let ptx_str = ptx_f64.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry norm_sq_reduce_f64"));

        let ptx_f32 = emit_norm_sq_reduce_f32(1000);
        assert!(ptx_f32.is_ok());
        let ptx_str_f32 = ptx_f32.expect("test: PTX gen should succeed");
        assert!(ptx_str_f32.contains(".entry norm_sq_reduce_f32"));
    }

    // -- Config accessor tests --

    #[test]
    fn plan_config_accessors() {
        let lanczos_config = LanczosConfig {
            max_iterations: 40,
            tolerance: 1e-8,
            num_eigenvalues: 10,
            which: EigenTarget::SmallestAlgebraic,
        };
        let plan = LanczosPlan::new(lanczos_config.clone(), 200).expect("test: valid config");
        assert_eq!(plan.config().max_iterations, 40);
        assert_eq!(plan.config().num_eigenvalues, 10);
        assert!((plan.config().tolerance - 1e-8).abs() < 1e-15);
        assert_eq!(plan.config().which, EigenTarget::SmallestAlgebraic);

        let arnoldi_config = ArnoldiConfig {
            max_iterations: 25,
            tolerance: 1e-12,
            num_eigenvalues: 6,
            which: EigenTarget::LargestAlgebraic,
        };
        let aplan = ArnoldiPlan::new(arnoldi_config, 300).expect("test: valid config");
        assert_eq!(aplan.config().max_iterations, 25);
        assert_eq!(aplan.config().num_eigenvalues, 6);
        assert_eq!(aplan.dimension(), 300);
    }
}
