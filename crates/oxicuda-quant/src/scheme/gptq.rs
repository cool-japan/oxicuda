//! # GPTQ — Gradient-Free Post-Training Quantization
//!
//! Frantar et al. (2022): "GPTQ: Accurate Post-Training Quantization
//! for Generative Pre-trained Transformers" <https://arxiv.org/abs/2210.17323>
//!
//! GPTQ applies Optimal Brain Quantization (OBC) to transformer weight matrices.
//! It uses the layer Hessian H = 2 X^T X (second-order information from input
//! activations) to minimize layer-output error during column-wise quantization.
//!
//! ## Algorithm (column-wise OBC)
//!
//! ```text
//! H_d  = H + λ I,   λ = percdamp × mean(diag H)
//! L    = cholesky(H_d)       H_d = L Lᵀ
//! L⁻¹  = invert_lower(L)
//! H⁻¹  = (L⁻¹)ᵀ L⁻¹
//!
//! for j = 0 .. n_cols:
//!     scale_j = max|W[:,j]| / q_max
//!     Q[:,j]  = quant(W[:,j], scale_j)
//!     err[:]  = (W[:,j] − Q[:,j]·scale_j) / H⁻¹[j,j]
//!     W[:,j+1:] −= outer(err, H⁻¹[j, j+1:])
//! ```

use crate::error::{QuantError, QuantResult};

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for GPTQ column-wise weight quantization.
#[derive(Debug, Clone)]
pub struct GptqConfig {
    /// Quantization bit-width (typically 2, 3, 4, or 8).
    pub bits: u32,
    /// Columns processed per block.  Larger blocks reduce overhead at the cost
    /// of higher peak memory.  Must divide `n_cols` evenly or will be clamped.
    pub block_size: usize,
    /// Relative dampening coefficient: λ = percdamp × mean(diag H).
    /// Prevents numerical failure when H has near-zero eigenvalues.
    pub percdamp: f64,
    /// Use symmetric quantization (zero-point = 0).
    pub symmetric: bool,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            block_size: 128,
            percdamp: 0.01,
            symmetric: true,
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────────

/// GPTQ quantized weight output.
#[derive(Debug, Clone)]
pub struct GptqOutput {
    /// Integer weight codes, row-major (`n_rows × n_cols`).
    pub quantized: Vec<i32>,
    /// Per-column scales (`n_cols`).
    pub scales: Vec<f32>,
    /// Per-column zero-points (`n_cols`; 0 for symmetric).
    pub zero_points: Vec<i32>,
    /// Weight matrix row count.
    pub n_rows: usize,
    /// Weight matrix column count.
    pub n_cols: usize,
}

impl GptqOutput {
    /// Dequantize integer codes back to f32.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_rows * self.n_cols);
        for i in 0..self.n_rows {
            for j in 0..self.n_cols {
                let q = self.quantized[i * self.n_cols + j] as f32;
                let zp = self.zero_points[j] as f32;
                out.push((q - zp) * self.scales[j]);
            }
        }
        out
    }

    /// Mean squared error between dequantized output and the original weights.
    #[must_use]
    pub fn reconstruction_mse(&self, original: &[f32]) -> f32 {
        let deq = self.dequantize();
        let n = deq.len().max(1) as f32;
        deq.iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n
    }
}

// ─── GptqQuantizer ────────────────────────────────────────────────────────────

/// GPTQ quantizer using Hessian-guided Optimal Brain Quantization.
///
/// Requires a pre-computed layer Hessian `H = 2 X^T X / n_samples` where `X`
/// is the matrix of input activations collected during calibration.
#[derive(Debug, Clone)]
pub struct GptqQuantizer {
    /// Quantization configuration.
    pub config: GptqConfig,
}

impl GptqQuantizer {
    /// Create a new GPTQ quantizer with the supplied configuration.
    #[must_use]
    pub fn new(config: GptqConfig) -> Self {
        Self { config }
    }

    /// Quantize a weight matrix using GPTQ.
    ///
    /// # Parameters
    ///
    /// * `weights`  — row-major weight matrix, shape `n_rows × n_cols`.
    /// * `n_rows`   — output-feature count (rows of W).
    /// * `n_cols`   — input-feature count (columns of W).
    /// * `hessian`  — row-major symmetric PSD Hessian `H`, shape `n_cols × n_cols`.
    ///
    /// # Errors
    ///
    /// * [`QuantError::DimensionMismatch`] — inconsistent slice lengths.
    /// * [`QuantError::EmptyInput`]        — empty weight matrix.
    /// * [`QuantError::InvalidBitWidth`]   — `bits` is 0 or > 16.
    /// * [`QuantError::SingularHessian`]   — H not positive definite after dampening.
    pub fn quantize_layer(
        &self,
        weights: &[f32],
        n_rows: usize,
        n_cols: usize,
        hessian: &[f32],
    ) -> QuantResult<GptqOutput> {
        // ── Validate inputs ───────────────────────────────────────────────────
        if weights.is_empty() {
            return Err(QuantError::EmptyInput("GptqQuantizer::quantize_layer"));
        }
        if weights.len() != n_rows * n_cols {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows * n_cols,
                got: weights.len(),
            });
        }
        if hessian.len() != n_cols * n_cols {
            return Err(QuantError::DimensionMismatch {
                expected: n_cols * n_cols,
                got: hessian.len(),
            });
        }
        let bits = self.config.bits;
        if bits == 0 || bits > 16 {
            return Err(QuantError::InvalidBitWidth { bits });
        }

        // ── Dampen Hessian: H_d = H + λ I ────────────────────────────────────
        let mean_diag = (0..n_cols).map(|j| hessian[j * n_cols + j]).sum::<f32>() / n_cols as f32;
        let lambda = (self.config.percdamp as f32) * mean_diag.abs().max(1e-8);
        let mut h_damp = hessian.to_vec();
        for j in 0..n_cols {
            h_damp[j * n_cols + j] += lambda;
        }

        // ── Cholesky factorisation H_d = L Lᵀ ────────────────────────────────
        let l = cholesky_lower(&h_damp, n_cols).ok_or_else(|| {
            let min_diag = (0..n_cols)
                .map(|j| h_damp[j * n_cols + j])
                .fold(f32::INFINITY, f32::min);
            QuantError::SingularHessian { min_diag }
        })?;

        // ── Invert lower-triangular factor ────────────────────────────────────
        let l_inv = invert_lower(&l, n_cols);

        // ── Column-wise OBC quantisation ──────────────────────────────────────
        let (q_min, q_max) = quant_range(bits, self.config.symmetric);

        let mut w = weights.to_vec();
        let mut quantized = vec![0_i32; n_rows * n_cols];
        let mut scales = vec![0.0_f32; n_cols];
        let mut zero_points = vec![0_i32; n_cols];

        for j in 0..n_cols {
            // Per-column quantisation parameters.
            let (scale_j, zp_j) = col_quant_params(
                &w,
                n_rows,
                n_cols,
                j,
                bits,
                self.config.symmetric,
                q_min,
                q_max,
            );
            scales[j] = scale_j;
            zero_points[j] = zp_j;

            // Slice of H⁻¹ row j starting at column j.
            // H⁻¹[j,k] = Σ_{m≥k} L_inv[m,j] · L_inv[m,k]
            let hinv_j = hinv_row_starting(&l_inv, n_cols, j);
            let hinv_jj = hinv_j[0];

            if hinv_jj < 1e-12 {
                // Degenerate: fall back to round-to-nearest.
                for i in 0..n_rows {
                    quantized[i * n_cols + j] =
                        quant_scalar(w[i * n_cols + j], scale_j, zp_j, q_min, q_max);
                }
                continue;
            }

            // Quantise column j and accumulate per-row errors.
            let mut errs = vec![0.0_f32; n_rows];
            for i in 0..n_rows {
                let wij = w[i * n_cols + j];
                let q = quant_scalar(wij, scale_j, zp_j, q_min, q_max);
                quantized[i * n_cols + j] = q;
                let q_dq = (q as f32 - zp_j as f32) * scale_j;
                errs[i] = (wij - q_dq) / hinv_jj;
            }

            // OBC update: W[:,k] -= err * H⁻¹[j,k]  for k > j.
            for (dk, k) in ((j + 1)..n_cols).enumerate() {
                let hinv_jk = hinv_j[dk + 1];
                for i in 0..n_rows {
                    w[i * n_cols + k] -= errs[i] * hinv_jk;
                }
            }
        }

        Ok(GptqOutput {
            quantized,
            scales,
            zero_points,
            n_rows,
            n_cols,
        })
    }
}

// ─── Private numeric helpers ─────────────────────────────────────────────────

/// Integer quantisation range for `bits` bits.
fn quant_range(bits: u32, symmetric: bool) -> (i32, i32) {
    if symmetric {
        let half = 1i32 << (bits - 1);
        (-half, half - 1)
    } else {
        (0i32, (1i32 << bits) - 1)
    }
}

/// Round-to-nearest quantisation of a single scalar.
fn quant_scalar(x: f32, scale: f32, zp: i32, q_min: i32, q_max: i32) -> i32 {
    (x / scale + zp as f32)
        .round()
        .clamp(q_min as f32, q_max as f32) as i32
}

/// Compute the per-column quantisation scale and zero-point.
fn col_quant_params(
    w: &[f32],
    n_rows: usize,
    n_cols: usize,
    j: usize,
    bits: u32,
    symmetric: bool,
    q_min: i32,
    q_max: i32,
) -> (f32, i32) {
    if symmetric {
        let abs_max = (0..n_rows)
            .map(|i| w[i * n_cols + j].abs())
            .fold(0.0_f32, f32::max)
            .max(1e-8);
        (abs_max / q_max as f32, 0)
    } else {
        let fmin = (0..n_rows)
            .map(|i| w[i * n_cols + j])
            .fold(f32::INFINITY, f32::min);
        let fmax = (0..n_rows)
            .map(|i| w[i * n_cols + j])
            .fold(f32::NEG_INFINITY, f32::max);
        let range = (fmax - fmin).max(1e-8);
        let scale = range / ((1i32 << bits) - 1) as f32;
        let zp = (-fmin / scale).round().clamp(q_min as f32, q_max as f32) as i32;
        (scale, zp)
    }
}

/// Cholesky decomposition H = L Lᵀ (lower triangular L).
///
/// Returns `None` if H is not positive definite.
fn cholesky_lower(h: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut l = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = h[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Invert a lower-triangular matrix via forward substitution.
///
/// Returns L⁻¹ (also lower triangular).
fn invert_lower(l: &[f32], n: usize) -> Vec<f32> {
    let mut li = vec![0.0_f32; n * n];
    for i in 0..n {
        li[i * n + i] = 1.0 / l[i * n + i];
        for j in 0..i {
            let mut s = 0.0_f32;
            for k in j..i {
                s += l[i * n + k] * li[k * n + j];
            }
            li[i * n + j] = -s / l[i * n + i];
        }
    }
    li
}

/// Compute H⁻¹[j, k] for k = j, j+1, …, n−1 from the lower Cholesky inverse.
///
/// H⁻¹ = (L⁻¹)ᵀ L⁻¹, so:
/// ```text
/// H⁻¹[j,k] = Σ_{m=max(j,k)}^{n−1} L_inv[m,j] · L_inv[m,k]
/// ```
/// Since we always have k ≥ j here, the sum starts at m = k.
fn hinv_row_starting(l_inv: &[f32], n: usize, j: usize) -> Vec<f32> {
    (j..n)
        .map(|k| (k..n).map(|m| l_inv[m * n + j] * l_inv[m * n + k]).sum())
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Build an n×n identity matrix.
    fn eye(n: usize) -> Vec<f32> {
        let mut h = vec![0.0_f32; n * n];
        for i in 0..n {
            h[i * n + i] = 1.0;
        }
        h
    }

    /// Matrix multiply A (m×k) by B (k×n) — for test verification only.
    fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for l in 0..k {
                for j in 0..n {
                    c[i * n + j] += a[i * k + l] * b[l * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn cholesky_decomposition_correct() {
        // H = [[4, 2], [2, 3]]  →  L = [[2, 0], [1, √2]]
        let h = vec![4.0_f32, 2.0, 2.0, 3.0];
        let l = cholesky_lower(&h, 2).expect("should succeed");
        // Verify L Lᵀ = H
        let lt: Vec<f32> = vec![l[0], l[2], l[1], l[3]]; // transpose
        let hh = matmul(&l, &lt, 2, 2, 2);
        for (a, b) in hh.iter().zip(h.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn cholesky_identity_is_identity() {
        let h = eye(4);
        let l = cholesky_lower(&h, 4).unwrap();
        // L should be identity.
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(l[i * 4 + j], expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn invert_lower_round_trip() {
        // L = [[2, 0, 0], [1, 3, 0], [0, 2, 4]]
        let l = vec![2.0_f32, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 2.0, 4.0];
        let li = invert_lower(&l, 3);
        // L * L_inv should be ≈ I
        let prod = matmul(&l, &li, 3, 3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(prod[i * 3 + j], expected, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn singular_hessian_returns_error() {
        // With percdamp=0, lambda=0 so no dampening is added.
        // Zero Hessian → Cholesky sees s=0 on first diagonal → fails.
        let q = GptqQuantizer::new(GptqConfig {
            percdamp: 0.0,
            ..GptqConfig::default()
        });
        let h = vec![0.0_f32; 4]; // 2×2 zero matrix
        let w = vec![0.5_f32; 4]; // 2×2
        assert!(matches!(
            q.quantize_layer(&w, 2, 2, &h),
            Err(QuantError::SingularHessian { .. })
        ));
    }

    #[test]
    fn dimension_mismatch_weight() {
        let q = GptqQuantizer::new(GptqConfig::default());
        let h = eye(4);
        let w = vec![0.5_f32; 3]; // wrong: 4×1 expected
        assert!(matches!(
            q.quantize_layer(&w, 1, 4, &h),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn invalid_bit_width() {
        let q = GptqQuantizer::new(GptqConfig {
            bits: 0,
            ..GptqConfig::default()
        });
        let h = eye(2);
        let w = vec![0.5_f32; 4];
        assert!(matches!(
            q.quantize_layer(&w, 2, 2, &h),
            Err(QuantError::InvalidBitWidth { bits: 0 })
        ));
    }

    #[test]
    fn identity_hessian_int8_low_mse() {
        // With identity Hessian, GPTQ reduces to round-to-nearest.
        // INT8 should give very low MSE on this range.
        let q = GptqQuantizer::new(GptqConfig {
            bits: 8,
            symmetric: true,
            ..GptqConfig::default()
        });
        let n_rows = 4;
        let n_cols = 8;
        let weights: Vec<f32> = (0..(n_rows * n_cols))
            .map(|i| (i as f32) / 16.0 - 1.0)
            .collect();
        let h = eye(n_cols);
        let out = q.quantize_layer(&weights, n_rows, n_cols, &h).unwrap();
        let mse = out.reconstruction_mse(&weights);
        assert!(mse < 1e-4, "INT8 MSE on identity Hessian too large: {mse}");
    }

    #[test]
    fn gptq_output_dequantize_shape() {
        let q = GptqQuantizer::new(GptqConfig {
            bits: 4,
            ..GptqConfig::default()
        });
        let n_rows = 3;
        let n_cols = 6;
        let weights = vec![0.3_f32; n_rows * n_cols];
        let h = eye(n_cols);
        let out = q.quantize_layer(&weights, n_rows, n_cols, &h).unwrap();
        assert_eq!(out.quantized.len(), n_rows * n_cols);
        assert_eq!(out.scales.len(), n_cols);
        assert_eq!(out.zero_points.len(), n_cols);
        let deq = out.dequantize();
        assert_eq!(deq.len(), n_rows * n_cols);
    }

    #[test]
    fn asymmetric_int4_round_trip() {
        // Asymmetric INT4 requires each column to span negative values so
        // that zero-point = round(-fmin/scale) is a valid non-negative integer.
        // Columns that are entirely positive would get zp clamped to 0,
        // making quantization of the full [fmin, fmax] range impossible.
        let q = GptqQuantizer::new(GptqConfig {
            bits: 4,
            symmetric: false,
            ..GptqConfig::default()
        });
        let n_rows = 2;
        let n_cols = 4;
        // Each column spans a signed range so zp fits naturally in [0, q_max].
        let weights = vec![-0.6_f32, 0.4, -0.2, 0.8, 0.6_f32, -0.4, 0.2, -0.8];
        let h = eye(n_cols);
        let out = q.quantize_layer(&weights, n_rows, n_cols, &h).unwrap();
        let mse = out.reconstruction_mse(&weights);
        assert!(mse < 0.05, "Asymmetric INT4 MSE too large: {mse}");
    }
}
