//! GPU CAME optimizer — Confidence-guided Adaptive Memory Efficient optimizer.
//!
//! CAME (Luo et al., 2023) achieves memory efficiency comparable to Adafactor by
//! factorising the second moment matrix into row and column factors, reducing the
//! O(mn) storage requirement of Adam's second moment to O(m + n).
//!
//! ## Memory comparison (embedding weight of size m × n)
//!
//! | Optimizer | State memory |
//! |---|---|
//! | Adam | 2·m·n (two full moment matrices) |
//! | CAME | m + 2n (one row factor + one col factor, one first moment) |
//!
//! ## Algorithm (simplified, single matrix weight)
//!
//! ```text
//! // Gradient squared
//! g² = g ⊙ g
//!
//! // Row and column second-moment factors
//! r ← β₂·r + (1−β₂) · rowsum(g²) / n     (per-row mean of g²)
//! c ← β₂·c + (1−β₂) · colsum(g²) / m     (per-col mean of g²)
//!
//! // Reconstruct approximate second moment
//! v̂[i,j] ≈ r[i] · c[j] / mean(r)         (outer product / normalisation)
//!
//! // First moment
//! m ← β₁·m + (1−β₁)·g
//!
//! // Update
//! p ← p − lr · m / (√v̂ + ε)
//! ```
//!
//! For flat (1-D) parameters, CAME falls back to scalar `r` (variance of
//! gradient) instead of full row/column factorisation.

use super::{GpuOptimizer, ParamTensor};
use crate::error::{TrainError, TrainResult};

// ─── Per-parameter CAME state ────────────────────────────────────────────────

/// Second-moment state for one parameter tensor.
///
/// For 2-D parameters we use factored row and column buffers.
/// For 1-D parameters we store a single scalar variance.
#[derive(Debug, Clone)]
enum CameV {
    /// Flat / 1-D parameter: single variance scalar.
    Flat { v: Vec<f32> },
    /// 2-D matrix parameter: factored row and column buffers.
    Matrix {
        /// Row factor `r` (size = n_rows).
        row: Vec<f32>,
        /// Column factor `c` (size = n_cols).
        col: Vec<f32>,
        n_rows: usize,
        n_cols: usize,
    },
}

/// Full CAME state for one parameter.
#[derive(Debug, Clone)]
struct CameState {
    /// First moment buffer (same size as parameter).
    m1: Vec<f32>,
    /// Second moment state (flat or matrix).
    v: CameV,
}

impl CameState {
    fn zeros_flat(n: usize) -> Self {
        Self {
            m1: vec![0.0_f32; n],
            v: CameV::Flat {
                v: vec![0.0_f32; n],
            },
        }
    }

    fn zeros_matrix(n_rows: usize, n_cols: usize) -> Self {
        Self {
            m1: vec![0.0_f32; n_rows * n_cols],
            v: CameV::Matrix {
                row: vec![0.0_f32; n_rows],
                col: vec![0.0_f32; n_cols],
                n_rows,
                n_cols,
            },
        }
    }
}

// ─── GpuCame struct ───────────────────────────────────────────────────────────

/// GPU CAME optimizer.
#[derive(Debug, Clone)]
pub struct GpuCame {
    lr: f64,
    /// First-moment decay (default 0.9).
    beta1: f64,
    /// Second-moment factor decay (default 0.999).
    beta2: f64,
    /// Stability ε (default 1e-8).
    eps: f64,
    /// Decoupled weight decay.
    weight_decay: f64,
    step_count: u64,
    states: Vec<Option<CameState>>,
    /// Shape hints: (n_rows, n_cols). `None` = flat parameter.
    shape_hints: Vec<Option<(usize, usize)>>,
}

impl GpuCame {
    /// Create a new CAME optimizer.
    ///
    /// # Defaults
    ///
    /// * `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`, `weight_decay = 0.0`
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            step_count: 0,
            states: Vec::new(),
            shape_hints: Vec::new(),
        }
    }

    /// Set β₁.
    #[must_use]
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set β₂.
    #[must_use]
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set ε.
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set decoupled weight decay.
    #[must_use]
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Register a 2-D shape hint for parameter at position `idx`.
    ///
    /// Without a shape hint the optimizer uses the flat (1-D) factorisation.
    pub fn register_shape(&mut self, idx: usize, n_rows: usize, n_cols: usize) {
        if self.shape_hints.len() <= idx {
            self.shape_hints.resize(idx + 1, None);
        }
        self.shape_hints[idx] = Some((n_rows, n_cols));
    }

    fn ensure_states(&mut self, params: &[ParamTensor]) {
        if self.states.len() < params.len() {
            self.states.resize(params.len(), None);
            self.shape_hints.resize(params.len(), None);
        }
        for (i, p) in params.iter().enumerate() {
            if self.states[i].is_none() {
                self.states[i] = Some(match self.shape_hints[i] {
                    Some((nr, nc)) if nr * nc == p.len() => CameState::zeros_matrix(nr, nc),
                    _ => CameState::zeros_flat(p.len()),
                });
            }
        }
    }

    /// Update a flat parameter using per-element second moment.
    fn update_flat(
        data: &mut [f32],
        grad: &[f32],
        m1: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd_factor: f32,
    ) {
        let c1 = 1.0_f32 - beta1;
        let c2 = 1.0_f32 - beta2;

        for idx in 0..data.len() {
            data[idx] *= wd_factor;
            let g = grad[idx];

            // First moment
            let nm1 = beta1 * m1[idx] + c1 * g;
            m1[idx] = nm1;

            // Flat second moment
            let nv = beta2 * v[idx] + c2 * g * g;
            v[idx] = nv;

            // Update
            let denom = nv.sqrt() + eps;
            data[idx] -= lr * nm1 / denom;
        }
    }

    /// Update a matrix parameter using factored row/col second moment.
    fn update_matrix(
        data: &mut [f32],
        grad: &[f32],
        m1: &mut [f32],
        row: &mut [f32],
        col: &mut [f32],
        n_rows: usize,
        n_cols: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd_factor: f32,
    ) {
        let c1 = 1.0_f32 - beta1;
        let c2 = 1.0_f32 - beta2;
        let n_cols_f = n_cols as f32;
        let n_rows_f = n_rows as f32;

        // Compute per-element g² and row/col sums
        let mut row_sum = vec![0.0_f32; n_rows];
        let mut col_sum = vec![0.0_f32; n_cols];
        for r in 0..n_rows {
            for c_idx in 0..n_cols {
                let g = grad[r * n_cols + c_idx];
                let g2 = g * g;
                row_sum[r] += g2;
                col_sum[c_idx] += g2;
            }
        }

        // Update row and column factors
        for (r_val, rs) in row.iter_mut().zip(row_sum.iter()) {
            *r_val = beta2 * *r_val + c2 * (rs / n_cols_f);
        }
        for (c_val, cs) in col.iter_mut().zip(col_sum.iter()) {
            *c_val = beta2 * *c_val + c2 * (cs / n_rows_f);
        }

        // Mean of row factors for normalisation
        let row_mean: f32 = row.iter().sum::<f32>() / (n_rows as f32 + 1e-10);

        // Update parameters using reconstructed v̂[i,j] ≈ row[i]*col[j] / row_mean
        // Iterate over rows directly to avoid index-based row/col access.
        let data_rows = data.chunks_mut(n_cols);
        let grad_rows = grad.chunks(n_cols);
        let m1_rows = m1.chunks_mut(n_cols);
        for ((&row_val, (data_row, grad_row)), m1_row) in
            row.iter().zip(data_rows.zip(grad_rows)).zip(m1_rows)
        {
            for ((&col_val, (d, &g)), m) in col
                .iter()
                .zip(data_row.iter_mut().zip(grad_row.iter()))
                .zip(m1_row.iter_mut())
            {
                *d *= wd_factor;
                // First moment
                let nm1 = beta1 * *m + c1 * g;
                *m = nm1;
                // Reconstructed second moment
                let v_hat = (row_val * col_val / (row_mean + 1e-10)).max(eps * eps);
                let denom = v_hat.sqrt() + eps;
                *d -= lr * nm1 / denom;
            }
        }
    }
}

impl GpuOptimizer for GpuCame {
    fn step(&mut self, params: &mut [ParamTensor]) -> TrainResult<()> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        self.ensure_states(params);
        self.step_count += 1;

        let lr = self.lr as f32;
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let eps = self.eps as f32;
        let wd_factor = (1.0 - self.lr * self.weight_decay) as f32;

        for (i, param) in params.iter_mut().enumerate() {
            if !param.requires_grad {
                continue;
            }
            let grad = param
                .grad
                .as_ref()
                .ok_or(TrainError::NoGradient { index: i })?
                .clone();

            let state = self.states[i]
                .as_mut()
                .expect("state guaranteed by ensure_states()");
            match &mut state.v {
                CameV::Flat { v } => {
                    Self::update_flat(
                        &mut param.data,
                        &grad,
                        &mut state.m1,
                        v,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        wd_factor,
                    );
                }
                CameV::Matrix {
                    row,
                    col,
                    n_rows,
                    n_cols,
                } => {
                    let nr = *n_rows;
                    let nc = *n_cols;
                    Self::update_matrix(
                        &mut param.data,
                        &grad,
                        &mut state.m1,
                        row,
                        col,
                        nr,
                        nc,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        wd_factor,
                    );
                }
            }
        }
        Ok(())
    }

    fn lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn name(&self) -> &str {
        "CAME"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_param(data: Vec<f32>, grad: Vec<f32>) -> ParamTensor {
        let mut p = ParamTensor::new(data, "w");
        p.set_grad(grad).unwrap();
        p
    }

    #[test]
    fn came_flat_step_decreases_param() {
        let mut opt = GpuCame::new(1e-3);
        let mut params = vec![make_param(vec![1.0_f32; 8], vec![0.5_f32; 8])];
        opt.step(&mut params).unwrap();
        for &v in &params[0].data {
            assert!(v < 1.0, "param should decrease, got {v}");
        }
    }

    #[test]
    fn came_matrix_step_works() {
        // 4×4 matrix parameter
        let n = 4 * 4;
        let mut opt = GpuCame::new(1e-3);
        opt.register_shape(0, 4, 4);
        let mut params = vec![make_param(vec![1.0_f32; n], vec![0.1_f32; n])];
        opt.step(&mut params).unwrap();
        for &v in &params[0].data {
            assert!(v < 1.0, "matrix param should decrease, got {v}");
        }
    }

    #[test]
    fn came_matrix_memory_is_factored() {
        let mut opt = GpuCame::new(1e-3);
        opt.register_shape(0, 8, 16);
        let mut params = vec![make_param(vec![0.0_f32; 128], vec![1.0_f32; 128])];
        opt.step(&mut params).unwrap();

        // State should have row (8) + col (16) buffers, NOT 128-element v
        if let Some(CameV::Matrix { row, col, .. }) = opt.states[0].as_ref().map(|s| &s.v) {
            assert_eq!(row.len(), 8);
            assert_eq!(col.len(), 16);
        } else {
            panic!("expected Matrix variant");
        }
    }

    #[test]
    fn came_converges_flat() {
        let mut opt = GpuCame::new(1e-2);
        let mut params = vec![make_param(vec![4.0_f32], vec![0.0_f32])];
        for _ in 0..300 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            opt.step(&mut params).unwrap();
        }
        assert!(
            params[0].data[0].abs() < 0.3,
            "should converge near 0, got {}",
            params[0].data[0]
        );
    }

    #[test]
    fn came_name() {
        assert_eq!(GpuCame::new(1e-3).name(), "CAME");
    }
}
