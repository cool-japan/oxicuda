//! GPU Adam optimizer with optional AMSGrad.
//!
//! Implements the Adam algorithm (Kingma & Ba, 2014) with:
//! * Bias-corrected first and second moment estimates
//! * Optional AMSGrad variant (Reddi et al., 2018) for improved convergence
//! * Fused PTX update kernel emulation (CPU-side reference matches PTX logic)
//!
//! ## Algorithm
//!
//! For each parameter `p` at step `t`:
//! ```text
//! m ← β₁·m + (1−β₁)·g
//! v ← β₂·v + (1−β₂)·g²
//!
//! m̂ = m / (1−β₁ᵗ)          (bias correction)
//! v̂ = v / (1−β₂ᵗ)
//!
//! if AMSGrad:  v̂_max ← max(v̂_max, v̂)
//!              p ← p − α · m̂ / (√v̂_max + ε)
//! else:        p ← p − α · m̂ / (√v̂ + ε)
//! ```

use super::{GpuOptimizer, ParamTensor, adam_bias_corrections};
use crate::error::{TrainError, TrainResult};

/// GPU Adam optimizer state.
///
/// Keeps first (`exp_avg`) and second (`exp_avg_sq`) moment buffers, one per
/// parameter element.
#[derive(Debug, Clone)]
pub struct GpuAdam {
    /// Learning rate.
    lr: f64,
    /// First-moment decay (default 0.9).
    beta1: f64,
    /// Second-moment decay (default 0.999).
    beta2: f64,
    /// Numerical stability term (default 1e-8).
    eps: f64,
    /// L2 weight decay applied to gradient (default 0 — use `GpuAdamW` for
    /// decoupled weight decay).
    weight_decay: f64,
    /// Enable AMSGrad variant.
    amsgrad: bool,
    /// Global step counter (incremented each `step` call).
    step_count: u64,
    /// Per-parameter first moment buffers.
    exp_avg: Vec<Vec<f32>>,
    /// Per-parameter second moment buffers.
    exp_avg_sq: Vec<Vec<f32>>,
    /// Per-parameter max second moment buffers (AMSGrad only).
    max_exp_avg_sq: Vec<Vec<f32>>,
}

impl GpuAdam {
    /// Create a new Adam optimizer.
    ///
    /// # Defaults
    ///
    /// * `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`
    /// * `weight_decay = 0`, `amsgrad = false`
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            step_count: 0,
            exp_avg: Vec::new(),
            exp_avg_sq: Vec::new(),
            max_exp_avg_sq: Vec::new(),
        }
    }

    /// Set `β₁`.
    #[must_use]
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set `β₂`.
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

    /// Set L2 weight decay (added to gradient before moment update).
    #[must_use]
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Enable AMSGrad variant.
    #[must_use]
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    /// Current global step.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    // ── Initialise state buffers for a set of parameters ─────────────────

    fn ensure_states(&mut self, params: &[ParamTensor]) {
        if self.exp_avg.len() < params.len() {
            self.exp_avg.resize_with(params.len(), Vec::new);
            self.exp_avg_sq.resize_with(params.len(), Vec::new);
            if self.amsgrad {
                self.max_exp_avg_sq.resize_with(params.len(), Vec::new);
            }
        }
        for (i, p) in params.iter().enumerate() {
            if self.exp_avg[i].len() != p.len() {
                self.exp_avg[i] = vec![0.0_f32; p.len()];
                self.exp_avg_sq[i] = vec![0.0_f32; p.len()];
                if self.amsgrad {
                    self.max_exp_avg_sq[i] = vec![0.0_f32; p.len()];
                }
            }
        }
    }

    // ── Fused element-wise Adam update (CPU reference matching PTX logic) ─

    fn apply_update(
        data: &mut [f32],
        grad: &[f32],
        m1: &mut [f32],
        m2: &mut [f32],
        mut max_m2: Option<&mut Vec<f32>>,
        step_size: f32,
        bc2_rsqrt: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
    ) {
        let c1 = 1.0_f32 - beta1;
        let c2 = 1.0_f32 - beta2;

        for idx in 0..data.len() {
            // L2 weight decay: add wd*p to gradient
            let g = grad[idx] + wd * data[idx];

            // Moment updates
            let nm1 = beta1 * m1[idx] + c1 * g;
            let nm2 = beta2 * m2[idx] + c2 * g * g;
            m1[idx] = nm1;
            m2[idx] = nm2;

            // Denominator: bias-corrected sqrt
            let v_hat_sq = if let Some(ref max_buf) = max_m2 {
                // AMSGrad: use running max
                let _ = max_buf; // will be accessed below
                nm2 * bc2_rsqrt * bc2_rsqrt
            } else {
                nm2 * bc2_rsqrt * bc2_rsqrt
            };

            let denom = if let Some(ref mut mx) = max_m2 {
                // mx[idx] = max(mx[idx], v_hat)
                let v_hat = v_hat_sq;
                if v_hat > mx[idx] {
                    mx[idx] = v_hat;
                }
                mx[idx].sqrt() + eps
            } else {
                v_hat_sq.sqrt() + eps
            };

            data[idx] -= step_size * nm1 / denom;
        }
    }
}

impl GpuOptimizer for GpuAdam {
    fn step(&mut self, params: &mut [ParamTensor]) -> TrainResult<()> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        self.ensure_states(params);
        self.step_count += 1;

        let (step_size, bc2_rsqrt) =
            adam_bias_corrections(self.lr, self.beta1, self.beta2, self.step_count);
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let eps = self.eps as f32;
        let wd = self.weight_decay as f32;
        let amsgrad = self.amsgrad;

        for (i, param) in params.iter_mut().enumerate() {
            if !param.requires_grad {
                continue;
            }
            let grad = param
                .grad
                .as_ref()
                .ok_or(TrainError::NoGradient { index: i })?;

            if amsgrad {
                let (m1, rest) = self.exp_avg.split_at_mut(i + 1);
                let _ = rest;
                let (m2, _) = self.exp_avg_sq.split_at_mut(i + 1);
                let (mx, _) = self.max_exp_avg_sq.split_at_mut(i + 1);
                Self::apply_update(
                    &mut param.data,
                    grad,
                    &mut m1[i],
                    &mut m2[i],
                    Some(&mut mx[i]),
                    step_size,
                    bc2_rsqrt,
                    beta1,
                    beta2,
                    eps,
                    wd,
                );
            } else {
                let m1 = &mut self.exp_avg[i];
                let m2 = &mut self.exp_avg_sq[i];
                Self::apply_update(
                    &mut param.data,
                    grad,
                    m1,
                    m2,
                    None,
                    step_size,
                    bc2_rsqrt,
                    beta1,
                    beta2,
                    eps,
                    wd,
                );
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
        if self.amsgrad { "AMSGrad" } else { "Adam" }
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
    fn adam_single_step_decreases_param() {
        // With a positive gradient, parameter should decrease.
        let mut opt = GpuAdam::new(1e-3);
        let mut params = vec![make_param(vec![1.0_f32; 4], vec![0.5_f32; 4])];
        opt.step(&mut params).unwrap();
        for &v in &params[0].data {
            assert!(v < 1.0_f32, "param should decrease, got {v}");
        }
    }

    #[test]
    fn adam_negative_gradient_increases_param() {
        let mut opt = GpuAdam::new(1e-3);
        let mut params = vec![make_param(vec![0.5_f32; 4], vec![-0.5_f32; 4])];
        opt.step(&mut params).unwrap();
        for &v in &params[0].data {
            assert!(v > 0.5_f32, "param should increase, got {v}");
        }
    }

    #[test]
    fn adam_multiple_steps_converge_to_zero_gradient() {
        // Minimise f(x) = x^2 via gradient 2x (gradient descent).
        // Using beta2=0.9 (same time-constant as beta1) so the second-moment
        // bias correction converges within ~20 steps; with the default
        // beta2=0.999 the correction takes ~1000 steps to stabilise.
        let mut opt = GpuAdam::new(1e-2).with_beta2(0.9);
        let mut params = vec![make_param(vec![2.0_f32], vec![0.0_f32])];

        for _ in 0..200 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            opt.step(&mut params).unwrap();
        }
        let x_final = params[0].data[0].abs();
        assert!(
            x_final < 0.1,
            "expected convergence near 0, got |x|={x_final}"
        );
    }

    #[test]
    fn adam_amsgrad_smoke() {
        let mut opt = GpuAdam::new(1e-3).with_amsgrad(true);
        let mut params = vec![make_param(vec![1.0_f32; 4], vec![0.1_f32; 4])];
        opt.step(&mut params).unwrap();
        opt.step(&mut params).unwrap();
        // Should not panic; max moment should grow
        assert!(opt.max_exp_avg_sq[0][0] > 0.0);
    }

    #[test]
    fn adam_empty_params_error() {
        let mut opt = GpuAdam::new(1e-3);
        let result = opt.step(&mut []);
        assert!(matches!(result, Err(TrainError::EmptyParams)));
    }

    #[test]
    fn adam_no_gradient_error() {
        let mut opt = GpuAdam::new(1e-3);
        let mut params = vec![ParamTensor::zeros(4, "w")]; // no grad set
        let result = opt.step(&mut params);
        assert!(matches!(result, Err(TrainError::NoGradient { index: 0 })));
    }

    #[test]
    fn adam_skips_no_grad_params() {
        let mut opt = GpuAdam::new(1e-3);
        let mut p = ParamTensor::new(vec![1.0_f32; 4], "frozen");
        p.requires_grad = false;
        let data_before = p.data.clone();
        let mut params = vec![p];
        opt.step(&mut params).unwrap();
        assert_eq!(
            params[0].data, data_before,
            "frozen param should not change"
        );
    }

    #[test]
    fn adam_weight_decay_effect() {
        // Weight decay increases the effective gradient, so convergence starts
        // slightly different but the optimiser should still work.
        let mut opt = GpuAdam::new(1e-3).with_weight_decay(1e-4);
        let mut params = vec![make_param(vec![1.0_f32; 4], vec![0.5_f32; 4])];
        opt.step(&mut params).unwrap();
        for &v in &params[0].data {
            assert!(v < 1.0, "param should decrease with weight decay");
        }
    }

    #[test]
    fn adam_name() {
        assert_eq!(GpuAdam::new(1e-3).name(), "Adam");
        assert_eq!(GpuAdam::new(1e-3).with_amsgrad(true).name(), "AMSGrad");
    }

    #[test]
    fn adam_set_lr() {
        let mut opt = GpuAdam::new(1e-3);
        opt.set_lr(1e-4);
        assert!((opt.lr() - 1e-4).abs() < 1e-12);
    }
}
