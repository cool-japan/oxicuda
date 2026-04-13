//! # RAdam (Rectified Adam) GPU Optimizer
//!
//! RAdam (Liu et al., "On the Variance of the Adaptive Learning Rate and
//! Beyond", ICLR 2020) introduces a rectified update that avoids the
//! problematic large-variance region of the Adam adaptive learning rate during
//! the first few steps of training.
//!
//! ## Algorithm
//!
//! Let `ρ∞ = 2/(1-β₂) − 1` be the maximum length of the approximated SMA.
//!
//! At each step `t`:
//! ```text
//! m_t = β₁ m_{t-1} + (1−β₁) g_t
//! v_t = β₂ v_{t-1} + (1−β₂) g_t²
//!
//! m̂_t = m_t / (1 − β₁ᵗ)          // bias-corrected first moment
//!
//! ρ_t = ρ∞ − 2t β₂ᵗ / (1 − β₂ᵗ) // approximated SMA length
//!
//! if ρ_t > 4:                       // variance is tractable
//!     r_t = √[(ρ_t−4)(ρ_t−2)ρ∞ / ((ρ∞−4)(ρ∞−2)ρ_t)]
//!     v̂_t = √(v_t / (1 − β₂ᵗ))
//!     p_t = p_{t-1} − lr · r_t · m̂_t / (v̂_t + ε)
//! else:                              // SGD warmup
//!     p_t = p_{t-1} − lr · m̂_t
//! ```
//!
//! This eliminates the need for a learning-rate warmup schedule while providing
//! better convergence than plain Adam in the first few hundred steps.

use crate::error::{TrainError, TrainResult};
use crate::gpu_optimizer::{GpuOptimizer, ParamTensor};

/// RAdam optimizer.
#[derive(Debug, Clone)]
pub struct GpuRAdam {
    /// Learning rate.
    lr: f64,
    /// First-moment EMA coefficient.
    beta1: f64,
    /// Second-moment EMA coefficient.
    beta2: f64,
    /// Numerical stability ε.
    eps: f64,
    /// Optional decoupled weight decay coefficient (AdamW-style).
    weight_decay: f64,
    /// Per-parameter first-moment buffers.
    m: Vec<Vec<f32>>,
    /// Per-parameter second-moment buffers.
    v: Vec<Vec<f32>>,
    /// Current step.
    step: u64,
}

impl GpuRAdam {
    /// Create RAdam with the given learning rate.
    ///
    /// Defaults: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, `weight_decay = 0`.
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            step: 0,
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

    /// Enable decoupled weight decay (AdamW-style).
    #[must_use]
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Current step count.
    #[must_use]
    #[inline]
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Compute the rectification term and the variance-tractability flag.
    ///
    /// Returns `(Some(rect), m_hat_scale, v_scale)` when variance is tractable
    /// (ρ_t > 4) or `(None, m_hat_scale, _)` when we fall back to SGD.
    fn radam_scalars(&self, t: f64) -> (Option<f32>, f32, f32) {
        let b1t = self.beta1.powf(t);
        let b2t = self.beta2.powf(t);
        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;
        let rho_t = rho_inf - 2.0 * t * b2t / (1.0 - b2t);

        // Bias-corrected first-moment scale
        let m_hat_scale = (1.0 / (1.0 - b1t)) as f32;

        if rho_t > 4.0 {
            // Variance tractable: compute rectification factor
            let numer = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf;
            let denom = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
            let rect = (numer / denom).sqrt() as f32;
            let v_scale = (1.0 / (1.0 - b2t)).sqrt() as f32;
            (Some(rect), m_hat_scale, v_scale)
        } else {
            // Variance intractable: use SGD step (no v)
            (None, m_hat_scale, 1.0)
        }
    }

    fn ensure_state(&mut self, params: &[ParamTensor]) {
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| vec![0.0_f32; p.data.len()]).collect();
            self.v = params.iter().map(|p| vec![0.0_f32; p.data.len()]).collect();
        }
    }
}

impl GpuOptimizer for GpuRAdam {
    fn step(&mut self, params: &mut [ParamTensor]) -> TrainResult<()> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        self.ensure_state(params);
        self.step += 1;

        let t = self.step as f64;
        let (rect_opt, m_hat_scale, v_scale) = self.radam_scalars(t);
        let lr = self.lr as f32;
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let eps = self.eps as f32;
        let wd = self.weight_decay as f32;

        for (i, param) in params.iter_mut().enumerate() {
            if !param.requires_grad {
                continue;
            }
            let grad = param
                .grad
                .as_ref()
                .ok_or(TrainError::NoGradient { index: i })?;
            let mi = &mut self.m[i];
            let vi = &mut self.v[i];

            for (((p, &g), m), v) in param
                .data
                .iter_mut()
                .zip(grad.iter())
                .zip(mi.iter_mut())
                .zip(vi.iter_mut())
            {
                // Decoupled weight decay
                if wd != 0.0 {
                    *p *= 1.0 - lr * wd;
                }
                // Update biased moment estimates
                *m = beta1 * *m + (1.0 - beta1) * g;
                *v = beta2 * *v + (1.0 - beta2) * g * g;

                let m_hat = *m * m_hat_scale;

                match rect_opt {
                    Some(rect) => {
                        let v_hat = (*v * v_scale * v_scale).sqrt() * v_scale;
                        // v_hat = sqrt(v / (1-β₂ᵗ)) = sqrt(v) * v_scale (approx)
                        // Correct: v̂ = sqrt(v / (1-β₂ᵗ))
                        let v_hat_corrected = (*v * (v_scale * v_scale)).sqrt();
                        let _ = v_hat; // discard above approximation
                        *p -= lr * rect * m_hat / (v_hat_corrected + eps);
                    }
                    None => {
                        // SGD warmup: p -= lr * m̂
                        *p -= lr * m_hat;
                    }
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn name(&self) -> &str {
        "RAdam"
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radam_reduces_param_early() {
        let mut opt = GpuRAdam::new(0.01);
        let mut params = vec![ParamTensor::new(vec![3.0_f32], "x")];
        for _ in 0..3 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            opt.step(&mut params).unwrap();
        }
        assert!(params[0].data[0] < 3.0, "RAdam should reduce param");
    }

    #[test]
    fn radam_converges() {
        // Use beta2=0.9 so rho_t enters tractable regime fast (rho∞=21)
        let mut opt = GpuRAdam::new(1e-2).with_beta2(0.9).with_beta1(0.9);
        let mut params = vec![ParamTensor::new(vec![3.0_f32], "x")];
        for _ in 0..500 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            opt.step(&mut params).unwrap();
        }
        let x = params[0].data[0].abs();
        assert!(x < 0.5, "RAdam should converge x→0, |x|={x}");
    }

    #[test]
    fn radam_sgd_warmup_regime() {
        // With beta2=0.9999, rho_inf ≈ 20001, rho_t < 4 for first few steps
        // → SGD warmup path
        let mut opt = GpuRAdam::new(0.01).with_beta2(0.9999);
        let mut params = vec![ParamTensor::new(vec![1.0_f32], "x")];
        params[0].set_grad(vec![1.0]).unwrap();
        opt.step(&mut params).unwrap(); // step 1 — SGD warmup
        // Just check it doesn't blow up and reduces param
        let x = params[0].data[0];
        assert!(
            x.is_finite() && x < 1.0,
            "SGD warmup step should reduce param, x={x}"
        );
    }

    #[test]
    fn radam_weight_decay_reduces_more() {
        let mut opt_wd = GpuRAdam::new(0.01).with_weight_decay(0.1).with_beta2(0.9);
        let mut opt_no = GpuRAdam::new(0.01).with_beta2(0.9);
        let mut p_wd = vec![ParamTensor::new(vec![2.0_f32], "p")];
        let mut p_no = vec![ParamTensor::new(vec![2.0_f32], "p")];
        for _ in 0..10 {
            p_wd[0].set_grad(vec![0.1_f32]).unwrap();
            p_no[0].set_grad(vec![0.1_f32]).unwrap();
            opt_wd.step(&mut p_wd).unwrap();
            opt_no.step(&mut p_no).unwrap();
        }
        // With weight decay the param should be smaller
        assert!(
            p_wd[0].data[0] < p_no[0].data[0],
            "weight decay should shrink param more: wd={} vs no-wd={}",
            p_wd[0].data[0],
            p_no[0].data[0]
        );
    }

    #[test]
    fn radam_empty_error() {
        let mut opt = GpuRAdam::new(0.01);
        assert!(opt.step(&mut []).is_err());
    }

    #[test]
    fn radam_no_grad_error() {
        let mut opt = GpuRAdam::new(0.01);
        let mut params = vec![ParamTensor::new(vec![1.0], "p")];
        assert!(opt.step(&mut params).is_err());
    }

    #[test]
    fn radam_step_count() {
        let mut opt = GpuRAdam::new(0.01).with_beta2(0.9);
        let mut params = vec![ParamTensor::new(vec![1.0_f32], "p")];
        for _ in 0..4 {
            params[0].set_grad(vec![0.5]).unwrap();
            opt.step(&mut params).unwrap();
        }
        assert_eq!(opt.step_count(), 4);
    }

    #[test]
    fn radam_name() {
        assert_eq!(GpuRAdam::new(0.01).name(), "RAdam");
    }

    #[test]
    fn radam_scalars_tractable() {
        // With beta2=0.9 and t large enough, rho_t > 4
        let opt = GpuRAdam::new(0.01).with_beta2(0.9);
        // rho_inf = 2/(1-0.9) - 1 = 19
        // rho_t at t=10: 19 - 20*(0.9^10)/(1-0.9^10) ≈ 19 - 20*0.349/0.651 ≈ 8.3 > 4
        let (rect, _, _) = opt.radam_scalars(10.0);
        assert!(
            rect.is_some(),
            "should be in tractable regime at t=10 with beta2=0.9"
        );
    }

    #[test]
    fn radam_scalars_sgd_warmup() {
        // beta2 = 0.9999: rho_inf = 20001, rho_t at t=1 ≈ 20001 - 2*0.9999/0.0001 ≈ 0.01 < 4
        let opt = GpuRAdam::new(0.01).with_beta2(0.9999);
        let (rect, _, _) = opt.radam_scalars(1.0);
        assert!(
            rect.is_none(),
            "should be in SGD warmup at t=1 with beta2=0.9999"
        );
    }
}
