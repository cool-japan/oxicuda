//! # RMSProp GPU Optimizer
//!
//! RMSProp (Hinton 2012) maintains a per-parameter exponential moving average
//! of squared gradients `v` and divides the update by `√(v + ε)`:
//!
//! ```text
//! v[i]  =  α * v[i]  +  (1 − α) * g[i]²
//! p[i] -=  lr / √(v[i] + ε)  * g[i]
//! ```
//!
//! **Centred RMSProp** additionally tracks the first-moment `μ` and uses the
//! variance `v − μ²` in the denominator, which can improve stability:
//!
//! ```text
//! μ[i]  =  α * μ[i]  +  (1 − α) * g[i]
//! v[i]  =  α * v[i]  +  (1 − α) * g[i]²
//! p[i] -=  lr / √(v[i] − μ[i]² + ε)  * g[i]
//! ```
//!
//! **Momentum** variant adds a velocity buffer to smooth the parameter update
//! trajectory:
//!
//! ```text
//! buf[i]  =  mom * buf[i]  +  g[i] / √(v[i] + ε)
//! p[i]   -=  lr * buf[i]
//! ```

use crate::error::{TrainError, TrainResult};
use crate::gpu_optimizer::{GpuOptimizer, ParamTensor};

/// RMSProp optimizer with optional centering and momentum.
#[derive(Debug, Clone)]
pub struct GpuRMSProp {
    /// Learning rate.
    lr: f64,
    /// Squared-gradient EMA smoothing coefficient α.
    alpha: f64,
    /// Numerical stability ε.
    eps: f64,
    /// Momentum coefficient (0 = disabled).
    momentum: f64,
    /// Whether to use centred RMSProp.
    centred: bool,
    /// Per-parameter second-moment buffers.
    v: Vec<Vec<f32>>,
    /// Per-parameter first-moment buffers (only used when `centred = true`).
    mu: Vec<Vec<f32>>,
    /// Per-parameter momentum velocity buffers.
    buf: Vec<Vec<f32>>,
    /// Number of optimizer steps taken.
    step: u64,
}

impl GpuRMSProp {
    /// Create RMSProp with the given learning rate.
    ///
    /// Defaults: `alpha = 0.99`, `eps = 1e-8`, `momentum = 0.0`,
    /// `centred = false`.
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            centred: false,
            v: Vec::new(),
            mu: Vec::new(),
            buf: Vec::new(),
            step: 0,
        }
    }

    /// Set the squared-gradient EMA coefficient.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the numerical stability ε.
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Enable momentum with coefficient `mom`.
    #[must_use]
    pub fn with_momentum(mut self, mom: f64) -> Self {
        self.momentum = mom;
        self
    }

    /// Enable centred RMSProp.
    #[must_use]
    pub fn centred(mut self) -> Self {
        self.centred = true;
        self
    }

    /// Current step count.
    #[must_use]
    #[inline]
    pub fn step_count(&self) -> u64 {
        self.step
    }

    fn ensure_state(&mut self, params: &[ParamTensor]) {
        if self.v.len() != params.len() {
            self.v = params.iter().map(|p| vec![0.0_f32; p.data.len()]).collect();
            self.mu = params.iter().map(|p| vec![0.0_f32; p.data.len()]).collect();
            self.buf = params.iter().map(|p| vec![0.0_f32; p.data.len()]).collect();
        }
    }
}

impl GpuOptimizer for GpuRMSProp {
    fn step(&mut self, params: &mut [ParamTensor]) -> TrainResult<()> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        self.ensure_state(params);
        self.step += 1;
        let lr = self.lr as f32;
        let alpha = self.alpha as f32;
        let one_minus_alpha = 1.0_f32 - alpha;
        let eps = self.eps as f32;
        let mom = self.momentum as f32;

        for (i, param) in params.iter_mut().enumerate() {
            if !param.requires_grad {
                continue;
            }
            let grad = param
                .grad
                .as_ref()
                .ok_or(TrainError::NoGradient { index: i })?;
            let vi = &mut self.v[i];
            let mui = &mut self.mu[i];
            let bufi = &mut self.buf[i];

            for (((p, &g), v), (mu, b)) in param
                .data
                .iter_mut()
                .zip(grad.iter())
                .zip(vi.iter_mut())
                .zip(mui.iter_mut().zip(bufi.iter_mut()))
            {
                // Update squared-gradient EMA
                *v = alpha * *v + one_minus_alpha * g * g;

                let denom = if self.centred {
                    // Update first-moment EMA and use variance
                    *mu = alpha * *mu + one_minus_alpha * g;
                    let variance = (*v - *mu * *mu).max(0.0_f32);
                    (variance + eps).sqrt()
                } else {
                    (*v + eps).sqrt()
                };

                if mom > 0.0 {
                    *b = mom * *b + g / denom;
                    *p -= lr * *b;
                } else {
                    *p -= lr * g / denom;
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
        "RMSProp"
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_param(val: f32) -> Vec<ParamTensor> {
        let mut p = ParamTensor::new(vec![val], "p");
        p.set_grad(vec![2.0 * val]).unwrap();
        vec![p]
    }

    #[test]
    fn rmsprop_reduces_param() {
        let mut opt = GpuRMSProp::new(0.01);
        let mut params = simple_param(3.0);
        opt.step(&mut params).unwrap();
        assert!(params[0].data[0] < 3.0);
    }

    #[test]
    fn rmsprop_converges() {
        let mut opt = GpuRMSProp::new(0.05).with_alpha(0.9);
        let mut params = vec![ParamTensor::new(vec![4.0_f32], "x")];
        for _ in 0..300 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            opt.step(&mut params).unwrap();
        }
        let x = params[0].data[0].abs();
        assert!(x < 0.5, "RMSProp should converge, |x|={x}");
    }

    #[test]
    fn rmsprop_centred_converges() {
        let mut opt = GpuRMSProp::new(0.05).with_alpha(0.9).centred();
        let mut params = vec![ParamTensor::new(vec![4.0_f32], "x")];
        for _ in 0..300 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            opt.step(&mut params).unwrap();
        }
        let x = params[0].data[0].abs();
        assert!(x < 0.5, "Centred RMSProp should converge, |x|={x}");
    }

    #[test]
    fn rmsprop_with_momentum_converges() {
        let mut opt = GpuRMSProp::new(0.02).with_alpha(0.9).with_momentum(0.9);
        let mut params = vec![ParamTensor::new(vec![4.0_f32], "x")];
        for _ in 0..300 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            opt.step(&mut params).unwrap();
        }
        let x = params[0].data[0].abs();
        assert!(x < 0.5, "RMSProp+momentum should converge, |x|={x}");
    }

    #[test]
    fn rmsprop_empty_error() {
        let mut opt = GpuRMSProp::new(0.01);
        assert!(opt.step(&mut []).is_err());
    }

    #[test]
    fn rmsprop_no_grad_error() {
        let mut opt = GpuRMSProp::new(0.01);
        let mut params = vec![ParamTensor::new(vec![1.0], "p")];
        assert!(opt.step(&mut params).is_err());
    }

    #[test]
    fn rmsprop_step_count() {
        let mut opt = GpuRMSProp::new(0.01);
        let mut params = simple_param(1.0);
        for _ in 0..7 {
            params[0].set_grad(vec![0.1]).unwrap();
            opt.step(&mut params).unwrap();
        }
        assert_eq!(opt.step_count(), 7);
    }

    #[test]
    fn rmsprop_name() {
        assert_eq!(GpuRMSProp::new(0.01).name(), "RMSProp");
    }
}
