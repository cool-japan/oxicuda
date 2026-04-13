//! GPU-accelerated optimizers (SGD, Adam, AdaGrad, RMSProp, LAMB).
//!
//! Each optimizer implements the [`Optimizer`] trait and operates directly
//! on GPU tensors, reading from `.grad` and updating the parameter data.

use super::error::TensorError;
use super::tensor::GpuTensor;

// ─── Optimizer trait ────────────────────────────────────────

/// Trait for parameter optimizers.
pub trait Optimizer {
    /// Perform one optimization step (update parameters from their gradients).
    fn step(&mut self, params: &mut [GpuTensor]) -> Result<(), TensorError>;

    /// Zero out all gradients on the given parameters.
    fn zero_grad(&mut self, params: &mut [GpuTensor]) {
        for p in params.iter_mut() {
            p.zero_grad();
        }
    }
}

// ─── SGD ────────────────────────────────────────────────────

/// Stochastic Gradient Descent with momentum, weight decay, dampening, and Nesterov.
#[derive(Debug, Clone)]
pub struct Sgd {
    /// Learning rate.
    pub lr: f64,
    /// Momentum factor.
    pub momentum: f64,
    /// Dampening for momentum.
    pub dampening: f64,
    /// L2 weight decay (regularization).
    pub weight_decay: f64,
    /// Whether to use Nesterov momentum.
    pub nesterov: bool,
    /// Velocity buffers (one per parameter tensor, flat).
    velocities: Vec<Vec<f64>>,
    /// Step counter.
    step_count: u64,
}

impl Sgd {
    /// Create a new SGD optimizer.
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocities: Vec::new(),
            step_count: 0,
        }
    }

    /// Set momentum.
    #[must_use]
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set dampening.
    #[must_use]
    pub fn with_dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    /// Set weight decay.
    #[must_use]
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Enable Nesterov momentum.
    #[must_use]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &mut [GpuTensor]) -> Result<(), TensorError> {
        // Initialize velocity buffers if needed
        if self.velocities.len() < params.len() {
            self.velocities.resize(params.len(), Vec::new());
        }

        for (i, param) in params.iter_mut().enumerate() {
            let grad = match param.grad() {
                Some(g) => g.host_data().to_vec(),
                None => continue,
            };

            // Ensure velocity buffer exists
            if self.velocities[i].len() != grad.len() {
                self.velocities[i] = vec![0.0; grad.len()];
            }

            let mut d_p = grad;

            // Weight decay: d_p += weight_decay * param
            if self.weight_decay != 0.0 {
                for (dp, &p) in d_p.iter_mut().zip(param.host_data.iter()) {
                    *dp += self.weight_decay * p;
                }
            }

            // Momentum
            if self.momentum != 0.0 {
                if self.step_count > 0 {
                    for (v, &dp) in self.velocities[i].iter_mut().zip(d_p.iter()) {
                        *v = self.momentum * *v + (1.0 - self.dampening) * dp;
                    }
                } else {
                    self.velocities[i].copy_from_slice(&d_p);
                }

                if self.nesterov {
                    for (dp, v) in d_p.iter_mut().zip(self.velocities[i].iter()) {
                        *dp += self.momentum * v;
                    }
                } else {
                    d_p.clone_from(&self.velocities[i]);
                }
            }

            // Update: param -= lr * d_p
            for (p, &dp) in param.host_data.iter_mut().zip(d_p.iter()) {
                *p -= self.lr * dp;
            }
        }

        self.step_count += 1;
        Ok(())
    }
}

// ─── Adam ───────────────────────────────────────────────────

/// Adam optimizer with optional AMSGrad and decoupled weight decay (AdamW).
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate.
    pub lr: f64,
    /// Exponential decay rate for first moment.
    pub beta1: f64,
    /// Exponential decay rate for second moment.
    pub beta2: f64,
    /// Numerical stability epsilon.
    pub eps: f64,
    /// Decoupled weight decay.
    pub weight_decay: f64,
    /// Whether to use AMSGrad variant.
    pub amsgrad: bool,
    /// First moment estimates.
    m: Vec<Vec<f64>>,
    /// Second moment estimates.
    v: Vec<Vec<f64>>,
    /// Max of second moment (AMSGrad).
    v_hat_max: Vec<Vec<f64>>,
    /// Step counter (global).
    t: u64,
}

impl Adam {
    /// Create a new Adam optimizer with defaults (lr=0.001, betas=(0.9,0.999), eps=1e-8).
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            m: Vec::new(),
            v: Vec::new(),
            v_hat_max: Vec::new(),
            t: 0,
        }
    }

    /// Set betas.
    #[must_use]
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set epsilon.
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

    /// Enable AMSGrad.
    #[must_use]
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

impl Optimizer for Adam {
    #[allow(clippy::needless_range_loop)]
    fn step(&mut self, params: &mut [GpuTensor]) -> Result<(), TensorError> {
        self.t += 1;

        if self.m.len() < params.len() {
            self.m.resize(params.len(), Vec::new());
            self.v.resize(params.len(), Vec::new());
            self.v_hat_max.resize(params.len(), Vec::new());
        }

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, param) in params.iter_mut().enumerate() {
            let grad = match param.grad() {
                Some(g) => g.host_data().to_vec(),
                None => continue,
            };

            let n = grad.len();
            if self.m[i].len() != n {
                self.m[i] = vec![0.0; n];
                self.v[i] = vec![0.0; n];
                if self.amsgrad {
                    self.v_hat_max[i] = vec![0.0; n];
                }
            }

            // Decoupled weight decay
            if self.weight_decay != 0.0 {
                for p in param.host_data.iter_mut() {
                    *p *= 1.0 - self.lr * self.weight_decay;
                }
            }

            // Update biased first and second moment
            for j in 0..n {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * grad[j];
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * grad[j] * grad[j];
            }

            // Compute step
            for j in 0..n {
                let m_hat = self.m[i][j] / bias_correction1;
                let v_hat = self.v[i][j] / bias_correction2;

                let denom = if self.amsgrad {
                    if self.v_hat_max[i].len() != n {
                        self.v_hat_max[i] = vec![0.0; n];
                    }
                    self.v_hat_max[i][j] = self.v_hat_max[i][j].max(v_hat);
                    self.v_hat_max[i][j].sqrt() + self.eps
                } else {
                    v_hat.sqrt() + self.eps
                };

                param.host_data[j] -= self.lr * m_hat / denom;
            }
        }

        Ok(())
    }
}

// ─── AdaGrad ────────────────────────────────────────────────

/// AdaGrad optimizer with per-parameter learning rate decay.
#[derive(Debug, Clone)]
pub struct AdaGrad {
    /// Initial learning rate.
    pub lr: f64,
    /// Learning rate decay.
    pub lr_decay: f64,
    /// Numerical stability epsilon.
    pub eps: f64,
    /// Weight decay.
    pub weight_decay: f64,
    /// Accumulated squared gradients.
    sum_sq: Vec<Vec<f64>>,
    /// Step counter.
    t: u64,
}

impl AdaGrad {
    /// Create a new AdaGrad optimizer.
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            lr_decay: 0.0,
            eps: 1e-10,
            weight_decay: 0.0,
            sum_sq: Vec::new(),
            t: 0,
        }
    }

    /// Set learning rate decay.
    #[must_use]
    pub fn with_lr_decay(mut self, decay: f64) -> Self {
        self.lr_decay = decay;
        self
    }

    /// Set epsilon.
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay.
    #[must_use]
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for AdaGrad {
    #[allow(clippy::needless_range_loop)]
    fn step(&mut self, params: &mut [GpuTensor]) -> Result<(), TensorError> {
        self.t += 1;
        let clr = self.lr / (1.0 + (self.t - 1) as f64 * self.lr_decay);

        if self.sum_sq.len() < params.len() {
            self.sum_sq.resize(params.len(), Vec::new());
        }

        for (i, param) in params.iter_mut().enumerate() {
            let mut grad = match param.grad() {
                Some(g) => g.host_data().to_vec(),
                None => continue,
            };

            let n = grad.len();
            if self.sum_sq[i].len() != n {
                self.sum_sq[i] = vec![0.0; n];
            }

            if self.weight_decay != 0.0 {
                for (g, &p) in grad.iter_mut().zip(param.host_data.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            for j in 0..n {
                self.sum_sq[i][j] += grad[j] * grad[j];
                param.host_data[j] -= clr * grad[j] / (self.sum_sq[i][j].sqrt() + self.eps);
            }
        }

        Ok(())
    }
}

// ─── RMSProp ────────────────────────────────────────────────

/// RMSProp optimizer.
#[derive(Debug, Clone)]
pub struct RmsProp {
    /// Learning rate.
    pub lr: f64,
    /// Smoothing constant (decay).
    pub alpha: f64,
    /// Numerical stability epsilon.
    pub eps: f64,
    /// Momentum factor.
    pub momentum: f64,
    /// Weight decay.
    pub weight_decay: f64,
    /// Whether to compute centered RMSProp.
    pub centered: bool,
    /// Running average of squared gradients.
    v: Vec<Vec<f64>>,
    /// Running average of gradients (centered mode).
    g_avg: Vec<Vec<f64>>,
    /// Momentum buffer.
    buf: Vec<Vec<f64>>,
}

impl RmsProp {
    /// Create a new RMSProp optimizer.
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            weight_decay: 0.0,
            centered: false,
            v: Vec::new(),
            g_avg: Vec::new(),
            buf: Vec::new(),
        }
    }

    /// Set alpha (smoothing).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set momentum.
    #[must_use]
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable centered mode.
    #[must_use]
    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    /// Set weight decay.
    #[must_use]
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for RmsProp {
    #[allow(clippy::needless_range_loop)]
    fn step(&mut self, params: &mut [GpuTensor]) -> Result<(), TensorError> {
        if self.v.len() < params.len() {
            self.v.resize(params.len(), Vec::new());
            self.g_avg.resize(params.len(), Vec::new());
            self.buf.resize(params.len(), Vec::new());
        }

        for (i, param) in params.iter_mut().enumerate() {
            let mut grad = match param.grad() {
                Some(g) => g.host_data().to_vec(),
                None => continue,
            };

            let n = grad.len();
            if self.v[i].len() != n {
                self.v[i] = vec![0.0; n];
                self.g_avg[i] = vec![0.0; n];
                self.buf[i] = vec![0.0; n];
            }

            if self.weight_decay != 0.0 {
                for (g, &p) in grad.iter_mut().zip(param.host_data.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            for j in 0..n {
                self.v[i][j] = self.alpha * self.v[i][j] + (1.0 - self.alpha) * grad[j] * grad[j];
            }

            let avg = if self.centered {
                for j in 0..n {
                    self.g_avg[i][j] = self.alpha * self.g_avg[i][j] + (1.0 - self.alpha) * grad[j];
                }
                &self.g_avg[i]
            } else {
                // Not centered — we still allocate g_avg but don't use it
                &self.g_avg[i]
            };

            for j in 0..n {
                let v_val = if self.centered {
                    self.v[i][j] - avg[j] * avg[j]
                } else {
                    self.v[i][j]
                };

                if self.momentum > 0.0 {
                    self.buf[i][j] =
                        self.momentum * self.buf[i][j] + grad[j] / (v_val.sqrt() + self.eps);
                    param.host_data[j] -= self.lr * self.buf[i][j];
                } else {
                    param.host_data[j] -= self.lr * grad[j] / (v_val.sqrt() + self.eps);
                }
            }
        }

        Ok(())
    }
}

// ─── LAMB ───────────────────────────────────────────────────

/// LAMB optimizer for large-batch training (Layer-wise Adaptive Moments).
///
/// Based on "Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes"
/// (You et al., 2019).
#[derive(Debug, Clone)]
pub struct Lamb {
    /// Learning rate.
    pub lr: f64,
    /// Beta1 for first moment.
    pub beta1: f64,
    /// Beta2 for second moment.
    pub beta2: f64,
    /// Epsilon.
    pub eps: f64,
    /// Weight decay.
    pub weight_decay: f64,
    /// First moment.
    m: Vec<Vec<f64>>,
    /// Second moment.
    v: Vec<Vec<f64>>,
    /// Step counter.
    t: u64,
}

impl Lamb {
    /// Create a new LAMB optimizer.
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Set betas.
    #[must_use]
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set weight decay.
    #[must_use]
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Lamb {
    #[allow(clippy::needless_range_loop)]
    fn step(&mut self, params: &mut [GpuTensor]) -> Result<(), TensorError> {
        self.t += 1;

        if self.m.len() < params.len() {
            self.m.resize(params.len(), Vec::new());
            self.v.resize(params.len(), Vec::new());
        }

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, param) in params.iter_mut().enumerate() {
            let grad = match param.grad() {
                Some(g) => g.host_data().to_vec(),
                None => continue,
            };

            let n = grad.len();
            if self.m[i].len() != n {
                self.m[i] = vec![0.0; n];
                self.v[i] = vec![0.0; n];
            }

            // Update moments
            for j in 0..n {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * grad[j];
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * grad[j] * grad[j];
            }

            // Bias-corrected Adam update direction
            let mut update = vec![0.0; n];
            for j in 0..n {
                let m_hat = self.m[i][j] / bias_correction1;
                let v_hat = self.v[i][j] / bias_correction2;
                update[j] = m_hat / (v_hat.sqrt() + self.eps);
            }

            // Add weight decay
            if self.weight_decay != 0.0 {
                for (u, &p) in update.iter_mut().zip(param.host_data.iter()) {
                    *u += self.weight_decay * p;
                }
            }

            // Layer-wise trust ratio
            let w_norm: f64 = param.host_data.iter().map(|&p| p * p).sum::<f64>().sqrt();
            let u_norm: f64 = update.iter().map(|&u| u * u).sum::<f64>().sqrt();

            let trust_ratio = if w_norm > 0.0 && u_norm > 0.0 {
                w_norm / u_norm
            } else {
                1.0
            };

            for (p, &u) in param.host_data.iter_mut().zip(update.iter()) {
                *p -= self.lr * trust_ratio * u;
            }
        }

        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_param(data: &[f64], grad_data: &[f64]) -> GpuTensor {
        let mut p = GpuTensor::from_host_f64(data, &[data.len()], 0).unwrap();
        p.set_requires_grad(true);
        let g = GpuTensor::from_host_f64(grad_data, &[grad_data.len()], 0).unwrap();
        p.accumulate_grad(&g).unwrap();
        p
    }

    #[test]
    fn test_sgd_basic() {
        let mut opt = Sgd::new(0.1);
        let mut params = vec![make_param(&[5.0, 10.0], &[1.0, 2.0])];
        opt.step(&mut params).unwrap();
        // p -= 0.1 * grad => 5 - 0.1 = 4.9, 10 - 0.2 = 9.8
        assert!((params[0].host_data[0] - 4.9).abs() < 1e-10);
        assert!((params[0].host_data[1] - 9.8).abs() < 1e-10);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut opt = Sgd::new(0.1).with_momentum(0.9);
        let mut params = vec![make_param(&[5.0], &[1.0])];
        opt.step(&mut params).unwrap();
        // First step: velocity = grad = 1.0, p -= 0.1 * 1.0 = 4.9
        assert!((params[0].host_data[0] - 4.9).abs() < 1e-10);
    }

    #[test]
    fn test_adam_basic() {
        let mut opt = Adam::new(0.01);
        let mut params = vec![make_param(&[5.0], &[1.0])];
        opt.step(&mut params).unwrap();
        // After one step, parameter should have decreased
        assert!(params[0].host_data[0] < 5.0);
    }

    #[test]
    fn test_adam_convergence() {
        // Simple quadratic: f(x) = x^2, grad = 2x. Minimum at x=0.
        let mut opt = Adam::new(0.1);
        let mut params = vec![make_param(&[3.0], &[6.0])];

        for _ in 0..100 {
            opt.step(&mut params).unwrap();
            // Recompute grad = 2 * x
            let new_grad = 2.0 * params[0].host_data[0];
            params[0].zero_grad();
            let g = GpuTensor::from_host_f64(&[new_grad], &[1], 0).unwrap();
            params[0].accumulate_grad(&g).unwrap();
        }
        // Should be close to 0
        assert!(params[0].host_data[0].abs() < 0.5);
    }

    #[test]
    fn test_adagrad_basic() {
        let mut opt = AdaGrad::new(0.1);
        let mut params = vec![make_param(&[5.0], &[1.0])];
        opt.step(&mut params).unwrap();
        assert!(params[0].host_data[0] < 5.0);
    }

    #[test]
    fn test_rmsprop_basic() {
        let mut opt = RmsProp::new(0.01);
        let mut params = vec![make_param(&[5.0], &[1.0])];
        opt.step(&mut params).unwrap();
        assert!(params[0].host_data[0] < 5.0);
    }

    #[test]
    fn test_lamb_basic() {
        let mut opt = Lamb::new(0.01);
        let mut params = vec![make_param(&[5.0], &[1.0])];
        opt.step(&mut params).unwrap();
        assert!(params[0].host_data[0] < 5.0);
    }

    #[test]
    fn test_zero_grad() {
        let mut opt = Sgd::new(0.1);
        let mut params = vec![make_param(&[5.0], &[1.0])];
        assert!(params[0].grad().is_some());
        opt.zero_grad(&mut params);
        assert!(params[0].grad().is_none());
    }

    #[test]
    fn test_adam_with_weight_decay() {
        let mut opt = Adam::new(0.01).with_weight_decay(0.01);
        let mut params = vec![make_param(&[5.0], &[0.0])];
        opt.step(&mut params).unwrap();
        // Even with zero grad, weight decay should shrink the parameter
        assert!(params[0].host_data[0] < 5.0);
    }

    #[test]
    fn test_lamb_convergence() {
        let mut opt = Lamb::new(0.1);
        let mut params = vec![make_param(&[3.0], &[6.0])];

        for _ in 0..100 {
            opt.step(&mut params).unwrap();
            let new_grad = 2.0 * params[0].host_data[0];
            params[0].zero_grad();
            let g = GpuTensor::from_host_f64(&[new_grad], &[1], 0).unwrap();
            params[0].accumulate_grad(&g).unwrap();
        }
        assert!(params[0].host_data[0].abs() < 1.0);
    }

    #[test]
    fn test_sgd_weight_decay() {
        let mut opt = Sgd::new(0.1).with_weight_decay(0.01);
        let mut params = vec![make_param(&[10.0], &[0.0])];
        opt.step(&mut params).unwrap();
        // weight decay adds 0.01*10 = 0.1 to grad, then p -= 0.1 * 0.1 = 9.99
        assert!((params[0].host_data[0] - 9.99).abs() < 1e-10);
    }

    #[test]
    fn test_no_grad_no_update() {
        let mut opt = Sgd::new(0.1);
        // Parameter without gradient
        let mut p = GpuTensor::from_host_f64(&[5.0], &[1], 0).unwrap();
        p.set_requires_grad(true);
        let mut params = vec![p];
        opt.step(&mut params).unwrap();
        // Should be unchanged
        assert!((params[0].host_data[0] - 5.0).abs() < 1e-10);
    }
}
