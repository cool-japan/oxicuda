//! # AdaGrad GPU Optimizer
//!
//! AdaGrad (Adaptive Gradient Algorithm, Duchi et al. 2011) maintains a
//! per-parameter *sum-of-squared-gradients* accumulator `G` and scales the
//! update by `1 / (√G + ε)`:
//!
//! ```text
//! G[i]  +=  g[i]²
//! p[i]  -=  lr / (√G[i] + ε) * g[i]
//! ```
//!
//! Unlike Adam, AdaGrad never forgets past gradients, making it effective for
//! sparse features (e.g. word embeddings) where important gradients are rare.
//! The downside is that `G` grows monotonically so the effective learning rate
//! decays to zero; for this reason AdaGrad is typically used with short
//! training runs or in conjunction with `initial_accumulator_value > 0` to
//! avoid very large early steps.

use crate::error::{TrainError, TrainResult};
use crate::gpu_optimizer::{GpuOptimizer, ParamTensor};

/// AdaGrad optimizer with per-parameter sum-of-squared-gradients accumulator.
#[derive(Debug, Clone)]
pub struct GpuAdaGrad {
    /// Learning rate.
    lr: f64,
    /// Numerical stability constant added to the denominator.
    eps: f64,
    /// Initial value for the `G` accumulator (warm-starts denominator).
    initial_accumulator: f64,
    /// Per-parameter G accumulator buffers.
    accumulator: Vec<Vec<f32>>,
    /// Number of steps taken.
    step: u64,
}

impl GpuAdaGrad {
    /// Create AdaGrad with the given learning rate.
    ///
    /// Defaults: `eps = 1e-10`, `initial_accumulator = 0.1`.
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            eps: 1e-10,
            initial_accumulator: 0.1,
            accumulator: Vec::new(),
            step: 0,
        }
    }

    /// Set the numerical stability term ε.
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the initial accumulator value (prevents very large first steps).
    #[must_use]
    pub fn with_initial_accumulator(mut self, val: f64) -> Self {
        self.initial_accumulator = val;
        self
    }

    /// Current step count.
    #[must_use]
    #[inline]
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Initialise (or re-initialise) accumulators from the parameter list.
    fn ensure_state(&mut self, params: &[ParamTensor]) {
        if self.accumulator.len() != params.len() {
            self.accumulator = params
                .iter()
                .map(|p| vec![self.initial_accumulator as f32; p.data.len()])
                .collect();
        }
    }
}

impl GpuOptimizer for GpuAdaGrad {
    fn step(&mut self, params: &mut [ParamTensor]) -> TrainResult<()> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        self.ensure_state(params);
        self.step += 1;
        let lr = self.lr as f32;
        let eps = self.eps as f32;

        for (i, param) in params.iter_mut().enumerate() {
            if !param.requires_grad {
                continue;
            }
            let grad = param
                .grad
                .as_ref()
                .ok_or(TrainError::NoGradient { index: i })?;
            let accum = &mut self.accumulator[i];
            for ((p, g), acc) in param.data.iter_mut().zip(grad.iter()).zip(accum.iter_mut()) {
                *acc += g * g;
                let denom = acc.sqrt() + eps;
                *p -= lr / denom * g;
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
        "AdaGrad"
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_param(val: f32) -> ParamTensor {
        let mut p = ParamTensor::new(vec![val], "p");
        p.set_grad(vec![2.0 * val]).unwrap();
        p
    }

    #[test]
    fn adagrad_reduces_param() {
        let mut opt = GpuAdaGrad::new(0.1);
        let mut params = vec![make_param(4.0)];
        opt.step(&mut params).unwrap();
        assert!(params[0].data[0] < 4.0, "param should decrease");
    }

    #[test]
    fn adagrad_converges() {
        let mut opt = GpuAdaGrad::new(0.5).with_initial_accumulator(0.0);
        let mut params = vec![ParamTensor::new(vec![3.0_f32], "x")];
        for _ in 0..200 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            opt.step(&mut params).unwrap();
        }
        let x = params[0].data[0].abs();
        assert!(x < 1.0, "AdaGrad should converge, |x|={x}");
    }

    #[test]
    fn adagrad_empty_error() {
        let mut opt = GpuAdaGrad::new(0.1);
        assert!(opt.step(&mut []).is_err());
    }

    #[test]
    fn adagrad_no_grad_error() {
        let mut opt = GpuAdaGrad::new(0.1);
        let mut params = vec![ParamTensor::new(vec![1.0], "p")];
        assert!(opt.step(&mut params).is_err());
    }

    #[test]
    fn adagrad_step_count() {
        let mut opt = GpuAdaGrad::new(0.1);
        let mut params = vec![make_param(1.0)];
        for _ in 0..5 {
            params[0].set_grad(vec![0.1]).unwrap();
            opt.step(&mut params).unwrap();
        }
        assert_eq!(opt.step_count(), 5);
    }

    #[test]
    fn adagrad_set_lr() {
        let mut opt = GpuAdaGrad::new(0.1);
        opt.set_lr(0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn adagrad_name() {
        assert_eq!(GpuAdaGrad::new(0.1).name(), "AdaGrad");
    }

    #[test]
    fn adagrad_accumulator_grows_monotonically() {
        let mut opt = GpuAdaGrad::new(0.01).with_initial_accumulator(0.0);
        let mut params = vec![ParamTensor::new(vec![1.0], "p")];
        let mut last_acc = 0.0_f32;
        for _ in 0..5 {
            params[0].set_grad(vec![1.0]).unwrap();
            opt.step(&mut params).unwrap();
            let acc = opt.accumulator[0][0];
            assert!(
                acc >= last_acc,
                "accumulator must be monotonically increasing"
            );
            last_acc = acc;
        }
    }
}
