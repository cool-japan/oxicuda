//! GPU AdamW optimizer — Adam with decoupled weight decay.
//!
//! AdamW (Loshchilov & Hutter, 2019) fixes the coupling of weight decay with
//! the adaptive learning rate in standard Adam by applying weight decay
//! **directly to the parameter** before the gradient step, independent of the
//! moment estimates.
//!
//! ## Difference from Adam
//!
//! | | Adam | AdamW |
//! |---|---|---|
//! | Weight decay | Added to gradient: `g' = g + λp` | Applied to param: `p *= (1 − lr·λ)` |
//! | Moment update | Uses `g'` (decays coupled) | Uses raw `g` (decoupled) |
//!
//! ## Algorithm
//!
//! ```text
//! p ← p · (1 − lr · λ)          (decoupled weight decay)
//! m ← β₁·m + (1−β₁)·g
//! v ← β₂·v + (1−β₂)·g²
//! p ← p − step_size · m̂ / (√v̂ + ε)
//! ```

use super::{GpuOptimizer, ParamTensor, adam_bias_corrections};
use crate::error::{TrainError, TrainResult};

/// GPU AdamW optimizer state.
#[derive(Debug, Clone)]
pub struct GpuAdamW {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    /// Decoupled weight decay coefficient λ.
    weight_decay: f64,
    step_count: u64,
    exp_avg: Vec<Vec<f32>>,
    exp_avg_sq: Vec<Vec<f32>>,
}

impl GpuAdamW {
    /// Create a new AdamW optimizer.
    ///
    /// # Defaults
    ///
    /// * `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`, `weight_decay = 1e-2`
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step_count: 0,
            exp_avg: Vec::new(),
            exp_avg_sq: Vec::new(),
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

    /// Set decoupled weight decay λ.
    #[must_use]
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Current step count.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    fn ensure_states(&mut self, params: &[ParamTensor]) {
        if self.exp_avg.len() < params.len() {
            self.exp_avg.resize_with(params.len(), Vec::new);
            self.exp_avg_sq.resize_with(params.len(), Vec::new);
        }
        for (i, p) in params.iter().enumerate() {
            if self.exp_avg[i].len() != p.len() {
                self.exp_avg[i] = vec![0.0_f32; p.len()];
                self.exp_avg_sq[i] = vec![0.0_f32; p.len()];
            }
        }
    }

    fn apply_update(
        data: &mut [f32],
        grad: &[f32],
        m1: &mut [f32],
        m2: &mut [f32],
        step_size: f32,
        bc2_rsqrt: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd_factor: f32,
    ) {
        let c1 = 1.0_f32 - beta1;
        let c2 = 1.0_f32 - beta2;

        for idx in 0..data.len() {
            // Decoupled weight decay: p *= (1 - lr*wd)
            data[idx] *= wd_factor;

            let g = grad[idx];

            // Moment updates (no weight decay contamination)
            let nm1 = beta1 * m1[idx] + c1 * g;
            let nm2 = beta2 * m2[idx] + c2 * g * g;
            m1[idx] = nm1;
            m2[idx] = nm2;

            // Bias-corrected denominator
            let sq_v_hat = (nm2 * bc2_rsqrt * bc2_rsqrt).sqrt();
            let denom = sq_v_hat + eps;

            // Parameter update
            data[idx] -= step_size * nm1 / denom;
        }
    }
}

impl GpuOptimizer for GpuAdamW {
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

        // Decoupled weight decay factor = 1 - lr*wd
        let wd_factor = (1.0 - self.lr * self.weight_decay) as f32;

        for (i, param) in params.iter_mut().enumerate() {
            if !param.requires_grad {
                continue;
            }
            let grad = param
                .grad
                .as_ref()
                .ok_or(TrainError::NoGradient { index: i })?;

            let m1 = &mut self.exp_avg[i];
            let m2 = &mut self.exp_avg_sq[i];
            Self::apply_update(
                &mut param.data,
                grad,
                m1,
                m2,
                step_size,
                bc2_rsqrt,
                beta1,
                beta2,
                eps,
                wd_factor,
            );
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
        "AdamW"
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
    fn adamw_step_decreases_param() {
        let mut opt = GpuAdamW::new(1e-3);
        let mut params = vec![make_param(vec![1.0_f32; 4], vec![0.5_f32; 4])];
        opt.step(&mut params).unwrap();
        for &v in &params[0].data {
            assert!(v < 1.0, "param should decrease");
        }
    }

    #[test]
    fn adamw_weight_decay_reduces_large_params() {
        // With large weight decay, even a zero gradient should shrink params.
        let mut opt = GpuAdamW::new(1.0).with_weight_decay(0.1);
        let mut params = vec![make_param(vec![10.0_f32; 4], vec![0.0_f32; 4])];
        opt.step(&mut params).unwrap();
        // p *= (1 - 1.0*0.1) = 0.9, so p shrinks from 10 to < 10
        for &v in &params[0].data {
            assert!(
                v < 10.0,
                "large param should be shrunk by weight decay, got {v}"
            );
        }
    }

    #[test]
    fn adamw_decoupled_wd_vs_adam_l2() {
        // AdamW with wd should produce different results than Adam with wd.
        use crate::gpu_optimizer::adam::GpuAdam;

        let data = vec![2.0_f32; 4];
        let grad = vec![1.0_f32; 4];

        let mut adam = GpuAdam::new(1e-3).with_weight_decay(0.01);
        let mut adamw = GpuAdamW::new(1e-3).with_weight_decay(0.01);

        let mut p_adam = vec![make_param(data.clone(), grad.clone())];
        let mut p_adamw = vec![make_param(data.clone(), grad.clone())];

        adam.step(&mut p_adam).unwrap();
        adamw.step(&mut p_adamw).unwrap();

        // Results should differ due to decoupled vs coupled weight decay.
        let differ = p_adam[0]
            .data
            .iter()
            .zip(p_adamw[0].data.iter())
            .any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(differ, "Adam and AdamW should produce different updates");
    }

    #[test]
    fn adamw_converges_to_zero() {
        // Use beta2=0.9 so the second-moment correction converges in ~20 steps;
        // the default beta2=0.999 would require ~1000+ steps for this quadratic.
        let mut opt = GpuAdamW::new(1e-2).with_weight_decay(0.0).with_beta2(0.9);
        let mut params = vec![make_param(vec![3.0_f32], vec![0.0_f32])];

        for _ in 0..300 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            opt.step(&mut params).unwrap();
        }
        assert!(
            params[0].data[0].abs() < 0.1,
            "should converge near 0, got {}",
            params[0].data[0]
        );
    }

    #[test]
    fn adamw_name() {
        assert_eq!(GpuAdamW::new(1e-3).name(), "AdamW");
    }
}
