//! GPU Lion optimizer — memory-efficient signed-update optimizer.
//!
//! Lion ("EvoLved Sign Momentum", Chen et al. 2023) achieves competitive
//! performance with Adam while requiring only **one** moment buffer per
//! parameter (half the memory of Adam's two buffers).
//!
//! ## Algorithm (per step)
//!
//! ```text
//! c ← β₁·m + (1−β₁)·g         (interpolate moment with gradient)
//! p ← p·(1 − lr·λ) − lr·sign(c) (apply decoupled WD + signed step)
//! m ← β₂·m + (1−β₂)·g          (update moment for next step)
//! ```
//!
//! Note: the update direction `c` uses `β₁` (default 0.9) while the moment
//! update uses `β₂` (default 0.99).  This asymmetry is intentional and
//! critical to Lion's behaviour.

use super::{GpuOptimizer, ParamTensor};
use crate::error::{TrainError, TrainResult};

/// GPU Lion optimizer.
#[derive(Debug, Clone)]
pub struct GpuLion {
    lr: f64,
    /// Interpolation coefficient for update direction (default 0.9).
    beta1: f64,
    /// Interpolation coefficient for moment update (default 0.99).
    beta2: f64,
    /// Decoupled weight decay coefficient λ.
    weight_decay: f64,
    step_count: u64,
    /// Per-parameter first moment buffers.
    exp_avg: Vec<Vec<f32>>,
}

impl GpuLion {
    /// Create a new Lion optimizer.
    ///
    /// # Defaults
    ///
    /// * `beta1 = 0.9`, `beta2 = 0.99`, `weight_decay = 0.0`
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
            step_count: 0,
            exp_avg: Vec::new(),
        }
    }

    /// Set β₁ (update direction interpolation).
    #[must_use]
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set β₂ (moment update interpolation).
    #[must_use]
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
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
        }
        for (i, p) in params.iter().enumerate() {
            if self.exp_avg[i].len() != p.len() {
                self.exp_avg[i] = vec![0.0_f32; p.len()];
            }
        }
    }

    fn apply_update(
        data: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        wd_factor: f32,
    ) {
        let c1 = 1.0_f32 - beta1;
        let c2 = 1.0_f32 - beta2;

        for idx in 0..data.len() {
            let g = grad[idx];
            let mi = m[idx];

            // c = beta1*m + (1-beta1)*g  (direction)
            let c = beta1 * mi + c1 * g;

            // sign(c): +1 / -1 / 0
            let sgn = if c > 0.0_f32 {
                1.0_f32
            } else if c < 0.0_f32 {
                -1.0_f32
            } else {
                0.0_f32
            };

            // Decoupled weight decay + signed step
            data[idx] = data[idx] * wd_factor - lr * sgn;

            // Moment update (uses beta2, NOT beta1)
            m[idx] = beta2 * mi + c2 * g;
        }
    }
}

impl GpuOptimizer for GpuLion {
    fn step(&mut self, params: &mut [ParamTensor]) -> TrainResult<()> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        self.ensure_states(params);
        self.step_count += 1;

        let lr = self.lr as f32;
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        // Decoupled WD factor = 1 - lr*wd
        let wd_factor = (1.0 - self.lr * self.weight_decay) as f32;

        for (i, param) in params.iter_mut().enumerate() {
            if !param.requires_grad {
                continue;
            }
            let grad = param
                .grad
                .as_ref()
                .ok_or(TrainError::NoGradient { index: i })?;

            let m = &mut self.exp_avg[i];
            Self::apply_update(&mut param.data, grad, m, lr, beta1, beta2, wd_factor);
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
        "Lion"
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
    fn lion_positive_grad_decreases_param() {
        let mut opt = GpuLion::new(1e-3);
        let mut params = vec![make_param(vec![1.0_f32; 4], vec![1.0_f32; 4])];
        opt.step(&mut params).unwrap();
        // sign(c) > 0, so p decreases
        for &v in &params[0].data {
            assert!(v < 1.0, "param should decrease, got {v}");
        }
    }

    #[test]
    fn lion_negative_grad_increases_param() {
        let mut opt = GpuLion::new(1e-3);
        let mut params = vec![make_param(vec![0.0_f32; 4], vec![-1.0_f32; 4])];
        opt.step(&mut params).unwrap();
        // sign(c) < 0, so p increases
        for &v in &params[0].data {
            assert!(v > 0.0, "param should increase, got {v}");
        }
    }

    #[test]
    fn lion_update_magnitude_equals_lr() {
        // With zero moment and gradient = 1, first step should change p by exactly lr
        let lr = 1e-2_f64;
        let mut opt = GpuLion::new(lr);
        let mut params = vec![make_param(vec![5.0_f32], vec![1.0_f32])];
        opt.step(&mut params).unwrap();
        // c = 0 + (1-beta1)*1.0 > 0, so sign = 1
        // p = p - lr * 1 = 5 - 0.01 = 4.99 (approximately — weight_decay=0)
        let expected = 5.0_f32 - lr as f32;
        assert!(
            (params[0].data[0] - expected).abs() < 1e-5,
            "expected ~{expected}, got {}",
            params[0].data[0]
        );
    }

    #[test]
    fn lion_zero_gradient_no_change_no_wd() {
        // zero gradient + zero weight decay → parameter should not change
        let mut opt = GpuLion::new(1e-3).with_weight_decay(0.0);
        let mut params = vec![make_param(vec![3.0_f32; 4], vec![0.0_f32; 4])];
        opt.step(&mut params).unwrap();
        for &v in &params[0].data {
            // sign(c) = sign(0) = 0, so p unchanged
            assert!((v - 3.0).abs() < 1e-6, "param should be unchanged, got {v}");
        }
    }

    #[test]
    fn lion_only_one_moment_buffer() {
        let mut opt = GpuLion::new(1e-3);
        let mut params = vec![make_param(vec![1.0_f32; 8], vec![0.5_f32; 8])];
        opt.step(&mut params).unwrap();
        // Only exp_avg should be populated (no exp_avg_sq)
        assert_eq!(opt.exp_avg.len(), 1);
        assert_eq!(opt.exp_avg[0].len(), 8);
    }

    #[test]
    fn lion_convergence() {
        // Lion takes exactly ±lr per step regardless of gradient magnitude.
        // With lr=1e-2 starting at x=5, 500 steps × 0.01 = 5 total displacement
        // so x crosses 0 around step 500 and the final |x| should be small.
        let mut opt = GpuLion::new(1e-2);
        let mut params = vec![make_param(vec![5.0_f32], vec![0.0_f32])];
        for _ in 0..500 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            opt.step(&mut params).unwrap();
        }
        let x = params[0].data[0].abs();
        assert!(x < 0.5, "should converge near 0, got |x|={x}");
    }

    #[test]
    fn lion_name() {
        assert_eq!(GpuLion::new(1e-3).name(), "Lion");
    }
}
