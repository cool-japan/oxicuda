//! GPU-resident optimizer state management.
//!
//! All optimizers in this module keep their state (moment buffers, velocity
//! vectors, etc.) in host-side `Vec<f32>` that mirror the conceptual GPU
//! buffers and apply fused PTX update algorithms element-wise.  The interface
//! is designed so that each optimizer can be trivially backed by real
//! `DeviceBuffer` allocations once real GPU execution is wired up.
//!
//! ## Trait hierarchy
//!
//! ```text
//! GpuOptimizer
//!   ├─ GpuAdam   (two moments, bias correction)
//!   ├─ GpuAdamW  (two moments, decoupled weight decay)
//!   ├─ GpuLion   (one moment, sign update — memory-efficient)
//!   ├─ GpuCame   (row/col factored second moment — memory-efficient)
//!   └─ GpuMuon   (Nesterov + Newton-Schulz orthogonalisation)
//! ```

pub mod adagrad;
pub mod adam;
pub mod adamw;
pub mod came;
pub mod lion;
pub mod muon;
pub mod radam;
pub mod rmsprop;

use crate::error::{TrainError, TrainResult};

// ─── Flat parameter buffer ────────────────────────────────────────────────────

/// A flattened, CPU-accessible representation of a single model parameter
/// together with its gradient.
///
/// In a real GPU training loop these would point to `DeviceBuffer` memory;
/// here they are `Vec<f32>` for portability.
#[derive(Debug, Clone)]
pub struct ParamTensor {
    /// Current parameter values (flat).
    pub data: Vec<f32>,
    /// Gradient (same length as `data`), or `None` if not yet computed.
    pub grad: Option<Vec<f32>>,
    /// Human-readable name for debugging / scheduler lookup.
    pub name: String,
    /// Whether this parameter requires gradient computation.
    pub requires_grad: bool,
}

impl ParamTensor {
    /// Create a parameter from a flat data vector.
    #[must_use]
    pub fn new(data: Vec<f32>, name: impl Into<String>) -> Self {
        Self {
            data,
            grad: None,
            name: name.into(),
            requires_grad: true,
        }
    }

    /// Create a zero-filled parameter.
    #[must_use]
    pub fn zeros(n: usize, name: impl Into<String>) -> Self {
        Self::new(vec![0.0_f32; n], name)
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the parameter has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Set the gradient to the provided vector.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if `grad.len() != self.data.len()`.
    pub fn set_grad(&mut self, grad: Vec<f32>) -> TrainResult<()> {
        if grad.len() != self.data.len() {
            return Err(TrainError::ShapeMismatch {
                expected: vec![self.data.len()],
                got: vec![grad.len()],
            });
        }
        self.grad = Some(grad);
        Ok(())
    }

    /// Zero out the gradient buffer (keeps allocation).
    pub fn zero_grad(&mut self) {
        if let Some(g) = &mut self.grad {
            g.iter_mut().for_each(|v| *v = 0.0_f32);
        }
    }

    /// Accumulate `other` gradient into `self.grad` (used for multi-step
    /// gradient accumulation).
    ///
    /// Initialises `self.grad` to a zero vector if it is `None`.
    pub fn accumulate_grad(&mut self, other: &[f32]) -> TrainResult<()> {
        if other.len() != self.data.len() {
            return Err(TrainError::ShapeMismatch {
                expected: vec![self.data.len()],
                got: vec![other.len()],
            });
        }
        let grad = self
            .grad
            .get_or_insert_with(|| vec![0.0_f32; self.data.len()]);
        for (g, o) in grad.iter_mut().zip(other.iter()) {
            *g += o;
        }
        Ok(())
    }
}

// ─── GpuOptimizer trait ───────────────────────────────────────────────────────

/// Common interface for all GPU-resident optimizers.
///
/// Implementors hold all per-parameter state buffers (moments, velocity, etc.)
/// and apply a fused update to `params` in one call to `step`.
pub trait GpuOptimizer {
    /// Perform one optimisation step: read `param.grad`, update state buffers,
    /// and write new parameter values to `param.data`.
    ///
    /// Gradients are **not** zeroed after the step — call [`GpuOptimizer::zero_grad`] if
    /// desired.
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] – `params` is empty.
    /// * [`TrainError::NoGradient`] – a parameter's gradient is `None`.
    fn step(&mut self, params: &mut [ParamTensor]) -> TrainResult<()>;

    /// Zero all gradients on the given parameters.
    fn zero_grad(&self, params: &mut [ParamTensor]) {
        for p in params.iter_mut() {
            p.zero_grad();
        }
    }

    /// Current learning rate (base, before scheduler scaling).
    fn lr(&self) -> f64;

    /// Update the learning rate (called by LR schedulers).
    fn set_lr(&mut self, lr: f64);

    /// Name tag for display / serialisation.
    fn name(&self) -> &str;
}

// ─── Optimizer state snapshot ────────────────────────────────────────────────

/// Serialisable snapshot of optimizer hyperparameters (used by checkpoint).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimizerConfig {
    /// Optimizer variant name (e.g. `"AdamW"`).
    pub name: String,
    /// Base learning rate.
    pub lr: f64,
    /// Current global step count.
    pub step: u64,
    /// Extra named hyperparameters (β₁, β₂, ε, wd, …).
    pub hparams: std::collections::HashMap<String, f64>,
}

// ─── Shared utility: bias-correction scalars ─────────────────────────────────

/// Compute Adam bias-correction scalars for step `t`.
///
/// Returns `(step_size, bc2_rsqrt)` where:
/// * `step_size  = lr / (1 − β₁ᵗ)`
/// * `bc2_rsqrt  = 1 / √(1 − β₂ᵗ)`
#[inline]
pub(crate) fn adam_bias_corrections(lr: f64, beta1: f64, beta2: f64, step: u64) -> (f32, f32) {
    let t = step as f64;
    let bc1 = 1.0 - beta1.powf(t);
    let bc2 = 1.0 - beta2.powf(t);
    let step_size = (lr / bc1) as f32;
    let bc2_rsqrt = (1.0 / bc2.sqrt()) as f32;
    (step_size, bc2_rsqrt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_tensor_set_grad_ok() {
        let mut p = ParamTensor::zeros(4, "w");
        p.set_grad(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(p.grad.as_ref().unwrap()[2], 3.0);
    }

    #[test]
    fn param_tensor_set_grad_shape_mismatch() {
        let mut p = ParamTensor::zeros(4, "w");
        assert!(p.set_grad(vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn param_tensor_accumulate_grad() {
        let mut p = ParamTensor::zeros(3, "w");
        p.accumulate_grad(&[1.0, 2.0, 3.0]).unwrap();
        p.accumulate_grad(&[0.5, 0.5, 0.5]).unwrap();
        let g = p.grad.unwrap();
        assert!((g[0] - 1.5).abs() < 1e-6);
        assert!((g[1] - 2.5).abs() < 1e-6);
        assert!((g[2] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn param_tensor_zero_grad() {
        let mut p = ParamTensor::zeros(3, "w");
        p.set_grad(vec![1.0, 2.0, 3.0]).unwrap();
        p.zero_grad();
        let g = p.grad.unwrap();
        assert_eq!(g, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn adam_bias_corrections_step1() {
        // At step 1 with standard Adam defaults: beta1=0.9, beta2=0.999, lr=1e-3
        let (step_size, bc2_rsqrt) = adam_bias_corrections(1e-3, 0.9, 0.999, 1);
        // bc1 = 1-0.9 = 0.1, step_size = 0.001/0.1 = 0.01
        assert!((step_size - 0.01_f32).abs() < 1e-5, "step_size={step_size}");
        // bc2 = 1-0.999 = 0.001, bc2_rsqrt = 1/sqrt(0.001) ≈ 31.62
        assert!(
            bc2_rsqrt > 31.0 && bc2_rsqrt < 32.0,
            "bc2_rsqrt={bc2_rsqrt}"
        );
    }

    #[test]
    fn param_tensor_len() {
        let p = ParamTensor::zeros(16, "x");
        assert_eq!(p.len(), 16);
        assert!(!p.is_empty());
        let empty = ParamTensor::zeros(0, "e");
        assert!(empty.is_empty());
    }
}
