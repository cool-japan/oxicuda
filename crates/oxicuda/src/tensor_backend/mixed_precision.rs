//! FP16/BF16 mixed-precision training with loss scaling and autocast.
//!
//! Provides [`GradScaler`] for dynamic loss scaling to prevent FP16 underflow,
//! and [`Autocast`] for automatic precision selection per operation.

use super::dtype::{PrecisionCategory, TensorDtype};
use super::error::TensorError;
use super::optimizer::Optimizer;
use super::tensor::GpuTensor;

// ─── GradScaler ─────────────────────────────────────────────

/// Dynamic loss scaler for mixed-precision training.
///
/// Scales the loss by a large factor before backward to prevent FP16 gradient
/// underflow, then unscales gradients before the optimizer step. Automatically
/// adjusts the scale factor based on whether inf/nan gradients are detected.
///
/// # Example
///
/// ```rust
/// use oxicuda::tensor_backend::mixed_precision::GradScaler;
///
/// let scaler = GradScaler::new(65536.0);
/// assert!((scaler.scale() - 65536.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct GradScaler {
    /// Current scale factor.
    scale_factor: f64,
    /// Growth factor applied when no inf/nan is found.
    growth_factor: f64,
    /// Backoff factor applied when inf/nan is detected.
    backoff_factor: f64,
    /// Number of consecutive steps without inf/nan required to grow.
    growth_interval: u64,
    /// Counter of consecutive good steps.
    good_steps: u64,
    /// Whether inf/nan was found in the latest unscale.
    found_inf: bool,
    /// Whether gradients have been unscaled (to prevent double-unscale).
    unscaled: bool,
}

impl GradScaler {
    /// Create a new grad scaler with the given initial scale.
    #[must_use]
    pub fn new(init_scale: f64) -> Self {
        Self {
            scale_factor: init_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            good_steps: 0,
            found_inf: false,
            unscaled: false,
        }
    }

    /// Current scale factor.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale_factor
    }

    /// Set custom growth factor.
    #[must_use]
    pub fn with_growth_factor(mut self, gf: f64) -> Self {
        self.growth_factor = gf;
        self
    }

    /// Set custom backoff factor.
    #[must_use]
    pub fn with_backoff_factor(mut self, bf: f64) -> Self {
        self.backoff_factor = bf;
        self
    }

    /// Set growth interval.
    #[must_use]
    pub fn with_growth_interval(mut self, interval: u64) -> Self {
        self.growth_interval = interval;
        self
    }

    /// Scale the loss before backward pass.
    ///
    /// Returns a new tensor with `loss * scale_factor`.
    pub fn scale_loss(&mut self, loss: &GpuTensor) -> Result<GpuTensor, TensorError> {
        self.unscaled = false;
        self.found_inf = false;
        let scaled_data: Vec<f64> = loss
            .host_data()
            .iter()
            .map(|&v| v * self.scale_factor)
            .collect();
        GpuTensor::from_host_f64(&scaled_data, loss.shape(), loss.device_id())
    }

    /// Unscale gradients on all parameters (divide by scale factor).
    ///
    /// Sets `found_inf = true` if any gradient contains inf or nan.
    pub fn unscale_gradients(&mut self, params: &mut [GpuTensor]) -> Result<(), TensorError> {
        if self.unscaled {
            return Err(TensorError::MixedPrecisionError(
                "gradients already unscaled".into(),
            ));
        }
        self.unscaled = true;
        let inv_scale = 1.0 / self.scale_factor;

        for param in params.iter_mut() {
            if let Some(grad) = &mut param.grad {
                for v in grad.host_data.iter_mut() {
                    *v *= inv_scale;
                    if v.is_nan() || v.is_infinite() {
                        self.found_inf = true;
                    }
                }
            }
        }
        Ok(())
    }

    /// Perform an optimizer step, skipping if inf/nan was detected.
    ///
    /// Returns `true` if the step was taken, `false` if skipped.
    pub fn step(
        &mut self,
        optimizer: &mut dyn Optimizer,
        params: &mut [GpuTensor],
    ) -> Result<bool, TensorError> {
        if !self.unscaled {
            self.unscale_gradients(params)?;
        }
        if self.found_inf {
            return Ok(false);
        }
        optimizer.step(params)?;
        Ok(true)
    }

    /// Update the scale factor after the step.
    ///
    /// If inf/nan was found, reduce the scale. Otherwise, count a good step
    /// and increase the scale after `growth_interval` consecutive good steps.
    pub fn update(&mut self) {
        if self.found_inf {
            self.scale_factor *= self.backoff_factor;
            self.good_steps = 0;
        } else {
            self.good_steps += 1;
            if self.good_steps >= self.growth_interval {
                self.scale_factor *= self.growth_factor;
                self.good_steps = 0;
            }
        }
        self.found_inf = false;
        self.unscaled = false;
    }

    /// Whether inf/nan was found in the last unscale.
    #[must_use]
    pub fn found_inf(&self) -> bool {
        self.found_inf
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new(65536.0)
    }
}

// ─── Autocast ───────────────────────────────────────────────

/// Automatic mixed-precision context.
///
/// When active, maps operations to their optimal precision:
/// - Matrix multiplications, convolutions → FP16 (fast on Tensor Cores)
/// - Reductions, softmax, loss computation → FP32 (numerically stable)
/// - Other operations → keep input precision
#[derive(Debug, Clone)]
pub struct Autocast {
    /// Whether autocast is enabled.
    enabled: bool,
    /// Target low-precision dtype (Float16 or BFloat16).
    low_precision_dtype: TensorDtype,
    /// Target full-precision dtype.
    full_precision_dtype: TensorDtype,
}

impl Autocast {
    /// Create a new autocast context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: true,
            low_precision_dtype: TensorDtype::Float16,
            full_precision_dtype: TensorDtype::Float32,
        }
    }

    /// Enable or disable autocast.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether autocast is currently enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set the low-precision dtype (Float16 or BFloat16).
    pub fn set_low_precision(&mut self, dtype: TensorDtype) -> Result<(), TensorError> {
        if !dtype.is_half() {
            return Err(TensorError::MixedPrecisionError(format!(
                "low precision dtype must be Float16 or BFloat16, got {dtype}"
            )));
        }
        self.low_precision_dtype = dtype;
        Ok(())
    }

    /// Get the low-precision dtype.
    #[must_use]
    pub fn low_precision_dtype(&self) -> TensorDtype {
        self.low_precision_dtype
    }

    /// Get the full-precision dtype.
    #[must_use]
    pub fn full_precision_dtype(&self) -> TensorDtype {
        self.full_precision_dtype
    }

    /// Determine the optimal dtype for a given operation category.
    #[must_use]
    pub fn resolve_dtype(
        &self,
        category: PrecisionCategory,
        input_dtype: TensorDtype,
    ) -> TensorDtype {
        if !self.enabled {
            return input_dtype;
        }
        match category {
            PrecisionCategory::LowPrecision => self.low_precision_dtype,
            PrecisionCategory::FullPrecision => self.full_precision_dtype,
            PrecisionCategory::PassThrough => input_dtype,
        }
    }

    /// Classify a standard operation into its precision category.
    #[must_use]
    pub fn classify_op(op_name: &str) -> PrecisionCategory {
        match op_name {
            "matmul" | "conv2d" | "linear" | "bmm" => PrecisionCategory::LowPrecision,
            "softmax" | "log_softmax" | "cross_entropy" | "mse_loss" | "l1_loss"
            | "smooth_l1_loss" | "nll_loss" | "layer_norm" | "batch_norm" | "group_norm"
            | "sum" | "mean" => PrecisionCategory::FullPrecision,
            _ => PrecisionCategory::PassThrough,
        }
    }

    /// Cast a tensor to the specified dtype (metadata-only, no data conversion
    /// in CPU-simulated mode since we always store f64 internally).
    pub fn cast_tensor(
        tensor: &GpuTensor,
        target_dtype: TensorDtype,
    ) -> Result<GpuTensor, TensorError> {
        if tensor.dtype() == target_dtype {
            return Ok(tensor.clone());
        }
        Ok(GpuTensor::from_parts(
            tensor.shape().to_vec(),
            target_dtype,
            tensor.device_id(),
            tensor.host_data().to_vec(),
            tensor.requires_grad(),
            None,
        ))
    }
}

impl Default for Autocast {
    fn default() -> Self {
        Self::new()
    }
}

// ─── RAII Autocast guard ────────────────────────────────────

use std::cell::RefCell;

thread_local! {
    static AUTOCAST_STACK: RefCell<Vec<Autocast>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard for autocast context.
///
/// While this guard exists, autocast is active on the current thread.
pub struct AutocastGuard {
    _private: (),
}

/// Enter an autocast context.
pub fn enter_autocast(autocast: Autocast) -> AutocastGuard {
    AUTOCAST_STACK.with(|stack| {
        stack.borrow_mut().push(autocast);
    });
    AutocastGuard { _private: () }
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        AUTOCAST_STACK.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

/// Get the current autocast context, if any.
pub fn current_autocast() -> Option<Autocast> {
    AUTOCAST_STACK.with(|stack| stack.borrow().last().cloned())
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_scaler_basic() {
        let scaler = GradScaler::new(1024.0);
        assert!((scaler.scale() - 1024.0).abs() < 1e-10);
        assert!(!scaler.found_inf());
    }

    #[test]
    fn test_grad_scaler_scale_loss() {
        let mut scaler = GradScaler::new(100.0);
        let loss = GpuTensor::from_host_f64(&[0.5], &[1], 0).unwrap();
        let scaled = scaler.scale_loss(&loss).unwrap();
        assert!((scaled.host_data()[0] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_scaler_unscale() {
        let mut scaler = GradScaler::new(100.0);
        let _loss = GpuTensor::from_host_f64(&[0.5], &[1], 0).unwrap();

        let mut param = GpuTensor::from_host_f64(&[1.0], &[1], 0).unwrap();
        param.set_requires_grad(true);
        let grad = GpuTensor::from_host_f64(&[200.0], &[1], 0).unwrap();
        param.accumulate_grad(&grad).unwrap();

        let mut params = vec![param];
        scaler.unscale_gradients(&mut params).unwrap();

        // 200 / 100 = 2.0
        let g = params[0].grad().unwrap();
        assert!((g.host_data()[0] - 2.0).abs() < 1e-10);
        assert!(!scaler.found_inf());
    }

    #[test]
    fn test_grad_scaler_inf_detection() {
        let mut scaler = GradScaler::new(1.0);

        let mut param = GpuTensor::from_host_f64(&[1.0], &[1], 0).unwrap();
        param.set_requires_grad(true);
        let grad = GpuTensor::from_host_f64(&[f64::INFINITY], &[1], 0).unwrap();
        param.accumulate_grad(&grad).unwrap();

        let mut params = vec![param];
        scaler.unscale_gradients(&mut params).unwrap();
        assert!(scaler.found_inf());
    }

    #[test]
    fn test_grad_scaler_update_growth() {
        let mut scaler = GradScaler::new(100.0).with_growth_interval(2);
        // Two consecutive good steps should double the scale
        scaler.found_inf = false;
        scaler.update();
        assert!((scaler.scale() - 100.0).abs() < 1e-10);
        scaler.update();
        assert!((scaler.scale() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_scaler_update_backoff() {
        let mut scaler = GradScaler::new(100.0);
        scaler.found_inf = true;
        scaler.update();
        assert!((scaler.scale() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_scaler_double_unscale_error() {
        let mut scaler = GradScaler::new(100.0);
        let mut params = vec![];
        scaler.unscale_gradients(&mut params).unwrap();
        // Second unscale should error
        assert!(scaler.unscale_gradients(&mut params).is_err());
    }

    #[test]
    fn test_autocast_new() {
        let ac = Autocast::new();
        assert!(ac.is_enabled());
        assert_eq!(ac.low_precision_dtype(), TensorDtype::Float16);
    }

    #[test]
    fn test_autocast_classify() {
        assert_eq!(
            Autocast::classify_op("matmul"),
            PrecisionCategory::LowPrecision
        );
        assert_eq!(
            Autocast::classify_op("softmax"),
            PrecisionCategory::FullPrecision
        );
        assert_eq!(
            Autocast::classify_op("relu"),
            PrecisionCategory::PassThrough
        );
    }

    #[test]
    fn test_autocast_resolve_dtype() {
        let ac = Autocast::new();
        assert_eq!(
            ac.resolve_dtype(PrecisionCategory::LowPrecision, TensorDtype::Float32),
            TensorDtype::Float16
        );
        assert_eq!(
            ac.resolve_dtype(PrecisionCategory::FullPrecision, TensorDtype::Float16),
            TensorDtype::Float32
        );
        assert_eq!(
            ac.resolve_dtype(PrecisionCategory::PassThrough, TensorDtype::Float64),
            TensorDtype::Float64
        );
    }

    #[test]
    fn test_autocast_disabled() {
        let mut ac = Autocast::new();
        ac.set_enabled(false);
        assert_eq!(
            ac.resolve_dtype(PrecisionCategory::LowPrecision, TensorDtype::Float32),
            TensorDtype::Float32
        );
    }

    #[test]
    fn test_autocast_set_low_precision() {
        let mut ac = Autocast::new();
        ac.set_low_precision(TensorDtype::BFloat16).unwrap();
        assert_eq!(ac.low_precision_dtype(), TensorDtype::BFloat16);
        assert!(ac.set_low_precision(TensorDtype::Float32).is_err());
    }

    #[test]
    fn test_autocast_guard() {
        assert!(current_autocast().is_none());
        {
            let _guard = enter_autocast(Autocast::new());
            let ac = current_autocast();
            assert!(ac.is_some());
            assert!(ac.unwrap().is_enabled());
        }
        assert!(current_autocast().is_none());
    }

    #[test]
    fn test_cast_tensor() {
        let t = GpuTensor::from_host_f64(&[1.0, 2.0], &[2], 0).unwrap();
        let casted = Autocast::cast_tensor(&t, TensorDtype::Float16).unwrap();
        assert_eq!(casted.dtype(), TensorDtype::Float16);
        assert!((casted.host_data()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cast_tensor_same_dtype() {
        let t = GpuTensor::from_host_f64(&[1.0], &[1], 0).unwrap();
        let casted = Autocast::cast_tensor(&t, TensorDtype::Float64).unwrap();
        assert_eq!(casted.dtype(), TensorDtype::Float64);
    }

    #[test]
    fn test_grad_scaler_default() {
        let scaler = GradScaler::default();
        assert!((scaler.scale() - 65536.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_scaler_step_with_inf() {
        let mut scaler = GradScaler::new(1.0);
        let mut param = GpuTensor::from_host_f64(&[1.0], &[1], 0).unwrap();
        param.set_requires_grad(true);
        let grad = GpuTensor::from_host_f64(&[f64::NAN], &[1], 0).unwrap();
        param.accumulate_grad(&grad).unwrap();

        let mut params = vec![param];
        let mut opt = crate::tensor_backend::optimizer::Sgd::new(0.1);
        let stepped = scaler.step(&mut opt, &mut params).unwrap();
        assert!(!stepped); // Should skip
    }
}
