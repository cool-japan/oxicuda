//! Gradient accumulation for micro-batch training.
//!
//! Gradient accumulation allows training with an effective batch size larger
//! than what fits in GPU memory by accumulating gradients over `k` micro-batches
//! before calling the optimiser's `step`.
//!
//! ## Usage
//!
//! ```rust
//! use oxicuda_train::grad_accum::GradientAccumulator;
//! use oxicuda_train::gpu_optimizer::ParamTensor;
//!
//! let mut accum = GradientAccumulator::new(4); // accumulate 4 micro-batches
//! let mut params = vec![ParamTensor::new(vec![1.0f32; 8], "w")];
//!
//! // Simulate 4 micro-batch gradient passes
//! for _ in 0..4 {
//!     params[0].set_grad(vec![0.25f32; 8]).unwrap();
//!     accum.accumulate(&mut params).unwrap();
//! }
//!
//! // After `k` accumulations, ready_to_step() returns true
//! assert!(accum.ready_to_step());
//!
//! // Gradients are averaged, then call optimizer.step() once
//! accum.finalise(&mut params).unwrap();
//! // params[0].grad now holds the averaged gradient
//! ```

use crate::error::{TrainError, TrainResult};
use crate::gpu_optimizer::ParamTensor;

// ─── GradientAccumulator ─────────────────────────────────────────────────────

/// Accumulates gradients over multiple micro-batches before an optimizer step.
///
/// The accumulator collects per-parameter gradient contributions and exposes
/// the averaged gradient once `k` micro-batches have been processed.
#[derive(Debug, Clone)]
pub struct GradientAccumulator {
    /// Number of micro-batches to accumulate.
    k: usize,
    /// Current micro-batch index (0..k).
    current: usize,
    /// Running sum of gradients for each parameter (indexed by param position).
    accumulated: Vec<Vec<f32>>,
    /// Whether to average (true) or sum (false) accumulated gradients.
    average: bool,
}

impl GradientAccumulator {
    /// Create a gradient accumulator for `k` micro-batches.
    ///
    /// Gradients will be **averaged** before the optimizer step.
    #[must_use]
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "accumulation steps must be >= 1");
        Self {
            k,
            current: 0,
            accumulated: Vec::new(),
            average: true,
        }
    }

    /// Create a gradient accumulator that **sums** gradients (no averaging).
    #[must_use]
    pub fn new_sum(k: usize) -> Self {
        let mut a = Self::new(k);
        a.average = false;
        a
    }

    /// Number of micro-batches to accumulate.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Current micro-batch count (resets after each full cycle).
    #[must_use]
    pub fn current_step(&self) -> usize {
        self.current
    }

    /// Returns `true` when enough micro-batches have been accumulated and the
    /// optimizer `step` should be called.
    #[must_use]
    pub fn ready_to_step(&self) -> bool {
        self.current >= self.k
    }

    /// Accumulate the current gradients from `params` into the running sum.
    ///
    /// After calling this `k` times, call [`GradientAccumulator::finalise`] to write the averaged
    /// gradient back to `params` before calling the optimizer.
    ///
    /// # Errors
    ///
    /// * `EmptyParams` if `params` is empty.
    /// * `NoGradient` if a parameter has no gradient set.
    pub fn accumulate(&mut self, params: &[ParamTensor]) -> TrainResult<()> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }

        // Initialise running buffers on first call or after reset.
        if self.accumulated.len() != params.len() {
            self.accumulated = params.iter().map(|p| vec![0.0_f32; p.len()]).collect();
        }

        for (i, p) in params.iter().enumerate() {
            if !p.requires_grad {
                continue;
            }
            let grad = p.grad.as_ref().ok_or(TrainError::NoGradient { index: i })?;
            if grad.len() != self.accumulated[i].len() {
                return Err(TrainError::ShapeMismatch {
                    expected: vec![self.accumulated[i].len()],
                    got: vec![grad.len()],
                });
            }
            for (acc, &g) in self.accumulated[i].iter_mut().zip(grad.iter()) {
                *acc += g;
            }
        }

        self.current += 1;
        Ok(())
    }

    /// Write the averaged (or summed) accumulated gradient back to `params`.
    ///
    /// Resets the accumulator for the next cycle.  Call this once
    /// `ready_to_step()` returns `true`, then immediately call the optimizer.
    ///
    /// # Errors
    ///
    /// Returns `TrainError::Internal` if called before `k` micro-batches have
    /// been accumulated (i.e., `ready_to_step()` is `false`).
    pub fn finalise(&mut self, params: &mut [ParamTensor]) -> TrainResult<()> {
        if !self.ready_to_step() {
            return Err(TrainError::Internal {
                msg: format!(
                    "finalise called before accumulation complete ({}/{} steps)",
                    self.current, self.k
                ),
            });
        }
        let scale = if self.average {
            1.0_f32 / self.k as f32
        } else {
            1.0_f32
        };

        for (i, p) in params.iter_mut().enumerate() {
            if !p.requires_grad {
                continue;
            }
            if i < self.accumulated.len() {
                let avg: Vec<f32> = self.accumulated[i].iter().map(|&v| v * scale).collect();
                p.grad = Some(avg);
            }
        }

        // Reset for next cycle
        self.reset();
        Ok(())
    }

    /// Reset the accumulator without writing gradients (used after skipping
    /// a step due to gradient overflow, etc.).
    pub fn reset(&mut self) {
        self.current = 0;
        for acc in self.accumulated.iter_mut() {
            acc.iter_mut().for_each(|v| *v = 0.0_f32);
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn param_with_grad(g: f32) -> ParamTensor {
        let mut p = ParamTensor::new(vec![1.0_f32; 4], "w");
        p.set_grad(vec![g; 4]).unwrap();
        p
    }

    #[test]
    fn accumulate_and_finalise_averages() {
        let k = 4_usize;
        let mut accum = GradientAccumulator::new(k);
        let mut params = vec![ParamTensor::new(vec![1.0_f32; 4], "w")];

        for step_val in [1.0_f32, 2.0, 3.0, 4.0] {
            params[0].set_grad(vec![step_val; 4]).unwrap();
            accum.accumulate(&params).unwrap();
        }

        assert!(accum.ready_to_step());
        accum.finalise(&mut params).unwrap();

        // Average of 1+2+3+4 = 10 / 4 = 2.5
        let g = params[0].grad.as_ref().unwrap();
        assert_abs_diff_eq!(g[0], 2.5_f32, epsilon = 1e-5);
    }

    #[test]
    fn accumulate_sum_mode() {
        let k = 3_usize;
        let mut accum = GradientAccumulator::new_sum(k);
        let mut params = vec![ParamTensor::new(vec![0.0_f32; 2], "w")];

        for _ in 0..k {
            params[0].set_grad(vec![1.0_f32; 2]).unwrap();
            accum.accumulate(&params).unwrap();
        }

        accum.finalise(&mut params).unwrap();
        let g = params[0].grad.as_ref().unwrap();
        // sum mode: 1+1+1 = 3
        assert_abs_diff_eq!(g[0], 3.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn finalise_before_ready_returns_error() {
        let mut accum = GradientAccumulator::new(4);
        let mut params = vec![param_with_grad(1.0)];
        accum.accumulate(&params).unwrap();
        // Only 1 step done, not ready
        let result = accum.finalise(&mut params);
        assert!(matches!(result, Err(TrainError::Internal { .. })));
    }

    #[test]
    fn reset_clears_state() {
        let k = 2_usize;
        let mut accum = GradientAccumulator::new(k);
        let params = vec![param_with_grad(5.0)];
        accum.accumulate(&params).unwrap();
        assert_eq!(accum.current_step(), 1);
        accum.reset();
        assert_eq!(accum.current_step(), 0);
        assert!(!accum.ready_to_step());
    }

    #[test]
    fn accumulate_multi_param() {
        let k = 2_usize;
        let mut accum = GradientAccumulator::new(k);
        let mut params = vec![
            ParamTensor::new(vec![0.0_f32; 2], "a"),
            ParamTensor::new(vec![0.0_f32; 2], "b"),
        ];

        for step in 0..k {
            let g_val = (step + 1) as f32;
            params[0].set_grad(vec![g_val; 2]).unwrap();
            params[1].set_grad(vec![g_val * 2.0; 2]).unwrap();
            accum.accumulate(&params).unwrap();
        }
        accum.finalise(&mut params).unwrap();

        // param a: (1+2)/2 = 1.5; param b: (2+4)/2 = 3.0
        let ga = &params[0].grad.as_ref().unwrap()[0];
        let gb = &params[1].grad.as_ref().unwrap()[0];
        assert_abs_diff_eq!(*ga, 1.5_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(*gb, 3.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn k_and_current_step() {
        let accum = GradientAccumulator::new(8);
        assert_eq!(accum.k(), 8);
        assert_eq!(accum.current_step(), 0);
    }
}
