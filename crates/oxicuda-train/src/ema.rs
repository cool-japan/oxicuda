//! # Exponential Moving Average (EMA) of Model Parameters
//!
//! [`crate::ema::ExponentialMovingAverage`] maintains *shadow parameters* that track a
//! smoothed version of the model parameters:
//!
//! ```text
//! shadow[i] = decay * shadow[i] + (1 - decay) * param[i]
//! ```
//!
//! EMA parameters are commonly used to produce better evaluation checkpoints
//! and are a core component of diffusion model training (e.g. DDPM, EDM2).
//!
//! ## Bias correction
//!
//! During the early stages of training (low step counts) the shadow parameters
//! are initialised to zero and thus biased toward zero.  The standard
//! *debiased* EMA applies a correction:
//!
//! ```text
//! effective_decay = min(decay, (1 + step) / (10 + step))
//! ```
//!
//! (This is the TensorFlow / Keras convention, also used by PyTorch's
//! `ExponentialMovingAverage`.)
//!
//! ## Usage
//!
//! ```rust
//! use oxicuda_train::ema::ExponentialMovingAverage;
//! use oxicuda_train::gpu_optimizer::ParamTensor;
//!
//! let mut params = vec![
//!     ParamTensor::new(vec![1.0_f32; 4], "w1"),
//!     ParamTensor::new(vec![2.0_f32; 2], "w2"),
//! ];
//!
//! let mut ema = ExponentialMovingAverage::new(0.999);
//!
//! // Training loop
//! for step in 0..100_u64 {
//!     // ... update params ...
//!     ema.update(&params).unwrap();
//!     let _ = step;
//! }
//!
//! // Copy EMA weights back for evaluation
//! let mut eval_params = params.clone();
//! ema.copy_to(&mut eval_params).unwrap();
//! ```

use crate::error::{TrainError, TrainResult};
use crate::gpu_optimizer::ParamTensor;

// ─── ExponentialMovingAverage ─────────────────────────────────────────────────

/// EMA decay modes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmaDecayMode {
    /// Fixed decay coefficient (no bias correction).
    Fixed,
    /// Bias-corrected decay: `min(decay, (1+step)/(10+step))`.
    ///
    /// Prevents the shadow parameters from being biased toward zero early in
    /// training.
    BiasCorrect,
}

/// Per-parameter decay override.
///
/// Allows different layers to use different decay rates (e.g. lower decay for
/// early layers that change more rapidly).
#[derive(Debug, Clone)]
pub struct LayerDecay {
    /// Parameter name (must match `ParamTensor::name`).
    pub name: String,
    /// Decay coefficient for this parameter.
    pub decay: f64,
}

/// Exponential Moving Average of model parameters.
///
/// Maintains a set of shadow buffers (`Vec<f32>`) that mirror the parameter
/// shapes and track their EMA across training steps.
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    /// Base decay coefficient.
    decay: f64,
    /// Shadow parameter buffers indexed by parameter position.
    shadows: Vec<Vec<f32>>,
    /// Whether shadow buffers have been initialised.
    initialised: bool,
    /// Current step count.
    step: u64,
    /// Bias-correction mode.
    mode: EmaDecayMode,
    /// Per-layer decay overrides (matched by name).
    layer_decays: Vec<LayerDecay>,
}

impl ExponentialMovingAverage {
    /// Create EMA with a fixed decay coefficient.
    ///
    /// Typical values: 0.999 (fast models), 0.9999 (slow / large models).
    #[must_use]
    pub fn new(decay: f64) -> Self {
        Self {
            decay,
            shadows: Vec::new(),
            initialised: false,
            step: 0,
            mode: EmaDecayMode::BiasCorrect,
            layer_decays: Vec::new(),
        }
    }

    /// Use fixed decay (no bias correction).
    #[must_use]
    pub fn with_fixed_decay(mut self) -> Self {
        self.mode = EmaDecayMode::Fixed;
        self
    }

    /// Register a per-layer decay override.
    #[must_use]
    pub fn with_layer_decay(mut self, name: impl Into<String>, decay: f64) -> Self {
        self.layer_decays.push(LayerDecay {
            name: name.into(),
            decay,
        });
        self
    }

    /// Current step count.
    #[must_use]
    #[inline]
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Base decay coefficient.
    #[must_use]
    #[inline]
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Effective decay for a given step (after bias correction if enabled).
    #[must_use]
    pub fn effective_decay(&self, step: u64, base: f64) -> f64 {
        match self.mode {
            EmaDecayMode::Fixed => base,
            EmaDecayMode::BiasCorrect => {
                let s = step as f64;
                let biased = (1.0 + s) / (10.0 + s);
                biased.min(base)
            }
        }
    }

    /// Return per-parameter effective decay for step `t`.
    fn param_decay(&self, param: &ParamTensor, step: u64) -> f64 {
        let base = self
            .layer_decays
            .iter()
            .find(|ld| ld.name == param.name)
            .map_or(self.decay, |ld| ld.decay);
        self.effective_decay(step, base)
    }

    /// Update shadow parameters from the current `params`.
    ///
    /// On the first call the shadows are initialised by cloning `params.data`.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ParamCountMismatch`] – if `params.len()` differs from
    ///   the number recorded at initialisation.
    /// * [`TrainError::ShapeMismatch`] – if any parameter changes shape.
    pub fn update(&mut self, params: &[ParamTensor]) -> TrainResult<()> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        if !self.initialised {
            // Initialise shadows as copies of current params.
            self.shadows = params.iter().map(|p| p.data.clone()).collect();
            self.initialised = true;
            self.step = 1;
            return Ok(());
        }
        if params.len() != self.shadows.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: self.shadows.len(),
                got: params.len(),
            });
        }
        self.step += 1;
        let step = self.step;
        // Pre-compute per-param decay values before mutably borrowing shadows.
        let decays: Vec<f32> = params
            .iter()
            .map(|p| self.param_decay(p, step) as f32)
            .collect();
        for ((shadow, param), &d) in self
            .shadows
            .iter_mut()
            .zip(params.iter())
            .zip(decays.iter())
        {
            if shadow.len() != param.data.len() {
                return Err(TrainError::ShapeMismatch {
                    expected: vec![shadow.len()],
                    got: vec![param.data.len()],
                });
            }
            let one_minus_d = 1.0_f32 - d;
            for (s, &p) in shadow.iter_mut().zip(param.data.iter()) {
                *s = d * *s + one_minus_d * p;
            }
        }
        Ok(())
    }

    /// Copy shadow parameters into `params.data` (for evaluation / checkpoint).
    ///
    /// # Errors
    ///
    /// * [`TrainError::StateNotInitialised`] – if `update` has never been
    ///   called.
    /// * [`TrainError::ParamCountMismatch`] / [`TrainError::ShapeMismatch`] –
    ///   shape mismatches.
    pub fn copy_to(&self, params: &mut [ParamTensor]) -> TrainResult<()> {
        if !self.initialised {
            return Err(TrainError::StateNotInitialised);
        }
        if params.len() != self.shadows.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: self.shadows.len(),
                got: params.len(),
            });
        }
        for (param, shadow) in params.iter_mut().zip(self.shadows.iter()) {
            if param.data.len() != shadow.len() {
                return Err(TrainError::ShapeMismatch {
                    expected: vec![shadow.len()],
                    got: vec![param.data.len()],
                });
            }
            param.data.copy_from_slice(shadow);
        }
        Ok(())
    }

    /// Restore `params.data` from the backup made before the last
    /// [`copy_to`][Self::copy_to].
    ///
    /// This is a convenience wrapper — it does the same as `copy_to` but reads
    /// from a separate set of `backup` values previously obtained with
    /// [`collect_shadows`][Self::collect_shadows].
    pub fn restore_from(params: &mut [ParamTensor], backup: &[Vec<f32>]) -> TrainResult<()> {
        if params.len() != backup.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: backup.len(),
                got: params.len(),
            });
        }
        for (param, bak) in params.iter_mut().zip(backup.iter()) {
            if param.data.len() != bak.len() {
                return Err(TrainError::ShapeMismatch {
                    expected: vec![bak.len()],
                    got: vec![param.data.len()],
                });
            }
            param.data.copy_from_slice(bak);
        }
        Ok(())
    }

    /// Return a copy of all current shadow parameter buffers.
    #[must_use]
    pub fn collect_shadows(&self) -> Vec<Vec<f32>> {
        self.shadows.clone()
    }

    /// Number of shadow parameters currently held.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.shadows.len()
    }

    /// Total shadow elements (sum of all buffer lengths).
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.shadows.iter().map(|s| s.len()).sum()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn params_from(vals: &[&[f32]]) -> Vec<ParamTensor> {
        vals.iter()
            .enumerate()
            .map(|(i, v)| ParamTensor::new(v.to_vec(), format!("p{i}")))
            .collect()
    }

    // ── basic update ─────────────────────────────────────────────────────────

    #[test]
    fn first_update_copies_params() {
        let mut ema = ExponentialMovingAverage::new(0.99);
        let params = params_from(&[&[1.0, 2.0, 3.0]]);
        ema.update(&params).unwrap();
        // After first update shadows == params exactly
        assert_eq!(ema.collect_shadows()[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn second_update_blends() {
        let mut ema = ExponentialMovingAverage::new(0.5).with_fixed_decay();
        let mut params = params_from(&[&[0.0]]);
        ema.update(&params).unwrap();
        // shadow = 0.0  (initial)
        params[0].data = vec![2.0];
        ema.update(&params).unwrap();
        // shadow = 0.5 * 0.0 + 0.5 * 2.0 = 1.0
        let s = ema.collect_shadows()[0][0];
        assert!((s - 1.0).abs() < 1e-6, "shadow={s}");
    }

    #[test]
    fn update_empty_params_error() {
        let mut ema = ExponentialMovingAverage::new(0.999);
        assert!(ema.update(&[]).is_err());
    }

    #[test]
    fn update_count_mismatch_error() {
        let mut ema = ExponentialMovingAverage::new(0.999);
        let p1 = params_from(&[&[1.0]]);
        ema.update(&p1).unwrap();
        let p2 = params_from(&[&[1.0], &[2.0]]);
        assert!(ema.update(&p2).is_err());
    }

    #[test]
    fn update_shape_mismatch_error() {
        let mut ema = ExponentialMovingAverage::new(0.999);
        let p1 = params_from(&[&[1.0, 2.0]]);
        ema.update(&p1).unwrap();
        let p2 = params_from(&[&[1.0, 2.0, 3.0]]);
        // shape differs: 2 → 3
        assert!(ema.update(&p2).is_err());
    }

    // ── copy_to ──────────────────────────────────────────────────────────────

    #[test]
    fn copy_to_overwrites_params() {
        let mut ema = ExponentialMovingAverage::new(0.5).with_fixed_decay();
        let mut params = params_from(&[&[0.0]]);
        ema.update(&params).unwrap(); // shadow = 0.0
        params[0].data = vec![4.0];
        ema.update(&params).unwrap(); // shadow = 0.5*0.0 + 0.5*4.0 = 2.0

        let mut eval_params = params.clone();
        ema.copy_to(&mut eval_params).unwrap();
        assert!((eval_params[0].data[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn copy_to_uninitialised_error() {
        let ema = ExponentialMovingAverage::new(0.999);
        let mut params = params_from(&[&[1.0]]);
        assert!(ema.copy_to(&mut params).is_err());
    }

    // ── restore_from ─────────────────────────────────────────────────────────

    #[test]
    fn restore_from_works() {
        let mut params = params_from(&[&[1.0, 2.0]]);
        let backup = vec![vec![5.0_f32, 6.0]];
        ExponentialMovingAverage::restore_from(&mut params, &backup).unwrap();
        assert_eq!(params[0].data, vec![5.0, 6.0]);
    }

    // ── bias correction ──────────────────────────────────────────────────────

    #[test]
    fn bias_corrected_decay_early_steps() {
        let ema = ExponentialMovingAverage::new(0.999);
        // At step 1: effective = min(0.999, 2/11) ≈ 0.1818 (far below base)
        let d = ema.effective_decay(1, 0.999);
        assert!(
            d < 0.999,
            "bias-corrected decay at step 1 should be < 0.999, got {d}"
        );
        // At step 9990: effective ≈ 0.999 (converged)
        let d2 = ema.effective_decay(9990, 0.999);
        assert!(
            d2 > 0.998,
            "decay should converge to base after many steps, got {d2}"
        );
    }

    #[test]
    fn fixed_decay_unchanged() {
        let ema = ExponentialMovingAverage::new(0.999).with_fixed_decay();
        let d = ema.effective_decay(1, 0.999);
        assert!(
            (d - 0.999).abs() < 1e-9,
            "fixed decay should equal base, got {d}"
        );
    }

    // ── per-layer decay ──────────────────────────────────────────────────────

    #[test]
    fn layer_decay_override() {
        let ema = ExponentialMovingAverage::new(0.999)
            .with_fixed_decay()
            .with_layer_decay("embed", 0.9);
        let embed = ParamTensor::new(vec![1.0], "embed");
        let other = ParamTensor::new(vec![1.0], "ffn");
        assert!((ema.param_decay(&embed, 10) - 0.9).abs() < 1e-9);
        assert!((ema.param_decay(&other, 10) - 0.999).abs() < 1e-9);
    }

    // ── metadata ─────────────────────────────────────────────────────────────

    #[test]
    fn total_elements_counts_all() {
        let mut ema = ExponentialMovingAverage::new(0.9);
        let params = params_from(&[&[1.0, 2.0], &[3.0, 4.0, 5.0]]);
        ema.update(&params).unwrap();
        assert_eq!(ema.total_elements(), 5);
    }

    #[test]
    fn num_params_matches() {
        let mut ema = ExponentialMovingAverage::new(0.9);
        let params = params_from(&[&[1.0], &[2.0], &[3.0]]);
        ema.update(&params).unwrap();
        assert_eq!(ema.num_params(), 3);
    }

    // ── convergence ──────────────────────────────────────────────────────────

    #[test]
    fn ema_tracks_constant_sequence() {
        // If params stay constant at 1.0, the shadow should converge to 1.0.
        let mut ema = ExponentialMovingAverage::new(0.9).with_fixed_decay();
        let params = params_from(&[&[1.0]]);
        for _ in 0..200 {
            ema.update(&params).unwrap();
        }
        let s = ema.collect_shadows()[0][0];
        assert!(
            (s - 1.0).abs() < 1e-4,
            "EMA should converge to 1.0, got {s}"
        );
    }

    #[test]
    fn ema_lags_behind_changing_params() {
        // Shadow should trail param as param ramps from 0 → 10.
        let mut ema = ExponentialMovingAverage::new(0.9).with_fixed_decay();
        let mut params = params_from(&[&[0.0]]);
        let mut shadow_vals = Vec::new();
        for i in 0..50_usize {
            params[0].data = vec![i as f32];
            ema.update(&params).unwrap();
            shadow_vals.push(ema.collect_shadows()[0][0]);
        }
        // Last shadow should be strictly less than 49.0 (lags behind)
        assert!(
            *shadow_vals.last().unwrap() < 49.0,
            "EMA should lag, last shadow={}",
            shadow_vals.last().unwrap()
        );
    }
}
