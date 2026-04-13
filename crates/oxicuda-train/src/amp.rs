//! # Automatic Mixed Precision — GradScaler
//!
//! `amp.rs` provides [`crate::amp::GradScaler`], a dynamic loss-scaling manager for
//! mixed-precision (FP16 / BF16) training on GPU.
//!
//! ## Background
//!
//! FP16 training is susceptible to gradient underflow: small gradient magnitudes
//! can flush to zero when represented in 16-bit.  The standard remedy is to
//! multiply the loss by a large *scale factor* S before back-propagation so that
//! the scaled gradients remain in the representable FP16 range.  Before the
//! optimizer step the gradients are divided by S again (*unscaled*).  If any
//! gradient element is `inf` or `NaN` after unscaling the step is *skipped* and
//! S is decreased; if no overflow is detected for `growth_interval` consecutive
//! steps, S is multiplied by `growth_factor`.
//!
//! ```text
//! Forward in FP16/BF16
//!   → loss × scale_factor  (scaled loss)
//! Backward
//!   → gradients ÷ scale_factor  (unscale)
//!   → check for inf/NaN  (overflow detection)
//!   → if overflow: skip step, scale /= backoff_factor
//!   → else:        optimizer.step(); if growth_interval hit: scale *= growth_factor
//! ```
//!
//! ## Quick start
//!
//! ```rust
//! use oxicuda_train::amp::{GradScaler, GradScalerConfig};
//! use oxicuda_train::gpu_optimizer::{GpuOptimizer, ParamTensor};
//! use oxicuda_train::gpu_optimizer::adamw::GpuAdamW;
//!
//! let mut scaler = GradScaler::default();
//! let mut opt    = GpuAdamW::new(1e-3);
//! let mut params = vec![ParamTensor::new(vec![1.0_f32; 4], "w")];
//!
//! // Simulate 10 training steps
//! for step in 0_u32..10 {
//!     // Fake scaled gradient (all finite)
//!     let scaled_grad: Vec<f32> = params[0].data.iter()
//!         .map(|&p| scaler.scale() as f32 * 2.0 * p)
//!         .collect();
//!     params[0].set_grad(scaled_grad).unwrap();
//!
//!     scaler.unscale(&mut params).unwrap();
//!     let did_step = scaler.step(&mut opt, &mut params).unwrap();
//!     scaler.update();
//!     let _ = (step, did_step);
//! }
//! ```

use crate::error::{TrainError, TrainResult};
use crate::gpu_optimizer::{GpuOptimizer, ParamTensor};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration knobs for [`GradScaler`].
#[derive(Debug, Clone)]
pub struct GradScalerConfig {
    /// Initial loss scale.  Default: 2¹⁶ = 65536.
    pub init_scale: f64,
    /// Multiplicative growth factor applied every `growth_interval` successful
    /// steps.  Default: 2.0.
    pub growth_factor: f64,
    /// Multiplicative backoff factor applied after an overflow is detected.
    /// Default: 0.5.
    pub backoff_factor: f64,
    /// Number of consecutive overflow-free steps before the scale is grown.
    /// Default: 2000.
    pub growth_interval: u32,
    /// Minimum allowed scale value.  If the scale would drop below this after a
    /// backoff the scaler is considered *overflowed beyond recovery* and returns
    /// [`TrainError::AmpMinScaleReached`].  Default: 1.0.
    pub min_scale: f64,
}

impl Default for GradScalerConfig {
    fn default() -> Self {
        Self {
            init_scale: 65536.0, // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            min_scale: 1.0,
        }
    }
}

// ─── GradScaler ───────────────────────────────────────────────────────────────

/// Dynamic loss-scale manager for mixed-precision training.
///
/// See module documentation for the full protocol.
#[derive(Debug, Clone)]
pub struct GradScaler {
    cfg: GradScalerConfig,
    /// Current scale factor.
    scale: f64,
    /// Consecutive steps without overflow.
    growth_tracker: u32,
    /// Number of optimizer steps that were actually taken (no overflow).
    steps_taken: u64,
    /// Number of optimizer steps skipped due to overflow.
    steps_skipped: u64,
    /// `true` after `unscale()` has been called for the current step, reset by
    /// `update()`.
    unscaled: bool,
    /// `true` if the most recent `unscale()` found inf/NaN.
    overflow: bool,
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new(GradScalerConfig::default())
    }
}

impl GradScaler {
    /// Create a new scaler with the given configuration.
    #[must_use]
    pub fn new(cfg: GradScalerConfig) -> Self {
        let scale = cfg.init_scale;
        Self {
            cfg,
            scale,
            growth_tracker: 0,
            steps_taken: 0,
            steps_skipped: 0,
            unscaled: false,
            overflow: false,
        }
    }

    /// Current scale factor.
    #[must_use]
    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Number of optimizer steps successfully taken.
    #[must_use]
    #[inline]
    pub fn steps_taken(&self) -> u64 {
        self.steps_taken
    }

    /// Number of steps skipped due to gradient overflow.
    #[must_use]
    #[inline]
    pub fn steps_skipped(&self) -> u64 {
        self.steps_skipped
    }

    /// Returns `true` if the last `unscale()` call detected overflow.
    #[must_use]
    #[inline]
    pub fn found_overflow(&self) -> bool {
        self.overflow
    }

    /// Scale a loss value by the current scale factor.
    ///
    /// Typically called as `let scaled_loss = scaler.scale_loss(raw_loss)`.
    #[must_use]
    #[inline]
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale as f32
    }

    /// Unscale gradients in-place by dividing by the current scale factor.
    ///
    /// Sets the internal overflow flag if any gradient element is `inf` or
    /// `NaN`.  Must be called **once per step** before [`step`][Self::step].
    ///
    /// # Errors
    ///
    /// * [`TrainError::InvalidState`] – if `unscale` is called twice without
    ///   an intervening `update`.
    pub fn unscale(&mut self, params: &mut [ParamTensor]) -> TrainResult<()> {
        if self.unscaled {
            return Err(TrainError::InvalidState(
                "unscale() called twice without update()".into(),
            ));
        }
        let inv_scale = (1.0 / self.scale) as f32;
        let mut found_inf = false;
        for p in params.iter_mut() {
            if let Some(g) = &mut p.grad {
                for v in g.iter_mut() {
                    *v *= inv_scale;
                    if !v.is_finite() {
                        found_inf = true;
                    }
                }
            }
        }
        self.overflow = found_inf;
        self.unscaled = true;
        Ok(())
    }

    /// Conditionally perform an optimizer step.
    ///
    /// * If overflow was detected during the preceding `unscale()`, the step is
    ///   **skipped** and `Ok(false)` is returned.
    /// * Otherwise `opt.step(params)` is called and `Ok(true)` is returned.
    ///
    /// # Errors
    ///
    /// * [`TrainError::InvalidState`] – if `unscale()` has not been called
    ///   first.
    /// * Propagates any error from the underlying optimizer `step`.
    pub fn step(
        &mut self,
        opt: &mut dyn GpuOptimizer,
        params: &mut [ParamTensor],
    ) -> TrainResult<bool> {
        if !self.unscaled {
            return Err(TrainError::InvalidState(
                "step() called before unscale()".into(),
            ));
        }
        if self.overflow {
            self.steps_skipped += 1;
            return Ok(false);
        }
        opt.step(params)?;
        self.steps_taken += 1;
        Ok(true)
    }

    /// Update the scale factor and reset internal per-step state.
    ///
    /// * On overflow: `scale *= backoff_factor`; `growth_tracker = 0`.
    /// * On success: `growth_tracker += 1`; if it reaches `growth_interval`,
    ///   `scale *= growth_factor` and `growth_tracker = 0`.
    ///
    /// # Errors
    ///
    /// * [`TrainError::AmpMinScaleReached`] – if `scale` drops below
    ///   `cfg.min_scale` after a backoff.
    pub fn update(&mut self) -> TrainResult<()> {
        if self.overflow {
            self.scale *= self.cfg.backoff_factor;
            self.growth_tracker = 0;
            if self.scale < self.cfg.min_scale {
                return Err(TrainError::AmpMinScaleReached(self.scale));
            }
        } else {
            self.growth_tracker += 1;
            if self.growth_tracker >= self.cfg.growth_interval {
                self.scale *= self.cfg.growth_factor;
                self.growth_tracker = 0;
            }
        }
        // Reset per-step flags.
        self.unscaled = false;
        self.overflow = false;
        Ok(())
    }

    /// Reset scaler to its initial state (useful for tests / restart).
    pub fn reset(&mut self) {
        self.scale = self.cfg.init_scale;
        self.growth_tracker = 0;
        self.steps_taken = 0;
        self.steps_skipped = 0;
        self.unscaled = false;
        self.overflow = false;
    }

    /// State summary for logging.
    #[must_use]
    pub fn state_summary(&self) -> AmpState {
        AmpState {
            scale: self.scale,
            growth_tracker: self.growth_tracker,
            steps_taken: self.steps_taken,
            steps_skipped: self.steps_skipped,
        }
    }
}

// ─── State snapshot ───────────────────────────────────────────────────────────

/// Read-only snapshot of GradScaler state for logging.
#[derive(Debug, Clone)]
pub struct AmpState {
    /// Current loss scale.
    pub scale: f64,
    /// Consecutive successful steps toward next growth event.
    pub growth_tracker: u32,
    /// Total optimizer steps taken.
    pub steps_taken: u64,
    /// Total optimizer steps skipped due to overflow.
    pub steps_skipped: u64,
}

// ─── Overflow detection utilities ────────────────────────────────────────────

/// Check whether any element of a flat slice is non-finite (`inf` or `NaN`).
#[must_use]
#[inline]
pub fn has_overflow(data: &[f32]) -> bool {
    data.iter().any(|&v| !v.is_finite())
}

/// PTX kernel source for checking `inf`/`NaN` in a flat float buffer.
///
/// Writes 1 to `result[0]` if any element is non-finite, leaves it at 0
/// otherwise.  The host should zero-init `result` before launch.
///
/// This is a GPU-side complement to the CPU-side `has_overflow` helper.
#[must_use]
pub fn overflow_check_ptx(sm: u32) -> String {
    let ver = if sm >= 80 { "8.0" } else { "7.5" };
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// overflow_check: flag any inf/NaN element
// .param .u64 ptr_data    (f32* input, n elements)
// .param .u64 ptr_result  (u32* output flag)
// .param .u32 n
.visible .entry overflow_check(
    .param .u64 ptr_data,
    .param .u64 ptr_result,
    .param .u32 n
)
{{
    .reg .u64  %addr, %res_addr;
    .reg .u32  %tid, %ntid, %ctaid, %nctaid, %idx, %n, %flag;
    .reg .f32  %val;
    .reg .pred %p_bounds, %p_inf;

    ld.param.u64  %addr,     [ptr_data];
    ld.param.u64  %res_addr, [ptr_result];
    ld.param.u32  %n,        [n];

    mov.u32 %tid,   %tid.x;
    mov.u32 %ntid,  %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mov.u32 %nctaid,%nctaid.x;

    mad.lo.u32 %idx, %ctaid, %ntid, %tid;

LOOP:
    setp.ge.u32 %p_bounds, %idx, %n;
    @%p_bounds bra DONE;

    // load element
    .reg .u64 %eaddr;
    mul.wide.u32 %eaddr, %idx, 4;
    add.u64  %eaddr, %addr, %eaddr;
    ld.global.f32 %val, [%eaddr];

    // abs(val) > val | val != val  →  inf or NaN
    testp.infinite.f32 %p_inf, %val;
    @%p_inf bra FOUND;
    testp.nan.f32      %p_inf, %val;
    @%p_inf bra FOUND;

    // advance grid-stride
    add.u32 %idx, %idx, %ntid;
    mul.wide.u32 %eaddr, %nctaid, %ntid;  // total_threads (reuse %eaddr)
    cvt.u32.u64 %flag, %eaddr;
    add.u32 %idx, %idx, %flag;
    sub.u32 %idx, %idx, %ntid;
    add.u32 %idx, %idx, %ntid;
    mov.u32 %flag, %ntid;
    mul.lo.u32 %flag, %nctaid, %flag;
    add.u32 %idx, %idx, %flag;
    sub.u32 %idx, %idx, %ntid;
    // --- simplified grid-stride: idx += gridDim.x * blockDim.x
    mov.u32 %flag, %ntid;
    mul.lo.u32 %flag, %nctaid, %flag;
    add.u32 %idx, %tid, 0;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    add.u32 %idx, %idx, %flag;
    // (reset to avoid accumulation bugs — recompute from scratch)
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    add.u32 %idx, %idx, %flag;
    sub.u32 %idx, %idx, %flag;
    add.u32 %idx, %idx, %flag;
    bra LOOP;

FOUND:
    mov.u32 %flag, 1;
    st.global.u32 [%res_addr], %flag;
DONE:
    ret;
}}
"#,
        ver = ver,
        sm = sm
    )
}

/// PTX kernel source for in-place gradient unscaling.
///
/// Multiplies every element of `data` by `inv_scale = 1 / scale_factor`.
#[must_use]
pub fn unscale_ptx(sm: u32) -> String {
    let ver = if sm >= 80 { "8.0" } else { "7.5" };
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// unscale: data[i] *= inv_scale  (in-place)
.visible .entry unscale_inplace(
    .param .u64 ptr_data,
    .param .u32 n,
    .param .f32 inv_scale
)
{{
    .reg .u64  %addr, %eaddr;
    .reg .u32  %tid, %ntid, %ctaid, %nctaid, %idx, %n, %stride;
    .reg .f32  %val, %inv;

    ld.param.u64  %addr, [ptr_data];
    ld.param.u32  %n,    [n];
    ld.param.f32  %inv,  [inv_scale];

    mov.u32 %tid,   %tid.x;
    mov.u32 %ntid,  %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mov.u32 %nctaid,%nctaid.x;

    mad.lo.u32 %idx,    %ctaid, %ntid, %tid;
    mul.lo.u32 %stride, %nctaid, %ntid;

LOOP:
    .reg .pred %p;
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    mul.wide.u32 %eaddr, %idx, 4;
    add.u64  %eaddr, %addr, %eaddr;
    ld.global.f32 %val, [%eaddr];
    mul.rn.f32    %val, %val, %inv;
    st.global.f32 [%eaddr], %val;

    add.u32 %idx, %idx, %stride;
    bra LOOP;
DONE:
    ret;
}}
"#,
        ver = ver,
        sm = sm
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_optimizer::adamw::GpuAdamW;

    fn make_params(vals: &[f32]) -> Vec<ParamTensor> {
        let mut p = ParamTensor::new(vals.to_vec(), "w");
        p.set_grad(vec![0.1_f32; vals.len()]).unwrap();
        vec![p]
    }

    // ── scale / unscale cycle ────────────────────────────────────────────────

    #[test]
    fn scale_loss() {
        let scaler = GradScaler::default();
        let loss = 1.0_f32;
        let scaled = scaler.scale_loss(loss);
        assert!((scaled - 65536.0_f32).abs() < 1.0, "scaled={scaled}");
    }

    #[test]
    fn unscale_divides_by_scale() {
        let mut scaler = GradScaler::new(GradScalerConfig {
            init_scale: 4.0,
            ..GradScalerConfig::default()
        });
        let mut params = make_params(&[1.0]);
        // set grad = 8.0 (= 4.0 * 2.0 simulating scale applied in backward)
        params[0].set_grad(vec![8.0_f32]).unwrap();
        scaler.unscale(&mut params).unwrap();
        // after unscale grad should be 8.0 / 4.0 = 2.0
        assert!((params[0].grad.as_ref().unwrap()[0] - 2.0_f32).abs() < 1e-6);
    }

    #[test]
    fn unscale_detects_inf() {
        let mut scaler = GradScaler::default();
        let mut params = make_params(&[1.0]);
        params[0].set_grad(vec![f32::INFINITY]).unwrap();
        scaler.unscale(&mut params).unwrap();
        assert!(scaler.found_overflow());
    }

    #[test]
    fn unscale_detects_nan() {
        let mut scaler = GradScaler::default();
        let mut params = make_params(&[1.0]);
        params[0].set_grad(vec![f32::NAN]).unwrap();
        scaler.unscale(&mut params).unwrap();
        assert!(scaler.found_overflow());
    }

    #[test]
    fn double_unscale_returns_error() {
        let mut scaler = GradScaler::default();
        let mut params = make_params(&[1.0]);
        scaler.unscale(&mut params).unwrap();
        let err = scaler.unscale(&mut params);
        assert!(err.is_err());
    }

    // ── overflow → skip step ─────────────────────────────────────────────────

    #[test]
    fn step_skipped_on_overflow() {
        let mut scaler = GradScaler::default();
        let mut opt = GpuAdamW::new(1e-3);
        let mut params = make_params(&[1.0]);
        // inject inf gradient
        params[0].set_grad(vec![f32::INFINITY]).unwrap();
        scaler.unscale(&mut params).unwrap();
        let did_step = scaler.step(&mut opt, &mut params).unwrap();
        assert!(!did_step, "should skip on overflow");
        assert_eq!(scaler.steps_skipped(), 1);
        assert_eq!(scaler.steps_taken(), 0);
    }

    #[test]
    fn step_taken_on_finite_gradients() {
        let mut scaler = GradScaler::new(GradScalerConfig {
            init_scale: 1.0,
            ..GradScalerConfig::default()
        });
        let mut opt = GpuAdamW::new(1e-3);
        let mut params = make_params(&[1.0]);
        params[0].set_grad(vec![0.5_f32]).unwrap();
        scaler.unscale(&mut params).unwrap();
        let did_step = scaler.step(&mut opt, &mut params).unwrap();
        assert!(did_step, "should take step on finite grads");
        assert_eq!(scaler.steps_taken(), 1);
    }

    #[test]
    fn step_without_unscale_returns_error() {
        let mut scaler = GradScaler::default();
        let mut opt = GpuAdamW::new(1e-3);
        let mut params = make_params(&[1.0]);
        let err = scaler.step(&mut opt, &mut params);
        assert!(err.is_err());
    }

    // ── update: scale adjustment ─────────────────────────────────────────────

    #[test]
    fn update_decreases_scale_on_overflow() {
        let mut scaler = GradScaler::new(GradScalerConfig {
            init_scale: 1024.0,
            backoff_factor: 0.5,
            ..GradScalerConfig::default()
        });
        let mut params = make_params(&[1.0]);
        params[0].set_grad(vec![f32::INFINITY]).unwrap();
        scaler.unscale(&mut params).unwrap();
        let _ = scaler.step(&mut GpuAdamW::new(1e-3), &mut params);
        scaler.update().unwrap();
        assert!(
            (scaler.scale() - 512.0).abs() < 1e-9,
            "scale={}",
            scaler.scale()
        );
    }

    #[test]
    fn update_grows_scale_after_interval() {
        let mut scaler = GradScaler::new(GradScalerConfig {
            init_scale: 1.0,
            growth_factor: 2.0,
            growth_interval: 3,
            ..GradScalerConfig::default()
        });
        let mut opt = GpuAdamW::new(1e-3);
        // 3 clean steps → scale doubles
        for _ in 0..3 {
            let mut params = make_params(&[1.0]);
            params[0].set_grad(vec![0.1_f32]).unwrap();
            scaler.unscale(&mut params).unwrap();
            scaler.step(&mut opt, &mut params).unwrap();
            scaler.update().unwrap();
        }
        assert!(
            (scaler.scale() - 2.0).abs() < 1e-9,
            "scale={}",
            scaler.scale()
        );
    }

    #[test]
    fn update_min_scale_error() {
        let mut scaler = GradScaler::new(GradScalerConfig {
            init_scale: 2.0,
            backoff_factor: 0.5,
            min_scale: 1.0,
            ..GradScalerConfig::default()
        });
        let mut params = make_params(&[1.0]);
        // Trigger two overflows: 2→1→error
        params[0].set_grad(vec![f32::INFINITY]).unwrap();
        scaler.unscale(&mut params).unwrap();
        let _ = scaler.step(&mut GpuAdamW::new(1e-3), &mut params);
        scaler.update().unwrap(); // scale: 2 → 1
        assert!((scaler.scale() - 1.0).abs() < 1e-9);

        let mut params2 = make_params(&[1.0]);
        params2[0].set_grad(vec![f32::INFINITY]).unwrap();
        scaler.unscale(&mut params2).unwrap();
        let _ = scaler.step(&mut GpuAdamW::new(1e-3), &mut params2);
        let err = scaler.update(); // scale would go to 0.5 < min_scale
        assert!(err.is_err(), "should error at min_scale");
    }

    // ── reset ────────────────────────────────────────────────────────────────

    #[test]
    fn reset_restores_initial_state() {
        let mut scaler = GradScaler::new(GradScalerConfig {
            init_scale: 128.0,
            ..GradScalerConfig::default()
        });
        let mut params = make_params(&[1.0]);
        params[0].set_grad(vec![f32::INFINITY]).unwrap();
        scaler.unscale(&mut params).unwrap();
        let _ = scaler.step(&mut GpuAdamW::new(1e-3), &mut params);
        scaler.update().unwrap();
        assert!(scaler.scale() < 128.0);

        scaler.reset();
        assert!((scaler.scale() - 128.0).abs() < 1e-9);
        assert_eq!(scaler.steps_taken(), 0);
        assert_eq!(scaler.steps_skipped(), 0);
    }

    // ── state_summary ────────────────────────────────────────────────────────

    #[test]
    fn state_summary_reflects_current_state() {
        let scaler = GradScaler::new(GradScalerConfig {
            init_scale: 1024.0,
            ..GradScalerConfig::default()
        });
        let s = scaler.state_summary();
        assert!((s.scale - 1024.0).abs() < 1e-9);
        assert_eq!(s.growth_tracker, 0);
        assert_eq!(s.steps_taken, 0);
        assert_eq!(s.steps_skipped, 0);
    }

    // ── has_overflow helper ──────────────────────────────────────────────────

    #[test]
    fn has_overflow_finite() {
        assert!(!has_overflow(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn has_overflow_inf() {
        assert!(has_overflow(&[1.0, f32::INFINITY, 3.0]));
    }

    #[test]
    fn has_overflow_nan() {
        assert!(has_overflow(&[f32::NAN]));
    }

    // ── PTX generation ───────────────────────────────────────────────────────

    #[test]
    fn overflow_check_ptx_sm80() {
        let ptx = overflow_check_ptx(80);
        assert!(ptx.contains("sm_80"));
        assert!(ptx.contains("overflow_check"));
    }

    #[test]
    fn unscale_ptx_sm75() {
        let ptx = unscale_ptx(75);
        assert!(ptx.contains("sm_75"));
        assert!(ptx.contains("unscale_inplace"));
    }

    // ── full loop (AMP simulation) ───────────────────────────────────────────

    #[test]
    fn amp_full_loop_converges() {
        let cfg = GradScalerConfig {
            init_scale: 4.0,
            growth_factor: 2.0,
            growth_interval: 5,
            backoff_factor: 0.5,
            min_scale: 0.25,
        };
        // Adam normalises the gradient to ~sign(g)*lr per step, so from x=4.0
        // with lr=5e-3 we need ~800 steps to reach |x| < 0.5.
        let mut scaler = GradScaler::new(cfg);
        let mut opt = GpuAdamW::new(5e-3).with_weight_decay(0.0).with_beta2(0.9);
        let mut params = vec![ParamTensor::new(vec![4.0_f32], "x")];

        for _ in 0..800 {
            let x = params[0].data[0];
            // scaled grad = scale * 2x
            let g = scaler.scale() as f32 * 2.0 * x;
            params[0].set_grad(vec![g]).unwrap();
            scaler.unscale(&mut params).unwrap();
            let _ = scaler.step(&mut opt, &mut params).unwrap();
            scaler.update().unwrap();
        }
        let x = params[0].data[0].abs();
        assert!(x < 0.5, "AMP loop should converge x→0, got |x|={x}");
        assert!(scaler.steps_taken() > 0);
    }
}
