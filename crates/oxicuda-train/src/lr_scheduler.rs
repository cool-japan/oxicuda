//! Learning rate schedulers.
//!
//! All schedulers implement the [`crate::lr_scheduler::LrScheduler`] trait.  They are stepped once
//! per epoch (or per optimiser step, depending on the scheduler) and return the
//! current learning rate to be applied via [`crate::gpu_optimizer::GpuOptimizer::set_lr`].
//!
//! ## Available schedulers
//!
//! | Type | Description |
//! |---|---|
//! | [`crate::lr_scheduler::ConstantLR`] | Fixed learning rate |
//! | [`crate::lr_scheduler::StepLR`] | Multiply by γ every `step_size` epochs |
//! | [`crate::lr_scheduler::MultiStepLR`] | Multiply by γ at specified milestones |
//! | [`crate::lr_scheduler::ExponentialLR`] | Multiply by γ every epoch |
//! | [`crate::lr_scheduler::CosineAnnealingLR`] | Cosine decay to `eta_min` over T_max epochs |
//! | [`crate::lr_scheduler::LinearWarmup`] | Linear warm-up from 0 to `base_lr` over `warmup_steps` |
//! | [`crate::lr_scheduler::WarmupCosine`] | Linear warm-up → cosine annealing |
//! | [`crate::lr_scheduler::PolynomialDecayLR`] | Polynomial decay from `base_lr` to `end_lr` |
//! | [`crate::lr_scheduler::OneCycleLR`] | 1cycle policy (warmup + annealing in one schedule) |
//! | [`crate::lr_scheduler::CyclicLR`] | Triangular or exp-range cyclical schedule |
//! | [`crate::lr_scheduler::ReduceLROnPlateau`] | Reduce LR when a metric stops improving |

use crate::error::TrainResult;

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Common interface for all learning rate schedulers.
pub trait LrScheduler {
    /// Advance the scheduler by one step and return the **new** learning rate.
    fn step(&mut self) -> f64;

    /// Current learning rate (last value returned by `step`, or initial LR).
    fn get_lr(&self) -> f64;

    /// Scheduler name for display.
    fn name(&self) -> &str;

    /// Current step count.
    fn steps_done(&self) -> u64;
}

// ─── ConstantLR ──────────────────────────────────────────────────────────────

/// A no-op scheduler that always returns the initial learning rate.
#[derive(Debug, Clone)]
pub struct ConstantLR {
    lr: f64,
    steps: u64,
}

impl ConstantLR {
    /// Create a constant-LR scheduler.
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self { lr, steps: 0 }
    }
}

impl LrScheduler for ConstantLR {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        self.lr
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn name(&self) -> &str {
        "ConstantLR"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── StepLR ──────────────────────────────────────────────────────────────────

/// Multiply the learning rate by `gamma` every `step_size` epochs.
#[derive(Debug, Clone)]
pub struct StepLR {
    base_lr: f64,
    current_lr: f64,
    gamma: f64,
    step_size: u64,
    steps: u64,
}

impl StepLR {
    /// Create a step-decay scheduler.
    ///
    /// * `base_lr` – initial LR
    /// * `step_size` – decay every `step_size` steps
    /// * `gamma` – multiplicative decay factor (e.g. 0.1)
    #[must_use]
    pub fn new(base_lr: f64, step_size: u64, gamma: f64) -> Self {
        Self {
            base_lr,
            current_lr: base_lr,
            gamma,
            step_size,
            steps: 0,
        }
    }

    /// Initial (base) learning rate passed at construction time.
    #[must_use]
    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }
}

impl LrScheduler for StepLR {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        if self.steps % self.step_size == 0 {
            self.current_lr *= self.gamma;
        }
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "StepLR"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── MultiStepLR ─────────────────────────────────────────────────────────────

/// Multiply the LR by `gamma` at each milestone step.
#[derive(Debug, Clone)]
pub struct MultiStepLR {
    base_lr: f64,
    current_lr: f64,
    gamma: f64,
    milestones: Vec<u64>,
    steps: u64,
}

impl MultiStepLR {
    /// Create a multi-step scheduler.
    ///
    /// `milestones` must be a sorted list of step indices at which to decay.
    #[must_use]
    pub fn new(base_lr: f64, mut milestones: Vec<u64>, gamma: f64) -> Self {
        milestones.sort_unstable();
        Self {
            base_lr,
            current_lr: base_lr,
            gamma,
            milestones,
            steps: 0,
        }
    }

    /// Initial (base) learning rate passed at construction time.
    #[must_use]
    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }
}

impl LrScheduler for MultiStepLR {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        if self.milestones.contains(&self.steps) {
            self.current_lr *= self.gamma;
        }
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "MultiStepLR"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── ExponentialLR ───────────────────────────────────────────────────────────

/// Multiply the LR by `gamma` every epoch.
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    base_lr: f64,
    current_lr: f64,
    gamma: f64,
    steps: u64,
}

impl ExponentialLR {
    /// Create an exponential decay scheduler.
    #[must_use]
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        Self {
            base_lr,
            current_lr: base_lr,
            gamma,
            steps: 0,
        }
    }

    /// Initial (base) learning rate passed at construction time.
    #[must_use]
    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }
}

impl LrScheduler for ExponentialLR {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        self.current_lr *= self.gamma;
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "ExponentialLR"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── CosineAnnealingLR ───────────────────────────────────────────────────────

/// Cosine annealing: LR decays from `eta_max` to `eta_min` over `T_max` steps,
/// then resets.
///
/// `lr(t) = eta_min + 0.5·(eta_max − eta_min)·(1 + cos(π·t / T_max))`
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    eta_max: f64,
    eta_min: f64,
    t_max: u64,
    current_lr: f64,
    steps: u64,
}

impl CosineAnnealingLR {
    /// Create a cosine annealing scheduler.
    ///
    /// * `eta_max` – peak LR (also the initial LR)
    /// * `eta_min` – minimum LR at the trough (default 0)
    /// * `t_max`   – half-cycle length in steps
    #[must_use]
    pub fn new(eta_max: f64, t_max: u64) -> Self {
        Self {
            eta_max,
            eta_min: 0.0,
            t_max,
            current_lr: eta_max,
            steps: 0,
        }
    }

    /// Set the minimum LR.
    #[must_use]
    pub fn with_eta_min(mut self, eta_min: f64) -> Self {
        self.eta_min = eta_min;
        self
    }
}

impl LrScheduler for CosineAnnealingLR {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        let t = (self.steps % (self.t_max + 1)) as f64;
        let cos_val = (std::f64::consts::PI * t / self.t_max as f64).cos();
        self.current_lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1.0 + cos_val);
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "CosineAnnealingLR"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── LinearWarmup ────────────────────────────────────────────────────────────

/// Linear warm-up from 0 to `base_lr` over `warmup_steps` steps, then holds.
#[derive(Debug, Clone)]
pub struct LinearWarmup {
    base_lr: f64,
    warmup_steps: u64,
    current_lr: f64,
    steps: u64,
}

impl LinearWarmup {
    /// Create a linear warm-up scheduler.
    #[must_use]
    pub fn new(base_lr: f64, warmup_steps: u64) -> Self {
        Self {
            base_lr,
            warmup_steps,
            current_lr: 0.0,
            steps: 0,
        }
    }
}

impl LrScheduler for LinearWarmup {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        self.current_lr = if self.steps < self.warmup_steps {
            self.base_lr * self.steps as f64 / self.warmup_steps as f64
        } else {
            self.base_lr
        };
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "LinearWarmup"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── WarmupCosine ────────────────────────────────────────────────────────────

/// Linear warm-up followed by cosine annealing.
///
/// Phase 1: linear from 0 → `base_lr` over `warmup_steps`
/// Phase 2: cosine decay from `base_lr` → `eta_min` over `total_steps − warmup_steps`
#[derive(Debug, Clone)]
pub struct WarmupCosine {
    base_lr: f64,
    eta_min: f64,
    warmup_steps: u64,
    total_steps: u64,
    current_lr: f64,
    steps: u64,
}

impl WarmupCosine {
    /// Create a warmup + cosine annealing scheduler.
    #[must_use]
    pub fn new(base_lr: f64, warmup_steps: u64, total_steps: u64) -> Self {
        Self {
            base_lr,
            eta_min: 0.0,
            warmup_steps,
            total_steps,
            current_lr: 0.0,
            steps: 0,
        }
    }

    /// Set the minimum LR for the cosine phase.
    #[must_use]
    pub fn with_eta_min(mut self, eta_min: f64) -> Self {
        self.eta_min = eta_min;
        self
    }
}

impl LrScheduler for WarmupCosine {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        self.current_lr = if self.steps <= self.warmup_steps {
            self.base_lr * self.steps as f64 / self.warmup_steps.max(1) as f64
        } else {
            let t = (self.steps - self.warmup_steps) as f64;
            let t_max = (self.total_steps - self.warmup_steps).max(1) as f64;
            let cos_val = (std::f64::consts::PI * t / t_max).cos();
            self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + cos_val)
        };
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "WarmupCosine"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── PolynomialDecayLR ───────────────────────────────────────────────────────

/// Polynomial decay from `base_lr` to `end_lr` over `total_steps`.
///
/// `lr(t) = (base_lr − end_lr) · (1 − t/total_steps)^power + end_lr`
#[derive(Debug, Clone)]
pub struct PolynomialDecayLR {
    base_lr: f64,
    end_lr: f64,
    total_steps: u64,
    power: f64,
    current_lr: f64,
    steps: u64,
}

impl PolynomialDecayLR {
    /// Create a polynomial decay scheduler.
    ///
    /// * `power = 1.0` gives linear decay, `power = 2.0` gives quadratic.
    #[must_use]
    pub fn new(base_lr: f64, end_lr: f64, total_steps: u64, power: f64) -> Self {
        Self {
            base_lr,
            end_lr,
            total_steps,
            power,
            current_lr: base_lr,
            steps: 0,
        }
    }
}

impl LrScheduler for PolynomialDecayLR {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        let progress = (self.steps.min(self.total_steps) as f64) / (self.total_steps as f64);
        let factor = (1.0 - progress).powf(self.power);
        self.current_lr = (self.base_lr - self.end_lr) * factor + self.end_lr;
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "PolynomialDecayLR"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── OneCycleLR ──────────────────────────────────────────────────────────────

/// 1cycle policy (Smith & Touvron).
///
/// Phase 1: LR increases from `base_lr` to `max_lr` (linear or cosine).
/// Phase 2: LR decreases from `max_lr` to `min_lr` (cosine).
///
/// The warmup phase spans `pct_start` fraction of `total_steps`.
#[derive(Debug, Clone)]
pub struct OneCycleLR {
    base_lr: f64,
    max_lr: f64,
    min_lr: f64,
    total_steps: u64,
    /// Fraction of total steps for the warmup phase.
    pct_start: f64,
    current_lr: f64,
    steps: u64,
}

impl OneCycleLR {
    /// Create a 1cycle scheduler.
    ///
    /// * `max_lr`       – peak learning rate
    /// * `total_steps`  – total training steps
    /// * `pct_start`    – warmup fraction (default 0.3 = 30% of steps)
    #[must_use]
    pub fn new(max_lr: f64, total_steps: u64) -> Self {
        let base_lr = max_lr / 25.0;
        Self {
            base_lr,
            max_lr,
            min_lr: base_lr / 1e4,
            total_steps,
            pct_start: 0.3,
            current_lr: base_lr,
            steps: 0,
        }
    }

    /// Set the warmup phase fraction.
    #[must_use]
    pub fn with_pct_start(mut self, pct: f64) -> Self {
        self.pct_start = pct.clamp(0.0, 1.0);
        self
    }

    /// Set the base (initial) LR.
    #[must_use]
    pub fn with_base_lr(mut self, lr: f64) -> Self {
        self.base_lr = lr;
        self.current_lr = lr;
        self
    }
}

impl LrScheduler for OneCycleLR {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        let t = self.steps.min(self.total_steps) as f64;
        let total = self.total_steps as f64;
        let warmup = self.pct_start * total;

        self.current_lr = if t <= warmup {
            // Cosine warmup: base → max
            let prog = t / warmup;
            let cos_val = (std::f64::consts::PI * (1.0 - prog)).cos();
            self.base_lr + 0.5 * (self.max_lr - self.base_lr) * (1.0 + cos_val)
        } else {
            // Cosine annealing: max → min
            let prog = (t - warmup) / (total - warmup);
            let cos_val = (std::f64::consts::PI * prog).cos();
            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + cos_val)
        };
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "OneCycleLR"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── CyclicLR ────────────────────────────────────────────────────────────────

/// Cyclical LR (Smith 2017): oscillates between `base_lr` and `max_lr`.
#[derive(Debug, Clone)]
pub struct CyclicLR {
    base_lr: f64,
    max_lr: f64,
    step_size: u64,
    current_lr: f64,
    steps: u64,
}

impl CyclicLR {
    /// Create a triangular cyclic LR.
    ///
    /// One full cycle spans `2 * step_size` steps.
    #[must_use]
    pub fn new(base_lr: f64, max_lr: f64, step_size: u64) -> Self {
        Self {
            base_lr,
            max_lr,
            step_size,
            current_lr: base_lr,
            steps: 0,
        }
    }
}

impl LrScheduler for CyclicLR {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        let cycle_len = 2 * self.step_size;
        let pos = (self.steps - 1) % cycle_len;
        let t = if pos < self.step_size {
            pos as f64 / self.step_size as f64
        } else {
            1.0 - (pos - self.step_size) as f64 / self.step_size as f64
        };
        self.current_lr = self.base_lr + (self.max_lr - self.base_lr) * t;
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "CyclicLR"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── ReduceLROnPlateau ────────────────────────────────────────────────────────

/// Reduce LR when a monitored metric stops improving.
///
/// After `patience` steps without improvement, LR is multiplied by `factor`.
/// The metric is considered improved if it changes by more than `threshold`.
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    current_lr: f64,
    factor: f64,
    patience: u64,
    threshold: f64,
    min_lr: f64,
    best: f64,
    /// Steps since last improvement.
    bad_steps: u64,
    /// Whether to minimise (default) or maximise the metric.
    minimize: bool,
    steps: u64,
}

impl ReduceLROnPlateau {
    /// Create a reduce-on-plateau scheduler.
    ///
    /// * `lr`      – initial LR
    /// * `factor`  – LR reduction factor (default 0.1)
    /// * `patience`– number of steps without improvement before reducing
    #[must_use]
    pub fn new(lr: f64, factor: f64, patience: u64) -> Self {
        Self {
            current_lr: lr,
            factor,
            patience,
            threshold: 1e-4,
            min_lr: 0.0,
            best: f64::INFINITY,
            bad_steps: 0,
            minimize: true,
            steps: 0,
        }
    }

    /// Set the minimum allowable LR.
    #[must_use]
    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Set the improvement threshold.
    #[must_use]
    pub fn with_threshold(mut self, thr: f64) -> Self {
        self.threshold = thr;
        self
    }

    /// Switch to maximise mode (e.g., for accuracy metrics).
    #[must_use]
    pub fn maximize(mut self) -> Self {
        self.minimize = false;
        self.best = f64::NEG_INFINITY;
        self
    }

    /// Advance the scheduler based on the observed metric value.
    ///
    /// Returns the (possibly reduced) learning rate.
    pub fn step_metric(&mut self, metric: f64) -> TrainResult<f64> {
        self.steps += 1;
        let improved = if self.minimize {
            metric < self.best - self.threshold.abs()
        } else {
            metric > self.best + self.threshold.abs()
        };

        if improved {
            self.best = metric;
            self.bad_steps = 0;
        } else {
            self.bad_steps += 1;
        }

        if self.bad_steps >= self.patience {
            self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
            self.bad_steps = 0;
        }

        Ok(self.current_lr)
    }
}

impl LrScheduler for ReduceLROnPlateau {
    fn step(&mut self) -> f64 {
        // When called without metric, just return current LR unchanged
        self.steps += 1;
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "ReduceLROnPlateau"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn constant_lr_stays_constant() {
        let mut sched = ConstantLR::new(1e-3);
        for _ in 0..10 {
            let lr = sched.step();
            assert_abs_diff_eq!(lr, 1e-3, epsilon = 1e-10);
        }
    }

    #[test]
    fn step_lr_decays_at_boundary() {
        let mut sched = StepLR::new(1.0, 3, 0.1);
        // Steps 1,2,3: at step 3 decay occurs
        sched.step();
        sched.step();
        let before = sched.step(); // step 3 — decay applied
        assert_abs_diff_eq!(before, 0.1, epsilon = 1e-8);
        sched.step();
        sched.step();
        let after_6 = sched.step(); // step 6 — second decay
        assert_abs_diff_eq!(after_6, 0.01, epsilon = 1e-8);
    }

    #[test]
    fn multi_step_lr_decays_at_milestones() {
        let mut sched = MultiStepLR::new(1.0, vec![2, 4], 0.1);
        sched.step(); // step 1 → 1.0
        let at_2 = sched.step(); // step 2 → 0.1
        assert_abs_diff_eq!(at_2, 0.1, epsilon = 1e-8);
        sched.step(); // step 3 → 0.1
        let at_4 = sched.step(); // step 4 → 0.01
        assert_abs_diff_eq!(at_4, 0.01, epsilon = 1e-8);
    }

    #[test]
    fn exponential_lr_decays_every_step() {
        let mut sched = ExponentialLR::new(1.0, 0.5);
        let lr1 = sched.step();
        let lr2 = sched.step();
        let lr3 = sched.step();
        assert_abs_diff_eq!(lr1, 0.5, epsilon = 1e-8);
        assert_abs_diff_eq!(lr2, 0.25, epsilon = 1e-8);
        assert_abs_diff_eq!(lr3, 0.125, epsilon = 1e-8);
    }

    #[test]
    fn cosine_annealing_starts_at_max() {
        let mut sched = CosineAnnealingLR::new(1.0, 100);
        let lr1 = sched.step();
        // At t=1, lr should be slightly below eta_max
        assert!(lr1 < 1.0 && lr1 > 0.999, "got {lr1}");
    }

    #[test]
    fn cosine_annealing_reaches_min_at_t_max() {
        let mut sched = CosineAnnealingLR::new(1.0, 10).with_eta_min(0.0);
        for _ in 0..10 {
            sched.step();
        }
        let lr = sched.get_lr();
        assert!(lr < 1e-6, "should reach eta_min at T_max, got {lr}");
    }

    #[test]
    fn linear_warmup_starts_near_zero() {
        let mut sched = LinearWarmup::new(1e-3, 100);
        let lr1 = sched.step();
        // After step 1, lr = 1e-3 * 1/100 = 1e-5
        assert_abs_diff_eq!(lr1, 1e-5, epsilon = 1e-8);
    }

    #[test]
    fn linear_warmup_reaches_base_lr() {
        let mut sched = LinearWarmup::new(1e-3, 10);
        for _ in 0..10 {
            sched.step();
        }
        let lr = sched.get_lr();
        assert_abs_diff_eq!(lr, 1e-3, epsilon = 1e-8);
        // After warmup, subsequent steps keep base_lr
        let lr_after = sched.step();
        assert_abs_diff_eq!(lr_after, 1e-3, epsilon = 1e-8);
    }

    #[test]
    fn warmup_cosine_warmup_phase() {
        let mut sched = WarmupCosine::new(1e-3, 5, 20);
        // Step 1 in warmup phase (5 warmup steps)
        let lr1 = sched.step();
        assert!(lr1 < 1e-3, "should be below base_lr during warmup");
        assert!(lr1 > 0.0, "should be positive");
    }

    #[test]
    fn warmup_cosine_annealing_phase() {
        let mut sched = WarmupCosine::new(1.0, 5, 20);
        // Step through warmup
        for _ in 0..5 {
            sched.step();
        }
        let after_warmup = sched.get_lr();
        // Continue into annealing
        for _ in 0..10 {
            sched.step();
        }
        let annealing = sched.get_lr();
        assert!(
            annealing < after_warmup,
            "annealing phase should decrease LR"
        );
    }

    #[test]
    fn polynomial_decay_linear() {
        // Power=1, linear decay from 1.0 to 0.0 over 100 steps
        let mut sched = PolynomialDecayLR::new(1.0, 0.0, 100, 1.0);
        for _ in 0..50 {
            sched.step();
        }
        let lr = sched.get_lr();
        assert_abs_diff_eq!(lr, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn one_cycle_warmup_then_decay() {
        let total = 100_u64;
        let mut sched = OneCycleLR::new(1.0, total);
        let mut lrs: Vec<f64> = Vec::new();
        for _ in 0..total {
            lrs.push(sched.step());
        }
        let peak = lrs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (peak - 1.0).abs() < 0.1,
            "peak should be near max_lr=1.0, got {peak}"
        );
    }

    #[test]
    fn cyclic_lr_oscillates() {
        let mut sched = CyclicLR::new(1e-4, 1e-2, 10);
        let lr5 = {
            for _ in 0..5 {
                sched.step();
            }
            sched.get_lr()
        };
        let lr10 = {
            for _ in 0..5 {
                sched.step();
            }
            sched.get_lr()
        };
        let lr15 = {
            for _ in 0..5 {
                sched.step();
            }
            sched.get_lr()
        };
        // LR should rise to max at step 10, then fall back
        assert!(
            lr10 >= lr5,
            "should be rising at half-cycle, lr5={lr5}, lr10={lr10}"
        );
        assert!(
            lr10 > lr15,
            "should fall after peak, lr10={lr10}, lr15={lr15}"
        );
    }

    #[test]
    fn reduce_on_plateau_reduces_after_patience() {
        let mut sched = ReduceLROnPlateau::new(1.0, 0.1, 3);
        sched.step_metric(1.0).unwrap(); // best = 1.0
        sched.step_metric(1.0).unwrap(); // no improvement #1
        sched.step_metric(1.0).unwrap(); // no improvement #2
        let reduced = sched.step_metric(1.0).unwrap(); // no improvement #3 → reduce
        assert_abs_diff_eq!(reduced, 0.1, epsilon = 1e-8);
    }

    #[test]
    fn reduce_on_plateau_resets_on_improvement() {
        let mut sched = ReduceLROnPlateau::new(1.0, 0.1, 3);
        sched.step_metric(1.0).unwrap();
        sched.step_metric(0.5).unwrap(); // improvement: reset counter
        sched.step_metric(0.5).unwrap(); // no improvement #1
        sched.step_metric(0.5).unwrap(); // no improvement #2
        // Still 2 bad steps, not yet reduced
        assert_abs_diff_eq!(sched.get_lr(), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn all_schedulers_implement_trait() {
        let schedulers: Vec<Box<dyn LrScheduler>> = vec![
            Box::new(ConstantLR::new(1e-3)),
            Box::new(StepLR::new(1e-3, 10, 0.1)),
            Box::new(MultiStepLR::new(1e-3, vec![10, 20], 0.1)),
            Box::new(ExponentialLR::new(1e-3, 0.99)),
            Box::new(CosineAnnealingLR::new(1e-3, 100)),
            Box::new(LinearWarmup::new(1e-3, 10)),
            Box::new(WarmupCosine::new(1e-3, 10, 100)),
            Box::new(PolynomialDecayLR::new(1e-3, 0.0, 100, 1.0)),
            Box::new(OneCycleLR::new(1e-2, 100)),
            Box::new(CyclicLR::new(1e-4, 1e-2, 10)),
        ];

        for mut s in schedulers {
            let lr = s.step();
            assert!(lr >= 0.0, "LR should be non-negative from {}", s.name());
            assert!(s.steps_done() >= 1);
        }
    }
}
