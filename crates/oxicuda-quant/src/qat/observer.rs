//! # Quantization Observers
//!
//! Observers calibrate the quantization range by accumulating statistics
//! over a calibration dataset and then deriving scale / zero-point.
//!
//! | Observer           | Calibration strategy                       |
//! |--------------------|--------------------------------------------|
//! | `MinMaxObserver`   | Global min/max over all observed data      |
//! | `MovingAvgObserver`| Exponential moving average of min/max      |
//! | `HistogramObserver`| Histogram + min-MSE clipping range search  |

use crate::error::{QuantError, QuantResult};

// ─── Observer trait ───────────────────────────────────────────────────────────

/// Common interface for all quantization observers.
pub trait Observer {
    /// Observe a batch of values.
    fn observe(&mut self, data: &[f32]);

    /// Compute `(scale, zero_point)` from accumulated statistics.
    ///
    /// # Errors
    ///
    /// Returns [`QuantError::CalibrationRequired`] if no data has been observed.
    fn compute_params(&self) -> QuantResult<(f32, i32)>;

    /// Reset all accumulated statistics.
    fn reset(&mut self);

    /// Whether any data has been observed.
    fn is_calibrated(&self) -> bool;
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

fn sym_scale(abs_max: f32, bits: u32) -> f32 {
    let q_max = (1i32 << (bits - 1)) as f32 - 1.0;
    abs_max.max(1e-8) / q_max
}

fn asym_scale_zp(min_val: f32, max_val: f32, bits: u32) -> (f32, i32) {
    let q_range = ((1u32 << bits) - 1) as f32;
    let range = (max_val - min_val).max(1e-8);
    let scale = range / q_range;
    let zp = (-min_val / scale).round().clamp(0.0, q_range) as i32;
    (scale, zp)
}

// ─── MinMaxObserver ──────────────────────────────────────────────────────────

/// Tracks the global minimum and maximum of observed values.
///
/// **Symmetric** quantization: `scale = max(|min|, |max|) / q_max`, `zp = 0`.
/// **Asymmetric** quantization: `scale = (max − min) / (2^bits − 1)`.
#[derive(Debug, Clone)]
pub struct MinMaxObserver {
    /// Running minimum value.
    pub min_val: f32,
    /// Running maximum value.
    pub max_val: f32,
    /// Quantization bit-width.
    pub bits: u32,
    /// Symmetric (zero-point = 0) vs asymmetric.
    pub symmetric: bool,
}

impl MinMaxObserver {
    /// Create a new MinMaxObserver.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is 0 or > 16.
    #[must_use]
    pub fn new(bits: u32, symmetric: bool) -> Self {
        assert!(bits > 0 && bits <= 16, "bits must be in [1, 16]");
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            bits,
            symmetric,
        }
    }
}

impl Observer for MinMaxObserver {
    fn observe(&mut self, data: &[f32]) {
        for &v in data {
            if v.is_finite() {
                if v < self.min_val {
                    self.min_val = v;
                }
                if v > self.max_val {
                    self.max_val = v;
                }
            }
        }
    }

    fn compute_params(&self) -> QuantResult<(f32, i32)> {
        if !self.is_calibrated() {
            return Err(QuantError::CalibrationRequired("MinMaxObserver"));
        }
        if self.symmetric {
            let abs_max = self.min_val.abs().max(self.max_val.abs());
            Ok((sym_scale(abs_max, self.bits), 0))
        } else {
            Ok(asym_scale_zp(self.min_val, self.max_val, self.bits))
        }
    }

    fn reset(&mut self) {
        self.min_val = f32::INFINITY;
        self.max_val = f32::NEG_INFINITY;
    }

    fn is_calibrated(&self) -> bool {
        self.min_val.is_finite() && self.max_val.is_finite()
    }
}

// ─── MovingAvgObserver ───────────────────────────────────────────────────────

/// Tracks an exponential moving average of per-batch min/max statistics.
///
/// Update rule:
/// ```text
/// min_val ← momentum × min_val + (1 − momentum) × batch_min
/// max_val ← momentum × max_val + (1 − momentum) × batch_max
/// ```
#[derive(Debug, Clone)]
pub struct MovingAvgObserver {
    /// Running EMA minimum.
    pub min_val: f32,
    /// Running EMA maximum.
    pub max_val: f32,
    /// EMA momentum (fraction of old statistics to retain, typically 0.9–0.99).
    pub momentum: f32,
    /// Quantization bit-width.
    pub bits: u32,
    /// Symmetric vs asymmetric quantization.
    pub symmetric: bool,
    initialized: bool,
}

impl MovingAvgObserver {
    /// Create a new MovingAvgObserver.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is 0 or > 16 or if `momentum` is not in (0, 1).
    #[must_use]
    pub fn new(bits: u32, symmetric: bool, momentum: f32) -> Self {
        assert!(bits > 0 && bits <= 16, "bits must be in [1, 16]");
        assert!(
            momentum > 0.0 && momentum < 1.0,
            "momentum must be in (0, 1), got {momentum}"
        );
        Self {
            min_val: 0.0,
            max_val: 0.0,
            momentum,
            bits,
            symmetric,
            initialized: false,
        }
    }
}

impl Observer for MovingAvgObserver {
    fn observe(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }
        let batch_min = data
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::INFINITY, f32::min);
        let batch_max = data
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        if !batch_min.is_finite() || !batch_max.is_finite() {
            return;
        }
        if !self.initialized {
            self.min_val = batch_min;
            self.max_val = batch_max;
            self.initialized = true;
        } else {
            let m = self.momentum;
            self.min_val = m * self.min_val + (1.0 - m) * batch_min;
            self.max_val = m * self.max_val + (1.0 - m) * batch_max;
        }
    }

    fn compute_params(&self) -> QuantResult<(f32, i32)> {
        if !self.is_calibrated() {
            return Err(QuantError::CalibrationRequired("MovingAvgObserver"));
        }
        if self.symmetric {
            let abs_max = self.min_val.abs().max(self.max_val.abs());
            Ok((sym_scale(abs_max, self.bits), 0))
        } else {
            Ok(asym_scale_zp(self.min_val, self.max_val, self.bits))
        }
    }

    fn reset(&mut self) {
        self.min_val = 0.0;
        self.max_val = 0.0;
        self.initialized = false;
    }

    fn is_calibrated(&self) -> bool {
        self.initialized
    }
}

// ─── HistogramObserver ───────────────────────────────────────────────────────

/// Calibrates using a fixed-width histogram and min-MSE clipping search.
///
/// Accumulates a histogram over the absolute range of all observed data.
/// `compute_params` searches over percentile clipping thresholds and returns
/// the range that minimises estimated quantization MSE.
#[derive(Debug, Clone)]
pub struct HistogramObserver {
    /// Histogram bin counts.
    bins: Vec<u64>,
    /// Left edge of the histogram range.
    range_min: f32,
    /// Right edge of the histogram range.
    range_max: f32,
    /// Number of histogram bins.
    n_bins: usize,
    /// Quantization bit-width.
    pub bits: u32,
    /// Symmetric vs asymmetric.
    pub symmetric: bool,
    initialized: bool,
}

impl HistogramObserver {
    /// Create a new HistogramObserver with `n_bins` bins.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is 0 or > 16 or `n_bins` is 0.
    #[must_use]
    pub fn new(bits: u32, symmetric: bool, n_bins: usize) -> Self {
        assert!(bits > 0 && bits <= 16, "bits must be in [1, 16]");
        assert!(n_bins > 0, "n_bins must be > 0");
        Self {
            bins: vec![0_u64; n_bins],
            range_min: 0.0,
            range_max: 0.0,
            n_bins,
            bits,
            symmetric,
            initialized: false,
        }
    }

    /// Bin width of the current histogram.
    fn bin_width(&self) -> f32 {
        (self.range_max - self.range_min) / self.n_bins as f32
    }

    /// Estimate the quantization MSE for the clipping range `[lo, hi]`.
    fn estimate_mse(&self, lo: f32, hi: f32) -> f32 {
        let bw = self.bin_width();
        let total: u64 = self.bins.iter().sum();
        if total == 0 || (hi - lo).abs() < 1e-12 {
            return f32::INFINITY;
        }

        let n_levels = ((1u32 << self.bits) - 1) as f32;
        let step = (hi - lo) / n_levels;

        let mut mse = 0.0_f32;
        for (b, &cnt) in self.bins.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            let center = self.range_min + (b as f32 + 0.5) * bw;
            let quant_val = if center <= lo {
                lo
            } else if center >= hi {
                hi
            } else {
                let idx = ((center - lo) / step).round();
                lo + idx * step
            };
            let err = center - quant_val;
            mse += cnt as f32 * err * err;
        }
        mse / total as f32
    }
}

impl Observer for HistogramObserver {
    fn observe(&mut self, data: &[f32]) {
        let finite: Vec<f32> = data.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return;
        }

        let d_min = finite.iter().copied().fold(f32::INFINITY, f32::min);
        let d_max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if !self.initialized {
            self.range_min = d_min;
            self.range_max = d_max;
            self.initialized = true;
        } else {
            // Expand range if needed (histogram bins are not re-bucketed for simplicity).
            if d_min < self.range_min {
                self.range_min = d_min;
            }
            if d_max > self.range_max {
                self.range_max = d_max;
            }
        }

        // Ensure non-trivial range.
        if (self.range_max - self.range_min).abs() < 1e-8 {
            self.range_max = self.range_min + 1e-8;
        }

        let bw = self.bin_width();
        for &v in &finite {
            let idx = ((v - self.range_min) / bw) as usize;
            let idx = idx.min(self.n_bins - 1);
            self.bins[idx] += 1;
        }
    }

    fn compute_params(&self) -> QuantResult<(f32, i32)> {
        if !self.is_calibrated() {
            return Err(QuantError::CalibrationRequired("HistogramObserver"));
        }

        // Search over 20 percentile thresholds (0.5% to 100% of histogram range).
        let n_search = 20_usize;
        let mut best_mse = f32::INFINITY;
        let mut best_lo = self.range_min;
        let mut best_hi = self.range_max;

        let total: u64 = self.bins.iter().sum();
        if total == 0 {
            return Err(QuantError::CalibrationRequired("HistogramObserver"));
        }

        // Find quantile boundaries.
        let percentiles: Vec<f32> = (1..=n_search).map(|i| i as f32 / n_search as f32).collect();

        for &pct in &percentiles {
            let threshold = (pct * total as f32) as u64;
            let mut cum = 0_u64;
            let mut cut_bin = self.n_bins - 1;
            for (b, &cnt) in self.bins.iter().enumerate() {
                cum += cnt;
                if cum >= threshold {
                    cut_bin = b;
                    break;
                }
            }
            let bw = self.bin_width();
            let hi = self.range_min + (cut_bin as f32 + 1.0) * bw;
            let lo = if self.symmetric { -hi } else { self.range_min };

            let mse = self.estimate_mse(lo, hi);
            if mse < best_mse {
                best_mse = mse;
                best_lo = lo;
                best_hi = hi;
            }
        }

        if self.symmetric {
            let abs_max = best_lo.abs().max(best_hi.abs());
            Ok((sym_scale(abs_max, self.bits), 0))
        } else {
            Ok(asym_scale_zp(best_lo, best_hi, self.bits))
        }
    }

    fn reset(&mut self) {
        self.bins.fill(0);
        self.range_min = 0.0;
        self.range_max = 0.0;
        self.initialized = false;
    }

    fn is_calibrated(&self) -> bool {
        self.initialized
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn minmax_symmetric_scale() {
        let mut obs = MinMaxObserver::new(8, true);
        obs.observe(&[-2.0_f32, -1.0, 0.5, 2.0]);
        let (scale, zp) = obs.compute_params().unwrap();
        // abs_max = 2.0, q_max = 127 → scale = 2/127
        assert_abs_diff_eq!(scale, 2.0 / 127.0, epsilon = 1e-6);
        assert_eq!(zp, 0);
    }

    #[test]
    fn minmax_asymmetric_scale_zp() {
        let mut obs = MinMaxObserver::new(8, false);
        obs.observe(&[0.0_f32, 1.0, 2.0, 3.0]);
        let (scale, zp) = obs.compute_params().unwrap();
        assert_abs_diff_eq!(scale, 3.0 / 255.0, epsilon = 1e-5);
        assert_eq!(zp, 0);
    }

    #[test]
    fn minmax_calibration_required() {
        let obs = MinMaxObserver::new(8, true);
        assert!(matches!(
            obs.compute_params(),
            Err(QuantError::CalibrationRequired(_))
        ));
    }

    #[test]
    fn minmax_reset() {
        let mut obs = MinMaxObserver::new(8, true);
        obs.observe(&[1.0_f32, 2.0]);
        obs.reset();
        assert!(!obs.is_calibrated());
    }

    #[test]
    fn moving_avg_first_batch_exact() {
        let mut obs = MovingAvgObserver::new(8, true, 0.9);
        obs.observe(&[-1.0_f32, 1.0]);
        // First batch: min=-1, max=1, no averaging yet.
        let (scale, zp) = obs.compute_params().unwrap();
        assert_abs_diff_eq!(scale, 1.0 / 127.0, epsilon = 1e-5);
        assert_eq!(zp, 0);
    }

    #[test]
    fn moving_avg_ema_update() {
        let mut obs = MovingAvgObserver::new(8, true, 0.9);
        obs.observe(&[2.0_f32, 2.0]); // first: min=2, max=2
        obs.observe(&[4.0_f32, 4.0]); // second: EMA
        // max_val = 0.9*2 + 0.1*4 = 2.2
        assert_abs_diff_eq!(obs.max_val, 2.2, epsilon = 1e-5);
    }

    #[test]
    fn moving_avg_calibration_required() {
        let obs = MovingAvgObserver::new(8, true, 0.9);
        assert!(matches!(
            obs.compute_params(),
            Err(QuantError::CalibrationRequired(_))
        ));
    }

    #[test]
    fn histogram_observer_calibrates() {
        let mut obs = HistogramObserver::new(8, true, 256);
        let data: Vec<f32> = (0..1024).map(|i| (i as f32 / 512.0) - 1.0).collect();
        obs.observe(&data);
        assert!(obs.is_calibrated());
        let (scale, zp) = obs.compute_params().unwrap();
        assert!(scale > 0.0, "scale must be positive: {scale}");
        assert_eq!(zp, 0, "symmetric: zp must be 0");
    }

    #[test]
    fn histogram_observer_reset() {
        let mut obs = HistogramObserver::new(8, true, 128);
        obs.observe(&[1.0_f32, 2.0]);
        obs.reset();
        assert!(!obs.is_calibrated());
    }

    #[test]
    fn histogram_observer_uncalibrated_error() {
        let obs = HistogramObserver::new(8, true, 64);
        assert!(matches!(
            obs.compute_params(),
            Err(QuantError::CalibrationRequired(_))
        ));
    }
}
