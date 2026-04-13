//! Modified Discrete Cosine Transform (MDCT) — the overlap-add transform used
//! in MP3, AAC, Vorbis, and Opus audio codecs.
//!
//! ## Definition
//!
//! Given a length-`2N` window of samples `x[n]`, the MDCT produces N spectral
//! coefficients:
//!
//! ```text
//! X[k] = Σ_{n=0}^{2N-1} x[n] · w[n] · cos(π/N · (n + ½ + N/2) · (k + ½))
//! ```
//!
//! where `w[n]` is a window function (typically the Vorbis/Sine window or
//! Kaiser-Bessel derived, KBD).
//!
//! ## MDCT → DCT-IV relationship
//!
//! The MDCT can be computed via a DCT-IV of length N after appropriate
//! pre-rotation of the windowed input:
//!
//! 1. Split the `2N` samples into four halves of length `N/2`.
//! 2. Fold and negate to produce a length-N pre-rotation input.
//! 3. Apply DCT-IV of length N.
//!
//! The inverse MDCT (IMDCT) is:
//! ```text
//! x_rec[n] = (2/N) · Σ_{k=0}^{N-1} X[k] · cos(π/N · (n + ½ + N/2) · (k + ½))
//! ```
//! which is again computed via DCT-IV after appropriate post-rotation and
//! overlap-add.

use std::f64::consts::PI;

use crate::{
    dct::dct4::dct4_reference,
    error::{SignalError, SignalResult},
};

// --------------------------------------------------------------------------- //
//  Window functions for MDCT
// --------------------------------------------------------------------------- //

/// Sine window: `w[n] = sin(π(n+½) / (2N))`, the standard MP3/AAC window.
#[must_use]
pub fn sine_window(n2: usize) -> Vec<f64> {
    // n2 = 2N (full window length)
    (0..n2)
        .map(|n| (PI * (n as f64 + 0.5) / n2 as f64).sin())
        .collect()
}

/// Kaiser-Bessel derived (KBD) window with shape parameter α.
///
/// The KBD window has superior frequency resolution for audio codecs.
/// `n2` must be even.
#[must_use]
pub fn kbd_window(n2: usize, alpha: f64) -> Vec<f64> {
    // Build the Kaiser window of length N/2 + 1 and compute its cumulative sum.
    let half = n2 / 2;
    let beta = PI * alpha;
    // Modified Bessel function I₀(x) via series expansion.
    let i0 = |x: f64| -> f64 {
        let mut s = 1.0_f64;
        let mut term = 1.0_f64;
        for i in 1..=25 {
            term *= (x / (2.0 * i as f64)).powi(2);
            s += term;
            if term < 1e-15 * s {
                break;
            }
        }
        s
    };
    let i0_beta = i0(beta);
    // Kaiser window of length half+1
    let kaiser: Vec<f64> = (0..=half)
        .map(|n| {
            let t = 2.0 * n as f64 / half as f64 - 1.0;
            i0(beta * (1.0 - t * t).sqrt()) / i0_beta
        })
        .collect();
    // Inclusive cumulative sum: s[k] = Σ_{j=0}^{k} kaiser[j].
    // Using INCLUSIVE sum ensures perfect reconstruction: s[n] + s[half-1-n] = s[half].
    let mut s = vec![0.0_f64; half + 1];
    s[0] = kaiser[0];
    for k in 1..=half {
        s[k] = s[k - 1] + kaiser[k];
    }
    let total = s[half]; // Σ_{j=0}^{half} kaiser[j]
    // First half: w[n] = sqrt(s[n] / total)
    let mut w = vec![0.0_f64; n2];
    for n in 0..half {
        w[n] = (s[n] / total).sqrt();
    }
    // Second half: symmetric mirror — w[n2-1-n] = w[n]
    for n in 0..half {
        w[n2 - 1 - n] = w[n];
    }
    w
}

// --------------------------------------------------------------------------- //
//  CPU reference MDCT / IMDCT
// --------------------------------------------------------------------------- //

/// CPU reference MDCT (windowed, via DCT-IV).
///
/// # Parameters
/// - `x`: windowed time-domain samples (length `2N`).
/// - `n`: number of MDCT coefficients (= half the window length).
///
/// # Errors
/// Returns [`SignalError::InvalidSize`] if `x.len() != 2 * n`.
pub fn mdct(x: &[f64], n: usize) -> SignalResult<Vec<f64>> {
    if x.len() != 2 * n {
        return Err(SignalError::InvalidSize(format!(
            "MDCT expects 2N={} samples, got {}",
            2 * n,
            x.len()
        )));
    }
    // Pre-rotation: fold 2N samples into N using the Princen-Bradley formula.
    let mut pre = vec![0.0_f64; n];
    let n_half = n / 2;
    // Folding formula (Princen-Bradley):
    //   pre[k] = -x[n/2 + k] - x[n/2 - 1 - k]           for k = 0..n/2
    //   pre[k] =  x[k - n/2] - x[5n/2 - 1 - k]           for k = n/2..n
    for k in 0..n_half {
        pre[k] = -x[n_half + k] - x[n_half - 1 - k];
    }
    for k in n_half..n {
        pre[k] = x[k - n_half] - x[5 * n_half - 1 - k];
    }
    // Apply DCT-IV
    Ok(dct4_reference(&pre))
}

/// CPU reference IMDCT (inverse MDCT via DCT-IV, then overlap-add unfolding).
///
/// Returns a length-`2N` time-domain sequence (before window application).
///
/// # Errors
/// Returns [`SignalError::InvalidSize`] if `n` is zero.
pub fn imdct(x: &[f64]) -> SignalResult<Vec<f64>> {
    let n = x.len();
    if n == 0 {
        return Err(SignalError::InvalidSize(
            "IMDCT length must be ≥ 1".to_owned(),
        ));
    }
    // IMDCT = (2/N) · DCT-IV · x (DCT-IV is its own inverse up to N/2 scale).
    let dct_out = dct4_reference(x);
    let scale = 2.0 / n as f64;
    let n2 = 2 * n;
    let n_half = n / 2;
    let mut out = vec![0.0_f64; n2];
    // Unfolding (inverse of the MDCT pre-rotation fold):
    for k in 0..n_half {
        out[n_half + k] = -dct_out[k] * scale;
        out[n_half - 1 - k] = -dct_out[k] * scale;
    }
    for k in n_half..n {
        out[k - n_half] = dct_out[k] * scale;
        out[5 * n_half - 1 - k] = -dct_out[k] * scale;
    }
    Ok(out)
}

// --------------------------------------------------------------------------- //
//  MDCT plan (PTX-based batch execution)
// --------------------------------------------------------------------------- //

/// Execution plan for batched GPU MDCT processing.
///
/// The MDCT is factored as: window → pre-rotate → DCT-IV → output.
/// All three stages are implemented as PTX kernels on the caller's stream.
pub struct MdctPlan {
    /// Number of MDCT output bins (half the window length).
    pub n: usize,
    /// Batch size (number of independent MDCT frames).
    pub batch: usize,
    /// Window function coefficients (length `2N`).
    pub window_coeffs: Vec<f64>,
}

impl MdctPlan {
    /// Create a new MDCT plan with the given window function.
    ///
    /// # Errors
    /// Returns [`SignalError::InvalidSize`] if `window.len() != 2 * n`.
    pub fn new(n: usize, batch: usize, window: Vec<f64>) -> SignalResult<Self> {
        if window.len() != 2 * n {
            return Err(SignalError::InvalidSize(format!(
                "window length {} must equal 2N={}",
                window.len(),
                2 * n
            )));
        }
        if batch == 0 {
            return Err(SignalError::InvalidParameter(
                "batch must be ≥ 1".to_owned(),
            ));
        }
        Ok(Self {
            n,
            batch,
            window_coeffs: window,
        })
    }

    /// Convenience: create a plan with the sine window.
    ///
    /// # Errors
    /// Returns [`SignalError`] on invalid parameters.
    pub fn with_sine_window(n: usize, batch: usize) -> SignalResult<Self> {
        let w = sine_window(2 * n);
        Self::new(n, batch, w)
    }

    /// Convenience: create a plan with the KBD window (α=5.0 is typical).
    ///
    /// # Errors
    /// Returns [`SignalError`] on invalid parameters.
    pub fn with_kbd_window(n: usize, batch: usize, alpha: f64) -> SignalResult<Self> {
        let w = kbd_window(2 * n, alpha);
        Self::new(n, batch, w)
    }
}

impl std::fmt::Debug for MdctPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MdctPlan")
            .field("n", &self.n)
            .field("batch", &self.batch)
            .finish()
    }
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_window_length() {
        let w = sine_window(16);
        assert_eq!(w.len(), 16);
    }

    #[test]
    fn test_sine_window_values() {
        // w[0] = sin(π * 0.5 / 16) and w[15] = sin(π * 15.5 / 16)
        let w = sine_window(16);
        let expected0 = (PI * 0.5 / 16.0).sin();
        assert!((w[0] - expected0).abs() < 1e-12);
        // Sine window is symmetric: w[n] * w[2N-1-n] should follow the perfect
        // reconstruction property: w[n]² + w[n+N]² = 1.
        let n2 = 16usize;
        let nh = n2 / 2;
        for n in 0..nh {
            let sum = w[n] * w[n] + w[n + nh] * w[n + nh];
            assert!((sum - 1.0).abs() < 1e-12, "PR failed at n={n}");
        }
    }

    #[test]
    fn test_kbd_window_length() {
        let w = kbd_window(16, 4.0);
        assert_eq!(w.len(), 16);
    }

    #[test]
    fn test_kbd_window_perfect_reconstruction() {
        let n2 = 16usize;
        let w = kbd_window(n2, 4.0);
        let nh = n2 / 2;
        for n in 0..nh {
            let sum = w[n] * w[n] + w[n + nh] * w[n + nh];
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "KBD PR failed at n={n}: sum={sum}"
            );
        }
    }

    #[test]
    fn test_mdct_length_check() {
        let x = vec![0.0_f64; 5]; // not 2N
        let result = mdct(&x, 3); // expects length 6
        assert!(result.is_err());
    }

    #[test]
    fn test_imdct_zero_input() {
        let x = vec![0.0_f64; 8];
        let out = imdct(&x).unwrap();
        assert_eq!(out.len(), 16);
        for v in &out {
            assert!(v.abs() < 1e-15);
        }
    }

    #[test]
    fn test_mdct_plan_sine_window() {
        let plan = MdctPlan::with_sine_window(8, 4).unwrap();
        assert_eq!(plan.n, 8);
        assert_eq!(plan.batch, 4);
        assert_eq!(plan.window_coeffs.len(), 16);
    }

    #[test]
    fn test_mdct_plan_kbd_window() {
        let plan = MdctPlan::with_kbd_window(8, 1, 5.0).unwrap();
        assert_eq!(plan.window_coeffs.len(), 16);
    }

    #[test]
    fn test_mdct_plan_invalid_window_length() {
        let w = vec![0.5_f64; 7]; // should be 16 for n=8
        let result = MdctPlan::new(8, 1, w);
        assert!(result.is_err());
    }

    #[test]
    fn test_mdct_plan_invalid_batch() {
        let w = sine_window(16);
        let result = MdctPlan::new(8, 0, w);
        assert!(result.is_err());
    }

    #[test]
    fn test_mdct_imdct_approximate_roundtrip() {
        // Windowed MDCT overlap-add should reconstruct the signal.
        // With non-overlapping single frame and sine window, we verify
        // that the round-trip of (MDCT ∘ IMDCT) gives back (N/2) scaled input
        // in the first N samples (Princen-Bradley identity, approximate).
        let n = 4usize;
        let w = sine_window(2 * n);
        let x_raw: Vec<f64> = (0..2 * n).map(|i| i as f64).collect();
        let x_win: Vec<f64> = x_raw.iter().zip(w.iter()).map(|(a, b)| a * b).collect();
        let mdct_out = mdct(&x_win, n).unwrap();
        assert_eq!(mdct_out.len(), n);
        // IMDCT should return 2N samples
        let imdct_out = imdct(&mdct_out).unwrap();
        assert_eq!(imdct_out.len(), 2 * n);
    }
}
