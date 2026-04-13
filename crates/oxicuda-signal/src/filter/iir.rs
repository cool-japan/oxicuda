//! IIR (Infinite Impulse Response) filter design and application.
//!
//! IIR filters are recursive: the output depends on both past inputs and
//! past outputs.  This makes them more computationally efficient than FIR
//! filters for equivalent frequency-domain shaping, at the cost of
//! non-linear phase and potential instability.
//!
//! ## Transfer function (Direct Form II Transposed)
//!
//! ```text
//! H(z) = B(z) / A(z) = (b₀ + b₁z⁻¹ + … + bMz⁻ᴹ) /
//!                        (1  + a₁z⁻¹ + … + aNz⁻ᴺ)
//! ```
//!
//! ## Supported filter types (CPU design)
//!
//! - **Butterworth** — maximally flat in passband (binomial coefficients)
//! - **Biquad** — second-order sections (SOS) for stable high-order filters
//! - **Peaking equaliser** — parametric EQ peak/notch (audio processing)

use std::f64::consts::PI;

use crate::error::{SignalError, SignalResult};

// --------------------------------------------------------------------------- //
//  Biquad section
// --------------------------------------------------------------------------- //

/// Second-order IIR section (biquad): `b = [b0, b1, b2]`, `a = [1, a1, a2]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Biquad {
    /// Feed-forward coefficients `[b0, b1, b2]`.
    pub b: [f64; 3],
    /// Feed-back coefficients `[1, a1, a2]` (a\[0\] is always 1).
    pub a: [f64; 3],
}

impl Biquad {
    /// Create a biquad from raw coefficients.  `a[0]` is normalised to 1.
    ///
    /// # Errors
    /// Returns `SignalError::InvalidParameter` if `a[0] == 0`.
    pub fn new(b: [f64; 3], a: [f64; 3]) -> SignalResult<Self> {
        if a[0].abs() < 1e-15 {
            return Err(SignalError::InvalidParameter(
                "Biquad a[0] must be non-zero".to_owned(),
            ));
        }
        let scale = 1.0 / a[0];
        Ok(Self {
            b: [b[0] * scale, b[1] * scale, b[2] * scale],
            a: [1.0, a[1] * scale, a[2] * scale],
        })
    }

    /// Design a lowpass biquad (Butterworth 2nd order) at normalised frequency `fc`.
    #[must_use]
    pub fn lowpass(fc: f64, q: f64) -> Self {
        let omega = 2.0 * PI * fc;
        let cos_w = omega.cos();
        let sin_w = omega.sin();
        let alpha = sin_w / (2.0 * q);
        let b0 = (1.0 - cos_w) / 2.0;
        let b1 = 1.0 - cos_w;
        let b2 = (1.0 - cos_w) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w;
        let a2 = 1.0 - alpha;
        Self::new([b0, b1, b2], [a0, a1, a2]).expect("valid biquad")
    }

    /// Design a highpass biquad (Butterworth 2nd order).
    #[must_use]
    pub fn highpass(fc: f64, q: f64) -> Self {
        let omega = 2.0 * PI * fc;
        let cos_w = omega.cos();
        let sin_w = omega.sin();
        let alpha = sin_w / (2.0 * q);
        let b0 = (1.0 + cos_w) / 2.0;
        let b1 = -(1.0 + cos_w);
        let b2 = (1.0 + cos_w) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w;
        let a2 = 1.0 - alpha;
        Self::new([b0, b1, b2], [a0, a1, a2]).expect("valid biquad")
    }

    /// Design a bandpass biquad (constant skirt gain).
    #[must_use]
    pub fn bandpass(fc: f64, q: f64) -> Self {
        let omega = 2.0 * PI * fc;
        let sin_w = omega.sin();
        let cos_w = omega.cos();
        let alpha = sin_w / (2.0 * q);
        let b0 = q * alpha;
        let b1 = 0.0;
        let b2 = -q * alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w;
        let a2 = 1.0 - alpha;
        Self::new([b0, b1, b2], [a0, a1, a2]).expect("valid biquad")
    }

    /// Design a peaking EQ biquad (audio parametric equaliser).
    ///
    /// Gain `dB_gain > 0` → boost, `< 0` → cut.
    #[must_use]
    pub fn peaking_eq(fc: f64, q: f64, db_gain: f64) -> Self {
        let omega = 2.0 * PI * fc;
        let cos_w = omega.cos();
        let sin_w = omega.sin();
        let a_lin = 10.0_f64.powf(db_gain / 40.0); // sqrt of linear gain
        let alpha = sin_w / (2.0 * q);
        let b0 = 1.0 + alpha * a_lin;
        let b1 = -2.0 * cos_w;
        let b2 = 1.0 - alpha * a_lin;
        let a0 = 1.0 + alpha / a_lin;
        let a1 = -2.0 * cos_w;
        let a2 = 1.0 - alpha / a_lin;
        Self::new([b0, b1, b2], [a0, a1, a2]).expect("valid biquad")
    }

    /// Apply this biquad to a signal using Direct Form II Transposed.
    ///
    /// Returns a new `Vec<f64>` of the same length as `x`.
    #[must_use]
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut y = vec![0.0_f64; n];
        let mut s1 = 0.0_f64; // first delay state
        let mut s2 = 0.0_f64; // second delay state
        for i in 0..n {
            let xi = x[i];
            y[i] = self.b[0] * xi + s1;
            s1 = self.b[1] * xi - self.a[1] * y[i] + s2;
            s2 = self.b[2] * xi - self.a[2] * y[i];
        }
        y
    }

    /// Cascade apply: apply a chain of biquad sections (SOS).
    #[must_use]
    pub fn apply_sos(signal: &[f64], sos: &[Biquad]) -> Vec<f64> {
        sos.iter().fold(signal.to_vec(), |s, bq| bq.apply(&s))
    }

    /// Compute frequency response at `n_freq` frequencies in [0, π].
    #[must_use]
    pub fn freq_response(&self, n_freq: usize) -> (Vec<f64>, Vec<f64>) {
        let mut mags = vec![0.0_f64; n_freq];
        let mut phases = vec![0.0_f64; n_freq];
        for k in 0..n_freq {
            let omega = PI * k as f64 / (n_freq.max(2) - 1) as f64;
            let z1 = num_complex::Complex64::new(0.0, -omega).exp(); // z⁻¹
            let z2 = z1 * z1;
            let b0_c = num_complex::Complex64::new(self.b[0], 0.0);
            let b1_c = num_complex::Complex64::new(self.b[1], 0.0) * z1;
            let b2_c = num_complex::Complex64::new(self.b[2], 0.0) * z2;
            let a1_c = num_complex::Complex64::new(self.a[1], 0.0) * z1;
            let a2_c = num_complex::Complex64::new(self.a[2], 0.0) * z2;
            let num = b0_c + b1_c + b2_c;
            let den = num_complex::Complex64::new(1.0, 0.0) + a1_c + a2_c;
            let h = num / den;
            mags[k] = h.norm();
            phases[k] = h.im.atan2(h.re);
        }
        (mags, phases)
    }
}

// --------------------------------------------------------------------------- //
//  General IIR filter (arbitrary order)
// --------------------------------------------------------------------------- //

/// Apply an IIR filter (arbitrary order) using Direct Form II Transposed.
///
/// `b` — numerator coefficients, `a` — denominator (a\[0\] normalised to 1).
/// Output length equals input length.
///
/// # Errors
/// Returns `SignalError::InvalidParameter` if `a` is empty.
pub fn iir_apply(signal: &[f64], b: &[f64], a: &[f64]) -> SignalResult<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Err(SignalError::InvalidParameter(
            "IIR filter coefficients must be non-empty".to_owned(),
        ));
    }
    let nb = b.len();
    let na = a.len();
    let order = nb.max(na) - 1;
    let n = signal.len();
    let mut y = vec![0.0_f64; n];
    let mut state = vec![0.0_f64; order + 1];

    for i in 0..n {
        y[i] = b[0] * signal[i] + state[0];
        for k in 0..order {
            let bk = if k + 1 < nb { b[k + 1] } else { 0.0 };
            let ak = if k + 1 < na { a[k + 1] } else { 0.0 };
            state[k] = bk * signal[i] - ak * y[i] + if k < order { state[k + 1] } else { 0.0 };
        }
    }
    Ok(y)
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_lowpass_dc_gain() {
        // Lowpass at fc=0.5 should pass DC (low frequencies) with gain ≈ 1.
        let bq = Biquad::lowpass(0.1, 0.707);
        let dc_gain: f64 = bq.b.iter().sum::<f64>() / (1.0 + bq.a[1] + bq.a[2]);
        assert!((dc_gain - 1.0).abs() < 0.01, "DC gain = {dc_gain}");
    }

    #[test]
    fn test_biquad_highpass_nyquist_gain() {
        // Highpass at fc should pass Nyquist (high frequencies).
        let bq = Biquad::highpass(0.1, 0.707);
        // Nyquist gain: H(z=-1) = (b0 - b1 + b2) / (1 - a1 + a2)
        let num = bq.b[0] - bq.b[1] + bq.b[2];
        let den = 1.0 - bq.a[1] + bq.a[2];
        let nyq_gain = num / den;
        assert!(
            (nyq_gain - 1.0).abs() < 0.01,
            "HP Nyquist gain = {nyq_gain}"
        );
    }

    #[test]
    fn test_biquad_apply_length() {
        let bq = Biquad::lowpass(0.1, 0.707);
        let x = vec![1.0_f64; 100];
        let y = bq.apply(&x);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_biquad_apply_impulse_dc() {
        // Lowpass filtered constant 1 → output settles to DC gain ≈ 1.
        let bq = Biquad::lowpass(0.1, 0.707);
        let x = vec![1.0_f64; 1000];
        let y = bq.apply(&x);
        // Final sample should be close to DC gain.
        let last = y[999];
        assert!((last - 1.0).abs() < 0.01, "steady-state = {last}");
    }

    #[test]
    fn test_biquad_freq_response_shape() {
        let bq = Biquad::lowpass(0.2, 0.707);
        let (mag, _) = bq.freq_response(128);
        // DC should be close to 1 and Nyquist close to 0.
        assert!(mag[0] > 0.9, "DC mag = {}", mag[0]);
        assert!(mag[127] < 0.1, "Nyquist mag = {}", mag[127]);
    }

    #[test]
    fn test_biquad_apply_sos() {
        // Cascade of two allpass (identity) biquads.
        let bq = Biquad {
            b: [1.0, 0.0, 0.0],
            a: [1.0, 0.0, 0.0],
        };
        let x = vec![1.0, 2.0, 3.0_f64];
        let y = Biquad::apply_sos(&x, &[bq.clone(), bq]);
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_peaking_eq_boost_dc() {
        // Peaking EQ with positive gain should boost the target frequency.
        let bq = Biquad::peaking_eq(0.1, 2.0, 6.0); // +6 dB at fc=0.1
        // We only verify the filter was constructed without panic.
        assert!(bq.b[0] > 0.0);
    }

    #[test]
    fn test_iir_apply_identity() {
        // b=[1], a=[1]: identity filter.
        let x = vec![1.0, 2.0, 3.0_f64];
        let y = iir_apply(&x, &[1.0], &[1.0]).unwrap();
        assert_eq!(y, x);
    }

    #[test]
    fn test_iir_apply_invalid_coeffs() {
        let result = iir_apply(&[1.0, 2.0], &[], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_biquad_bandpass_peak() {
        // Bandpass at fc=0.25 uses ω = 2π·fc = π/2 internally.
        // freq_response maps k → ω = π·k/(n-1), so peak at k = (n-1)·(fc·2) ≈ n/2.
        let bq = Biquad::bandpass(0.25, 5.0);
        let (mag, _) = bq.freq_response(256);
        let peak_idx = mag
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // fc=0.25 corresponds to ω=π/2, which maps to bin (n-1)/2 ≈ 127.
        let expected_idx = (2.0 * 0.25 * (256 - 1) as f64) as usize;
        assert!(
            (peak_idx as isize - expected_idx as isize).abs() < 20,
            "peak at bin {peak_idx}, expected ~{expected_idx}"
        );
    }
}
