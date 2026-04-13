//! Mel filterbank computation for audio feature extraction.
//!
//! The mel scale converts linear frequency (Hz) to the perceptual mel scale:
//!
//! ```text
//! m = 2595 · log₁₀(1 + f / 700)
//! f = 700 · (10^(m / 2595) - 1)
//! ```
//!
//! A mel filterbank consists of M triangular filters equally spaced on the mel
//! scale between `f_min` and `f_max`.  Each filter is a trapezoid whose
//! response is 1 at the centre frequency and linearly decays to 0 at the
//! adjacent centres.
//!
//! ## GPU Strategy
//!
//! The filterbank matrix `F ∈ ℝ^{M × (N/2+1)}` (sparse with triangular bands)
//! is pre-computed on the host and stored as a dense matrix.  Applying the
//! filterbank to a power spectrogram frame is then a GEMM:
//! `mel = F × power_frame`.

use crate::error::{SignalError, SignalResult};

// --------------------------------------------------------------------------- //
//  Mel scale conversions
// --------------------------------------------------------------------------- //

/// Convert frequency in Hz to mel scale.
#[must_use]
pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel value to frequency in Hz.
#[must_use]
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

// --------------------------------------------------------------------------- //
//  Filterbank construction
// --------------------------------------------------------------------------- //

/// Configuration for a mel filterbank.
#[derive(Debug, Clone)]
pub struct MelFilterbankConfig {
    /// Number of mel filters.
    pub n_mels: usize,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// FFT size (N_fft).
    pub n_fft: usize,
    /// Minimum frequency (Hz).
    pub f_min: f64,
    /// Maximum frequency (Hz).  Defaults to `sample_rate / 2`.
    pub f_max: f64,
    /// Whether to normalise each filter to unit area (slaney-style).
    pub norm: bool,
}

impl MelFilterbankConfig {
    /// Create a new filterbank config with sensible defaults.
    ///
    /// # Errors
    /// Returns `SignalError::InvalidParameter` if `n_mels == 0` or
    /// `f_min >= f_max`.
    pub fn new(
        n_mels: usize,
        sample_rate: f64,
        n_fft: usize,
        f_min: f64,
        f_max: f64,
    ) -> SignalResult<Self> {
        if n_mels == 0 {
            return Err(SignalError::InvalidParameter(
                "n_mels must be ≥ 1".to_owned(),
            ));
        }
        if f_min >= f_max {
            return Err(SignalError::InvalidParameter(format!(
                "f_min ({f_min}) must be < f_max ({f_max})"
            )));
        }
        Ok(Self {
            n_mels,
            sample_rate,
            n_fft,
            f_min,
            f_max,
            norm: true,
        })
    }

    /// Number of FFT bins (N_fft / 2 + 1 for one-sided spectrum).
    #[must_use]
    pub const fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// Compute the mel filterbank matrix.
///
/// Returns a matrix of shape `[n_mels, n_bins]` in row-major order.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] on invalid config.
pub fn mel_filterbank(config: &MelFilterbankConfig) -> SignalResult<Vec<f64>> {
    let n_bins = config.n_bins();
    let n_mels = config.n_mels;

    // Frequency of each FFT bin in Hz.
    let bin_freqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * config.sample_rate / config.n_fft as f64)
        .collect();

    // Mel centre frequencies: M+2 equally spaced points on mel scale.
    let m_min = hz_to_mel(config.f_min);
    let m_max = hz_to_mel(config.f_max);
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|m| mel_to_hz(m_min + (m_max - m_min) * m as f64 / (n_mels + 1) as f64))
        .collect();

    let mut matrix = vec![0.0_f64; n_mels * n_bins];

    for m in 0..n_mels {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];

        for k in 0..n_bins {
            let f = bin_freqs[k];
            let val = if f >= f_left && f <= f_center {
                (f - f_left) / (f_center - f_left)
            } else if f > f_center && f <= f_right {
                (f_right - f) / (f_right - f_center)
            } else {
                0.0
            };
            matrix[m * n_bins + k] = val;
        }

        // Slaney normalisation: divide by filter bandwidth in Hz.
        if config.norm {
            let bandwidth = f_right - f_left;
            if bandwidth > 0.0 {
                for k in 0..n_bins {
                    matrix[m * n_bins + k] /= bandwidth;
                }
            }
        }
    }

    Ok(matrix)
}

/// Apply a pre-computed mel filterbank to a power spectrogram frame.
///
/// `filterbank` has shape `[n_mels, n_bins]`.
/// `power_frame` has length `n_bins`.
/// Returns a mel-scaled vector of length `n_mels`.
///
/// # Errors
/// Returns `SignalError::DimensionMismatch` on shape mismatch.
pub fn apply_filterbank(
    filterbank: &[f64],
    power_frame: &[f64],
    n_mels: usize,
) -> SignalResult<Vec<f64>> {
    let n_bins = power_frame.len();
    if filterbank.len() != n_mels * n_bins {
        return Err(SignalError::DimensionMismatch {
            expected: format!("{}", n_mels * n_bins),
            got: format!("{}", filterbank.len()),
        });
    }
    let mut out = vec![0.0_f64; n_mels];
    for m in 0..n_mels {
        out[m] = filterbank[m * n_bins..]
            .iter()
            .zip(power_frame.iter())
            .take(n_bins)
            .map(|(f, p)| f * p)
            .sum();
    }
    Ok(out)
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_mel_roundtrip() {
        for &hz in &[100.0, 440.0, 1000.0, 8000.0, 16000.0_f64] {
            let m = hz_to_mel(hz);
            let hz2 = mel_to_hz(m);
            assert!((hz - hz2).abs() < 1e-8, "roundtrip failed for {hz}");
        }
    }

    #[test]
    fn test_mel_scale_monotone() {
        let freqs = [100.0, 500.0, 1000.0, 4000.0, 8000.0_f64];
        let mels: Vec<f64> = freqs.iter().map(|&f| hz_to_mel(f)).collect();
        for w in mels.windows(2) {
            assert!(w[1] > w[0], "mel scale must be monotone");
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let cfg = MelFilterbankConfig::new(40, 22050.0, 512, 0.0, 8000.0).unwrap();
        let fb = mel_filterbank(&cfg).unwrap();
        assert_eq!(fb.len(), 40 * cfg.n_bins());
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let cfg = MelFilterbankConfig::new(40, 22050.0, 512, 0.0, 8000.0).unwrap();
        let fb = mel_filterbank(&cfg).unwrap();
        assert!(fb.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_mel_filterbank_invalid_n_mels() {
        let result = MelFilterbankConfig::new(0, 22050.0, 512, 0.0, 8000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_mel_filterbank_invalid_freq_range() {
        let result = MelFilterbankConfig::new(40, 22050.0, 512, 8000.0, 100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_filterbank_dc() {
        // All-ones power frame: mel output = row sums of filterbank.
        let cfg = MelFilterbankConfig::new(4, 8000.0, 64, 0.0, 4000.0).unwrap();
        let fb = mel_filterbank(&cfg).unwrap();
        let n_bins = cfg.n_bins();
        let power = vec![1.0_f64; n_bins];
        let mel_out = apply_filterbank(&fb, &power, 4).unwrap();
        assert_eq!(mel_out.len(), 4);
        // Each filter's output should be positive
        assert!(mel_out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_apply_filterbank_zero_input() {
        let cfg = MelFilterbankConfig::new(4, 8000.0, 64, 0.0, 4000.0).unwrap();
        let fb = mel_filterbank(&cfg).unwrap();
        let n_bins = cfg.n_bins();
        let power = vec![0.0_f64; n_bins];
        let mel_out = apply_filterbank(&fb, &power, 4).unwrap();
        assert!(mel_out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_apply_filterbank_shape_mismatch() {
        let fb = vec![1.0_f64; 10];
        let power = vec![1.0_f64; 5];
        let result = apply_filterbank(&fb, &power, 3); // 3*5=15 ≠ 10
        assert!(result.is_err());
    }
}
