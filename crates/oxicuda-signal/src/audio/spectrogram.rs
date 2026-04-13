//! Spectrogram variants: magnitude, power, dB-scaled, and chroma.
//!
//! Provides a unified interface for computing and manipulating spectrograms
//! built on top of the STFT module.

use crate::{
    audio::stft::{StftConfig, power_spectrogram, stft_reference},
    error::{SignalError, SignalResult},
    types::WindowType,
};

// --------------------------------------------------------------------------- //
//  Spectrogram types
// --------------------------------------------------------------------------- //

/// Type of spectrogram to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrogramType {
    /// Magnitude: `|X[t, k]|`
    Magnitude,
    /// Power: `|X[t, k]|²`
    Power,
    /// Decibel-scaled power: `10·log₁₀(power + ref_power)`
    PowerDb,
    /// Decibel-scaled magnitude: `20·log₁₀(magnitude + ref_mag)`
    MagnitudeDb,
}

/// Configuration for spectrogram computation.
#[derive(Debug, Clone)]
pub struct SpectrogramConfig {
    /// FFT size.
    pub n_fft: usize,
    /// Hop size in samples.
    pub hop_len: usize,
    /// Window length.
    pub win_len: usize,
    /// Window type.
    pub window: WindowType,
    /// Type of spectrogram to compute.
    pub spec_type: SpectrogramType,
    /// Reference value for dB scaling (top-normalise by `max` if 0.0).
    pub ref_value: f64,
    /// Minimum dB value (floor): values below this are clipped.
    pub amin: f64,
}

impl SpectrogramConfig {
    /// Create a standard power spectrogram config.
    ///
    /// # Errors
    /// Returns `SignalError::InvalidParameter` if `hop_len == 0`.
    pub fn new(
        n_fft: usize,
        hop_len: usize,
        window: WindowType,
        spec_type: SpectrogramType,
    ) -> SignalResult<Self> {
        if hop_len == 0 {
            return Err(SignalError::InvalidParameter(
                "hop_len must be ≥ 1".to_owned(),
            ));
        }
        Ok(Self {
            n_fft,
            hop_len,
            win_len: n_fft,
            window,
            spec_type,
            ref_value: 1.0,
            amin: 1e-10,
        })
    }

    /// Number of frequency bins.
    #[must_use]
    pub const fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

// --------------------------------------------------------------------------- //
//  Spectrogram computation
// --------------------------------------------------------------------------- //

/// Compute a spectrogram of the given type.
///
/// Returns a flat array of shape `[n_frames, n_bins]`.
///
/// # Errors
/// Returns [`SignalError`] if the signal is too short.
pub fn spectrogram(signal: &[f64], config: &SpectrogramConfig) -> SignalResult<Vec<f64>> {
    let stft_cfg = StftConfig::new(config.n_fft, config.win_len, config.hop_len, config.window)?;
    let stft_out = stft_reference(signal, &stft_cfg)?;
    let n_bins = stft_cfg.num_bins();
    let pow = power_spectrogram(&stft_out, n_bins);

    let out: Vec<f64> = match config.spec_type {
        SpectrogramType::Power => pow,
        SpectrogramType::Magnitude => pow.iter().map(|v| v.sqrt()).collect(),
        SpectrogramType::PowerDb => {
            let ref_val = if config.ref_value == 0.0 {
                pow.iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(config.amin)
            } else {
                config.ref_value
            };
            pow.iter()
                .map(|&p| 10.0 * (p.max(config.amin) / ref_val).log10())
                .collect()
        }
        SpectrogramType::MagnitudeDb => {
            let mag: Vec<f64> = pow.iter().map(|v| v.sqrt()).collect();
            let ref_val = if config.ref_value == 0.0 {
                mag.iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(config.amin)
            } else {
                config.ref_value
            };
            mag.iter()
                .map(|&m| 20.0 * (m.max(config.amin) / ref_val).log10())
                .collect()
        }
    };

    Ok(out)
}

// --------------------------------------------------------------------------- //
//  Chroma features
// --------------------------------------------------------------------------- //

/// Compute chroma (pitch class profile) features from a power spectrogram.
///
/// Maps FFT bins to 12 semitone bins (pitch classes A–G#).
///
/// # Arguments
/// - `power` — flat power spectrogram `[n_frames, n_bins]`
/// - `n_bins` — number of FFT bins
/// - `sample_rate` — sample rate in Hz
/// - `n_fft` — FFT size
/// - `n_chroma` — number of chroma bins (typically 12)
///
/// Returns flat array `[n_frames, n_chroma]`.
///
/// # Errors
/// Returns `SignalError::InvalidParameter` if `n_chroma == 0`.
pub fn chroma_from_power(
    power: &[f64],
    n_bins: usize,
    sample_rate: f64,
    n_fft: usize,
    n_chroma: usize,
) -> SignalResult<Vec<f64>> {
    if n_chroma == 0 {
        return Err(SignalError::InvalidParameter(
            "n_chroma must be ≥ 1".to_owned(),
        ));
    }
    let n_frames = power.len() / n_bins;
    if n_frames == 0 || power.len() % n_bins != 0 {
        return Err(SignalError::InvalidSize(
            "power length must be divisible by n_bins".to_owned(),
        ));
    }

    // Frequency of each FFT bin.
    let bin_freqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect();

    // Chroma bin for each FFT bin (skip DC).
    // chroma_bin = round(12 * log₂(f / A4)) mod 12, where A4 = 440 Hz.
    let a4 = 440.0_f64;
    let chroma_map: Vec<usize> = bin_freqs
        .iter()
        .map(|&f| {
            if f <= 0.0 {
                0
            } else {
                let bin = (n_chroma as f64 * (f / a4).log2()).round() as isize;
                bin.rem_euclid(n_chroma as isize) as usize
            }
        })
        .collect();

    let mut out = vec![0.0_f64; n_frames * n_chroma];
    for t in 0..n_frames {
        for k in 0..n_bins {
            let c = chroma_map[k];
            out[t * n_chroma + c] += power[t * n_bins + k];
        }
    }

    // L2-normalise each frame.
    for t in 0..n_frames {
        let frame = &mut out[t * n_chroma..(t + 1) * n_chroma];
        let norm: f64 = frame.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for v in frame.iter_mut() {
                *v /= norm;
            }
        }
    }

    Ok(out)
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_sine(n: usize, freq: f64, sr: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn test_power_spectrogram_shape() {
        let x = make_sine(512, 440.0, 8000.0);
        let cfg = SpectrogramConfig::new(64, 32, WindowType::Hann, SpectrogramType::Power).unwrap();
        let s = spectrogram(&x, &cfg).unwrap();
        let n_frames = StftConfig::new(64, 64, 32, WindowType::Hann)
            .unwrap()
            .num_frames(512);
        assert_eq!(s.len(), n_frames * cfg.n_bins());
    }

    #[test]
    fn test_power_spectrogram_non_negative() {
        let x = make_sine(256, 440.0, 8000.0);
        let cfg = SpectrogramConfig::new(32, 16, WindowType::Hann, SpectrogramType::Power).unwrap();
        let s = spectrogram(&x, &cfg).unwrap();
        assert!(s.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_magnitude_spectrogram_non_negative() {
        let x = make_sine(256, 440.0, 8000.0);
        let cfg =
            SpectrogramConfig::new(32, 16, WindowType::Hann, SpectrogramType::Magnitude).unwrap();
        let s = spectrogram(&x, &cfg).unwrap();
        assert!(s.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_power_db_spectrogram_finite() {
        let x = make_sine(256, 440.0, 8000.0);
        let cfg =
            SpectrogramConfig::new(32, 16, WindowType::Hann, SpectrogramType::PowerDb).unwrap();
        let s = spectrogram(&x, &cfg).unwrap();
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_spectrogram_invalid_hop() {
        let result = SpectrogramConfig::new(32, 0, WindowType::Hann, SpectrogramType::Power);
        assert!(result.is_err());
    }

    #[test]
    fn test_chroma_shape() {
        let power = vec![1.0_f64; 3 * 17]; // 3 frames, 17 bins
        let chroma = chroma_from_power(&power, 17, 8000.0, 32, 12).unwrap();
        assert_eq!(chroma.len(), 3 * 12);
    }

    #[test]
    fn test_chroma_l2_normalized() {
        let power = vec![1.0_f64; 3 * 17];
        let chroma = chroma_from_power(&power, 17, 8000.0, 32, 12).unwrap();
        for t in 0..3 {
            let norm: f64 = chroma[t * 12..(t + 1) * 12]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "frame {t} norm={norm}");
        }
    }

    #[test]
    fn test_chroma_invalid_n_chroma() {
        let power = vec![1.0_f64; 10];
        let result = chroma_from_power(&power, 5, 8000.0, 8, 0);
        assert!(result.is_err());
    }
}
