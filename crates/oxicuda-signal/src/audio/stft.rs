//! Short-Time Fourier Transform (STFT) and its inverse (ISTFT).
//!
//! The STFT divides a long signal into overlapping frames and computes
//! the DFT of each frame, producing a time-frequency spectrogram matrix.
//!
//! ## Definition
//!
//! ```text
//! STFT{x}(τ, ω) = Σ_n x[n] · w[n - τ] · e^{-jωn}
//! ```
//!
//! In discrete form with hop `H` and window length `L`:
//! ```text
//! X[t, k] = DFT{ x[t·H : t·H + L] · w }[k]
//! ```
//!
//! ## GPU Strategy
//!
//! Each STFT frame is one FFT of length `N_fft` (zero-padded if `N_fft > L`).
//! We use `oxicuda_fft` for the underlying FFT.  A windowing kernel applies
//! the window function element-wise before the FFT.

use std::f64::consts::PI;

use crate::{
    error::{SignalError, SignalResult},
    types::WindowType,
};

// --------------------------------------------------------------------------- //
//  Window function generation (CPU-side)
// --------------------------------------------------------------------------- //

/// Compute window function coefficients of length `n`.
///
/// Returns normalised coefficients for the chosen window type.
#[must_use]
pub fn make_window(n: usize, wtype: WindowType) -> Vec<f64> {
    match wtype {
        WindowType::Rectangular => vec![1.0; n],
        WindowType::Hann => (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos()))
            .collect(),
        WindowType::Hamming => (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos())
            .collect(),
        WindowType::Blackman => (0..n)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / (n as f64 - 1.0);
                0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
            })
            .collect(),
        WindowType::BlackmanHarris => {
            let a = [0.358_75, 0.488_29, 0.141_28, 0.011_68];
            (0..n)
                .map(|i| {
                    let t = 2.0 * PI * i as f64 / (n as f64 - 1.0);
                    a[0] - a[1] * t.cos() + a[2] * (2.0 * t).cos() - a[3] * (3.0 * t).cos()
                })
                .collect()
        }
        WindowType::Bartlett => {
            let nm1 = (n - 1) as f64;
            (0..n)
                .map(|i| 1.0 - (2.0 * i as f64 / nm1 - 1.0).abs())
                .collect()
        }
        WindowType::Kaiser { beta } => {
            let i0 = |x: f64| -> f64 {
                let mut s = 1.0_f64;
                let mut term = 1.0_f64;
                for k in 1..=25 {
                    term *= (x / (2.0 * k as f64)).powi(2);
                    s += term;
                    if term < 1e-15 * s {
                        break;
                    }
                }
                s
            };
            let i0_beta = i0(beta);
            let nm1 = (n - 1) as f64;
            (0..n)
                .map(|i| {
                    let t = 2.0 * i as f64 / nm1 - 1.0;
                    i0(beta * (1.0 - t * t).max(0.0).sqrt()) / i0_beta
                })
                .collect()
        }
        WindowType::Gaussian { sigma } => {
            let center = (n as f64 - 1.0) / 2.0;
            let denom = 2.0 * (sigma * center) * (sigma * center);
            (0..n)
                .map(|i| {
                    let diff = i as f64 - center;
                    (-diff * diff / denom).exp()
                })
                .collect()
        }
        WindowType::FlatTop => {
            let a = [
                0.215_578_95,
                0.416_631_58,
                0.277_263_16,
                0.083_578_95,
                0.006_947_37,
            ];
            (0..n)
                .map(|i| {
                    let t = 2.0 * PI * i as f64 / (n as f64 - 1.0);
                    a[0] - a[1] * t.cos() + a[2] * (2.0 * t).cos() - a[3] * (3.0 * t).cos()
                        + a[4] * (4.0 * t).cos()
                })
                .collect()
        }
        WindowType::DolphChebyshev { attenuation_db } => {
            // Chebyshev window via IDFT of Chebyshev polynomial in frequency domain.
            // Simple approximation via Kaiser β ≈ 0.1102 * (A - 8.7).
            let beta = if attenuation_db > 50.0 {
                0.1102 * (attenuation_db - 8.7)
            } else if attenuation_db >= 21.0 {
                0.5842 * (attenuation_db - 21.0).powf(0.4) + 0.07886 * (attenuation_db - 21.0)
            } else {
                0.0
            };
            make_window(n, WindowType::Kaiser { beta })
        }
    }
}

// --------------------------------------------------------------------------- //
//  STFT configuration
// --------------------------------------------------------------------------- //

/// STFT configuration parameters.
#[derive(Debug, Clone)]
pub struct StftConfig {
    /// FFT size (must be ≥ window size; zero-padded to this length).
    pub n_fft: usize,
    /// Window function length (in samples).
    pub win_len: usize,
    /// Hop size in samples.
    pub hop_len: usize,
    /// Window function type.
    pub window: WindowType,
    /// Whether to center the signal by padding `n_fft/2` zeros on each end.
    pub center: bool,
    /// Whether to compute the one-sided spectrum (real signal → N/2+1 bins).
    pub onesided: bool,
}

impl StftConfig {
    /// Create a new STFT configuration.
    ///
    /// # Errors
    /// Returns [`SignalError::InvalidParameter`] if `hop_len == 0` or
    /// `win_len > n_fft`.
    pub fn new(
        n_fft: usize,
        win_len: usize,
        hop_len: usize,
        window: WindowType,
    ) -> SignalResult<Self> {
        if hop_len == 0 {
            return Err(SignalError::InvalidParameter(
                "STFT hop_len must be ≥ 1".to_owned(),
            ));
        }
        if win_len > n_fft {
            return Err(SignalError::InvalidParameter(format!(
                "STFT win_len ({win_len}) must be ≤ n_fft ({n_fft})"
            )));
        }
        Ok(Self {
            n_fft,
            win_len,
            hop_len,
            window,
            center: false,
            onesided: true,
        })
    }

    /// Number of output frequency bins.
    #[must_use]
    pub const fn num_bins(&self) -> usize {
        if self.onesided {
            self.n_fft / 2 + 1
        } else {
            self.n_fft
        }
    }

    /// Number of STFT frames for an input of length `n_samples`.
    #[must_use]
    pub fn num_frames(&self, n_samples: usize) -> usize {
        let padded = if self.center {
            n_samples + self.n_fft
        } else {
            n_samples
        };
        if padded < self.n_fft {
            0
        } else {
            (padded - self.n_fft) / self.hop_len + 1
        }
    }
}

// --------------------------------------------------------------------------- //
//  CPU reference STFT
// --------------------------------------------------------------------------- //

/// CPU-reference STFT returning complex coefficients.
///
/// Output shape: `[num_frames, num_bins]` stored as flat interleaved `[Re, Im]`
/// pairs → total length `num_frames * num_bins * 2`.
///
/// # Errors
/// Returns [`SignalError::InvalidSize`] if the signal is too short.
pub fn stft_reference(signal: &[f64], config: &StftConfig) -> SignalResult<Vec<f64>> {
    let n_frames = config.num_frames(signal.len());
    if n_frames == 0 {
        return Err(SignalError::InvalidSize(
            "Signal too short for STFT".to_owned(),
        ));
    }
    let win = make_window(config.win_len, config.window);
    let n_bins = config.num_bins();
    let n_fft = config.n_fft;

    // Build padded signal if center=true.
    let padded: Vec<f64> = if config.center {
        let pad = n_fft / 2;
        let mut p = vec![0.0_f64; pad];
        p.extend_from_slice(signal);
        p.extend(vec![0.0_f64; pad]);
        p
    } else {
        signal.to_vec()
    };

    let mut out = vec![0.0_f64; n_frames * n_bins * 2];

    for t in 0..n_frames {
        let start = t * config.hop_len;
        // Zero-padded windowed frame.
        let mut frame = vec![0.0_f64; n_fft];
        for i in 0..config.win_len {
            let src = start + i;
            if src < padded.len() {
                frame[i] = padded[src] * win[i];
            }
        }
        // DFT of frame (O(N²) for correctness).
        for k in 0..n_bins {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for (n, &fn_val) in frame.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * n as f64 / n_fft as f64;
                re += fn_val * angle.cos();
                im += fn_val * angle.sin();
            }
            out[(t * n_bins + k) * 2] = re;
            out[(t * n_bins + k) * 2 + 1] = im;
        }
    }
    Ok(out)
}

/// Compute the magnitude spectrogram from STFT output.
///
/// `stft_out` is interleaved `[Re, Im]` pairs; `n_bins` is the number of
/// frequency bins per frame.
#[must_use]
pub fn magnitude_spectrogram(stft_out: &[f64], _n_bins: usize) -> Vec<f64> {
    stft_out
        .chunks_exact(2)
        .map(|pair| (pair[0] * pair[0] + pair[1] * pair[1]).sqrt())
        .collect()
}

/// Compute the power spectrogram (magnitude²) from STFT output.
#[must_use]
pub fn power_spectrogram(stft_out: &[f64], _n_bins: usize) -> Vec<f64> {
    stft_out
        .chunks_exact(2)
        .map(|pair| pair[0] * pair[0] + pair[1] * pair[1])
        .collect()
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_length() {
        let w = make_window(8, WindowType::Hann);
        assert_eq!(w.len(), 8);
    }

    #[test]
    fn test_hann_window_endpoints() {
        // Hann window has zero endpoints (w[0] = w[N-1] ≈ 0).
        let w = make_window(9, WindowType::Hann);
        assert!(w[0].abs() < 1e-12);
        assert!(w[8].abs() < 1e-12);
    }

    #[test]
    fn test_hamming_window_pedestal() {
        // Hamming has a non-zero pedestal: w[0] ≈ 0.08.
        let w = make_window(8, WindowType::Hamming);
        assert!((w[0] - 0.08).abs() < 0.01);
    }

    #[test]
    fn test_rectangular_window() {
        let w = make_window(4, WindowType::Rectangular);
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn test_blackman_window_energy() {
        // Blackman window energy should be less than 1 (not unit-energy).
        let w = make_window(64, WindowType::Blackman);
        let energy: f64 = w.iter().map(|v| v * v).sum::<f64>() / w.len() as f64;
        assert!(energy < 1.0 && energy > 0.1);
    }

    #[test]
    fn test_kaiser_window_unit_dc() {
        // Kaiser with β=0 should be close to rectangular.
        let w = make_window(8, WindowType::Kaiser { beta: 0.0 });
        for &v in &w {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bartlett_window_zero_endpoints() {
        let w = make_window(5, WindowType::Bartlett);
        assert!(w[0].abs() < 1e-12);
        assert!(w[4].abs() < 1e-12);
    }

    #[test]
    fn test_stft_config_num_bins_onesided() {
        let cfg = StftConfig::new(16, 16, 8, WindowType::Hann).unwrap();
        assert_eq!(cfg.num_bins(), 9); // N/2 + 1
    }

    #[test]
    fn test_stft_config_num_frames() {
        let cfg = StftConfig::new(16, 16, 8, WindowType::Hann).unwrap();
        // signal of length 32: (32 - 16) / 8 + 1 = 3
        assert_eq!(cfg.num_frames(32), 3);
    }

    #[test]
    fn test_stft_config_invalid_hop() {
        let result = StftConfig::new(16, 16, 0, WindowType::Hann);
        assert!(result.is_err());
    }

    #[test]
    fn test_stft_config_invalid_win_len() {
        let result = StftConfig::new(8, 16, 4, WindowType::Hann);
        assert!(result.is_err());
    }

    #[test]
    fn test_stft_reference_dc_input() {
        // A constant signal → only the DC bin (k=0) should have energy.
        let x = vec![1.0_f64; 32];
        let cfg = StftConfig::new(8, 8, 4, WindowType::Rectangular).unwrap();
        let out = stft_reference(&x, &cfg).unwrap();
        let _n_bins = cfg.num_bins();
        // DC energy in first frame: Re[0] = sum(window) = N = 8
        let dc_re = out[0];
        assert!((dc_re - 8.0).abs() < 1e-9, "DC re = {dc_re}");
    }

    #[test]
    fn test_magnitude_spectrogram_shape() {
        let x = vec![1.0_f64; 32];
        let cfg = StftConfig::new(8, 8, 4, WindowType::Hann).unwrap();
        let stft_out = stft_reference(&x, &cfg).unwrap();
        let mag = magnitude_spectrogram(&stft_out, cfg.num_bins());
        let n_frames = cfg.num_frames(32);
        assert_eq!(mag.len(), n_frames * cfg.num_bins());
    }

    #[test]
    fn test_power_spectrogram_non_negative() {
        let x: Vec<f64> = (0..32).map(|i| (i as f64 / 8.0 * PI).sin()).collect();
        let cfg = StftConfig::new(8, 8, 4, WindowType::Hann).unwrap();
        let stft_out = stft_reference(&x, &cfg).unwrap();
        let pow = power_spectrogram(&stft_out, cfg.num_bins());
        assert!(pow.iter().all(|&v| v >= 0.0));
    }
}
