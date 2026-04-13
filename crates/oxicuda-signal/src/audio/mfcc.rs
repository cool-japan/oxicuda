//! Mel-Frequency Cepstral Coefficients (MFCCs).
//!
//! MFCCs are the standard feature representation for automatic speech
//! recognition (ASR) and audio classification.
//!
//! ## Pipeline
//!
//! ```text
//! signal  →  STFT  →  power spec  →  mel filterbank  →  log  →  DCT-II  →  MFCCs
//! ```
//!
//! 1. Compute the STFT of the input signal.
//! 2. Compute the power spectrogram.
//! 3. Apply a mel filterbank to each frame (output: `[T, M]` log-mel matrix).
//! 4. Apply DCT-II along the mel axis → `[T, n_mfcc]` MFCC matrix.
//!
//! The DCT-II here is the same as in JPEG image compression, providing
//! decorrelated coefficients.

use crate::{
    audio::mel::{MelFilterbankConfig, apply_filterbank, mel_filterbank},
    audio::stft::{StftConfig, power_spectrogram, stft_reference},
    dct::dct2::dct2_reference,
    error::{SignalError, SignalResult},
    types::WindowType,
};

// --------------------------------------------------------------------------- //
//  MFCC configuration
// --------------------------------------------------------------------------- //

/// Configuration for MFCC feature extraction.
#[derive(Debug, Clone)]
pub struct MfccConfig {
    /// Number of MFCC coefficients to return.
    pub n_mfcc: usize,
    /// Number of mel filterbank channels.
    pub n_mels: usize,
    /// FFT size.
    pub n_fft: usize,
    /// Hop size in samples.
    pub hop_len: usize,
    /// Window length in samples.
    pub win_len: usize,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Minimum frequency for mel filterbank (Hz).
    pub f_min: f64,
    /// Maximum frequency for mel filterbank (Hz).
    pub f_max: f64,
    /// Whether to use log-mel (true) or raw-mel (false) before DCT.
    pub use_log: bool,
    /// Log offset to avoid log(0): `log(mel + log_offset)`.
    pub log_offset: f64,
    /// Window function type.
    pub window: WindowType,
}

impl MfccConfig {
    /// Create a standard MFCC configuration.
    ///
    /// Defaults: Hann window, `f_min=0`, `f_max=sample_rate/2`, log-mel.
    ///
    /// # Errors
    /// Returns [`SignalError::InvalidParameter`] if `n_mfcc > n_mels`.
    pub fn new(
        n_mfcc: usize,
        n_mels: usize,
        n_fft: usize,
        hop_len: usize,
        sample_rate: f64,
    ) -> SignalResult<Self> {
        if n_mfcc > n_mels {
            return Err(SignalError::InvalidParameter(format!(
                "n_mfcc ({n_mfcc}) must be ≤ n_mels ({n_mels})"
            )));
        }
        if hop_len == 0 {
            return Err(SignalError::InvalidParameter(
                "hop_len must be ≥ 1".to_owned(),
            ));
        }
        Ok(Self {
            n_mfcc,
            n_mels,
            n_fft,
            hop_len,
            win_len: n_fft,
            sample_rate,
            f_min: 0.0,
            f_max: sample_rate / 2.0,
            use_log: true,
            log_offset: 1e-10,
            window: WindowType::Hann,
        })
    }
}

// --------------------------------------------------------------------------- //
//  MFCC computation (CPU reference)
// --------------------------------------------------------------------------- //

/// Compute MFCCs for a mono audio signal.
///
/// Returns a flat array of shape `[n_frames, n_mfcc]`.
///
/// # Errors
/// Returns [`SignalError`] on invalid config or signal too short.
pub fn mfcc(signal: &[f64], config: &MfccConfig) -> SignalResult<Vec<f64>> {
    // 1. STFT
    let stft_cfg = StftConfig::new(config.n_fft, config.win_len, config.hop_len, config.window)?;
    let stft_out = stft_reference(signal, &stft_cfg)?;
    let n_bins = stft_cfg.num_bins();
    let n_frames = stft_cfg.num_frames(signal.len());

    // 2. Power spectrogram: shape [n_frames, n_bins]
    let power = power_spectrogram(&stft_out, n_bins);

    // 3. Mel filterbank: shape [n_mels, n_bins] → apply to each frame
    let mel_cfg = MelFilterbankConfig::new(
        config.n_mels,
        config.sample_rate,
        config.n_fft,
        config.f_min,
        config.f_max,
    )?;
    let fb = mel_filterbank(&mel_cfg)?;

    // Mel energies: shape [n_frames, n_mels]
    let mut mel_energy = vec![0.0_f64; n_frames * config.n_mels];
    for t in 0..n_frames {
        let frame = &power[t * n_bins..(t + 1) * n_bins];
        let mel_frame = apply_filterbank(&fb, frame, config.n_mels)?;
        for m in 0..config.n_mels {
            mel_energy[t * config.n_mels + m] = mel_frame[m];
        }
    }

    // 4. Log compression
    if config.use_log {
        for v in mel_energy.iter_mut() {
            *v = (*v + config.log_offset).ln();
        }
    }

    // 5. DCT-II along mel axis → take first n_mfcc coefficients
    let mut mfcc_out = vec![0.0_f64; n_frames * config.n_mfcc];
    for t in 0..n_frames {
        let mel_frame = &mel_energy[t * config.n_mels..(t + 1) * config.n_mels];
        let dct_out = dct2_reference(mel_frame);
        for k in 0..config.n_mfcc {
            mfcc_out[t * config.n_mfcc + k] = dct_out[k];
        }
    }

    Ok(mfcc_out)
}

// --------------------------------------------------------------------------- //
//  Delta MFCCs (velocity and acceleration coefficients)
// --------------------------------------------------------------------------- //

/// Compute delta (velocity) features using the standard regression formula.
///
/// For each frame `t` and coefficient `c`:
/// ```text
/// Δ[t, c] = Σ_{m=1}^{width} m · (feat[t+m, c] - feat[t-m, c]) / (2 · Σ_{m=1}^{width} m²)
/// ```
///
/// Boundary frames use edge-replication (constant padding).
///
/// `feats` has shape `[n_frames, n_feats]` stored flat.
/// `width` is the number of context frames on each side (typically 2).
#[must_use]
pub fn delta_features(feats: &[f64], n_frames: usize, n_feats: usize, width: usize) -> Vec<f64> {
    let denom: f64 = 2.0 * (1..=width).map(|m| (m * m) as f64).sum::<f64>();
    let mut delta = vec![0.0_f64; n_frames * n_feats];
    for t in 0..n_frames {
        for c in 0..n_feats {
            let mut num = 0.0_f64;
            for m in 1..=width as isize {
                let tp = (t as isize + m).clamp(0, n_frames as isize - 1) as usize;
                let tm = (t as isize - m).clamp(0, n_frames as isize - 1) as usize;
                num += m as f64 * (feats[tp * n_feats + c] - feats[tm * n_feats + c]);
            }
            delta[t * n_feats + c] = num / denom;
        }
    }
    delta
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_signal(freq: f64, sr: f64, dur: f64) -> Vec<f64> {
        let n = (sr * dur) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn test_mfcc_config_valid() {
        let cfg = MfccConfig::new(13, 40, 512, 160, 16000.0);
        assert!(cfg.is_ok());
    }

    #[test]
    fn test_mfcc_config_n_mfcc_gt_n_mels() {
        let cfg = MfccConfig::new(50, 40, 512, 160, 16000.0);
        assert!(cfg.is_err());
    }

    #[test]
    fn test_mfcc_config_zero_hop() {
        let cfg = MfccConfig::new(13, 40, 512, 0, 16000.0);
        assert!(cfg.is_err());
    }

    #[test]
    fn test_mfcc_output_shape() {
        // 0.5 seconds of 440 Hz sine at 16 kHz.
        let sr = 8000.0_f64;
        let x = sine_signal(440.0, sr, 0.5);
        let cfg = MfccConfig::new(13, 40, 256, 128, sr).unwrap();
        let out = mfcc(&x, &cfg).unwrap();
        let n_frames = StftConfig::new(256, 256, 128, WindowType::Hann)
            .unwrap()
            .num_frames(x.len());
        assert_eq!(out.len(), n_frames * 13);
    }

    #[test]
    fn test_mfcc_finite_values() {
        let sr = 8000.0_f64;
        let x = sine_signal(440.0, sr, 0.2);
        let cfg = MfccConfig::new(13, 40, 256, 128, sr).unwrap();
        let out = mfcc(&x, &cfg).unwrap();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "MFCC contains non-finite values"
        );
    }

    #[test]
    fn test_delta_features_shape() {
        let feats = vec![1.0_f64; 10 * 13]; // 10 frames, 13 feats
        let delta = delta_features(&feats, 10, 13, 2);
        assert_eq!(delta.len(), 10 * 13);
    }

    #[test]
    fn test_delta_features_constant_zero() {
        // Constant features → delta = 0
        let feats = vec![1.0_f64; 5 * 4];
        let delta = delta_features(&feats, 5, 4, 2);
        assert!(delta.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn test_delta_features_linear_ramp() {
        // Features that are linear ramps → delta should be approximately constant
        let n_frames = 10usize;
        let n_feats = 1usize;
        let feats: Vec<f64> = (0..n_frames).map(|t| t as f64).collect();
        let delta = delta_features(&feats, n_frames, n_feats, 2);
        // Interior frames: delta ≈ 1.0
        for (t, &dt) in delta.iter().enumerate().skip(2).take(n_frames - 4) {
            assert!((dt - 1.0).abs() < 1e-10, "delta[{t}]={}", dt);
        }
    }
}
