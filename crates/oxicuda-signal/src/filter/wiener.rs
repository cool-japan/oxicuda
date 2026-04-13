//! Wiener filter for optimal linear filtering / denoising.
//!
//! The Wiener filter minimises the mean-square error between the estimated
//! signal and the true signal, assuming stationary stochastic processes:
//!
//! ```text
//! H(ω) = S_xy(ω) / S_xx(ω)
//! ```
//!
//! where `S_xy` is the cross-spectral density and `S_xx` is the power spectral
//! density of the noisy observation.
//!
//! ## Non-parametric (frequency-domain) Wiener filter
//!
//! Given a noisy signal `y = s + n` with power spectral densities `S_s(ω)` and
//! `S_n(ω)`:
//!
//! ```text
//! H(ω) = S_s(ω) / (S_s(ω) + S_n(ω))
//! ```
//!
//! In practice `S_s(ω)` is estimated from the noisy signal by subtracting a
//! noise power estimate (computed from a noise-only segment or a noise floor
//! estimate).
//!
//! ## GPU strategy
//!
//! 1. Compute the STFT of the noisy signal.
//! 2. For each time-frequency bin, apply the Wiener gain `H = S_s / (S_s + S_n)`.
//! 3. Compute the inverse STFT.
//!
//! The Wiener gain computation is a trivial per-bin multiply, suitable for a
//! single GPU kernel.

use crate::error::{SignalError, SignalResult};

// --------------------------------------------------------------------------- //
//  Spectral estimation
// --------------------------------------------------------------------------- //

/// Estimate the noise power spectral density from a noise-only segment.
///
/// Returns a vector of length `n_bins` with the average noise power per bin.
///
/// `noise_stft` is the STFT of the noise segment: interleaved `[Re, Im]`
/// of shape `[n_frames, n_bins]`.
#[must_use]
pub fn estimate_noise_psd(noise_stft: &[f64], n_bins: usize) -> Vec<f64> {
    if noise_stft.is_empty() || n_bins == 0 {
        return vec![];
    }
    let n_frames = noise_stft.len() / (2 * n_bins);
    if n_frames == 0 {
        return vec![0.0; n_bins];
    }
    let mut psd = vec![0.0_f64; n_bins];
    for t in 0..n_frames {
        for k in 0..n_bins {
            let re = noise_stft[(t * n_bins + k) * 2];
            let im = noise_stft[(t * n_bins + k) * 2 + 1];
            psd[k] += re * re + im * im;
        }
    }
    for v in psd.iter_mut() {
        *v /= n_frames as f64;
    }
    psd
}

// --------------------------------------------------------------------------- //
//  Wiener gain computation
// --------------------------------------------------------------------------- //

/// Compute the per-bin Wiener gain `H[k] = max(0, 1 - noise_psd[k] / signal_psd[k])`.
///
/// This is the "spectral subtraction" / Wiener filter gain assuming
/// `signal_psd[k] = noisy_psd[k]` (i.e., we use the noisy observation as
/// the estimate of signal + noise PSD).
///
/// - `noisy_psd` — PSD of the noisy observation (length `n_bins`)
/// - `noise_psd` — estimated PSD of the noise (length `n_bins`)
/// - `over_sub` — over-subtraction factor α (typically 1.0–2.0)
/// - `floor` — spectral floor β to prevent musical noise (typically 0.001–0.01)
///
/// Returns gains in `[floor, 1]`.
#[must_use]
pub fn wiener_gain(noisy_psd: &[f64], noise_psd: &[f64], over_sub: f64, floor: f64) -> Vec<f64> {
    noisy_psd
        .iter()
        .zip(noise_psd.iter())
        .map(|(&y, &n)| {
            if y <= 0.0 {
                floor
            } else {
                let gain = 1.0 - over_sub * n / y;
                gain.max(floor)
            }
        })
        .collect()
}

// --------------------------------------------------------------------------- //
//  Apply Wiener gain to STFT
// --------------------------------------------------------------------------- //

/// Apply per-bin Wiener gains to an STFT magnitude/phase representation.
///
/// `stft_out` — interleaved `[Re, Im]` of shape `[n_frames, n_bins]`.
/// `gains` — per-bin gain `[n_bins]` (applied to all frames).
/// Modifies `stft_out` in-place by multiplying each bin's magnitude by its gain.
///
/// # Errors
/// Returns `SignalError::DimensionMismatch` if lengths are inconsistent.
pub fn apply_wiener_gains(stft_out: &mut [f64], gains: &[f64], n_bins: usize) -> SignalResult<()> {
    if stft_out.len() % (2 * n_bins) != 0 {
        return Err(SignalError::DimensionMismatch {
            expected: format!("stft_out length multiple of 2*{n_bins}"),
            got: format!("{}", stft_out.len()),
        });
    }
    if gains.len() != n_bins {
        return Err(SignalError::DimensionMismatch {
            expected: format!("gains length = {n_bins}"),
            got: format!("{}", gains.len()),
        });
    }
    let n_frames = stft_out.len() / (2 * n_bins);
    for t in 0..n_frames {
        for k in 0..n_bins {
            let g = gains[k];
            stft_out[(t * n_bins + k) * 2] *= g;
            stft_out[(t * n_bins + k) * 2 + 1] *= g;
        }
    }
    Ok(())
}

// --------------------------------------------------------------------------- //
//  Adaptive Wiener filter (1D, time-domain)
// --------------------------------------------------------------------------- //

/// Apply a local (block-adaptive) Wiener filter to a 1D signal.
///
/// For each local block of length `block_size`, the local mean and variance
/// are computed; the Wiener gain is:
/// ```text
/// g = max(0, (σ² - noise_var) / σ²)
/// output[n] = μ + g · (x[n] - μ)
/// ```
///
/// `noise_var` is the estimated noise variance (assumed constant).
#[must_use]
pub fn local_wiener_1d(signal: &[f64], block_size: usize, noise_var: f64) -> Vec<f64> {
    let n = signal.len();
    let half = block_size / 2;
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let block = &signal[lo..hi];
        let len = block.len() as f64;
        let mu: f64 = block.iter().sum::<f64>() / len;
        let var: f64 = block.iter().map(|&v| (v - mu) * (v - mu)).sum::<f64>() / len;
        let gain = if var <= 0.0 {
            0.0
        } else {
            ((var - noise_var) / var).max(0.0)
        };
        out[i] = mu + gain * (signal[i] - mu);
    }
    out
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_noise_psd_shape() {
        // 2 frames, 4 bins → noise_stft length = 2 * 2 * 4 = 16 (interleaved)
        let noise_stft = vec![1.0_f64; 16];
        let psd = estimate_noise_psd(&noise_stft, 4);
        assert_eq!(psd.len(), 4);
    }

    #[test]
    fn test_estimate_noise_psd_dc() {
        // Re=1, Im=0 for all → psd[k] = 1²+0² = 1 for all k.
        let mut noise_stft = vec![0.0_f64; 2 * 3 * 4]; // 3 frames, 4 bins
        // Set Re=1 for all bins, Im=0
        for t in 0..3 {
            for k in 0..4 {
                noise_stft[(t * 4 + k) * 2] = 1.0;
            }
        }
        let psd = estimate_noise_psd(&noise_stft, 4);
        for v in &psd {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_wiener_gain_no_noise() {
        let noisy_psd = vec![1.0, 2.0, 3.0_f64];
        let noise_psd = vec![0.0, 0.0, 0.0_f64];
        let gains = wiener_gain(&noisy_psd, &noise_psd, 1.0, 0.01);
        // All gains should be 1.0 (no noise to subtract)
        for &g in &gains {
            assert!((g - 1.0).abs() < 1e-12, "gain={g}");
        }
    }

    #[test]
    fn test_wiener_gain_full_noise() {
        let noisy_psd = vec![1.0, 1.0, 1.0_f64];
        let noise_psd = vec![1.0, 1.0, 1.0_f64]; // SNR = 0 dB
        let floor = 0.01;
        let gains = wiener_gain(&noisy_psd, &noise_psd, 1.0, floor);
        // All gains should be at the floor
        for &g in &gains {
            assert!((g - floor).abs() < 1e-12, "gain={g}");
        }
    }

    #[test]
    fn test_wiener_gain_floor_respected() {
        let noisy = vec![1.0_f64];
        let noise = vec![10.0_f64]; // noise > signal
        let floor = 0.05;
        let gains = wiener_gain(&noisy, &noise, 1.0, floor);
        assert!(gains[0] >= floor);
    }

    #[test]
    fn test_apply_wiener_gains_unity() {
        let mut stft = vec![2.0_f64; 8]; // 2 bins, 2 frames
        let gains = vec![1.0, 1.0_f64];
        apply_wiener_gains(&mut stft, &gains, 2).unwrap();
        assert!(stft.iter().all(|&v| (v - 2.0).abs() < 1e-12));
    }

    #[test]
    fn test_apply_wiener_gains_zero() {
        let mut stft = vec![5.0_f64; 8];
        let gains = vec![0.0, 0.0_f64];
        apply_wiener_gains(&mut stft, &gains, 2).unwrap();
        assert!(stft.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn test_apply_wiener_gains_mismatch() {
        let mut stft = vec![1.0_f64; 8]; // should be multiple of 2*n_bins
        let gains = vec![1.0_f64; 3]; // mismatch
        assert!(apply_wiener_gains(&mut stft, &gains, 3).is_err());
    }

    #[test]
    fn test_local_wiener_1d_clean_signal() {
        // Signal with zero noise_var → output = signal
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = local_wiener_1d(&x, 3, 0.0);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-10, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_local_wiener_1d_high_noise_smoothing() {
        // With very large noise_var relative to signal variance,
        // the output should be smoothed toward the local mean.
        let x = vec![1.0, 10.0, 1.0, 10.0, 1.0_f64];
        let y = local_wiener_1d(&x, 5, 100.0); // noise >> signal variance
        // Central sample gain → floor → output ≈ local mean
        let local_mean = x.iter().sum::<f64>() / x.len() as f64;
        assert!(
            (y[2] - local_mean).abs() < 1.0,
            "y[2]={} vs mean={local_mean}",
            y[2]
        );
    }
}
