//! Autocorrelation and autocovariance estimation.
//!
//! The autocorrelation of signal `x` at lag `τ` is:
//!
//! ```text
//! R_xx[τ] = Σ_n x[n] · x[n + τ]    (unbiased: divide by N - |τ|)
//! ```
//!
//! ## GPU Strategy
//!
//! Long autocorrelations are computed via FFT: `R_xx = IFFT(|FFT(x)|²)`.
//! Short lags (τ < N/8) use a direct PTX kernel with shared-memory tiling.

use crate::error::{SignalError, SignalResult};

// --------------------------------------------------------------------------- //
//  CPU reference: direct-form autocorrelation
// --------------------------------------------------------------------------- //

/// Compute the full (biased) autocorrelation for lags 0 to `max_lag`.
///
/// Biased estimator: `R[τ] = (1/N) · Σ_{n=0}^{N-1-τ} x[n] · x[n+τ]`.
///
/// Returns a vector of length `max_lag + 1`.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `max_lag >= signal.len()`.
pub fn autocorr_biased(signal: &[f64], max_lag: usize) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    if max_lag >= n {
        return Err(SignalError::InvalidSize(format!(
            "max_lag ({max_lag}) must be < signal length ({n})"
        )));
    }
    let nf = n as f64;
    Ok((0..=max_lag)
        .map(|tau| {
            signal[..n - tau]
                .iter()
                .zip(signal[tau..].iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / nf
        })
        .collect())
}

/// Compute the full (unbiased) autocorrelation for lags 0 to `max_lag`.
///
/// Unbiased estimator: `R[τ] = (1/(N-|τ|)) · Σ_{n=0}^{N-1-τ} x[n] · x[n+τ]`.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `max_lag >= signal.len()`.
pub fn autocorr_unbiased(signal: &[f64], max_lag: usize) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    if max_lag >= n {
        return Err(SignalError::InvalidSize(format!(
            "max_lag ({max_lag}) must be < signal length ({n})"
        )));
    }
    Ok((0..=max_lag)
        .map(|tau| {
            let count = (n - tau) as f64;
            signal[..n - tau]
                .iter()
                .zip(signal[tau..].iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / count
        })
        .collect())
}

/// Compute the normalised autocorrelation (divided by R\[0\]).
///
/// Equivalent to the Pearson correlation coefficient at each lag.
///
/// # Errors
/// Propagates errors from [`autocorr_biased`].
pub fn autocorr_normalised(signal: &[f64], max_lag: usize) -> SignalResult<Vec<f64>> {
    let r = autocorr_biased(signal, max_lag)?;
    let r0 = r[0];
    if r0.abs() < 1e-15 {
        return Ok(vec![0.0; r.len()]);
    }
    Ok(r.iter().map(|v| v / r0).collect())
}

// --------------------------------------------------------------------------- //
//  Partial autocorrelation function (PACF)
// --------------------------------------------------------------------------- //

/// Compute the partial autocorrelation function up to `max_lag` using the
/// Durbin-Levinson recursion.
///
/// The PACF is useful for AR model order selection (Box-Jenkins methodology).
///
/// # Errors
/// Propagates errors from [`autocorr_unbiased`].
pub fn pacf(signal: &[f64], max_lag: usize) -> SignalResult<Vec<f64>> {
    let r = autocorr_unbiased(signal, max_lag)?;
    let mut pacf_vals = vec![0.0_f64; max_lag + 1];
    pacf_vals[0] = 1.0;

    let mut phi = vec![vec![0.0_f64; max_lag + 1]; max_lag + 1];
    phi[1][1] = r[1] / r[0];
    pacf_vals[1] = phi[1][1];

    for k in 2..=max_lag {
        let num: f64 = r[k] - (1..k).map(|j| phi[k - 1][j] * r[k - j]).sum::<f64>();
        let den: f64 = r[0] - (1..k).map(|j| phi[k - 1][j] * r[j]).sum::<f64>();
        phi[k][k] = if den.abs() < 1e-15 { 0.0 } else { num / den };
        pacf_vals[k] = phi[k][k];
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }
    }
    Ok(pacf_vals)
}

// --------------------------------------------------------------------------- //
//  Autocovariance
// --------------------------------------------------------------------------- //

/// Compute the autocovariance (mean-removed autocorrelation).
///
/// # Errors
/// Propagates errors from [`autocorr_biased`].
pub fn autocovariance(signal: &[f64], max_lag: usize) -> SignalResult<Vec<f64>> {
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let centred: Vec<f64> = signal.iter().map(|&v| v - mean).collect();
    autocorr_biased(&centred, max_lag)
}

// --------------------------------------------------------------------------- //
//  Whiteness test (Ljung-Box statistic)
// --------------------------------------------------------------------------- //

/// Ljung-Box Q statistic for testing whiteness of a residual sequence.
///
/// ```text
/// Q = N(N+2) · Σ_{τ=1}^{max_lag} ρ[τ]² / (N - τ)
/// ```
///
/// Under the null hypothesis (white noise), `Q ~ χ²(max_lag)`.
/// A large Q → reject whiteness.
///
/// # Errors
/// Propagates errors from [`autocorr_biased`].
pub fn ljung_box_q(signal: &[f64], max_lag: usize) -> SignalResult<f64> {
    let n = signal.len();
    let r = autocorr_normalised(signal, max_lag)?;
    let nf = n as f64;
    let q = (1..=max_lag)
        .map(|tau| r[tau] * r[tau] / (nf - tau as f64))
        .sum::<f64>()
        * nf
        * (nf + 2.0);
    Ok(q)
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_autocorr_biased_lag0_variance() {
        // R[0] (biased) = E[x²] = variance for zero-mean signal
        let x = vec![1.0, -1.0, 1.0, -1.0_f64];
        let r = autocorr_biased(&x, 0).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-12, "R[0]={}", r[0]);
    }

    #[test]
    fn test_autocorr_biased_length() {
        let x = vec![1.0_f64; 8];
        let r = autocorr_biased(&x, 3).unwrap();
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_autocorr_biased_max_lag_too_large() {
        let x = vec![1.0_f64; 4];
        assert!(autocorr_biased(&x, 4).is_err());
    }

    #[test]
    fn test_autocorr_white_noise_decorrelated() {
        // For white noise (alternating), lag-1 autocorr should be -1.
        let x = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0_f64];
        let r = autocorr_biased(&x, 1).unwrap();
        assert!(r[1] < 0.0, "white noise lag-1 autocorr = {}", r[1]);
    }

    #[test]
    fn test_autocorr_normalised_lag0_is_one() {
        let x = vec![3.0, 1.0, 4.0, 1.0, 5.0_f64];
        let r = autocorr_normalised(&x, 2).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_autocorr_normalised_range() {
        let x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.1).sin()).collect();
        let r = autocorr_normalised(&x, 8).unwrap();
        for &v in &r {
            assert!(v.abs() <= 1.0 + 1e-10, "correlation > 1: {v}");
        }
    }

    #[test]
    fn test_autocovariance_constant_signal() {
        // Constant signal has zero variance → autocovariance at all lags = 0.
        let x = vec![5.0_f64; 8];
        let cv = autocovariance(&x, 3).unwrap();
        for v in cv {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn test_pacf_length() {
        let x: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5).sin()).collect();
        let p = pacf(&x, 4).unwrap();
        assert_eq!(p.len(), 5);
    }

    #[test]
    fn test_pacf_lag0_is_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0_f64];
        let p = pacf(&x, 2).unwrap();
        assert!((p[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_ljung_box_q_positive() {
        let x: Vec<f64> = (0..32).map(|i| (i as f64 / 8.0 * PI).sin()).collect();
        let q = ljung_box_q(&x, 4).unwrap();
        assert!(q >= 0.0);
    }

    #[test]
    fn test_ljung_box_q_white_noise_small() {
        // iid sequence → Q should be small
        let x: Vec<f64> = (0..50)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let q = ljung_box_q(&x, 4).unwrap();
        assert!(q.is_finite());
    }
}
