//! Cross-correlation and convolution between two signals.
//!
//! The cross-correlation of signals `x` and `y` at lag `τ` is:
//!
//! ```text
//! R_xy[τ] = Σ_n x[n] · y[n + τ]
//! ```
//!
//! This differs from convolution by the absence of time reversal.
//!
//! ## GPU strategy
//!
//! For long signals: FFT-based cross-correlation in O(N log N):
//! ```text
//! R_xy = IFFT(FFT(x)* · FFT(y))
//! ```
//! where `FFT(x)*` is the complex conjugate of FFT(x).

use crate::error::{SignalError, SignalResult};

// --------------------------------------------------------------------------- //
//  CPU reference: direct cross-correlation
// --------------------------------------------------------------------------- //

/// Compute the cross-correlation of `x` and `y` for lags `-(max_lag)` to `max_lag`.
///
/// Positive lag τ: `R_xy[τ] = Σ x[n] · y[n + τ]`.
/// Negative lag τ: `R_xy[-τ] = Σ x[n + τ] · y[n]`.
///
/// Returns a vector of length `2 * max_lag + 1`.  Element `i` corresponds
/// to lag `τ = i - max_lag`.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `max_lag >= min(x.len(), y.len())`.
pub fn crosscorr(x: &[f64], y: &[f64], max_lag: usize) -> SignalResult<Vec<f64>> {
    let nx = x.len();
    let ny = y.len();
    if max_lag >= nx || max_lag >= ny {
        return Err(SignalError::InvalidSize(format!(
            "max_lag ({max_lag}) must be < min(x.len(), y.len()) = {}",
            nx.min(ny)
        )));
    }
    let n_out = 2 * max_lag + 1;
    let mut out = vec![0.0_f64; n_out];
    for (i, lag) in (-(max_lag as isize)..=(max_lag as isize)).enumerate() {
        let mut sum = 0.0_f64;
        for (n, &xn) in x.iter().enumerate().take(nx) {
            let m = n as isize + lag;
            if m >= 0 && (m as usize) < ny {
                sum += xn * y[m as usize];
            }
        }
        out[i] = sum;
    }
    Ok(out)
}

/// Normalised cross-correlation coefficient in [-1, 1]:
/// ```text
/// ρ_xy[τ] = R_xy[τ] / √(R_xx[0] · R_yy[0])
/// ```
///
/// # Errors
/// Propagates errors from [`crosscorr`].
pub fn crosscorr_normalised(x: &[f64], y: &[f64], max_lag: usize) -> SignalResult<Vec<f64>> {
    let rxy = crosscorr(x, y, max_lag)?;
    let rxx: f64 = x.iter().map(|v| v * v).sum();
    let ryy: f64 = y.iter().map(|v| v * v).sum();
    let denom = (rxx * ryy).sqrt();
    if denom < 1e-15 {
        return Ok(vec![0.0; rxy.len()]);
    }
    Ok(rxy.iter().map(|&v| v / denom).collect())
}

// --------------------------------------------------------------------------- //
//  Linear convolution (direct form)
// --------------------------------------------------------------------------- //

/// Compute the linear convolution of `x` and `h` (full output).
///
/// Output length = `x.len() + h.len() - 1`.
#[must_use]
pub fn convolve(x: &[f64], h: &[f64]) -> Vec<f64> {
    let nx = x.len();
    let nh = h.len();
    let n_out = nx + nh - 1;
    let mut out = vec![0.0_f64; n_out];
    for i in 0..nx {
        for j in 0..nh {
            out[i + j] += x[i] * h[j];
        }
    }
    out
}

/// Compute the circular convolution of `x` and `h` (both length `n`).
///
/// # Errors
/// Returns `SignalError::DimensionMismatch` if lengths differ.
pub fn convolve_circular(x: &[f64], h: &[f64]) -> SignalResult<Vec<f64>> {
    if x.len() != h.len() {
        return Err(SignalError::DimensionMismatch {
            expected: format!("h.len() = {}", x.len()),
            got: format!("{}", h.len()),
        });
    }
    let n = x.len();
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            out[(i + j) % n] += x[i] * h[j];
        }
    }
    Ok(out)
}

// --------------------------------------------------------------------------- //
//  Lag-finding utilities
// --------------------------------------------------------------------------- //

/// Find the lag at which the cross-correlation achieves its maximum.
///
/// Returns `(lag, peak_value)` where lag is in `[-max_lag, max_lag]`.
///
/// # Errors
/// Propagates errors from [`crosscorr`].
pub fn find_delay(x: &[f64], y: &[f64], max_lag: usize) -> SignalResult<(isize, f64)> {
    let r = crosscorr(x, y, max_lag)?;
    let (idx, &peak) = r
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| {
            SignalError::InvalidSize(
                "cross-correlation output is empty (max_lag check failed)".to_owned(),
            )
        })?;
    let lag = idx as isize - max_lag as isize;
    Ok((lag, peak))
}

/// Cross-power spectral density (CPSD): estimate the time delay between two
/// signals using the Generalised Cross-Correlation with Phase Transform (GCC-PHAT).
///
/// Returns the estimated lag in samples.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if signals are too short.
pub fn gcc_phat(x: &[f64], y: &[f64]) -> SignalResult<isize> {
    use std::f64::consts::PI;
    let n = x.len().min(y.len());
    if n < 2 {
        return Err(SignalError::InvalidSize(
            "GCC-PHAT requires ≥ 2 samples".to_owned(),
        ));
    }
    // Compute DFT of x and y (O(N²) for correctness).
    let n_half = n / 2 + 1;
    let mut gxx = vec![(0.0_f64, 0.0_f64); n_half];
    let mut gyy = vec![(0.0_f64, 0.0_f64); n_half];
    for k in 0..n_half {
        for m in 0..n {
            let angle = -2.0 * PI * k as f64 * m as f64 / n as f64;
            let (c, s) = (angle.cos(), angle.sin());
            gxx[k].0 += x[m] * c;
            gxx[k].1 += x[m] * s;
            gyy[k].0 += y[m] * c;
            gyy[k].1 += y[m] * s;
        }
    }
    // PHAT weighting: normalise cross-spectrum by its magnitude.
    let mut gcorr = vec![0.0_f64; n];
    for k in 0..n_half {
        // Cross-spectrum: X* · Y
        let re = gxx[k].0 * gyy[k].0 + gxx[k].1 * gyy[k].1;
        let im = gxx[k].0 * gyy[k].1 - gxx[k].1 * gyy[k].0;
        let mag = (re * re + im * im).sqrt();
        let (re_n, im_n) = if mag > 1e-15 {
            (re / mag, im / mag)
        } else {
            (0.0, 0.0)
        };
        // Inverse DFT contribution.
        for (m, gc) in gcorr.iter_mut().enumerate().take(n) {
            let angle = 2.0 * PI * k as f64 * m as f64 / n as f64;
            *gc += re_n * angle.cos() - im_n * angle.sin();
        }
    }
    // Find peak (excluding conjugate half: look only in [-N/2, N/2]).
    let half = n / 2;
    let mut best_lag = 0isize;
    let mut best_val = f64::NEG_INFINITY;
    for (m, &gc) in gcorr.iter().enumerate().take(n) {
        let lag = if m <= half {
            m as isize
        } else {
            m as isize - n as isize
        };
        if gc > best_val {
            best_val = gc;
            best_lag = lag;
        }
    }
    Ok(best_lag)
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crosscorr_self_is_autocorr() {
        let x = vec![1.0, 2.0, 3.0_f64];
        let r = crosscorr(&x, &x, 2).unwrap();
        // R_xx[0] at centre
        assert!((r[2] - (1.0 + 4.0 + 9.0)).abs() < 1e-10);
    }

    #[test]
    fn test_crosscorr_length() {
        let x = vec![1.0_f64; 5];
        let y = vec![1.0_f64; 5];
        let r = crosscorr(&x, &y, 2).unwrap();
        assert_eq!(r.len(), 5); // 2*2+1
    }

    #[test]
    fn test_crosscorr_lag_too_large() {
        let x = vec![1.0_f64; 3];
        let y = vec![1.0_f64; 3];
        assert!(crosscorr(&x, &y, 3).is_err());
    }

    #[test]
    fn test_crosscorr_normalised_lag0_self() {
        // Self cross-correlation at lag 0 should be 1.
        let x = vec![1.0, 2.0, 3.0_f64];
        let r = crosscorr_normalised(&x, &x, 2).unwrap();
        assert!((r[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_crosscorr_shifted_signal() {
        // y = x shifted by 2 samples: peak should be at lag 2.
        let x = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0_f64];
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0_f64];
        let (lag, _) = find_delay(&x, &y, 4).unwrap();
        assert_eq!(lag, 2, "expected lag=2, got {lag}");
    }

    #[test]
    fn test_convolve_identity() {
        // Convolving with [1] (identity) returns the input.
        let x = vec![1.0, 2.0, 3.0_f64];
        let h = vec![1.0_f64];
        let y = convolve(&x, &h);
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_convolve_output_length() {
        let x = vec![1.0_f64; 5];
        let h = vec![1.0_f64; 3];
        let y = convolve(&x, &h);
        assert_eq!(y.len(), 7);
    }

    #[test]
    fn test_convolve_circular_shape_mismatch() {
        let x = vec![1.0_f64; 4];
        let h = vec![1.0_f64; 3];
        assert!(convolve_circular(&x, &h).is_err());
    }

    #[test]
    fn test_convolve_circular_identity() {
        let n = 4usize;
        let x = vec![1.0, 2.0, 3.0, 4.0_f64];
        let mut h = vec![0.0_f64; n];
        h[0] = 1.0; // impulse
        let y = convolve_circular(&x, &h).unwrap();
        assert_eq!(y, x);
    }

    #[test]
    fn test_gcc_phat_no_delay() {
        // If x == y, delay should be 0.
        let x: Vec<f64> = (0..8).map(|i| (i as f64).sin()).collect();
        let lag = gcc_phat(&x, &x).unwrap();
        assert_eq!(lag, 0);
    }
}
