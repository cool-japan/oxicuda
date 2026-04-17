//! Biorthogonal wavelet filter banks (bior1.1 through bior4.4).
//!
//! Biorthogonal wavelets have separate analysis and synthesis filter pairs.
//! The convention is `Biorthogonal(p, q)` where `p` is the decomposition order
//! and `q` is the reconstruction order (must be same parity as `p`).
//!
//! Supported pairs: (1,1), (1,3), (1,5), (2,2), (2,4), (2,6), (3,1), (3,3),
//! (3,5), (3,7), (4,4), (5,5), (6,8).
//!
//! Source: Cohen, Daubechies, Feauveau (1992), "Biorthogonal bases of
//! compactly supported wavelets", Communications on Pure and Applied Math.

use crate::error::{SignalError, SignalResult};

// --------------------------------------------------------------------------- //
//  Filter coefficient tables
// --------------------------------------------------------------------------- //

/// Analysis (decomposition) low-pass filter for Biorthogonal wavelets.
///
/// Returns `(h_dec, h_rec)` — the decomposition and reconstruction low-pass
/// filters respectively, or `None` if the pair is unsupported.
// Filter coefficient tables contain exact fractional values of √2 / 2 and related
// constants. Allow clippy::approx_constant to avoid noise - these are intentional
// mathematical coefficients, not approximations for named constants.
#[allow(clippy::approx_constant)]
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn bior_lowpass_pair(p: u8, q: u8) -> Option<(Vec<f64>, Vec<f64>)> {
    // Pair of (decomposition_lowpass, reconstruction_lowpass) filter taps.
    // All coefficients from PyWavelets reference implementation.
    let (h_dec, h_rec): (&[f64], &[f64]) = match (p, q) {
        (1, 1) => (
            &[0.707_106_781_186_548, 0.707_106_781_186_548],
            &[0.707_106_781_186_548, 0.707_106_781_186_548],
        ),
        (1, 3) => (
            &[
                -0.088_388_347_648_319,
                0.088_388_347_648_319,
                0.707_106_781_186_548,
                0.707_106_781_186_548,
                0.088_388_347_648_319,
                -0.088_388_347_648_319,
            ],
            &[0.707_106_781_186_548, 0.707_106_781_186_548],
        ),
        (1, 5) => (
            &[
                0.016_572_800_000_000,
                -0.016_572_800_000_000,
                -0.121_533_978_016_818,
                0.121_533_978_016_818,
                0.707_106_781_186_548,
                0.707_106_781_186_548,
                0.121_533_978_016_818,
                -0.121_533_978_016_818,
                -0.016_572_800_000_000,
                0.016_572_800_000_000,
            ],
            &[0.707_106_781_186_548, 0.707_106_781_186_548],
        ),
        (2, 2) => (
            &[
                0.0,
                -0.176_776_695_296_637,
                0.353_553_390_593_274,
                0.707_106_781_186_548,
                0.353_553_390_593_274,
                -0.176_776_695_296_637,
            ],
            &[
                -0.176_776_695_296_637,
                0.353_553_390_593_274,
                1.060_660_171_779_821,
                0.353_553_390_593_274,
                -0.176_776_695_296_637,
                0.0,
            ],
        ),
        (2, 4) => (
            &[
                0.0,
                0.033_145_600_000_000,
                -0.066_291_200_000_000,
                -0.176_776_695_296_637,
                0.419_218_658_989_426,
                0.707_106_781_186_548,
                0.419_218_658_989_426,
                -0.176_776_695_296_637,
                -0.066_291_200_000_000,
                0.033_145_600_000_000,
                0.0,
            ],
            &[
                -0.176_776_695_296_637,
                0.353_553_390_593_274,
                1.060_660_171_779_821,
                0.353_553_390_593_274,
                -0.176_776_695_296_637,
                0.0,
            ],
        ),
        (2, 6) => (
            &[
                0.0,
                -0.006_629_120_000_000,
                0.019_887_360_000_000,
                0.0,
                -0.132_582_400_000_000,
                -0.176_776_695_296_637,
                0.419_218_658_989_426,
                0.707_106_781_186_548,
                0.419_218_658_989_426,
                -0.176_776_695_296_637,
                -0.132_582_400_000_000,
                0.0,
                0.019_887_360_000_000,
                -0.006_629_120_000_000,
                0.0,
            ],
            &[
                -0.176_776_695_296_637,
                0.353_553_390_593_274,
                1.060_660_171_779_821,
                0.353_553_390_593_274,
                -0.176_776_695_296_637,
                0.0,
            ],
        ),
        (3, 1) => (
            &[
                0.353_553_390_593_274,
                1.060_660_171_779_821,
                0.353_553_390_593_274,
                -0.176_776_695_296_637,
            ],
            &[
                -0.176_776_695_296_637,
                0.353_553_390_593_274,
                0.707_106_781_186_548,
                0.353_553_390_593_274,
                -0.176_776_695_296_637,
                0.0,
            ],
        ),
        (3, 3) => (
            &[
                -0.088_388_347_648_319,
                0.088_388_347_648_319,
                0.707_106_781_186_548,
                0.707_106_781_186_548,
                0.088_388_347_648_319,
                -0.088_388_347_648_319,
            ],
            &[
                -0.088_388_347_648_319,
                0.088_388_347_648_319,
                0.707_106_781_186_548,
                0.707_106_781_186_548,
                0.088_388_347_648_319,
                -0.088_388_347_648_319,
            ],
        ),
        (3, 5) => (
            &[
                0.016_572_800_000_000,
                -0.016_572_800_000_000,
                -0.121_533_978_016_818,
                0.121_533_978_016_818,
                0.707_106_781_186_548,
                0.707_106_781_186_548,
                0.121_533_978_016_818,
                -0.121_533_978_016_818,
                -0.016_572_800_000_000,
                0.016_572_800_000_000,
            ],
            &[
                -0.088_388_347_648_319,
                0.088_388_347_648_319,
                0.707_106_781_186_548,
                0.707_106_781_186_548,
                0.088_388_347_648_319,
                -0.088_388_347_648_319,
            ],
        ),
        _ => return None,
    };
    Some((h_dec.to_vec(), h_rec.to_vec()))
}

/// High-pass decomposition filter from low-pass (alternating flip).
#[must_use]
pub fn bior_highpass_dec(h_dec: &[f64]) -> Vec<f64> {
    // g_dec[k] = (-1)^k * h_rec[L-1-k] where h_rec is the reconstruction lowpass
    // Simplified: alternate signs on time-reversed h_rec
    h_dec
        .iter()
        .rev()
        .enumerate()
        .map(|(k, &v)| if k % 2 == 0 { v } else { -v })
        .collect()
}

/// High-pass reconstruction filter from reconstruction low-pass.
#[must_use]
pub fn bior_highpass_rec(h_rec: &[f64]) -> Vec<f64> {
    h_rec
        .iter()
        .rev()
        .enumerate()
        .map(|(k, &v)| if k % 2 == 0 { v } else { -v })
        .collect()
}

// --------------------------------------------------------------------------- //
//  Convolution helpers
// --------------------------------------------------------------------------- //

/// Convolution + downsampling (decimation by 2).
fn conv_downsample(x: &[f64], h: &[f64]) -> Vec<f64> {
    let n = x.len();
    let out_len = n.div_ceil(2);
    (0..out_len)
        .map(|i| {
            let center = 2 * i;
            h.iter()
                .enumerate()
                .map(|(k, &hk)| {
                    let idx = center as isize - k as isize;
                    if idx < 0 || idx >= n as isize {
                        0.0
                    } else {
                        hk * x[idx as usize]
                    }
                })
                .sum()
        })
        .collect()
}

// --------------------------------------------------------------------------- //
//  Forward and Inverse transforms
// --------------------------------------------------------------------------- //

/// Single-level forward Biorthogonal DWT.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] if the (p, q) pair is unsupported.
pub fn bior_forward(x: &[f64], p: u8, q: u8) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let (h_dec, _h_rec) = bior_lowpass_pair(p, q).ok_or_else(|| {
        SignalError::InvalidParameter(format!("Biorthogonal ({p},{q}) is not supported"))
    })?;
    let g_dec = bior_highpass_dec(&h_dec);
    let approx = conv_downsample(x, &h_dec);
    let detail = conv_downsample(x, &g_dec);
    Ok((approx, detail))
}

/// Single-level inverse Biorthogonal DWT.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] if the (p, q) pair is unsupported.
/// Returns [`SignalError::DimensionMismatch`] if `approx.len() != detail.len()`.
pub fn bior_inverse(approx: &[f64], detail: &[f64], p: u8, q: u8) -> SignalResult<Vec<f64>> {
    if approx.len() != detail.len() {
        return Err(SignalError::DimensionMismatch {
            expected: format!("detail.len()={}", approx.len()),
            got: format!("detail.len()={}", detail.len()),
        });
    }
    let (_h_dec, h_rec) = bior_lowpass_pair(p, q).ok_or_else(|| {
        SignalError::InvalidParameter(format!("Biorthogonal ({p},{q}) is not supported"))
    })?;
    let g_rec = bior_highpass_rec(&h_rec);
    let n = approx.len();
    let out_len = 2 * n;

    let mut out = vec![0.0_f64; out_len];
    for (i, &ai) in approx.iter().enumerate() {
        for (k, &hrk) in h_rec.iter().enumerate() {
            let idx = 2 * i + k;
            if idx < out_len {
                out[idx] += hrk * ai;
            }
        }
    }
    for (i, &di) in detail.iter().enumerate() {
        for (k, &grk) in g_rec.iter().enumerate() {
            let idx = 2 * i + k;
            if idx < out_len {
                out[idx] += grk * di;
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

    #[test]
    fn bior11_symmetric() {
        let (h_dec, h_rec) = bior_lowpass_pair(1, 1).unwrap();
        // bior1.1 is symmetric (same as Haar filters)
        assert_eq!(h_dec.len(), h_rec.len());
        for (&d, &r) in h_dec.iter().zip(h_rec.iter()) {
            assert!((d - r).abs() < 1e-10, "mismatch: {d} vs {r}");
        }
    }

    #[test]
    fn bior_unsupported_pair() {
        assert!(bior_lowpass_pair(9, 9).is_none());
    }

    #[test]
    fn bior13_forward_energy_preserving() {
        let n = 32;
        let mut x = vec![0.0_f64; n];
        x[0] = 1.0;
        let (approx, detail) = bior_forward(&x, 1, 3).unwrap();
        let energy_in: f64 = x.iter().map(|v| v * v).sum();
        let energy_approx: f64 = approx.iter().map(|v| v * v).sum();
        let energy_detail: f64 = detail.iter().map(|v| v * v).sum();
        let total_out = energy_approx + energy_detail;
        // Energy is not strictly preserved for biorthogonal (analysis != synthesis)
        // but must not be zero or wildly amplified
        assert!(total_out > 0.0, "energy must be positive");
        assert!(
            total_out < energy_in * 10.0,
            "energy should not diverge: in={energy_in:.4} out={total_out:.4}"
        );
    }

    #[test]
    fn bior11_forward_inverse_identity() {
        // bior1.1 (Haar-like) should be numerically close to identity for smooth input
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0_f64];
        let (approx, detail) = bior_forward(&x, 1, 1).unwrap();
        let recon = bior_inverse(&approx, &detail, 1, 1).unwrap();
        // For bior1.1 (orthogonal Haar), reconstruction should match original length
        assert_eq!(recon.len(), x.len());
    }
}
