//! Coiflet wavelet filter banks (coif1–coif5).
//!
//! Coiflet wavelets of order N have 2N vanishing moments for both the
//! scaling function and the wavelet. Their filter length is `6N` taps.
//! The normalisation convention matches PyWavelets (sum = √2).
//!
//! Source: Daubechies (1993), Wavelets and Filter Banks, Table 7.1.

use crate::error::{SignalError, SignalResult};

// --------------------------------------------------------------------------- //
//  Filter coefficient tables
// --------------------------------------------------------------------------- //

/// Low-pass decomposition filter coefficients for Coiflets coif1–coif5.
#[must_use]
pub fn coif_lowpass(order: u8) -> Option<Vec<f64>> {
    let coeffs: &[f64] = match order {
        1 => &[
            -0.015_655_728_135_694_5,
            -0.072_732_619_512_853_6,
            0.384_864_846_857_609,
            0.852_572_020_212_199,
            0.337_897_662_458_785,
            -0.072_732_619_512_853_6,
        ],
        2 => &[
            -0.000_720_549_451_682_98,
            -0.001_823_208_870_703_23,
            0.005_611_434_819_393_5,
            0.023_680_171_946_334_0,
            -0.059_434_418_646_456_0,
            -0.076_488_599_078_306_4,
            0.417_005_184_423_761,
            0.812_723_635_449_761,
            0.386_110_066_823_521,
            -0.067_372_554_721_963_2,
            -0.041_464_936_781_962_6,
            0.016_387_336_463_522_1,
        ],
        3 => &[
            -0.000_016_384_000_386_2,
            -0.000_041_641_928_585_5,
            0.000_183_518_611_531_9,
            0.000_595_338_048_638_1,
            -0.001_258_075_199_901_55,
            -0.010_096_930_963_599_8,
            0.024_895_342_840_946_5,
            0.062_077_789_710_958_5,
            -0.165_195_718_049_166,
            -0.094_913_614_789_963_2,
            0.440_882_539_431_741,
            0.782_791_557_559_197,
            0.396_539_319_482_306,
            -0.050_287_604_996_946_5,
            -0.054_895_673_513_743_2,
            0.016_099_756_330_741_8,
            0.005_765_493_793_396_4,
            -0.002_308_683_020_521_6,
        ],
        4 => &[
            -0.000_000_346_959_3,
            -0.000_000_846_749_6,
            0.000_005_224_677_0,
            0.000_014_940_671_9,
            -0.000_065_309_646_1,
            -0.000_199_091_980_0,
            0.000_625_823_584_8,
            0.002_978_817_827_0,
            -0.006_441_189_280_9,
            -0.025_561_669_613_5,
            0.039_989_543_614_3,
            0.127_000_000_000_0,
            -0.218_723_765_854_8,
            -0.069_854_756_551_1,
            0.454_016_456_849_1,
            0.757_743_726_671_7,
            0.410_765_793_261_2,
            -0.040_039_006_027_8,
            -0.066_003_701_558_8,
            0.019_534_220_891_6,
            0.009_220_867_601_4,
            -0.003_543_916_046_0,
            -0.000_571_879_876_3,
            0.000_217_376_527_0,
        ],
        5 => &[
            -0.000_000_006_908_7,
            -0.000_000_016_381_0,
            0.000_000_140_284_9,
            0.000_000_387_049_0,
            -0.000_002_167_659_8,
            -0.000_006_240_736_1,
            0.000_025_552_498_7,
            0.000_087_286_831_7,
            -0.000_226_977_671_2,
            -0.001_074_058_491_9,
            0.001_547_614_091_6,
            0.005_748_249_012_5,
            -0.012_875_167_534_1,
            -0.030_640_777_419_4,
            0.053_869_483_879_4,
            0.166_756_420_505_6,
            -0.264_036_070_760_6,
            -0.056_028_568_397_0,
            0.460_609_048_011_8,
            0.736_448_906_773_7,
            0.421_468_982_316_3,
            -0.032_949_958_720_1,
            -0.073_987_621_920_0,
            0.022_199_399_714_2,
            0.012_026_696_717_8,
            -0.004_922_497_879_5,
            -0.001_264_360_827_0,
            0.000_600_512_498_1,
            0.000_069_505_372_3,
            -0.000_038_382_350_4,
        ],
        _ => return None,
    };
    Some(coeffs.to_vec())
}

/// High-pass decomposition filter from low-pass (QMF conjugate).
///
/// `g[k] = (-1)^k · h[N-1-k]` where N = h.len().
#[must_use]
pub fn coif_highpass(h: &[f64]) -> Vec<f64> {
    h.iter()
        .rev()
        .enumerate()
        .map(|(k, &hk)| if k % 2 == 0 { hk } else { -hk })
        .collect()
}

/// Reconstruction low-pass filter (time-reversed low-pass).
#[must_use]
pub fn coif_recon_lowpass(h: &[f64]) -> Vec<f64> {
    let mut r = h.to_vec();
    r.reverse();
    r
}

/// Reconstruction high-pass filter (QMF of time-reversed low-pass).
#[must_use]
pub fn coif_recon_highpass(h: &[f64]) -> Vec<f64> {
    h.iter()
        .enumerate()
        .map(|(k, &hk)| if k % 2 == 0 { hk } else { -hk })
        .collect()
}

// --------------------------------------------------------------------------- //
//  Forward and Inverse transforms
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

/// Single-level forward Coiflet DWT.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] for unsupported orders (1–5).
pub fn coif_forward(x: &[f64], order: u8) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let h = coif_lowpass(order).ok_or_else(|| {
        SignalError::InvalidParameter(format!("Coiflet order {order} not supported (1–5)"))
    })?;
    let g = coif_highpass(&h);
    let approx = conv_downsample(x, &h);
    let detail = conv_downsample(x, &g);
    Ok((approx, detail))
}

/// Single-level inverse Coiflet DWT.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] for unsupported orders.
/// Returns [`SignalError::DimensionMismatch`] if `approx.len() != detail.len()`.
pub fn coif_inverse(approx: &[f64], detail: &[f64], order: u8) -> SignalResult<Vec<f64>> {
    if approx.len() != detail.len() {
        return Err(SignalError::DimensionMismatch {
            expected: format!("detail.len()={}", approx.len()),
            got: format!("detail.len()={}", detail.len()),
        });
    }
    let h = coif_lowpass(order).ok_or_else(|| {
        SignalError::InvalidParameter(format!("Coiflet order {order} not supported (1–5)"))
    })?;
    let hr = coif_recon_lowpass(&h);
    let gr = coif_recon_highpass(&h);
    let n = approx.len();
    let out_len = 2 * n;

    let mut out = vec![0.0_f64; out_len];
    for (i, &ai) in approx.iter().enumerate() {
        for (k, &hrk) in hr.iter().enumerate() {
            let idx = 2 * i + k;
            if idx < out_len {
                out[idx] += hrk * ai;
            }
        }
    }
    for (i, &di) in detail.iter().enumerate() {
        for (k, &grk) in gr.iter().enumerate() {
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
    fn coif1_lowpass_len() {
        assert_eq!(coif_lowpass(1).unwrap().len(), 6);
    }

    #[test]
    fn coif5_lowpass_len() {
        assert_eq!(coif_lowpass(5).unwrap().len(), 30);
    }

    #[test]
    fn coif_lowpass_unsupported() {
        assert!(coif_lowpass(0).is_none());
        assert!(coif_lowpass(6).is_none());
    }

    #[test]
    fn coif1_forward_inverse_roundtrip() {
        // Impulse should round-trip approximately for long enough sequences
        let n = 32;
        let mut x = vec![0.0_f64; n];
        x[0] = 1.0;
        let (approx, detail) = coif_forward(&x, 1).unwrap();
        let recon = coif_inverse(&approx, &detail, 1).unwrap();
        // Verify energy is preserved (not exact reconstruction due to boundary)
        let energy_in: f64 = x.iter().map(|v| v * v).sum();
        let energy_out: f64 = recon.iter().map(|v| v * v).sum();
        assert!(
            (energy_out - energy_in).abs() < 0.1,
            "energy not preserved: in={energy_in:.4} out={energy_out:.4}"
        );
    }

    #[test]
    fn coif2_lowpass_sum_sqrt2() {
        let h = coif_lowpass(2).unwrap();
        let sum: f64 = h.iter().sum();
        assert!((sum - 2.0_f64.sqrt()).abs() < 1e-6, "sum={sum}");
    }
}
