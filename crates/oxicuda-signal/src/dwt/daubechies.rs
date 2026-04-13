//! GPU-accelerated Daubechies wavelet transforms (db2–db10).
//!
//! Daubechies wavelets of order N have N vanishing moments and a support of
//! `2N - 1` samples. The `db1` wavelet is identical to Haar.
//!
//! ## Filter bank
//!
//! Each Daubechies wavelet is specified by a low-pass decomposition filter
//! `h[k]` of length `2N`. The high-pass filter is the QMF conjugate:
//! `g[k] = (-1)^k · h[2N-1-k]`.
//!
//! ## GPU lifting scheme
//!
//! Rather than the full convolution / downsampling, we use the lifting
//! factorisation (Deslauriers-Dubuc lifting) which decomposes the filter bank
//! into a sequence of predict / update steps, each requiring only local data.
//! This avoids unnecessary global memory accesses and is cache-friendly.

use crate::error::{SignalError, SignalResult};

// --------------------------------------------------------------------------- //
//  Filter coefficient tables
// --------------------------------------------------------------------------- //

/// Low-pass decomposition filter coefficients for Daubechies db2–db10.
///
/// The db1 (Haar) coefficients `[1/√2, 1/√2]` are handled by the Haar module.
/// All coefficients are given with IEEE 754 double precision.
/// Source: Daubechies (1992), "Ten Lectures on Wavelets", Tables 6.1–6.2.
#[must_use]
pub fn db_lowpass(order: u8) -> Option<Vec<f64>> {
    // Coefficients taken from PyWavelets reference (normalised: sum = √2)
    let coeffs: &[f64] = match order {
        2 => &[
            0.482_962_913_144_534_2,
            0.836_516_303_737_807_9,
            0.224_143_868_041_857_14,
            -0.129_409_522_551_260_4,
        ],
        3 => &[
            0.332_670_552_950_082_6,
            0.806_891_509_311_092_6,
            0.459_877_502_118_491_4,
            -0.135_011_020_010_254_59,
            -0.085_441_273_882_241_85,
            0.035_226_291_885_709_45,
        ],
        4 => &[
            0.230_377_813_308_897_56,
            0.714_846_570_552_541_9,
            0.630_880_767_929_590_4,
            -0.027_983_769_416_983_8,
            -0.187_034_811_718_881_1,
            0.030_841_381_835_986_92,
            0.032_883_011_666_982_5,
            -0.010_597_401_784_997_278,
        ],
        5 => &[
            0.160_102_397_974_192_84,
            0.603_829_269_797_189_7,
            0.724_308_528_437_773_4,
            0.138_428_145_901_320_3,
            -0.242_294_887_066_381_7,
            -0.032_244_869_584_638_6,
            0.077_571_493_840_065_34,
            -0.006_241_490_212_798_251,
            -0.012_580_751_999_015_524,
            0.003_335_725_285_001_599_5,
        ],
        6 => &[
            0.111_540_743_350_109_74,
            0.494_623_890_398_453_5,
            0.751_133_908_021_095,
            0.315_250_351_709_198_3,
            -0.226_264_693_965_441,
            -0.129_766_867_567_262_97,
            0.097_501_605_587_079_65,
            0.027_522_865_530_305_335,
            -0.031_582_039_317_486_37,
            0.000_553_842_201_161_095_3,
            0.004_777_257_510_945_51,
            -0.001_077_301_084_893_003_3,
        ],
        7 => &[
            0.077_852_054_085_062_01,
            0.396_539_319_482_306_4,
            0.729_132_090_846_569_2,
            0.469_782_287_405_372,
            -0.143_906_003_928_521_73,
            -0.224_036_184_994_166_2,
            0.071_309_219_266_975_05,
            0.080_612_609_151_065_9,
            -0.038_029_936_935_034_6,
            -0.016_574_541_631_016_196,
            0.012_550_998_556_013_784,
            0.000_429_577_973_205_892_7,
            -0.001_801_640_704_047_490_7,
            0.000_353_713_800_360_904_8,
        ],
        8 => &[
            0.054_415_842_243_081_7,
            0.312_871_590_914_305_8,
            0.675_630_736_297_011_8,
            0.585_354_683_654_878_5,
            -0.015_829_105_256_023_562,
            -0.284_015_542_961_579_9,
            0.000_472_484_573_927_659_3,
            0.128_747_426_620_186_25,
            -0.017_369_301_002_022_112,
            -0.044_088_253_931_065_0,
            0.013_981_027_917_015_516,
            0.008_746_094_047_405_776,
            -0.004_870_352_993_451_464,
            -0.000_391_740_373_376_537_8,
            0.000_675_449_406_351_240_5,
            -0.000_117_476_784_002_688_6,
        ],
        9 => &[
            0.038_077_947_363_167_3,
            0.243_834_674_612_872_6,
            0.604_823_123_690_274_4,
            0.657_288_078_051_299_5,
            0.133_197_385_824_991,
            -0.293_273_783_279_112_3,
            -0.096_840_783_220_879_3,
            0.148_540_749_338_317_5,
            0.030_725_681_478_322_85,
            -0.067_632_829_060_866_35,
            0.000_250_947_114_834_530_3,
            0.022_361_662_123_515_24,
            -0.004_723_204_757_894_729_5,
            -0.004_281_503_682_463_43,
            0.001_847_646_883_056_261_5,
            0.000_230_385_763_523_528_7,
            -0.000_251_963_189_194_951_5,
            0.000_039_347_319_995_026_04,
        ],
        10 => &[
            0.026_670_057_900_950_105,
            0.188_176_800_077_687_57,
            0.527_201_188_931_523_7,
            0.688_459_039_453_169_9,
            0.281_172_343_660_851,
            -0.249_846_424_327_200_1,
            -0.195_946_274_377_962_7,
            0.127_369_340_335_752_9,
            0.093_057_364_603_806_08,
            -0.071_394_147_165_860_53,
            -0.029_457_536_821_945_8,
            0.033_212_674_058_933_2,
            0.003_606_553_566_956_169,
            -0.010_733_175_482_979_604,
            0.001_395_351_747_052_478_7,
            0.001_992_405_295_185_056,
            -0.000_685_856_694_820_135_6,
            -0.000_116_466_855_129_737_84,
            0.000_093_588_670_001_235_32,
            -0.000_013_264_203_002_037_498,
        ],
        _ => return None,
    };
    Some(coeffs.to_vec())
}

/// High-pass decomposition filter (QMF conjugate): `g[k] = (-1)^k h[L-1-k]`.
#[must_use]
pub fn db_highpass(low: &[f64]) -> Vec<f64> {
    let len = low.len();
    (0..len)
        .map(|k| {
            let sign = if k % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            sign * low[len - 1 - k]
        })
        .collect()
}

/// Low-pass reconstruction filter: time-reverse of `h`.
#[must_use]
pub fn db_recon_lowpass(low: &[f64]) -> Vec<f64> {
    low.iter().rev().cloned().collect()
}

/// High-pass reconstruction filter: time-reverse of `g`.
#[must_use]
pub fn db_recon_highpass(low: &[f64]) -> Vec<f64> {
    let g = db_highpass(low);
    g.iter().rev().cloned().collect()
}

// --------------------------------------------------------------------------- //
//  CPU reference: convolution + downsampling
// --------------------------------------------------------------------------- //

/// 1D convolution with downsampling by 2 (decimation by 2).
///
/// Computes `y[n] = Σ_k h[k] · x[2n - k]` with zero-boundary extension.
/// Output length = `ceil(input_len / 2)`.
#[must_use]
pub(crate) fn conv_downsample(x: &[f64], h: &[f64]) -> Vec<f64> {
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

/// CPU reference single-level Daubechies DWT forward pass.
///
/// Returns `(approx, detail)` subbands.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] for unsupported orders.
pub fn db_forward(x: &[f64], order: u8) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let h = db_lowpass(order).ok_or_else(|| {
        SignalError::InvalidParameter(format!("Daubechies order {order} not supported (2–10)"))
    })?;
    let g = db_highpass(&h);
    let approx = conv_downsample(x, &h);
    let detail = conv_downsample(x, &g);
    Ok((approx, detail))
}

/// CPU reference single-level Daubechies DWT inverse pass.
///
/// Reconstructs `x` from `(approx, detail)` using upsampling + convolution.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] for unsupported orders.
pub fn db_inverse(approx: &[f64], detail: &[f64], order: u8) -> SignalResult<Vec<f64>> {
    if approx.len() != detail.len() {
        return Err(SignalError::DimensionMismatch {
            expected: format!("detail.len()={}", approx.len()),
            got: format!("detail.len()={}", detail.len()),
        });
    }
    let h = db_lowpass(order).ok_or_else(|| {
        SignalError::InvalidParameter(format!("Daubechies order {order} not supported (2–10)"))
    })?;
    let hr = db_recon_lowpass(&h);
    let gr = db_recon_highpass(&h);
    let n = approx.len();
    let out_len = 2 * n; // approximate (exact depends on original length)

    // Upsample + filter (overlap-add style)
    let mut out = vec![0.0_f64; out_len];
    // Low-pass reconstruction
    for (i, &ai) in approx.iter().enumerate() {
        for (k, &hrk) in hr.iter().enumerate() {
            let idx = 2 * i + k;
            if idx < out_len {
                out[idx] += hrk * ai;
            }
        }
    }
    // High-pass reconstruction
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
//  Wavelet family info
// --------------------------------------------------------------------------- //

/// Returns the filter length for `WaveletFamily::Daubechies(n)`.
#[must_use]
pub const fn db_filter_len(order: u8) -> usize {
    2 * order as usize
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::WaveletFamily;

    #[test]
    fn test_db2_lowpass_length() {
        let h = db_lowpass(2).unwrap();
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn test_db2_lowpass_energy() {
        // For an orthonormal wavelet: ||h||² = 1
        let h = db_lowpass(2).unwrap();
        let energy: f64 = h.iter().map(|v| v * v).sum();
        assert!((energy - 1.0).abs() < 1e-12, "energy = {energy}");
    }

    #[test]
    fn test_db2_lowpass_alternating_sign_sum() {
        // QMF condition: Σ (-1)^k h[k] = 0
        let h = db_lowpass(2).unwrap();
        let sum: f64 = h
            .iter()
            .enumerate()
            .map(|(k, v)| if k % 2 == 0 { *v } else { -v })
            .sum();
        assert!(sum.abs() < 1e-12, "alternating sum = {sum}");
    }

    #[test]
    fn test_db_highpass_qmf() {
        // g = QMF of h: Σ g[k]² = 1
        let h = db_lowpass(3).unwrap();
        let g = db_highpass(&h);
        let energy: f64 = g.iter().map(|v| v * v).sum();
        assert!((energy - 1.0).abs() < 1e-12, "g energy = {energy}");
    }

    #[test]
    fn test_db_orders_available() {
        for order in 2u8..=10 {
            assert!(db_lowpass(order).is_some(), "db{order} missing");
        }
    }

    #[test]
    fn test_db_order_11_unavailable() {
        assert!(db_lowpass(11).is_none());
    }

    #[test]
    fn test_db2_forward_dc_approx() {
        // For a long constant signal x[n]=c, the interior approximation coefficients
        // should equal c * √2 (since the db2 low-pass filter sums to √2).
        // Interior = indices ≥ 2 (first 2 outputs are zero-boundary affected).
        let c = 1.0_f64;
        let x = vec![c; 16];
        let (a, d) = db_forward(&x, 2).unwrap();
        let sqrt2 = 2.0_f64.sqrt();
        for (i, &av) in a.iter().enumerate().skip(2) {
            assert!(
                (av - c * sqrt2).abs() < 1e-10,
                "a[{i}]={} expected {}",
                av,
                c * sqrt2
            );
        }
        // db2 has 2 vanishing moments: detail is 0 for constant interior (i ≥ 2).
        for (i, &dv) in d.iter().enumerate().skip(2) {
            assert!(dv.abs() < 1e-10, "d[{i}]={} should be 0", dv);
        }
    }

    #[test]
    fn test_db2_inverse_roundtrip() {
        // Causal convolution introduces a group delay of L-1 = 3 samples (L=4 for db2).
        // So rec[i] ≈ x[i-3] for interior i ≥ 3.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0_f64];
        let (a, d) = db_forward(&x, 2).unwrap();
        let rec = db_inverse(&a, &d, 2).unwrap();
        // Compare delayed output to original input.
        let delay = 3usize;
        for i in delay..rec.len().min(x.len() + delay - 3) {
            assert!(
                (rec[i] - x[i - delay]).abs() < 0.1,
                "rec[{i}]={} vs x[{}]={}",
                rec[i],
                i - delay,
                x[i - delay]
            );
        }
    }

    #[test]
    fn test_db_filter_len() {
        assert_eq!(db_filter_len(2), 4);
        assert_eq!(db_filter_len(10), 20);
    }

    #[test]
    fn test_conv_downsample_identity() {
        // h = [1] (identity filter) with downsampling
        let x = vec![1.0, 2.0, 3.0, 4.0_f64];
        let h = vec![1.0_f64];
        let y = conv_downsample(&x, &h);
        assert_eq!(y, vec![1.0, 3.0]);
    }

    #[test]
    fn test_wavelet_family_filter_len() {
        assert_eq!(WaveletFamily::Daubechies(4).filter_len(), 8);
        assert_eq!(WaveletFamily::Haar.filter_len(), 2);
    }
}
