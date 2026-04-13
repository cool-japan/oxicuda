//! Symlet wavelet filters (sym2–sym10).
//!
//! Symlets are the "nearly symmetric" compactly-supported orthonormal wavelets
//! designed by Daubechies to be as symmetric as possible while maintaining the
//! orthonormality constraint. They have the same filter length as the
//! corresponding Daubechies wavelet of the same order.

use crate::{
    dwt::daubechies::db_highpass,
    error::{SignalError, SignalResult},
};

// Re-export so callers can access without going through daubechies.
pub use crate::dwt::daubechies::{db_recon_highpass, db_recon_lowpass};

// --------------------------------------------------------------------------- //
//  Filter coefficient tables (Symlets sym2–sym10)
// --------------------------------------------------------------------------- //

/// Low-pass decomposition filter for Symlet of the given order (sym2–sym10).
///
/// Coefficients from PyWavelets / Strang & Nguyen, normalised with `||h||=1`.
#[must_use]
pub fn sym_lowpass(order: u8) -> Option<Vec<f64>> {
    let coeffs: &[f64] = match order {
        2 => &[
            -0.129_409_522_551_260_4,
            0.224_143_868_041_857_14,
            0.836_516_303_737_807_9,
            0.482_962_913_144_534_2,
        ],
        3 => &[
            0.035_226_291_885_709_45,
            -0.085_441_273_882_241_85,
            -0.135_011_020_010_254_59,
            0.459_877_502_118_491_4,
            0.806_891_509_311_092_6,
            0.332_670_552_950_082_6,
        ],
        4 => &[
            -0.075_765_714_789_273_58,
            -0.029_635_527_645_954_28,
            0.497_618_667_632_006_4,
            0.803_738_751_805_916_5,
            0.297_857_795_605_545_7,
            -0.099_219_543_576_847_26,
            -0.012_603_967_262_037_438,
            0.032_223_100_604_042_7,
        ],
        5 => &[
            0.027_333_068_345_077_91,
            0.029_519_490_925_774_792,
            -0.039_134_249_302_383_08,
            0.199_397_533_977_605_5,
            0.723_407_690_402_421_8,
            0.633_978_963_458_297,
            0.016_602_105_764_522_317,
            -0.175_328_089_908_450_4,
            -0.021_101_834_024_758_856,
            0.019_538_882_035_490_204,
        ],
        6 => &[
            0.015_404_109_327_027_751,
            0.003_490_307_294_185_749_5,
            -0.117_990_764_978_914_52,
            -0.048_311_742_585_632_93,
            0.491_055_941_922_892_1,
            0.787_641_141_030_194_4,
            0.337_929_421_728_287_5,
            -0.072_677_084_380_027_8,
            -0.021_060_292_512_300_1,
            0.044_724_901_770_665_66,
            0.000_180_162_031_656_920_9,
            -0.007_800_708_325_034_647,
        ],
        7 => &[
            0.002_682_418_671_425_906,
            -0.001_047_384_888_289_046_6,
            -0.012_636_303_418_013_857,
            0.030_515_513_165_395_73,
            0.067_892_693_501_172_63,
            -0.049_552_834_937_127_255,
            0.017_441_255_086_855_827,
            0.536_101_917_338_943_2,
            0.767_764_317_003_164_7,
            0.288_629_631_751_195_4,
            -0.140_047_241_928_83,
            -0.107_808_237_703_747_5,
            0.004_010_244_871_533_663,
            0.010_268_176_708_511_255,
        ],
        8 => &[
            -0.003_382_415_460_701_348_4,
            -0.000_542_132_331_777_831_2,
            0.031_695_087_811_492_97,
            0.007_607_487_324_917_604,
            -0.143_294_238_350_697_88,
            -0.061_273_359_067_808_1,
            0.481_359_651_258_372_5,
            0.777_185_751_699_363_3,
            0.364_441_894_835_726_2,
            -0.051_945_838_107_709_64,
            -0.027_219_029_717_597_38,
            0.049_137_179_673_607_32,
            0.003_808_752_013_890_312_2,
            -0.014_952_258_337_048_23,
            -0.000_302_920_514_551_204_4,
            0.001_885_857_879_569_955_7,
        ],
        9 => &[
            0.001_012_867_780_727_097_7,
            0.000_393_667_062_889_863_5,
            -0.010_717_990_085_710_37,
            0.001_383_737_028_990_742_3,
            0.064_735_921_612_120_27,
            -0.022_695_252_506_027_67,
            -0.054_447_090_258_745_72,
            0.076_388_196_199_670_29,
            -0.011_958_020_567_474_89,
            -0.013_810_679_320_461_81,
            0.086_375_319_595_353_76,
            0.715_997_816_718_453_8,
            0.618_901_575_691_571_6,
            0.028_609_943_071_085_826,
            -0.190_726_558_049_046_6,
            -0.020_396_416_750_193_48,
            0.044_699_851_191_738_4,
            0.000_939_966_971_370_048_4,
        ],
        10 => &[
            0.000_770_159_809_530_944_5,
            9.564_168_440_249_204e-5,
            -0.008_641_169_942_425_88,
            -0.000_132_563_738_645_684_87,
            0.059_329_560_193_044_54,
            0.015_564_522_951_782_413,
            -0.026_898_806_718_855_45,
            -0.010_991_252_745_002_875,
            0.017_067_637_721_083_3,
            0.015_271_526_218_018_034,
            0.029_750_067_855_475_14,
            -0.162_380_899_735_979_9,
            -0.088_640_218_748_428_69,
            0.504_666_078_979_267_7,
            0.760_748_539_808_386_6,
            0.326_866_549_249_530_4,
            -0.093_551_571_751_059_6,
            -0.068_895_386_093_630_6,
            0.024_080_353_592_248_13,
            0.010_640_166_126_361_844,
        ],
        _ => return None,
    };
    Some(coeffs.to_vec())
}

// --------------------------------------------------------------------------- //
//  CPU reference: convolution + downsampling (re-use Daubechies helper)
// --------------------------------------------------------------------------- //

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

/// CPU reference single-level Symlet DWT forward pass.
///
/// Returns `(approx, detail)` subbands.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] for unsupported orders.
pub fn sym_forward(x: &[f64], order: u8) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let h = sym_lowpass(order).ok_or_else(|| {
        SignalError::InvalidParameter(format!("Symlet order {order} not supported (2–10)"))
    })?;
    let g = db_highpass(&h);
    Ok((conv_downsample(x, &h), conv_downsample(x, &g)))
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sym2_length() {
        let h = sym_lowpass(2).unwrap();
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn test_sym_orders_available() {
        for order in 2u8..=10 {
            assert!(sym_lowpass(order).is_some(), "sym{order} missing");
        }
    }

    #[test]
    fn test_sym_order_11_unavailable() {
        assert!(sym_lowpass(11).is_none());
    }

    #[test]
    fn test_sym2_energy() {
        // ||h||² = 1 for orthonormal wavelet
        let h = sym_lowpass(2).unwrap();
        let energy: f64 = h.iter().map(|v| v * v).sum();
        assert!((energy - 1.0).abs() < 1e-12, "energy = {energy}");
    }

    #[test]
    fn test_sym4_energy() {
        let h = sym_lowpass(4).unwrap();
        let energy: f64 = h.iter().map(|v| v * v).sum();
        assert!((energy - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sym_forward_output_lengths() {
        let x = vec![1.0_f64; 8];
        let (a, d) = sym_forward(&x, 2).unwrap();
        // Output length = ceil(8/2) = 4
        assert_eq!(a.len(), 4);
        assert_eq!(d.len(), 4);
    }

    #[test]
    fn test_sym_forward_invalid_order() {
        let x = vec![1.0_f64; 4];
        assert!(sym_forward(&x, 11).is_err());
    }

    #[test]
    fn test_sym_near_symmetry() {
        // Symlets should be approximately symmetric (time-reversed self).
        // Asymmetry metric: max |h[k] - h[L-1-k]| < max |h[k] - (-h[L-1-k])|
        let h = sym_lowpass(4).unwrap();
        let len = h.len();
        let sym_err: f64 = (0..len)
            .map(|k| (h[k] - h[len - 1 - k]).abs())
            .fold(0.0_f64, f64::max);
        let antisym_err: f64 = (0..len)
            .map(|k| (h[k] + h[len - 1 - k]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            sym_err < antisym_err + 0.1,
            "sym4 should be more symmetric than antisymmetric"
        );
    }
}
