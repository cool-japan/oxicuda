//! Radix-2 butterfly PTX generation.
//!
//! Implements the 2-point Discrete Fourier Transform (DFT) butterfly:
//!
//! ```text
//!   X[0] = a + b * W
//!   X[1] = a - b * W
//! ```
//!
//! where `W = exp(-2*pi*i*k/N)` is the twiddle factor.
#![allow(dead_code)]

use oxicuda_ptx::builder::BodyBuilder;

use crate::ptx_helpers::{ComplexRegs, complex_add, complex_mul, complex_sub, load_twiddle_imm};
use crate::types::FftPrecision;

// ---------------------------------------------------------------------------
// Radix-2 butterfly without twiddle (stage 0 / twiddle = 1)
// ---------------------------------------------------------------------------

/// Emits a radix-2 butterfly with trivial twiddle factor (W = 1):
///
/// ```text
///   out0 = in0 + in1
///   out1 = in0 - in1
/// ```
pub(crate) fn emit_radix2_butterfly_trivial(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    in0: &ComplexRegs,
    in1: &ComplexRegs,
) -> (ComplexRegs, ComplexRegs) {
    b.comment("radix-2 butterfly (trivial twiddle)");
    let out0 = complex_add(b, precision, in0, in1);
    let out1 = complex_sub(b, precision, in0, in1);
    (out0, out1)
}

// ---------------------------------------------------------------------------
// Radix-2 butterfly with explicit twiddle
// ---------------------------------------------------------------------------

/// Emits a radix-2 butterfly with an explicit twiddle factor:
///
/// ```text
///   t    = in1 * W(k, N)
///   out0 = in0 + t
///   out1 = in0 - t
/// ```
///
/// The twiddle factor `W_N^k = exp(sign * 2*pi*i*k/N)` is loaded as an
/// immediate constant (precomputed at PTX generation time).
pub(crate) fn emit_radix2_butterfly(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    in0: &ComplexRegs,
    in1: &ComplexRegs,
    k: u32,
    n: u32,
    direction_sign: f64,
) -> (ComplexRegs, ComplexRegs) {
    b.comment(&format!("radix-2 butterfly W({k},{n})"));

    // Load twiddle factor as immediate
    let w = load_twiddle_imm(b, precision, k, n, direction_sign);

    // t = in1 * W
    let t = complex_mul(b, precision, in1, &w);

    // out0 = in0 + t
    let out0 = complex_add(b, precision, in0, &t);

    // out1 = in0 - t
    let out1 = complex_sub(b, precision, in0, &t);

    (out0, out1)
}

// ---------------------------------------------------------------------------
// Full radix-2 stage for N elements
// ---------------------------------------------------------------------------

/// Emits a complete radix-2 stage operating on `n` complex elements stored
/// in the `data` slice.  After execution, `data` contains the butterflied
/// results.
///
/// `stride` is the distance between butterfly partners at this stage
/// (i.e., `N / 2` for the first stage, `N / 4` for the second, etc.).
pub(crate) fn emit_radix2_stage(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    data: &mut [ComplexRegs],
    stride: usize,
    n: u32,
    direction_sign: f64,
) {
    b.comment(&format!("radix-2 stage: N={n}, stride={stride}"));

    let half = data.len() / 2;
    for i in 0..half {
        let group = i / stride;
        let pos_in_group = i % stride;

        #[allow(clippy::cast_possible_truncation)]
        let k = (group * pos_in_group) as u32;

        let j = i + half;
        let in0 = data[i].clone();
        let in1 = data[j].clone();

        let (out0, out1) = if k == 0 {
            emit_radix2_butterfly_trivial(b, precision, &in0, &in1)
        } else {
            emit_radix2_butterfly(b, precision, &in0, &in1, k, n, direction_sign)
        };

        data[i] = out0;
        data[j] = out1;
    }
}

// ---------------------------------------------------------------------------
// Utility: number of radix-2 stages for a power-of-2 size
// ---------------------------------------------------------------------------

/// Returns `log2(n)` for a power of 2, or `None` if `n` is not a power of 2.
pub(crate) fn log2_exact(n: usize) -> Option<u32> {
    if n == 0 || (n & (n - 1)) != 0 {
        return None;
    }
    Some(n.trailing_zeros())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn log2_exact_powers() {
        assert_eq!(log2_exact(1), Some(0));
        assert_eq!(log2_exact(2), Some(1));
        assert_eq!(log2_exact(1024), Some(10));
        assert_eq!(log2_exact(0), None);
        assert_eq!(log2_exact(3), None);
    }

    #[test]
    fn log2_exact_more_powers() {
        assert_eq!(log2_exact(4), Some(2));
        assert_eq!(log2_exact(8), Some(3));
        assert_eq!(log2_exact(16), Some(4));
        assert_eq!(log2_exact(32), Some(5));
        assert_eq!(log2_exact(64), Some(6));
        assert_eq!(log2_exact(128), Some(7));
        assert_eq!(log2_exact(256), Some(8));
        assert_eq!(log2_exact(512), Some(9));
        // Non-powers should return None
        assert_eq!(log2_exact(5), None);
        assert_eq!(log2_exact(6), None);
        assert_eq!(log2_exact(7), None);
        assert_eq!(log2_exact(9), None);
        assert_eq!(log2_exact(1023), None);
    }

    /// Verify the radix-2 butterfly formula at the mathematical level.
    ///
    /// For a 2-point DFT with W = exp(-2πi*0/2) = 1:
    ///   X[0] = A + 1*B = A + B
    ///   X[1] = A - 1*B = A - B
    ///
    /// Test with A = (3, 0), B = (1, 0):
    ///   X[0] = (4, 0), X[1] = (2, 0)
    #[test]
    fn radix2_butterfly_trivial_size2_math() {
        let a_re = 3.0_f64;
        let a_im = 0.0_f64;
        let b_re = 1.0_f64;
        let b_im = 0.0_f64;

        // Twiddle W = 1 (trivial, k=0)
        let w_re = 1.0_f64;
        let w_im = 0.0_f64;

        // t = B * W (complex multiply)
        let t_re = b_re * w_re - b_im * w_im;
        let t_im = b_re * w_im + b_im * w_re;

        let out0_re = a_re + t_re;
        let out0_im = a_im + t_im;
        let out1_re = a_re - t_re;
        let out1_im = a_im - t_im;

        assert!((out0_re - 4.0).abs() < 1e-10, "X[0].re = {out0_re}");
        assert!(out0_im.abs() < 1e-10, "X[0].im = {out0_im}");
        assert!((out1_re - 2.0).abs() < 1e-10, "X[1].re = {out1_re}");
        assert!(out1_im.abs() < 1e-10, "X[1].im = {out1_im}");
    }

    /// For a 2-point DFT with equal inputs [1, 1], twiddle W_2^0 = 1:
    ///   X[0] = 1 + 1 = 2
    ///   X[1] = 1 - 1 = 0
    #[test]
    fn radix2_butterfly_equal_inputs() {
        let a_re = 1.0_f64;
        let a_im = 0.0_f64;
        let b_re = 1.0_f64;
        let b_im = 0.0_f64;

        // W = 1
        let t_re = b_re;
        let t_im = b_im;

        let out0_re = a_re + t_re;
        let out0_im = a_im + t_im;
        let out1_re = a_re - t_re;
        let out1_im = a_im - t_im;

        assert!((out0_re - 2.0).abs() < 1e-10, "X[0].re should be 2.0");
        assert!(out0_im.abs() < 1e-10, "X[0].im should be 0.0");
        assert!(out1_re.abs() < 1e-10, "X[1].re should be 0.0");
        assert!(out1_im.abs() < 1e-10, "X[1].im should be 0.0");
    }

    /// Verify the N=4 twiddle factor used in radix-2 stage 1.
    ///
    /// W_4^1 = exp(-2πi/4) = exp(-iπ/2) = (0, -1)
    ///
    /// Test butterfly with A=(1,0), B=(0,1), W=(0,-1):
    ///   t = B * W = (0,1)*(0,-1) = (0*0 - 1*(-1), 0*(-1) + 1*0) = (1, 0)
    ///   X[0] = A + t = (2, 0)
    ///   X[1] = A - t = (0, 0)
    #[test]
    fn radix2_butterfly_quarter_rotation_twiddle() {
        let a_re = 1.0_f64;
        let a_im = 0.0_f64;
        let b_re = 0.0_f64;
        let b_im = 1.0_f64;

        // W_4^1 = (cos(-π/2), sin(-π/2)) = (0, -1)
        let w_re = 0.0_f64;
        let w_im = -1.0_f64;
        let t_re = b_re * w_re - b_im * w_im;
        let t_im = b_re * w_im + b_im * w_re;

        assert!((t_re - 1.0).abs() < 1e-10, "t.re = {t_re}");
        assert!(t_im.abs() < 1e-10, "t.im = {t_im}");

        let out0_re = a_re + t_re;
        let out0_im = a_im + t_im;
        let out1_re = a_re - t_re;
        let out1_im = a_im - t_im;

        assert!((out0_re - 2.0).abs() < 1e-10, "X[0].re = {out0_re}");
        assert!(out0_im.abs() < 1e-10, "X[0].im = {out0_im}");
        assert!(out1_re.abs() < 1e-10, "X[1].re = {out1_re}");
        assert!(out1_im.abs() < 1e-10, "X[1].im = {out1_im}");
    }

    /// Verify that the unit-circle property holds for twiddle factors:
    /// |W_N^k|^2 = |exp(-2πi*k/N)|^2 = 1 for all k, N.
    #[test]
    fn radix2_twiddle_unit_circle_property() {
        let n = 8_u32;
        for k in 0..n {
            let angle = -2.0 * PI * k as f64 / n as f64;
            let w_re = angle.cos();
            let w_im = angle.sin();
            let magnitude_sq = w_re * w_re + w_im * w_im;
            assert!(
                (magnitude_sq - 1.0).abs() < 1e-10,
                "W_{n}^{k} magnitude^2 = {magnitude_sq}, expected 1.0"
            );
        }
    }

    /// Verify the conjugate symmetry of radix-2 twiddle factors:
    /// W_N^{N-k} = conj(W_N^k)
    #[test]
    fn radix2_twiddle_conjugate_symmetry() {
        let n = 8_u32;
        for k in 1..n {
            let angle_k = -2.0 * PI * k as f64 / n as f64;
            let w_k = (angle_k.cos(), angle_k.sin());

            let angle_nk = -2.0 * PI * (n - k) as f64 / n as f64;
            let w_nk = (angle_nk.cos(), angle_nk.sin());

            // conj(W_N^k) should equal W_N^{N-k}
            assert!(
                (w_k.0 - w_nk.0).abs() < 1e-10,
                "W_{n}^{k}.re ({}) != W_{n}^{}.re ({})",
                w_k.0,
                n - k,
                w_nk.0
            );
            assert!(
                (w_k.1 + w_nk.1).abs() < 1e-10,
                "conj(W_{n}^{k}).im ({}) != W_{n}^{}.im ({})",
                -w_k.1,
                n - k,
                w_nk.1
            );
        }
    }

    /// Verify the stage count for radix-2 FFT sizes.
    #[test]
    fn log2_exact_stage_count_for_fft() {
        // An N=8 FFT requires 3 radix-2 stages: 8→4→2→1
        assert_eq!(log2_exact(8), Some(3));
        // An N=16 FFT requires 4 stages
        assert_eq!(log2_exact(16), Some(4));
        // An N=1024 FFT requires 10 stages
        assert_eq!(log2_exact(1024), Some(10));
        // Non-power-of-2 cannot be handled by pure radix-2
        assert_eq!(log2_exact(12), None);
        assert_eq!(log2_exact(100), None);
    }
}
