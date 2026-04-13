//! Mixed-radix butterfly PTX generation for non-power-of-2 factors.
//!
//! Supports radix 3, 5, and 7 butterflies using the general
//! N-point DFT formula with precomputed twiddle factors.
#![allow(dead_code)]

use oxicuda_ptx::builder::BodyBuilder;

use crate::ptx_helpers::{ComplexRegs, complex_add, complex_mul, load_float_imm, load_twiddle_imm};
use crate::types::FftPrecision;

// ---------------------------------------------------------------------------
// Generic small-N butterfly
// ---------------------------------------------------------------------------

/// Emits a generic N-point DFT butterfly using the direct formula:
///
/// ```text
///   X[k] = sum_{j=0}^{N-1}  x[j] * W_N^{k*j}
/// ```
///
/// This is O(N^2) per butterfly, but for small N (3, 5, 7) the constant
/// factor is acceptable and avoids complex decomposition logic.
pub(crate) fn emit_radix_n_butterfly(
    b: &mut BodyBuilder<'_>,
    n: u32,
    precision: FftPrecision,
    inputs: &[ComplexRegs],
    direction_sign: f64,
) -> Vec<ComplexRegs> {
    b.comment(&format!("radix-{n} butterfly (generic DFT)"));

    let mut outputs = Vec::with_capacity(n as usize);

    for k in 0..n {
        // X[k] = sum_j x[j] * W_N^{k*j}
        // Start with x[0] (W_N^0 = 1 for all k)
        let mut acc = inputs[0].clone();

        for j in 1..n {
            let twiddle_exp = (k * j) % n;
            if twiddle_exp == 0 {
                // W_N^0 = 1, just add
                acc = complex_add(b, precision, &acc, &inputs[j as usize]);
            } else {
                let w = load_twiddle_imm(b, precision, twiddle_exp, n, direction_sign);
                let term = complex_mul(b, precision, &inputs[j as usize], &w);
                acc = complex_add(b, precision, &acc, &term);
            }
        }

        outputs.push(acc);
    }

    outputs
}

// ---------------------------------------------------------------------------
// Radix-3 butterfly (optimised)
// ---------------------------------------------------------------------------

/// Emits an optimised radix-3 butterfly.
///
/// Uses the identity `W_3 = exp(-2*pi*i/3) = -1/2 - i*sqrt(3)/2` to
/// reduce multiplications compared to the generic DFT.
///
/// ```text
///   t1 = x[1] + x[2]
///   t2 = x[1] - x[2]
///   X[0] = x[0] + t1
///   X[1] = x[0] - t1/2 + j*sqrt(3)/2 * t2  (adjusted for direction)
///   X[2] = x[0] - t1/2 - j*sqrt(3)/2 * t2  (adjusted for direction)
/// ```
pub(crate) fn emit_radix3_butterfly(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    inputs: &[ComplexRegs; 3],
    direction_sign: f64,
) -> [ComplexRegs; 3] {
    b.comment("radix-3 butterfly (optimised)");

    let half = load_float_imm(b, precision, 0.5);
    let sqrt3_half = load_float_imm(b, precision, std::f64::consts::FRAC_PI_3.sin());
    // sin(pi/3) = sqrt(3)/2

    // t1 = x[1] + x[2]
    let t1 = complex_add(b, precision, &inputs[1], &inputs[2]);

    // t2 = x[1] - x[2]
    let t2_re =
        crate::ptx_helpers::sub_float(b, precision, inputs[1].re.clone(), inputs[2].re.clone());
    let t2_im =
        crate::ptx_helpers::sub_float(b, precision, inputs[1].im.clone(), inputs[2].im.clone());

    // X[0] = x[0] + t1
    let x0 = complex_add(b, precision, &inputs[0], &t1);

    // half_t1 = t1 * 0.5
    let half_t1_re = crate::ptx_helpers::mul_float(b, precision, t1.re.clone(), half.clone());
    let half_t1_im = crate::ptx_helpers::mul_float(b, precision, t1.im.clone(), half.clone());

    // base = x[0] - half_t1
    let base_re = crate::ptx_helpers::sub_float(b, precision, inputs[0].re.clone(), half_t1_re);
    let base_im = crate::ptx_helpers::sub_float(b, precision, inputs[0].im.clone(), half_t1_im);

    // rot = sqrt3_half * t2 * direction_sign
    // For forward (sign=-1): rot_re = sqrt3_half * t2_im, rot_im = -sqrt3_half * t2_re
    // For inverse (sign=+1): rot_re = -sqrt3_half * t2_im, rot_im = sqrt3_half * t2_re
    let s3h_t2_re = crate::ptx_helpers::mul_float(b, precision, sqrt3_half.clone(), t2_im.clone());
    let s3h_t2_im = crate::ptx_helpers::mul_float(b, precision, sqrt3_half, t2_re);

    let (rot_re, rot_im) = if direction_sign < 0.0 {
        // Forward: rot = (s3h * t2_im, -s3h * t2_re)
        let neg_im = crate::ptx_helpers::neg_float(b, precision, s3h_t2_im);
        (s3h_t2_re, neg_im)
    } else {
        // Inverse: rot = (-s3h * t2_im, s3h * t2_re)
        let neg_re = crate::ptx_helpers::neg_float(b, precision, s3h_t2_re);
        (neg_re, s3h_t2_im)
    };

    // X[1] = base + rot
    let x1_re = crate::ptx_helpers::add_float(b, precision, base_re.clone(), rot_re.clone());
    let x1_im = crate::ptx_helpers::add_float(b, precision, base_im.clone(), rot_im.clone());

    // X[2] = base - rot
    let x2_re = crate::ptx_helpers::sub_float(b, precision, base_re, rot_re);
    let x2_im = crate::ptx_helpers::sub_float(b, precision, base_im, rot_im);

    [
        x0,
        ComplexRegs {
            re: x1_re,
            im: x1_im,
        },
        ComplexRegs {
            re: x2_re,
            im: x2_im,
        },
    ]
}

// ---------------------------------------------------------------------------
// Radix-5 butterfly
// ---------------------------------------------------------------------------

/// Emits a radix-5 butterfly using the generic N-point DFT formula.
///
/// While a fully optimised radix-5 exists (using the pentagonal identity),
/// the generic approach produces correct results with acceptable overhead
/// for the relatively rare case of factor-of-5 FFTs.
pub(crate) fn emit_radix5_butterfly(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    inputs: &[ComplexRegs; 5],
    direction_sign: f64,
) -> Vec<ComplexRegs> {
    emit_radix_n_butterfly(b, 5, precision, inputs.as_slice(), direction_sign)
}

// ---------------------------------------------------------------------------
// Radix-7 butterfly
// ---------------------------------------------------------------------------

/// Emits a radix-7 butterfly using the generic N-point DFT formula.
pub(crate) fn emit_radix7_butterfly(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    inputs: &[ComplexRegs; 7],
    direction_sign: f64,
) -> Vec<ComplexRegs> {
    emit_radix_n_butterfly(b, 7, precision, inputs.as_slice(), direction_sign)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn mixed_radix_module_loads() {
        // Smoke test: the module compiles and functions are accessible.
    }
}
