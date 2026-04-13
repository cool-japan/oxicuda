//! Radix-8 butterfly PTX generation.
//!
//! Implements the 8-point DFT butterfly, decomposed as:
//!
//! 1. Two interleaved radix-4 butterflies on even/odd indexed elements.
//! 2. Twiddle factor application with 8th roots of unity.
//! 3. Final combination stage.
//!
//! The 8th roots of unity `W_8^k` for `k = 0..7` have special forms
//! that allow optimised computation (e.g. `W_8^0 = 1`, `W_8^2 = -j`).
#![allow(dead_code)]

use oxicuda_ptx::builder::BodyBuilder;

use crate::ptx_helpers::{ComplexRegs, complex_add, complex_mul, complex_sub, load_twiddle_imm};
use crate::types::FftPrecision;

// ---------------------------------------------------------------------------
// Pre-computed W_8 twiddle factors
// ---------------------------------------------------------------------------

/// Twiddle values for 8th roots of unity: `W_8^k = exp(-2*pi*i*k/8)`.
///
/// ```text
///   W_8^0 = ( 1,        0       )
///   W_8^1 = ( sqrt(2)/2, -sqrt(2)/2 )  [forward]
///   W_8^2 = ( 0,       -1       )
///   W_8^3 = (-sqrt(2)/2, -sqrt(2)/2 )
///   W_8^4 = (-1,        0       )
///   W_8^5 = (-sqrt(2)/2,  sqrt(2)/2 )
///   W_8^6 = ( 0,        1       )
///   W_8^7 = ( sqrt(2)/2,  sqrt(2)/2 )
/// ```
const SQRT2_OVER_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Returns the (cos, sin) pair for `W_8^k` in the forward direction.
fn twiddle_8_forward(k: u32) -> (f64, f64) {
    match k % 8 {
        0 => (1.0, 0.0),
        1 => (SQRT2_OVER_2, -SQRT2_OVER_2),
        2 => (0.0, -1.0),
        3 => (-SQRT2_OVER_2, -SQRT2_OVER_2),
        4 => (-1.0, 0.0),
        5 => (-SQRT2_OVER_2, SQRT2_OVER_2),
        6 => (0.0, 1.0),
        7 => (SQRT2_OVER_2, SQRT2_OVER_2),
        _ => unreachable!(), // k%8 is always 0..7; silence the match-must-be-exhaustive lint
    }
}

// ---------------------------------------------------------------------------
// Radix-8 butterfly (trivial twiddle)
// ---------------------------------------------------------------------------

/// Emits a radix-8 butterfly with trivial outer twiddle factors.
///
/// This performs the full 8-point DFT using the decimation-in-frequency
/// decomposition:
///
/// 1. Split into even/odd halves.
/// 2. Apply 4-point DFT to each half.
/// 3. Combine with W_8 twiddles.
pub(crate) fn emit_radix8_butterfly_trivial(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    inputs: &[ComplexRegs; 8],
    direction_sign: f64,
) -> [ComplexRegs; 8] {
    b.comment("radix-8 butterfly (trivial outer twiddle)");

    // Stage 1: sum/difference pairs
    let mut u = Vec::with_capacity(8);
    for i in 0..4 {
        u.push(complex_add(b, precision, &inputs[i], &inputs[i + 4]));
    }
    let mut v = Vec::with_capacity(4);
    for i in 0..4 {
        v.push(complex_sub(b, precision, &inputs[i], &inputs[i + 4]));
    }

    // Stage 2: apply W_8 twiddles to the difference terms
    let mut v_tw = Vec::with_capacity(4);
    for (i, vi) in v.iter().enumerate() {
        if i == 0 {
            // W_8^0 = 1, no multiplication needed
            v_tw.push(vi.clone());
        } else {
            #[allow(clippy::cast_possible_truncation)]
            let tw = load_twiddle_imm(b, precision, i as u32, 8, direction_sign);
            v_tw.push(complex_mul(b, precision, vi, &tw));
        }
    }

    // Stage 3: two 4-point DFTs (on u[0..4] and v_tw[0..4])
    // We perform them inline using radix-2 decomposition.

    // 4-point DFT on u: split into pairs
    let u02_p = complex_add(b, precision, &u[0], &u[2]);
    let u02_m = complex_sub(b, precision, &u[0], &u[2]);
    let u13_p = complex_add(b, precision, &u[1], &u[3]);
    let u13_m = complex_sub(b, precision, &u[1], &u[3]);

    // Apply W_4 rotation to u13_m
    let w4_1 = load_twiddle_imm(b, precision, 1, 4, direction_sign);
    let u13_m_rot = complex_mul(b, precision, &u13_m, &w4_1);

    let y0 = complex_add(b, precision, &u02_p, &u13_p);
    let y1 = complex_add(b, precision, &u02_m, &u13_m_rot);
    let y2 = complex_sub(b, precision, &u02_p, &u13_p);
    let y3 = complex_sub(b, precision, &u02_m, &u13_m_rot);

    // 4-point DFT on v_tw
    let v02_p = complex_add(b, precision, &v_tw[0], &v_tw[2]);
    let v02_m = complex_sub(b, precision, &v_tw[0], &v_tw[2]);
    let v13_p = complex_add(b, precision, &v_tw[1], &v_tw[3]);
    let v13_m = complex_sub(b, precision, &v_tw[1], &v_tw[3]);

    let v13_m_rot = complex_mul(b, precision, &v13_m, &w4_1);

    let y4 = complex_add(b, precision, &v02_p, &v13_p);
    let y5 = complex_add(b, precision, &v02_m, &v13_m_rot);
    let y6 = complex_sub(b, precision, &v02_p, &v13_p);
    let y7 = complex_sub(b, precision, &v02_m, &v13_m_rot);

    // Output in bit-reversed order: [y0, y4, y2, y6, y1, y5, y3, y7]
    [y0, y4, y2, y6, y1, y5, y3, y7]
}

// ---------------------------------------------------------------------------
// Radix-8 butterfly with explicit twiddle factors
// ---------------------------------------------------------------------------

/// Emits a radix-8 butterfly with explicit per-element twiddle factors.
///
/// Each input `inputs[i]` is first multiplied by `W_N^(k*i)` before
/// the 8-point DFT.
pub(crate) fn emit_radix8_butterfly(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    inputs: &[ComplexRegs; 8],
    k: u32,
    n: u32,
    direction_sign: f64,
) -> [ComplexRegs; 8] {
    b.comment(&format!("radix-8 butterfly W({k},{n})"));

    // Apply outer twiddle factors
    let mut twiddled: Vec<ComplexRegs> = Vec::with_capacity(8);
    for i in 0..8u32 {
        if i == 0 {
            twiddled.push(inputs[0].clone());
        } else {
            let tw = load_twiddle_imm(b, precision, k * i, n, direction_sign);
            twiddled.push(complex_mul(b, precision, &inputs[i as usize], &tw));
        }
    }

    let arr: [ComplexRegs; 8] = [
        twiddled[0].clone(),
        twiddled[1].clone(),
        twiddled[2].clone(),
        twiddled[3].clone(),
        twiddled[4].clone(),
        twiddled[5].clone(),
        twiddled[6].clone(),
        twiddled[7].clone(),
    ];

    emit_radix8_butterfly_trivial(b, precision, &arr, direction_sign)
}

// ---------------------------------------------------------------------------
// Radix-8 stage for a full array
// ---------------------------------------------------------------------------

/// Emits a complete radix-8 stage operating on the `data` slice.
///
/// The `data` slice length must be divisible by 8.
pub(crate) fn emit_radix8_stage(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    data: &mut [ComplexRegs],
    stride: usize,
    n: u32,
    direction_sign: f64,
) {
    b.comment(&format!("radix-8 stage: N={n}, stride={stride}"));

    let eighth = data.len() / 8;
    for i in 0..eighth {
        let group = i / stride;
        let pos = i % stride;

        #[allow(clippy::cast_possible_truncation)]
        let k = (group * pos) as u32;

        let indices: [usize; 8] = [
            i,
            i + eighth,
            i + 2 * eighth,
            i + 3 * eighth,
            i + 4 * eighth,
            i + 5 * eighth,
            i + 6 * eighth,
            i + 7 * eighth,
        ];

        let inputs: [ComplexRegs; 8] = [
            data[indices[0]].clone(),
            data[indices[1]].clone(),
            data[indices[2]].clone(),
            data[indices[3]].clone(),
            data[indices[4]].clone(),
            data[indices[5]].clone(),
            data[indices[6]].clone(),
            data[indices[7]].clone(),
        ];

        let outputs = if k == 0 {
            emit_radix8_butterfly_trivial(b, precision, &inputs, direction_sign)
        } else {
            emit_radix8_butterfly(b, precision, &inputs, k, n, direction_sign)
        };

        for (idx, out) in indices.iter().zip(outputs.iter()) {
            data[*idx] = out.clone();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twiddle_8_symmetry() {
        // W_8^0 should be (1, 0)
        let (c, s) = twiddle_8_forward(0);
        assert!((c - 1.0).abs() < 1e-12);
        assert!(s.abs() < 1e-12);

        // W_8^4 should be (-1, 0)
        let (c, s) = twiddle_8_forward(4);
        assert!((c + 1.0).abs() < 1e-12);
        assert!(s.abs() < 1e-12);
    }
}
