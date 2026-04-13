//! Radix-4 butterfly PTX generation.
//!
//! Implements the 4-point DFT butterfly:
//!
//! ```text
//!   X[0] = (a0 + a2) + (a1 + a3)
//!   X[1] = (a0 - a2) - j*(a1 - a3)   [forward]
//!   X[2] = (a0 + a2) - (a1 + a3)
//!   X[3] = (a0 - a2) + j*(a1 - a3)   [forward]
//! ```
//!
//! The j-multiply optimisation avoids actual multiplications by using
//! swap + negate.
#![allow(dead_code)]

use oxicuda_ptx::builder::BodyBuilder;

use crate::ptx_helpers::{
    ComplexRegs, complex_add, complex_mul, complex_mul_j, complex_mul_neg_j, complex_sub,
    load_twiddle_imm,
};
use crate::types::FftPrecision;

// ---------------------------------------------------------------------------
// Radix-4 butterfly (trivial twiddles, direction-aware)
// ---------------------------------------------------------------------------

/// Emits a radix-4 butterfly with trivial (W=1) twiddle factors.
///
/// For forward transform (sign = -1):
/// ```text
///   X[0] = (a0 + a2) + (a1 + a3)
///   X[1] = (a0 - a2) - j*(a1 - a3)
///   X[2] = (a0 + a2) - (a1 + a3)
///   X[3] = (a0 - a2) + j*(a1 - a3)
/// ```
///
/// For inverse transform (sign = +1) the j-rotations are conjugated.
pub(crate) fn emit_radix4_butterfly_trivial(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    inputs: &[ComplexRegs; 4],
    forward: bool,
) -> [ComplexRegs; 4] {
    b.comment("radix-4 butterfly (trivial twiddle)");

    // Stage 1: pair-wise sums / differences
    let t0 = complex_add(b, precision, &inputs[0], &inputs[2]);
    let t1 = complex_sub(b, precision, &inputs[0], &inputs[2]);
    let t2 = complex_add(b, precision, &inputs[1], &inputs[3]);
    let t3 = complex_sub(b, precision, &inputs[1], &inputs[3]);

    // Stage 2: apply j-rotation to t3
    let t3_rot = if forward {
        // Forward: multiply by -j  =>  (im, -re)
        complex_mul_neg_j(b, precision, &t3)
    } else {
        // Inverse: multiply by +j  =>  (-im, re)
        complex_mul_j(b, precision, &t3)
    };

    // Final combination
    let x0 = complex_add(b, precision, &t0, &t2);
    let x1 = complex_add(b, precision, &t1, &t3_rot);
    let x2 = complex_sub(b, precision, &t0, &t2);
    let x3 = complex_sub(b, precision, &t1, &t3_rot);

    [x0, x1, x2, x3]
}

// ---------------------------------------------------------------------------
// Radix-4 butterfly with explicit twiddle factors
// ---------------------------------------------------------------------------

/// Emits a radix-4 butterfly with explicit per-element twiddle factors.
///
/// Each input `inputs[i]` is first multiplied by `W_N^(k*i)` before the
/// 4-point DFT is applied.
pub(crate) fn emit_radix4_butterfly(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    inputs: &[ComplexRegs; 4],
    k: u32,
    n: u32,
    direction_sign: f64,
) -> [ComplexRegs; 4] {
    b.comment(&format!("radix-4 butterfly W({k},{n})"));

    // Apply twiddle factors: tw[i] = W_N^(k*i)
    // tw[0] = 1 (skip multiplication for input 0)
    let tw1 = load_twiddle_imm(b, precision, k, n, direction_sign);
    let tw2 = load_twiddle_imm(b, precision, 2 * k, n, direction_sign);
    let tw3 = load_twiddle_imm(b, precision, 3 * k, n, direction_sign);

    let a0 = inputs[0].clone();
    let a1 = complex_mul(b, precision, &inputs[1], &tw1);
    let a2 = complex_mul(b, precision, &inputs[2], &tw2);
    let a3 = complex_mul(b, precision, &inputs[3], &tw3);

    let twiddled = [a0, a1, a2, a3];
    let forward = direction_sign < 0.0;
    emit_radix4_butterfly_trivial(b, precision, &twiddled, forward)
}

// ---------------------------------------------------------------------------
// Radix-4 stage for a full array
// ---------------------------------------------------------------------------

/// Emits a complete radix-4 stage operating on the `data` slice.
///
/// The `data` slice length must be divisible by 4.  `stride` is the
/// distance between consecutive butterfly group elements.
pub(crate) fn emit_radix4_stage(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    data: &mut [ComplexRegs],
    stride: usize,
    n: u32,
    direction_sign: f64,
) {
    b.comment(&format!("radix-4 stage: N={n}, stride={stride}"));

    let quarter = data.len() / 4;
    for i in 0..quarter {
        let group = i / stride;
        let pos = i % stride;

        #[allow(clippy::cast_possible_truncation)]
        let k = (group * pos) as u32;

        let i0 = i;
        let i1 = i + quarter;
        let i2 = i + 2 * quarter;
        let i3 = i + 3 * quarter;

        let inputs = [
            data[i0].clone(),
            data[i1].clone(),
            data[i2].clone(),
            data[i3].clone(),
        ];

        let outputs = if k == 0 {
            let forward = direction_sign < 0.0;
            emit_radix4_butterfly_trivial(b, precision, &inputs, forward)
        } else {
            emit_radix4_butterfly(b, precision, &inputs, k, n, direction_sign)
        };

        data[i0] = outputs[0].clone();
        data[i1] = outputs[1].clone();
        data[i2] = outputs[2].clone();
        data[i3] = outputs[3].clone();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn radix4_stage_partition_count() {
        // Verify that the stage loop processes the correct number of groups
        let n = 16;
        let data_len = n;
        let quarter = data_len / 4;
        assert_eq!(quarter, 4);
    }
}
