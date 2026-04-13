//! GPU-accelerated DCT-III — the inverse of DCT-II.
//!
//! The DCT-III of length N is:
//!
//! ```text
//! x[n] = X[0]/2 + Σ_{k=1}^{N-1} X[k] · cos(π·k·(n + ½) / N)
//!       = Re( W[n]^{-1} · IFFT(Y)[n] )
//! ```
//!
//! where `Y[k] = X[k] / 2` for k>0, and `Y[0] = X[0]`.  This is simply the
//! inverse of the DCT-II up to a 2N normalisation factor, achieved by:
//!
//! 1. Pre-multiply each spectral coefficient by the conjugate twiddle
//!    `W*[k] = exp(+jπk/(2N))`.
//! 2. Compute the IFFT of the result.
//! 3. Permute the output back from the even/odd interleaved ordering.

use std::f64::consts::PI;

use oxicuda_ptx::arch::SmVersion;

use crate::{
    ptx_helpers::{bounds_check, global_tid_1d, ptx_header},
    types::SignalPrecision,
};

// --------------------------------------------------------------------------- //
//  CPU reference
// --------------------------------------------------------------------------- //

/// CPU reference DCT-III using the naive O(N²) definition.
///
/// This is the inverse of [`super::dct2::dct2_reference`] up to a factor
/// of `1/(2N)` when the orthonormal convention is applied.
#[must_use]
pub fn dct3_reference(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let sum: f64 = (1..n)
                .map(|k| x[k] * (PI * k as f64 * (i as f64 + 0.5) / n as f64).cos())
                .sum();
            x[0] * 0.5 + sum
        })
        .collect()
}

// --------------------------------------------------------------------------- //
//  PTX: pre-twiddle kernel (spectral domain → complex buffer)
// --------------------------------------------------------------------------- //

/// Emits the PTX kernel that converts DCT-III input coefficients into a
/// complex buffer for IFFT by applying conjugate twiddle factors:
/// ```text
/// buf_re[k] = X[k] · cos(πk/(2N))
/// buf_im[k] = X[k] · sin(πk/(2N))   // positive imaginary for IFFT conjugate
/// ```
/// X\[0\] is halved as per the DCT-III definition.
pub fn emit_pretwiddle_kernel(prec: SignalPrecision, sm: SmVersion) -> String {
    let ty = match prec {
        SignalPrecision::F32 => "f32",
        SignalPrecision::F64 => "f64",
    };
    let bytes = match prec {
        SignalPrecision::F32 => 4u64,
        SignalPrecision::F64 => 8u64,
    };
    let header = ptx_header(sm);
    format!(
        r"{header}
// Kernel: dct3_pretwiddle
// Params: x_ptr (input real, u64), buf_ptr (interleaved complex, u64),
//         tw_ptr (twiddle, interleaved cos/sin, u64), n (u64)
.visible .entry dct3_pretwiddle(
    .param .u64 x_ptr,
    .param .u64 buf_ptr,
    .param .u64 tw_ptr,
    .param .u64 n_param
)
{{
    {tid_preamble}
    .reg .u64 %x_base, %buf_base, %tw_base, %n;
    .reg .u64 %off1, %off2, %addr;
    .reg .{ty} %xk, %cw, %sw, %re_out, %im_out, %half;
    .reg .pred %p_oob, %p_k0;

    ld.param.u64    %x_base,   [x_ptr];
    ld.param.u64    %buf_base, [buf_ptr];
    ld.param.u64    %tw_base,  [tw_ptr];
    ld.param.u64    %n,        [n_param];

    {bounds}

    // Load X[k]
    mul.lo.u64      %off1, %tid64, {bytes};
    add.u64         %addr, %x_base, %off1;
    ld.global.{ty}  %xk, [%addr];

    // Apply X[0] *= 0.5
    setp.eq.u64     %p_k0, %tid64, 0;
    @%p_k0 {{
        mul.{ty}    %half, %xk, 0f3F000000;  // 0.5 in f32
        mov.{ty}    %xk, %half;
    }}

    // Load twiddle cos(πk/2N) and sin(πk/2N)
    mul.lo.u64      %off2, %tid64, {bytes2};
    add.u64         %addr, %tw_base, %off2;
    ld.global.{ty}  %cw, [%addr];
    add.u64         %addr, %addr, {bytes};
    ld.global.{ty}  %sw, [%addr];   // stored as -sin; negate for conjugate

    // buf_re = xk * cw,  buf_im = xk * (-sw) = xk * sin
    mul.{ty}        %re_out, %xk, %cw;
    neg.{ty}        %im_out, %sw;
    mul.{ty}        %im_out, %xk, %im_out;

    // Store interleaved [re, im]
    add.u64         %addr, %buf_base, %off2;
    st.global.{ty}  [%addr], %re_out;
    add.u64         %addr, %addr, {bytes};
    st.global.{ty}  [%addr], %im_out;

done_pretwiddle:
    ret;
}}
",
        header = header,
        ty = ty,
        bytes = bytes,
        bytes2 = bytes * 2,
        tid_preamble = global_tid_1d(),
        bounds = bounds_check("%tid64", "%n", "done_pretwiddle"),
    )
}

// --------------------------------------------------------------------------- //
//  PTX: output un-permute kernel
// --------------------------------------------------------------------------- //

/// Un-permutes the IFFT output back from the DCT-II even/odd ordering.
///
/// The inverse of the DCT-II permutation:
/// ```text
/// n even: x[n]     = y[n/2]
/// n odd:  x[N-1-n] = y[n/2]   →   x[n] = y[(N-1-n)/2]
/// ```
pub fn emit_unpermute_kernel(prec: SignalPrecision, sm: SmVersion) -> String {
    let ty = match prec {
        SignalPrecision::F32 => "f32",
        SignalPrecision::F64 => "f64",
    };
    let bytes = match prec {
        SignalPrecision::F32 => 4u64,
        SignalPrecision::F64 => 8u64,
    };
    let header = ptx_header(sm);
    format!(
        r"{header}
// Kernel: dct3_unpermute
// Params: y_ptr (IFFT output, real part, u64), x_ptr (DCT-III output, u64), n (u64)
.visible .entry dct3_unpermute(
    .param .u64 y_ptr,
    .param .u64 x_ptr,
    .param .u64 n_param
)
{{
    {tid_preamble}
    .reg .u64 %y_base, %x_base, %n;
    .reg .u64 %off, %in_idx, %addr;
    .reg .{ty} %val;
    .reg .pred %p_oob, %p_even;
    .reg .u32  %tid_mod2;
    .reg .u64  %half;

    ld.param.u64    %y_base, [y_ptr];
    ld.param.u64    %x_base, [x_ptr];
    ld.param.u64    %n,      [n_param];

    {bounds}

    // tid_mod2 = tid & 1
    and.b64         %in_idx, %tid64, 1;
    cvt.u32.u64     %tid_mod2, %in_idx;
    setp.eq.u32     %p_even, %tid_mod2, 0;

    // half = tid / 2
    shr.u64         %half, %tid64, 1;

    // even tid n: read from y[n/2] = y[half]
    // odd  tid n: read from y[(N-1-n)/2] = y[(N-1-tid)/2]
    @%p_even mov.u64 %in_idx, %half;
    @!%p_even {{
        sub.u64 %in_idx, %n, 1;
        sub.u64 %in_idx, %in_idx, %tid64;
        shr.u64 %in_idx, %in_idx, 1;
    }}

    // Load y[in_idx]  (real part only; IFFT interleaved → step by 2 scalars)
    // The caller stores the full real IFFT output as stride-2 from interleaved.
    // Convention: y_ptr points to real part buffer (already extracted by caller).
    mul.lo.u64      %off, %in_idx, {bytes};
    add.u64         %addr, %y_base, %off;
    ld.global.{ty}  %val, [%addr];

    // Store to x[tid]
    mul.lo.u64      %off, %tid64, {bytes};
    add.u64         %addr, %x_base, %off;
    st.global.{ty}  [%addr], %val;

done_unpermute:
    ret;
}}
",
        header = header,
        ty = ty,
        bytes = bytes,
        tid_preamble = global_tid_1d(),
        bounds = bounds_check("%tid64", "%n", "done_unpermute"),
    )
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct3_reference_inverse_of_dct2() {
        use crate::dct::dct2::dct2_reference;
        // With the unnormalized convention (no leading factor of 2 in DCT-II),
        // DCT-III(DCT-II(x)) = (N/2) * x.
        let x = vec![1.0, 2.0, -1.0, 0.5_f64];
        let n = x.len();
        let xk = dct2_reference(&x);
        let xrec = dct3_reference(&xk);
        let scale = n as f64 / 2.0;
        for i in 0..n {
            assert!(
                (xrec[i] - scale * x[i]).abs() < 1e-10,
                "round-trip failed at i={i}: got {}, expected {}",
                xrec[i],
                scale * x[i]
            );
        }
    }

    #[test]
    fn test_dct3_reference_dc() {
        // DCT-III of [N, 0, …, 0] should give x[n] = N/2 for all n.
        let n = 8usize;
        let x: Vec<f64> = (0..n)
            .map(|k| if k == 0 { n as f64 } else { 0.0 })
            .collect();
        let out = dct3_reference(&x);
        for val in &out {
            assert!((*val - n as f64 / 2.0).abs() < 1e-10, "got {val}");
        }
    }

    #[test]
    fn test_pretwiddle_ptx_has_entry() {
        let ptx = emit_pretwiddle_kernel(SignalPrecision::F32, SmVersion::Sm80);
        assert!(ptx.contains(".visible .entry dct3_pretwiddle"));
    }

    #[test]
    fn test_unpermute_ptx_has_entry() {
        let ptx = emit_unpermute_kernel(SignalPrecision::F64, SmVersion::Sm90);
        assert!(ptx.contains(".visible .entry dct3_unpermute"));
        assert!(ptx.contains(".target sm_90"));
    }

    #[test]
    fn test_pretwiddle_ptx_sm120() {
        let ptx = emit_pretwiddle_kernel(SignalPrecision::F32, SmVersion::Sm120);
        assert!(ptx.contains("sm_120"));
        assert!(ptx.contains("8.7"));
    }

    #[test]
    fn test_dct3_reference_linearity() {
        let a = 3.0_f64;
        let b = -2.0_f64;
        let x1 = vec![1.0, 0.5, -0.5, 0.25_f64];
        let x2 = vec![0.0, 1.0, 2.0, -1.0_f64];
        let combo: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(a_, b_)| a * a_ + b * b_)
            .collect();
        let d1 = dct3_reference(&x1);
        let d2 = dct3_reference(&x2);
        let dc = dct3_reference(&combo);
        for i in 0..4 {
            let expected = a * d1[i] + b * d2[i];
            assert!((dc[i] - expected).abs() < 1e-10);
        }
    }
}
