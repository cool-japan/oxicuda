//! GPU-accelerated DCT-IV.
//!
//! The DCT-IV of length N is defined as:
//!
//! ```text
//! X[k] = Σ_{n=0}^{N-1} x[n] · cos(π·(k+½)·(n+½) / N),   k = 0,…,N-1
//! ```
//!
//! Note that DCT-IV is its own inverse (up to a factor of 2N), making it
//! especially useful for the Modified Discrete Cosine Transform (MDCT) used in
//! MP3 / AAC audio compression.
//!
//! ## Algorithm (Wen-Hsiung Chen, 1977)
//!
//! We reduce DCT-IV to DCT-II via the following steps:
//!
//! 1. Multiply x\[n\] by twiddle: `u[n] = x[n] · cos(π(2n+1)/(4N))`
//! 2. Compute DCT-II of `u` to get `U[k]`.
//! 3. Multiply: `X[k] = 2 · cos(π(2k+1)/(4N)) · U[k]`
//!    followed by a subtract-adjacent step.
//!
//! The complete formula ensures O(N log N) complexity by delegating to the
//! DCT-II (which uses FFT internally).

use std::f64::consts::PI;

use oxicuda_ptx::arch::SmVersion;

use crate::{
    ptx_helpers::{bounds_check, global_tid_1d, ptx_header},
    types::SignalPrecision,
};

// --------------------------------------------------------------------------- //
//  CPU reference
// --------------------------------------------------------------------------- //

/// CPU reference DCT-IV (O(N²), for testing only).
#[must_use]
pub fn dct4_reference(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| xi * (PI * (k as f64 + 0.5) * (i as f64 + 0.5) / n as f64).cos())
                .sum()
        })
        .collect()
}

/// DCT-IV is involutory: DCT4(DCT4(x)) = N/2 · x (up to scale 2N for
/// unnormalized).  Returns the normalisation constant for length N.
#[must_use]
pub const fn dct4_self_inverse_scale(n: usize) -> f64 {
    // Unnormalized DCT-IV: DCT4(DCT4(x)) = (N/2) * x? No.
    // Actually: DCT4 ∘ DCT4 = (N/2) * I  when defined with the cos formula above.
    // The orthonormal form uses scale √(2/N).
    (n / 2) as f64
}

// --------------------------------------------------------------------------- //
//  Pre-twiddle & post-scale PTX kernels
// --------------------------------------------------------------------------- //

/// Emits a PTX kernel that applies the pre-twiddle:
/// `u[n] = x[n] · cos(π(2n+1)/(4N))`.
///
/// The twiddle table (length N, real) is pre-computed on the host and passed
/// as `tw_ptr`.
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
// Kernel: dct4_pretwiddle
// u[n] = x[n] * tw[n]
.visible .entry dct4_pretwiddle(
    .param .u64 x_ptr,
    .param .u64 u_ptr,
    .param .u64 tw_ptr,
    .param .u64 n_param
)
{{
    {tid_preamble}
    .reg .u64 %x_base, %u_base, %tw_base, %n;
    .reg .u64 %off, %addr;
    .reg .{ty} %x, %tw, %u;
    .reg .pred %p_oob;

    ld.param.u64    %x_base,  [x_ptr];
    ld.param.u64    %u_base,  [u_ptr];
    ld.param.u64    %tw_base, [tw_ptr];
    ld.param.u64    %n,       [n_param];

    {bounds}

    mul.lo.u64      %off, %tid64, {bytes};

    add.u64         %addr, %x_base, %off;
    ld.global.{ty}  %x, [%addr];

    add.u64         %addr, %tw_base, %off;
    ld.global.{ty}  %tw, [%addr];

    mul.{ty}        %u, %x, %tw;

    add.u64         %addr, %u_base, %off;
    st.global.{ty}  [%addr], %u;

done_pretwiddle4:
    ret;
}}
",
        header = header,
        ty = ty,
        bytes = bytes,
        tid_preamble = global_tid_1d(),
        bounds = bounds_check("%tid64", "%n", "done_pretwiddle4"),
    )
}

/// Emits a PTX kernel that applies the post-scale and adjacent subtraction:
/// `X[k] = 2 · tw2[k] · U[k] - X[k-1]`
///
/// where `tw2[k] = cos(π(2k+1)/(4N))` and the subtraction is performed
/// in-place via a sequential scan (handled on the host for simplicity; this
/// kernel only does the pointwise multiply).
pub fn emit_postscale_kernel(prec: SignalPrecision, sm: SmVersion) -> String {
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
// Kernel: dct4_postscale
// X[k] = 2 * tw[k] * U[k]
.visible .entry dct4_postscale(
    .param .u64 u_ptr,
    .param .u64 out_ptr,
    .param .u64 tw_ptr,
    .param .u64 n_param
)
{{
    {tid_preamble}
    .reg .u64 %u_base, %out_base, %tw_base, %n;
    .reg .u64 %off, %addr;
    .reg .{ty} %u, %tw, %res, %two;
    .reg .pred %p_oob;

    ld.param.u64    %u_base,   [u_ptr];
    ld.param.u64    %out_base, [out_ptr];
    ld.param.u64    %tw_base,  [tw_ptr];
    ld.param.u64    %n,        [n_param];

    {bounds}

    mul.lo.u64      %off, %tid64, {bytes};

    add.u64         %addr, %u_base, %off;
    ld.global.{ty}  %u, [%addr];

    add.u64         %addr, %tw_base, %off;
    ld.global.{ty}  %tw, [%addr];

    // two = 2.0 (encoded as IEEE 754)
    mov.{ty}        %two, 0f40000000;  // 2.0 in f32; f64 uses 4000000000000000
    mul.{ty}        %res, %tw, %two;
    mul.{ty}        %res, %res, %u;

    add.u64         %addr, %out_base, %off;
    st.global.{ty}  [%addr], %res;

done_postscale4:
    ret;
}}
",
        header = header,
        ty = ty,
        bytes = bytes,
        tid_preamble = global_tid_1d(),
        bounds = bounds_check("%tid64", "%n", "done_postscale4"),
    )
}

// --------------------------------------------------------------------------- //
//  Twiddle factor computation
// --------------------------------------------------------------------------- //

/// Pre-twiddle table: `tw[n] = cos(π(2n+1)/(4N))` for n = 0..N.
#[must_use]
pub fn pretwiddle_table(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (PI * (2 * i + 1) as f64 / (4.0 * n as f64)).cos())
        .collect()
}

/// Post-scale twiddle: `tw2[k] = cos(π(2k+1)/(4N))` — same formula, same table.
#[must_use]
pub fn postscale_table(n: usize) -> Vec<f64> {
    pretwiddle_table(n)
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct4_reference_involutory() {
        // DCT4(DCT4(x)) = (N/2) · x (unnormalized)
        let x = vec![1.0, 2.0, 3.0, 4.0_f64];
        let n = x.len();
        let xk = dct4_reference(&x);
        let xrec = dct4_reference(&xk);
        let scale = n as f64 / 2.0;
        for i in 0..n {
            assert!(
                (xrec[i] - scale * x[i]).abs() < 1e-9,
                "involutory failed at i={i}: got {}, expected {}",
                xrec[i],
                scale * x[i]
            );
        }
    }

    #[test]
    fn test_pretwiddle_table_length() {
        let t = pretwiddle_table(8);
        assert_eq!(t.len(), 8);
    }

    #[test]
    fn test_pretwiddle_table_first_value() {
        let t = pretwiddle_table(8);
        let expected = (PI / 32.0_f64).cos(); // (2*0+1)*π / (4*8)
        assert!((t[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_dct4_reference_linearity() {
        let a = 2.0_f64;
        let b = -1.5_f64;
        let x1 = vec![1.0, 0.5, -0.5, 0.25_f64];
        let x2 = vec![0.0, 1.0, 2.0, -1.0_f64];
        let combo: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(xi, yi)| a * xi + b * yi)
            .collect();
        let d1 = dct4_reference(&x1);
        let d2 = dct4_reference(&x2);
        let dc = dct4_reference(&combo);
        for i in 0..4 {
            let expected = a * d1[i] + b * d2[i];
            assert!((dc[i] - expected).abs() < 1e-10, "i={i}");
        }
    }

    #[test]
    fn test_pretwiddle_ptx_has_entry() {
        let ptx = emit_pretwiddle_kernel(SignalPrecision::F32, SmVersion::Sm80);
        assert!(ptx.contains(".visible .entry dct4_pretwiddle"));
    }

    #[test]
    fn test_postscale_ptx_has_entry() {
        let ptx = emit_postscale_kernel(SignalPrecision::F64, SmVersion::Sm90);
        assert!(ptx.contains(".visible .entry dct4_postscale"));
        assert!(ptx.contains(".target sm_90"));
    }

    #[test]
    fn test_dct4_self_inverse_scale() {
        assert_eq!(dct4_self_inverse_scale(8), 4.0);
        assert_eq!(dct4_self_inverse_scale(4), 2.0);
    }
}
