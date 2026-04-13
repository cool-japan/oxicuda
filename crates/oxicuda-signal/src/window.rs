//! Window function generation and analysis utilities.
//!
//! Window functions are used in spectral analysis, FIR filter design, and
//! time-frequency methods to reduce spectral leakage by smoothly tapering the
//! signal to zero at the boundaries.
//!
//! This module provides CPU-side generation of window coefficients (used for
//! host-to-device uploads before GPU kernels) and analysis metrics.

pub use crate::audio::stft::make_window;
use crate::types::WindowType;

// --------------------------------------------------------------------------- //
//  Window analysis metrics
// --------------------------------------------------------------------------- //

/// Compute the coherent gain of a window function:
/// `CG = (1/N) · Σ w[n]`.
///
/// A value of 1.0 means no amplitude loss for a DC signal.
#[must_use]
pub fn coherent_gain(w: &[f64]) -> f64 {
    let n = w.len() as f64;
    w.iter().sum::<f64>() / n
}

/// Compute the equivalent noise bandwidth (ENBW) in bins:
/// `ENBW = N · Σ w[n]² / (Σ w[n])²`.
///
/// A rectangular window has ENBW = 1.0 bin.
#[must_use]
pub fn enbw(w: &[f64]) -> f64 {
    let n = w.len() as f64;
    let sum_sq: f64 = w.iter().map(|v| v * v).sum();
    let sum: f64 = w.iter().sum();
    n * sum_sq / (sum * sum)
}

/// Compute the process gain: `PG = 1 / ENBW` (in dB: `-10·log₁₀(ENBW)`).
#[must_use]
pub fn process_gain_db(w: &[f64]) -> f64 {
    -10.0 * enbw(w).log10()
}

/// Compute the peak sidelobe level by evaluating the DTFT of the window and
/// returning the maximum sidelobe amplitude in dB.
///
/// Uses a zero-padded DFT of size `nfft` (should be ≥ 4·N for accuracy).
#[must_use]
pub fn peak_sidelobe_db(w: &[f64], nfft: usize) -> f64 {
    use std::f64::consts::PI;
    let n = w.len();
    // Evaluate the DTFT only over positive frequencies (k = 0..nfft/2+1).
    // The DFT second half (k > nfft/2) is the complex conjugate mirror of
    // the first half for real windows, so including it causes aliasing.
    //
    // Main lobe half-width: use 4×(nfft/N) bins to accommodate wide windows
    // such as Hann (2×) and Blackman (3×), which have wider main lobes than
    // the rectangular window.
    let main_lobe_bins = (4 * nfft / n.max(1)).max(4);
    let mut peak_side = 0.0_f64;
    let mut peak_main = 0.0_f64;

    let k_max = nfft / 2 + 1;
    for k in 0..k_max {
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for (i, &wi) in w.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * i as f64 / nfft as f64;
            re += wi * angle.cos();
            im += wi * angle.sin();
        }
        let mag = (re * re + im * im).sqrt();
        if k <= main_lobe_bins {
            peak_main = peak_main.max(mag);
        } else {
            peak_side = peak_side.max(mag);
        }
    }

    if peak_main == 0.0 || peak_side == 0.0 {
        return f64::NEG_INFINITY;
    }
    20.0 * (peak_side / peak_main).log10()
}

/// Return all common window types as a slice for batch analysis.
#[must_use]
pub fn standard_window_types() -> Vec<WindowType> {
    vec![
        WindowType::Rectangular,
        WindowType::Hann,
        WindowType::Hamming,
        WindowType::Blackman,
        WindowType::BlackmanHarris,
        WindowType::Bartlett,
        WindowType::Kaiser { beta: 8.6 },
        WindowType::FlatTop,
    ]
}

// --------------------------------------------------------------------------- //
//  PTX kernel for window application
// --------------------------------------------------------------------------- //

use crate::{
    ptx_helpers::{bounds_check, global_tid_1d, ptx_header},
    types::SignalPrecision,
};
use oxicuda_ptx::arch::SmVersion;

/// Emits a PTX kernel that applies a window function in-place:
/// ```text
/// x[tid] *= w[tid]
/// ```
/// Both `x_ptr` and `w_ptr` have the same length `n`.
pub fn emit_window_apply_kernel(prec: SignalPrecision, sm: SmVersion) -> String {
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
// Kernel: window_apply
// x[tid] *= w[tid]
.visible .entry window_apply(
    .param .u64 x_ptr,
    .param .u64 w_ptr,
    .param .u64 n_param
)
{{
    {tid_preamble}
    .reg .u64 %x_base, %w_base, %n;
    .reg .u64 %off, %addr;
    .reg .{ty} %x, %w;
    .reg .pred %p_oob;

    ld.param.u64    %x_base, [x_ptr];
    ld.param.u64    %w_base, [w_ptr];
    ld.param.u64    %n,      [n_param];

    {bounds}

    mul.lo.u64      %off,  %tid64, {bytes};
    add.u64         %addr, %x_base, %off;
    ld.global.{ty}  %x, [%addr];
    add.u64         %addr, %w_base, %off;
    ld.global.{ty}  %w, [%addr];
    mul.{ty}        %x, %x, %w;
    add.u64         %addr, %x_base, %off;
    st.global.{ty}  [%addr], %x;

done_window_apply:
    ret;
}}
",
        header = header,
        ty = ty,
        bytes = bytes,
        tid_preamble = global_tid_1d(),
        bounds = bounds_check("%tid64", "%n", "done_window_apply"),
    )
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherent_gain_rectangular() {
        let w = make_window(8, WindowType::Rectangular);
        let cg = coherent_gain(&w);
        assert!((cg - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_coherent_gain_hann() {
        let w = make_window(64, WindowType::Hann);
        let cg = coherent_gain(&w);
        // Hann CG ≈ 0.5
        assert!((cg - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_enbw_rectangular() {
        let w = make_window(64, WindowType::Rectangular);
        let e = enbw(&w);
        assert!((e - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_enbw_hann() {
        let w = make_window(64, WindowType::Hann);
        let e = enbw(&w);
        // Hann ENBW ≈ 1.5 (symmetric formula with n=64 gives ~1.52 due to endpoint taper).
        assert!((e - 1.5).abs() < 0.03, "Hann ENBW = {e}");
    }

    #[test]
    fn test_process_gain_db_rectangular() {
        let w = make_window(64, WindowType::Rectangular);
        let pg = process_gain_db(&w);
        assert!(pg.abs() < 1e-9);
    }

    #[test]
    fn test_peak_sidelobe_rectangular_negative() {
        // Rectangular window has ~-13 dB sidelobe level.
        let w = make_window(64, WindowType::Rectangular);
        let psl = peak_sidelobe_db(&w, 512);
        assert!(psl < -10.0, "rectangular PSL = {psl} dB");
    }

    #[test]
    fn test_peak_sidelobe_blackman_better_than_hann() {
        let wh = make_window(64, WindowType::Hann);
        let wb = make_window(64, WindowType::Blackman);
        let psl_hann = peak_sidelobe_db(&wh, 1024);
        let psl_blackman = peak_sidelobe_db(&wb, 1024);
        assert!(
            psl_blackman < psl_hann,
            "Blackman ({psl_blackman}) should have lower sidelobes than Hann ({psl_hann})"
        );
    }

    #[test]
    fn test_standard_window_types_non_empty() {
        assert!(!standard_window_types().is_empty());
    }

    #[test]
    fn test_window_apply_ptx_entry() {
        let ptx = emit_window_apply_kernel(SignalPrecision::F32, SmVersion::Sm80);
        assert!(ptx.contains(".visible .entry window_apply"));
    }

    #[test]
    fn test_window_apply_ptx_sm90() {
        let ptx = emit_window_apply_kernel(SignalPrecision::F64, SmVersion::Sm90);
        assert!(ptx.contains("sm_90"));
    }
}
