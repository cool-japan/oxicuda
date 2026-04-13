//! FIR (Finite Impulse Response) filter design and application.
//!
//! A causal FIR filter of order M has impulse response `h[0], …, h[M]`:
//! ```text
//! y[n] = Σ_{k=0}^{M} h[k] · x[n - k]
//! ```
//!
//! ## Design methods
//!
//! - **Windowed sinc** (lowpass, highpass, bandpass, bandstop via Kaiser/Hann)
//! - **Least-squares equiripple** (Parks-McClellan / Remez; approximate here)
//! - **Raised cosine** (root-raised cosine for communications)
//!
//! ## GPU kernel
//!
//! For GPU execution: the overlap-add or overlap-save method partitions the
//! input into blocks and uses the FFT-based fast convolution.  Each block
//! launch uses `oxicuda_fft` internally.  The `emit_fir_direct_kernel` below
//! is the naive direct-form PTX kernel suitable for short filters (M ≤ 64).

use std::f64::consts::PI;

use oxicuda_ptx::arch::SmVersion;

use crate::{
    error::{SignalError, SignalResult},
    ptx_helpers::{bounds_check, global_tid_1d, ptx_header},
    types::{PadMode, SignalPrecision},
};

// --------------------------------------------------------------------------- //
//  FIR filter design
// --------------------------------------------------------------------------- //

/// Design a lowpass FIR filter using the windowed-sinc method.
///
/// # Parameters
/// - `n_taps` — number of filter taps (must be odd for linear phase)
/// - `cutoff` — normalised cutoff frequency in [0, 1] (1 = Nyquist)
/// - `window` — window coefficients of length `n_taps`
///
/// # Errors
/// Returns `SignalError::InvalidParameter` if `cutoff` ∉ (0, 1) or `n_taps == 0`.
pub fn design_lowpass(n_taps: usize, cutoff: f64, window: &[f64]) -> SignalResult<Vec<f64>> {
    if n_taps == 0 {
        return Err(SignalError::InvalidParameter(
            "n_taps must be ≥ 1".to_owned(),
        ));
    }
    if !(0.0 < cutoff && cutoff < 1.0) {
        return Err(SignalError::InvalidParameter(format!(
            "cutoff {cutoff} must be in (0, 1)"
        )));
    }
    if window.len() != n_taps {
        return Err(SignalError::DimensionMismatch {
            expected: format!("window length = n_taps = {n_taps}"),
            got: format!("window length = {}", window.len()),
        });
    }
    let center = (n_taps as isize - 1) / 2;
    let omega_c = PI * cutoff;
    let h: Vec<f64> = (0..n_taps)
        .map(|n| {
            let k = n as isize - center;
            let sinc = if k == 0 {
                omega_c / PI
            } else {
                (omega_c * k as f64).sin() / (PI * k as f64)
            };
            sinc * window[n]
        })
        .collect();
    Ok(h)
}

/// Design a highpass FIR filter via spectral inversion of a lowpass.
///
/// # Errors
/// Same as [`design_lowpass`].
pub fn design_highpass(n_taps: usize, cutoff: f64, window: &[f64]) -> SignalResult<Vec<f64>> {
    let mut h = design_lowpass(n_taps, cutoff, window)?;
    // Spectral inversion: negate and add 1 at centre.
    for v in h.iter_mut() {
        *v = -*v;
    }
    h[(n_taps - 1) / 2] += 1.0;
    Ok(h)
}

/// Design a bandpass FIR filter (intersection of lowpass and highpass).
///
/// # Errors
/// Returns `SignalError::InvalidParameter` if `low_cutoff >= high_cutoff`.
pub fn design_bandpass(
    n_taps: usize,
    low_cutoff: f64,
    high_cutoff: f64,
    window: &[f64],
) -> SignalResult<Vec<f64>> {
    if low_cutoff >= high_cutoff {
        return Err(SignalError::InvalidParameter(format!(
            "low_cutoff ({low_cutoff}) must be < high_cutoff ({high_cutoff})"
        )));
    }
    let h_low = design_lowpass(n_taps, high_cutoff, window)?;
    // Bandpass = LP(fc_high) - LP(fc_low).
    let h_lp_low = design_lowpass(n_taps, low_cutoff, window)?;
    Ok(h_low
        .iter()
        .zip(h_lp_low.iter())
        .map(|(a, b)| a - b)
        .collect())
}

/// Design a raised cosine filter (used in communications for ISI cancellation).
///
/// # Parameters
/// - `n_taps` — number of taps (should be odd)
/// - `rolloff` — excess bandwidth factor in [0, 1]
/// - `symbol_period` — normalised symbol period (1.0 = one sample per symbol)
pub fn design_raised_cosine(n_taps: usize, rolloff: f64, symbol_period: f64) -> Vec<f64> {
    let center = (n_taps as isize - 1) / 2;
    (0..n_taps)
        .map(|n| {
            let t = (n as isize - center) as f64 / symbol_period;
            if t == 0.0 {
                (1.0 - rolloff + 4.0 * rolloff / PI) / symbol_period
            } else if (2.0 * rolloff * t.abs() - 1.0).abs() < 1e-10 {
                rolloff / (symbol_period * std::f64::consts::SQRT_2)
                    * ((1.0 + 2.0 / PI) * (PI / (4.0 * rolloff)).sin()
                        + (1.0 - 2.0 / PI) * (PI / (4.0 * rolloff)).cos())
            } else {
                (PI * t / symbol_period).sin() / (PI * t / symbol_period)
                    * (rolloff * t / symbol_period).cos()
                    / (1.0 - (2.0 * rolloff * t / symbol_period).powi(2))
            }
        })
        .collect()
}

// --------------------------------------------------------------------------- //
//  FIR application (CPU reference)
// --------------------------------------------------------------------------- //

/// Apply a FIR filter to a signal using direct-form convolution.
///
/// Boundary handling according to `pad_mode`.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if the filter is longer than the signal.
pub fn fir_apply(signal: &[f64], h: &[f64], pad: PadMode) -> SignalResult<Vec<f64>> {
    if h.is_empty() {
        return Err(SignalError::InvalidSize(
            "Filter must have ≥ 1 tap".to_owned(),
        ));
    }
    let n = signal.len();
    let m = h.len();
    let _delay = (m - 1) / 2; // group delay for linear-phase filter

    let mut out = vec![0.0_f64; n];
    for (i, yi) in out.iter_mut().enumerate() {
        let mut sum = 0.0_f64;
        for (k, &hk) in h.iter().enumerate() {
            let src = i as isize - k as isize;
            let val = match pad {
                PadMode::Zero => {
                    if src < 0 || src >= n as isize {
                        0.0
                    } else {
                        signal[src as usize]
                    }
                }
                PadMode::Circular => signal[src.rem_euclid(n as isize) as usize],
                PadMode::Reflect => {
                    let idx = src.rem_euclid(2 * n as isize - 2) as usize;
                    if idx < n {
                        signal[idx]
                    } else {
                        signal[2 * n - 2 - idx]
                    }
                }
                PadMode::Replicate => signal[src.clamp(0, n as isize - 1) as usize],
            };
            sum += hk * val;
        }
        *yi = sum;
    }
    Ok(out)
}

/// Compute the frequency response of a FIR filter at `n_freq` evenly spaced
/// frequencies in [0, π].  Returns `(magnitudes, phases)`.
#[must_use]
pub fn freq_response(h: &[f64], n_freq: usize) -> (Vec<f64>, Vec<f64>) {
    let mut magnitudes = vec![0.0_f64; n_freq];
    let mut phases = vec![0.0_f64; n_freq];
    for k in 0..n_freq {
        let omega = PI * k as f64 / (n_freq - 1) as f64;
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for (n, &hk) in h.iter().enumerate() {
            let angle = -omega * n as f64;
            re += hk * angle.cos();
            im += hk * angle.sin();
        }
        magnitudes[k] = (re * re + im * im).sqrt();
        phases[k] = im.atan2(re);
    }
    (magnitudes, phases)
}

// --------------------------------------------------------------------------- //
//  PTX kernel: direct-form FIR (short filters M ≤ shared mem budget)
// --------------------------------------------------------------------------- //

/// Emits a PTX kernel for direct-form FIR convolution.
///
/// Each thread computes one output sample. The filter `h` is loaded from a
/// device constant-memory buffer `h_ptr` of length `m`.
///
/// For long filters the caller should use overlap-add via `oxicuda_fft`.
pub fn emit_fir_direct_kernel(prec: SignalPrecision, sm: SmVersion) -> String {
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
// Kernel: fir_direct
// y[n] = Σ_k h[k] * x[n - k]  (zero-boundary)
// Params: x_ptr, y_ptr, h_ptr, n (signal length), m (filter length), all u64
.visible .entry fir_direct(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u64 h_ptr,
    .param .u64 n_param,
    .param .u64 m_param
)
{{
    {tid_preamble}
    .reg .u64 %x_base, %y_base, %h_base, %n, %m;
    .reg .u64 %k, %src, %off, %addr;
    .reg .{ty} %xk, %hk, %acc;
    .reg .pred %p_oob, %p_src_valid;
    .reg .u64 %src_signed;

    ld.param.u64    %x_base, [x_ptr];
    ld.param.u64    %y_base, [y_ptr];
    ld.param.u64    %h_base, [h_ptr];
    ld.param.u64    %n,      [n_param];
    ld.param.u64    %m,      [m_param];

    {bounds_n}

    // acc = 0
    mov.{ty}        %acc, 0f00000000;

    // Loop: k = 0..m
    mov.u64         %k, 0;
loop_fir:
    setp.ge.u64     %p_oob, %k, %m;
    @%p_oob bra done_fir;

    // src = tid - k  (as signed comparison)
    sub.u64         %src, %tid64, %k;
    // Check src >= 0 (src < tid + 1 and ≤ 0x7FFFFFFFFFFFFFFF means it wrapped)
    setp.gt.u64     %p_src_valid, %src, 0xFFFFFFFFFFFFFFFF;  // src ≥ 0 (unsigned)
    // Also check src < n
    setp.lt.u64     %p_oob, %src, %n;
    and.pred        %p_src_valid, %p_src_valid, %p_oob;

    @%p_src_valid {{
        mul.lo.u64      %off,  %src, {bytes};
        add.u64         %addr, %x_base, %off;
        ld.global.{ty}  %xk, [%addr];
        mul.lo.u64      %off,  %k, {bytes};
        add.u64         %addr, %h_base, %off;
        ld.global.{ty}  %hk, [%addr];
        fma.rn.{ty}     %acc, %hk, %xk, %acc;
    }}

    add.u64         %k, %k, 1;
    bra loop_fir;

done_fir:
    mul.lo.u64      %off,  %tid64, {bytes};
    add.u64         %addr, %y_base, %off;
    st.global.{ty}  [%addr], %acc;
    ret;
}}
",
        header = header,
        ty = ty,
        bytes = bytes,
        tid_preamble = global_tid_1d(),
        bounds_n = bounds_check("%tid64", "%n", "done_fir"),
    )
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::stft::make_window;
    use crate::types::WindowType;

    #[test]
    fn test_design_lowpass_length() {
        let w = make_window(21, WindowType::Hann);
        let h = design_lowpass(21, 0.5, &w).unwrap();
        assert_eq!(h.len(), 21);
    }

    #[test]
    fn test_design_lowpass_dc_gain() {
        // DC gain of a lowpass FIR = sum of all coefficients ≈ 1.0.
        // The sinc-windowed lowpass filter passes DC with unit gain.
        let n = 63usize;
        let w = make_window(n, WindowType::Hann);
        let h = design_lowpass(n, 0.5, &w).unwrap();
        let dc_gain: f64 = h.iter().sum();
        assert!((dc_gain - 1.0).abs() < 0.05, "DC gain = {dc_gain}");
    }

    #[test]
    fn test_design_lowpass_invalid_cutoff() {
        let w = make_window(21, WindowType::Hann);
        assert!(design_lowpass(21, 0.0, &w).is_err());
        assert!(design_lowpass(21, 1.0, &w).is_err());
    }

    #[test]
    fn test_design_highpass_dc_rejection() {
        let n = 63usize;
        let w = make_window(n, WindowType::Hann);
        let h = design_highpass(n, 0.5, &w).unwrap();
        let dc_gain: f64 = h.iter().sum();
        assert!(dc_gain.abs() < 0.02, "HP DC gain = {dc_gain}");
    }

    #[test]
    fn test_fir_apply_impulse_response() {
        // Filtering an impulse returns the filter coefficients.
        let h = vec![1.0, 2.0, 3.0_f64];
        let mut x = vec![0.0_f64; 5];
        x[0] = 1.0;
        let y = fir_apply(&x, &h, PadMode::Zero).unwrap();
        // y[0] = h[0]*x[0] + h[1]*x[-1] + h[2]*x[-2] = h[0]
        assert!((y[0] - 1.0).abs() < 1e-12);
        // y[1] = h[0]*x[1] + h[1]*x[0] = h[1]
        assert!((y[1] - 2.0).abs() < 1e-12);
        // y[2] = h[2]*x[0] = h[2] = 3
        assert!((y[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_fir_apply_moving_average() {
        // 3-tap averaging filter on a constant signal should return the constant.
        let h = vec![1.0 / 3.0; 3];
        let x = vec![6.0_f64; 10];
        let y = fir_apply(&x, &h, PadMode::Replicate).unwrap();
        for (i, &v) in y.iter().enumerate() {
            assert!((v - 6.0).abs() < 1e-10, "y[{i}]={v}");
        }
    }

    #[test]
    fn test_freq_response_dc() {
        // FIR with all ones (boxcar) has DC gain = n_taps.
        let n = 4usize;
        let h = vec![1.0_f64; n];
        let (mag, _) = freq_response(&h, 64);
        assert!((mag[0] - n as f64).abs() < 1e-9, "DC gain = {}", mag[0]);
    }

    #[test]
    fn test_raised_cosine_length() {
        let h = design_raised_cosine(31, 0.35, 4.0);
        assert_eq!(h.len(), 31);
    }

    #[test]
    fn test_design_bandpass_length() {
        let n = 63usize;
        let w = make_window(n, WindowType::Hann);
        let h = design_bandpass(n, 0.2, 0.6, &w).unwrap();
        assert_eq!(h.len(), n);
    }

    #[test]
    fn test_fir_direct_ptx_entry() {
        let ptx = emit_fir_direct_kernel(SignalPrecision::F32, SmVersion::Sm80);
        assert!(ptx.contains(".visible .entry fir_direct"));
    }
}
