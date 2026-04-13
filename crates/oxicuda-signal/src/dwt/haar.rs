//! GPU-accelerated Haar wavelet transform.
//!
//! The Haar wavelet is the simplest orthonormal wavelet. Each level of the
//! discrete wavelet transform splits the signal into a low-pass (approximation)
//! and high-pass (detail) subband using 2-tap filters:
//!
//! ```text
//! low  filter h[n]: [1/√2,  1/√2]
//! high filter g[n]: [1/√2, -1/√2]
//! ```
//!
//! The entire transform is lossless and can be computed in-place via a
//! lifting scheme:
//! 1. **Predict** step (high-pass): `d[n] = x[2n+1] - x[2n]`
//! 2. **Update** step (low-pass):   `s[n] = x[2n] + d[n]/2`
//!
//! (with normalisation applied after both steps).
//!
//! ## GPU Strategy
//!
//! Each decomposition level launches one PTX kernel that, for a half-warp
//! (or larger thread block), computes the predict/update in shared memory to
//! avoid global-memory round-trips for adjacent elements.

use oxicuda_ptx::arch::SmVersion;

use crate::{
    error::{SignalError, SignalResult},
    ptx_helpers::{bounds_check, global_tid_1d, ptx_header},
    types::SignalPrecision,
};

// --------------------------------------------------------------------------- //
//  CPU reference — lifting scheme
// --------------------------------------------------------------------------- //

/// CPU reference Haar DWT (forward, 1D, in-place).
///
/// Transforms `data` in-place: the first `len/2` elements become the
/// approximation (low-pass), the next `len/2` the detail (high-pass).
///
/// `len` must be even.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `len` is odd or zero.
pub fn haar_forward(data: &mut [f64], len: usize) -> SignalResult<()> {
    if len == 0 || len % 2 != 0 {
        return Err(SignalError::InvalidSize(format!(
            "Haar transform length must be even and ≥ 2, got {len}"
        )));
    }
    let mut tmp = vec![0.0_f64; len];
    let scale = std::f64::consts::FRAC_1_SQRT_2;
    for i in 0..len / 2 {
        tmp[i] = (data[2 * i] + data[2 * i + 1]) * scale;
        tmp[i + len / 2] = (data[2 * i] - data[2 * i + 1]) * scale;
    }
    data[..len].copy_from_slice(&tmp);
    Ok(())
}

/// CPU reference Haar DWT (inverse, 1D, in-place).
///
/// Inverts [`haar_forward`]: reconstructs the original signal from `data`.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `len` is odd or zero.
pub fn haar_inverse(data: &mut [f64], len: usize) -> SignalResult<()> {
    if len == 0 || len % 2 != 0 {
        return Err(SignalError::InvalidSize(format!(
            "Haar inverse length must be even and ≥ 2, got {len}"
        )));
    }
    let mut tmp = vec![0.0_f64; len];
    let scale = std::f64::consts::FRAC_1_SQRT_2;
    for i in 0..len / 2 {
        tmp[2 * i] = (data[i] + data[i + len / 2]) * scale;
        tmp[2 * i + 1] = (data[i] - data[i + len / 2]) * scale;
    }
    data[..len].copy_from_slice(&tmp);
    Ok(())
}

// --------------------------------------------------------------------------- //
//  PTX kernel: single-level Haar forward
// --------------------------------------------------------------------------- //

/// Emits a PTX kernel for one level of the Haar DWT forward pass.
///
/// Thread `tid` computes:
/// ```text
/// approx[tid] = (x[2*tid] + x[2*tid+1]) * scale
/// detail[tid] = (x[2*tid] - x[2*tid+1]) * scale
/// ```
/// where `scale = 1/√2`.
///
/// Output is written to two separate buffers `approx_ptr` and `detail_ptr`,
/// each of length `half = len/2`.
pub fn emit_haar_forward_kernel(prec: SignalPrecision, sm: SmVersion) -> String {
    let ty = match prec {
        SignalPrecision::F32 => "f32",
        SignalPrecision::F64 => "f64",
    };
    let bytes = match prec {
        SignalPrecision::F32 => 4u64,
        SignalPrecision::F64 => 8u64,
    };
    // 1/√2 in IEEE 754 hex
    let inv_sqrt2_bits = match prec {
        SignalPrecision::F32 => format!("0f{:08X}", (std::f32::consts::FRAC_1_SQRT_2).to_bits()),
        SignalPrecision::F64 => {
            format!("0d{:016X}", (std::f64::consts::FRAC_1_SQRT_2).to_bits())
        }
    };
    let header = ptx_header(sm);
    format!(
        r"{header}
// Kernel: haar_forward_level
// Params: x_ptr (u64), approx_ptr (u64), detail_ptr (u64), half (u64)
.visible .entry haar_forward_level(
    .param .u64 x_ptr,
    .param .u64 approx_ptr,
    .param .u64 detail_ptr,
    .param .u64 half_param
)
{{
    {tid_preamble}
    .reg .u64 %x_base, %ap_base, %dt_base, %half;
    .reg .u64 %off2, %off1, %addr;
    .reg .{ty} %xe, %xo, %ap, %dt, %scale;
    .reg .pred %p_oob;

    ld.param.u64    %x_base,  [x_ptr];
    ld.param.u64    %ap_base, [approx_ptr];
    ld.param.u64    %dt_base, [detail_ptr];
    ld.param.u64    %half,    [half_param];

    {bounds}

    // off2 = 2 * tid * bytes  (index into x at even position)
    shl.b64         %off2, %tid64, 1;
    mul.lo.u64      %off2, %off2, {bytes};
    // off1 = tid * bytes
    mul.lo.u64      %off1, %tid64, {bytes};

    // Load x[2*tid]
    add.u64         %addr, %x_base, %off2;
    ld.global.{ty}  %xe, [%addr];

    // Load x[2*tid + 1]
    add.u64         %addr, %addr, {bytes};
    ld.global.{ty}  %xo, [%addr];

    // scale = 1/√2
    mov.{ty}        %scale, {inv_sqrt2};

    // approx = (xe + xo) * scale
    add.{ty}        %ap, %xe, %xo;
    mul.{ty}        %ap, %ap, %scale;

    // detail = (xe - xo) * scale
    sub.{ty}        %dt, %xe, %xo;
    mul.{ty}        %dt, %dt, %scale;

    // Store approx[tid]
    add.u64         %addr, %ap_base, %off1;
    st.global.{ty}  [%addr], %ap;

    // Store detail[tid]
    add.u64         %addr, %dt_base, %off1;
    st.global.{ty}  [%addr], %dt;

done_haar_fwd:
    ret;
}}
",
        header = header,
        ty = ty,
        bytes = bytes,
        inv_sqrt2 = inv_sqrt2_bits,
        tid_preamble = global_tid_1d(),
        bounds = bounds_check("%tid64", "%half", "done_haar_fwd"),
    )
}

/// Emits a PTX kernel for one level of the Haar DWT inverse pass.
///
/// Thread `tid` computes:
/// ```text
/// x[2*tid]   = (approx[tid] + detail[tid]) * scale
/// x[2*tid+1] = (approx[tid] - detail[tid]) * scale
/// ```
pub fn emit_haar_inverse_kernel(prec: SignalPrecision, sm: SmVersion) -> String {
    let ty = match prec {
        SignalPrecision::F32 => "f32",
        SignalPrecision::F64 => "f64",
    };
    let bytes = match prec {
        SignalPrecision::F32 => 4u64,
        SignalPrecision::F64 => 8u64,
    };
    let inv_sqrt2_bits = match prec {
        SignalPrecision::F32 => format!("0f{:08X}", (std::f32::consts::FRAC_1_SQRT_2).to_bits()),
        SignalPrecision::F64 => {
            format!("0d{:016X}", (std::f64::consts::FRAC_1_SQRT_2).to_bits())
        }
    };
    let header = ptx_header(sm);
    format!(
        r"{header}
// Kernel: haar_inverse_level
.visible .entry haar_inverse_level(
    .param .u64 approx_ptr,
    .param .u64 detail_ptr,
    .param .u64 x_ptr,
    .param .u64 half_param
)
{{
    {tid_preamble}
    .reg .u64 %ap_base, %dt_base, %x_base, %half;
    .reg .u64 %off2, %off1, %addr;
    .reg .{ty} %ap, %dt, %xe, %xo, %scale;
    .reg .pred %p_oob;

    ld.param.u64    %ap_base, [approx_ptr];
    ld.param.u64    %dt_base, [detail_ptr];
    ld.param.u64    %x_base,  [x_ptr];
    ld.param.u64    %half,    [half_param];

    {bounds}

    mul.lo.u64      %off1, %tid64, {bytes};
    shl.b64         %off2, %tid64, 1;
    mul.lo.u64      %off2, %off2, {bytes};

    add.u64         %addr, %ap_base, %off1;
    ld.global.{ty}  %ap, [%addr];
    add.u64         %addr, %dt_base, %off1;
    ld.global.{ty}  %dt, [%addr];

    mov.{ty}        %scale, {inv_sqrt2};

    add.{ty}        %xe, %ap, %dt;
    mul.{ty}        %xe, %xe, %scale;
    sub.{ty}        %xo, %ap, %dt;
    mul.{ty}        %xo, %xo, %scale;

    add.u64         %addr, %x_base, %off2;
    st.global.{ty}  [%addr], %xe;
    add.u64         %addr, %addr, {bytes};
    st.global.{ty}  [%addr], %xo;

done_haar_inv:
    ret;
}}
",
        header = header,
        ty = ty,
        bytes = bytes,
        inv_sqrt2 = inv_sqrt2_bits,
        tid_preamble = global_tid_1d(),
        bounds = bounds_check("%tid64", "%half", "done_haar_inv"),
    )
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_forward_dc() {
        // [c, c, c, c] → approx = [c·√2, c·√2], detail = [0, 0]
        let mut d = vec![1.0_f64; 4];
        haar_forward(&mut d, 4).unwrap();
        let sqrt2 = std::f64::consts::SQRT_2;
        for (i, &dv) in d.iter().enumerate().take(2) {
            assert!((dv - sqrt2).abs() < 1e-12, "approx[{i}]={}", dv);
        }
        for (i, &dv) in d.iter().enumerate().skip(2).take(2) {
            assert!(dv.abs() < 1e-12, "detail[{i}]={}", dv);
        }
    }

    #[test]
    fn test_haar_forward_inverse_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0_f64];
        let mut d = original.clone();
        haar_forward(&mut d, 8).unwrap();
        haar_inverse(&mut d, 8).unwrap();
        for (a, b) in original.iter().zip(d.iter()) {
            assert!((a - b).abs() < 1e-12, "round-trip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_haar_forward_orthogonality() {
        // Energy preservation: sum |d|² = sum |x|²
        let x = vec![1.0, 2.0, 3.0, 4.0_f64];
        let energy_in: f64 = x.iter().map(|v| v * v).sum();
        let mut d = x.clone();
        haar_forward(&mut d, 4).unwrap();
        let energy_out: f64 = d.iter().map(|v| v * v).sum();
        assert!((energy_in - energy_out).abs() < 1e-12);
    }

    #[test]
    fn test_haar_forward_invalid_size() {
        let mut d = vec![1.0_f64; 3];
        assert!(haar_forward(&mut d, 3).is_err());
    }

    #[test]
    fn test_haar_forward_ptx_entry() {
        let ptx = emit_haar_forward_kernel(SignalPrecision::F32, SmVersion::Sm80);
        assert!(ptx.contains(".visible .entry haar_forward_level"));
    }

    #[test]
    fn test_haar_inverse_ptx_entry() {
        let ptx = emit_haar_inverse_kernel(SignalPrecision::F64, SmVersion::Sm90);
        assert!(ptx.contains(".visible .entry haar_inverse_level"));
        assert!(ptx.contains("sm_90"));
    }

    #[test]
    fn test_haar_forward_impulse() {
        // x = [1, 0, 0, 0]: approx[0] = 1/√2, detail[0] = 1/√2, rest 0
        let mut d = vec![1.0, 0.0, 0.0, 0.0_f64];
        haar_forward(&mut d, 4).unwrap();
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((d[0] - expected).abs() < 1e-12);
        assert!((d[2] - expected).abs() < 1e-12);
        assert!(d[1].abs() < 1e-12);
        assert!(d[3].abs() < 1e-12);
    }
}
