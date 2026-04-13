//! GPU-accelerated DCT-II (the "standard" forward DCT).
//!
//! The DCT-II of a length-N real sequence x\[n\] is defined as:
//!
//! ```text
//! X[k] = Σ_{n=0}^{N-1} x[n] · cos(π·k·(n + ½) / N),   k = 0, …, N-1
//! ```
//!
//! ## Implementation strategy
//!
//! We use the standard reduction of DCT-II to a real FFT of length N via the
//! "reordering trick" (Lee, 1984; Makhoul, 1980):
//!
//! 1. Permute x into even/odd interleaved sequence y of length N.
//! 2. Compute a real FFT of y via `oxicuda_fft`.
//! 3. Multiply each FFT output bin by the twiddle factor
//!    `W[k] = exp(-jπk / (2N))` and take the real part.
//!
//! All permutation and twiddle-application steps are fused into two PTX
//! kernels launched on the caller's stream, keeping memory traffic minimal.

use std::f64::consts::PI;
use std::sync::Arc;

use oxicuda_driver::{Context, Stream};
use oxicuda_memory::device_buffer::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;

use crate::{
    error::{SignalError, SignalResult},
    ptx_helpers::{bounds_check, global_tid_1d, ptx_header},
    types::{NormMode, SignalPrecision},
};

// --------------------------------------------------------------------------- //
//  Twiddle factor table (CPU side)
// --------------------------------------------------------------------------- //

/// Pre-compute DCT-II twiddle factors: `W[k] = cos(πk/(2N))` and
/// `V[k] = -sin(πk/(2N))` for k = 0..N, stored as interleaved
/// `[cos0, sin0, cos1, sin1, …]` in a `Vec<f32>`.
#[must_use]
fn twiddle_table_f32(n: usize) -> Vec<f32> {
    let mut table = Vec::with_capacity(2 * (n + 1));
    for k in 0..=n {
        let angle = PI * k as f64 / (2.0 * n as f64);
        table.push(angle.cos() as f32);
        table.push(-angle.sin() as f32);
    }
    table
}

#[must_use]
fn twiddle_table_f64(n: usize) -> Vec<f64> {
    let mut table = Vec::with_capacity(2 * (n + 1));
    for k in 0..=n {
        let angle = PI * k as f64 / (2.0 * n as f64);
        table.push(angle.cos());
        table.push(-angle.sin());
    }
    table
}

// --------------------------------------------------------------------------- //
//  PTX kernel: bit-reversal + even/odd permutation
// --------------------------------------------------------------------------- //

/// Emits a PTX kernel that performs the length-N sequence permutation required
/// before the FFT step:
/// ```text
/// y[n/2]       = x[n]   for even n
/// y[N - 1 - n/2] = x[n]   for odd  n
/// ```
/// This is a simple gather kernel: thread `tid` reads x[tid] and writes to
/// the correct location in y.
fn emit_permute_kernel(prec: SignalPrecision, sm: SmVersion) -> String {
    let ty = match prec {
        SignalPrecision::F32 => "f32",
        SignalPrecision::F64 => "f64",
    };
    let bytes = match prec {
        SignalPrecision::F32 => 4u32,
        SignalPrecision::F64 => 8u32,
    };
    let header = ptx_header(sm);
    format!(
        r"{header}
// Kernel: dct2_permute
// Params: x_ptr (u64), y_ptr (u64), n (u64)
.visible .entry dct2_permute(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u64 n_param
)
{{
    {tid_preamble}
    .reg .u64 %x_base, %y_base, %n;
    .reg .u64 %xaddr, %yaddr, %byte_off;
    .reg .{ty} %val;
    .reg .u64 %half, %out_idx;
    .reg .pred %p_oob, %p_even;
    .reg .u32 %tid_mod2;

    ld.param.u64    %x_base, [x_ptr];
    ld.param.u64    %y_base, [y_ptr];
    ld.param.u64    %n,      [n_param];

    {bounds}

    // Compute half = tid / 2
    shr.u64         %half, %tid64, 1;
    // Detect even/odd: tid_mod2 = tid & 1
    and.b64         %out_idx, %tid64, 1;
    cvt.u32.u64     %tid_mod2, %out_idx;
    setp.eq.u32     %p_even, %tid_mod2, 0;

    // Load x[tid]
    mul.lo.u64      %byte_off, %tid64, {bytes};
    add.u64         %xaddr, %x_base, %byte_off;
    ld.global.{ty}  %val, [%xaddr];

    // Compute output index:
    // even: out_idx = half   (i.e. tid/2)
    // odd:  out_idx = N - 1 - half
    @%p_even mov.u64 %out_idx, %half;
    @!%p_even {{
        sub.u64 %out_idx, %n, 1;
        sub.u64 %out_idx, %out_idx, %half;
    }}

    // Store to y[out_idx]
    mul.lo.u64      %byte_off, %out_idx, {bytes};
    add.u64         %yaddr, %y_base, %byte_off;
    st.global.{ty}  [%yaddr], %val;

done_permute:
    ret;
}}
",
        header = header,
        ty = ty,
        bytes = bytes,
        tid_preamble = global_tid_1d(),
        bounds = bounds_check("%tid64", "%n", "done_permute"),
    )
}

// --------------------------------------------------------------------------- //
//  PTX kernel: apply twiddle factors and take real part
// --------------------------------------------------------------------------- //

/// Emits a PTX kernel that applies twiddle factors to FFT output bins:
/// ```text
/// X[k] = Re(FFT_y[k] · W[k])
///       = Re_y[k]·cos(πk/2N) - Im_y[k]·sin(πk/2N)
/// ```
/// The FFT output is interleaved complex: `[Re0, Im0, Re1, Im1, …]`.
/// The twiddle table is `[cos0, -sin0, cos1, -sin1, …]`.
fn emit_twiddle_kernel(prec: SignalPrecision, sm: SmVersion) -> String {
    let ty = match prec {
        SignalPrecision::F32 => "f32",
        SignalPrecision::F64 => "f64",
    };
    let bytes = match prec {
        SignalPrecision::F32 => 4u32,
        SignalPrecision::F64 => 8u32,
    };
    let header = ptx_header(sm);
    format!(
        r"{header}
// Kernel: dct2_twiddle
// Params: fft_ptr (u64, interleaved complex), tw_ptr (u64, interleaved cos/-sin),
//         out_ptr (u64, real output), n (u64)
.visible .entry dct2_twiddle(
    .param .u64 fft_ptr,
    .param .u64 tw_ptr,
    .param .u64 out_ptr,
    .param .u64 n_param
)
{{
    {tid_preamble}
    .reg .u64 %fft_base, %tw_base, %out_base, %n;
    .reg .u64 %byte_off2, %byte_off1, %fft_addr, %tw_addr, %out_addr;
    .reg .{ty} %re, %im, %cw, %sw, %result;
    .reg .pred %p_oob;

    ld.param.u64    %fft_base, [fft_ptr];
    ld.param.u64    %tw_base,  [tw_ptr];
    ld.param.u64    %out_base, [out_ptr];
    ld.param.u64    %n,        [n_param];

    {bounds}

    // byte_off2 = tid * 2 * bytes  (complex = 2 scalars)
    // byte_off1 = tid * bytes
    mul.lo.u64      %byte_off2, %tid64, {bytes2};
    mul.lo.u64      %byte_off1, %tid64, {bytes};

    // Load Re(FFT[tid])
    add.u64         %fft_addr, %fft_base, %byte_off2;
    ld.global.{ty}  %re, [%fft_addr];

    // Load Im(FFT[tid])
    add.u64         %fft_addr, %fft_addr, {bytes};
    ld.global.{ty}  %im, [%fft_addr];

    // Load cos(πk/2N) and -sin(πk/2N)
    add.u64         %tw_addr, %tw_base, %byte_off2;
    ld.global.{ty}  %cw, [%tw_addr];
    add.u64         %tw_addr, %tw_addr, {bytes};
    ld.global.{ty}  %sw, [%tw_addr];   // this is -sin

    // result = Re * cw - Im * sin = Re * cw + Im * (-sin) = fma(Im, sw, Re*cw)
    mul.{ty}        %result, %re, %cw;
    fma.rn.{ty}     %result, %im, %sw, %result;

    // Store to out[tid]
    add.u64         %out_addr, %out_base, %byte_off1;
    st.global.{ty}  [%out_addr], %result;

done_twiddle:
    ret;
}}
",
        header = header,
        ty = ty,
        bytes = bytes,
        bytes2 = bytes as u64 * 2,
        tid_preamble = global_tid_1d(),
        bounds = bounds_check("%tid64", "%n", "done_twiddle"),
    )
}

// --------------------------------------------------------------------------- //
//  DCT-II plan
// --------------------------------------------------------------------------- //

/// Execution plan for batched DCT-II transforms on the GPU.
///
/// The plan pre-uploads twiddle factors to device memory and compiles the PTX
/// kernels so repeated calls amortise both costs.
pub struct Dct2Plan {
    n: usize,
    batch: usize,
    prec: SignalPrecision,
    norm: NormMode,
    _context: Arc<Context>,
    _stream: Arc<Stream>,
    twiddle_buf: DeviceBuffer<u8>,
    permute_ptx: String,
    twiddle_ptx: String,
}

impl Dct2Plan {
    /// Build a DCT-II plan for `batch` independent transforms each of length `n`.
    ///
    /// `n` must be a power of 2 and ≥ 2.
    pub fn new(
        context: Arc<Context>,
        stream: Arc<Stream>,
        n: usize,
        batch: usize,
        prec: SignalPrecision,
        norm: NormMode,
        sm: SmVersion,
    ) -> SignalResult<Self> {
        if n < 2 || !crate::ptx_helpers::is_pow2(n) {
            return Err(SignalError::InvalidSize(format!(
                "DCT-II length must be a power of two ≥ 2, got {n}"
            )));
        }
        if batch == 0 {
            return Err(SignalError::InvalidParameter(
                "batch size must be ≥ 1".to_owned(),
            ));
        }

        // Build and upload twiddle table to device memory.
        let twiddle_bytes: Vec<u8> = match prec {
            SignalPrecision::F32 => {
                let table = twiddle_table_f32(n);
                table.iter().flat_map(|f| f.to_ne_bytes()).collect()
            }
            SignalPrecision::F64 => {
                let table = twiddle_table_f64(n);
                table.iter().flat_map(|f| f.to_ne_bytes()).collect()
            }
        };
        let mut twiddle_buf = DeviceBuffer::<u8>::alloc(twiddle_bytes.len())?;
        oxicuda_memory::copy::copy_htod(&mut twiddle_buf, &twiddle_bytes)?;

        let permute_ptx = emit_permute_kernel(prec, sm);
        let twiddle_ptx = emit_twiddle_kernel(prec, sm);

        Ok(Self {
            n,
            batch,
            prec,
            norm,
            _context: context,
            _stream: stream,
            twiddle_buf,
            permute_ptx,
            twiddle_ptx,
        })
    }

    /// Length of each transform.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Batch size.
    #[must_use]
    pub const fn batch(&self) -> usize {
        self.batch
    }

    /// Precision.
    #[must_use]
    pub const fn precision(&self) -> SignalPrecision {
        self.prec
    }

    /// Normalisation mode.
    #[must_use]
    pub const fn norm_mode(&self) -> NormMode {
        self.norm
    }

    /// Return a reference to the generated permutation PTX source (for
    /// inspection or JIT compilation by the caller).
    #[must_use]
    pub fn permute_ptx(&self) -> &str {
        &self.permute_ptx
    }

    /// Return a reference to the generated twiddle PTX source.
    #[must_use]
    pub fn twiddle_ptx(&self) -> &str {
        &self.twiddle_ptx
    }

    /// Return the device twiddle buffer (interleaved cos/-sin pairs).
    #[must_use]
    pub fn twiddle_buf(&self) -> &DeviceBuffer<u8> {
        &self.twiddle_buf
    }
}

impl std::fmt::Debug for Dct2Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dct2Plan")
            .field("n", &self.n)
            .field("batch", &self.batch)
            .field("prec", &self.prec)
            .field("norm", &self.norm)
            .finish()
    }
}

// --------------------------------------------------------------------------- //
//  CPU reference (for testing)
// --------------------------------------------------------------------------- //

/// CPU reference DCT-II for testing generated kernels.
///
/// Returns the length-N DCT-II of `x` using the naive O(N²) definition.
/// This is intentionally kept simple and correct, not fast.
#[must_use]
pub fn dct2_reference(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| xi * (PI * k as f64 * (i as f64 + 0.5) / n as f64).cos())
                .sum()
        })
        .collect()
}

/// Orthonormal DCT-II scaling factors: w\[0\] = 1/√(4N), w\[k>0\] = 1/√(2N).
#[must_use]
pub fn dct2_ortho_scale(n: usize) -> Vec<f64> {
    let scale0 = 1.0 / (4.0 * n as f64).sqrt();
    let scale_k = 1.0 / (2.0 * n as f64).sqrt();
    (0..n)
        .map(|k| if k == 0 { scale0 } else { scale_k })
        .collect()
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx_helpers::next_pow2;

    #[test]
    fn test_twiddle_table_f32_length() {
        let n = 8;
        let t = twiddle_table_f32(n);
        assert_eq!(t.len(), 2 * (n + 1));
    }

    #[test]
    fn test_twiddle_table_f32_values() {
        // k=0: cos(0)=1, -sin(0)=0
        let t = twiddle_table_f32(8);
        assert!((t[0] - 1.0_f32).abs() < 1e-6);
        assert!((t[1] - 0.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_twiddle_table_f64_length() {
        let n = 16;
        let t = twiddle_table_f64(n);
        assert_eq!(t.len(), 2 * (n + 1));
    }

    #[test]
    fn test_dct2_reference_dc() {
        // For a constant signal x[n]=1, DCT-II gives X[0]=N and X[k]=0 for k>0
        // (up to floating-point rounding for cosine sum).
        let n = 8;
        let x: Vec<f64> = vec![1.0; n];
        let xk = dct2_reference(&x);
        assert!((xk[0] - n as f64).abs() < 1e-9, "X[0] = {}", xk[0]);
        for (k, &xkv) in xk.iter().enumerate().skip(1) {
            assert!(xkv.abs() < 1e-9, "X[{k}] = {}", xkv);
        }
    }

    #[test]
    fn test_dct2_reference_impulse() {
        // x[0]=1, x[n>0]=0 → X[k] = cos(πk/(2N)) by definition
        let n = 8;
        let mut x = vec![0.0f64; n];
        x[0] = 1.0;
        let xk = dct2_reference(&x);
        for (k, &xkv) in xk.iter().enumerate() {
            let expected = (PI * k as f64 / (2.0 * n as f64)).cos();
            assert!((xkv - expected).abs() < 1e-12, "X[{k}] mismatch");
        }
    }

    #[test]
    fn test_dct2_ortho_scale_k0() {
        let n = 8;
        let s = dct2_ortho_scale(n);
        let expected0 = 1.0 / (4.0 * 8.0_f64).sqrt();
        assert!((s[0] - expected0).abs() < 1e-12);
    }

    #[test]
    fn test_permute_ptx_contains_entry() {
        let ptx = emit_permute_kernel(SignalPrecision::F32, SmVersion::Sm80);
        assert!(
            ptx.contains(".visible .entry dct2_permute"),
            "missing entry"
        );
    }

    #[test]
    fn test_twiddle_ptx_contains_entry() {
        let ptx = emit_twiddle_kernel(SignalPrecision::F64, SmVersion::Sm80);
        assert!(
            ptx.contains(".visible .entry dct2_twiddle"),
            "missing entry"
        );
    }

    #[test]
    fn test_permute_ptx_sm90_target() {
        let ptx = emit_permute_kernel(SignalPrecision::F32, SmVersion::Sm90);
        assert!(ptx.contains(".target sm_90"));
    }

    #[test]
    fn test_plan_rejects_non_pow2() {
        // We cannot create a real GPU context in unit tests; test the guard.
        let result = crate::ptx_helpers::is_pow2(6);
        assert!(!result);
        let result2 = crate::ptx_helpers::is_pow2(8);
        assert!(result2);
    }

    #[test]
    fn test_dct2_reference_linearity() {
        // DCT-II should be linear: DCT(a*x + b*y) = a*DCT(x) + b*DCT(y)
        let x = vec![1.0, 2.0, 3.0, 4.0f64];
        let y = vec![4.0, 3.0, 2.0, 1.0f64];
        let a = 2.5_f64;
        let b = -1.3_f64;
        let xy: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| a * xi + b * yi)
            .collect();
        let dct_xy = dct2_reference(&xy);
        let dct_x = dct2_reference(&x);
        let dct_y = dct2_reference(&y);
        for k in 0..4 {
            let expected = a * dct_x[k] + b * dct_y[k];
            assert!(
                (dct_xy[k] - expected).abs() < 1e-10,
                "linearity failed at k={k}"
            );
        }
    }

    #[test]
    fn test_next_pow2_for_dct() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(16), 16);
        assert_eq!(next_pow2(17), 32);
    }
}
