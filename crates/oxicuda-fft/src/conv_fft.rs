//! High-level convolution via FFT helpers.
//!
//! Provides FFT-based convolution, cross-correlation, and 2D convolution plans
//! that generate PTX kernels for element-wise operations in the frequency domain.
//!
//! ## Algorithm
//!
//! FFT convolution computes `signal * kernel` by:
//! 1. Zero-pad both signal and kernel to length `fft_len` (next power of 2)
//! 2. FFT both padded sequences
//! 3. Pointwise complex multiply in frequency domain
//! 4. Inverse FFT the product
//! 5. Extract the output region based on [`ConvolutionMode`]
//!
//! This is O(N log N) versus O(N*M) for direct convolution, which is
//! advantageous when the kernel is large.

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{FftError, FftResult};
use crate::ptx_helpers::{ptx_float_type, ptx_type_suffix};
use crate::transforms::real_fft::PtxModule;
use crate::types::FftPrecision;

// ---------------------------------------------------------------------------
// ConvolutionMode
// ---------------------------------------------------------------------------

/// Determines the output size of a convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvolutionMode {
    /// Full convolution output of size `m + n - 1`, where `m` is the signal
    /// length and `n` is the kernel length.
    Full,
    /// Output has the same size as the larger of the two inputs.
    Same,
    /// Only the portion computed without zero-padded edges.
    /// Output size is `max(m, n) - min(m, n) + 1`.
    Valid,
}

impl ConvolutionMode {
    /// Computes the output length for a 1D convolution given signal and kernel
    /// lengths.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::InvalidSize`] if `Valid` mode produces zero or
    /// negative output (i.e., both inputs are zero length).
    fn output_len(self, signal_len: usize, kernel_len: usize) -> FftResult<usize> {
        let result = match self {
            ConvolutionMode::Full => signal_len
                .checked_add(kernel_len)
                .and_then(|s| s.checked_sub(1))
                .ok_or_else(|| {
                    FftError::InvalidSize(format!(
                        "overflow computing full convolution output: \
                             signal_len={signal_len}, kernel_len={kernel_len}"
                    ))
                })?,
            ConvolutionMode::Same => signal_len.max(kernel_len),
            ConvolutionMode::Valid => {
                let big = signal_len.max(kernel_len);
                let small = signal_len.min(kernel_len);
                big.checked_sub(small)
                    .and_then(|d| d.checked_add(1))
                    .ok_or_else(|| {
                        FftError::InvalidSize(format!(
                            "overflow computing valid convolution output: \
                             signal_len={signal_len}, kernel_len={kernel_len}"
                        ))
                    })?
            }
        };
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// next_fft_size — power-of-2 padding
// ---------------------------------------------------------------------------

/// Returns the smallest power of 2 that is >= `n`.
///
/// # Panics
///
/// This function does not panic. If `n` is zero, returns 1.
/// If `n` exceeds the largest representable power of 2 for `usize`,
/// returns `n` unchanged (saturating behavior).
pub fn next_fft_size(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    // For values that are already powers of 2, return them directly.
    // Otherwise, find the next power of 2.
    let result = n.checked_next_power_of_two();
    match result {
        Some(p) => p,
        // Overflow: n is too large for the next power of 2
        None => n,
    }
}

// ---------------------------------------------------------------------------
// FftConvPlan — 1D convolution via FFT
// ---------------------------------------------------------------------------

/// A plan for performing 1D convolution via FFT.
///
/// This plan precomputes the padded FFT size and can generate PTX kernels
/// for the pointwise multiply and zero-padding steps.
#[derive(Debug, Clone)]
pub struct FftConvPlan {
    /// Length of the input signal.
    signal_len: usize,
    /// Length of the convolution kernel.
    kernel_len: usize,
    /// Convolution mode determining output size.
    mode: ConvolutionMode,
    /// Padded FFT size (next power of 2 >= signal_len + kernel_len - 1).
    fft_len: usize,
    /// Floating-point precision for the computation.
    precision: FftPrecision,
}

impl FftConvPlan {
    /// Creates a new 1D FFT convolution plan.
    ///
    /// # Arguments
    ///
    /// * `signal_len` - Length of the input signal
    /// * `kernel_len` - Length of the convolution kernel
    /// * `mode` - [`ConvolutionMode`] determining output size
    /// * `precision` - [`FftPrecision::Single`] or [`FftPrecision::Double`]
    ///
    /// # Errors
    ///
    /// Returns [`FftError::InvalidSize`] if either length is zero or if the
    /// combined length overflows.
    pub fn new(
        signal_len: usize,
        kernel_len: usize,
        mode: ConvolutionMode,
        precision: FftPrecision,
    ) -> FftResult<Self> {
        if signal_len == 0 {
            return Err(FftError::InvalidSize(
                "signal length must be > 0".to_string(),
            ));
        }
        if kernel_len == 0 {
            return Err(FftError::InvalidSize(
                "kernel length must be > 0".to_string(),
            ));
        }

        // Full convolution length = signal_len + kernel_len - 1
        let full_len = signal_len
            .checked_add(kernel_len)
            .and_then(|s| s.checked_sub(1))
            .ok_or_else(|| {
                FftError::InvalidSize(format!(
                    "overflow computing full convolution length: \
                     signal_len={signal_len}, kernel_len={kernel_len}"
                ))
            })?;

        let fft_len = next_fft_size(full_len);

        // Validate that the requested mode produces a valid output length
        let _output = mode.output_len(signal_len, kernel_len)?;

        Ok(Self {
            signal_len,
            kernel_len,
            mode,
            fft_len,
            precision,
        })
    }

    /// Returns the output length based on the convolution mode.
    ///
    /// - `Full`: `signal_len + kernel_len - 1`
    /// - `Same`: `max(signal_len, kernel_len)`
    /// - `Valid`: `max(signal_len, kernel_len) - min(signal_len, kernel_len) + 1`
    pub fn output_len(&self) -> usize {
        // Safe: validated in `new()`
        self.mode
            .output_len(self.signal_len, self.kernel_len)
            .unwrap_or(0)
    }

    /// Returns the padded FFT size (always a power of 2).
    pub fn fft_len(&self) -> usize {
        self.fft_len
    }

    /// Returns the signal length.
    pub fn signal_len(&self) -> usize {
        self.signal_len
    }

    /// Returns the kernel length.
    pub fn kernel_len(&self) -> usize {
        self.kernel_len
    }

    /// Returns the convolution mode.
    pub fn mode(&self) -> ConvolutionMode {
        self.mode
    }

    /// Returns the precision.
    pub fn precision(&self) -> FftPrecision {
        self.precision
    }

    /// Computes the total workspace size in bytes needed for the convolution.
    ///
    /// This includes space for:
    /// - Padded signal (complex): `fft_len` complex elements
    /// - Padded kernel (complex): `fft_len` complex elements
    /// - Output buffer (complex): `fft_len` complex elements
    pub fn workspace_size(&self) -> usize {
        // 3 complex arrays of fft_len each
        3 * self.fft_len * self.precision.complex_bytes()
    }

    /// Generates a PTX kernel for element-wise complex multiplication in
    /// the frequency domain.
    ///
    /// The kernel multiplies two complex arrays element-by-element:
    /// ```text
    /// out[i] = signal_fft[i] * kernel_fft[i]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FftError::PtxGeneration`] if PTX code generation fails.
    pub fn generate_pointwise_multiply_ptx(&self) -> FftResult<PtxModule> {
        generate_complex_multiply_kernel(self.fft_len, self.precision, ComplexMultiplyOp::Standard)
    }

    /// Generates a PTX kernel for zero-padding a signal to `fft_len`.
    ///
    /// The kernel copies the first `input_len` elements from the input
    /// to the output and fills the remaining positions with zeros.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::PtxGeneration`] if PTX code generation fails.
    pub fn generate_zero_pad_ptx(&self) -> FftResult<PtxModule> {
        generate_zero_pad_kernel(self.fft_len, self.precision)
    }
}

// ---------------------------------------------------------------------------
// CrossCorrelationPlan
// ---------------------------------------------------------------------------

/// A plan for computing cross-correlation via FFT.
///
/// Cross-correlation is like convolution but the kernel is conjugated in
/// the frequency domain instead of being reversed in the time domain:
/// ```text
/// (f ★ g)[n] = Σ_m f*[m] g[m + n]
/// ```
///
/// In the frequency domain:
/// ```text
/// F{f ★ g} = conj(F{f}) · F{g}
/// ```
#[derive(Debug, Clone)]
pub struct CrossCorrelationPlan {
    /// Length of the first signal.
    signal_len: usize,
    /// Length of the second signal (kernel).
    kernel_len: usize,
    /// Floating-point precision.
    precision: FftPrecision,
    /// Padded FFT size.
    fft_len: usize,
}

impl CrossCorrelationPlan {
    /// Creates a new cross-correlation plan.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::InvalidSize`] if either length is zero.
    pub fn new(signal_len: usize, kernel_len: usize, precision: FftPrecision) -> FftResult<Self> {
        if signal_len == 0 {
            return Err(FftError::InvalidSize(
                "signal length must be > 0".to_string(),
            ));
        }
        if kernel_len == 0 {
            return Err(FftError::InvalidSize(
                "kernel length must be > 0".to_string(),
            ));
        }

        let full_len = signal_len
            .checked_add(kernel_len)
            .and_then(|s| s.checked_sub(1))
            .ok_or_else(|| {
                FftError::InvalidSize(format!(
                    "overflow computing cross-correlation length: \
                     signal_len={signal_len}, kernel_len={kernel_len}"
                ))
            })?;

        let fft_len = next_fft_size(full_len);

        Ok(Self {
            signal_len,
            kernel_len,
            precision,
            fft_len,
        })
    }

    /// Returns the output length (same as full convolution: `m + n - 1`).
    pub fn output_len(&self) -> usize {
        self.signal_len + self.kernel_len - 1
    }

    /// Returns the padded FFT size.
    pub fn fft_len(&self) -> usize {
        self.fft_len
    }

    /// Returns the signal length.
    pub fn signal_len(&self) -> usize {
        self.signal_len
    }

    /// Returns the kernel length.
    pub fn kernel_len(&self) -> usize {
        self.kernel_len
    }

    /// Returns the precision.
    pub fn precision(&self) -> FftPrecision {
        self.precision
    }

    /// Computes workspace size in bytes for the cross-correlation.
    pub fn workspace_size(&self) -> usize {
        3 * self.fft_len * self.precision.complex_bytes()
    }

    /// Generates a PTX kernel for conjugate-multiply in the frequency domain.
    ///
    /// The kernel computes:
    /// ```text
    /// out[i] = conj(a[i]) * b[i]
    /// ```
    ///
    /// This is the key difference from regular convolution: the first operand
    /// is conjugated before multiplication.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::PtxGeneration`] if PTX code generation fails.
    pub fn generate_conj_multiply_ptx(&self) -> FftResult<PtxModule> {
        generate_complex_multiply_kernel(self.fft_len, self.precision, ComplexMultiplyOp::Conjugate)
    }
}

// ---------------------------------------------------------------------------
// FftConv2dPlan — 2D convolution via FFT
// ---------------------------------------------------------------------------

/// A plan for performing 2D convolution via FFT.
///
/// The 2D convolution is computed by:
/// 1. Zero-pad both input and kernel to `(fft_h, fft_w)` (powers of 2)
/// 2. Compute 2D FFT of both (row FFTs then column FFTs)
/// 3. Pointwise complex multiply in frequency domain
/// 4. 2D inverse FFT
/// 5. Extract output region based on mode
#[derive(Debug, Clone)]
pub struct FftConv2dPlan {
    /// Input height.
    input_h: usize,
    /// Input width.
    input_w: usize,
    /// Kernel height.
    kernel_h: usize,
    /// Kernel width.
    kernel_w: usize,
    /// Convolution mode.
    mode: ConvolutionMode,
    /// Floating-point precision.
    precision: FftPrecision,
    /// Padded FFT height (power of 2).
    fft_h: usize,
    /// Padded FFT width (power of 2).
    fft_w: usize,
}

impl FftConv2dPlan {
    /// Creates a new 2D FFT convolution plan.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::InvalidSize`] if any dimension is zero.
    pub fn new(
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        mode: ConvolutionMode,
        precision: FftPrecision,
    ) -> FftResult<Self> {
        if input_h == 0 || input_w == 0 {
            return Err(FftError::InvalidSize(
                "input dimensions must be > 0".to_string(),
            ));
        }
        if kernel_h == 0 || kernel_w == 0 {
            return Err(FftError::InvalidSize(
                "kernel dimensions must be > 0".to_string(),
            ));
        }

        let full_h = input_h
            .checked_add(kernel_h)
            .and_then(|s| s.checked_sub(1))
            .ok_or_else(|| {
                FftError::InvalidSize(format!(
                    "overflow in height: input_h={input_h}, kernel_h={kernel_h}"
                ))
            })?;

        let full_w = input_w
            .checked_add(kernel_w)
            .and_then(|s| s.checked_sub(1))
            .ok_or_else(|| {
                FftError::InvalidSize(format!(
                    "overflow in width: input_w={input_w}, kernel_w={kernel_w}"
                ))
            })?;

        let fft_h = next_fft_size(full_h);
        let fft_w = next_fft_size(full_w);

        // Validate mode produces valid output
        let _oh = mode.output_len(input_h, kernel_h)?;
        let _ow = mode.output_len(input_w, kernel_w)?;

        Ok(Self {
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            mode,
            precision,
            fft_h,
            fft_w,
        })
    }

    /// Returns the output height based on convolution mode.
    pub fn output_h(&self) -> usize {
        self.mode
            .output_len(self.input_h, self.kernel_h)
            .unwrap_or(0)
    }

    /// Returns the output width based on convolution mode.
    pub fn output_w(&self) -> usize {
        self.mode
            .output_len(self.input_w, self.kernel_w)
            .unwrap_or(0)
    }

    /// Returns the padded FFT height.
    pub fn fft_h(&self) -> usize {
        self.fft_h
    }

    /// Returns the padded FFT width.
    pub fn fft_w(&self) -> usize {
        self.fft_w
    }

    /// Returns the input dimensions.
    pub fn input_dims(&self) -> (usize, usize) {
        (self.input_h, self.input_w)
    }

    /// Returns the kernel dimensions.
    pub fn kernel_dims(&self) -> (usize, usize) {
        (self.kernel_h, self.kernel_w)
    }

    /// Computes total workspace size in bytes for 2D convolution.
    ///
    /// Includes space for:
    /// - Padded input (complex): `fft_h * fft_w` complex elements
    /// - Padded kernel (complex): `fft_h * fft_w` complex elements
    /// - Output buffer (complex): `fft_h * fft_w` complex elements
    pub fn workspace_size(&self) -> usize {
        let total_elements = self.fft_h * self.fft_w;
        3 * total_elements * self.precision.complex_bytes()
    }

    /// Generates a PTX kernel for element-wise complex multiplication of
    /// two 2D frequency-domain arrays.
    ///
    /// The kernel treats the 2D arrays as flat 1D arrays of `fft_h * fft_w`
    /// complex elements and multiplies them element-by-element.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::PtxGeneration`] if PTX code generation fails.
    pub fn generate_pointwise_multiply_2d_ptx(&self) -> FftResult<PtxModule> {
        let total = self.fft_h * self.fft_w;
        generate_complex_multiply_kernel(total, self.precision, ComplexMultiplyOp::Standard)
    }

    /// Generates a PTX kernel for zero-padding a 2D input to `(fft_h, fft_w)`.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::PtxGeneration`] if PTX code generation fails.
    pub fn generate_zero_pad_2d_ptx(&self) -> FftResult<PtxModule> {
        generate_zero_pad_2d_kernel(self.fft_h, self.fft_w, self.precision)
    }
}

// ---------------------------------------------------------------------------
// Internal: complex multiply operation type
// ---------------------------------------------------------------------------

/// Selects between standard and conjugate complex multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComplexMultiplyOp {
    /// Standard: `out = a * b`
    Standard,
    /// Conjugate: `out = conj(a) * b`
    Conjugate,
}

// ---------------------------------------------------------------------------
// PTX kernel generators
// ---------------------------------------------------------------------------

/// Block size for element-wise kernels.
const ELEMENTWISE_BLOCK_SIZE: u32 = 256;

/// Generates a PTX kernel for complex element-wise multiply.
///
/// Supports both standard multiply and conjugate multiply (for
/// cross-correlation).
fn generate_complex_multiply_kernel(
    n: usize,
    precision: FftPrecision,
    op: ComplexMultiplyOp,
) -> FftResult<PtxModule> {
    let float_ty = ptx_float_type(precision);
    let suffix = ptx_type_suffix(precision);
    let op_name = match op {
        ComplexMultiplyOp::Standard => "pointwise_mul",
        ComplexMultiplyOp::Conjugate => "conj_mul",
    };
    let kernel_name = format!("fft_conv_{op_name}_{suffix}_n{n}");

    let sm = SmVersion::Sm75;
    let elem_bytes = precision.element_bytes();

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("a_ptr", PtxType::U64) // input array A (complex)
        .param("b_ptr", PtxType::U64) // input array B (complex)
        .param("out_ptr", PtxType::U64) // output array (complex)
        .param("count", PtxType::U32) // number of complex elements
        .max_threads_per_block(ELEMENTWISE_BLOCK_SIZE)
        .body(move |b| {
            b.comment(&format!(
                "Element-wise complex {}: N={n}",
                match op {
                    ComplexMultiplyOp::Standard => "multiply",
                    ComplexMultiplyOp::Conjugate => "conjugate-multiply",
                }
            ));

            // tid = threadIdx.x + blockIdx.x * blockDim.x
            let tid = b.thread_id_x();
            let bid = b.block_id_x();
            let bsz = b.block_dim_x();
            let global_tid = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {global_tid}, {bid}, {bsz}, {tid};"));

            // Bounds check
            let count_reg = b.load_param_u32("count");
            let pred = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {pred}, {global_tid}, {count_reg};"));
            b.raw_ptx(&format!("@{pred} bra $L_exit;"));

            // Compute byte offset: global_tid * 2 * elem_bytes (complex = 2 floats)
            let complex_bytes = elem_bytes * 2;
            let offset_u32 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "mul.lo.u32 {offset_u32}, {global_tid}, {complex_bytes};"
            ));
            let offset = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("cvt.u64.u32 {offset}, {offset_u32};"));

            // Load base pointers
            let a_base = b.load_param_u64("a_ptr");
            let b_base = b.load_param_u64("b_ptr");
            let out_base = b.load_param_u64("out_ptr");

            // Compute element addresses
            let a_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {a_addr}, {a_base}, {offset};"));
            let b_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {b_addr}, {b_base}, {offset};"));
            let out_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {out_addr}, {out_base}, {offset};"));

            // Load a.re, a.im
            let a_re = b.alloc_reg(float_ty);
            b.raw_ptx(&format!("ld.global.{suffix} {a_re}, [{a_addr}];"));
            let a_im_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {a_im_addr}, {a_addr}, {elem_bytes};"));
            let a_im = b.alloc_reg(float_ty);
            b.raw_ptx(&format!("ld.global.{suffix} {a_im}, [{a_im_addr}];"));

            // Load b.re, b.im
            let b_re = b.alloc_reg(float_ty);
            b.raw_ptx(&format!("ld.global.{suffix} {b_re}, [{b_addr}];"));
            let b_im_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {b_im_addr}, {b_addr}, {elem_bytes};"));
            let b_im = b.alloc_reg(float_ty);
            b.raw_ptx(&format!("ld.global.{suffix} {b_im}, [{b_im_addr}];"));

            // For conjugate mode: negate a_im
            // Standard:  out = (a_re + j*a_im) * (b_re + j*b_im)
            // Conjugate: out = (a_re - j*a_im) * (b_re + j*b_im)
            let effective_a_im = if op == ComplexMultiplyOp::Conjugate {
                let neg_aim = b.alloc_reg(float_ty);
                b.raw_ptx(&format!("neg.{suffix} {neg_aim}, {a_im};"));
                neg_aim
            } else {
                a_im
            };

            // Complex multiply:
            // out_re = a_re * b_re - a_im_eff * b_im
            // out_im = a_re * b_im + a_im_eff * b_re
            let out_re = b.alloc_reg(float_ty);
            b.raw_ptx(&format!("mul.rn.{suffix} {out_re}, {a_re}, {b_re};"));
            let neg_aim_eff = b.alloc_reg(float_ty);
            b.raw_ptx(&format!("neg.{suffix} {neg_aim_eff}, {effective_a_im};"));
            let out_re_final = b.alloc_reg(float_ty);
            b.raw_ptx(&format!(
                "fma.rn.{suffix} {out_re_final}, {neg_aim_eff}, {b_im}, {out_re};"
            ));

            let out_im = b.alloc_reg(float_ty);
            b.raw_ptx(&format!("mul.rn.{suffix} {out_im}, {a_re}, {b_im};"));
            let out_im_final = b.alloc_reg(float_ty);
            b.raw_ptx(&format!(
                "fma.rn.{suffix} {out_im_final}, {effective_a_im}, {b_re}, {out_im};"
            ));

            // Store result
            b.raw_ptx(&format!("st.global.{suffix} [{out_addr}], {out_re_final};"));
            let out_im_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {out_im_addr}, {out_addr}, {elem_bytes};"));
            b.raw_ptx(&format!(
                "st.global.{suffix} [{out_im_addr}], {out_im_final};"
            ));

            b.raw_ptx("$L_exit:");
        })
        .build()?;

    Ok(PtxModule {
        source: ptx,
        entry_name: kernel_name,
        block_size: ELEMENTWISE_BLOCK_SIZE,
    })
}

/// Generates a PTX kernel for zero-padding a 1D signal.
///
/// Copies `input_len` elements from input to output, pads the rest with zeros.
/// The `input_len` is passed as a kernel parameter at runtime.
fn generate_zero_pad_kernel(fft_len: usize, precision: FftPrecision) -> FftResult<PtxModule> {
    let float_ty = ptx_float_type(precision);
    let suffix = ptx_type_suffix(precision);
    let kernel_name = format!("fft_conv_zero_pad_{suffix}_n{fft_len}");
    let sm = SmVersion::Sm75;
    let elem_bytes = precision.element_bytes();

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("input_len", PtxType::U32) // actual signal length
        .param("total_len", PtxType::U32) // fft_len (padded size)
        .max_threads_per_block(ELEMENTWISE_BLOCK_SIZE)
        .body(move |b| {
            b.comment(&format!("Zero-pad kernel: fft_len={fft_len}"));

            let tid = b.thread_id_x();
            let bid = b.block_id_x();
            let bsz = b.block_dim_x();
            let global_tid = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {global_tid}, {bid}, {bsz}, {tid};"));

            // Bounds check
            let total = b.load_param_u32("total_len");
            let pred_exit = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {pred_exit}, {global_tid}, {total};"));
            b.raw_ptx(&format!("@{pred_exit} bra $L_zp_exit;"));

            // Compute byte offsets for real values (not complex pairs here;
            // the kernel handles individual float elements)
            let offset_u32 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "mul.lo.u32 {offset_u32}, {global_tid}, {elem_bytes};"
            ));
            let offset = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("cvt.u64.u32 {offset}, {offset_u32};"));

            let out_base = b.load_param_u64("output_ptr");
            let out_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {out_addr}, {out_base}, {offset};"));

            // Check if within input range
            let input_len = b.load_param_u32("input_len");
            let pred_copy = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!(
                "setp.lt.u32 {pred_copy}, {global_tid}, {input_len};"
            ));
            b.raw_ptx(&format!("@!{pred_copy} bra $L_zero;"));

            // Copy from input
            let in_base = b.load_param_u64("input_ptr");
            let in_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {in_addr}, {in_base}, {offset};"));

            let val = b.alloc_reg(float_ty);
            b.raw_ptx(&format!("ld.global.{suffix} {val}, [{in_addr}];"));
            b.raw_ptx(&format!("st.global.{suffix} [{out_addr}], {val};"));
            b.raw_ptx("bra $L_zp_exit;");

            // Store zero
            b.raw_ptx("$L_zero:");
            let zero_val = b.alloc_reg(float_ty);
            match precision {
                FftPrecision::Single => {
                    b.raw_ptx(&format!("mov.b32 {zero_val}, 0F00000000;"));
                }
                FftPrecision::Double => {
                    b.raw_ptx(&format!("mov.b64 {zero_val}, 0D0000000000000000;"));
                }
            }
            b.raw_ptx(&format!("st.global.{suffix} [{out_addr}], {zero_val};"));

            b.raw_ptx("$L_zp_exit:");
        })
        .build()?;

    Ok(PtxModule {
        source: ptx,
        entry_name: kernel_name,
        block_size: ELEMENTWISE_BLOCK_SIZE,
    })
}

/// Generates a PTX kernel for zero-padding a 2D input.
///
/// The kernel handles row-major layout: for each output position (row, col),
/// if it is within the original input dimensions, copy; otherwise store zero.
fn generate_zero_pad_2d_kernel(
    fft_h: usize,
    fft_w: usize,
    precision: FftPrecision,
) -> FftResult<PtxModule> {
    let float_ty = ptx_float_type(precision);
    let suffix = ptx_type_suffix(precision);
    let kernel_name = format!("fft_conv_zero_pad_2d_{suffix}_h{fft_h}_w{fft_w}");
    let sm = SmVersion::Sm75;
    let elem_bytes = precision.element_bytes();
    let total_elements = fft_h * fft_w;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("input_h", PtxType::U32)
        .param("input_w", PtxType::U32)
        .param("out_w", PtxType::U32) // fft_w
        .param("total_count", PtxType::U32) // fft_h * fft_w
        .max_threads_per_block(ELEMENTWISE_BLOCK_SIZE)
        .body(move |b| {
            b.comment(&format!(
                "2D zero-pad: fft_h={fft_h}, fft_w={fft_w}, total={total_elements}"
            ));

            let tid = b.thread_id_x();
            let bid = b.block_id_x();
            let bsz = b.block_dim_x();
            let global_tid = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {global_tid}, {bid}, {bsz}, {tid};"));

            // Bounds check
            let total_count = b.load_param_u32("total_count");
            let pred_exit = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!(
                "setp.ge.u32 {pred_exit}, {global_tid}, {total_count};"
            ));
            b.raw_ptx(&format!("@{pred_exit} bra $L_2d_exit;"));

            // Compute row, col from linear index
            let out_w = b.load_param_u32("out_w");
            let row = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("div.u32 {row}, {global_tid}, {out_w};"));
            let col = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("rem.u32 {col}, {global_tid}, {out_w};"));

            // Output byte offset
            let out_offset_u32 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "mul.lo.u32 {out_offset_u32}, {global_tid}, {elem_bytes};"
            ));
            let out_offset = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("cvt.u64.u32 {out_offset}, {out_offset_u32};"));

            let out_base = b.load_param_u64("output_ptr");
            let out_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {out_addr}, {out_base}, {out_offset};"));

            // Check if (row, col) is within input bounds
            let input_h = b.load_param_u32("input_h");
            let input_w = b.load_param_u32("input_w");

            let pred_row = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lt.u32 {pred_row}, {row}, {input_h};"));
            let pred_col = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lt.u32 {pred_col}, {col}, {input_w};"));
            let pred_in_bounds = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!(
                "and.pred {pred_in_bounds}, {pred_row}, {pred_col};"
            ));
            b.raw_ptx(&format!("@!{pred_in_bounds} bra $L_2d_zero;"));

            // Compute input linear index: row * input_w + col
            let in_linear = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {in_linear}, {row}, {input_w}, {col};"));
            let in_offset_u32 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "mul.lo.u32 {in_offset_u32}, {in_linear}, {elem_bytes};"
            ));
            let in_offset = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("cvt.u64.u32 {in_offset}, {in_offset_u32};"));

            let in_base = b.load_param_u64("input_ptr");
            let in_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {in_addr}, {in_base}, {in_offset};"));

            let val = b.alloc_reg(float_ty);
            b.raw_ptx(&format!("ld.global.{suffix} {val}, [{in_addr}];"));
            b.raw_ptx(&format!("st.global.{suffix} [{out_addr}], {val};"));
            b.raw_ptx("bra $L_2d_exit;");

            b.raw_ptx("$L_2d_zero:");
            let zero_val = b.alloc_reg(float_ty);
            match precision {
                FftPrecision::Single => {
                    b.raw_ptx(&format!("mov.b32 {zero_val}, 0F00000000;"));
                }
                FftPrecision::Double => {
                    b.raw_ptx(&format!("mov.b64 {zero_val}, 0D0000000000000000;"));
                }
            }
            b.raw_ptx(&format!("st.global.{suffix} [{out_addr}], {zero_val};"));

            b.raw_ptx("$L_2d_exit:");
        })
        .build()?;

    Ok(PtxModule {
        source: ptx,
        entry_name: kernel_name,
        block_size: ELEMENTWISE_BLOCK_SIZE,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- next_fft_size tests --------------------------------------------------

    #[test]
    fn next_fft_size_zero() {
        assert_eq!(next_fft_size(0), 1);
    }

    #[test]
    fn next_fft_size_one() {
        assert_eq!(next_fft_size(1), 1);
    }

    #[test]
    fn next_fft_size_power_of_two() {
        assert_eq!(next_fft_size(16), 16);
        assert_eq!(next_fft_size(256), 256);
        assert_eq!(next_fft_size(1024), 1024);
    }

    #[test]
    fn next_fft_size_non_power_of_two() {
        assert_eq!(next_fft_size(3), 4);
        assert_eq!(next_fft_size(5), 8);
        assert_eq!(next_fft_size(100), 128);
        assert_eq!(next_fft_size(1000), 1024);
        assert_eq!(next_fft_size(1025), 2048);
    }

    // -- ConvolutionMode output_len tests -------------------------------------

    #[test]
    fn output_len_full_mode() {
        // Full: m + n - 1
        let plan = FftConvPlan::new(128, 32, ConvolutionMode::Full, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert_eq!(p.output_len(), 128 + 32 - 1);
        }
    }

    #[test]
    fn output_len_same_mode() {
        // Same: max(m, n)
        let plan = FftConvPlan::new(128, 32, ConvolutionMode::Same, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert_eq!(p.output_len(), 128);
        }
    }

    #[test]
    fn output_len_valid_mode() {
        // Valid: max(m, n) - min(m, n) + 1
        let plan = FftConvPlan::new(128, 32, ConvolutionMode::Valid, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert_eq!(p.output_len(), 128 - 32 + 1);
        }
    }

    #[test]
    fn output_len_valid_mode_equal_sizes() {
        let plan = FftConvPlan::new(64, 64, ConvolutionMode::Valid, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert_eq!(p.output_len(), 1);
        }
    }

    // -- FFT padding size tests -----------------------------------------------

    #[test]
    fn fft_padding_size() {
        // signal=100, kernel=50 => full=149 => next_pow2=256
        let plan = FftConvPlan::new(100, 50, ConvolutionMode::Full, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert_eq!(p.fft_len(), 256);
        }
    }

    #[test]
    fn fft_padding_exact_power_of_two() {
        // signal=64, kernel=1 => full=64 => next_pow2=64
        let plan = FftConvPlan::new(64, 1, ConvolutionMode::Full, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert_eq!(p.fft_len(), 64);
        }
    }

    // -- PTX generation tests -------------------------------------------------

    #[test]
    fn ptx_pointwise_multiply_generates() {
        let plan = FftConvPlan::new(256, 64, ConvolutionMode::Full, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let ptx = p.generate_pointwise_multiply_ptx();
            assert!(ptx.is_ok());
            if let Ok(module) = ptx {
                assert!(!module.source.is_empty());
                assert!(module.entry_name.contains("pointwise_mul"));
                assert!(module.source.contains(".entry"));
            }
        }
    }

    #[test]
    fn ptx_zero_pad_generates() {
        let plan = FftConvPlan::new(128, 32, ConvolutionMode::Full, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let ptx = p.generate_zero_pad_ptx();
            assert!(ptx.is_ok());
            if let Ok(module) = ptx {
                assert!(!module.source.is_empty());
                assert!(module.entry_name.contains("zero_pad"));
            }
        }
    }

    #[test]
    fn ptx_pointwise_multiply_double_precision() {
        let plan = FftConvPlan::new(512, 128, ConvolutionMode::Full, FftPrecision::Double);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let ptx = p.generate_pointwise_multiply_ptx();
            assert!(ptx.is_ok());
            if let Ok(module) = ptx {
                assert!(module.entry_name.contains("f64"));
                assert!(module.source.contains("f64"));
            }
        }
    }

    // -- Cross-correlation tests ----------------------------------------------

    #[test]
    fn cross_correlation_plan_creation() {
        let plan = CrossCorrelationPlan::new(256, 64, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert_eq!(p.output_len(), 256 + 64 - 1);
            assert_eq!(p.fft_len(), 512);
        }
    }

    #[test]
    fn cross_correlation_conj_multiply_ptx() {
        let plan = CrossCorrelationPlan::new(128, 32, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let ptx = p.generate_conj_multiply_ptx();
            assert!(ptx.is_ok());
            if let Ok(module) = ptx {
                assert!(module.entry_name.contains("conj_mul"));
                assert!(!module.source.is_empty());
                // Should contain negation for conjugation
                assert!(module.source.contains("neg"));
            }
        }
    }

    // -- 2D convolution tests -------------------------------------------------

    #[test]
    fn conv2d_plan_creation() {
        let plan = FftConv2dPlan::new(64, 64, 3, 3, ConvolutionMode::Same, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert_eq!(p.output_h(), 64);
            assert_eq!(p.output_w(), 64);
            // fft_h = next_pow2(64 + 3 - 1) = next_pow2(66) = 128
            assert_eq!(p.fft_h(), 128);
            assert_eq!(p.fft_w(), 128);
        }
    }

    #[test]
    fn conv2d_full_mode_output_dims() {
        let plan = FftConv2dPlan::new(32, 48, 5, 7, ConvolutionMode::Full, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert_eq!(p.output_h(), 32 + 5 - 1); // 36
            assert_eq!(p.output_w(), 48 + 7 - 1); // 54
        }
    }

    #[test]
    fn conv2d_ptx_pointwise_multiply_2d() {
        let plan = FftConv2dPlan::new(32, 32, 3, 3, ConvolutionMode::Full, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let ptx = p.generate_pointwise_multiply_2d_ptx();
            assert!(ptx.is_ok());
            if let Ok(module) = ptx {
                assert!(!module.source.is_empty());
                assert!(module.entry_name.contains("pointwise_mul"));
            }
        }
    }

    // -- Workspace calculation tests ------------------------------------------

    #[test]
    fn workspace_size_1d() {
        let plan = FftConvPlan::new(256, 64, ConvolutionMode::Full, FftPrecision::Single);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            // fft_len = next_pow2(256 + 64 - 1) = next_pow2(319) = 512
            // workspace = 3 * 512 * 8 (complex f32 = 8 bytes) = 12288
            assert_eq!(p.fft_len(), 512);
            assert_eq!(p.workspace_size(), 3 * 512 * 8);
        }
    }

    #[test]
    fn workspace_size_2d() {
        let plan = FftConv2dPlan::new(16, 16, 3, 3, ConvolutionMode::Full, FftPrecision::Double);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            // fft_h = next_pow2(16 + 3 - 1) = 32
            // fft_w = next_pow2(16 + 3 - 1) = 32
            // workspace = 3 * 32 * 32 * 16 (complex f64 = 16 bytes) = 49152
            assert_eq!(p.fft_h(), 32);
            assert_eq!(p.fft_w(), 32);
            assert_eq!(p.workspace_size(), 3 * 32 * 32 * 16);
        }
    }

    // -- Error handling tests -------------------------------------------------

    #[test]
    fn error_zero_signal_len() {
        let result = FftConvPlan::new(0, 32, ConvolutionMode::Full, FftPrecision::Single);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn error_zero_kernel_len() {
        let result = FftConvPlan::new(128, 0, ConvolutionMode::Full, FftPrecision::Single);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn error_cross_corr_zero_signal() {
        let result = CrossCorrelationPlan::new(0, 32, FftPrecision::Single);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn error_conv2d_zero_input() {
        let result = FftConv2dPlan::new(0, 32, 3, 3, ConvolutionMode::Full, FftPrecision::Single);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn error_conv2d_zero_kernel() {
        let result = FftConv2dPlan::new(32, 32, 0, 3, ConvolutionMode::Full, FftPrecision::Single);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }
}
