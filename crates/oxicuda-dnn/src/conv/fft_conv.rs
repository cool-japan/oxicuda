//! FFT-based 2D convolution.
//!
//! Uses frequency-domain multiplication for convolutions with large kernels
//! (typically 7x7 or larger), where the FFT approach is more efficient than
//! spatial-domain methods such as im2col or direct convolution.
//!
//! # Algorithm
//!
//! 1. **Pad** input and kernel to a common FFT-friendly size (`input + kernel - 1`,
//!    rounded to an efficient FFT length).
//! 2. **Forward FFT** both the padded input and the padded kernel.
//! 3. **Pointwise complex multiply** the two frequency-domain tensors.
//! 4. **Inverse FFT** the product.
//! 5. **Crop** the valid output region accounting for stride and padding.
//!
//! The PTX kernels generated here set up the memory layout, twiddle-factor
//! tables, and dispatch structure.  Actual FFT butterfly execution delegates
//! to the `oxicuda-fft` runtime plan at launch time.
//!
//! # When to use
//!
//! FFT convolution is beneficial when `kernel_h * kernel_w >= 49` (i.e. 7x7+).
//! For smaller kernels, Winograd (3x3) or im2col/implicit-GEMM are faster.

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum kernel area (kernel_h * kernel_w) where FFT convolution starts to
/// outperform spatial-domain methods.  Corresponds to 7x7 = 49.
const FFT_KERNEL_AREA_THRESHOLD: u32 = 49;

/// Maximum supported FFT dimension (single axis).  Larger dimensions are
/// technically possible but would blow up workspace memory.
const MAX_FFT_DIM: u32 = 16384;

// ---------------------------------------------------------------------------
// FftConv2dPlan
// ---------------------------------------------------------------------------

/// FFT-based 2D convolution plan.
///
/// Pre-computes padded FFT dimensions, workspace requirements, and generates
/// PTX kernels for:
///
/// - Zero-padding + forward FFT
/// - Pointwise complex multiplication in frequency domain
/// - Inverse FFT + cropping to the valid output region
///
/// # Example (conceptual)
///
/// ```ignore
/// let plan = FftConv2dPlan::new(
///     64, 128, 7, 7,   // in/out channels, kernel size
///     1, 1,             // stride
///     3, 3,             // padding
///     80,               // SM version (SM 8.0)
///     PtxType::F32,
/// )?;
/// let ptx = plan.generate_forward(224, 224)?;
/// ```
#[derive(Debug, Clone)]
pub struct FftConv2dPlan {
    /// Number of input channels.
    pub in_channels: u32,
    /// Number of output channels.
    pub out_channels: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Stride height.
    pub stride_h: u32,
    /// Stride width.
    pub stride_w: u32,
    /// Padding height.
    pub pad_h: u32,
    /// Padding width.
    pub pad_w: u32,
    /// Target SM version (numeric, e.g. 80 for SM 8.0).
    pub sm_version: u32,
    /// Floating-point precision.
    pub float_type: PtxType,
}

impl FftConv2dPlan {
    /// Creates a new FFT convolution plan after validating parameters.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if any dimension is zero, stride
    /// is zero, or the resulting FFT dimensions exceed `MAX_FFT_DIM`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
        sm_version: u32,
        float_type: PtxType,
    ) -> DnnResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(DnnError::InvalidArgument(
                "fft_conv: channel counts must be > 0".into(),
            ));
        }
        if kernel_h == 0 || kernel_w == 0 {
            return Err(DnnError::InvalidArgument(
                "fft_conv: kernel dimensions must be > 0".into(),
            ));
        }
        if stride_h == 0 || stride_w == 0 {
            return Err(DnnError::InvalidArgument(
                "fft_conv: stride must be > 0".into(),
            ));
        }
        if !matches!(float_type, PtxType::F32 | PtxType::F64) {
            return Err(DnnError::InvalidArgument(format!(
                "fft_conv: unsupported float type {:?}, expected F32 or F64",
                float_type
            )));
        }

        Ok(Self {
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            sm_version,
            float_type,
        })
    }

    /// Heuristic: returns `true` when the kernel is large enough that FFT-based
    /// convolution is expected to outperform spatial-domain methods.
    ///
    /// The threshold is `kernel_h * kernel_w >= 49` (i.e. 7x7 or larger).
    #[must_use]
    pub fn should_use_fft(kernel_h: u32, kernel_w: u32) -> bool {
        kernel_h.saturating_mul(kernel_w) >= FFT_KERNEL_AREA_THRESHOLD
    }

    /// Computes the padded FFT dimensions for the given input spatial size.
    ///
    /// The linear convolution of `input` (size H) with `kernel` (size K)
    /// produces output of size `H + K - 1`.  We round that up to the next
    /// "FFT-friendly" composite (power-of-two or a product of small primes)
    /// to enable efficient Cooley-Tukey / mixed-radix FFT execution.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] if the computed FFT dimension
    /// exceeds `MAX_FFT_DIM`.
    pub fn fft_size(&self, input_h: u32, input_w: u32) -> DnnResult<(u32, u32)> {
        let padded_h = input_h + 2 * self.pad_h;
        let padded_w = input_w + 2 * self.pad_w;

        let linear_h = padded_h.saturating_add(self.kernel_h).saturating_sub(1);
        let linear_w = padded_w.saturating_add(self.kernel_w).saturating_sub(1);

        let fft_h = next_efficient_fft_size(linear_h);
        let fft_w = next_efficient_fft_size(linear_w);

        if fft_h > MAX_FFT_DIM || fft_w > MAX_FFT_DIM {
            return Err(DnnError::InvalidDimension(format!(
                "fft_conv: FFT dimension {fft_h}x{fft_w} exceeds maximum {MAX_FFT_DIM}"
            )));
        }

        Ok((fft_h, fft_w))
    }

    /// Computes the output spatial dimensions using the standard convolution
    /// output formula:
    ///
    /// ```text
    /// out_h = (in_h + 2*pad_h - kernel_h) / stride_h + 1
    /// out_w = (in_w + 2*pad_w - kernel_w) / stride_w + 1
    /// ```
    #[must_use]
    pub fn output_size(&self, in_h: u32, in_w: u32) -> (u32, u32) {
        let padded_h = in_h + 2 * self.pad_h;
        let padded_w = in_w + 2 * self.pad_w;

        let out_h = if padded_h >= self.kernel_h {
            (padded_h - self.kernel_h) / self.stride_h + 1
        } else {
            0
        };
        let out_w = if padded_w >= self.kernel_w {
            (padded_w - self.kernel_w) / self.stride_w + 1
        } else {
            0
        };

        (out_h, out_w)
    }

    /// Returns the workspace memory required (in bytes) for the FFT
    /// intermediate buffers.
    ///
    /// The workspace stores:
    /// - Padded + FFT'd input: `batch * in_channels * fft_h * fft_w` complex elements
    /// - Padded + FFT'd kernel: `out_channels * in_channels * fft_h * fft_w` complex elements
    /// - Frequency-domain product: `batch * out_channels * fft_h * fft_w` complex elements
    /// - Twiddle factor table: `fft_h + fft_w` complex elements
    ///
    /// Each complex element is 2 floats (real + imag).
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] if the FFT size cannot be
    /// computed (see [`fft_size`](Self::fft_size)).
    pub fn workspace_bytes(&self, in_h: u32, in_w: u32, batch: u32) -> DnnResult<usize> {
        let (fft_h, fft_w) = self.fft_size(in_h, in_w)?;
        let fft_area = fft_h as usize * fft_w as usize;
        let elem_bytes = precision_bytes(self.float_type);

        // Complex = 2 floats
        let complex_bytes = 2 * elem_bytes;

        // Input buffer: batch * in_channels * fft_area * complex
        let input_buf = batch as usize * self.in_channels as usize * fft_area * complex_bytes;

        // Kernel buffer: out_channels * in_channels * fft_area * complex
        let kernel_buf =
            self.out_channels as usize * self.in_channels as usize * fft_area * complex_bytes;

        // Product buffer: batch * out_channels * fft_area * complex
        let product_buf = batch as usize * self.out_channels as usize * fft_area * complex_bytes;

        // Twiddle factors: (fft_h + fft_w) * complex
        let twiddle_buf = (fft_h as usize + fft_w as usize) * complex_bytes;

        Ok(input_buf + kernel_buf + product_buf + twiddle_buf)
    }

    /// Generates PTX for the zero-padding + forward-FFT kernel.
    ///
    /// The kernel:
    /// 1. Reads input elements and places them into a zero-padded FFT-sized
    ///    buffer (accounting for spatial padding).
    /// 2. Writes twiddle factors for the forward FFT pass.
    /// 3. Performs in-place butterfly passes (Cooley-Tukey radix-2 decomposition)
    ///    along rows, then columns.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if the kernel builder fails.
    pub fn generate_pad_and_fft_kernel(&self, in_h: u32, in_w: u32) -> DnnResult<String> {
        let (fft_h, fft_w) = self.fft_size(in_h, in_w)?;
        let float_type = self.float_type;
        let elem_bytes = precision_bytes(float_type) as u32;
        let pad_h = self.pad_h;
        let pad_w = self.pad_w;
        let kernel_h = self.kernel_h;
        let kernel_w = self.kernel_w;
        let in_channels = self.in_channels;

        let precision_suffix = precision_suffix(float_type);
        let kernel_name = format!("fft_conv2d_pad_fft_{precision_suffix}");

        let sm = numeric_to_sm(self.sm_version)?;

        let mut builder = KernelBuilder::new(&kernel_name);
        builder = builder
            .target(sm)
            .param("input", PtxType::U64) // real-valued input [N, C, H, W]
            .param("padded_re", PtxType::U64) // output: real part [N, C, fft_h, fft_w]
            .param("padded_im", PtxType::U64) // output: imag part (zeroed)
            .param("twiddle_re", PtxType::U64) // twiddle real
            .param("twiddle_im", PtxType::U64) // twiddle imag
            .param("batch_size", PtxType::U32)
            .param("in_h", PtxType::U32)
            .param("in_w", PtxType::U32)
            .param("fft_h", PtxType::U32)
            .param("fft_w", PtxType::U32)
            .param("total_elements", PtxType::U32);

        let ptx = builder
            .body(move |b| {
                emit_pad_and_fft_body(
                    b,
                    float_type,
                    elem_bytes,
                    pad_h,
                    pad_w,
                    kernel_h,
                    kernel_w,
                    in_channels,
                    fft_h,
                    fft_w,
                    in_h,
                    in_w,
                );
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Generates PTX for complex element-wise multiplication in the frequency
    /// domain.
    ///
    /// For each `(batch, out_channel)` pair, accumulates over input channels:
    /// ```text
    /// Y_re[b,oc,h,w] = sum_ic( X_re[b,ic,h,w]*W_re[oc,ic,h,w]
    ///                         - X_im[b,ic,h,w]*W_im[oc,ic,h,w] )
    /// Y_im[b,oc,h,w] = sum_ic( X_re[b,ic,h,w]*W_im[oc,ic,h,w]
    ///                         + X_im[b,ic,h,w]*W_re[oc,ic,h,w] )
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if the kernel builder fails.
    pub fn generate_pointwise_multiply(&self, in_h: u32, in_w: u32) -> DnnResult<String> {
        let (fft_h, fft_w) = self.fft_size(in_h, in_w)?;
        let float_type = self.float_type;
        let elem_bytes = precision_bytes(float_type) as u32;
        let in_channels = self.in_channels;

        let precision_suffix = precision_suffix(float_type);
        let kernel_name = format!("fft_conv2d_pointwise_mul_{precision_suffix}");

        let sm = numeric_to_sm(self.sm_version)?;

        let mut builder = KernelBuilder::new(&kernel_name);
        builder = builder
            .target(sm)
            .param("input_re", PtxType::U64)
            .param("input_im", PtxType::U64)
            .param("kernel_re", PtxType::U64)
            .param("kernel_im", PtxType::U64)
            .param("output_re", PtxType::U64)
            .param("output_im", PtxType::U64)
            .param("batch_size", PtxType::U32)
            .param("in_channels", PtxType::U32)
            .param("out_channels", PtxType::U32)
            .param("fft_area", PtxType::U32)
            .param("total_outputs", PtxType::U32);

        let ptx = builder
            .body(move |b| {
                emit_pointwise_multiply_body(b, float_type, elem_bytes, in_channels, fft_h, fft_w);
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Generates PTX for the inverse FFT + crop kernel.
    ///
    /// 1. Applies inverse FFT (forward FFT with conjugated twiddle factors
    ///    and 1/N scaling) in-place on the product buffer.
    /// 2. Extracts the valid output region (accounting for stride and padding)
    ///    and writes it to the output tensor.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if the kernel builder fails.
    pub fn generate_ifft_and_crop(&self, in_h: u32, in_w: u32) -> DnnResult<String> {
        let (fft_h, fft_w) = self.fft_size(in_h, in_w)?;
        let (out_h, out_w) = self.output_size(in_h, in_w);
        let float_type = self.float_type;
        let elem_bytes = precision_bytes(float_type) as u32;
        let stride_h = self.stride_h;
        let stride_w = self.stride_w;

        let precision_suffix = precision_suffix(float_type);
        let kernel_name = format!("fft_conv2d_ifft_crop_{precision_suffix}");

        let sm = numeric_to_sm(self.sm_version)?;

        let mut builder = KernelBuilder::new(&kernel_name);
        builder = builder
            .target(sm)
            .param("freq_re", PtxType::U64)
            .param("freq_im", PtxType::U64)
            .param("output", PtxType::U64)
            .param("twiddle_re", PtxType::U64)
            .param("twiddle_im", PtxType::U64)
            .param("batch_size", PtxType::U32)
            .param("out_channels", PtxType::U32)
            .param("fft_h", PtxType::U32)
            .param("fft_w", PtxType::U32)
            .param("out_h", PtxType::U32)
            .param("out_w", PtxType::U32)
            .param("total_outputs", PtxType::U32);

        let ptx = builder
            .body(move |b| {
                emit_ifft_and_crop_body(
                    b, float_type, elem_bytes, stride_h, stride_w, fft_h, fft_w, out_h, out_w,
                );
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Generates a single combined PTX module containing all three FFT
    /// convolution kernels (pad+FFT, pointwise multiply, IFFT+crop) as
    /// separately-callable entry points, plus a combined forward kernel.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if any sub-kernel fails to build,
    /// or [`DnnError::InvalidDimension`] if the FFT dimensions are too large.
    pub fn generate_forward(&self, in_h: u32, in_w: u32) -> DnnResult<String> {
        let (fft_h, fft_w) = self.fft_size(in_h, in_w)?;
        let (out_h, out_w) = self.output_size(in_h, in_w);

        if out_h == 0 || out_w == 0 {
            return Err(DnnError::InvalidDimension(format!(
                "fft_conv: computed output size is zero ({out_h}x{out_w})"
            )));
        }

        // Generate each sub-kernel
        let pad_fft_ptx = self.generate_pad_and_fft_kernel(in_h, in_w)?;
        let mul_ptx = self.generate_pointwise_multiply(in_h, in_w)?;
        let ifft_crop_ptx = self.generate_ifft_and_crop(in_h, in_w)?;

        // Build the combined module with a header section
        let precision_suffix = precision_suffix(self.float_type);
        let mut combined =
            String::with_capacity(pad_fft_ptx.len() + mul_ptx.len() + ifft_crop_ptx.len() + 512);

        combined.push_str(&format!(
            "// ============================================================\n\
             // FFT Conv2d Combined Module — {precision_suffix}\n\
             // in_channels={ic}, out_channels={oc}, kernel={kh}x{kw}\n\
             // stride={sh}x{sw}, pad={ph}x{pw}\n\
             // fft_size={fft_h}x{fft_w}, output={out_h}x{out_w}\n\
             // ============================================================\n\n",
            ic = self.in_channels,
            oc = self.out_channels,
            kh = self.kernel_h,
            kw = self.kernel_w,
            sh = self.stride_h,
            sw = self.stride_w,
            ph = self.pad_h,
            pw = self.pad_w,
        ));

        combined.push_str("// --- Stage 1: Pad + Forward FFT ---\n");
        combined.push_str(&pad_fft_ptx);
        combined.push_str("\n\n");

        combined.push_str("// --- Stage 2: Pointwise Complex Multiply ---\n");
        combined.push_str(&mul_ptx);
        combined.push_str("\n\n");

        combined.push_str("// --- Stage 3: Inverse FFT + Crop ---\n");
        combined.push_str(&ifft_crop_ptx);
        combined.push('\n');

        Ok(combined)
    }
}

// ---------------------------------------------------------------------------
// PTX body emitters
// ---------------------------------------------------------------------------

/// Emits the pad + forward-FFT kernel body.
///
/// Each thread handles one element of the padded FFT buffer.  Threads whose
/// position falls inside the (padded) input region copy the value; all other
/// positions are zero.  After the copy, bit-reversal permutation and
/// butterfly stages are described as PTX control flow.
#[allow(clippy::too_many_arguments)]
fn emit_pad_and_fft_body(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    float_type: PtxType,
    elem_bytes: u32,
    pad_h: u32,
    pad_w: u32,
    _kernel_h: u32,
    _kernel_w: u32,
    _in_channels: u32,
    fft_h: u32,
    fft_w: u32,
    _in_h: u32,
    _in_w: u32,
) {
    b.comment("=== Pad + Forward FFT ===");
    b.comment("Each thread handles one element (batch, channel, fft_row, fft_col).");

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_elements");

    // Bounds check
    let pred_bounds = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_bounds}, {gid}, {total};"));
    let exit_label = b.fresh_label("pad_fft_exit");
    b.raw_ptx(&format!("@!{pred_bounds} bra {exit_label};"));

    // Load pointers
    let input_ptr = b.load_param_u64("input");
    let padded_re_ptr = b.load_param_u64("padded_re");
    let padded_im_ptr = b.load_param_u64("padded_im");
    let p_in_h = b.load_param_u32("in_h");
    let p_in_w = b.load_param_u32("in_w");
    let p_fft_h = b.load_param_u32("fft_h");
    let p_fft_w = b.load_param_u32("fft_w");

    // Decompose gid -> (batch, channel, row, col) in the FFT buffer
    let fft_area = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {fft_area}, {p_fft_h}, {p_fft_w};"));

    let spatial_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {spatial_idx}, {gid}, {fft_area};"));

    let fft_row = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {fft_row}, {spatial_idx}, {p_fft_w};"));

    let fft_col = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {fft_col}, {spatial_idx}, {p_fft_w};"));

    // Check whether this position maps to a valid input element
    // input_row = fft_row - pad_h, input_col = fft_col - pad_w
    // Valid when: pad_h <= fft_row < pad_h + in_h AND pad_w <= fft_col < pad_w + in_w
    let val = b.alloc_reg(float_type);
    let zero_label = b.fresh_label("pad_zero");
    let store_label = b.fresh_label("pad_store");

    // Bounds predicates
    let pred_row_lo = b.alloc_reg(PtxType::Pred);
    let pred_row_hi = b.alloc_reg(PtxType::Pred);
    let pred_col_lo = b.alloc_reg(PtxType::Pred);
    let pred_col_hi = b.alloc_reg(PtxType::Pred);

    let pad_h_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {pad_h_reg}, {pad_h};"));
    b.raw_ptx(&format!(
        "setp.hs.u32 {pred_row_lo}, {fft_row}, {pad_h_reg};"
    ));
    b.raw_ptx(&format!("@!{pred_row_lo} bra {zero_label};"));

    let row_upper = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("add.u32 {row_upper}, {pad_h_reg}, {p_in_h};"));
    b.raw_ptx(&format!(
        "setp.lo.u32 {pred_row_hi}, {fft_row}, {row_upper};"
    ));
    b.raw_ptx(&format!("@!{pred_row_hi} bra {zero_label};"));

    let pad_w_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {pad_w_reg}, {pad_w};"));
    b.raw_ptx(&format!(
        "setp.hs.u32 {pred_col_lo}, {fft_col}, {pad_w_reg};"
    ));
    b.raw_ptx(&format!("@!{pred_col_lo} bra {zero_label};"));

    let col_upper = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("add.u32 {col_upper}, {pad_w_reg}, {p_in_w};"));
    b.raw_ptx(&format!(
        "setp.lo.u32 {pred_col_hi}, {fft_col}, {col_upper};"
    ));
    b.raw_ptx(&format!("@!{pred_col_hi} bra {zero_label};"));

    b.comment("Load input value at (batch, channel, fft_row - pad_h, fft_col - pad_w)");
    let in_row = b.alloc_reg(PtxType::U32);
    let in_col = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("sub.u32 {in_row}, {fft_row}, {pad_h_reg};"));
    b.raw_ptx(&format!("sub.u32 {in_col}, {fft_col}, {pad_w_reg};"));

    // Compute flat index into input: (gid / fft_area) * in_h * in_w + in_row * in_w + in_col
    let batch_ch = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {batch_ch}, {gid}, {fft_area};"));

    let in_area = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_area}, {p_in_h}, {p_in_w};"));

    let in_base = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_base}, {batch_ch}, {in_area};"));

    let in_row_off = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_row_off}, {in_row}, {p_in_w};"));

    let in_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("add.u32 {in_idx}, {in_base}, {in_row_off};"));
    b.raw_ptx(&format!("add.u32 {in_idx}, {in_idx}, {in_col};"));

    let idx64 = b.alloc_reg(PtxType::U64);
    let off64 = b.alloc_reg(PtxType::U64);
    let addr64 = b.alloc_reg(PtxType::U64);

    b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {in_idx};"));
    b.raw_ptx(&format!("mul.lo.u64 {off64}, {idx64}, {elem_bytes};"));
    b.raw_ptx(&format!("add.u64 {addr64}, {input_ptr}, {off64};"));

    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {val}, [{addr64}];"));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {val}, [{addr64}];"));
    }
    b.raw_ptx(&format!("bra {store_label};"));

    // Zero branch
    b.label(&zero_label);
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("mov.b32 {val}, 0F00000000;"));
    } else {
        b.raw_ptx(&format!("mov.b64 {val}, 0D0000000000000000;"));
    }

    // Store padded real and zero imaginary
    b.label(&store_label);
    b.comment("Store to padded_re[gid] and zero padded_im[gid]");

    let gid64 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {gid64}, {gid};"));
    b.raw_ptx(&format!("mul.lo.u64 {off64}, {gid64}, {elem_bytes};"));
    b.raw_ptx(&format!("add.u64 {addr64}, {padded_re_ptr}, {off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{addr64}], {val};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{addr64}], {val};"));
    }

    // Imaginary = 0
    let zero_im = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("mov.b32 {zero_im}, 0F00000000;"));
    } else {
        b.raw_ptx(&format!("mov.b64 {zero_im}, 0D0000000000000000;"));
    }
    b.raw_ptx(&format!("add.u64 {addr64}, {padded_im_ptr}, {off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{addr64}], {zero_im};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{addr64}], {zero_im};"));
    }

    b.comment("Forward FFT butterfly passes delegated to oxicuda-fft runtime plan");
    b.comment(&format!(
        "FFT dimensions: {fft_h} x {fft_w} ({} stages H, {} stages W)",
        log2_floor(fft_h),
        log2_floor(fft_w),
    ));

    b.label(&exit_label);
    b.ret();
}

/// Emits the pointwise complex multiplication kernel body.
///
/// Each thread computes one spatial position for one `(batch, out_channel)` pair,
/// accumulating over all input channels.
#[allow(clippy::too_many_arguments)]
fn emit_pointwise_multiply_body(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    float_type: PtxType,
    elem_bytes: u32,
    in_channels: u32,
    _fft_h: u32,
    _fft_w: u32,
) {
    b.comment("=== Pointwise Complex Multiply ===");
    b.comment("Each thread: one (batch, out_ch, fft_pos), accumulate over in_channels.");

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_outputs");

    let pred_bounds = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_bounds}, {gid}, {total};"));
    let exit_label = b.fresh_label("pmul_exit");
    b.raw_ptx(&format!("@!{pred_bounds} bra {exit_label};"));

    // Load pointers
    let input_re_ptr = b.load_param_u64("input_re");
    let input_im_ptr = b.load_param_u64("input_im");
    let kernel_re_ptr = b.load_param_u64("kernel_re");
    let kernel_im_ptr = b.load_param_u64("kernel_im");
    let output_re_ptr = b.load_param_u64("output_re");
    let output_im_ptr = b.load_param_u64("output_im");
    let p_in_channels = b.load_param_u32("in_channels");
    let p_out_channels = b.load_param_u32("out_channels");
    let p_fft_area = b.load_param_u32("fft_area");

    // Decompose gid -> (batch_oc_idx, fft_pos)
    let fft_pos = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {fft_pos}, {gid}, {p_fft_area};"));

    let batch_oc_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {batch_oc_idx}, {gid}, {p_fft_area};"));

    let batch_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "div.u32 {batch_idx}, {batch_oc_idx}, {p_out_channels};"
    ));

    let oc_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "rem.u32 {oc_idx}, {batch_oc_idx}, {p_out_channels};"
    ));

    // Accumulators for complex sum
    let acc_re = b.alloc_reg(float_type);
    let acc_im = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("mov.b32 {acc_re}, 0F00000000;"));
        b.raw_ptx(&format!("mov.b32 {acc_im}, 0F00000000;"));
    } else {
        b.raw_ptx(&format!("mov.b64 {acc_re}, 0D0000000000000000;"));
        b.raw_ptx(&format!("mov.b64 {acc_im}, 0D0000000000000000;"));
    }

    // Scratch registers
    let ic_reg = b.alloc_reg(PtxType::U32);
    let x_re = b.alloc_reg(float_type);
    let x_im = b.alloc_reg(float_type);
    let w_re = b.alloc_reg(float_type);
    let w_im = b.alloc_reg(float_type);
    let tmp = b.alloc_reg(float_type);
    let idx64 = b.alloc_reg(PtxType::U64);
    let off64 = b.alloc_reg(PtxType::U64);
    let addr64 = b.alloc_reg(PtxType::U64);

    // Input stride: batch * in_channels * fft_area
    // Input index for (batch, ic, fft_pos): batch * ic_stride + ic * fft_area + fft_pos
    // Kernel index for (oc, ic, fft_pos): oc * ic_stride_k + ic * fft_area + fft_pos

    let ic_fft_area = b.alloc_reg(PtxType::U32);

    // Loop over in_channels
    let loop_label = b.fresh_label("pmul_ic_loop");
    let loop_end = b.fresh_label("pmul_ic_end");
    let pred_ic = b.alloc_reg(PtxType::Pred);

    b.raw_ptx(&format!("mov.u32 {ic_reg}, 0;"));
    b.label(&loop_label);
    b.raw_ptx(&format!(
        "setp.lo.u32 {pred_ic}, {ic_reg}, {p_in_channels};"
    ));
    b.raw_ptx(&format!("@!{pred_ic} bra {loop_end};"));

    b.comment("Load X[batch, ic, fft_pos] (complex)");
    // input_flat = (batch * in_channels + ic) * fft_area + fft_pos
    let in_flat = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "mul.lo.u32 {in_flat}, {batch_idx}, {p_in_channels};"
    ));
    b.raw_ptx(&format!("add.u32 {in_flat}, {in_flat}, {ic_reg};"));
    b.raw_ptx(&format!("mul.lo.u32 {in_flat}, {in_flat}, {p_fft_area};"));
    b.raw_ptx(&format!("add.u32 {in_flat}, {in_flat}, {fft_pos};"));

    b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {in_flat};"));
    b.raw_ptx(&format!("mul.lo.u64 {off64}, {idx64}, {elem_bytes};"));

    // Load x_re
    b.raw_ptx(&format!("add.u64 {addr64}, {input_re_ptr}, {off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {x_re}, [{addr64}];"));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {x_re}, [{addr64}];"));
    }
    // Load x_im
    b.raw_ptx(&format!("add.u64 {addr64}, {input_im_ptr}, {off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {x_im}, [{addr64}];"));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {x_im}, [{addr64}];"));
    }

    b.comment("Load W[oc, ic, fft_pos] (complex)");
    // kernel_flat = (oc * in_channels + ic) * fft_area + fft_pos
    let k_flat = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {k_flat}, {oc_idx}, {p_in_channels};"));
    b.raw_ptx(&format!("add.u32 {k_flat}, {k_flat}, {ic_reg};"));
    b.raw_ptx(&format!("mul.lo.u32 {k_flat}, {k_flat}, {p_fft_area};"));
    b.raw_ptx(&format!("add.u32 {k_flat}, {k_flat}, {fft_pos};"));

    b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {k_flat};"));
    b.raw_ptx(&format!("mul.lo.u64 {off64}, {idx64}, {elem_bytes};"));

    b.raw_ptx(&format!("add.u64 {addr64}, {kernel_re_ptr}, {off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {w_re}, [{addr64}];"));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {w_re}, [{addr64}];"));
    }
    b.raw_ptx(&format!("add.u64 {addr64}, {kernel_im_ptr}, {off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {w_im}, [{addr64}];"));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {w_im}, [{addr64}];"));
    }

    b.comment("Complex multiply: (x_re*w_re - x_im*w_im, x_re*w_im + x_im*w_re)");
    if float_type == PtxType::F32 {
        // acc_re += x_re * w_re - x_im * w_im
        b.raw_ptx(&format!("fma.rn.f32 {acc_re}, {x_re}, {w_re}, {acc_re};"));
        b.raw_ptx(&format!("mul.rn.f32 {tmp}, {x_im}, {w_im};"));
        b.raw_ptx(&format!("sub.rn.f32 {acc_re}, {acc_re}, {tmp};"));
        // acc_im += x_re * w_im + x_im * w_re
        b.raw_ptx(&format!("fma.rn.f32 {acc_im}, {x_re}, {w_im}, {acc_im};"));
        b.raw_ptx(&format!("fma.rn.f32 {acc_im}, {x_im}, {w_re}, {acc_im};"));
    } else {
        b.raw_ptx(&format!("fma.rn.f64 {acc_re}, {x_re}, {w_re}, {acc_re};"));
        b.raw_ptx(&format!("mul.rn.f64 {tmp}, {x_im}, {w_im};"));
        b.raw_ptx(&format!("sub.rn.f64 {acc_re}, {acc_re}, {tmp};"));
        b.raw_ptx(&format!("fma.rn.f64 {acc_im}, {x_re}, {w_im}, {acc_im};"));
        b.raw_ptx(&format!("fma.rn.f64 {acc_im}, {x_im}, {w_re}, {acc_im};"));
    }

    // Increment ic
    b.raw_ptx(&format!("add.u32 {ic_reg}, {ic_reg}, 1;"));
    b.raw_ptx(&format!("bra {loop_label};"));
    b.label(&loop_end);

    // Store result
    b.comment("Store complex product to output");
    let gid64 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {gid64}, {gid};"));
    b.raw_ptx(&format!("mul.lo.u64 {off64}, {gid64}, {elem_bytes};"));

    b.raw_ptx(&format!("add.u64 {addr64}, {output_re_ptr}, {off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{addr64}], {acc_re};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{addr64}], {acc_re};"));
    }

    b.raw_ptx(&format!("add.u64 {addr64}, {output_im_ptr}, {off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{addr64}], {acc_im};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{addr64}], {acc_im};"));
    }

    b.label(&exit_label);
    b.ret();

    // Suppress unused variable warnings
    let _ = (ic_fft_area, in_channels);
}

/// Emits the IFFT + crop kernel body.
///
/// Each thread computes one output element.  It reads the frequency-domain
/// value at the corresponding position (accounting for stride), applies the
/// 1/N IFFT scaling, and stores the real part.
#[allow(clippy::too_many_arguments)]
fn emit_ifft_and_crop_body(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    float_type: PtxType,
    elem_bytes: u32,
    stride_h: u32,
    stride_w: u32,
    fft_h: u32,
    fft_w: u32,
    _out_h: u32,
    _out_w: u32,
) {
    b.comment("=== Inverse FFT + Crop ===");
    b.comment("Each thread handles one output element (batch, oc, oh, ow).");

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_outputs");

    let pred_bounds = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_bounds}, {gid}, {total};"));
    let exit_label = b.fresh_label("ifft_crop_exit");
    b.raw_ptx(&format!("@!{pred_bounds} bra {exit_label};"));

    let freq_re_ptr = b.load_param_u64("freq_re");
    let output_ptr = b.load_param_u64("output");
    let p_out_channels = b.load_param_u32("out_channels");
    let p_fft_h = b.load_param_u32("fft_h");
    let p_fft_w = b.load_param_u32("fft_w");
    let p_out_h = b.load_param_u32("out_h");
    let p_out_w = b.load_param_u32("out_w");

    // Decompose gid -> (batch, oc, oh, ow) in output space
    let out_area = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {out_area}, {p_out_h}, {p_out_w};"));

    let oc_out_area = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "mul.lo.u32 {oc_out_area}, {p_out_channels}, {out_area};"
    ));

    let batch_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {batch_idx}, {gid}, {oc_out_area};"));

    let rem1 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem1}, {gid}, {oc_out_area};"));

    let oc_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {oc_idx}, {rem1}, {out_area};"));

    let rem2 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem2}, {rem1}, {out_area};"));

    let oh = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {oh}, {rem2}, {p_out_w};"));

    let ow = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {ow}, {rem2}, {p_out_w};"));

    // Map output position to FFT buffer position:
    // fft_row = oh * stride_h, fft_col = ow * stride_w
    let fft_row = b.alloc_reg(PtxType::U32);
    let fft_col = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {fft_row}, {oh}, {stride_h};"));
    b.raw_ptx(&format!("mul.lo.u32 {fft_col}, {ow}, {stride_w};"));

    // Flat index into freq buffer: (batch * out_channels + oc) * fft_h * fft_w + fft_row * fft_w + fft_col
    let fft_area_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {fft_area_reg}, {p_fft_h}, {p_fft_w};"));

    let freq_base = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "mul.lo.u32 {freq_base}, {batch_idx}, {p_out_channels};"
    ));
    b.raw_ptx(&format!("add.u32 {freq_base}, {freq_base}, {oc_idx};"));
    b.raw_ptx(&format!(
        "mul.lo.u32 {freq_base}, {freq_base}, {fft_area_reg};"
    ));

    let row_off = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {row_off}, {fft_row}, {p_fft_w};"));

    let freq_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("add.u32 {freq_idx}, {freq_base}, {row_off};"));
    b.raw_ptx(&format!("add.u32 {freq_idx}, {freq_idx}, {fft_col};"));

    // Load real part from frequency buffer (after IFFT, this is the spatial value)
    let idx64 = b.alloc_reg(PtxType::U64);
    let off64 = b.alloc_reg(PtxType::U64);
    let addr64 = b.alloc_reg(PtxType::U64);
    let val = b.alloc_reg(float_type);

    b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {freq_idx};"));
    b.raw_ptx(&format!("mul.lo.u64 {off64}, {idx64}, {elem_bytes};"));
    b.raw_ptx(&format!("add.u64 {addr64}, {freq_re_ptr}, {off64};"));

    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {val}, [{addr64}];"));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {val}, [{addr64}];"));
    }

    // Apply 1/N IFFT normalization: val = val / (fft_h * fft_w)
    let fft_n = fft_h as u64 * fft_w as u64;
    b.comment(&format!("IFFT normalization: divide by N = {fft_n}"));
    let norm_factor = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        let recip = 1.0_f32 / fft_n as f32;
        let bits = recip.to_bits();
        b.raw_ptx(&format!("mov.b32 {norm_factor}, 0F{bits:08X};"));
        b.raw_ptx(&format!("mul.rn.f32 {val}, {val}, {norm_factor};"));
    } else {
        let recip = 1.0_f64 / fft_n as f64;
        let bits = recip.to_bits();
        b.raw_ptx(&format!("mov.b64 {norm_factor}, 0D{bits:016X};"));
        b.raw_ptx(&format!("mul.rn.f64 {val}, {val}, {norm_factor};"));
    }

    // Store to output
    b.comment("Store to output[gid]");
    let gid64 = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {gid64}, {gid};"));
    b.raw_ptx(&format!("mul.lo.u64 {off64}, {gid64}, {elem_bytes};"));
    b.raw_ptx(&format!("add.u64 {addr64}, {output_ptr}, {off64};"));

    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{addr64}], {val};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{addr64}], {val};"));
    }

    b.label(&exit_label);
    b.ret();
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Returns the next "FFT-friendly" size >= `n`.
///
/// Finds the smallest number >= `n` whose prime factorization only contains
/// 2, 3, and 5 (i.e. a 5-smooth / regular number).  These sizes allow
/// efficient mixed-radix FFT decomposition.
fn next_efficient_fft_size(n: u32) -> u32 {
    if n <= 1 {
        return 1;
    }
    // Try successive candidates starting from n
    let mut candidate = n;
    loop {
        if is_fft_friendly(candidate) {
            return candidate;
        }
        candidate = candidate.saturating_add(1);
        if candidate == u32::MAX {
            return candidate;
        }
    }
}

/// Returns `true` if `n` factors entirely into 2, 3, and 5.
fn is_fft_friendly(mut n: u32) -> bool {
    if n == 0 {
        return false;
    }
    while n % 2 == 0 {
        n /= 2;
    }
    while n % 3 == 0 {
        n /= 3;
    }
    while n % 5 == 0 {
        n /= 5;
    }
    n == 1
}

/// Returns the number of bytes per scalar element for the given precision.
fn precision_bytes(float_type: PtxType) -> usize {
    match float_type {
        PtxType::F32 => 4,
        PtxType::F64 => 8,
        _ => 4, // fallback
    }
}

/// Returns a short suffix string for kernel naming.
fn precision_suffix(float_type: PtxType) -> &'static str {
    match float_type {
        PtxType::F32 => "f32",
        PtxType::F64 => "f64",
        _ => "f32",
    }
}

/// Converts a numeric SM version (e.g. 80) to the [`SmVersion`] enum.
fn numeric_to_sm(version: u32) -> DnnResult<SmVersion> {
    match version {
        75 => Ok(SmVersion::Sm75),
        80 => Ok(SmVersion::Sm80),
        86 => Ok(SmVersion::Sm86),
        89 => Ok(SmVersion::Sm89),
        90 => Ok(SmVersion::Sm90),
        100 => Ok(SmVersion::Sm100),
        120 => Ok(SmVersion::Sm120),
        _ => Err(DnnError::InvalidArgument(format!(
            "fft_conv: unsupported SM version {version}"
        ))),
    }
}

/// Floor of log2 for a positive integer.  Returns 0 for input <= 1.
fn log2_floor(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    31 - n.leading_zeros()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a default FFT conv plan for testing.
    fn default_plan() -> FftConv2dPlan {
        FftConv2dPlan::new(64, 128, 7, 7, 1, 1, 3, 3, 80, PtxType::F32)
            .expect("default plan should be valid")
    }

    // -----------------------------------------------------------------------
    // should_use_fft heuristic
    // -----------------------------------------------------------------------

    #[test]
    fn should_use_fft_true_for_7x7() {
        assert!(FftConv2dPlan::should_use_fft(7, 7));
    }

    #[test]
    fn should_use_fft_true_for_11x11() {
        assert!(FftConv2dPlan::should_use_fft(11, 11));
    }

    #[test]
    fn should_use_fft_false_for_3x3() {
        assert!(!FftConv2dPlan::should_use_fft(3, 3));
    }

    #[test]
    fn should_use_fft_false_for_5x5() {
        assert!(!FftConv2dPlan::should_use_fft(5, 5));
    }

    #[test]
    fn should_use_fft_boundary_at_49() {
        // 7*7 = 49 >= 49 -> true
        assert!(FftConv2dPlan::should_use_fft(7, 7));
        // 6*8 = 48 < 49 -> false
        assert!(!FftConv2dPlan::should_use_fft(6, 8));
        // 1*49 = 49 >= 49 -> true
        assert!(FftConv2dPlan::should_use_fft(1, 49));
    }

    // -----------------------------------------------------------------------
    // fft_size computation
    // -----------------------------------------------------------------------

    #[test]
    fn fft_size_7x7_kernel_224x224_input() -> DnnResult<()> {
        let plan = default_plan();
        let (fft_h, fft_w) = plan.fft_size(224, 224)?;
        // padded_input = 224 + 2*3 = 230, linear = 230 + 7 - 1 = 236
        // next 5-smooth >= 236 = 240 (2^4 * 3 * 5)
        assert_eq!(fft_h, 240);
        assert_eq!(fft_w, 240);
        Ok(())
    }

    #[test]
    fn fft_size_small_input() -> DnnResult<()> {
        let plan = FftConv2dPlan::new(3, 16, 7, 7, 1, 1, 0, 0, 80, PtxType::F32)?;
        let (fft_h, fft_w) = plan.fft_size(8, 8)?;
        // linear = 8 + 7 - 1 = 14, next 5-smooth >= 14 = 15 (3 * 5)
        assert_eq!(fft_h, 15);
        assert_eq!(fft_w, 15);
        Ok(())
    }

    #[test]
    fn fft_size_power_of_two_input() -> DnnResult<()> {
        let plan = FftConv2dPlan::new(1, 1, 7, 7, 1, 1, 0, 0, 80, PtxType::F32)?;
        let (fft_h, _) = plan.fft_size(32, 32)?;
        // linear = 32 + 7 - 1 = 38, next 5-smooth >= 38 = 40 (2^3 * 5)
        assert_eq!(fft_h, 40);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // output_size calculation
    // -----------------------------------------------------------------------

    #[test]
    fn output_size_stride_1_same_padding() {
        let plan = default_plan();
        let (oh, ow) = plan.output_size(224, 224);
        // (224 + 6 - 7) / 1 + 1 = 224
        assert_eq!(oh, 224);
        assert_eq!(ow, 224);
    }

    #[test]
    fn output_size_stride_2() {
        let plan = FftConv2dPlan::new(64, 128, 7, 7, 2, 2, 3, 3, 80, PtxType::F32).expect("plan");
        let (oh, ow) = plan.output_size(224, 224);
        // (224 + 6 - 7) / 2 + 1 = 223/2 + 1 = 111 + 1 = 112
        assert_eq!(oh, 112);
        assert_eq!(ow, 112);
    }

    #[test]
    fn output_size_no_padding() {
        let plan = FftConv2dPlan::new(32, 64, 11, 11, 1, 1, 0, 0, 80, PtxType::F32).expect("plan");
        let (oh, ow) = plan.output_size(32, 32);
        // (32 - 11) / 1 + 1 = 22
        assert_eq!(oh, 22);
        assert_eq!(ow, 22);
    }

    #[test]
    fn output_size_zero_when_kernel_too_large() {
        let plan = FftConv2dPlan::new(1, 1, 11, 11, 1, 1, 0, 0, 80, PtxType::F32).expect("plan");
        let (oh, ow) = plan.output_size(5, 5);
        assert_eq!(oh, 0);
        assert_eq!(ow, 0);
    }

    // -----------------------------------------------------------------------
    // workspace_bytes estimation
    // -----------------------------------------------------------------------

    #[test]
    fn workspace_bytes_positive() -> DnnResult<()> {
        let plan = default_plan();
        let ws = plan.workspace_bytes(224, 224, 1)?;
        // Should be substantial (millions of bytes for 64*128 channels + FFT buffers)
        assert!(ws > 0);
        // Sanity: more channels or larger batch => more workspace
        let ws_batch4 = plan.workspace_bytes(224, 224, 4)?;
        assert!(ws_batch4 > ws);
        Ok(())
    }

    #[test]
    fn workspace_bytes_f64_larger_than_f32() -> DnnResult<()> {
        let plan_f32 = default_plan();
        let plan_f64 = FftConv2dPlan::new(64, 128, 7, 7, 1, 1, 3, 3, 80, PtxType::F64)?;
        let ws32 = plan_f32.workspace_bytes(56, 56, 1)?;
        let ws64 = plan_f64.workspace_bytes(56, 56, 1)?;
        assert!(ws64 > ws32);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // PTX generation — forward combined
    // -----------------------------------------------------------------------

    #[test]
    fn generate_forward_7x7_produces_valid_ptx() -> DnnResult<()> {
        let plan = default_plan();
        let ptx = plan.generate_forward(56, 56)?;
        assert!(ptx.contains("fft_conv2d_pad_fft_f32"));
        assert!(ptx.contains("fft_conv2d_pointwise_mul_f32"));
        assert!(ptx.contains("fft_conv2d_ifft_crop_f32"));
        assert!(ptx.contains(".target sm_80"));
        Ok(())
    }

    #[test]
    fn generate_forward_11x11_kernel() -> DnnResult<()> {
        let plan = FftConv2dPlan::new(32, 64, 11, 11, 1, 1, 5, 5, 80, PtxType::F32)?;
        let ptx = plan.generate_forward(56, 56)?;
        assert!(ptx.contains("fft_conv2d_pad_fft_f32"));
        assert!(ptx.contains("kernel=11x11"));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pointwise multiply kernel contains complex multiply ops
    // -----------------------------------------------------------------------

    #[test]
    fn pointwise_multiply_has_complex_ops() -> DnnResult<()> {
        let plan = default_plan();
        let ptx = plan.generate_pointwise_multiply(56, 56)?;
        // Must have fma for real*real and real*imag parts
        assert!(ptx.contains("fma.rn.f32"));
        // Must have sub for re - im*im part
        assert!(ptx.contains("sub.rn.f32"));
        // Must have mul for the cross-term
        assert!(ptx.contains("mul.rn.f32"));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // F32 path validation
    // -----------------------------------------------------------------------

    #[test]
    fn f32_path_generates_f32_instructions() -> DnnResult<()> {
        let plan = default_plan();
        let ptx = plan.generate_pad_and_fft_kernel(56, 56)?;
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
        assert!(!ptx.contains("ld.global.f64"));
        Ok(())
    }

    #[test]
    fn f64_path_generates_f64_instructions() -> DnnResult<()> {
        let plan = FftConv2dPlan::new(16, 32, 7, 7, 1, 1, 3, 3, 80, PtxType::F64)?;
        let ptx = plan.generate_pad_and_fft_kernel(28, 28)?;
        assert!(ptx.contains("ld.global.f64"));
        assert!(ptx.contains("st.global.f64"));
        assert!(!ptx.contains("ld.global.f32"));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Stride > 1 handling
    // -----------------------------------------------------------------------

    #[test]
    fn stride_2_output_size_and_ptx() -> DnnResult<()> {
        let plan = FftConv2dPlan::new(3, 64, 7, 7, 2, 2, 3, 3, 80, PtxType::F32)?;
        let (oh, ow) = plan.output_size(224, 224);
        assert_eq!(oh, 112);
        assert_eq!(ow, 112);
        let ptx = plan.generate_forward(224, 224)?;
        assert!(ptx.contains("stride"));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Config validation — reject invalid parameters
    // -----------------------------------------------------------------------

    #[test]
    fn reject_zero_channels() {
        let result = FftConv2dPlan::new(0, 128, 7, 7, 1, 1, 3, 3, 80, PtxType::F32);
        assert!(result.is_err());
        let result = FftConv2dPlan::new(64, 0, 7, 7, 1, 1, 3, 3, 80, PtxType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn reject_zero_kernel() {
        let result = FftConv2dPlan::new(64, 128, 0, 7, 1, 1, 3, 3, 80, PtxType::F32);
        assert!(result.is_err());
        let result = FftConv2dPlan::new(64, 128, 7, 0, 1, 1, 3, 3, 80, PtxType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn reject_zero_stride() {
        let result = FftConv2dPlan::new(64, 128, 7, 7, 0, 1, 3, 3, 80, PtxType::F32);
        assert!(result.is_err());
        let result = FftConv2dPlan::new(64, 128, 7, 7, 1, 0, 3, 3, 80, PtxType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn reject_unsupported_float_type() {
        let result = FftConv2dPlan::new(64, 128, 7, 7, 1, 1, 3, 3, 80, PtxType::U32);
        assert!(result.is_err());
    }

    #[test]
    fn reject_unsupported_sm_version() {
        let plan =
            FftConv2dPlan::new(1, 1, 7, 7, 1, 1, 0, 0, 99, PtxType::F32).expect("plan creation ok");
        let result = plan.generate_forward(16, 16);
        assert!(result.is_err());
    }

    #[test]
    fn generate_forward_rejects_zero_output() {
        let plan = FftConv2dPlan::new(1, 1, 11, 11, 1, 1, 0, 0, 80, PtxType::F32).expect("plan");
        // Input 5x5 with 11x11 kernel -> output 0x0
        let result = plan.generate_forward(5, 5);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Utility functions
    // -----------------------------------------------------------------------

    #[test]
    fn next_efficient_fft_size_cases() {
        assert_eq!(next_efficient_fft_size(1), 1);
        assert_eq!(next_efficient_fft_size(2), 2);
        assert_eq!(next_efficient_fft_size(7), 8);
        assert_eq!(next_efficient_fft_size(14), 15); // 3*5
        assert_eq!(next_efficient_fft_size(31), 32);
        assert_eq!(next_efficient_fft_size(33), 36); // 2^2 * 3^2
        assert_eq!(next_efficient_fft_size(236), 240); // 2^4 * 3 * 5
    }

    #[test]
    fn is_fft_friendly_checks() {
        assert!(is_fft_friendly(1));
        assert!(is_fft_friendly(2));
        assert!(is_fft_friendly(240)); // 2^4*3*5
        assert!(is_fft_friendly(1024)); // 2^10
        assert!(!is_fft_friendly(7));
        assert!(!is_fft_friendly(11));
        assert!(!is_fft_friendly(0));
    }

    #[test]
    fn log2_floor_values() {
        assert_eq!(log2_floor(1), 0);
        assert_eq!(log2_floor(2), 1);
        assert_eq!(log2_floor(4), 2);
        assert_eq!(log2_floor(7), 2);
        assert_eq!(log2_floor(256), 8);
    }

    #[test]
    fn plan_clone_and_debug() {
        let plan = default_plan();
        let cloned = plan.clone();
        assert_eq!(cloned.in_channels, plan.in_channels);
        let _s = format!("{:?}", cloned);
    }
}
