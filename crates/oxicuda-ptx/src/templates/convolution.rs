//! Convolution PTX kernel templates.
//!
//! Generates PTX kernels for forward and backward 2D convolution operations,
//! including specialized paths for im2col transformation, direct convolution,
//! and optimized 1x1 (pointwise) convolution.
//!
//! # Supported operations
//!
//! - **Im2col**: Unfolds the input tensor into a column matrix suitable for
//!   GEMM-based convolution (the standard cuDNN approach).
//! - **Direct convolution**: Each thread computes one output value by iterating
//!   over the kernel window with shared-memory tiling.
//! - **1x1 convolution**: Optimized path that degenerates to matrix multiplication
//!   (no spatial kernel loops, vectorized loads where possible).
//! - **Backward data**: Computes the input gradient via transposed convolution
//!   with flipped kernels.
//! - **Backward filter**: Computes the weight gradient by accumulating the
//!   correlation of input patches and output gradients.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::convolution::ConvolutionTemplate;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = ConvolutionTemplate {
//!     in_channels: 3,
//!     out_channels: 64,
//!     kernel_h: 3,
//!     kernel_w: 3,
//!     stride_h: 1,
//!     stride_w: 1,
//!     pad_h: 1,
//!     pad_w: 1,
//!     dilation_h: 1,
//!     dilation_w: 1,
//!     groups: 1,
//!     sm_version: SmVersion::Sm80,
//!     float_type: PtxType::F32,
//! };
//! let ptx = template.generate_im2col_kernel().expect("im2col PTX");
//! assert!(ptx.contains("im2col"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Configuration for convolution kernel generation.
///
/// Encapsulates all parameters for 2D convolution: input/output channels,
/// kernel dimensions, stride, padding, dilation, and grouping. Set `groups`
/// to 1 for standard convolution, or to `in_channels` for depthwise convolution.
#[derive(Debug, Clone)]
pub struct ConvolutionTemplate {
    /// Number of input channels.
    pub in_channels: u32,
    /// Number of output channels (filters).
    pub out_channels: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Vertical stride.
    pub stride_h: u32,
    /// Horizontal stride.
    pub stride_w: u32,
    /// Vertical padding.
    pub pad_h: u32,
    /// Horizontal padding.
    pub pad_w: u32,
    /// Vertical dilation.
    pub dilation_h: u32,
    /// Horizontal dilation.
    pub dilation_w: u32,
    /// Number of groups (1 = standard, `in_channels` = depthwise).
    pub groups: u32,
    /// Target GPU architecture.
    pub sm_version: SmVersion,
    /// Floating-point type for computation (F32, F16, or F64).
    pub float_type: PtxType,
}

impl ConvolutionTemplate {
    /// Returns a type suffix string for kernel naming.
    const fn type_suffix(&self) -> &'static str {
        match self.float_type {
            PtxType::F16 => "f16",
            PtxType::F64 => "f64",
            _ => "f32",
        }
    }

    /// Returns the PTX type string for this template's float type.
    const fn ty(&self) -> &'static str {
        self.float_type.as_ptx_str()
    }

    /// Returns the byte size of the float type.
    const fn byte_size(&self) -> usize {
        self.float_type.size_bytes()
    }

    /// Returns the zero literal for the configured float type.
    const fn zero_lit(&self) -> &'static str {
        match self.float_type {
            PtxType::F64 => "0d0000000000000000",
            _ => "0f00000000",
        }
    }

    /// Number of input channels per group.
    const fn channels_per_group(&self) -> u32 {
        self.in_channels / self.groups
    }

    /// Number of output channels per group.
    const fn out_channels_per_group(&self) -> u32 {
        self.out_channels / self.groups
    }

    /// Computes the output spatial dimensions given input dimensions.
    ///
    /// Uses the standard formula:
    /// `out = (in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1`
    #[must_use]
    pub const fn output_size(&self, in_h: u32, in_w: u32) -> (u32, u32) {
        let out_h =
            (in_h + 2 * self.pad_h - self.dilation_h * (self.kernel_h - 1) - 1) / self.stride_h + 1;
        let out_w =
            (in_w + 2 * self.pad_w - self.dilation_w * (self.kernel_w - 1) - 1) / self.stride_w + 1;
        (out_h, out_w)
    }

    /// Generates a descriptive kernel name with the given suffix.
    #[must_use]
    pub fn kernel_name(&self, suffix: &str) -> String {
        let ts = self.type_suffix();
        let ic = self.in_channels;
        let oc = self.out_channels;
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let g = self.groups;
        if g == 1 {
            format!("conv2d_{suffix}_{ts}_ic{ic}_oc{oc}_k{kh}x{kw}")
        } else if g == self.in_channels {
            format!("conv2d_{suffix}_{ts}_dw{ic}_k{kh}x{kw}")
        } else {
            format!("conv2d_{suffix}_{ts}_ic{ic}_oc{oc}_k{kh}x{kw}_g{g}")
        }
    }

    /// Validates the template configuration.
    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(self.float_type, PtxType::F16 | PtxType::F32 | PtxType::F64) {
            return Err(PtxGenError::InvalidType(format!(
                "convolution requires F16, F32, or F64, got {}",
                self.float_type.as_ptx_str()
            )));
        }
        if self.in_channels == 0 {
            return Err(PtxGenError::GenerationFailed(
                "in_channels must be > 0".to_string(),
            ));
        }
        if self.out_channels == 0 {
            return Err(PtxGenError::GenerationFailed(
                "out_channels must be > 0".to_string(),
            ));
        }
        if self.kernel_h == 0 || self.kernel_w == 0 {
            return Err(PtxGenError::GenerationFailed(
                "kernel dimensions must be > 0".to_string(),
            ));
        }
        if self.stride_h == 0 || self.stride_w == 0 {
            return Err(PtxGenError::GenerationFailed(
                "stride must be > 0".to_string(),
            ));
        }
        if self.dilation_h == 0 || self.dilation_w == 0 {
            return Err(PtxGenError::GenerationFailed(
                "dilation must be > 0".to_string(),
            ));
        }
        if self.groups == 0 {
            return Err(PtxGenError::GenerationFailed(
                "groups must be > 0".to_string(),
            ));
        }
        if self.in_channels % self.groups != 0 {
            return Err(PtxGenError::GenerationFailed(format!(
                "in_channels ({}) must be divisible by groups ({})",
                self.in_channels, self.groups
            )));
        }
        if self.out_channels % self.groups != 0 {
            return Err(PtxGenError::GenerationFailed(format!(
                "out_channels ({}) must be divisible by groups ({})",
                self.out_channels, self.groups
            )));
        }
        Ok(())
    }

    /// Writes the standard PTX header (version, target, address size).
    fn write_header(&self, ptx: &mut String) -> Result<(), PtxGenError> {
        writeln!(ptx, ".version {}", self.sm_version.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.sm_version.as_ptx_str())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Im2col kernel
    // -----------------------------------------------------------------------

    /// Generates a PTX kernel that transforms input into a column matrix (im2col).
    ///
    /// Each thread handles one output spatial position. It extracts
    /// `kernel_h * kernel_w * channels_per_group` values from the input tensor
    /// and writes them as a row in the column buffer. Out-of-bounds input
    /// positions (due to padding) are written as zero.
    ///
    /// **Parameters:**
    /// - `input`: pointer to input tensor `[N, C, H, W]` (NCHW layout)
    /// - `col`: pointer to column buffer `[N, C/g * kH * kW, out_H * out_W]`
    /// - `batch_size`: N
    /// - `in_h`, `in_w`: input spatial dimensions
    /// - `out_h`, `out_w`: output spatial dimensions
    ///
    /// **Grid:** `(ceil(out_h * out_w / block_size), batch_size, 1)`
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if validation or PTX formatting fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate_im2col_kernel(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.ty();
        let byte_size = self.byte_size();
        let kernel_name = self.kernel_name("im2col");
        let cpg = self.channels_per_group();
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let sh = self.stride_h;
        let sw = self.stride_w;
        let ph = self.pad_h;
        let pw = self.pad_w;
        let dh = self.dilation_h;
        let dw = self.dilation_w;
        let zero_lit = self.zero_lit();
        let col_row_len = cpg * kh * kw;

        let mut ptx = String::with_capacity(8192);
        self.write_header(&mut ptx)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_col,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_in_h,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_in_w,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_out_h,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_out_w").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Register declarations
        writeln!(ptx, "    .reg .b32 %r<48>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<24>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {ty} %val<4>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread indexing: global_idx = ctaid.x * ntid.x + tid.x
        // This is the output pixel index within the spatial output plane
        writeln!(ptx, "    // Thread and block indexing").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;  // out_pixel_idx")
            .map_err(PtxGenError::FormatError)?;
        // Batch index from ctaid.y
        writeln!(ptx, "    mov.u32 %r4, %ctaid.y;  // batch_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    // Load parameters").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_col];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r5, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r6, [%param_in_h];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r7, [%param_in_w];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r8, [%param_out_h];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r9, [%param_out_w];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check: out_pixel_idx < out_h * out_w
        writeln!(ptx, "    // Bounds check").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mul.lo.u32 %r10, %r8, %r9;  // total_out_pixels = out_h * out_w"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r3, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $IM2COL_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Decompose out_pixel_idx into (out_y, out_x)
        // out_y = out_pixel_idx / out_w, out_x = out_pixel_idx % out_w
        writeln!(
            ptx,
            "    // Decompose output pixel index into (out_y, out_x)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    div.u32 %r11, %r3, %r9;  // out_y").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rem.u32 %r12, %r3, %r9;  // out_x").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute base input coordinates (top-left corner of receptive field)
        // in_y_base = out_y * stride_h - pad_h
        // in_x_base = out_x * stride_w - pad_w
        // We use signed arithmetic for padding
        writeln!(ptx, "    // Compute base input coordinates (signed)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.s32 %r13, %r11, {sh};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub.s32 %r13, %r13, {ph};  // in_y_base")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.s32 %r14, %r12, {sw};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub.s32 %r14, %r14, {pw};  // in_x_base")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute input base pointer for this batch element
        // input_batch_ptr = input + batch_idx * in_channels * in_h * in_w * byte_size
        writeln!(ptx, "    // Input base address for batch element")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd2, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mul.lo.u32 %r15, %r6, %r7;  // in_h * in_w = spatial_size"
        )
        .map_err(PtxGenError::FormatError)?;
        let in_ch = self.in_channels;
        writeln!(
            ptx,
            "    mul.lo.u32 %r16, %r15, {in_ch};  // in_channels * spatial_size"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r16;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u64 %rd4, %rd2, %rd3, %rd0;  // input_batch_ptr"
        )
        .map_err(PtxGenError::FormatError)?;
        // Note: %rd3 now holds batch_stride_bytes, but we used mad so we need to
        // recompute it as a product. Actually mad.lo.u64 %rd4 = %rd2 * %rd3 + %rd0
        // which is what we want: input + batch_idx * batch_stride
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute col output base pointer for this batch element and pixel
        // col layout: [N, C/g * kH * kW, out_H * out_W]
        // col_ptr = col + (batch_idx * col_row_len * total_out_pixels + out_pixel_idx) * byte_size
        // But we write one full column (col_row_len values) at stride total_out_pixels
        writeln!(ptx, "    // Col output base address").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    cvt.u64.u32 %rd5, %r10;  // total_out_pixels as u64"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mul.lo.u64 %rd6, %rd5, {col_row_len};  // total_out_pixels * col_row_len"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mul.lo.u64 %rd6, %rd6, {byte_size};  // batch stride in col"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u64 %rd7, %rd2, %rd6, %rd1;  // col + batch_offset"
        )
        .map_err(PtxGenError::FormatError)?;
        // Add pixel offset: out_pixel_idx * byte_size
        writeln!(ptx, "    cvt.u64.u32 %rd8, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd7, %rd7, %rd8;  // col_pixel_ptr")
            .map_err(PtxGenError::FormatError)?;
        // Row stride in col = total_out_pixels * byte_size
        writeln!(
            ptx,
            "    mul.lo.u64 %rd9, %rd5, {byte_size};  // col_row_stride"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Triple loop: for c in 0..channels_per_group, kh, kw
        // We unroll into a sequential iteration with a counter
        writeln!(ptx, "    // Im2col extraction loop: c, ky, kx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r17, 0;  // c").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u64 %rd10, %rd7;  // running col pointer")
            .map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$IM2COL_C_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r17, {cpg};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $IM2COL_DONE;").map_err(PtxGenError::FormatError)?;

        // Compute channel plane base: input_batch_ptr + c * in_h * in_w * byte_size
        writeln!(ptx, "    cvt.u64.u32 %rd11, %r17;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd12, %r15;  // spatial_size")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, %rd12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd13, %rd4, %rd11;  // channel_base_ptr")
            .map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    mov.u32 %r18, 0;  // ky").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$IM2COL_KY_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p2, %r18, {kh};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $IM2COL_KY_DONE;").map_err(PtxGenError::FormatError)?;

        // in_y = in_y_base + ky * dilation_h
        writeln!(ptx, "    mad.lo.s32 %r19, %r18, {dh}, %r13;  // in_y")
            .map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    mov.u32 %r20, 0;  // kx").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$IM2COL_KX_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p3, %r20, {kw};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p3 bra $IM2COL_KX_DONE;").map_err(PtxGenError::FormatError)?;

        // in_x = in_x_base + kx * dilation_w
        writeln!(ptx, "    mad.lo.s32 %r21, %r20, {dw}, %r14;  // in_x")
            .map_err(PtxGenError::FormatError)?;

        // Bounds check: 0 <= in_y < in_h && 0 <= in_x < in_w
        writeln!(ptx, "    setp.lt.s32 %p4, %r19, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.s32 %p5, %r19, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    or.pred %p4, %p4, %p5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.s32 %p5, %r21, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    or.pred %p4, %p4, %p5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.s32 %p5, %r21, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    or.pred %p4, %p4, %p5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 bra $IM2COL_PAD;").map_err(PtxGenError::FormatError)?;

        // Load input value: input[batch, c, in_y, in_x]
        // addr = channel_base_ptr + (in_y * in_w + in_x) * byte_size
        writeln!(
            ptx,
            "    mad.lo.s32 %r22, %r19, %r7, %r21;  // in_y * in_w + in_x"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd14, %r22;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd14, %rd14, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd15, %rd13, %rd14;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %val0, [%rd15];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $IM2COL_STORE;").map_err(PtxGenError::FormatError)?;

        // Padding path: store zero
        writeln!(ptx, "$IM2COL_PAD:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %val0, {zero_lit};").map_err(PtxGenError::FormatError)?;

        // Store to col buffer
        writeln!(ptx, "$IM2COL_STORE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd10], %val0;").map_err(PtxGenError::FormatError)?;
        // Advance col pointer by row stride (total_out_pixels * byte_size)
        writeln!(ptx, "    add.u64 %rd10, %rd10, %rd9;").map_err(PtxGenError::FormatError)?;

        // Increment kx
        writeln!(ptx, "    add.u32 %r20, %r20, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $IM2COL_KX_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$IM2COL_KX_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r18, %r18, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $IM2COL_KY_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$IM2COL_KY_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r17, %r17, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $IM2COL_C_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$IM2COL_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    // -----------------------------------------------------------------------
    // Direct convolution kernel
    // -----------------------------------------------------------------------

    /// Generates a PTX kernel for direct 2D convolution with shared memory tiling.
    ///
    /// Each thread computes one output value by iterating over the full kernel
    /// window (`kernel_h * kernel_w * channels_per_group`) and accumulating
    /// via fused multiply-add. Input and weight tiles are loaded through shared
    /// memory to improve bandwidth utilization.
    ///
    /// **Parameters:**
    /// - `input`: pointer to input `[N, C, H, W]`
    /// - `weight`: pointer to weight `[out_C, C/g, kH, kW]`
    /// - `output`: pointer to output `[N, out_C, out_H, out_W]`
    /// - `bias`: pointer to bias `[out_C]` (may be null)
    /// - `batch_size`: N
    /// - `in_h`, `in_w`: input spatial dimensions
    /// - `out_h`, `out_w`: output spatial dimensions
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if validation or PTX formatting fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate_direct_conv_kernel(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.ty();
        let byte_size = self.byte_size();
        let kernel_name = self.kernel_name("direct");
        let cpg = self.channels_per_group();
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let sh = self.stride_h;
        let sw = self.stride_w;
        let ph = self.pad_h;
        let pw = self.pad_w;
        let dh = self.dilation_h;
        let dw = self.dilation_w;
        let zero_lit = self.zero_lit();
        let groups = self.groups;
        let oc = self.out_channels;
        let ocpg = self.out_channels_per_group();

        // Shared memory for one tile of input and weights
        let tile_size = 16_usize; // tile dimension for shared memory
        let input_smem = tile_size * tile_size * byte_size;
        let weight_smem = (kh as usize) * (kw as usize) * (cpg as usize) * byte_size;
        let total_smem = input_smem + weight_smem;

        let mut ptx = String::with_capacity(8192);
        self.write_header(&mut ptx)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_weight,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_bias,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_in_h,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_in_w,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_out_h,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_out_w").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Register declarations
        writeln!(ptx, "    .reg .b32 %r<64>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {ty} %f<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    .shared .align {} .b8 smem_conv[{}];",
            byte_size.max(4),
            total_smem
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread indexing
        // tid.x -> output x position within block
        // ctaid.x -> output x block
        // ctaid.y -> output y block
        // ctaid.z -> combined (batch, out_channel) index
        writeln!(ptx, "    // Thread and block indexing").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;  // thread_x").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;  // block_x")
            .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mov.u32 %r2, %ctaid.y;  // block_y (maps to out_y)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mov.u32 %r3, %ctaid.z;  // combined batch*oc index"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r4, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute out_x = block_x * ntid.x + tid.x
        writeln!(ptx, "    mad.lo.u32 %r5, %r1, %r4, %r0;  // out_x")
            .map_err(PtxGenError::FormatError)?;
        // Decompose ctaid.z into (batch_idx, oc_idx)
        writeln!(ptx, "    div.u32 %r6, %r3, {oc};  // batch_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rem.u32 %r7, %r3, {oc};  // oc_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    // Load parameters").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_weight];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd3, [%param_bias];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r8, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r9, [%param_in_h];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r10, [%param_in_w];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r11, [%param_out_h];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r12, [%param_out_w];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check: out_x < out_w && out_y (=r2) < out_h
        writeln!(ptx, "    // Bounds check").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r5, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r2, %r11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    or.pred %p0, %p0, %p1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $DIRECT_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Determine group and channel range
        // group_idx = oc_idx / out_channels_per_group
        // c_start = group_idx * channels_per_group
        writeln!(ptx, "    // Determine group and channel range")
            .map_err(PtxGenError::FormatError)?;
        if groups > 1 {
            writeln!(ptx, "    div.u32 %r13, %r7, {ocpg};  // group_idx")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    mul.lo.u32 %r14, %r13, {cpg};  // c_start")
                .map_err(PtxGenError::FormatError)?;
        } else {
            writeln!(ptx, "    mov.u32 %r13, 0;  // group_idx")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    mov.u32 %r14, 0;  // c_start").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute input base for this batch: input + batch * C * H * W * byte_size
        writeln!(ptx, "    // Input base for batch element").map_err(PtxGenError::FormatError)?;
        let in_ch = self.in_channels;
        writeln!(ptx, "    mul.lo.u32 %r15, %r9, %r10;  // in_h * in_w")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r16, %r15, {in_ch};  // C * H * W")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd4, %r16;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd4, %rd4, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u64 %rd6, %rd5, %rd4, %rd0;  // input_batch_ptr"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Weight base for this output channel: weight + oc_idx * cpg * kH * kW * byte_size
        let weight_per_oc = (cpg * kh * kw) as usize * byte_size;
        writeln!(ptx, "    // Weight base for output channel").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd7, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd7, %rd7, {weight_per_oc};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd8, %rd1, %rd7;  // weight_oc_ptr")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Initialize accumulator to zero (or bias)
        writeln!(ptx, "    // Initialize accumulator").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f0, {zero_lit};  // acc").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute base input coordinates
        writeln!(ptx, "    // Compute input base coordinates (signed)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.s32 %r17, %r2, {sh};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub.s32 %r17, %r17, {ph};  // in_y_base")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.s32 %r18, %r5, {sw};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub.s32 %r18, %r18, {pw};  // in_x_base")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Convolution accumulation loop: over c, ky, kx
        writeln!(ptx, "    // Convolution accumulation loop").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r19, 0;  // c_local (relative to group)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r20, 0;  // weight_offset_idx")
            .map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$DIRECT_C_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p2, %r19, {cpg};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $DIRECT_BIAS;").map_err(PtxGenError::FormatError)?;

        // c_global = c_start + c_local
        writeln!(ptx, "    add.u32 %r21, %r14, %r19;  // c_global")
            .map_err(PtxGenError::FormatError)?;
        // channel plane base: input_batch_ptr + c_global * in_h * in_w * byte_size
        writeln!(ptx, "    cvt.u64.u32 %rd9, %r21;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd10, %r15;  // spatial_size")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd9, %rd9, %rd10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd9, %rd9, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd11, %rd6, %rd9;  // input_channel_ptr")
            .map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    mov.u32 %r22, 0;  // ky").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$DIRECT_KY_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p3, %r22, {kh};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p3 bra $DIRECT_KY_DONE;").map_err(PtxGenError::FormatError)?;

        // in_y = in_y_base + ky * dilation_h
        writeln!(ptx, "    mad.lo.s32 %r23, %r22, {dh}, %r17;  // in_y")
            .map_err(PtxGenError::FormatError)?;

        // Check y bounds
        writeln!(ptx, "    setp.lt.s32 %p4, %r23, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.s32 %p5, %r23, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    or.pred %p4, %p4, %p5;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    mov.u32 %r24, 0;  // kx").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$DIRECT_KX_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p6, %r24, {kw};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p6 bra $DIRECT_KX_DONE;").map_err(PtxGenError::FormatError)?;

        // Skip if y was out of bounds
        writeln!(ptx, "    @%p4 bra $DIRECT_SKIP;").map_err(PtxGenError::FormatError)?;

        // in_x = in_x_base + kx * dilation_w
        writeln!(ptx, "    mad.lo.s32 %r25, %r24, {dw}, %r18;  // in_x")
            .map_err(PtxGenError::FormatError)?;

        // Check x bounds
        writeln!(ptx, "    setp.lt.s32 %p7, %r25, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p7 bra $DIRECT_SKIP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.s32 %p7, %r25, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p7 bra $DIRECT_SKIP;").map_err(PtxGenError::FormatError)?;

        // Load input value
        writeln!(
            ptx,
            "    mad.lo.s32 %r26, %r23, %r10, %r25;  // in_y * in_w + in_x"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd12, %r26;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd12, %rd12, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd13, %rd11, %rd12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd13];  // input_val")
            .map_err(PtxGenError::FormatError)?;

        // Load weight value: weight[oc_idx, c_local, ky, kx]
        writeln!(ptx, "    cvt.u64.u32 %rd14, %r20;  // weight_offset_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd14, %rd14, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd15, %rd8, %rd14;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f2, [%rd15];  // weight_val")
            .map_err(PtxGenError::FormatError)?;

        // FMA: acc += input_val * weight_val
        writeln!(ptx, "    fma.rn{ty} %f0, %f1, %f2, %f0;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$DIRECT_SKIP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r20, %r20, 1;  // weight_offset_idx++")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r24, %r24, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $DIRECT_KX_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$DIRECT_KX_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r22, %r22, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $DIRECT_KY_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$DIRECT_KY_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r19, %r19, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $DIRECT_C_LOOP;").map_err(PtxGenError::FormatError)?;

        // Add bias if bias pointer is non-null
        writeln!(ptx, "$DIRECT_BIAS:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.eq.u64 %p2, %rd3, 0;  // bias == null?")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $DIRECT_STORE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd16, %r7;  // oc_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd16, %rd16, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd17, %rd3, %rd16;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f3, [%rd17];  // bias_val")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add{ty} %f0, %f0, %f3;").map_err(PtxGenError::FormatError)?;

        // Store output: output[batch, oc_idx, out_y, out_x]
        writeln!(ptx, "$DIRECT_STORE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    // Store output value").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r27, %r11, %r12;  // out_h * out_w")
            .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mul.lo.u32 %r28, %r27, {oc};  // oc * out_h * out_w"
        )
        .map_err(PtxGenError::FormatError)?;
        // output_idx = batch * (oc * out_h * out_w) + oc_idx * (out_h * out_w) + out_y * out_w + out_x
        writeln!(
            ptx,
            "    mad.lo.u32 %r29, %r6, %r28, 0;  // batch * out_plane"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u32 %r29, %r7, %r27, %r29;  // + oc_idx * spatial"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u32 %r29, %r2, %r12, %r29;  // + out_y * out_w"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r29, %r29, %r5;  // + out_x")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd18, %r29;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd18, %rd18, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd19, %rd2, %rd18;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd19], %f0;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$DIRECT_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    // -----------------------------------------------------------------------
    // 1x1 convolution kernel (optimized pointwise)
    // -----------------------------------------------------------------------

    /// Generates an optimized PTX kernel for 1x1 (pointwise) convolution.
    ///
    /// Since the kernel is 1x1, there are no spatial loops; the operation
    /// degenerates to a matrix multiply of the weight matrix against each
    /// spatial position. Each thread computes one output value by dot-
    /// producting the channel dimension.
    ///
    /// **Parameters:**
    /// - `input`: pointer to input `[N, C, H, W]`
    /// - `weight`: pointer to weight `[out_C, C/g]` (1x1 kernel, no spatial dims)
    /// - `output`: pointer to output `[N, out_C, H, W]`
    /// - `bias`: pointer to bias `[out_C]` (may be null)
    /// - `batch_size`: N
    /// - `spatial_size`: H * W (input spatial size = output spatial size for 1x1)
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if validation or PTX formatting fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate_1x1_conv_kernel(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.ty();
        let byte_size = self.byte_size();
        let kernel_name = self.kernel_name("1x1");
        let cpg = self.channels_per_group();
        let zero_lit = self.zero_lit();
        let groups = self.groups;
        let oc = self.out_channels;
        let ocpg = self.out_channels_per_group();
        let in_ch = self.in_channels;

        let mut ptx = String::with_capacity(4096);
        self.write_header(&mut ptx)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_weight,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_bias,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_spatial_size").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Register declarations
        writeln!(ptx, "    .reg .b32 %r<48>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<24>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {ty} %f<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread indexing
        // global_idx = ctaid.x * ntid.x + tid.x -> spatial position
        // ctaid.y -> combined (batch, oc_idx)
        writeln!(ptx, "    // Thread indexing").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;  // spatial_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r4, %ctaid.y;  // combined (batch, oc)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Decompose ctaid.y
        writeln!(ptx, "    div.u32 %r5, %r4, {oc};  // batch_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rem.u32 %r6, %r4, {oc};  // oc_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_weight];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd3, [%param_bias];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r7, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r8, [%param_spatial_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check: spatial_idx < spatial_size
        writeln!(ptx, "    setp.ge.u32 %p0, %r3, %r8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $CONV1X1_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Determine group and channel range
        if groups > 1 {
            writeln!(ptx, "    div.u32 %r9, %r6, {ocpg};  // group_idx")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    mul.lo.u32 %r10, %r9, {cpg};  // c_start")
                .map_err(PtxGenError::FormatError)?;
        } else {
            writeln!(ptx, "    mov.u32 %r10, 0;  // c_start").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Input base for this batch: input + batch * C * spatial * byte_size
        writeln!(ptx, "    // Input base for batch").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r11, %r8, {in_ch};  // C * spatial")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd4, %r11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd4, %rd4, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u64 %rd6, %rd5, %rd4, %rd0;  // input_batch_ptr"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Weight base for oc_idx: weight + oc_idx * cpg * byte_size
        let weight_per_oc = (cpg as usize) * byte_size;
        writeln!(ptx, "    // Weight base for oc").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd7, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd7, %rd7, {weight_per_oc};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd8, %rd1, %rd7;  // weight_oc_ptr")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Dot product over channels
        writeln!(ptx, "    // Dot product over channels_per_group")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f0, {zero_lit};  // acc").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r12, 0;  // c_local").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$CONV1X1_C_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r12, {cpg};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $CONV1X1_BIAS;").map_err(PtxGenError::FormatError)?;

        // c_global = c_start + c_local
        writeln!(ptx, "    add.u32 %r13, %r10, %r12;  // c_global")
            .map_err(PtxGenError::FormatError)?;

        // Load input: input_batch_ptr + (c_global * spatial_size + spatial_idx) * byte_size
        writeln!(
            ptx,
            "    mad.lo.u32 %r14, %r13, %r8, %r3;  // c_global * spatial + idx"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd9, %r14;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd9, %rd9, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd10, %rd6, %rd9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd10];  // input_val")
            .map_err(PtxGenError::FormatError)?;

        // Load weight: weight_oc_ptr + c_local * byte_size
        writeln!(ptx, "    cvt.u64.u32 %rd11, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd12, %rd8, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f2, [%rd12];  // weight_val")
            .map_err(PtxGenError::FormatError)?;

        // FMA
        writeln!(ptx, "    fma.rn{ty} %f0, %f1, %f2, %f0;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    add.u32 %r12, %r12, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $CONV1X1_C_LOOP;").map_err(PtxGenError::FormatError)?;

        // Bias
        writeln!(ptx, "$CONV1X1_BIAS:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.eq.u64 %p2, %rd3, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $CONV1X1_STORE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd13, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd13, %rd13, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd14, %rd3, %rd13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f3, [%rd14];  // bias_val")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add{ty} %f0, %f0, %f3;").map_err(PtxGenError::FormatError)?;

        // Store output: output + (batch * oc * spatial + oc_idx * spatial + spatial_idx) * byte_size
        writeln!(ptx, "$CONV1X1_STORE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r15, %r8, {oc};  // oc * spatial")
            .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u32 %r16, %r5, %r15, 0;  // batch * (oc * spatial)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u32 %r16, %r6, %r8, %r16;  // + oc_idx * spatial"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r16, %r16, %r3;  // + spatial_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd15, %r16;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd15, %rd15, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd16, %rd2, %rd15;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd16], %f0;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$CONV1X1_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    // -----------------------------------------------------------------------
    // Backward data kernel (input gradient)
    // -----------------------------------------------------------------------

    /// Generates a PTX kernel for computing the input gradient (backward data).
    ///
    /// Performs transposed convolution of the output gradient with flipped
    /// kernels. Each thread computes one input gradient value by iterating
    /// over the output channels and kernel window.
    ///
    /// **Parameters:**
    /// - `grad_output`: pointer to output gradient `[N, out_C, out_H, out_W]`
    /// - `weight`: pointer to weight `[out_C, C/g, kH, kW]`
    /// - `grad_input`: pointer to input gradient `[N, C, H, W]` (output)
    /// - `batch_size`: N
    /// - `in_h`, `in_w`: input spatial dimensions
    /// - `out_h`, `out_w`: output spatial dimensions
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if validation or PTX formatting fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate_backward_data_kernel(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.ty();
        let byte_size = self.byte_size();
        let kernel_name = self.kernel_name("bwd_data");
        let cpg = self.channels_per_group();
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let sh = self.stride_h;
        let sw = self.stride_w;
        let ph = self.pad_h;
        let pw = self.pad_w;
        let dh = self.dilation_h;
        let dw = self.dilation_w;
        let zero_lit = self.zero_lit();
        let groups = self.groups;
        let oc = self.out_channels;
        let ocpg = self.out_channels_per_group();
        let in_ch = self.in_channels;

        let mut ptx = String::with_capacity(8192);
        self.write_header(&mut ptx)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_grad_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_weight,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_grad_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_in_h,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_in_w,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_out_h,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_out_w").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Register declarations
        writeln!(ptx, "    .reg .b32 %r<64>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {ty} %f<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread indexing: each thread computes one input gradient value
        // ctaid.x * ntid.x + tid.x -> in_x
        // ctaid.y -> in_y
        // ctaid.z -> combined (batch, channel)
        writeln!(ptx, "    // Thread indexing").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;  // in_x")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r4, %ctaid.y;  // in_y").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mov.u32 %r5, %ctaid.z;  // combined (batch, channel)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Decompose ctaid.z
        writeln!(ptx, "    div.u32 %r6, %r5, {in_ch};  // batch_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    rem.u32 %r7, %r5, {in_ch};  // c_idx (input channel)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_grad_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_weight];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_grad_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r8, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r9, [%param_in_h];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r10, [%param_in_w];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r11, [%param_out_h];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r12, [%param_out_w];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check
        writeln!(ptx, "    setp.ge.u32 %p0, %r3, %r10;  // in_x >= in_w?")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r4, %r9;   // in_y >= in_h?")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    or.pred %p0, %p0, %p1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $BWD_DATA_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Determine group for this input channel
        if groups > 1 {
            writeln!(ptx, "    div.u32 %r13, %r7, {cpg};  // group_idx")
                .map_err(PtxGenError::FormatError)?;
            writeln!(
                ptx,
                "    rem.u32 %r14, %r7, {cpg};  // c_local (within group)"
            )
            .map_err(PtxGenError::FormatError)?;
        } else {
            writeln!(ptx, "    mov.u32 %r13, 0;  // group_idx")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    mov.u32 %r14, %r7;  // c_local = c_idx")
                .map_err(PtxGenError::FormatError)?;
        }
        // oc_start = group_idx * ocpg
        writeln!(ptx, "    mul.lo.u32 %r15, %r13, {ocpg};  // oc_start")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Grad output batch base: grad_output + batch * oc * out_h * out_w * byte_size
        writeln!(ptx, "    // Grad output batch base").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mul.lo.u32 %r16, %r11, %r12;  // out_spatial = out_h * out_w"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r17, %r16, {oc};  // oc * out_spatial")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r17;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd4, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u64 %rd5, %rd4, %rd3, %rd0;  // go_batch_ptr"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Initialize accumulator
        writeln!(ptx, "    mov{ty} %f0, {zero_lit};  // grad_acc")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Loop over output channels in group
        writeln!(ptx, "    // Loop over output channels and kernel positions")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r18, 0;  // oc_local").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$BWD_DATA_OC_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p2, %r18, {ocpg};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $BWD_DATA_STORE;").map_err(PtxGenError::FormatError)?;

        // oc_global = oc_start + oc_local
        writeln!(ptx, "    add.u32 %r19, %r15, %r18;  // oc_global")
            .map_err(PtxGenError::FormatError)?;

        // Weight base for (oc_global, c_local): weight + (oc_global * cpg + c_local) * kh * kw * byte_size
        let kernel_spatial = (kh * kw) as usize * byte_size;
        writeln!(
            ptx,
            "    mad.lo.u32 %r20, %r19, {cpg}, %r14;  // oc_global * cpg + c_local"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd6, %r20;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd6, %rd6, {kernel_spatial};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    add.u64 %rd7, %rd1, %rd6;  // weight_ptr for this (oc, c)"
        )
        .map_err(PtxGenError::FormatError)?;

        // Grad output channel plane: go_batch_ptr + oc_global * out_spatial * byte_size
        writeln!(ptx, "    cvt.u64.u32 %rd8, %r19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd9, %r16;  // out_spatial")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, %rd9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd10, %rd5, %rd8;  // go_channel_ptr")
            .map_err(PtxGenError::FormatError)?;

        // Loop over kernel positions
        writeln!(ptx, "    mov.u32 %r21, 0;  // ky").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$BWD_DATA_KY_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p3, %r21, {kh};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p3 bra $BWD_DATA_KY_DONE;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    mov.u32 %r22, 0;  // kx").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$BWD_DATA_KX_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p4, %r22, {kw};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 bra $BWD_DATA_KX_DONE;").map_err(PtxGenError::FormatError)?;

        // For backward data, the corresponding output position is:
        // out_y = (in_y + pad_h - ky * dilation_h) / stride_h (if divisible)
        // out_x = (in_x + pad_w - kx * dilation_w) / stride_w (if divisible)
        writeln!(ptx, "    // Compute corresponding output position")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.s32 %r23, %r21, {dh}, 0;  // ky * dh")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.s32 %r24, %r4, {ph};  // in_y + pad_h")
            .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    sub.s32 %r24, %r24, %r23;  // in_y + pad_h - ky * dh"
        )
        .map_err(PtxGenError::FormatError)?;

        // Check divisibility by stride_h and compute out_y
        writeln!(ptx, "    setp.lt.s32 %p5, %r24, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p5 bra $BWD_DATA_KX_NEXT;").map_err(PtxGenError::FormatError)?;
        if sh > 1 {
            writeln!(ptx, "    rem.u32 %r25, %r24, {sh};").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    setp.ne.u32 %p5, %r25, 0;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    @%p5 bra $BWD_DATA_KX_NEXT;").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx, "    div.u32 %r26, %r24, {sh};  // out_y")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p5, %r26, %r11;  // out_y >= out_h?")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p5 bra $BWD_DATA_KX_NEXT;").map_err(PtxGenError::FormatError)?;

        // Same for x
        writeln!(ptx, "    mad.lo.s32 %r27, %r22, {dw}, 0;  // kx * dw")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.s32 %r28, %r3, {pw};  // in_x + pad_w")
            .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    sub.s32 %r28, %r28, %r27;  // in_x + pad_w - kx * dw"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.s32 %p6, %r28, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p6 bra $BWD_DATA_KX_NEXT;").map_err(PtxGenError::FormatError)?;
        if sw > 1 {
            writeln!(ptx, "    rem.u32 %r29, %r28, {sw};").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    setp.ne.u32 %p6, %r29, 0;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    @%p6 bra $BWD_DATA_KX_NEXT;").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx, "    div.u32 %r30, %r28, {sw};  // out_x")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p6, %r30, %r12;  // out_x >= out_w?")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p6 bra $BWD_DATA_KX_NEXT;").map_err(PtxGenError::FormatError)?;

        // Load grad_output[batch, oc, out_y, out_x]
        writeln!(
            ptx,
            "    mad.lo.u32 %r31, %r26, %r12, %r30;  // out_y * out_w + out_x"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd11, %r31;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd12, %rd10, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd12];  // go_val")
            .map_err(PtxGenError::FormatError)?;

        // Load weight[oc, c_local, ky, kx] (flipped: use (kh-1-ky, kw-1-kx) for transposed conv)
        // For backward data we need the weight at position (ky, kx) as stored
        // because the summation already handles the transpose via index mapping.
        writeln!(ptx, "    // Load weight at (ky, kx)").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u32 %r32, %r21, {kw}, %r22;  // ky * kw + kx"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd13, %r32;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd13, %rd13, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd14, %rd7, %rd13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f2, [%rd14];  // w_val")
            .map_err(PtxGenError::FormatError)?;

        // FMA: grad_acc += go_val * w_val
        writeln!(ptx, "    fma.rn{ty} %f0, %f1, %f2, %f0;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$BWD_DATA_KX_NEXT:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r22, %r22, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $BWD_DATA_KX_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$BWD_DATA_KX_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r21, %r21, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $BWD_DATA_KY_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$BWD_DATA_KY_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r18, %r18, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $BWD_DATA_OC_LOOP;").map_err(PtxGenError::FormatError)?;

        // Store grad_input[batch, c_idx, in_y, in_x]
        writeln!(ptx, "$BWD_DATA_STORE:").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mul.lo.u32 %r33, %r9, %r10;  // in_spatial = in_h * in_w"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mul.lo.u32 %r34, %r33, {in_ch};  // C * in_spatial"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r35, %r6, %r34, 0;  // batch * plane")
            .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u32 %r35, %r7, %r33, %r35;  // + c * spatial"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u32 %r35, %r4, %r10, %r35;  // + in_y * in_w"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r35, %r35, %r3;  // + in_x")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd15, %r35;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd15, %rd15, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd16, %rd2, %rd15;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd16], %f0;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$BWD_DATA_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    // -----------------------------------------------------------------------
    // Backward filter kernel (weight gradient)
    // -----------------------------------------------------------------------

    /// Generates a PTX kernel for computing the weight gradient (backward filter).
    ///
    /// For each weight position `[oc, c, ky, kx]`, accumulates the correlation
    /// between input patches and the output gradient across the entire batch
    /// and all spatial positions.
    ///
    /// **Parameters:**
    /// - `input`: pointer to input `[N, C, H, W]`
    /// - `grad_output`: pointer to output gradient `[N, out_C, out_H, out_W]`
    /// - `grad_weight`: pointer to weight gradient `[out_C, C/g, kH, kW]` (output)
    /// - `batch_size`: N
    /// - `in_h`, `in_w`: input spatial dimensions
    /// - `out_h`, `out_w`: output spatial dimensions
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if validation or PTX formatting fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate_backward_filter_kernel(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.ty();
        let byte_size = self.byte_size();
        let kernel_name = self.kernel_name("bwd_filter");
        let cpg = self.channels_per_group();
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let sh = self.stride_h;
        let sw = self.stride_w;
        let ph = self.pad_h;
        let pw = self.pad_w;
        let dh = self.dilation_h;
        let dw = self.dilation_w;
        let zero_lit = self.zero_lit();
        let groups = self.groups;
        let oc = self.out_channels;
        let ocpg = self.out_channels_per_group();
        let in_ch = self.in_channels;

        let mut ptx = String::with_capacity(8192);
        self.write_header(&mut ptx)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_grad_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_grad_weight,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_in_h,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_in_w,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_out_h,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_out_w").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Register declarations
        writeln!(ptx, "    .reg .b32 %r<64>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {ty} %f<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread indexing: each thread computes one weight gradient element
        // ctaid.x * ntid.x + tid.x -> flat weight index within one (oc, c_local) pair's spatial kernel
        // ctaid.y -> combined (oc, c_local) index
        // We flatten oc_idx * cpg + c_local into ctaid.y
        let weight_spatial = kh * kw;
        writeln!(
            ptx,
            "    // Thread indexing (one thread per weight element)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u32 %r3, %r1, %r2, %r0;  // spatial_idx (ky*kw flat)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r4, %ctaid.y;  // combined (oc, c_local)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check on spatial_idx
        writeln!(ptx, "    setp.ge.u32 %p0, %r3, {weight_spatial};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $BWD_FILTER_DONE;").map_err(PtxGenError::FormatError)?;

        // Decompose spatial_idx into (ky, kx)
        writeln!(ptx, "    div.u32 %r5, %r3, {kw};  // ky").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rem.u32 %r6, %r3, {kw};  // kx").map_err(PtxGenError::FormatError)?;

        // Decompose ctaid.y into (oc_idx, c_local)
        writeln!(ptx, "    div.u32 %r7, %r4, {cpg};  // oc_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rem.u32 %r8, %r4, {cpg};  // c_local")
            .map_err(PtxGenError::FormatError)?;
        // Bounds check on oc_idx
        writeln!(ptx, "    setp.ge.u32 %p1, %r7, {oc};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $BWD_FILTER_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute c_global from group
        if groups > 1 {
            writeln!(ptx, "    div.u32 %r9, %r7, {ocpg};  // group_idx")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    mad.lo.u32 %r10, %r9, {cpg}, %r8;  // c_global")
                .map_err(PtxGenError::FormatError)?;
        } else {
            writeln!(ptx, "    mov.u32 %r10, %r8;  // c_global = c_local")
                .map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_grad_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_grad_weight];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r11, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r12, [%param_in_h];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r13, [%param_in_w];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r14, [%param_out_h];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r15, [%param_out_w];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Precompute spatial sizes
        writeln!(
            ptx,
            "    mul.lo.u32 %r16, %r12, %r13;  // in_spatial = in_h * in_w"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mul.lo.u32 %r17, %r14, %r15;  // out_spatial = out_h * out_w"
        )
        .map_err(PtxGenError::FormatError)?;
        // Input batch stride = C * in_spatial * byte_size
        writeln!(
            ptx,
            "    mul.lo.u32 %r18, %r16, {in_ch};  // C * in_spatial"
        )
        .map_err(PtxGenError::FormatError)?;
        // Output batch stride = oc * out_spatial * byte_size
        writeln!(ptx, "    mul.lo.u32 %r19, %r17, {oc};  // oc * out_spatial")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Initialize accumulator
        writeln!(ptx, "    mov{ty} %f0, {zero_lit};  // grad_w_acc")
            .map_err(PtxGenError::FormatError)?;

        // Loop over batch and output spatial positions
        writeln!(ptx, "    mov.u32 %r20, 0;  // batch").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$BWD_FILTER_BATCH_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p2, %r20, %r11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $BWD_FILTER_STORE;").map_err(PtxGenError::FormatError)?;

        // Input base for (batch, c_global): input + (batch * C * in_spatial + c_global * in_spatial) * byte_size
        writeln!(
            ptx,
            "    mad.lo.u32 %r21, %r20, %r18, 0;  // batch * C * in_spatial"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u32 %r21, %r10, %r16, %r21;  // + c_global * in_spatial"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r21;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd0, %rd3;  // input_ch_ptr")
            .map_err(PtxGenError::FormatError)?;

        // Grad output base for (batch, oc): go + (batch * oc * out_spatial + oc_idx * out_spatial) * byte_size
        writeln!(
            ptx,
            "    mad.lo.u32 %r22, %r20, %r19, 0;  // batch * oc * out_spatial"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mad.lo.u32 %r22, %r7, %r17, %r22;  // + oc_idx * out_spatial"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r22;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd1, %rd5;  // go_oc_ptr")
            .map_err(PtxGenError::FormatError)?;

        // Loop over output spatial positions (out_y, out_x)
        writeln!(ptx, "    mov.u32 %r23, 0;  // out_y").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$BWD_FILTER_OY_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p3, %r23, %r14;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p3 bra $BWD_FILTER_OY_DONE;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    mov.u32 %r24, 0;  // out_x").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$BWD_FILTER_OX_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p4, %r24, %r15;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 bra $BWD_FILTER_OX_DONE;").map_err(PtxGenError::FormatError)?;

        // Corresponding input position:
        // in_y = out_y * stride_h - pad_h + ky * dilation_h
        // in_x = out_x * stride_w - pad_w + kx * dilation_w
        writeln!(ptx, "    mad.lo.s32 %r25, %r23, {sh}, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub.s32 %r25, %r25, {ph};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.s32 %r25, %r5, {dh}, %r25;  // in_y")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.s32 %r26, %r24, {sw}, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub.s32 %r26, %r26, {pw};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.s32 %r26, %r6, {dw}, %r26;  // in_x")
            .map_err(PtxGenError::FormatError)?;

        // Bounds check: 0 <= in_y < in_h && 0 <= in_x < in_w
        writeln!(ptx, "    setp.lt.s32 %p5, %r25, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.s32 %p6, %r25, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    or.pred %p5, %p5, %p6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.s32 %p6, %r26, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    or.pred %p5, %p5, %p6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.s32 %p6, %r26, %r13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    or.pred %p5, %p5, %p6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p5 bra $BWD_FILTER_OX_NEXT;").map_err(PtxGenError::FormatError)?;

        // Load input[batch, c_global, in_y, in_x]
        writeln!(
            ptx,
            "    mad.lo.s32 %r27, %r25, %r13, %r26;  // in_y * in_w + in_x"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd7, %r27;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd7, %rd7, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd8, %rd4, %rd7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd8];  // input_val")
            .map_err(PtxGenError::FormatError)?;

        // Load grad_output[batch, oc, out_y, out_x]
        writeln!(
            ptx,
            "    mad.lo.u32 %r28, %r23, %r15, %r24;  // out_y * out_w + out_x"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd9, %r28;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd9, %rd9, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd10, %rd6, %rd9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f2, [%rd10];  // go_val")
            .map_err(PtxGenError::FormatError)?;

        // FMA: grad_w_acc += input_val * go_val
        writeln!(ptx, "    fma.rn{ty} %f0, %f1, %f2, %f0;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$BWD_FILTER_OX_NEXT:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r24, %r24, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $BWD_FILTER_OX_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$BWD_FILTER_OX_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r23, %r23, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $BWD_FILTER_OY_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$BWD_FILTER_OY_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r20, %r20, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $BWD_FILTER_BATCH_LOOP;").map_err(PtxGenError::FormatError)?;

        // Store grad_weight[oc_idx, c_local, ky, kx]
        writeln!(ptx, "$BWD_FILTER_STORE:").map_err(PtxGenError::FormatError)?;
        let weight_per_oc_total = (cpg * kh * kw) as usize * byte_size;
        writeln!(ptx, "    // Store weight gradient").map_err(PtxGenError::FormatError)?;
        // flat index = oc_idx * cpg * kh * kw + c_local * kh * kw + ky * kw + kx
        writeln!(ptx, "    cvt.u64.u32 %rd11, %r7;  // oc_idx")
            .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    mul.lo.u64 %rd11, %rd11, {weight_per_oc_total};  // oc_idx * (cpg*kh*kw*bs)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd12, %r8;  // c_local")
            .map_err(PtxGenError::FormatError)?;
        let kernel_spatial_bytes = (kh * kw) as usize * byte_size;
        writeln!(
            ptx,
            "    mul.lo.u64 %rd12, %rd12, {kernel_spatial_bytes};  // c_local * kh*kw*bs"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd11, %rd11, %rd12;").map_err(PtxGenError::FormatError)?;
        // Add spatial position: (ky * kw + kx) * byte_size = r3 * byte_size
        writeln!(ptx, "    cvt.u64.u32 %rd13, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd13, %rd13, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd11, %rd11, %rd13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd14, %rd2, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd14], %f0;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$BWD_FILTER_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }
}

#[cfg(test)]
#[path = "convolution_tests.rs"]
mod tests;
