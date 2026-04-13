//! Transposed convolution (fractionally-strided convolution / deconvolution).
//!
//! Implements the upsampling counterpart to regular convolution, commonly used
//! in decoder networks (autoencoders, VAEs), generative models (GANs), and
//! semantic segmentation (upsampling paths).
//!
//! The implementation uses the **col2im + GEMM** approach:
//! 1. Reshape weights: `[C_in, C_out/groups, kH, kW]` → column matrix
//! 2. GEMM: multiply transposed weight by input columns
//! 3. col2im: scatter GEMM output columns back to spatial output positions
//!
//! This is equivalent to inserting zeros into the input (fractional striding)
//! and performing a regular forward convolution, but avoids the overhead of
//! the explicit zero-insertion.

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// TransposeConvConfig
// ---------------------------------------------------------------------------

/// Configuration for a transposed convolution operation.
///
/// A transposed convolution (also called fractionally-strided convolution or
/// deconvolution) computes an output that would produce the given input if
/// a regular convolution with the same parameters were applied. It is the
/// gradient of convolution with respect to data.
///
/// # Output size formula
///
/// ```text
/// out_h = (in_h - 1) * stride_h - 2 * pad_h
///       + dilation_h * (kernel_h - 1) + output_pad_h + 1
/// ```
///
/// The `output_pad_h/w` parameters resolve the ambiguity that arises because
/// multiple input sizes can map to the same output size under a given stride.
#[derive(Debug, Clone)]
pub struct TransposeConvConfig {
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Stride height.
    pub stride_h: usize,
    /// Stride width.
    pub stride_w: usize,
    /// Padding height.
    pub pad_h: usize,
    /// Padding width.
    pub pad_w: usize,
    /// Output padding height (for resolving stride ambiguity).
    pub output_pad_h: usize,
    /// Output padding width.
    pub output_pad_w: usize,
    /// Dilation height.
    pub dilation_h: usize,
    /// Dilation width.
    pub dilation_w: usize,
    /// Number of groups (1 = normal, >1 = grouped transposed conv).
    pub groups: usize,
}

impl TransposeConvConfig {
    /// Computes the output spatial dimensions.
    ///
    /// Uses the standard transposed convolution output formula:
    /// ```text
    /// out_h = (in_h - 1) * stride_h - 2 * pad_h
    ///       + dilation_h * (kernel_h - 1) + output_pad_h + 1
    /// ```
    #[must_use]
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = (input_h.saturating_sub(1)) * self.stride_h
            + self.dilation_h * (self.kernel_h.saturating_sub(1))
            + self.output_pad_h
            + 1;
        // Subtract 2 * pad_h only if there's enough room.
        let out_h = out_h.saturating_sub(2 * self.pad_h);

        let out_w = (input_w.saturating_sub(1)) * self.stride_w
            + self.dilation_w * (self.kernel_w.saturating_sub(1))
            + self.output_pad_w
            + 1;
        let out_w = out_w.saturating_sub(2 * self.pad_w);

        (out_h, out_w)
    }

    /// Validates the transposed convolution configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if:
    /// - `kernel_h` or `kernel_w` is zero
    /// - `stride_h` or `stride_w` is zero
    /// - `output_pad_h >= stride_h` or `output_pad_w >= stride_w`
    /// - `groups` is zero
    /// - `in_channels` is not divisible by `groups`
    /// - `out_channels` is not divisible by `groups`
    /// - `in_channels` or `out_channels` is zero
    /// - `dilation_h` or `dilation_w` is zero
    pub fn validate(&self) -> DnnResult<()> {
        if self.kernel_h == 0 || self.kernel_w == 0 {
            return Err(DnnError::InvalidArgument(
                "transposed conv: kernel dimensions must be > 0".into(),
            ));
        }
        if self.stride_h == 0 || self.stride_w == 0 {
            return Err(DnnError::InvalidArgument(
                "transposed conv: stride must be > 0".into(),
            ));
        }
        if self.dilation_h == 0 || self.dilation_w == 0 {
            return Err(DnnError::InvalidArgument(
                "transposed conv: dilation must be > 0".into(),
            ));
        }
        if self.output_pad_h >= self.stride_h {
            return Err(DnnError::InvalidArgument(format!(
                "transposed conv: output_pad_h ({}) must be < stride_h ({})",
                self.output_pad_h, self.stride_h
            )));
        }
        if self.output_pad_w >= self.stride_w {
            return Err(DnnError::InvalidArgument(format!(
                "transposed conv: output_pad_w ({}) must be < stride_w ({})",
                self.output_pad_w, self.stride_w
            )));
        }
        if self.groups == 0 {
            return Err(DnnError::InvalidArgument(
                "transposed conv: groups must be > 0".into(),
            ));
        }
        if self.in_channels == 0 || self.out_channels == 0 {
            return Err(DnnError::InvalidArgument(
                "transposed conv: channel counts must be > 0".into(),
            ));
        }
        if self.in_channels % self.groups != 0 {
            return Err(DnnError::InvalidArgument(format!(
                "transposed conv: in_channels ({}) not divisible by groups ({})",
                self.in_channels, self.groups
            )));
        }
        if self.out_channels % self.groups != 0 {
            return Err(DnnError::InvalidArgument(format!(
                "transposed conv: out_channels ({}) not divisible by groups ({})",
                self.out_channels, self.groups
            )));
        }
        Ok(())
    }

    /// Returns the number of input channels per group.
    #[must_use]
    pub fn in_channels_per_group(&self) -> usize {
        if self.groups == 0 {
            return 0;
        }
        self.in_channels / self.groups
    }

    /// Returns the number of output channels per group.
    #[must_use]
    pub fn out_channels_per_group(&self) -> usize {
        if self.groups == 0 {
            return 0;
        }
        self.out_channels / self.groups
    }

    /// Returns the effective kernel height accounting for dilation.
    #[must_use]
    pub fn effective_kernel_h(&self) -> usize {
        self.dilation_h * (self.kernel_h.saturating_sub(1)) + 1
    }

    /// Returns the effective kernel width accounting for dilation.
    #[must_use]
    pub fn effective_kernel_w(&self) -> usize {
        self.dilation_w * (self.kernel_w.saturating_sub(1)) + 1
    }
}

// ---------------------------------------------------------------------------
// TransposeConvPlan
// ---------------------------------------------------------------------------

/// Execution plan for a transposed convolution.
///
/// Pre-computes all derived dimensions and workspace requirements so that
/// the actual execution can proceed without redundant calculations.
#[derive(Debug, Clone)]
pub struct TransposeConvPlan {
    /// The validated configuration.
    pub config: TransposeConvConfig,
    /// Batch size (N).
    pub batch_size: usize,
    /// Input spatial height.
    pub input_h: usize,
    /// Input spatial width.
    pub input_w: usize,
    /// Computed output height.
    pub output_h: usize,
    /// Computed output width.
    pub output_w: usize,
    /// Workspace size needed in bytes for the col2im intermediate buffer.
    pub workspace_bytes: usize,
}

impl TransposeConvPlan {
    /// Creates a plan for transposed convolution.
    ///
    /// Validates the config and pre-computes output dimensions and workspace
    /// requirements.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if the config is invalid or
    /// if `batch_size`, `input_h`, or `input_w` is zero.
    pub fn create(
        config: TransposeConvConfig,
        batch_size: usize,
        input_h: usize,
        input_w: usize,
    ) -> DnnResult<Self> {
        config.validate()?;

        if batch_size == 0 {
            return Err(DnnError::InvalidArgument(
                "transposed conv: batch_size must be > 0".into(),
            ));
        }
        if input_h == 0 || input_w == 0 {
            return Err(DnnError::InvalidArgument(
                "transposed conv: input spatial dimensions must be > 0".into(),
            ));
        }

        let (output_h, output_w) = config.output_size(input_h, input_w);

        if output_h == 0 || output_w == 0 {
            return Err(DnnError::InvalidDimension(format!(
                "transposed conv: computed output size is zero ({output_h}x{output_w})"
            )));
        }

        // Workspace for the col2im intermediate column matrix.
        // Shape: [in_channels_per_group * kernel_h * kernel_w, output_h * output_w]
        // per batch element per group. We compute for the full batch.
        //
        // For the GEMM phase:
        //   weight: [in_channels_per_group, out_channels_per_group * kH * kW]
        //   input column: [in_channels_per_group, input_h * input_w]
        //   GEMM result: [out_channels_per_group * kH * kW, input_h * input_w]
        //
        // The col2im kernel scatters this into [out_channels_per_group, output_h, output_w].
        let out_channels_per_group = config.out_channels_per_group();
        let col_rows = out_channels_per_group * config.kernel_h * config.kernel_w;
        let col_cols = input_h * input_w;
        let elements_per_sample = col_rows * col_cols;
        // Total across all groups and batch elements.
        // Each group is processed independently, so we need workspace for
        // at least one group x one batch at a time. We allocate for the
        // largest single (batch, group) slice.
        let workspace_elements = elements_per_sample;
        // 4 bytes for f32, 8 bytes for f64 — use 8 as upper bound.
        let workspace_bytes = workspace_elements * 8;

        Ok(Self {
            config,
            batch_size,
            input_h,
            input_w,
            output_h,
            output_w,
            workspace_bytes,
        })
    }

    /// Returns the workspace size needed in bytes.
    #[must_use]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes
    }

    /// Returns workspace size for a specific precision ("f32" or "f64").
    #[must_use]
    pub fn workspace_size_for_precision(&self, precision: &str) -> usize {
        let elem_bytes = match precision {
            "f32" => 4,
            "f64" => 8,
            _ => 8, // default to larger
        };
        let out_channels_per_group = self.config.out_channels_per_group();
        let col_rows = out_channels_per_group * self.config.kernel_h * self.config.kernel_w;
        let col_cols = self.input_h * self.input_w;
        col_rows * col_cols * elem_bytes
    }
}

// ---------------------------------------------------------------------------
// PTX Generation — col2im scatter kernel
// ---------------------------------------------------------------------------

/// Parameters for the col2im kernel body emission.
#[derive(Debug, Clone, Copy)]
struct Col2imParams {
    float_type: PtxType,
    elem_bytes: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
}

/// Generates PTX for the col2im scatter kernel.
///
/// The col2im kernel is the inverse of im2col: it scatters column matrix
/// elements back to spatial output positions. Each thread handles one element
/// in the output tensor and accumulates all contributing column entries.
///
/// Given the GEMM output of shape `[out_channels_per_group * kH * kW, in_H * in_W]`,
/// each output pixel `(n, c_out, oh, ow)` accumulates contributions from
/// column positions where:
/// ```text
/// oh = ih * stride_h - pad_h + kh * dilation_h
/// ow = iw * stride_w - pad_w + kw * dilation_w
/// ```
/// for all valid `(ih, iw, kh, kw)` combinations.
///
/// # Errors
///
/// Returns [`DnnError::PtxGeneration`] if kernel building fails.
pub fn generate_col2im_ptx(
    config: &TransposeConvConfig,
    precision: &str,
    sm_version: SmVersion,
) -> DnnResult<String> {
    let kernel_name = format!("col2im_transpose_conv_{precision}");
    let float_type = match precision {
        "f32" => PtxType::F32,
        "f64" => PtxType::F64,
        other => {
            return Err(DnnError::InvalidArgument(format!(
                "unsupported precision for col2im: {other}"
            )));
        }
    };
    let elem_bytes: u32 = match precision {
        "f32" => 4,
        _ => 8,
    };

    // Capture config values for the closure (must be 'static).
    let params = Col2imParams {
        float_type,
        elem_bytes,
        kernel_h: config.kernel_h as u32,
        kernel_w: config.kernel_w as u32,
        stride_h: config.stride_h as u32,
        stride_w: config.stride_w as u32,
        pad_h: config.pad_h as u32,
        pad_w: config.pad_w as u32,
        dilation_h: config.dilation_h as u32,
        dilation_w: config.dilation_w as u32,
    };

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm_version)
        // Pointers to column matrix (GEMM result) and output tensor
        .param("col_matrix", PtxType::U64)
        .param("output", PtxType::U64)
        // Dimensions
        .param("out_channels_per_group", PtxType::U32)
        .param("out_h", PtxType::U32)
        .param("out_w", PtxType::U32)
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("total_output_elements", PtxType::U32)
        .body(move |b| {
            emit_col2im_body(b, &params);
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Emits the col2im kernel body using raw PTX branching to avoid
/// register move/borrow issues with closures.
///
/// Each thread computes one output element by finding all (kh, kw, ih, iw)
/// combinations from the GEMM column matrix that contribute to this output
/// position, and accumulating their values.
fn emit_col2im_body(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>, p: &Col2imParams) {
    let float_type = p.float_type;
    let elem_bytes = p.elem_bytes;
    let kernel_h = p.kernel_h;
    let kernel_w = p.kernel_w;
    let stride_h = p.stride_h;
    let stride_w = p.stride_w;
    let pad_h = p.pad_h;
    let pad_w = p.pad_w;
    let dilation_h = p.dilation_h;
    let dilation_w = p.dilation_w;
    b.comment("=== Col2im Transpose Convolution Scatter Kernel ===");
    b.comment("Each thread handles one output element (c_out, oh, ow).");

    // Bounds check: gid < total
    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_output_elements");
    let pred_bounds = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_bounds}, {gid}, {total};"));
    let exit_label = b.fresh_label("exit");
    b.raw_ptx(&format!("@!{pred_bounds} bra {exit_label};"));

    b.comment("Load parameters");
    let col_ptr = b.load_param_u64("col_matrix");
    let out_ptr = b.load_param_u64("output");
    let _out_cpg = b.load_param_u32("out_channels_per_group");
    let out_h = b.load_param_u32("out_h");
    let out_w = b.load_param_u32("out_w");
    let in_h = b.load_param_u32("in_h");
    let in_w = b.load_param_u32("in_w");

    b.comment("Decompose gid -> (c_out, oh, ow)");
    let out_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {out_hw}, {out_h}, {out_w};"));

    let c_out = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {c_out}, {gid}, {out_hw};"));
    let remainder = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {remainder}, {gid}, {out_hw};"));
    let oh = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {oh}, {remainder}, {out_w};"));
    let ow = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {ow}, {remainder}, {out_w};"));

    b.comment("Initialize accumulator to zero");
    let acc = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        let zero_bits: u32 = 0f32.to_bits();
        b.raw_ptx(&format!("mov.b32 {acc}, 0F{zero_bits:08X};"));
    } else {
        let zero_bits: u64 = 0f64.to_bits();
        b.raw_ptx(&format!("mov.b64 {acc}, 0D{zero_bits:016X};"));
    }

    let in_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_hw}, {in_h}, {in_w};"));

    let kh_kw_val = kernel_h * kernel_w;
    let kh_kw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {kh_kw}, {kh_kw_val};"));

    // Scratch registers reused across loop iterations
    let oh_plus_pad = b.alloc_reg(PtxType::U32);
    let ow_plus_pad = b.alloc_reg(PtxType::U32);
    let h_offset = b.alloc_reg(PtxType::U32);
    let w_offset = b.alloc_reg(PtxType::U32);
    let h_mod = b.alloc_reg(PtxType::U32);
    let w_mod = b.alloc_reg(PtxType::U32);
    let ih = b.alloc_reg(PtxType::U32);
    let iw = b.alloc_reg(PtxType::U32);
    let col_row = b.alloc_reg(PtxType::U32);
    let col_idx = b.alloc_reg(PtxType::U32);
    let row_offset = b.alloc_reg(PtxType::U32);
    let ih_times_inw = b.alloc_reg(PtxType::U32);
    let spatial_idx = b.alloc_reg(PtxType::U32);
    let c_kh_kw_tmp = b.alloc_reg(PtxType::U32);
    let pred_h_ge = b.alloc_reg(PtxType::Pred);
    let pred_h_mod = b.alloc_reg(PtxType::Pred);
    let pred_ih = b.alloc_reg(PtxType::Pred);
    let pred_w_ge = b.alloc_reg(PtxType::Pred);
    let pred_w_mod = b.alloc_reg(PtxType::Pred);
    let pred_iw = b.alloc_reg(PtxType::Pred);
    let load_addr = b.alloc_reg(PtxType::U64);
    let loaded_val = b.alloc_reg(float_type);
    let idx64 = b.alloc_reg(PtxType::U64);
    let offset64 = b.alloc_reg(PtxType::U64);

    b.comment("Unrolled kernel loop over (kh, kw)");
    for kh_val in 0..kernel_h {
        for kw_val in 0..kernel_w {
            let kh_dil = kh_val * dilation_h;
            let kw_dil = kw_val * dilation_w;
            let kh_times_kw_plus_kw = kh_val * kernel_w + kw_val;

            let skip = b.fresh_label(&format!("skip_kh{kh_val}_kw{kw_val}"));

            b.comment(&format!("kh={kh_val}, kw={kw_val}"));

            // oh_plus_pad = oh + pad_h
            b.raw_ptx(&format!("add.u32 {oh_plus_pad}, {oh}, {pad_h};"));
            // check oh_plus_pad >= kh_dil
            b.raw_ptx(&format!(
                "setp.hs.u32 {pred_h_ge}, {oh_plus_pad}, {kh_dil};"
            ));
            b.raw_ptx(&format!("@!{pred_h_ge} bra {skip};"));

            // h_offset = oh_plus_pad - kh_dil
            b.raw_ptx(&format!("sub.u32 {h_offset}, {oh_plus_pad}, {kh_dil};"));
            // check h_offset % stride_h == 0
            b.raw_ptx(&format!("rem.u32 {h_mod}, {h_offset}, {stride_h};"));
            b.raw_ptx(&format!("setp.eq.u32 {pred_h_mod}, {h_mod}, 0;"));
            b.raw_ptx(&format!("@!{pred_h_mod} bra {skip};"));

            // ih = h_offset / stride_h; check ih < in_h
            b.raw_ptx(&format!("div.u32 {ih}, {h_offset}, {stride_h};"));
            b.raw_ptx(&format!("setp.lo.u32 {pred_ih}, {ih}, {in_h};"));
            b.raw_ptx(&format!("@!{pred_ih} bra {skip};"));

            // ow_plus_pad = ow + pad_w; check >= kw_dil
            b.raw_ptx(&format!("add.u32 {ow_plus_pad}, {ow}, {pad_w};"));
            b.raw_ptx(&format!(
                "setp.hs.u32 {pred_w_ge}, {ow_plus_pad}, {kw_dil};"
            ));
            b.raw_ptx(&format!("@!{pred_w_ge} bra {skip};"));

            // w_offset = ow_plus_pad - kw_dil; check % stride_w == 0
            b.raw_ptx(&format!("sub.u32 {w_offset}, {ow_plus_pad}, {kw_dil};"));
            b.raw_ptx(&format!("rem.u32 {w_mod}, {w_offset}, {stride_w};"));
            b.raw_ptx(&format!("setp.eq.u32 {pred_w_mod}, {w_mod}, 0;"));
            b.raw_ptx(&format!("@!{pred_w_mod} bra {skip};"));

            // iw = w_offset / stride_w; check iw < in_w
            b.raw_ptx(&format!("div.u32 {iw}, {w_offset}, {stride_w};"));
            b.raw_ptx(&format!("setp.lo.u32 {pred_iw}, {iw}, {in_w};"));
            b.raw_ptx(&format!("@!{pred_iw} bra {skip};"));

            // col_row = c_out * kh_kw + kh_times_kw_plus_kw
            b.raw_ptx(&format!("mul.lo.u32 {c_kh_kw_tmp}, {c_out}, {kh_kw};"));
            b.raw_ptx(&format!(
                "add.u32 {col_row}, {c_kh_kw_tmp}, {kh_times_kw_plus_kw};"
            ));

            // col_idx = col_row * in_hw + ih * in_w + iw
            b.raw_ptx(&format!("mul.lo.u32 {row_offset}, {col_row}, {in_hw};"));
            b.raw_ptx(&format!("mul.lo.u32 {ih_times_inw}, {ih}, {in_w};"));
            b.raw_ptx(&format!("add.u32 {spatial_idx}, {ih_times_inw}, {iw};"));
            b.raw_ptx(&format!("add.u32 {col_idx}, {row_offset}, {spatial_idx};"));

            // Load col_matrix[col_idx]
            b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {col_idx};"));
            b.raw_ptx(&format!("mul.lo.u64 {offset64}, {idx64}, {elem_bytes};"));
            b.raw_ptx(&format!("add.u64 {load_addr}, {col_ptr}, {offset64};"));
            if float_type == PtxType::F32 {
                b.raw_ptx(&format!("ld.global.f32 {loaded_val}, [{load_addr}];"));
                b.raw_ptx(&format!("add.rn.f32 {acc}, {acc}, {loaded_val};"));
            } else {
                b.raw_ptx(&format!("ld.global.f64 {loaded_val}, [{load_addr}];"));
                b.raw_ptx(&format!("add.rn.f64 {acc}, {acc}, {loaded_val};"));
            }

            b.label(&skip);
        }
    }

    b.comment("Store accumulated result to output");
    let out_idx64 = b.alloc_reg(PtxType::U64);
    let out_off64 = b.alloc_reg(PtxType::U64);
    let out_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {out_idx64}, {gid};"));
    b.raw_ptx(&format!(
        "mul.lo.u64 {out_off64}, {out_idx64}, {elem_bytes};"
    ));
    b.raw_ptx(&format!("add.u64 {out_addr}, {out_ptr}, {out_off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{out_addr}], {acc};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{out_addr}], {acc};"));
    }

    b.label(&exit_label);
    b.ret();
}

// ---------------------------------------------------------------------------
// PTX Generation — weight reshape kernel (grouped transposed conv)
// ---------------------------------------------------------------------------

/// Parameters for the weight reshape kernel body emission.
#[derive(Debug, Clone, Copy)]
struct WeightReshapeParams {
    float_type: PtxType,
    elem_bytes: u32,
    in_cpg: u32,
    out_cpg: u32,
    kernel_h: u32,
    kernel_w: u32,
}

/// Generates PTX for the weight reshaping kernel used in grouped transposed
/// convolution.
///
/// For grouped transposed convolution, the weight tensor is shaped as
/// `[in_channels, out_channels/groups, kH, kW]` and needs to be reshaped
/// and transposed to `[groups, out_channels/groups, in_channels/groups, kH, kW]`
/// for per-group GEMM operations.
///
/// Each thread handles one element of the output reshaped weight tensor.
///
/// # Errors
///
/// Returns [`DnnError::PtxGeneration`] if kernel building fails.
/// Returns [`DnnError::InvalidArgument`] for unsupported precision.
pub fn generate_weight_reshape_ptx(
    config: &TransposeConvConfig,
    precision: &str,
    sm_version: SmVersion,
) -> DnnResult<String> {
    let kernel_name = format!("weight_reshape_transpose_conv_{precision}");
    let float_type = match precision {
        "f32" => PtxType::F32,
        "f64" => PtxType::F64,
        other => {
            return Err(DnnError::InvalidArgument(format!(
                "unsupported precision for weight reshape: {other}"
            )));
        }
    };
    let elem_bytes: u32 = match precision {
        "f32" => 4,
        _ => 8,
    };

    let groups = config.groups as u32;
    let in_cpg = config.in_channels_per_group() as u32;
    let out_cpg = config.out_channels_per_group() as u32;
    let kernel_h = config.kernel_h as u32;
    let kernel_w = config.kernel_w as u32;

    let wr_params = WeightReshapeParams {
        float_type,
        elem_bytes,
        in_cpg,
        out_cpg,
        kernel_h,
        kernel_w,
    };

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm_version)
        .param("src_weight", PtxType::U64)
        .param("dst_weight", PtxType::U64)
        .param("total_elements", PtxType::U32)
        .body(move |b| {
            emit_weight_reshape_body(b, groups, &wr_params);
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Emits the weight reshape kernel body using raw PTX branching.
///
/// Source layout:  `[in_channels, out_channels/groups, kH, kW]`
/// Dest layout:    `[groups, out_channels/groups, in_channels/groups, kH, kW]`
///
/// Each thread maps one destination element back to the source and copies it.
fn emit_weight_reshape_body(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    _groups: u32,
    p: &WeightReshapeParams,
) {
    let float_type = p.float_type;
    let elem_bytes = p.elem_bytes;
    let in_cpg = p.in_cpg;
    let out_cpg = p.out_cpg;
    let kernel_h = p.kernel_h;
    let kernel_w = p.kernel_w;
    b.comment("=== Weight Reshape for Grouped Transpose Conv ===");
    b.comment("src: [in_channels, out_cpg, kH, kW]");
    b.comment("dst: [groups, out_cpg, in_cpg, kH, kW]");

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_elements");

    // Bounds check
    let pred_bounds = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_bounds}, {gid}, {total};"));
    let exit_label = b.fresh_label("wr_exit");
    b.raw_ptx(&format!("@!{pred_bounds} bra {exit_label};"));

    let src_ptr = b.load_param_u64("src_weight");
    let dst_ptr = b.load_param_u64("dst_weight");

    b.comment("Decompose dst index: gid -> (g, oc, ic, kh, kw)");
    let kh_kw = kernel_h * kernel_w;
    let ic_kh_kw = in_cpg * kh_kw;
    let oc_ic_kh_kw = out_cpg * ic_kh_kw;

    // g = gid / (out_cpg * in_cpg * kH * kW)
    let g = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {g}, {gid}, {oc_ic_kh_kw};"));
    let rem1 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem1}, {gid}, {oc_ic_kh_kw};"));

    // oc = rem1 / (in_cpg * kH * kW)
    let oc = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {oc}, {rem1}, {ic_kh_kw};"));
    let rem2 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem2}, {rem1}, {ic_kh_kw};"));

    // ic = rem2 / (kH * kW)
    let ic = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {ic}, {rem2}, {kh_kw};"));
    let rem3 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem3}, {rem2}, {kh_kw};"));

    // kh = rem3 / kW, kw_reg = rem3 % kW
    let kh = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {kh}, {rem3}, {kernel_w};"));
    let kw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {kw}, {rem3}, {kernel_w};"));

    b.comment("Compute source index");
    // in_channel = g * in_cpg + ic
    let in_ch = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_ch}, {g}, {in_cpg};"));
    b.raw_ptx(&format!("add.u32 {in_ch}, {in_ch}, {ic};"));

    // src_idx = in_channel * (out_cpg * kH * kW) + oc * (kH * kW) + kh * kW + kw
    let out_cpg_khkw = out_cpg * kh_kw;
    let src_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {src_idx}, {in_ch}, {out_cpg_khkw};"));
    let oc_offset = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {oc_offset}, {oc}, {kh_kw};"));
    b.raw_ptx(&format!("add.u32 {src_idx}, {src_idx}, {oc_offset};"));
    let kh_off = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {kh_off}, {kh}, {kernel_w};"));
    b.raw_ptx(&format!("add.u32 {src_idx}, {src_idx}, {kh_off};"));
    b.raw_ptx(&format!("add.u32 {src_idx}, {src_idx}, {kw};"));

    b.comment("Load from source, store to destination");
    // src addr
    let src_idx64 = b.alloc_reg(PtxType::U64);
    let src_off64 = b.alloc_reg(PtxType::U64);
    let src_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {src_idx64}, {src_idx};"));
    b.raw_ptx(&format!(
        "mul.lo.u64 {src_off64}, {src_idx64}, {elem_bytes};"
    ));
    b.raw_ptx(&format!("add.u64 {src_addr}, {src_ptr}, {src_off64};"));

    // dst addr
    let dst_idx64 = b.alloc_reg(PtxType::U64);
    let dst_off64 = b.alloc_reg(PtxType::U64);
    let dst_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {dst_idx64}, {gid};"));
    b.raw_ptx(&format!(
        "mul.lo.u64 {dst_off64}, {dst_idx64}, {elem_bytes};"
    ));
    b.raw_ptx(&format!("add.u64 {dst_addr}, {dst_ptr}, {dst_off64};"));

    let val = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {val}, [{src_addr}];"));
        b.raw_ptx(&format!("st.global.f32 [{dst_addr}], {val};"));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {val}, [{src_addr}];"));
        b.raw_ptx(&format!("st.global.f64 [{dst_addr}], {val};"));
    }

    b.label(&exit_label);
    b.ret();
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a basic config for testing.
    fn basic_config() -> TransposeConvConfig {
        TransposeConvConfig {
            in_channels: 64,
            out_channels: 32,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            pad_h: 1,
            pad_w: 1,
            output_pad_h: 1,
            output_pad_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        }
    }

    // -----------------------------------------------------------------------
    // Output size tests
    // -----------------------------------------------------------------------

    #[test]
    fn output_size_basic() {
        let cfg = basic_config();
        let (oh, ow) = cfg.output_size(4, 4);
        // (4-1)*2 - 2*1 + 1*(3-1) + 1 + 1 = 6 - 2 + 2 + 2 = 8
        assert_eq!(oh, 8);
        assert_eq!(ow, 8);
    }

    #[test]
    fn output_size_stride2_doubles_spatial() {
        // stride=2 with no padding, no output_pad, kernel=1: output = (in-1)*2 + 1
        let cfg = TransposeConvConfig {
            in_channels: 16,
            out_channels: 16,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            output_pad_h: 0,
            output_pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };
        let (oh, ow) = cfg.output_size(4, 4);
        assert_eq!(oh, 7);
        assert_eq!(ow, 7);
        // With output_pad=1: (4-1)*2 + 0 + 0 + 1 + 1 = 8
        let cfg2 = TransposeConvConfig {
            output_pad_h: 1,
            output_pad_w: 1,
            ..cfg
        };
        let (oh2, ow2) = cfg2.output_size(4, 4);
        assert_eq!(oh2, 8);
        assert_eq!(ow2, 8);
    }

    #[test]
    fn output_size_with_padding_reduces_output() {
        let cfg = TransposeConvConfig {
            in_channels: 16,
            out_channels: 16,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            output_pad_h: 0,
            output_pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };
        let (oh_no_pad, ow_no_pad) = cfg.output_size(4, 4);
        let cfg_padded = TransposeConvConfig {
            pad_h: 1,
            pad_w: 1,
            ..cfg
        };
        let (oh_pad, ow_pad) = cfg_padded.output_size(4, 4);
        assert!(oh_pad < oh_no_pad, "padding should reduce output height");
        assert!(ow_pad < ow_no_pad, "padding should reduce output width");
    }

    #[test]
    fn output_size_with_dilation_increases_effective_kernel() {
        let cfg = TransposeConvConfig {
            in_channels: 16,
            out_channels: 16,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            output_pad_h: 0,
            output_pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };
        let (oh1, _) = cfg.output_size(4, 4);

        let cfg_dilated = TransposeConvConfig {
            dilation_h: 2,
            dilation_w: 2,
            ..cfg
        };
        let (oh2, _) = cfg_dilated.output_size(4, 4);
        assert!(
            oh2 > oh1,
            "dilation should increase output via effective kernel size"
        );
    }

    #[test]
    fn output_size_output_padding_resolves_ambiguity() {
        // Two configs identical except for output_pad produce different output sizes.
        let cfg_a = TransposeConvConfig {
            in_channels: 16,
            out_channels: 16,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            pad_h: 1,
            pad_w: 1,
            output_pad_h: 0,
            output_pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };
        let cfg_b = TransposeConvConfig {
            output_pad_h: 1,
            output_pad_w: 1,
            ..cfg_a.clone()
        };
        let (oh_a, ow_a) = cfg_a.output_size(4, 4);
        let (oh_b, ow_b) = cfg_b.output_size(4, 4);
        assert_eq!(oh_b, oh_a + 1, "output_pad_h=1 should add 1 to output_h");
        assert_eq!(ow_b, ow_a + 1, "output_pad_w=1 should add 1 to output_w");
    }

    // -----------------------------------------------------------------------
    // Validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn validate_ok() {
        let cfg = basic_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_kernel_zero() {
        let cfg = TransposeConvConfig {
            kernel_h: 0,
            ..basic_config()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("kernel"));
    }

    #[test]
    fn validate_stride_zero() {
        let cfg = TransposeConvConfig {
            stride_h: 0,
            ..basic_config()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("stride"));
    }

    #[test]
    fn validate_output_pad_ge_stride() {
        let cfg = TransposeConvConfig {
            output_pad_h: 2, // == stride_h
            ..basic_config()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("output_pad_h"));
    }

    #[test]
    fn validate_groups_divides_channels() {
        let cfg = TransposeConvConfig {
            groups: 3, // 64 % 3 != 0
            ..basic_config()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("not divisible"));
    }

    #[test]
    fn validate_groups_zero() {
        let cfg = TransposeConvConfig {
            groups: 0,
            ..basic_config()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("groups"));
    }

    #[test]
    fn validate_channels_zero() {
        let cfg = TransposeConvConfig {
            in_channels: 0,
            ..basic_config()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("channel"));
    }

    #[test]
    fn validate_dilation_zero() {
        let cfg = TransposeConvConfig {
            dilation_h: 0,
            ..basic_config()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("dilation"));
    }

    // -----------------------------------------------------------------------
    // Grouped transposed conv validation
    // -----------------------------------------------------------------------

    #[test]
    fn grouped_transpose_conv_validation() {
        let cfg = TransposeConvConfig {
            in_channels: 64,
            out_channels: 64,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            output_pad_h: 0,
            output_pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 4,
        };
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.in_channels_per_group(), 16);
        assert_eq!(cfg.out_channels_per_group(), 16);
    }

    // -----------------------------------------------------------------------
    // Plan tests
    // -----------------------------------------------------------------------

    #[test]
    fn plan_creation_and_workspace_size() {
        let cfg = basic_config();
        let plan = TransposeConvPlan::create(cfg, 4, 8, 8);
        assert!(plan.is_ok());
        let plan = plan.expect("plan creation should succeed in test");
        assert_eq!(plan.batch_size, 4);
        assert!(plan.output_h > 0);
        assert!(plan.output_w > 0);
        assert!(plan.workspace_size() > 0, "workspace must be positive");
    }

    #[test]
    fn plan_workspace_positive_for_valid_config() {
        let cfg = TransposeConvConfig {
            in_channels: 32,
            out_channels: 16,
            kernel_h: 4,
            kernel_w: 4,
            stride_h: 2,
            stride_w: 2,
            pad_h: 1,
            pad_w: 1,
            output_pad_h: 0,
            output_pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };
        let plan =
            TransposeConvPlan::create(cfg, 1, 4, 4).expect("plan creation should succeed in test");
        assert!(plan.workspace_size() > 0);
        // Verify f32 workspace is half of f64
        let ws_f32 = plan.workspace_size_for_precision("f32");
        let ws_f64 = plan.workspace_size_for_precision("f64");
        assert_eq!(ws_f64, ws_f32 * 2);
    }

    #[test]
    fn plan_rejects_zero_batch() {
        let cfg = basic_config();
        let err = TransposeConvPlan::create(cfg, 0, 8, 8).unwrap_err();
        assert!(err.to_string().contains("batch_size"));
    }

    #[test]
    fn plan_rejects_zero_spatial() {
        let cfg = basic_config();
        let err = TransposeConvPlan::create(cfg, 1, 0, 8).unwrap_err();
        assert!(err.to_string().contains("spatial"));
    }

    // -----------------------------------------------------------------------
    // PTX generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn col2im_ptx_generates_valid_ptx() {
        let cfg = basic_config();
        let ptx = generate_col2im_ptx(&cfg, "f32", SmVersion::Sm80);
        assert!(ptx.is_ok());
        let text = ptx.expect("col2im PTX should generate in test");
        assert!(text.contains(".entry col2im_transpose_conv_f32"));
        assert!(text.contains(".visible"));
    }

    #[test]
    fn col2im_ptx_contains_target_directive() {
        let cfg = basic_config();
        let ptx =
            generate_col2im_ptx(&cfg, "f32", SmVersion::Sm80).expect("PTX gen should succeed");
        assert!(
            ptx.contains(".target sm_80"),
            "PTX must contain target directive"
        );
    }

    #[test]
    fn col2im_ptx_f64_variant() {
        let cfg = basic_config();
        let ptx =
            generate_col2im_ptx(&cfg, "f64", SmVersion::Sm80).expect("PTX gen should succeed");
        assert!(ptx.contains("col2im_transpose_conv_f64"));
        // f64 loads/stores use .f64
        assert!(ptx.contains(".f64"));
    }

    #[test]
    fn col2im_ptx_rejects_invalid_precision() {
        let cfg = basic_config();
        let result = generate_col2im_ptx(&cfg, "f16", SmVersion::Sm80);
        assert!(result.is_err());
    }

    #[test]
    fn weight_reshape_ptx_generation() {
        let cfg = TransposeConvConfig {
            in_channels: 64,
            out_channels: 64,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            output_pad_h: 0,
            output_pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 4,
        };
        let ptx = generate_weight_reshape_ptx(&cfg, "f32", SmVersion::Sm80);
        assert!(ptx.is_ok());
        let text = ptx.expect("weight reshape PTX should generate in test");
        assert!(text.contains("weight_reshape_transpose_conv_f32"));
        assert!(text.contains(".target sm_80"));
    }

    #[test]
    fn weight_reshape_ptx_rejects_invalid_precision() {
        let cfg = basic_config();
        let result = generate_weight_reshape_ptx(&cfg, "bf16", SmVersion::Sm80);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Effective kernel size
    // -----------------------------------------------------------------------

    #[test]
    fn effective_kernel_with_dilation() {
        let cfg = TransposeConvConfig {
            dilation_h: 2,
            dilation_w: 3,
            ..basic_config()
        };
        // effective_h = 2 * (3-1) + 1 = 5
        assert_eq!(cfg.effective_kernel_h(), 5);
        // effective_w = 3 * (3-1) + 1 = 7
        assert_eq!(cfg.effective_kernel_w(), 7);
    }
}
