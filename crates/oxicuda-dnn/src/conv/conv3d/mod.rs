//! 3D volumetric convolution for deep learning.
//!
//! Implements GPU-accelerated 3D convolution primitives targeting:
//! - **Video understanding** — spatiotemporal convolution on `[N, C, D, H, W]` tensors
//! - **Medical imaging** — 3D CT/MRI volume processing
//! - **Point cloud processing** — voxelised 3D representations
//!
//! # Algorithms
//!
//! | Algorithm | Best for | Workspace |
//! |-----------|----------|-----------|
//! | [`Im2colGemm`](Conv3dAlgorithm::Im2colGemm) | General-purpose | Yes |
//! | [`Direct1x1x1`](Conv3dAlgorithm::Direct1x1x1) | 1×1×1 point-wise | None |
//! | [`DirectSmall`](Conv3dAlgorithm::DirectSmall) | 3×3×3 kernels | Shared mem |
//!
//! # Layout
//!
//! Input tensors are expected in NCDHW format: `[batch, channels, depth, height, width]`.
//! The im2col approach flattens the volumetric patch into a column matrix of shape
//! `[C_in/groups × kD × kH × kW,  oD × oH × oW]` and then calls GEMM with the
//! weight matrix `[C_out/groups, C_in/groups × kD × kH × kW]`.

use oxicuda_ptx::arch::SmVersion;

use crate::error::{DnnError, DnnResult};

mod conv3d_ptx;
pub use conv3d_ptx::{
    generate_col2im3d_ptx, generate_direct3d_ptx, generate_im2col3d_ptx, generate_wgrad3d_ptx,
};

// ---------------------------------------------------------------------------
// Conv3dConfig
// ---------------------------------------------------------------------------

/// Configuration for a 3D (volumetric) convolution operation.
///
/// Captures all hyper-parameters — kernel size, stride, padding, dilation,
/// and grouping — for computing output dimensions and selecting an algorithm.
#[derive(Debug, Clone)]
pub struct Conv3dConfig {
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Kernel depth.
    pub kernel_d: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Stride along the depth axis.
    pub stride_d: usize,
    /// Stride along the height axis.
    pub stride_h: usize,
    /// Stride along the width axis.
    pub stride_w: usize,
    /// Zero-padding along the depth axis.
    pub pad_d: usize,
    /// Zero-padding along the height axis.
    pub pad_h: usize,
    /// Zero-padding along the width axis.
    pub pad_w: usize,
    /// Dilation along the depth axis.
    pub dilation_d: usize,
    /// Dilation along the height axis.
    pub dilation_h: usize,
    /// Dilation along the width axis.
    pub dilation_w: usize,
    /// Number of groups (1 = standard convolution, >1 = grouped convolution).
    pub groups: usize,
}

impl Conv3dConfig {
    /// Computes the output spatial dimensions `(out_d, out_h, out_w)`.
    ///
    /// Uses the standard convolution output formula per axis:
    /// ```text
    /// out = (in + 2·pad − dilation·(kernel − 1) − 1) / stride + 1
    /// ```
    #[must_use]
    pub fn output_size(
        &self,
        input_d: usize,
        input_h: usize,
        input_w: usize,
    ) -> (usize, usize, usize) {
        let out_d = self.output_dim(
            input_d,
            self.kernel_d,
            self.pad_d,
            self.stride_d,
            self.dilation_d,
        );
        let out_h = self.output_dim(
            input_h,
            self.kernel_h,
            self.pad_h,
            self.stride_h,
            self.dilation_h,
        );
        let out_w = self.output_dim(
            input_w,
            self.kernel_w,
            self.pad_w,
            self.stride_w,
            self.dilation_w,
        );
        (out_d, out_h, out_w)
    }

    /// Computes a single output dimension. Returns 0 on underflow.
    fn output_dim(
        &self,
        input: usize,
        kernel: usize,
        pad: usize,
        stride: usize,
        dilation: usize,
    ) -> usize {
        let effective_kernel = dilation.saturating_mul(kernel.saturating_sub(1));
        let numerator = input
            .saturating_add(2 * pad)
            .saturating_sub(effective_kernel)
            .saturating_sub(1);
        if stride == 0 {
            return 0;
        }
        numerator / stride + 1
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if:
    /// - Any kernel dimension is zero
    /// - Any stride is zero
    /// - Any dilation is zero
    /// - `groups` is zero
    /// - `in_channels` or `out_channels` is zero
    /// - `in_channels` or `out_channels` is not divisible by `groups`
    pub fn validate(&self) -> DnnResult<()> {
        if self.kernel_d == 0 || self.kernel_h == 0 || self.kernel_w == 0 {
            return Err(DnnError::InvalidArgument(
                "conv3d: kernel dimensions must be > 0".into(),
            ));
        }
        if self.stride_d == 0 || self.stride_h == 0 || self.stride_w == 0 {
            return Err(DnnError::InvalidArgument(
                "conv3d: stride must be > 0".into(),
            ));
        }
        if self.dilation_d == 0 || self.dilation_h == 0 || self.dilation_w == 0 {
            return Err(DnnError::InvalidArgument(
                "conv3d: dilation must be > 0".into(),
            ));
        }
        if self.groups == 0 {
            return Err(DnnError::InvalidArgument(
                "conv3d: groups must be > 0".into(),
            ));
        }
        if self.in_channels == 0 || self.out_channels == 0 {
            return Err(DnnError::InvalidArgument(
                "conv3d: channel counts must be > 0".into(),
            ));
        }
        if self.in_channels % self.groups != 0 {
            return Err(DnnError::InvalidArgument(format!(
                "conv3d: in_channels ({}) not divisible by groups ({})",
                self.in_channels, self.groups
            )));
        }
        if self.out_channels % self.groups != 0 {
            return Err(DnnError::InvalidArgument(format!(
                "conv3d: out_channels ({}) not divisible by groups ({})",
                self.out_channels, self.groups
            )));
        }
        Ok(())
    }

    /// Returns `true` if this is a 1×1×1 convolution (reducible to GEMM).
    #[must_use]
    pub fn is_1x1x1(&self) -> bool {
        self.kernel_d == 1
            && self.kernel_h == 1
            && self.kernel_w == 1
            && self.stride_d == 1
            && self.stride_h == 1
            && self.stride_w == 1
            && self.pad_d == 0
            && self.pad_h == 0
            && self.pad_w == 0
    }

    /// Returns `true` if this is a 3×3×3 convolution.
    #[must_use]
    pub fn is_3x3x3(&self) -> bool {
        self.kernel_d == 3 && self.kernel_h == 3 && self.kernel_w == 3
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

    /// Returns the effective kernel depth accounting for dilation.
    #[must_use]
    pub fn effective_kernel_d(&self) -> usize {
        self.dilation_d * (self.kernel_d.saturating_sub(1)) + 1
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
// Conv3dAlgorithm
// ---------------------------------------------------------------------------

/// Algorithm selection for 3D convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv3dAlgorithm {
    /// im2col + GEMM — general-purpose, works for all configurations.
    Im2colGemm,
    /// Direct 1×1×1 via GEMM reshape (no im2col needed).
    Direct1x1x1,
    /// Direct convolution optimised for small 3×3×3 kernels.
    DirectSmall,
}

// ---------------------------------------------------------------------------
// Conv3dPlan
// ---------------------------------------------------------------------------

/// Execution plan for a 3D convolution.
///
/// Pre-computes all derived dimensions and workspace requirements so that
/// the actual kernel launch can proceed without redundant calculations.
#[derive(Debug, Clone)]
pub struct Conv3dPlan {
    /// The validated configuration.
    pub config: Conv3dConfig,
    /// Batch size (N).
    pub batch_size: usize,
    /// Input spatial depth.
    pub input_d: usize,
    /// Input spatial height.
    pub input_h: usize,
    /// Input spatial width.
    pub input_w: usize,
    /// Computed output depth.
    pub output_d: usize,
    /// Computed output height.
    pub output_h: usize,
    /// Computed output width.
    pub output_w: usize,
    /// Workspace size needed in bytes for the im2col intermediate buffer.
    pub workspace_bytes: usize,
    /// Selected algorithm.
    pub algorithm: Conv3dAlgorithm,
}

impl Conv3dPlan {
    /// Creates a plan for 3D convolution.
    ///
    /// Validates the config and pre-computes output dimensions, workspace
    /// requirements, and selects the optimal algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if the config is invalid or
    /// if `batch_size` or any spatial dimension is zero.
    /// Returns [`DnnError::InvalidDimension`] if the computed output is zero.
    pub fn create(
        config: Conv3dConfig,
        batch_size: usize,
        input_d: usize,
        input_h: usize,
        input_w: usize,
    ) -> DnnResult<Self> {
        config.validate()?;

        if batch_size == 0 {
            return Err(DnnError::InvalidArgument(
                "conv3d: batch_size must be > 0".into(),
            ));
        }
        if input_d == 0 || input_h == 0 || input_w == 0 {
            return Err(DnnError::InvalidArgument(
                "conv3d: input spatial dimensions must be > 0".into(),
            ));
        }

        let (output_d, output_h, output_w) = config.output_size(input_d, input_h, input_w);

        if output_d == 0 || output_h == 0 || output_w == 0 {
            return Err(DnnError::InvalidDimension(format!(
                "conv3d: computed output size is zero ({output_d}x{output_h}x{output_w})"
            )));
        }

        let algorithm = Self::select_algorithm(&config);

        // Workspace for the im2col column matrix.
        // Shape: [C_in/groups * kD * kH * kW, oD * oH * oW]
        // For Direct1x1x1 no im2col is needed so workspace is 0.
        let workspace_bytes = match algorithm {
            Conv3dAlgorithm::Direct1x1x1 => 0,
            Conv3dAlgorithm::Im2colGemm | Conv3dAlgorithm::DirectSmall => {
                let in_cpg = config.in_channels_per_group();
                let col_rows = in_cpg * config.kernel_d * config.kernel_h * config.kernel_w;
                let col_cols = output_d * output_h * output_w;
                // 8 bytes (f64 upper bound) per element.
                col_rows * col_cols * 8
            }
        };

        Ok(Self {
            config,
            batch_size,
            input_d,
            input_h,
            input_w,
            output_d,
            output_h,
            output_w,
            workspace_bytes,
            algorithm,
        })
    }

    /// Returns the workspace size needed in bytes.
    #[must_use]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes
    }

    /// Creates a plan for 3D convolution (convenience constructor with `u32` params).
    ///
    /// This is an alias for [`create`](Self::create) that accepts `u32` parameters
    /// matching the typical neural network dimension type.
    ///
    /// # Errors
    ///
    /// Same as [`create`](Self::create).
    pub fn new(
        config: Conv3dConfig,
        batch_size: u32,
        in_d: u32,
        in_h: u32,
        in_w: u32,
    ) -> DnnResult<Self> {
        Self::create(
            config,
            batch_size as usize,
            in_d as usize,
            in_h as usize,
            in_w as usize,
        )
    }

    /// Returns the full output shape as `(N, C_out, D_out, H_out, W_out)`.
    #[must_use]
    pub fn output_shape(&self) -> (u32, u32, u32, u32, u32) {
        (
            self.batch_size as u32,
            self.config.out_channels as u32,
            self.output_d as u32,
            self.output_h as u32,
            self.output_w as u32,
        )
    }

    /// Generates the forward-pass PTX kernel.
    ///
    /// Uses the im2col3d approach for general configurations, or the
    /// direct 3×3×3 kernel when applicable. Returns the PTX assembly string.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if kernel building fails.
    pub fn generate_forward_ptx(&self) -> DnnResult<String> {
        let sm = SmVersion::Sm80;
        match self.algorithm {
            Conv3dAlgorithm::DirectSmall => generate_direct3d_ptx(&self.config, "f32", sm),
            Conv3dAlgorithm::Im2colGemm | Conv3dAlgorithm::Direct1x1x1 => generate_im2col3d_ptx(
                &self.config,
                self.batch_size,
                self.input_d,
                self.input_h,
                self.input_w,
                "f32",
                sm,
            ),
        }
    }

    /// Generates the backward-data PTX kernel (col2im3d scatter).
    ///
    /// Produces a kernel that scatters column-matrix gradients back to
    /// the 3D input spatial positions. Used for computing `dL/dX`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if kernel building fails.
    pub fn generate_backward_data_ptx(&self) -> DnnResult<String> {
        let sm = SmVersion::Sm80;
        generate_col2im3d_ptx(&self.config, "f32", sm)
    }

    /// Generates the backward-filter PTX kernel (weight gradient).
    ///
    /// Produces a kernel that computes `dL/dW` by accumulating
    /// input × output-gradient products over spatial positions.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if kernel building fails.
    pub fn generate_backward_filter_ptx(&self) -> DnnResult<String> {
        let sm = SmVersion::Sm80;
        generate_wgrad3d_ptx(
            &self.config,
            self.batch_size,
            self.input_d,
            self.input_h,
            self.input_w,
            "f32",
            sm,
        )
    }

    /// Returns workspace size for a specific precision (`"f32"` or `"f64"`).
    #[must_use]
    pub fn workspace_size_for_precision(&self, precision: &str) -> usize {
        if self.algorithm == Conv3dAlgorithm::Direct1x1x1 {
            return 0;
        }
        let elem_bytes: usize = match precision {
            "f32" => 4,
            "f64" => 8,
            _ => 8,
        };
        let in_cpg = self.config.in_channels_per_group();
        let col_rows = in_cpg * self.config.kernel_d * self.config.kernel_h * self.config.kernel_w;
        let col_cols = self.output_d * self.output_h * self.output_w;
        col_rows * col_cols * elem_bytes
    }

    /// Selects the optimal algorithm for the given configuration.
    fn select_algorithm(config: &Conv3dConfig) -> Conv3dAlgorithm {
        if config.is_1x1x1() {
            Conv3dAlgorithm::Direct1x1x1
        } else if config.is_3x3x3() {
            Conv3dAlgorithm::DirectSmall
        } else {
            Conv3dAlgorithm::Im2colGemm
        }
    }
}
// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: basic Conv3dConfig for testing.
    fn basic_config() -> Conv3dConfig {
        Conv3dConfig {
            in_channels: 64,
            out_channels: 128,
            kernel_d: 3,
            kernel_h: 3,
            kernel_w: 3,
            stride_d: 1,
            stride_h: 1,
            stride_w: 1,
            pad_d: 1,
            pad_h: 1,
            pad_w: 1,
            dilation_d: 1,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        }
    }

    // -----------------------------------------------------------------------
    // Conv3dConfig — output_size
    // -----------------------------------------------------------------------

    #[test]
    fn output_size_no_padding_stride1() {
        let cfg = Conv3dConfig {
            pad_d: 0,
            pad_h: 0,
            pad_w: 0,
            ..basic_config()
        };
        // out = (8 + 0 - 1*(3-1) - 1)/1 + 1 = (8-3)/1 + 1 = 6
        let (od, oh, ow) = cfg.output_size(8, 8, 8);
        assert_eq!((od, oh, ow), (6, 6, 6));
    }

    #[test]
    fn output_size_with_padding() {
        let cfg = basic_config(); // pad=1 each
        // out = (8 + 2 - 2 - 1)/1 + 1 = 8
        let (od, oh, ow) = cfg.output_size(8, 8, 8);
        assert_eq!((od, oh, ow), (8, 8, 8));
    }

    #[test]
    fn output_size_with_stride() {
        let cfg = Conv3dConfig {
            stride_d: 2,
            stride_h: 2,
            stride_w: 2,
            ..basic_config()
        };
        // out = (8 + 2 - 2 - 1)/2 + 1 = 7/2 + 1 = 3 + 1 = 4
        let (od, oh, ow) = cfg.output_size(8, 8, 8);
        assert_eq!((od, oh, ow), (4, 4, 4));
    }

    #[test]
    fn output_size_with_dilation() {
        let cfg = Conv3dConfig {
            dilation_d: 2,
            dilation_h: 2,
            dilation_w: 2,
            pad_d: 0,
            pad_h: 0,
            pad_w: 0,
            ..basic_config()
        };
        // effective kernel = 2*(3-1)+1 = 5
        // out = (8 + 0 - 5)/1 + 1 = 4
        let (od, oh, ow) = cfg.output_size(8, 8, 8);
        assert_eq!((od, oh, ow), (4, 4, 4));
    }

    #[test]
    fn output_size_asymmetric() {
        let cfg = Conv3dConfig {
            kernel_d: 3,
            kernel_h: 5,
            kernel_w: 1,
            stride_d: 1,
            stride_h: 2,
            stride_w: 1,
            pad_d: 1,
            pad_h: 2,
            pad_w: 0,
            dilation_d: 1,
            dilation_h: 1,
            dilation_w: 1,
            ..basic_config()
        };
        // D: (16 + 2 - 2 -1)/1 + 1 = 16
        // H: (32 + 4 - 4 -1)/2 + 1 = 31/2 + 1 = 16
        // W: (64 + 0 - 0 -1)/1 + 1 = 64
        let (od, oh, ow) = cfg.output_size(16, 32, 64);
        assert_eq!((od, oh, ow), (16, 16, 64));
    }

    // -----------------------------------------------------------------------
    // Conv3dConfig — validate
    // -----------------------------------------------------------------------

    #[test]
    fn validate_kernel_zero_fails() {
        let cfg = Conv3dConfig {
            kernel_d: 0,
            ..basic_config()
        };
        assert!(cfg.validate().is_err());
        let cfg2 = Conv3dConfig {
            kernel_h: 0,
            ..basic_config()
        };
        assert!(cfg2.validate().is_err());
        let cfg3 = Conv3dConfig {
            kernel_w: 0,
            ..basic_config()
        };
        assert!(cfg3.validate().is_err());
    }

    #[test]
    fn validate_stride_zero_fails() {
        let cfg = Conv3dConfig {
            stride_d: 0,
            ..basic_config()
        };
        assert!(cfg.validate().is_err());
        let cfg2 = Conv3dConfig {
            stride_h: 0,
            ..basic_config()
        };
        assert!(cfg2.validate().is_err());
        let cfg3 = Conv3dConfig {
            stride_w: 0,
            ..basic_config()
        };
        assert!(cfg3.validate().is_err());
    }

    #[test]
    fn validate_dilation_zero_fails() {
        let cfg = Conv3dConfig {
            dilation_d: 0,
            ..basic_config()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_groups_divides_channels() {
        let cfg = Conv3dConfig {
            groups: 3, // 64 not divisible by 3
            ..basic_config()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_groups_zero_fails() {
        let cfg = Conv3dConfig {
            groups: 0,
            ..basic_config()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_channels_zero_fails() {
        let cfg = Conv3dConfig {
            in_channels: 0,
            ..basic_config()
        };
        assert!(cfg.validate().is_err());
        let cfg2 = Conv3dConfig {
            out_channels: 0,
            ..basic_config()
        };
        assert!(cfg2.validate().is_err());
    }

    #[test]
    fn validate_valid_config_passes() {
        assert!(basic_config().validate().is_ok());
    }

    #[test]
    fn validate_grouped_config_passes() {
        let cfg = Conv3dConfig {
            in_channels: 64,
            out_channels: 128,
            groups: 4,
            ..basic_config()
        };
        assert!(cfg.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Conv3dConfig — is_1x1x1
    // -----------------------------------------------------------------------

    #[test]
    fn is_1x1x1_detection() {
        let cfg = Conv3dConfig {
            in_channels: 64,
            out_channels: 128,
            kernel_d: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_d: 1,
            stride_h: 1,
            stride_w: 1,
            pad_d: 0,
            pad_h: 0,
            pad_w: 0,
            dilation_d: 1,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };
        assert!(cfg.is_1x1x1());
        assert!(!basic_config().is_1x1x1());
    }

    #[test]
    fn is_1x1x1_with_stride_not_detected() {
        let cfg = Conv3dConfig {
            kernel_d: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_d: 2,
            stride_h: 1,
            stride_w: 1,
            pad_d: 0,
            pad_h: 0,
            pad_w: 0,
            ..basic_config()
        };
        assert!(!cfg.is_1x1x1());
    }

    // -----------------------------------------------------------------------
    // Conv3dPlan
    // -----------------------------------------------------------------------

    #[test]
    fn plan_creation_with_workspace() {
        let plan = Conv3dPlan::create(basic_config(), 2, 8, 8, 8);
        assert!(plan.is_ok());
        let plan = plan.expect("plan creation should succeed in test");
        assert_eq!(plan.output_d, 8);
        assert_eq!(plan.output_h, 8);
        assert_eq!(plan.output_w, 8);
        assert!(plan.workspace_bytes > 0);
    }

    #[test]
    fn plan_algorithm_1x1x1_picks_direct() {
        let cfg = Conv3dConfig {
            kernel_d: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_d: 1,
            stride_h: 1,
            stride_w: 1,
            pad_d: 0,
            pad_h: 0,
            pad_w: 0,
            ..basic_config()
        };
        let plan = Conv3dPlan::create(cfg, 1, 8, 8, 8);
        assert!(plan.is_ok());
        let plan = plan.expect("plan creation should succeed in test");
        assert_eq!(plan.algorithm, Conv3dAlgorithm::Direct1x1x1);
        assert_eq!(plan.workspace_bytes, 0);
    }

    #[test]
    fn plan_algorithm_3x3x3_picks_direct_small() {
        let plan = Conv3dPlan::create(basic_config(), 1, 8, 8, 8);
        assert!(plan.is_ok());
        let plan = plan.expect("plan creation should succeed in test");
        assert_eq!(plan.algorithm, Conv3dAlgorithm::DirectSmall);
    }

    #[test]
    fn plan_algorithm_5x5x5_picks_im2col() {
        let cfg = Conv3dConfig {
            kernel_d: 5,
            kernel_h: 5,
            kernel_w: 5,
            pad_d: 2,
            pad_h: 2,
            pad_w: 2,
            ..basic_config()
        };
        let plan = Conv3dPlan::create(cfg, 1, 8, 8, 8);
        assert!(plan.is_ok());
        let plan = plan.expect("plan creation should succeed in test");
        assert_eq!(plan.algorithm, Conv3dAlgorithm::Im2colGemm);
    }

    #[test]
    fn plan_rejects_zero_batch() {
        let result = Conv3dPlan::create(basic_config(), 0, 8, 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn plan_rejects_zero_spatial_dims() {
        assert!(Conv3dPlan::create(basic_config(), 1, 0, 8, 8).is_err());
        assert!(Conv3dPlan::create(basic_config(), 1, 8, 0, 8).is_err());
        assert!(Conv3dPlan::create(basic_config(), 1, 8, 8, 0).is_err());
    }

    #[test]
    fn plan_workspace_positive_for_im2col() {
        let cfg = Conv3dConfig {
            kernel_d: 5,
            kernel_h: 5,
            kernel_w: 5,
            pad_d: 2,
            pad_h: 2,
            pad_w: 2,
            ..basic_config()
        };
        let plan = Conv3dPlan::create(cfg, 1, 16, 16, 16);
        assert!(plan.is_ok());
        let plan = plan.expect("plan creation should succeed in test");
        assert!(plan.workspace_size() > 0);
    }

    #[test]
    fn plan_workspace_for_precision() {
        let cfg = Conv3dConfig {
            kernel_d: 5,
            kernel_h: 5,
            kernel_w: 5,
            pad_d: 2,
            pad_h: 2,
            pad_w: 2,
            ..basic_config()
        };
        let plan = Conv3dPlan::create(cfg, 1, 8, 8, 8);
        assert!(plan.is_ok());
        let plan = plan.expect("plan creation should succeed in test");
        let f32_ws = plan.workspace_size_for_precision("f32");
        let f64_ws = plan.workspace_size_for_precision("f64");
        assert_eq!(f64_ws, f32_ws * 2);
    }

    // -----------------------------------------------------------------------
    // PTX generation — im2col3d
    // -----------------------------------------------------------------------

    #[test]
    fn im2col3d_ptx_f32() {
        let cfg = basic_config();
        let ptx = generate_im2col3d_ptx(&cfg, 1, 8, 8, 8, "f32", SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("PTX generation should succeed in test");
        assert!(ptx.contains("im2col3d_f32"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn im2col3d_ptx_f64() {
        let cfg = basic_config();
        let ptx = generate_im2col3d_ptx(&cfg, 1, 8, 8, 8, "f64", SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("PTX generation should succeed in test");
        assert!(ptx.contains("im2col3d_f64"));
    }

    #[test]
    fn im2col3d_ptx_contains_target() {
        let cfg = basic_config();
        let ptx = generate_im2col3d_ptx(&cfg, 1, 8, 8, 8, "f32", SmVersion::Sm90);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("PTX generation should succeed in test");
        assert!(ptx.contains(".target sm_90"));
    }

    #[test]
    fn im2col3d_ptx_rejects_invalid_precision() {
        let cfg = basic_config();
        let result = generate_im2col3d_ptx(&cfg, 1, 8, 8, 8, "f16", SmVersion::Sm80);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // PTX generation — col2im3d
    // -----------------------------------------------------------------------

    #[test]
    fn col2im3d_ptx_f32() {
        let cfg = basic_config();
        let ptx = generate_col2im3d_ptx(&cfg, "f32", SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("PTX generation should succeed in test");
        assert!(ptx.contains("col2im3d_f32"));
    }

    #[test]
    fn col2im3d_ptx_f64() {
        let cfg = basic_config();
        let ptx = generate_col2im3d_ptx(&cfg, "f64", SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("PTX generation should succeed in test");
        assert!(ptx.contains("col2im3d_f64"));
    }

    // -----------------------------------------------------------------------
    // PTX generation — direct 3×3×3
    // -----------------------------------------------------------------------

    #[test]
    fn direct3d_ptx_f32() {
        let cfg = basic_config();
        let ptx = generate_direct3d_ptx(&cfg, "f32", SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("PTX generation should succeed in test");
        assert!(ptx.contains("direct3d_3x3x3_f32"));
    }

    #[test]
    fn direct3d_ptx_contains_kernel_size() {
        let cfg = basic_config();
        let ptx = generate_direct3d_ptx(&cfg, "f32", SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("PTX generation should succeed in test");
        // The 27-tap kernel should reference weight offsets via "27" literal.
        assert!(ptx.contains("3x3x3"));
    }

    #[test]
    fn direct3d_ptx_rejects_non_3x3x3() {
        let cfg = Conv3dConfig {
            kernel_d: 5,
            kernel_h: 5,
            kernel_w: 5,
            pad_d: 2,
            pad_h: 2,
            pad_w: 2,
            ..basic_config()
        };
        let result = generate_direct3d_ptx(&cfg, "f32", SmVersion::Sm80);
        assert!(result.is_err());
    }

    #[test]
    fn direct3d_ptx_rejects_invalid_precision() {
        let cfg = basic_config();
        let result = generate_direct3d_ptx(&cfg, "bf16", SmVersion::Sm80);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Config helpers
    // -----------------------------------------------------------------------

    #[test]
    fn effective_kernel_sizes() {
        let cfg = Conv3dConfig {
            dilation_d: 2,
            dilation_h: 3,
            dilation_w: 1,
            ..basic_config()
        };
        assert_eq!(cfg.effective_kernel_d(), 5); // 2*(3-1)+1
        assert_eq!(cfg.effective_kernel_h(), 7); // 3*(3-1)+1
        assert_eq!(cfg.effective_kernel_w(), 3); // 1*(3-1)+1
    }

    #[test]
    fn channels_per_group() {
        let cfg = Conv3dConfig {
            in_channels: 64,
            out_channels: 128,
            groups: 4,
            ..basic_config()
        };
        assert_eq!(cfg.in_channels_per_group(), 16);
        assert_eq!(cfg.out_channels_per_group(), 32);
    }

    // -----------------------------------------------------------------------
    // Conv3dPlan — new() constructor
    // -----------------------------------------------------------------------

    #[test]
    fn plan_new_u32_constructor() {
        let plan = Conv3dPlan::new(basic_config(), 2, 8, 8, 8);
        assert!(plan.is_ok());
        let plan = plan.expect("plan new should succeed in test");
        assert_eq!(plan.batch_size, 2);
        assert_eq!(plan.input_d, 8);
        assert_eq!(plan.output_d, 8);
    }

    #[test]
    fn plan_new_rejects_zero_batch() {
        let result = Conv3dPlan::new(basic_config(), 0, 8, 8, 8);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Conv3dPlan — output_shape()
    // -----------------------------------------------------------------------

    #[test]
    fn output_shape_basic() {
        let plan =
            Conv3dPlan::new(basic_config(), 4, 16, 16, 16).expect("plan should succeed in test");
        let (n, c_out, od, oh, ow) = plan.output_shape();
        assert_eq!(n, 4);
        assert_eq!(c_out, 128);
        assert_eq!(od, 16);
        assert_eq!(oh, 16);
        assert_eq!(ow, 16);
    }

    #[test]
    fn output_shape_strided() {
        let cfg = Conv3dConfig {
            stride_d: 2,
            stride_h: 2,
            stride_w: 2,
            ..basic_config()
        };
        let plan = Conv3dPlan::new(cfg, 1, 8, 8, 8).expect("plan should succeed in test");
        let (n, c_out, od, oh, ow) = plan.output_shape();
        assert_eq!(n, 1);
        assert_eq!(c_out, 128);
        assert_eq!(od, 4);
        assert_eq!(oh, 4);
        assert_eq!(ow, 4);
    }

    #[test]
    fn output_shape_grouped() {
        let cfg = Conv3dConfig {
            groups: 4,
            ..basic_config()
        };
        let plan = Conv3dPlan::new(cfg, 2, 8, 8, 8).expect("plan should succeed in test");
        let (n, c_out, _od, _oh, _ow) = plan.output_shape();
        assert_eq!(n, 2);
        assert_eq!(c_out, 128);
    }

    // -----------------------------------------------------------------------
    // Conv3dPlan — generate_forward_ptx()
    // -----------------------------------------------------------------------

    #[test]
    fn forward_ptx_3x3x3() {
        let plan =
            Conv3dPlan::new(basic_config(), 1, 8, 8, 8).expect("plan should succeed in test");
        assert_eq!(plan.algorithm, Conv3dAlgorithm::DirectSmall);
        let ptx = plan.generate_forward_ptx();
        assert!(ptx.is_ok());
        let ptx = ptx.expect("forward PTX should succeed in test");
        assert!(ptx.contains("direct3d_3x3x3"));
    }

    #[test]
    fn forward_ptx_5x5x5_im2col() {
        let cfg = Conv3dConfig {
            kernel_d: 5,
            kernel_h: 5,
            kernel_w: 5,
            pad_d: 2,
            pad_h: 2,
            pad_w: 2,
            ..basic_config()
        };
        let plan = Conv3dPlan::new(cfg, 1, 8, 8, 8).expect("plan should succeed in test");
        assert_eq!(plan.algorithm, Conv3dAlgorithm::Im2colGemm);
        let ptx = plan.generate_forward_ptx();
        assert!(ptx.is_ok());
        let ptx = ptx.expect("forward PTX should succeed in test");
        assert!(ptx.contains("im2col3d"));
    }

    #[test]
    fn forward_ptx_1x1x1_direct() {
        let cfg = Conv3dConfig {
            kernel_d: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_d: 1,
            stride_h: 1,
            stride_w: 1,
            pad_d: 0,
            pad_h: 0,
            pad_w: 0,
            ..basic_config()
        };
        let plan = Conv3dPlan::new(cfg, 1, 8, 8, 8).expect("plan should succeed in test");
        assert_eq!(plan.algorithm, Conv3dAlgorithm::Direct1x1x1);
        let ptx = plan.generate_forward_ptx();
        assert!(ptx.is_ok());
    }

    // -----------------------------------------------------------------------
    // Conv3dPlan — generate_backward_data_ptx()
    // -----------------------------------------------------------------------

    #[test]
    fn backward_data_ptx_basic() {
        let plan =
            Conv3dPlan::new(basic_config(), 1, 8, 8, 8).expect("plan should succeed in test");
        let ptx = plan.generate_backward_data_ptx();
        assert!(ptx.is_ok());
        let ptx = ptx.expect("backward data PTX should succeed in test");
        assert!(ptx.contains("col2im3d"));
    }

    #[test]
    fn backward_data_ptx_dilated() {
        let cfg = Conv3dConfig {
            dilation_d: 2,
            dilation_h: 2,
            dilation_w: 2,
            pad_d: 2,
            pad_h: 2,
            pad_w: 2,
            ..basic_config()
        };
        let plan = Conv3dPlan::new(cfg, 2, 16, 16, 16).expect("plan should succeed in test");
        let ptx = plan.generate_backward_data_ptx();
        assert!(ptx.is_ok());
    }

    // -----------------------------------------------------------------------
    // Conv3dPlan — generate_backward_filter_ptx()
    // -----------------------------------------------------------------------

    #[test]
    fn backward_filter_ptx_basic() {
        let plan =
            Conv3dPlan::new(basic_config(), 1, 8, 8, 8).expect("plan should succeed in test");
        let ptx = plan.generate_backward_filter_ptx();
        assert!(ptx.is_ok());
        let ptx = ptx.expect("backward filter PTX should succeed in test");
        assert!(ptx.contains("wgrad3d"));
    }

    #[test]
    fn backward_filter_ptx_grouped() {
        let cfg = Conv3dConfig {
            groups: 4,
            ..basic_config()
        };
        let plan = Conv3dPlan::new(cfg, 2, 8, 8, 8).expect("plan should succeed in test");
        let ptx = plan.generate_backward_filter_ptx();
        assert!(ptx.is_ok());
        let ptx = ptx.expect("wgrad PTX should succeed in test");
        assert!(ptx.contains("wgrad3d"));
    }

    #[test]
    fn backward_filter_ptx_strided() {
        let cfg = Conv3dConfig {
            stride_d: 2,
            stride_h: 2,
            stride_w: 2,
            ..basic_config()
        };
        let plan = Conv3dPlan::new(cfg, 1, 16, 16, 16).expect("plan should succeed in test");
        let ptx = plan.generate_backward_filter_ptx();
        assert!(ptx.is_ok());
    }

    // -----------------------------------------------------------------------
    // PTX generation — wgrad3d
    // -----------------------------------------------------------------------

    #[test]
    fn wgrad3d_ptx_f32() {
        let cfg = basic_config();
        let ptx = generate_wgrad3d_ptx(&cfg, 1, 8, 8, 8, "f32", SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("wgrad PTX should succeed in test");
        assert!(ptx.contains("wgrad3d_f32"));
    }

    #[test]
    fn wgrad3d_ptx_f64() {
        let cfg = basic_config();
        let ptx = generate_wgrad3d_ptx(&cfg, 1, 8, 8, 8, "f64", SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("wgrad PTX should succeed in test");
        assert!(ptx.contains("wgrad3d_f64"));
    }

    #[test]
    fn wgrad3d_ptx_rejects_invalid_precision() {
        let cfg = basic_config();
        let result = generate_wgrad3d_ptx(&cfg, 1, 8, 8, 8, "f16", SmVersion::Sm80);
        assert!(result.is_err());
    }

    #[test]
    fn wgrad3d_ptx_contains_target() {
        let cfg = basic_config();
        let ptx = generate_wgrad3d_ptx(&cfg, 1, 8, 8, 8, "f32", SmVersion::Sm90);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("wgrad PTX should succeed in test");
        assert!(ptx.contains(".target sm_90"));
    }

    // -----------------------------------------------------------------------
    // workspace_size()
    // -----------------------------------------------------------------------

    #[test]
    fn workspace_size_via_plan_method() {
        let plan =
            Conv3dPlan::new(basic_config(), 1, 8, 8, 8).expect("plan should succeed in test");
        // DirectSmall still uses workspace for im2col
        assert!(plan.workspace_size() > 0);
    }

    #[test]
    fn workspace_size_1x1x1_is_zero() {
        let cfg = Conv3dConfig {
            kernel_d: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_d: 1,
            stride_h: 1,
            stride_w: 1,
            pad_d: 0,
            pad_h: 0,
            pad_w: 0,
            ..basic_config()
        };
        let plan = Conv3dPlan::new(cfg, 1, 8, 8, 8).expect("plan should succeed in test");
        assert_eq!(plan.workspace_size(), 0);
    }
}
