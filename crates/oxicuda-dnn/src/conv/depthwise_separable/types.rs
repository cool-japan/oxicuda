//! Configuration types for depthwise separable convolutions.

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// ActivationType
// ---------------------------------------------------------------------------

/// Activation function subset for depthwise separable convolutions.
///
/// These are the activations most commonly used in MobileNet, EfficientNet,
/// and related efficient architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationType {
    /// No activation (identity).
    None,
    /// ReLU: `max(0, x)`.
    Relu,
    /// Capped ReLU: `min(6, max(0, x))`.
    Relu6,
    /// Sigmoid Linear Unit (Swish): `x * sigmoid(x)`.
    Silu,
    /// Hard Swish: `x * min(6, max(0, x + 3)) / 6`.
    HardSwish,
}

impl ActivationType {
    /// Returns a short identifier used in kernel names.
    #[must_use]
    pub fn kernel_suffix(self) -> &'static str {
        match self {
            Self::None => "identity",
            Self::Relu => "relu",
            Self::Relu6 => "relu6",
            Self::Silu => "silu",
            Self::HardSwish => "hardswish",
        }
    }
}

// ---------------------------------------------------------------------------
// DepthwiseSeparableConfig
// ---------------------------------------------------------------------------

/// Configuration for a depthwise separable convolution.
///
/// The depthwise stage applies one filter per input channel (possibly with a
/// `depth_multiplier > 1`). The pointwise stage is a 1×1 convolution that
/// projects from `channels * depth_multiplier` intermediate channels down to
/// `out_channels`.
#[derive(Debug, Clone)]
pub struct DepthwiseSeparableConfig {
    /// Number of input channels (depthwise operates per-channel).
    pub channels: usize,
    /// Number of output channels (pointwise 1×1 projects to this).
    pub out_channels: usize,
    /// Depthwise kernel height.
    pub kernel_h: usize,
    /// Depthwise kernel width.
    pub kernel_w: usize,
    /// Stride height.
    pub stride_h: usize,
    /// Stride width.
    pub stride_w: usize,
    /// Padding height.
    pub pad_h: usize,
    /// Padding width.
    pub pad_w: usize,
    /// Dilation height for depthwise conv.
    pub dilation_h: usize,
    /// Dilation width for depthwise conv.
    pub dilation_w: usize,
    /// Channel multiplier for depthwise (typically 1).
    pub depth_multiplier: usize,
    /// Activation after depthwise stage.
    pub depthwise_activation: ActivationType,
    /// Activation after pointwise stage.
    pub pointwise_activation: ActivationType,
    /// Whether to fuse batch normalisation after depthwise.
    pub depthwise_bn: bool,
    /// Whether to fuse batch normalisation after pointwise.
    pub pointwise_bn: bool,
}

impl DepthwiseSeparableConfig {
    /// Computes output spatial dimensions.
    ///
    /// The depthwise stage controls spatial dimensions:
    /// ```text
    /// out_h = (in_h + 2*pad_h - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
    /// out_w = (in_w + 2*pad_w - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
    /// ```
    #[must_use]
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let effective_kh = self.dilation_h * (self.kernel_h.saturating_sub(1)) + 1;
        let effective_kw = self.dilation_w * (self.kernel_w.saturating_sub(1)) + 1;

        let padded_h = input_h + 2 * self.pad_h;
        let padded_w = input_w + 2 * self.pad_w;

        let out_h = if padded_h >= effective_kh {
            (padded_h - effective_kh) / self.stride_h.max(1) + 1
        } else {
            0
        };
        let out_w = if padded_w >= effective_kw {
            (padded_w - effective_kw) / self.stride_w.max(1) + 1
        } else {
            0
        };

        (out_h, out_w)
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] when parameters are invalid.
    pub fn validate(&self) -> DnnResult<()> {
        if self.kernel_h == 0 || self.kernel_w == 0 {
            return Err(DnnError::InvalidArgument(
                "depthwise separable: kernel dimensions must be > 0".into(),
            ));
        }
        if self.stride_h == 0 || self.stride_w == 0 {
            return Err(DnnError::InvalidArgument(
                "depthwise separable: stride must be > 0".into(),
            ));
        }
        if self.dilation_h == 0 || self.dilation_w == 0 {
            return Err(DnnError::InvalidArgument(
                "depthwise separable: dilation must be > 0".into(),
            ));
        }
        if self.channels == 0 {
            return Err(DnnError::InvalidArgument(
                "depthwise separable: channels must be > 0".into(),
            ));
        }
        if self.out_channels == 0 {
            return Err(DnnError::InvalidArgument(
                "depthwise separable: out_channels must be > 0".into(),
            ));
        }
        if self.depth_multiplier == 0 {
            return Err(DnnError::InvalidArgument(
                "depthwise separable: depth_multiplier must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Total intermediate channels after depthwise stage.
    #[must_use]
    pub fn depthwise_out_channels(&self) -> usize {
        self.channels * self.depth_multiplier
    }
}

// ---------------------------------------------------------------------------
// DepthwiseSeparablePlan
// ---------------------------------------------------------------------------

/// Shared-memory budget (bytes) for deciding whether full fusion is feasible.
pub(super) const SHARED_MEM_BUDGET: usize = 48 * 1024; // 48 KiB

/// Execution plan for a fused depthwise separable convolution.
///
/// Pre-computes all derived dimensions and workspace requirements.
#[derive(Debug, Clone)]
pub struct DepthwiseSeparablePlan {
    /// The validated configuration.
    pub config: DepthwiseSeparableConfig,
    /// Batch size.
    pub batch_size: usize,
    /// Input spatial height.
    pub input_h: usize,
    /// Input spatial width.
    pub input_w: usize,
    /// Computed output height.
    pub output_h: usize,
    /// Computed output width.
    pub output_w: usize,
    /// Workspace bytes for intermediate depthwise output (when not fused).
    pub workspace_bytes: usize,
    /// Whether both stages can be fully fused into one kernel.
    pub is_fully_fused: bool,
}

impl DepthwiseSeparablePlan {
    /// Creates an execution plan.
    ///
    /// Validates the config, computes output dimensions, workspace, and
    /// decides whether the two stages can be fully fused.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] on bad parameters.
    pub fn create(
        config: DepthwiseSeparableConfig,
        batch_size: usize,
        input_h: usize,
        input_w: usize,
    ) -> DnnResult<Self> {
        config.validate()?;

        if batch_size == 0 {
            return Err(DnnError::InvalidArgument(
                "depthwise separable: batch_size must be > 0".into(),
            ));
        }
        if input_h == 0 || input_w == 0 {
            return Err(DnnError::InvalidArgument(
                "depthwise separable: input spatial dimensions must be > 0".into(),
            ));
        }

        let (output_h, output_w) = config.output_size(input_h, input_w);

        if output_h == 0 || output_w == 0 {
            return Err(DnnError::InvalidDimension(format!(
                "depthwise separable: computed output size is zero ({output_h}x{output_w})"
            )));
        }

        // Determine whether full fusion into a single kernel is feasible.
        // The intermediate depthwise output for one sample must fit in shared
        // memory: dw_out_channels * output_h * output_w * sizeof(f32).
        let dw_out_channels = config.depthwise_out_channels();
        let intermediate_elements_per_sample = dw_out_channels * output_h * output_w;
        let intermediate_bytes_per_sample = intermediate_elements_per_sample * 4; // f32.
        let is_fully_fused = intermediate_bytes_per_sample <= SHARED_MEM_BUDGET;

        // Workspace: when not fully fused we need to materialise the
        // intermediate depthwise output for the entire batch.
        let workspace_bytes = if is_fully_fused {
            0
        } else {
            batch_size * intermediate_elements_per_sample * 4
        };

        Ok(Self {
            config,
            batch_size,
            input_h,
            input_w,
            output_h,
            output_w,
            workspace_bytes,
            is_fully_fused,
        })
    }

    /// Returns workspace size in bytes.
    #[must_use]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes
    }
}
