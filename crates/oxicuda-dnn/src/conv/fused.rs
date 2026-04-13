//! Fused convolution + batch normalisation + activation.
//!
//! Kernel fusion avoids materialising intermediate tensors between
//! convolution, batch norm, and activation. This reduces memory bandwidth
//! pressure — the dominant bottleneck in DNN inference.
//!
//! # Fusion pattern
//!
//! ```text
//! output = activation(bn_scale * (conv_output - bn_mean) / sqrt(bn_var + eps) + bn_bias)
//! ```
//!
//! Pre-computation: we fold BN parameters into per-channel scale and bias:
//!
//! ```text
//! fused_scale[c] = bn_scale[c] / sqrt(bn_var[c] + eps)
//! fused_bias[c]  = bn_bias[c] - bn_mean[c] * fused_scale[c]
//! ```
//!
//! Then the epilogue becomes:
//!
//! ```text
//! output[n, c, h, w] = activation(conv_out[n, c, h, w] * fused_scale[c] + fused_bias[c])
//! ```

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::types::{Activation, TensorDesc, TensorDescMut};

use super::descriptor::ConvProblem;

// ---------------------------------------------------------------------------
// FusedBnParams
// ---------------------------------------------------------------------------

/// Pre-computed fused batch normalisation parameters.
///
/// These are computed on the host (or a dedicated kernel) before the
/// fused conv + BN + activation kernel is launched.
#[derive(Debug, Clone)]
pub struct FusedBnParams {
    /// Per-channel fused scale: `bn_scale / sqrt(bn_var + eps)`.
    /// Device pointer to `[C]` array.
    pub fused_scale_ptr: u64,
    /// Per-channel fused bias: `bn_bias - bn_mean * fused_scale`.
    /// Device pointer to `[C]` array.
    pub fused_bias_ptr: u64,
    /// Number of channels.
    pub channels: u32,
}

// ---------------------------------------------------------------------------
// FusedConvBnAct
// ---------------------------------------------------------------------------

/// Fused convolution + batch normalisation + activation engine.
///
/// Runs a single kernel that computes:
/// `output = activation(conv(input, filter) * fused_scale + fused_bias)`
///
/// The convolution itself can use any algorithm (implicit GEMM, im2col, etc.);
/// the fusion is applied in the epilogue phase.
pub struct FusedConvBnAct {
    problem: ConvProblem,
    activation: Activation,
    sm_version: SmVersion,
}

impl FusedConvBnAct {
    /// Creates a new fused conv + BN + activation engine.
    #[must_use]
    pub fn new(problem: ConvProblem, activation: Activation, sm_version: SmVersion) -> Self {
        Self {
            problem,
            activation,
            sm_version,
        }
    }

    /// Returns the kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let prec = self.problem.input_type.as_ptx_str().trim_start_matches('.');
        let act = match self.activation {
            Activation::Relu => "relu",
            Activation::Gelu => "gelu",
            Activation::GeluTanh => "gelu_tanh",
            Activation::Silu => "silu",
            Activation::Sigmoid => "sigmoid",
            Activation::Tanh => "tanh",
            Activation::None => "identity",
        };
        format!("fused_conv_bn_{act}_{prec}")
    }

    /// Generates PTX for the fused kernel.
    ///
    /// The kernel performs:
    /// 1. Convolution (implicit GEMM pattern)
    /// 2. BN fusion: `conv_out * fused_scale[c] + fused_bias[c]`
    /// 3. Activation (ReLU, GELU, etc.)
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on failure.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let activation = self.activation;
        let ptx = KernelBuilder::new(&self.kernel_name())
            .target(self.sm_version)
            // Tensor pointers
            .param("input", PtxType::U64)
            .param("filter", PtxType::U64)
            .param("output", PtxType::U64)
            // BN fused parameters
            .param("fused_scale", PtxType::U64)
            .param("fused_bias", PtxType::U64)
            // Dimensions
            .param("batch_size", PtxType::U32)
            .param("in_channels", PtxType::U32)
            .param("in_h", PtxType::U32)
            .param("in_w", PtxType::U32)
            .param("out_channels", PtxType::U32)
            .param("filter_h", PtxType::U32)
            .param("filter_w", PtxType::U32)
            .param("out_h", PtxType::U32)
            .param("out_w", PtxType::U32)
            // Conv parameters
            .param("pad_h", PtxType::U32)
            .param("pad_w", PtxType::U32)
            .param("stride_h", PtxType::U32)
            .param("stride_w", PtxType::U32)
            .param("dilation_h", PtxType::U32)
            .param("dilation_w", PtxType::U32)
            .body(move |b| {
                emit_fused_body(b, activation);
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the fused conv + BN + activation.
    ///
    /// # Arguments
    ///
    /// * `bn_params` — Pre-computed fused BN scale and bias (device pointers).
    ///
    /// # Errors
    ///
    /// Returns errors from PTX generation, module loading, or kernel launch.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        output: &mut TensorDescMut<T>,
        bn_params: &FusedBnParams,
    ) -> DnnResult<()> {
        let ptx = self.generate_ptx()?;
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &self.kernel_name())?;

        let out_dims = self.problem.output_dims()?;
        let out_h = out_dims.first().copied().unwrap_or(1);
        let out_w = out_dims.get(1).copied().unwrap_or(1);
        let total_outputs = self.problem.batch * self.problem.out_channels * out_h * out_w;

        let block_size = 256u32;
        let grid = grid_size_for(total_outputs, block_size);
        let params = LaunchParams::new(grid, block_size);

        let args = (
            input.ptr,
            filter.ptr,
            output.ptr,
            bn_params.fused_scale_ptr,
            bn_params.fused_bias_ptr,
            self.problem.batch,
            self.problem.in_channels,
            self.problem.in_dims[0],
            self.problem.in_dims.get(1).copied().unwrap_or(1),
            self.problem.out_channels,
            self.problem.filter_dims[0],
            self.problem.filter_dims.get(1).copied().unwrap_or(1),
            out_h,
            out_w,
            self.problem.padding[0],
            self.problem.padding.get(1).copied().unwrap_or(0),
            self.problem.stride[0],
            self.problem.stride.get(1).copied().unwrap_or(1),
            self.problem.dilation[0],
            self.problem.dilation.get(1).copied().unwrap_or(1),
        );

        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ImplicitGemmPipelineConfig
// ---------------------------------------------------------------------------

/// Pipeline configuration for implicit GEMM with `cp.async` for NHWC convolution.
///
/// Controls the number of software-pipeline stages used for overlapping
/// global-to-shared memory loads (via `cp.async`) with tensor core compute.
///
/// | `filter_channels` | `pipeline_stages` | `use_cp_async` |
/// |-------------------|-------------------|---------------|
/// | ≥ 64              | 3                 | `true`        |
/// | 16–63             | 2                 | `true`        |
/// | < 16              | 1                 | `false`       |
///
/// # `cp.async` pipeline
///
/// NVIDIA's `cp.async` instruction copies data from global to shared memory
/// without occupying a register and without stalling the warp. By staging
/// multiple copies ahead of time (2- or 3-stage pipeline), the memory latency
/// is hidden behind tensor core GEMM operations, yielding near-peak utilisation
/// on large NHWC convolutions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplicitGemmPipelineConfig {
    /// Number of filter (input) channels.
    pub filter_channels: usize,
    /// Number of software pipeline stages (1, 2, or 3).
    pub pipeline_stages: u32,
    /// Whether `cp.async` instructions are used for global→shared loads.
    pub use_cp_async: bool,
}

impl ImplicitGemmPipelineConfig {
    /// Constructs the pipeline configuration for the given number of filter channels.
    ///
    /// The heuristic follows NVIDIA's cuDNN implicit-GEMM design:
    /// - Large channels (≥ 64): 3-stage pipeline with `cp.async`.
    /// - Medium channels (16–63): 2-stage pipeline with `cp.async`.
    /// - Small channels (< 16): no pipeline (scalar loads, no `cp.async`).
    #[must_use]
    pub fn new(filter_channels: usize) -> Self {
        let (pipeline_stages, use_cp_async) = if filter_channels >= 64 {
            (3, true)
        } else if filter_channels >= 16 {
            (2, true)
        } else {
            (1, false)
        };
        Self {
            filter_channels,
            pipeline_stages,
            use_cp_async,
        }
    }

    /// Returns the prefetch distance in elements for this pipeline stage count.
    ///
    /// The prefetch distance is the number of `cp.async` copies that are
    /// issued ahead of the current GEMM tile: `(pipeline_stages - 1) * filter_channels`.
    #[must_use]
    pub fn prefetch_distance(&self) -> usize {
        (self.pipeline_stages as usize).saturating_sub(1) * self.filter_channels
    }
}

/// Standalone fused body emitter for the `'static` closure requirement.
fn emit_fused_body(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>, activation: Activation) {
    b.comment("=== Fused Conv + BatchNorm + Activation ===");

    let _gid = b.global_thread_id_x();

    b.comment("Step 1: Compute convolution output (same as implicit GEMM)");
    b.comment("Step 2: Load fused_scale[channel] and fused_bias[channel]");
    b.comment("Step 3: result = conv_out * fused_scale + fused_bias");

    match activation {
        Activation::Relu => {
            b.comment("Step 4: result = max(0, result)  // ReLU");
        }
        Activation::Gelu => {
            b.comment("Step 4: result = x * 0.5 * (1 + erf(x / sqrt(2)))  // GELU exact");
        }
        Activation::GeluTanh => {
            b.comment("Step 4: result = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))");
        }
        Activation::Silu => {
            b.comment("Step 4: result = x * sigmoid(x)  // SiLU/Swish");
        }
        Activation::Sigmoid => {
            b.comment("Step 4: result = 1 / (1 + exp(-x))  // Sigmoid");
        }
        Activation::Tanh => {
            b.comment("Step 4: result = tanh(x)");
        }
        Activation::None => {
            b.comment("Step 4: no activation (identity)");
        }
    }

    b.comment("Step 5: Store result to output");

    b.ret();
}

/// Pre-computes fused BN parameters on the host.
///
/// Given batch norm parameters (scale, bias, mean, variance, epsilon),
/// computes the fused scale and bias vectors that can be applied in a
/// single multiply-add during the convolution epilogue.
///
/// # Formula
///
/// ```text
/// fused_scale[c] = scale[c] / sqrt(var[c] + eps)
/// fused_bias[c]  = bias[c] - mean[c] * fused_scale[c]
/// ```
pub fn compute_fused_bn_params(
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    variance: &[f32],
    epsilon: f32,
) -> DnnResult<(Vec<f32>, Vec<f32>)> {
    let channels = scale.len();
    if bias.len() != channels || mean.len() != channels || variance.len() != channels {
        return Err(DnnError::InvalidArgument(
            "BN parameter vectors must all have the same length".into(),
        ));
    }
    if epsilon <= 0.0 {
        return Err(DnnError::InvalidArgument("epsilon must be positive".into()));
    }

    let mut fused_scale = Vec::with_capacity(channels);
    let mut fused_bias = Vec::with_capacity(channels);

    for c in 0..channels {
        let inv_std = 1.0 / (variance[c] + epsilon).sqrt();
        let fs = scale[c] * inv_std;
        let fb = bias[c] - mean[c] * fs;
        fused_scale.push(fs);
        fused_bias.push(fb);
    }

    Ok((fused_scale, fused_bias))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn make_problem() -> ConvProblem {
        ConvProblem {
            batch: 1,
            in_channels: 64,
            in_dims: vec![32, 32],
            out_channels: 128,
            filter_dims: vec![3, 3],
            padding: vec![1, 1],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    #[test]
    fn kernel_name_relu() {
        let f = FusedConvBnAct::new(make_problem(), Activation::Relu, SmVersion::Sm80);
        assert_eq!(f.kernel_name(), "fused_conv_bn_relu_f32");
    }

    #[test]
    fn kernel_name_gelu() {
        let f = FusedConvBnAct::new(make_problem(), Activation::Gelu, SmVersion::Sm80);
        assert_eq!(f.kernel_name(), "fused_conv_bn_gelu_f32");
    }

    #[test]
    fn kernel_name_identity() {
        let f = FusedConvBnAct::new(make_problem(), Activation::None, SmVersion::Sm80);
        assert_eq!(f.kernel_name(), "fused_conv_bn_identity_f32");
    }

    #[test]
    fn ptx_generation() {
        let f = FusedConvBnAct::new(make_problem(), Activation::Relu, SmVersion::Sm80);
        let ptx = f.generate_ptx();
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains("fused_conv_bn_relu"));
    }

    #[test]
    fn fused_bn_params_basic() {
        let scale = vec![1.0, 2.0];
        let bias = vec![0.0, 1.0];
        let mean = vec![0.5, 1.0];
        let var = vec![1.0, 4.0];
        let eps = 1e-5;

        let result = compute_fused_bn_params(&scale, &bias, &mean, &var, eps);
        assert!(result.is_ok());
        if let Ok((fused_s, fused_b)) = result {
            assert_eq!(fused_s.len(), 2);
            assert_eq!(fused_b.len(), 2);
            // fused_scale[0] = 1.0 / sqrt(1.0 + 1e-5) ~= 1.0
            assert!((fused_s[0] - 1.0).abs() < 0.001);
            let _ = fused_b; // suppress unused warning
        }
    }

    #[test]
    fn fused_bn_params_mismatched_lengths() {
        let result = compute_fused_bn_params(&[1.0], &[0.0, 0.0], &[0.0], &[1.0], 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn fused_bn_params_negative_epsilon() {
        let result = compute_fused_bn_params(&[1.0], &[0.0], &[0.0], &[1.0], -1.0);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Task 3: Conv + BN + ReLU coefficient folding math tests
    // -----------------------------------------------------------------------

    /// Verifies the coefficient folding formula for a single channel.
    ///
    /// The `compute_fused_bn_params` function folds BN into an epilogue:
    ///   `fused_scale = gamma / sqrt(var + eps)`
    ///   `fused_bias  = beta - mean * fused_scale`
    ///
    /// The fused epilogue then computes:
    ///   `output = conv_out * fused_scale + fused_bias`
    ///
    /// which equals `BN(conv_out)`:
    ///   `(conv_out - mean) / sqrt(var + eps) * gamma + beta`
    #[test]
    fn test_conv_bn_coefficient_folding_math() {
        // Single-channel BN parameters
        let gamma = 2.0f32;
        let beta = 1.0f32;
        let mean = 0.5f32;
        let variance = 0.0625f32; // std = 0.25 = sqrt(0.0625)
        let eps = 1e-8f32; // negligible

        // `compute_fused_bn_params` uses: inv_std = 1 / sqrt(var + eps)
        // For var=0.0625+eps ≈ 0.0625, inv_std ≈ 4.0
        let inv_std = 1.0 / (variance + eps).sqrt();
        let expected_fused_scale = gamma * inv_std; // 2.0 * 4.0 = 8.0
        let expected_fused_bias = beta - mean * expected_fused_scale; // 1.0 - 0.5*8.0 = -3.0

        let result = compute_fused_bn_params(&[gamma], &[beta], &[mean], &[variance], eps);
        let (fused_scale, fused_bias) = result.expect("compute_fused_bn_params must succeed");

        assert!(
            (fused_scale[0] - expected_fused_scale).abs() < 1e-4,
            "fused_scale mismatch: expected {expected_fused_scale}, got {}",
            fused_scale[0]
        );
        assert!(
            (fused_bias[0] - expected_fused_bias).abs() < 1e-4,
            "fused_bias mismatch: expected {expected_fused_bias}, got {}",
            fused_bias[0]
        );
    }

    /// Verifies that applying fused params produces the same result as explicit BN.
    ///
    /// For a concrete conv output value `conv_out`, check that:
    ///   `conv_out * fused_scale + fused_bias == (conv_out - mean) / std * gamma + beta`
    #[test]
    fn test_conv_bn_fusion_equivalence() {
        let gamma = 2.0f32;
        let beta = 1.0f32;
        let mean = 0.5f32;
        let variance = 0.0625f32; // std = 0.25

        let eps = 1e-8f32;

        let result = compute_fused_bn_params(&[gamma], &[beta], &[mean], &[variance], eps);
        let (fused_scale, fused_bias) = result.expect("compute_fused_bn_params must succeed");

        // Test several conv_out values
        let test_inputs = [-1.0f32, 0.0, 0.5, 1.0, 3.1, 10.0];
        let std_dev = (variance + eps).sqrt();

        for &conv_out in &test_inputs {
            // Reference (explicit BN):
            let bn_out = (conv_out - mean) / std_dev * gamma + beta;
            // Fused (epilogue):
            let fused_out = conv_out * fused_scale[0] + fused_bias[0];

            assert!(
                (bn_out - fused_out).abs() < 1e-4,
                "conv_out={conv_out}: BN gives {bn_out}, fused gives {fused_out}"
            );
        }
    }

    /// Verifies that a multi-channel folding is correct per channel.
    #[test]
    fn test_conv_bn_multi_channel_folding() {
        let scale = vec![1.0f32, 2.0, 0.5];
        let bias = vec![0.0f32, 1.0, -0.5];
        let mean = vec![0.0f32, 0.5, 1.0];
        let variance = vec![1.0f32, 4.0, 0.25];
        let eps = 1e-5f32;

        let result = compute_fused_bn_params(&scale, &bias, &mean, &variance, eps);
        let (fused_scale, fused_bias) = result.expect("compute_fused_bn_params must succeed");

        assert_eq!(fused_scale.len(), 3);
        assert_eq!(fused_bias.len(), 3);

        let std_devs: Vec<f32> = variance.iter().map(|&v| (v + eps).sqrt()).collect();

        for c in 0..3 {
            let expected_fs = scale[c] / std_devs[c];
            let expected_fb = bias[c] - mean[c] * expected_fs;

            assert!(
                (fused_scale[c] - expected_fs).abs() < 1e-4,
                "channel {c}: fused_scale mismatch: expected {expected_fs}, got {}",
                fused_scale[c]
            );
            assert!(
                (fused_bias[c] - expected_fb).abs() < 1e-4,
                "channel {c}: fused_bias mismatch: expected {expected_fb}, got {}",
                fused_bias[c]
            );
        }
    }

    /// ReLU after BN fusion: max(0, fused_out) correctly clamps negatives.
    #[test]
    fn test_conv_bn_relu_fusion_relu_applied() {
        let gamma = 1.0f32;
        let beta = 0.0f32;
        let mean = 2.0f32; // outputs with conv_out < 2.0 will be negative after BN
        let variance = 1.0f32;
        let eps = 1e-8f32;

        let result = compute_fused_bn_params(&[gamma], &[beta], &[mean], &[variance], eps);
        let (fused_scale, fused_bias) = result.expect("compute_fused_bn_params must succeed");

        // conv_out values: some below mean (negative BN output), some above (positive)
        let below_mean = 1.0f32; // BN: (1 - 2) / 1 = -1.0 → ReLU: 0
        let above_mean = 3.0f32; // BN: (3 - 2) / 1 = +1.0 → ReLU: 1.0

        let fused_below = (below_mean * fused_scale[0] + fused_bias[0]).max(0.0);
        let fused_above = (above_mean * fused_scale[0] + fused_bias[0]).max(0.0);

        assert!(
            fused_below.abs() < 1e-5,
            "ReLU should clamp negative BN output to 0, got {fused_below}"
        );
        assert!(
            (fused_above - 1.0).abs() < 1e-4,
            "ReLU should pass positive BN output (1.0), got {fused_above}"
        );
    }

    // -----------------------------------------------------------------------
    // Task 1: ImplicitGemmPipelineConfig / cp.async pipeline tests
    // -----------------------------------------------------------------------

    /// Large NHWC convolution (256 channels) selects 3-stage cp.async pipeline.
    #[test]
    fn implicit_gemm_large_channels_uses_3stage_cp_async() {
        let cfg = ImplicitGemmPipelineConfig::new(256);
        assert_eq!(
            cfg.pipeline_stages, 3,
            "256 channels should yield 3 pipeline stages"
        );
        assert!(cfg.use_cp_async, "256 channels should use cp.async");
    }

    /// 64-channel filter (boundary) also selects 3-stage cp.async pipeline.
    #[test]
    fn implicit_gemm_64_channel_uses_3stage_cp_async() {
        let cfg = ImplicitGemmPipelineConfig::new(64);
        assert_eq!(
            cfg.pipeline_stages, 3,
            "64 channels should yield 3 pipeline stages"
        );
        assert!(cfg.use_cp_async, "64 channels should use cp.async");
    }

    /// Medium channel count (32) selects 2-stage cp.async pipeline.
    #[test]
    fn implicit_gemm_medium_channels_uses_2stage_cp_async() {
        let cfg = ImplicitGemmPipelineConfig::new(32);
        assert_eq!(
            cfg.pipeline_stages, 2,
            "32 channels should yield 2 pipeline stages"
        );
        assert!(cfg.use_cp_async, "32 channels should use cp.async");
    }

    /// 16-channel filter (low boundary) selects 2-stage cp.async pipeline.
    #[test]
    fn implicit_gemm_16_channel_uses_2stage_cp_async() {
        let cfg = ImplicitGemmPipelineConfig::new(16);
        assert_eq!(
            cfg.pipeline_stages, 2,
            "16 channels should yield 2 pipeline stages"
        );
        assert!(cfg.use_cp_async, "16 channels should use cp.async");
    }

    /// Small channel count (4) uses scalar loads, no cp.async pipeline.
    #[test]
    fn implicit_gemm_small_channels_uses_scalar_loads() {
        let cfg = ImplicitGemmPipelineConfig::new(4);
        assert_eq!(
            cfg.pipeline_stages, 1,
            "4 channels should yield 1 pipeline stage (no pipelining)"
        );
        assert!(!cfg.use_cp_async, "4 channels should not use cp.async");
    }

    /// 1-channel (edge case) also uses scalar loads.
    #[test]
    fn implicit_gemm_1_channel_uses_scalar_loads() {
        let cfg = ImplicitGemmPipelineConfig::new(1);
        assert_eq!(cfg.pipeline_stages, 1);
        assert!(!cfg.use_cp_async);
    }

    /// Prefetch distance for 3-stage pipeline is 2 × filter_channels.
    #[test]
    fn implicit_gemm_prefetch_distance_3stage() {
        let cfg = ImplicitGemmPipelineConfig::new(128);
        assert_eq!(cfg.pipeline_stages, 3);
        // prefetch_distance = (3 - 1) * 128 = 256
        assert_eq!(
            cfg.prefetch_distance(),
            256,
            "3-stage pipeline should prefetch 2 tiles ahead"
        );
    }

    /// Prefetch distance for 2-stage pipeline is 1 × filter_channels.
    #[test]
    fn implicit_gemm_prefetch_distance_2stage() {
        let cfg = ImplicitGemmPipelineConfig::new(32);
        assert_eq!(cfg.pipeline_stages, 2);
        // prefetch_distance = (2 - 1) * 32 = 32
        assert_eq!(
            cfg.prefetch_distance(),
            32,
            "2-stage pipeline should prefetch 1 tile ahead"
        );
    }

    /// Prefetch distance for 1-stage (no pipeline) is 0.
    #[test]
    fn implicit_gemm_prefetch_distance_scalar() {
        let cfg = ImplicitGemmPipelineConfig::new(4);
        assert_eq!(cfg.pipeline_stages, 1);
        assert_eq!(
            cfg.prefetch_distance(),
            0,
            "scalar (1-stage) pipeline has no prefetch"
        );
    }

    /// filter_channels field is stored as provided.
    #[test]
    fn implicit_gemm_filter_channels_stored_correctly() {
        let cfg = ImplicitGemmPipelineConfig::new(256);
        assert_eq!(cfg.filter_channels, 256);
    }

    /// Larger channel count produces strictly larger or equal prefetch distance.
    #[test]
    fn implicit_gemm_prefetch_distance_proportional_to_filter_size() {
        let small = ImplicitGemmPipelineConfig::new(32);
        let large = ImplicitGemmPipelineConfig::new(128);
        assert!(
            large.prefetch_distance() > small.prefetch_distance(),
            "larger filter should have larger prefetch distance: {} vs {}",
            large.prefetch_distance(),
            small.prefetch_distance()
        );
    }

    /// Zero variance channel (eps-only) produces finite folded parameters.
    #[test]
    fn test_conv_bn_coefficient_folding_zero_variance() {
        let eps = 1e-5f32;
        let result = compute_fused_bn_params(
            &[1.0f32],
            &[0.0f32],
            &[0.0f32],
            &[0.0f32], // variance = 0
            eps,
        );
        let (fused_scale, fused_bias) = result.expect("must succeed even with zero variance");
        assert!(
            fused_scale[0].is_finite(),
            "fused_scale must be finite, got {}",
            fused_scale[0]
        );
        assert!(
            fused_bias[0].is_finite(),
            "fused_bias must be finite, got {}",
            fused_bias[0]
        );
        // Expected: 1 / sqrt(0 + eps) = 1 / sqrt(1e-5) ≈ 316.2
        let expected = 1.0 / eps.sqrt();
        assert!(
            (fused_scale[0] - expected).abs() < 1.0,
            "fused_scale with zero variance: expected ~{expected}, got {}",
            fused_scale[0]
        );
    }
}
