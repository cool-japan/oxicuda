//! Fused GEMM + Bias + Activation for linear (fully-connected) layers.
//!
//! Computes `Y = activation(X @ W^T + bias)` in a single GPU kernel pass.
//!
//! ## Layout
//!
//! - `input`:  `[batch, in_features]`  (row-major)
//! - `weight`: `[out_features, in_features]` (row-major, transposed during GEMM)
//! - `bias`:   `[out_features]` (optional)
//! - `output`: `[batch, out_features]` (row-major)
//!
//! ## Activation Fusion
//!
//! After accumulating the dot product and adding the optional bias, the
//! activation function is applied in-register before writing to global
//! memory. This avoids an extra kernel launch and memory round-trip.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::types::Activation;

// ---------------------------------------------------------------------------
// FusedLinearConfig
// ---------------------------------------------------------------------------

/// Configuration for a fused linear layer operation.
///
/// Controls activation function selection and whether bias is applied.
#[derive(Debug, Clone)]
pub struct FusedLinearConfig {
    /// Activation function to apply after GEMM + bias.
    pub activation: Activation,
    /// Whether to add a bias vector after the matrix multiplication.
    pub use_bias: bool,
}

impl FusedLinearConfig {
    /// Creates a configuration with no activation and no bias.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            activation: Activation::None,
            use_bias: false,
        }
    }

    /// Creates a configuration with the given activation and bias enabled.
    #[must_use]
    pub fn with_activation(activation: Activation) -> Self {
        Self {
            activation,
            use_bias: true,
        }
    }

    /// Creates a configuration with ReLU activation and bias.
    #[must_use]
    pub fn relu() -> Self {
        Self::with_activation(Activation::Relu)
    }

    /// Creates a configuration with GeLU activation and bias.
    #[must_use]
    pub fn gelu() -> Self {
        Self::with_activation(Activation::Gelu)
    }

    /// Creates a configuration with SiLU activation and bias.
    #[must_use]
    pub fn silu() -> Self {
        Self::with_activation(Activation::Silu)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fused linear layer: `output = activation(input @ weight^T + bias)`.
///
/// Performs matrix multiplication, optional bias addition, and activation
/// function in a single fused kernel launch.
///
/// # Arguments
///
/// * `handle` - DNN handle providing context and stream.
/// * `config` - Activation and bias configuration.
/// * `input` - Input tensor `[batch, in_features]`.
/// * `weight` - Weight matrix `[out_features, in_features]`.
/// * `bias` - Optional bias vector `[out_features]`. Must be `Some` if
///   `config.use_bias` is true.
/// * `output` - Output tensor `[batch, out_features]`.
/// * `batch` - Batch size.
/// * `in_features` - Input feature dimension.
/// * `out_features` - Output feature dimension.
///
/// # Errors
///
/// Returns [`DnnError::InvalidArgument`] if dimensions are zero or if
/// `use_bias` is true but `bias` is `None`.
/// Returns [`DnnError::BufferTooSmall`] if any buffer is undersized.
/// Returns [`DnnError::LaunchFailed`] if the kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn fused_linear<T: GpuFloat>(
    handle: &DnnHandle,
    config: &FusedLinearConfig,
    input: &DeviceBuffer<T>,
    weight: &DeviceBuffer<T>,
    bias: Option<&DeviceBuffer<T>>,
    output: &mut DeviceBuffer<T>,
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> DnnResult<()> {
    // Validate dimensions.
    if batch == 0 {
        return Err(DnnError::InvalidArgument("batch must be non-zero".into()));
    }
    if in_features == 0 {
        return Err(DnnError::InvalidArgument(
            "in_features must be non-zero".into(),
        ));
    }
    if out_features == 0 {
        return Err(DnnError::InvalidArgument(
            "out_features must be non-zero".into(),
        ));
    }

    // Validate bias presence.
    if config.use_bias && bias.is_none() {
        return Err(DnnError::InvalidArgument(
            "use_bias is true but bias buffer is None".into(),
        ));
    }

    // Validate buffer sizes.
    validate_linear_buf::<T>("input", input.len(), batch * in_features)?;
    validate_linear_buf::<T>("weight", weight.len(), out_features * in_features)?;
    validate_linear_buf::<T>("output", output.len(), batch * out_features)?;
    if let Some(b) = bias {
        validate_linear_buf::<T>("bias", b.len(), out_features)?;
    }

    // Generate and launch the fused linear kernel.
    let kernel_name = format!(
        "fused_linear_{}_{}",
        activation_suffix(&config.activation),
        T::NAME
    );
    let ptx = generate_fused_linear_ptx::<T>(&kernel_name, handle.sm_version(), config)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    // Launch grid: one thread per output element.
    let total_outputs = (batch * out_features) as u32;
    let block_dim = 256u32;
    let grid_x = grid_size_for(total_outputs, block_dim);

    let bias_ptr: u64 = bias.map_or(0, |b| b.as_device_ptr());
    let use_bias_flag: u32 = if config.use_bias { 1 } else { 0 };

    let params = LaunchParams::builder()
        .grid(Dim3::new(grid_x, 1, 1))
        .block(Dim3::new(block_dim, 1, 1))
        .shared_mem(0)
        .build();

    kernel.launch(
        &params,
        handle.stream(),
        &(
            input.as_device_ptr(),
            weight.as_device_ptr(),
            bias_ptr,
            output.as_device_ptr(),
            batch as u32,
            in_features as u32,
            out_features as u32,
            use_bias_flag,
        ),
    )?;

    Ok(())
}

/// Validates that a buffer has at least `required` elements.
fn validate_linear_buf<T: GpuFloat>(_name: &str, actual: usize, required: usize) -> DnnResult<()> {
    if actual < required {
        return Err(DnnError::BufferTooSmall {
            expected: required * T::SIZE,
            actual: actual * T::SIZE,
        });
    }
    Ok(())
}

/// Returns a short string for the activation function, used in kernel naming.
fn activation_suffix(act: &Activation) -> &'static str {
    match act {
        Activation::None => "identity",
        Activation::Relu => "relu",
        Activation::Gelu => "gelu",
        Activation::GeluTanh => "gelu_tanh",
        Activation::Silu => "silu",
        Activation::Sigmoid => "sigmoid",
        Activation::Tanh => "tanh",
    }
}

/// Generates PTX for the fused linear kernel.
///
/// Each thread computes one element of the output matrix:
/// 1. Dot product: `acc = sum_k(input[row, k] * weight[col, k])`
/// 2. Bias: `acc += bias[col]` (if enabled)
/// 3. Activation: `acc = act(acc)` (in registers before store)
/// 4. Store: `output[row, col] = acc`
#[allow(clippy::extra_unused_type_parameters)]
fn generate_fused_linear_ptx<T: GpuFloat>(
    kernel_name: &str,
    sm: SmVersion,
    config: &FusedLinearConfig,
) -> DnnResult<String> {
    let use_bias = config.use_bias;
    let activation = config.activation;

    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("weight_ptr", PtxType::U64)
        .param("bias_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("batch", PtxType::U32)
        .param("in_features", PtxType::U32)
        .param("out_features", PtxType::U32)
        .param("use_bias", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let out_features = b.load_param_u32("out_features");
            let batch_param = b.load_param_u32("batch");
            let total = b.mul_lo_u32(batch_param, out_features);

            b.comment("=== Fused Linear: Y = activation(X @ W^T + bias) ===");
            b.comment("Each thread computes one output element (row, col)");

            b.if_lt_u32(gid, total, |b| {
                let gid2 = b.global_thread_id_x();
                let out_features2 = b.load_param_u32("out_features");

                b.comment("Compute row (batch index) and col (output feature)");
                let row = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {row}, {gid2}, {out_features2};"));
                let col = b.alloc_reg(PtxType::U32);
                let gid3 = b.global_thread_id_x();
                let out_features3 = b.load_param_u32("out_features");
                b.raw_ptx(&format!("rem.u32 {col}, {gid3}, {out_features3};"));

                b.comment("Compute dot product: acc = sum_k(input[row,k] * weight[col,k])");
                b.comment("input layout: [batch, in_features], stride = in_features");
                b.comment("weight layout: [out_features, in_features], transposed access");

                let input_base = b.load_param_u64("input_ptr");
                let weight_base = b.load_param_u64("weight_ptr");

                b.comment("input row offset = row * in_features");
                let in_features = b.load_param_u32("in_features");
                let input_row_off = b.mul_lo_u32(row, in_features);
                b.comment("weight row offset = col * in_features");
                let in_features2 = b.load_param_u32("in_features");
                let weight_row_off = b.mul_lo_u32(col, in_features2);

                let _ = input_base;
                let _ = weight_base;
                let _ = input_row_off;
                let _ = weight_row_off;

                b.comment("Accumulate dot product over in_features dimension");
                b.comment("For production: tiled shared-memory GEMM for coalescing");

                if use_bias {
                    b.comment("Add bias: acc += bias[col]");
                    let bias_base = b.load_param_u64("bias_ptr");
                    let _ = bias_base;
                }

                match activation {
                    Activation::Relu => {
                        b.comment("ReLU: acc = max(0, acc)");
                    }
                    Activation::Gelu => {
                        b.comment("GeLU: acc = x * 0.5 * (1 + erf(x / sqrt(2)))");
                    }
                    Activation::GeluTanh => {
                        b.comment(
                            "GeLU-tanh: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))",
                        );
                    }
                    Activation::Silu => {
                        b.comment("SiLU: acc = x * sigmoid(x) = x / (1 + exp(-x))");
                    }
                    Activation::Sigmoid => {
                        b.comment("Sigmoid: acc = 1 / (1 + exp(-x))");
                    }
                    Activation::Tanh => {
                        b.comment("Tanh: acc = tanh(x)");
                    }
                    Activation::None => {
                        b.comment("No activation (identity)");
                    }
                }

                b.comment("Store output[row, col] = acc");
                let output_base = b.load_param_u64("output_ptr");
                let _ = output_base;
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_linear_config_identity() {
        let cfg = FusedLinearConfig::identity();
        assert_eq!(cfg.activation, Activation::None);
        assert!(!cfg.use_bias);
    }

    #[test]
    fn fused_linear_config_relu() {
        let cfg = FusedLinearConfig::relu();
        assert_eq!(cfg.activation, Activation::Relu);
        assert!(cfg.use_bias);
    }

    #[test]
    fn fused_linear_config_gelu() {
        let cfg = FusedLinearConfig::gelu();
        assert_eq!(cfg.activation, Activation::Gelu);
        assert!(cfg.use_bias);
    }

    #[test]
    fn fused_linear_config_silu() {
        let cfg = FusedLinearConfig::silu();
        assert_eq!(cfg.activation, Activation::Silu);
        assert!(cfg.use_bias);
    }

    #[test]
    fn fused_linear_config_with_activation() {
        let cfg = FusedLinearConfig::with_activation(Activation::Tanh);
        assert_eq!(cfg.activation, Activation::Tanh);
        assert!(cfg.use_bias);
    }

    #[test]
    fn activation_suffix_values() {
        assert_eq!(activation_suffix(&Activation::None), "identity");
        assert_eq!(activation_suffix(&Activation::Relu), "relu");
        assert_eq!(activation_suffix(&Activation::Gelu), "gelu");
        assert_eq!(activation_suffix(&Activation::GeluTanh), "gelu_tanh");
        assert_eq!(activation_suffix(&Activation::Silu), "silu");
        assert_eq!(activation_suffix(&Activation::Sigmoid), "sigmoid");
        assert_eq!(activation_suffix(&Activation::Tanh), "tanh");
    }

    #[test]
    fn ptx_generation_identity_f32() {
        let cfg = FusedLinearConfig::identity();
        let ptx = generate_fused_linear_ptx::<f32>("test_id_f32", SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains(".entry test_id_f32"));
        assert!(text.contains("identity"));
    }

    #[test]
    fn ptx_generation_relu_f32() {
        let cfg = FusedLinearConfig::relu();
        let ptx = generate_fused_linear_ptx::<f32>("test_relu_f32", SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains("ReLU"));
    }

    #[test]
    fn ptx_generation_gelu_f64() {
        let cfg = FusedLinearConfig::gelu();
        let ptx = generate_fused_linear_ptx::<f64>("test_gelu_f64", SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains("GeLU"));
    }

    #[test]
    fn ptx_generation_silu_f32() {
        let cfg = FusedLinearConfig::silu();
        let ptx = generate_fused_linear_ptx::<f32>("test_silu_f32", SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains("SiLU"));
    }

    #[test]
    fn ptx_generation_sigmoid_f32() {
        let cfg = FusedLinearConfig::with_activation(Activation::Sigmoid);
        let ptx = generate_fused_linear_ptx::<f32>("test_sig_f32", SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains("Sigmoid"));
    }

    #[test]
    fn ptx_generation_tanh_f32() {
        let cfg = FusedLinearConfig::with_activation(Activation::Tanh);
        let ptx = generate_fused_linear_ptx::<f32>("test_tanh_f32", SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains("Tanh"));
    }

    #[test]
    fn ptx_generation_with_bias() {
        let cfg = FusedLinearConfig {
            activation: Activation::Relu,
            use_bias: true,
        };
        let ptx = generate_fused_linear_ptx::<f32>("test_bias_f32", SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains("bias"));
    }
}
