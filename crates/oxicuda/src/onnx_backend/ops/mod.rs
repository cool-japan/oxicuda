//! ONNX operator registry and common helpers.
//!
//! Maps ONNX operator type strings to host-side implementations
//! for simulated inference.

use std::collections::HashMap;

use super::ir::*;

mod activation;
mod conv;
mod elementwise;
mod matmul;
mod misc;
mod norm;
mod reduction;
mod shape_ops;

// ─── Input / attribute helpers ──────────────────────────────

/// Extract a required input tensor by positional index.
pub fn get_required_input<'a>(
    inputs: &[Option<&'a OnnxTensor>],
    idx: usize,
    name: &str,
) -> OnnxResult<&'a OnnxTensor> {
    inputs
        .get(idx)
        .and_then(|o| *o)
        .ok_or_else(|| OnnxError::InvalidData(format!("missing required input '{name}' at {idx}")))
}

/// Extract an optional input tensor by positional index.
pub fn get_optional_input<'a>(
    inputs: &[Option<&'a OnnxTensor>],
    idx: usize,
) -> Option<&'a OnnxTensor> {
    inputs.get(idx).and_then(|o| *o)
}

/// Get an integer attribute with a default.
pub fn get_int_attr(
    attrs: &HashMap<String, AttributeValue>,
    name: &str,
    default: i64,
) -> OnnxResult<i64> {
    match attrs.get(name) {
        Some(v) => v.as_int(),
        None => Ok(default),
    }
}

/// Get a float attribute with a default.
pub fn get_float_attr(
    attrs: &HashMap<String, AttributeValue>,
    name: &str,
    default: f64,
) -> OnnxResult<f64> {
    match attrs.get(name) {
        Some(v) => v.as_float(),
        None => Ok(default),
    }
}

/// Get an optional integer-list attribute.
pub fn get_ints_attr<'a>(
    attrs: &'a HashMap<String, AttributeValue>,
    name: &str,
) -> OnnxResult<Option<&'a [i64]>> {
    match attrs.get(name) {
        Some(v) => Ok(Some(v.as_ints()?)),
        None => Ok(None),
    }
}

// ─── Common elementwise helpers ─────────────────────────────

/// Apply a binary f32 operation with broadcasting.
pub fn binary_elementwise_f32(
    inputs: &[Option<&OnnxTensor>],
    op: impl Fn(f32, f32) -> f32,
) -> OnnxResult<Vec<OnnxTensor>> {
    let a = get_required_input(inputs, 0, "A")?;
    let b = get_required_input(inputs, 1, "B")?;
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let total: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let mut result = Vec::with_capacity(total);
    for i in 0..total {
        let multi = flat_to_multi(i, &out_shape);
        let a_idx = broadcast_index(&multi, &a.shape, &out_shape);
        let b_idx = broadcast_index(&multi, &b.shape, &out_shape);
        result.push(op(a_data[a_idx], b_data[b_idx]));
    }
    Ok(vec![OnnxTensor::from_f32(&result, out_shape)])
}

/// Apply a unary f32 operation element-wise.
pub fn unary_elementwise_f32(
    inputs: &[Option<&OnnxTensor>],
    op: impl Fn(f32) -> f32,
) -> OnnxResult<Vec<OnnxTensor>> {
    let a = get_required_input(inputs, 0, "X")?;
    let data = a.as_f32()?;
    let result: Vec<f32> = data.iter().map(|&x| op(x)).collect();
    Ok(vec![OnnxTensor::from_f32(&result, a.shape.clone())])
}

// ─── Operator registry ──────────────────────────────────────

/// Registry that maps ONNX operator names to host-side implementations.
pub struct OpRegistry;

impl OpRegistry {
    /// Create a new operator registry.
    pub fn new() -> Self {
        Self
    }

    /// Execute an operator by name.
    #[allow(clippy::too_many_lines)]
    pub fn execute(
        &self,
        op_type: &str,
        inputs: &[Option<&OnnxTensor>],
        attrs: &HashMap<String, AttributeValue>,
    ) -> OnnxResult<Vec<OnnxTensor>> {
        match op_type {
            // Elementwise
            "Add" => elementwise::execute_add(inputs, attrs),
            "Sub" => elementwise::execute_sub(inputs, attrs),
            "Mul" => elementwise::execute_mul(inputs, attrs),
            "Div" => elementwise::execute_div(inputs, attrs),
            "Relu" => elementwise::execute_relu(inputs, attrs),
            "Sigmoid" => elementwise::execute_sigmoid(inputs, attrs),
            "Tanh" => elementwise::execute_tanh(inputs, attrs),
            "Exp" => elementwise::execute_exp(inputs, attrs),
            "Log" => elementwise::execute_log(inputs, attrs),
            "Sqrt" => elementwise::execute_sqrt(inputs, attrs),
            "Abs" => elementwise::execute_abs(inputs, attrs),
            "Neg" => elementwise::execute_neg(inputs, attrs),
            "LeakyRelu" => elementwise::execute_leaky_relu(inputs, attrs),
            "Elu" => elementwise::execute_elu(inputs, attrs),
            "Selu" => elementwise::execute_selu(inputs, attrs),
            "Softplus" => elementwise::execute_softplus(inputs, attrs),
            "Clip" => elementwise::execute_clip(inputs, attrs),
            "Where" => elementwise::execute_where(inputs, attrs),
            "Cast" => elementwise::execute_cast(inputs, attrs),
            "Pow" => elementwise::execute_pow(inputs, attrs),
            "Ceil" => elementwise::execute_ceil(inputs, attrs),
            "Floor" => elementwise::execute_floor(inputs, attrs),
            "Round" => elementwise::execute_round(inputs, attrs),
            "Sign" => elementwise::execute_sign(inputs, attrs),
            "Reciprocal" => elementwise::execute_reciprocal(inputs, attrs),
            // Reduction
            "ReduceSum" => reduction::execute_reduce_sum(inputs, attrs),
            "ReduceMean" => reduction::execute_reduce_mean(inputs, attrs),
            "ReduceMax" => reduction::execute_reduce_max(inputs, attrs),
            "ReduceMin" => reduction::execute_reduce_min(inputs, attrs),
            "ReduceProd" => reduction::execute_reduce_prod(inputs, attrs),
            // Matrix
            "MatMul" => matmul::execute_matmul(inputs, attrs),
            "Gemm" => matmul::execute_gemm(inputs, attrs),
            // Convolution / Pooling
            "Conv" => conv::execute_conv(inputs, attrs),
            "ConvTranspose" => conv::execute_conv_transpose(inputs, attrs),
            "MaxPool" => conv::execute_max_pool(inputs, attrs),
            "AveragePool" => conv::execute_average_pool(inputs, attrs),
            "GlobalAveragePool" => conv::execute_global_average_pool(inputs, attrs),
            // Normalization
            "BatchNormalization" => norm::execute_batch_normalization(inputs, attrs),
            "LayerNormalization" => norm::execute_layer_normalization(inputs, attrs),
            "InstanceNormalization" => norm::execute_instance_normalization(inputs, attrs),
            "GroupNormalization" => norm::execute_group_normalization(inputs, attrs),
            "FlashAttention" => norm::execute_flash_attention(inputs, attrs),
            // Shape manipulation
            "Reshape" => shape_ops::execute_reshape(inputs, attrs),
            "Transpose" => shape_ops::execute_transpose(inputs, attrs),
            "Squeeze" => shape_ops::execute_squeeze(inputs, attrs),
            "Unsqueeze" => shape_ops::execute_unsqueeze(inputs, attrs),
            "Flatten" => shape_ops::execute_flatten(inputs, attrs),
            "Concat" => shape_ops::execute_concat(inputs, attrs),
            "Split" => shape_ops::execute_split(inputs, attrs),
            "Gather" => shape_ops::execute_gather(inputs, attrs),
            "Slice" => shape_ops::execute_slice(inputs, attrs),
            "Pad" => shape_ops::execute_pad(inputs, attrs),
            "Expand" => shape_ops::execute_expand(inputs, attrs),
            "Tile" => shape_ops::execute_tile(inputs, attrs),
            // Activation
            "Softmax" => activation::execute_softmax(inputs, attrs),
            "LogSoftmax" => activation::execute_log_softmax(inputs, attrs),
            // Misc
            "Identity" => misc::execute_identity(inputs, attrs),
            "Dropout" => misc::execute_dropout(inputs, attrs),
            "Constant" => misc::execute_constant(inputs, attrs),
            "Shape" => misc::execute_shape(inputs, attrs),
            "Size" => misc::execute_size(inputs, attrs),
            // Fused ops (produced by fusion passes)
            "FusedConvBn" => conv::execute_conv(inputs, attrs),
            "FusedConvBnRelu" => conv::execute_conv(inputs, attrs),
            "FusedBiasedMatMul" => matmul::execute_gemm(inputs, attrs),
            "FusedFMA" => elementwise::execute_fma(inputs, attrs),
            _ => Err(OnnxError::UnsupportedOp(op_type.to_string())),
        }
    }

    /// Check whether an operator is supported.
    pub fn is_supported(&self, op_type: &str) -> bool {
        SUPPORTED_OPS.contains(&op_type)
    }

    /// List all supported operator names.
    pub fn supported_ops(&self) -> &'static [&'static str] {
        SUPPORTED_OPS
    }
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

static SUPPORTED_OPS: &[&str] = &[
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Relu",
    "Sigmoid",
    "Tanh",
    "Exp",
    "Log",
    "Sqrt",
    "Abs",
    "Neg",
    "LeakyRelu",
    "Elu",
    "Selu",
    "Softplus",
    "Clip",
    "Where",
    "Cast",
    "Pow",
    "Ceil",
    "Floor",
    "Round",
    "Sign",
    "Reciprocal",
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "MatMul",
    "Gemm",
    "Conv",
    "ConvTranspose",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "BatchNormalization",
    "LayerNormalization",
    "InstanceNormalization",
    "GroupNormalization",
    "FlashAttention",
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Flatten",
    "Concat",
    "Split",
    "Gather",
    "Slice",
    "Pad",
    "Expand",
    "Tile",
    "Softmax",
    "LogSoftmax",
    "Identity",
    "Dropout",
    "Constant",
    "Shape",
    "Size",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_supported_count() {
        let reg = OpRegistry::new();
        assert!(reg.supported_ops().len() >= 60);
    }

    #[test]
    fn test_registry_unsupported_op() {
        let reg = OpRegistry::new();
        let result = reg.execute("NonExistentOp", &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_is_supported() {
        let reg = OpRegistry::new();
        assert!(reg.is_supported("Add"));
        assert!(reg.is_supported("Conv"));
        assert!(reg.is_supported("Softmax"));
        assert!(!reg.is_supported("FakeOp"));
    }
}
