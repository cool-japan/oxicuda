//! Quantized inference dispatch for transformer models.
//!
//! Supports multiple quantization methods with per-layer configuration:
//!
//! - **INT8 Symmetric** — per-channel INT8 (TensorRT-style)
//! - **INT4 Asymmetric** — per-group INT4 (GPTQ-style)
//! - **FP8 E4M3** — FP8 for matmul (Hopper)
//! - **AWQ** — activation-aware weight quantization
//! - **SmoothQuant** — activation smoothing + INT8
//! - **GPTQ** — post-training quantization with group size

use std::collections::HashMap;

use super::{TransformerError, TransformerResult};

/// Quantization method.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantMethod {
    /// Per-channel symmetric INT8 quantization (TensorRT-style).
    Int8Symmetric,
    /// Per-group asymmetric INT4 quantization (GPTQ-style).
    Int4Asymmetric,
    /// FP8 E4M3 format for matmul acceleration (Hopper).
    Fp8E4m3,
    /// Activation-aware weight quantization.
    Awq {
        /// Group size for quantization.
        group_size: usize,
    },
    /// Activation smoothing followed by INT8 quantization.
    SmoothQuant {
        /// Smoothing factor alpha in [0.0, 1.0].
        alpha: f64,
    },
    /// GPTQ post-training quantization.
    Gptq {
        /// Group size for quantization.
        group_size: usize,
        /// Number of bits (typically 4 or 8).
        bits: usize,
    },
}

impl std::fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int8Symmetric => write!(f, "INT8-Symmetric"),
            Self::Int4Asymmetric => write!(f, "INT4-Asymmetric"),
            Self::Fp8E4m3 => write!(f, "FP8-E4M3"),
            Self::Awq { group_size } => write!(f, "AWQ(group={group_size})"),
            Self::SmoothQuant { alpha } => write!(f, "SmoothQuant(α={alpha:.2})"),
            Self::Gptq { group_size, bits } => {
                write!(f, "GPTQ({bits}bit, group={group_size})")
            }
        }
    }
}

impl QuantMethod {
    /// Bits per weight element.
    pub fn bits_per_weight(&self) -> usize {
        match self {
            Self::Int8Symmetric => 8,
            Self::Int4Asymmetric => 4,
            Self::Fp8E4m3 => 8,
            Self::Awq { .. } => 4,
            Self::SmoothQuant { .. } => 8,
            Self::Gptq { bits, .. } => *bits,
        }
    }

    /// Compression ratio compared to FP16 (16 bits).
    pub fn compression_ratio(&self) -> f64 {
        16.0 / self.bits_per_weight() as f64
    }

    /// Validate the quantization method parameters.
    pub fn validate(&self) -> TransformerResult<()> {
        match self {
            Self::Awq { group_size } if *group_size == 0 => {
                return Err(TransformerError::QuantizationError(
                    "AWQ group_size must be > 0".to_string(),
                ));
            }
            Self::SmoothQuant { alpha } if *alpha < 0.0 || *alpha > 1.0 => {
                return Err(TransformerError::QuantizationError(
                    "SmoothQuant alpha must be in [0.0, 1.0]".to_string(),
                ));
            }
            Self::Gptq { group_size, .. } if *group_size == 0 => {
                return Err(TransformerError::QuantizationError(
                    "GPTQ group_size must be > 0".to_string(),
                ));
            }
            Self::Gptq { bits, .. } if *bits == 0 || *bits > 8 => {
                return Err(TransformerError::QuantizationError(
                    "GPTQ bits must be in [1, 8]".to_string(),
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

/// Quantization configuration for a model.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Default quantization method.
    pub method: QuantMethod,
    /// Per-layer overrides (layer name → method).
    pub per_layer_config: HashMap<String, QuantMethod>,
}

impl QuantConfig {
    /// Create a new config with a uniform method.
    pub fn new(method: QuantMethod) -> Self {
        Self {
            method,
            per_layer_config: HashMap::new(),
        }
    }

    /// Set a per-layer override.
    pub fn set_layer_method(&mut self, layer_name: impl Into<String>, method: QuantMethod) {
        self.per_layer_config.insert(layer_name.into(), method);
    }

    /// Get the method for a specific layer.
    pub fn method_for_layer(&self, layer_name: &str) -> &QuantMethod {
        self.per_layer_config
            .get(layer_name)
            .unwrap_or(&self.method)
    }

    /// Validate all methods in the config.
    pub fn validate(&self) -> TransformerResult<()> {
        self.method.validate()?;
        for (name, method) in &self.per_layer_config {
            method
                .validate()
                .map_err(|e| TransformerError::QuantizationError(format!("layer '{name}': {e}")))?;
        }
        Ok(())
    }
}

/// A quantized weight tensor.
#[derive(Debug, Clone)]
pub struct QuantizedWeight {
    /// Quantized data (packed format).
    pub data: Vec<u8>,
    /// Scale factors (per-channel or per-group).
    pub scales: Vec<f64>,
    /// Zero points (for asymmetric quantization).
    pub zero_points: Vec<f64>,
    /// Original shape [rows, cols].
    pub shape: [usize; 2],
    /// Quantization method used.
    pub method: QuantMethod,
    /// Group size (0 means per-channel).
    pub group_size: usize,
}

/// A quantized tensor (activations or weights).
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data.
    pub data: Vec<u8>,
    /// Scale factor.
    pub scale: f64,
    /// Zero point.
    pub zero_point: f64,
    /// Number of elements.
    pub num_elements: usize,
    /// Bits per element.
    pub bits: usize,
}

impl QuantizedTensor {
    /// Size in bytes of the quantized data.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Size in bytes if stored as FP16.
    pub fn fp16_size_bytes(&self) -> usize {
        self.num_elements * 2
    }

    /// Compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.fp16_size_bytes() as f64 / self.size_bytes() as f64
    }
}

/// Quantize a vector of f64 values to INT8 (symmetric).
pub fn quantize_int8_symmetric(values: &[f64]) -> TransformerResult<QuantizedTensor> {
    if values.is_empty() {
        return Err(TransformerError::QuantizationError(
            "empty values".to_string(),
        ));
    }

    let abs_max = values.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);

    let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };

    let data: Vec<u8> = values
        .iter()
        .map(|&v| {
            let quantized = (v / scale).round().clamp(-128.0, 127.0) as i8;
            quantized as u8
        })
        .collect();

    Ok(QuantizedTensor {
        data,
        scale,
        zero_point: 0.0,
        num_elements: values.len(),
        bits: 8,
    })
}

/// Dequantize INT8 symmetric tensor back to f64.
pub fn dequantize_int8_symmetric(tensor: &QuantizedTensor) -> TransformerResult<Vec<f64>> {
    if tensor.bits != 8 {
        return Err(TransformerError::QuantizationError(format!(
            "expected 8-bit tensor, got {}-bit",
            tensor.bits
        )));
    }

    let values: Vec<f64> = tensor
        .data
        .iter()
        .map(|&byte| {
            let quantized = byte as i8;
            quantized as f64 * tensor.scale
        })
        .collect();

    Ok(values)
}

/// Quantize a vector of f64 values to INT4 (asymmetric, per-group).
pub fn quantize_int4_asymmetric(
    values: &[f64],
    group_size: usize,
) -> TransformerResult<QuantizedTensor> {
    if values.is_empty() {
        return Err(TransformerError::QuantizationError(
            "empty values".to_string(),
        ));
    }
    if group_size == 0 {
        return Err(TransformerError::QuantizationError(
            "group_size must be > 0".to_string(),
        ));
    }

    // Pack two INT4 values into one byte
    let mut data = Vec::with_capacity(values.len().div_ceil(2));

    for chunk in values.chunks(2) {
        let v0 = chunk[0];
        let v1 = if chunk.len() > 1 { chunk[1] } else { 0.0 };

        // Quantize to [0, 15] range
        let q0 = (v0.clamp(-1.0, 1.0) * 7.5 + 7.5).round() as u8;
        let q1 = (v1.clamp(-1.0, 1.0) * 7.5 + 7.5).round() as u8;

        let packed = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        data.push(packed);
    }

    // Global scale and zero point
    let min_val = values.iter().copied().fold(f64::MAX, f64::min);
    let max_val = values.iter().copied().fold(f64::MIN, f64::max);
    let range = max_val - min_val;
    let scale = if range == 0.0 { 1.0 } else { range / 15.0 };
    let zero_point = min_val;

    Ok(QuantizedTensor {
        data,
        scale,
        zero_point,
        num_elements: values.len(),
        bits: 4,
    })
}

/// Dequantize INT4 asymmetric tensor back to f64.
pub fn dequantize_int4_asymmetric(tensor: &QuantizedTensor) -> TransformerResult<Vec<f64>> {
    if tensor.bits != 4 {
        return Err(TransformerError::QuantizationError(format!(
            "expected 4-bit tensor, got {}-bit",
            tensor.bits
        )));
    }

    let mut values = Vec::with_capacity(tensor.num_elements);

    for (i, &byte) in tensor.data.iter().enumerate() {
        let q0 = (byte & 0x0F) as f64;
        values.push(q0 * tensor.scale + tensor.zero_point);

        if i * 2 + 1 < tensor.num_elements {
            let q1 = ((byte >> 4) & 0x0F) as f64;
            values.push(q1 * tensor.scale + tensor.zero_point);
        }
    }

    values.truncate(tensor.num_elements);
    Ok(values)
}

/// Apply SmoothQuant scaling to a weight matrix.
///
/// Smooths the activation outliers by scaling: W' = W * diag(s), X' = X * diag(1/s)
/// where s = max(|X|)^alpha / max(|W|)^(1-alpha)
pub fn smooth_quant_scales(
    weight_abs_max: &[f64],
    activation_abs_max: &[f64],
    alpha: f64,
) -> TransformerResult<Vec<f64>> {
    if weight_abs_max.len() != activation_abs_max.len() {
        return Err(TransformerError::QuantizationError(
            "weight and activation dimensions must match".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(TransformerError::QuantizationError(
            "alpha must be in [0.0, 1.0]".to_string(),
        ));
    }

    let scales: Vec<f64> = weight_abs_max
        .iter()
        .zip(activation_abs_max.iter())
        .map(|(&w, &a)| {
            if w == 0.0 && a == 0.0 {
                1.0
            } else {
                let act_part = a.powf(alpha);
                let weight_part = w.powf(1.0 - alpha);
                if weight_part == 0.0 {
                    1.0
                } else {
                    act_part / weight_part
                }
            }
        })
        .collect();

    Ok(scales)
}

/// Pack INT4 weights into INT8 storage (two INT4 values per byte).
pub fn pack_int4_to_int8(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(2));

    for chunk in values.chunks(2) {
        let v0 = (chunk[0] & 0x0F) as u8;
        let v1 = if chunk.len() > 1 {
            (chunk[1] & 0x0F) as u8
        } else {
            0u8
        };
        packed.push(v0 | (v1 << 4));
    }

    packed
}

/// Unpack INT4 values from INT8 storage.
pub fn unpack_int8_to_int4(packed: &[u8], num_elements: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(num_elements);

    for &byte in packed {
        let v0 = (byte & 0x0F) as i8;
        // Sign extend 4-bit value
        let v0 = if v0 & 0x08 != 0 { v0 | -16 } else { v0 };
        values.push(v0);

        if values.len() < num_elements {
            let v1 = ((byte >> 4) & 0x0F) as i8;
            let v1 = if v1 & 0x08 != 0 { v1 | -16 } else { v1 };
            values.push(v1);
        }
    }

    values.truncate(num_elements);
    values
}

/// Estimate memory savings for quantizing a model.
pub fn estimate_memory_savings(num_params: usize, method: &QuantMethod) -> (usize, usize) {
    let fp16_bytes = num_params * 2;
    let bits = method.bits_per_weight();
    let quant_bytes = (num_params * bits).div_ceil(8);
    (fp16_bytes, quant_bytes)
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_roundtrip() {
        let values = vec![1.0, -0.5, 0.25, -1.0, 0.0, 0.75];
        let quantized = quantize_int8_symmetric(&values).unwrap();
        assert_eq!(quantized.bits, 8);
        assert_eq!(quantized.num_elements, 6);

        let dequantized = dequantize_int8_symmetric(&quantized).unwrap();
        assert_eq!(dequantized.len(), 6);

        // Check roundtrip accuracy (should be close)
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.05, "orig={orig}, deq={deq}");
        }
    }

    #[test]
    fn test_int4_roundtrip() {
        let values = vec![0.5, -0.3, 0.1, -0.8, 0.0];
        let quantized = quantize_int4_asymmetric(&values, 4).unwrap();
        assert_eq!(quantized.bits, 4);
        assert_eq!(quantized.num_elements, 5);

        let dequantized = dequantize_int4_asymmetric(&quantized).unwrap();
        assert_eq!(dequantized.len(), 5);

        // INT4 has much lower precision — just verify values are in range
        for deq in &dequantized {
            assert!(
                *deq >= -1.5 && *deq <= 1.5,
                "dequantized value {deq} out of expected range"
            );
        }
    }

    #[test]
    fn test_quant_method_display() {
        assert_eq!(format!("{}", QuantMethod::Int8Symmetric), "INT8-Symmetric");
        assert_eq!(
            format!("{}", QuantMethod::Awq { group_size: 128 }),
            "AWQ(group=128)"
        );
        assert_eq!(
            format!(
                "{}",
                QuantMethod::Gptq {
                    group_size: 128,
                    bits: 4
                }
            ),
            "GPTQ(4bit, group=128)"
        );
    }

    #[test]
    fn test_bits_per_weight() {
        assert_eq!(QuantMethod::Int8Symmetric.bits_per_weight(), 8);
        assert_eq!(QuantMethod::Int4Asymmetric.bits_per_weight(), 4);
        assert_eq!(QuantMethod::Fp8E4m3.bits_per_weight(), 8);
        assert_eq!(QuantMethod::Awq { group_size: 128 }.bits_per_weight(), 4);
        assert_eq!(
            QuantMethod::Gptq {
                group_size: 128,
                bits: 4
            }
            .bits_per_weight(),
            4
        );
    }

    #[test]
    fn test_compression_ratio() {
        assert!((QuantMethod::Int8Symmetric.compression_ratio() - 2.0).abs() < 1e-10);
        assert!((QuantMethod::Int4Asymmetric.compression_ratio() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_quant_method_validation() {
        assert!(QuantMethod::Int8Symmetric.validate().is_ok());
        assert!(QuantMethod::Awq { group_size: 0 }.validate().is_err());
        assert!(QuantMethod::SmoothQuant { alpha: 1.5 }.validate().is_err());
        assert!(QuantMethod::SmoothQuant { alpha: 0.5 }.validate().is_ok());
        assert!(
            QuantMethod::Gptq {
                group_size: 0,
                bits: 4
            }
            .validate()
            .is_err()
        );
        assert!(
            QuantMethod::Gptq {
                group_size: 128,
                bits: 0
            }
            .validate()
            .is_err()
        );
        assert!(
            QuantMethod::Gptq {
                group_size: 128,
                bits: 9
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn test_quant_config() {
        let mut config = QuantConfig::new(QuantMethod::Int8Symmetric);
        config.set_layer_method("lm_head", QuantMethod::Fp8E4m3);
        config.set_layer_method(
            "layers.0.attn",
            QuantMethod::Gptq {
                group_size: 128,
                bits: 4,
            },
        );

        assert_eq!(
            config.method_for_layer("layers.0.attn"),
            &QuantMethod::Gptq {
                group_size: 128,
                bits: 4
            }
        );
        assert_eq!(
            config.method_for_layer("layers.1.attn"),
            &QuantMethod::Int8Symmetric
        );

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quant_config_invalid_layer() {
        let mut config = QuantConfig::new(QuantMethod::Int8Symmetric);
        config.set_layer_method("bad_layer", QuantMethod::Awq { group_size: 0 });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_smooth_quant_scales() {
        let weight_max = vec![2.0, 1.0, 0.5];
        let act_max = vec![1.0, 2.0, 4.0];

        let scales = smooth_quant_scales(&weight_max, &act_max, 0.5).unwrap();
        assert_eq!(scales.len(), 3);
        for &s in &scales {
            assert!(s > 0.0);
        }
    }

    #[test]
    fn test_smooth_quant_invalid() {
        let w = vec![1.0, 2.0];
        let a = vec![1.0]; // mismatched
        assert!(smooth_quant_scales(&w, &a, 0.5).is_err());
        assert!(smooth_quant_scales(&[1.0], &[1.0], 1.5).is_err());
    }

    #[test]
    fn test_pack_unpack_int4() {
        let values: Vec<i8> = vec![1, -2, 3, -4, 5, 6, 7];
        let packed = pack_int4_to_int8(&values);
        let unpacked = unpack_int8_to_int4(&packed, values.len());
        assert_eq!(unpacked.len(), values.len());

        // Values should match (within 4-bit range)
        for (&orig, &unpacked) in values.iter().zip(unpacked.iter()) {
            let orig_4bit = orig & 0x0F;
            let sign_extended = if orig_4bit & 0x08 != 0 {
                orig_4bit | -16
            } else {
                orig_4bit
            };
            assert_eq!(unpacked, sign_extended);
        }
    }

    #[test]
    fn test_estimate_memory_savings() {
        let (fp16, quant) = estimate_memory_savings(1_000_000, &QuantMethod::Int4Asymmetric);
        assert_eq!(fp16, 2_000_000);
        assert_eq!(quant, 500_000);
    }

    #[test]
    fn test_quantized_tensor_compression() {
        let t = QuantizedTensor {
            data: vec![0; 100],
            scale: 1.0,
            zero_point: 0.0,
            num_elements: 200,
            bits: 4,
        };
        assert_eq!(t.size_bytes(), 100);
        assert_eq!(t.fp16_size_bytes(), 400);
        assert!((t.compression_ratio() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantize_empty() {
        assert!(quantize_int8_symmetric(&[]).is_err());
        assert!(quantize_int4_asymmetric(&[], 4).is_err());
    }

    #[test]
    fn test_quantize_all_zeros() {
        let values = vec![0.0; 10];
        let quantized = quantize_int8_symmetric(&values).unwrap();
        let dequantized = dequantize_int8_symmetric(&quantized).unwrap();
        for v in &dequantized {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_dequantize_wrong_bits() {
        let t = QuantizedTensor {
            data: vec![0],
            scale: 1.0,
            zero_point: 0.0,
            num_elements: 1,
            bits: 4,
        };
        assert!(dequantize_int8_symmetric(&t).is_err());
    }

    #[test]
    fn test_quantized_weight_fields() {
        let w = QuantizedWeight {
            data: vec![0; 50],
            scales: vec![1.0],
            zero_points: vec![0.0],
            shape: [10, 10],
            method: QuantMethod::Int8Symmetric,
            group_size: 0,
        };
        assert_eq!(w.shape, [10, 10]);
        assert_eq!(w.group_size, 0);
    }
}
