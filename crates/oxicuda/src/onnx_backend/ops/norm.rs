//! Normalization ONNX operators and FlashAttention.

use std::collections::HashMap;

use super::{get_float_attr, get_int_attr, get_required_input};
use crate::onnx_backend::ir::*;

/// BatchNormalization(X, scale, B, mean, var) -> normalized output.
/// Inference mode: Y = scale * (X - mean) / sqrt(var + epsilon) + B
pub fn execute_batch_normalization(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "X")?;
    let scale = get_required_input(inputs, 1, "scale")?;
    let bias = get_required_input(inputs, 2, "B")?;
    let mean = get_required_input(inputs, 3, "mean")?;
    let var = get_required_input(inputs, 4, "var")?;

    let x_data = x.as_f32()?;
    let scale_data = scale.as_f32()?;
    let bias_data = bias.as_f32()?;
    let mean_data = mean.as_f32()?;
    let var_data = var.as_f32()?;
    let epsilon = get_float_attr(attrs, "epsilon", 1e-5)? as f32;

    if x.shape.len() < 2 {
        return Err(OnnxError::ShapeMismatch(
            "BatchNorm requires at least 2D input".into(),
        ));
    }

    let c = x.shape[1];
    let total: usize = x.shape.iter().product();
    let spatial: usize = x.shape[2..].iter().product();
    let mut result = vec![0.0f32; total];

    for i in 0..total {
        let ch = (i / spatial) % c;
        let inv_std = 1.0 / (var_data[ch] + epsilon).sqrt();
        result[i] = scale_data[ch] * (x_data[i] - mean_data[ch]) * inv_std + bias_data[ch];
    }

    Ok(vec![OnnxTensor::from_f32(&result, x.shape.clone())])
}

/// LayerNormalization(X, scale, B?) -> normalized over last axis.
pub fn execute_layer_normalization(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "X")?;
    let scale = get_required_input(inputs, 1, "scale")?;
    let bias = inputs.get(2).and_then(|o| *o);
    let x_data = x.as_f32()?;
    let scale_data = scale.as_f32()?;
    let epsilon = get_float_attr(attrs, "epsilon", 1e-5)? as f32;
    let axis = get_int_attr(attrs, "axis", -1)?;

    let rank = x.shape.len() as i64;
    let norm_axis = if axis < 0 { rank + axis } else { axis } as usize;

    let outer: usize = x.shape[..norm_axis].iter().product();
    let inner: usize = x.shape[norm_axis..].iter().product();
    let total = outer * inner;
    let mut result = vec![0.0f32; total];

    let bias_data = if let Some(b) = bias {
        Some(b.as_f32()?)
    } else {
        None
    };

    for o in 0..outer {
        let base = o * inner;
        // Compute mean
        let mut sum = 0.0f32;
        for i in 0..inner {
            sum += x_data[base + i];
        }
        let mean = sum / inner as f32;

        // Compute variance
        let mut var_sum = 0.0f32;
        for i in 0..inner {
            let diff = x_data[base + i] - mean;
            var_sum += diff * diff;
        }
        let var = var_sum / inner as f32;
        let inv_std = 1.0 / (var + epsilon).sqrt();

        // Normalize
        for i in 0..inner {
            let norm = (x_data[base + i] - mean) * inv_std;
            let scaled = norm * scale_data[i % scale_data.len()];
            result[base + i] = if let Some(ref bd) = bias_data {
                scaled + bd[i % bd.len()]
            } else {
                scaled
            };
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, x.shape.clone())])
}

/// InstanceNormalization(X, scale, B) -> per-instance normalization.
pub fn execute_instance_normalization(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "X")?;
    let scale = get_required_input(inputs, 1, "scale")?;
    let bias = get_required_input(inputs, 2, "B")?;
    let x_data = x.as_f32()?;
    let scale_data = scale.as_f32()?;
    let bias_data = bias.as_f32()?;
    let epsilon = get_float_attr(attrs, "epsilon", 1e-5)? as f32;

    if x.shape.len() < 3 {
        return Err(OnnxError::ShapeMismatch(
            "InstanceNorm requires at least 3D input".into(),
        ));
    }

    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product();
    let total = n * c * spatial;
    let mut result = vec![0.0f32; total];

    for batch in 0..n {
        for ch in 0..c {
            let base = (batch * c + ch) * spatial;
            let mut sum = 0.0f32;
            for i in 0..spatial {
                sum += x_data[base + i];
            }
            let mean = sum / spatial as f32;

            let mut var_sum = 0.0f32;
            for i in 0..spatial {
                let diff = x_data[base + i] - mean;
                var_sum += diff * diff;
            }
            let var = var_sum / spatial as f32;
            let inv_std = 1.0 / (var + epsilon).sqrt();

            for i in 0..spatial {
                result[base + i] =
                    scale_data[ch] * (x_data[base + i] - mean) * inv_std + bias_data[ch];
            }
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, x.shape.clone())])
}

/// GroupNormalization(X, scale, bias) -> group normalization.
pub fn execute_group_normalization(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "X")?;
    let scale = get_required_input(inputs, 1, "scale")?;
    let bias = get_required_input(inputs, 2, "bias")?;
    let x_data = x.as_f32()?;
    let scale_data = scale.as_f32()?;
    let bias_data = bias.as_f32()?;
    let epsilon = get_float_attr(attrs, "epsilon", 1e-5)? as f32;
    let num_groups = get_int_attr(attrs, "num_groups", 1)? as usize;

    if x.shape.len() < 3 {
        return Err(OnnxError::ShapeMismatch(
            "GroupNorm requires at least 3D".into(),
        ));
    }

    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product();
    let channels_per_group = c / num_groups;
    let group_size = channels_per_group * spatial;
    let total = n * c * spatial;
    let mut result = vec![0.0f32; total];

    for batch in 0..n {
        for g in 0..num_groups {
            let group_base = (batch * c + g * channels_per_group) * spatial;
            // Compute mean/var over the group
            let mut sum = 0.0f32;
            for i in 0..group_size {
                sum += x_data[group_base + i];
            }
            let mean = sum / group_size as f32;

            let mut var_sum = 0.0f32;
            for i in 0..group_size {
                let d = x_data[group_base + i] - mean;
                var_sum += d * d;
            }
            let var = var_sum / group_size as f32;
            let inv_std = 1.0 / (var + epsilon).sqrt();

            for ci in 0..channels_per_group {
                let ch = g * channels_per_group + ci;
                let base = (batch * c + ch) * spatial;
                for si in 0..spatial {
                    let norm = (x_data[base + si] - mean) * inv_std;
                    result[base + si] = scale_data[ch] * norm + bias_data[ch];
                }
            }
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, x.shape.clone())])
}

/// FlashAttention(Q, K, V) -> scaled dot-product attention.
/// Custom operator for fused attention: softmax(Q * K^T / sqrt(d)) * V.
pub fn execute_flash_attention(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let q = get_required_input(inputs, 0, "Q")?;
    let k = get_required_input(inputs, 1, "K")?;
    let v = get_required_input(inputs, 2, "V")?;
    let q_data = q.as_f32()?;
    let k_data = k.as_f32()?;
    let v_data = v.as_f32()?;

    // Q, K, V shape: [batch, seq_len, d_model] or [batch, heads, seq_len, d_k]
    if q.shape.len() < 2 {
        return Err(OnnxError::ShapeMismatch(
            "FlashAttention requires at least 2D".into(),
        ));
    }

    let rank = q.shape.len();
    let d_k = q.shape[rank - 1];
    let seq_q = q.shape[rank - 2];
    let seq_k = k.shape[rank - 2];
    let scale = 1.0 / (d_k as f32).sqrt();

    let batch_dims: usize = q.shape[..rank - 2].iter().product::<usize>().max(1);
    let qk_stride_q = seq_q * d_k;
    let qk_stride_k = seq_k * d_k;
    let d_v = v.shape[rank - 1];
    let qk_stride_v = seq_k * d_v;

    let mut result = vec![0.0f32; batch_dims * seq_q * d_v];

    for b in 0..batch_dims {
        let q_base = b * qk_stride_q;
        let k_base = b * qk_stride_k;
        let v_base = b * qk_stride_v;
        let o_base = b * seq_q * d_v;

        for i in 0..seq_q {
            // Compute attention scores for row i
            let mut scores = vec![0.0f32; seq_k];
            for j in 0..seq_k {
                let mut dot = 0.0f32;
                for d in 0..d_k {
                    dot += q_data[q_base + i * d_k + d] * k_data[k_base + j * d_k + d];
                }
                scores[j] = dot * scale;
            }

            // Softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                exp_sum += *s;
            }
            if exp_sum > 0.0 {
                for s in &mut scores {
                    *s /= exp_sum;
                }
            }

            // Multiply by V
            for d in 0..d_v {
                let mut sum = 0.0f32;
                for j in 0..seq_k {
                    sum += scores[j] * v_data[v_base + j * d_v + d];
                }
                result[o_base + i * d_v + d] = sum;
            }
        }
    }

    let mut out_shape = q.shape[..rank - 2].to_vec();
    out_shape.push(seq_q);
    out_shape.push(d_v);

    Ok(vec![OnnxTensor::from_f32(&result, out_shape)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_f32_near(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < eps, "index {i}: {a} != {e} (eps={eps})");
        }
    }

    #[test]
    fn test_batch_norm() {
        // X=[1,2,1,1], scale=[1,1], bias=[0,0], mean=[0,0], var=[1,1]
        let x = OnnxTensor::from_f32(&[2.0, 4.0], vec![1, 2, 1, 1]);
        let scale = OnnxTensor::from_f32(&[1.0, 1.0], vec![2]);
        let bias = OnnxTensor::from_f32(&[0.0, 0.0], vec![2]);
        let mean = OnnxTensor::from_f32(&[0.0, 0.0], vec![2]);
        let var = OnnxTensor::from_f32(&[1.0, 1.0], vec![2]);
        let r = execute_batch_normalization(
            &[Some(&x), Some(&scale), Some(&bias), Some(&mean), Some(&var)],
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(r[0].shape, vec![1, 2, 1, 1]);
        assert_f32_near(&r[0].as_f32().unwrap(), &[2.0, 4.0], 1e-4);
    }

    #[test]
    fn test_batch_norm_with_stats() {
        let x = OnnxTensor::from_f32(&[10.0], vec![1, 1, 1, 1]);
        let scale = OnnxTensor::from_f32(&[2.0], vec![1]);
        let bias = OnnxTensor::from_f32(&[1.0], vec![1]);
        let mean = OnnxTensor::from_f32(&[10.0], vec![1]);
        let var = OnnxTensor::from_f32(&[4.0], vec![1]);
        let mut attrs = HashMap::new();
        attrs.insert("epsilon".into(), AttributeValue::Float(0.0));
        let r = execute_batch_normalization(
            &[Some(&x), Some(&scale), Some(&bias), Some(&mean), Some(&var)],
            &attrs,
        )
        .unwrap();
        // (10 - 10) / sqrt(4) * 2 + 1 = 0 * 1 + 1 = 1
        assert_f32_near(&r[0].as_f32().unwrap(), &[1.0], 1e-5);
    }

    #[test]
    fn test_layer_norm() {
        // Normalize over last dim of [1,4]
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let scale = OnnxTensor::from_f32(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let bias = OnnxTensor::from_f32(&[0.0, 0.0, 0.0, 0.0], vec![4]);
        let r =
            execute_layer_normalization(&[Some(&x), Some(&scale), Some(&bias)], &HashMap::new())
                .unwrap();
        // mean=2.5, var=1.25,  output ~= [-1.342, -0.447, 0.447, 1.342]
        let out = r[0].as_f32().unwrap();
        assert!((out[0] + out[3]).abs() < 1e-5); // symmetric
        assert!(out[0] < 0.0 && out[3] > 0.0);
    }

    #[test]
    fn test_instance_norm() {
        let x = OnnxTensor::from_f32(&[1.0, 3.0, 5.0, 7.0], vec![1, 2, 1, 2]);
        let scale = OnnxTensor::from_f32(&[1.0, 1.0], vec![2]);
        let bias = OnnxTensor::from_f32(&[0.0, 0.0], vec![2]);
        let r =
            execute_instance_normalization(&[Some(&x), Some(&scale), Some(&bias)], &HashMap::new())
                .unwrap();
        let out = r[0].as_f32().unwrap();
        // Each channel normalized independently over spatial dims
        assert_f32_near(&out[0..2], &[-1.0, 1.0], 1e-4);
        assert_f32_near(&out[2..4], &[-1.0, 1.0], 1e-4);
    }

    #[test]
    fn test_group_norm() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 1, 2]);
        let scale = OnnxTensor::from_f32(&[1.0, 1.0], vec![2]);
        let bias = OnnxTensor::from_f32(&[0.0, 0.0], vec![2]);
        let mut attrs = HashMap::new();
        attrs.insert("num_groups".into(), AttributeValue::Int(1)); // all channels in 1 group
        let r =
            execute_group_normalization(&[Some(&x), Some(&scale), Some(&bias)], &attrs).unwrap();
        let out = r[0].as_f32().unwrap();
        // All 4 values normalized together
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn test_flash_attention() {
        // Simple 2x2 attention
        let q = OnnxTensor::from_f32(&[1.0, 0.0, 0.0, 1.0], vec![1, 2, 2]);
        let k = OnnxTensor::from_f32(&[1.0, 0.0, 0.0, 1.0], vec![1, 2, 2]);
        let v = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let r = execute_flash_attention(&[Some(&q), Some(&k), Some(&v)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![1, 2, 2]);
        // Attention should weight V rows by softmax of Q*K^T/sqrt(2)
        let out = r[0].as_f32().unwrap();
        assert_eq!(out.len(), 4);
    }
}
