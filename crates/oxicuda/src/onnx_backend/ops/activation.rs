//! Activation ONNX operators (Softmax, LogSoftmax).

use std::collections::HashMap;

use super::{get_int_attr, get_required_input};
use crate::onnx_backend::ir::*;

/// Softmax(X, axis) -> softmax along axis.
pub fn execute_softmax(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "input")?;
    let data = x.as_f32()?;
    let rank = x.shape.len() as i64;
    let axis = get_int_attr(attrs, "axis", -1)?;
    let a = if axis < 0 { rank + axis } else { axis } as usize;

    let outer: usize = x.shape[..a].iter().product::<usize>().max(1);
    let dim = x.shape[a];
    let inner: usize = x.shape[a + 1..].iter().product::<usize>().max(1);

    let mut result = data.clone();

    for o in 0..outer {
        for i in 0..inner {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }

            // Compute exp and sum
            let mut sum = 0.0f32;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                let e = (data[idx] - max_val).exp();
                result[idx] = e;
                sum += e;
            }

            // Normalize
            if sum > 0.0 {
                for d in 0..dim {
                    let idx = (o * dim + d) * inner + i;
                    result[idx] /= sum;
                }
            }
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, x.shape.clone())])
}

/// LogSoftmax(X, axis) -> log(softmax(x)) along axis.
pub fn execute_log_softmax(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "input")?;
    let data = x.as_f32()?;
    let rank = x.shape.len() as i64;
    let axis = get_int_attr(attrs, "axis", -1)?;
    let a = if axis < 0 { rank + axis } else { axis } as usize;

    let outer: usize = x.shape[..a].iter().product::<usize>().max(1);
    let dim = x.shape[a];
    let inner: usize = x.shape[a + 1..].iter().product::<usize>().max(1);

    let mut result = data.clone();

    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }

            let mut log_sum_exp = 0.0f32;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                log_sum_exp += (data[idx] - max_val).exp();
            }
            log_sum_exp = log_sum_exp.ln() + max_val;

            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                result[idx] = data[idx] - log_sum_exp;
            }
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, x.shape.clone())])
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
    fn test_softmax() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0], vec![1, 3]);
        let r = execute_softmax(&[Some(&x)], &HashMap::new()).unwrap();
        let out = r[0].as_f32().unwrap();
        // Sum should be 1
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Values should be monotonically increasing
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
    }

    #[test]
    fn test_softmax_batch() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".into(), AttributeValue::Int(1));
        let r = execute_softmax(&[Some(&x)], &attrs).unwrap();
        let out = r[0].as_f32().unwrap();
        // Each row should sum to 1
        let sum0: f32 = out[0..3].iter().sum();
        let sum1: f32 = out[3..6].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5);
        assert!((sum1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0], vec![1, 3]);
        let r = execute_log_softmax(&[Some(&x)], &HashMap::new()).unwrap();
        let out = r[0].as_f32().unwrap();
        // All values should be <= 0
        assert!(out.iter().all(|&v| v <= 0.0));
        // exp(logsoftmax) should sum to 1
        let sum: f32 = out.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_numerically_stable() {
        // Large values shouldn't cause overflow
        let x = OnnxTensor::from_f32(&[1000.0, 1001.0, 1002.0], vec![1, 3]);
        let r = execute_softmax(&[Some(&x)], &HashMap::new()).unwrap();
        let out = r[0].as_f32().unwrap();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_softmax_equals_manual() {
        let x = OnnxTensor::from_f32(&[0.0, 0.0, 0.0], vec![1, 3]);
        let r = execute_softmax(&[Some(&x)], &HashMap::new()).unwrap();
        let out = r[0].as_f32().unwrap();
        // Uniform distribution
        assert_f32_near(&out, &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1e-5);
    }
}
