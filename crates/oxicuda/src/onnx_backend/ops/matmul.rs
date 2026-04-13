//! Matrix ONNX operators (MatMul, Gemm).

use std::collections::HashMap;

use super::{get_float_attr, get_int_attr, get_optional_input, get_required_input};
use crate::onnx_backend::ir::*;

/// MatMul(A, B) -> matrix multiplication with batch broadcasting.
pub fn execute_matmul(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let a = get_required_input(inputs, 0, "A")?;
    let b = get_required_input(inputs, 1, "B")?;
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;

    if a.shape.len() < 2 || b.shape.len() < 2 {
        return execute_matmul_low_rank(a, b, &a_data, &b_data);
    }

    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    let m = a.shape[a_rank - 2];
    let k = a.shape[a_rank - 1];
    let n = b.shape[b_rank - 1];

    if b.shape[b_rank - 2] != k {
        return Err(OnnxError::ShapeMismatch(format!(
            "MatMul inner dimensions mismatch: {} vs {}",
            k,
            b.shape[b_rank - 2]
        )));
    }

    // Broadcast batch dimensions
    let a_batch = &a.shape[..a_rank - 2];
    let b_batch = &b.shape[..b_rank - 2];
    let out_batch = broadcast_shapes(a_batch, b_batch)?;

    let mut out_shape = out_batch.clone();
    out_shape.push(m);
    out_shape.push(n);

    let batch_total: usize = if out_batch.is_empty() {
        1
    } else {
        out_batch.iter().product()
    };
    let a_mat_stride = m * k;
    let b_mat_stride = k * n;
    let out_mat_stride = m * n;

    let mut result = vec![0.0f32; batch_total * out_mat_stride];

    for bi in 0..batch_total {
        let multi = flat_to_multi(bi, &out_batch);
        let a_bi = broadcast_index(&multi, a_batch, &out_batch) * a_mat_stride;
        let b_bi = broadcast_index(&multi, b_batch, &out_batch) * b_mat_stride;
        let out_bi = bi * out_mat_stride;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[a_bi + i * k + p] * b_data[b_bi + p * n + j];
                }
                result[out_bi + i * n + j] = sum;
            }
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, out_shape)])
}

/// Handle 1-D vector cases for MatMul.
fn execute_matmul_low_rank(
    a: &OnnxTensor,
    b: &OnnxTensor,
    a_data: &[f32],
    b_data: &[f32],
) -> OnnxResult<Vec<OnnxTensor>> {
    // 1D x 1D -> scalar (dot product)
    if a.shape.len() == 1 && b.shape.len() == 1 {
        if a.shape[0] != b.shape[0] {
            return Err(OnnxError::ShapeMismatch(format!(
                "MatMul 1D x 1D: {} vs {}",
                a.shape[0], b.shape[0]
            )));
        }
        let dot: f32 = a_data.iter().zip(b_data).map(|(x, y)| x * y).sum();
        return Ok(vec![OnnxTensor::from_f32(&[dot], vec![])]);
    }

    // 1D x 2D -> [N]  (treat 1D as row vector)
    if a.shape.len() == 1 && b.shape.len() >= 2 {
        let k = a.shape[0];
        let b_rank = b.shape.len();
        if b.shape[b_rank - 2] != k {
            return Err(OnnxError::ShapeMismatch("MatMul 1D x 2D mismatch".into()));
        }
        let n = b.shape[b_rank - 1];
        let mut out_shape = b.shape[..b_rank - 2].to_vec();
        out_shape.push(n);

        let batch_total: usize = b.shape[..b_rank - 2].iter().product::<usize>().max(1);
        let stride = k * n;
        let mut result = Vec::with_capacity(batch_total * n);
        for bi in 0..batch_total {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[p] * b_data[bi * stride + p * n + j];
                }
                result.push(sum);
            }
        }
        return Ok(vec![OnnxTensor::from_f32(&result, out_shape)]);
    }

    // 2D x 1D -> [M]  (treat 1D as column vector)
    if a.shape.len() >= 2 && b.shape.len() == 1 {
        let a_rank = a.shape.len();
        let k = a.shape[a_rank - 1];
        if b.shape[0] != k {
            return Err(OnnxError::ShapeMismatch("MatMul 2D x 1D mismatch".into()));
        }
        let m = a.shape[a_rank - 2];
        let mut out_shape = a.shape[..a_rank - 2].to_vec();
        out_shape.push(m);

        let batch_total = a.shape[..a_rank - 2].iter().product::<usize>().max(1);
        let stride = m * k;
        let mut result = Vec::with_capacity(batch_total * m);
        for bi in 0..batch_total {
            for i in 0..m {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[bi * stride + i * k + p] * b_data[p];
                }
                result.push(sum);
            }
        }
        return Ok(vec![OnnxTensor::from_f32(&result, out_shape)]);
    }

    Err(OnnxError::ShapeMismatch(format!(
        "MatMul unsupported shapes: {:?} x {:?}",
        a.shape, b.shape
    )))
}

/// Gemm(A, B, C) -> alpha * op(A) * op(B) + beta * C.
#[allow(clippy::too_many_lines)]
pub fn execute_gemm(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let a = get_required_input(inputs, 0, "A")?;
    let b = get_required_input(inputs, 1, "B")?;
    let c = get_optional_input(inputs, 2);
    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;

    let alpha = get_float_attr(attrs, "alpha", 1.0)? as f32;
    let beta = get_float_attr(attrs, "beta", 1.0)? as f32;
    let trans_a = get_int_attr(attrs, "transA", 0)? != 0;
    let trans_b = get_int_attr(attrs, "transB", 0)? != 0;

    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(OnnxError::ShapeMismatch("Gemm requires 2D inputs".into()));
    }

    let (m, k_a) = if trans_a {
        (a.shape[1], a.shape[0])
    } else {
        (a.shape[0], a.shape[1])
    };
    let (k_b, n) = if trans_b {
        (b.shape[1], b.shape[0])
    } else {
        (b.shape[0], b.shape[1])
    };

    if k_a != k_b {
        return Err(OnnxError::ShapeMismatch(format!(
            "Gemm inner dimension mismatch: {k_a} vs {k_b}"
        )));
    }
    let k = k_a;

    let mut result = vec![0.0f32; m * n];

    // Compute alpha * A * B
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_val = if trans_a {
                    a_data[p * a.shape[1] + i]
                } else {
                    a_data[i * a.shape[1] + p]
                };
                let b_val = if trans_b {
                    b_data[j * b.shape[1] + p]
                } else {
                    b_data[p * b.shape[1] + j]
                };
                sum += a_val * b_val;
            }
            result[i * n + j] = alpha * sum;
        }
    }

    // Add beta * C (with broadcasting)
    if let Some(c_tensor) = c {
        let c_data = c_tensor.as_f32()?;
        let out_shape = vec![m, n];
        for i in 0..m {
            for j in 0..n {
                let multi = vec![i, j];
                let ci = broadcast_index(&multi, &c_tensor.shape, &out_shape);
                result[i * n + j] += beta * c_data[ci];
            }
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, vec![m, n])])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_attrs() -> HashMap<String, AttributeValue> {
        HashMap::new()
    }

    fn assert_f32_near(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < eps, "index {i}: {a} != {e} (eps={eps})");
        }
    }

    #[test]
    fn test_matmul_2d() {
        // [2,3] x [3,2] -> [2,2]
        let a = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = OnnxTensor::from_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let r = execute_matmul(&[Some(&a), Some(&b)], &empty_attrs()).unwrap();
        assert_eq!(r[0].shape, vec![2, 2]);
        // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
        assert_f32_near(&r[0].as_f32().unwrap(), &[58.0, 64.0, 139.0, 154.0], 1e-5);
    }

    #[test]
    fn test_matmul_batch() {
        // [2,2,3] x [2,3,2] -> [2,2,2]
        let a = OnnxTensor::from_f32(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![2, 2, 3],
        );
        let b = OnnxTensor::from_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 3, 2],
        );
        let r = execute_matmul(&[Some(&a), Some(&b)], &empty_attrs()).unwrap();
        assert_eq!(r[0].shape, vec![2, 2, 2]);
    }

    #[test]
    fn test_matmul_1d_dot() {
        let a = OnnxTensor::from_f32(&[1.0, 2.0, 3.0], vec![3]);
        let b = OnnxTensor::from_f32(&[4.0, 5.0, 6.0], vec![3]);
        let r = execute_matmul(&[Some(&a), Some(&b)], &empty_attrs()).unwrap();
        assert!(r[0].shape.is_empty());
        assert_f32_near(&r[0].as_f32().unwrap(), &[32.0], 1e-5);
    }

    #[test]
    fn test_gemm_basic() {
        // Y = alpha * A * B + beta * C
        let a = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = OnnxTensor::from_f32(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = OnnxTensor::from_f32(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let mut attrs = HashMap::new();
        attrs.insert("alpha".into(), AttributeValue::Float(1.0));
        attrs.insert("beta".into(), AttributeValue::Float(1.0));
        let r = execute_gemm(&[Some(&a), Some(&b), Some(&c)], &attrs).unwrap();
        assert_eq!(r[0].shape, vec![2, 2]);
        // [1*5+2*7+1, 1*6+2*8+1, 3*5+4*7+1, 3*6+4*8+1] = [20, 23, 44, 51]
        assert_f32_near(&r[0].as_f32().unwrap(), &[20.0, 23.0, 44.0, 51.0], 1e-5);
    }

    #[test]
    fn test_gemm_transb() {
        let a = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = OnnxTensor::from_f32(&[5.0, 7.0, 6.0, 8.0], vec![2, 2]); // transposed
        let mut attrs = HashMap::new();
        attrs.insert("transB".into(), AttributeValue::Int(1));
        let r = execute_gemm(&[Some(&a), Some(&b)], &attrs).unwrap();
        // A * B^T: same result as A * [[5,6],[7,8]]
        assert_f32_near(&r[0].as_f32().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-5);
    }

    #[test]
    fn test_gemm_bias_broadcast() {
        let a = OnnxTensor::from_f32(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let b = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = OnnxTensor::from_f32(&[10.0, 20.0], vec![1, 2]); // broadcast
        let mut attrs = HashMap::new();
        attrs.insert("beta".into(), AttributeValue::Float(1.0));
        let r = execute_gemm(&[Some(&a), Some(&b), Some(&c)], &attrs).unwrap();
        assert_f32_near(&r[0].as_f32().unwrap(), &[11.0, 22.0, 13.0, 24.0], 1e-5);
    }
}
