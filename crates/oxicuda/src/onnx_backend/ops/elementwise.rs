//! Elementwise ONNX operators.

use std::collections::HashMap;

use super::{binary_elementwise_f32, get_float_attr, get_required_input, unary_elementwise_f32};
use crate::onnx_backend::ir::*;

// ─── Binary ops ─────────────────────────────────────────────

/// Add(A, B) -> element-wise addition with broadcasting.
pub fn execute_add(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    binary_elementwise_f32(inputs, |a, b| a + b)
}

/// Sub(A, B) -> element-wise subtraction with broadcasting.
pub fn execute_sub(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    binary_elementwise_f32(inputs, |a, b| a - b)
}

/// Mul(A, B) -> element-wise multiplication with broadcasting.
pub fn execute_mul(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    binary_elementwise_f32(inputs, |a, b| a * b)
}

/// Div(A, B) -> element-wise division with broadcasting.
pub fn execute_div(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    binary_elementwise_f32(inputs, |a, b| if b != 0.0 { a / b } else { f32::NAN })
}

/// Pow(A, B) -> element-wise power with broadcasting.
pub fn execute_pow(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    binary_elementwise_f32(inputs, |a, b| a.powf(b))
}

// ─── Unary ops ──────────────────────────────────────────────

/// Relu(X) -> max(0, x).
pub fn execute_relu(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, |x| x.max(0.0))
}

/// Sigmoid(X) -> 1 / (1 + exp(-x)).
pub fn execute_sigmoid(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, |x| 1.0 / (1.0 + (-x).exp()))
}

/// Tanh(X) -> hyperbolic tangent.
pub fn execute_tanh(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, f32::tanh)
}

/// Exp(X) -> e^x.
pub fn execute_exp(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, f32::exp)
}

/// Log(X) -> natural log.
pub fn execute_log(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, f32::ln)
}

/// Sqrt(X) -> square root.
pub fn execute_sqrt(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, f32::sqrt)
}

/// Abs(X) -> absolute value.
pub fn execute_abs(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, f32::abs)
}

/// Neg(X) -> negation.
pub fn execute_neg(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, |x| -x)
}

/// Ceil(X) -> ceiling.
pub fn execute_ceil(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, f32::ceil)
}

/// Floor(X) -> floor.
pub fn execute_floor(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, f32::floor)
}

/// Round(X) -> round to nearest even.
pub fn execute_round(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, f32::round)
}

/// Sign(X) -> signum.
pub fn execute_sign(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, |x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

/// Reciprocal(X) -> 1/x.
pub fn execute_reciprocal(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, |x| if x != 0.0 { 1.0 / x } else { f32::NAN })
}

/// Softplus(X) -> ln(1 + exp(x)).
pub fn execute_softplus(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    unary_elementwise_f32(inputs, |x| {
        if x > 20.0 {
            x // numerical stability
        } else {
            (1.0 + x.exp()).ln()
        }
    })
}

// ─── Parameterized unary ops ────────────────────────────────

/// LeakyRelu(X, alpha) -> max(alpha*x, x).
pub fn execute_leaky_relu(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let alpha = get_float_attr(attrs, "alpha", 0.01)? as f32;
    unary_elementwise_f32(inputs, move |x| if x >= 0.0 { x } else { alpha * x })
}

/// Elu(X, alpha) -> x if x>=0, alpha*(exp(x)-1) otherwise.
pub fn execute_elu(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let alpha = get_float_attr(attrs, "alpha", 1.0)? as f32;
    unary_elementwise_f32(
        inputs,
        move |x| {
            if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }
        },
    )
}

/// Selu(X, alpha, gamma) -> gamma * (x if x>=0, alpha*(exp(x)-1) otherwise).
pub fn execute_selu(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let alpha = get_float_attr(attrs, "alpha", 1.673_263_168_334_961)? as f32;
    let gamma = get_float_attr(attrs, "gamma", 1.050_700_783_920_288)? as f32;
    unary_elementwise_f32(inputs, move |x| {
        if x >= 0.0 {
            gamma * x
        } else {
            gamma * (alpha * (x.exp() - 1.0))
        }
    })
}

/// Clip(X, min, max) -> clamp to [min, max].
pub fn execute_clip(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "input")?;
    let data = x.as_f32()?;

    // Opset 11+: min/max as optional inputs
    let min_val = if let Some(min_t) = inputs.get(1).and_then(|o| *o) {
        let v = min_t.as_f32()?;
        v.first().copied().unwrap_or(f32::NEG_INFINITY)
    } else {
        get_float_attr(attrs, "min", f64::from(f32::NEG_INFINITY))? as f32
    };

    let max_val = if let Some(max_t) = inputs.get(2).and_then(|o| *o) {
        let v = max_t.as_f32()?;
        v.first().copied().unwrap_or(f32::INFINITY)
    } else {
        get_float_attr(attrs, "max", f64::from(f32::INFINITY))? as f32
    };

    let result: Vec<f32> = data.iter().map(|&v| v.clamp(min_val, max_val)).collect();
    Ok(vec![OnnxTensor::from_f32(&result, x.shape.clone())])
}

// ─── Ternary / special ops ──────────────────────────────────

/// Where(condition, X, Y) -> select X where true, Y where false.
pub fn execute_where(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let cond = get_required_input(inputs, 0, "condition")?;
    let x = get_required_input(inputs, 1, "X")?;
    let y = get_required_input(inputs, 2, "Y")?;
    let cond_data = cond.as_bool()?;
    let x_data = x.as_f32()?;
    let y_data = y.as_f32()?;

    // Three-way broadcast
    let shape_xy = broadcast_shapes(&x.shape, &y.shape)?;
    let out_shape = broadcast_shapes(&cond.shape, &shape_xy)?;
    let total: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let mut result = Vec::with_capacity(total);
    for i in 0..total {
        let multi = flat_to_multi(i, &out_shape);
        let ci = broadcast_index(&multi, &cond.shape, &out_shape);
        let xi = broadcast_index(&multi, &x.shape, &out_shape);
        let yi = broadcast_index(&multi, &y.shape, &out_shape);
        result.push(if cond_data[ci] {
            x_data[xi]
        } else {
            y_data[yi]
        });
    }
    Ok(vec![OnnxTensor::from_f32(&result, out_shape)])
}

/// Cast(X, to=dtype) -> type conversion.
pub fn execute_cast(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "input")?;
    let to = attrs
        .get("to")
        .ok_or_else(|| OnnxError::InvalidAttribute("Cast requires 'to' attribute".into()))?
        .as_int()?;

    let target_dtype = match to {
        1 => DataType::Float32,
        2 => DataType::Uint8,
        3 => DataType::Int8,
        6 => DataType::Int32,
        7 => DataType::Int64,
        9 => DataType::Bool,
        10 => DataType::Float16,
        11 => DataType::Float64,
        other => {
            return Err(OnnxError::TypeError(format!(
                "unsupported Cast target type: {other}"
            )));
        }
    };

    cast_tensor(x, target_dtype)
}

/// FMA(A, B, C) -> A * B + C (fused multiply-add).
pub fn execute_fma(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let a = get_required_input(inputs, 0, "A")?;
    let b = get_required_input(inputs, 1, "B")?;
    let c = get_required_input(inputs, 2, "C")?;

    let a_data = a.as_f32()?;
    let b_data = b.as_f32()?;
    let c_data = c.as_f32()?;

    let ab_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let out_shape = broadcast_shapes(&ab_shape, &c.shape)?;
    let total: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let mut result = Vec::with_capacity(total);
    for i in 0..total {
        let multi = flat_to_multi(i, &out_shape);
        let ai = broadcast_index(&multi, &a.shape, &out_shape);
        let bi = broadcast_index(&multi, &b.shape, &out_shape);
        let ci = broadcast_index(&multi, &c.shape, &out_shape);
        result.push(a_data[ai].mul_add(b_data[bi], c_data[ci]));
    }
    Ok(vec![OnnxTensor::from_f32(&result, out_shape)])
}

// ─── Cast helpers ───────────────────────────────────────────

// ONNX uses specific integer codes for data types; we handle a subset
// of common int16 as Int32 storage internally (not full Float16 compute).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum Int16Placeholder {}

fn cast_tensor(x: &OnnxTensor, target: DataType) -> OnnxResult<Vec<OnnxTensor>> {
    // Fast path: same type
    if x.dtype == target {
        return Ok(vec![x.clone()]);
    }

    // Get source values as f64 intermediary
    let values = tensor_to_f64(x)?;

    // Convert to target
    let out = match target {
        DataType::Float32 => {
            let v: Vec<f32> = values.iter().map(|&x| x as f32).collect();
            OnnxTensor::from_f32(&v, x.shape.clone())
        }
        DataType::Float64 => OnnxTensor::from_f64(&values, x.shape.clone()),
        DataType::Int32 => {
            let v: Vec<i32> = values.iter().map(|&x| x as i32).collect();
            OnnxTensor::from_i32(&v, x.shape.clone())
        }
        DataType::Int64 => {
            let v: Vec<i64> = values.iter().map(|&x| x as i64).collect();
            OnnxTensor::from_i64(&v, x.shape.clone())
        }
        DataType::Bool => {
            let v: Vec<bool> = values.iter().map(|&x| x != 0.0).collect();
            OnnxTensor::from_bool(&v, x.shape.clone())
        }
        other => {
            return Err(OnnxError::TypeError(format!(
                "Cast to {other} not yet implemented"
            )));
        }
    };
    Ok(vec![out])
}

fn tensor_to_f64(x: &OnnxTensor) -> OnnxResult<Vec<f64>> {
    match x.dtype {
        DataType::Float32 => Ok(x.as_f32()?.iter().map(|&v| f64::from(v)).collect()),
        DataType::Float64 => x.as_f64(),
        DataType::Int32 => Ok(x.as_i32()?.iter().map(|&v| f64::from(v)).collect()),
        DataType::Int64 => Ok(x.as_i64()?.iter().map(|&v| v as f64).collect()),
        DataType::Bool => Ok(x
            .as_bool()?
            .iter()
            .map(|&v| if v { 1.0 } else { 0.0 })
            .collect()),
        other => Err(OnnxError::TypeError(format!(
            "Cast from {other} not implemented"
        ))),
    }
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
    fn test_add() {
        let a = OnnxTensor::from_f32(&[1.0, 2.0, 3.0], vec![3]);
        let b = OnnxTensor::from_f32(&[10.0, 20.0, 30.0], vec![3]);
        let r = execute_add(&[Some(&a), Some(&b)], &empty_attrs()).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_add_broadcast() {
        let a = OnnxTensor::from_f32(&[1.0, 2.0, 3.0], vec![3]);
        let b = OnnxTensor::from_f32(&[10.0], vec![1]);
        let r = execute_add(&[Some(&a), Some(&b)], &empty_attrs()).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_sub() {
        let a = OnnxTensor::from_f32(&[10.0, 20.0], vec![2]);
        let b = OnnxTensor::from_f32(&[1.0, 2.0], vec![2]);
        let r = execute_sub(&[Some(&a), Some(&b)], &empty_attrs()).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![9.0, 18.0]);
    }

    #[test]
    fn test_mul() {
        let a = OnnxTensor::from_f32(&[2.0, 3.0], vec![2]);
        let b = OnnxTensor::from_f32(&[4.0, 5.0], vec![2]);
        let r = execute_mul(&[Some(&a), Some(&b)], &empty_attrs()).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_div() {
        let a = OnnxTensor::from_f32(&[10.0, 20.0], vec![2]);
        let b = OnnxTensor::from_f32(&[2.0, 5.0], vec![2]);
        let r = execute_div(&[Some(&a), Some(&b)], &empty_attrs()).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![5.0, 4.0]);
    }

    #[test]
    fn test_relu() {
        let x = OnnxTensor::from_f32(&[-1.0, 0.0, 1.0, 2.0], vec![4]);
        let r = execute_relu(&[Some(&x)], &empty_attrs()).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let x = OnnxTensor::from_f32(&[0.0], vec![1]);
        let r = execute_sigmoid(&[Some(&x)], &empty_attrs()).unwrap();
        assert_f32_near(&r[0].as_f32().unwrap(), &[0.5], 1e-6);
    }

    #[test]
    fn test_tanh() {
        let x = OnnxTensor::from_f32(&[0.0], vec![1]);
        let r = execute_tanh(&[Some(&x)], &empty_attrs()).unwrap();
        assert_f32_near(&r[0].as_f32().unwrap(), &[0.0], 1e-6);
    }

    #[test]
    fn test_exp_log() {
        let x = OnnxTensor::from_f32(&[1.0], vec![1]);
        let r = execute_exp(&[Some(&x)], &empty_attrs()).unwrap();
        assert_f32_near(&r[0].as_f32().unwrap(), &[std::f32::consts::E], 1e-5);
        let r2 = execute_log(&[Some(&r[0])], &empty_attrs()).unwrap();
        assert_f32_near(&r2[0].as_f32().unwrap(), &[1.0], 1e-5);
    }

    #[test]
    fn test_abs_neg() {
        let x = OnnxTensor::from_f32(&[-3.0, 2.0], vec![2]);
        let r_abs = execute_abs(&[Some(&x)], &empty_attrs()).unwrap();
        assert_eq!(r_abs[0].as_f32().unwrap(), vec![3.0, 2.0]);
        let r_neg = execute_neg(&[Some(&x)], &empty_attrs()).unwrap();
        assert_eq!(r_neg[0].as_f32().unwrap(), vec![3.0, -2.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let x = OnnxTensor::from_f32(&[-10.0, 5.0], vec![2]);
        let mut attrs = HashMap::new();
        attrs.insert("alpha".into(), AttributeValue::Float(0.1));
        let r = execute_leaky_relu(&[Some(&x)], &attrs).unwrap();
        assert_f32_near(&r[0].as_f32().unwrap(), &[-1.0, 5.0], 1e-6);
    }

    #[test]
    fn test_clip() {
        let x = OnnxTensor::from_f32(&[-5.0, 0.0, 5.0, 10.0], vec![4]);
        let min_t = OnnxTensor::from_f32(&[0.0], vec![]);
        let max_t = OnnxTensor::from_f32(&[6.0], vec![]);
        let r = execute_clip(&[Some(&x), Some(&min_t), Some(&max_t)], &empty_attrs()).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![0.0, 0.0, 5.0, 6.0]);
    }

    #[test]
    fn test_where_op() {
        let cond = OnnxTensor::from_bool(&[true, false, true], vec![3]);
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0], vec![3]);
        let y = OnnxTensor::from_f32(&[10.0, 20.0, 30.0], vec![3]);
        let r = execute_where(&[Some(&cond), Some(&x), Some(&y)], &empty_attrs()).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![1.0, 20.0, 3.0]);
    }

    #[test]
    fn test_cast_f32_to_i32() {
        let x = OnnxTensor::from_f32(&[1.5, 2.7, -3.1], vec![3]);
        let mut attrs = HashMap::new();
        attrs.insert("to".into(), AttributeValue::Int(6)); // INT32
        let r = execute_cast(&[Some(&x)], &attrs).unwrap();
        assert_eq!(r[0].dtype, DataType::Int32);
        assert_eq!(r[0].as_i32().unwrap(), vec![1, 2, -3]);
    }

    #[test]
    fn test_fma() {
        let a = OnnxTensor::from_f32(&[2.0, 3.0], vec![2]);
        let b = OnnxTensor::from_f32(&[4.0, 5.0], vec![2]);
        let c = OnnxTensor::from_f32(&[1.0, 1.0], vec![2]);
        let r = execute_fma(&[Some(&a), Some(&b), Some(&c)], &empty_attrs()).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![9.0, 16.0]);
    }

    #[test]
    fn test_pow() {
        let a = OnnxTensor::from_f32(&[2.0, 3.0], vec![2]);
        let b = OnnxTensor::from_f32(&[3.0, 2.0], vec![2]);
        let r = execute_pow(&[Some(&a), Some(&b)], &empty_attrs()).unwrap();
        assert_f32_near(&r[0].as_f32().unwrap(), &[8.0, 9.0], 1e-5);
    }

    #[test]
    fn test_ceil_floor_round() {
        let x = OnnxTensor::from_f32(&[1.3, 2.7, -0.5], vec![3]);
        let rc = execute_ceil(&[Some(&x)], &empty_attrs()).unwrap();
        assert_eq!(rc[0].as_f32().unwrap(), vec![2.0, 3.0, 0.0]);
        let rf = execute_floor(&[Some(&x)], &empty_attrs()).unwrap();
        assert_eq!(rf[0].as_f32().unwrap(), vec![1.0, 2.0, -1.0]);
        let rr = execute_round(&[Some(&x)], &empty_attrs()).unwrap();
        assert_eq!(rr[0].as_f32().unwrap(), vec![1.0, 3.0, -1.0]);
    }

    #[test]
    fn test_sign() {
        let x = OnnxTensor::from_f32(&[-3.0, 0.0, 5.0], vec![3]);
        let r = execute_sign(&[Some(&x)], &empty_attrs()).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![-1.0, 0.0, 1.0]);
    }
}
