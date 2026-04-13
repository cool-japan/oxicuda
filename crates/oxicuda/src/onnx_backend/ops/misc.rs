//! Miscellaneous ONNX operators (Identity, Dropout, Constant, Shape, Size).

use std::collections::HashMap;

use super::get_required_input;
use crate::onnx_backend::ir::*;

/// Identity(X) -> pass-through.
pub fn execute_identity(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "input")?;
    Ok(vec![x.clone()])
}

/// Dropout(data) -> identity in inference mode.
pub fn execute_dropout(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "data")?;
    // In inference mode, dropout is an identity op.
    // Second output (mask) is all-true.
    let mask = OnnxTensor::from_bool(&vec![true; x.element_count()], x.shape.clone());
    Ok(vec![x.clone(), mask])
}

/// Constant() -> constant tensor from attribute.
pub fn execute_constant(
    _inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    if let Some(AttributeValue::Tensor(t)) = attrs.get("value") {
        return Ok(vec![t.clone()]);
    }
    if let Some(AttributeValue::Float(v)) = attrs.get("value_float") {
        return Ok(vec![OnnxTensor::scalar_f32(*v as f32)]);
    }
    if let Some(AttributeValue::Int(v)) = attrs.get("value_int") {
        return Ok(vec![OnnxTensor::scalar_i64(*v)]);
    }
    if let Some(AttributeValue::Floats(v)) = attrs.get("value_floats") {
        let f32s: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let len = f32s.len();
        return Ok(vec![OnnxTensor::from_f32(&f32s, vec![len])]);
    }
    if let Some(AttributeValue::Ints(v)) = attrs.get("value_ints") {
        let len = v.len();
        return Ok(vec![OnnxTensor::from_i64(v, vec![len])]);
    }
    Err(OnnxError::InvalidAttribute(
        "Constant: no value attribute found".into(),
    ))
}

/// Shape(data) -> int64 tensor of input shape.
pub fn execute_shape(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "data")?;
    let shape_vals: Vec<i64> = x.shape.iter().map(|&d| d as i64).collect();
    let len = shape_vals.len();
    Ok(vec![OnnxTensor::from_i64(&shape_vals, vec![len])])
}

/// Size(data) -> scalar int64 of total element count.
pub fn execute_size(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "data")?;
    Ok(vec![OnnxTensor::scalar_i64(x.element_count() as i64)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0], vec![3]);
        let r = execute_identity(&[Some(&x)], &HashMap::new()).unwrap();
        assert_eq!(r[0], x);
    }

    #[test]
    fn test_dropout_inference() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = execute_dropout(&[Some(&x)], &HashMap::new()).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0], x);
        // Mask should be all true
        assert_eq!(r[1].as_bool().unwrap(), vec![true; 4]);
    }

    #[test]
    fn test_constant_tensor() {
        let t = OnnxTensor::from_f32(&[1.0, 2.0], vec![2]);
        let mut attrs = HashMap::new();
        attrs.insert("value".into(), AttributeValue::Tensor(t.clone()));
        let r = execute_constant(&[], &attrs).unwrap();
        assert_eq!(r[0], t);
    }

    #[test]
    fn test_constant_float() {
        let mut attrs = HashMap::new();
        attrs.insert("value_float".into(), AttributeValue::Float(7.125));
        let r = execute_constant(&[], &attrs).unwrap();
        let v = r[0].as_f32().unwrap();
        assert!((v[0] - 7.125).abs() < 1e-3);
    }

    #[test]
    fn test_constant_int() {
        let mut attrs = HashMap::new();
        attrs.insert("value_int".into(), AttributeValue::Int(42));
        let r = execute_constant(&[], &attrs).unwrap();
        assert_eq!(r[0].as_i64().unwrap(), vec![42]);
    }

    #[test]
    fn test_shape() {
        let x = OnnxTensor::from_f32(&[0.0; 24], vec![2, 3, 4]);
        let r = execute_shape(&[Some(&x)], &HashMap::new()).unwrap();
        assert_eq!(r[0].as_i64().unwrap(), vec![2, 3, 4]);
        assert_eq!(r[0].shape, vec![3]);
    }

    #[test]
    fn test_size() {
        let x = OnnxTensor::from_f32(&[0.0; 24], vec![2, 3, 4]);
        let r = execute_size(&[Some(&x)], &HashMap::new()).unwrap();
        assert_eq!(r[0].as_i64().unwrap(), vec![24]);
        assert!(r[0].shape.is_empty());
    }
}
