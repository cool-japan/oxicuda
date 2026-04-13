//! Reduction ONNX operators (ReduceSum, ReduceMean, etc.).

use std::collections::HashMap;

use super::{get_int_attr, get_ints_attr, get_required_input};
use crate::onnx_backend::ir::*;

// ─── Generic reduction helper ───────────────────────────────

fn execute_reduce(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
    init: f32,
    fold: fn(f32, f32) -> f32,
    finalize: fn(f32, usize) -> f32,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "data")?;
    let data = x.as_f32()?;
    let rank = x.shape.len();
    let keepdims = get_int_attr(attrs, "keepdims", 1)? != 0;

    // Determine axes to reduce
    let axes: Vec<usize> = if let Some(axes_attr) = get_ints_attr(attrs, "axes")? {
        axes_attr
            .iter()
            .map(|&a| normalize_axis(a, rank))
            .collect::<OnnxResult<Vec<_>>>()?
    } else if let Some(axes_input) = inputs.get(1).and_then(|o| *o) {
        // Opset 18+: axes as second input
        let a = axes_input.as_i64()?;
        a.iter()
            .map(|&v| normalize_axis(v, rank))
            .collect::<OnnxResult<Vec<_>>>()?
    } else {
        // Reduce all axes
        (0..rank).collect()
    };

    if rank == 0 {
        // Scalar input: pass through
        return Ok(vec![x.clone()]);
    }

    // Compute output shape
    let mut out_shape: Vec<usize> = Vec::new();
    for (i, &dim) in x.shape.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(dim);
        }
    }

    let out_total: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let mut result = vec![init; out_total];
    let mut counts = vec![0usize; out_total];

    for (i, &val) in data.iter().enumerate() {
        let multi = flat_to_multi(i, &x.shape);
        // Map to output index
        let mut out_idx = Vec::new();
        for (j, &idx) in multi.iter().enumerate() {
            if axes.contains(&j) {
                if keepdims {
                    out_idx.push(0);
                }
            } else {
                out_idx.push(idx);
            }
        }
        let oi = if out_idx.is_empty() {
            0
        } else {
            multi_to_flat(&out_idx, &out_shape)
        };
        result[oi] = fold(result[oi], val);
        counts[oi] += 1;
    }

    // Apply finalization (e.g., divide by count for mean)
    for (r, &c) in result.iter_mut().zip(counts.iter()) {
        *r = finalize(*r, c);
    }

    Ok(vec![OnnxTensor::from_f32(&result, out_shape)])
}

fn normalize_axis(axis: i64, rank: usize) -> OnnxResult<usize> {
    let r = rank as i64;
    if axis < -r || axis >= r {
        return Err(OnnxError::InvalidAttribute(format!(
            "axis {axis} out of range for rank {rank}"
        )));
    }
    Ok(if axis < 0 {
        (r + axis) as usize
    } else {
        axis as usize
    })
}

// ─── Operators ──────────────────────────────────────────────

/// ReduceSum over specified axes.
pub fn execute_reduce_sum(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    execute_reduce(inputs, attrs, 0.0, |acc, v| acc + v, |v, _| v)
}

/// ReduceMean over specified axes.
pub fn execute_reduce_mean(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    execute_reduce(
        inputs,
        attrs,
        0.0,
        |acc, v| acc + v,
        |v, c| {
            if c > 0 { v / c as f32 } else { 0.0 }
        },
    )
}

/// ReduceMax over specified axes.
pub fn execute_reduce_max(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    execute_reduce(inputs, attrs, f32::NEG_INFINITY, f32::max, |v, _| v)
}

/// ReduceMin over specified axes.
pub fn execute_reduce_min(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    execute_reduce(inputs, attrs, f32::INFINITY, f32::min, |v, _| v)
}

/// ReduceProd over specified axes.
pub fn execute_reduce_prod(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    execute_reduce(inputs, attrs, 1.0, |acc, v| acc * v, |v, _| v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs_with_axes(axes: &[i64], keepdims: bool) -> HashMap<String, AttributeValue> {
        let mut m = HashMap::new();
        m.insert("axes".into(), AttributeValue::Ints(axes.to_vec()));
        m.insert("keepdims".into(), AttributeValue::Int(i64::from(keepdims)));
        m
    }

    #[test]
    fn test_reduce_sum_axis0() {
        // [[1,2,3],[4,5,6]] -> reduce axis 0 -> [5,7,9]
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let attrs = attrs_with_axes(&[0], false);
        let r = execute_reduce_sum(&[Some(&x)], &attrs).unwrap();
        assert_eq!(r[0].shape, vec![3]);
        assert_eq!(r[0].as_f32().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_reduce_sum_axis1_keepdims() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let attrs = attrs_with_axes(&[1], true);
        let r = execute_reduce_sum(&[Some(&x)], &attrs).unwrap();
        assert_eq!(r[0].shape, vec![2, 1]);
        assert_eq!(r[0].as_f32().unwrap(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_reduce_mean() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let attrs = attrs_with_axes(&[1], false);
        let r = execute_reduce_mean(&[Some(&x)], &attrs).unwrap();
        assert_eq!(r[0].shape, vec![2]);
        assert_eq!(r[0].as_f32().unwrap(), vec![1.5, 3.5]);
    }

    #[test]
    fn test_reduce_max() {
        let x = OnnxTensor::from_f32(&[1.0, 5.0, 3.0, 2.0, 4.0, 0.0], vec![2, 3]);
        let attrs = attrs_with_axes(&[1], false);
        let r = execute_reduce_max(&[Some(&x)], &attrs).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![5.0, 4.0]);
    }

    #[test]
    fn test_reduce_min() {
        let x = OnnxTensor::from_f32(&[1.0, 5.0, 3.0, 2.0, 4.0, 0.0], vec![2, 3]);
        let attrs = attrs_with_axes(&[1], false);
        let r = execute_reduce_min(&[Some(&x)], &attrs).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![1.0, 0.0]);
    }

    #[test]
    fn test_reduce_prod() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let attrs = attrs_with_axes(&[1], false);
        let r = execute_reduce_prod(&[Some(&x)], &attrs).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![2.0, 12.0]);
    }

    #[test]
    fn test_reduce_sum_all_axes() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let attrs = attrs_with_axes(&[0, 1], false);
        let r = execute_reduce_sum(&[Some(&x)], &attrs).unwrap();
        assert!(r[0].shape.is_empty());
        assert_eq!(r[0].as_f32().unwrap(), vec![10.0]);
    }

    #[test]
    fn test_reduce_negative_axis() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let attrs = attrs_with_axes(&[-1], false);
        let r = execute_reduce_sum(&[Some(&x)], &attrs).unwrap();
        assert_eq!(r[0].shape, vec![2]);
        assert_eq!(r[0].as_f32().unwrap(), vec![6.0, 15.0]);
    }
}
