//! Shape manipulation ONNX operators.

use std::collections::HashMap;

use super::{get_int_attr, get_ints_attr, get_optional_input, get_required_input};
use crate::onnx_backend::ir::*;

/// Reshape(data, shape) -> reshaped tensor.
pub fn execute_reshape(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let data = get_required_input(inputs, 0, "data")?;
    let shape_input = get_required_input(inputs, 1, "shape")?;
    let shape_vals = shape_input.as_i64()?;

    let elem_count = data.element_count();
    let mut new_shape: Vec<usize> = Vec::with_capacity(shape_vals.len());
    let mut neg_idx = None;

    for (i, &s) in shape_vals.iter().enumerate() {
        if s == -1 {
            if neg_idx.is_some() {
                return Err(OnnxError::ShapeMismatch(
                    "Reshape: at most one -1 allowed".into(),
                ));
            }
            neg_idx = Some(i);
            new_shape.push(0); // placeholder
        } else if s == 0 {
            // Copy from input shape
            let dim =
                data.shape.get(i).copied().ok_or_else(|| {
                    OnnxError::ShapeMismatch("Reshape: 0 index out of range".into())
                })?;
            new_shape.push(dim);
        } else {
            new_shape.push(s as usize);
        }
    }

    if let Some(idx) = neg_idx {
        let known_product: usize = new_shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != idx)
            .map(|(_, &v)| v)
            .product();
        if known_product == 0 {
            return Err(OnnxError::ShapeMismatch(
                "Reshape: zero in known dimensions".into(),
            ));
        }
        new_shape[idx] = elem_count / known_product;
    }

    Ok(vec![OnnxTensor {
        data: data.data.clone(),
        dtype: data.dtype,
        shape: new_shape,
    }])
}

/// Transpose(data, perm?) -> permuted dimensions.
pub fn execute_transpose(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let data = get_required_input(inputs, 0, "data")?;
    let src = data.as_f32()?;
    let rank = data.shape.len();

    let perm: Vec<usize> = if let Some(p) = get_ints_attr(attrs, "perm")? {
        p.iter().map(|&v| v as usize).collect()
    } else {
        // Default: reverse dimensions
        (0..rank).rev().collect()
    };

    let mut new_shape = vec![0usize; rank];
    for (i, &p) in perm.iter().enumerate() {
        new_shape[i] = data.shape[p];
    }

    let total: usize = data.shape.iter().product::<usize>().max(1);
    let mut result = vec![0.0f32; total];

    for (i, &val) in src.iter().enumerate() {
        let src_multi = flat_to_multi(i, &data.shape);
        let mut dst_multi = vec![0usize; rank];
        for (j, &p) in perm.iter().enumerate() {
            dst_multi[j] = src_multi[p];
        }
        let dst_idx = multi_to_flat(&dst_multi, &new_shape);
        result[dst_idx] = val;
    }

    Ok(vec![OnnxTensor::from_f32(&result, new_shape)])
}

/// Squeeze(data, axes?) -> remove size-1 dimensions.
pub fn execute_squeeze(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let data = get_required_input(inputs, 0, "data")?;

    let axes: Vec<usize> = if let Some(axes_input) = get_optional_input(inputs, 1) {
        // Opset 13+: axes as input
        axes_input
            .as_i64()?
            .iter()
            .map(|&a| normalize_axis(a, data.shape.len()))
            .collect::<OnnxResult<_>>()?
    } else if let Some(ax) = get_ints_attr(attrs, "axes")? {
        ax.iter()
            .map(|&a| normalize_axis(a, data.shape.len()))
            .collect::<OnnxResult<_>>()?
    } else {
        // Squeeze all size-1 dims
        data.shape
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 1)
            .map(|(i, _)| i)
            .collect()
    };

    let new_shape: Vec<usize> = data
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes.contains(i))
        .map(|(_, &d)| d)
        .collect();

    Ok(vec![OnnxTensor {
        data: data.data.clone(),
        dtype: data.dtype,
        shape: new_shape,
    }])
}

/// Unsqueeze(data, axes) -> insert size-1 dimensions.
pub fn execute_unsqueeze(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let data = get_required_input(inputs, 0, "data")?;

    let axes_raw: Vec<i64> = if let Some(axes_input) = get_optional_input(inputs, 1) {
        axes_input.as_i64()?
    } else if let Some(ax) = get_ints_attr(attrs, "axes")? {
        ax.to_vec()
    } else {
        return Err(OnnxError::InvalidAttribute(
            "Unsqueeze requires axes".into(),
        ));
    };

    let new_rank = data.shape.len() + axes_raw.len();
    let mut axes: Vec<usize> = axes_raw
        .iter()
        .map(|&a| {
            if a < 0 {
                (new_rank as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    axes.sort_unstable();

    let mut new_shape = Vec::with_capacity(new_rank);
    let mut src_i = 0;
    for i in 0..new_rank {
        if axes.contains(&i) {
            new_shape.push(1);
        } else {
            new_shape.push(data.shape[src_i]);
            src_i += 1;
        }
    }

    Ok(vec![OnnxTensor {
        data: data.data.clone(),
        dtype: data.dtype,
        shape: new_shape,
    }])
}

/// Flatten(data, axis) -> reshape to 2D.
pub fn execute_flatten(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let data = get_required_input(inputs, 0, "data")?;
    let axis = get_int_attr(attrs, "axis", 1)?;
    let rank = data.shape.len() as i64;
    let a = if axis < 0 { rank + axis } else { axis } as usize;

    let d0: usize = data.shape[..a].iter().product::<usize>().max(1);
    let d1: usize = data.shape[a..].iter().product::<usize>().max(1);

    Ok(vec![OnnxTensor {
        data: data.data.clone(),
        dtype: data.dtype,
        shape: vec![d0, d1],
    }])
}

/// Concat(inputs..., axis) -> concatenate along axis.
pub fn execute_concat(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let axis = attrs
        .get("axis")
        .ok_or_else(|| OnnxError::InvalidAttribute("Concat requires 'axis'".into()))?
        .as_int()?;

    // Collect all non-None inputs
    let tensors: Vec<&OnnxTensor> = inputs.iter().filter_map(|o| *o).collect();
    if tensors.is_empty() {
        return Err(OnnxError::InvalidData("Concat: no inputs".into()));
    }

    let rank = tensors[0].shape.len();
    let a = normalize_axis(axis, rank)?;

    // Validate shapes match on non-concat dims
    for t in &tensors[1..] {
        for (i, (&d1, &d2)) in tensors[0].shape.iter().zip(t.shape.iter()).enumerate() {
            if i != a && d1 != d2 {
                return Err(OnnxError::ShapeMismatch(format!(
                    "Concat: dim {i} mismatch: {d1} vs {d2}"
                )));
            }
        }
    }

    let concat_dim: usize = tensors.iter().map(|t| t.shape[a]).sum();
    let mut out_shape = tensors[0].shape.clone();
    out_shape[a] = concat_dim;

    let out_total: usize = out_shape.iter().product();
    let mut result = vec![0.0f32; out_total];

    let mut offset = 0usize;
    for t in &tensors {
        let t_data = t.as_f32()?;
        let _t_total: usize = t.shape.iter().product();
        for (i, &val) in t_data.iter().enumerate() {
            let mut multi = flat_to_multi(i, &t.shape);
            multi[a] += offset;
            let oi = multi_to_flat(&multi, &out_shape);
            result[oi] = val;
        }
        offset += t.shape[a];
    }

    Ok(vec![OnnxTensor::from_f32(&result, out_shape)])
}

/// Split(data, split?, axis) -> split into multiple outputs.
pub fn execute_split(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let data = get_required_input(inputs, 0, "data")?;
    let axis = get_int_attr(attrs, "axis", 0)?;
    let rank = data.shape.len();
    let a = normalize_axis(axis, rank)?;
    let dim_size = data.shape[a];

    // Determine split sizes
    let splits: Vec<usize> = if let Some(split_input) = get_optional_input(inputs, 1) {
        split_input.as_i64()?.iter().map(|&v| v as usize).collect()
    } else if let Some(s) = get_ints_attr(attrs, "split")? {
        s.iter().map(|&v| v as usize).collect()
    } else {
        // Default: equal splits of 2
        let num_outputs = attrs
            .get("num_outputs")
            .map(|v| v.as_int())
            .transpose()?
            .unwrap_or(2) as usize;
        let base = dim_size / num_outputs;
        let remainder = dim_size % num_outputs;
        let mut splits = vec![base; num_outputs];
        for s in splits.iter_mut().take(remainder) {
            *s += 1;
        }
        splits
    };

    let data_f32 = data.as_f32()?;
    let _data_total: usize = data.shape.iter().product();
    let mut results = Vec::with_capacity(splits.len());
    let mut offset = 0usize;

    for &split_size in &splits {
        let mut out_shape = data.shape.clone();
        out_shape[a] = split_size;
        let out_total: usize = out_shape.iter().product();
        let mut out_data = vec![0.0f32; out_total];

        for (i, &val) in data_f32.iter().enumerate() {
            let multi = flat_to_multi(i, &data.shape);
            if multi[a] >= offset && multi[a] < offset + split_size {
                let mut out_multi = multi.clone();
                out_multi[a] -= offset;
                let oi = multi_to_flat(&out_multi, &out_shape);
                out_data[oi] = val;
            }
        }

        results.push(OnnxTensor::from_f32(&out_data, out_shape));
        offset += split_size;
    }

    Ok(results)
}

/// Gather(data, indices, axis) -> gather elements.
pub fn execute_gather(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let data = get_required_input(inputs, 0, "data")?;
    let indices = get_required_input(inputs, 1, "indices")?;
    let data_f32 = data.as_f32()?;
    let idx_vals = indices.as_i64()?;
    let axis = get_int_attr(attrs, "axis", 0)?;
    let a = normalize_axis(axis, data.shape.len())?;

    let dim_at_axis = data.shape[a];

    // Output shape: data.shape[:axis] + indices.shape + data.shape[axis+1:]
    let mut out_shape = data.shape[..a].to_vec();
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&data.shape[a + 1..]);

    let out_total: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let mut result = vec![0.0f32; out_total];

    let indices_total: usize = if indices.shape.is_empty() {
        1
    } else {
        indices.shape.iter().product()
    };

    // Pre-compute strides
    let outer: usize = data.shape[..a].iter().product::<usize>().max(1);
    let inner: usize = data.shape[a + 1..].iter().product::<usize>().max(1);

    for o in 0..outer {
        for (ii, &raw_idx) in idx_vals.iter().enumerate().take(indices_total) {
            let idx = if raw_idx < 0 {
                (dim_at_axis as i64 + raw_idx) as usize
            } else {
                raw_idx as usize
            };
            if idx >= dim_at_axis {
                return Err(OnnxError::InvalidData(format!(
                    "Gather: index {raw_idx} out of range for dim {dim_at_axis}"
                )));
            }
            for inn in 0..inner {
                let src = (o * dim_at_axis + idx) * inner + inn;
                let dst = (o * indices_total + ii) * inner + inn;
                result[dst] = data_f32[src];
            }
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, out_shape)])
}

/// Slice(data, starts, ends, axes?, steps?) -> slice along axes.
#[allow(clippy::too_many_lines)]
pub fn execute_slice(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let data = get_required_input(inputs, 0, "data")?;
    let starts_t = get_required_input(inputs, 1, "starts")?;
    let ends_t = get_required_input(inputs, 2, "ends")?;
    let axes_t = get_optional_input(inputs, 3);
    let steps_t = get_optional_input(inputs, 4);

    let starts = starts_t.as_i64()?;
    let ends = ends_t.as_i64()?;
    let rank = data.shape.len();

    let axes: Vec<usize> = if let Some(at) = axes_t {
        at.as_i64()?
            .iter()
            .map(|&a| normalize_axis(a, rank))
            .collect::<OnnxResult<_>>()?
    } else {
        (0..starts.len()).collect()
    };

    let steps: Vec<i64> = if let Some(st) = steps_t {
        st.as_i64()?
    } else {
        vec![1; axes.len()]
    };

    // Compute ranges for each axis
    let mut ranges: Vec<(i64, i64, i64)> =
        (0..rank).map(|i| (0, data.shape[i] as i64, 1)).collect();

    for (i, &ax) in axes.iter().enumerate() {
        let dim = data.shape[ax] as i64;
        let step = steps[i];
        let mut s = starts[i];
        let mut e = ends[i];

        // Clamp
        if s < 0 {
            s += dim;
        }
        if e < 0 {
            e += dim;
        }
        s = s.clamp(0, dim);
        e = e.clamp(0, dim);

        // Cap to i64::MAX sentinel handling
        if e > dim {
            e = dim;
        }

        ranges[ax] = (s, e, step);
    }

    // Compute output shape
    let mut out_shape = Vec::with_capacity(rank);
    for &(s, e, step) in &ranges {
        let len = if step > 0 {
            ((e - s) as f64 / step as f64).ceil().max(0.0) as usize
        } else if step < 0 {
            ((s - e) as f64 / (-step) as f64).ceil().max(0.0) as usize
        } else {
            return Err(OnnxError::InvalidAttribute(
                "Slice: step cannot be 0".into(),
            ));
        };
        out_shape.push(len);
    }

    let data_f32 = data.as_f32()?;
    let out_total: usize = out_shape.iter().product::<usize>().max(1);
    let mut result = vec![0.0f32; out_total];

    for (oi, result_val) in result.iter_mut().enumerate().take(out_total) {
        let out_multi = flat_to_multi(oi, &out_shape);
        let mut src_multi = vec![0usize; rank];
        for (d, &om) in out_multi.iter().enumerate() {
            let (s, _, step) = ranges[d];
            src_multi[d] = (s + om as i64 * step) as usize;
        }
        let si = multi_to_flat(&src_multi, &data.shape);
        *result_val = data_f32[si];
    }

    Ok(vec![OnnxTensor::from_f32(&result, out_shape)])
}

/// Pad(data, pads, constant_value?) -> padded tensor.
pub fn execute_pad(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let data = get_required_input(inputs, 0, "data")?;
    let pads_t = get_required_input(inputs, 1, "pads")?;
    let const_val = get_optional_input(inputs, 2);
    let data_f32 = data.as_f32()?;
    let pads = pads_t.as_i64()?;
    let rank = data.shape.len();

    let pad_val = if let Some(cv) = const_val {
        cv.as_f32()?.first().copied().unwrap_or(0.0)
    } else {
        0.0
    };

    let _mode = attrs
        .get("mode")
        .and_then(|v| v.as_string().ok())
        .unwrap_or("constant");

    // pads format: [begin_0, begin_1, ..., end_0, end_1, ...]
    let mut out_shape = Vec::with_capacity(rank);
    for i in 0..rank {
        let before = pads[i] as usize;
        let after = pads[rank + i] as usize;
        out_shape.push(data.shape[i] + before + after);
    }

    let out_total: usize = out_shape.iter().product::<usize>().max(1);
    let mut result = vec![pad_val; out_total];

    for (i, &val) in data_f32.iter().enumerate() {
        let multi = flat_to_multi(i, &data.shape);
        let mut out_multi = Vec::with_capacity(rank);
        for (d, &idx) in multi.iter().enumerate() {
            out_multi.push(idx + pads[d] as usize);
        }
        let oi = multi_to_flat(&out_multi, &out_shape);
        result[oi] = val;
    }

    Ok(vec![OnnxTensor::from_f32(&result, out_shape)])
}

/// Expand(data, shape) -> broadcast data to target shape.
pub fn execute_expand(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let data = get_required_input(inputs, 0, "data")?;
    let shape_t = get_required_input(inputs, 1, "shape")?;
    let target_shape: Vec<usize> = shape_t.as_i64()?.iter().map(|&v| v as usize).collect();
    let data_f32 = data.as_f32()?;

    let out_shape = broadcast_shapes(&data.shape, &target_shape)?;
    let total: usize = out_shape.iter().product::<usize>().max(1);
    let mut result = Vec::with_capacity(total);

    for i in 0..total {
        let multi = flat_to_multi(i, &out_shape);
        let src_idx = broadcast_index(&multi, &data.shape, &out_shape);
        result.push(data_f32[src_idx]);
    }

    Ok(vec![OnnxTensor::from_f32(&result, out_shape)])
}

/// Tile(data, repeats) -> tile/repeat tensor.
pub fn execute_tile(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let data = get_required_input(inputs, 0, "data")?;
    let repeats_t = get_required_input(inputs, 1, "repeats")?;
    let repeats: Vec<usize> = repeats_t.as_i64()?.iter().map(|&v| v as usize).collect();
    let data_f32 = data.as_f32()?;

    let mut out_shape = Vec::with_capacity(data.shape.len());
    for (i, &d) in data.shape.iter().enumerate() {
        let r = repeats.get(i).copied().unwrap_or(1);
        out_shape.push(d * r);
    }

    let total: usize = out_shape.iter().product::<usize>().max(1);
    let mut result = Vec::with_capacity(total);

    for i in 0..total {
        let multi = flat_to_multi(i, &out_shape);
        let src_multi: Vec<usize> = multi
            .iter()
            .enumerate()
            .map(|(d, &idx)| idx % data.shape[d])
            .collect();
        let si = multi_to_flat(&src_multi, &data.shape);
        result.push(data_f32[si]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape() {
        let data = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let shape = OnnxTensor::from_i64(&[3, 2], vec![2]);
        let r = execute_reshape(&[Some(&data), Some(&shape)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![3, 2]);
        assert_eq!(r[0].as_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_with_neg1() {
        let data = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let shape = OnnxTensor::from_i64(&[-1, 2], vec![2]);
        let r = execute_reshape(&[Some(&data), Some(&shape)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![3, 2]);
    }

    #[test]
    fn test_transpose() {
        let data = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let mut attrs = HashMap::new();
        attrs.insert("perm".into(), AttributeValue::Ints(vec![1, 0]));
        let r = execute_transpose(&[Some(&data)], &attrs).unwrap();
        assert_eq!(r[0].shape, vec![3, 2]);
        assert_eq!(r[0].as_f32().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_squeeze() {
        let data = OnnxTensor::from_f32(&[1.0, 2.0, 3.0], vec![1, 3, 1]);
        let axes = OnnxTensor::from_i64(&[0, 2], vec![2]);
        let r = execute_squeeze(&[Some(&data), Some(&axes)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![3]);
    }

    #[test]
    fn test_unsqueeze() {
        let data = OnnxTensor::from_f32(&[1.0, 2.0, 3.0], vec![3]);
        let axes = OnnxTensor::from_i64(&[0, 2], vec![2]);
        let r = execute_unsqueeze(&[Some(&data), Some(&axes)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![1, 3, 1]);
    }

    #[test]
    fn test_flatten() {
        let data = OnnxTensor::from_f32(&[1.0; 24], vec![2, 3, 4]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".into(), AttributeValue::Int(1));
        let r = execute_flatten(&[Some(&data)], &attrs).unwrap();
        assert_eq!(r[0].shape, vec![2, 12]);
    }

    #[test]
    fn test_concat() {
        let a = OnnxTensor::from_f32(&[1.0, 2.0], vec![1, 2]);
        let b = OnnxTensor::from_f32(&[3.0, 4.0], vec![1, 2]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".into(), AttributeValue::Int(0));
        let r = execute_concat(&[Some(&a), Some(&b)], &attrs).unwrap();
        assert_eq!(r[0].shape, vec![2, 2]);
        assert_eq!(r[0].as_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_concat_axis1() {
        let a = OnnxTensor::from_f32(&[1.0, 2.0], vec![1, 2]);
        let b = OnnxTensor::from_f32(&[3.0], vec![1, 1]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".into(), AttributeValue::Int(1));
        let r = execute_concat(&[Some(&a), Some(&b)], &attrs).unwrap();
        assert_eq!(r[0].shape, vec![1, 3]);
        assert_eq!(r[0].as_f32().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_split() {
        let data = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
        let split = OnnxTensor::from_i64(&[2, 4], vec![2]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".into(), AttributeValue::Int(0));
        let r = execute_split(&[Some(&data), Some(&split)], &attrs).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].as_f32().unwrap(), vec![1.0, 2.0]);
        assert_eq!(r[1].as_f32().unwrap(), vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_gather() {
        let data = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let indices = OnnxTensor::from_i64(&[0, 3, 1], vec![3]);
        let r = execute_gather(&[Some(&data), Some(&indices)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![3]);
        assert_eq!(r[0].as_f32().unwrap(), vec![1.0, 4.0, 2.0]);
    }

    #[test]
    fn test_slice() {
        let data = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let starts = OnnxTensor::from_i64(&[1], vec![1]);
        let ends = OnnxTensor::from_i64(&[4], vec![1]);
        let axes = OnnxTensor::from_i64(&[0], vec![1]);
        let r = execute_slice(
            &[Some(&data), Some(&starts), Some(&ends), Some(&axes)],
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(r[0].shape, vec![3]);
        assert_eq!(r[0].as_f32().unwrap(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pad() {
        let data = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let pads = OnnxTensor::from_i64(&[1, 0, 0, 1], vec![4]); // before=[1,0], after=[0,1]
        let r = execute_pad(&[Some(&data), Some(&pads)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![3, 3]);
        assert_eq!(
            r[0].as_f32().unwrap(),
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0]
        );
    }

    #[test]
    fn test_expand() {
        let data = OnnxTensor::from_f32(&[1.0, 2.0, 3.0], vec![1, 3]);
        let shape = OnnxTensor::from_i64(&[3, 3], vec![2]);
        let r = execute_expand(&[Some(&data), Some(&shape)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![3, 3]);
        assert_eq!(
            r[0].as_f32().unwrap(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_tile() {
        let data = OnnxTensor::from_f32(&[1.0, 2.0], vec![2]);
        let repeats = OnnxTensor::from_i64(&[3], vec![1]);
        let r = execute_tile(&[Some(&data), Some(&repeats)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![6]);
        assert_eq!(r[0].as_f32().unwrap(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }
}
