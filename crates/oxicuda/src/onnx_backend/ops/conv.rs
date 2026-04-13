//! Convolution and pooling ONNX operators.

use std::collections::HashMap;

use super::{get_int_attr, get_ints_attr, get_optional_input, get_required_input};
use crate::onnx_backend::ir::*;

// ─── Helper: extract conv/pool spatial parameters ───────────

struct SpatialParams {
    strides: Vec<usize>,
    pads: Vec<usize>, // [top, left, bottom, right] for 2D
    dilations: Vec<usize>,
    group: usize,
}

fn get_spatial_params(
    attrs: &HashMap<String, AttributeValue>,
    spatial_rank: usize,
) -> OnnxResult<SpatialParams> {
    let strides = if let Some(s) = get_ints_attr(attrs, "strides")? {
        s.iter().map(|&v| v as usize).collect()
    } else {
        vec![1; spatial_rank]
    };
    let pads = if let Some(p) = get_ints_attr(attrs, "pads")? {
        p.iter().map(|&v| v as usize).collect()
    } else {
        vec![0; spatial_rank * 2]
    };
    let dilations = if let Some(d) = get_ints_attr(attrs, "dilations")? {
        d.iter().map(|&v| v as usize).collect()
    } else {
        vec![1; spatial_rank]
    };
    let group = get_int_attr(attrs, "group", 1)? as usize;
    Ok(SpatialParams {
        strides,
        pads,
        dilations,
        group,
    })
}

fn output_size(
    input: usize,
    pad_begin: usize,
    pad_end: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
) -> usize {
    let effective_kernel = dilation * (kernel - 1) + 1;
    (input + pad_begin + pad_end - effective_kernel) / stride + 1
}

// ─── Conv ───────────────────────────────────────────────────

/// Conv(X, W, B?) -> convolution.
#[allow(clippy::too_many_lines)]
pub fn execute_conv(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "X")?;
    let w = get_required_input(inputs, 1, "W")?;
    let bias = get_optional_input(inputs, 2);
    let x_data = x.as_f32()?;
    let w_data = w.as_f32()?;

    // Only 2D convolution for now: X=[N,C,H,W], W=[OC,IC/g,KH,KW]
    if x.shape.len() != 4 || w.shape.len() != 4 {
        return Err(OnnxError::UnsupportedOp(
            "Conv only supports 2D (NCHW)".into(),
        ));
    }

    let n = x.shape[0];
    let _ic = x.shape[1];
    let ih = x.shape[2];
    let iw = x.shape[3];
    let oc = w.shape[0];
    let ic_per_group = w.shape[1];
    let kh = w.shape[2];
    let kw = w.shape[3];

    let sp = get_spatial_params(attrs, 2)?;
    let oh = output_size(
        ih,
        sp.pads[0],
        sp.pads[2],
        kh,
        sp.dilations[0],
        sp.strides[0],
    );
    let ow = output_size(
        iw,
        sp.pads[1],
        sp.pads[3],
        kw,
        sp.dilations[1],
        sp.strides[1],
    );
    let oc_per_group = oc / sp.group;

    let bias_data = if let Some(b) = bias {
        Some(b.as_f32()?)
    } else {
        None
    };

    let mut result = vec![0.0f32; n * oc * oh * ow];

    for batch in 0..n {
        for g in 0..sp.group {
            for oc_i in 0..oc_per_group {
                let abs_oc = g * oc_per_group + oc_i;
                let b_val = bias_data
                    .as_ref()
                    .and_then(|b| b.get(abs_oc).copied())
                    .unwrap_or(0.0);

                for y in 0..oh {
                    for x_pos in 0..ow {
                        let mut sum = b_val;
                        for ic_i in 0..ic_per_group {
                            let abs_ic = g * ic_per_group + ic_i;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = y * sp.strides[0] + ky * sp.dilations[0];
                                    let ix = x_pos * sp.strides[1] + kx * sp.dilations[1];
                                    // Check padding
                                    if iy >= sp.pads[0]
                                        && ix >= sp.pads[1]
                                        && (iy - sp.pads[0]) < ih
                                        && (ix - sp.pads[1]) < iw
                                    {
                                        let src_y = iy - sp.pads[0];
                                        let src_x = ix - sp.pads[1];
                                        let x_idx =
                                            ((batch * _ic + abs_ic) * ih + src_y) * iw + src_x;
                                        let w_idx =
                                            ((abs_oc * ic_per_group + ic_i) * kh + ky) * kw + kx;
                                        sum += x_data[x_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }
                        let out_idx = ((batch * oc + abs_oc) * oh + y) * ow + x_pos;
                        result[out_idx] = sum;
                    }
                }
            }
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, vec![n, oc, oh, ow])])
}

/// ConvTranspose(X, W, B?) -> transposed convolution.
#[allow(clippy::too_many_lines)]
pub fn execute_conv_transpose(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "X")?;
    let w = get_required_input(inputs, 1, "W")?;
    let bias = get_optional_input(inputs, 2);
    let x_data = x.as_f32()?;
    let w_data = w.as_f32()?;

    if x.shape.len() != 4 || w.shape.len() != 4 {
        return Err(OnnxError::UnsupportedOp(
            "ConvTranspose only supports 2D".into(),
        ));
    }

    let n = x.shape[0];
    let ic = x.shape[1];
    let ih = x.shape[2];
    let iw_dim = x.shape[3];
    // W shape for ConvTranspose: [IC, OC/g, KH, KW]
    let oc_per_group = w.shape[1];
    let kh = w.shape[2];
    let kw = w.shape[3];

    let sp = get_spatial_params(attrs, 2)?;
    let group = sp.group;
    let oc = oc_per_group * group;
    let ic_per_group = ic / group;

    let output_padding_h = if let Some(op) = get_ints_attr(attrs, "output_padding")? {
        op.first().copied().unwrap_or(0) as usize
    } else {
        0
    };
    let output_padding_w = if let Some(op) = get_ints_attr(attrs, "output_padding")? {
        op.get(1).copied().unwrap_or(0) as usize
    } else {
        0
    };

    let oh = sp.strides[0] * (ih - 1) + sp.dilations[0] * (kh - 1) + 1 - sp.pads[0] - sp.pads[2]
        + output_padding_h;
    let ow =
        sp.strides[1] * (iw_dim - 1) + sp.dilations[1] * (kw - 1) + 1 - sp.pads[1] - sp.pads[3]
            + output_padding_w;

    let bias_data = if let Some(b) = bias {
        Some(b.as_f32()?)
    } else {
        None
    };

    let mut result = vec![0.0f32; n * oc * oh * ow];

    // Initialize with bias
    if let Some(ref bd) = bias_data {
        for batch in 0..n {
            for c in 0..oc {
                let bv = bd.get(c).copied().unwrap_or(0.0);
                for y in 0..oh {
                    for xp in 0..ow {
                        result[((batch * oc + c) * oh + y) * ow + xp] = bv;
                    }
                }
            }
        }
    }

    // Transposed convolution: scatter input into output
    for batch in 0..n {
        for g in 0..group {
            for ic_i in 0..ic_per_group {
                let abs_ic = g * ic_per_group + ic_i;
                for iy in 0..ih {
                    for ix in 0..iw_dim {
                        let x_val = x_data[((batch * ic + abs_ic) * ih + iy) * iw_dim + ix];
                        for oc_i in 0..oc_per_group {
                            let abs_oc = g * oc_per_group + oc_i;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let oy_raw = iy * sp.strides[0] + ky * sp.dilations[0];
                                    let ox_raw = ix * sp.strides[1] + kx * sp.dilations[1];
                                    if oy_raw >= sp.pads[0] && ox_raw >= sp.pads[1] {
                                        let oy = oy_raw - sp.pads[0];
                                        let ox = ox_raw - sp.pads[1];
                                        if oy < oh && ox < ow {
                                            let w_idx = ((abs_ic * oc_per_group + oc_i) * kh + ky)
                                                * kw
                                                + kx;
                                            let out_idx =
                                                ((batch * oc + abs_oc) * oh + oy) * ow + ox;
                                            result[out_idx] += x_val * w_data[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, vec![n, oc, oh, ow])])
}

/// MaxPool(X) -> max pooling.
pub fn execute_max_pool(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "X")?;
    let x_data = x.as_f32()?;

    if x.shape.len() != 4 {
        return Err(OnnxError::UnsupportedOp("MaxPool requires 4D input".into()));
    }

    let n = x.shape[0];
    let c = x.shape[1];
    let ih = x.shape[2];
    let iw = x.shape[3];

    let kernel_shape = get_ints_attr(attrs, "kernel_shape")?
        .ok_or_else(|| OnnxError::InvalidAttribute("MaxPool requires kernel_shape".into()))?;
    let kh = kernel_shape[0] as usize;
    let kw = kernel_shape[1] as usize;

    let sp = get_spatial_params(attrs, 2)?;
    let oh = output_size(
        ih,
        sp.pads[0],
        sp.pads[2],
        kh,
        sp.dilations[0],
        sp.strides[0],
    );
    let ow = output_size(
        iw,
        sp.pads[1],
        sp.pads[3],
        kw,
        sp.dilations[1],
        sp.strides[1],
    );

    let mut result = vec![f32::NEG_INFINITY; n * c * oh * ow];

    for batch in 0..n {
        for ch in 0..c {
            for y in 0..oh {
                for xp in 0..ow {
                    let mut max_val = f32::NEG_INFINITY;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = y * sp.strides[0] + ky * sp.dilations[0];
                            let ix = xp * sp.strides[1] + kx * sp.dilations[1];
                            if iy >= sp.pads[0]
                                && ix >= sp.pads[1]
                                && (iy - sp.pads[0]) < ih
                                && (ix - sp.pads[1]) < iw
                            {
                                let src_y = iy - sp.pads[0];
                                let src_x = ix - sp.pads[1];
                                let idx = ((batch * c + ch) * ih + src_y) * iw + src_x;
                                if x_data[idx] > max_val {
                                    max_val = x_data[idx];
                                }
                            }
                        }
                    }
                    result[((batch * c + ch) * oh + y) * ow + xp] = max_val;
                }
            }
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, vec![n, c, oh, ow])])
}

/// AveragePool(X) -> average pooling.
pub fn execute_average_pool(
    inputs: &[Option<&OnnxTensor>],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "X")?;
    let x_data = x.as_f32()?;

    if x.shape.len() != 4 {
        return Err(OnnxError::UnsupportedOp(
            "AveragePool requires 4D input".into(),
        ));
    }

    let n = x.shape[0];
    let c = x.shape[1];
    let ih = x.shape[2];
    let iw = x.shape[3];

    let kernel_shape = get_ints_attr(attrs, "kernel_shape")?
        .ok_or_else(|| OnnxError::InvalidAttribute("AveragePool needs kernel_shape".into()))?;
    let kh = kernel_shape[0] as usize;
    let kw = kernel_shape[1] as usize;

    let sp = get_spatial_params(attrs, 2)?;
    let count_include_pad = get_int_attr(attrs, "count_include_pad", 0)? != 0;
    let oh = output_size(
        ih,
        sp.pads[0],
        sp.pads[2],
        kh,
        sp.dilations[0],
        sp.strides[0],
    );
    let ow = output_size(
        iw,
        sp.pads[1],
        sp.pads[3],
        kw,
        sp.dilations[1],
        sp.strides[1],
    );

    let mut result = vec![0.0f32; n * c * oh * ow];

    for batch in 0..n {
        for ch in 0..c {
            for y in 0..oh {
                for xp in 0..ow {
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = y * sp.strides[0] + ky * sp.dilations[0];
                            let ix = xp * sp.strides[1] + kx * sp.dilations[1];
                            if iy >= sp.pads[0]
                                && ix >= sp.pads[1]
                                && (iy - sp.pads[0]) < ih
                                && (ix - sp.pads[1]) < iw
                            {
                                let src_y = iy - sp.pads[0];
                                let src_x = ix - sp.pads[1];
                                let idx = ((batch * c + ch) * ih + src_y) * iw + src_x;
                                sum += x_data[idx];
                                count += 1;
                            } else if count_include_pad {
                                count += 1;
                            }
                        }
                    }
                    let divisor = if count_include_pad {
                        (kh * kw) as f32
                    } else if count > 0 {
                        count as f32
                    } else {
                        1.0
                    };
                    result[((batch * c + ch) * oh + y) * ow + xp] = sum / divisor;
                }
            }
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, vec![n, c, oh, ow])])
}

/// GlobalAveragePool(X) -> average over all spatial dimensions.
pub fn execute_global_average_pool(
    inputs: &[Option<&OnnxTensor>],
    _attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<OnnxTensor>> {
    let x = get_required_input(inputs, 0, "X")?;
    let x_data = x.as_f32()?;

    if x.shape.len() != 4 {
        return Err(OnnxError::UnsupportedOp(
            "GlobalAveragePool requires 4D".into(),
        ));
    }

    let n = x.shape[0];
    let c = x.shape[1];
    let h = x.shape[2];
    let w = x.shape[3];
    let spatial = h * w;

    let mut result = vec![0.0f32; n * c];
    for batch in 0..n {
        for ch in 0..c {
            let mut sum = 0.0f32;
            let base = (batch * c + ch) * spatial;
            for i in 0..spatial {
                sum += x_data[base + i];
            }
            result[batch * c + ch] = sum / spatial as f32;
        }
    }

    Ok(vec![OnnxTensor::from_f32(&result, vec![n, c, 1, 1])])
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
    fn test_conv_1x1() {
        // 1x1 conv: just a linear transformation per pixel
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let w = OnnxTensor::from_f32(&[2.0], vec![1, 1, 1, 1]);
        let r = execute_conv(&[Some(&x), Some(&w)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![1, 1, 2, 2]);
        assert_eq!(r[0].as_f32().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_conv_3x3_nopad() {
        // 3x3 conv on 3x3 input -> 1x1 output
        let x = OnnxTensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
        );
        let w = OnnxTensor::from_f32(&[1.0; 9], vec![1, 1, 3, 3]);
        let r = execute_conv(&[Some(&x), Some(&w)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![1, 1, 1, 1]);
        assert_f32_near(&r[0].as_f32().unwrap(), &[45.0], 1e-5);
    }

    #[test]
    fn test_conv_with_bias() {
        let x = OnnxTensor::from_f32(&[1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
        let w = OnnxTensor::from_f32(&[1.0], vec![1, 1, 1, 1]);
        let b = OnnxTensor::from_f32(&[10.0], vec![1]);
        let r = execute_conv(&[Some(&x), Some(&w), Some(&b)], &HashMap::new()).unwrap();
        assert_eq!(r[0].as_f32().unwrap(), vec![11.0, 11.0, 11.0, 11.0]);
    }

    #[test]
    fn test_max_pool() {
        // 2x2 max pool on 4x4 input
        let x = OnnxTensor::from_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            vec![1, 1, 4, 4],
        );
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".into(), AttributeValue::Ints(vec![2, 2]));
        attrs.insert("strides".into(), AttributeValue::Ints(vec![2, 2]));
        let r = execute_max_pool(&[Some(&x)], &attrs).unwrap();
        assert_eq!(r[0].shape, vec![1, 1, 2, 2]);
        assert_eq!(r[0].as_f32().unwrap(), vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_average_pool() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".into(), AttributeValue::Ints(vec![2, 2]));
        attrs.insert("strides".into(), AttributeValue::Ints(vec![2, 2]));
        let r = execute_average_pool(&[Some(&x)], &attrs).unwrap();
        assert_eq!(r[0].shape, vec![1, 1, 1, 1]);
        assert_f32_near(&r[0].as_f32().unwrap(), &[2.5], 1e-5);
    }

    #[test]
    fn test_global_average_pool() {
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 2, 2, 2]);
        let r = execute_global_average_pool(&[Some(&x)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![1, 2, 1, 1]);
        assert_f32_near(&r[0].as_f32().unwrap(), &[2.5, 6.5], 1e-5);
    }

    #[test]
    fn test_conv_transpose_basic() {
        // Simple 1x1 conv transpose
        let x = OnnxTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let w = OnnxTensor::from_f32(&[1.0], vec![1, 1, 1, 1]);
        let r = execute_conv_transpose(&[Some(&x), Some(&w)], &HashMap::new()).unwrap();
        assert_eq!(r[0].shape, vec![1, 1, 2, 2]);
        assert_f32_near(&r[0].as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-5);
    }
}
