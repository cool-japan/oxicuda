//! Shape inference engine for ONNX operators.
//!
//! Infers output shapes from input shapes, operator type, and attributes
//! without actually executing the operators.

use std::collections::HashMap;

use super::ir::*;

/// Shape inference engine.
pub struct ShapeInference;

impl ShapeInference {
    /// Infer the output shapes for a single operator node.
    ///
    /// `input_shapes` are the shapes of the node's inputs (in order).
    /// Returns shapes for each output tensor.
    #[allow(clippy::too_many_lines)]
    pub fn infer(
        op_type: &str,
        input_shapes: &[&TensorShape],
        attrs: &HashMap<String, AttributeValue>,
    ) -> OnnxResult<Vec<TensorShape>> {
        match op_type {
            // ── Elementwise binary (broadcast) ──
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Where" => infer_broadcast(input_shapes),

            // ── Elementwise unary (pass-through) ──
            "Relu" | "Sigmoid" | "Tanh" | "Exp" | "Log" | "Sqrt" | "Abs" | "Neg" | "LeakyRelu"
            | "Elu" | "Selu" | "Softplus" | "Clip" | "Ceil" | "Floor" | "Round" | "Sign"
            | "Reciprocal" => infer_passthrough(input_shapes),

            "Cast" => infer_passthrough(input_shapes),

            // ── Reduction ──
            "ReduceSum" | "ReduceMean" | "ReduceMax" | "ReduceMin" | "ReduceProd" => {
                infer_reduction(input_shapes, attrs)
            }

            // ── MatMul ──
            "MatMul" => infer_matmul(input_shapes),
            "Gemm" | "FusedBiasedMatMul" => infer_gemm(input_shapes, attrs),

            // ── Convolution / Pooling ──
            "Conv" | "FusedConvBn" | "FusedConvBnRelu" => infer_conv(input_shapes, attrs),
            "ConvTranspose" => infer_conv_transpose(input_shapes, attrs),
            "MaxPool" | "AveragePool" => infer_pool(input_shapes, attrs),
            "GlobalAveragePool" => infer_global_pool(input_shapes),

            // ── Normalization ──
            "BatchNormalization"
            | "LayerNormalization"
            | "InstanceNormalization"
            | "GroupNormalization"
            | "FlashAttention" => infer_passthrough(input_shapes),

            // ── Activation ──
            "Softmax" | "LogSoftmax" => infer_passthrough(input_shapes),

            // ── Shape manipulation ──
            "Reshape" => infer_reshape(input_shapes),
            "Transpose" => infer_transpose(input_shapes, attrs),
            "Squeeze" => infer_squeeze(input_shapes, attrs),
            "Unsqueeze" => infer_unsqueeze(input_shapes, attrs),
            "Flatten" => infer_flatten(input_shapes, attrs),
            "Concat" => infer_concat(input_shapes, attrs),
            "Split" => infer_split(input_shapes, attrs),
            "Gather" => infer_gather(input_shapes, attrs),
            "Slice" => Ok(vec![input_shapes.first().map_or_else(
                || TensorShape::new(vec![]),
                |s| {
                    // Slice output is dynamic without concrete starts/ends
                    TensorShape::new(vec![None; s.rank()])
                },
            )]),
            "Pad" => Ok(vec![TensorShape::new(vec![
                None;
                input_shapes
                    .first()
                    .map_or(0, |s| s.rank())
            ])]),
            "Expand" => infer_expand(input_shapes),
            "Tile" => Ok(vec![TensorShape::new(vec![
                None;
                input_shapes
                    .first()
                    .map_or(0, |s| s.rank())
            ])]),

            // ── Misc ──
            "Identity" | "Dropout" => infer_passthrough(input_shapes),
            "Constant" => infer_constant(attrs),
            "Shape" => {
                let rank = input_shapes.first().map_or(0, |s| s.rank());
                Ok(vec![TensorShape::fixed(vec![rank])])
            }
            "Size" => Ok(vec![TensorShape::fixed(vec![])]),

            // ── Fused ──
            "FusedFMA" => infer_broadcast(input_shapes),

            _ => Err(OnnxError::UnsupportedOp(format!(
                "shape inference for '{op_type}'"
            ))),
        }
    }

    /// Infer shapes for all nodes in a graph, propagating through the topology.
    pub fn infer_graph(graph: &Graph) -> OnnxResult<HashMap<String, TensorShape>> {
        let mut shapes: HashMap<String, TensorShape> = HashMap::new();

        // Seed with graph inputs
        for info in &graph.inputs {
            shapes.insert(info.name.clone(), info.shape.clone());
        }

        // Seed with initializers
        for (name, tensor) in &graph.initializers {
            shapes.insert(name.clone(), TensorShape::fixed(tensor.shape.clone()));
        }

        // Process nodes (assuming they're in topological order or close)
        // Multiple passes to handle forward references
        for _pass in 0..3 {
            for node in &graph.nodes {
                let input_shapes: Vec<Option<&TensorShape>> = node
                    .inputs
                    .iter()
                    .map(|name| {
                        if name.is_empty() {
                            None
                        } else {
                            shapes.get(name.as_str())
                        }
                    })
                    .collect();

                // Skip if any required input is missing
                let available: Vec<&TensorShape> = input_shapes.iter().filter_map(|o| *o).collect();
                if available.is_empty() && !node.inputs.is_empty() {
                    continue;
                }

                if let Ok(outputs) = Self::infer(&node.op_type, &available, &node.attributes) {
                    for (i, shape) in outputs.into_iter().enumerate() {
                        if let Some(name) = node.outputs.get(i) {
                            if !name.is_empty() {
                                shapes.insert(name.clone(), shape);
                            }
                        }
                    }
                }
            }
        }

        Ok(shapes)
    }
}

// ─── Inference helpers ──────────────────────────────────────

fn infer_passthrough(inputs: &[&TensorShape]) -> OnnxResult<Vec<TensorShape>> {
    let shape = inputs
        .first()
        .ok_or_else(|| OnnxError::ShapeMismatch("no inputs".into()))?;
    Ok(vec![(*shape).clone()])
}

fn infer_broadcast(inputs: &[&TensorShape]) -> OnnxResult<Vec<TensorShape>> {
    if inputs.is_empty() {
        return Err(OnnxError::ShapeMismatch("no inputs for broadcast".into()));
    }
    let mut result = inputs[0].clone();
    for &input in &inputs[1..] {
        result = broadcast_tensor_shapes(&result, input)?;
    }
    Ok(vec![result])
}

fn broadcast_tensor_shapes(a: &TensorShape, b: &TensorShape) -> OnnxResult<TensorShape> {
    let max_rank = a.rank().max(b.rank());
    let mut dims = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let da = if i < max_rank - a.rank() {
            Some(1)
        } else {
            a.dims[i + a.rank() - max_rank]
        };
        let db = if i < max_rank - b.rank() {
            Some(1)
        } else {
            b.dims[i + b.rank() - max_rank]
        };

        match (da, db) {
            (Some(x), Some(y)) if x == y => dims.push(Some(x)),
            (Some(1), d) | (d, Some(1)) => dims.push(d),
            (Some(x), Some(y)) => {
                return Err(OnnxError::ShapeMismatch(format!(
                    "cannot broadcast {x} and {y}"
                )));
            }
            (None, _) | (_, None) => dims.push(None), // dynamic
        }
    }
    Ok(TensorShape::new(dims))
}

fn infer_reduction(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    let input = inputs
        .first()
        .ok_or_else(|| OnnxError::ShapeMismatch("no input".into()))?;
    let rank = input.rank();
    let keepdims = attrs
        .get("keepdims")
        .and_then(|v| v.as_int().ok())
        .unwrap_or(1)
        != 0;

    let axes: Vec<usize> = if let Some(AttributeValue::Ints(ax)) = attrs.get("axes") {
        ax.iter()
            .map(|&a| {
                if a < 0 {
                    (rank as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    } else {
        (0..rank).collect() // reduce all
    };

    let mut out_dims = Vec::new();
    for (i, d) in input.dims.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                out_dims.push(Some(1));
            }
        } else {
            out_dims.push(*d);
        }
    }
    Ok(vec![TensorShape::new(out_dims)])
}

fn infer_matmul(inputs: &[&TensorShape]) -> OnnxResult<Vec<TensorShape>> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch("MatMul needs 2 inputs".into()));
    }
    let a = inputs[0];
    let b = inputs[1];

    // 1D x 1D → scalar
    if a.rank() == 1 && b.rank() == 1 {
        return Ok(vec![TensorShape::fixed(vec![])]);
    }

    // 1D x 2D
    if a.rank() == 1 && b.rank() >= 2 {
        let mut out = b.dims[..b.rank() - 2].to_vec();
        out.push(b.dims[b.rank() - 1]);
        return Ok(vec![TensorShape::new(out)]);
    }

    // 2D x 1D
    if a.rank() >= 2 && b.rank() == 1 {
        let out = a.dims[..a.rank() - 1].to_vec();
        return Ok(vec![TensorShape::new(out)]);
    }

    // General: broadcast batch dims, then [M, N]
    let a_batch = &a.dims[..a.rank() - 2];
    let b_batch = &b.dims[..b.rank() - 2];
    let batch_shape = broadcast_opt_dims(a_batch, b_batch)?;

    let m = a.dims[a.rank() - 2];
    let n = b.dims[b.rank() - 1];

    let mut out = batch_shape;
    out.push(m);
    out.push(n);
    Ok(vec![TensorShape::new(out)])
}

fn broadcast_opt_dims(a: &[Option<usize>], b: &[Option<usize>]) -> OnnxResult<Vec<Option<usize>>> {
    let max_rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_rank);
    for i in 0..max_rank {
        let da = if i < max_rank - a.len() {
            Some(1)
        } else {
            a[i + a.len() - max_rank]
        };
        let db = if i < max_rank - b.len() {
            Some(1)
        } else {
            b[i + b.len() - max_rank]
        };
        match (da, db) {
            (Some(x), Some(y)) if x == y => result.push(Some(x)),
            (Some(1), d) | (d, Some(1)) => result.push(d),
            (None, _) | (_, None) => result.push(None),
            (Some(x), Some(y)) => {
                return Err(OnnxError::ShapeMismatch(format!(
                    "cannot broadcast batch dims {x} and {y}"
                )));
            }
        }
    }
    Ok(result)
}

fn infer_gemm(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch("Gemm needs >=2 inputs".into()));
    }
    let a = inputs[0];
    let b = inputs[1];
    let trans_a = attrs
        .get("transA")
        .and_then(|v| v.as_int().ok())
        .unwrap_or(0)
        != 0;
    let trans_b = attrs
        .get("transB")
        .and_then(|v| v.as_int().ok())
        .unwrap_or(0)
        != 0;

    let m = if trans_a {
        a.dims.get(1).copied().flatten()
    } else {
        a.dims.first().copied().flatten()
    };
    let n = if trans_b {
        b.dims.first().copied().flatten()
    } else {
        b.dims.get(1).copied().flatten()
    };

    Ok(vec![TensorShape::new(vec![m, n])])
}

fn infer_conv(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch("Conv needs >=2 inputs".into()));
    }
    let x = inputs[0]; // [N, C, H, W]
    let w = inputs[1]; // [OC, IC/g, KH, KW]

    if x.rank() != 4 || w.rank() != 4 {
        return Ok(vec![TensorShape::new(vec![None; 4])]);
    }

    let n = x.dims[0];
    let oc = w.dims[0];

    let strides = get_spatial_ints(attrs, "strides", 2);
    let pads = get_spatial_ints(attrs, "pads", 4);
    let dilations = get_spatial_ints(attrs, "dilations", 2);

    let oh = compute_conv_dim(
        x.dims[2],
        w.dims[2],
        pads[0],
        pads[2],
        dilations[0],
        strides[0],
    );
    let ow = compute_conv_dim(
        x.dims[3],
        w.dims[3],
        pads[1],
        pads[3],
        dilations[1],
        strides[1],
    );

    Ok(vec![TensorShape::new(vec![n, oc, oh, ow])])
}

fn infer_conv_transpose(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch(
            "ConvTranspose needs >=2 inputs".into(),
        ));
    }
    let x = inputs[0];
    let w = inputs[1]; // [IC, OC/g, KH, KW]

    if x.rank() != 4 || w.rank() != 4 {
        return Ok(vec![TensorShape::new(vec![None; 4])]);
    }

    let n = x.dims[0];
    let group = attrs
        .get("group")
        .and_then(|v| v.as_int().ok())
        .unwrap_or(1) as usize;
    let oc = w.dims[1].map(|v| v * group);

    let strides = get_spatial_ints(attrs, "strides", 2);
    let pads = get_spatial_ints(attrs, "pads", 4);
    let dilations = get_spatial_ints(attrs, "dilations", 2);

    let oh = compute_conv_transpose_dim(
        x.dims[2],
        w.dims[2],
        pads[0],
        pads[2],
        dilations[0],
        strides[0],
    );
    let ow = compute_conv_transpose_dim(
        x.dims[3],
        w.dims[3],
        pads[1],
        pads[3],
        dilations[1],
        strides[1],
    );

    Ok(vec![TensorShape::new(vec![n, oc, oh, ow])])
}

fn infer_pool(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    let x = inputs
        .first()
        .ok_or_else(|| OnnxError::ShapeMismatch("Pool needs input".into()))?;
    if x.rank() != 4 {
        return Ok(vec![TensorShape::new(vec![None; 4])]);
    }

    let kernel = attrs
        .get("kernel_shape")
        .and_then(|v| v.as_ints().ok())
        .map(|v| v.iter().map(|&i| Some(i as usize)).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![None; 2]);
    let strides = get_spatial_ints(attrs, "strides", 2);
    let pads = get_spatial_ints(attrs, "pads", 4);

    let oh = compute_conv_dim(x.dims[2], kernel[0], pads[0], pads[2], Some(1), strides[0]);
    let ow = compute_conv_dim(x.dims[3], kernel[1], pads[1], pads[3], Some(1), strides[1]);

    Ok(vec![TensorShape::new(vec![x.dims[0], x.dims[1], oh, ow])])
}

fn infer_global_pool(inputs: &[&TensorShape]) -> OnnxResult<Vec<TensorShape>> {
    let x = inputs
        .first()
        .ok_or_else(|| OnnxError::ShapeMismatch("GlobalPool needs input".into()))?;
    if x.rank() != 4 {
        return Ok(vec![TensorShape::new(vec![None; 4])]);
    }
    Ok(vec![TensorShape::new(vec![
        x.dims[0],
        x.dims[1],
        Some(1),
        Some(1),
    ])])
}

fn infer_reshape(inputs: &[&TensorShape]) -> OnnxResult<Vec<TensorShape>> {
    // Without concrete shape values, we can only return dynamic
    if inputs.len() < 2 {
        return Ok(vec![TensorShape::new(vec![])]);
    }
    let target = inputs[1];
    // If the shape tensor has a known rank, use that
    if let Some(rank) = target.dims.first().and_then(|d| *d) {
        Ok(vec![TensorShape::new(vec![None; rank])])
    } else {
        Ok(vec![TensorShape::new(vec![None])])
    }
}

fn infer_transpose(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    let input = inputs
        .first()
        .ok_or_else(|| OnnxError::ShapeMismatch("no input".into()))?;

    let perm: Vec<usize> = if let Some(AttributeValue::Ints(p)) = attrs.get("perm") {
        p.iter().map(|&v| v as usize).collect()
    } else {
        (0..input.rank()).rev().collect()
    };

    let out_dims: Vec<Option<usize>> = perm.iter().map(|&p| input.dims[p]).collect();
    Ok(vec![TensorShape::new(out_dims)])
}

fn infer_squeeze(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    let input = inputs
        .first()
        .ok_or_else(|| OnnxError::ShapeMismatch("no input".into()))?;

    let axes: Option<Vec<usize>> = if let Some(AttributeValue::Ints(ax)) = attrs.get("axes") {
        Some(
            ax.iter()
                .map(|&a| {
                    if a < 0 {
                        (input.rank() as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect(),
        )
    } else {
        None
    };

    let out_dims: Vec<Option<usize>> = input
        .dims
        .iter()
        .enumerate()
        .filter(|(i, d)| {
            if let Some(ref ax) = axes {
                !ax.contains(i)
            } else {
                *d != &Some(1)
            }
        })
        .map(|(_, d)| *d)
        .collect();
    Ok(vec![TensorShape::new(out_dims)])
}

fn infer_unsqueeze(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    let input = inputs
        .first()
        .ok_or_else(|| OnnxError::ShapeMismatch("no input".into()))?;

    let axes: Vec<usize> = if let Some(AttributeValue::Ints(ax)) = attrs.get("axes") {
        let new_rank = input.rank() + ax.len();
        ax.iter()
            .map(|&a| {
                if a < 0 {
                    (new_rank as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    } else {
        return Ok(vec![(*input).clone()]);
    };

    let new_rank = input.rank() + axes.len();
    let mut out_dims = Vec::with_capacity(new_rank);
    let mut src_i = 0;
    for i in 0..new_rank {
        if axes.contains(&i) {
            out_dims.push(Some(1));
        } else {
            out_dims.push(input.dims[src_i]);
            src_i += 1;
        }
    }
    Ok(vec![TensorShape::new(out_dims)])
}

fn infer_flatten(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    let input = inputs
        .first()
        .ok_or_else(|| OnnxError::ShapeMismatch("no input".into()))?;
    let axis = attrs.get("axis").and_then(|v| v.as_int().ok()).unwrap_or(1);
    let a = if axis < 0 {
        (input.rank() as i64 + axis) as usize
    } else {
        axis as usize
    };

    let d0 = input.dims[..a]
        .iter()
        .try_fold(1usize, |acc, d| d.map(|v| acc * v));
    let d1 = input.dims[a..]
        .iter()
        .try_fold(1usize, |acc, d| d.map(|v| acc * v));

    Ok(vec![TensorShape::new(vec![d0, d1])])
}

fn infer_concat(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    if inputs.is_empty() {
        return Err(OnnxError::ShapeMismatch("Concat: no inputs".into()));
    }
    let axis = attrs.get("axis").and_then(|v| v.as_int().ok()).unwrap_or(0);
    let rank = inputs[0].rank();
    let a = if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    };

    let mut out_dims = inputs[0].dims.clone();
    let concat_dim: Option<usize> = inputs
        .iter()
        .map(|s| s.dims[a])
        .try_fold(0usize, |acc, d| d.map(|v| acc + v));
    out_dims[a] = concat_dim;

    Ok(vec![TensorShape::new(out_dims)])
}

fn infer_split(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    let input = inputs
        .first()
        .ok_or_else(|| OnnxError::ShapeMismatch("no input".into()))?;
    let axis = attrs.get("axis").and_then(|v| v.as_int().ok()).unwrap_or(0);
    let a = if axis < 0 {
        (input.rank() as i64 + axis) as usize
    } else {
        axis as usize
    };

    let num_outputs = if let Some(AttributeValue::Ints(splits)) = attrs.get("split") {
        splits
            .iter()
            .map(|&s| {
                let mut shape = input.dims.clone();
                shape[a] = Some(s as usize);
                TensorShape::new(shape)
            })
            .collect()
    } else {
        let n = attrs
            .get("num_outputs")
            .and_then(|v| v.as_int().ok())
            .unwrap_or(2) as usize;
        (0..n)
            .map(|_| {
                let mut shape = input.dims.clone();
                shape[a] = input.dims[a].map(|d| d / n);
                TensorShape::new(shape)
            })
            .collect()
    };

    Ok(num_outputs)
}

fn infer_gather(
    inputs: &[&TensorShape],
    attrs: &HashMap<String, AttributeValue>,
) -> OnnxResult<Vec<TensorShape>> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch("Gather needs 2 inputs".into()));
    }
    let data = inputs[0];
    let indices = inputs[1];
    let axis = attrs.get("axis").and_then(|v| v.as_int().ok()).unwrap_or(0);
    let a = if axis < 0 {
        (data.rank() as i64 + axis) as usize
    } else {
        axis as usize
    };

    let mut out_dims = data.dims[..a].to_vec();
    out_dims.extend_from_slice(&indices.dims);
    out_dims.extend_from_slice(&data.dims[a + 1..]);
    Ok(vec![TensorShape::new(out_dims)])
}

fn infer_expand(inputs: &[&TensorShape]) -> OnnxResult<Vec<TensorShape>> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch("Expand needs 2 inputs".into()));
    }
    // Without concrete shape values, output is dynamic with same rank
    let rank = inputs[0]
        .rank()
        .max(inputs[1].dims.first().and_then(|d| *d).unwrap_or(0));
    Ok(vec![TensorShape::new(vec![None; rank])])
}

fn infer_constant(attrs: &HashMap<String, AttributeValue>) -> OnnxResult<Vec<TensorShape>> {
    if let Some(AttributeValue::Tensor(t)) = attrs.get("value") {
        return Ok(vec![TensorShape::fixed(t.shape.clone())]);
    }
    if attrs.contains_key("value_float") || attrs.contains_key("value_int") {
        return Ok(vec![TensorShape::fixed(vec![])]);
    }
    if let Some(AttributeValue::Floats(v)) = attrs.get("value_floats") {
        return Ok(vec![TensorShape::fixed(vec![v.len()])]);
    }
    if let Some(AttributeValue::Ints(v)) = attrs.get("value_ints") {
        return Ok(vec![TensorShape::fixed(vec![v.len()])]);
    }
    Ok(vec![TensorShape::new(vec![])])
}

// ─── Spatial dimension helpers ──────────────────────────────

fn get_spatial_ints(
    attrs: &HashMap<String, AttributeValue>,
    name: &str,
    default_len: usize,
) -> Vec<Option<usize>> {
    attrs
        .get(name)
        .and_then(|v| v.as_ints().ok())
        .map(|v| v.iter().map(|&i| Some(i as usize)).collect())
        .unwrap_or_else(|| {
            let default_val = if name == "pads" { 0 } else { 1 };
            vec![Some(default_val); default_len]
        })
}

fn compute_conv_dim(
    input: Option<usize>,
    kernel: Option<usize>,
    pad_begin: Option<usize>,
    pad_end: Option<usize>,
    dilation: Option<usize>,
    stride: Option<usize>,
) -> Option<usize> {
    let i = input?;
    let k = kernel?;
    let pb = pad_begin.unwrap_or(0);
    let pe = pad_end.unwrap_or(0);
    let d = dilation.unwrap_or(1);
    let s = stride.unwrap_or(1);
    let effective_k = d * (k - 1) + 1;
    Some((i + pb + pe - effective_k) / s + 1)
}

fn compute_conv_transpose_dim(
    input: Option<usize>,
    kernel: Option<usize>,
    pad_begin: Option<usize>,
    pad_end: Option<usize>,
    dilation: Option<usize>,
    stride: Option<usize>,
) -> Option<usize> {
    let i = input?;
    let k = kernel?;
    let pb = pad_begin.unwrap_or(0);
    let pe = pad_end.unwrap_or(0);
    let d = dilation.unwrap_or(1);
    let s = stride.unwrap_or(1);
    Some(s * (i - 1) + d * (k - 1) + 1 - pb - pe)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed(dims: &[usize]) -> TensorShape {
        TensorShape::fixed(dims.to_vec())
    }

    #[test]
    fn test_elementwise_broadcast_shape() {
        let a = fixed(&[3, 4]);
        let b = fixed(&[1, 4]);
        let r = ShapeInference::infer("Add", &[&a, &b], &HashMap::new()).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_unary_passthrough() {
        let a = fixed(&[2, 3, 4]);
        let r = ShapeInference::infer("Relu", &[&a], &HashMap::new()).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn test_matmul_shape() {
        let a = fixed(&[2, 3]);
        let b = fixed(&[3, 4]);
        let r = ShapeInference::infer("MatMul", &[&a, &b], &HashMap::new()).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![2, 4]);
    }

    #[test]
    fn test_matmul_batch_shape() {
        let a = fixed(&[5, 2, 3]);
        let b = fixed(&[5, 3, 4]);
        let r = ShapeInference::infer("MatMul", &[&a, &b], &HashMap::new()).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![5, 2, 4]);
    }

    #[test]
    fn test_conv_shape() {
        let x = fixed(&[1, 3, 32, 32]);
        let w = fixed(&[16, 3, 3, 3]);
        let r = ShapeInference::infer("Conv", &[&x, &w], &HashMap::new()).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![1, 16, 30, 30]);
    }

    #[test]
    fn test_conv_with_padding() {
        let x = fixed(&[1, 3, 32, 32]);
        let w = fixed(&[16, 3, 3, 3]);
        let mut attrs = HashMap::new();
        attrs.insert("pads".into(), AttributeValue::Ints(vec![1, 1, 1, 1]));
        let r = ShapeInference::infer("Conv", &[&x, &w], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![1, 16, 32, 32]);
    }

    #[test]
    fn test_pool_shape() {
        let x = fixed(&[1, 3, 32, 32]);
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".into(), AttributeValue::Ints(vec![2, 2]));
        attrs.insert("strides".into(), AttributeValue::Ints(vec![2, 2]));
        let r = ShapeInference::infer("MaxPool", &[&x], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![1, 3, 16, 16]);
    }

    #[test]
    fn test_global_pool_shape() {
        let x = fixed(&[1, 512, 7, 7]);
        let r = ShapeInference::infer("GlobalAveragePool", &[&x], &HashMap::new()).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![1, 512, 1, 1]);
    }

    #[test]
    fn test_flatten_shape() {
        let x = fixed(&[2, 3, 4, 5]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".into(), AttributeValue::Int(2));
        let r = ShapeInference::infer("Flatten", &[&x], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![6, 20]);
    }

    #[test]
    fn test_transpose_shape() {
        let x = fixed(&[2, 3, 4]);
        let mut attrs = HashMap::new();
        attrs.insert("perm".into(), AttributeValue::Ints(vec![2, 0, 1]));
        let r = ShapeInference::infer("Transpose", &[&x], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![4, 2, 3]);
    }

    #[test]
    fn test_concat_shape() {
        let a = fixed(&[2, 3]);
        let b = fixed(&[2, 4]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".into(), AttributeValue::Int(1));
        let r = ShapeInference::infer("Concat", &[&a, &b], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![2, 7]);
    }

    #[test]
    fn test_reduction_shape() {
        let x = fixed(&[3, 4, 5]);
        let mut attrs = HashMap::new();
        attrs.insert("axes".into(), AttributeValue::Ints(vec![1]));
        attrs.insert("keepdims".into(), AttributeValue::Int(1));
        let r = ShapeInference::infer("ReduceSum", &[&x], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![3, 1, 5]);
    }

    #[test]
    fn test_reduction_no_keepdims() {
        let x = fixed(&[3, 4, 5]);
        let mut attrs = HashMap::new();
        attrs.insert("axes".into(), AttributeValue::Ints(vec![1]));
        attrs.insert("keepdims".into(), AttributeValue::Int(0));
        let r = ShapeInference::infer("ReduceSum", &[&x], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![3, 5]);
    }

    #[test]
    fn test_gemm_shape() {
        let a = fixed(&[4, 3]);
        let b = fixed(&[3, 5]);
        let r = ShapeInference::infer("Gemm", &[&a, &b], &HashMap::new()).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![4, 5]);
    }

    #[test]
    fn test_gemm_trans_shape() {
        let a = fixed(&[3, 4]);
        let b = fixed(&[3, 5]);
        let mut attrs = HashMap::new();
        attrs.insert("transA".into(), AttributeValue::Int(1));
        let r = ShapeInference::infer("Gemm", &[&a, &b], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![4, 5]);
    }

    #[test]
    fn test_shape_inference_graph() {
        let graph = Graph {
            name: "test".into(),
            nodes: vec![Node {
                op_type: "Relu".into(),
                name: "relu".into(),
                inputs: vec!["X".into()],
                outputs: vec!["Y".into()],
                attributes: HashMap::new(),
            }],
            inputs: vec![TensorInfo {
                name: "X".into(),
                dtype: DataType::Float32,
                shape: TensorShape::fixed(vec![2, 3]),
            }],
            outputs: vec![TensorInfo {
                name: "Y".into(),
                dtype: DataType::Float32,
                shape: TensorShape::fixed(vec![2, 3]),
            }],
            initializers: HashMap::new(),
        };

        let shapes = ShapeInference::infer_graph(&graph).unwrap();
        assert_eq!(
            shapes.get("Y").map(|s| s.to_concrete().ok()),
            Some(Some(vec![2, 3]))
        );
    }

    #[test]
    fn test_dynamic_broadcast() {
        let a = TensorShape::new(vec![None, Some(4)]);
        let b = TensorShape::new(vec![Some(3), Some(1)]);
        let r = ShapeInference::infer("Add", &[&a, &b], &HashMap::new()).unwrap();
        // First dim dynamic, second dim 4
        assert_eq!(r[0].dims, vec![None, Some(4)]);
    }

    #[test]
    fn test_constant_shape() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "value".into(),
            AttributeValue::Tensor(OnnxTensor::from_f32(&[1.0, 2.0, 3.0], vec![3])),
        );
        let r = ShapeInference::infer("Constant", &[], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![3]);
    }

    #[test]
    fn test_shape_op_inference() {
        let x = fixed(&[2, 3, 4]);
        let r = ShapeInference::infer("Shape", &[&x], &HashMap::new()).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![3]);
    }

    #[test]
    fn test_size_op_inference() {
        let x = fixed(&[2, 3, 4]);
        let r = ShapeInference::infer("Size", &[&x], &HashMap::new()).unwrap();
        assert!(r[0].dims.is_empty()); // scalar
    }

    #[test]
    fn test_gather_shape() {
        let data = fixed(&[5, 4, 3]);
        let indices = fixed(&[2]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".into(), AttributeValue::Int(0));
        let r = ShapeInference::infer("Gather", &[&data, &indices], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![2, 4, 3]);
    }

    #[test]
    fn test_squeeze_shape() {
        let x = fixed(&[1, 3, 1, 4]);
        let mut attrs = HashMap::new();
        attrs.insert("axes".into(), AttributeValue::Ints(vec![0, 2]));
        let r = ShapeInference::infer("Squeeze", &[&x], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_unsqueeze_shape() {
        let x = fixed(&[3, 4]);
        let mut attrs = HashMap::new();
        attrs.insert("axes".into(), AttributeValue::Ints(vec![0, 3]));
        let r = ShapeInference::infer("Unsqueeze", &[&x], &attrs).unwrap();
        assert_eq!(r[0].to_concrete().unwrap(), vec![1, 3, 4, 1]);
    }
}
