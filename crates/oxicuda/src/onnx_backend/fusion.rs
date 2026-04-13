//! Operator fusion passes for ONNX graphs.
//!
//! Detects common patterns and replaces them with fused operators:
//! - Conv + BatchNorm → FusedConvBn
//! - Conv + BatchNorm + Relu → FusedConvBnRelu
//! - MatMul + Add → FusedBiasedMatMul
//! - Mul + Add → FusedFMA (fused multiply-add)
//! - Transpose + MatMul → transposed MatMul

use std::collections::HashMap;

use super::ir::*;

/// Result of applying fusion passes.
#[derive(Debug, Clone, Default)]
pub struct FusionResult {
    /// Number of fusions applied.
    pub fusions_applied: usize,
    /// Names of fused operator groups.
    pub fused_ops: Vec<String>,
}

impl FusionResult {
    /// Merge another fusion result into this one.
    pub fn merge(&mut self, other: FusionResult) {
        self.fusions_applied += other.fusions_applied;
        self.fused_ops.extend(other.fused_ops);
    }
}

/// Applies fusion passes to an ONNX graph.
pub struct FusionPass;

impl FusionPass {
    /// Apply all fusion patterns to the graph (in-place).
    pub fn apply(graph: &mut Graph) -> FusionResult {
        let mut result = FusionResult::default();
        // Order matters: Conv+BN+Relu before Conv+BN
        result.merge(Self::fuse_conv_bn_relu(graph));
        result.merge(Self::fuse_conv_bn(graph));
        result.merge(Self::fuse_matmul_add(graph));
        result.merge(Self::fuse_mul_add(graph));
        result.merge(Self::fuse_transpose_matmul(graph));
        result
    }

    /// Fuse Conv + BatchNormalization → FusedConvBn.
    fn fuse_conv_bn(graph: &mut Graph) -> FusionResult {
        let mut result = FusionResult::default();
        let patterns = find_two_node_pattern(graph, "Conv", "BatchNormalization");

        for (conv_idx, bn_idx) in patterns.into_iter().rev() {
            if let Some(fused) = fuse_conv_bn_nodes(graph, conv_idx, bn_idx) {
                remove_and_replace(graph, &[conv_idx, bn_idx], fused);
                result.fusions_applied += 1;
                result.fused_ops.push("Conv+BN → FusedConvBn".into());
            }
        }
        result
    }

    /// Fuse Conv + BatchNormalization + Relu → FusedConvBnRelu.
    fn fuse_conv_bn_relu(graph: &mut Graph) -> FusionResult {
        let mut result = FusionResult::default();
        let patterns = find_three_node_pattern(graph, "Conv", "BatchNormalization", "Relu");

        for (conv_idx, bn_idx, relu_idx) in patterns.into_iter().rev() {
            if let Some(fused) = fuse_conv_bn_relu_nodes(graph, conv_idx, bn_idx, relu_idx) {
                remove_and_replace(graph, &[conv_idx, bn_idx, relu_idx], fused);
                result.fusions_applied += 1;
                result
                    .fused_ops
                    .push("Conv+BN+Relu → FusedConvBnRelu".into());
            }
        }
        result
    }

    /// Fuse MatMul + Add → FusedBiasedMatMul.
    fn fuse_matmul_add(graph: &mut Graph) -> FusionResult {
        let mut result = FusionResult::default();
        let patterns = find_two_node_pattern(graph, "MatMul", "Add");

        for (mm_idx, add_idx) in patterns.into_iter().rev() {
            if let Some(fused) = fuse_matmul_add_nodes(graph, mm_idx, add_idx) {
                remove_and_replace(graph, &[mm_idx, add_idx], fused);
                result.fusions_applied += 1;
                result
                    .fused_ops
                    .push("MatMul+Add → FusedBiasedMatMul".into());
            }
        }
        result
    }

    /// Fuse Mul + Add → FusedFMA.
    fn fuse_mul_add(graph: &mut Graph) -> FusionResult {
        let mut result = FusionResult::default();
        let patterns = find_two_node_pattern(graph, "Mul", "Add");

        for (mul_idx, add_idx) in patterns.into_iter().rev() {
            if let Some(fused) = fuse_mul_add_nodes(graph, mul_idx, add_idx) {
                remove_and_replace(graph, &[mul_idx, add_idx], fused);
                result.fusions_applied += 1;
                result.fused_ops.push("Mul+Add → FusedFMA".into());
            }
        }
        result
    }

    /// Fuse Transpose + MatMul → transposed MatMul (Gemm with transA).
    fn fuse_transpose_matmul(graph: &mut Graph) -> FusionResult {
        let mut result = FusionResult::default();
        let patterns = find_two_node_pattern(graph, "Transpose", "MatMul");

        for (trans_idx, mm_idx) in patterns.into_iter().rev() {
            if let Some(fused) = fuse_transpose_matmul_nodes(graph, trans_idx, mm_idx) {
                remove_and_replace(graph, &[trans_idx, mm_idx], fused);
                result.fusions_applied += 1;
                result
                    .fused_ops
                    .push("Transpose+MatMul → Gemm(transA)".into());
            }
        }
        result
    }
}

// ─── Pattern matching helpers ───────────────────────────────

/// Find pairs of (A, B) where A's output is B's input and both are single-consumer.
fn find_two_node_pattern(graph: &Graph, op_a: &str, op_b: &str) -> Vec<(usize, usize)> {
    let consumer_count = build_consumer_count(graph);
    let mut pairs = Vec::new();

    for (i, node_a) in graph.nodes.iter().enumerate() {
        if node_a.op_type != op_a {
            continue;
        }
        let a_output = match node_a.outputs.first() {
            Some(o) if !o.is_empty() => o.as_str(),
            _ => continue,
        };

        // Must have exactly one consumer
        if consumer_count.get(a_output).copied().unwrap_or(0) != 1 {
            continue;
        }

        for (j, node_b) in graph.nodes.iter().enumerate() {
            if j == i || node_b.op_type != op_b {
                continue;
            }
            if node_b.inputs.first().map(String::as_str) == Some(a_output) {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// Find triples (A, B, C) where A→B→C are chained.
fn find_three_node_pattern(
    graph: &Graph,
    op_a: &str,
    op_b: &str,
    op_c: &str,
) -> Vec<(usize, usize, usize)> {
    let consumer_count = build_consumer_count(graph);
    let mut triples = Vec::new();

    for (i, node_a) in graph.nodes.iter().enumerate() {
        if node_a.op_type != op_a {
            continue;
        }
        let a_out = match node_a.outputs.first() {
            Some(o) if !o.is_empty() => o.as_str(),
            _ => continue,
        };
        if consumer_count.get(a_out).copied().unwrap_or(0) != 1 {
            continue;
        }

        for (j, node_b) in graph.nodes.iter().enumerate() {
            if j == i || node_b.op_type != op_b {
                continue;
            }
            if node_b.inputs.first().map(String::as_str) != Some(a_out) {
                continue;
            }
            let b_out = match node_b.outputs.first() {
                Some(o) if !o.is_empty() => o.as_str(),
                _ => continue,
            };
            if consumer_count.get(b_out).copied().unwrap_or(0) != 1 {
                continue;
            }

            for (k, node_c) in graph.nodes.iter().enumerate() {
                if k == i || k == j || node_c.op_type != op_c {
                    continue;
                }
                if node_c.inputs.first().map(String::as_str) == Some(b_out) {
                    triples.push((i, j, k));
                }
            }
        }
    }
    triples
}

fn build_consumer_count(graph: &Graph) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for node in &graph.nodes {
        for input in &node.inputs {
            if !input.is_empty() {
                *counts.entry(input.clone()).or_insert(0) += 1;
            }
        }
    }
    // Graph outputs count as consumers
    for out in &graph.outputs {
        *counts.entry(out.name.clone()).or_insert(0) += 1;
    }
    counts
}

// ─── Node fusion constructors ───────────────────────────────

fn fuse_conv_bn_nodes(graph: &Graph, conv_idx: usize, bn_idx: usize) -> Option<Node> {
    let conv = &graph.nodes[conv_idx];
    let bn = &graph.nodes[bn_idx];

    // Conv inputs: X, W, B(opt)
    // BN inputs: <conv_out>, scale, B, mean, var
    let conv_input = conv.inputs.first()?.clone();
    let conv_weight = conv.inputs.get(1)?.clone();
    let conv_bias = conv.inputs.get(2).cloned().unwrap_or_default();
    let bn_output = bn.outputs.first()?.clone();

    // Merge attributes from Conv
    let mut attrs = conv.attributes.clone();
    // Carry BN scale/bias/mean/var as special attributes
    for (key, val) in &bn.attributes {
        attrs.insert(format!("bn_{key}"), val.clone());
    }

    // Build fused node: inputs = [X, W, B, scale, bn_bias, mean, var]
    let mut inputs = vec![conv_input, conv_weight, conv_bias];
    inputs.extend(bn.inputs.iter().skip(1).cloned());

    Some(Node {
        op_type: "FusedConvBn".into(),
        name: format!("{}_fused_{}", conv.name, bn.name),
        inputs,
        outputs: vec![bn_output],
        attributes: attrs,
    })
}

fn fuse_conv_bn_relu_nodes(
    graph: &Graph,
    conv_idx: usize,
    bn_idx: usize,
    relu_idx: usize,
) -> Option<Node> {
    let conv = &graph.nodes[conv_idx];
    let bn = &graph.nodes[bn_idx];
    let relu = &graph.nodes[relu_idx];

    let conv_input = conv.inputs.first()?.clone();
    let conv_weight = conv.inputs.get(1)?.clone();
    let conv_bias = conv.inputs.get(2).cloned().unwrap_or_default();
    let relu_output = relu.outputs.first()?.clone();

    let mut attrs = conv.attributes.clone();
    for (key, val) in &bn.attributes {
        attrs.insert(format!("bn_{key}"), val.clone());
    }
    attrs.insert("fused_relu".into(), AttributeValue::Int(1));

    let mut inputs = vec![conv_input, conv_weight, conv_bias];
    inputs.extend(bn.inputs.iter().skip(1).cloned());

    Some(Node {
        op_type: "FusedConvBnRelu".into(),
        name: format!("{}_fused_{}_relu", conv.name, bn.name),
        inputs,
        outputs: vec![relu_output],
        attributes: attrs,
    })
}

fn fuse_matmul_add_nodes(graph: &Graph, mm_idx: usize, add_idx: usize) -> Option<Node> {
    let mm = &graph.nodes[mm_idx];
    let add = &graph.nodes[add_idx];

    let a = mm.inputs.first()?.clone();
    let b = mm.inputs.get(1)?.clone();
    let add_output = add.outputs.first()?.clone();

    // Find the bias: the Add input that isn't the MatMul output
    let mm_out = mm.outputs.first()?;
    let bias = add
        .inputs
        .iter()
        .find(|inp| inp.as_str() != mm_out.as_str())?
        .clone();

    let mut attrs = HashMap::new();
    attrs.insert("alpha".into(), AttributeValue::Float(1.0));
    attrs.insert("beta".into(), AttributeValue::Float(1.0));

    Some(Node {
        op_type: "FusedBiasedMatMul".into(),
        name: format!("{}_fused_{}", mm.name, add.name),
        inputs: vec![a, b, bias],
        outputs: vec![add_output],
        attributes: attrs,
    })
}

fn fuse_mul_add_nodes(graph: &Graph, mul_idx: usize, add_idx: usize) -> Option<Node> {
    let mul = &graph.nodes[mul_idx];
    let add = &graph.nodes[add_idx];

    let a = mul.inputs.first()?.clone();
    let b = mul.inputs.get(1)?.clone();
    let add_output = add.outputs.first()?.clone();

    let mul_out = mul.outputs.first()?;
    let c = add
        .inputs
        .iter()
        .find(|inp| inp.as_str() != mul_out.as_str())?
        .clone();

    Some(Node {
        op_type: "FusedFMA".into(),
        name: format!("{}_fused_{}", mul.name, add.name),
        inputs: vec![a, b, c],
        outputs: vec![add_output],
        attributes: HashMap::new(),
    })
}

fn fuse_transpose_matmul_nodes(graph: &Graph, trans_idx: usize, mm_idx: usize) -> Option<Node> {
    let trans = &graph.nodes[trans_idx];
    let mm = &graph.nodes[mm_idx];

    // Transpose input → MatMul's left (A) input
    let trans_input = trans.inputs.first()?.clone();
    let trans_output = trans.outputs.first()?;
    let mm_output = mm.outputs.first()?.clone();

    // Check which MatMul input is the transposed one
    let (a, b, trans_side) = if mm.inputs.first().map(String::as_str) == Some(trans_output.as_str())
    {
        let other = mm.inputs.get(1)?.clone();
        (trans_input, other, "A")
    } else if mm.inputs.get(1).map(String::as_str) == Some(trans_output.as_str()) {
        let other = mm.inputs.first()?.clone();
        (other, trans_input, "B")
    } else {
        return None;
    };

    let mut attrs = HashMap::new();
    attrs.insert("alpha".into(), AttributeValue::Float(1.0));
    attrs.insert("beta".into(), AttributeValue::Float(0.0));
    if trans_side == "A" {
        attrs.insert("transA".into(), AttributeValue::Int(1));
    } else {
        attrs.insert("transB".into(), AttributeValue::Int(1));
    }

    Some(Node {
        op_type: "Gemm".into(),
        name: format!("{}_fused_{}", trans.name, mm.name),
        inputs: vec![a, b],
        outputs: vec![mm_output],
        attributes: attrs,
    })
}

/// Remove old nodes and insert the fused node.
fn remove_and_replace(graph: &mut Graph, remove_indices: &[usize], fused: Node) {
    // Remove in reverse order to preserve indices
    let mut sorted = remove_indices.to_vec();
    sorted.sort_unstable();
    for &idx in sorted.iter().rev() {
        if idx < graph.nodes.len() {
            graph.nodes.remove(idx);
        }
    }
    graph.nodes.push(fused);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(op: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            op_type: op.into(),
            name: name.into(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attributes: HashMap::new(),
        }
    }

    fn make_info(name: &str) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dtype: DataType::Float32,
            shape: TensorShape::fixed(vec![1]),
        }
    }

    #[test]
    fn test_conv_bn_fusion() {
        let mut graph = Graph {
            name: "test".into(),
            nodes: vec![
                make_node("Conv", "conv", &["X", "W"], &["conv_out"]),
                make_node(
                    "BatchNormalization",
                    "bn",
                    &["conv_out", "scale", "bias", "mean", "var"],
                    &["Y"],
                ),
            ],
            inputs: vec![make_info("X")],
            outputs: vec![make_info("Y")],
            initializers: HashMap::new(),
        };

        let result = FusionPass::apply(&mut graph);
        assert_eq!(result.fusions_applied, 1);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "FusedConvBn");
    }

    #[test]
    fn test_conv_bn_relu_fusion() {
        let mut graph = Graph {
            name: "test".into(),
            nodes: vec![
                make_node("Conv", "conv", &["X", "W"], &["conv_out"]),
                make_node(
                    "BatchNormalization",
                    "bn",
                    &["conv_out", "s", "b", "m", "v"],
                    &["bn_out"],
                ),
                make_node("Relu", "relu", &["bn_out"], &["Y"]),
            ],
            inputs: vec![make_info("X")],
            outputs: vec![make_info("Y")],
            initializers: HashMap::new(),
        };

        let result = FusionPass::apply(&mut graph);
        assert_eq!(result.fusions_applied, 1);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "FusedConvBnRelu");
    }

    #[test]
    fn test_matmul_add_fusion() {
        let mut graph = Graph {
            name: "test".into(),
            nodes: vec![
                make_node("MatMul", "mm", &["A", "B"], &["mm_out"]),
                make_node("Add", "add", &["mm_out", "bias"], &["Y"]),
            ],
            inputs: vec![make_info("A")],
            outputs: vec![make_info("Y")],
            initializers: HashMap::new(),
        };

        let result = FusionPass::apply(&mut graph);
        assert_eq!(result.fusions_applied, 1);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "FusedBiasedMatMul");
        assert!(graph.nodes[0].inputs.contains(&"bias".to_string()));
    }

    #[test]
    fn test_mul_add_fma_fusion() {
        let mut graph = Graph {
            name: "test".into(),
            nodes: vec![
                make_node("Mul", "mul", &["A", "B"], &["mul_out"]),
                make_node("Add", "add", &["mul_out", "C"], &["Y"]),
            ],
            inputs: vec![make_info("A")],
            outputs: vec![make_info("Y")],
            initializers: HashMap::new(),
        };

        let result = FusionPass::apply(&mut graph);
        assert_eq!(result.fusions_applied, 1);
        assert_eq!(graph.nodes[0].op_type, "FusedFMA");
    }

    #[test]
    fn test_transpose_matmul_fusion() {
        let mut graph = Graph {
            name: "test".into(),
            nodes: vec![
                make_node("Transpose", "trans", &["A"], &["At"]),
                make_node("MatMul", "mm", &["At", "B"], &["Y"]),
            ],
            inputs: vec![make_info("A")],
            outputs: vec![make_info("Y")],
            initializers: HashMap::new(),
        };

        let result = FusionPass::apply(&mut graph);
        assert_eq!(result.fusions_applied, 1);
        assert_eq!(graph.nodes[0].op_type, "Gemm");
        assert_eq!(
            graph.nodes[0].attributes.get("transA"),
            Some(&AttributeValue::Int(1))
        );
    }

    #[test]
    fn test_no_fusion_multi_consumer() {
        // Conv output used by both BN and another node → no fusion
        let mut graph = Graph {
            name: "test".into(),
            nodes: vec![
                make_node("Conv", "conv", &["X", "W"], &["conv_out"]),
                make_node(
                    "BatchNormalization",
                    "bn",
                    &["conv_out", "s", "b", "m", "v"],
                    &["Y1"],
                ),
                make_node("Relu", "relu", &["conv_out"], &["Y2"]),
            ],
            inputs: vec![make_info("X")],
            outputs: vec![make_info("Y1"), make_info("Y2")],
            initializers: HashMap::new(),
        };

        let result = FusionPass::apply(&mut graph);
        assert_eq!(result.fusions_applied, 0);
        assert_eq!(graph.nodes.len(), 3);
    }

    #[test]
    fn test_multiple_fusions() {
        // Two independent chains: Conv+BN and Mul+Add
        let mut graph = Graph {
            name: "test".into(),
            nodes: vec![
                make_node("Conv", "conv", &["X", "W"], &["conv_out"]),
                make_node(
                    "BatchNormalization",
                    "bn",
                    &["conv_out", "s", "b", "m", "v"],
                    &["Y1"],
                ),
                make_node("Mul", "mul", &["A", "B"], &["mul_out"]),
                make_node("Add", "add", &["mul_out", "C"], &["Y2"]),
            ],
            inputs: vec![make_info("X"), make_info("A")],
            outputs: vec![make_info("Y1"), make_info("Y2")],
            initializers: HashMap::new(),
        };

        let result = FusionPass::apply(&mut graph);
        assert_eq!(result.fusions_applied, 2);
        assert_eq!(graph.nodes.len(), 2);
    }
}
