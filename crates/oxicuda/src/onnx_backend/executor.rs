//! ONNX graph execution engine.
//!
//! Provides [`GraphExecutor`] which performs topological-sort-based
//! sequential execution of an ONNX graph.

use std::collections::{HashMap, VecDeque};

use super::ir::*;
use super::ops::OpRegistry;

/// Executes an ONNX graph sequentially in topologically sorted order.
pub struct GraphExecutor {
    graph: Graph,
    execution_order: Vec<usize>,
}

impl GraphExecutor {
    /// Build an executor from a graph. Performs topological sort up front.
    pub fn new(graph: Graph) -> OnnxResult<Self> {
        let execution_order = topological_sort(&graph)?;
        Ok(Self {
            graph,
            execution_order,
        })
    }

    /// Get the execution order (node indices).
    pub fn execution_order(&self) -> &[usize] {
        &self.execution_order
    }

    /// Get a reference to the underlying graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Execute the graph with the given input tensors.
    ///
    /// Returns a map of output tensor name → tensor.
    pub fn run(
        &self,
        inputs: HashMap<String, OnnxTensor>,
    ) -> OnnxResult<HashMap<String, OnnxTensor>> {
        let mut tensors: HashMap<String, OnnxTensor> = HashMap::new();

        // Load graph initializers (weights)
        for (name, tensor) in &self.graph.initializers {
            tensors.insert(name.clone(), tensor.clone());
        }

        // Load user-provided inputs
        for (name, tensor) in inputs {
            tensors.insert(name, tensor);
        }

        // Execute each node in topological order
        let registry = OpRegistry::new();
        for &idx in &self.execution_order {
            let node = &self.graph.nodes[idx];

            // Collect inputs (preserving positional indices; empty name → None)
            let input_refs: Vec<Option<&OnnxTensor>> = node
                .inputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        tensors.get(name.as_str())
                    }
                })
                .collect();

            let outputs = registry.execute(&node.op_type, &input_refs, &node.attributes)?;

            // Store outputs
            for (output_name, output_tensor) in node.outputs.iter().zip(outputs) {
                if !output_name.is_empty() {
                    tensors.insert(output_name.clone(), output_tensor);
                }
            }
        }

        // Extract graph outputs
        let mut result = HashMap::new();
        for output_info in &self.graph.outputs {
            if let Some(tensor) = tensors.remove(&output_info.name) {
                result.insert(output_info.name.clone(), tensor);
            }
        }

        Ok(result)
    }
}

/// Topological sort of graph nodes using Kahn's algorithm.
fn topological_sort(graph: &Graph) -> OnnxResult<Vec<usize>> {
    let n = graph.nodes.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Build tensor→producer mapping
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        for output in &node.outputs {
            if !output.is_empty() {
                producer.insert(output.as_str(), i);
            }
        }
    }

    // Build adjacency list and in-degree counts
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    let mut in_degree = vec![0u32; n];

    for (i, node) in graph.nodes.iter().enumerate() {
        for input in &node.inputs {
            if let Some(&j) = producer.get(input.as_str()) {
                if j != i {
                    adj[j].push(i);
                    in_degree[i] += 1;
                }
            }
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(node_idx) = queue.pop_front() {
        order.push(node_idx);
        for &next in &adj[node_idx] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                queue.push_back(next);
            }
        }
    }

    if order.len() != n {
        return Err(OnnxError::InvalidGraph(format!(
            "cycle detected: sorted {} of {} nodes",
            order.len(),
            n
        )));
    }

    Ok(order)
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
            shape: TensorShape::fixed(vec![2]),
        }
    }

    #[test]
    fn test_topological_sort_linear() {
        // A -> Relu -> B -> Relu -> C
        let graph = Graph {
            name: "linear".into(),
            nodes: vec![
                make_node("Relu", "relu0", &["A"], &["B"]),
                make_node("Relu", "relu1", &["B"], &["C"]),
            ],
            inputs: vec![make_info("A")],
            outputs: vec![make_info("C")],
            initializers: HashMap::new(),
        };
        let order = topological_sort(&graph).unwrap();
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn test_topological_sort_diamond() {
        // X -> [Add1, Add2] -> Add3 -> Y
        let graph = Graph {
            name: "diamond".into(),
            nodes: vec![
                make_node("Relu", "left", &["X"], &["L"]),
                make_node("Neg", "right", &["X"], &["R"]),
                make_node("Add", "merge", &["L", "R"], &["Y"]),
            ],
            inputs: vec![make_info("X")],
            outputs: vec![make_info("Y")],
            initializers: HashMap::new(),
        };
        let order = topological_sort(&graph).unwrap();
        // left and right can be in any order, but merge must come last
        assert_eq!(*order.last().unwrap_or(&999), 2);
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_topological_sort_cycle() {
        let graph = Graph {
            name: "cycle".into(),
            nodes: vec![
                make_node("Relu", "a", &["B"], &["A"]),
                make_node("Relu", "b", &["A"], &["B"]),
            ],
            inputs: vec![],
            outputs: vec![],
            initializers: HashMap::new(),
        };
        assert!(topological_sort(&graph).is_err());
    }

    #[test]
    fn test_executor_linear_chain() {
        // X -> Relu -> Y
        let graph = Graph {
            name: "test".into(),
            nodes: vec![make_node("Relu", "relu0", &["X"], &["Y"])],
            inputs: vec![make_info("X")],
            outputs: vec![make_info("Y")],
            initializers: HashMap::new(),
        };

        let exec = GraphExecutor::new(graph).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("X".into(), OnnxTensor::from_f32(&[-1.0, 2.0], vec![2]));

        let result = exec.run(inputs).unwrap();
        let y = result.get("Y").unwrap();
        assert_eq!(y.as_f32().unwrap(), vec![0.0, 2.0]);
    }

    #[test]
    fn test_executor_diamond() {
        // X -> Relu -> L
        // X -> Neg  -> R
        // L + R -> Y
        let graph = Graph {
            name: "diamond".into(),
            nodes: vec![
                make_node("Relu", "relu", &["X"], &["L"]),
                make_node("Neg", "neg", &["X"], &["R"]),
                make_node("Add", "add", &["L", "R"], &["Y"]),
            ],
            inputs: vec![TensorInfo {
                name: "X".into(),
                dtype: DataType::Float32,
                shape: TensorShape::fixed(vec![3]),
            }],
            outputs: vec![TensorInfo {
                name: "Y".into(),
                dtype: DataType::Float32,
                shape: TensorShape::fixed(vec![3]),
            }],
            initializers: HashMap::new(),
        };

        let exec = GraphExecutor::new(graph).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("X".into(), OnnxTensor::from_f32(&[-1.0, 2.0, 0.0], vec![3]));

        let result = exec.run(inputs).unwrap();
        let y = result.get("Y").unwrap();
        // relu([-1,2,0]) = [0,2,0], neg([-1,2,0]) = [1,-2,0]
        // add = [1, 0, 0]
        assert_eq!(y.as_f32().unwrap(), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_executor_skip_connection() {
        // X -> Relu -> T1
        // T1 + X -> Y   (skip connection)
        let graph = Graph {
            name: "skip".into(),
            nodes: vec![
                make_node("Relu", "relu", &["X"], &["T1"]),
                make_node("Add", "add", &["T1", "X"], &["Y"]),
            ],
            inputs: vec![TensorInfo {
                name: "X".into(),
                dtype: DataType::Float32,
                shape: TensorShape::fixed(vec![3]),
            }],
            outputs: vec![TensorInfo {
                name: "Y".into(),
                dtype: DataType::Float32,
                shape: TensorShape::fixed(vec![3]),
            }],
            initializers: HashMap::new(),
        };

        let exec = GraphExecutor::new(graph).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("X".into(), OnnxTensor::from_f32(&[-2.0, 3.0, 1.0], vec![3]));

        let result = exec.run(inputs).unwrap();
        let y = result.get("Y").unwrap();
        // relu([-2,3,1]) = [0,3,1], add([0,3,1], [-2,3,1]) = [-2,6,2]
        assert_eq!(y.as_f32().unwrap(), vec![-2.0, 6.0, 2.0]);
    }

    #[test]
    fn test_executor_with_initializer() {
        // Initializer (weight) + input -> Add -> Y
        let mut initializers = HashMap::new();
        initializers.insert("W".into(), OnnxTensor::from_f32(&[10.0, 20.0], vec![2]));

        let graph = Graph {
            name: "with_init".into(),
            nodes: vec![make_node("Add", "add", &["X", "W"], &["Y"])],
            inputs: vec![make_info("X")],
            outputs: vec![make_info("Y")],
            initializers,
        };

        let exec = GraphExecutor::new(graph).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("X".into(), OnnxTensor::from_f32(&[1.0, 2.0], vec![2]));

        let result = exec.run(inputs).unwrap();
        let y = result.get("Y").unwrap();
        assert_eq!(y.as_f32().unwrap(), vec![11.0, 22.0]);
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph {
            name: "empty".into(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            initializers: HashMap::new(),
        };
        let exec = GraphExecutor::new(graph).unwrap();
        let result = exec.run(HashMap::new()).unwrap();
        assert!(result.is_empty());
    }
}
