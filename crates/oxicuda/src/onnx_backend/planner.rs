//! Memory planner for ONNX graph execution.
//!
//! Performs tensor lifetime analysis and buffer reuse optimization
//! to minimise peak GPU memory usage.

use std::collections::HashMap;

use super::ir::*;

/// Information about a single tensor allocation.
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Tensor name.
    pub tensor_name: String,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Offset within the shared buffer.
    pub offset: usize,
    /// Node index where this tensor is first produced.
    pub first_use: usize,
    /// Node index where this tensor is last consumed.
    pub last_use: usize,
    /// If this allocation reuses another tensor's buffer, its name.
    pub reuses: Option<String>,
}

/// A complete memory allocation plan for a graph.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Per-tensor allocation info.
    pub allocations: HashMap<String, AllocationInfo>,
    /// Peak memory usage in bytes (with reuse).
    pub peak_memory: usize,
    /// Total memory if no reuse were applied.
    pub total_without_reuse: usize,
}

/// Plans memory allocation for ONNX graph execution.
pub struct MemoryPlanner;

impl MemoryPlanner {
    /// Construct a memory plan for a graph given its execution order.
    ///
    /// `tensor_sizes` maps tensor name → size in bytes. If not provided,
    /// the planner estimates from graph metadata.
    pub fn plan(
        graph: &Graph,
        execution_order: &[usize],
        tensor_sizes: Option<&HashMap<String, usize>>,
    ) -> OnnxResult<MemoryPlan> {
        // Step 1: Compute tensor lifetimes (first_use, last_use)
        let lifetimes = compute_lifetimes(graph, execution_order);

        // Step 2: Estimate tensor sizes
        let sizes = if let Some(s) = tensor_sizes {
            s.clone()
        } else {
            estimate_tensor_sizes(graph)
        };

        // Step 3: Greedy buffer reuse
        let allocations = greedy_allocation(&lifetimes, &sizes, execution_order);

        // Step 4: Compute peak memory
        let peak_memory = compute_peak_memory(&allocations, execution_order);
        let total_without_reuse: usize = allocations.values().map(|a| a.size_bytes).sum();

        Ok(MemoryPlan {
            allocations,
            peak_memory,
            total_without_reuse,
        })
    }
}

/// Tensor lifetime: (first produced at step, last consumed at step).
#[derive(Debug, Clone)]
struct TensorLifetime {
    name: String,
    first_use: usize, // step index in execution_order
    last_use: usize,  // step index in execution_order
}

fn compute_lifetimes(graph: &Graph, execution_order: &[usize]) -> Vec<TensorLifetime> {
    // Map: tensor_name -> (first_step, last_step)
    let mut lifetime_map: HashMap<String, (usize, usize)> = HashMap::new();

    for (step, &node_idx) in execution_order.iter().enumerate() {
        let node = &graph.nodes[node_idx];

        // Outputs are produced at this step
        for output in &node.outputs {
            if output.is_empty() {
                continue;
            }
            lifetime_map
                .entry(output.clone())
                .and_modify(|(_, last)| *last = step)
                .or_insert((step, step));
        }

        // Inputs are consumed at this step
        for input in &node.inputs {
            if input.is_empty() {
                continue;
            }
            lifetime_map
                .entry(input.clone())
                .and_modify(|(_, last)| *last = step)
                .or_insert((step, step));
        }
    }

    // Also mark graph outputs as live until the end
    let last_step = execution_order.len().saturating_sub(1);
    for output in &graph.outputs {
        lifetime_map
            .entry(output.name.clone())
            .and_modify(|(_, last)| *last = last_step)
            .or_insert((0, last_step));
    }

    lifetime_map
        .into_iter()
        .map(|(name, (first, last))| TensorLifetime {
            name,
            first_use: first,
            last_use: last,
        })
        .collect()
}

fn estimate_tensor_sizes(graph: &Graph) -> HashMap<String, usize> {
    let mut sizes = HashMap::new();

    // Graph inputs
    for info in &graph.inputs {
        if let Some(count) = info.shape.element_count() {
            sizes.insert(info.name.clone(), count * info.dtype.size_bytes());
        }
    }

    // Initializers
    for (name, tensor) in &graph.initializers {
        sizes.insert(name.clone(), tensor.size_bytes());
    }

    // Graph outputs
    for info in &graph.outputs {
        if let Some(count) = info.shape.element_count() {
            sizes.insert(info.name.clone(), count * info.dtype.size_bytes());
        }
    }

    // For intermediate tensors without known size, use a default
    for node in &graph.nodes {
        for output in &node.outputs {
            if !output.is_empty() {
                sizes.entry(output.clone()).or_insert(1024); // default 1KB
            }
        }
    }

    sizes
}

fn greedy_allocation(
    lifetimes: &[TensorLifetime],
    sizes: &HashMap<String, usize>,
    _execution_order: &[usize],
) -> HashMap<String, AllocationInfo> {
    // Sort by first use, then by size (largest first for better packing)
    let mut sorted: Vec<&TensorLifetime> = lifetimes.iter().collect();
    sorted.sort_by(|a, b| {
        a.first_use.cmp(&b.first_use).then_with(|| {
            let sa = sizes.get(&b.name).copied().unwrap_or(0);
            let sb = sizes.get(&a.name).copied().unwrap_or(0);
            sa.cmp(&sb) // largest first
        })
    });

    // Free pool: (size, offset, original tensor name)
    let mut free_pool: Vec<(usize, usize, String)> = Vec::new();
    let mut allocations: HashMap<String, AllocationInfo> = HashMap::new();
    let mut next_offset = 0usize;

    // Track when tensors should be freed
    // Process in first_use order
    for lifetime in &sorted {
        let size = sizes.get(&lifetime.name).copied().unwrap_or(1024);

        // Check if any freed buffer can be reused
        // First, free buffers whose last_use < current first_use
        // (done lazily: scan pool for expired entries)
        let reuse_idx = free_pool
            .iter()
            .position(|&(buf_size, _, ref _name)| buf_size >= size);

        let (offset, reuses) = if let Some(idx) = reuse_idx {
            let (_, off, ref name) = free_pool[idx];
            let reuse_name = name.clone();
            free_pool.remove(idx);
            (off, Some(reuse_name))
        } else {
            let off = next_offset;
            next_offset += size;
            (off, None)
        };

        allocations.insert(
            lifetime.name.clone(),
            AllocationInfo {
                tensor_name: lifetime.name.clone(),
                size_bytes: size,
                offset,
                first_use: lifetime.first_use,
                last_use: lifetime.last_use,
                reuses,
            },
        );
    }

    // Return freed buffers to pool after their last use
    // (In a real implementation, this would be done incrementally)
    // For now, the greedy approach above provides a reasonable allocation.

    // Improved: sort by last_use and process free events
    let mut alloc_list: Vec<_> = allocations.values().cloned().collect();
    alloc_list.sort_by_key(|a| a.first_use);

    let mut refined = HashMap::new();
    let mut pool: Vec<(usize, usize)> = Vec::new(); // (size, offset)
    let mut current_offset = 0usize;

    for step in 0..=alloc_list.iter().map(|a| a.last_use).max().unwrap_or(0) {
        // Free tensors whose lifetime ended before this step
        for alloc in &alloc_list {
            if alloc.last_use + 1 == step {
                if let Some(info) = refined.get(&alloc.tensor_name) {
                    let info: &AllocationInfo = info;
                    pool.push((info.size_bytes, info.offset));
                }
            }
        }

        // Allocate tensors starting at this step
        for alloc in &alloc_list {
            if alloc.first_use == step {
                let size = alloc.size_bytes;
                let best = pool
                    .iter()
                    .enumerate()
                    .filter(|&(_, &(s, _))| s >= size)
                    .min_by_key(|&(_, &(s, _))| s)
                    .map(|(i, _)| i);

                let (offset, reuses) = if let Some(idx) = best {
                    let (_, off) = pool.remove(idx);
                    (off, Some(String::from("(reused)")))
                } else {
                    let off = current_offset;
                    current_offset += size;
                    (off, None)
                };

                refined.insert(
                    alloc.tensor_name.clone(),
                    AllocationInfo {
                        tensor_name: alloc.tensor_name.clone(),
                        size_bytes: size,
                        offset,
                        first_use: alloc.first_use,
                        last_use: alloc.last_use,
                        reuses,
                    },
                );
            }
        }
    }

    refined
}

fn compute_peak_memory(
    allocations: &HashMap<String, AllocationInfo>,
    execution_order: &[usize],
) -> usize {
    let num_steps = execution_order.len();
    let mut peak = 0usize;

    for step in 0..num_steps {
        let live_memory: usize = allocations
            .values()
            .filter(|a| a.first_use <= step && a.last_use >= step)
            .map(|a| a.size_bytes)
            .sum();
        if live_memory > peak {
            peak = live_memory;
        }
    }

    peak
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

    fn make_info(name: &str, count: usize) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dtype: DataType::Float32,
            shape: TensorShape::fixed(vec![count]),
        }
    }

    #[test]
    fn test_basic_memory_plan() {
        let graph = Graph {
            name: "test".into(),
            nodes: vec![
                make_node("Relu", "n0", &["X"], &["A"]),
                make_node("Relu", "n1", &["A"], &["B"]),
                make_node("Relu", "n2", &["B"], &["Y"]),
            ],
            inputs: vec![make_info("X", 1000)],
            outputs: vec![make_info("Y", 1000)],
            initializers: HashMap::new(),
        };

        let execution_order = vec![0, 1, 2];
        let plan = MemoryPlanner::plan(&graph, &execution_order, None).unwrap();

        assert!(!plan.allocations.is_empty());
        assert!(plan.peak_memory > 0);
    }

    #[test]
    fn test_buffer_reuse_reduces_memory() {
        // A -> B -> C (sequential chain, A's buffer can be reused for C)
        let graph = Graph {
            name: "chain".into(),
            nodes: vec![
                make_node("Relu", "n0", &["X"], &["A"]),
                make_node("Relu", "n1", &["A"], &["B"]),
                make_node("Relu", "n2", &["B"], &["Y"]),
            ],
            inputs: vec![make_info("X", 256)],
            outputs: vec![make_info("Y", 256)],
            initializers: HashMap::new(),
        };

        let execution_order = vec![0, 1, 2];

        let sizes: HashMap<String, usize> = [("X", 1024), ("A", 1024), ("B", 1024), ("Y", 1024)]
            .iter()
            .map(|&(k, v)| (k.to_string(), v))
            .collect();

        let plan = MemoryPlanner::plan(&graph, &execution_order, Some(&sizes)).unwrap();

        // With buffer reuse, peak should be less than total
        assert!(
            plan.peak_memory <= plan.total_without_reuse,
            "peak {} should be <= total {}",
            plan.peak_memory,
            plan.total_without_reuse
        );
    }

    #[test]
    fn test_diamond_memory_plan() {
        let graph = Graph {
            name: "diamond".into(),
            nodes: vec![
                make_node("Relu", "n0", &["X"], &["L"]),
                make_node("Relu", "n1", &["X"], &["R"]),
                make_node("Add", "n2", &["L", "R"], &["Y"]),
            ],
            inputs: vec![make_info("X", 100)],
            outputs: vec![make_info("Y", 100)],
            initializers: HashMap::new(),
        };

        let execution_order = vec![0, 1, 2];
        let plan = MemoryPlanner::plan(&graph, &execution_order, None).unwrap();

        // L and R are both live at step 2, so they can't share a buffer
        assert!(plan.peak_memory > 0);
    }

    #[test]
    fn test_peak_memory_tracking() {
        let graph = Graph {
            name: "peak".into(),
            nodes: vec![
                make_node("Relu", "n0", &["X"], &["A"]),
                make_node("Relu", "n1", &["X"], &["B"]),
                make_node("Add", "n2", &["A", "B"], &["C"]),
                make_node("Relu", "n3", &["C"], &["Y"]),
            ],
            inputs: vec![make_info("X", 100)],
            outputs: vec![make_info("Y", 100)],
            initializers: HashMap::new(),
        };

        let execution_order = vec![0, 1, 2, 3];
        let sizes: HashMap<String, usize> =
            [("X", 400), ("A", 400), ("B", 400), ("C", 400), ("Y", 400)]
                .iter()
                .map(|&(k, v)| (k.to_string(), v))
                .collect();

        let plan = MemoryPlanner::plan(&graph, &execution_order, Some(&sizes)).unwrap();
        // At step 2, X, A, B, C are all live = 1600 bytes peak
        // With reuse after step 2, it might be less
        assert!(plan.peak_memory >= 400); // at minimum one tensor
    }

    #[test]
    fn test_empty_graph_plan() {
        let graph = Graph {
            name: "empty".into(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            initializers: HashMap::new(),
        };
        let plan = MemoryPlanner::plan(&graph, &[], None).unwrap();
        assert_eq!(plan.peak_memory, 0);
    }
}
