//! CUDA Graph capture backend.
//!
//! Converts an [`ExecutionPlan`] into an `oxicuda_driver::graph::Graph`,
//! which can be instantiated and launched with minimal CPU overhead via the
//! CUDA driver's graph API.
//!
//! # Mapping
//!
//! | `PlanStep`          | `driver::graph::GraphNode`         |
//! |---------------------|------------------------------------|
//! | `KernelLaunch`      | `GraphNode::KernelLaunch`          |
//! | `Memcpy`            | `GraphNode::Memcpy`                |
//! | `Memset`            | `GraphNode::Memset`                |
//! | `Barrier`           | `GraphNode::Empty`                 |
//! | `EventRecord/Wait`  | `GraphNode::Empty` (sync barrier)  |
//! | `HostCallback`      | `GraphNode::Empty` (placeholder)   |
//!
//! The dependency edges from the original plan are preserved: each step
//! depends on all steps that produced an event it is waiting for.
//!
//! # Limitations
//!
//! * Stream-parallel execution is not yet mapped to separate CUDA stream
//!   arguments — all nodes are enqueued on the same graph-internal stream.
//!   Proper multi-stream CUDA graph support requires the `cuGraphAddNode`
//!   API with `cuGraphNodeParams`, which is future work.
//! * `EventRecord`/`EventWait` steps collapse into `Empty` nodes with a
//!   dependency edge, which enforces ordering without the overhead of true
//!   event signalling within a CUDA graph.

use oxicuda_driver::graph::{Graph, MemcpyDirection};

use crate::error::GraphResult;
use crate::executor::plan::{ExecutionPlan, PlanStep};
use crate::node::MemcpyDir;

/// Converts a `MemcpyDir` to the driver's `MemcpyDirection`.
fn to_driver_dir(dir: &MemcpyDir) -> MemcpyDirection {
    match dir {
        MemcpyDir::HostToDevice => MemcpyDirection::HostToDevice,
        MemcpyDir::DeviceToHost => MemcpyDirection::DeviceToHost,
        MemcpyDir::DeviceToDevice | MemcpyDir::PeerToPeer { .. } => MemcpyDirection::DeviceToDevice,
    }
}

/// Captures an [`ExecutionPlan`] into a `driver::graph::Graph`.
///
/// Returns the resulting graph, ready to be instantiated via
/// [`Graph::instantiate`].
///
/// # Errors
///
/// Returns a [`GraphError`] if any step cannot be added to the graph.
///
/// [`GraphError`]: crate::error::GraphError
pub fn capture(plan: &ExecutionPlan) -> GraphResult<Graph> {
    let mut graph = Graph::new();

    // Map from plan step index → driver graph node index.
    let mut step_to_node: Vec<Option<usize>> = vec![None; plan.steps.len()];

    // Map from event_id → the step index that recorded it.
    let mut event_record_step: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();

    // First pass: add all nodes to the driver graph.
    for (i, step) in plan.steps.iter().enumerate() {
        let node_idx = match step {
            PlanStep::KernelLaunch {
                function_name,
                config,
                ..
            } => graph.add_kernel_node(
                function_name,
                config.grid,
                config.block,
                config.shared_mem_bytes,
            ),
            PlanStep::Memcpy {
                dir, size_bytes, ..
            } => graph.add_memcpy_node(to_driver_dir(dir), *size_bytes),
            PlanStep::Memset {
                size_bytes, value, ..
            } => graph.add_memset_node(*size_bytes, *value),
            PlanStep::EventRecord { event_id, .. } => {
                // Record which step emitted this event.
                event_record_step.insert(*event_id, i);
                graph.add_empty_node()
            }
            PlanStep::EventWait { .. } => graph.add_empty_node(),
            PlanStep::HostCallback { .. } | PlanStep::Barrier { .. } => graph.add_empty_node(),
        };
        step_to_node[i] = Some(node_idx);
    }

    // Second pass: add dependency edges.
    // Strategy: the plan steps are in topological submission order,
    // so for each step we add an edge from the immediately preceding
    // step on the same stream (sequential ordering within a stream)
    // and from the EventRecord step for each EventWait.

    // Track the most recent step on each stream.
    let mut last_on_stream: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();

    for (i, step) in plan.steps.iter().enumerate() {
        let sid = step.stream().0;
        let my_node = step_to_node[i].ok_or_else(|| {
            crate::error::GraphError::Internal(format!("step {i} node not populated"))
        })?;

        // Within-stream ordering: depend on the previous step on this stream.
        if let Some(&prev_step) = last_on_stream.get(&sid) {
            let prev_node = step_to_node[prev_step].ok_or_else(|| {
                crate::error::GraphError::Internal(format!("step {prev_step} node not populated"))
            })?;
            graph.add_dependency(prev_node, my_node).ok();
        }
        last_on_stream.insert(sid, i);

        // Cross-stream ordering: EventWait depends on the EventRecord.
        if let PlanStep::EventWait { event_id, .. } = step {
            if let Some(&record_step) = event_record_step.get(event_id) {
                let record_node = step_to_node[record_step].ok_or_else(|| {
                    crate::error::GraphError::Internal(format!(
                        "event record step {record_step} node not populated"
                    ))
                })?;
                graph.add_dependency(record_node, my_node).ok();
            }
        }
    }

    Ok(graph)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::executor::plan::ExecutionPlan;
    use crate::graph::ComputeGraph;
    use crate::node::MemcpyDir;

    fn simple_graph() -> ComputeGraph {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let up = b.add_memcpy("up", MemcpyDir::HostToDevice, 1024);
        let k = b.add_kernel("gemm", 4, 256, 0).fusible(false).finish();
        let dn = b.add_memcpy("dn", MemcpyDir::DeviceToHost, 1024);
        b.chain(&[up, k, dn]);
        b.build().unwrap()
    }

    #[test]
    fn cuda_graph_capture_no_panic() {
        let g = simple_graph();
        let plan = ExecutionPlan::build(&g, 1).unwrap();
        let dg = capture(&plan).unwrap();
        assert!(dg.node_count() > 0);
    }

    #[test]
    fn cuda_graph_node_count_matches_steps() {
        let g = simple_graph();
        let plan = ExecutionPlan::build(&g, 1).unwrap();
        let step_count = plan.steps.len();
        let dg = capture(&plan).unwrap();
        assert_eq!(dg.node_count(), step_count);
    }

    #[test]
    fn cuda_graph_has_dependencies() {
        // Linear chain → must have edges.
        let g = simple_graph();
        let plan = ExecutionPlan::build(&g, 1).unwrap();
        let dg = capture(&plan).unwrap();
        assert!(dg.dependency_count() > 0);
    }

    #[test]
    fn cuda_graph_is_dag_instantiatable() {
        let g = simple_graph();
        let plan = ExecutionPlan::build(&g, 1).unwrap();
        let dg = capture(&plan).unwrap();
        // The graph should be a valid DAG (instantiate runs topological sort).
        let exec = dg.instantiate();
        assert!(exec.is_ok(), "captured graph must be a valid DAG");
    }

    #[test]
    fn cuda_graph_event_sync_adds_edge() {
        // Manually build a plan with EventRecord → EventWait across streams.
        let steps = vec![
            PlanStep::KernelLaunch {
                nodes: vec![crate::node::NodeId(0)],
                function_name: "k0".into(),
                config: crate::node::KernelConfig::linear(1, 32, 0),
                stream: crate::node::StreamId(0),
            },
            PlanStep::EventRecord {
                event_id: 0,
                stream: crate::node::StreamId(0),
            },
            PlanStep::EventWait {
                event_id: 0,
                stream: crate::node::StreamId(1),
            },
            PlanStep::KernelLaunch {
                nodes: vec![crate::node::NodeId(1)],
                function_name: "k1".into(),
                config: crate::node::KernelConfig::linear(1, 32, 0),
                stream: crate::node::StreamId(1),
            },
        ];
        let plan = ExecutionPlan {
            steps,
            num_streams: 2,
            pool_bytes: 0,
            kernel_count_original: 2,
            kernel_count_fused: 2,
            event_count: 1,
        };
        let dg = capture(&plan).unwrap();
        // The EventWait node must depend on the EventRecord node.
        assert!(dg.dependency_count() >= 1);
        let exec = dg.instantiate();
        assert!(exec.is_ok());
    }

    #[test]
    fn cuda_graph_memset_node_recorded() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        b.add_memset("zero", 4096, 0x00);
        let g = b.build().unwrap();
        let plan = ExecutionPlan::build(&g, 1).unwrap();
        let dg = capture(&plan).unwrap();
        assert_eq!(dg.node_count(), 1);
        // Should be a Memset node in the driver graph.
        let nodes = dg.nodes();
        assert!(matches!(
            nodes[0],
            oxicuda_driver::graph::GraphNode::Memset { .. }
        ));
    }

    #[test]
    fn cuda_graph_barrier_becomes_empty_node() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        b.add_barrier("sync");
        let g = b.build().unwrap();
        let plan = ExecutionPlan::build(&g, 1).unwrap();
        let dg = capture(&plan).unwrap();
        let nodes = dg.nodes();
        // Barrier → Empty node.
        assert!(matches!(nodes[0], oxicuda_driver::graph::GraphNode::Empty));
    }
}
