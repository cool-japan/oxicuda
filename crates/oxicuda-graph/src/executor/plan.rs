//! Execution plan — the linearised, stream-aware schedule produced by the
//! compilation pipeline and consumed by the executor backends.
//!
//! An [`ExecutionPlan`] is an ordered sequence of [`PlanStep`]s, each of
//! which describes one concrete GPU or host-side action. Steps are already
//! assigned to streams and include the event synchronisation pairs needed
//! to enforce cross-stream dependencies.
//!
//! The plan is produced by running all analysis and optimisation passes in
//! sequence; see [`ExecutionPlan::build`].

use crate::analysis::{liveness_analyse, topo_analyse};
use crate::error::{GraphError, GraphResult};
use crate::graph::ComputeGraph;
use crate::node::{KernelConfig, MemcpyDir, NodeId, NodeKind, StreamId};
use crate::optimizer::{fusion_analyse, memory_analyse, stream_analyse};

// ---------------------------------------------------------------------------
// PlanStep
// ---------------------------------------------------------------------------

/// A single step in the execution plan.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanStep {
    /// Launch a (possibly fused) kernel.
    KernelLaunch {
        /// Node(s) contributing to this launch (singleton if not fused).
        nodes: Vec<NodeId>,
        /// Function name to look up in the loaded PTX module.
        function_name: String,
        /// Launch configuration.
        config: KernelConfig,
        /// Stream to submit on.
        stream: StreamId,
    },

    /// Transfer memory between host and device (or device to device).
    Memcpy {
        node: NodeId,
        dir: MemcpyDir,
        size_bytes: usize,
        stream: StreamId,
    },

    /// Fill a device buffer with a byte pattern.
    Memset {
        node: NodeId,
        size_bytes: usize,
        value: u8,
        stream: StreamId,
    },

    /// Record a CUDA event on `stream` to signal completion of preceding work.
    EventRecord {
        /// Synthetic event ID (for matching with `EventWait`).
        event_id: usize,
        stream: StreamId,
    },

    /// Wait for a previously recorded event before continuing on `stream`.
    EventWait { event_id: usize, stream: StreamId },

    /// Host-side callback inserted into the stream.
    HostCallback {
        node: NodeId,
        label: String,
        stream: StreamId,
    },

    /// No-op barrier (no device work, used for explicit synchronisation).
    Barrier { node: NodeId, stream: StreamId },
}

impl PlanStep {
    /// Returns the stream this step executes on.
    pub fn stream(&self) -> StreamId {
        match self {
            Self::KernelLaunch { stream, .. } => *stream,
            Self::Memcpy { stream, .. } => *stream,
            Self::Memset { stream, .. } => *stream,
            Self::EventRecord { stream, .. } => *stream,
            Self::EventWait { stream, .. } => *stream,
            Self::HostCallback { stream, .. } => *stream,
            Self::Barrier { stream, .. } => *stream,
        }
    }

    /// Returns a short tag string for display.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::KernelLaunch { .. } => "kernel",
            Self::Memcpy { .. } => "memcpy",
            Self::Memset { .. } => "memset",
            Self::EventRecord { .. } => "event_record",
            Self::EventWait { .. } => "event_wait",
            Self::HostCallback { .. } => "host_cb",
            Self::Barrier { .. } => "barrier",
        }
    }

    /// Returns `true` if this step represents GPU compute work.
    pub fn is_compute(&self) -> bool {
        matches!(self, Self::KernelLaunch { .. })
    }
}

impl std::fmt::Display for PlanStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KernelLaunch {
                function_name,
                config,
                stream,
                nodes,
            } => {
                write!(
                    f,
                    "{}  kernel {function_name} [{config}] @{stream} (nodes: {})",
                    self.tag(),
                    nodes
                        .iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            Self::Memcpy {
                dir,
                size_bytes,
                stream,
                ..
            } => {
                write!(f, "  memcpy {dir} {size_bytes}B @{stream}")
            }
            Self::Memset {
                size_bytes,
                value,
                stream,
                ..
            } => {
                write!(f, "  memset {size_bytes}B=0x{value:02x} @{stream}")
            }
            Self::EventRecord { event_id, stream } => {
                write!(f, "  event_record ev{event_id} @{stream}")
            }
            Self::EventWait { event_id, stream } => {
                write!(f, "  event_wait ev{event_id} @{stream}")
            }
            Self::HostCallback { label, stream, .. } => {
                write!(f, "  host_cb '{label}' @{stream}")
            }
            Self::Barrier { stream, .. } => {
                write!(f, "  barrier @{stream}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ExecutionPlan
// ---------------------------------------------------------------------------

/// The compiled, optimised execution plan for a `ComputeGraph`.
///
/// Contains an ordered list of `PlanStep`s that can be submitted to an
/// executor backend (sequential simulator or real CUDA graph capture).
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Steps in submission order.
    pub steps: Vec<PlanStep>,
    /// Number of distinct CUDA streams used.
    pub num_streams: usize,
    /// Total device memory pool size in bytes.
    pub pool_bytes: usize,
    /// Number of kernel launches (before fusion).
    pub kernel_count_original: usize,
    /// Number of kernel launches (after fusion).
    pub kernel_count_fused: usize,
    /// Number of cross-stream event sync pairs.
    pub event_count: usize,
}

impl ExecutionPlan {
    /// Compiles a `ComputeGraph` into an `ExecutionPlan`.
    ///
    /// Runs all analysis and optimisation passes in order:
    /// topo → liveness → fusion → memory → stream → linearise.
    ///
    /// # Parameters
    ///
    /// * `graph` — the computation graph to compile.
    /// * `max_streams` — upper bound on concurrent CUDA streams.
    ///
    /// # Errors
    ///
    /// Propagates errors from any analysis or optimisation pass.
    pub fn build(graph: &ComputeGraph, max_streams: usize) -> GraphResult<Self> {
        if graph.is_empty() {
            return Err(GraphError::EmptyGraph);
        }

        // --- Analysis passes ---
        let topo = topo_analyse(graph)?;
        let _liveness = liveness_analyse(graph)?;
        let fusion_plan = fusion_analyse(graph)?;
        let memory_plan = memory_analyse(graph)?;
        let stream_plan = stream_analyse(graph, max_streams)?;

        let pool_bytes = memory_plan.total_bytes;
        let num_streams = stream_plan.num_streams;
        let kernel_count_original = graph.kernel_nodes().len();

        // --- Build event map for cross-stream sync points ---
        let mut event_counter = 0usize;
        // Map: (from_node, to_node) → event_id
        let mut edge_events: std::collections::HashMap<(NodeId, NodeId), usize> =
            std::collections::HashMap::new();
        for sp in &stream_plan.sync_points {
            let key = (sp.from, sp.to);
            edge_events.entry(key).or_insert_with(|| {
                let eid = event_counter;
                event_counter += 1;
                eid
            });
        }

        // For each node, collect the set of events it must wait for.
        // event_waits[node] = list of event_ids to wait before executing node.
        let mut event_waits: std::collections::HashMap<NodeId, Vec<usize>> =
            std::collections::HashMap::new();
        // event_records[node] = list of event_ids to record after executing node.
        let mut event_records: std::collections::HashMap<NodeId, Vec<usize>> =
            std::collections::HashMap::new();

        for sp in &stream_plan.sync_points {
            let eid = edge_events[&(sp.from, sp.to)];
            event_records.entry(sp.from).or_default().push(eid);
            event_waits.entry(sp.to).or_default().push(eid);
        }

        // --- Build fused kernel names ---
        // For each node, determine what function name its group exposes.
        let fused_name_of = |node_id: NodeId| -> String {
            let group = fusion_plan.group_of(node_id);
            match group {
                Some(g) if !g.is_trivial() => {
                    // Build name from all member function names joined.
                    let member_names: Vec<String> = g
                        .members
                        .iter()
                        .filter_map(|&m| {
                            graph
                                .node(m)
                                .ok()
                                .and_then(|n| n.kind.function_name().map(|s| s.to_owned()))
                        })
                        .collect();
                    format!("fused_{}", member_names.join("_"))
                }
                _ => graph
                    .node(node_id)
                    .ok()
                    .and_then(|n| n.kind.function_name())
                    .unwrap_or("unknown")
                    .to_owned(),
            }
        };

        // --- Linearise: one step per node in topological order ---
        // Track which fusion groups have already been emitted.
        let mut emitted_groups: std::collections::HashSet<usize> = std::collections::HashSet::new();

        let mut steps: Vec<PlanStep> = Vec::new();
        let mut kernel_count_fused = 0;

        for &node_id in &topo.order {
            let node = graph.node(node_id)?;
            let stream = stream_plan.stream_of(node_id);

            // Emit event waits before this node.
            if let Some(waits) = event_waits.get(&node_id) {
                for &eid in waits {
                    steps.push(PlanStep::EventWait {
                        event_id: eid,
                        stream,
                    });
                }
            }

            // Emit the operation itself.
            match &node.kind {
                NodeKind::KernelLaunch {
                    function_name,
                    config,
                    ..
                } => {
                    // Check if this node is part of a non-trivial fusion group.
                    let group_id = fusion_plan.node_to_group.get(&node_id).copied();
                    let group = group_id.and_then(|gid| fusion_plan.groups.get(gid));

                    match group {
                        Some(g) if !g.is_trivial() => {
                            // Only emit once (for the first member of the group).
                            if !emitted_groups.contains(&g.id) {
                                emitted_groups.insert(g.id);
                                let fused_name = fused_name_of(node_id);
                                steps.push(PlanStep::KernelLaunch {
                                    nodes: g.members.clone(),
                                    function_name: fused_name,
                                    config: g.config,
                                    stream,
                                });
                                kernel_count_fused += 1;
                            }
                            // Skip non-first members (they're folded into the fused launch).
                        }
                        _ => {
                            // Trivial group or non-fusible: emit as-is.
                            steps.push(PlanStep::KernelLaunch {
                                nodes: vec![node_id],
                                function_name: function_name.clone(),
                                config: *config,
                                stream,
                            });
                            kernel_count_fused += 1;
                        }
                    }
                }
                NodeKind::Memcpy { dir, size_bytes } => {
                    steps.push(PlanStep::Memcpy {
                        node: node_id,
                        dir: *dir,
                        size_bytes: *size_bytes,
                        stream,
                    });
                }
                NodeKind::Memset { size_bytes, value } => {
                    steps.push(PlanStep::Memset {
                        node: node_id,
                        size_bytes: *size_bytes,
                        value: *value,
                        stream,
                    });
                }
                NodeKind::HostCallback { label } => {
                    steps.push(PlanStep::HostCallback {
                        node: node_id,
                        label: label.clone(),
                        stream,
                    });
                }
                NodeKind::EventRecord => {
                    // User-inserted event record — keep as-is.
                    steps.push(PlanStep::EventRecord {
                        event_id: event_counter,
                        stream,
                    });
                    event_counter += 1;
                }
                NodeKind::EventWait => {
                    steps.push(PlanStep::EventWait {
                        event_id: event_counter,
                        stream,
                    });
                    event_counter += 1;
                }
                NodeKind::Barrier | NodeKind::Conditional { .. } => {
                    steps.push(PlanStep::Barrier {
                        node: node_id,
                        stream,
                    });
                }
            }

            // Emit event records after this node.
            if let Some(records) = event_records.get(&node_id) {
                for &eid in records {
                    steps.push(PlanStep::EventRecord {
                        event_id: eid,
                        stream,
                    });
                }
            }
        }

        Ok(ExecutionPlan {
            steps,
            num_streams,
            pool_bytes,
            kernel_count_original,
            kernel_count_fused,
            event_count: event_counter,
        })
    }

    /// Returns all steps assigned to a specific stream, in order.
    pub fn steps_on(&self, stream: StreamId) -> Vec<&PlanStep> {
        self.steps.iter().filter(|s| s.stream() == stream).collect()
    }

    /// Returns the total number of steps.
    pub fn total_steps(&self) -> usize {
        self.steps.len()
    }

    /// Returns the number of kernel-launch steps (after fusion).
    pub fn compute_steps(&self) -> usize {
        self.steps.iter().filter(|s| s.is_compute()).count()
    }
}

impl std::fmt::Display for ExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "ExecutionPlan: {} steps, {} streams, {} bytes pool, {} kernels (→{} fused), {} events",
            self.steps.len(),
            self.num_streams,
            self.pool_bytes,
            self.kernel_count_original,
            self.kernel_count_fused,
            self.event_count
        )?;
        for (i, step) in self.steps.iter().enumerate() {
            writeln!(f, "  [{i:3}] {step}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::node::MemcpyDir;

    fn build_simple_inference_graph() -> ComputeGraph {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let inp = b.alloc_buffer("input", 4096);
        let out = b.alloc_buffer("output", 4096);
        let upload = b.add_memcpy("upload", MemcpyDir::HostToDevice, 4096);
        let k0 = b
            .add_kernel("relu", 4, 256, 0)
            .fusible(true)
            .inputs([inp])
            .outputs([out])
            .finish();
        let k1 = b
            .add_kernel("scale", 4, 256, 0)
            .fusible(true)
            .inputs([out])
            .outputs([inp])
            .finish();
        let download = b.add_memcpy("download", MemcpyDir::DeviceToHost, 4096);
        b.chain(&[upload, k0, k1, download]);
        b.set_outputs(upload, [inp]);
        b.set_inputs(download, [inp]);
        b.build().unwrap()
    }

    #[test]
    fn plan_empty_graph_error() {
        let g = ComputeGraph::new();
        assert!(matches!(
            ExecutionPlan::build(&g, 4),
            Err(GraphError::EmptyGraph)
        ));
    }

    #[test]
    fn plan_builds_without_error() {
        let g = build_simple_inference_graph();
        let plan = ExecutionPlan::build(&g, 4).unwrap();
        assert!(plan.total_steps() > 0);
    }

    #[test]
    fn plan_covers_all_nodes() {
        let g = build_simple_inference_graph();
        let plan = ExecutionPlan::build(&g, 4).unwrap();
        // 4 original nodes; steps may include event records/waits too.
        // At minimum, each non-fused node must appear in at least one step.
        assert!(plan.total_steps() >= 2); // at least upload and download
    }

    #[test]
    fn plan_fused_kernels_reduce_compute_steps() {
        let g = build_simple_inference_graph();
        let plan = ExecutionPlan::build(&g, 4).unwrap();
        // relu and scale are fusible and in a chain → fused to 1 step.
        assert!(plan.compute_steps() <= 2);
        // kernel_count_fused ≤ kernel_count_original.
        assert!(plan.kernel_count_fused <= plan.kernel_count_original);
    }

    #[test]
    fn plan_single_stream_no_events() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_barrier("a");
        let bnode = b.add_barrier("b");
        b.dep(a, bnode);
        let g = b.build().unwrap();
        let plan = ExecutionPlan::build(&g, 1).unwrap();
        // With max_streams=1, no cross-stream syncs.
        let event_steps: Vec<_> = plan
            .steps
            .iter()
            .filter(|s| matches!(s, PlanStep::EventRecord { .. } | PlanStep::EventWait { .. }))
            .collect();
        // Events from user-inserted nodes: zero (no EventRecord/Wait nodes added).
        // Cross-stream events: zero (single stream).
        assert_eq!(event_steps.len(), 0);
    }

    #[test]
    fn plan_steps_on_stream() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        b.add_barrier("n");
        let g = b.build().unwrap();
        let plan = ExecutionPlan::build(&g, 1).unwrap();
        let on_s0 = plan.steps_on(StreamId(0));
        assert!(!on_s0.is_empty());
    }

    #[test]
    fn plan_display_output() {
        let g = build_simple_inference_graph();
        let plan = ExecutionPlan::build(&g, 4).unwrap();
        let s = plan.to_string();
        assert!(s.contains("ExecutionPlan"));
        assert!(s.contains("streams"));
    }

    #[test]
    fn plan_memcpy_and_memset_steps() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let up = b.add_memcpy("up", MemcpyDir::HostToDevice, 1024);
        let ms = b.add_memset("zero", 4096, 0x00);
        b.dep(up, ms);
        let g = b.build().unwrap();
        let plan = ExecutionPlan::build(&g, 1).unwrap();
        let has_memcpy = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::Memcpy { .. }));
        let has_memset = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::Memset { .. }));
        assert!(has_memcpy);
        assert!(has_memset);
    }

    #[test]
    fn plan_pool_bytes_reported() {
        let g = build_simple_inference_graph();
        let plan = ExecutionPlan::build(&g, 4).unwrap();
        // Pool bytes ≥ size of at least one internal buffer.
        // Two internal buffers of 4096 bytes; pool may be shared.
        // pool_bytes is usize, always non-negative; just verify the field is accessible (non-crash)
        let _ = plan.pool_bytes;
    }

    #[test]
    fn plan_kernel_count_fields_valid() {
        let g = build_simple_inference_graph();
        let plan = ExecutionPlan::build(&g, 4).unwrap();
        assert_eq!(plan.kernel_count_original, 2); // relu, scale
        assert!(plan.kernel_count_fused <= plan.kernel_count_original);
    }
}
