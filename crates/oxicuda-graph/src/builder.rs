//! Ergonomic builder API for constructing `ComputeGraph`s.
//!
//! `GraphBuilder` provides a fluent interface for assembling a computation
//! graph, handling ID allocation and optional automatic data-flow edge
//! inference transparently.
//!
//! # Example
//!
//! ```rust
//! # use oxicuda_graph::builder::GraphBuilder;
//! # use oxicuda_graph::node::MemcpyDir;
//! let mut b = GraphBuilder::new();
//!
//! let upload   = b.add_memcpy("upload",   MemcpyDir::HostToDevice, 4096);
//! let weights  = b.alloc_buffer("weights",  4096);
//! let compute  = b.add_kernel("gemm", 32, 256, 0).fusible(false).finish();
//! let download = b.add_memcpy("download", MemcpyDir::DeviceToHost, 4096);
//!
//! b.dep(upload, compute).dep(compute, download);
//!
//! let graph = b.build().unwrap();
//! assert_eq!(graph.node_count(), 3);
//! ```

use crate::error::GraphResult;
use crate::graph::ComputeGraph;
use crate::node::{
    BufferDescriptor, BufferId, GraphNode, KernelConfig, MemcpyDir, NodeId, NodeKind, StreamId,
};

// ---------------------------------------------------------------------------
// KernelNodeBuilder — fluent kernel-node configuration
// ---------------------------------------------------------------------------

/// Fluent builder for a single kernel launch node.
///
/// Obtained via [`GraphBuilder::add_kernel`].
pub struct KernelNodeBuilder<'a> {
    parent: &'a mut GraphBuilder,
    name: String,
    function_name: String,
    config: KernelConfig,
    fusible: bool,
    inputs: Vec<BufferId>,
    outputs: Vec<BufferId>,
    stream: Option<StreamId>,
    cost: u64,
}

impl<'a> KernelNodeBuilder<'a> {
    /// Marks this kernel as fusible with adjacent element-wise kernels.
    #[must_use]
    pub fn fusible(mut self, v: bool) -> Self {
        self.fusible = v;
        self
    }

    /// Sets input buffer IDs for this kernel.
    #[must_use]
    pub fn inputs(mut self, ids: impl IntoIterator<Item = BufferId>) -> Self {
        self.inputs.extend(ids);
        self
    }

    /// Sets output buffer IDs for this kernel.
    #[must_use]
    pub fn outputs(mut self, ids: impl IntoIterator<Item = BufferId>) -> Self {
        self.outputs.extend(ids);
        self
    }

    /// Sets the preferred stream for this node.
    #[must_use]
    pub fn on_stream(mut self, s: StreamId) -> Self {
        self.stream = Some(s);
        self
    }

    /// Sets the cost hint for the scheduler.
    #[must_use]
    pub fn cost(mut self, c: u64) -> Self {
        self.cost = c;
        self
    }

    /// Finishes configuration and registers the node with the parent builder.
    ///
    /// Returns the allocated `NodeId`.
    pub fn finish(self) -> NodeId {
        let kind = NodeKind::KernelLaunch {
            function_name: self.function_name,
            config: self.config,
            fusible: self.fusible,
        };
        let mut node = GraphNode::new(NodeId(0), kind)
            .with_inputs(self.inputs)
            .with_outputs(self.outputs)
            .with_name(self.name)
            .with_cost(self.cost);
        if let Some(s) = self.stream {
            node = node.with_stream(s);
        }
        self.parent.graph.add_node(node)
    }
}

// ---------------------------------------------------------------------------
// GraphBuilder
// ---------------------------------------------------------------------------

/// Builds a [`ComputeGraph`] via a fluent API.
///
/// The builder accumulates nodes, buffers, and explicit dependency edges.
/// Calling [`build`](GraphBuilder::build) finalises the graph and optionally
/// infers data-flow edges from buffer read/write relationships.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    graph: ComputeGraph,
    /// Whether to automatically infer data-flow edges from buffer I/O.
    auto_infer_edges: bool,
}

impl GraphBuilder {
    /// Creates a new empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: ComputeGraph::new(),
            auto_infer_edges: true,
        }
    }

    /// Configures whether buffer-based data-flow edges are inferred on build.
    ///
    /// Default: `true`.
    #[must_use]
    pub fn with_auto_infer_edges(mut self, v: bool) -> Self {
        self.auto_infer_edges = v;
        self
    }

    // -----------------------------------------------------------------------
    // Buffer allocation
    // -----------------------------------------------------------------------

    /// Allocates a named device buffer and returns its `BufferId`.
    pub fn alloc_buffer(&mut self, name: &str, size_bytes: usize) -> BufferId {
        self.graph
            .add_buffer(BufferDescriptor::new(BufferId(0), size_bytes).with_name(name))
    }

    /// Allocates an external (caller-managed) buffer.
    pub fn alloc_external_buffer(&mut self, name: &str, size_bytes: usize) -> BufferId {
        self.graph.add_buffer(
            BufferDescriptor::new(BufferId(0), size_bytes)
                .with_name(name)
                .external(),
        )
    }

    // -----------------------------------------------------------------------
    // Node addition helpers
    // -----------------------------------------------------------------------

    /// Starts a fluent kernel-node configuration.
    ///
    /// Call `.finish()` on the returned builder to register the node.
    pub fn add_kernel(
        &mut self,
        name: &str,
        num_blocks: u32,
        threads_per_block: u32,
        shared_mem: u32,
    ) -> KernelNodeBuilder<'_> {
        KernelNodeBuilder {
            parent: self,
            name: name.to_owned(),
            function_name: name.to_owned(),
            config: KernelConfig::linear(num_blocks, threads_per_block, shared_mem),
            fusible: true,
            inputs: Vec::new(),
            outputs: Vec::new(),
            stream: None,
            cost: 1,
        }
    }

    /// Adds a kernel launch node with the given function name and 3-D grid/block.
    pub fn add_kernel_3d(
        &mut self,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) -> NodeId {
        let kind = NodeKind::KernelLaunch {
            function_name: name.to_owned(),
            config: KernelConfig {
                grid,
                block,
                shared_mem_bytes: shared_mem,
            },
            fusible: false,
        };
        self.graph
            .add_node(GraphNode::new(NodeId(0), kind).with_name(name))
    }

    /// Adds a host-to-device or device-to-host memcpy node.
    pub fn add_memcpy(&mut self, name: &str, dir: MemcpyDir, size_bytes: usize) -> NodeId {
        let kind = NodeKind::Memcpy { dir, size_bytes };
        self.graph
            .add_node(GraphNode::new(NodeId(0), kind).with_name(name))
    }

    /// Adds a memset node.
    pub fn add_memset(&mut self, name: &str, size_bytes: usize, value: u8) -> NodeId {
        let kind = NodeKind::Memset { size_bytes, value };
        self.graph
            .add_node(GraphNode::new(NodeId(0), kind).with_name(name))
    }

    /// Adds an event-record node.
    pub fn add_event_record(&mut self, name: &str) -> NodeId {
        self.graph
            .add_node(GraphNode::new(NodeId(0), NodeKind::EventRecord).with_name(name))
    }

    /// Adds an event-wait node.
    pub fn add_event_wait(&mut self, name: &str) -> NodeId {
        self.graph
            .add_node(GraphNode::new(NodeId(0), NodeKind::EventWait).with_name(name))
    }

    /// Adds a barrier (no-op synchronisation) node.
    pub fn add_barrier(&mut self, name: &str) -> NodeId {
        self.graph
            .add_node(GraphNode::new(NodeId(0), NodeKind::Barrier).with_name(name))
    }

    /// Adds a host-callback node.
    pub fn add_host_callback(&mut self, name: &str) -> NodeId {
        let kind = NodeKind::HostCallback {
            label: name.to_owned(),
        };
        self.graph
            .add_node(GraphNode::new(NodeId(0), kind).with_name(name))
    }

    /// Adds a raw `GraphNode` (full control over the node).
    pub fn add_raw(&mut self, node: GraphNode) -> NodeId {
        self.graph.add_node(node)
    }

    // -----------------------------------------------------------------------
    // Convenience: annotate existing nodes
    // -----------------------------------------------------------------------

    /// Attaches input buffers to an existing node.
    ///
    /// Returns `self` for chaining.
    pub fn set_inputs(
        &mut self,
        node: NodeId,
        inputs: impl IntoIterator<Item = BufferId>,
    ) -> &mut Self {
        if let Ok(n) = self.graph.node_mut(node) {
            n.inputs.extend(inputs);
        }
        self
    }

    /// Attaches output buffers to an existing node.
    pub fn set_outputs(
        &mut self,
        node: NodeId,
        outputs: impl IntoIterator<Item = BufferId>,
    ) -> &mut Self {
        if let Ok(n) = self.graph.node_mut(node) {
            n.outputs.extend(outputs);
        }
        self
    }

    // -----------------------------------------------------------------------
    // Dependency edges
    // -----------------------------------------------------------------------

    /// Adds a dependency edge `from → to`.
    ///
    /// Returns `&mut self` for chaining. On error, the builder records the
    /// error and `build()` will propagate it.
    pub fn dep(&mut self, from: NodeId, to: NodeId) -> &mut Self {
        // Errors are propagated lazily via build().
        let _ = self.graph.add_edge(from, to);
        self
    }

    /// Adds a linear dependency chain: `nodes[0] → nodes[1] → … → nodes[n-1]`.
    pub fn chain(&mut self, nodes: &[NodeId]) -> &mut Self {
        for w in nodes.windows(2) {
            self.dep(w[0], w[1]);
        }
        self
    }

    /// Adds "fan-in" edges: all `predecessors` must complete before `join`.
    pub fn fan_in(&mut self, predecessors: &[NodeId], join: NodeId) -> &mut Self {
        for &p in predecessors {
            self.dep(p, join);
        }
        self
    }

    /// Adds "fan-out" edges: `fork` must complete before all `successors` start.
    pub fn fan_out(&mut self, fork: NodeId, successors: &[NodeId]) -> &mut Self {
        for &s in successors {
            self.dep(fork, s);
        }
        self
    }

    // -----------------------------------------------------------------------
    // Finalise
    // -----------------------------------------------------------------------

    /// Finalises the builder and returns the assembled `ComputeGraph`.
    ///
    /// If `auto_infer_edges` is `true` (the default), data-flow dependency
    /// edges are inferred from buffer input/output annotations before
    /// returning.
    ///
    /// # Errors
    ///
    /// Propagates any `GraphError` accumulated during the build process
    /// (e.g., cycles introduced by data-flow edge inference).
    pub fn build(mut self) -> GraphResult<ComputeGraph> {
        if self.auto_infer_edges {
            self.graph.infer_data_edges()?;
        }
        Ok(self.graph)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::MemcpyDir;

    #[test]
    fn builder_empty_build() {
        let b = GraphBuilder::new();
        let g = b.build().unwrap();
        assert!(g.is_empty());
    }

    #[test]
    fn builder_alloc_buffer() {
        let mut b = GraphBuilder::new();
        let buf0 = b.alloc_buffer("input", 1024);
        let buf1 = b.alloc_buffer("output", 2048);
        assert_eq!(buf0, BufferId(0));
        assert_eq!(buf1, BufferId(1));
        let g = b.build().unwrap();
        assert_eq!(g.buffer_count(), 2);
        assert_eq!(g.buffer(buf0).unwrap().size_bytes, 1024);
    }

    #[test]
    fn builder_add_kernel_chain() {
        let mut b = GraphBuilder::new();
        let k0 = b.add_kernel("k0", 4, 256, 0).finish();
        let k1 = b.add_kernel("k1", 4, 256, 0).finish();
        let k2 = b.add_kernel("k2", 4, 256, 0).finish();
        b.chain(&[k0, k1, k2]);
        let g = b.build().unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
        assert!(g.is_reachable(k0, k2));
    }

    #[test]
    fn builder_kernel_fusible_flag() {
        let mut b = GraphBuilder::new();
        let id = b.add_kernel("custom", 1, 32, 0).fusible(false).finish();
        let g = b.build().unwrap();
        assert!(!g.node(id).unwrap().kind.is_fusible());
    }

    #[test]
    fn builder_add_memcpy() {
        let mut b = GraphBuilder::new();
        let up = b.add_memcpy("upload", MemcpyDir::HostToDevice, 4096);
        let g = b.build().unwrap();
        assert_eq!(g.node_count(), 1);
        let node = g.node(up).unwrap();
        assert!(node.kind.is_memory_op());
    }

    #[test]
    fn builder_add_memset() {
        let mut b = GraphBuilder::new();
        let ms = b.add_memset("zero", 8192, 0x00);
        let g = b.build().unwrap();
        let node = g.node(ms).unwrap();
        if let NodeKind::Memset { value, .. } = node.kind {
            assert_eq!(value, 0x00);
        } else {
            panic!("expected Memset");
        }
    }

    #[test]
    fn builder_add_barrier_and_event() {
        let mut b = GraphBuilder::new();
        let barrier = b.add_barrier("sync");
        let ev = b.add_event_record("ev0");
        let ew = b.add_event_wait("ew0");
        b.chain(&[barrier, ev, ew]);
        let g = b.build().unwrap();
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn builder_fan_in_fan_out() {
        let mut b = GraphBuilder::new();
        let src = b.add_barrier("src");
        let a = b.add_kernel("a", 1, 32, 0).finish();
        let c = b.add_kernel("c", 1, 32, 0).finish();
        let d = b.add_kernel("d", 1, 32, 0).finish();
        let sink = b.add_barrier("sink");
        b.fan_out(src, &[a, c, d]);
        b.fan_in(&[a, c, d], sink);
        let g = b.build().unwrap();
        assert_eq!(g.edge_count(), 6);
        assert!(g.is_reachable(src, sink));
        assert_eq!(g.sources(), vec![src]);
        assert_eq!(g.sinks(), vec![sink]);
    }

    #[test]
    fn builder_dep_returns_self_for_chaining() {
        let mut b = GraphBuilder::new();
        let a = b.add_barrier("a");
        let b_node = b.add_barrier("b");
        let c = b.add_barrier("c");
        b.dep(a, b_node).dep(b_node, c);
        let g = b.build().unwrap();
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn builder_auto_infer_edges_from_buffers() {
        let mut b = GraphBuilder::new();
        let buf = b.alloc_buffer("tmp", 512);
        let writer = b.add_barrier("writer");
        let reader = b.add_barrier("reader");
        b.set_outputs(writer, [buf]);
        b.set_inputs(reader, [buf]);
        let g = b.build().unwrap(); // auto_infer_edges=true
        assert!(g.is_reachable(writer, reader));
    }

    #[test]
    fn builder_no_auto_infer_edges_disabled() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let buf = b.alloc_buffer("tmp", 512);
        let writer = b.add_barrier("writer");
        let reader = b.add_barrier("reader");
        b.set_outputs(writer, [buf]);
        b.set_inputs(reader, [buf]);
        let g = b.build().unwrap();
        // No inferred edges, so no reachability unless explicitly added.
        assert!(!g.is_reachable(writer, reader));
    }

    #[test]
    fn builder_add_raw_node() {
        let mut b = GraphBuilder::new();
        let n = GraphNode::new(NodeId(0), NodeKind::Barrier)
            .with_name("raw_barrier")
            .with_cost(42);
        let id = b.add_raw(n);
        let g = b.build().unwrap();
        assert_eq!(g.node(id).unwrap().cost_hint, 42);
    }

    #[test]
    fn builder_add_host_callback() {
        let mut b = GraphBuilder::new();
        let cb = b.add_host_callback("sync_point");
        let g = b.build().unwrap();
        if let NodeKind::HostCallback { label } = &g.node(cb).unwrap().kind {
            assert_eq!(label, "sync_point");
        } else {
            panic!("expected HostCallback");
        }
    }

    #[test]
    fn builder_set_inputs_and_outputs() {
        let mut b = GraphBuilder::new();
        let buf0 = b.alloc_buffer("a", 1);
        let buf1 = b.alloc_buffer("b", 1);
        let node = b.add_barrier("n");
        b.set_inputs(node, [buf0]);
        b.set_outputs(node, [buf1]);
        let g = b.build().unwrap();
        let n = g.node(node).unwrap();
        assert!(n.inputs.contains(&buf0));
        assert!(n.outputs.contains(&buf1));
    }

    #[test]
    fn builder_add_kernel_with_3d_grid() {
        let mut b = GraphBuilder::new();
        let id = b.add_kernel_3d("conv2d", (4, 4, 1), (8, 8, 1), 0);
        let g = b.build().unwrap();
        if let NodeKind::KernelLaunch { config, .. } = g.node(id).unwrap().kind {
            assert_eq!(config.grid, (4, 4, 1));
            assert_eq!(config.block, (8, 8, 1));
        } else {
            panic!("expected KernelLaunch");
        }
    }

    #[test]
    fn builder_alloc_external_buffer() {
        let mut b = GraphBuilder::new();
        let id = b.alloc_external_buffer("model_weights", 65536);
        let g = b.build().unwrap();
        assert!(g.buffer(id).unwrap().external);
    }

    #[test]
    fn builder_kernel_with_buffers_and_cost() {
        let mut b = GraphBuilder::new();
        let inp = b.alloc_buffer("input", 4096);
        let out = b.alloc_buffer("output", 4096);
        let id = b
            .add_kernel("scale", 8, 128, 0)
            .inputs([inp])
            .outputs([out])
            .cost(100)
            .finish();
        let g = b.build().unwrap();
        let n = g.node(id).unwrap();
        assert_eq!(n.cost_hint, 100);
        assert!(n.inputs.contains(&inp));
        assert!(n.outputs.contains(&out));
    }
}
