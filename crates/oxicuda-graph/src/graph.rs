//! The `ComputeGraph` — a directed acyclic graph of GPU operations.
//!
//! `ComputeGraph` is the central data structure of `oxicuda-graph`. It
//! stores nodes (GPU operations), directed dependency edges, and buffer
//! descriptors. Analysis and optimisation passes operate on this structure
//! before it is lowered to an `ExecutionPlan` by the executor.
//!
//! # Design
//!
//! * Nodes are stored in a `Vec` indexed by a dense `u32` slot.
//! * Adjacency is maintained as two maps: `successors` and `predecessors`,
//!   each mapping `NodeId → Vec<NodeId>`. This makes both forward and
//!   backward traversals O(degree).
//! * Cycle detection happens eagerly on `add_edge` using DFS, so the
//!   invariant "this graph is a DAG" always holds after successful mutation.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{GraphError, GraphResult};
use crate::node::{BufferDescriptor, BufferId, GraphNode, NodeId};

// ---------------------------------------------------------------------------
// ComputeGraph
// ---------------------------------------------------------------------------

/// A directed acyclic graph (DAG) of GPU operations.
///
/// Nodes represent individual GPU operations ([`GraphNode`]). Directed edges
/// express execution-order dependencies: an edge `a → b` means `b` cannot
/// begin until `a` has completed. Buffers referenced by nodes are registered
/// separately via [`add_buffer`](ComputeGraph::add_buffer).
///
/// # Guarantees
///
/// * The graph is always a DAG — [`add_edge`](ComputeGraph::add_edge) returns
///   [`GraphError::CycleDetected`] if adding the edge would create a cycle.
/// * Node IDs are unique and stable: nodes are never moved or re-indexed.
/// * Buffer IDs are unique and stable.
#[derive(Debug, Clone)]
pub struct ComputeGraph {
    /// Nodes stored in insertion order (indexed by NodeId.0).
    nodes: Vec<GraphNode>,
    /// Successor adjacency: node → nodes that depend on it.
    successors: Vec<Vec<NodeId>>,
    /// Predecessor adjacency: node → nodes it depends on.
    predecessors: Vec<Vec<NodeId>>,
    /// Buffer metadata (indexed by BufferId.0).
    buffers: Vec<BufferDescriptor>,
    /// Next node id to allocate.
    next_node: u32,
    /// Next buffer id to allocate.
    next_buf: u32,
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeGraph {
    /// Creates an empty computation graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            successors: Vec::new(),
            predecessors: Vec::new(),
            buffers: Vec::new(),
            next_node: 0,
            next_buf: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Node management
    // -----------------------------------------------------------------------

    /// Adds a node to the graph and returns its assigned `NodeId`.
    ///
    /// The node's `id` field will be overwritten with the allocated ID.
    pub fn add_node(&mut self, mut node: GraphNode) -> NodeId {
        let id = NodeId(self.next_node);
        node.id = id;
        self.next_node += 1;
        self.nodes.push(node);
        self.successors.push(Vec::new());
        self.predecessors.push(Vec::new());
        id
    }

    /// Returns a reference to the node with the given ID, or an error.
    pub fn node(&self, id: NodeId) -> GraphResult<&GraphNode> {
        self.nodes
            .get(id.0 as usize)
            .ok_or(GraphError::NodeNotFound(id))
    }

    /// Returns a mutable reference to the node with the given ID, or an error.
    pub fn node_mut(&mut self, id: NodeId) -> GraphResult<&mut GraphNode> {
        self.nodes
            .get_mut(id.0 as usize)
            .ok_or(GraphError::NodeNotFound(id))
    }

    /// Returns a slice of all nodes in insertion order.
    #[inline]
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Returns the total number of nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // -----------------------------------------------------------------------
    // Buffer management
    // -----------------------------------------------------------------------

    /// Registers a buffer and returns its assigned `BufferId`.
    ///
    /// The buffer's `id` field will be overwritten with the allocated ID.
    pub fn add_buffer(&mut self, mut buf: BufferDescriptor) -> BufferId {
        let id = BufferId(self.next_buf);
        buf.id = id;
        self.next_buf += 1;
        self.buffers.push(buf);
        id
    }

    /// Returns a reference to the buffer descriptor, or an error.
    pub fn buffer(&self, id: BufferId) -> GraphResult<&BufferDescriptor> {
        self.buffers
            .get(id.0 as usize)
            .ok_or(GraphError::NodeNotFound(NodeId(id.0)))
    }

    /// Returns a slice of all buffer descriptors.
    #[inline]
    pub fn buffers(&self) -> &[BufferDescriptor] {
        &self.buffers
    }

    /// Returns the number of registered buffers.
    #[inline]
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    // -----------------------------------------------------------------------
    // Edge management
    // -----------------------------------------------------------------------

    /// Adds a directed dependency edge `from → to`.
    ///
    /// This means `to` will not begin execution until `from` has completed.
    ///
    /// # Errors
    ///
    /// * [`GraphError::NodeNotFound`] if either ID is invalid.
    /// * [`GraphError::CycleDetected`] if the edge would create a cycle.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) -> GraphResult<()> {
        let n = self.nodes.len();
        if from.0 as usize >= n {
            return Err(GraphError::NodeNotFound(from));
        }
        if to.0 as usize >= n {
            return Err(GraphError::NodeNotFound(to));
        }
        if from == to {
            return Err(GraphError::CycleDetected { from, to });
        }
        // Check whether adding from→to would create a cycle: it does iff
        // `from` is reachable from `to` in the current graph.
        if self.is_reachable(to, from) {
            return Err(GraphError::CycleDetected { from, to });
        }
        // Avoid duplicate edges.
        if !self.successors[from.0 as usize].contains(&to) {
            self.successors[from.0 as usize].push(to);
            self.predecessors[to.0 as usize].push(from);
        }
        Ok(())
    }

    /// Returns whether node `src` can reach node `dst` via directed edges.
    ///
    /// Uses iterative BFS to avoid stack overflow on deep graphs.
    pub fn is_reachable(&self, src: NodeId, dst: NodeId) -> bool {
        if src == dst {
            return true;
        }
        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        queue.push_back(src);
        visited[src.0 as usize] = true;
        while let Some(curr) = queue.pop_front() {
            for &next in &self.successors[curr.0 as usize] {
                if next == dst {
                    return true;
                }
                if !visited[next.0 as usize] {
                    visited[next.0 as usize] = true;
                    queue.push_back(next);
                }
            }
        }
        false
    }

    /// Returns successors of `id` (nodes that depend on it).
    pub fn successors(&self, id: NodeId) -> GraphResult<&[NodeId]> {
        if id.0 as usize >= self.nodes.len() {
            return Err(GraphError::NodeNotFound(id));
        }
        Ok(&self.successors[id.0 as usize])
    }

    /// Returns predecessors of `id` (nodes it depends on).
    pub fn predecessors(&self, id: NodeId) -> GraphResult<&[NodeId]> {
        if id.0 as usize >= self.nodes.len() {
            return Err(GraphError::NodeNotFound(id));
        }
        Ok(&self.predecessors[id.0 as usize])
    }

    /// Returns the total number of directed edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.successors.iter().map(|v| v.len()).sum()
    }

    /// Returns all edges as `(from, to)` pairs.
    pub fn edges(&self) -> Vec<(NodeId, NodeId)> {
        let mut edges = Vec::new();
        for (i, succs) in self.successors.iter().enumerate() {
            for &to in succs {
                edges.push((NodeId(i as u32), to));
            }
        }
        edges
    }

    // -----------------------------------------------------------------------
    // Source / sink queries
    // -----------------------------------------------------------------------

    /// Returns all nodes with no predecessors (entry points of the graph).
    pub fn sources(&self) -> Vec<NodeId> {
        self.predecessors
            .iter()
            .enumerate()
            .filter(|(_, preds)| preds.is_empty())
            .map(|(i, _)| NodeId(i as u32))
            .collect()
    }

    /// Returns all nodes with no successors (terminal nodes of the graph).
    pub fn sinks(&self) -> Vec<NodeId> {
        self.successors
            .iter()
            .enumerate()
            .filter(|(_, succs)| succs.is_empty())
            .map(|(i, _)| NodeId(i as u32))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Topological sort (Kahn's algorithm)
    // -----------------------------------------------------------------------

    /// Returns nodes in topological order (all predecessors before successors).
    ///
    /// Uses Kahn's BFS algorithm. Because the graph is guaranteed to be a DAG
    /// after successful `add_edge` calls, this will always succeed unless the
    /// graph is empty.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::EmptyGraph`] if there are no nodes.
    pub fn topological_order(&self) -> GraphResult<Vec<NodeId>> {
        if self.nodes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }
        let n = self.nodes.len();
        let mut in_degree: Vec<u32> = self.predecessors.iter().map(|p| p.len() as u32).collect();
        let mut queue: VecDeque<NodeId> = (0..n)
            .filter(|&i| in_degree[i] == 0)
            .map(|i| NodeId(i as u32))
            .collect();
        let mut order = Vec::with_capacity(n);
        while let Some(id) = queue.pop_front() {
            order.push(id);
            for &succ in &self.successors[id.0 as usize] {
                let d = &mut in_degree[succ.0 as usize];
                *d -= 1;
                if *d == 0 {
                    queue.push_back(succ);
                }
            }
        }
        // Safety: the DAG invariant guarantees all nodes are reachable.
        debug_assert_eq!(
            order.len(),
            n,
            "topological sort incomplete — internal invariant broken"
        );
        Ok(order)
    }

    // -----------------------------------------------------------------------
    // Buffer-derived data-flow edges
    // -----------------------------------------------------------------------

    /// Infers and adds control edges from buffer data-flow.
    ///
    /// For every buffer `b`, if node `a` writes `b` (has `b` in outputs) and
    /// node `c` reads `b` (has `b` in inputs), adds the dependency edge `a → c`.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::CycleDetected`] if any inferred edge creates a cycle.
    pub fn infer_data_edges(&mut self) -> GraphResult<()> {
        // Build writer map: buffer → list of nodes that write it.
        let mut writers: HashMap<BufferId, Vec<NodeId>> = HashMap::new();
        for node in &self.nodes {
            for &buf in &node.outputs {
                writers.entry(buf).or_default().push(node.id);
            }
        }
        // For each node that reads a buffer, add edge from all writers → this node.
        let reader_data: Vec<(NodeId, Vec<BufferId>)> = self
            .nodes
            .iter()
            .map(|n| (n.id, n.inputs.clone()))
            .collect();
        for (reader_id, inputs) in reader_data {
            for buf in inputs {
                if let Some(node_writers) = writers.get(&buf) {
                    for &writer_id in node_writers {
                        if writer_id != reader_id {
                            // Ignore duplicate-edge or cycle errors here: cycles
                            // introduced by explicit buffer write-after-write are
                            // a user error caught by add_edge.
                            self.add_edge(writer_id, reader_id)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Subgraph extraction
    // -----------------------------------------------------------------------

    /// Returns the set of all nodes reachable (forward) from the given roots.
    pub fn reachable_from(&self, roots: &[NodeId]) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut stack: Vec<NodeId> = roots.to_vec();
        while let Some(id) = stack.pop() {
            if visited.insert(id) {
                for &s in &self.successors[id.0 as usize] {
                    if !visited.contains(&s) {
                        stack.push(s);
                    }
                }
            }
        }
        visited
    }

    /// Returns the set of all nodes that can reach any of the given targets (reverse reachability).
    pub fn reaching(&self, targets: &[NodeId]) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut stack: Vec<NodeId> = targets.to_vec();
        while let Some(id) = stack.pop() {
            if visited.insert(id) {
                for &p in &self.predecessors[id.0 as usize] {
                    if !visited.contains(&p) {
                        stack.push(p);
                    }
                }
            }
        }
        visited
    }

    // -----------------------------------------------------------------------
    // Graph properties
    // -----------------------------------------------------------------------

    /// Returns `true` if the graph has no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the longest path length (in edges) from any source to any sink.
    ///
    /// This is the critical path length — a lower bound on the sequential
    /// execution depth of the graph.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::EmptyGraph`] if the graph is empty.
    pub fn critical_path_length(&self) -> GraphResult<usize> {
        let order = self.topological_order()?;
        let n = self.nodes.len();
        let mut dist = vec![0usize; n];
        let mut max_len = 0usize;
        for id in &order {
            let d = dist[id.0 as usize];
            max_len = max_len.max(d);
            for &succ in &self.successors[id.0 as usize] {
                let nd = d + 1;
                if nd > dist[succ.0 as usize] {
                    dist[succ.0 as usize] = nd;
                }
            }
        }
        Ok(max_len)
    }

    /// Returns the maximum fan-in (in-degree) among all nodes.
    pub fn max_in_degree(&self) -> usize {
        self.predecessors.iter().map(|p| p.len()).max().unwrap_or(0)
    }

    /// Returns the maximum fan-out (out-degree) among all nodes.
    pub fn max_out_degree(&self) -> usize {
        self.successors.iter().map(|s| s.len()).max().unwrap_or(0)
    }

    /// Returns the number of parallel chains (independent execution paths).
    ///
    /// This is an upper bound on the number of streams that can be usefully
    /// exploited for concurrent execution.
    pub fn parallelism_width(&self) -> GraphResult<usize> {
        // Width = maximum number of nodes at the same topological "level"
        // (BFS layer from sources).
        if self.nodes.is_empty() {
            return Ok(0);
        }
        let n = self.nodes.len();
        let mut level = vec![0usize; n];
        let mut in_degree: Vec<u32> = self.predecessors.iter().map(|p| p.len() as u32).collect();
        let mut queue: VecDeque<NodeId> = (0..n)
            .filter(|&i| in_degree[i] == 0)
            .map(|i| NodeId(i as u32))
            .collect();
        let mut max_width = queue.len();
        while let Some(id) = queue.pop_front() {
            for &succ in &self.successors[id.0 as usize] {
                let nl = level[id.0 as usize] + 1;
                if nl > level[succ.0 as usize] {
                    level[succ.0 as usize] = nl;
                }
                let d = &mut in_degree[succ.0 as usize];
                *d -= 1;
                if *d == 0 {
                    queue.push_back(succ);
                }
            }
            // Compute width at each level by scanning all level values.
        }
        // Count nodes per level.
        let max_level = *level.iter().max().unwrap_or(&0);
        let mut width_at_level = vec![0usize; max_level + 1];
        for &lv in &level {
            width_at_level[lv] += 1;
        }
        max_width = max_width.max(*width_at_level.iter().max().unwrap_or(&0));
        Ok(max_width)
    }

    // -----------------------------------------------------------------------
    // Kernel-type queries
    // -----------------------------------------------------------------------

    /// Returns all nodes that are kernel launches (compute nodes).
    pub fn kernel_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.kind.is_compute())
            .map(|n| n.id)
            .collect()
    }

    /// Returns all nodes that are fusible kernel launches.
    pub fn fusible_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.kind.is_fusible())
            .map(|n| n.id)
            .collect()
    }

    // -----------------------------------------------------------------------
    // DOT format serialisation
    // -----------------------------------------------------------------------

    /// Renders the graph in Graphviz DOT format for visualisation.
    pub fn to_dot(&self) -> String {
        let mut s = String::from("digraph ComputeGraph {\n  rankdir=TB;\n");
        for node in &self.nodes {
            let label = node.display_name();
            let shape = if node.kind.is_compute() {
                "box"
            } else if node.kind.is_memory_op() {
                "parallelogram"
            } else {
                "ellipse"
            };
            s.push_str(&format!(
                "  {} [label=\"{} ({})\", shape={shape}];\n",
                node.id.0,
                label,
                node.kind.tag()
            ));
        }
        for (i, succs) in self.successors.iter().enumerate() {
            for &to in succs {
                s.push_str(&format!("  {} -> {};\n", i, to.0));
            }
        }
        s.push('}');
        s
    }
}

impl std::fmt::Display for ComputeGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ComputeGraph({} nodes, {} edges, {} buffers)",
            self.node_count(),
            self.edge_count(),
            self.buffer_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{BufferDescriptor, KernelConfig, MemcpyDir, NodeKind};

    fn kernel_node(name: &str) -> GraphNode {
        GraphNode::new(
            NodeId(0),
            NodeKind::KernelLaunch {
                function_name: name.into(),
                config: KernelConfig::linear(1, 32, 0),
                fusible: true,
            },
        )
    }

    fn barrier_node() -> GraphNode {
        GraphNode::new(NodeId(0), NodeKind::Barrier)
    }

    fn memcpy_node(dir: MemcpyDir, size: usize) -> GraphNode {
        GraphNode::new(
            NodeId(0),
            NodeKind::Memcpy {
                dir,
                size_bytes: size,
            },
        )
    }

    // --- Basic construction ---

    #[test]
    fn new_graph_is_empty() {
        let g = ComputeGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.buffer_count(), 0);
    }

    #[test]
    fn default_is_empty() {
        let g = ComputeGraph::default();
        assert!(g.is_empty());
    }

    #[test]
    fn add_node_assigns_sequential_ids() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        assert_eq!(a, NodeId(0));
        assert_eq!(b, NodeId(1));
        assert_eq!(c, NodeId(2));
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn node_lookup_valid() {
        let mut g = ComputeGraph::new();
        let id = g.add_node(kernel_node("add"));
        assert!(g.node(id).is_ok());
        assert_eq!(g.node(id).unwrap().kind.function_name(), Some("add"));
    }

    #[test]
    fn node_lookup_invalid() {
        let g = ComputeGraph::new();
        assert!(matches!(
            g.node(NodeId(0)),
            Err(GraphError::NodeNotFound(_))
        ));
    }

    #[test]
    fn node_mut_allows_modification() {
        let mut g = ComputeGraph::new();
        let id = g.add_node(barrier_node());
        g.node_mut(id).unwrap().cost_hint = 42;
        assert_eq!(g.node(id).unwrap().cost_hint, 42);
    }

    // --- Buffer management ---

    #[test]
    fn add_buffer_assigns_ids() {
        let mut g = ComputeGraph::new();
        let b0 = g.add_buffer(BufferDescriptor::new(BufferId(0), 1024));
        let b1 = g.add_buffer(BufferDescriptor::new(BufferId(0), 2048));
        assert_eq!(b0, BufferId(0));
        assert_eq!(b1, BufferId(1));
        assert_eq!(g.buffer_count(), 2);
    }

    #[test]
    fn buffer_lookup_invalid() {
        let g = ComputeGraph::new();
        assert!(g.buffer(BufferId(0)).is_err());
    }

    // --- Edge management ---

    #[test]
    fn add_edge_valid() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(kernel_node("a"));
        let b = g.add_node(kernel_node("b"));
        assert!(g.add_edge(a, b).is_ok());
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.successors(a).unwrap(), &[b]);
        assert_eq!(g.predecessors(b).unwrap(), &[a]);
    }

    #[test]
    fn add_edge_self_loop_rejected() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        assert!(matches!(
            g.add_edge(a, a),
            Err(GraphError::CycleDetected { .. })
        ));
    }

    #[test]
    fn add_edge_cycle_rejected() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        assert!(matches!(
            g.add_edge(b, a),
            Err(GraphError::CycleDetected { .. })
        ));
    }

    #[test]
    fn add_edge_invalid_node_rejected() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        assert!(matches!(
            g.add_edge(a, NodeId(99)),
            Err(GraphError::NodeNotFound(_))
        ));
        assert!(matches!(
            g.add_edge(NodeId(99), a),
            Err(GraphError::NodeNotFound(_))
        ));
    }

    #[test]
    fn add_edge_duplicate_is_idempotent() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(a, b).unwrap(); // second call must succeed and NOT duplicate
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn edges_returns_all() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(a, c).unwrap();
        let mut edges = g.edges();
        edges.sort();
        assert_eq!(edges, vec![(a, b), (a, c)]);
    }

    // --- Reachability ---

    #[test]
    fn is_reachable_direct() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        assert!(g.is_reachable(a, b));
        assert!(!g.is_reachable(b, a));
    }

    #[test]
    fn is_reachable_transitive() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(b, c).unwrap();
        assert!(g.is_reachable(a, c));
        assert!(!g.is_reachable(c, a));
    }

    #[test]
    fn is_reachable_disconnected() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        assert!(!g.is_reachable(a, b));
        assert!(!g.is_reachable(b, a));
    }

    // --- Sources / sinks ---

    #[test]
    fn sources_and_sinks_linear_chain() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(b, c).unwrap();
        let sources = g.sources();
        let sinks = g.sinks();
        assert_eq!(sources, vec![a]);
        assert_eq!(sinks, vec![c]);
    }

    #[test]
    fn sources_and_sinks_diamond() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node()); // source
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        let d = g.add_node(barrier_node()); // sink
        g.add_edge(a, b).unwrap();
        g.add_edge(a, c).unwrap();
        g.add_edge(b, d).unwrap();
        g.add_edge(c, d).unwrap();
        assert_eq!(g.sources(), vec![a]);
        assert_eq!(g.sinks(), vec![d]);
    }

    // --- Topological sort ---

    #[test]
    fn topological_order_linear() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(b, c).unwrap();
        let order = g.topological_order().unwrap();
        let pos_a = order.iter().position(|&x| x == a).unwrap();
        let pos_b = order.iter().position(|&x| x == b).unwrap();
        let pos_c = order.iter().position(|&x| x == c).unwrap();
        assert!(pos_a < pos_b && pos_b < pos_c);
    }

    #[test]
    fn topological_order_diamond() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        let d = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(a, c).unwrap();
        g.add_edge(b, d).unwrap();
        g.add_edge(c, d).unwrap();
        let order = g.topological_order().unwrap();
        assert_eq!(order.len(), 4);
        let pos = |n: NodeId| order.iter().position(|&x| x == n).unwrap();
        assert!(pos(a) < pos(b));
        assert!(pos(a) < pos(c));
        assert!(pos(b) < pos(d));
        assert!(pos(c) < pos(d));
    }

    #[test]
    fn topological_order_empty_graph() {
        let g = ComputeGraph::new();
        assert!(matches!(g.topological_order(), Err(GraphError::EmptyGraph)));
    }

    #[test]
    fn topological_order_isolated_nodes() {
        let mut g = ComputeGraph::new();
        g.add_node(barrier_node());
        g.add_node(barrier_node());
        g.add_node(barrier_node());
        let order = g.topological_order().unwrap();
        assert_eq!(order.len(), 3);
    }

    // --- Data-flow edge inference ---

    #[test]
    fn infer_data_edges_connects_writer_to_reader() {
        let mut g = ComputeGraph::new();
        let buf = g.add_buffer(BufferDescriptor::new(BufferId(0), 1024));
        let writer = g.add_node(GraphNode::new(NodeId(0), NodeKind::Barrier).with_outputs([buf]));
        let reader = g.add_node(GraphNode::new(NodeId(0), NodeKind::Barrier).with_inputs([buf]));
        g.infer_data_edges().unwrap();
        assert!(g.is_reachable(writer, reader));
    }

    #[test]
    fn infer_data_edges_multiple_readers() {
        let mut g = ComputeGraph::new();
        let buf = g.add_buffer(BufferDescriptor::new(BufferId(0), 1024));
        let writer = g.add_node(GraphNode::new(NodeId(0), NodeKind::Barrier).with_outputs([buf]));
        let r1 = g.add_node(GraphNode::new(NodeId(0), NodeKind::Barrier).with_inputs([buf]));
        let r2 = g.add_node(GraphNode::new(NodeId(0), NodeKind::Barrier).with_inputs([buf]));
        g.infer_data_edges().unwrap();
        assert!(g.is_reachable(writer, r1));
        assert!(g.is_reachable(writer, r2));
    }

    // --- Critical path, degrees ---

    #[test]
    fn critical_path_linear_chain() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        let d = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(b, c).unwrap();
        g.add_edge(c, d).unwrap();
        assert_eq!(g.critical_path_length().unwrap(), 3);
    }

    #[test]
    fn critical_path_diamond() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        let d = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(a, c).unwrap();
        g.add_edge(b, d).unwrap();
        g.add_edge(c, d).unwrap();
        assert_eq!(g.critical_path_length().unwrap(), 2);
    }

    #[test]
    fn max_degrees_computed_correctly() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        let d = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(a, c).unwrap();
        g.add_edge(a, d).unwrap();
        assert_eq!(g.max_out_degree(), 3);
        assert_eq!(g.max_in_degree(), 1);
    }

    // --- Subgraph reachability ---

    #[test]
    fn reachable_from_set() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        let d = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(b, c).unwrap();
        // d is isolated
        let reach = g.reachable_from(&[a]);
        assert!(reach.contains(&a));
        assert!(reach.contains(&b));
        assert!(reach.contains(&c));
        assert!(!reach.contains(&d));
    }

    #[test]
    fn reaching_set() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(b, c).unwrap();
        let reaching = g.reaching(&[c]);
        assert!(reaching.contains(&a));
        assert!(reaching.contains(&b));
        assert!(reaching.contains(&c));
    }

    // --- Kernel / fusible queries ---

    #[test]
    fn kernel_nodes_returns_compute_only() {
        let mut g = ComputeGraph::new();
        g.add_node(kernel_node("k0"));
        g.add_node(barrier_node());
        g.add_node(memcpy_node(MemcpyDir::HostToDevice, 1024));
        g.add_node(kernel_node("k1"));
        let kernels = g.kernel_nodes();
        assert_eq!(kernels.len(), 2);
    }

    #[test]
    fn fusible_nodes_returns_fusible_only() {
        let mut g = ComputeGraph::new();
        g.add_node(kernel_node("fusible")); // fusible=true in helper
        g.add_node(GraphNode::new(
            NodeId(0),
            NodeKind::KernelLaunch {
                function_name: "custom".into(),
                config: KernelConfig::linear(1, 32, 0),
                fusible: false,
            },
        ));
        assert_eq!(g.fusible_nodes().len(), 1);
    }

    // --- Parallelism width ---

    #[test]
    fn parallelism_width_linear_is_one() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        g.add_edge(b, c).unwrap();
        assert_eq!(g.parallelism_width().unwrap(), 1);
    }

    #[test]
    fn parallelism_width_fork_join() {
        let mut g = ComputeGraph::new();
        let src = g.add_node(barrier_node());
        let b = g.add_node(barrier_node());
        let c = g.add_node(barrier_node());
        let d = g.add_node(barrier_node());
        let sink = g.add_node(barrier_node());
        g.add_edge(src, b).unwrap();
        g.add_edge(src, c).unwrap();
        g.add_edge(src, d).unwrap();
        g.add_edge(b, sink).unwrap();
        g.add_edge(c, sink).unwrap();
        g.add_edge(d, sink).unwrap();
        assert_eq!(g.parallelism_width().unwrap(), 3);
    }

    // --- DOT output ---

    #[test]
    fn to_dot_contains_node_labels() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(kernel_node("my_kernel").with_name("k0"));
        let b = g.add_node(barrier_node());
        g.add_edge(a, b).unwrap();
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("k0"));
        assert!(dot.contains("->"));
    }

    // --- Display ---

    #[test]
    fn display_shows_counts() {
        let mut g = ComputeGraph::new();
        g.add_node(barrier_node());
        g.add_node(barrier_node());
        g.add_buffer(BufferDescriptor::new(BufferId(0), 1));
        let s = g.to_string();
        assert!(s.contains("2 nodes"));
        assert!(s.contains("1 buffers"));
    }
}
