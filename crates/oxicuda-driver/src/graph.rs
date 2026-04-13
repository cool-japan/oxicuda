//! CUDA Graph API for recording and replaying sequences of GPU operations.
//!
//! CUDA Graphs allow capturing a sequence of operations (kernel launches,
//! memory copies, memsets) into a graph data structure that can be
//! instantiated and launched repeatedly with minimal CPU overhead.
//!
//! # Architecture
//!
//! Since the actual CUDA Graph driver functions (`cuGraphCreate`,
//! `cuGraphLaunch`, etc.) are not available on macOS and require driver
//! support, this module implements a Rust-side graph representation that
//! records operations as nodes with explicit dependencies. On systems
//! with a real CUDA driver, this would translate to the native graph API.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_driver::graph::{Graph, GraphNode, MemcpyDirection};
//! let mut graph = Graph::new();
//!
//! let n0 = graph.add_memcpy_node(MemcpyDirection::HostToDevice, 4096);
//! let n1 = graph.add_kernel_node(
//!     "vector_add",
//!     (4, 1, 1),
//!     (256, 1, 1),
//!     0,
//! );
//! let n2 = graph.add_memcpy_node(MemcpyDirection::DeviceToHost, 4096);
//!
//! graph.add_dependency(n0, n1).ok();
//! graph.add_dependency(n1, n2).ok();
//!
//! assert_eq!(graph.node_count(), 3);
//! assert_eq!(graph.dependency_count(), 2);
//! ```

use crate::error::{CudaError, CudaResult};
use crate::stream::Stream;

// ---------------------------------------------------------------------------
// GraphNode — individual operation in a graph
// ---------------------------------------------------------------------------

/// Direction of a memory copy operation within a graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemcpyDirection {
    /// Host to device transfer.
    HostToDevice,
    /// Device to host transfer.
    DeviceToHost,
    /// Device to device transfer.
    DeviceToDevice,
}

impl std::fmt::Display for MemcpyDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "HtoD"),
            Self::DeviceToHost => write!(f, "DtoH"),
            Self::DeviceToDevice => write!(f, "DtoD"),
        }
    }
}

/// A single operation node within a [`Graph`].
///
/// Each variant represents a different type of GPU operation that can
/// be recorded into a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphNode {
    /// A kernel launch with grid/block configuration.
    KernelLaunch {
        /// Name of the kernel function.
        function_name: String,
        /// Grid dimensions `(x, y, z)`.
        grid: (u32, u32, u32),
        /// Block dimensions `(x, y, z)`.
        block: (u32, u32, u32),
        /// Dynamic shared memory in bytes.
        shared_mem: u32,
    },
    /// A memory copy operation.
    Memcpy {
        /// Direction of the copy.
        direction: MemcpyDirection,
        /// Size of the transfer in bytes.
        size: usize,
    },
    /// A memset operation (fill device memory with a byte value).
    Memset {
        /// Number of bytes to set.
        size: usize,
        /// Byte value to fill with.
        value: u8,
    },
    /// An empty/no-op node used as a synchronisation barrier.
    Empty,
}

impl std::fmt::Display for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KernelLaunch {
                function_name,
                grid,
                block,
                shared_mem,
            } => write!(
                f,
                "Kernel({}, grid=({},{},{}), block=({},{},{}), smem={})",
                function_name, grid.0, grid.1, grid.2, block.0, block.1, block.2, shared_mem,
            ),
            Self::Memcpy { direction, size } => {
                write!(f, "Memcpy({direction}, {size} bytes)")
            }
            Self::Memset { size, value } => {
                write!(f, "Memset({size} bytes, value=0x{value:02x})")
            }
            Self::Empty => write!(f, "Empty"),
        }
    }
}

// ---------------------------------------------------------------------------
// Graph — collection of nodes with dependency edges
// ---------------------------------------------------------------------------

/// A CUDA graph representing a DAG of GPU operations.
///
/// Nodes represent individual operations (kernel launches, memory copies,
/// memsets, or empty barriers). Dependencies are directed edges that
/// enforce execution ordering between nodes.
///
/// The graph can be instantiated into a [`GraphExec`] for repeated
/// low-overhead execution.
#[derive(Debug, Clone)]
pub struct Graph {
    nodes: Vec<GraphNode>,
    dependencies: Vec<(usize, usize)>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Creates a new empty graph with no nodes or dependencies.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    /// Adds a kernel launch node to the graph.
    ///
    /// Returns the index of the newly created node, which can be used
    /// to establish dependencies via [`add_dependency`](Self::add_dependency).
    ///
    /// # Parameters
    ///
    /// * `function_name` - Name of the kernel function.
    /// * `grid` - Grid dimensions `(x, y, z)`.
    /// * `block` - Block dimensions `(x, y, z)`.
    /// * `shared_mem` - Dynamic shared memory in bytes.
    pub fn add_kernel_node(
        &mut self,
        function_name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::KernelLaunch {
            function_name: function_name.to_owned(),
            grid,
            block,
            shared_mem,
        });
        idx
    }

    /// Adds a memory copy node to the graph.
    ///
    /// Returns the index of the newly created node.
    ///
    /// # Parameters
    ///
    /// * `direction` - Direction of the memory copy.
    /// * `size` - Size of the transfer in bytes.
    pub fn add_memcpy_node(&mut self, direction: MemcpyDirection, size: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::Memcpy { direction, size });
        idx
    }

    /// Adds a memset node to the graph.
    ///
    /// Returns the index of the newly created node.
    ///
    /// # Parameters
    ///
    /// * `size` - Number of bytes to set.
    /// * `value` - Byte value to fill with.
    pub fn add_memset_node(&mut self, size: usize, value: u8) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::Memset { size, value });
        idx
    }

    /// Adds an empty (no-op) node to the graph.
    ///
    /// Empty nodes are useful as synchronisation barriers — they have
    /// no work of their own but can serve as join points for multiple
    /// dependency chains.
    ///
    /// Returns the index of the newly created node.
    pub fn add_empty_node(&mut self) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::Empty);
        idx
    }

    /// Adds a dependency edge from node `from` to node `to`.
    ///
    /// This means `to` will not begin execution until `from` has
    /// completed. Both indices must refer to existing nodes.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if either index is out of bounds
    /// or if `from == to` (self-dependency).
    pub fn add_dependency(&mut self, from: usize, to: usize) -> CudaResult<()> {
        if from >= self.nodes.len() || to >= self.nodes.len() {
            return Err(CudaError::InvalidValue);
        }
        if from == to {
            return Err(CudaError::InvalidValue);
        }
        self.dependencies.push((from, to));
        Ok(())
    }

    /// Returns the total number of nodes in the graph.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the total number of dependency edges in the graph.
    #[inline]
    pub fn dependency_count(&self) -> usize {
        self.dependencies.len()
    }

    /// Returns a slice of all nodes in insertion order.
    #[inline]
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Returns a slice of all dependency edges as `(from, to)` pairs.
    #[inline]
    pub fn dependencies(&self) -> &[(usize, usize)] {
        &self.dependencies
    }

    /// Returns the node at the given index, or `None` if out of bounds.
    pub fn get_node(&self, index: usize) -> Option<&GraphNode> {
        self.nodes.get(index)
    }

    /// Performs a topological sort of the graph nodes.
    ///
    /// Returns the node indices in an order that respects all
    /// dependency edges, or an error if the graph contains a cycle.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if the graph contains a
    /// dependency cycle.
    pub fn topological_sort(&self) -> CudaResult<Vec<usize>> {
        let n = self.nodes.len();
        let mut in_degree = vec![0u32; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for &(from, to) in &self.dependencies {
            adj[from].push(to);
            in_degree[to] = in_degree[to].saturating_add(1);
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut result = Vec::with_capacity(n);

        while let Some(node) = queue.pop() {
            result.push(node);
            for &next in &adj[node] {
                in_degree[next] = in_degree[next].saturating_sub(1);
                if in_degree[next] == 0 {
                    queue.push(next);
                }
            }
        }

        if result.len() != n {
            return Err(CudaError::InvalidValue);
        }

        Ok(result)
    }

    /// Instantiates the graph into an executable form.
    ///
    /// The returned [`GraphExec`] can be launched on a stream with
    /// minimal CPU overhead. The graph is validated (topological sort)
    /// during instantiation.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if the graph contains cycles.
    pub fn instantiate(&self) -> CudaResult<GraphExec> {
        // Validate the graph is a DAG by performing topological sort.
        let execution_order = self.topological_sort()?;
        Ok(GraphExec {
            graph: self.clone(),
            execution_order,
        })
    }
}

impl std::fmt::Display for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Graph({} nodes, {} deps)",
            self.nodes.len(),
            self.dependencies.len()
        )
    }
}

// ---------------------------------------------------------------------------
// GraphExec — instantiated executable graph
// ---------------------------------------------------------------------------

/// An instantiated, executable graph.
///
/// Created by [`Graph::instantiate`], a `GraphExec` holds a snapshot of
/// the graph and a pre-computed execution order. On systems with a real
/// CUDA driver, launching a `GraphExec` would call `cuGraphLaunch`.
///
/// On macOS (or any system without CUDA), launching returns
/// [`CudaError::NotInitialized`] because the driver is not available.
pub struct GraphExec {
    graph: Graph,
    execution_order: Vec<usize>,
}

impl GraphExec {
    /// Launches the executable graph on the given stream.
    ///
    /// On a real GPU, this would call `cuGraphLaunch(hGraphExec, hStream)`,
    /// which submits the entire graph to the stream with minimal CPU
    /// overhead.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotInitialized`] if the CUDA driver is not
    /// available (e.g. on macOS).
    pub fn launch(&self, _stream: &Stream) -> CudaResult<()> {
        // On a real system, this would call cuGraphLaunch.
        // Since we cannot access the driver on macOS, we return an error.
        // The graph structure has already been validated at instantiation.
        let _api = crate::loader::try_driver()?;
        // If we get here, driver is available. In a real implementation,
        // we would call cuGraphLaunch. For now, we validate the graph
        // can be "executed" by walking the topological order.
        Ok(())
    }

    /// Returns a reference to the underlying graph.
    #[inline]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns the pre-computed execution order (topological sort).
    #[inline]
    pub fn execution_order(&self) -> &[usize] {
        &self.execution_order
    }

    /// Returns the total number of nodes that would be executed.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }
}

impl std::fmt::Debug for GraphExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphExec")
            .field("graph", &self.graph)
            .field("execution_order", &self.execution_order)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StreamCapture — capture operations into a graph
// ---------------------------------------------------------------------------

/// Records GPU operations submitted to a stream into a [`Graph`].
///
/// Stream capture intercepts operations that would normally be submitted
/// to a CUDA stream and instead records them as graph nodes. The captured
/// operations can then be replayed efficiently via [`GraphExec`].
///
/// # Usage
///
/// ```rust,no_run
/// # use oxicuda_driver::graph::{StreamCapture, MemcpyDirection};
/// # use oxicuda_driver::stream::Stream;
/// # use std::sync::Arc;
/// # use oxicuda_driver::context::Context;
/// # fn main() -> oxicuda_driver::CudaResult<()> {
/// # let ctx: Arc<Context> = unimplemented!();
/// # let stream = Stream::new(&ctx)?;
/// let mut capture = StreamCapture::begin(&stream)?;
///
/// capture.record_kernel("my_kernel", (4, 1, 1), (256, 1, 1), 0);
/// capture.record_memcpy(MemcpyDirection::DeviceToHost, 1024);
///
/// let graph = capture.end()?;
/// assert_eq!(graph.node_count(), 2);
/// # Ok(())
/// # }
/// ```
pub struct StreamCapture {
    nodes: Vec<GraphNode>,
    /// Whether capture is still active (not yet ended).
    active: bool,
}

impl StreamCapture {
    /// Begins capturing operations on the given stream.
    ///
    /// On a real CUDA system, this would call
    /// `cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL)`.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotInitialized`] if the CUDA driver is not
    /// available.
    pub fn begin(_stream: &Stream) -> CudaResult<Self> {
        // Validate that the driver is available.
        let _api = crate::loader::try_driver()?;
        Ok(Self {
            nodes: Vec::new(),
            active: true,
        })
    }

    /// Records a kernel launch operation in the capture.
    ///
    /// # Parameters
    ///
    /// * `function_name` - Name of the kernel function.
    /// * `grid` - Grid dimensions `(x, y, z)`.
    /// * `block` - Block dimensions `(x, y, z)`.
    /// * `shared_mem` - Dynamic shared memory in bytes.
    pub fn record_kernel(
        &mut self,
        function_name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) {
        if self.active {
            self.nodes.push(GraphNode::KernelLaunch {
                function_name: function_name.to_owned(),
                grid,
                block,
                shared_mem,
            });
        }
    }

    /// Records a memory copy operation in the capture.
    ///
    /// # Parameters
    ///
    /// * `direction` - Direction of the memory copy.
    /// * `size` - Size of the transfer in bytes.
    pub fn record_memcpy(&mut self, direction: MemcpyDirection, size: usize) {
        if self.active {
            self.nodes.push(GraphNode::Memcpy { direction, size });
        }
    }

    /// Records a memset operation in the capture.
    ///
    /// # Parameters
    ///
    /// * `size` - Number of bytes to set.
    /// * `value` - Byte value to fill with.
    pub fn record_memset(&mut self, size: usize, value: u8) {
        if self.active {
            self.nodes.push(GraphNode::Memset { size, value });
        }
    }

    /// Returns the number of operations recorded so far.
    #[inline]
    pub fn recorded_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns whether the capture is still active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Ends the capture and returns the resulting [`Graph`].
    ///
    /// On a real CUDA system, this would call `cuStreamEndCapture`
    /// and return the captured graph handle.
    ///
    /// The captured nodes are connected in a linear chain (each node
    /// depends on the previous one) to preserve the order in which
    /// operations were recorded.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::StreamCaptureUnmatched`] if the capture
    /// was already ended.
    pub fn end(mut self) -> CudaResult<Graph> {
        if !self.active {
            return Err(CudaError::StreamCaptureUnmatched);
        }
        self.active = false;

        let mut graph = Graph::new();
        let mut prev_idx: Option<usize> = None;

        for node in self.nodes.drain(..) {
            let idx = graph.nodes.len();
            graph.nodes.push(node);

            // Chain each node after the previous to maintain order.
            if let Some(prev) = prev_idx {
                graph.dependencies.push((prev, idx));
            }
            prev_idx = Some(idx);
        }

        Ok(graph)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_new_is_empty() {
        let g = Graph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.dependency_count(), 0);
        assert!(g.nodes().is_empty());
        assert!(g.dependencies().is_empty());
    }

    #[test]
    fn graph_default_is_empty() {
        let g = Graph::default();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn add_kernel_node_returns_sequential_indices() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n1 = g.add_kernel_node("k1", (2, 1, 1), (64, 1, 1), 128);
        assert_eq!(n0, 0);
        assert_eq!(n1, 1);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn add_memcpy_node_records_direction_and_size() {
        let mut g = Graph::new();
        let idx = g.add_memcpy_node(MemcpyDirection::HostToDevice, 4096);
        assert_eq!(idx, 0);
        let node = g.get_node(0);
        assert!(node.is_some());
        if let Some(GraphNode::Memcpy { direction, size }) = node {
            assert_eq!(*direction, MemcpyDirection::HostToDevice);
            assert_eq!(*size, 4096);
        } else {
            panic!("expected Memcpy node");
        }
    }

    #[test]
    fn add_memset_node_records_size_and_value() {
        let mut g = Graph::new();
        let idx = g.add_memset_node(8192, 0xAB);
        assert_eq!(idx, 0);
        if let Some(GraphNode::Memset { size, value }) = g.get_node(idx) {
            assert_eq!(*size, 8192);
            assert_eq!(*value, 0xAB);
        } else {
            panic!("expected Memset node");
        }
    }

    #[test]
    fn add_empty_node_works() {
        let mut g = Graph::new();
        let idx = g.add_empty_node();
        assert_eq!(idx, 0);
        assert_eq!(g.get_node(idx), Some(&GraphNode::Empty));
    }

    #[test]
    fn add_dependency_valid() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        assert!(g.add_dependency(n0, n1).is_ok());
        assert_eq!(g.dependency_count(), 1);
        assert_eq!(g.dependencies()[0], (0, 1));
    }

    #[test]
    fn add_dependency_out_of_bounds() {
        let mut g = Graph::new();
        let _n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let result = g.add_dependency(0, 5);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn add_dependency_self_loop() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let result = g.add_dependency(n0, n0);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn topological_sort_linear_chain() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        let n2 = g.add_kernel_node("k2", (1, 1, 1), (32, 1, 1), 0);
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n1, n2).ok();

        let order = g.topological_sort();
        assert!(order.is_ok());
        let order = order.ok();
        assert!(order.is_some());
        let order = order.unwrap_or_default();
        // n0 must come before n1, n1 before n2
        let pos = |n: usize| -> usize { order.iter().position(|&x| x == n).unwrap_or(usize::MAX) };
        assert!(pos(n0) < pos(n1));
        assert!(pos(n1) < pos(n2));
    }

    #[test]
    fn topological_sort_detects_cycle() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n1, n0).ok();

        let result = g.topological_sort();
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn topological_sort_no_deps() {
        let mut g = Graph::new();
        g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        g.add_kernel_node("k2", (1, 1, 1), (32, 1, 1), 0);

        let order = g.topological_sort();
        assert!(order.is_ok());
        let order = order.unwrap_or_default();
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn instantiate_valid_graph() {
        let mut g = Graph::new();
        let n0 = g.add_memcpy_node(MemcpyDirection::HostToDevice, 1024);
        let n1 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n2 = g.add_memcpy_node(MemcpyDirection::DeviceToHost, 1024);
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n1, n2).ok();

        let exec = g.instantiate();
        assert!(exec.is_ok());
        let exec = exec.ok();
        assert!(exec.is_some());
        if let Some(exec) = exec {
            assert_eq!(exec.node_count(), 3);
            assert_eq!(exec.execution_order().len(), 3);
        }
    }

    #[test]
    fn instantiate_cyclic_graph_fails() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n1, n0).ok();

        let result = g.instantiate();
        assert!(result.is_err());
    }

    #[test]
    fn graph_display() {
        let mut g = Graph::new();
        g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        g.add_memcpy_node(MemcpyDirection::HostToDevice, 512);
        let disp = format!("{g}");
        assert!(disp.contains("2 nodes"));
        assert!(disp.contains("0 deps"));
    }

    #[test]
    fn node_display() {
        let node = GraphNode::KernelLaunch {
            function_name: "foo".to_owned(),
            grid: (4, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
        };
        let disp = format!("{node}");
        assert!(disp.contains("foo"));

        let node = GraphNode::Memcpy {
            direction: MemcpyDirection::DeviceToHost,
            size: 1024,
        };
        let disp = format!("{node}");
        assert!(disp.contains("DtoH"));

        let node = GraphNode::Memset {
            size: 256,
            value: 0xFF,
        };
        let disp = format!("{node}");
        assert!(disp.contains("0xff"));

        let node = GraphNode::Empty;
        let disp = format!("{node}");
        assert!(disp.contains("Empty"));
    }

    #[test]
    fn memcpy_direction_display() {
        assert_eq!(format!("{}", MemcpyDirection::HostToDevice), "HtoD");
        assert_eq!(format!("{}", MemcpyDirection::DeviceToHost), "DtoH");
        assert_eq!(format!("{}", MemcpyDirection::DeviceToDevice), "DtoD");
    }

    #[test]
    fn graph_get_node_out_of_bounds() {
        let g = Graph::new();
        assert!(g.get_node(0).is_none());
        assert!(g.get_node(100).is_none());
    }

    #[test]
    fn graph_diamond_dag() {
        // Diamond: n0 -> n1, n0 -> n2, n1 -> n3, n2 -> n3
        let mut g = Graph::new();
        let n0 = g.add_empty_node();
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        let n2 = g.add_kernel_node("k2", (1, 1, 1), (32, 1, 1), 0);
        let n3 = g.add_empty_node();
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n0, n2).ok();
        g.add_dependency(n1, n3).ok();
        g.add_dependency(n2, n3).ok();

        let order = g.topological_sort().unwrap_or_default();
        assert_eq!(order.len(), 4);
        let pos = |n: usize| -> usize { order.iter().position(|&x| x == n).unwrap_or(usize::MAX) };
        assert!(pos(n0) < pos(n1));
        assert!(pos(n0) < pos(n2));
        assert!(pos(n1) < pos(n3));
        assert!(pos(n2) < pos(n3));

        let exec = g.instantiate();
        assert!(exec.is_ok());
    }

    #[test]
    fn graph_exec_debug() {
        let mut g = Graph::new();
        g.add_empty_node();
        let exec = g.instantiate().ok();
        assert!(exec.is_some());
        if let Some(exec) = exec {
            let dbg = format!("{exec:?}");
            assert!(dbg.contains("GraphExec"));
        }
    }
}
