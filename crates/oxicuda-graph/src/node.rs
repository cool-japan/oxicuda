//! Core graph node types for the OxiCUDA computation graph.
//!
//! This module defines the fundamental building blocks of the computation
//! graph: typed identifiers (`NodeId`, `BufferId`, `StreamId`) and the
//! `GraphNode` which describes a single GPU operation along with its
//! data-flow edges (input and output buffer references).

use std::fmt;

// ---------------------------------------------------------------------------
// Typed identifier wrappers
// ---------------------------------------------------------------------------

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "N{}", self.0)
    }
}

/// Unique identifier for a logical buffer (device memory region) in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId(pub u32);

impl fmt::Display for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B{}", self.0)
    }
}

/// Identifier for a CUDA stream assignment.
///
/// `StreamId(0)` is the default stream. Streams with IDs ≥ 1 are
/// independent execution queues that can run concurrently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(pub u32);

impl StreamId {
    /// The default (null) stream.
    pub const DEFAULT: Self = Self(0);
}

impl fmt::Display for StreamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            write!(f, "S:default")
        } else {
            write!(f, "S{}", self.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel launch configuration
// ---------------------------------------------------------------------------

/// GPU grid/block launch configuration for a kernel node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelConfig {
    /// Grid dimensions `(x, y, z)`.
    pub grid: (u32, u32, u32),
    /// Block dimensions `(x, y, z)`.
    pub block: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
}

impl KernelConfig {
    /// Creates a simple 1-D launch configuration.
    ///
    /// * `num_blocks` — number of thread-blocks in the x dimension.
    /// * `threads_per_block` — threads per block in the x dimension.
    /// * `shared_mem_bytes` — bytes of dynamic shared memory per block.
    #[must_use]
    pub fn linear(num_blocks: u32, threads_per_block: u32, shared_mem_bytes: u32) -> Self {
        Self {
            grid: (num_blocks, 1, 1),
            block: (threads_per_block, 1, 1),
            shared_mem_bytes,
        }
    }

    /// Total number of threads launched.
    #[must_use]
    pub fn total_threads(&self) -> u64 {
        let g = self.grid;
        let b = self.block;
        u64::from(g.0)
            * u64::from(g.1)
            * u64::from(g.2)
            * u64::from(b.0)
            * u64::from(b.1)
            * u64::from(b.2)
    }

    /// Total number of thread-blocks.
    #[must_use]
    pub fn total_blocks(&self) -> u64 {
        let g = self.grid;
        u64::from(g.0) * u64::from(g.1) * u64::from(g.2)
    }
}

impl fmt::Display for KernelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (gx, gy, gz) = self.grid;
        let (bx, by, bz) = self.block;
        write!(
            f,
            "grid=({gx},{gy},{gz}) block=({bx},{by},{bz}) smem={}B",
            self.shared_mem_bytes
        )
    }
}

// ---------------------------------------------------------------------------
// Memcpy direction
// ---------------------------------------------------------------------------

/// Direction of a device memory copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemcpyDir {
    /// Host-to-device transfer (upload).
    HostToDevice,
    /// Device-to-host transfer (download).
    DeviceToHost,
    /// Device-to-device transfer (on-device copy).
    DeviceToDevice,
    /// Peer-to-peer transfer across devices.
    PeerToPeer { src_device: u32, dst_device: u32 },
}

impl fmt::Display for MemcpyDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "HtoD"),
            Self::DeviceToHost => write!(f, "DtoH"),
            Self::DeviceToDevice => write!(f, "DtoD"),
            Self::PeerToPeer {
                src_device,
                dst_device,
            } => {
                write!(f, "PeerToPeer(dev{src_device}→dev{dst_device})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NodeKind — the operation performed by a node
// ---------------------------------------------------------------------------

/// The operation kind associated with a [`GraphNode`].
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    /// Launches a PTX/CUDA kernel.
    KernelLaunch {
        /// Kernel function name (used to look up the loaded module).
        function_name: String,
        /// Grid/block configuration.
        config: KernelConfig,
        /// Whether this kernel is eligible for fusion with adjacent element-wise kernels.
        fusible: bool,
    },

    /// Copies memory between host and device, or device to device.
    Memcpy {
        /// Transfer direction.
        dir: MemcpyDir,
        /// Transfer size in bytes.
        size_bytes: usize,
    },

    /// Fills a device buffer with a byte pattern.
    Memset {
        /// Number of bytes to fill.
        size_bytes: usize,
        /// Byte fill value.
        value: u8,
    },

    /// Records a CUDA event (signals completion of preceding work).
    EventRecord,

    /// Waits for a recorded event before proceeding.
    EventWait,

    /// A host-side callback inserted into the stream.
    HostCallback {
        /// Human-readable label for debugging and visualisation.
        label: String,
    },

    /// A no-op synchronisation barrier (join point for multiple dependency chains).
    Barrier,

    /// A conditional subgraph fork (CUDA graph conditional node equivalent).
    Conditional {
        /// Condition variable — a boolean device buffer.
        condition_buf: BufferId,
        /// Which branch is the "true" path (node IDs within this graph).
        true_branch: Vec<NodeId>,
        /// Which branch is the "false" path.
        false_branch: Vec<NodeId>,
    },
}

impl NodeKind {
    /// Returns `true` if this node performs device computation (not just memory movement).
    #[must_use]
    pub fn is_compute(&self) -> bool {
        matches!(self, Self::KernelLaunch { .. })
    }

    /// Returns `true` if this node is a memory transfer.
    #[must_use]
    pub fn is_memory_op(&self) -> bool {
        matches!(self, Self::Memcpy { .. } | Self::Memset { .. })
    }

    /// Returns `true` if this kernel is marked as fusible with adjacent element-wise kernels.
    #[must_use]
    pub fn is_fusible(&self) -> bool {
        match self {
            Self::KernelLaunch { fusible, .. } => *fusible,
            _ => false,
        }
    }

    /// Returns the function name for kernel launch nodes.
    #[must_use]
    pub fn function_name(&self) -> Option<&str> {
        match self {
            Self::KernelLaunch { function_name, .. } => Some(function_name.as_str()),
            _ => None,
        }
    }

    /// Returns a short human-readable tag for this operation kind.
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            Self::KernelLaunch { .. } => "kernel",
            Self::Memcpy { .. } => "memcpy",
            Self::Memset { .. } => "memset",
            Self::EventRecord => "event_record",
            Self::EventWait => "event_wait",
            Self::HostCallback { .. } => "host_cb",
            Self::Barrier => "barrier",
            Self::Conditional { .. } => "conditional",
        }
    }
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelLaunch {
                function_name,
                config,
                fusible,
            } => {
                let fuse = if *fusible { " [fusible]" } else { "" };
                write!(f, "KernelLaunch({function_name}, {config}{fuse})")
            }
            Self::Memcpy { dir, size_bytes } => {
                write!(f, "Memcpy({dir}, {size_bytes}B)")
            }
            Self::Memset { size_bytes, value } => {
                write!(f, "Memset({size_bytes}B, 0x{value:02x})")
            }
            Self::EventRecord => write!(f, "EventRecord"),
            Self::EventWait => write!(f, "EventWait"),
            Self::HostCallback { label } => write!(f, "HostCallback({label})"),
            Self::Barrier => write!(f, "Barrier"),
            Self::Conditional { condition_buf, .. } => {
                write!(f, "Conditional(cond={condition_buf})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GraphNode — a node in the computation DAG
// ---------------------------------------------------------------------------

/// A single node in the OxiCUDA computation graph.
///
/// Each node represents one GPU operation. Data-flow is captured through
/// explicit `inputs` and `outputs` buffer lists: an output buffer of one
/// node becomes an input buffer of its successors, forming the data
/// dependency graph alongside the explicit control dependency edges stored
/// in `ComputeGraph`.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identity within the graph.
    pub id: NodeId,
    /// The GPU operation this node performs.
    pub kind: NodeKind,
    /// Buffers consumed by this operation (read-only or read-write).
    pub inputs: Vec<BufferId>,
    /// Buffers produced or written by this operation.
    pub outputs: Vec<BufferId>,
    /// Preferred stream for this node (hint to the stream partitioner).
    pub stream_hint: Option<StreamId>,
    /// Estimated compute cost in "abstract units" for scheduling heuristics.
    pub cost_hint: u64,
    /// Optional human-readable name for debugging and profiling.
    pub name: Option<String>,
}

impl GraphNode {
    /// Creates a new node with the given identity and operation kind.
    #[must_use]
    pub fn new(id: NodeId, kind: NodeKind) -> Self {
        Self {
            id,
            kind,
            inputs: Vec::new(),
            outputs: Vec::new(),
            stream_hint: None,
            cost_hint: 1,
            name: None,
        }
    }

    /// Attaches input buffer references to this node (builder pattern).
    #[must_use]
    pub fn with_inputs(mut self, inputs: impl IntoIterator<Item = BufferId>) -> Self {
        self.inputs.extend(inputs);
        self
    }

    /// Attaches output buffer references to this node (builder pattern).
    #[must_use]
    pub fn with_outputs(mut self, outputs: impl IntoIterator<Item = BufferId>) -> Self {
        self.outputs.extend(outputs);
        self
    }

    /// Sets the preferred stream for this node (builder pattern).
    #[must_use]
    pub fn with_stream(mut self, stream: StreamId) -> Self {
        self.stream_hint = Some(stream);
        self
    }

    /// Sets the cost hint for scheduling (builder pattern).
    #[must_use]
    pub fn with_cost(mut self, cost: u64) -> Self {
        self.cost_hint = cost;
        self
    }

    /// Sets the display name for this node (builder pattern).
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the node's display name, falling back to its ID.
    #[must_use]
    pub fn display_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| self.id.to_string())
    }

    /// Returns `true` if this node has no input buffers (source node).
    #[must_use]
    pub fn is_source(&self) -> bool {
        self.inputs.is_empty()
    }

    /// Returns `true` if this node has no output buffers (sink node).
    #[must_use]
    pub fn is_sink(&self) -> bool {
        self.outputs.is_empty()
    }
}

impl fmt::Display for GraphNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.display_name();
        write!(f, "[{}] {}", name, self.kind)?;
        if !self.inputs.is_empty() {
            let ins: Vec<String> = self.inputs.iter().map(|b| b.to_string()).collect();
            write!(f, " ←({})", ins.join(","))?;
        }
        if !self.outputs.is_empty() {
            let outs: Vec<String> = self.outputs.iter().map(|b| b.to_string()).collect();
            write!(f, " →({})", outs.join(","))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BufferDescriptor — metadata for a logical buffer
// ---------------------------------------------------------------------------

/// Metadata describing a logical device buffer in the graph.
///
/// Buffers are abstract references to device memory regions. The memory
/// planner uses buffer descriptors to assign physical allocations while
/// maximising reuse across non-overlapping lifetimes.
#[derive(Debug, Clone)]
pub struct BufferDescriptor {
    /// Unique identifier.
    pub id: BufferId,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Optional human-readable name.
    pub name: Option<String>,
    /// If `true`, this buffer is externally managed (not handled by the planner).
    pub external: bool,
    /// Required alignment in bytes (must be a power of two).
    pub alignment: usize,
}

impl BufferDescriptor {
    /// Creates a new buffer descriptor.
    #[must_use]
    pub fn new(id: BufferId, size_bytes: usize) -> Self {
        Self {
            id,
            size_bytes,
            name: None,
            external: false,
            alignment: 256,
        }
    }

    /// Marks this buffer as externally managed (builder pattern).
    #[must_use]
    pub fn external(mut self) -> Self {
        self.external = true;
        self
    }

    /// Sets the buffer's required alignment (builder pattern).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `align` is not a power of two or is zero.
    #[must_use]
    pub fn with_alignment(mut self, align: usize) -> Self {
        debug_assert!(
            align > 0 && align.is_power_of_two(),
            "alignment must be a non-zero power of two"
        );
        self.alignment = align;
        self
    }

    /// Sets a human-readable name (builder pattern).
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl fmt::Display for BufferDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.name.as_deref().unwrap_or("<unnamed>");
        let ext = if self.external { " [ext]" } else { "" };
        write!(
            f,
            "Buffer({}, {} \"{}\"{})",
            self.id, self.size_bytes, name, ext
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- NodeId, BufferId, StreamId ---

    #[test]
    fn node_id_display() {
        assert_eq!(NodeId(0).to_string(), "N0");
        assert_eq!(NodeId(42).to_string(), "N42");
    }

    #[test]
    fn buffer_id_display() {
        assert_eq!(BufferId(0).to_string(), "B0");
        assert_eq!(BufferId(7).to_string(), "B7");
    }

    #[test]
    fn stream_id_default_display() {
        assert_eq!(StreamId::DEFAULT.to_string(), "S:default");
        assert_eq!(StreamId(1).to_string(), "S1");
    }

    #[test]
    fn node_id_ordering() {
        assert!(NodeId(1) > NodeId(0));
        assert!(NodeId(0) < NodeId(100));
        let mut ids = vec![NodeId(3), NodeId(1), NodeId(2)];
        ids.sort();
        assert_eq!(ids, vec![NodeId(1), NodeId(2), NodeId(3)]);
    }

    // --- KernelConfig ---

    #[test]
    fn kernel_config_linear() {
        let cfg = KernelConfig::linear(4, 256, 1024);
        assert_eq!(cfg.grid, (4, 1, 1));
        assert_eq!(cfg.block, (256, 1, 1));
        assert_eq!(cfg.shared_mem_bytes, 1024);
    }

    #[test]
    fn kernel_config_total_threads() {
        let cfg = KernelConfig {
            grid: (4, 2, 1),
            block: (32, 4, 2),
            shared_mem_bytes: 0,
        };
        assert_eq!(cfg.total_threads(), 4 * 2 * 32 * 4 * 2);
    }

    #[test]
    fn kernel_config_total_blocks() {
        let cfg = KernelConfig {
            grid: (3, 3, 3),
            block: (8, 8, 8),
            shared_mem_bytes: 0,
        };
        assert_eq!(cfg.total_blocks(), 27);
    }

    #[test]
    fn kernel_config_display() {
        let cfg = KernelConfig::linear(8, 128, 0);
        let s = cfg.to_string();
        assert!(s.contains("grid=(8,1,1)"));
        assert!(s.contains("block=(128,1,1)"));
    }

    // --- MemcpyDir ---

    #[test]
    fn memcpy_dir_display() {
        assert_eq!(MemcpyDir::HostToDevice.to_string(), "HtoD");
        assert_eq!(MemcpyDir::DeviceToHost.to_string(), "DtoH");
        assert_eq!(MemcpyDir::DeviceToDevice.to_string(), "DtoD");
        let p2p = MemcpyDir::PeerToPeer {
            src_device: 0,
            dst_device: 1,
        };
        assert!(p2p.to_string().contains("PeerToPeer"));
    }

    // --- NodeKind ---

    #[test]
    fn node_kind_is_compute() {
        let k = NodeKind::KernelLaunch {
            function_name: "add".into(),
            config: KernelConfig::linear(1, 32, 0),
            fusible: true,
        };
        assert!(k.is_compute());
        assert!(
            !NodeKind::Memset {
                size_bytes: 1024,
                value: 0
            }
            .is_compute()
        );
    }

    #[test]
    fn node_kind_is_memory_op() {
        let m = NodeKind::Memcpy {
            dir: MemcpyDir::HostToDevice,
            size_bytes: 128,
        };
        assert!(m.is_memory_op());
        let ms = NodeKind::Memset {
            size_bytes: 64,
            value: 0xff,
        };
        assert!(ms.is_memory_op());
        let k = NodeKind::KernelLaunch {
            function_name: "k".into(),
            config: KernelConfig::linear(1, 32, 0),
            fusible: false,
        };
        assert!(!k.is_memory_op());
    }

    #[test]
    fn node_kind_is_fusible() {
        let f = NodeKind::KernelLaunch {
            function_name: "add".into(),
            config: KernelConfig::linear(1, 32, 0),
            fusible: true,
        };
        assert!(f.is_fusible());
        let nf = NodeKind::KernelLaunch {
            function_name: "custom".into(),
            config: KernelConfig::linear(1, 32, 0),
            fusible: false,
        };
        assert!(!nf.is_fusible());
        assert!(!NodeKind::Barrier.is_fusible());
    }

    #[test]
    fn node_kind_function_name() {
        let k = NodeKind::KernelLaunch {
            function_name: "scale".into(),
            config: KernelConfig::linear(1, 32, 0),
            fusible: false,
        };
        assert_eq!(k.function_name(), Some("scale"));
        assert_eq!(NodeKind::Barrier.function_name(), None);
    }

    #[test]
    fn node_kind_tag() {
        assert_eq!(
            NodeKind::KernelLaunch {
                function_name: "x".into(),
                config: KernelConfig::linear(1, 1, 0),
                fusible: false
            }
            .tag(),
            "kernel"
        );
        assert_eq!(
            NodeKind::Memcpy {
                dir: MemcpyDir::HostToDevice,
                size_bytes: 1
            }
            .tag(),
            "memcpy"
        );
        assert_eq!(
            NodeKind::Memset {
                size_bytes: 1,
                value: 0
            }
            .tag(),
            "memset"
        );
        assert_eq!(NodeKind::EventRecord.tag(), "event_record");
        assert_eq!(NodeKind::EventWait.tag(), "event_wait");
        assert_eq!(NodeKind::Barrier.tag(), "barrier");
        assert_eq!(
            NodeKind::HostCallback { label: "x".into() }.tag(),
            "host_cb"
        );
    }

    #[test]
    fn node_kind_display_kernel() {
        let k = NodeKind::KernelLaunch {
            function_name: "vector_add".into(),
            config: KernelConfig::linear(4, 256, 0),
            fusible: true,
        };
        let s = k.to_string();
        assert!(s.contains("vector_add"));
        assert!(s.contains("fusible"));
    }

    #[test]
    fn node_kind_display_memcpy() {
        let m = NodeKind::Memcpy {
            dir: MemcpyDir::DeviceToHost,
            size_bytes: 4096,
        };
        let s = m.to_string();
        assert!(s.contains("DtoH"));
        assert!(s.contains("4096"));
    }

    // --- GraphNode ---

    #[test]
    fn graph_node_new() {
        let n = GraphNode::new(NodeId(5), NodeKind::Barrier);
        assert_eq!(n.id, NodeId(5));
        assert!(n.inputs.is_empty());
        assert!(n.outputs.is_empty());
        assert!(n.stream_hint.is_none());
        assert_eq!(n.cost_hint, 1);
        assert!(n.name.is_none());
    }

    #[test]
    fn graph_node_builder_pattern() {
        let n = GraphNode::new(NodeId(0), NodeKind::Barrier)
            .with_inputs([BufferId(0), BufferId(1)])
            .with_outputs([BufferId(2)])
            .with_stream(StreamId(1))
            .with_cost(100)
            .with_name("barrier_join");
        assert_eq!(n.inputs.len(), 2);
        assert_eq!(n.outputs.len(), 1);
        assert_eq!(n.stream_hint, Some(StreamId(1)));
        assert_eq!(n.cost_hint, 100);
        assert_eq!(n.display_name(), "barrier_join");
    }

    #[test]
    fn graph_node_is_source_and_sink() {
        let source = GraphNode::new(NodeId(0), NodeKind::Barrier).with_outputs([BufferId(0)]);
        assert!(source.is_source());
        assert!(!source.is_sink());

        let sink = GraphNode::new(NodeId(1), NodeKind::Barrier).with_inputs([BufferId(0)]);
        assert!(!sink.is_source());
        assert!(sink.is_sink());
    }

    #[test]
    fn graph_node_display_name_fallback() {
        let n = GraphNode::new(NodeId(7), NodeKind::Barrier);
        assert_eq!(n.display_name(), "N7");
    }

    #[test]
    fn graph_node_display() {
        let n = GraphNode::new(
            NodeId(3),
            NodeKind::Memcpy {
                dir: MemcpyDir::HostToDevice,
                size_bytes: 512,
            },
        )
        .with_inputs([BufferId(0)])
        .with_outputs([BufferId(1)])
        .with_name("upload");
        let s = n.to_string();
        assert!(s.contains("upload"));
        assert!(s.contains("HtoD"));
        assert!(s.contains("B0"));
        assert!(s.contains("B1"));
    }

    // --- BufferDescriptor ---

    #[test]
    fn buffer_descriptor_new() {
        let b = BufferDescriptor::new(BufferId(0), 4096);
        assert_eq!(b.id, BufferId(0));
        assert_eq!(b.size_bytes, 4096);
        assert!(!b.external);
        assert_eq!(b.alignment, 256);
    }

    #[test]
    fn buffer_descriptor_builder() {
        let b = BufferDescriptor::new(BufferId(1), 1024)
            .external()
            .with_alignment(512)
            .with_name("weights");
        assert!(b.external);
        assert_eq!(b.alignment, 512);
        assert_eq!(b.name.as_deref(), Some("weights"));
    }

    #[test]
    fn buffer_descriptor_display() {
        let b = BufferDescriptor::new(BufferId(2), 8192).with_name("activations");
        let s = b.to_string();
        assert!(s.contains("B2"));
        assert!(s.contains("8192"));
        assert!(s.contains("activations"));
    }
}
