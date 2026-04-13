//! OxiCUDA Graph — CUDA Graph execution engine.
//!
//! `oxicuda-graph` provides a high-level, Rust-native computation graph
//! framework that sits above the raw CUDA driver API. It models GPU
//! workloads as directed acyclic graphs (DAGs) of operations, applies a
//! suite of optimisation passes, and lowers the result to an
//! `oxicuda_driver::graph::Graph` for low-overhead CUDA graph submission.
//!
//! # Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────────────┐
//!  │              GraphBuilder                   │  ← ergonomic API
//!  └──────────────────┬──────────────────────────┘
//!                     │ .build()
//!  ┌──────────────────▼──────────────────────────┐
//!  │             ComputeGraph                    │  ← DAG of GraphNodes
//!  └──────────────────┬──────────────────────────┘
//!           ┌─────────┴──────────┐
//!           │   Analysis Passes  │
//!           │  ┌─────────────┐   │
//!           │  │ topo        │   │  ASAP/ALAP, slack, levels
//!           │  │ liveness    │   │  buffer live intervals
//!           │  │ dominance   │   │  Lengauer–Tarjan dominator tree
//!           │  └─────────────┘   │
//!           └─────────┬──────────┘
//!           ┌─────────┴──────────┐
//!           │ Optimisation Passes│
//!           │  ┌─────────────┐   │
//!           │  │ fusion      │   │  element-wise kernel merging
//!           │  │ memory      │   │  interval-graph buffer colouring
//!           │  │ stream      │   │  multi-stream partitioning
//!           │  └─────────────┘   │
//!           └─────────┬──────────┘
//!  ┌──────────────────▼──────────────────────────┐
//!  │             ExecutionPlan                   │  ← ordered PlanSteps
//!  └──────────────────┬──────────────────────────┘
//!           ┌─────────┴──────────┐
//!           │   Executor Backends│
//!           │  ┌─────────────┐   │
//!           │  │ sequential  │   │  CPU simulator (testing)
//!           │  │ cuda_graph  │   │  driver::graph::Graph capture
//!           │  └─────────────┘   │
//!           └────────────────────┘
//! ```
//!
//! # Quick start
//!
//! ```rust
//! use oxicuda_graph::builder::GraphBuilder;
//! use oxicuda_graph::executor::{ExecutionPlan, SequentialExecutor};
//! use oxicuda_graph::node::MemcpyDir;
//!
//! // 1. Build a computation graph.
//! let mut b = GraphBuilder::new();
//!
//! let buf_in  = b.alloc_buffer("input",  4096);
//! let buf_mid = b.alloc_buffer("mid",    4096);
//! let buf_out = b.alloc_buffer("output", 4096);
//!
//! let upload   = b.add_memcpy("upload",   MemcpyDir::HostToDevice, 4096);
//! let k0       = b.add_kernel("relu",   4, 256, 0)
//!                 .fusible(true)
//!                 .inputs([buf_in])
//!                 .outputs([buf_mid])
//!                 .finish();
//! let k1       = b.add_kernel("scale",  4, 256, 0)
//!                 .fusible(true)
//!                 .inputs([buf_mid])
//!                 .outputs([buf_out])
//!                 .finish();
//! let download = b.add_memcpy("download", MemcpyDir::DeviceToHost, 4096);
//!
//! b.chain(&[upload, k0, k1, download]);
//!
//! let graph = b.build().unwrap();
//!
//! // 2. Compile to an execution plan.
//! let plan = ExecutionPlan::build(&graph, /*max_streams=*/ 4).unwrap();
//!
//! // 3. Execute (sequential CPU simulation).
//! let stats = SequentialExecutor::new(&plan).run().unwrap();
//! assert!(stats.kernels_launched <= 2);  // relu+scale may be fused to 1
//! ```

#![forbid(unsafe_code)]

pub mod analysis;
pub mod builder;
pub mod error;
pub mod executor;
pub mod graph;
pub mod node;
pub mod optimizer;

// Re-export the most commonly used types at the crate root.
pub use builder::GraphBuilder;
pub use error::{GraphError, GraphResult};
pub use executor::{ExecutionPlan, PlanStep, SequentialExecutor};
pub use graph::ComputeGraph;
pub use node::{BufferId, GraphNode, KernelConfig, MemcpyDir, NodeId, NodeKind, StreamId};
