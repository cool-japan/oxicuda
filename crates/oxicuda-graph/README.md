# oxicuda-graph

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-graph` is a high-level CUDA Graph execution engine that models GPU workloads as directed acyclic graphs (DAGs) of operations. It applies analysis and optimisation passes — operator fusion, buffer lifetime colouring, and multi-stream partitioning — and lowers the result to an `oxicuda_driver::graph::Graph` for low-overhead CUDA Graph submission.

## Features

- Ergonomic `GraphBuilder` API for constructing kernel, memcpy, and memset nodes with explicit buffer dependencies
- **Analysis passes**: topological ordering (ASAP/ALAP scheduling, slack, levels), buffer liveness intervals, and a Lengauer–Tarjan dominator tree
- **Optimisation passes**: element-wise kernel fusion, interval-graph buffer colouring (minimises peak allocation), and multi-stream partitioning
- `ExecutionPlan` lowering to ordered `PlanStep` sequences ready for `SequentialExecutor` (CPU simulation) or `CudaGraphExecutor`
- Execution statistics: kernels launched, nodes visited, fused ops count
- `#![forbid(unsafe_code)]` — fully safe Rust

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxicuda-graph = "0.1.3"
```

```rust
use oxicuda_graph::builder::GraphBuilder;
use oxicuda_graph::executor::{ExecutionPlan, SequentialExecutor};
use oxicuda_graph::node::MemcpyDir;

let mut b = GraphBuilder::new();
let buf_in  = b.alloc_buffer("input",  4096);
let buf_out = b.alloc_buffer("output", 4096);
let upload  = b.add_memcpy("upload", MemcpyDir::HostToDevice, 4096);
let kernel  = b.add_kernel("relu", 4, 256, 0)
               .fusible(true).inputs([buf_in]).outputs([buf_out]).finish();
let download = b.add_memcpy("download", MemcpyDir::DeviceToHost, 4096);
b.chain(&[upload, kernel, download]);

let graph = b.build().unwrap();
let plan  = ExecutionPlan::build(&graph, 4).unwrap();
let stats = SequentialExecutor::new(&plan).run().unwrap();
println!("kernels launched: {}", stats.kernels_launched);
```

## Status

**v0.1.3** (2026-04-17) — 175 tests passing

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
