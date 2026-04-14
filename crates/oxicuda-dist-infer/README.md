# oxicuda-dist-infer

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-dist-infer` (Vol.12) is a production-grade distributed GPU inference engine for large language models. It implements three orthogonal parallelism strategies — tensor parallelism (TP), sequence parallelism (SP), and expert parallelism (EP) — along with a distributed KV cache and affinity-aware request routing to efficiently scale LLM serving across GPU clusters.

## Features

- **Tensor parallelism**: `ColumnLinear` and `RowLinear` shards with all-gather and all-reduce collectives
- **Sequence parallelism**: `SeqSplitter` for chunked sequence extraction, `BoundaryExchange` for cross-rank attention
- **Expert parallelism**: `TopKRouter` (softmax top-K selection) and `ExpertDispatcher` (scatter/gather) for Mixture-of-Experts models
- **Distributed KV cache**: `CachePartition` with least-loaded block assignment, sequence migration, and rebalancing
- **Request routing**: `RoutingPolicy` supporting round-robin, least-loaded, and prefix-affinity dispatch
- **PTX kernels**: Five ready-to-JIT-compile PTX kernel strings covering all parallelism axes (SM75–SM120)
- `world_size = tp × sp × ep` — degrees multiply cleanly for flexible cluster configurations

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxicuda-dist-infer = "0.1.2"
```

```rust
use oxicuda_dist_infer::{DistInferHandle, ParallelismConfig, SmVersion};
use oxicuda_dist_infer::tensor_parallel::column_parallel::ColumnLinear;

// Create a rank-0 handle for 4-way tensor parallelism.
let handle = DistInferHandle::new(
    0,                      // device_id
    SmVersion(80),          // SM version (A100)
    0,                      // rank
    ParallelismConfig { tp: 4, sp: 1, ep: 1 },
)?;

// Shard a weight matrix across 4 GPUs.
let full_weight = vec![1.0_f32; 512 * 512];
let layer = ColumnLinear::from_full_weight(handle, 512, 512, &full_weight, None)?;
let output = layer.local_forward(&input, batch_size)?;
```

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
