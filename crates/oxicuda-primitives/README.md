# oxicuda-primitives

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-primitives` is the CUB-equivalent library for OxiCUDA, providing high-performance parallel GPU primitives — radix sort, merge sort, scan, reduce, select, and histogram — implemented as PTX code generators. All kernels are produced as PTX source strings at runtime and JIT-compiled via `cuModuleLoadData`, with zero dependency on the CUDA SDK at build time.

## Features

- **Warp primitives** (`warp`): warp-level reduce and inclusive/exclusive scan using `shfl.sync.*` shuffle instructions
- **Block primitives** (`block`): block-level reduce and scan via shared-memory ping-pong buffers
- **Device-wide algorithms** (`device`): multi-pass device reduce, scan, select (stream compaction), and histogram
- **Sort** (`sort`): device-wide radix sort (digit-by-digit, configurable radix bits) and merge sort
- **PTX helpers** (`ptx_helpers`): shared utilities for PTX code generation, type mapping, and operation encoding
- Configurable via `DeviceReduceConfig`, `ScanConfig`, `SortConfig`, etc. — choose data type, operation, and tuning parameters at runtime
- Supports SM75 through SM120 (Turing, Ampere, Hopper, Blackwell)

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxicuda-primitives = "0.1.3"
```

```rust
use oxicuda_primitives::device::reduce::{DeviceReduceConfig, DeviceReduceTemplate};
use oxicuda_primitives::ptx_helpers::ReduceOp;
use oxicuda_ptx::ir::PtxType;
use oxicuda_ptx::arch::SmVersion;

let cfg = DeviceReduceConfig::new(ReduceOp::Sum, PtxType::F32);
let (pass1_ptx, pass2_ptx) = DeviceReduceTemplate::new(cfg)
    .generate(SmVersion::Sm80)?;

// JIT-compile pass1_ptx and pass2_ptx via the CUDA driver API.
assert!(pass1_ptx.contains("device_reduce_pass1_sum_f32"));
```

## Status

- **Version**: 0.1.3 (2026-04-17)
- **Tests**: 142 passing

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
