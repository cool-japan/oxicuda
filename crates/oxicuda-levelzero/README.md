# oxicuda-levelzero

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-levelzero` provides a `LevelZeroBackend` that implements the `ComputeBackend` trait from `oxicuda-backend` using Intel's oneAPI Level Zero API. It targets Intel Arc, Xe, and integrated GPUs on Linux and Windows by loading `libze_loader.so` / `ze_loader.dll` at runtime via `libloading`, with no C SDK dependency at compile time.

## Features

- `LevelZeroBackend` implementing the full `ComputeBackend` trait (GEMM, conv2D, attention, reduce, unary/binary ops, memory management)
- Runtime driver loading via `libloading` — zero link-time dependency on the Intel GPU driver
- SPIR-V kernel support via the `spirv` module for cross-vendor shader compilation
- Intel Xe Matrix Extensions (XMX) SPIR-V generators via the `spirv_xmx` module — cooperative-matrix GEMM for Xe/Arc/Ponte Vecchio GPUs
- Sub-group optimized kernels via the `spirv_subgroup` module — warp-level reduction, scan, and GEMM using `GroupNonUniform` opcodes
- Neural-network SPIR-V kernels via the `spirv_nn` module — Conv2D and scaled dot-product attention
- Multi-tile / multi-device dispatch via the `multi_tile` module — distributes workloads across Intel Max series GPU sub-device tiles
- Device enumeration and selection via the `device` module
- Memory management with unified shared memory (USM) via the `memory` module
- Compiles on all platforms; returns `UnsupportedPlatform` on macOS so cross-platform workspaces are unaffected

## Platform Support

| Platform | Status |
|----------|--------|
| Linux (Intel GPU) | Full support via `libze_loader.so` |
| Windows (Intel GPU) | Full support via `ze_loader.dll` |
| macOS | Compile-only; returns `UnsupportedPlatform` at runtime |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxicuda-levelzero = "0.1.4"
```

```rust
use oxicuda_levelzero::LevelZeroBackend;
use oxicuda_backend::ComputeBackend;

let mut backend = LevelZeroBackend::new();
backend.init()?;

let ptr = backend.alloc(1024)?;
// ... copy data, run kernels ...
backend.free(ptr)?;
```

## Status

- **Version**: 0.1.4 (2026-04-18)
- **Tests**: 104 passing

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
