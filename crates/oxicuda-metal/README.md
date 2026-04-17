# oxicuda-metal

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-metal` provides a `MetalBackend` that implements the `ComputeBackend` trait from `oxicuda-backend` using Apple's Metal API. It targets Apple Silicon and Intel Mac GPUs through Metal compute pipelines, enabling GPU-accelerated inference on macOS without any CUDA dependency. On non-macOS platforms the crate compiles cleanly and returns `UnsupportedPlatform` at runtime, keeping cross-platform workspaces unaffected.

## Features

- `MetalBackend` implementing the full `ComputeBackend` trait (GEMM, conv2D, attention, reduce, unary/binary ops, memory management)
- Shared-mode `MTLBuffer` pool via `MetalMemoryManager` for efficient host-visible GPU allocation
- `MetalDevice` abstraction for device enumeration and command queue management
- `msl` module with MSL source-string generators for GEMM, element-wise ops, and reductions — usable directly for custom kernel compilation
- `mps` module with Metal Performance Shaders (MPS) interop for hardware-accelerated GEMM, image convolution, and normalization via `MPSMatrixMultiplication`
- Metal compute `pipeline` module for shader compilation and dispatch
- Conditional macOS compilation: the `metal` crate is only linked on `target_os = "macos"`

## Platform Support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon / Intel) | Full support via Metal API |
| Linux / Windows | Compile-only; returns `UnsupportedPlatform` at runtime |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxicuda-metal = "0.1.3"
```

```rust
use oxicuda_metal::MetalBackend;
use oxicuda_backend::ComputeBackend;

let mut backend = MetalBackend::new();
backend.init()?;

let ptr = backend.alloc(256)?;
// ... copy data, launch Metal kernels ...
backend.free(ptr)?;
```

## Status

- **Version**: 0.1.3 (2026-04-17)
- **Tests**: 152 passing

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
