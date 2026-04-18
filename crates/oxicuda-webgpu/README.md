# oxicuda-webgpu

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-webgpu` is the cross-platform GPU compute backend for OxiCUDA, targeting Vulkan, Metal, Direct3D 12, and the browser WebGPU API from a single Rust crate via `wgpu`. It implements the `ComputeBackend` trait from `oxicuda-backend` and provides a `WebGpuBackend` with a pooled buffer allocator and a WGSL shader generator for common kernels such as GEMM, element-wise ops, and reductions.

## Features

- **Cross-platform** — Single backend targets Vulkan (Linux/Windows), Metal (macOS/iOS), Direct3D 12 (Windows), and browser WebGPU via `wgpu`
- **WGSL shader generation** — The `shader` module generates WGSL source strings at runtime for GEMM, element-wise, and reduction kernels
- **Buffer pool** — `WebGpuMemoryManager` maintains a `u64` handle → `wgpu::Buffer` map for efficient allocation and reuse
- **Backend abstraction** — Implements `oxicuda-backend`'s `ComputeBackend` for interoperability with the rest of the OxiCUDA ecosystem
- **Pure Rust** — Zero C/Fortran in the default feature set; `wgpu` handles all native API calls

## Architecture

```
WebGpuBackend  (implements ComputeBackend)
      │
  WebGpuDevice       ← wgpu Instance + Adapter + Device + Queue
      │
  WebGpuMemoryManager  ← buffer pool (u64 handle → wgpu::Buffer)
```

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
oxicuda-webgpu = "0.1.4"
```

```rust
use oxicuda_webgpu::WebGpuBackend;
use oxicuda_backend::ComputeBackend;

let mut backend = WebGpuBackend::new();
backend.init().expect("WebGPU init failed");

// Allocate 1 KiB on the GPU
let ptr = backend.alloc(1024).expect("alloc failed");
backend.free(ptr).expect("free failed");
```

## Status

- **Version**: 0.1.4 (2026-04-18)
- **Tests**: 91 passing

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
