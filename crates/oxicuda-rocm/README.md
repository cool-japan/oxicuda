# oxicuda-rocm

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-rocm` is the AMD ROCm/HIP backend for OxiCUDA. It implements the `ComputeBackend` trait using the AMD HIP runtime (`libamdhip64.so`), loaded dynamically at runtime via `libloading` — no compile-time SDK dependency is required. The crate provides device enumeration, memory management, and kernel dispatch for AMD GPUs on Linux.

## Features

- **Dynamic HIP loading** — `libamdhip64.so` is loaded at runtime; no HIP SDK needed at compile time
- **hipRTC integration** — runtime compilation of HIP kernel source strings via `libhiprtc.so`; no ROCm toolchain required at compile time
- **hipBLAS interop** — optional high-performance BLAS via `libhipblas.so` with automatic kernel fallback when not installed
- **Device management** — Enumerate and select AMD GPU devices
- **Memory operations** — Allocate, free, and transfer device memory
- **Multi-GPU support** — `MultiDeviceDispatcher` distributes matrix work across all available AMD GPUs
- **Backend abstraction** — Implements `oxicuda-backend`'s `ComputeBackend` trait for unified multi-GPU-API programming
- **Pure Rust** — Zero C/Fortran in the default feature set

## Platform Support

| Platform       | Status                                      |
|----------------|---------------------------------------------|
| Linux (AMD GPU)| Full support via `libamdhip64.so`           |
| Windows        | Not supported (`UnsupportedPlatform`)       |
| macOS          | Not supported (`UnsupportedPlatform`)       |

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
oxicuda-rocm = "0.1.3"
```

```rust
use oxicuda_rocm::RocmBackend;

let backend = RocmBackend::new();
match backend.init() {
    Ok(()) => println!("ROCm backend initialised"),
    Err(e) => println!("ROCm not available: {e}"),
}
```

## Status

- **Version**: 0.1.3 (2026-04-17)
- **Tests**: 104 passing

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
