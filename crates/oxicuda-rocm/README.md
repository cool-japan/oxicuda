# oxicuda-rocm

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-rocm` is the AMD ROCm/HIP backend for OxiCUDA. It implements the `ComputeBackend` trait using the AMD HIP runtime (`libamdhip64.so`), loaded dynamically at runtime via `libloading` — no compile-time SDK dependency is required. The crate provides device enumeration, memory management, and kernel dispatch for AMD GPUs on Linux.

## Features

- **Dynamic HIP loading** — `libamdhip64.so` is loaded at runtime; no HIP SDK needed at compile time
- **Device management** — Enumerate and select AMD GPU devices
- **Memory operations** — Allocate, free, and transfer device memory
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
oxicuda-rocm = "0.1.0"
```

```rust
use oxicuda_rocm::RocmBackend;

let backend = RocmBackend::new();
match backend.init() {
    Ok(()) => println!("ROCm backend initialised"),
    Err(e) => println!("ROCm not available: {e}"),
}
```

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
