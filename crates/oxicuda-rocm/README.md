# oxicuda-rocm

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-rocm` is the AMD ROCm/HIP backend for OxiCUDA. It implements the `ComputeBackend` trait using the AMD HIP runtime (`libamdhip64.so`), loaded dynamically at runtime via `libloading` — no compile-time SDK dependency is required. The crate provides device enumeration, memory management, and kernel dispatch for AMD GPUs on Linux.

## Features

- **Dynamic HIP loading** — `libamdhip64.so` is loaded at runtime; no HIP SDK needed at compile time
- **hipRTC integration** — runtime compilation of HIP kernel source strings via `libhiprtc.so`; no ROCm toolchain required at compile time
- **hipBLAS / hipBLASLt interop** — optional high-performance BLAS via `libhipblas.so` / `libhipblaslt.so` (fused-GEMM epilogues) with automatic kernel fallback when not installed
- **HIP kernel codegen** — elementwise, GEMM (naive / LDS-tiled / wave-aware), reduction, softmax, LayerNorm, prefix-scan, transpose, attention, conv2d, plus MFMA/WMMA/FP8 matrix-core micro-kernels
- **`gfx*` architecture model** — per-arch VGPR/SGPR/LDS limits, wavefront width, matrix-core capability tables (CDNA1/2/3, RDNA2/3)
- **Host-side planners** — occupancy calculator, launch-config validation, stream/event ordering, hipGraph DAG, memory-pool suballocator, xGMI peer topology
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
oxicuda-rocm = "0.4.1"
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

- **Version**: 0.5.2 (2026-07-27)
- **Tests**: 217 passing (host-side / codegen; device-execution paths require AMD ROCm hardware)

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
