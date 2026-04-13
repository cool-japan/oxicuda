# oxicuda-driver

Dynamic, safe Rust bindings for the NVIDIA CUDA Driver API.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-driver` is a pure Rust wrapper around the CUDA Driver API (`cuda.h`).
Unlike traditional approaches that require the CUDA Toolkit at build time,
this crate loads the driver shared library entirely at **runtime** via
[`libloading`](https://crates.io/crates/libloading). No `cuda.h`, no
`libcuda.so` symlink, no `nvcc` -- the crate compiles on any standard Rust
toolchain.

The actual GPU driver is discovered the first time you call `init()` or
`try_driver()`. A global `OnceLock` singleton caches the loaded function
pointers for the lifetime of the process, so subsequent calls are
essentially free.

All public APIs return `CudaResult<T>` rather than panicking. The
`CudaError` enum covers roughly 100 CUDA driver error codes as
strongly-typed variants, making match-based error handling straightforward.
RAII wrappers (`Context`, `Stream`, `Event`, `Module`) automatically release
GPU resources on `Drop`.

## Modules

| Module      | Description                                                  |
|-------------|--------------------------------------------------------------|
| `ffi`       | Raw C-compatible types (`CUdevice`, `CUcontext`, etc.)       |
| `error`     | `CudaError` (~100 variants), `CudaResult`, `check()` helper |
| `loader`    | Runtime library loading with `OnceLock` singleton            |
| `device`    | Device enumeration, attribute queries, `best_device()`       |
| `context`   | RAII CUDA context bound to a device                          |
| `stream`    | Asynchronous command queue within a context                  |
| `event`     | Timing and synchronisation markers on streams                |
| `module`    | PTX/cubin loading, JIT compilation, function lookup          |
| `occupancy` | Occupancy-based launch configuration queries                 |

## Quick Start

```rust,no_run
use oxicuda_driver::prelude::*;

// Initialise the CUDA driver (loads libcuda at runtime).
init()?;

// Pick the first available GPU and create a context.
let dev = Device::get(0)?;
let _ctx = Context::new(&dev)?;

// Load a PTX module and look up a kernel.
let module = Module::from_ptx(ptx_source)?;
let kernel = module.get_function("vector_add")?;
# Ok::<(), oxicuda_driver::CudaError>(())
```

## Features

| Feature     | Description                               |
|-------------|-------------------------------------------|
| `gpu-tests` | Enable tests that require a physical GPU  |

## Runtime Library Resolution

| Platform | Library searched               |
|----------|--------------------------------|
| Linux    | `libcuda.so`, `libcuda.so.1`   |
| Windows  | `nvcuda.dll`                   |
| macOS    | *(returns `UnsupportedPlatform` at runtime -- NVIDIA dropped macOS support)* |

## Platform Support

| Platform | Status                                     |
|----------|--------------------------------------------|
| Linux    | Full support (NVIDIA driver 525+)          |
| Windows  | Full support (NVIDIA driver 525+)          |
| macOS    | Compile only (UnsupportedPlatform at runtime) |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
