# oxicuda-runtime

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-runtime` is a Pure Rust implementation of the CUDA Runtime API (`libcudart`) surface, built on top of `oxicuda-driver`'s dynamic driver loader. It exposes ergonomic Rust types for streams, events, device pointers, and kernel dimensions, covering the full breadth of commonly used `cudaXxx` functions — without any CUDA SDK build-time dependency.

## Features

- **Device management** — `cudaGetDeviceCount`, `cudaSetDevice`, `cudaGetDeviceProperties`, `cudaDeviceSynchronize`, `cudaDeviceReset`
- **Memory** — `cudaMalloc`, `cudaFree`, `cudaMallocHost`, `cudaMallocManaged`, `cudaMallocPitch`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`, `cudaMemGetInfo`
- **Streams** — Create/destroy/synchronize CUDA streams with priority and flag support
- **Events** — `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`
- **Kernel launch** — `cudaLaunchKernel`, `cudaFuncGetAttributes`, PTX module loading/unloading
- **Peer access** — Multi-GPU peer memory copies and access enable/disable
- **Profiler** — `cudaProfilerStart`/`cudaProfilerStop` with a `ProfilerGuard` RAII wrapper
- **Texture & surface objects** — Full `CudaTextureObject`, `CudaSurfaceObject`, array descriptors

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
oxicuda-runtime = "0.1.0"
```

```rust
use oxicuda_runtime::{device, memory, stream, event};

// Select device 0
device::set_device(0)?;

// Allocate 1 MiB of device memory and zero it
let d_buf = memory::malloc(1 << 20)?;
memory::memset(d_buf, 0, 1 << 20)?;

// Create a stream and record an event
let s = stream::stream_create()?;
let e = event::event_create()?;
event::event_record(e, s)?;
event::event_synchronize(e)?;

// Cleanup
event::event_destroy(e)?;
stream::stream_destroy(s)?;
memory::free(d_buf)?;
```

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
