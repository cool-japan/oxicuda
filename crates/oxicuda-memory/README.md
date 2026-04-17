# oxicuda-memory

Type-safe GPU memory management with Rust ownership semantics.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-memory` provides safe, RAII-based wrappers around CUDA memory
allocation and transfer operations. Every buffer type owns its GPU (or
pinned-host) allocation and automatically frees it on `Drop`, preventing
leaks without requiring manual cleanup.

The crate enforces compile-time type safety through generics (`T: Copy`)
and validates sizes at runtime, returning `CudaError::InvalidValue` for
mismatches rather than panicking. `Drop` implementations log errors via
`tracing::warn` instead of panicking, ensuring safe teardown even when
the CUDA context has already been destroyed.

The `copy` module provides freestanding transfer functions that mirror the
CUDA driver `cuMemcpy*` family (`copy_htod`, `copy_dtoh`, `copy_dtod`)
with both synchronous and async variants. For convenience, `DeviceBuffer`
also exposes methods like `copy_from_host()` and `copy_to_host()` directly.

## Modules

| Module               | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `device_buffer`      | `DeviceBuffer<T>` (VRAM) and `DeviceSlice<T>` (sub-range)         |
| `host_buffer`        | `PinnedBuffer<T>` -- page-locked host memory for fast DMA         |
| `unified`            | `UnifiedBuffer<T>` -- CUDA managed memory (host+device)           |
| `zero_copy`          | `MappedBuffer<T>` -- zero-copy host-mapped device-accessible mem  |
| `copy`               | Freestanding `copy_htod`, `copy_dtoh`, `copy_dtod` (sync + async) |
| `copy_2d3d`          | `Memcpy2DParams`, `Memcpy3DParams`, 2D/3D memory copy helpers      |
| `pool`               | `MemoryPool` -- stream-ordered allocation (behind `pool` feature)  |
| `pool_stats`         | `AllocationHistogram`, `PoolReport`, fragmentation metrics         |
| `aligned`            | Aligned allocation options (256-byte / 512-byte alignment)         |
| `buffer_view`        | Type-safe buffer view and reinterpret-cast helpers                 |
| `host_registered`    | Host-registered memory (`cuMemHostRegister`)                       |
| `managed_hints`      | CUDA 12+ managed memory hints, migration policy, prefetch plans    |
| `virtual_memory`     | `VirtualAddressRange`, `PhysicalAllocation`, virtual memory mgmt   |
| `peer_copy`          | Multi-GPU peer memory copy (`copy_peer`, `copy_peer_async`)        |
| `memory_info`        | Device memory info queries (`cuMemGetInfo`)                        |
| `bandwidth_profiler` | Transfer timing and throughput measurement hooks                   |

## Quick Start

```rust,no_run
use oxicuda_driver::prelude::*;
use oxicuda_memory::prelude::*;

init()?;
let dev = Device::get(0)?;
let _ctx = Context::new(&dev)?;

// Allocate a device buffer and upload host data.
let host_data = vec![1.0f32; 1024];
let mut gpu_buf = DeviceBuffer::<f32>::from_slice(&host_data)?;

// Download results back to the host.
let mut result = vec![0.0f32; 1024];
gpu_buf.copy_to_host(&mut result)?;
# Ok::<(), oxicuda_driver::CudaError>(())
```

## Buffer Types

| Type               | Location        | Description                            |
|--------------------|-----------------|----------------------------------------|
| `DeviceBuffer<T>`  | Device (VRAM)   | Primary GPU-side buffer                |
| `DeviceSlice<T>`   | Device (VRAM)   | Borrowed sub-range of a device buffer  |
| `PinnedBuffer<T>`  | Host (pinned)   | Page-locked host memory for fast DMA   |
| `UnifiedBuffer<T>` | Unified/managed | Accessible from both host and device   |
| `MappedBuffer<T>`  | Host-mapped     | Zero-copy host-mapped device-accessible memory |
| `MemoryPool`       | Device pool     | Stream-ordered allocation (CUDA 11.2+) |

## Features

| Feature     | Description                                       |
|-------------|---------------------------------------------------|
| `pool`      | Enable stream-ordered memory pool (CUDA 11.2+)    |
| `gpu-tests` | Enable integration tests that require a real GPU  |

## Platform Support

| Platform | Status                                        |
|----------|-----------------------------------------------|
| Linux    | Full support (NVIDIA driver 525+)             |
| Windows  | Full support (NVIDIA driver 525+)             |
| macOS    | Compile only (UnsupportedPlatform at runtime) |

## Status

| Item       | Value              |
|------------|--------------------|
| Version    | 0.1.3 (2026-04-17) |
| Tests      | 201 passing        |
| Warnings   | 0                  |
| `unwrap()` | 0                  |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
