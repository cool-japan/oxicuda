# oxicuda-launch

Type-safe GPU kernel launch infrastructure for the OxiCUDA ecosystem.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-launch` provides ergonomic, type-safe abstractions for launching
CUDA GPU kernels. It builds on top of `oxicuda-driver` and `oxicuda-memory`
to turn raw `cuLaunchKernel` calls into safe, builder-pattern Rust code.

The `Kernel` struct wraps a `Function` handle and holds an `Arc<Module>` to
ensure the PTX module remains loaded for the kernel's lifetime. Arguments
are passed through the `KernelArgs` trait, which converts typed Rust values
into the `*mut c_void` array that the driver expects. Tuple implementations
are provided for up to 24 `Copy` elements, covering virtually all practical
kernel signatures.

The `launch!` macro offers a concise syntax for kernel dispatch, while
`LaunchParams` and its builder provide explicit control over grid size,
block size, shared memory, and stream selection. The `grid_size_for(n,
block_size)` helper computes the minimum number of blocks needed to cover
a given number of work items.

## Modules

| Module   | Description                                                   |
|----------|---------------------------------------------------------------|
| `grid`   | `Dim3` struct, `grid_size_for()` helper                       |
| `params` | `LaunchParams` and `LaunchParamsBuilder` (builder pattern)    |
| `kernel` | `Kernel` struct, `KernelArgs` trait (tuple impls up to 24)    |
| `macros` | `launch!` macro for ergonomic kernel dispatch                 |

## Quick Start

```rust,no_run
use std::sync::Arc;
use oxicuda_driver::{init, Device, Context, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams, Dim3, grid_size_for, launch};

init()?;
let dev = Device::get(0)?;
let ctx = Arc::new(Context::new(&dev)?);

// Load PTX and create a kernel.
let module = Arc::new(Module::from_ptx(ptx_source)?);
let kernel = Kernel::from_module(module, "vector_add")?;

// Configure launch dimensions.
let n: u32 = 1024;
let block_size = 256u32;
let grid = grid_size_for(n, block_size);
let stream = Stream::new(&ctx)?;

// Launch with the macro.
let (a_ptr, b_ptr, c_ptr) = (0u64, 0u64, 0u64);
launch!(kernel, grid(grid), block(block_size), &stream, &(a_ptr, b_ptr, c_ptr, n))?;
stream.synchronize()?;
# Ok::<(), oxicuda_driver::CudaError>(())
```

## Dim3 Conversions

`Dim3` supports convenient `From` conversions for common dimension patterns:

| Input type              | Result                 |
|-------------------------|------------------------|
| `u32`                   | `Dim3 { x, y: 1, z: 1 }` |
| `(u32, u32)`            | `Dim3 { x, y, z: 1 }`    |
| `(u32, u32, u32)`       | `Dim3 { x, y, z }`       |

## Features

| Feature     | Description                               |
|-------------|-------------------------------------------|
| `gpu-tests` | Enable tests that require a physical GPU  |

## Platform Support

| Platform | Status                                        |
|----------|-----------------------------------------------|
| Linux    | Full support (NVIDIA driver 525+)             |
| Windows  | Full support (NVIDIA driver 525+)             |
| macOS    | Compile only (UnsupportedPlatform at runtime) |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
