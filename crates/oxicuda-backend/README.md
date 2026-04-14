# oxicuda-backend

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-backend` defines the `ComputeBackend` trait — a unified, object-safe abstraction over GPU compute APIs (CUDA, ROCm, Metal, Level Zero). Higher-level crates such as SciRS2, ToRSh, and oxionnx program against this trait rather than any specific GPU API, enabling transparent backend switching at runtime without recompilation.

## Features

- Object-safe `ComputeBackend` trait usable as `Box<dyn ComputeBackend>` or `&dyn ComputeBackend`
- General matrix multiply (`gemm`), 2D convolution (`conv2d_forward`), and scaled dot-product attention
- Element-wise unary and binary operations (ReLU, sigmoid, tanh, exp, log, sqrt, abs, neg; add, sub, mul, div, max, min)
- Reduction operations along any axis (sum, max, min, mean)
- Device memory management: `alloc`, `free`, `copy_htod`, `copy_dtoh`, `synchronize`
- Rich error type (`BackendError`) covering unsupported ops, device errors, OOM, and uninitialized state
- Zero external dependencies — built entirely on `std`

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxicuda-backend = "0.1.1"
```

```rust
use oxicuda_backend::{ComputeBackend, BackendTranspose, BackendResult};

fn run_gemm(backend: &dyn ComputeBackend) -> BackendResult<()> {
    let a = backend.alloc(64 * 8)?;   // 64×8 f64 matrix
    let b = backend.alloc(8 * 32)?;   // 8×32 f64 matrix
    let c = backend.alloc(64 * 32)?;  // output 64×32

    backend.gemm(
        BackendTranspose::NoTrans, BackendTranspose::NoTrans,
        64, 32, 8,
        1.0, a, 64, b, 8, 0.0, c, 64,
    )?;
    backend.synchronize()?;
    backend.free(a)?;
    backend.free(b)?;
    backend.free(c)?;
    Ok(())
}
```

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
