# oxicuda-blas

GPU-accelerated Basic Linear Algebra Subprograms (BLAS) -- a pure Rust cuBLAS equivalent.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-blas` provides a comprehensive BLAS library that generates and launches
GPU kernels entirely from Rust, with no C/Fortran dependencies. It covers all
three classical BLAS levels plus batched operations, elementwise transforms, and
reductions commonly needed by higher-level ML frameworks.

The GEMM engine is architecture-aware: it inspects the device SM version at
runtime and selects optimal tile sizes and instruction strategies for Turing,
Ampere, Ada Lovelace, Hopper, and Blackwell GPUs. Both SIMT and Tensor Core
paths are supported, with split-K parallelisation for tall-skinny matrices and
fused epilogue operations (bias, ReLU, GELU) to eliminate redundant memory
traffic.

Precision support spans the full spectrum from FP64 down to FP8 (E4M3/E5M2),
including TF32 and mixed-precision modes for training and inference workloads.

## Modules

| Module | Description |
|--------|-------------|
| `handle` | `BlasHandle` -- binds operations to a CUDA context and stream |
| `types` | `GpuFloat`, `MatrixDesc`, `VectorDesc`, layout/transpose enums |
| `level1` | Vector ops: axpy, scal, dot, nrm2, asum, iamax, copy, swap |
| `level2` | Matrix-vector ops: gemv, symv, trmv, trsv, ger, syr |
| `level3` | Matrix-matrix ops: GEMM, symm, trsm, syrk, syr2k, trmm |
| `batched` | batched_gemm, strided_gemm, grouped_gemm |
| `elementwise` | Unary (relu, gelu, sigmoid, silu, tanh) and binary (add, mul, scale) |
| `reduction` | sum, max, min, mean, variance, softmax (warp/block/multipass) |
| `precision` | Per-type PTX builders for f64, f32, f16, bf16, tf32, fp8 |
| `error` | `BlasError` / `BlasResult` |

## Quick Start

```rust,no_run
use std::sync::Arc;
use oxicuda_driver::Context;
use oxicuda_blas::prelude::*;

fn main() -> BlasResult<()> {
    // Obtain a CUDA context (see oxicuda-driver docs)
    let ctx: Arc<Context> = unimplemented!();

    // Create a BLAS handle bound to the context
    let handle = BlasHandle::new(&ctx)?;

    // Allocate device vectors, then call Level-1 routines:
    //   axpy(&handle, n, alpha, &x, 1, &mut y, 1)?;
    //   let nrm = nrm2(&handle, n, &x, 1)?;

    // Level-3 GEMM with automatic kernel dispatch:
    //   gemm_api::gemm(&handle, transa, transb, m, n, k,
    //                  alpha, &a, lda, &b, ldb, beta, &mut c, ldc)?;
    Ok(())
}
```

## Supported Operations

### BLAS Level 1 (vector-vector)

`axpy`, `scal`, `dot`, `nrm2`, `asum`, `iamax`, `copy`, `swap`

### BLAS Level 2 (matrix-vector)

`gemv`, `symv`, `trmv`, `trsv`, `ger`, `syr`

### BLAS Level 3 (matrix-matrix)

| Operation | Notes |
|-----------|-------|
| GEMM | SIMT + Tensor Core dispatcher, architecture-aware tile selection |
| symm | Symmetric matrix multiply |
| trsm | Triangular solve (multiple RHS) |
| syrk / syr2k | Symmetric rank-k / rank-2k update |
| trmm | Triangular matrix multiply |

### Batched GEMM

`batched_gemm`, `strided_gemm`, `grouped_gemm`

### GEMM Advanced Features

- Split-K parallelisation for large K dimensions
- Epilogue fusion: `LinearCombination`, `+ReLU`, `+GELU`, `+Bias`
- Architecture-specific tiles: Turing / Ampere / Ada / Hopper / Blackwell

## Feature Flags

| Feature | Description |
|---------|-------------|
| `f16` | Enable FP16 / BF16 support via the `half` crate |
| `tensor-core` | Enable Tensor Core GEMM paths |
| `all-precisions` | Shorthand for `f16` (and future precision gates) |

## Precision Support

| Type | Status |
|------|--------|
| f64 | Default |
| f32 | Default |
| f16 / bf16 | Behind `f16` feature |
| tf32 | Default (requires Ampere+) |
| fp8 E4M3 / E5M2 | Default (requires Hopper+) |
| Mixed precision | Automatic promotion / demotion |

## Performance Targets

The GEMM engine targets 95% of cuBLAS throughput on square matrices at
representative sizes (M=N=K in 512..8192) across supported architectures.
Batched and reduction kernels aim for comparable device occupancy.

## Status

| Item | Value |
|------|-------|
| Version | 0.1.4 |
| Release date | 2026-04-18 |
| Tests | 645 passing |
| Warnings | 0 (clippy clean) |
| `unwrap()` | 0 (production code) |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
