# oxicuda-sparse

GPU-accelerated sparse matrix operations -- pure Rust cuSPARSE equivalent.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-sparse` provides a full suite of GPU-accelerated sparse matrix operations
implemented entirely in Rust, targeting feature parity with NVIDIA's cuSPARSE. It
supports five standard sparse storage formats with efficient conversions between them,
and offers all the key sparse linear algebra primitives needed for scientific computing
and machine learning workloads.

The SpMV implementation includes multiple algorithm variants -- scalar, vector, and
adaptive -- allowing the autotuner or user to select the best strategy for a given
sparsity pattern. SpGEMM uses a two-phase symbolic + numeric approach to minimize
memory allocation overhead. SpTRSV employs level-set parallelism for efficient
triangular solves on the GPU.

Preconditioners (ILU(0) and IC(0)) are included for use with iterative solvers
in `oxicuda-solver`. The `SparseHandle` integrates with the BLAS handle from
`oxicuda-blas` for seamless interoperability.

## Sparse Formats

| Format      | Type         | Description                          |
|-------------|--------------|--------------------------------------|
| `CsrMatrix` | CSR          | Compressed Sparse Row (primary)      |
| `CscMatrix` | CSC          | Compressed Sparse Column             |
| `CooMatrix` | COO          | Coordinate (triplet) format          |
| `BsrMatrix` | BSR          | Block Sparse Row                     |
| `EllMatrix` | ELL          | ELLPACK (fixed entries per row)      |

Format conversions are available in the `format::convert` module (CSR to/from CSC, COO, BSR, ELL).

## Operations

| Operation | Function   | Description                                      |
|-----------|------------|--------------------------------------------------|
| SpMV      | `ops::spmv` | Sparse matrix-vector multiply: y = alpha*A*x + beta*y |
| SpMM      | `ops::spmm` | Sparse-dense matrix multiply: C = alpha*A*B + beta*C  |
| SpGEMM    | `ops::spgemm` | Sparse-sparse multiply: C = A*B (two-phase)      |
| SpTRSV    | `ops::sptrsv` | Sparse triangular solve: L*x = b or U*x = b      |
| SDDMM     | `ops::sddmm` | Sampled Dense-Dense Matrix Multiply               |

## Preconditioners

- **ILU(0)** -- Incomplete LU factorization (zero fill-in) for general systems
- **IC(0)** -- Incomplete Cholesky factorization for symmetric positive definite systems

## Quick Start

```rust,no_run
use oxicuda_sparse::prelude::*;

// Create a CSR matrix from raw arrays
let row_ptr = vec![0u32, 2, 4, 6];
let col_idx = vec![0u32, 1, 0, 1, 0, 1];
let values  = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
let csr = CsrMatrix::new(3, 2, row_ptr, col_idx, values);

// With a GPU context:
// let handle = SparseHandle::new(&ctx)?;
// spmv(&handle, 1.0, &csr_gpu, &x, 0.0, &mut y, SpMVAlgo::Vector)?;
```

## Feature Flags

| Feature | Description                           |
|---------|---------------------------------------|
| `f16`   | Half-precision (fp16) sparse support  |

## Status

| Metric | Value |
|--------|-------|
| Version | 0.1.3 |
| Tests passing | 322 |
| Release date | 2026-04-17 |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
