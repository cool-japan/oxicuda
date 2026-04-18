# oxicuda-solver

GPU-accelerated matrix decompositions and linear solvers -- pure Rust cuSOLVER equivalent.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-solver` provides GPU-accelerated dense matrix decompositions and iterative
sparse solvers implemented entirely in Rust, targeting feature parity with NVIDIA's
cuSOLVER. The dense solvers cover all the fundamental factorizations -- LU, QR,
Cholesky, SVD, and eigendecomposition -- plus derived operations like matrix inverse,
determinant, and least squares.

The dense LU uses a blocked right-looking algorithm with partial pivoting. QR employs
blocked Householder reflections. SVD combines Jacobi rotations with Golub-Kahan
bidiagonalization. Eigendecomposition proceeds via tridiagonalization followed by
QR iteration. Condition number estimation uses Hager's algorithm.

Iterative sparse solvers include CG (for SPD systems), BiCGSTAB (for non-symmetric
systems), and GMRES with restart. These solvers accept matrix-free closures for SpMV,
making them composable with any sparse format or custom operator. A direct sparse
solver via dense LU is also provided for small systems.

## Dense Decompositions

| Decomposition     | Module          | Description                            |
|-------------------|-----------------|----------------------------------------|
| LU                | `dense::lu`     | P*A = L*U with partial pivoting        |
| QR                | `dense::qr`     | A = Q*R via blocked Householder        |
| Cholesky          | `dense::cholesky` | A = L*L^T for SPD matrices           |
| SVD               | `dense::svd`    | A = U*S*V^T (Jacobi + bidiag)         |
| Eigendecomposition| `dense::eig`    | A = Q*L*Q^T for symmetric matrices     |
| Inverse           | `dense::inverse` | A^{-1} via LU factorization          |
| Determinant       | `dense::det`    | det(A) and log-det via LU              |
| Least Squares     | `dense::lstsq`  | min \|\|Ax - b\|\| via QR             |

## Iterative Sparse Solvers

| Solver    | Module            | Applicable Systems                     |
|-----------|-------------------|----------------------------------------|
| CG        | `sparse::cg`      | Symmetric positive definite            |
| BiCGSTAB  | `sparse::bicgstab` | General non-symmetric                 |
| GMRES(m)  | `sparse::gmres`   | General, with restart parameter m      |
| Direct LU | `sparse::direct`  | Small sparse systems via dense LU      |

## Helpers

- **Pivot selection** (`helpers::pivot`) -- Partial and complete pivoting strategies
- **Condition number** (`helpers::condition`) -- Hager's 1-norm condition estimator

## Quick Start

```rust,no_run
use oxicuda_solver::prelude::*;

// With a GPU context:
// let handle = SolverHandle::new(&ctx)?;
//
// // LU decomposition of a 4x4 matrix
// let lu = handle.lu(&matrix, 4)?;
//
// // SVD
// let svd = handle.svd(&matrix, 4, 4, SvdJob::All)?;
//
// // Iterative solve (CG)
// let x = handle.cg(|v, out| spmv(a, v, out), &b, 1e-6, 1000)?;
```

## Feature Flags

| Feature          | Description                              |
|------------------|------------------------------------------|
| `f16`            | Half-precision (fp16) solver support     |
| `sparse-solvers` | Enable sparse iterative solvers (pulls in `oxicuda-sparse`) |

## Status

| Metric | Value |
|--------|-------|
| Version | 0.1.4 |
| Tests passing | 387 |
| Release date | 2026-04-18 |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
