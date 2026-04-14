# oxicuda-solver TODO

GPU-accelerated matrix decompositions and linear solvers, serving as a pure Rust equivalent to NVIDIA's cuSOLVER library. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.5).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 15,804 SLoC (40 files) -- Estimated: 76K-122K SLoC (estimation.md Vol.5 solver portion)**

Current implementation covers eight dense decompositions (LU, QR, Cholesky, SVD, eigendecomposition, inverse, determinant, least squares), four iterative sparse solvers (CG, BiCGSTAB, GMRES, direct), and helper utilities (pivoting, condition number estimation).

### Completed

- [x] Error handling (error.rs) -- SolverError, SolverResult<T>
- [x] Solver handle (handle.rs) -- SolverHandle context management
- [x] PTX helpers (ptx_helpers.rs) -- Shared PTX code generation utilities
- [x] LU factorization (dense/lu.rs) -- P*A = L*U with partial pivoting
- [x] QR factorization (dense/qr.rs) -- Blocked Householder reflections (A = Q*R)
- [x] Cholesky decomposition (dense/cholesky.rs) -- SPD factorization (A = L*L^T) with solve
- [x] SVD (dense/svd.rs) -- Singular Value Decomposition (A = U*S*V^T)
- [x] Eigendecomposition (dense/eig.rs) -- Symmetric eigenvalue decomposition (A = Q*L*Q^T)
- [x] Matrix inverse (dense/inverse.rs) -- Inverse via LU factorization
- [x] Determinant (dense/det.rs) -- Determinant and log-determinant via LU
- [x] Least squares (dense/lstsq.rs) -- Least squares solver via QR (min ||A*x - b||)
- [x] Pivot helpers (helpers/pivot.rs) -- Pivot selection and row swapping utilities
- [x] Condition number (helpers/condition.rs) -- Matrix condition number estimation
- [x] CG solver (sparse/cg.rs) -- Conjugate Gradient for SPD systems with CgConfig
- [x] BiCGSTAB solver (sparse/bicgstab.rs) -- Biconjugate Gradient Stabilized with BiCgStabConfig
- [x] GMRES solver (sparse/gmres.rs) -- GMRES(m) with restart and GmresConfig
- [x] Direct sparse solver (sparse/direct.rs) -- Direct solver via dense LU for small systems

### Future Enhancements

- [x] Batched LU/QR/Cholesky (batched.rs) -- BatchedSolver for many small matrices (4x4 to 64x64), batched_solve (P0)
- [x] Randomized SVD (randomized_svd.rs) -- Halko-Martinsson-Tropp 2011, configurable rank/oversampling/power iterations (P0)
- [x] Divide-and-conquer SVD (dense/dc_svd.rs) -- recursive bidiagonal splitting for faster convergence (P1)
- [x] Symmetric indefinite factorization (dense/ldlt.rs) -- Bunch-Kaufman LDL^T for symmetric indefinite systems (P1)
- [x] Preconditioned CG/GMRES (preconditioned.rs) -- Preconditioner trait, Jacobi, PCG, PGMRES (P1)
- [x] Flexible GMRES (sparse/fgmres.rs) -- FGMRES allowing variable preconditioner per iteration (P1)
- [x] Band matrix solvers (dense/band.rs) -- banded LU, Cholesky, solve with O(n*b^2) complexity (P1)
- [x] Tridiagonal/pentadiagonal solvers (tridiagonal.rs) -- Thomas algorithm, cyclic reduction, batched (P1)
- [x] Non-symmetric eigenvalue -- QZ algorithm for generalized eigenvalue problem A*x = lambda*B*x (P2)
- [x] Matrix functions (dense/matrix_functions.rs) -- Matrix exponential (expm), logarithm (logm), square root (sqrtm) via Pade approximation (P2)
- [x] Sparse direct solvers -- Supernodal Cholesky and multifrontal LU (sparse/direct_factorization.rs) (P2)
- [x] Nested dissection ordering -- Graph-based fill-reducing ordering for sparse direct solvers (P2)
- [x] Tensor decomposition (tensor_decomp.rs) -- CP, Tucker, and Tensor-Train decompositions for multi-dimensional arrays on GPU (P1)
- [x] ODE/PDE solver (ode_pde.rs) -- Runge-Kutta (RK4/RK45), implicit Euler, and method-of-lines PDE solver with GPU-accelerated right-hand-side evaluation (P1)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API (runtime loading) | Yes |
| oxicuda-memory | GPU memory management | Yes |
| oxicuda-launch | Kernel launch infrastructure | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| oxicuda-blas | Dense BLAS for decomposition kernels | Yes |
| oxicuda-sparse (optional) | Sparse matrix types for sparse solvers | Yes |
| thiserror | Error derive macros | Yes |
| half (optional) | FP16 support | Yes |

## Quality Status

- Tests: 373 passing
- All production code uses Result/Option (no unwrap)
- clippy::all and missing_docs warnings enabled
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- Matrix-free iterative solvers accept closure-based SpMV
- Feature-gated sparse solver integration (`sparse-solvers` feature)

## Estimation vs Actual

| Metric | Estimated (Vol.5 solver) | Actual |
|--------|-------------------------|--------|
| SLoC | 76K-122K | 13,981 |
| Files | ~15-20 | 40 |
| Coverage | Full cuSOLVER parity | Core decompositions + iterative |
| Ratio | -- | ~3.4% of estimate |

The estimation included exhaustive batched variants, band/tridiagonal specializations, and sparse direct solvers. The current implementation covers the essential decomposition and iterative solver foundation.

---

## Blueprint Quality Gates (Vol.5 Sec 9)

### Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| S9 | LU factorization + linear solve — correctness | P0 | [x] |
| S10 | QR factorization — correctness (Householder) | P0 | [x] |
| S11 | Cholesky decomposition — correctness for SPD matrices | P0 | [x] |
| S12 | SVD — correctness (U, Σ, Vᵀ reconstruction) | P0 | [x] |
| S13 | Eigenvalue decomposition — correctness for symmetric matrices | P0 | [x] |
| S14 | Batched SVD — correctness for batch ≥ 32 matrices | P1 | [x] |

### Performance Requirements

| # | Requirement | Target | Status |
|---|-------------|--------|--------|
| P4 | LU factorization, 4096×4096 F64 | ≥ 90% cuSOLVER throughput | [ ] |
| P5 | SVD, 1024×1024 F64 | ≥ 85% cuSOLVER throughput | [ ] |
| P6 | Cholesky, 4096×4096 F64 | ≥ 90% cuSOLVER throughput | [ ] |

---

## Numerical Accuracy Requirements (Vol.5 Sec 9)

### Backward Error Bounds

| Decomposition | Backward Error Criterion | Reference |
|--------------|------------------------|-----------|
| LU | ‖PA - LU‖_F / (‖A‖_F × n × ε) < threshold | LAPACK Working Note 13 |
| QR | ‖A - QR‖_F / (‖A‖_F × n × ε) < threshold | LAPACK Working Note 13 |
| Cholesky | ‖A - LLᵀ‖_F / (‖A‖_F × n × ε) < threshold | LAPACK Working Note 13 |
| SVD | ‖A - UΣVᵀ‖_F / ‖A‖_F < n × ε | Golub & Van Loan §8 |
| Eigenvalue | ‖AX - XΛ‖ / (‖A‖ × ε) < threshold | LAPACK Working Note 13 |

### Absolute Tolerances

| Decomposition | FP32 | FP64 |
|--------------|------|------|
| LU solve residual ‖Ax - b‖/‖b‖ | < 1e-5 | < 1e-12 |
| QR residual | < 1e-5 | < 1e-12 |
| Cholesky solve residual | < 1e-5 | < 1e-12 |
| SVD reconstruction | < 1e-5 | < 1e-12 |

---

## Architecture-Specific Deepening Opportunities

### GEMM/TRSM Integration
- [x] LU factorization uses Vol.3 TRSM and GEMM internally — integration verified
- [x] Cholesky uses Vol.3 SYRK internally — integration verified
- [x] Panel factorization block size tuned: LU block_size=64, Cholesky block_size=64, QR block_size=32

### Ampere (sm_80) / Hopper (sm_90)
- [ ] Randomized SVD uses `cuBLAS` equivalent GEMM for sketch — achieves ≥ 85% throughput for large matrices
- [x] Divide & Conquer SVD: bidiagonalization on GPU for matrices N ≥ 1024

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These represent the gap between current implementation depth and blueprint-grade production requirements.

- [ ] All S9–S14 functional requirements verified on GPU hardware
- [x] Backward error bounds verified for LU, QR, Cholesky, SVD per LAPACK standards
- [x] Absolute tolerance test suite: all decompositions × FP32/FP64 × random/ill-conditioned matrices
- [ ] Performance benchmarks P4–P6 measured and documented (vs cuSOLVER reference)
- [ ] Batched LU: 1000 × (64×64) matrices, throughput vs cuSOLVER batched API
- [x] Iterative refinement: CG/BiCGSTAB/GMRES convergence verified on standard test problems
- [x] Sparse direct solver: supernodal Cholesky vs iterative for well-conditioned SPD systems
