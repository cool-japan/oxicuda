//! # OxiCUDA Solver — GPU-Accelerated Matrix Decompositions
//!
//! This crate provides GPU-accelerated matrix decompositions and linear solvers,
//! serving as a pure Rust equivalent to NVIDIA's cuSOLVER library.
//!
//! ## Dense decompositions
//!
//! - **LU** — LU factorization with partial pivoting (`P * A = L * U`)
//! - **QR** — QR factorization via blocked Householder reflections (`A = Q * R`)
//! - **Cholesky** — Cholesky decomposition for SPD matrices (`A = L * L^T`)
//! - **SVD** — Singular Value Decomposition (`A = U * Σ * V^T`)
//! - **DC-SVD** — Divide-and-Conquer SVD for medium-to-large matrices
//! - **LDL^T** — Bunch-Kaufman factorization for symmetric indefinite matrices
//! - **Eigendecomposition** — Symmetric eigenvalue decomposition (`A = Q * Λ * Q^T`)
//! - **Inverse** — Matrix inverse via LU (`A^{-1}`)
//! - **Determinant** — Determinant and log-determinant via LU
//! - **Least squares** — Least squares solver via QR (`min ||A*x - b||`)
//!
//! ## Dense solvers
//!
//! - **Tridiagonal** — Thomas algorithm and cyclic reduction for tridiagonal systems
//! - **Band** — LU and Cholesky for banded matrices (O(n*b^2) complexity)
//!
//! ## Iterative sparse solvers
//!
//! - **CG** — Conjugate Gradient for SPD systems
//! - **BiCGSTAB** — Biconjugate Gradient Stabilized for non-symmetric systems
//! - **GMRES(m)** — Generalized Minimal Residual with restart
//! - **FGMRES(m)** — Flexible GMRES with variable preconditioner
//! - **Direct** — Direct sparse solver via dense LU (small systems)
//! - **Preconditioned CG** — PCG with pluggable preconditioners (Identity, Jacobi, ILU, IC)
//! - **Preconditioned GMRES** — Left-preconditioned GMRES(m) with restart
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxicuda_solver::prelude::*;
//! ```
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

#![warn(clippy::all)]
#![warn(missing_docs)]

pub mod dense;
pub mod error;
pub mod handle;
pub mod helpers;
pub mod sparse;

#[allow(dead_code)]
pub(crate) mod ptx_helpers;

pub use dense::{
    AdvectionEquation1D, BandMatrix, BatchAlgorithm, BatchConfig, BatchedResult, BatchedSolver,
    Bdf2Solver, BoundaryCondition, CpAlsConfig, CpDecomposition, DcSvdConfig, EigJob, EulerSolver,
    Grid1D, Grid2D, HeatEquation1D, ImplicitEulerSolver, LdltResult, LuResult, Matrix, OdeConfig,
    OdeMethod, OdeSolution, OdeSystem, PdeConfig, Poisson1D, RandomizedSvdConfig,
    RandomizedSvdResult, Rk4Solver, Rk45Solver, StepResult, SvdJob, SvdResult, Tensor, TtConfig,
    TtDecomposition, TuckerConfig, TuckerDecomposition, WaveEquation1D,
};
pub use error::{SolverError, SolverResult};
pub use handle::SolverHandle;
pub use sparse::{
    AdjacencyGraph, EliminationTree, FgmresConfig, MultifrontalLUSolver, NestedDissectionOrdering,
    OrderingQuality, Permutation, SupernodalCholeskySolver, SupernodalStructure,
    SymbolicFactorization, fgmres, sparse_cholesky_solve, sparse_lu_solve,
};

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::dense::*;
    pub use crate::error::{SolverError, SolverResult};
    pub use crate::handle::SolverHandle;
    pub use crate::sparse::direct_factorization::{
        EliminationTree, MultifrontalLUSolver, SupernodalCholeskySolver, SupernodalStructure,
        SymbolicFactorization, sparse_cholesky_solve, sparse_lu_solve,
    };
    pub use crate::sparse::nested_dissection::{
        AdjacencyGraph, NestedDissectionOrdering, OrderingQuality, Permutation,
    };
}
