//! Iterative sparse linear solvers.
//!
//! Provides matrix-free iterative methods for solving large sparse linear systems
//! `A * x = b`. The solvers accept a closure `spmv: F` that computes the
//! sparse matrix-vector product `y = A * x`, enabling use with any sparse
//! matrix format or even matrix-free operators.
//!
//! # Solvers
//!
//! - **CG** ([`cg`]): Conjugate Gradient for symmetric positive definite systems.
//! - **BiCGSTAB** ([`bicgstab`]): Biconjugate Gradient Stabilized for non-symmetric systems.
//! - **GMRES(m)** ([`gmres`]): Generalized Minimal Residual with restart for general systems.
//! - **Direct** ([`direct`]): Direct sparse solver via dense LU (for small-to-medium systems).

pub mod bicgstab;
pub mod cg;
pub mod direct;
pub mod direct_factorization;
pub mod fgmres;
pub mod gmres;
pub mod nested_dissection;
pub mod preconditioned;

pub use bicgstab::{BiCgStabConfig, bicgstab_solve};
pub use cg::{CgConfig, cg_solve};
pub use direct::prefer_direct_solver;
pub use direct_factorization::{
    EliminationTree, MultifrontalLUSolver, SupernodalCholeskySolver, SupernodalStructure,
    SymbolicFactorization, sparse_cholesky_solve, sparse_lu_solve,
};
pub use fgmres::{FgmresConfig, fgmres};
pub use gmres::{GmresConfig, gmres_solve};
pub use nested_dissection::{
    AdjacencyGraph, NestedDissectionOrdering, OrderingQuality, Permutation,
};
pub use preconditioned::{
    IdentityPreconditioner, IterativeSolverResult, JacobiPreconditioner, PcgConfig, PgmresConfig,
    Preconditioner, preconditioned_cg, preconditioned_gmres,
};
