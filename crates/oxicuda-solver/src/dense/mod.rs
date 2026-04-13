//! Dense matrix decompositions and solvers.
//!
//! This module provides GPU-accelerated implementations of the core dense
//! linear algebra algorithms: LU, QR, Cholesky, SVD, eigendecomposition,
//! matrix inverse, determinant, and least squares.

pub mod band;
pub mod batched;
pub mod cholesky;
pub mod dc_svd;
pub mod det;
pub mod eig;
pub mod inverse;
pub mod ldlt;
pub mod lstsq;
pub mod lu;
pub mod matrix_functions;
pub mod ode_pde;
pub mod qr;
pub mod qz;
pub mod randomized_svd;
pub mod svd;
pub mod tensor_decomp;
pub mod tridiagonal;

// Re-exports of primary types and functions.
pub use band::{BandMatrix, band_cholesky, band_lu, band_solve};
pub use batched::{BatchAlgorithm, BatchConfig, BatchedResult, BatchedSolver};
pub use cholesky::{cholesky, cholesky_solve};
pub use dc_svd::{DcSvdConfig, dc_svd};
pub use det::{determinant, log_determinant};
pub use eig::{EigJob, syevd};
pub use inverse::inverse;
pub use ldlt::{LdltResult, ldlt, ldlt_solve};
pub use lstsq::lstsq;
pub use lu::{LuResult, lu_factorize, lu_solve};
pub use matrix_functions::{
    MatrixExpConfig, MatrixExpPlan, MatrixLogConfig, MatrixLogPlan, MatrixSqrtConfig,
    MatrixSqrtPlan,
};
pub use ode_pde::{
    AdvectionEquation1D, Bdf2Solver, BoundaryCondition, EulerSolver, Grid1D, Grid2D,
    HeatEquation1D, ImplicitEulerSolver, OdeConfig, OdeMethod, OdeSolution, OdeSystem, PdeConfig,
    Poisson1D, Rk4Solver, Rk45Solver, StepResult, WaveEquation1D, numerical_jacobian,
    solve_tridiagonal as ode_solve_tridiagonal,
};
pub use qr::{qr_factorize, qr_generate_q, qr_solve};
pub use qz::{
    BalanceStrategy, EigenvalueType, QzConfig, QzPlan, QzResult, QzStep, ShiftStrategy,
    classify_eigenvalue, estimate_qz_flops, plan_qz, qz_host, validate_qz_config,
};
pub use randomized_svd::{RandomizedSvdConfig, RandomizedSvdResult, randomized_svd};
pub use svd::{SvdJob, SvdResult, svd};
pub use tensor_decomp::{
    CpAlsConfig, CpDecomposition, Matrix, Tensor, TtConfig, TtDecomposition, TuckerConfig,
    TuckerDecomposition, cp_als, hadamard_product, khatri_rao_product, mode_n_product, tt_svd,
    tucker_hooi, tucker_hosvd,
};
pub use tridiagonal::{batched_tridiagonal_solve, tridiagonal_solve};
