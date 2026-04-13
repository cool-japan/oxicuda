//! Sparse matrix operations.
//!
//! This module provides GPU-accelerated sparse matrix operations:
//!
//! - [`fn@spmv`] -- Sparse matrix-vector multiplication (y = alpha*A*x + beta*y)
//! - [`fn@spmm`] -- Sparse matrix-dense matrix multiplication (C = alpha*A*B + beta*C)
//! - [`spgemm`] -- Sparse matrix-sparse matrix multiplication (C = A*B)
//! - [`fn@sptrsv`] -- Sparse triangular solve (L*x = b or U*x = b)
//! - [`fn@sddmm`] -- Sampled Dense-Dense Matrix Multiply
//! - [`krylov`] -- Krylov subspace methods (Lanczos & Arnoldi iteration)
//! - [`matrix_powers`] -- Sparse matrix powers (A^k) and polynomial evaluation

pub mod auto_spmv;
pub mod batched;
pub mod krylov;
pub mod matrix_powers;
pub mod mixed_precision_spmv;
pub mod sddmm;
pub mod spgemm;
pub mod spgemm_estimate;
pub mod spgemm_merge;
pub mod spmm;
pub mod spmv;
pub mod spmv_bsr;
pub mod spmv_csr5;
pub mod spmv_ell;
pub mod sptrsv;
pub mod tensor;

pub use auto_spmv::{
    RecommendedFormat, SpMatFormat, analyze_sparsity, auto_spmv, recommend_format, select_format,
};
pub use batched::{
    BatchScheduler, BatchedSpGEMM, BatchedSpMV, BatchedSpMVPlan, BatchedTriSolve, Strategy,
    UniformBatchedSpMV, batched_spmv_cpu, generate_batched_spmv_ptx, mixed_precision_spmv_cpu,
};
pub use krylov::{
    ArnoldiConfig, ArnoldiPlan, ArnoldiResult, EigenTarget, LanczosConfig, LanczosPlan,
    LanczosResult,
};
pub use matrix_powers::{
    MatrixPowerConfig, MatrixPowerResult, estimate_power_nnz, sparse_identity,
    sparse_matrix_polynomial, sparse_matrix_power,
};
pub use mixed_precision_spmv::{
    ComputePrecision, MixedPrecisionConfig, MixedPrecisionPlan, MixedPrecisionStats, MixedSpMVAlgo,
    StoragePrecision, estimate_precision_loss, generate_mixed_scalar_spmv_ptx,
    generate_mixed_vector_spmv_ptx, generate_packed_vector_spmv_ptx, plan_mixed_precision_spmv,
    validate_mixed_precision_config,
};
pub use sddmm::sddmm;
pub use spgemm::{spgemm_numeric, spgemm_symbolic};
pub use spgemm_estimate::{
    EstimationMethod, SpGEMMEstimate, auto_estimate_spgemm, count_nnz_exact, estimate_nnz_sampling,
    estimate_nnz_upper_bound, estimate_spgemm_memory,
};
pub use spgemm_merge::spgemm_merge;
pub use spmm::spmm;
pub use spmv::{SpMVAlgo, spmv};
pub use spmv_bsr::spmv_bsr;
pub use spmv_csr5::csr5_spmv;
pub use spmv_ell::spmv_ell;
pub use sptrsv::sptrsv;
pub use tensor::{
    EdgeFeatures, GnnSparseConfig, MessagePassingOp, add_self_loops, compute_degree_matrix, gather,
    scatter_reduce, sparse_attention_message, sparse_message_passing, sparse_row_softmax,
    symmetric_normalize,
};
