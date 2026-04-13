//! # OxiCUDA Sparse -- GPU-Accelerated Sparse Matrix Operations
//!
//! This crate provides GPU-accelerated sparse matrix operations,
//! serving as a pure Rust equivalent to NVIDIA's cuSPARSE library.
//!
//! ## Sparse formats
//!
//! Multiple storage formats are supported via the [`mod@format`] module:
//! - [`CsrMatrix`] -- Compressed Sparse Row (primary format)
//! - [`CscMatrix`] -- Compressed Sparse Column
//! - [`CooMatrix`] -- Coordinate (triplet)
//! - [`BsrMatrix`] -- Block Sparse Row
//! - [`EllMatrix`] -- ELLPACK
//! - [`HybMatrix`] -- HYB (Hybrid ELL+COO)
//!
//! ## Operations
//!
//! - [`fn@ops::spmv`] -- Sparse matrix-vector multiply (`y = alpha*A*x + beta*y`)
//! - [`fn@ops::spmm`] -- Sparse-dense matrix multiply (`C = alpha*A*B + beta*C`)
//! - [`ops::spgemm`] -- Sparse-sparse matrix multiply (`C = A*B`)
//! - [`fn@ops::sptrsv`] -- Sparse triangular solve (`L*x = b` or `U*x = b`)
//! - [`fn@ops::sddmm`] -- Sampled Dense-Dense Matrix Multiply
//!
//! ## Preconditioners
//!
//! - [`fn@preconditioner::ilu0`] -- Incomplete LU(0) for general systems
//! - [`fn@preconditioner::ic0`] -- Incomplete Cholesky(0) for SPD systems
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxicuda_sparse::prelude::*;
//! ```

#![warn(clippy::all)]
#![warn(missing_docs)]

pub mod error;
pub mod format;
pub mod handle;
pub mod ops;
pub mod preconditioner;
pub(crate) mod ptx_helpers;

pub use error::{SparseError, SparseResult};
pub use format::{
    BsrMatrix, CooMatrix, CscMatrix, Csr5Matrix, CsrMatrix, EllMatrix, HybMatrix, HybPartition,
    HybStatistics,
};
pub use handle::SparseHandle;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::error::{SparseError, SparseResult};
    pub use crate::format::{
        BsrMatrix, CooMatrix, CscMatrix, Csr5Matrix, CsrMatrix, EllMatrix, HybMatrix, HybPartition,
        HybStatistics, amd_ordering, inverse_permutation, permute_csr, rcm_ordering,
    };
    pub use crate::handle::SparseHandle;
    pub use crate::ops::{
        BatchScheduler, BatchedSpGEMM, BatchedSpMV, BatchedSpMVPlan, BatchedTriSolve,
        RecommendedFormat, SpMVAlgo, SpMatFormat, Strategy, UniformBatchedSpMV, analyze_sparsity,
        auto_spmv, batched_spmv_cpu, csr5_spmv, generate_batched_spmv_ptx,
        mixed_precision_spmv_cpu, recommend_format, select_format, spgemm_merge, spmm, spmv,
        spmv_bsr, spmv_ell,
    };
    pub use crate::ops::{
        EdgeFeatures, GnnSparseConfig, MessagePassingOp, add_self_loops, compute_degree_matrix,
        gather, scatter_reduce, sparse_attention_message, sparse_message_passing,
        sparse_row_softmax, symmetric_normalize,
    };
    pub use crate::preconditioner::{IlukConfig, IlukFactorization};
}
