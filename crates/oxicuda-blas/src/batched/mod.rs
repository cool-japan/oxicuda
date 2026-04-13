//! Batched GEMM operations.
//!
//! This module provides three flavours of batched matrix multiplication:
//!
//! - **Pointer-array batched** ([`batched_gemm`]) — each batch element has an
//!   independent device pointer for A, B, C, and D.
//! - **Strided batched** ([`strided_gemm`]) — all batch elements share the
//!   same base pointer with a fixed stride between consecutive matrices.
//! - **Grouped** ([`grouped_gemm`]) — each problem in the group may have
//!   entirely different dimensions and pointers.

pub mod batched_cholesky;
pub mod batched_gemm;
pub mod grouped_gemm;
pub mod multi_stream_batched;
pub mod strided_gemm;

pub use batched_cholesky::{
    BatchedCholeskyConfig, BatchedCholeskyPlan, BatchedCholeskyResult, CholeskyStep,
    estimate_cholesky_flops, generate_diagonal_cholesky_ptx, generate_panel_trsm_ptx,
    generate_schur_update_ptx, plan_batched_cholesky, validate_batched_cholesky,
};
pub use batched_gemm::gemm_batched;
pub use grouped_gemm::{GroupedGemmProblem, gemm_grouped};
pub use multi_stream_batched::{
    MultiStreamBatchedConfig, StreamDistribution, distribute_batches, gemm_batched_multi_stream,
};
pub use strided_gemm::gemm_strided_batched;
