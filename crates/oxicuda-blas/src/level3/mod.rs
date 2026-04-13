//! BLAS Level 3 — matrix-matrix operations.
//!
//! This module provides GPU-accelerated Level 3 BLAS routines:
//!
//! | Routine | Operation |
//! |---------|-----------|
//! | [`fn@gemm`] | General matrix multiply: `C = alpha * op(A) * op(B) + beta * C` |
//! | [`fn@symm`] | Symmetric matrix multiply: `C = alpha * A * B + beta * C` |
//! | [`fn@trsm`] | Triangular solve: `op(A) * X = alpha * B` |
//! | [`fn@syrk`] | Symmetric rank-k update: `C = alpha * A * A^T + beta * C` |
//! | [`fn@syr2k`] | Symmetric rank-2k update |
//! | [`fn@trmm`] | Triangular matrix multiply: `B = alpha * op(A) * B` |
//! | [`fn@batched_trsm`] | Batched triangular solve (many small systems) |
//! | [`fn@stream_k_gemm`] | Stream-K GEMM with dynamic load balancing |
//!
//! The GEMM dispatcher is the core engine, selecting optimal tile
//! configurations, generating PTX via [`oxicuda_ptx::templates::gemm::GemmTemplate`], and caching
//! compiled kernels.

pub mod batched_trsm;
pub mod gemm;
pub mod gemm_api;
pub mod persistent_gemm;
pub mod stream_k;
pub mod symm;
pub mod syr2k;
pub mod syrk;
pub mod syrk_tc;
pub mod trmm;
pub mod trsm;

pub use batched_trsm::batched_trsm;
pub use gemm::dispatch::{GemmCategory, GemmDispatcher, GemmProblem, TileConfig};
pub use gemm::epilogue::EpilogueOp;
pub use gemm_api::gemm;
pub use persistent_gemm::{PersistentGemmConfig, persistent_gemm};
pub use stream_k::{StreamKConfig, stream_k_gemm};
pub use symm::symm;
pub use syr2k::syr2k;
pub use syrk::syrk;
pub use trmm::trmm;
pub use trsm::trsm;
