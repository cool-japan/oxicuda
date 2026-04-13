//! BLAS Level 2 -- matrix-vector operations.
//!
//! This module provides GPU-accelerated implementations of the classic BLAS
//! Level 2 routines. Each operation launches one or more PTX kernels via the
//! [`BlasHandle`](crate::handle::BlasHandle).
//!
//! | Function  | Operation                                          |
//! |-----------|----------------------------------------------------|
//! | [`fn@gemv`]  | y = alpha * op(A) * x + beta * y                   |
//! | [`fn@symv`]  | y = alpha * A * x + beta * y  (A symmetric)        |
//! | [`fn@trmv`]  | x = op(A) * x               (A triangular)         |
//! | [`fn@trsv`]  | solve op(A) * x = b         (A triangular)         |
//! | [`fn@ger`]   | A = alpha * x * y^T + A     (rank-1 update)        |
//! | [`fn@syr`]   | A = alpha * x * x^T + A     (symmetric rank-1)     |

pub mod gemv;
pub mod ger;
pub mod symv;
pub mod syr;
pub mod trmv;
pub mod trsv;

pub use gemv::gemv;
pub use ger::ger;
pub use symv::symv;
pub use syr::syr;
pub use trmv::trmv;
pub use trsv::trsv;
