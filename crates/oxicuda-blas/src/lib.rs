//! # OxiCUDA BLAS — GPU-Accelerated BLAS Operations
//!
//! This crate provides GPU-accelerated Basic Linear Algebra Subprograms (BLAS),
//! serving as a pure Rust equivalent to cuBLAS.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use oxicuda_driver::Context;
//! use oxicuda_blas::handle::BlasHandle;
//!
//! # fn main() -> Result<(), oxicuda_blas::error::BlasError> {
//! # let ctx: Arc<Context> = unimplemented!();
//! let handle = BlasHandle::new(&ctx)?;
//! // ... call BLAS routines via the handle ...
//! # Ok(())
//! # }
//! ```

#![warn(clippy::all)]
#![warn(missing_docs)]

pub mod algorithm_selection;
pub mod batched;
pub mod complex_gemm;
pub mod elementwise;
pub mod error;
pub mod handle;
pub mod level1;
pub mod level2;
pub mod level3;
pub mod precision;
pub mod reduction;
pub mod types;

#[cfg(test)]
mod test_matrices;

pub use algorithm_selection::{
    AlgorithmConfig, AlgorithmHeuristic, AlgorithmId, AlgorithmSelector, EpiloguePreference,
    SwizzleMode,
};
pub use error::{BlasError, BlasResult};
pub use handle::BlasHandle;
pub use types::{
    DiagType, E4M3, E5M2, FillMode, GpuFloat, Layout, MathMode, MatrixDesc, MatrixDescMut,
    PointerMode, Side, Transpose, VectorDesc,
};

/// Convenience re-exports for common BLAS usage.
///
/// ```rust,no_run
/// use oxicuda_blas::prelude::*;
/// ```
pub mod prelude {
    // Algorithm selection (cuBLASLt-style)
    pub use crate::algorithm_selection::{
        AlgorithmConfig, AlgorithmHeuristic, AlgorithmId, AlgorithmSelector, EpiloguePreference,
        SwizzleMode,
    };

    // Core types
    pub use crate::error::{BlasError, BlasResult};
    pub use crate::handle::BlasHandle;
    pub use crate::types::{
        DiagType, E4M3, E5M2, FillMode, GpuFloat, Layout, MathMode, MatrixDesc, MatrixDescMut,
        PointerMode, Side, Transpose, VectorDesc,
    };

    // BLAS Level 1
    pub use crate::level1::{asum, axpy, copy_vec, dot, iamax, nrm2, scal, swap};

    // BLAS Level 2
    pub use crate::level2::{gemv, ger, symv, syr, trmv, trsv};

    // BLAS Level 3
    pub use crate::level3::persistent_gemm::PersistentGemmConfig;
    pub use crate::level3::stream_k::StreamKConfig;
    pub use crate::level3::{
        batched_trsm, gemm_api, persistent_gemm, stream_k, symm, syr2k, syrk, trmm, trsm,
    };

    // Complex GEMM/GEMV
    pub use crate::complex_gemm::{complex_gemm, complex_gemv};

    // Batched operations
    pub use crate::batched::{batched_gemm, grouped_gemm, strided_gemm};

    // Elementwise operations
    pub use crate::elementwise;

    // Reduction operations
    pub use crate::reduction;
}
