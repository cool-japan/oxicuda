//! Sparse matrix storage formats.
//!
//! This module provides GPU-backed sparse matrix types in multiple standard
//! formats:
//!
//! - [`CsrMatrix`] -- Compressed Sparse Row
//! - [`CscMatrix`] -- Compressed Sparse Column
//! - [`CooMatrix`] -- Coordinate (triplet) format
//! - [`BsrMatrix`] -- Block Sparse Row
//! - [`EllMatrix`] -- ELLPACK format
//! - [`HybMatrix`] -- HYB (Hybrid ELL+COO) format
//!
//! Format conversion routines are in the [`convert`] sub-module.

pub mod bsr;
pub mod convert;
pub mod coo;
pub mod csc;
pub mod csr;
pub mod csr5;
pub mod ell;
pub mod hyb;
pub mod reorder;

pub use bsr::BsrMatrix;
pub use coo::CooMatrix;
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
pub use csr5::Csr5Matrix;
pub use ell::EllMatrix;
pub use hyb::{HybMatrix, HybPartition, HybStatistics};
pub use reorder::{amd_ordering, inverse_permutation, permute_csr, rcm_ordering};
