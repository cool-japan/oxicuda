//! Tensor Core instruction generation helpers.
//!
//! This module provides configuration types and helper functions for NVIDIA
//! Tensor Core instructions across multiple architecture generations:
//!
//! - [`crate::tensor_core::wmma`]: WMMA (Warp Matrix Multiply-Accumulate) for Volta/Turing+
//! - [`crate::tensor_core::mma`]: MMA (`mma.sync.aligned`) for Ampere+
//! - [`crate::tensor_core::wgmma`]: WGMMA (Warp Group MMA) for Hopper+
//!
//! Each sub-module defines a configuration struct that computes fragment sizes,
//! register counts, and validates type combinations for the corresponding
//! instruction family.

pub mod mma;
pub mod wgmma;
pub mod wmma;
