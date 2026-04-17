//! OxiCUDA Metal backend — GPU compute via Apple Metal API (macOS only).
//!
//! Provides a [`MetalBackend`] that implements the [`oxicuda_backend::ComputeBackend`] trait
//! from `oxicuda-backend`, using Metal to target Apple Silicon and Intel Mac
//! GPUs through Apple's first-party compute API.
//!
//! On non-macOS platforms, all operations return
//! [`MetalError::UnsupportedPlatform`] — the crate compiles successfully
//! everywhere so that cross-platform workspaces (including Linux CI) are
//! unaffected.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use oxicuda_metal::MetalBackend;
//! use oxicuda_backend::ComputeBackend;
//!
//! let mut backend = MetalBackend::new();
//! backend.init().expect("Metal init failed");
//!
//! // Allocate 256 bytes on the GPU.
//! let ptr = backend.alloc(256).expect("alloc failed");
//! backend.free(ptr).expect("free failed");
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────┐
//! │          MetalBackend                    │
//! │  (implements ComputeBackend)             │
//! └──────────────┬───────────────────────────┘
//!                │
//!       ┌────────▼────────┐
//!       │  MetalDevice    │  ← metal::Device (macOS only)
//!       └────────┬────────┘
//!                │
//!    ┌───────────▼────────────┐
//!    │  MetalMemoryManager    │  ← shared-mode MTLBuffer pool
//!    └────────────────────────┘
//! ```
//!
//! # MSL Kernel Generation
//!
//! The [`msl`] module provides helpers that produce MSL source strings for
//! common compute kernels (GEMM, element-wise ops, reductions).  These are
//! used by the pipeline layer and can also be called directly for custom
//! kernel compilation.

pub mod ane;
pub mod backend;
pub mod device;
pub mod error;
pub mod fft;
pub mod memory;
pub mod mps;
pub mod msl;
pub mod pipeline;

pub use backend::MetalBackend;
pub use error::{MetalError, MetalResult};
pub use fft::{MetalFftBuffer, MetalFftDirection, MetalFftPlan};
