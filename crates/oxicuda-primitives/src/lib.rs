//! OxiCUDA Primitives — CUB-equivalent high-performance parallel GPU primitives.
//!
//! This crate provides PTX code generators for GPU parallel algorithms without
//! any dependency on the CUDA SDK.  All kernels are generated as PTX source
//! strings at runtime and JIT-compiled via `cuModuleLoadData`.
//!
//! # Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`warp`] | Warp-level reduce and scan using `shfl.sync.*` |
//! | [`block`] | Block-level reduce and scan via shared memory |
//! | [`device`] | Device-wide reduce, scan, select, and histogram |
//! | [`sort`] | Device-wide radix sort and merge sort |
//! | [`ptx_helpers`] | Shared PTX code-generation utilities |
//! | [`handle`] | Execution-context holder with SM version info |
//! | [`error`] | Error and result types |
//!
//! # Quick start
//!
//! ```
//! use oxicuda_primitives::device::reduce::{DeviceReduceConfig, DeviceReduceTemplate};
//! use oxicuda_primitives::ptx_helpers::ReduceOp;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = DeviceReduceConfig::new(ReduceOp::Sum, PtxType::F32);
//! let (pass1_ptx, pass2_ptx) = DeviceReduceTemplate::new(cfg)
//!     .generate(SmVersion::Sm80)
//!     .expect("PTX generation failed");
//!
//! // JIT-compile and launch pass1_ptx and pass2_ptx via the CUDA driver API.
//! assert!(pass1_ptx.contains("device_reduce_pass1_sum_f32"));
//! ```

pub mod block;
pub mod device;
pub mod error;
pub mod handle;
pub mod ptx_helpers;
pub mod sort;
pub mod warp;

pub use error::{PrimitivesError, PrimitivesResult};
pub use handle::PrimitivesHandle;
pub use ptx_helpers::{PrimitiveType, ReduceOp};
