//! # OxiCUDA Launch
//!
//! **Type-safe GPU kernel launch infrastructure for the OxiCUDA ecosystem.**
//!
//! This crate provides ergonomic, type-safe abstractions for launching CUDA
//! GPU kernels. It builds on top of [`oxicuda_driver`] to offer:
//!
//! - **[`Dim3`]** — 3-dimensional grid and block size specification with
//!   convenient conversions from `u32`, `(u32, u32)`, and `(u32, u32, u32)`.
//!
//! - **[`LaunchParams`]** — kernel launch configuration (grid, block, shared
//!   memory) with a builder pattern via [`LaunchParamsBuilder`].
//!
//! - **[`Kernel`]** — a launchable kernel wrapper that manages module lifetime
//!   via `Arc<Module>` and provides occupancy query delegation.
//!
//! - **[`KernelArgs`]** — a trait for type-safe kernel argument passing,
//!   implemented for tuples of `Copy` types up to 24 elements.
//!
//! - **[`launch!`]** — a convenience macro for concise kernel launches.
//!
//! - **[`grid_size_for`]** — a helper to compute the minimum grid size
//!   needed to cover a given number of work items.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use oxicuda_driver::{init, Device, Context, Module, Stream};
//! use oxicuda_launch::{Kernel, LaunchParams, Dim3, grid_size_for, launch};
//!
//! # fn main() -> oxicuda_driver::CudaResult<()> {
//! init()?;
//! let dev = Device::get(0)?;
//! let ctx = Arc::new(Context::new(&dev)?);
//!
//! // Load PTX and create a kernel.
//! let ptx = ""; // In practice, use include_str! or load from file.
//! let module = Arc::new(Module::from_ptx(ptx)?);
//! let kernel = Kernel::from_module(module, "vector_add")?;
//!
//! // Configure launch dimensions.
//! let n: u32 = 1024;
//! let block_size = 256u32;
//! let grid = grid_size_for(n, block_size);
//!
//! let stream = Stream::new(&ctx)?;
//!
//! // Launch with the macro.
//! let (a_ptr, b_ptr, c_ptr) = (0u64, 0u64, 0u64);
//! launch!(kernel, grid(grid), block(block_size), &stream, &(a_ptr, b_ptr, c_ptr, n))?;
//!
//! stream.synchronize()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Crate features
//!
//! | Feature     | Description                              |
//! |-------------|------------------------------------------|
//! | `gpu-tests` | Enable tests that require a physical GPU |

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

// ---------------------------------------------------------------------------
// Module declarations
// ---------------------------------------------------------------------------

pub mod arg_serialize;
pub mod async_launch;
pub mod cluster;
pub mod cooperative;
pub mod dynamic_parallelism;
pub mod error;
pub mod graph_launch;
pub mod grid;
pub mod kernel;
pub mod macros;
pub mod multi_stream;
pub mod named_args;
pub mod params;
pub mod telemetry;
pub mod trace;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use arg_serialize::{
    ArgType, LaunchLog, LaunchLogger, LaunchSummary, SerializableKernelArgs, SerializedArg,
};
pub use async_launch::{
    AsyncKernel, AsyncLaunchConfig, CompletionStatus, ErasedKernelArgs, LaunchCompletion,
    LaunchTiming, PollStrategy, TimedLaunchCompletion, multi_launch_async,
};
pub use cluster::{ClusterDim, ClusterLaunchParams, cluster_launch};
pub use cooperative::CooperativeLaunch;
pub use dynamic_parallelism::{
    ChildKernelSpec, DynamicLaunchPlan, DynamicParallelismConfig, GridSpec,
};
pub use error::LaunchError;
pub use graph_launch::{GraphLaunchCapture, LaunchRecord};
pub use grid::{Dim3, auto_grid_2d, auto_grid_for, grid_size_for};
pub use kernel::{Kernel, KernelArgs};
pub use multi_stream::{multi_stream_launch, multi_stream_launch_uniform};
pub use named_args::{ArgBuilder, NamedKernelArgs};
pub use params::{LaunchParams, LaunchParamsBuilder};
pub use telemetry::{
    KernelStats, LaunchTelemetry, TelemetryCollector, TelemetryExporter, TelemetrySummary,
    estimate_occupancy,
};
pub use trace::KernelSpanGuard;

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

/// Convenient glob import for common OxiCUDA Launch types.
///
/// ```rust
/// use oxicuda_launch::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        ArgBuilder, ArgType, AsyncKernel, AsyncLaunchConfig, ChildKernelSpec, ClusterDim,
        ClusterLaunchParams, CompletionStatus, CooperativeLaunch, Dim3, DynamicLaunchPlan,
        DynamicParallelismConfig, GraphLaunchCapture, GridSpec, Kernel, KernelArgs,
        LaunchCompletion, LaunchError, LaunchLog, LaunchLogger, LaunchParams, LaunchParamsBuilder,
        LaunchRecord, LaunchSummary, LaunchTiming, NamedKernelArgs, PollStrategy,
        SerializableKernelArgs, SerializedArg, TimedLaunchCompletion, auto_grid_2d, auto_grid_for,
        cluster_launch, grid_size_for, multi_launch_async, multi_stream_launch,
        multi_stream_launch_uniform,
    };
}
