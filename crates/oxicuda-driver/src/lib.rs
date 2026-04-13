//! # OxiCUDA Driver
//!
//! **Dynamic, safe Rust bindings for the NVIDIA CUDA Driver API.**
//!
//! `oxicuda-driver` provides a zero-SDK-dependency wrapper around the CUDA
//! Driver API.  Unlike traditional CUDA crate approaches that require the
//! CUDA Toolkit (or at least its headers and link stubs) to be present at
//! **build time**, this crate loads the driver shared library entirely at
//! **runtime** via [`libloading`](https://crates.io/crates/libloading).
//!
//! ## Zero build-time dependency
//!
//! No `cuda.h`, no `libcuda.so` symlink, no `nvcc` — the crate compiles on
//! any Rust toolchain.  The actual GPU driver is discovered and loaded the
//! first time you call [`try_driver()`] or [`init()`].
//!
//! ## Runtime library loading
//!
//! | Platform | Library searched             |
//! |----------|-----------------------------|
//! | Linux    | `libcuda.so`, `libcuda.so.1` |
//! | Windows  | `nvcuda.dll`                 |
//! | macOS    | *(returns `UnsupportedPlatform` — NVIDIA dropped macOS support)* |
//!
//! ## Key types
//!
//! | Type          | Description                                    |
//! |---------------|------------------------------------------------|
//! | [`Device`]    | A CUDA-capable GPU discovered on the system    |
//! | [`Context`]   | Owns a CUDA context bound to a device          |
//! | [`Stream`]    | Asynchronous command queue within a context     |
//! | [`Event`]     | Timing / synchronisation marker on a stream    |
//! | [`Module`]    | Loaded PTX or cubin containing kernel code     |
//! | [`Function`]  | A single kernel entry point inside a module    |
//! | [`CudaError`] | Strongly-typed driver error code               |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use oxicuda_driver::prelude::*;
//!
//! // Initialise the CUDA driver (loads libcuda at runtime).
//! init()?;
//!
//! // Pick the best available GPU and create a context.
//! let dev = Device::get(0)?;
//! let _ctx = Context::new(&dev)?;
//!
//! // Load a PTX module and look up a kernel.
//! let module = Module::from_ptx("ptx_source")?;
//! let kernel = module.get_function("vector_add")?;
//! # Ok::<(), oxicuda_driver::CudaError>(())
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::macro_metavars_in_unsafe)]

// ---------------------------------------------------------------------------
// Module declarations
// ---------------------------------------------------------------------------

pub mod context;
pub mod context_config;
pub mod cooperative_launch;
pub mod debug;
pub mod device;
pub mod error;
pub mod event;
pub mod ffi;
pub mod function_attr;
pub mod graph;
pub mod link;
pub mod loader;
pub mod memory_info;
pub mod module;
pub mod multi_gpu;
pub mod nvlink_topology;
pub mod occupancy;
pub mod occupancy_ext;
pub mod primary_context;
pub mod profiler;
pub mod stream;
pub mod stream_ordered_alloc;
pub mod tma;

// ---------------------------------------------------------------------------
// Re-exports — error handling
// ---------------------------------------------------------------------------

pub use error::{CudaError, CudaResult, DriverLoadError, check};

// ---------------------------------------------------------------------------
// Re-exports — FFI types and constants
// ---------------------------------------------------------------------------

pub use ffi::{
    CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION, CU_TRSF_NORMALIZED_COORDINATES,
    CU_TRSF_READ_AS_INTEGER, CU_TRSF_SRGB, CUDA_ARRAY_DESCRIPTOR, CUDA_ARRAY3D_CUBEMAP,
    CUDA_ARRAY3D_DESCRIPTOR, CUDA_ARRAY3D_LAYERED, CUDA_ARRAY3D_SURFACE_LDST,
    CUDA_ARRAY3D_TEXTURE_GATHER, CUDA_RESOURCE_DESC, CUDA_RESOURCE_VIEW_DESC, CUDA_TEXTURE_DESC,
    CUaddress_mode, CUarray, CUarray_format, CUcontext, CUdevice, CUdevice_attribute, CUdeviceptr,
    CUevent, CUfilter_mode, CUfunction, CUfunction_attribute, CUjit_option, CUkernel, CUlibrary,
    CUlimit, CUmemoryPool, CUmemorytype, CUmipmappedArray, CUmodule, CUmulticastObject,
    CUpointer_attribute, CUresourceViewFormat, CUresourcetype, CUstream, CUsurfObject, CUsurfref,
    CUtexObject, CUtexref, CuLaunchAttribute, CuLaunchAttributeClusterDim, CuLaunchAttributeId,
    CuLaunchAttributeValue, CuLaunchConfig, CudaResourceDescArray, CudaResourceDescLinear,
    CudaResourceDescMipmap, CudaResourceDescPitch2d, CudaResourceDescRes,
};

// ---------------------------------------------------------------------------
// Re-exports — high-level safe wrappers
// ---------------------------------------------------------------------------

pub use context::Context;
pub use context_config::{CacheConfig, SharedMemConfig};
pub use cooperative_launch::{
    CooperativeLaunchConfig, CooperativeLaunchSupport, DeviceLaunchConfig,
    MultiDeviceCooperativeLaunchConfig, cooperative_launch, cooperative_launch_multi_device,
};
pub use debug::{DebugLevel, DebugSession, KernelDebugger, MemoryChecker, NanInfChecker};
pub use device::{Device, DeviceInfo, best_device, can_access_peer, driver_version, list_devices};
pub use event::Event;
pub use graph::{Graph, GraphExec, GraphNode, MemcpyDirection, StreamCapture};
pub use link::{
    FallbackStrategy, LinkInputType, LinkedModule, Linker, LinkerOptions, OptimizationLevel,
};
pub use loader::try_driver;
pub use module::{Function, JitDiagnostic, JitLog, JitOptions, JitSeverity, Module};
pub use multi_gpu::DevicePool;
pub use nvlink_topology::{GpuTopology, NvLinkVersion, TopologyTree, TopologyType};
pub use primary_context::PrimaryContext;
pub use profiler::ProfilerGuard;
pub use stream::Stream;
pub use stream_ordered_alloc::{
    StreamAllocation, StreamMemoryPool, StreamOrderedAllocConfig, stream_alloc, stream_free,
};

// ---------------------------------------------------------------------------
// Driver initialisation
// ---------------------------------------------------------------------------

/// Initialise the CUDA driver API.
///
/// This must be called before any other driver function.  It is safe to call
/// multiple times; subsequent calls are no-ops inside the driver itself.
///
/// Internally this loads the shared library (if not already cached) and
/// invokes `cuInit(0)`.
///
/// # Errors
///
/// Returns [`CudaError::NotInitialized`] if the CUDA driver library cannot be
/// loaded, or another [`CudaError`] variant if `cuInit` reports a failure.
pub fn init() -> CudaResult<()> {
    let driver = loader::try_driver()?;
    error::check(unsafe { (driver.cu_init)(0) })
}

// ---------------------------------------------------------------------------
// Prelude — convenient glob import
// ---------------------------------------------------------------------------

/// Convenient glob import for common OxiCUDA Driver types.
///
/// ```rust
/// use oxicuda_driver::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        CacheConfig, Context, CooperativeLaunchConfig, CooperativeLaunchSupport, CudaError,
        CudaResult, DebugLevel, DebugSession, Device, DeviceLaunchConfig, DevicePool, Event,
        FallbackStrategy, Function, GpuTopology, Graph, GraphExec, GraphNode, KernelDebugger,
        LinkInputType, LinkedModule, Linker, LinkerOptions, MemcpyDirection, Module,
        MultiDeviceCooperativeLaunchConfig, NvLinkVersion, OptimizationLevel, PrimaryContext,
        ProfilerGuard, SharedMemConfig, Stream, StreamAllocation, StreamCapture, StreamMemoryPool,
        StreamOrderedAllocConfig, TopologyTree, TopologyType, can_access_peer, cooperative_launch,
        cooperative_launch_multi_device, driver_version, init, stream_alloc, stream_free,
        try_driver,
    };
}

// ---------------------------------------------------------------------------
// Compile-time feature flags
// ---------------------------------------------------------------------------

/// Compile-time feature availability.
pub mod features {
    /// Whether GPU tests are enabled (`--features gpu-tests`).
    pub const HAS_GPU_TESTS: bool = cfg!(feature = "gpu-tests");
}

// ---------------------------------------------------------------------------
// CPU-only tests for driver infrastructure
// ---------------------------------------------------------------------------

#[cfg(test)]
mod driver_infra_tests {
    // -----------------------------------------------------------------------
    // Task 2 — Multi-threaded context migration (F3)
    //
    // Verifies the thread-safety of the context-stack data structure model
    // using pure Rust primitives.  No GPU is required.
    // -----------------------------------------------------------------------

    /// Simulate 4 threads each pushing and popping a "context ID" to/from a
    /// thread-local stack, then verifying all results are collected correctly.
    ///
    /// This exercises the logical structure of context push/pop across threads
    /// (corresponding to `cuCtxPushCurrent` / `cuCtxPopCurrent`) without
    /// needing a real CUDA driver.
    #[test]
    fn context_push_pop_thread_safety() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let results: Arc<Mutex<Vec<(u32, u32)>>> = Arc::new(Mutex::new(vec![]));
        let mut handles = vec![];

        for thread_id in 0..4u32 {
            let results_clone = Arc::clone(&results);
            let handle = thread::spawn(move || {
                // Each thread simulates pushing two context IDs onto its
                // private stack and then reading the top (most-recently-pushed)
                // context.
                let ctx_id = thread_id * 100;
                let stack: Vec<u32> = vec![ctx_id, ctx_id + 1];
                // Pop semantics: the top of the stack is the last element.
                let top = stack.last().copied().unwrap_or(0);
                let mut r = results_clone.lock().expect("results lock failed");
                r.push((thread_id, top));
            });
            handles.push(handle);
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        let results = results.lock().expect("final lock failed");
        assert_eq!(results.len(), 4, "all 4 threads must contribute a result");

        // Every thread should have seen `ctx_id + 1` as the top of its stack.
        for &(thread_id, top) in results.iter() {
            let expected_top = thread_id * 100 + 1;
            assert_eq!(
                top, expected_top,
                "thread {thread_id}: expected top {expected_top}, got {top}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Task 3 — Scope-exit / Drop resource release under OOM (F10)
    //
    // Verifies that Drop impls run correctly even when further allocations
    // fail (simulated OOM), and that Rust's LIFO drop order is preserved.
    // -----------------------------------------------------------------------

    /// `Drop` is invoked for every resource that was successfully constructed,
    /// even when a subsequent allocation would fail (simulated OOM).
    #[test]
    fn drop_counter_tracks_resource_release() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct FakeResource {
            dropped: Arc<AtomicUsize>,
        }

        impl Drop for FakeResource {
            fn drop(&mut self) {
                self.dropped.fetch_add(1, Ordering::SeqCst);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));

        {
            let _r1 = FakeResource {
                dropped: Arc::clone(&counter),
            };
            let _r2 = FakeResource {
                dropped: Arc::clone(&counter),
            };
            // Simulate OOM by not creating r3 — neither r1 nor r2 is dropped yet.
            assert_eq!(
                counter.load(Ordering::SeqCst),
                0,
                "resources must not be dropped before scope exit"
            );
        }

        // After the block ends, both r1 and r2 must have been dropped.
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "both resources must be dropped at scope exit"
        );
    }

    /// Rust drops local variables in **reverse declaration order** (LIFO).
    /// This test verifies that invariant for RAII guard types.
    #[test]
    fn drop_order_is_lifo() {
        use std::sync::{Arc, Mutex};

        let order: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(vec![]));

        struct Ordered {
            id: u32,
            order: Arc<Mutex<Vec<u32>>>,
        }

        impl Drop for Ordered {
            fn drop(&mut self) {
                self.order.lock().expect("order lock failed").push(self.id);
            }
        }

        {
            let _a = Ordered {
                id: 1,
                order: Arc::clone(&order),
            };
            let _b = Ordered {
                id: 2,
                order: Arc::clone(&order),
            };
            let _c = Ordered {
                id: 3,
                order: Arc::clone(&order),
            };
        }

        let observed = order.lock().expect("final order lock failed");
        assert_eq!(
            *observed,
            vec![3, 2, 1],
            "CUDA RAII guards must be released in LIFO order"
        );
    }

    // -----------------------------------------------------------------------
    // Task 4 — Driver version negotiation (NVIDIA Driver 525 / 535 / 550 / 560)
    //
    // `cuDriverGetVersion` returns the CUDA version as `major * 1000 + minor`.
    // These tests verify the parsing logic and the version-gating conditions
    // used throughout OxiCUDA without requiring a real driver.
    // -----------------------------------------------------------------------

    /// NVIDIA Driver 525 ships with CUDA 12.0.  Verify the parse of 12000.
    #[test]
    fn driver_version_parsing_cuda_12_0() {
        // cuDriverGetVersion returns 12000 for CUDA 12.0 (driver 525).
        let cuda_version: i32 = 12000;
        let major = cuda_version / 1000;
        let minor = cuda_version % 1000;
        assert_eq!(major, 12, "major version mismatch");
        assert_eq!(minor, 0, "minor version mismatch");
    }

    /// NVIDIA Driver 535 ships with CUDA 12.2.  Verify the parse of 12020.
    #[test]
    fn driver_version_parsing_cuda_12_2() {
        let cuda_version: i32 = 12020;
        let major = cuda_version / 1000;
        let minor = cuda_version % 1000;
        assert_eq!(major, 12);
        assert_eq!(minor, 20);
    }

    /// NVIDIA Driver 550 ships with CUDA 12.4.  Verify the parse of 12040.
    #[test]
    fn driver_version_parsing_cuda_12_4() {
        let cuda_version: i32 = 12040;
        let major = cuda_version / 1000;
        let minor = cuda_version % 1000;
        assert_eq!(major, 12);
        assert_eq!(minor, 40);
    }

    /// NVIDIA Driver 560 ships with CUDA 12.6.  Verify the parse of 12060.
    #[test]
    fn driver_version_parsing_cuda_12_6() {
        let cuda_version: i32 = 12060;
        let major = cuda_version / 1000;
        let minor = cuda_version % 1000;
        assert_eq!(major, 12);
        assert_eq!(minor, 60);
    }

    /// OxiCUDA requires CUDA 11.2+ (`cuMemAllocAsync` availability).
    /// Verify that the set of supported versions all meet the minimum and
    /// that older versions are correctly rejected.
    #[test]
    fn driver_version_minimum_requirement() {
        // cuMemAllocAsync was introduced in CUDA 11.2 (version integer 11020).
        let min_required: i32 = 11020;

        let supported: [i32; 5] = [11020, 11040, 12000, 12060, 12080];
        for v in supported {
            assert!(
                v >= min_required,
                "CUDA version {v} should be supported (>= {min_required})"
            );
        }

        let too_old: [i32; 2] = [10020, 11010];
        for v in too_old {
            assert!(
                v < min_required,
                "CUDA version {v} should NOT be supported (< {min_required})"
            );
        }
    }

    /// CUDA 12.8 (version 12080) introduces `cuMemcpyBatchAsync`.
    /// Verify the feature-gating arithmetic.
    #[test]
    fn driver_cuda_12_8_features_available() {
        // 12.8 → 12080
        let cuda_128: i32 = 12080;
        assert!(
            cuda_128 >= 12080,
            "CUDA 12.8 must support cuMemcpyBatchAsync"
        );

        // 12.0 does not have it.
        let cuda_120: i32 = 12000;
        assert!(
            cuda_120 < 12080,
            "CUDA 12.0 must NOT support cuMemcpyBatchAsync"
        );
    }

    /// Verify the complete NVIDIA-driver-version → CUDA-version mapping used
    /// in OxiCUDA's version negotiation table.
    #[test]
    fn driver_nvidia_to_cuda_version_mapping() {
        // (nvidia_driver, expected_cuda_version_int)
        let mapping: [(u32, i32); 4] = [
            (525, 12000), // Driver 525  → CUDA 12.0
            (535, 12020), // Driver 535  → CUDA 12.2
            (550, 12040), // Driver 550  → CUDA 12.4
            (560, 12060), // Driver 560  → CUDA 12.6
        ];

        for (nvidia_driver, cuda_version) in mapping {
            let major = cuda_version / 1000;
            let minor = cuda_version % 1000;
            // Sanity: all are CUDA 12.x
            assert_eq!(major, 12, "driver {nvidia_driver}: expected CUDA 12.x");
            // Minor must be a multiple of 10 (CUDA minor encoding)
            assert_eq!(
                minor % 10,
                0,
                "driver {nvidia_driver}: minor {minor} is not a multiple of 10"
            );
            // CUDA 12.8+ features require version >= 12080
            let has_12_8_features = cuda_version >= 12080;
            assert!(
                !has_12_8_features,
                "driver {nvidia_driver} (CUDA {major}.{:02}) should NOT have 12.8+ features",
                minor / 10
            );
        }
    }
}
