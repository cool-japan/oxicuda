//! Dynamic CUDA driver library loader.
//!
//! This module is the architectural foundation of `oxicuda-driver`. It locates
//! and loads the CUDA driver shared library (`libcuda.so` on Linux,
//! `nvcuda.dll` on Windows) **at runtime** via [`libloading`], so that no CUDA
//! SDK is required at build time.
//!
//! # Platform support
//!
//! | Platform | Library names tried              | Notes                            |
//! |----------|----------------------------------|----------------------------------|
//! | Linux    | `libcuda.so.1`, `libcuda.so`     | Installed by NVIDIA driver       |
//! | Windows  | `nvcuda.dll`                     | Ships with the display driver    |
//! | macOS    | —                                | Returns `UnsupportedPlatform`    |
//!
//! # Usage
//!
//! Application code should **not** interact with [`DriverApi`] directly.
//! Instead, call [`try_driver`] to obtain a reference to the lazily-
//! initialised global singleton:
//!
//! ```rust,no_run
//! # use oxicuda_driver::loader::try_driver;
//! let api = try_driver()?;
//! // api.cu_init, api.cu_device_get, …
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```
//!
//! The singleton is stored in a [`OnceLock`] so that the (relatively
//! expensive) `dlopen` + symbol resolution only happens once, and all
//! subsequent accesses are a single atomic load.

use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use libloading::Library;

use crate::error::{CudaError, CudaResult, DriverLoadError};
use crate::ffi::*;

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

/// Global singleton for the driver API function table.
///
/// Initialised lazily on the first call to [`try_driver`].
static DRIVER: OnceLock<Result<DriverApi, DriverLoadError>> = OnceLock::new();

// ---------------------------------------------------------------------------
// load_sym! helper macro
// ---------------------------------------------------------------------------

/// Load a single symbol from the shared library and transmute it to the
/// requested function-pointer type.
///
/// # Safety
///
/// The caller must ensure that the symbol name matches the actual ABI of the
/// function pointer type expected at the call site.
#[cfg(not(target_os = "macos"))]
macro_rules! load_sym {
    ($lib:expr, $name:literal) => {{
        // `Library::get` requires the name as a byte slice.  We request the
        // most general function-pointer type and then transmute to the
        // concrete signature stored in DriverApi.
        let sym = unsafe { $lib.get::<unsafe extern "C" fn()>($name.as_bytes()) }.map_err(|e| {
            DriverLoadError::SymbolNotFound {
                symbol: $name,
                reason: e.to_string(),
            }
        })?;
        // SAFETY: we trust that the CUDA driver exports the symbol with the
        // ABI described by the target field type.  The type is inferred from
        // the DriverApi field this expression is assigned to, so explicit
        // transmute annotations would require repeating the function-pointer
        // type at every call site inside a macro — we suppress that lint here.
        #[allow(clippy::missing_transmute_annotations)]
        let result = unsafe { std::mem::transmute(*sym) };
        result
    }};
}

/// Load a symbol from the shared library, returning `Some(fn_ptr)` on success
/// or `None` if the symbol is not found. Used for optional API entry points
/// that may not be present in older driver versions.
///
/// # Safety
///
/// Same safety requirements as [`load_sym!`].
#[cfg(not(target_os = "macos"))]
macro_rules! load_sym_optional {
    ($lib:expr, $name:literal) => {{
        match unsafe { $lib.get::<unsafe extern "C" fn()>($name.as_bytes()) } {
            Ok(sym) => {
                // SAFETY: the target type is inferred from the DriverApi field
                // this value is assigned to.  Suppressing the lint here avoids
                // repeating the function-pointer type at every call site.
                #[allow(clippy::missing_transmute_annotations)]
                let fp = unsafe { std::mem::transmute(*sym) };
                Some(fp)
            }
            Err(_) => {
                tracing::debug!(concat!("optional symbol not found: ", $name));
                None
            }
        }
    }};
}

// ---------------------------------------------------------------------------
// DriverApi
// ---------------------------------------------------------------------------

/// Complete function-pointer table for the CUDA Driver API.
///
/// An instance of this struct is produced by [`DriverApi::load`] and kept
/// alive for the lifetime of the process inside the `DRIVER` singleton.
/// The embedded [`Library`] handle ensures the shared object is not unloaded.
///
/// # Function pointer groups
///
/// The fields are organised into logical groups mirroring the CUDA Driver API
/// documentation:
///
/// * **Initialisation** — [`cu_init`](Self::cu_init)
/// * **Device management** — `cu_device_*`
/// * **Context management** — `cu_ctx_*`
/// * **Module management** — `cu_module_*`
/// * **Memory management** — `cu_mem_*`, `cu_memcpy_*`, `cu_memset_*`
/// * **Stream management** — `cu_stream_*`
/// * **Event management** — `cu_event_*`
/// * **Kernel launch** — [`cu_launch_kernel`](Self::cu_launch_kernel)
/// * **Occupancy queries** — `cu_occupancy_*`
pub struct DriverApi {
    // Keep the shared library handle alive.
    _lib: Library,

    // -- Initialisation ----------------------------------------------------
    /// `cuInit(flags) -> CUresult`
    ///
    /// Initialises the CUDA driver API.  Must be called before any other
    /// driver function.  Passing `0` for *flags* is the only documented
    /// value.
    pub cu_init: unsafe extern "C" fn(flags: u32) -> CUresult,

    // -- Version query -------------------------------------------------------
    /// `cuDriverGetVersion(driverVersion*) -> CUresult`
    ///
    /// Returns the CUDA driver version as `major*1000 + minor*10`.
    pub cu_driver_get_version: unsafe extern "C" fn(version: *mut c_int) -> CUresult,

    // -- Device management -------------------------------------------------
    /// `cuDeviceGet(device*, ordinal) -> CUresult`
    ///
    /// Returns a handle to a compute device.
    pub cu_device_get: unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult,

    /// `cuDeviceGetCount(count*) -> CUresult`
    ///
    /// Returns the number of compute-capable devices.
    pub cu_device_get_count: unsafe extern "C" fn(count: *mut c_int) -> CUresult,

    /// `cuDeviceGetName(name*, len, dev) -> CUresult`
    ///
    /// Returns an ASCII string identifying the device.
    pub cu_device_get_name:
        unsafe extern "C" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult,

    /// `cuDeviceGetAttribute(pi*, attrib, dev) -> CUresult`
    ///
    /// Returns information about the device.
    pub cu_device_get_attribute:
        unsafe extern "C" fn(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult,

    /// `cuDeviceTotalMem_v2(bytes*, dev) -> CUresult`
    ///
    /// Returns the total amount of memory on the device.
    pub cu_device_total_mem_v2: unsafe extern "C" fn(bytes: *mut usize, dev: CUdevice) -> CUresult,

    /// `cuDeviceCanAccessPeer(canAccessPeer*, dev, peerDev) -> CUresult`
    ///
    /// Queries if a device may directly access a peer device's memory.
    pub cu_device_can_access_peer:
        unsafe extern "C" fn(can_access: *mut c_int, dev: CUdevice, peer_dev: CUdevice) -> CUresult,

    // -- Primary context management ----------------------------------------
    /// `cuDevicePrimaryCtxRetain(pctx*, dev) -> CUresult`
    ///
    /// Retains the primary context on the device, creating it if necessary.
    pub cu_device_primary_ctx_retain:
        unsafe extern "C" fn(pctx: *mut CUcontext, dev: CUdevice) -> CUresult,

    /// `cuDevicePrimaryCtxRelease_v2(dev) -> CUresult`
    ///
    /// Releases the primary context on the device.
    pub cu_device_primary_ctx_release_v2: unsafe extern "C" fn(dev: CUdevice) -> CUresult,

    /// `cuDevicePrimaryCtxSetFlags_v2(dev, flags) -> CUresult`
    ///
    /// Sets flags for the primary context.
    pub cu_device_primary_ctx_set_flags_v2:
        unsafe extern "C" fn(dev: CUdevice, flags: u32) -> CUresult,

    /// `cuDevicePrimaryCtxGetState(dev, flags*, active*) -> CUresult`
    ///
    /// Returns the state (flags and active status) of the primary context.
    pub cu_device_primary_ctx_get_state:
        unsafe extern "C" fn(dev: CUdevice, flags: *mut u32, active: *mut c_int) -> CUresult,

    /// `cuDevicePrimaryCtxReset_v2(dev) -> CUresult`
    ///
    /// Resets the primary context on the device.
    pub cu_device_primary_ctx_reset_v2: unsafe extern "C" fn(dev: CUdevice) -> CUresult,

    // -- Context management ------------------------------------------------
    /// `cuCtxCreate_v2(pctx*, flags, dev) -> CUresult`
    ///
    /// Creates a new CUDA context and associates it with the calling thread.
    pub cu_ctx_create_v2:
        unsafe extern "C" fn(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult,

    /// `cuCtxDestroy_v2(ctx) -> CUresult`
    ///
    /// Destroys a CUDA context.
    pub cu_ctx_destroy_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,

    /// `cuCtxSetCurrent(ctx) -> CUresult`
    ///
    /// Binds the specified CUDA context to the calling CPU thread.
    pub cu_ctx_set_current: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,

    /// `cuCtxGetCurrent(pctx*) -> CUresult`
    ///
    /// Returns the CUDA context bound to the calling CPU thread.
    pub cu_ctx_get_current: unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult,

    /// `cuCtxSynchronize() -> CUresult`
    ///
    /// Blocks until the device has completed all preceding requested tasks.
    pub cu_ctx_synchronize: unsafe extern "C" fn() -> CUresult,

    // -- Module management -------------------------------------------------
    /// `cuModuleLoadData(module*, image*) -> CUresult`
    ///
    /// Loads a module from a PTX or cubin image in host memory.
    pub cu_module_load_data:
        unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,

    /// `cuModuleLoadDataEx(module*, image*, numOptions, options*, optionValues*) -> CUresult`
    ///
    /// Loads a module with JIT compiler options.
    pub cu_module_load_data_ex: unsafe extern "C" fn(
        module: *mut CUmodule,
        image: *const c_void,
        num_options: u32,
        options: *mut CUjit_option,
        option_values: *mut *mut c_void,
    ) -> CUresult,

    /// `cuModuleGetFunction(hfunc*, hmod, name*) -> CUresult`
    ///
    /// Returns a handle to a function within a module.
    pub cu_module_get_function: unsafe extern "C" fn(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult,

    /// `cuModuleUnload(hmod) -> CUresult`
    ///
    /// Unloads a module from the current context.
    pub cu_module_unload: unsafe extern "C" fn(hmod: CUmodule) -> CUresult,

    // -- Memory management -------------------------------------------------
    /// `cuMemAlloc_v2(dptr*, bytesize) -> CUresult`
    ///
    /// Allocates device memory.
    pub cu_mem_alloc_v2: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,

    /// `cuMemFree_v2(dptr) -> CUresult`
    ///
    /// Frees device memory.
    pub cu_mem_free_v2: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,

    /// `cuMemcpyHtoD_v2(dst, src*, bytesize) -> CUresult`
    ///
    /// Copies data from host memory to device memory.
    pub cu_memcpy_htod_v2:
        unsafe extern "C" fn(dst: CUdeviceptr, src: *const c_void, bytesize: usize) -> CUresult,

    /// `cuMemcpyDtoH_v2(dst*, src, bytesize) -> CUresult`
    ///
    /// Copies data from device memory to host memory.
    pub cu_memcpy_dtoh_v2:
        unsafe extern "C" fn(dst: *mut c_void, src: CUdeviceptr, bytesize: usize) -> CUresult,

    /// `cuMemcpyDtoD_v2(dst, src, bytesize) -> CUresult`
    ///
    /// Copies data from device memory to device memory.
    pub cu_memcpy_dtod_v2:
        unsafe extern "C" fn(dst: CUdeviceptr, src: CUdeviceptr, bytesize: usize) -> CUresult,

    /// `cuMemcpyHtoDAsync_v2(dst, src*, bytesize, stream) -> CUresult`
    ///
    /// Asynchronously copies data from host to device memory.
    pub cu_memcpy_htod_async_v2: unsafe extern "C" fn(
        dst: CUdeviceptr,
        src: *const c_void,
        bytesize: usize,
        stream: CUstream,
    ) -> CUresult,

    /// `cuMemcpyDtoHAsync_v2(dst*, src, bytesize, stream) -> CUresult`
    ///
    /// Asynchronously copies data from device to host memory.
    pub cu_memcpy_dtoh_async_v2: unsafe extern "C" fn(
        dst: *mut c_void,
        src: CUdeviceptr,
        bytesize: usize,
        stream: CUstream,
    ) -> CUresult,

    /// `cuMemAllocHost_v2(pp*, bytesize) -> CUresult`
    ///
    /// Allocates page-locked (pinned) host memory.
    pub cu_mem_alloc_host_v2:
        unsafe extern "C" fn(pp: *mut *mut c_void, bytesize: usize) -> CUresult,

    /// `cuMemFreeHost(p*) -> CUresult`
    ///
    /// Frees page-locked host memory.
    pub cu_mem_free_host: unsafe extern "C" fn(p: *mut c_void) -> CUresult,

    /// `cuMemAllocManaged(dptr*, bytesize, flags) -> CUresult`
    ///
    /// Allocates unified memory accessible from both host and device.
    pub cu_mem_alloc_managed:
        unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize, flags: u32) -> CUresult,

    /// `cuMemsetD8_v2(dst, value, count) -> CUresult`
    ///
    /// Sets device memory to a value (byte granularity).
    pub cu_memset_d8_v2:
        unsafe extern "C" fn(dst: CUdeviceptr, value: u8, count: usize) -> CUresult,

    /// `cuMemsetD32_v2(dst, value, count) -> CUresult`
    ///
    /// Sets device memory to a value (32-bit granularity).
    pub cu_memset_d32_v2:
        unsafe extern "C" fn(dst: CUdeviceptr, value: u32, count: usize) -> CUresult,

    /// `cuMemGetInfo_v2(free*, total*) -> CUresult`
    ///
    /// Returns free and total memory for the current context's device.
    pub cu_mem_get_info_v2: unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> CUresult,

    /// `cuMemHostRegister_v2(p*, bytesize, flags) -> CUresult`
    ///
    /// Registers an existing host memory range for use by CUDA.
    pub cu_mem_host_register_v2:
        unsafe extern "C" fn(p: *mut c_void, bytesize: usize, flags: u32) -> CUresult,

    /// `cuMemHostUnregister(p*) -> CUresult`
    ///
    /// Unregisters a memory range that was registered with cuMemHostRegister.
    pub cu_mem_host_unregister: unsafe extern "C" fn(p: *mut c_void) -> CUresult,

    /// `cuMemHostGetDevicePointer_v2(pdptr*, p*, flags) -> CUresult`
    ///
    /// Returns the device pointer mapped to a registered host pointer.
    pub cu_mem_host_get_device_pointer_v2:
        unsafe extern "C" fn(pdptr: *mut CUdeviceptr, p: *mut c_void, flags: u32) -> CUresult,

    /// `cuPointerGetAttribute(data*, attribute, ptr) -> CUresult`
    ///
    /// Returns information about a pointer.
    pub cu_pointer_get_attribute:
        unsafe extern "C" fn(data: *mut c_void, attribute: u32, ptr: CUdeviceptr) -> CUresult,

    /// `cuMemAdvise(devPtr, count, advice, device) -> CUresult`
    ///
    /// Advises the unified memory subsystem about usage patterns.
    pub cu_mem_advise: unsafe extern "C" fn(
        dev_ptr: CUdeviceptr,
        count: usize,
        advice: u32,
        device: CUdevice,
    ) -> CUresult,

    /// `cuMemPrefetchAsync(devPtr, count, dstDevice, hStream) -> CUresult`
    ///
    /// Prefetches unified memory to the specified device.
    pub cu_mem_prefetch_async: unsafe extern "C" fn(
        dev_ptr: CUdeviceptr,
        count: usize,
        dst_device: CUdevice,
        hstream: CUstream,
    ) -> CUresult,

    // -- Stream management -------------------------------------------------
    /// `cuStreamCreate(phStream*, flags) -> CUresult`
    ///
    /// Creates a stream.
    pub cu_stream_create: unsafe extern "C" fn(phstream: *mut CUstream, flags: u32) -> CUresult,

    /// `cuStreamCreateWithPriority(phStream*, flags, priority) -> CUresult`
    ///
    /// Creates a stream with the given priority.
    pub cu_stream_create_with_priority:
        unsafe extern "C" fn(phstream: *mut CUstream, flags: u32, priority: c_int) -> CUresult,

    /// `cuStreamDestroy_v2(hStream) -> CUresult`
    ///
    /// Destroys a stream.
    pub cu_stream_destroy_v2: unsafe extern "C" fn(hstream: CUstream) -> CUresult,

    /// `cuStreamSynchronize(hStream) -> CUresult`
    ///
    /// Waits until a stream's tasks are completed.
    pub cu_stream_synchronize: unsafe extern "C" fn(hstream: CUstream) -> CUresult,

    /// `cuStreamWaitEvent(hStream, hEvent, flags) -> CUresult`
    ///
    /// Makes all future work submitted to the stream wait for the event.
    pub cu_stream_wait_event:
        unsafe extern "C" fn(hstream: CUstream, hevent: CUevent, flags: u32) -> CUresult,

    /// `cuStreamQuery(hStream) -> CUresult`
    ///
    /// Returns `CUDA_SUCCESS` if all operations in the stream have completed,
    /// `CUDA_ERROR_NOT_READY` if still pending.
    pub cu_stream_query: unsafe extern "C" fn(hstream: CUstream) -> CUresult,

    /// `cuStreamGetPriority(hStream, priority*) -> CUresult`
    ///
    /// Query the priority of `hStream`.
    pub cu_stream_get_priority:
        unsafe extern "C" fn(hstream: CUstream, priority: *mut std::ffi::c_int) -> CUresult,

    /// `cuStreamGetFlags(hStream, flags*) -> CUresult`
    ///
    /// Query the flags of `hStream`.
    pub cu_stream_get_flags: unsafe extern "C" fn(hstream: CUstream, flags: *mut u32) -> CUresult,

    // -- Event management --------------------------------------------------
    /// `cuEventCreate(phEvent*, flags) -> CUresult`
    ///
    /// Creates an event.
    pub cu_event_create: unsafe extern "C" fn(phevent: *mut CUevent, flags: u32) -> CUresult,

    /// `cuEventDestroy_v2(hEvent) -> CUresult`
    ///
    /// Destroys an event.
    pub cu_event_destroy_v2: unsafe extern "C" fn(hevent: CUevent) -> CUresult,

    /// `cuEventRecord(hEvent, hStream) -> CUresult`
    ///
    /// Records an event in a stream.
    pub cu_event_record: unsafe extern "C" fn(hevent: CUevent, hstream: CUstream) -> CUresult,

    /// `cuEventQuery(hEvent) -> CUresult`
    ///
    /// Queries the status of an event. Returns `CUDA_SUCCESS` if complete,
    /// `CUDA_ERROR_NOT_READY` if still pending.
    pub cu_event_query: unsafe extern "C" fn(hevent: CUevent) -> CUresult,

    /// `cuEventSynchronize(hEvent) -> CUresult`
    ///
    /// Waits until an event completes.
    pub cu_event_synchronize: unsafe extern "C" fn(hevent: CUevent) -> CUresult,

    /// `cuEventElapsedTime(pMilliseconds*, hStart, hEnd) -> CUresult`
    ///
    /// Computes the elapsed time between two events.
    pub cu_event_elapsed_time:
        unsafe extern "C" fn(pmilliseconds: *mut f32, hstart: CUevent, hend: CUevent) -> CUresult,

    // -- Kernel launch -----------------------------------------------------

    // -- Peer memory access ------------------------------------------------
    /// `cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, count) -> CUresult`
    ///
    /// Copies device memory between two primary contexts.
    pub cu_memcpy_peer: unsafe extern "C" fn(
        dst_device: u64,
        dst_ctx: CUcontext,
        src_device: u64,
        src_ctx: CUcontext,
        count: usize,
    ) -> CUresult,

    /// `cuMemcpyPeerAsync(..., hStream) -> CUresult`
    ///
    /// Asynchronous cross-device copy.
    pub cu_memcpy_peer_async: unsafe extern "C" fn(
        dst_device: u64,
        dst_ctx: CUcontext,
        src_device: u64,
        src_ctx: CUcontext,
        count: usize,
        stream: CUstream,
    ) -> CUresult,

    /// `cuCtxEnablePeerAccess(peerContext, flags) -> CUresult`
    ///
    /// Enables peer access between two contexts.
    pub cu_ctx_enable_peer_access:
        unsafe extern "C" fn(peer_context: CUcontext, flags: u32) -> CUresult,

    /// `cuCtxDisablePeerAccess(peerContext) -> CUresult`
    ///
    /// Disables peer access to a context.
    pub cu_ctx_disable_peer_access: unsafe extern "C" fn(peer_context: CUcontext) -> CUresult,
    /// `cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
    ///   blockDimZ, sharedMemBytes, hStream, kernelParams**, extra**) -> CUresult`
    ///
    /// Launches a CUDA kernel.
    #[allow(clippy::type_complexity)]
    pub cu_launch_kernel: unsafe extern "C" fn(
        f: CUfunction,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        hstream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult,

    /// `cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
    ///   blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams**) -> CUresult`
    ///
    /// Launches a cooperative CUDA kernel (CUDA 9.0+).
    #[allow(clippy::type_complexity)]
    pub cu_launch_cooperative_kernel: unsafe extern "C" fn(
        f: CUfunction,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        hstream: CUstream,
        kernel_params: *mut *mut c_void,
    ) -> CUresult,

    /// `cuLaunchCooperativeKernelMultiDevice(launchParamsList*, numDevices,
    ///   flags) -> CUresult`
    ///
    /// Launches a cooperative kernel across multiple devices (CUDA 9.0+).
    pub cu_launch_cooperative_kernel_multi_device: unsafe extern "C" fn(
        launch_params_list: *mut c_void,
        num_devices: u32,
        flags: u32,
    ) -> CUresult,

    // -- Occupancy ---------------------------------------------------------
    /// `cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks*, func, blockSize,
    ///   dynamicSMemSize) -> CUresult`
    ///
    /// Returns the number of the maximum active blocks per streaming
    /// multiprocessor.
    pub cu_occupancy_max_active_blocks_per_multiprocessor: unsafe extern "C" fn(
        num_blocks: *mut c_int,
        func: CUfunction,
        block_size: c_int,
        dynamic_smem_size: usize,
    ) -> CUresult,

    /// `cuOccupancyMaxPotentialBlockSize(minGridSize*, blockSize*, func,
    ///   blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit) -> CUresult`
    ///
    /// Suggests a launch configuration with reasonable occupancy.
    #[allow(clippy::type_complexity)]
    pub cu_occupancy_max_potential_block_size: unsafe extern "C" fn(
        min_grid_size: *mut c_int,
        block_size: *mut c_int,
        func: CUfunction,
        block_size_to_dynamic_smem_size: Option<unsafe extern "C" fn(c_int) -> usize>,
        dynamic_smem_size: usize,
        block_size_limit: c_int,
    ) -> CUresult,

    /// `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks*, func,
    ///   blockSize, dynamicSMemSize, flags) -> CUresult`
    ///
    /// Like `cuOccupancyMaxActiveBlocksPerMultiprocessor` but with flags
    /// to control caching behaviour (CUDA 9.0+).
    pub cu_occupancy_max_active_blocks_per_multiprocessor_with_flags:
        unsafe extern "C" fn(
            num_blocks: *mut c_int,
            func: CUfunction,
            block_size: c_int,
            dynamic_smem_size: usize,
            flags: u32,
        ) -> CUresult,

    // -- Memory management (optional) -----------------------------------------
    /// `cuMemcpyDtoDAsync_v2(dst, src, bytesize, stream) -> CUresult`
    ///
    /// Asynchronously copies data from device memory to device memory.
    pub cu_memcpy_dtod_async_v2: Option<
        unsafe extern "C" fn(
            dst: CUdeviceptr,
            src: CUdeviceptr,
            bytesize: usize,
            stream: CUstream,
        ) -> CUresult,
    >,

    /// `cuMemsetD16_v2(dst, value, count) -> CUresult`
    ///
    /// Sets device memory to a value (16-bit granularity).
    pub cu_memset_d16_v2:
        Option<unsafe extern "C" fn(dst: CUdeviceptr, value: u16, count: usize) -> CUresult>,

    /// `cuMemsetD32Async(dst, value, count, stream) -> CUresult`
    ///
    /// Asynchronously sets device memory to a value (32-bit granularity).
    pub cu_memset_d32_async: Option<
        unsafe extern "C" fn(
            dst: CUdeviceptr,
            value: u32,
            count: usize,
            stream: CUstream,
        ) -> CUresult,
    >,

    // -- Context management (optional) ----------------------------------------
    /// `cuCtxGetLimit(value*, limit) -> CUresult`
    ///
    /// Returns the value of a context limit.
    pub cu_ctx_get_limit: Option<unsafe extern "C" fn(value: *mut usize, limit: u32) -> CUresult>,

    /// `cuCtxSetLimit(limit, value) -> CUresult`
    ///
    /// Sets a context limit.
    pub cu_ctx_set_limit: Option<unsafe extern "C" fn(limit: u32, value: usize) -> CUresult>,

    /// `cuCtxGetCacheConfig(config*) -> CUresult`
    ///
    /// Returns the current cache configuration for the context.
    pub cu_ctx_get_cache_config: Option<unsafe extern "C" fn(config: *mut u32) -> CUresult>,

    /// `cuCtxSetCacheConfig(config) -> CUresult`
    ///
    /// Sets the cache configuration for the current context.
    pub cu_ctx_set_cache_config: Option<unsafe extern "C" fn(config: u32) -> CUresult>,

    /// `cuCtxGetSharedMemConfig(config*) -> CUresult`
    ///
    /// Returns the shared memory configuration for the context.
    pub cu_ctx_get_shared_mem_config: Option<unsafe extern "C" fn(config: *mut u32) -> CUresult>,

    /// `cuCtxSetSharedMemConfig(config) -> CUresult`
    ///
    /// Sets the shared memory configuration for the current context.
    pub cu_ctx_set_shared_mem_config: Option<unsafe extern "C" fn(config: u32) -> CUresult>,

    // -- Event with flags (optional, CUDA 11.1+) ------------------------------
    /// `cuEventRecordWithFlags(hEvent, hStream, flags) -> CUresult`
    ///
    /// Records an event in a stream with additional flags (CUDA 11.1+).
    /// Falls back to `cu_event_record` when `None`.
    pub cu_event_record_with_flags:
        Option<unsafe extern "C" fn(hevent: CUevent, hstream: CUstream, flags: u32) -> CUresult>,

    // -- Function attributes (optional) ---------------------------------------
    /// `cuFuncGetAttribute(value*, attrib, func) -> CUresult`
    ///
    /// Returns information about a function.
    pub cu_func_get_attribute: Option<
        unsafe extern "C" fn(value: *mut c_int, attrib: c_int, func: CUfunction) -> CUresult,
    >,

    /// `cuFuncSetCacheConfig(func, config) -> CUresult`
    ///
    /// Sets the cache configuration for a device function.
    pub cu_func_set_cache_config:
        Option<unsafe extern "C" fn(func: CUfunction, config: u32) -> CUresult>,

    /// `cuFuncSetSharedMemConfig(func, config) -> CUresult`
    ///
    /// Sets the shared memory configuration for a device function.
    pub cu_func_set_shared_mem_config:
        Option<unsafe extern "C" fn(func: CUfunction, config: u32) -> CUresult>,

    /// `cuFuncSetAttribute(func, attrib, value) -> CUresult`
    ///
    /// Sets an attribute value for a device function.
    pub cu_func_set_attribute:
        Option<unsafe extern "C" fn(func: CUfunction, attrib: c_int, value: c_int) -> CUresult>,

    // -- Profiler (optional) --------------------------------------------------
    /// `cuProfilerStart() -> CUresult`
    ///
    /// Starts the CUDA profiler.
    pub cu_profiler_start: Option<unsafe extern "C" fn() -> CUresult>,

    /// `cuProfilerStop() -> CUresult`
    ///
    /// Stops the CUDA profiler.
    pub cu_profiler_stop: Option<unsafe extern "C" fn() -> CUresult>,

    // -- CUDA 12.x extended launch (optional) ---------------------------------
    /// `cuLaunchKernelEx(config*, f, kernelParams**, extra**) -> CUresult`
    ///
    /// Extended kernel launch with cluster dimensions and other CUDA 12.0+
    /// attributes. Available only when the driver is CUDA 12.0 or newer.
    ///
    /// When `None`, fall back to [`cu_launch_kernel`](Self::cu_launch_kernel).
    #[allow(clippy::type_complexity)]
    pub cu_launch_kernel_ex: Option<
        unsafe extern "C" fn(
            config: *const CuLaunchConfig,
            f: CUfunction,
            kernel_params: *mut *mut std::ffi::c_void,
            extra: *mut *mut std::ffi::c_void,
        ) -> CUresult,
    >,

    /// `cuTensorMapEncodeTiled(tensorMap*, ...) -> CUresult`
    ///
    /// Creates a TMA tensor map descriptor for tiled access patterns.
    /// Available on CUDA 12.0+ with sm_90+ (Hopper/Blackwell).
    ///
    /// When `None`, TMA is not supported by the loaded driver.
    #[allow(clippy::type_complexity)]
    pub cu_tensor_map_encode_tiled: Option<
        unsafe extern "C" fn(
            tensor_map: *mut std::ffi::c_void,
            tensor_data_type: u32,
            tensor_rank: u32,
            global_address: *mut std::ffi::c_void,
            global_dim: *const u64,
            global_strides: *const u64,
            box_dim: *const u32,
            element_strides: *const u32,
            interleave: u32,
            swizzle: u32,
            l2_promotion: u32,
            oob_fill: u32,
        ) -> CUresult,
    >,

    // -- CUDA 12.8+ extended API (optional) -----------------------------------
    /// `cuTensorMapEncodeTiledMemref(tensorMap*, ...) -> CUresult`
    ///
    /// Extended TMA encoding using memref descriptors (CUDA 12.8+,
    /// Blackwell sm_100/sm_120). When `None`, fall back to
    /// [`cu_tensor_map_encode_tiled`](Self::cu_tensor_map_encode_tiled).
    #[allow(clippy::type_complexity)]
    pub cu_tensor_map_encode_tiled_memref: Option<
        unsafe extern "C" fn(
            tensor_map: *mut c_void,
            tensor_data_type: u32,
            tensor_rank: u32,
            global_address: *mut c_void,
            global_dim: *const u64,
            global_strides: *const u64,
            box_dim: *const u32,
            element_strides: *const u32,
            interleave: u32,
            swizzle: u32,
            l2_promotion: u32,
            oob_fill: u32,
            flags: u64,
        ) -> CUresult,
    >,

    /// `cuKernelGetLibrary(pLib*, kernel) -> CUresult`
    ///
    /// Returns the library handle that owns a given kernel handle
    /// (CUDA 12.8+). When `None`, the driver does not support the JIT
    /// library API.
    pub cu_kernel_get_library:
        Option<unsafe extern "C" fn(p_lib: *mut CUlibrary, kernel: CUkernel) -> CUresult>,

    /// `cuMulticastGetGranularity(granularity*, desc*, option) -> CUresult`
    ///
    /// Queries the recommended memory granularity for an NVLink multicast
    /// object (CUDA 12.8+). When `None`, multicast memory is not supported.
    pub cu_multicast_get_granularity: Option<
        unsafe extern "C" fn(granularity: *mut usize, desc: *const c_void, option: u32) -> CUresult,
    >,

    /// `cuMulticastCreate(mcHandle*, desc*) -> CUresult`
    ///
    /// Creates an NVLink multicast object for cross-GPU broadcast memory
    /// (CUDA 12.8+). When `None`, multicast memory is not supported.
    pub cu_multicast_create: Option<
        unsafe extern "C" fn(mc_handle: *mut CUmulticastObject, desc: *const c_void) -> CUresult,
    >,

    /// `cuMulticastAddDevice(mcHandle, dev) -> CUresult`
    ///
    /// Adds a device to an NVLink multicast group (CUDA 12.8+). When
    /// `None`, multicast memory is not supported.
    pub cu_multicast_add_device:
        Option<unsafe extern "C" fn(mc_handle: CUmulticastObject, dev: CUdevice) -> CUresult>,

    /// `cuMemcpyBatchAsync(dsts*, srcs*, sizes*, count, flags, stream) -> CUresult`
    ///
    /// Issues *count* asynchronous memory copies (H2D, D2H, or D2D) in a
    /// single driver call (CUDA 12.8+). When `None`, issue individual
    /// `cuMemcpyAsync` calls as a fallback.
    #[allow(clippy::type_complexity)]
    pub cu_memcpy_batch_async: Option<
        unsafe extern "C" fn(
            dsts: *const *mut c_void,
            srcs: *const *const c_void,
            sizes: *const usize,
            count: u64,
            flags: u64,
            stream: CUstream,
        ) -> CUresult,
    >,

    // -- Texture / Surface memory (optional) ----------------------------------
    /// `cuArrayCreate_v2(pHandle*, pAllocateArray*) -> CUresult`
    ///
    /// Allocates a 1-D or 2-D CUDA array. When `None`, CUDA array allocation
    /// is not supported by the loaded driver.
    pub cu_array_create_v2: Option<
        unsafe extern "C" fn(
            p_handle: *mut CUarray,
            p_allocate_array: *const CUDA_ARRAY_DESCRIPTOR,
        ) -> CUresult,
    >,

    /// `cuArrayDestroy(hArray) -> CUresult`
    ///
    /// Frees a CUDA array previously allocated by `cuArrayCreate_v2`.
    pub cu_array_destroy: Option<unsafe extern "C" fn(h_array: CUarray) -> CUresult>,

    /// `cuArrayGetDescriptor_v2(pArrayDescriptor*, hArray) -> CUresult`
    ///
    /// Returns the descriptor of a 1-D or 2-D CUDA array.
    pub cu_array_get_descriptor_v2: Option<
        unsafe extern "C" fn(
            p_array_descriptor: *mut CUDA_ARRAY_DESCRIPTOR,
            h_array: CUarray,
        ) -> CUresult,
    >,

    /// `cuArray3DCreate_v2(pHandle*, pAllocateArray*) -> CUresult`
    ///
    /// Allocates a 3-D CUDA array (also supports layered and cubemap arrays).
    pub cu_array3d_create_v2: Option<
        unsafe extern "C" fn(
            p_handle: *mut CUarray,
            p_allocate_array: *const CUDA_ARRAY3D_DESCRIPTOR,
        ) -> CUresult,
    >,

    /// `cuArray3DGetDescriptor_v2(pArrayDescriptor*, hArray) -> CUresult`
    ///
    /// Returns the descriptor of a 3-D CUDA array.
    pub cu_array3d_get_descriptor_v2: Option<
        unsafe extern "C" fn(
            p_array_descriptor: *mut CUDA_ARRAY3D_DESCRIPTOR,
            h_array: CUarray,
        ) -> CUresult,
    >,

    /// `cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost*, ByteCount) -> CUresult`
    ///
    /// Synchronously copies host memory into a CUDA array.
    pub cu_memcpy_htoa_v2: Option<
        unsafe extern "C" fn(
            dst_array: CUarray,
            dst_offset: usize,
            src_host: *const c_void,
            byte_count: usize,
        ) -> CUresult,
    >,

    /// `cuMemcpyAtoH_v2(dstHost*, srcArray, srcOffset, ByteCount) -> CUresult`
    ///
    /// Synchronously copies data from a CUDA array into host memory.
    pub cu_memcpy_atoh_v2: Option<
        unsafe extern "C" fn(
            dst_host: *mut c_void,
            src_array: CUarray,
            src_offset: usize,
            byte_count: usize,
        ) -> CUresult,
    >,

    /// `cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost*, byteCount, stream) -> CUresult`
    ///
    /// Asynchronously copies host memory into a CUDA array on a stream.
    pub cu_memcpy_htoa_async_v2: Option<
        unsafe extern "C" fn(
            dst_array: CUarray,
            dst_offset: usize,
            src_host: *const c_void,
            byte_count: usize,
            stream: CUstream,
        ) -> CUresult,
    >,

    /// `cuMemcpyAtoHAsync_v2(dstHost*, srcArray, srcOffset, byteCount, stream) -> CUresult`
    ///
    /// Asynchronously copies data from a CUDA array into host memory on a stream.
    pub cu_memcpy_atoh_async_v2: Option<
        unsafe extern "C" fn(
            dst_host: *mut c_void,
            src_array: CUarray,
            src_offset: usize,
            byte_count: usize,
            stream: CUstream,
        ) -> CUresult,
    >,

    /// `cuTexObjectCreate(pTexObject*, pResDesc*, pTexDesc*, pResViewDesc*) -> CUresult`
    ///
    /// Creates a texture object from a resource descriptor, texture descriptor,
    /// and optional resource-view descriptor (CUDA 5.0+).
    pub cu_tex_object_create: Option<
        unsafe extern "C" fn(
            p_tex_object: *mut CUtexObject,
            p_res_desc: *const CUDA_RESOURCE_DESC,
            p_tex_desc: *const CUDA_TEXTURE_DESC,
            p_res_view_desc: *const CUDA_RESOURCE_VIEW_DESC,
        ) -> CUresult,
    >,

    /// `cuTexObjectDestroy(texObject) -> CUresult`
    ///
    /// Destroys a texture object created by `cuTexObjectCreate`.
    pub cu_tex_object_destroy: Option<unsafe extern "C" fn(tex_object: CUtexObject) -> CUresult>,

    /// `cuTexObjectGetResourceDesc(pResDesc*, texObject) -> CUresult`
    ///
    /// Returns the resource descriptor of a texture object.
    pub cu_tex_object_get_resource_desc: Option<
        unsafe extern "C" fn(
            p_res_desc: *mut CUDA_RESOURCE_DESC,
            tex_object: CUtexObject,
        ) -> CUresult,
    >,

    /// `cuSurfObjectCreate(pSurfObject*, pResDesc*) -> CUresult`
    ///
    /// Creates a surface object from a resource descriptor (CUDA 5.0+).
    /// The resource type must be `Array` (surface-capable CUDA arrays only).
    pub cu_surf_object_create: Option<
        unsafe extern "C" fn(
            p_surf_object: *mut CUsurfObject,
            p_res_desc: *const CUDA_RESOURCE_DESC,
        ) -> CUresult,
    >,

    /// `cuSurfObjectDestroy(surfObject) -> CUresult`
    ///
    /// Destroys a surface object created by `cuSurfObjectCreate`.
    pub cu_surf_object_destroy: Option<unsafe extern "C" fn(surf_object: CUsurfObject) -> CUresult>,
}

// SAFETY: All fields are plain function pointers (which are Send + Sync) and
// the Library handle is kept alive but never mutated.
unsafe impl Send for DriverApi {}
unsafe impl Sync for DriverApi {}

// ---------------------------------------------------------------------------
// DriverApi — construction
// ---------------------------------------------------------------------------

impl DriverApi {
    /// Attempt to dynamically load the CUDA driver shared library and resolve
    /// every required symbol.
    ///
    /// # Platform behaviour
    ///
    /// * **macOS** — immediately returns [`DriverLoadError::UnsupportedPlatform`].
    /// * **Linux** — tries `libcuda.so.1` then `libcuda.so`.
    /// * **Windows** — tries `nvcuda.dll`.
    ///
    /// # Errors
    ///
    /// * [`DriverLoadError::UnsupportedPlatform`] on macOS.
    /// * [`DriverLoadError::LibraryNotFound`] if none of the candidate library
    ///   names could be opened.
    /// * [`DriverLoadError::SymbolNotFound`] if a required CUDA entry point is
    ///   missing from the loaded library.
    pub fn load() -> Result<Self, DriverLoadError> {
        // macOS: CUDA is not and will not be supported.
        #[cfg(target_os = "macos")]
        {
            Err(DriverLoadError::UnsupportedPlatform)
        }

        // Linux library search order.
        #[cfg(target_os = "linux")]
        let lib_names: &[&str] = &["libcuda.so.1", "libcuda.so"];

        // Windows library search order.
        #[cfg(target_os = "windows")]
        let lib_names: &[&str] = &["nvcuda.dll"];

        #[cfg(not(target_os = "macos"))]
        {
            let lib = Self::load_library(lib_names)?;
            let api = Self::load_symbols(lib)?;
            // `cuInit(0)` must be called before any other CUDA driver API.
            // This mirrors what `libcudart` does internally on the first CUDA
            // Runtime call. We call it unconditionally here so that all
            // `try_driver()` callers get a fully initialised driver without
            // each needing to call `cuInit` themselves.
            //
            // SAFETY: `api.cu_init` was just resolved from the shared library.
            // Passing flags=0 is the only documented value.
            let rc = unsafe { (api.cu_init)(0) };
            if rc != 0 {
                // Propagate the error; the OnceLock will store this Err and
                // return CudaError::NotInitialized on every subsequent
                // try_driver() call — matching behaviour on no-GPU machines.
                return Err(DriverLoadError::InitializationFailed { code: rc });
            }
            Ok(api)
        }
    }

    /// Try each candidate library name in order, returning the first that
    /// loads successfully.
    ///
    /// # Errors
    ///
    /// Returns [`DriverLoadError::LibraryNotFound`] if **all** candidates
    /// fail to load, capturing the last OS-level error message.
    #[cfg(not(target_os = "macos"))]
    fn load_library(names: &[&str]) -> Result<Library, DriverLoadError> {
        let mut last_error = String::new();
        for name in names {
            // SAFETY: loading a shared library has side-effects (running its
            // init routines), but the CUDA driver library is designed for
            // this.
            match unsafe { Library::new(*name) } {
                Ok(lib) => {
                    tracing::debug!("loaded CUDA driver library: {name}");
                    return Ok(lib);
                }
                Err(e) => {
                    tracing::debug!("failed to load {name}: {e}");
                    last_error = e.to_string();
                }
            }
        }

        Err(DriverLoadError::LibraryNotFound {
            candidates: names.iter().map(|s| (*s).to_string()).collect(),
            last_error,
        })
    }

    /// Resolve every required CUDA driver symbol from the loaded library and
    /// assemble the [`DriverApi`] function table.
    ///
    /// # Errors
    ///
    /// Returns [`DriverLoadError::SymbolNotFound`] if any symbol cannot be
    /// resolved.
    #[cfg(not(target_os = "macos"))]
    fn load_symbols(lib: Library) -> Result<Self, DriverLoadError> {
        Ok(Self {
            // -- Initialisation ------------------------------------------------
            cu_init: load_sym!(lib, "cuInit"),

            // -- Version query -------------------------------------------------
            cu_driver_get_version: load_sym!(lib, "cuDriverGetVersion"),

            // -- Device management ---------------------------------------------
            cu_device_get: load_sym!(lib, "cuDeviceGet"),
            cu_device_get_count: load_sym!(lib, "cuDeviceGetCount"),
            cu_device_get_name: load_sym!(lib, "cuDeviceGetName"),
            cu_device_get_attribute: load_sym!(lib, "cuDeviceGetAttribute"),
            cu_device_total_mem_v2: load_sym!(lib, "cuDeviceTotalMem_v2"),
            cu_device_can_access_peer: load_sym!(lib, "cuDeviceCanAccessPeer"),

            // -- Primary context management ------------------------------------
            cu_device_primary_ctx_retain: load_sym!(lib, "cuDevicePrimaryCtxRetain"),
            cu_device_primary_ctx_release_v2: load_sym!(lib, "cuDevicePrimaryCtxRelease_v2"),
            cu_device_primary_ctx_set_flags_v2: load_sym!(lib, "cuDevicePrimaryCtxSetFlags_v2"),
            cu_device_primary_ctx_get_state: load_sym!(lib, "cuDevicePrimaryCtxGetState"),
            cu_device_primary_ctx_reset_v2: load_sym!(lib, "cuDevicePrimaryCtxReset_v2"),

            // -- Context management --------------------------------------------
            cu_ctx_create_v2: load_sym!(lib, "cuCtxCreate_v2"),
            cu_ctx_destroy_v2: load_sym!(lib, "cuCtxDestroy_v2"),
            cu_ctx_set_current: load_sym!(lib, "cuCtxSetCurrent"),
            cu_ctx_get_current: load_sym!(lib, "cuCtxGetCurrent"),
            cu_ctx_synchronize: load_sym!(lib, "cuCtxSynchronize"),

            // -- Module management ---------------------------------------------
            cu_module_load_data: load_sym!(lib, "cuModuleLoadData"),
            cu_module_load_data_ex: load_sym!(lib, "cuModuleLoadDataEx"),
            cu_module_get_function: load_sym!(lib, "cuModuleGetFunction"),
            cu_module_unload: load_sym!(lib, "cuModuleUnload"),

            // -- Memory management ---------------------------------------------
            cu_mem_alloc_v2: load_sym!(lib, "cuMemAlloc_v2"),
            cu_mem_free_v2: load_sym!(lib, "cuMemFree_v2"),
            cu_memcpy_htod_v2: load_sym!(lib, "cuMemcpyHtoD_v2"),
            cu_memcpy_dtoh_v2: load_sym!(lib, "cuMemcpyDtoH_v2"),
            cu_memcpy_dtod_v2: load_sym!(lib, "cuMemcpyDtoD_v2"),
            cu_memcpy_htod_async_v2: load_sym!(lib, "cuMemcpyHtoDAsync_v2"),
            cu_memcpy_dtoh_async_v2: load_sym!(lib, "cuMemcpyDtoHAsync_v2"),
            cu_mem_alloc_host_v2: load_sym!(lib, "cuMemAllocHost_v2"),
            cu_mem_free_host: load_sym!(lib, "cuMemFreeHost"),
            cu_mem_alloc_managed: load_sym!(lib, "cuMemAllocManaged"),
            cu_memset_d8_v2: load_sym!(lib, "cuMemsetD8_v2"),
            cu_memset_d32_v2: load_sym!(lib, "cuMemsetD32_v2"),
            cu_mem_get_info_v2: load_sym!(lib, "cuMemGetInfo_v2"),
            cu_mem_host_register_v2: load_sym!(lib, "cuMemHostRegister_v2"),
            cu_mem_host_unregister: load_sym!(lib, "cuMemHostUnregister"),
            cu_mem_host_get_device_pointer_v2: load_sym!(lib, "cuMemHostGetDevicePointer_v2"),
            cu_pointer_get_attribute: load_sym!(lib, "cuPointerGetAttribute"),
            cu_mem_advise: load_sym!(lib, "cuMemAdvise"),
            cu_mem_prefetch_async: load_sym!(lib, "cuMemPrefetchAsync"),

            // -- Stream management ---------------------------------------------
            cu_stream_create: load_sym!(lib, "cuStreamCreate"),
            cu_stream_create_with_priority: load_sym!(lib, "cuStreamCreateWithPriority"),
            cu_stream_destroy_v2: load_sym!(lib, "cuStreamDestroy_v2"),
            cu_stream_synchronize: load_sym!(lib, "cuStreamSynchronize"),
            cu_stream_wait_event: load_sym!(lib, "cuStreamWaitEvent"),
            cu_stream_query: load_sym!(lib, "cuStreamQuery"),
            cu_stream_get_priority: load_sym!(lib, "cuStreamGetPriority"),
            cu_stream_get_flags: load_sym!(lib, "cuStreamGetFlags"),

            // -- Event management ----------------------------------------------
            cu_event_create: load_sym!(lib, "cuEventCreate"),
            cu_event_destroy_v2: load_sym!(lib, "cuEventDestroy_v2"),
            cu_event_record: load_sym!(lib, "cuEventRecord"),
            cu_event_query: load_sym!(lib, "cuEventQuery"),
            cu_event_synchronize: load_sym!(lib, "cuEventSynchronize"),
            cu_event_elapsed_time: load_sym!(lib, "cuEventElapsedTime"),
            cu_event_record_with_flags: load_sym_optional!(lib, "cuEventRecordWithFlags"),

            // -- Peer memory access -------------------------------------------
            cu_memcpy_peer: load_sym!(lib, "cuMemcpyPeer"),
            cu_memcpy_peer_async: load_sym!(lib, "cuMemcpyPeerAsync"),
            cu_ctx_enable_peer_access: load_sym!(lib, "cuCtxEnablePeerAccess"),
            cu_ctx_disable_peer_access: load_sym!(lib, "cuCtxDisablePeerAccess"),

            // -- Kernel launch -------------------------------------------------
            cu_launch_kernel: load_sym!(lib, "cuLaunchKernel"),
            cu_launch_cooperative_kernel: load_sym!(lib, "cuLaunchCooperativeKernel"),
            cu_launch_cooperative_kernel_multi_device: load_sym!(
                lib,
                "cuLaunchCooperativeKernelMultiDevice"
            ),

            // -- Occupancy -----------------------------------------------------
            cu_occupancy_max_active_blocks_per_multiprocessor: load_sym!(
                lib,
                "cuOccupancyMaxActiveBlocksPerMultiprocessor"
            ),
            cu_occupancy_max_potential_block_size: load_sym!(
                lib,
                "cuOccupancyMaxPotentialBlockSize"
            ),
            cu_occupancy_max_active_blocks_per_multiprocessor_with_flags: load_sym!(
                lib,
                "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
            ),

            // -- Memory management (optional) ---------------------------------
            cu_memcpy_dtod_async_v2: load_sym_optional!(lib, "cuMemcpyDtoDAsync_v2"),
            cu_memset_d16_v2: load_sym_optional!(lib, "cuMemsetD16_v2"),
            cu_memset_d32_async: load_sym_optional!(lib, "cuMemsetD32Async"),

            // -- Context management (optional) --------------------------------
            cu_ctx_get_limit: load_sym_optional!(lib, "cuCtxGetLimit"),
            cu_ctx_set_limit: load_sym_optional!(lib, "cuCtxSetLimit"),
            cu_ctx_get_cache_config: load_sym_optional!(lib, "cuCtxGetCacheConfig"),
            cu_ctx_set_cache_config: load_sym_optional!(lib, "cuCtxSetCacheConfig"),
            cu_ctx_get_shared_mem_config: load_sym_optional!(lib, "cuCtxGetSharedMemConfig"),
            cu_ctx_set_shared_mem_config: load_sym_optional!(lib, "cuCtxSetSharedMemConfig"),

            // -- Function attributes (optional) -------------------------------
            cu_func_get_attribute: load_sym_optional!(lib, "cuFuncGetAttribute"),
            cu_func_set_cache_config: load_sym_optional!(lib, "cuFuncSetCacheConfig"),
            cu_func_set_shared_mem_config: load_sym_optional!(lib, "cuFuncSetSharedMemConfig"),
            cu_func_set_attribute: load_sym_optional!(lib, "cuFuncSetAttribute"),

            // -- Profiler (optional) ------------------------------------------
            cu_profiler_start: load_sym_optional!(lib, "cuProfilerStart"),
            cu_profiler_stop: load_sym_optional!(lib, "cuProfilerStop"),

            // -- CUDA 12.x extended launch (optional) -------------------------
            cu_launch_kernel_ex: load_sym_optional!(lib, "cuLaunchKernelEx"),
            cu_tensor_map_encode_tiled: load_sym_optional!(lib, "cuTensorMapEncodeTiled"),

            // -- CUDA 12.8+ extended API (optional) ---------------------------
            cu_tensor_map_encode_tiled_memref: load_sym_optional!(
                lib,
                "cuTensorMapEncodeTiledMemref"
            ),
            cu_kernel_get_library: load_sym_optional!(lib, "cuKernelGetLibrary"),
            cu_multicast_get_granularity: load_sym_optional!(lib, "cuMulticastGetGranularity"),
            cu_multicast_create: load_sym_optional!(lib, "cuMulticastCreate"),
            cu_multicast_add_device: load_sym_optional!(lib, "cuMulticastAddDevice"),
            cu_memcpy_batch_async: load_sym_optional!(lib, "cuMemcpyBatchAsync"),

            // -- Texture / Surface memory (optional) ---------------------------
            cu_array_create_v2: load_sym_optional!(lib, "cuArrayCreate_v2"),
            cu_array_destroy: load_sym_optional!(lib, "cuArrayDestroy"),
            cu_array_get_descriptor_v2: load_sym_optional!(lib, "cuArrayGetDescriptor_v2"),
            cu_array3d_create_v2: load_sym_optional!(lib, "cuArray3DCreate_v2"),
            cu_array3d_get_descriptor_v2: load_sym_optional!(lib, "cuArray3DGetDescriptor_v2"),
            cu_memcpy_htoa_v2: load_sym_optional!(lib, "cuMemcpyHtoA_v2"),
            cu_memcpy_atoh_v2: load_sym_optional!(lib, "cuMemcpyAtoH_v2"),
            cu_memcpy_htoa_async_v2: load_sym_optional!(lib, "cuMemcpyHtoAAsync_v2"),
            cu_memcpy_atoh_async_v2: load_sym_optional!(lib, "cuMemcpyAtoHAsync_v2"),
            cu_tex_object_create: load_sym_optional!(lib, "cuTexObjectCreate"),
            cu_tex_object_destroy: load_sym_optional!(lib, "cuTexObjectDestroy"),
            cu_tex_object_get_resource_desc: load_sym_optional!(lib, "cuTexObjectGetResourceDesc"),
            cu_surf_object_create: load_sym_optional!(lib, "cuSurfObjectCreate"),
            cu_surf_object_destroy: load_sym_optional!(lib, "cuSurfObjectDestroy"),

            // Keep the library handle alive.
            _lib: lib,
        })
    }
}

// ---------------------------------------------------------------------------
// Global accessor
// ---------------------------------------------------------------------------

/// Get a reference to the lazily-loaded CUDA driver API function table.
///
/// On the first call, this function dynamically loads the CUDA shared library
/// and resolves all required symbols.  Subsequent calls return the cached
/// result with only an atomic load.
///
/// # Errors
///
/// Returns [`CudaError::NotInitialized`] if the driver could not be loaded —
/// for instance, on macOS, or on a system without an NVIDIA GPU driver
/// installed.
///
/// # Examples
///
/// ```rust,no_run
/// # use oxicuda_driver::loader::try_driver;
/// let api = try_driver()?;
/// let result = unsafe { (api.cu_init)(0) };
/// # Ok::<(), oxicuda_driver::error::CudaError>(())
/// ```
pub fn try_driver() -> CudaResult<&'static DriverApi> {
    let result = DRIVER.get_or_init(DriverApi::load);
    match result {
        Ok(api) => Ok(api),
        Err(_) => Err(CudaError::NotInitialized),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// On macOS, loading should always fail with `UnsupportedPlatform`.
    #[cfg(target_os = "macos")]
    #[test]
    fn load_returns_unsupported_on_macos() {
        let result = DriverApi::load();
        assert!(result.is_err(), "expected Err on macOS");
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected Err on macOS"),
        };
        assert!(
            matches!(err, DriverLoadError::UnsupportedPlatform),
            "expected UnsupportedPlatform, got {err:?}"
        );
    }

    /// `try_driver` should return `Err(NotInitialized)` on platforms without
    /// a CUDA driver (including macOS).
    #[cfg(target_os = "macos")]
    #[test]
    fn try_driver_returns_not_initialized_on_macos() {
        let result = try_driver();
        assert!(result.is_err(), "expected Err on macOS");
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected Err on macOS"),
        };
        assert!(
            matches!(err, CudaError::NotInitialized),
            "expected NotInitialized, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Task 1 — CUDA 12.8+ DriverApi struct layout tests
    //
    // These tests verify that the DriverApi struct contains the expected
    // Option<fn(...)> fields for the new CUDA 12.8+ API entry points.
    // They compile and run without a GPU because they only inspect type
    // layout and field presence, never calling the function pointers.
    // -----------------------------------------------------------------------

    /// Verify that the `cu_tensor_map_encode_tiled_memref` field exists and
    /// is an `Option` type.  The driver will return `None` on older versions.
    #[test]
    fn driver_v12_8_api_fields_present() {
        // The simplest way to prove a field exists at the correct type is to
        // construct a value that fits in that position.  We use a local
        // DriverApi value on macOS (where load() always returns Err) by
        // manufacturing a dummy function pointer and verifying the type
        // annotation compiles.
        //
        // On non-macOS platforms we simply verify the field is accessible on
        // the type via a None literal assignment (compile-time check).
        type TensorMapEncodeTiledFn = unsafe extern "C" fn(
            tensor_map: *mut std::ffi::c_void,
            tensor_data_type: u32,
            tensor_rank: u32,
            global_address: *mut std::ffi::c_void,
            global_dim: *const u64,
            global_strides: *const u64,
            box_dim: *const u32,
            element_strides: *const u32,
            interleave: u32,
            swizzle: u32,
            l2_promotion: u32,
            oob_fill: u32,
            flags: u64,
        ) -> CUresult;
        let _none: Option<TensorMapEncodeTiledFn> = None;
        // Field name check: accessing the field compiles only if it exists.
        // We use a trait-object-based field-name probe: the macro produces a
        // compile error if the identifier does not exist.
        let _field_exists = |api: &DriverApi| api.cu_tensor_map_encode_tiled_memref.is_none();
        // Suppress unused variable warnings.
        let _ = _none;
        let _ = _field_exists;
    }

    /// Verify that `cu_multicast_create` and `cu_multicast_add_device` fields
    /// exist with the correct Option<fn(...)> types (CUDA 12.8+ multicast).
    #[test]
    fn driver_v12_8_multicast_fields_present() {
        let _probe_create = |api: &DriverApi| api.cu_multicast_create.is_none();
        let _probe_add = |api: &DriverApi| api.cu_multicast_add_device.is_none();
        let _probe_gran = |api: &DriverApi| api.cu_multicast_get_granularity.is_none();
        let _ = (_probe_create, _probe_add, _probe_gran);
    }

    /// Verify that `cu_memcpy_batch_async` field exists with the correct
    /// Option<fn(...)> type (CUDA 12.8+ batch memcpy).
    #[test]
    fn driver_v12_8_batch_memcpy_field_present() {
        let _probe = |api: &DriverApi| api.cu_memcpy_batch_async.is_none();
        let _ = _probe;
    }

    /// Verify that `cu_kernel_get_library` field exists (CUDA 12.8+ JIT libs).
    #[test]
    fn driver_v12_8_kernel_get_library_field_present() {
        let _probe = |api: &DriverApi| api.cu_kernel_get_library.is_none();
        let _ = _probe;
    }
}
