//! Raw CUDA Driver API FFI types, constants, and enums.
//!
//! This module provides the low-level type definitions that mirror the CUDA Driver API
//! (`cuda.h`). No functions are defined here — only types, opaque pointer aliases,
//! result-code constants, and `#[repr]` enums used by the dynamically loaded driver
//! entry points.
//!
//! # Safety
//!
//! All pointer types in this module are raw pointers intended for FFI use.
//! They must only be used through the safe wrappers provided by higher-level
//! modules in `oxicuda-driver`.

use std::ffi::c_void;
use std::fmt;

// ---------------------------------------------------------------------------
// Core scalar type aliases
// ---------------------------------------------------------------------------

/// Return code from every CUDA Driver API call.
///
/// A value of `0` (`CUDA_SUCCESS`) indicates success; any other value is an
/// error code. See the `CUDA_*` constants below for the full catalogue.
pub type CUresult = u32;

/// Ordinal identifier for a CUDA-capable device (0-based).
pub type CUdevice = i32;

/// Device-side pointer (64-bit address in GPU virtual memory).
pub type CUdeviceptr = u64;

// ---------------------------------------------------------------------------
// Opaque handle helpers
// ---------------------------------------------------------------------------

macro_rules! define_handle {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(pub *mut c_void);

        // SAFETY: CUDA handles are thread-safe when used with proper
        // synchronisation via the driver API.
        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({:p})", stringify!($name), self.0)
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self(std::ptr::null_mut())
            }
        }

        impl $name {
            /// Returns `true` if the handle is null (uninitialised).
            #[inline]
            pub fn is_null(self) -> bool {
                self.0.is_null()
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Handle types
// ---------------------------------------------------------------------------

define_handle! {
    /// Opaque handle to a CUDA context.
    CUcontext
}

define_handle! {
    /// Opaque handle to a loaded CUDA module (PTX / cubin).
    CUmodule
}

define_handle! {
    /// Opaque handle to a CUDA kernel function within a module.
    CUfunction
}

define_handle! {
    /// Opaque handle to a CUDA stream (command queue).
    CUstream
}

define_handle! {
    /// Opaque handle to a CUDA event (used for timing and synchronisation).
    CUevent
}

define_handle! {
    /// Opaque handle to a CUDA memory pool (`cuMemPool*` family).
    CUmemoryPool
}

define_handle! {
    /// Opaque handle to a CUDA texture reference (legacy API).
    CUtexref
}

define_handle! {
    /// Opaque handle to a CUDA surface reference (legacy API).
    CUsurfref
}

define_handle! {
    /// Opaque handle to a CUDA texture object (modern bindless API).
    CUtexObject
}

define_handle! {
    /// Opaque handle to a CUDA surface object (modern bindless API).
    CUsurfObject
}

define_handle! {
    /// Opaque handle to a CUDA kernel (CUDA 12.8+ library-based kernels).
    ///
    /// Used with `cuKernelGetLibrary` to retrieve the library a kernel
    /// belongs to.
    CUkernel
}

define_handle! {
    /// Opaque handle to a CUDA library (CUDA 12.8+ JIT library API).
    ///
    /// Retrieved via `cuKernelGetLibrary` to identify the JIT-compiled
    /// library that contains a given kernel.
    CUlibrary
}

define_handle! {
    /// Opaque handle to an NVLink multicast object (CUDA 12.8+).
    ///
    /// Used with `cuMulticastCreate`, `cuMulticastAddDevice`, and related
    /// functions to manage NVLink multicast memory regions across devices.
    CUmulticastObject
}

// =========================================================================
// CUmemorytype — memory type identifiers
// =========================================================================

/// Memory type identifiers returned by pointer attribute queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
pub enum CUmemorytype {
    /// Host (system) memory.
    Host = 1,
    /// Device (GPU) memory.
    Device = 2,
    /// Array memory.
    Array = 3,
    /// Unified (managed) memory.
    Unified = 4,
}

// =========================================================================
// CUpointer_attribute — pointer attribute query keys
// =========================================================================

/// Pointer attribute identifiers passed to `cuPointerGetAttribute`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum CUpointer_attribute {
    /// Query the CUDA context associated with a pointer.
    Context = 1,
    /// Query the memory type (host / device / unified) of a pointer.
    MemoryType = 2,
    /// Query the device pointer corresponding to a host pointer.
    DevicePointer = 3,
    /// Query the host pointer corresponding to a device pointer.
    HostPointer = 4,
    /// Query whether the memory is managed (unified).
    IsManaged = 9,
    /// Query the device ordinal for the pointer.
    DeviceOrdinal = 10,
}

// =========================================================================
// CUlimit — context limit identifiers
// =========================================================================

/// Context limit identifiers for `cuCtxSetLimit` / `cuCtxGetLimit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
pub enum CUlimit {
    /// Stack size for each GPU thread.
    StackSize = 0,
    /// Size of the printf FIFO.
    PrintfFifoSize = 1,
    /// Size of the heap used by `malloc()` on the device.
    MallocHeapSize = 2,
    /// Maximum nesting depth of a device runtime launch.
    DevRuntimeSyncDepth = 3,
    /// Maximum number of outstanding device runtime launches.
    DevRuntimePendingLaunchCount = 4,
    /// L2 cache fetch granularity.
    MaxL2FetchGranularity = 5,
    /// Maximum persisting L2 cache size.
    PersistingL2CacheSize = 6,
}

// =========================================================================
// CUfunction_attribute — function attribute query keys
// =========================================================================

/// Function attribute identifiers passed to `cuFuncGetAttribute`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum CUfunction_attribute {
    /// Maximum threads per block for this function.
    MaxThreadsPerBlock = 0,
    /// Shared memory used by this function (bytes).
    SharedSizeBytes = 1,
    /// Size of user-allocated constant memory (bytes).
    ConstSizeBytes = 2,
    /// Size of local memory used by each thread (bytes).
    LocalSizeBytes = 3,
    /// Number of registers used by each thread.
    NumRegs = 4,
    /// PTX virtual architecture version.
    PtxVersion = 5,
    /// Binary architecture version.
    BinaryVersion = 6,
    /// Whether this function has been cached.
    CacheModeCa = 7,
    /// Maximum dynamic shared memory size (bytes).
    MaxDynamicSharedSizeBytes = 8,
    /// Preferred shared memory carve-out.
    PreferredSharedMemoryCarveout = 9,
}

// =========================================================================
// CUresult constants — every documented CUDA Driver API error code
// =========================================================================

/// The API call returned with no errors.
pub const CUDA_SUCCESS: CUresult = 0;

/// One or more parameters passed to the API call are not acceptable.
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;

/// The API call failed because it was unable to allocate enough memory.
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;

/// The CUDA driver has not been initialised via `cuInit`.
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;

/// The CUDA driver is shutting down.
pub const CUDA_ERROR_DEINITIALIZED: CUresult = 4;

/// Profiler is not initialised for this run.
pub const CUDA_ERROR_PROFILER_DISABLED: CUresult = 5;

/// (Deprecated) Profiler not started.
pub const CUDA_ERROR_PROFILER_NOT_INITIALIZED: CUresult = 6;

/// (Deprecated) Profiler already started.
pub const CUDA_ERROR_PROFILER_ALREADY_STARTED: CUresult = 7;

/// (Deprecated) Profiler already stopped.
pub const CUDA_ERROR_PROFILER_ALREADY_STOPPED: CUresult = 8;

/// Stub library loaded instead of the real driver.
pub const CUDA_ERROR_STUB_LIBRARY: CUresult = 34;

/// Device-side assert triggered.
pub const CUDA_ERROR_DEVICE_UNAVAILABLE: CUresult = 46;

/// No CUDA-capable device is detected.
pub const CUDA_ERROR_NO_DEVICE: CUresult = 100;

/// The device ordinal supplied is out of range.
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;

/// The device does not have a valid licence.
pub const CUDA_ERROR_DEVICE_NOT_LICENSED: CUresult = 102;

/// The PTX or cubin image is invalid.
pub const CUDA_ERROR_INVALID_IMAGE: CUresult = 200;

/// The supplied context is not valid.
pub const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;

/// (Deprecated) Context already current.
pub const CUDA_ERROR_CONTEXT_ALREADY_CURRENT: CUresult = 202;

/// A map or register operation has failed.
pub const CUDA_ERROR_MAP_FAILED: CUresult = 205;

/// An unmap or unregister operation has failed.
pub const CUDA_ERROR_UNMAP_FAILED: CUresult = 206;

/// The specified array is currently mapped.
pub const CUDA_ERROR_ARRAY_IS_MAPPED: CUresult = 207;

/// The resource is already mapped.
pub const CUDA_ERROR_ALREADY_MAPPED: CUresult = 208;

/// There is no kernel image available for execution on the device.
pub const CUDA_ERROR_NO_BINARY_FOR_GPU: CUresult = 209;

/// A resource has already been acquired.
pub const CUDA_ERROR_ALREADY_ACQUIRED: CUresult = 210;

/// The resource is not mapped.
pub const CUDA_ERROR_NOT_MAPPED: CUresult = 211;

/// A mapped resource is not available for access as an array.
pub const CUDA_ERROR_NOT_MAPPED_AS_ARRAY: CUresult = 212;

/// A mapped resource is not available for access as a pointer.
pub const CUDA_ERROR_NOT_MAPPED_AS_POINTER: CUresult = 213;

/// An uncorrectable ECC error was detected.
pub const CUDA_ERROR_ECC_UNCORRECTABLE: CUresult = 214;

/// A PTX JIT limit has been reached.
pub const CUDA_ERROR_UNSUPPORTED_LIMIT: CUresult = 215;

/// The context already has work from another thread bound to it.
pub const CUDA_ERROR_CONTEXT_ALREADY_IN_USE: CUresult = 216;

/// Peer access is not supported across the given devices.
pub const CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: CUresult = 217;

/// The PTX JIT compilation was disabled or the PTX is invalid.
pub const CUDA_ERROR_INVALID_PTX: CUresult = 218;

/// Invalid graphics context.
pub const CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: CUresult = 219;

/// NVLINK is uncorrectable.
pub const CUDA_ERROR_NVLINK_UNCORRECTABLE: CUresult = 220;

/// JIT compiler not found.
pub const CUDA_ERROR_JIT_COMPILER_NOT_FOUND: CUresult = 221;

/// Unsupported PTX version.
pub const CUDA_ERROR_UNSUPPORTED_PTX_VERSION: CUresult = 222;

/// JIT compilation disabled.
pub const CUDA_ERROR_JIT_COMPILATION_DISABLED: CUresult = 223;

/// Unsupported exec-affinity type.
pub const CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY: CUresult = 224;

/// Unsupported device-side synchronisation on this device.
pub const CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC: CUresult = 225;

/// The requested source is invalid.
pub const CUDA_ERROR_INVALID_SOURCE: CUresult = 300;

/// The named file was not found.
pub const CUDA_ERROR_FILE_NOT_FOUND: CUresult = 301;

/// A shared-object symbol lookup failed.
pub const CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: CUresult = 302;

/// The shared-object init function failed.
pub const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: CUresult = 303;

/// An OS call failed.
pub const CUDA_ERROR_OPERATING_SYSTEM: CUresult = 304;

/// The supplied handle is invalid.
pub const CUDA_ERROR_INVALID_HANDLE: CUresult = 400;

/// The requested resource is in an illegal state.
pub const CUDA_ERROR_ILLEGAL_STATE: CUresult = 401;

/// A loss-less compression buffer was detected while doing uncompressed access.
pub const CUDA_ERROR_LOSSY_QUERY: CUresult = 402;

/// A named symbol was not found.
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;

/// The operation is not ready (asynchronous).
pub const CUDA_ERROR_NOT_READY: CUresult = 600;

/// An illegal memory address was encountered.
pub const CUDA_ERROR_ILLEGAL_ADDRESS: CUresult = 700;

/// The kernel launch uses too many resources (registers / shared memory).
pub const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: CUresult = 701;

/// The kernel launch exceeded the time-out enforced by the driver.
pub const CUDA_ERROR_LAUNCH_TIMEOUT: CUresult = 702;

/// A launch did not occur on a compatible texturing mode.
pub const CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: CUresult = 703;

/// Peer access already enabled.
pub const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: CUresult = 704;

/// Peer access has not been enabled.
pub const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: CUresult = 705;

/// The primary context has already been initialised.
pub const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: CUresult = 708;

/// The context is being destroyed.
pub const CUDA_ERROR_CONTEXT_IS_DESTROYED: CUresult = 709;

/// A 64-bit device assertion triggered.
pub const CUDA_ERROR_ASSERT: CUresult = 710;

/// Hardware resources to enable peer access are exhausted.
pub const CUDA_ERROR_TOO_MANY_PEERS: CUresult = 711;

/// The host-side memory region is already registered.
pub const CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: CUresult = 712;

/// The host-side memory region is not registered.
pub const CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: CUresult = 713;

/// Hardware stack overflow on the device.
pub const CUDA_ERROR_HARDWARE_STACK_ERROR: CUresult = 714;

/// Illegal instruction encountered on the device.
pub const CUDA_ERROR_ILLEGAL_INSTRUCTION: CUresult = 715;

/// Misaligned address on the device.
pub const CUDA_ERROR_MISALIGNED_ADDRESS: CUresult = 716;

/// Invalid address space.
pub const CUDA_ERROR_INVALID_ADDRESS_SPACE: CUresult = 717;

/// Invalid program counter on the device.
pub const CUDA_ERROR_INVALID_PC: CUresult = 718;

/// The kernel launch failed.
pub const CUDA_ERROR_LAUNCH_FAILED: CUresult = 719;

/// Cooperative launch is too large for the device/kernel.
pub const CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: CUresult = 720;

/// The API call is not permitted in the active context.
pub const CUDA_ERROR_NOT_PERMITTED: CUresult = 800;

/// The API call is not supported by the current driver/device combination.
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;

/// System not ready for CUDA operations.
pub const CUDA_ERROR_SYSTEM_NOT_READY: CUresult = 802;

/// System driver mismatch.
pub const CUDA_ERROR_SYSTEM_DRIVER_MISMATCH: CUresult = 803;

/// Old-style context incompatible with CUDA 3.2+ API.
pub const CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: CUresult = 804;

/// MPS connection failed.
pub const CUDA_ERROR_MPS_CONNECTION_FAILED: CUresult = 805;

/// MPS RPC failure.
pub const CUDA_ERROR_MPS_RPC_FAILURE: CUresult = 806;

/// MPS server not ready.
pub const CUDA_ERROR_MPS_SERVER_NOT_READY: CUresult = 807;

/// MPS maximum clients reached.
pub const CUDA_ERROR_MPS_MAX_CLIENTS_REACHED: CUresult = 808;

/// MPS maximum connections reached.
pub const CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED: CUresult = 809;

/// MPS client terminated.
pub const CUDA_ERROR_MPS_CLIENT_TERMINATED: CUresult = 810;

/// CDP not supported.
pub const CUDA_ERROR_CDP_NOT_SUPPORTED: CUresult = 811;

/// CDP version mismatch.
pub const CUDA_ERROR_CDP_VERSION_MISMATCH: CUresult = 812;

/// Stream capture unsupported.
pub const CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: CUresult = 900;

/// Stream capture invalidated.
pub const CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: CUresult = 901;

/// Stream capture merge not permitted.
pub const CUDA_ERROR_STREAM_CAPTURE_MERGE: CUresult = 902;

/// Stream capture unmatched.
pub const CUDA_ERROR_STREAM_CAPTURE_UNMATCHED: CUresult = 903;

/// Stream capture unjoined.
pub const CUDA_ERROR_STREAM_CAPTURE_UNJOINED: CUresult = 904;

/// Stream capture isolation violation.
pub const CUDA_ERROR_STREAM_CAPTURE_ISOLATION: CUresult = 905;

/// Implicit stream in graph capture.
pub const CUDA_ERROR_STREAM_CAPTURE_IMPLICIT: CUresult = 906;

/// Captured event error.
pub const CUDA_ERROR_CAPTURED_EVENT: CUresult = 907;

/// Stream capture wrong thread.
pub const CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD: CUresult = 908;

/// The async operation timed out.
pub const CUDA_ERROR_TIMEOUT: CUresult = 909;

/// The graph update failed.
pub const CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE: CUresult = 910;

/// External device error.
pub const CUDA_ERROR_EXTERNAL_DEVICE: CUresult = 911;

/// Invalid cluster size.
pub const CUDA_ERROR_INVALID_CLUSTER_SIZE: CUresult = 912;

/// Function not loaded.
pub const CUDA_ERROR_FUNCTION_NOT_LOADED: CUresult = 913;

/// Invalid resource type.
pub const CUDA_ERROR_INVALID_RESOURCE_TYPE: CUresult = 914;

/// Invalid resource configuration.
pub const CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION: CUresult = 915;

/// An unknown internal error occurred.
pub const CUDA_ERROR_UNKNOWN: CUresult = 999;

// =========================================================================
// CUdevice_attribute — device property query keys
// =========================================================================

/// Device attribute identifiers passed to `cuDeviceGetAttribute`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum CUdevice_attribute {
    /// Maximum number of threads per block.
    MaxThreadsPerBlock = 1,
    /// Maximum x-dimension of a block.
    MaxBlockDimX = 2,
    /// Maximum y-dimension of a block.
    MaxBlockDimY = 3,
    /// Maximum z-dimension of a block.
    MaxBlockDimZ = 4,
    /// Maximum x-dimension of a grid.
    MaxGridDimX = 5,
    /// Maximum y-dimension of a grid.
    MaxGridDimY = 6,
    /// Maximum z-dimension of a grid.
    MaxGridDimZ = 7,
    /// Maximum shared memory available per block (bytes).
    MaxSharedMemoryPerBlock = 8,
    /// Total amount of constant memory on the device (bytes).
    TotalConstantMemory = 9,
    /// Warp size in threads.
    WarpSize = 10,
    /// Maximum pitch allowed by memory copies (bytes).
    MaxPitch = 11,
    /// Maximum number of 32-bit registers per block.
    MaxRegistersPerBlock = 12,
    /// Peak clock frequency in kHz.
    ClockRate = 13,
    /// Alignment requirement for textures.
    TextureAlignment = 14,
    /// Device can possibly copy memory and execute a kernel concurrently.
    GpuOverlap = 15,
    /// Number of multiprocessors on the device.
    MultiprocessorCount = 16,
    /// Whether there is a run-time limit on kernels.
    KernelExecTimeout = 17,
    /// Device is integrated (shares host memory).
    Integrated = 18,
    /// Device can map host memory with `cuMemHostAlloc` / `cuMemHostRegister`.
    CanMapHostMemory = 19,
    /// Compute mode: default, exclusive, prohibited, etc.
    ComputeMode = 20,
    /// Maximum 1D texture width.
    MaxTexture1DWidth = 21,
    /// Maximum 2D texture width.
    MaxTexture2DWidth = 22,
    /// Maximum 2D texture height.
    MaxTexture2DHeight = 23,
    /// Maximum 3D texture width.
    MaxTexture3DWidth = 24,
    /// Maximum 3D texture height.
    MaxTexture3DHeight = 25,
    /// Maximum 3D texture depth.
    MaxTexture3DDepth = 26,
    /// Maximum 2D layered texture width.
    MaxTexture2DLayeredWidth = 27,
    /// Maximum 2D layered texture height.
    MaxTexture2DLayeredHeight = 28,
    /// Maximum layers in a 2D layered texture.
    MaxTexture2DLayeredLayers = 29,
    /// Alignment requirement for surfaces.
    SurfaceAlignment = 30,
    /// Device can execute multiple kernels concurrently.
    ConcurrentKernels = 31,
    /// Device supports ECC memory.
    EccEnabled = 32,
    /// PCI bus ID of the device.
    PciBusId = 33,
    /// PCI device ID of the device.
    PciDeviceId = 34,
    /// Device is using TCC (Tesla Compute Cluster) driver model.
    TccDriver = 35,
    /// Peak memory clock frequency in kHz.
    MemoryClockRate = 36,
    /// Global memory bus width in bits.
    GlobalMemoryBusWidth = 37,
    /// Size of L2 cache in bytes.
    L2CacheSize = 38,
    /// Maximum resident threads per multiprocessor.
    MaxThreadsPerMultiprocessor = 39,
    /// Number of asynchronous engines.
    AsyncEngineCount = 40,
    /// Device shares a unified address space with the host.
    UnifiedAddressing = 41,
    /// Maximum 1D layered texture width.
    MaxTexture1DLayeredWidth = 42,
    /// Maximum layers in a 1D layered texture.
    MaxTexture1DLayeredLayers = 43,
    /// Maximum 2D texture width if CUDA 2D memory allocation is bound.
    MaxTexture2DGatherWidth = 44,
    /// Maximum 2D texture height if CUDA 2D memory allocation is bound.
    MaxTexture2DGatherHeight = 45,
    /// Alternate maximum 3D texture width.
    MaxTexture3DWidthAlt = 47,
    /// Alternate maximum 3D texture height.
    MaxTexture3DHeightAlt = 48,
    /// Alternate maximum 3D texture depth.
    MaxTexture3DDepthAlt = 49,
    /// PCI domain ID.
    PciDomainId = 50,
    /// Texture pitch alignment.
    TexturePitchAlignment = 51,
    /// Maximum 1D mipmapped texture width.
    MaxTexture1DMipmappedWidth2 = 52,
    /// Maximum width for a cubemap texture.
    MaxTextureCubemapWidth = 54,
    /// Maximum width for a cubemap layered texture.
    MaxTextureCubemapLayeredWidth = 55,
    /// Maximum layers in a cubemap layered texture.
    MaxTextureCubemapLayeredLayers = 56,
    /// Maximum 1D surface width.
    MaxSurface1DWidth = 57,
    /// Maximum 2D surface width.
    MaxSurface2DWidth = 58,
    /// Maximum 2D surface height.
    MaxSurface2DHeight = 59,
    /// Maximum 3D surface width.
    MaxSurface3DWidth = 60,
    /// Maximum 3D surface height.
    MaxSurface3DHeight = 61,
    /// Maximum 3D surface depth.
    MaxSurface3DDepth = 62,
    /// Maximum cubemap surface width.
    MaxSurfaceCubemapWidth = 63,
    /// Maximum 1D layered surface width.
    MaxSurface1DLayeredWidth = 64,
    /// Maximum layers in a 1D layered surface.
    MaxSurface1DLayeredLayers = 65,
    /// Maximum 2D layered surface width.
    MaxSurface2DLayeredWidth = 66,
    /// Maximum 2D layered surface height.
    MaxSurface2DLayeredHeight = 67,
    /// Maximum layers in a 2D layered surface.
    MaxSurface2DLayeredLayers = 68,
    /// Maximum cubemap layered surface width.
    MaxSurfaceCubemapLayeredWidth = 69,
    /// Maximum layers in a cubemap layered surface.
    MaxSurfaceCubemapLayeredLayers = 70,
    /// Maximum 1D linear texture width (deprecated).
    MaxTexture1DLinearWidth = 71,
    /// Maximum 2D linear texture width.
    MaxTexture2DLinearWidth = 72,
    /// Maximum 2D linear texture height.
    MaxTexture2DLinearHeight = 73,
    /// Maximum 2D linear texture pitch (bytes).
    MaxTexture2DLinearPitch = 74,
    /// Major compute capability version number.
    ComputeCapabilityMajor = 75,
    /// Minor compute capability version number.
    ComputeCapabilityMinor = 76,
    /// Maximum mipmapped 2D texture width.
    MaxTexture2DMipmappedWidth = 77,
    /// Maximum mipmapped 2D texture height.
    MaxTexture2DMipmappedHeight = 78,
    /// Maximum mipmapped 1D texture width.
    MaxTexture1DMipmappedWidth = 79,
    /// Device supports stream priorities.
    StreamPrioritiesSupported = 80,
    /// Maximum shared memory per multiprocessor (bytes).
    MaxSharedMemoryPerMultiprocessor = 81,
    /// Maximum registers per multiprocessor.
    MaxRegistersPerMultiprocessor = 82,
    /// Device supports managed memory.
    ManagedMemory = 83,
    /// Device is on a multi-GPU board.
    IsMultiGpuBoard = 84,
    /// Unique identifier for the multi-GPU board group.
    MultiGpuBoardGroupId = 85,
    /// Host-visible native-atomic support for float operations.
    HostNativeAtomicSupported = 86,
    /// Ratio of single-to-double precision performance.
    SingleToDoublePrecisionPerfRatio = 87,
    /// Device supports pageable memory access.
    PageableMemoryAccess = 88,
    /// Device can access host registered memory at the same virtual address.
    ConcurrentManagedAccess = 89,
    /// Device supports compute preemption.
    ComputePreemptionSupported = 90,
    /// Device can access host memory via pageable accesses.
    CanUseHostPointerForRegisteredMem = 91,
    /// Reserved attribute (CUDA internal, value 92).
    Reserved92 = 92,
    /// Reserved attribute (CUDA internal, value 93).
    Reserved93 = 93,
    /// Reserved attribute (CUDA internal, value 94).
    Reserved94 = 94,
    /// Device supports cooperative kernel launches.
    CooperativeLaunch = 95,
    /// Device supports cooperative kernel launches across multiple GPUs.
    CooperativeMultiDeviceLaunch = 96,
    /// Maximum optin shared memory per block.
    MaxSharedMemoryPerBlockOptin = 97,
    /// Device supports flushing of outstanding remote writes.
    CanFlushRemoteWrites = 98,
    /// Device supports host-side memory-register functions.
    HostRegisterSupported = 99,
    /// Device supports pageable memory access using host page tables.
    PageableMemoryAccessUsesHostPageTables = 100,
    /// Device supports direct access to managed memory on the host.
    DirectManagedMemAccessFromHost = 101,
    /// Device supports virtual memory management APIs.
    VirtualMemoryManagementSupported = 102,
    /// Device supports handle-type POSIX file descriptors for IPC.
    HandleTypePosixFileDescriptorSupported = 103,
    /// Device supports handle-type Win32 handles for IPC.
    HandleTypeWin32HandleSupported = 104,
    /// Device supports handle-type Win32 KMT handles for IPC.
    HandleTypeWin32KmtHandleSupported = 105,
    /// Maximum blocks per multiprocessor.
    MaxBlocksPerMultiprocessor = 106,
    /// Device supports generic compression for memory.
    GenericCompressionSupported = 107,
    /// Maximum persisting L2 cache size (bytes).
    MaxPersistingL2CacheSize = 108,
    /// Maximum access-policy window size for L2 cache.
    MaxAccessPolicyWindowSize = 109,
    /// Device supports RDMA APIs via `cuMemRangeGetAttribute`.
    GpuDirectRdmaWithCudaVmmSupported = 110,
    /// Free memory / total memory on the device accessible via `cuMemGetInfo`.
    AccessPolicyMaxWindowSize = 111,
    /// Reserved range of shared memory per SM (bytes).
    ReservedSharedMemoryPerBlock = 112,
    /// Device supports timeline semaphore interop.
    TimelineSemaphoreInteropSupported = 113,
    /// Device supports memory pools (`cudaMallocAsync`).
    MemoryPoolsSupported = 115,
    /// GPU direct RDMA is supported.
    GpuDirectRdmaSupported = 116,
    /// GPU direct RDMA flush-writes order.
    GpuDirectRdmaFlushWritesOptions = 117,
    /// GPU direct RDMA writes ordering.
    GpuDirectRdmaWritesOrdering = 118,
    /// Memory pool supported handle types.
    MemoryPoolSupportedHandleTypes = 119,
    /// Device supports cluster launch.
    ClusterLaunch = 120,
    /// Deferred mapping CUDA array supported.
    DeferredMappingCudaArraySupported = 121,
    /// Device supports IPC event handles.
    IpcEventSupported = 122,
    /// Device supports mem-sync domain count.
    MemSyncDomainCount = 123,
    /// Device supports tensor-map access to data.
    TensorMapAccessSupported = 124,
    /// Unified function pointers supported.
    UnifiedFunctionPointers = 125,
    /// NUMA config.
    NumaConfig = 127,
    /// NUMA id.
    NumaId = 128,
    /// Multicast supported.
    /// Device supports getting the minimum required per-block shared memory
    /// for a cooperative launch via the extended attributes.
    MaxTimelineSemaphoreInteropSupported = 129,
    /// Device supports memory sync domain operations.
    MemSyncDomainSupported = 130,
    /// Device supports GPU-Direct Fabric.
    GpuDirectRdmaFabricSupported = 131,
    /// Device supports multicast.
    MulticastSupported = 132,
    /// Device supports MPS features.
    MpsEnabled = 133,
    /// Host-NUMA identifier.
    HostNumaId = 134,
}

// =========================================================================
// CUjit_option — options for the JIT compiler
// =========================================================================

/// JIT compilation options passed to `cuModuleLoadDataEx` and related functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum CUjit_option {
    /// Maximum number of registers that a thread may use.
    MaxRegisters = 0,
    /// Number of threads per block for the JIT target.
    ThreadsPerBlock = 1,
    /// Wall-clock time (ms) for compilation.
    WallTime = 2,
    /// Pointer to a buffer for info log output.
    InfoLogBuffer = 3,
    /// Size (bytes) of the info-log buffer.
    InfoLogBufferSizeBytes = 4,
    /// Pointer to a buffer for error log output.
    ErrorLogBuffer = 5,
    /// Size (bytes) of the error-log buffer.
    ErrorLogBufferSizeBytes = 6,
    /// Optimisation level (0-4).
    OptimizationLevel = 7,
    /// Determines the target based on the current attached context.
    TargetFromCuContext = 8,
    /// Specific compute target (sm_XX).
    Target = 9,
    /// Fallback strategy when exact match is not found.
    FallbackStrategy = 10,
    /// Specifies whether to generate debug info.
    GenerateDebugInfo = 11,
    /// Generate verbose log messages.
    LogVerbose = 12,
    /// Generate line-number information.
    GenerateLineInfo = 13,
    /// Cache mode (on / off).
    CacheMode = 14,
    /// (Internal) New SM3X option.
    Sm3xOpt = 15,
    /// Fast compile flag.
    FastCompile = 16,
    /// Global symbol names.
    GlobalSymbolNames = 17,
    /// Global symbol addresses.
    GlobalSymbolAddresses = 18,
    /// Number of global symbols.
    GlobalSymbolCount = 19,
    /// LTO flag.
    Lto = 20,
    /// FTZ (flush-to-zero) flag.
    Ftz = 21,
    /// Prec-div flag.
    PrecDiv = 22,
    /// Prec-sqrt flag.
    PrecSqrt = 23,
    /// FMA flag.
    Fma = 24,
    /// Referenced kernel names.
    ReferencedKernelNames = 25,
    /// Referenced kernel count.
    ReferencedKernelCount = 26,
    /// Referenced variable names.
    ReferencedVariableNames = 27,
    /// Referenced variable count.
    ReferencedVariableCount = 28,
    /// Optimise unused device variables.
    OptimizeUnusedDeviceVariables = 29,
    /// Position-independent code.
    PositionIndependentCode = 30,
}

// =========================================================================
// CUjitInputType — input types for the linker
// =========================================================================

/// Input types for `cuLinkAddData` / `cuLinkAddFile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
pub enum CUjitInputType {
    /// PTX source code.
    Ptx = 1,
    /// Compiled device code (cubin).
    Cubin = 2,
    /// Fat binary bundle.
    Fatbin = 3,
    /// Relocatable device object.
    Object = 4,
    /// Device code library.
    Library = 5,
}

// =========================================================================
// Submodules — extracted per refactoring policy (<2000 lines per file)
// =========================================================================

#[path = "ffi_constants.rs"]
mod ffi_constants;
pub use ffi_constants::*;

#[path = "ffi_launch.rs"]
mod ffi_launch;
pub use ffi_launch::*;

#[path = "ffi_descriptors.rs"]
mod ffi_descriptors;
pub use ffi_descriptors::*;

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_success_is_zero() {
        assert_eq!(CUDA_SUCCESS, 0);
    }

    #[test]
    fn test_opaque_types_are_pointer_sized() {
        assert_eq!(
            std::mem::size_of::<CUcontext>(),
            std::mem::size_of::<*mut c_void>()
        );
        assert_eq!(
            std::mem::size_of::<CUmodule>(),
            std::mem::size_of::<*mut c_void>()
        );
        assert_eq!(
            std::mem::size_of::<CUstream>(),
            std::mem::size_of::<*mut c_void>()
        );
        assert_eq!(
            std::mem::size_of::<CUevent>(),
            std::mem::size_of::<*mut c_void>()
        );
        assert_eq!(
            std::mem::size_of::<CUfunction>(),
            std::mem::size_of::<*mut c_void>()
        );
        assert_eq!(
            std::mem::size_of::<CUmemoryPool>(),
            std::mem::size_of::<*mut c_void>()
        );
    }

    #[test]
    fn test_handle_default_is_null() {
        assert!(CUcontext::default().is_null());
        assert!(CUmodule::default().is_null());
        assert!(CUfunction::default().is_null());
        assert!(CUstream::default().is_null());
        assert!(CUevent::default().is_null());
        assert!(CUmemoryPool::default().is_null());
    }

    #[test]
    fn test_device_attribute_repr() {
        // Original variants
        assert_eq!(CUdevice_attribute::MaxThreadsPerBlock as i32, 1);
        assert_eq!(CUdevice_attribute::WarpSize as i32, 10);
        assert_eq!(CUdevice_attribute::MultiprocessorCount as i32, 16);
        assert_eq!(CUdevice_attribute::ComputeCapabilityMajor as i32, 75);
        assert_eq!(CUdevice_attribute::ComputeCapabilityMinor as i32, 76);
        assert_eq!(CUdevice_attribute::MaxBlocksPerMultiprocessor as i32, 106);
        assert_eq!(CUdevice_attribute::L2CacheSize as i32, 38);
        assert_eq!(
            CUdevice_attribute::MaxSharedMemoryPerMultiprocessor as i32,
            81
        );
        assert_eq!(CUdevice_attribute::ManagedMemory as i32, 83);

        // New variants
        assert_eq!(CUdevice_attribute::MaxTexture2DGatherWidth as i32, 44);
        assert_eq!(CUdevice_attribute::MaxTexture2DGatherHeight as i32, 45);
        assert_eq!(CUdevice_attribute::MaxTexture3DWidthAlt as i32, 47);
        assert_eq!(CUdevice_attribute::MaxTexture3DHeightAlt as i32, 48);
        assert_eq!(CUdevice_attribute::MaxTexture3DDepthAlt as i32, 49);
        assert_eq!(CUdevice_attribute::MaxTexture1DMipmappedWidth2 as i32, 52);
        assert_eq!(CUdevice_attribute::Reserved92 as i32, 92);
        assert_eq!(CUdevice_attribute::Reserved93 as i32, 93);
        assert_eq!(CUdevice_attribute::Reserved94 as i32, 94);
        assert_eq!(
            CUdevice_attribute::VirtualMemoryManagementSupported as i32,
            102
        );
        assert_eq!(
            CUdevice_attribute::HandleTypePosixFileDescriptorSupported as i32,
            103
        );
        assert_eq!(
            CUdevice_attribute::HandleTypeWin32HandleSupported as i32,
            104
        );
        assert_eq!(
            CUdevice_attribute::HandleTypeWin32KmtHandleSupported as i32,
            105
        );
        assert_eq!(CUdevice_attribute::AccessPolicyMaxWindowSize as i32, 111);
        assert_eq!(CUdevice_attribute::ReservedSharedMemoryPerBlock as i32, 112);
        assert_eq!(
            CUdevice_attribute::TimelineSemaphoreInteropSupported as i32,
            113
        );
        assert_eq!(CUdevice_attribute::MemoryPoolsSupported as i32, 115);
        assert_eq!(CUdevice_attribute::ClusterLaunch as i32, 120);
        assert_eq!(CUdevice_attribute::UnifiedFunctionPointers as i32, 125);
        assert_eq!(
            CUdevice_attribute::MaxTimelineSemaphoreInteropSupported as i32,
            129
        );
        assert_eq!(CUdevice_attribute::MemSyncDomainSupported as i32, 130);
        assert_eq!(CUdevice_attribute::GpuDirectRdmaFabricSupported as i32, 131);
    }

    #[test]
    fn test_jit_option_repr() {
        assert_eq!(CUjit_option::MaxRegisters as u32, 0);
        assert_eq!(CUjit_option::ThreadsPerBlock as u32, 1);
        assert_eq!(CUjit_option::WallTime as u32, 2);
        assert_eq!(CUjit_option::InfoLogBuffer as u32, 3);
        assert_eq!(CUjit_option::InfoLogBufferSizeBytes as u32, 4);
        assert_eq!(CUjit_option::ErrorLogBuffer as u32, 5);
        assert_eq!(CUjit_option::ErrorLogBufferSizeBytes as u32, 6);
        assert_eq!(CUjit_option::OptimizationLevel as u32, 7);
        assert_eq!(CUjit_option::Target as u32, 9);
        assert_eq!(CUjit_option::FallbackStrategy as u32, 10);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_error_code_ranges() {
        // Basic errors: 1-8
        assert!(CUDA_ERROR_INVALID_VALUE < 10);
        // Device errors: 100-102
        assert!((100..=102).contains(&CUDA_ERROR_NO_DEVICE));
        assert!((100..=102).contains(&CUDA_ERROR_INVALID_DEVICE));
        assert!((100..=102).contains(&CUDA_ERROR_DEVICE_NOT_LICENSED));
        // Image/context errors: 200+
        assert!(CUDA_ERROR_INVALID_IMAGE >= 200);
        // Launch errors: 700+
        assert!(CUDA_ERROR_LAUNCH_FAILED >= 700);
        assert!(CUDA_ERROR_ILLEGAL_ADDRESS >= 700);
        assert!(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES >= 700);
        // Stream capture errors: 900+
        assert!(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED >= 900);
        // Unknown is 999
        assert_eq!(CUDA_ERROR_UNKNOWN, 999);
    }

    #[test]
    fn test_handle_debug_format() {
        let ctx = CUcontext::default();
        let debug_str = format!("{ctx:?}");
        assert!(debug_str.starts_with("CUcontext("));
    }

    #[test]
    fn test_handle_equality() {
        let a = CUcontext::default();
        let b = CUcontext::default();
        assert_eq!(a, b);
    }

    #[test]
    fn test_new_handle_types_are_pointer_sized() {
        assert_eq!(
            std::mem::size_of::<CUtexref>(),
            std::mem::size_of::<*mut c_void>()
        );
        assert_eq!(
            std::mem::size_of::<CUsurfref>(),
            std::mem::size_of::<*mut c_void>()
        );
        assert_eq!(
            std::mem::size_of::<CUtexObject>(),
            std::mem::size_of::<*mut c_void>()
        );
        assert_eq!(
            std::mem::size_of::<CUsurfObject>(),
            std::mem::size_of::<*mut c_void>()
        );
    }

    #[test]
    fn test_new_handle_defaults_are_null() {
        assert!(CUtexref::default().is_null());
        assert!(CUsurfref::default().is_null());
        assert!(CUtexObject::default().is_null());
        assert!(CUsurfObject::default().is_null());
    }

    #[test]
    fn test_memory_type_enum() {
        assert_eq!(CUmemorytype::Host as u32, 1);
        assert_eq!(CUmemorytype::Device as u32, 2);
        assert_eq!(CUmemorytype::Array as u32, 3);
        assert_eq!(CUmemorytype::Unified as u32, 4);
    }

    #[test]
    fn test_pointer_attribute_enum() {
        assert_eq!(CUpointer_attribute::Context as u32, 1);
        assert_eq!(CUpointer_attribute::MemoryType as u32, 2);
        assert_eq!(CUpointer_attribute::DevicePointer as u32, 3);
        assert_eq!(CUpointer_attribute::HostPointer as u32, 4);
        assert_eq!(CUpointer_attribute::IsManaged as u32, 9);
        assert_eq!(CUpointer_attribute::DeviceOrdinal as u32, 10);
    }

    #[test]
    fn test_limit_enum() {
        assert_eq!(CUlimit::StackSize as u32, 0);
        assert_eq!(CUlimit::PrintfFifoSize as u32, 1);
        assert_eq!(CUlimit::MallocHeapSize as u32, 2);
        assert_eq!(CUlimit::DevRuntimeSyncDepth as u32, 3);
        assert_eq!(CUlimit::DevRuntimePendingLaunchCount as u32, 4);
        assert_eq!(CUlimit::MaxL2FetchGranularity as u32, 5);
        assert_eq!(CUlimit::PersistingL2CacheSize as u32, 6);
    }

    #[test]
    fn test_function_attribute_enum() {
        assert_eq!(CUfunction_attribute::MaxThreadsPerBlock as i32, 0);
        assert_eq!(CUfunction_attribute::SharedSizeBytes as i32, 1);
        assert_eq!(CUfunction_attribute::NumRegs as i32, 4);
        assert_eq!(CUfunction_attribute::PtxVersion as i32, 5);
        assert_eq!(CUfunction_attribute::BinaryVersion as i32, 6);
        assert_eq!(CUfunction_attribute::MaxDynamicSharedSizeBytes as i32, 8);
        assert_eq!(
            CUfunction_attribute::PreferredSharedMemoryCarveout as i32,
            9
        );
    }
}
