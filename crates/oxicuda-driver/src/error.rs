//! Error types for the OxiCUDA driver crate.
//!
//! This module provides [`CudaError`], the primary error type returned by
//! driver API wrappers, [`DriverLoadError`] for library-loading failures,
//! the [`check`] function for converting raw result codes, and the
//! `cuda_call!` macro for ergonomic unsafe FFI calls.

use crate::ffi;

// =========================================================================
// CudaError — one variant per CUDA Driver API error code
// =========================================================================

/// Primary error type for CUDA driver API calls.
///
/// Each variant maps to a specific `CUresult` code from the CUDA Driver API.
/// The [`Unknown`](CudaError::Unknown) variant is the catch-all for codes not
/// explicitly listed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum CudaError {
    // ----- Basic errors (1-8) -----
    /// `CUDA_ERROR_INVALID_VALUE` (1)
    #[error("CUDA: invalid value")]
    InvalidValue,

    /// `CUDA_ERROR_OUT_OF_MEMORY` (2)
    #[error("CUDA: out of device memory")]
    OutOfMemory,

    /// `CUDA_ERROR_NOT_INITIALIZED` (3)
    #[error("CUDA: not initialized")]
    NotInitialized,

    /// `CUDA_ERROR_DEINITIALIZED` (4)
    #[error("CUDA: deinitialized")]
    Deinitialized,

    /// `CUDA_ERROR_PROFILER_DISABLED` (5)
    #[error("CUDA: profiler disabled")]
    ProfilerDisabled,

    /// `CUDA_ERROR_PROFILER_NOT_INITIALIZED` (6)
    #[error("CUDA: profiler not initialized (deprecated)")]
    ProfilerNotInitialized,

    /// `CUDA_ERROR_PROFILER_ALREADY_STARTED` (7)
    #[error("CUDA: profiler already started (deprecated)")]
    ProfilerAlreadyStarted,

    /// `CUDA_ERROR_PROFILER_ALREADY_STOPPED` (8)
    #[error("CUDA: profiler already stopped (deprecated)")]
    ProfilerAlreadyStopped,

    // ----- Stub / unavailable (34, 46) -----
    /// `CUDA_ERROR_STUB_LIBRARY` (34)
    #[error("CUDA: stub library loaded instead of real driver")]
    StubLibrary,

    /// `CUDA_ERROR_DEVICE_UNAVAILABLE` (46)
    #[error("CUDA: device unavailable")]
    DeviceUnavailable,

    // ----- Device errors (100-102) -----
    /// `CUDA_ERROR_NO_DEVICE` (100)
    #[error("CUDA: no device found")]
    NoDevice,

    /// `CUDA_ERROR_INVALID_DEVICE` (101)
    #[error("CUDA: invalid device")]
    InvalidDevice,

    /// `CUDA_ERROR_DEVICE_NOT_LICENSED` (102)
    #[error("CUDA: device not licensed")]
    DeviceNotLicensed,

    // ----- Image / context / map errors (200-225) -----
    /// `CUDA_ERROR_INVALID_IMAGE` (200)
    #[error("CUDA: invalid PTX/cubin image")]
    InvalidImage,

    /// `CUDA_ERROR_INVALID_CONTEXT` (201)
    #[error("CUDA: invalid context")]
    InvalidContext,

    /// `CUDA_ERROR_CONTEXT_ALREADY_CURRENT` (202)
    #[error("CUDA: context already current (deprecated)")]
    ContextAlreadyCurrent,

    /// `CUDA_ERROR_MAP_FAILED` (205)
    #[error("CUDA: map failed")]
    MapFailed,

    /// `CUDA_ERROR_UNMAP_FAILED` (206)
    #[error("CUDA: unmap failed")]
    UnmapFailed,

    /// `CUDA_ERROR_ARRAY_IS_MAPPED` (207)
    #[error("CUDA: array is mapped")]
    ArrayIsMapped,

    /// `CUDA_ERROR_ALREADY_MAPPED` (208)
    #[error("CUDA: already mapped")]
    AlreadyMapped,

    /// `CUDA_ERROR_NO_BINARY_FOR_GPU` (209)
    #[error("CUDA: no binary for GPU")]
    NoBinaryForGpu,

    /// `CUDA_ERROR_ALREADY_ACQUIRED` (210)
    #[error("CUDA: already acquired")]
    AlreadyAcquired,

    /// `CUDA_ERROR_NOT_MAPPED` (211)
    #[error("CUDA: not mapped")]
    NotMapped,

    /// `CUDA_ERROR_NOT_MAPPED_AS_ARRAY` (212)
    #[error("CUDA: not mapped as array")]
    NotMappedAsArray,

    /// `CUDA_ERROR_NOT_MAPPED_AS_POINTER` (213)
    #[error("CUDA: not mapped as pointer")]
    NotMappedAsPointer,

    /// `CUDA_ERROR_ECC_UNCORRECTABLE` (214)
    #[error("CUDA: uncorrectable ECC error")]
    EccUncorrectable,

    /// `CUDA_ERROR_UNSUPPORTED_LIMIT` (215)
    #[error("CUDA: unsupported limit")]
    UnsupportedLimit,

    /// `CUDA_ERROR_CONTEXT_ALREADY_IN_USE` (216)
    #[error("CUDA: context already in use")]
    ContextAlreadyInUse,

    /// `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED` (217)
    #[error("CUDA: peer access unsupported")]
    PeerAccessUnsupported,

    /// `CUDA_ERROR_INVALID_PTX` (218)
    #[error("CUDA: invalid PTX")]
    InvalidPtx,

    /// `CUDA_ERROR_INVALID_GRAPHICS_CONTEXT` (219)
    #[error("CUDA: invalid graphics context")]
    InvalidGraphicsContext,

    /// `CUDA_ERROR_NVLINK_UNCORRECTABLE` (220)
    #[error("CUDA: NVLINK uncorrectable error")]
    NvlinkUncorrectable,

    /// `CUDA_ERROR_JIT_COMPILER_NOT_FOUND` (221)
    #[error("CUDA: JIT compiler not found")]
    JitCompilerNotFound,

    /// `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (222)
    #[error("CUDA: unsupported PTX version")]
    UnsupportedPtxVersion,

    /// `CUDA_ERROR_JIT_COMPILATION_DISABLED` (223)
    #[error("CUDA: JIT compilation disabled")]
    JitCompilationDisabled,

    /// `CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY` (224)
    #[error("CUDA: unsupported exec affinity")]
    UnsupportedExecAffinity,

    /// `CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC` (225)
    #[error("CUDA: unsupported device-side sync")]
    UnsupportedDevsideSync,

    // ----- Source / file / shared-object errors (300-304) -----
    /// `CUDA_ERROR_INVALID_SOURCE` (300)
    #[error("CUDA: invalid source")]
    InvalidSource,

    /// `CUDA_ERROR_FILE_NOT_FOUND` (301)
    #[error("CUDA: file not found")]
    FileNotFound,

    /// `CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND` (302)
    #[error("CUDA: shared object symbol not found")]
    SharedObjectSymbolNotFound,

    /// `CUDA_ERROR_SHARED_OBJECT_INIT_FAILED` (303)
    #[error("CUDA: shared object init failed")]
    SharedObjectInitFailed,

    /// `CUDA_ERROR_OPERATING_SYSTEM` (304)
    #[error("CUDA: operating system error")]
    OperatingSystem,

    // ----- Handle / state errors (400-402) -----
    /// `CUDA_ERROR_INVALID_HANDLE` (400)
    #[error("CUDA: invalid handle")]
    InvalidHandle,

    /// `CUDA_ERROR_ILLEGAL_STATE` (401)
    #[error("CUDA: illegal state")]
    IllegalState,

    /// `CUDA_ERROR_LOSSY_QUERY` (402)
    #[error("CUDA: lossy query")]
    LossyQuery,

    // ----- Lookup error (500) -----
    /// `CUDA_ERROR_NOT_FOUND` (500)
    #[error("CUDA: symbol not found")]
    NotFound,

    // ----- Readiness error (600) -----
    /// `CUDA_ERROR_NOT_READY` (600)
    #[error("CUDA: not ready (async operation pending)")]
    NotReady,

    // ----- Launch / address / peer errors (700-720) -----
    /// `CUDA_ERROR_ILLEGAL_ADDRESS` (700)
    #[error("CUDA: illegal memory address")]
    IllegalAddress,

    /// `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` (701)
    #[error("CUDA: kernel launch out of resources (registers/shared memory)")]
    LaunchOutOfResources,

    /// `CUDA_ERROR_LAUNCH_TIMEOUT` (702)
    #[error("CUDA: kernel launch timeout")]
    LaunchTimeout,

    /// `CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING` (703)
    #[error("CUDA: launch incompatible texturing")]
    LaunchIncompatibleTexturing,

    /// `CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED` (704)
    #[error("CUDA: peer access already enabled")]
    PeerAccessAlreadyEnabled,

    /// `CUDA_ERROR_PEER_ACCESS_NOT_ENABLED` (705)
    #[error("CUDA: peer access not enabled")]
    PeerAccessNotEnabled,

    /// `CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE` (708)
    #[error("CUDA: primary context active")]
    PrimaryContextActive,

    /// `CUDA_ERROR_CONTEXT_IS_DESTROYED` (709)
    #[error("CUDA: context is destroyed")]
    ContextIsDestroyed,

    /// `CUDA_ERROR_ASSERT` (710)
    #[error("CUDA: device-side assert triggered")]
    Assert,

    /// `CUDA_ERROR_TOO_MANY_PEERS` (711)
    #[error("CUDA: too many peers")]
    TooManyPeers,

    /// `CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED` (712)
    #[error("CUDA: host memory already registered")]
    HostMemoryAlreadyRegistered,

    /// `CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED` (713)
    #[error("CUDA: host memory not registered")]
    HostMemoryNotRegistered,

    /// `CUDA_ERROR_HARDWARE_STACK_ERROR` (714)
    #[error("CUDA: hardware stack error")]
    HardwareStackError,

    /// `CUDA_ERROR_ILLEGAL_INSTRUCTION` (715)
    #[error("CUDA: illegal instruction")]
    IllegalInstruction,

    /// `CUDA_ERROR_MISALIGNED_ADDRESS` (716)
    #[error("CUDA: misaligned address")]
    MisalignedAddress,

    /// `CUDA_ERROR_INVALID_ADDRESS_SPACE` (717)
    #[error("CUDA: invalid address space")]
    InvalidAddressSpace,

    /// `CUDA_ERROR_INVALID_PC` (718)
    #[error("CUDA: invalid program counter")]
    InvalidPc,

    /// `CUDA_ERROR_LAUNCH_FAILED` (719)
    #[error("CUDA: kernel launch failed")]
    LaunchFailed,

    /// `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` (720)
    #[error("CUDA: cooperative launch too large")]
    CooperativeLaunchTooLarge,

    // ----- Permission / support errors (800-812) -----
    /// `CUDA_ERROR_NOT_PERMITTED` (800)
    #[error("CUDA: not permitted")]
    NotPermitted,

    /// `CUDA_ERROR_NOT_SUPPORTED` (801)
    #[error("CUDA: not supported")]
    NotSupported,

    /// `CUDA_ERROR_SYSTEM_NOT_READY` (802)
    #[error("CUDA: system not ready")]
    SystemNotReady,

    /// `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` (803)
    #[error("CUDA: system driver mismatch")]
    SystemDriverMismatch,

    /// `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE` (804)
    #[error("CUDA: compat not supported on device")]
    CompatNotSupportedOnDevice,

    /// `CUDA_ERROR_MPS_CONNECTION_FAILED` (805)
    #[error("CUDA: MPS connection failed")]
    MpsConnectionFailed,

    /// `CUDA_ERROR_MPS_RPC_FAILURE` (806)
    #[error("CUDA: MPS RPC failure")]
    MpsRpcFailure,

    /// `CUDA_ERROR_MPS_SERVER_NOT_READY` (807)
    #[error("CUDA: MPS server not ready")]
    MpsServerNotReady,

    /// `CUDA_ERROR_MPS_MAX_CLIENTS_REACHED` (808)
    #[error("CUDA: MPS max clients reached")]
    MpsMaxClientsReached,

    /// `CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED` (809)
    #[error("CUDA: MPS max connections reached")]
    MpsMaxConnectionsReached,

    /// `CUDA_ERROR_MPS_CLIENT_TERMINATED` (810)
    #[error("CUDA: MPS client terminated")]
    MpsClientTerminated,

    /// `CUDA_ERROR_CDP_NOT_SUPPORTED` (811)
    #[error("CUDA: CDP not supported")]
    CdpNotSupported,

    /// `CUDA_ERROR_CDP_VERSION_MISMATCH` (812)
    #[error("CUDA: CDP version mismatch")]
    CdpVersionMismatch,

    // ----- Stream capture / graph errors (900-915) -----
    /// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` (900)
    #[error("CUDA: stream capture unsupported")]
    StreamCaptureUnsupported,

    /// `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` (901)
    #[error("CUDA: stream capture invalidated")]
    StreamCaptureInvalidated,

    /// `CUDA_ERROR_STREAM_CAPTURE_MERGE` (902)
    #[error("CUDA: stream capture merge not permitted")]
    StreamCaptureMerge,

    /// `CUDA_ERROR_STREAM_CAPTURE_UNMATCHED` (903)
    #[error("CUDA: stream capture unmatched")]
    StreamCaptureUnmatched,

    /// `CUDA_ERROR_STREAM_CAPTURE_UNJOINED` (904)
    #[error("CUDA: stream capture unjoined")]
    StreamCaptureUnjoined,

    /// `CUDA_ERROR_STREAM_CAPTURE_ISOLATION` (905)
    #[error("CUDA: stream capture isolation violation")]
    StreamCaptureIsolation,

    /// `CUDA_ERROR_STREAM_CAPTURE_IMPLICIT` (906)
    #[error("CUDA: implicit stream in graph capture")]
    StreamCaptureImplicit,

    /// `CUDA_ERROR_CAPTURED_EVENT` (907)
    #[error("CUDA: captured event error")]
    CapturedEvent,

    /// `CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD` (908)
    #[error("CUDA: stream capture wrong thread")]
    StreamCaptureWrongThread,

    /// `CUDA_ERROR_TIMEOUT` (909)
    #[error("CUDA: async operation timed out")]
    Timeout,

    /// `CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE` (910)
    #[error("CUDA: graph exec update failure")]
    GraphExecUpdateFailure,

    /// `CUDA_ERROR_EXTERNAL_DEVICE` (911)
    #[error("CUDA: external device error")]
    ExternalDevice,

    /// `CUDA_ERROR_INVALID_CLUSTER_SIZE` (912)
    #[error("CUDA: invalid cluster size")]
    InvalidClusterSize,

    /// `CUDA_ERROR_FUNCTION_NOT_LOADED` (913)
    #[error("CUDA: function not loaded")]
    FunctionNotLoaded,

    /// `CUDA_ERROR_INVALID_RESOURCE_TYPE` (914)
    #[error("CUDA: invalid resource type")]
    InvalidResourceType,

    /// `CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION` (915)
    #[error("CUDA: invalid resource configuration")]
    InvalidResourceConfiguration,

    // ----- Catch-all -----
    /// Unknown error code not covered by any other variant.
    #[error("CUDA: unknown error (code {0})")]
    Unknown(u32),
}

impl CudaError {
    /// Convert a raw `CUresult` code into the corresponding [`CudaError`] variant.
    ///
    /// This should not be called with `CUDA_SUCCESS` (0); prefer using [`check`]
    /// which returns `Ok(())` for success.
    #[allow(clippy::too_many_lines)]
    pub fn from_raw(code: u32) -> Self {
        match code {
            ffi::CUDA_ERROR_INVALID_VALUE => Self::InvalidValue,
            ffi::CUDA_ERROR_OUT_OF_MEMORY => Self::OutOfMemory,
            ffi::CUDA_ERROR_NOT_INITIALIZED => Self::NotInitialized,
            ffi::CUDA_ERROR_DEINITIALIZED => Self::Deinitialized,
            ffi::CUDA_ERROR_PROFILER_DISABLED => Self::ProfilerDisabled,
            ffi::CUDA_ERROR_PROFILER_NOT_INITIALIZED => Self::ProfilerNotInitialized,
            ffi::CUDA_ERROR_PROFILER_ALREADY_STARTED => Self::ProfilerAlreadyStarted,
            ffi::CUDA_ERROR_PROFILER_ALREADY_STOPPED => Self::ProfilerAlreadyStopped,
            ffi::CUDA_ERROR_STUB_LIBRARY => Self::StubLibrary,
            ffi::CUDA_ERROR_DEVICE_UNAVAILABLE => Self::DeviceUnavailable,
            ffi::CUDA_ERROR_NO_DEVICE => Self::NoDevice,
            ffi::CUDA_ERROR_INVALID_DEVICE => Self::InvalidDevice,
            ffi::CUDA_ERROR_DEVICE_NOT_LICENSED => Self::DeviceNotLicensed,
            ffi::CUDA_ERROR_INVALID_IMAGE => Self::InvalidImage,
            ffi::CUDA_ERROR_INVALID_CONTEXT => Self::InvalidContext,
            ffi::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => Self::ContextAlreadyCurrent,
            ffi::CUDA_ERROR_MAP_FAILED => Self::MapFailed,
            ffi::CUDA_ERROR_UNMAP_FAILED => Self::UnmapFailed,
            ffi::CUDA_ERROR_ARRAY_IS_MAPPED => Self::ArrayIsMapped,
            ffi::CUDA_ERROR_ALREADY_MAPPED => Self::AlreadyMapped,
            ffi::CUDA_ERROR_NO_BINARY_FOR_GPU => Self::NoBinaryForGpu,
            ffi::CUDA_ERROR_ALREADY_ACQUIRED => Self::AlreadyAcquired,
            ffi::CUDA_ERROR_NOT_MAPPED => Self::NotMapped,
            ffi::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Self::NotMappedAsArray,
            ffi::CUDA_ERROR_NOT_MAPPED_AS_POINTER => Self::NotMappedAsPointer,
            ffi::CUDA_ERROR_ECC_UNCORRECTABLE => Self::EccUncorrectable,
            ffi::CUDA_ERROR_UNSUPPORTED_LIMIT => Self::UnsupportedLimit,
            ffi::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => Self::ContextAlreadyInUse,
            ffi::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => Self::PeerAccessUnsupported,
            ffi::CUDA_ERROR_INVALID_PTX => Self::InvalidPtx,
            ffi::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => Self::InvalidGraphicsContext,
            ffi::CUDA_ERROR_NVLINK_UNCORRECTABLE => Self::NvlinkUncorrectable,
            ffi::CUDA_ERROR_JIT_COMPILER_NOT_FOUND => Self::JitCompilerNotFound,
            ffi::CUDA_ERROR_UNSUPPORTED_PTX_VERSION => Self::UnsupportedPtxVersion,
            ffi::CUDA_ERROR_JIT_COMPILATION_DISABLED => Self::JitCompilationDisabled,
            ffi::CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY => Self::UnsupportedExecAffinity,
            ffi::CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC => Self::UnsupportedDevsideSync,
            ffi::CUDA_ERROR_INVALID_SOURCE => Self::InvalidSource,
            ffi::CUDA_ERROR_FILE_NOT_FOUND => Self::FileNotFound,
            ffi::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => Self::SharedObjectSymbolNotFound,
            ffi::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => Self::SharedObjectInitFailed,
            ffi::CUDA_ERROR_OPERATING_SYSTEM => Self::OperatingSystem,
            ffi::CUDA_ERROR_INVALID_HANDLE => Self::InvalidHandle,
            ffi::CUDA_ERROR_ILLEGAL_STATE => Self::IllegalState,
            ffi::CUDA_ERROR_LOSSY_QUERY => Self::LossyQuery,
            ffi::CUDA_ERROR_NOT_FOUND => Self::NotFound,
            ffi::CUDA_ERROR_NOT_READY => Self::NotReady,
            ffi::CUDA_ERROR_ILLEGAL_ADDRESS => Self::IllegalAddress,
            ffi::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => Self::LaunchOutOfResources,
            ffi::CUDA_ERROR_LAUNCH_TIMEOUT => Self::LaunchTimeout,
            ffi::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => Self::LaunchIncompatibleTexturing,
            ffi::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => Self::PeerAccessAlreadyEnabled,
            ffi::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => Self::PeerAccessNotEnabled,
            ffi::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => Self::PrimaryContextActive,
            ffi::CUDA_ERROR_CONTEXT_IS_DESTROYED => Self::ContextIsDestroyed,
            ffi::CUDA_ERROR_ASSERT => Self::Assert,
            ffi::CUDA_ERROR_TOO_MANY_PEERS => Self::TooManyPeers,
            ffi::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => Self::HostMemoryAlreadyRegistered,
            ffi::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => Self::HostMemoryNotRegistered,
            ffi::CUDA_ERROR_HARDWARE_STACK_ERROR => Self::HardwareStackError,
            ffi::CUDA_ERROR_ILLEGAL_INSTRUCTION => Self::IllegalInstruction,
            ffi::CUDA_ERROR_MISALIGNED_ADDRESS => Self::MisalignedAddress,
            ffi::CUDA_ERROR_INVALID_ADDRESS_SPACE => Self::InvalidAddressSpace,
            ffi::CUDA_ERROR_INVALID_PC => Self::InvalidPc,
            ffi::CUDA_ERROR_LAUNCH_FAILED => Self::LaunchFailed,
            ffi::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE => Self::CooperativeLaunchTooLarge,
            ffi::CUDA_ERROR_NOT_PERMITTED => Self::NotPermitted,
            ffi::CUDA_ERROR_NOT_SUPPORTED => Self::NotSupported,
            ffi::CUDA_ERROR_SYSTEM_NOT_READY => Self::SystemNotReady,
            ffi::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => Self::SystemDriverMismatch,
            ffi::CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => Self::CompatNotSupportedOnDevice,
            ffi::CUDA_ERROR_MPS_CONNECTION_FAILED => Self::MpsConnectionFailed,
            ffi::CUDA_ERROR_MPS_RPC_FAILURE => Self::MpsRpcFailure,
            ffi::CUDA_ERROR_MPS_SERVER_NOT_READY => Self::MpsServerNotReady,
            ffi::CUDA_ERROR_MPS_MAX_CLIENTS_REACHED => Self::MpsMaxClientsReached,
            ffi::CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED => Self::MpsMaxConnectionsReached,
            ffi::CUDA_ERROR_MPS_CLIENT_TERMINATED => Self::MpsClientTerminated,
            ffi::CUDA_ERROR_CDP_NOT_SUPPORTED => Self::CdpNotSupported,
            ffi::CUDA_ERROR_CDP_VERSION_MISMATCH => Self::CdpVersionMismatch,
            ffi::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED => Self::StreamCaptureUnsupported,
            ffi::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED => Self::StreamCaptureInvalidated,
            ffi::CUDA_ERROR_STREAM_CAPTURE_MERGE => Self::StreamCaptureMerge,
            ffi::CUDA_ERROR_STREAM_CAPTURE_UNMATCHED => Self::StreamCaptureUnmatched,
            ffi::CUDA_ERROR_STREAM_CAPTURE_UNJOINED => Self::StreamCaptureUnjoined,
            ffi::CUDA_ERROR_STREAM_CAPTURE_ISOLATION => Self::StreamCaptureIsolation,
            ffi::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT => Self::StreamCaptureImplicit,
            ffi::CUDA_ERROR_CAPTURED_EVENT => Self::CapturedEvent,
            ffi::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD => Self::StreamCaptureWrongThread,
            ffi::CUDA_ERROR_TIMEOUT => Self::Timeout,
            ffi::CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE => Self::GraphExecUpdateFailure,
            ffi::CUDA_ERROR_EXTERNAL_DEVICE => Self::ExternalDevice,
            ffi::CUDA_ERROR_INVALID_CLUSTER_SIZE => Self::InvalidClusterSize,
            ffi::CUDA_ERROR_FUNCTION_NOT_LOADED => Self::FunctionNotLoaded,
            ffi::CUDA_ERROR_INVALID_RESOURCE_TYPE => Self::InvalidResourceType,
            ffi::CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION => Self::InvalidResourceConfiguration,
            ffi::CUDA_ERROR_UNKNOWN => Self::Unknown(999),
            other => Self::Unknown(other),
        }
    }

    /// Returns `true` for errors that indicate a **fatal**, non-recoverable
    /// GPU state — meaning the current CUDA context is likely corrupted and
    /// should be destroyed before attempting further GPU work.
    ///
    /// # Rationale
    ///
    /// Most CUDA errors are caused by incorrect API usage (e.g. bad argument,
    /// unsupported feature) and do not invalidate the GPU context. A small
    /// subset — hardware faults, illegal instructions, or illegal memory
    /// accesses — leave the device in an indeterminate state.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxicuda_driver::CudaError;
    ///
    /// // Hardware fault — fatal
    /// assert!(CudaError::IllegalAddress.is_fatal());
    /// assert!(CudaError::LaunchFailed.is_fatal());
    /// assert!(CudaError::HardwareStackError.is_fatal());
    ///
    /// // Bad argument — recoverable
    /// assert!(!CudaError::InvalidValue.is_fatal());
    /// assert!(!CudaError::OutOfMemory.is_fatal());
    /// assert!(!CudaError::NotReady.is_fatal());
    /// ```
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            // Hardware / device-side faults that corrupt context state.
            Self::IllegalAddress
                | Self::LaunchFailed
                | Self::HardwareStackError
                | Self::IllegalInstruction
                | Self::MisalignedAddress
                | Self::InvalidAddressSpace
                | Self::InvalidPc
                | Self::Assert
                | Self::EccUncorrectable
                | Self::NvlinkUncorrectable
                // Driver / context state is unrecoverable.
                | Self::ContextIsDestroyed
                | Self::Deinitialized
        )
    }

    /// Returns `true` for errors that are clearly caused by incorrect API
    /// usage rather than hardware faults or resource exhaustion.
    ///
    /// These errors indicate the caller passed bad arguments and the GPU
    /// state is intact.
    ///
    /// ```
    /// use oxicuda_driver::CudaError;
    ///
    /// assert!(CudaError::InvalidValue.is_usage_error());
    /// assert!(CudaError::InvalidDevice.is_usage_error());
    /// assert!(!CudaError::OutOfMemory.is_usage_error());
    /// ```
    #[must_use]
    pub fn is_usage_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidValue
                | Self::InvalidDevice
                | Self::InvalidContext
                | Self::InvalidHandle
                | Self::InvalidImage
                | Self::InvalidPtx
                | Self::InvalidSource
                | Self::InvalidClusterSize
                | Self::NoDevice
                | Self::UnsupportedLimit
                | Self::NotSupported
                | Self::NotPermitted
        )
    }

    /// Convert this error back to its raw `CUresult` code.
    #[allow(clippy::too_many_lines)]
    pub fn as_raw(&self) -> u32 {
        match self {
            Self::InvalidValue => ffi::CUDA_ERROR_INVALID_VALUE,
            Self::OutOfMemory => ffi::CUDA_ERROR_OUT_OF_MEMORY,
            Self::NotInitialized => ffi::CUDA_ERROR_NOT_INITIALIZED,
            Self::Deinitialized => ffi::CUDA_ERROR_DEINITIALIZED,
            Self::ProfilerDisabled => ffi::CUDA_ERROR_PROFILER_DISABLED,
            Self::ProfilerNotInitialized => ffi::CUDA_ERROR_PROFILER_NOT_INITIALIZED,
            Self::ProfilerAlreadyStarted => ffi::CUDA_ERROR_PROFILER_ALREADY_STARTED,
            Self::ProfilerAlreadyStopped => ffi::CUDA_ERROR_PROFILER_ALREADY_STOPPED,
            Self::StubLibrary => ffi::CUDA_ERROR_STUB_LIBRARY,
            Self::DeviceUnavailable => ffi::CUDA_ERROR_DEVICE_UNAVAILABLE,
            Self::NoDevice => ffi::CUDA_ERROR_NO_DEVICE,
            Self::InvalidDevice => ffi::CUDA_ERROR_INVALID_DEVICE,
            Self::DeviceNotLicensed => ffi::CUDA_ERROR_DEVICE_NOT_LICENSED,
            Self::InvalidImage => ffi::CUDA_ERROR_INVALID_IMAGE,
            Self::InvalidContext => ffi::CUDA_ERROR_INVALID_CONTEXT,
            Self::ContextAlreadyCurrent => ffi::CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
            Self::MapFailed => ffi::CUDA_ERROR_MAP_FAILED,
            Self::UnmapFailed => ffi::CUDA_ERROR_UNMAP_FAILED,
            Self::ArrayIsMapped => ffi::CUDA_ERROR_ARRAY_IS_MAPPED,
            Self::AlreadyMapped => ffi::CUDA_ERROR_ALREADY_MAPPED,
            Self::NoBinaryForGpu => ffi::CUDA_ERROR_NO_BINARY_FOR_GPU,
            Self::AlreadyAcquired => ffi::CUDA_ERROR_ALREADY_ACQUIRED,
            Self::NotMapped => ffi::CUDA_ERROR_NOT_MAPPED,
            Self::NotMappedAsArray => ffi::CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
            Self::NotMappedAsPointer => ffi::CUDA_ERROR_NOT_MAPPED_AS_POINTER,
            Self::EccUncorrectable => ffi::CUDA_ERROR_ECC_UNCORRECTABLE,
            Self::UnsupportedLimit => ffi::CUDA_ERROR_UNSUPPORTED_LIMIT,
            Self::ContextAlreadyInUse => ffi::CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
            Self::PeerAccessUnsupported => ffi::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
            Self::InvalidPtx => ffi::CUDA_ERROR_INVALID_PTX,
            Self::InvalidGraphicsContext => ffi::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
            Self::NvlinkUncorrectable => ffi::CUDA_ERROR_NVLINK_UNCORRECTABLE,
            Self::JitCompilerNotFound => ffi::CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
            Self::UnsupportedPtxVersion => ffi::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
            Self::JitCompilationDisabled => ffi::CUDA_ERROR_JIT_COMPILATION_DISABLED,
            Self::UnsupportedExecAffinity => ffi::CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY,
            Self::UnsupportedDevsideSync => ffi::CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC,
            Self::InvalidSource => ffi::CUDA_ERROR_INVALID_SOURCE,
            Self::FileNotFound => ffi::CUDA_ERROR_FILE_NOT_FOUND,
            Self::SharedObjectSymbolNotFound => ffi::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
            Self::SharedObjectInitFailed => ffi::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
            Self::OperatingSystem => ffi::CUDA_ERROR_OPERATING_SYSTEM,
            Self::InvalidHandle => ffi::CUDA_ERROR_INVALID_HANDLE,
            Self::IllegalState => ffi::CUDA_ERROR_ILLEGAL_STATE,
            Self::LossyQuery => ffi::CUDA_ERROR_LOSSY_QUERY,
            Self::NotFound => ffi::CUDA_ERROR_NOT_FOUND,
            Self::NotReady => ffi::CUDA_ERROR_NOT_READY,
            Self::IllegalAddress => ffi::CUDA_ERROR_ILLEGAL_ADDRESS,
            Self::LaunchOutOfResources => ffi::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
            Self::LaunchTimeout => ffi::CUDA_ERROR_LAUNCH_TIMEOUT,
            Self::LaunchIncompatibleTexturing => ffi::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
            Self::PeerAccessAlreadyEnabled => ffi::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
            Self::PeerAccessNotEnabled => ffi::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
            Self::PrimaryContextActive => ffi::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
            Self::ContextIsDestroyed => ffi::CUDA_ERROR_CONTEXT_IS_DESTROYED,
            Self::Assert => ffi::CUDA_ERROR_ASSERT,
            Self::TooManyPeers => ffi::CUDA_ERROR_TOO_MANY_PEERS,
            Self::HostMemoryAlreadyRegistered => ffi::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
            Self::HostMemoryNotRegistered => ffi::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
            Self::HardwareStackError => ffi::CUDA_ERROR_HARDWARE_STACK_ERROR,
            Self::IllegalInstruction => ffi::CUDA_ERROR_ILLEGAL_INSTRUCTION,
            Self::MisalignedAddress => ffi::CUDA_ERROR_MISALIGNED_ADDRESS,
            Self::InvalidAddressSpace => ffi::CUDA_ERROR_INVALID_ADDRESS_SPACE,
            Self::InvalidPc => ffi::CUDA_ERROR_INVALID_PC,
            Self::LaunchFailed => ffi::CUDA_ERROR_LAUNCH_FAILED,
            Self::CooperativeLaunchTooLarge => ffi::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
            Self::NotPermitted => ffi::CUDA_ERROR_NOT_PERMITTED,
            Self::NotSupported => ffi::CUDA_ERROR_NOT_SUPPORTED,
            Self::SystemNotReady => ffi::CUDA_ERROR_SYSTEM_NOT_READY,
            Self::SystemDriverMismatch => ffi::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
            Self::CompatNotSupportedOnDevice => ffi::CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE,
            Self::MpsConnectionFailed => ffi::CUDA_ERROR_MPS_CONNECTION_FAILED,
            Self::MpsRpcFailure => ffi::CUDA_ERROR_MPS_RPC_FAILURE,
            Self::MpsServerNotReady => ffi::CUDA_ERROR_MPS_SERVER_NOT_READY,
            Self::MpsMaxClientsReached => ffi::CUDA_ERROR_MPS_MAX_CLIENTS_REACHED,
            Self::MpsMaxConnectionsReached => ffi::CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED,
            Self::MpsClientTerminated => ffi::CUDA_ERROR_MPS_CLIENT_TERMINATED,
            Self::CdpNotSupported => ffi::CUDA_ERROR_CDP_NOT_SUPPORTED,
            Self::CdpVersionMismatch => ffi::CUDA_ERROR_CDP_VERSION_MISMATCH,
            Self::StreamCaptureUnsupported => ffi::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
            Self::StreamCaptureInvalidated => ffi::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
            Self::StreamCaptureMerge => ffi::CUDA_ERROR_STREAM_CAPTURE_MERGE,
            Self::StreamCaptureUnmatched => ffi::CUDA_ERROR_STREAM_CAPTURE_UNMATCHED,
            Self::StreamCaptureUnjoined => ffi::CUDA_ERROR_STREAM_CAPTURE_UNJOINED,
            Self::StreamCaptureIsolation => ffi::CUDA_ERROR_STREAM_CAPTURE_ISOLATION,
            Self::StreamCaptureImplicit => ffi::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT,
            Self::CapturedEvent => ffi::CUDA_ERROR_CAPTURED_EVENT,
            Self::StreamCaptureWrongThread => ffi::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD,
            Self::Timeout => ffi::CUDA_ERROR_TIMEOUT,
            Self::GraphExecUpdateFailure => ffi::CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE,
            Self::ExternalDevice => ffi::CUDA_ERROR_EXTERNAL_DEVICE,
            Self::InvalidClusterSize => ffi::CUDA_ERROR_INVALID_CLUSTER_SIZE,
            Self::FunctionNotLoaded => ffi::CUDA_ERROR_FUNCTION_NOT_LOADED,
            Self::InvalidResourceType => ffi::CUDA_ERROR_INVALID_RESOURCE_TYPE,
            Self::InvalidResourceConfiguration => ffi::CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION,
            Self::Unknown(code) => *code,
        }
    }
}

// =========================================================================
// CudaResult type alias
// =========================================================================

/// Convenience result alias used throughout the crate.
pub type CudaResult<T> = Result<T, CudaError>;

// =========================================================================
// check() — convert raw CUresult to CudaResult
// =========================================================================

/// Convert a raw [`CUresult`](ffi::CUresult) into a [`CudaResult<()>`].
///
/// Returns `Ok(())` for `CUDA_SUCCESS` (0), otherwise wraps the code in the
/// corresponding [`CudaError`] variant.
#[inline(always)]
pub fn check(result: u32) -> CudaResult<()> {
    if result == 0 {
        Ok(())
    } else {
        Err(CudaError::from_raw(result))
    }
}

// =========================================================================
// cuda_call! macro
// =========================================================================

/// Invoke a raw CUDA Driver API function and convert the result to [`CudaResult`].
///
/// The expression is evaluated inside an `unsafe` block and its `CUresult`
/// return value is passed through [`check`].
///
/// # Examples
///
/// ```ignore
/// cuda_call!(cuInit(0))?;
/// ```
#[macro_export]
macro_rules! cuda_call {
    ($expr:expr) => {
        $crate::error::check(unsafe { $expr })
    };
}

// =========================================================================
// DriverLoadError
// =========================================================================

/// Errors that can occur while dynamically loading `libcuda.so` / `nvcuda.dll`.
#[derive(Debug, thiserror::Error)]
pub enum DriverLoadError {
    /// The CUDA driver library was not found on the system.
    #[error("CUDA driver library not found (tried: {candidates:?}): {last_error}")]
    LibraryNotFound {
        /// Library names that were attempted.
        candidates: Vec<String>,
        /// OS-level error from the last attempt.
        last_error: String,
    },

    /// The shared library was loaded but a required symbol was missing.
    #[error("failed to load symbol '{symbol}': {reason}")]
    SymbolNotFound {
        /// Name of the missing symbol (e.g. `"cuInit"`).
        symbol: &'static str,
        /// OS-level error description.
        reason: String,
    },

    /// The CUDA driver library loaded and all symbols resolved, but `cuInit(0)`
    /// returned a non-zero error code.
    ///
    /// This typically happens when there is no CUDA-capable GPU installed
    /// (`CUDA_ERROR_NO_DEVICE` = 100) or when the installed driver is
    /// incompatible with the hardware.
    #[error("cuInit(0) failed with CUDA error code {code}")]
    InitializationFailed {
        /// Raw CUDA error code returned by `cuInit`.
        code: u32,
    },

    /// CUDA is not supported on this operating system (e.g. macOS).
    #[error("CUDA driver not supported on this platform")]
    UnsupportedPlatform,
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_success() {
        assert!(check(0).is_ok());
    }

    #[test]
    fn test_check_error() {
        let result = check(1);
        assert!(result.is_err());
        assert_eq!(result.err(), Some(CudaError::InvalidValue));
    }

    #[test]
    fn test_check_out_of_memory() {
        let result = check(2);
        assert_eq!(result.err(), Some(CudaError::OutOfMemory));
    }

    #[test]
    fn test_check_not_initialized() {
        let result = check(3);
        assert_eq!(result.err(), Some(CudaError::NotInitialized));
    }

    #[test]
    fn test_from_raw_roundtrip() {
        // Test a representative sample of error codes round-trip correctly
        let codes: &[u32] = &[
            1, 2, 3, 4, 5, 6, 7, 8, 34, 46, 100, 101, 102, 200, 201, 202, 205, 206, 207, 208, 209,
            210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 300,
            301, 302, 303, 304, 400, 401, 402, 500, 600, 700, 701, 702, 703, 704, 705, 708, 709,
            710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 800, 801, 802, 803, 804, 805,
            806, 807, 808, 809, 810, 811, 812, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909,
            910, 911, 912, 913, 914, 915, 999,
        ];
        for &code in codes {
            let err = CudaError::from_raw(code);
            assert_eq!(
                err.as_raw(),
                code,
                "round-trip failed for code {code}: got {:?}",
                err
            );
        }
    }

    #[test]
    fn test_unknown_code() {
        let err = CudaError::from_raw(12345);
        assert_eq!(err, CudaError::Unknown(12345));
        assert_eq!(err.as_raw(), 12345);
    }

    #[test]
    fn test_unknown_code_display() {
        let err = CudaError::from_raw(12345);
        let msg = format!("{err}");
        assert!(
            msg.contains("12345"),
            "display should contain the code: {msg}"
        );
    }

    #[test]
    fn test_error_display_messages() {
        assert_eq!(
            format!("{}", CudaError::InvalidValue),
            "CUDA: invalid value"
        );
        assert_eq!(
            format!("{}", CudaError::OutOfMemory),
            "CUDA: out of device memory"
        );
        assert_eq!(
            format!("{}", CudaError::LaunchFailed),
            "CUDA: kernel launch failed"
        );
        assert_eq!(
            format!("{}", CudaError::NotReady),
            "CUDA: not ready (async operation pending)"
        );
    }

    #[test]
    fn test_error_is_copy() {
        let err = CudaError::InvalidValue;
        let copy = err;
        assert_eq!(err, copy);
    }

    #[test]
    fn test_error_implements_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<CudaError>();
        assert_error::<DriverLoadError>();
    }

    #[test]
    fn test_driver_load_error_display() {
        let err = DriverLoadError::LibraryNotFound {
            candidates: vec!["libcuda.so".to_string()],
            last_error: "no such file or directory".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("libcuda.so"));

        let err = DriverLoadError::SymbolNotFound {
            symbol: "cuInit",
            reason: "not found".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("cuInit"));

        let err = DriverLoadError::UnsupportedPlatform;
        let msg = format!("{err}");
        assert!(msg.contains("not supported"));
    }

    #[test]
    fn test_cuda_call_macro() {
        // Simulate a successful call via a function that returns CUresult
        unsafe fn fake_success() -> u32 {
            0
        }
        unsafe fn fake_launch_failed() -> u32 {
            719
        }

        let result = cuda_call!(fake_success());
        assert!(result.is_ok());

        let result = cuda_call!(fake_launch_failed());
        assert_eq!(result.err(), Some(CudaError::LaunchFailed));
    }

    #[test]
    fn test_error_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CudaError::InvalidValue);
        set.insert(CudaError::OutOfMemory);
        set.insert(CudaError::InvalidValue); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_cuda_result_type_alias() {
        fn returns_ok() -> CudaResult<i32> {
            Ok(42)
        }
        fn returns_err() -> CudaResult<i32> {
            Err(CudaError::NoDevice)
        }
        assert_eq!(returns_ok().ok(), Some(42));
        assert_eq!(returns_err().err(), Some(CudaError::NoDevice));
    }

    // =========================================================================
    // Error injection test suite
    // =========================================================================

    /// Verifies every commonly-used CudaError variant has a non-empty Display.
    #[test]
    fn test_all_common_cuda_error_variants_have_display() {
        let errors: &[CudaError] = &[
            // Basic errors (1-8)
            CudaError::InvalidValue,
            CudaError::OutOfMemory,
            CudaError::NotInitialized,
            CudaError::Deinitialized,
            CudaError::ProfilerDisabled,
            CudaError::ProfilerNotInitialized,
            CudaError::ProfilerAlreadyStarted,
            CudaError::ProfilerAlreadyStopped,
            // Stub / unavailable
            CudaError::StubLibrary,
            CudaError::DeviceUnavailable,
            // Device errors
            CudaError::NoDevice,
            CudaError::InvalidDevice,
            CudaError::DeviceNotLicensed,
            // Image / context errors
            CudaError::InvalidImage,
            CudaError::InvalidContext,
            CudaError::ContextAlreadyCurrent,
            CudaError::MapFailed,
            CudaError::UnmapFailed,
            CudaError::EccUncorrectable,
            // Source / file errors
            CudaError::InvalidSource,
            CudaError::FileNotFound,
            CudaError::SharedObjectSymbolNotFound,
            // Handle errors
            CudaError::InvalidHandle,
            CudaError::IllegalState,
            CudaError::NotFound,
            // Readiness
            CudaError::NotReady,
            // Launch / address errors
            CudaError::IllegalAddress,
            CudaError::LaunchOutOfResources,
            CudaError::LaunchTimeout,
            CudaError::LaunchFailed,
            CudaError::CooperativeLaunchTooLarge,
            // Permission errors
            CudaError::NotPermitted,
            CudaError::NotSupported,
            // Hardware faults
            CudaError::HardwareStackError,
            CudaError::IllegalInstruction,
            CudaError::MisalignedAddress,
            CudaError::InvalidAddressSpace,
            CudaError::InvalidPc,
            // Peer access
            CudaError::PeerAccessAlreadyEnabled,
            CudaError::PeerAccessNotEnabled,
            CudaError::ContextIsDestroyed,
            // Stream capture
            CudaError::StreamCaptureUnsupported,
            CudaError::StreamCaptureInvalidated,
            CudaError::InvalidClusterSize,
            CudaError::FunctionNotLoaded,
            // Catch-all
            CudaError::Unknown(99999),
        ];
        for err in errors {
            let display = format!("{err}");
            assert!(
                !display.is_empty(),
                "CudaError::{err:?} has an empty Display string"
            );
        }
    }

    /// Verifies the code→error→code round-trip for the most critical codes.
    #[test]
    fn test_cuda_error_from_result_code() {
        let pairs: &[(u32, CudaError)] = &[
            (1, CudaError::InvalidValue),
            (2, CudaError::OutOfMemory),
            (3, CudaError::NotInitialized),
            (4, CudaError::Deinitialized),
            (100, CudaError::NoDevice),
            (101, CudaError::InvalidDevice),
            (200, CudaError::InvalidImage),
            (201, CudaError::InvalidContext),
            (400, CudaError::InvalidHandle),
            (500, CudaError::NotFound),
            (600, CudaError::NotReady),
            (700, CudaError::IllegalAddress),
            (701, CudaError::LaunchOutOfResources),
            (702, CudaError::LaunchTimeout),
            (719, CudaError::LaunchFailed),
            (800, CudaError::NotPermitted),
            (801, CudaError::NotSupported),
            (912, CudaError::InvalidClusterSize),
            (999, CudaError::Unknown(999)),
        ];
        for &(code, ref expected) in pairs {
            let got = CudaError::from_raw(code);
            assert_eq!(
                &got, expected,
                "from_raw({code}) should be {expected:?}, got {got:?}"
            );
            assert_eq!(
                got.as_raw(),
                code,
                "as_raw() round-trip failed for code {code}"
            );
        }
    }

    /// Verifies the `is_fatal` classification for a representative sample.
    #[test]
    fn test_cuda_error_is_fatal() {
        // Fatal — hardware fault / context corruption.
        let fatal: &[CudaError] = &[
            CudaError::IllegalAddress,
            CudaError::LaunchFailed,
            CudaError::HardwareStackError,
            CudaError::IllegalInstruction,
            CudaError::MisalignedAddress,
            CudaError::InvalidAddressSpace,
            CudaError::InvalidPc,
            CudaError::Assert,
            CudaError::EccUncorrectable,
            CudaError::NvlinkUncorrectable,
            CudaError::ContextIsDestroyed,
            CudaError::Deinitialized,
        ];
        for err in fatal {
            assert!(
                err.is_fatal(),
                "CudaError::{err:?} should be classified as fatal"
            );
        }

        // Non-fatal — recoverable errors or usage mistakes.
        let non_fatal: &[CudaError] = &[
            CudaError::InvalidValue,
            CudaError::OutOfMemory,
            CudaError::NotInitialized,
            CudaError::NoDevice,
            CudaError::NotReady,
            CudaError::LaunchOutOfResources,
            CudaError::LaunchTimeout,
            CudaError::NotPermitted,
            CudaError::NotSupported,
            CudaError::InvalidClusterSize,
            CudaError::Unknown(99),
        ];
        for err in non_fatal {
            assert!(
                !err.is_fatal(),
                "CudaError::{err:?} should NOT be classified as fatal"
            );
        }
    }

    /// Verifies the `is_usage_error` classification.
    #[test]
    fn test_cuda_error_is_usage_error() {
        let usage: &[CudaError] = &[
            CudaError::InvalidValue,
            CudaError::InvalidDevice,
            CudaError::InvalidContext,
            CudaError::InvalidHandle,
            CudaError::NoDevice,
        ];
        for err in usage {
            assert!(
                err.is_usage_error(),
                "CudaError::{err:?} should be a usage error"
            );
        }

        let non_usage: &[CudaError] = &[
            CudaError::OutOfMemory,
            CudaError::LaunchFailed,
            CudaError::IllegalAddress,
            CudaError::NotReady,
        ];
        for err in non_usage {
            assert!(
                !err.is_usage_error(),
                "CudaError::{err:?} should NOT be a usage error"
            );
        }
    }

    /// Verifies that `is_fatal` and `is_usage_error` are mutually exclusive.
    #[test]
    fn test_fatal_and_usage_error_are_disjoint() {
        // A sample of errors that should not be both fatal AND usage errors.
        let all_errors: &[CudaError] = &[
            CudaError::InvalidValue,
            CudaError::OutOfMemory,
            CudaError::LaunchFailed,
            CudaError::IllegalAddress,
            CudaError::HardwareStackError,
            CudaError::NotReady,
            CudaError::InvalidDevice,
        ];
        for err in all_errors {
            assert!(
                !(err.is_fatal() && err.is_usage_error()),
                "CudaError::{err:?} cannot be both fatal and a usage error"
            );
        }
    }

    /// Simulates error injection: build a Result from a raw error code and
    /// verify downstream handling branches on the correct classification.
    #[test]
    fn test_error_injection_simulation() {
        // Inject CUDA_ERROR_ILLEGAL_ADDRESS (700) — should trigger fatal handling.
        let injected = check(700);
        let err = injected.expect_err("error code 700 must be an Err");
        assert_eq!(err, CudaError::IllegalAddress);
        assert!(err.is_fatal(), "IllegalAddress must be fatal");
        assert!(!err.is_usage_error());

        // Inject CUDA_ERROR_INVALID_VALUE (1) — should trigger usage-error handling.
        let injected = check(1);
        let err = injected.expect_err("error code 1 must be an Err");
        assert_eq!(err, CudaError::InvalidValue);
        assert!(!err.is_fatal(), "InvalidValue must not be fatal");
        assert!(err.is_usage_error());

        // Inject unknown code — should propagate as Unknown.
        let injected = check(55555);
        let err = injected.expect_err("unknown code must be an Err");
        assert!(matches!(err, CudaError::Unknown(55555)));
        assert!(!err.is_fatal());
        assert!(!err.is_usage_error());
    }
}
