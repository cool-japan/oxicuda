//! CUDA Runtime API error types.
//!
//! The Runtime API uses `cudaError_t` (a C enum) as its primary error code.
//! This module maps all standard `cudaError_t` values to [`CudaRtError`] and
//! provides [`CudaRtResult`] for ergonomic `?` propagation.

use thiserror::Error;

// =========================================================================
// CudaRtError
// =========================================================================

/// CUDA Runtime API error.
///
/// Each variant corresponds to a distinct `cudaError_t` value from the
/// CUDA Runtime specification. The integer discriminant matches the canonical
/// `cudaError_t` value so that raw codes received through FFI can be
/// converted cheaply via [`from_code`](CudaRtError::from_code).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Error)]
#[repr(u32)]
pub enum CudaRtError {
    /// `cudaErrorInvalidValue` (1) — an invalid value was supplied.
    #[error("CUDA RT: invalid value")]
    InvalidValue = 1,

    /// `cudaErrorMemoryAllocation` (2) — device memory allocation failed.
    #[error("CUDA RT: out of device memory")]
    MemoryAllocation = 2,

    /// `cudaErrorInitializationError` (3) — CUDA initialization failed.
    #[error("CUDA RT: initialization error")]
    InitializationError = 3,

    /// `cudaErrorCudartUnloading` (4) — CUDA runtime is unloading.
    #[error("CUDA RT: CUDA runtime is unloading")]
    CudartUnloading = 4,

    /// `cudaErrorProfilerDisabled` (5) — profiler was disabled.
    #[error("CUDA RT: profiler disabled")]
    ProfilerDisabled = 5,

    /// `cudaErrorInvalidConfiguration` (9) — invalid kernel launch configuration.
    #[error("CUDA RT: invalid launch configuration")]
    InvalidConfiguration = 9,

    /// `cudaErrorInvalidPitchValue` (12)
    #[error("CUDA RT: invalid pitch value")]
    InvalidPitchValue = 12,

    /// `cudaErrorInvalidSymbol` (13)
    #[error("CUDA RT: invalid symbol")]
    InvalidSymbol = 13,

    /// `cudaErrorInvalidHostPointer` (16)
    #[error("CUDA RT: invalid host pointer")]
    InvalidHostPointer = 16,

    /// `cudaErrorInvalidDevicePointer` (17)
    #[error("CUDA RT: invalid device pointer")]
    InvalidDevicePointer = 17,

    /// `cudaErrorInvalidTexture` (18)
    #[error("CUDA RT: invalid texture")]
    InvalidTexture = 18,

    /// `cudaErrorInvalidTextureBinding` (19)
    #[error("CUDA RT: invalid texture binding")]
    InvalidTextureBinding = 19,

    /// `cudaErrorInvalidChannelDescriptor` (20)
    #[error("CUDA RT: invalid channel descriptor")]
    InvalidChannelDescriptor = 20,

    /// `cudaErrorInvalidMemcpyDirection` (21)
    #[error("CUDA RT: invalid memcpy direction")]
    InvalidMemcpyDirection = 21,

    /// `cudaErrorInvalidFilterSetting` (26)
    #[error("CUDA RT: invalid filter setting")]
    InvalidFilterSetting = 26,

    /// `cudaErrorInvalidNormSetting` (27)
    #[error("CUDA RT: invalid norm setting")]
    InvalidNormSetting = 27,

    /// `cudaErrorStubLibrary` (34) — stub library loaded instead of real driver.
    #[error("CUDA RT: stub library")]
    StubLibrary = 34,

    /// `cudaErrorInsufficientDriver` (35) — installed driver is too old.
    #[error("CUDA RT: installed driver version is insufficient")]
    InsufficientDriver = 35,

    /// `cudaErrorCallRequiresNewerDriver` (36)
    #[error("CUDA RT: call requires a newer driver version")]
    CallRequiresNewerDriver = 36,

    /// `cudaErrorInvalidSurface` (37)
    #[error("CUDA RT: invalid surface")]
    InvalidSurface = 37,

    /// `cudaErrorDuplicateVariableName` (43)
    #[error("CUDA RT: duplicate variable name")]
    DuplicateVariableName = 43,

    /// `cudaErrorDuplicateTextureName` (44)
    #[error("CUDA RT: duplicate texture name")]
    DuplicateTextureName = 44,

    /// `cudaErrorDuplicateSurfaceName` (45)
    #[error("CUDA RT: duplicate surface name")]
    DuplicateSurfaceName = 45,

    /// `cudaErrorDevicesUnavailable` (46)
    #[error("CUDA RT: all CUDA-capable devices are busy or unavailable")]
    DevicesUnavailable = 46,

    /// `cudaErrorIncompatibleDriverContext` (49)
    #[error("CUDA RT: incompatible driver context")]
    IncompatibleDriverContext = 49,

    /// `cudaErrorMissingConfiguration` (52)
    #[error("CUDA RT: missing configuration")]
    MissingConfiguration = 52,

    /// `cudaErrorLaunchMaxDepthExceeded` (65)
    #[error("CUDA RT: launch max depth exceeded (dynamic parallelism)")]
    LaunchMaxDepthExceeded = 65,

    /// `cudaErrorLaunchFileScopedTex` (66)
    #[error("CUDA RT: attempted launch of file-scoped texture")]
    LaunchFileScopedTex = 66,

    /// `cudaErrorLaunchFileScopedSurf` (67)
    #[error("CUDA RT: attempted launch of file-scoped surface")]
    LaunchFileScopedSurf = 67,

    /// `cudaErrorSyncDepthExceeded` (68)
    #[error("CUDA RT: sync depth exceeded")]
    SyncDepthExceeded = 68,

    /// `cudaErrorLaunchPendingCountExceeded` (69)
    #[error("CUDA RT: launch pending count exceeded")]
    LaunchPendingCountExceeded = 69,

    /// `cudaErrorInvalidDeviceFunction` (98)
    #[error("CUDA RT: invalid device function")]
    InvalidDeviceFunction = 98,

    /// `cudaErrorNoDevice` (100) — no CUDA-capable device is detected.
    #[error("CUDA RT: no CUDA-capable device found")]
    NoDevice = 100,

    /// `cudaErrorInvalidDevice` (101) — the specified device is invalid.
    #[error("CUDA RT: invalid device ordinal")]
    InvalidDevice = 101,

    /// `cudaErrorDeviceNotLicensed` (102)
    #[error("CUDA RT: device not licensed for compute")]
    DeviceNotLicensed = 102,

    /// `cudaErrorSoftwareValidityNotEstablished` (103)
    #[error("CUDA RT: software validity not established")]
    SoftwareValidityNotEstablished = 103,

    /// `cudaErrorStartupFailure` (127)
    #[error("CUDA RT: startup failure")]
    StartupFailure = 127,

    /// `cudaErrorInvalidKernelImage` (200)
    #[error("CUDA RT: invalid kernel image")]
    InvalidKernelImage = 200,

    /// `cudaErrorDeviceUninitialized` (201)
    #[error("CUDA RT: device uninitialized")]
    DeviceUninitialized = 201,

    /// `cudaErrorMapBufferObjectFailed` (205)
    #[error("CUDA RT: map buffer object failed")]
    MapBufferObjectFailed = 205,

    /// `cudaErrorUnmapBufferObjectFailed` (206)
    #[error("CUDA RT: unmap buffer object failed")]
    UnmapBufferObjectFailed = 206,

    /// `cudaErrorArrayIsMapped` (207)
    #[error("CUDA RT: array is mapped")]
    ArrayIsMapped = 207,

    /// `cudaErrorAlreadyMapped` (208)
    #[error("CUDA RT: already mapped")]
    AlreadyMapped = 208,

    /// `cudaErrorNoKernelImageForDevice` (209)
    #[error("CUDA RT: no kernel image for this device")]
    NoKernelImageForDevice = 209,

    /// `cudaErrorAlreadyAcquired` (210)
    #[error("CUDA RT: already acquired")]
    AlreadyAcquired = 210,

    /// `cudaErrorNotMapped` (211)
    #[error("CUDA RT: not mapped")]
    NotMapped = 211,

    /// `cudaErrorNotMappedAsArray` (212)
    #[error("CUDA RT: not mapped as array")]
    NotMappedAsArray = 212,

    /// `cudaErrorNotMappedAsPointer` (213)
    #[error("CUDA RT: not mapped as pointer")]
    NotMappedAsPointer = 213,

    /// `cudaErrorECCUncorrectable` (214)
    #[error("CUDA RT: uncorrectable ECC error")]
    EccUncorrectable = 214,

    /// `cudaErrorUnsupportedLimit` (215)
    #[error("CUDA RT: unsupported limit")]
    UnsupportedLimit = 215,

    /// `cudaErrorDeviceAlreadyInUse` (216)
    #[error("CUDA RT: device already in use")]
    DeviceAlreadyInUse = 216,

    /// `cudaErrorPeerAccessUnsupported` (217)
    #[error("CUDA RT: peer access unsupported")]
    PeerAccessUnsupported = 217,

    /// `cudaErrorInvalidPtx` (218)
    #[error("CUDA RT: invalid PTX")]
    InvalidPtx = 218,

    /// `cudaErrorInvalidGraphicsContext` (219)
    #[error("CUDA RT: invalid graphics context")]
    InvalidGraphicsContext = 219,

    /// `cudaErrorNvlinkUncorrectable` (220)
    #[error("CUDA RT: NVLink uncorrectable error")]
    NvlinkUncorrectable = 220,

    /// `cudaErrorJitCompilerNotFound` (221)
    #[error("CUDA RT: JIT compiler not found")]
    JitCompilerNotFound = 221,

    /// `cudaErrorUnsupportedPtxVersion` (222)
    #[error("CUDA RT: unsupported PTX version")]
    UnsupportedPtxVersion = 222,

    /// `cudaErrorJitCompilationDisabled` (223)
    #[error("CUDA RT: JIT compilation disabled")]
    JitCompilationDisabled = 223,

    /// `cudaErrorUnsupportedExecAffinity` (224)
    #[error("CUDA RT: unsupported exec affinity")]
    UnsupportedExecAffinity = 224,

    /// `cudaErrorUnsupportedDevSideSync` (225)
    #[error("CUDA RT: unsupported device-side sync")]
    UnsupportedDevSideSync = 225,

    /// `cudaErrorInvalidSource` (300)
    #[error("CUDA RT: invalid source")]
    InvalidSource = 300,

    /// `cudaErrorFileNotFound` (301)
    #[error("CUDA RT: file not found")]
    FileNotFound = 301,

    /// `cudaErrorSharedObjectSymbolNotFound` (302)
    #[error("CUDA RT: shared object symbol not found")]
    SharedObjectSymbolNotFound = 302,

    /// `cudaErrorSharedObjectInitFailed` (303)
    #[error("CUDA RT: shared object init failed")]
    SharedObjectInitFailed = 303,

    /// `cudaErrorOperatingSystem` (304)
    #[error("CUDA RT: operating system error")]
    OperatingSystem = 304,

    /// `cudaErrorInvalidResourceHandle` (400)
    #[error("CUDA RT: invalid resource handle")]
    InvalidResourceHandle = 400,

    /// `cudaErrorIllegalState` (401)
    #[error("CUDA RT: illegal state")]
    IllegalState = 401,

    /// `cudaErrorLostConnection` (402)
    #[error("CUDA RT: lost connection to client")]
    LostConnection = 402,

    /// `cudaErrorSymbolNotFound` (500)
    #[error("CUDA RT: named symbol not found")]
    SymbolNotFound = 500,

    /// `cudaErrorNotReady` (600) — async operation not yet complete.
    #[error("CUDA RT: operation not ready")]
    NotReady = 600,

    /// `cudaErrorIllegalAddress` (700)
    #[error("CUDA RT: illegal memory access")]
    IllegalAddress = 700,

    /// `cudaErrorLaunchOutOfResources` (701)
    #[error("CUDA RT: launch out of resources")]
    LaunchOutOfResources = 701,

    /// `cudaErrorLaunchTimeout` (702)
    #[error("CUDA RT: launch timed out")]
    LaunchTimeout = 702,

    /// `cudaErrorLaunchIncompatibleTexturing` (703)
    #[error("CUDA RT: incompatible texturing mode")]
    LaunchIncompatibleTexturing = 703,

    /// `cudaErrorPeerAccessAlreadyEnabled` (704)
    #[error("CUDA RT: peer access already enabled")]
    PeerAccessAlreadyEnabled = 704,

    /// `cudaErrorPeerAccessNotEnabled` (705)
    #[error("CUDA RT: peer access not enabled")]
    PeerAccessNotEnabled = 705,

    /// `cudaErrorSetOnActiveProcess` (708)
    #[error("CUDA RT: cannot set while active process")]
    SetOnActiveProcess = 708,

    /// `cudaErrorContextIsDestroyed` (709)
    #[error("CUDA RT: context is destroyed")]
    ContextIsDestroyed = 709,

    /// `cudaErrorAssert` (710) — device-side assertion triggered.
    #[error("CUDA RT: device assertion triggered")]
    Assert = 710,

    /// `cudaErrorTooManyPeers` (711)
    #[error("CUDA RT: too many peer mappings")]
    TooManyPeers = 711,

    /// `cudaErrorHostMemoryAlreadyRegistered` (712)
    #[error("CUDA RT: host memory already registered")]
    HostMemoryAlreadyRegistered = 712,

    /// `cudaErrorHostMemoryNotRegistered` (713)
    #[error("CUDA RT: host memory not registered")]
    HostMemoryNotRegistered = 713,

    /// `cudaErrorHardwareStackError` (714)
    #[error("CUDA RT: hardware stack error")]
    HardwareStackError = 714,

    /// `cudaErrorIllegalInstruction` (715)
    #[error("CUDA RT: illegal instruction")]
    IllegalInstruction = 715,

    /// `cudaErrorMisalignedAddress` (716)
    #[error("CUDA RT: misaligned address")]
    MisalignedAddress = 716,

    /// `cudaErrorInvalidAddressSpace` (717)
    #[error("CUDA RT: invalid address space")]
    InvalidAddressSpace = 717,

    /// `cudaErrorInvalidPc` (718)
    #[error("CUDA RT: invalid program counter")]
    InvalidPc = 718,

    /// `cudaErrorLaunchFailure` (719) — exception during kernel execution.
    #[error("CUDA RT: exception during kernel launch")]
    LaunchFailure = 719,

    /// `cudaErrorCooperativeLaunchTooLarge` (720)
    #[error("CUDA RT: cooperative launch too large")]
    CooperativeLaunchTooLarge = 720,

    /// `cudaErrorNotPermitted` (800)
    #[error("CUDA RT: not permitted")]
    NotPermitted = 800,

    /// `cudaErrorNotSupported` (801)
    #[error("CUDA RT: not supported")]
    NotSupported = 801,

    /// `cudaErrorSystemNotReady` (802)
    #[error("CUDA RT: system not ready")]
    SystemNotReady = 802,

    /// `cudaErrorSystemDriverMismatch` (803)
    #[error("CUDA RT: system driver mismatch")]
    SystemDriverMismatch = 803,

    /// `cudaErrorCompatNotSupportedOnDevice` (804)
    #[error("CUDA RT: compat mode not supported on device")]
    CompatNotSupportedOnDevice = 804,

    /// `cudaErrorMpsConnectionFailed` (805)
    #[error("CUDA RT: MPS connection failed")]
    MpsConnectionFailed = 805,

    /// `cudaErrorMpsRpcFailure` (806)
    #[error("CUDA RT: MPS RPC failure")]
    MpsRpcFailure = 806,

    /// `cudaErrorMpsServerNotReady` (807)
    #[error("CUDA RT: MPS server not ready")]
    MpsServerNotReady = 807,

    /// `cudaErrorMpsMaxClientsReached` (808)
    #[error("CUDA RT: MPS max clients reached")]
    MpsMaxClientsReached = 808,

    /// `cudaErrorMpsMaxConnectionsReached` (809)
    #[error("CUDA RT: MPS max connections reached")]
    MpsMaxConnectionsReached = 809,

    /// `cudaErrorMpsClientTerminated` (810)
    #[error("CUDA RT: MPS client terminated")]
    MpsClientTerminated = 810,

    /// `cudaErrorCdpNotSupported` (811)
    #[error("CUDA RT: CDP not supported")]
    CdpNotSupported = 811,

    /// `cudaErrorCdpVersionMismatch` (812)
    #[error("CUDA RT: CDP version mismatch")]
    CdpVersionMismatch = 812,

    /// `cudaErrorStreamCaptureUnsupported` (900)
    #[error("CUDA RT: stream capture unsupported")]
    StreamCaptureUnsupported = 900,

    /// `cudaErrorStreamCaptureInvalidated` (901)
    #[error("CUDA RT: stream capture invalidated")]
    StreamCaptureInvalidated = 901,

    /// `cudaErrorStreamCaptureMerge` (902)
    #[error("CUDA RT: stream capture merge failed")]
    StreamCaptureMerge = 902,

    /// `cudaErrorStreamCaptureUnmatched` (903)
    #[error("CUDA RT: stream capture unmatched")]
    StreamCaptureUnmatched = 903,

    /// `cudaErrorStreamCaptureUnjoined` (904)
    #[error("CUDA RT: stream capture unjoined")]
    StreamCaptureUnjoined = 904,

    /// `cudaErrorStreamCaptureIsolation` (905)
    #[error("CUDA RT: stream capture isolation")]
    StreamCaptureIsolation = 905,

    /// `cudaErrorStreamCaptureImplicit` (906)
    #[error("CUDA RT: stream capture implicit")]
    StreamCaptureImplicit = 906,

    /// `cudaErrorCapturedEvent` (907)
    #[error("CUDA RT: captured event")]
    CapturedEvent = 907,

    /// `cudaErrorStreamCaptureWrongThread` (908)
    #[error("CUDA RT: stream capture wrong thread")]
    StreamCaptureWrongThread = 908,

    /// `cudaErrorTimeout` (909)
    #[error("CUDA RT: timeout")]
    Timeout = 909,

    /// `cudaErrorGraphExecUpdateFailure` (910)
    #[error("CUDA RT: CUDA graph exec update failure")]
    GraphExecUpdateFailure = 910,

    /// `cudaErrorExternalDevice` (911)
    #[error("CUDA RT: external device error")]
    ExternalDevice = 911,

    /// `cudaErrorInvalidClusterSize` (912)
    #[error("CUDA RT: invalid cluster size")]
    InvalidClusterSize = 912,

    /// `cudaErrorFunctionNotLoaded` (913)
    #[error("CUDA RT: function not loaded")]
    FunctionNotLoaded = 913,

    /// `cudaErrorInvalidResourceType` (914)
    #[error("CUDA RT: invalid resource type")]
    InvalidResourceType = 914,

    /// `cudaErrorInvalidResourceConfiguration` (915)
    #[error("CUDA RT: invalid resource configuration")]
    InvalidResourceConfiguration = 915,

    /// `cudaErrorUnknown` (999) — unknown error.
    #[error("CUDA RT: unknown error")]
    Unknown = 999,

    /// Driver could not be loaded at runtime (libcuda.so not found).
    #[error("CUDA RT: CUDA driver not available")]
    DriverNotAvailable,

    /// No GPU available (driver loaded but 0 CUDA devices).
    #[error("CUDA RT: no GPU device available")]
    NoGpu,

    /// Runtime context not initialised (call cudaSetDevice first).
    #[error("CUDA RT: device not set — call cudaSetDevice before using runtime API")]
    DeviceNotSet,
}

/// Convenience alias for `Result<T, CudaRtError>`.
pub type CudaRtResult<T> = Result<T, CudaRtError>;

impl CudaRtError {
    /// Convert a raw `cudaError_t` integer to [`CudaRtError`].
    ///
    /// Returns `Ok(())` for `cudaSuccess` (0) and maps all other values.
    #[must_use]
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => None, // cudaSuccess
            1 => Some(Self::InvalidValue),
            2 => Some(Self::MemoryAllocation),
            3 => Some(Self::InitializationError),
            4 => Some(Self::CudartUnloading),
            5 => Some(Self::ProfilerDisabled),
            9 => Some(Self::InvalidConfiguration),
            12 => Some(Self::InvalidPitchValue),
            13 => Some(Self::InvalidSymbol),
            16 => Some(Self::InvalidHostPointer),
            17 => Some(Self::InvalidDevicePointer),
            18 => Some(Self::InvalidTexture),
            19 => Some(Self::InvalidTextureBinding),
            20 => Some(Self::InvalidChannelDescriptor),
            21 => Some(Self::InvalidMemcpyDirection),
            26 => Some(Self::InvalidFilterSetting),
            27 => Some(Self::InvalidNormSetting),
            34 => Some(Self::StubLibrary),
            35 => Some(Self::InsufficientDriver),
            36 => Some(Self::CallRequiresNewerDriver),
            37 => Some(Self::InvalidSurface),
            43 => Some(Self::DuplicateVariableName),
            44 => Some(Self::DuplicateTextureName),
            45 => Some(Self::DuplicateSurfaceName),
            46 => Some(Self::DevicesUnavailable),
            49 => Some(Self::IncompatibleDriverContext),
            52 => Some(Self::MissingConfiguration),
            65 => Some(Self::LaunchMaxDepthExceeded),
            66 => Some(Self::LaunchFileScopedTex),
            67 => Some(Self::LaunchFileScopedSurf),
            68 => Some(Self::SyncDepthExceeded),
            69 => Some(Self::LaunchPendingCountExceeded),
            98 => Some(Self::InvalidDeviceFunction),
            100 => Some(Self::NoDevice),
            101 => Some(Self::InvalidDevice),
            102 => Some(Self::DeviceNotLicensed),
            103 => Some(Self::SoftwareValidityNotEstablished),
            127 => Some(Self::StartupFailure),
            200 => Some(Self::InvalidKernelImage),
            201 => Some(Self::DeviceUninitialized),
            205 => Some(Self::MapBufferObjectFailed),
            206 => Some(Self::UnmapBufferObjectFailed),
            207 => Some(Self::ArrayIsMapped),
            208 => Some(Self::AlreadyMapped),
            209 => Some(Self::NoKernelImageForDevice),
            210 => Some(Self::AlreadyAcquired),
            211 => Some(Self::NotMapped),
            212 => Some(Self::NotMappedAsArray),
            213 => Some(Self::NotMappedAsPointer),
            214 => Some(Self::EccUncorrectable),
            215 => Some(Self::UnsupportedLimit),
            216 => Some(Self::DeviceAlreadyInUse),
            217 => Some(Self::PeerAccessUnsupported),
            218 => Some(Self::InvalidPtx),
            219 => Some(Self::InvalidGraphicsContext),
            220 => Some(Self::NvlinkUncorrectable),
            221 => Some(Self::JitCompilerNotFound),
            222 => Some(Self::UnsupportedPtxVersion),
            223 => Some(Self::JitCompilationDisabled),
            224 => Some(Self::UnsupportedExecAffinity),
            225 => Some(Self::UnsupportedDevSideSync),
            300 => Some(Self::InvalidSource),
            301 => Some(Self::FileNotFound),
            302 => Some(Self::SharedObjectSymbolNotFound),
            303 => Some(Self::SharedObjectInitFailed),
            304 => Some(Self::OperatingSystem),
            400 => Some(Self::InvalidResourceHandle),
            401 => Some(Self::IllegalState),
            402 => Some(Self::LostConnection),
            500 => Some(Self::SymbolNotFound),
            600 => Some(Self::NotReady),
            700 => Some(Self::IllegalAddress),
            701 => Some(Self::LaunchOutOfResources),
            702 => Some(Self::LaunchTimeout),
            703 => Some(Self::LaunchIncompatibleTexturing),
            704 => Some(Self::PeerAccessAlreadyEnabled),
            705 => Some(Self::PeerAccessNotEnabled),
            708 => Some(Self::SetOnActiveProcess),
            709 => Some(Self::ContextIsDestroyed),
            710 => Some(Self::Assert),
            711 => Some(Self::TooManyPeers),
            712 => Some(Self::HostMemoryAlreadyRegistered),
            713 => Some(Self::HostMemoryNotRegistered),
            714 => Some(Self::HardwareStackError),
            715 => Some(Self::IllegalInstruction),
            716 => Some(Self::MisalignedAddress),
            717 => Some(Self::InvalidAddressSpace),
            718 => Some(Self::InvalidPc),
            719 => Some(Self::LaunchFailure),
            720 => Some(Self::CooperativeLaunchTooLarge),
            800 => Some(Self::NotPermitted),
            801 => Some(Self::NotSupported),
            802 => Some(Self::SystemNotReady),
            803 => Some(Self::SystemDriverMismatch),
            804 => Some(Self::CompatNotSupportedOnDevice),
            805 => Some(Self::MpsConnectionFailed),
            806 => Some(Self::MpsRpcFailure),
            807 => Some(Self::MpsServerNotReady),
            808 => Some(Self::MpsMaxClientsReached),
            809 => Some(Self::MpsMaxConnectionsReached),
            810 => Some(Self::MpsClientTerminated),
            811 => Some(Self::CdpNotSupported),
            812 => Some(Self::CdpVersionMismatch),
            900 => Some(Self::StreamCaptureUnsupported),
            901 => Some(Self::StreamCaptureInvalidated),
            902 => Some(Self::StreamCaptureMerge),
            903 => Some(Self::StreamCaptureUnmatched),
            904 => Some(Self::StreamCaptureUnjoined),
            905 => Some(Self::StreamCaptureIsolation),
            906 => Some(Self::StreamCaptureImplicit),
            907 => Some(Self::CapturedEvent),
            908 => Some(Self::StreamCaptureWrongThread),
            909 => Some(Self::Timeout),
            910 => Some(Self::GraphExecUpdateFailure),
            911 => Some(Self::ExternalDevice),
            912 => Some(Self::InvalidClusterSize),
            913 => Some(Self::FunctionNotLoaded),
            914 => Some(Self::InvalidResourceType),
            915 => Some(Self::InvalidResourceConfiguration),
            _ => Some(Self::Unknown),
        }
    }

    /// Return `true` if the error is transient (e.g. `NotReady`).
    #[must_use]
    pub fn is_transient(self) -> bool {
        matches!(self, Self::NotReady)
    }

    /// Return `true` if the error indicates no CUDA hardware is present.
    #[must_use]
    pub fn is_no_device(self) -> bool {
        matches!(self, Self::NoDevice | Self::NoGpu | Self::DeviceNotSet)
    }
}

/// Convert a `CudaError` (driver error) to `CudaRtError`.
impl From<oxicuda_driver::error::CudaError> for CudaRtError {
    fn from(e: oxicuda_driver::error::CudaError) -> Self {
        // Map the most common driver errors to their runtime equivalents.
        use oxicuda_driver::error::CudaError as D;
        match e {
            D::InvalidValue => Self::InvalidValue,
            D::OutOfMemory => Self::MemoryAllocation,
            D::NotInitialized => Self::InitializationError,
            D::Deinitialized => Self::CudartUnloading,
            D::ProfilerDisabled => Self::ProfilerDisabled,
            D::StubLibrary => Self::StubLibrary,
            D::DeviceUnavailable => Self::DevicesUnavailable,
            D::NoDevice => Self::NoDevice,
            D::InvalidDevice => Self::InvalidDevice,
            D::InvalidImage => Self::InvalidKernelImage,
            D::NotReady => Self::NotReady,
            D::IllegalAddress => Self::IllegalAddress,
            D::LaunchOutOfResources => Self::LaunchOutOfResources,
            D::LaunchTimeout => Self::LaunchTimeout,
            D::LaunchFailed => Self::LaunchFailure,
            D::NotPermitted => Self::NotPermitted,
            D::NotSupported => Self::NotSupported,
            D::Unknown(_) => Self::Unknown,
            _ => Self::Unknown,
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_code_success_returns_none() {
        assert!(CudaRtError::from_code(0).is_none());
    }

    #[test]
    fn from_code_known_errors() {
        assert_eq!(CudaRtError::from_code(1), Some(CudaRtError::InvalidValue));
        assert_eq!(
            CudaRtError::from_code(2),
            Some(CudaRtError::MemoryAllocation)
        );
        assert_eq!(CudaRtError::from_code(100), Some(CudaRtError::NoDevice));
        assert_eq!(
            CudaRtError::from_code(719),
            Some(CudaRtError::LaunchFailure)
        );
        assert_eq!(CudaRtError::from_code(999), Some(CudaRtError::Unknown));
    }

    #[test]
    fn from_code_unrecognised_becomes_unknown() {
        assert_eq!(CudaRtError::from_code(12345), Some(CudaRtError::Unknown));
    }

    #[test]
    fn is_transient_only_for_not_ready() {
        assert!(CudaRtError::NotReady.is_transient());
        assert!(!CudaRtError::LaunchFailure.is_transient());
    }

    #[test]
    fn is_no_device_variants() {
        assert!(CudaRtError::NoDevice.is_no_device());
        assert!(CudaRtError::NoGpu.is_no_device());
        assert!(CudaRtError::DeviceNotSet.is_no_device());
        assert!(!CudaRtError::InvalidValue.is_no_device());
    }

    #[test]
    fn error_display() {
        let e = CudaRtError::MemoryAllocation;
        assert!(e.to_string().contains("out of device memory"));
    }

    #[test]
    fn from_driver_error() {
        let rt: CudaRtError = oxicuda_driver::error::CudaError::OutOfMemory.into();
        assert_eq!(rt, CudaRtError::MemoryAllocation);
    }
}
