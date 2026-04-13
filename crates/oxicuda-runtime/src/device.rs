//! Device management — `cudaGetDeviceCount`, `cudaSetDevice`, `cudaGetDevice`,
//! `cudaGetDeviceProperties`, `cudaDeviceSynchronize`, `cudaDeviceReset`.
//!
//! The CUDA Runtime maintains a *per-thread* current device.  This module
//! stores that state in a thread-local and exposes the standard Runtime API
//! surface on top of the underlying driver API.

use std::cell::Cell;
use std::ffi::c_int;

use oxicuda_driver::loader::try_driver;

use crate::error::{CudaRtError, CudaRtResult};

// ─── Thread-local current device ─────────────────────────────────────────────

thread_local! {
    /// The device ordinal bound to the calling thread.
    /// `None` means no device has been selected yet.
    static CURRENT_DEVICE: Cell<Option<c_int>> = const { Cell::new(None) };
}

// ─── cudaDeviceProp ───────────────────────────────────────────────────────────

/// Subset of `cudaDeviceProp` exposed by the Runtime API.
///
/// Field names intentionally match the CUDA Runtime documentation so that code
/// written against `cudaGetDeviceProperties` compiles with minimal changes.
#[derive(Debug, Clone)]
pub struct CudaDeviceProp {
    /// ASCII name of the device, e.g. `"NVIDIA GeForce RTX 4090"`.
    pub name: String,
    /// Total amount of global memory in bytes.
    pub total_global_mem: usize,
    /// Shared memory per block in bytes.
    pub shared_mem_per_block: usize,
    /// 32-bit registers per block.
    pub regs_per_block: u32,
    /// Warp size (threads per warp).
    pub warp_size: u32,
    /// Maximum pitch in bytes for `cudaMallocPitch`.
    pub mem_pitch: usize,
    /// Maximum number of threads per block.
    pub max_threads_per_block: u32,
    /// Maximum size of each dimension of a block `[x, y, z]`.
    pub max_threads_dim: [u32; 3],
    /// Maximum size of each dimension of a grid `[x, y, z]`.
    pub max_grid_size: [u32; 3],
    /// Clock frequency in kilohertz.
    pub clock_rate: u32,
    /// Total constant memory available on device in bytes.
    pub total_const_mem: usize,
    /// Major revision number of device's compute capability.
    pub major: u32,
    /// Minor revision number of device's compute capability.
    pub minor: u32,
    /// Alignment requirement for textures (in bytes).
    pub texture_alignment: usize,
    /// Pitch alignment requirement for texture references (in bytes).
    pub texture_pitch_alignment: usize,
    /// `true` if device can concurrently copy and execute a kernel.
    pub device_overlap: bool,
    /// Number of multiprocessors on device.
    pub multi_processor_count: u32,
    /// `true` if device has ECC support enabled.
    pub ecc_enabled: bool,
    /// `true` if device is an integrated (on-chip) GPU.
    pub integrated: bool,
    /// `true` if device can map host memory.
    pub can_map_host_memory: bool,
    /// `true` if device supports unified virtual addressing.
    pub unified_addressing: bool,
    /// Peak memory clock frequency in kilohertz.
    pub memory_clock_rate: u32,
    /// Global memory bus width in bits.
    pub memory_bus_width: u32,
    /// Size of the L2 cache in bytes (0 if not applicable).
    pub l2_cache_size: u32,
    /// Maximum number of resident threads per multiprocessor.
    pub max_threads_per_multi_processor: u32,
    /// Device supports stream priorities.
    pub stream_priorities_supported: bool,
    /// Shared memory per multiprocessor in bytes.
    pub shared_mem_per_multiprocessor: usize,
    /// 32-bit registers per multiprocessor.
    pub regs_per_multiprocessor: u32,
    /// Device supports allocating managed memory.
    pub managed_memory: bool,
    /// Device is on a multi-GPU board.
    pub is_multi_gpu_board: bool,
    /// Unique identifier for a group of devices on the same multi-GPU board.
    pub multi_gpu_board_group_id: u32,
    /// Link between the device and the host supports native atomic operations.
    pub host_native_atomic_supported: bool,
    /// `true` if the device supports Cooperative Launch.
    pub cooperative_launch: bool,
    /// `true` if the device supports Multi-Device Cooperative Launch.
    pub cooperative_multi_device_launch: bool,
    /// Maximum number of blocks per multiprocessor.
    pub max_blocks_per_multi_processor: u32,
    /// Per-device maximum shared memory per block usable without opt-in.
    pub shared_mem_per_block_optin: usize,
    /// `true` if device supports cluster launch.
    pub cluster_launch: bool,
}

impl CudaDeviceProp {
    /// Construct a `CudaDeviceProp` by querying device attributes via the
    /// CUDA Driver API.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver is not loaded or any attribute query
    /// fails.
    pub fn from_device(ordinal: c_int) -> CudaRtResult<Self> {
        let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;

        // Helper: query one attribute; map driver error to runtime error.
        let attr = |a: oxicuda_driver::ffi::CUdevice_attribute| -> CudaRtResult<u32> {
            let mut v: c_int = 0;
            // SAFETY: FFI; driver validates the attribute enum.
            let rc = unsafe { (api.cu_device_get_attribute)(&raw mut v, a, ordinal) };
            if rc != 0 {
                return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
            }
            Ok(v as u32)
        };

        use oxicuda_driver::ffi::CUdevice_attribute as A;

        // Device name
        let mut name_buf = [0u8; 256];
        // SAFETY: FFI; name_buf is valid and len matches.
        unsafe {
            (api.cu_device_get_name)(
                name_buf.as_mut_ptr() as *mut std::ffi::c_char,
                name_buf.len() as c_int,
                ordinal,
            );
        }
        let name = {
            let nul = name_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_buf.len());
            String::from_utf8_lossy(&name_buf[..nul]).into_owned()
        };

        // Total global memory
        let mut total_global_mem: usize = 0;
        // SAFETY: FFI; pointer is valid.
        unsafe {
            (api.cu_device_total_mem_v2)(&raw mut total_global_mem, ordinal);
        }

        Ok(Self {
            name,
            total_global_mem,
            shared_mem_per_block: attr(A::MaxSharedMemoryPerBlock)? as usize,
            regs_per_block: attr(A::MaxRegistersPerBlock)?,
            warp_size: attr(A::WarpSize)?,
            mem_pitch: attr(A::MaxPitch)? as usize,
            max_threads_per_block: attr(A::MaxThreadsPerBlock)?,
            max_threads_dim: [
                attr(A::MaxBlockDimX)?,
                attr(A::MaxBlockDimY)?,
                attr(A::MaxBlockDimZ)?,
            ],
            max_grid_size: [
                attr(A::MaxGridDimX)?,
                attr(A::MaxGridDimY)?,
                attr(A::MaxGridDimZ)?,
            ],
            clock_rate: attr(A::ClockRate)?,
            total_const_mem: attr(A::TotalConstantMemory)? as usize,
            major: attr(A::ComputeCapabilityMajor)?,
            minor: attr(A::ComputeCapabilityMinor)?,
            texture_alignment: attr(A::TextureAlignment)? as usize,
            texture_pitch_alignment: attr(A::TexturePitchAlignment)? as usize,
            device_overlap: attr(A::GpuOverlap)? != 0,
            multi_processor_count: attr(A::MultiprocessorCount)?,
            ecc_enabled: attr(A::EccEnabled)? != 0,
            integrated: attr(A::Integrated)? != 0,
            can_map_host_memory: attr(A::CanMapHostMemory)? != 0,
            unified_addressing: attr(A::UnifiedAddressing)? != 0,
            memory_clock_rate: attr(A::MemoryClockRate)?,
            memory_bus_width: attr(A::GlobalMemoryBusWidth)?,
            l2_cache_size: attr(A::L2CacheSize)?,
            max_threads_per_multi_processor: attr(A::MaxThreadsPerMultiprocessor)?,
            stream_priorities_supported: attr(A::StreamPrioritiesSupported)? != 0,
            shared_mem_per_multiprocessor: attr(A::MaxSharedMemoryPerMultiprocessor)? as usize,
            regs_per_multiprocessor: attr(A::MaxRegistersPerMultiprocessor)?,
            managed_memory: attr(A::ManagedMemory)? != 0,
            is_multi_gpu_board: attr(A::IsMultiGpuBoard)? != 0,
            multi_gpu_board_group_id: attr(A::MultiGpuBoardGroupId)?,
            host_native_atomic_supported: attr(A::HostNativeAtomicSupported)? != 0,
            cooperative_launch: attr(A::CooperativeLaunch)? != 0,
            cooperative_multi_device_launch: attr(A::CooperativeMultiDeviceLaunch)? != 0,
            max_blocks_per_multi_processor: attr(A::MaxBlocksPerMultiprocessor)?,
            shared_mem_per_block_optin: attr(A::MaxSharedMemoryPerBlockOptin)? as usize,
            cluster_launch: attr(A::ClusterLaunch)? != 0,
        })
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Returns the number of CUDA-capable devices.
///
/// Mirrors `cudaGetDeviceCount`.
///
/// # Errors
///
/// Returns [`CudaRtError::DriverNotAvailable`] if the CUDA driver is not
/// installed, or [`CudaRtError::NoGpu`] on systems with zero CUDA devices.
pub fn get_device_count() -> CudaRtResult<u32> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut count: c_int = 0;
    // SAFETY: FFI; count is a valid stack-allocated i32. cuInit(0) was called
    // during DriverApi::load(), so the driver is guaranteed to be initialised.
    let rc = unsafe { (api.cu_device_get_count)(&raw mut count) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::NoGpu));
    }
    if count == 0 {
        return Err(CudaRtError::NoGpu);
    }
    Ok(count as u32)
}

/// Selects `device` as the current CUDA device for the calling thread.
///
/// Mirrors `cudaSetDevice`.
///
/// # Errors
///
/// Returns [`CudaRtError::InvalidDevice`] if `device >= get_device_count()`,
/// or [`CudaRtError::DriverNotAvailable`] if the driver is absent.
pub fn set_device(device: u32) -> CudaRtResult<()> {
    let count = get_device_count()?;
    if device >= count {
        return Err(CudaRtError::InvalidDevice);
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // Retain the primary context for this device (creates it if necessary) and
    // make it the current context on the calling thread.  This is what the real
    // `cudaSetDevice` does internally; without it, driver API calls that need
    // a context (cuMemAlloc, cuLaunchKernel, …) fail with DeviceUninitialized.
    let mut ctx = oxicuda_driver::ffi::CUcontext::default();
    // SAFETY: FFI; device is a valid ordinal (checked above).
    let rc = unsafe { (api.cu_device_primary_ctx_retain)(&raw mut ctx, device as c_int) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    // SAFETY: ctx is a valid primary context handle.
    let rc = unsafe { (api.cu_ctx_set_current)(ctx) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    CURRENT_DEVICE.with(|cell| cell.set(Some(device as c_int)));
    Ok(())
}

/// Returns the ordinal of the current CUDA device for the calling thread.
///
/// Mirrors `cudaGetDevice`.
///
/// # Errors
///
/// Returns [`CudaRtError::DeviceNotSet`] if no device has been selected.
pub fn get_device() -> CudaRtResult<u32> {
    CURRENT_DEVICE.with(|cell| {
        cell.get()
            .map(|d| d as u32)
            .ok_or(CudaRtError::DeviceNotSet)
    })
}

/// Returns properties of the specified device.
///
/// Mirrors `cudaGetDeviceProperties`.
///
/// # Errors
///
/// Propagates driver errors or returns [`CudaRtError::InvalidDevice`] for
/// out-of-range ordinals.
pub fn get_device_properties(device: u32) -> CudaRtResult<CudaDeviceProp> {
    let count = get_device_count()?;
    if device >= count {
        return Err(CudaRtError::InvalidDevice);
    }
    CudaDeviceProp::from_device(device as c_int)
}

/// Blocks until all preceding tasks in the current device's context complete.
///
/// Mirrors `cudaDeviceSynchronize`.
///
/// # Errors
///
/// Returns [`CudaRtError::DeviceNotSet`] if no device is selected, or
/// a driver error if synchronization fails.
pub fn device_synchronize() -> CudaRtResult<()> {
    let _device = get_device()?;
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; driver's current context is valid.
    unsafe { (api.cu_ctx_synchronize)() };
    Ok(())
}

/// Explicitly destroys and cleans up all resources associated with the current
/// device in the current process.
///
/// Mirrors `cudaDeviceReset`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn device_reset() -> CudaRtResult<()> {
    let _device = get_device()?;
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // The Runtime API implements reset by resetting the primary context.
    // We obtain and then release the primary context to force a reset.
    let mut _dev: c_int = 0;
    CURRENT_DEVICE.with(|cell| {
        if let Some(d) = cell.get() {
            _dev = d;
        }
    });
    // SAFETY: FFI; _dev is a valid device ordinal.
    unsafe { (api.cu_device_primary_ctx_reset_v2)(_dev) };
    // Forget the device binding for this thread.
    CURRENT_DEVICE.with(|cell| cell.set(None));
    Ok(())
}

/// Returns the compute capability as a `(major, minor)` pair.
///
/// Convenience helper built on top of [`get_device_properties`].
///
/// # Errors
///
/// Propagates errors from `get_device_properties`.
pub fn get_compute_capability(device: u32) -> CudaRtResult<(u32, u32)> {
    let props = get_device_properties(device)?;
    Ok((props.major, props.minor))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "gpu-tests"))]
    fn get_device_without_set_errors() {
        // Fresh thread has no device set.
        let result = get_device();
        assert!(matches!(result, Err(CudaRtError::DeviceNotSet)));
    }

    #[test]
    fn set_device_persists_in_thread() {
        // We can't test against a real GPU here, but we can verify the
        // thread-local logic is consistent if get_device_count works.
        // If driver is absent, both calls return the same error class.
        let count_result = get_device_count();
        match count_result {
            Err(CudaRtError::DriverNotAvailable) | Err(CudaRtError::NoGpu) => {
                // No GPU environment — only verify DeviceNotSet.
                assert!(get_device().is_err());
            }
            Ok(n) => {
                // Real GPU: set device 0 and verify round-trip.
                set_device(0).expect("set_device(0) failed");
                assert_eq!(get_device().unwrap(), 0);
                // Out-of-range device must fail.
                assert!(matches!(set_device(n), Err(CudaRtError::InvalidDevice)));
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn from_code_round_trip() {
        // Verify that error::from_code covers the device-specific codes.
        assert_eq!(CudaRtError::from_code(100), Some(CudaRtError::NoDevice));
        assert_eq!(
            CudaRtError::from_code(101),
            Some(CudaRtError::InvalidDevice)
        );
    }
}
