//! CUDA device enumeration and attribute queries.
//!
//! This module provides the [`Device`] type, which wraps a `CUdevice` handle
//! obtained from the CUDA driver. Devices are identified by ordinal (0, 1, 2,
//! ...) and expose a rich set of convenience methods for querying hardware
//! capabilities such as compute capability, memory size, multiprocessor count,
//! and various limits.
//!
//! # Examples
//!
//! ```no_run
//! use oxicuda_driver::device::{Device, list_devices};
//!
//! oxicuda_driver::init()?;
//! for dev in list_devices()? {
//!     println!("{}: compute {}.{}",
//!         dev.name()?,
//!         dev.compute_capability()?.0,
//!         dev.compute_capability()?.1,
//!     );
//! }
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::ffi::{c_char, c_int};

use crate::error::CudaResult;
use crate::ffi::{CUdevice, CUdevice_attribute};
use crate::loader::try_driver;

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

/// Represents a CUDA-capable GPU device.
///
/// Wraps a `CUdevice` handle obtained from the driver API. Devices are
/// identified by a zero-based ordinal index. The handle is a lightweight
/// integer that can be freely copied.
///
/// # Examples
///
/// ```no_run
/// use oxicuda_driver::device::Device;
///
/// oxicuda_driver::init()?;
/// let device = Device::get(0)?;
/// println!("GPU: {}", device.name()?);
/// println!("Memory: {} MB", device.total_memory()? / (1024 * 1024));
/// let (major, minor) = device.compute_capability()?;
/// println!("Compute: {major}.{minor}");
/// # Ok::<(), oxicuda_driver::error::CudaError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    /// Raw CUDA device handle (integer ordinal internally).
    raw: CUdevice,
    /// The ordinal used to obtain this device.
    ordinal: i32,
}

impl Device {
    // -- Construction --------------------------------------------------------

    /// Get a device handle by ordinal (0-indexed).
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidDevice`](crate::error::CudaError::InvalidDevice) if the ordinal is out of range,
    /// or [`CudaError::NotInitialized`](crate::error::CudaError::NotInitialized) if the driver has not been loaded.
    pub fn get(ordinal: i32) -> CudaResult<Self> {
        let driver = try_driver()?;
        let mut raw: CUdevice = 0;
        crate::error::check(unsafe { (driver.cu_device_get)(&mut raw, ordinal) })?;
        Ok(Self { raw, ordinal })
    }

    /// Get the number of CUDA-capable devices in the system.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver cannot enumerate devices.
    pub fn count() -> CudaResult<i32> {
        let driver = try_driver()?;
        let mut count: std::ffi::c_int = 0;
        crate::error::check(unsafe { (driver.cu_device_get_count)(&mut count) })?;
        Ok(count)
    }

    // -- Identity ------------------------------------------------------------

    /// Get the device name (e.g., `"NVIDIA A100-SXM4-80GB"`).
    ///
    /// The returned string is an ASCII identifier provided by the driver.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver call fails.
    pub fn name(&self) -> CudaResult<String> {
        let driver = try_driver()?;
        let mut buf = [0u8; 256];
        crate::error::check(unsafe {
            (driver.cu_device_get_name)(buf.as_mut_ptr() as *mut c_char, 256, self.raw)
        })?;
        let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..len]).into_owned())
    }

    /// Get total device memory in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver call fails.
    pub fn total_memory(&self) -> CudaResult<usize> {
        let driver = try_driver()?;
        let mut bytes: usize = 0;
        crate::error::check(unsafe { (driver.cu_device_total_mem_v2)(&mut bytes, self.raw) })?;
        Ok(bytes)
    }

    // -- Generic attribute query --------------------------------------------

    /// Query an arbitrary device attribute.
    ///
    /// This is the low-level building block for all the convenience methods
    /// below. Callers can use any [`CUdevice_attribute`] variant directly.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute is not supported or the driver call
    /// fails.
    pub fn attribute(&self, attr: CUdevice_attribute) -> CudaResult<i32> {
        let driver = try_driver()?;
        let mut value: std::ffi::c_int = 0;
        crate::error::check(unsafe {
            (driver.cu_device_get_attribute)(&mut value, attr, self.raw)
        })?;
        Ok(value)
    }

    // -- Compute capability -------------------------------------------------

    /// Get compute capability as `(major, minor)`.
    ///
    /// For example, an A100 returns `(8, 0)` and an RTX 4090 returns `(8, 9)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver call fails.
    pub fn compute_capability(&self) -> CudaResult<(i32, i32)> {
        let major = self.attribute(CUdevice_attribute::ComputeCapabilityMajor)?;
        let minor = self.attribute(CUdevice_attribute::ComputeCapabilityMinor)?;
        Ok((major, minor))
    }

    // -- Thread / block / grid limits ---------------------------------------

    /// Get the maximum number of threads per block.
    pub fn max_threads_per_block(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxThreadsPerBlock)
    }

    /// Get the maximum block dimensions as `(x, y, z)`.
    pub fn max_block_dim(&self) -> CudaResult<(i32, i32, i32)> {
        Ok((
            self.attribute(CUdevice_attribute::MaxBlockDimX)?,
            self.attribute(CUdevice_attribute::MaxBlockDimY)?,
            self.attribute(CUdevice_attribute::MaxBlockDimZ)?,
        ))
    }

    /// Get the maximum grid dimensions as `(x, y, z)`.
    pub fn max_grid_dim(&self) -> CudaResult<(i32, i32, i32)> {
        Ok((
            self.attribute(CUdevice_attribute::MaxGridDimX)?,
            self.attribute(CUdevice_attribute::MaxGridDimY)?,
            self.attribute(CUdevice_attribute::MaxGridDimZ)?,
        ))
    }

    /// Get the maximum number of threads per multiprocessor.
    pub fn max_threads_per_multiprocessor(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxThreadsPerMultiprocessor)
    }

    /// Get the maximum number of blocks per multiprocessor.
    pub fn max_blocks_per_multiprocessor(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxBlocksPerMultiprocessor)
    }

    // -- Multiprocessor / warp ----------------------------------------------

    /// Get the number of streaming multiprocessors (SMs) on the device.
    pub fn multiprocessor_count(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MultiprocessorCount)
    }

    /// Get the warp size in threads (typically 32 for all NVIDIA GPUs).
    pub fn warp_size(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::WarpSize)
    }

    // -- Memory hierarchy ---------------------------------------------------

    /// Get the maximum shared memory per block in bytes.
    pub fn max_shared_memory_per_block(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxSharedMemoryPerBlock)
    }

    /// Get the maximum shared memory per multiprocessor in bytes.
    pub fn max_shared_memory_per_multiprocessor(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxSharedMemoryPerMultiprocessor)
    }

    /// Get the maximum opt-in shared memory per block in bytes.
    ///
    /// This is the upper bound achievable via
    /// `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`.
    pub fn max_shared_memory_per_block_optin(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxSharedMemoryPerBlockOptin)
    }

    /// Get the maximum number of 32-bit registers per block.
    pub fn max_registers_per_block(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxRegistersPerBlock)
    }

    /// Get the maximum number of 32-bit registers per multiprocessor.
    pub fn max_registers_per_multiprocessor(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxRegistersPerMultiprocessor)
    }

    /// Get the L2 cache size in bytes.
    pub fn l2_cache_size(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::L2CacheSize)
    }

    /// Get the total constant memory on the device in bytes.
    pub fn total_constant_memory(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::TotalConstantMemory)
    }

    // -- Clock / bus --------------------------------------------------------

    /// Get the core clock rate in kHz.
    pub fn clock_rate_khz(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::ClockRate)
    }

    /// Get the memory clock rate in kHz.
    pub fn memory_clock_rate_khz(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MemoryClockRate)
    }

    /// Get the global memory bus width in bits.
    pub fn memory_bus_width(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::GlobalMemoryBusWidth)
    }

    // -- PCI topology -------------------------------------------------------

    /// Get the PCI bus ID of the device.
    pub fn pci_bus_id(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::PciBusId)
    }

    /// Get the PCI device ID.
    pub fn pci_device_id(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::PciDeviceId)
    }

    /// Get the PCI domain ID.
    pub fn pci_domain_id(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::PciDomainId)
    }

    // -- Feature / capability flags -----------------------------------------

    /// Check if the device supports managed (unified) memory.
    pub fn supports_managed_memory(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::ManagedMemory)? != 0)
    }

    /// Check if the device supports concurrent managed memory access.
    pub fn supports_concurrent_managed_access(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::ConcurrentManagedAccess)? != 0)
    }

    /// Check if the device supports concurrent kernel execution.
    pub fn supports_concurrent_kernels(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::ConcurrentKernels)? != 0)
    }

    /// Check if the device supports cooperative kernel launches.
    pub fn supports_cooperative_launch(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::CooperativeLaunch)? != 0)
    }

    /// Check if ECC memory is enabled on the device.
    pub fn ecc_enabled(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::EccEnabled)? != 0)
    }

    /// Check if the device is integrated (shares memory with the host).
    pub fn is_integrated(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::Integrated)? != 0)
    }

    /// Check if the device can map host memory into its address space.
    pub fn can_map_host_memory(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::CanMapHostMemory)? != 0)
    }

    /// Check if the device uses a unified address space with the host.
    pub fn supports_unified_addressing(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::UnifiedAddressing)? != 0)
    }

    /// Check if the device supports stream priorities.
    pub fn supports_stream_priorities(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::StreamPrioritiesSupported)? != 0)
    }

    /// Check if the device supports compute preemption.
    pub fn supports_compute_preemption(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::ComputePreemptionSupported)? != 0)
    }

    /// Get the number of asynchronous engines (copy engines).
    pub fn async_engine_count(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::AsyncEngineCount)
    }

    /// Check if the device is on a multi-GPU board.
    pub fn is_multi_gpu_board(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::IsMultiGpuBoard)? != 0)
    }

    /// Check if there is a kernel execution timeout enforced by the OS.
    pub fn has_kernel_exec_timeout(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::KernelExecTimeout)? != 0)
    }

    // -- Compute mode / driver model ----------------------------------------

    /// Get the compute mode (0=default, 1=exclusive-thread, 2=prohibited,
    /// 3=exclusive-process).
    pub fn compute_mode(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::ComputeMode)
    }

    /// Check if the device uses the TCC (Tesla Compute Cluster) driver model.
    ///
    /// TCC mode disables the display driver, giving full GPU resources to
    /// compute workloads.
    pub fn tcc_driver(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::TccDriver)? != 0)
    }

    /// Get the multi-GPU board group identifier.
    ///
    /// Devices on the same board share the same group ID.
    pub fn multi_gpu_board_group_id(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MultiGpuBoardGroupId)
    }

    // -- Memory features (extended) -----------------------------------------

    /// Get the maximum persisting L2 cache size in bytes (Ampere+).
    pub fn max_persisting_l2_cache_size(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxPersistingL2CacheSize)
    }

    /// Check if the device supports generic memory compression.
    pub fn supports_generic_compression(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::GenericCompressionSupported)? != 0)
    }

    /// Check if the device supports pageable memory access.
    pub fn supports_pageable_memory_access(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::PageableMemoryAccess)? != 0)
    }

    /// Check if pageable memory access uses host page tables.
    pub fn pageable_memory_uses_host_page_tables(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::PageableMemoryAccessUsesHostPageTables)? != 0)
    }

    /// Check if the device supports direct managed memory access from the host.
    pub fn supports_direct_managed_mem_from_host(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::DirectManagedMemAccessFromHost)? != 0)
    }

    /// Get memory pool supported handle types as a bitmask.
    pub fn memory_pool_supported_handle_types(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MemoryPoolSupportedHandleTypes)
    }

    // -- Advanced features --------------------------------------------------

    /// Check if the device supports host-visible native atomic operations.
    pub fn supports_host_native_atomics(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::HostNativeAtomicSupported)? != 0)
    }

    /// Get the ratio of single-precision to double-precision performance.
    ///
    /// A higher value means the GPU is relatively faster at FP32 than FP64.
    pub fn single_to_double_perf_ratio(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::SingleToDoublePrecisionPerfRatio)
    }

    /// Check if the device supports cooperative multi-device kernel launches.
    pub fn supports_cooperative_multi_device_launch(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::CooperativeMultiDeviceLaunch)? != 0)
    }

    /// Check if the device supports flushing outstanding remote writes.
    pub fn supports_flush_remote_writes(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::CanFlushRemoteWrites)? != 0)
    }

    /// Check if the device supports host-side memory register functions.
    pub fn supports_host_register(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::HostRegisterSupported)? != 0)
    }

    /// Check if the device can use host pointers for registered memory.
    pub fn can_use_host_pointer_for_registered_mem(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::CanUseHostPointerForRegisteredMem)? != 0)
    }

    /// Check if the device supports GPU Direct RDMA.
    pub fn supports_gpu_direct_rdma(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::GpuDirectRdmaSupported)? != 0)
    }

    /// Check if the device supports tensor-map access (Hopper+).
    pub fn supports_tensor_map_access(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::TensorMapAccessSupported)? != 0)
    }

    /// Check if the device supports multicast operations.
    pub fn supports_multicast(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::MulticastSupported)? != 0)
    }

    /// Check if Multi-Process Service (MPS) is enabled on the device.
    pub fn mps_enabled(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::MpsEnabled)? != 0)
    }

    // -- Texture / surface limits -------------------------------------------

    /// Get the maximum 1D texture width.
    pub fn max_texture_1d_width(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxTexture1DWidth)
    }

    /// Get the maximum 2D texture dimensions as `(width, height)`.
    pub fn max_texture_2d_dims(&self) -> CudaResult<(i32, i32)> {
        Ok((
            self.attribute(CUdevice_attribute::MaxTexture2DWidth)?,
            self.attribute(CUdevice_attribute::MaxTexture2DHeight)?,
        ))
    }

    /// Get the maximum 3D texture dimensions as `(width, height, depth)`.
    pub fn max_texture_3d_dims(&self) -> CudaResult<(i32, i32, i32)> {
        Ok((
            self.attribute(CUdevice_attribute::MaxTexture3DWidth)?,
            self.attribute(CUdevice_attribute::MaxTexture3DHeight)?,
            self.attribute(CUdevice_attribute::MaxTexture3DDepth)?,
        ))
    }

    /// Check if the device can copy memory and execute a kernel concurrently.
    pub fn gpu_overlap(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::GpuOverlap)? != 0)
    }

    /// Get the maximum pitch for memory copies in bytes.
    pub fn max_pitch(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxPitch)
    }

    /// Get the texture alignment requirement in bytes.
    pub fn texture_alignment(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::TextureAlignment)
    }

    /// Get the surface alignment requirement in bytes.
    pub fn surface_alignment(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::SurfaceAlignment)
    }

    /// Check if the device supports deferred mapping of CUDA arrays.
    pub fn supports_deferred_mapping(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::DeferredMappingCudaArraySupported)? != 0)
    }

    // -- Memory pool / async features ---------------------------------------

    /// Check if the device supports memory pools (`cudaMallocAsync`).
    pub fn supports_memory_pools(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::MemoryPoolsSupported)? != 0)
    }

    /// Check if the device supports cluster launch (Hopper+).
    pub fn supports_cluster_launch(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::ClusterLaunch)? != 0)
    }

    /// Check if the device supports virtual memory management APIs.
    pub fn supports_virtual_memory_management(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::VirtualMemoryManagementSupported)? != 0)
    }

    /// Check if the device supports POSIX file descriptor handles for IPC.
    pub fn supports_handle_type_posix_fd(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::HandleTypePosixFileDescriptorSupported)? != 0)
    }

    /// Check if the device supports Win32 handles for IPC.
    pub fn supports_handle_type_win32(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::HandleTypeWin32HandleSupported)? != 0)
    }

    /// Check if the device supports Win32 KMT handles for IPC.
    pub fn supports_handle_type_win32_kmt(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::HandleTypeWin32KmtHandleSupported)? != 0)
    }

    /// Check if the device supports GPU Direct RDMA with CUDA VMM.
    pub fn supports_gpu_direct_rdma_vmm(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::GpuDirectRdmaWithCudaVmmSupported)? != 0)
    }

    /// Get the GPU Direct RDMA flush-writes options bitmask.
    pub fn gpu_direct_rdma_flush_writes_options(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::GpuDirectRdmaFlushWritesOptions)
    }

    /// Get the GPU Direct RDMA writes ordering.
    pub fn gpu_direct_rdma_writes_ordering(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::GpuDirectRdmaWritesOrdering)
    }

    /// Get the maximum access-policy window size for L2 cache.
    pub fn max_access_policy_window_size(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MaxAccessPolicyWindowSize)
    }

    /// Get the reserved shared memory per block in bytes.
    pub fn reserved_shared_memory_per_block(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::ReservedSharedMemoryPerBlock)
    }

    /// Check if timeline semaphore interop is supported.
    pub fn supports_timeline_semaphore_interop(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::TimelineSemaphoreInteropSupported)? != 0)
    }

    /// Check if memory sync domain operations are supported.
    pub fn supports_mem_sync_domain(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::MemSyncDomainSupported)? != 0)
    }

    /// Get the number of memory sync domains.
    pub fn mem_sync_domain_count(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::MemSyncDomainCount)
    }

    /// Check if GPU-Direct Fabric (RDMA) is supported.
    pub fn supports_gpu_direct_rdma_fabric(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::GpuDirectRdmaFabricSupported)? != 0)
    }

    /// Check if unified function pointers are supported.
    pub fn supports_unified_function_pointers(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::UnifiedFunctionPointers)? != 0)
    }

    /// Check if IPC event handles are supported.
    pub fn supports_ipc_events(&self) -> CudaResult<bool> {
        Ok(self.attribute(CUdevice_attribute::IpcEventSupported)? != 0)
    }

    /// Get the NUMA configuration of the device.
    pub fn numa_config(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::NumaConfig)
    }

    /// Get the NUMA ID of the device.
    pub fn numa_id(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::NumaId)
    }

    /// Get the host NUMA ID of the device.
    pub fn host_numa_id(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::HostNumaId)
    }

    /// Get the texture pitch alignment requirement in bytes.
    pub fn texture_pitch_alignment(&self) -> CudaResult<i32> {
        self.attribute(CUdevice_attribute::TexturePitchAlignment)
    }

    // -- Structured info ----------------------------------------------------

    /// Gather comprehensive device information in a single call.
    ///
    /// Returns a [`DeviceInfo`] with all key properties. Individual attribute
    /// query failures are silently replaced with default values (`0` / `false`)
    /// so that the call succeeds even on older drivers that lack some attributes.
    ///
    /// # Errors
    ///
    /// Returns an error only if the device name or total memory cannot be
    /// queried (fundamental properties).
    pub fn info(&self) -> CudaResult<DeviceInfo> {
        let name = self.name()?;
        let total_memory_bytes = self.total_memory()?;
        let (cc_major, cc_minor) = self.compute_capability().unwrap_or((0, 0));

        Ok(DeviceInfo {
            name,
            ordinal: self.ordinal,
            compute_capability: (cc_major, cc_minor),
            total_memory_bytes,
            multiprocessor_count: self.multiprocessor_count().unwrap_or(0),
            max_threads_per_block: self.max_threads_per_block().unwrap_or(0),
            max_threads_per_sm: self.max_threads_per_multiprocessor().unwrap_or(0),
            warp_size: self.warp_size().unwrap_or(0),
            clock_rate_mhz: self.clock_rate_khz().unwrap_or(0) as f64 / 1000.0,
            memory_clock_rate_mhz: self.memory_clock_rate_khz().unwrap_or(0) as f64 / 1000.0,
            memory_bus_width_bits: self.memory_bus_width().unwrap_or(0),
            l2_cache_bytes: self.l2_cache_size().unwrap_or(0),
            max_shared_memory_per_block: self.max_shared_memory_per_block().unwrap_or(0),
            max_shared_memory_per_sm: self.max_shared_memory_per_multiprocessor().unwrap_or(0),
            max_registers_per_block: self.max_registers_per_block().unwrap_or(0),
            ecc_enabled: self.ecc_enabled().unwrap_or(false),
            tcc_driver: self.tcc_driver().unwrap_or(false),
            compute_mode: self.compute_mode().unwrap_or(0),
            supports_cooperative_launch: self.supports_cooperative_launch().unwrap_or(false),
            supports_managed_memory: self.supports_managed_memory().unwrap_or(false),
            max_persisting_l2_cache_bytes: self.max_persisting_l2_cache_size().unwrap_or(0),
            async_engine_count: self.async_engine_count().unwrap_or(0),
            supports_memory_pools: self.supports_memory_pools().unwrap_or(false),
            supports_gpu_direct_rdma: self.supports_gpu_direct_rdma().unwrap_or(false),
            supports_cluster_launch: self.supports_cluster_launch().unwrap_or(false),
            supports_concurrent_kernels: self.supports_concurrent_kernels().unwrap_or(false),
            supports_unified_addressing: self.supports_unified_addressing().unwrap_or(false),
            max_blocks_per_sm: self.max_blocks_per_multiprocessor().unwrap_or(0),
            single_to_double_perf_ratio: self.single_to_double_perf_ratio().unwrap_or(0),
        })
    }

    // -- Raw access ---------------------------------------------------------

    /// Get the raw `CUdevice` handle for use with FFI calls.
    #[inline]
    pub fn raw(&self) -> CUdevice {
        self.raw
    }

    /// Get the device ordinal that was used to obtain this handle.
    #[inline]
    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Device({})", self.ordinal)
    }
}

// ---------------------------------------------------------------------------
// DeviceInfo — structured summary
// ---------------------------------------------------------------------------

/// Comprehensive device information gathered in a single call.
///
/// All fields are populated via convenience methods on [`Device`]. Fields that
/// fail to query (e.g. on an older driver) default to `0` / `false`.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Human-readable device name.
    pub name: String,
    /// Zero-based device ordinal.
    pub ordinal: i32,
    /// Compute capability `(major, minor)`.
    pub compute_capability: (i32, i32),
    /// Total device memory in bytes.
    pub total_memory_bytes: usize,
    /// Number of streaming multiprocessors.
    pub multiprocessor_count: i32,
    /// Maximum threads per block.
    pub max_threads_per_block: i32,
    /// Maximum threads per streaming multiprocessor.
    pub max_threads_per_sm: i32,
    /// Warp size in threads.
    pub warp_size: i32,
    /// Core clock rate in MHz.
    pub clock_rate_mhz: f64,
    /// Memory clock rate in MHz.
    pub memory_clock_rate_mhz: f64,
    /// Memory bus width in bits.
    pub memory_bus_width_bits: i32,
    /// L2 cache size in bytes.
    pub l2_cache_bytes: i32,
    /// Maximum shared memory per block in bytes.
    pub max_shared_memory_per_block: i32,
    /// Maximum shared memory per SM in bytes.
    pub max_shared_memory_per_sm: i32,
    /// Maximum 32-bit registers per block.
    pub max_registers_per_block: i32,
    /// ECC memory enabled.
    pub ecc_enabled: bool,
    /// TCC driver mode.
    pub tcc_driver: bool,
    /// Compute mode (0=default, 1=exclusive-thread, 2=prohibited, 3=exclusive-process).
    pub compute_mode: i32,
    /// Cooperative kernel launch support.
    pub supports_cooperative_launch: bool,
    /// Managed (unified) memory support.
    pub supports_managed_memory: bool,
    /// Maximum persisting L2 cache size in bytes (Ampere+).
    pub max_persisting_l2_cache_bytes: i32,
    /// Number of async copy engines.
    pub async_engine_count: i32,
    /// Supports memory pools (`cudaMallocAsync`).
    pub supports_memory_pools: bool,
    /// Supports GPU Direct RDMA.
    pub supports_gpu_direct_rdma: bool,
    /// Supports cluster launch (Hopper+).
    pub supports_cluster_launch: bool,
    /// Concurrent kernel execution supported.
    pub supports_concurrent_kernels: bool,
    /// Unified addressing supported.
    pub supports_unified_addressing: bool,
    /// Maximum blocks per SM.
    pub max_blocks_per_sm: i32,
    /// Single-to-double precision performance ratio.
    pub single_to_double_perf_ratio: i32,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mem_mb = self.total_memory_bytes / (1024 * 1024);
        let (major, minor) = self.compute_capability;
        writeln!(f, "Device {}: {}", self.ordinal, self.name)?;
        writeln!(f, "  Compute capability : {major}.{minor}")?;
        writeln!(f, "  Total memory       : {mem_mb} MB")?;
        writeln!(f, "  SMs                : {}", self.multiprocessor_count)?;
        writeln!(f, "  Max threads/block  : {}", self.max_threads_per_block)?;
        writeln!(f, "  Max threads/SM     : {}", self.max_threads_per_sm)?;
        writeln!(f, "  Warp size          : {}", self.warp_size)?;
        writeln!(f, "  Core clock         : {:.1} MHz", self.clock_rate_mhz)?;
        writeln!(
            f,
            "  Memory clock       : {:.1} MHz",
            self.memory_clock_rate_mhz
        )?;
        writeln!(
            f,
            "  Memory bus         : {} bits",
            self.memory_bus_width_bits
        )?;
        writeln!(
            f,
            "  L2 cache           : {} KB",
            self.l2_cache_bytes / 1024
        )?;
        writeln!(
            f,
            "  Shared mem/block   : {} KB",
            self.max_shared_memory_per_block / 1024
        )?;
        writeln!(
            f,
            "  Shared mem/SM      : {} KB",
            self.max_shared_memory_per_sm / 1024
        )?;
        writeln!(f, "  Registers/block    : {}", self.max_registers_per_block)?;
        writeln!(f, "  ECC                : {}", self.ecc_enabled)?;
        writeln!(f, "  TCC driver         : {}", self.tcc_driver)?;
        writeln!(f, "  Compute mode       : {}", self.compute_mode)?;
        writeln!(
            f,
            "  Cooperative launch : {}",
            self.supports_cooperative_launch
        )?;
        writeln!(f, "  Managed memory     : {}", self.supports_managed_memory)?;
        writeln!(
            f,
            "  Persist L2 cache   : {} KB",
            self.max_persisting_l2_cache_bytes / 1024
        )?;
        writeln!(f, "  Async engines      : {}", self.async_engine_count)?;
        writeln!(f, "  Memory pools       : {}", self.supports_memory_pools)?;
        writeln!(
            f,
            "  GPU Direct RDMA    : {}",
            self.supports_gpu_direct_rdma
        )?;
        writeln!(f, "  Cluster launch     : {}", self.supports_cluster_launch)?;
        writeln!(
            f,
            "  Concurrent kernels : {}",
            self.supports_concurrent_kernels
        )?;
        writeln!(
            f,
            "  Unified addressing : {}",
            self.supports_unified_addressing
        )?;
        writeln!(f, "  Max blocks/SM      : {}", self.max_blocks_per_sm)?;
        write!(
            f,
            "  FP32/FP64 ratio    : {}",
            self.single_to_double_perf_ratio
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// List all available CUDA devices.
///
/// Returns a vector of [`Device`] handles for every GPU visible to the driver.
/// The vector is empty if no CUDA-capable devices are present.
///
/// # Errors
///
/// Returns an error if the driver cannot be loaded or device enumeration fails.
///
/// # Examples
///
/// ```no_run
/// use oxicuda_driver::device::list_devices;
///
/// oxicuda_driver::init()?;
/// for dev in list_devices()? {
///     println!("{}: {} MB", dev.name()?, dev.total_memory()? / (1024 * 1024));
/// }
/// # Ok::<(), oxicuda_driver::error::CudaError>(())
/// ```
pub fn list_devices() -> CudaResult<Vec<Device>> {
    let count = Device::count()?;
    let mut devices = Vec::with_capacity(count as usize);
    for i in 0..count {
        devices.push(Device::get(i)?);
    }
    Ok(devices)
}

/// Query the CUDA driver version.
///
/// Returns the version as `major * 1000 + minor * 10`, e.g. 12060 for CUDA 12.6.
/// This is a system-wide property that does not depend on a specific device.
///
/// # Errors
///
/// Returns an error if the driver cannot be loaded or `cuDriverGetVersion` fails.
pub fn driver_version() -> CudaResult<i32> {
    let driver = try_driver()?;
    let mut version: c_int = 0;
    crate::error::check(unsafe { (driver.cu_driver_get_version)(&mut version) })?;
    Ok(version)
}

/// Query whether peer access is possible between two devices.
///
/// Peer access allows one GPU to directly access another GPU's memory
/// without going through the host. This requires both devices to support
/// peer access and be connected via a suitable interconnect (e.g., NVLink
/// or PCIe peer-to-peer).
///
/// # Parameters
///
/// * `device` — the device that would initiate the access.
/// * `peer` — the device whose memory would be accessed.
///
/// # Errors
///
/// Returns an error if the driver cannot be loaded or the query fails.
pub fn can_access_peer(device: &Device, peer: &Device) -> CudaResult<bool> {
    let driver = try_driver()?;
    let mut can_access: c_int = 0;
    crate::error::check(unsafe {
        (driver.cu_device_can_access_peer)(&mut can_access, device.raw(), peer.raw())
    })?;
    Ok(can_access != 0)
}

/// Find the device with the most total memory.
///
/// Returns `None` if no CUDA devices are available.
///
/// # Errors
///
/// Returns an error if device enumeration or memory queries fail.
///
/// # Examples
///
/// ```no_run
/// use oxicuda_driver::device::best_device;
///
/// oxicuda_driver::init()?;
/// if let Some(dev) = best_device()? {
///     println!("Best GPU: {} ({} MB)", dev.name()?, dev.total_memory()? / (1024 * 1024));
/// }
/// # Ok::<(), oxicuda_driver::error::CudaError>(())
/// ```
pub fn best_device() -> CudaResult<Option<Device>> {
    let devices = list_devices()?;
    if devices.is_empty() {
        return Ok(None);
    }
    let mut best = devices[0];
    let mut best_mem = best.total_memory()?;
    for dev in devices.iter().skip(1) {
        let mem = dev.total_memory()?;
        if mem > best_mem {
            best = *dev;
            best_mem = mem;
        }
    }
    Ok(Some(best))
}
