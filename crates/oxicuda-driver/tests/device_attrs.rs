//! Tests for extended device attribute convenience methods and DeviceInfo.
//!
//! Since we may run on macOS without an NVIDIA GPU, all tests use if-let
//! patterns: they attempt to initialise the driver and get a device, but
//! gracefully skip when the platform is unsupported.

use oxicuda_driver::CudaResult;
use oxicuda_driver::device::{Device, DeviceInfo};

// ---------------------------------------------------------------------------
// Helper: try to get device 0, returning None on unsupported platform
// ---------------------------------------------------------------------------

fn try_device() -> Option<Device> {
    if oxicuda_driver::init().is_err() {
        return None;
    }
    Device::get(0).ok()
}

// ---------------------------------------------------------------------------
// Compile-time type checks — ensure methods return the expected types
// ---------------------------------------------------------------------------

/// Verify bool-returning methods have the correct signature.
#[allow(dead_code)]
fn assert_bool_signatures(dev: &Device) {
    let _: CudaResult<bool> = dev.tcc_driver();
    let _: CudaResult<bool> = dev.supports_generic_compression();
    let _: CudaResult<bool> = dev.supports_pageable_memory_access();
    let _: CudaResult<bool> = dev.pageable_memory_uses_host_page_tables();
    let _: CudaResult<bool> = dev.supports_direct_managed_mem_from_host();
    let _: CudaResult<bool> = dev.supports_host_native_atomics();
    let _: CudaResult<bool> = dev.supports_cooperative_multi_device_launch();
    let _: CudaResult<bool> = dev.supports_flush_remote_writes();
    let _: CudaResult<bool> = dev.supports_host_register();
    let _: CudaResult<bool> = dev.can_use_host_pointer_for_registered_mem();
    let _: CudaResult<bool> = dev.supports_gpu_direct_rdma();
    let _: CudaResult<bool> = dev.supports_tensor_map_access();
    let _: CudaResult<bool> = dev.supports_multicast();
    let _: CudaResult<bool> = dev.mps_enabled();
    let _: CudaResult<bool> = dev.gpu_overlap();
    let _: CudaResult<bool> = dev.supports_deferred_mapping();
    // New methods
    let _: CudaResult<bool> = dev.supports_memory_pools();
    let _: CudaResult<bool> = dev.supports_cluster_launch();
    let _: CudaResult<bool> = dev.supports_virtual_memory_management();
    let _: CudaResult<bool> = dev.supports_handle_type_posix_fd();
    let _: CudaResult<bool> = dev.supports_handle_type_win32();
    let _: CudaResult<bool> = dev.supports_handle_type_win32_kmt();
    let _: CudaResult<bool> = dev.supports_gpu_direct_rdma_vmm();
    let _: CudaResult<bool> = dev.supports_timeline_semaphore_interop();
    let _: CudaResult<bool> = dev.supports_mem_sync_domain();
    let _: CudaResult<bool> = dev.supports_gpu_direct_rdma_fabric();
    let _: CudaResult<bool> = dev.supports_unified_function_pointers();
    let _: CudaResult<bool> = dev.supports_ipc_events();
}

/// Verify i32-returning methods have the correct signature.
#[allow(dead_code)]
fn assert_i32_signatures(dev: &Device) {
    let _: CudaResult<i32> = dev.compute_mode();
    let _: CudaResult<i32> = dev.multi_gpu_board_group_id();
    let _: CudaResult<i32> = dev.max_persisting_l2_cache_size();
    let _: CudaResult<i32> = dev.memory_pool_supported_handle_types();
    let _: CudaResult<i32> = dev.single_to_double_perf_ratio();
    let _: CudaResult<i32> = dev.max_texture_1d_width();
    let _: CudaResult<i32> = dev.max_pitch();
    let _: CudaResult<i32> = dev.texture_alignment();
    let _: CudaResult<i32> = dev.surface_alignment();
    // New methods
    let _: CudaResult<i32> = dev.gpu_direct_rdma_flush_writes_options();
    let _: CudaResult<i32> = dev.gpu_direct_rdma_writes_ordering();
    let _: CudaResult<i32> = dev.max_access_policy_window_size();
    let _: CudaResult<i32> = dev.reserved_shared_memory_per_block();
    let _: CudaResult<i32> = dev.mem_sync_domain_count();
    let _: CudaResult<i32> = dev.numa_config();
    let _: CudaResult<i32> = dev.numa_id();
    let _: CudaResult<i32> = dev.host_numa_id();
    let _: CudaResult<i32> = dev.texture_pitch_alignment();
}

/// Verify tuple-returning methods have the correct signature.
#[allow(dead_code)]
fn assert_tuple_signatures(dev: &Device) {
    let _: CudaResult<(i32, i32)> = dev.max_texture_2d_dims();
    let _: CudaResult<(i32, i32, i32)> = dev.max_texture_3d_dims();
}

// ---------------------------------------------------------------------------
// Compute mode / driver model tests
// ---------------------------------------------------------------------------

#[test]
fn test_compute_mode() {
    if let Some(dev) = try_device() {
        let mode = dev.compute_mode();
        assert!(mode.is_ok());
        let val = mode.unwrap_or(-1);
        // Compute mode is 0..=3
        assert!((0..=3).contains(&val), "unexpected compute mode: {val}");
    }
}

#[test]
fn test_tcc_driver() {
    if let Some(dev) = try_device() {
        // Just verify it returns Ok(bool)
        let result = dev.tcc_driver();
        assert!(result.is_ok());
    }
}

#[test]
fn test_multi_gpu_board_group_id() {
    if let Some(dev) = try_device() {
        let result = dev.multi_gpu_board_group_id();
        assert!(result.is_ok());
    }
}

// ---------------------------------------------------------------------------
// Memory features tests
// ---------------------------------------------------------------------------

#[test]
fn test_max_persisting_l2_cache_size() {
    if let Some(dev) = try_device() {
        let result = dev.max_persisting_l2_cache_size();
        assert!(result.is_ok());
        // Must be non-negative
        assert!(result.unwrap_or(-1) >= 0);
    }
}

#[test]
fn test_supports_generic_compression() {
    if let Some(dev) = try_device() {
        let result = dev.supports_generic_compression();
        assert!(result.is_ok());
    }
}

#[test]
fn test_supports_pageable_memory_access() {
    if let Some(dev) = try_device() {
        let result = dev.supports_pageable_memory_access();
        assert!(result.is_ok());
    }
}

#[test]
fn test_pageable_memory_uses_host_page_tables() {
    if let Some(dev) = try_device() {
        let result = dev.pageable_memory_uses_host_page_tables();
        assert!(result.is_ok());
    }
}

#[test]
fn test_supports_direct_managed_mem_from_host() {
    if let Some(dev) = try_device() {
        let result = dev.supports_direct_managed_mem_from_host();
        assert!(result.is_ok());
    }
}

#[test]
fn test_memory_pool_supported_handle_types() {
    if let Some(dev) = try_device() {
        let result = dev.memory_pool_supported_handle_types();
        assert!(result.is_ok());
    }
}

// ---------------------------------------------------------------------------
// Advanced feature tests
// ---------------------------------------------------------------------------

#[test]
fn test_supports_host_native_atomics() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_host_native_atomics().is_ok());
    }
}

#[test]
fn test_single_to_double_perf_ratio() {
    if let Some(dev) = try_device() {
        let result = dev.single_to_double_perf_ratio();
        assert!(result.is_ok());
        assert!(result.unwrap_or(-1) >= 0);
    }
}

#[test]
fn test_supports_cooperative_multi_device_launch() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_cooperative_multi_device_launch().is_ok());
    }
}

#[test]
fn test_supports_flush_remote_writes() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_flush_remote_writes().is_ok());
    }
}

#[test]
fn test_supports_host_register() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_host_register().is_ok());
    }
}

#[test]
fn test_can_use_host_pointer_for_registered_mem() {
    if let Some(dev) = try_device() {
        assert!(dev.can_use_host_pointer_for_registered_mem().is_ok());
    }
}

#[test]
fn test_supports_gpu_direct_rdma() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_gpu_direct_rdma().is_ok());
    }
}

#[test]
fn test_supports_tensor_map_access() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_tensor_map_access().is_ok());
    }
}

#[test]
fn test_supports_multicast() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_multicast().is_ok());
    }
}

#[test]
fn test_mps_enabled() {
    if let Some(dev) = try_device() {
        assert!(dev.mps_enabled().is_ok());
    }
}

// ---------------------------------------------------------------------------
// Texture / surface limit tests
// ---------------------------------------------------------------------------

#[test]
fn test_max_texture_1d_width() {
    if let Some(dev) = try_device() {
        let result = dev.max_texture_1d_width();
        assert!(result.is_ok());
        assert!(
            result.unwrap_or(-1) > 0,
            "1D texture width should be positive"
        );
    }
}

#[test]
fn test_max_texture_2d_dims() {
    if let Some(dev) = try_device() {
        let result = dev.max_texture_2d_dims();
        assert!(result.is_ok());
        if let Ok((w, h)) = result {
            assert!(
                w > 0 && h > 0,
                "2D texture dims should be positive: ({w}, {h})"
            );
        }
    }
}

#[test]
fn test_max_texture_3d_dims() {
    if let Some(dev) = try_device() {
        let result = dev.max_texture_3d_dims();
        assert!(result.is_ok());
        if let Ok((w, h, d)) = result {
            assert!(
                w > 0 && h > 0 && d > 0,
                "3D texture dims should be positive: ({w}, {h}, {d})"
            );
        }
    }
}

#[test]
fn test_gpu_overlap() {
    if let Some(dev) = try_device() {
        assert!(dev.gpu_overlap().is_ok());
    }
}

#[test]
fn test_max_pitch() {
    if let Some(dev) = try_device() {
        let result = dev.max_pitch();
        assert!(result.is_ok());
        assert!(result.unwrap_or(-1) > 0, "max pitch should be positive");
    }
}

#[test]
fn test_texture_alignment() {
    if let Some(dev) = try_device() {
        let result = dev.texture_alignment();
        assert!(result.is_ok());
        assert!(
            result.unwrap_or(-1) > 0,
            "texture alignment should be positive"
        );
    }
}

#[test]
fn test_surface_alignment() {
    if let Some(dev) = try_device() {
        let result = dev.surface_alignment();
        assert!(result.is_ok());
    }
}

#[test]
fn test_supports_deferred_mapping() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_deferred_mapping().is_ok());
    }
}

// ---------------------------------------------------------------------------
// DeviceInfo tests
// ---------------------------------------------------------------------------

#[test]
fn test_device_info_struct_creation() {
    // Verify DeviceInfo can be constructed manually (no GPU required).
    let info = DeviceInfo {
        name: "Test GPU".to_string(),
        ordinal: 0,
        compute_capability: (8, 0),
        total_memory_bytes: 80 * 1024 * 1024 * 1024, // 80 GB
        multiprocessor_count: 108,
        max_threads_per_block: 1024,
        max_threads_per_sm: 2048,
        warp_size: 32,
        clock_rate_mhz: 1410.0,
        memory_clock_rate_mhz: 1215.0,
        memory_bus_width_bits: 5120,
        l2_cache_bytes: 40 * 1024 * 1024, // 40 MB
        max_shared_memory_per_block: 49152,
        max_shared_memory_per_sm: 167936,
        max_registers_per_block: 65536,
        ecc_enabled: true,
        tcc_driver: true,
        compute_mode: 0,
        supports_cooperative_launch: true,
        supports_managed_memory: true,
        max_persisting_l2_cache_bytes: 40 * 1024 * 1024,
        async_engine_count: 3,
        supports_memory_pools: true,
        supports_gpu_direct_rdma: true,
        supports_cluster_launch: false,
        supports_concurrent_kernels: true,
        supports_unified_addressing: true,
        max_blocks_per_sm: 32,
        single_to_double_perf_ratio: 2,
    };

    assert_eq!(info.name, "Test GPU");
    assert_eq!(info.compute_capability, (8, 0));
    assert_eq!(info.multiprocessor_count, 108);
    assert_eq!(info.warp_size, 32);
    assert!(info.ecc_enabled);
    assert!(info.supports_cooperative_launch);
    assert!(info.supports_memory_pools);
    assert!(info.supports_concurrent_kernels);
    assert_eq!(info.async_engine_count, 3);
    assert_eq!(info.max_blocks_per_sm, 32);
}

#[test]
fn test_device_info_display() {
    let info = DeviceInfo {
        name: "NVIDIA A100".to_string(),
        ordinal: 0,
        compute_capability: (8, 0),
        total_memory_bytes: 80 * 1024 * 1024 * 1024,
        multiprocessor_count: 108,
        max_threads_per_block: 1024,
        max_threads_per_sm: 2048,
        warp_size: 32,
        clock_rate_mhz: 1410.0,
        memory_clock_rate_mhz: 1215.0,
        memory_bus_width_bits: 5120,
        l2_cache_bytes: 40 * 1024 * 1024,
        max_shared_memory_per_block: 49152,
        max_shared_memory_per_sm: 167936,
        max_registers_per_block: 65536,
        ecc_enabled: true,
        tcc_driver: false,
        compute_mode: 0,
        supports_cooperative_launch: true,
        supports_managed_memory: true,
        max_persisting_l2_cache_bytes: 40 * 1024 * 1024,
        async_engine_count: 3,
        supports_memory_pools: true,
        supports_gpu_direct_rdma: true,
        supports_cluster_launch: false,
        supports_concurrent_kernels: true,
        supports_unified_addressing: true,
        max_blocks_per_sm: 32,
        single_to_double_perf_ratio: 2,
    };

    let display = format!("{info}");
    assert!(display.contains("NVIDIA A100"));
    assert!(display.contains("8.0"));
    assert!(display.contains("108"));
    assert!(display.contains("1024"));
    assert!(display.contains("Managed memory"));
    assert!(display.contains("Memory pools"));
    assert!(display.contains("Async engines"));
}

#[test]
fn test_device_info_debug() {
    let info = DeviceInfo {
        name: "GPU".to_string(),
        ordinal: 0,
        compute_capability: (7, 5),
        total_memory_bytes: 0,
        multiprocessor_count: 0,
        max_threads_per_block: 0,
        max_threads_per_sm: 0,
        warp_size: 0,
        clock_rate_mhz: 0.0,
        memory_clock_rate_mhz: 0.0,
        memory_bus_width_bits: 0,
        l2_cache_bytes: 0,
        max_shared_memory_per_block: 0,
        max_shared_memory_per_sm: 0,
        max_registers_per_block: 0,
        ecc_enabled: false,
        tcc_driver: false,
        compute_mode: 0,
        supports_cooperative_launch: false,
        supports_managed_memory: false,
        max_persisting_l2_cache_bytes: 0,
        async_engine_count: 0,
        supports_memory_pools: false,
        supports_gpu_direct_rdma: false,
        supports_cluster_launch: false,
        supports_concurrent_kernels: false,
        supports_unified_addressing: false,
        max_blocks_per_sm: 0,
        single_to_double_perf_ratio: 0,
    };

    // Debug should include field names
    let dbg = format!("{info:?}");
    assert!(dbg.contains("DeviceInfo"));
    assert!(dbg.contains("compute_capability"));
}

#[test]
fn test_device_info_clone() {
    let info = DeviceInfo {
        name: "GPU".to_string(),
        ordinal: 1,
        compute_capability: (9, 0),
        total_memory_bytes: 1024,
        multiprocessor_count: 1,
        max_threads_per_block: 1,
        max_threads_per_sm: 1,
        warp_size: 32,
        clock_rate_mhz: 1.0,
        memory_clock_rate_mhz: 1.0,
        memory_bus_width_bits: 128,
        l2_cache_bytes: 0,
        max_shared_memory_per_block: 0,
        max_shared_memory_per_sm: 0,
        max_registers_per_block: 0,
        ecc_enabled: false,
        tcc_driver: false,
        compute_mode: 2,
        supports_cooperative_launch: false,
        supports_managed_memory: false,
        max_persisting_l2_cache_bytes: 0,
        async_engine_count: 0,
        supports_memory_pools: false,
        supports_gpu_direct_rdma: false,
        supports_cluster_launch: false,
        supports_concurrent_kernels: false,
        supports_unified_addressing: false,
        max_blocks_per_sm: 0,
        single_to_double_perf_ratio: 0,
    };

    let cloned = info.clone();
    assert_eq!(cloned.name, info.name);
    assert_eq!(cloned.ordinal, info.ordinal);
    assert_eq!(cloned.compute_capability, info.compute_capability);
    assert_eq!(cloned.compute_mode, 2);
}

#[test]
fn test_device_info_from_device() {
    if let Some(dev) = try_device() {
        let result = dev.info();
        assert!(result.is_ok());
        if let Ok(info) = result {
            assert!(!info.name.is_empty(), "device name should not be empty");
            assert!(info.total_memory_bytes > 0, "total memory should be > 0");
            assert!(info.warp_size > 0, "warp size should be > 0");
            assert!(
                info.max_threads_per_block > 0,
                "max threads/block should be > 0"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Graceful error on macOS (no driver)
// ---------------------------------------------------------------------------

#[test]
fn test_methods_return_error_without_gpu() {
    // On macOS (or systems without NVIDIA driver), init() fails.
    // Verify that Device::get(0) returns an error rather than panicking.
    if oxicuda_driver::init().is_err() {
        let result = Device::get(0);
        assert!(result.is_err(), "Device::get should fail without a driver");
    }
}

// ---------------------------------------------------------------------------
// Bool conversion correctness: verify != 0 mapping
// ---------------------------------------------------------------------------

#[test]
fn test_bool_conversion_pattern() {
    // This is a compile-time + documentation test ensuring our bool methods
    // use the `!= 0` pattern (positive values = true, zero = false).
    // We verify by calling every bool method and checking that they all
    // return CudaResult<bool> (already guaranteed by assert_bool_signatures).
    if let Some(dev) = try_device() {
        // All these should succeed; we just confirm the type is bool.
        for result in [
            dev.tcc_driver(),
            dev.supports_generic_compression(),
            dev.supports_pageable_memory_access(),
            dev.pageable_memory_uses_host_page_tables(),
            dev.supports_direct_managed_mem_from_host(),
            dev.supports_host_native_atomics(),
            dev.supports_cooperative_multi_device_launch(),
            dev.supports_flush_remote_writes(),
            dev.supports_host_register(),
            dev.can_use_host_pointer_for_registered_mem(),
            dev.supports_gpu_direct_rdma(),
            dev.supports_tensor_map_access(),
            dev.supports_multicast(),
            dev.mps_enabled(),
            dev.gpu_overlap(),
            dev.supports_deferred_mapping(),
            // New methods
            dev.supports_memory_pools(),
            dev.supports_cluster_launch(),
            dev.supports_virtual_memory_management(),
            dev.supports_gpu_direct_rdma_vmm(),
            dev.supports_timeline_semaphore_interop(),
            dev.supports_mem_sync_domain(),
            dev.supports_gpu_direct_rdma_fabric(),
            dev.supports_unified_function_pointers(),
            dev.supports_ipc_events(),
        ] {
            assert!(result.is_ok(), "bool method failed: {result:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Attribute mapping correctness (uses the generic attribute() method)
// ---------------------------------------------------------------------------

#[test]
fn test_compute_mode_matches_raw_attribute() {
    if let Some(dev) = try_device() {
        use oxicuda_driver::CUdevice_attribute;
        let via_method = dev.compute_mode();
        let via_raw = dev.attribute(CUdevice_attribute::ComputeMode);
        assert_eq!(via_method.ok(), via_raw.ok());
    }
}

#[test]
fn test_max_persisting_l2_matches_raw_attribute() {
    if let Some(dev) = try_device() {
        use oxicuda_driver::CUdevice_attribute;
        let via_method = dev.max_persisting_l2_cache_size();
        let via_raw = dev.attribute(CUdevice_attribute::MaxPersistingL2CacheSize);
        assert_eq!(via_method.ok(), via_raw.ok());
    }
}

#[test]
fn test_max_texture_1d_matches_raw_attribute() {
    if let Some(dev) = try_device() {
        use oxicuda_driver::CUdevice_attribute;
        let via_method = dev.max_texture_1d_width();
        let via_raw = dev.attribute(CUdevice_attribute::MaxTexture1DWidth);
        assert_eq!(via_method.ok(), via_raw.ok());
    }
}

// ---------------------------------------------------------------------------
// New enum variant discriminant tests
// ---------------------------------------------------------------------------

#[test]
fn test_new_device_attribute_discriminants() {
    use oxicuda_driver::CUdevice_attribute;

    // Memory pool / async attributes
    assert_eq!(CUdevice_attribute::MemoryPoolsSupported as i32, 115);
    assert_eq!(CUdevice_attribute::ClusterLaunch as i32, 120);

    // IPC handle types
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

    // RDMA and fabric
    assert_eq!(
        CUdevice_attribute::GpuDirectRdmaWithCudaVmmSupported as i32,
        110
    );
    assert_eq!(CUdevice_attribute::GpuDirectRdmaFabricSupported as i32, 131);

    // Timeline semaphore and sync domain
    assert_eq!(
        CUdevice_attribute::TimelineSemaphoreInteropSupported as i32,
        113
    );
    assert_eq!(CUdevice_attribute::MemSyncDomainSupported as i32, 130);

    // Unified function pointers
    assert_eq!(CUdevice_attribute::UnifiedFunctionPointers as i32, 125);

    // Texture gather
    assert_eq!(CUdevice_attribute::MaxTexture2DGatherWidth as i32, 44);
    assert_eq!(CUdevice_attribute::MaxTexture2DGatherHeight as i32, 45);

    // Alt 3D texture
    assert_eq!(CUdevice_attribute::MaxTexture3DWidthAlt as i32, 47);
    assert_eq!(CUdevice_attribute::MaxTexture3DHeightAlt as i32, 48);
    assert_eq!(CUdevice_attribute::MaxTexture3DDepthAlt as i32, 49);

    // Reserved shared memory
    assert_eq!(CUdevice_attribute::ReservedSharedMemoryPerBlock as i32, 112);

    // Access policy window
    assert_eq!(CUdevice_attribute::AccessPolicyMaxWindowSize as i32, 111);
}

#[test]
fn test_no_discriminant_overlap() {
    use oxicuda_driver::CUdevice_attribute;

    // Collect all variant discriminant values and ensure no duplicates.
    let values: Vec<i32> = vec![
        CUdevice_attribute::MaxThreadsPerBlock as i32,
        CUdevice_attribute::MaxBlockDimX as i32,
        CUdevice_attribute::MaxBlockDimY as i32,
        CUdevice_attribute::MaxBlockDimZ as i32,
        CUdevice_attribute::MaxGridDimX as i32,
        CUdevice_attribute::MaxGridDimY as i32,
        CUdevice_attribute::MaxGridDimZ as i32,
        CUdevice_attribute::MaxSharedMemoryPerBlock as i32,
        CUdevice_attribute::TotalConstantMemory as i32,
        CUdevice_attribute::WarpSize as i32,
        CUdevice_attribute::MaxPitch as i32,
        CUdevice_attribute::MaxRegistersPerBlock as i32,
        CUdevice_attribute::ClockRate as i32,
        CUdevice_attribute::TextureAlignment as i32,
        CUdevice_attribute::GpuOverlap as i32,
        CUdevice_attribute::MultiprocessorCount as i32,
        CUdevice_attribute::KernelExecTimeout as i32,
        CUdevice_attribute::Integrated as i32,
        CUdevice_attribute::CanMapHostMemory as i32,
        CUdevice_attribute::ComputeMode as i32,
        CUdevice_attribute::MaxTexture1DWidth as i32,
        CUdevice_attribute::MaxTexture2DWidth as i32,
        CUdevice_attribute::MaxTexture2DHeight as i32,
        CUdevice_attribute::MaxTexture3DWidth as i32,
        CUdevice_attribute::MaxTexture3DHeight as i32,
        CUdevice_attribute::MaxTexture3DDepth as i32,
        CUdevice_attribute::MaxTexture2DLayeredWidth as i32,
        CUdevice_attribute::MaxTexture2DLayeredHeight as i32,
        CUdevice_attribute::MaxTexture2DLayeredLayers as i32,
        CUdevice_attribute::SurfaceAlignment as i32,
        CUdevice_attribute::ConcurrentKernels as i32,
        CUdevice_attribute::EccEnabled as i32,
        CUdevice_attribute::PciBusId as i32,
        CUdevice_attribute::PciDeviceId as i32,
        CUdevice_attribute::TccDriver as i32,
        CUdevice_attribute::MemoryClockRate as i32,
        CUdevice_attribute::GlobalMemoryBusWidth as i32,
        CUdevice_attribute::L2CacheSize as i32,
        CUdevice_attribute::MaxThreadsPerMultiprocessor as i32,
        CUdevice_attribute::AsyncEngineCount as i32,
        CUdevice_attribute::UnifiedAddressing as i32,
        CUdevice_attribute::MaxTexture1DLayeredWidth as i32,
        CUdevice_attribute::MaxTexture1DLayeredLayers as i32,
        CUdevice_attribute::MaxTexture2DGatherWidth as i32,
        CUdevice_attribute::MaxTexture2DGatherHeight as i32,
        CUdevice_attribute::MaxTexture3DWidthAlt as i32,
        CUdevice_attribute::MaxTexture3DHeightAlt as i32,
        CUdevice_attribute::MaxTexture3DDepthAlt as i32,
        CUdevice_attribute::PciDomainId as i32,
        CUdevice_attribute::TexturePitchAlignment as i32,
        CUdevice_attribute::MaxTextureCubemapWidth as i32,
        CUdevice_attribute::MaxTextureCubemapLayeredWidth as i32,
        CUdevice_attribute::MaxTextureCubemapLayeredLayers as i32,
        CUdevice_attribute::MaxSurface1DWidth as i32,
        CUdevice_attribute::MaxSurface2DWidth as i32,
        CUdevice_attribute::MaxSurface2DHeight as i32,
        CUdevice_attribute::MaxSurface3DWidth as i32,
        CUdevice_attribute::MaxSurface3DHeight as i32,
        CUdevice_attribute::MaxSurface3DDepth as i32,
        CUdevice_attribute::MaxSurfaceCubemapWidth as i32,
        CUdevice_attribute::MaxSurface1DLayeredWidth as i32,
        CUdevice_attribute::MaxSurface1DLayeredLayers as i32,
        CUdevice_attribute::MaxSurface2DLayeredWidth as i32,
        CUdevice_attribute::MaxSurface2DLayeredHeight as i32,
        CUdevice_attribute::MaxSurface2DLayeredLayers as i32,
        CUdevice_attribute::MaxSurfaceCubemapLayeredWidth as i32,
        CUdevice_attribute::MaxSurfaceCubemapLayeredLayers as i32,
        CUdevice_attribute::MaxTexture1DLinearWidth as i32,
        CUdevice_attribute::MaxTexture2DLinearWidth as i32,
        CUdevice_attribute::MaxTexture2DLinearHeight as i32,
        CUdevice_attribute::MaxTexture2DLinearPitch as i32,
        CUdevice_attribute::ComputeCapabilityMajor as i32,
        CUdevice_attribute::ComputeCapabilityMinor as i32,
        CUdevice_attribute::MaxTexture2DMipmappedWidth as i32,
        CUdevice_attribute::MaxTexture2DMipmappedHeight as i32,
        CUdevice_attribute::MaxTexture1DMipmappedWidth as i32,
        CUdevice_attribute::StreamPrioritiesSupported as i32,
        CUdevice_attribute::MaxSharedMemoryPerMultiprocessor as i32,
        CUdevice_attribute::MaxRegistersPerMultiprocessor as i32,
        CUdevice_attribute::ManagedMemory as i32,
        CUdevice_attribute::IsMultiGpuBoard as i32,
        CUdevice_attribute::MultiGpuBoardGroupId as i32,
        CUdevice_attribute::HostNativeAtomicSupported as i32,
        CUdevice_attribute::SingleToDoublePrecisionPerfRatio as i32,
        CUdevice_attribute::PageableMemoryAccess as i32,
        CUdevice_attribute::ConcurrentManagedAccess as i32,
        CUdevice_attribute::ComputePreemptionSupported as i32,
        CUdevice_attribute::CanUseHostPointerForRegisteredMem as i32,
        CUdevice_attribute::CooperativeLaunch as i32,
        CUdevice_attribute::CooperativeMultiDeviceLaunch as i32,
        CUdevice_attribute::MaxSharedMemoryPerBlockOptin as i32,
        CUdevice_attribute::CanFlushRemoteWrites as i32,
        CUdevice_attribute::HostRegisterSupported as i32,
        CUdevice_attribute::PageableMemoryAccessUsesHostPageTables as i32,
        CUdevice_attribute::DirectManagedMemAccessFromHost as i32,
        CUdevice_attribute::VirtualMemoryManagementSupported as i32,
        CUdevice_attribute::HandleTypePosixFileDescriptorSupported as i32,
        CUdevice_attribute::HandleTypeWin32HandleSupported as i32,
        CUdevice_attribute::HandleTypeWin32KmtHandleSupported as i32,
        CUdevice_attribute::MaxBlocksPerMultiprocessor as i32,
        CUdevice_attribute::GenericCompressionSupported as i32,
        CUdevice_attribute::MaxPersistingL2CacheSize as i32,
        CUdevice_attribute::MaxAccessPolicyWindowSize as i32,
        CUdevice_attribute::GpuDirectRdmaWithCudaVmmSupported as i32,
        CUdevice_attribute::AccessPolicyMaxWindowSize as i32,
        CUdevice_attribute::ReservedSharedMemoryPerBlock as i32,
        CUdevice_attribute::TimelineSemaphoreInteropSupported as i32,
        CUdevice_attribute::MemoryPoolsSupported as i32,
        CUdevice_attribute::GpuDirectRdmaSupported as i32,
        CUdevice_attribute::GpuDirectRdmaFlushWritesOptions as i32,
        CUdevice_attribute::GpuDirectRdmaWritesOrdering as i32,
        CUdevice_attribute::MemoryPoolSupportedHandleTypes as i32,
        CUdevice_attribute::ClusterLaunch as i32,
        CUdevice_attribute::DeferredMappingCudaArraySupported as i32,
        CUdevice_attribute::IpcEventSupported as i32,
        CUdevice_attribute::MemSyncDomainCount as i32,
        CUdevice_attribute::TensorMapAccessSupported as i32,
        CUdevice_attribute::UnifiedFunctionPointers as i32,
        CUdevice_attribute::NumaConfig as i32,
        CUdevice_attribute::NumaId as i32,
        CUdevice_attribute::MaxTimelineSemaphoreInteropSupported as i32,
        CUdevice_attribute::MemSyncDomainSupported as i32,
        CUdevice_attribute::GpuDirectRdmaFabricSupported as i32,
        CUdevice_attribute::MulticastSupported as i32,
        CUdevice_attribute::MpsEnabled as i32,
        CUdevice_attribute::HostNumaId as i32,
    ];

    // Check for duplicates
    let mut sorted = values.clone();
    sorted.sort();
    for window in sorted.windows(2) {
        assert_ne!(
            window[0], window[1],
            "duplicate discriminant value: {}",
            window[0]
        );
    }
}

// ---------------------------------------------------------------------------
// New convenience method tests
// ---------------------------------------------------------------------------

#[test]
fn test_supports_memory_pools() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_memory_pools().is_ok());
    }
}

#[test]
fn test_supports_cluster_launch() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_cluster_launch().is_ok());
    }
}

#[test]
fn test_supports_virtual_memory_management() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_virtual_memory_management().is_ok());
    }
}

#[test]
fn test_gpu_direct_rdma_flush_writes_options() {
    if let Some(dev) = try_device() {
        assert!(dev.gpu_direct_rdma_flush_writes_options().is_ok());
    }
}

#[test]
fn test_gpu_direct_rdma_writes_ordering() {
    if let Some(dev) = try_device() {
        assert!(dev.gpu_direct_rdma_writes_ordering().is_ok());
    }
}

#[test]
fn test_max_access_policy_window_size() {
    if let Some(dev) = try_device() {
        let result = dev.max_access_policy_window_size();
        assert!(result.is_ok());
        assert!(result.unwrap_or(-1) >= 0);
    }
}

#[test]
fn test_supports_ipc_events() {
    if let Some(dev) = try_device() {
        assert!(dev.supports_ipc_events().is_ok());
    }
}

#[test]
fn test_numa_config() {
    if let Some(dev) = try_device() {
        assert!(dev.numa_config().is_ok());
    }
}

#[test]
fn test_numa_id() {
    if let Some(dev) = try_device() {
        assert!(dev.numa_id().is_ok());
    }
}

#[test]
fn test_host_numa_id() {
    if let Some(dev) = try_device() {
        assert!(dev.host_numa_id().is_ok());
    }
}

#[test]
fn test_texture_pitch_alignment() {
    if let Some(dev) = try_device() {
        let result = dev.texture_pitch_alignment();
        assert!(result.is_ok());
        assert!(
            result.unwrap_or(-1) > 0,
            "pitch alignment should be positive"
        );
    }
}

#[test]
fn test_mem_sync_domain_count() {
    if let Some(dev) = try_device() {
        assert!(dev.mem_sync_domain_count().is_ok());
    }
}

#[test]
fn test_reserved_shared_memory_per_block() {
    if let Some(dev) = try_device() {
        let result = dev.reserved_shared_memory_per_block();
        assert!(result.is_ok());
        assert!(result.unwrap_or(-1) >= 0);
    }
}

#[test]
fn test_device_info_new_fields() {
    if let Some(dev) = try_device() {
        if let Ok(info) = dev.info() {
            assert!(info.async_engine_count >= 0);
            assert!(info.max_blocks_per_sm >= 0);
            assert!(info.single_to_double_perf_ratio >= 0);
            assert!(info.max_persisting_l2_cache_bytes >= 0);
        }
    }
}

#[test]
fn test_memory_pools_matches_raw_attribute() {
    if let Some(dev) = try_device() {
        use oxicuda_driver::CUdevice_attribute;
        let via_method = dev.supports_memory_pools();
        let via_raw = dev.attribute(CUdevice_attribute::MemoryPoolsSupported);
        if let (Ok(method_val), Ok(raw_val)) = (via_method, via_raw) {
            assert_eq!(method_val, raw_val != 0);
        }
    }
}
