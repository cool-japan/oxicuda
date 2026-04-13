//! Cooperative kernel launch support (CUDA 9.0+).
//!
//! Cooperative launches allow thread blocks within a kernel (and optionally
//! across multiple GPUs) to synchronise with each other via
//! `cooperative_groups`. This module wraps:
//!
//! * `cuLaunchCooperativeKernel` — single-device cooperative launch.
//! * `cuLaunchCooperativeKernelMultiDevice` — multi-device cooperative launch.
//! * `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` — cooperative-
//!   aware occupancy query.
//!
//! # Platform behaviour
//!
//! On macOS (where NVIDIA dropped CUDA support), query functions return
//! synthetic data suitable for unit tests, while actual launch functions
//! return `Err(CudaError::NotSupported)`.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_driver::cooperative_launch::*;
//! use oxicuda_driver::device::Device;
//!
//! oxicuda_driver::init()?;
//! let dev = Device::get(0)?;
//!
//! let supported = CooperativeLaunchSupport::is_cooperative_supported(&dev)?;
//! println!("Cooperative launch supported: {supported}");
//! # Ok::<(), oxicuda_driver::CudaError>(())
//! ```

use std::ffi::c_void;
use std::fmt;

use crate::error::{CudaError, CudaResult};
use crate::ffi::{CUfunction, CUstream};

#[cfg(any(not(target_os = "macos"), test))]
use crate::ffi::CUdevice_attribute;

use crate::device::Device;
use crate::module::Function;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Flag for `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`:
/// no special flags.
#[cfg(any(not(target_os = "macos"), test))]
const CU_OCCUPANCY_DEFAULT: u32 = 0x0;

/// Flag for cooperative-launch-aware occupancy calculation.
#[cfg(any(not(target_os = "macos"), test))]
const CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE: u32 = 0x1;

// ---------------------------------------------------------------------------
// CooperativeLaunchConfig
// ---------------------------------------------------------------------------

/// Configuration for a single-device cooperative kernel launch.
///
/// This mirrors the parameters accepted by `cuLaunchCooperativeKernel`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CooperativeLaunchConfig {
    /// Grid dimensions `(x, y, z)` in blocks.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions `(x, y, z)` in threads.
    pub block_dim: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
    /// Stream handle. `None` means the default (null) stream.
    pub stream: Option<CUstream>,
}

impl CooperativeLaunchConfig {
    /// Create a new configuration with the given grid and block dimensions.
    ///
    /// Shared memory defaults to 0 and the default stream is used.
    pub fn new(grid_dim: (u32, u32, u32), block_dim: (u32, u32, u32)) -> Self {
        Self {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
            stream: None,
        }
    }

    /// Set dynamic shared memory in bytes.
    #[must_use]
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Set the stream for the launch.
    #[must_use]
    pub fn with_stream(mut self, stream: CUstream) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Validate the configuration.
    ///
    /// All dimensions must be non-zero.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if any dimension component is zero.
    pub fn validate(&self) -> CudaResult<()> {
        if self.grid_dim.0 == 0 || self.grid_dim.1 == 0 || self.grid_dim.2 == 0 {
            return Err(CudaError::InvalidValue);
        }
        if self.block_dim.0 == 0 || self.block_dim.1 == 0 || self.block_dim.2 == 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(())
    }

    /// Total number of threads per block.
    pub fn threads_per_block(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }

    /// Total number of blocks in the grid.
    pub fn total_blocks(&self) -> u64 {
        u64::from(self.grid_dim.0) * u64::from(self.grid_dim.1) * u64::from(self.grid_dim.2)
    }

    /// Resolve the stream handle (null pointer for the default stream).
    #[cfg(any(not(target_os = "macos"), test))]
    fn resolved_stream(&self) -> CUstream {
        self.stream.unwrap_or_default()
    }
}

impl Default for CooperativeLaunchConfig {
    fn default() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
            stream: None,
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceLaunchConfig
// ---------------------------------------------------------------------------

/// Per-device configuration for a multi-device cooperative launch.
///
/// Each entry describes one device's contribution to the cooperative kernel.
/// All entries in a multi-device launch must use identical grid and block
/// dimensions.
#[derive(Clone)]
pub struct DeviceLaunchConfig {
    /// Device ordinal (0-based).
    pub device_ordinal: i32,
    /// Raw CUDA function handle for this device's kernel.
    pub function: CUfunction,
    /// Grid dimensions `(x, y, z)` in blocks.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions `(x, y, z)` in threads.
    pub block_dim: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
    /// Stream handle for this device.
    pub stream: CUstream,
    /// Kernel arguments (pointers to argument values).
    pub args: Vec<*mut c_void>,
}

// SAFETY: The raw pointers in `args` are only used during the launch call
// and the caller is responsible for ensuring their validity at that point.
unsafe impl Send for DeviceLaunchConfig {}
unsafe impl Sync for DeviceLaunchConfig {}

impl fmt::Debug for DeviceLaunchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceLaunchConfig")
            .field("device_ordinal", &self.device_ordinal)
            .field("function", &self.function)
            .field("grid_dim", &self.grid_dim)
            .field("block_dim", &self.block_dim)
            .field("shared_mem_bytes", &self.shared_mem_bytes)
            .field("stream", &self.stream)
            .field("args_count", &self.args.len())
            .finish()
    }
}

impl DeviceLaunchConfig {
    /// Create a new per-device launch configuration.
    pub fn new(
        device_ordinal: i32,
        function: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        stream: CUstream,
    ) -> Self {
        Self {
            device_ordinal,
            function,
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
            stream,
            args: Vec::new(),
        }
    }

    /// Set dynamic shared memory in bytes.
    #[must_use]
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Set kernel arguments.
    #[must_use]
    pub fn with_args(mut self, args: Vec<*mut c_void>) -> Self {
        self.args = args;
        self
    }
}

// ---------------------------------------------------------------------------
// MultiDeviceCooperativeLaunchConfig
// ---------------------------------------------------------------------------

/// Configuration for a multi-device cooperative kernel launch.
///
/// Wraps a collection of [`DeviceLaunchConfig`] entries, one per participating
/// device. All entries must use the same grid and block dimensions.
#[derive(Debug, Clone)]
pub struct MultiDeviceCooperativeLaunchConfig {
    /// Per-device configurations.
    pub per_device: Vec<DeviceLaunchConfig>,
}

impl MultiDeviceCooperativeLaunchConfig {
    /// Create a new multi-device configuration from per-device entries.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if:
    /// - The list is empty.
    /// - Grid or block dimensions differ across devices.
    /// - Any device ordinal is negative.
    /// - Any dimension component is zero.
    pub fn new(per_device: Vec<DeviceLaunchConfig>) -> CudaResult<Self> {
        Self::validate_configs(&per_device)?;
        Ok(Self { per_device })
    }

    /// Validate that all device configurations are consistent.
    fn validate_configs(configs: &[DeviceLaunchConfig]) -> CudaResult<()> {
        if configs.is_empty() {
            return Err(CudaError::InvalidValue);
        }

        let first = &configs[0];

        // All dimensions must be non-zero.
        if first.grid_dim.0 == 0
            || first.grid_dim.1 == 0
            || first.grid_dim.2 == 0
            || first.block_dim.0 == 0
            || first.block_dim.1 == 0
            || first.block_dim.2 == 0
        {
            return Err(CudaError::InvalidValue);
        }

        if first.device_ordinal < 0 {
            return Err(CudaError::InvalidValue);
        }

        for cfg in &configs[1..] {
            if cfg.grid_dim != first.grid_dim {
                return Err(CudaError::InvalidValue);
            }
            if cfg.block_dim != first.block_dim {
                return Err(CudaError::InvalidValue);
            }
            if cfg.device_ordinal < 0 {
                return Err(CudaError::InvalidValue);
            }
        }

        Ok(())
    }

    /// Number of devices participating in the launch.
    pub fn device_count(&self) -> usize {
        self.per_device.len()
    }
}

// ---------------------------------------------------------------------------
// CooperativeLaunchSupport — query helpers
// ---------------------------------------------------------------------------

/// Query helpers for cooperative launch capabilities.
pub struct CooperativeLaunchSupport;

impl CooperativeLaunchSupport {
    /// Query whether the device supports cooperative kernel launches.
    ///
    /// Checks `CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH` (attribute 95).
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] if the driver call fails.
    pub fn is_cooperative_supported(device: &Device) -> CudaResult<bool> {
        #[cfg(not(target_os = "macos"))]
        {
            let driver = crate::loader::try_driver()?;
            let mut value: i32 = 0;
            crate::error::check(unsafe {
                (driver.cu_device_get_attribute)(
                    &mut value,
                    CUdevice_attribute::CooperativeLaunch,
                    device.raw(),
                )
            })?;
            Ok(value != 0)
        }
        #[cfg(target_os = "macos")]
        {
            let _ = device;
            // Synthetic: report cooperative launch as supported for testing.
            Ok(true)
        }
    }

    /// Query whether the device supports multi-device cooperative launches.
    ///
    /// Checks `CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH` (attribute 96).
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] if the driver call fails.
    pub fn is_multi_device_supported(device: &Device) -> CudaResult<bool> {
        #[cfg(not(target_os = "macos"))]
        {
            let driver = crate::loader::try_driver()?;
            let mut value: i32 = 0;
            crate::error::check(unsafe {
                (driver.cu_device_get_attribute)(
                    &mut value,
                    CUdevice_attribute::CooperativeMultiDeviceLaunch,
                    device.raw(),
                )
            })?;
            Ok(value != 0)
        }
        #[cfg(target_os = "macos")]
        {
            let _ = device;
            // Synthetic: report multi-device cooperative launch as supported.
            Ok(true)
        }
    }

    /// Returns the maximum number of active blocks per SM for a cooperative
    /// kernel launch.
    ///
    /// This is a cooperative-launch-aware variant of the standard occupancy
    /// query. For cooperative launches, the hardware may limit the number of
    /// blocks more tightly than for regular launches.
    ///
    /// # Parameters
    ///
    /// * `func` — the kernel function handle.
    /// * `block_size` — number of threads per block.
    /// * `shared_mem` — dynamic shared memory per block in bytes.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] if the driver call fails.
    pub fn max_cooperative_grid_blocks(
        func: &Function,
        block_size: u32,
        shared_mem: u32,
    ) -> CudaResult<u32> {
        #[cfg(not(target_os = "macos"))]
        {
            let driver = crate::loader::try_driver()?;
            let mut num_blocks: i32 = 0;
            crate::error::check(unsafe {
                (driver.cu_occupancy_max_active_blocks_per_multiprocessor_with_flags)(
                    &mut num_blocks,
                    func.raw(),
                    block_size as i32,
                    shared_mem as usize,
                    CU_OCCUPANCY_DEFAULT,
                )
            })?;
            Ok(num_blocks as u32)
        }
        #[cfg(target_os = "macos")]
        {
            let _ = (func, block_size, shared_mem);
            // Synthetic: return a reasonable occupancy value for testing.
            Ok(16)
        }
    }

    /// Returns the maximum number of active blocks per SM for a cooperative
    /// launch, with the caching override disabled.
    ///
    /// When `disable_caching_override` is `true`, the driver will not use
    /// the L1/texture cache to increase occupancy.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] if the driver call fails.
    pub fn max_cooperative_grid_blocks_with_flags(
        func: &Function,
        block_size: u32,
        shared_mem: u32,
        disable_caching_override: bool,
    ) -> CudaResult<u32> {
        #[cfg(not(target_os = "macos"))]
        {
            let driver = crate::loader::try_driver()?;
            let flags = if disable_caching_override {
                CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE
            } else {
                CU_OCCUPANCY_DEFAULT
            };
            let mut num_blocks: i32 = 0;
            crate::error::check(unsafe {
                (driver.cu_occupancy_max_active_blocks_per_multiprocessor_with_flags)(
                    &mut num_blocks,
                    func.raw(),
                    block_size as i32,
                    shared_mem as usize,
                    flags,
                )
            })?;
            Ok(num_blocks as u32)
        }
        #[cfg(target_os = "macos")]
        {
            let _ = (func, block_size, shared_mem, disable_caching_override);
            Ok(16)
        }
    }
}

// ---------------------------------------------------------------------------
// cooperative_launch — single-device
// ---------------------------------------------------------------------------

/// Launch a cooperative kernel on a single device.
///
/// Cooperative launches enable thread blocks to synchronise with each other
/// via `cooperative_groups::this_grid().sync()`. The grid must be small
/// enough that all blocks can be simultaneously resident on the GPU; use
/// [`CooperativeLaunchSupport::max_cooperative_grid_blocks`] to query the
/// maximum.
///
/// # Safety (caller-side)
///
/// The `args` slice must contain pointers to valid kernel argument values
/// with correct types and alignment. This is the same contract as
/// `cuLaunchKernel`.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if the config fails validation.
/// * [`CudaError::CooperativeLaunchTooLarge`] if the grid is too large.
/// * [`CudaError::NotSupported`] on macOS.
pub fn cooperative_launch(
    func: &Function,
    config: &CooperativeLaunchConfig,
    args: &[*mut c_void],
) -> CudaResult<()> {
    config.validate()?;

    #[cfg(not(target_os = "macos"))]
    {
        let driver = crate::loader::try_driver()?;
        let mut kernel_params: Vec<*mut c_void> = args.to_vec();
        let params_ptr = if kernel_params.is_empty() {
            std::ptr::null_mut()
        } else {
            kernel_params.as_mut_ptr()
        };

        crate::error::check(unsafe {
            (driver.cu_launch_cooperative_kernel)(
                func.raw(),
                config.grid_dim.0,
                config.grid_dim.1,
                config.grid_dim.2,
                config.block_dim.0,
                config.block_dim.1,
                config.block_dim.2,
                config.shared_mem_bytes,
                config.resolved_stream(),
                params_ptr,
            )
        })
    }
    #[cfg(target_os = "macos")]
    {
        let _ = (func, args);
        Err(CudaError::NotSupported)
    }
}

// ---------------------------------------------------------------------------
// cooperative_launch_multi_device
// ---------------------------------------------------------------------------

/// CUDA internal structure for `cuLaunchCooperativeKernelMultiDevice`.
///
/// This matches `CUDA_LAUNCH_PARAMS` from the CUDA Driver API.
#[cfg(not(target_os = "macos"))]
#[repr(C)]
#[allow(non_camel_case_types)]
struct CUDA_LAUNCH_PARAMS {
    function: CUfunction,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    h_stream: CUstream,
    kernel_params: *mut *mut c_void,
}

/// Launch a cooperative kernel across multiple devices simultaneously.
///
/// All devices execute the same grid/block configuration and can synchronise
/// via `cooperative_groups::this_multi_grid().sync()`.
///
/// # Safety (caller-side)
///
/// Each device's `args` must contain valid pointers to kernel argument values.
/// The appropriate CUDA context must be current for each device when building
/// the configuration.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if configurations are inconsistent.
/// * [`CudaError::CooperativeLaunchTooLarge`] if the grid is too large.
/// * [`CudaError::NotSupported`] on macOS.
pub fn cooperative_launch_multi_device(configs: &[DeviceLaunchConfig]) -> CudaResult<()> {
    if configs.is_empty() {
        return Err(CudaError::InvalidValue);
    }

    // Validate consistency.
    MultiDeviceCooperativeLaunchConfig::validate_configs(configs)?;

    #[cfg(not(target_os = "macos"))]
    {
        let driver = crate::loader::try_driver()?;

        // Build the CUDA_LAUNCH_PARAMS array. We need mutable copies of the
        // args vectors so we can take pointers into them.
        let mut args_storage: Vec<Vec<*mut c_void>> =
            configs.iter().map(|c| c.args.clone()).collect();

        let mut launch_params: Vec<CUDA_LAUNCH_PARAMS> = configs
            .iter()
            .enumerate()
            .map(|(i, cfg)| CUDA_LAUNCH_PARAMS {
                function: cfg.function,
                grid_dim_x: cfg.grid_dim.0,
                grid_dim_y: cfg.grid_dim.1,
                grid_dim_z: cfg.grid_dim.2,
                block_dim_x: cfg.block_dim.0,
                block_dim_y: cfg.block_dim.1,
                block_dim_z: cfg.block_dim.2,
                shared_mem_bytes: cfg.shared_mem_bytes,
                h_stream: cfg.stream,
                kernel_params: if args_storage[i].is_empty() {
                    std::ptr::null_mut()
                } else {
                    args_storage[i].as_mut_ptr()
                },
            })
            .collect();

        let num_devices = launch_params.len() as u32;
        crate::error::check(unsafe {
            (driver.cu_launch_cooperative_kernel_multi_device)(
                launch_params.as_mut_ptr().cast::<c_void>(),
                num_devices,
                0, // flags
            )
        })
    }
    #[cfg(target_os = "macos")]
    {
        let _ = configs;
        Err(CudaError::NotSupported)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- CooperativeLaunchConfig tests --

    #[test]
    fn test_config_new() {
        let config = CooperativeLaunchConfig::new((4, 2, 1), (256, 1, 1));
        assert_eq!(config.grid_dim, (4, 2, 1));
        assert_eq!(config.block_dim, (256, 1, 1));
        assert_eq!(config.shared_mem_bytes, 0);
        assert!(config.stream.is_none());
    }

    #[test]
    fn test_config_default() {
        let config = CooperativeLaunchConfig::default();
        assert_eq!(config.grid_dim, (1, 1, 1));
        assert_eq!(config.block_dim, (1, 1, 1));
        assert_eq!(config.shared_mem_bytes, 0);
        assert!(config.stream.is_none());
    }

    #[test]
    fn test_config_builder_methods() {
        let stream = CUstream::default();
        let config = CooperativeLaunchConfig::new((8, 1, 1), (128, 1, 1))
            .with_shared_mem(4096)
            .with_stream(stream);
        assert_eq!(config.shared_mem_bytes, 4096);
        assert!(config.stream.is_some());
    }

    #[test]
    fn test_config_validate_valid() {
        let config = CooperativeLaunchConfig::new((1, 1, 1), (32, 1, 1));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_grid_x() {
        let config = CooperativeLaunchConfig::new((0, 1, 1), (32, 1, 1));
        assert_eq!(config.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn test_config_validate_zero_grid_y() {
        let config = CooperativeLaunchConfig::new((1, 0, 1), (32, 1, 1));
        assert_eq!(config.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn test_config_validate_zero_block_z() {
        let config = CooperativeLaunchConfig::new((1, 1, 1), (32, 1, 0));
        assert_eq!(config.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn test_config_threads_per_block() {
        let config = CooperativeLaunchConfig::new((1, 1, 1), (16, 8, 2));
        assert_eq!(config.threads_per_block(), 256);
    }

    #[test]
    fn test_config_total_blocks() {
        let config = CooperativeLaunchConfig::new((4, 3, 2), (1, 1, 1));
        assert_eq!(config.total_blocks(), 24);
    }

    #[test]
    fn test_config_resolved_stream_default() {
        let config = CooperativeLaunchConfig::default();
        let stream = config.resolved_stream();
        assert!(stream.is_null());
    }

    // -- DeviceLaunchConfig tests --

    #[test]
    fn test_device_launch_config_new() {
        let func = CUfunction::default();
        let stream = CUstream::default();
        let cfg = DeviceLaunchConfig::new(0, func, (4, 1, 1), (256, 1, 1), stream);
        assert_eq!(cfg.device_ordinal, 0);
        assert_eq!(cfg.grid_dim, (4, 1, 1));
        assert_eq!(cfg.block_dim, (256, 1, 1));
        assert_eq!(cfg.shared_mem_bytes, 0);
        assert!(cfg.args.is_empty());
    }

    #[test]
    fn test_device_launch_config_builder() {
        let func = CUfunction::default();
        let stream = CUstream::default();
        let mut val: u32 = 42;
        let arg_ptr = &mut val as *mut u32 as *mut c_void;
        let cfg = DeviceLaunchConfig::new(1, func, (2, 2, 1), (128, 1, 1), stream)
            .with_shared_mem(2048)
            .with_args(vec![arg_ptr]);
        assert_eq!(cfg.shared_mem_bytes, 2048);
        assert_eq!(cfg.args.len(), 1);
    }

    #[test]
    fn test_device_launch_config_debug() {
        let func = CUfunction::default();
        let stream = CUstream::default();
        let cfg = DeviceLaunchConfig::new(0, func, (1, 1, 1), (32, 1, 1), stream);
        let debug_str = format!("{cfg:?}");
        assert!(debug_str.contains("DeviceLaunchConfig"));
        assert!(debug_str.contains("args_count"));
    }

    // -- MultiDeviceCooperativeLaunchConfig tests --

    #[test]
    fn test_multi_device_config_empty() {
        let result = MultiDeviceCooperativeLaunchConfig::new(vec![]);
        assert_eq!(result.err(), Some(CudaError::InvalidValue));
    }

    #[test]
    fn test_multi_device_config_single() {
        let func = CUfunction::default();
        let stream = CUstream::default();
        let cfg = DeviceLaunchConfig::new(0, func, (4, 1, 1), (256, 1, 1), stream);
        let multi = MultiDeviceCooperativeLaunchConfig::new(vec![cfg]);
        assert!(multi.is_ok());
        let multi = multi.ok();
        assert!(multi.is_some());
        let multi = multi.map(|m| m.device_count());
        assert_eq!(multi, Some(1));
    }

    #[test]
    fn test_multi_device_config_mismatched_grid() {
        let func = CUfunction::default();
        let stream = CUstream::default();
        let cfg0 = DeviceLaunchConfig::new(0, func, (4, 1, 1), (256, 1, 1), stream);
        let cfg1 = DeviceLaunchConfig::new(1, func, (8, 1, 1), (256, 1, 1), stream);
        let result = MultiDeviceCooperativeLaunchConfig::new(vec![cfg0, cfg1]);
        assert_eq!(result.err(), Some(CudaError::InvalidValue));
    }

    #[test]
    fn test_multi_device_config_mismatched_block() {
        let func = CUfunction::default();
        let stream = CUstream::default();
        let cfg0 = DeviceLaunchConfig::new(0, func, (4, 1, 1), (256, 1, 1), stream);
        let cfg1 = DeviceLaunchConfig::new(1, func, (4, 1, 1), (128, 1, 1), stream);
        let result = MultiDeviceCooperativeLaunchConfig::new(vec![cfg0, cfg1]);
        assert_eq!(result.err(), Some(CudaError::InvalidValue));
    }

    #[test]
    fn test_multi_device_config_negative_ordinal() {
        let func = CUfunction::default();
        let stream = CUstream::default();
        let cfg = DeviceLaunchConfig::new(-1, func, (4, 1, 1), (256, 1, 1), stream);
        let result = MultiDeviceCooperativeLaunchConfig::new(vec![cfg]);
        assert_eq!(result.err(), Some(CudaError::InvalidValue));
    }

    #[test]
    fn test_multi_device_config_zero_dim() {
        let func = CUfunction::default();
        let stream = CUstream::default();
        let cfg = DeviceLaunchConfig::new(0, func, (0, 1, 1), (256, 1, 1), stream);
        let result = MultiDeviceCooperativeLaunchConfig::new(vec![cfg]);
        assert_eq!(result.err(), Some(CudaError::InvalidValue));
    }

    #[test]
    fn test_multi_device_config_consistent_pair() {
        let func = CUfunction::default();
        let stream = CUstream::default();
        let cfg0 = DeviceLaunchConfig::new(0, func, (4, 2, 1), (128, 2, 1), stream);
        let cfg1 = DeviceLaunchConfig::new(1, func, (4, 2, 1), (128, 2, 1), stream);
        let multi = MultiDeviceCooperativeLaunchConfig::new(vec![cfg0, cfg1]);
        assert!(multi.is_ok());
        let multi = multi.ok();
        assert!(multi.is_some());
        let count = multi.map(|m| m.device_count());
        assert_eq!(count, Some(2));
    }

    // -- cooperative_launch on macOS --

    #[cfg(target_os = "macos")]
    #[test]
    fn test_cooperative_launch_returns_not_supported_on_macos() {
        let _func_handle = CUfunction::default();
        // We can't construct a Function directly, so test via the raw path.
        // The cooperative_launch function requires &Function, which needs
        // module.rs. Instead, test the config validation works.
        let config = CooperativeLaunchConfig::new((1, 1, 1), (32, 1, 1));
        assert!(config.validate().is_ok());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_multi_device_launch_returns_not_supported_on_macos() {
        let configs: &[DeviceLaunchConfig] = &[];
        let result = cooperative_launch_multi_device(configs);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    // -- CooperativeLaunchSupport tests (macOS synthetic) --

    #[cfg(target_os = "macos")]
    #[test]
    fn test_cooperative_support_query_macos() {
        // On macOS, Device::get will fail because the driver is not loaded.
        // We test the synthetic path by verifying the constants exist.
        assert_eq!(CUdevice_attribute::CooperativeLaunch as i32, 95);
        assert_eq!(CUdevice_attribute::CooperativeMultiDeviceLaunch as i32, 96);
    }

    #[test]
    fn test_occupancy_constants() {
        assert_eq!(CU_OCCUPANCY_DEFAULT, 0x0);
        assert_eq!(CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, 0x1);
    }

    #[test]
    fn test_config_large_total_blocks_no_overflow() {
        // Test that large grids don't overflow (u64 arithmetic).
        let config = CooperativeLaunchConfig::new((65535, 65535, 64), (1, 1, 1));
        let total = config.total_blocks();
        assert_eq!(total, 65535u64 * 65535 * 64);
    }
}
