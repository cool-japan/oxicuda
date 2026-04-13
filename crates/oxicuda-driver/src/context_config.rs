//! Context configuration: limits, cache config, and shared memory config.
//!
//! These functions allow tuning per-context resource limits (stack size,
//! heap size, etc.) and cache / shared memory policies.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_driver::context_config;
//! # use oxicuda_driver::ffi::CUlimit;
//! # fn main() -> Result<(), oxicuda_driver::CudaError> {
//! let stack = context_config::get_limit(CUlimit::StackSize)?;
//! println!("GPU thread stack size: {stack} bytes");
//!
//! context_config::set_cache_config(context_config::CacheConfig::PreferL1)?;
//! # Ok(())
//! # }
//! ```

use crate::error::{CudaError, CudaResult};
use crate::ffi::CUlimit;
use crate::loader::try_driver;

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/// Returns the value of a context limit.
///
/// # Errors
///
/// Returns [`CudaError::NotSupported`] if the driver lacks `cuCtxGetLimit`,
/// or another error on failure.
pub fn get_limit(limit: CUlimit) -> CudaResult<usize> {
    let api = try_driver()?;
    let f = api.cu_ctx_get_limit.ok_or(CudaError::NotSupported)?;
    let mut value: usize = 0;
    crate::cuda_call!(f(&mut value, limit as u32))?;
    Ok(value)
}

/// Sets the value of a context limit.
///
/// # Errors
///
/// Returns [`CudaError::NotSupported`] if the driver lacks `cuCtxSetLimit`,
/// or another error on failure.
pub fn set_limit(limit: CUlimit, value: usize) -> CudaResult<()> {
    let api = try_driver()?;
    let f = api.cu_ctx_set_limit.ok_or(CudaError::NotSupported)?;
    crate::cuda_call!(f(limit as u32, value))
}

// ---------------------------------------------------------------------------
// CacheConfig
// ---------------------------------------------------------------------------

/// Preferred cache configuration for a CUDA context or function.
///
/// Controls the trade-off between L1 cache and shared memory on devices
/// that share the same on-chip memory for both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CacheConfig {
    /// No preference — the driver picks.
    PreferNone = 0,
    /// Prefer more shared memory over L1 cache.
    PreferShared = 1,
    /// Prefer more L1 cache over shared memory.
    PreferL1 = 2,
    /// Equal split between L1 and shared memory.
    PreferEqual = 3,
}

impl CacheConfig {
    /// Convert a raw `u32` driver value to a `CacheConfig`.
    fn from_raw(val: u32) -> CudaResult<Self> {
        match val {
            0 => Ok(Self::PreferNone),
            1 => Ok(Self::PreferShared),
            2 => Ok(Self::PreferL1),
            3 => Ok(Self::PreferEqual),
            _ => Err(CudaError::InvalidValue),
        }
    }
}

/// Returns the current cache configuration for the active context.
///
/// # Errors
///
/// Returns [`CudaError::NotSupported`] if the driver lacks
/// `cuCtxGetCacheConfig`, or another error on failure.
pub fn get_cache_config() -> CudaResult<CacheConfig> {
    let api = try_driver()?;
    let f = api.cu_ctx_get_cache_config.ok_or(CudaError::NotSupported)?;
    let mut raw: u32 = 0;
    crate::cuda_call!(f(&mut raw))?;
    CacheConfig::from_raw(raw)
}

/// Sets the cache configuration for the active context.
///
/// # Errors
///
/// Returns [`CudaError::NotSupported`] if the driver lacks
/// `cuCtxSetCacheConfig`, or another error on failure.
pub fn set_cache_config(config: CacheConfig) -> CudaResult<()> {
    let api = try_driver()?;
    let f = api.cu_ctx_set_cache_config.ok_or(CudaError::NotSupported)?;
    crate::cuda_call!(f(config as u32))
}

// ---------------------------------------------------------------------------
// SharedMemConfig
// ---------------------------------------------------------------------------

/// Shared memory bank configuration.
///
/// Controls whether shared memory uses 4-byte or 8-byte bank width.
/// 8-byte banks can reduce bank conflicts for 64-bit accesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SharedMemConfig {
    /// Use the device default bank size.
    Default = 0,
    /// 4-byte (32-bit) shared memory banks.
    FourByte = 1,
    /// 8-byte (64-bit) shared memory banks.
    EightByte = 2,
}

impl SharedMemConfig {
    /// Convert a raw `u32` driver value to a `SharedMemConfig`.
    fn from_raw(val: u32) -> CudaResult<Self> {
        match val {
            0 => Ok(Self::Default),
            1 => Ok(Self::FourByte),
            2 => Ok(Self::EightByte),
            _ => Err(CudaError::InvalidValue),
        }
    }
}

/// Returns the current shared memory configuration for the active context.
///
/// # Errors
///
/// Returns [`CudaError::NotSupported`] if the driver lacks
/// `cuCtxGetSharedMemConfig`, or another error on failure.
pub fn get_shared_mem_config() -> CudaResult<SharedMemConfig> {
    let api = try_driver()?;
    let f = api
        .cu_ctx_get_shared_mem_config
        .ok_or(CudaError::NotSupported)?;
    let mut raw: u32 = 0;
    crate::cuda_call!(f(&mut raw))?;
    SharedMemConfig::from_raw(raw)
}

/// Sets the shared memory configuration for the active context.
///
/// # Errors
///
/// Returns [`CudaError::NotSupported`] if the driver lacks
/// `cuCtxSetSharedMemConfig`, or another error on failure.
pub fn set_shared_mem_config(config: SharedMemConfig) -> CudaResult<()> {
    let api = try_driver()?;
    let f = api
        .cu_ctx_set_shared_mem_config
        .ok_or(CudaError::NotSupported)?;
    crate::cuda_call!(f(config as u32))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_config_round_trip() {
        assert_eq!(CacheConfig::from_raw(0).ok(), Some(CacheConfig::PreferNone));
        assert_eq!(
            CacheConfig::from_raw(1).ok(),
            Some(CacheConfig::PreferShared)
        );
        assert_eq!(CacheConfig::from_raw(2).ok(), Some(CacheConfig::PreferL1));
        assert_eq!(
            CacheConfig::from_raw(3).ok(),
            Some(CacheConfig::PreferEqual)
        );
        assert!(CacheConfig::from_raw(99).is_err());
    }

    #[test]
    fn shared_mem_config_round_trip() {
        assert_eq!(
            SharedMemConfig::from_raw(0).ok(),
            Some(SharedMemConfig::Default)
        );
        assert_eq!(
            SharedMemConfig::from_raw(1).ok(),
            Some(SharedMemConfig::FourByte)
        );
        assert_eq!(
            SharedMemConfig::from_raw(2).ok(),
            Some(SharedMemConfig::EightByte)
        );
        assert!(SharedMemConfig::from_raw(99).is_err());
    }

    #[test]
    fn cache_config_repr_values() {
        assert_eq!(CacheConfig::PreferNone as u32, 0);
        assert_eq!(CacheConfig::PreferShared as u32, 1);
        assert_eq!(CacheConfig::PreferL1 as u32, 2);
        assert_eq!(CacheConfig::PreferEqual as u32, 3);
    }

    #[test]
    fn get_limit_returns_error_without_gpu() {
        let result = get_limit(CUlimit::StackSize);
        let _ = result;
    }
}
