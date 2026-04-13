//! Kernel launch API.
//!
//! Implements `cudaLaunchKernel`, `cudaFuncGetAttributes`, and
//! `cudaFuncSetAttribute` on top of the CUDA Driver API.
//!
//! # Design
//!
//! In the CUDA Runtime, kernels are typically invoked via `<<<...>>>` syntax
//! which the NVCC compiler rewrites into `cudaLaunchKernel` calls.  Since
//! OxiCUDA never uses NVCC, callers must use the driver-level module/function
//! handle pair directly.  This module therefore exposes a slightly lower-level
//! surface that accepts a [`CudaFunction`] instead of a raw symbol pointer.

use std::ffi::c_void;

use oxicuda_driver::ffi::{CUfunction, CUmodule};
use oxicuda_driver::loader::try_driver;

use crate::error::{CudaRtError, CudaRtResult};
use crate::stream::CudaStream;

// ─── Re-exports ───────────────────────────────────────────────────────────────

/// A compiled GPU kernel function (alias for the driver's `CUfunction`).
pub type CudaFunction = CUfunction;

/// A compiled GPU module (alias for the driver's `CUmodule`).
pub type CudaModule = CUmodule;

// ─── Dim3 ────────────────────────────────────────────────────────────────────

/// 3-D grid / block dimensions for kernel launches.
///
/// Mirrors CUDA's `dim3` struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dim3 {
    /// X dimension.
    pub x: u32,
    /// Y dimension.
    pub y: u32,
    /// Z dimension.
    pub z: u32,
}

impl Dim3 {
    /// Construct a 1-D dimension (y = z = 1).
    #[must_use]
    pub const fn one_d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Construct a 2-D dimension (z = 1).
    #[must_use]
    pub const fn two_d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Construct a full 3-D dimension.
    #[must_use]
    pub const fn three_d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total number of threads / blocks.
    #[must_use]
    pub fn volume(self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }
}

impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Self::one_d(x)
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from((x, y): (u32, u32)) -> Self {
        Self::two_d(x, y)
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self::three_d(x, y, z)
    }
}

// ─── FuncAttributes ──────────────────────────────────────────────────────────

/// Attributes of a compiled kernel function.
///
/// Mirrors `cudaFuncAttributes`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FuncAttributes {
    /// Size in bytes of statically-allocated shared memory per block.
    pub shared_size_bytes: usize,
    /// Size in bytes of the constant memory used by the function.
    pub const_size_bytes: usize,
    /// Size in bytes of local memory used by each thread.
    pub local_size_bytes: usize,
    /// Maximum number of threads per block the function can use.
    pub max_threads_per_block: u32,
    /// Number of registers used by each thread.
    pub num_regs: u32,
    /// PTX virtual architecture of the function.
    pub ptx_version: u32,
    /// Binary architecture of the function (same as compute capability × 10).
    pub binary_version: u32,
    /// Cache mode configuration.
    pub cache_mode_ca: bool,
    /// Maximum dynamic shared memory per block.
    pub max_dynamic_shared_size_bytes: usize,
    /// Preferred shared memory carveout.
    pub preferred_shared_memory_carveout: i32,
}

/// Attribute selector for `cudaFuncSetAttribute`.
///
/// Mirrors `cudaFuncAttribute`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FuncAttribute {
    /// Maximum dynamic shared memory size.
    MaxDynamicSharedMemorySize = 8,
    /// Preferred shared memory / L1 carveout (0–100).
    PreferredSharedMemoryCarveout = 9,
}

// ─── Module / function loading ────────────────────────────────────────────────

/// Load a PTX module from a null-terminated byte string.
///
/// Mirrors the driver's `cuModuleLoadDataEx`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn module_load_ptx(ptx: &[u8]) -> CudaRtResult<CudaModule> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut module = CUmodule::default();
    // Ensure null termination.
    let mut ptx_owned;
    let ptx_ptr = if ptx.last().copied() == Some(0) {
        ptx.as_ptr()
    } else {
        ptx_owned = ptx.to_vec();
        ptx_owned.push(0);
        ptx_owned.as_ptr()
    };
    // SAFETY: FFI; ptx_ptr points to null-terminated PTX text.
    let rc = unsafe {
        (api.cu_module_load_data_ex)(
            &raw mut module,
            ptx_ptr as *const c_void,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidPtx));
    }
    Ok(module)
}

/// Get a function handle by name from a loaded module.
///
/// Mirrors the driver's `cuModuleGetFunction`.
///
/// # Errors
///
/// Returns [`CudaRtError::SymbolNotFound`] if the function does not exist.
pub fn module_get_function(module: CudaModule, name: &str) -> CudaRtResult<CudaFunction> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut func = CUfunction::default();
    let name_cstr = std::ffi::CString::new(name).map_err(|_| CudaRtError::InvalidSymbol)?;
    // SAFETY: FFI; name_cstr is null-terminated.
    let rc = unsafe { (api.cu_module_get_function)(&raw mut func, module, name_cstr.as_ptr()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::SymbolNotFound));
    }
    Ok(func)
}

/// Unload a previously loaded module.
///
/// Mirrors `cuModuleUnload`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn module_unload(module: CudaModule) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; module is valid.
    let rc = unsafe { (api.cu_module_unload)(module) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(())
}

// ─── Kernel launch ────────────────────────────────────────────────────────────

/// Launch a CUDA kernel.
///
/// Mirrors `cudaLaunchKernel` (with explicit function handle).
///
/// # Parameters
///
/// - `func` — compiled kernel function (from [`module_get_function`]).
/// - `grid` — grid dimensions.
/// - `block` — block dimensions.
/// - `args` — mutable slice of pointers to kernel arguments; each element
///   must point to the actual argument value, as required by `cuLaunchKernel`.
/// - `shared_mem` — dynamic shared memory in bytes.
/// - `stream` — CUDA stream on which to enqueue the launch.
///
/// # Safety
///
/// - `func` must be a valid kernel handle.
/// - Each `args[i]` pointer must point to a value whose type matches the
///   kernel's `i`-th parameter.
/// - `shared_mem` must not exceed the device's maximum shared memory per block.
///
/// # Errors
///
/// Propagates driver errors.
pub unsafe fn launch_kernel(
    func: CudaFunction,
    grid: Dim3,
    block: Dim3,
    args: &mut [*mut c_void],
    shared_mem: u32,
    stream: CudaStream,
) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; caller guarantees func, args, and stream are valid.
    let rc = unsafe {
        (api.cu_launch_kernel)(
            func,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            shared_mem,
            stream.raw(),
            args.as_mut_ptr(),
            std::ptr::null_mut(), // extra (unused)
        )
    };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::LaunchFailure));
    }
    Ok(())
}

/// Query attributes of a compiled kernel.
///
/// Mirrors `cudaFuncGetAttributes`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn func_get_attributes(func: CudaFunction) -> CudaRtResult<FuncAttributes> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;

    // cu_func_get_attribute is optional (not available on very old drivers).
    let get_attr_fn = api.cu_func_get_attribute.ok_or(CudaRtError::NotSupported)?;
    let attr = |a: oxicuda_driver::ffi::CUfunction_attribute| -> CudaRtResult<i32> {
        let mut v: std::ffi::c_int = 0;
        // SAFETY: FFI.
        let rc = unsafe { get_attr_fn(&raw mut v, a as std::ffi::c_int, func) };
        if rc != 0 {
            return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDeviceFunction));
        }
        Ok(v)
    };

    use oxicuda_driver::ffi::CUfunction_attribute as FA;
    Ok(FuncAttributes {
        shared_size_bytes: attr(FA::SharedSizeBytes)? as usize,
        const_size_bytes: attr(FA::ConstSizeBytes)? as usize,
        local_size_bytes: attr(FA::LocalSizeBytes)? as usize,
        max_threads_per_block: attr(FA::MaxThreadsPerBlock)? as u32,
        num_regs: attr(FA::NumRegs)? as u32,
        ptx_version: attr(FA::PtxVersion)? as u32,
        binary_version: attr(FA::BinaryVersion)? as u32,
        cache_mode_ca: attr(FA::CacheModeCa)? != 0,
        max_dynamic_shared_size_bytes: attr(FA::MaxDynamicSharedSizeBytes)? as usize,
        preferred_shared_memory_carveout: attr(FA::PreferredSharedMemoryCarveout)?,
    })
}

/// Set a kernel attribute.
///
/// Mirrors `cudaFuncSetAttribute`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn func_set_attribute(func: CudaFunction, attr: FuncAttribute, value: i32) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // cu_func_set_attribute is optional (not available on very old drivers).
    let set_attr_fn = api.cu_func_set_attribute.ok_or(CudaRtError::NotSupported)?;
    // SAFETY: FFI.
    let rc = unsafe { set_attr_fn(func, attr as std::ffi::c_int, value) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDeviceFunction));
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim3_one_d() {
        let d = Dim3::one_d(128);
        assert_eq!(d.x, 128);
        assert_eq!(d.y, 1);
        assert_eq!(d.z, 1);
        assert_eq!(d.volume(), 128);
    }

    #[test]
    fn dim3_from_u32() {
        let d: Dim3 = 256u32.into();
        assert_eq!(d.x, 256);
    }

    #[test]
    fn dim3_from_tuple() {
        let d: Dim3 = (32u32, 8u32).into();
        assert_eq!(d.volume(), 256);
        let d3: Dim3 = (4u32, 4u32, 4u32).into();
        assert_eq!(d3.volume(), 64);
    }

    #[test]
    fn dim3_volume() {
        assert_eq!(Dim3::three_d(2, 3, 4).volume(), 24);
    }

    #[test]
    fn module_load_ptx_without_gpu_errors() {
        let ptx = b"// empty\n\0";
        let _ = module_load_ptx(ptx); // must not panic
    }
}
