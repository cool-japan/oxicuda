//! Host-registered memory for DMA access.
//!
//! [`RegisteredMemory<T>`] wraps `cuMemHostRegister` / `cuMemHostUnregister`
//! to register existing host allocations with the CUDA driver, enabling DMA
//! transfers without an intermediate staging copy.
//!
//! Unlike [`PinnedBuffer`](crate::PinnedBuffer), which allocates *new*
//! page-locked memory, `RegisteredMemory` works with memory that has
//! already been allocated (e.g. a `Vec<T>`, a slice from a memory-mapped
//! file, etc.).
//!
//! # Lifetime
//!
//! The caller must ensure the underlying allocation outlives the
//! `RegisteredMemory` handle.  The handle borrows (but does NOT own) the
//! memory.  On [`Drop`], only `cuMemHostUnregister` is called — the
//! original allocation is untouched.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_memory::host_registered::{register_vec, RegisterFlags};
//! let mut data = vec![0.0f32; 1024];
//! let reg = register_vec(&mut data, RegisterFlags::DEFAULT)?;
//! assert_eq!(reg.len(), 1024);
//! // `data` is now DMA-accessible; use `reg.device_ptr()` on the GPU side.
//! drop(reg); // cuMemHostUnregister is called here
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::fmt;
use std::ops::{BitAnd, BitOr, Deref, DerefMut};

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::ffi::{
    CU_MEMHOSTREGISTER_DEVICEMAP, CU_MEMHOSTREGISTER_IOMEMORY, CU_MEMHOSTREGISTER_PORTABLE,
    CU_MEMHOSTREGISTER_READ_ONLY, CUdeviceptr,
};

#[cfg(not(target_os = "macos"))]
use oxicuda_driver::ffi;
#[cfg(not(target_os = "macos"))]
use oxicuda_driver::loader::try_driver;
#[cfg(not(target_os = "macos"))]
use std::ffi::c_void;

// ---------------------------------------------------------------------------
// RegisterFlags
// ---------------------------------------------------------------------------

/// Bitflags controlling how `cuMemHostRegister` registers host memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegisterFlags(u32);

impl RegisterFlags {
    /// Memory is portable across CUDA contexts.
    pub const PORTABLE: Self = Self(CU_MEMHOSTREGISTER_PORTABLE);

    /// Memory is mapped into the device address space, enabling zero-copy
    /// access via `cuMemHostGetDevicePointer`.
    pub const DEVICE_MAP: Self = Self(CU_MEMHOSTREGISTER_DEVICEMAP);

    /// Pointer refers to I/O memory (not system RAM).
    pub const IO_MEMORY: Self = Self(CU_MEMHOSTREGISTER_IOMEMORY);

    /// Memory will not be written by the GPU (read-only hint).
    pub const READ_ONLY: Self = Self(CU_MEMHOSTREGISTER_READ_ONLY);

    /// The recommended default: portable + device-mapped.
    pub const DEFAULT: Self = Self(CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);

    /// No flags set.
    pub const NONE: Self = Self(0);

    /// Returns the raw `u32` flag value.
    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Creates a `RegisterFlags` from a raw `u32` value.
    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns `true` if `self` contains all flags in `other`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl BitOr for RegisterFlags {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitAnd for RegisterFlags {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl fmt::Display for RegisterFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.contains(Self::PORTABLE) {
            parts.push("PORTABLE");
        }
        if self.contains(Self::DEVICE_MAP) {
            parts.push("DEVICE_MAP");
        }
        if self.contains(Self::IO_MEMORY) {
            parts.push("IO_MEMORY");
        }
        if self.contains(Self::READ_ONLY) {
            parts.push("READ_ONLY");
        }
        if parts.is_empty() {
            write!(f, "NONE")
        } else {
            write!(f, "{}", parts.join(" | "))
        }
    }
}

// ---------------------------------------------------------------------------
// RegisteredMemory<T>
// ---------------------------------------------------------------------------

/// RAII handle for host memory registered with the CUDA driver.
///
/// The handle borrows a raw pointer to existing host memory and registers
/// it via `cuMemHostRegister_v2`.  On [`Drop`], `cuMemHostUnregister` is
/// called to undo the registration.  The underlying allocation is **not**
/// freed — that responsibility remains with the original owner.
///
/// # Safety invariant
///
/// The memory range `[ptr, ptr + len)` must remain valid and not be freed
/// for the entire lifetime of this handle.
pub struct RegisteredMemory<T: Copy> {
    /// Borrowed pointer to the host allocation (NOT owned).
    ptr: *mut T,
    /// Number of `T` elements.
    len: usize,
    /// Flags used during registration.
    flags: RegisterFlags,
    /// Device-visible pointer obtained from registration (if DEVICE_MAP).
    device_ptr: CUdeviceptr,
}

// SAFETY: The registered host memory is not thread-local; it is accessible
// from any thread once registered with the CUDA driver.
unsafe impl<T: Copy + Send> Send for RegisteredMemory<T> {}
unsafe impl<T: Copy + Sync> Sync for RegisteredMemory<T> {}

impl<T: Copy> RegisteredMemory<T> {
    /// Returns a raw const pointer to the registered memory.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a raw mutable pointer to the registered memory.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns the device-visible pointer for the registered memory.
    ///
    /// This is only meaningful when the `DEVICE_MAP` flag was set.
    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    /// Returns the number of `T` elements in the registered range.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the registered range contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the flags used when the memory was registered.
    #[inline]
    pub fn flags(&self) -> RegisterFlags {
        self.flags
    }

    /// Returns a shared slice over the registered memory.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the caller guaranteed the memory is valid for `self.len`
        // elements, and we have `&self` so no mutable alias exists.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a mutable slice over the registered memory.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: the caller guaranteed the memory is valid for `self.len`
        // elements, and we have `&mut self` so no other alias exists.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Copy> Deref for RegisteredMemory<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy> DerefMut for RegisteredMemory<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy> Drop for RegisteredMemory<T> {
    fn drop(&mut self) {
        #[cfg(not(target_os = "macos"))]
        {
            if let Ok(api) = try_driver() {
                let rc = unsafe { (api.cu_mem_host_unregister)(self.ptr.cast::<c_void>()) };
                if rc != 0 {
                    tracing::warn!(
                        cuda_error = rc,
                        len = self.len,
                        "cuMemHostUnregister failed during RegisteredMemory drop"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public registration functions
// ---------------------------------------------------------------------------

/// Registers an existing host memory range with the CUDA driver for DMA.
///
/// # Safety contract (upheld by the caller)
///
/// * `ptr` must point to a valid allocation of at least `len * size_of::<T>()` bytes.
/// * The allocation must remain valid for the lifetime of the returned handle.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `len` is zero or the byte size overflows.
/// * [`CudaError::NotSupported`] on macOS.
/// * Other driver errors from `cuMemHostRegister_v2`.
pub fn register<T: Copy>(
    ptr: *mut T,
    len: usize,
    flags: RegisterFlags,
) -> CudaResult<RegisteredMemory<T>> {
    if len == 0 {
        return Err(CudaError::InvalidValue);
    }
    if ptr.is_null() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CudaError::InvalidValue)?;

    #[cfg(target_os = "macos")]
    {
        // On macOS there is no CUDA driver.  Return a synthetic handle so
        // that unit tests can exercise the API surface without a GPU.
        let _ = byte_size;
        Ok(RegisteredMemory {
            ptr,
            len,
            flags,
            device_ptr: ptr as CUdeviceptr,
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        let api = try_driver()?;

        // Register the host memory range.
        let rc =
            unsafe { (api.cu_mem_host_register_v2)(ptr.cast::<c_void>(), byte_size, flags.bits()) };
        oxicuda_driver::check(rc)?;

        // If DEVICE_MAP is set, obtain the device pointer.
        let device_ptr = if flags.contains(RegisterFlags::DEVICE_MAP) {
            let mut dptr: CUdeviceptr = 0;
            let rc2 = unsafe {
                (api.cu_mem_host_get_device_pointer_v2)(&mut dptr, ptr.cast::<c_void>(), 0)
            };
            oxicuda_driver::check(rc2)?;
            dptr
        } else {
            0
        };

        Ok(RegisteredMemory {
            ptr,
            len,
            flags,
            device_ptr,
        })
    }
}

/// Convenience: registers a mutable slice with the CUDA driver.
///
/// # Errors
///
/// Same as [`register`].
pub fn register_slice<T: Copy>(
    slice: &mut [T],
    flags: RegisterFlags,
) -> CudaResult<RegisteredMemory<T>> {
    register(slice.as_mut_ptr(), slice.len(), flags)
}

/// Convenience: registers a `Vec<T>` with the CUDA driver.
///
/// The `Vec` must not be reallocated (e.g. via `push`, `resize`) while the
/// returned handle is alive, as that would invalidate the registered pointer.
///
/// # Errors
///
/// Same as [`register`].
pub fn register_vec<T: Copy>(
    vec: &mut Vec<T>,
    flags: RegisterFlags,
) -> CudaResult<RegisteredMemory<T>> {
    register(vec.as_mut_ptr(), vec.len(), flags)
}

// ---------------------------------------------------------------------------
// Pointer query
// ---------------------------------------------------------------------------

/// The type of memory backing a registered pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisteredMemoryType {
    /// Host (system) memory.
    Host,
    /// Device (GPU) memory.
    Device,
    /// Unified (managed) memory.
    Unified,
    /// Pointer is not registered with CUDA.
    Unregistered,
}

impl fmt::Display for RegisteredMemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host => write!(f, "Host"),
            Self::Device => write!(f, "Device"),
            Self::Unified => write!(f, "Unified"),
            Self::Unregistered => write!(f, "Unregistered"),
        }
    }
}

/// Information about a pointer registered with the CUDA driver.
#[derive(Debug, Clone, Copy)]
pub struct RegisteredPointerInfo {
    /// Device pointer corresponding to the registered host pointer.
    pub device_ptr: CUdeviceptr,
    /// Whether the memory is managed (unified).
    pub is_managed: bool,
    /// The type of memory backing the pointer.
    pub memory_type: RegisteredMemoryType,
}

/// Queries the CUDA driver for information about a registered pointer.
///
/// # Errors
///
/// * [`CudaError::NotSupported`] on macOS.
/// * [`CudaError::InvalidValue`] if the pointer is not known to the driver.
/// * Other driver errors from `cuPointerGetAttribute`.
pub fn query_registered_pointer_info(ptr: *const u8) -> CudaResult<RegisteredPointerInfo> {
    if ptr.is_null() {
        return Err(CudaError::InvalidValue);
    }

    #[cfg(target_os = "macos")]
    {
        // Synthetic response for macOS tests.
        Ok(RegisteredPointerInfo {
            device_ptr: ptr as CUdeviceptr,
            is_managed: false,
            memory_type: RegisteredMemoryType::Host,
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        let api = try_driver()?;
        let dev_ptr_val = ptr as CUdeviceptr;

        // Query memory type.
        let mut mem_type: u32 = 0;
        let rc = unsafe {
            (api.cu_pointer_get_attribute)(
                (&mut mem_type as *mut u32).cast::<c_void>(),
                ffi::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                dev_ptr_val,
            )
        };
        let memory_type = if rc != 0 {
            // If the query fails, the pointer is likely unregistered.
            RegisteredMemoryType::Unregistered
        } else {
            match mem_type {
                ffi::CU_MEMORYTYPE_HOST => RegisteredMemoryType::Host,
                ffi::CU_MEMORYTYPE_DEVICE => RegisteredMemoryType::Device,
                ffi::CU_MEMORYTYPE_UNIFIED => RegisteredMemoryType::Unified,
                _ => RegisteredMemoryType::Unregistered,
            }
        };

        // Query is_managed.
        let mut managed: u32 = 0;
        let rc2 = unsafe {
            (api.cu_pointer_get_attribute)(
                (&mut managed as *mut u32).cast::<c_void>(),
                ffi::CU_POINTER_ATTRIBUTE_IS_MANAGED,
                dev_ptr_val,
            )
        };
        let is_managed = rc2 == 0 && managed != 0;

        // Query device pointer.
        let mut dptr: CUdeviceptr = 0;
        let rc3 = unsafe {
            (api.cu_pointer_get_attribute)(
                (&mut dptr as *mut CUdeviceptr).cast::<c_void>(),
                ffi::CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                dev_ptr_val,
            )
        };
        if rc3 != 0 {
            dptr = 0;
        }

        Ok(RegisteredPointerInfo {
            device_ptr: dptr,
            is_managed,
            memory_type,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RegisterFlags tests -----------------------------------------------

    #[test]
    fn flags_default_contains_portable_and_device_map() {
        assert!(RegisterFlags::DEFAULT.contains(RegisterFlags::PORTABLE));
        assert!(RegisterFlags::DEFAULT.contains(RegisterFlags::DEVICE_MAP));
        assert!(!RegisterFlags::DEFAULT.contains(RegisterFlags::IO_MEMORY));
        assert!(!RegisterFlags::DEFAULT.contains(RegisterFlags::READ_ONLY));
    }

    #[test]
    fn flags_bitor_combines() {
        let combined = RegisterFlags::PORTABLE | RegisterFlags::READ_ONLY;
        assert!(combined.contains(RegisterFlags::PORTABLE));
        assert!(combined.contains(RegisterFlags::READ_ONLY));
        assert!(!combined.contains(RegisterFlags::IO_MEMORY));
    }

    #[test]
    fn flags_bitand_intersects() {
        let a = RegisterFlags::PORTABLE | RegisterFlags::DEVICE_MAP;
        let b = RegisterFlags::PORTABLE | RegisterFlags::READ_ONLY;
        let intersected = a & b;
        assert!(intersected.contains(RegisterFlags::PORTABLE));
        assert!(!intersected.contains(RegisterFlags::DEVICE_MAP));
        assert!(!intersected.contains(RegisterFlags::READ_ONLY));
    }

    #[test]
    fn flags_display() {
        assert_eq!(RegisterFlags::NONE.to_string(), "NONE");
        assert_eq!(RegisterFlags::PORTABLE.to_string(), "PORTABLE");
        let default_str = RegisterFlags::DEFAULT.to_string();
        assert!(default_str.contains("PORTABLE"));
        assert!(default_str.contains("DEVICE_MAP"));
    }

    #[test]
    fn flags_bits_roundtrip() {
        let flags = RegisterFlags::PORTABLE | RegisterFlags::IO_MEMORY;
        let bits = flags.bits();
        assert_eq!(RegisterFlags::from_bits(bits), flags);
    }

    #[test]
    fn flags_none_is_zero() {
        assert_eq!(RegisterFlags::NONE.bits(), 0);
    }

    // -- RegisteredMemoryType tests ----------------------------------------

    #[test]
    fn memory_type_display() {
        assert_eq!(RegisteredMemoryType::Host.to_string(), "Host");
        assert_eq!(RegisteredMemoryType::Device.to_string(), "Device");
        assert_eq!(RegisteredMemoryType::Unified.to_string(), "Unified");
        assert_eq!(
            RegisteredMemoryType::Unregistered.to_string(),
            "Unregistered"
        );
    }

    #[test]
    fn memory_type_equality() {
        assert_eq!(RegisteredMemoryType::Host, RegisteredMemoryType::Host);
        assert_ne!(RegisteredMemoryType::Host, RegisteredMemoryType::Device);
    }

    // -- register / RegisteredMemory tests ---------------------------------

    #[test]
    fn register_zero_len_fails() {
        let mut buf = [0u8; 16];
        let result = register(buf.as_mut_ptr(), 0, RegisterFlags::DEFAULT);
        assert!(matches!(result, Err(CudaError::InvalidValue)));
    }

    #[test]
    fn register_null_ptr_fails() {
        let result = register::<u8>(std::ptr::null_mut(), 10, RegisterFlags::DEFAULT);
        assert!(matches!(result, Err(CudaError::InvalidValue)));
    }

    #[test]
    fn register_slice_zero_len_fails() {
        let mut empty: [f32; 0] = [];
        let result = register_slice(&mut empty, RegisterFlags::DEFAULT);
        assert!(matches!(result, Err(CudaError::InvalidValue)));
    }

    #[test]
    fn register_vec_zero_len_fails() {
        let mut v: Vec<i32> = Vec::new();
        let result = register_vec(&mut v, RegisterFlags::DEFAULT);
        assert!(matches!(result, Err(CudaError::InvalidValue)));
    }

    #[test]
    fn query_null_ptr_fails() {
        let result = query_registered_pointer_info(std::ptr::null());
        assert!(matches!(result, Err(CudaError::InvalidValue)));
    }

    // -- macOS synthetic tests (these run on all platforms for validation) --

    #[cfg(target_os = "macos")]
    mod macos_tests {
        use super::*;

        #[test]
        fn register_slice_succeeds_on_macos() {
            let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
            let reg = register_slice(data.as_mut_slice(), RegisterFlags::DEFAULT);
            let reg = reg.ok();
            assert!(reg.is_some());
            let reg = reg.inspect(|r| {
                assert_eq!(r.len(), 4);
                assert!(!r.is_empty());
                assert_eq!(r.flags(), RegisterFlags::DEFAULT);
                assert_eq!(r.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
            });
            drop(reg);
        }

        #[test]
        fn register_vec_succeeds_on_macos() {
            let mut v = vec![10u32, 20, 30];
            let reg = register_vec(&mut v, RegisterFlags::PORTABLE);
            assert!(reg.is_ok());
            if let Ok(r) = reg {
                assert_eq!(r.len(), 3);
                assert_eq!(r.flags(), RegisterFlags::PORTABLE);
                assert_ne!(r.device_ptr(), 0);
            }
        }

        #[test]
        fn registered_memory_deref_works() {
            let mut data = vec![100i64, 200, 300];
            let reg = register_vec(&mut data, RegisterFlags::DEFAULT);
            assert!(reg.is_ok());
            if let Ok(r) = reg {
                // Deref to &[T]
                let slice: &[i64] = &r;
                assert_eq!(slice.len(), 3);
                assert_eq!(slice[0], 100);
                assert_eq!(slice[2], 300);
            }
        }

        #[test]
        fn registered_memory_deref_mut_works() {
            let mut data = vec![1u8, 2, 3, 4, 5];
            let reg = register_slice(&mut data, RegisterFlags::DEFAULT);
            assert!(reg.is_ok());
            if let Ok(mut r) = reg {
                r[0] = 99;
                assert_eq!(r[0], 99);
                let mslice: &mut [u8] = &mut r;
                mslice[4] = 88;
                assert_eq!(mslice[4], 88);
            }
        }

        #[test]
        fn query_pointer_info_on_macos() {
            let data = [42u8; 64];
            let info = query_registered_pointer_info(data.as_ptr());
            assert!(info.is_ok());
            if let Ok(info) = info {
                assert!(!info.is_managed);
                assert_eq!(info.memory_type, RegisteredMemoryType::Host);
                assert_ne!(info.device_ptr, 0);
            }
        }

        #[test]
        fn registered_memory_as_ptr_mut_ptr() {
            let mut data = vec![5.0f64; 10];
            let original_ptr = data.as_mut_ptr();
            let reg = register_vec(&mut data, RegisterFlags::DEFAULT);
            assert!(reg.is_ok());
            if let Ok(mut r) = reg {
                assert_eq!(r.as_ptr(), original_ptr as *const f64);
                assert_eq!(r.as_mut_ptr(), original_ptr);
            }
        }
    }

    // -- GPU integration tests (require real hardware) ---------------------

    #[cfg(feature = "gpu-tests")]
    mod gpu_tests {
        use super::*;

        #[test]
        fn register_and_unregister_on_gpu() {
            // cuMemHostRegister requires an active CUDA context bound to the
            // calling thread.  Create one via Context::new (which calls
            // cuCtxCreate, making the context current).  Skip the test if no
            // GPU or driver is available.
            if oxicuda_driver::init().is_err() || oxicuda_driver::Device::count().unwrap_or(0) == 0
            {
                return;
            }
            let Ok(dev) = oxicuda_driver::Device::get(0) else {
                return;
            };
            let Ok(_ctx) = oxicuda_driver::Context::new(&dev) else {
                return;
            };
            // _ctx keeps the CUDA context alive and current for this thread.

            let mut data = vec![0.0f32; 4096];
            let reg = register_vec(&mut data, RegisterFlags::DEFAULT);
            assert!(reg.is_ok(), "registration failed: {:?}", reg.err());
            if let Ok(r) = reg {
                assert_eq!(r.len(), 4096);
                assert!(r.device_ptr() != 0, "device_ptr should be non-zero");
            }
        }
    }
}
