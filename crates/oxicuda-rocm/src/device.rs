//! ROCm/HIP device wrapper.
//!
//! Loads `libamdhip64.so` at runtime via `libloading` and extracts the HIP
//! runtime function pointers needed by the rest of this crate.
//!
//! On non-Linux platforms every constructor returns
//! [`RocmError::UnsupportedPlatform`].

#[cfg(target_os = "linux")]
use std::ffi::c_int;
#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use libloading::Library;

use crate::error::{RocmError, RocmResult};

// ─── HIP constants ────────────────────────────────────────────────────────────

/// Returned by all HIP runtime functions when the call succeeded.
#[cfg(target_os = "linux")]
pub(crate) const HIP_SUCCESS: i32 = 0;
/// Host → Device transfer kind for `hipMemcpy`.
#[cfg(target_os = "linux")]
pub(crate) const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
/// Device → Host transfer kind for `hipMemcpy`.
#[cfg(target_os = "linux")]
pub(crate) const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
/// Device → Device transfer kind for `hipMemcpy`.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub(crate) const HIP_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

// ─── HIP device properties ───────────────────────────────────────────────────

/// Minimal `repr(C)` projection of `hipDeviceProp_t`.
///
/// The full struct is ~700 bytes depending on the HIP version.  We only
/// need the `name` field (first 256 bytes), and we pad to 1024 bytes to
/// safely accommodate any HIP version without reading out-of-bounds.
#[cfg(target_os = "linux")]
#[repr(C)]
pub(crate) struct HipDeviceProperties {
    /// Null-terminated ASCII device name (e.g. `"AMD Radeon RX 7900 XTX"`).
    pub name: [u8; 256],
    _padding: [u8; 768],
}

// ─── Function pointer types ───────────────────────────────────────────────────

/// `hipError_t hipInit(unsigned int flags)`
#[cfg(target_os = "linux")]
pub(crate) type HipInitFn = unsafe extern "C" fn(flags: u32) -> i32;
/// `hipError_t hipGetDeviceCount(int *count)`
#[cfg(target_os = "linux")]
pub(crate) type HipGetDeviceCountFn = unsafe extern "C" fn(count: *mut c_int) -> i32;
/// `hipError_t hipSetDevice(int deviceId)`
#[cfg(target_os = "linux")]
pub(crate) type HipSetDeviceFn = unsafe extern "C" fn(device_id: c_int) -> i32;
/// `hipError_t hipGetDevice(int *deviceId)`
#[cfg(target_os = "linux")]
pub(crate) type HipGetDeviceFn = unsafe extern "C" fn(device_id: *mut c_int) -> i32;
/// `hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId)`
#[cfg(target_os = "linux")]
pub(crate) type HipGetDevicePropertiesFn =
    unsafe extern "C" fn(props: *mut HipDeviceProperties, device_id: c_int) -> i32;
/// `hipError_t hipDeviceSynchronize(void)`
#[cfg(target_os = "linux")]
pub(crate) type HipDeviceSynchronizeFn = unsafe extern "C" fn() -> i32;
/// `hipError_t hipMalloc(void **ptr, size_t size)`
#[cfg(target_os = "linux")]
pub(crate) type HipMallocFn =
    unsafe extern "C" fn(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
/// `hipError_t hipFree(void *ptr)`
#[cfg(target_os = "linux")]
pub(crate) type HipFreeFn = unsafe extern "C" fn(ptr: *mut std::ffi::c_void) -> i32;
/// `hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind)`
#[cfg(target_os = "linux")]
pub(crate) type HipMemcpyFn = unsafe extern "C" fn(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    size_bytes: usize,
    kind: i32,
) -> i32;

// ─── HipApi ──────────────────────────────────────────────────────────────────

/// Holds the loaded HIP shared library and all extracted function pointers.
///
/// `_lib` **must** be stored here (and not dropped earlier) because the
/// function pointers are technically pointers into the loaded DSO.  As long
/// as `_lib` is alive the pointers remain valid.
#[cfg(target_os = "linux")]
pub(crate) struct HipApi {
    /// Keep the library alive — dropping this would invalidate all fn ptrs.
    _lib: Library,
    pub hip_init: HipInitFn,
    pub hip_get_device_count: HipGetDeviceCountFn,
    pub hip_set_device: HipSetDeviceFn,
    #[allow(dead_code)]
    pub hip_get_device: HipGetDeviceFn,
    pub hip_get_device_properties: HipGetDevicePropertiesFn,
    pub hip_device_synchronize: HipDeviceSynchronizeFn,
    pub hip_malloc: HipMallocFn,
    pub hip_free: HipFreeFn,
    pub hip_memcpy: HipMemcpyFn,
}

// ─── RocmDevice ───────────────────────────────────────────────────────────────

/// An AMD ROCm/HIP GPU device.
///
/// On non-Linux platforms [`RocmDevice::new`] always returns
/// [`RocmError::UnsupportedPlatform`].
pub struct RocmDevice {
    /// Loaded HIP API (Linux only).
    #[cfg(target_os = "linux")]
    pub(crate) api: Arc<HipApi>,
    /// Index of the selected device (Linux only).
    #[cfg(target_os = "linux")]
    #[allow(dead_code)]
    pub(crate) device_index: c_int,
    /// Human-readable device name (e.g. `"AMD Radeon RX 7900 XTX"`).
    device_name: String,
}

impl RocmDevice {
    /// Acquire the first available AMD GPU via the HIP runtime.
    ///
    /// # Errors
    ///
    /// - [`RocmError::UnsupportedPlatform`] — not running on Linux.
    /// - [`RocmError::LibraryNotFound`] — `libamdhip64.so` could not be loaded.
    /// - [`RocmError::HipError`] — a HIP API call returned a non-zero code.
    /// - [`RocmError::NoSuitableDevice`] — no AMD GPU was found.
    pub fn new() -> RocmResult<Self> {
        #[cfg(target_os = "linux")]
        {
            // ── 1. Load libamdhip64.so ─────────────────────────────────────
            // SAFETY: We immediately check the return value; the library handle
            // is stored in HipApi._lib for its entire lifetime.
            let lib = unsafe { Library::new("libamdhip64.so") }
                .map_err(|e| RocmError::LibraryNotFound(e.to_string()))?;

            // ── 2. Extract symbols ─────────────────────────────────────────
            macro_rules! sym {
                ($lib:expr, $name:expr, $ty:ty) => {{
                    // SAFETY: We pass a correctly-typed function pointer type and
                    // a null-terminated symbol name.  The returned Symbol borrows
                    // from `$lib`; we immediately copy the fn-ptr value (`*`) so
                    // we do not hold a borrow on `lib`.
                    *unsafe { $lib.get::<$ty>($name) }
                        .map_err(|e| RocmError::LibraryNotFound(e.to_string()))?
                }};
            }

            let hip_init: HipInitFn = sym!(lib, b"hipInit\0", HipInitFn);
            let hip_get_device_count: HipGetDeviceCountFn =
                sym!(lib, b"hipGetDeviceCount\0", HipGetDeviceCountFn);
            let hip_set_device: HipSetDeviceFn = sym!(lib, b"hipSetDevice\0", HipSetDeviceFn);
            let hip_get_device: HipGetDeviceFn = sym!(lib, b"hipGetDevice\0", HipGetDeviceFn);
            let hip_get_device_properties: HipGetDevicePropertiesFn =
                sym!(lib, b"hipGetDeviceProperties\0", HipGetDevicePropertiesFn);
            let hip_device_synchronize: HipDeviceSynchronizeFn =
                sym!(lib, b"hipDeviceSynchronize\0", HipDeviceSynchronizeFn);
            let hip_malloc: HipMallocFn = sym!(lib, b"hipMalloc\0", HipMallocFn);
            let hip_free: HipFreeFn = sym!(lib, b"hipFree\0", HipFreeFn);
            let hip_memcpy: HipMemcpyFn = sym!(lib, b"hipMemcpy\0", HipMemcpyFn);

            let api = HipApi {
                _lib: lib,
                hip_init,
                hip_get_device_count,
                hip_set_device,
                hip_get_device,
                hip_get_device_properties,
                hip_device_synchronize,
                hip_malloc,
                hip_free,
                hip_memcpy,
            };

            // ── 3. hipInit(0) ──────────────────────────────────────────────
            // SAFETY: `hip_init` is a valid function pointer loaded from the HIP
            // runtime.  Flag 0 is the only currently-defined value.
            let rc = unsafe { (api.hip_init)(0) };
            if rc != HIP_SUCCESS {
                return Err(RocmError::HipError(rc, "hipInit failed".into()));
            }

            // ── 4. Count devices ───────────────────────────────────────────
            let mut count: c_int = 0;
            // SAFETY: `hip_get_device_count` is a valid function pointer;
            // `count` is stack-allocated and properly aligned.
            let rc = unsafe { (api.hip_get_device_count)(&mut count) };
            if rc != HIP_SUCCESS {
                return Err(RocmError::HipError(rc, "hipGetDeviceCount failed".into()));
            }
            if count == 0 {
                return Err(RocmError::NoSuitableDevice);
            }

            // ── 5. Read device properties for device 0 ────────────────────
            // SAFETY: `HipDeviceProperties` has `repr(C)` and is large enough
            // to hold any HIP version's `hipDeviceProp_t` (padded to 1024 B).
            let mut props = HipDeviceProperties {
                name: [0u8; 256],
                _padding: [0u8; 768],
            };
            let rc = unsafe { (api.hip_get_device_properties)(&mut props, 0) };
            if rc != HIP_SUCCESS {
                return Err(RocmError::HipError(
                    rc,
                    "hipGetDeviceProperties failed".into(),
                ));
            }

            // Convert null-terminated C string to Rust String.
            let name_end = props.name.iter().position(|&b| b == 0).unwrap_or(256);
            let device_name = String::from_utf8_lossy(&props.name[..name_end]).into_owned();

            // ── 6. Select device 0 ────────────────────────────────────────
            // SAFETY: `hip_set_device` is a valid function pointer; device 0
            // exists (we checked count > 0 above).
            let rc = unsafe { (api.hip_set_device)(0) };
            if rc != HIP_SUCCESS {
                return Err(RocmError::HipError(rc, "hipSetDevice failed".into()));
            }

            tracing::info!("ROCm device selected: {device_name}");

            Ok(Self {
                api: Arc::new(api),
                device_index: 0,
                device_name,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(RocmError::UnsupportedPlatform)
        }
    }

    /// Human-readable device name.
    pub fn name(&self) -> &str {
        &self.device_name
    }
}

// SAFETY: HIP API calls are thread-safe per the ROCm documentation.
// `device_index` is read-only after construction.
// `Arc<HipApi>` allows multiple owners, and fn-ptrs inside HipApi are
// stateless (they go through the driver, which manages its own locking).
unsafe impl Send for RocmDevice {}
// SAFETY: See `Send` impl above.  All shared-state access is mediated by the
// HIP runtime's own internal synchronisation.
unsafe impl Sync for RocmDevice {}

impl std::fmt::Debug for RocmDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RocmDevice({})", self.device_name)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "linux")]
    fn rocm_device_new_graceful_on_linux() {
        match RocmDevice::new() {
            Ok(dev) => {
                assert!(!dev.name().is_empty());
                let dbg = format!("{dev:?}");
                assert!(dbg.contains("RocmDevice"));
            }
            Err(RocmError::LibraryNotFound(_)) => {
                // Acceptable — no ROCm stack installed.
            }
            Err(RocmError::NoSuitableDevice) => {
                // Acceptable — no AMD GPU present.
            }
            Err(RocmError::HipError(_, _)) => {
                // Acceptable — HIP runtime reported an issue.
            }
            Err(e) => {
                // Any other error is also non-fatal — just must not panic.
                let _ = format!("ROCm device init error (non-fatal): {e}");
            }
        }
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn rocm_device_unsupported_on_non_linux() {
        let result = RocmDevice::new();
        assert!(matches!(result, Err(RocmError::UnsupportedPlatform)));
    }
}
