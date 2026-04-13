//! Level Zero device wrapper.
//!
//! Loads `libze_loader.so` (Linux) or `ze_loader.dll` (Windows) at runtime via
//! `libloading` and exposes a safe Rust handle.  On macOS every constructor
//! returns [`LevelZeroError::UnsupportedPlatform`].

use crate::error::{LevelZeroError, LevelZeroResult};

// ─── Platform-specific imports and type definitions ──────────────────────────

#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::{ffi::c_void, sync::Arc};

#[cfg(any(target_os = "linux", target_os = "windows"))]
use libloading::Library;

// ─── Level Zero opaque handle types (Linux + Windows only) ───────────────────

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeDriverHandle = *mut c_void;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeDeviceHandle = *mut c_void;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeContextHandle = *mut c_void;

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(crate) type ZeCommandQueueHandle = *mut c_void;

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(crate) type ZeCommandListHandle = *mut c_void;

// ─── Level Zero result / type constants ──────────────────────────────────────

#[cfg(any(target_os = "linux", target_os = "windows"))]
const ZE_RESULT_SUCCESS: u32 = 0;

#[cfg(any(target_os = "linux", target_os = "windows"))]
const ZE_DEVICE_TYPE_GPU: u32 = 1;

#[cfg(any(target_os = "linux", target_os = "windows"))]
const ZE_STRUCTURE_TYPE_CONTEXT_DESC: u32 = 0xb;

#[cfg(any(target_os = "linux", target_os = "windows"))]
const ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC: u32 = 0xf;

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(crate) const ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC: u32 = 0x9;

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(crate) const ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: u32 = 0x1;

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(crate) const ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: u32 = 0x2;

#[cfg(any(target_os = "linux", target_os = "windows"))]
const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x3;

// ─── Level Zero descriptor and property structs (Linux + Windows only) ───────

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[repr(C)]
pub(crate) struct ZeContextDesc {
    pub(crate) stype: u32,
    pub(crate) p_next: *const c_void,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[repr(C)]
pub(crate) struct ZeCommandQueueDesc {
    stype: u32,
    p_next: *const c_void,
    ordinal: u32,
    index: u32,
    flags: u32,
    mode: u32,
    priority: u32,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[repr(C)]
pub(crate) struct ZeCommandListDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub command_queue_group_ordinal: u32,
    pub flags: u32,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[repr(C)]
pub(crate) struct ZeDeviceMemAllocDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub ordinal: u32,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[repr(C)]
pub(crate) struct ZeHostMemAllocDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[repr(C)]
pub(crate) struct ZeDeviceProperties {
    stype: u32,
    p_next: *const c_void,
    device_type: u32,
    vendor_id: u32,
    device_id: u32,
    _flags: u32,
    _sub_device_ids: [u32; 64],
    _timer_resolution: u64,
    _timestamp_valid_bits: u32,
    _kernel_timestamp_valid_bits: u32,
    name: [u8; 256],
    _max_mem_alloc_size: u64,
    _num_threads_per_eu: u32,
    _physical_eu_simd_width: u32,
    _num_eu_per_sub_slice: u32,
    _num_sub_slices_per_slice: u32,
    _num_slices: u32,
    _timer_resolution_ns: u64,
    _uuid: [u8; 16],
}

// ─── Level Zero function pointer types (Linux + Windows only) ────────────────

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeInitFn = unsafe extern "C" fn(flags: u32) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeDriverGetFn = unsafe extern "C" fn(count: *mut u32, drivers: *mut ZeDriverHandle) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeDeviceGetFn = unsafe extern "C" fn(
    driver: ZeDriverHandle,
    count: *mut u32,
    devices: *mut ZeDeviceHandle,
) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeDeviceGetPropertiesFn =
    unsafe extern "C" fn(device: ZeDeviceHandle, props: *mut ZeDeviceProperties) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeContextCreateFn = unsafe extern "C" fn(
    driver: ZeDriverHandle,
    desc: *const ZeContextDesc,
    context: *mut ZeContextHandle,
) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeContextDestroyFn = unsafe extern "C" fn(context: ZeContextHandle) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeCommandQueueCreateFn = unsafe extern "C" fn(
    context: ZeContextHandle,
    device: ZeDeviceHandle,
    desc: *const ZeCommandQueueDesc,
    queue: *mut ZeCommandQueueHandle,
) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeCommandQueueDestroyFn = unsafe extern "C" fn(queue: ZeCommandQueueHandle) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeCommandQueueSynchronizeFn =
    unsafe extern "C" fn(queue: ZeCommandQueueHandle, timeout: u64) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeCommandQueueExecuteCommandListsFn = unsafe extern "C" fn(
    queue: ZeCommandQueueHandle,
    count: u32,
    lists: *const ZeCommandListHandle,
    fence: usize,
) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeCommandListCreateFn = unsafe extern "C" fn(
    context: ZeContextHandle,
    device: ZeDeviceHandle,
    desc: *const ZeCommandListDesc,
    list: *mut ZeCommandListHandle,
) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeCommandListDestroyFn = unsafe extern "C" fn(list: ZeCommandListHandle) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeCommandListCloseFn = unsafe extern "C" fn(list: ZeCommandListHandle) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeCommandListResetFn = unsafe extern "C" fn(list: ZeCommandListHandle) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeCommandListAppendMemoryCopyFn = unsafe extern "C" fn(
    list: ZeCommandListHandle,
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
    signal_event: usize,
    wait_count: u32,
    wait_events: *const usize,
) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeMemAllocDeviceFn = unsafe extern "C" fn(
    context: ZeContextHandle,
    desc: *const ZeDeviceMemAllocDesc,
    size: usize,
    alignment: usize,
    device: ZeDeviceHandle,
    ptr: *mut *mut c_void,
) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeMemAllocHostFn = unsafe extern "C" fn(
    context: ZeContextHandle,
    desc: *const ZeHostMemAllocDesc,
    size: usize,
    alignment: usize,
    ptr: *mut *mut c_void,
) -> u32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type ZeMemFreeFn = unsafe extern "C" fn(context: ZeContextHandle, ptr: *mut c_void) -> u32;

// ─── L0Api — dynamically-loaded function table (Linux + Windows only) ─────────

/// Holds the loaded `libze_loader` library and all extracted function pointers.
///
/// The `Library` field keeps the shared object alive for the lifetime of this struct.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(crate) struct L0Api {
    /// The loaded shared library — must outlive all function pointer calls.
    _lib: Library,
    pub ze_init: ZeInitFn,
    pub ze_driver_get: ZeDriverGetFn,
    pub ze_device_get: ZeDeviceGetFn,
    pub ze_device_get_properties: ZeDeviceGetPropertiesFn,
    pub ze_context_create: ZeContextCreateFn,
    pub ze_context_destroy: ZeContextDestroyFn,
    pub ze_command_queue_create: ZeCommandQueueCreateFn,
    pub ze_command_queue_destroy: ZeCommandQueueDestroyFn,
    pub ze_command_queue_synchronize: ZeCommandQueueSynchronizeFn,
    pub ze_command_queue_execute_command_lists: ZeCommandQueueExecuteCommandListsFn,
    pub ze_command_list_create: ZeCommandListCreateFn,
    pub ze_command_list_destroy: ZeCommandListDestroyFn,
    pub ze_command_list_close: ZeCommandListCloseFn,
    #[allow(dead_code)]
    pub ze_command_list_reset: ZeCommandListResetFn,
    pub ze_command_list_append_memory_copy: ZeCommandListAppendMemoryCopyFn,
    pub ze_mem_alloc_device: ZeMemAllocDeviceFn,
    pub ze_mem_alloc_host: ZeMemAllocHostFn,
    pub ze_mem_free: ZeMemFreeFn,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl L0Api {
    /// Load the Level Zero loader library and extract all function pointers.
    ///
    /// # Safety
    ///
    /// The returned `L0Api` must not outlive the process image that loaded it.
    /// All function pointers are valid only as long as the `Library` is alive
    /// (which is guaranteed by the `_lib` field).
    unsafe fn load() -> LevelZeroResult<Self> {
        #[cfg(target_os = "linux")]
        let lib_name = "libze_loader.so.1";
        #[cfg(target_os = "windows")]
        let lib_name = "ze_loader.dll";

        // SAFETY: We are loading a well-known system library by its standard
        // filename. The resulting Library keeps the shared object alive.
        let lib = unsafe {
            Library::new(lib_name)
                .map_err(|e| LevelZeroError::LibraryNotFound(format!("{lib_name}: {e}")))?
        };

        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                // SAFETY: We are loading a symbol from the Level Zero loader
                // that we know to exist with the correct type signature.
                // The symbol lifetime is bounded by `lib`.
                *unsafe {
                    lib.get::<$ty>($name).map_err(|e| {
                        LevelZeroError::LibraryNotFound(format!(
                            "symbol {}: {e}",
                            stringify!($name)
                        ))
                    })?
                }
            }};
        }

        let ze_init = sym!(b"zeInit\0", ZeInitFn);
        let ze_driver_get = sym!(b"zeDriverGet\0", ZeDriverGetFn);
        let ze_device_get = sym!(b"zeDeviceGet\0", ZeDeviceGetFn);
        let ze_device_get_properties = sym!(b"zeDeviceGetProperties\0", ZeDeviceGetPropertiesFn);
        let ze_context_create = sym!(b"zeContextCreate\0", ZeContextCreateFn);
        let ze_context_destroy = sym!(b"zeContextDestroy\0", ZeContextDestroyFn);
        let ze_command_queue_create = sym!(b"zeCommandQueueCreate\0", ZeCommandQueueCreateFn);
        let ze_command_queue_destroy = sym!(b"zeCommandQueueDestroy\0", ZeCommandQueueDestroyFn);
        let ze_command_queue_synchronize =
            sym!(b"zeCommandQueueSynchronize\0", ZeCommandQueueSynchronizeFn);
        let ze_command_queue_execute_command_lists = sym!(
            b"zeCommandQueueExecuteCommandLists\0",
            ZeCommandQueueExecuteCommandListsFn
        );
        let ze_command_list_create = sym!(b"zeCommandListCreate\0", ZeCommandListCreateFn);
        let ze_command_list_destroy = sym!(b"zeCommandListDestroy\0", ZeCommandListDestroyFn);
        let ze_command_list_close = sym!(b"zeCommandListClose\0", ZeCommandListCloseFn);
        let ze_command_list_reset = sym!(b"zeCommandListReset\0", ZeCommandListResetFn);
        let ze_command_list_append_memory_copy = sym!(
            b"zeCommandListAppendMemoryCopy\0",
            ZeCommandListAppendMemoryCopyFn
        );
        let ze_mem_alloc_device = sym!(b"zeMemAllocDevice\0", ZeMemAllocDeviceFn);
        let ze_mem_alloc_host = sym!(b"zeMemAllocHost\0", ZeMemAllocHostFn);
        let ze_mem_free = sym!(b"zeMemFree\0", ZeMemFreeFn);

        Ok(Self {
            _lib: lib,
            ze_init,
            ze_driver_get,
            ze_device_get,
            ze_device_get_properties,
            ze_context_create,
            ze_context_destroy,
            ze_command_queue_create,
            ze_command_queue_destroy,
            ze_command_queue_synchronize,
            ze_command_queue_execute_command_lists,
            ze_command_list_create,
            ze_command_list_destroy,
            ze_command_list_close,
            ze_command_list_reset,
            ze_command_list_append_memory_copy,
            ze_mem_alloc_device,
            ze_mem_alloc_host,
            ze_mem_free,
        })
    }
}

// ─── LevelZeroDevice ─────────────────────────────────────────────────────────

/// An Intel GPU device accessed via the Level Zero API.
///
/// On non-Linux/Windows platforms, [`LevelZeroDevice::new`] always returns
/// [`LevelZeroError::UnsupportedPlatform`].
pub struct LevelZeroDevice {
    /// The loaded Level Zero API (Linux and Windows only).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) api: Arc<L0Api>,
    /// The Level Zero context handle (Linux and Windows only).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) context: ZeContextHandle,
    /// The selected GPU device handle (Linux and Windows only).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) device: ZeDeviceHandle,
    /// The command queue handle (Linux and Windows only).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) queue: ZeCommandQueueHandle,
    /// Human-readable device name.
    device_name: String,
}

impl LevelZeroDevice {
    /// Open the first Intel GPU found via the Level Zero loader.
    ///
    /// Returns [`LevelZeroError::UnsupportedPlatform`] on macOS.
    pub fn new() -> LevelZeroResult<Self> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            // SAFETY: L0Api::load performs dlopen/LoadLibrary and symbol
            // resolution. All further calls are guarded by result checks.
            let api = Arc::new(unsafe { L0Api::load()? });

            // Step 1: Initialize Level Zero runtime.
            // SAFETY: ze_init is a valid function pointer from the loaded library.
            let rc = unsafe { (api.ze_init)(0) };
            if rc != ZE_RESULT_SUCCESS {
                return Err(LevelZeroError::ZeError(rc, "zeInit failed".into()));
            }

            // Step 2: Enumerate drivers.
            let mut driver_count: u32 = 0;
            // SAFETY: Passing null for the driver array to query the count.
            let rc =
                unsafe { (api.ze_driver_get)(&mut driver_count as *mut u32, std::ptr::null_mut()) };
            if rc != ZE_RESULT_SUCCESS {
                return Err(LevelZeroError::ZeError(
                    rc,
                    "zeDriverGet (count) failed".into(),
                ));
            }
            if driver_count == 0 {
                return Err(LevelZeroError::NoSuitableDevice);
            }

            let mut drivers: Vec<ZeDriverHandle> =
                vec![std::ptr::null_mut(); driver_count as usize];
            // SAFETY: `drivers` is allocated for exactly `driver_count` elements.
            let rc =
                unsafe { (api.ze_driver_get)(&mut driver_count as *mut u32, drivers.as_mut_ptr()) };
            if rc != ZE_RESULT_SUCCESS {
                return Err(LevelZeroError::ZeError(
                    rc,
                    "zeDriverGet (enumerate) failed".into(),
                ));
            }

            let driver = drivers[0];

            // Step 3: Enumerate devices and find the first GPU.
            let mut device_count: u32 = 0;
            // SAFETY: Passing null to query the device count for this driver.
            let rc = unsafe {
                (api.ze_device_get)(driver, &mut device_count as *mut u32, std::ptr::null_mut())
            };
            if rc != ZE_RESULT_SUCCESS {
                return Err(LevelZeroError::ZeError(
                    rc,
                    "zeDeviceGet (count) failed".into(),
                ));
            }
            if device_count == 0 {
                return Err(LevelZeroError::NoSuitableDevice);
            }

            let mut devices: Vec<ZeDeviceHandle> =
                vec![std::ptr::null_mut(); device_count as usize];
            // SAFETY: `devices` is allocated for exactly `device_count` elements.
            let rc = unsafe {
                (api.ze_device_get)(driver, &mut device_count as *mut u32, devices.as_mut_ptr())
            };
            if rc != ZE_RESULT_SUCCESS {
                return Err(LevelZeroError::ZeError(
                    rc,
                    "zeDeviceGet (enumerate) failed".into(),
                ));
            }

            // Step 4: Find the first GPU device and read its properties.
            let mut chosen_device: Option<ZeDeviceHandle> = None;
            let mut device_name = String::from("Intel GPU");

            for &dev in &devices {
                // SAFETY: ZeDeviceProperties is #[repr(C)] and fully initialized
                // (zeroed) before passing to the API.
                let mut props =
                    unsafe { std::mem::MaybeUninit::<ZeDeviceProperties>::zeroed().assume_init() };
                props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                props.p_next = std::ptr::null();

                // SAFETY: `dev` is a valid device handle; `props` is properly
                // initialized with the correct stype field.
                let rc = unsafe {
                    (api.ze_device_get_properties)(dev, &mut props as *mut ZeDeviceProperties)
                };
                if rc != ZE_RESULT_SUCCESS {
                    continue;
                }

                if props.device_type == ZE_DEVICE_TYPE_GPU {
                    // Extract null-terminated name string.
                    let name_len = props
                        .name
                        .iter()
                        .position(|&b| b == 0)
                        .unwrap_or(props.name.len());
                    device_name = String::from_utf8_lossy(&props.name[..name_len]).into_owned();
                    chosen_device = Some(dev);
                    break;
                }
            }

            let device = chosen_device.ok_or(LevelZeroError::NoSuitableDevice)?;

            // Step 5: Create a context.
            let ctx_desc = ZeContextDesc {
                stype: ZE_STRUCTURE_TYPE_CONTEXT_DESC,
                p_next: std::ptr::null(),
            };
            let mut context: ZeContextHandle = std::ptr::null_mut();
            // SAFETY: `ctx_desc` is valid; `context` is a valid output pointer.
            let rc = unsafe {
                (api.ze_context_create)(driver, &ctx_desc, &mut context as *mut ZeContextHandle)
            };
            if rc != ZE_RESULT_SUCCESS {
                return Err(LevelZeroError::ZeError(rc, "zeContextCreate failed".into()));
            }

            // Step 6: Create a command queue.
            let queue_desc = ZeCommandQueueDesc {
                stype: ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                p_next: std::ptr::null(),
                ordinal: 0,
                index: 0,
                flags: 0,
                mode: 0, // default mode
                priority: 0,
            };
            let mut queue: ZeCommandQueueHandle = std::ptr::null_mut();
            // SAFETY: `context`, `device`, and `queue_desc` are valid; `queue`
            // is a valid output pointer.
            let rc = unsafe {
                (api.ze_command_queue_create)(
                    context,
                    device,
                    &queue_desc,
                    &mut queue as *mut ZeCommandQueueHandle,
                )
            };
            if rc != ZE_RESULT_SUCCESS {
                // Clean up context before returning error.
                // SAFETY: context was successfully created above.
                unsafe { (api.ze_context_destroy)(context) };
                return Err(LevelZeroError::ZeError(
                    rc,
                    "zeCommandQueueCreate failed".into(),
                ));
            }

            tracing::info!("Level Zero device selected: {device_name}");

            Ok(Self {
                api,
                context,
                device,
                queue,
                device_name,
            })
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            Err(LevelZeroError::UnsupportedPlatform)
        }
    }

    /// Human-readable device name (e.g. `"Intel UHD Graphics 770"`).
    pub fn name(&self) -> &str {
        &self.device_name
    }
}

// ─── Drop ────────────────────────────────────────────────────────────────────

impl Drop for LevelZeroDevice {
    fn drop(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            // SAFETY: `queue` and `context` were successfully created in `new()`
            // and have not been freed yet.  We destroy in reverse creation order.
            unsafe {
                (self.api.ze_command_queue_destroy)(self.queue);
                (self.api.ze_context_destroy)(self.context);
            }
        }
    }
}

// ─── Send + Sync ─────────────────────────────────────────────────────────────

// SAFETY: `LevelZeroDevice` holds raw Level Zero opaque pointers that are
// conceptually owned by this struct.  The Level Zero specification states that
// context and command-queue handles may be used from any thread as long as
// external synchronization is provided.  We guarantee exclusive ownership via
// `Arc<LevelZeroDevice>` and `Mutex` in the memory manager.
unsafe impl Send for LevelZeroDevice {}
// SAFETY: See `Send` impl above.  Shared immutable access is safe because all
// mutable operations go through synchronised command lists.
unsafe impl Sync for LevelZeroDevice {}

// ─── Debug ───────────────────────────────────────────────────────────────────

impl std::fmt::Debug for LevelZeroDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LevelZeroDevice({})", self.device_name)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn level_zero_device_graceful_init() {
        match LevelZeroDevice::new() {
            Ok(dev) => {
                assert!(!dev.name().is_empty());
                let dbg = format!("{dev:?}");
                assert!(dbg.contains("LevelZeroDevice"));
            }
            Err(LevelZeroError::LibraryNotFound(_)) => {
                // Acceptable: Level Zero loader not installed on this machine.
            }
            Err(LevelZeroError::NoSuitableDevice) => {
                // Acceptable: no Intel GPU present.
            }
            Err(LevelZeroError::ZeError(_, _)) => {
                // Acceptable: Level Zero runtime error (e.g. missing driver).
            }
            Err(e) => {
                // Any other error must not panic — just log it.
                let _ = format!("Level Zero device init error (non-fatal): {e}");
            }
        }
    }

    #[test]
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    fn level_zero_device_unsupported_on_macos() {
        let result = LevelZeroDevice::new();
        assert!(matches!(result, Err(LevelZeroError::UnsupportedPlatform)));
    }
}
