//! # OxiCUDA Runtime
//!
//! Pure-Rust implementation of the **CUDA Runtime API** (`libcudart`) surface,
//! built on top of `oxicuda-driver`'s dynamic driver loader.
//!
//! ## Coverage
//!
//! | Module       | API functions                                                       |
//! |--------------|---------------------------------------------------------------------|
//! | [`device`]   | `cudaGetDeviceCount`, `cudaSetDevice`, `cudaGetDevice`, `cudaGetDeviceProperties`, `cudaDeviceSynchronize`, `cudaDeviceReset` |
//! | [`memory`]   | `cudaMalloc`, `cudaFree`, `cudaMallocHost`, `cudaFreeHost`, `cudaMallocManaged`, `cudaMallocPitch`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`, `cudaMemGetInfo` |
//! | [`stream`]   | `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamCreateWithPriority`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamQuery`, `cudaStreamWaitEvent`, `cudaStreamGetPriority`, `cudaStreamGetFlags` |
//! | [`event`]    | `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventDestroy`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventQuery`, `cudaEventElapsedTime` |
//! | [`launch`]   | `cudaLaunchKernel` (explicit function handle), `cudaFuncGetAttributes`, `cudaFuncSetAttribute`, `module_load_ptx`, `module_get_function`, `module_unload` |
//! | [`peer`]     | `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`, `cudaDeviceDisablePeerAccess`, `cudaMemcpyPeer`, `cudaMemcpyPeerAsync` |
//! | [`profiler`] | `cudaProfilerStart`, `cudaProfilerStop`, [`profiler::ProfilerGuard`] |
//! | [`error`]    | [`CudaRtError`], [`CudaRtResult`]                                   |
//!
//! ## Design goals
//!
//! - **Zero CUDA SDK build-time dependency**: just like `oxicuda-driver`, the
//!   runtime crate only needs the NVIDIA driver (`libcuda.so` / `nvcuda.dll`)
//!   at *run* time.
//! - **Ergonomic Rust API**: strong types for streams, events, device pointers,
//!   and kernel dimensions instead of raw pointers.
//! - **No unwrap**: all fallible operations return `Result`.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use oxicuda_runtime::{device, memory, stream, event};
//! use oxicuda_runtime::memory::MemcpyKind;
//!
//! // Select device 0.
//! device::set_device(0)?;
//!
//! // Allocate 1 MiB of device memory.
//! let d_buf = memory::malloc(1 << 20)?;
//!
//! // Zero it.
//! memory::memset(d_buf, 0, 1 << 20)?;
//!
//! // Create a stream, record an event.
//! let s = stream::stream_create()?;
//! let e = event::event_create()?;
//! event::event_record(e, s)?;
//! event::event_synchronize(e)?;
//!
//! // Cleanup.
//! event::event_destroy(e)?;
//! stream::stream_destroy(s)?;
//! memory::free(d_buf)?;
//! # Ok::<(), oxicuda_runtime::error::CudaRtError>(())
//! ```

// ─── Modules ─────────────────────────────────────────────────────────────────

pub mod device;
pub mod error;
pub mod event;
pub mod launch;
pub mod memory;
pub mod peer;
pub mod profiler;
pub mod stream;
pub mod texture;

// ─── Top-level re-exports ────────────────────────────────────────────────────

pub use device::CudaDeviceProp;
pub use error::{CudaRtError, CudaRtResult};
pub use event::{CudaEvent, EventFlags};
pub use launch::{CudaFunction, CudaModule, Dim3, FuncAttribute, FuncAttributes};
pub use memory::DevicePtr;
pub use stream::{CudaStream, StreamFlags};
pub use texture::{
    AddressMode, Array3DFlags, ArrayFormat, CudaArray, CudaArray3D, CudaSurfaceObject,
    CudaTextureObject, FilterMode, ResourceDesc, ResourceViewDesc, TextureDesc,
};

// ─── Convenience API (flat namespace) ────────────────────────────────────────

/// Returns the number of CUDA-capable devices (mirrors `cudaGetDeviceCount`).
pub fn get_device_count() -> CudaRtResult<u32> {
    device::get_device_count()
}

/// Set the current device for this thread (mirrors `cudaSetDevice`).
pub fn set_device(ordinal: u32) -> CudaRtResult<()> {
    device::set_device(ordinal)
}

/// Get the current device for this thread (mirrors `cudaGetDevice`).
pub fn get_device() -> CudaRtResult<u32> {
    device::get_device()
}

/// Block until all device operations complete (mirrors `cudaDeviceSynchronize`).
pub fn device_synchronize() -> CudaRtResult<()> {
    device::device_synchronize()
}

/// Allocate device memory (mirrors `cudaMalloc`).
pub fn cuda_malloc(size: usize) -> CudaRtResult<DevicePtr> {
    memory::malloc(size)
}

/// Free device memory (mirrors `cudaFree`).
pub fn cuda_free(ptr: DevicePtr) -> CudaRtResult<()> {
    memory::free(ptr)
}

/// Zero device memory (mirrors `cudaMemset`).
pub fn cuda_memset(ptr: DevicePtr, value: u8, count: usize) -> CudaRtResult<()> {
    memory::memset(ptr, value, count)
}

/// Copy host slice → device (typed helper, no raw pointers).
pub fn memcpy_h2d<T: Copy>(dst: DevicePtr, src: &[T]) -> CudaRtResult<()> {
    memory::memcpy_h2d(dst, src)
}

/// Copy device → host slice (typed helper, no raw pointers).
pub fn memcpy_d2h<T: Copy>(dst: &mut [T], src: DevicePtr) -> CudaRtResult<()> {
    memory::memcpy_d2h(dst, src)
}

/// Copy between device allocations.
pub fn memcpy_d2d(dst: DevicePtr, src: DevicePtr, bytes: usize) -> CudaRtResult<()> {
    memory::memcpy_d2d(dst, src, bytes)
}

// ─── Integration tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the flat API delegates correctly without panicking.
    #[test]
    fn flat_api_no_panic() {
        // These must all return Result, not panic, regardless of GPU presence.
        let _ = get_device_count();
        let _ = get_device();
        let _ = cuda_malloc(0);
        let _ = cuda_free(DevicePtr::NULL);
        let _ = cuda_memset(DevicePtr::NULL, 0, 0);
    }

    #[test]
    fn device_ptr_arithmetic() {
        let base = DevicePtr(0x1000);
        assert_eq!(base.offset(16), DevicePtr(0x1010));
        assert_eq!(base.offset(-16), DevicePtr(0x0FF0));
    }

    #[test]
    fn dim3_convenience() {
        let d = Dim3::one_d(1024);
        assert_eq!(d.volume(), 1024);
        let d2 = Dim3::two_d(32, 8);
        assert_eq!(d2.volume(), 256);
    }

    #[test]
    fn error_display_non_empty() {
        let e = CudaRtError::MemoryAllocation;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn stream_flags_constants() {
        assert_eq!(StreamFlags::DEFAULT.0, 0);
        assert_eq!(StreamFlags::NON_BLOCKING.0, 1);
    }

    #[test]
    fn event_flags_constants() {
        assert_eq!(EventFlags::DEFAULT.0, 0);
        assert_eq!(EventFlags::DISABLE_TIMING.0, 2);
    }
}
