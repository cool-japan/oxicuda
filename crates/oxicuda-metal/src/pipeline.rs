//! Metal compute pipeline wrapper.
//!
//! A [`MetalComputePipeline`] compiles MSL source into a
//! `metal::ComputePipelineState` and holds a retain of the **device-wide**
//! `metal::CommandQueue` owned by [`crate::device::MetalDevice`], so dispatching
//! needs no extra plumbing.  On non-macOS platforms every constructor returns
//! [`MetalError::UnsupportedPlatform`].

use crate::{
    device::MetalDevice,
    error::{MetalError, MetalResult},
    memory::MetalMemoryManager,
};

// ─── Command-buffer completion checking ──────────────────────────────────────

/// Map a finished command buffer's status onto a [`MetalResult`].
///
/// Anything other than `Completed` (device lost, GPU timeout/TDR, a Metal
/// validation failure, an out-of-memory at encode time, …) becomes an error so
/// callers can never mistake a failed GPU submission for a successful one and
/// read back stale or uninitialised buffer contents.
#[cfg(target_os = "macos")]
pub(crate) fn status_to_result(
    status: metal::MTLCommandBufferStatus,
    what: &str,
) -> MetalResult<()> {
    match status {
        metal::MTLCommandBufferStatus::Completed => Ok(()),
        other => Err(MetalError::CommandBufferError(format!(
            "GPU work for '{what}' finished with status {other:?} instead of Completed"
        ))),
    }
}

/// Commit `command_buffer`, block until the GPU finishes it, and turn a
/// non-`Completed` status into an error.
///
/// This is the single place the crate implements the
/// commit → `waitUntilCompleted` → `status()` sequence; every synchronous
/// dispatch path routes through it so a GPU-side failure can never be reported
/// as `Ok`.
#[cfg(target_os = "macos")]
pub(crate) fn commit_and_wait(
    command_buffer: &metal::CommandBufferRef,
    what: &str,
) -> MetalResult<()> {
    command_buffer.commit();
    command_buffer.wait_until_completed();
    status_to_result(command_buffer.status(), what)
}

// ─── MetalComputePipeline ─────────────────────────────────────────────────────

/// A compiled Metal compute pipeline together with its command queue.
///
/// Created by compiling an MSL source string through
/// [`MetalComputePipeline::new`].  The `command_queue` field is an independent
/// retain of the **one queue owned by the device**, not a fresh queue — several
/// hundred cached pipeline variants therefore still share a single
/// `MTLCommandQueue`.
pub struct MetalComputePipeline {
    /// The compiled pipeline state — only present on macOS.
    /// Used by [`MetalComputePipeline::dispatch`].
    #[cfg(target_os = "macos")]
    pub(crate) pipeline_state: metal::ComputePipelineState,
    /// A retain of the device-wide command queue — only present on macOS.
    /// Used by [`MetalComputePipeline::dispatch`].
    #[cfg(target_os = "macos")]
    pub(crate) command_queue: metal::CommandQueue,
    /// The MSL entry-point function name (kept for diagnostics).
    function_name: String,
}

impl MetalComputePipeline {
    /// Compile `msl_source` and look up `function_name` inside the resulting
    /// library, then create a compute pipeline state.
    ///
    /// Returns:
    /// * [`MetalError::ShaderCompilation`] if the MSL fails to compile.
    /// * [`MetalError::PipelineCreation`] if the PSO cannot be created.
    /// * [`MetalError::UnsupportedPlatform`] on non-macOS.
    pub fn new(device: &MetalDevice, msl_source: &str, function_name: &str) -> MetalResult<Self> {
        #[cfg(target_os = "macos")]
        {
            let opts = metal::CompileOptions::new();
            let library = device
                .device
                .new_library_with_source(msl_source, &opts)
                .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

            let function = library
                .get_function(function_name, None)
                .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

            let pipeline_state = device
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MetalError::PipelineCreation(e.to_string()))?;

            // Retain the *device's* queue rather than creating a new one: a
            // fresh `MTLCommandQueue` per cached pipeline variant is both
            // wasteful and an ordering hazard (command buffers on different
            // queues are unordered relative to each other).
            let command_queue = device.command_queue().to_owned();

            Ok(Self {
                pipeline_state,
                command_queue,
                function_name: function_name.to_string(),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (device, msl_source, function_name);
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// The MSL function name this pipeline was compiled for.
    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    /// Dispatch this compiled pipeline over `total_threads` GPU threads (1-D).
    ///
    /// Binding layout:
    /// * each handle in `handles` is resolved to its `metal::Buffer` through
    ///   `memory` and bound to `buffer(0)`, `buffer(1)`, … in order;
    /// * each blob in `scalar_bytes` is bound with `set_bytes` to the buffer
    ///   index immediately following the buffers — `buffer(handles.len())`,
    ///   `buffer(handles.len() + 1)`, … — so a kernel that declares `K` device
    ///   buffers followed by `S` `constant` scalars maps one-to-one.
    ///
    /// The threadgroup width is `min(max_total_threads_per_threadgroup, total_threads)`
    /// and the grid is rounded up to whole threadgroups, so the kernel **must**
    /// bounds-check its `thread_position_in_grid` against the element count.
    ///
    /// The call is synchronous: it commits the command buffer and waits for GPU
    /// completion before returning, matching the crate's other compute ops.
    ///
    /// Returns [`MetalError::UnsupportedPlatform`] on non-macOS, and
    /// [`MetalError::InvalidArgument`] for an unknown buffer handle or an empty
    /// `scalar_bytes` entry.
    pub fn dispatch(
        &self,
        memory: &MetalMemoryManager,
        handles: &[u64],
        scalar_bytes: &[&[u8]],
        total_threads: usize,
    ) -> MetalResult<()> {
        #[cfg(target_os = "macos")]
        {
            if total_threads == 0 {
                return Ok(());
            }
            for blob in scalar_bytes {
                if blob.is_empty() {
                    return Err(MetalError::InvalidArgument(
                        "scalar_bytes entries must be non-empty".into(),
                    ));
                }
            }
            // Resolve handles to independent buffer retains under the lock, then
            // release the lock *before* encoding + the blocking GPU wait. The
            // encoder (and command buffer) retain their bound resources until
            // completion, so a concurrent `free()` cannot invalidate them; this
            // keeps unrelated `alloc`/`free`/`copy_*` calls off the critical path
            // during the (potentially long) kernel run.
            let bound: Vec<metal::Buffer> = {
                let buffers = memory.lock_buffers()?;
                let mut v = Vec::with_capacity(handles.len());
                for handle in handles {
                    let info = buffers.get(handle).ok_or_else(|| {
                        MetalError::InvalidArgument(format!("unknown buffer handle {handle}"))
                    })?;
                    v.push(info.buffer.to_owned());
                }
                v
            };
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline_state);
            for (slot, buffer) in bound.iter().enumerate() {
                encoder.set_buffer(slot as u64, Some(buffer), 0);
            }
            let scalar_base = handles.len() as u64;
            for (offset, blob) in scalar_bytes.iter().enumerate() {
                encoder.set_bytes(
                    scalar_base + offset as u64,
                    blob.len() as u64,
                    blob.as_ptr() as *const std::ffi::c_void,
                );
            }
            let max_tg = self.pipeline_state.max_total_threads_per_threadgroup();
            let tg = max_tg.min(total_threads as u64).max(1);
            let groups = (total_threads as u64).div_ceil(tg);
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(groups, 1, 1),
                metal::MTLSize::new(tg, 1, 1),
            );
            encoder.end_encoding();
            // Surface GPU-side failures (device lost, timeout/TDR, …) instead of
            // returning Ok on a command buffer that finished in an error state.
            commit_and_wait(command_buffer, &self.function_name)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (memory, handles, scalar_bytes, total_threads);
            Err(MetalError::UnsupportedPlatform)
        }
    }
}

impl std::fmt::Debug for MetalComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalComputePipeline(fn={})", self.function_name)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    #[cfg(target_os = "macos")]
    fn try_device() -> Option<MetalDevice> {
        MetalDevice::new().ok()
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_compile_valid_msl() {
        let Some(dev) = try_device() else {
            return;
        };
        let src = crate::msl::gemm_msl();
        let p = MetalComputePipeline::new(&dev, src, "gemm_f32")
            .expect("pipeline creation from valid MSL should succeed");
        assert_eq!(p.function_name(), "gemm_f32");
        let dbg = format!("{p:?}");
        assert!(dbg.contains("gemm_f32"));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_bad_msl_returns_shader_error() {
        let Some(dev) = try_device() else {
            return;
        };
        let bad_src = "this is not valid MSL !!!";
        let err = MetalComputePipeline::new(&dev, bad_src, "nope").unwrap_err();
        assert!(matches!(err, MetalError::ShaderCompilation(_)));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_missing_function_returns_error() {
        let Some(dev) = try_device() else {
            return;
        };
        let src = crate::msl::gemm_msl();
        let err = MetalComputePipeline::new(&dev, src, "nonexistent_function").unwrap_err();
        assert!(matches!(err, MetalError::ShaderCompilation(_)));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn status_completed_is_ok_everything_else_is_err() {
        use metal::MTLCommandBufferStatus as S;
        assert!(status_to_result(S::Completed, "unit").is_ok());
        for status in [
            S::NotEnqueued,
            S::Enqueued,
            S::Committed,
            S::Scheduled,
            S::Error,
        ] {
            let err = status_to_result(status, "unit-op")
                .expect_err("a non-Completed status must map to an error");
            match err {
                MetalError::CommandBufferError(msg) => {
                    assert!(msg.contains("unit-op"), "message should name the op: {msg}");
                    assert!(
                        msg.contains(&format!("{status:?}")),
                        "message should name the status: {msg}"
                    );
                }
                other => panic!("expected CommandBufferError, got {other:?}"),
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipelines_share_one_device_command_queue() {
        let Some(dev) = try_device() else {
            return;
        };
        let a = MetalComputePipeline::new(&dev, crate::msl::gemm_msl(), "gemm_f32")
            .expect("pipeline a");
        let b = MetalComputePipeline::new(&dev, &crate::msl::binary_msl("add"), "binary_f32")
            .expect("pipeline b");
        // `CommandQueueRef` is a zero-sized foreign wrapper living at the
        // Objective-C object's own address, so comparing the reference
        // addresses compares the underlying MTLCommandQueue identities.
        let device_queue: *const metal::CommandQueueRef = dev.command_queue();
        let queue_a: *const metal::CommandQueueRef = &*a.command_queue;
        let queue_b: *const metal::CommandQueueRef = &*b.command_queue;
        assert!(
            std::ptr::eq(queue_a, device_queue),
            "pipelines must retain the device queue, not create their own"
        );
        assert!(std::ptr::eq(queue_a, queue_b));
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn pipeline_unsupported_on_non_macos() {
        // On non-macOS we can't even construct a MetalDevice, so just verify
        // the UnsupportedPlatform error is what MetalDevice returns.
        let result = MetalDevice::new();
        assert!(matches!(result, Err(MetalError::UnsupportedPlatform)));
    }
}
