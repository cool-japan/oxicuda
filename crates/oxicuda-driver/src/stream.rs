//! CUDA stream management.
//!
//! Streams are command queues on the GPU. Commands within a stream
//! execute in order. Different streams can execute concurrently.
//!
//! # Example
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use oxicuda_driver::context::Context;
//! # use oxicuda_driver::stream::Stream;
//! # fn main() -> Result<(), oxicuda_driver::error::CudaError> {
//! // Assuming `ctx` is an Arc<Context> obtained from Context::new(...)
//! # let ctx: Arc<Context> = unimplemented!();
//! let stream = Stream::new(&ctx)?;
//! // ... enqueue work on the stream ...
//! stream.synchronize()?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use crate::context::Context;
use crate::error::CudaResult;
use crate::event::Event;
use crate::ffi::{CU_STREAM_NON_BLOCKING, CUcontext, CUstream};
use crate::loader::try_driver;

/// Creates a raw non-blocking stream **in `ctx`**, restoring the thread's
/// previously-current context afterward.
///
/// `cuStreamCreate` targets whichever context is current on the calling
/// thread, so a [`Stream`] that stores an [`Arc<Context>`] must make that
/// context current for the duration of the create call — otherwise the stream
/// would silently belong to some unrelated context that merely happened to be
/// current, and work later enqueued on it would run in the wrong context
/// (device pointers from `ctx` would be invalid there). The previous current
/// context is captured and restored so this is transparent to the caller.
fn create_stream_in_ctx(
    api: &crate::loader::DriverApi,
    ctx: &Context,
    create: impl FnOnce(&mut CUstream) -> u32,
) -> CudaResult<CUstream> {
    // Capture the thread's current context (null if none) so we can restore it.
    let mut prev = CUcontext::default();
    // SAFETY: `cu_ctx_get_current` was resolved from the driver and `prev` is a
    // valid out-pointer. A non-zero rc leaves `prev` null, so we restore to the
    // "no context" state, which is the correct fallback.
    let _ = unsafe { (api.cu_ctx_get_current)(&mut prev) };
    // Bind `ctx` for the duration of the create call.
    crate::cuda_call!((api.cu_ctx_set_current)(ctx.raw()))?;
    let mut raw = CUstream::default();
    let rc = create(&mut raw);
    // Restore the previous context regardless of whether create succeeded.
    // SAFETY: `prev` is either a context that was current a moment ago or null.
    let _ = unsafe { (api.cu_ctx_set_current)(prev) };
    crate::error::check(rc)?;
    Ok(raw)
}

/// A CUDA stream (GPU command queue).
///
/// Streams provide ordered, asynchronous execution of GPU commands.
/// Commands enqueued on the same stream execute sequentially, while
/// commands on different streams may execute concurrently.
///
/// The stream holds an [`Arc<Context>`] to ensure the parent context
/// outlives the stream.
///
/// # A `Stream` is a shared handle, not a unique owner
///
/// Cloning yields a second handle to the **same** queue — [`Stream::raw`]
/// returns the same `CUstream` — and the queue is destroyed once the last
/// handle drops. That is what lets two subsystems which each want to hold
/// "their" stream be collapsed onto one queue: `oxicuda-dnn`'s `DnnHandle` and
/// the `BlasHandle` nested inside it now share one, so a convolution's output
/// is ordered before a GEMM that reads it by stream semantics alone — no
/// event choreography, no host rendezvous, and a capture of the pair is a
/// linear chain rather than a fork/join.
///
/// `Clone` is written out rather than derived so the doc comment can say what
/// it means: this is another reference to one queue, not a copy of it.
pub struct Stream {
    /// The driver-owned queue, destroyed when the last handle drops.
    inner: Arc<StreamInner>,
}

impl Clone for Stream {
    /// Another handle to the same queue. See the type docs.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// The driver-owned half of a [`Stream`]: destroyed exactly once, when the
/// last handle to the queue drops.
struct StreamInner {
    /// Raw CUDA stream handle.
    raw: CUstream,
    /// Keeps the parent context alive for the lifetime of the stream.
    ctx: Arc<Context>,
}

// `Stream` is `Send + Sync` by auto-derivation from its fields: an
// `Arc<StreamInner>` over a `CUstream` handle and an `Arc<Context>` (and
// `Context` is itself `Send + Sync`). The CUDA Driver API is thread-safe, so
// no manual `unsafe impl` is required.

impl Stream {
    /// Creates a new stream with [`CU_STREAM_NON_BLOCKING`] flag.
    ///
    /// Non-blocking streams do not implicitly synchronise with the
    /// default (NULL) stream, allowing maximum concurrency.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the driver
    /// call fails (e.g. invalid context, out of resources).
    pub fn new(ctx: &Arc<Context>) -> CudaResult<Self> {
        let api = try_driver()?;
        // Bind the stream to `ctx` (not merely to whatever context happens to be
        // current), matching the `Arc<Context>` this stream stores and keeps
        // alive. See [`create_stream_in_ctx`].
        let raw = create_stream_in_ctx(api, ctx, |raw| unsafe {
            (api.cu_stream_create)(raw, CU_STREAM_NON_BLOCKING)
        })?;
        Ok(Self {
            inner: Arc::new(StreamInner {
                raw,
                ctx: Arc::clone(ctx),
            }),
        })
    }

    /// Creates a new stream with the specified priority and
    /// [`CU_STREAM_NON_BLOCKING`] flag.
    ///
    /// Lower numerical values indicate higher priority. The valid range
    /// can be queried via `cuCtxGetStreamPriorityRange`.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the priority
    /// is out of range or the driver call otherwise fails.
    pub fn with_priority(ctx: &Arc<Context>, priority: i32) -> CudaResult<Self> {
        let api = try_driver()?;
        // Bind the stream to `ctx`; see [`Stream::new`] / [`create_stream_in_ctx`].
        let raw = create_stream_in_ctx(api, ctx, |raw| unsafe {
            (api.cu_stream_create_with_priority)(raw, CU_STREAM_NON_BLOCKING, priority)
        })?;
        Ok(Self {
            inner: Arc::new(StreamInner {
                raw,
                ctx: Arc::clone(ctx),
            }),
        })
    }

    /// Blocks the calling thread until all previously enqueued commands
    /// in this stream have completed.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if any enqueued
    /// operation failed or the driver reports an error.
    pub fn synchronize(&self) -> CudaResult<()> {
        let api = try_driver()?;
        crate::cuda_call!((api.cu_stream_synchronize)(self.inner.raw))
    }

    /// Makes all future work submitted to this stream wait until
    /// the given event has been recorded and completed.
    ///
    /// This is the primary mechanism for inter-stream synchronisation:
    /// record an [`Event`] on one stream, then call `wait_event` on
    /// another stream to establish an ordering dependency.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the driver
    /// call fails (e.g. invalid event handle).
    pub fn wait_event(&self, event: &Event) -> CudaResult<()> {
        let api = try_driver()?;
        // flags = 0 is the only documented value.
        crate::cuda_call!((api.cu_stream_wait_event)(self.inner.raw, event.raw(), 0))
    }

    /// Returns the raw [`CUstream`] handle.
    ///
    /// # Safety (caller)
    ///
    /// The caller must not destroy or otherwise invalidate the handle
    /// while this `Stream` is still alive.
    #[inline]
    pub fn raw(&self) -> CUstream {
        self.inner.raw
    }

    /// Whether `self` and `other` are handles to the **same** driver queue.
    ///
    /// The question a caller asks before deciding that stream order alone
    /// sequences two pieces of work: on one queue it does, on two it does not
    /// and an event is required. Compares the driver handle rather than the
    /// `Arc`, so a queue reached through two independently-built handles (were
    /// that ever possible) still answers truthfully.
    #[inline]
    #[must_use]
    pub fn is_same_queue(&self, other: &Self) -> bool {
        self.inner.raw == other.inner.raw
    }

    /// Returns a reference to the parent [`Context`].
    #[inline]
    pub fn context(&self) -> &Arc<Context> {
        &self.inner.ctx
    }
}

impl Drop for StreamInner {
    fn drop(&mut self) {
        if let Ok(api) = try_driver() {
            let rc = unsafe { (api.cu_stream_destroy_v2)(self.raw) };
            if rc != 0 {
                tracing::warn!(
                    cuda_error = rc,
                    stream = ?self.raw,
                    "cuStreamDestroy_v2 failed during drop"
                );
            }
        }
    }
}

#[cfg(test)]
mod multi_stream_tests {
    use super::*;
    use crate::device::Device;
    use crate::ffi::CUdeviceptr;
    use crate::module::Module;
    use std::ffi::c_void;

    /// Grid-stride in-place doubling kernel, arch-portable (`.target sm_70`).
    const DOUBLE_PTX: &str = "\
.version 7.0
.target sm_70
.address_size 64
.visible .entry dbl(
    .param .u64 ptr,
    .param .u32 n
)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<2>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [ptr];
    ld.param.u32 %r0, [n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    mov.u32 %r5, %nctaid.x;
    mul.lo.u32 %r6, %r5, %r2;
$LOOP:
    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra $DONE;
    mul.wide.u32 %rd1, %r4, 4;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.f32 %f0, [%rd2];
    add.f32 %f0, %f0, %f0;
    st.global.f32 [%rd2], %f0;
    add.u32 %r4, %r4, %r6;
    bra $LOOP;
$DONE:
    ret;
}
";

    /// Launch the doubling kernel on `dptr` over `stream` (raw FFI).
    fn launch_double(
        api: &crate::loader::DriverApi,
        func: &crate::module::Function,
        stream: &Stream,
        dptr: CUdeviceptr,
        n: usize,
    ) -> CudaResult<()> {
        let mut dptr_arg = dptr;
        let mut n_arg: u32 = n as u32;
        let mut params: [*mut c_void; 2] = [
            (&mut dptr_arg as *mut CUdeviceptr).cast(),
            (&mut n_arg as *mut u32).cast(),
        ];
        crate::error::check(unsafe {
            (api.cu_launch_kernel)(
                func.raw(),
                8,
                1,
                1,
                128,
                1,
                1,
                0,
                stream.raw(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    }

    /// Real-hardware multi-stream test: run the doubling kernel concurrently on
    /// two independent streams over two buffers, plus a cross-stream dependency
    /// (stream B waits on an event recorded on stream A before doubling a buffer
    /// A already doubled — so it ends up x4). Verifies both the concurrent and
    /// the ordered results. No-op without a GPU.
    #[test]
    fn two_streams_concurrent_and_cross_stream_event() {
        let Ok(dev) = Device::get(0) else {
            return;
        };
        let ctx = match Context::new(&dev) {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let stream_a = match Stream::new(&ctx) {
            Ok(s) => s,
            Err(_) => return,
        };
        let stream_b = match Stream::new(&ctx) {
            Ok(s) => s,
            Err(_) => return,
        };
        let api = try_driver().expect("driver present");

        let module = match Module::from_ptx(DOUBLE_PTX) {
            Ok(m) => m,
            Err(_) => return,
        };
        let func = module.get_function("dbl").expect("dbl");

        const N: usize = 2048;
        let bytes = N * std::mem::size_of::<f32>();
        let a_in: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_in: Vec<f32> = (0..N).map(|i| i as f32 + 1000.0).collect();

        let mut da: CUdeviceptr = 0;
        let mut db: CUdeviceptr = 0;
        crate::error::check(unsafe { (api.cu_mem_alloc_v2)(&mut da, bytes) }).expect("alloc a");
        crate::error::check(unsafe { (api.cu_mem_alloc_v2)(&mut db, bytes) }).expect("alloc b");

        let result = (|| -> CudaResult<(Vec<f32>, Vec<f32>)> {
            crate::error::check(unsafe {
                (api.cu_memcpy_htod_v2)(da, a_in.as_ptr().cast(), bytes)
            })?;
            crate::error::check(unsafe {
                (api.cu_memcpy_htod_v2)(db, b_in.as_ptr().cast(), bytes)
            })?;

            // Concurrent: double A on stream A, double B on stream B.
            launch_double(api, &func, &stream_a, da, N)?;
            launch_double(api, &func, &stream_b, db, N)?;

            // Cross-stream dependency: record an event on A after its kernel,
            // make B wait on it, then double A again on B (A -> x4).
            let evt = Event::new()?;
            evt.record(&stream_a)?;
            stream_b.wait_event(&evt)?;
            launch_double(api, &func, &stream_b, da, N)?;

            stream_a.synchronize()?;
            stream_b.synchronize()?;

            let mut a_out = vec![0.0f32; N];
            let mut b_out = vec![0.0f32; N];
            crate::error::check(unsafe {
                (api.cu_memcpy_dtoh_v2)(a_out.as_mut_ptr().cast(), da, bytes)
            })?;
            crate::error::check(unsafe {
                (api.cu_memcpy_dtoh_v2)(b_out.as_mut_ptr().cast(), db, bytes)
            })?;
            Ok((a_out, b_out))
        })();

        let _ = unsafe { (api.cu_mem_free_v2)(da) };
        let _ = unsafe { (api.cu_mem_free_v2)(db) };

        let (a_out, b_out) = result.expect("multi-stream round-trip");
        // A was doubled twice (stream A, then stream B after the event) => x4.
        for (i, &v) in a_out.iter().enumerate() {
            assert!(
                (v - 4.0 * i as f32).abs() <= 1e-4,
                "stream A buffer element {i}: got {v}, expected {}",
                4.0 * i as f32
            );
        }
        // B was doubled once on stream B => x2.
        for (i, &v) in b_out.iter().enumerate() {
            let want = 2.0 * (i as f32 + 1000.0);
            assert!(
                (v - want).abs() <= 1e-3,
                "stream B buffer element {i}: got {v}, expected {want}"
            );
        }
    }
}
