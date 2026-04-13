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
use crate::ffi::{CU_STREAM_NON_BLOCKING, CUstream};
use crate::loader::try_driver;

/// A CUDA stream (GPU command queue).
///
/// Streams provide ordered, asynchronous execution of GPU commands.
/// Commands enqueued on the same stream execute sequentially, while
/// commands on different streams may execute concurrently.
///
/// The stream holds an [`Arc<Context>`] to ensure the parent context
/// outlives the stream.
pub struct Stream {
    /// Raw CUDA stream handle.
    raw: CUstream,
    /// Keeps the parent context alive for the lifetime of the stream.
    ctx: Arc<Context>,
}

// SAFETY: CUDA streams are safe to send between threads when properly
// synchronised via the driver API. The Arc<Context> is already Send.
unsafe impl Send for Stream {}

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
        let mut raw = CUstream::default();
        crate::cuda_call!((api.cu_stream_create)(&mut raw, CU_STREAM_NON_BLOCKING))?;
        Ok(Self {
            raw,
            ctx: Arc::clone(ctx),
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
        let mut raw = CUstream::default();
        crate::cuda_call!((api.cu_stream_create_with_priority)(
            &mut raw,
            CU_STREAM_NON_BLOCKING,
            priority
        ))?;
        Ok(Self {
            raw,
            ctx: Arc::clone(ctx),
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
        crate::cuda_call!((api.cu_stream_synchronize)(self.raw))
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
        crate::cuda_call!((api.cu_stream_wait_event)(self.raw, event.raw(), 0))
    }

    /// Returns the raw [`CUstream`] handle.
    ///
    /// # Safety (caller)
    ///
    /// The caller must not destroy or otherwise invalidate the handle
    /// while this `Stream` is still alive.
    #[inline]
    pub fn raw(&self) -> CUstream {
        self.raw
    }

    /// Returns a reference to the parent [`Context`].
    #[inline]
    pub fn context(&self) -> &Arc<Context> {
        &self.ctx
    }
}

impl Drop for Stream {
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
