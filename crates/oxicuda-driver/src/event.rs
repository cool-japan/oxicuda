//! CUDA event management for timing and synchronisation.
//!
//! Events can be recorded on a stream and used to measure elapsed time
//! between GPU operations or to synchronise streams.
//!
//! # Timing example
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use oxicuda_driver::event::Event;
//! # use oxicuda_driver::stream::Stream;
//! # use oxicuda_driver::context::Context;
//! # fn main() -> Result<(), oxicuda_driver::error::CudaError> {
//! # let ctx: Arc<Context> = unimplemented!();
//! let stream = Stream::new(&ctx)?;
//! let start = Event::new()?;
//! let end = Event::new()?;
//!
//! start.record(&stream)?;
//! // ... enqueue GPU work on `stream` ...
//! end.record(&stream)?;
//! end.synchronize()?;
//!
//! let ms = Event::elapsed_time(&start, &end)?;
//! println!("kernel took {ms:.3} ms");
//! # Ok(())
//! # }
//! ```

use crate::error::CudaResult;
use crate::ffi::{CU_EVENT_DEFAULT, CUevent};
use crate::loader::try_driver;
use crate::stream::Stream;

/// A CUDA event for timing and synchronisation.
///
/// Events are lightweight markers that can be recorded into a
/// [`Stream`]. They support two primary use-cases:
///
/// 1. **Timing** — measure elapsed GPU time between two recorded events
///    via [`Event::elapsed_time`].
/// 2. **Synchronisation** — make one stream wait for work recorded in
///    another stream via [`Stream::wait_event`].
pub struct Event {
    /// Raw CUDA event handle.
    raw: CUevent,
}

// SAFETY: CUDA events are safe to send between threads when properly
// synchronised via the driver API.
unsafe impl Send for Event {}

impl Event {
    /// Creates a new event with [`CU_EVENT_DEFAULT`] flags.
    ///
    /// Default events record timing data. Use [`Event::with_flags`] to
    /// create events with different characteristics (e.g. disable timing
    /// for lower overhead).
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the driver
    /// call fails.
    pub fn new() -> CudaResult<Self> {
        Self::with_flags(CU_EVENT_DEFAULT)
    }

    /// Creates a new event with the specified flags.
    ///
    /// Common flag values (from [`crate::ffi`]):
    ///
    /// | Constant                  | Value | Description                    |
    /// |---------------------------|-------|--------------------------------|
    /// | `CU_EVENT_DEFAULT`        | 0     | Default (records timing)       |
    /// | `CU_EVENT_BLOCKING_SYNC`  | 1     | Use blocking synchronisation   |
    /// | `CU_EVENT_DISABLE_TIMING` | 2     | Disable timing (lower overhead)|
    /// | `CU_EVENT_INTERPROCESS`   | 4     | Usable across processes        |
    ///
    /// Flags can be combined with bitwise OR.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the flags
    /// are invalid or the driver call otherwise fails.
    pub fn with_flags(flags: u32) -> CudaResult<Self> {
        let api = try_driver()?;
        let mut raw = CUevent::default();
        crate::cuda_call!((api.cu_event_create)(&mut raw, flags))?;
        Ok(Self { raw })
    }

    /// Records this event on the given stream.
    ///
    /// The event captures the point in the stream's command queue at
    /// which it was recorded. Subsequent calls to [`Event::synchronize`]
    /// or [`Event::elapsed_time`] reference this recorded point.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the stream
    /// or event handle is invalid.
    pub fn record(&self, stream: &Stream) -> CudaResult<()> {
        let api = try_driver()?;
        crate::cuda_call!((api.cu_event_record)(self.raw, stream.raw()))
    }

    /// Queries whether this event has completed.
    ///
    /// Returns `Ok(true)` if the event (and all preceding work in its
    /// stream) has completed, `Ok(false)` if it is still pending.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the event
    /// was not recorded or an unexpected driver error occurs (errors
    /// other than `NotReady`).
    pub fn query(&self) -> CudaResult<bool> {
        let api = try_driver()?;
        let rc = unsafe { (api.cu_event_query)(self.raw) };
        if rc == 0 {
            Ok(true)
        } else if rc == crate::ffi::CUDA_ERROR_NOT_READY {
            Ok(false)
        } else {
            Err(crate::error::CudaError::from_raw(rc))
        }
    }

    /// Blocks the calling thread until this event has been recorded
    /// and all preceding work in its stream has completed.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the event
    /// was not recorded or the driver reports an error.
    pub fn synchronize(&self) -> CudaResult<()> {
        let api = try_driver()?;
        crate::cuda_call!((api.cu_event_synchronize)(self.raw))
    }

    /// Computes the elapsed time in milliseconds between two recorded
    /// events.
    ///
    /// Both `start` and `end` must have been previously recorded on a
    /// stream, and `end` must have completed (e.g. via
    /// [`Event::synchronize`]).
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if either event
    /// has not been recorded, or if timing data is not available (e.g.
    /// the events were created with `CU_EVENT_DISABLE_TIMING`).
    pub fn elapsed_time(start: &Event, end: &Event) -> CudaResult<f32> {
        let api = try_driver()?;
        let mut ms: f32 = 0.0;
        crate::cuda_call!((api.cu_event_elapsed_time)(&mut ms, start.raw, end.raw))?;
        Ok(ms)
    }

    /// Returns the raw [`CUevent`] handle.
    ///
    /// # Safety (caller)
    ///
    /// The caller must not destroy or otherwise invalidate the handle
    /// while this `Event` is still alive.
    #[inline]
    pub fn raw(&self) -> CUevent {
        self.raw
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(api) = try_driver() {
            let rc = unsafe { (api.cu_event_destroy_v2)(self.raw) };
            if rc != 0 {
                tracing::warn!(
                    cuda_error = rc,
                    event = ?self.raw,
                    "cuEventDestroy_v2 failed during drop"
                );
            }
        }
    }
}
