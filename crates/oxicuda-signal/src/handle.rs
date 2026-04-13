//! Session handle for GPU signal processing operations.

use std::sync::Arc;

use oxicuda_driver::{Context, Stream};

use crate::error::SignalResult;

/// Central handle that owns GPU context and stream for signal operations.
///
/// All compute-intensive signal processing functions take a `&SignalHandle` so
/// they share the same CUDA context and stream, enabling implicit serialisation
/// without extra synchronisation overhead.
///
/// # Example
/// ```rust,no_run
/// use oxicuda_signal::handle::SignalHandle;
/// let handle = SignalHandle::new().unwrap();
/// ```
#[derive(Clone)]
pub struct SignalHandle {
    context: Arc<Context>,
    stream: Arc<Stream>,
}

impl SignalHandle {
    /// Create a handle on the default GPU device using the primary context
    /// and a new stream.
    ///
    /// Returns `SignalError::Cuda` if no CUDA-capable GPU is available.
    pub fn new() -> SignalResult<Self> {
        use oxicuda_driver::{best_device, device::Device, init};
        init()?;
        let device = if let Some(d) = best_device()? {
            d
        } else {
            Device::get(0)?
        };
        let context = Arc::new(Context::new(&device)?);
        let stream = Arc::new(Stream::new(&context)?);
        Ok(Self { context, stream })
    }

    /// Create a handle from an existing context and stream (for advanced users
    /// who manage their own CUDA resources).
    #[must_use]
    pub fn from_parts(context: Arc<Context>, stream: Arc<Stream>) -> Self {
        Self { context, stream }
    }

    /// Returns a reference to the underlying CUDA context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns a reference to the CUDA stream used for asynchronous launches.
    #[must_use]
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Block the CPU until all GPU operations queued on this handle's stream
    /// have completed.
    pub fn synchronize(&self) -> SignalResult<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

impl std::fmt::Debug for SignalHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignalHandle").finish_non_exhaustive()
    }
}
