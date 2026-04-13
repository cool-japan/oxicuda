//! DNN handle management.
//!
//! [`DnnHandle`] is the central object for all DNN operations, analogous to
//! `cudnnHandle_t` in cuDNN.  It owns a CUDA stream, a [`BlasHandle`] for
//! matrix operations, a [`PtxCache`] for kernel caching, and an optional
//! workspace buffer.
//!
//! # Example
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use oxicuda_driver::Context;
//! # use oxicuda_dnn::handle::DnnHandle;
//! # fn main() -> Result<(), oxicuda_dnn::error::DnnError> {
//! # let ctx: Arc<Context> = unimplemented!();
//! let mut handle = DnnHandle::new(&ctx)?;
//! handle.set_workspace(1 << 20)?; // 1 MiB workspace
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use oxicuda_blas::BlasHandle;
use oxicuda_driver::{Context, Stream};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::cache::PtxCache;

use crate::error::{DnnError, DnnResult};

/// Central handle for DNN operations.
///
/// Every DNN routine requires a `DnnHandle`.  The handle binds operations to
/// a specific CUDA context and stream, maintains a BLAS sub-handle for GEMM
/// workloads (e.g. im2col convolutions), caches compiled PTX kernels, and
/// optionally holds a pre-allocated workspace buffer.
///
/// # Thread safety
///
/// `DnnHandle` is `Send` but **not** `Sync`.  Each thread should create its
/// own handle (possibly sharing the same [`Arc<Context>`]).
pub struct DnnHandle {
    /// The CUDA context this handle is bound to.
    context: Arc<Context>,
    /// The stream on which DNN kernels are launched.
    stream: Stream,
    /// Sub-handle for BLAS operations (GEMM inside convolution, etc.).
    blas_handle: BlasHandle,
    /// On-disk cache for compiled PTX modules.
    ptx_cache: PtxCache,
    /// SM architecture of the device, cached for kernel selection.
    sm_version: SmVersion,
    /// Optional pre-allocated workspace buffer for algorithms that need
    /// temporary storage (e.g. im2col, Winograd transforms).
    workspace: Option<DeviceBuffer<u8>>,
}

impl DnnHandle {
    /// Creates a new DNN handle with a freshly-allocated default stream.
    ///
    /// The device's compute capability is queried once and cached.  A
    /// [`BlasHandle`] and [`PtxCache`] are created internally.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::Cuda`] if stream creation or device query fails.
    /// Returns [`DnnError::Blas`] if the BLAS handle cannot be created.
    /// Returns [`DnnError::Io`] if the PTX cache directory cannot be created.
    pub fn new(ctx: &Arc<Context>) -> DnnResult<Self> {
        let stream = Stream::new(ctx)?;
        Self::build(ctx, stream)
    }

    /// Creates a new DNN handle bound to an existing stream.
    ///
    /// This avoids allocating an extra stream when the caller already has
    /// one (e.g. from a training pipeline).
    ///
    /// # Errors
    ///
    /// Same as [`new`](Self::new) except stream creation cannot fail.
    pub fn with_stream(ctx: &Arc<Context>, stream: Stream) -> DnnResult<Self> {
        Self::build(ctx, stream)
    }

    /// Shared construction logic.
    fn build(ctx: &Arc<Context>, stream: Stream) -> DnnResult<Self> {
        let device = ctx.device();
        let (major, minor) = device.compute_capability()?;
        let sm_version = SmVersion::from_compute_capability(major, minor).ok_or_else(|| {
            DnnError::UnsupportedOperation(format!(
                "unsupported compute capability: {major}.{minor}"
            ))
        })?;

        // Create a separate stream for the internal BLAS handle so that
        // BLAS and DNN launches can be overlapped when appropriate.
        let blas_stream = Stream::new(ctx)?;
        let blas_handle = BlasHandle::with_stream(ctx, blas_stream)?;
        let ptx_cache = PtxCache::new()?;

        Ok(Self {
            context: Arc::clone(ctx),
            stream,
            blas_handle,
            ptx_cache,
            sm_version,
            workspace: None,
        })
    }

    // -- Accessors -----------------------------------------------------------

    /// Returns a reference to the CUDA context.
    #[inline]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns a reference to the stream used for kernel launches.
    #[inline]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Returns a reference to the internal BLAS handle.
    #[inline]
    pub fn blas(&self) -> &BlasHandle {
        &self.blas_handle
    }

    /// Returns a mutable reference to the internal BLAS handle.
    #[inline]
    pub fn blas_mut(&mut self) -> &mut BlasHandle {
        &mut self.blas_handle
    }

    /// Returns the SM version of the bound device.
    #[inline]
    pub fn sm_version(&self) -> SmVersion {
        self.sm_version
    }

    /// Returns a reference to the PTX cache.
    #[inline]
    pub fn ptx_cache(&self) -> &PtxCache {
        &self.ptx_cache
    }

    /// Returns a reference to the workspace buffer, if one has been allocated.
    #[inline]
    pub fn workspace(&self) -> Option<&DeviceBuffer<u8>> {
        self.workspace.as_ref()
    }

    /// Returns a mutable reference to the workspace buffer, if allocated.
    #[inline]
    pub fn workspace_mut(&mut self) -> Option<&mut DeviceBuffer<u8>> {
        self.workspace.as_mut()
    }

    // -- Mutators ------------------------------------------------------------

    /// Replaces the stream used for subsequent DNN operations.
    ///
    /// The previous stream is **not** synchronised; the caller must ensure
    /// all in-flight work has completed before swapping streams.
    pub fn set_stream(&mut self, stream: Stream) {
        self.stream = stream;
    }

    /// Allocates (or re-allocates) the workspace buffer with at least `bytes`
    /// bytes of device memory.
    ///
    /// If the current workspace is already large enough, this is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `bytes` is zero.
    /// Returns [`DnnError::Cuda`] if device memory allocation fails.
    pub fn set_workspace(&mut self, bytes: usize) -> DnnResult<()> {
        if bytes == 0 {
            return Err(DnnError::InvalidArgument(
                "workspace size must be non-zero".into(),
            ));
        }
        // Skip re-allocation if current workspace is already sufficient.
        if let Some(ref ws) = self.workspace {
            if ws.len() >= bytes {
                return Ok(());
            }
        }
        let buf = DeviceBuffer::<u8>::alloc(bytes)?;
        self.workspace = Some(buf);
        Ok(())
    }

    /// Drops the current workspace buffer, freeing device memory.
    pub fn clear_workspace(&mut self) {
        self.workspace = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SmVersion must be Copy for the getter to return by value.
    #[test]
    fn sm_version_is_copy() {
        let v = SmVersion::Sm80;
        let v2 = v;
        assert_eq!(v, v2);
    }
}
