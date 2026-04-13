//! Sparse handle management.
//!
//! [`SparseHandle`] is the central object for all sparse operations, analogous
//! to `cusparseHandle_t` in cuSPARSE. It owns a CUDA stream, a [`BlasHandle`]
//! for dense sub-operations, a [`PtxCache`] for kernel caching, and caches the
//! target SM version.
//!
//! # Example
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use oxicuda_driver::Context;
//! # use oxicuda_sparse::handle::SparseHandle;
//! # fn main() -> Result<(), oxicuda_sparse::error::SparseError> {
//! # let ctx: Arc<Context> = unimplemented!();
//! let handle = SparseHandle::new(&ctx)?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use oxicuda_blas::BlasHandle;
use oxicuda_driver::{Context, Stream};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::cache::PtxCache;

use crate::error::{SparseError, SparseResult};

/// Central handle for sparse matrix operations.
///
/// Every sparse routine requires a `SparseHandle`. The handle binds operations
/// to a specific CUDA context and stream, maintains a BLAS sub-handle for
/// dense operations (e.g. SpMM output accumulation), caches compiled PTX
/// kernels, and stores the device's SM version for kernel dispatch.
///
/// # Thread safety
///
/// `SparseHandle` is `Send` but **not** `Sync`. Each thread should create its
/// own handle (possibly sharing the same [`Arc<Context>`]).
pub struct SparseHandle {
    /// The CUDA context this handle is bound to.
    context: Arc<Context>,
    /// The stream on which sparse kernels are launched.
    stream: Stream,
    /// Sub-handle for BLAS operations.
    blas_handle: BlasHandle,
    /// On-disk cache for compiled PTX modules.
    ptx_cache: PtxCache,
    /// SM architecture of the device, cached for kernel selection.
    sm_version: SmVersion,
}

impl SparseHandle {
    /// Creates a new sparse handle with a freshly-allocated default stream.
    ///
    /// The device's compute capability is queried once and cached.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] if stream creation or device query fails.
    pub fn new(ctx: &Arc<Context>) -> SparseResult<Self> {
        let stream = Stream::new(ctx)?;
        Self::build(ctx, stream)
    }

    /// Creates a new sparse handle bound to an existing stream.
    ///
    /// # Errors
    ///
    /// Same as [`new`](Self::new) except stream creation cannot fail.
    pub fn with_stream(ctx: &Arc<Context>, stream: Stream) -> SparseResult<Self> {
        Self::build(ctx, stream)
    }

    /// Shared construction logic.
    fn build(ctx: &Arc<Context>, stream: Stream) -> SparseResult<Self> {
        let device = ctx.device();
        let (major, minor) = device.compute_capability()?;
        let sm_version = SmVersion::from_compute_capability(major, minor).ok_or_else(|| {
            SparseError::InternalError(format!("unsupported compute capability: {major}.{minor}"))
        })?;

        let blas_stream = Stream::new(ctx)?;
        let blas_handle = BlasHandle::with_stream(ctx, blas_stream)?;
        let ptx_cache = PtxCache::new()?;

        Ok(Self {
            context: Arc::clone(ctx),
            stream,
            blas_handle,
            ptx_cache,
            sm_version,
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
    pub fn blas_handle(&self) -> &BlasHandle {
        &self.blas_handle
    }

    /// Returns a mutable reference to the internal BLAS handle.
    #[inline]
    pub fn blas_handle_mut(&mut self) -> &mut BlasHandle {
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

    // -- Mutators ------------------------------------------------------------

    /// Replaces the stream used for subsequent sparse operations.
    pub fn set_stream(&mut self, stream: Stream) {
        self.stream = stream;
    }
}

#[cfg(test)]
mod tests {
    use oxicuda_ptx::arch::SmVersion;

    #[test]
    fn sm_version_is_copy() {
        let v = SmVersion::Sm80;
        let v2 = v;
        assert_eq!(v, v2);
    }
}
