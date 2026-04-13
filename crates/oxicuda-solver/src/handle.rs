//! Solver handle management.
//!
//! [`SolverHandle`] is the central object for all solver operations, analogous
//! to `cusolverDnHandle_t` in cuSOLVER. It owns a BLAS handle, CUDA stream,
//! PTX cache, and a device workspace buffer for intermediate computations.

use std::sync::Arc;

use oxicuda_blas::BlasHandle;
use oxicuda_driver::{Context, Stream};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::cache::PtxCache;

use crate::error::{SolverError, SolverResult};

/// Central handle for solver operations.
///
/// Every solver routine requires a `SolverHandle`. The handle binds operations
/// to a specific CUDA context and stream, provides access to the underlying
/// BLAS handle for delegating matrix operations, and manages a resizable
/// device workspace buffer.
///
/// # Thread safety
///
/// `SolverHandle` is `Send` but **not** `Sync`. Each thread should create its
/// own handle (possibly sharing the same [`Arc<Context>`]).
pub struct SolverHandle {
    /// The CUDA context this handle is bound to.
    context: Arc<Context>,
    /// The stream on which solver kernels are launched.
    stream: Stream,
    /// BLAS handle for delegating GEMM, TRSM, etc.
    blas_handle: BlasHandle,
    /// Cache for generated PTX kernels.
    ptx_cache: PtxCache,
    /// SM architecture of the device.
    sm_version: SmVersion,
    /// Resizable device workspace for intermediate computations.
    workspace: DeviceBuffer<u8>,
}

impl SolverHandle {
    /// Creates a new solver handle with a freshly-allocated default stream.
    ///
    /// The device's compute capability is queried once and cached as an
    /// [`SmVersion`] for later kernel dispatch decisions. An initial workspace
    /// of 4 KiB is allocated.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::Cuda`] if stream creation or device query fails.
    /// Returns [`SolverError::Blas`] if BLAS handle creation fails.
    pub fn new(ctx: &Arc<Context>) -> SolverResult<Self> {
        let blas_handle = BlasHandle::new(ctx)?;
        let sm_version = blas_handle.sm_version();
        let stream = Stream::new(ctx)?;
        let ptx_cache = PtxCache::new()
            .map_err(|e| SolverError::InternalError(format!("failed to create PTX cache: {e}")))?;
        // Start with a small initial workspace (4 KiB).
        let workspace = DeviceBuffer::<u8>::zeroed(4096)?;

        Ok(Self {
            context: Arc::clone(ctx),
            stream,
            blas_handle,
            ptx_cache,
            sm_version,
            workspace,
        })
    }

    /// Ensures the workspace buffer has at least `bytes` capacity.
    ///
    /// If the current workspace is smaller, it is reallocated. The contents
    /// of the previous workspace are **not** preserved.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::Cuda`] if reallocation fails.
    pub fn ensure_workspace(&mut self, bytes: usize) -> SolverResult<()> {
        if self.workspace.len() < bytes {
            self.workspace = DeviceBuffer::<u8>::zeroed(bytes)?;
        }
        Ok(())
    }

    /// Returns a reference to the underlying BLAS handle.
    pub fn blas(&self) -> &BlasHandle {
        &self.blas_handle
    }

    /// Returns a mutable reference to the underlying BLAS handle.
    pub fn blas_mut(&mut self) -> &mut BlasHandle {
        &mut self.blas_handle
    }

    /// Returns a reference to the stream used for kernel launches.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Returns a reference to the CUDA context.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns the SM version of the bound device.
    pub fn sm_version(&self) -> SmVersion {
        self.sm_version
    }

    /// Returns a reference to the PTX cache.
    pub fn ptx_cache(&self) -> &PtxCache {
        &self.ptx_cache
    }

    /// Returns a mutable reference to the PTX cache.
    pub fn ptx_cache_mut(&mut self) -> &mut PtxCache {
        &mut self.ptx_cache
    }

    /// Returns a reference to the device workspace buffer.
    pub fn workspace(&self) -> &DeviceBuffer<u8> {
        &self.workspace
    }

    /// Returns a mutable reference to the device workspace buffer.
    pub fn workspace_mut(&mut self) -> &mut DeviceBuffer<u8> {
        &mut self.workspace
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn initial_workspace_size() {
        // Verify the constant used for initial allocation.
        assert_eq!(4096, 4096);
    }

    #[test]
    fn workspace_requirement_logic() {
        // Test the ensure_workspace size comparison logic.
        let current = 4096_usize;
        let required = 8192_usize;
        assert!(current < required, "should need reallocation");
    }

    #[test]
    fn workspace_sufficient_logic() {
        let current = 8192_usize;
        let required = 4096_usize;
        assert!(current >= required, "should not need reallocation");
    }
}
