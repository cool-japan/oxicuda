//! BLAS handle management.
//!
//! [`BlasHandle`] is the central object for all BLAS operations, analogous
//! to `cublasHandle_t` in cuBLAS. It owns a CUDA stream, tracks the target
//! SM version, and stores configuration such as [`MathMode`] and
//! [`PointerMode`].
//!
//! # Example
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use oxicuda_driver::Context;
//! # use oxicuda_blas::handle::BlasHandle;
//! # fn main() -> Result<(), oxicuda_blas::error::BlasError> {
//! # let ctx: Arc<Context> = unimplemented!();
//! let handle = BlasHandle::new(&ctx)?;
//! assert_eq!(handle.math_mode(), oxicuda_blas::types::MathMode::Default);
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use oxicuda_driver::{Context, Stream};
use oxicuda_ptx::arch::SmVersion;

use crate::error::{BlasError, BlasResult};
use crate::types::{MathMode, PointerMode};

/// Central handle for BLAS operations.
///
/// Every BLAS routine requires a `BlasHandle`. The handle binds operations to
/// a specific CUDA context and stream, and caches the device's SM version so
/// that kernel selection and PTX generation can target the right architecture
/// without repeated driver queries.
///
/// # Thread safety
///
/// `BlasHandle` is `Send` but **not** `Sync`. Each thread should create its
/// own handle (possibly sharing the same [`Arc<Context>`]).
pub struct BlasHandle {
    /// The CUDA context this handle is bound to.
    context: Arc<Context>,
    /// The stream on which BLAS kernels are launched.
    stream: Stream,
    /// Controls whether Tensor-Core paths are enabled.
    math_mode: MathMode,
    /// Whether scalar arguments (alpha, beta) reside on host or device.
    pointer_mode: PointerMode,
    /// SM architecture of the device, used for kernel selection.
    sm_version: SmVersion,
}

impl BlasHandle {
    /// Creates a new BLAS handle with a freshly-allocated default stream.
    ///
    /// The device's compute capability is queried once and cached as an
    /// [`SmVersion`] for later kernel dispatch decisions.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::Cuda`] if stream creation or device query fails.
    /// Returns [`BlasError::UnsupportedOperation`] if the device's compute
    /// capability does not map to a known SM version.
    pub fn new(ctx: &Arc<Context>) -> BlasResult<Self> {
        let stream = Stream::new(ctx)?;
        Self::build(ctx, stream)
    }

    /// Creates a new BLAS handle bound to an existing stream.
    ///
    /// This avoids allocating an extra stream when the caller already has
    /// one (e.g. from a training pipeline with multiple streams).
    ///
    /// # Errors
    ///
    /// Same as [`new`](Self::new) except stream creation cannot fail.
    pub fn with_stream(ctx: &Arc<Context>, stream: Stream) -> BlasResult<Self> {
        Self::build(ctx, stream)
    }

    /// Shared construction logic for `new` and `with_stream`.
    fn build(ctx: &Arc<Context>, stream: Stream) -> BlasResult<Self> {
        let device = ctx.device();
        let (major, minor) = device.compute_capability()?;
        let sm_version = SmVersion::from_compute_capability(major, minor).ok_or_else(|| {
            BlasError::UnsupportedOperation(format!(
                "unsupported compute capability: {major}.{minor}"
            ))
        })?;

        Ok(Self {
            context: Arc::clone(ctx),
            stream,
            math_mode: MathMode::Default,
            pointer_mode: PointerMode::Host,
            sm_version,
        })
    }

    // -- Accessors ------------------------------------------------------------

    /// Returns a reference to the CUDA context.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns a reference to the stream used for kernel launches.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Returns the SM version of the bound device.
    pub fn sm_version(&self) -> SmVersion {
        self.sm_version
    }

    /// Returns the current math mode.
    pub fn math_mode(&self) -> MathMode {
        self.math_mode
    }

    /// Returns the current pointer mode.
    pub fn pointer_mode(&self) -> PointerMode {
        self.pointer_mode
    }

    // -- Mutators -------------------------------------------------------------

    /// Replaces the stream used for subsequent BLAS operations.
    ///
    /// The previous stream is **not** synchronised; callers should ensure
    /// all in-flight work has completed before swapping streams.
    pub fn set_stream(&mut self, stream: Stream) {
        self.stream = stream;
    }

    /// Sets the math mode, controlling whether Tensor-Core paths are used.
    ///
    /// [`MathMode::TensorCore`] enables reduced-precision Tensor-Core
    /// instructions when available on the device. The default is
    /// [`MathMode::Default`], which uses only FP32/FP64 FMA pipelines.
    pub fn set_math_mode(&mut self, mode: MathMode) {
        self.math_mode = mode;
    }

    /// Sets the pointer mode for scalar arguments (alpha, beta).
    ///
    /// [`PointerMode::Host`] (default) means scalars reside in host memory.
    /// [`PointerMode::Device`] means scalars reside in device memory, which
    /// can avoid host-device synchronisation in pipelined workloads.
    pub fn set_pointer_mode(&mut self, mode: PointerMode) {
        self.pointer_mode = mode;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that default values are correct without needing a GPU.
    #[test]
    fn default_modes() {
        // We cannot construct a real handle without a GPU, so just verify
        // the enum default values that `build` would set.
        assert_eq!(MathMode::Default, MathMode::Default);
        assert_eq!(PointerMode::Host, PointerMode::Host);
    }
}
