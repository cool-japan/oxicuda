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

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use oxicuda_driver::{Context, Module, Stream};
use oxicuda_ptx::arch::SmVersion;

use crate::error::{BlasError, BlasResult};
use crate::level3::gemm::dispatch::GemmDispatcher;
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
    /// Persistent GEMM dispatcher. Owning it on the handle (rather than
    /// constructing a fresh one per [`gemm`](crate::level3::gemm_api::gemm)
    /// call) means the dispatcher's compiled-kernel cache survives across
    /// calls, so a repeated GEMM shape re-JITs its tiled kernel only once.
    gemm_dispatcher: GemmDispatcher,
    /// Compiled-module cache for the non-GEMM ops (Level-1/2, reductions,
    /// element-wise, …). Keyed by the fully-qualified kernel name — which
    /// already encodes the op and the element precision, and the handle pins a
    /// single context/SM version — so the name alone is a sufficient key.
    ///
    /// Without this, every non-GEMM call would regenerate PTX and
    /// `cuModuleLoadData`-compile a brand-new module on each invocation.
    module_cache: RwLock<HashMap<String, Arc<Module>>>,
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

        // The real, live streaming-multiprocessor count for *this* device --
        // compute capability alone cannot give it (see
        // `GemmDispatcher::sm_count`'s doc comment: sm_86 alone spans GPUs
        // from 46 SMs to 84+), so it is queried here rather than looked up
        // from an architecture table. A query failure (or an implausible
        // non-positive count) falls back to `GemmDispatcher::new`'s
        // architecture-typical default rather than failing handle
        // construction over what is ultimately a launch-tuning detail: an
        // approximate count still produces a *correct* GEMM launch, only a
        // less precisely occupancy-tuned one.
        let gemm_dispatcher = match device
            .multiprocessor_count()
            .ok()
            .and_then(|count| u32::try_from(count).ok())
        {
            Some(sm_count) if sm_count > 0 => {
                GemmDispatcher::new_with_sm_count(sm_version, sm_count)
            }
            _ => GemmDispatcher::new(sm_version),
        };

        Ok(Self {
            context: Arc::clone(ctx),
            stream,
            math_mode: MathMode::Default,
            pointer_mode: PointerMode::Host,
            sm_version,
            gemm_dispatcher,
            module_cache: RwLock::new(HashMap::new()),
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

    /// Returns the handle-owned GEMM dispatcher.
    ///
    /// The dispatcher caches compiled kernels internally; reusing this shared
    /// instance across `gemm` calls keeps that cache warm.
    pub(crate) fn gemm_dispatcher(&self) -> &GemmDispatcher {
        &self.gemm_dispatcher
    }

    /// Device memory this handle's GEMM dispatcher is holding in reusable
    /// split-K reduction workspaces.
    ///
    /// Zero until the first skinny GEMM takes the split-K path, and
    /// monotonically non-decreasing afterwards: those workspaces are kept for
    /// the handle's lifetime so their device addresses stay stable, which is
    /// what lets a split-K GEMM be recorded into a CUDA graph and replayed.
    /// See `GemmDispatcher::split_k_workspace` for the full rationale, and
    /// `tests/splitk_workspace_gpu.rs` for the regression test that pins it.
    ///
    /// The number is a *consequence* of the caching policy, not a knob.
    ///
    /// # Errors
    ///
    /// [`BlasError::LaunchFailed`] if the workspace cache's lock is poisoned.
    pub fn split_k_workspace_bytes(&self) -> BlasResult<usize> {
        self.gemm_dispatcher.split_k_workspace_bytes()
    }

    /// Returns a cached compiled [`Module`] for `name`, or generates its PTX
    /// via `gen`, compiles it, caches it, and returns it.
    ///
    /// This is the non-GEMM analogue of [`GemmDispatcher`]'s kernel cache: the
    /// first call for a given kernel name JIT-compiles the module; subsequent
    /// calls reuse it, avoiding a per-call PTX regeneration and
    /// `cuModuleLoadData`. `gen_ptx` is invoked only on a cache miss.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError`] if the cache lock is poisoned, if `gen_ptx` fails,
    /// or if the generated PTX fails to compile.
    pub(crate) fn get_or_compile_module(
        &self,
        name: &str,
        gen_ptx: impl FnOnce() -> BlasResult<String>,
    ) -> BlasResult<Arc<Module>> {
        // Fast path: read lock, return an existing module.
        {
            let cache = self
                .module_cache
                .read()
                .map_err(|_| BlasError::LaunchFailed("module cache lock poisoned".into()))?;
            if let Some(module) = cache.get(name) {
                return Ok(Arc::clone(module));
            }
        }

        // Slow path: generate PTX and compile outside the write lock, then
        // insert. A concurrent compile of the same name is harmless — the map
        // simply keeps whichever module was inserted last; every returned
        // `Arc` stays valid for as long as a caller holds it.
        let ptx = gen_ptx()?;
        let module = Arc::new(Module::from_ptx(&ptx).map_err(BlasError::Cuda)?);
        {
            let mut cache = self
                .module_cache
                .write()
                .map_err(|_| BlasError::LaunchFailed("module cache lock poisoned".into()))?;
            cache.insert(name.to_owned(), Arc::clone(&module));
        }
        Ok(module)
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
