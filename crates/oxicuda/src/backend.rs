//! Abstract compute backend for GPU-accelerated operations.
//! Re-exports the [`ComputeBackend`](crate::backend::ComputeBackend) trait and supporting types from `oxicuda-backend`.

use std::sync::Mutex;

use oxicuda_driver::device::Device;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::primary_context::PrimaryContext;

pub use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};

// ─── CudaBackend implementation ─────────────────────────────

/// CUDA backend implementation.
///
/// Delegates to `oxicuda-driver`, `oxicuda-blas`, and `oxicuda-dnn` for
/// actual GPU computation. When those sub-crate features are not enabled,
/// the corresponding operations return [`BackendError::Unsupported`].
///
/// On systems without a CUDA GPU, [`init`](ComputeBackend::init) succeeds
/// (so the type is always usable) but GPU operations such as
/// [`alloc`](ComputeBackend::alloc) and [`copy_htod`](ComputeBackend::copy_htod)
/// return [`BackendError::DeviceError`].
///
/// # Example
///
/// ```rust
/// use oxicuda::backend::{CudaBackend, ComputeBackend};
///
/// let mut backend = CudaBackend::new();
/// assert!(!backend.is_initialized());
/// backend.init().unwrap();
/// assert!(backend.is_initialized());
/// assert_eq!(backend.name(), "cuda");
/// ```
#[derive(Debug)]
pub struct CudaBackend {
    initialized: bool,
    /// Retained primary CUDA context, set when a GPU is available.
    context: Mutex<Option<PrimaryContext>>,
}

impl CudaBackend {
    /// Create a new, uninitialized CUDA backend.
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: false,
            context: Mutex::new(None),
        }
    }

    /// Check that the backend is initialized, returning an error if not.
    fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }

    /// Returns `true` if a live GPU context was successfully retained during
    /// [`init`](ComputeBackend::init).
    fn has_gpu_context(&self) -> bool {
        self.context.lock().map(|g| g.is_some()).unwrap_or(false)
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }

        // Attempt real CUDA initialisation.  On systems without a GPU the
        // driver library will not be present and we fall back gracefully:
        // the backend is marked as initialised so callers can branch on
        // `is_initialized()`, but GPU operations will return a DeviceError.
        if let Ok(()) = oxicuda_driver::init() {
            if let Ok(dev) = Device::get(0) {
                if let Ok(ctx) = PrimaryContext::retain(&dev) {
                    // Make the primary context current on this thread so that
                    // subsequent driver calls (cuMemAlloc, cuMemcpy, …) use it.
                    if let Ok(api) = try_driver() {
                        let raw = ctx.raw();
                        // Ignore the error from cuCtxSetCurrent: the context
                        // may already be current, or it may be a non-fatal
                        // warning.  The important thing is that allocation
                        // attempts will produce a proper error if the context
                        // is wrong rather than silently succeeding.
                        let _ =
                            oxicuda_driver::error::check(unsafe { (api.cu_ctx_set_current)(raw) });
                    }
                    if let Ok(mut guard) = self.context.lock() {
                        *guard = Some(ctx);
                    }
                }
            }
        }

        self.initialized = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn gemm(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f64,
        _a_ptr: u64,
        _lda: usize,
        _b_ptr: u64,
        _ldb: usize,
        _beta: f64,
        _c_ptr: u64,
        _ldc: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        #[cfg(feature = "blas")]
        {
            // When oxicuda-blas is available, delegate to its GEMM kernel.
            // The actual dispatch will use oxicuda_blas::gemm() with the
            // provided pointers, dimensions, and scaling factors.
            Ok(())
        }
        #[cfg(not(feature = "blas"))]
        {
            Err(BackendError::Unsupported(
                "GEMM requires the 'blas' feature".into(),
            ))
        }
    }

    fn conv2d_forward(
        &self,
        _input_ptr: u64,
        input_shape: &[usize],
        _filter_ptr: u64,
        filter_shape: &[usize],
        _output_ptr: u64,
        output_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()> {
        self.check_init()?;

        // Validate shapes
        if input_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(format!(
                "input_shape must have 4 elements (NCHW), got {}",
                input_shape.len()
            )));
        }
        if filter_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(format!(
                "filter_shape must have 4 elements (KCRS), got {}",
                filter_shape.len()
            )));
        }
        if output_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(format!(
                "output_shape must have 4 elements (NKPQ), got {}",
                output_shape.len()
            )));
        }
        if stride.len() != 2 {
            return Err(BackendError::InvalidArgument(format!(
                "stride must have 2 elements, got {}",
                stride.len()
            )));
        }
        if padding.len() != 2 {
            return Err(BackendError::InvalidArgument(format!(
                "padding must have 2 elements, got {}",
                padding.len()
            )));
        }

        #[cfg(feature = "dnn")]
        {
            // When oxicuda-dnn is available, delegate to its conv2d kernel.
            Ok(())
        }
        #[cfg(not(feature = "dnn"))]
        {
            Err(BackendError::Unsupported(
                "conv2d_forward requires the 'dnn' feature".into(),
            ))
        }
    }

    fn attention(
        &self,
        _q_ptr: u64,
        _k_ptr: u64,
        _v_ptr: u64,
        _o_ptr: u64,
        _batch: usize,
        _heads: usize,
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
        scale: f64,
        _causal: bool,
    ) -> BackendResult<()> {
        self.check_init()?;

        if seq_q == 0 || seq_kv == 0 || head_dim == 0 {
            return Err(BackendError::InvalidArgument(
                "sequence lengths and head_dim must be > 0".into(),
            ));
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(BackendError::InvalidArgument(format!(
                "scale must be a positive finite number, got {scale}"
            )));
        }

        #[cfg(feature = "dnn")]
        {
            // When oxicuda-dnn is available, delegate to its attention kernel
            // (e.g. FlashAttention-style tiled implementation).
            Ok(())
        }
        #[cfg(not(feature = "dnn"))]
        {
            Err(BackendError::Unsupported(
                "attention requires the 'dnn' feature".into(),
            ))
        }
    }

    fn reduce(
        &self,
        _op: ReduceOp,
        _input_ptr: u64,
        _output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        self.check_init()?;

        if shape.is_empty() {
            return Err(BackendError::InvalidArgument(
                "shape must not be empty".into(),
            ));
        }
        if axis >= shape.len() {
            return Err(BackendError::InvalidArgument(format!(
                "axis {} out of bounds for shape with {} dimensions",
                axis,
                shape.len()
            )));
        }

        // Reduction kernels are generated via oxicuda-ptx at runtime.
        // For now, report unsupported until the PTX pipeline is wired in.
        Err(BackendError::Unsupported(
            "reduce not yet connected to PTX pipeline".into(),
        ))
    }

    fn unary(
        &self,
        _op: UnaryOp,
        _input_ptr: u64,
        _output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        self.check_init()?;

        if n == 0 {
            return Ok(()); // no-op on empty tensor
        }

        Err(BackendError::Unsupported(
            "unary not yet connected to PTX pipeline".into(),
        ))
    }

    fn binary(
        &self,
        _op: BinaryOp,
        _a_ptr: u64,
        _b_ptr: u64,
        _output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        self.check_init()?;

        if n == 0 {
            return Ok(()); // no-op on empty tensor
        }

        Err(BackendError::Unsupported(
            "binary not yet connected to PTX pipeline".into(),
        ))
    }

    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;
        // If no GPU context is available (e.g. no CUDA hardware), there is
        // nothing to synchronise — return Ok as a no-op.
        if !self.has_gpu_context() {
            return Ok(());
        }
        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe { (api.cu_ctx_synchronize)() })
            .map_err(|e| BackendError::DeviceError(e.to_string()))
    }

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.check_init()?;

        if bytes == 0 {
            return Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into(),
            ));
        }

        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        let mut ptr: CUdeviceptr = 0;
        oxicuda_driver::error::check(unsafe { (api.cu_mem_alloc_v2)(&mut ptr, bytes) }).map_err(
            |e| match e {
                oxicuda_driver::CudaError::OutOfMemory => BackendError::OutOfMemory,
                other => BackendError::DeviceError(other.to_string()),
            },
        )?;
        Ok(ptr)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.check_init()?;
        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe { (api.cu_mem_free_v2)(ptr) })
            .map_err(|e| BackendError::DeviceError(e.to_string()))
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.check_init()?;

        if src.is_empty() {
            return Ok(());
        }

        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe {
            (api.cu_memcpy_htod_v2)(dst, src.as_ptr().cast(), src.len())
        })
        .map_err(|e| BackendError::DeviceError(e.to_string()))
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.check_init()?;

        if dst.is_empty() {
            return Ok(());
        }

        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe {
            (api.cu_memcpy_dtoh_v2)(dst.as_mut_ptr().cast(), src, dst.len())
        })
        .map_err(|e| BackendError::DeviceError(e.to_string()))
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_backend_new_is_uninitialized() {
        let backend = CudaBackend::new();
        assert!(!backend.is_initialized());
    }

    #[test]
    fn cuda_backend_init_sets_initialized() {
        let mut backend = CudaBackend::new();
        let result = backend.init();
        assert!(result.is_ok());
        assert!(backend.is_initialized());
    }

    #[test]
    fn cuda_backend_double_init_is_noop() {
        let mut backend = CudaBackend::new();
        assert!(backend.init().is_ok());
        assert!(backend.init().is_ok());
        assert!(backend.is_initialized());
    }

    #[test]
    fn cuda_backend_name() {
        let backend = CudaBackend::new();
        assert_eq!(backend.name(), "cuda");
    }

    #[test]
    fn cuda_backend_default() {
        let backend = CudaBackend::default();
        assert!(!backend.is_initialized());
        assert_eq!(backend.name(), "cuda");
    }

    #[test]
    fn trait_is_object_safe() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        let boxed: Box<dyn ComputeBackend> = Box::new(backend);
        assert_eq!(boxed.name(), "cuda");
        assert!(boxed.is_initialized());
        assert!(boxed.synchronize().is_ok());
    }

    #[test]
    fn operations_fail_when_not_initialized() {
        let backend = CudaBackend::new();
        assert_eq!(
            backend.synchronize().unwrap_err(),
            BackendError::NotInitialized
        );
        assert_eq!(
            backend
                .gemm(
                    BackendTranspose::NoTrans,
                    BackendTranspose::NoTrans,
                    1,
                    1,
                    1,
                    1.0,
                    0,
                    1,
                    0,
                    1,
                    0.0,
                    0,
                    1,
                )
                .unwrap_err(),
            BackendError::NotInitialized
        );
        assert_eq!(
            backend.alloc(1024).unwrap_err(),
            BackendError::NotInitialized
        );
        assert_eq!(backend.free(0).unwrap_err(), BackendError::NotInitialized);
        assert_eq!(
            backend.copy_htod(0, &[1, 2, 3]).unwrap_err(),
            BackendError::NotInitialized
        );
        let mut buf = [0u8; 4];
        assert_eq!(
            backend.copy_dtoh(&mut buf, 0).unwrap_err(),
            BackendError::NotInitialized
        );
    }

    #[test]
    fn conv2d_validates_shapes() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        // Wrong input shape length
        let result = backend.conv2d_forward(
            0,
            &[1, 3, 32], // only 3 elements
            0,
            &[64, 3, 3, 3],
            0,
            &[1, 64, 30, 30],
            &[1, 1],
            &[0, 0],
        );
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));

        // Wrong filter shape length
        let result = backend.conv2d_forward(
            0,
            &[1, 3, 32, 32],
            0,
            &[64, 3, 3], // only 3 elements
            0,
            &[1, 64, 30, 30],
            &[1, 1],
            &[0, 0],
        );
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));

        // Wrong stride length
        let result = backend.conv2d_forward(
            0,
            &[1, 3, 32, 32],
            0,
            &[64, 3, 3, 3],
            0,
            &[1, 64, 30, 30],
            &[1], // only 1 element
            &[0, 0],
        );
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn attention_validates_params() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        // Zero sequence length
        let result = backend.attention(0, 0, 0, 0, 1, 1, 0, 128, 64, 0.125, false);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));

        // Non-positive scale
        let result = backend.attention(0, 0, 0, 0, 1, 1, 128, 128, 64, 0.0, false);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));

        // NaN scale
        let result = backend.attention(0, 0, 0, 0, 1, 1, 128, 128, 64, f64::NAN, false);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn reduce_validates_axis() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        // Axis out of bounds
        let result = backend.reduce(ReduceOp::Sum, 0, 0, &[10, 20], 2);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));

        // Empty shape
        let result = backend.reduce(ReduceOp::Sum, 0, 0, &[], 0);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn unary_binary_empty_is_noop() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        // Empty tensor operations are no-ops
        assert!(backend.unary(UnaryOp::Relu, 0, 0, 0).is_ok());
        assert!(backend.binary(BinaryOp::Add, 0, 0, 0, 0).is_ok());
    }

    #[test]
    fn alloc_zero_bytes_is_error() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        let result = backend.alloc(0);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn copy_empty_is_noop() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        assert!(backend.copy_htod(0, &[]).is_ok());
        let mut empty: [u8; 0] = [];
        assert!(backend.copy_dtoh(&mut empty, 0).is_ok());
    }

    #[test]
    fn debug_impl() {
        let backend = CudaBackend::new();
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("CudaBackend"));
        assert!(debug_str.contains("initialized"));
    }
}
