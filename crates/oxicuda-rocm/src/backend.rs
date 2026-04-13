//! [`RocmBackend`] — the main entry point for the `oxicuda-rocm` crate.
//!
//! Implements the [`ComputeBackend`] trait from `oxicuda-backend` using
//! the AMD ROCm/HIP runtime for GPU compute on Linux.

use std::sync::Arc;

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};

use crate::{device::RocmDevice, memory::RocmMemoryManager};

// ─── Backend struct ───────────────────────────────────────────────────────────

/// AMD ROCm/HIP GPU compute backend.
///
/// On Linux this loads `libamdhip64.so` at runtime, selects the first AMD
/// GPU device, and manages explicit device-memory allocations.
///
/// On non-Linux platforms (macOS, Windows) every operation returns
/// [`BackendError::DeviceError`] wrapping [`crate::error::RocmError::UnsupportedPlatform`].
///
/// # Lifecycle
///
/// 1. `RocmBackend::new()` — create an uninitialised backend.
/// 2. `init()` — load HIP runtime, select device, set up memory manager.
/// 3. Use `alloc`, `copy_htod`, compute ops, `copy_dtoh`, `free`.
/// 4. `synchronize()` — wait for all pending GPU work to complete.
#[derive(Debug)]
pub struct RocmBackend {
    device: Option<Arc<RocmDevice>>,
    memory: Option<Arc<RocmMemoryManager>>,
    initialized: bool,
}

impl RocmBackend {
    /// Create a new, uninitialised ROCm backend.
    pub fn new() -> Self {
        Self {
            device: None,
            memory: None,
            initialized: false,
        }
    }

    /// Return an error if the backend has not been initialised yet.
    fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }

    /// Convenience accessor: return the memory manager or `NotInitialized`.
    fn memory_manager(&self) -> BackendResult<&Arc<RocmMemoryManager>> {
        self.memory.as_ref().ok_or(BackendError::NotInitialized)
    }
}

impl Default for RocmBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ComputeBackend impl ──────────────────────────────────────────────────────

impl ComputeBackend for RocmBackend {
    fn name(&self) -> &str {
        "rocm"
    }

    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }
        match RocmDevice::new() {
            Ok(dev) => {
                let dev = Arc::new(dev);
                tracing::info!("ROCm backend initialised on: {}", dev.name());
                let memory = RocmMemoryManager::new(Arc::clone(&dev));
                self.device = Some(dev);
                self.memory = Some(Arc::new(memory));
                self.initialized = true;
                Ok(())
            }
            Err(e) => Err(BackendError::from(e)),
        }
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    // ── Compute operations ────────────────────────────────────────────────────

    fn gemm(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
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
        // Zero-dimension matrices are trivially complete.
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        Err(BackendError::Unsupported("rocm: gemm not yet wired".into()))
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

        if input_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "input_shape must have 4 elements (NCHW)".into(),
            ));
        }
        if filter_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "filter_shape must have 4 elements (KCFHFW)".into(),
            ));
        }
        if output_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "output_shape must have 4 elements (NKOhOw)".into(),
            ));
        }
        if stride.len() != 2 {
            return Err(BackendError::InvalidArgument(
                "stride must have 2 elements [sh, sw]".into(),
            ));
        }
        if padding.len() != 2 {
            return Err(BackendError::InvalidArgument(
                "padding must have 2 elements [ph, pw]".into(),
            ));
        }

        Err(BackendError::Unsupported(
            "rocm: conv2d_forward not yet wired".into(),
        ))
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
                "seq_q, seq_kv, and head_dim must all be > 0".into(),
            ));
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(BackendError::InvalidArgument(format!(
                "scale must be a positive finite number, got {scale}"
            )));
        }

        Err(BackendError::Unsupported(
            "rocm: attention not yet wired".into(),
        ))
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
                "axis {axis} is out of bounds for shape of length {}",
                shape.len()
            )));
        }

        Err(BackendError::Unsupported(
            "rocm: reduce not yet wired".into(),
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
            return Ok(());
        }
        Err(BackendError::Unsupported(
            "rocm: unary not yet wired".into(),
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
            return Ok(());
        }
        Err(BackendError::Unsupported(
            "rocm: binary not yet wired".into(),
        ))
    }

    // ── Synchronisation ───────────────────────────────────────────────────────

    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;
        #[cfg(target_os = "linux")]
        {
            if let Some(dev) = &self.device {
                // SAFETY: `hip_device_synchronize` is a valid fn ptr from the
                // loaded HIP library.  It blocks until all pending GPU work on
                // the current device completes.
                let rc = unsafe { (dev.api.hip_device_synchronize)() };
                if rc != crate::device::HIP_SUCCESS {
                    return Err(BackendError::DeviceError(format!(
                        "hipDeviceSynchronize failed (rc={rc})"
                    )));
                }
            }
        }
        Ok(())
    }

    // ── Memory management ─────────────────────────────────────────────────────

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.check_init()?;
        if bytes == 0 {
            return Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into(),
            ));
        }
        self.memory_manager()?
            .alloc(bytes)
            .map_err(BackendError::from)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.check_init()?;
        self.memory_manager()?.free(ptr).map_err(BackendError::from)
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.check_init()?;
        if src.is_empty() {
            return Ok(());
        }
        self.memory_manager()?
            .copy_to_device(dst, src)
            .map_err(BackendError::from)
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.check_init()?;
        if dst.is_empty() {
            return Ok(());
        }
        self.memory_manager()?
            .copy_from_device(dst, src)
            .map_err(BackendError::from)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::{BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp};

    // ── 1. Structural ─────────────────────────────────────────────────────────

    #[test]
    fn rocm_backend_new_uninitialized() {
        let b = RocmBackend::new();
        assert!(!b.is_initialized());
    }

    #[test]
    fn rocm_backend_name() {
        let b = RocmBackend::new();
        assert_eq!(b.name(), "rocm");
    }

    #[test]
    fn rocm_backend_default() {
        let b = RocmBackend::default();
        assert!(!b.is_initialized());
        assert_eq!(b.name(), "rocm");
    }

    #[test]
    fn backend_debug_impl() {
        let b = RocmBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("RocmBackend"));
    }

    #[test]
    fn backend_object_safe() {
        let b: Box<dyn ComputeBackend> = Box::new(RocmBackend::new());
        assert_eq!(b.name(), "rocm");
    }

    // ── 2. Not-initialized guards ─────────────────────────────────────────────

    #[test]
    fn backend_not_initialized_gemm() {
        let b = RocmBackend::new();
        let result = b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            0,
            4,
            0.0,
            0,
            4,
        );
        assert_eq!(result, Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_alloc() {
        let b = RocmBackend::new();
        assert_eq!(b.alloc(1024), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_synchronize() {
        let b = RocmBackend::new();
        assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_free() {
        let b = RocmBackend::new();
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_htod() {
        let b = RocmBackend::new();
        assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_dtoh() {
        let b = RocmBackend::new();
        let mut buf = [0u8; 4];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
    }

    // ── 3. Graceful init failure ──────────────────────────────────────────────

    #[test]
    fn init_graceful_failure() {
        // Verify that init() returns a Result and never panics on any platform.
        let mut b = RocmBackend::new();
        let _result = b.init();
        // Both Ok(()) and Err(_) are acceptable.
    }

    // ── 4. Post-init tests (skip if no AMD GPU / not on Linux) ────────────────

    fn try_init() -> Option<RocmBackend> {
        let mut b = RocmBackend::new();
        match b.init() {
            Ok(()) => Some(b),
            Err(_) => None,
        }
    }

    #[test]
    fn alloc_zero_bytes_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.alloc(0),
            Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into()
            ))
        );
    }

    #[test]
    fn copy_htod_empty_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.copy_htod(0, &[]), Ok(()));
    }

    #[test]
    fn copy_dtoh_empty_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.copy_dtoh(&mut [], 0), Ok(()));
    }

    #[test]
    fn gemm_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                0,
                0,
                1.0,
                0,
                1,
                0,
                1,
                0.0,
                0,
                1
            ),
            Ok(())
        );
    }

    #[test]
    fn unary_zero_n_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.unary(UnaryOp::Relu, 0, 0, 0), Ok(()));
    }

    #[test]
    fn binary_zero_n_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.binary(BinaryOp::Add, 0, 0, 0, 0), Ok(()));
    }

    #[test]
    fn synchronize_after_init() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.synchronize(), Ok(()));
    }

    #[test]
    fn reduce_empty_shape_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.reduce(ReduceOp::Sum, 0, 0, &[], 0),
            Err(BackendError::InvalidArgument(
                "shape must not be empty".into()
            ))
        );
    }

    #[test]
    fn reduce_axis_out_of_bounds_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.reduce(ReduceOp::Sum, 0, 0, &[4, 4], 5),
            Err(BackendError::InvalidArgument(
                "axis 5 is out of bounds for shape of length 2".into()
            ))
        );
    }

    #[test]
    fn attention_zero_seq_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 0, 8, 64, 0.125, false),
            Err(BackendError::InvalidArgument(
                "seq_q, seq_kv, and head_dim must all be > 0".into()
            ))
        );
    }

    #[test]
    fn attention_invalid_scale_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, 0.0, false),
            Err(BackendError::InvalidArgument(
                "scale must be a positive finite number, got 0".into()
            ))
        );
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, -1.0, false),
            Err(BackendError::InvalidArgument(
                "scale must be a positive finite number, got -1".into()
            ))
        );
        assert!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, f64::INFINITY, false)
                .is_err()
        );
    }

    #[test]
    fn conv2d_wrong_input_rank() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.conv2d_forward(
                0,
                &[1, 3, 32],
                0,
                &[16, 3, 3, 3],
                0,
                &[1, 16, 30, 30],
                &[1, 1],
                &[0, 0]
            ),
            Err(BackendError::InvalidArgument(
                "input_shape must have 4 elements (NCHW)".into()
            ))
        );
    }

    #[test]
    fn conv2d_wrong_filter_rank() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.conv2d_forward(
                0,
                &[1, 3, 32, 32],
                0,
                &[16, 3, 3],
                0,
                &[1, 16, 30, 30],
                &[1, 1],
                &[0, 0]
            ),
            Err(BackendError::InvalidArgument(
                "filter_shape must have 4 elements (KCFHFW)".into()
            ))
        );
    }

    #[test]
    fn init_idempotent() {
        let Some(mut b) = try_init() else {
            return;
        };
        assert_eq!(b.init(), Ok(()));
        assert!(b.is_initialized());
    }

    #[test]
    fn alloc_copy_roundtrip() {
        let Some(b) = try_init() else {
            return;
        };
        let src: Vec<u8> = (0u8..64).collect();
        let handle = match b.alloc(src.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(handle, &src).expect("copy_htod");
        let mut dst = vec![0u8; src.len()];
        b.copy_dtoh(&mut dst, handle).expect("copy_dtoh");
        assert_eq!(src, dst);
        b.free(handle).expect("free");
    }

    #[test]
    fn double_init_is_noop() {
        let Some(mut b) = try_init() else {
            return;
        };
        assert!(b.is_initialized());
        assert_eq!(b.init(), Ok(()));
        assert!(b.is_initialized());
    }

    #[test]
    fn alloc_and_free_basic() {
        let Some(b) = try_init() else {
            return;
        };
        let handle = match b.alloc(512) {
            Ok(h) => h,
            Err(_) => return,
        };
        assert!(handle > 0);
        b.free(handle).expect("free should succeed");
    }
}
