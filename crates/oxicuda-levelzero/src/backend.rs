//! [`LevelZeroBackend`] — the main entry point for the oxicuda-levelzero crate.
//!
//! Implements the [`ComputeBackend`] trait from `oxicuda-backend` using
//! Intel's Level Zero API for GPU compute on Linux and Windows.

use std::sync::Arc;

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};

use crate::{device::LevelZeroDevice, memory::LevelZeroMemoryManager};

// ─── Backend struct ───────────────────────────────────────────────────────────

/// Intel Level Zero GPU compute backend.
///
/// On Linux and Windows this selects the first Intel GPU via the Level Zero
/// loader library (`libze_loader.so` / `ze_loader.dll`) and allocates device
/// memory through the Level Zero memory model.
///
/// On macOS every operation returns [`BackendError::DeviceError`] wrapping
/// [`crate::error::LevelZeroError::UnsupportedPlatform`].
///
/// # Lifecycle
///
/// 1. `LevelZeroBackend::new()` — create an uninitialised backend.
/// 2. `init()` — load the Level Zero driver and select a GPU.
/// 3. Use `alloc`, `copy_htod`, compute ops, `copy_dtoh`, `free`.
/// 4. `synchronize()` — wait for all pending GPU work to finish.
#[derive(Debug)]
pub struct LevelZeroBackend {
    device: Option<Arc<LevelZeroDevice>>,
    memory: Option<Arc<LevelZeroMemoryManager>>,
    initialized: bool,
}

impl LevelZeroBackend {
    /// Create a new, uninitialised Level Zero backend.
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

    /// Convenience accessor: get the memory manager or return `NotInitialized`.
    fn memory(&self) -> BackendResult<&Arc<LevelZeroMemoryManager>> {
        self.memory.as_ref().ok_or(BackendError::NotInitialized)
    }
}

impl Default for LevelZeroBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ComputeBackend impl ──────────────────────────────────────────────────────

impl ComputeBackend for LevelZeroBackend {
    fn name(&self) -> &str {
        "level-zero"
    }

    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }
        match LevelZeroDevice::new() {
            Ok(dev) => {
                let dev = Arc::new(dev);
                tracing::info!("Level Zero backend initialised on: {}", dev.name());
                let memory = LevelZeroMemoryManager::new(Arc::clone(&dev));
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
        Err(BackendError::Unsupported(
            "level-zero: gemm not yet wired".into(),
        ))
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
            "level-zero: conv2d not yet wired".into(),
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
            "level-zero: attention not yet wired".into(),
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
            "level-zero: reduce not yet wired".into(),
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
            "level-zero: unary not yet wired".into(),
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
            "level-zero: binary not yet wired".into(),
        ))
    }

    // ── Synchronisation ───────────────────────────────────────────────────────

    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            if let Some(dev) = &self.device {
                let api = &dev.api;
                let queue = dev.queue;
                // SAFETY: `queue` is a valid command queue handle and the
                // backend is initialized.  u64::MAX means "wait indefinitely".
                let rc = unsafe { (api.ze_command_queue_synchronize)(queue, u64::MAX) };
                if rc != 0 {
                    return Err(BackendError::DeviceError(format!(
                        "zeCommandQueueSynchronize failed: 0x{rc:08x}"
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
        self.memory()?.alloc(bytes).map_err(BackendError::from)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.check_init()?;
        self.memory()?.free(ptr).map_err(BackendError::from)
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.check_init()?;
        if src.is_empty() {
            return Ok(());
        }
        self.memory()?
            .copy_to_device(dst, src)
            .map_err(BackendError::from)
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.check_init()?;
        if dst.is_empty() {
            return Ok(());
        }
        self.memory()?
            .copy_from_device(dst, src)
            .map_err(BackendError::from)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::{BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp};

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn level_zero_backend_new_uninitialized() {
        let b = LevelZeroBackend::new();
        assert!(!b.is_initialized());
    }

    #[test]
    fn level_zero_backend_name() {
        let b = LevelZeroBackend::new();
        assert_eq!(b.name(), "level-zero");
    }

    #[test]
    fn level_zero_backend_default() {
        let b = LevelZeroBackend::default();
        assert!(!b.is_initialized());
        assert_eq!(b.name(), "level-zero");
    }

    #[test]
    fn backend_debug_impl() {
        let b = LevelZeroBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("LevelZeroBackend"));
    }

    // ── Object-safety smoke test ──────────────────────────────────────────────

    #[test]
    fn backend_object_safe() {
        let b: Box<dyn ComputeBackend> = Box::new(LevelZeroBackend::new());
        assert_eq!(b.name(), "level-zero");
    }

    // ── Not-initialized guards ────────────────────────────────────────────────

    #[test]
    fn backend_not_initialized_gemm() {
        let b = LevelZeroBackend::new();
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
        let b = LevelZeroBackend::new();
        assert_eq!(b.alloc(1024), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_synchronize() {
        let b = LevelZeroBackend::new();
        assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_free() {
        let b = LevelZeroBackend::new();
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_htod() {
        let b = LevelZeroBackend::new();
        assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_dtoh() {
        let b = LevelZeroBackend::new();
        let mut buf = [0u8; 4];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
    }

    // ── Helper: try to get an initialised backend (skip if no GPU or no loader) ─

    fn try_init() -> Option<LevelZeroBackend> {
        let mut b = LevelZeroBackend::new();
        match b.init() {
            Ok(()) => Some(b),
            Err(_) => None,
        }
    }

    // ── Graceful init failure ─────────────────────────────────────────────────

    #[test]
    fn init_graceful_failure() {
        // Verify that init() returns a Result and never panics.
        let mut b = LevelZeroBackend::new();
        let _result = b.init();
        // Ok or Err — both are acceptable.
    }

    // ── Zero-size / trivial-OK paths (post-init) ──────────────────────────────

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

    // ── Argument validation (post-init) ───────────────────────────────────────

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

    // ── Init is idempotent ────────────────────────────────────────────────────

    #[test]
    fn init_idempotent() {
        let Some(mut b) = try_init() else {
            return;
        };
        assert_eq!(b.init(), Ok(()));
        assert!(b.is_initialized());
    }

    // ── alloc/free/copy roundtrip ─────────────────────────────────────────────

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

    // ── Double init is a no-op ────────────────────────────────────────────────

    #[test]
    fn double_init_is_noop() {
        let Some(mut b) = try_init() else {
            return;
        };
        let first = b.is_initialized();
        let _ = b.init();
        assert_eq!(first, b.is_initialized());
    }

    // ── alloc and free basic ──────────────────────────────────────────────────

    #[test]
    fn alloc_and_free_basic() {
        let Some(b) = try_init() else {
            return;
        };
        match b.alloc(128) {
            Ok(handle) => {
                assert!(handle > 0);
                b.free(handle).expect("free should succeed");
            }
            Err(_) => {
                // Allocation failure is acceptable in environments without GPU.
            }
        }
    }
}
