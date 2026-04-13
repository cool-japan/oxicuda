//! [`VulkanBackend`] — implements [`ComputeBackend`] via Vulkan + SPIR-V.
//!
//! # Initialisation
//!
//! Call [`VulkanBackend::init`] before any other operation.  On macOS (or any
//! system without a Vulkan driver), `init` returns
//! `Err(BackendError::DeviceError(...))` rather than panicking.
//!
//! # Compute operations
//!
//! All compute kernels (`gemm`, `conv2d_forward`, `attention`, `reduce`,
//! `unary`, `binary`) currently return `BackendError::Unsupported`.  The
//! memory management pipeline (alloc / free / copy_htod / copy_dtoh) and
//! synchronisation are fully implemented via Vulkan buffer objects backed by
//! host-visible, host-coherent memory.

use std::sync::Arc;

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};

use crate::device::VulkanDevice;
use crate::memory::VulkanMemoryManager;

/// Vulkan compute backend.
///
/// Create with `VulkanBackend::new()`, then call `init()` to select a device.
#[derive(Debug)]
pub struct VulkanBackend {
    device: Option<Arc<VulkanDevice>>,
    memory: Option<Arc<VulkanMemoryManager>>,
    initialized: bool,
}

impl VulkanBackend {
    /// Create an uninitialised backend.  Call [`init`](Self::init) before use.
    pub fn new() -> Self {
        Self {
            device: None,
            memory: None,
            initialized: false,
        }
    }

    /// Return `Err(NotInitialized)` if `init` has not been called successfully.
    fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }

    /// Convenience: get the memory manager or return NotInitialized.
    fn memory_manager(&self) -> BackendResult<&VulkanMemoryManager> {
        self.check_init()?;
        self.memory.as_deref().ok_or(BackendError::NotInitialized)
    }
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }

        let device = VulkanDevice::new().map_err(BackendError::from)?;

        tracing::info!(device = device.device_name(), "Vulkan backend initialised");

        let device = Arc::new(device);
        let memory = Arc::new(VulkanMemoryManager::new(Arc::clone(&device)));

        self.device = Some(device);
        self.memory = Some(memory);
        self.initialized = true;

        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    // ── Compute operations ───────────────────────────────────────────────────

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
        // Zero-dimension GEMM is a no-op.
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        Err(BackendError::Unsupported(
            "vulkan: gemm not yet wired to SPIR-V kernel".into(),
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
        _stride: &[usize],
        _padding: &[usize],
    ) -> BackendResult<()> {
        self.check_init()?;

        // Validate shapes: all must have rank 4.
        if input_shape.len() != 4 || filter_shape.len() != 4 || output_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "conv2d_forward: input, filter, and output must each have rank 4 (NCHW)".into(),
            ));
        }

        Err(BackendError::Unsupported(
            "vulkan: conv2d_forward not yet wired to SPIR-V kernel".into(),
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
        _head_dim: usize,
        scale: f64,
        _causal: bool,
    ) -> BackendResult<()> {
        self.check_init()?;

        if seq_q == 0 || seq_kv == 0 {
            return Err(BackendError::InvalidArgument(
                "attention: seq_q and seq_kv must be > 0".into(),
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(BackendError::InvalidArgument(
                "attention: scale must be a positive finite number".into(),
            ));
        }

        Err(BackendError::Unsupported(
            "vulkan: attention not yet wired to SPIR-V kernel".into(),
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
                "reduce: shape must not be empty".into(),
            ));
        }
        if axis >= shape.len() {
            return Err(BackendError::InvalidArgument(format!(
                "reduce: axis {axis} out of bounds for shape of rank {}",
                shape.len()
            )));
        }

        Err(BackendError::Unsupported(
            "vulkan: reduce not yet wired to SPIR-V kernel".into(),
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
            "vulkan: unary not yet wired to SPIR-V kernel".into(),
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
            "vulkan: binary not yet wired to SPIR-V kernel".into(),
        ))
    }

    // ── Synchronisation ──────────────────────────────────────────────────────

    fn synchronize(&self) -> BackendResult<()> {
        if !self.initialized {
            // No device — nothing to synchronise.
            return Ok(());
        }
        if let Some(dev) = &self.device {
            dev.wait_idle().map_err(BackendError::from)?;
        }
        Ok(())
    }

    // ── Memory management ────────────────────────────────────────────────────

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.memory_manager()?
            .alloc(bytes)
            .map_err(BackendError::from)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.memory_manager()?.free(ptr).map_err(BackendError::from)
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        if src.is_empty() {
            return Ok(());
        }
        self.memory_manager()?
            .copy_to_device(dst, src)
            .map_err(BackendError::from)
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        if dst.is_empty() {
            return Ok(());
        }
        self.memory_manager()?
            .copy_from_device(dst, src)
            .map_err(BackendError::from)
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::ComputeBackend;

    // ── Structural / compile-time tests ─────────────────────

    #[test]
    fn vulkan_backend_new_uninitialized() {
        let b = VulkanBackend::new();
        assert!(!b.is_initialized());
    }

    #[test]
    fn vulkan_backend_name() {
        let b = VulkanBackend::new();
        assert_eq!(b.name(), "vulkan");
    }

    #[test]
    fn vulkan_backend_default() {
        let b = VulkanBackend::default();
        assert!(!b.is_initialized());
        assert_eq!(b.name(), "vulkan");
    }

    #[test]
    fn backend_debug_impl() {
        let b = VulkanBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("VulkanBackend"));
    }

    /// Verify that `VulkanBackend` can be used as a `Box<dyn ComputeBackend>`.
    #[test]
    fn backend_object_safe() {
        let b: Box<dyn ComputeBackend> = Box::new(VulkanBackend::new());
        assert_eq!(b.name(), "vulkan");
        assert!(!b.is_initialized());
    }

    // ── Not-initialised guard tests ──────────────────────────

    #[test]
    fn backend_not_initialized_returns_error() {
        let b = VulkanBackend::new();
        assert_eq!(b.alloc(64), Err(BackendError::NotInitialized));
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
        assert_eq!(b.copy_htod(1, &[0u8; 8]), Err(BackendError::NotInitialized));
        let mut buf = [0u8; 8];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
    }

    #[test]
    fn alloc_zero_bytes_error() {
        // alloc(0) — not initialised path first.
        let b = VulkanBackend::new();
        assert!(b.alloc(0).is_err());
    }

    // ── Empty / zero-dimension no-ops (no device needed) ────

    /// copy_htod with empty src is a no-op even before init.
    #[test]
    fn copy_htod_empty_noop() {
        let b = VulkanBackend::new();
        // Empty slice -> early return Ok(()) regardless of init state.
        assert_eq!(b.copy_htod(0, &[]), Ok(()));
    }

    /// copy_dtoh with empty dst is a no-op even before init.
    #[test]
    fn copy_dtoh_empty_noop() {
        let b = VulkanBackend::new();
        let buf: &mut [u8] = &mut [];
        assert_eq!(b.copy_dtoh(buf, 0), Ok(()));
    }

    /// GEMM with any zero dimension is a no-op (requires init guard to pass first).
    /// We test the invalid-arg path (not-initialized).
    #[test]
    fn gemm_uninit_returns_not_initialized() {
        let b = VulkanBackend::new();
        let r = b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            0,
            0,
            0,
            1.0,
            0,
            0,
            0,
            0,
            0.0,
            0,
            0,
        );
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    #[test]
    fn unary_zero_n_uninit_returns_not_initialized() {
        let b = VulkanBackend::new();
        assert_eq!(
            b.unary(UnaryOp::Relu, 0, 0, 0),
            Err(BackendError::NotInitialized)
        );
    }

    #[test]
    fn binary_zero_n_uninit_returns_not_initialized() {
        let b = VulkanBackend::new();
        assert_eq!(
            b.binary(BinaryOp::Add, 0, 0, 0, 0),
            Err(BackendError::NotInitialized)
        );
    }

    #[test]
    fn reduce_empty_shape_error() {
        let b = VulkanBackend::new();
        // Not initialised — returns NotInitialized first.
        let r = b.reduce(ReduceOp::Sum, 0, 0, &[], 0);
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    #[test]
    fn attention_zero_seq_error() {
        let b = VulkanBackend::new();
        let r = b.attention(0, 0, 0, 0, 1, 1, 0, 0, 64, 0.125, false);
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    #[test]
    fn attention_invalid_scale_error() {
        let b = VulkanBackend::new();
        let r = b.attention(0, 0, 0, 0, 1, 1, 4, 4, 64, -1.0, false);
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    #[test]
    fn conv2d_wrong_shape_ranks() {
        let b = VulkanBackend::new();
        // Not initialised first.
        let r = b.conv2d_forward(
            0,
            &[1, 1, 4],
            0,
            &[1, 1, 3, 3],
            0,
            &[1, 1, 2, 2],
            &[1, 1],
            &[0, 0],
        );
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    /// `synchronize()` on an uninitialised backend is a no-op.
    #[test]
    fn synchronize_uninit_noop() {
        let b = VulkanBackend::new();
        assert_eq!(b.synchronize(), Ok(()));
    }

    // ── init() graceful failure test ─────────────────────────

    /// On macOS (no Vulkan driver) init() returns Err and does not panic.
    /// On Linux with a Vulkan driver this will succeed — both cases are correct.
    #[test]
    fn init_graceful_failure() {
        let mut b = VulkanBackend::new();
        let result = b.init();
        match result {
            Ok(()) => {
                // Vulkan is available on this system — verify post-init state.
                assert!(b.is_initialized());
                // Double-init is a no-op.
                assert_eq!(b.init(), Ok(()));
            }
            Err(e) => {
                // Vulkan not available — backend must not be initialised.
                assert!(!b.is_initialized());
                // Must be a DeviceError wrapping the library-not-found message.
                assert!(
                    matches!(e, BackendError::DeviceError(_)),
                    "expected DeviceError, got {e:?}"
                );
            }
        }
    }

    // ── Post-init behaviour (only runs when Vulkan is available) ────────────

    #[test]
    fn alloc_free_round_trip_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            // No Vulkan — skip.
            return;
        }
        let handle = b.alloc(256).expect("alloc 256 bytes");
        b.free(handle).expect("free handle");
    }

    #[test]
    fn copy_round_trip_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        let data: Vec<u8> = (0u8..64).collect();
        let handle = b.alloc(64).expect("alloc");
        b.copy_htod(handle, &data).expect("copy_htod");
        let mut recv = vec![0u8; 64];
        b.copy_dtoh(&mut recv, handle).expect("copy_dtoh");
        assert_eq!(data, recv, "round-trip copy must preserve data");
        b.free(handle).expect("free");
    }

    #[test]
    fn alloc_zero_after_init_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        // alloc(0) must return an error even after init.
        assert!(matches!(b.alloc(0), Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn gemm_zero_dims_noop_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        // m == 0 -> no-op
        assert_eq!(
            b.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                4,
                4,
                1.0,
                0,
                4,
                0,
                4,
                0.0,
                0,
                4
            ),
            Ok(())
        );
        // n == 0 -> no-op
        assert_eq!(
            b.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                4,
                0,
                4,
                1.0,
                0,
                4,
                0,
                4,
                0.0,
                0,
                4
            ),
            Ok(())
        );
        // k == 0 -> no-op
        assert_eq!(
            b.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                4,
                4,
                0,
                1.0,
                0,
                4,
                0,
                4,
                0.0,
                0,
                4
            ),
            Ok(())
        );
    }

    #[test]
    fn unary_zero_n_noop_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert_eq!(b.unary(UnaryOp::Relu, 0, 0, 0), Ok(()));
    }

    #[test]
    fn binary_zero_n_noop_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert_eq!(b.binary(BinaryOp::Add, 0, 0, 0, 0), Ok(()));
    }

    #[test]
    fn reduce_empty_shape_error_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert!(matches!(
            b.reduce(ReduceOp::Sum, 0, 0, &[], 0),
            Err(BackendError::InvalidArgument(_))
        ));
    }

    #[test]
    fn attention_zero_seq_error_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert!(matches!(
            b.attention(0, 0, 0, 0, 1, 1, 0, 4, 64, 0.125, false),
            Err(BackendError::InvalidArgument(_))
        ));
    }

    #[test]
    fn attention_invalid_scale_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert!(matches!(
            b.attention(0, 0, 0, 0, 1, 1, 4, 4, 64, 0.0, false),
            Err(BackendError::InvalidArgument(_))
        ));
        assert!(matches!(
            b.attention(0, 0, 0, 0, 1, 1, 4, 4, 64, f64::INFINITY, false),
            Err(BackendError::InvalidArgument(_))
        ));
    }

    #[test]
    fn conv2d_wrong_shape_ranks_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        // input rank 3 is invalid.
        assert!(matches!(
            b.conv2d_forward(
                0,
                &[1, 3, 4],
                0,
                &[1, 3, 3, 3],
                0,
                &[1, 1, 2, 2],
                &[1, 1],
                &[0, 0]
            ),
            Err(BackendError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synchronize_after_init_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert_eq!(b.synchronize(), Ok(()));
    }
}
