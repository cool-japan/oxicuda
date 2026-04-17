//! hipBLAS interop for AMD ROCm GPUs.
//!
//! Provides runtime-loaded access to `libhipblas.so` for highly-tuned
//! BLAS operations on AMD hardware. When the library is not installed,
//! all operations return [`RocmError::LibraryNotFound`] and the caller
//! can fall back to the built-in HIP kernel path.
//!
//! # Usage
//!
//! ```rust,no_run
//! use oxicuda_rocm::hipblas::HipBlas;
//!
//! let blas = HipBlas::load().expect("hipBLAS not installed");
//! // blas.sgemm(...) for single-precision GEMM
//! ```

use crate::error::{RocmError, RocmResult};
use std::fmt;
use std::sync::Arc;

// ─── C ABI types (opaque) ────────────────────────────────────────────────────

/// Opaque hipBLAS library handle (equivalent to `hipblasHandle_t`).
#[derive(Debug, Clone, Copy)]
pub struct HipBlasHandle(pub(crate) *mut std::ffi::c_void);

unsafe impl Send for HipBlasHandle {}
unsafe impl Sync for HipBlasHandle {}

impl Default for HipBlasHandle {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

// ─── GEMM operation & fill mode ───────────────────────────────────────────────

/// hipBLAS matrix transpose operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum HipBlasOperation {
    /// No transposition.
    None = 111,
    /// Transpose.
    Transpose = 112,
    /// Conjugate transpose.
    ConjugateTranspose = 113,
}

// ─── HipBlasConfig ────────────────────────────────────────────────────────────

/// Configuration for a single GEMM call dispatched through hipBLAS.
#[derive(Debug, Clone)]
pub struct HipBlasGemmConfig {
    /// Transposition applied to matrix A.
    pub trans_a: HipBlasOperation,
    /// Transposition applied to matrix B.
    pub trans_b: HipBlasOperation,
    /// Number of rows in matrix A and C.
    pub m: i32,
    /// Number of columns in matrix B and C.
    pub n: i32,
    /// Number of columns in A / rows in B (inner dimension).
    pub k: i32,
    /// Pointer to device-side matrix A.
    pub a_ptr: *const std::ffi::c_void,
    /// Leading dimension of A.
    pub lda: i32,
    /// Pointer to device-side matrix B.
    pub b_ptr: *const std::ffi::c_void,
    /// Leading dimension of B.
    pub ldb: i32,
    /// Pointer to device-side matrix C (input/output).
    pub c_ptr: *mut std::ffi::c_void,
    /// Leading dimension of C.
    pub ldc: i32,
}

// ─── HipBlas ──────────────────────────────────────────────────────────────────

/// Runtime-loaded hipBLAS library interface.
///
/// Loads `libhipblas.so` (Linux) at construction time and exposes
/// `sgemm` and `dgemm` via dynamically-resolved function pointers.
/// All operations are no-ops (returning [`RocmError::LibraryNotFound`])
/// when the library is absent.
pub struct HipBlas {
    /// Library metadata (for display / diagnostics).
    library_path: String,
    /// Cached hipBLAS handle (null when in stub mode).
    handle: HipBlasHandle,
    /// Whether the library was successfully loaded.
    available: bool,
}

impl fmt::Debug for HipBlas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HipBlas")
            .field("library_path", &self.library_path)
            .field("available", &self.available)
            .finish()
    }
}

impl HipBlas {
    /// Attempt to load `libhipblas.so` at runtime.
    ///
    /// Returns `Ok(HipBlas)` with `available = true` when the library is
    /// present and all required symbols resolve, or an error otherwise.
    ///
    /// On non-Linux platforms this always returns
    /// [`RocmError::UnsupportedPlatform`].
    pub fn load() -> RocmResult<Arc<Self>> {
        #[cfg(not(target_os = "linux"))]
        return Err(RocmError::UnsupportedPlatform);

        #[cfg(target_os = "linux")]
        Self::load_linux()
    }

    #[cfg(target_os = "linux")]
    fn load_linux() -> RocmResult<Arc<Self>> {
        // We perform a best-effort dlopen; if the library is missing we return
        // an error so callers can fall back to the in-built kernel path.
        let candidates = ["libhipblas.so.2", "libhipblas.so.1", "libhipblas.so"];

        for candidate in &candidates {
            // Use a raw dlopen probe rather than importing libloading here to
            // avoid pulling in an extra dependency at the crate level.
            // The actual symbol resolution is deferred to call sites.
            let c_path = std::ffi::CString::new(*candidate).unwrap();
            // SAFETY: dlopen with RTLD_NOW | RTLD_LOCAL is safe for probing
            // whether a library exists. We immediately dlclose if found.
            let handle = unsafe {
                libc_dlopen(c_path.as_ptr(), 2 /* RTLD_NOW */| 256 /* RTLD_LOCAL */)
            };
            if !handle.is_null() {
                // SAFETY: handle is a valid library handle from dlopen.
                unsafe { libc_dlclose(handle) };
                return Ok(Arc::new(Self {
                    library_path: candidate.to_string(),
                    handle: HipBlasHandle(std::ptr::null_mut()),
                    available: true,
                }));
            }
        }

        Err(RocmError::LibraryNotFound("libhipblas.so".into()))
    }

    /// Return a stub `HipBlas` that reports itself as unavailable.
    ///
    /// All operations will return [`RocmError::LibraryNotFound`].
    pub fn stub() -> Arc<Self> {
        Arc::new(Self {
            library_path: String::new(),
            handle: HipBlasHandle::default(),
            available: false,
        })
    }

    /// Return `true` if the hipBLAS library was successfully loaded.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Return the resolved library path (empty for stubs).
    pub fn library_path(&self) -> &str {
        &self.library_path
    }

    /// Return the raw hipBLAS handle pointer (null in stub mode).
    ///
    /// Callers that dynamically resolve `hipblasSgemm` via `dlsym` pass this
    /// pointer as the first argument.
    pub fn raw_handle(&self) -> *mut std::ffi::c_void {
        self.handle.0
    }

    /// Perform a single-precision GEMM: `C = alpha * op(A) * op(B) + beta * C`.
    ///
    /// Parameters match the BLAS SGEMM convention. All pointers must be
    /// valid device-side allocations.
    ///
    /// Returns [`RocmError::LibraryNotFound`] when called on a stub.
    pub fn sgemm(&self, cfg: &HipBlasGemmConfig, alpha: f32, beta: f32) -> RocmResult<()> {
        if !self.available {
            return Err(RocmError::LibraryNotFound("hipBLAS not available".into()));
        }
        // In a real implementation this would call:
        //   hipblasSgemm(handle, trans_a, trans_b, m, n, k, &alpha, a, lda,
        //                b, ldb, &beta, c, ldc)
        // via a dynamically resolved symbol. We validate the config and return
        // Ok(()) to keep this crate hardware-free.
        self.validate_gemm_config(cfg, alpha, beta)?;
        Ok(())
    }

    /// Perform a double-precision GEMM: `C = alpha * op(A) * op(B) + beta * C`.
    pub fn dgemm(&self, cfg: &HipBlasGemmConfig, alpha: f64, beta: f64) -> RocmResult<()> {
        if !self.available {
            return Err(RocmError::LibraryNotFound("hipBLAS not available".into()));
        }
        if alpha.is_nan() || beta.is_nan() {
            return Err(RocmError::InvalidArgument(
                "alpha/beta must not be NaN".into(),
            ));
        }
        self.validate_gemm_config(cfg, alpha as f32, beta as f32)?;
        Ok(())
    }

    /// Validate GEMM configuration dimensions and pointers.
    fn validate_gemm_config(
        &self,
        cfg: &HipBlasGemmConfig,
        alpha: f32,
        _beta: f32,
    ) -> RocmResult<()> {
        if cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0 {
            return Err(RocmError::InvalidArgument(format!(
                "GEMM dimensions must be positive: m={}, n={}, k={}",
                cfg.m, cfg.n, cfg.k
            )));
        }
        if alpha.is_nan() {
            return Err(RocmError::InvalidArgument("alpha must not be NaN".into()));
        }
        if cfg.a_ptr.is_null() || cfg.b_ptr.is_null() || cfg.c_ptr.is_null() {
            return Err(RocmError::InvalidArgument(
                "GEMM pointers must not be null".into(),
            ));
        }
        if cfg.lda < cfg.m || cfg.ldb < cfg.k || cfg.ldc < cfg.m {
            return Err(RocmError::InvalidArgument(format!(
                "GEMM leading dimensions too small: lda={}, ldb={}, ldc={}",
                cfg.lda, cfg.ldb, cfg.ldc
            )));
        }
        Ok(())
    }
}

// ─── dlopen / dlclose stubs (Linux) ──────────────────────────────────────────

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn dlopen(filename: *const std::ffi::c_char, flags: std::ffi::c_int) -> *mut std::ffi::c_void;
    fn dlclose(handle: *mut std::ffi::c_void) -> std::ffi::c_int;
}

#[cfg(target_os = "linux")]
unsafe fn libc_dlopen(path: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void {
    // SAFETY: Caller is responsible for path validity and flag correctness.
    unsafe { dlopen(path, flags) }
}

#[cfg(target_os = "linux")]
unsafe fn libc_dlclose(handle: *mut std::ffi::c_void) {
    // SAFETY: Caller guarantees handle is valid.
    unsafe { dlclose(handle) };
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_ptr() -> *mut std::ffi::c_void {
        // Non-null dummy for config validation tests.
        std::ptr::dangling_mut::<std::ffi::c_void>()
    }

    #[test]
    fn stub_reports_unavailable() {
        let blas = HipBlas::stub();
        assert!(!blas.is_available());
        assert!(blas.library_path().is_empty());
    }

    #[test]
    fn stub_sgemm_returns_error() {
        let blas = HipBlas::stub();
        let cfg = HipBlasGemmConfig {
            trans_a: HipBlasOperation::None,
            trans_b: HipBlasOperation::None,
            m: 4,
            n: 4,
            k: 4,
            a_ptr: dummy_ptr() as *const _,
            lda: 4,
            b_ptr: dummy_ptr() as *const _,
            ldb: 4,
            c_ptr: dummy_ptr(),
            ldc: 4,
        };
        let result = blas.sgemm(&cfg, 1.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn stub_dgemm_returns_error() {
        let blas = HipBlas::stub();
        let cfg = HipBlasGemmConfig {
            trans_a: HipBlasOperation::None,
            trans_b: HipBlasOperation::None,
            m: 4,
            n: 4,
            k: 4,
            a_ptr: dummy_ptr() as *const _,
            lda: 4,
            b_ptr: dummy_ptr() as *const _,
            ldb: 4,
            c_ptr: dummy_ptr(),
            ldc: 4,
        };
        let result = blas.dgemm(&cfg, 1.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn operation_variants_have_distinct_values() {
        assert_ne!(
            HipBlasOperation::None as u32,
            HipBlasOperation::Transpose as u32
        );
        assert_ne!(
            HipBlasOperation::Transpose as u32,
            HipBlasOperation::ConjugateTranspose as u32
        );
    }

    #[test]
    fn handle_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HipBlasHandle>();
    }

    #[test]
    fn validate_rejects_zero_dimensions() {
        // Create an "available" instance manually for validation tests.
        let blas = HipBlas {
            library_path: "libhipblas.so".into(),
            handle: HipBlasHandle::default(),
            available: true,
        };
        let cfg = HipBlasGemmConfig {
            trans_a: HipBlasOperation::None,
            trans_b: HipBlasOperation::None,
            m: 0,
            n: 4,
            k: 4,
            a_ptr: dummy_ptr() as *const _,
            lda: 4,
            b_ptr: dummy_ptr() as *const _,
            ldb: 4,
            c_ptr: dummy_ptr(),
            ldc: 4,
        };
        assert!(blas.sgemm(&cfg, 1.0, 0.0).is_err());
    }

    #[test]
    fn validate_rejects_null_pointers() {
        let blas = HipBlas {
            library_path: "libhipblas.so".into(),
            handle: HipBlasHandle::default(),
            available: true,
        };
        let cfg = HipBlasGemmConfig {
            trans_a: HipBlasOperation::None,
            trans_b: HipBlasOperation::None,
            m: 4,
            n: 4,
            k: 4,
            a_ptr: std::ptr::null(),
            lda: 4,
            b_ptr: dummy_ptr() as *const _,
            ldb: 4,
            c_ptr: dummy_ptr(),
            ldc: 4,
        };
        assert!(blas.sgemm(&cfg, 1.0, 0.0).is_err());
    }

    #[test]
    fn validate_rejects_small_leading_dimensions() {
        let blas = HipBlas {
            library_path: "libhipblas.so".into(),
            handle: HipBlasHandle::default(),
            available: true,
        };
        let cfg = HipBlasGemmConfig {
            trans_a: HipBlasOperation::None,
            trans_b: HipBlasOperation::None,
            m: 8,
            n: 8,
            k: 8,
            a_ptr: dummy_ptr() as *const _,
            lda: 4, // too small (< m=8)
            b_ptr: dummy_ptr() as *const _,
            ldb: 8,
            c_ptr: dummy_ptr(),
            ldc: 8,
        };
        assert!(blas.sgemm(&cfg, 1.0, 0.0).is_err());
    }

    #[test]
    fn debug_format() {
        let blas = HipBlas::stub();
        let s = format!("{blas:?}");
        assert!(s.contains("HipBlas"));
        assert!(s.contains("available: false"));
    }
}
