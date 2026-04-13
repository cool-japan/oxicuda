//! CUDA context management with RAII semantics.
//!
//! A CUDA **context** is the primary interface through which a CPU thread
//! interacts with a GPU. It owns driver state such as loaded modules, allocated
//! memory, and streams. This module provides the [`Context`] type, an RAII
//! wrapper around `CUcontext` that automatically calls `cuCtxDestroy` on drop.
//!
//! # Thread safety
//!
//! CUDA contexts can be migrated between threads via `cuCtxSetCurrent`. The
//! [`Context`] type implements [`Send`] to allow this. It does **not** implement
//! [`Sync`] because the driver binds a context to a single thread at a time.
//! Use [`Arc<Context>`](std::sync::Arc) together with explicit
//! [`set_current`](Context::set_current) calls when sharing across threads.
//!
//! # Examples
//!
//! ```no_run
//! use oxicuda_driver::context::Context;
//! use oxicuda_driver::device::Device;
//!
//! oxicuda_driver::init()?;
//! let device = Device::get(0)?;
//! let ctx = Context::new(&device)?;
//! ctx.set_current()?;
//! // ... launch kernels, allocate memory ...
//! ctx.synchronize()?;
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use crate::device::Device;
use crate::error::CudaResult;
use crate::ffi::CUcontext;
use crate::loader::try_driver;

// ---------------------------------------------------------------------------
// Scheduling flags
// ---------------------------------------------------------------------------

/// Context scheduling flags passed to [`Context::with_flags`].
///
/// These control how the CPU thread behaves while waiting for GPU operations.
pub mod flags {
    /// Let the driver choose the optimal scheduling policy.
    pub const SCHED_AUTO: u32 = 0x00;

    /// Actively spin (busy-wait) while waiting for GPU results. Lowest latency
    /// but consumes a full CPU core.
    pub const SCHED_SPIN: u32 = 0x01;

    /// Yield the CPU time-slice to other threads while waiting. Good for
    /// multi-threaded applications.
    pub const SCHED_YIELD: u32 = 0x02;

    /// Block the calling thread on a synchronisation primitive. Lowest CPU
    /// usage but slightly higher latency.
    pub const SCHED_BLOCKING_SYNC: u32 = 0x04;

    /// Enable mapped pinned allocations in this context.
    pub const MAP_HOST: u32 = 0x08;

    /// Keep local memory allocation after launch (deprecated flag kept for
    /// completeness).
    pub const LMEM_RESIZE_TO_MAX: u32 = 0x10;
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// RAII wrapper for a CUDA context.
///
/// A context is created on a specific [`Device`] and becomes the active
/// context for the calling thread. When the `Context` is dropped,
/// `cuCtxDestroy_v2` is called automatically.
///
/// # Examples
///
/// ```no_run
/// use oxicuda_driver::context::Context;
/// use oxicuda_driver::device::Device;
///
/// oxicuda_driver::init()?;
/// let dev = Device::get(0)?;
/// let ctx = Context::new(&dev)?;
/// println!("Context on device {}", ctx.device().ordinal());
/// ctx.synchronize()?;
/// // ctx is destroyed when it goes out of scope
/// # Ok::<(), oxicuda_driver::error::CudaError>(())
/// ```
pub struct Context {
    /// The raw CUDA context handle.
    raw: CUcontext,
    /// The device this context was created on.
    device: Device,
}

impl Context {
    // -- Construction --------------------------------------------------------

    /// Create a new context on the given device with default flags
    /// ([`flags::SCHED_AUTO`]).
    ///
    /// The new context is automatically pushed onto the calling thread's
    /// context stack and becomes the current context.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver cannot create the context (e.g., device
    /// is invalid, out of resources).
    pub fn new(device: &Device) -> CudaResult<Self> {
        Self::with_flags(device, flags::SCHED_AUTO)
    }

    /// Create a new context on the given device with specific scheduling flags.
    ///
    /// See the [`flags`] module for available values. Multiple flags can be
    /// combined with bitwise OR.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver cannot create the context.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oxicuda_driver::context::{Context, flags};
    /// use oxicuda_driver::device::Device;
    ///
    /// oxicuda_driver::init()?;
    /// let dev = Device::get(0)?;
    /// let ctx = Context::with_flags(&dev, flags::SCHED_BLOCKING_SYNC)?;
    /// # Ok::<(), oxicuda_driver::error::CudaError>(())
    /// ```
    pub fn with_flags(device: &Device, flags: u32) -> CudaResult<Self> {
        let driver = try_driver()?;
        let mut raw = CUcontext::default();
        crate::error::check(unsafe { (driver.cu_ctx_create_v2)(&mut raw, flags, device.raw()) })?;
        Ok(Self {
            raw,
            device: *device,
        })
    }

    // -- Current context management -----------------------------------------

    /// Set this context as the current context for the calling thread.
    ///
    /// Any previous context on this thread is detached (but not destroyed).
    ///
    /// # Errors
    ///
    /// Returns an error if the driver call fails.
    pub fn set_current(&self) -> CudaResult<()> {
        let driver = try_driver()?;
        crate::error::check(unsafe { (driver.cu_ctx_set_current)(self.raw) })
    }

    /// Get the raw handle of the current context for the calling thread.
    ///
    /// Returns `None` if no context is bound to the current thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver call fails.
    pub fn current_raw() -> CudaResult<Option<CUcontext>> {
        let driver = try_driver()?;
        let mut ctx = CUcontext::default();
        crate::error::check(unsafe { (driver.cu_ctx_get_current)(&mut ctx) })?;
        if ctx.is_null() {
            Ok(None)
        } else {
            Ok(Some(ctx))
        }
    }

    // -- Synchronisation ----------------------------------------------------

    /// Block until all pending GPU operations in this context have completed.
    ///
    /// This sets the context as current before synchronising to ensure the
    /// correct context is targeted.
    ///
    /// # Errors
    ///
    /// Returns an error if any GPU operation failed or the driver call fails.
    pub fn synchronize(&self) -> CudaResult<()> {
        self.set_current()?;
        let driver = try_driver()?;
        crate::error::check(unsafe { (driver.cu_ctx_synchronize)() })
    }

    // -- Scoped execution ---------------------------------------------------

    /// Execute a closure with this context set as current, then restore the
    /// previous context.
    ///
    /// This is useful when temporarily switching contexts. The previous
    /// context (if any) is restored even if the closure returns an error.
    ///
    /// # Errors
    ///
    /// Propagates any error from the closure. Context-restoration errors are
    /// logged but do not override the closure result.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oxicuda_driver::context::Context;
    /// use oxicuda_driver::device::Device;
    ///
    /// oxicuda_driver::init()?;
    /// let dev = Device::get(0)?;
    /// let ctx = Context::new(&dev)?;
    /// let result = ctx.scoped(|| {
    ///     // ctx is current here
    ///     Ok(42)
    /// })?;
    /// assert_eq!(result, 42);
    /// # Ok::<(), oxicuda_driver::error::CudaError>(())
    /// ```
    pub fn scoped<F, R>(&self, f: F) -> CudaResult<R>
    where
        F: FnOnce() -> CudaResult<R>,
    {
        // Save the currently active context (may be None).
        let prev = Self::current_raw()?;

        // Activate this context.
        self.set_current()?;

        // Run the user closure.
        let result = f();

        // Restore the previous context. A null CUcontext detaches any context
        // from the current thread, which is the correct behaviour when there
        // was no previous context.
        let restore_ctx = prev.unwrap_or_default();
        if let Ok(driver) = try_driver() {
            if let Err(e) = crate::error::check(unsafe { (driver.cu_ctx_set_current)(restore_ctx) })
            {
                tracing::warn!("failed to restore previous context: {e}");
            }
        }

        result
    }

    // -- Accessors ----------------------------------------------------------

    /// Get a reference to the [`Device`] this context was created on.
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the raw `CUcontext` handle for use with FFI calls.
    #[inline]
    pub fn raw(&self) -> CUcontext {
        self.raw
    }

    /// Returns `true` if this context is the current context on the calling
    /// thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver call fails.
    pub fn is_current(&self) -> CudaResult<bool> {
        match Self::current_raw()? {
            Some(ctx) => Ok(ctx == self.raw),
            None => Ok(false),
        }
    }
}

// ---------------------------------------------------------------------------
// Drop
// ---------------------------------------------------------------------------

impl Drop for Context {
    /// Destroy the CUDA context.
    ///
    /// Errors during destruction are logged via `tracing::warn` but never
    /// propagated (destructors must not panic).
    fn drop(&mut self) {
        if let Ok(driver) = try_driver() {
            let result = unsafe { (driver.cu_ctx_destroy_v2)(self.raw) };
            if result != 0 {
                tracing::warn!(
                    "cuCtxDestroy_v2 failed with error code {result} during Context drop \
                     (device ordinal {})",
                    self.device.ordinal()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

// SAFETY: CUDA contexts can be migrated between threads via cuCtxSetCurrent.
// The caller is responsible for calling set_current() on the new thread.
unsafe impl Send for Context {}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("raw", &self.raw)
            .field("device", &self.device)
            .finish()
    }
}
