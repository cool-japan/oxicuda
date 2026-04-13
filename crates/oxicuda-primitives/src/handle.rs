//! Session handle for OxiCUDA Primitives operations.
//!
//! [`PrimitivesHandle`] encapsulates the CUDA context, stream, and SM version
//! that all primitives kernels are bound to.  Create one handle per session
//! (device context + stream combination) and reuse it across multiple calls.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use oxicuda_primitives::handle::PrimitivesHandle;
//! use oxicuda_driver::{Context, Device, Stream};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! oxicuda_driver::init()?;
//! let dev = Device::get(0)?;
//! let ctx = Arc::new(Context::new(&dev)?);
//! let stream = Arc::new(Stream::new(&ctx)?);
//! let cc = dev.compute_capability()?;
//! let handle = PrimitivesHandle::from_arc(ctx, stream, cc);
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use oxicuda_driver::{Context, Stream};
use oxicuda_ptx::arch::SmVersion;

// ─── SmVersion conversion helper ────────────────────────────────────────────

/// Convert a `(major, minor)` compute-capability pair to an [`SmVersion`].
///
/// Unknown / future capability pairs default to `Sm80`.
fn cc_to_sm(major: i32, minor: i32) -> SmVersion {
    match (major, minor) {
        (7, 5) => SmVersion::Sm75,
        (8, 0) => SmVersion::Sm80,
        (8, 6) => SmVersion::Sm86,
        (8, 9) => SmVersion::Sm89,
        (9, 0) => SmVersion::Sm90,
        (10, 0) => SmVersion::Sm100,
        (12, 0) => SmVersion::Sm120,
        // Default: if above 10.x treat as Sm80 safe baseline
        _ => SmVersion::Sm80,
    }
}

// ─── PrimitivesHandle ────────────────────────────────────────────────────────

/// Session handle for parallel-primitive operations.
///
/// Wraps the CUDA context, stream, and target SM version.  All primitives that
/// need to JIT-compile PTX kernels use the embedded SM version to select the
/// optimal code path.
///
/// The handle is cheap to clone (all fields are reference-counted).
#[derive(Clone)]
pub struct PrimitivesHandle {
    /// CUDA context owning the device memory and kernels.
    pub(crate) ctx: Arc<Context>,
    /// Stream on which kernel launches are serialised.
    pub(crate) stream: Arc<Stream>,
    /// Target SM version used for PTX kernel generation.
    pub(crate) sm: SmVersion,
}

impl PrimitivesHandle {
    /// Create a new primitives handle from the given context, stream, and
    /// compute capability.
    ///
    /// # Arguments
    ///
    /// * `ctx` — CUDA context (RAII guard; kept alive as long as the handle).
    /// * `stream` — CUDA stream for asynchronous kernel launches.
    /// * `cc` — `(major, minor)` compute capability of the target device
    ///   (e.g. `(8, 0)` for Ampere A100).
    #[must_use]
    pub fn new(ctx: Context, stream: Stream, cc: (i32, i32)) -> Self {
        Self {
            ctx: Arc::new(ctx),
            stream: Arc::new(stream),
            sm: cc_to_sm(cc.0, cc.1),
        }
    }

    /// Create a primitives handle from an already-shared context and stream.
    ///
    /// Useful when the caller holds the context in an `Arc` and wants to avoid
    /// an extra clone.
    #[must_use]
    pub fn from_arc(ctx: Arc<Context>, stream: Arc<Stream>, cc: (i32, i32)) -> Self {
        Self {
            ctx,
            stream,
            sm: cc_to_sm(cc.0, cc.1),
        }
    }

    /// Return the target [`SmVersion`] this handle was created for.
    #[must_use]
    pub fn sm_version(&self) -> SmVersion {
        self.sm
    }

    /// Return the underlying [`Context`].
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Return the underlying [`Stream`].
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

impl std::fmt::Debug for PrimitivesHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimitivesHandle")
            .field("sm", &self.sm)
            .finish_non_exhaustive()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify cc_to_sm mapping for well-known architectures.
    #[test]
    fn cc_to_sm_known_architectures() {
        assert_eq!(cc_to_sm(7, 5), SmVersion::Sm75);
        assert_eq!(cc_to_sm(8, 0), SmVersion::Sm80);
        assert_eq!(cc_to_sm(8, 6), SmVersion::Sm86);
        assert_eq!(cc_to_sm(8, 9), SmVersion::Sm89);
        assert_eq!(cc_to_sm(9, 0), SmVersion::Sm90);
        assert_eq!(cc_to_sm(10, 0), SmVersion::Sm100);
        assert_eq!(cc_to_sm(12, 0), SmVersion::Sm120);
    }

    /// Unknown compute capability falls back to Sm80.
    #[test]
    fn cc_to_sm_unknown_fallback() {
        assert_eq!(cc_to_sm(6, 0), SmVersion::Sm80);
        assert_eq!(cc_to_sm(99, 99), SmVersion::Sm80);
    }

    /// Debug impl does not panic.
    #[test]
    fn handle_debug_impl() {
        // We cannot construct a real PrimitivesHandle without GPU, so test the
        // cc_to_sm conversion that is used internally instead.
        let sm = cc_to_sm(8, 0);
        assert_eq!(format!("{sm:?}"), "Sm80");
    }
}
