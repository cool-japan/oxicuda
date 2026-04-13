//! Launch telemetry / tracing integration.
//!
//! This module provides the [`kernel_launch_span!`] macro that emits a
//! [`tracing`] instrumentation span around every kernel launch when the
//! `tracing` Cargo feature is enabled. When the feature is disabled the
//! macro expands to a unit expression `()` with zero overhead.
//!
//! # Feature gate
//!
//! Add `tracing` to the features in `Cargo.toml` to enable live spans:
//!
//! ```toml
//! [dependencies]
//! oxicuda-launch = { version = "*", features = ["tracing"] }
//! ```
//!
//! # Span fields
//!
//! When enabled, each span carries:
//!
//! | Field    | Type     | Description                              |
//! |----------|----------|------------------------------------------|
//! | `kernel` | `&str`   | Function name as passed to `Kernel::new` |
//! | `grid`   | Debug    | Grid dimensions `(x, y, z)`             |
//! | `block`  | Debug    | Block dimensions `(x, y, z)`            |
//!
//! # Example (with tracing enabled)
//!
//! ```rust
//! use oxicuda_launch::trace::kernel_launch_span;
//!
//! let _span = kernel_launch_span!("my_kernel", (4u32, 1u32, 1u32), (256u32, 1u32, 1u32));
//! // The span is entered while `_span` is in scope.
//! ```

// =========================================================================
// kernel_launch_span! macro
// =========================================================================

/// Emit a tracing span for a kernel launch.
///
/// # Arguments
///
/// * `$name` — kernel function name (string literal or `&str`).
/// * `$grid` — grid dimensions (anything that implements `Debug`).
/// * `$block` — block dimensions (anything that implements `Debug`).
///
/// # Behaviour
///
/// * With the `tracing` feature: returns an [`tracing::Span`] entered at
///   `INFO` level.  Assign the result to a variable; the span closes when
///   the variable is dropped.
/// * Without the `tracing` feature: expands to `()`.
#[cfg(feature = "tracing")]
#[macro_export]
macro_rules! kernel_launch_span {
    ($name:expr, $grid:expr, $block:expr) => {
        tracing::info_span!(
            "kernel_launch",
            kernel = $name,
            grid   = ?$grid,
            block  = ?$block,
        )
    };
}

/// No-op version used when the `tracing` feature is disabled.
#[cfg(not(feature = "tracing"))]
#[macro_export]
macro_rules! kernel_launch_span {
    ($name:expr, $grid:expr, $block:expr) => {
        ()
    };
}

// Re-export so callers can use `oxicuda_launch::trace::kernel_launch_span`.
pub use kernel_launch_span;

// =========================================================================
// KernelSpanGuard — RAII wrapper that enters/exits the span
// =========================================================================

/// RAII guard that enters a kernel-launch tracing span and exits on drop.
///
/// When the `tracing` feature is disabled this struct is zero-sized.
///
/// # Usage
///
/// ```rust
/// use oxicuda_launch::trace::KernelSpanGuard;
///
/// let _guard = KernelSpanGuard::enter("vector_add", (4u32, 1u32, 1u32), (256u32, 1u32, 1u32));
/// // Span active here — dropped at end of scope.
/// ```
#[must_use = "The span is closed when the guard is dropped; don't discard it."]
pub struct KernelSpanGuard {
    #[cfg(feature = "tracing")]
    _entered: tracing::span::EnteredSpan,
}

impl KernelSpanGuard {
    /// Enter a new kernel-launch span.
    ///
    /// The span is active (entered) until this guard is dropped.
    pub fn enter<G, B>(kernel_name: &str, grid: G, block: B) -> Self
    where
        G: std::fmt::Debug,
        B: std::fmt::Debug,
    {
        #[cfg(feature = "tracing")]
        {
            let span = tracing::info_span!(
                "kernel_launch",
                kernel = kernel_name,
                grid   = ?grid,
                block  = ?block,
            );
            Self {
                _entered: span.entered(),
            }
        }

        #[cfg(not(feature = "tracing"))]
        {
            // Suppress "unused variable" warnings in no-tracing builds.
            let _ = kernel_name;
            let _ = grid;
            let _ = block;
            Self {}
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// KernelSpanGuard must be zero-sized when tracing is disabled, or a
    /// reasonable size when enabled.
    #[test]
    fn test_kernel_span_guard_size() {
        // Zero bytes without tracing; non-zero with tracing (contains EnteredSpan).
        #[cfg(feature = "tracing")]
        assert!(std::mem::size_of::<KernelSpanGuard>() > 0);

        #[cfg(not(feature = "tracing"))]
        assert_eq!(std::mem::size_of::<KernelSpanGuard>(), 0);
    }

    #[test]
    fn test_kernel_span_guard_drop() {
        // Should not panic when dropped.
        {
            let _guard =
                KernelSpanGuard::enter("test_kernel", (1u32, 1u32, 1u32), (32u32, 1u32, 1u32));
        }
        // If we reach here, no panic occurred.
    }

    #[test]
    fn test_kernel_span_guard_enter_and_exit() {
        let guard = KernelSpanGuard::enter("matmul", (16u32, 16u32, 1u32), (16u32, 16u32, 1u32));
        // Guard is active; doing some work.
        let _ = 1 + 1;
        drop(guard); // Span exited here.
    }

    #[test]
    fn test_macro_invocation_compiles() {
        // Verify the macro expands without error.
        let _span = kernel_launch_span!("test_kernel", (4u32, 1u32, 1u32), (256u32, 1u32, 1u32));
        // On tracing builds _span is a Span; on no-tracing builds it is ().
        let _ = _span;
    }
}
