//! Convenience macros for kernel launching.
//!
//! The [`launch!`](crate::launch) macro provides a concise syntax for launching GPU kernels
//! without manually constructing [`LaunchParams`](crate::LaunchParams).
//!
//! The [`named_args!`](macro@crate::named_args) macro provides a zero-overhead way to pass named
//! kernel arguments — names are stripped at compile time, producing the same
//! tuple as positional arguments with no runtime overhead.
//!
//! The [`launch_named!`](crate::launch_named) macro combines named-argument syntax with the
//! `launch!` convenience macro.

/// Launch a GPU kernel with a concise syntax.
///
/// This macro constructs [`LaunchParams`](crate::LaunchParams) from the
/// provided grid and block dimensions, then calls [`Kernel::launch`](crate::Kernel::launch).
///
/// # Syntax
///
/// ```text
/// launch!(kernel, grid(G), block(B), shared(S), stream, args)?;
/// launch!(kernel, grid(G), block(B), stream, args)?;  // shared_mem = 0
/// ```
///
/// Where:
/// - `kernel` — a [`Kernel`](crate::Kernel) instance.
/// - `G` — grid dimensions (anything convertible to [`Dim3`](crate::Dim3)).
/// - `B` — block dimensions (anything convertible to [`Dim3`](crate::Dim3)).
/// - `S` — dynamic shared memory in bytes (`u32`).
/// - `stream` — a reference to a [`Stream`](oxicuda_driver::Stream).
/// - `args` — a reference to a tuple implementing [`KernelArgs`](crate::KernelArgs).
///
/// # Returns
///
/// `CudaResult<()>` — use `?` to propagate errors.
///
/// # Examples
///
/// ```rust,no_run
/// # use oxicuda_launch::*;
/// # fn main() -> oxicuda_driver::CudaResult<()> {
/// # let kernel: Kernel = todo!();
/// # let stream: oxicuda_driver::Stream = todo!();
/// let n: u32 = 1024;
/// let a_ptr: u64 = 0;
/// let b_ptr: u64 = 0;
/// let c_ptr: u64 = 0;
///
/// // With explicit shared memory
/// launch!(kernel, grid(4u32), block(256u32), shared(0), &stream, &(a_ptr, b_ptr, c_ptr, n))?;
///
/// // Without shared memory (defaults to 0)
/// launch!(kernel, grid(4u32), block(256u32), &stream, &(a_ptr, b_ptr, c_ptr, n))?;
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! launch {
    ($kernel:expr, grid($g:expr), block($b:expr), shared($s:expr), $stream:expr, $args:expr) => {{
        let params = $crate::LaunchParams::new($g, $b).with_shared_mem($s);
        $kernel.launch(&params, $stream, $args)
    }};
    ($kernel:expr, grid($g:expr), block($b:expr), $stream:expr, $args:expr) => {{
        let params = $crate::LaunchParams::new($g, $b);
        $kernel.launch(&params, $stream, $args)
    }};
}

/// Build a kernel argument tuple from named fields.
///
/// Names are stripped at compile time; the result is identical to a plain
/// positional tuple with zero runtime or size overhead.
///
/// # Syntax
///
/// ```text
/// named_args!(name1: value1, name2: value2, ...)
/// ```
///
/// # Returns
///
/// A tuple `(value1, value2, ...)` with the same types as the values.
///
/// # Examples
///
/// ```rust
/// use oxicuda_launch::named_args;
///
/// let n: u32 = 1024;
/// let alpha: f32 = 2.0;
///
/// // Named form — more readable, identical output to positional.
/// let named = named_args!(n: n, alpha: alpha);
/// // Positional form — for comparison.
/// let positional = (n, alpha);
///
/// assert_eq!(named, positional);
/// ```
#[macro_export]
macro_rules! named_args {
    // Base case: nothing → unit tuple.
    () => { () };

    // One or more `name: value` pairs — strip all names, keep values as tuple.
    ($($name:ident : $val:expr),+ $(,)?) => {
        ($($val,)*)
    };
}

/// Launch a GPU kernel with named argument syntax.
///
/// This macro strips the argument names at compile time and delegates to
/// [`launch!`](crate::launch), so there is zero runtime overhead versus the positional form.
///
/// # Syntax
///
/// ```text
/// launch_named!(kernel, grid(G), block(B), shared(S), stream, {
///     name1: value1,
///     name2: value2,
/// })?;
///
/// // Without explicit shared memory (defaults to 0):
/// launch_named!(kernel, grid(G), block(B), stream, {
///     name1: value1,
///     name2: value2,
/// })?;
/// ```
///
/// # Examples
///
/// ```rust,no_run
/// # use oxicuda_launch::*;
/// # fn main() -> oxicuda_driver::CudaResult<()> {
/// # let kernel: Kernel = todo!();
/// # let stream: oxicuda_driver::Stream = todo!();
/// let n: u32 = 1024;
/// let a_ptr: u64 = 0;
/// let b_ptr: u64 = 0;
/// let c_ptr: u64 = 0;
///
/// launch_named!(kernel, grid(4u32), block(256u32), &stream, {
///     n: n,
///     a: a_ptr,
///     b: b_ptr,
///     c: c_ptr,
/// })?;
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! launch_named {
    // With explicit shared memory.
    ($kernel:expr, grid($g:expr), block($b:expr), shared($s:expr), $stream:expr, {
        $($name:ident : $val:expr),+ $(,)?
    }) => {{
        let args = $crate::named_args!($($name: $val),+);
        $crate::launch!($kernel, grid($g), block($b), shared($s), $stream, &args)
    }};

    // Without shared memory (defaults to 0).
    ($kernel:expr, grid($g:expr), block($b:expr), $stream:expr, {
        $($name:ident : $val:expr),+ $(,)?
    }) => {{
        let args = $crate::named_args!($($name: $val),+);
        $crate::launch!($kernel, grid($g), block($b), $stream, &args)
    }};
}

// ---------------------------------------------------------------------------
// Tests for named_args! and launch_named! macro (no GPU required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    #[test]
    fn test_named_args_produces_correct_tuple_two_fields() {
        let n = 1024u32;
        let alpha = 2.0f32;

        let pos = (n, alpha);
        let named = named_args!(n: n, alpha: alpha);

        assert_eq!(pos, named);
    }

    #[test]
    fn test_named_args_single_field() {
        let x = 42u64;
        let named = named_args!(x: x);
        // A single-element named_args! produces a 1-tuple: (x,)
        assert_eq!(named.0, 42u64);
    }

    #[test]
    fn test_named_args_three_fields_order_preserved() {
        let a = 1u32;
        let b = 2u64;
        let c = 3.0f32;

        let named = named_args!(a: a, b: b, c: c);
        assert_eq!(named.0, 1u32);
        assert_eq!(named.1, 2u64);
        assert!((named.2 - 3.0f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_named_args_no_extra_size_vs_positional() {
        // The macro produces the same tuple type — sizes must match.
        assert_eq!(
            size_of::<(u32, f32)>(),
            size_of::<(u32, f32)>(),
            "named_args! tuple must be the same size as positional tuple"
        );
        // Verify via type inference: named_args! produces the same type as
        // its positional equivalent.
        let n = 1024u32;
        let alpha = 2.0f32;
        let named = named_args!(n: n, alpha: alpha);
        // If this compiles, the types are identical.
        let _: (u32, f32) = named;
    }

    #[test]
    fn test_named_args_trailing_comma_allowed() {
        let x = 7u32;
        let y = 8u64;
        // Trailing comma must be accepted without compile error.
        let named = named_args!(x: x, y: y,);
        assert_eq!(named.0, 7u32);
        assert_eq!(named.1, 8u64);
    }

    #[test]
    fn test_named_args_expressions_evaluated() {
        // Values in named_args! can be arbitrary expressions.
        let named = named_args!(result: 2u32 + 3u32, factor: 1.5f32 * 2.0f32);
        assert_eq!(named.0, 5u32);
        assert!((named.1 - 3.0f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_named_args_four_fields() {
        let n = 1024u32;
        let a: u64 = 0x1000;
        let b: u64 = 0x2000;
        let c: u64 = 0x3000;

        let named = named_args!(n: n, a: a, b: b, c: c);
        assert_eq!(named.0, 1024u32);
        assert_eq!(named.1, 0x1000u64);
        assert_eq!(named.2, 0x2000u64);
        assert_eq!(named.3, 0x3000u64);
    }
}
