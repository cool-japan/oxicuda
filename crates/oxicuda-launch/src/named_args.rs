//! Named kernel arguments for enhanced debuggability and type safety.
//!
//! This module provides [`NamedKernelArgs`], a trait that extends
//! [`KernelArgs`] with human-readable argument names, and [`ArgBuilder`],
//! a builder that constructs argument pointer arrays with associated
//! names for logging and debugging.
//!
//! # Motivation
//!
//! Standard kernel arguments are positional tuples with no names. When
//! debugging kernel launches, it is helpful to know which argument
//! corresponds to which kernel parameter. `NamedKernelArgs` bridges
//! this gap by associating names with arguments.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_launch::named_args::ArgBuilder;
//!
//! let a_ptr: u64 = 0x1000;
//! let n: u32 = 1024;
//! let mut builder = ArgBuilder::new();
//! builder.add("a_ptr", &a_ptr).add("n", &n);
//! assert_eq!(builder.names(), &["a_ptr", "n"]);
//! let ptrs = builder.build();
//! assert_eq!(ptrs.len(), 2);
//! ```

use std::ffi::c_void;

use crate::kernel::KernelArgs;

// ---------------------------------------------------------------------------
// NamedKernelArgs trait
// ---------------------------------------------------------------------------

/// Extension of [`KernelArgs`] that provides argument metadata.
///
/// Types implementing this trait can report the names and count of
/// their kernel arguments, which is useful for debugging, logging,
/// and validation.
///
/// # Safety
///
/// Implementors must uphold the same invariants as [`KernelArgs`].
/// The names returned by `arg_names` must correspond one-to-one with
/// the pointers returned by `as_param_ptrs`.
pub unsafe trait NamedKernelArgs: KernelArgs {
    /// Returns the names of all kernel arguments in order.
    fn arg_names() -> &'static [&'static str];

    /// Returns the number of kernel arguments.
    fn arg_count() -> usize {
        Self::arg_names().len()
    }
}

// ---------------------------------------------------------------------------
// ArgEntry
// ---------------------------------------------------------------------------

/// An entry in the argument builder, holding a pointer and its name.
#[derive(Debug)]
struct ArgEntry {
    /// Pointer to the argument value.
    ptr: *mut c_void,
    /// Human-readable name for the argument.
    name: String,
}

// ---------------------------------------------------------------------------
// ArgBuilder
// ---------------------------------------------------------------------------

/// A builder for constructing named kernel argument arrays.
///
/// Collects typed argument values along with their names, then
/// produces the `Vec<*mut c_void>` array needed by `cuLaunchKernel`.
///
/// # Example
///
/// ```rust
/// use oxicuda_launch::named_args::ArgBuilder;
///
/// let x: f32 = 3.14;
/// let n: u32 = 512;
/// let mut builder = ArgBuilder::new();
/// builder.add("x", &x).add("n", &n);
/// assert_eq!(builder.arg_count(), 2);
/// let ptrs = builder.build();
/// assert_eq!(ptrs.len(), 2);
/// ```
pub struct ArgBuilder {
    /// Collected argument entries.
    args: Vec<ArgEntry>,
}

impl ArgBuilder {
    /// Creates a new empty argument builder.
    #[inline]
    pub fn new() -> Self {
        Self { args: Vec::new() }
    }

    /// Creates a new argument builder with the given initial capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            args: Vec::with_capacity(capacity),
        }
    }

    /// Adds a named argument to the builder.
    ///
    /// The pointer to `val` is stored. The caller must ensure that `val`
    /// remains valid (not moved or dropped) until the kernel launch
    /// using the built pointer array completes.
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn add<T: Copy>(&mut self, name: &str, val: &T) -> &mut Self {
        self.args.push(ArgEntry {
            ptr: val as *const T as *mut c_void,
            name: name.to_owned(),
        });
        self
    }

    /// Builds the argument pointer array for `cuLaunchKernel`.
    ///
    /// Returns the raw pointer array. The names are consumed; use
    /// [`names`](Self::names) before calling `build` if you need them.
    pub fn build(self) -> Vec<*mut c_void> {
        self.args.into_iter().map(|entry| entry.ptr).collect()
    }

    /// Returns the names of all added arguments in order.
    pub fn names(&self) -> Vec<&str> {
        self.args.iter().map(|entry| entry.name.as_str()).collect()
    }

    /// Returns the number of arguments added so far.
    #[inline]
    pub fn arg_count(&self) -> usize {
        self.args.len()
    }

    /// Returns `true` if no arguments have been added.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }

    /// Returns a human-readable summary of the arguments.
    pub fn summary(&self) -> String {
        let parts: Vec<String> = self
            .args
            .iter()
            .map(|entry| format!("{}={:p}", entry.name, entry.ptr))
            .collect();
        format!("ArgBuilder[{}]", parts.join(", "))
    }
}

impl Default for ArgBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ArgBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArgBuilder")
            .field("count", &self.args.len())
            .field("names", &self.names())
            .finish()
    }
}

// Implement NamedKernelArgs for () (no arguments).
//
// SAFETY: Returns an empty name array, consistent with the empty
// pointer array from KernelArgs for ().
unsafe impl NamedKernelArgs for () {
    fn arg_names() -> &'static [&'static str] {
        &[]
    }
}

// ---------------------------------------------------------------------------
// FixedNamedArgs — stack-allocated named argument array
// ---------------------------------------------------------------------------

/// A single named argument entry: a static string name paired with a raw
/// const pointer to the argument value (already on the stack in the caller).
///
/// The pointer is `*const c_void` so no allocation occurs; it is the
/// caller's responsibility to ensure the pointed-to value outlives the
/// kernel launch.
#[derive(Debug, Clone, Copy)]
pub struct NamedArgEntry {
    /// The human-readable name of this kernel argument.
    pub name: &'static str,
    /// Raw const pointer to the argument value.
    pub ptr: *const c_void,
}

// SAFETY: NamedArgEntry only holds a raw pointer to a value that lives
// at least as long as the kernel launch (caller guarantees this).
// The pointer is never dereferenced inside this module.
unsafe impl Send for NamedArgEntry {}
unsafe impl Sync for NamedArgEntry {}

/// A const-generic, stack-allocated array of named kernel arguments.
///
/// `FixedNamedArgs<N>` stores exactly `N` [`NamedArgEntry`] values on
/// the stack — zero heap allocation, zero indirection overhead compared
/// to a plain positional tuple.
///
/// # Invariants
///
/// Every pointed-to value must remain valid (not moved or dropped) until
/// after the kernel launch that consumes this struct completes.
///
/// # Example
///
/// ```rust
/// use oxicuda_launch::named_args::{FixedNamedArgs, NamedArgEntry};
///
/// let n: u32 = 1024;
/// let alpha: f32 = 2.0;
///
/// let args = FixedNamedArgs::new([
///     NamedArgEntry { name: "n",     ptr: &n     as *const u32 as *const std::ffi::c_void },
///     NamedArgEntry { name: "alpha", ptr: &alpha as *const f32  as *const std::ffi::c_void },
/// ]);
///
/// assert_eq!(args.len(), 2);
/// assert_eq!(args.names()[0], "n");
/// assert_eq!(args.names()[1], "alpha");
/// ```
pub struct FixedNamedArgs<const N: usize> {
    /// The argument entries, stored inline on the stack.
    entries: [NamedArgEntry; N],
}

impl<const N: usize> FixedNamedArgs<N> {
    /// Creates a new `FixedNamedArgs` from an array of [`NamedArgEntry`] values.
    #[inline]
    pub const fn new(entries: [NamedArgEntry; N]) -> Self {
        Self { entries }
    }

    /// Returns the number of arguments.
    #[inline]
    pub const fn len(&self) -> usize {
        N
    }

    /// Returns `true` if there are no arguments.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Returns the argument names in declaration order.
    pub fn names(&self) -> [&'static str; N] {
        let mut out = [""; N];
        for (i, entry) in self.entries.iter().enumerate() {
            out[i] = entry.name;
        }
        out
    }

    /// Returns a mutable array of `*mut c_void` pointers suitable for
    /// passing directly to `cuLaunchKernel` as the `kernelParams` array.
    ///
    /// # Safety
    ///
    /// The returned pointers are valid only as long as the original values
    /// that were passed when constructing the entries remain in scope.
    pub fn as_ptr_array(&self) -> [*mut c_void; N] {
        let mut out = [std::ptr::null_mut::<c_void>(); N];
        for (i, entry) in self.entries.iter().enumerate() {
            // Cast const → mut: cuLaunchKernel's ABI takes `void**` but
            // never mutates the argument values.
            out[i] = entry.ptr as *mut c_void;
        }
        out
    }

    /// Returns an immutable slice over the entries for inspection.
    #[inline]
    pub fn entries(&self) -> &[NamedArgEntry; N] {
        &self.entries
    }
}

impl<const N: usize> std::fmt::Debug for FixedNamedArgs<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("FixedNamedArgs");
        ds.field("len", &N);
        for entry in &self.entries {
            ds.field(entry.name, &entry.ptr);
        }
        ds.finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arg_builder_new_empty() {
        let builder = ArgBuilder::new();
        assert!(builder.is_empty());
        assert_eq!(builder.arg_count(), 0);
    }

    #[test]
    fn arg_builder_add_and_names() {
        let x: u32 = 42;
        let y: f64 = 2.78;
        let mut builder = ArgBuilder::new();
        builder.add("x", &x).add("y", &y);
        assert_eq!(builder.arg_count(), 2);
        assert_eq!(builder.names(), vec!["x", "y"]);
    }

    #[test]
    fn arg_builder_build_pointer_count() {
        let a: u64 = 0x1000;
        let b: u32 = 512;
        let c: f32 = 1.0;
        let mut builder = ArgBuilder::new();
        builder.add("a", &a).add("b", &b).add("c", &c);
        let ptrs = builder.build();
        assert_eq!(ptrs.len(), 3);
    }

    #[test]
    fn arg_builder_build_pointers_valid() {
        let val: u32 = 99;
        let mut builder = ArgBuilder::new();
        builder.add("val", &val);
        let ptrs = builder.build();
        assert_eq!(ptrs.len(), 1);
        let read_back = unsafe { *(ptrs[0] as *const u32) };
        assert_eq!(read_back, 99);
    }

    #[test]
    fn arg_builder_summary() {
        let x: u32 = 10;
        let mut builder = ArgBuilder::new();
        builder.add("x", &x);
        let s = builder.summary();
        assert!(s.starts_with("ArgBuilder["));
        assert!(s.contains("x="));
    }

    #[test]
    fn arg_builder_debug() {
        let builder = ArgBuilder::new();
        let dbg = format!("{builder:?}");
        assert!(dbg.contains("ArgBuilder"));
        assert!(dbg.contains("count"));
    }

    #[test]
    fn arg_builder_default() {
        let builder = ArgBuilder::default();
        assert!(builder.is_empty());
    }

    #[test]
    fn arg_builder_with_capacity() {
        let builder = ArgBuilder::with_capacity(8);
        assert!(builder.is_empty());
        assert_eq!(builder.arg_count(), 0);
    }

    #[test]
    fn named_kernel_args_trait_exists() {
        // Verify the trait compiles with the safety requirement.
        fn assert_named<T: NamedKernelArgs>() {
            let _names = T::arg_names();
            let _count = T::arg_count();
        }
        // We cannot call assert_named without an implementor, but
        // the existence of the function is enough.
        let _ = assert_named::<()> as *const ();
    }

    // -----------------------------------------------------------------------
    // FixedNamedArgs tests
    // -----------------------------------------------------------------------

    #[test]
    fn fixed_named_args_zero_size() {
        let args: FixedNamedArgs<0> = FixedNamedArgs::new([]);
        assert!(args.is_empty());
        assert_eq!(args.len(), 0);
        let ptrs = args.as_ptr_array();
        assert_eq!(ptrs.len(), 0);
    }

    #[test]
    fn fixed_named_args_single_u32() {
        let n: u32 = 1024;
        let args = FixedNamedArgs::new([NamedArgEntry {
            name: "n",
            ptr: &n as *const u32 as *const c_void,
        }]);
        assert_eq!(args.len(), 1);
        assert!(!args.is_empty());
        assert_eq!(args.names(), ["n"]);

        let ptrs = args.as_ptr_array();
        assert_eq!(ptrs.len(), 1);
        // Verify the pointer round-trips to the original value.
        let read_back = unsafe { *(ptrs[0] as *const u32) };
        assert_eq!(read_back, 1024);
    }

    #[test]
    fn fixed_named_args_two_entries_order_preserved() {
        let n: u32 = 512;
        let alpha: f32 = std::f32::consts::PI;
        let args = FixedNamedArgs::new([
            NamedArgEntry {
                name: "n",
                ptr: &n as *const u32 as *const c_void,
            },
            NamedArgEntry {
                name: "alpha",
                ptr: &alpha as *const f32 as *const c_void,
            },
        ]);

        assert_eq!(args.len(), 2);
        let names = args.names();
        assert_eq!(names[0], "n");
        assert_eq!(names[1], "alpha");

        let ptrs = args.as_ptr_array();
        let n_back = unsafe { *(ptrs[0] as *const u32) };
        let a_back = unsafe { *(ptrs[1] as *const f32) };
        assert_eq!(n_back, 512);
        assert!((a_back - std::f32::consts::PI).abs() < f32::EPSILON);
    }

    #[test]
    fn fixed_named_args_no_size_overhead_vs_entry_array() {
        use std::mem::size_of;
        // FixedNamedArgs<N> must have the same size as [NamedArgEntry; N].
        assert_eq!(
            size_of::<FixedNamedArgs<4>>(),
            size_of::<[NamedArgEntry; 4]>(),
            "FixedNamedArgs<4> must not add any size overhead"
        );
        assert_eq!(
            size_of::<FixedNamedArgs<1>>(),
            size_of::<[NamedArgEntry; 1]>(),
            "FixedNamedArgs<1> must not add any size overhead"
        );
    }

    #[test]
    fn fixed_named_args_ptr_array_length_matches_n() {
        let a: u64 = 0x1000_0000;
        let b: u32 = 256;
        let c: f64 = 1.0;
        let args = FixedNamedArgs::new([
            NamedArgEntry {
                name: "a",
                ptr: &a as *const u64 as *const c_void,
            },
            NamedArgEntry {
                name: "b",
                ptr: &b as *const u32 as *const c_void,
            },
            NamedArgEntry {
                name: "c",
                ptr: &c as *const f64 as *const c_void,
            },
        ]);
        let ptrs = args.as_ptr_array();
        assert_eq!(ptrs.len(), 3);
    }

    #[test]
    fn fixed_named_args_debug_contains_len() {
        let n: u32 = 42;
        let args = FixedNamedArgs::new([NamedArgEntry {
            name: "n",
            ptr: &n as *const u32 as *const c_void,
        }]);
        let dbg = format!("{args:?}");
        assert!(dbg.contains("FixedNamedArgs"), "Debug output: {dbg}");
        assert!(
            dbg.contains('1') || dbg.contains("len"),
            "Debug output: {dbg}"
        );
    }

    #[test]
    fn fixed_named_args_entries_accessor() {
        let x: f32 = 7.0;
        let args = FixedNamedArgs::new([NamedArgEntry {
            name: "x",
            ptr: &x as *const f32 as *const c_void,
        }]);
        let entries = args.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "x");
    }

    // ---------------------------------------------------------------------------
    // Quality gate tests (CPU-only)
    // ---------------------------------------------------------------------------

    #[test]
    fn named_kernel_args_empty() {
        // NamedKernelArgs for () has no args: arg_names is empty, arg_count is 0.
        let names = <() as NamedKernelArgs>::arg_names();
        assert!(names.is_empty(), "() must have no arg names");
        let count = <() as NamedKernelArgs>::arg_count();
        assert_eq!(count, 0, "() must have arg_count == 0");
    }

    #[test]
    fn named_kernel_args_add_and_count() {
        // After adding 3 named args to ArgBuilder, arg_count() == 3.
        let a: u32 = 1;
        let b: f64 = 2.0;
        let c: u64 = 3;
        let mut builder = ArgBuilder::new();
        builder.add("a", &a).add("b", &b).add("c", &c);
        assert_eq!(
            builder.arg_count(),
            3,
            "ArgBuilder with 3 args must report arg_count == 3"
        );
        // Names must be in insertion order
        let names = builder.names();
        assert_eq!(names[0], "a");
        assert_eq!(names[1], "b");
        assert_eq!(names[2], "c");
    }
}
