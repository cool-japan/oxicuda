//! Type-safe GPU kernel management and argument passing.
//!
//! This module provides the [`Kernel`] struct for launching GPU kernels
//! and the [`KernelArgs`] trait for type-safe argument passing to CUDA
//! kernel functions.
//!
//! # Architecture
//!
//! A [`Kernel`] wraps a [`Function`] handle and holds an `Arc<Module>`
//! to ensure the PTX module remains loaded for the kernel's lifetime.
//! Arguments are passed via the [`KernelArgs`] trait, which converts
//! typed Rust values into the `*mut c_void` array that `cuLaunchKernel`
//! expects.
//!
//! # Tuple arguments
//!
//! The [`KernelArgs`] trait is implemented for tuples of `Copy` types
//! up to 24 elements. Each element must be `Copy` because kernel
//! arguments are passed by value to the GPU.
//!
//! # Example
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use oxicuda_driver::{Module, Stream, Context, Device};
//! # use oxicuda_launch::{Kernel, LaunchParams, Dim3};
//! # fn main() -> oxicuda_driver::CudaResult<()> {
//! # oxicuda_driver::init()?;
//! # let dev = Device::get(0)?;
//! # let ctx = Arc::new(Context::new(&dev)?);
//! # let ptx = "";
//! let module = Arc::new(Module::from_ptx(ptx)?);
//! let kernel = Kernel::from_module(module, "vector_add")?;
//!
//! let stream = Stream::new(&ctx)?;
//! let params = LaunchParams::new(4u32, 256u32);
//!
//! // Launch with typed arguments: (a_ptr, b_ptr, c_ptr, n)
//! let args = (0u64, 0u64, 0u64, 1024u32);
//! kernel.launch(&params, &stream, &args)?;
//! # Ok(())
//! # }
//! ```

use std::ffi::c_void;
use std::sync::Arc;

use oxicuda_driver::error::CudaResult;
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::module::{Function, Module};
use oxicuda_driver::stream::Stream;

use crate::params::LaunchParams;
use crate::trace::KernelSpanGuard;

// ---------------------------------------------------------------------------
// KernelArgs trait
// ---------------------------------------------------------------------------

/// Trait for types that can be passed as kernel arguments.
///
/// Kernel arguments must be convertible to an array of void pointers
/// that `cuLaunchKernel` accepts. Each pointer points to the argument
/// value on the host; the CUDA driver copies the values to the GPU
/// before the kernel executes.
///
/// # Safety
///
/// Implementors must ensure that:
/// - `as_param_ptrs` returns valid pointers to the argument values.
/// - The pointed-to values remain valid for the duration of the kernel launch
///   (i.e., until `cuLaunchKernel` returns).
/// - The argument types and sizes match what the kernel expects.
pub unsafe trait KernelArgs {
    /// Convert arguments to an array of void pointers for `cuLaunchKernel`.
    ///
    /// Each element in the returned `Vec` is a pointer to one kernel argument.
    /// The CUDA driver reads the value through each pointer and copies it
    /// to the GPU.
    fn as_param_ptrs(&self) -> Vec<*mut c_void>;
}

// ---------------------------------------------------------------------------
// KernelArgs — unit type (no arguments)
// ---------------------------------------------------------------------------

/// Implementation for kernels that take no arguments.
///
/// # Safety
///
/// Returns an empty pointer array, which is valid for zero-argument kernels.
unsafe impl KernelArgs for () {
    #[inline]
    fn as_param_ptrs(&self) -> Vec<*mut c_void> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// KernelArgs — tuple implementations via macro
// ---------------------------------------------------------------------------

/// Generates [`KernelArgs`] implementations for tuples of `Copy` types.
///
/// Each tuple element is converted to a `*mut c_void` by taking
/// a reference to the element and casting through `*const T`.
macro_rules! impl_kernel_args_tuple {
    ($($idx:tt: $T:ident),+) => {
        /// # Safety
        ///
        /// The pointers returned point into `self`, which must remain
        /// valid (i.e., not moved or dropped) until `cuLaunchKernel` returns.
        unsafe impl<$($T: Copy),+> KernelArgs for ($($T,)+) {
            #[inline]
            fn as_param_ptrs(&self) -> Vec<*mut c_void> {
                vec![
                    $(&self.$idx as *const $T as *mut c_void,)+
                ]
            }
        }
    };
}

impl_kernel_args_tuple!(0: A);
impl_kernel_args_tuple!(0: A, 1: B);
impl_kernel_args_tuple!(0: A, 1: B, 2: C);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X);

// ---------------------------------------------------------------------------
// Kernel struct
// ---------------------------------------------------------------------------

/// A launchable GPU kernel with module lifetime management.
///
/// Holds an `Arc<Module>` to ensure the PTX module remains loaded
/// as long as any `Kernel` references it. This is important because
/// [`Function`] handles become invalid once their parent module is
/// unloaded.
///
/// # Creating a kernel
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use oxicuda_driver::Module;
/// # use oxicuda_launch::Kernel;
/// # fn main() -> oxicuda_driver::CudaResult<()> {
/// # let ptx = "";
/// let module = Arc::new(Module::from_ptx(ptx)?);
/// let kernel = Kernel::from_module(module, "my_kernel")?;
/// println!("loaded kernel: {}", kernel.name());
/// # Ok(())
/// # }
/// ```
///
/// # Launching
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use oxicuda_driver::{Module, Stream, Context, Device};
/// # use oxicuda_launch::{Kernel, LaunchParams};
/// # fn main() -> oxicuda_driver::CudaResult<()> {
/// # oxicuda_driver::init()?;
/// # let dev = Device::get(0)?;
/// # let ctx = Arc::new(Context::new(&dev)?);
/// # let ptx = "";
/// # let module = Arc::new(Module::from_ptx(ptx)?);
/// # let kernel = Kernel::from_module(module, "my_kernel")?;
/// let stream = Stream::new(&ctx)?;
/// let params = LaunchParams::new(4u32, 256u32);
/// kernel.launch(&params, &stream, &(42u32, 1024u32))?;
/// # Ok(())
/// # }
/// ```
pub struct Kernel {
    /// The underlying CUDA function handle.
    function: Function,
    /// Keeps the parent module alive as long as this kernel exists.
    _module: Arc<Module>,
    /// The kernel function name (for debugging and diagnostics).
    name: String,
}

impl Kernel {
    /// Creates a new `Kernel` from a module and function name.
    ///
    /// Looks up the named function in the module. The `Arc<Module>` ensures
    /// the module is not unloaded while this kernel exists.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotFound`](oxicuda_driver::CudaError::NotFound) if no
    /// function with the given name exists in the module, or another
    /// [`CudaError`](oxicuda_driver::CudaError) on driver failure.
    pub fn from_module(module: Arc<Module>, name: &str) -> CudaResult<Self> {
        let function = module.get_function(name)?;
        Ok(Self {
            function,
            _module: module,
            name: name.to_owned(),
        })
    }

    /// Launches the kernel with the given parameters and arguments on a stream.
    ///
    /// This is the primary entry point for kernel execution. It calls
    /// `cuLaunchKernel` with the specified grid/block dimensions, shared
    /// memory, stream, and kernel arguments.
    ///
    /// The launch is asynchronous — it returns immediately and the kernel
    /// executes on the GPU. Use [`Stream::synchronize`] to wait for completion.
    ///
    /// # Type safety
    ///
    /// The `args` parameter accepts any type implementing [`KernelArgs`],
    /// including tuples of `Copy` types up to 24 elements. The caller is
    /// responsible for ensuring the argument types match the kernel signature.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](oxicuda_driver::CudaError) if the launch fails
    /// (e.g., invalid dimensions, insufficient resources, driver error).
    pub fn launch<A: KernelArgs>(
        &self,
        params: &LaunchParams,
        stream: &Stream,
        args: &A,
    ) -> CudaResult<()> {
        // Emit a tracing span for this kernel launch (no-op when the
        // `tracing` feature is disabled).
        let _span = KernelSpanGuard::enter(
            &self.name,
            (params.grid.x, params.grid.y, params.grid.z),
            (params.block.x, params.block.y, params.block.z),
        );

        let driver = try_driver()?;
        let mut param_ptrs = args.as_param_ptrs();
        oxicuda_driver::error::check(unsafe {
            (driver.cu_launch_kernel)(
                self.function.raw(),
                params.grid.x,
                params.grid.y,
                params.grid.z,
                params.block.x,
                params.block.y,
                params.block.z,
                params.shared_mem_bytes,
                stream.raw(),
                param_ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    }

    /// Returns the kernel function name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the underlying [`Function`] handle.
    ///
    /// This can be used for occupancy queries and other function-level
    /// operations provided by `oxicuda-driver`.
    #[inline]
    pub fn function(&self) -> &Function {
        &self.function
    }

    /// Returns the maximum number of active blocks per streaming multiprocessor
    /// for a given block size and dynamic shared memory.
    ///
    /// Delegates to [`Function::max_active_blocks_per_sm`].
    ///
    /// # Parameters
    ///
    /// * `block_size` — number of threads per block.
    /// * `dynamic_smem` — dynamic shared memory per block in bytes.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](oxicuda_driver::CudaError) if the query fails.
    pub fn max_active_blocks_per_sm(
        &self,
        block_size: i32,
        dynamic_smem: usize,
    ) -> CudaResult<i32> {
        self.function
            .max_active_blocks_per_sm(block_size, dynamic_smem)
    }

    /// Returns the optimal block size for this kernel and the minimum
    /// grid size to achieve maximum occupancy.
    ///
    /// Delegates to [`Function::optimal_block_size`].
    ///
    /// Returns `(min_grid_size, optimal_block_size)`.
    ///
    /// # Parameters
    ///
    /// * `dynamic_smem` — dynamic shared memory per block in bytes.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](oxicuda_driver::CudaError) if the query fails.
    pub fn optimal_block_size(&self, dynamic_smem: usize) -> CudaResult<(i32, i32)> {
        self.function.optimal_block_size(dynamic_smem)
    }
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("name", &self.name)
            .field("function", &self.function)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kernel({})", self.name)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_args_empty() {
        let args = ();
        let ptrs = args.as_param_ptrs();
        assert!(ptrs.is_empty());
    }

    #[test]
    fn single_arg_ptr_valid() {
        let args = (42u32,);
        let ptrs = args.as_param_ptrs();
        assert_eq!(ptrs.len(), 1);
        // Verify the pointer actually points to the value.
        let val_ptr = ptrs[0] as *const u32;
        assert_eq!(unsafe { *val_ptr }, 42u32);
    }

    #[test]
    fn two_args_ptr_valid() {
        let args = (10u32, 20u64);
        let ptrs = args.as_param_ptrs();
        assert_eq!(ptrs.len(), 2);
        assert_eq!(unsafe { *(ptrs[0] as *const u32) }, 10u32);
        assert_eq!(unsafe { *(ptrs[1] as *const u64) }, 20u64);
    }

    #[test]
    fn four_args_ptr_valid() {
        let args = (1u32, 2u64, 3.0f32, 4.0f64);
        let ptrs = args.as_param_ptrs();
        assert_eq!(ptrs.len(), 4);
        assert_eq!(unsafe { *(ptrs[0] as *const u32) }, 1u32);
        assert_eq!(unsafe { *(ptrs[1] as *const u64) }, 2u64);
        assert!((unsafe { *(ptrs[2] as *const f32) } - 3.0f32).abs() < f32::EPSILON);
        assert!((unsafe { *(ptrs[3] as *const f64) } - 4.0f64).abs() < f64::EPSILON);
    }

    #[test]
    fn twelve_args_count() {
        let args = (
            1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32, 12u32,
        );
        let ptrs = args.as_param_ptrs();
        assert_eq!(ptrs.len(), 12);
        for (i, ptr) in ptrs.iter().enumerate() {
            let val = unsafe { *(*ptr as *const u32) };
            assert_eq!(val, (i as u32) + 1);
        }
    }

    // ---------------------------------------------------------------------------
    // Quality gate tests (CPU-only, E2E PTX chain parameter verification)
    // ---------------------------------------------------------------------------

    #[test]
    fn launch_params_grid_calculation_e2e() {
        // Given n = 1_048_576 (1M elements) and block_size = 256,
        // grid_size_for must return exactly 4096 (ceiling division).
        let n: u32 = 1_048_576;
        let block_size: u32 = 256;
        let grid = crate::grid::grid_size_for(n, block_size);
        assert_eq!(
            grid, 4096,
            "grid_size_for(1M, 256) must be 4096, got {grid}"
        );
        // Also verify via arithmetic: 1_048_576 / 256 == 4096 exactly
        assert_eq!(
            n % block_size,
            0,
            "n must be exactly divisible by block_size"
        );
    }

    #[test]
    fn launch_params_stores_grid_and_block() {
        // LaunchParams::new(4096, 256) must record grid==4096 and block==256.
        let params = LaunchParams::new(4096u32, 256u32);
        assert_eq!(
            params.grid.x, 4096,
            "grid.x must be 4096, got {}",
            params.grid.x
        );
        assert_eq!(
            params.block.x, 256,
            "block.x must be 256, got {}",
            params.block.x
        );
        assert_eq!(params.shared_mem_bytes, 0);
        // Total threads: 4096 * 256 = 1_048_576
        assert_eq!(params.total_threads(), 1_048_576);
    }

    #[test]
    fn named_args_builder_chain() {
        // ArgBuilder::new().add("a", &1u32).add("b", &2.0f32).build() must have length 2.
        use crate::named_args::ArgBuilder;
        let a: u32 = 1;
        let b: f32 = 2.0;
        let mut builder = ArgBuilder::new();
        builder.add("a", &a).add("b", &b);
        assert_eq!(
            builder.arg_count(),
            2,
            "ArgBuilder with 2 pushes must have length 2"
        );
        let ptrs = builder.build();
        assert_eq!(ptrs.len(), 2, "build() must return 2 pointers");
    }
}
