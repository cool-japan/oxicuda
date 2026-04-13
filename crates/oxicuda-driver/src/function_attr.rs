//! Safe wrappers for querying and configuring CUDA function attributes.
//!
//! This module extends [`Function`] with methods for inspecting kernel
//! resource usage (registers, shared memory, etc.) and tuning launch
//! parameters such as dynamic shared memory limits.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_driver::module::{Module, Function};
//! # fn main() -> Result<(), oxicuda_driver::CudaError> {
//! # let module = Module::from_ptx("...")?;
//! # let func = module.get_function("my_kernel")?;
//! let regs = func.num_registers()?;
//! let smem = func.shared_memory_bytes()?;
//! println!("kernel uses {regs} registers, {smem} bytes shared mem");
//! # Ok(())
//! # }
//! ```

use std::ffi::c_int;

use crate::error::{CudaError, CudaResult};
use crate::ffi::CUfunction_attribute;
use crate::loader::try_driver;
use crate::module::Function;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Fetch a single integer function attribute from the driver.
fn get_attribute(func: &Function, attrib: CUfunction_attribute) -> CudaResult<i32> {
    let api = try_driver()?;
    let f = api.cu_func_get_attribute.ok_or(CudaError::NotSupported)?;
    let mut value: c_int = 0;
    crate::cuda_call!(f(&mut value, attrib as i32, func.raw()))?;
    Ok(value)
}

/// Set a single integer function attribute via the driver.
fn set_attribute(func: &Function, attrib: CUfunction_attribute, value: i32) -> CudaResult<()> {
    let api = try_driver()?;
    let f = api.cu_func_set_attribute.ok_or(CudaError::NotSupported)?;
    crate::cuda_call!(f(func.raw(), attrib as i32, value))
}

// ---------------------------------------------------------------------------
// Function attribute methods
// ---------------------------------------------------------------------------

impl Function {
    /// Returns the number of registers used by each thread of this kernel.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotSupported`] if the driver lacks
    /// `cuFuncGetAttribute`, or another error on failure.
    pub fn num_registers(&self) -> CudaResult<i32> {
        get_attribute(self, CUfunction_attribute::NumRegs)
    }

    /// Returns the static shared memory used by this kernel (bytes).
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotSupported`] if the driver lacks
    /// `cuFuncGetAttribute`, or another error on failure.
    pub fn shared_memory_bytes(&self) -> CudaResult<i32> {
        get_attribute(self, CUfunction_attribute::SharedSizeBytes)
    }

    /// Returns the maximum number of threads per block for this kernel.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] on failure.
    pub fn max_threads_per_block_attr(&self) -> CudaResult<i32> {
        get_attribute(self, CUfunction_attribute::MaxThreadsPerBlock)
    }

    /// Returns the local memory used by each thread of this kernel (bytes).
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] on failure.
    pub fn local_memory_bytes(&self) -> CudaResult<i32> {
        get_attribute(self, CUfunction_attribute::LocalSizeBytes)
    }

    /// Returns the PTX virtual architecture version for this kernel.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] on failure.
    pub fn ptx_version(&self) -> CudaResult<i32> {
        get_attribute(self, CUfunction_attribute::PtxVersion)
    }

    /// Returns the binary (SASS) architecture version for this kernel.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] on failure.
    pub fn binary_version(&self) -> CudaResult<i32> {
        get_attribute(self, CUfunction_attribute::BinaryVersion)
    }

    /// Returns the maximum dynamic shared memory size (bytes) for this kernel.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] on failure.
    pub fn max_dynamic_shared_memory(&self) -> CudaResult<i32> {
        get_attribute(self, CUfunction_attribute::MaxDynamicSharedSizeBytes)
    }

    /// Sets the maximum dynamic shared memory size (bytes) for this kernel.
    ///
    /// This must be called before launching the kernel if you need more
    /// dynamic shared memory than the default limit.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotSupported`] if the driver lacks
    /// `cuFuncSetAttribute`, or another error on failure.
    pub fn set_max_dynamic_shared_memory(&self, bytes: i32) -> CudaResult<()> {
        set_attribute(self, CUfunction_attribute::MaxDynamicSharedSizeBytes, bytes)
    }

    /// Sets the preferred shared memory carve-out (percentage 0-100).
    ///
    /// A value of 0 means use the device default. Values between 1 and 100
    /// indicate the desired percentage of L1 cache to use as shared memory.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotSupported`] if the driver lacks
    /// `cuFuncSetAttribute`, or another error on failure.
    pub fn set_preferred_shared_memory_carveout(&self, percent: i32) -> CudaResult<()> {
        set_attribute(
            self,
            CUfunction_attribute::PreferredSharedMemoryCarveout,
            percent,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn function_attribute_enum_values() {
        use crate::ffi::CUfunction_attribute;
        assert_eq!(CUfunction_attribute::MaxThreadsPerBlock as i32, 0);
        assert_eq!(CUfunction_attribute::NumRegs as i32, 4);
        assert_eq!(CUfunction_attribute::PtxVersion as i32, 5);
        assert_eq!(CUfunction_attribute::MaxDynamicSharedSizeBytes as i32, 8);
    }
}
