//! GPU occupancy queries for performance optimisation.
//!
//! Occupancy measures how effectively GPU resources (warps, registers,
//! shared memory) are utilised. These queries help select launch
//! configurations that maximise hardware utilisation.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_driver::module::Module;
//! # fn main() -> Result<(), oxicuda_driver::error::CudaError> {
//! # let module: Module = unimplemented!();
//! let func = module.get_function("my_kernel")?;
//!
//! // Query the optimal block size for maximum occupancy.
//! let (min_grid_size, optimal_block_size) = func.optimal_block_size(0)?;
//! println!("optimal: grid >= {min_grid_size}, block = {optimal_block_size}");
//!
//! // Query active blocks per SM for a specific block size.
//! let active = func.max_active_blocks_per_sm(256, 0)?;
//! println!("active blocks per SM with 256 threads: {active}");
//! # Ok(())
//! # }
//! ```

use crate::error::CudaResult;
use crate::loader::try_driver;
use crate::module::Function;

impl Function {
    /// Returns the maximum number of active blocks per streaming
    /// multiprocessor for a given block size and dynamic shared memory.
    ///
    /// This is useful for evaluating different block sizes to find
    /// the configuration that achieves the highest occupancy.
    ///
    /// # Parameters
    ///
    /// * `block_size` — number of threads per block.
    /// * `dynamic_smem` — dynamic shared memory per block in bytes
    ///   (set to `0` if the kernel does not use dynamic shared memory).
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the function
    /// handle is invalid or the driver call fails.
    pub fn max_active_blocks_per_sm(
        &self,
        block_size: i32,
        dynamic_smem: usize,
    ) -> CudaResult<i32> {
        let api = try_driver()?;
        let mut num_blocks: i32 = 0;
        crate::cuda_call!((api.cu_occupancy_max_active_blocks_per_multiprocessor)(
            &mut num_blocks,
            self.raw(),
            block_size,
            dynamic_smem,
        ))?;
        Ok(num_blocks)
    }

    /// Suggests an optimal launch configuration that maximises
    /// multiprocessor occupancy.
    ///
    /// Returns `(min_grid_size, optimal_block_size)` where:
    ///
    /// * `min_grid_size` — the minimum number of blocks needed to
    ///   achieve maximum occupancy across all SMs.
    /// * `optimal_block_size` — the block size (number of threads)
    ///   that achieves maximum occupancy.
    ///
    /// # Parameters
    ///
    /// * `dynamic_smem` — dynamic shared memory per block in bytes
    ///   (set to `0` if the kernel does not use dynamic shared memory).
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the function
    /// handle is invalid or the driver call fails.
    pub fn optimal_block_size(&self, dynamic_smem: usize) -> CudaResult<(i32, i32)> {
        let api = try_driver()?;
        let mut min_grid_size: i32 = 0;
        let mut block_size: i32 = 0;
        crate::cuda_call!((api.cu_occupancy_max_potential_block_size)(
            &mut min_grid_size,
            &mut block_size,
            self.raw(),
            None, // no dynamic smem callback
            dynamic_smem,
            0, // no block size limit
        ))?;
        Ok((min_grid_size, block_size))
    }
}
