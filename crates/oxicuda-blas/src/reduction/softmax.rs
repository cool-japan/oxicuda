//! Numerically stable row-wise softmax on device buffers.
//!
//! Uses the three-pass algorithm from [`SoftmaxTemplate`]:
//! 1. Find row maximum: `m = max(x[0..cols])`
//! 2. Exponentiate and sum: `s = sum(exp(x[i] - m))`
//! 3. Normalize: `y[i] = exp(x[i] - m) / s`
//!
//! The implementation strategy is selected based on the number of columns
//! (elements per row):
//! - `cols <= 32`: warp-level shuffle reduction (1 warp per row)
//! - `cols <= 1024`: shared-memory block reduction (1 block per row)
//! - `cols > 1024`: currently unsupported (returns an error)

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::templates::softmax::SoftmaxTemplate;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

/// Builds a softmax kernel from the PTX template.
fn build_softmax_kernel(
    handle: &BlasHandle,
    ptx_type: oxicuda_ptx::ir::PtxType,
    row_size: u32,
) -> BlasResult<(Kernel, String)> {
    let template = SoftmaxTemplate {
        precision: ptx_type,
        target: handle.sm_version(),
        row_size,
    };
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("softmax (row_size={row_size}): {e}")))?;
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| BlasError::LaunchFailed(format!("module load for softmax: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Computes row-wise softmax over a 2-D matrix stored in row-major order.
///
/// For each row `r` in `[0, rows)`:
///
/// ```text
/// m = max(input[r, 0..cols])
/// output[r, j] = exp(input[r, j] - m) / sum_j(exp(input[r, j] - m))
/// ```
///
/// The implementation uses a numerically stable algorithm that subtracts
/// the row maximum before exponentiation to prevent overflow.
///
/// # Strategy selection
///
/// | `cols`         | Strategy                      | Threads/row |
/// |----------------|-------------------------------|-------------|
/// | `<= 32`        | Warp shuffle reduction        | 32          |
/// | `33..=1024`    | Shared memory block reduction | `cols`*     |
/// | `> 1024`       | Error (not yet supported)     | N/A         |
///
/// (*) Rounded up to the nearest power of two.
///
/// # Arguments
///
/// * `handle` -- BLAS handle bound to a CUDA context and stream.
/// * `rows` -- number of rows (batch size).
/// * `cols` -- number of columns (elements per row).
/// * `input` -- device buffer containing the input matrix in row-major
///   layout, at least `rows * cols` elements.
/// * `output` -- device buffer for the result matrix, same layout, at
///   least `rows * cols` elements.
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if buffers are too small,
/// [`BlasError::InvalidDimension`] if `rows` or `cols` is zero, or
/// [`BlasError::UnsupportedOperation`] if `cols > 1024`.
pub fn softmax<T: GpuFloat>(
    handle: &BlasHandle,
    rows: u32,
    cols: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if rows == 0 || cols == 0 {
        return Err(BlasError::InvalidDimension(
            "softmax requires rows > 0 and cols > 0".to_string(),
        ));
    }
    if cols > 1024 {
        return Err(BlasError::UnsupportedOperation(format!(
            "softmax: cols={cols} exceeds the current limit of 1024; \
             multi-block softmax not yet implemented"
        )));
    }

    let total_elements = rows as usize * cols as usize;
    if input.len() < total_elements {
        return Err(BlasError::BufferTooSmall {
            expected: total_elements,
            actual: input.len(),
        });
    }
    if output.len() < total_elements {
        return Err(BlasError::BufferTooSmall {
            expected: total_elements,
            actual: output.len(),
        });
    }

    let (kernel, _) = build_softmax_kernel(handle, T::PTX_TYPE, cols)?;

    // Launch configuration depends on the strategy:
    // - Warp shuffle (cols <= 32): each warp handles one row, so we need
    //   `rows` warps total. Block size = 256 (8 warps per block).
    // - Shared memory (cols > 32): one block per row, block size = cols
    //   rounded up to power of 2.
    let (grid, block) = if cols <= 32 {
        // Warps per block = block_size / 32
        let block_size: u32 = 256;
        let warps_per_block = block_size / 32;
        let num_blocks = grid_size_for(rows, warps_per_block);
        (num_blocks, block_size)
    } else {
        // One block per row, block size = next power of two >= cols
        let block_size = cols.next_power_of_two();
        (rows, block_size)
    };

    let params = LaunchParams::new(grid, block);
    // Softmax kernel signature: (input_ptr, output_ptr, batch_size)
    let args = (input.as_device_ptr(), output.as_device_ptr(), rows);

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("softmax: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;
    use oxicuda_ptx::ir::PtxType;
    use oxicuda_ptx::templates::softmax::SoftmaxTemplate;

    #[test]
    fn ptx_template_generates_softmax_warp_f32() {
        let template = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 32,
        };
        let ptx = template
            .generate()
            .expect("warp softmax PTX should generate");
        assert!(ptx.contains("softmax_f32_r32"));
        assert!(ptx.contains("shfl.sync"));
    }

    #[test]
    fn ptx_template_generates_softmax_block_f32() {
        let template = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 128,
        };
        let ptx = template
            .generate()
            .expect("block softmax PTX should generate");
        assert!(ptx.contains("softmax_f32_r128"));
    }

    #[test]
    fn ptx_template_rejects_large_row_size() {
        let template = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 2048,
        };
        assert!(template.generate().is_err());
    }

    #[test]
    fn warp_launch_config() {
        // cols=16, rows=100 => warp strategy
        let block_size: u32 = 256;
        let warps_per_block = block_size / 32;
        let num_blocks = grid_size_for(100, warps_per_block);
        // 8 warps per block, 100 rows => ceil(100/8) = 13 blocks
        assert_eq!(num_blocks, 13);
    }

    #[test]
    fn block_launch_config() {
        // cols=100, rows=50 => block strategy, block_size = 128 (next pow2)
        let cols: u32 = 100;
        let block_size = cols.next_power_of_two();
        assert_eq!(block_size, 128);
    }

    #[test]
    fn softmax_warp_small_row() {
        let template = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 8,
        };
        let ptx = template
            .generate()
            .expect("small warp softmax should generate");
        assert!(ptx.contains("softmax_f32_r8"));
    }
}
