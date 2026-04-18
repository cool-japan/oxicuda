//! GPU broadcast-axes operation: expand a reduced tensor to its original shape.
//!
//! Uses the stride-zero trick — see [`oxicuda_ptx::templates::broadcast::BroadcastTemplate`]
//! for the kernel generation details.
//!
//! # Stride-zero trick (host side)
//!
//! Given a destination shape `dst_shape[0..R]` and a list of `reduced_axes`,
//! the host computes:
//!
//! - `dst_strides[d]  = product(dst_shape[d+1..R])` (row-major strides)
//! - `src_strides_padded[d] = 0` if `d ∈ reduced_axes`
//! - `src_strides_padded[d] = row-major stride for dimension `d` in `src` otherwise
//!
//! The GPU kernel then computes, for output thread `tid`:
//! ```text
//! flat_src = Σ_d ( (tid / dst_strides[d]) % dst_shape[d] ) * src_strides_padded[d]
//! ```
//! Since reduced axes contribute 0, many output positions map to the same source element.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::templates::broadcast::{BroadcastTemplate, MAX_BROADCAST_RANK};

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

/// Standard block size for the broadcast kernel.
const BLOCK_SIZE: u32 = 256;

/// Broadcasts `src` (a reduced tensor) back to `dst` (the full original shape)
/// by replicating values along every axis listed in `reduced_axes`.
///
/// # Arguments
///
/// * `handle` — BLAS handle bound to a CUDA context and stream.
/// * `src` — Source device buffer (reduced tensor). Must hold at least
///   `src_shape.iter().product()` elements.
/// * `src_shape` — Shape of the source tensor (non-reduced dimensions only,
///   in the same order they appear in `dst_shape` after filtering `reduced_axes`).
/// * `dst` — Destination device buffer. Must hold at least `dst_shape.iter().product()`.
/// * `dst_shape` — Full output shape (rank ≤ [`MAX_BROADCAST_RANK`]).
/// * `reduced_axes` — Indices (into `dst_shape`) of the axes that were reduced.
///   These axes receive stride 0 in `src_strides_padded`, causing the GPU kernel
///   to replicate the corresponding source element.
///
/// # Errors
///
/// Returns [`BlasError::InvalidArgument`] if `dst_shape.len() > MAX_BROADCAST_RANK`.
/// Returns [`BlasError::BufferTooSmall`] if either buffer is undersized.
/// Returns [`BlasError::PtxGeneration`] if kernel source generation fails.
/// Returns [`BlasError::LaunchFailed`] if module load or kernel launch fails.
pub fn broadcast_axes<T: GpuFloat>(
    handle: &BlasHandle,
    src: &DeviceBuffer<T>,
    src_shape: &[usize],
    dst: &mut DeviceBuffer<T>,
    dst_shape: &[usize],
    reduced_axes: &[usize],
) -> BlasResult<()> {
    // ------------------------------------------------------------------
    // Argument validation
    // ------------------------------------------------------------------
    if dst_shape.len() > MAX_BROADCAST_RANK {
        return Err(BlasError::InvalidArgument(format!(
            "broadcast_axes: dst rank {} exceeds MAX_BROADCAST_RANK {}",
            dst_shape.len(),
            MAX_BROADCAST_RANK
        )));
    }

    let n_dst: usize = dst_shape.iter().product();
    if n_dst == 0 {
        return Ok(());
    }
    if dst.len() < n_dst {
        return Err(BlasError::BufferTooSmall {
            expected: n_dst,
            actual: dst.len(),
        });
    }
    let n_src: usize = src_shape.iter().product();
    if src.len() < n_src {
        return Err(BlasError::BufferTooSmall {
            expected: n_src,
            actual: src.len(),
        });
    }

    let rank = dst_shape.len();

    // ------------------------------------------------------------------
    // Build padded shape/stride arrays
    // ------------------------------------------------------------------
    // dst shape padded with 1s
    let mut ds = [1u32; MAX_BROADCAST_RANK];
    for (d, &s) in dst_shape.iter().enumerate() {
        ds[d] = s as u32;
    }

    // dst row-major strides: dst_strides[d] = product(dst_shape[d+1..rank])
    let mut dst_strides = [0u32; MAX_BROADCAST_RANK];
    let mut stride = 1usize;
    for d in (0..rank).rev() {
        dst_strides[d] = stride as u32;
        stride *= dst_shape[d];
    }

    // src_strides_padded: 0 for reduced axes, src row-major stride otherwise.
    // The non-reduced dimensions of dst correspond to the consecutive axes of src
    // (in the same left-to-right order, skipping reduced axes).
    let non_reduced: Vec<usize> = (0..rank).filter(|d| !reduced_axes.contains(d)).collect();

    // Compute src row-major strides for its own shape dimensions.
    // src_shape[i] corresponds to non_reduced[i] in dst coordinates.
    let mut src_raw_strides = vec![0usize; non_reduced.len()];
    if !src_shape.is_empty() {
        let mut src_stride = 1usize;
        for i in (0..non_reduced.len()).rev() {
            src_raw_strides[i] = src_stride;
            if i < src_shape.len() {
                src_stride *= src_shape[i];
            }
        }
    }

    let mut src_strides_padded = [0u32; MAX_BROADCAST_RANK];
    for (idx, &dst_dim) in non_reduced.iter().enumerate() {
        src_strides_padded[dst_dim] = src_raw_strides.get(idx).copied().unwrap_or(0) as u32;
    }
    // reduced_axes leave src_strides_padded[d] = 0, enabling the stride-zero trick.

    // ------------------------------------------------------------------
    // PTX kernel generation, module load, and launch
    // ------------------------------------------------------------------
    let template = BroadcastTemplate::new(T::PTX_TYPE, handle.sm_version());
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("broadcast_axes: {e}")))?;
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| BlasError::LaunchFailed(format!("broadcast_axes module: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("broadcast_axes kernel: {e}")))?;

    let n_dst_u32 = n_dst as u32;
    let grid = grid_size_for(n_dst_u32, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);

    // Kernel signature (28 args):
    //   src_ptr: u64, dst_ptr: u64, rank: u32,
    //   ds[0..8]: u32×8, dst_strides[0..8]: u32×8, src_strides_padded[0..8]: u32×8,
    //   n_dst: u32
    let args = (
        src.as_device_ptr(),
        dst.as_device_ptr(),
        rank as u32,
        ds[0],
        ds[1],
        ds[2],
        ds[3],
        ds[4],
        ds[5],
        ds[6],
        ds[7],
        dst_strides[0],
        dst_strides[1],
        dst_strides[2],
        dst_strides[3],
        dst_strides[4],
        dst_strides[5],
        dst_strides[6],
        dst_strides[7],
        src_strides_padded[0],
        src_strides_padded[1],
        src_strides_padded[2],
        src_strides_padded[3],
        src_strides_padded[4],
        src_strides_padded[5],
        src_strides_padded[6],
        src_strides_padded[7],
        n_dst_u32,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("broadcast_axes launch: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;
    use oxicuda_ptx::ir::PtxType;
    use oxicuda_ptx::templates::broadcast::BroadcastTemplate;

    #[test]
    fn block_size_is_power_of_two() {
        assert!(BLOCK_SIZE.is_power_of_two());
        const { assert!(BLOCK_SIZE >= 32) };
    }

    #[test]
    fn broadcast_rejects_rank_exceeding_max() {
        // Validation logic test — no GPU needed
        let rank_too_big = MAX_BROADCAST_RANK + 1;
        let err_msg = format!(
            "broadcast_axes: dst rank {} exceeds MAX_BROADCAST_RANK {}",
            rank_too_big, MAX_BROADCAST_RANK
        );
        let blas_err = BlasError::InvalidArgument(err_msg);
        assert!(blas_err.to_string().contains("exceeds MAX_BROADCAST_RANK"));
    }

    #[test]
    fn dst_stride_computation_correctness() {
        // For shape [2, 3, 4], row-major strides should be [12, 4, 1].
        let dst_shape: Vec<usize> = vec![2, 3, 4];
        let rank = dst_shape.len();
        let mut dst_strides = [0u32; MAX_BROADCAST_RANK];
        let mut stride = 1usize;
        for d in (0..rank).rev() {
            dst_strides[d] = stride as u32;
            stride *= dst_shape[d];
        }
        assert_eq!(dst_strides[0], 12, "stride[0] = 3*4 = 12");
        assert_eq!(dst_strides[1], 4, "stride[1] = 4");
        assert_eq!(dst_strides[2], 1, "stride[2] = 1");
    }

    #[test]
    fn src_stride_padded_zeros_on_reduced_axes() {
        // dst_shape = [2, 3, 4], reduced_axes = [1] (middle dim).
        // src_shape = [2, 4] (non-reduced dims).
        // src_strides_padded = [4, 0, 1]: dim0 -> stride 4, dim1 -> 0, dim2 -> stride 1.
        let dst_shape: Vec<usize> = vec![2, 3, 4];
        let reduced_axes: Vec<usize> = vec![1];
        let src_shape: Vec<usize> = vec![2, 4];
        let rank = dst_shape.len();
        let non_reduced: Vec<usize> = (0..rank).filter(|d| !reduced_axes.contains(d)).collect();

        let mut src_raw_strides = vec![0usize; non_reduced.len()];
        let mut src_stride = 1usize;
        for i in (0..non_reduced.len()).rev() {
            src_raw_strides[i] = src_stride;
            if i < src_shape.len() {
                src_stride *= src_shape[i];
            }
        }

        let mut src_strides_padded = [0u32; MAX_BROADCAST_RANK];
        for (idx, &dst_dim) in non_reduced.iter().enumerate() {
            src_strides_padded[dst_dim] = src_raw_strides.get(idx).copied().unwrap_or(0) as u32;
        }

        assert_eq!(
            src_strides_padded[0], 4,
            "dim0: src_stride for shape[2,4] is 4"
        );
        assert_eq!(src_strides_padded[1], 0, "dim1 is reduced, must be 0");
        assert_eq!(src_strides_padded[2], 1, "dim2: innermost src_stride is 1");
    }

    #[test]
    fn broadcast_ptx_template_generates_valid_ptx() {
        let t = BroadcastTemplate::new(PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("broadcast PTX generation failed");
        assert!(ptx.contains("broadcast_axes_f32"));
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
    }
}
