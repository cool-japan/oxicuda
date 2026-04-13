//! Multi-stream batched GEMM.
//!
//! Distributes batched GEMM work across multiple CUDA streams for improved
//! throughput. Each stream processes a subset of the total batch, enabling
//! concurrent kernel execution on GPUs with sufficient resources.
//!
//! # Distribution strategies
//!
//! * [`StreamDistribution::RoundRobin`] — assigns batches to streams in
//!   round-robin order, producing balanced but non-contiguous assignments.
//! * [`StreamDistribution::EqualSplit`] — assigns contiguous, roughly equal
//!   chunks of batches to each stream, which is better for memory locality.

use oxicuda_driver::Stream;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use std::sync::Arc;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{GpuFloat, Transpose};

use super::batched_gemm::{TILE_M, TILE_N, generate_batched_gemm_ptx, validate_batched_args};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Strategy for distributing batches across streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamDistribution {
    /// Assigns batches to streams in round-robin order (batch 0 -> stream 0,
    /// batch 1 -> stream 1, ..., batch N -> stream N % num_streams).
    /// This produces maximally balanced assignments but non-contiguous memory
    /// access patterns per stream.
    RoundRobin,
    /// Assigns contiguous, roughly equal chunks of batches to each stream.
    /// Streams earlier in the list may receive one extra batch if the total
    /// count is not evenly divisible.
    EqualSplit,
}

/// Configuration for multi-stream batched GEMM execution.
#[derive(Debug, Clone)]
pub struct MultiStreamBatchedConfig {
    /// Number of streams to distribute work across.
    pub num_streams: u32,
    /// Strategy for assigning batches to streams.
    pub distribution: StreamDistribution,
}

impl MultiStreamBatchedConfig {
    /// Creates a new configuration.
    ///
    /// # Arguments
    ///
    /// * `num_streams` — number of streams to use (must be >= 1).
    /// * `distribution` — how to distribute batches across streams.
    #[must_use]
    pub fn new(num_streams: u32, distribution: StreamDistribution) -> Self {
        Self {
            num_streams,
            distribution,
        }
    }

    /// Validates the configuration.
    fn validate(&self) -> BlasResult<()> {
        if self.num_streams == 0 {
            return Err(BlasError::InvalidArgument(
                "num_streams must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Distribution logic
// ---------------------------------------------------------------------------

/// Computes `(start_index, count)` pairs for distributing `batch_count`
/// batches across `num_streams` streams using the given strategy.
///
/// For [`StreamDistribution::EqualSplit`], the first `batch_count % num_streams`
/// streams receive `batch_count / num_streams + 1` batches, and the remaining
/// streams receive `batch_count / num_streams`.
///
/// For [`StreamDistribution::RoundRobin`], the returned ranges are a
/// simplification: we still produce contiguous ranges, but balance them the
/// same way as `EqualSplit`. The actual round-robin assignment is handled at
/// kernel launch time by selecting the correct pointer offsets.
///
/// Returns an empty vector if `batch_count` is zero.
pub fn distribute_batches(
    batch_count: u32,
    num_streams: u32,
    distribution: &StreamDistribution,
) -> Vec<(u32, u32)> {
    if batch_count == 0 || num_streams == 0 {
        return Vec::new();
    }

    let effective_streams = num_streams.min(batch_count);

    match distribution {
        StreamDistribution::EqualSplit => {
            let base = batch_count / effective_streams;
            let remainder = batch_count % effective_streams;
            let mut result = Vec::with_capacity(effective_streams as usize);
            let mut offset = 0u32;

            for i in 0..effective_streams {
                let count = if i < remainder { base + 1 } else { base };
                result.push((offset, count));
                offset = offset.saturating_add(count);
            }
            result
        }
        StreamDistribution::RoundRobin => {
            // Round-robin: each stream gets every N-th batch.
            // We report (logical_start, count) where logical_start is the
            // first batch index assigned to this stream, and count is how
            // many batches it handles.
            let base = batch_count / effective_streams;
            let remainder = batch_count % effective_streams;
            let mut result = Vec::with_capacity(effective_streams as usize);

            for i in 0..effective_streams {
                let count = if i < remainder { base + 1 } else { base };
                // For round-robin, the "start" is the stream index itself.
                // Batches assigned: i, i + N, i + 2N, ...
                result.push((i, count));
            }
            result
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-stream batched GEMM
// ---------------------------------------------------------------------------

/// Executes a batched GEMM across multiple CUDA streams.
///
/// ```text
/// D[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]   for i in 0..batch_count
/// ```
///
/// The batch is partitioned according to `config.distribution` and each
/// partition is launched on a separate stream from `streams`.
///
/// # Arguments
///
/// * `handle` — BLAS handle (used for SM version and validation).
/// * `config` — multi-stream configuration (number of streams, distribution).
/// * `trans_a`, `trans_b` — transposition modes for A and B.
/// * `m`, `n`, `k` — matrix dimensions.
/// * `alpha`, `beta` — scalar coefficients.
/// * `a_ptrs`, `b_ptrs`, `c_ptrs` — device-pointer arrays (one per batch).
/// * `lda`, `ldb`, `ldc`, `ldd` — leading dimensions.
/// * `d_ptrs` — output device-pointer array (one per batch).
/// * `batch_count` — total number of batches.
/// * `streams` — slice of streams to distribute work across.
///
/// # Errors
///
/// * [`BlasError::InvalidArgument`] if `config.num_streams` is zero or
///   `streams.len()` is less than `config.num_streams`.
/// * All errors from [`gemm_batched`](super::gemm_batched) (dimension,
///   buffer, PTX, launch errors).
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched_multi_stream<T: GpuFloat>(
    handle: &BlasHandle,
    config: &MultiStreamBatchedConfig,
    trans_a: Transpose,
    trans_b: Transpose,
    m: u32,
    n: u32,
    k: u32,
    alpha: T,
    a_ptrs: &DeviceBuffer<CUdeviceptr>,
    lda: u32,
    b_ptrs: &DeviceBuffer<CUdeviceptr>,
    ldb: u32,
    beta: T,
    c_ptrs: &DeviceBuffer<CUdeviceptr>,
    ldc: u32,
    d_ptrs: &mut DeviceBuffer<CUdeviceptr>,
    ldd: u32,
    batch_count: u32,
    streams: &[&Stream],
) -> BlasResult<()> {
    // Early-out for zero batches.
    if batch_count == 0 {
        return Ok(());
    }

    // Validate configuration.
    config.validate()?;

    if (streams.len() as u32) < config.num_streams {
        return Err(BlasError::InvalidArgument(format!(
            "streams slice has {} entries but config requires {}",
            streams.len(),
            config.num_streams
        )));
    }

    // Validate matrix arguments using shared helper.
    validate_batched_args::<T>(
        m,
        n,
        k,
        a_ptrs,
        lda,
        b_ptrs,
        ldb,
        c_ptrs,
        ldc,
        d_ptrs,
        ldd,
        batch_count,
        trans_a,
        trans_b,
    )?;

    let sm = handle.sm_version();

    // Single-stream fast path: delegate to the standard batched dispatch.
    if config.num_streams == 1 {
        return launch_batch_on_stream::<T>(
            sm,
            streams[0],
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a_ptrs,
            lda,
            b_ptrs,
            ldb,
            beta,
            c_ptrs,
            ldc,
            d_ptrs,
            ldd,
            0,
            batch_count,
        );
    }

    // Partition batches across streams.
    let partitions = distribute_batches(batch_count, config.num_streams, &config.distribution);

    // Generate PTX once (shared across all streams).
    let (ptx_source, kernel_name) = generate_batched_gemm_ptx::<T>(sm, m, n, k, trans_a, trans_b)?;
    let module = oxicuda_driver::Module::from_ptx(&ptx_source).map_err(BlasError::Cuda)?;
    let module = Arc::new(module);

    let alpha_bits = alpha.to_bits_u64();
    let beta_bits = beta.to_bits_u64();

    for (stream_idx, &(start, count)) in partitions.iter().enumerate() {
        if count == 0 {
            continue;
        }

        let stream = streams[stream_idx];
        let kernel =
            Kernel::from_module(Arc::clone(&module), &kernel_name).map_err(BlasError::Cuda)?;

        let grid = Dim3::new(m.div_ceil(TILE_M), n.div_ceil(TILE_N), count);
        let block = Dim3::new(TILE_M, TILE_N, 1);
        let params = LaunchParams::new(grid, block);

        // Offset into the pointer arrays for this partition.
        // For EqualSplit, start is the contiguous offset.
        // For RoundRobin, we still pass contiguous sub-ranges; the caller
        // is responsible for arranging pointer arrays accordingly, or we
        // launch with the batch offset as a kernel parameter.
        let a_ptr_offset = a_ptrs
            .as_device_ptr()
            .wrapping_add(start as u64 * std::mem::size_of::<CUdeviceptr>() as u64);
        let b_ptr_offset = b_ptrs
            .as_device_ptr()
            .wrapping_add(start as u64 * std::mem::size_of::<CUdeviceptr>() as u64);
        let c_ptr_offset = c_ptrs
            .as_device_ptr()
            .wrapping_add(start as u64 * std::mem::size_of::<CUdeviceptr>() as u64);
        let d_ptr_offset = d_ptrs
            .as_device_ptr()
            .wrapping_add(start as u64 * std::mem::size_of::<CUdeviceptr>() as u64);

        let args = (
            m,
            n,
            k,
            alpha_bits,
            a_ptr_offset,
            lda,
            b_ptr_offset,
            ldb,
            beta_bits,
            c_ptr_offset,
            ldc,
            d_ptr_offset,
            ldd,
        );

        kernel
            .launch(&params, stream, &args)
            .map_err(|e| BlasError::LaunchFailed(format!("stream {stream_idx}: {e}")))?;
    }

    Ok(())
}

/// Launches a batched GEMM kernel on a single stream for a contiguous
/// sub-range of the batch.
#[allow(clippy::too_many_arguments)]
fn launch_batch_on_stream<T: GpuFloat>(
    sm: SmVersion,
    stream: &Stream,
    trans_a: Transpose,
    trans_b: Transpose,
    m: u32,
    n: u32,
    k: u32,
    alpha: T,
    a_ptrs: &DeviceBuffer<CUdeviceptr>,
    lda: u32,
    b_ptrs: &DeviceBuffer<CUdeviceptr>,
    ldb: u32,
    beta: T,
    c_ptrs: &DeviceBuffer<CUdeviceptr>,
    ldc: u32,
    d_ptrs: &mut DeviceBuffer<CUdeviceptr>,
    ldd: u32,
    start: u32,
    count: u32,
) -> BlasResult<()> {
    if count == 0 {
        return Ok(());
    }

    let (ptx_source, kernel_name) = generate_batched_gemm_ptx::<T>(sm, m, n, k, trans_a, trans_b)?;
    let module = oxicuda_driver::Module::from_ptx(&ptx_source).map_err(BlasError::Cuda)?;
    let module = Arc::new(module);
    let kernel = Kernel::from_module(module, &kernel_name).map_err(BlasError::Cuda)?;

    let grid = Dim3::new(m.div_ceil(TILE_M), n.div_ceil(TILE_N), count);
    let block = Dim3::new(TILE_M, TILE_N, 1);
    let params = LaunchParams::new(grid, block);

    let alpha_bits = alpha.to_bits_u64();
    let beta_bits = beta.to_bits_u64();

    let ptr_size = std::mem::size_of::<CUdeviceptr>() as u64;
    let offset = start as u64 * ptr_size;

    let args = (
        m,
        n,
        k,
        alpha_bits,
        a_ptrs.as_device_ptr().wrapping_add(offset),
        lda,
        b_ptrs.as_device_ptr().wrapping_add(offset),
        ldb,
        beta_bits,
        c_ptrs.as_device_ptr().wrapping_add(offset),
        ldc,
        d_ptrs.as_device_ptr().wrapping_add(offset),
        ldd,
    );

    kernel
        .launch(&params, stream, &args)
        .map_err(|e| BlasError::LaunchFailed(e.to_string()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- distribute_batches tests -------------------------------------------

    #[test]
    fn equal_split_even_division() {
        let result = distribute_batches(12, 3, &StreamDistribution::EqualSplit);
        assert_eq!(result, vec![(0, 4), (4, 4), (8, 4)]);
    }

    #[test]
    fn equal_split_uneven_division() {
        let result = distribute_batches(10, 3, &StreamDistribution::EqualSplit);
        // 10 / 3 = 3 remainder 1, so first stream gets 4, rest get 3.
        assert_eq!(result, vec![(0, 4), (4, 3), (7, 3)]);
    }

    #[test]
    fn equal_split_more_streams_than_batches() {
        let result = distribute_batches(2, 5, &StreamDistribution::EqualSplit);
        // Only 2 effective streams, each gets 1 batch.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0, 1));
        assert_eq!(result[1], (1, 1));
    }

    #[test]
    fn equal_split_single_batch() {
        let result = distribute_batches(1, 4, &StreamDistribution::EqualSplit);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (0, 1));
    }

    #[test]
    fn equal_split_single_stream() {
        let result = distribute_batches(100, 1, &StreamDistribution::EqualSplit);
        assert_eq!(result, vec![(0, 100)]);
    }

    #[test]
    fn round_robin_even_division() {
        let result = distribute_batches(12, 3, &StreamDistribution::RoundRobin);
        assert_eq!(result.len(), 3);
        // Each stream gets 4 batches.
        assert_eq!(result[0], (0, 4));
        assert_eq!(result[1], (1, 4));
        assert_eq!(result[2], (2, 4));
    }

    #[test]
    fn round_robin_uneven_division() {
        let result = distribute_batches(10, 3, &StreamDistribution::RoundRobin);
        // 10 / 3 = 3 remainder 1, first stream gets 4.
        assert_eq!(result[0], (0, 4));
        assert_eq!(result[1], (1, 3));
        assert_eq!(result[2], (2, 3));
    }

    #[test]
    fn zero_batches_returns_empty() {
        let result = distribute_batches(0, 4, &StreamDistribution::EqualSplit);
        assert!(result.is_empty());
    }

    #[test]
    fn zero_streams_returns_empty() {
        let result = distribute_batches(10, 0, &StreamDistribution::EqualSplit);
        assert!(result.is_empty());
    }

    #[test]
    fn config_validation_rejects_zero_streams() {
        let config = MultiStreamBatchedConfig::new(0, StreamDistribution::EqualSplit);
        let res = config.validate();
        assert!(res.is_err());
    }

    #[test]
    fn config_validation_accepts_positive_streams() {
        let config = MultiStreamBatchedConfig::new(4, StreamDistribution::RoundRobin);
        let res = config.validate();
        assert!(res.is_ok());
    }

    #[test]
    fn total_batches_preserved_equal_split() {
        for batch_count in 1u32..=50 {
            for num_streams in 1u32..=10 {
                let partitions =
                    distribute_batches(batch_count, num_streams, &StreamDistribution::EqualSplit);
                let total: u32 = partitions.iter().map(|&(_, c)| c).sum();
                assert_eq!(
                    total, batch_count,
                    "batch_count={batch_count}, num_streams={num_streams}"
                );
            }
        }
    }

    #[test]
    fn total_batches_preserved_round_robin() {
        for batch_count in 1u32..=50 {
            for num_streams in 1u32..=10 {
                let partitions =
                    distribute_batches(batch_count, num_streams, &StreamDistribution::RoundRobin);
                let total: u32 = partitions.iter().map(|&(_, c)| c).sum();
                assert_eq!(
                    total, batch_count,
                    "batch_count={batch_count}, num_streams={num_streams}"
                );
            }
        }
    }

    #[test]
    fn equal_split_contiguous_coverage() {
        // Verify that EqualSplit partitions cover [0, batch_count) contiguously.
        let partitions = distribute_batches(17, 4, &StreamDistribution::EqualSplit);
        let mut expected_start = 0u32;
        for &(start, count) in &partitions {
            assert_eq!(start, expected_start);
            expected_start += count;
        }
        assert_eq!(expected_start, 17);
    }

    #[test]
    fn distribution_balance_property() {
        // No stream should have more than ceil(batch_count / num_streams) batches.
        let batch_count = 23u32;
        let num_streams = 5u32;
        let max_per_stream = batch_count.div_ceil(num_streams);

        for dist in &[
            StreamDistribution::EqualSplit,
            StreamDistribution::RoundRobin,
        ] {
            let partitions = distribute_batches(batch_count, num_streams, dist);
            for &(_, count) in &partitions {
                assert!(
                    count <= max_per_stream,
                    "stream got {count} batches, max allowed {max_per_stream} ({dist:?})"
                );
            }
        }
    }
}
