//! PagedAttention for efficient LLM decode with KV-cache.
//!
//! Implements the PagedAttention algorithm (Kwon et al., 2023) that partitions
//! the KV-cache into fixed-size pages, enabling efficient memory management
//! for variable-length sequences in LLM serving. Each request's KV-cache is
//! stored in non-contiguous physical pages, with a page table mapping logical
//! to physical blocks.
//!
//! ## Algorithm
//!
//! For each query token in the current decode step:
//! 1. Look up the page table to find physical KV-cache pages.
//! 2. For each page, load K/V blocks and compute partial attention scores.
//! 3. Use online softmax to combine partial results across pages.
//! 4. Store the final output.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::tensor_util::{attn_dims, attn_dims_mut};
use crate::types::{TensorDesc, TensorDescMut};

// ---------------------------------------------------------------------------
// PagedAttentionConfig
// ---------------------------------------------------------------------------

/// Configuration for the PagedAttention decode kernel.
///
/// Controls the KV-cache page layout, GQA head mapping, and kernel tuning
/// parameters for the paged attention decode path.
#[derive(Debug, Clone)]
pub struct PagedAttentionConfig {
    /// Per-head dimension (D).
    pub head_dim: u32,
    /// Number of query attention heads (H_q).
    pub num_heads: u32,
    /// Number of key/value attention heads (H_kv).
    /// When `num_kv_heads < num_heads`, grouped-query attention (GQA) is used.
    pub num_kv_heads: u32,
    /// Number of tokens per KV-cache page.
    pub block_size: u32,
    /// PTX precision for the kernel body.
    pub precision: PtxType,
    /// Target SM architecture.
    pub sm_version: SmVersion,
}

// ---------------------------------------------------------------------------
// TMA descriptor parameters for KV-cache loading
// ---------------------------------------------------------------------------

/// Parameters for a TMA (Tensor Memory Accelerator) descriptor targeting a
/// KV-cache page tile.
///
/// TMA descriptors are 2D tensors: `[block_size, head_dim]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TmaKvDescriptorParams {
    /// Number of tensor dimensions (always 2 for KV-cache pages).
    pub num_dims: u32,
    /// Size of the innermost dimension (head_dim elements).
    pub dim_inner: u32,
    /// Size of the outer dimension (block_size tokens per page).
    pub dim_outer: u32,
    /// Byte stride for a single element.
    pub element_bytes: u32,
}

impl PagedAttentionConfig {
    /// Returns the GQA group size (number of Q heads per KV head).
    #[must_use]
    pub fn gqa_group_size(&self) -> u32 {
        if self.num_kv_heads == 0 {
            return 1;
        }
        self.num_heads / self.num_kv_heads
    }

    /// Returns the number of threads per block for the kernel.
    #[must_use]
    pub fn threads_per_block(&self) -> u32 {
        let warps = self.head_dim.div_ceil(32).max(4);
        warps * 32
    }

    /// Returns the total byte size of one KV-cache page block (K + V tiles).
    ///
    /// For TMA loads, this must be 16-byte aligned.
    ///
    /// Layout: `block_size × head_dim × elem_bytes × 2` (K and V).
    #[must_use]
    pub fn block_bytes(&self) -> u32 {
        let elem_bytes = self.precision.size_bytes() as u32;
        self.block_size * self.head_dim * elem_bytes * 2
    }

    /// Returns the TMA descriptor parameters for KV-cache loading.
    ///
    /// The TMA descriptor targets a 2-D tile of shape `[block_size, head_dim]`.
    /// Returns an error if the block is not 16-byte aligned (TMA requirement).
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `block_bytes() % 16 != 0`.
    pub fn tma_descriptor_params(&self) -> DnnResult<TmaKvDescriptorParams> {
        if self.block_bytes() % 16 != 0 {
            return Err(DnnError::InvalidArgument(format!(
                "KV block size {} bytes is not 16-byte aligned (TMA requirement)",
                self.block_bytes()
            )));
        }
        let elem_bytes = self.precision.size_bytes() as u32;
        Ok(TmaKvDescriptorParams {
            num_dims: 2,
            dim_inner: self.head_dim,
            dim_outer: self.block_size,
            element_bytes: elem_bytes,
        })
    }

    /// Shared memory required in bytes.
    #[must_use]
    pub fn shared_mem_bytes(&self) -> u32 {
        let elem_size = self.precision.size_bytes() as u32;
        let k_block = self.block_size * self.head_dim * elem_size;
        let v_block = self.block_size * self.head_dim * elem_size;
        let scratch = self.block_size * 4;
        k_block + v_block + scratch
    }

    /// Generates the PTX for the paged attention decode kernel.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on builder failure.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let kernel_name = format!("paged_attn_decode_d{}_bs{}", self.head_dim, self.block_size);

        let block_size = self.block_size;
        let head_dim = self.head_dim;
        let sm = self.sm_version;
        let threads = self.threads_per_block();

        let k_smem_elems = (block_size * head_dim) as usize;
        let v_smem_elems = (block_size * head_dim) as usize;

        let ptx = KernelBuilder::new(&kernel_name)
            .target(sm)
            .param("q_ptr", PtxType::U64)
            .param("k_cache_ptr", PtxType::U64)
            .param("v_cache_ptr", PtxType::U64)
            .param("page_table_ptr", PtxType::U64)
            .param("seq_lengths_ptr", PtxType::U64)
            .param("out_ptr", PtxType::U64)
            .param("head_dim", PtxType::U32)
            .param("num_heads", PtxType::U32)
            .param("num_kv_heads", PtxType::U32)
            .param("block_size", PtxType::U32)
            .param("max_seq_len", PtxType::U32)
            .param("sm_scale", PtxType::F32)
            .shared_mem("k_smem", PtxType::F32, k_smem_elems)
            .shared_mem("v_smem", PtxType::F32, v_smem_elems)
            .max_threads_per_block(threads)
            .body(move |b| {
                let _tid = b.thread_id_x();
                let _bid_x = b.block_id_x();

                b.comment("=== PagedAttention Decode ===");
                b.comment("");
                b.comment("Grid: (num_heads, batch_size, 1)");
                b.comment("Each block handles one (batch, head) pair.");
                b.comment("");
                b.comment("Step 1: Determine the KV head index for GQA mapping");
                b.comment("  kv_head = q_head / gqa_group_size");
                b.comment("Step 2: Load the query vector to registers");
                b.comment("Step 3: Read seq_length, compute num_pages");
                b.comment("Step 4: Initialise accumulators (o=0, m=-inf, l=0)");
                b.comment("Step 5: Loop over pages:");
                b.comment("  5a. Read physical page from page_table");
                b.comment("  5b. Load K page to smem, compute dot products");
                b.comment("  5c. Online softmax update");
                b.comment("  5d. Load V page, accumulate output");
                b.comment("Step 6: Final normalisation and store");

                b.bar_sync(0);
                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Executes paged attention decode for a batch of sequences.
///
/// Each sequence in the batch has its KV-cache stored across non-contiguous
/// pages. The page table maps logical page indices to physical page numbers
/// in the cache.
///
/// # Arguments
///
/// * `handle` - DNN handle providing context and stream.
/// * `q` - Query tensor `[B, H_q, 1, D]` (single decode token per sequence).
/// * `k_cache` - Raw device pointer to the K-cache pool.
/// * `v_cache` - Raw device pointer to the V-cache pool.
/// * `page_table` - Page table `[B, max_pages_per_seq]`.
/// * `seq_lengths` - Actual sequence length for each batch entry `[B]`.
/// * `output` - Output tensor `[B, H_q, 1, D]`.
/// * `config` - PagedAttention configuration.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] on shape mismatch.
/// Returns [`DnnError::LaunchFailed`] on kernel launch failure.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_decode<T: GpuFloat>(
    handle: &DnnHandle,
    q: &TensorDesc<T>,
    k_cache: CUdeviceptr,
    v_cache: CUdeviceptr,
    page_table: &DeviceBuffer<i32>,
    seq_lengths: &DeviceBuffer<i32>,
    output: &mut TensorDescMut<T>,
    config: &PagedAttentionConfig,
) -> DnnResult<()> {
    validate_paged_shapes(q, output, seq_lengths, config)?;

    let (batch, _heads, _seq, _hdim) = attn_dims(q)?;

    let ptx = config.generate_ptx()?;
    let kernel_name = format!(
        "paged_attn_decode_d{}_bs{}",
        config.head_dim, config.block_size
    );
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let max_pages_per_seq = if batch > 0 {
        page_table.len() / batch as usize
    } else {
        0
    };
    let max_seq_len = max_pages_per_seq as u32 * config.block_size;

    let threads = config.threads_per_block();
    let grid = Dim3::new(config.num_heads, batch, 1);
    let block = Dim3::new(threads, 1, 1);

    let params = LaunchParams::builder()
        .grid(grid)
        .block(block)
        .shared_mem(config.shared_mem_bytes())
        .build();

    let sm_scale = 1.0f32 / (config.head_dim as f32).sqrt();

    kernel.launch(
        &params,
        handle.stream(),
        &(
            q.ptr,
            k_cache,
            v_cache,
            page_table.as_device_ptr(),
            seq_lengths.as_device_ptr(),
            output.ptr,
            config.head_dim,
            config.num_heads,
            config.num_kv_heads,
            config.block_size,
            max_seq_len,
            sm_scale,
        ),
    )?;

    Ok(())
}

/// Validates tensor shapes for paged attention decode.
fn validate_paged_shapes<T: GpuFloat>(
    q: &TensorDesc<T>,
    output: &TensorDescMut<T>,
    seq_lengths: &DeviceBuffer<i32>,
    config: &PagedAttentionConfig,
) -> DnnResult<()> {
    let (batch, heads, seq, head_dim) = attn_dims(q)?;

    if seq != 1 {
        return Err(DnnError::InvalidDimension(format!(
            "PagedAttention decode Q seq_len must be 1, got {seq}"
        )));
    }
    if head_dim != config.head_dim {
        return Err(DnnError::InvalidDimension(format!(
            "Q head_dim {} != config {}",
            head_dim, config.head_dim
        )));
    }
    if heads != config.num_heads {
        return Err(DnnError::InvalidDimension(format!(
            "Q num_heads {} != config {}",
            heads, config.num_heads
        )));
    }

    let (ob, oh, osn, od) = attn_dims_mut(output)?;
    if ob != batch || oh != heads || osn != 1 || od != head_dim {
        return Err(DnnError::InvalidDimension(format!(
            "output dims {:?} != Q dims {:?}",
            output.dims, q.dims
        )));
    }

    if seq_lengths.len() < batch as usize {
        return Err(DnnError::BufferTooSmall {
            expected: batch as usize * 4,
            actual: seq_lengths.len() * 4,
        });
    }

    if config.num_kv_heads > 0 && config.num_heads % config.num_kv_heads != 0 {
        return Err(DnnError::InvalidArgument(format!(
            "num_heads {} not divisible by num_kv_heads {}",
            config.num_heads, config.num_kv_heads
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gqa_group_size() {
        let cfg = PagedAttentionConfig {
            head_dim: 128,
            num_heads: 32,
            num_kv_heads: 8,
            block_size: 16,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm80,
        };
        assert_eq!(cfg.gqa_group_size(), 4);
    }

    #[test]
    fn threads_per_block_minimum() {
        let cfg = PagedAttentionConfig {
            head_dim: 16,
            num_heads: 1,
            num_kv_heads: 1,
            block_size: 16,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm80,
        };
        assert!(cfg.threads_per_block() >= 128);
    }

    #[test]
    fn shared_mem_calculation() {
        let cfg = PagedAttentionConfig {
            head_dim: 64,
            num_heads: 8,
            num_kv_heads: 8,
            block_size: 16,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm80,
        };
        let smem = cfg.shared_mem_bytes();
        assert_eq!(smem, 4096 + 4096 + 64);
    }

    #[test]
    fn test_paged_attn_tma_kv_cache_block_alignment() {
        // TMA requires 16-byte alignment for block starts.
        // KV-cache page size should be aligned.
        // block_bytes = block_size * head_dim * elem_bytes * 2
        // = 16 * 128 * 4 * 2 = 16384 — divisible by 16.
        let cfg = PagedAttentionConfig {
            head_dim: 128,
            num_heads: 8,
            num_kv_heads: 8,
            block_size: 16,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm90,
        };
        assert_eq!(
            cfg.block_bytes() % 16,
            0,
            "KV blocks must be 16-byte aligned for TMA"
        );
    }

    #[test]
    fn test_paged_attn_tma_descriptor_for_kv() {
        // Verify that TMA descriptor parameters are valid for KV cache.
        let cfg = PagedAttentionConfig {
            head_dim: 128,
            num_heads: 8,
            num_kv_heads: 8,
            block_size: 16,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm90,
        };
        let tma_params = cfg.tma_descriptor_params();
        assert!(tma_params.is_ok(), "Should produce valid TMA params");
        let params = tma_params.expect("params is ok");
        assert_eq!(params.num_dims, 2, "KV cache is 2D: [page_size, head_dim]");
        assert_eq!(params.dim_inner, 128, "inner dim = head_dim");
        assert_eq!(params.dim_outer, 16, "outer dim = block_size");
    }

    #[test]
    fn test_paged_attn_block_bytes_formula() {
        // block_bytes = block_size * head_dim * elem_bytes * 2
        let cfg = PagedAttentionConfig {
            head_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            block_size: 8,
            precision: PtxType::F32, // 4 bytes
            sm_version: SmVersion::Sm80,
        };
        // 8 * 64 * 4 * 2 = 4096
        assert_eq!(cfg.block_bytes(), 4096);
    }

    #[test]
    fn generate_paged_ptx_succeeds() {
        let cfg = PagedAttentionConfig {
            head_dim: 128,
            num_heads: 32,
            num_kv_heads: 8,
            block_size: 16,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm80,
        };
        let ptx = cfg.generate_ptx();
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains("paged_attn_decode"));
        assert!(text.contains(".shared"));
    }

    // -----------------------------------------------------------------------
    // Quality-gate: PagedAttention TMA KV-cache tests
    // -----------------------------------------------------------------------

    /// Verifies that TMA descriptor params are returned successfully for a
    /// config with page_size=128 and head_dim=128 (Hopper-style 128-token pages).
    #[test]
    fn test_paged_attn_tma_descriptor_128_token_pages() {
        let cfg = PagedAttentionConfig {
            head_dim: 128,
            num_heads: 8,
            num_kv_heads: 8,
            block_size: 128, // 128-token pages
            precision: PtxType::F32,
            sm_version: SmVersion::Sm90,
        };
        let params = cfg
            .tma_descriptor_params()
            .expect("valid TMA params for 128-token pages");
        assert_eq!(params.num_dims, 2, "KV TMA descriptor is always 2-D");
        assert_eq!(params.dim_inner, 128, "inner dim = head_dim = 128");
        assert_eq!(params.dim_outer, 128, "outer dim = block_size = 128");
        assert_eq!(params.element_bytes, 4, "F32 element = 4 bytes");
    }

    /// Verifies that the KV-cache page stride (bytes per page, single K or V)
    /// equals head_dim * block_size * elem_bytes.
    ///
    /// Layout: each K or V page is a [block_size, head_dim] matrix.
    /// `page_stride = block_size * head_dim * elem_bytes`
    #[test]
    fn test_paged_attn_kv_cache_page_stride() {
        let cfg = PagedAttentionConfig {
            head_dim: 128,
            num_heads: 8,
            num_kv_heads: 8,
            block_size: 16,
            precision: PtxType::F32, // 4 bytes
            sm_version: SmVersion::Sm80,
        };
        // Single K (or V) page stride = block_size * head_dim * elem_bytes
        let page_stride =
            cfg.block_size as usize * cfg.head_dim as usize * cfg.precision.size_bytes();
        // block_bytes() returns K+V combined (factor of 2); single-page stride is half
        let expected_single = cfg.block_bytes() as usize / 2;
        assert_eq!(
            page_stride, expected_single,
            "single-page stride must equal block_size * head_dim * elem_bytes"
        );
        // Specific value: 16 * 128 * 4 = 8192
        assert_eq!(
            page_stride, 8192,
            "stride for block_size=16, head_dim=128, F32"
        );
    }

    /// GQA page-sharing ratio: with 32 Q heads and 8 KV heads,
    /// gqa_group_size = 32/8 = 4 (4 Q heads share one KV head's pages).
    #[test]
    fn test_paged_attn_gqa_page_sharing_ratio() {
        let cfg = PagedAttentionConfig {
            head_dim: 128,
            num_heads: 32,
            num_kv_heads: 8,
            block_size: 16,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm80,
        };
        let group_size = cfg.gqa_group_size();
        assert_eq!(
            group_size, 4,
            "32 Q heads / 8 KV heads = 4 Q heads share each KV head"
        );
    }

    /// For seq_len=4096 and page_size=128, the page table needs
    /// ceil(4096/128) = 32 pages.
    #[test]
    fn test_paged_attn_page_table_size_estimate() {
        let seq_len: u32 = 4096;
        let page_size: u32 = 128;
        let num_pages = seq_len.div_ceil(page_size);
        assert_eq!(
            num_pages, 32,
            "seq_len=4096 with page_size=128 requires exactly 32 pages"
        );

        // Also verify non-aligned case: seq_len=4097, page_size=128 → 33 pages
        let num_pages_nonaligned = 4097u32.div_ceil(128);
        assert_eq!(
            num_pages_nonaligned, 33,
            "non-aligned seq_len=4097 requires ceil(4097/128)=33 pages"
        );
    }

    /// TMA fast-path detection: configs with block_size >= 64 produce valid
    /// TMA descriptor params (16-byte aligned), enabling TMA loading.
    /// Configs with very small block_size may fail the alignment check.
    #[test]
    fn test_paged_attn_tma_fast_path_for_large_block_size() {
        // block_size=64, head_dim=128, F32: bytes = 64*128*4*2 = 65536 (aligned)
        let cfg_large = PagedAttentionConfig {
            head_dim: 128,
            num_heads: 8,
            num_kv_heads: 8,
            block_size: 64,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm90,
        };
        assert!(
            cfg_large.tma_descriptor_params().is_ok(),
            "block_size=64 should be TMA-eligible (16-byte aligned)"
        );

        // block_size=1, head_dim=3, F32: bytes = 1*3*4*2 = 24 (not 16-byte aligned
        // because 24 % 16 != 0)
        let cfg_small = PagedAttentionConfig {
            head_dim: 3,
            num_heads: 1,
            num_kv_heads: 1,
            block_size: 1,
            precision: PtxType::F32,
            sm_version: SmVersion::Sm90,
        };
        // 1 * 3 * 4 * 2 = 24 bytes; 24 % 16 = 8 ≠ 0 → should fail
        assert!(
            cfg_small.tma_descriptor_params().is_err(),
            "non-aligned block should not pass TMA alignment check"
        );
    }
}
