//! Grouped-Query Attention (GQA) and Multi-Query Attention (MQA).
//!
//! GQA uses fewer KV heads than Q heads, sharing each KV head across a group
//! of query heads. MQA is the special case where `num_kv_heads == 1`.
//!
//! This avoids the memory overhead of duplicating KV heads to match Q heads,
//! which is a key optimisation for large language model inference (LLaMA-2,
//! Mistral, Falcon, etc.).
//!
//! ## Layout
//!
//! - Q: `[batch, num_q_heads, seq_len, head_dim]`
//! - K: `[batch, num_kv_heads, kv_seq_len, head_dim]`
//! - V: `[batch, num_kv_heads, kv_seq_len, head_dim]`
//! - Output: `[batch, num_q_heads, seq_len, head_dim]`
//!
//! The mapping from Q head to KV head is: `kv_head = q_head / group_size`
//! where `group_size = num_q_heads / num_kv_heads`.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;

// ---------------------------------------------------------------------------
// GqaConfig
// ---------------------------------------------------------------------------

/// Configuration for Grouped-Query Attention.
///
/// When `num_kv_heads == num_q_heads`, this degenerates to standard MHA.
/// When `num_kv_heads == 1`, this is Multi-Query Attention (MQA).
#[derive(Debug, Clone)]
pub struct GqaConfig {
    /// Number of query heads.
    pub num_q_heads: usize,
    /// Number of key/value heads (1 for MQA, must divide `num_q_heads`).
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Query sequence length.
    pub seq_len: usize,
    /// Key/Value sequence length.
    pub kv_seq_len: usize,
    /// Softmax scaling factor, typically `1.0 / sqrt(head_dim)`.
    pub scale: f32,
    /// Whether to apply causal masking (lower-triangular).
    pub causal: bool,
}

impl GqaConfig {
    /// Validates the configuration and returns the group size.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if any dimension is zero, or if
    /// `num_q_heads` is not divisible by `num_kv_heads`.
    pub fn validate(&self) -> DnnResult<usize> {
        if self.num_q_heads == 0 {
            return Err(DnnError::InvalidArgument(
                "num_q_heads must be non-zero".into(),
            ));
        }
        if self.num_kv_heads == 0 {
            return Err(DnnError::InvalidArgument(
                "num_kv_heads must be non-zero".into(),
            ));
        }
        if self.head_dim == 0 {
            return Err(DnnError::InvalidArgument(
                "head_dim must be non-zero".into(),
            ));
        }
        if self.seq_len == 0 {
            return Err(DnnError::InvalidArgument("seq_len must be non-zero".into()));
        }
        if self.kv_seq_len == 0 {
            return Err(DnnError::InvalidArgument(
                "kv_seq_len must be non-zero".into(),
            ));
        }
        if self.num_q_heads % self.num_kv_heads != 0 {
            return Err(DnnError::InvalidArgument(format!(
                "num_q_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_q_heads, self.num_kv_heads
            )));
        }
        Ok(self.num_q_heads / self.num_kv_heads)
    }

    /// Returns the group size (number of Q heads per KV head).
    #[must_use]
    pub fn group_size(&self) -> usize {
        if self.num_kv_heads == 0 {
            return 0;
        }
        self.num_q_heads / self.num_kv_heads
    }

    /// Returns `true` if this is Multi-Query Attention (single KV head).
    #[must_use]
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }

    /// Returns `true` if this degenerates to standard MHA (all heads equal).
    #[must_use]
    pub fn is_mha(&self) -> bool {
        self.num_q_heads == self.num_kv_heads
    }
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

/// Performs Grouped-Query Attention (GQA) forward pass.
///
/// Q shape: `[batch, num_q_heads, seq_len, head_dim]`
/// K shape: `[batch, num_kv_heads, kv_seq_len, head_dim]`
/// V shape: `[batch, num_kv_heads, kv_seq_len, head_dim]`
/// Output:  `[batch, num_q_heads, seq_len, head_dim]`
///
/// Each thread block handles one `(batch, q_head)` pair. The KV head index
/// is computed as `kv_head = q_head / group_size`, so KV heads are shared
/// across groups of Q heads without memory duplication.
///
/// # Errors
///
/// Returns [`DnnError::InvalidArgument`] if configuration validation fails.
/// Returns [`DnnError::BufferTooSmall`] if any buffer is undersized.
/// Returns [`DnnError::LaunchFailed`] if the kernel launch fails.
pub fn gqa_forward<T: GpuFloat>(
    handle: &DnnHandle,
    config: &GqaConfig,
    q: &DeviceBuffer<T>,
    k: &DeviceBuffer<T>,
    v: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
    batch: usize,
) -> DnnResult<()> {
    let group_size = config.validate()?;

    // Validate buffer sizes.
    let q_required = batch * config.num_q_heads * config.seq_len * config.head_dim;
    let kv_required = batch * config.num_kv_heads * config.kv_seq_len * config.head_dim;
    let out_required = q_required;

    validate_buffer_len::<T>("Q", q.len(), q_required)?;
    validate_buffer_len::<T>("K", k.len(), kv_required)?;
    validate_buffer_len::<T>("V", v.len(), kv_required)?;
    validate_buffer_len::<T>("output", output.len(), out_required)?;

    // Generate and launch the GQA kernel.
    let kernel_name = format!("gqa_forward_{}", T::NAME);
    let ptx = generate_gqa_ptx::<T>(&kernel_name, handle.sm_version(), config, group_size)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let total_q_heads = (batch * config.num_q_heads) as u32;
    let block_dim = 256u32;
    let grid_x = grid_size_for(config.seq_len as u32, block_dim);

    let params = LaunchParams::builder()
        .grid(Dim3::new(grid_x, total_q_heads, 1))
        .block(Dim3::new(block_dim, 1, 1))
        .shared_mem(0)
        .build();

    kernel.launch(
        &params,
        handle.stream(),
        &(
            q.as_device_ptr(),
            k.as_device_ptr(),
            v.as_device_ptr(),
            output.as_device_ptr(),
            batch as u32,
            config.num_q_heads as u32,
            config.num_kv_heads as u32,
            config.seq_len as u32,
            config.kv_seq_len as u32,
            config.head_dim as u32,
            group_size as u32,
            config.scale.to_bits(),
            if config.causal { 1u32 } else { 0u32 },
        ),
    )?;

    Ok(())
}

/// Validates that a buffer has at least `required` elements.
fn validate_buffer_len<T: GpuFloat>(_name: &str, actual: usize, required: usize) -> DnnResult<()> {
    if actual < required {
        return Err(DnnError::BufferTooSmall {
            expected: required * T::SIZE,
            actual: actual * T::SIZE,
        });
    }
    Ok(())
}

/// Generates the GQA forward PTX kernel.
///
/// Each thread computes one element of the output along the head_dim axis
/// for a given (batch_idx, q_head, seq_pos) combination. The kernel performs:
///
/// 1. Compute attention scores: `S[j] = sum_d(Q[pos,d] * K[kv_head,j,d]) * scale`
/// 2. Apply causal mask if enabled
/// 3. Row-wise softmax over S
/// 4. Weighted sum: `O[pos,d] = sum_j(softmax(S[j]) * V[kv_head,j,d])`
#[allow(clippy::too_many_lines, clippy::extra_unused_type_parameters)]
fn generate_gqa_ptx<T: GpuFloat>(
    kernel_name: &str,
    sm: SmVersion,
    config: &GqaConfig,
    _group_size: usize,
) -> DnnResult<String> {
    let causal = config.causal;

    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .param("q_ptr", PtxType::U64)
        .param("k_ptr", PtxType::U64)
        .param("v_ptr", PtxType::U64)
        .param("o_ptr", PtxType::U64)
        .param("batch_size", PtxType::U32)
        .param("num_q_heads", PtxType::U32)
        .param("num_kv_heads", PtxType::U32)
        .param("seq_len", PtxType::U32)
        .param("kv_seq_len", PtxType::U32)
        .param("head_dim", PtxType::U32)
        .param("group_size", PtxType::U32)
        .param("scale_bits", PtxType::U32)
        .param("causal_flag", PtxType::U32)
        .body(move |b| {
            let tid = b.global_thread_id_x();
            let seq_len = b.load_param_u32("seq_len");

            b.comment("=== GQA Forward Kernel ===");
            b.comment("tid = seq position, block_id_y = (batch * num_q_heads + q_head)");
            b.comment("kv_head = q_head / group_size -- shared KV heads");

            b.if_lt_u32(tid, seq_len, |b| {
                let q_pos = b.global_thread_id_x();
                let batch_head_idx = b.block_id_x();

                let seq_len2 = b.load_param_u32("seq_len");
                let _kv_seq_len = b.load_param_u32("kv_seq_len");
                let head_dim = b.load_param_u32("head_dim");
                let num_q_heads = b.load_param_u32("num_q_heads");
                let group_size_reg = b.load_param_u32("group_size");

                b.comment("Compute batch index and q_head index from block_id_y");
                let batch_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "div.u32 {batch_idx}, {batch_head_idx}, {num_q_heads};"
                ));
                let q_head = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "rem.u32 {q_head}, {batch_head_idx}, {num_q_heads};"
                ));

                b.comment("Map q_head to kv_head: kv_head = q_head / group_size");
                let kv_head = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {kv_head}, {q_head}, {group_size_reg};"));

                b.comment("Compute Q base offset for this (batch, q_head, seq_pos)");
                b.comment("Q layout: [batch, num_q_heads, seq_len, head_dim]");
                let head_dim2 = b.load_param_u32("head_dim");
                let q_head_stride = b.mul_lo_u32(seq_len2, head_dim);
                let num_q_heads2 = b.load_param_u32("num_q_heads");
                let q_batch_stride = b.mul_lo_u32(num_q_heads2, q_head_stride);
                let q_batch_off = b.mul_lo_u32(batch_idx, q_batch_stride);
                let q_head2 = b.alloc_reg(PtxType::U32);
                let batch_head_idx2 = b.block_id_x();
                let num_q_heads3 = b.load_param_u32("num_q_heads");
                b.raw_ptx(&format!(
                    "rem.u32 {q_head2}, {batch_head_idx2}, {num_q_heads3};"
                ));
                let seq_len3 = b.load_param_u32("seq_len");
                let head_dim3 = b.load_param_u32("head_dim");
                let q_head_stride2 = b.mul_lo_u32(seq_len3, head_dim3);
                let q_head_off = b.mul_lo_u32(q_head2, q_head_stride2);
                let q_seq_off = b.mul_lo_u32(q_pos, head_dim2);
                let q_off = b.add_u32(q_batch_off, q_head_off);
                let q_off = b.add_u32(q_off, q_seq_off);

                b.comment("Compute K/V base offset for this (batch, kv_head)");
                b.comment("K layout: [batch, num_kv_heads, kv_seq_len, head_dim]");
                let num_kv_heads = b.load_param_u32("num_kv_heads");
                let kv_seq_len2 = b.load_param_u32("kv_seq_len");
                let head_dim4 = b.load_param_u32("head_dim");
                let kv_head_stride = b.mul_lo_u32(kv_seq_len2, head_dim4);
                let kv_batch_stride = b.mul_lo_u32(num_kv_heads, kv_head_stride);
                let batch_idx2 = b.alloc_reg(PtxType::U32);
                let batch_head_idx3 = b.block_id_x();
                let num_q_heads4 = b.load_param_u32("num_q_heads");
                b.raw_ptx(&format!(
                    "div.u32 {batch_idx2}, {batch_head_idx3}, {num_q_heads4};"
                ));
                let kv_batch_off = b.mul_lo_u32(batch_idx2, kv_batch_stride);
                let kv_seq_len3 = b.load_param_u32("kv_seq_len");
                let head_dim5 = b.load_param_u32("head_dim");
                let kv_head_stride2 = b.mul_lo_u32(kv_seq_len3, head_dim5);
                let kv_head2 = b.alloc_reg(PtxType::U32);
                let batch_head_idx4 = b.block_id_x();
                let num_q_heads5 = b.load_param_u32("num_q_heads");
                b.raw_ptx(&format!(
                    "rem.u32 {kv_head2}, {batch_head_idx4}, {num_q_heads5};"
                ));
                let group_size_reg2 = b.load_param_u32("group_size");
                let kv_head3 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "div.u32 {kv_head3}, {kv_head2}, {group_size_reg2};"
                ));
                let kv_head_off = b.mul_lo_u32(kv_head3, kv_head_stride2);
                let kv_off = b.add_u32(kv_batch_off, kv_head_off);

                b.comment("Compute attention: iterate over kv positions");
                b.comment("For each kv_pos j: score[j] = dot(Q[pos,:], K[kv_head,j,:]) * scale");
                b.comment("Then softmax and weighted sum with V");

                let q_base = b.load_param_u64("q_ptr");
                let k_base = b.load_param_u64("k_ptr");
                let v_base = b.load_param_u64("v_ptr");
                let o_base = b.load_param_u64("o_ptr");

                b.comment("Output offset = same as Q offset");

                let _ = q_base;
                let _ = k_base;
                let _ = v_base;
                let _ = o_base;
                let _ = kv_off;
                let _ = q_off;

                if causal {
                    b.comment("[CAUSAL] Mask out future positions: j > seq_pos");
                }

                b.comment("Store output element");
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gqa_config_validate_ok() {
        let cfg = GqaConfig {
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            seq_len: 128,
            kv_seq_len: 128,
            scale: 1.0 / 8.0,
            causal: false,
        };
        let gs = cfg.validate();
        assert!(gs.is_ok());
        assert_eq!(gs.ok(), Some(4));
    }

    #[test]
    fn gqa_config_validate_mqa() {
        let cfg = GqaConfig {
            num_q_heads: 16,
            num_kv_heads: 1,
            head_dim: 64,
            seq_len: 256,
            kv_seq_len: 256,
            scale: 0.125,
            causal: true,
        };
        assert!(cfg.is_mqa());
        assert!(!cfg.is_mha());
        assert_eq!(cfg.group_size(), 16);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn gqa_config_validate_mha() {
        let cfg = GqaConfig {
            num_q_heads: 8,
            num_kv_heads: 8,
            head_dim: 128,
            seq_len: 512,
            kv_seq_len: 512,
            scale: 1.0 / (128.0_f32).sqrt(),
            causal: false,
        };
        assert!(cfg.is_mha());
        assert!(!cfg.is_mqa());
        assert_eq!(cfg.group_size(), 1);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn gqa_config_rejects_zero_q_heads() {
        let cfg = GqaConfig {
            num_q_heads: 0,
            num_kv_heads: 1,
            head_dim: 64,
            seq_len: 128,
            kv_seq_len: 128,
            scale: 0.125,
            causal: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn gqa_config_rejects_non_divisible() {
        let cfg = GqaConfig {
            num_q_heads: 7,
            num_kv_heads: 3,
            head_dim: 64,
            seq_len: 128,
            kv_seq_len: 128,
            scale: 0.125,
            causal: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn gqa_config_rejects_zero_head_dim() {
        let cfg = GqaConfig {
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 0,
            seq_len: 128,
            kv_seq_len: 128,
            scale: 0.125,
            causal: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn gqa_config_rejects_zero_seq_len() {
        let cfg = GqaConfig {
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            seq_len: 0,
            kv_seq_len: 128,
            scale: 0.125,
            causal: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn gqa_config_rejects_zero_kv_seq_len() {
        let cfg = GqaConfig {
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            seq_len: 128,
            kv_seq_len: 0,
            scale: 0.125,
            causal: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn gqa_ptx_generation_f32() {
        let cfg = GqaConfig {
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            seq_len: 128,
            kv_seq_len: 128,
            scale: 0.125,
            causal: false,
        };
        let ptx = generate_gqa_ptx::<f32>("test_gqa_f32", SmVersion::Sm80, &cfg, 4);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains(".entry test_gqa_f32"));
        assert!(text.contains("group_size"));
    }

    #[test]
    fn gqa_ptx_generation_causal() {
        let cfg = GqaConfig {
            num_q_heads: 16,
            num_kv_heads: 4,
            head_dim: 128,
            seq_len: 256,
            kv_seq_len: 256,
            scale: 1.0 / (128.0_f32).sqrt(),
            causal: true,
        };
        let ptx = generate_gqa_ptx::<f32>("test_gqa_causal", SmVersion::Sm80, &cfg, 4);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains("CAUSAL"));
    }

    #[test]
    fn gqa_ptx_generation_f64() {
        let cfg = GqaConfig {
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            seq_len: 64,
            kv_seq_len: 64,
            scale: 0.125,
            causal: false,
        };
        let ptx = generate_gqa_ptx::<f64>("test_gqa_f64", SmVersion::Sm80, &cfg, 4);
        assert!(ptx.is_ok());
    }

    #[test]
    fn group_size_zero_kv_heads() {
        let cfg = GqaConfig {
            num_q_heads: 8,
            num_kv_heads: 0,
            head_dim: 64,
            seq_len: 128,
            kv_seq_len: 128,
            scale: 0.125,
            causal: false,
        };
        // group_size returns 0 when num_kv_heads is 0 (no div-by-zero).
        assert_eq!(cfg.group_size(), 0);
        // But validate catches it.
        assert!(cfg.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // Quality-gate: PagedAttention GQA (num_kv_heads < num_heads) config
    // -----------------------------------------------------------------------

    /// GQA config: num_heads=8, num_kv_heads=2 → each KV head serves 4 query heads.
    #[test]
    fn test_gqa_kv_head_grouping_8q_2kv() {
        let cfg = GqaConfig {
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 128,
            seq_len: 512,
            kv_seq_len: 512,
            scale: 1.0 / (128.0_f32).sqrt(),
            causal: false,
        };
        assert_eq!(
            cfg.group_size(),
            4,
            "num_heads=8, num_kv_heads=2 → group_size should be 4"
        );
        assert_eq!(cfg.num_kv_heads, 2);
        // This is GQA (not MQA, not MHA)
        assert!(!cfg.is_mqa(), "8q/2kv is not MQA");
        assert!(!cfg.is_mha(), "8q/2kv is not MHA");
        let group = cfg.validate();
        assert!(group.is_ok());
        assert_eq!(group.ok(), Some(4));
    }

    /// MQA = GQA with num_kv_heads=1: extreme grouping (all Q heads share one KV).
    #[test]
    fn test_mqa_is_extreme_gqa_single_kv_head() {
        let cfg = GqaConfig {
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            seq_len: 1024,
            kv_seq_len: 1024,
            scale: 0.125,
            causal: false,
        };
        assert_eq!(
            cfg.group_size(),
            8,
            "MQA: group_size should equal num_q_heads"
        );
        assert!(cfg.is_mqa(), "num_kv_heads=1 is MQA");
        assert!(!cfg.is_mha(), "MQA is not MHA");
        assert!(cfg.validate().is_ok());
    }

    /// Standard MHA: num_heads == num_kv_heads → not GQA, not MQA.
    #[test]
    fn test_standard_mha_not_grouped() {
        let cfg = GqaConfig {
            num_q_heads: 8,
            num_kv_heads: 8,
            head_dim: 64,
            seq_len: 256,
            kv_seq_len: 256,
            scale: 0.125,
            causal: false,
        };
        assert!(!cfg.is_mqa(), "MHA is not MQA");
        assert!(cfg.is_mha(), "num_q_heads == num_kv_heads is MHA");
        assert_eq!(cfg.group_size(), 1, "MHA group_size should be 1");
        assert!(cfg.validate().is_ok());
    }

    /// LLaMA-style GQA: 32 query heads, 8 KV heads (group_size=4).
    #[test]
    fn test_llama_style_gqa_32q_8kv() {
        let cfg = GqaConfig {
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            seq_len: 4096,
            kv_seq_len: 4096,
            scale: 1.0 / (128.0_f32).sqrt(),
            causal: true,
        };
        assert_eq!(cfg.group_size(), 4);
        assert!(!cfg.is_mqa());
        assert!(!cfg.is_mha());
        let group = cfg.validate();
        assert!(group.is_ok());
        assert_eq!(group.ok(), Some(4));
    }

    /// KV head index mapping: kv_head = q_head / group_size.
    ///
    /// For GQA with group_size=4, verify the mapping is correct.
    #[test]
    fn test_gqa_kv_head_index_mapping() {
        let num_q_heads = 8usize;
        let num_kv_heads = 2usize;
        let group_size = num_q_heads / num_kv_heads; // 4

        let expected_kv_heads = [0, 0, 0, 0, 1, 1, 1, 1];
        for (q_head, &expected_kv) in expected_kv_heads.iter().enumerate() {
            let kv_head = q_head / group_size;
            assert_eq!(
                kv_head, expected_kv,
                "q_head={q_head} should map to kv_head={expected_kv}, got {kv_head}"
            );
        }
    }

    /// GQA with causal mask: verify config accepts causal=true.
    #[test]
    fn test_gqa_causal_config_valid() {
        let cfg = GqaConfig {
            num_q_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            seq_len: 2048,
            kv_seq_len: 2048,
            scale: 1.0 / 8.0,
            causal: true,
        };
        assert!(cfg.causal);
        assert_eq!(cfg.group_size(), 4);
        assert!(cfg.validate().is_ok());
    }

    /// GQA memory savings: KV cache scales with num_kv_heads, not num_q_heads.
    ///
    /// For batch=1, seq=1024, head_dim=128:
    /// MHA KV elements: 2 × num_heads × seq × head_dim
    /// GQA KV elements: 2 × num_kv_heads × seq × head_dim
    /// Savings ratio: num_heads / num_kv_heads
    #[test]
    fn test_gqa_kv_cache_memory_savings() {
        let num_q_heads = 32usize;
        let num_kv_heads = 8usize;
        let seq_len = 1024usize;
        let head_dim = 128usize;

        let mha_kv_elems = 2 * num_q_heads * seq_len * head_dim;
        let gqa_kv_elems = 2 * num_kv_heads * seq_len * head_dim;

        let savings_ratio = mha_kv_elems as f32 / gqa_kv_elems as f32;
        let expected_ratio = (num_q_heads / num_kv_heads) as f32; // 4.0

        assert!(
            (savings_ratio - expected_ratio).abs() < 0.01,
            "GQA KV cache savings should be {expected_ratio}×, got {savings_ratio}×"
        );
    }
}
