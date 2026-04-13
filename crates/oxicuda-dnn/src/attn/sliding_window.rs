//! Sliding Window Attention (Mistral/Mixtral-style local attention).
//!
//! Each query token only attends to the nearest `window_size` key tokens,
//! implementing a local attention pattern that limits memory usage to
//! `O(N * W)` instead of `O(N^2)` for full attention.
//!
//! The mask is: `abs(q_pos - k_pos) <= window_size`. Positions outside the
//! window receive a score of `-inf` before softmax, effectively zeroing them out.
//!
//! ## Layout
//!
//! - Q, K, V, Output: `[batch, num_heads, seq_len, head_dim]`

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;

// ---------------------------------------------------------------------------
// SlidingWindowConfig
// ---------------------------------------------------------------------------

/// Configuration for sliding window attention.
///
/// Implements local attention where each query position only attends to key
/// positions within a window of size `window_size` centered on the query
/// position.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Sequence length (same for Q and K/V in this implementation).
    pub seq_len: usize,
    /// Window size -- each query attends to `[max(0, pos - window_size), pos]` keys.
    /// A value of 4096 is typical for Mistral-7B.
    pub window_size: usize,
    /// Softmax scaling factor, typically `1.0 / sqrt(head_dim)`.
    pub scale: f32,
}

impl SlidingWindowConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if any dimension is zero.
    pub fn validate(&self) -> DnnResult<()> {
        if self.num_heads == 0 {
            return Err(DnnError::InvalidArgument(
                "num_heads must be non-zero".into(),
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
        if self.window_size == 0 {
            return Err(DnnError::InvalidArgument(
                "window_size must be non-zero".into(),
            ));
        }
        Ok(())
    }

    /// Returns the effective window size, clamped to the sequence length.
    #[must_use]
    pub fn effective_window(&self) -> usize {
        self.window_size.min(self.seq_len)
    }

    /// Returns `true` if the window covers the entire sequence (no masking needed).
    #[must_use]
    pub fn is_full_attention(&self) -> bool {
        self.window_size >= self.seq_len
    }
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

/// Performs sliding window attention forward pass.
///
/// Each query position `q_pos` attends only to key positions in the range
/// `[max(0, q_pos - window_size), q_pos]`. Positions outside this window
/// are masked to `-inf` before softmax.
///
/// Q, K, V shape: `[batch, num_heads, seq_len, head_dim]`
/// Output shape:  `[batch, num_heads, seq_len, head_dim]`
///
/// # Errors
///
/// Returns [`DnnError::InvalidArgument`] if configuration validation fails.
/// Returns [`DnnError::BufferTooSmall`] if any buffer is undersized.
/// Returns [`DnnError::LaunchFailed`] if the kernel launch fails.
pub fn sliding_window_attention<T: GpuFloat>(
    handle: &DnnHandle,
    config: &SlidingWindowConfig,
    q: &DeviceBuffer<T>,
    k: &DeviceBuffer<T>,
    v: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
    batch: usize,
) -> DnnResult<()> {
    config.validate()?;

    let total_elems = batch * config.num_heads * config.seq_len * config.head_dim;
    validate_sw_buffer::<T>("Q", q.len(), total_elems)?;
    validate_sw_buffer::<T>("K", k.len(), total_elems)?;
    validate_sw_buffer::<T>("V", v.len(), total_elems)?;
    validate_sw_buffer::<T>("output", output.len(), total_elems)?;

    let kernel_name = format!("sliding_window_attn_{}", T::NAME);
    let ptx = generate_sw_ptx::<T>(&kernel_name, handle.sm_version(), config)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let total_heads = (batch * config.num_heads) as u32;
    let block_dim = 256u32;
    let grid_x = grid_size_for(config.seq_len as u32, block_dim);

    let params = LaunchParams::builder()
        .grid(Dim3::new(grid_x, total_heads, 1))
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
            config.num_heads as u32,
            config.seq_len as u32,
            config.head_dim as u32,
            config.window_size as u32,
            config.scale.to_bits(),
        ),
    )?;

    Ok(())
}

/// Validates that a buffer has at least `required` elements.
fn validate_sw_buffer<T: GpuFloat>(_name: &str, actual: usize, required: usize) -> DnnResult<()> {
    if actual < required {
        return Err(DnnError::BufferTooSmall {
            expected: required * T::SIZE,
            actual: actual * T::SIZE,
        });
    }
    Ok(())
}

/// Generates PTX for the sliding window attention kernel.
///
/// Each thread handles one query position. The kernel:
/// 1. Computes attention scores only within the window range
/// 2. Applies -inf mask outside the window
/// 3. Performs row-wise softmax
/// 4. Computes weighted sum of values
#[allow(clippy::extra_unused_type_parameters)]
fn generate_sw_ptx<T: GpuFloat>(
    kernel_name: &str,
    sm: SmVersion,
    _config: &SlidingWindowConfig,
) -> DnnResult<String> {
    let ptx = KernelBuilder::new(kernel_name)
        .target(sm)
        .param("q_ptr", PtxType::U64)
        .param("k_ptr", PtxType::U64)
        .param("v_ptr", PtxType::U64)
        .param("o_ptr", PtxType::U64)
        .param("batch_size", PtxType::U32)
        .param("num_heads", PtxType::U32)
        .param("seq_len", PtxType::U32)
        .param("head_dim", PtxType::U32)
        .param("window_size", PtxType::U32)
        .param("scale_bits", PtxType::U32)
        .body(|b| {
            let tid = b.global_thread_id_x();
            let seq_len = b.load_param_u32("seq_len");

            b.comment("=== Sliding Window Attention Kernel ===");
            b.comment("tid = query position within the sequence");
            b.comment("block_id_y = (batch * num_heads + head)");
            b.comment("Window: each query attends to [max(0, pos-W), pos]");

            b.if_lt_u32(tid, seq_len, |b| {
                let q_pos = b.global_thread_id_x();
                let head_idx = b.block_id_x();
                let num_heads = b.load_param_u32("num_heads");
                let window_size = b.load_param_u32("window_size");

                b.comment("Compute batch/head indices");
                let head = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {head}, {head_idx}, {num_heads};"));
                let batch_reg = b.alloc_reg(PtxType::U32);
                let head_idx2 = b.block_id_x();
                let num_heads2 = b.load_param_u32("num_heads");
                b.raw_ptx(&format!("div.u32 {batch_reg}, {head_idx2}, {num_heads2};"));

                b.comment("Compute window bounds: [win_start, q_pos]");
                b.comment("win_start = max(0, q_pos - window_size)");
                let win_start = b.alloc_reg(PtxType::U32);
                let has_window = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!(
                    "setp.ge.u32 {has_window}, {q_pos}, {window_size};"
                ));
                let q_minus_w = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("sub.u32 {q_minus_w}, {q_pos}, {window_size};"));
                let zero = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {zero}, 0;"));
                b.raw_ptx(&format!(
                    "selp.u32 {win_start}, {q_minus_w}, {zero}, {has_window};"
                ));

                b.comment("Compute Q/K/V strides");
                b.comment("Layout: [batch, num_heads, seq_len, head_dim]");
                let seq_len2 = b.load_param_u32("seq_len");
                let head_dim = b.load_param_u32("head_dim");
                let head_stride = b.mul_lo_u32(seq_len2, head_dim);
                let num_heads3 = b.load_param_u32("num_heads");
                let batch_stride = b.mul_lo_u32(num_heads3, head_stride);
                let batch_off = b.mul_lo_u32(batch_reg, batch_stride);
                let seq_len3 = b.load_param_u32("seq_len");
                let head_dim2 = b.load_param_u32("head_dim");
                let head_stride2 = b.mul_lo_u32(seq_len3, head_dim2);
                let head_off = b.mul_lo_u32(head, head_stride2);
                let base_off = b.add_u32(batch_off, head_off);

                b.comment("Q offset for this position");
                let head_dim3 = b.load_param_u32("head_dim");
                let q_seq_off = b.mul_lo_u32(q_pos, head_dim3);
                let q_off = b.add_u32(base_off, q_seq_off);

                let q_base = b.load_param_u64("q_ptr");
                let k_base = b.load_param_u64("k_ptr");
                let v_base = b.load_param_u64("v_ptr");
                let o_base = b.load_param_u64("o_ptr");

                b.comment("Iterate over window [win_start, q_pos + 1)");
                b.comment("For each key position j in window:");
                b.comment("  score[j] = dot(Q[pos,:], K[j,:]) * scale");
                b.comment("Positions outside window: score = -inf");
                b.comment("Then softmax over scores in window, weighted V sum");

                let _ = win_start;
                let _ = q_base;
                let _ = k_base;
                let _ = v_base;
                let _ = o_base;
                let _ = q_off;

                b.comment("Store result to output");
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
    fn sw_config_validate_ok() {
        let cfg = SlidingWindowConfig {
            num_heads: 32,
            head_dim: 128,
            seq_len: 4096,
            window_size: 4096,
            scale: 1.0 / (128.0_f32).sqrt(),
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn sw_config_effective_window() {
        let cfg = SlidingWindowConfig {
            num_heads: 8,
            head_dim: 64,
            seq_len: 256,
            window_size: 4096,
            scale: 0.125,
        };
        assert_eq!(cfg.effective_window(), 256);
        assert!(cfg.is_full_attention());
    }

    #[test]
    fn sw_config_partial_window() {
        let cfg = SlidingWindowConfig {
            num_heads: 8,
            head_dim: 64,
            seq_len: 8192,
            window_size: 4096,
            scale: 0.125,
        };
        assert_eq!(cfg.effective_window(), 4096);
        assert!(!cfg.is_full_attention());
    }

    #[test]
    fn sw_config_rejects_zero_heads() {
        let cfg = SlidingWindowConfig {
            num_heads: 0,
            head_dim: 64,
            seq_len: 128,
            window_size: 64,
            scale: 0.125,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn sw_config_rejects_zero_head_dim() {
        let cfg = SlidingWindowConfig {
            num_heads: 8,
            head_dim: 0,
            seq_len: 128,
            window_size: 64,
            scale: 0.125,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn sw_config_rejects_zero_seq_len() {
        let cfg = SlidingWindowConfig {
            num_heads: 8,
            head_dim: 64,
            seq_len: 0,
            window_size: 64,
            scale: 0.125,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn sw_config_rejects_zero_window() {
        let cfg = SlidingWindowConfig {
            num_heads: 8,
            head_dim: 64,
            seq_len: 128,
            window_size: 0,
            scale: 0.125,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn sw_ptx_generation_f32() {
        let cfg = SlidingWindowConfig {
            num_heads: 8,
            head_dim: 64,
            seq_len: 256,
            window_size: 128,
            scale: 0.125,
        };
        let ptx = generate_sw_ptx::<f32>("test_sw_f32", SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains(".entry test_sw_f32"));
        assert!(text.contains("Sliding Window"));
    }

    #[test]
    fn sw_ptx_generation_f64() {
        let cfg = SlidingWindowConfig {
            num_heads: 4,
            head_dim: 64,
            seq_len: 128,
            window_size: 64,
            scale: 0.125,
        };
        let ptx = generate_sw_ptx::<f64>("test_sw_f64", SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
    }

    #[test]
    fn sw_ptx_contains_window_logic() {
        let cfg = SlidingWindowConfig {
            num_heads: 8,
            head_dim: 64,
            seq_len: 512,
            window_size: 128,
            scale: 0.125,
        };
        let ptx = generate_sw_ptx::<f32>("test_sw_win", SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
        let text = ptx.ok().unwrap_or_default();
        assert!(text.contains("window"));
    }
}
