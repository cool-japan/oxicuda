//! Session handle carrying model and hardware configuration for inference.

// ─── InferHandle ─────────────────────────────────────────────────────────────

/// Immutable configuration for an inference session.
///
/// Carries both the transformer model topology (layers, heads, head dim) and
/// the hardware context (device, SM version) required to select correct PTX
/// kernel variants and size the KV cache correctly.
#[derive(Debug, Clone)]
pub struct InferHandle {
    /// CUDA device ordinal (0-based).
    pub device: u32,
    /// SM version integer (e.g., 80 for Ampere A100, 90 for Hopper H100).
    pub sm_version: u32,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of attention heads (query heads; equals KV heads for MHA).
    pub n_heads: usize,
    /// Number of KV heads (< n_heads for GQA / MQA; equals n_heads for MHA).
    pub n_kv_heads: usize,
    /// Per-head dimension (d_k = d_v = head_dim).
    pub head_dim: usize,
    /// Vocabulary size (number of output logits per token).
    pub vocab_size: usize,
    /// Tokens per KV cache block (PagedAttention block size).
    pub block_size: usize,
    /// Maximum sequence length supported by this session.
    pub max_seq_len: usize,
}

impl InferHandle {
    /// Create a new handle with multi-head attention (n_kv_heads = n_heads).
    #[must_use]
    pub fn new(
        device: u32,
        sm_version: u32,
        n_layers: usize,
        n_heads: usize,
        head_dim: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            device,
            sm_version,
            n_layers,
            n_heads,
            n_kv_heads: n_heads,
            head_dim,
            vocab_size,
            block_size: 16,
            max_seq_len: 4096,
        }
    }

    /// Override KV head count for Grouped-Query Attention (GQA) or MQA.
    ///
    /// # Panics
    /// Panics if `n_kv_heads` does not evenly divide `self.n_heads`.
    #[must_use]
    pub fn with_gqa(mut self, n_kv_heads: usize) -> Self {
        assert_eq!(
            self.n_heads % n_kv_heads,
            0,
            "n_heads ({}) must be divisible by n_kv_heads ({})",
            self.n_heads,
            n_kv_heads,
        );
        self.n_kv_heads = n_kv_heads;
        self
    }

    /// Override the PagedAttention block size.
    #[must_use]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Override the maximum supported sequence length.
    #[must_use]
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// PTX `.version` directive string for this SM.
    #[must_use]
    pub fn ptx_version_str(&self) -> &'static str {
        if self.sm_version >= 100 {
            "8.7"
        } else if self.sm_version >= 90 {
            "8.4"
        } else if self.sm_version >= 80 {
            "8.0"
        } else {
            "7.5"
        }
    }

    /// Number of float32 elements in a single K **or** V block (per KV head).
    #[must_use]
    pub fn kv_block_elements(&self) -> usize {
        self.block_size * self.n_kv_heads * self.head_dim
    }

    /// Bytes consumed by a single K+V block (both K and V, f32).
    #[must_use]
    pub fn kv_block_bytes(&self) -> usize {
        2 * self.kv_block_elements() * 4
    }

    /// Attention scale factor √(1/head_dim) = 1/√head_dim.
    #[must_use]
    pub fn attention_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Default test handle: Ampere SM 80, tiny model (4 layers, 4 heads, 64-dim, vocab 256).
    #[must_use]
    pub fn default_test_handle() -> Self {
        Self::new(0, 80, 4, 4, 64, 256)
    }

    /// GQA test handle (4 query heads, 2 KV heads).
    #[must_use]
    pub fn gqa_test_handle() -> Self {
        Self::new(0, 80, 4, 4, 64, 256).with_gqa(2)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn default_handle_fields() {
        let h = InferHandle::default_test_handle();
        assert_eq!(h.n_layers, 4);
        assert_eq!(h.n_heads, 4);
        assert_eq!(h.n_kv_heads, 4);
        assert_eq!(h.head_dim, 64);
        assert_eq!(h.vocab_size, 256);
        assert_eq!(h.block_size, 16);
        assert_eq!(h.max_seq_len, 4096);
    }

    #[test]
    fn gqa_handle() {
        let h = InferHandle::gqa_test_handle();
        assert_eq!(h.n_heads, 4);
        assert_eq!(h.n_kv_heads, 2);
    }

    #[test]
    fn kv_block_elements() {
        let h = InferHandle::default_test_handle();
        // block_size=16, n_kv_heads=4, head_dim=64
        assert_eq!(h.kv_block_elements(), 16 * 4 * 64);
    }

    #[test]
    fn kv_block_bytes() {
        let h = InferHandle::default_test_handle();
        assert_eq!(h.kv_block_bytes(), 2 * 16 * 4 * 64 * 4);
    }

    #[test]
    fn attention_scale() {
        let h = InferHandle::default_test_handle();
        assert_abs_diff_eq!(h.attention_scale(), 1.0 / 8.0, epsilon = 1e-6);
    }

    #[test]
    fn ptx_version_str() {
        assert_eq!(
            InferHandle::new(0, 80, 1, 1, 32, 64).ptx_version_str(),
            "8.0"
        );
        assert_eq!(
            InferHandle::new(0, 90, 1, 1, 32, 64).ptx_version_str(),
            "8.4"
        );
        assert_eq!(
            InferHandle::new(0, 100, 1, 1, 32, 64).ptx_version_str(),
            "8.7"
        );
        assert_eq!(
            InferHandle::new(0, 75, 1, 1, 32, 64).ptx_version_str(),
            "7.5"
        );
    }

    #[test]
    fn with_block_size_and_max_seq_len() {
        let h = InferHandle::default_test_handle()
            .with_block_size(32)
            .with_max_seq_len(8192);
        assert_eq!(h.block_size, 32);
        assert_eq!(h.max_seq_len, 8192);
    }
}
