//! Transformer layer building blocks.
//!
//! # Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`attention`] | Multi-head attention (MHA/GQA) with KV-cache |
//! | [`embedding`] | Token embedding, learned positional embedding, RoPE |
//! | [`ffn`]       | MLP (GELU) and SwiGLU feed-forward networks |
//! | [`norm`]      | RMSNorm and LayerNorm |
//! | [`transformer`] | GPT-2 and LLaMA transformer blocks; `PastKvCache` |

pub mod attention;
pub mod embedding;
pub mod ffn;
pub mod norm;
pub mod transformer;

pub use attention::{LayerKvCache, MultiHeadAttention};
pub use embedding::{LearnedPositionalEmbedding, RotaryEmbedding, TokenEmbedding};
pub use ffn::{MlpFfn, SwiGluFfn, gelu, silu};
pub use norm::{LayerNorm, RmsNorm};
pub use transformer::{GptBlock, LlamaBlock, PastKvCache};
