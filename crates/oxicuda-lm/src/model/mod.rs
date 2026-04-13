//! Complete LLM model implementations.
//!
//! | Module | Model family |
//! |--------|--------------|
//! | [`gpt`]   | GPT-2 (token+positional embedding, LayerNorm, MLP FFN, weight-tied LM head) |
//! | [`llama`] | LLaMA-2/3, Mistral (RMSNorm, GQA, RoPE, SwiGLU, independent LM head) |
//! | [`weights`] | Weight loading utilities for both families |

pub mod gpt;
pub mod llama;
pub mod weights;

pub use gpt::Gpt2Model;
pub use llama::LlamaModel;
pub use weights::{load_gpt2_block, load_llama_block};
