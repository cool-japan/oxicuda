//! FlashAttention GPU kernels for efficient attention computation.
//!
//! This module provides tiled, IO-aware attention implementations that
//! avoid materialising the full `[N, N]` attention matrix:
//!
//! | Sub-module  | Description                                       |
//! |-------------|---------------------------------------------------|
//! | [`forward`] | FlashAttention-2 forward pass                     |
//! | [`backward`]| FlashAttention-2 backward pass for training       |
//! | [`paged`]   | PagedAttention for KV-cache based LLM inference   |
//! | [`decode`]  | Single-query decode attention (non-paged)         |
//! | [`hopper`]  | FlashAttention-3 for Hopper+ GPUs (sm_90+)        |

pub mod backward;
pub mod decode;
pub mod forward;
pub mod hopper;
pub mod paged;
