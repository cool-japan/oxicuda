//! Executor backends for the OxiCUDA inference engine.
//!
//! * [`model_runner`]       — `ModelRunner` trait + `MockModelRunner` for tests.
//! * [`attention_backend`]  — CPU reference PagedAttention implementation.

pub mod attention_backend;
pub mod model_runner;

pub use attention_backend::{AttentionConfig, paged_attention_cpu};
pub use model_runner::{MockModelRunner, ModelRunner, RunnerStats};
