//! Normalization operations for DNN.
//!
//! This module implements GPU-accelerated normalization layers commonly used
//! in deep learning models:
//!
//! - [`fn@layer_norm`] -- Layer Normalization (Transformer default)
//! - [`fn@rms_norm`] / [`fused_add_rms_norm`] -- RMS Normalization (LLaMA, Gemma)
//! - [`batch_norm_forward`] -- Batch Normalization (CNN standard)
//! - [`fn@group_norm`] -- Group Normalization (per-group channel normalisation)
//! - [`instance_norm`] -- Instance Normalization (style transfer, image generation)
//! - [`scale_norm`] -- Scale Normalization (efficient transformers)
//! - [`power_norm`] -- Power Normalization (improved training stability)
//! - Fused normalization + activation patterns ([`fused_layer_norm_relu`],
//!   [`fused_rms_norm_silu`])

pub mod batch_norm;
pub mod fused_norm;
pub mod group_norm;
pub mod instance_norm;
pub mod layer_norm;
pub mod power_norm;
pub mod rms_norm;
pub mod scale_norm;

pub use batch_norm::batch_norm_forward;
pub use fused_norm::{fused_layer_norm_relu, fused_rms_norm_silu};
pub use group_norm::group_norm;
pub use instance_norm::{InstanceNormConfig, InstanceNormPlan};
pub use layer_norm::layer_norm;
pub use power_norm::{PowerNormConfig, PowerNormPlan};
pub use rms_norm::{fused_add_rms_norm, rms_norm};
pub use scale_norm::{ScaleNormConfig, ScaleNormPlan};
