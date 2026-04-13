//! Mixture of Experts (MoE) module.
//!
//! Provides GPU-accelerated MoE primitives for transformer models such as
//! Mixtral, Switch Transformer, and GShard. The module implements:
//!
//! - **Routing** ([`routing`]) — top-k expert selection with fused softmax.
//! - **Permutation** ([`permute`]) — token scatter/gather by expert assignment.
//! - **Grouped GEMM** ([`grouped_gemm`]) — MoE-specific batched GEMM wrapper.
//! - **Fused MoE** ([`mod@fused_moe`]) — end-to-end fused kernel combining
//!   permute + GEMM + activation + GEMM + unpermute.
//! - **Auxiliary Loss** ([`aux_loss`]) — Switch Transformer style load-balancing
//!   and z-loss for training stability.
//! - **Capacity** ([`capacity`]) — expert capacity factor tuning with overflow
//!   masking and dynamic capacity adjustment.
//! - **Monitoring** ([`monitoring`]) — runtime expert utilization tracking and
//!   imbalance detection.
//!
//! # Architecture
//!
//! The MoE layer routes each input token to its top-k experts, executes
//! per-expert FFN layers (two linear projections with an activation in
//! between), and combines the expert outputs weighted by the routing scores.

pub mod aux_loss;
pub mod capacity;
pub mod fused_moe;
pub mod grouped_gemm;
pub mod monitoring;
pub mod permute;
pub mod routing;

pub use aux_loss::{AuxLossConfig, AuxLossPlan};
pub use capacity::{CapacityConfig, CapacityPlan};
pub use fused_moe::{Fp8Type, FusedMoeConfig, fused_moe};
pub use grouped_gemm::moe_grouped_gemm;
pub use monitoring::{ImbalanceLevel, MoeMonitor, MoeUtilizationReport};
pub use permute::{permute_tokens, unpermute_tokens};
pub use routing::{MoeConfig, moe_routing};
