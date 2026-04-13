//! GEMM (General Matrix Multiply) implementation.
//!
//! This module contains the core GEMM machinery:
//!
//! - [`dispatch`]: Kernel selection, compilation, and caching.
//! - [`simt`]: SIMT (non-Tensor-Core) GEMM helpers for small matrices.
//! - [`tensor_core`]: Tensor Core configuration and validation.
//! - [`splitk`]: Split-K parallelisation for tall-skinny K dimensions.
//! - [`cooperative`]: Multi-CTA cooperative GEMM with cluster synchronisation.
//! - [`epilogue`]: Post-GEMM fused operations (ReLU, GELU, bias, etc.).
//! - [`warp_specialized`]: Hopper+ warp-specialized producer/consumer GEMM.

pub mod bandwidth_opt;
pub mod cooperative;
pub mod dispatch;
pub mod epilogue;
pub mod fusion;
pub mod simt;
pub mod splitk;
pub mod tensor_core;
pub mod tiles;
pub mod warp_specialized;

pub use bandwidth_opt::{
    ArithmeticIntensityAnalysis, BandwidthGemmConfig, BandwidthPrecision, BandwidthStrategy,
    BandwidthTileConfig, analyze_intensity, is_bandwidth_limited, select_bandwidth_tiles,
};
pub use cooperative::{
    CoopGemmStats, CoopPrecision, CoopReductionStrategy, CoopWorkPartition, CooperativeGemmConfig,
    CooperativeGemmPlan,
};
pub use dispatch::{GemmCategory, GemmDispatcher, GemmProblem, TileConfig};
pub use epilogue::EpilogueOp;
pub use fusion::{
    FusedKernelPlan, FusedOp, FusiblePair, FusionOpportunity, FusionPass, FusionPlan, FusionStage,
    FusionStrategy, FusionType, GemmGraph, GemmNode, GemmOp, NodeInput, estimate_chain_flops,
    minimum_chain_flops, optimal_chain_order,
};
pub use simt::SimtGemmBuilder;
pub use splitk::SplitKConfig;
pub use tensor_core::TensorCoreValidator;
pub use tiles::{RectangularTile, TileSelector};
pub use warp_specialized::WarpSpecializedGemm;
