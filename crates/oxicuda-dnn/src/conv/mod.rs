//! Convolution operations for deep learning.
//!
//! This module provides GPU-accelerated convolution primitives covering
//! forward propagation (fprop), backward data gradient (dgrad), and
//! backward filter gradient (wgrad). Multiple algorithms are supported
//! and automatically selected based on problem dimensions and target
//! architecture.
//!
//! # Algorithms
//!
//! | Algorithm | Best for | Workspace |
//! |-----------|----------|-----------|
//! | [`ImplicitGemm`](fprop::implicit_gemm) | General-purpose | None |
//! | [`Im2colGemm`](fprop::im2col_gemm) | Medium feature maps | Yes |
//! | [`Winograd`](fprop::winograd) | 3x3 kernels, stride 1 | Yes |
//! | [`Direct`](fprop::direct) | 1x1 and depthwise | None |
//! | [`FftConv2d`](fft_conv::FftConv2dPlan) | Large kernels (7x7+) | Yes |
//!
//! # Public API
//!
//! Use the functions in [`api`] for high-level convolution operations:
//!
//! - [`conv_forward`] — forward convolution
//! - [`conv_backward_data`] — gradient w.r.t. input
//! - [`conv_backward_filter`] — gradient w.r.t. weights
//! - [`conv_bn_relu`] — fused convolution + batch norm + ReLU

pub mod algo_select;
pub mod api;
pub mod conv3d;
pub mod deformable;
pub mod depthwise_separable;
pub mod descriptor;
pub mod dgrad;
pub mod fft_conv;
pub mod fprop;
pub mod fused;
pub mod transpose_conv;
pub mod wgrad;

pub use api::{conv_backward_data, conv_backward_filter, conv_bn_relu, conv_forward};
pub use conv3d::{
    Conv3dAlgorithm, Conv3dConfig, Conv3dPlan, generate_col2im3d_ptx, generate_direct3d_ptx,
    generate_im2col3d_ptx, generate_wgrad3d_ptx,
};
pub use deformable::{
    DeformableConvConfig, DeformableConvPlan, generate_deformable_conv_backward_input_ptx,
    generate_deformable_conv_backward_offset_ptx, generate_deformable_conv_backward_weight_ptx,
    generate_deformable_conv_forward_ptx,
};
pub use depthwise_separable::{
    ActivationType, DepthwiseSeparableConfig, DepthwiseSeparablePlan, generate_depthwise_conv_ptx,
    generate_fused_dw_pw_ptx, generate_pointwise_conv_ptx,
};
pub use descriptor::ConvProblem;
pub use fft_conv::FftConv2dPlan;
pub use transpose_conv::{
    TransposeConvConfig, TransposeConvPlan, generate_col2im_ptx, generate_weight_reshape_ptx,
};
