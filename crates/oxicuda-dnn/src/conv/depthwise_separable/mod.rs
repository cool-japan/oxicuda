//! Fused depthwise separable convolution.
//!
//! Implements the MobileNet / EfficientNet pattern of depthwise + pointwise
//! convolution with optional batch normalisation and activation fusion.
//!
//! # Architecture
//!
//! A depthwise separable convolution factors a standard convolution into:
//!
//! 1. **Depthwise convolution** — one filter per input channel (groups = C).
//! 2. **Pointwise convolution** — 1×1 convolution mixing channels.
//!
//! Fusing both stages eliminates the intermediate tensor write/read, saving
//! memory bandwidth. For small spatial sizes the two stages can even be
//! executed inside a single kernel launch.
//!
//! # Activation functions
//!
//! [`ActivationType`] covers the activations most commonly paired with
//! separable convolutions in mobile-class networks: ReLU, ReLU6, SiLU,
//! and HardSwish.

mod depthwise;
mod fused;
mod helpers;
mod pointwise;
mod types;

pub use depthwise::generate_depthwise_conv_ptx;
pub use fused::generate_fused_dw_pw_ptx;
pub use pointwise::generate_pointwise_conv_ptx;
pub use types::{ActivationType, DepthwiseSeparableConfig, DepthwiseSeparablePlan};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DnnResult;
    use oxicuda_ptx::arch::SmVersion;

    fn default_config() -> DepthwiseSeparableConfig {
        DepthwiseSeparableConfig {
            channels: 32,
            out_channels: 64,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            depth_multiplier: 1,
            depthwise_activation: ActivationType::None,
            pointwise_activation: ActivationType::None,
            depthwise_bn: false,
            pointwise_bn: false,
        }
    }

    // -----------------------------------------------------------------------
    // Config output_size
    // -----------------------------------------------------------------------

    #[test]
    fn config_output_size_basic() {
        let cfg = default_config();
        // 3x3 kernel, pad=1, stride=1 -> same spatial dim
        let (oh, ow) = cfg.output_size(16, 16);
        assert_eq!((oh, ow), (16, 16));
    }

    #[test]
    fn config_output_size_with_stride() {
        let mut cfg = default_config();
        cfg.stride_h = 2;
        cfg.stride_w = 2;
        // (16 + 2 - 3) / 2 + 1 = 8
        let (oh, ow) = cfg.output_size(16, 16);
        assert_eq!((oh, ow), (8, 8));
    }

    #[test]
    fn config_output_size_with_padding() {
        let mut cfg = default_config();
        cfg.pad_h = 0;
        cfg.pad_w = 0;
        // (16 + 0 - 3) / 1 + 1 = 14
        let (oh, ow) = cfg.output_size(16, 16);
        assert_eq!((oh, ow), (14, 14));
    }

    #[test]
    fn config_output_size_with_dilation() {
        let mut cfg = default_config();
        cfg.dilation_h = 2;
        cfg.dilation_w = 2;
        cfg.pad_h = 2;
        cfg.pad_w = 2;
        // effective_kh = 2*(3-1)+1 = 5, padded_h = 16+4 = 20
        // out_h = (20 - 5) / 1 + 1 = 16
        let (oh, ow) = cfg.output_size(16, 16);
        assert_eq!((oh, ow), (16, 16));
    }

    #[test]
    fn config_output_size_asymmetric() {
        let mut cfg = default_config();
        cfg.kernel_h = 5;
        cfg.kernel_w = 3;
        cfg.pad_h = 2;
        cfg.pad_w = 1;
        cfg.stride_h = 2;
        cfg.stride_w = 1;
        // out_h = (32 + 4 - 5) / 2 + 1 = 16
        // out_w = (24 + 2 - 3) / 1 + 1 = 24
        let (oh, ow) = cfg.output_size(32, 24);
        assert_eq!((oh, ow), (16, 24));
    }

    #[test]
    fn config_output_size_zero_when_kernel_exceeds_input() {
        let mut cfg = default_config();
        cfg.kernel_h = 5;
        cfg.kernel_w = 5;
        cfg.pad_h = 0;
        cfg.pad_w = 0;
        // 2 + 0 - 5 < 0 -> 0
        let (oh, ow) = cfg.output_size(2, 2);
        assert_eq!((oh, ow), (0, 0));
    }

    // -----------------------------------------------------------------------
    // Config validate
    // -----------------------------------------------------------------------

    #[test]
    fn config_validate_kernel_gt_zero() {
        let mut cfg = default_config();
        cfg.kernel_h = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_stride_gt_zero() {
        let mut cfg = default_config();
        cfg.stride_h = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_channels_gt_zero() {
        let mut cfg = default_config();
        cfg.channels = 0;
        assert!(cfg.validate().is_err());

        let mut cfg2 = default_config();
        cfg2.out_channels = 0;
        assert!(cfg2.validate().is_err());
    }

    #[test]
    fn config_validate_depth_multiplier_gt_zero() {
        let mut cfg = default_config();
        cfg.depth_multiplier = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_valid_passes() {
        let cfg = default_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_depthwise_out_channels() {
        let mut cfg = default_config();
        assert_eq!(cfg.depthwise_out_channels(), 32);

        cfg.depth_multiplier = 4;
        assert_eq!(cfg.depthwise_out_channels(), 128);
    }

    #[test]
    fn config_validate_dilation_gt_zero() {
        let mut cfg = default_config();
        cfg.dilation_h = 0;
        assert!(cfg.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // Plan
    // -----------------------------------------------------------------------

    #[test]
    fn plan_creation() -> DnnResult<()> {
        let cfg = default_config();
        let p = DepthwiseSeparablePlan::create(cfg, 2, 16, 16)?;
        assert_eq!(p.output_h, 16);
        assert_eq!(p.output_w, 16);
        assert_eq!(p.batch_size, 2);
        Ok(())
    }

    #[test]
    fn plan_workspace_size() -> DnnResult<()> {
        let cfg = default_config();
        let plan = DepthwiseSeparablePlan::create(cfg, 1, 16, 16)?;
        // 32 * 16 * 16 * 4 = 32768 < 49152 -> fully fused
        assert!(plan.is_fully_fused);
        assert_eq!(plan.workspace_size(), 0);
        Ok(())
    }

    #[test]
    fn plan_reject_zero_batch() {
        let cfg = default_config();
        let plan = DepthwiseSeparablePlan::create(cfg, 0, 16, 16);
        assert!(plan.is_err());
    }

    #[test]
    fn plan_reject_zero_spatial() {
        let cfg = default_config();
        assert!(DepthwiseSeparablePlan::create(cfg.clone(), 1, 0, 16).is_err());
        assert!(DepthwiseSeparablePlan::create(cfg, 1, 16, 0).is_err());
    }

    #[test]
    fn plan_fusion_detection_large() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.channels = 256;
        cfg.depth_multiplier = 1;
        // 256 * 64 * 64 * 4 = 4,194,304 >> 48 KiB -> not fused
        let plan = DepthwiseSeparablePlan::create(cfg, 1, 64, 64)?;
        assert!(!plan.is_fully_fused);
        assert!(plan.workspace_size() > 0);
        Ok(())
    }

    #[test]
    fn plan_fusion_detection_small() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.channels = 8;
        cfg.depth_multiplier = 1;
        // 8 * 4 * 4 * 4 = 512 <= 48 KiB -> fused
        let plan = DepthwiseSeparablePlan::create(cfg, 1, 4, 4)?;
        assert!(plan.is_fully_fused);
        assert_eq!(plan.workspace_size(), 0);
        Ok(())
    }

    #[test]
    fn plan_workspace_bytes_match_intermediate_size() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.channels = 256;
        cfg.depth_multiplier = 1;
        cfg.out_channels = 512;
        let batch = 4;
        let plan = DepthwiseSeparablePlan::create(cfg, batch, 64, 64)?;
        assert!(!plan.is_fully_fused);
        // workspace = batch * dw_out_channels * out_h * out_w * sizeof(f32)
        let expected = batch * 256 * 64 * 64 * 4;
        assert_eq!(plan.workspace_size(), expected);
        Ok(())
    }

    #[test]
    fn plan_with_depth_multiplier() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.channels = 16;
        cfg.depth_multiplier = 4;
        cfg.out_channels = 128;
        let plan = DepthwiseSeparablePlan::create(cfg, 1, 8, 8)?;
        assert_eq!(plan.config.depthwise_out_channels(), 64);
        assert_eq!(plan.output_h, 8);
        assert_eq!(plan.output_w, 8);
        Ok(())
    }

    #[test]
    fn plan_reject_zero_output() {
        // kernel too large for input with no padding -> output = 0
        let mut cfg = default_config();
        cfg.kernel_h = 7;
        cfg.kernel_w = 7;
        cfg.pad_h = 0;
        cfg.pad_w = 0;
        let result = DepthwiseSeparablePlan::create(cfg, 1, 3, 3);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // PTX Generation
    // -----------------------------------------------------------------------

    #[test]
    fn depthwise_conv_ptx_generation_f32() -> DnnResult<()> {
        let cfg = default_config();
        let text = generate_depthwise_conv_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains(".target sm_80"));
        assert!(text.contains("depthwise_separable_dw_identity_f32"));
        Ok(())
    }

    #[test]
    fn depthwise_conv_ptx_with_relu() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.depthwise_activation = ActivationType::Relu;
        let text = generate_depthwise_conv_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("depthwise_separable_dw_relu_f32"));
        assert!(text.contains("max.f32"));
        Ok(())
    }

    #[test]
    fn depthwise_conv_ptx_with_relu6() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.depthwise_activation = ActivationType::Relu6;
        let text = generate_depthwise_conv_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("depthwise_separable_dw_relu6_f32"));
        assert!(text.contains("max.f32"));
        assert!(text.contains("min.f32"));
        Ok(())
    }

    #[test]
    fn depthwise_conv_ptx_with_silu() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.depthwise_activation = ActivationType::Silu;
        let text = generate_depthwise_conv_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("depthwise_separable_dw_silu_f32"));
        assert!(text.contains("ex2.approx"));
        Ok(())
    }

    #[test]
    fn depthwise_conv_ptx_with_hardswish() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.depthwise_activation = ActivationType::HardSwish;
        let text = generate_depthwise_conv_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("depthwise_separable_dw_hardswish_f32"));
        assert!(text.contains("div.rn.f32"));
        Ok(())
    }

    #[test]
    fn depthwise_conv_ptx_f64() -> DnnResult<()> {
        let cfg = default_config();
        let text = generate_depthwise_conv_ptx(&cfg, "f64", SmVersion::Sm80)?;
        assert!(text.contains("depthwise_separable_dw_identity_f64"));
        assert!(text.contains("ld.global.f64"));
        assert!(text.contains("st.global.f64"));
        Ok(())
    }

    #[test]
    fn depthwise_conv_ptx_with_bn() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.depthwise_bn = true;
        let text = generate_depthwise_conv_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("Batch Normalization"));
        assert!(text.contains("rsqrt.approx"));
        Ok(())
    }

    #[test]
    fn pointwise_conv_ptx_generation_f32() -> DnnResult<()> {
        let cfg = default_config();
        let text = generate_pointwise_conv_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains(".target sm_80"));
        assert!(text.contains("depthwise_separable_pw_identity_f32"));
        Ok(())
    }

    #[test]
    fn pointwise_conv_ptx_with_relu() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.pointwise_activation = ActivationType::Relu;
        let text = generate_pointwise_conv_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("depthwise_separable_pw_relu_f32"));
        assert!(text.contains("max.f32"));
        Ok(())
    }

    #[test]
    fn pointwise_conv_ptx_f64() -> DnnResult<()> {
        let cfg = default_config();
        let text = generate_pointwise_conv_ptx(&cfg, "f64", SmVersion::Sm80)?;
        assert!(text.contains("depthwise_separable_pw_identity_f64"));
        assert!(text.contains("ld.global.f64"));
        Ok(())
    }

    #[test]
    fn pointwise_conv_ptx_with_bn() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.pointwise_bn = true;
        let text = generate_pointwise_conv_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("Batch Normalization"));
        Ok(())
    }

    #[test]
    fn fused_dw_pw_ptx_generation() -> DnnResult<()> {
        let cfg = default_config();
        let text = generate_fused_dw_pw_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("fused_dw_pw_identity_identity_f32"));
        assert!(text.contains("bar.sync"));
        Ok(())
    }

    #[test]
    fn fused_dw_pw_ptx_with_activations() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.depthwise_activation = ActivationType::Relu;
        cfg.pointwise_activation = ActivationType::Silu;
        let text = generate_fused_dw_pw_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("fused_dw_pw_relu_silu_f32"));
        // Should have both max (relu) and ex2 (silu)
        assert!(text.contains("max.f32"));
        assert!(text.contains("ex2.approx"));
        Ok(())
    }

    #[test]
    fn fused_dw_pw_ptx_f64() -> DnnResult<()> {
        let cfg = default_config();
        let text = generate_fused_dw_pw_ptx(&cfg, "f64", SmVersion::Sm80)?;
        assert!(text.contains("fused_dw_pw_identity_identity_f64"));
        assert!(text.contains("fma.rn.f64"));
        assert!(text.contains("st.global.f64"));
        Ok(())
    }

    #[test]
    fn fused_dw_pw_ptx_contains_depthwise_and_pointwise_logic() -> DnnResult<()> {
        let cfg = default_config();
        let text = generate_fused_dw_pw_ptx(&cfg, "f32", SmVersion::Sm80)?;
        // Should contain depthwise conv comments/logic
        assert!(text.contains("depthwise"));
        // Should contain pointwise accumulation
        assert!(text.contains("pointwise") || text.contains("Pointwise") || text.contains("PW"));
        // Should contain fma for both stages
        assert!(text.contains("fma.rn.f32"));
        // Should contain global stores for final output
        assert!(text.contains("st.global.f32"));
        Ok(())
    }

    #[test]
    fn fused_dw_pw_ptx_1x1_kernel() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.kernel_h = 1;
        cfg.kernel_w = 1;
        cfg.pad_h = 0;
        cfg.pad_w = 0;
        let text = generate_fused_dw_pw_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("fused_dw_pw_identity_identity_f32"));
        Ok(())
    }

    #[test]
    fn fused_dw_pw_ptx_5x5_kernel() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.kernel_h = 5;
        cfg.kernel_w = 5;
        cfg.pad_h = 2;
        cfg.pad_w = 2;
        let text = generate_fused_dw_pw_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("fused_dw_pw_identity_identity_f32"));
        // 5x5 kernel generates skip labels per spatial position
        let skip_count = text.matches("fused_dw_skip_kh").count();
        assert!(
            skip_count >= 25,
            "expected at least 25 skip labels for 5x5, got {skip_count}"
        );
        Ok(())
    }

    #[test]
    fn fused_dw_pw_ptx_with_bn_both_stages() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.depthwise_bn = true;
        cfg.pointwise_bn = true;
        // BN params are registered in the kernel but the fused body
        // does not call emit_bn_epilogue yet; ensure it at least
        // generates valid PTX with the extra params.
        let text = generate_fused_dw_pw_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("dw_bn_gamma"));
        assert!(text.contains("pw_bn_gamma"));
        Ok(())
    }

    #[test]
    fn activation_type_variants() {
        assert_eq!(ActivationType::None.kernel_suffix(), "identity");
        assert_eq!(ActivationType::Relu.kernel_suffix(), "relu");
        assert_eq!(ActivationType::Relu6.kernel_suffix(), "relu6");
        assert_eq!(ActivationType::Silu.kernel_suffix(), "silu");
        assert_eq!(ActivationType::HardSwish.kernel_suffix(), "hardswish");
    }

    #[test]
    fn ptx_target_directive_sm75() -> DnnResult<()> {
        let cfg = default_config();
        let text = generate_depthwise_conv_ptx(&cfg, "f32", SmVersion::Sm75)?;
        assert!(text.contains(".target sm_75"));
        Ok(())
    }

    #[test]
    fn ptx_target_directive_sm90() -> DnnResult<()> {
        let cfg = default_config();
        let text = generate_pointwise_conv_ptx(&cfg, "f32", SmVersion::Sm90)?;
        assert!(text.contains(".target sm_90"));
        Ok(())
    }

    #[test]
    fn depthwise_conv_ptx_unsupported_precision() {
        let cfg = default_config();
        let ptx = generate_depthwise_conv_ptx(&cfg, "f16", SmVersion::Sm80);
        assert!(ptx.is_err());
    }

    #[test]
    fn parse_precision_valid_and_invalid() -> DnnResult<()> {
        let (ty32, sz32) = helpers::parse_precision("f32")?;
        assert_eq!(ty32, oxicuda_ptx::ir::PtxType::F32);
        assert_eq!(sz32, 4);
        let (ty64, sz64) = helpers::parse_precision("f64")?;
        assert_eq!(ty64, oxicuda_ptx::ir::PtxType::F64);
        assert_eq!(sz64, 8);
        assert!(helpers::parse_precision("f16").is_err());
        assert!(helpers::parse_precision("").is_err());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Fused kernel correctness properties
    // -----------------------------------------------------------------------

    #[test]
    fn fused_ptx_has_channel_loop() -> DnnResult<()> {
        let cfg = default_config();
        let text = generate_fused_dw_pw_ptx(&cfg, "f32", SmVersion::Sm80)?;
        // The fused kernel should have a channel loop
        assert!(text.contains("fused_ch_loop"));
        assert!(text.contains("fused_ch_loop_end"));
        Ok(())
    }

    #[test]
    fn fused_ptx_hardswish_both_stages() -> DnnResult<()> {
        let mut cfg = default_config();
        cfg.depthwise_activation = ActivationType::HardSwish;
        cfg.pointwise_activation = ActivationType::HardSwish;
        let text = generate_fused_dw_pw_ptx(&cfg, "f32", SmVersion::Sm80)?;
        assert!(text.contains("fused_dw_pw_hardswish_hardswish_f32"));
        let div_count = text.matches("div.rn.f32").count();
        assert!(
            div_count >= 2,
            "expected at least 2 div.rn.f32, got {div_count}"
        );
        Ok(())
    }

    #[test]
    fn config_clone_and_debug() {
        let cfg = default_config();
        let cloned = cfg.clone();
        assert_eq!(cloned.channels, cfg.channels);
        let _s = format!("{:?}", cloned);
    }

    #[test]
    fn activation_type_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ActivationType::Relu);
        set.insert(ActivationType::Relu);
        assert_eq!(set.len(), 1);
        set.insert(ActivationType::Silu);
        assert_eq!(set.len(), 2);
    }
}
