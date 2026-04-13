//! Tests for deformable convolution (DCNv2).

use super::*;
use crate::error::DnnResult;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

/// Helper to create a basic 3x3 DCNv2 config for testing.
fn basic_config_3x3() -> DeformableConvConfig {
    DeformableConvConfig {
        in_channels: 64,
        out_channels: 64,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        offset_groups: 1,
        use_modulation: true,
        sm_version: SmVersion::Sm80,
        float_type: PtxType::F32,
    }
}

/// Helper to create a 5x5 DCNv2 config.
fn basic_config_5x5() -> DeformableConvConfig {
    DeformableConvConfig {
        kernel_h: 5,
        kernel_w: 5,
        pad_h: 2,
        pad_w: 2,
        ..basic_config_3x3()
    }
}

/// Helper to create a DCNv1 config (no modulation).
fn dcnv1_config() -> DeformableConvConfig {
    DeformableConvConfig {
        use_modulation: false,
        ..basic_config_3x3()
    }
}

// ---------------------------------------------------------------------------
// Config validation tests
// ---------------------------------------------------------------------------

#[test]
fn validate_valid_config() {
    let cfg = basic_config_3x3();
    assert!(cfg.validate().is_ok());
}

#[test]
fn validate_zero_kernel() {
    let mut cfg = basic_config_3x3();
    cfg.kernel_h = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_zero_stride() {
    let mut cfg = basic_config_3x3();
    cfg.stride_h = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_zero_dilation() {
    let mut cfg = basic_config_3x3();
    cfg.dilation_h = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_zero_channels() {
    let mut cfg = basic_config_3x3();
    cfg.in_channels = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_zero_offset_groups() {
    let mut cfg = basic_config_3x3();
    cfg.offset_groups = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_indivisible_offset_groups() {
    let mut cfg = basic_config_3x3();
    cfg.in_channels = 64;
    cfg.offset_groups = 3; // 64 % 3 != 0
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_unsupported_float_type() {
    let mut cfg = basic_config_3x3();
    cfg.float_type = PtxType::F64;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_f16_accepted() {
    let mut cfg = basic_config_3x3();
    cfg.float_type = PtxType::F16;
    assert!(cfg.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Output size tests
// ---------------------------------------------------------------------------

#[test]
fn output_size_same_padding_3x3() {
    let cfg = basic_config_3x3();
    let (oh, ow) = cfg.output_size(16, 16);
    // (16 + 2*1 - 1*(3-1) - 1) / 1 + 1 = (16 + 2 - 2 - 1) / 1 + 1 = 16
    assert_eq!(oh, 16);
    assert_eq!(ow, 16);
}

#[test]
fn output_size_stride2() {
    let mut cfg = basic_config_3x3();
    cfg.stride_h = 2;
    cfg.stride_w = 2;
    let (oh, ow) = cfg.output_size(16, 16);
    // (16 + 2 - 2 - 1) / 2 + 1 = 15 / 2 + 1 = 7 + 1 = 8
    assert_eq!(oh, 8);
    assert_eq!(ow, 8);
}

#[test]
fn output_size_dilation2() {
    let mut cfg = basic_config_3x3();
    cfg.dilation_h = 2;
    cfg.dilation_w = 2;
    cfg.pad_h = 2;
    cfg.pad_w = 2;
    let (oh, ow) = cfg.output_size(16, 16);
    // effective_k = 2*(3-1)+1 = 5
    // (16 + 4 - 5) / 1 + 1 = 16
    assert_eq!(oh, 16);
    assert_eq!(ow, 16);
}

#[test]
fn output_size_no_padding() {
    let mut cfg = basic_config_3x3();
    cfg.pad_h = 0;
    cfg.pad_w = 0;
    let (oh, ow) = cfg.output_size(16, 16);
    // (16 + 0 - 2 - 1) / 1 + 1 = 14
    assert_eq!(oh, 14);
    assert_eq!(ow, 14);
}

#[test]
fn output_size_5x5() {
    let cfg = basic_config_5x5();
    let (oh, ow) = cfg.output_size(16, 16);
    // (16 + 4 - 4 - 1) / 1 + 1 = 16
    assert_eq!(oh, 16);
    assert_eq!(ow, 16);
}

#[test]
fn output_size_stride2_dilation2() {
    let mut cfg = basic_config_3x3();
    cfg.stride_h = 2;
    cfg.stride_w = 2;
    cfg.dilation_h = 2;
    cfg.dilation_w = 2;
    cfg.pad_h = 2;
    cfg.pad_w = 2;
    let (oh, ow) = cfg.output_size(16, 16);
    // effective_k = 5; (16 + 4 - 5) / 2 + 1 = 15/2 + 1 = 7 + 1 = 8
    assert_eq!(oh, 8);
    assert_eq!(ow, 8);
}

// ---------------------------------------------------------------------------
// Derived dimensions tests
// ---------------------------------------------------------------------------

#[test]
fn offset_channels_calculation() {
    let cfg = basic_config_3x3();
    // 2 * 3 * 3 * 1 = 18
    assert_eq!(cfg.offset_channels(), 18);
}

#[test]
fn mask_channels_calculation() {
    let cfg = basic_config_3x3();
    // 3 * 3 * 1 = 9
    assert_eq!(cfg.mask_channels(), 9);
}

#[test]
fn channels_per_offset_group_calculation() {
    let cfg = basic_config_3x3();
    assert_eq!(cfg.channels_per_offset_group(), 64);

    let mut cfg2 = basic_config_3x3();
    cfg2.offset_groups = 4;
    assert_eq!(cfg2.channels_per_offset_group(), 16);
}

#[test]
fn effective_kernel_size() {
    let cfg = basic_config_3x3();
    assert_eq!(cfg.effective_kernel_h(), 3);
    assert_eq!(cfg.effective_kernel_w(), 3);

    let mut cfg2 = basic_config_3x3();
    cfg2.dilation_h = 2;
    cfg2.dilation_w = 3;
    assert_eq!(cfg2.effective_kernel_h(), 5);
    assert_eq!(cfg2.effective_kernel_w(), 7);
}

// ---------------------------------------------------------------------------
// Plan creation tests
// ---------------------------------------------------------------------------

#[test]
fn plan_creation_valid() {
    let cfg = basic_config_3x3();
    let plan = DeformableConvPlan::new(cfg);
    assert!(plan.is_ok());
}

#[test]
fn plan_creation_invalid_config() {
    let mut cfg = basic_config_3x3();
    cfg.kernel_h = 0;
    let plan = DeformableConvPlan::new(cfg);
    assert!(plan.is_err());
}

#[test]
fn plan_output_size() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    let plan = DeformableConvPlan::new(cfg)?;
    let (oh, ow) = plan.output_size(16, 16);
    assert_eq!(oh, 16);
    assert_eq!(ow, 16);
    Ok(())
}

// ---------------------------------------------------------------------------
// Forward PTX generation tests
// ---------------------------------------------------------------------------

#[test]
fn forward_ptx_3x3_f32() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_forward()?;
    assert!(!ptx.is_empty());
    assert!(ptx.contains("deformable_conv_forward_f32_3x3"));
    assert!(ptx.contains(".entry"));
    assert!(ptx.contains("bilinear") || ptx.contains("Deformable"));
    Ok(())
}

#[test]
fn forward_ptx_5x5_f32() -> DnnResult<()> {
    let cfg = basic_config_5x5();
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_forward()?;
    assert!(!ptx.is_empty());
    assert!(ptx.contains("deformable_conv_forward_f32_5x5"));
    Ok(())
}

#[test]
fn forward_ptx_f16() -> DnnResult<()> {
    let mut cfg = basic_config_3x3();
    cfg.float_type = PtxType::F16;
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_forward()?;
    assert!(!ptx.is_empty());
    assert!(ptx.contains("deformable_conv_forward_f16_3x3"));
    Ok(())
}

#[test]
fn forward_ptx_dcnv1_no_modulation() -> DnnResult<()> {
    let cfg = dcnv1_config();
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_forward()?;
    assert!(!ptx.is_empty());
    // DCNv1 should not contain mask loading code
    assert!(!ptx.contains("modulation mask"));
    Ok(())
}

#[test]
fn forward_ptx_dcnv2_with_modulation() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    assert!(cfg.use_modulation);
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_forward()?;
    assert!(!ptx.is_empty());
    // DCNv2 should contain mask handling
    assert!(ptx.contains("modulation") || ptx.contains("mask"));
    Ok(())
}

#[test]
fn forward_ptx_multiple_offset_groups() -> DnnResult<()> {
    let mut cfg = basic_config_3x3();
    cfg.offset_groups = 4;
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_forward()?;
    assert!(!ptx.is_empty());
    Ok(())
}

// ---------------------------------------------------------------------------
// Backward PTX generation tests
// ---------------------------------------------------------------------------

#[test]
fn backward_input_ptx_3x3() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_backward_input()?;
    assert!(!ptx.is_empty());
    assert!(ptx.contains("backward_input"));
    Ok(())
}

#[test]
fn backward_offset_ptx_3x3() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_backward_offset()?;
    assert!(!ptx.is_empty());
    assert!(ptx.contains("backward_offset"));
    Ok(())
}

#[test]
fn backward_weight_ptx_3x3() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_backward_weight()?;
    assert!(!ptx.is_empty());
    assert!(ptx.contains("backward_weight"));
    Ok(())
}

#[test]
fn backward_input_f16() -> DnnResult<()> {
    let mut cfg = basic_config_3x3();
    cfg.float_type = PtxType::F16;
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_backward_input()?;
    assert!(!ptx.is_empty());
    assert!(ptx.contains("f16"));
    Ok(())
}

#[test]
fn backward_weight_5x5() -> DnnResult<()> {
    let cfg = basic_config_5x5();
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_backward_weight()?;
    assert!(!ptx.is_empty());
    assert!(ptx.contains("5x5"));
    Ok(())
}

// ---------------------------------------------------------------------------
// Convenience function tests
// ---------------------------------------------------------------------------

#[test]
fn convenience_forward_generates_ptx() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    let ptx = generate_deformable_conv_forward_ptx(&cfg)?;
    assert!(!ptx.is_empty());
    Ok(())
}

#[test]
fn convenience_backward_input_generates_ptx() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    let ptx = generate_deformable_conv_backward_input_ptx(&cfg)?;
    assert!(!ptx.is_empty());
    Ok(())
}

#[test]
fn convenience_backward_offset_generates_ptx() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    let ptx = generate_deformable_conv_backward_offset_ptx(&cfg)?;
    assert!(!ptx.is_empty());
    Ok(())
}

#[test]
fn convenience_backward_weight_generates_ptx() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    let ptx = generate_deformable_conv_backward_weight_ptx(&cfg)?;
    assert!(!ptx.is_empty());
    Ok(())
}

// ---------------------------------------------------------------------------
// Bilinear interpolation correctness in PTX
// ---------------------------------------------------------------------------

#[test]
fn forward_ptx_contains_bilinear_ops() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_forward()?;
    // Should contain floor operation for bilinear interpolation
    assert!(ptx.contains("cvt.rmi"));
    // Should contain 4-corner sampling (tl, tr, bl, br pattern)
    assert!(ptx.contains("mul.rn.f32"));
    assert!(ptx.contains("add.rn.f32"));
    Ok(())
}

#[test]
fn forward_ptx_loads_offsets() -> DnnResult<()> {
    let cfg = basic_config_3x3();
    let plan = DeformableConvPlan::new(cfg)?;
    let ptx = plan.generate_forward()?;
    // Must load from global memory (offset tensor)
    assert!(ptx.contains("ld.global"));
    // Must store to global memory (output tensor)
    assert!(ptx.contains("st.global"));
    Ok(())
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn output_size_tiny_input() {
    let cfg = basic_config_3x3();
    let (oh, ow) = cfg.output_size(1, 1);
    // (1 + 2 - 2 - 1) / 1 + 1 = 1
    assert_eq!(oh, 1);
    assert_eq!(ow, 1);
}

#[test]
fn output_size_input_smaller_than_kernel_no_padding() {
    let mut cfg = basic_config_3x3();
    cfg.pad_h = 0;
    cfg.pad_w = 0;
    let (oh, ow) = cfg.output_size(2, 2);
    // (2 + 0 - 2 - 1) / 1 + 1 = 0; padded_h (2) < effective_kh (3)
    assert_eq!(oh, 0);
    assert_eq!(ow, 0);
}

#[test]
fn plan_with_offset_groups_equal_in_channels() -> DnnResult<()> {
    let mut cfg = basic_config_3x3();
    cfg.offset_groups = cfg.in_channels;
    let plan = DeformableConvPlan::new(cfg)?;
    assert_eq!(plan.config().channels_per_offset_group(), 1);
    let ptx = plan.generate_forward()?;
    assert!(!ptx.is_empty());
    Ok(())
}
