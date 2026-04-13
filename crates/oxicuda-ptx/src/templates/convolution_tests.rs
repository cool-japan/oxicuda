//! Tests for convolution PTX kernel templates.

use super::ConvolutionTemplate;
use crate::arch::SmVersion;
use crate::ir::PtxType;

fn default_template() -> ConvolutionTemplate {
    ConvolutionTemplate {
        in_channels: 3,
        out_channels: 64,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 1,
        sm_version: SmVersion::Sm80,
        float_type: PtxType::F32,
    }
}

// --- Output size ---

#[test]
fn output_size_3x3_pad1() {
    let t = default_template();
    let (oh, ow) = t.output_size(32, 32);
    assert_eq!(oh, 32);
    assert_eq!(ow, 32);
}

#[test]
fn output_size_no_pad() {
    let mut t = default_template();
    t.pad_h = 0;
    t.pad_w = 0;
    let (oh, ow) = t.output_size(32, 32);
    assert_eq!(oh, 30);
    assert_eq!(ow, 30);
}

#[test]
fn output_size_5x5_stride2() {
    let t = ConvolutionTemplate {
        in_channels: 3,
        out_channels: 64,
        kernel_h: 5,
        kernel_w: 5,
        stride_h: 2,
        stride_w: 2,
        pad_h: 2,
        pad_w: 2,
        dilation_h: 1,
        dilation_w: 1,
        groups: 1,
        sm_version: SmVersion::Sm80,
        float_type: PtxType::F32,
    };
    let (oh, ow) = t.output_size(224, 224);
    assert_eq!(oh, 112);
    assert_eq!(ow, 112);
}

#[test]
fn output_size_dilation() {
    let t = ConvolutionTemplate {
        in_channels: 64,
        out_channels: 64,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 2,
        pad_w: 2,
        dilation_h: 2,
        dilation_w: 2,
        groups: 1,
        sm_version: SmVersion::Sm80,
        float_type: PtxType::F32,
    };
    let (oh, ow) = t.output_size(32, 32);
    assert_eq!(oh, 32);
    assert_eq!(ow, 32);
}

// --- Kernel name ---

#[test]
fn kernel_name_standard() {
    let t = default_template();
    let name = t.kernel_name("im2col");
    assert_eq!(name, "conv2d_im2col_f32_ic3_oc64_k3x3");
}

#[test]
fn kernel_name_depthwise() {
    let t = ConvolutionTemplate {
        in_channels: 32,
        out_channels: 32,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 32,
        sm_version: SmVersion::Sm80,
        float_type: PtxType::F32,
    };
    let name = t.kernel_name("direct");
    assert_eq!(name, "conv2d_direct_f32_dw32_k3x3");
}

#[test]
fn kernel_name_grouped() {
    let t = ConvolutionTemplate {
        in_channels: 64,
        out_channels: 128,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 4,
        sm_version: SmVersion::Sm80,
        float_type: PtxType::F32,
    };
    let name = t.kernel_name("direct");
    assert_eq!(name, "conv2d_direct_f32_ic64_oc128_k3x3_g4");
}

// --- Im2col kernel ---

#[test]
fn im2col_3x3_generates_ptx() {
    let t = default_template();
    let ptx = t.generate_im2col_kernel().expect("im2col generation");
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".target sm_80"));
    assert!(ptx.contains("im2col"));
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));
    assert!(ptx.contains("$IM2COL_PAD"));
}

#[test]
fn im2col_5x5_generates_ptx() {
    let mut t = default_template();
    t.kernel_h = 5;
    t.kernel_w = 5;
    t.pad_h = 2;
    t.pad_w = 2;
    let ptx = t.generate_im2col_kernel().expect("5x5 im2col");
    assert!(ptx.contains("im2col"));
    assert!(ptx.contains(".f32"));
}

#[test]
fn im2col_1x1_generates_ptx() {
    let mut t = default_template();
    t.kernel_h = 1;
    t.kernel_w = 1;
    t.pad_h = 0;
    t.pad_w = 0;
    let ptx = t.generate_im2col_kernel().expect("1x1 im2col");
    assert!(ptx.contains("im2col"));
}

// --- Direct conv kernel ---

#[test]
fn direct_conv_generates_ptx() {
    let t = default_template();
    let ptx = t.generate_direct_conv_kernel().expect("direct conv");
    assert!(ptx.contains("direct"));
    assert!(ptx.contains("fma.rn.f32"));
    assert!(ptx.contains("$DIRECT_C_LOOP"));
    assert!(ptx.contains("$DIRECT_BIAS"));
}

#[test]
fn direct_conv_grouped() {
    let t = ConvolutionTemplate {
        in_channels: 64,
        out_channels: 128,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 4,
        sm_version: SmVersion::Sm80,
        float_type: PtxType::F32,
    };
    let ptx = t.generate_direct_conv_kernel().expect("grouped conv");
    assert!(ptx.contains("group_idx"));
    assert!(ptx.contains("div.u32"));
}

#[test]
fn direct_conv_depthwise() {
    let t = ConvolutionTemplate {
        in_channels: 32,
        out_channels: 32,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 32,
        sm_version: SmVersion::Sm80,
        float_type: PtxType::F32,
    };
    let ptx = t.generate_direct_conv_kernel().expect("depthwise conv");
    // Depthwise: groups > 1, channels_per_group = 1
    assert!(ptx.contains("group_idx"));
}

// --- 1x1 conv kernel ---

#[test]
fn conv_1x1_generates_ptx() {
    let mut t = default_template();
    t.kernel_h = 1;
    t.kernel_w = 1;
    t.pad_h = 0;
    t.pad_w = 0;
    let ptx = t.generate_1x1_conv_kernel().expect("1x1 conv");
    assert!(ptx.contains("1x1"));
    assert!(ptx.contains("fma.rn.f32"));
    assert!(ptx.contains("$CONV1X1_C_LOOP"));
    assert!(ptx.contains("$CONV1X1_BIAS"));
}

#[test]
fn conv_1x1_grouped() {
    let t = ConvolutionTemplate {
        in_channels: 64,
        out_channels: 64,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
        dilation_h: 1,
        dilation_w: 1,
        groups: 4,
        sm_version: SmVersion::Sm80,
        float_type: PtxType::F32,
    };
    let ptx = t.generate_1x1_conv_kernel().expect("1x1 grouped");
    assert!(ptx.contains("group_idx"));
}

// --- Backward data kernel ---

#[test]
fn backward_data_generates_ptx() {
    let t = default_template();
    let ptx = t.generate_backward_data_kernel().expect("bwd data");
    assert!(ptx.contains("bwd_data"));
    assert!(ptx.contains("fma.rn.f32"));
    assert!(ptx.contains("$BWD_DATA_OC_LOOP"));
    assert!(ptx.contains("$BWD_DATA_STORE"));
}

#[test]
fn backward_data_stride2() {
    let mut t = default_template();
    t.stride_h = 2;
    t.stride_w = 2;
    let ptx = t.generate_backward_data_kernel().expect("bwd data s2");
    // Stride > 1 generates divisibility checks
    assert!(ptx.contains("rem.u32"));
}

// --- Backward filter kernel ---

#[test]
fn backward_filter_generates_ptx() {
    let t = default_template();
    let ptx = t.generate_backward_filter_kernel().expect("bwd filter");
    assert!(ptx.contains("bwd_filter"));
    assert!(ptx.contains("fma.rn.f32"));
    assert!(ptx.contains("$BWD_FILTER_BATCH_LOOP"));
    assert!(ptx.contains("$BWD_FILTER_STORE"));
}

#[test]
fn backward_filter_grouped() {
    let t = ConvolutionTemplate {
        in_channels: 64,
        out_channels: 128,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 4,
        sm_version: SmVersion::Sm80,
        float_type: PtxType::F32,
    };
    let ptx = t
        .generate_backward_filter_kernel()
        .expect("bwd filter grouped");
    assert!(ptx.contains("group_idx"));
}

// --- Float type variations ---

#[test]
fn im2col_f16() {
    let mut t = default_template();
    t.float_type = PtxType::F16;
    let ptx = t.generate_im2col_kernel().expect("f16 im2col");
    assert!(ptx.contains(".f16"));
}

#[test]
fn direct_conv_f64() {
    let mut t = default_template();
    t.float_type = PtxType::F64;
    let ptx = t.generate_direct_conv_kernel().expect("f64 direct");
    assert!(ptx.contains(".f64"));
    assert!(ptx.contains("fma.rn.f64"));
}

#[test]
fn conv_1x1_f16() {
    let mut t = default_template();
    t.kernel_h = 1;
    t.kernel_w = 1;
    t.pad_h = 0;
    t.pad_w = 0;
    t.float_type = PtxType::F16;
    let ptx = t.generate_1x1_conv_kernel().expect("f16 1x1");
    assert!(ptx.contains(".f16"));
}

// --- Stride / dilation / padding combinations ---

#[test]
fn direct_conv_stride_dilation_pad() {
    let t = ConvolutionTemplate {
        in_channels: 16,
        out_channels: 32,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 2,
        stride_w: 2,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 2,
        dilation_w: 2,
        groups: 1,
        sm_version: SmVersion::Sm90,
        float_type: PtxType::F32,
    };
    let ptx = t
        .generate_direct_conv_kernel()
        .expect("stride+dilation conv");
    assert!(ptx.contains(".target sm_90"));
    assert!(ptx.contains("direct"));
}

// --- Validation errors ---

#[test]
fn validate_invalid_type() {
    let mut t = default_template();
    t.float_type = PtxType::U32;
    assert!(t.generate_im2col_kernel().is_err());
}

#[test]
fn validate_zero_channels() {
    let mut t = default_template();
    t.in_channels = 0;
    assert!(t.generate_direct_conv_kernel().is_err());
}

#[test]
fn validate_groups_not_divisible() {
    let mut t = default_template();
    t.groups = 2; // 3 % 2 != 0
    assert!(t.generate_direct_conv_kernel().is_err());
}

#[test]
fn validate_zero_kernel() {
    let mut t = default_template();
    t.kernel_h = 0;
    assert!(t.generate_im2col_kernel().is_err());
}

#[test]
fn validate_zero_stride() {
    let mut t = default_template();
    t.stride_h = 0;
    assert!(t.generate_direct_conv_kernel().is_err());
}

#[test]
fn validate_zero_dilation() {
    let mut t = default_template();
    t.dilation_h = 0;
    assert!(t.generate_backward_data_kernel().is_err());
}
