//! Quantization and dequantization operations for DNN.
//!
//! This module provides GPU-accelerated quantization primitives:
//!
//! - [`fp8_quantize`] — FP8 E4M3 quantization/dequantization.
//! - [`int8_quantize`] — INT8 symmetric quantization/dequantization.
//! - [`block_scale`] — Block-scaled quantization for Blackwell FP4.

pub mod block_scale;
pub mod fp8_quantize;
pub mod gptq_awq;
pub mod int4_quantize;
pub mod int8_quantize;
pub mod qat;

pub use block_scale::quantize_block_scaled;
pub use fp8_quantize::{dequantize_from_fp8, quantize_to_fp8};
pub use gptq_awq::{
    AwqChannelScales, AwqConfig, GptqConfig, GptqState, QuantMethodTag, QuantizedWeight,
    WeightQuantMethod, WeightQuantPlan,
};
pub use int4_quantize::{Int4QuantConfig, dequantize_int4, quantize_to_int4, quantize_to_nf4};
pub use int8_quantize::{BlockQuantizedInt8, dequantize_from_int8, quantize_to_int8};
pub use qat::{FakeQuantize, QatBitWidth, QatConfig, QuantParams};
