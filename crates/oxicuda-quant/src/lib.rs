//! # oxicuda-quant — GPU-Accelerated Quantization & Model Compression Engine
//!
//! `oxicuda-quant` provides a comprehensive suite of post-training quantization
//! (PTQ), quantization-aware training (QAT), pruning, knowledge distillation,
//! and mixed-precision analysis tools.
//!
//! ## Feature overview
//!
//! | Category    | Highlights                                                  |
//! |-------------|-------------------------------------------------------------|
//! | Schemes     | MinMax INT4/8, NF4 (QLoRA), FP8 E4M3/E5M2, GPTQ, SmoothQuant |
//! | QAT         | MinMax / MovingAvg / Histogram observers, FakeQuantize (STE) |
//! | Pruning     | Magnitude unstructured, channel / filter / head structured  |
//! | Distillation| KL / MSE / cosine response + feature distillation           |
//! | Analysis    | Layer sensitivity, compression metrics, mixed-precision policy |
//! | GPU kernels | PTX kernels for fake-quant, INT8 quant/dequant, NF4, pruning |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! # use oxicuda_quant::scheme::minmax::{MinMaxQuantizer, QuantScheme, QuantGranularity};
//! let q = MinMaxQuantizer::int8_symmetric();
//! let data = vec![-1.0_f32, 0.0, 0.5, 1.0];
//! let params = q.calibrate(&data).unwrap();
//! let codes  = q.quantize(&data, &params).unwrap();
//! let deq    = q.dequantize(&codes, &params);
//! ```

// ─── Lints ───────────────────────────────────────────────────────────────────

#![allow(clippy::module_name_repetitions)]

// ─── Modules ─────────────────────────────────────────────────────────────────

pub mod analysis;
pub mod distill;
pub mod error;
pub mod pruning;
pub mod qat;
pub mod scheme;

/// PTX kernel source strings for GPU-side quantization operations.
pub mod ptx_kernels;

// ─── Top-level re-exports ────────────────────────────────────────────────────

pub use error::{QuantError, QuantResult};
