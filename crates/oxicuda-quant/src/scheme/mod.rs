//! # Quantization Schemes
//!
//! This module exposes a suite of post-training quantization (PTQ) strategies:
//!
//! | Module        | Scheme                                      | Primary use |
//! |---------------|---------------------------------------------|-------------|
//! | `minmax`      | Min-Max calibration (INT4/INT8)             | General PTQ |
//! | `nf4`         | NormalFloat4 (QLoRA)                        | 4-bit weights |
//! | `fp8`         | FP8 E4M3 / E5M2 (Hopper / Blackwell)        | Training & inference |
//! | `gptq`        | GPTQ Hessian-guided quantization            | LLM weights |
//! | `smooth_quant`| SmoothQuant activation–weight migration     | LLM activations |

pub mod fp8;
pub mod gptq;
pub mod minmax;
pub mod nf4;
pub mod smooth_quant;

pub use fp8::{Fp8Codec, Fp8Format};
pub use gptq::{GptqConfig, GptqOutput, GptqQuantizer};
pub use minmax::{MinMaxQuantizer, QuantGranularity, QuantParams, QuantScheme};
pub use nf4::{NF4_LUT, Nf4Quantizer};
pub use smooth_quant::{SmoothQuantConfig, SmoothQuantMigrator};
