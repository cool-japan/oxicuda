//! Precision-specific GEMM optimizations and configuration.
//!
//! Each submodule provides architecture-aware tile configurations for a
//! specific floating-point precision. The [`crate::level3::TileConfig`] struct describes
//! how a GEMM kernel should be tiled across shared memory, warps, and
//! pipeline stages for optimal throughput on a given GPU.
//!
//! Mixed-precision helpers (e.g. FP16 inputs with FP32 accumulators) live
//! in the `mixed` submodule.

mod f32_ops;
mod f64_ops;
pub mod fp4_fp6_ops;
mod fp8_ops;
pub mod int_ops;
mod mixed;
mod tf32_ops;

#[cfg(feature = "f16")]
mod bf16_ops;
#[cfg(feature = "f16")]
mod f16_ops;

pub use f32_ops::F32Config;
pub use f64_ops::F64Config;
pub use fp4_fp6_ops::{
    Fp4Format, Fp4GemmConfig, Fp4Quantizer, Fp6Format, Fp6GemmConfig, Fp6Quantizer,
    MicroScalingConfig, MicroScalingQuantizer, PackedFp4, PackedFp6, ScalingFormat,
    ScalingGranularity, SubByteAccumulator, generate_fp4_dequantize_ptx, generate_fp4_gemm_ptx,
    generate_fp6_dequantize_ptx, generate_fp6_gemm_ptx, select_fp4_tile, select_fp6_tile,
};
pub use fp8_ops::{Fp8Config, Fp8Format};
pub use int_ops::{
    AccType, Int4GemmConfig, Int8GemmConfig, generate_int4_gemm_ptx, generate_int8_gemm_ptx,
    pack_int4, unpack_int4,
};
pub use mixed::MixedPrecisionConfig;
pub use tf32_ops::Tf32Config;

#[cfg(feature = "f16")]
pub use bf16_ops::Bf16Config;
#[cfg(feature = "f16")]
pub use f16_ops::F16Config;
