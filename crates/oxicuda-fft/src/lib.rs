//! # OxiCUDA FFT -- GPU-Accelerated FFT Operations
//!
//! This crate provides GPU-accelerated Fast Fourier Transform operations,
//! serving as a pure Rust equivalent to NVIDIA's cuFFT library.
//!
//! ## Architecture
//!
//! The crate is organized into the following modules:
//!
//! | Module        | Purpose                                              |
//! |---------------|------------------------------------------------------|
//! | [`types`]     | Core types: `Complex<T>`, `FftType`, `FftDirection`  |
//! | [`error`]     | Error types and `FftResult<T>` alias                 |
//! | [`plan`]      | FFT plan creation and strategy selection              |
//! | [`execute`]   | High-level `FftHandle` executor                      |
//! | [`transforms`]| C2C, R2C, C2R, 2-D, and 3-D transform dispatch      |
//! | [`kernels`]   | PTX kernel generators (Stockham, batch, transpose)   |
//! | [`radix`]     | Radix butterfly implementations (2, 4, 8, mixed)     |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxicuda_fft::prelude::*;
//!
//! // Create a 1-D C2C plan for 1024 elements
//! let plan = FftPlan::new_1d(1024, FftType::C2C, 1).expect("plan");
//!
//! // With a GPU context:
//! // let handle = FftHandle::new(&ctx)?;
//! // handle.execute(&plan, input, output, FftDirection::Forward)?;
//! ```

#![warn(clippy::all)]
#![warn(missing_docs)]

pub mod callbacks;
pub mod conv_fft;
pub mod error;
pub mod execute;
pub mod half_precision;
pub mod inverse_scaling;
pub mod kernels;
pub mod multi_gpu;
pub mod out_of_core;
pub mod plan;
pub mod pruned;
pub(crate) mod ptx_helpers;
pub mod radix;
pub mod transforms;
pub mod types;

pub use callbacks::{
    CallbackOp, CallbackType, FftCallbackConfig, FftCallbackPlan, WindowFunction,
    generate_load_callback_ptx, generate_store_callback_ptx, generate_window_ptx, plan_callbacks,
    validate_callback_config, window_coefficient,
};
pub use conv_fft::{
    ConvolutionMode, CrossCorrelationPlan, FftConv2dPlan, FftConvPlan, next_fft_size,
};
pub use error::{FftError, FftResult};
pub use execute::FftHandle;
pub use half_precision::{
    AccumulationMode, HalfPrecisionFftConfig, HalfPrecisionFftPlan, HalfRoundingMode,
};
pub use inverse_scaling::{
    ScalingConfig, ScalingMode, ScalingPlan, compute_scale_factor, plan_scaling,
    scaling_error_bound, validate_scaling_config,
};
pub use multi_gpu::{MultiGpuFftConfig, MultiGpuFftPlan, SlabDecomposition};
pub use out_of_core::{LargeFftConfig, OutOfCoreConfig, OutOfCorePlan, OutOfCoreStrategy};
pub use plan::FftPlan;
pub use pruned::{PruneMode, PrunedFftConfig, PrunedFftPlan, PrunedStage, plan_pruned_fft};
pub use types::{Complex, FftDirection, FftPrecision, FftType};

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::callbacks::{
        CallbackOp, CallbackType, FftCallbackConfig, FftCallbackPlan, WindowFunction,
        plan_callbacks, validate_callback_config, window_coefficient,
    };
    pub use crate::conv_fft::{
        ConvolutionMode, CrossCorrelationPlan, FftConv2dPlan, FftConvPlan, next_fft_size,
    };
    pub use crate::error::{FftError, FftResult};
    pub use crate::execute::FftHandle;
    pub use crate::half_precision::{
        AccumulationMode, HalfPrecisionFftConfig, HalfPrecisionFftPlan, HalfRoundingMode,
    };
    pub use crate::inverse_scaling::{ScalingConfig, ScalingMode, ScalingPlan};
    pub use crate::out_of_core::{OutOfCoreConfig, OutOfCorePlan, OutOfCoreStrategy};
    pub use crate::plan::FftPlan;
    pub use crate::pruned::{PruneMode, PrunedFftConfig, PrunedFftPlan, PrunedStage};
    pub use crate::types::{Complex, FftDirection, FftPrecision, FftType};
}
