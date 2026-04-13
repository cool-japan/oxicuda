//! OxiCUDA Signal — GPU-accelerated signal, audio, and image processing.
//!
//! A pure-Rust signal processing library for the OxiCUDA ecosystem, providing
//! CPU reference implementations and PTX-generating GPU kernels for:
//!
//! | Module | Operations |
//! |---|---|
//! | [`dct`] | DCT-II, DCT-III (IDCT), DCT-IV, MDCT/IMDCT |
//! | [`dwt`] | Haar, Daubechies db2–db10, Symlets, multi-level DWT |
//! | [`audio`] | STFT, mel filterbank, MFCC, chroma, spectrograms |
//! | [`filter`] | FIR (windowed-sinc, raised-cosine), IIR (biquad SOS), Wiener |
//! | [`correlation`] | Autocorrelation, cross-correlation, GCC-PHAT, convolution |
//! | [`image`] | NMS (greedy, soft, heatmap), morphology, Gaussian blur, Sobel |
//! | [`window`] | Window functions and analysis metrics |
//!
//! ## Design principles
//!
//! - **No `unwrap()`**: All fallible operations return `SignalResult<T>`.
//! - **CPU + GPU**: Each operation has a CPU reference and PTX kernel generator.
//! - **Zero-copy PTX**: Kernels are emitted as `String` at runtime, compiled by
//!   the CUDA JIT via `cuModuleLoadData` — no pre-compiled `.ptx` files.
//! - **No warnings**: Zero clippy/rustc warnings across the entire crate.

pub mod audio;
pub mod correlation;
pub mod dct;
pub mod dwt;
pub mod error;
pub mod filter;
pub mod handle;
pub mod image;
pub mod ptx_helpers;
pub mod types;
pub mod window;

pub use error::{SignalError, SignalResult};
pub use handle::SignalHandle;
pub use types::{
    NormMode, PadMode, SignalPrecision, StructuringElement, TransformDirection, WaveletFamily,
    WindowType,
};

/// Convenience prelude — import everything needed for typical signal-processing
/// pipelines in one `use oxicuda_signal::prelude::*;` statement.
pub mod prelude {
    pub use crate::audio::{
        MfccConfig, SpectrogramConfig, SpectrogramType, StftConfig, chroma_from_power,
        magnitude_spectrogram, mfcc, power_spectrogram, spectrogram, stft_reference,
    };
    pub use crate::correlation::{
        autocorr_biased, autocorr_normalised, autocorr_unbiased, autocovariance, convolve,
        convolve_circular, crosscorr, crosscorr_normalised, find_delay, gcc_phat, ljung_box_q,
        pacf,
    };
    pub use crate::dct::{
        Dct2Plan, MdctPlan, dct2_reference, dct3_reference, dct4_reference, imdct, mdct,
    };
    pub use crate::dwt::{
        WaveletDecomposition, hard_threshold, multilevel_forward, multilevel_inverse,
        soft_threshold, universal_threshold,
    };
    pub use crate::error::{SignalError, SignalResult};
    pub use crate::filter::{
        Biquad, apply_wiener_gains, design_bandpass, design_highpass, design_lowpass,
        design_raised_cosine, estimate_noise_psd, fir_apply, iir_apply, local_wiener_1d,
        wiener_gain,
    };
    pub use crate::image::{
        BBox, SoftNmsDecay, close, dilate, erode, gaussian_blur, morphological_gradient,
        nms_greedy, nms_heatmap, nms_soft, open, sobel, sobel_magnitude,
    };
    pub use crate::types::{
        NormMode, PadMode, SignalPrecision, StructuringElement, TransformDirection, WaveletFamily,
        WindowType,
    };
    pub use crate::window::make_window;
}
