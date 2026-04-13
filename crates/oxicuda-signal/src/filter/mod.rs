//! Digital filter design and application (FIR, IIR, Wiener).

pub mod fir;
pub mod iir;
pub mod wiener;

pub use fir::{
    design_bandpass, design_highpass, design_lowpass, design_raised_cosine, emit_fir_direct_kernel,
    fir_apply, freq_response as fir_freq_response,
};
pub use iir::{Biquad, iir_apply};
pub use wiener::{apply_wiener_gains, estimate_noise_psd, local_wiener_1d, wiener_gain};
