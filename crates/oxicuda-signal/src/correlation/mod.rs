//! Correlation and convolution operations.

pub mod autocorr;
pub mod crosscorr;

pub use autocorr::{
    autocorr_biased, autocorr_normalised, autocorr_unbiased, autocovariance, ljung_box_q, pacf,
};
pub use crosscorr::{
    convolve, convolve_circular, crosscorr, crosscorr_normalised, find_delay, gcc_phat,
};
