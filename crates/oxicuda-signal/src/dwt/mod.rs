//! Discrete Wavelet Transform (DWT) family.
//!
//! Provides GPU-accelerated implementations of:
//! - **Haar** — simplest orthonormal wavelet (db1)
//! - **Daubechies** (db2–db10) — compact support, maximal vanishing moments
//! - **Symlets** (sym2–sym10) — near-symmetric Daubechies variants
//! - **Multi-level** — successive application over dyadic tree

pub mod daubechies;
pub mod haar;
pub mod multilevel;
pub mod sym;

pub use daubechies::{
    db_filter_len, db_forward, db_highpass, db_inverse, db_lowpass, db_recon_highpass,
    db_recon_lowpass,
};
pub use haar::{emit_haar_forward_kernel, emit_haar_inverse_kernel, haar_forward, haar_inverse};
pub use multilevel::{
    WaveletDecomposition, hard_threshold, multilevel_forward, multilevel_inverse, soft_threshold,
    universal_threshold,
};
pub use sym::{sym_forward, sym_lowpass};
