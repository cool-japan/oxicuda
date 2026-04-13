//! Discrete Cosine Transform (DCT) family.
//!
//! Provides GPU-accelerated implementations of:
//! - **DCT-II** — the "standard" forward DCT used in JPEG, MP3, H.264
//! - **DCT-III** — the inverse of DCT-II
//! - **DCT-IV** — the involutory transform underlying MDCT
//! - **MDCT** — Modified DCT for audio coding (MP3, AAC, Vorbis, Opus)

pub mod dct2;
pub mod dct3;
pub mod dct4;
pub mod mdct;

pub use dct2::{Dct2Plan, dct2_ortho_scale, dct2_reference};
pub use dct3::{
    dct3_reference, emit_pretwiddle_kernel as emit_dct3_pretwiddle_kernel, emit_unpermute_kernel,
};
pub use dct4::{
    dct4_reference, emit_postscale_kernel, emit_pretwiddle_kernel as emit_dct4_pretwiddle_kernel,
    postscale_table, pretwiddle_table,
};
pub use mdct::{MdctPlan, imdct, kbd_window, mdct, sine_window};
