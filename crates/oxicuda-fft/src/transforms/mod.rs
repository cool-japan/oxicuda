//! FFT transform execution routines.
//!
//! Each sub-module implements a specific transform type or dimensionality:
//!
//! | Module  | Transform                                      |
//! |---------|------------------------------------------------|
//! | [`c2c`] | Complex-to-complex (forward and inverse)        |
//! | [`r2c`] | Real-to-complex (forward, Hermitian output)     |
//! | [`c2r`] | Complex-to-real (inverse of R2C)                |
//! | [`fft2d`] | 2-D FFT via row/column 1-D passes + transpose |
//! | [`fft3d`] | 3-D FFT via three axis passes + transposes    |

pub mod c2c;
pub mod c2r;
pub mod fft2d;
pub mod fft3d;
pub mod r2c;
pub mod real_fft;
