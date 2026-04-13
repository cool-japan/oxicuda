//! Resize (interpolation) operations for DNN.
//!
//! This module provides GPU-accelerated 2D image resizing:
//!
//! - [`nearest`] — Nearest-neighbor interpolation.
//! - [`bilinear`] — Bilinear interpolation with optional corner alignment.
//! - [`bicubic`] — Bicubic interpolation using 4x4 neighbor windows.

pub mod bicubic;
pub mod bilinear;
pub mod nearest;

pub use bicubic::resize_bicubic;
pub use bilinear::resize_bilinear;
pub use nearest::resize_nearest;
