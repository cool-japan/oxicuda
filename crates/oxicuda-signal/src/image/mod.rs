//! Image processing primitives.
//!
//! Covers the most common GPU-accelerated image operations used in computer
//! vision pipelines:
//!
//! | Module | Operations |
//! |---|---|
//! | [`nms`] | Greedy NMS, Soft-NMS, 2D heatmap NMS |
//! | [`morphology`] | Erosion, dilation, open, close, top-hat, black-hat, gradient |
//! | [`mod@gaussian_blur`] | Separable Gaussian blur (horizontal + vertical passes) |
//! | [`mod@sobel`] | Sobel Gx/Gy, gradient magnitude and orientation |

pub mod gaussian_blur;
pub mod morphology;
pub mod nms;
pub mod sobel;

pub use gaussian_blur::{
    emit_gaussian_blur_h_kernel, emit_gaussian_blur_v_kernel, gaussian_blur, gaussian_blur_h,
    gaussian_blur_v, gaussian_kernel_1d, gaussian_radius_from_sigma,
};
pub use morphology::{
    blackhat, close, dilate, emit_dilation_kernel, emit_erosion_kernel, erode, generate_se_mask,
    morphological_gradient, open, tophat,
};
pub use nms::{BBox, SoftNmsDecay, iou, nms_greedy, nms_heatmap, nms_soft};
pub use sobel::{
    emit_sobel_x_kernel, emit_sobel_y_kernel, sobel, sobel_angle, sobel_magnitude, sobel_x, sobel_y,
};
