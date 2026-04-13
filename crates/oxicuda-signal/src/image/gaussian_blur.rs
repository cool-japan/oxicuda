//! Separable Gaussian blur for 2D grayscale images.
//!
//! A 2D Gaussian convolution is separable:
//! ```text
//! G_2D(x, y) = G_1D(x) · G_1D(y)
//! ```
//! so the blur can be applied as two successive 1D convolutions — a horizontal
//! pass followed by a vertical pass — reducing complexity from O(N·k²) to
//! O(N·2k) per pixel.
//!
//! ## Kernel construction
//!
//! The 1D Gaussian kernel of radius `r` has `2r + 1` taps:
//! ```text
//! k[i] = exp(-(i − r)² / (2σ²))   for i = 0 … 2r
//! ```
//! then normalised to unit sum.  A radius of `r = ⌈3σ⌉` captures ≥ 99.7 % of
//! the Gaussian mass.
//!
//! ## Boundary handling
//!
//! Both CPU and GPU implementations use **zero-padding**: pixels outside the
//! image boundary contribute zero weight.  This causes mild edge darkening
//! proportional to the fraction of the kernel that falls outside.
//!
//! ## GPU strategy
//!
//! Two PTX kernels:
//! - `signal_gaussian_blur_h_f32` — horizontal pass, one thread per pixel.
//! - `signal_gaussian_blur_v_f32` — vertical pass, one thread per pixel.
//!
//! The 1D kernel coefficient array is passed as a device `*f32` pointer.
//! The loop counter is converted to a signed integer before computing offsets,
//! so that the `(k − radius)` subtraction is performed in signed arithmetic.

use oxicuda_ptx::arch::SmVersion;

use crate::{
    error::{SignalError, SignalResult},
    ptx_helpers::ptx_header,
};

// --------------------------------------------------------------------------- //
//  Kernel construction
// --------------------------------------------------------------------------- //

/// Compute the radius for a given σ: `r = max(1, ⌈3σ⌉)`.
#[must_use]
pub fn gaussian_radius_from_sigma(sigma: f64) -> usize {
    ((3.0 * sigma).ceil() as usize).max(1)
}

/// Generate a normalised 1D Gaussian kernel of the given `radius` and `sigma`.
///
/// Kernel length = `2 * radius + 1`.  Values sum to 1.0.
///
/// If `sigma` ≤ 0, the kernel degenerates to a unit impulse at the centre.
#[must_use]
pub fn gaussian_kernel_1d(sigma: f64, radius: usize) -> Vec<f64> {
    let len = 2 * radius + 1;
    if sigma <= 0.0 {
        let mut k = vec![0.0f64; len];
        k[radius] = 1.0;
        return k;
    }
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut k: Vec<f64> = (0..len)
        .map(|i| {
            let d = i as f64 - radius as f64;
            (-d * d / two_sigma_sq).exp()
        })
        .collect();
    let sum: f64 = k.iter().sum();
    if sum > 1e-15 {
        for v in &mut k {
            *v /= sum;
        }
    }
    k
}

// --------------------------------------------------------------------------- //
//  CPU reference: separable passes
// --------------------------------------------------------------------------- //

/// Horizontal 1D convolution pass (row-wise, zero-padded boundaries).
///
/// Returns an image of the same dimensions as the input.
#[must_use]
pub fn gaussian_blur_h(image: &[f32], height: usize, width: usize, kernel: &[f64]) -> Vec<f32> {
    let radius = (kernel.len() - 1) / 2;
    let mut out = vec![0.0f32; height * width];
    for r in 0..height {
        for c in 0..width {
            let mut acc = 0.0f64;
            for (k_idx, &kv) in kernel.iter().enumerate() {
                let nc = c as isize + k_idx as isize - radius as isize;
                if nc >= 0 && nc < width as isize {
                    acc += kv * image[r * width + nc as usize] as f64;
                }
            }
            out[r * width + c] = acc as f32;
        }
    }
    out
}

/// Vertical 1D convolution pass (column-wise, zero-padded boundaries).
///
/// Returns an image of the same dimensions as the input.
#[must_use]
pub fn gaussian_blur_v(image: &[f32], height: usize, width: usize, kernel: &[f64]) -> Vec<f32> {
    let radius = (kernel.len() - 1) / 2;
    let mut out = vec![0.0f32; height * width];
    for r in 0..height {
        for c in 0..width {
            let mut acc = 0.0f64;
            for (k_idx, &kv) in kernel.iter().enumerate() {
                let nr = r as isize + k_idx as isize - radius as isize;
                if nr >= 0 && nr < height as isize {
                    acc += kv * image[nr as usize * width + c] as f64;
                }
            }
            out[r * width + c] = acc as f32;
        }
    }
    out
}

/// Apply separable Gaussian blur to a 2D image.
///
/// # Arguments
/// - `image` — flat row-major image of shape `[height, width]`
/// - `height`, `width` — spatial dimensions
/// - `sigma` — Gaussian standard deviation (pixels)
/// - `radius` — optional kernel half-width; if `None`, defaults to `⌈3σ⌉`
///
/// # Errors
/// - `SignalError::InvalidSize` if `image.len() != height * width`.
/// - `SignalError::InvalidParameter` if `sigma < 0.0`.
pub fn gaussian_blur(
    image: &[f32],
    height: usize,
    width: usize,
    sigma: f64,
    radius: Option<usize>,
) -> SignalResult<Vec<f32>> {
    if image.len() != height * width {
        return Err(SignalError::InvalidSize(format!(
            "image length {} ≠ {height} × {width} = {}",
            image.len(),
            height * width
        )));
    }
    if sigma < 0.0 {
        return Err(SignalError::InvalidParameter(
            "sigma must be ≥ 0".to_owned(),
        ));
    }
    let r = radius.unwrap_or_else(|| gaussian_radius_from_sigma(sigma));
    let kernel = gaussian_kernel_1d(sigma, r);
    let tmp = gaussian_blur_h(image, height, width, &kernel);
    Ok(gaussian_blur_v(&tmp, height, width, &kernel))
}

// --------------------------------------------------------------------------- //
//  PTX kernel generation
// --------------------------------------------------------------------------- //

/// Emit the PTX horizontal Gaussian blur kernel for f32 images.
///
/// Kernel entry: `signal_gaussian_blur_h_f32`
///
/// Parameters: `(in: *f32, out: *f32, kernel: *f32, height: u32, width: u32, radius: u32, n: u64)`
#[must_use]
pub fn emit_gaussian_blur_h_kernel(sm: SmVersion) -> String {
    emit_blur_kernel(sm, true)
}

/// Emit the PTX vertical Gaussian blur kernel for f32 images.
///
/// Kernel entry: `signal_gaussian_blur_v_f32`
///
/// Parameters: `(in: *f32, out: *f32, kernel: *f32, height: u32, width: u32, radius: u32, n: u64)`
#[must_use]
pub fn emit_gaussian_blur_v_kernel(sm: SmVersion) -> String {
    emit_blur_kernel(sm, false)
}

fn emit_blur_kernel(sm: SmVersion, horizontal: bool) -> String {
    // Register plan (shared between passes):
    //   %rd0 = in, %rd1 = out, %rd2 = kernel, %rd3 = n, %rd4 = tid, %rd5 = temp addr
    //   %r0 = height, %r1 = width, %r2 = radius, %r3 = klen (2*radius+1)
    //   %r4 = tid tmp, %r5 = ctaid, %r6 = ntid, %r7 = row, %r8 = col
    //   %r9 = pixel_idx tmp
    //   %r11 = k (loop counter, u32)
    //   %s0 = nr (s32), %s1 = nc (s32)
    //   %s2 = height (s32), %s3 = width (s32)
    //   %s4 = s_row or s_col (fixed component, s32)
    //   %s5 = offset = k - radius (s32)
    //   %s6 = k as s32 (cvt from %r11)
    //   %s7 = radius as s32 (set once before loop)
    //   %f0 = accumulator, %f1 = kernel[k], %f2 = pixel

    let name = if horizontal {
        "signal_gaussian_blur_h_f32"
    } else {
        "signal_gaussian_blur_v_f32"
    };

    // For the horizontal pass:
    //   nr = row (fixed)  →  s0 = s4 (s_row, set once)
    //   nc = col + (k - radius)  →  s1 = s4_col + s5
    // For the vertical pass:
    //   nr = row + (k - radius)  →  s0 = s4_row + s5
    //   nc = col (fixed)  →  s1 = s4 (s_col, set once)

    // Compute (nr, nc) inside the loop — emitted as a PTX string fragment.
    // %s6 = k (signed), %s7 = radius (signed), %s4 = fixed dimension (row or col).
    let (fixed_reg, fixed_setup, varying_reg, bounds_reg) = if horizontal {
        // Fixed: nr = row. Varying: nc = col + (k - radius).
        (
            "%s0",
            "    cvt.s32.u32  %s4, %r7;\n    mov.s32      %s0, %s4;\n",
            "%s1",
            "%s3",
        )
    } else {
        // Fixed: nc = col. Varying: nr = row + (k - radius).
        (
            "%s1",
            "    cvt.s32.u32  %s4, %r8;\n    mov.s32      %s1, %s4;\n",
            "%s0",
            "%s2",
        )
    };

    // Varying dimension setup inside the loop.
    // Horizontal: nc = col + (k - radius): cvt col to s32, compute offset, add.
    // Vertical:   nr = row + (k - radius): cvt row to s32, compute offset, add.
    let (varying_src_reg, varying_dim_name) = if horizontal {
        ("%r8", "col") // nc = col + offset
    } else {
        ("%r7", "row") // nr = row + offset
    };

    let varying_inner = format!(
        "    cvt.s32.u32  %s6, %r11;\n\
         \x20   sub.s32      %s5, %s6, %s7;\n\
         \x20   cvt.s32.u32  %s4, {varying_src_reg};\n\
         \x20   add.s32      {varying_reg}, %s4, %s5;   // {varying_dim_name} + (k - radius)\n"
    );

    // Bounds check: the varying dimension must be in [0, bounds_reg).
    // (The fixed dimension is always valid since it equals row or col.)
    let bounds_inner = format!(
        "    setp.lt.s32  %p2, {varying_reg}, 0;\n\
         \x20   @%p2 bra     skip_k;\n\
         \x20   setp.ge.s32  %p2, {varying_reg}, {bounds_reg};\n\
         \x20   @%p2 bra     skip_k;\n"
    );

    let _ = fixed_reg; // used only in fixed_setup
    let hdr = ptx_header(sm);

    format!(
        r"{hdr}
.visible .entry {name}(
    .param .u64 param_in,
    .param .u64 param_out,
    .param .u64 param_kernel,
    .param .u32 param_height,
    .param .u32 param_width,
    .param .u32 param_radius,
    .param .u64 param_n
)
{{
    .reg .pred  %p<4>;
    .reg .u32   %r<14>;
    .reg .u64   %rd<8>;
    .reg .f32   %f<4>;
    .reg .s32   %s<10>;

    ld.param.u64 %rd0, [param_in];
    ld.param.u64 %rd1, [param_out];
    ld.param.u64 %rd2, [param_kernel];
    ld.param.u32 %r0,  [param_height];
    ld.param.u32 %r1,  [param_width];
    ld.param.u32 %r2,  [param_radius];
    ld.param.u64 %rd3, [param_n];

    // klen = 2*radius + 1
    mad.lo.u32   %r3,  %r2, 2, 1;

    mov.u32      %r4,  %tid.x;
    mov.u32      %r5,  %ctaid.x;
    mov.u32      %r6,  %ntid.x;
    mad.lo.u32   %r4,  %r5, %r6, %r4;
    cvt.u64.u32  %rd4, %r4;

    setp.ge.u64  %p0, %rd4, %rd3;
    @%p0 bra     done;

    div.u32      %r7, %r4, %r1;   // row
    rem.u32      %r8, %r4, %r1;   // col

    // Signed constants used throughout.
    cvt.s32.u32  %s2, %r0;    // height (s32)
    cvt.s32.u32  %s3, %r1;    // width  (s32)
    cvt.s32.u32  %s7, %r2;    // radius (s32)

    // Set the fixed dimension (row or col) once before the loop.
{fixed_setup}
    mov.f32      %f0, 0f00000000;  // accumulator
    mov.u32      %r11, 0;          // k = 0

loop_k:
    setp.ge.u32  %p1, %r11, %r3;
    @%p1 bra     loop_k_end;

    // Compute varying dimension offset: k - radius (signed).
{varying_inner}
    // Bounds check (fixed dimension is always in-bounds by construction).
{bounds_inner}
    // Load kernel[k] (f32).
    cvt.u64.u32  %rd5, %r11;
    shl.b64      %rd5, %rd5, 2;
    add.u64      %rd5, %rd2, %rd5;
    ld.global.f32 %f1, [%rd5];

    // pixel_idx = nr * width + nc.
    cvt.u32.s32  %r9, %s0;
    mul.lo.u32   %r9, %r9, %r1;
    cvt.u32.s32  %r10, %s1;
    add.u32      %r9,  %r9, %r10;
    cvt.u64.u32  %rd5, %r9;
    shl.b64      %rd5, %rd5, 2;
    add.u64      %rd5, %rd0, %rd5;
    ld.global.f32 %f2, [%rd5];

    fma.rn.f32   %f0, %f1, %f2, %f0;

skip_k:
    add.u32      %r11, %r11, 1;
    bra          loop_k;

loop_k_end:
    shl.b64      %rd4, %rd4, 2;
    add.u64      %rd4, %rd1, %rd4;
    st.global.f32 [%rd4], %f0;

done:
    ret;
}}
",
        name = name,
        fixed_setup = fixed_setup,
        varying_inner = varying_inner,
        bounds_inner = bounds_inner,
    )
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_sums_to_one() {
        let k = gaussian_kernel_1d(1.0, 3);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "kernel sum = {sum}");
    }

    #[test]
    fn test_kernel_is_symmetric() {
        let k = gaussian_kernel_1d(2.0, 6);
        let n = k.len();
        for i in 0..n / 2 {
            assert!((k[i] - k[n - 1 - i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_kernel_zero_sigma_is_impulse() {
        let k = gaussian_kernel_1d(0.0, 3);
        assert_eq!(k[3], 1.0, "centre tap should be 1");
        for (i, &v) in k.iter().enumerate() {
            if i != 3 {
                assert_eq!(v, 0.0, "non-centre tap {i} should be 0");
            }
        }
    }

    #[test]
    fn test_gaussian_radius_from_sigma() {
        assert_eq!(gaussian_radius_from_sigma(1.0), 3);
        assert_eq!(gaussian_radius_from_sigma(2.0), 6);
        assert_eq!(gaussian_radius_from_sigma(0.1), 1);
    }

    #[test]
    fn test_blur_uniform_image_is_identity() {
        // A constant image blurred with any Gaussian stays constant in the interior.
        // Border pixels are affected by zero-padding (edge darkening); skip them.
        // sigma=1.0 → radius=3, so check interior pixels starting at row/col 3.
        let size = 15usize;
        let img = vec![7.0f32; size * size];
        let result = gaussian_blur(&img, size, size, 1.0, None).unwrap();
        let radius = gaussian_radius_from_sigma(1.0);
        for r in radius..size - radius {
            for c in radius..size - radius {
                let v = result[r * size + c];
                assert!((v - 7.0).abs() < 1e-4, "expected 7.0 at ({r},{c}), got {v}");
            }
        }
    }

    #[test]
    fn test_blur_single_impulse_energy() {
        // The blur of a single impulse should conserve mass (sum of weights).
        let mut img = vec![0.0f32; 49];
        img[3 * 7 + 3] = 1.0;
        let result = gaussian_blur(&img, 7, 7, 1.0, Some(2)).unwrap();
        let sum: f32 = result.iter().sum();
        // Zero-padding means total mass ≤ 1.0 (some weight leaks outside).
        assert!(sum > 0.5 && sum <= 1.0 + 1e-5, "impulse energy = {sum}");
    }

    #[test]
    fn test_blur_h_identity_pass_no_column_shift() {
        // With impulse kernel, h-pass should return the original image.
        let img: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let k_impulse = gaussian_kernel_1d(0.0, 1); // unit impulse at centre
        let out = gaussian_blur_h(&img, 3, 3, &k_impulse);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blur_v_identity_pass() {
        let img: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let k_impulse = gaussian_kernel_1d(0.0, 1);
        let out = gaussian_blur_v(&img, 3, 3, &k_impulse);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_h_then_v_matches_full_blur() {
        // Verify that manual h+v == gaussian_blur().
        let img: Vec<f32> = (0..49).map(|i| (i as f32) * 0.1).collect();
        let k = gaussian_kernel_1d(1.0, 3);
        let tmp = gaussian_blur_h(&img, 7, 7, &k);
        let manual = gaussian_blur_v(&tmp, 7, 7, &k);
        let full = gaussian_blur(&img, 7, 7, 1.0, Some(3)).unwrap();
        for (a, b) in manual.iter().zip(full.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blur_invalid_size() {
        let img = vec![1.0f32; 10];
        assert!(gaussian_blur(&img, 3, 4, 1.0, None).is_err());
    }

    #[test]
    fn test_blur_negative_sigma() {
        let img = vec![1.0f32; 9];
        assert!(gaussian_blur(&img, 3, 3, -1.0, None).is_err());
    }

    #[test]
    fn test_emit_h_kernel_ptx() {
        let ptx = emit_gaussian_blur_h_kernel(SmVersion::Sm80);
        assert!(ptx.contains("signal_gaussian_blur_h_f32"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("loop_k:"));
    }

    #[test]
    fn test_emit_v_kernel_ptx() {
        let ptx = emit_gaussian_blur_v_kernel(SmVersion::Sm80);
        assert!(ptx.contains("signal_gaussian_blur_v_f32"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("loop_k:"));
    }
}
