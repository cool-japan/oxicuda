//! Morphological image processing operations.
//!
//! Implements erosion, dilation, opening, closing, white/black top-hat,
//! and morphological gradient for 2D single-channel f32 images.
//!
//! ## Mathematical definitions
//!
//! For a structuring element B centred at the origin:
//! ```text
//! (A ⊖ B)[p] = min_{b ∈ B} A[p + b]   (erosion)
//! (A ⊕ B)[p] = max_{b ∈ B} A[p - b]   (dilation)
//! ```
//!
//! - **Opening**  = erode → dilate (removes small bright structures)
//! - **Closing**  = dilate → erode (fills small dark holes)
//! - **Top-hat**  = image − open(image)
//! - **Black-hat** = close(image) − image
//! - **Gradient** = dilate(image) − erode(image)
//!
//! ## GPU strategy
//!
//! Each GPU thread handles one output pixel.  The structuring-element mask
//! is stored in a global-memory `u8` array (1 = active, 0 = inactive).
//! Row and column dimensions are passed as `u32` kernel parameters.
//! Boundary policy: out-of-bounds pixels are skipped (zero-pad semantics —
//! they do not participate in the min/max reduction).

use oxicuda_ptx::arch::SmVersion;

use crate::{
    error::{SignalError, SignalResult},
    ptx_helpers::ptx_header,
    types::StructuringElement,
};

// --------------------------------------------------------------------------- //
//  Structuring-element mask generation
// --------------------------------------------------------------------------- //

/// Generate the flat SE mask as a `Vec<u8>` (1 = active, 0 = inactive).
///
/// Returns `(mask, se_height, se_width)`.
#[must_use]
pub fn generate_se_mask(se: StructuringElement) -> (Vec<u8>, usize, usize) {
    match se {
        StructuringElement::Rectangle { height, width } => {
            let h = height as usize;
            let w = width as usize;
            (vec![1u8; h * w], h, w)
        }
        StructuringElement::Ellipse { height, width } => {
            let h = height as usize;
            let w = width as usize;
            // Centre coordinates (float).
            let cy = (h as f64 - 1.0) / 2.0;
            let cx = (w as f64 - 1.0) / 2.0;
            let ry = cy.max(1e-9);
            let rx = cx.max(1e-9);
            let mask: Vec<u8> = (0..h)
                .flat_map(|r| {
                    (0..w).map(move |c| {
                        let dy = (r as f64 - cy) / ry;
                        let dx = (c as f64 - cx) / rx;
                        u8::from(dy * dy + dx * dx <= 1.0)
                    })
                })
                .collect();
            (mask, h, w)
        }
        StructuringElement::Cross { radius } => {
            let r = radius as usize;
            let size = 2 * r + 1;
            let mut mask = vec![0u8; size * size];
            // Centre row.
            for c in 0..size {
                mask[r * size + c] = 1;
            }
            // Centre column.
            for rr in 0..size {
                mask[rr * size + r] = 1;
            }
            (mask, size, size)
        }
    }
}

// --------------------------------------------------------------------------- //
//  CPU reference: single-pass morphology
// --------------------------------------------------------------------------- //

fn morphology_1pass(
    image: &[f32],
    img_h: usize,
    img_w: usize,
    mask: &[u8],
    se_h: usize,
    se_w: usize,
    is_erosion: bool,
) -> Vec<f32> {
    let init = if is_erosion {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    let half_h = se_h / 2;
    let half_w = se_w / 2;
    let mut out = vec![0.0f32; img_h * img_w];
    for r in 0..img_h {
        for c in 0..img_w {
            let mut val = init;
            for dr in 0..se_h {
                for dc in 0..se_w {
                    if mask[dr * se_w + dc] == 0 {
                        continue;
                    }
                    let nr = r as isize + dr as isize - half_h as isize;
                    let nc = c as isize + dc as isize - half_w as isize;
                    if nr < 0 || nr >= img_h as isize || nc < 0 || nc >= img_w as isize {
                        // Skip OOB — does not affect the min/max accumulator.
                        continue;
                    }
                    let pixel = image[nr as usize * img_w + nc as usize];
                    val = if is_erosion {
                        val.min(pixel)
                    } else {
                        val.max(pixel)
                    };
                }
            }
            // Guard: if no valid pixel was loaded (all SE neighbours OOB), output 0.
            out[r * img_w + c] = if val.is_infinite() { 0.0 } else { val };
        }
    }
    out
}

// --------------------------------------------------------------------------- //
//  Public morphology operations
// --------------------------------------------------------------------------- //

/// Morphological erosion: `(A ⊖ B)[p] = min_{b ∈ B} A[p + b]`.
///
/// OOB pixels are zero-padded (they do not participate in the minimum).
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `image.len() != img_h * img_w`.
pub fn erode(
    image: &[f32],
    img_h: usize,
    img_w: usize,
    se: StructuringElement,
) -> SignalResult<Vec<f32>> {
    if image.len() != img_h * img_w {
        return Err(SignalError::InvalidSize(format!(
            "image length {} ≠ {img_h} × {img_w} = {}",
            image.len(),
            img_h * img_w
        )));
    }
    let (mask, se_h, se_w) = generate_se_mask(se);
    Ok(morphology_1pass(
        image, img_h, img_w, &mask, se_h, se_w, true,
    ))
}

/// Morphological dilation: `(A ⊕ B)[p] = max_{b ∈ B} A[p - b]`.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `image.len() != img_h * img_w`.
pub fn dilate(
    image: &[f32],
    img_h: usize,
    img_w: usize,
    se: StructuringElement,
) -> SignalResult<Vec<f32>> {
    if image.len() != img_h * img_w {
        return Err(SignalError::InvalidSize(format!(
            "image length {} ≠ {img_h} × {img_w} = {}",
            image.len(),
            img_h * img_w
        )));
    }
    let (mask, se_h, se_w) = generate_se_mask(se);
    Ok(morphology_1pass(
        image, img_h, img_w, &mask, se_h, se_w, false,
    ))
}

/// Morphological opening: erosion then dilation.
///
/// Removes small bright objects and smooths contours.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `image.len() != img_h * img_w`.
pub fn open(
    image: &[f32],
    img_h: usize,
    img_w: usize,
    se: StructuringElement,
) -> SignalResult<Vec<f32>> {
    let eroded = erode(image, img_h, img_w, se)?;
    dilate(&eroded, img_h, img_w, se)
}

/// Morphological closing: dilation then erosion.
///
/// Fills small dark holes and bridges nearby bright regions.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `image.len() != img_h * img_w`.
pub fn close(
    image: &[f32],
    img_h: usize,
    img_w: usize,
    se: StructuringElement,
) -> SignalResult<Vec<f32>> {
    let dilated = dilate(image, img_h, img_w, se)?;
    erode(&dilated, img_h, img_w, se)
}

/// White top-hat transform: `image − open(image)`.
///
/// Extracts small bright features relative to the local background.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `image.len() != img_h * img_w`.
pub fn tophat(
    image: &[f32],
    img_h: usize,
    img_w: usize,
    se: StructuringElement,
) -> SignalResult<Vec<f32>> {
    let opened = open(image, img_h, img_w, se)?;
    Ok(image
        .iter()
        .zip(opened.iter())
        .map(|(&a, &b)| a - b)
        .collect())
}

/// Black-hat transform: `close(image) − image`.
///
/// Extracts small dark features relative to the local background.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `image.len() != img_h * img_w`.
pub fn blackhat(
    image: &[f32],
    img_h: usize,
    img_w: usize,
    se: StructuringElement,
) -> SignalResult<Vec<f32>> {
    let closed = close(image, img_h, img_w, se)?;
    Ok(closed
        .iter()
        .zip(image.iter())
        .map(|(&a, &b)| a - b)
        .collect())
}

/// Morphological gradient: `dilate(image) − erode(image)`.
///
/// Highlights boundaries by computing the difference between dilation and erosion.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `image.len() != img_h * img_w`.
pub fn morphological_gradient(
    image: &[f32],
    img_h: usize,
    img_w: usize,
    se: StructuringElement,
) -> SignalResult<Vec<f32>> {
    let d = dilate(image, img_h, img_w, se)?;
    let e = erode(image, img_h, img_w, se)?;
    Ok(d.iter().zip(e.iter()).map(|(&a, &b)| a - b).collect())
}

// --------------------------------------------------------------------------- //
//  PTX kernel generation
// --------------------------------------------------------------------------- //

/// Emit the PTX erosion kernel for f32 images.
///
/// Kernel entry: `signal_erosion_f32`
/// Parameters: `(in: *f32, out: *f32, mask: *u8, height: u32, width: u32, se_h: u32, se_w: u32, n: u64)`
#[must_use]
pub fn emit_erosion_kernel(sm: SmVersion) -> String {
    emit_morphology_kernel(sm, true)
}

/// Emit the PTX dilation kernel for f32 images.
///
/// Kernel entry: `signal_dilation_f32`
/// Parameters: `(in: *f32, out: *f32, mask: *u8, height: u32, width: u32, se_h: u32, se_w: u32, n: u64)`
#[must_use]
pub fn emit_dilation_kernel(sm: SmVersion) -> String {
    emit_morphology_kernel(sm, false)
}

fn emit_morphology_kernel(sm: SmVersion, is_erosion: bool) -> String {
    let (name, init_imm, reduce_op) = if is_erosion {
        ("signal_erosion_f32", "0f7F800000", "min.f32")
    } else {
        ("signal_dilation_f32", "0fFF800000", "max.f32")
    };
    // IEEE 754 bits for the init sentinel used in the infinity guard.
    let guard_imm = if is_erosion {
        "0f7F800000"
    } else {
        "0fFF800000"
    };

    let hdr = ptx_header(sm);
    format!(
        r"{hdr}
.visible .entry {name}(
    .param .u64 param_in,
    .param .u64 param_out,
    .param .u64 param_mask,
    .param .u32 param_height,
    .param .u32 param_width,
    .param .u32 param_se_h,
    .param .u32 param_se_w,
    .param .u64 param_n
)
{{
    .reg .pred  %p<4>;
    .reg .u32   %r<20>;
    .reg .u64   %rd<10>;
    .reg .f32   %f<4>;
    .reg .s32   %s<12>;

    ld.param.u64 %rd0, [param_in];
    ld.param.u64 %rd1, [param_out];
    ld.param.u64 %rd2, [param_mask];
    ld.param.u32 %r0,  [param_height];
    ld.param.u32 %r1,  [param_width];
    ld.param.u32 %r2,  [param_se_h];
    ld.param.u32 %r3,  [param_se_w];
    ld.param.u64 %rd3, [param_n];

    mov.u32      %r4,  %tid.x;
    mov.u32      %r5,  %ctaid.x;
    mov.u32      %r6,  %ntid.x;
    mad.lo.u32   %r4,  %r5, %r6, %r4;
    cvt.u64.u32  %rd4, %r4;

    setp.ge.u64  %p0, %rd4, %rd3;
    @%p0 bra     done;

    div.u32      %r7,  %r4, %r1;
    rem.u32      %r8,  %r4, %r1;
    shr.u32      %r9,  %r2, 1;
    shr.u32      %r10, %r3, 1;

    mov.f32      %f0, {init_imm};
    mov.u32      %r11, 0;

loop_dr:
    setp.ge.u32  %p1, %r11, %r2;
    @%p1 bra     loop_dr_end;
    mov.u32      %r12, 0;

loop_dc:
    setp.ge.u32  %p1, %r12, %r3;
    @%p1 bra     loop_dc_end;

    mad.lo.u32   %r13, %r11, %r3, %r12;
    cvt.u64.u32  %rd5, %r13;
    add.u64      %rd5, %rd2, %rd5;
    ld.global.u8 %r14, [%rd5];
    setp.eq.u32  %p2,  %r14, 0;
    @%p2 bra     next_dc;

    cvt.s32.u32  %s0, %r7;
    cvt.s32.u32  %s1, %r11;
    cvt.s32.u32  %s2, %r9;
    sub.s32      %s3, %s1, %s2;
    add.s32      %s3, %s0, %s3;

    cvt.s32.u32  %s4, %r8;
    cvt.s32.u32  %s5, %r12;
    cvt.s32.u32  %s6, %r10;
    sub.s32      %s7, %s5, %s6;
    add.s32      %s7, %s4, %s7;

    setp.lt.s32  %p2, %s3, 0;
    @%p2 bra     next_dc;
    cvt.s32.u32  %s8, %r0;
    setp.ge.s32  %p2, %s3, %s8;
    @%p2 bra     next_dc;
    setp.lt.s32  %p2, %s7, 0;
    @%p2 bra     next_dc;
    cvt.s32.u32  %s9, %r1;
    setp.ge.s32  %p2, %s7, %s9;
    @%p2 bra     next_dc;

    cvt.u32.s32  %r15, %s3;
    mul.lo.u32   %r15, %r15, %r1;
    cvt.u32.s32  %r16, %s7;
    add.u32      %r15, %r15, %r16;
    cvt.u64.u32  %rd5, %r15;
    shl.b64      %rd5, %rd5, 2;
    add.u64      %rd5, %rd0, %rd5;
    ld.global.f32 %f1, [%rd5];
    {reduce_op}  %f0, %f0, %f1;

next_dc:
    add.u32      %r12, %r12, 1;
    bra          loop_dc;

loop_dc_end:
    add.u32      %r11, %r11, 1;
    bra          loop_dr;

loop_dr_end:
    // Guard: if val is still the sentinel (all SE neighbours were OOB), write 0.
    mov.f32      %f2, {guard_imm};
    setp.eq.f32  %p3, %f0, %f2;
    selp.f32     %f0, 0f00000000, %f0, %p3;

    shl.b64      %rd4, %rd4, 2;
    add.u64      %rd4, %rd1, %rd4;
    st.global.f32 [%rd4], %f0;

done:
    ret;
}}
"
    )
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    fn rect3x3() -> StructuringElement {
        StructuringElement::Rectangle {
            height: 3,
            width: 3,
        }
    }

    fn cross3() -> StructuringElement {
        StructuringElement::Cross { radius: 1 }
    }

    #[test]
    fn test_generate_se_rect() {
        let (mask, h, w) = generate_se_mask(StructuringElement::Rectangle {
            height: 2,
            width: 3,
        });
        assert_eq!(h, 2);
        assert_eq!(w, 3);
        assert!(mask.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_generate_se_cross() {
        let (mask, h, w) = generate_se_mask(StructuringElement::Cross { radius: 1 });
        assert_eq!(h, 3);
        assert_eq!(w, 3);
        // Centre row (row=1) and centre col (col=1) are active.
        assert_eq!(mask[3], 1); // centre row, left
        assert_eq!(mask[4], 1); // centre
        assert_eq!(mask[5], 1); // centre row, right
        assert_eq!(mask[1], 1); // centre col, top
        assert_eq!(mask[7], 1); // centre col, bottom
        assert_eq!(mask[0], 0); // corner not set
    }

    #[test]
    fn test_generate_se_ellipse_1x1() {
        // A 1×1 ellipse is just the centre pixel.
        let (mask, h, w) = generate_se_mask(StructuringElement::Ellipse {
            height: 1,
            width: 1,
        });
        assert_eq!(h, 1);
        assert_eq!(w, 1);
        assert_eq!(mask[0], 1);
    }

    #[test]
    fn test_erode_uniform_image() {
        // Uniform image → erosion is identity.
        let img = vec![5.0f32; 9];
        let result = erode(&img, 3, 3, rect3x3()).unwrap();
        for v in &result {
            assert!((v - 5.0).abs() < 1e-6, "expected 5.0, got {v}");
        }
    }

    #[test]
    fn test_dilate_uniform_image() {
        // Uniform image → dilation is identity.
        let img = vec![5.0f32; 9];
        let result = dilate(&img, 3, 3, rect3x3()).unwrap();
        for v in &result {
            assert!((v - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_erode_removes_peak() {
        // 5×5 image; bright spike at centre.
        let mut img = vec![0.0f32; 25];
        img[2 * 5 + 2] = 1.0;
        let result = erode(&img, 5, 5, rect3x3()).unwrap();
        // Erosion with 3×3 rect should zero out a single-pixel spike.
        assert!(result.iter().all(|&v| v == 0.0), "spike should be eroded");
    }

    #[test]
    fn test_dilate_expands_peak() {
        // Single bright pixel; dilation with 3×3 rect → 3×3 region.
        let mut img = vec![0.0f32; 25];
        img[2 * 5 + 2] = 1.0;
        let result = dilate(&img, 5, 5, rect3x3()).unwrap();
        // The 3×3 neighbourhood of (2,2) should all be 1.
        for dr in 0usize..3 {
            for dc in 0usize..3 {
                assert_eq!(result[(1 + dr) * 5 + (1 + dc)], 1.0);
            }
        }
    }

    #[test]
    fn test_open_removes_noise() {
        // 5×5 background=1, single noise pixel=10.
        let mut img = vec![1.0f32; 25];
        img[2 * 5 + 2] = 10.0;
        let result = open(&img, 5, 5, rect3x3()).unwrap();
        // Opening removes the noise spike; result should be ≈1 everywhere.
        assert!(
            result.iter().all(|&v| (v - 1.0).abs() < 1e-5),
            "noise spike not removed"
        );
    }

    #[test]
    fn test_close_fills_hole() {
        // 5×5 background=1, single dark hole=0.
        let mut img = vec![1.0f32; 25];
        img[2 * 5 + 2] = 0.0;
        let result = close(&img, 5, 5, rect3x3()).unwrap();
        // Closing fills the hole; result should be ≈1 everywhere.
        assert!(
            result.iter().all(|&v| (v - 1.0).abs() < 1e-5),
            "hole not filled"
        );
    }

    #[test]
    fn test_tophat_detects_peak() {
        let mut img = vec![0.5f32; 25];
        img[2 * 5 + 2] = 1.0;
        let th = tophat(&img, 5, 5, rect3x3()).unwrap();
        // Top-hat highlights the bright peak.
        assert!(th[2 * 5 + 2] > 0.0);
    }

    #[test]
    fn test_morphological_gradient_at_edge() {
        // Gradient of a step edge should be non-zero at the transition.
        let img: Vec<f32> = (0..25).map(|i| if i % 5 < 3 { 1.0 } else { 0.0 }).collect();
        let grad = morphological_gradient(&img, 5, 5, cross3()).unwrap();
        let has_nonzero = grad.iter().any(|&v| v > 0.0);
        assert!(has_nonzero, "gradient should be nonzero at step edge");
    }

    #[test]
    fn test_erode_invalid_size() {
        let img = vec![1.0f32; 10];
        assert!(erode(&img, 3, 4, rect3x3()).is_err());
    }

    #[test]
    fn test_emit_erosion_ptx_has_entry() {
        let ptx = emit_erosion_kernel(SmVersion::Sm80);
        assert!(ptx.contains("signal_erosion_f32"), "missing kernel entry");
        assert!(ptx.contains("min.f32"), "missing min reduction");
    }

    #[test]
    fn test_emit_dilation_ptx_has_entry() {
        let ptx = emit_dilation_kernel(SmVersion::Sm80);
        assert!(ptx.contains("signal_dilation_f32"), "missing kernel entry");
        assert!(ptx.contains("max.f32"), "missing max reduction");
    }
}
