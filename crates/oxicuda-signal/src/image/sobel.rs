//! Sobel edge detection for 2D grayscale images.
//!
//! The Sobel operator approximates the image gradient via 3×3 convolution:
//!
//! ```text
//! Kx = [[-1, 0, 1],          Ky = [[-1, -2, -1],
//!        [-2, 0, 2],                [ 0,  0,  0],
//!        [-1, 0, 1]]                [ 1,  2,  1]]
//! ```
//!
//! giving horizontal gradient `Gx` and vertical gradient `Gy`.
//!
//! The gradient magnitude is `|G| = √(Gx² + Gy²)` and the orientation is
//! `θ = atan2(Gy, Gx)` in radians.
//!
//! ## Boundary handling
//!
//! Both the CPU reference and PTX kernels use **zero-padding**: the image is
//! treated as surrounded by zeros, so the 3×3 neighbourhood at a border pixel
//! silently loads 0 for any out-of-bounds position.
//!
//! ## GPU strategy
//!
//! Two separate PTX kernels compute `Gx` and `Gy` independently, one thread
//! per output pixel.  Each kernel loads exactly the 6 non-zero Sobel taps via
//! fully-unrolled predicated loads (no loop), giving maximum instruction-level
//! parallelism.

use oxicuda_ptx::arch::SmVersion;

use crate::{
    error::{SignalError, SignalResult},
    ptx_helpers::ptx_header,
};

// --------------------------------------------------------------------------- //
//  CPU reference implementations
// --------------------------------------------------------------------------- //

/// Helper: load pixel at `(row + dr, col + dc)` with zero-padding.
#[inline]
fn load_zeropad(image: &[f32], h: usize, w: usize, r: isize, c: isize) -> f32 {
    if r < 0 || r >= h as isize || c < 0 || c >= w as isize {
        0.0
    } else {
        image[r as usize * w + c as usize]
    }
}

/// Compute the horizontal Sobel gradient `Gx` for each pixel.
///
/// `Gx = (ne − nw) + 2·(e − w) + (se − sw)` where n/s/e/w indicate
/// north/south/east/west neighbours.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `image.len() != height * width`.
pub fn sobel_x(image: &[f32], height: usize, width: usize) -> SignalResult<Vec<f32>> {
    if image.len() != height * width {
        return Err(SignalError::InvalidSize(format!(
            "image length {} ≠ {height} × {width} = {}",
            image.len(),
            height * width
        )));
    }
    let mut gx = vec![0.0f32; height * width];
    for r in 0..height {
        for c in 0..width {
            let ri = r as isize;
            let ci = c as isize;
            let hi = height as isize;
            let wi = width as isize;
            let nw = load_zeropad(image, hi as usize, wi as usize, ri - 1, ci - 1);
            let ne = load_zeropad(image, hi as usize, wi as usize, ri - 1, ci + 1);
            let w_px = load_zeropad(image, hi as usize, wi as usize, ri, ci - 1);
            let e_px = load_zeropad(image, hi as usize, wi as usize, ri, ci + 1);
            let sw = load_zeropad(image, hi as usize, wi as usize, ri + 1, ci - 1);
            let se = load_zeropad(image, hi as usize, wi as usize, ri + 1, ci + 1);
            gx[r * width + c] = (ne - nw) + 2.0 * (e_px - w_px) + (se - sw);
        }
    }
    Ok(gx)
}

/// Compute the vertical Sobel gradient `Gy` for each pixel.
///
/// `Gy = (sw − nw) + 2·(s − n) + (se − ne)`.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `image.len() != height * width`.
pub fn sobel_y(image: &[f32], height: usize, width: usize) -> SignalResult<Vec<f32>> {
    if image.len() != height * width {
        return Err(SignalError::InvalidSize(format!(
            "image length {} ≠ {height} × {width} = {}",
            image.len(),
            height * width
        )));
    }
    let mut gy = vec![0.0f32; height * width];
    for r in 0..height {
        for c in 0..width {
            let ri = r as isize;
            let ci = c as isize;
            let hi = height as isize;
            let wi = width as isize;
            let nw = load_zeropad(image, hi as usize, wi as usize, ri - 1, ci - 1);
            let n_px = load_zeropad(image, hi as usize, wi as usize, ri - 1, ci);
            let ne = load_zeropad(image, hi as usize, wi as usize, ri - 1, ci + 1);
            let sw = load_zeropad(image, hi as usize, wi as usize, ri + 1, ci - 1);
            let s_px = load_zeropad(image, hi as usize, wi as usize, ri + 1, ci);
            let se = load_zeropad(image, hi as usize, wi as usize, ri + 1, ci + 1);
            gy[r * width + c] = (sw - nw) + 2.0 * (s_px - n_px) + (se - ne);
        }
    }
    Ok(gy)
}

/// Compute the gradient magnitude `|G| = √(Gx² + Gy²)` element-wise.
///
/// Both slices must have the same length.
#[must_use]
pub fn sobel_magnitude(gx: &[f32], gy: &[f32]) -> Vec<f32> {
    gx.iter()
        .zip(gy.iter())
        .map(|(&gx_v, &gy_v)| (gx_v * gx_v + gy_v * gy_v).sqrt())
        .collect()
}

/// Compute the gradient orientation `θ = atan2(Gy, Gx)` in radians, element-wise.
///
/// Result is in `[−π, π]`.  Both slices must have the same length.
#[must_use]
pub fn sobel_angle(gx: &[f32], gy: &[f32]) -> Vec<f32> {
    gx.iter()
        .zip(gy.iter())
        .map(|(&gx_v, &gy_v)| gy_v.atan2(gx_v))
        .collect()
}

/// Compute `Gx`, `Gy`, gradient magnitude, and orientation in a single pass.
///
/// Returns `(gx, gy, magnitude, angle)`.
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `image.len() != height * width`.
#[allow(clippy::type_complexity)]
pub fn sobel(
    image: &[f32],
    height: usize,
    width: usize,
) -> SignalResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let gx = sobel_x(image, height, width)?;
    let gy = sobel_y(image, height, width)?;
    let mag = sobel_magnitude(&gx, &gy);
    let ang = sobel_angle(&gx, &gy);
    Ok((gx, gy, mag, ang))
}

// --------------------------------------------------------------------------- //
//  PTX kernel generation
// --------------------------------------------------------------------------- //

/// Generate PTX for loading pixel at `(row + dr, col + dc)` with zero-padding.
///
/// Stores the result into float register `f_reg` (e.g. `"%f1"`).
/// Uses unique label `ln{idx}` to avoid collision across unrolled loads.
///
/// Assumes the following registers are already set up in the calling kernel:
/// - `%s0` = current row (s32)
/// - `%s1` = current col (s32)
/// - `%s2` = height (s32)
/// - `%s3` = width  (s32)
/// - `%r1`  = width  (u32)
/// - `%rd0` = base address of input
///
/// Temporaries used (safe to reuse between calls since they are sequential):
/// `%s4`, `%s5`, `%r9`, `%r10`, `%rd5`, `%p2`, `%p3`.
fn load_neighbor_ptx(f_reg: &str, idx: usize, dr: i32, dc: i32) -> String {
    let skip = format!("ln{idx}");
    format!(
        "    add.s32       %s4, %s0, {dr};\n\
         \x20   add.s32       %s5, %s1, {dc};\n\
         \x20   mov.f32       {f_reg}, 0f00000000;\n\
         \x20   setp.lt.s32   %p2, %s4, 0;\n\
         \x20   @%p2 bra      {skip};\n\
         \x20   setp.ge.s32   %p2, %s4, %s2;\n\
         \x20   @%p2 bra      {skip};\n\
         \x20   setp.lt.s32   %p3, %s5, 0;\n\
         \x20   @%p3 bra      {skip};\n\
         \x20   setp.ge.s32   %p3, %s5, %s3;\n\
         \x20   @%p3 bra      {skip};\n\
         \x20   cvt.u32.s32   %r9,  %s4;\n\
         \x20   mul.lo.u32    %r9,  %r9, %r1;\n\
         \x20   cvt.u32.s32   %r10, %s5;\n\
         \x20   add.u32       %r9,  %r9, %r10;\n\
         \x20   cvt.u64.u32   %rd5, %r9;\n\
         \x20   shl.b64       %rd5, %rd5, 2;\n\
         \x20   add.u64       %rd5, %rd0, %rd5;\n\
         \x20   ld.global.f32 {f_reg}, [%rd5];\n\
{skip}:\n"
    )
}

/// Emit the PTX Sobel Gx kernel for f32 images.
///
/// Kernel entry: `signal_sobel_gx_f32`
///
/// Parameters: `(in: *f32, out: *f32, height: u32, width: u32, n: u64)`
#[must_use]
pub fn emit_sobel_x_kernel(sm: SmVersion) -> String {
    // Loads for Gx (only non-zero Sobel taps):
    //   %f1 = p_nw  (r-1, c-1)   weight -1
    //   %f2 = p_ne  (r-1, c+1)   weight +1
    //   %f3 = p_w   (r,   c-1)   weight -2
    //   %f4 = p_e   (r,   c+1)   weight +2
    //   %f5 = p_sw  (r+1, c-1)   weight -1
    //   %f6 = p_se  (r+1, c+1)   weight +1
    // Gx = (f2-f1) + 2*(f4-f3) + (f6-f5)
    let loads = format!(
        "{}{}{}{}{}{}",
        load_neighbor_ptx("%f1", 0, -1, -1),
        load_neighbor_ptx("%f2", 1, -1, 1),
        load_neighbor_ptx("%f3", 2, 0, -1),
        load_neighbor_ptx("%f4", 3, 0, 1),
        load_neighbor_ptx("%f5", 4, 1, -1),
        load_neighbor_ptx("%f6", 5, 1, 1),
    );
    let compute = "\
    // Gx = (ne - nw) + 2*(e - w) + (se - sw)\n\
    sub.f32      %f0, %f2, %f1;\n\
    sub.f32      %f7, %f4, %f3;\n\
    fma.rn.f32   %f0, %f7, 0f40000000, %f0;\n\
    sub.f32      %f7, %f6, %f5;\n\
    add.f32      %f0, %f0, %f7;\n";
    emit_sobel_kernel(sm, "signal_sobel_gx_f32", &loads, compute)
}

/// Emit the PTX Sobel Gy kernel for f32 images.
///
/// Kernel entry: `signal_sobel_gy_f32`
///
/// Parameters: `(in: *f32, out: *f32, height: u32, width: u32, n: u64)`
#[must_use]
pub fn emit_sobel_y_kernel(sm: SmVersion) -> String {
    // Loads for Gy (only non-zero Sobel taps):
    //   %f1 = p_nw  (r-1, c-1)   weight -1
    //   %f2 = p_n   (r-1, c)     weight -2
    //   %f3 = p_ne  (r-1, c+1)   weight -1
    //   %f4 = p_sw  (r+1, c-1)   weight +1
    //   %f5 = p_s   (r+1, c)     weight +2
    //   %f6 = p_se  (r+1, c+1)   weight +1
    // Gy = (f4-f1) + 2*(f5-f2) + (f6-f3)
    let loads = format!(
        "{}{}{}{}{}{}",
        load_neighbor_ptx("%f1", 0, -1, -1),
        load_neighbor_ptx("%f2", 1, -1, 0),
        load_neighbor_ptx("%f3", 2, -1, 1),
        load_neighbor_ptx("%f4", 3, 1, -1),
        load_neighbor_ptx("%f5", 4, 1, 0),
        load_neighbor_ptx("%f6", 5, 1, 1),
    );
    let compute = "\
    // Gy = (sw - nw) + 2*(s - n) + (se - ne)\n\
    sub.f32      %f0, %f4, %f1;\n\
    sub.f32      %f7, %f5, %f2;\n\
    fma.rn.f32   %f0, %f7, 0f40000000, %f0;\n\
    sub.f32      %f7, %f6, %f3;\n\
    add.f32      %f0, %f0, %f7;\n";
    emit_sobel_kernel(sm, "signal_sobel_gy_f32", &loads, compute)
}

fn emit_sobel_kernel(sm: SmVersion, name: &str, loads: &str, compute: &str) -> String {
    // Register plan:
    //   %rd0 = in, %rd1 = out, %rd3 = n, %rd4 = tid (→ out offset), %rd5 = temp
    //   %r0 = height, %r1 = width
    //   %r4 = tid tmp, %r5 = ctaid, %r6 = ntid, %r7 = row, %r8 = col
    //   %r9, %r10 = pixel_idx temporaries
    //   %s0 = row (s32), %s1 = col (s32), %s2 = height (s32), %s3 = width (s32)
    //   %s4, %s5 = per-load temporaries
    //   %f0 = result, %f1–%f6 = neighbours, %f7 = computation temporary
    //   %p0 = tid oob, %p2, %p3 = per-load bounds predicates
    let hdr = ptx_header(sm);
    format!(
        r"{hdr}
.visible .entry {name}(
    .param .u64 param_in,
    .param .u64 param_out,
    .param .u32 param_height,
    .param .u32 param_width,
    .param .u64 param_n
)
{{
    .reg .pred  %p<4>;
    .reg .u32   %r<14>;
    .reg .u64   %rd<8>;
    .reg .f32   %f<8>;
    .reg .s32   %s<8>;

    ld.param.u64 %rd0, [param_in];
    ld.param.u64 %rd1, [param_out];
    ld.param.u32 %r0,  [param_height];
    ld.param.u32 %r1,  [param_width];
    ld.param.u64 %rd3, [param_n];

    mov.u32      %r4,  %tid.x;
    mov.u32      %r5,  %ctaid.x;
    mov.u32      %r6,  %ntid.x;
    mad.lo.u32   %r4,  %r5, %r6, %r4;
    cvt.u64.u32  %rd4, %r4;

    setp.ge.u64  %p0,  %rd4, %rd3;
    @%p0 bra     done;

    div.u32      %r7, %r4, %r1;   // row
    rem.u32      %r8, %r4, %r1;   // col

    cvt.s32.u32  %s0, %r7;        // row (s32)
    cvt.s32.u32  %s1, %r8;        // col (s32)
    cvt.s32.u32  %s2, %r0;        // height (s32)
    cvt.s32.u32  %s3, %r1;        // width  (s32)

    // Load 6 neighbours with zero-padding.
{loads}
    // Compute gradient component.
{compute}
    // Write output.
    shl.b64      %rd4, %rd4, 2;
    add.u64      %rd4, %rd1, %rd4;
    st.global.f32 [%rd4], %f0;

done:
    ret;
}}
",
        name = name,
        loads = loads,
        compute = compute,
    )
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_sobel_x_constant_image_zero() {
        // Constant image has zero horizontal gradient in the interior.
        // Border pixels are affected by zero-padding and may be non-zero.
        let img = vec![3.0f32; 25];
        let gx = sobel_x(&img, 5, 5).unwrap();
        for r in 1..4 {
            for c in 1..4 {
                let v = gx[r * 5 + c];
                assert!(
                    v == 0.0,
                    "constant image interior Gx should be 0 at ({r},{c}), got {v}"
                );
            }
        }
    }

    #[test]
    fn test_sobel_y_constant_image_zero() {
        // Constant image has zero vertical gradient in the interior.
        let img = vec![3.0f32; 25];
        let gy = sobel_y(&img, 5, 5).unwrap();
        for r in 1..4 {
            for c in 1..4 {
                let v = gy[r * 5 + c];
                assert!(
                    v == 0.0,
                    "constant image interior Gy should be 0 at ({r},{c}), got {v}"
                );
            }
        }
    }

    #[test]
    fn test_sobel_x_vertical_edge() {
        // Step edge in column direction: left half = 0, right half = 1.
        // Horizontal Sobel should detect this.
        let img: Vec<f32> = (0..25).map(|i| if i % 5 < 3 { 0.0 } else { 1.0 }).collect();
        let gx = sobel_x(&img, 5, 5).unwrap();
        let max_abs = gx.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_abs > 0.0, "should detect vertical edge with Gx");
    }

    #[test]
    fn test_sobel_y_horizontal_edge() {
        // Step edge in row direction: top half = 0, bottom half = 1.
        let img: Vec<f32> = (0..25).map(|i| if i / 5 < 3 { 0.0 } else { 1.0 }).collect();
        let gy = sobel_y(&img, 5, 5).unwrap();
        let max_abs = gy.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_abs > 0.0, "should detect horizontal edge with Gy");
    }

    #[test]
    fn test_sobel_x_on_ramp_interior() {
        // A row-ramp p[r][c] = c has Gx ≠ 0 in the interior.
        let img: Vec<f32> = (0..25).map(|i| (i % 5) as f32).collect();
        let gx = sobel_x(&img, 5, 5).unwrap();
        // At interior pixel (2, 2): Gx = (1 + 2 + 1) = 4 (all taps non-zero).
        // Actually: ne - nw = (0-4) - (0-0) ... let me compute manually.
        // For img[r][c] = c: ne = c+1, nw = c-1, e = c+1, w = c-1, se = c+1, sw = c-1
        // Gx = (ne-nw) + 2*(e-w) + (se-sw) = 2 + 4 + 2 = 8? Let me check:
        // ne = img[r-1][c+1] = (c+1), nw = img[r-1][c-1] = (c-1)
        // => ne - nw = 2
        // e = img[r][c+1] = (c+1), w = img[r][c-1] = (c-1)
        // => e - w = 2
        // se = img[r+1][c+1] = (c+1), sw = img[r+1][c-1] = (c-1)
        // => se - sw = 2
        // Gx = 2 + 2*2 + 2 = 8
        let interior_gx = gx[2 * 5 + 2];
        assert!(
            (interior_gx - 8.0).abs() < 1e-4,
            "Gx at interior = {interior_gx}"
        );
    }

    #[test]
    fn test_sobel_magnitude_non_negative() {
        let img: Vec<f32> = (0..25).map(|i| (i as f32).sin()).collect();
        let gx = sobel_x(&img, 5, 5).unwrap();
        let gy = sobel_y(&img, 5, 5).unwrap();
        let mag = sobel_magnitude(&gx, &gy);
        assert!(mag.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_sobel_angle_range() {
        let img: Vec<f32> = (0..25).map(|i| (i as f32 * 0.1).cos()).collect();
        let gx = sobel_x(&img, 5, 5).unwrap();
        let gy = sobel_y(&img, 5, 5).unwrap();
        let ang = sobel_angle(&gx, &gy);
        for &a in &ang {
            assert!(
                (-PI - 1e-5..=PI + 1e-5).contains(&a),
                "angle out of range: {a}"
            );
        }
    }

    #[test]
    fn test_sobel_returns_four_outputs() {
        let img = vec![1.0f32; 9];
        let (gx, gy, mag, ang) = sobel(&img, 3, 3).unwrap();
        assert_eq!(gx.len(), 9);
        assert_eq!(gy.len(), 9);
        assert_eq!(mag.len(), 9);
        assert_eq!(ang.len(), 9);
    }

    #[test]
    fn test_sobel_invalid_size() {
        let img = vec![1.0f32; 8];
        assert!(sobel_x(&img, 3, 3).is_err());
        assert!(sobel_y(&img, 3, 3).is_err());
    }

    #[test]
    fn test_emit_sobel_x_ptx_structure() {
        let ptx = emit_sobel_x_kernel(SmVersion::Sm80);
        assert!(ptx.contains("signal_sobel_gx_f32"), "missing entry name");
        // Should have 6 unique load labels.
        for i in 0..6 {
            assert!(ptx.contains(&format!("ln{i}:")), "missing label ln{i}");
        }
        assert!(ptx.contains("fma.rn.f32"), "missing FMA instruction");
    }

    #[test]
    fn test_emit_sobel_y_ptx_structure() {
        let ptx = emit_sobel_y_kernel(SmVersion::Sm80);
        assert!(ptx.contains("signal_sobel_gy_f32"), "missing entry name");
        for i in 0..6 {
            assert!(ptx.contains(&format!("ln{i}:")), "missing label ln{i}");
        }
    }
}
