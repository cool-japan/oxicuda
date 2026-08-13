//! Winograd transform matrices and their host-side reference application.
//!
//! Every Winograd stage in this crate is a **congruence transform**
//! `out = M * x * M^T` for one of three matrices:
//!
//! | Stage | Matrix | Shape (F(2,3)) | Meaning |
//! |---|---|---|---|
//! | input | `B^T` | 4x4 | `V = B^T d B`, and `B = (B^T)^T` |
//! | filter | `G` | 4x3 | `U = G g G^T` |
//! | output | `A^T` | 2x4 | `Y = A^T m A`, and `A = (A^T)^T` |
//!
//! Writing all three as `M x M^T` is not a notational convenience — it is the
//! property [`congruence`] and the PTX emitter both rely on, so a single
//! generic routine drives every stage from the same constants the tests check.
//! The `B`/`A`/`G^T` constants below are therefore *derived* data kept for
//! readers (and for the dgrad/wgrad modules that quote them); the transforms
//! only ever read `BT_F2X3`, `G_F2X3` and `AT_F2X3`.
//!
//! # Convention
//!
//! `F(2,3)` computes `y_i = sum_j d_{i+j} * g_j` — **cross-correlation**, the
//! cuDNN convention used by every other forward engine in this crate (no 180
//! degree kernel flip). [`tests::f2x3_tile_matches_direct_cross_correlation`]
//! pins that down against a direct 2x2-from-4x4 evaluation.

// ---------------------------------------------------------------------------
// F(2,3) matrices
// ---------------------------------------------------------------------------

/// `B^T` for F(2,3): the 4x4 input transform.
///
/// ```text
/// B^T = [[1,  0, -1,  0],
///        [0,  1,  1,  0],
///        [0, -1,  1,  0],
///        [0,  1,  0, -1]]
/// ```
#[rustfmt::skip]
pub(crate) const BT_F2X3: [[f32; 4]; 4] = [
    [ 1.0,  0.0, -1.0,  0.0],
    [ 0.0,  1.0,  1.0,  0.0],
    [ 0.0, -1.0,  1.0,  0.0],
    [ 0.0,  1.0,  0.0, -1.0],
];

/// `B` for F(2,3): the transpose of [`BT_F2X3`].
///
/// Derived data: the emitters build the right-hand factor from `B^T` itself
/// (`M x M^T`), so nothing outside the consistency test reads this. It is
/// `cfg(test)` rather than deleted because
/// [`tests::transpose_constants_are_consistent`] is what proves the
/// documented matrix and the emitted kernel agree.
#[cfg(test)]
#[rustfmt::skip]
pub(crate) const B_F2X3: [[f32; 4]; 4] = [
    [ 1.0,  0.0,  0.0,  0.0],
    [ 0.0,  1.0, -1.0,  1.0],
    [-1.0,  1.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0, -1.0],
];

/// `A^T` for F(2,3): the 2x4 output transform.
///
/// ```text
/// A^T = [[1, 1,  1,  0],
///        [0, 1, -1, -1]]
/// ```
#[rustfmt::skip]
pub(crate) const AT_F2X3: [[f32; 4]; 2] = [
    [1.0,  1.0,  1.0,  0.0],
    [0.0,  1.0, -1.0, -1.0],
];

/// `A` for F(2,3): the transpose of [`AT_F2X3`]. Derived data — see [`B_F2X3`].
#[cfg(test)]
#[rustfmt::skip]
pub(crate) const A_F2X3: [[f32; 2]; 4] = [
    [1.0,  0.0],
    [1.0,  1.0],
    [1.0, -1.0],
    [0.0, -1.0],
];

/// `G` for F(2,3): the 4x3 filter transform.
///
/// ```text
/// G = [[1,    0,    0  ],
///      [0.5,  0.5,  0.5],
///      [0.5, -0.5,  0.5],
///      [0,    0,    1  ]]
/// ```
#[rustfmt::skip]
pub(crate) const G_F2X3: [[f32; 3]; 4] = [
    [1.0,     0.0,    0.0  ],
    [0.5,     0.5,    0.5  ],
    [0.5,    -0.5,    0.5  ],
    [0.0,     0.0,    1.0  ],
];

/// `G^T` for F(2,3): the transpose of [`G_F2X3`]. Derived data — see [`B_F2X3`].
#[cfg(test)]
#[rustfmt::skip]
pub(crate) const GT_F2X3: [[f32; 4]; 3] = [
    [1.0,  0.5,  0.5, 0.0],
    [0.0,  0.5, -0.5, 0.0],
    [0.0,  0.5,  0.5, 1.0],
];

// ---------------------------------------------------------------------------
// F(4,3) matrices
// ---------------------------------------------------------------------------
//
// The forward engine implements F(2,3) only (see
// `WinogradTileSize::forward_supported`), so today these are consumed purely
// by the dgrad/wgrad modules' tests, which quote them as their own reference
// coefficients. They are `cfg(test)` for exactly that reason; a forward F(4,3)
// engine would drop the attribute and nothing else.

/// `B^T` for F(4,3): the 6x6 input transform.
#[cfg(test)]
#[rustfmt::skip]
pub(crate) const BT_F4X3: [[f32; 6]; 6] = [
    [ 4.0,  0.0, -5.0,  0.0,  1.0, 0.0],
    [ 0.0, -4.0, -4.0,  1.0,  1.0, 0.0],
    [ 0.0,  4.0, -4.0, -1.0,  1.0, 0.0],
    [ 0.0, -2.0, -1.0,  2.0,  1.0, 0.0],
    [ 0.0,  2.0, -1.0, -2.0,  1.0, 0.0],
    [ 0.0,  4.0,  0.0, -5.0,  0.0, 1.0],
];

/// `A^T` for F(4,3): the 4x6 output transform.
#[cfg(test)]
#[rustfmt::skip]
pub(crate) const AT_F4X3: [[f32; 6]; 4] = [
    [1.0,  1.0,  1.0,  1.0,  1.0, 0.0],
    [0.0,  1.0, -1.0,  2.0, -2.0, 0.0],
    [0.0,  1.0,  1.0,  4.0,  4.0, 0.0],
    [0.0,  1.0, -1.0,  8.0, -8.0, 1.0],
];

/// `G` for F(4,3): the 6x3 filter transform.
#[cfg(test)]
#[rustfmt::skip]
pub(crate) const G_F4X3: [[f32; 3]; 6] = [
    [ 1.0/4.0,   0.0,       0.0      ],
    [-1.0/6.0,  -1.0/6.0,  -1.0/6.0  ],
    [-1.0/6.0,   1.0/6.0,  -1.0/6.0  ],
    [ 1.0/24.0,  1.0/12.0,  1.0/6.0  ],
    [ 1.0/24.0, -1.0/12.0,  1.0/6.0  ],
    [ 0.0,       0.0,       1.0      ],
];

// ---------------------------------------------------------------------------
// Host-side reference transforms
// ---------------------------------------------------------------------------

/// Applies the congruence transform `out = M * x * M^T` in `f64`.
///
/// `m` is `rows x cols` (given as slices so the F(2,3) `A^T` (2x4), `B^T`
/// (4x4) and `G` (4x3) shapes all go through the same routine); `x` is
/// `cols x cols`; the result is `rows x rows`.
///
/// This is the exact operation the generated PTX performs, expressed in one
/// place so the kernel emitter and the CPU oracle cannot drift apart: the
/// emitter walks the same two stages over the same `m`.
///
/// # Panics
///
/// Panics if the row lengths of `m` or `x` disagree with the deduced
/// dimensions. Only reachable from in-crate callers with compile-time-known
/// matrices.
///
/// `cfg(test)`: the GPU does the real work, and this exists to check it — the
/// unit tests below and the on-device oracle in `gpu_tests::conv_fprop`.
#[cfg(test)]
pub(crate) fn congruence(m: &[&[f32]], x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = m.len();
    let cols = m[0].len();
    assert!(m.iter().all(|r| r.len() == cols), "M is not rectangular");
    assert_eq!(x.len(), cols, "x must be cols x cols");
    assert!(x.iter().all(|r| r.len() == cols), "x is not square");

    // Stage 1: t = M * x   (rows x cols)
    let mut t = vec![vec![0.0f64; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = 0.0f64;
            for (k, x_k) in x.iter().enumerate() {
                acc += f64::from(m[i][k]) * x_k[j];
            }
            t[i][j] = acc;
        }
    }

    // Stage 2: out = t * M^T   (rows x rows); column j of M^T is row j of M.
    let mut out = vec![vec![0.0f64; rows]; rows];
    for i in 0..rows {
        for j in 0..rows {
            let mut acc = 0.0f64;
            for k in 0..cols {
                acc += t[i][k] * f64::from(m[j][k]);
            }
            out[i][j] = acc;
        }
    }
    out
}

/// `B^T` as a slice-of-slices view, for [`congruence`].
pub(crate) fn bt_f2x3_view() -> Vec<&'static [f32]> {
    BT_F2X3.iter().map(|r| r.as_slice()).collect()
}

/// `G` as a slice-of-slices view, for [`congruence`].
pub(crate) fn g_f2x3_view() -> Vec<&'static [f32]> {
    G_F2X3.iter().map(|r| r.as_slice()).collect()
}

/// `A^T` as a slice-of-slices view, for [`congruence`].
pub(crate) fn at_f2x3_view() -> Vec<&'static [f32]> {
    AT_F2X3.iter().map(|r| r.as_slice()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_rows(x: &[[f64; 4]; 4]) -> Vec<Vec<f64>> {
        x.iter().map(|r| r.to_vec()).collect()
    }

    /// Every "transpose" constant really is the transpose of its partner. The
    /// PTX emitter derives the right-hand factor from the left-hand matrix
    /// (`M x M^T`), so a mismatch here would mean the documented constants and
    /// the emitted kernel disagree.
    #[test]
    fn transpose_constants_are_consistent() {
        for i in 0..4 {
            for j in 0..4 {
                assert!((B_F2X3[i][j] - BT_F2X3[j][i]).abs() < 1e-12, "B vs B^T");
            }
        }
        for i in 0..4 {
            for j in 0..2 {
                assert!((A_F2X3[i][j] - AT_F2X3[j][i]).abs() < 1e-12, "A vs A^T");
            }
        }
        for i in 0..3 {
            for j in 0..4 {
                assert!((GT_F2X3[i][j] - G_F2X3[j][i]).abs() < 1e-12, "G^T vs G");
            }
        }
    }

    /// The three-stage F(2,3) pipeline must reproduce a direct 2x2-output
    /// cross-correlation of a 4x4 patch with a 3x3 filter, exactly (to `f64`
    /// round-off). This is the mathematical contract the whole engine rests
    /// on; if it fails, no amount of GPU work can be correct.
    #[test]
    fn f2x3_tile_matches_direct_cross_correlation() {
        // A deterministic, non-symmetric patch and filter: symmetric data can
        // hide a transposed transform.
        let mut d = [[0.0f64; 4]; 4];
        for (i, row) in d.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (i as f64) * 3.25 - (j as f64) * 1.5 + ((i * 4 + j) as f64) * 0.125;
            }
        }
        let mut g = [[0.0f64; 3]; 3];
        for (i, row) in g.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = 0.5 - (i as f64) * 0.25 + (j as f64) * 0.75;
            }
        }

        // Direct reference: y[i][j] = sum_{r,s} d[i+r][j+s] * g[r][s].
        let mut expected = [[0.0f64; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0;
                for r in 0..3 {
                    for s in 0..3 {
                        acc += d[i + r][j + s] * g[r][s];
                    }
                }
                expected[i][j] = acc;
            }
        }

        // Winograd: Y = A^T ((G g G^T) . (B^T d B)) A.
        let bt = bt_f2x3_view();
        let gv = g_f2x3_view();
        let at = at_f2x3_view();
        let v = congruence(&bt, &to_rows(&d));
        let g_rows: Vec<Vec<f64>> = g.iter().map(|r| r.to_vec()).collect();
        let u = congruence(&gv, &g_rows);
        let mut m = vec![vec![0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = u[i][j] * v[i][j];
            }
        }
        let y = congruence(&at, &m);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (y[i][j] - expected[i][j]).abs() < 1e-9,
                    "F(2,3) tile mismatch at ({i},{j}): winograd={} direct={}",
                    y[i][j],
                    expected[i][j]
                );
            }
        }
    }

    /// A delta filter must transform to the outer product `G[:,1] G[:,1]^T`:
    /// an independent check that the filter stage really is `G g G^T` and not,
    /// say, `G g G` (which agrees for symmetric `g`).
    #[test]
    fn delta_filter_transform_is_outer_product() {
        let g = vec![
            vec![0.0f64, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        let u = congruence(&g_f2x3_view(), &g);
        for i in 0..4 {
            for j in 0..4 {
                let expected = f64::from(G_F2X3[i][1]) * f64::from(G_F2X3[j][1]);
                assert!((u[i][j] - expected).abs() < 1e-12, "U[{i}][{j}]");
            }
        }
    }

    /// The transform of an all-zero patch is all-zero — the property the
    /// kernel's zero-padding path depends on (out-of-range taps contribute
    /// nothing, so a fully padded tile produces a zero output tile).
    #[test]
    fn zero_patch_transforms_to_zero() {
        let d = vec![vec![0.0f64; 4]; 4];
        let v = congruence(&bt_f2x3_view(), &d);
        assert!(v.iter().flatten().all(|&x| x == 0.0));
    }

    /// Shape sanity for the F(4,3) constants retained for dgrad/wgrad.
    #[test]
    fn f4x3_matrix_shapes() {
        assert_eq!(BT_F4X3.len(), 6);
        assert_eq!(BT_F4X3[0].len(), 6);
        assert_eq!(AT_F4X3.len(), 4);
        assert_eq!(AT_F4X3[0].len(), 6);
        assert_eq!(G_F4X3.len(), 6);
        assert_eq!(G_F4X3[0].len(), 3);
    }
}
