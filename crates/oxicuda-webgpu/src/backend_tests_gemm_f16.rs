//! Multi-tile numeric correctness tests for [`shader::gemm_wgsl_f16`]'s tiled
//! kernel (`WebGpuBackend::gemm_f16`).
//!
//! Split into a sibling file (mirroring `backend_tests_gpu_ops.rs` /
//! `backend_tests_pipeline.rs`) to keep `backend_tests.rs` under the
//! 2 000-line refactoring policy.
//!
//! # Why these tests exist specifically
//!
//! `gemm_wgsl_f16` used to be a plain per-thread dot-product loop with no
//! shared-memory tiling (`tile_size` only sized the workgroup). It was
//! rewritten to mirror `gemm_wgsl`'s `tile_size × tile_size` shared-memory
//! tiling — `var<workgroup> tile_a`/`tile_b`, staged in `f16`, widened to
//! `f32` only at the point of multiplication. That rewrite introduces a
//! failure mode the untiled kernel could not have: the tiled staging loop
//! reads `t * tile_size + lc` / `t * tile_size + lr`, which routinely exceeds
//! `params.k` / `params.m` on the last (partial) tile whenever a dimension is
//! not an exact multiple of `tile_size` — an unguarded `load_a`/`load_b`
//! would silently read a neighbouring row/column instead of contributing a
//! zero pad, corrupting the result (not crashing, so nothing short of a
//! numeric check would catch it).
//!
//! The existing f16 tests in `backend_tests.rs`
//! (`gemm_f16_matches_reference_2x3_times_3x2`,
//! `gemm_f16_honours_transpose_and_lda`) all use `k <= 3` against a
//! `tile_size` of 16, so `num_tiles == ceil(k / 16) == 1` — the partial-tile
//! path this rewrite added guards for is never exercised by them. The tests
//! below use `k = 40` (`ceil(40 / 16) == 3`, with a non-multiple-of-16
//! remainder on the last tile) specifically to exercise that path, across
//! every transpose combination and with a padded (non-packed) `lda`.
//!
//! Every operand is rounded through `half::f16` *before* being fed to the
//! `f32` CPU oracle (`cpu_gemm`, defined in `backend_tests.rs`), so the
//! oracle computes over exactly the values the GPU actually sees — isolating
//! the tiling/transpose/lda logic under test from ordinary f16
//! upload-quantisation noise, which would otherwise dominate a naive
//! f32-input-vs-f16-output comparison at these shapes.

use super::*;

/// Round every element of `data` through `half::f16` round-trip, so a CPU
/// oracle computed over the result matches exactly what the GPU's `f16`
/// storage buffers will contain after `upload_f16`.
fn round_to_f16(data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&x| half::f16::from_f32(x).to_f32())
        .collect()
}

/// Relative tolerance appropriate for f16 storage (~3-4 decimal digits of
/// precision): the accumulate itself is f32, so this bounds only the final
/// f16-rounding of the stored result, not accumulated tiling error.
fn f16_tol(expected: f32) -> f32 {
    1e-2 * (1.0 + expected.abs())
}

#[allow(clippy::too_many_arguments)]
fn run_gemm_f16_multi_tile_case(
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    m: usize,
    n: usize,
    k: usize,
) {
    let Some(b) = try_init() else { return };
    if !b.supports_f16() {
        return; // Adapter lacks SHADER_F16; nothing to exercise.
    }

    let ta = trans_a != BackendTranspose::NoTrans;
    let tb = trans_b != BackendTranspose::NoTrans;

    let a_data: Vec<f32> = (0..m * k).map(|x| (x as f32) * 0.02 - 1.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|x| (x as f32) * 0.015 + 0.3).collect();
    let c_init: Vec<f32> = (0..m * n).map(|x| (x as f32) * 0.01).collect();
    let alpha = 1.25f32;
    let beta = 0.75f32;

    let a_r = round_to_f16(&a_data);
    let b_r = round_to_f16(&b_data);
    let c_r = round_to_f16(&c_init);

    let expected = cpu_gemm(ta, tb, m, n, k, alpha, &a_r, &b_r, beta, &c_r);

    let a_h = upload_f16(&b, &a_r);
    let b_h = upload_f16(&b, &b_r);
    let c_h = upload_f16(&b, &c_r);

    b.gemm_f16(
        trans_a,
        trans_b,
        m,
        n,
        k,
        alpha as f64,
        a_h,
        if ta { m } else { k },
        b_h,
        if tb { k } else { n },
        beta as f64,
        c_h,
        n,
    )
    .expect("gemm_f16 multi-tile transpose case");

    let result = download_f16(&b, c_h, m * n);
    for (idx, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < f16_tol(*e),
            "trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, k={k}, slot={idx}: \
             got {r}, expected {e}"
        );
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

/// `k = 40` forces `ceil(40 / 16) == 3` tiles with a non-multiple-of-16
/// remainder on the last one; `m = 20`, `n = 18` also exceed the tile so
/// every axis' partial-tile guard is exercised. Every transpose combination
/// runs through the same shape.
#[test]
fn gemm_f16_multi_tile_matches_reference_across_transpose_combinations() {
    for &ta in &[BackendTranspose::NoTrans, BackendTranspose::Trans] {
        for &tb in &[BackendTranspose::NoTrans, BackendTranspose::Trans] {
            run_gemm_f16_multi_tile_case(ta, tb, 20, 18, 40);
        }
    }
}

/// TN with a genuinely padded `lda` (physical row stride wider than the
/// packed extent) at a multi-tile shape — proves `load_a`'s `lda`/transpose
/// handling and its last-partial-tile bounds guard compose correctly, not
/// just each in isolation. Mirrors `gemm_f16_honours_transpose_and_lda`
/// (which uses a tiny 2×3 shape, `num_tiles == 1`) at a shape that forces
/// multiple tile passes.
#[test]
fn gemm_f16_transpose_with_padded_lda_multi_tile_matches_reference() {
    let Some(b) = try_init() else { return };
    if !b.supports_f16() {
        return;
    }

    let m = 20usize;
    let k = 40usize;
    let n = 18usize;
    let lda_a = m + 3; // Padded well past the packed extent (m).
    let ldb = n;
    let ldc = n;

    // A_logical[row][i], row in 0..m, i in 0..k.
    let a_logical = |row: usize, i: usize| -> f32 { ((row * k + i) as f32) * 0.02 - 1.0 };

    // Packed (unpadded) transpose, k rows x m cols row-major: exactly the
    // physical layout `cpu_gemm(trans_a=true, ...)` expects.
    let mut a_packed_t = vec![0.0f32; k * m];
    for row in 0..m {
        for i in 0..k {
            a_packed_t[i * m + row] = a_logical(row, i);
        }
    }
    // Padded physical layout at `lda_a` — the extra columns hold a sentinel
    // the kernel must never read; if it did, the result would be wildly off
    // (not subtly), since the sentinel is far outside the data's range.
    let mut a_padded_t = vec![-9999.0f32; k * lda_a];
    for row in 0..m {
        for i in 0..k {
            a_padded_t[i * lda_a + row] = a_logical(row, i);
        }
    }

    let b_data: Vec<f32> = (0..k * n).map(|x| (x as f32) * 0.015 + 0.3).collect();
    let c_init: Vec<f32> = (0..m * n).map(|x| (x as f32) * 0.01).collect();
    let alpha = 1.5f32;
    let beta = 0.5f32;

    let a_packed_t_r = round_to_f16(&a_packed_t);
    let a_padded_t_r = round_to_f16(&a_padded_t);
    let b_r = round_to_f16(&b_data);
    let c_r = round_to_f16(&c_init);

    let expected = cpu_gemm(true, false, m, n, k, alpha, &a_packed_t_r, &b_r, beta, &c_r);

    let a_h = upload_f16(&b, &a_padded_t_r);
    let b_h = upload_f16(&b, &b_r);
    let c_h = upload_f16(&b, &c_r);

    b.gemm_f16(
        BackendTranspose::Trans,
        BackendTranspose::NoTrans,
        m,
        n,
        k,
        alpha as f64,
        a_h,
        lda_a,
        b_h,
        ldb,
        beta as f64,
        c_h,
        ldc,
    )
    .expect("gemm_f16 TN, padded lda, multi-tile k");

    let result = download_f16(&b, c_h, m * n);
    for (idx, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < f16_tol(*e),
            "slot={idx}: got {r}, expected {e}"
        );
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}
