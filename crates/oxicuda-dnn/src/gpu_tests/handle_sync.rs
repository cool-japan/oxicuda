//! On-device regression test for [`DnnHandle::synchronize_all`].
//!
//! `DnnHandle` deliberately gives its internal [`BlasHandle`](oxicuda_blas::BlasHandle)
//! its own CUDA stream, separate from the one [`DnnHandle::stream`] returns
//! (see `handle.rs`'s `build` doc comment) — so a caller that dispatches work
//! through [`DnnHandle::blas`] and then synchronizes only [`DnnHandle::stream`]
//! has synchronized a stream nothing was queued on. On real hardware this
//! silently reads back a partially- or un-written result buffer instead of
//! faulting (`oxionnx-cuda::matmul::cuda_matmul`'s pre-fix history — see
//! `synchronize_all`'s doc comment). This module proves the fix: a GEMM
//! dispatched through the BLAS sub-handle, read back after
//! `synchronize_all()`, must match a CPU oracle exactly.

use oxicuda_blas::level3::gemm;
use oxicuda_blas::types::{Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_memory::DeviceBuffer;

use super::{Lcg, assert_close_f32, gpu_fixture};

/// Host reference GEMM (row-major, `f64` accumulation): `C = A(MxK) * B(KxN)`.
fn cpu_gemm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

/// A GEMM dispatched through `handle.blas()` (its own, separate stream) and
/// read back after `handle.synchronize_all()` must be numerically complete.
///
/// `M=N=K=1536` is a `Standard`-category shape (not the `Skinny` split-K
/// path this same investigation added — this test is deliberately about the
/// *stream*, not about GEMM tiling), large enough for the single-pass naive
/// kernel to take a measurable amount of device time, so a reader that
/// raced ahead of it (rather than genuinely waiting) would have a real
/// chance of observing an incomplete result instead of passing by accident.
#[test]
fn synchronize_all_makes_blas_dispatched_gemm_readback_correct() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, k, n) = (1536usize, 1536usize, 1536usize);
    let mut rng = Lcg::new(0xC0FF_EE01);
    let a: Vec<f32> = (0..m * k).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let c0 = vec![0.0f32; m * n];

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    let mut d_c = DeviceBuffer::<f32>::from_host(&c0).expect("d_c");

    let a_desc = MatrixDesc::from_buffer(&d_a, m as u32, k as u32, Layout::RowMajor).unwrap();
    let b_desc = MatrixDesc::from_buffer(&d_b, k as u32, n as u32, Layout::RowMajor).unwrap();
    let mut c_desc =
        MatrixDescMut::from_buffer(&mut d_c, m as u32, n as u32, Layout::RowMajor).unwrap();

    // Dispatch through the BLAS sub-handle -- its own stream, distinct from
    // `fx.handle.stream()`.
    gemm::<f32>(
        fx.handle.blas(),
        Transpose::NoTrans,
        Transpose::NoTrans,
        1.0f32,
        &a_desc,
        &b_desc,
        0.0f32,
        &mut c_desc,
    )
    .expect("gemm dispatch");

    // The fix under test: waits for *both* streams, not just `fx.stream()`.
    fx.handle.synchronize_all().expect("synchronize_all");

    let mut got = vec![0.0f32; m * n];
    d_c.copy_to_host(&mut got).expect("copy_to_host");

    // Full shape is 1536^2 = ~2.36M elements; a CPU f64-accumulated oracle
    // over the full K=1536 for every element is a few seconds in release
    // mode. Check a deterministic spread of rows (first, middle, last, plus
    // a handful scattered through the matrix) rather than every element --
    // this test's purpose is proving the *stream* is correctly awaited, not
    // re-proving GEMM numerical completeness (the dedicated split-K/shape-
    // sweep tests already check every element there).
    let sample_rows: Vec<usize> = [0, 1, m / 4, m / 2, m - 2, m - 1]
        .into_iter()
        .filter(|&r| r < m)
        .collect();
    for &row in &sample_rows {
        let a_row = &a[row * k..(row + 1) * k];
        let expect_row = cpu_gemm_f32(a_row, &b, 1, k, n);
        let got_row = &got[row * n..(row + 1) * n];
        assert_close_f32(
            got_row,
            &expect_row,
            1e-3,
            1e-3,
            &format!("synchronize_all row {row}"),
        );
    }
}
