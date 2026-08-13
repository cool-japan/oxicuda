//! On-device shape-sweep regression test for [`GemmDispatcher`]'s GEMM launch
//! occupancy.
//!
//! # Background
//!
//! An investigation into `oxiface`'s CUDA performance reported that
//! `GemmDispatcher::compute_grid`/`compute_block` (fed by
//! `skinny_tile_config`) under-provisions the launch for skinny/small-`M`
//! shapes — concretely, the ArcFace/InSwapper-dominant shape `M=1, K=25088,
//! N=512` was measured launching only 512 total threads across 8 CTAs — and
//! hypothesised this leaves output elements uncovered (reads back as `0.0`,
//! consistent with a fresh/zeroed allocation the launch never touches).
//!
//! Direct on-device reproduction (this test's own history) shows the launch
//! *does* cover every output element — `GemmTemplate::generate`'s naive
//! kernel walks the flattened `M*N` space with a 64-bit grid-stride loop
//! sized from the kernel's own runtime `gridDim`/`blockDim` queries, which is
//! correct for any grid the dispatcher hands it, including the exact 8-CTA/
//! 512-thread grid measured above. The *occupancy* half of the report is
//! real, though: 512 threads each doing a serial 25088-element reduction is
//! a severe under-utilisation of an Ampere-class GPU (tens of thousands of
//! schedulable threads), measured here at ~17x slower than necessary (see
//! `zz_bench_splitk.rs`/this crate's history) — `GemmDispatcher::dispatch`
//! now routes exactly this shape (`Skinny` category, small `M*N`, large `K`)
//! through a genuine two-pass split-K launch (`dispatch_skinny_split_k`,
//! `level3::gemm::splitk`) that parallelises the *reduction* itself instead
//! of capping out at one thread per output element.
//!
//! This file is the numeric regression test for both: every shape below is
//! checked *element-for-element* (not sampled) against a naive `f64`-
//! accumulated CPU oracle, through the exact production entry point
//! (`level3::gemm::gemm`, i.e. `GemmDispatcher::dispatch`) — not a
//! hand-rolled grid/block the test computes itself. `1024^3`/`2048^3` are
//! included because a sibling investigation stream measured their
//! *throughput* but never checked their numeric correctness.

use std::sync::Arc;

use oxicuda_blas::handle::BlasHandle;
use oxicuda_blas::level3::gemm;
use oxicuda_blas::types::{Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

// ---------------------------------------------------------------------------
// Fixture & helpers
// ---------------------------------------------------------------------------

/// Acquire a GPU handle, or `None` when no driver / device is present (the
/// suite then skips, matching this crate's `src/gpu_tests.rs` convention).
fn try_handle() -> Option<(Arc<Context>, BlasHandle)> {
    oxicuda_driver::init().ok()?;
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    let handle = BlasHandle::new(&ctx).ok()?;
    Some((ctx, handle))
}

/// A small deterministic LCG (mirrors `src/gpu_tests.rs`'s `Lcg`, duplicated
/// here as this file is a separate `tests/` compilation unit).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    /// Uniform `f32` in `[lo, hi)`.
    fn range_f32(&mut self, lo: f64, hi: f64) -> f32 {
        let unit = f64::from(self.next_u32()) / 4_294_967_296.0;
        (lo + (hi - lo) * unit) as f32
    }
}

fn make_matrix(rng: &mut Lcg, len: usize, lo: f64, hi: f64) -> Vec<f32> {
    (0..len).map(|_| rng.range_f32(lo, hi)).collect()
}

/// Naive `f64`-accumulated CPU oracle: `C = A(MxK) @ B(KxN)`, row-major.
/// Single-threaded; used for every shape except the `1024^3`/`2048^3` cubes.
fn cpu_gemm_f64(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

/// Same oracle as [`cpu_gemm_f64`], parallelised over row-chunks with
/// `std::thread::scope` (no extra crate — every worker just reduces its own
/// row range). Used only for `1024^3`/`2048^3`, where the serial version
/// would take tens of seconds.
fn cpu_gemm_f64_parallel(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let workers = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .clamp(1, m.max(1));
    let mut out = vec![0.0f32; m * n];
    let rows_per_worker = m.div_ceil(workers);

    std::thread::scope(|scope| {
        for (chunk_idx, out_chunk) in out.chunks_mut(rows_per_worker * n).enumerate() {
            let row_start = chunk_idx * rows_per_worker;
            scope.spawn(move || {
                let rows_here = out_chunk.len() / n;
                for local_i in 0..rows_here {
                    let i = row_start + local_i;
                    for j in 0..n {
                        let mut acc = 0.0f64;
                        for p in 0..k {
                            acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
                        }
                        out_chunk[local_i * n + j] = acc as f32;
                    }
                }
            });
        }
    });
    out
}

/// Runs the *production* GEMM entry point (`level3::gemm::gemm`, i.e.
/// `GemmDispatcher::dispatch` with whatever tile config / grid / block /
/// launch-kind it chooses for this shape — nothing here overrides that) and
/// returns the host-side result.
fn run_gemm(handle: &BlasHandle, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let c0 = vec![0.0f32; m * n];
    let d_a = DeviceBuffer::<f32>::from_host(a).expect("upload A");
    let d_b = DeviceBuffer::<f32>::from_host(b).expect("upload B");
    let mut d_c = DeviceBuffer::<f32>::from_host(&c0).expect("upload C0");

    let a_desc =
        MatrixDesc::from_buffer(&d_a, m as u32, k as u32, Layout::RowMajor).expect("A descriptor");
    let b_desc =
        MatrixDesc::from_buffer(&d_b, k as u32, n as u32, Layout::RowMajor).expect("B descriptor");
    let mut c_desc = MatrixDescMut::from_buffer(&mut d_c, m as u32, n as u32, Layout::RowMajor)
        .expect("C descriptor");

    gemm::<f32>(
        handle,
        Transpose::NoTrans,
        Transpose::NoTrans,
        1.0f32,
        &a_desc,
        &b_desc,
        0.0f32,
        &mut c_desc,
    )
    .expect("GEMM dispatch");
    handle.stream().synchronize().expect("stream sync");

    let mut got = vec![0.0f32; m * n];
    d_c.copy_to_host(&mut got).expect("copy C back");
    got
}

/// Asserts every element of `got` agrees with `expect` (the `f64` oracle),
/// reporting every disagreement found (not just the first) so a real
/// coverage gap (as opposed to an isolated rounding edge case) is obvious
/// from the failure message shape.
///
/// Tolerance: `f32` accumulation (the GPU's `fma.rn.f32` per K-step) against
/// an `f64`-accumulated oracle diverges by an amount that grows with `K`; an
/// absolute floor plus a relative term scaled to the oracle's own magnitude
/// comfortably separates "expected f32-vs-f64 rounding drift" from "a whole
/// element was never written" (which shows up as an exact `0.0` against a
/// nonzero expected value, or vice versa).
fn assert_matches_oracle(got: &[f32], expect: &[f32], tag: &str) {
    assert_eq!(got.len(), expect.len(), "{tag}: length mismatch");
    let mut mismatches = Vec::new();
    for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
        let tol = 1e-4 + 2e-3 * f64::from(e.abs());
        if f64::from((g - e).abs()) > tol {
            mismatches.push((i, g, e));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{tag}: {} / {} elements mismatched (showing up to 10): {:?}",
        mismatches.len(),
        got.len(),
        &mismatches[..mismatches.len().min(10)],
    );
}

// ---------------------------------------------------------------------------
// The M sweep (task-mandated shapes)
// ---------------------------------------------------------------------------

/// `M` in `{1, 2, 7, 8, 16, 17, 32, 64, 100, 128, 1024, 2048}` crossed with
/// three `(K, N)` pairs — small/square, non-power-of-two, and a moderate
/// rectangle — covering both the `Skinny` category (M < 32, which is the
/// occupancy fix's target) and `Standard` (M >= 32, exercising the
/// unmodified single-pass path for a regression check on *that* code too).
/// Every element of every shape is checked; nothing here is sampled.
#[test]
fn m_sweep_matches_cpu_oracle_exactly() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("no CUDA device present, skipping m_sweep_matches_cpu_oracle_exactly");
        return;
    };

    let m_values = [1usize, 2, 7, 8, 16, 17, 32, 64, 100, 128, 1024, 2048];
    let kn_pairs = [(64usize, 64usize), (257, 129), (512, 384)];
    let mut rng = Lcg::new(0x5EED_F00D_1234_5678);

    for &m in &m_values {
        for &(k, n) in &kn_pairs {
            let a = make_matrix(&mut rng, m * k, -1.0, 1.0);
            let b = make_matrix(&mut rng, k * n, -1.0, 1.0);
            let got = run_gemm(&handle, &a, &b, m, k, n);
            let expect = cpu_gemm_f64(&a, &b, m, k, n);
            assert_matches_oracle(&got, &expect, &format!("M={m} K={k} N={n}"));
        }
    }
}

/// The two shapes named explicitly in the bug report: ArcFace's `1x512`
/// embedding projection (`M=1, K=25088, N=512`) and InSwapper's `1x512` emap
/// projection batched to `M=8`. Both are well inside the split-K trigger
/// (`M*N < 65536`, `K >= 512`), so this is also the direct regression test
/// for `dispatch_skinny_split_k` at oxiface's actual production shapes.
#[test]
fn arcface_and_inswapper_dominant_shapes_match_cpu_oracle() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!(
            "no CUDA device present, skipping arcface_and_inswapper_dominant_shapes_match_cpu_oracle"
        );
        return;
    };

    let mut rng = Lcg::new(0xA4CE_FACE_0000_0001);
    for &m in &[1usize, 8] {
        let (k, n) = (25088usize, 512usize);
        let a = make_matrix(&mut rng, m * k, -1.0, 1.0);
        let b = make_matrix(&mut rng, k * n, -1.0, 1.0);
        let got = run_gemm(&handle, &a, &b, m, k, n);
        let expect = cpu_gemm_f64(&a, &b, m, k, n);
        assert_matches_oracle(
            &got,
            &expect,
            &format!("M={m} K={k} N={n} (ArcFace/InSwapper)"),
        );
    }
}

/// The exact `64x64x64` all-ones reproduction from the bug report: the
/// report claimed elements past a fixed cutoff (~1024) silently read back
/// `0.0` instead of the correct `64.0`. Checked element-for-element.
#[test]
fn square_64_all_ones_matches_64_everywhere() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("no CUDA device present, skipping square_64_all_ones_matches_64_everywhere");
        return;
    };

    let (m, k, n) = (64usize, 64usize, 64usize);
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let got = run_gemm(&handle, &a, &b, m, k, n);
    for (i, &v) in got.iter().enumerate() {
        assert!(
            (v - 64.0).abs() < 1e-3,
            "element {i}: got {v}, expected 64.0 (64x64x64 all-ones reproduction)"
        );
    }
}

// ---------------------------------------------------------------------------
// Large square cubes: re-verify NUMERIC correctness, not just throughput.
// ---------------------------------------------------------------------------

/// `1024^3`. A sibling investigation stream measured this shape's
/// *throughput* only; this checks every one of its ~1.05M output elements
/// against the parallel `f64` oracle.
#[test]
fn square_1024_cubed_matches_cpu_oracle() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("no CUDA device present, skipping square_1024_cubed_matches_cpu_oracle");
        return;
    };
    let (m, k, n) = (1024usize, 1024usize, 1024usize);
    let mut rng = Lcg::new(0x1024_1024_1024_1024);
    let a = make_matrix(&mut rng, m * k, -1.0, 1.0);
    let b = make_matrix(&mut rng, k * n, -1.0, 1.0);
    let got = run_gemm(&handle, &a, &b, m, k, n);
    let expect = cpu_gemm_f64_parallel(&a, &b, m, k, n);
    assert_matches_oracle(&got, &expect, "M=K=N=1024");
}

/// `2048^3`. Same rationale as the 1024 cube; ~4.19M output elements.
#[test]
fn square_2048_cubed_matches_cpu_oracle() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("no CUDA device present, skipping square_2048_cubed_matches_cpu_oracle");
        return;
    };
    let (m, k, n) = (2048usize, 2048usize, 2048usize);
    let mut rng = Lcg::new(0x2048_2048_2048_2048);
    let a = make_matrix(&mut rng, m * k, -1.0, 1.0);
    let b = make_matrix(&mut rng, k * n, -1.0, 1.0);
    let got = run_gemm(&handle, &a, &b, m, k, n);
    let expect = cpu_gemm_f64_parallel(&a, &b, m, k, n);
    assert_matches_oracle(&got, &expect, "M=K=N=2048");
}
