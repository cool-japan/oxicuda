//! On-device regression test for the **cached split-K reduction workspace**.
//!
//! Run with:
//!
//! ```text
//! cargo test -p oxicuda-blas --release --test splitk_workspace_gpu -- --test-threads=1
//! ```
//!
//! # What changed, and what has to stay true
//!
//! `GemmDispatcher::dispatch_skinny_split_k` used to allocate its
//! `split_factor * m * n` reduction workspace with `cuMemAlloc` inside every
//! call and free it at the end. That had two consequences, one of them fatal
//! to a feature built on top of it:
//!
//! * **`cuMemAlloc` is forbidden during CUDA stream capture.** The driver
//!   rejects it with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, so *no* split-K
//!   GEMM could be recorded into a CUDA graph — and split-K is exactly the
//!   shape class (`m*n < 65536`, `k >= 512`) that repeated small-batch
//!   inference GEMMs fall into. `oxionnx-cuda`'s `graph_cache` depends on this
//!   fix; without it, its whole target workload was uncapturable.
//! * **The workspace pointer was different on every call.** A recorded graph
//!   stores addresses, so a per-call workspace would have baked in an address
//!   that was freed before the first replay.
//!
//! So the three properties below are asserted directly:
//!
//! 1. The numbers are unchanged — checked element-for-element against an
//!    `f64`-accumulated CPU oracle, because a workspace bug shows up as a
//!    correctly-shaped output with wrong values, never as an error.
//! 2. Repeating a shape reuses one workspace: the cache's held bytes stop
//!    growing after the first call, and a *different* shape adds to them.
//! 3. A split-K GEMM records into a real, driver-backed CUDA graph, and
//!    replaying that graph reproduces the same numbers bit for bit.
//!
//! Skips cleanly (rather than failing) with no CUDA driver / device, matching
//! this crate's `src/gpu_tests.rs` convention.

use std::sync::Arc;

use oxicuda_blas::handle::BlasHandle;
use oxicuda_blas::level3::gemm_api::gemm;
use oxicuda_blas::types::{Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_driver::ffi::{CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, CUdeviceptr};
use oxicuda_driver::graph::StreamGraphCapture;
use oxicuda_driver::{Context, Device, Stream};
use oxicuda_memory::DeviceBuffer;

/// Acquire a GPU handle, or `None` when no driver / device is present.
///
/// The handle owns a freshly created `CU_STREAM_NON_BLOCKING` stream, which
/// stream capture requires (the legacy default stream cannot be captured).
fn try_handle() -> Option<(Arc<Context>, BlasHandle)> {
    oxicuda_driver::init().ok()?;
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    ctx.set_current().ok()?;
    let stream = Stream::new(&ctx).ok()?;
    let handle = BlasHandle::with_stream(&ctx, stream).ok()?;
    Some((ctx, handle))
}

/// A small deterministic LCG, so a failure is reproducible.
fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = f64::from((state >> 32) as u32) / 4_294_967_296.0;
            (unit * 2.0 - 1.0) as f32
        })
        .collect()
}

/// Row-major `A[m,k] @ B[k,n]`, accumulated in `f64` so the oracle is
/// independent of the kernel's summation order.
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f64;
            for i in 0..k {
                acc += f64::from(a[row * k + i]) * f64::from(b[i * n + col]);
            }
            out[row * n + col] = acc as f32;
        }
    }
    out
}

/// Device-side operands for one split-K-shaped GEMM.
struct Case {
    m: usize,
    k: usize,
    n: usize,
    host_a: Vec<f32>,
    host_b: Vec<f32>,
    d_a: DeviceBuffer<f32>,
    d_b: DeviceBuffer<f32>,
    d_c: DeviceBuffer<f32>,
}

impl Case {
    /// `m*n` below 65536 with `k >= 512` is precisely the split-K rule in
    /// `GemmDispatcher::should_use_split_k_workspace`, so every case built
    /// here takes the path under test.
    fn new(m: usize, k: usize, n: usize, seed: u64) -> Option<Self> {
        assert!(
            m * n < 65_536 && k >= 512,
            "case {m}x{k}x{n} does not take the split-K path, so it would test nothing",
        );
        let host_a = pseudo_random(m * k, seed);
        let host_b = pseudo_random(k * n, seed ^ 0xFFFF);
        let mut d_a = DeviceBuffer::<f32>::alloc(m * k).ok()?;
        let mut d_b = DeviceBuffer::<f32>::alloc(k * n).ok()?;
        d_a.copy_from_host(&host_a).ok()?;
        d_b.copy_from_host(&host_b).ok()?;
        Some(Self {
            m,
            k,
            n,
            host_a,
            host_b,
            d_a,
            d_b,
            d_c: DeviceBuffer::<f32>::alloc(m * n).ok()?,
        })
    }

    /// Issue the GEMM on the handle's stream. Does not synchronise.
    fn issue(&mut self, handle: &BlasHandle) -> Result<(), Box<dyn std::error::Error>> {
        let desc_a = MatrixDesc::<f32>::from_buffer(
            &self.d_a,
            self.m as u32,
            self.k as u32,
            Layout::RowMajor,
        )?;
        let desc_b = MatrixDesc::<f32>::from_buffer(
            &self.d_b,
            self.k as u32,
            self.n as u32,
            Layout::RowMajor,
        )?;
        let mut desc_c = MatrixDescMut::<f32>::from_buffer(
            &mut self.d_c,
            self.m as u32,
            self.n as u32,
            Layout::RowMajor,
        )?;
        gemm(
            handle,
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0_f32,
            &desc_a,
            &desc_b,
            0.0_f32,
            &mut desc_c,
        )?;
        Ok(())
    }

    /// Issue, synchronise, and read the result back.
    fn run(&mut self, handle: &BlasHandle) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.issue(handle)?;
        handle.stream().synchronize()?;
        let mut out = vec![0.0_f32; self.m * self.n];
        self.d_c.copy_to_host(&mut out)?;
        Ok(out)
    }

    fn expected(&self) -> Vec<f32> {
        cpu_matmul(&self.host_a, &self.host_b, self.m, self.k, self.n)
    }
}

/// Assert two results agree to a relative tolerance appropriate for a
/// `k`-long f32 reduction against an f64 oracle.
fn assert_close(got: &[f32], want: &[f32], k: usize, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    // The kernel accumulates in f32 over `k` terms in a different order than
    // the oracle's f64 sum, so agreement is relative, not exact.
    let tol = 1e-4_f32 * (k as f32).sqrt();
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let diff = (g - w).abs();
        let scale = w.abs().max(1.0);
        assert!(
            diff / scale <= tol,
            "{what}: element {i} is {g} but the CPU oracle says {w} (|diff| {diff}, tol {tol})",
        );
    }
}

#[test]
fn a_repeated_split_k_shape_reuses_one_workspace_and_keeps_its_numbers() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("no CUDA device -- skipping");
        return;
    };
    assert_eq!(
        handle.split_k_workspace_bytes().expect("cache readable"),
        0,
        "a fresh dispatcher must hold no split-K workspace",
    );

    // InSwapper's AdaIN projection shape.
    let Some(mut case) = Case::new(1, 512, 2048, 7) else {
        eprintln!("device allocation failed -- skipping");
        return;
    };
    let want = case.expected();

    let first = case.run(&handle).expect("first split-K GEMM");
    assert_close(&first, &want, case.k, "first call");
    let after_first = handle
        .split_k_workspace_bytes()
        .expect("cache readable after the first call");
    assert!(
        after_first > 0,
        "the first split-K GEMM cached no workspace, so nothing is being reused",
    );

    // Repeating the same shape must not grow the cache — that is the whole
    // claim — and must not change the answer.
    for call in 1..=4 {
        let again = case.run(&handle).expect("repeated split-K GEMM");
        assert_close(&again, &want, case.k, &format!("repeat {call}"));
        assert_eq!(
            handle.split_k_workspace_bytes().expect("cache readable"),
            after_first,
            "repeat {call} allocated a second workspace for a shape already cached",
        );
    }

    // A *different* shape is a different key, so it does allocate — otherwise
    // the assertion above would also pass for a cache that never stores
    // anything after the first entry.
    let Some(mut other) = Case::new(1, 512, 1024, 11) else {
        eprintln!("device allocation failed -- skipping");
        return;
    };
    let other_want = other.expected();
    let other_got = other.run(&handle).expect("second shape");
    assert_close(&other_got, &other_want, other.k, "second shape");
    assert!(
        handle.split_k_workspace_bytes().expect("cache readable") > after_first,
        "a distinct split-K shape must get its own workspace",
    );
}

#[test]
fn a_split_k_gemm_records_into_a_driver_backed_graph_and_replays_exactly() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("no CUDA device -- skipping");
        return;
    };
    // ArcFace's embedding head — the most repeated split-K shape in the face
    // pipeline, and the one that could not be captured at all before the
    // workspace was cached.
    let Some(mut case) = Case::new(1, 25088, 512, 13) else {
        eprintln!("device allocation failed -- skipping");
        return;
    };
    let want = case.expected();

    // One ordinary call first: it populates the workspace cache (and the
    // compiled-kernel caches), so the capture below contains nothing but
    // kernel launches.
    let ordinary = case.run(&handle).expect("warm-up GEMM");
    assert_close(&ordinary, &want, case.k, "ordinary launch");

    // Poison the output so a replay that does nothing is visible.
    const SENTINEL: f32 = -98_765.5;
    case.d_c
        .copy_from_host(&vec![SENTINEL; case.m * case.n])
        .expect("sentinel");

    let capture = StreamGraphCapture::begin(handle.stream(), CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .expect("begin capture");
    let issued = case.issue(&handle);
    assert!(
        issued.is_ok(),
        "a split-K GEMM must be capturable now that its workspace is cached; got {issued:?}",
    );
    let exec = capture.end().expect("end capture");
    assert!(
        exec.is_driver_backed(),
        "the captured graph is not driver-backed, so replaying it would compute nothing",
    );
    assert!(
        exec.node_count() >= 2,
        "split-K is a partial pass plus a reduction pass, so the capture must hold at least two \
         nodes; got {}",
        exec.node_count(),
    );

    // Capture records; it must not execute.
    let mut mid = vec![0.0_f32; case.m * case.n];
    case.d_c.copy_to_host(&mut mid).expect("read back");
    assert!(
        mid.iter().all(|&v| (v - SENTINEL).abs() < f32::EPSILON),
        "capture executed the recorded GEMM instead of recording it",
    );

    exec.launch(handle.stream()).expect("replay");
    handle.stream().synchronize().expect("sync");
    let mut replayed = vec![0.0_f32; case.m * case.n];
    case.d_c.copy_to_host(&mut replayed).expect("read back");
    assert_close(&replayed, &want, case.k, "replayed graph");
    // Against the ordinary launch the recording was made from, exactness is
    // the right bar: it is the same kernels over the same addresses.
    assert_eq!(
        replayed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ordinary.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a replayed split-K graph differs from the ordinary launch it recorded",
    );
}

#[test]
fn a_cached_workspace_keeps_the_same_device_address_across_calls() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("no CUDA device -- skipping");
        return;
    };
    // Address stability is what makes a recorded graph replayable, and it is
    // not observable through the public API directly — so it is established
    // the way a graph would rely on it: record once, then run many ordinary
    // GEMMs of the same shape (each of which would have re-allocated the
    // workspace under the old behaviour, very likely at a different address),
    // then replay. A moved workspace shows up as a wrong replay.
    let Some(mut case) = Case::new(1, 512, 1024, 23) else {
        eprintln!("device allocation failed -- skipping");
        return;
    };
    let want = case.expected();
    let ordinary = case.run(&handle).expect("warm-up");

    let capture = StreamGraphCapture::begin(handle.stream(), CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .expect("begin capture");
    case.issue(&handle).expect("record");
    let exec = capture.end().expect("end capture");
    assert!(exec.is_driver_backed());

    // Churn: twenty more ordinary GEMMs of the same shape, plus allocations
    // and frees in between, all of which would land on top of a workspace that
    // had been released.
    for _ in 0..20 {
        let _ = case.run(&handle).expect("churn GEMM");
        let churn = DeviceBuffer::<f32>::alloc(1 << 16).expect("churn alloc");
        let churn_ptr: CUdeviceptr = churn.as_device_ptr();
        assert_ne!(churn_ptr, 0, "a churn allocation must be real");
        drop(churn);
    }

    exec.launch(handle.stream()).expect("replay after churn");
    handle.stream().synchronize().expect("sync");
    let mut replayed = vec![0.0_f32; case.m * case.n];
    case.d_c.copy_to_host(&mut replayed).expect("read back");
    assert_close(&replayed, &want, case.k, "replay after churn");
    assert_eq!(
        replayed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ordinary.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the replay diverged after unrelated allocation churn -- the split-K workspace the graph \
         recorded did not survive",
    );
}
