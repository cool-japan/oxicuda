//! On-device tests for [`DnnHandle`]'s reusable pinned staging buffer.
//!
//! The handle now owns one page-locked host allocation
//! ([`oxicuda_memory::StagingBuffer`]) that backs its host↔device transfers, so
//! a pipeline running the same models once per frame pins host memory once
//! instead of per call, and moves tensors by DMA out of page-locked memory
//! rather than through the driver's pageable bounce buffer.
//!
//! Two properties matter and are both asserted here:
//!
//! * **Correctness.** The staged path must round-trip data exactly, and — the
//!   sharper case — a staged readback must be correctly ordered against work
//!   dispatched on the handle's *other* stream (the BLAS sub-handle's). That
//!   ordering is established device-side with an event rather than by blocking
//!   the host; [`staged_readback_sees_a_gemm_dispatched_through_blas`] is the
//!   proof it actually holds, and would fail by returning zeros if it did not.
//!
//! * **Reuse.** The point of the buffer is that it is allocated once. A
//!   `StagingBuffer` that silently re-pinned per call would still be correct and
//!   would still pass every data test while being slower than what it replaced,
//!   so reuse is asserted explicitly via the allocation counter.
//!
//! Every test skips cleanly when no CUDA device is present.

#![cfg(feature = "gpu-tests")]

use std::sync::Arc;

use oxicuda_blas::level3::gemm;
use oxicuda_blas::types::{Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_dnn::handle::DnnHandle;
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

/// A live DNN handle, or `None` when no CUDA driver/device is available.
fn try_handle() -> Option<DnnHandle> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let dev = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&dev).ok()?);
    DnnHandle::new(&ctx).ok()
}

/// Adding a [`StagingBuffer`](oxicuda_memory::StagingBuffer) and a reusable
/// [`Event`](oxicuda_driver::Event) to the handle must not cost it its
/// documented `Send`-but-not-`Sync` threading contract — a pipeline that moves
/// a handle onto a decoder thread depends on it, and losing it would be a
/// compile error far away from here.
#[test]
fn handle_is_still_send() {
    fn assert_send<T: Send>() {}
    assert_send::<DnnHandle>();
}

macro_rules! handle_or_skip {
    () => {
        match try_handle() {
            Some(h) => h,
            None => {
                eprintln!("skipping: no CUDA driver/device");
                return;
            }
        }
    };
}

/// The fill-in-place upload and read-in-place download must round-trip exactly.
///
/// This is the fast path (no pageable source slice, no copy out), so it is the
/// one whose correctness most needs pinning down.
#[test]
fn staged_round_trip_is_exact() {
    let mut handle = handle_or_skip!();
    // InSwapper's 128x128x3 output tensor -- a real shape from the target
    // workload, and below the auto-stage threshold so it exercises staging.
    let n = 128 * 128 * 3;
    let mut d = DeviceBuffer::<f32>::alloc(n).expect("alloc");

    handle
        .upload_staged_with(&mut d, n, |dst: &mut [f32]| {
            for (i, v) in dst.iter_mut().enumerate() {
                *v = (i % 1013) as f32 * 0.125 - 3.0;
            }
        })
        .expect("upload_staged_with");

    let got = handle
        .download_staged_into(&d, n)
        .expect("download_staged_into");
    assert_eq!(got.len(), n);
    for (i, &v) in got.iter().enumerate() {
        let want = (i % 1013) as f32 * 0.125 - 3.0;
        assert_eq!(v, want, "element {i}: staged round trip corrupted data");
    }
}

/// The slice-taking wrappers must round-trip on both sides of the auto-stage
/// threshold — the size-based path switch must not change observable behaviour.
#[test]
fn slice_wrappers_round_trip_above_and_below_threshold() {
    let mut handle = handle_or_skip!();

    // Below the 512 KiB default threshold (staged), then well above it (direct).
    for n in [64 * 1024usize, 1024 * 1024usize] {
        let mut d = DeviceBuffer::<f32>::alloc(n).expect("alloc");
        let src: Vec<f32> = (0..n).map(|i| (i % 617) as f32 * 0.5).collect();
        let mut dst = vec![0.0f32; n];

        handle.upload_staged(&mut d, &src).expect("upload_staged");
        handle
            .download_staged(&mut dst, &d)
            .expect("download_staged");
        assert_eq!(dst, src, "round trip failed for n = {n}");
    }

    let stats = handle.staging_stats();
    assert!(stats.staged_transfers > 0, "small transfers should stage");
    assert!(
        stats.direct_transfers > 0,
        "transfers past the threshold should bypass staging"
    );
}

/// The staging buffer must be pinned **once** and reused for the rest of the
/// run — that is the entire reason it lives on the handle.
///
/// A per-call `cuMemAllocHost_v2` would be slower than the pageable path it
/// replaces while passing every data-correctness test, so this is asserted
/// directly rather than inferred.
#[test]
fn hot_loop_pins_host_memory_exactly_once() {
    let mut handle = handle_or_skip!();
    let n = 112 * 112 * 3; // ArcFace input.
    let mut d = DeviceBuffer::<f32>::alloc(n).expect("alloc");
    let mut out = vec![0.0f32; n];

    // 200 frames' worth of the same shape, as the video pipeline would run.
    for frame in 0..200u32 {
        handle
            .upload_staged_with(&mut d, n, |dst: &mut [f32]| {
                dst.fill(frame as f32);
            })
            .expect("upload");
        handle.download_staged(&mut out, &d).expect("download");
        assert_eq!(out[0], frame as f32);
        assert_eq!(out[n - 1], frame as f32);
    }

    let stats = handle.staging_stats();
    assert_eq!(
        stats.allocations, 1,
        "handle re-pinned host memory {} times across a steady-state loop; the \
         staging buffer must be allocated once and reused",
        stats.allocations
    );
    assert_eq!(stats.staged_transfers, 400);
    assert_eq!(stats.direct_transfers, 0);
}

/// `reserve_staging` must pre-pin so that the steady-state loop allocates
/// nothing at all.
#[test]
fn reserve_staging_keeps_allocation_out_of_the_loop() {
    let mut handle = handle_or_skip!();
    let n = 640 * 640 * 3; // SCRFD input -- the largest tensor in the pipeline.
    handle
        .reserve_staging(n * std::mem::size_of::<f32>())
        .expect("reserve_staging");
    assert_eq!(handle.staging_stats().allocations, 1);

    let mut d = DeviceBuffer::<f32>::alloc(n).expect("alloc");
    for _ in 0..8 {
        handle
            .upload_staged_with(&mut d, n, |dst: &mut [f32]| dst.fill(2.5))
            .expect("upload");
    }
    assert_eq!(
        handle.staging_stats().allocations,
        1,
        "a pre-sized staging buffer must never re-pin"
    );

    // And a smaller shape reuses the same, larger allocation.
    let small = 112 * 112 * 3;
    let mut d_small = DeviceBuffer::<f32>::alloc(small).expect("alloc small");
    handle
        .upload_staged_with(&mut d_small, small, |dst: &mut [f32]| dst.fill(1.0))
        .expect("upload small");
    assert_eq!(handle.staging_stats().allocations, 1);
}

/// **The ordering guarantee.** A staged readback must observe a GEMM dispatched
/// through [`DnnHandle::blas`], with no `synchronize_all()` in between.
///
/// The BLAS sub-handle runs on its own `CU_STREAM_NON_BLOCKING` stream,
/// deliberately unordered against the handle's launch stream. A readback
/// enqueued on the launch stream would therefore be free to run before the GEMM
/// and return the output buffer's prior contents — zeros, which look like a
/// plausible result rather than an error. `upload_staged*` / `download_staged*`
/// close that by recording an event on the BLAS stream and having the launch
/// stream wait on it (a device-side dependency, so the host never blocks).
///
/// Without that join this test fails: `got` comes back all zeros while the CPU
/// oracle is non-zero.
#[test]
fn staged_readback_sees_a_gemm_dispatched_through_blas() {
    let mut handle = handle_or_skip!();

    // Large enough that the GEMM takes real device time, so a readback that
    // raced ahead would genuinely observe an unfinished buffer rather than
    // passing by luck.
    let (m, k, n) = (512usize, 512usize, 512usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) / 64.0)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i % 89) as f32 - 44.0) / 64.0)
        .collect();

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    let mut d_c = DeviceBuffer::<f32>::zeroed(m * n).expect("d_c");

    let a_desc = MatrixDesc::from_buffer(&d_a, m as u32, k as u32, Layout::RowMajor).expect("a");
    let b_desc = MatrixDesc::from_buffer(&d_b, k as u32, n as u32, Layout::RowMajor).expect("b");
    {
        let mut c_desc =
            MatrixDescMut::from_buffer(&mut d_c, m as u32, n as u32, Layout::RowMajor).expect("c");
        gemm::<f32>(
            handle.blas(),
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0f32,
            &a_desc,
            &b_desc,
            0.0f32,
            &mut c_desc,
        )
        .expect("gemm dispatch");
    }

    // Deliberately NO synchronize_all() here: the staged download's own
    // cross-stream join is the only thing making this correct.
    let got = handle
        .download_staged_into(&d_c, m * n)
        .expect("download_staged_into")
        .to_vec();

    assert!(
        got.iter().any(|&v| v != 0.0),
        "staged readback returned an all-zero matrix -- it ran before the GEMM \
         on the BLAS stream had written anything"
    );

    // Spot-check a spread of rows against an f64 CPU oracle rather than
    // recomputing all 262k elements.
    for &row in &[0usize, 1, m / 3, m / 2, m - 1] {
        for &col in &[0usize, 7, n / 2, n - 1] {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += f64::from(a[row * k + p]) * f64::from(b[p * n + col]);
            }
            let want = acc as f32;
            let have = got[row * n + col];
            assert!(
                (have - want).abs() <= 1e-3 * want.abs().max(1.0),
                "({row},{col}): staged readback gave {have}, oracle {want}"
            );
        }
    }
}

/// Invalid extents must be rejected rather than silently transferring a
/// truncated tensor.
#[test]
fn mismatched_lengths_are_rejected() {
    let mut handle = handle_or_skip!();
    let mut d = DeviceBuffer::<f32>::alloc(256).expect("alloc");

    assert!(handle.upload_staged(&mut d, &vec![0.0f32; 128]).is_err());
    assert!(handle.download_staged(&mut vec![0.0f32; 128], &d).is_err());
    assert!(
        handle
            .upload_staged_with(&mut d, 128, |_: &mut [f32]| {})
            .is_err()
    );
    assert!(handle.download_staged_into(&d, 128).is_err());
}
