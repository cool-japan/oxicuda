//! Regression tests for GPU-side dispatch ordering with **zero host-side
//! synchronisation** between ops.
//!
//! Split into a sibling file (mirroring `backend_tests_gpu_ops.rs`) to keep
//! `backend_tests.rs` under the 2 000-line refactoring policy.
//!
//! These tests exist because two related changes removed every per-op
//! `device.poll(...)` call from the compute-dispatch methods (`gemm`,
//! `unary`, `binary`, `reduce`, `reduce_nd`, `batched_gemm`, `gemm_f16`,
//! `conv2d_forward`, `attention`) and introduced a bind-group + dedicated-
//! uniform-buffer cache keyed on the operand handles, so a call with the same
//! operand handles as a previous call reuses (and `write_buffer`s fresh
//! parameters into) the *same* underlying `wgpu::Buffer` and `wgpu::BindGroup`
//! objects instead of allocating new ones:
//!
//! * Removing the per-op poll relies on `wgpu::Queue` executing everything
//!   submitted to it — both `submit()` (compute dispatches) and
//!   `write_buffer()` (uniform parameter uploads) — in the exact order the
//!   calls were issued from the host ("queue timeline" ordering, part of the
//!   WebGPU/wgpu contract). A later dispatch reading an earlier dispatch's
//!   output, or a host readback via `copy_from_device`/`synchronize()`, is
//!   correctly ordered without any host-side wait in between.
//! * Reusing one uniform buffer across many calls goes further: it requires
//!   that dispatch *N*'s compute pass observes the `write_buffer` call that
//!   happened immediately before *its own* `submit()` — not a later call's
//!   `write_buffer` racing ahead of it. This is the same queue-timeline
//!   guarantee, but it is worth its own test because a *caching* bug (e.g.
//!   silently reusing a stale bind group that still points at an orphaned
//!   uniform buffer, or writing the new parameters to the wrong buffer)
//!   would surface here specifically, not in the fresh-buffer-per-call tests
//!   above it in `backend_tests.rs`.
//!
//! Roughly, the tests below fall into four groups:
//!
//! 1. **Ordering under zero host sync** (`chained_unary_ops_...`,
//!    `chained_gemm_reusing_same_handles_...`): issue many dispatches with
//!    **no** `synchronize()`/`copy_dtoh` in the loop, then perform exactly
//!    one final readback — if either queue-timeline guarantee above is
//!    violated, the final value silently reflects a stale or out-of-order
//!    intermediate result instead of the last op actually issued.
//! 2. **Cache/handle-lifetime interaction** (`bind_group_cache_entry_is_evicted_on_free`,
//!    `bind_group_cache_order_does_not_grow_unbounded_...`): a `wgpu::BindGroup`
//!    retains a strong reference to every buffer it binds, so
//!    `WebGpuBackend::free` must evict any cache entry mentioning a freed
//!    handle both from the live entry map *and* from the cache's
//!    insertion-order tracking — otherwise either that handle's GPU memory
//!    (map) or plain host memory (order tracking) is kept alive
//!    indefinitely instead of being bounded by the cache's small cap.
//! 3. **`synchronize()`'s hardening** (`synchronize_surfaces_recorded_uncaptured_error`,
//!    `synchronize_is_ok_with_nothing_pending`): with no per-op poll left
//!    anywhere, `synchronize()` is the *only* place a caller who skips
//!    `copy_dtoh` learns that submitted work actually failed, so its
//!    `let _ = dev.device.poll(...)` was hardened to propagate a real error
//!    instead of discarding it — and must still report `Ok(())` when
//!    nothing actually failed.

use super::*;

/// Eight `Neg` unary dispatches chained output-to-input, with **no** host
/// sync between any of them, followed by exactly one `copy_dtoh`.
///
/// Regression test for `perf-9` ("wgpu blocks on
/// `device.poll(wait_indefinitely())` after EVERY submit") — its proposed
/// fix explicitly calls for "a chained-op regression test (8 unary ops then
/// `copy_dtoh`) to prove ordering holds" once the per-op polls are removed.
///
/// Eight negations is an even count, so the expected output is exactly the
/// original input — a double negation *and* a reordering/dropped-dispatch
/// bug would both generically fail this equality, unlike an idempotent op
/// (e.g. `Relu`) which would mask a reordering.
#[test]
fn chained_unary_ops_with_no_intermediate_sync_stay_ordered() {
    let Some(b) = try_init() else { return };

    let input = [-3.5f32, 0.0, 2.25, -100.0, 7.0, -0.001, 42.0, -8.0];
    let n = input.len();

    // 9 buffers: buf[0] holds the original input, buf[8] receives the 8th
    // (final) negation. buf[i] = neg(buf[i-1]). The tail 8 start zeroed so a
    // skipped dispatch is visible as "still zero" rather than coincidentally
    // correct leftover memory.
    let mut bufs: Vec<u64> = vec![upload_f32(&b, &input)];
    bufs.extend((0..8).map(|_| upload_f32(&b, &vec![0.0f32; n])));

    for i in 0..8 {
        b.unary(UnaryOp::Neg, bufs[i], bufs[i + 1], n)
            .unwrap_or_else(|e| panic!("unary Neg step {i} failed: {e:?}"));
        // Deliberately no synchronize()/copy_dtoh here.
    }

    let result = download_f32(&b, bufs[8], n);
    for (r, e) in result.iter().zip(input.iter()) {
        assert!(
            (r - e).abs() < 1e-5,
            "chained 8x-negate result {result:?} does not match original input {input:?} \
             (got {r}, expected {e}) — dispatch ordering broke without a per-op poll"
        );
    }

    for h in bufs {
        b.free(h).expect("free");
    }
}

/// Thirty `gemm` dispatches reusing the *same* `(a_ptr, b_ptr, c_ptr)`
/// handle triple with a distinct `alpha` each time, no host sync between any
/// of them, followed by exactly one final readback.
///
/// This is the scenario the bind-group + dedicated-uniform-buffer cache
/// (`WebGpuBackend`'s `bind_group_cache`) exists for — every call after the
/// first reuses the same cached `wgpu::BindGroup` and the same small
/// `wgpu::Buffer` for `GemmParams`, `write_buffer`-ing fresh bytes into it
/// each time. The test passes trivially before that cache exists (every call
/// then allocates its own fresh uniform buffer and bind group, so there is
/// nothing to race), and becomes a real check of the cache's correctness
/// once it is wired in: only the *last* alpha's result can survive if
/// dispatch *N* ever failed to observe dispatch *N*'s own `write_buffer`.
#[test]
fn chained_gemm_reusing_same_handles_observes_latest_uniform_each_call() {
    let Some(b) = try_init() else { return };

    // A (2x3) * B (3x2) = fixed 2x2 product; scale it by a different alpha
    // each iteration and overwrite C in place, beta = 0 so C never
    // accumulates across iterations.
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bm = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let base = [58.0f32, 64.0, 139.0, 154.0]; // A*B with alpha=1, beta=0

    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &bm);
    let c_h = upload_f32(&b, &[0.0f32; 4]);

    let nt = BackendTranspose::NoTrans;
    const ITERS: usize = 30;
    let mut last_alpha = 0.0f64;
    for i in 0..ITERS {
        // Widely spaced, easily distinguished values so an out-of-order
        // final result is very unlikely to coincidentally match.
        let alpha = (i as f64) * 1000.0 + 7.0;
        last_alpha = alpha;
        b.gemm(nt, nt, 2, 2, 3, alpha, a_h, 3, b_h, 2, 0.0, c_h, 2)
            .unwrap_or_else(|e| panic!("gemm iteration {i} (alpha={alpha}) failed: {e:?}"));
        // Deliberately no synchronize()/copy_dtoh here.
    }

    let result = download_f32(&b, c_h, 4);
    for (r, base_v) in result.iter().zip(base.iter()) {
        let expected = base_v * (last_alpha as f32);
        // f32 accumulation over a k=3 dot product at alpha ~= 29007: a
        // relative tolerance is appropriate at this magnitude.
        let tol = (expected.abs() * 1e-5).max(0.5);
        assert!(
            (r - expected).abs() < tol,
            "reused-handle gemm chain result {result:?} does not reflect the last \
             alpha={last_alpha} (expected ~{expected}, got {r}) — a stale cached uniform \
             buffer or bind group would produce exactly this symptom"
        );
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

/// Freeing a handle that was used in a cached `gemm` dispatch must evict
/// exactly that bind-group cache entry — and only that one, leaving an
/// unrelated cached entry (from an unrelated set of operand handles) intact.
///
/// A cached `wgpu::BindGroup` retains a strong reference to every buffer it
/// binds (the same guarantee this crate already relies on for an in-flight
/// dispatch surviving a concurrent `free()` on an unrelated handle), so an
/// entry left in the cache after `free()` would keep that "freed" handle's
/// GPU memory alive indefinitely. See `cache.rs`'s module doc, "Freed-buffer
/// memory".
#[test]
fn bind_group_cache_entry_is_evicted_on_free() {
    let Some(b) = try_init() else { return };
    let nt = BackendTranspose::NoTrans;

    // Entry 1: a1/b1/c1.
    let a1 = upload_f32(&b, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b1 = upload_f32(&b, &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let c1 = upload_f32(&b, &[0.0f32; 4]);
    b.gemm(nt, nt, 2, 2, 3, 1.0, a1, 3, b1, 2, 0.0, c1, 2)
        .expect("gemm entry 1");

    // Entry 2: a2/b2/c2, a completely unrelated set of handles.
    let a2 = upload_f32(&b, &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let b2 = upload_f32(&b, &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let c2 = upload_f32(&b, &[0.0f32; 4]);
    b.gemm(nt, nt, 2, 2, 3, 1.0, a2, 3, b2, 2, 0.0, c2, 2)
        .expect("gemm entry 2");

    let entries_after_two_gemms = b
        .bind_group_cache
        .lock()
        .expect("bind_group_cache lock")
        .len();
    assert_eq!(
        entries_after_two_gemms, 2,
        "two gemm calls with disjoint operand handles should populate two distinct cache entries"
    );

    // Freeing any one handle from entry 1 must evict entry 1 specifically.
    b.free(c1).expect("free c1");

    let entries_after_free = b
        .bind_group_cache
        .lock()
        .expect("bind_group_cache lock")
        .len();
    assert_eq!(
        entries_after_free, 1,
        "freeing a handle referenced by one cached bind group must evict only that entry"
    );

    // Entry 2 must still be usable: a fresh gemm with a2/b2/c2 (the exact
    // handles the surviving cache entry points at) must still succeed and
    // still be correct — proving the survivor was not corrupted by the
    // neighbouring eviction.
    b.gemm(nt, nt, 2, 2, 3, 1.0, a2, 3, b2, 2, 0.0, c2, 2)
        .expect("gemm reusing entry 2's cached bind group after an unrelated eviction");
    let result = download_f32(&b, c2, 4);
    // A2=[[7,8,9],[10,11,12]] (2x3) times B2=[[1,0],[0,1],[1,1]] (3x2):
    // C[0]=[7+0+9, 0+8+9]=[16,17]; C[1]=[10+0+12, 0+11+12]=[22,23].
    let expected = [16.0f32, 17.0, 22.0, 23.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-3, "got {r}, expected {e}");
    }

    b.free(a1).expect("free a1");
    b.free(b1).expect("free b1");
    b.free(a2).expect("free a2");
    b.free(b2).expect("free b2");
    b.free(c2).expect("free c2");
}

/// `synchronize()` must surface a recorded uncaptured wgpu error instead of
/// silently discarding it.
///
/// Before the per-op polls were removed, a caller could rely on the very
/// next compute op (or `copy_dtoh`) to poll again and notice a problem;
/// `synchronize()` itself was `let _ = dev.device.poll(...); Ok(())` — an
/// unconditional `Ok`. Now that no compute op polls at all, `synchronize()`
/// is the only wait a caller who never calls `copy_dtoh` gets, so it must
/// both wait for real and propagate what it observes. This test triggers a
/// real uncaptured error directly against the raw device (the same
/// technique `device.rs`'s and `memory.rs`'s own tests use to prove the
/// non-fatal handler works), bypassing every higher-level guard in
/// `WebGpuBackend`/`WebGpuMemoryManager`, so it exercises `synchronize()`'s
/// own error path specifically.
#[test]
fn synchronize_surfaces_recorded_uncaptured_error() {
    let Some(b) = try_init() else { return };

    // `WebGpuBackend::device()` is a private accessor, but this test lives
    // in a descendant module of `backend` and can see it directly.
    let dev = b.device().expect("device() after successful init");
    assert!(
        dev.poll_error().is_none(),
        "precondition: no error recorded yet"
    );

    let _bogus = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oxicuda-webgpu-test-oversize"),
        size: u64::MAX,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let err = b
        .synchronize()
        .expect_err("synchronize() should surface the recorded uncaptured error, not Ok(())");
    assert!(
        matches!(err, BackendError::DeviceError(_)),
        "got {err:?}, expected a DeviceError wrapping the uncaptured wgpu error"
    );
}

/// A `WebGpuBackend` with no recorded error and no pending failure must
/// still report `Ok(())` from `synchronize()` — the hardening must not turn
/// the common (nothing went wrong) case into a false-positive error.
///
/// Deliberately uses *separate* input/output buffers: `unary`/`binary`'s
/// bind group declares `input`/`lhs`/`rhs` as `read` and `output` as
/// `read_write` at different bindings, and wgpu's usage-scope validation
/// rejects the same buffer being bound as both within one dispatch — an
/// aliased (`input_ptr == output_ptr`) call is not a pattern any test in
/// this crate exercises or the `ComputeBackend` trait doc promises, so this
/// test (like every other one in the suite) sticks to the supported,
/// non-aliased shape.
#[test]
fn synchronize_is_ok_with_nothing_pending() {
    let Some(b) = try_init() else { return };
    let a_h = upload_f32(&b, &[1.0f32, 2.0, 3.0, 4.0]);
    let out_h = upload_f32(&b, &[0.0f32; 4]);
    b.unary(UnaryOp::Relu, a_h, out_h, 4)
        .expect("unary before synchronize");
    b.synchronize()
        .expect("synchronize() must be Ok when nothing failed");
    b.free(a_h).expect("free");
    b.free(out_h).expect("free");
}

/// A workload that allocates fresh handles, dispatches once, and frees them
/// every iteration must not grow `BindGroupCache`'s insertion-order tracking
/// (`order`) without bound.
///
/// `evict_handle` removes dead keys from `entries` (the map itself), but
/// `order` (a `VecDeque<BindGroupKey>`, each owning a `String` + `Vec<u64>`)
/// was only ever pruned lazily from the *front* during `insert`'s own
/// eviction path — a key that is used once and freed before the cache ever
/// filled up left a permanent dead entry in `order`. This test's every
/// iteration frees all three handles a cached entry references, so by the
/// end `entries` and `order` should both be empty, not `order.len() ==
/// ITERS`.
#[test]
fn bind_group_cache_order_does_not_grow_unbounded_across_alloc_dispatch_free() {
    let Some(b) = try_init() else { return };
    let nt = BackendTranspose::NoTrans;
    const ITERS: usize = 200;

    for _ in 0..ITERS {
        let a_h = upload_f32(&b, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b_h = upload_f32(&b, &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let c_h = upload_f32(&b, &[0.0f32; 4]);
        b.gemm(nt, nt, 2, 2, 3, 1.0, a_h, 3, b_h, 2, 0.0, c_h, 2)
            .expect("gemm");
        b.free(a_h).expect("free a");
        b.free(b_h).expect("free b");
        b.free(c_h).expect("free c");
    }

    let cache = b.bind_group_cache.lock().expect("bind_group_cache lock");
    assert_eq!(
        cache.len(),
        0,
        "every entry's handles were freed every iteration; none should remain live"
    );
    assert_eq!(
        cache.order_len(),
        0,
        "order should track exactly the live entries (0), not accumulate one \
         dead key per iteration ({ITERS} would indicate the pre-fix unbounded-growth bug)"
    );
}
