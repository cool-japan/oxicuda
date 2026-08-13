//! Edge-case coverage for `WebGpuBackend` that is not otherwise exercised by
//! `conformance_gemm.rs` / `conformance_ops.rs`: a non-multiple-of-4
//! allocation size, oversized allocation, and `batched_gemm`'s
//! `batch_count > 65_535` rejection.
//!
//! Every test degrades to a no-op skip when no wgpu adapter is present
//! (headless CI), matching the crate's existing `try_init` convention (see
//! `src/backend_tests.rs`).

use oxicuda_backend::{BackendError, BackendTranspose, ComputeBackend};
use oxicuda_webgpu::WebGpuBackend;

fn try_init() -> Option<WebGpuBackend> {
    let mut b = WebGpuBackend::new();
    b.init().ok().map(|()| b)
}

// ─── non-multiple-of-4 allocation ────────────────────────────────────────────

/// `WebGpuMemoryManager::alloc` rounds the requested size up to
/// `wgpu::COPY_BUFFER_ALIGNMENT` (4 bytes) rather than rejecting a
/// non-aligned request — the physical `wgpu::Buffer` is larger than asked
/// for, but every byte the caller actually requested must still round-trip
/// correctly through `copy_htod`/`copy_dtoh`, which is the property this
/// test checks (not just that `alloc` returns `Ok`).
#[test]
fn non_multiple_of_4_alloc_rounds_up_and_roundtrips() {
    let Some(backend) = try_init() else { return };

    // 13 bytes: not a multiple of 4, and not a multiple of 16 either, so a
    // hidden assumption of either alignment would be caught.
    const LEN: usize = 13;
    let ptr = backend
        .alloc(LEN)
        .expect("a non-multiple-of-4 alloc size must still succeed (rounds up internally)");

    let data: Vec<u8> = (0u8..LEN as u8)
        .map(|i| i.wrapping_mul(17).wrapping_add(1))
        .collect();
    backend
        .copy_htod(ptr, &data)
        .expect("copy_htod of exactly LEN bytes");

    let mut back = vec![0u8; LEN];
    backend
        .copy_dtoh(&mut back, ptr)
        .expect("copy_dtoh of exactly LEN bytes");
    assert_eq!(
        back, data,
        "the requested (unaligned) byte range must round-trip exactly"
    );

    backend.free(ptr).expect("free");
}

// ─── oversized alloc ─────────────────────────────────────────────────────────

/// An allocation request larger than the adapter's `max_buffer_size` /
/// `max_storage_buffer_binding_size` is rejected with `OutOfMemory` rather
/// than attempting (and failing inside) `create_buffer` — see
/// `WebGpuMemoryManager::alloc`'s pre-flight `size > limits...` guard.
#[test]
fn oversized_alloc_returns_out_of_memory() {
    let Some(backend) = try_init() else { return };

    // 1 TiB: no real wgpu adapter's max_buffer_size reaches this, and it is
    // far below any risk of overflowing the size-rounding arithmetic in the
    // allocator's own checks (`next_multiple_of` on a u64).
    const HUGE: usize = 1usize << 40;
    let result = backend.alloc(HUGE);
    assert_eq!(
        result,
        Err(BackendError::OutOfMemory),
        "a 1 TiB allocation must be rejected as OutOfMemory"
    );

    // The backend must still be usable afterwards — an OOM alloc must not
    // poison the allocator state.
    let ptr = backend
        .alloc(64)
        .expect("small alloc after OOM must still work");
    backend.free(ptr).expect("free after OOM must still work");
}

// ─── batched_gemm batch_count > 65_535 ───────────────────────────────────────

/// `WebGpuBackend::batched_gemm` dispatches the batch axis as the `z`
/// dimension of a `wgpu` workgroup grid, which is hard-capped at
/// `MAX_WORKGROUPS_PER_DIM = 65_535` (`planner.rs`) per the WebGPU spec's
/// `maxComputeWorkgroupsPerDimension` floor. `batch_count = 70_000` must
/// therefore be rejected with a clean, typed error *before* any GPU state is
/// created for a dispatch that could never legally run — not a wgpu
/// validation panic. Mirrors this crate's own in-tree coverage of the same
/// scenario (`backend_tests_gpu_ops.rs`) as an independent, external
/// (integration-level) confirmation, and contrasts with `MetalBackend`,
/// which has no such cap (see that crate's `edge_cases.rs`).
#[test]
fn batch_count_over_65535_is_rejected() {
    let Some(backend) = try_init() else { return };

    let (m, n, k, batch_count) = (1usize, 1usize, 1usize, 70_000usize);
    let a_ptr = backend.alloc(batch_count * 4).expect("alloc a");
    let b_ptr = backend.alloc(batch_count * 4).expect("alloc b");
    let c_ptr = backend.alloc(batch_count * 4).expect("alloc c");

    let result = backend.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        m,
        n,
        k,
        1.0,
        a_ptr,
        k,
        m * k,
        b_ptr,
        n,
        k * n,
        0.0,
        c_ptr,
        n,
        m * n,
        batch_count,
    );
    assert!(
        matches!(result, Err(BackendError::InvalidArgument(_))),
        "batch_count=70_000 must be rejected with InvalidArgument (exceeds the \
         65_535 per-dimension workgroup cap), got {result:?}"
    );
}
