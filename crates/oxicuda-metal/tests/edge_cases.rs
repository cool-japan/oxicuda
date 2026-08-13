//! Edge-case coverage for `MetalBackend` that is not otherwise exercised by
//! `conformance_gemm.rs` / `conformance_ops.rs`: device-to-device copy
//! aliasing, oversized allocation, and an unusually large batched-GEMM
//! `batch_count`.
//!
//! Every test degrades to a no-op skip when no Metal device is present
//! (headless CI, non-macOS), matching the crate's existing `try_init`
//! convention (see `src/backend/gpu_tests.rs`).

use oxicuda_backend::{BackendError, BackendTranspose, ComputeBackend};
use oxicuda_metal::MetalBackend;

fn try_init() -> Option<MetalBackend> {
    let mut b = MetalBackend::new();
    b.init().ok().map(|()| b)
}

fn upload_f32(be: &MetalBackend, data: &[f32]) -> u64 {
    let ptr = be.alloc((data.len() * 4).max(4)).expect("alloc");
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    be.copy_htod(ptr, &bytes).expect("copy_htod");
    ptr
}
fn download_f32(be: &MetalBackend, ptr: u64, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * 4];
    be.copy_dtoh(&mut bytes, ptr).expect("copy_dtoh");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect()
}

// ─── copy_dtod aliasing ──────────────────────────────────────────────────────

/// `MetalBackend::copy_dtod` explicitly rejects `src == dst` with
/// `InvalidArgument` (see `MetalMemoryManager::copy_device_to_device`'s doc:
/// "requires distinct src and dst handles") rather than performing an
/// aliased copy whose correctness would depend on the underlying blit
/// encoder's overlap handling. A distinct, non-aliased `src`/`dst` pair must
/// still copy correctly.
#[test]
fn copy_dtod_rejects_aliasing_but_roundtrips_when_distinct() {
    let Some(backend) = try_init() else { return };

    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = upload_f32(&backend, &data);

    // Aliased copy: same handle for both src and dst — rejected.
    let aliased = backend.copy_dtod(a, a, data.len() * 4);
    assert!(
        matches!(aliased, Err(BackendError::InvalidArgument(_))),
        "copy_dtod(a, a, ..) must be rejected as aliasing, got {aliased:?}"
    );
    // The rejection must not have corrupted `a`.
    assert_eq!(download_f32(&backend, a, data.len()), data);

    // Distinct, non-aliased copy: must succeed and round-trip exactly.
    let b = backend.alloc(data.len() * 4).expect("alloc b");
    backend
        .copy_dtod(b, a, data.len() * 4)
        .expect("distinct copy_dtod must succeed");
    assert_eq!(download_f32(&backend, b, data.len()), data);

    // Source is unchanged by a device-to-device copy out of it.
    assert_eq!(download_f32(&backend, a, data.len()), data);
}

// ─── oversized alloc ─────────────────────────────────────────────────────────

/// An allocation request larger than the device's `max_buffer_length` is
/// rejected with `OutOfMemory` rather than attempting (and likely crashing
/// on) `newBufferWithLength:` — see `MetalMemoryManager::alloc`'s pre-flight
/// `bytes as u64 > max_len` guard.
#[test]
fn oversized_alloc_returns_out_of_memory() {
    let Some(backend) = try_init() else { return };

    // 1 TiB: no real Metal device's max_buffer_length reaches this, and it
    // is far below any risk of overflowing the `usize`/`u64` arithmetic in
    // the allocator's own size checks.
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

// ─── batched_gemm with an unusually large batch_count ────────────────────────

/// `WebGpuBackend::batched_gemm` documents (and, per its own in-crate test
/// `backend_tests_gpu_ops.rs`, enforces) a hard `batch_count <= 65_535`
/// rejection, inherited from the WebGPU workgroup-per-dimension dispatch
/// limit. `MetalBackend` has no such documented cap — `MTLSize`'s grid `z`
/// dimension is not bound to the WebGPU/D3D 65 535 convention — so this
/// checks what the Metal path *actually does* at `batch_count = 70_000`
/// rather than assuming it shares WebGPU's limit. `m = n = k = 1` keeps the
/// three buffers a few hundred KiB regardless of the outcome.
#[test]
fn batched_gemm_batch_count_70000() {
    let Some(backend) = try_init() else { return };

    let (m, n, k, batch_count) = (1usize, 1usize, 1usize, 70_000usize);
    // C[b] = A[b] * B[b] with every A[b] = 2.0, B[b] = 3.0 -> every C[b] = 6.0.
    let a: Vec<f32> = vec![2.0; batch_count];
    let b: Vec<f32> = vec![3.0; batch_count];
    let c: Vec<f32> = vec![0.0; batch_count];

    let a_ptr = upload_f32(&backend, &a);
    let b_ptr = upload_f32(&backend, &b);
    let c_ptr = upload_f32(&backend, &c);

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

    // Metal imposes no WebGPU-style 65 535 dispatch-grid cap on the batch
    // axis, so this succeeds and every batch computes the right answer —
    // the opposite contract from `oxicuda-webgpu`'s hard rejection at the
    // same batch_count (see that crate's `edge_cases.rs`). Asserted against
    // the value actually observed on real hardware, not inferred.
    result.unwrap_or_else(|e| panic!("batch_count=70_000 was expected to succeed on Metal (no WebGPU-style dispatch cap), got {e}"));
    let got = download_f32(&backend, c_ptr, batch_count);
    assert!(
        got.iter().all(|&v| (v - 6.0).abs() < 1e-4),
        "every one of the 70,000 batches must compute 2.0*3.0 = 6.0"
    );
}
