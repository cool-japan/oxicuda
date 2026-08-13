//! Tests for `WebGpuBackend`.
//!
//! Extracted from `backend.rs` so the production module stays under the
//! 2 000-line refactoring policy.  Included into `backend.rs` via
//! `#[path = "backend_tests.rs"] mod tests;` so all tests still live in the
//! `backend::tests` namespace.

use super::*;
use oxicuda_backend::{BackendTranspose, BinaryOp, ReduceOp, UnaryOp};

// ── Construction ──────────────────────────────────────────────────────────

#[test]
fn webgpu_backend_new_uninitialized() {
    let b = WebGpuBackend::new();
    assert!(!b.is_initialized());
}

#[test]
fn webgpu_backend_name() {
    let b = WebGpuBackend::new();
    assert_eq!(b.name(), "webgpu");
}

#[test]
fn webgpu_backend_default() {
    let b = WebGpuBackend::default();
    assert!(!b.is_initialized());
    assert_eq!(b.name(), "webgpu");
}

#[test]
fn backend_debug_impl() {
    let b = WebGpuBackend::new();
    let s = format!("{b:?}");
    assert!(s.contains("WebGpuBackend"));
}

// ── Object-safety smoke test ──────────────────────────────────────────────

#[test]
fn backend_object_safe() {
    let b: Box<dyn ComputeBackend> = Box::new(WebGpuBackend::new());
    assert_eq!(b.name(), "webgpu");
}

// ── Not-initialized guards ────────────────────────────────────────────────

#[test]
fn backend_not_initialized_gemm() {
    let b = WebGpuBackend::new();
    let result = b.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        4,
        4,
        4,
        1.0,
        0,
        4,
        0,
        4,
        0.0,
        0,
        4,
    );
    assert_eq!(result, Err(BackendError::NotInitialized));
}

#[test]
fn backend_not_initialized_alloc() {
    let b = WebGpuBackend::new();
    let result = b.alloc(1024);
    assert_eq!(result, Err(BackendError::NotInitialized));
}

#[test]
fn backend_not_initialized_synchronize() {
    let b = WebGpuBackend::new();
    assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
}

#[test]
fn backend_not_initialized_free() {
    let b = WebGpuBackend::new();
    assert_eq!(b.free(1), Err(BackendError::NotInitialized));
}

#[test]
fn backend_not_initialized_copy_htod() {
    let b = WebGpuBackend::new();
    assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));
}

#[test]
fn backend_not_initialized_copy_dtoh() {
    let b = WebGpuBackend::new();
    let mut buf = [0u8; 4];
    assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
}

// ── Zero-size / trivial-OK paths (no GPU needed) ─────────────────────────

/// These tests exercise the "no-op for zero size" branches.  We need the
/// backend to be initialised, but if no GPU is available we skip.
///
/// Skipping keeps the suite green on headless CI, but it also means a device
/// that silently stops initialising turns every GPU test into a vacuous pass.
/// Set `OXICUDA_REQUIRE_GPU=1` to make that condition a failure instead — see
/// `gpu_device_is_live_when_required`.
fn try_init() -> Option<WebGpuBackend> {
    let mut b = WebGpuBackend::new();
    match b.init() {
        Ok(()) => Some(b),
        Err(_) => {
            assert!(
                !require_gpu(),
                "OXICUDA_REQUIRE_GPU=1 but no WebGPU adapter initialised"
            );
            None
        }
    }
}

/// Whether the caller demands that the GPU tests really run on a device.
fn require_gpu() -> bool {
    std::env::var("OXICUDA_REQUIRE_GPU").is_ok_and(|v| v == "1")
}

#[test]
fn gemm_zero_size_after_init() {
    let Some(b) = try_init() else {
        return;
    };
    let result = b.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        0,
        0,
        0,
        1.0,
        0,
        1,
        0,
        1,
        0.0,
        0,
        1,
    );
    assert_eq!(result, Ok(()));
}

#[test]
fn unary_zero_elements_after_init() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(b.unary(UnaryOp::Relu, 0, 0, 0), Ok(()));
}

#[test]
fn binary_zero_elements_after_init() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(b.binary(BinaryOp::Add, 0, 0, 0, 0), Ok(()));
}

#[test]
fn copy_htod_empty_noop() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(b.copy_htod(0, &[]), Ok(()));
}

#[test]
fn copy_dtoh_empty_noop() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(b.copy_dtoh(&mut [], 0), Ok(()));
}

#[test]
fn alloc_zero_bytes_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.alloc(0),
        Err(BackendError::InvalidArgument(
            "cannot allocate 0 bytes".into()
        ))
    );
}

#[test]
fn synchronize_after_init() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(b.synchronize(), Ok(()));
}

// ── Argument validation (post-init) ───────────────────────────────────────

#[test]
fn reduce_empty_shape_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.reduce(ReduceOp::Sum, 0, 0, &[], 0),
        Err(BackendError::InvalidArgument(
            "shape must not be empty".into()
        ))
    );
}

#[test]
fn reduce_axis_out_of_bounds_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.reduce(ReduceOp::Sum, 0, 0, &[4, 4], 5),
        Err(BackendError::InvalidArgument(
            "axis 5 is out of bounds for shape of length 2".into()
        ))
    );
}

#[test]
fn attention_zero_seq_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.attention(0, 0, 0, 0, 1, 1, 0, 8, 64, 0.125, false),
        Err(BackendError::InvalidArgument(
            "seq_q, seq_kv, and head_dim must all be > 0".into()
        ))
    );
}

#[test]
fn attention_nonpositive_scale_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, 0.0, false),
        Err(BackendError::InvalidArgument(
            "scale must be a positive finite number, got 0".into()
        ))
    );
    assert_eq!(
        b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, -1.0, false),
        Err(BackendError::InvalidArgument(
            "scale must be a positive finite number, got -1".into()
        ))
    );
    assert!(
        b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, f64::INFINITY, false)
            .is_err()
    );
}

#[test]
fn conv2d_wrong_input_shape_error() {
    let Some(b) = try_init() else {
        return;
    };
    // 3-element input_shape — should fail.
    assert_eq!(
        b.conv2d_forward(
            0,
            &[1, 3, 32],
            0,
            &[16, 3, 3, 3],
            0,
            &[1, 16, 30, 30],
            &[1, 1],
            &[0, 0]
        ),
        Err(BackendError::InvalidArgument(
            "input_shape must have 4 elements (NCHW)".into()
        ))
    );
}

#[test]
fn conv2d_wrong_filter_shape_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.conv2d_forward(
            0,
            &[1, 3, 32, 32],
            0,
            &[16, 3, 3],
            0,
            &[1, 16, 30, 30],
            &[1, 1],
            &[0, 0]
        ),
        Err(BackendError::InvalidArgument(
            "filter_shape must have 4 elements (KCFHFW)".into()
        ))
    );
}

#[test]
fn conv2d_wrong_stride_shape_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.conv2d_forward(
            0,
            &[1, 3, 32, 32],
            0,
            &[16, 3, 3, 3],
            0,
            &[1, 16, 30, 30],
            &[1], // <-- wrong
            &[0, 0],
        ),
        Err(BackendError::InvalidArgument(
            "stride must have 2 elements [sh, sw]".into()
        ))
    );
}

// ── Init is idempotent ────────────────────────────────────────────────────

#[test]
fn init_idempotent() {
    let Some(mut b) = try_init() else {
        return;
    };
    // Second call must succeed without error.
    assert_eq!(b.init(), Ok(()));
    assert!(b.is_initialized());
}

// ── Graceful failure ──────────────────────────────────────────────────────

#[test]
fn webgpu_init_graceful_failure() {
    // We cannot force a failure, but we can at least verify that init()
    // returns a Result and never panics.
    let mut b = WebGpuBackend::new();
    let _result = b.init(); // Ok or Err — both are acceptable.
    // No panic => test passes.
}

// ── GPU compute tests ─────────────────────────────────────────────────────
//
// These helpers upload f32 slices and read back results, exercising the
// full shader → pipeline → dispatch path.

/// Helper: upload `data` (f32 slice) to a new GPU buffer, return its handle.
fn upload_f32(b: &WebGpuBackend, data: &[f32]) -> u64 {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let h = b.alloc(bytes.len()).expect("alloc");
    b.copy_htod(h, &bytes).expect("copy_htod");
    h
}

/// Helper: download `n` f32 values from a GPU buffer handle.
fn download_f32(b: &WebGpuBackend, h: u64, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    b.copy_dtoh(&mut bytes, h).expect("copy_dtoh");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn unary_neg_small() {
    let Some(b) = try_init() else { return };
    let input = [1.0f32, -2.0, 3.0, 0.0];
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(input.len() * 4).expect("alloc output");

    b.unary(UnaryOp::Neg, in_h, out_h, input.len())
        .expect("unary neg");

    let result = download_f32(&b, out_h, input.len());
    let expected = [-1.0f32, 2.0, -3.0, 0.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
    }

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn unary_abs_small() {
    let Some(b) = try_init() else { return };
    let input = [-3.0f32, 4.0, -5.0, 0.0];
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(input.len() * 4).expect("alloc output");

    b.unary(UnaryOp::Abs, in_h, out_h, input.len())
        .expect("unary abs");

    let result = download_f32(&b, out_h, input.len());
    let expected = [3.0f32, 4.0, 5.0, 0.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
    }

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn binary_add_small() {
    let Some(b) = try_init() else { return };
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let bv = [10.0f32, 20.0, 30.0, 40.0];
    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &bv);
    let out_h = b.alloc(a.len() * 4).expect("alloc output");

    b.binary(BinaryOp::Add, a_h, b_h, out_h, a.len())
        .expect("binary add");

    let result = download_f32(&b, out_h, a.len());
    let expected = [11.0f32, 22.0, 33.0, 44.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn binary_mul_small() {
    let Some(b) = try_init() else { return };
    let a = [2.0f32, 3.0, 4.0, 5.0];
    let bv = [10.0f32, 10.0, 10.0, 10.0];
    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &bv);
    let out_h = b.alloc(a.len() * 4).expect("alloc output");

    b.binary(BinaryOp::Mul, a_h, b_h, out_h, a.len())
        .expect("binary mul");

    let result = download_f32(&b, out_h, a.len());
    let expected = [20.0f32, 30.0, 40.0, 50.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(out_h).expect("free");
}

// ── Aliased input/output rejection ──────────────────────────────────────
//
// `elementwise_wgsl`/`binary_wgsl` bind their output as `read_write` at a
// different binding than the (`read`-only) input(s); wgpu's usage-scope
// validation rejects the same buffer being bound as both within one
// dispatch. Before this check existed, an aliased call would still return
// `Ok(())` (the validation error only ever reached the non-fatal
// uncaptured-error slot, see `WebGpuDevice::poll_error`) and the failure
// would surface later, misattributed, from whatever unrelated `alloc`/
// `copy_*`/`synchronize()` call happened to be the next one to drain that
// slot. Rejecting it here — before any dispatch — makes it a clean,
// correctly attributed error at the call that actually caused it.

#[test]
fn unary_rejects_aliased_input_and_output() {
    let Some(b) = try_init() else { return };
    let h = b.alloc(16).expect("alloc");

    let err = b.unary(UnaryOp::Relu, h, h, 4).unwrap_err();
    assert!(
        matches!(err, BackendError::InvalidArgument(_)),
        "got {err:?}"
    );

    b.free(h).expect("free");
}

#[test]
fn binary_rejects_output_aliased_with_either_input() {
    let Some(b) = try_init() else { return };
    let a_h = b.alloc(16).expect("alloc a");
    let b_h = b.alloc(16).expect("alloc b");

    let err_a = b.binary(BinaryOp::Add, a_h, b_h, a_h, 4).unwrap_err();
    assert!(
        matches!(err_a, BackendError::InvalidArgument(_)),
        "a_ptr aliased with output_ptr: got {err_a:?}"
    );

    let err_b = b.binary(BinaryOp::Add, a_h, b_h, b_h, 4).unwrap_err();
    assert!(
        matches!(err_b, BackendError::InvalidArgument(_)),
        "b_ptr aliased with output_ptr: got {err_b:?}"
    );

    b.free(a_h).expect("free a");
    b.free(b_h).expect("free b");
}

#[test]
fn binary_allows_the_two_inputs_to_alias_each_other() {
    // `a_ptr == b_ptr` (neither aliased with `output_ptr`) binds the same
    // buffer to two `read`-only bindings, which does not conflict — only
    // aliasing with the `read_write` output is rejected.
    let Some(b) = try_init() else { return };
    let a_h = upload_f32(&b, &[1.0f32, 2.0, 3.0, 4.0]);
    let out_h = b.alloc(16).expect("alloc output");

    b.binary(BinaryOp::Add, a_h, a_h, out_h, 4)
        .expect("binary with both inputs aliased to each other must be allowed");

    let result = download_f32(&b, out_h, 4);
    let expected = [2.0f32, 4.0, 6.0, 8.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn reduce_sum_small() {
    let Some(b) = try_init() else { return };
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(4).expect("alloc output"); // single f32

    b.reduce(ReduceOp::Sum, in_h, out_h, &[4], 0)
        .expect("reduce sum");

    let result = download_f32(&b, out_h, 1);
    assert!(
        (result[0] - 10.0).abs() < 1e-5,
        "expected 10.0, got {}",
        result[0]
    );

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn reduce_sum_over_65536_elements() {
    // Regression: the 1-D reduce previously dropped every partial past index
    // 255 (i.e. every input element beyond 65 536) because the final pass only
    // touched 256 partials.  100 000 ones must sum to exactly 100 000.
    let Some(b) = try_init() else { return };
    let n = 100_000usize;
    let input = vec![1.0f32; n];
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(4).expect("alloc output");

    b.reduce(ReduceOp::Sum, in_h, out_h, &[n], 0)
        .expect("reduce sum large");

    let result = download_f32(&b, out_h, 1);
    assert!(
        (result[0] - n as f32).abs() < 1.0,
        "expected {n}, got {}",
        result[0]
    );

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn reduce_max_over_65536_elements() {
    // The true maximum lives past element 65 535; the pre-fix kernel would miss
    // it because those partials were never folded in.
    let Some(b) = try_init() else { return };
    let n = 100_000usize;
    let mut input = vec![0.5f32; n];
    input[90_000] = 42.0; // maximum beyond the first 65 536 elements
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(4).expect("alloc output");

    b.reduce(ReduceOp::Max, in_h, out_h, &[n], 0)
        .expect("reduce max large");

    let result = download_f32(&b, out_h, 1);
    assert!(
        (result[0] - 42.0).abs() < 1e-4,
        "expected 42.0, got {}",
        result[0]
    );

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn reduce_max_small() {
    let Some(b) = try_init() else { return };
    let input = [1.0f32, 5.0, 3.0, 2.0];
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(4).expect("alloc output");

    b.reduce(ReduceOp::Max, in_h, out_h, &[4], 0)
        .expect("reduce max");

    let result = download_f32(&b, out_h, 1);
    assert!(
        (result[0] - 5.0).abs() < 1e-5,
        "expected 5.0, got {}",
        result[0]
    );

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn reduce_mean_small() {
    let Some(b) = try_init() else { return };
    let input = [2.0f32, 4.0, 6.0, 8.0];
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(4).expect("alloc output");

    b.reduce(ReduceOp::Mean, in_h, out_h, &[4], 0)
        .expect("reduce mean");

    let result = download_f32(&b, out_h, 1);
    assert!(
        (result[0] - 5.0).abs() < 1e-5,
        "expected 5.0, got {}",
        result[0]
    );

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn gemm_identity_2x2() {
    let Some(b) = try_init() else { return };
    // A = [[1,2],[3,4]], B = [[1,0],[0,1]] (identity), C = zeros
    // C = 1.0 * A * I + 0.0 * C = A
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let eye = [1.0f32, 0.0, 0.0, 1.0];
    let c_init = [0.0f32; 4];

    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &eye);
    let c_h = upload_f32(&b, &c_init);

    b.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        2,
        2,
        2,
        1.0,
        a_h,
        2,
        b_h,
        2,
        0.0,
        c_h,
        2,
    )
    .expect("gemm");

    let result = download_f32(&b, c_h, 4);
    for (r, e) in result.iter().zip(a.iter()) {
        assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

#[test]
fn gemm_2x3_times_3x2() {
    let Some(b) = try_init() else { return };
    // A 2x3, B 3x2 → C 2x2
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bm = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let c_init = [0.0f32; 4];

    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &bm);
    let c_h = upload_f32(&b, &c_init);

    b.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        2,
        2,
        3,
        1.0,
        a_h,
        3,
        b_h,
        2,
        0.0,
        c_h,
        2,
    )
    .expect("gemm");

    // Expected: [[58, 64], [139, 154]]
    let result = download_f32(&b, c_h, 4);
    let expected = [58.0f32, 64.0, 139.0, 154.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-4, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

#[test]
fn gemm_alpha_beta() {
    let Some(b) = try_init() else { return };
    // C = 2.0 * A * B + 3.0 * C
    // A = [[1,0],[0,1]], B = [[1,0],[0,1]], C = [[1,1],[1,1]]
    // C = 2*I + 3*ones = [[5,3],[3,5]]
    let a = [1.0f32, 0.0, 0.0, 1.0];
    let bm = [1.0f32, 0.0, 0.0, 1.0];
    let c_init = [1.0f32, 1.0, 1.0, 1.0];

    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &bm);
    let c_h = upload_f32(&b, &c_init);

    b.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        2,
        2,
        2,
        2.0,
        a_h,
        2,
        b_h,
        2,
        3.0,
        c_h,
        2,
    )
    .expect("gemm alpha+beta");

    let result = download_f32(&b, c_h, 4);
    let expected = [5.0f32, 3.0, 3.0, 5.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-4, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

#[test]
fn gemm_honours_padded_lda() {
    let Some(b) = try_init() else { return };
    // Logical A = [[1,2],[3,4]] stored row-major with a leading dimension of 4
    // (two real columns + two padding columns per row).  A correct kernel must
    // read a[r*lda + i]; the pre-fix kernel read a[r*k + i] and returned wrong
    // values.
    let a_padded = [1.0f32, 2.0, 99.0, 99.0, 3.0, 4.0, 99.0, 99.0];
    let eye = [1.0f32, 0.0, 0.0, 1.0];
    let c_init = [0.0f32; 4];

    let a_h = upload_f32(&b, &a_padded);
    let b_h = upload_f32(&b, &eye);
    let c_h = upload_f32(&b, &c_init);

    b.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        2,
        2,
        2,
        1.0,
        a_h,
        4, // lda > packed (k=2): padded rows
        b_h,
        2,
        0.0,
        c_h,
        2,
    )
    .expect("gemm with padded lda");

    let result = download_f32(&b, c_h, 4);
    let expected = [1.0f32, 2.0, 3.0, 4.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

#[test]
fn gemm_rejects_too_small_lda() {
    let Some(b) = try_init() else { return };
    let a_h = upload_f32(&b, &[1.0f32, 2.0, 3.0, 4.0]);
    let b_h = upload_f32(&b, &[1.0f32, 0.0, 0.0, 1.0]);
    let c_h = upload_f32(&b, &[0.0f32; 4]);

    // lda = 1 is smaller than the packed extent k = 2 → clean InvalidArgument.
    let err = b
        .gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a_h,
            1,
            b_h,
            2,
            0.0,
            c_h,
            2,
        )
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)));

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

// ── Transposed GEMM tests ─────────────────────────────────────────────
//
// These exercise the WGSL `trans_a` / `trans_b` uniform paths, which were
// previously rejected with `BackendError::Unsupported`.

/// Reference CPU GEMM: `C = alpha * op(A) * op(B) + beta * C`, row-major.
///
/// `op(A)` is the logical `m × k` operand; when `trans_a` is true the buffer
/// `a` holds its transpose (`k × m`).  Likewise for `op(B)` / `b`.
fn cpu_gemm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                let av = if trans_a {
                    a[i * m + row]
                } else {
                    a[row * k + i]
                };
                let bv = if trans_b {
                    b[col * k + i]
                } else {
                    b[i * n + col]
                };
                acc += av * bv;
            }
            let idx = row * n + col;
            out[idx] = alpha * acc + beta * c[idx];
        }
    }
    out
}

/// Drive `gemm` for one transpose combination and compare against `cpu_gemm`.
fn run_gemm_transpose_case(
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    m: usize,
    n: usize,
    k: usize,
) {
    let Some(b) = try_init() else { return };

    let ta = trans_a != BackendTranspose::NoTrans;
    let tb = trans_b != BackendTranspose::NoTrans;

    // `a` holds op(A) (m×k) or its transpose (k×m); same idea for `b`.
    let a_data: Vec<f32> = (0..m * k).map(|x| (x as f32) * 0.5 - 1.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|x| (x as f32) * 0.25 + 0.3).collect();
    let c_init: Vec<f32> = (0..m * n).map(|x| (x as f32) * 0.1).collect();
    let alpha = 2.0f32;
    let beta = 3.0f32;

    let expected = cpu_gemm(ta, tb, m, n, k, alpha, &a_data, &b_data, beta, &c_init);

    let a_h = upload_f32(&b, &a_data);
    let b_h = upload_f32(&b, &b_data);
    let c_h = upload_f32(&b, &c_init);

    b.gemm(
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
    .expect("gemm transpose");

    let result = download_f32(&b, c_h, m * n);
    for (idx, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-3 * (1.0 + e.abs()),
            "trans_a={trans_a}, trans_b={trans_b}, slot={idx}: got {r}, expected {e}"
        );
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

#[test]
fn gemm_nn() {
    run_gemm_transpose_case(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        3,
        4,
        5,
    );
}

#[test]
fn gemm_nt() {
    run_gemm_transpose_case(BackendTranspose::NoTrans, BackendTranspose::Trans, 3, 4, 5);
}

#[test]
fn gemm_tn() {
    run_gemm_transpose_case(BackendTranspose::Trans, BackendTranspose::NoTrans, 3, 4, 5);
}

#[test]
fn gemm_tt() {
    run_gemm_transpose_case(BackendTranspose::Trans, BackendTranspose::Trans, 3, 4, 5);
}

#[test]
fn gemm_transpose_larger_than_tile() {
    // Dimensions exceeding the 16×16 tile size (see `gpu_limits`/
    // `planner::plan_workgroup_square`) exercise multi-tile k loops for
    // every transpose combination.
    for &ta in &[BackendTranspose::NoTrans, BackendTranspose::Trans] {
        for &tb in &[BackendTranspose::NoTrans, BackendTranspose::Trans] {
            run_gemm_transpose_case(ta, tb, 17, 13, 23);
        }
    }
}

#[test]
fn gemm_conjtrans_treated_as_trans() {
    // For real f32 buffers ConjTrans must behave exactly like Trans.
    run_gemm_transpose_case(
        BackendTranspose::ConjTrans,
        BackendTranspose::ConjTrans,
        4,
        4,
        4,
    );
}

#[test]
fn gemm_transpose_known_values() {
    // A_logical = [[1,2,3],[4,5,6]] (2×3).  Stored transposed (3×2) for TN.
    // B = [[1,0],[0,1],[1,1]] (3×2).  C = A_logical * B = [[4,5],[10,11]].
    let Some(b) = try_init() else { return };

    let a_transposed = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]; // column-major 3×2
    let bm = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let c_init = [0.0f32; 4];

    let a_h = upload_f32(&b, &a_transposed);
    let b_h = upload_f32(&b, &bm);
    let c_h = upload_f32(&b, &c_init);

    b.gemm(
        BackendTranspose::Trans,
        BackendTranspose::NoTrans,
        2,
        2,
        3,
        1.0,
        a_h,
        2,
        b_h,
        2,
        0.0,
        c_h,
        2,
    )
    .expect("gemm TN");

    let result = download_f32(&b, c_h, 4);
    let expected = [4.0f32, 5.0, 10.0, 11.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-4, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

// ── Conv2D tests ──────────────────────────────────────────────────────

#[test]
fn conv2d_identity_1x1() {
    // 1×1 convolution with single channel, no padding, stride=1
    // input: 1×1×3×3, filter: 1×1×1×1 (weight=2.0), output: 1×1×3×3
    let Some(b) = try_init() else { return };
    let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let filter = [2.0f32];
    let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();

    let in_h = upload_f32(&b, &input);
    let f_h = upload_f32(&b, &filter);
    let out_h = b.alloc(9 * 4).expect("alloc output");

    b.conv2d_forward(
        in_h,
        &[1, 1, 3, 3],
        f_h,
        &[1, 1, 1, 1],
        out_h,
        &[1, 1, 3, 3],
        &[1, 1],
        &[0, 0],
    )
    .expect("conv2d");

    let result = download_f32(&b, out_h, 9);
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
    }

    b.free(in_h).expect("free");
    b.free(f_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn conv2d_3x3_no_padding() {
    // input: 1×1×4×4, filter: 1×1×3×3 (all ones), stride=1, pad=0
    // output: 1×1×2×2
    let Some(b) = try_init() else { return };
    let input: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let filter = [1.0f32; 9];

    let in_h = upload_f32(&b, &input);
    let f_h = upload_f32(&b, &filter);
    let out_h = b.alloc(4 * 4).expect("alloc output");

    b.conv2d_forward(
        in_h,
        &[1, 1, 4, 4],
        f_h,
        &[1, 1, 3, 3],
        out_h,
        &[1, 1, 2, 2],
        &[1, 1],
        &[0, 0],
    )
    .expect("conv2d");

    let result = download_f32(&b, out_h, 4);
    // top-left 3×3 sum: 0+1+2+4+5+6+8+9+10 = 45
    assert!((result[0] - 45.0).abs() < 1e-4, "got {}", result[0]);
    // top-right 3×3 sum: 1+2+3+5+6+7+9+10+11 = 54
    assert!((result[1] - 54.0).abs() < 1e-4, "got {}", result[1]);

    b.free(in_h).expect("free");
    b.free(f_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn conv2d_with_padding() {
    // input: 1×1×2×2, filter: 1×1×3×3 (all ones), stride=1, pad=1
    // output: 1×1×2×2
    // With padding=1 around a 2×2 input, the output is also 2×2.
    let Some(b) = try_init() else { return };
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let filter = [1.0f32; 9];

    let in_h = upload_f32(&b, &input);
    let f_h = upload_f32(&b, &filter);
    let out_h = b.alloc(4 * 4).expect("alloc output");

    b.conv2d_forward(
        in_h,
        &[1, 1, 2, 2],
        f_h,
        &[1, 1, 3, 3],
        out_h,
        &[1, 1, 2, 2],
        &[1, 1],
        &[1, 1],
    )
    .expect("conv2d");

    let result = download_f32(&b, out_h, 4);
    // Top-left output: only 4 of 9 filter taps hit valid input
    // input[0,0]=1, input[0,1]=2, input[1,0]=3, input[1,1]=4 => sum=10
    assert!((result[0] - 10.0).abs() < 1e-4, "got {}", result[0]);

    b.free(in_h).expect("free");
    b.free(f_h).expect("free");
    b.free(out_h).expect("free");
}

// ── Attention tests ───────────────────────────────────────────────────

#[test]
fn attention_uniform_weights() {
    // 1 head, seq_q=1, seq_kv=2, head_dim=2, no causal
    // Q = [1, 0], K = [[1, 0], [1, 0]], V = [[1, 2], [3, 4]]
    // scores = [1*scale, 1*scale] => equal weights => O = mean(V) = [2, 3]
    let Some(b) = try_init() else { return };

    let q = [1.0f32, 0.0];
    let k = [1.0f32, 0.0, 1.0, 0.0];
    let v = [1.0f32, 2.0, 3.0, 4.0];

    let q_h = upload_f32(&b, &q);
    let k_h = upload_f32(&b, &k);
    let v_h = upload_f32(&b, &v);
    let o_h = b.alloc(2 * 4).expect("alloc output");

    b.attention(q_h, k_h, v_h, o_h, 1, 1, 1, 2, 2, 1.0, false)
        .expect("attention");

    let result = download_f32(&b, o_h, 2);
    // Equal scores → equal softmax weights → average of V rows
    assert!(
        (result[0] - 2.0).abs() < 1e-4,
        "got {}, expected 2.0",
        result[0]
    );
    assert!(
        (result[1] - 3.0).abs() < 1e-4,
        "got {}, expected 3.0",
        result[1]
    );

    b.free(q_h).expect("free");
    b.free(k_h).expect("free");
    b.free(v_h).expect("free");
    b.free(o_h).expect("free");
}

#[test]
fn attention_causal_single_token() {
    // 1 head, seq_q=2, seq_kv=2, head_dim=1, causal
    // Q = [1, 1], K = [1, 1], V = [10, 20]
    // sq=0: only sees sk=0 → O[0] = V[0] = 10
    // sq=1: sees sk=0,1 with equal scores → O[1] = (10+20)/2 = 15
    let Some(b) = try_init() else { return };

    let q = [1.0f32, 1.0];
    let k = [1.0f32, 1.0];
    let v = [10.0f32, 20.0];

    let q_h = upload_f32(&b, &q);
    let k_h = upload_f32(&b, &k);
    let v_h = upload_f32(&b, &v);
    let o_h = b.alloc(2 * 4).expect("alloc output");

    b.attention(q_h, k_h, v_h, o_h, 1, 1, 2, 2, 1, 1.0, true)
        .expect("attention causal");

    let result = download_f32(&b, o_h, 2);
    assert!(
        (result[0] - 10.0).abs() < 1e-4,
        "got {}, expected 10.0",
        result[0]
    );
    assert!(
        (result[1] - 15.0).abs() < 1e-4,
        "got {}, expected 15.0",
        result[1]
    );

    b.free(q_h).expect("free");
    b.free(k_h).expect("free");
    b.free(v_h).expect("free");
    b.free(o_h).expect("free");
}

// ── Batched GEMM tests ─────────────────────────────────────────────

#[test]
fn batched_gemm_not_initialized() {
    let b = WebGpuBackend::new();
    let result = b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        4,
        4,
        4,
        1.0,
        0,
        4,
        16,
        0,
        4,
        16,
        0.0,
        0,
        4,
        16,
        2,
    );
    assert_eq!(result, Err(BackendError::NotInitialized));
}

#[test]
fn batched_gemm_zero_batch_noop() {
    let Some(b) = try_init() else { return };
    let result = b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        4,
        4,
        4,
        1.0,
        0,
        4,
        16,
        0,
        4,
        16,
        0.0,
        0,
        4,
        16,
        0, // batch_count = 0
    );
    assert_eq!(result, Ok(()));
}

#[test]
fn batched_gemm_zero_dims_noop() {
    let Some(b) = try_init() else { return };
    // m = 0
    let result = b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        0,
        4,
        4,
        1.0,
        0,
        4,
        16,
        0,
        4,
        16,
        0.0,
        0,
        4,
        16,
        2,
    );
    assert_eq!(result, Ok(()));
    // n = 0
    let result = b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        4,
        0,
        4,
        1.0,
        0,
        4,
        16,
        0,
        4,
        16,
        0.0,
        0,
        4,
        16,
        2,
    );
    assert_eq!(result, Ok(()));
    // k = 0
    let result = b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        4,
        4,
        0,
        1.0,
        0,
        4,
        16,
        0,
        4,
        16,
        0.0,
        0,
        4,
        16,
        2,
    );
    assert_eq!(result, Ok(()));
}

#[test]
fn batched_gemm_identity_2x2() {
    let Some(b) = try_init() else { return };
    // 2 batches of 2x2 identity multiply
    // batch 0: A0=[[1,2],[3,4]] * I = [[1,2],[3,4]]
    // batch 1: A1=[[5,6],[7,8]] * I = [[5,6],[7,8]]
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let eye = [1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let c_init = [0.0f32; 8];

    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &eye);
    let c_h = upload_f32(&b, &c_init);

    b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        2,
        2,
        2,
        1.0,
        a_h,
        2,
        4, // stride_a = 2*2 = 4
        b_h,
        2,
        4, // stride_b = 4
        0.0,
        c_h,
        2,
        4, // stride_c = 4
        2, // batch_count
    )
    .expect("batched_gemm");

    let result = download_f32(&b, c_h, 8);
    for (r, e) in result.iter().zip(a.iter()) {
        assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

// ── Batched transposed GEMM tests ──────────────────────────────────
//
// Exercise the WGSL `trans_a` / `trans_b` uniforms in the batched kernel,
// previously rejected with `BackendError::Unsupported`.

/// Drive `batched_gemm` for one transpose combination and compare per batch
/// against `cpu_gemm`.
fn run_batched_gemm_transpose_case(
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
) {
    let Some(b) = try_init() else { return };

    let ta = trans_a != BackendTranspose::NoTrans;
    let tb = trans_b != BackendTranspose::NoTrans;

    let stride_a = m * k;
    let stride_b = k * n;
    let stride_c = m * n;

    let a_data: Vec<f32> = (0..stride_a * batch_count)
        .map(|x| (x as f32) * 0.3 - 0.7)
        .collect();
    let b_data: Vec<f32> = (0..stride_b * batch_count)
        .map(|x| (x as f32) * 0.15 + 0.2)
        .collect();
    let c_init: Vec<f32> = (0..stride_c * batch_count)
        .map(|x| (x as f32) * 0.05)
        .collect();
    let alpha = 1.5f32;
    let beta = 0.5f32;

    // Per-batch expected output via the CPU reference.
    let mut expected = vec![0.0f32; stride_c * batch_count];
    for batch in 0..batch_count {
        let a_slice = &a_data[batch * stride_a..(batch + 1) * stride_a];
        let b_slice = &b_data[batch * stride_b..(batch + 1) * stride_b];
        let c_slice = &c_init[batch * stride_c..(batch + 1) * stride_c];
        let out = cpu_gemm(ta, tb, m, n, k, alpha, a_slice, b_slice, beta, c_slice);
        expected[batch * stride_c..(batch + 1) * stride_c].copy_from_slice(&out);
    }

    let a_h = upload_f32(&b, &a_data);
    let b_h = upload_f32(&b, &b_data);
    let c_h = upload_f32(&b, &c_init);

    b.batched_gemm(
        trans_a,
        trans_b,
        m,
        n,
        k,
        alpha as f64,
        a_h,
        if ta { m } else { k },
        stride_a,
        b_h,
        if tb { k } else { n },
        stride_b,
        beta as f64,
        c_h,
        n,
        stride_c,
        batch_count,
    )
    .expect("batched_gemm transpose");

    let result = download_f32(&b, c_h, stride_c * batch_count);
    for (idx, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-3 * (1.0 + e.abs()),
            "trans_a={trans_a}, trans_b={trans_b}, slot={idx}: got {r}, expected {e}"
        );
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

#[test]
fn batched_gemm_nn() {
    run_batched_gemm_transpose_case(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        3,
        4,
        5,
        3,
    );
}

#[test]
fn batched_gemm_nt() {
    run_batched_gemm_transpose_case(
        BackendTranspose::NoTrans,
        BackendTranspose::Trans,
        3,
        4,
        5,
        3,
    );
}

#[test]
fn batched_gemm_tn() {
    run_batched_gemm_transpose_case(
        BackendTranspose::Trans,
        BackendTranspose::NoTrans,
        3,
        4,
        5,
        3,
    );
}

#[test]
fn batched_gemm_tt() {
    run_batched_gemm_transpose_case(BackendTranspose::Trans, BackendTranspose::Trans, 3, 4, 5, 3);
}

#[test]
fn batched_gemm_transpose_larger_than_tile() {
    // Dimensions exceeding the 16×16 tile force multi-tile k loops per batch.
    for &ta in &[BackendTranspose::NoTrans, BackendTranspose::Trans] {
        for &tb in &[BackendTranspose::NoTrans, BackendTranspose::Trans] {
            run_batched_gemm_transpose_case(ta, tb, 11, 19, 23, 2);
        }
    }
}

// ── FP16 GEMM tests ─────────────────────────────────────────────────

#[test]
fn gemm_f16_not_initialized() {
    let b = WebGpuBackend::new();
    let result = b.gemm_f16(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        4,
        4,
        4,
        1.0,
        0,
        4,
        0,
        4,
        0.0,
        0,
        4,
    );
    assert_eq!(result, Err(BackendError::NotInitialized));
}

#[test]
fn gemm_f16_zero_dims_noop() {
    let Some(b) = try_init() else { return };
    let nt = BackendTranspose::NoTrans;
    assert_eq!(
        b.gemm_f16(nt, nt, 0, 4, 4, 1.0, 0, 4, 0, 4, 0.0, 0, 4),
        Ok(())
    );
    assert_eq!(
        b.gemm_f16(nt, nt, 4, 0, 4, 1.0, 0, 4, 0, 4, 0.0, 0, 4),
        Ok(())
    );
    assert_eq!(
        b.gemm_f16(nt, nt, 4, 4, 0, 1.0, 0, 4, 0, 4, 0.0, 0, 4),
        Ok(())
    );
}

#[test]
fn gemm_f16_real_dims_never_panics() {
    // Regression: with real dimensions the f16 GEMM used to unconditionally
    // build a module declaring `enable f16;` on a device created without the
    // SHADER_F16 feature, which surfaced as a process-fatal validation error.
    // It must now either run (feature enabled) or return a typed Unsupported
    // error — never panic.
    let Some(b) = try_init() else { return };
    // 2×2 f16 operands: 2 bytes per element.
    let a = b.alloc(4 * 2).expect("alloc a");
    let bm = b.alloc(4 * 2).expect("alloc b");
    let c = b.alloc(4 * 2).expect("alloc c");
    let zeros = [0u8; 8];
    b.copy_htod(a, &zeros).expect("htod a");
    b.copy_htod(bm, &zeros).expect("htod b");
    b.copy_htod(c, &zeros).expect("htod c");

    let nt = BackendTranspose::NoTrans;
    match b.gemm_f16(nt, nt, 2, 2, 2, 1.0, a, 2, bm, 2, 0.0, c, 2) {
        Ok(()) => {}
        Err(BackendError::Unsupported(_)) => {}
        Err(e) => panic!("unexpected gemm_f16 error: {e:?}"),
    }

    b.free(a).expect("free");
    b.free(bm).expect("free");
    b.free(c).expect("free");
}

/// Upload an f32 slice to the GPU as f16 (half-precision), return the handle.
fn upload_f16(b: &WebGpuBackend, data: &[f32]) -> u64 {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect();
    let h = b.alloc(bytes.len()).expect("alloc f16");
    b.copy_htod(h, &bytes).expect("copy_htod f16");
    h
}

/// Download `n` f16 values from a GPU buffer handle, widened back to f32.
fn download_f16(b: &WebGpuBackend, h: u64, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 2];
    b.copy_dtoh(&mut bytes, h).expect("copy_dtoh f16");
    bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

#[test]
fn gemm_f16_matches_reference_2x3_times_3x2() {
    let Some(b) = try_init() else { return };
    if !b.supports_f16() {
        return; // Adapter lacks SHADER_F16; nothing to exercise.
    }
    // Same shape/values as the f32 `gemm_2x3_times_3x2` test.
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bm = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let c_init = [0.0f32; 4];

    let a_h = upload_f16(&b, &a);
    let b_h = upload_f16(&b, &bm);
    let c_h = upload_f16(&b, &c_init);

    let nt = BackendTranspose::NoTrans;
    b.gemm_f16(nt, nt, 2, 2, 3, 1.0, a_h, 3, b_h, 2, 0.0, c_h, 2)
        .expect("gemm_f16");

    let result = download_f16(&b, c_h, 4);
    let expected = [58.0f32, 64.0, 139.0, 154.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        // f16 has ~3 decimal digits of precision; these exact integer-valued
        // products are still representable, so a loose tolerance suffices.
        assert!((r - e).abs() < 0.5, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

#[test]
fn gemm_f16_honours_transpose_and_lda() {
    // Mirrors `gemm_transpose_known_values`: A stored transposed (TN), lda
    // padded, proving the f16 kernel's `load_a`/`load_b` (added to fix
    // finding webgpu-14) are actually wired up rather than ignored.
    let Some(b) = try_init() else { return };
    if !b.supports_f16() {
        return;
    }
    // A_logical = [[1,2,3],[4,5,6]] (2×3), stored transposed (3×2) with a
    // padded lda of 3 (one padding column per stored row).
    let a_transposed_padded = [1.0f32, 4.0, 99.0, 2.0, 5.0, 99.0, 3.0, 6.0, 99.0];
    let bm = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let c_init = [0.0f32; 4];

    let a_h = upload_f16(&b, &a_transposed_padded);
    let b_h = upload_f16(&b, &bm);
    let c_h = upload_f16(&b, &c_init);

    b.gemm_f16(
        BackendTranspose::Trans,
        BackendTranspose::NoTrans,
        2,
        2,
        3,
        1.0,
        a_h,
        3, // lda > packed extent (m=2): one padding column per row
        b_h,
        2,
        0.0,
        c_h,
        2,
    )
    .expect("gemm_f16 TN with padded lda");

    let result = download_f16(&b, c_h, 4);
    let expected = [4.0f32, 5.0, 10.0, 11.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 0.5, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

#[test]
fn gemm_f16_rejects_too_small_lda() {
    let Some(b) = try_init() else { return };
    if !b.supports_f16() {
        return;
    }
    let a_h = upload_f16(&b, &[1.0f32, 2.0, 3.0, 4.0]);
    let b_h = upload_f16(&b, &[1.0f32, 0.0, 0.0, 1.0]);
    let c_h = upload_f16(&b, &[0.0f32; 4]);

    let nt = BackendTranspose::NoTrans;
    let err = b
        .gemm_f16(nt, nt, 2, 2, 2, 1.0, a_h, 1, b_h, 2, 0.0, c_h, 2)
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)));

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

// ── N-D reduce tests ──────────────────────────────────────────────────
//
// Each test computes the same reduction on the host and compares.

/// Reference CPU reduction along `axis` of a row-major tensor.
fn cpu_reduce(data: &[f32], shape: &[usize], axis: usize, op: ReduceOp) -> Vec<f32> {
    let outer: usize = shape[..axis].iter().product();
    let dk: usize = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let total = outer * inner;

    let neutral = match op {
        ReduceOp::Sum | ReduceOp::Mean => 0.0f32,
        ReduceOp::Max => f32::NEG_INFINITY,
        ReduceOp::Min => f32::INFINITY,
    };

    let mut out = vec![neutral; total];
    for o in 0..outer {
        for i in 0..dk {
            for j in 0..inner {
                let v = data[o * dk * inner + i * inner + j];
                let slot = o * inner + j;
                out[slot] = match op {
                    ReduceOp::Sum | ReduceOp::Mean => out[slot] + v,
                    ReduceOp::Max => out[slot].max(v),
                    ReduceOp::Min => out[slot].min(v),
                };
            }
        }
    }
    if op == ReduceOp::Mean && dk > 0 {
        for v in &mut out {
            *v /= dk as f32;
        }
    }
    out
}

fn run_nd_reduce_case(data: Vec<f32>, shape: Vec<usize>, axis: usize, op: ReduceOp) {
    let Some(b) = try_init() else { return };

    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let out_len = outer * inner;
    let expected = cpu_reduce(&data, &shape, axis, op);

    let in_h = upload_f32(&b, &data);
    let out_h = b.alloc(out_len * 4).expect("alloc output");

    b.reduce(op, in_h, out_h, &shape, axis).expect("reduce nd");

    let result = download_f32(&b, out_h, out_len);
    for (idx, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        // Use a slightly relaxed tolerance — large dk can accumulate error.
        assert!(
            (r - e).abs() < 1e-3 * (1.0 + e.abs()),
            "axis={axis}, op={op:?}, slot={idx}: got {r}, expected {e}"
        );
    }

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn reduce_2d_sum_axis0() {
    // shape [3, 4], reduce axis 0 → length 4
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![3, 4], 0, ReduceOp::Sum);
}

#[test]
fn reduce_2d_sum_axis1() {
    // shape [3, 4], reduce axis 1 → length 3
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![3, 4], 1, ReduceOp::Sum);
}

#[test]
fn reduce_2d_max_axis0() {
    let data = vec![1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 6.0, 4.0, 9.0, 0.0, 1.5, 2.5];
    run_nd_reduce_case(data, vec![3, 4], 0, ReduceOp::Max);
}

#[test]
fn reduce_2d_min_axis1() {
    let data = vec![1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 6.0, 4.0, 9.0, 0.0, 1.5, 2.5];
    run_nd_reduce_case(data, vec![3, 4], 1, ReduceOp::Min);
}

#[test]
fn reduce_2d_mean_axis0() {
    // Mean along axis 0 should divide by dk = 3 inside the shader.
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![3, 4], 0, ReduceOp::Mean);
}

#[test]
fn reduce_3d_sum_axis0() {
    // shape [2, 3, 4] → output [3, 4]
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![2, 3, 4], 0, ReduceOp::Sum);
}

#[test]
fn reduce_3d_sum_axis1() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![2, 3, 4], 1, ReduceOp::Sum);
}

#[test]
fn reduce_3d_sum_axis2() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![2, 3, 4], 2, ReduceOp::Sum);
}

#[test]
fn reduce_3d_mean_axis1() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![2, 3, 4], 1, ReduceOp::Mean);
}

#[test]
fn reduce_4d_sum_axis2() {
    // shape [2, 3, 4, 2] → output [2, 3, 2]
    let data: Vec<f32> = (0..48).map(|x| x as f32 * 0.5).collect();
    run_nd_reduce_case(data, vec![2, 3, 4, 2], 2, ReduceOp::Sum);
}

#[test]
fn reduce_4d_max_axis3() {
    let data: Vec<f32> = (0..48).map(|x| ((x * 13) % 47) as f32).collect();
    run_nd_reduce_case(data, vec![2, 3, 4, 2], 3, ReduceOp::Max);
}

#[test]
fn reduce_2d_dk_one_is_copy() {
    // shape [3, 1], reduce along axis 1 → output [3], values copied.
    let data = vec![7.0f32, -2.5, 3.25];
    run_nd_reduce_case(data, vec![3, 1], 1, ReduceOp::Sum);
}

#[test]
fn reduce_3d_dk_one_axis0() {
    // shape [1, 3, 4], reduce axis 0 → output [3, 4], values copied.
    let data: Vec<f32> = (0..12).map(|x| x as f32 + 0.5).collect();
    run_nd_reduce_case(data, vec![1, 3, 4], 0, ReduceOp::Mean);
}

#[test]
fn reduce_2d_single_output_via_nd() {
    // shape [1, 5], axis 1 → output [1].  Exercises outer=1, inner=1
    // through the N-D path (not the 1-D fast path because shape.len() > 1).
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    run_nd_reduce_case(data, vec![1, 5], 1, ReduceOp::Sum);
}

#[test]
fn reduce_large_dk_strided_loop() {
    // dk = 1024 exercises the per-thread strided loop (256 threads, 4 iter).
    let data: Vec<f32> = (0..1024).map(|x| (x as f32) * 0.001).collect();
    run_nd_reduce_case(data, vec![1, 1024], 1, ReduceOp::Sum);
}

#[test]
fn attention_dominant_key() {
    // 1 head, seq_q=1, seq_kv=2, head_dim=2, no causal
    // Q = [1, 0], K = [[10, 0], [0, 0]], V = [[100, 200], [0, 0]]
    // score[0] = 10*scale, score[1] = 0*scale
    // With large enough difference, softmax saturates → O ≈ V[0]
    let Some(b) = try_init() else { return };

    let q = [1.0f32, 0.0];
    let k = [10.0f32, 0.0, 0.0, 0.0];
    let v = [100.0f32, 200.0, 0.0, 0.0];

    let q_h = upload_f32(&b, &q);
    let k_h = upload_f32(&b, &k);
    let v_h = upload_f32(&b, &v);
    let o_h = b.alloc(2 * 4).expect("alloc output");

    // scale=1.0 gives scores 10 vs 0 → softmax ≈ [1, 0]
    b.attention(q_h, k_h, v_h, o_h, 1, 1, 1, 2, 2, 1.0, false)
        .expect("attention dominant");

    let result = download_f32(&b, o_h, 2);
    assert!(
        (result[0] - 100.0).abs() < 0.1,
        "got {}, expected ~100",
        result[0]
    );
    assert!(
        (result[1] - 200.0).abs() < 0.1,
        "got {}, expected ~200",
        result[1]
    );

    b.free(q_h).expect("free");
    b.free(k_h).expect("free");
    b.free(v_h).expect("free");
    b.free(o_h).expect("free");
}

// GPU-vs-CPU dispatch-wiring regression tests (attention/conv2d oracle
// comparisons, batched_gemm/reduce validation) — split into a sibling file to
// keep this one under the 2 000-line refactoring policy.  See that file's
// module doc for what it covers.
#[path = "backend_tests_gpu_ops.rs"]
mod gpu_ops_tests;

// Dispatch-ordering / caching regression tests (zero host sync between
// chained ops) — split into a sibling file for the same reason as
// `gpu_ops_tests` above.  See that file's module doc for what it covers.
#[path = "backend_tests_pipeline.rs"]
mod pipeline_tests;

// Multi-tile numeric correctness tests for the tiled `gemm_f16` kernel —
// split into a sibling file for the same reason as `gpu_ops_tests` above.
// See that file's module doc for what it covers.
#[path = "backend_tests_gemm_f16.rs"]
mod gemm_f16_tests;
