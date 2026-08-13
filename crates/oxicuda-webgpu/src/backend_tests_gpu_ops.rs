//! GPU-vs-CPU dispatch-wiring regression tests, split out of `backend_tests.rs`
//! (as `backend::tests::gpu_ops_tests`, via `#[path = ...] mod
//! gpu_ops_tests;`) purely to keep that file under the workspace's 2 000-line
//! refactoring policy — mirrors how `backend_tests.rs` itself is split out of
//! `backend.rs`.
//!
//! `conv2d_forward` and `attention` now dispatch to the real GPU kernels
//! (`shader::conv2d_wgsl` / `shader::attention_wgsl`) instead of always
//! running the CPU fallback loop.  Every conv2d/attention test in the parent
//! file already exercises that path and passes, which is itself a
//! GPU-vs-hand-computed-value check — the tests below add: (1) an explicit
//! comparison against the `conv2d_cpu_reference` / `attention_cpu_reference`
//! oracles for shapes too complex to hand-compute, (2) shapes wide enough to
//! span more than one workgroup (every test in the parent file fits in a
//! single tile), (3) the direct regression test for the `o_buf[..] +=`
//! stale-accumulation bug that motivated zero-initialising `attention_wgsl`'s
//! output row, and (4) `batched_gemm`/`reduce` validation regressions
//! (checked u32 casts, dispatch-grid cap, buffer-size checks).

use super::*;

// ── GPU-vs-CPU dispatch wiring regressions ────────────────────────────────
//
// `conv2d_forward` and `attention` now dispatch to the real GPU kernels
// (`shader::conv2d_wgsl` / `shader::attention_wgsl`) instead of always
// running the CPU fallback loop.  Every existing conv2d/attention test above
// already exercises that path and passes, which is itself a GPU-vs-hand-
// computed-value check — the tests below add: (1) an explicit comparison
// against the `conv2d_cpu_reference` / `attention_cpu_reference` oracles for
// shapes too complex to hand-compute, (2) shapes wide enough to span more
// than one workgroup (every test above fits in a single tile), and (3) the
// direct regression test for the `o_buf[..] +=` stale-accumulation bug that
// motivated zero-initialising `attention_wgsl`'s output row.

#[test]
fn attention_reused_output_buffer_does_not_accumulate_stale_data() {
    // Regression for the attention_wgsl accumulation bug (finding webgpu-9):
    // pass 2 does `o_buf[..] += w * v_buf[..]`, so dispatching into a buffer
    // that already holds a PREVIOUS attention result must not leak that
    // result into the new sum.  Call `attention` twice into the *same*
    // output handle with two different inputs; the second result must equal
    // the second computation alone, not first + second.
    let Some(b) = try_init() else { return };

    let o_h = b.alloc(2 * 4).expect("alloc output");

    // First call: uniform weights -> O = mean(V) = [2, 3] (same values as
    // `attention_uniform_weights`).
    let q1 = [1.0f32, 0.0];
    let k1 = [1.0f32, 0.0, 1.0, 0.0];
    let v1 = [1.0f32, 2.0, 3.0, 4.0];
    let q1_h = upload_f32(&b, &q1);
    let k1_h = upload_f32(&b, &k1);
    let v1_h = upload_f32(&b, &v1);
    b.attention(q1_h, k1_h, v1_h, o_h, 1, 1, 1, 2, 2, 1.0, false)
        .expect("attention #1");
    let first = download_f32(&b, o_h, 2);
    assert!((first[0] - 2.0).abs() < 1e-4);
    assert!((first[1] - 3.0).abs() < 1e-4);

    // Second call into the SAME o_h, with a completely different result
    // (dominant-key case, same values as `attention_dominant_key`):
    // O ≈ V[0] = [100, 200].
    let q2 = [1.0f32, 0.0];
    let k2 = [10.0f32, 0.0, 0.0, 0.0];
    let v2 = [100.0f32, 200.0, 0.0, 0.0];
    let q2_h = upload_f32(&b, &q2);
    let k2_h = upload_f32(&b, &k2);
    let v2_h = upload_f32(&b, &v2);
    b.attention(q2_h, k2_h, v2_h, o_h, 1, 1, 1, 2, 2, 1.0, false)
        .expect("attention #2 (reused buffer)");
    let second = download_f32(&b, o_h, 2);

    // Pre-fix, `second` would be roughly `first + [100, 200]` = `[102, 203]`
    // (or worse, depending on whatever garbage the buffer held) instead of
    // exactly `[100, 200]`.
    assert!(
        (second[0] - 100.0).abs() < 0.1,
        "got {}, expected ~100 (stale accumulation bug?)",
        second[0]
    );
    assert!(
        (second[1] - 200.0).abs() < 0.1,
        "got {}, expected ~200 (stale accumulation bug?)",
        second[1]
    );

    b.free(q1_h).expect("free");
    b.free(k1_h).expect("free");
    b.free(v1_h).expect("free");
    b.free(q2_h).expect("free");
    b.free(k2_h).expect("free");
    b.free(v2_h).expect("free");
    b.free(o_h).expect("free");
}

/// Drive `conv2d_forward` (real GPU dispatch) and compare against
/// `conv2d_cpu_reference` — the same oracle function the production CPU
/// fallback path uses, so the two paths are proven to agree rather than
/// merely both existing.
#[allow(clippy::too_many_arguments)]
fn run_conv2d_gpu_vs_cpu_case(
    batch: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    k_out: usize,
    fh: usize,
    fw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) {
    let Some(b) = try_init() else { return };

    let oh = (h_in + 2 * ph - fh) / sh + 1;
    let ow = (w_in + 2 * pw - fw) / sw + 1;

    let in_data: Vec<f32> = (0..batch * c_in * h_in * w_in)
        .map(|x| ((x % 13) as f32) * 0.3 - 1.5)
        .collect();
    let f_data: Vec<f32> = (0..k_out * c_in * fh * fw)
        .map(|x| ((x % 7) as f32) * 0.2 - 0.6)
        .collect();

    let expected = conv2d_cpu_reference(
        &in_data, &f_data, batch, c_in, h_in, w_in, k_out, fh, fw, oh, ow, sh, sw, ph, pw,
    );

    let in_h = upload_f32(&b, &in_data);
    let f_h = upload_f32(&b, &f_data);
    let out_h = b.alloc(batch * k_out * oh * ow * 4).expect("alloc output");

    b.conv2d_forward(
        in_h,
        &[batch, c_in, h_in, w_in],
        f_h,
        &[k_out, c_in, fh, fw],
        out_h,
        &[batch, k_out, oh, ow],
        &[sh, sw],
        &[ph, pw],
    )
    .expect("conv2d_forward");

    let result = download_f32(&b, out_h, batch * k_out * oh * ow);
    for (idx, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-3 * (1.0 + e.abs()),
            "batch={batch} c_in={c_in} k_out={k_out} slot={idx}: got {r}, expected {e}"
        );
    }

    b.free(in_h).expect("free");
    b.free(f_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn conv2d_multi_channel_multi_filter_multi_batch() {
    // c_in > 1, k_out > 1, batch > 1 — every conv2d test above this point
    // used c_in = k_out = batch = 1.
    run_conv2d_gpu_vs_cpu_case(2, 3, 6, 6, 4, 3, 3, 1, 1, 0, 0);
}

#[test]
fn conv2d_spans_multiple_workgroups() {
    // ow = 20 exceeds the shader's @workgroup_size(8, 8) tile in X, and
    // batch * k_out * oh = 1 * 5 * 20 = 100 exceeds it in Y — exercises a
    // multi-workgroup dispatch grid, not just a single (possibly
    // partially-out-of-range) tile.
    run_conv2d_gpu_vs_cpu_case(1, 2, 22, 22, 5, 3, 3, 1, 1, 0, 0);
}

#[test]
fn conv2d_strided_with_padding_multi_channel() {
    run_conv2d_gpu_vs_cpu_case(1, 2, 9, 9, 3, 3, 3, 2, 2, 1, 1);
}

#[test]
fn attention_spans_multiple_workgroups() {
    // batch_heads * seq_q = 2 * 40 = 80 exceeds @workgroup_size(64) — spans
    // more than one workgroup's worth of (bh, sq) rows, unlike every
    // attention test above (all single-digit seq_q/seq_kv).  Causal, so this
    // also exercises the per-element mask across a wide seq_q range.
    let Some(b) = try_init() else { return };

    let batch = 2usize;
    let heads = 1usize;
    let seq_q = 40usize;
    let seq_kv = 6usize;
    let head_dim = 8usize;
    let batch_heads = batch * heads;

    let q_data: Vec<f32> = (0..batch_heads * seq_q * head_dim)
        .map(|x| ((x % 11) as f32) * 0.1 - 0.5)
        .collect();
    let k_data: Vec<f32> = (0..batch_heads * seq_kv * head_dim)
        .map(|x| ((x % 7) as f32) * 0.15 + 0.2)
        .collect();
    let v_data: Vec<f32> = (0..batch_heads * seq_kv * head_dim)
        .map(|x| ((x % 5) as f32) * 0.3 - 0.6)
        .collect();

    let expected = attention_cpu_reference(
        &q_data,
        &k_data,
        &v_data,
        batch_heads,
        seq_q,
        seq_kv,
        head_dim,
        0.125,
        true,
    );

    let q_h = upload_f32(&b, &q_data);
    let k_h = upload_f32(&b, &k_data);
    let v_h = upload_f32(&b, &v_data);
    let o_h = b
        .alloc(batch_heads * seq_q * head_dim * 4)
        .expect("alloc output");

    b.attention(
        q_h, k_h, v_h, o_h, batch, heads, seq_q, seq_kv, head_dim, 0.125, true,
    )
    .expect("attention multi-workgroup");

    let result = download_f32(&b, o_h, batch_heads * seq_q * head_dim);
    for (idx, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-3 * (1.0 + e.abs()),
            "slot={idx}: got {r}, expected {e}"
        );
    }

    b.free(q_h).expect("free");
    b.free(k_h).expect("free");
    b.free(v_h).expect("free");
    b.free(o_h).expect("free");
}

// ── batched_gemm validation regressions (checked u32 casts, dispatch cap) ──

#[test]
fn batched_gemm_rejects_oversize_batch_count() {
    let Some(b) = try_init() else { return };
    // batch_count exceeds max_compute_workgroups_per_dimension (65 535); the
    // dispatch Z axis cannot express it, so this must be a clean typed error
    // (checked before any buffer lookup — see the `plan_dispatch_2d` call
    // moved ahead of pipeline/bind-group creation in `batched_gemm`) rather
    // than an aborted process (no uncaptured-error handler is installed for
    // wgpu validation failures) or a misleading "unknown handle" error.
    let err = b
        .batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            0,
            2,
            4,
            0,
            2,
            4,
            0.0,
            0,
            2,
            4,
            70_000, // batch_count > 65_535
        )
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)));
}

#[test]
fn batched_gemm_rejects_oversize_stride() {
    let Some(b) = try_init() else { return };
    // stride_a above u32::MAX must be a clean typed error, not a silent
    // wraparound into a small (wrong) stride that reads the wrong batch
    // slice.
    let oversize = (u32::MAX as usize) + 1;
    let err = b
        .batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            0,
            2,
            oversize,
            0,
            2,
            4,
            0.0,
            0,
            2,
            4,
            2,
        )
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)));
}

// ── reduce() buffer-size validation regressions ────────────────────────────

#[test]
fn reduce_1d_rejects_undersized_input_buffer() {
    let Some(b) = try_init() else { return };
    // Allocate an input buffer holding only 2 f32s but claim shape [8]: the
    // shader would otherwise read past the buffer (WGSL robust access
    // returns zeros / drops writes silently) instead of erroring.
    let in_h = b.alloc(2 * 4).expect("alloc undersized input");
    let out_h = b.alloc(4).expect("alloc output");
    let err = b.reduce(ReduceOp::Sum, in_h, out_h, &[8], 0).unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)));
    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

// Note: there is no `reduce_1d_rejects_undersized_output_buffer` test — the
// 1-D scalar reduction path needs exactly 4 bytes, and `alloc()` rounds every
// nonzero request up to `COPY_BUFFER_ALIGNMENT` (4) bytes, so a buffer this
// small cannot be constructed through the public API any more.  The `< 4`
// check in `reduce()`'s pass-2 bind group remains as defence in depth (e.g.
// against a future change to that rounding), covered indirectly by
// `reduce_nd_rejects_undersized_output_buffer` below, which needs 16 bytes
// and so stays genuinely reachable regardless of the 4-byte floor.

#[test]
fn reduce_nd_rejects_undersized_input_buffer() {
    let Some(b) = try_init() else { return };
    // shape [3, 4] needs 12 f32s of input; allocate only 4.
    let in_h = b.alloc(4 * 4).expect("alloc undersized input");
    let out_h = b.alloc(4 * 4).expect("alloc output");
    let err = b
        .reduce(ReduceOp::Sum, in_h, out_h, &[3, 4], 0)
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)));
    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn reduce_nd_rejects_undersized_output_buffer() {
    let Some(b) = try_init() else { return };
    // shape [3, 4], axis 0 -> output needs 4 f32s; allocate only 1.
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let in_h = upload_f32(&b, &data);
    let out_h = b.alloc(4).expect("alloc undersized output");
    let err = b
        .reduce(ReduceOp::Sum, in_h, out_h, &[3, 4], 0)
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)));
    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

// ── GPU-liveness witness ──────────────────────────────────────────────────

/// Proves the GPU-vs-CPU tests in this file are not passing vacuously.
///
/// `conv2d_forward` / `attention` fall back to `conv2d_cpu_reference` /
/// `attention_cpu_reference` when the dispatch grid exceeds
/// `max_workgroups_per_dim` — and those same functions are the *oracles* the
/// tests compare against. If a shape ever crossed that threshold the
/// comparison would silently become "CPU == CPU", which is trivially true.
///
/// This asserts the dispatch grid resolves to `Some` for the exact shapes
/// `conv2d_multi_channel_multi_filter_multi_batch`,
/// `conv2d_spans_multiple_workgroups`,
/// `conv2d_strided_with_padding_multi_channel` and
/// `attention_spans_multiple_workgroups` use, so each really dispatches a
/// kernel. Combined with `OXICUDA_REQUIRE_GPU=1` (which turns a missing
/// adapter into a failure), a green suite means kernels actually ran.
#[test]
fn gpu_device_is_live_when_required() {
    // (batch, k_out, oh, ow) for each conv2d case above, post-convolution.
    // 6x6 filter 3x3 stride 1 pad 0 -> 4x4; 22x22 -> 20x20; 9x9 s2 p1 -> 5x5.
    for (batch, k_out, oh, ow) in [
        (3usize, 4usize, 4usize, 4usize),
        (1, 5, 20, 20),
        (1, 3, 5, 5),
    ] {
        assert!(
            conv2d_gpu_dispatch_grid(batch, k_out, oh, ow).is_some(),
            "conv2d {batch}x{k_out}x{oh}x{ow} must dispatch on the GPU, not the CPU fallback"
        );
    }

    // `attention_spans_multiple_workgroups`: batch_heads = 2, seq_q = 40.
    assert!(
        attention_gpu_dispatch_grid(2, 40).is_some(),
        "attention 2x40 must dispatch on the GPU, not the CPU fallback"
    );

    let Some(b) = try_init() else {
        assert!(
            !require_gpu(),
            "OXICUDA_REQUIRE_GPU=1 but no WebGPU adapter is available"
        );
        return;
    };

    // A live adapter must report real limits — `max_workgroups_per_dim` is the
    // value the dispatch-grid decisions above are made against.
    let caps = b.capabilities();
    assert!(
        caps.max_threads_per_block > 0,
        "an initialised adapter must report a nonzero threads-per-workgroup limit"
    );
    let adapter_name = b
        .device
        .as_ref()
        .map(|d| d.adapter_name.clone())
        .unwrap_or_default();
    assert!(
        !adapter_name.is_empty(),
        "an initialised backend must hold an adapter with a real name"
    );
    println!(
        "WEBGPU WITNESS: adapter={adapter_name:?} max_threads={} max_workgroups_per_dim={}",
        caps.max_threads_per_block,
        gpu_limits().max_workgroups_per_dim
    );

    // NOTE: `WebGpuBackend` does not override `ComputeBackend::available_devices`,
    // so it inherits the trait default (an empty list) even when an adapter is
    // live. Deliberately not asserted here; reported as a gap.
}
