//! Cross-backend conformance for the ops that share `f32` row-major layout
//! identically across `oxicuda_backend::CpuBackend` and `MetalBackend`:
//! `unary`, `binary`, `reduce`, `conv2d_forward`, `attention`, and (Metal
//! only — `WebGpuBackend` has no override) `softmax`.
//!
//! Unlike GEMM (see `conformance_gemm.rs`), none of these ops have a
//! documented layout divergence between backends, so the same uploaded byte
//! buffer is fed to both `CpuBackend` and `MetalBackend` directly — no
//! logical/storage conversion is needed.
//!
//! Every GPU test degrades to a no-op skip when no Metal device is present
//! (headless CI, non-macOS), matching the crate's existing `try_init`
//! convention (see `src/backend/gpu_tests.rs`).

use oxicuda_backend::{BackendError, BinaryOp, ComputeBackend, CpuBackend, ReduceOp, UnaryOp};
use oxicuda_metal::MetalBackend;

// ─── Shared helpers ──────────────────────────────────────────────────────────

fn try_init() -> Option<MetalBackend> {
    let mut b = MetalBackend::new();
    b.init().ok().map(|()| b)
}

/// Deterministic pseudo-random `f32` values in `[lo, hi)`, no `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Full 32-bit range, normalized by the full 32-bit span — NOT a
        // narrower mask divided by a mismatched divisor, which silently
        // produces values orders of magnitude outside [0, 1) (caught by
        // `unary_ops_match_cpu_reference` sending `tanh` an argument large
        // enough that Metal's kernel saturated to NaN while `f32::tanh`
        // correctly saturated to 1.0 — a bug in this helper, not in Metal).
        let bits = (self.0 >> 32) as u32;
        (f64::from(bits) / f64::from(u32::MAX)) as f32 // [0, 1)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_unit() * (hi - lo)
    }
    fn vec(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len).map(|_| self.range(lo, hi)).collect()
    }
}

fn upload_f32<B: ComputeBackend>(be: &B, data: &[f32]) -> u64 {
    let ptr = be.alloc((data.len() * 4).max(4)).expect("alloc");
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    be.copy_htod(ptr, &bytes).expect("copy_htod");
    ptr
}
fn download_f32<B: ComputeBackend>(be: &B, ptr: u64, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * 4];
    be.copy_dtoh(&mut bytes, ptr).expect("copy_dtoh");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let limit = tol * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= limit,
            "{what}: element {i} = {g}, expected {w} (tolerance {limit})"
        );
    }
}

// ─── unary / binary ────────────────────────────────────────────────────────

#[test]
fn unary_ops_match_cpu_reference() {
    let Some(metal) = try_init() else { return };
    let cpu = CpuBackend::new();
    let n = 257; // deliberately not a power of two / not a multiple of any tile size.

    let ops: &[(UnaryOp, f32, f32)] = &[
        (UnaryOp::Relu, -3.0, 3.0),
        (UnaryOp::Sigmoid, -6.0, 6.0),
        (UnaryOp::Tanh, -3.0, 3.0),
        (UnaryOp::Exp, -4.0, 4.0),
        (UnaryOp::Log, 0.01, 10.0), // strictly positive domain
        (UnaryOp::Sqrt, 0.0, 10.0), // non-negative domain
        (UnaryOp::Abs, -5.0, 5.0),
        (UnaryOp::Neg, -5.0, 5.0),
    ];

    for &(op, lo, hi) in ops {
        let mut rng = Lcg::new(0x0000_1234 ^ (op as u64));
        let input = rng.vec(n, lo, hi);

        let cpu_in = upload_f32(&cpu, &input);
        let cpu_out = cpu.alloc(n * 4).expect("alloc");
        cpu.unary(op, cpu_in, cpu_out, n).expect("cpu unary");
        let want = download_f32(&cpu, cpu_out, n);

        let m_in = upload_f32(&metal, &input);
        let m_out = metal.alloc(n * 4).expect("alloc");
        metal.unary(op, m_in, m_out, n).expect("metal unary");
        let got = download_f32(&metal, m_out, n);

        assert_close(&got, &want, 1e-4, &format!("unary {op}"));
    }
}

#[test]
fn binary_ops_match_cpu_reference() {
    let Some(metal) = try_init() else { return };
    let cpu = CpuBackend::new();
    let n = 259;

    for &op in &[
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Max,
        BinaryOp::Min,
    ] {
        let mut rng = Lcg::new(0x0000_5678 ^ (op as u64));
        let a = rng.vec(n, -5.0, 5.0);
        // Keep `b` away from zero so `Div` stays finite and comparable.
        let b: Vec<f32> = (0..n)
            .map(|_| {
                let v = rng.range(0.5, 5.0);
                if rng.next_unit() < 0.5 { -v } else { v }
            })
            .collect();

        let cpu_a = upload_f32(&cpu, &a);
        let cpu_b = upload_f32(&cpu, &b);
        let cpu_out = cpu.alloc(n * 4).expect("alloc");
        cpu.binary(op, cpu_a, cpu_b, cpu_out, n)
            .expect("cpu binary");
        let want = download_f32(&cpu, cpu_out, n);

        let m_a = upload_f32(&metal, &a);
        let m_b = upload_f32(&metal, &b);
        let m_out = metal.alloc(n * 4).expect("alloc");
        metal.binary(op, m_a, m_b, m_out, n).expect("metal binary");
        let got = download_f32(&metal, m_out, n);

        assert_close(&got, &want, 1e-4, &format!("binary {op}"));
    }
}

// ─── reduce ────────────────────────────────────────────────────────────────

#[test]
fn reduce_matches_cpu_all_ops_various_shapes() {
    let Some(metal) = try_init() else { return };
    let cpu = CpuBackend::new();

    // (shape, axis) pairs: a 2-D shape reduced on each axis, a non-power-of-
    // two 1-D shape, and a size-1 shape (a degenerate but legal reduction).
    let cases: &[(&[usize], usize)] = &[
        (&[6, 7], 0),
        (&[6, 7], 1),
        (&[100], 0), // not a power of two
        (&[1], 0),
    ];

    for &op in &[ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::Mean] {
        for &(shape, axis) in cases {
            let total: usize = shape.iter().product();
            let outer: usize = shape[..axis].iter().product();
            let inner: usize = shape[axis + 1..].iter().product();
            let out_len = outer * inner;

            let mut rng = Lcg::new(0x0000_9ABC ^ (op as u64) ^ (total as u64) ^ (axis as u64));
            let input = rng.vec(total, -10.0, 10.0);

            let cpu_in = upload_f32(&cpu, &input);
            let cpu_out = cpu.alloc(out_len.max(1) * 4).expect("alloc");
            cpu.reduce(op, cpu_in, cpu_out, shape, axis)
                .expect("cpu reduce");
            let want = download_f32(&cpu, cpu_out, out_len);

            let m_in = upload_f32(&metal, &input);
            let m_out = metal.alloc(out_len.max(1) * 4).expect("alloc");
            metal
                .reduce(op, m_in, m_out, shape, axis)
                .expect("metal reduce");
            let got = download_f32(&metal, m_out, out_len);

            assert_close(
                &got,
                &want,
                1e-3,
                &format!("reduce {op} shape={shape:?} axis={axis}"),
            );
        }
    }
}

/// Dedicated `ReduceOp::Min` check with a curated (non-random) array so a
/// copy/paste bug that quietly turned `Min` into `Max` or `Sum` cannot hide
/// behind a coincidentally-matching random draw.
#[test]
fn reduce_min_finds_the_true_minimum() {
    let Some(metal) = try_init() else { return };
    let cpu = CpuBackend::new();

    // Row-major [2, 5]; the minimum of each row is placed at a different
    // column so a wrong axis/stride cannot accidentally look right.
    let data: [f32; 10] = [
        5.0, 3.0, -7.0, 9.0, 2.0, // row 0 min = -7 at col 2
        -1.0, -4.0, 0.5, -2.0, 8.0, // row 1 min = -4 at col 1
    ];
    let shape = [2usize, 5usize];

    let cpu_in = upload_f32(&cpu, &data);
    let cpu_out = cpu.alloc(2 * 4).expect("alloc");
    cpu.reduce(ReduceOp::Min, cpu_in, cpu_out, &shape, 1)
        .expect("cpu reduce min");
    let cpu_got = download_f32(&cpu, cpu_out, 2);
    assert_eq!(cpu_got, vec![-7.0, -4.0], "cpu oracle sanity check");

    let m_in = upload_f32(&metal, &data);
    let m_out = metal.alloc(2 * 4).expect("alloc");
    metal
        .reduce(ReduceOp::Min, m_in, m_out, &shape, 1)
        .expect("metal reduce min");
    let got = download_f32(&metal, m_out, 2);
    assert_close(&got, &cpu_got, 1e-5, "metal reduce Min vs cpu oracle");
}

/// A zero-length dimension in the reduced shape is rejected with
/// `InvalidArgument` on Metal — the documented behaviour after the fix that
/// closed the "reduce Mean over a zero-length axis silently yields NaN"
/// gap. Written against Metal's *own observed* behaviour rather than an
/// assumed shared cross-backend contract: `CpuBackend::reduce` performs no
/// such check at all (it returns the reduction identity for the op instead
/// of erroring), so this is deliberately not a cross-backend comparison.
#[test]
fn reduce_zero_length_dimension_is_rejected() {
    let Some(metal) = try_init() else { return };
    let ptr = metal.alloc(4).expect("alloc");
    let out = metal.alloc(4).expect("alloc");
    let result = metal.reduce(ReduceOp::Mean, ptr, out, &[0, 3], 0);
    assert!(
        matches!(result, Err(BackendError::InvalidArgument(_))),
        "expected InvalidArgument for a zero-length reduced dimension, got {result:?}"
    );
}

// ─── conv2d ────────────────────────────────────────────────────────────────

#[test]
fn conv2d_matches_cpu_reference() {
    let Some(metal) = try_init() else { return };
    let cpu = CpuBackend::new();

    let (n, c_in, h, w) = (1usize, 2usize, 5usize, 5usize);
    let (k_out, fh, fw) = (3usize, 3usize, 3usize);
    let (sh, sw) = (1usize, 1usize);
    let (ph, pw) = (1usize, 1usize);
    let oh = (h + 2 * ph - fh) / sh + 1;
    let ow = (w + 2 * pw - fw) / sw + 1;

    let mut rng = Lcg::new(0xC0432D);
    let input = rng.vec(n * c_in * h * w, -2.0, 2.0);
    let filter = rng.vec(k_out * c_in * fh * fw, -1.0, 1.0);

    let cpu_in = upload_f32(&cpu, &input);
    let cpu_filt = upload_f32(&cpu, &filter);
    let cpu_out = cpu.alloc(n * k_out * oh * ow * 4).expect("alloc");
    cpu.conv2d_forward(
        cpu_in,
        &[n, c_in, h, w],
        cpu_filt,
        &[k_out, c_in, fh, fw],
        cpu_out,
        &[n, k_out, oh, ow],
        &[sh, sw],
        &[ph, pw],
    )
    .expect("cpu conv2d");
    let want = download_f32(&cpu, cpu_out, n * k_out * oh * ow);

    let m_in = upload_f32(&metal, &input);
    let m_filt = upload_f32(&metal, &filter);
    let m_out = metal.alloc(n * k_out * oh * ow * 4).expect("alloc");
    metal
        .conv2d_forward(
            m_in,
            &[n, c_in, h, w],
            m_filt,
            &[k_out, c_in, fh, fw],
            m_out,
            &[n, k_out, oh, ow],
            &[sh, sw],
            &[ph, pw],
        )
        .expect("metal conv2d");
    let got = download_f32(&metal, m_out, n * k_out * oh * ow);

    assert_close(&got, &want, 1e-3, "conv2d_forward");
}

// ─── attention ─────────────────────────────────────────────────────────────

#[test]
fn attention_matches_cpu_reference_causal_and_noncausal() {
    let Some(metal) = try_init() else { return };
    let cpu = CpuBackend::new();

    let (batch, heads, seq, head_dim) = (1usize, 2usize, 4usize, 8usize);
    let scale = 1.0 / (head_dim as f64).sqrt();
    let qkv_elems = batch * heads * seq * head_dim;

    for causal in [false, true] {
        let mut rng = Lcg::new(0xA77E17 ^ u64::from(causal));
        let q = rng.vec(qkv_elems, -1.0, 1.0);
        let k = rng.vec(qkv_elems, -1.0, 1.0);
        let v = rng.vec(qkv_elems, -1.0, 1.0);

        let cpu_q = upload_f32(&cpu, &q);
        let cpu_k = upload_f32(&cpu, &k);
        let cpu_v = upload_f32(&cpu, &v);
        let cpu_o = cpu.alloc(qkv_elems * 4).expect("alloc");
        cpu.attention(
            cpu_q, cpu_k, cpu_v, cpu_o, batch, heads, seq, seq, head_dim, scale, causal,
        )
        .expect("cpu attention");
        let want = download_f32(&cpu, cpu_o, qkv_elems);

        let m_q = upload_f32(&metal, &q);
        let m_k = upload_f32(&metal, &k);
        let m_v = upload_f32(&metal, &v);
        let m_o = metal.alloc(qkv_elems * 4).expect("alloc");
        metal
            .attention(
                m_q, m_k, m_v, m_o, batch, heads, seq, seq, head_dim, scale, causal,
            )
            .expect("metal attention");
        let got = download_f32(&metal, m_o, qkv_elems);

        assert_close(&got, &want, 5e-3, &format!("attention causal={causal}"));
    }
}

// ─── softmax (Metal only — WebGpuBackend has no override) ──────────────────

#[test]
fn softmax_matches_cpu_reference_last_axis() {
    let Some(metal) = try_init() else { return };
    let cpu = CpuBackend::new();

    let shape = [3usize, 11usize]; // 11: not a power of two, exercises the
    // exact-for-any-width tree-reduction fix.
    let axis = 1;
    let total: usize = shape.iter().product();

    let mut rng = Lcg::new(0x50F7_A0AA);
    let input = rng.vec(total, -5.0, 5.0);

    let cpu_in = upload_f32(&cpu, &input);
    let cpu_out = cpu.alloc(total * 4).expect("alloc");
    cpu.softmax(cpu_in, cpu_out, &shape, axis)
        .expect("cpu softmax");
    let want = download_f32(&cpu, cpu_out, total);

    let m_in = upload_f32(&metal, &input);
    let m_out = metal.alloc(total * 4).expect("alloc");
    metal
        .softmax(m_in, m_out, &shape, axis)
        .expect("metal softmax");
    let got = download_f32(&metal, m_out, total);

    assert_close(&got, &want, 1e-4, "softmax last axis");
}
