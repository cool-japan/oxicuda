//! On-device numeric tests for the GPU paths wired up in [`super::nn`] and the
//! v2 GEMM dispatchers in [`super::types`].
//!
//! Every test here runs a real kernel on a real Metal device and compares the
//! result against a host oracle — the class of test the crate previously had
//! almost none of, and the only kind that can catch a mis-plumbed transpose
//! flag, a wrong leading dimension or a truncated accumulator.
//!
//! All tests degrade to a no-op when no Metal device is available (headless CI,
//! non-macOS), so the suite stays green everywhere. Set `OXICUDA_REQUIRE_GPU=1`
//! to forbid that degradation — see [`gpu_device_is_live_when_required`], which
//! also asserts that attention takes the GPU kernel rather than its host
//! fallback.

#![cfg(test)]

use oxicuda_backend::{BackendError, BackendTranspose, ComputeBackend};

use super::nn::{AttentionGeometry, attention_host};
use super::types::MetalBackend;

// ─── Shared helpers ──────────────────────────────────────────────────────────

/// Initialise a backend, or `None` when this machine has no Metal device.
///
/// Every GPU test in this module degrades to a no-op on `None` so the suite
/// stays green on headless CI. That portability has a sharp edge: if the device
/// silently stops initialising, *all* of these tests pass while executing
/// nothing. Set `OXICUDA_REQUIRE_GPU=1` to turn that silence into a failure —
/// see [`gpu_device_is_live_when_required`].
fn try_init() -> Option<MetalBackend> {
    let mut b = MetalBackend::new();
    let backend = b.init().ok().map(|()| b);
    if backend.is_none() && require_gpu() {
        panic!("OXICUDA_REQUIRE_GPU=1 but no Metal device initialised");
    }
    backend
}

/// Whether the caller demands that the GPU tests really run on a device.
fn require_gpu() -> bool {
    std::env::var("OXICUDA_REQUIRE_GPU").is_ok_and(|v| v == "1")
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic pseudo-random values in `[-2, 2)`, no `rand` dependency.
///
/// Used to fill **padding** as well as live elements: a kernel that reads across
/// a row boundary then picks up a distinct nonzero value and the comparison
/// fails, which an all-zero padding would hide.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 33) as u32 % 4001;
        (bits as f32 - 2000.0) / 1000.0
    }
    fn vec(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next_f32()).collect()
    }
}

/// Allocate a device buffer holding `data`, or `None` if allocation fails.
fn upload(backend: &MetalBackend, data: &[f32]) -> Option<u64> {
    let bytes = f32_to_bytes(data);
    let handle = backend.alloc(bytes.len()).ok()?;
    backend.copy_htod(handle, &bytes).ok()?;
    Some(handle)
}

/// Read `len` `f32` elements back from a device buffer.
fn download(backend: &MetalBackend, handle: u64, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * 4];
    backend.copy_dtoh(&mut bytes, handle).expect("device→host");
    bytes_to_f32(&bytes)
}

/// Assert `got ≈ want` element-wise with a mixed absolute/relative tolerance.
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

// ─── GEMM: transpose modes and padded leading dimensions ─────────────────────

/// Host oracle for the row-major GEMM contract the v2 kernels implement.
///
/// Mirrors [`crate::msl::gemm_msl_v2`]'s addressing table exactly and
/// accumulates in `f64` so the comparison measures the GPU's error, not the
/// oracle's.
#[allow(clippy::too_many_arguments)]
fn gemm_oracle(
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    let a_at = |r: usize, i: usize| -> f64 {
        let idx = if trans_a == BackendTranspose::NoTrans {
            r * lda + i
        } else {
            i * lda + r
        };
        f64::from(a[idx])
    };
    let b_at = |i: usize, col: usize| -> f64 {
        let idx = if trans_b == BackendTranspose::NoTrans {
            i * ldb + col
        } else {
            col * ldb + i
        };
        f64::from(b[idx])
    };
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += a_at(r, i) * b_at(i, col);
            }
            let dst = &mut c[r * ldc + col];
            *dst = (f64::from(alpha) * acc + f64::from(beta) * f64::from(*dst)) as f32;
        }
    }
}

/// The headline P2 test: all four transpose combinations, at dimensions that
/// are **not** multiples of the 32×32 output tile, with every leading dimension
/// **padded** beyond its packed minimum.
///
/// A kernel that ignores `trans_*` fails here; one that assumes `lda == k`
/// fails here; one that only guards the tile origin (rather than every store)
/// corrupts `C`'s padding columns and fails here. A test with `lda == k` and
/// `NoTrans` passes under both the old and the new indexing and would prove
/// nothing.
#[test]
fn gemm_matches_the_oracle_for_every_transpose_with_padded_leading_dims() {
    let Some(backend) = try_init() else { return };
    const M: usize = 17;
    const N: usize = 13;
    const K: usize = 11;
    let alpha = 1.5f32;
    let beta = 0.75f32;

    for trans_a in [BackendTranspose::NoTrans, BackendTranspose::Trans] {
        for trans_b in [BackendTranspose::NoTrans, BackendTranspose::Trans] {
            let (min_lda, min_ldb, min_ldc) =
                super::types::packed_gemm_lds(trans_a, trans_b, M, N, K);
            let (lda, ldb, ldc) = (min_lda + 3, min_ldb + 5, min_ldc + 2);
            let a_rows = if trans_a == BackendTranspose::NoTrans {
                M
            } else {
                K
            };
            let b_rows = if trans_b == BackendTranspose::NoTrans {
                K
            } else {
                N
            };

            let mut rng = Lcg::new(0xC0FFEE ^ (u64::from(trans_a == BackendTranspose::Trans) << 1));
            let a = rng.vec(a_rows * lda);
            let b = rng.vec(b_rows * ldb);
            let c_init = rng.vec(M * ldc);

            let mut want = c_init.clone();
            gemm_oracle(
                trans_a, trans_b, M, N, K, alpha, &a, lda, &b, ldb, beta, &mut want, ldc,
            );

            let (Some(ah), Some(bh), Some(ch)) = (
                upload(&backend, &a),
                upload(&backend, &b),
                upload(&backend, &c_init),
            ) else {
                return;
            };
            backend
                .gemm(
                    trans_a,
                    trans_b,
                    M,
                    N,
                    K,
                    f64::from(alpha),
                    ah,
                    lda,
                    bh,
                    ldb,
                    f64::from(beta),
                    ch,
                    ldc,
                )
                .expect("gemm must accept every transpose combination");
            let got = download(&backend, ch, M * ldc);

            // Only the logical m×n block is defined; the padding columns of C
            // must be left exactly as they were.
            for r in 0..M {
                for col in 0..N {
                    let idx = r * ldc + col;
                    let limit = 1e-4 * want[idx].abs().max(1.0);
                    assert!(
                        (got[idx] - want[idx]).abs() <= limit,
                        "{trans_a:?}/{trans_b:?} C[{r},{col}] = {}, expected {}",
                        got[idx],
                        want[idx]
                    );
                }
                for col in N..ldc {
                    let idx = r * ldc + col;
                    assert_eq!(
                        got[idx], c_init[idx],
                        "{trans_a:?}/{trans_b:?} wrote into C's padding at row {r}, column {col}"
                    );
                }
            }
            for h in [ah, bh, ch] {
                backend.free(h).expect("free");
            }
        }
    }
}

/// `beta == 0` must ignore `C`'s previous contents entirely, even when they are
/// non-finite — the kernel guards the read rather than multiplying by zero.
#[test]
fn gemm_with_beta_zero_never_reads_c() {
    let Some(backend) = try_init() else { return };
    let (m, n, k) = (4usize, 4usize, 4usize);
    let mut rng = Lcg::new(7);
    let a = rng.vec(m * k);
    let b = rng.vec(k * n);
    let c_init = vec![f32::NAN; m * n];
    let mut want = vec![0.0f32; m * n];
    gemm_oracle(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        m,
        n,
        k,
        1.0,
        &a,
        k,
        &b,
        n,
        0.0,
        &mut want,
        n,
    );
    let (Some(ah), Some(bh), Some(ch)) = (
        upload(&backend, &a),
        upload(&backend, &b),
        upload(&backend, &c_init),
    ) else {
        return;
    };
    backend
        .gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            m,
            n,
            k,
            1.0,
            ah,
            k,
            bh,
            n,
            0.0,
            ch,
            n,
        )
        .expect("gemm");
    let got = download(&backend, ch, m * n);
    assert_close(&got, &want, 1e-4, "beta=0 gemm");
    for h in [ah, bh, ch] {
        backend.free(h).expect("free");
    }
}

/// Batched GEMM with a transpose and independent per-batch strides.
#[test]
fn batched_gemm_matches_the_oracle_with_transpose_and_strides() {
    let Some(backend) = try_init() else { return };
    const M: usize = 9;
    const N: usize = 7;
    const K: usize = 5;
    const BATCH: usize = 3;
    let trans_a = BackendTranspose::NoTrans;
    let trans_b = BackendTranspose::Trans;
    let (lda, ldb, ldc) = super::types::packed_gemm_lds(trans_a, trans_b, M, N, K);
    // Strides deliberately exceed the matrix size so a dispatcher that ignored
    // them (or reused the packed size) reads the wrong batch element.
    let stride_a = M * lda + 4;
    let stride_b = N * ldb + 6;
    let stride_c = M * ldc + 2;

    let mut rng = Lcg::new(31_337);
    let a = rng.vec(stride_a * BATCH);
    let b = rng.vec(stride_b * BATCH);
    let c_init = rng.vec(stride_c * BATCH);

    let mut want = c_init.clone();
    for batch in 0..BATCH {
        let mut block = want[batch * stride_c..batch * stride_c + M * ldc].to_vec();
        gemm_oracle(
            trans_a,
            trans_b,
            M,
            N,
            K,
            2.0,
            &a[batch * stride_a..],
            lda,
            &b[batch * stride_b..],
            ldb,
            -0.5,
            &mut block,
            ldc,
        );
        want[batch * stride_c..batch * stride_c + M * ldc].copy_from_slice(&block);
    }

    let (Some(ah), Some(bh), Some(ch)) = (
        upload(&backend, &a),
        upload(&backend, &b),
        upload(&backend, &c_init),
    ) else {
        return;
    };
    backend
        .batched_gemm(
            trans_a, trans_b, M, N, K, 2.0, ah, lda, stride_a, bh, ldb, stride_b, -0.5, ch, ldc,
            stride_c, BATCH,
        )
        .expect("batched_gemm");
    let got = download(&backend, ch, stride_c * BATCH);
    for batch in 0..BATCH {
        for r in 0..M {
            for col in 0..N {
                let idx = batch * stride_c + r * ldc + col;
                let limit = 1e-4 * want[idx].abs().max(1.0);
                assert!(
                    (got[idx] - want[idx]).abs() <= limit,
                    "batch {batch} C[{r},{col}] = {}, expected {}",
                    got[idx],
                    want[idx]
                );
            }
        }
    }
    for h in [ah, bh, ch] {
        backend.free(h).expect("free");
    }
}

/// One compiled pipeline must serve every GEMM shape: the v2 kernels take their
/// dimensions from a parameter buffer, so shape must not enter the cache key.
#[test]
fn gemm_serves_every_shape_from_a_single_pipeline() {
    let Some(backend) = try_init() else { return };
    let mut rng = Lcg::new(11);
    for (m, n, k) in [(4usize, 4usize, 4usize), (7, 3, 5), (33, 31, 17)] {
        let a = rng.vec(m * k);
        let b = rng.vec(k * n);
        let c = vec![0.0f32; m * n];
        let (Some(ah), Some(bh), Some(ch)) = (
            upload(&backend, &a),
            upload(&backend, &b),
            upload(&backend, &c),
        ) else {
            return;
        };
        backend
            .gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                m,
                n,
                k,
                1.0,
                ah,
                k,
                bh,
                n,
                0.0,
                ch,
                n,
            )
            .expect("gemm");
        for h in [ah, bh, ch] {
            backend.free(h).expect("free");
        }
    }
    assert_eq!(
        backend.pipeline_cache_len(),
        1,
        "three GEMM shapes must share one compiled pipeline"
    );
}

// ─── Conv2D ──────────────────────────────────────────────────────────────────

/// Host oracle for `conv2d_forward`, structurally identical to the loop this
/// backend used to run on the CPU.
#[allow(clippy::too_many_arguments)]
fn conv2d_oracle(
    input: &[f32],
    filter: &[f32],
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    k_out: usize,
    fh: usize,
    fw: usize,
    oh: usize,
    ow: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n * k_out * oh * ow];
    for b in 0..n {
        for kf in 0..k_out {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0.0f32;
                    for ci in 0..c_in {
                        for fy in 0..fh {
                            for fx in 0..fw {
                                let iy = (oy * stride_h + fy) as isize - pad_h as isize;
                                let ix = (ox * stride_w + fx) as isize - pad_w as isize;
                                if iy >= 0
                                    && (iy as usize) < h_in
                                    && ix >= 0
                                    && (ix as usize) < w_in
                                {
                                    let iy = iy as usize;
                                    let ix = ix as usize;
                                    acc += input[((b * c_in + ci) * h_in + iy) * w_in + ix]
                                        * filter[((kf * c_in + ci) * fh + fy) * fw + fx];
                                }
                            }
                        }
                    }
                    out[((b * k_out + kf) * oh + oy) * ow + ox] = acc;
                }
            }
        }
    }
    out
}

/// Multi-batch, multi-channel convolution with a stride of 2 and asymmetric
/// padding, at extents that are not multiples of anything convenient.
#[test]
fn conv2d_stride2_padded_matches_the_host_oracle() {
    let Some(backend) = try_init() else { return };
    let (n, c_in, h_in, w_in) = (2usize, 3usize, 7usize, 5usize);
    let (k_out, fh, fw) = (4usize, 3usize, 3usize);
    let (stride_h, stride_w, pad_h, pad_w) = (2usize, 2usize, 1usize, 1usize);
    let oh = (h_in + 2 * pad_h - fh) / stride_h + 1;
    let ow = (w_in + 2 * pad_w - fw) / stride_w + 1;

    let mut rng = Lcg::new(0xABCD);
    let input = rng.vec(n * c_in * h_in * w_in);
    let filter = rng.vec(k_out * c_in * fh * fw);
    let want = conv2d_oracle(
        &input, &filter, n, c_in, h_in, w_in, k_out, fh, fw, oh, ow, stride_h, stride_w, pad_h,
        pad_w,
    );

    let (Some(ih), Some(fh_h), Some(oh_h)) = (
        upload(&backend, &input),
        upload(&backend, &filter),
        upload(&backend, &vec![0.0f32; want.len()]),
    ) else {
        return;
    };
    backend
        .conv2d_forward(
            ih,
            &[n, c_in, h_in, w_in],
            fh_h,
            &[k_out, c_in, fh, fw],
            oh_h,
            &[n, k_out, oh, ow],
            &[stride_h, stride_w],
            &[pad_h, pad_w],
        )
        .expect("conv2d");
    let got = download(&backend, oh_h, want.len());
    assert_close(&got, &want, 1e-4, "conv2d stride=2 pad=1");
    for h in [ih, fh_h, oh_h] {
        backend.free(h).expect("free");
    }
}

/// A zero stride used to compile into a kernel that read the same window for
/// every output element; it must be rejected on the host instead.
#[test]
fn conv2d_rejects_a_zero_stride() {
    let Some(backend) = try_init() else { return };
    assert!(matches!(
        backend.conv2d_forward(
            0,
            &[1, 1, 3, 3],
            0,
            &[1, 1, 1, 1],
            0,
            &[1, 1, 3, 3],
            &[0, 1],
            &[0, 0]
        ),
        Err(BackendError::InvalidArgument(_))
    ));
}

/// Two different convolution shapes must share one compiled pipeline — the
/// whole point of moving the shapes into a parameter buffer. With the old
/// shape-baking generator this would be two entries and would churn the cache.
#[test]
fn conv2d_serves_every_shape_from_a_single_pipeline() {
    let Some(backend) = try_init() else { return };
    for (h_in, w_in) in [(4usize, 4usize), (6, 5)] {
        let input = vec![1.0f32; h_in * w_in];
        let filter = vec![1.0f32; 9];
        let oh = h_in - 2;
        let ow = w_in - 2;
        let (Some(ih), Some(fh_h), Some(oh_h)) = (
            upload(&backend, &input),
            upload(&backend, &filter),
            upload(&backend, &vec![0.0f32; oh * ow]),
        ) else {
            return;
        };
        backend
            .conv2d_forward(
                ih,
                &[1, 1, h_in, w_in],
                fh_h,
                &[1, 1, 3, 3],
                oh_h,
                &[1, 1, oh, ow],
                &[1, 1],
                &[0, 0],
            )
            .expect("conv2d");
        for h in [ih, fh_h, oh_h] {
            backend.free(h).expect("free");
        }
    }
    assert_eq!(
        backend.pipeline_cache_len(),
        1,
        "two convolution shapes must share one compiled pipeline"
    );
}

// ─── Attention ───────────────────────────────────────────────────────────────

/// The GPU online-softmax kernel against the host two-pass reference, at a
/// non-power-of-two `seq_kv`, `seq_q != seq_kv`, and a `head_dim` that is not a
/// multiple of the 32-lane SIMD width — so the strided per-lane loops, the
/// running-max rescaling and the causal bound are all exercised.
#[test]
fn attention_matches_the_host_oracle_dense_and_causal() {
    let Some(backend) = try_init() else { return };
    let geom_base = AttentionGeometry {
        batch_heads: 2,
        seq_q: 5,
        seq_kv: 257,
        head_dim: 40,
        scale: 0.125,
        causal: false,
    };
    // Guard against a vacuous comparison: `dispatch_attention` falls back to
    // `attention_host` when no SIMD-group count fits, and then "GPU == host"
    // would be trivially true. Assert the GPU path is the one under test.
    // (1024 is every current Apple GPU's `maxTotalThreadsPerThreadgroup`; the
    // binding constraint here is threadgroup memory, which is device-probed.)
    assert!(
        backend
            .attention_simdgroups(geom_base.head_dim, 1024)
            .is_some(),
        "this head_dim must run on the GPU, not the host fallback"
    );
    let mut rng = Lcg::new(0x5EED);
    let q = rng.vec(geom_base.batch_heads * geom_base.seq_q * geom_base.head_dim);
    let kv_len = geom_base.batch_heads * geom_base.seq_kv * geom_base.head_dim;
    let k = rng.vec(kv_len);
    let v = rng.vec(kv_len);
    let o_len = q.len();

    for causal in [false, true] {
        let geom = AttentionGeometry {
            causal,
            ..geom_base
        };
        let (Some(qh), Some(kh), Some(vh), Some(gpu_o), Some(host_o)) = (
            upload(&backend, &q),
            upload(&backend, &k),
            upload(&backend, &v),
            upload(&backend, &vec![0.0f32; o_len]),
            upload(&backend, &vec![0.0f32; o_len]),
        ) else {
            return;
        };
        backend
            .attention(
                qh,
                kh,
                vh,
                gpu_o,
                geom.batch_heads,
                1,
                geom.seq_q,
                geom.seq_kv,
                geom.head_dim,
                geom.scale,
                causal,
            )
            .expect("attention");
        attention_host(&backend, qh, kh, vh, host_o, geom).expect("host attention");
        let got = download(&backend, gpu_o, o_len);
        let want = download(&backend, host_o, o_len);
        assert_close(&got, &want, 1e-3, if causal { "causal" } else { "dense" });
        for h in [qh, kh, vh, gpu_o, host_o] {
            backend.free(h).expect("free");
        }
    }
}

/// Causal masking must actually mask: with `seq_q == seq_kv` the first query
/// attends to exactly one key, so its output equals `V[0]` regardless of the
/// other keys' values.
#[test]
fn attention_causal_first_query_sees_only_the_first_value() {
    let Some(backend) = try_init() else { return };
    let (seq, head_dim) = (4usize, 2usize);
    let q = vec![1.0f32; seq * head_dim];
    let k = vec![1.0f32; seq * head_dim];
    let mut v = vec![0.0f32; seq * head_dim];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = (i + 1) as f32;
    }
    let (Some(qh), Some(kh), Some(vh), Some(oh)) = (
        upload(&backend, &q),
        upload(&backend, &k),
        upload(&backend, &v),
        upload(&backend, &vec![0.0f32; seq * head_dim]),
    ) else {
        return;
    };
    backend
        .attention(qh, kh, vh, oh, 1, 1, seq, seq, head_dim, 1.0, true)
        .expect("attention");
    let got = download(&backend, oh, seq * head_dim);
    assert_close(&got[..head_dim], &v[..head_dim], 1e-5, "causal row 0");
    for h in [qh, kh, vh, oh] {
        backend.free(h).expect("free");
    }
}

/// Attention shapes live in a parameter buffer, so two different sequence
/// lengths must share one pipeline.
#[test]
fn attention_serves_every_shape_from_a_single_pipeline() {
    let Some(backend) = try_init() else { return };
    for seq in [2usize, 3] {
        let len = seq * 4;
        let (Some(qh), Some(kh), Some(vh), Some(oh)) = (
            upload(&backend, &vec![0.5f32; len]),
            upload(&backend, &vec![0.5f32; len]),
            upload(&backend, &vec![0.5f32; len]),
            upload(&backend, &vec![0.0f32; len]),
        ) else {
            return;
        };
        backend
            .attention(qh, kh, vh, oh, 1, 1, seq, seq, 4, 0.5, false)
            .expect("attention");
        for h in [qh, kh, vh, oh] {
            backend.free(h).expect("free");
        }
    }
    assert_eq!(
        backend.pipeline_cache_len(),
        1,
        "two attention shapes must share one compiled pipeline"
    );
}

// ─── Softmax ─────────────────────────────────────────────────────────────────

fn softmax_oracle(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Row-wise softmax at a non-power-of-two width, including a row whose dynamic
/// range would overflow `exp` without the max subtraction.
#[test]
fn softmax_matches_a_stable_cpu_oracle() {
    let Some(backend) = try_init() else { return };
    let (rows, cols) = (4usize, 37usize);
    let mut rng = Lcg::new(0xF00D);
    let mut input = rng.vec(rows * cols);
    // A row that only a numerically-stable implementation survives.
    input[cols] = 1e30;
    input[cols + 1] = -1e30;

    let mut want = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        want.extend(softmax_oracle(&input[r * cols..(r + 1) * cols]));
    }

    let (Some(ih), Some(oh)) = (
        upload(&backend, &input),
        upload(&backend, &vec![0.0f32; rows * cols]),
    ) else {
        return;
    };
    backend
        .softmax(ih, oh, &[rows, cols], 1)
        .expect("softmax must be supported on the last axis");
    let got = download(&backend, oh, rows * cols);
    assert_close(&got, &want, 1e-5, "softmax");
    // Every row must still sum to one.
    for r in 0..rows {
        let sum: f32 = got[r * cols..(r + 1) * cols].iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {r} sums to {sum}");
    }
    for h in [ih, oh] {
        backend.free(h).expect("free");
    }
}

/// A 1-D tensor is a single row; `axis = 0` is its last axis.
#[test]
fn softmax_handles_a_one_dimensional_tensor() {
    let Some(backend) = try_init() else { return };
    let input = vec![1.0f32, 2.0, 3.0];
    let want = softmax_oracle(&input);
    let (Some(ih), Some(oh)) = (
        upload(&backend, &input),
        upload(&backend, &vec![0.0f32; input.len()]),
    ) else {
        return;
    };
    backend.softmax(ih, oh, &[3], 0).expect("1-D softmax");
    let got = download(&backend, oh, input.len());
    assert_close(&got, &want, 1e-6, "1-D softmax");
    for h in [ih, oh] {
        backend.free(h).expect("free");
    }
}

/// The kernel is row-contiguous, so a non-final axis must be refused rather
/// than silently reinterpreted as a different reduction.
#[test]
fn softmax_rejects_a_non_final_axis_and_bad_shapes() {
    let Some(backend) = try_init() else { return };
    assert!(matches!(
        backend.softmax(0, 0, &[3, 4], 0),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        backend.softmax(0, 0, &[3, 4], 2),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        backend.softmax(0, 0, &[], 0),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        backend.softmax(0, 0, &[3, 0], 1),
        Err(BackendError::InvalidArgument(_))
    ));
}

// ─── Layer normalisation ─────────────────────────────────────────────────────

#[test]
fn layer_norm_matches_a_cpu_oracle() {
    let Some(backend) = try_init() else { return };
    let (rows, cols) = (3usize, 19usize);
    let eps = 1e-5f32;
    let mut rng = Lcg::new(0xBEEF);
    let input = rng.vec(rows * cols);
    let gamma = rng.vec(cols);
    let beta = rng.vec(cols);

    let mut want = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / cols as f64;
        let var = row
            .iter()
            .map(|&x| (f64::from(x) - mean) * (f64::from(x) - mean))
            .sum::<f64>()
            / cols as f64;
        let inv_std = 1.0 / (var + f64::from(eps)).sqrt();
        for c in 0..cols {
            want[r * cols + c] = (((f64::from(row[c]) - mean) * inv_std) * f64::from(gamma[c])
                + f64::from(beta[c])) as f32;
        }
    }

    let (Some(ih), Some(gh), Some(bh), Some(oh)) = (
        upload(&backend, &input),
        upload(&backend, &gamma),
        upload(&backend, &beta),
        upload(&backend, &vec![0.0f32; rows * cols]),
    ) else {
        return;
    };
    backend
        .layer_norm(ih, gh, bh, oh, rows, cols, eps)
        .expect("layer_norm");
    let got = download(&backend, oh, rows * cols);
    assert_close(&got, &want, 1e-4, "layer_norm");
    for h in [ih, gh, bh, oh] {
        backend.free(h).expect("free");
    }
}

#[test]
fn layer_norm_rejects_degenerate_arguments() {
    let Some(backend) = try_init() else { return };
    assert!(matches!(
        backend.layer_norm(0, 0, 0, 0, 2, 0, 1e-5),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        backend.layer_norm(0, 0, 0, 0, 2, 4, -1.0),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        backend.layer_norm(0, 0, 0, 0, 2, 4, f32::NAN),
        Err(BackendError::InvalidArgument(_))
    ));
    // Zero rows is a legitimate no-op, not an error.
    assert!(backend.layer_norm(0, 0, 0, 0, 0, 4, 1e-5).is_ok());
}

// ─── Scan ────────────────────────────────────────────────────────────────────

#[test]
fn scan_inclusive_and_exclusive_match_a_cpu_oracle() {
    let Some(backend) = try_init() else { return };
    let n = 1000usize;
    let mut rng = Lcg::new(0x1234_5678);
    let input = rng.vec(n);

    for exclusive in [false, true] {
        let mut want = Vec::with_capacity(n);
        let mut acc = 0.0f32;
        for &x in &input {
            if exclusive {
                want.push(acc);
                acc += x;
            } else {
                acc += x;
                want.push(acc);
            }
        }
        let (Some(ih), Some(oh)) = (upload(&backend, &input), upload(&backend, &vec![0.0f32; n]))
        else {
            return;
        };
        backend.scan(ih, oh, n, exclusive).expect("scan");
        let got = download(&backend, oh, n);
        // The Hillis-Steele tree associates differently from the sequential
        // oracle, so compare with a tolerance that scales with the running sum.
        assert_close(
            &got,
            &want,
            1e-4,
            if exclusive { "exclusive" } else { "inclusive" },
        );
        for h in [ih, oh] {
            backend.free(h).expect("free");
        }
    }
}

/// The single-threadgroup scan silently truncated anything past `tg_size`; it
/// must now refuse the input instead.
#[test]
fn scan_rejects_more_than_one_threadgroup_of_elements() {
    let Some(backend) = try_init() else { return };
    let Some(ih) = upload(&backend, &[1.0f32; 8]) else {
        return;
    };
    assert!(matches!(
        backend.scan(ih, ih, 100_000, false),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(backend.scan(ih, ih, 0, false).is_ok());
    backend.free(ih).expect("free");
}

// ─── Capabilities and device enumeration ─────────────────────────────────────

/// Before `init` there is no device to interrogate, so the conservative default
/// profile is the honest answer — never a fabricated GPU profile.
#[test]
fn capabilities_before_init_are_the_conservative_default() {
    let backend = MetalBackend::new();
    assert_eq!(
        backend.capabilities(),
        oxicuda_backend::Capabilities::default()
    );
    assert_eq!(backend.available_devices().expect("no error"), Vec::new());
}

#[test]
fn capabilities_report_the_probed_device() {
    let Some(backend) = try_init() else { return };
    let caps = backend.capabilities();
    let device = backend.device.as_ref().expect("initialised");
    let probed = device.capabilities();

    assert!(caps.supports_fp16, "MSL `half` is always available");
    assert!(!caps.supports_bf16, "no bf16 kernel ships in this crate");
    assert!(!caps.supports_fp8);
    assert!(!caps.peer_access);
    assert!(!caps.cluster_launch);
    assert!(!caps.async_copy);
    assert_eq!(caps.tensor_cores, probed.simdgroup_matrix);
    assert_eq!(caps.unified_memory, probed.unified_memory);
    assert_eq!(
        caps.max_threads_per_block as usize,
        probed.max_threads_per_threadgroup
    );
    assert_eq!(
        caps.max_shared_mem_per_block as usize,
        probed.threadgroup_memory
    );
    assert_eq!(caps.warp_size, 32);
    // A real GPU must not be reported with the CPU profile.
    assert_ne!(caps, oxicuda_backend::Capabilities::default());
}

#[test]
fn available_devices_describes_the_metal_gpu() {
    let Some(backend) = try_init() else { return };
    let devices = backend.available_devices().expect("enumeration");
    assert_eq!(devices.len(), 1, "this backend drives one device");
    let info = &devices[0];
    let device = backend.device.as_ref().expect("initialised");
    assert_eq!(info.ordinal, 0);
    assert_eq!(info.name, device.name());
    assert!(!info.name.is_empty());
    assert_eq!(info.total_memory_bytes, device.max_buffer_length());
    assert!(info.total_memory_bytes > 0);
    assert_eq!(
        info.compute_capability,
        (device.capabilities().family.generation(), 0)
    );
    assert_eq!(info.capabilities, backend.capabilities());
    // `Display` must not panic on the real values.
    assert!(format!("{info}").contains(&info.name));
}

// ─── Asynchronous dispatch (opt-in) ──────────────────────────────────────────

/// Synchronous dispatch is the default, and it must leave nothing in flight.
#[test]
fn dispatch_is_synchronous_by_default() {
    let Some(backend) = try_init() else { return };
    assert!(!backend.async_dispatch(), "async dispatch must be opt-in");
    let values = [-1.0f32, 0.0, 2.0, -3.5];
    let (Some(ih), Some(oh)) = (
        upload(&backend, &values),
        upload(&backend, &vec![0.0f32; values.len()]),
    ) else {
        return;
    };
    backend
        .unary(oxicuda_backend::UnaryOp::Relu, ih, oh, values.len())
        .expect("relu");
    assert_eq!(
        backend.inflight_count(),
        0,
        "a synchronous dispatch has already been awaited when it returns"
    );
    for h in [ih, oh] {
        backend.free(h).expect("free");
    }
}

/// With async dispatch on, work accumulates until a synchronisation point —
/// and `synchronize` is one.
#[test]
fn async_dispatch_queues_work_until_synchronize() {
    let Some(backend) = try_init() else { return };
    let values = vec![1.0f32; 64];
    let (Some(ih), Some(oh)) = (
        upload(&backend, &values),
        upload(&backend, &vec![0.0f32; values.len()]),
    ) else {
        return;
    };
    backend.set_async_dispatch(true).expect("enable async");
    assert!(backend.async_dispatch());
    for _ in 0..3 {
        backend
            .unary(oxicuda_backend::UnaryOp::Relu, ih, oh, values.len())
            .expect("relu");
    }
    assert_eq!(
        backend.inflight_count(),
        3,
        "async dispatches must be tracked, not awaited"
    );
    backend.synchronize().expect("synchronize");
    assert_eq!(backend.inflight_count(), 0, "synchronize drains the queue");

    // Turning async off is itself a flush.
    backend
        .unary(oxicuda_backend::UnaryOp::Relu, ih, oh, values.len())
        .expect("relu");
    assert_eq!(backend.inflight_count(), 1);
    backend.set_async_dispatch(false).expect("disable async");
    assert_eq!(backend.inflight_count(), 0);
    for h in [ih, oh] {
        backend.free(h).expect("free");
    }
}

/// Every path that can *observe* a buffer must synchronise first, so async mode
/// is numerically indistinguishable from synchronous mode.
#[test]
fn async_dispatch_gives_identical_results_and_reads_synchronize() {
    let Some(backend) = try_init() else { return };
    let mut rng = Lcg::new(0xA5A5);
    let a = rng.vec(1024);
    let b = rng.vec(1024);
    let want: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x + y).max(0.0))
        .collect();

    backend.set_async_dispatch(true).expect("enable async");
    let (Some(ah), Some(bh), Some(sum), Some(out)) = (
        upload(&backend, &a),
        upload(&backend, &b),
        upload(&backend, &vec![0.0f32; a.len()]),
        upload(&backend, &vec![0.0f32; a.len()]),
    ) else {
        return;
    };
    backend
        .binary(oxicuda_backend::BinaryOp::Add, ah, bh, sum, a.len())
        .expect("add");
    backend
        .unary(oxicuda_backend::UnaryOp::Relu, sum, out, a.len())
        .expect("relu");
    // No explicit synchronize: the read-back must do it.
    let got = download(&backend, out, a.len());
    assert_eq!(
        backend.inflight_count(),
        0,
        "copy_dtoh must be a synchronisation point"
    );
    assert_close(&got, &want, 1e-6, "async add+relu chain");

    backend.set_async_dispatch(false).expect("disable async");
    for h in [ah, bh, sum, out] {
        backend.free(h).expect("free");
    }
}

/// `free` must synchronise: the allocator's reuse pool can hand the very same
/// buffer to the next `alloc`, so releasing one still referenced by a running
/// kernel would let two logical allocations alias.
#[test]
fn free_synchronizes_before_the_buffer_can_be_reused() {
    let Some(backend) = try_init() else { return };
    let values = vec![2.0f32; 256];
    let (Some(ih), Some(oh)) = (
        upload(&backend, &values),
        upload(&backend, &vec![0.0f32; values.len()]),
    ) else {
        return;
    };
    backend.set_async_dispatch(true).expect("enable async");
    backend
        .unary(oxicuda_backend::UnaryOp::Relu, ih, oh, values.len())
        .expect("relu");
    assert_eq!(backend.inflight_count(), 1);
    backend.free(oh).expect("free");
    assert_eq!(
        backend.inflight_count(),
        0,
        "free must be a synchronisation point"
    );
    backend.set_async_dispatch(false).expect("disable async");
    backend.free(ih).expect("free");
}

/// The two-pass flat reduction allocates a scratch buffer, encodes into it and
/// frees it through the memory manager directly — bypassing
/// `ComputeBackend::free`'s synchronisation point. In async mode it must
/// therefore await its own work before releasing the scratch, or the allocator's
/// reuse pool can hand a still-in-use buffer to the next `alloc`.
#[test]
fn async_two_pass_reduction_is_correct_and_leaves_nothing_in_flight() {
    let Some(backend) = try_init() else { return };
    // Above TWO_PASS_REDUCE_THRESHOLD (4096) so the chunked path is taken.
    let n = 10_000usize;
    let input: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let want: f32 = input.iter().sum();
    let (Some(ih), Some(oh)) = (upload(&backend, &input), upload(&backend, &[0.0f32])) else {
        return;
    };
    backend.set_async_dispatch(true).expect("enable async");
    backend
        .reduce(oxicuda_backend::ReduceOp::Sum, ih, oh, &[n], 0)
        .expect("reduce");
    assert_eq!(
        backend.inflight_count(),
        0,
        "the flat reduction must await its own work before freeing its scratch"
    );
    let got = download(&backend, oh, 1);
    assert!(
        (got[0] - want).abs() <= 1e-3 * want.abs(),
        "sum = {}, expected {want}",
        got[0]
    );
    backend.set_async_dispatch(false).expect("disable async");
    for h in [ih, oh] {
        backend.free(h).expect("free");
    }
}

/// `launch_custom_kernel` runs caller-supplied code against caller-chosen
/// buffers, and `MetalComputePipeline::dispatch` waits only on its *own* command
/// buffer — so the entry point must drain the in-flight queue first, exactly
/// like a host read does.
#[test]
fn launch_custom_kernel_synchronizes_before_running_opaque_code() {
    let Some(backend) = try_init() else { return };
    const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void double_it(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    output[gid] = input[gid] * 2.0f;
}
"#;
    let values = vec![-1.0f32, 0.0, 3.0, -4.0, 5.5, 6.0, -7.0, 8.0];
    let n = values.len();
    let (Some(ih), Some(mid), Some(oh)) = (
        upload(&backend, &values),
        upload(&backend, &vec![0.0f32; n]),
        upload(&backend, &vec![0.0f32; n]),
    ) else {
        return;
    };
    backend.set_async_dispatch(true).expect("enable async");
    // Queue work that writes `mid` and do NOT synchronise explicitly.
    backend
        .unary(oxicuda_backend::UnaryOp::Relu, ih, mid, n)
        .expect("relu");
    assert_eq!(backend.inflight_count(), 1, "the relu is still in flight");

    let count = u32::try_from(n).expect("fits");
    backend
        .launch_custom_kernel(SRC, "double_it", &[mid, oh], &[&count.to_le_bytes()], n)
        .expect("custom kernel");
    assert_eq!(
        backend.inflight_count(),
        0,
        "launch_custom_kernel must be a synchronisation point"
    );

    let want: Vec<f32> = values.iter().map(|&v| v.max(0.0) * 2.0).collect();
    let got = download(&backend, oh, n);
    assert_close(&got, &want, 1e-6, "relu then custom double");

    backend.set_async_dispatch(false).expect("disable async");
    for h in [ih, mid, oh] {
        backend.free(h).expect("free");
    }
}

// ─── GPU-liveness witness ────────────────────────────────────────────────────

/// Proves the numeric tests above are not passing vacuously.
///
/// Two distinct failure modes hide behind a green suite:
///
/// 1. **No device.** `try_init` returns `None` and every test returns early, so
///    the suite is green having run no kernel at all. `OXICUDA_REQUIRE_GPU=1`
///    makes that a hard failure.
/// 2. **A live host fallback.** `dispatch_attention` falls back to
///    [`attention_host`] when no SIMD-group count fits — and `attention_host` is
///    also the oracle the attention tests compare against, so a fallback makes
///    "GPU == host" trivially true. Asserting `attention_simdgroups(..).is_some()`
///    for the exact `head_dim`s those tests use is the one check that separates
///    "the kernel ran" from "the CPU loop matched itself".
///
/// `conv2d` needs no such guard: `dispatch_conv2d` has no host path whatsoever,
/// so a green conv2d oracle test is necessarily GPU execution.
#[test]
fn gpu_device_is_live_when_required() {
    let Some(backend) = try_init() else {
        assert!(
            !require_gpu(),
            "OXICUDA_REQUIRE_GPU=1 but no Metal device is available"
        );
        return;
    };

    let devices = backend.available_devices().expect("available_devices");
    assert!(
        !devices.is_empty(),
        "an initialised backend must report a device"
    );
    assert!(
        !devices[0].name.is_empty(),
        "the device must have a real name"
    );

    let caps = backend.capabilities();
    assert_eq!(
        caps.warp_size, 32,
        "Apple GPUs execute in 32-wide SIMD groups"
    );
    assert!(caps.supports_fp16, "MSL `half` is a first-class scalar");
    assert!(
        caps.max_threads_per_block >= 32,
        "a threadgroup must hold at least one SIMD group"
    );

    // The `head_dim`s exercised by the attention tests in this module must all
    // take the GPU path, or those tests compare `attention_host` with itself.
    for head_dim in [2usize, 4, 40] {
        assert!(
            backend
                .attention_simdgroups(head_dim, u64::from(caps.max_threads_per_block))
                .is_some(),
            "head_dim {head_dim} must dispatch on the GPU, not the host fallback"
        );
    }

    println!(
        "GPU WITNESS: device={:?} warp={} max_threads={} shared={}B",
        devices[0].name, caps.warp_size, caps.max_threads_per_block, caps.max_shared_mem_per_block
    );
}
