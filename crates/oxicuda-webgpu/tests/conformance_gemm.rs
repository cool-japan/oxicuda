//! Cross-backend GEMM conformance: `WebGpuBackend` vs
//! `oxicuda_backend::CpuBackend` vs a from-scratch `f64` oracle, plus
//! `WebGpuBackend`-only coverage of transpose combinations, padded leading
//! dimensions, batched GEMM, and the FP16 GEMM path.
//!
//! # Why an independent oracle, not just `CpuBackend`
//!
//! `CpuBackend::gemm` and `WebGpuBackend::gemm` implement the *same*
//! mathematical contract (`C = alpha * op(A) * op(B) + beta * C`) but with
//! **different storage conventions**: the CPU reference is column-major
//! `f64` (see `oxicuda_backend::ComputeBackend::gemm`'s doc), while
//! `WebGpuBackend::gemm` is row-major `f32` (its own doc: "The WGSL tiled
//! GEMM kernel handles every NN / NT / TN / TT combination at runtime").
//! A byte buffer cannot be shared between them, so every test here:
//!
//! 1. Generates one **logical** (mathematical, storage-agnostic) pair of
//!    matrices `A` (m×k) and `B` (k×n) plus an initial `C0` (m×n).
//! 2. Computes a plain triple-loop `f64` oracle for
//!    `alpha * A @ B + beta * C0`.
//! 3. Materializes `A`/`B`/`C0` as **column-major `f64`** for `CpuBackend`
//!    and as **row-major `f32`** for `WebGpuBackend`, and checks each
//!    backend's own output against the *same* oracle.
//!
//! Every GPU test degrades to a no-op skip when no wgpu adapter is present
//! (headless CI), matching the crate's existing `try_init` convention (see
//! `src/backend_tests.rs`).

use oxicuda_backend::{BackendTranspose, ComputeBackend, CpuBackend};
use oxicuda_webgpu::WebGpuBackend;

// ─── Shared helpers ──────────────────────────────────────────────────────────

fn try_init() -> Option<WebGpuBackend> {
    let mut b = WebGpuBackend::new();
    b.init().ok().map(|()| b)
}

/// Deterministic pseudo-random `f64` values in `[-2, 2)`, no `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 33) as u32 % 4001;
        (f64::from(bits) - 2000.0) / 1000.0
    }
    fn vec(&mut self, len: usize) -> Vec<f64> {
        (0..len).map(|_| self.next_f64()).collect()
    }
}

/// Row-major triple-loop `f64` GEMM oracle: `alpha * A @ B + beta * C0`,
/// with `A` logically `m×k` and `B` logically `k×n`, both row-major.
fn ref_gemm_f64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c0: &[f64],
) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = alpha * acc + beta * c0[i * n + j];
        }
    }
    c
}

/// Transpose a logical `rows×cols` row-major matrix into `cols×rows`
/// row-major — used to materialize the *stored* buffer for a transposed
/// operand under a given `BackendTranspose` mode.
fn transpose_row_major(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = m[r * cols + c];
        }
    }
    out
}

// ── CpuBackend (f64, column-major) helpers ───────────────────────────────────

fn upload_f64(be: &CpuBackend, data: &[f64]) -> u64 {
    let ptr = be.alloc(data.len() * 8).expect("cpu alloc");
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    be.copy_htod(ptr, &bytes).expect("cpu copy_htod");
    ptr
}
fn download_f64(be: &CpuBackend, ptr: u64, len: usize) -> Vec<f64> {
    let mut bytes = vec![0u8; len * 8];
    be.copy_dtoh(&mut bytes, ptr).expect("cpu copy_dtoh");
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect()
}
/// Upload a logical `rows×cols` row-major matrix as `CpuBackend`'s expected
/// column-major buffer (transposing on the way in).
fn upload_col_major_from_logical(
    be: &CpuBackend,
    logical: &[f64],
    rows: usize,
    cols: usize,
) -> u64 {
    let mut col_major = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            col_major[c * rows + r] = logical[r * cols + c];
        }
    }
    upload_f64(be, &col_major)
}
/// Download a `CpuBackend` column-major `rows×cols` result back into logical
/// row-major order, for comparison against [`ref_gemm_f64`].
fn download_logical_from_col_major(
    be: &CpuBackend,
    ptr: u64,
    rows: usize,
    cols: usize,
) -> Vec<f64> {
    let col_major = download_f64(be, ptr, rows * cols);
    let mut logical = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            logical[r * cols + c] = col_major[c * rows + r];
        }
    }
    logical
}

// ── WebGpuBackend (f32, row-major) helpers ───────────────────────────────────

fn upload_f32(be: &WebGpuBackend, data: &[f32]) -> u64 {
    let ptr = be.alloc((data.len() * 4).max(4)).expect("webgpu alloc");
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    be.copy_htod(ptr, &bytes).expect("webgpu copy_htod");
    ptr
}
fn download_f32(be: &WebGpuBackend, ptr: u64, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * 4];
    be.copy_dtoh(&mut bytes, ptr).expect("webgpu copy_dtoh");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect()
}

fn assert_close_f64(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let limit = tol * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= limit,
            "{what}: element {i} = {g}, expected {w} (tolerance {limit})"
        );
    }
}
fn assert_close_f32(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let limit = tol * w.abs().max(1.0);
        assert!(
            (f64::from(g) - w).abs() <= limit,
            "{what}: element {i} = {g}, expected {w} (tolerance {limit})"
        );
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

/// `CpuBackend` and `WebGpuBackend` agree with each other (via the shared f64
/// oracle) on a NoTrans/NoTrans GEMM with a nontrivial alpha/beta and a
/// nonzero initial C — the core cross-backend numeric contract.
#[test]
fn gemm_cpu_and_webgpu_agree_with_oracle_notrans() {
    let Some(webgpu) = try_init() else { return };
    let cpu = CpuBackend::new();

    let (m, n, k) = (4, 5, 3);
    let mut rng = Lcg::new(0xC0FFEE);
    let a = rng.vec(m * k);
    let b = rng.vec(k * n);
    let c0 = rng.vec(m * n);
    let (alpha, beta) = (1.7, 0.6);

    let oracle = ref_gemm_f64(m, n, k, alpha, &a, &b, beta, &c0);

    // CPU path (column-major f64). Column-major lda/ldb/ldc are row counts
    // (m, k, m), not column counts — easy to get backwards coming from the
    // row-major GPU side below, where lda = k instead.
    let a_cpu = upload_col_major_from_logical(&cpu, &a, m, k);
    let b_cpu = upload_col_major_from_logical(&cpu, &b, k, n);
    let c_cpu = upload_col_major_from_logical(&cpu, &c0, m, n);
    cpu.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        m,
        n,
        k,
        alpha,
        a_cpu,
        m,
        b_cpu,
        k,
        beta,
        c_cpu,
        m,
    )
    .expect("cpu gemm");
    let got_cpu = download_logical_from_col_major(&cpu, c_cpu, m, n);
    assert_close_f64(&got_cpu, &oracle, 1e-9, "cpu gemm vs f64 oracle");

    // WebGPU path (row-major f32) — same logical matrices, direct layout.
    let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = c0.iter().map(|&v| v as f32).collect();
    let a_g = upload_f32(&webgpu, &a32);
    let b_g = upload_f32(&webgpu, &b32);
    let c_g = upload_f32(&webgpu, &c32);
    webgpu
        .gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            m,
            n,
            k,
            alpha,
            a_g,
            k,
            b_g,
            n,
            beta,
            c_g,
            n,
        )
        .expect("webgpu gemm");
    let got_webgpu = download_f32(&webgpu, c_g, m * n);
    // f32 accumulation over k=3 terms plus an f64->f32 downcast of the
    // operands: a loose-but-real relative tolerance.
    assert_close_f32(&got_webgpu, &oracle, 5e-3, "webgpu gemm vs f64 oracle");
}

/// Every transpose combination (`NoTrans`/`Trans`/`ConjTrans`, which is
/// documented to collapse to `Trans` for real data) and a padded leading
/// dimension on every operand, all checked against the same f64 oracle.
#[test]
fn gemm_matches_oracle_every_transpose_combo_and_padded_ld() {
    let Some(webgpu) = try_init() else { return };
    let (m, n, k) = (5, 4, 3);
    let mut rng = Lcg::new(0xFACADE);
    let a = rng.vec(m * k);
    let b = rng.vec(k * n);
    let oracle = ref_gemm_f64(m, n, k, 1.0, &a, &b, 0.0, &vec![0.0; m * n]);

    for &trans_a in &[
        BackendTranspose::NoTrans,
        BackendTranspose::Trans,
        BackendTranspose::ConjTrans,
    ] {
        for &trans_b in &[
            BackendTranspose::NoTrans,
            BackendTranspose::Trans,
            BackendTranspose::ConjTrans,
        ] {
            // Stored operand shape/layout depends on the transpose flag: op(A)
            // is always logically m×k, so a Trans-stored A is physically k×m.
            let (a_stored, a_rows, a_cols) = if trans_a == BackendTranspose::NoTrans {
                (a.clone(), m, k)
            } else {
                (transpose_row_major(&a, m, k), k, m)
            };
            let (b_stored, b_rows, b_cols) = if trans_b == BackendTranspose::NoTrans {
                (b.clone(), k, n)
            } else {
                (transpose_row_major(&b, k, n), n, k)
            };

            // Pad every leading dimension by a few extra columns of garbage,
            // proving the kernel honours `lda`/`ldb`/`ldc` rather than
            // assuming a tightly packed buffer.
            const PAD: usize = 2;
            let lda = a_cols + PAD;
            let ldb = b_cols + PAD;
            let ldc = n + PAD;

            let mut a_padded = vec![0.0f32; a_rows * lda];
            for r in 0..a_rows {
                for c in 0..a_cols {
                    a_padded[r * lda + c] = a_stored[r * a_cols + c] as f32;
                }
            }
            let mut b_padded = vec![0.0f32; b_rows * ldb];
            for r in 0..b_rows {
                for c in 0..b_cols {
                    b_padded[r * ldb + c] = b_stored[r * b_cols + c] as f32;
                }
            }
            let c_padded = vec![0.0f32; m * ldc];

            let a_ptr = upload_f32(&webgpu, &a_padded);
            let b_ptr = upload_f32(&webgpu, &b_padded);
            let c_ptr = upload_f32(&webgpu, &c_padded);

            webgpu
                .gemm(
                    trans_a, trans_b, m, n, k, 1.0, a_ptr, lda, b_ptr, ldb, 0.0, c_ptr, ldc,
                )
                .unwrap_or_else(|e| panic!("webgpu gemm({trans_a:?},{trans_b:?}) failed: {e}"));

            // Extract the logical m×n result out of the padded ldc buffer.
            let c_out = download_f32(&webgpu, c_ptr, m * ldc);
            let mut got = vec![0.0f32; m * n];
            for r in 0..m {
                got[r * n..(r + 1) * n].copy_from_slice(&c_out[r * ldc..r * ldc + n]);
            }
            assert_close_f32(
                &got,
                &oracle,
                5e-3,
                &format!("webgpu gemm({trans_a:?},{trans_b:?}) padded ld vs oracle"),
            );
        }
    }
}

/// Batched GEMM numeric check: `WebGpuBackend::batched_gemm` (a single
/// dispatch covering the whole batch via the `z` workgroup axis) against a
/// *manually* per-batch `CpuBackend::gemm` oracle.
///
/// Deliberately does **not** use `ComputeBackend::batched_gemm`'s default
/// implementation on the CPU side: that default offsets pointers assuming
/// 4-byte (`f32`) elements while `CpuBackend::gemm` reads 8-byte `f64`
/// elements (see `CpuBackend`'s own `batched_gemm_default_runs_on_cpu` test,
/// which sidesteps the mismatch by using a single batch). Looping
/// `CpuBackend::gemm` here, with correct 8-byte-element offsets computed by
/// hand, avoids that trap entirely.
#[test]
fn batched_gemm_matches_per_batch_cpu_oracle() {
    let Some(webgpu) = try_init() else { return };
    let cpu = CpuBackend::new();

    let (m, n, k, batch_count) = (3usize, 2usize, 4usize, 4usize);
    let mut rng = Lcg::new(0xBA7C4);
    let batches_a: Vec<Vec<f64>> = (0..batch_count).map(|_| rng.vec(m * k)).collect();
    let batches_b: Vec<Vec<f64>> = (0..batch_count).map(|_| rng.vec(k * n)).collect();

    // Per-batch f64 oracle via a fresh CpuBackend::gemm call per batch —
    // correct regardless of the trait default's element-size assumption.
    let mut oracle_batches: Vec<Vec<f64>> = Vec::with_capacity(batch_count);
    for b in 0..batch_count {
        let a_ptr = upload_col_major_from_logical(&cpu, &batches_a[b], m, k);
        let b_ptr = upload_col_major_from_logical(&cpu, &batches_b[b], k, n);
        let c_ptr = upload_col_major_from_logical(&cpu, &vec![0.0; m * n], m, n);
        cpu.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            m,
            n,
            k,
            1.0,
            a_ptr,
            m,
            b_ptr,
            k,
            0.0,
            c_ptr,
            m,
        )
        .expect("cpu per-batch gemm");
        oracle_batches.push(download_logical_from_col_major(&cpu, c_ptr, m, n));
        cpu.free(a_ptr).expect("free");
        cpu.free(b_ptr).expect("free");
        cpu.free(c_ptr).expect("free");
    }

    // WebGPU path: one contiguous f32 row-major buffer per operand, natural
    // (tightly packed) per-batch strides.
    let mut a_flat = Vec::with_capacity(batch_count * m * k);
    let mut b_flat = Vec::with_capacity(batch_count * k * n);
    for b in 0..batch_count {
        a_flat.extend(batches_a[b].iter().map(|&v| v as f32));
        b_flat.extend(batches_b[b].iter().map(|&v| v as f32));
    }
    let c_flat = vec![0.0f32; batch_count * m * n];

    let a_ptr = upload_f32(&webgpu, &a_flat);
    let b_ptr = upload_f32(&webgpu, &b_flat);
    let c_ptr = upload_f32(&webgpu, &c_flat);

    webgpu
        .batched_gemm(
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
        )
        .expect("webgpu batched_gemm");

    let got_flat = download_f32(&webgpu, c_ptr, batch_count * m * n);
    for b in 0..batch_count {
        let got_batch = &got_flat[b * m * n..(b + 1) * m * n];
        assert_close_f32(
            got_batch,
            &oracle_batches[b],
            5e-3,
            &format!("batched_gemm batch {b} vs per-batch cpu oracle"),
        );
    }
}

/// FP16 GEMM numeric check: round the logical operands to f16-representable
/// `f32` values (matching the "store in half, accumulate in f32" contract
/// documented on `ComputeBackend::gemm_mixed_precision`, which
/// `WebGpuBackend::gemm_f16` implements as an inherent method) before
/// computing the oracle, then compare against the real kernel's f16-packed
/// output. `beta = 0` sidesteps a *second*, independent precision question —
/// whether the kernel's `C` accumulator/readback also round-trips through
/// half — since with `beta = 0` the initial value of `C` cannot affect the
/// result at all.
///
/// Skips (not fails) when the adapter lacks the `SHADER_F16` feature —
/// `supports_f16()` is the documented, non-panicking way to check before
/// calling `gemm_f16`, which otherwise returns `Unsupported`.
#[test]
fn gemm_f16_matches_f16_rounded_oracle() {
    let Some(webgpu) = try_init() else { return };
    if !webgpu.supports_f16() {
        return;
    }

    let (m, n, k) = (4usize, 4usize, 4usize);
    let mut rng = Lcg::new(0xF16F16);
    let a: Vec<f32> = rng.vec(m * k).iter().map(|&v| v as f32).collect();
    let b: Vec<f32> = rng.vec(k * n).iter().map(|&v| v as f32).collect();

    // Oracle: round inputs to f16-representable f32, accumulate in f32 —
    // exactly CpuBackend::gemm_mixed_precision's documented contract.
    let a_rounded: Vec<f32> = a
        .iter()
        .map(|&v| oxicuda_backend::round_to_f16(v))
        .collect();
    let b_rounded: Vec<f32> = b
        .iter()
        .map(|&v| oxicuda_backend::round_to_f16(v))
        .collect();
    let mut oracle = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a_rounded[i * k + p] * b_rounded[p * n + j];
            }
            oracle[i * n + j] = f64::from(acc);
        }
    }

    let a_bits: Vec<u16> = a
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let b_bits: Vec<u16> = b
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let c_bits = vec![0u16; m * n];

    let a_ptr = webgpu.alloc(a_bits.len() * 2).expect("alloc a");
    let b_ptr = webgpu.alloc(b_bits.len() * 2).expect("alloc b");
    let c_ptr = webgpu.alloc(c_bits.len() * 2).expect("alloc c");
    let to_bytes = |d: &[u16]| -> Vec<u8> { d.iter().flat_map(|v| v.to_le_bytes()).collect() };
    webgpu.copy_htod(a_ptr, &to_bytes(&a_bits)).expect("htod a");
    webgpu.copy_htod(b_ptr, &to_bytes(&b_bits)).expect("htod b");
    webgpu.copy_htod(c_ptr, &to_bytes(&c_bits)).expect("htod c");

    webgpu
        .gemm_f16(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            m,
            n,
            k,
            1.0,
            a_ptr,
            k,
            b_ptr,
            n,
            0.0,
            c_ptr,
            n,
        )
        .expect("webgpu gemm_f16");

    let mut out_bytes = vec![0u8; m * n * 2];
    webgpu.copy_dtoh(&mut out_bytes, c_ptr).expect("dtoh c");
    let got: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    // Half-precision throughout (inputs *and* the final store), so a wider
    // relative tolerance than the f32 GEMM tests above.
    assert_close_f32(&got, &oracle, 3e-2, "webgpu gemm_f16 vs f16-rounded oracle");
}
