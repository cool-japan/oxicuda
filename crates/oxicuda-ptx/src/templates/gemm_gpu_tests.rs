//! On-device numeric coverage for [`GemmTemplate::generate`]'s row/col
//! strength-reduction (see the comment above `$TILE_LOOP` inside
//! `generate`), specifically forcing the grid-stride loop to iterate more
//! than once per thread -- the code path the strength-reduced row/col
//! *advance* actually exercises.
//!
//! `oxicuda-blas`'s GPU suite (`gpu_tests.rs`,
//! `simt_gemm_f32_square_matches_host` and friends) already validates
//! `generate()`'s *production* launch geometry end-to-end, but that launch
//! is sized `m * n` threads wide (one thread per output element), so every
//! thread's grid-stride loop body runs exactly once and the advance step
//! at the bottom of the loop is dead code for that test: it computes new
//! row/col values that are never read before the thread's very next
//! iteration hits the bounds check and exits. This file's coverage is
//! deliberately complementary to (not redundant with) that suite: a
//! *deliberately undersized* launch (far fewer threads than output
//! elements) forces dozens of advances -- and, since the launch width is
//! chosen not to divide the column count, dozens of row-carries -- per
//! thread.
//!
//! Requires the `gpu-tests` feature and, to do anything beyond a no-op, a
//! live CUDA driver + device; every test returns early (skips) when
//! neither is present.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::arch::SmVersion;
use crate::ir::PtxType;

use super::{EpilogueKind, GemmTemplate};

/// A live CUDA context + stream, plus the device's SM version.
struct GpuFixture {
    _ctx: Arc<Context>,
    stream: Stream,
    sm: SmVersion,
}

/// Acquires a GPU fixture, or `None` when no driver / device is present.
fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let dev = Device::get(0).ok()?;
    let (major, minor) = dev.compute_capability().ok()?;
    let sm = SmVersion::from_compute_capability(major, minor).unwrap_or(SmVersion::Sm80);
    let ctx = Arc::new(Context::new(&dev).ok()?);
    let stream = Stream::new(&ctx).ok()?;
    Some(GpuFixture {
        _ctx: ctx,
        stream,
        sm,
    })
}

/// Relative-with-absolute-floor closeness test, matching
/// `oxicuda_blas::gpu_tests::close_f32`'s tolerance convention for FP32
/// GEMM comparisons in this workspace.
fn close_f32(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel.mul_add(a.abs().max(b.abs()), abs)
}

/// Host reference GEMM (row-major): `C = alpha * A(MxK) * B(KxN) + beta * C`,
/// with the *same* per-element FMA accumulation order as the kernel's
/// `$K_LOOP` (`k = 0..K`, `acc = fma(A[row,k], B[k,col], acc)`), so this is
/// a same-order re-derivation, not just an approximate check.
#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
fn cpu_gemm_f32(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc = a[row * k + p].mul_add(b[p * n + col], acc);
            }
            out[row * n + col] = alpha.mul_add(acc, beta * c[row * n + col]);
        }
    }
    out
}

/// Launches `generate()`'s kernel for a `37 x 29 x 11` problem (1073 output
/// elements -- deliberately not a multiple of the 16-wide launch below) with
/// only 16 threads total, forcing every thread through
/// `ceil(1073 / 16) = 68` grid-stride iterations. `stride_rem_n = 16 % 29 =
/// 16` (`stride_div_n = 16 / 29 = 0`), so *every* row advance in this test
/// comes from the carry branch, not the `stride_div_n` baseline term --
/// deliberately the strongest exercise of the carry logic a single launch
/// geometry can give it.
/// Threads launched by
/// [`generate_grid_stride_multi_iteration_matches_cpu_oracle`] -- deliberately
/// far fewer than the problem's `M*N` (1073) output elements.
const GRID_STRIDE_TEST_THREADS: u32 = 16;

#[test]
#[allow(clippy::many_single_char_names)]
fn generate_grid_stride_multi_iteration_matches_cpu_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let (m, n, k): (usize, usize, usize) = (37, 29, 11);
    let template = GemmTemplate {
        tile_m: 64,
        tile_n: 64,
        tile_k: 8,
        warp_m: 32,
        warp_n: 32,
        precision: PtxType::F32,
        accumulator: PtxType::F32,
        use_tensor_core: false,
        stages: 1,
        target: fx.sm,
        epilogue: EpilogueKind::LinearCombination,
    };
    let ptx = template.generate().expect("generate() must succeed");
    let module = Module::from_ptx(&ptx)
        .unwrap_or_else(|e| panic!("PTX JIT compile failed: {e}\n--- PTX ---\n{ptx}"));
    let kernel = Kernel::from_module(Arc::new(module), &template.kernel_name())
        .unwrap_or_else(|e| panic!("kernel lookup failed: {e}"));

    // Deterministic, non-uniform inputs (a fixed affine-congruential
    // sequence, not an RNG draw -- reproducible across runs).
    let a: Vec<f32> = (0..m * k)
        .map(|i| f32::from(u8::try_from((i * 37 + 11) % 97).unwrap_or(0)) * 0.1 - 4.0)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| f32::from(u8::try_from((i * 53 + 7) % 101).unwrap_or(0)) * 0.1 - 4.0)
        .collect();
    let c0: Vec<f32> = (0..m * n)
        .map(|i| f32::from(u8::try_from((i * 13 + 3) % 89).unwrap_or(0)) * 0.1 - 3.0)
        .collect();
    let (alpha, beta) = (1.7f32, 0.6f32);

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a alloc");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b alloc");
    let d_c = DeviceBuffer::<f32>::from_host(&c0).expect("d_c alloc");

    let params = LaunchParams::new(
        Dim3::new(1, 1, 1),
        Dim3::new(GRID_STRIDE_TEST_THREADS, 1, 1),
    );
    let args = (
        d_a.as_device_ptr(),
        d_b.as_device_ptr(),
        d_c.as_device_ptr(),
        u32::try_from(m).expect("m fits u32"),
        u32::try_from(n).expect("n fits u32"),
        u32::try_from(k).expect("k fits u32"),
        u64::from(alpha.to_bits()),
        u64::from(beta.to_bits()),
    );
    kernel
        .launch(&params, &fx.stream, &args)
        .expect("kernel launch");
    fx.stream.synchronize().expect("synchronize");

    let mut gpu_result = vec![0.0f32; m * n];
    d_c.copy_to_host(&mut gpu_result).expect("copy_to_host");

    let expected = cpu_gemm_f32(&a, &b, &c0, m, n, k, alpha, beta);

    for i in 0..m * n {
        let (row, col) = (i / n, i % n);
        assert!(
            close_f32(gpu_result[i], expected[i], 1e-4, 1e-4),
            "mismatch at [{row},{col}] (flattened index {i} of {}): gpu={}, \
             cpu={} -- this element is reached only after {} grid-stride \
             advances of the thread that owns it (16 threads covering {} \
             elements), the exact path the row/col strength-reduction \
             replaced a per-iteration 64-bit div/rem with",
            m * n,
            gpu_result[i],
            expected[i],
            i / GRID_STRIDE_TEST_THREADS as usize,
            m * n,
        );
    }
}
