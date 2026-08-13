//! On-device GPU validation for the GEMM PTX emitted by `oxicuda-blas`.
//!
//! The production dispatcher (`level3::gemm::dispatch::GemmDispatcher`)
//! compiles its `Template`-launch-kind kernels *exclusively* from
//! [`GemmTemplate::generate`] (the grid-stride SIMT path); its `Simt`-launch-
//! kind kernels (transposed operands, triangle-masked SYRK/SYR2K writes) come
//! from a separate builder, `super::simt::SimtGemmBuilder`, not from
//! `GemmTemplate` at all. Neither dispatcher path calls
//! [`GemmTemplate::generate_pipelined_skeleton`] -- as the name says, it is a
//! structural skeleton (see its own doc comment), not a second production
//! kernel; this suite exercises it directly, independent of the dispatcher,
//! precisely because nothing else ever launches it. These unit-test suites
//! previously only checked the emitted PTX *as a string* (instruction
//! presence). This module instead JIT-compiles the PTX for the live device
//! via `Module::from_ptx`, launches it on the real CUDA GPU through
//! `oxicuda-launch`, copies the results back, and asserts equivalence to an
//! independent CPU re-derivation.
//!
//! ## Honest kernel accounting
//!
//! * **Validated against a CPU oracle (numerically complete).** The SIMT kernel
//!   from [`GemmTemplate::generate`] — this is the exact PTX the production
//!   dispatcher launches (`dispatch.rs` `Template` launch kind). It performs a
//!   full grid-stride `C = alpha * A*B + beta * C` with per-element
//!   K-reduction and precision conversion. Validated here for `F32` and `F64`
//!   across square / non-square shapes, non-trivial `alpha`/`beta`, and a
//!   deliberate-corruption probe that proves the launch reads device memory
//!   (non-vacuous).
//!
//! * **Structural launch only (no numeric oracle).** The pipelined kernel from
//!   [`GemmTemplate::generate_pipelined_skeleton`] is a software-pipeline
//!   *skeleton*, dispatched by nothing in production: it emits a correct
//!   `cp.async` prologue / steady-state / drain schedule and an
//!   `mma.sync.aligned.m16n8k16` (or an FMA placeholder), but the `mma`
//!   operand registers are not loaded from the staged shared-memory tiles and
//!   the store writes a single accumulator lane. It therefore does **not**
//!   compute a complete GEMM, and asserting a numeric result would be
//!   dishonest. We instead validate that (a) every config assembles under
//!   `ptxas` for the live architecture, and (b) both the f32-FMA and the real
//!   Ampere f16 `mma.sync` variants JIT-load and launch fault-free on the
//!   device. This validates the async-copy machinery and the HMMA instruction
//!   itself execute on hardware.
//!
//! Every device test returns early (skips) when no CUDA device is present, so
//! the suite stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::{EpilogueKind, GemmTemplate, PtxType, SmVersion};

// ---------------------------------------------------------------------------
// Fixture & shared helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
struct GpuFixture {
    ctx: Arc<Context>,
    sm: SmVersion,
}

/// Acquire a GPU fixture, or `None` when no driver / device is present.
fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let dev = Device::get(0).ok()?;
    let (major, minor) = dev.compute_capability().ok()?;
    let sm = SmVersion::from_compute_capability(major, minor).unwrap_or(SmVersion::Sm80);
    let ctx = Context::new(&dev).ok()?;
    Some(GpuFixture {
        ctx: Arc::new(ctx),
        sm,
    })
}

/// JIT-compile `ptx` and look up `entry`, returning a launchable kernel.
///
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real bug in the
/// generated kernel, surfaced as a test failure rather than silently skipped.
fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx).unwrap_or_else(|e| {
        panic!("PTX JIT compile failed for `{entry}`: {e}\n--- PTX ---\n{ptx}")
    });
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

/// Extracts the `.visible .entry NAME(` identifier from a PTX module.
fn entry_name(ptx: &str) -> String {
    let marker = ".visible .entry ";
    let start = ptx
        .find(marker)
        .map(|p| p + marker.len())
        .expect("module must contain a .visible .entry");
    let rest = &ptx[start..];
    let end = rest
        .find(|c: char| c == '(' || c.is_whitespace())
        .unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

/// `ceil(n / block)` as a 1-D grid size (at least 1).
fn grid_1d(n: u32, block: u32) -> u32 {
    n.div_ceil(block).max(1)
}

/// Relative-with-absolute-floor closeness test for FP32 comparisons.
fn close_f32(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Asserts two FP32 slices agree within tolerance, reporting the first mismatch.
fn assert_close_f32(gpu: &[f32], cpu: &[f32], rel: f32, abs: f32, tag: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{tag}: length mismatch");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            close_f32(g, c, rel, abs),
            "{tag}: element {i} mismatch gpu={g} cpu={c} (rel={rel:e} abs={abs:e})"
        );
    }
}

/// Asserts two FP64 slices agree within tolerance, reporting the first mismatch.
fn assert_close_f64(gpu: &[f64], cpu: &[f64], rel: f64, abs: f64, tag: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{tag}: length mismatch");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (g - c).abs() <= rel * g.abs().max(c.abs()) + abs,
            "{tag}: element {i} mismatch gpu={g} cpu={c} (rel={rel:e} abs={abs:e})"
        );
    }
}

/// A small deterministic LCG. Normalisation divides by `2^32` (never `2^31`).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    /// Uniform in `[0, 1)` via division by `2^32`.
    fn unit(&mut self) -> f64 {
        f64::from(self.next_u32()) / 4_294_967_296.0
    }
    /// Uniform `f64` in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Builds the SIMT (`.generate()`) GEMM template the production dispatcher uses,
/// for the given precision/accumulator. `use_tensor_core` only affects the
/// kernel-name suffix for the SIMT path; we keep it `false` (the "naive" name).
fn simt_template(precision: PtxType, accumulator: PtxType) -> GemmTemplate {
    GemmTemplate {
        tile_m: 16,
        tile_n: 16,
        tile_k: 16,
        warp_m: 16,
        warp_n: 16,
        precision,
        accumulator,
        use_tensor_core: false,
        stages: 1,
        target: SmVersion::Sm86,
        epilogue: EpilogueKind::LinearCombination,
    }
}

/// `ptxas` pre-screen: assembles `ptx` for its own declared `.target`. Returns
/// `Ok(())` on success or when `ptxas` is unavailable, the captured stderr on
/// assembler rejection.
fn ptxas_assembles(ptx: &str, tag: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::Command;

    let ptxas = "/usr/local/cuda/bin/ptxas";
    if !std::path::Path::new(ptxas).exists() {
        return Ok(());
    }
    let arch = ptx
        .lines()
        .find_map(|l| l.trim().strip_prefix(".target "))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "sm_86".to_string());

    let dir = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let in_path = dir.join(format!("oxiblas_{tag}_{stamp}.ptx"));
    let out_path = dir.join(format!("oxiblas_{tag}_{stamp}.cubin"));

    {
        let mut f = match std::fs::File::create(&in_path) {
            Ok(f) => f,
            Err(_) => return Ok(()),
        };
        if f.write_all(ptx.as_bytes()).is_err() {
            return Ok(());
        }
    }

    let output = Command::new(ptxas)
        .arg(format!("-arch={arch}"))
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .output();

    let result = match output {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => Err(format!(
            "ptxas rejected `{tag}` (arch={arch}): {}",
            String::from_utf8_lossy(&o.stderr)
        )),
        Err(_) => Ok(()),
    };

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    result
}

/// Row-major GEMM dimensions: `A` is `MxK`, `B` is `KxN`, `C` is `MxN`.
#[derive(Clone, Copy)]
struct Shape {
    m: usize,
    n: usize,
    k: usize,
}

impl Shape {
    fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }
    fn elems(&self) -> usize {
        self.m * self.n
    }
}

/// Host reference GEMM (row-major): `C = alpha * A(MxK) * B(KxN) + beta * C`.
fn cpu_gemm_f32(a: &[f32], b: &[f32], c: &[f32], s: Shape, alpha: f32, beta: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; s.m * s.n];
    for i in 0..s.m {
        for j in 0..s.n {
            let mut acc = 0.0f32;
            for p in 0..s.k {
                acc = a[i * s.k + p].mul_add(b[p * s.n + j], acc);
            }
            out[i * s.n + j] = alpha * acc + beta * c[i * s.n + j];
        }
    }
    out
}

/// Host reference GEMM (row-major) in FP64.
fn cpu_gemm_f64(a: &[f64], b: &[f64], c: &[f64], s: Shape, alpha: f64, beta: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; s.m * s.n];
    for i in 0..s.m {
        for j in 0..s.n {
            let mut acc = 0.0f64;
            for p in 0..s.k {
                acc = a[i * s.k + p].mul_add(b[p * s.n + j], acc);
            }
            out[i * s.n + j] = alpha * acc + beta * c[i * s.n + j];
        }
    }
    out
}

// ===========================================================================
// 0. ptxas pre-screen of every emitted GEMM PTX (no GPU required)
// ===========================================================================

/// Every (precision, tensor-core, stages) config the dispatcher can emit must
/// assemble cleanly under `ptxas` for `sm_86`. Catches invalid-PTX regressions
/// cheaply, including the f16/bf16 `mma.sync.m16n8k16` HMMA encoding.
#[test]
fn ptxas_prescreen_all_gemm_configs() {
    let mut failures: Vec<String> = Vec::new();
    let mut check = |ptx: &str, tag: &str| {
        if let Err(e) = ptxas_assembles(ptx, tag) {
            failures.push(e);
        }
    };

    // SIMT path across precisions (and mixed-precision accumulation).
    for (prec, acc, tag) in [
        (PtxType::F32, PtxType::F32, "simt_f32_f32"),
        (PtxType::F64, PtxType::F64, "simt_f64_f64"),
        (PtxType::F16, PtxType::F32, "simt_f16_f32"),
        (PtxType::BF16, PtxType::F32, "simt_bf16_f32"),
        (PtxType::F16, PtxType::F64, "simt_f16_f64"),
    ] {
        let ptx = simt_template(prec, acc).generate().expect("simt generate");
        check(&ptx, tag);
    }

    // Pipelined cp.async path: f32-FMA placeholder and f16/bf16 tensor-core MMA.
    for stages in [2u32, 3, 4] {
        let mut t = simt_template(PtxType::F32, PtxType::F32);
        t.tile_m = 16;
        t.tile_n = 8;
        t.tile_k = 16;
        t.stages = stages;
        t.use_tensor_core = false;
        let ptx = t
            .generate_pipelined_skeleton()
            .expect("pipelined fma generate");
        check(&ptx, &format!("pipe_fma_{stages}stage"));

        for (prec, tag) in [(PtxType::F16, "f16"), (PtxType::BF16, "bf16")] {
            let mut tc = simt_template(prec, PtxType::F32);
            tc.tile_m = 16;
            tc.tile_n = 8;
            tc.tile_k = 16;
            tc.warp_n = 8;
            tc.stages = stages;
            tc.use_tensor_core = true;
            let ptx = tc
                .generate_pipelined_skeleton()
                .expect("pipelined mma generate");
            check(&ptx, &format!("pipe_mma_{tag}_{stages}stage"));
        }
    }

    assert!(
        failures.is_empty(),
        "ptxas rejected {} GEMM config(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ===========================================================================
// 1. SIMT GEMM — numeric CPU-vs-GPU validation (the production kernel)
// ===========================================================================

/// Launches the SIMT GEMM PTX on the device and returns C.
fn run_simt_gemm_f32(
    fx: &GpuFixture,
    a: &[f32],
    b: &[f32],
    c0: &[f32],
    s: Shape,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let stream = Stream::new(&fx.ctx).expect("stream");
    let mut t = simt_template(PtxType::F32, PtxType::F32);
    t.target = fx.sm;
    let ptx = t.generate().expect("ptx");
    let kernel = load_kernel(&ptx, &entry_name(&ptx));

    let d_a = DeviceBuffer::<f32>::from_host(a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(b).expect("d_b");
    let d_c = DeviceBuffer::<f32>::from_host(c0).expect("d_c");

    let (m, n, k) = (s.m as u32, s.n as u32, s.k as u32);
    let block = 128u32;
    let params = LaunchParams::new(grid_1d(m * n, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                m,
                n,
                k,
                alpha,
                beta,
            ),
        )
        .expect("launch simt gemm");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; s.elems()];
    d_c.copy_to_host(&mut got).expect("copy");
    got
}

#[test]
fn simt_gemm_f32_square_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let s = Shape::new(64, 64, 64);
    let mut rng = Lcg::new(0xA1B2_C3D4);
    let a: Vec<f32> = (0..s.m * s.k)
        .map(|_| rng.range(-1.0, 1.0) as f32)
        .collect();
    let b: Vec<f32> = (0..s.k * s.n)
        .map(|_| rng.range(-1.0, 1.0) as f32)
        .collect();
    let c0: Vec<f32> = (0..s.elems())
        .map(|_| rng.range(-0.5, 0.5) as f32)
        .collect();
    let (alpha, beta) = (1.0f32, 0.0f32);

    let got = run_simt_gemm_f32(&fx, &a, &b, &c0, s, alpha, beta);
    let expect = cpu_gemm_f32(&a, &b, &c0, s, alpha, beta);
    assert_close_f32(&got, &expect, 1e-4, 1e-4, "simt_f32_square");
}

#[test]
fn simt_gemm_f32_nonsquare_with_alpha_beta_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let s = Shape::new(48, 32, 40);
    let mut rng = Lcg::new(0x0F1E_2D3C);
    let a: Vec<f32> = (0..s.m * s.k)
        .map(|_| rng.range(-1.5, 1.5) as f32)
        .collect();
    let b: Vec<f32> = (0..s.k * s.n)
        .map(|_| rng.range(-1.5, 1.5) as f32)
        .collect();
    let c0: Vec<f32> = (0..s.elems())
        .map(|_| rng.range(-1.0, 1.0) as f32)
        .collect();
    let (alpha, beta) = (0.75f32, -1.25f32);

    let got = run_simt_gemm_f32(&fx, &a, &b, &c0, s, alpha, beta);
    let expect = cpu_gemm_f32(&a, &b, &c0, s, alpha, beta);
    assert_close_f32(&got, &expect, 1e-4, 1e-4, "simt_f32_nonsquare_ab");
}

/// Deliberate-corruption probe: perturbing one input element must change the
/// output, proving the launch genuinely reads device memory (non-vacuous).
#[test]
fn simt_gemm_f32_corruption_probe_is_detected() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let s = Shape::new(16, 16, 16);
    let mut rng = Lcg::new(0xDEAD_BEEF);
    let a: Vec<f32> = (0..s.m * s.k)
        .map(|_| rng.range(-1.0, 1.0) as f32)
        .collect();
    let b: Vec<f32> = (0..s.k * s.n)
        .map(|_| rng.range(-1.0, 1.0) as f32)
        .collect();
    let c0 = vec![0.0f32; s.elems()];

    let clean = run_simt_gemm_f32(&fx, &a, &b, &c0, s, 1.0, 0.0);
    let expect = cpu_gemm_f32(&a, &b, &c0, s, 1.0, 0.0);
    assert_close_f32(&clean, &expect, 1e-4, 1e-4, "probe_clean");

    // Corrupt A[0,0] (feeds row 0 of C) and re-run; row 0 must change.
    let mut a_bad = a.clone();
    a_bad[0] += 7.0;
    let dirty = run_simt_gemm_f32(&fx, &a_bad, &b, &c0, s, 1.0, 0.0);
    let changed = clean
        .iter()
        .zip(dirty.iter())
        .any(|(&x, &y)| (x - y).abs() > 1e-3);
    assert!(
        changed,
        "corrupting an input A element did not change the GEMM output — \
         the kernel may not be reading device memory"
    );
}

#[test]
fn simt_gemm_f64_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");
    let s = Shape::new(32, 24, 28);
    let mut rng = Lcg::new(0x5566_7788);
    let a: Vec<f64> = (0..s.m * s.k).map(|_| rng.range(-1.0, 1.0)).collect();
    let b: Vec<f64> = (0..s.k * s.n).map(|_| rng.range(-1.0, 1.0)).collect();
    let c0: Vec<f64> = (0..s.elems()).map(|_| rng.range(-0.5, 0.5)).collect();
    let (alpha, beta) = (1.3f64, 0.4f64);

    let mut t = simt_template(PtxType::F64, PtxType::F64);
    t.target = fx.sm;
    let ptx = t.generate().expect("ptx");
    let kernel = load_kernel(&ptx, &entry_name(&ptx));

    let d_a = DeviceBuffer::<f64>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f64>::from_host(&b).expect("d_b");
    let d_c = DeviceBuffer::<f64>::from_host(&c0).expect("d_c");

    let (m, n, k) = (s.m as u32, s.n as u32, s.k as u32);
    let block = 128u32;
    let params = LaunchParams::new(grid_1d(m * n, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                m,
                n,
                k,
                alpha,
                beta,
            ),
        )
        .expect("launch f64 gemm");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f64; s.elems()];
    d_c.copy_to_host(&mut got).expect("copy");
    let expect = cpu_gemm_f64(&a, &b, &c0, s, alpha, beta);
    assert_close_f64(&got, &expect, 1e-12, 1e-12, "simt_f64");
}

// ===========================================================================
// 2. Pipelined cp.async / tensor-core — structural on-device launch
// ===========================================================================

/// The `cp.async` software-pipeline (f32 FMA-placeholder variant) must JIT-load
/// and launch fault-free on the device. This validates the async-copy schedule
/// (prologue prefetch, `cp.async.commit_group` / `cp.async.wait_group` /
/// `bar.sync` steady state, drain epilogue) executes on hardware. The kernel is
/// a pipeline *skeleton* (see module docs) so its output is not a meaningful
/// GEMM — only fault-free execution is asserted here.
#[test]
fn pipelined_cp_async_fma_launches_on_device() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");

    let mut t = simt_template(PtxType::F32, PtxType::F32);
    t.target = fx.sm;
    t.tile_m = 16;
    t.tile_n = 8;
    t.tile_k = 16;
    t.warp_n = 8;
    t.stages = 3;
    t.use_tensor_core = false;
    let ptx = t.generate_pipelined_skeleton().expect("pipelined ptx");
    let kernel = load_kernel(&ptx, &entry_name(&ptx));

    // Staging tiles: A(16x16), B(16x8); C holds the single written accumulator.
    let a = vec![1.0f32; 16 * 16];
    let b = vec![1.0f32; 16 * 8];
    let c = vec![0.0f32; 16 * 8];
    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    let d_c = DeviceBuffer::<f32>::from_host(&c).expect("d_c");

    let smem = (16 * 16 + 16 * 8) * 4 * 3; // (a_tile + b_tile) * f32 * stages
    let params = LaunchParams::new(1u32, 32u32).with_shared_mem(smem);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                16u32,
                8u32,
                16u32,
                1.0f32,
                0.0f32,
            ),
        )
        .expect("launch pipelined fma");
    stream
        .synchronize()
        .expect("sync — cp.async pipeline faulted on device");
}

/// The real Ampere `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` HMMA
/// variant must JIT-load and launch fault-free on the device with `half::f16`
/// inputs. This proves the tensor-core instruction itself executes on the
/// A4000 (sm_86). As above, the kernel is a pipeline skeleton, so only
/// fault-free execution is asserted — not a numeric GEMM result.
#[test]
fn pipelined_tensor_core_mma_f16_launches_on_device() {
    use half::f16;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    // The m16n8k16 f16 HMMA requires sm_80+. Skip on older silicon.
    if fx.sm < SmVersion::Sm80 {
        return;
    }
    let stream = Stream::new(&fx.ctx).expect("stream");

    let mut t = simt_template(PtxType::F16, PtxType::F32);
    t.target = fx.sm;
    t.tile_m = 16;
    t.tile_n = 8;
    t.tile_k = 16;
    t.warp_n = 8;
    t.stages = 3;
    t.use_tensor_core = true;
    let ptx = t.generate_pipelined_skeleton().expect("pipelined mma ptx");
    assert!(
        ptx.contains("mma.sync.aligned.m16n8k16"),
        "expected Ampere m16n8k16 HMMA in pipelined tensor-core PTX"
    );
    let kernel = load_kernel(&ptx, &entry_name(&ptx));

    let a = vec![f16::from_f32(1.0); 16 * 16];
    let b = vec![f16::from_f32(1.0); 16 * 8];
    // C is written in accumulator (f32) precision: st.global.f32.
    let c = vec![0.0f32; 16 * 8];
    let d_a = DeviceBuffer::<f16>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f16>::from_host(&b).expect("d_b");
    let d_c = DeviceBuffer::<f32>::from_host(&c).expect("d_c");

    let smem = (16 * 16 + 16 * 8) * 2 * 3; // (a_tile + b_tile) * f16 * stages
    let params = LaunchParams::new(1u32, 32u32).with_shared_mem(smem);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                16u32,
                8u32,
                16u32,
                1.0f32,
                0.0f32,
            ),
        )
        .expect("launch pipelined mma f16");
    stream
        .synchronize()
        .expect("sync — Ampere mma.sync HMMA faulted on device");
}
