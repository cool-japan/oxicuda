//! On-device GPU validation for the hand-written PTX kernels emitted by
//! `oxicuda-dnn`, run against the live CUDA device.
//!
//! Each submodule covers one subsystem. Tests either drive the crate's public
//! operation API (`DnnHandle` + device-buffer-backed `TensorDesc`s) or JIT the
//! kernel PTX directly via `Module::from_ptx`, launch on the real device, copy
//! results back, and assert equivalence to an independent CPU re-derivation.
//!
//! ## Honesty contract
//!
//! Kernels are classified (see the discovery inventory) as:
//! * **complete** — computes a numerically-correct result a CPU oracle can
//!   check → `numeric_cpu_oracle` test.
//! * **fragment** — assembles + launches but is a structural skeleton / single
//!   tile / proxy → `load_launch_only` test (assert it runs fault-free; do NOT
//!   assert a numeric result that would be wrong), UNLESS the test sizes inputs
//!   so a single-tile fragment is exact, in which case a numeric oracle is
//!   used and the constraint is documented.
//! * **hopper_blocked** — only emits sm_90+ PTX (wgmma/TMA/FP8 mma) → skipped
//!   on this sm_86 device; at most a `ptxas`/Hopper-target prescreen.
//!
//! Every test returns early (skips) when no CUDA device is present, so the
//! suite stays green on CPU-only machines.

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::Kernel;
use oxicuda_ptx::arch::SmVersion;

use crate::handle::DnnHandle;

// Subsystem submodules (one per discovery cluster).
mod attn;
mod conv_fprop;
mod conv_fused;
mod conv_other;
mod conv_winograd;
mod handle_sync;
mod moe_linear;
mod norm;
mod pool_resize;
mod quantize;
mod rnn_misc;

// ---------------------------------------------------------------------------
// Shared fixture & helpers
// ---------------------------------------------------------------------------

/// A live DNN handle (owns a CUDA context + stream) plus the device SM version.
pub(crate) struct GpuFixture {
    pub handle: DnnHandle,
    pub sm: SmVersion,
}

impl GpuFixture {
    /// The launch stream bound to the handle.
    pub fn stream(&self) -> &Stream {
        self.handle.stream()
    }
}

/// Acquire a GPU fixture, or `None` when no driver / device is present.
pub(crate) fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let dev = Device::get(0).ok()?;
    let (major, minor) = dev.compute_capability().ok()?;
    let sm = SmVersion::from_compute_capability(major, minor).unwrap_or(SmVersion::Sm80);
    let ctx = Arc::new(Context::new(&dev).ok()?);
    let handle = DnnHandle::new(&ctx).ok()?;
    Some(GpuFixture { handle, sm })
}

/// JIT-compile `ptx` and look up `entry`, returning a launchable kernel.
///
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real bug in
/// the kernel string, surfaced as a test failure rather than silently skipped.
pub(crate) fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx).unwrap_or_else(|e| {
        panic!(
            "PTX JIT compile failed for `{entry}`: {e}\n--- PTX (first 1200 chars) ---\n{}",
            &ptx[..ptx.len().min(1200)]
        )
    });
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

/// Extracts the first `.visible .entry NAME(` identifier from a PTX module.
pub(crate) fn entry_name(ptx: &str) -> String {
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

/// `ceil(n / d)` as a `u32` grid size (at least 1).
pub(crate) fn ceil_div(n: u32, d: u32) -> u32 {
    n.div_ceil(d).max(1)
}

/// Relative-with-absolute-floor closeness test for FP32.
pub(crate) fn close_f32(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Asserts two FP32 slices agree within tolerance, reporting the first mismatch.
pub(crate) fn assert_close_f32(gpu: &[f32], cpu: &[f32], rel: f32, abs: f32, tag: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{tag}: length mismatch");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            close_f32(g, c, rel, abs),
            "{tag}: element {i} mismatch gpu={g} cpu={c} (rel={rel:e} abs={abs:e})"
        );
    }
}

/// Relative L2 error `||got - want||_2 / ||want||_2` between an FP32 result and
/// an FP64 reference, falling back to the absolute norm when the reference is
/// identically zero.
///
/// The right metric for a kernel that computes the *same* mathematical result
/// by a *different* association of the sum (Winograd, split-K, any tree
/// reduction): such a kernel can differ from the reference by an unbounded
/// relative amount on a single catastrophically-cancelling element while the
/// tensor as a whole is accurate to several digits. An element-wise bound
/// either fails spuriously there or has to be loosened until it stops
/// detecting real bugs.
pub(crate) fn rel_l2_error(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "rel_l2_error: length mismatch");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        let d = f64::from(g) - w;
        num += d * d;
        den += w * w;
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

/// Asserts two FP64 slices agree within tolerance, reporting the first mismatch.
pub(crate) fn assert_close_f64(gpu: &[f64], cpu: &[f64], rel: f64, abs: f64, tag: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{tag}: length mismatch");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (g - c).abs() <= rel * g.abs().max(c.abs()) + abs,
            "{tag}: element {i} mismatch gpu={g} cpu={c} (rel={rel:e} abs={abs:e})"
        );
    }
}

/// A small deterministic LCG. Normalisation divides by `2^32` (never `2^31`).
pub(crate) struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    /// Uniform in `[0, 1)` via division by `2^32`.
    pub fn unit(&mut self) -> f64 {
        f64::from(self.next_u32()) / 4_294_967_296.0
    }
    /// Uniform `f32` in `[lo, hi)`.
    pub fn range_f32(&mut self, lo: f64, hi: f64) -> f32 {
        (lo + (hi - lo) * self.unit()) as f32
    }
    /// Uniform `f64` in `[lo, hi)`.
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Best-effort `ptxas` pre-screen: assembles `ptx` for its declared `.target`.
/// Returns `Ok(())` on success or when `ptxas` is unavailable, the captured
/// stderr on assembler rejection.
pub(crate) fn ptxas_assembles(ptx: &str, tag: &str) -> Result<(), String> {
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
    let in_path = dir.join(format!("oxidnn_{tag}_{stamp}.ptx"));
    let out_path = dir.join(format!("oxidnn_{tag}_{stamp}.cubin"));

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
