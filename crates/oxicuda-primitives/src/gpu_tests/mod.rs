//! On-device GPU validation for the generated PTX kernels in this crate.
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! [`oxicuda_driver::Module::from_ptx`], launches it on the real CUDA device
//! through `oxicuda-launch`, copies the results back, and asserts numerical
//! equivalence to a CPU oracle (the crate's own `host_reference` /
//! per-module `reference_*` functions where one exists, otherwise an
//! independent Rust re-derivation of the kernel's documented arithmetic).
//!
//! Every test returns early (skips) when no CUDA device is present, so the
//! suite stays green on CPU-only machines.
//!
//! ## PTX bugs found and fixed on a real RTX A4000 (sm_86)
//!
//! These kernels had **never** been compiled by `ptxas` before this campaign
//! (the crate's unit tests only string-match the generated PTX). Pre-screening
//! every entry with `ptxas -arch=sm_86` plus on-device numerical checks
//! surfaced several systemic bug classes; see the crate notes / commit message
//! for the full list. The headline ones:
//!
//! * **Shadowed `%tid` special register** — every kernel declared a user
//!   register `.reg .u32 %tid` and then read `mov.u32 %tid, %tid.x;`. Because
//!   the user register shadows the built-in `%tid`, `ptxas` parsed `%tid.x` as
//!   an (illegal) video selector and rejected *every* kernel. Fixed by renaming
//!   the user register to `%ltid`.
//! * **`mad.lo.u64` with 32-bit operands** — address and global-index math used
//!   `mad.lo.u64 %addr, %idx32, size, %base64`, mixing a 32-bit index into a
//!   64-bit multiply-add (ptxas: "Arguments mismatch"). Fixed to
//!   `mad.wide.u32` for address math and `cvt.u64.u32` + `mad.wide.u32` for the
//!   `blockIdx*blockDim + tid` global index.
//! * **Invalid f32/f64 immediates** — min/max reduction identities and sort
//!   sentinels were written `0x7F800000` / `0x7FF0…`, which `mov.f32`/`mov.f64`
//!   reject; fixed to the PTX float-literal forms `0f7F800000` / `0d7FF0…`.
//! * **`cvt.u32.pred`** — bitonic sort materialised a predicate into an int via
//!   the non-existent `cvt.u32.pred`; fixed to `selp.u32 d, 1, 0, p`.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_ptx::arch::SmVersion;

mod reduce_scan;
mod sort;
mod stream_compaction;
mod warp_vec;

// ---------------------------------------------------------------------------
// Shared fixture + helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
pub(crate) struct GpuFixture {
    pub(crate) ctx: Arc<Context>,
    pub(crate) sm: SmVersion,
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
    let ctx = Context::new(&dev).ok()?;
    Some(GpuFixture {
        ctx: Arc::new(ctx),
        sm,
    })
}

impl GpuFixture {
    pub(crate) fn stream(&self) -> Stream {
        Stream::new(&self.ctx).expect("create stream")
    }
}

/// JIT-compile `ptx` and look up `entry`, returning a launchable kernel.
///
/// A `Module::from_ptx` failure means ptxas rejected the PTX — surfaced as a
/// test failure (panic) rather than silently skipped.
pub(crate) fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx)
        .unwrap_or_else(|e| panic!("PTX JIT compile failed for `{entry}`: {e}\n{ptx}"));
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

/// `ceil(n / block)` as a 1-D grid size (minimum 1).
pub(crate) fn grid_1d(n: u32, block: u32) -> u32 {
    n.div_ceil(block).max(1)
}

/// A small, deterministic linear-congruential generator for test data.
///
/// Normalisation divides by `2^32` (never `2^31`), matching the workspace RNG
/// policy. Only used to fabricate reproducible input vectors for the oracles.
pub(crate) struct Lcg {
    state: u64,
}

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Next raw `u32`.
    pub(crate) fn next_u32(&mut self) -> u32 {
        // Numerical Recipes LCG constants, take the high 32 bits.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// `u32` in `[0, bound)`.
    pub(crate) fn below(&mut self, bound: u32) -> u32 {
        self.next_u32() % bound
    }

    /// `f32` in `[lo, hi)` (normalised by `2^32`).
    pub(crate) fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = f64::from(self.next_u32()) / 4_294_967_296.0_f64; // 2^32
        (f64::from(lo) + f64::from(hi - lo) * unit) as f32
    }
}

/// Map a kernel's [`SmVersion`] for launch params: `(grid, block)` shorthand.
pub(crate) fn params(grid: u32, block: u32) -> LaunchParams {
    LaunchParams::new(grid, block)
}
