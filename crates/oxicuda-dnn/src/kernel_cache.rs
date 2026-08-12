//! Handle-owned cache of JIT-compiled kernels, plus occupancy-aware launch
//! geometry derived from the compiled code object.
//!
//! # Why this exists
//!
//! Every DNN operation in this crate generates PTX text and hands it to
//! [`Module::from_ptx`], which is a real `cuModuleLoadData` JIT compile. Doing
//! that on *every* call is fine for a train-once workload and ruinous for an
//! inference pipeline that re-invokes the same handful of kernels once per
//! video frame: the JIT tax (measured at ~194 us per call on an RTX A4000,
//! driver 550.144.03) dwarfs the kernels themselves, which typically run in
//! single-digit microseconds at batch 1.
//!
//! [`KernelCache`] gives each [`DnnHandle`](crate::handle::DnnHandle) an
//! in-memory map from a *code-generation key* to the compiled
//! [`Module`]/[`Kernel`], so a repeated call is a hash lookup and an
//! `Arc` clone instead of a JIT compile. It is the DNN analogue of
//! `BlasHandle::get_or_compile_module`, which already does exactly this for the
//! non-GEMM BLAS routines.
//!
//! # The keying contract (read before adding a call site)
//!
//! The cache key must capture **everything the generated PTX depends on**. Most
//! engines in this crate fold their code-gen constants into the kernel entry
//! name, in which case the entry name *is* a sufficient key. Some do not — for
//! example the forward convolution kernels bake channels-per-group and the
//! filter extent into the instruction stream as immediates, and (before this
//! module existed) named the entry only after the precision and layout. Keying
//! such a kernel by name alone would hand back a module compiled for a
//! *different* problem shape and silently produce wrong results.
//!
//! Two remedies are acceptable, and both are used in this crate:
//!
//! 1. Extend the entry name so it discriminates every code-gen constant
//!    (preferred — it also makes the PTX self-describing).
//! 2. Pass a key that is a strict superset of the entry name via
//!    [`KernelCache::get_or_compile_kernel`]'s `key` parameter.
//!
//! Because an engine's `sm_version` is supplied at construction and is *not*
//! required to match the handle's device, the target architecture is folded
//! into the key by [`cache_key`] as well: the same entry name targeting
//! `sm_80` and `sm_86` are different modules.
//!
//! # Occupancy-aware launch configuration
//!
//! Historically every convolution engine hard-coded `block_size = 256` and
//! derived the grid from it. That ignores the kernel's actual register and
//! shared-memory footprint, which is only knowable once the code object exists.
//! Now that the compiled [`Kernel`] is cached, the driver's
//! `cuOccupancyMaxPotentialBlockSize` can be queried against the real
//! [`Function`](oxicuda_driver::module::Function) — once per compiled kernel,
//! memoised in [`CachedKernel`] — and the answer reused on every launch.
//!
//! See [`elementwise_block_size`] for the exact policy and its rationale.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::arch::SmVersion;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Launch-configuration policy constants
// ---------------------------------------------------------------------------

/// Threads-per-block used when no occupancy answer is available.
///
/// This is the value every conv engine used to hard-code. It is retained as
/// the fallback so that a device or driver that refuses the occupancy query
/// reproduces the previous, known-good launch geometry exactly rather than
/// inventing a new one.
pub(crate) const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Warp-aligned block sizes the launch configurator will choose from,
/// descending.
///
/// The driver's suggestion is snapped *down* onto this ladder. Every entry is
/// a multiple of the 32-thread warp, so no candidate can produce a partially
/// populated warp in every block, and halving any entry lands on another entry
/// (which is what makes the spread loop in [`elementwise_block_size`] safe).
const BLOCK_LADDER: [u32; 6] = [1024, 512, 256, 128, 64, 32];

/// Smallest block the "spread across SMs" loop will shrink to.
///
/// Below two warps per block the scheduler loses more to per-block launch
/// overhead than it gains from extra block-level parallelism, so the loop
/// stops here even if the problem is still too small to fill the device.
const SPREAD_FLOOR: u32 = 64;

// ---------------------------------------------------------------------------
// Cache keys
// ---------------------------------------------------------------------------

/// Builds a cache key from a kernel entry name and the SM version its PTX was
/// generated for.
///
/// The entry name must already discriminate every *other* code-generation
/// constant baked into the PTX (see the module docs). The architecture is
/// appended here because engines carry their own `sm_version`, independent of
/// the handle they are executed against.
#[must_use]
pub(crate) fn cache_key(entry: &str, sm: SmVersion) -> String {
    format!("{entry}@{}", sm.as_ptx_str())
}

/// Builds a cache key for a kernel whose entry name does **not** encode every
/// code-generation constant, by appending the missing ones as `extra`.
///
/// This is remedy (2) from the module docs, used where renaming the entry point
/// would churn a widely-referenced public kernel name for no functional gain.
/// `extra` must list every code-generation input the name omits — and only
/// discrete, bounded ones: a key that varies with a per-call runtime quantity
/// (a sequence length, an element count) would compile and retain a fresh
/// module on every call, which is worse than the JIT it replaces.
#[must_use]
pub(crate) fn cache_key_with(entry: &str, sm: SmVersion, extra: &str) -> String {
    format!("{entry}[{extra}]@{}", sm.as_ptx_str())
}

// ---------------------------------------------------------------------------
// CachedKernel
// ---------------------------------------------------------------------------

/// A JIT-compiled kernel plus its memoised occupancy hint.
///
/// Handed out as an `Arc` so a launch site can hold it across a call without
/// keeping the cache locked.
pub(crate) struct CachedKernel {
    /// The launchable kernel; keeps its parent module loaded.
    kernel: Kernel,
    /// `(min_grid_size, block_size)` from `cuOccupancyMaxPotentialBlockSize`
    /// for a zero-dynamic-shared-memory launch, or `None` if the driver
    /// declined to answer. Queried at most once per compiled kernel.
    occupancy: OnceLock<Option<(u32, u32)>>,
}

impl CachedKernel {
    /// Wraps a freshly looked-up kernel; the occupancy query is deferred to
    /// first use.
    fn new(kernel: Kernel) -> Self {
        Self {
            kernel,
            occupancy: OnceLock::new(),
        }
    }

    /// The underlying launchable kernel.
    #[inline]
    pub(crate) fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    /// The memoised `(min_grid_size, block_size)` occupancy suggestion for a
    /// launch with no dynamic shared memory.
    ///
    /// A driver failure is cached as `None` (rather than retried per call) so
    /// a device that cannot answer costs one query, not one per launch.
    fn occupancy_hint(&self) -> Option<(u32, u32)> {
        *self.occupancy.get_or_init(|| {
            let (min_grid, block) = self.kernel.optimal_block_size(0).ok()?;
            let min_grid = u32::try_from(min_grid).ok()?;
            let block = u32::try_from(block).ok()?;
            if block == 0 {
                None
            } else {
                Some((min_grid, block))
            }
        })
    }

    /// Occupancy-aware 1-D launch parameters for a kernel that assigns one
    /// thread per element and guards itself with `if gid < total`.
    ///
    /// # Grid coverage
    ///
    /// The grid is always `total.div_ceil(block)`, so `grid * block >= total`
    /// for every block size this can select: no output element can be left
    /// uncovered regardless of what the occupancy query returns. The kernel's
    /// own `gid < total` guard retires the surplus threads in the last block.
    /// **This invariant is the whole safety argument for varying the block
    /// size — do not use this helper for a kernel that assumes a particular
    /// `blockDim.x`** (e.g. one sizing a shared-memory reduction buffer from
    /// it, or one written as a fixed number of tiles).
    pub(crate) fn launch_1d(&self, total: u32) -> LaunchParams {
        let block = elementwise_block_size(total, self.occupancy_hint());
        LaunchParams::new(grid_size_for(total, block), block)
    }
}

// ---------------------------------------------------------------------------
// Launch configuration policy
// ---------------------------------------------------------------------------

/// Chooses threads-per-block for a one-thread-per-element launch of `total`
/// elements, given the driver's `(min_grid_size, block_size)` occupancy hint.
///
/// # Policy
///
/// 1. **No hint** (query unsupported or failed) → [`DEFAULT_BLOCK_SIZE`], i.e.
///    exactly the legacy hard-coded geometry.
/// 2. **Snap down** the driver's suggestion onto [`BLOCK_LADDER`]. The driver
///    may return a non-power-of-two such as 768; halving that repeatedly would
///    eventually leave the warp grid (768 → 384 → 192 → 96 → 48), so the value
///    is first snapped to the largest ladder entry that does not exceed it.
/// 3. **Spread small problems.** `min_grid_size` is the block count needed to
///    saturate every SM at that block size. If the problem cannot produce that
///    many blocks, halve the block size (which raises the block count without
///    changing the total thread count, so more SMs get work) until it can, or
///    until [`SPREAD_FLOOR`] is reached. Without this step a small
///    convolution — common at batch 1 — could be handed a 1024-thread block
///    and collapse onto a single SM, which would be markedly *slower* than the
///    256 this replaces.
///
/// The returned value is always a multiple of 32 and in `1..=1024`, so it is a
/// legal `blockDim.x` on every supported architecture.
#[must_use]
pub(crate) fn elementwise_block_size(total: u32, hint: Option<(u32, u32)>) -> u32 {
    let Some((min_grid, opt_block)) = hint else {
        return DEFAULT_BLOCK_SIZE;
    };

    // Snap the driver's answer down onto the warp-aligned ladder. A suggestion
    // below one warp (which no real driver returns) falls back to a full warp.
    let mut block = BLOCK_LADDER
        .into_iter()
        .find(|&candidate| candidate <= opt_block)
        .unwrap_or(32);

    while block > SPREAD_FLOOR && grid_size_for(total, block) < min_grid {
        block /= 2;
    }

    block
}

// ---------------------------------------------------------------------------
// KernelCache
// ---------------------------------------------------------------------------

/// Separator between the module key and the entry name inside a kernel-cache
/// key. ASCII unit separator: it cannot occur in a generated kernel name, so a
/// composed key can never collide with a module key.
const KEY_SEP: char = '\u{1f}';

/// In-memory cache of JIT-compiled modules and the kernels looked up from them.
///
/// Owned by a [`DnnHandle`](crate::handle::DnnHandle), so entries live exactly
/// as long as the handle (and therefore the CUDA context the modules were
/// loaded into).
pub(crate) struct KernelCache {
    /// Compiled modules, keyed by code-generation key.
    modules: RwLock<HashMap<String, Arc<Module>>>,
    /// Kernels, keyed by `"{module_key}{KEY_SEP}{entry}"`. Caching these too
    /// saves a `cuModuleGetFunction` per launch and — more importantly — gives
    /// the occupancy answer somewhere stable to be memoised.
    kernels: RwLock<HashMap<String, Arc<CachedKernel>>>,
}

impl KernelCache {
    /// Creates an empty cache.
    pub(crate) fn new() -> Self {
        Self {
            modules: RwLock::new(HashMap::new()),
            kernels: RwLock::new(HashMap::new()),
        }
    }

    /// Returns the compiled [`Module`] for `key`, generating and JIT-compiling
    /// its PTX on a miss.
    ///
    /// `gen_ptx` is invoked **only** on a cache miss, so PTX generation cost is
    /// also paid once per key rather than per call.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::LaunchFailed`] if the cache lock is poisoned,
    /// whatever `gen_ptx` returns on generation failure, or
    /// [`DnnError::Cuda`] if the PTX fails to JIT-compile.
    pub(crate) fn get_or_compile_module(
        &self,
        key: &str,
        gen_ptx: impl FnOnce() -> DnnResult<String>,
    ) -> DnnResult<Arc<Module>> {
        // Fast path: shared read lock, hand back the existing module.
        {
            let cache = self.modules.read().map_err(|_| poisoned("module"))?;
            if let Some(module) = cache.get(key) {
                return Ok(Arc::clone(module));
            }
        }

        // Slow path: generate and compile *outside* the write lock so a long
        // JIT never blocks other lookups. A concurrent compile of the same key
        // is harmless — the map keeps whichever module was inserted last, and
        // every `Arc` already handed out stays valid for as long as its holder
        // keeps it.
        let ptx = gen_ptx()?;
        let module = Arc::new(Module::from_ptx(&ptx)?);
        {
            let mut cache = self.modules.write().map_err(|_| poisoned("module"))?;
            cache.insert(key.to_owned(), Arc::clone(&module));
        }
        Ok(module)
    }

    /// Returns the [`CachedKernel`] for entry point `entry` within the module
    /// identified by `key`, compiling the module on a miss.
    ///
    /// `key` must discriminate every code-generation constant baked into the
    /// PTX — see the module-level docs. Use [`cache_key`] to fold in the target
    /// architecture.
    ///
    /// # Errors
    ///
    /// As [`Self::get_or_compile_module`], plus [`DnnError::Cuda`] if `entry`
    /// is not present in the compiled module.
    pub(crate) fn get_or_compile_kernel(
        &self,
        key: &str,
        entry: &str,
        gen_ptx: impl FnOnce() -> DnnResult<String>,
    ) -> DnnResult<Arc<CachedKernel>> {
        let kernel_key = format!("{key}{KEY_SEP}{entry}");
        {
            let cache = self.kernels.read().map_err(|_| poisoned("kernel"))?;
            if let Some(kernel) = cache.get(&kernel_key) {
                return Ok(Arc::clone(kernel));
            }
        }

        let module = self.get_or_compile_module(key, gen_ptx)?;
        let kernel = Arc::new(CachedKernel::new(Kernel::from_module(module, entry)?));
        {
            let mut cache = self.kernels.write().map_err(|_| poisoned("kernel"))?;
            cache.insert(kernel_key, Arc::clone(&kernel));
        }
        Ok(kernel)
    }

    /// Number of distinct modules currently compiled and held.
    ///
    /// Used by the on-device tests to prove that a repeated call is a cache
    /// hit rather than a re-compile.
    #[cfg(all(test, feature = "gpu-tests"))]
    pub(crate) fn module_count(&self) -> usize {
        self.modules.read().map(|c| c.len()).unwrap_or(0)
    }
}

impl Default for KernelCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds the error reported when a cache lock has been poisoned by a panic in
/// another thread.
fn poisoned(which: &str) -> DnnError {
    DnnError::LaunchFailed(format!("{which} cache lock poisoned"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole safety argument for varying the block size: whatever block
    /// size the policy picks, `ceil(total / block) * block >= total`, so every
    /// element is covered by some thread.
    #[test]
    fn grid_always_covers_every_element() {
        let totals = [
            0u32,
            1,
            31,
            32,
            33,
            63,
            64,
            65,
            255,
            256,
            257,
            1000,
            1024,
            1025,
            4095,
            100_000,
            1_638_400,
            16_777_216,
            u32::MAX,
        ];
        let hints = [
            None,
            Some((1u32, 1024u32)),
            Some((48, 1024)),
            Some((384, 1024)),
            Some((96, 512)),
            Some((192, 256)),
            Some((768, 128)),
            Some((1, 64)),
            Some((100_000, 1024)),
            Some((u32::MAX, 1024)),
        ];
        for total in totals {
            for hint in hints {
                let block = elementwise_block_size(total, hint);
                let grid = grid_size_for(total, block);
                assert!(
                    u64::from(grid) * u64::from(block) >= u64::from(total),
                    "under-coverage: total={total} hint={hint:?} block={block} grid={grid}"
                );
            }
        }
    }

    /// Every selected block size must be a legal `blockDim.x`: warp-aligned,
    /// non-zero, and within the 1024-thread hardware limit.
    #[test]
    fn block_size_is_always_a_legal_warp_multiple() {
        let hints = [
            None,
            Some((1u32, 1u32)),
            Some((1, 31)),
            Some((1, 32)),
            Some((1, 96)),
            Some((1, 768)),
            Some((1, 1024)),
            Some((10_000, 1024)),
        ];
        for hint in hints {
            for total in [0u32, 1, 512, 1_000_000] {
                let block = elementwise_block_size(total, hint);
                assert!(block > 0 && block <= 1024, "illegal block {block}");
                assert_eq!(block % 32, 0, "block {block} is not warp-aligned");
            }
        }
    }

    /// With no occupancy answer the configurator must reproduce the legacy
    /// hard-coded geometry byte for byte, so an unsupported driver cannot
    /// change behaviour.
    #[test]
    fn missing_hint_reproduces_legacy_geometry() {
        for total in [1u32, 255, 256, 257, 1_638_400] {
            assert_eq!(elementwise_block_size(total, None), 256);
            assert_eq!(
                grid_size_for(total, elementwise_block_size(total, None)),
                grid_size_for(total, 256)
            );
        }
    }

    /// A problem far too small to fill the device must not be collapsed onto a
    /// single huge block: the spread step trades block size for block count.
    #[test]
    fn small_problem_is_spread_across_sms() {
        // 4096 elements, driver suggests 1024 threads and wants >= 384 blocks.
        let block = elementwise_block_size(4096, Some((384, 1024)));
        assert_eq!(block, SPREAD_FLOOR, "expected shrink to the spread floor");
        assert_eq!(grid_size_for(4096, block), 64);
    }

    /// A problem large enough to saturate the device keeps the driver's
    /// occupancy-optimal block size.
    #[test]
    fn large_problem_keeps_the_occupancy_optimum() {
        // 1.6M elements at 1024 threads is 1600 blocks, comfortably above the
        // 384 the driver asked for.
        assert_eq!(elementwise_block_size(1_638_400, Some((384, 1024))), 1024);
    }

    /// A non-power-of-two suggestion snaps *down* onto the ladder — never up,
    /// which could exceed the resources the driver said were available.
    #[test]
    fn suggestion_snaps_down_to_the_ladder() {
        assert_eq!(elementwise_block_size(1_000_000, Some((1, 768))), 512);
        assert_eq!(elementwise_block_size(1_000_000, Some((1, 300))), 256);
        assert_eq!(elementwise_block_size(1_000_000, Some((1, 1023))), 512);
        assert_eq!(elementwise_block_size(1_000_000, Some((1, 1024))), 1024);
    }

    /// Keys must separate the same entry name compiled for different targets.
    #[test]
    fn cache_key_discriminates_sm_version() {
        assert_ne!(
            cache_key("conv1x1_f32_nchw", SmVersion::Sm80),
            cache_key("conv1x1_f32_nchw", SmVersion::Sm86)
        );
        assert!(cache_key("k", SmVersion::Sm86).starts_with("k@"));
    }
}
