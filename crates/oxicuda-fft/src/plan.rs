//! FFT plan creation and strategy selection.
//!
//! An [`FftPlan`] describes the size, transform type, batch count, and
//! precision of an FFT, together with the decomposition strategy (radix
//! factorisation) and compiled GPU kernels.  Plans are created once and
//! executed many times.
#![allow(dead_code)]

use std::sync::Arc;

use oxicuda_driver::Module;

use crate::error::{FftError, FftResult};
use crate::types::{FftDirection, FftPrecision, FftType};

// ---------------------------------------------------------------------------
// FftStrategy — how the FFT size is decomposed
// ---------------------------------------------------------------------------

/// Describes how an FFT of size N is decomposed into radix stages.
///
/// For example, N = 1024 might be factorised as `[8, 8, 4, 4]`,
/// meaning four Stockham stages with radices 8, 8, 4, 4.
#[derive(Debug, Clone)]
pub struct FftStrategy {
    /// Radices for each stage (innermost first).
    pub radices: Vec<u32>,
    /// Strides for each stage (cumulative product of previous radices).
    pub strides: Vec<u32>,
    /// If `true`, the entire FFT fits in a single kernel using shared memory.
    /// Typically true for N <= 4096.
    pub single_kernel: bool,
}

/// Describes which kernel variant should be used for execution.
///
/// The planner selects the best variant based on FFT size, batch count,
/// and whether the size is a power of 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelVariant {
    /// Standard Stockham kernel (default).
    Stockham,
    /// Bank-conflict-free Stockham (power-of-2, N >= 64, N <= 4096).
    BankConflictFree,
    /// Fused batched kernel (N <= 1024, large batch count).
    FusedBatch,
    /// Per-stage kernels for large FFTs (N > 4096).
    MultiPass,
}

// ---------------------------------------------------------------------------
// CompiledFftKernel — a ready-to-launch GPU kernel
// ---------------------------------------------------------------------------

/// A compiled PTX kernel ready for launch.
#[derive(Clone)]
pub struct CompiledFftKernel {
    /// The loaded CUDA module containing the kernel.
    pub module: Arc<Module>,
    /// The entry-point function name within the module.
    pub function_name: String,
    /// Bytes of dynamic shared memory required.
    pub shared_mem_bytes: u32,
    /// Number of threads per block.
    pub block_size: u32,
}

impl std::fmt::Debug for CompiledFftKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledFftKernel")
            .field("function_name", &self.function_name)
            .field("shared_mem_bytes", &self.shared_mem_bytes)
            .field("block_size", &self.block_size)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// FftPlan — the main plan object
// ---------------------------------------------------------------------------

/// A fully specified FFT execution plan.
///
/// The plan holds the transform parameters, the chosen decomposition
/// strategy, and (after compilation) the GPU kernels needed to execute
/// the transform.
#[derive(Debug, Clone)]
pub struct FftPlan {
    /// FFT sizes along each dimension (1-D has one element, 2-D has two, etc.).
    pub sizes: Vec<usize>,
    /// The kind of transform (C2C, R2C, C2R).
    pub transform_type: FftType,
    /// Number of independent transforms to perform.
    pub batch: usize,
    /// Default direction (can be overridden at execution time for C2C).
    pub direction_default: FftDirection,
    /// Floating-point precision.
    pub precision: FftPrecision,
    /// Decomposition strategy for the first (or only) dimension.
    pub strategy: FftStrategy,
    /// Compiled GPU kernels, one per stage (or a single kernel for small N).
    pub compiled_kernels: Vec<CompiledFftKernel>,
    /// Size of temporary device buffer required (in bytes).
    pub temp_buffer_bytes: usize,
    /// Selected kernel variant for this plan.
    pub kernel_variant: KernelVariant,
}

impl FftPlan {
    /// Creates a 1-D FFT plan.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::InvalidSize`] if `n` is zero or cannot be
    /// factorised into supported radices.
    /// Returns [`FftError::InvalidBatch`] if `batch` is zero.
    pub fn new_1d(n: usize, transform_type: FftType, batch: usize) -> FftResult<Self> {
        validate_size(n)?;
        validate_batch(batch)?;

        let strategy = plan_strategy(n)?;
        let temp_buffer_bytes =
            compute_temp_buffer_bytes(n, batch, transform_type, FftPrecision::Single);
        let kernel_variant = select_kernel_variant(n, batch);

        Ok(Self {
            sizes: vec![n],
            transform_type,
            batch,
            direction_default: FftDirection::Forward,
            precision: FftPrecision::Single,
            strategy,
            compiled_kernels: Vec::new(),
            temp_buffer_bytes,
            kernel_variant,
        })
    }

    /// Creates a 2-D FFT plan.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::InvalidSize`] if either dimension is zero.
    /// Returns [`FftError::InvalidBatch`] if `batch` is zero.
    pub fn new_2d(nx: usize, ny: usize, transform_type: FftType, batch: usize) -> FftResult<Self> {
        validate_size(nx)?;
        validate_size(ny)?;
        validate_batch(batch)?;

        let strategy = plan_strategy(nx)?;
        let total_elements = nx * ny * batch;
        let temp_buffer_bytes = total_elements * FftPrecision::Single.complex_bytes();
        let kernel_variant = select_kernel_variant(nx, batch);

        Ok(Self {
            sizes: vec![nx, ny],
            transform_type,
            batch,
            direction_default: FftDirection::Forward,
            precision: FftPrecision::Single,
            strategy,
            compiled_kernels: Vec::new(),
            temp_buffer_bytes,
            kernel_variant,
        })
    }

    /// Creates a 3-D FFT plan.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::InvalidSize`] if any dimension is zero.
    /// Returns [`FftError::InvalidBatch`] if `batch` is zero.
    pub fn new_3d(
        nx: usize,
        ny: usize,
        nz: usize,
        transform_type: FftType,
        batch: usize,
    ) -> FftResult<Self> {
        validate_size(nx)?;
        validate_size(ny)?;
        validate_size(nz)?;
        validate_batch(batch)?;

        let strategy = plan_strategy(nx)?;
        let total_elements = nx * ny * nz * batch;
        let temp_buffer_bytes = total_elements * FftPrecision::Single.complex_bytes();
        let kernel_variant = select_kernel_variant(nx, batch);

        Ok(Self {
            sizes: vec![nx, ny, nz],
            transform_type,
            batch,
            direction_default: FftDirection::Forward,
            precision: FftPrecision::Single,
            strategy,
            compiled_kernels: Vec::new(),
            temp_buffer_bytes,
            kernel_variant,
        })
    }

    /// Sets the precision for this plan.
    pub fn with_precision(mut self, precision: FftPrecision) -> Self {
        self.precision = precision;
        // Recompute temp buffer with new precision
        self.temp_buffer_bytes = self.estimated_workspace_bytes();
        self
    }

    /// Sets the default direction for this plan.
    pub fn with_direction(mut self, direction: FftDirection) -> Self {
        self.direction_default = direction;
        self
    }

    /// Returns the estimated workspace size in bytes.
    ///
    /// For single-kernel plans this is zero (everything fits in shared memory).
    /// For multi-pass plans this equals one full-sized complex buffer.
    pub fn estimated_workspace_bytes(&self) -> usize {
        if self.strategy.single_kernel {
            return 0;
        }
        let total_elements: usize = self.sizes.iter().product::<usize>() * self.batch;
        total_elements * self.precision.complex_bytes()
    }

    /// Returns the number of dimensions (1, 2, or 3).
    pub fn ndim(&self) -> usize {
        self.sizes.len()
    }

    /// Returns the total number of complex elements per batch.
    pub fn elements_per_batch(&self) -> usize {
        self.sizes.iter().product()
    }
}

// ---------------------------------------------------------------------------
// Strategy planning
// ---------------------------------------------------------------------------

/// Determine the decomposition strategy for a 1-D FFT of size `n`.
///
/// Preferred radices are 8, 4, 2 (powers of 2), then 3, 5, 7 for
/// mixed-radix support.  If `n` cannot be fully factorised, the plan
/// falls back to Bluestein (recorded as a single "radix" equal to `n`).
fn plan_strategy(n: usize) -> FftResult<FftStrategy> {
    let radices = factorize(n);

    // Check if all factors are supported
    let all_supported = radices.iter().all(|&r| matches!(r, 2 | 3 | 4 | 5 | 7 | 8));

    if radices.is_empty() {
        return Err(FftError::InvalidSize(format!(
            "cannot factorise size {n} into supported radices"
        )));
    }

    let single_kernel = n <= 4096 && all_supported;

    // Compute strides (cumulative product of radices so far)
    let mut strides = Vec::with_capacity(radices.len());
    let mut stride: u32 = 1;
    for &r in &radices {
        strides.push(stride);
        stride = stride.saturating_mul(r);
    }

    Ok(FftStrategy {
        radices,
        strides,
        single_kernel,
    })
}

/// Factorize `n` into a sequence of small radices.
///
/// Preferred order: 8, 4, 2, 3, 5, 7.  If `n` has a prime factor
/// larger than 7, it is included as-is (requiring Bluestein).
fn factorize(mut n: usize) -> Vec<u32> {
    let mut factors = Vec::new();

    // Extract factors of 8 first (radix-8 butterfly)
    while n % 8 == 0 {
        factors.push(8);
        n /= 8;
    }

    // Extract factors of 4
    while n % 4 == 0 {
        factors.push(4);
        n /= 4;
    }

    // Extract factors of 2
    while n % 2 == 0 {
        factors.push(2);
        n /= 2;
    }

    // Extract factors of 3
    while n % 3 == 0 {
        factors.push(3);
        n /= 3;
    }

    // Extract factors of 5
    while n % 5 == 0 {
        factors.push(5);
        n /= 5;
    }

    // Extract factors of 7
    while n % 7 == 0 {
        factors.push(7);
        n /= 7;
    }

    // If anything remains, it is a prime > 7; include it as-is
    if n > 1 {
        #[allow(clippy::cast_possible_truncation)]
        factors.push(n as u32);
    }

    factors
}

/// Selects the best kernel variant for the given FFT size and batch count.
///
/// Priority:
/// 1. **FusedBatch**: N <= 1024 and batch >= 2 (amortise launch overhead)
/// 2. **BankConflictFree**: N is power of 2, 64 <= N <= 4096 (eliminate bank conflicts)
/// 3. **MultiPass**: N > 4096 (requires per-stage global memory)
/// 4. **Stockham**: fallback for all other cases
fn select_kernel_variant(n: usize, batch: usize) -> KernelVariant {
    use crate::kernels::bank_conflict_free::should_use_bcf;
    use crate::kernels::fused_batch::should_use_fused_batch;

    if should_use_fused_batch(n) && batch >= 2 {
        return KernelVariant::FusedBatch;
    }
    if should_use_bcf(n) {
        return KernelVariant::BankConflictFree;
    }
    if n > 4096 {
        return KernelVariant::MultiPass;
    }
    KernelVariant::Stockham
}

/// Compute the temporary buffer size for a 1-D FFT.
fn compute_temp_buffer_bytes(
    n: usize,
    batch: usize,
    _transform_type: FftType,
    precision: FftPrecision,
) -> usize {
    // For single-kernel (small N) plans, no temp buffer is needed.
    if n <= 4096 {
        return 0;
    }
    n * batch * precision.complex_bytes()
}

/// Validate that `n` is a legal FFT size.
fn validate_size(n: usize) -> FftResult<()> {
    if n == 0 {
        return Err(FftError::InvalidSize("FFT size must be > 0".to_string()));
    }
    Ok(())
}

/// Validate that `batch` is a legal batch count.
fn validate_batch(batch: usize) -> FftResult<()> {
    if batch == 0 {
        return Err(FftError::InvalidBatch(
            "batch count must be >= 1".to_string(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorize_power_of_2() {
        let f = factorize(1024);
        let product: u32 = f.iter().product();
        assert_eq!(product, 1024);
        assert!(f.iter().all(|&r| matches!(r, 2 | 4 | 8)));
    }

    #[test]
    fn factorize_mixed() {
        let f = factorize(360); // 8 * 45 = 8 * 9 * 5 => 8, 3, 3, 5
        let product: u32 = f.iter().product();
        assert_eq!(product, 360);
    }

    #[test]
    fn factorize_prime() {
        let f = factorize(13);
        assert_eq!(f, vec![13]);
    }

    #[test]
    fn plan_1d_basic() {
        let plan = FftPlan::new_1d(256, FftType::C2C, 1);
        assert!(plan.is_ok());
        let plan = plan.ok();
        assert!(plan.is_some());
        let plan = plan.map(|p| {
            assert_eq!(p.sizes, vec![256]);
            assert!(p.strategy.single_kernel);
        });
        let _ = plan;
    }

    #[test]
    fn plan_1d_zero_size() {
        let result = FftPlan::new_1d(0, FftType::C2C, 1);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn plan_1d_zero_batch() {
        let result = FftPlan::new_1d(256, FftType::C2C, 0);
        assert!(matches!(result, Err(FftError::InvalidBatch(_))));
    }

    #[test]
    fn plan_2d() {
        let plan = FftPlan::new_2d(64, 64, FftType::C2C, 1);
        assert!(plan.is_ok());
    }

    #[test]
    fn plan_3d() {
        let plan = FftPlan::new_3d(32, 32, 32, FftType::C2C, 1);
        assert!(plan.is_ok());
    }

    #[test]
    fn workspace_single_kernel() {
        let plan = FftPlan::new_1d(256, FftType::C2C, 1);
        let plan = plan.ok();
        assert!(plan.is_some());
        if let Some(p) = plan {
            assert_eq!(p.estimated_workspace_bytes(), 0);
        }
    }

    #[test]
    fn workspace_large() {
        let plan = FftPlan::new_1d(8192, FftType::C2C, 1);
        let plan = plan.ok();
        assert!(plan.is_some());
        if let Some(p) = plan {
            assert!(p.estimated_workspace_bytes() > 0);
        }
    }
}
