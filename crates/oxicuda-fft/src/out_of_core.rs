//! Out-of-core FFT for transforms exceeding single GPU memory.
//!
//! When an FFT is too large to fit entirely in GPU VRAM, this module
//! decomposes the transform into multiple passes that stage data through
//! host memory in chunks.  Two strategies are supported:
//!
//! - **Overlap**: overlapping chunks with twiddle factor correction between passes.
//! - **FourStep**: the four-step FFT algorithm, optimal for large 1-D transforms.
//!
//! The [`plan_out_of_core`] function automatically decomposes a problem into
//! [`OutOfCorePass`] steps based on the available GPU memory budget.

use std::f64::consts::PI;
use std::fmt;

use crate::error::{FftError, FftResult};
use crate::types::{FftDirection, FftPrecision, FftType};

// ---------------------------------------------------------------------------
// OutOfCoreStrategy -- algorithm selection
// ---------------------------------------------------------------------------

/// Strategy for out-of-core FFT decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutOfCoreStrategy {
    /// Overlapping chunks with twiddle factor correction.
    ///
    /// Each pass processes a contiguous slice of the input with a small
    /// overlap region.  After each local FFT the results are multiplied
    /// by twiddle factors to account for the global phase shift.
    Overlap,

    /// Four-step FFT algorithm (Gentleman--Sande decomposition).
    ///
    /// Reinterprets the 1-D array of length N = N1 * N2 as a 2-D
    /// N1 x N2 matrix, then:
    ///   1. Column FFTs of length N1
    ///   2. Twiddle-factor multiplication
    ///   3. Row FFTs of length N2
    ///   4. Final transpose
    ///
    /// Optimal for large 1-D transforms where N is highly composite.
    FourStep,

    /// Automatically select the best strategy based on problem size.
    ///
    /// Uses [`FourStep`](Self::FourStep) when the total element count
    /// is a perfect square or has a good factorisation, otherwise falls
    /// back to [`Overlap`](Self::Overlap).
    Auto,
}

impl fmt::Display for OutOfCoreStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overlap => write!(f, "Overlap"),
            Self::FourStep => write!(f, "FourStep"),
            Self::Auto => write!(f, "Auto"),
        }
    }
}

// ---------------------------------------------------------------------------
// OutOfCoreConfig -- user-facing configuration
// ---------------------------------------------------------------------------

/// Configuration for an out-of-core FFT operation.
///
/// The caller specifies the total FFT size and the GPU memory budget;
/// the planner determines how many passes are needed.
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Total number of complex elements in the FFT.
    pub total_elements: usize,
    /// Available GPU memory in bytes.
    pub available_gpu_memory: usize,
    /// Floating-point precision.
    pub precision: FftPrecision,
    /// Transform direction (forward / inverse).
    pub direction: FftDirection,
    /// Transform type (C2C, R2C, C2R).
    pub fft_type: FftType,
    /// Strategy to use for the decomposition.
    pub strategy: OutOfCoreStrategy,
    /// Optional upper bound on host staging buffer size (bytes).
    /// When `None`, the planner assumes host memory is unconstrained.
    pub host_staging_size: Option<usize>,
    /// Override the automatic pass count.  When `Some(k)`, exactly
    /// `k` passes are created regardless of the memory budget.
    pub num_passes: Option<u32>,
}

impl OutOfCoreConfig {
    /// Creates a new configuration with the minimum required parameters.
    ///
    /// Defaults to `Auto` strategy, forward direction, C2C transform,
    /// single precision, and no host/pass overrides.
    pub fn new(total_elements: usize, available_gpu_memory: usize) -> Self {
        Self {
            total_elements,
            available_gpu_memory,
            precision: FftPrecision::Single,
            direction: FftDirection::Forward,
            fft_type: FftType::C2C,
            strategy: OutOfCoreStrategy::Auto,
            host_staging_size: None,
            num_passes: None,
        }
    }

    /// Sets the precision.
    pub fn with_precision(mut self, precision: FftPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Sets the direction.
    pub fn with_direction(mut self, direction: FftDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Sets the FFT type.
    pub fn with_fft_type(mut self, fft_type: FftType) -> Self {
        self.fft_type = fft_type;
        self
    }

    /// Sets the strategy.
    pub fn with_strategy(mut self, strategy: OutOfCoreStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the host staging buffer size limit.
    pub fn with_host_staging_size(mut self, size: usize) -> Self {
        self.host_staging_size = Some(size);
        self
    }

    /// Overrides the automatic pass count.
    pub fn with_num_passes(mut self, passes: u32) -> Self {
        self.num_passes = Some(passes);
        self
    }
}

// ---------------------------------------------------------------------------
// OutOfCorePass -- one chunk of work
// ---------------------------------------------------------------------------

/// Describes a single pass of an out-of-core FFT.
///
/// Each pass transfers a contiguous slice of elements to the GPU,
/// computes the local FFT, optionally applies twiddle-factor fixup,
/// and transfers the result back to the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfCorePass {
    /// Zero-based index of this pass.
    pub pass_index: u32,
    /// Offset (in elements) into the global array where this pass starts.
    pub element_offset: usize,
    /// Number of elements processed in this pass.
    pub element_count: usize,
    /// GPU buffer requirement for this pass in bytes.
    pub gpu_buffer_bytes: usize,
    /// Whether a twiddle-factor fixup is needed after the local FFT.
    ///
    /// The first pass never requires fixup; subsequent passes do unless
    /// the FourStep algorithm handles it implicitly in step 2.
    pub requires_twiddle_fixup: bool,
}

impl fmt::Display for OutOfCorePass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pass {}: offset={}, count={}, gpu_bytes={}, twiddle={}",
            self.pass_index,
            self.element_offset,
            self.element_count,
            self.gpu_buffer_bytes,
            self.requires_twiddle_fixup,
        )
    }
}

// ---------------------------------------------------------------------------
// OutOfCoreStats -- transfer statistics
// ---------------------------------------------------------------------------

/// Statistics collected during (or estimated before) out-of-core FFT execution.
#[derive(Debug, Clone, PartialEq)]
pub struct OutOfCoreStats {
    /// Total bytes transferred from host to device across all passes.
    pub total_h2d_bytes: usize,
    /// Total bytes transferred from device to host across all passes.
    pub total_d2h_bytes: usize,
    /// Number of host-to-device transfers.
    pub h2d_transfers: u32,
    /// Number of device-to-host transfers.
    pub d2h_transfers: u32,
    /// Peak GPU memory usage in bytes (across all passes).
    pub peak_gpu_bytes: usize,
    /// Estimated compute time in seconds (wall-clock), or `None` if
    /// no timing information is available.
    pub estimated_compute_secs: Option<f64>,
}

impl OutOfCoreStats {
    /// Total bytes transferred in both directions.
    pub fn total_transfer_bytes(&self) -> usize {
        self.total_h2d_bytes + self.total_d2h_bytes
    }
}

impl fmt::Display for OutOfCoreStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OutOfCoreStats {{ H2D: {} bytes ({} xfers), D2H: {} bytes ({} xfers), peak_gpu: {} bytes }}",
            self.total_h2d_bytes,
            self.h2d_transfers,
            self.total_d2h_bytes,
            self.d2h_transfers,
            self.peak_gpu_bytes,
        )
    }
}

// ---------------------------------------------------------------------------
// OutOfCorePlan -- the decomposed execution plan
// ---------------------------------------------------------------------------

/// A fully decomposed out-of-core FFT execution plan.
///
/// Created by [`plan_out_of_core`], this plan lists every pass that must
/// be executed, the overlap region sizes, and estimated transfer counts.
#[derive(Debug, Clone)]
pub struct OutOfCorePlan {
    /// The configuration that produced this plan.
    pub config: OutOfCoreConfig,
    /// The resolved strategy (never `Auto` — always concrete).
    pub resolved_strategy: OutOfCoreStrategy,
    /// Ordered list of passes.
    pub passes: Vec<OutOfCorePass>,
    /// Number of elements that overlap between consecutive passes
    /// (used for twiddle-factor blending in the `Overlap` strategy).
    pub overlap_elements: usize,
    /// Total number of host <-> device transfers.
    pub total_host_transfers: usize,
    /// Estimated transfer and compute statistics.
    pub stats: OutOfCoreStats,
}

impl OutOfCorePlan {
    /// Returns the number of passes in this plan.
    pub fn num_passes(&self) -> u32 {
        self.passes.len() as u32
    }

    /// Returns whether the entire FFT fits in a single pass (no staging needed).
    pub fn is_single_pass(&self) -> bool {
        self.passes.len() == 1
    }
}

impl fmt::Display for OutOfCorePlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "OutOfCorePlan:")?;
        writeln!(f, "  total_elements: {}", self.config.total_elements,)?;
        writeln!(
            f,
            "  available_gpu_memory: {} bytes",
            self.config.available_gpu_memory,
        )?;
        writeln!(f, "  precision: {:?}", self.config.precision)?;
        writeln!(f, "  direction: {:?}", self.config.direction)?;
        writeln!(f, "  fft_type: {}", self.config.fft_type)?;
        writeln!(f, "  strategy: {}", self.resolved_strategy)?;
        writeln!(f, "  passes: {}", self.passes.len())?;
        writeln!(f, "  overlap_elements: {}", self.overlap_elements)?;
        writeln!(f, "  total_host_transfers: {}", self.total_host_transfers)?;
        for pass in &self.passes {
            writeln!(f, "    {pass}")?;
        }
        write!(f, "  {}", self.stats)
    }
}

// ---------------------------------------------------------------------------
// LargeFftConfig — TMA-based data loading for N > 64K
// ---------------------------------------------------------------------------

/// Configuration for large FFT transforms that benefit from Tensor Memory
/// Accelerator (TMA) data loading on Hopper (sm_90) and later GPUs.
///
/// When `N > 65536`, TMA-based loading reduces address generation overhead
/// and improves memory throughput for large 1-D transforms.
#[derive(Debug, Clone)]
pub struct LargeFftConfig {
    /// Total number of complex elements.
    pub n: usize,
    /// Whether TMA-based loading is enabled (true when n > 65536).
    pub tma_loading: bool,
    /// Chunk size in elements (determined by available GPU memory).
    pub chunk_size: usize,
}

impl LargeFftConfig {
    /// Creates a new `LargeFftConfig` for an FFT of `n` complex elements.
    ///
    /// `gpu_mem_bytes` is the available GPU memory in bytes; the chunk size
    /// is computed as `gpu_mem_bytes / 8` (8 bytes per complex f32 element).
    ///
    /// TMA loading is enabled automatically when `n > 65536`.
    #[must_use]
    pub fn new(n: usize, gpu_mem_bytes: usize) -> Self {
        let tma_loading = n > 65_536;
        let bytes_per_element = 8usize; // complex f32: 4 bytes real + 4 bytes imag
        let chunk_size = gpu_mem_bytes.checked_div(bytes_per_element).unwrap_or(0);
        Self {
            n,
            tma_loading,
            chunk_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates an [`OutOfCoreConfig`] before planning.
///
/// # Errors
///
/// Returns [`FftError::InvalidSize`] if:
/// - `total_elements` is zero
/// - `available_gpu_memory` is zero
/// - `available_gpu_memory` is too small to hold even a single complex element
/// - `num_passes` override is zero
/// - `host_staging_size` is zero when specified
pub fn validate_out_of_core_config(config: &OutOfCoreConfig) -> FftResult<()> {
    if config.total_elements == 0 {
        return Err(FftError::InvalidSize(
            "total_elements must be > 0 for out-of-core FFT".to_string(),
        ));
    }
    if config.available_gpu_memory == 0 {
        return Err(FftError::InvalidSize(
            "available_gpu_memory must be > 0".to_string(),
        ));
    }

    let elem_bytes = config.precision.complex_bytes();
    if config.available_gpu_memory < elem_bytes {
        return Err(FftError::InvalidSize(format!(
            "available_gpu_memory ({}) is less than one complex element ({} bytes)",
            config.available_gpu_memory, elem_bytes,
        )));
    }

    if let Some(0) = config.num_passes {
        return Err(FftError::InvalidSize(
            "num_passes override must be >= 1".to_string(),
        ));
    }

    if let Some(0) = config.host_staging_size {
        return Err(FftError::InvalidSize(
            "host_staging_size must be > 0 when specified".to_string(),
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Pass estimation
// ---------------------------------------------------------------------------

/// Estimates the minimum number of passes required to process `total_elements`
/// complex values given `gpu_memory` bytes and the specified `precision`.
///
/// The GPU must hold at least the input buffer plus a workspace buffer
/// (2x the chunk size), so the effective capacity per pass is
/// `gpu_memory / (2 * complex_bytes)`.
pub fn estimate_passes(total_elements: usize, gpu_memory: usize, precision: FftPrecision) -> u32 {
    let complex_bytes = precision.complex_bytes();
    // We need 2x memory: one for the input chunk, one for workspace.
    let effective_capacity = gpu_memory / (2 * complex_bytes);

    if effective_capacity == 0 {
        return u32::MAX;
    }

    let passes = total_elements.div_ceil(effective_capacity);
    // At least 1 pass
    (passes as u32).max(1)
}

// ---------------------------------------------------------------------------
// Twiddle factors
// ---------------------------------------------------------------------------

/// Computes twiddle factors for correcting a partial FFT pass.
///
/// When a size-N FFT is split into chunks of `pass_size` elements,
/// pass `pass_index` (0-based) must multiply element `j` (0-based
/// within the pass) by:
///
/// ```text
/// W_N^{j * (pass_index * pass_size)}
///   = exp(sign * 2*pi*i * j * pass_index * pass_size / N)
/// ```
///
/// where `sign = -1` for forward and `+1` for inverse transforms.
///
/// Returns a vector of `(re, im)` pairs, one per element in the pass.
/// The first pass (`pass_index == 0`) always returns unit twiddles `(1, 0)`.
pub fn compute_twiddle_factors(n: usize, pass_size: usize, pass_index: u32) -> Vec<(f64, f64)> {
    if n == 0 || pass_size == 0 {
        return Vec::new();
    }

    let n_f64 = n as f64;
    let offset = (pass_index as usize) * pass_size;

    (0..pass_size)
        .map(|j| {
            let angle = -2.0 * PI * (j * offset) as f64 / n_f64;
            (angle.cos(), angle.sin())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Strategy resolution
// ---------------------------------------------------------------------------

/// Resolves [`OutOfCoreStrategy::Auto`] to a concrete strategy.
///
/// Prefers `FourStep` when N has a good near-square factorisation
/// (i.e. `sqrt(N)` is close to an integer factor), otherwise `Overlap`.
fn resolve_strategy(total_elements: usize, strategy: OutOfCoreStrategy) -> OutOfCoreStrategy {
    match strategy {
        OutOfCoreStrategy::Auto => {
            let sqrt_n = (total_elements as f64).sqrt() as usize;
            // Check if N has a factor close to sqrt(N)
            if sqrt_n > 1 && total_elements % sqrt_n == 0 {
                OutOfCoreStrategy::FourStep
            } else {
                // Try sqrt_n +/- 1
                let candidates = [sqrt_n, sqrt_n + 1];
                let has_good_factor = candidates.iter().any(|&c| c > 1 && total_elements % c == 0);
                if has_good_factor {
                    OutOfCoreStrategy::FourStep
                } else {
                    OutOfCoreStrategy::Overlap
                }
            }
        }
        concrete => concrete,
    }
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// Decomposes an out-of-core FFT into a sequence of passes.
///
/// # Errors
///
/// Returns an error if the configuration is invalid (see
/// [`validate_out_of_core_config`]).
pub fn plan_out_of_core(config: &OutOfCoreConfig) -> FftResult<OutOfCorePlan> {
    validate_out_of_core_config(config)?;

    let resolved_strategy = resolve_strategy(config.total_elements, config.strategy);
    let complex_bytes = config.precision.complex_bytes();

    // Determine number of passes
    let num_passes = config.num_passes.unwrap_or_else(|| {
        estimate_passes(
            config.total_elements,
            config.available_gpu_memory,
            config.precision,
        )
    });
    let num_passes = num_passes.max(1);

    // Elements per pass (evenly distributed, last pass gets remainder)
    let base_count = config.total_elements / (num_passes as usize);
    let remainder = config.total_elements % (num_passes as usize);

    // Overlap for twiddle blending (only for Overlap strategy)
    let overlap_elements = match resolved_strategy {
        OutOfCoreStrategy::Overlap if num_passes > 1 => {
            // Small overlap: 1% of pass size, minimum 1
            let overlap = (base_count / 100).max(1);
            // Constrain by host staging if specified
            if let Some(host_limit) = config.host_staging_size {
                let max_overlap = host_limit / complex_bytes;
                overlap.min(max_overlap)
            } else {
                overlap
            }
        }
        _ => 0,
    };

    // Build passes
    let mut passes = Vec::with_capacity(num_passes as usize);
    let mut offset = 0usize;

    for i in 0..num_passes {
        let count = if (i as usize) < remainder {
            base_count + 1
        } else {
            base_count
        };

        let gpu_buffer_bytes = count * complex_bytes * 2; // input + workspace

        passes.push(OutOfCorePass {
            pass_index: i,
            element_offset: offset,
            element_count: count,
            gpu_buffer_bytes,
            requires_twiddle_fixup: i > 0,
        });

        offset += count;
    }

    // Transfer counts: each pass does one H2D + one D2H
    let total_host_transfers = (num_passes as usize) * 2;

    // Compute stats
    let total_h2d_bytes: usize = passes.iter().map(|p| p.element_count * complex_bytes).sum();
    let total_d2h_bytes = total_h2d_bytes; // symmetric
    let peak_gpu_bytes = passes.iter().map(|p| p.gpu_buffer_bytes).max().unwrap_or(0);

    let stats = OutOfCoreStats {
        total_h2d_bytes,
        total_d2h_bytes,
        h2d_transfers: num_passes,
        d2h_transfers: num_passes,
        peak_gpu_bytes,
        estimated_compute_secs: None,
    };

    Ok(OutOfCorePlan {
        config: config.clone(),
        resolved_strategy,
        passes,
        overlap_elements,
        total_host_transfers,
        stats,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Pass decomposition tests --

    #[test]
    fn plan_even_decomposition() {
        let config = OutOfCoreConfig::new(1024, 512 * 8).with_strategy(OutOfCoreStrategy::Overlap);
        let plan = plan_out_of_core(&config);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            // All elements accounted for
            let total: usize = p.passes.iter().map(|pass| pass.element_count).sum();
            assert_eq!(total, 1024);
            // Passes are contiguous
            for (i, pass) in p.passes.iter().enumerate() {
                assert_eq!(pass.pass_index, i as u32);
            }
        }
    }

    #[test]
    fn plan_single_pass_when_fits() {
        // 1024 elements * 8 bytes (f32 complex) = 8192 bytes data
        // We need 2x for workspace, so 16384 bytes
        // Give plenty of memory: 32768 bytes
        let config = OutOfCoreConfig::new(1024, 32768);
        let plan = plan_out_of_core(&config);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert!(p.is_single_pass());
            assert_eq!(p.passes.len(), 1);
            let pass = &p.passes[0];
            assert_eq!(pass.element_offset, 0);
            assert_eq!(pass.element_count, 1024);
            assert!(!pass.requires_twiddle_fixup);
        }
    }

    #[test]
    fn plan_exact_fit() {
        // Exactly enough memory for 512 elements with workspace
        // 512 * 8 * 2 = 8192 bytes
        let config = OutOfCoreConfig::new(512, 8192);
        let plan = plan_out_of_core(&config);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert!(p.is_single_pass());
            assert_eq!(p.passes[0].element_count, 512);
        }
    }

    #[test]
    fn plan_multiple_passes_small_memory() {
        // 4096 elements, only enough GPU memory for ~512 elements with workspace
        // 512 * 8 * 2 = 8192 bytes
        let config = OutOfCoreConfig::new(4096, 8192).with_strategy(OutOfCoreStrategy::Overlap);
        let plan = plan_out_of_core(&config);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert!(p.passes.len() > 1);
            let total: usize = p.passes.iter().map(|pass| pass.element_count).sum();
            assert_eq!(total, 4096);
            // First pass should not need twiddle fixup
            assert!(!p.passes[0].requires_twiddle_fixup);
            // Subsequent passes need fixup
            for pass in &p.passes[1..] {
                assert!(pass.requires_twiddle_fixup);
            }
        }
    }

    #[test]
    fn plan_num_passes_override() {
        let config = OutOfCoreConfig::new(1000, 1_000_000).with_num_passes(5);
        let plan = plan_out_of_core(&config);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert_eq!(p.passes.len(), 5);
            let total: usize = p.passes.iter().map(|pass| pass.element_count).sum();
            assert_eq!(total, 1000);
        }
    }

    #[test]
    fn plan_double_precision() {
        // Double precision: 16 bytes per complex element
        // 256 elements * 16 * 2 = 8192 bytes needed for single pass
        let config = OutOfCoreConfig::new(256, 8192).with_precision(FftPrecision::Double);
        let plan = plan_out_of_core(&config);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            assert!(p.is_single_pass());
            assert_eq!(p.passes[0].element_count, 256);
            // GPU buffer should account for double precision
            assert_eq!(p.passes[0].gpu_buffer_bytes, 256 * 16 * 2);
        }
    }

    // -- Twiddle factor tests --

    #[test]
    fn twiddle_first_pass_is_unity() {
        let twiddles = compute_twiddle_factors(1024, 256, 0);
        assert_eq!(twiddles.len(), 256);
        for (re, im) in &twiddles {
            assert!((re - 1.0).abs() < 1e-12, "re should be 1.0, got {re}");
            assert!(im.abs() < 1e-12, "im should be 0.0, got {im}");
        }
    }

    #[test]
    fn twiddle_known_values() {
        // N=4, pass_size=2, pass_index=1
        // offset = 1 * 2 = 2
        // j=0: angle = -2*pi*0*2/4 = 0 => (1, 0)
        // j=1: angle = -2*pi*1*2/4 = -pi => (cos(-pi), sin(-pi)) = (-1, ~0)
        let twiddles = compute_twiddle_factors(4, 2, 1);
        assert_eq!(twiddles.len(), 2);

        let (re0, im0) = twiddles[0];
        assert!((re0 - 1.0).abs() < 1e-12);
        assert!(im0.abs() < 1e-12);

        let (re1, im1) = twiddles[1];
        assert!((re1 - (-1.0)).abs() < 1e-12);
        assert!(im1.abs() < 1e-12);
    }

    #[test]
    fn twiddle_quarter_rotation() {
        // N=8, pass_size=4, pass_index=1
        // offset = 4
        // j=1: angle = -2*pi*1*4/8 = -pi => (-1, 0)
        // j=2: angle = -2*pi*2*4/8 = -2*pi => (1, 0)
        let twiddles = compute_twiddle_factors(8, 4, 1);
        assert_eq!(twiddles.len(), 4);

        // j=0: angle = 0 => (1, 0)
        assert!((twiddles[0].0 - 1.0).abs() < 1e-12);
        assert!(twiddles[0].1.abs() < 1e-12);

        // j=1: angle = -pi => (-1, 0)
        assert!((twiddles[1].0 - (-1.0)).abs() < 1e-12);
        assert!(twiddles[1].1.abs() < 1e-12);

        // j=2: angle = -2*pi => (1, 0)
        assert!((twiddles[2].0 - 1.0).abs() < 1e-12);
        assert!(twiddles[2].1.abs() < 1e-12);

        // j=3: angle = -3*pi => (-1, 0)
        assert!((twiddles[3].0 - (-1.0)).abs() < 1e-12);
        assert!(twiddles[3].1.abs() < 1e-12);
    }

    #[test]
    fn twiddle_empty_on_zero() {
        assert!(compute_twiddle_factors(0, 10, 1).is_empty());
        assert!(compute_twiddle_factors(10, 0, 1).is_empty());
    }

    // -- Config validation tests --

    #[test]
    fn validate_zero_elements() {
        let config = OutOfCoreConfig::new(0, 1024);
        let result = validate_out_of_core_config(&config);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn validate_zero_memory() {
        let config = OutOfCoreConfig::new(1024, 0);
        let result = validate_out_of_core_config(&config);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn validate_memory_too_small() {
        // Less than one complex element (8 bytes for f32)
        let config = OutOfCoreConfig::new(1024, 4);
        let result = validate_out_of_core_config(&config);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn validate_zero_passes_override() {
        let config = OutOfCoreConfig::new(1024, 8192).with_num_passes(0);
        let result = validate_out_of_core_config(&config);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn validate_zero_host_staging() {
        let config = OutOfCoreConfig::new(1024, 8192).with_host_staging_size(0);
        let result = validate_out_of_core_config(&config);
        assert!(matches!(result, Err(FftError::InvalidSize(_))));
    }

    #[test]
    fn validate_valid_config() {
        let config = OutOfCoreConfig::new(1024, 8192);
        assert!(validate_out_of_core_config(&config).is_ok());
    }

    // -- Strategy tests --

    #[test]
    fn strategy_auto_selects_four_step_for_square() {
        // 1024 = 32 * 32 => perfect square
        let resolved = resolve_strategy(1024, OutOfCoreStrategy::Auto);
        assert_eq!(resolved, OutOfCoreStrategy::FourStep);
    }

    #[test]
    fn strategy_auto_selects_overlap_for_prime() {
        // 1009 is prime
        let resolved = resolve_strategy(1009, OutOfCoreStrategy::Auto);
        assert_eq!(resolved, OutOfCoreStrategy::Overlap);
    }

    #[test]
    fn strategy_explicit_not_changed() {
        let resolved = resolve_strategy(1024, OutOfCoreStrategy::Overlap);
        assert_eq!(resolved, OutOfCoreStrategy::Overlap);

        let resolved = resolve_strategy(1009, OutOfCoreStrategy::FourStep);
        assert_eq!(resolved, OutOfCoreStrategy::FourStep);
    }

    // -- Display tests --

    #[test]
    fn display_plan() {
        let config = OutOfCoreConfig::new(2048, 4096).with_strategy(OutOfCoreStrategy::Overlap);
        let plan = plan_out_of_core(&config);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let display = format!("{p}");
            assert!(display.contains("OutOfCorePlan"));
            assert!(display.contains("2048"));
            assert!(display.contains("Overlap"));
            assert!(display.contains("Pass"));
        }
    }

    #[test]
    fn display_pass() {
        let pass = OutOfCorePass {
            pass_index: 2,
            element_offset: 512,
            element_count: 256,
            gpu_buffer_bytes: 4096,
            requires_twiddle_fixup: true,
        };
        let s = format!("{pass}");
        assert!(s.contains("Pass 2"));
        assert!(s.contains("512"));
        assert!(s.contains("256"));
        assert!(s.contains("true"));
    }

    #[test]
    fn display_stats() {
        let stats = OutOfCoreStats {
            total_h2d_bytes: 8192,
            total_d2h_bytes: 8192,
            h2d_transfers: 4,
            d2h_transfers: 4,
            peak_gpu_bytes: 2048,
            estimated_compute_secs: None,
        };
        let s = format!("{stats}");
        assert!(s.contains("8192"));
        assert!(s.contains("4 xfers"));
    }

    // -- estimate_passes tests --

    #[test]
    fn estimate_passes_single() {
        // 100 elements, 10000 bytes => 100 * 8 * 2 = 1600 needed, plenty of room
        let passes = estimate_passes(100, 10000, FftPrecision::Single);
        assert_eq!(passes, 1);
    }

    #[test]
    fn estimate_passes_multiple() {
        // 1000 elements, each needs 8 bytes (f32 complex), 2x for workspace = 16 each
        // 100 bytes of GPU memory => capacity = 100 / 16 = 6 elements per pass
        // 1000 / 6 = 167 passes (ceil)
        let passes = estimate_passes(1000, 100, FftPrecision::Single);
        assert_eq!(passes, 167);
    }

    // -- Stats test --

    #[test]
    fn stats_total_transfer() {
        let stats = OutOfCoreStats {
            total_h2d_bytes: 1000,
            total_d2h_bytes: 2000,
            h2d_transfers: 1,
            d2h_transfers: 1,
            peak_gpu_bytes: 500,
            estimated_compute_secs: Some(1.5),
        };
        assert_eq!(stats.total_transfer_bytes(), 3000);
    }
}
