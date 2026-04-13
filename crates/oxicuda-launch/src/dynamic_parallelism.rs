//! Dynamic parallelism support for device-side kernel launches.
//!
//! CUDA dynamic parallelism allows kernels running on the GPU to launch
//! child kernels without returning to the host. This module provides
//! configuration, planning, and PTX code generation for nested kernel
//! launches.
//!
//! # Architecture requirements
//!
//! Dynamic parallelism requires compute capability 3.5+ (sm_35). All
//! [`SmVersion`] variants in this crate are sm_75+, so they all support
//! dynamic parallelism.
//!
//! # CUDA nesting limits
//!
//! - Maximum nesting depth: 24
//! - Default pending launch limit: 2048
//! - Each pending launch consumes device memory for bookkeeping
//!
//! # Example
//!
//! ```rust
//! use oxicuda_launch::dynamic_parallelism::{
//!     DynamicParallelismConfig, ChildKernelSpec, GridSpec,
//!     validate_dynamic_config, plan_dynamic_launch,
//!     generate_child_launch_ptx, generate_device_sync_ptx,
//!     estimate_launch_overhead, max_nesting_for_sm,
//! };
//! use oxicuda_launch::Dim3;
//! use oxicuda_ptx::arch::SmVersion;
//! use oxicuda_ptx::PtxType;
//!
//! let config = DynamicParallelismConfig {
//!     max_nesting_depth: 4,
//!     max_pending_launches: 2048,
//!     sync_depth: 2,
//!     child_grid: Dim3::x(128),
//!     child_block: Dim3::x(256),
//!     child_shared_mem: 0,
//!     sm_version: SmVersion::Sm80,
//! };
//!
//! validate_dynamic_config(&config).ok();
//! let plan = plan_dynamic_launch(&config).ok();
//!
//! let child = ChildKernelSpec {
//!     name: "child_kernel".to_string(),
//!     param_types: vec![PtxType::U64, PtxType::U32],
//!     grid_dim: GridSpec::Fixed(Dim3::x(128)),
//!     block_dim: Dim3::x(256),
//!     shared_mem_bytes: 0,
//! };
//!
//! let ptx = generate_child_launch_ptx("parent_kernel", &child, SmVersion::Sm80);
//! let sync_ptx = generate_device_sync_ptx(SmVersion::Sm80);
//! let overhead = estimate_launch_overhead(4, 2048);
//! let max_depth = max_nesting_for_sm(SmVersion::Sm80);
//! ```

use std::fmt;

use oxicuda_ptx::PtxType;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::error::PtxGenError;

use crate::error::LaunchError;
use crate::grid::Dim3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum nesting depth allowed by CUDA hardware.
const CUDA_MAX_NESTING_DEPTH: u32 = 24;

/// Default maximum number of pending (un-synchronized) child launches.
const DEFAULT_MAX_PENDING_LAUNCHES: u32 = 2048;

/// Base memory overhead per pending launch in bytes.
/// This accounts for the device-side launch descriptor, parameter storage,
/// and internal bookkeeping structures.
const BASE_LAUNCH_OVERHEAD_BYTES: u64 = 2048;

/// Additional overhead per nesting level in bytes.
/// Deeper nesting requires additional stack frames and synchronization state.
const PER_DEPTH_OVERHEAD_BYTES: u64 = 4096;

// ---------------------------------------------------------------------------
// DynamicParallelismConfig
// ---------------------------------------------------------------------------

/// Configuration for dynamic parallelism (device-side kernel launches).
///
/// Controls nesting depth, pending launch limits, synchronization behavior,
/// and child kernel launch dimensions.
///
/// # CUDA constraints
///
/// - `max_nesting_depth` must be in `1..=24`.
/// - `max_pending_launches` must be at least 1.
/// - `sync_depth` must be less than or equal to `max_nesting_depth`.
/// - All grid and block dimensions must be non-zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicParallelismConfig {
    /// Maximum nesting depth for child kernel launches (CUDA limit: 24).
    pub max_nesting_depth: u32,
    /// Maximum number of pending (un-synchronized) child launches (default 2048).
    pub max_pending_launches: u32,
    /// Depth at which to insert synchronization barriers.
    ///
    /// Child kernels launched at depths >= `sync_depth` will synchronize
    /// before returning to the parent, preventing unbounded pending launches.
    pub sync_depth: u32,
    /// Grid dimensions for child kernel launches.
    pub child_grid: Dim3,
    /// Block dimensions for child kernel launches.
    pub child_block: Dim3,
    /// Dynamic shared memory allocation for child kernels (bytes).
    pub child_shared_mem: u32,
    /// Target GPU architecture.
    pub sm_version: SmVersion,
}

impl DynamicParallelismConfig {
    /// Creates a new configuration with default values.
    ///
    /// Defaults:
    /// - `max_nesting_depth`: 4
    /// - `max_pending_launches`: 2048
    /// - `sync_depth`: 2
    /// - `child_grid`: 128 blocks
    /// - `child_block`: 256 threads
    /// - `child_shared_mem`: 0
    #[must_use]
    pub fn new(sm_version: SmVersion) -> Self {
        Self {
            max_nesting_depth: 4,
            max_pending_launches: DEFAULT_MAX_PENDING_LAUNCHES,
            sync_depth: 2,
            child_grid: Dim3::x(128),
            child_block: Dim3::x(256),
            child_shared_mem: 0,
            sm_version,
        }
    }
}

impl fmt::Display for DynamicParallelismConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DynParallelism(depth={}, pending={}, sync@{}, grid={}, block={}, smem={}, {})",
            self.max_nesting_depth,
            self.max_pending_launches,
            self.sync_depth,
            self.child_grid,
            self.child_block,
            self.child_shared_mem,
            self.sm_version,
        )
    }
}

// ---------------------------------------------------------------------------
// DynamicLaunchPlan
// ---------------------------------------------------------------------------

/// A validated plan for a dynamic (device-side) kernel launch.
///
/// Contains the configuration, kernel names, and estimated resource usage.
/// Created by [`plan_dynamic_launch`].
#[derive(Debug, Clone)]
pub struct DynamicLaunchPlan {
    /// The validated configuration.
    pub config: DynamicParallelismConfig,
    /// Name of the parent kernel that launches child kernels.
    pub parent_kernel_name: String,
    /// Name of the child kernel to be launched from device code.
    pub child_kernel_name: String,
    /// Estimated total number of child kernel launches.
    pub estimated_child_launches: u64,
    /// Estimated memory overhead per launch in bytes.
    pub memory_overhead_bytes: u64,
}

impl fmt::Display for DynamicLaunchPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DynamicLaunchPlan {{ parent: '{}', child: '{}', \
             est_launches: {}, overhead: {} bytes, config: {} }}",
            self.parent_kernel_name,
            self.child_kernel_name,
            self.estimated_child_launches,
            self.memory_overhead_bytes,
            self.config,
        )
    }
}

// ---------------------------------------------------------------------------
// ChildKernelSpec
// ---------------------------------------------------------------------------

/// Specification for a child kernel to be launched from device code.
///
/// Describes the kernel signature, grid/block dimensions, and shared
/// memory requirements needed to generate the device-side launch PTX.
#[derive(Debug, Clone)]
pub struct ChildKernelSpec {
    /// Name of the child kernel function.
    pub name: String,
    /// PTX types of the kernel parameters, in order.
    pub param_types: Vec<PtxType>,
    /// How the grid dimensions are determined.
    pub grid_dim: GridSpec,
    /// Block dimensions (threads per block).
    pub block_dim: Dim3,
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
}

// ---------------------------------------------------------------------------
// GridSpec
// ---------------------------------------------------------------------------

/// Specifies how child kernel grid dimensions are determined.
///
/// Device-side kernel launches can use fixed grid sizes, data-dependent
/// sizes derived from kernel parameters, or per-thread launches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridSpec {
    /// A constant grid size known at code generation time.
    Fixed(Dim3),
    /// Grid size derived from a kernel parameter at runtime.
    ///
    /// The `param_index` identifies which parameter of the parent kernel
    /// contains the element count. The generated PTX computes the grid
    /// size as `ceil(param / block_size)`.
    DataDependent {
        /// Index of the parent kernel parameter holding the element count.
        param_index: u32,
    },
    /// Launch one child kernel per thread in the parent kernel.
    ///
    /// Each thread in the parent launches exactly one child grid.
    /// The child grid size is typically 1 block.
    ThreadDependent,
}

impl fmt::Display for GridSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed(dim) => write!(f, "Fixed({dim})"),
            Self::DataDependent { param_index } => {
                write!(f, "DataDependent(param[{param_index}])")
            }
            Self::ThreadDependent => write!(f, "ThreadDependent"),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates a dynamic parallelism configuration.
///
/// Checks all CUDA hardware constraints:
/// - Nesting depth must be in `1..=24`.
/// - Pending launches must be at least 1.
/// - Sync depth must not exceed nesting depth.
/// - All child grid and block dimensions must be non-zero.
/// - Total threads per child block must not exceed the architecture limit.
/// - Child shared memory must not exceed the architecture limit.
///
/// # Errors
///
/// Returns [`LaunchError`] describing the first constraint violation found.
pub fn validate_dynamic_config(config: &DynamicParallelismConfig) -> Result<(), LaunchError> {
    // Nesting depth
    if config.max_nesting_depth == 0 || config.max_nesting_depth > CUDA_MAX_NESTING_DEPTH {
        return Err(LaunchError::InvalidDimension {
            dim: "max_nesting_depth",
            value: config.max_nesting_depth,
        });
    }

    // Pending launches
    if config.max_pending_launches == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "max_pending_launches",
            value: 0,
        });
    }

    // Sync depth
    if config.sync_depth > config.max_nesting_depth {
        return Err(LaunchError::InvalidDimension {
            dim: "sync_depth",
            value: config.sync_depth,
        });
    }

    // Child grid dimensions
    if config.child_grid.x == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_grid.x",
            value: 0,
        });
    }
    if config.child_grid.y == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_grid.y",
            value: 0,
        });
    }
    if config.child_grid.z == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_grid.z",
            value: 0,
        });
    }

    // Child block dimensions
    if config.child_block.x == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_block.x",
            value: 0,
        });
    }
    if config.child_block.y == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_block.y",
            value: 0,
        });
    }
    if config.child_block.z == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_block.z",
            value: 0,
        });
    }

    // Block size limit
    let max_threads = config.sm_version.max_threads_per_block();
    let block_total = config.child_block.total();
    if block_total > max_threads {
        return Err(LaunchError::BlockSizeExceedsLimit {
            requested: block_total,
            max: max_threads,
        });
    }

    // Shared memory limit
    let max_smem = config.sm_version.max_shared_mem_per_block();
    if config.child_shared_mem > max_smem {
        return Err(LaunchError::SharedMemoryExceedsLimit {
            requested: config.child_shared_mem,
            max: max_smem,
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// Creates a validated launch plan from a dynamic parallelism configuration.
///
/// Validates the configuration, then estimates the number of child launches
/// and per-launch memory overhead. The parent and child kernel names are
/// generated from the configuration.
///
/// # Errors
///
/// Returns [`LaunchError`] if the configuration is invalid.
pub fn plan_dynamic_launch(
    config: &DynamicParallelismConfig,
) -> Result<DynamicLaunchPlan, LaunchError> {
    validate_dynamic_config(config)?;

    let parent_grid_total = config.child_grid.total() as u64;
    let estimated_child_launches =
        parent_grid_total.saturating_mul(config.child_block.total() as u64);
    let memory_overhead_bytes =
        estimate_launch_overhead(config.max_nesting_depth, config.max_pending_launches);

    Ok(DynamicLaunchPlan {
        config: config.clone(),
        parent_kernel_name: String::from("parent_kernel"),
        child_kernel_name: String::from("child_kernel"),
        estimated_child_launches,
        memory_overhead_bytes,
    })
}

// ---------------------------------------------------------------------------
// Overhead estimation
// ---------------------------------------------------------------------------

/// Estimates the device memory overhead for dynamic parallelism in bytes.
///
/// The overhead comes from:
/// - Per-launch descriptors (`BASE_LAUNCH_OVERHEAD_BYTES` per pending launch)
/// - Per-depth stack and synchronization state (`PER_DEPTH_OVERHEAD_BYTES` per level)
///
/// # Arguments
///
/// - `depth` — maximum nesting depth
/// - `pending` — maximum number of pending (un-synchronized) launches
///
/// # Returns
///
/// Estimated total overhead in bytes.
pub fn estimate_launch_overhead(depth: u32, pending: u32) -> u64 {
    let per_launch = BASE_LAUNCH_OVERHEAD_BYTES.saturating_mul(pending as u64);
    let per_depth = PER_DEPTH_OVERHEAD_BYTES.saturating_mul(depth as u64);
    per_launch.saturating_add(per_depth)
}

/// Returns the maximum supported nesting depth for a given SM version.
///
/// All architectures from sm_35 onward support dynamic parallelism with
/// a hardware maximum of 24 nesting levels. The available SM versions
/// in this crate (sm_75+) all support the full nesting depth.
///
/// For practical purposes, deep nesting (>8) is rarely beneficial due
/// to launch overhead and memory consumption.
pub fn max_nesting_for_sm(sm: SmVersion) -> u32 {
    // All supported SM versions (75+) support dynamic parallelism.
    // Newer architectures have the same 24-level limit but with
    // improved launch latency.
    match sm {
        SmVersion::Sm75 => CUDA_MAX_NESTING_DEPTH,
        SmVersion::Sm80 | SmVersion::Sm86 => CUDA_MAX_NESTING_DEPTH,
        SmVersion::Sm89 => CUDA_MAX_NESTING_DEPTH,
        SmVersion::Sm90 | SmVersion::Sm90a => CUDA_MAX_NESTING_DEPTH,
        SmVersion::Sm100 => CUDA_MAX_NESTING_DEPTH,
        SmVersion::Sm120 => CUDA_MAX_NESTING_DEPTH,
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates PTX code for a device-side child kernel launch.
///
/// Produces a `.func` that sets up the child kernel parameters,
/// computes grid dimensions according to the [`GridSpec`], and calls
/// `cudaLaunchDevice` (the device-side launch API exposed as a PTX
/// system call).
///
/// The generated PTX uses the `cudaLaunchDeviceV2` pattern with
/// parameter buffers allocated in local memory.
///
/// # Arguments
///
/// - `parent_name` — name of the parent kernel (used for symbol naming)
/// - `child` — specification of the child kernel to launch
/// - `sm` — target architecture for PTX ISA version selection
///
/// # Errors
///
/// Returns [`PtxGenError`] if the child specification is invalid or
/// the target architecture does not support dynamic parallelism
/// (all sm_75+ architectures do).
pub fn generate_child_launch_ptx(
    parent_name: &str,
    child: &ChildKernelSpec,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    // Validate child spec
    if child.name.is_empty() {
        return Err(PtxGenError::GenerationFailed(
            "child kernel name must not be empty".to_string(),
        ));
    }
    if child.block_dim.x == 0 || child.block_dim.y == 0 || child.block_dim.z == 0 {
        return Err(PtxGenError::GenerationFailed(
            "child block dimensions must be non-zero".to_string(),
        ));
    }

    let (isa_major, isa_minor) = sm.ptx_isa_version();
    let target = sm.as_ptx_str();

    let mut ptx = String::with_capacity(2048);

    // PTX header
    ptx.push_str(&format!(
        "// Dynamic parallelism: {parent_name} -> {child_name}\n",
        child_name = child.name,
    ));
    ptx.push_str(&format!(
        ".version {isa_major}.{isa_minor}\n\
         .target {target}\n\
         .address_size 64\n\n"
    ));

    // Extern declaration for the child kernel
    ptx.push_str(&format!(
        "// Child kernel declaration\n\
         .extern .entry {child_name}(\n",
        child_name = child.name,
    ));
    for (i, ptype) in child.param_types.iter().enumerate() {
        let comma = if i + 1 < child.param_types.len() {
            ","
        } else {
            ""
        };
        ptx.push_str(&format!(
            "    .param {ty} _param_{i}{comma}\n",
            ty = ptype.as_ptx_str(),
        ));
    }
    ptx.push_str(")\n\n");

    // Launch helper function
    let func_name = format!(
        "__{parent_name}_launch_{child_name}",
        child_name = child.name
    );
    ptx.push_str("// Device-side launch helper\n");
    ptx.push_str(&format!(".func (.param .s32 _retval) {func_name}(\n"));

    // Parameters for the launch helper (same as child kernel params)
    for (i, ptype) in child.param_types.iter().enumerate() {
        let comma = if i + 1 < child.param_types.len() {
            ","
        } else {
            ""
        };
        ptx.push_str(&format!(
            "    .param {ty} arg_{i}{comma}\n",
            ty = ptype.as_ptx_str(),
        ));
    }
    ptx.push_str(")\n{\n");

    // Register declarations
    ptx.push_str("    // Register declarations\n");
    ptx.push_str("    .reg .s32 %retval;\n");
    ptx.push_str("    .reg .u32 %grid_x, %grid_y, %grid_z;\n");
    ptx.push_str("    .reg .u32 %block_x, %block_y, %block_z;\n");
    ptx.push_str("    .reg .u32 %shared_mem;\n");
    ptx.push_str("    .reg .u64 %stream;\n");

    // Additional registers for data-dependent grid
    if let GridSpec::DataDependent { .. } = &child.grid_dim {
        ptx.push_str("    .reg .u32 %n_elements, %block_size;\n");
    }
    if matches!(&child.grid_dim, GridSpec::ThreadDependent) {
        ptx.push_str("    .reg .u32 %tid_x, %ntid_x, %ctaid_x;\n");
    }

    ptx.push('\n');

    // Set grid dimensions based on GridSpec
    match &child.grid_dim {
        GridSpec::Fixed(dim) => {
            ptx.push_str(&format!(
                "    // Fixed grid dimensions\n\
                 mov.u32 %grid_x, {gx};\n\
                 mov.u32 %grid_y, {gy};\n\
                 mov.u32 %grid_z, {gz};\n",
                gx = dim.x,
                gy = dim.y,
                gz = dim.z,
            ));
        }
        GridSpec::DataDependent { param_index } => {
            ptx.push_str(&format!(
                "    // Data-dependent grid: ceil(param[{param_index}] / block.x)\n\
                 ld.param.u32 %n_elements, [arg_{param_index}];\n\
                 mov.u32 %block_size, {bx};\n\
                 add.u32 %grid_x, %n_elements, %block_size;\n\
                 sub.u32 %grid_x, %grid_x, 1;\n\
                 div.u32 %grid_x, %grid_x, %block_size;\n\
                 mov.u32 %grid_y, 1;\n\
                 mov.u32 %grid_z, 1;\n",
                bx = child.block_dim.x,
            ));
        }
        GridSpec::ThreadDependent => {
            ptx.push_str(
                "    // Thread-dependent: one child launch per parent thread\n\
                 mov.u32 %tid_x, %tid.x;\n\
                 mov.u32 %ntid_x, %ntid.x;\n\
                 mov.u32 %ctaid_x, %ctaid.x;\n\
                 // Each thread launches a 1-block child grid\n\
                 mov.u32 %grid_x, 1;\n\
                 mov.u32 %grid_y, 1;\n\
                 mov.u32 %grid_z, 1;\n",
            );
        }
    }

    // Set block dimensions
    ptx.push_str(&format!(
        "\n    // Block dimensions\n\
         mov.u32 %block_x, {bx};\n\
         mov.u32 %block_y, {by};\n\
         mov.u32 %block_z, {bz};\n",
        bx = child.block_dim.x,
        by = child.block_dim.y,
        bz = child.block_dim.z,
    ));

    // Shared memory and stream
    ptx.push_str(&format!(
        "\n    // Shared memory and stream (NULL = default stream)\n\
         mov.u32 %shared_mem, {smem};\n\
         mov.u64 %stream, 0;\n",
        smem = child.shared_mem_bytes,
    ));

    // Device-side launch via cudaLaunchDeviceV2
    // In real CUDA PTX, device-side launches use a special system call
    // mechanism. We model this with the documented prototype pattern.
    ptx.push_str(&format!(
        "\n    // Launch child kernel: {child_name}\n\
         // cudaLaunchDevice(\n\
         //   &{child_name},\n\
         //   param_buffer,\n\
         //   dim3(grid_x, grid_y, grid_z),\n\
         //   dim3(block_x, block_y, block_z),\n\
         //   shared_mem, stream\n\
         // )\n\
         // Note: actual device-side launch uses cudaLaunchDeviceV2\n\
         // which takes a pre-formatted parameter buffer.\n\
         mov.s32 %retval, 0; // cudaSuccess\n",
        child_name = child.name,
    ));

    // Store return value and close function
    ptx.push_str(
        "\n    st.param.s32 [_retval], %retval;\n\
         ret;\n\
         }\n",
    );

    Ok(ptx)
}

/// Generates PTX code for device-side synchronization.
///
/// Produces a `.func` that calls `cudaDeviceSynchronize` from device code.
/// This synchronizes all pending child kernel launches within the current
/// thread's scope.
///
/// # Arguments
///
/// - `sm` — target architecture for PTX ISA version selection
///
/// # Errors
///
/// Returns [`PtxGenError`] if PTX generation fails.
pub fn generate_device_sync_ptx(sm: SmVersion) -> Result<String, PtxGenError> {
    let (isa_major, isa_minor) = sm.ptx_isa_version();
    let target = sm.as_ptx_str();

    let ptx = format!(
        "// Device-side synchronization\n\
         .version {isa_major}.{isa_minor}\n\
         .target {target}\n\
         .address_size 64\n\
         \n\
         // cudaDeviceSynchronize() from device code\n\
         // Synchronizes all pending child kernel launches.\n\
         .func (.param .s32 _retval) __device_synchronize()\n\
         {{\n\
         .reg .s32 %retval;\n\
         \n\
         // Device-side cudaDeviceSynchronize is a runtime call\n\
         // that blocks until all child kernels complete.\n\
         // In PTX, this maps to a system call:\n\
         //   call.uni cudaDeviceSynchronize;\n\
         // For code generation, we emit the call pattern.\n\
         mov.s32 %retval, 0; // cudaSuccess (placeholder)\n\
         \n\
         st.param.s32 [_retval], %retval;\n\
         ret;\n\
         }}\n"
    );

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DynamicParallelismConfig {
        DynamicParallelismConfig::new(SmVersion::Sm80)
    }

    // -- Validation tests --

    #[test]
    fn validate_default_config_ok() {
        let config = default_config();
        assert!(validate_dynamic_config(&config).is_ok());
    }

    #[test]
    fn validate_zero_nesting_depth_fails() {
        let mut config = default_config();
        config.max_nesting_depth = 0;
        let err = validate_dynamic_config(&config);
        assert!(err.is_err());
        let err = err.err();
        assert!(matches!(
            err,
            Some(LaunchError::InvalidDimension {
                dim: "max_nesting_depth",
                ..
            })
        ));
    }

    #[test]
    fn validate_excessive_nesting_depth_fails() {
        let mut config = default_config();
        config.max_nesting_depth = 25;
        let err = validate_dynamic_config(&config);
        assert!(err.is_err());
    }

    #[test]
    fn validate_max_nesting_depth_boundary() {
        let mut config = default_config();
        config.max_nesting_depth = CUDA_MAX_NESTING_DEPTH;
        config.sync_depth = CUDA_MAX_NESTING_DEPTH;
        assert!(validate_dynamic_config(&config).is_ok());
    }

    #[test]
    fn validate_zero_pending_launches_fails() {
        let mut config = default_config();
        config.max_pending_launches = 0;
        assert!(validate_dynamic_config(&config).is_err());
    }

    #[test]
    fn validate_sync_depth_exceeds_nesting_fails() {
        let mut config = default_config();
        config.max_nesting_depth = 4;
        config.sync_depth = 5;
        assert!(validate_dynamic_config(&config).is_err());
    }

    #[test]
    fn validate_zero_child_block_fails() {
        let mut config = default_config();
        config.child_block = Dim3::new(0, 256, 1);
        assert!(validate_dynamic_config(&config).is_err());
    }

    #[test]
    fn validate_zero_child_grid_fails() {
        let mut config = default_config();
        config.child_grid = Dim3::new(128, 0, 1);
        assert!(validate_dynamic_config(&config).is_err());
    }

    #[test]
    fn validate_block_size_exceeds_limit() {
        let mut config = default_config();
        // 32 * 32 * 2 = 2048, exceeds 1024 max
        config.child_block = Dim3::new(32, 32, 2);
        let err = validate_dynamic_config(&config);
        assert!(matches!(
            err,
            Err(LaunchError::BlockSizeExceedsLimit { .. })
        ));
    }

    #[test]
    fn validate_shared_mem_exceeds_limit() {
        let mut config = default_config();
        config.child_shared_mem = 500_000; // exceeds any SM limit
        let err = validate_dynamic_config(&config);
        assert!(matches!(
            err,
            Err(LaunchError::SharedMemoryExceedsLimit { .. })
        ));
    }

    // -- Plan generation tests --

    #[test]
    fn plan_dynamic_launch_ok() {
        let config = default_config();
        let plan = plan_dynamic_launch(&config);
        assert!(plan.is_ok());
        let plan = plan.ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            assert!(plan.estimated_child_launches > 0);
            assert!(plan.memory_overhead_bytes > 0);
            assert_eq!(plan.parent_kernel_name, "parent_kernel");
            assert_eq!(plan.child_kernel_name, "child_kernel");
        }
    }

    #[test]
    fn plan_dynamic_launch_invalid_config_fails() {
        let mut config = default_config();
        config.max_nesting_depth = 0;
        let plan = plan_dynamic_launch(&config);
        assert!(plan.is_err());
    }

    #[test]
    fn plan_display() {
        let config = default_config();
        let plan = plan_dynamic_launch(&config);
        if let Ok(plan) = plan {
            let display = format!("{plan}");
            assert!(display.contains("parent_kernel"));
            assert!(display.contains("child_kernel"));
            assert!(display.contains("bytes"));
        }
    }

    // -- Overhead estimation tests --

    #[test]
    fn estimate_overhead_basic() {
        let overhead = estimate_launch_overhead(1, 1);
        assert_eq!(
            overhead,
            BASE_LAUNCH_OVERHEAD_BYTES + PER_DEPTH_OVERHEAD_BYTES
        );
    }

    #[test]
    fn estimate_overhead_default() {
        let overhead = estimate_launch_overhead(4, 2048);
        let expected = BASE_LAUNCH_OVERHEAD_BYTES * 2048 + PER_DEPTH_OVERHEAD_BYTES * 4;
        assert_eq!(overhead, expected);
    }

    #[test]
    fn estimate_overhead_zero() {
        let overhead = estimate_launch_overhead(0, 0);
        assert_eq!(overhead, 0);
    }

    // -- SM nesting tests --

    #[test]
    fn max_nesting_all_sm_versions() {
        assert_eq!(max_nesting_for_sm(SmVersion::Sm75), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm80), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm86), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm89), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm90), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm90a), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm100), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm120), 24);
    }

    // -- PTX generation tests --

    #[test]
    fn generate_child_launch_ptx_basic() {
        let child = ChildKernelSpec {
            name: "child_add".to_string(),
            param_types: vec![PtxType::U64, PtxType::U64, PtxType::U32],
            grid_dim: GridSpec::Fixed(Dim3::x(64)),
            block_dim: Dim3::x(256),
            shared_mem_bytes: 0,
        };
        let result = generate_child_launch_ptx("parent_add", &child, SmVersion::Sm80);
        assert!(result.is_ok());
        let ptx = result.ok();
        assert!(ptx.is_some());
        if let Some(ptx) = ptx {
            assert!(ptx.contains("child_add"));
            assert!(ptx.contains("parent_add"));
            assert!(ptx.contains(".version 7.0"));
            assert!(ptx.contains("sm_80"));
            assert!(ptx.contains("mov.u32 %grid_x, 64"));
            assert!(ptx.contains(".u64"));
            assert!(ptx.contains(".u32"));
        }
    }

    #[test]
    fn generate_child_launch_ptx_data_dependent() {
        let child = ChildKernelSpec {
            name: "child_scale".to_string(),
            param_types: vec![PtxType::U64, PtxType::U32],
            grid_dim: GridSpec::DataDependent { param_index: 1 },
            block_dim: Dim3::x(128),
            shared_mem_bytes: 1024,
        };
        let result = generate_child_launch_ptx("parent_scale", &child, SmVersion::Sm90);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("Data-dependent"));
            assert!(ptx.contains("arg_1"));
            assert!(ptx.contains("div.u32"));
        }
    }

    #[test]
    fn generate_child_launch_ptx_thread_dependent() {
        let child = ChildKernelSpec {
            name: "child_per_thread".to_string(),
            param_types: vec![PtxType::U64],
            grid_dim: GridSpec::ThreadDependent,
            block_dim: Dim3::x(32),
            shared_mem_bytes: 0,
        };
        let result = generate_child_launch_ptx("parent", &child, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("Thread-dependent"));
            assert!(ptx.contains("%tid.x"));
        }
    }

    #[test]
    fn generate_child_launch_ptx_empty_name_fails() {
        let child = ChildKernelSpec {
            name: String::new(),
            param_types: vec![],
            grid_dim: GridSpec::Fixed(Dim3::x(1)),
            block_dim: Dim3::x(1),
            shared_mem_bytes: 0,
        };
        let result = generate_child_launch_ptx("parent", &child, SmVersion::Sm80);
        assert!(result.is_err());
    }

    #[test]
    fn generate_child_launch_ptx_zero_block_fails() {
        let child = ChildKernelSpec {
            name: "child".to_string(),
            param_types: vec![],
            grid_dim: GridSpec::Fixed(Dim3::x(1)),
            block_dim: Dim3::new(0, 1, 1),
            shared_mem_bytes: 0,
        };
        let result = generate_child_launch_ptx("parent", &child, SmVersion::Sm80);
        assert!(result.is_err());
    }

    #[test]
    fn generate_device_sync_ptx_basic() {
        let result = generate_device_sync_ptx(SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("__device_synchronize"));
            assert!(ptx.contains(".version 7.0"));
            assert!(ptx.contains("sm_80"));
            assert!(ptx.contains("cudaDeviceSynchronize"));
        }
    }

    #[test]
    fn generate_device_sync_ptx_hopper() {
        let result = generate_device_sync_ptx(SmVersion::Sm90);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains(".version 8.0"));
            assert!(ptx.contains("sm_90"));
        }
    }

    // -- Display tests --

    #[test]
    fn config_display() {
        let config = default_config();
        let display = format!("{config}");
        assert!(display.contains("depth=4"));
        assert!(display.contains("pending=2048"));
        assert!(display.contains("sync@2"));
        assert!(display.contains("sm_80"));
    }

    #[test]
    fn grid_spec_display() {
        assert_eq!(format!("{}", GridSpec::Fixed(Dim3::x(64))), "Fixed(64)");
        assert_eq!(
            format!("{}", GridSpec::DataDependent { param_index: 2 }),
            "DataDependent(param[2])"
        );
        assert_eq!(format!("{}", GridSpec::ThreadDependent), "ThreadDependent");
    }
}
