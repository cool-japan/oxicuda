//! # `OxiCUDA` PTX -- Pure Rust PTX Code Generation DSL
//!
//! `oxicuda-ptx` provides a complete Rust-native DSL and intermediate representation
//! for generating NVIDIA PTX (Parallel Thread Execution) assembly code at runtime.
//! It eliminates the dependency on `nvcc`, the proprietary CUDA Toolkit, or any
//! C/C++ compiler toolchain -- PTX text is constructed entirely from safe Rust code.
//!
//! ## Crate Architecture
//!
//! The crate is organized into six major subsystems:
//!
//! | Module         | Purpose                                                   |
//! |----------------|-----------------------------------------------------------|
//! | [`ir`]         | Typed intermediate representation for PTX instructions    |
//! | [`builder`]    | Ergonomic fluent builder API for kernel construction      |
//! | [`templates`]  | High-level templates for GEMM, reduction, softmax, etc.   |
//! | [`tensor_core`]| Tensor Core instruction helpers (WMMA, MMA, WGMMA)       |
//! | [`emit`]       | PTX text printer and validation passes                    |
//! | [`arch`]       | Architecture definitions and capability queries (`sm_75`+)  |
//! | [`cache`]      | Disk-based PTX kernel cache for avoiding regeneration     |
//! | [`error`]      | Error types for all PTX generation failure modes          |
//!
//! ## Supported Architectures
//!
//! PTX generation targets NVIDIA architectures from Turing through Blackwell:
//!
//! - **`sm_75`** -- Turing (RTX 20xx, T4)
//! - **`sm_80` / `sm_86`** -- Ampere (A100, RTX 30xx)
//! - **`sm_89`** -- Ada Lovelace (RTX 40xx, L40)
//! - **`sm_90` / `sm_90a`** -- Hopper (H100, H200)
//! - **`sm_100`** -- Blackwell (B100, B200)
//! - **`sm_120`** -- Next-generation Blackwell
//!
//! ## Design Principles
//!
//! 1. **Type-safe IR**: Every PTX register carries its type, preventing mismatched
//!    operand types at construction time rather than at `ptxas` assembly time.
//! 2. **Zero external tools**: No `nvcc`, `ptxas`, or CUDA Toolkit installation
//!    required for PTX text generation (only needed for final binary compilation).
//! 3. **Architecture-aware**: Templates and builders automatically select optimal
//!    instruction sequences based on the target [`SmVersion`].
//! 4. **Composable**: The IR, builder, and template layers compose freely --
//!    templates produce the same IR types that manual builders do.
//! 5. **Cacheable**: Generated PTX is deterministic and can be cached to disk
//!    via [`PtxCache`], keyed by kernel name, parameters, and target architecture.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxicuda_ptx::prelude::*;
//!
//! // Build a vector-add kernel targeting Ampere
//! let ptx = KernelBuilder::new("vector_add")
//!     .target(SmVersion::Sm80)
//!     .param("a_ptr", PtxType::U64)
//!     .param("b_ptr", PtxType::U64)
//!     .param("c_ptr", PtxType::U64)
//!     .param("n", PtxType::U32)
//!     .body(|b| {
//!         let gid = b.global_thread_id_x();
//!         // ... load, add, store ...
//!     })
//!     .build()
//!     .expect("PTX generation failed");
//! ```
//!
//! ## Low-Level IR Usage
//!
//! ```rust
//! use oxicuda_ptx::ir::*;
//!
//! let mut alloc = RegisterAllocator::new();
//! let tid = alloc.alloc(PtxType::U32);
//!
//! let inst = Instruction::MovSpecial {
//!     dst: tid,
//!     special: SpecialReg::TidX,
//! };
//! assert!(inst.emit().contains("%tid.x"));
//! ```
//!
//! ## Template-Based Generation
//!
//! For common patterns, use the high-level templates which handle shared memory,
//! thread coordination, and architecture-specific optimizations automatically:
//!
//! - [`templates::elementwise`] -- Unary/binary elementwise ops (add, relu, sigmoid)
//! - [`templates::reduction`] -- Parallel block-level reductions (sum, max, min)
//! - [`templates::gemm`] -- Matrix multiplication kernels
//! - [`templates::softmax`] -- Numerically stable row-wise softmax
//!
//! ## Tensor Core Support
//!
//! The [`tensor_core`] module provides configuration and code generation helpers
//! for NVIDIA Tensor Core instructions across three generations:
//!
//! - [`tensor_core::wmma`] -- WMMA for Volta/Turing+ (`wmma.load`, `wmma.mma`)
//! - [`tensor_core::mma`] -- MMA for Ampere+ (`mma.sync.aligned`)
//! - [`tensor_core::wgmma`] -- WGMMA for Hopper+ (warp-group level MMA)

// ---------------------------------------------------------------------------
// Lint configuration
// ---------------------------------------------------------------------------
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::private_intra_doc_links)]
#![deny(unsafe_code)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]

// ---------------------------------------------------------------------------
// Module declarations
// ---------------------------------------------------------------------------

/// Static analysis passes for PTX instruction sequences.
pub mod analysis;

/// NVIDIA GPU architecture definitions and capability queries.
pub mod arch;

/// Ergonomic fluent builder API for PTX kernel construction.
pub mod builder;

/// Disk-based PTX kernel cache with content-addressable storage.
pub mod cache;

/// PTX text emission (printer) and structural validation.
pub mod emit;

/// Error types for PTX code generation.
pub mod error;

/// Typed intermediate representation for PTX instructions.
pub mod ir;

/// Profile-guided code generation using autotune/profiling data.
pub mod profile_guided;

/// High-level parameterized kernel templates for common GPU workloads.
pub mod templates;

/// Tensor Core instruction configuration and generation helpers.
pub mod tensor_core;

/// Visual PTX explorer for terminal-based PTX analysis.
pub mod tui_explorer;

// ---------------------------------------------------------------------------
// Public re-exports of key types
// ---------------------------------------------------------------------------

// Analysis
pub use analysis::bank_conflict::{BankConflict, BankConflictReport, analyze_bank_conflicts};
pub use analysis::constant_folding::fold_constants;
pub use analysis::dead_code::eliminate_dead_code;
pub use analysis::instruction_scheduling::{
    SchedulingReport, SchedulingStrategy, schedule_instructions,
};
pub use analysis::register_pressure::{RegisterPressureReport, analyze_register_pressure};
pub use analysis::strength_reduction::reduce_strength;

// Architecture
pub use arch::{ArchCapabilities, SmVersion};

// IR core types
pub use ir::{
    AtomOp, BasicBlock, CacheQualifier, CmpOp, FenceScope, ImmValue, Instruction, MemorySpace,
    MmaShape, MulMode, Operand, PtxFunction, PtxModule, PtxType, Register, RegisterAllocator,
    RoundingMode, SpecialReg, SurfaceOp, TextureDim, VectorWidth, WgmmaShape, WmmaLayout, WmmaOp,
    WmmaShape,
};

// Builder API
pub use builder::{BodyBuilder, KernelBuilder};

// Error type
pub use error::PtxGenError;

// Cache
pub use cache::{PtxCache, PtxCacheKey};

// Profile-guided optimisation
pub use profile_guided::{
    Bottleneck, BranchProfile, CodeGenDecision, HotSpot, KernelProfile, MemoryAccessProfile,
    ProfileData, ProfileGuidedOptimizer, ProfileMetrics, StallReason, TileConfig,
    apply_profile_decisions,
};

// TUI Explorer
pub use tui_explorer::{ExplorerConfig, PtxExplorer};

// Emit utilities
pub use emit::printer::{emit_function, emit_function_standalone, emit_module, try_emit_module};
pub use emit::validator::{
    ValidationError, ValidationResult, validate_ptx, validate_ptx_for_target,
};

// Templates
pub use templates::broadcast::{BroadcastTemplate, MAX_BROADCAST_RANK};
pub use templates::elementwise::{ElementwiseOp, ElementwiseTemplate};
pub use templates::gemm::{EpilogueKind, GemmTemplate};
pub use templates::reduction::{ReductionOp, ReductionTemplate};
pub use templates::softmax::SoftmaxTemplate;

// Tensor Core configurations
pub use tensor_core::mma::MmaConfig;
pub use tensor_core::wgmma::WgmmaConfig;
pub use tensor_core::wmma::WmmaConfig;

// ---------------------------------------------------------------------------
// Prelude module
// ---------------------------------------------------------------------------

/// Convenient wildcard import for the most commonly needed types.
///
/// ```rust
/// use oxicuda_ptx::prelude::*;
/// ```
///
/// This re-exports the types you need for typical kernel construction:
/// architecture selection, the builder API, core IR types, error handling,
/// and the PTX cache.
pub mod prelude {
    // Architecture
    pub use crate::arch::{ArchCapabilities, SmVersion};

    // Builder API (primary entry point for most users)
    pub use crate::builder::{BodyBuilder, KernelBuilder};

    // Core IR types frequently used in builder closures
    pub use crate::ir::{
        AtomOp, BasicBlock, CacheQualifier, CmpOp, FenceScope, ImmValue, Instruction, MemorySpace,
        MulMode, Operand, PtxFunction, PtxModule, PtxType, Register, RegisterAllocator,
        RoundingMode, SpecialReg, VectorWidth,
    };

    // Tensor Core shape types (needed for MMA instructions in body closures)
    pub use crate::ir::{MmaShape, WgmmaShape, WmmaLayout, WmmaOp, WmmaShape};

    // Error type
    pub use crate::error::PtxGenError;

    // Cache
    pub use crate::cache::{PtxCache, PtxCacheKey};

    // Emit utilities
    pub use crate::emit::printer::emit_module;
    pub use crate::emit::validator::{ValidationResult, validate_ptx};

    // Template types
    pub use crate::templates::elementwise::{ElementwiseOp, ElementwiseTemplate};
    pub use crate::templates::gemm::{EpilogueKind, GemmTemplate};
    pub use crate::templates::reduction::{ReductionOp, ReductionTemplate};
    pub use crate::templates::softmax::SoftmaxTemplate;

    // Tensor Core configurations
    pub use crate::tensor_core::mma::MmaConfig;
    pub use crate::tensor_core::wgmma::WgmmaConfig;
    pub use crate::tensor_core::wmma::WmmaConfig;
}

// ---------------------------------------------------------------------------
// Features documentation module
// ---------------------------------------------------------------------------

/// Feature flags for `oxicuda-ptx`.
///
/// Currently `oxicuda-ptx` has no optional feature gates -- all functionality
/// is available by default as part of the pure-Rust design philosophy.
///
/// Future feature flags may include:
///
/// - `serde`: Serialization support for IR types and cache keys
/// - `rayon`: Parallel template generation for batch kernel compilation
/// - `tracing`: Structured logging and diagnostics for the code generation pipeline
pub mod features {}
