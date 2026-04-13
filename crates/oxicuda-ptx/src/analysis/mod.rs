//! Analysis passes for PTX intermediate representation.
//!
//! This module provides static analysis utilities for PTX instruction sequences:
//!
//! - [`crate::analysis::register_pressure`]: Live register analysis to estimate peak register
//!   usage, detect spill risk, and predict occupancy impact.
//! - [`crate::analysis::dead_code`]: Dead code elimination that removes instructions whose
//!   results are never consumed, while preserving side-effecting operations.
//! - [`crate::analysis::constant_folding`]: Evaluate constant expressions at compile time,
//!   replacing arithmetic on immediates with precomputed results.
//! - [`crate::analysis::strength_reduction`]: Replace expensive operations with cheaper
//!   equivalents (e.g., multiply by power-of-2 → shift).
//! - [`crate::analysis::bank_conflict`]: Shared memory bank conflict detection that identifies
//!   stride patterns causing serialization across warp threads.
//! - [`crate::analysis::arch_legality`]: Per-architecture instruction legality checking that
//!   validates instructions are legal for the target SM version.

pub mod arch_legality;
pub mod bank_conflict;
pub mod constant_folding;
pub mod dead_code;
pub mod instruction_scheduling;
pub mod kernel_fusion;
pub mod register_pressure;
pub mod strength_reduction;

pub use arch_legality::{LegalityReport, LegalityViolation, check_instruction_legality};
pub use bank_conflict::{BankConflict, BankConflictReport, analyze_bank_conflicts};
pub use constant_folding::{
    ConstantFoldingReport, fold_constant_branches, fold_constants, fold_constants_report,
};
pub use dead_code::eliminate_dead_code;
pub use instruction_scheduling::{SchedulingReport, SchedulingStrategy, schedule_instructions};
pub use kernel_fusion::{
    AccessPattern, DataDependency, FusionAnalysis, FusionCandidate, FusionConstraint, FusionPlan,
    FusionReport, FusionType, plan_fusion,
};
pub use register_pressure::{RegisterPressureReport, analyze_register_pressure};
pub use strength_reduction::{StrengthReductionReport, reduce_strength, reduce_strength_report};
