//! PTX Intermediate Representation.
//!
//! This module defines the typed IR for PTX code generation. The IR models
//! the PTX ISA at a level close to the textual assembly, providing type safety
//! and register allocation while keeping the mapping to PTX text straightforward.

mod block;
mod function;
mod instruction;
mod module;
mod operand;
mod register;
mod texture;
mod types;

pub use block::BasicBlock;
pub use function::PtxFunction;
pub use instruction::{
    GridDepAction, Instruction, MmaShape, ReduxOp, SetmaxnregAction, StmatrixShape, WgmmaShape,
    WmmaLayout, WmmaOp, WmmaShape,
};
pub use module::PtxModule;
pub use operand::{ImmValue, Operand};
pub use register::{Register, RegisterAllocator};
pub use texture::{SurfaceOp, TextureDim};
pub use types::{
    AtomOp, CacheQualifier, CmpOp, FenceScope, MemorySpace, MulMode, PtxType, RoundingMode,
    SpecialReg, VectorWidth,
};
