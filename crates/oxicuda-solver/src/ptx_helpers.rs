//! Solver-specific PTX code generation helpers.
//!
//! Provides reusable PTX code fragments for common solver operations such as
//! finding the maximum absolute value in a column, swapping rows, scaling
//! columns below a pivot, and performing rank-1 updates on trailing matrices.

use oxicuda_blas::GpuFloat;
use oxicuda_ptx::builder::BodyBuilder;
use oxicuda_ptx::ir::{PtxType, Register};

/// Block size used by solver PTX kernels.
pub const SOLVER_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// Float arithmetic helpers (precision-generic)
// ---------------------------------------------------------------------------

/// Loads a float from global memory at `addr`, returning a register of the
/// appropriate float type.
pub fn load_global_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, addr: Register) -> Register {
    let float_ty = T::PTX_TYPE;
    let dst = b.alloc_reg(float_ty);
    b.raw_ptx(&format!(
        "ld.global{} {dst}, [{addr}];",
        float_ty.as_ptx_str()
    ));
    dst
}

/// Stores a float register to global memory at `addr`.
pub fn store_global_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, addr: Register, val: Register) {
    let float_ty = T::PTX_TYPE;
    b.raw_ptx(&format!(
        "st.global{} [{addr}], {val};",
        float_ty.as_ptx_str()
    ));
}

/// Multiplies two float registers, returning the result.
pub fn mul_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, a: Register, x: Register) -> Register {
    let float_ty = T::PTX_TYPE;
    let dst = b.alloc_reg(float_ty);
    b.raw_ptx(&format!("mul{} {dst}, {a}, {x};", float_ty.as_ptx_str()));
    dst
}

/// Adds two float registers, returning the result.
pub fn add_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, a: Register, x: Register) -> Register {
    let float_ty = T::PTX_TYPE;
    let dst = b.alloc_reg(float_ty);
    b.raw_ptx(&format!("add{} {dst}, {a}, {x};", float_ty.as_ptx_str()));
    dst
}

/// Subtracts `y` from `x` (`x - y`), returning the result.
pub fn sub_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, x: Register, y: Register) -> Register {
    let float_ty = T::PTX_TYPE;
    let dst = b.alloc_reg(float_ty);
    b.raw_ptx(&format!("sub{} {dst}, {x}, {y};", float_ty.as_ptx_str()));
    dst
}

/// Divides `x` by `y` (`x / y`), returning the result.
pub fn div_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, x: Register, y: Register) -> Register {
    let float_ty = T::PTX_TYPE;
    let dst = b.alloc_reg(float_ty);
    b.raw_ptx(&format!("div.rn{} {dst}, {x}, {y};", float_ty.as_ptx_str()));
    dst
}

/// Computes the absolute value of a float register.
pub fn abs_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, x: Register) -> Register {
    let float_ty = T::PTX_TYPE;
    let dst = b.alloc_reg(float_ty);
    b.raw_ptx(&format!("abs{} {dst}, {x};", float_ty.as_ptx_str()));
    dst
}

/// Computes the square root of a float register.
pub fn sqrt_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, x: Register) -> Register {
    let float_ty = T::PTX_TYPE;
    let dst = b.alloc_reg(float_ty);
    b.raw_ptx(&format!("sqrt.rn{} {dst}, {x};", float_ty.as_ptx_str()));
    dst
}

/// Emits a fused multiply-add: `dst = a * b + c`.
pub fn fma_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a_reg: Register,
    b_reg: Register,
    c_reg: Register,
) -> Register {
    let float_ty = T::PTX_TYPE;
    let dst = b.alloc_reg(float_ty);
    b.raw_ptx(&format!(
        "fma.rn{} {dst}, {a_reg}, {b_reg}, {c_reg};",
        float_ty.as_ptx_str()
    ));
    dst
}

/// Loads a zero constant for the given float type.
pub fn zero_float<T: GpuFloat>(b: &mut BodyBuilder<'_>) -> Register {
    let float_ty = T::PTX_TYPE;
    let dst = b.alloc_reg(float_ty);
    let zero_bits: u64 = T::gpu_zero().to_bits_u64();
    let bits_reg = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {bits_reg}, {zero_bits};"));
    // Reinterpret bits as float.
    if float_ty == PtxType::F32 {
        let bits32 = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("cvt.u32.u64 {bits32}, {bits_reg};"));
        b.raw_ptx(&format!("mov.b32 {dst}, {bits32};"));
    } else {
        b.raw_ptx(&format!("mov.b64 {dst}, {bits_reg};"));
    }
    dst
}

/// Loads a one constant for the given float type.
pub fn one_float<T: GpuFloat>(b: &mut BodyBuilder<'_>) -> Register {
    let float_ty = T::PTX_TYPE;
    let dst = b.alloc_reg(float_ty);
    let one_bits: u64 = T::gpu_one().to_bits_u64();
    let bits_reg = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {bits_reg}, {one_bits};"));
    if float_ty == PtxType::F32 {
        let bits32 = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("cvt.u32.u64 {bits32}, {bits_reg};"));
        b.raw_ptx(&format!("mov.b32 {dst}, {bits32};"));
    } else {
        b.raw_ptx(&format!("mov.b64 {dst}, {bits_reg};"));
    }
    dst
}

/// Computes a byte offset address: `base + index * elem_bytes`.
pub fn byte_offset<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    base: Register,
    index: Register,
) -> Register {
    b.byte_offset_addr(base, index, T::size_u32())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_block_size_reasonable() {
        let block_size = SOLVER_BLOCK_SIZE;
        assert!(block_size > 0);
        assert!(block_size <= 1024);
        assert_eq!(
            SOLVER_BLOCK_SIZE & (SOLVER_BLOCK_SIZE - 1),
            0,
            "must be power of 2"
        );
    }
}
