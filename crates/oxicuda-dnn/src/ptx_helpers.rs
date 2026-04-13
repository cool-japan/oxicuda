//! Shared PTX code-generation helpers for DNN kernels.
//!
//! These functions encapsulate common patterns like loading/storing floats,
//! reinterpreting bit patterns, and float arithmetic across different precisions.
#![allow(dead_code)]

use oxicuda_blas::GpuFloat;
use oxicuda_ptx::builder::BodyBuilder;
use oxicuda_ptx::ir::{PtxType, Register};

/// Returns the PTX type suffix for `T` without a leading dot (e.g. `"f32"`, `"f64"`).
pub(crate) fn ptx_type_name<T: GpuFloat>() -> &'static str {
    // PtxType::as_ptx_str() returns ".f32", ".f64", etc.
    // Strip the leading dot.
    let s = T::PTX_TYPE.as_ptx_str();
    if let Some(stripped) = s.strip_prefix('.') {
        stripped
    } else {
        s
    }
}

/// Reinterprets a u64 bit pattern as the float type `T`.
pub(crate) fn reinterpret_bits_to_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    bits: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let bits32 = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("cvt.u32.u64 {bits32}, {bits};"));
        let fval = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("mov.b32 {fval}, {bits32};"));
        fval
    } else {
        let fval = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("mov.b64 {fval}, {bits};"));
        fval
    }
}

/// Loads a float value from a global memory address.
pub(crate) fn load_global_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, addr: Register) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.load_global_f32(addr)
    } else {
        b.load_global_f64(addr)
    }
}

/// Stores a float value to a global memory address.
pub(crate) fn store_global_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    addr: Register,
    val: Register,
) {
    if T::PTX_TYPE == PtxType::F32 {
        b.store_global_f32(addr, val);
    } else {
        b.store_global_f64(addr, val);
    }
}

/// Emits a multiplication: `dst = a * b`.
pub(crate) fn mul_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let dst = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("mul.rn.f32 {dst}, {a}, {bv};"));
        dst
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("mul.rn.f64 {dst}, {a}, {bv};"));
        dst
    }
}

/// Emits an addition: `dst = a + b`.
pub(crate) fn add_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.add_f32(a, bv)
    } else {
        b.add_f64(a, bv)
    }
}

/// Emits a fused multiply-add: `dst = a * bv + c`.
pub(crate) fn fma_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
    c: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.fma_f32(a, bv, c)
    } else {
        b.fma_f64(a, bv, c)
    }
}

/// Emits `max(a, b)` for the given float type.
pub(crate) fn max_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.max_f32(a, bv)
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("max.f64 {dst}, {a}, {bv};"));
        dst
    }
}

/// Emits a division: `dst = a / b`.
pub(crate) fn div_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let dst = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("div.rn.f32 {dst}, {a}, {bv};"));
        dst
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("div.rn.f64 {dst}, {a}, {bv};"));
        dst
    }
}

/// Loads an immediate float constant into a register.
pub(crate) fn load_float_imm<T: GpuFloat>(b: &mut BodyBuilder<'_>, val: f64) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let r = b.alloc_reg(PtxType::F32);
        let bits = (val as f32).to_bits();
        b.raw_ptx(&format!("mov.b32 {r}, 0F{bits:08X};"));
        r
    } else {
        let r = b.alloc_reg(PtxType::F64);
        let bits = val.to_bits();
        b.raw_ptx(&format!("mov.b64 {r}, 0D{bits:016X};"));
        r
    }
}

/// Converts a u32 register to the float type T.
pub(crate) fn cvt_u32_to_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, src: Register) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let dst = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("cvt.rn.f32.u32 {dst}, {src};"));
        dst
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("cvt.rn.f64.u32 {dst}, {src};"));
        dst
    }
}

/// Converts a float register to u32 (round toward zero / truncate).
pub(crate) fn cvt_float_to_u32<T: GpuFloat>(b: &mut BodyBuilder<'_>, src: Register) -> Register {
    let dst = b.alloc_reg(PtxType::U32);
    if T::PTX_TYPE == PtxType::F32 {
        b.raw_ptx(&format!("cvt.rzi.u32.f32 {dst}, {src};"));
    } else {
        b.raw_ptx(&format!("cvt.rzi.u32.f64 {dst}, {src};"));
    }
    dst
}

/// Emits a set-predicate-greater-than comparison for floats.
pub(crate) fn setp_gt_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    let pred = b.alloc_reg(PtxType::Pred);
    if T::PTX_TYPE == PtxType::F32 {
        b.raw_ptx(&format!("setp.gt.f32 {pred}, {a}, {bv};"));
    } else {
        b.raw_ptx(&format!("setp.gt.f64 {pred}, {a}, {bv};"));
    }
    pred
}

/// Selects `a` if `pred` is true, else `b` (float register).
pub(crate) fn selp_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
    pred: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let dst = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("selp.f32 {dst}, {a}, {bv}, {pred};"));
        dst
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("selp.f64 {dst}, {a}, {bv}, {pred};"));
        dst
    }
}

/// Emits `abs(src)` for the given float type.
pub(crate) fn abs_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, src: Register) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.abs_f32(src)
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("abs.f64 {dst}, {src};"));
        dst
    }
}
