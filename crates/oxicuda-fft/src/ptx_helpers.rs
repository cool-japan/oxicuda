//! Shared PTX code-generation helpers for FFT kernels.
//!
//! These functions encapsulate common patterns for complex arithmetic,
//! twiddle factor computation, and precision-polymorphic float operations
//! within PTX body builders.
#![allow(dead_code)]

use oxicuda_ptx::builder::BodyBuilder;
use oxicuda_ptx::ir::{PtxType, Register};

use crate::types::FftPrecision;

// ---------------------------------------------------------------------------
// Precision mapping
// ---------------------------------------------------------------------------

/// Returns the PTX float type for the given precision.
pub(crate) fn ptx_float_type(precision: FftPrecision) -> PtxType {
    match precision {
        FftPrecision::Single => PtxType::F32,
        FftPrecision::Double => PtxType::F64,
    }
}

/// Returns the PTX type suffix string (without leading dot) for formatting.
pub(crate) fn ptx_type_suffix(precision: FftPrecision) -> &'static str {
    match precision {
        FftPrecision::Single => "f32",
        FftPrecision::Double => "f64",
    }
}

// ---------------------------------------------------------------------------
// Immediate loading
// ---------------------------------------------------------------------------

/// Loads a float immediate constant into a register.
pub(crate) fn load_float_imm(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    val: f64,
) -> Register {
    match precision {
        FftPrecision::Single => {
            let r = b.alloc_reg(PtxType::F32);
            let bits = (val as f32).to_bits();
            b.raw_ptx(&format!("mov.b32 {r}, 0F{bits:08X};"));
            r
        }
        FftPrecision::Double => {
            let r = b.alloc_reg(PtxType::F64);
            let bits = val.to_bits();
            b.raw_ptx(&format!("mov.b64 {r}, 0D{bits:016X};"));
            r
        }
    }
}

// ---------------------------------------------------------------------------
// Float arithmetic
// ---------------------------------------------------------------------------

/// Emits `dst = a + b` for the given precision.
pub(crate) fn add_float(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    a: Register,
    bv: Register,
) -> Register {
    match precision {
        FftPrecision::Single => b.add_f32(a, bv),
        FftPrecision::Double => b.add_f64(a, bv),
    }
}

/// Emits `dst = a - b` for the given precision.
pub(crate) fn sub_float(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    a: Register,
    bv: Register,
) -> Register {
    match precision {
        FftPrecision::Single => b.sub_f32(a, bv),
        FftPrecision::Double => b.sub_f64(a, bv),
    }
}

/// Emits `dst = a * b` for the given precision.
pub(crate) fn mul_float(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    a: Register,
    bv: Register,
) -> Register {
    let ty = ptx_float_type(precision);
    let suffix = ptx_type_suffix(precision);
    let dst = b.alloc_reg(ty);
    b.raw_ptx(&format!("mul.rn.{suffix} {dst}, {a}, {bv};"));
    dst
}

/// Emits `dst = a * b + c` (fused multiply-add) for the given precision.
pub(crate) fn fma_float(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    a: Register,
    bv: Register,
    c: Register,
) -> Register {
    match precision {
        FftPrecision::Single => b.fma_f32(a, bv, c),
        FftPrecision::Double => b.fma_f64(a, bv, c),
    }
}

/// Emits `dst = -src` for the given precision.
pub(crate) fn neg_float(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    src: Register,
) -> Register {
    let ty = ptx_float_type(precision);
    let suffix = ptx_type_suffix(precision);
    let dst = b.alloc_reg(ty);
    b.raw_ptx(&format!("neg.{suffix} {dst}, {src};"));
    dst
}

// ---------------------------------------------------------------------------
// Complex arithmetic (pair of registers: re, im)
// ---------------------------------------------------------------------------

/// Represents a complex value as a pair of PTX registers.
#[derive(Debug, Clone)]
pub(crate) struct ComplexRegs {
    /// Real part register.
    pub re: Register,
    /// Imaginary part register.
    pub im: Register,
}

/// Emits complex addition: `(a_re + b_re, a_im + b_im)`.
pub(crate) fn complex_add(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    a: &ComplexRegs,
    bv: &ComplexRegs,
) -> ComplexRegs {
    let re = add_float(b, precision, a.re.clone(), bv.re.clone());
    let im = add_float(b, precision, a.im.clone(), bv.im.clone());
    ComplexRegs { re, im }
}

/// Emits complex subtraction: `(a_re - b_re, a_im - b_im)`.
pub(crate) fn complex_sub(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    a: &ComplexRegs,
    bv: &ComplexRegs,
) -> ComplexRegs {
    let re = sub_float(b, precision, a.re.clone(), bv.re.clone());
    let im = sub_float(b, precision, a.im.clone(), bv.im.clone());
    ComplexRegs { re, im }
}

/// Emits complex multiplication: `(a_re*b_re - a_im*b_im, a_re*b_im + a_im*b_re)`.
///
/// Uses FMA for the second operand of each component:
///   re = a_re * b_re;  re = -a_im * b_im + re;
///   im = a_re * b_im;  im =  a_im * b_re + im;
pub(crate) fn complex_mul(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    a: &ComplexRegs,
    bv: &ComplexRegs,
) -> ComplexRegs {
    // re = a.re * b.re
    let re_partial = mul_float(b, precision, a.re.clone(), bv.re.clone());
    // re = re - a.im * b.im  =>  re = (-a.im) * b.im + re_partial
    let neg_aim = neg_float(b, precision, a.im.clone());
    let re = fma_float(b, precision, neg_aim, bv.im.clone(), re_partial);

    // im = a.re * b.im
    let im_partial = mul_float(b, precision, a.re.clone(), bv.im.clone());
    // im = a.im * b.re + im_partial
    let im = fma_float(b, precision, a.im.clone(), bv.re.clone(), im_partial);

    ComplexRegs { re, im }
}

/// Emits multiplication by `j` (the imaginary unit): `(-im, re)`.
///
/// This is an optimisation over general complex multiply: no multiplications
/// needed, just a negate and swap.
pub(crate) fn complex_mul_j(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    a: &ComplexRegs,
) -> ComplexRegs {
    let re = neg_float(b, precision, a.im.clone());
    let im = a.re.clone();
    ComplexRegs { re, im }
}

/// Emits multiplication by `-j`: `(im, -re)`.
pub(crate) fn complex_mul_neg_j(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    a: &ComplexRegs,
) -> ComplexRegs {
    let re = a.im.clone();
    let im = neg_float(b, precision, a.re.clone());
    ComplexRegs { re, im }
}

// ---------------------------------------------------------------------------
// Twiddle factor generation
// ---------------------------------------------------------------------------

/// Loads a precomputed twiddle factor `W_N^k = exp(-2*pi*i*k/N)` as an
/// immediate complex constant (for small N, used at code-gen time).
///
/// The `direction_sign` should be `-1.0` for forward and `+1.0` for inverse.
pub(crate) fn load_twiddle_imm(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    k: u32,
    n: u32,
    direction_sign: f64,
) -> ComplexRegs {
    let angle = direction_sign * 2.0 * std::f64::consts::PI * f64::from(k) / f64::from(n);
    let cos_val = angle.cos();
    let sin_val = angle.sin();
    let re = load_float_imm(b, precision, cos_val);
    let im = load_float_imm(b, precision, sin_val);
    ComplexRegs { re, im }
}

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------

/// Loads a complex value (re, im) from global memory at the given address.
///
/// The address points to the real part; the imaginary part is at
/// `addr + element_size`.
pub(crate) fn load_complex_global(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    addr: Register,
) -> ComplexRegs {
    let elem_bytes = precision.element_bytes();
    match precision {
        FftPrecision::Single => {
            let re = b.load_global_f32(addr.clone());
            // Offset to imaginary part
            let offset_reg = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {offset_reg}, {addr}, {elem_bytes};"));
            let im = b.load_global_f32(offset_reg);
            ComplexRegs { re, im }
        }
        FftPrecision::Double => {
            let re = b.load_global_f64(addr.clone());
            let offset_reg = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {offset_reg}, {addr}, {elem_bytes};"));
            let im = b.load_global_f64(offset_reg);
            ComplexRegs { re, im }
        }
    }
}

/// Stores a complex value (re, im) to global memory at the given address.
pub(crate) fn store_complex_global(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    addr: Register,
    val: &ComplexRegs,
) {
    let elem_bytes = precision.element_bytes();
    match precision {
        FftPrecision::Single => {
            b.store_global_f32(addr.clone(), val.re.clone());
            let offset_reg = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {offset_reg}, {addr}, {elem_bytes};"));
            b.store_global_f32(offset_reg, val.im.clone());
        }
        FftPrecision::Double => {
            b.store_global_f64(addr.clone(), val.re.clone());
            let offset_reg = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {offset_reg}, {addr}, {elem_bytes};"));
            b.store_global_f64(offset_reg, val.im.clone());
        }
    }
}

/// Converts a `u32` index to the float type matching the given precision.
pub(crate) fn cvt_u32_to_float(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    src: Register,
) -> Register {
    let suffix = ptx_type_suffix(precision);
    let ty = ptx_float_type(precision);
    let dst = b.alloc_reg(ty);
    b.raw_ptx(&format!("cvt.rn.{suffix}.u32 {dst}, {src};"));
    dst
}
