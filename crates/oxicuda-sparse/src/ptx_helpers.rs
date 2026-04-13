//! Sparse-specific PTX code-generation helpers.
//!
//! These functions encapsulate common patterns needed by sparse kernels:
//! warp shuffle reductions, atomic floating-point adds, and guarded
//! memory loads with bounds checking.
#![allow(dead_code)]

use oxicuda_blas::GpuFloat;
use oxicuda_ptx::builder::BodyBuilder;
use oxicuda_ptx::ir::{PtxType, Register};

/// Returns the PTX type suffix without a leading dot (e.g. `"f32"`, `"f64"`).
pub(crate) fn ptx_suffix<T: GpuFloat>() -> &'static str {
    let s = T::PTX_TYPE.as_ptx_str();
    s.strip_prefix('.').unwrap_or(s)
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

/// Emits a multiplication: `dst = a * bv`.
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

/// Emits an addition: `dst = a + bv`.
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

/// Emits warp shuffle reduction (sum) over a register using `shfl.sync.down`.
///
/// Reduces `val` across all lanes in the warp (32 threads) using XOR shuffle.
/// After this, lane 0 holds the sum of all lanes.
pub(crate) fn emit_warp_reduce_sum<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    val: Register,
) -> Register {
    let suffix = ptx_suffix::<T>();
    let bit_width = if T::PTX_TYPE == PtxType::F32 {
        "b32"
    } else {
        "b64"
    };

    let mut current = val;
    // Unroll the warp reduction: offsets 16, 8, 4, 2, 1
    for offset in [16u32, 8, 4, 2, 1] {
        let shuffled = b.alloc_reg(T::PTX_TYPE);
        b.raw_ptx(&format!(
            "shfl.sync.down.{bit_width} {shuffled}, {current}, {offset}, 31, 0xFFFFFFFF;"
        ));
        let sum = b.alloc_reg(T::PTX_TYPE);
        b.raw_ptx(&format!("add.{suffix} {sum}, {current}, {shuffled};"));
        current = sum;
    }
    current
}

/// Emits an atomic add for floating-point values.
///
/// Uses `atom.global.add.f32` (or f64 on sm_60+).
pub(crate) fn emit_atomic_add_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    addr: Register,
    val: Register,
) {
    let suffix = ptx_suffix::<T>();
    let _old = b.alloc_reg(T::PTX_TYPE);
    b.raw_ptx(&format!(
        "atom.global.add.{suffix} {_old}, [{addr}], {val};"
    ));
}

/// Emits a guarded global load with bounds checking.
///
/// Loads `T` from `base_addr + idx * sizeof(T)` if `idx < bound`,
/// otherwise returns zero.
pub(crate) fn emit_guarded_load<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    base_addr: Register,
    idx: Register,
    bound: Register,
) -> Register {
    let elem_bytes = T::SIZE as u32;
    let zero = load_float_imm::<T>(b, 0.0);

    let pred = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred}, {idx}, {bound};"));

    let addr = b.byte_offset_addr(base_addr, idx, elem_bytes);
    let loaded = load_global_float::<T>(b, addr);

    let result = b.alloc_reg(T::PTX_TYPE);
    let suffix = ptx_suffix::<T>();
    b.raw_ptx(&format!(
        "selp.{suffix} {result}, {loaded}, {zero}, {pred};"
    ));
    result
}
