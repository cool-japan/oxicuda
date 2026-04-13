//! Shared PTX generation utilities for the signal processing crate.
//!
//! Follows the same pattern as `oxicuda-primitives/ptx_helpers.rs`:
//! each subsystem generates a complete PTX module as a `String` and
//! loads it via `cuModuleLoadData` at first use.

use oxicuda_ptx::arch::SmVersion;

use crate::types::SignalPrecision;

/// Emit the PTX `.version` and `.target` header lines for a given SM.
///
/// Uses PTX ISA 8.5 for sm_100+, 8.0 for sm_90/sm_90a, 7.5 for sm_80/86/89,
/// and 7.0 for sm_75.
#[must_use]
pub fn ptx_header(sm: SmVersion) -> String {
    let (ptx_ver, target) = match sm {
        SmVersion::Sm120 => ("8.7", "sm_120"),
        SmVersion::Sm100 => ("8.5", "sm_100"),
        SmVersion::Sm90a => ("8.0", "sm_90a"),
        SmVersion::Sm90 => ("8.0", "sm_90"),
        SmVersion::Sm89 => ("7.5", "sm_89"),
        SmVersion::Sm86 => ("7.5", "sm_86"),
        SmVersion::Sm80 => ("7.5", "sm_80"),
        SmVersion::Sm75 => ("7.0", "sm_75"),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n")
}

/// PTX register type prefix for a given precision.
#[must_use]
pub const fn ptx_type_str(prec: SignalPrecision) -> &'static str {
    match prec {
        SignalPrecision::F32 => "f32",
        SignalPrecision::F64 => "f64",
    }
}

/// Byte size of one scalar for the given precision (used in `.align` / `ld.global`).
#[must_use]
pub const fn ptx_type_bytes(prec: SignalPrecision) -> u32 {
    match prec {
        SignalPrecision::F32 => 4,
        SignalPrecision::F64 => 8,
    }
}

/// Emit a typed register declaration block.
///
/// ```text
/// .reg .f32 %f<N>;
/// ```
#[must_use]
pub fn reg_decl(prec: SignalPrecision, name: &str, count: u32) -> String {
    format!(".reg .{} %{name}<{count}>;\n", ptx_type_str(prec))
}

/// Emit a `ld.global` instruction: `ld.global.<type> dst, [src]`.
#[must_use]
pub fn ld_global(prec: SignalPrecision, dst: &str, src: &str) -> String {
    format!("ld.global.{} {dst}, [{src}];\n", ptx_type_str(prec))
}

/// Emit a `st.global` instruction: `st.global.<type> [dst], src`.
#[must_use]
pub fn st_global(prec: SignalPrecision, dst: &str, src: &str) -> String {
    format!("st.global.{} [{dst}], {src};\n", ptx_type_str(prec))
}

/// Emit an `add` instruction for a given precision.
#[must_use]
pub fn add_instr(prec: SignalPrecision, dst: &str, a: &str, b: &str) -> String {
    format!("add.{} {dst}, {a}, {b};\n", ptx_type_str(prec))
}

/// Emit a `mul` instruction for a given precision.
#[must_use]
pub fn mul_instr(prec: SignalPrecision, dst: &str, a: &str, b: &str) -> String {
    format!("mul.{} {dst}, {a}, {b};\n", ptx_type_str(prec))
}

/// Emit an `fma.rn` instruction.
#[must_use]
pub fn fma_instr(prec: SignalPrecision, dst: &str, a: &str, b: &str, c: &str) -> String {
    format!("fma.rn.{} {dst}, {a}, {b}, {c};\n", ptx_type_str(prec))
}

/// Emit a `mov` instruction for moving an immediate f32 or f64 constant.
#[must_use]
pub fn mov_imm(prec: SignalPrecision, dst: &str, val: f64) -> String {
    match prec {
        SignalPrecision::F32 => {
            // PTX f32 immediate is hex-encoded IEEE 754 bits.
            let bits = (val as f32).to_bits();
            format!("mov.f32 {dst}, 0f{bits:08X};\n")
        }
        SignalPrecision::F64 => {
            let bits = val.to_bits();
            format!("mov.f64 {dst}, 0d{bits:016X};\n")
        }
    }
}

/// Global thread index computation preamble (1D grid).
///
/// Emits the registers and instructions to compute:
/// ```text
/// tid = blockIdx.x * blockDim.x + threadIdx.x
/// ```
/// into register `%tid`.
#[must_use]
pub fn global_tid_1d() -> String {
    r"
    .reg .u32 %tidx, %bidx, %bdimx;
    .reg .u64 %tid64, %tmp64;
    mov.u32     %tidx,  %tid.x;
    mov.u32     %bidx,  %ctaid.x;
    mov.u32     %bdimx, %ntid.x;
    mad.lo.u32  %tidx,  %bidx, %bdimx, %tidx;
    cvt.u64.u32 %tid64, %tidx;
"
    .to_owned()
}

/// Emit a grid-stride loop guard: branch to `done_label` if `tid >= n`.
#[must_use]
pub fn bounds_check(tid: &str, n_reg: &str, done_label: &str) -> String {
    format!("    setp.ge.u64 %p_oob, {tid}, {n_reg};\n    @%p_oob bra {done_label};\n")
}

/// Compute the next power-of-two >= `n` (CPU-side helper for launch config).
#[must_use]
pub const fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// True if `n` is a power of 2.
#[must_use]
pub const fn is_pow2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_header_sm80() {
        let h = ptx_header(SmVersion::Sm80);
        assert!(h.contains(".version 7.5"));
        assert!(h.contains(".target sm_80"));
        assert!(h.contains(".address_size 64"));
    }

    #[test]
    fn test_ptx_header_sm90() {
        let h = ptx_header(SmVersion::Sm90);
        assert!(h.contains(".version 8.0"));
        assert!(h.contains(".target sm_90"));
    }

    #[test]
    fn test_ptx_header_sm120() {
        let h = ptx_header(SmVersion::Sm120);
        assert!(h.contains("8.7"));
        assert!(h.contains("sm_120"));
    }

    #[test]
    fn test_mov_imm_f32() {
        let s = mov_imm(SignalPrecision::F32, "%a", 1.0);
        assert!(s.contains("mov.f32"));
        assert!(s.contains("3F800000")); // IEEE 754 for 1.0f32
    }

    #[test]
    fn test_mov_imm_f64() {
        let s = mov_imm(SignalPrecision::F64, "%a", 1.0);
        assert!(s.contains("mov.f64"));
        assert!(s.contains("3FF0000000000000")); // IEEE 754 for 1.0f64
    }

    #[test]
    fn test_next_pow2() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(8), 8);
        assert_eq!(next_pow2(1000), 1024);
    }

    #[test]
    fn test_is_pow2() {
        assert!(is_pow2(1));
        assert!(is_pow2(64));
        assert!(!is_pow2(0));
        assert!(!is_pow2(3));
        assert!(!is_pow2(1000));
    }

    #[test]
    fn test_reg_decl() {
        let s = reg_decl(SignalPrecision::F32, "a", 16);
        assert_eq!(s, ".reg .f32 %a<16>;\n");
    }

    #[test]
    fn test_ptx_type_bytes() {
        assert_eq!(ptx_type_bytes(SignalPrecision::F32), 4);
        assert_eq!(ptx_type_bytes(SignalPrecision::F64), 8);
    }
}
