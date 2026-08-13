//! Vectorized shared-memory load/store and TF32 rounding helpers.
//!
//! Fills three gaps identified by the CUDA-perf roofline audit against
//! [`BodyBuilder`]'s existing scalar / global-vectorized surface:
//!
//! - [`load_shared_f32x4`](BodyBuilder::load_shared_f32x4) /
//!   [`store_shared_f32x4`](BodyBuilder::store_shared_f32x4) mirror the
//!   already-verified [`load_global_f32x4`](BodyBuilder::load_global_f32x4)
//!   (which lowers to `LDG.E.128` in SASS) for the shared address space,
//!   emitting `ld.shared.v4.f32` / `st.shared.v4.f32`.
//! - [`store_global_f32x4`](BodyBuilder::store_global_f32x4) is the missing
//!   store-side counterpart of `load_global_f32x4`, emitting
//!   `st.global.v4.f32`.
//! - [`cvt_f32_to_tf32`](BodyBuilder::cvt_f32_to_tf32) rounds an `f32` value
//!   into TensorFloat-32 for `mma.sync` TF32 operands, using the only
//!   rounding mode `ptxas` accepts for this conversion on `sm_80`-`sm_89`
//!   (`cvt.rna.tf32.f32`) -- see its doc comment for the `cvt.rn.tf32.f32`
//!   sm_90-only trap and the `mov.b32`-truncation-bias trap.
//!
//! On-device numeric coverage for all four lives in
//! `vector_mem_ops_gpu_tests.rs` (gated on the `gpu-tests` feature).

use super::BodyBuilder;
use crate::ir::{Instruction, Operand, PtxType, Register, RoundingMode};

impl BodyBuilder<'_> {
    // ════════════════════════════════════════════════════════════════════
    //  Vectorized shared-memory load / store
    // ════════════════════════════════════════════════════════════════════

    /// Loads four `f32` values from shared memory as a vectorized `.v4` load.
    ///
    /// Mirrors [`load_global_f32x4`](Self::load_global_f32x4) for the shared
    /// address space. Returns an array of 4 registers containing the loaded
    /// values.
    ///
    /// `addr` must be 16-byte aligned. Declare the backing allocation with
    /// [`KernelBuilder::shared_mem_aligned`](crate::builder::KernelBuilder::shared_mem_aligned)
    /// (`align = 16`), **not** the default
    /// [`shared_mem`](crate::builder::KernelBuilder::shared_mem): a plain
    /// `f32` array's default alignment is only 4 bytes (its own element
    /// size), which `ptxas` accepts syntactically but is insufficient for
    /// `ld.shared.v4.f32`.
    pub fn load_shared_f32x4(&mut self, addr: &Register) -> [Register; 4] {
        let r0 = self.regs.alloc(PtxType::F32);
        let r1 = self.regs.alloc(PtxType::F32);
        let r2 = self.regs.alloc(PtxType::F32);
        let r3 = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Raw(format!(
            "ld.shared.v4.f32 {{{r0}, {r1}, {r2}, {r3}}}, [{addr}];"
        )));
        [r0, r1, r2, r3]
    }

    /// Stores four `f32` values to shared memory as a vectorized `.v4` store.
    ///
    /// The store-side counterpart of
    /// [`load_shared_f32x4`](Self::load_shared_f32x4); emits
    /// `st.shared.v4.f32 [addr], {v0, v1, v2, v3}`. `addr` must be 16-byte
    /// aligned -- see [`load_shared_f32x4`](Self::load_shared_f32x4) for how
    /// to declare such an allocation.
    pub fn store_shared_f32x4(&mut self, addr: &Register, vals: &[Register; 4]) {
        self.emit(Instruction::Raw(format!(
            "st.shared.v4.f32 [{addr}], {{{}, {}, {}, {}}};",
            vals[0], vals[1], vals[2], vals[3]
        )));
    }

    /// Stores four `f32` values to global memory as a vectorized `.v4` store.
    ///
    /// The store-side counterpart of
    /// [`load_global_f32x4`](Self::load_global_f32x4); emits
    /// `st.global.v4.f32 [addr], {v0, v1, v2, v3}`. `addr` must be 16-byte
    /// aligned (any pointer returned by `oxicuda-memory`'s `DeviceBuffer`
    /// allocators satisfies this -- `cuMemAlloc` always returns memory
    /// aligned to at least 256 bytes).
    pub fn store_global_f32x4(&mut self, addr: &Register, vals: &[Register; 4]) {
        self.emit(Instruction::Raw(format!(
            "st.global.v4.f32 [{addr}], {{{}, {}, {}, {}}};",
            vals[0], vals[1], vals[2], vals[3]
        )));
    }

    // ════════════════════════════════════════════════════════════════════
    //  TF32 rounding (Ampere+)
    // ════════════════════════════════════════════════════════════════════

    /// Converts (rounds) an `f32` register to TensorFloat-32 (`.tf32`), the
    /// reduced-mantissa format `mma.sync` TF32 operands are staged in.
    ///
    /// Emits `cvt.rna.tf32.f32 dst, src` -- **`.rna`** (round-to-nearest,
    /// ties-away-from-zero), deliberately not `.rn`. This is not a
    /// stylistic choice; both alternatives are traps:
    ///
    /// - **`cvt.rn.tf32.f32` is `sm_90`+ only.** The PTX ISA does define a
    ///   `.rn` form of this conversion, but `ptxas` rejects it on Ampere/Ada
    ///   (`sm_80`-`sm_89` -- this workspace's primary target, the RTX
    ///   A4000) with "not supported". `.rna` is the form `ptxas` actually
    ///   accepts there, and it remains valid on `sm_90`+ too, so it is the
    ///   only architecture-portable choice.
    /// - **`mov.b32` truncation is not a substitute.** Reinterpreting the
    ///   `f32` bit pattern and masking off the low 13 mantissa bits (plain
    ///   truncation towards zero) needs no `cvt` instruction at all, but it
    ///   is **downward-biased by construction** (magnitude never rounds up),
    ///   with a typical relative error around `1e-3`. That is not "close
    ///   enough": it sits right at the edge of (and regularly exceeds) this
    ///   codebase's `OXIONNX_CUDA_VERIFY` shadow-verification tolerances
    ///   (`ATOL 1e-4` / `RTOL 1e-3` in `oxionnx-cuda`). A TF32 GEMM/conv
    ///   path that truncates instead of calling this method reads as
    ///   numerically plausible in a quick spot check and then fails (or
    ///   silently drifts low) under that shadow verifier. Always round via
    ///   this method rather than truncating.
    ///
    /// The returned register's *declared* storage class is a plain 32-bit
    /// register (`.b32`, via [`PtxType::reg_type`]) -- TF32 shares F32's bit
    /// layout (1 sign + 8 exponent + 23-bit-container mantissa, with only
    /// the top 10 mantissa bits significant after rounding), so it can be
    /// read back with an ordinary `.f32` load/store, or fed directly into
    /// [`mma_m16n8k8_tf32_f32`](Self::mma_m16n8k8_tf32_f32) operands.
    pub fn cvt_f32_to_tf32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::TF32);
        self.emit(Instruction::Cvt {
            rnd: Some(RoundingMode::Rna),
            dst_ty: PtxType::TF32,
            src_ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;
    use crate::builder::KernelBuilder;

    /// Helper: build a kernel with the given body and return the PTX text.
    fn build_with_body<F>(f: F) -> String
    where
        F: FnOnce(&mut BodyBuilder<'_>) + 'static,
    {
        KernelBuilder::new("test")
            .target(SmVersion::Sm80)
            .param("in_ptr", PtxType::U64)
            .param("out_ptr", PtxType::U64)
            .shared_mem_aligned("tile", PtxType::F32, 4, 16)
            .body(f)
            .build()
            .expect("build should succeed")
    }

    // ── load_shared_f32x4 ──────────────────────────────────────────────

    #[test]
    fn load_shared_f32x4_emits_vectorized_shared_load() {
        let ptx = build_with_body(|b| {
            let addr = b.load_param_u64("in_ptr");
            let _vals = b.load_shared_f32x4(&addr);
            b.ret();
        });
        assert!(
            ptx.contains("ld.shared.v4.f32"),
            "expected a vectorized shared load:\n{ptx}"
        );
        // Braced 4-register destination list, PTX vector-load syntax.
        assert!(
            ptx.contains("ld.shared.v4.f32 {") && ptx.contains("}, [") && ptx.contains("];"),
            "expected `ld.shared.v4.f32 {{r0, r1, r2, r3}}, [addr];` shape:\n{ptx}"
        );
    }

    #[test]
    fn load_shared_f32x4_allocates_four_distinct_registers() {
        let ptx = build_with_body(|b| {
            let addr = b.load_param_u64("in_ptr");
            let vals = b.load_shared_f32x4(&addr);
            let names: std::collections::HashSet<_> = vals.iter().map(|r| r.name.clone()).collect();
            assert_eq!(names.len(), 4, "the four loaded registers must be distinct");
            b.ret();
        });
        assert!(ptx.contains("ld.shared.v4.f32"));
    }

    // ── store_shared_f32x4 ─────────────────────────────────────────────

    #[test]
    fn store_shared_f32x4_emits_vectorized_shared_store() {
        let ptx = build_with_body(|b| {
            let addr = b.load_param_u64("in_ptr");
            let vals = b.load_shared_f32x4(&addr);
            b.store_shared_f32x4(&addr, &vals);
            b.ret();
        });
        assert!(
            ptx.contains("st.shared.v4.f32"),
            "expected a vectorized shared store:\n{ptx}"
        );
        assert!(
            ptx.contains("st.shared.v4.f32 [") && ptx.contains("], {") && ptx.contains("};"),
            "expected `st.shared.v4.f32 [addr], {{v0, v1, v2, v3}};` shape:\n{ptx}"
        );
    }

    // ── store_global_f32x4 ─────────────────────────────────────────────

    #[test]
    fn store_global_f32x4_emits_vectorized_global_store() {
        let ptx = build_with_body(|b| {
            let in_addr = b.load_param_u64("in_ptr");
            let out_addr = b.load_param_u64("out_ptr");
            let vals = b.load_global_f32x4(&in_addr);
            b.store_global_f32x4(&out_addr, &vals);
            b.ret();
        });
        assert!(
            ptx.contains("st.global.v4.f32"),
            "expected a vectorized global store:\n{ptx}"
        );
        assert!(
            ptx.contains("st.global.v4.f32 [") && ptx.contains("], {") && ptx.contains("};"),
            "expected `st.global.v4.f32 [addr], {{v0, v1, v2, v3}};` shape:\n{ptx}"
        );
    }

    #[test]
    fn store_global_f32x4_references_the_stored_registers_in_order() {
        // Deterministic allocation order: `in_ptr`/`out_ptr` are the first
        // two `%rd` (u64) allocations (`%rd0`/`%rd1`), and
        // `load_global_f32x4` allocates the first four `%f` (f32)
        // registers (`%f0`..`%f3`) in order -- so the exact emitted
        // instruction text is predictable without inspecting the PTX from
        // inside the body closure (the closure runs before `build()`
        // returns the finished text).
        let ptx = build_with_body(|b| {
            let in_addr = b.load_param_u64("in_ptr");
            let out_addr = b.load_param_u64("out_ptr");
            let vals = b.load_global_f32x4(&in_addr);
            b.store_global_f32x4(&out_addr, &vals);
            b.ret();
        });
        let expected = "st.global.v4.f32 [%rd1], {%f0, %f1, %f2, %f3};";
        assert!(
            ptx.contains(expected),
            "expected exact instruction text {expected:?} in:\n{ptx}"
        );
    }

    // ── cvt_f32_to_tf32 ────────────────────────────────────────────────

    #[test]
    fn cvt_f32_to_tf32_emits_rna_not_rn() {
        let ptx = build_with_body(|b| {
            let addr = b.load_param_u64("in_ptr");
            let x = b.load_global_f32(addr);
            let _y = b.cvt_f32_to_tf32(x);
            b.ret();
        });
        assert!(
            ptx.contains("cvt.rna.tf32.f32"),
            "expected cvt.rna.tf32.f32 (the only form ptxas accepts on sm_80-sm_89):\n{ptx}"
        );
        // The sm_90-only `.rn` form must never be emitted by this method.
        assert!(
            !ptx.contains("cvt.rn.tf32.f32"),
            "must not emit the sm_90-only cvt.rn.tf32.f32 form:\n{ptx}"
        );
    }

    #[test]
    fn cvt_f32_to_tf32_result_is_usable_as_an_mma_operand() {
        // The destination register's declared storage class must be a plain
        // 32-bit bank (shared with ordinary u32/s32/f32 temporaries), not a
        // `.tf32`-declared register (which is not legal PTX) -- confirmed
        // indirectly: the register must round-trip through a plain f32
        // store without any additional conversion instruction.
        let ptx = build_with_body(|b| {
            let in_addr = b.load_param_u64("in_ptr");
            let out_addr = b.load_param_u64("out_ptr");
            let x = b.load_global_f32(in_addr);
            let y = b.cvt_f32_to_tf32(x);
            b.store_global_f32(out_addr, y);
            b.ret();
        });
        assert!(ptx.contains("cvt.rna.tf32.f32"));
        assert!(
            ptx.contains("st.global.f32"),
            "TF32 result must be storable via a plain f32 store:\n{ptx}"
        );
        // No `.reg .tf32` declaration should ever appear -- TF32 registers
        // are declared at their b32 storage-width class.
        assert!(
            !ptx.contains(".reg .tf32"),
            ".tf32 is not a valid .reg declaration type:\n{ptx}"
        );
    }
}
