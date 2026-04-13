//! Tests for PTX instruction emission.

use super::super::operand::ImmValue;
use super::*;

fn make_reg(name: &str, ty: PtxType) -> Register {
    Register {
        name: name.to_string(),
        ty,
    }
}

fn make_reg_op(name: &str, ty: PtxType) -> Operand {
    Operand::Register(make_reg(name, ty))
}

#[test]
fn emit_add_f32() {
    let inst = Instruction::Add {
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        a: make_reg_op("%f1", PtxType::F32),
        b: make_reg_op("%f2", PtxType::F32),
    };
    assert_eq!(inst.emit(), "add.f32 %f0, %f1, %f2;");
}

#[test]
fn emit_sub_s32() {
    let inst = Instruction::Sub {
        ty: PtxType::S32,
        dst: make_reg("%r0", PtxType::S32),
        a: make_reg_op("%r1", PtxType::S32),
        b: make_reg_op("%r2", PtxType::S32),
    };
    assert_eq!(inst.emit(), "sub.s32 %r0, %r1, %r2;");
}

#[test]
fn emit_mul_lo_u32() {
    let inst = Instruction::Mul {
        ty: PtxType::U32,
        mode: MulMode::Lo,
        dst: make_reg("%r0", PtxType::U32),
        a: make_reg_op("%r1", PtxType::U32),
        b: make_reg_op("%r2", PtxType::U32),
    };
    assert_eq!(inst.emit(), "mul.lo.u32 %r0, %r1, %r2;");
}

#[test]
fn emit_mad_wide_s32() {
    let inst = Instruction::Mad {
        ty: PtxType::S32,
        mode: MulMode::Wide,
        dst: make_reg("%rd0", PtxType::S64),
        a: make_reg_op("%r0", PtxType::S32),
        b: make_reg_op("%r1", PtxType::S32),
        c: make_reg_op("%rd1", PtxType::S64),
    };
    assert_eq!(inst.emit(), "mad.wide.s32 %rd0, %r0, %r1, %rd1;");
}

#[test]
fn emit_fma_rn_f32() {
    let inst = Instruction::Fma {
        rnd: RoundingMode::Rn,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        a: make_reg_op("%f1", PtxType::F32),
        b: make_reg_op("%f2", PtxType::F32),
        c: make_reg_op("%f3", PtxType::F32),
    };
    assert_eq!(inst.emit(), "fma.rn.f32 %f0, %f1, %f2, %f3;");
}

#[test]
fn emit_neg_f32() {
    let inst = Instruction::Neg {
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "neg.f32 %f0, %f1;");
}

#[test]
fn emit_abs_s32() {
    let inst = Instruction::Abs {
        ty: PtxType::S32,
        dst: make_reg("%r0", PtxType::S32),
        src: make_reg_op("%r1", PtxType::S32),
    };
    assert_eq!(inst.emit(), "abs.s32 %r0, %r1;");
}

#[test]
fn emit_min_max() {
    let min = Instruction::Min {
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        a: make_reg_op("%r1", PtxType::U32),
        b: make_reg_op("%r2", PtxType::U32),
    };
    assert_eq!(min.emit(), "min.u32 %r0, %r1, %r2;");

    let max = Instruction::Max {
        ty: PtxType::F64,
        dst: make_reg("%f0", PtxType::F64),
        a: make_reg_op("%f1", PtxType::F64),
        b: make_reg_op("%f2", PtxType::F64),
    };
    assert_eq!(max.emit(), "max.f64 %f0, %f1, %f2;");
}

#[test]
fn emit_setp() {
    let inst = Instruction::SetP {
        cmp: CmpOp::Lt,
        ty: PtxType::U32,
        dst: make_reg("%p0", PtxType::Pred),
        a: make_reg_op("%r0", PtxType::U32),
        b: make_reg_op("%r1", PtxType::U32),
    };
    assert_eq!(inst.emit(), "setp.lt.u32 %p0, %r0, %r1;");
}

#[test]
fn emit_load_global() {
    let inst = Instruction::Load {
        space: MemorySpace::Global,
        qualifier: CacheQualifier::None,
        vec: VectorWidth::V1,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        addr: Operand::Address {
            base: make_reg("%rd0", PtxType::U64),
            offset: None,
        },
    };
    assert_eq!(inst.emit(), "ld.global.f32 %f0, [%rd0];");
}

#[test]
fn emit_load_with_offset() {
    let inst = Instruction::Load {
        space: MemorySpace::Global,
        qualifier: CacheQualifier::Cg,
        vec: VectorWidth::V1,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        addr: Operand::Address {
            base: make_reg("%rd0", PtxType::U64),
            offset: Some(16),
        },
    };
    assert_eq!(inst.emit(), "ld.global.cg.f32 %f0, [%rd0+16];");
}

#[test]
fn emit_store_shared() {
    let inst = Instruction::Store {
        space: MemorySpace::Shared,
        qualifier: CacheQualifier::None,
        vec: VectorWidth::V1,
        ty: PtxType::F32,
        addr: Operand::Address {
            base: make_reg("%r0", PtxType::U32),
            offset: None,
        },
        src: make_reg("%f0", PtxType::F32),
    };
    assert_eq!(inst.emit(), "st.shared.f32 [%r0], %f0;");
}

#[test]
fn emit_cp_async() {
    let inst = Instruction::CpAsync {
        bytes: 16,
        dst_shared: Operand::Register(make_reg("%r0", PtxType::U32)),
        src_global: Operand::Register(make_reg("%rd0", PtxType::U64)),
    };
    assert_eq!(inst.emit(), "cp.async.ca.shared.global [%r0], [%rd0], 16;");
}

#[test]
fn emit_cp_async_commit_wait() {
    assert_eq!(Instruction::CpAsyncCommit.emit(), "cp.async.commit_group;");
    let wait = Instruction::CpAsyncWait { n: 0 };
    assert_eq!(wait.emit(), "cp.async.wait_group 0;");
}

#[test]
fn emit_cvt() {
    let inst = Instruction::Cvt {
        rnd: Some(RoundingMode::Rn),
        dst_ty: PtxType::F32,
        src_ty: PtxType::F16,
        dst: make_reg("%f0", PtxType::F32),
        src: make_reg_op("%f1", PtxType::F16),
    };
    assert_eq!(inst.emit(), "cvt.rn.f32.f16 %f0, %f1;");

    let inst_no_rnd = Instruction::Cvt {
        rnd: None,
        dst_ty: PtxType::U32,
        src_ty: PtxType::U16,
        dst: make_reg("%r0", PtxType::U32),
        src: make_reg_op("%r1", PtxType::U16),
    };
    assert_eq!(inst_no_rnd.emit(), "cvt.u32.u16 %r0, %r1;");
}

#[test]
fn emit_branch_unconditional() {
    let inst = Instruction::Branch {
        target: "loop".to_string(),
        predicate: None,
    };
    assert_eq!(inst.emit(), "bra $loop;");
}

#[test]
fn emit_branch_conditional() {
    let inst = Instruction::Branch {
        target: "label".to_string(),
        predicate: Some((make_reg("%p0", PtxType::Pred), false)),
    };
    assert_eq!(inst.emit(), "@%p0 bra $label;");
}

#[test]
fn emit_branch_negated_predicate() {
    let inst = Instruction::Branch {
        target: "exit".to_string(),
        predicate: Some((make_reg("%p1", PtxType::Pred), true)),
    };
    assert_eq!(inst.emit(), "@!%p1 bra $exit;");
}

#[test]
fn emit_label_and_return() {
    assert_eq!(Instruction::Label("done".to_string()).emit(), "$done:");
    assert_eq!(Instruction::Return.emit(), "ret;");
}

#[test]
fn emit_bar_sync() {
    assert_eq!(Instruction::BarSync { id: 0 }.emit(), "bar.sync 0;");
}

#[test]
fn emit_bar_arrive() {
    let inst = Instruction::BarArrive { id: 1, count: 128 };
    assert_eq!(inst.emit(), "bar.arrive 1, 128;");
}

#[test]
fn emit_fence() {
    let inst = Instruction::FenceAcqRel {
        scope: FenceScope::Gpu,
    };
    assert_eq!(inst.emit(), "fence.acq_rel.gpu;");
}

#[test]
fn emit_mov_special() {
    let inst = Instruction::MovSpecial {
        dst: make_reg("%r0", PtxType::U32),
        special: SpecialReg::TidX,
    };
    assert_eq!(inst.emit(), "mov.u32 %r0, %tid.x;");
}

#[test]
fn emit_load_param() {
    let inst = Instruction::LoadParam {
        ty: PtxType::U64,
        dst: make_reg("%rd0", PtxType::U64),
        param_name: "_param_0".to_string(),
    };
    assert_eq!(inst.emit(), "ld.param.u64 %rd0, [_param_0];");
}

#[test]
fn emit_comment_and_raw() {
    assert_eq!(
        Instruction::Comment("loop start".to_string()).emit(),
        "// loop start"
    );
    assert_eq!(
        Instruction::Raw("custom.op;".to_string()).emit(),
        "custom.op;"
    );
}

#[test]
fn emit_mma_sync() {
    let d = vec![
        make_reg("%f0", PtxType::F32),
        make_reg("%f1", PtxType::F32),
        make_reg("%f2", PtxType::F32),
        make_reg("%f3", PtxType::F32),
    ];
    let a = vec![
        make_reg("%r0", PtxType::U32),
        make_reg("%r1", PtxType::U32),
        make_reg("%r2", PtxType::U32),
        make_reg("%r3", PtxType::U32),
    ];
    let b = vec![make_reg("%r4", PtxType::U32), make_reg("%r5", PtxType::U32)];
    let c = vec![
        make_reg("%f4", PtxType::F32),
        make_reg("%f5", PtxType::F32),
        make_reg("%f6", PtxType::F32),
        make_reg("%f7", PtxType::F32),
    ];
    let inst = Instruction::Mma {
        shape: MmaShape::M16N8K16,
        a_ty: PtxType::F16,
        b_ty: PtxType::F16,
        c_ty: PtxType::F32,
        d_ty: PtxType::F32,
        d_regs: d,
        a_regs: a,
        b_regs: b,
        c_regs: c,
    };
    assert_eq!(
        inst.emit(),
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f0, %f1, %f2, %f3}, {%r0, %r1, %r2, %r3}, {%r4, %r5}, {%f4, %f5, %f6, %f7};"
    );
}

// -- Bit Manipulation emit tests ----------------------------------------

#[test]
fn emit_brev_b32() {
    let inst = Instruction::Brev {
        ty: PtxType::B32,
        dst: make_reg("%r0", PtxType::B32),
        src: make_reg_op("%r1", PtxType::B32),
    };
    assert_eq!(inst.emit(), "brev.b32 %r0, %r1;");
}

#[test]
fn emit_brev_b64() {
    let inst = Instruction::Brev {
        ty: PtxType::B64,
        dst: make_reg("%rd0", PtxType::B64),
        src: make_reg_op("%rd1", PtxType::B64),
    };
    assert_eq!(inst.emit(), "brev.b64 %rd0, %rd1;");
}

#[test]
fn emit_clz_b32() {
    let inst = Instruction::Clz {
        ty: PtxType::B32,
        dst: make_reg("%r0", PtxType::U32),
        src: make_reg_op("%r1", PtxType::B32),
    };
    assert_eq!(inst.emit(), "clz.b32 %r0, %r1;");
}

#[test]
fn emit_popc_b32() {
    let inst = Instruction::Popc {
        ty: PtxType::B32,
        dst: make_reg("%r0", PtxType::U32),
        src: make_reg_op("%r1", PtxType::B32),
    };
    assert_eq!(inst.emit(), "popc.b32 %r0, %r1;");
}

#[test]
fn emit_popc_b64() {
    let inst = Instruction::Popc {
        ty: PtxType::B64,
        dst: make_reg("%r0", PtxType::U32),
        src: make_reg_op("%rd0", PtxType::B64),
    };
    assert_eq!(inst.emit(), "popc.b64 %r0, %rd0;");
}

#[test]
fn emit_bfind_u32() {
    let inst = Instruction::Bfind {
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        src: make_reg_op("%r1", PtxType::U32),
    };
    assert_eq!(inst.emit(), "bfind.u32 %r0, %r1;");
}

#[test]
fn emit_bfind_s32() {
    let inst = Instruction::Bfind {
        ty: PtxType::S32,
        dst: make_reg("%r0", PtxType::U32),
        src: make_reg_op("%r1", PtxType::S32),
    };
    assert_eq!(inst.emit(), "bfind.s32 %r0, %r1;");
}

#[test]
fn emit_bfe_u32() {
    let inst = Instruction::Bfe {
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        src: make_reg_op("%r1", PtxType::U32),
        start: make_reg_op("%r2", PtxType::U32),
        len: make_reg_op("%r3", PtxType::U32),
    };
    assert_eq!(inst.emit(), "bfe.u32 %r0, %r1, %r2, %r3;");
}

#[test]
fn emit_bfe_s32() {
    let inst = Instruction::Bfe {
        ty: PtxType::S32,
        dst: make_reg("%r0", PtxType::S32),
        src: make_reg_op("%r1", PtxType::S32),
        start: make_reg_op("%r2", PtxType::U32),
        len: make_reg_op("%r3", PtxType::U32),
    };
    assert_eq!(inst.emit(), "bfe.s32 %r0, %r1, %r2, %r3;");
}

#[test]
fn emit_bfi_b32() {
    let inst = Instruction::Bfi {
        ty: PtxType::B32,
        dst: make_reg("%r0", PtxType::B32),
        insert: make_reg_op("%r1", PtxType::B32),
        base: make_reg_op("%r2", PtxType::B32),
        start: make_reg_op("%r3", PtxType::U32),
        len: make_reg_op("%r4", PtxType::U32),
    };
    assert_eq!(inst.emit(), "bfi.b32 %r0, %r1, %r2, %r3, %r4;");
}

// -- Special Math emit tests --------------------------------------------

#[test]
fn emit_rcp_rn_f32() {
    let inst = Instruction::Rcp {
        rnd: Some(RoundingMode::Rn),
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "rcp.rn.f32 %f0, %f1;");
}

#[test]
fn emit_rcp_no_rounding() {
    let inst = Instruction::Rcp {
        rnd: None,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "rcp.f32 %f0, %f1;");
}

#[test]
fn emit_rcp_rn_f64() {
    let inst = Instruction::Rcp {
        rnd: Some(RoundingMode::Rn),
        ty: PtxType::F64,
        dst: make_reg("%fd0", PtxType::F64),
        src: make_reg_op("%fd1", PtxType::F64),
    };
    assert_eq!(inst.emit(), "rcp.rn.f64 %fd0, %fd1;");
}

#[test]
fn emit_rsqrt_approx_f32() {
    let inst = Instruction::Rsqrt {
        approx: true,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "rsqrt.approx.f32 %f0, %f1;");
}

#[test]
fn emit_rsqrt_no_approx() {
    let inst = Instruction::Rsqrt {
        approx: false,
        ty: PtxType::F64,
        dst: make_reg("%fd0", PtxType::F64),
        src: make_reg_op("%fd1", PtxType::F64),
    };
    assert_eq!(inst.emit(), "rsqrt.f64 %fd0, %fd1;");
}

#[test]
fn emit_sqrt_rn_f32() {
    let inst = Instruction::Sqrt {
        rnd: Some(RoundingMode::Rn),
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "sqrt.rn.f32 %f0, %f1;");
}

#[test]
fn emit_sqrt_rn_f64() {
    let inst = Instruction::Sqrt {
        rnd: Some(RoundingMode::Rn),
        ty: PtxType::F64,
        dst: make_reg("%fd0", PtxType::F64),
        src: make_reg_op("%fd1", PtxType::F64),
    };
    assert_eq!(inst.emit(), "sqrt.rn.f64 %fd0, %fd1;");
}

#[test]
fn emit_ex2_approx_f32() {
    let inst = Instruction::Ex2 {
        approx: true,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "ex2.approx.f32 %f0, %f1;");
}

#[test]
fn emit_lg2_approx_f32() {
    let inst = Instruction::Lg2 {
        approx: true,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "lg2.approx.f32 %f0, %f1;");
}

#[test]
fn emit_sin_approx_f32() {
    let inst = Instruction::Sin {
        approx: true,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "sin.approx.f32 %f0, %f1;");
}

#[test]
fn emit_cos_approx_f32() {
    let inst = Instruction::Cos {
        approx: true,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "cos.approx.f32 %f0, %f1;");
}

#[test]
fn emit_sin_no_approx() {
    let inst = Instruction::Sin {
        approx: false,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "sin.f32 %f0, %f1;");
}

// -- Atomic instruction emit tests --------------------------------------

#[test]
fn emit_atom_global_add_f32() {
    let inst = Instruction::Atom {
        space: MemorySpace::Global,
        op: AtomOp::Add,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        addr: make_reg_op("%rd0", PtxType::U64),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "atom.global.add.f32 %f0, [%rd0], %f1;");
}

#[test]
fn emit_atom_global_add_u32() {
    let inst = Instruction::Atom {
        space: MemorySpace::Global,
        op: AtomOp::Add,
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        addr: make_reg_op("%rd0", PtxType::U64),
        src: make_reg_op("%r1", PtxType::U32),
    };
    assert_eq!(inst.emit(), "atom.global.add.u32 %r0, [%rd0], %r1;");
}

#[test]
fn emit_atom_shared_add_f32() {
    let inst = Instruction::Atom {
        space: MemorySpace::Shared,
        op: AtomOp::Add,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        addr: make_reg_op("%r0", PtxType::U32),
        src: make_reg_op("%f1", PtxType::F32),
    };
    assert_eq!(inst.emit(), "atom.shared.add.f32 %f0, [%r0], %f1;");
}

#[test]
fn emit_atom_global_exch_u32() {
    let inst = Instruction::Atom {
        space: MemorySpace::Global,
        op: AtomOp::Exch,
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        addr: make_reg_op("%rd0", PtxType::U64),
        src: make_reg_op("%r1", PtxType::U32),
    };
    assert_eq!(inst.emit(), "atom.global.exch.u32 %r0, [%rd0], %r1;");
}

#[test]
fn emit_atom_global_min_max() {
    let min_inst = Instruction::Atom {
        space: MemorySpace::Global,
        op: AtomOp::Min,
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        addr: make_reg_op("%rd0", PtxType::U64),
        src: make_reg_op("%r1", PtxType::U32),
    };
    assert_eq!(min_inst.emit(), "atom.global.min.u32 %r0, [%rd0], %r1;");

    let max_inst = Instruction::Atom {
        space: MemorySpace::Global,
        op: AtomOp::Max,
        ty: PtxType::S32,
        dst: make_reg("%r0", PtxType::S32),
        addr: make_reg_op("%rd0", PtxType::U64),
        src: make_reg_op("%r1", PtxType::S32),
    };
    assert_eq!(max_inst.emit(), "atom.global.max.s32 %r0, [%rd0], %r1;");
}

#[test]
fn emit_atom_global_bitwise_ops() {
    for (op, name) in [
        (AtomOp::And, "and"),
        (AtomOp::Or, "or"),
        (AtomOp::Xor, "xor"),
    ] {
        let inst = Instruction::Atom {
            space: MemorySpace::Global,
            op,
            ty: PtxType::B32,
            dst: make_reg("%r0", PtxType::B32),
            addr: make_reg_op("%rd0", PtxType::U64),
            src: make_reg_op("%r1", PtxType::B32),
        };
        assert_eq!(
            inst.emit(),
            format!("atom.global.{name}.b32 %r0, [%rd0], %r1;")
        );
    }
}

#[test]
fn emit_atom_global_inc_dec() {
    let inc = Instruction::Atom {
        space: MemorySpace::Global,
        op: AtomOp::Inc,
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        addr: make_reg_op("%rd0", PtxType::U64),
        src: make_reg_op("%r1", PtxType::U32),
    };
    assert_eq!(inc.emit(), "atom.global.inc.u32 %r0, [%rd0], %r1;");

    let dec = Instruction::Atom {
        space: MemorySpace::Global,
        op: AtomOp::Dec,
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        addr: make_reg_op("%rd0", PtxType::U64),
        src: make_reg_op("%r1", PtxType::U32),
    };
    assert_eq!(dec.emit(), "atom.global.dec.u32 %r0, [%rd0], %r1;");
}

#[test]
fn emit_atom_cas_global_u32() {
    let inst = Instruction::AtomCas {
        space: MemorySpace::Global,
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        addr: make_reg_op("%rd0", PtxType::U64),
        compare: make_reg_op("%r1", PtxType::U32),
        value: make_reg_op("%r2", PtxType::U32),
    };
    assert_eq!(inst.emit(), "atom.global.cas.u32 %r0, [%rd0], %r1, %r2;");
}

#[test]
fn emit_atom_cas_global_u64() {
    let inst = Instruction::AtomCas {
        space: MemorySpace::Global,
        ty: PtxType::U64,
        dst: make_reg("%rd0", PtxType::U64),
        addr: make_reg_op("%rd1", PtxType::U64),
        compare: make_reg_op("%rd2", PtxType::U64),
        value: make_reg_op("%rd3", PtxType::U64),
    };
    assert_eq!(inst.emit(), "atom.global.cas.u64 %rd0, [%rd1], %rd2, %rd3;");
}

#[test]
fn emit_atom_cas_shared() {
    let inst = Instruction::AtomCas {
        space: MemorySpace::Shared,
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        addr: make_reg_op("%r1", PtxType::U32),
        compare: make_reg_op("%r2", PtxType::U32),
        value: make_reg_op("%r3", PtxType::U32),
    };
    assert_eq!(inst.emit(), "atom.shared.cas.u32 %r0, [%r1], %r2, %r3;");
}

#[test]
fn emit_red_global_add_f32() {
    let inst = Instruction::Red {
        space: MemorySpace::Global,
        op: AtomOp::Add,
        ty: PtxType::F32,
        addr: make_reg_op("%rd0", PtxType::U64),
        src: make_reg_op("%f0", PtxType::F32),
    };
    assert_eq!(inst.emit(), "red.global.add.f32 [%rd0], %f0;");
}

#[test]
fn emit_red_global_add_u32() {
    let inst = Instruction::Red {
        space: MemorySpace::Global,
        op: AtomOp::Add,
        ty: PtxType::U32,
        addr: make_reg_op("%rd0", PtxType::U64),
        src: make_reg_op("%r0", PtxType::U32),
    };
    assert_eq!(inst.emit(), "red.global.add.u32 [%rd0], %r0;");
}

#[test]
fn emit_red_shared_add_f32() {
    let inst = Instruction::Red {
        space: MemorySpace::Shared,
        op: AtomOp::Add,
        ty: PtxType::F32,
        addr: make_reg_op("%r0", PtxType::U32),
        src: make_reg_op("%f0", PtxType::F32),
    };
    assert_eq!(inst.emit(), "red.shared.add.f32 [%r0], %f0;");
}

#[test]
fn emit_atom_global_add_u64() {
    let inst = Instruction::Atom {
        space: MemorySpace::Global,
        op: AtomOp::Add,
        ty: PtxType::U64,
        dst: make_reg("%rd0", PtxType::U64),
        addr: make_reg_op("%rd1", PtxType::U64),
        src: make_reg_op("%rd2", PtxType::U64),
    };
    assert_eq!(inst.emit(), "atom.global.add.u64 %rd0, [%rd1], %rd2;");
}

#[test]
fn emit_atom_global_add_f64() {
    let inst = Instruction::Atom {
        space: MemorySpace::Global,
        op: AtomOp::Add,
        ty: PtxType::F64,
        dst: make_reg("%fd0", PtxType::F64),
        addr: make_reg_op("%rd0", PtxType::U64),
        src: make_reg_op("%fd1", PtxType::F64),
    };
    assert_eq!(inst.emit(), "atom.global.add.f64 %fd0, [%rd0], %fd1;");
}

// -- MadLo tests -------------------------------------------------------

#[test]
fn emit_mad_lo_s32() {
    let inst = Instruction::MadLo {
        typ: PtxType::S32,
        dst: make_reg("%r0", PtxType::S32),
        a: make_reg_op("%r1", PtxType::S32),
        b: make_reg_op("%r2", PtxType::S32),
        c: make_reg_op("%r3", PtxType::S32),
    };
    assert_eq!(inst.emit(), "mad.lo.s32 %r0, %r1, %r2, %r3;");
}

#[test]
fn emit_mad_lo_u32() {
    let inst = Instruction::MadLo {
        typ: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        a: make_reg_op("%r1", PtxType::U32),
        b: make_reg_op("%r2", PtxType::U32),
        c: make_reg_op("%r3", PtxType::U32),
    };
    assert_eq!(inst.emit(), "mad.lo.u32 %r0, %r1, %r2, %r3;");
}

#[test]
fn emit_mad_lo_s64() {
    let inst = Instruction::MadLo {
        typ: PtxType::S64,
        dst: make_reg("%rd0", PtxType::S64),
        a: make_reg_op("%rd1", PtxType::S64),
        b: make_reg_op("%rd2", PtxType::S64),
        c: make_reg_op("%rd3", PtxType::S64),
    };
    assert_eq!(inst.emit(), "mad.lo.s64 %rd0, %rd1, %rd2, %rd3;");
}

#[test]
fn emit_mad_lo_u64() {
    let inst = Instruction::MadLo {
        typ: PtxType::U64,
        dst: make_reg("%rd0", PtxType::U64),
        a: make_reg_op("%rd1", PtxType::U64),
        b: make_reg_op("%rd2", PtxType::U64),
        c: make_reg_op("%rd3", PtxType::U64),
    };
    assert_eq!(inst.emit(), "mad.lo.u64 %rd0, %rd1, %rd2, %rd3;");
}

// -- MadHi tests -------------------------------------------------------

#[test]
fn emit_mad_hi_s32() {
    let inst = Instruction::MadHi {
        typ: PtxType::S32,
        dst: make_reg("%r0", PtxType::S32),
        a: make_reg_op("%r1", PtxType::S32),
        b: make_reg_op("%r2", PtxType::S32),
        c: make_reg_op("%r3", PtxType::S32),
    };
    assert_eq!(inst.emit(), "mad.hi.s32 %r0, %r1, %r2, %r3;");
}

#[test]
fn emit_mad_hi_u32() {
    let inst = Instruction::MadHi {
        typ: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        a: make_reg_op("%r1", PtxType::U32),
        b: make_reg_op("%r2", PtxType::U32),
        c: make_reg_op("%r3", PtxType::U32),
    };
    assert_eq!(inst.emit(), "mad.hi.u32 %r0, %r1, %r2, %r3;");
}

#[test]
fn emit_mad_hi_s64() {
    let inst = Instruction::MadHi {
        typ: PtxType::S64,
        dst: make_reg("%rd0", PtxType::S64),
        a: make_reg_op("%rd1", PtxType::S64),
        b: make_reg_op("%rd2", PtxType::S64),
        c: make_reg_op("%rd3", PtxType::S64),
    };
    assert_eq!(inst.emit(), "mad.hi.s64 %rd0, %rd1, %rd2, %rd3;");
}

#[test]
fn emit_mad_hi_u64() {
    let inst = Instruction::MadHi {
        typ: PtxType::U64,
        dst: make_reg("%rd0", PtxType::U64),
        a: make_reg_op("%rd1", PtxType::U64),
        b: make_reg_op("%rd2", PtxType::U64),
        c: make_reg_op("%rd3", PtxType::U64),
    };
    assert_eq!(inst.emit(), "mad.hi.u64 %rd0, %rd1, %rd2, %rd3;");
}

// -- MadWide tests -----------------------------------------------------

#[test]
fn emit_mad_wide_s16() {
    let inst = Instruction::MadWide {
        src_typ: PtxType::S16,
        dst: make_reg("%r0", PtxType::S32),
        a: make_reg_op("%rh0", PtxType::S16),
        b: make_reg_op("%rh1", PtxType::S16),
        c: make_reg_op("%r1", PtxType::S32),
    };
    assert_eq!(inst.emit(), "mad.wide.s16 %r0, %rh0, %rh1, %r1;");
}

#[test]
fn emit_mad_wide_u16() {
    let inst = Instruction::MadWide {
        src_typ: PtxType::U16,
        dst: make_reg("%r0", PtxType::U32),
        a: make_reg_op("%rh0", PtxType::U16),
        b: make_reg_op("%rh1", PtxType::U16),
        c: make_reg_op("%r1", PtxType::U32),
    };
    assert_eq!(inst.emit(), "mad.wide.u16 %r0, %rh0, %rh1, %r1;");
}

#[test]
fn emit_mad_wide_s32_to_s64() {
    let inst = Instruction::MadWide {
        src_typ: PtxType::S32,
        dst: make_reg("%rd0", PtxType::S64),
        a: make_reg_op("%r0", PtxType::S32),
        b: make_reg_op("%r1", PtxType::S32),
        c: make_reg_op("%rd1", PtxType::S64),
    };
    assert_eq!(inst.emit(), "mad.wide.s32 %rd0, %r0, %r1, %rd1;");
}

#[test]
fn emit_mad_wide_u32_to_u64() {
    let inst = Instruction::MadWide {
        src_typ: PtxType::U32,
        dst: make_reg("%rd0", PtxType::U64),
        a: make_reg_op("%r0", PtxType::U32),
        b: make_reg_op("%r1", PtxType::U32),
        c: make_reg_op("%rd1", PtxType::U64),
    };
    assert_eq!(inst.emit(), "mad.wide.u32 %rd0, %r0, %r1, %rd1;");
}

// -- Video instruction emit tests ----------------------------------------

#[test]
fn emit_dp4a_u32_u32() {
    let inst = Instruction::Dp4a {
        dst: make_reg("%r0", PtxType::S32),
        a: make_reg_op("%r1", PtxType::U32),
        b: make_reg_op("%r2", PtxType::U32),
        c: make_reg_op("%r3", PtxType::S32),
        signed_a: false,
        signed_b: false,
    };
    assert_eq!(inst.emit(), "dp4a.u32.u32 %r0, %r1, %r2, %r3;");
}

#[test]
fn emit_dp4a_s32_s32() {
    let inst = Instruction::Dp4a {
        dst: make_reg("%r0", PtxType::S32),
        a: make_reg_op("%r1", PtxType::S32),
        b: make_reg_op("%r2", PtxType::S32),
        c: make_reg_op("%r3", PtxType::S32),
        signed_a: true,
        signed_b: true,
    };
    assert_eq!(inst.emit(), "dp4a.s32.s32 %r0, %r1, %r2, %r3;");
}

#[test]
fn emit_dp4a_s32_u32() {
    let inst = Instruction::Dp4a {
        dst: make_reg("%r0", PtxType::S32),
        a: make_reg_op("%r1", PtxType::S32),
        b: make_reg_op("%r2", PtxType::U32),
        c: make_reg_op("%r3", PtxType::S32),
        signed_a: true,
        signed_b: false,
    };
    assert_eq!(inst.emit(), "dp4a.s32.u32 %r0, %r1, %r2, %r3;");
}

#[test]
fn emit_dp4a_u32_s32() {
    let inst = Instruction::Dp4a {
        dst: make_reg("%r0", PtxType::S32),
        a: make_reg_op("%r1", PtxType::U32),
        b: make_reg_op("%r2", PtxType::S32),
        c: make_reg_op("%r3", PtxType::S32),
        signed_a: false,
        signed_b: true,
    };
    assert_eq!(inst.emit(), "dp4a.u32.s32 %r0, %r1, %r2, %r3;");
}

#[test]
fn emit_dp2a_lo_u32_u32() {
    let inst = Instruction::Dp2a {
        dst: make_reg("%r0", PtxType::S32),
        a: make_reg_op("%r1", PtxType::U32),
        b: make_reg_op("%r2", PtxType::U32),
        c: make_reg_op("%r3", PtxType::S32),
        signed_a: false,
        signed_b: false,
        lo: true,
    };
    assert_eq!(inst.emit(), "dp2a.lo.u32.u32 %r0, %r1, %r2, %r3;");
}

#[test]
fn emit_dp2a_hi_s32_s32() {
    let inst = Instruction::Dp2a {
        dst: make_reg("%r0", PtxType::S32),
        a: make_reg_op("%r1", PtxType::S32),
        b: make_reg_op("%r2", PtxType::S32),
        c: make_reg_op("%r3", PtxType::S32),
        signed_a: true,
        signed_b: true,
        lo: false,
    };
    assert_eq!(inst.emit(), "dp2a.hi.s32.s32 %r0, %r1, %r2, %r3;");
}

// -- Pragma emit tests ---------------------------------------------------

#[test]
fn emit_pragma_unroll() {
    let inst = Instruction::Pragma("unroll 4".to_string());
    assert_eq!(inst.emit(), ".pragma \"unroll 4\";");
}

#[test]
fn emit_pragma_nounroll() {
    let inst = Instruction::Pragma("nounroll".to_string());
    assert_eq!(inst.emit(), ".pragma \"nounroll\";");
}

// -- Texture / Surface emit tests ------------------------------------------

#[test]
fn emit_tex_1d() {
    let inst = Instruction::Tex1d {
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        tex_ref: "my_tex".to_string(),
        coord: make_reg_op("%r0", PtxType::S32),
    };
    assert_eq!(inst.emit(), "tex.1d.v4.f32.s32 %f0, [my_tex, {%r0}];");
}

#[test]
fn emit_tex_2d() {
    let inst = Instruction::Tex2d {
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        tex_ref: "my_tex2d".to_string(),
        coord_x: make_reg_op("%r0", PtxType::S32),
        coord_y: make_reg_op("%r1", PtxType::S32),
    };
    assert_eq!(
        inst.emit(),
        "tex.2d.v4.f32.s32 %f0, [my_tex2d, {%r0, %r1}];"
    );
}

#[test]
fn emit_tex_3d() {
    let inst = Instruction::Tex3d {
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        tex_ref: "my_tex3d".to_string(),
        coord_x: make_reg_op("%r0", PtxType::S32),
        coord_y: make_reg_op("%r1", PtxType::S32),
        coord_z: make_reg_op("%r2", PtxType::S32),
    };
    assert_eq!(
        inst.emit(),
        "tex.3d.v4.f32.s32 %f0, [my_tex3d, {%r0, %r1, %r2}];"
    );
}

#[test]
fn emit_surf_load() {
    let inst = Instruction::SurfLoad {
        ty: PtxType::B32,
        dst: make_reg("%r0", PtxType::B32),
        surf_ref: "my_surf".to_string(),
        coord: make_reg_op("%r1", PtxType::S32),
    };
    assert_eq!(inst.emit(), "suld.b.1d.b32 %r0, [my_surf, {%r1}];");
}

#[test]
fn emit_surf_store() {
    let inst = Instruction::SurfStore {
        ty: PtxType::B32,
        surf_ref: "my_surf".to_string(),
        coord: make_reg_op("%r0", PtxType::S32),
        src: make_reg("%r1", PtxType::B32),
    };
    assert_eq!(inst.emit(), "sust.b.1d.b32 [my_surf, {%r0}], %r1;");
}

#[test]
fn emit_tex_1d_f16() {
    let inst = Instruction::Tex1d {
        ty: PtxType::F16,
        dst: make_reg("%h0", PtxType::F16),
        tex_ref: "tex_f16".to_string(),
        coord: make_reg_op("%r0", PtxType::S32),
    };
    assert_eq!(inst.emit(), "tex.1d.v4.f16.s32 %h0, [tex_f16, {%r0}];");
}

#[test]
fn emit_tex_2d_b32() {
    let inst = Instruction::Tex2d {
        ty: PtxType::B32,
        dst: make_reg("%r0", PtxType::B32),
        tex_ref: "tex_b32".to_string(),
        coord_x: make_reg_op("%r1", PtxType::S32),
        coord_y: make_reg_op("%r2", PtxType::S32),
    };
    assert_eq!(inst.emit(), "tex.2d.v4.b32.s32 %r0, [tex_b32, {%r1, %r2}];");
}

#[test]
fn emit_surf_load_u32() {
    let inst = Instruction::SurfLoad {
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        surf_ref: "surf_u32".to_string(),
        coord: make_reg_op("%r1", PtxType::S32),
    };
    assert_eq!(inst.emit(), "suld.b.1d.u32 %r0, [surf_u32, {%r1}];");
}

#[test]
fn emit_surf_store_f32() {
    let inst = Instruction::SurfStore {
        ty: PtxType::F32,
        surf_ref: "surf_f32".to_string(),
        coord: make_reg_op("%r0", PtxType::S32),
        src: make_reg("%f0", PtxType::F32),
    };
    assert_eq!(inst.emit(), "sust.b.1d.f32 [surf_f32, {%r0}], %f0;");
}

#[test]
fn builder_tex_1d() {
    use crate::arch::SmVersion;
    use crate::ir::RegisterAllocator;

    let mut regs = RegisterAllocator::new();
    let mut instructions = Vec::new();
    let params: Vec<String> = vec![];
    let mut builder =
        crate::BodyBuilder::new(&mut regs, &mut instructions, &params, SmVersion::Sm80);
    let coord = Operand::Immediate(ImmValue::U32(42));
    let dst = builder.tex_1d(PtxType::F32, "my_tex", coord);
    assert_eq!(dst.ty, PtxType::F32);
    assert_eq!(instructions.len(), 1);
    assert!(instructions[0].emit().starts_with("tex.1d.v4.f32.s32"));
}

#[test]
fn builder_tex_2d() {
    use crate::arch::SmVersion;
    use crate::ir::RegisterAllocator;

    let mut regs = RegisterAllocator::new();
    let mut instructions = Vec::new();
    let params: Vec<String> = vec![];
    let mut builder =
        crate::BodyBuilder::new(&mut regs, &mut instructions, &params, SmVersion::Sm80);
    let cx = Operand::Immediate(ImmValue::U32(10));
    let cy = Operand::Immediate(ImmValue::U32(20));
    let dst = builder.tex_2d(PtxType::F32, "tex2d", cx, cy);
    assert_eq!(dst.ty, PtxType::F32);
    assert_eq!(instructions.len(), 1);
    assert!(instructions[0].emit().starts_with("tex.2d.v4.f32.s32"));
}

#[test]
fn builder_tex_3d() {
    use crate::arch::SmVersion;
    use crate::ir::RegisterAllocator;

    let mut regs = RegisterAllocator::new();
    let mut instructions = Vec::new();
    let params: Vec<String> = vec![];
    let mut builder =
        crate::BodyBuilder::new(&mut regs, &mut instructions, &params, SmVersion::Sm80);
    let cx = Operand::Immediate(ImmValue::U32(1));
    let cy = Operand::Immediate(ImmValue::U32(2));
    let cz = Operand::Immediate(ImmValue::U32(3));
    let dst = builder.tex_3d(PtxType::F32, "tex3d", cx, cy, cz);
    assert_eq!(dst.ty, PtxType::F32);
    assert_eq!(instructions.len(), 1);
    assert!(instructions[0].emit().starts_with("tex.3d.v4.f32.s32"));
}

#[test]
fn builder_surf_load() {
    use crate::arch::SmVersion;
    use crate::ir::RegisterAllocator;

    let mut regs = RegisterAllocator::new();
    let mut instructions = Vec::new();
    let params: Vec<String> = vec![];
    let mut builder =
        crate::BodyBuilder::new(&mut regs, &mut instructions, &params, SmVersion::Sm80);
    let coord = Operand::Immediate(ImmValue::U32(0));
    let dst = builder.surf_load(PtxType::B32, "my_surf", coord);
    assert_eq!(dst.ty, PtxType::B32);
    assert_eq!(instructions.len(), 1);
    assert!(instructions[0].emit().starts_with("suld.b.1d.b32"));
}

#[test]
fn builder_surf_store() {
    use crate::arch::SmVersion;
    use crate::ir::RegisterAllocator;

    let mut regs = RegisterAllocator::new();
    let mut instructions = Vec::new();
    let params: Vec<String> = vec![];
    let src = regs.alloc(PtxType::F32);
    let mut builder =
        crate::BodyBuilder::new(&mut regs, &mut instructions, &params, SmVersion::Sm80);
    let coord = Operand::Immediate(ImmValue::U32(0));
    builder.surf_store(PtxType::F32, "my_surf", coord, src);
    assert_eq!(instructions.len(), 1);
    assert!(instructions[0].emit().starts_with("sust.b.1d.f32"));
}

#[test]
fn tex_instructions_in_scheduling() {
    use crate::analysis::instruction_scheduling::{SchedulingStrategy, schedule_instructions};

    let instructions = vec![
        Instruction::Tex1d {
            ty: PtxType::F32,
            dst: make_reg("%f0", PtxType::F32),
            tex_ref: "tex".to_string(),
            coord: make_reg_op("%r0", PtxType::S32),
        },
        // Independent arithmetic that can be scheduled alongside the tex fetch
        Instruction::Add {
            ty: PtxType::F32,
            dst: make_reg("%f1", PtxType::F32),
            a: make_reg_op("%f2", PtxType::F32),
            b: make_reg_op("%f3", PtxType::F32),
        },
        // Use the texture result
        Instruction::Add {
            ty: PtxType::F32,
            dst: make_reg("%f4", PtxType::F32),
            a: make_reg_op("%f0", PtxType::F32),
            b: make_reg_op("%f1", PtxType::F32),
        },
    ];
    let (scheduled, report) = schedule_instructions(&instructions, SchedulingStrategy::MaxIlp);
    assert_eq!(scheduled.len(), 3);
    assert_eq!(report.original_count, 3);
}

// ---------------------------------------------------------------------------
// PTX 8.x instruction emission tests
// ---------------------------------------------------------------------------

#[test]
fn emit_redux_add() {
    let inst = Instruction::Redux {
        op: super::ReduxOp::Add,
        dst: make_reg("%r0", PtxType::U32),
        src: make_reg_op("%r1", PtxType::U32),
        membership_mask: 0xFFFF_FFFF,
    };
    assert_eq!(inst.emit(), "redux.sync.add.u32 %r0, %r1, 0xffffffff;");
}

#[test]
fn emit_redux_max() {
    let inst = Instruction::Redux {
        op: super::ReduxOp::Max,
        dst: make_reg("%r0", PtxType::U32),
        src: make_reg_op("%r1", PtxType::U32),
        membership_mask: 0x0000_FFFF,
    };
    assert_eq!(inst.emit(), "redux.sync.max.u32 %r0, %r1, 0x0000ffff;");
}

#[test]
fn emit_stmatrix_m8n8x4() {
    let inst = Instruction::Stmatrix {
        dst_addr: make_reg_op("%r0", PtxType::U32),
        src: make_reg("%r1", PtxType::B32),
        shape: super::StmatrixShape::M8n8x4,
        trans: false,
    };
    assert_eq!(
        inst.emit(),
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r0], {%r1};"
    );
}

#[test]
fn emit_stmatrix_transposed() {
    let inst = Instruction::Stmatrix {
        dst_addr: make_reg_op("%r0", PtxType::U32),
        src: make_reg("%r1", PtxType::B32),
        shape: super::StmatrixShape::M8n8x2,
        trans: true,
    };
    assert_eq!(
        inst.emit(),
        "stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%r0], {%r1};"
    );
}

#[test]
fn emit_elect_sync() {
    let inst = Instruction::ElectSync {
        dst: make_reg("%p0", PtxType::Pred),
        membership_mask: 0xFFFF_FFFF,
    };
    assert_eq!(inst.emit(), "elect.sync %p0, 0xffffffff;");
}

#[test]
fn emit_setmaxnreg_inc() {
    let inst = Instruction::Setmaxnreg {
        reg_count: 128,
        action: super::SetmaxnregAction::Inc,
    };
    assert_eq!(inst.emit(), "setmaxnreg.inc 128;");
}

#[test]
fn emit_setmaxnreg_dec() {
    let inst = Instruction::Setmaxnreg {
        reg_count: 64,
        action: super::SetmaxnregAction::Dec,
    };
    assert_eq!(inst.emit(), "setmaxnreg.dec 64;");
}

#[test]
fn emit_griddepcontrol_launch() {
    let inst = Instruction::Griddepcontrol {
        action: super::GridDepAction::LaunchDependents,
    };
    assert_eq!(inst.emit(), "griddepcontrol.launch_dependents;");
}

#[test]
fn emit_griddepcontrol_wait() {
    let inst = Instruction::Griddepcontrol {
        action: super::GridDepAction::Wait,
    };
    assert_eq!(inst.emit(), "griddepcontrol.wait;");
}

#[test]
fn emit_fence_proxy() {
    let inst = Instruction::FenceProxy {
        scope: FenceScope::Gpu,
        space: MemorySpace::Shared,
    };
    assert_eq!(inst.emit(), "fence.proxy.async.gpu.shared;");
}

#[test]
fn emit_mbarrier_init() {
    let inst = Instruction::MbarrierInit {
        addr: make_reg_op("%r0", PtxType::U64),
        count: make_reg_op("%r1", PtxType::U32),
    };
    assert_eq!(inst.emit(), "mbarrier.init.shared.b64 [%r0], %r1;");
}

#[test]
fn emit_mbarrier_arrive() {
    let inst = Instruction::MbarrierArrive {
        addr: make_reg_op("%r0", PtxType::U64),
    };
    assert_eq!(inst.emit(), "mbarrier.arrive.shared.b64 [%r0];");
}

#[test]
fn emit_mbarrier_wait() {
    let inst = Instruction::MbarrierWait {
        addr: make_reg_op("%r0", PtxType::U64),
        phase: make_reg_op("%r1", PtxType::U32),
    };
    assert_eq!(
        inst.emit(),
        "mbarrier.try_wait.parity.shared.b64 [%r0], %r1;"
    );
}
