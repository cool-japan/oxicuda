//! Tests for the `WarpVec` / `WarpMask` layer and the underlying warp ops.
//!
//! Two tiers:
//!
//! 1. **Text tier** — build kernels and assert the exact lowering (opcodes,
//!    lane-selector `c` encodings, fast-path selection) in the emitted PTX.
//! 2. **Assembler tier** — run `ptxas -arch=sm_86` over a battery of kernels
//!    covering every lowering path, so the syntax is validated by the real
//!    toolchain even before the on-device suites downstream run them.

use super::super::warp_ops::{FULL_WARP_MASK, WARP_SIZE, is_valid_warp_width, shfl_c_value};
use super::{WarpMask, WarpReduceOp, WarpScanMode, WarpVec};
use crate::arch::SmVersion;
use crate::builder::KernelBuilder;
use crate::ir::{PtxType, Register, ShflMode};

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

#[test]
fn warp_width_validation() {
    for w in [2, 4, 8, 16, 32] {
        assert!(is_valid_warp_width(w), "width {w} must be valid");
    }
    for w in [0, 1, 3, 6, 24, 33, 64] {
        assert!(!is_valid_warp_width(w), "width {w} must be invalid");
    }
}

#[test]
fn shfl_c_encoding_matches_cuda_intrinsics() {
    // Full warp: idx/down/bfly clamp to 31, up clamps to 0.
    assert_eq!(shfl_c_value(ShflMode::Bfly, 32), 0x1F);
    assert_eq!(shfl_c_value(ShflMode::Idx, 32), 0x1F);
    assert_eq!(shfl_c_value(ShflMode::Down, 32), 0x1F);
    assert_eq!(shfl_c_value(ShflMode::Up, 32), 0);
    // Segmented: ((32 - width) << 8) | clamp.
    assert_eq!(shfl_c_value(ShflMode::Bfly, 16), 0x101F);
    assert_eq!(shfl_c_value(ShflMode::Up, 16), 0x1000);
    assert_eq!(shfl_c_value(ShflMode::Idx, 8), 0x181F);
    assert_eq!(shfl_c_value(ShflMode::Down, 2), 0x1E1F);
    assert_eq!(FULL_WARP_MASK, 0xFFFF_FFFF);
    assert_eq!(WARP_SIZE, 32);
}

// ---------------------------------------------------------------------------
// Text-tier kernels
// ---------------------------------------------------------------------------

/// Builds a one-warp kernel: loads one f32 per lane from `x` (and `y`), runs
/// `body`, stores the result register per lane to `out`.
fn build_f32_kernel<F>(target: SmVersion, f: F) -> String
where
    F: FnOnce(&mut super::BodyBuilder<'_>, WarpVec, WarpVec) -> Register + 'static,
{
    KernelBuilder::new("warp_vec_test")
        .target(target)
        .param("out", PtxType::U64)
        .param("x", PtxType::U64)
        .param("y", PtxType::U64)
        .body(move |b| {
            let x_ptr = b.load_param_u64("x");
            let y_ptr = b.load_param_u64("y");
            let lane = b.lane_id();
            let x_addr = b.f32_elem_addr(x_ptr, lane.clone());
            let y_addr = b.f32_elem_addr(y_ptr, lane.clone());
            let x_val = b.load_global_f32(x_addr);
            let y_val = b.load_global_f32(y_addr);
            let x = WarpVec::from_register(x_val).expect("x vec");
            let y = WarpVec::from_register(y_val).expect("y vec");
            let result = f(b, x, y);
            let out_ptr = b.load_param_u64("out");
            let out_addr = b.f32_elem_addr(out_ptr, lane);
            b.store_global_f32(out_addr, result);
            b.ret();
        })
        .build()
        .expect("kernel build")
}

#[test]
fn reduce_sum_f32_uses_butterfly_allreduce() {
    let ptx = build_f32_kernel(SmVersion::Sm86, |b, x, _y| {
        x.reduce_sum(b).expect("reduce").into_register()
    });
    // log2(32) = 5 butterfly rounds, all at full-warp clamp 31.
    assert_eq!(
        ptx.matches("shfl.sync.bfly.b32").count(),
        5,
        "expected 5 butterfly rounds:\n{ptx}"
    );
    assert!(ptx.contains(", 31, 0xffffffff;"), "full-warp c operand:\n{ptx}");
    // Floats never take the integer redux path.
    assert!(!ptx.contains("redux.sync"), "f32 must not use redux:\n{ptx}");
    // Butterfly all-reduce needs no trailing broadcast.
    assert!(!ptx.contains("shfl.sync.idx"), "no broadcast expected:\n{ptx}");
}

#[test]
fn reduce_sum_u32_uses_redux_on_sm80_plus() {
    let ptx = KernelBuilder::new("redux_test")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .body(|b| {
            let lanes = WarpVec::lane_ids(b);
            let total = lanes.reduce_sum(b).expect("reduce");
            let out_ptr = b.load_param_u64("out");
            b.store_global_u32(out_ptr, total.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    assert!(
        ptx.contains("redux.sync.add.u32"),
        "u32 sum on sm_86 must lower to redux:\n{ptx}"
    );
    assert!(!ptx.contains("shfl.sync.bfly"), "no butterfly expected:\n{ptx}");
}

#[test]
fn reduce_sum_u32_falls_back_to_butterfly_on_sm75() {
    let ptx = KernelBuilder::new("redux_fallback")
        .target(SmVersion::Sm75)
        .param("out", PtxType::U64)
        .body(|b| {
            let lanes = WarpVec::lane_ids(b);
            let total = lanes.reduce_sum(b).expect("reduce");
            let out_ptr = b.load_param_u64("out");
            b.store_global_u32(out_ptr, total.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    assert!(!ptx.contains("redux.sync"), "sm_75 has no redux:\n{ptx}");
    assert_eq!(ptx.matches("shfl.sync.bfly.b32").count(), 5, "{ptx}");
}

#[test]
fn reduce_prod_never_uses_redux() {
    let ptx = KernelBuilder::new("prod_test")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .body(|b| {
            let lanes = WarpVec::lane_ids(b);
            let prod = lanes.reduce(b, WarpReduceOp::Prod).expect("reduce");
            let out_ptr = b.load_param_u64("out");
            b.store_global_u32(out_ptr, prod.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    assert!(!ptx.contains("redux.sync"), "prod has no redux form:\n{ptx}");
    assert!(ptx.contains("mul.lo.u32"), "integer product combine:\n{ptx}");
}

#[test]
fn reduce_f64_routes_through_pack_unpack() {
    let ptx = KernelBuilder::new("f64_reduce")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .param("x", PtxType::U64)
        .body(|b| {
            let x_ptr = b.load_param_u64("x");
            let lane = b.lane_id();
            let addr = b.f64_elem_addr(x_ptr, lane.clone());
            let val = b.load_global_f64(addr);
            let x = WarpVec::from_register(val).expect("f64 vec");
            let sum = x.reduce_sum(b).expect("reduce");
            let out_ptr = b.load_param_u64("out");
            let out_addr = b.f64_elem_addr(out_ptr, lane);
            b.store_global_f64(out_addr, sum.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    // 5 rounds × (unpack + 2 half-shuffles + pack).
    assert_eq!(
        ptx.matches("shfl.sync.bfly.b32").count(),
        10,
        "two 32-bit shuffles per round:\n{ptx}"
    );
    assert_eq!(ptx.matches("mov.b64 {").count(), 5, "one unpack per round:\n{ptx}");
    assert!(ptx.contains("add.f64"), "f64 combine:\n{ptx}");
}

#[test]
fn segmented_reduce_uses_segment_encoded_c_operand() {
    let ptx = KernelBuilder::new("seg_reduce")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .param("x", PtxType::U64)
        .body(|b| {
            let x_ptr = b.load_param_u64("x");
            let lane = b.lane_id();
            let addr = b.f32_elem_addr(x_ptr, lane.clone());
            let val = b.load_global_f32(addr);
            let x = WarpVec::from_register_segmented(val, 16).expect("seg vec");
            let sum = x.reduce_sum(b).expect("reduce");
            let out_ptr = b.load_param_u64("out");
            let out_addr = b.f32_elem_addr(out_ptr, lane);
            b.store_global_f32(out_addr, sum.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    // Width 16 → 4 rounds, c = ((32-16)<<8)|0x1f = 4127.
    assert_eq!(ptx.matches("shfl.sync.bfly.b32").count(), 4, "{ptx}");
    assert!(
        ptx.contains(", 4127, 0xffffffff;"),
        "segmented c operand 0x101f:\n{ptx}"
    );
}

#[test]
fn scan_sum_inclusive_uses_guarded_up_shuffles() {
    let ptx = KernelBuilder::new("scan_test")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .body(|b| {
            let lanes = WarpVec::lane_ids(b);
            let scanned = lanes.scan_sum(b, WarpScanMode::Inclusive).expect("scan");
            let out_ptr = b.load_param_u64("out");
            b.store_global_u32(out_ptr, scanned.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    // Hillis-Steele: 5 up-shuffle rounds, each writing the in-range predicate
    // (`dst|pred`) and selecting via selp.
    assert_eq!(ptx.matches("shfl.sync.up.b32").count(), 5, "{ptx}");
    assert_eq!(ptx.matches("|%p").count(), 5, "guard predicate per round:\n{ptx}");
    assert_eq!(ptx.matches("selp.u32").count(), 5, "guarded select per round:\n{ptx}");
    // Up-shuffles clamp to 0 at full width.
    assert!(ptx.contains(", 0, 0xffffffff;"), "up-shuffle clamp:\n{ptx}");
}

#[test]
fn scan_sum_exclusive_adds_final_shift() {
    let ptx = KernelBuilder::new("scan_excl")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .body(|b| {
            let lanes = WarpVec::lane_ids(b);
            let scanned = lanes.scan_sum(b, WarpScanMode::Exclusive).expect("scan");
            let out_ptr = b.load_param_u64("out");
            b.store_global_u32(out_ptr, scanned.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    // 5 scan rounds + 1 final shift-down-by-one round.
    assert_eq!(ptx.matches("shfl.sync.up.b32").count(), 6, "{ptx}");
    assert_eq!(ptx.matches("selp.u32").count(), 6, "{ptx}");
}

#[test]
fn mask_votes_and_select_lower_to_vote_and_selp() {
    let ptx = build_f32_kernel(SmVersion::Sm86, |b, x, y| {
        let mask = x.gt(b, &y).expect("gt");
        let any = mask.any(b).expect("any");
        let all = mask.all(b).expect("all");
        let combined = any.and(b, &all).expect("and");
        let flipped = combined.not(b);
        let picked = flipped.select(b, &x, &y).expect("select");
        let count = mask.count(b).expect("count");
        let _ballot = mask.ballot(b);
        let _ = count;
        picked.into_register()
    });
    assert!(ptx.contains("setp.gt.f32"), "float ordered compare:\n{ptx}");
    assert!(ptx.contains("vote.sync.any.pred"), "{ptx}");
    assert!(ptx.contains("vote.sync.all.pred"), "{ptx}");
    assert!(ptx.contains("vote.sync.ballot.b32"), "{ptx}");
    assert!(ptx.contains("and.pred"), "{ptx}");
    assert!(ptx.contains("not.pred"), "{ptx}");
    assert!(ptx.contains("selp.f32"), "{ptx}");
    assert!(ptx.contains("popc.b32"), "count via popc:\n{ptx}");
}

#[test]
fn segmented_mask_any_uses_ballot_and_segment_mask() {
    let ptx = KernelBuilder::new("seg_any")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .param("x", PtxType::U64)
        .body(|b| {
            let x_ptr = b.load_param_u64("x");
            let lane = b.lane_id();
            let addr = b.f32_elem_addr(x_ptr, lane.clone());
            let val = b.load_global_f32(addr);
            let x = WarpVec::from_register_segmented(val, 8).expect("seg vec");
            let zero_reg = b.mov_typed(
                PtxType::F32,
                crate::ir::Operand::Immediate(crate::ir::ImmValue::F32(0.0)),
            );
            let zero = WarpVec::from_register_segmented(zero_reg, 8).expect("zero vec");
            let mask = x.gt(b, &zero).expect("gt");
            let any = mask.any(b).expect("any");
            let one = WarpVec::splat_f32(b, 1.0);
            let one = one.with_width(8).expect("width");
            let none = WarpVec::splat_f32(b, 0.0);
            let none = none.with_width(8).expect("width");
            let flag = any.select(b, &one, &none).expect("select");
            let out_ptr = b.load_param_u64("out");
            let out_addr = b.f32_elem_addr(out_ptr, lane);
            b.store_global_f32(out_addr, flag.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    // Segmented ANY: ballot ∩ per-lane segment mask, compared against zero.
    assert!(ptx.contains("vote.sync.ballot.b32"), "{ptx}");
    assert!(ptx.contains("shl.b32"), "segment mask shift:\n{ptx}");
    assert!(ptx.contains("setp.ne.b32"), "{ptx}");
    // No full-warp vote.any in the segmented path.
    assert!(!ptx.contains("vote.sync.any"), "{ptx}");
}

#[test]
fn shuffles_lower_to_expected_modes() {
    let ptx = build_f32_kernel(SmVersion::Sm86, |b, x, _y| {
        let bc = x.broadcast(b, 7).expect("broadcast");
        let rev = bc.reverse(b).expect("reverse");
        let up = rev.shuffle_up(b, 3).expect("up");
        let down = up.shuffle_down(b, 2).expect("down");
        let lanes = WarpVec::lane_ids(b);
        let dynamic = down.shuffle_idx(b, &lanes).expect("idx");
        dynamic.into_register()
    });
    assert!(ptx.contains("shfl.sync.idx.b32"), "{ptx}");
    // broadcast(7): lane operand 7; reverse: bfly 31.
    assert!(ptx.contains(", 7, 31, 0xffffffff;"), "broadcast lane 7:\n{ptx}");
    assert!(ptx.contains("shfl.sync.bfly.b32"), "{ptx}");
    assert!(ptx.contains(", 31, 31, 0xffffffff;"), "reverse = bfly 31:\n{ptx}");
    assert!(ptx.contains("shfl.sync.up.b32"), "{ptx}");
    assert!(ptx.contains(", 3, 0, 0xffffffff;"), "shuffle_up 3:\n{ptx}");
    assert!(ptx.contains("shfl.sync.down.b32"), "{ptx}");
    assert!(ptx.contains(", 2, 31, 0xffffffff;"), "shuffle_down 2:\n{ptx}");
}

#[test]
fn elementwise_lowering_picks_typed_opcodes() {
    let ptx = build_f32_kernel(SmVersion::Sm86, |b, x, y| {
        let product = x.mul(b, &y).expect("mul");
        let fused = product.fma(b, &x, &y).expect("fma");
        let rect = fused.relu(b).expect("relu");
        let rooted = rect.sqrt(b).expect("sqrt");
        let magnitude = rooted.abs(b).expect("abs");
        let negated = magnitude.neg(b).expect("neg");
        let low = negated.min(b, &x).expect("min");
        let high = low.max(b, &y).expect("max");
        let sum = high.add(b, &x).expect("add");
        let dif = sum.sub(b, &y).expect("sub");
        dif.into_register()
    });
    for needle in [
        "mul.rn.f32",
        "fma.rn.f32",
        "max.f32",
        "sqrt.rn.f32",
        "abs.f32",
        "neg.f32",
        "min.f32",
        "add.f32",
        "sub.f32",
        // relu materializes +0.0 as a hex float literal.
        "0f00000000",
    ] {
        assert!(ptx.contains(needle), "missing `{needle}`:\n{ptx}");
    }
}

#[test]
fn integer_mul_uses_lo_form() {
    let ptx = KernelBuilder::new("int_mul")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .body(|b| {
            let lanes = WarpVec::lane_ids(b);
            let sq = lanes.mul(b, &lanes).expect("mul");
            let out_ptr = b.load_param_u64("out");
            b.store_global_u32(out_ptr, sq.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    assert!(ptx.contains("mul.lo.u32"), "{ptx}");
    assert!(!ptx.contains("mul.rn"), "{ptx}");
}

#[test]
fn unsigned_comparison_picks_unsigned_operators() {
    let ptx = KernelBuilder::new("cmp_test")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .body(|b| {
            let lanes = WarpVec::lane_ids(b);
            let threshold = WarpVec::splat_u32(b, 16);
            let m1 = lanes.gt(b, &threshold).expect("gt");
            let m2 = lanes.le(b, &threshold).expect("le");
            let both = m1.or(b, &m2).expect("or");
            let one = WarpVec::splat_u32(b, 1);
            let zero = WarpVec::splat_u32(b, 0);
            let flag = both.select(b, &one, &zero).expect("select");
            let out_ptr = b.load_param_u64("out");
            b.store_global_u32(out_ptr, flag.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    assert!(ptx.contains("setp.hi.u32"), "unsigned gt → hi:\n{ptx}");
    assert!(ptx.contains("setp.ls.u32"), "unsigned le → ls:\n{ptx}");
    assert!(ptx.contains("or.pred"), "{ptx}");
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

#[test]
fn constructors_reject_bad_inputs() {
    let pred = Register {
        name: "%p0".into(),
        ty: PtxType::Pred,
    };
    assert!(WarpVec::from_register(pred.clone()).is_err(), "pred elements");
    let f32_reg = Register {
        name: "%f0".into(),
        ty: PtxType::F32,
    };
    assert!(
        WarpVec::from_register_segmented(f32_reg.clone(), 3).is_err(),
        "width 3"
    );
    assert!(
        WarpVec::from_register_segmented(f32_reg.clone(), 64).is_err(),
        "width 64"
    );
    assert!(WarpMask::from_predicate(f32_reg).is_err(), "non-pred mask");
    assert!(WarpMask::from_predicate_segmented(pred, 5).is_err(), "width 5");
}

#[test]
fn operations_reject_type_and_width_mismatches() {
    KernelBuilder::new("err_test")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .body(|b| {
            let f = WarpVec::splat_f32(b, 1.0);
            let u = WarpVec::splat_u32(b, 1);
            assert!(f.add(b, &u).is_err(), "type mismatch");

            let f16w = f.with_width(16).expect("width");
            assert!(f.add(b, &f16w).is_err(), "width mismatch");

            assert!(u.fma(b, &u, &u).is_err(), "fma on integers");
            assert!(u.sqrt(b).is_err(), "sqrt on integers");
            assert!(u.neg(b).is_err(), "neg on unsigned");
            assert!(u.relu(b).is_err(), "relu on unsigned");
            assert!(f.reduce(b, WarpReduceOp::BitAnd).is_err(), "bitand on float");
            assert!(f.broadcast(b, 32).is_err(), "lane out of range");
            assert!(f.butterfly(b, 32).is_err(), "xor mask out of range");
            assert!(f.shuffle_idx(b, &f).is_err(), "float index vector");

            let sm = f.gt(b, &f).expect("mask").not(b);
            let sm16 = WarpMask::from_predicate_segmented(sm.predicate().clone(), 16)
                .expect("segmented mask");
            assert!(sm16.select(b, &f, &f).is_err(), "mask/value width mismatch");
            b.ret();
        })
        .build()
        .expect("kernel build");
}

// ---------------------------------------------------------------------------
// Assembler tier: ptxas -arch=sm_86 over every lowering path
// ---------------------------------------------------------------------------

fn find_ptxas() -> Option<std::path::PathBuf> {
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            for name in ["ptxas", "ptxas.exe"] {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    let fallback = std::path::PathBuf::from("/usr/local/cuda/bin/ptxas");
    if fallback.is_file() {
        return Some(fallback);
    }
    None
}

/// Assembles `ptx` with `ptxas -arch=sm_86`, panicking with the full
/// diagnostics and PTX text on rejection.
fn assert_assembles(ptxas: &std::path::Path, label: &str, ptx: &str) {
    let mut ptx_path = std::env::temp_dir();
    ptx_path.push(format!("oxicuda_warpvec_{label}_{}.ptx", std::process::id()));
    std::fs::write(&ptx_path, ptx).expect("write PTX to temp file");
    let cubin = ptx_path.with_extension("cubin");
    let output = std::process::Command::new(ptxas)
        .arg("-arch=sm_86")
        .arg(&ptx_path)
        .arg("-o")
        .arg(&cubin)
        .output()
        .expect("invoke ptxas");
    let _ = std::fs::remove_file(&ptx_path);
    let _ = std::fs::remove_file(&cubin);
    assert!(
        output.status.success(),
        "ptxas rejected `{label}`:\n{}{}\n--- PTX ---\n{ptx}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Every lowering path through the layer, validated by the real assembler.
#[test]
#[allow(clippy::too_many_lines)]
fn warp_vec_kernels_assemble_for_sm86() {
    let Some(ptxas) = find_ptxas() else {
        println!("skipping: ptxas not found on PATH");
        return;
    };

    // relu_dot: the module-doc example shape (mul → relu → reduce_sum).
    let relu_dot = build_f32_kernel(SmVersion::Sm86, |b, x, y| {
        x.mul(b, &y)
            .expect("mul")
            .relu(b)
            .expect("relu")
            .reduce_sum(b)
            .expect("reduce")
            .into_register()
    });
    assert_assembles(&ptxas, "relu_dot", &relu_dot);

    // Reductions: redux fast path, butterfly, f64 composite, segmented,
    // prod / min / max / bitwise.
    let reductions = KernelBuilder::new("reductions")
        .target(SmVersion::Sm86)
        .param("out32", PtxType::U64)
        .param("out64", PtxType::U64)
        .param("x", PtxType::U64)
        .body(|b| {
            let lane = b.lane_id();
            let lanes = WarpVec::lane_ids(b);
            let redux_sum = lanes.reduce_sum(b).expect("redux sum");
            let redux_and = lanes.reduce(b, WarpReduceOp::BitAnd).expect("redux and");
            let prod = lanes.reduce_prod(b).expect("prod");
            let partial = redux_sum
                .add(b, &redux_and)
                .expect("add")
                .add(b, &prod)
                .expect("add");

            let x_ptr = b.load_param_u64("x");
            let f64_addr = b.f64_elem_addr(x_ptr, lane.clone());
            let f64_val = b.load_global_f64(f64_addr);
            let dvec = WarpVec::from_register(f64_val).expect("f64 vec");
            let dmin = dvec.reduce_min(b).expect("min");
            let dmax = dvec.reduce_max(b).expect("max");
            let dsum = dmin.add(b, &dmax).expect("add");

            let seg = lanes.with_width(4).expect("seg");
            let seg_max = seg.reduce_max(b).expect("seg max");
            let seg_max_full = seg_max.with_width(32).expect("re-widen");
            let total = partial.add(b, &seg_max_full).expect("add");

            let out32 = b.load_param_u64("out32");
            let out32_addr = b.f32_elem_addr(out32, lane.clone());
            b.store_global_u32(out32_addr, total.into_register());
            let out64 = b.load_param_u64("out64");
            let out64_addr = b.f64_elem_addr(out64, lane);
            b.store_global_f64(out64_addr, dsum.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    assert_assembles(&ptxas, "reductions", &reductions);

    // Scans: inclusive u32, exclusive f32, segmented inclusive.
    let scans = KernelBuilder::new("scans")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .body(|b| {
            let lane = b.lane_id();
            let lanes = WarpVec::lane_ids(b);
            let incl = lanes.scan_sum(b, WarpScanMode::Inclusive).expect("incl");
            let seg = lanes.with_width(8).expect("seg");
            let seg_incl = seg.scan_sum(b, WarpScanMode::Inclusive).expect("seg incl");
            let sum = incl.add(b, &seg_incl.with_width(32).expect("w")).expect("add");

            let fvec = WarpVec::splat_f32(b, 1.5);
            let fexcl = fvec.scan_sum(b, WarpScanMode::Exclusive).expect("excl");
            let _ = fexcl;

            let out_ptr = b.load_param_u64("out");
            let out_addr = b.f32_elem_addr(out_ptr, lane);
            b.store_global_u32(out_addr, sum.into_register());
            b.ret();
        })
        .build()
        .expect("kernel build");
    assert_assembles(&ptxas, "scans", &scans);

    // Masks: votes, ballot, count, segmented any/all, select, logic.
    let masks = build_f32_kernel(SmVersion::Sm86, |b, x, y| {
        let gt = x.gt(b, &y).expect("gt");
        let ballot_any = gt.any(b).expect("any");
        let ballot_all = gt.all(b).expect("all");
        let logic = ballot_any
            .and(b, &ballot_all)
            .expect("and")
            .or(b, &gt)
            .expect("or")
            .xor(b, &gt)
            .expect("xor")
            .not(b);
        let count = gt.count(b).expect("count");
        let _ = count;

        let x8 = x.with_width(8).expect("w8");
        let y8 = y.with_width(8).expect("w8");
        let seg_mask = x8.lt(b, &y8).expect("lt");
        let seg_any = seg_mask.any(b).expect("seg any");
        let seg_all = seg_mask.all(b).expect("seg all");
        let seg_pick = seg_any.select(b, &x8, &y8).expect("select");
        let seg_pick2 = seg_all.select(b, &seg_pick, &y8).expect("select");

        let final_pick = logic.select(b, &x, &seg_pick2.with_width(32).expect("w")).expect("select");
        final_pick.into_register()
    });
    assert_assembles(&ptxas, "masks", &masks);

    // Shuffles: broadcast / reverse / up / down / dynamic idx / u64 shuffle.
    let shuffles = KernelBuilder::new("shuffles")
        .target(SmVersion::Sm86)
        .param("out", PtxType::U64)
        .param("x", PtxType::U64)
        .body(|b| {
            let lane = b.lane_id();
            let x_ptr = b.load_param_u64("x");
            let addr = b.f64_elem_addr(x_ptr, lane.clone());
            let dval = b.load_global_f64(addr);
            let dvec = WarpVec::from_register(dval).expect("f64 vec");
            let drot = dvec.shuffle_down(b, 1).expect("f64 shuffle");

            let lanes = WarpVec::lane_ids(b);
            let bc = lanes.broadcast(b, 7).expect("broadcast");
            let rev = lanes.reverse(b).expect("reverse");
            let up = lanes.shuffle_up(b, 3).expect("up");
            let dynamic = lanes.shuffle_idx(b, &rev).expect("idx");
            let sum = bc
                .add(b, &rev)
                .expect("add")
                .add(b, &up)
                .expect("add")
                .add(b, &dynamic)
                .expect("add");

            let out_ptr = b.load_param_u64("out");
            let out_addr = b.f64_elem_addr(out_ptr.clone(), lane);
            b.store_global_f64(out_addr, drot.into_register());
            let out_addr2 = b.f32_elem_addr(out_ptr, sum.register().clone());
            let zero = b.mov_typed(
                PtxType::U32,
                crate::ir::Operand::Immediate(crate::ir::ImmValue::U32(0)),
            );
            b.store_global_u32(out_addr2, zero);
            b.ret();
        })
        .build()
        .expect("kernel build");
    assert_assembles(&ptxas, "shuffles", &shuffles);
}
