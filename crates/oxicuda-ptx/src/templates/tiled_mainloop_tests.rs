//! Host-side tests for the CTA-tiled mainloop emitter.
//!
//! Split out of `tiled_mainloop.rs` to keep both files well inside the
//! workspace's 2000-line budget.
//!
//! Three things are checked here, none of which needs a GPU:
//!
//! 1. **Configuration validation** — every constraint the emitted code relies
//!    on is rejected at build time rather than miscompiled.
//! 2. **The index mapping's bank behaviour** — the column split
//!    ([`TiledAccumulators::col_offset`]) is asserted to be a bijection *and*
//!    to give a conflict-free bank pattern for the 8-thread phases a
//!    `ld.shared.v4` is serviced in. This is the property the whole mapping
//!    exists for, and it is checkable as pure arithmetic.
//! 3. **The emitted instruction stream** — exact instruction counts (FMAs,
//!    vector loads, barriers) and, when `ptxas` is installed, that a real
//!    assembler accepts the module and does not spill.

use super::*;
use crate::arch::SmVersion;
use crate::builder::KernelBuilder;
use crate::ir::Register as PtxRegister;

// ---------------------------------------------------------------------------
// A minimal row-major GEMM driver, used as the emitter's test harness
// ---------------------------------------------------------------------------

/// Builds `C[m][n] = sum_k A[m][k] * B[k][n]` for row-major operands with the
/// tiled mainloop, and returns the PTX text.
///
/// Deliberately a *plain* GEMM rather than a convolution: it is the second
/// consumer shape the emitter's callback interface claims to serve, so the
/// fact that it can be expressed in a few lines here is itself part of the
/// test.
fn build_row_major_gemm(cfg: TiledMainloopConfig) -> String {
    KernelBuilder::new("tiled_gemm_probe")
        .target(SmVersion::Sm86)
        .max_threads_per_block(TILED_MAINLOOP_THREADS)
        .param("a", PtxType::U64)
        .param("b", PtxType::U64)
        .param("c", PtxType::U64)
        .param("m_dim", PtxType::U32)
        .param("n_dim", PtxType::U32)
        .param("k_dim", PtxType::U32)
        .shared_mem_aligned("smem_a", PtxType::F32, cfg.smem_a_elems() as usize, 16)
        .shared_mem_aligned("smem_b", PtxType::F32, cfg.smem_b_elems() as usize, 16)
        .body(move |bb| {
            let a_ptr = bb.load_param_u64("a");
            let b_ptr = bb.load_param_u64("b");
            let c_ptr = bb.load_param_u64("c");
            let m_dim = bb.load_param_u32("m_dim");
            let n_dim = bb.load_param_u32("n_dim");
            let k_dim = bb.load_param_u32("k_dim");

            let block_row = bb.alloc_reg(PtxType::U32);
            let block_col = bb.alloc_reg(PtxType::U32);
            bb.raw_ptx(&format!("mov.u32 {block_row}, %ctaid.y;"));
            bb.raw_ptx(&format!("mov.u32 {block_col}, %ctaid.x;"));
            let row0 = bb.alloc_reg(PtxType::U32);
            let col0 = bb.alloc_reg(PtxType::U32);
            bb.raw_ptx(&format!(
                "mul.lo.u32 {row0}, {block_row}, {};",
                cfg.tile_m()
            ));
            bb.raw_ptx(&format!(
                "mul.lo.u32 {col0}, {block_col}, {};",
                cfg.tile_n()
            ));

            let n_ktiles = bb.alloc_reg(PtxType::U32);
            bb.raw_ptx(&format!("add.u32 {n_ktiles}, {k_dim}, {};", cfg.tile_k - 1));
            bb.raw_ptx(&format!("div.u32 {n_ktiles}, {n_ktiles}, {};", cfg.tile_k));

            let a_step = bb.alloc_reg(PtxType::U64);
            bb.raw_ptx(&format!("mov.u64 {a_step}, {};", cfg.tile_k * 4));
            let b_step = bb.alloc_reg(PtxType::U64);
            let b_step32 = bb.alloc_reg(PtxType::U32);
            bb.raw_ptx(&format!(
                "mul.lo.u32 {b_step32}, {n_dim}, {};",
                cfg.tile_k * 4
            ));
            bb.raw_ptx(&format!("cvt.u64.u32 {b_step}, {b_step32};"));

            let acc = emit_probe_mainloop(
                bb,
                cfg,
                &ProbeOperands {
                    a_ptr,
                    b_ptr,
                    a_step,
                    b_step,
                    row0: row0.clone(),
                    col0: col0.clone(),
                    m_dim: m_dim.clone(),
                    n_dim: n_dim.clone(),
                    k_dim,
                },
                &n_ktiles,
            );

            emit_probe_epilogue(
                bb,
                &acc,
                &ProbeEpilogue {
                    c_ptr,
                    row0,
                    col0,
                    m_dim,
                    n_dim,
                },
            );
            bb.ret();
        })
        .build()
        .expect("probe kernel build")
}

/// The probe's operand registers, as the mainloop callbacks need them.
struct ProbeOperands {
    a_ptr: PtxRegister,
    b_ptr: PtxRegister,
    a_step: PtxRegister,
    b_step: PtxRegister,
    row0: PtxRegister,
    col0: PtxRegister,
    m_dim: PtxRegister,
    n_dim: PtxRegister,
    k_dim: PtxRegister,
}

/// Drives the emitter with plain row-major operand addressing.
fn emit_probe_mainloop(
    bb: &mut BodyBuilder<'_>,
    cfg: TiledMainloopConfig,
    ops: &ProbeOperands,
    n_ktiles: &PtxRegister,
) -> TiledAccumulators {
    emit_tiled_mainloop(
        bb,
        cfg,
        "smem_a",
        "smem_b",
        n_ktiles,
        |bb, slots| {
            slots
                .iter()
                .map(|slot| {
                    // A[row0 + outer][inner]
                    let row = bb.add_u32(ops.row0.clone(), slot.outer.clone());
                    let ok = bb.alloc_reg(PtxType::Pred);
                    bb.raw_ptx(&format!("setp.lo.u32 {ok}, {row}, {};", ops.m_dim));
                    let idx = bb.mad_lo_u32(row, ops.k_dim.clone(), slot.inner.clone());
                    let ptr = bb.byte_offset_addr(ops.a_ptr.clone(), idx, 4);
                    GlobalTap {
                        ptr,
                        advance_bytes: ops.a_step.clone(),
                        valid: Some(ok),
                    }
                })
                .collect()
        },
        |bb, slots| {
            slots
                .iter()
                .map(|slot| {
                    // B[inner][col0 + outer]
                    let col = bb.add_u32(ops.col0.clone(), slot.outer.clone());
                    let ok = bb.alloc_reg(PtxType::Pred);
                    bb.raw_ptx(&format!("setp.lo.u32 {ok}, {col}, {};", ops.n_dim));
                    let idx = bb.mad_lo_u32(slot.inner.clone(), ops.n_dim.clone(), col);
                    let ptr = bb.byte_offset_addr(ops.b_ptr.clone(), idx, 4);
                    GlobalTap {
                        ptr,
                        advance_bytes: ops.b_step.clone(),
                        valid: Some(ok),
                    }
                })
                .collect()
        },
    )
    .expect("mainloop emission")
}

/// What the probe's epilogue needs from the preamble.
struct ProbeEpilogue {
    c_ptr: PtxRegister,
    row0: PtxRegister,
    col0: PtxRegister,
    m_dim: PtxRegister,
    n_dim: PtxRegister,
}

/// Scalar, fully predicated epilogue: correctness only, this probe is never
/// launched for throughput.
fn emit_probe_epilogue(bb: &mut BodyBuilder<'_>, acc: &TiledAccumulators, ctx: &ProbeEpilogue) {
    let cfg = acc.config();
    let my_row = bb.add_u32(ctx.row0.clone(), acc.row_base().clone());
    let my_col = bb.add_u32(ctx.col0.clone(), acc.col_base().clone());
    for m in 0..cfg.thread_m {
        for n in 0..cfg.thread_n {
            let r_off = bb.mov_imm_u32(TiledAccumulators::row_offset(m));
            let c_off = bb.mov_imm_u32(TiledAccumulators::col_offset(n));
            let row = bb.add_u32(my_row.clone(), r_off);
            let col = bb.add_u32(my_col.clone(), c_off);
            let pr = bb.alloc_reg(PtxType::Pred);
            let pc = bb.alloc_reg(PtxType::Pred);
            let pok = bb.alloc_reg(PtxType::Pred);
            bb.raw_ptx(&format!("setp.lo.u32 {pr}, {row}, {};", ctx.m_dim));
            bb.raw_ptx(&format!("setp.lo.u32 {pc}, {col}, {};", ctx.n_dim));
            bb.raw_ptx(&format!("and.pred {pok}, {pr}, {pc};"));
            let idx = bb.mad_lo_u32(row, ctx.n_dim.clone(), col);
            let addr = bb.byte_offset_addr(ctx.c_ptr.clone(), idx, 4);
            bb.raw_ptx(&format!(
                "@{pok} st.global.f32 [{addr}], {};",
                acc.acc(m, n)
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[test]
fn canonical_config_geometry() {
    let cfg = TiledMainloopConfig::canonical();
    assert_eq!(cfg.tile_m(), 128);
    assert_eq!(cfg.tile_n(), 128);
    assert_eq!(cfg.tile_k, 8);
    assert_eq!(cfg.accumulators(), 64);
    assert_eq!(cfg.smem_a_pitch(), 132);
    assert_eq!(cfg.smem_a_elems(), 132 * 8);
    assert_eq!(cfg.smem_b_elems(), 128 * 8);
    // 128*8 A elements + 128*8 B elements + padding, all f32.
    assert_eq!(cfg.smem_bytes(), 4 * (132 * 8 + 128 * 8));
    // 1024 elements per tile over 256 threads.
    assert_eq!(cfg.a_stage_iters(), 4);
    assert_eq!(cfg.b_stage_iters(), 4);
    cfg.validate().expect("canonical config must validate");
}

/// `tile_k = R*S` for a 3x3 convolution: 9 does not divide 256, so the last
/// staging pass is partial and the emitter must guard it.
#[test]
fn conv_3x3_config_geometry() {
    let cfg = TiledMainloopConfig {
        thread_m: 8,
        thread_n: 8,
        tile_k: 9,
    };
    cfg.validate().expect("3x3 config must validate");
    assert_eq!(cfg.a_stage_iters(), 5, "128*9 = 1152 elements over 256");
    assert_eq!(cfg.b_stage_iters(), 5);
    assert!(cfg.smem_bytes() < MAX_STATIC_SMEM_BYTES);
}

#[test]
fn validate_rejects_non_multiple_of_four_thread_n() {
    for thread_n in [1u32, 2, 3, 5, 6, 7] {
        let cfg = TiledMainloopConfig {
            thread_m: 8,
            thread_n,
            tile_k: 8,
        };
        assert!(
            cfg.validate().is_err(),
            "thread_n={thread_n} must be rejected: column fragments are v4 loads"
        );
    }
}

#[test]
fn validate_rejects_oversized_register_tile() {
    for (thread_m, thread_n) in [(9u32, 8u32), (16, 8), (8, 12), (8, 16)] {
        let cfg = TiledMainloopConfig {
            thread_m,
            thread_n,
            tile_k: 8,
        };
        assert!(
            cfg.validate().is_err(),
            "{thread_m}x{thread_n} register tile must be rejected"
        );
    }
}

#[test]
fn validate_rejects_zero_tile_k() {
    let cfg = TiledMainloopConfig {
        thread_m: 8,
        thread_n: 8,
        tile_k: 0,
    };
    assert!(cfg.validate().is_err());
}

/// A 7x7 convolution's `tile_k = 49` overflows the static shared-memory budget
/// at a 128-row tile; the caller is expected to shrink `thread_m`, and
/// `validate` is what tells it to.
#[test]
fn validate_rejects_over_budget_shared_memory() {
    let big = TiledMainloopConfig {
        thread_m: 8,
        thread_n: 8,
        tile_k: 49,
    };
    assert!(
        big.smem_bytes() > MAX_STATIC_SMEM_BYTES,
        "this case is only meaningful if it really is over budget"
    );
    assert!(big.validate().is_err());

    let shrunk = TiledMainloopConfig {
        thread_m: 4,
        thread_n: 8,
        tile_k: 49,
    };
    assert!(shrunk.smem_bytes() <= MAX_STATIC_SMEM_BYTES);
    shrunk.validate().expect("half-height tile fits");
}

// ---------------------------------------------------------------------------
// Index mapping
// ---------------------------------------------------------------------------

/// The `(tx, n) -> column` map must cover `0..tile_n` exactly once: a
/// duplicate would make two threads accumulate the same output, and a gap
/// would leave an output element unwritten.
#[test]
fn column_mapping_is_a_bijection_onto_the_tile() {
    for thread_n in [4u32, 8] {
        let cfg = TiledMainloopConfig {
            thread_m: 8,
            thread_n,
            tile_k: 8,
        };
        let mut seen = vec![0u32; cfg.tile_n() as usize];
        for tx in 0..THREADS_X {
            for n in 0..thread_n {
                let col = tx * 4 + TiledAccumulators::col_offset(n);
                seen[col as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "thread_n={thread_n}: column coverage must be exactly 1 everywhere, got {seen:?}"
        );
    }
}

/// The row map must likewise tile `0..tile_m` exactly once.
#[test]
fn row_mapping_is_a_bijection_onto_the_tile() {
    for thread_m in [2u32, 4, 8] {
        let cfg = TiledMainloopConfig {
            thread_m,
            thread_n: 8,
            tile_k: 8,
        };
        let mut seen = vec![0u32; cfg.tile_m() as usize];
        for ty in 0..THREADS_Y {
            for m in 0..thread_m {
                let row = ty * thread_m + TiledAccumulators::row_offset(m);
                seen[row as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "thread_m={thread_m}");
    }
}

/// The reason the column split exists: `ld.shared.v4` is serviced in phases of
/// 8 threads, and within a phase the 8 x 16 B accesses must hit all 32 banks
/// exactly once. A contiguous `tx * thread_n` mapping (asserted here to be
/// worse) hits only half the banks, doubling every column load.
#[test]
fn column_split_is_bank_conflict_free_per_phase() {
    let cfg = TiledMainloopConfig::canonical();
    let banks_touched = |first_col: u32| -> Vec<u32> {
        // One `ld.shared.v4.f32` reads 4 consecutive floats == 4 banks.
        (0..4).map(|j| (first_col + j) % 32).collect()
    };

    // Split mapping, group 0: threads tx = 0..7 in one phase.
    let mut hits = [0u32; 32];
    for tx in 0..8u32 {
        for bank in banks_touched(tx * 4 + TiledAccumulators::col_offset(0)) {
            hits[bank as usize] += 1;
        }
    }
    assert!(
        hits.iter().all(|&h| h == 1),
        "split mapping must touch each bank exactly once per phase, got {hits:?}"
    );

    // The rejected alternative: contiguous per-thread columns.
    let mut naive = [0u32; 32];
    for tx in 0..8u32 {
        for bank in banks_touched(tx * cfg.thread_n) {
            naive[bank as usize] += 1;
        }
    }
    assert!(
        naive.iter().any(|&h| h > 1),
        "the contiguous mapping is supposed to be the conflicting one"
    );
}

/// Every `ld.shared.v4.f32` the mainloop issues must be 16-byte aligned. Both
/// fragment bases and every immediate offset are checked as arithmetic here,
/// which is stronger than reading it out of the PTX text.
#[test]
fn every_vector_shared_access_is_sixteen_byte_aligned() {
    for cfg in [
        TiledMainloopConfig::canonical(),
        TiledMainloopConfig {
            thread_m: 8,
            thread_n: 8,
            tile_k: 9,
        },
        TiledMainloopConfig {
            thread_m: 4,
            thread_n: 8,
            tile_k: 25,
        },
    ] {
        for k in 0..cfg.tile_k {
            for ty in 0..THREADS_Y {
                for g in 0..(cfg.thread_m / 4) {
                    let byte = (ty * cfg.thread_m + k * cfg.smem_a_pitch()) * 4 + g * 16;
                    assert_eq!(byte % 16, 0, "A fragment misaligned: {cfg:?} k={k} ty={ty}");
                }
            }
            for tx in 0..THREADS_X {
                for g in 0..(cfg.thread_n / 4) {
                    let byte = tx * 16 + (k * cfg.tile_n()) * 4 + g * COL_GROUP_STRIDE * 4;
                    assert_eq!(byte % 16, 0, "B fragment misaligned: {cfg:?} k={k} tx={tx}");
                }
            }
        }
    }
}

/// Staging must cover every element of both tiles exactly once across the
/// 256 threads and their `*_stage_iters` passes — with the out-of-tile slots
/// of a partial pass guarded off.
#[test]
fn staging_covers_each_tile_element_exactly_once() {
    for cfg in [
        TiledMainloopConfig::canonical(),
        TiledMainloopConfig {
            thread_m: 8,
            thread_n: 8,
            tile_k: 9,
        },
        TiledMainloopConfig {
            thread_m: 2,
            thread_n: 4,
            tile_k: 25,
        },
    ] {
        let a_total = cfg.tile_m() * cfg.tile_k;
        let mut a_seen = vec![0u32; a_total as usize];
        for tid in 0..TILED_MAINLOOP_THREADS {
            for i in 0..cfg.a_stage_iters() {
                let idx = tid + i * TILED_MAINLOOP_THREADS;
                if idx < a_total {
                    a_seen[idx as usize] += 1;
                }
            }
        }
        assert!(a_seen.iter().all(|&c| c == 1), "A staging gap: {cfg:?}");

        let b_total = cfg.tile_k * cfg.tile_n();
        let mut b_seen = vec![0u32; b_total as usize];
        for tid in 0..TILED_MAINLOOP_THREADS {
            for i in 0..cfg.b_stage_iters() {
                let idx = tid + i * TILED_MAINLOOP_THREADS;
                if idx < b_total {
                    b_seen[idx as usize] += 1;
                }
            }
        }
        assert!(b_seen.iter().all(|&c| c == 1), "B staging gap: {cfg:?}");
    }
}

// ---------------------------------------------------------------------------
// Emitted instruction stream
// ---------------------------------------------------------------------------

#[test]
fn emitted_mainloop_has_the_expected_instruction_mix() {
    let cfg = TiledMainloopConfig::canonical();
    let ptx = build_row_major_gemm(cfg);

    assert_eq!(
        ptx.matches("fma.rn.f32").count(),
        (cfg.tile_k * cfg.accumulators()) as usize,
        "one FMA per (k, register-tile element), fully unrolled"
    );

    // A: thread_m/4 vector loads per k; B: thread_n/4 per k.
    assert_eq!(
        ptx.matches("ld.shared.v4.f32").count(),
        (cfg.tile_k * (cfg.thread_m / 4 + cfg.thread_n / 4)) as usize
    );

    // Exactly two barriers per k-step: one after staging, one after compute.
    assert_eq!(ptx.matches("bar.sync").count(), 2);

    // One shared store per slot per k-step...
    let slots = (cfg.a_stage_iters() + cfg.b_stage_iters()) as usize;
    assert_eq!(ptx.matches("st.shared.f32").count(), slots);
    // ...but *two* global-load sites per slot: the prologue's first slice and
    // the in-loop prefetch of the next one. That doubling is the register
    // prefetch pipeline, not redundancy -- see `prefetch_precedes_the_compute`.
    assert_eq!(ptx.matches("ld.global.f32").count(), 2 * slots);

    assert!(ptx.contains(".shared .align 16 .b8 smem_a"));
    assert!(ptx.contains(".shared .align 16 .b8 smem_b"));
    assert!(ptx.contains(".maxntid 256, 1, 1"));
}

/// The pipeline's defining property: inside the k-loop, the next slice's
/// global loads are issued **before** the register-tile update, so the memory
/// latency overlaps the compute instead of stalling in front of it.
///
/// Asserted on instruction order in the emitted text, because that is the only
/// thing that makes the difference: the same instructions in the other order
/// are a correct but latency-exposed kernel.
#[test]
fn prefetch_precedes_the_compute() {
    let ptx = build_row_major_gemm(TiledMainloopConfig::canonical());
    let first_barrier = ptx.find("bar.sync").expect("staging barrier");
    let after_barrier = &ptx[first_barrier..];

    let prefetch = after_barrier
        .find("ld.global.f32")
        .expect("in-loop prefetch must exist");
    let compute = after_barrier
        .find("fma.rn.f32")
        .expect("register-tile update must exist");
    assert!(
        prefetch < compute,
        "the next slice's loads must be issued before the FMAs that hide them"
    );

    // ...and the prefetch must be skipped on the final iteration, or it would
    // read one K slice past the end of both operands.
    let guard = after_barrier[..prefetch]
        .lines()
        .rev()
        .take(6)
        .any(|l| l.contains("bra") && l.contains('@'));
    assert!(
        guard,
        "the in-loop prefetch must be guarded by a last-iteration branch"
    );
}

/// A partial staging pass must be predicated; a full one must not be. Both
/// polarities are checked so a future change that guards everything (a silent
/// throughput loss) or nothing (a silent out-of-tile write) is caught.
#[test]
fn partial_staging_pass_is_guarded_and_full_passes_are_not() {
    let even = build_row_major_gemm(TiledMainloopConfig::canonical());
    assert!(
        !even.contains("st.shared.f32") || !even.contains("@%p") || {
            // Canonical tiles divide evenly: no guarded shared store at all.
            !even
                .lines()
                .any(|l| l.trim_start().starts_with('@') && l.contains("st.shared.f32"))
        },
        "an evenly-divided tile must not emit a guarded shared store"
    );

    let odd = build_row_major_gemm(TiledMainloopConfig {
        thread_m: 8,
        thread_n: 8,
        tile_k: 9,
    });
    assert!(
        odd.lines()
            .any(|l| l.trim_start().starts_with('@') && l.contains("st.shared.f32")),
        "a partial staging pass must guard its shared store"
    );
}

/// The k-loop must contain no address arithmetic for the fragment reads: every
/// `ld.shared` inside it addresses a loop-invariant base with an immediate
/// offset.
#[test]
fn fragment_reads_use_immediate_offsets() {
    let ptx = build_row_major_gemm(TiledMainloopConfig::canonical());
    let offsets = ptx
        .lines()
        .filter(|l| l.contains("ld.shared.v4.f32"))
        .filter(|l| l.contains('+'))
        .count();
    // Every fragment load except the two at offset zero carries an immediate.
    assert!(
        offsets >= (TiledMainloopConfig::canonical().tile_k * 4 - 2) as usize,
        "fragment loads must use immediate offsets, saw {offsets}"
    );
}

#[test]
fn ptxas_accepts_the_emitted_module_without_spilling() {
    use std::io::Write;
    use std::process::Command;

    for cfg in [
        TiledMainloopConfig::canonical(),
        TiledMainloopConfig {
            thread_m: 8,
            thread_n: 8,
            tile_k: 9,
        },
        TiledMainloopConfig {
            thread_m: 4,
            thread_n: 8,
            tile_k: 9,
        },
    ] {
        let ptx = build_row_major_gemm(cfg);
        let dir = std::env::temp_dir().join(format!(
            "oxicuda_tiled_mainloop_{}_{}_{}",
            cfg.thread_m, cfg.thread_n, cfg.tile_k
        ));
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let src = dir.join("probe.ptx");
        let Ok(mut f) = std::fs::File::create(&src) else {
            return;
        };
        if f.write_all(ptx.as_bytes()).is_err() {
            return;
        }
        drop(f);
        let out = dir.join("probe.cubin");
        let Ok(result) = Command::new("ptxas")
            .arg("-arch=sm_86")
            .arg("-v")
            .arg(&src)
            .arg("-o")
            .arg(&out)
            .output()
        else {
            // No CUDA toolkit on this host: the structural assertions above
            // still ran.
            return;
        };
        let stderr = String::from_utf8_lossy(&result.stderr).to_string();
        assert!(
            result.status.success(),
            "ptxas rejected the {cfg:?} mainloop:\n{stderr}"
        );
        assert!(
            stderr.contains("0 bytes spill stores"),
            "the {cfg:?} mainloop must not spill:\n{stderr}"
        );
        // scratch: keep dir
    }
}
