//! FlashAttention-3 kernel body emission for Hopper+ GPUs.
//!
//! This module contains the actual PTX instruction emission for the forward and
//! backward passes. It is split from `hopper.rs` to keep each file under 2000
//! lines (refactoring policy).
//!
//! ## Forward Pass Algorithm (FlashAttention-2 online softmax)
//!
//! For each Q-block tile:
//! 1. Load Q tile to shared memory (producers via cp.async).
//! 2. Initialise O_acc = 0, m_i = -inf, l_i = 0.
//! 3. Loop over KV tiles (ping-pong pipeline):
//!    a. Load K/V tiles (producers).
//!    b. Compute S = scale * Q @ K^T via m16n8k16 MMA.
//!    c. Online softmax: update m_i, compute P = exp(S - m_i), update l_i.
//!    d. Rescale O_acc by exp(m_old - m_new).
//!    e. Accumulate O_acc += P @ V via m16n8k16 MMA.
//! 4. Normalise O_out = O_acc / l_i, store logsumexp = m_i + ln(l_i).
//!
//! ## MMA Fragment Layout (m16n8k16, Ampere+)
//!
//! Each warp operates on a 16x8x16 tile:
//! - A fragment (Q/P): 8 f16 values packed into 4 b32 registers.
//! - B fragment (K^T/V^T): 4 f16 values packed into 2 b32 registers.
//! - D/C accumulator: 4 f32 registers (2 rows x 2 cols per thread).

use oxicuda_ptx::ir::{FenceScope, PtxType, Register};
use oxicuda_ptx::prelude::*;

// ---------------------------------------------------------------------------
// Public re-exports used by hopper.rs
// ---------------------------------------------------------------------------

/// Returns a short string suffix for the given PTX floating-point type.
pub(super) fn float_type_suffix(ty: PtxType) -> &'static str {
    match ty {
        PtxType::F16 => "f16",
        PtxType::BF16 => "bf16",
        PtxType::F32 => "f32",
        PtxType::F64 => "f64",
        _ => "unk",
    }
}

// ---------------------------------------------------------------------------
// Low-level PTX helpers
// ---------------------------------------------------------------------------

/// Move an f32 immediate into a fresh register.
/// Emits `mov.f32 dst, 0F<hex>;`
fn mov_imm_f32(b: &mut BodyBuilder<'_>, val: f32) -> Register {
    let dst = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("mov.f32 {dst}, 0F{:08X};", val.to_bits()));
    dst
}

/// Butterfly shuffle of an f32 value across a warp lane offset.
fn shfl_bfly_f32(b: &mut BodyBuilder<'_>, src: &Register, offset: u32) -> Register {
    let dst = b.alloc_reg(PtxType::F32);
    let pred = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!(
        "shfl.sync.bfly.b32 {dst}|{pred}, {src}, {offset}, 0x1f, 0xFFFFFFFF;"
    ));
    dst
}

/// Warp-level reduction for the maximum of an f32 value.
fn warp_reduce_max_f32(b: &mut BodyBuilder<'_>, val: Register) -> Register {
    let mut acc = val;
    for offset in [16u32, 8, 4, 2, 1] {
        let shuffled = shfl_bfly_f32(b, &acc, offset);
        acc = b.max_f32(acc, shuffled);
    }
    acc
}

/// Warp-level reduction for the sum of an f32 value.
fn warp_reduce_sum_f32(b: &mut BodyBuilder<'_>, val: Register) -> Register {
    let mut acc = val;
    for offset in [16u32, 8, 4, 2, 1] {
        let shuffled = shfl_bfly_f32(b, &acc, offset);
        acc = b.add_f32(acc, shuffled);
    }
    acc
}

/// Compute exp(x) ≈ ex2(x * log2(e)) using hardware fast ex2.
fn fast_exp_f32(b: &mut BodyBuilder<'_>, x: Register) -> Register {
    // Pre-compute all arguments before calling fma to avoid double-borrow.
    let log2e = mov_imm_f32(b, std::f32::consts::LOG2_E);
    let zero = mov_imm_f32(b, 0.0f32);
    let scaled = b.fma_f32(x, log2e, zero);
    b.ex2_approx_f32(scaled)
}

/// Load an m16n8k16 A-fragment from a u32 shared memory address.
/// Returns 4 B32 registers holding the fragment.
fn ldmatrix_a_x4(b: &mut BodyBuilder<'_>, smem_addr_u32: &Register) -> [Register; 4] {
    let r0 = b.alloc_reg(PtxType::B32);
    let r1 = b.alloc_reg(PtxType::B32);
    let r2 = b.alloc_reg(PtxType::B32);
    let r3 = b.alloc_reg(PtxType::B32);
    b.raw_ptx(&format!(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{r0}, {r1}, {r2}, {r3}}}, [{smem_addr_u32}];"
    ));
    [r0, r1, r2, r3]
}

/// Load an m16n8k16 B-fragment (transposed) from a u32 shared memory address.
/// Returns 2 B32 registers holding the fragment.
fn ldmatrix_b_x2_trans(b: &mut BodyBuilder<'_>, smem_addr_u32: &Register) -> [Register; 2] {
    let r0 = b.alloc_reg(PtxType::B32);
    let r1 = b.alloc_reg(PtxType::B32);
    b.raw_ptx(&format!(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {{{r0}, {r1}}}, [{smem_addr_u32}];"
    ));
    [r0, r1]
}

/// Tag a B32 register as F16 for passing to `mma_m16n8k16_f16_f32`.
///
/// `ldmatrix` loads packed f16x2 values into B32 registers. The PTX MMA
/// instruction declares its A/B operands as f16, but `ptxas` interprets the
/// bits of any 32-bit register file entry as a pair of f16 values regardless
/// of the declared register type. Retagging here is semantically correct
/// because the underlying bit pattern is already two f16 halves.
fn b32_as_f16(r: Register) -> Register {
    Register {
        name: r.name,
        ty: PtxType::F16,
    }
}

/// Emit one m16n8k16 MMA instruction.
fn emit_mma_tile(
    b: &mut BodyBuilder<'_>,
    a_regs: [Register; 4],
    b_regs: [Register; 2],
    c_regs: [Register; 4],
) -> [Register; 4] {
    let a_f16: Vec<Register> = a_regs.into_iter().map(b32_as_f16).collect();
    let b_f16: Vec<Register> = b_regs.into_iter().map(b32_as_f16).collect();
    b.mma_m16n8k16_f16_f32(&a_f16, &b_f16, &c_regs)
}

/// Allocate and zero-initialise n f32 accumulator registers.
fn alloc_f32_accum(b: &mut BodyBuilder<'_>, n: usize) -> Vec<Register> {
    (0..n)
        .map(|_| {
            let r = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("mov.f32 {r}, 0F00000000;"));
            r
        })
        .collect()
}

/// Compute smem base address (u32) for a named shared memory variable.
fn smem_base_u32(b: &mut BodyBuilder<'_>, name: &str) -> Register {
    let r = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {r}, %{name};"));
    r
}

/// Compute a u32 element offset (index * 2 bytes for f16) added to a u32 base.
fn smem_add_offset_u32(b: &mut BodyBuilder<'_>, base: Register, byte_offset: u32) -> Register {
    let off = b.mov_imm_u32(byte_offset);
    b.add_u32(base, off)
}

/// Scale an f32 register by scale and assign back. Returns the scaled register.
fn scale_f32(b: &mut BodyBuilder<'_>, val: Register, scale: Register) -> Register {
    let zero = mov_imm_f32(b, 0.0f32);
    b.fma_f32(val, scale, zero)
}

// ---------------------------------------------------------------------------
// Forward pass body
// ---------------------------------------------------------------------------

/// Emits the FlashAttention-3 forward pass kernel body with real PTX instructions.
///
/// Emits:
/// - `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` for QK^T and P@V.
/// - `ldmatrix.sync.aligned.m8n8` for loading MMA fragments.
/// - `shfl.sync.bfly.b32` for warp-level softmax reductions.
/// - `ex2.approx.f32` for exp() approximation.
/// - `cp.async` and `bar.sync` for the ping-pong pipeline.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_fa3_forward_body(
    b: &mut BodyBuilder<'_>,
    block_m: u32,
    block_n: u32,
    head_dim: u32,
    num_producer_warps: u32,
    pingpong_stages: u32,
    causal: bool,
    float_type: PtxType,
) {
    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    let tid = b.thread_id_x();
    let bid_x = b.block_id_x();

    let _seq_q = b.load_param_u32("seq_len_q");
    let _seq_kv = b.load_param_u32("seq_len_kv");
    let _hdim = b.load_param_u32("head_dim");
    let _nheads = b.load_param_u32("num_heads");
    let scale = b.load_param_f32("sm_scale");
    let nkv_tiles = b.load_param_u32("num_kv_tiles");

    let q_base = b.load_param_u64("q_ptr");
    let k_base = b.load_param_u64("k_ptr");
    let v_base = b.load_param_u64("v_ptr");
    let o_base = b.load_param_u64("o_ptr");
    let lse_base = b.load_param_u64("lse_ptr");

    b.comment("=== FlashAttention-3 Forward Pass (Hopper, real PTX) ===");
    b.comment(&format!(
        "Config: block_m={block_m} block_n={block_n} head_dim={head_dim} float_type={}",
        float_type_suffix(float_type)
    ));
    b.comment(&format!(
        "Warp spec: {num_producer_warps} producer warps, Ping-pong stages: {pingpong_stages}, causal={causal}"
    ));

    let elem_bytes = if matches!(float_type, PtxType::F16 | PtxType::BF16) {
        2u32
    } else {
        4u32
    };

    // -----------------------------------------------------------------------
    // Warp role
    // -----------------------------------------------------------------------
    b.comment("--- warp_id = tid >> 5 ---");
    let five = b.mov_imm_u32(5);
    let warp_id = b.shr_u32(tid.clone(), five);
    let producer_thresh = b.mov_imm_u32(num_producer_warps);

    // -----------------------------------------------------------------------
    // Load Q tile (producers)
    // -----------------------------------------------------------------------
    b.comment("--- Load Q tile via cp.async ---");
    let q_tile_bytes = block_m * head_dim * elem_bytes;
    let qt_stride_reg = b.mov_imm_u32(q_tile_bytes);
    let qt_off = b.mul_wide_u32_to_u64(bid_x.clone(), qt_stride_reg);
    let q_global = b.add_u64(q_base, qt_off);
    let q_smem_u32 = smem_base_u32(b, "q_smem");
    let q_smem_u64 = b.cvt_u32_to_u64(q_smem_u32);
    b.cp_async_128bit(q_smem_u64, q_global);
    b.cp_async_commit();
    b.cp_async_wait(0);
    b.bar_sync(0);

    // -----------------------------------------------------------------------
    // Accumulator init (logically consumer-only; all threads allocate)
    // -----------------------------------------------------------------------
    b.comment("--- Init O_acc, m_i, l_i ---");

    let m_tiles = (block_m / 16) as usize;
    let n_tiles_pv = (head_dim / 8) as usize;
    let n_tiles_qk = (block_n / 8) as usize;

    let mut o_acc = alloc_f32_accum(b, m_tiles * n_tiles_pv * 4);

    // stat_len = m_tiles * 2 (2 logical rows per m16 tile per warp thread)
    let stat_len = m_tiles * 2;
    let mut m_i: Vec<Register> = (0..stat_len)
        .map(|_| {
            let r = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("mov.f32 {r}, 0FFF800000;")); // -inf
            r
        })
        .collect();
    let mut l_i: Vec<Register> = (0..stat_len)
        .map(|_| {
            let r = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("mov.f32 {r}, 0F00000000;")); // 0.0
            r
        })
        .collect();

    // -----------------------------------------------------------------------
    // KV-tile loop
    // -----------------------------------------------------------------------
    b.comment("--- KV-tile loop ---");
    let kv_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {kv_idx}, 0;"));
    let loop_lbl = b.fresh_label("kv_loop");
    let loop_end_lbl = b.fresh_label("kv_loop_end");
    b.label(&loop_lbl);

    // Loop guard
    let loop_pred = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ge.u32 {loop_pred}, {kv_idx}, {nkv_tiles};"));
    b.raw_ptx(&format!("@{loop_pred} bra {loop_end_lbl};"));

    // stage = kv_idx % pingpong_stages
    let stage_mod = b.mov_imm_u32(pingpong_stages);
    let stage = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {stage}, {kv_idx}, {stage_mod};"));

    // -------------------------------------------------------------------
    // Producer: load K and V tiles
    // -------------------------------------------------------------------
    b.comment("  Producer: cp.async K and V tiles");
    b.if_lt_u32(warp_id.clone(), producer_thresh.clone(), |b| {
        let kv_bytes = block_n * head_dim * elem_bytes;
        let kv_stride = b.mov_imm_u32(kv_bytes);
        let kv_off = b.mul_wide_u32_to_u64(kv_idx.clone(), kv_stride);

        let k_global = b.add_u64(k_base.clone(), kv_off.clone());
        let v_global = b.add_u64(v_base.clone(), kv_off);

        // Stage offset in smem
        let stage_bytes = block_n * head_dim * elem_bytes;
        let sb_reg = b.mov_imm_u32(stage_bytes);
        let stage_off = b.mul_wide_u32_to_u64(stage.clone(), sb_reg);

        let k_smem_u32 = smem_base_u32(b, "k_smem");
        let k_smem_u64 = b.cvt_u32_to_u64(k_smem_u32);
        let k_smem_addr = b.add_u64(k_smem_u64, stage_off.clone());
        b.cp_async_128bit(k_smem_addr, k_global);

        let v_smem_u32 = smem_base_u32(b, "v_smem");
        let v_smem_u64 = b.cvt_u32_to_u64(v_smem_u32);
        let v_smem_addr = b.add_u64(v_smem_u64, stage_off);
        b.cp_async_128bit(v_smem_addr, v_global);

        b.cp_async_commit();
    });

    b.bar_sync(1); // producer signals consumer
    b.cp_async_wait(0);

    // -------------------------------------------------------------------
    // Consumer: S = Q @ K^T
    // -------------------------------------------------------------------
    b.comment("  Consumer: S = Q_smem @ K_smem^T via mma.sync.aligned.m16n8k16");
    b.if_ge_u32(warp_id.clone(), producer_thresh.clone(), |b| {
        let k_steps = (head_dim / 16) as usize;
        let mut s_acc = alloc_f32_accum(b, m_tiles * n_tiles_qk * 4);

        for m_t in 0..m_tiles {
            for n_t in 0..n_tiles_qk {
                let s_base = (m_t * n_tiles_qk + n_t) * 4;
                let c_init = [
                    s_acc[s_base].clone(),
                    s_acc[s_base + 1].clone(),
                    s_acc[s_base + 2].clone(),
                    s_acc[s_base + 3].clone(),
                ];
                let mut running_c = c_init;

                for k_t in 0..k_steps {
                    // Q fragment address (u32 smem)
                    let q_byte_off = (m_t as u32 * 16 * head_dim + k_t as u32 * 16) * 2;
                    let q_base_u32 = smem_base_u32(b, "q_smem");
                    let q_addr_u32 = smem_add_offset_u32(b, q_base_u32, q_byte_off);

                    // K fragment address (u32 smem + stage offset)
                    let k_byte_off = (n_t as u32 * head_dim + k_t as u32 * 16) * 2;
                    let k_stage_bytes = block_n * head_dim * elem_bytes;
                    let k_stage_reg = b.mov_imm_u32(k_stage_bytes);
                    let k_stage_off32 = b.mul_lo_u32(stage.clone(), k_stage_reg);
                    let k_base_u32 = smem_base_u32(b, "k_smem");
                    let k_with_stage = b.add_u32(k_base_u32, k_stage_off32);
                    let k_addr_u32 = smem_add_offset_u32(b, k_with_stage, k_byte_off);

                    let a_regs = ldmatrix_a_x4(b, &q_addr_u32);
                    let br_regs = ldmatrix_b_x2_trans(b, &k_addr_u32);
                    running_c = emit_mma_tile(b, a_regs, br_regs, running_c);
                }

                // Scale S by sm_scale
                for i in 0..4 {
                    s_acc[s_base + i] = scale_f32(b, running_c[i].clone(), scale.clone());
                }
            }
        }

        // -------------------------------------------------------------------
        // Causal mask
        // -------------------------------------------------------------------
        if causal {
            b.comment("  [CAUSAL] Apply causal mask to S");
            for m_t in 0..m_tiles {
                for n_t in 0..n_tiles_qk {
                    let s_base = (m_t * n_tiles_qk + n_t) * 4;
                    let n_col = b.mov_imm_u32(n_t as u32 * 8);
                    let m_row = b.mov_imm_u32(m_t as u32 * 16);
                    let cmp_pred = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.gt.u32 {cmp_pred}, {n_col}, {m_row};"));
                    let neginf = mov_imm_f32(b, f32::NEG_INFINITY);
                    for k in 0..4usize {
                        let masked = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!(
                            "selp.f32 {masked}, {neginf}, {}, {cmp_pred};",
                            s_acc[s_base + k]
                        ));
                        s_acc[s_base + k] = masked;
                    }
                }
            }
        }

        // -------------------------------------------------------------------
        // Online softmax: for each logical row, update m_i, l_i, rescale O_acc
        // -------------------------------------------------------------------
        b.comment("  Online softmax update");
        for m_t in 0..m_tiles {
            for row in 0..2usize {
                let stat = m_t * 2 + row;

                // Row max across n_tiles_qk
                let mut row_max = mov_imm_f32(b, f32::NEG_INFINITY);
                for n_t in 0..n_tiles_qk {
                    let sb = (m_t * n_tiles_qk + n_t) * 4;
                    let s0 = s_acc[sb + row * 2].clone();
                    let s1 = s_acc[sb + row * 2 + 1].clone();
                    row_max = b.max_f32(row_max, s0);
                    row_max = b.max_f32(row_max, s1);
                }
                let warp_max = warp_reduce_max_f32(b, row_max);
                let m_new = b.max_f32(m_i[stat].clone(), warp_max);

                // correction = exp(m_old - m_new)
                let m_diff = b.sub_f32(m_i[stat].clone(), m_new.clone());
                let corr = fast_exp_f32(b, m_diff);

                // Rescale O_acc row
                for n_t in 0..n_tiles_pv {
                    let oi = (m_t * n_tiles_pv + n_t) * 4;
                    for k in 0..4usize {
                        if k / 2 == row {
                            o_acc[oi + k] = scale_f32(b, o_acc[oi + k].clone(), corr.clone());
                        }
                    }
                }

                // P = exp(S - m_new); p_sum = rowsum(P)
                let mut p_sum = mov_imm_f32(b, 0.0f32);
                for n_t in 0..n_tiles_qk {
                    let sb = (m_t * n_tiles_qk + n_t) * 4;
                    for k in 0..4usize {
                        if k / 2 == row {
                            let shifted = b.sub_f32(s_acc[sb + k].clone(), m_new.clone());
                            let p_val = fast_exp_f32(b, shifted);
                            p_sum = b.add_f32(p_sum, p_val.clone());
                            s_acc[sb + k] = p_val;
                        }
                    }
                }
                let p_sum_w = warp_reduce_sum_f32(b, p_sum);
                let l_corr = scale_f32(b, l_i[stat].clone(), corr.clone());
                let l_new = b.add_f32(l_corr, p_sum_w);

                m_i[stat] = m_new;
                l_i[stat] = l_new;
            }
        }

        // -------------------------------------------------------------------
        // O_acc += P @ V
        // -------------------------------------------------------------------
        b.comment("  Accumulate O_acc += P @ V via mma.sync.aligned.m16n8k16");
        let k_steps_pv = (block_n / 16) as usize;

        for m_t in 0..m_tiles {
            for n_t in 0..n_tiles_pv {
                let oi = (m_t * n_tiles_pv + n_t) * 4;
                let c_init = [
                    o_acc[oi].clone(),
                    o_acc[oi + 1].clone(),
                    o_acc[oi + 2].clone(),
                    o_acc[oi + 3].clone(),
                ];
                let mut running_o = c_init;

                for k_t in 0..k_steps_pv {
                    // P fragment from softmax_smem scratch
                    let p_byte_off = m_t as u32 * 16 * block_n * 2 + k_t as u32 * 16 * 2;
                    let p_base_u32 = smem_base_u32(b, "softmax_smem");
                    let p_addr_u32 = smem_add_offset_u32(b, p_base_u32, p_byte_off);

                    // V fragment from v_smem (with stage offset)
                    let v_byte_off = n_t as u32 * head_dim * 2 + k_t as u32 * 16 * 2;
                    let v_stage_b = block_n * head_dim * elem_bytes;
                    let v_stage_reg = b.mov_imm_u32(v_stage_b);
                    let v_stage_off = b.mul_lo_u32(stage.clone(), v_stage_reg);
                    let v_base_u32 = smem_base_u32(b, "v_smem");
                    let v_with_stage = b.add_u32(v_base_u32, v_stage_off);
                    let v_addr_u32 = smem_add_offset_u32(b, v_with_stage, v_byte_off);

                    let a_regs = ldmatrix_a_x4(b, &p_addr_u32);
                    let br_regs = ldmatrix_b_x2_trans(b, &v_addr_u32);
                    running_o = emit_mma_tile(b, a_regs, br_regs, running_o);
                }

                o_acc[oi..(oi + 4)].clone_from_slice(&running_o);
            }
        }

        b.bar_sync(2); // consumer done
    });

    // Advance loop counter
    let one = b.mov_imm_u32(1);
    b.raw_ptx(&format!("add.u32 {kv_idx}, {kv_idx}, {one};"));
    b.branch(&loop_lbl);
    b.label(&loop_end_lbl);

    // -----------------------------------------------------------------------
    // Final normalise O = O_acc / l_i and store
    // -----------------------------------------------------------------------
    b.comment("--- Final normalise and store ---");
    b.if_ge_u32(warp_id.clone(), producer_thresh, |b| {
        // Write logsumexp and normalise O
        for m_t in 0..m_tiles {
            for row in 0..2usize {
                let stat = m_t * 2 + row;
                let l_inv = b.rcp_approx_f32(l_i[stat].clone());

                // Normalise O row
                for n_t in 0..n_tiles_pv {
                    let oi = (m_t * n_tiles_pv + n_t) * 4;
                    for k in 0..4usize {
                        if k / 2 == row {
                            o_acc[oi + k] = scale_f32(b, o_acc[oi + k].clone(), l_inv.clone());
                        }
                    }
                }

                // logsumexp = m_i + ln(l_i) = m_i + lg2(l_i) * ln(2)
                let lg2_l = b.lg2_approx_f32(l_i[stat].clone());
                let ln2 = mov_imm_f32(b, std::f32::consts::LN_2);
                let zero_lse = mov_imm_f32(b, 0.0f32);
                let log_l = b.fma_f32(lg2_l, ln2, zero_lse);
                let lse_val = b.add_f32(m_i[stat].clone(), log_l);

                // Store lse: lse_ptr + (bid_x * block_m + m_t * 16 + row * 8) * 4
                let lse_row_off = m_t as u32 * 16 + row as u32 * 8;
                let lse_row_reg = b.mov_imm_u32(lse_row_off);
                let bm_reg = b.mov_imm_u32(block_m);
                let lse_row = b.mad_lo_u32(bid_x.clone(), bm_reg, lse_row_reg);
                let lse_addr = b.f32_elem_addr(lse_base.clone(), lse_row);
                b.store_global_f32(lse_addr, lse_val);
            }
        }

        // Store O to global memory
        for m_t in 0..m_tiles {
            for n_t in 0..n_tiles_pv {
                let oi = (m_t * n_tiles_pv + n_t) * 4;
                for k in 0..4usize {
                    let row_off = m_t as u32 * 16 + (k / 2) as u32 * 8;
                    let col_off = n_t as u32 * 8 + (k % 2) as u32 * 4;
                    let flat_reg = b.mov_imm_u32(row_off * head_dim + col_off);
                    let bm_hd_reg = b.mov_imm_u32(block_m * head_dim);
                    let o_row = b.mad_lo_u32(bid_x.clone(), bm_hd_reg, flat_reg);
                    let o_addr = b.f32_elem_addr(o_base.clone(), o_row);
                    b.store_global_f32(o_addr, o_acc[oi + k].clone());
                }
            }
        }
    });

    b.fence_acq_rel(FenceScope::Cta);
    b.bar_sync(0);

    let _ = (tid, warp_id, nkv_tiles, kv_idx, stage);
    b.ret();
}

// ---------------------------------------------------------------------------
// Backward pass body
// ---------------------------------------------------------------------------

/// Emits the FlashAttention-3 backward pass kernel body with real PTX instructions.
///
/// Each block (indexed by `bid_x`) handles one KV tile and iterates over Q tiles:
/// 1. Producers load K_j, V_j once; then Q_i, dO_i per Q tile.
/// 2. Consumers recompute S_ij, P_ij; accumulate dV, dK; atomically update dQ.
/// 3. Store dK_j, dV_j.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_fa3_backward_body(
    b: &mut BodyBuilder<'_>,
    block_m: u32,
    block_n: u32,
    head_dim: u32,
    num_producer_warps: u32,
    pingpong_stages: u32,
    causal: bool,
    float_type: PtxType,
) {
    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    let tid = b.thread_id_x();
    let bid_x = b.block_id_x();

    let _seq_q = b.load_param_u32("seq_len_q");
    let _seq_kv = b.load_param_u32("seq_len_kv");
    let _hdim = b.load_param_u32("head_dim");
    let _nheads = b.load_param_u32("num_heads");
    let scale = b.load_param_f32("sm_scale");
    let nq_tiles = b.load_param_u32("num_q_tiles");

    let q_base = b.load_param_u64("q_ptr");
    let k_base = b.load_param_u64("k_ptr");
    let v_base = b.load_param_u64("v_ptr");
    let _o_base = b.load_param_u64("o_ptr");
    let do_base = b.load_param_u64("do_ptr");
    let lse_base = b.load_param_u64("lse_ptr");
    let di_base = b.load_param_u64("di_ptr");
    let dq_base = b.load_param_u64("dq_ptr");
    let dk_base = b.load_param_u64("dk_ptr");
    let dv_base = b.load_param_u64("dv_ptr");

    b.comment("=== FlashAttention-3 Backward Pass (Hopper, real PTX) ===");
    b.comment(&format!(
        "Config: block_m={block_m} block_n={block_n} head_dim={head_dim} float_type={} causal={causal}",
        float_type_suffix(float_type)
    ));
    b.comment("Launched with grid (num_kv_tiles, B*H, 1). Each block handles one KV tile.");

    let elem_bytes = if matches!(float_type, PtxType::F16 | PtxType::BF16) {
        2u32
    } else {
        4u32
    };

    // -----------------------------------------------------------------------
    // Warp role
    // -----------------------------------------------------------------------
    let five = b.mov_imm_u32(5);
    let warp_id = b.shr_u32(tid.clone(), five);
    let producer_thresh = b.mov_imm_u32(num_producer_warps);

    // -----------------------------------------------------------------------
    // Load K_j and V_j (once per block)
    // -----------------------------------------------------------------------
    b.comment("--- Load K_j and V_j tiles (producers) ---");
    b.if_lt_u32(warp_id.clone(), producer_thresh.clone(), |b| {
        let kv_bytes = block_n * head_dim * elem_bytes;
        let kv_stride = b.mov_imm_u32(kv_bytes);
        let kv_off = b.mul_wide_u32_to_u64(bid_x.clone(), kv_stride);

        let k_global = b.add_u64(k_base.clone(), kv_off.clone());
        let v_global = b.add_u64(v_base.clone(), kv_off);

        let k_smem_u32 = smem_base_u32(b, "k_smem");
        let k_smem_u64 = b.cvt_u32_to_u64(k_smem_u32);
        b.cp_async_128bit(k_smem_u64, k_global);

        let v_smem_u32 = smem_base_u32(b, "v_smem");
        let v_smem_u64 = b.cvt_u32_to_u64(v_smem_u32);
        b.cp_async_128bit(v_smem_u64, v_global);

        b.cp_async_commit();
    });
    b.cp_async_wait(0);
    b.bar_sync(0);

    // -----------------------------------------------------------------------
    // dK and dV accumulators
    // -----------------------------------------------------------------------
    b.comment("--- Init dK_j, dV_j accumulators ---");
    let m_tiles_bm = (block_m / 16) as usize;
    let n_tiles_dkv = (block_n / 8) as usize; // N-dimension for dK/dV tiles
    let n_tiles_hd = (head_dim / 8) as usize;

    let dk_len = n_tiles_dkv * n_tiles_hd * 4;
    let dv_len = n_tiles_dkv * n_tiles_hd * 4;
    let mut dk_acc = alloc_f32_accum(b, dk_len);
    let mut dv_acc = alloc_f32_accum(b, dv_len);

    // -----------------------------------------------------------------------
    // Q-tile loop
    // -----------------------------------------------------------------------
    b.comment("--- Q-tile loop ---");
    let q_tile_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {q_tile_idx}, 0;"));
    let q_loop_lbl = b.fresh_label("q_loop");
    let q_end_lbl = b.fresh_label("q_loop_end");

    b.label(&q_loop_lbl);
    let q_loop_pred = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!(
        "setp.ge.u32 {q_loop_pred}, {q_tile_idx}, {nq_tiles};"
    ));
    b.raw_ptx(&format!("@{q_loop_pred} bra {q_end_lbl};"));

    let stage_mod = b.mov_imm_u32(pingpong_stages);
    let stage = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {stage}, {q_tile_idx}, {stage_mod};"));

    // Producer: load Q_i and dO_i
    b.comment("  Producer: load Q_i and dO_i tiles");
    b.if_lt_u32(warp_id.clone(), producer_thresh.clone(), |b| {
        let q_bytes = block_m * head_dim * elem_bytes;
        let q_stride = b.mov_imm_u32(q_bytes);
        let q_off = b.mul_wide_u32_to_u64(q_tile_idx.clone(), q_stride);

        let q_global = b.add_u64(q_base.clone(), q_off.clone());
        let do_global = b.add_u64(do_base.clone(), q_off);

        let sb_bytes = block_m * head_dim * elem_bytes;
        let sb_reg = b.mov_imm_u32(sb_bytes);
        let sb_off = b.mul_wide_u32_to_u64(stage.clone(), sb_reg);

        let q_smem_u32 = smem_base_u32(b, "q_smem");
        let q_smem_u64 = b.cvt_u32_to_u64(q_smem_u32);
        let q_smem_dst = b.add_u64(q_smem_u64, sb_off.clone());
        b.cp_async_128bit(q_smem_dst, q_global);

        let do_smem_u32 = smem_base_u32(b, "do_smem");
        let do_smem_u64 = b.cvt_u32_to_u64(do_smem_u32);
        let do_smem_dst = b.add_u64(do_smem_u64, sb_off);
        b.cp_async_128bit(do_smem_dst, do_global);

        b.cp_async_commit();
    });
    b.bar_sync(1);
    b.cp_async_wait(0);

    // Consumer: recompute S_ij and accumulate gradients
    b.comment("  Consumer: recompute S_ij, P_ij; accumulate dV, dK, dQ");
    b.if_ge_u32(warp_id.clone(), producer_thresh.clone(), |b| {
        let n_tiles_s = (block_n / 8) as usize;
        let k_steps = (head_dim / 16) as usize;

        // ----------------------------------------------------------------
        // Recompute S_ij = sm_scale * Q_i @ K_j^T
        // ----------------------------------------------------------------
        b.comment("  Recompute S_ij via mma.sync.aligned.m16n8k16");
        let mut s_acc = alloc_f32_accum(b, m_tiles_bm * n_tiles_s * 4);

        for m_t in 0..m_tiles_bm {
            for n_t in 0..n_tiles_s {
                let sb = (m_t * n_tiles_s + n_t) * 4;
                let c_init = [
                    s_acc[sb].clone(),
                    s_acc[sb + 1].clone(),
                    s_acc[sb + 2].clone(),
                    s_acc[sb + 3].clone(),
                ];
                let mut running_c = c_init;

                for k_t in 0..k_steps {
                    // Q from q_smem (stage-buffered)
                    let q_byte_off = m_t as u32 * 16 * head_dim * 2 + k_t as u32 * 16 * 2;
                    let q_stage_b = block_m * head_dim * elem_bytes;
                    let q_stage_reg = b.mov_imm_u32(q_stage_b);
                    let q_stage_off = b.mul_lo_u32(stage.clone(), q_stage_reg);
                    let q_base_u32 = smem_base_u32(b, "q_smem");
                    let q_with_st = b.add_u32(q_base_u32, q_stage_off);
                    let q_addr_u32 = smem_add_offset_u32(b, q_with_st, q_byte_off);

                    // K from k_smem (no stage for K_j — loaded once)
                    let k_byte_off = n_t as u32 * head_dim * 2 + k_t as u32 * 16 * 2;
                    let k_base_u32 = smem_base_u32(b, "k_smem");
                    let k_addr_u32 = smem_add_offset_u32(b, k_base_u32, k_byte_off);

                    let a_regs = ldmatrix_a_x4(b, &q_addr_u32);
                    let br_regs = ldmatrix_b_x2_trans(b, &k_addr_u32);
                    running_c = emit_mma_tile(b, a_regs, br_regs, running_c);
                }

                for i in 0..4 {
                    s_acc[sb + i] = scale_f32(b, running_c[i].clone(), scale.clone());
                }
            }
        }

        // Causal mask
        if causal {
            b.comment("  [CAUSAL] S_ij mask");
            for m_t in 0..m_tiles_bm {
                for n_t in 0..n_tiles_s {
                    let sb = (m_t * n_tiles_s + n_t) * 4;
                    let n_col = b.mov_imm_u32(n_t as u32 * 8);
                    let m_row = b.mov_imm_u32(m_t as u32 * 16);
                    let cmp_pred = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.gt.u32 {cmp_pred}, {n_col}, {m_row};"));
                    let neginf = mov_imm_f32(b, f32::NEG_INFINITY);
                    for k in 0..4usize {
                        let masked = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!(
                            "selp.f32 {masked}, {neginf}, {}, {cmp_pred};",
                            s_acc[sb + k]
                        ));
                        s_acc[sb + k] = masked;
                    }
                }
            }
        }

        // ----------------------------------------------------------------
        // Recompute P_ij = exp(S_ij - lse_i)
        // ----------------------------------------------------------------
        b.comment("  Recompute P_ij = exp(S_ij - lse_i)");
        for m_t in 0..m_tiles_bm {
            for row in 0..2usize {
                let lse_off_val = m_t as u32 * 16 + row as u32 * 8;
                let lse_off_reg = b.mov_imm_u32(lse_off_val);
                let bm_reg = b.mov_imm_u32(block_m);
                let lse_row = b.mad_lo_u32(q_tile_idx.clone(), bm_reg, lse_off_reg);
                let lse_addr = b.f32_elem_addr(lse_base.clone(), lse_row);
                let lse_val = b.load_global_f32(lse_addr);

                for n_t in 0..n_tiles_s {
                    let sb = (m_t * n_tiles_s + n_t) * 4;
                    for k in 0..4usize {
                        if k / 2 == row {
                            let shifted = b.sub_f32(s_acc[sb + k].clone(), lse_val.clone());
                            s_acc[sb + k] = fast_exp_f32(b, shifted);
                        }
                    }
                }
            }
        }

        // ----------------------------------------------------------------
        // dV_j += P_ij^T @ dO_i  (MMA: [block_n x head_dim])
        // ----------------------------------------------------------------
        b.comment("  dV_j += P^T @ dO_i via mma.sync.aligned.m16n8k16");
        let k_steps_p = m_tiles_bm; // block_m / 16

        for n_t in 0..n_tiles_dkv {
            for hd_t in 0..n_tiles_hd {
                let dv_idx = (n_t * n_tiles_hd + hd_t) * 4;
                let c_init = [
                    dv_acc[dv_idx].clone(),
                    dv_acc[dv_idx + 1].clone(),
                    dv_acc[dv_idx + 2].clone(),
                    dv_acc[dv_idx + 3].clone(),
                ];
                let mut running_dv = c_init;

                for k_t in 0..k_steps_p {
                    // P^T: load from softmax_smem scratch
                    let p_byte_off = n_t as u32 * block_m * 2 + k_t as u32 * 16 * 2;
                    let p_base_u32 = smem_base_u32(b, "softmax_smem");
                    let p_addr_u32 = smem_add_offset_u32(b, p_base_u32, p_byte_off);

                    // dO: from do_smem (stage-buffered)
                    let do_byte_off = k_t as u32 * 16 * head_dim * 2 + hd_t as u32 * 8 * 2;
                    let do_stage_b = block_m * head_dim * elem_bytes;
                    let do_stage_reg = b.mov_imm_u32(do_stage_b);
                    let do_stage_off = b.mul_lo_u32(stage.clone(), do_stage_reg);
                    let do_base_u32 = smem_base_u32(b, "do_smem");
                    let do_with_st = b.add_u32(do_base_u32, do_stage_off);
                    let do_addr_u32 = smem_add_offset_u32(b, do_with_st, do_byte_off);

                    let a_regs = ldmatrix_a_x4(b, &p_addr_u32);
                    let br_regs = ldmatrix_b_x2_trans(b, &do_addr_u32);
                    running_dv = emit_mma_tile(b, a_regs, br_regs, running_dv);
                }

                dv_acc[dv_idx..(dv_idx + 4)].clone_from_slice(&running_dv);
            }
        }

        // ----------------------------------------------------------------
        // dP_ij = dO_i @ V_j^T  (MMA: [block_m x block_n])
        // ----------------------------------------------------------------
        b.comment("  dP_ij = dO_i @ V_j^T via mma.sync.aligned.m16n8k16");
        let k_steps_v = (head_dim / 16) as usize;
        let n_tiles_dp = (block_n / 8) as usize;
        let mut dp_acc = alloc_f32_accum(b, m_tiles_bm * n_tiles_dp * 4);

        for m_t in 0..m_tiles_bm {
            for n_t in 0..n_tiles_dp {
                let dp_idx = (m_t * n_tiles_dp + n_t) * 4;
                let c_init = [
                    dp_acc[dp_idx].clone(),
                    dp_acc[dp_idx + 1].clone(),
                    dp_acc[dp_idx + 2].clone(),
                    dp_acc[dp_idx + 3].clone(),
                ];
                let mut running_dp = c_init;

                for k_t in 0..k_steps_v {
                    let do_byte_off = m_t as u32 * 16 * head_dim * 2 + k_t as u32 * 16 * 2;
                    let do_stage_b = block_m * head_dim * elem_bytes;
                    let do_stage_reg = b.mov_imm_u32(do_stage_b);
                    let do_stage_off = b.mul_lo_u32(stage.clone(), do_stage_reg);
                    let do_base_u32 = smem_base_u32(b, "do_smem");
                    let do_with_st = b.add_u32(do_base_u32, do_stage_off);
                    let do_addr_u32 = smem_add_offset_u32(b, do_with_st, do_byte_off);

                    let v_byte_off = n_t as u32 * head_dim * 2 + k_t as u32 * 16 * 2;
                    let v_base_u32 = smem_base_u32(b, "v_smem");
                    let v_addr_u32 = smem_add_offset_u32(b, v_base_u32, v_byte_off);

                    let a_regs = ldmatrix_a_x4(b, &do_addr_u32);
                    let br_regs = ldmatrix_b_x2_trans(b, &v_addr_u32);
                    running_dp = emit_mma_tile(b, a_regs, br_regs, running_dp);
                }

                dp_acc[dp_idx..(dp_idx + 4)].clone_from_slice(&running_dp);
            }
        }

        // ----------------------------------------------------------------
        // dS_ij = P_ij * (dP_ij - D_i)
        // ----------------------------------------------------------------
        b.comment("  dS_ij = P_ij * (dP_ij - D_i)");
        let mut ds_acc = alloc_f32_accum(b, m_tiles_bm * n_tiles_dp * 4);

        for m_t in 0..m_tiles_bm {
            for row in 0..2usize {
                let d_off_val = m_t as u32 * 16 + row as u32 * 8;
                let d_off_reg = b.mov_imm_u32(d_off_val);
                let bm_reg = b.mov_imm_u32(block_m);
                let di_row = b.mad_lo_u32(q_tile_idx.clone(), bm_reg, d_off_reg);
                let di_addr = b.f32_elem_addr(di_base.clone(), di_row);
                let d_i = b.load_global_f32(di_addr);

                for n_t in 0..n_tiles_dp {
                    let ib = (m_t * n_tiles_dp + n_t) * 4;
                    for k in 0..4usize {
                        if k / 2 == row {
                            let p_val = s_acc[ib + k].clone();
                            let dp_val = dp_acc[ib + k].clone();
                            let dp_min_d = b.sub_f32(dp_val, d_i.clone());
                            let zero_ds = mov_imm_f32(b, 0.0f32);
                            ds_acc[ib + k] = b.fma_f32(p_val, dp_min_d, zero_ds);
                        }
                    }
                }
            }
        }

        // ----------------------------------------------------------------
        // dQ_i += sm_scale * dS_ij @ K_j  (atomic add to global)
        // ----------------------------------------------------------------
        b.comment("  dQ_i += sm_scale * dS_ij @ K_j via MMA + atom.add.global.f32");
        let k_steps_dk = (block_n / 16) as usize;

        for m_t in 0..m_tiles_bm {
            for hd_t in 0..n_tiles_hd {
                let mut dq_tile = alloc_f32_accum(b, 4);

                for k_t in 0..k_steps_dk {
                    let ds_byte_off = m_t as u32 * 16 * block_n * 2 + k_t as u32 * 16 * 2;
                    let ds_base_u32 = smem_base_u32(b, "softmax_smem");
                    let ds_addr_u32 = smem_add_offset_u32(b, ds_base_u32, ds_byte_off);

                    let k_byte_off = hd_t as u32 * 8 * 2 + k_t as u32 * 16 * 2;
                    let k_base_u32 = smem_base_u32(b, "k_smem");
                    let k_addr_u32 = smem_add_offset_u32(b, k_base_u32, k_byte_off);

                    let a_regs = ldmatrix_a_x4(b, &ds_addr_u32);
                    let br_regs = ldmatrix_b_x2_trans(b, &k_addr_u32);
                    let c_init = [
                        dq_tile[0].clone(),
                        dq_tile[1].clone(),
                        dq_tile[2].clone(),
                        dq_tile[3].clone(),
                    ];
                    let new_dq = emit_mma_tile(b, a_regs, br_regs, c_init);
                    dq_tile[..4].clone_from_slice(&new_dq);
                }

                // Atomic add scaled dQ to global dQ buffer
                #[allow(clippy::needless_range_loop)]
                for k in 0..4usize {
                    let row_off = m_t as u32 * 16 + (k / 2) as u32 * 8;
                    let col_off = hd_t as u32 * 8 + (k % 2) as u32 * 4;
                    let flat_reg = b.mov_imm_u32(row_off * head_dim + col_off);
                    let bm_reg = b.mov_imm_u32(block_m * head_dim);
                    let dq_row_off = b.mad_lo_u32(q_tile_idx.clone(), bm_reg, flat_reg);
                    let dq_addr = b.f32_elem_addr(dq_base.clone(), dq_row_off);
                    let scaled_dq = scale_f32(b, dq_tile[k].clone(), scale.clone());
                    let _ = b.atom_global_add_f32(dq_addr, scaled_dq);
                }
            }
        }

        // ----------------------------------------------------------------
        // dK_j += sm_scale * dS_ij^T @ Q_i  (MMA accumulate)
        // ----------------------------------------------------------------
        b.comment("  dK_j += sm_scale * dS^T @ Q_i via mma.sync.aligned.m16n8k16");
        let k_steps_dq = m_tiles_bm; // block_m / 16

        for n_t in 0..n_tiles_dkv {
            for hd_t in 0..n_tiles_hd {
                let dk_idx = (n_t * n_tiles_hd + hd_t) * 4;
                let c_init = [
                    dk_acc[dk_idx].clone(),
                    dk_acc[dk_idx + 1].clone(),
                    dk_acc[dk_idx + 2].clone(),
                    dk_acc[dk_idx + 3].clone(),
                ];
                let mut running_dk = c_init;

                for k_t in 0..k_steps_dq {
                    let ds_byte_off = n_t as u32 * block_m * 2 + k_t as u32 * 16 * 2;
                    let ds_base_u32 = smem_base_u32(b, "softmax_smem");
                    let ds_addr_u32 = smem_add_offset_u32(b, ds_base_u32, ds_byte_off);

                    let q_byte_off = k_t as u32 * 16 * head_dim * 2 + hd_t as u32 * 8 * 2;
                    let q_stage_b = block_m * head_dim * elem_bytes;
                    let q_stage_reg = b.mov_imm_u32(q_stage_b);
                    let q_stage_off = b.mul_lo_u32(stage.clone(), q_stage_reg);
                    let q_base_u32 = smem_base_u32(b, "q_smem");
                    let q_with_st = b.add_u32(q_base_u32, q_stage_off);
                    let q_addr_u32 = smem_add_offset_u32(b, q_with_st, q_byte_off);

                    let a_regs = ldmatrix_a_x4(b, &ds_addr_u32);
                    let br_regs = ldmatrix_b_x2_trans(b, &q_addr_u32);
                    running_dk = emit_mma_tile(b, a_regs, br_regs, running_dk);
                }

                dk_acc[dk_idx..(dk_idx + 4)].clone_from_slice(&running_dk);
            }
        }

        b.bar_sync(2); // consumer done
        let _ = (s_acc, ds_acc, dp_acc);
    });

    // Advance Q-tile index
    let one = b.mov_imm_u32(1);
    b.raw_ptx(&format!("add.u32 {q_tile_idx}, {q_tile_idx}, {one};"));
    b.branch(&q_loop_lbl);
    b.label(&q_end_lbl);

    // -----------------------------------------------------------------------
    // Store dK_j and dV_j
    // -----------------------------------------------------------------------
    b.comment("--- Store dK_j and dV_j to global memory ---");
    b.if_ge_u32(warp_id.clone(), producer_thresh, |b| {
        // Store dK (scaled by sm_scale)
        for n_t in 0..n_tiles_dkv {
            for hd_t in 0..n_tiles_hd {
                let dk_idx = (n_t * n_tiles_hd + hd_t) * 4;
                for k in 0..4usize {
                    let row_off = n_t as u32 * 8 + (k / 2) as u32 * 4;
                    let col_off = hd_t as u32 * 8 + (k % 2) as u32 * 4;
                    let flat_reg = b.mov_imm_u32(row_off * head_dim + col_off);
                    let bn_reg = b.mov_imm_u32(block_n * head_dim);
                    let dk_row = b.mad_lo_u32(bid_x.clone(), bn_reg, flat_reg);
                    let dk_addr = b.f32_elem_addr(dk_base.clone(), dk_row);
                    let scaled = scale_f32(b, dk_acc[dk_idx + k].clone(), scale.clone());
                    b.store_global_f32(dk_addr, scaled);
                }
            }
        }

        // Store dV (no scale)
        for n_t in 0..n_tiles_dkv {
            for hd_t in 0..n_tiles_hd {
                let dv_idx = (n_t * n_tiles_hd + hd_t) * 4;
                for k in 0..4usize {
                    let row_off = n_t as u32 * 8 + (k / 2) as u32 * 4;
                    let col_off = hd_t as u32 * 8 + (k % 2) as u32 * 4;
                    let flat_reg = b.mov_imm_u32(row_off * head_dim + col_off);
                    let bn_reg = b.mov_imm_u32(block_n * head_dim);
                    let dv_row = b.mad_lo_u32(bid_x.clone(), bn_reg, flat_reg);
                    let dv_addr = b.f32_elem_addr(dv_base.clone(), dv_row);
                    b.store_global_f32(dv_addr, dv_acc[dv_idx + k].clone());
                }
            }
        }
    });

    b.fence_acq_rel(FenceScope::Cta);
    b.bar_sync(0);

    let _ = (tid, warp_id, nq_tiles, q_tile_idx, stage);
    b.ret();
}
