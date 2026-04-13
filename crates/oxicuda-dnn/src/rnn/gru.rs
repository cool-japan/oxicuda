//! GRU (Gated Recurrent Unit) cell implementation.
//!
//! Implements the standard GRU cell equations:
//!
//! ```text
//! z_t = sigmoid(W_xz * x_t + W_hz * h_{t-1} + b_z)   // update gate
//! r_t = sigmoid(W_xr * x_t + W_hr * h_{t-1} + b_r)   // reset gate
//! h_candidate = tanh(W_xh * x_t + r_t * (W_hh * h_{t-1}) + b_h)
//! h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate
//! ```
//!
//! Each GPU thread handles one `(batch, hidden_unit)` pair.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::ptx_helpers::*;

/// Block size for GRU gate-fusion kernels.
const GRU_BLOCK: u32 = 256;

// ---------------------------------------------------------------------------
// Weight descriptor
// ---------------------------------------------------------------------------

/// Weight matrices and biases for a single GRU layer.
///
/// The three gate weights (update, reset, candidate) are stored concatenated:
///
/// - `w_x`: shape `[3 * hidden_size, input_size]` — input-to-hidden weights
/// - `w_h`: shape `[3 * hidden_size, hidden_size]` — hidden-to-hidden weights
/// - `bias`: shape `[3 * hidden_size]` — combined biases
///
/// Gate order: `[z (update), r (reset), h (candidate)]`.
pub struct GruWeights<'a, T: GpuFloat> {
    /// Input-to-hidden weight matrix `[3H, input_size]`.
    pub w_x: &'a DeviceBuffer<T>,
    /// Hidden-to-hidden weight matrix `[3H, hidden_size]`.
    pub w_h: &'a DeviceBuffer<T>,
    /// Bias vector `[3H]`.
    pub bias: &'a DeviceBuffer<T>,
    /// Input feature dimension.
    pub input_size: usize,
    /// Hidden state dimension.
    pub hidden_size: usize,
}

impl<'a, T: GpuFloat> GruWeights<'a, T> {
    /// Validates that all weight buffers have correct sizes.
    fn validate(&self) -> DnnResult<()> {
        let three_h = 3 * self.hidden_size;

        let w_x_required = three_h * self.input_size;
        if self.w_x.len() < w_x_required {
            return Err(DnnError::BufferTooSmall {
                expected: w_x_required,
                actual: self.w_x.len(),
            });
        }

        let w_h_required = three_h * self.hidden_size;
        if self.w_h.len() < w_h_required {
            return Err(DnnError::BufferTooSmall {
                expected: w_h_required,
                actual: self.w_h.len(),
            });
        }

        let bias_required = three_h;
        if self.bias.len() < bias_required {
            return Err(DnnError::BufferTooSmall {
                expected: bias_required,
                actual: self.bias.len(),
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Single timestep forward
// ---------------------------------------------------------------------------

/// Performs a GRU forward pass for a single timestep.
///
/// Given input `x_t` of shape `[batch_size, input_size]` and previous hidden
/// state `h_{t-1}` of shape `[batch_size, hidden_size]`, computes the new
/// hidden state `h_t`.
///
/// # Errors
///
/// Returns errors if buffers are undersized, dimensions are invalid, or PTX
/// generation / kernel launch fails.
pub fn gru_cell_forward<T: GpuFloat>(
    handle: &DnnHandle,
    weights: &GruWeights<'_, T>,
    batch_size: usize,
    x_t: &DeviceBuffer<T>,
    h_prev: &DeviceBuffer<T>,
    h_out: &mut DeviceBuffer<T>,
) -> DnnResult<()> {
    let hidden_size = weights.hidden_size;
    let input_size = weights.input_size;

    if batch_size == 0 || hidden_size == 0 || input_size == 0 {
        return Err(DnnError::InvalidArgument(
            "GRU: batch_size, hidden_size, and input_size must be non-zero".into(),
        ));
    }

    weights.validate()?;

    let bh = batch_size * hidden_size;
    let bi = batch_size * input_size;

    if x_t.len() < bi {
        return Err(DnnError::BufferTooSmall {
            expected: bi,
            actual: x_t.len(),
        });
    }
    if h_prev.len() < bh {
        return Err(DnnError::BufferTooSmall {
            expected: bh,
            actual: h_prev.len(),
        });
    }
    if h_out.len() < bh {
        return Err(DnnError::BufferTooSmall {
            expected: bh,
            actual: h_out.len(),
        });
    }

    let ptx = generate_gru_fused_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel_name = format!("dnn_gru_fused_{}", T::NAME);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let total_threads = bh as u32;
    let grid = grid_size_for(total_threads, GRU_BLOCK);
    let params = LaunchParams::new(grid, GRU_BLOCK);

    let args = (
        x_t.as_device_ptr(),
        h_prev.as_device_ptr(),
        weights.w_x.as_device_ptr(),
        weights.w_h.as_device_ptr(),
        weights.bias.as_device_ptr(),
        h_out.as_device_ptr(),
        batch_size as u32,
        hidden_size as u32,
        input_size as u32,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("GRU cell forward: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Sequence forward
// ---------------------------------------------------------------------------

/// Performs a GRU forward pass over a full sequence.
///
/// Processes `seq_len` timesteps sequentially. Input `x` has shape
/// `[seq_len, batch_size, input_size]`. Initial state `h_0` has shape
/// `[batch_size, hidden_size]`.
///
/// Output `h_seq` has shape `[seq_len, batch_size, hidden_size]` and
/// contains the hidden state at each timestep. The final hidden state
/// is written to `h_n`.
///
/// # Errors
///
/// Returns errors if buffers are undersized or any timestep kernel fails.
#[allow(clippy::too_many_arguments)]
pub fn gru_sequence_forward<T: GpuFloat>(
    handle: &DnnHandle,
    weights: &GruWeights<'_, T>,
    seq_len: usize,
    batch_size: usize,
    x: &DeviceBuffer<T>,
    h_0: &DeviceBuffer<T>,
    h_seq: &mut DeviceBuffer<T>,
    h_n: &mut DeviceBuffer<T>,
) -> DnnResult<()> {
    let hidden_size = weights.hidden_size;
    let input_size = weights.input_size;

    if seq_len == 0 {
        return Err(DnnError::InvalidArgument(
            "GRU sequence: seq_len must be non-zero".into(),
        ));
    }
    if batch_size == 0 || hidden_size == 0 || input_size == 0 {
        return Err(DnnError::InvalidArgument(
            "GRU sequence: batch_size, hidden_size, input_size must be non-zero".into(),
        ));
    }

    let bh = batch_size * hidden_size;
    let bi = batch_size * input_size;

    let x_required = seq_len * bi;
    if x.len() < x_required {
        return Err(DnnError::BufferTooSmall {
            expected: x_required,
            actual: x.len(),
        });
    }
    let h_seq_required = seq_len * bh;
    if h_seq.len() < h_seq_required {
        return Err(DnnError::BufferTooSmall {
            expected: h_seq_required,
            actual: h_seq.len(),
        });
    }
    if h_0.len() < bh {
        return Err(DnnError::BufferTooSmall {
            expected: bh,
            actual: h_0.len(),
        });
    }
    if h_n.len() < bh {
        return Err(DnnError::BufferTooSmall {
            expected: bh,
            actual: h_n.len(),
        });
    }

    // Generate the kernel once
    let ptx = generate_gru_fused_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel_name = format!("dnn_gru_fused_{}", T::NAME);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let total_threads = bh as u32;
    let grid = grid_size_for(total_threads, GRU_BLOCK);
    let params = LaunchParams::new(grid, GRU_BLOCK);

    let elem_bytes = T::SIZE as u64;
    let bh_bytes = (bh as u64) * elem_bytes;
    let bi_bytes = (bi as u64) * elem_bytes;

    let x_base = x.as_device_ptr();
    let h_seq_base = h_seq.as_device_ptr();

    // Timestep 0: h_prev = h_0
    let args_0 = (
        x_base,
        h_0.as_device_ptr(),
        weights.w_x.as_device_ptr(),
        weights.w_h.as_device_ptr(),
        weights.bias.as_device_ptr(),
        h_seq_base,
        batch_size as u32,
        hidden_size as u32,
        input_size as u32,
    );

    kernel
        .launch(&params, handle.stream(), &args_0)
        .map_err(|e| DnnError::LaunchFailed(format!("GRU sequence t=0: {e}")))?;

    // Timesteps 1..seq_len
    for t in 1..seq_len {
        let x_ptr = x_base + (t as u64) * bi_bytes;
        let h_prev_ptr = h_seq_base + ((t - 1) as u64) * bh_bytes;
        let h_out_ptr = h_seq_base + (t as u64) * bh_bytes;

        let args_t = (
            x_ptr,
            h_prev_ptr,
            weights.w_x.as_device_ptr(),
            weights.w_h.as_device_ptr(),
            weights.bias.as_device_ptr(),
            h_out_ptr,
            batch_size as u32,
            hidden_size as u32,
            input_size as u32,
        );

        kernel
            .launch(&params, handle.stream(), &args_t)
            .map_err(|e| DnnError::LaunchFailed(format!("GRU sequence t={t}: {e}")))?;
    }

    // Copy final hidden state: h_seq[last] -> h_n via copy kernel
    let copy_ptx = generate_copy_kernel_ptx_gru::<T>(handle.sm_version())?;
    let copy_mod = Arc::new(Module::from_ptx(&copy_ptx)?);
    let copy_name = format!("dnn_gru_copy_{}", T::NAME);
    let copy_kernel_fn = Kernel::from_module(copy_mod, &copy_name)?;

    let copy_n = bh as u32;
    let copy_grid = grid_size_for(copy_n, GRU_BLOCK);
    let copy_params = LaunchParams::new(copy_grid, GRU_BLOCK);
    let h_last_ptr = h_seq_base + ((seq_len - 1) as u64) * bh_bytes;
    let copy_args = (h_last_ptr, h_n.as_device_ptr(), copy_n);

    copy_kernel_fn
        .launch(&copy_params, handle.stream(), &copy_args)
        .map_err(|e| DnnError::LaunchFailed(format!("GRU copy final h: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates a fused GRU gate-computation PTX kernel.
///
/// Each thread handles one `(batch, hidden_unit)` pair and computes:
/// 1. z = sigmoid(W_xz * x + W_hz * h + b_z)  (update gate)
/// 2. r = sigmoid(W_xr * x + W_hr * h + b_r)  (reset gate)
/// 3. h_cand = tanh(W_xh * x + r * (W_hh * h) + b_h)  (candidate)
/// 4. h_t = (1 - z) * h_prev + z * h_cand
fn generate_gru_fused_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let kernel_name = format!("dnn_gru_fused_{}", T::NAME);

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .max_threads_per_block(GRU_BLOCK)
        .param("x_ptr", PtxType::U64)
        .param("h_prev_ptr", PtxType::U64)
        .param("w_x_ptr", PtxType::U64)
        .param("w_h_ptr", PtxType::U64)
        .param("bias_ptr", PtxType::U64)
        .param("h_out_ptr", PtxType::U64)
        .param("batch_size", PtxType::U32)
        .param("hidden_size", PtxType::U32)
        .param("input_size", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let batch_size_reg = b.load_param_u32("batch_size");
            let hidden_size_reg = b.load_param_u32("hidden_size");
            let input_size_reg = b.load_param_u32("input_size");

            let total = b.mul_lo_u32(batch_size_reg.clone(), hidden_size_reg.clone());

            b.if_lt_u32(gid.clone(), total, move |b| {
                let batch_idx = b.alloc_reg(PtxType::U32);
                let hidden_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {batch_idx}, {gid}, {hidden_size_reg};"));
                b.raw_ptx(&format!("rem.u32 {hidden_idx}, {gid}, {hidden_size_reg};"));

                let x_ptr = b.load_param_u64("x_ptr");
                let h_prev_ptr = b.load_param_u64("h_prev_ptr");
                let w_x_ptr = b.load_param_u64("w_x_ptr");
                let w_h_ptr = b.load_param_u64("w_h_ptr");
                let bias_ptr = b.load_param_u64("bias_ptr");

                // Initialize three gate accumulators with bias
                // Gate order: z=0, r=1, h_cand=2
                let mut gate_accum = Vec::new();
                for gate in 0u32..3 {
                    let bias_offset = b.alloc_reg(PtxType::U32);
                    let gate_offset = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mul.lo.u32 {gate_offset}, {hidden_size_reg}, {gate};"
                    ));
                    b.raw_ptx(&format!(
                        "add.u32 {bias_offset}, {gate_offset}, {hidden_idx};"
                    ));
                    let bias_addr =
                        b.byte_offset_addr(bias_ptr.clone(), bias_offset, T::size_u32());
                    let acc = load_global_float::<T>(b, bias_addr);
                    gate_accum.push(acc);
                }

                // Accumulate W_x * x for all 3 gates
                let k_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {k_reg}, 0;"));

                let loop_k = b.fresh_label("gru_k_loop");
                let end_k = b.fresh_label("gru_k_end");
                b.label(&loop_k);
                let p_k = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_k}, {k_reg}, {input_size_reg};"));
                b.branch_if(p_k, &end_k);

                let x_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {x_offset}, {batch_idx}, {input_size_reg}, {k_reg};"
                ));
                let x_addr = b.byte_offset_addr(x_ptr.clone(), x_offset, T::size_u32());
                let x_val = load_global_float::<T>(b, x_addr);

                for gate in 0u32..3 {
                    let w_row = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {w_row}, {hidden_size_reg}, {gate}, {hidden_idx};"
                    ));
                    let w_offset = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {w_offset}, {w_row}, {input_size_reg}, {k_reg};"
                    ));
                    let w_addr = b.byte_offset_addr(w_x_ptr.clone(), w_offset, T::size_u32());
                    let w_val = load_global_float::<T>(b, w_addr);
                    let new_acc =
                        fma_float::<T>(b, w_val, x_val.clone(), gate_accum[gate as usize].clone());
                    gate_accum[gate as usize] = new_acc;
                }

                b.raw_ptx(&format!("add.u32 {k_reg}, {k_reg}, 1;"));
                b.branch(&loop_k);
                b.label(&end_k);

                // Accumulate W_h * h_prev for z and r gates (gates 0, 1)
                // For candidate (gate 2), we need r * (W_hh * h), so accumulate separately
                let mut wh_h_accum_zr = [load_float_imm::<T>(b, 0.0), load_float_imm::<T>(b, 0.0)];
                let mut wh_h_accum_cand = load_float_imm::<T>(b, 0.0);

                let kh_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {kh_reg}, 0;"));

                let loop_kh = b.fresh_label("gru_kh_loop");
                let end_kh = b.fresh_label("gru_kh_end");
                b.label(&loop_kh);
                let p_kh = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_kh}, {kh_reg}, {hidden_size_reg};"));
                b.branch_if(p_kh, &end_kh);

                let h_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {h_offset}, {batch_idx}, {hidden_size_reg}, {kh_reg};"
                ));
                let h_addr = b.byte_offset_addr(h_prev_ptr.clone(), h_offset, T::size_u32());
                let h_val = load_global_float::<T>(b, h_addr);

                // z gate (0) and r gate (1)
                for (gi, acc) in wh_h_accum_zr.iter_mut().enumerate() {
                    let gate = gi as u32;
                    let w_row = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {w_row}, {hidden_size_reg}, {gate}, {hidden_idx};"
                    ));
                    let w_offset = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {w_offset}, {w_row}, {hidden_size_reg}, {kh_reg};"
                    ));
                    let w_addr = b.byte_offset_addr(w_h_ptr.clone(), w_offset, T::size_u32());
                    let w_val = load_global_float::<T>(b, w_addr);
                    let new_acc = fma_float::<T>(b, w_val, h_val.clone(), acc.clone());
                    *acc = new_acc;
                }

                // candidate gate (2): accumulate W_hh * h separately
                {
                    let gate = 2u32;
                    let w_row = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {w_row}, {hidden_size_reg}, {gate}, {hidden_idx};"
                    ));
                    let w_offset = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {w_offset}, {w_row}, {hidden_size_reg}, {kh_reg};"
                    ));
                    let w_addr = b.byte_offset_addr(w_h_ptr.clone(), w_offset, T::size_u32());
                    let w_val = load_global_float::<T>(b, w_addr);
                    wh_h_accum_cand = fma_float::<T>(b, w_val, h_val, wh_h_accum_cand);
                }

                b.raw_ptx(&format!("add.u32 {kh_reg}, {kh_reg}, 1;"));
                b.branch(&loop_kh);
                b.label(&end_kh);

                // Add W_h*h to z and r gate accumulators
                gate_accum[0] = add_float::<T>(b, gate_accum[0].clone(), wh_h_accum_zr[0].clone());
                gate_accum[1] = add_float::<T>(b, gate_accum[1].clone(), wh_h_accum_zr[1].clone());

                // Apply sigmoid to z and r
                let one = load_float_imm::<T>(b, 1.0);
                let neg_one = load_float_imm::<T>(b, -1.0);

                // z = sigmoid(gate_accum[0])
                let neg_z = mul_float::<T>(b, gate_accum[0].clone(), neg_one.clone());
                let exp_neg_z = emit_approx_exp_gru::<T>(b, neg_z);
                let denom_z = add_float::<T>(b, one.clone(), exp_neg_z);
                let z_gate = div_float::<T>(b, one.clone(), denom_z);

                // r = sigmoid(gate_accum[1])
                let neg_r = mul_float::<T>(b, gate_accum[1].clone(), neg_one.clone());
                let exp_neg_r = emit_approx_exp_gru::<T>(b, neg_r);
                let denom_r = add_float::<T>(b, one.clone(), exp_neg_r);
                let r_gate = div_float::<T>(b, one.clone(), denom_r);

                // candidate: h_cand = tanh(gate_accum[2] + r * wh_h_accum_cand)
                let r_wh = mul_float::<T>(b, r_gate, wh_h_accum_cand);
                let cand_pre = add_float::<T>(b, gate_accum[2].clone(), r_wh);

                // tanh via 2*sigmoid(2x) - 1
                let two = load_float_imm::<T>(b, 2.0);
                let two_cand = mul_float::<T>(b, cand_pre, two.clone());
                let neg_two_cand = mul_float::<T>(b, two_cand, neg_one.clone());
                let exp_neg_2c = emit_approx_exp_gru::<T>(b, neg_two_cand);
                let denom_c = add_float::<T>(b, one.clone(), exp_neg_2c);
                let sig_2c = div_float::<T>(b, one.clone(), denom_c);
                let two_sig = mul_float::<T>(b, two, sig_2c);
                let h_cand = add_float::<T>(b, two_sig, neg_one);

                // h_t = (1 - z) * h_prev + z * h_cand
                let h_idx_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {h_idx_offset}, {batch_idx}, {hidden_size_reg}, {hidden_idx};"
                ));
                let h_prev_addr =
                    b.byte_offset_addr(h_prev_ptr, h_idx_offset.clone(), T::size_u32());
                let h_prev_val = load_global_float::<T>(b, h_prev_addr);

                let neg_one_z = load_float_imm::<T>(b, -1.0);
                let neg_z = mul_float::<T>(b, z_gate.clone(), neg_one_z);
                let one_minus_z = add_float::<T>(b, one, neg_z);
                let term1 = mul_float::<T>(b, one_minus_z, h_prev_val);
                let term2 = mul_float::<T>(b, z_gate, h_cand);
                let h_new = add_float::<T>(b, term1, term2);

                // Store h_t
                let h_out_ptr = b.load_param_u64("h_out_ptr");
                let h_out_addr = b.byte_offset_addr(h_out_ptr, h_idx_offset, T::size_u32());
                store_global_float::<T>(b, h_out_addr, h_new);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("GRU fused gate: {e}")))?;

    Ok(ptx)
}

/// Generates a simple D2D copy kernel used for final state extraction.
fn generate_copy_kernel_ptx_gru<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let kernel_name = format!("dnn_gru_copy_{}", T::NAME);

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .max_threads_per_block(GRU_BLOCK)
        .param("src_ptr", PtxType::U64)
        .param("dst_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let src = b.load_param_u64("src_ptr");
                let dst = b.load_param_u64("dst_ptr");
                let src_addr = b.byte_offset_addr(src, gid.clone(), T::size_u32());
                let val = load_global_float::<T>(b, src_addr);
                let dst_addr = b.byte_offset_addr(dst, gid, T::size_u32());
                store_global_float::<T>(b, dst_addr, val);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("GRU copy kernel: {e}")))?;

    Ok(ptx)
}

/// Emits an approximate exp(x) using `ex2.approx` for GRU gates.
fn emit_approx_exp_gru<T: GpuFloat>(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    x: oxicuda_ptx::ir::Register,
) -> oxicuda_ptx::ir::Register {
    let log2_e = load_float_imm::<T>(b, std::f64::consts::LOG2_E);
    let scaled = mul_float::<T>(b, x, log2_e);

    if T::PTX_TYPE == PtxType::F32 {
        let result = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("ex2.approx.f32 {result}, {scaled};"));
        result
    } else {
        let f32_val = b.cvt_f64_to_f32(scaled);
        let exp_f32 = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("ex2.approx.f32 {exp_f32}, {f32_val};"));
        b.cvt_f32_to_f64(exp_f32)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gru_fused_ptx_f32() {
        let ptx = generate_gru_fused_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("should generate");
        assert!(ptx_str.contains("dnn_gru_fused_f32"));
        assert!(ptx_str.contains("ex2.approx.f32"));
    }

    #[test]
    fn gru_fused_ptx_f64() {
        let ptx = generate_gru_fused_ptx::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("should generate");
        assert!(ptx_str.contains("dnn_gru_fused_f64"));
    }

    #[test]
    fn gru_fused_ptx_sm90() {
        let ptx = generate_gru_fused_ptx::<f32>(SmVersion::Sm90);
        assert!(ptx.is_ok());
    }

    #[test]
    fn gru_fused_ptx_sm75() {
        let ptx = generate_gru_fused_ptx::<f32>(SmVersion::Sm75);
        assert!(ptx.is_ok());
    }

    #[test]
    fn gru_fused_ptx_contains_gates() {
        let ptx = generate_gru_fused_ptx::<f32>(SmVersion::Sm80).expect("should generate");
        assert!(ptx.contains("gru_k_loop"));
        assert!(ptx.contains("gru_kh_loop"));
    }

    #[test]
    fn gru_fused_ptx_has_store() {
        let ptx = generate_gru_fused_ptx::<f32>(SmVersion::Sm80).expect("should generate");
        assert!(ptx.contains("st.global"));
    }
}
