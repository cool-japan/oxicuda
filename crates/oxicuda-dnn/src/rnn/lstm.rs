//! LSTM (Long Short-Term Memory) cell implementation.
//!
//! Implements the standard LSTM cell equations:
//!
//! ```text
//! i_t = sigmoid(W_xi * x_t + W_hi * h_{t-1} + b_i)
//! f_t = sigmoid(W_xf * x_t + W_hf * h_{t-1} + b_f)
//! g_t = tanh(W_xg * x_t + W_hg * h_{t-1} + b_g)
//! o_t = sigmoid(W_xo * x_t + W_ho * h_{t-1} + b_o)
//! c_t = f_t * c_{t-1} + i_t * g_t
//! h_t = o_t * tanh(c_t)
//! ```
//!
//! The PTX kernel fuses all four gate computations so that each GPU thread
//! handles one hidden unit, computing all gates and the state update in a
//! single pass.

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

/// Block size for LSTM gate-fusion kernels.
const LSTM_BLOCK: u32 = 256;

// ---------------------------------------------------------------------------
// Weight descriptor
// ---------------------------------------------------------------------------

/// Weight matrices and biases for a single LSTM layer.
///
/// The four gate weights (input, forget, candidate, output) are stored
/// concatenated along the first axis:
///
/// - `w_x`: shape `[4 * hidden_size, input_size]` — input-to-hidden weights
/// - `w_h`: shape `[4 * hidden_size, hidden_size]` — hidden-to-hidden weights
/// - `bias`: shape `[4 * hidden_size]` — combined biases for all four gates
///
/// Gate order within the concatenated dimension: `[i, f, g, o]`.
pub struct LstmWeights<'a, T: GpuFloat> {
    /// Input-to-hidden weight matrix `[4H, input_size]`.
    pub w_x: &'a DeviceBuffer<T>,
    /// Hidden-to-hidden weight matrix `[4H, hidden_size]`.
    pub w_h: &'a DeviceBuffer<T>,
    /// Bias vector `[4H]`.
    pub bias: &'a DeviceBuffer<T>,
    /// Input feature dimension.
    pub input_size: usize,
    /// Hidden state dimension.
    pub hidden_size: usize,
}

impl<'a, T: GpuFloat> LstmWeights<'a, T> {
    /// Validates that all weight buffers have correct sizes.
    fn validate(&self) -> DnnResult<()> {
        let four_h = 4 * self.hidden_size;

        let w_x_required = four_h * self.input_size;
        if self.w_x.len() < w_x_required {
            return Err(DnnError::BufferTooSmall {
                expected: w_x_required,
                actual: self.w_x.len(),
            });
        }

        let w_h_required = four_h * self.hidden_size;
        if self.w_h.len() < w_h_required {
            return Err(DnnError::BufferTooSmall {
                expected: w_h_required,
                actual: self.w_h.len(),
            });
        }

        let bias_required = four_h;
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

/// Performs an LSTM forward pass for a single timestep.
///
/// Given input `x_t` of shape `[batch_size, input_size]` and previous states
/// `h_{t-1}`, `c_{t-1}` of shape `[batch_size, hidden_size]`, computes the
/// new hidden state `h_t` and cell state `c_t`.
///
/// This function generates and launches a fused PTX kernel where each thread
/// handles one `(batch, hidden_unit)` pair, computing all four gates and the
/// cell/hidden state update.
///
/// # Errors
///
/// Returns errors if buffers are undersized, dimensions are invalid, or PTX
/// generation / kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn lstm_cell_forward<T: GpuFloat>(
    handle: &DnnHandle,
    weights: &LstmWeights<'_, T>,
    batch_size: usize,
    x_t: &DeviceBuffer<T>,
    h_prev: &DeviceBuffer<T>,
    c_prev: &DeviceBuffer<T>,
    h_out: &mut DeviceBuffer<T>,
    c_out: &mut DeviceBuffer<T>,
) -> DnnResult<()> {
    let hidden_size = weights.hidden_size;
    let input_size = weights.input_size;

    // Validate dimensions
    if batch_size == 0 || hidden_size == 0 || input_size == 0 {
        return Err(DnnError::InvalidArgument(
            "LSTM: batch_size, hidden_size, and input_size must be non-zero".into(),
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
    if c_prev.len() < bh {
        return Err(DnnError::BufferTooSmall {
            expected: bh,
            actual: c_prev.len(),
        });
    }
    if h_out.len() < bh {
        return Err(DnnError::BufferTooSmall {
            expected: bh,
            actual: h_out.len(),
        });
    }
    if c_out.len() < bh {
        return Err(DnnError::BufferTooSmall {
            expected: bh,
            actual: c_out.len(),
        });
    }

    // Generate and launch the fused LSTM gate kernel
    let ptx = generate_lstm_fused_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel_name = format!("dnn_lstm_fused_{}", T::NAME);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let total_threads = bh as u32;
    let grid = grid_size_for(total_threads, LSTM_BLOCK);
    let params = LaunchParams::new(grid, LSTM_BLOCK);

    let args = (
        x_t.as_device_ptr(),
        h_prev.as_device_ptr(),
        c_prev.as_device_ptr(),
        weights.w_x.as_device_ptr(),
        weights.w_h.as_device_ptr(),
        weights.bias.as_device_ptr(),
        h_out.as_device_ptr(),
        c_out.as_device_ptr(),
        batch_size as u32,
        hidden_size as u32,
        input_size as u32,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("LSTM cell forward: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Sequence forward
// ---------------------------------------------------------------------------

/// Performs an LSTM forward pass over a full sequence.
///
/// Processes `seq_len` timesteps sequentially. Input `x` has shape
/// `[seq_len, batch_size, input_size]`. Initial states `h_0` and `c_0`
/// have shape `[batch_size, hidden_size]`.
///
/// The output `h_seq` has shape `[seq_len, batch_size, hidden_size]` and
/// contains the hidden state at each timestep. The final hidden and cell
/// states are written to `h_n` and `c_n`.
///
/// # Errors
///
/// Returns errors if buffers are undersized or any timestep kernel fails.
#[allow(clippy::too_many_arguments)]
pub fn lstm_sequence_forward<T: GpuFloat>(
    handle: &DnnHandle,
    weights: &LstmWeights<'_, T>,
    seq_len: usize,
    batch_size: usize,
    x: &DeviceBuffer<T>,
    h_0: &DeviceBuffer<T>,
    c_0: &DeviceBuffer<T>,
    h_seq: &mut DeviceBuffer<T>,
    h_n: &mut DeviceBuffer<T>,
    c_n: &mut DeviceBuffer<T>,
) -> DnnResult<()> {
    let hidden_size = weights.hidden_size;
    let input_size = weights.input_size;

    if seq_len == 0 {
        return Err(DnnError::InvalidArgument(
            "LSTM sequence: seq_len must be non-zero".into(),
        ));
    }
    if batch_size == 0 || hidden_size == 0 || input_size == 0 {
        return Err(DnnError::InvalidArgument(
            "LSTM sequence: batch_size, hidden_size, input_size must be non-zero".into(),
        ));
    }

    let bh = batch_size * hidden_size;
    let bi = batch_size * input_size;

    // Validate full-sequence buffers
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
    if c_0.len() < bh {
        return Err(DnnError::BufferTooSmall {
            expected: bh,
            actual: c_0.len(),
        });
    }
    if h_n.len() < bh {
        return Err(DnnError::BufferTooSmall {
            expected: bh,
            actual: h_n.len(),
        });
    }
    if c_n.len() < bh {
        return Err(DnnError::BufferTooSmall {
            expected: bh,
            actual: c_n.len(),
        });
    }

    // Generate the kernel once, reuse across timesteps
    let ptx = generate_lstm_fused_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel_name = format!("dnn_lstm_fused_{}", T::NAME);
    let kernel = Kernel::from_module(module, &kernel_name)?;

    let total_threads = bh as u32;
    let grid = grid_size_for(total_threads, LSTM_BLOCK);
    let params = LaunchParams::new(grid, LSTM_BLOCK);

    let elem_bytes = T::SIZE as u64;
    let bh_bytes = (bh as u64) * elem_bytes;
    let bi_bytes = (bi as u64) * elem_bytes;

    // First timestep uses h_0, c_0 as input
    // Subsequent timesteps read from the h_seq / c_n (ping-pong via temp buffers)
    // We use h_n and c_n as the "current state" scratch space.

    // Step 0: h_prev = h_0, c_prev = c_0, output goes to h_seq[0] and c_n
    let x_base = x.as_device_ptr();
    let h_seq_base = h_seq.as_device_ptr();

    // Timestep 0
    let args_0 = (
        x_base,
        h_0.as_device_ptr(),
        c_0.as_device_ptr(),
        weights.w_x.as_device_ptr(),
        weights.w_h.as_device_ptr(),
        weights.bias.as_device_ptr(),
        h_seq_base,          // h_out -> h_seq[0]
        c_n.as_device_ptr(), // c_out -> c_n (temp for c_t)
        batch_size as u32,
        hidden_size as u32,
        input_size as u32,
    );

    kernel
        .launch(&params, handle.stream(), &args_0)
        .map_err(|e| DnnError::LaunchFailed(format!("LSTM sequence t=0: {e}")))?;

    // Timesteps 1..seq_len-1
    // h_prev reads from h_seq[t-1], c_prev reads from c_n (which holds c_{t-1})
    // h_out writes to h_seq[t], c_out writes to h_n (swap with c_n)
    for t in 1..seq_len {
        let x_ptr = x_base + (t as u64) * bi_bytes;
        let h_prev_ptr = h_seq_base + ((t - 1) as u64) * bh_bytes;
        let h_out_ptr = h_seq_base + (t as u64) * bh_bytes;

        // Alternate c storage between c_n and h_n to avoid aliasing
        let (c_prev_ptr, c_out_ptr) = if t % 2 == 1 {
            (c_n.as_device_ptr(), h_n.as_device_ptr())
        } else {
            (h_n.as_device_ptr(), c_n.as_device_ptr())
        };

        let args_t = (
            x_ptr,
            h_prev_ptr,
            c_prev_ptr,
            weights.w_x.as_device_ptr(),
            weights.w_h.as_device_ptr(),
            weights.bias.as_device_ptr(),
            h_out_ptr,
            c_out_ptr,
            batch_size as u32,
            hidden_size as u32,
            input_size as u32,
        );

        kernel
            .launch(&params, handle.stream(), &args_t)
            .map_err(|e| DnnError::LaunchFailed(format!("LSTM sequence t={t}: {e}")))?;
    }

    // Copy final states to h_n and c_n.
    //
    // After the loop, the final hidden state is in h_seq[seq_len-1].
    // The final cell state is in c_n if the last step wrote there (even index
    // after step 0), or h_n if odd index. We need both h_n and c_n to hold
    // the correct final states.
    //
    // Rather than raw D2D memcpy with pointer offsets, we launch a simple
    // copy kernel to transfer h_seq[last] -> h_n. For the cell state, if it
    // ended up in h_n, we first copy it to c_n using copy_from_device, then
    // overwrite h_n.

    // Determine where the final cell state landed
    if seq_len > 1 && (seq_len - 1) % 2 == 1 {
        // Final c is currently in h_n; copy h_n -> c_n before overwriting
        c_n.copy_from_device(h_n)?;
    }
    // else: final c is already in c_n (step 0 wrote there, or even-indexed last step)

    // Copy final hidden state: h_seq[last] -> h_n
    // We use a PTX copy kernel that reads from an offset within h_seq
    let copy_ptx = generate_copy_kernel_ptx::<T>(handle.sm_version())?;
    let copy_mod = Arc::new(Module::from_ptx(&copy_ptx)?);
    let copy_name = format!("dnn_copy_{}", T::NAME);
    let copy_kernel = Kernel::from_module(copy_mod, &copy_name)?;

    let copy_n = bh as u32;
    let copy_grid = grid_size_for(copy_n, LSTM_BLOCK);
    let copy_params = LaunchParams::new(copy_grid, LSTM_BLOCK);
    let h_last_ptr = h_seq_base + ((seq_len - 1) as u64) * bh_bytes;
    let copy_args = (h_last_ptr, h_n.as_device_ptr(), copy_n);

    copy_kernel
        .launch(&copy_params, handle.stream(), &copy_args)
        .map_err(|e| DnnError::LaunchFailed(format!("LSTM copy final h: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates a fused LSTM gate-computation PTX kernel.
///
/// Each thread handles one `(batch, hidden_unit)` pair and computes:
/// 1. Dot products: `gate[g] = W_x[g] . x + W_h[g] . h_prev + bias[g]` for g in {i,f,g,o}
/// 2. Activations: sigmoid for i, f, o; tanh for g
/// 3. Cell update: `c_t = f * c_prev + i * g`
/// 4. Hidden update: `h_t = o * tanh(c_t)`
fn generate_lstm_fused_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let kernel_name = format!("dnn_lstm_fused_{}", T::NAME);

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .max_threads_per_block(LSTM_BLOCK)
        .param("x_ptr", PtxType::U64)
        .param("h_prev_ptr", PtxType::U64)
        .param("c_prev_ptr", PtxType::U64)
        .param("w_x_ptr", PtxType::U64)
        .param("w_h_ptr", PtxType::U64)
        .param("bias_ptr", PtxType::U64)
        .param("h_out_ptr", PtxType::U64)
        .param("c_out_ptr", PtxType::U64)
        .param("batch_size", PtxType::U32)
        .param("hidden_size", PtxType::U32)
        .param("input_size", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let batch_size_reg = b.load_param_u32("batch_size");
            let hidden_size_reg = b.load_param_u32("hidden_size");
            let input_size_reg = b.load_param_u32("input_size");

            // total_threads = batch_size * hidden_size
            let total = b.mul_lo_u32(batch_size_reg.clone(), hidden_size_reg.clone());

            b.if_lt_u32(gid.clone(), total, move |b| {
                // Decode batch index and hidden index
                let batch_idx = b.alloc_reg(PtxType::U32);
                let hidden_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {batch_idx}, {gid}, {hidden_size_reg};"));
                b.raw_ptx(&format!("rem.u32 {hidden_idx}, {gid}, {hidden_size_reg};"));

                let x_ptr = b.load_param_u64("x_ptr");
                let h_prev_ptr = b.load_param_u64("h_prev_ptr");
                let c_prev_ptr = b.load_param_u64("c_prev_ptr");
                let w_x_ptr = b.load_param_u64("w_x_ptr");
                let w_h_ptr = b.load_param_u64("w_h_ptr");
                let bias_ptr = b.load_param_u64("bias_ptr");

                // Compute 4H (stride for gates in weight matrices)
                let four_h = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shl.b32 {four_h}, {hidden_size_reg}, 2;"));

                // Initialize four gate accumulators with bias
                // Gate order: i=0, f=1, g=2, o=3
                // bias offset for gate g, hidden j: bias[(g * H) + j]
                let mut gate_accum = Vec::new();
                for gate in 0u32..4 {
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

                // Accumulate W_x * x: for each input feature k
                // gate_accum[g] += W_x[(g*H + j) * input_size + k] * x[batch * input_size + k]
                let k_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {k_reg}, 0;"));

                let loop_k = b.fresh_label("lstm_k_loop");
                let end_k = b.fresh_label("lstm_k_end");
                b.label(&loop_k);
                let p_k = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_k}, {k_reg}, {input_size_reg};"));
                b.branch_if(p_k, &end_k);

                // x_val = x[batch * input_size + k]
                let x_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {x_offset}, {batch_idx}, {input_size_reg}, {k_reg};"
                ));
                let x_addr = b.byte_offset_addr(x_ptr.clone(), x_offset, T::size_u32());
                let x_val = load_global_float::<T>(b, x_addr);

                // For each gate, accumulate w_x * x_val
                for gate in 0u32..4 {
                    // W_x row index = gate * H + hidden_idx
                    // W_x col index = k
                    // linear offset = (gate * H + hidden_idx) * input_size + k
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

                // Accumulate W_h * h_prev: for each hidden feature k
                let kh_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {kh_reg}, 0;"));

                let loop_kh = b.fresh_label("lstm_kh_loop");
                let end_kh = b.fresh_label("lstm_kh_end");
                b.label(&loop_kh);
                let p_kh = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_kh}, {kh_reg}, {hidden_size_reg};"));
                b.branch_if(p_kh, &end_kh);

                // h_val = h_prev[batch * hidden_size + k]
                let h_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {h_offset}, {batch_idx}, {hidden_size_reg}, {kh_reg};"
                ));
                let h_addr = b.byte_offset_addr(h_prev_ptr.clone(), h_offset, T::size_u32());
                let h_val = load_global_float::<T>(b, h_addr);

                for gate in 0u32..4 {
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
                    let new_acc =
                        fma_float::<T>(b, w_val, h_val.clone(), gate_accum[gate as usize].clone());
                    gate_accum[gate as usize] = new_acc;
                }

                b.raw_ptx(&format!("add.u32 {kh_reg}, {kh_reg}, 1;"));
                b.branch(&loop_kh);
                b.label(&end_kh);

                // Apply activations:
                // i = sigmoid(gate_accum[0])
                // f = sigmoid(gate_accum[1])
                // g = tanh(gate_accum[2])
                // o = sigmoid(gate_accum[3])

                // sigmoid(x) = 1 / (1 + exp(-x))
                let one = load_float_imm::<T>(b, 1.0);
                let neg_one = load_float_imm::<T>(b, -1.0);

                // i_gate = sigmoid
                let neg_i = mul_float::<T>(b, gate_accum[0].clone(), neg_one.clone());
                let exp_neg_i = emit_approx_exp::<T>(b, neg_i);
                let denom_i = add_float::<T>(b, one.clone(), exp_neg_i);
                let i_gate = div_float::<T>(b, one.clone(), denom_i);

                // f_gate = sigmoid
                let neg_f = mul_float::<T>(b, gate_accum[1].clone(), neg_one.clone());
                let exp_neg_f = emit_approx_exp::<T>(b, neg_f);
                let denom_f = add_float::<T>(b, one.clone(), exp_neg_f);
                let f_gate = div_float::<T>(b, one.clone(), denom_f);

                // g_gate = tanh (using ex2 approximation: tanh(x) = 2*sigmoid(2x) - 1)
                let two = load_float_imm::<T>(b, 2.0);
                let two_g = mul_float::<T>(b, gate_accum[2].clone(), two.clone());
                let neg_two_g = mul_float::<T>(b, two_g, neg_one.clone());
                let exp_neg_2g = emit_approx_exp::<T>(b, neg_two_g);
                let denom_g = add_float::<T>(b, one.clone(), exp_neg_2g);
                let sig_2g = div_float::<T>(b, one.clone(), denom_g);
                let two_sig = mul_float::<T>(b, two.clone(), sig_2g);
                let g_gate = add_float::<T>(b, two_sig, neg_one.clone());

                // o_gate = sigmoid
                let neg_o = mul_float::<T>(b, gate_accum[3].clone(), neg_one.clone());
                let exp_neg_o = emit_approx_exp::<T>(b, neg_o);
                let denom_o = add_float::<T>(b, one.clone(), exp_neg_o);
                let o_gate = div_float::<T>(b, one.clone(), denom_o);

                // Cell state update: c_t = f * c_prev + i * g
                let c_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {c_offset}, {batch_idx}, {hidden_size_reg}, {hidden_idx};"
                ));
                let c_prev_addr = b.byte_offset_addr(c_prev_ptr, c_offset.clone(), T::size_u32());
                let c_prev_val = load_global_float::<T>(b, c_prev_addr);

                let fc = mul_float::<T>(b, f_gate, c_prev_val);
                let ig = mul_float::<T>(b, i_gate, g_gate);
                let c_new = add_float::<T>(b, fc, ig);

                // Hidden state update: h_t = o * tanh(c_t)
                // tanh(c_t) = 2 * sigmoid(2 * c_t) - 1
                let two_c = mul_float::<T>(b, c_new.clone(), two.clone());
                let neg_two_c = mul_float::<T>(b, two_c, neg_one);
                let exp_neg_2c = emit_approx_exp::<T>(b, neg_two_c);
                let denom_c = add_float::<T>(b, one.clone(), exp_neg_2c);
                let sig_2c = div_float::<T>(b, one, denom_c);
                let two_sig_c = mul_float::<T>(b, two, sig_2c);
                let neg_one_2 = load_float_imm::<T>(b, -1.0);
                let tanh_c = add_float::<T>(b, two_sig_c, neg_one_2);
                let h_new = mul_float::<T>(b, o_gate, tanh_c);

                // Store c_t and h_t
                let c_out_ptr = b.load_param_u64("c_out_ptr");
                let c_out_addr = b.byte_offset_addr(c_out_ptr, c_offset.clone(), T::size_u32());
                store_global_float::<T>(b, c_out_addr, c_new);

                let h_out_ptr = b.load_param_u64("h_out_ptr");
                let h_out_addr = b.byte_offset_addr(h_out_ptr, c_offset, T::size_u32());
                store_global_float::<T>(b, h_out_addr, h_new);
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("LSTM fused gate: {e}")))?;

    Ok(ptx)
}

/// Generates a simple D2D copy kernel used for final state extraction.
fn generate_copy_kernel_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let kernel_name = format!("dnn_copy_{}", T::NAME);

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .max_threads_per_block(LSTM_BLOCK)
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
        .map_err(|e| DnnError::PtxGeneration(format!("copy kernel: {e}")))?;

    Ok(ptx)
}

/// Emits an approximate exp(x) using the `ex2.approx` PTX instruction.
///
/// `exp(x) = 2^(x * log2(e))` where `log2(e) ~ 1.4426950408889634`.
fn emit_approx_exp<T: GpuFloat>(
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
        // f64: no native ex2.approx, convert to f32, compute, convert back
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
    fn lstm_fused_ptx_f32() {
        let ptx = generate_lstm_fused_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("should generate");
        assert!(ptx_str.contains("dnn_lstm_fused_f32"));
        assert!(ptx_str.contains("ex2.approx.f32"));
    }

    #[test]
    fn lstm_fused_ptx_f64() {
        let ptx = generate_lstm_fused_ptx::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("should generate");
        assert!(ptx_str.contains("dnn_lstm_fused_f64"));
    }

    #[test]
    fn lstm_fused_ptx_sm90() {
        let ptx = generate_lstm_fused_ptx::<f32>(SmVersion::Sm90);
        assert!(ptx.is_ok());
    }

    #[test]
    fn lstm_fused_ptx_sm75() {
        let ptx = generate_lstm_fused_ptx::<f32>(SmVersion::Sm75);
        assert!(ptx.is_ok());
    }

    #[test]
    fn lstm_fused_ptx_contains_gates() {
        let ptx = generate_lstm_fused_ptx::<f32>(SmVersion::Sm80).expect("should generate");
        // Should have the k-loop for input accumulation
        assert!(ptx.contains("lstm_k_loop"));
        // Should have the kh-loop for hidden accumulation
        assert!(ptx.contains("lstm_kh_loop"));
    }

    #[test]
    fn lstm_fused_ptx_has_store() {
        let ptx = generate_lstm_fused_ptx::<f32>(SmVersion::Sm80).expect("should generate");
        assert!(ptx.contains("st.global"));
    }
}
