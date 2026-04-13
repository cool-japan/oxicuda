//! INT4 and NF4 (NormalFloat4) quantization and dequantization.
//!
//! INT4 packs two 4-bit values per byte for weight-only quantization
//! (used in QLoRA, GPTQ, AWQ). NF4 uses lookup values placed at normal
//! distribution quantiles for better information-theoretic encoding.
//!
//! ## Packing format
//!
//! Two INT4 values are packed into a single `u8`:
//! - Low nibble (bits 3:0) = even-indexed element
//! - High nibble (bits 7:4) = odd-indexed element

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

/// Block size for INT4/NF4 quantization kernels.
const INT4_QUANT_BLOCK: u32 = 256;

/// INT4 representable range (symmetric: [-8, 7], asymmetric: [0, 15]).
const INT4_SYM_MIN: f64 = -8.0;
const INT4_SYM_MAX: f64 = 7.0;
const INT4_ASYM_MAX: f64 = 15.0;

/// NF4 lookup table: 16 values placed at quantiles of a standard normal
/// distribution, providing information-theoretically optimal 4-bit encoding
/// for normally distributed weights.
const NF4_LOOKUP: [f64; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for INT4 quantization.
///
/// Group-wise quantization divides the weight tensor into contiguous groups
/// of `group_size` elements, computing a separate scale (and optionally zero
/// point) for each group.
#[derive(Debug, Clone, Copy)]
pub struct Int4QuantConfig {
    /// Number of elements per quantization group (typically 32 or 128).
    pub group_size: usize,
    /// If `true`, use symmetric quantization (range [-8, 7], zero point = 0).
    /// If `false`, use asymmetric quantization (range [0, 15], with zero point).
    pub symmetric: bool,
}

impl Int4QuantConfig {
    /// Creates a new INT4 quantization configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `group_size` is zero.
    pub fn new(group_size: usize, symmetric: bool) -> DnnResult<Self> {
        if group_size == 0 {
            return Err(DnnError::InvalidArgument(
                "INT4 group_size must be non-zero".into(),
            ));
        }
        Ok(Self {
            group_size,
            symmetric,
        })
    }

    /// Returns the number of quantization groups for `n` elements.
    #[must_use]
    pub fn num_groups(&self, n: usize) -> usize {
        n.div_ceil(self.group_size)
    }

    /// Returns the number of packed output bytes for `n` elements.
    ///
    /// Two 4-bit values are packed per byte, so output size = ceil(n / 2).
    #[must_use]
    pub fn packed_bytes(&self, n: usize) -> usize {
        n.div_ceil(2)
    }
}

// ---------------------------------------------------------------------------
// INT4 quantization
// ---------------------------------------------------------------------------

/// Quantizes floating-point data to INT4, packed two values per byte.
///
/// Computes per-group scale factors (and zero points for asymmetric mode).
/// The packed output uses the low nibble for even-indexed elements and the
/// high nibble for odd-indexed elements within each byte.
///
/// # Arguments
///
/// * `handle` — DNN handle.
/// * `input` — source floating-point data.
/// * `output` — packed INT4 output (`ceil(n/2)` bytes).
/// * `scales` — per-group scale factors (`num_groups` elements).
/// * `zeros` — per-group zero points (only used for asymmetric mode).
/// * `n` — total number of elements.
/// * `config` — quantization configuration.
///
/// # Errors
///
/// Returns errors if buffers are undersized or PTX generation fails.
#[allow(clippy::too_many_arguments)]
pub fn quantize_to_int4<T: GpuFloat>(
    handle: &DnnHandle,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<T>,
    zeros: &mut DeviceBuffer<T>,
    n: usize,
    config: &Int4QuantConfig,
) -> DnnResult<()> {
    if n == 0 {
        return Ok(());
    }

    let num_groups = config.num_groups(n);
    let packed_bytes = config.packed_bytes(n);

    if input.len() < n {
        return Err(DnnError::BufferTooSmall {
            expected: n,
            actual: input.len(),
        });
    }
    if output.len() < packed_bytes {
        return Err(DnnError::BufferTooSmall {
            expected: packed_bytes,
            actual: output.len(),
        });
    }
    if scales.len() < num_groups {
        return Err(DnnError::BufferTooSmall {
            expected: num_groups,
            actual: scales.len(),
        });
    }
    if !config.symmetric && zeros.len() < num_groups {
        return Err(DnnError::BufferTooSmall {
            expected: num_groups,
            actual: zeros.len(),
        });
    }

    // Step 1: Compute per-group scales (and zeros for asymmetric)
    let scale_ptx = generate_int4_scale_ptx::<T>(handle.sm_version(), config)?;
    let scale_mod = Arc::new(Module::from_ptx(&scale_ptx)?);
    let scale_name = format!("dnn_int4_scale_{}", T::NAME);
    let scale_kernel = Kernel::from_module(scale_mod, &scale_name)?;

    let scale_grid = grid_size_for(num_groups as u32, INT4_QUANT_BLOCK);
    let scale_params = LaunchParams::new(scale_grid, INT4_QUANT_BLOCK);
    let scale_args = (
        input.as_device_ptr(),
        scales.as_device_ptr(),
        zeros.as_device_ptr(),
        n as u32,
        config.group_size as u32,
        num_groups as u32,
    );

    scale_kernel
        .launch(&scale_params, handle.stream(), &scale_args)
        .map_err(|e| DnnError::LaunchFailed(format!("INT4 scale compute: {e}")))?;

    // Step 2: Quantize and pack
    let quant_ptx = generate_int4_pack_ptx::<T>(handle.sm_version(), config)?;
    let quant_mod = Arc::new(Module::from_ptx(&quant_ptx)?);
    let quant_name = format!("dnn_int4_pack_{}", T::NAME);
    let quant_kernel = Kernel::from_module(quant_mod, &quant_name)?;

    // Each thread processes one output byte (2 elements)
    let quant_grid = grid_size_for(packed_bytes as u32, INT4_QUANT_BLOCK);
    let quant_params = LaunchParams::new(quant_grid, INT4_QUANT_BLOCK);
    let quant_args = (
        input.as_device_ptr(),
        scales.as_device_ptr(),
        zeros.as_device_ptr(),
        output.as_device_ptr(),
        n as u32,
        config.group_size as u32,
        packed_bytes as u32,
    );

    quant_kernel
        .launch(&quant_params, handle.stream(), &quant_args)
        .map_err(|e| DnnError::LaunchFailed(format!("INT4 pack: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// INT4 dequantization
// ---------------------------------------------------------------------------

/// Dequantizes packed INT4 data back to floating-point.
///
/// `out[i] = (nibble[i] - zero[group]) * scale[group]`
///
/// # Errors
///
/// Returns errors if buffers are undersized.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_int4<T: GpuFloat>(
    handle: &DnnHandle,
    input: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<T>,
    zeros: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
    n: usize,
    config: &Int4QuantConfig,
) -> DnnResult<()> {
    if n == 0 {
        return Ok(());
    }

    let packed_bytes = config.packed_bytes(n);
    let num_groups = config.num_groups(n);

    if input.len() < packed_bytes {
        return Err(DnnError::BufferTooSmall {
            expected: packed_bytes,
            actual: input.len(),
        });
    }
    if scales.len() < num_groups {
        return Err(DnnError::BufferTooSmall {
            expected: num_groups,
            actual: scales.len(),
        });
    }
    if output.len() < n {
        return Err(DnnError::BufferTooSmall {
            expected: n,
            actual: output.len(),
        });
    }

    let ptx = generate_int4_unpack_ptx::<T>(handle.sm_version(), config)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let name = format!("dnn_int4_unpack_{}", T::NAME);
    let kernel = Kernel::from_module(module, &name)?;

    let grid = grid_size_for(packed_bytes as u32, INT4_QUANT_BLOCK);
    let params = LaunchParams::new(grid, INT4_QUANT_BLOCK);
    let args = (
        input.as_device_ptr(),
        scales.as_device_ptr(),
        zeros.as_device_ptr(),
        output.as_device_ptr(),
        n as u32,
        config.group_size as u32,
        packed_bytes as u32,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("INT4 unpack: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// NF4 quantization
// ---------------------------------------------------------------------------

/// Quantizes floating-point data to NF4 (NormalFloat4), packed two values
/// per byte.
///
/// NF4 uses a fixed lookup table of 16 values placed at normal distribution
/// quantiles. Each input value is mapped to the nearest NF4 codeword. This
/// provides information-theoretically optimal 4-bit encoding for weights
/// that follow a normal distribution (common in pretrained LLMs).
///
/// Scales are computed per-group: `scale = absmax(group)`. Input is
/// normalized to `[-1, 1]` before NF4 mapping.
///
/// # Errors
///
/// Returns errors if buffers are undersized or PTX generation fails.
pub fn quantize_to_nf4<T: GpuFloat>(
    handle: &DnnHandle,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<T>,
    n: usize,
    group_size: usize,
) -> DnnResult<()> {
    if n == 0 {
        return Ok(());
    }
    if group_size == 0 {
        return Err(DnnError::InvalidArgument(
            "NF4 group_size must be non-zero".into(),
        ));
    }

    let num_groups = n.div_ceil(group_size);
    let packed_bytes = n.div_ceil(2);

    if input.len() < n {
        return Err(DnnError::BufferTooSmall {
            expected: n,
            actual: input.len(),
        });
    }
    if output.len() < packed_bytes {
        return Err(DnnError::BufferTooSmall {
            expected: packed_bytes,
            actual: output.len(),
        });
    }
    if scales.len() < num_groups {
        return Err(DnnError::BufferTooSmall {
            expected: num_groups,
            actual: scales.len(),
        });
    }

    // Step 1: Compute per-group absmax scales
    let sym_config = Int4QuantConfig {
        group_size,
        symmetric: true,
    };
    let scale_ptx = generate_int4_scale_ptx::<T>(handle.sm_version(), &sym_config)?;
    let scale_mod = Arc::new(Module::from_ptx(&scale_ptx)?);
    let scale_name = format!("dnn_int4_scale_{}", T::NAME);
    let scale_kernel = Kernel::from_module(scale_mod, &scale_name)?;

    // For NF4, zeros buffer is unused (symmetric), but we need a dummy pointer
    let dummy_zeros = DeviceBuffer::<T>::alloc(1)?;
    let scale_grid = grid_size_for(num_groups as u32, INT4_QUANT_BLOCK);
    let scale_params = LaunchParams::new(scale_grid, INT4_QUANT_BLOCK);
    let scale_args = (
        input.as_device_ptr(),
        scales.as_device_ptr(),
        dummy_zeros.as_device_ptr(),
        n as u32,
        group_size as u32,
        num_groups as u32,
    );

    scale_kernel
        .launch(&scale_params, handle.stream(), &scale_args)
        .map_err(|e| DnnError::LaunchFailed(format!("NF4 scale compute: {e}")))?;

    // Step 2: NF4 quantize and pack
    let nf4_ptx = generate_nf4_pack_ptx::<T>(handle.sm_version())?;
    let nf4_mod = Arc::new(Module::from_ptx(&nf4_ptx)?);
    let nf4_name = format!("dnn_nf4_pack_{}", T::NAME);
    let nf4_kernel = Kernel::from_module(nf4_mod, &nf4_name)?;

    let nf4_grid = grid_size_for(packed_bytes as u32, INT4_QUANT_BLOCK);
    let nf4_params = LaunchParams::new(nf4_grid, INT4_QUANT_BLOCK);
    let nf4_args = (
        input.as_device_ptr(),
        scales.as_device_ptr(),
        output.as_device_ptr(),
        n as u32,
        group_size as u32,
        packed_bytes as u32,
    );

    nf4_kernel
        .launch(&nf4_params, handle.stream(), &nf4_args)
        .map_err(|e| DnnError::LaunchFailed(format!("NF4 pack: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// PTX generation: INT4 scale computation
// ---------------------------------------------------------------------------

/// Generates PTX for per-group scale (and optional zero-point) computation.
fn generate_int4_scale_ptx<T: GpuFloat>(
    sm: SmVersion,
    config: &Int4QuantConfig,
) -> DnnResult<String> {
    let kernel_name = format!("dnn_int4_scale_{}", T::NAME);
    let symmetric = config.symmetric;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .max_threads_per_block(INT4_QUANT_BLOCK)
        .param("in_ptr", PtxType::U64)
        .param("scale_ptr", PtxType::U64)
        .param("zero_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("group_size", PtxType::U32)
        .param("num_groups", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let num_groups_reg = b.load_param_u32("num_groups");

            b.if_lt_u32(gid.clone(), num_groups_reg, move |b| {
                let in_ptr = b.load_param_u64("in_ptr");
                let scale_ptr = b.load_param_u64("scale_ptr");
                let n_reg = b.load_param_u32("n");
                let group_size_reg = b.load_param_u32("group_size");

                // group_start = gid * group_size
                let group_start = b.mul_lo_u32(gid.clone(), group_size_reg.clone());
                // group_end = min(group_start + group_size, n)
                let group_end_raw = b.add_u32(group_start.clone(), group_size_reg);
                let p_end = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.gt.u32 {p_end}, {group_end_raw}, {n_reg};"));
                let group_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "selp.u32 {group_end}, {n_reg}, {group_end_raw}, {p_end};"
                ));

                if symmetric {
                    // Symmetric: find absmax, scale = absmax / 8
                    let max_val = load_float_imm::<T>(b, 0.0);
                    let i_reg = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {i_reg}, {group_start};"));

                    let loop_lbl = b.fresh_label("i4s_loop");
                    let end_lbl = b.fresh_label("i4s_end");
                    b.label(&loop_lbl);
                    let p_i = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.ge.u32 {p_i}, {i_reg}, {group_end};"));
                    b.branch_if(p_i, &end_lbl);

                    let addr = b.byte_offset_addr(in_ptr.clone(), i_reg.clone(), T::size_u32());
                    let val = load_global_float::<T>(b, addr);
                    let abs_val = abs_float::<T>(b, val);
                    let new_max = max_float::<T>(b, max_val.clone(), abs_val);
                    b.raw_ptx(&format!(
                        "mov.{} {max_val}, {new_max};",
                        T::PTX_TYPE.as_ptx_str().trim_start_matches('.')
                    ));

                    b.raw_ptx(&format!("add.u32 {i_reg}, {i_reg}, 1;"));
                    b.branch(&loop_lbl);
                    b.label(&end_lbl);

                    // scale = absmax / 8.0 (INT4 symmetric range is [-8, 7])
                    let eight = load_float_imm::<T>(b, INT4_SYM_MAX.abs());
                    let eps = load_float_imm::<T>(b, 1e-12);
                    let safe_max = max_float::<T>(b, max_val, eps);
                    let scale = div_float::<T>(b, safe_max, eight);

                    let scale_addr = b.byte_offset_addr(scale_ptr, gid, T::size_u32());
                    store_global_float::<T>(b, scale_addr, scale);
                } else {
                    // Asymmetric: find min and max, scale = (max - min) / 15
                    let min_val = load_float_imm::<T>(b, f64::MAX);
                    let max_val = load_float_imm::<T>(b, f64::MIN);
                    let i_reg = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {i_reg}, {group_start};"));

                    let loop_lbl = b.fresh_label("i4a_loop");
                    let end_lbl = b.fresh_label("i4a_end");
                    b.label(&loop_lbl);
                    let p_i = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.ge.u32 {p_i}, {i_reg}, {group_end};"));
                    b.branch_if(p_i, &end_lbl);

                    let addr = b.byte_offset_addr(in_ptr.clone(), i_reg.clone(), T::size_u32());
                    let val = load_global_float::<T>(b, addr);

                    // min
                    let p_lt = setp_gt_float::<T>(b, min_val.clone(), val.clone());
                    let new_min = selp_float::<T>(b, val.clone(), min_val.clone(), p_lt);
                    b.raw_ptx(&format!(
                        "mov.{} {min_val}, {new_min};",
                        T::PTX_TYPE.as_ptx_str().trim_start_matches('.')
                    ));

                    // max
                    let new_max = max_float::<T>(b, max_val.clone(), val);
                    b.raw_ptx(&format!(
                        "mov.{} {max_val}, {new_max};",
                        T::PTX_TYPE.as_ptx_str().trim_start_matches('.')
                    ));

                    b.raw_ptx(&format!("add.u32 {i_reg}, {i_reg}, 1;"));
                    b.branch(&loop_lbl);
                    b.label(&end_lbl);

                    // scale = (max - min) / 15.0
                    let neg_one_a = load_float_imm::<T>(b, -1.0);
                    let neg_min = mul_float::<T>(b, min_val.clone(), neg_one_a);
                    let range = add_float::<T>(b, max_val, neg_min);
                    let fifteen = load_float_imm::<T>(b, INT4_ASYM_MAX);
                    let eps = load_float_imm::<T>(b, 1e-12);
                    let safe_range = max_float::<T>(b, range, eps);
                    let scale = div_float::<T>(b, safe_range, fifteen);

                    let scale_addr = b.byte_offset_addr(scale_ptr, gid.clone(), T::size_u32());
                    store_global_float::<T>(b, scale_addr, scale.clone());

                    // zero = min / scale (the zero point)
                    let zero_ptr = b.load_param_u64("zero_ptr");
                    let neg_one_b = load_float_imm::<T>(b, -1.0);
                    let neg_min2 = mul_float::<T>(b, min_val, neg_one_b);
                    let zero = div_float::<T>(b, neg_min2, scale);
                    let zero_addr = b.byte_offset_addr(zero_ptr, gid, T::size_u32());
                    store_global_float::<T>(b, zero_addr, zero);
                }
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("INT4 scale: {e}")))?;

    Ok(ptx)
}

/// Generates PTX for INT4 quantization + packing kernel.
fn generate_int4_pack_ptx<T: GpuFloat>(
    sm: SmVersion,
    config: &Int4QuantConfig,
) -> DnnResult<String> {
    let kernel_name = format!("dnn_int4_pack_{}", T::NAME);
    let symmetric = config.symmetric;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .max_threads_per_block(INT4_QUANT_BLOCK)
        .param("in_ptr", PtxType::U64)
        .param("scale_ptr", PtxType::U64)
        .param("zero_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("group_size", PtxType::U32)
        .param("packed_n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let packed_n = b.load_param_u32("packed_n");

            b.if_lt_u32(gid.clone(), packed_n, move |b| {
                let in_ptr = b.load_param_u64("in_ptr");
                let scale_ptr = b.load_param_u64("scale_ptr");
                let n_reg = b.load_param_u32("n");
                let group_size_reg = b.load_param_u32("group_size");

                // Each thread processes 2 elements: elem_idx = gid * 2
                let elem_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shl.b32 {elem_idx}, {gid}, 1;"));

                // Process low nibble (even element)
                let low_nibble = quantize_one_int4::<T>(
                    b,
                    &in_ptr,
                    &scale_ptr,
                    &elem_idx,
                    &n_reg,
                    &group_size_reg,
                    symmetric,
                );

                // Process high nibble (odd element)
                let one_u32_a = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {one_u32_a}, 1;"));
                let odd_idx = b.add_u32(elem_idx, one_u32_a);
                let high_nibble = quantize_one_int4::<T>(
                    b,
                    &in_ptr,
                    &scale_ptr,
                    &odd_idx,
                    &n_reg,
                    &group_size_reg,
                    symmetric,
                );

                // Pack: result = (high << 4) | low
                let shifted_high = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shl.b32 {shifted_high}, {high_nibble}, 4;"));
                let packed = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("or.b32 {packed}, {shifted_high}, {low_nibble};"));

                // Store as u8
                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, 1u32);
                b.raw_ptx(&format!("st.global.u8 [{out_addr}], {packed};"));
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("INT4 pack: {e}")))?;

    Ok(ptx)
}

/// Helper: quantizes one element to a 4-bit integer (returned as u32 in [0,15]).
fn quantize_one_int4<T: GpuFloat>(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    in_ptr: &oxicuda_ptx::ir::Register,
    scale_ptr: &oxicuda_ptx::ir::Register,
    elem_idx: &oxicuda_ptx::ir::Register,
    n_reg: &oxicuda_ptx::ir::Register,
    group_size_reg: &oxicuda_ptx::ir::Register,
    symmetric: bool,
) -> oxicuda_ptx::ir::Register {
    // Check bounds
    let p_oob = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ge.u32 {p_oob}, {elem_idx}, {n_reg};"));

    // Default value for out-of-bounds
    let result = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {result}, 0;"));

    let skip_lbl = b.fresh_label("i4_skip");
    b.branch_if(p_oob, &skip_lbl);

    // group_idx = elem_idx / group_size
    let group_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "div.u32 {group_idx}, {elem_idx}, {group_size_reg};"
    ));

    // Load scale
    let scale_addr = b.byte_offset_addr(scale_ptr.clone(), group_idx, T::size_u32());
    let scale = load_global_float::<T>(b, scale_addr);

    // Load value
    let val_addr = b.byte_offset_addr(in_ptr.clone(), elem_idx.clone(), T::size_u32());
    let val = load_global_float::<T>(b, val_addr);

    // Quantize: q = round(val / scale) for symmetric
    let scaled = div_float::<T>(b, val, scale);

    if symmetric {
        // Clamp to [-8, 7] then add 8 to get [0, 15]
        let min_v = load_float_imm::<T>(b, INT4_SYM_MIN);
        let max_v = load_float_imm::<T>(b, INT4_SYM_MAX);
        let clamped = max_float::<T>(b, scaled, min_v);
        let p_gt = setp_gt_float::<T>(b, clamped.clone(), max_v.clone());
        let clamped2 = selp_float::<T>(b, max_v, clamped, p_gt);

        // Round and convert to u32
        let eight = load_float_imm::<T>(b, 8.0);
        let shifted = add_float::<T>(b, clamped2, eight);
        let as_u32 = cvt_float_to_u32::<T>(b, shifted);
        // Clamp to [0, 15]
        let p_over = b.alloc_reg(PtxType::Pred);
        b.raw_ptx(&format!("setp.gt.u32 {p_over}, {as_u32}, 15;"));
        let fifteen = {
            let fifteen_r = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {fifteen_r}, 15;"));
            fifteen_r
        };
        b.raw_ptx(&format!(
            "selp.u32 {result}, {fifteen}, {as_u32}, {p_over};"
        ));
    } else {
        // Asymmetric: q = round(val / scale + zero), clamp to [0, 15]
        let max_v = load_float_imm::<T>(b, INT4_ASYM_MAX);
        let zero_v = load_float_imm::<T>(b, 0.0);
        let clamped = max_float::<T>(b, scaled, zero_v);
        let p_gt = setp_gt_float::<T>(b, clamped.clone(), max_v.clone());
        let clamped2 = selp_float::<T>(b, max_v, clamped, p_gt);
        let as_u32 = cvt_float_to_u32::<T>(b, clamped2);
        let p_over = b.alloc_reg(PtxType::Pred);
        b.raw_ptx(&format!("setp.gt.u32 {p_over}, {as_u32}, 15;"));
        let fifteen = {
            let fifteen_r = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {fifteen_r}, 15;"));
            fifteen_r
        };
        b.raw_ptx(&format!(
            "selp.u32 {result}, {fifteen}, {as_u32}, {p_over};"
        ));
    }

    b.label(&skip_lbl);
    result
}

/// Generates PTX for INT4 dequantization (unpack + dequant).
fn generate_int4_unpack_ptx<T: GpuFloat>(
    sm: SmVersion,
    config: &Int4QuantConfig,
) -> DnnResult<String> {
    let kernel_name = format!("dnn_int4_unpack_{}", T::NAME);
    let symmetric = config.symmetric;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .max_threads_per_block(INT4_QUANT_BLOCK)
        .param("in_ptr", PtxType::U64)
        .param("scale_ptr", PtxType::U64)
        .param("zero_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("group_size", PtxType::U32)
        .param("packed_n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let packed_n = b.load_param_u32("packed_n");

            b.if_lt_u32(gid.clone(), packed_n, move |b| {
                let in_ptr = b.load_param_u64("in_ptr");
                let scale_ptr = b.load_param_u64("scale_ptr");
                let out_ptr = b.load_param_u64("out_ptr");
                let n_reg = b.load_param_u32("n");
                let group_size_reg = b.load_param_u32("group_size");

                // Load packed byte
                let in_addr = b.byte_offset_addr(in_ptr, gid.clone(), 1u32);
                let packed = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("ld.global.u8 {packed}, [{in_addr}];"));

                // Extract nibbles
                let low = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("and.b32 {low}, {packed}, 15;"));
                let high = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shr.u32 {high}, {packed}, 4;"));
                b.raw_ptx(&format!("and.b32 {high}, {high}, 15;"));

                // Element indices
                let even_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shl.b32 {even_idx}, {gid}, 1;"));
                let one_u32_b = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {one_u32_b}, 1;"));
                let odd_idx = b.add_u32(even_idx.clone(), one_u32_b);

                // Dequantize and store even element
                dequant_and_store_int4::<T>(
                    b,
                    &low,
                    &even_idx,
                    &n_reg,
                    &scale_ptr,
                    &out_ptr,
                    &group_size_reg,
                    symmetric,
                );

                // Dequantize and store odd element
                dequant_and_store_int4::<T>(
                    b,
                    &high,
                    &odd_idx,
                    &n_reg,
                    &scale_ptr,
                    &out_ptr,
                    &group_size_reg,
                    symmetric,
                );
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("INT4 unpack: {e}")))?;

    Ok(ptx)
}

/// Helper: dequantize a single INT4 nibble and store result.
#[allow(clippy::too_many_arguments)]
fn dequant_and_store_int4<T: GpuFloat>(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    nibble: &oxicuda_ptx::ir::Register,
    elem_idx: &oxicuda_ptx::ir::Register,
    n_reg: &oxicuda_ptx::ir::Register,
    scale_ptr: &oxicuda_ptx::ir::Register,
    out_ptr: &oxicuda_ptx::ir::Register,
    group_size_reg: &oxicuda_ptx::ir::Register,
    symmetric: bool,
) {
    let p_oob = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ge.u32 {p_oob}, {elem_idx}, {n_reg};"));
    let skip_lbl = b.fresh_label("i4d_skip");
    b.branch_if(p_oob, &skip_lbl);

    let group_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "div.u32 {group_idx}, {elem_idx}, {group_size_reg};"
    ));

    let scale_addr = b.byte_offset_addr(scale_ptr.clone(), group_idx, T::size_u32());
    let scale = load_global_float::<T>(b, scale_addr);

    let float_val = cvt_u32_to_float::<T>(b, nibble.clone());

    let result = if symmetric {
        // val = (nibble - 8) * scale
        let eight = load_float_imm::<T>(b, 8.0);
        let neg_one_imm = load_float_imm::<T>(b, -1.0);
        let neg_eight = mul_float::<T>(b, eight, neg_one_imm);
        let shifted = add_float::<T>(b, float_val, neg_eight);
        mul_float::<T>(b, shifted, scale)
    } else {
        // val = nibble * scale (zero point already baked into the packed value)
        mul_float::<T>(b, float_val, scale)
    };

    let out_addr = b.byte_offset_addr(out_ptr.clone(), elem_idx.clone(), T::size_u32());
    store_global_float::<T>(b, out_addr, result);

    b.label(&skip_lbl);
}

// ---------------------------------------------------------------------------
// PTX generation: NF4
// ---------------------------------------------------------------------------

/// Generates PTX for NF4 quantization + packing.
///
/// Each thread processes two elements: normalize by scale, find the nearest
/// NF4 codeword via linear scan of the 16-entry lookup table, pack two
/// 4-bit indices per byte.
fn generate_nf4_pack_ptx<T: GpuFloat>(sm: SmVersion) -> DnnResult<String> {
    let kernel_name = format!("dnn_nf4_pack_{}", T::NAME);

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .max_threads_per_block(INT4_QUANT_BLOCK)
        .param("in_ptr", PtxType::U64)
        .param("scale_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("group_size", PtxType::U32)
        .param("packed_n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let packed_n = b.load_param_u32("packed_n");

            b.if_lt_u32(gid.clone(), packed_n, move |b| {
                let in_ptr = b.load_param_u64("in_ptr");
                let scale_ptr = b.load_param_u64("scale_ptr");
                let n_reg = b.load_param_u32("n");
                let group_size_reg = b.load_param_u32("group_size");

                let even_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shl.b32 {even_idx}, {gid}, 1;"));
                let one_u32_c = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {one_u32_c}, 1;"));
                let odd_idx = b.add_u32(even_idx.clone(), one_u32_c);

                // Quantize even element to NF4 code
                let low_code = nf4_quantize_one::<T>(
                    b,
                    &in_ptr,
                    &scale_ptr,
                    &even_idx,
                    &n_reg,
                    &group_size_reg,
                );

                // Quantize odd element to NF4 code
                let high_code = nf4_quantize_one::<T>(
                    b,
                    &in_ptr,
                    &scale_ptr,
                    &odd_idx,
                    &n_reg,
                    &group_size_reg,
                );

                // Pack
                let shifted = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shl.b32 {shifted}, {high_code}, 4;"));
                let packed = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("or.b32 {packed}, {shifted}, {low_code};"));

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.byte_offset_addr(out_ptr, gid, 1u32);
                b.raw_ptx(&format!("st.global.u8 [{out_addr}], {packed};"));
            });

            b.ret();
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(format!("NF4 pack: {e}")))?;

    Ok(ptx)
}

/// Quantizes a single element to its nearest NF4 code (0-15).
fn nf4_quantize_one<T: GpuFloat>(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    in_ptr: &oxicuda_ptx::ir::Register,
    scale_ptr: &oxicuda_ptx::ir::Register,
    elem_idx: &oxicuda_ptx::ir::Register,
    n_reg: &oxicuda_ptx::ir::Register,
    group_size_reg: &oxicuda_ptx::ir::Register,
) -> oxicuda_ptx::ir::Register {
    let result = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {result}, 0;"));

    let p_oob = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ge.u32 {p_oob}, {elem_idx}, {n_reg};"));
    let skip_lbl = b.fresh_label("nf4_skip");
    b.branch_if(p_oob, &skip_lbl);

    // group_idx = elem_idx / group_size
    let group_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "div.u32 {group_idx}, {elem_idx}, {group_size_reg};"
    ));

    let scale_addr = b.byte_offset_addr(scale_ptr.clone(), group_idx, T::size_u32());
    let scale = load_global_float::<T>(b, scale_addr);
    let eps = load_float_imm::<T>(b, 1e-12);
    let safe_scale = max_float::<T>(b, scale, eps);

    let val_addr = b.byte_offset_addr(in_ptr.clone(), elem_idx.clone(), T::size_u32());
    let val = load_global_float::<T>(b, val_addr);

    // Normalize to [-1, 1]
    let normalized = div_float::<T>(b, val, safe_scale);

    // Find nearest NF4 code via midpoint comparison
    // We use a binary-search-style approach with hardcoded thresholds
    // Midpoints between consecutive NF4 values:
    let midpoints: Vec<f64> = (0..15)
        .map(|i| (NF4_LOOKUP[i] + NF4_LOOKUP[i + 1]) / 2.0)
        .collect();

    // Linear scan: start with code=0, increment if normalized > midpoint[i]
    b.raw_ptx(&format!("mov.u32 {result}, 0;"));
    for (i, &mp) in midpoints.iter().enumerate() {
        let threshold = load_float_imm::<T>(b, mp);
        let p_gt = setp_gt_float::<T>(b, normalized.clone(), threshold);
        let next_code = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("mov.u32 {next_code}, {};", i + 1));
        b.raw_ptx(&format!(
            "selp.u32 {result}, {next_code}, {result}, {p_gt};"
        ));
    }

    b.label(&skip_lbl);
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int4_config_valid() {
        let cfg = Int4QuantConfig::new(32, true);
        assert!(cfg.is_ok());
    }

    #[test]
    fn int4_config_zero_group() {
        let cfg = Int4QuantConfig::new(0, true);
        assert!(cfg.is_err());
    }

    #[test]
    fn int4_num_groups() {
        let cfg = Int4QuantConfig::new(32, true).expect("valid");
        assert_eq!(cfg.num_groups(128), 4);
        assert_eq!(cfg.num_groups(129), 5);
        assert_eq!(cfg.num_groups(0), 0);
    }

    #[test]
    fn int4_packed_bytes() {
        let cfg = Int4QuantConfig::new(32, true).expect("valid");
        assert_eq!(cfg.packed_bytes(128), 64);
        assert_eq!(cfg.packed_bytes(129), 65);
        assert_eq!(cfg.packed_bytes(1), 1);
    }

    #[test]
    fn int4_scale_ptx_symmetric_f32() {
        let cfg = Int4QuantConfig::new(32, true).expect("valid");
        let ptx = generate_int4_scale_ptx::<f32>(SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("should generate");
        assert!(ptx_str.contains("dnn_int4_scale_f32"));
    }

    #[test]
    fn int4_scale_ptx_asymmetric_f32() {
        let cfg = Int4QuantConfig::new(128, false).expect("valid");
        let ptx = generate_int4_scale_ptx::<f32>(SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
    }

    #[test]
    fn int4_pack_ptx_f32() {
        let cfg = Int4QuantConfig::new(32, true).expect("valid");
        let ptx = generate_int4_pack_ptx::<f32>(SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("should generate");
        assert!(ptx_str.contains("dnn_int4_pack_f32"));
    }

    #[test]
    fn int4_unpack_ptx_f32() {
        let cfg = Int4QuantConfig::new(32, true).expect("valid");
        let ptx = generate_int4_unpack_ptx::<f32>(SmVersion::Sm80, &cfg);
        assert!(ptx.is_ok());
    }

    #[test]
    fn nf4_pack_ptx_f32() {
        let ptx = generate_nf4_pack_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("should generate");
        assert!(ptx_str.contains("dnn_nf4_pack_f32"));
    }

    #[test]
    fn nf4_lookup_table_sorted() {
        for i in 1..NF4_LOOKUP.len() {
            assert!(
                NF4_LOOKUP[i] > NF4_LOOKUP[i - 1],
                "NF4 lookup not sorted at index {i}"
            );
        }
    }

    #[test]
    fn nf4_lookup_table_range() {
        assert!((NF4_LOOKUP[0] - (-1.0)).abs() < 1e-10);
        assert!((NF4_LOOKUP[15] - 1.0).abs() < 1e-10);
    }
}
