//! Mixed-precision sparse matrix-vector multiplication (SpMV).
//!
//! Provides FP16/BF16 storage with FP32 accumulation for memory-bandwidth-bound
//! SpMV operations. This approach halves the memory footprint of matrix values
//! while maintaining FP32 numerical quality during the dot-product accumulation,
//! yielding up to 2x bandwidth savings on bandwidth-limited GPUs.
//!
//! Three kernel strategies are available:
//! - **Scalar**: one thread per row, FP16 load -> FP32 FMA
//! - **Vector**: one warp per row with FP32 shuffle reduction
//! - **VectorPacked**: loads two FP16 values per 32-bit memory transaction (2x bandwidth)
//! - **Auto**: auto-selects based on average nnz per row

use oxicuda_ptx::prelude::*;

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Storage precision for matrix values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoragePrecision {
    /// IEEE 754 half-precision (FP16, 10-bit mantissa).
    Fp16,
    /// Brain floating-point (BF16, 7-bit mantissa, wider exponent range).
    Bf16,
}

impl StoragePrecision {
    /// Returns the PTX type corresponding to this storage precision.
    #[must_use]
    pub const fn ptx_type(self) -> PtxType {
        match self {
            Self::Fp16 => PtxType::F16,
            Self::Bf16 => PtxType::BF16,
        }
    }

    /// Returns the packed (x2) PTX type for this precision.
    #[must_use]
    pub const fn packed_ptx_type(self) -> PtxType {
        match self {
            Self::Fp16 => PtxType::F16x2,
            Self::Bf16 => PtxType::BF16x2,
        }
    }

    /// Bytes per element.
    #[must_use]
    pub const fn element_bytes(self) -> u32 {
        2
    }

    /// Mantissa bits (for error analysis).
    #[must_use]
    pub const fn mantissa_bits(self) -> u32 {
        match self {
            Self::Fp16 => 10,
            Self::Bf16 => 7,
        }
    }

    /// Returns the PTX type suffix string without the leading dot.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Fp16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

/// Compute/accumulation precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputePrecision {
    /// IEEE 754 single-precision (32-bit).
    Fp32,
    /// IEEE 754 double-precision (64-bit).
    Fp64,
}

impl ComputePrecision {
    /// Returns the PTX type for the compute precision.
    #[must_use]
    pub const fn ptx_type(self) -> PtxType {
        match self {
            Self::Fp32 => PtxType::F32,
            Self::Fp64 => PtxType::F64,
        }
    }

    /// Bytes per element.
    #[must_use]
    pub const fn element_bytes(self) -> u32 {
        match self {
            Self::Fp32 => 4,
            Self::Fp64 => 8,
        }
    }

    /// Returns the PTX suffix without the dot.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Fp32 => "f32",
            Self::Fp64 => "f64",
        }
    }

    /// Returns the bit-move suffix for `mov.bN` instructions.
    #[must_use]
    pub const fn bit_suffix(self) -> &'static str {
        match self {
            Self::Fp32 => "b32",
            Self::Fp64 => "b64",
        }
    }
}

/// Algorithm selection for mixed-precision SpMV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MixedSpMVAlgo {
    /// One thread per row. Best for very sparse matrices (< 4 nnz/row).
    Scalar,
    /// One warp (32 threads) per row with FP32 warp shuffle reduction.
    Vector,
    /// Vector algorithm with packed 2xFP16 loads (doubles effective bandwidth).
    /// Each 32-bit load fetches two FP16 values simultaneously.
    VectorPacked,
    /// Automatically selects the best algorithm based on matrix structure.
    Auto,
}

/// Configuration for mixed-precision SpMV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MixedPrecisionConfig {
    /// Storage precision for matrix values (FP16 or BF16).
    pub storage_precision: StoragePrecision,
    /// Compute/accumulation precision (FP32 for now).
    pub compute_precision: ComputePrecision,
    /// Algorithm selection.
    pub algorithm: MixedSpMVAlgo,
    /// Target GPU architecture.
    pub sm_version: SmVersion,
}

impl MixedPrecisionConfig {
    /// Creates a new configuration with FP16 storage and FP32 accumulation.
    #[must_use]
    pub const fn fp16_fp32(algo: MixedSpMVAlgo, sm: SmVersion) -> Self {
        Self {
            storage_precision: StoragePrecision::Fp16,
            compute_precision: ComputePrecision::Fp32,
            algorithm: algo,
            sm_version: sm,
        }
    }

    /// Creates a new configuration with BF16 storage and FP32 accumulation.
    #[must_use]
    pub const fn bf16_fp32(algo: MixedSpMVAlgo, sm: SmVersion) -> Self {
        Self {
            storage_precision: StoragePrecision::Bf16,
            compute_precision: ComputePrecision::Fp32,
            algorithm: algo,
            sm_version: sm,
        }
    }
}

// ---------------------------------------------------------------------------
// Execution plan
// ---------------------------------------------------------------------------

/// Pre-computed execution plan for mixed-precision SpMV.
///
/// Contains the resolved configuration, estimated bandwidth savings, and
/// memory footprint information needed to launch the kernel.
#[derive(Debug, Clone)]
pub struct MixedPrecisionPlan {
    /// Resolved configuration (Auto replaced with concrete algorithm).
    pub config: MixedPrecisionConfig,
    /// Number of matrix rows.
    pub rows: u32,
    /// Number of matrix columns.
    pub cols: u32,
    /// Number of non-zero entries.
    pub nnz: u64,
    /// Memory footprint in bytes for the FP16/BF16 values array.
    pub values_bytes: u64,
    /// Memory footprint in bytes for the same values if stored in compute precision.
    pub values_bytes_full: u64,
}

impl MixedPrecisionPlan {
    /// Returns the bandwidth savings ratio (e.g., ~2.0 for FP16 vs FP32 values).
    ///
    /// This only accounts for the values array; row_offsets and col_indices remain
    /// unchanged as i32 arrays.
    #[must_use]
    pub fn bandwidth_savings_ratio(&self) -> f64 {
        if self.values_bytes == 0 {
            return 1.0;
        }
        self.values_bytes_full as f64 / self.values_bytes as f64
    }

    /// Estimates peak achievable GFLOPS assuming the kernel is perfectly
    /// memory-bandwidth-bound.
    ///
    /// Uses a simple roofline model: each non-zero requires one FMA (2 flops),
    /// plus the memory traffic to load the value, column index, and vector element.
    #[must_use]
    pub fn estimated_gflops(&self, peak_bandwidth_gb_s: f64) -> f64 {
        if self.nnz == 0 {
            return 0.0;
        }
        // Bytes per non-zero: storage_bytes (value) + 4 (col_idx) + compute_bytes (x[col])
        let bytes_per_nnz = self.config.storage_precision.element_bytes() as f64
            + 4.0
            + self.config.compute_precision.element_bytes() as f64;
        let total_bytes = bytes_per_nnz * self.nnz as f64;
        // 2 flops per nnz (one multiply, one add in FMA)
        let total_flops = 2.0 * self.nnz as f64;
        // bandwidth in bytes/s
        let bandwidth_bytes_s = peak_bandwidth_gb_s * 1e9;
        // time = bytes / bandwidth
        let time_s = total_bytes / bandwidth_bytes_s;
        // GFLOPS = flops / time / 1e9
        total_flops / time_s / 1e9
    }

    /// Returns the average non-zeros per row.
    #[must_use]
    pub fn avg_nnz_per_row(&self) -> f64 {
        if self.rows == 0 {
            return 0.0;
        }
        self.nnz as f64 / self.rows as f64
    }
}

/// Performance statistics from a mixed-precision SpMV execution.
#[derive(Debug, Clone)]
pub struct MixedPrecisionStats {
    /// Elapsed wall-clock time in microseconds.
    pub elapsed_us: f64,
    /// Achieved GFLOPS (2 * nnz / elapsed).
    pub gflops: f64,
    /// Achieved bandwidth in GB/s.
    pub bandwidth_gb_s: f64,
    /// Estimated relative precision loss (upper bound).
    pub precision_loss_bound: f64,
}

// ---------------------------------------------------------------------------
// Threshold for auto algorithm selection
// ---------------------------------------------------------------------------

/// Average nnz/row threshold above which Vector is preferred over Scalar.
const VECTOR_THRESHOLD: f64 = 4.0;

/// Average nnz/row threshold above which VectorPacked is preferred over Vector.
const PACKED_THRESHOLD: f64 = 32.0;

/// Block size for scalar kernels.
const SCALAR_BLOCK_SIZE: u32 = 256;

/// Block size for vector/packed kernels (must be multiple of 32).
const VECTOR_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// Plan creation
// ---------------------------------------------------------------------------

/// Creates an execution plan for mixed-precision SpMV.
///
/// Resolves the `Auto` algorithm based on the matrix dimensions, validates
/// the configuration, and computes memory/bandwidth estimates.
///
/// # Errors
///
/// Returns [`SparseError::InvalidArgument`] if the configuration is invalid
/// (e.g., BF16 on unsupported architecture, zero dimensions).
pub fn plan_mixed_precision_spmv(
    config: &MixedPrecisionConfig,
    rows: u32,
    cols: u32,
    nnz: u64,
) -> SparseResult<MixedPrecisionPlan> {
    validate_mixed_precision_config(config)?;

    let avg_nnz = if rows > 0 {
        nnz as f64 / rows as f64
    } else {
        0.0
    };

    // Resolve Auto algorithm
    let resolved_algo = match config.algorithm {
        MixedSpMVAlgo::Auto => {
            if avg_nnz >= PACKED_THRESHOLD {
                MixedSpMVAlgo::VectorPacked
            } else if avg_nnz >= VECTOR_THRESHOLD {
                MixedSpMVAlgo::Vector
            } else {
                MixedSpMVAlgo::Scalar
            }
        }
        other => other,
    };

    let resolved_config = MixedPrecisionConfig {
        algorithm: resolved_algo,
        ..*config
    };

    let values_bytes = nnz * config.storage_precision.element_bytes() as u64;
    let values_bytes_full = nnz * config.compute_precision.element_bytes() as u64;

    Ok(MixedPrecisionPlan {
        config: resolved_config,
        rows,
        cols,
        nnz,
        values_bytes,
        values_bytes_full,
    })
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

/// Validates a mixed-precision configuration.
///
/// Checks that BF16 is only used on architectures that support it (sm_80+)
/// and that the compute precision is compatible.
///
/// # Errors
///
/// Returns [`SparseError::InvalidArgument`] if the config is invalid.
pub fn validate_mixed_precision_config(config: &MixedPrecisionConfig) -> SparseResult<()> {
    // BF16 requires Ampere (sm_80) or newer
    if config.storage_precision == StoragePrecision::Bf16 {
        let (major, _minor) = config.sm_version.ptx_isa_version();
        if major < 7 {
            return Err(SparseError::InvalidArgument(
                "BF16 storage requires sm_80 (Ampere) or newer; \
                 the selected SM version does not support BF16 instructions"
                    .to_string(),
            ));
        }
    }

    // FP64 compute is only meaningful with FP16 storage (BF16 range can overflow in F32
    // but F64 compute with BF16 is valid)
    if config.compute_precision == ComputePrecision::Fp64 {
        // FP64 compute is supported but rarely beneficial for SpMV on modern GPUs.
        // Allow it but warn via documentation that throughput may be low.
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Precision loss estimation
// ---------------------------------------------------------------------------

/// Estimates the theoretical relative error bound for mixed-precision SpMV.
///
/// For a dot product of length `n` with FP16 storage and FP32 accumulation,
/// the relative error is bounded by:
///
///   `n * eps_storage + n * eps_compute`
///
/// where `eps_storage = 2^{-(p+1)}` for `p` mantissa bits, and
/// `eps_compute = 2^{-24}` for FP32.
///
/// This is a pessimistic (worst-case) bound; typical error is much smaller.
#[must_use]
pub fn estimate_precision_loss(nnz_per_row: f64, storage: StoragePrecision) -> f64 {
    let eps_storage = match storage {
        StoragePrecision::Fp16 => f64::powi(2.0, -11), // 2^-11 for 10-bit mantissa
        StoragePrecision::Bf16 => f64::powi(2.0, -8),  // 2^-8 for 7-bit mantissa
    };
    let eps_compute: f64 = f64::powi(2.0, -24); // FP32

    // Standard floating-point error accumulation bound for length-n dot product
    nnz_per_row * eps_storage + nnz_per_row * eps_compute
}

// ---------------------------------------------------------------------------
// PTX Generation: Scalar kernel
// ---------------------------------------------------------------------------

/// Generates PTX for scalar mixed-precision SpMV (one thread per row).
///
/// Each thread loads FP16/BF16 values from global memory, converts them
/// to FP32, performs FMA accumulation in FP32, and writes the FP32 result.
///
/// Kernel signature:
/// ```text
/// .entry mixed_spmv_scalar(
///     .param .u64 row_ptr,     // i32 CSR row offsets
///     .param .u64 col_idx,     // i32 column indices
///     .param .u64 values,      // FP16/BF16 values
///     .param .u64 x_ptr,       // FP32 dense vector x
///     .param .u64 y_ptr,       // FP32 dense vector y
///     .param .u32 alpha_bits,  // FP32 alpha as u32 bits
///     .param .u32 beta_bits,   // FP32 beta as u32 bits
///     .param .u32 num_rows
/// )
/// ```
///
/// # Errors
///
/// Returns [`PtxGenError`] if kernel assembly fails.
pub fn generate_mixed_scalar_spmv_ptx(
    config: &MixedPrecisionConfig,
) -> Result<String, PtxGenError> {
    let storage = config.storage_precision;
    let compute = config.compute_precision;
    let sm = config.sm_version;
    let storage_suffix = storage.suffix();
    let compute_suffix = compute.suffix();
    let compute_bit = compute.bit_suffix();
    let elem_bytes = storage.element_bytes();

    KernelBuilder::new("mixed_spmv_scalar")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U32)
        .param("beta_bits", PtxType::U32)
        .param("num_rows", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, num_rows, move |b| {
                let row = gid_inner;
                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");

                // Load alpha/beta as FP32 from bit patterns
                let alpha_bits = b.load_param_u32("alpha_bits");
                let alpha = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {alpha}, {alpha_bits};"));

                let beta_bits = b.load_param_u32("beta_bits");
                let beta = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {beta}, {beta_bits};"));

                // Load row_ptr[row] and row_ptr[row+1]
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), row.clone(), 4);
                let row_start = b.load_global_i32(rp_addr);

                let row_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_plus_1}, {row}, 1;"));
                let rp_addr_next = b.byte_offset_addr(row_ptr_base, row_plus_1, 4);
                let row_end = b.load_global_i32(rp_addr_next);

                // Initialize FP32 accumulator to 0
                let acc = b.alloc_reg(compute.ptx_type());
                let zero_bits: u32 = 0u32;
                b.raw_ptx(&format!("mov.{compute_bit} {acc}, 0F{zero_bits:08X};"));

                // Loop setup
                let loop_label = b.fresh_label("mpspmv_loop");
                let done_label = b.fresh_label("mpspmv_done");

                let k = b.alloc_reg(PtxType::U32);
                let rs_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {rs_u32}, {row_start};"));
                b.raw_ptx(&format!("mov.u32 {k}, {rs_u32};"));

                let re_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {re_u32}, {row_end};"));

                b.label(&loop_label);
                let pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {pred}, {k}, {re_u32};"));
                b.raw_ptx(&format!("@!{pred} bra {done_label};"));

                // Load col_idx[k]
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), k.clone(), 4);
                let col = b.load_global_i32(ci_addr);
                let col_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_u32}, {col};"));

                // Load FP16/BF16 value and convert to FP32
                let v_addr = b.byte_offset_addr(values_base.clone(), k.clone(), elem_bytes);
                let val_half = b.alloc_reg(storage.ptx_type());
                b.raw_ptx(&format!(
                    "ld.global.{storage_suffix} {val_half}, [{v_addr}];"
                ));
                let val_fp32 = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "cvt.{compute_suffix}.{storage_suffix} {val_fp32}, {val_half};"
                ));

                // Load x[col] (already FP32)
                let x_addr = b.byte_offset_addr(x_ptr.clone(), col_u32, compute.element_bytes());
                let x_val = b.load_global_f32(x_addr);

                // FMA: acc += val_fp32 * x_val
                let new_acc = b.fma_f32(val_fp32, x_val, acc.clone());
                b.raw_ptx(&format!("mov.{compute_suffix} {acc}, {new_acc};"));

                // k++
                b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
                b.branch(&loop_label);
                b.label(&done_label);

                // y = alpha * acc + beta * y_old
                let y_addr = b.byte_offset_addr(y_ptr, row, compute.element_bytes());
                let y_old = b.load_global_f32(y_addr.clone());

                let alpha_acc = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {alpha_acc}, {alpha}, {acc};"
                ));

                let beta_y = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {beta_y}, {beta}, {y_old};"
                ));

                let result = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "add.{compute_suffix} {result}, {alpha_acc}, {beta_y};"
                ));

                b.store_global_f32(y_addr, result);
            });

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// PTX Generation: Vector kernel (warp-parallel)
// ---------------------------------------------------------------------------

/// Generates PTX for vector mixed-precision SpMV (one warp per row).
///
/// Each warp cooperatively processes one row. Lanes stride over the non-zeros,
/// load FP16/BF16, convert to FP32, accumulate, and finally reduce via warp
/// shuffles. Lane 0 writes the final result.
///
/// # Errors
///
/// Returns [`PtxGenError`] if kernel assembly fails.
pub fn generate_mixed_vector_spmv_ptx(
    config: &MixedPrecisionConfig,
) -> Result<String, PtxGenError> {
    let storage = config.storage_precision;
    let compute = config.compute_precision;
    let sm = config.sm_version;
    let storage_suffix = storage.suffix();
    let compute_suffix = compute.suffix();
    let compute_bit = compute.bit_suffix();
    let elem_bytes = storage.element_bytes();

    KernelBuilder::new("mixed_spmv_vector")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U32)
        .param("beta_bits", PtxType::U32)
        .param("num_rows", PtxType::U32)
        .body(move |b| {
            let tid_global = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");

            // Lane within warp
            let lane = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("and.b32 {lane}, {tid_global}, 31;"));

            // Warp ID
            let warp_id = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {warp_id}, {tid_global}, 5;"));

            let warp_id_inner = warp_id.clone();
            let lane_inner = lane.clone();
            b.if_lt_u32(warp_id, num_rows, move |b| {
                let row = warp_id_inner;
                let lane = lane_inner;

                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");

                let alpha_bits = b.load_param_u32("alpha_bits");
                let alpha = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {alpha}, {alpha_bits};"));

                let beta_bits = b.load_param_u32("beta_bits");
                let beta = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {beta}, {beta_bits};"));

                // Load row bounds
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), row.clone(), 4);
                let row_start_i32 = b.load_global_i32(rp_addr);
                let row_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_start}, {row_start_i32};"));

                let row_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_plus_1}, {row}, 1;"));
                let rp_addr_next = b.byte_offset_addr(row_ptr_base, row_plus_1, 4);
                let row_end_i32 = b.load_global_i32(rp_addr_next);
                let row_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_end}, {row_end_i32};"));

                // Each lane starts at row_start + lane, stride 32
                let acc = b.alloc_reg(compute.ptx_type());
                let zero_bits: u32 = 0u32;
                b.raw_ptx(&format!("mov.{compute_bit} {acc}, 0F{zero_bits:08X};"));

                let k = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {k}, {row_start}, {lane};"));

                let loop_label = b.fresh_label("mpspmv_vloop");
                let done_label = b.fresh_label("mpspmv_vdone");

                b.label(&loop_label);
                let pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {pred}, {k}, {row_end};"));
                b.raw_ptx(&format!("@!{pred} bra {done_label};"));

                // Load col and half-precision value
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), k.clone(), 4);
                let col_i32 = b.load_global_i32(ci_addr);
                let col_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_u32}, {col_i32};"));

                let v_addr = b.byte_offset_addr(values_base.clone(), k.clone(), elem_bytes);
                let val_half = b.alloc_reg(storage.ptx_type());
                b.raw_ptx(&format!("ld.global.{storage_suffix} {val_half}, [{v_addr}];"));
                let val_fp32 = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "cvt.{compute_suffix}.{storage_suffix} {val_fp32}, {val_half};"
                ));

                let x_addr = b.byte_offset_addr(
                    x_ptr.clone(),
                    col_u32,
                    compute.element_bytes(),
                );
                let x_val = b.load_global_f32(x_addr);

                // FMA accumulate
                let new_acc = b.fma_f32(val_fp32, x_val, acc.clone());
                b.raw_ptx(&format!("mov.{compute_suffix} {acc}, {new_acc};"));

                // k += 32
                b.raw_ptx(&format!("add.u32 {k}, {k}, 32;"));
                b.branch(&loop_label);
                b.label(&done_label);

                // Warp shuffle reduction in FP32
                let mut current = acc;
                for offset in [16u32, 8, 4, 2, 1] {
                    let shuffled = b.alloc_reg(compute.ptx_type());
                    b.raw_ptx(&format!(
                        "shfl.sync.down.{compute_bit} {shuffled}, {current}, {offset}, 31, 0xFFFFFFFF;"
                    ));
                    let sum = b.alloc_reg(compute.ptx_type());
                    b.raw_ptx(&format!(
                        "add.{compute_suffix} {sum}, {current}, {shuffled};"
                    ));
                    current = sum;
                }

                // Lane 0 writes result
                let is_lane_0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {is_lane_0}, {lane}, 0;"));

                let skip_label = b.fresh_label("mpspmv_skip");
                b.raw_ptx(&format!("@!{is_lane_0} bra {skip_label};"));

                let y_addr = b.byte_offset_addr(y_ptr, row, compute.element_bytes());
                let y_old = b.load_global_f32(y_addr.clone());

                let alpha_acc = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {alpha_acc}, {alpha}, {current};"
                ));
                let beta_y = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {beta_y}, {beta}, {y_old};"
                ));
                let result = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "add.{compute_suffix} {result}, {alpha_acc}, {beta_y};"
                ));

                b.store_global_f32(y_addr, result);

                b.label(&skip_label);
            });

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// PTX Generation: Packed Vector kernel (2xFP16 loads)
// ---------------------------------------------------------------------------

/// Generates PTX for packed vector mixed-precision SpMV.
///
/// This kernel loads two FP16 values per 32-bit memory transaction using
/// the `f16x2` type, effectively doubling the bandwidth utilization for
/// the values array. Each pair of FP16 values is unpacked, converted to
/// FP32, and accumulated separately.
///
/// The kernel handles odd-length rows by processing the last element
/// individually when the row has an odd number of non-zeros.
///
/// # Errors
///
/// Returns [`PtxGenError`] if kernel assembly fails.
pub fn generate_packed_vector_spmv_ptx(
    config: &MixedPrecisionConfig,
) -> Result<String, PtxGenError> {
    let storage = config.storage_precision;
    let compute = config.compute_precision;
    let sm = config.sm_version;
    let storage_suffix = storage.suffix();
    let compute_suffix = compute.suffix();
    let compute_bit = compute.bit_suffix();
    let elem_bytes = storage.element_bytes();

    KernelBuilder::new("mixed_spmv_packed")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U32)
        .param("beta_bits", PtxType::U32)
        .param("num_rows", PtxType::U32)
        .body(move |b| {
            let tid_global = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");

            let lane = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("and.b32 {lane}, {tid_global}, 31;"));

            let warp_id = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {warp_id}, {tid_global}, 5;"));

            let warp_id_inner = warp_id.clone();
            let lane_inner = lane.clone();
            b.if_lt_u32(warp_id, num_rows, move |b| {
                let row = warp_id_inner;
                let lane = lane_inner;

                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");

                let alpha_bits = b.load_param_u32("alpha_bits");
                let alpha = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {alpha}, {alpha_bits};"));

                let beta_bits = b.load_param_u32("beta_bits");
                let beta = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {beta}, {beta_bits};"));

                // Load row bounds
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), row.clone(), 4);
                let row_start_i32 = b.load_global_i32(rp_addr);
                let row_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_start}, {row_start_i32};"));

                let row_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_plus_1}, {row}, 1;"));
                let rp_addr_next = b.byte_offset_addr(row_ptr_base, row_plus_1, 4);
                let row_end_i32 = b.load_global_i32(rp_addr_next);
                let row_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_end}, {row_end_i32};"));

                // Compute nnz for this row and the "paired" end (rounded down to even)
                let nnz_row = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("sub.u32 {nnz_row}, {row_end}, {row_start};"));
                let nnz_even = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("and.b32 {nnz_even}, {nnz_row}, 0xFFFFFFFE;"));
                let row_end_even = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_end_even}, {row_start}, {nnz_even};"));

                // Accumulator
                let acc = b.alloc_reg(compute.ptx_type());
                let zero_bits: u32 = 0u32;
                b.raw_ptx(&format!("mov.{compute_bit} {acc}, 0F{zero_bits:08X};"));

                // Packed loop: each lane processes pairs, stride = 32*2 = 64 elements
                // Lane processes element indices: row_start + lane*2, row_start + lane*2 + 64, ...
                let k = b.alloc_reg(PtxType::U32);
                let lane_x2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shl.b32 {lane_x2}, {lane}, 1;"));
                b.raw_ptx(&format!("add.u32 {k}, {row_start}, {lane_x2};"));

                let packed_loop = b.fresh_label("mpspmv_packed_loop");
                let packed_done = b.fresh_label("mpspmv_packed_done");

                b.label(&packed_loop);
                let pred_pair = b.alloc_reg(PtxType::Pred);
                // k+1 must be < row_end_even to process a full pair
                let k_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {k_plus_1}, {k}, 1;"));
                b.raw_ptx(&format!("setp.ls.u32 {pred_pair}, {k_plus_1}, {row_end_even};"));
                b.raw_ptx(&format!("@!{pred_pair} bra {packed_done};"));

                // Load packed 2xFP16 as a 32-bit value
                // Address = values_base + k * 2 (each FP16 is 2 bytes)
                let v_addr = b.byte_offset_addr(values_base.clone(), k.clone(), elem_bytes);
                let packed_val = b.alloc_reg(PtxType::B32);
                b.raw_ptx(&format!("ld.global.b32 {packed_val}, [{v_addr}];"));

                // Unpack low half (first FP16)
                let val_lo = b.alloc_reg(storage.ptx_type());
                b.raw_ptx(&format!("mov.b16 {val_lo}, {{0}}; // placeholder"));
                // Extract low 16 bits
                let lo_bits = b.alloc_reg(PtxType::B16);
                b.raw_ptx("{ .reg .b16 __hi;");
                b.raw_ptx(&format!("mov.b32 {{{lo_bits}, __hi}}, {packed_val}; }}"));
                let val_lo_h = b.alloc_reg(storage.ptx_type());
                b.raw_ptx(&format!("mov.b16 {val_lo_h}, {lo_bits};"));
                let val_lo_f32 = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "cvt.{compute_suffix}.{storage_suffix} {val_lo_f32}, {val_lo_h};"
                ));

                // Extract high 16 bits
                let hi_bits = b.alloc_reg(PtxType::B16);
                b.raw_ptx("{ .reg .b16 __lo;");
                b.raw_ptx(&format!("mov.b32 {{__lo, {hi_bits}}}, {packed_val}; }}"));
                let val_hi_h = b.alloc_reg(storage.ptx_type());
                b.raw_ptx(&format!("mov.b16 {val_hi_h}, {hi_bits};"));
                let val_hi_f32 = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "cvt.{compute_suffix}.{storage_suffix} {val_hi_f32}, {val_hi_h};"
                ));

                // Load two column indices and x values
                let ci_addr_0 = b.byte_offset_addr(col_idx_base.clone(), k.clone(), 4);
                let col_0_i32 = b.load_global_i32(ci_addr_0);
                let col_0 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_0}, {col_0_i32};"));

                let ci_addr_1 = b.byte_offset_addr(col_idx_base.clone(), k_plus_1.clone(), 4);
                let col_1_i32 = b.load_global_i32(ci_addr_1);
                let col_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_1}, {col_1_i32};"));

                let x_addr_0 = b.byte_offset_addr(x_ptr.clone(), col_0, compute.element_bytes());
                let x_val_0 = b.load_global_f32(x_addr_0);

                let x_addr_1 = b.byte_offset_addr(x_ptr.clone(), col_1, compute.element_bytes());
                let x_val_1 = b.load_global_f32(x_addr_1);

                // FMA: acc += val_lo * x0; acc += val_hi * x1
                let acc1 = b.fma_f32(val_lo_f32, x_val_0, acc.clone());
                b.raw_ptx(&format!("mov.{compute_suffix} {acc}, {acc1};"));
                let acc2 = b.fma_f32(val_hi_f32, x_val_1, acc.clone());
                b.raw_ptx(&format!("mov.{compute_suffix} {acc}, {acc2};"));

                // k += 64 (32 lanes * 2 elements per lane)
                b.raw_ptx(&format!("add.u32 {k}, {k}, 64;"));
                b.branch(&packed_loop);
                b.label(&packed_done);

                // Handle remainder: if nnz is odd, one element is left unpaired
                let has_remainder = b.alloc_reg(PtxType::Pred);
                let nnz_odd = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("and.b32 {nnz_odd}, {nnz_row}, 1;"));
                b.raw_ptx(&format!("setp.ne.u32 {has_remainder}, {nnz_odd}, 0;"));

                let remainder_done = b.fresh_label("mpspmv_rem_done");
                b.raw_ptx(&format!("@!{has_remainder} bra {remainder_done};"));

                // Only lane 0 handles the remainder element
                let is_lane_0_rem = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {is_lane_0_rem}, {lane}, 0;"));
                b.raw_ptx(&format!("@!{is_lane_0_rem} bra {remainder_done};"));

                // Last element index = row_end - 1
                let last_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("sub.u32 {last_idx}, {row_end}, 1;"));

                let last_v_addr =
                    b.byte_offset_addr(values_base.clone(), last_idx.clone(), elem_bytes);
                let last_val_h = b.alloc_reg(storage.ptx_type());
                b.raw_ptx(&format!(
                    "ld.global.{storage_suffix} {last_val_h}, [{last_v_addr}];"
                ));
                let last_val_f32 = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "cvt.{compute_suffix}.{storage_suffix} {last_val_f32}, {last_val_h};"
                ));

                let last_ci_addr = b.byte_offset_addr(col_idx_base, last_idx, 4);
                let last_col_i32 = b.load_global_i32(last_ci_addr);
                let last_col = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {last_col}, {last_col_i32};"));

                let last_x_addr =
                    b.byte_offset_addr(x_ptr, last_col, compute.element_bytes());
                let last_x_val = b.load_global_f32(last_x_addr);

                let acc_rem = b.fma_f32(last_val_f32, last_x_val, acc.clone());
                b.raw_ptx(&format!("mov.{compute_suffix} {acc}, {acc_rem};"));

                b.label(&remainder_done);

                // Warp shuffle reduction in FP32
                let mut current = acc;
                for offset in [16u32, 8, 4, 2, 1] {
                    let shuffled = b.alloc_reg(compute.ptx_type());
                    b.raw_ptx(&format!(
                        "shfl.sync.down.{compute_bit} {shuffled}, {current}, {offset}, 31, 0xFFFFFFFF;"
                    ));
                    let sum = b.alloc_reg(compute.ptx_type());
                    b.raw_ptx(&format!(
                        "add.{compute_suffix} {sum}, {current}, {shuffled};"
                    ));
                    current = sum;
                }

                // Lane 0 writes the final result
                let is_lane_0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {is_lane_0}, {lane}, 0;"));

                let skip_label = b.fresh_label("mpspmv_pskip");
                b.raw_ptx(&format!("@!{is_lane_0} bra {skip_label};"));

                let y_addr = b.byte_offset_addr(y_ptr, row, compute.element_bytes());
                let y_old = b.load_global_f32(y_addr.clone());

                let alpha_acc = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {alpha_acc}, {alpha}, {current};"
                ));
                let beta_y = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {beta_y}, {beta}, {y_old};"
                ));
                let result = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "add.{compute_suffix} {result}, {alpha_acc}, {beta_y};"
                ));

                b.store_global_f32(y_addr, result);

                b.label(&skip_label);
            });

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// Helper: resolve block/grid sizes
// ---------------------------------------------------------------------------

/// Returns `(grid_size, block_size)` for a scalar kernel.
#[must_use]
pub fn scalar_launch_params(num_rows: u32) -> (u32, u32) {
    let block = SCALAR_BLOCK_SIZE;
    let grid = num_rows.div_ceil(block);
    (grid, block)
}

/// Returns `(grid_size, block_size)` for a vector/packed kernel.
#[must_use]
pub fn vector_launch_params(num_rows: u32) -> (u32, u32) {
    let block = VECTOR_BLOCK_SIZE;
    let warps_per_block = block / 32;
    let grid = num_rows.div_ceil(warps_per_block);
    (grid, block)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_fp16_config(algo: MixedSpMVAlgo) -> MixedPrecisionConfig {
        MixedPrecisionConfig::fp16_fp32(algo, SmVersion::Sm80)
    }

    fn default_bf16_config(algo: MixedSpMVAlgo) -> MixedPrecisionConfig {
        MixedPrecisionConfig::bf16_fp32(algo, SmVersion::Sm80)
    }

    // --- PTX Generation tests ---

    #[test]
    fn scalar_ptx_fp16_generates() {
        let config = default_fp16_config(MixedSpMVAlgo::Scalar);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains(".entry mixed_spmv_scalar"));
        assert!(ptx.contains(".target sm_80"));
        // Should contain FP16 -> FP32 conversion
        assert!(ptx.contains("cvt.f32.f16"));
    }

    #[test]
    fn scalar_ptx_bf16_generates() {
        let config = default_bf16_config(MixedSpMVAlgo::Scalar);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains("cvt.f32.bf16"));
    }

    #[test]
    fn vector_ptx_fp16_generates() {
        let config = default_fp16_config(MixedSpMVAlgo::Vector);
        let ptx = generate_mixed_vector_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains(".entry mixed_spmv_vector"));
        assert!(ptx.contains("shfl.sync.down"));
        assert!(ptx.contains("cvt.f32.f16"));
    }

    #[test]
    fn vector_ptx_bf16_generates() {
        let config = default_bf16_config(MixedSpMVAlgo::Vector);
        let ptx = generate_mixed_vector_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains("cvt.f32.bf16"));
        assert!(ptx.contains("shfl.sync.down"));
    }

    #[test]
    fn packed_ptx_fp16_generates() {
        let config = default_fp16_config(MixedSpMVAlgo::VectorPacked);
        let ptx = generate_packed_vector_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains(".entry mixed_spmv_packed"));
        // Should use 32-bit loads for packed FP16
        assert!(ptx.contains("ld.global.b32"));
        assert!(ptx.contains("shfl.sync.down"));
    }

    #[test]
    fn packed_ptx_bf16_generates() {
        let config = default_bf16_config(MixedSpMVAlgo::VectorPacked);
        let ptx = generate_packed_vector_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains("cvt.f32.bf16"));
    }

    // --- Config validation tests ---

    #[test]
    fn validate_fp16_on_turing() {
        let config = MixedPrecisionConfig::fp16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm75);
        assert!(validate_mixed_precision_config(&config).is_ok());
    }

    #[test]
    fn validate_bf16_on_turing_fails() {
        let config = MixedPrecisionConfig::bf16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm75);
        let result = validate_mixed_precision_config(&config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.expect_err("test"));
        assert!(err_msg.contains("BF16"));
    }

    #[test]
    fn validate_bf16_on_ampere_ok() {
        let config = MixedPrecisionConfig::bf16_fp32(MixedSpMVAlgo::Vector, SmVersion::Sm80);
        assert!(validate_mixed_precision_config(&config).is_ok());
    }

    // --- Plan tests ---

    #[test]
    fn plan_auto_selects_scalar_for_sparse() {
        let config = default_fp16_config(MixedSpMVAlgo::Auto);
        // 1000 rows, 2000 nnz => avg 2 nnz/row => should select Scalar
        let plan = plan_mixed_precision_spmv(&config, 1000, 5000, 2000);
        assert!(plan.is_ok());
        let plan = plan.expect("test");
        assert_eq!(plan.config.algorithm, MixedSpMVAlgo::Scalar);
    }

    #[test]
    fn plan_auto_selects_vector_for_moderate() {
        let config = default_fp16_config(MixedSpMVAlgo::Auto);
        // 1000 rows, 10000 nnz => avg 10 nnz/row => should select Vector
        let plan = plan_mixed_precision_spmv(&config, 1000, 5000, 10000);
        assert!(plan.is_ok());
        let plan = plan.expect("test");
        assert_eq!(plan.config.algorithm, MixedSpMVAlgo::Vector);
    }

    #[test]
    fn plan_auto_selects_packed_for_dense() {
        let config = default_fp16_config(MixedSpMVAlgo::Auto);
        // 1000 rows, 50000 nnz => avg 50 nnz/row => should select VectorPacked
        let plan = plan_mixed_precision_spmv(&config, 1000, 5000, 50000);
        assert!(plan.is_ok());
        let plan = plan.expect("test");
        assert_eq!(plan.config.algorithm, MixedSpMVAlgo::VectorPacked);
    }

    // --- Bandwidth estimation tests ---

    #[test]
    fn bandwidth_savings_fp16_vs_fp32() {
        let config = default_fp16_config(MixedSpMVAlgo::Scalar);
        let plan = plan_mixed_precision_spmv(&config, 1000, 1000, 10000).expect("test");
        // FP16 = 2 bytes, FP32 = 4 bytes => ratio = 2.0
        let ratio = plan.bandwidth_savings_ratio();
        assert!((ratio - 2.0).abs() < 1e-10, "Expected ~2.0, got {ratio}");
    }

    #[test]
    fn estimated_gflops_positive() {
        let config = default_fp16_config(MixedSpMVAlgo::Vector);
        let plan = plan_mixed_precision_spmv(&config, 10000, 10000, 100000).expect("test");
        // With 1000 GB/s peak bandwidth (A100-like)
        let gflops = plan.estimated_gflops(1000.0);
        assert!(gflops > 0.0, "Expected positive GFLOPS, got {gflops}");
        // Sanity: should be in a reasonable range for roofline
        assert!(gflops < 1000.0, "GFLOPS suspiciously high: {gflops}");
    }

    // --- Precision loss estimation tests ---

    #[test]
    fn precision_loss_fp16_bounded() {
        let loss = estimate_precision_loss(100.0, StoragePrecision::Fp16);
        // eps_fp16 = 2^-11 ~= 4.88e-4, eps_fp32 = 2^-24 ~= 5.96e-8
        // bound = 100 * (4.88e-4 + 5.96e-8) ~ 0.0488
        assert!(loss > 0.0);
        assert!(loss < 0.1, "Precision loss bound too large: {loss}");
    }

    #[test]
    fn precision_loss_bf16_larger_than_fp16() {
        let loss_fp16 = estimate_precision_loss(50.0, StoragePrecision::Fp16);
        let loss_bf16 = estimate_precision_loss(50.0, StoragePrecision::Bf16);
        // BF16 has fewer mantissa bits, so larger error bound
        assert!(
            loss_bf16 > loss_fp16,
            "BF16 loss ({loss_bf16}) should exceed FP16 loss ({loss_fp16})"
        );
    }

    // --- Launch parameter tests ---

    #[test]
    fn scalar_launch_params_correct() {
        let (grid, block) = scalar_launch_params(1000);
        assert_eq!(block, 256);
        assert_eq!(grid, 4); // ceil(1000/256) = 4
    }

    #[test]
    fn vector_launch_params_correct() {
        let (grid, block) = vector_launch_params(1000);
        assert_eq!(block, 256);
        // warps_per_block = 256/32 = 8, grid = ceil(1000/8) = 125
        assert_eq!(grid, 125);
    }

    // --- Mixed-precision PTX instruction accuracy tests ---

    /// Verify that scalar FP16→FP32 kernel contains `cvt.f32.f16` conversion.
    ///
    /// This ensures FP16 values are widened to FP32 before accumulation,
    /// which is the core requirement for mixed-precision SpMV numerical quality.
    #[test]
    fn mixed_precision_scalar_ptx_contains_fp16_to_fp32_conversion() {
        let config = MixedPrecisionConfig::fp16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm80);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX gen failed: {ptx:?}");
        let ptx = ptx.expect("test");

        // Must contain FP16 → FP32 widening conversion
        assert!(
            ptx.contains("cvt.f32.f16"),
            "scalar kernel must contain cvt.f32.f16 for FP16→FP32 widening"
        );
    }

    /// Verify that scalar FP16→FP32 kernel contains `fma.rn.f32` accumulation.
    ///
    /// The round-to-nearest FMA is required for correct FP32 accumulation quality.
    #[test]
    fn mixed_precision_scalar_ptx_contains_fma_rn_f32_accumulation() {
        let config = MixedPrecisionConfig::fp16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm80);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX gen failed: {ptx:?}");
        let ptx = ptx.expect("test");

        // Must contain FP32 FMA with round-to-nearest mode for accuracy
        assert!(
            ptx.contains("fma.rn.f32"),
            "scalar kernel must contain fma.rn.f32 for FP32 accumulation"
        );
    }

    /// Verify vector FP16→FP32 kernel contains both conversion and FMA instructions.
    #[test]
    fn mixed_precision_vector_ptx_contains_conversion_and_fma() {
        let config = MixedPrecisionConfig::fp16_fp32(MixedSpMVAlgo::Vector, SmVersion::Sm80);
        let ptx = generate_mixed_vector_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX gen failed: {ptx:?}");
        let ptx = ptx.expect("test");

        assert!(
            ptx.contains("cvt.f32.f16"),
            "vector kernel must contain cvt.f32.f16 for FP16→FP32 widening"
        );
        assert!(
            ptx.contains("fma.rn.f32"),
            "vector kernel must contain fma.rn.f32 for FP32 accumulation"
        );
    }

    /// Verify BF16→FP32 kernel uses `cvt.f32.bf16` (not `cvt.f32.f16`).
    #[test]
    fn mixed_precision_bf16_ptx_uses_bf16_conversion() {
        let config = MixedPrecisionConfig::bf16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm80);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX gen failed: {ptx:?}");
        let ptx = ptx.expect("test");

        assert!(
            ptx.contains("cvt.f32.bf16"),
            "BF16 kernel must contain cvt.f32.bf16 for BF16→FP32 widening"
        );
        // BF16 kernel should NOT use the FP16 conversion instruction
        assert!(
            !ptx.contains("cvt.f32.f16"),
            "BF16 kernel must NOT contain cvt.f32.f16"
        );
    }

    /// Verify that `mul.rn.f32` is used for alpha/beta scaling (not just fma).
    #[test]
    fn mixed_precision_scalar_ptx_uses_rn_mode_for_scaling() {
        let config = MixedPrecisionConfig::fp16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm80);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX gen failed: {ptx:?}");
        let ptx = ptx.expect("test");

        // The alpha/beta scaling uses mul.rn.f32
        assert!(
            ptx.contains("mul.rn.f32"),
            "scalar kernel must contain mul.rn.f32 for alpha/beta scaling"
        );
    }

    /// Verify precision loss estimation is monotone: more nnz → larger error bound.
    #[test]
    fn precision_loss_monotone_in_nnz_per_row() {
        let nnz_values = [1.0_f64, 10.0, 100.0, 1000.0];
        let mut prev_loss = 0.0_f64;
        for &nnz in &nnz_values {
            let loss = estimate_precision_loss(nnz, StoragePrecision::Fp16);
            assert!(
                loss >= prev_loss,
                "precision loss should increase with nnz/row: nnz={nnz}, loss={loss}, prev={prev_loss}"
            );
            prev_loss = loss;
        }
    }

    /// Verify that the precision loss is proportional to nnz/row.
    ///
    /// From the formula: loss = nnz * (eps_storage + eps_compute),
    /// so doubling nnz should double the loss.
    #[test]
    fn precision_loss_linear_in_nnz_per_row() {
        let loss_10 = estimate_precision_loss(10.0, StoragePrecision::Fp16);
        let loss_20 = estimate_precision_loss(20.0, StoragePrecision::Fp16);

        // loss_20 should be exactly 2 * loss_10 (linear in nnz)
        assert!(
            (loss_20 - 2.0 * loss_10).abs() < 1e-15,
            "Precision loss should be linear: loss(20)={loss_20} should be 2*loss(10)={loss_10}"
        );
    }

    /// Verify that the bandwidth savings ratio for BF16 is also ~2x vs FP32.
    #[test]
    fn bandwidth_savings_bf16_vs_fp32() {
        let config = default_bf16_config(MixedSpMVAlgo::Scalar);
        let plan = plan_mixed_precision_spmv(&config, 1000, 1000, 10000).expect("test");
        // BF16 = 2 bytes, FP32 = 4 bytes => ratio = 2.0
        let ratio = plan.bandwidth_savings_ratio();
        assert!(
            (ratio - 2.0).abs() < 1e-10,
            "BF16 bandwidth savings ratio should be ~2.0, got {ratio}"
        );
    }

    /// Verify the avg_nnz_per_row calculation is correct.
    #[test]
    fn mixed_plan_avg_nnz_per_row_calculation() {
        let config = default_fp16_config(MixedSpMVAlgo::Scalar);
        // 100 rows, 500 nnz => avg = 5.0
        let plan = plan_mixed_precision_spmv(&config, 100, 1000, 500).expect("test");
        let avg = plan.avg_nnz_per_row();
        assert!(
            (avg - 5.0).abs() < 1e-10,
            "avg_nnz_per_row should be 5.0, got {avg}"
        );
    }
}
