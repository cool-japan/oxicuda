//! Parallel scan (prefix sum) kernel templates.
//!
//! This module generates PTX kernels for work-efficient parallel scan (prefix sum)
//! over device arrays using the Blelloch algorithm. The scan uses a two-kernel
//! approach:
//!
//! 1. **Block-level scan**: Each thread block performs a local scan on its segment
//!    using shared memory with up-sweep (reduce) and down-sweep (propagate) phases.
//!    Each block also writes its aggregate to a separate `block_sums` array.
//!
//! 2. **Block-sum propagation**: After all block-level scans complete, this kernel
//!    adds the scanned block aggregates to each element, completing the global scan.
//!
//! Supported operations: sum, product, min, max.
//! Supported types: F32, F64, U32, U64, S32, S64.
//! Supported kinds: inclusive, exclusive.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::scan::{ScanTemplate, ScanOp, ScanKind};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32)
//!     .threads_per_block(256);
//! let ptx = template.generate(SmVersion::Sm80).expect("PTX generation failed");
//! assert!(ptx.contains("scan_sum_exclusive"));
//! assert!(ptx.contains("scan_propagate"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Scan operation types.
///
/// Each variant determines the identity element and the combining operation
/// used during the up-sweep and down-sweep phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanOp {
    /// Addition (prefix sum): identity = 0, combine = add.
    Sum,
    /// Multiplication (prefix product): identity = 1, combine = mul.
    Product,
    /// Minimum (prefix min): identity = +INF / MAX, combine = min.
    Min,
    /// Maximum (prefix max): identity = -INF / MIN, combine = max.
    Max,
}

impl ScanOp {
    /// Returns a short lowercase name for display and kernel naming.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Product => "product",
            Self::Min => "min",
            Self::Max => "max",
        }
    }

    /// Returns the identity element literal for this operation and type.
    ///
    /// The identity element `e` satisfies `op(e, x) == x` for all `x`.
    #[must_use]
    #[allow(clippy::match_same_arms)]
    pub fn identity_literal(&self, ty: PtxType) -> String {
        match (self, ty) {
            // Sum: identity = 0
            (Self::Sum, PtxType::F64) => "0d0000000000000000".to_string(),
            (Self::Sum, PtxType::F32) => "0f00000000".to_string(),
            (Self::Sum, PtxType::U32 | PtxType::U64 | PtxType::S32 | PtxType::S64) => {
                "0".to_string()
            }
            // Product: identity = 1
            (Self::Product, PtxType::F64) => "0d3FF0000000000000".to_string(),
            (Self::Product, PtxType::F32) => "0f3F800000".to_string(),
            (Self::Product, PtxType::U32 | PtxType::U64 | PtxType::S32 | PtxType::S64) => {
                "1".to_string()
            }
            // Max: identity = -INF / MIN
            (Self::Max, PtxType::F64) => "0dFFF0000000000000".to_string(),
            (Self::Max, PtxType::F32) => "0fFF800000".to_string(),
            (Self::Max, PtxType::U32 | PtxType::U64) => "0".to_string(),
            (Self::Max, PtxType::S32) => "-2147483648".to_string(),
            (Self::Max, PtxType::S64) => "-9223372036854775808".to_string(),
            // Min: identity = +INF / MAX
            (Self::Min, PtxType::F64) => "0d7FF0000000000000".to_string(),
            (Self::Min, PtxType::F32) => "0f7F800000".to_string(),
            (Self::Min, PtxType::U32) => "4294967295".to_string(),
            (Self::Min, PtxType::U64) => "18446744073709551615".to_string(),
            (Self::Min, PtxType::S32) => "2147483647".to_string(),
            (Self::Min, PtxType::S64) => "9223372036854775807".to_string(),
            // Unsupported types get 0 (validation should catch these)
            _ => "0".to_string(),
        }
    }

    /// Returns the PTX instruction that combines two values for this scan.
    fn combine_instruction(self, ty_str: &str) -> String {
        match self {
            Self::Sum => format!("add{ty_str}"),
            Self::Product => format!("mul{ty_str}"),
            Self::Min => format!("min{ty_str}"),
            Self::Max => format!("max{ty_str}"),
        }
    }
}

/// Whether the scan is inclusive or exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanKind {
    /// Inclusive: `output[i] = op(input[0], ..., input[i])`.
    Inclusive,
    /// Exclusive: `output[i] = op(input[0], ..., input[i-1])`, `output[0] = identity`.
    Exclusive,
}

impl ScanKind {
    /// Returns a short lowercase name for kernel naming.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Inclusive => "inclusive",
            Self::Exclusive => "exclusive",
        }
    }
}

/// Configuration for scan (prefix sum) kernel template.
///
/// The template generates two kernels:
/// 1. A block-level scan kernel that processes each block's segment and stores
///    per-block aggregates.
/// 2. A propagation kernel that adds scanned block aggregates to each element.
///
/// For a complete large-array scan, the caller should:
/// 1. Launch `block_scan` over the input.
/// 2. Recursively scan the `block_sums` array (using the same template or a
///    single-block scan if small enough).
/// 3. Launch `propagate` to distribute the scanned block sums.
///
/// # Items per thread
///
/// The `items_per_thread` field controls how many elements each thread processes.
/// Higher values increase arithmetic intensity and reduce launch overhead at the
/// cost of more registers per thread. Must be a power of 2.
///
/// Each block processes `threads_per_block * items_per_thread` elements.
#[derive(Debug, Clone)]
pub struct ScanTemplate {
    /// Scan operation (sum, product, min, max).
    pub op: ScanOp,
    /// Inclusive or exclusive scan.
    pub kind: ScanKind,
    /// Data precision.
    pub precision: PtxType,
    /// Threads per block (must be a power of 2, default 256).
    pub threads_per_block: u32,
    /// Number of elements each thread loads (must be a power of 2, default 2).
    pub items_per_thread: u32,
}

impl ScanTemplate {
    /// Creates a new scan template with default thread count (256) and 2 items per thread.
    #[must_use]
    pub const fn new(op: ScanOp, kind: ScanKind, precision: PtxType) -> Self {
        Self {
            op,
            kind,
            precision,
            threads_per_block: 256,
            items_per_thread: 2,
        }
    }

    /// Sets the number of threads per block (must be a power of 2).
    #[must_use]
    pub const fn threads_per_block(mut self, n: u32) -> Self {
        self.threads_per_block = n;
        self
    }

    /// Sets the number of elements each thread processes (must be a power of 2, >= 2).
    #[must_use]
    pub const fn items_per_thread(mut self, n: u32) -> Self {
        self.items_per_thread = n;
        self
    }

    /// Returns the number of elements processed per block.
    #[must_use]
    pub const fn elements_per_block(&self) -> u32 {
        self.threads_per_block * self.items_per_thread
    }

    /// Validates the template configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if:
    /// - `threads_per_block` is 0 or not a power of 2
    /// - `precision` is not a supported type for scan
    pub fn validate(&self) -> Result<(), PtxGenError> {
        if self.threads_per_block == 0 {
            return Err(PtxGenError::GenerationFailed(
                "threads_per_block must be > 0".to_string(),
            ));
        }
        if !self.threads_per_block.is_power_of_two() {
            return Err(PtxGenError::GenerationFailed(format!(
                "threads_per_block must be a power of 2, got {}",
                self.threads_per_block
            )));
        }
        if self.items_per_thread < 2 {
            return Err(PtxGenError::GenerationFailed(
                "items_per_thread must be >= 2".to_string(),
            ));
        }
        if !self.items_per_thread.is_power_of_two() {
            return Err(PtxGenError::GenerationFailed(format!(
                "items_per_thread must be a power of 2, got {}",
                self.items_per_thread
            )));
        }
        if !matches!(
            self.precision,
            PtxType::F32 | PtxType::F64 | PtxType::U32 | PtxType::U64 | PtxType::S32 | PtxType::S64
        ) {
            return Err(PtxGenError::InvalidType(format!(
                "scan requires F32, F64, U32, U64, S32, or S64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        Ok(())
    }

    /// Returns the kernel function name for the block-level scan.
    #[must_use]
    pub fn block_scan_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        if self.items_per_thread == 2 {
            format!(
                "scan_{}_{}_{}_bs{}",
                self.op.as_str(),
                self.kind.as_str(),
                type_str,
                self.threads_per_block
            )
        } else {
            format!(
                "scan_{}_{}_{}_bs{}_ipt{}",
                self.op.as_str(),
                self.kind.as_str(),
                type_str,
                self.threads_per_block,
                self.items_per_thread,
            )
        }
    }

    /// Returns the kernel function name for the propagation kernel.
    #[must_use]
    pub fn propagate_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        if self.items_per_thread == 2 {
            format!(
                "scan_propagate_{}_{}_bs{}",
                self.op.as_str(),
                type_str,
                self.threads_per_block
            )
        } else {
            format!(
                "scan_propagate_{}_{}_bs{}_ipt{}",
                self.op.as_str(),
                type_str,
                self.threads_per_block,
                self.items_per_thread,
            )
        }
    }

    /// Generates the block-level scan kernel PTX.
    ///
    /// # Kernel parameters
    ///
    /// - `input`: pointer to input array (`.u64`)
    /// - `output`: pointer to output array (`.u64`)
    /// - `block_sums`: pointer to per-block aggregate array (`.u64`)
    /// - `n`: number of elements (`.u32`)
    ///
    /// Each block processes `2 * threads_per_block` elements using the Blelloch
    /// work-efficient scan algorithm with shared memory.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] on validation failure or formatting error.
    #[allow(clippy::too_many_lines)]
    pub fn generate_block_scan(&self, sm: SmVersion) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let tpb = self.threads_per_block;
        let ipt = self.items_per_thread;
        let elements_per_block = self.elements_per_block();
        let smem_bytes = elements_per_block as usize * byte_size;
        let identity = self.op.identity_literal(self.precision);
        let combine = self.op.combine_instruction(ty);
        let kernel_name = self.block_scan_name();
        let log2_n = log2_u32(elements_per_block);
        let is_float = self.precision.is_float();
        let reg_type_str = reg_type_decl(self.precision);

        // We need enough registers for items_per_thread loads + scratch
        let val_reg_count = (ipt + 8).max(16);
        let reg_count = (ipt * 2 + 16).max(32);
        let rd_reg_count = (ipt * 2 + 8).max(16);
        let p_reg_count = (ipt + 4).max(8);

        let mut ptx = String::with_capacity(8192);

        emit_header(&mut ptx, sm)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_block_sums,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_n").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Declarations
        writeln!(ptx, "    .maxntid {tpb}, 1, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .u32 %r<{reg_count}>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .u64 %rd<{rd_reg_count}>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {reg_type_str} %val<{val_reg_count}>;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<{p_reg_count}>;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    .shared .align {} .b8 smem_scan[{}];",
            byte_size.max(4),
            smem_bytes
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread indices
        writeln!(ptx, "    // Thread and block indices").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        // gid_base = bid * elements_per_block + tid * items_per_thread
        writeln!(ptx, "    mul.lo.u32 %r3, %r1, {elements_per_block};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r16, %r0, {ipt};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r4, %r3, %r16;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load params
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_block_sums];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r6, [%param_n];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u64 %rd3, smem_scan;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load items_per_thread elements per thread into shared memory
        // For each item i in [0..ipt):
        //   gid_i = gid_base + i
        //   smem[tid * ipt + i] = (gid_i < n) ? input[gid_i] : identity
        for i in 0..ipt {
            let gid_reg = format!("%r{}", 17 + i);
            let smem_idx_reg = format!("%r{}", 17 + ipt + i);
            let val_reg = format!("%val{i}");
            let pred_reg = format!("%p{i}");
            let tmp_rd = format!("%rd{}", 4 + i);
            let tmp_rd2 = format!("%rd{}", 4 + ipt + i);
            writeln!(ptx, "    // Load element {i} into smem[tid*{ipt}+{i}]")
                .map_err(PtxGenError::FormatError)?;
            emit_identity_mov(&mut ptx, &val_reg, &identity, ty, is_float)?;
            writeln!(ptx, "    add.u32 {gid_reg}, %r4, {i};").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    setp.lt.u32 {pred_reg}, {gid_reg}, %r6;")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    @!{pred_reg} bra $SKIP_LOAD{i};")
                .map_err(PtxGenError::FormatError)?;
            emit_global_load(&mut ptx, &val_reg, "%rd0", &gid_reg, &tmp_rd, ty, byte_size)?;
            writeln!(ptx, "$SKIP_LOAD{i}:").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    add.u32 {smem_idx_reg}, %r16, {i};")
                .map_err(PtxGenError::FormatError)?;
            emit_smem_store_by_idx(
                &mut ptx,
                &val_reg,
                "%rd3",
                &smem_idx_reg,
                &tmp_rd2,
                ty,
                byte_size,
            )?;
            writeln!(ptx).map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // For backward compatibility: set %r5 = gid_base + ipt - 1 (last gid for this thread)
        // and keep %r4 as gid_base
        writeln!(ptx, "    add.u32 %r5, %r4, {};", ipt - 1).map_err(PtxGenError::FormatError)?;

        // === Up-sweep (reduce) phase ===
        emit_up_sweep(
            &mut ptx,
            &combine,
            ty,
            byte_size,
            tpb,
            elements_per_block,
            log2_n,
        )?;

        // Save block total and set last element to identity for exclusive scan
        emit_save_block_total(
            &mut ptx,
            ty,
            byte_size,
            elements_per_block,
            &identity,
            is_float,
            self.kind,
        )?;

        // === Down-sweep phase ===
        emit_down_sweep(
            &mut ptx,
            &combine,
            ty,
            byte_size,
            elements_per_block,
            log2_n,
        )?;

        // Write results back to global memory
        emit_write_results(&mut ptx, &combine, ty, byte_size, ipt, self.kind)?;

        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    /// Generates the block-sum propagation kernel PTX.
    ///
    /// After the block-level scan and a scan of `block_sums`, this kernel adds
    /// the scanned block sum to every element in each block (except block 0).
    ///
    /// # Kernel parameters
    ///
    /// - `output`: pointer to partially-scanned output array (`.u64`)
    /// - `block_sums`: pointer to scanned block sums array (`.u64`)
    /// - `n`: number of elements (`.u32`)
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] on validation failure or formatting error.
    pub fn generate_propagate(&self, sm: SmVersion) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let tpb = self.threads_per_block;
        let ipt = self.items_per_thread;
        let elements_per_block = self.elements_per_block();
        let combine = self.op.combine_instruction(ty);
        let kernel_name = self.propagate_name();
        let reg_type_str = reg_type_decl(self.precision);
        let val_reg_count = (ipt + 4).max(8);
        let r_reg_count = (ipt + 8).max(16);
        let p_reg_count = (ipt + 2).max(4);

        let mut ptx = String::with_capacity(2048);

        emit_header(&mut ptx, sm)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_block_sums,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_n").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    .maxntid {tpb}, 1, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .u32 %r<{r_reg_count}>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .u64 %rd<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {reg_type_str} %val<{val_reg_count}>;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<{p_reg_count}>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Skip block 0 entirely
        writeln!(ptx, "    // Skip block 0 (already correct)").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.eq.u32 %p0, %r0, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $PROP_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute global base index for this thread
        writeln!(ptx, "    mov.u32 %r1, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r2, %r0, {elements_per_block};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r6, %r1, {ipt};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r3, %r2, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_block_sums];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r5, [%param_n];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load block_sums[blockIdx.x]
        writeln!(ptx, "    // Load block_sums[blockIdx.x]").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd2, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd2, %rd2, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd3, %rd1, %rd2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %val0, [%rd3];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Propagate to each of items_per_thread elements
        for i in 0..ipt {
            let gid_reg = format!("%r{}", 7 + i);
            let pred_idx = 1 + i;
            writeln!(ptx, "    // Propagate to element {i}").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    add.u32 {gid_reg}, %r3, {i};").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    setp.lt.u32 %p{pred_idx}, {gid_reg}, %r5;")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    @!%p{pred_idx} bra $SKIP_PROP{i};")
                .map_err(PtxGenError::FormatError)?;
            let val_idx = i + 1;
            emit_global_load(
                &mut ptx,
                &format!("%val{val_idx}"),
                "%rd0",
                &gid_reg,
                "%rd4",
                ty,
                byte_size,
            )?;
            writeln!(ptx, "    {combine} %val{val_idx}, %val0, %val{val_idx};")
                .map_err(PtxGenError::FormatError)?;
            emit_global_store(
                &mut ptx,
                &format!("%val{val_idx}"),
                "%rd0",
                &gid_reg,
                "%rd4",
                ty,
                byte_size,
            )?;
            writeln!(ptx, "$SKIP_PROP{i}:").map_err(PtxGenError::FormatError)?;
            writeln!(ptx).map_err(PtxGenError::FormatError)?;
        }

        writeln!(ptx, "$PROP_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    /// Generates both the block-level scan and propagation kernels as a single
    /// PTX module string.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] on validation failure or formatting error.
    pub fn generate(&self, sm: SmVersion) -> Result<String, PtxGenError> {
        self.validate()?;

        let scan_ptx = self.generate_block_scan(sm)?;
        let prop_ptx = self.generate_propagate(sm)?;

        let mut combined = scan_ptx;
        combined.push('\n');

        // Skip the header lines (.version, .target, .address_size, blank) from propagate
        let prop_body = skip_ptx_header(&prop_ptx);
        combined.push_str(prop_body);

        Ok(combined)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Computes floor(log2(n)) for a positive power-of-2 integer.
fn log2_u32(n: u32) -> u32 {
    debug_assert!(n > 0 && n.is_power_of_two());
    n.trailing_zeros()
}

/// Emits the PTX header (version, target, `address_size`).
fn emit_header(ptx: &mut String, sm: SmVersion) -> Result<(), PtxGenError> {
    writeln!(ptx, ".version {}", sm.ptx_version()).map_err(PtxGenError::FormatError)?;
    writeln!(ptx, ".target {}", sm.as_ptx_str()).map_err(PtxGenError::FormatError)?;
    writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
    writeln!(ptx).map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Returns the register type declaration string for a given `PtxType`.
const fn reg_type_decl(ty: PtxType) -> &'static str {
    match ty {
        PtxType::F32 => ".f32",
        PtxType::F64 => ".f64",
        PtxType::U32 | PtxType::S32 => ".u32",
        PtxType::U64 | PtxType::S64 => ".u64",
        _ => ".b32",
    }
}

/// Emits a `mov` instruction to set a register to the identity value.
fn emit_identity_mov(
    ptx: &mut String,
    dst: &str,
    identity: &str,
    ty: &str,
    _is_float: bool,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    mov{ty} {dst}, {identity};").map_err(PtxGenError::FormatError)
}

/// Emits a load from global memory: `dst = base[idx]`.
fn emit_global_load(
    ptx: &mut String,
    dst: &str,
    base: &str,
    idx: &str,
    tmp: &str,
    ty: &str,
    byte_size: usize,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    cvt.u64.u32 {tmp}, {idx};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 {tmp}, {tmp}, {byte_size};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 {tmp}, {base}, {tmp};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.global{ty} {dst}, [{tmp}];").map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits a store to global memory: `base[idx] = src`.
fn emit_global_store(
    ptx: &mut String,
    src: &str,
    base: &str,
    idx: &str,
    tmp: &str,
    ty: &str,
    byte_size: usize,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    cvt.u64.u32 {tmp}, {idx};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 {tmp}, {tmp}, {byte_size};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 {tmp}, {base}, {tmp};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    st.global{ty} [{tmp}], {src};").map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits a store to shared memory: `smem[idx] = src`.
fn emit_smem_store_by_idx(
    ptx: &mut String,
    src: &str,
    smem_base: &str,
    idx: &str,
    tmp: &str,
    ty: &str,
    byte_size: usize,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    cvt.u64.u32 {tmp}, {idx};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 {tmp}, {tmp}, {byte_size};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 {tmp}, {smem_base}, {tmp};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    st.shared{ty} [{tmp}], {src};").map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits a load from shared memory: `dst = smem[idx]`.
fn emit_smem_load_by_idx(
    ptx: &mut String,
    dst: &str,
    smem_base: &str,
    idx: &str,
    tmp: &str,
    ty: &str,
    byte_size: usize,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    cvt.u64.u32 {tmp}, {idx};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 {tmp}, {tmp}, {byte_size};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 {tmp}, {smem_base}, {tmp};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.shared{ty} {dst}, [{tmp}];").map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits the up-sweep (reduce) phase of the Blelloch scan.
fn emit_up_sweep(
    ptx: &mut String,
    combine: &str,
    ty: &str,
    byte_size: usize,
    _tpb: u32,
    elements_per_block: u32,
    log2_n: u32,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    // === Up-sweep (reduce) phase ===").map_err(PtxGenError::FormatError)?;
    for d in 0..log2_n {
        let stride: u32 = 1 << (d + 1);
        let half: u32 = 1 << d;
        let active = elements_per_block / stride;
        let left_off = half - 1;
        let right_off = stride - 1;
        writeln!(ptx, "    // Level {d}: stride={stride}, active={active}")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.u32 %p2, %r0, {active};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p2 bra $UP_SKIP_{d};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r8, %r0, {stride};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r9, %r8, {left_off};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r10, %r8, {right_off};").map_err(PtxGenError::FormatError)?;
        emit_smem_load_by_idx(ptx, "%val2", "%rd3", "%r9", "%rd8", ty, byte_size)?;
        emit_smem_load_by_idx(ptx, "%val3", "%rd3", "%r10", "%rd9", ty, byte_size)?;
        writeln!(ptx, "    {combine} %val3, %val2, %val3;").map_err(PtxGenError::FormatError)?;
        emit_smem_store_by_idx(ptx, "%val3", "%rd3", "%r10", "%rd9", ty, byte_size)?;
        writeln!(ptx, "$UP_SKIP_{d}:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
    }
    writeln!(ptx).map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits the block total save and (for exclusive scan) identity insertion.
fn emit_save_block_total(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    elements_per_block: u32,
    identity: &str,
    is_float: bool,
    kind: ScanKind,
) -> Result<(), PtxGenError> {
    let last_idx = elements_per_block - 1;
    writeln!(ptx, "    // Save block total to block_sums[bid]")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    setp.eq.u32 %p3, %r0, 0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    @!%p3 bra $SKIP_SAVE_TOTAL;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mov.u32 %r11, {last_idx};").map_err(PtxGenError::FormatError)?;
    emit_smem_load_by_idx(ptx, "%val4", "%rd3", "%r11", "%rd10", ty, byte_size)?;
    writeln!(ptx, "    cvt.u64.u32 %rd11, %r1;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {byte_size};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd12, %rd2, %rd11;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    st.global{ty} [%rd12], %val4;").map_err(PtxGenError::FormatError)?;
    if kind == ScanKind::Exclusive {
        emit_identity_mov(ptx, "%val5", identity, ty, is_float)?;
        emit_smem_store_by_idx(ptx, "%val5", "%rd3", "%r11", "%rd10", ty, byte_size)?;
    }
    writeln!(ptx, "$SKIP_SAVE_TOTAL:").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx).map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits the down-sweep phase of the Blelloch scan.
fn emit_down_sweep(
    ptx: &mut String,
    combine: &str,
    ty: &str,
    byte_size: usize,
    elements_per_block: u32,
    log2_n: u32,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    // === Down-sweep phase ===").map_err(PtxGenError::FormatError)?;
    for d in (0..log2_n).rev() {
        let stride: u32 = 1 << (d + 1);
        let half: u32 = 1 << d;
        let active = elements_per_block / stride;
        let left_off = half - 1;
        let right_off = stride - 1;
        writeln!(ptx, "    // Level {d}: stride={stride}, active={active}")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.u32 %p4, %r0, {active};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p4 bra $DOWN_SKIP_{d};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r12, %r0, {stride};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r13, %r12, {left_off};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r14, %r12, {right_off};").map_err(PtxGenError::FormatError)?;
        // Load left and right
        emit_smem_load_by_idx(ptx, "%val2", "%rd3", "%r13", "%rd8", ty, byte_size)?;
        emit_smem_load_by_idx(ptx, "%val3", "%rd3", "%r14", "%rd9", ty, byte_size)?;
        // smem[left] = smem[right]
        emit_smem_store_by_idx(ptx, "%val3", "%rd3", "%r13", "%rd8", ty, byte_size)?;
        // smem[right] = op(old_left, old_right)
        writeln!(ptx, "    {combine} %val3, %val2, %val3;").map_err(PtxGenError::FormatError)?;
        emit_smem_store_by_idx(ptx, "%val3", "%rd3", "%r14", "%rd9", ty, byte_size)?;
        writeln!(ptx, "$DOWN_SKIP_{d}:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
    }
    writeln!(ptx).map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits the result write-back phase for `items_per_thread` elements per thread.
fn emit_write_results(
    ptx: &mut String,
    combine: &str,
    ty: &str,
    byte_size: usize,
    ipt: u32,
    kind: ScanKind,
) -> Result<(), PtxGenError> {
    // %r16 = tid * ipt, %r4 = gid_base (already computed)
    if kind == ScanKind::Inclusive {
        writeln!(
            ptx,
            "    // Convert exclusive to inclusive: output[i] = op(excl[i], input[i])"
        )
        .map_err(PtxGenError::FormatError)?;
    } else {
        writeln!(ptx, "    // Write exclusive scan results to global memory")
            .map_err(PtxGenError::FormatError)?;
    }

    for i in 0..ipt {
        let gid_reg = format!("%r{}", 17 + i);
        let smem_idx_reg = format!("%r{}", 17 + ipt + i);
        let pred_idx = 5 + i;
        // Compute gid_i = gid_base + i
        writeln!(ptx, "    add.u32 {gid_reg}, %r4, {i};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.u32 %p{pred_idx}, {gid_reg}, %r6;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p{pred_idx} bra $SKIP_WRITE{i};")
            .map_err(PtxGenError::FormatError)?;
        // smem index = tid * ipt + i
        writeln!(ptx, "    add.u32 {smem_idx_reg}, %r16, {i};")
            .map_err(PtxGenError::FormatError)?;
        emit_smem_load_by_idx(ptx, "%val2", "%rd3", &smem_idx_reg, "%rd8", ty, byte_size)?;
        if kind == ScanKind::Inclusive {
            emit_global_load(ptx, "%val3", "%rd0", &gid_reg, "%rd9", ty, byte_size)?;
            writeln!(ptx, "    {combine} %val2, %val2, %val3;")
                .map_err(PtxGenError::FormatError)?;
        }
        emit_global_store(ptx, "%val2", "%rd1", &gid_reg, "%rd10", ty, byte_size)?;
        writeln!(ptx, "$SKIP_WRITE{i}:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;
    }
    Ok(())
}

/// Skips the PTX header lines (`.version`, `.target`, `.address_size`, blank line)
/// and returns the rest.
fn skip_ptx_header(ptx: &str) -> &str {
    let mut lines_skipped = 0;
    let mut pos = 0;
    for (i, ch) in ptx.char_indices() {
        if ch == '\n' {
            lines_skipped += 1;
            pos = i + 1;
            if lines_skipped >= 4 {
                break;
            }
        }
    }
    ptx.get(pos..).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;

    #[test]
    fn scan_template_creation_defaults() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32);
        assert_eq!(t.op, ScanOp::Sum);
        assert_eq!(t.kind, ScanKind::Exclusive);
        assert_eq!(t.precision, PtxType::F32);
        assert_eq!(t.threads_per_block, 256);
    }

    #[test]
    fn validate_rejects_non_power_of_2() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(100);
        assert!(t.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_threads() {
        let t =
            ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32).threads_per_block(0);
        assert!(t.validate().is_err());
    }

    #[test]
    fn validate_rejects_unsupported_type() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::Pred);
        assert!(t.validate().is_err());
    }

    #[test]
    fn generate_block_scan_f32_inclusive() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Inclusive, PtxType::F32)
            .threads_per_block(256);
        let ptx = t
            .generate_block_scan(SmVersion::Sm80)
            .expect("should generate block scan");
        assert!(ptx.contains(".entry scan_sum_inclusive_f32_bs256"));
        assert!(ptx.contains("bar.sync 0"));
        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("smem_scan"));
    }

    #[test]
    fn generate_block_scan_f32_exclusive() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(256);
        let ptx = t
            .generate_block_scan(SmVersion::Sm80)
            .expect("should generate exclusive block scan");
        assert!(ptx.contains(".entry scan_sum_exclusive_f32_bs256"));
        assert!(ptx.contains("bar.sync 0"));
        assert!(ptx.contains(".shared"));
    }

    #[test]
    fn generate_propagate_valid() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(256);
        let ptx = t
            .generate_propagate(SmVersion::Sm80)
            .expect("should generate propagate kernel");
        assert!(ptx.contains(".entry scan_propagate_sum_f32_bs256"));
        assert!(ptx.contains("block_sums"));
    }

    #[test]
    fn generate_both_kernels() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(256);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate both kernels");
        assert!(ptx.contains("scan_sum_exclusive_f32_bs256"));
        assert!(ptx.contains("scan_propagate_sum_f32_bs256"));
    }

    #[test]
    fn ptx_contains_target_directive() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32);
        let ptx = t
            .generate_block_scan(SmVersion::Sm80)
            .expect("should generate");
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn ptx_contains_shared_memory() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(256);
        let ptx = t
            .generate_block_scan(SmVersion::Sm80)
            .expect("should generate");
        // 2 * 256 * 4 = 2048 bytes
        assert!(ptx.contains("smem_scan[2048]"));
    }

    #[test]
    fn identity_literal_sum() {
        assert_eq!(ScanOp::Sum.identity_literal(PtxType::F32), "0f00000000");
        assert_eq!(ScanOp::Sum.identity_literal(PtxType::U32), "0");
    }

    #[test]
    fn identity_literal_product() {
        assert_eq!(ScanOp::Product.identity_literal(PtxType::F32), "0f3F800000");
        assert_eq!(ScanOp::Product.identity_literal(PtxType::U32), "1");
    }

    #[test]
    fn identity_literal_min_max() {
        assert_eq!(ScanOp::Min.identity_literal(PtxType::F32), "0f7F800000");
        assert_eq!(
            ScanOp::Min.identity_literal(PtxType::F64),
            "0d7FF0000000000000"
        );
        assert_eq!(ScanOp::Min.identity_literal(PtxType::U32), "4294967295");
        assert_eq!(ScanOp::Min.identity_literal(PtxType::S32), "2147483647");

        assert_eq!(ScanOp::Max.identity_literal(PtxType::F32), "0fFF800000");
        assert_eq!(
            ScanOp::Max.identity_literal(PtxType::F64),
            "0dFFF0000000000000"
        );
        assert_eq!(ScanOp::Max.identity_literal(PtxType::U32), "0");
        assert_eq!(ScanOp::Max.identity_literal(PtxType::S32), "-2147483648");
    }

    #[test]
    fn scan_op_as_str() {
        assert_eq!(ScanOp::Sum.as_str(), "sum");
        assert_eq!(ScanOp::Product.as_str(), "product");
        assert_eq!(ScanOp::Min.as_str(), "min");
        assert_eq!(ScanOp::Max.as_str(), "max");
    }

    #[test]
    fn scan_f64_precision() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F64)
            .threads_per_block(128);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate f64 scan");
        assert!(ptx.contains("scan_sum_exclusive_f64_bs128"));
        assert!(ptx.contains(".f64"));
    }

    #[test]
    fn scan_u32_precision() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Inclusive, PtxType::U32)
            .threads_per_block(256);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate u32 scan");
        assert!(ptx.contains("scan_sum_inclusive_u32_bs256"));
        assert!(ptx.contains(".u32"));
    }

    #[test]
    fn scan_kind_is_copy() {
        let k = ScanKind::Inclusive;
        let k2 = k;
        assert_eq!(k, k2);
    }

    #[test]
    fn scan_op_is_copy() {
        let op = ScanOp::Sum;
        let op2 = op;
        assert_eq!(op, op2);
    }

    #[test]
    fn kernel_name_format() {
        let t = ScanTemplate::new(ScanOp::Product, ScanKind::Inclusive, PtxType::F32)
            .threads_per_block(512);
        assert_eq!(t.block_scan_name(), "scan_product_inclusive_f32_bs512");
        assert_eq!(t.propagate_name(), "scan_propagate_product_f32_bs512");
    }

    #[test]
    fn scan_max_f32() {
        let t = ScanTemplate::new(ScanOp::Max, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(256);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate max scan");
        assert!(ptx.contains("max.f32"));
    }

    #[test]
    fn scan_min_f32() {
        let t = ScanTemplate::new(ScanOp::Min, ScanKind::Inclusive, PtxType::F32)
            .threads_per_block(256);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate min scan");
        assert!(ptx.contains("min.f32"));
    }

    #[test]
    fn scan_product_f32() {
        let t = ScanTemplate::new(ScanOp::Product, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(256);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate product scan");
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn validate_accepts_all_supported_types() {
        for ty in [
            PtxType::F32,
            PtxType::F64,
            PtxType::U32,
            PtxType::U64,
            PtxType::S32,
            PtxType::S64,
        ] {
            let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, ty);
            assert!(t.validate().is_ok(), "should accept {ty:?}");
        }
    }

    #[test]
    fn validate_rejects_f16() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F16);
        assert!(t.validate().is_err());
    }

    #[test]
    fn scan_s32_precision() {
        let t = ScanTemplate::new(ScanOp::Min, ScanKind::Exclusive, PtxType::S32)
            .threads_per_block(256);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate s32 scan");
        assert!(ptx.contains("scan_min_exclusive_s32_bs256"));
        assert!(ptx.contains("min.s32"));
    }

    // ---- items_per_thread tests ----

    #[test]
    fn default_items_per_thread_is_two() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32);
        assert_eq!(t.items_per_thread, 2);
        assert_eq!(t.elements_per_block(), 512);
    }

    #[test]
    fn items_per_thread_four_generates_valid_ptx() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(128)
            .items_per_thread(4);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate with ipt=4");
        assert!(ptx.contains("scan_sum_exclusive_f32_bs128_ipt4"));
        assert!(ptx.contains("scan_propagate_sum_f32_bs128_ipt4"));
        // 128 * 4 * 4 bytes = 2048 bytes shared memory
        assert!(ptx.contains("smem_scan[2048]"));
        assert!(ptx.contains("bar.sync 0"));
    }

    #[test]
    fn items_per_thread_eight_f64_inclusive() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Inclusive, PtxType::F64)
            .threads_per_block(64)
            .items_per_thread(8);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate with ipt=8");
        assert!(ptx.contains("scan_sum_inclusive_f64_bs64_ipt8"));
        // 64 * 8 * 8 bytes = 4096 bytes shared memory
        assert!(ptx.contains("smem_scan[4096]"));
        assert!(ptx.contains(".f64"));
    }

    #[test]
    fn validate_rejects_items_per_thread_one() {
        let t =
            ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32).items_per_thread(1);
        assert!(t.validate().is_err());
    }

    #[test]
    fn validate_rejects_items_per_thread_non_power_of_two() {
        let t =
            ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32).items_per_thread(3);
        assert!(t.validate().is_err());
    }

    #[test]
    fn kernel_name_with_custom_ipt() {
        let t = ScanTemplate::new(ScanOp::Max, ScanKind::Inclusive, PtxType::F64)
            .threads_per_block(256)
            .items_per_thread(4);
        assert_eq!(t.block_scan_name(), "scan_max_inclusive_f64_bs256_ipt4");
        assert_eq!(t.propagate_name(), "scan_propagate_max_f64_bs256_ipt4");
    }

    #[test]
    fn kernel_name_default_ipt_no_suffix() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(256)
            .items_per_thread(2);
        // Default ipt=2 should not include _ipt suffix
        assert_eq!(t.block_scan_name(), "scan_sum_exclusive_f32_bs256");
        assert_eq!(t.propagate_name(), "scan_propagate_sum_f32_bs256");
    }

    #[test]
    fn elements_per_block_calculation() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(128)
            .items_per_thread(4);
        assert_eq!(t.elements_per_block(), 512);

        let t2 = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(256)
            .items_per_thread(8);
        assert_eq!(t2.elements_per_block(), 2048);
    }

    #[test]
    fn items_per_thread_four_product_exclusive() {
        let t = ScanTemplate::new(ScanOp::Product, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(256)
            .items_per_thread(4);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate product scan with ipt=4");
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("$SKIP_LOAD0"));
        assert!(ptx.contains("$SKIP_LOAD1"));
        assert!(ptx.contains("$SKIP_LOAD2"));
        assert!(ptx.contains("$SKIP_LOAD3"));
    }

    #[test]
    fn propagate_kernel_ipt4_has_four_elements() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Exclusive, PtxType::F32)
            .threads_per_block(128)
            .items_per_thread(4);
        let ptx = t
            .generate_propagate(SmVersion::Sm80)
            .expect("should generate propagate with ipt=4");
        assert!(ptx.contains("$SKIP_PROP0"));
        assert!(ptx.contains("$SKIP_PROP1"));
        assert!(ptx.contains("$SKIP_PROP2"));
        assert!(ptx.contains("$SKIP_PROP3"));
    }

    #[test]
    fn scan_block_size_32_ipt2() {
        let t = ScanTemplate::new(ScanOp::Sum, ScanKind::Inclusive, PtxType::F32)
            .threads_per_block(32)
            .items_per_thread(2);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate with bs=32, ipt=2");
        assert!(ptx.contains(".entry scan_sum_inclusive_f32_bs32"));
        // 32 * 2 * 4 = 256 bytes shared memory
        assert!(ptx.contains("smem_scan[256]"));
    }
}
