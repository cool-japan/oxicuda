//! Device-wide parallel prefix scan (exclusive / inclusive).
//!
//! Extends the block-level Blelloch scan to arbitrary array sizes using a
//! classic three-kernel approach:
//!
//! 1. **Block scan** — each block independently scans its `block_size`
//!    elements and writes the block aggregate to `block_sums[]`.
//! 2. **Aggregate scan** — a single block scans `block_sums[]` to produce
//!    exclusive prefix sums of the block aggregates.
//! 3. **Propagate** — each block adds the corresponding scanned aggregate to
//!    every element it owns, completing the device-wide scan.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::device::scan::{DeviceScanConfig, DeviceScanTemplate};
//! use oxicuda_primitives::ptx_helpers::ReduceOp;
//! use oxicuda_primitives::warp::scan::ScanKind;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = DeviceScanConfig {
//!     op: ReduceOp::Sum,
//!     ty: PtxType::F32,
//!     block_size: 256,
//!     kind: ScanKind::Exclusive,
//! };
//! let t = DeviceScanTemplate::new(cfg);
//! let (block_ptx, agg_ptx, prop_ptx) = t.generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(block_ptx.contains("device_scan_block_sum"));
//! assert!(agg_ptx.contains("device_scan_aggregate_sum"));
//! assert!(prop_ptx.contains("device_scan_propagate_sum"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::ptx_helpers::{ReduceOp, ptx_header, ptx_type_str};
use crate::warp::scan::ScanKind;

/// Configuration for a device-wide prefix scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceScanConfig {
    /// The scan operation.
    pub op: ReduceOp,
    /// Element type.
    pub ty: PtxType,
    /// Threads per block (power of 2, 32–1024).
    pub block_size: u32,
    /// Inclusive or exclusive scan.
    pub kind: ScanKind,
}

impl DeviceScanConfig {
    /// Number of blocks needed for `n` elements.
    #[must_use]
    pub fn num_blocks(&self, n: u64) -> u64 {
        n.div_ceil(self.block_size as u64)
    }

    /// Bytes of temporary storage needed for `n` elements (one entry per block).
    #[must_use]
    pub fn temp_bytes(&self, n: u64) -> u64 {
        let elem = match self.ty {
            PtxType::F64 | PtxType::U64 | PtxType::S64 => 8u64,
            _ => 4,
        };
        self.num_blocks(n) * elem
    }

    /// Kernel name for the block-scan pass.
    #[must_use]
    pub fn block_kernel_name(&self) -> String {
        format!(
            "device_scan_block_{}_{}_{}_bs{}",
            self.op.name(),
            ptx_type_str(self.ty),
            self.kind.name(),
            self.block_size
        )
    }

    /// Kernel name for the aggregate-scan pass (exclusive on block sums).
    #[must_use]
    pub fn aggregate_kernel_name(&self) -> String {
        format!(
            "device_scan_aggregate_{}_{}",
            self.op.name(),
            ptx_type_str(self.ty)
        )
    }

    /// Kernel name for the propagation pass.
    #[must_use]
    pub fn propagate_kernel_name(&self) -> String {
        format!(
            "device_scan_propagate_{}_{}_{}_bs{}",
            self.op.name(),
            ptx_type_str(self.ty),
            self.kind.name(),
            self.block_size
        )
    }
}

/// PTX code generator for device-wide prefix scan (three-kernel approach).
pub struct DeviceScanTemplate {
    /// Configuration.
    pub cfg: DeviceScanConfig,
}

impl DeviceScanTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: DeviceScanConfig) -> Self {
        Self { cfg }
    }

    /// Generate all three PTX kernels.
    ///
    /// Returns `(block_scan_ptx, aggregate_scan_ptx, propagate_ptx)`.
    ///
    /// # Errors
    ///
    /// Returns a string error on generation failure.
    pub fn generate(&self, sm: SmVersion) -> Result<(String, String, String), String> {
        let block = self.generate_block_scan(sm)?;
        let agg = self.generate_aggregate_scan(sm)?;
        let prop = self.generate_propagate(sm)?;
        Ok((block, agg, prop))
    }

    // ── Kernel 1: block-local scan + write block aggregate ──────────────────

    fn generate_block_scan(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.block_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let op = self.cfg.op.ptx_instr(self.cfg.ty);
        let bs = self.cfg.block_size;
        let is_exclusive = self.cfg.kind == ScanKind::Exclusive;
        let is_64bit = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        let eb: u32 = if is_64bit { 8 } else { 4 };

        let identity = self.identity_str();

        let mut out = ptx_header(sm);
        writeln!(out, ".shared .align {eb} .{ty} blk_scan_smem[{bs}];")
            .map_err(|e| e.to_string())?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_output,\n    \
             .param .u64 param_block_sums,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %val, %left;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u32    %tid, %bid, %left_idx, %right_idx, %stride;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %ptr_in, %ptr_out, %ptr_sums;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %elem_addr, %smem_base, %left_addr, %right_addr;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_out,  [param_output];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_sums, [param_block_sums];")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,   [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %n,         [param_n];").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %smem_base, blk_scan_smem;").map_err(|e| e.to_string())?;

        // Load input
        writeln!(out, "    setp.ge.u64 %p, %gid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %elem_addr, %gid, {eb}, %ptr_in;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.global.{ty} %val, [%elem_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %elem_addr, %tid, {eb}, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.{ty} [%elem_addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

        // Up-sweep
        let mut stride = 1u32;
        while stride < bs {
            writeln!(out, "    add.u32  %right_idx, %tid, 1;").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mul.lo.u32 %right_idx, %right_idx, {};",
                2 * stride
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    sub.u32  %right_idx, %right_idx, 1;").map_err(|e| e.to_string())?;
            writeln!(out, "    sub.u32  %left_idx, %right_idx, {stride};")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    setp.ge.u32 %p, %right_idx, {bs};").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p bra UP_{name}_{stride};").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mad.lo.u64 %left_addr,  %left_idx,  {eb}, %smem_base;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mad.lo.u64 %right_addr, %right_idx, {eb}, %smem_base;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    ld.shared.{ty} %left, [%left_addr];").map_err(|e| e.to_string())?;
            writeln!(out, "    ld.shared.{ty} %val,  [%right_addr];").map_err(|e| e.to_string())?;
            writeln!(out, "    {op} %val, %val, %left;").map_err(|e| e.to_string())?;
            writeln!(out, "    st.shared.{ty} [%right_addr], %val;").map_err(|e| e.to_string())?;
            writeln!(out, "UP_{name}_{stride}:").map_err(|e| e.to_string())?;
            writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
            stride *= 2;
        }

        // Save block aggregate from smem[bs-1] before clearing
        writeln!(out, "    setp.ne.u32 %p, %tid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra SAVE_AGG_{name};").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u64 %elem_addr, {}, {eb}, %smem_base;",
            bs - 1
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.shared.{ty} %val, [%elem_addr];").map_err(|e| e.to_string())?;
        // Store aggregate at block_sums[bid]
        writeln!(out, "    mad.lo.u64 %elem_addr, %bid, {eb}, %ptr_sums;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.{ty} [%elem_addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "SAVE_AGG_{name}:").map_err(|e| e.to_string())?;

        // Clear last element for exclusive scan
        if is_exclusive {
            writeln!(out, "    setp.ne.u32 %p, %tid, 0;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p bra CLEAR_LAST_{name};").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mad.lo.u64 %elem_addr, {}, {eb}, %smem_base;",
                bs - 1
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;
            writeln!(out, "    st.shared.{ty} [%elem_addr], %val;").map_err(|e| e.to_string())?;
            writeln!(out, "CLEAR_LAST_{name}:").map_err(|e| e.to_string())?;
            writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

            // Down-sweep
            stride = bs / 2;
            loop {
                writeln!(out, "    add.u32  %right_idx, %tid, 1;").map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "    mul.lo.u32 %right_idx, %right_idx, {};",
                    2 * stride
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "    sub.u32  %right_idx, %right_idx, 1;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    sub.u32  %left_idx, %right_idx, {stride};")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    setp.ge.u32 %p, %right_idx, {bs};")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    @%p bra DN_{name}_{stride};").map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "    mad.lo.u64 %left_addr,  %left_idx,  {eb}, %smem_base;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "    mad.lo.u64 %right_addr, %right_idx, {eb}, %smem_base;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "    ld.shared.{ty} %left, [%left_addr];")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    ld.shared.{ty} %val,  [%right_addr];")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    st.shared.{ty} [%left_addr],  %val;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    {op} %val, %val, %left;").map_err(|e| e.to_string())?;
                writeln!(out, "    st.shared.{ty} [%right_addr], %val;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "DN_{name}_{stride}:").map_err(|e| e.to_string())?;
                writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
                if stride == 1 {
                    break;
                }
                stride /= 2;
            }
        }

        // Write output
        writeln!(out, "    setp.lt.u64 %p, %gid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %elem_addr, %tid, {eb}, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    @%p ld.shared.{ty} %val, [%elem_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %elem_addr, %gid, {eb}, %ptr_out;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    @%p st.global.{ty} [%elem_addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    // ── Kernel 2: exclusive scan of block aggregates ─────────────────────────

    fn generate_aggregate_scan(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.aggregate_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let op = self.cfg.op.ptx_instr(self.cfg.ty);
        let identity = self.identity_str();
        let is_64bit = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        let eb: u32 = if is_64bit { 8 } else { 4 };
        let bs: u32 = 1024; // use max block size for aggregate scan (≤1024 blocks)

        let mut out = ptx_header(sm);
        writeln!(out, ".shared .align {eb} .{ty} agg_smem[{bs}];").map_err(|e| e.to_string())?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_block_sums,\n    \
             .param .u32 param_nblocks\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %val, %left;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %tid, %nb, %left_idx, %right_idx;")
            .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %ptr_sums, %elem_addr, %smem_base, %left_addr, %right_addr;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_sums, [param_block_sums];")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %nb,        [param_nblocks];")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %smem_base, agg_smem;").map_err(|e| e.to_string())?;

        writeln!(out, "    setp.ge.u32 %p, %tid, %nb;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %elem_addr, %tid, {eb}, %ptr_sums;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.global.{ty} %val, [%elem_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %elem_addr, %tid, {eb}, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.{ty} [%elem_addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

        // Up-sweep on block_sums (variable length, up to bs)
        let mut stride = 1u32;
        while stride < bs {
            writeln!(out, "    add.u32  %right_idx, %tid, 1;").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mul.lo.u32 %right_idx, %right_idx, {};",
                2 * stride
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    sub.u32  %right_idx, %right_idx, 1;").map_err(|e| e.to_string())?;
            writeln!(out, "    sub.u32  %left_idx, %right_idx, {stride};")
                .map_err(|e| e.to_string())?;
            // Guard: right_idx < nb AND right_idx < bs
            writeln!(out, "    setp.ge.u32 %p, %right_idx, %nb;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p bra AGG_UP_{name}_{stride};").map_err(|e| e.to_string())?;
            writeln!(out, "    setp.ge.u32 %p, %right_idx, {bs};").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p bra AGG_UP_{name}_{stride};").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mad.lo.u64 %left_addr,  %left_idx,  {eb}, %smem_base;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mad.lo.u64 %right_addr, %right_idx, {eb}, %smem_base;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    ld.shared.{ty} %left, [%left_addr];").map_err(|e| e.to_string())?;
            writeln!(out, "    ld.shared.{ty} %val,  [%right_addr];").map_err(|e| e.to_string())?;
            writeln!(out, "    {op} %val, %val, %left;").map_err(|e| e.to_string())?;
            writeln!(out, "    st.shared.{ty} [%right_addr], %val;").map_err(|e| e.to_string())?;
            writeln!(out, "AGG_UP_{name}_{stride}:").map_err(|e| e.to_string())?;
            writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
            stride *= 2;
        }

        // Clear last and down-sweep (exclusive)
        writeln!(out, "    setp.ne.u32 %p, %tid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra AGG_CLR_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    sub.u32 %right_idx, %nb, 1;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u64 %elem_addr, %right_idx, {eb}, %smem_base;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.{ty} [%elem_addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "AGG_CLR_{name}:").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

        // Simple down-sweep using nb/2 … 1
        let mut stride = bs / 2;
        loop {
            writeln!(out, "    add.u32  %right_idx, %tid, 1;").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mul.lo.u32 %right_idx, %right_idx, {};",
                2 * stride
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    sub.u32  %right_idx, %right_idx, 1;").map_err(|e| e.to_string())?;
            writeln!(out, "    sub.u32  %left_idx, %right_idx, {stride};")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    setp.ge.u32 %p, %right_idx, %nb;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p bra AGG_DN_{name}_{stride};").map_err(|e| e.to_string())?;
            writeln!(out, "    setp.ge.u32 %p, %right_idx, {bs};").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p bra AGG_DN_{name}_{stride};").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mad.lo.u64 %left_addr,  %left_idx,  {eb}, %smem_base;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mad.lo.u64 %right_addr, %right_idx, {eb}, %smem_base;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    ld.shared.{ty} %left, [%left_addr];").map_err(|e| e.to_string())?;
            writeln!(out, "    ld.shared.{ty} %val,  [%right_addr];").map_err(|e| e.to_string())?;
            writeln!(out, "    st.shared.{ty} [%left_addr],  %val;").map_err(|e| e.to_string())?;
            writeln!(out, "    {op} %val, %val, %left;").map_err(|e| e.to_string())?;
            writeln!(out, "    st.shared.{ty} [%right_addr], %val;").map_err(|e| e.to_string())?;
            writeln!(out, "AGG_DN_{name}_{stride}:").map_err(|e| e.to_string())?;
            writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
            if stride == 1 {
                break;
            }
            stride /= 2;
        }

        // Write back scanned aggregates
        writeln!(out, "    setp.lt.u32 %p, %tid, %nb;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %elem_addr, %tid, {eb}, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    @%p ld.shared.{ty} %val, [%elem_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %elem_addr, %tid, {eb}, %ptr_sums;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    @%p st.global.{ty} [%elem_addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    // ── Kernel 3: add scanned aggregate to each element ─────────────────────

    fn generate_propagate(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.propagate_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let op = self.cfg.op.ptx_instr(self.cfg.ty);
        let bs = self.cfg.block_size;
        let is_64bit = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        let eb: u32 = if is_64bit { 8 } else { 4 };

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_output,\n    \
             .param .u64 param_block_sums,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %val, %agg;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %tid, %bid;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %ptr_out, %ptr_sums, %addr;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_out,  [param_output];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_sums, [param_block_sums];")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %n,         [param_n];").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(|e| e.to_string())?;

        // Load block aggregate from block_sums[bid]
        writeln!(out, "    mad.lo.u64 %addr, %bid, {eb}, %ptr_sums;").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.global.{ty} %agg, [%addr];").map_err(|e| e.to_string())?;

        // Add aggregate to element
        writeln!(out, "    setp.lt.u64 %p, %gid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %addr, %gid, {eb}, %ptr_out;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p ld.global.{ty} %val, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p {op} %val, %val, %agg;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p st.global.{ty} [%addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    fn identity_str(&self) -> &'static str {
        match (self.cfg.op, self.cfg.ty) {
            (ReduceOp::Sum, PtxType::F32) => "0f00000000",
            (ReduceOp::Sum, _) => "0",
            (ReduceOp::Product, PtxType::F32) => "0f3F800000",
            (ReduceOp::Product, _) => "1",
            (ReduceOp::Min, PtxType::F32) => "0x7F800000",
            (ReduceOp::Min, PtxType::U32) => "0xFFFFFFFF",
            (ReduceOp::Min, _) => "0x7FFFFFFF",
            (ReduceOp::Max, PtxType::F32) => "0xFF800000",
            (ReduceOp::Max, PtxType::U32) => "0",
            (ReduceOp::Max, _) => "0x80000000",
            (ReduceOp::And, _) => "0xFFFFFFFF",
            (ReduceOp::Or | ReduceOp::Xor, _) => "0",
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    fn cfg(op: ReduceOp, ty: PtxType, kind: ScanKind) -> DeviceScanConfig {
        DeviceScanConfig {
            op,
            ty,
            block_size: 256,
            kind,
        }
    }

    #[test]
    fn kernel_names_contain_op_type() {
        let c = cfg(ReduceOp::Sum, PtxType::F32, ScanKind::Exclusive);
        assert!(
            c.block_kernel_name().contains("sum"),
            "{}",
            c.block_kernel_name()
        );
        assert!(
            c.block_kernel_name().contains("f32"),
            "{}",
            c.block_kernel_name()
        );
        assert!(
            c.aggregate_kernel_name().contains("sum"),
            "{}",
            c.aggregate_kernel_name()
        );
        assert!(
            c.propagate_kernel_name().contains("sum"),
            "{}",
            c.propagate_kernel_name()
        );
    }

    #[test]
    fn num_blocks_and_temp_bytes() {
        let c = cfg(ReduceOp::Sum, PtxType::F32, ScanKind::Exclusive);
        assert_eq!(c.num_blocks(256), 1);
        assert_eq!(c.num_blocks(257), 2);
        assert_eq!(c.temp_bytes(256), 4); // 1 block × 4 bytes/f32
    }

    #[test]
    fn generate_three_kernels_sum_f32() {
        let t = DeviceScanTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, ScanKind::Exclusive));
        let (blk, agg, prop) = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(blk.contains("device_scan_block_sum"), "block: {blk}");
        assert!(agg.contains("device_scan_aggregate_sum"), "agg: {agg}");
        assert!(prop.contains("device_scan_propagate_sum"), "prop: {prop}");
    }

    #[test]
    fn block_kernel_has_shared_memory() {
        let t = DeviceScanTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, ScanKind::Exclusive));
        let (blk, _, _) = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(blk.contains(".shared"), "must use smem\n{blk}");
    }

    #[test]
    fn block_kernel_writes_block_sums() {
        let t = DeviceScanTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, ScanKind::Exclusive));
        let (blk, _, _) = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(blk.contains("ptr_sums"), "must write block sums\n{blk}");
    }

    #[test]
    fn propagate_adds_aggregate() {
        let t = DeviceScanTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, ScanKind::Exclusive));
        let (_, _, prop) = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(
            prop.contains("add.f32"),
            "propagate must add aggregate\n{prop}"
        );
    }

    #[test]
    fn inclusive_scan_has_no_down_sweep_label() {
        let t = DeviceScanTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, ScanKind::Inclusive));
        let (blk, _, _) = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(
            !blk.contains("DN_"),
            "inclusive scan should skip down-sweep\n{blk}"
        );
    }

    #[test]
    fn max_u32_uses_max_instr_in_block() {
        let t = DeviceScanTemplate::new(cfg(ReduceOp::Max, PtxType::U32, ScanKind::Exclusive));
        let (blk, _, _) = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(blk.contains("max.u32"), "must use max.u32\n{blk}");
    }
}
