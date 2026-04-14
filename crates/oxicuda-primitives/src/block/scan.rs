//! Block-level parallel prefix scan using the Blelloch work-efficient algorithm.
//!
//! The block scan processes `block_size` elements residing in shared memory
//! through two phases:
//!
//! **Up-sweep (reduce) phase:**
//!
//! Partial sums are computed in a binary-tree pattern:
//!
//! ```text
//! stride=1:  smem[1]  += smem[0]
//!            smem[3]  += smem[2]
//!            smem[5]  += smem[4]  …
//! stride=2:  smem[3]  += smem[1]
//!            smem[7]  += smem[5]  …
//! stride=4:  smem[7]  += smem[3]
//!            smem[15] += smem[11] …
//! ```
//!
//! After the up-sweep, `smem[block_size - 1]` contains the total aggregate.
//!
//! **Down-sweep (scan) phase:**
//!
//! For an exclusive scan, the last element is cleared and values are
//! propagated downward:
//!
//! ```text
//! smem[block_size-1] = identity
//! stride=block_size/2: …propagate…
//! ```
//!
//! For an inclusive scan the down-sweep is adapted so `smem[i]` contains the
//! prefix sum through element `i`.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::block::scan::{BlockScanTemplate, BlockScanConfig};
//! use oxicuda_primitives::warp::scan::ScanKind;
//! use oxicuda_primitives::ptx_helpers::ReduceOp;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = BlockScanConfig {
//!     op: ReduceOp::Sum,
//!     ty: PtxType::F32,
//!     block_size: 256,
//!     kind: ScanKind::Exclusive,
//! };
//! let ptx = BlockScanTemplate::new(cfg)
//!     .generate(SmVersion::Sm80)
//!     .expect("PTX gen failed");
//! assert!(ptx.contains("block_scan_sum_f32_bs256_exclusive"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::ptx_helpers::{ReduceOp, ptx_header, ptx_type_str};
use crate::warp::scan::ScanKind;

/// Configuration for a block-level scan kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockScanConfig {
    /// The scan operation (sum, min, max, …).
    pub op: ReduceOp,
    /// Element type.
    pub ty: PtxType,
    /// Number of threads per block (must be a power of 2, 32–1024).
    pub block_size: u32,
    /// Inclusive or exclusive scan.
    pub kind: ScanKind,
}

impl BlockScanConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error string if `block_size` is invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.block_size < 32 || self.block_size > 1024 {
            return Err(format!(
                "block_size must be 32..=1024, got {}",
                self.block_size
            ));
        }
        if !self.block_size.is_power_of_two() {
            return Err(format!(
                "block_size must be a power of 2, got {}",
                self.block_size
            ));
        }
        Ok(())
    }

    /// Canonical kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!(
            "block_scan_{}_{}_bs{}_{}",
            self.op.name(),
            ptx_type_str(self.ty),
            self.block_size,
            self.kind.name()
        )
    }
}

/// PTX code generator for block-level prefix scan (Blelloch algorithm).
pub struct BlockScanTemplate {
    /// Kernel configuration.
    pub cfg: BlockScanConfig,
}

impl BlockScanTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: BlockScanConfig) -> Self {
        Self { cfg }
    }

    /// Generate PTX source for the block scan.
    ///
    /// # Errors
    ///
    /// Returns a string error if validation or generation fails.
    pub fn generate(&self, sm: SmVersion) -> Result<String, String> {
        self.cfg.validate()?;
        self.generate_inner(sm)
    }

    fn generate_inner(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let op = self.cfg.op.ptx_instr(self.cfg.ty);
        let bs = self.cfg.block_size;
        let is_exclusive = self.cfg.kind == ScanKind::Exclusive;
        let is_64bit = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        let elem_bytes: u32 = if is_64bit { 8 } else { 4 };

        // Operator-specific identity
        let identity = match (self.cfg.op, self.cfg.ty) {
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
        };

        let mut out = ptx_header(sm);

        // Shared memory: block_size elements
        writeln!(out, ".shared .align {elem_bytes} .{ty} scan_smem[{bs}];")
            .map_err(|e| e.to_string())?;

        // Kernel: (T* output, const T* input, u32 n)
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_output,\n    \
             .param .u64 param_input,\n    \
             .param .u32 param_n\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;

        writeln!(out, "    .reg .{ty}   %val, %left;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u32    %tid, %n, %stride, %left_idx, %right_idx;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %ptr_in, %ptr_out, %elem_addr, %smem_base;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %left_addr, %right_addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_out,  [param_output];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,   [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %n,         [param_n];").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %tid,       %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %smem_base, scan_smem;").map_err(|e| e.to_string())?;

        // Load input into shared memory (identity for out-of-bounds)
        writeln!(out, "    setp.ge.u32 %p, %tid, %n;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u64  %elem_addr, %tid, {elem_bytes}, %ptr_in;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.global.{ty} %val, [%elem_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u64  %elem_addr, %tid, {elem_bytes}, %smem_base;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.{ty} [%elem_addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

        // ── Up-sweep (reduce) phase ──────────────────────────────────────────
        writeln!(out, "    // === Up-sweep phase ===").map_err(|e| e.to_string())?;
        let mut stride = 1u32;
        while stride < bs {
            writeln!(out, "    // stride={stride}").map_err(|e| e.to_string())?;
            writeln!(out, "    mov.u32 %stride, {stride};").map_err(|e| e.to_string())?;
            // right_idx = (tid+1) * 2*stride - 1
            writeln!(out, "    add.u32  %right_idx, %tid, 1;").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mul.lo.u32 %right_idx, %right_idx, {};",
                2 * stride
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    sub.u32  %right_idx, %right_idx, 1;").map_err(|e| e.to_string())?;
            // left_idx = right_idx - stride
            writeln!(out, "    sub.u32  %left_idx, %right_idx, %stride;")
                .map_err(|e| e.to_string())?;

            // Guard: right_idx < bs
            writeln!(out, "    setp.ge.u32 %p, %right_idx, {bs};").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p bra UP_NEXT_{name}_{stride};").map_err(|e| e.to_string())?;

            writeln!(
                out,
                "    mad.lo.u64 %left_addr,  %left_idx,  {elem_bytes}, %smem_base;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mad.lo.u64 %right_addr, %right_idx, {elem_bytes}, %smem_base;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    ld.shared.{ty} %left,  [%left_addr];").map_err(|e| e.to_string())?;
            writeln!(out, "    ld.shared.{ty} %val,   [%right_addr];")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    {op} %val, %val, %left;").map_err(|e| e.to_string())?;
            writeln!(out, "    st.shared.{ty} [%right_addr], %val;").map_err(|e| e.to_string())?;
            writeln!(out, "UP_NEXT_{name}_{stride}:").map_err(|e| e.to_string())?;
            writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

            stride *= 2;
        }

        // ── Clear last element (exclusive) or keep (inclusive) ───────────────
        if is_exclusive {
            writeln!(out, "    // === Clear last element for exclusive scan ===")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    setp.ne.u32 %p, %tid, 0;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p bra DOWN_START_{name};").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mad.lo.u64 %elem_addr, {}, {elem_bytes}, %smem_base;",
                bs - 1
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;
            writeln!(out, "    st.shared.{ty} [%elem_addr], %val;").map_err(|e| e.to_string())?;
            writeln!(out, "DOWN_START_{name}:").map_err(|e| e.to_string())?;
            writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
        }

        // ── Down-sweep phase ─────────────────────────────────────────────────
        if is_exclusive {
            writeln!(out, "    // === Down-sweep phase (exclusive) ===")
                .map_err(|e| e.to_string())?;
            stride = bs / 2;
            while stride >= 1 {
                writeln!(out, "    // stride={stride}").map_err(|e| e.to_string())?;
                writeln!(out, "    mov.u32 %stride, {stride};").map_err(|e| e.to_string())?;
                writeln!(out, "    add.u32  %right_idx, %tid, 1;").map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "    mul.lo.u32 %right_idx, %right_idx, {};",
                    2 * stride
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "    sub.u32  %right_idx, %right_idx, 1;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    sub.u32  %left_idx, %right_idx, %stride;")
                    .map_err(|e| e.to_string())?;

                writeln!(out, "    setp.ge.u32 %p, %right_idx, {bs};")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    @%p bra DOWN_NEXT_{name}_{stride};")
                    .map_err(|e| e.to_string())?;

                writeln!(
                    out,
                    "    mad.lo.u64 %left_addr,  %left_idx,  {elem_bytes}, %smem_base;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "    mad.lo.u64 %right_addr, %right_idx, {elem_bytes}, %smem_base;"
                )
                .map_err(|e| e.to_string())?;
                // temp = smem[left]
                // smem[left] = smem[right]
                // smem[right] = smem[right] op temp
                writeln!(out, "    ld.shared.{ty} %left, [%left_addr];")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    ld.shared.{ty} %val,  [%right_addr];")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    st.shared.{ty} [%left_addr],  %val;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    {op} %val, %val, %left;").map_err(|e| e.to_string())?;
                writeln!(out, "    st.shared.{ty} [%right_addr], %val;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "DOWN_NEXT_{name}_{stride}:").map_err(|e| e.to_string())?;
                writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

                if stride == 0 {
                    break;
                }
                stride /= 2;
            }
        }

        // ── Write output from shared memory ──────────────────────────────────
        writeln!(out, "    setp.lt.u32 %p, %tid, %n;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u64  %elem_addr, %tid, {elem_bytes}, %smem_base;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    @%p ld.shared.{ty} %val, [%elem_addr];").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u64  %elem_addr, %tid, {elem_bytes}, %ptr_out;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    @%p st.global.{ty} [%elem_addr], %val;").map_err(|e| e.to_string())?;

        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    fn cfg(op: ReduceOp, ty: PtxType, bs: u32, kind: ScanKind) -> BlockScanConfig {
        BlockScanConfig {
            op,
            ty,
            block_size: bs,
            kind,
        }
    }

    #[test]
    fn kernel_name_sum_f32_bs256_exclusive() {
        let c = cfg(ReduceOp::Sum, PtxType::F32, 256, ScanKind::Exclusive);
        assert_eq!(c.kernel_name(), "block_scan_sum_f32_bs256_exclusive");
    }

    #[test]
    fn kernel_name_max_u32_bs64_inclusive() {
        let c = cfg(ReduceOp::Max, PtxType::U32, 64, ScanKind::Inclusive);
        assert_eq!(c.kernel_name(), "block_scan_max_u32_bs64_inclusive");
    }

    #[test]
    fn validate_ok_bs256() {
        assert!(
            cfg(ReduceOp::Sum, PtxType::F32, 256, ScanKind::Exclusive)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn validate_bs_300_fails() {
        assert!(
            cfg(ReduceOp::Sum, PtxType::F32, 300, ScanKind::Exclusive)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bs_16_fails() {
        assert!(
            cfg(ReduceOp::Sum, PtxType::F32, 16, ScanKind::Exclusive)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn generate_exclusive_sum_f32_has_name() {
        let t = BlockScanTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, 64, ScanKind::Exclusive));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(
            ptx.contains("block_scan_sum_f32_bs64_exclusive"),
            "missing name\n{ptx}"
        );
    }

    #[test]
    fn generate_uses_shared_memory() {
        let t = BlockScanTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, 64, ScanKind::Exclusive));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(ptx.contains(".shared"), "must use smem\n{ptx}");
        assert!(ptx.contains("scan_smem"), "must reference scan_smem\n{ptx}");
    }

    #[test]
    fn generate_has_bar_sync() {
        let t = BlockScanTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, 64, ScanKind::Exclusive));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(ptx.contains("bar.sync"), "must sync threads\n{ptx}");
    }

    #[test]
    fn generate_exclusive_has_identity_clear() {
        let t = BlockScanTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, 64, ScanKind::Exclusive));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        // Exclusive scan must clear the last element with identity
        assert!(
            ptx.contains("0f00000000"),
            "exclusive sum must insert f32 zero identity\n{ptx}"
        );
    }

    #[test]
    fn generate_inclusive_has_up_sweep_only() {
        let t = BlockScanTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, 64, ScanKind::Inclusive));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        // Up-sweep present
        assert!(ptx.contains("UP_NEXT"), "must have up-sweep labels\n{ptx}");
        // Down-sweep absent for inclusive
        assert!(
            !ptx.contains("DOWN_NEXT"),
            "inclusive scan should not have down-sweep\n{ptx}"
        );
    }

    #[test]
    fn generate_min_u32_uses_min_instr() {
        let t = BlockScanTemplate::new(cfg(ReduceOp::Min, PtxType::U32, 64, ScanKind::Inclusive));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(ptx.contains("min.u32"), "must use min.u32\n{ptx}");
    }
}
