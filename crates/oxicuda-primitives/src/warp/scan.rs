//! Warp-level prefix scan using PTX shuffle-up instructions.
//!
//! A warp scan computes the prefix (inclusive or exclusive) of an operator
//! across all 32 threads using `shfl.sync.up.b32` — a shift-up shuffle that
//! propagates values from lower-numbered lanes to higher-numbered lanes.
//!
//! # Algorithm (Hillis-Steele)
//!
//! For an inclusive scan (prefix sum for example):
//!
//! ```text
//! Round 0 (offset=1):  lane i receives lane (i-1)'s value → add to own value
//! Round 1 (offset=2):  lane i receives lane (i-2)'s value → add to own value
//! Round 2 (offset=4):  lane i receives lane (i-4)'s value → add to own value
//! Round 3 (offset=8):  lane i receives lane (i-8)'s value → add to own value
//! Round 4 (offset=16): lane i receives lane(i-16)'s value → add to own value
//! ```
//!
//! After 5 rounds, lane `i` holds `input[0] ⊕ input[1] ⊕ … ⊕ input[i]`.
//!
//! For an exclusive scan the identity value is shifted in at lane 0, and lane
//! `i` holds `input[0] ⊕ … ⊕ input[i-1]` (lane 0 gets the identity).
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::warp::scan::{WarpScanTemplate, WarpScanConfig, ScanKind};
//! use oxicuda_primitives::ptx_helpers::ReduceOp;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = WarpScanConfig {
//!     op: ReduceOp::Sum,
//!     ty: PtxType::F32,
//!     kind: ScanKind::Inclusive,
//! };
//! let ptx = WarpScanTemplate::new(cfg)
//!     .generate(SmVersion::Sm80)
//!     .expect("PTX gen failed");
//! assert!(ptx.contains("warp_scan_sum_f32_inclusive"));
//! assert!(ptx.contains("shfl.sync.up"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::ptx_helpers::{ReduceOp, ptx_header, ptx_type_str};

// ─── ScanKind ────────────────────────────────────────────────────────────────

/// Whether the scan produces inclusive or exclusive output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanKind {
    /// `result[i] = op(input[0], …, input[i])` — includes the current element.
    Inclusive,
    /// `result[i] = op(input[0], …, input[i-1])` — excludes the current element.
    ///
    /// Lane 0 always receives the operator identity.
    Exclusive,
}

impl ScanKind {
    /// Short name for use in kernel names.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Inclusive => "inclusive",
            Self::Exclusive => "exclusive",
        }
    }
}

// ─── WarpScanConfig ───────────────────────────────────────────────────────────

/// Configuration for a warp-level scan kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WarpScanConfig {
    /// The scan operation.
    pub op: ReduceOp,
    /// Element type.
    pub ty: PtxType,
    /// Inclusive or exclusive scan.
    pub kind: ScanKind,
}

impl WarpScanConfig {
    /// Canonical kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!(
            "warp_scan_{}_{}_{}",
            self.op.name(),
            ptx_type_str(self.ty),
            self.kind.name()
        )
    }
}

// ─── WarpScanTemplate ────────────────────────────────────────────────────────

/// PTX code generator for a warp-level prefix scan kernel.
pub struct WarpScanTemplate {
    /// Kernel configuration.
    pub cfg: WarpScanConfig,
}

impl WarpScanTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: WarpScanConfig) -> Self {
        Self { cfg }
    }

    /// Generate the PTX source for this warp scan.
    ///
    /// # Errors
    ///
    /// Returns a string describing the error if PTX generation fails.
    pub fn generate(&self, sm: SmVersion) -> Result<String, String> {
        let is_64bit = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        if is_64bit {
            self.generate_64bit(sm)
        } else {
            self.generate_32bit(sm)
        }
    }

    // ── 32-bit path ────────────────────────────────────────────────────────

    fn generate_32bit(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let op = self.cfg.op.ptx_instr(self.cfg.ty);
        let is_exclusive = self.cfg.kind == ScanKind::Exclusive;

        // Identity for the exclusive scan's lane-0 initial value
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

        writeln!(out, "    .reg .{ty}   %val, %shfl;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %tid, %n, %mask, %laneid, %offset;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %ptr_in, %ptr_out, %addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p, %q;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_out, [param_output];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %n,        [param_n];").map_err(|e| e.to_string())?;

        writeln!(out, "    mov.u32 %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    and.b32 %laneid, %tid, 31;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32 %mask, 0xFFFFFFFF;").map_err(|e| e.to_string())?;

        writeln!(out, "    setp.ge.u32 %p, %tid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %addr, %tid, 4, %ptr_in;").map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.global.{ty} %val, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;

        // For exclusive scan: save original value, shift down by one lane
        if is_exclusive {
            writeln!(
                out,
                "    // Exclusive: shift val right by 1 lane (insert identity at lane 0)"
            )
            .map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    shfl.sync.up.b32 %shfl, %val, 1, 0, %mask;  // src from lane i-1"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    setp.ne.u32 %q, %laneid, 0;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%q  mov.{ty} %val, %shfl;").map_err(|e| e.to_string())?;
            writeln!(out, "    @!%q mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;
        }

        // Hillis-Steele rounds with shfl.sync.up
        for offset in [1u32, 2, 4, 8, 16] {
            writeln!(out, "    mov.u32 %offset, {offset};").map_err(|e| e.to_string())?;
            // shfl.sync.up: dst ← src from lane (laneid - offset) if laneid >= offset, else src unchanged
            writeln!(out, "    shfl.sync.up.b32 %shfl, %val, %offset, 0, %mask;")
                .map_err(|e| e.to_string())?;
            // Only apply op if this lane received a valid predecessor (laneid >= offset)
            writeln!(out, "    setp.ge.u32 %q, %laneid, %offset;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%q {op} %val, %val, %shfl;").map_err(|e| e.to_string())?;
        }

        // Write output
        writeln!(out, "    setp.lt.u32 %p, %tid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %addr, %tid, 4, %ptr_out;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p st.global.{ty} [%addr], %val;").map_err(|e| e.to_string())?;

        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    // ── 64-bit path (split lo/hi) ──────────────────────────────────────────

    fn generate_64bit(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let op = self.cfg.op.ptx_instr(self.cfg.ty);
        let is_exclusive = self.cfg.kind == ScanKind::Exclusive;

        let mut out = ptx_header(sm);

        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_output,\n    \
             .param .u64 param_input,\n    \
             .param .u32 param_n\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;

        writeln!(out, "    .reg .{ty}   %val, %shfl64;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %lo, %hi, %shfl_lo, %shfl_hi;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %tid, %n, %mask, %laneid, %offset;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %ptr_in, %ptr_out, %addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p, %q;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_out, [param_output];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %n,        [param_n];").map_err(|e| e.to_string())?;

        writeln!(out, "    mov.u32 %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    and.b32 %laneid, %tid, 31;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32 %mask, 0xFFFFFFFF;").map_err(|e| e.to_string())?;

        writeln!(out, "    setp.ge.u32 %p, %tid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %addr, %tid, 8, %ptr_in;").map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.global.{ty} %val, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.b64 {{%lo, %hi}}, %val;").map_err(|e| e.to_string())?;

        if is_exclusive {
            writeln!(out, "    shfl.sync.up.b32 %shfl_lo, %lo, 1, 0, %mask;")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    shfl.sync.up.b32 %shfl_hi, %hi, 1, 0, %mask;")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    setp.ne.u32 %q, %laneid, 0;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%q  mov.b64 %val, {{%shfl_lo, %shfl_hi}};")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    @!%q mov.{ty} %val, 0;").map_err(|e| e.to_string())?;
            writeln!(out, "    mov.b64 {{%lo, %hi}}, %val;").map_err(|e| e.to_string())?;
        }

        for offset in [1u32, 2, 4, 8, 16] {
            writeln!(out, "    mov.u32 %offset, {offset};").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    shfl.sync.up.b32 %shfl_lo, %lo, %offset, 0, %mask;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    shfl.sync.up.b32 %shfl_hi, %hi, %offset, 0, %mask;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    mov.b64 %shfl64, {{%shfl_lo, %shfl_hi}};")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    setp.ge.u32 %q, %laneid, %offset;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%q {op} %val, %val, %shfl64;").map_err(|e| e.to_string())?;
            writeln!(out, "    mov.b64 {{%lo, %hi}}, %val;").map_err(|e| e.to_string())?;
        }

        writeln!(out, "    setp.lt.u32 %p, %tid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %addr, %tid, 8, %ptr_out;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p st.global.{ty} [%addr], %val;").map_err(|e| e.to_string())?;

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

    fn make(op: ReduceOp, ty: PtxType, kind: ScanKind) -> WarpScanTemplate {
        WarpScanTemplate::new(WarpScanConfig { op, ty, kind })
    }

    #[test]
    fn kernel_name_inclusive_sum_f32() {
        let t = make(ReduceOp::Sum, PtxType::F32, ScanKind::Inclusive);
        assert_eq!(t.cfg.kernel_name(), "warp_scan_sum_f32_inclusive");
    }

    #[test]
    fn kernel_name_exclusive_min_u32() {
        let t = make(ReduceOp::Min, PtxType::U32, ScanKind::Exclusive);
        assert_eq!(t.cfg.kernel_name(), "warp_scan_min_u32_exclusive");
    }

    #[test]
    fn generate_contains_shfl_up() {
        let t = make(ReduceOp::Sum, PtxType::F32, ScanKind::Inclusive);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(ptx.contains("shfl.sync.up.b32"), "must use shfl.up\n{ptx}");
    }

    #[test]
    fn generate_five_rounds_inclusive() {
        let t = make(ReduceOp::Sum, PtxType::F32, ScanKind::Inclusive);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        for offset in [1, 2, 4, 8, 16] {
            assert!(
                ptx.contains(&format!("mov.u32 %offset, {offset};")),
                "missing offset={offset}\n{ptx}"
            );
        }
    }

    #[test]
    fn generate_exclusive_inserts_identity_at_lane0() {
        let t = make(ReduceOp::Sum, PtxType::F32, ScanKind::Exclusive);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        // Exclusive scan must contain a lane-0 identity insertion
        assert!(
            ptx.contains("setp.ne.u32 %q, %laneid, 0"),
            "missing lane-0 guard\n{ptx}"
        );
    }

    #[test]
    fn generate_max_f64_splits_lo_hi() {
        let t = make(ReduceOp::Max, PtxType::F64, ScanKind::Inclusive);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(ptx.contains("%lo"), "must have %lo\n{ptx}");
        assert!(ptx.contains("%hi"), "must have %hi\n{ptx}");
    }

    #[test]
    fn generate_product_uses_mul() {
        let t = make(ReduceOp::Product, PtxType::F32, ScanKind::Inclusive);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(ptx.contains("mul.f32"), "must use mul\n{ptx}");
    }

    #[test]
    fn generate_min_uses_min_instr() {
        let t = make(ReduceOp::Min, PtxType::U32, ScanKind::Inclusive);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(ptx.contains("min.u32"), "must use min\n{ptx}");
    }

    #[test]
    fn scan_kind_names_distinct() {
        assert_ne!(ScanKind::Inclusive.name(), ScanKind::Exclusive.name());
    }

    #[test]
    fn config_hash_equality() {
        use std::collections::HashSet;
        let cfg1 = WarpScanConfig {
            op: ReduceOp::Sum,
            ty: PtxType::F32,
            kind: ScanKind::Inclusive,
        };
        let cfg2 = WarpScanConfig {
            op: ReduceOp::Sum,
            ty: PtxType::F32,
            kind: ScanKind::Inclusive,
        };
        let mut set = HashSet::new();
        set.insert(cfg1);
        assert!(set.contains(&cfg2));
    }
}
