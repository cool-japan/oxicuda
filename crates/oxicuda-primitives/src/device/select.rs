//! Device-wide stream compaction — keep elements that satisfy a predicate.
//!
//! The full pipeline uses three GPU passes:
//!
//! 1. **Flag** — write a `u32` flag (`1` = keep, `0` = discard) for each element.
//! 2. **Exclusive scan** — compute prefix sums of the flag array to obtain the
//!    output index for each selected element.  Run this step with
//!    [`crate::device::scan::DeviceScanTemplate`] using `ReduceOp::Sum` and
//!    `ScanKind::Exclusive`.
//! 3. **Gather** — scatter input elements to their compact output positions.
//!
//! # Supported predicates
//!
//! | [`SelectPredicate`]  | Condition kept                              |
//! |---------------------|---------------------------------------------|
//! | `NonZero`           | element != 0                                |
//! | `Positive`          | element > 0 (unsigned: same as `NonZero`)   |
//! | `Negative`          | element < 0 (unsigned: always empty result) |
//! | `FlagArray`         | caller-supplied `u32` flag array is nonzero  |
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::device::select::{
//!     DeviceSelectConfig, DeviceSelectTemplate, SelectPredicate,
//! };
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = DeviceSelectConfig {
//!     ty: PtxType::F32,
//!     pred: SelectPredicate::Positive,
//!     block_size: 256,
//! };
//! let t = DeviceSelectTemplate::new(cfg);
//! let (flag_ptx, gather_ptx) = t.generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(flag_ptx.contains("device_select_flag_positive_f32_bs256"));
//! assert!(gather_ptx.contains("device_select_gather_positive_f32_bs256"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::ptx_helpers::{ptx_header, ptx_type_str};

// ─── SelectPredicate ─────────────────────────────────────────────────────────

/// Predicate controlling which elements are retained during stream compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectPredicate {
    /// Keep elements whose value is not equal to zero.
    NonZero,
    /// Keep elements strictly greater than zero.
    ///
    /// For unsigned integer types, equivalent to `NonZero`.
    Positive,
    /// Keep elements strictly less than zero.
    ///
    /// For unsigned integer types, always produces an empty result.
    Negative,
    /// Keep elements where the caller-supplied `u32` flag array is nonzero at
    /// the same index.  The flag kernel normalises the flag array to `{0, 1}`.
    FlagArray,
}

impl SelectPredicate {
    /// Short name used in generated kernel identifiers.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::NonZero => "nonzero",
            Self::Positive => "positive",
            Self::Negative => "negative",
            Self::FlagArray => "flagged",
        }
    }

    /// PTX `setp` instruction that writes `%select_pred`.
    ///
    /// For `FlagArray` the comparison is performed against `%flag_u32`
    /// (a `u32`-typed register); for all other predicates the comparison is
    /// against `%val` (of the configured element type).
    fn comparison_ptx(self, ty: PtxType) -> String {
        let ty_str = ptx_type_str(ty);
        let zero = ptx_zero_literal(ty);
        let is_unsigned = matches!(ty, PtxType::U32 | PtxType::U64);

        match self {
            Self::NonZero => {
                format!("    setp.ne.{ty_str} %select_pred, %val, {zero};")
            }
            Self::Positive => {
                if is_unsigned {
                    format!("    setp.ne.{ty_str} %select_pred, %val, {zero};")
                } else {
                    format!("    setp.gt.{ty_str} %select_pred, %val, {zero};")
                }
            }
            Self::Negative => {
                if is_unsigned {
                    // Unsigned values are never negative — predicate is always false.
                    // setp.ne.u32 p, 0, 0  evaluates 0 ≠ 0  → false.
                    "    setp.ne.u32 %select_pred, 0, 0;".to_string()
                } else {
                    format!("    setp.lt.{ty_str} %select_pred, %val, {zero};")
                }
            }
            Self::FlagArray => "    setp.ne.u32 %select_pred, %flag_u32, 0;".to_string(),
        }
    }
}

/// PTX immediate zero literal for the given element type.
fn ptx_zero_literal(ty: PtxType) -> &'static str {
    match ty {
        PtxType::F32 => "0f00000000",
        PtxType::F64 => "0d0000000000000000",
        _ => "0",
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for device-wide stream compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceSelectConfig {
    /// Element type of the input and output data arrays.
    pub ty: PtxType,
    /// Predicate that determines which elements are retained.
    pub pred: SelectPredicate,
    /// Threads per block (power of 2, `32`–`1024`).
    pub block_size: u32,
}

impl DeviceSelectConfig {
    /// Kernel name for the flag-generation pass.
    ///
    /// For `FlagArray` the name encodes the flag-read type (`u32`) and the data
    /// element type separately, e.g. `device_select_flag_flagarray_u32_f32_bs256`.
    /// For all other predicates only the element type appears:
    /// `device_select_flag_positive_f32_bs256`.
    #[must_use]
    pub fn flag_kernel_name(&self) -> String {
        if self.pred == SelectPredicate::FlagArray {
            format!(
                "device_select_flag_flagarray_u32_{}_bs{}",
                ptx_type_str(self.ty),
                self.block_size,
            )
        } else {
            format!(
                "device_select_flag_{}_{}_bs{}",
                self.pred.name(),
                ptx_type_str(self.ty),
                self.block_size,
            )
        }
    }

    /// Kernel name for the gather (scatter-to-compact-output) pass.
    #[must_use]
    pub fn gather_kernel_name(&self) -> String {
        format!(
            "device_select_gather_{}_{}_bs{}",
            self.pred.name(),
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

// ─── Template ────────────────────────────────────────────────────────────────

/// PTX code generator for device-wide stream compaction.
///
/// Produces two PTX strings: the **flag** kernel and the **gather** kernel.
/// Between them the caller must run an exclusive prefix scan (sum) on the
/// `u32` flag array; the scan output provides the output indices used by the
/// gather kernel.
pub struct DeviceSelectTemplate {
    /// Configuration.
    pub cfg: DeviceSelectConfig,
}

impl DeviceSelectTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: DeviceSelectConfig) -> Self {
        Self { cfg }
    }

    /// Generate both PTX kernels as `(flag_ptx, gather_ptx)`.
    ///
    /// # Errors
    ///
    /// Returns a `String` error message on generation failure.
    pub fn generate(&self, sm: SmVersion) -> Result<(String, String), String> {
        let flag = self.generate_flag_kernel(sm)?;
        let gather = self.generate_gather_kernel(sm)?;
        Ok((flag, gather))
    }

    // ── Kernel 1: materialise per-element flag array ─────────────────────────

    fn generate_flag_kernel(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.flag_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let is_64bit = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        let eb: u32 = if is_64bit { 8 } else { 4 };
        let cmp = self.cfg.pred.comparison_ptx(self.cfg.ty);
        let is_flagarr = self.cfg.pred == SelectPredicate::FlagArray;

        let mut out = ptx_header(sm);

        if is_flagarr {
            // Flag kernel reads an external u32 flag array, normalises to 0/1.
            writeln!(
                out,
                ".visible .entry {name}(\n    \
                 .param .u64 param_out_flags,\n    \
                 .param .u64 param_in_flags,\n    \
                 .param .u64 param_n\n)"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "{{").map_err(|e| e.to_string())?;
            writeln!(out, "    .reg .u32    %flag_u32, %out_flag, %tid, %bid;")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    .reg .u64    %n, %gid, %ptr_in, %ptr_out, %addr;")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    .reg .pred   %p, %select_pred;").map_err(|e| e.to_string())?;
            writeln!(out, "    ld.param.u64 %ptr_out, [param_out_flags];")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    ld.param.u64 %ptr_in,  [param_in_flags];")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    ld.param.u64 %n,        [param_n];").map_err(|e| e.to_string())?;
            writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(|e| e.to_string())?;
            writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(|e| e.to_string())?;
            writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(|e| e.to_string())?;
            writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p ret;").map_err(|e| e.to_string())?;
            writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr_in;")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    ld.global.u32 %flag_u32, [%addr];").map_err(|e| e.to_string())?;
        } else {
            // Flag kernel reads the typed input data and applies a comparison.
            writeln!(
                out,
                ".visible .entry {name}(\n    \
                 .param .u64 param_flags,\n    \
                 .param .u64 param_input,\n    \
                 .param .u64 param_n\n)"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "{{").map_err(|e| e.to_string())?;
            writeln!(out, "    .reg .{ty}   %val;").map_err(|e| e.to_string())?;
            writeln!(out, "    .reg .u32    %out_flag, %tid, %bid;").map_err(|e| e.to_string())?;
            writeln!(out, "    .reg .u64    %n, %gid, %ptr_in, %ptr_out, %addr;")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    .reg .pred   %p, %select_pred;").map_err(|e| e.to_string())?;
            writeln!(out, "    ld.param.u64 %ptr_out, [param_flags];")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    ld.param.u64 %n,        [param_n];").map_err(|e| e.to_string())?;
            writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(|e| e.to_string())?;
            writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(|e| e.to_string())?;
            writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(|e| e.to_string())?;
            writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p ret;").map_err(|e| e.to_string())?;
            writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    ld.global.{ty} %val, [%addr];").map_err(|e| e.to_string())?;
        }

        // Apply the predicate comparison.
        writeln!(out, "{cmp}").map_err(|e| e.to_string())?;
        writeln!(out, "    selp.u32     %out_flag, 1, 0, %select_pred;")
            .map_err(|e| e.to_string())?;

        // Write flag to output flag array (always u32, stride 4).
        if is_flagarr {
            writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr_out;")
                .map_err(|e| e.to_string())?;
        } else {
            writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr_out;")
                .map_err(|e| e.to_string())?;
        }
        writeln!(out, "    st.global.u32 [%addr], %out_flag;").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    // ── Kernel 2: scatter selected elements to the compact output ────────────

    fn generate_gather_kernel(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.gather_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let is_64bit = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        let eb: u32 = if is_64bit { 8 } else { 4 };

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_output,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_flags,\n    \
             .param .u64 param_offsets,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %val;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %flag, %tid, %bid;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %n, %gid, %offset;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %ptr_in, %ptr_out, %ptr_flags, %ptr_off, %addr;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p, %keep;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_out,   [param_output];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,    [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_flags, [param_flags];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_off,   [param_offsets];")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %n,          [param_n];").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(|e| e.to_string())?;

        // Out-of-bounds guard.
        writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p ret;").map_err(|e| e.to_string())?;

        // Load flag and early-exit if not selected.
        writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr_flags;").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.global.u32 %flag, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ne.u32  %keep, %flag, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @!%keep ret;").map_err(|e| e.to_string())?;

        // Load output index from the exclusive scan of the flag array (u64).
        writeln!(out, "    mad.lo.u64   %addr, %gid, 8, %ptr_off;").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.global.u64 %offset, [%addr];").map_err(|e| e.to_string())?;

        // Load input element.
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.global.{ty} %val, [%addr];").map_err(|e| e.to_string())?;

        // Scatter to compact output.
        writeln!(out, "    mad.lo.u64   %addr, %offset, {eb}, %ptr_out;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.{ty} [%addr], %val;").map_err(|e| e.to_string())?;
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

    fn cfg(ty: PtxType, pred: SelectPredicate) -> DeviceSelectConfig {
        DeviceSelectConfig {
            ty,
            pred,
            block_size: 256,
        }
    }

    #[test]
    fn flag_kernel_name_contains_pred_and_type() {
        let c = cfg(PtxType::F32, SelectPredicate::Positive);
        let n = c.flag_kernel_name();
        assert!(n.contains("positive"), "{n}");
        assert!(n.contains("f32"), "{n}");
        assert!(n.contains("bs256"), "{n}");
    }

    #[test]
    fn gather_kernel_name_contains_pred_and_type() {
        let c = cfg(PtxType::U32, SelectPredicate::NonZero);
        let n = c.gather_kernel_name();
        assert!(n.contains("nonzero"), "{n}");
        assert!(n.contains("u32"), "{n}");
    }

    #[test]
    fn flag_ptx_nonzero_f32_contains_setp_ne() {
        let t = DeviceSelectTemplate::new(cfg(PtxType::F32, SelectPredicate::NonZero));
        let ptx = t.generate_flag_kernel(SmVersion::Sm80).unwrap();
        assert!(ptx.contains("setp.ne.f32"), "PTX: {ptx}");
        assert!(ptx.contains("0f00000000"), "PTX: {ptx}");
    }

    #[test]
    fn flag_ptx_positive_s32_contains_setp_gt() {
        let t = DeviceSelectTemplate::new(cfg(PtxType::S32, SelectPredicate::Positive));
        let ptx = t.generate_flag_kernel(SmVersion::Sm80).unwrap();
        assert!(ptx.contains("setp.gt.s32"), "PTX: {ptx}");
    }

    #[test]
    fn flag_ptx_negative_u32_is_always_false() {
        let t = DeviceSelectTemplate::new(cfg(PtxType::U32, SelectPredicate::Negative));
        let ptx = t.generate_flag_kernel(SmVersion::Sm80).unwrap();
        // Unsigned Negative uses setp.ne.u32 0,0 → always false.
        assert!(ptx.contains("setp.ne.u32 %select_pred, 0, 0"), "PTX: {ptx}");
    }

    #[test]
    fn flag_ptx_flagarray_reads_u32() {
        let t = DeviceSelectTemplate::new(cfg(PtxType::F32, SelectPredicate::FlagArray));
        let ptx = t.generate_flag_kernel(SmVersion::Sm80).unwrap();
        assert!(ptx.contains("param_in_flags"), "PTX: {ptx}");
        assert!(ptx.contains("ld.global.u32"), "PTX: {ptx}");
        assert!(ptx.contains("setp.ne.u32"), "PTX: {ptx}");
    }

    #[test]
    fn gather_ptx_loads_flag_and_offset() {
        let t = DeviceSelectTemplate::new(cfg(PtxType::F32, SelectPredicate::NonZero));
        let ptx = t.generate_gather_kernel(SmVersion::Sm80).unwrap();
        assert!(ptx.contains("param_flags"), "PTX: {ptx}");
        assert!(ptx.contains("param_offsets"), "PTX: {ptx}");
        assert!(ptx.contains("ld.global.u32"), "PTX: {ptx}");
        assert!(ptx.contains("ld.global.u64"), "PTX: {ptx}");
        assert!(ptx.contains("ld.global.f32"), "PTX: {ptx}");
        assert!(ptx.contains("st.global.f32"), "PTX: {ptx}");
    }

    #[test]
    fn generate_both_kernels_succeeds() {
        let t = DeviceSelectTemplate::new(cfg(PtxType::U32, SelectPredicate::NonZero));
        let (flag_ptx, gather_ptx) = t.generate(SmVersion::Sm80).unwrap();
        assert!(!flag_ptx.is_empty());
        assert!(!gather_ptx.is_empty());
    }

    #[test]
    fn gather_ptx_f64_uses_8byte_stride() {
        let t = DeviceSelectTemplate::new(cfg(PtxType::F64, SelectPredicate::NonZero));
        let ptx = t.generate_gather_kernel(SmVersion::Sm80).unwrap();
        assert!(ptx.contains("ld.global.f64"), "PTX: {ptx}");
        assert!(ptx.contains("st.global.f64"), "PTX: {ptx}");
    }

    #[test]
    fn pred_names_unique() {
        use std::collections::HashSet;
        let names: HashSet<_> = [
            SelectPredicate::NonZero,
            SelectPredicate::Positive,
            SelectPredicate::Negative,
            SelectPredicate::FlagArray,
        ]
        .iter()
        .map(|p| p.name())
        .collect();
        assert_eq!(names.len(), 4);
    }
}
