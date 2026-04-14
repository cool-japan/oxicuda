//! Warp-level parallel reduction using PTX shuffle instructions.
//!
//! A warp reduction computes a single aggregate (sum, max, min, product, …)
//! across all 32 threads in a CUDA warp using `shfl.sync.bfly` — a
//! butterfly-network XOR shuffle that requires only `log2(32) = 5` rounds.
//!
//! # Algorithm
//!
//! Each round halves the active thread count by folding the upper half into
//! the lower half:
//!
//! ```text
//! Round 0 (offset=16): lane i  ← op(lane i, lane i^16)   [threads 0..15 hold]
//! Round 1 (offset= 8): lane i  ← op(lane i, lane i^ 8)   [threads 0.. 7 hold]
//! Round 2 (offset= 4): lane i  ← op(lane i, lane i^ 4)   [threads 0.. 3 hold]
//! Round 3 (offset= 2): lane i  ← op(lane i, lane i^ 2)   [threads 0.. 1 hold]
//! Round 4 (offset= 1): lane i  ← op(lane i, lane i^ 1)   [thread  0 holds result]
//! ```
//!
//! After 5 rounds, lane 0 holds the warp aggregate.  The optional
//! "broadcast" step uses `shfl.sync.idx` to broadcast lane 0's result to
//! all lanes, so every thread in the warp sees the final value.
//!
//! # PTX generation
//!
//! The kernels in this module are expressed as PTX source strings that use
//! `shfl.sync.bfly.b32` (sm_80+ with `.full_mask`) for 32-bit types and a
//! composite shuffle + combine for 64-bit types.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::warp::reduce::{WarpReduceTemplate, WarpReduceConfig};
//! use oxicuda_primitives::ptx_helpers::ReduceOp;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = WarpReduceConfig {
//!     op: ReduceOp::Sum,
//!     ty: PtxType::F32,
//!     broadcast: true,
//! };
//! let template = WarpReduceTemplate::new(cfg);
//! let ptx = template.generate(SmVersion::Sm80).expect("PTX gen failed");
//! assert!(ptx.contains("warp_reduce_sum_f32"));
//! assert!(ptx.contains("shfl.sync.bfly"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::ptx_helpers::{ReduceOp, ptx_header, ptx_type_str};

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for a warp-level reduction kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WarpReduceConfig {
    /// The combining operation (sum, min, max, …).
    pub op: ReduceOp,
    /// Element type.
    pub ty: PtxType,
    /// If `true`, lane 0's result is broadcast to all 32 lanes.
    pub broadcast: bool,
}

impl WarpReduceConfig {
    /// Canonical kernel name derived from this configuration.
    ///
    /// E.g. `"warp_reduce_sum_f32"` or `"warp_reduce_max_u32_bcast"`.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let suffix = if self.broadcast { "_bcast" } else { "" };
        format!(
            "warp_reduce_{}_{}{}",
            self.op.name(),
            ptx_type_str(self.ty),
            suffix
        )
    }
}

// ─── Template ────────────────────────────────────────────────────────────────

/// PTX code generator for a warp-level reduction kernel.
pub struct WarpReduceTemplate {
    /// Kernel configuration.
    pub cfg: WarpReduceConfig,
}

impl WarpReduceTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: WarpReduceConfig) -> Self {
        Self { cfg }
    }

    /// Generate the PTX source for the configured warp reduction.
    ///
    /// # Errors
    ///
    /// Returns a string describing the error if PTX generation fails (should
    /// only occur for unsupported type combinations).
    pub fn generate(&self, sm: SmVersion) -> Result<String, String> {
        let is_64bit = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        if is_64bit {
            self.generate_64bit(sm)
        } else {
            self.generate_32bit(sm)
        }
    }

    // ── 32-bit path (uses shfl.sync.bfly.b32) ─────────────────────────────

    fn generate_32bit(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let op = self.cfg.op.ptx_instr(self.cfg.ty);
        let broadcast = self.cfg.broadcast;

        let mut out = ptx_header(sm);

        // Kernel signature: (T* result, const T* input, u32 n)
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_result,\n    \
             .param .u64 param_input,\n    \
             .param .u32 param_n\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;

        // Register declarations
        writeln!(out, "    .reg .{ty}   %val;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %shfl;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %tid, %n, %mask, %laneid;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %ptr_in, %ptr_out, %addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %offset;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;

        // Load parameters
        writeln!(out, "    ld.param.u64 %ptr_out, [param_result];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %n,        [param_n];").map_err(|e| e.to_string())?;

        // Compute lane ID and global thread ID
        writeln!(out, "    mov.u32 %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    and.b32 %laneid, %tid, 31;   // lane = tid & 31")
            .map_err(|e| e.to_string())?;

        // Load input element (guard: if tid >= n load identity)
        // Identity for the operation
        let identity = match self.cfg.op {
            ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor => {
                if matches!(self.cfg.ty, PtxType::F32) {
                    "0f00000000"
                } else {
                    "0"
                }
            }
            ReduceOp::Product => {
                if matches!(self.cfg.ty, PtxType::F32) {
                    "0f3F800000"
                } else {
                    "1"
                }
            }
            ReduceOp::Min => {
                if matches!(self.cfg.ty, PtxType::F32) {
                    "0x7F800000"
                } else {
                    "0x7FFFFFFF"
                }
            }
            ReduceOp::Max => {
                if matches!(self.cfg.ty, PtxType::F32) {
                    "0xFF800000"
                } else {
                    "0x80000000"
                }
            }
            ReduceOp::And => "0xFFFFFFFF",
        };
        writeln!(out, "    setp.ge.u32  %p, %tid, %n;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u64   %addr, %tid, {}, %ptr_in;",
            std::mem::size_of::<f32>()
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.global.{ty} %val, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;

        // Full warp mask
        writeln!(out, "    mov.u32 %mask, 0xFFFFFFFF;").map_err(|e| e.to_string())?;

        // 5 rounds of butterfly shuffle-and-reduce
        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(out, "    mov.u32 %offset, {offset};").map_err(|e| e.to_string())?;
            // shfl.sync.bfly.b32 dst, src, offset, mask_and_clamp, membermask
            // clamp = 31 (0x1F) means warp-size-1
            writeln!(
                out,
                "    shfl.sync.bfly.b32 %shfl, %val, %offset, 31, %mask;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    {op} %val, %val, %shfl;").map_err(|e| e.to_string())?;
        }

        // Optional broadcast: every lane gets lane-0's result
        if broadcast {
            writeln!(
                out,
                "    shfl.sync.idx.b32 %val, %val, 0, 31, %mask;   // broadcast"
            )
            .map_err(|e| e.to_string())?;
        }

        // Write result from lane 0 (or all lanes if broadcast)
        if broadcast {
            writeln!(
                out,
                "    mad.lo.u64 %addr, %tid, {}, %ptr_out;",
                std::mem::size_of::<f32>()
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    st.global.{ty} [%addr], %val;").map_err(|e| e.to_string())?;
        } else {
            writeln!(out, "    setp.ne.u32 %p, %laneid, 0;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p bra SKIP_{name};  // only lane 0 writes")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    st.global.{ty} [%ptr_out], %val;").map_err(|e| e.to_string())?;
            writeln!(out, "SKIP_{name}:").map_err(|e| e.to_string())?;
        }

        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    // ── 64-bit path (split into two 32-bit shuffles) ───────────────────────

    fn generate_64bit(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let op = self.cfg.op.ptx_instr(self.cfg.ty);

        let mut out = ptx_header(sm);

        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_result,\n    \
             .param .u64 param_input,\n    \
             .param .u32 param_n\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;

        writeln!(out, "    .reg .{ty}   %val;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %lo, %hi, %shfl_lo, %shfl_hi;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %shfl64;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %tid, %n, %mask, %laneid, %offset;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %ptr_in, %ptr_out, %addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_out, [param_result];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %n,        [param_n];").map_err(|e| e.to_string())?;

        writeln!(out, "    mov.u32 %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    and.b32 %laneid, %tid, 31;").map_err(|e| e.to_string())?;

        writeln!(out, "    setp.ge.u32 %p, %tid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %addr, %tid, 8, %ptr_in;").map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.global.{ty} %val, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, 0;").map_err(|e| e.to_string())?;

        writeln!(out, "    mov.u32 %mask, 0xFFFFFFFF;").map_err(|e| e.to_string())?;
        // Split 64-bit value into lo/hi 32-bit words
        writeln!(out, "    mov.b64 {{%lo, %hi}}, %val;").map_err(|e| e.to_string())?;

        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(out, "    mov.u32 %offset, {offset};").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    shfl.sync.bfly.b32 %shfl_lo, %lo, %offset, 31, %mask;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    shfl.sync.bfly.b32 %shfl_hi, %hi, %offset, 31, %mask;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    mov.b64 %shfl64, {{%shfl_lo, %shfl_hi}};")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    {op} %val, %val, %shfl64;").map_err(|e| e.to_string())?;
            writeln!(out, "    mov.b64 {{%lo, %hi}}, %val;").map_err(|e| e.to_string())?;
        }

        writeln!(out, "    setp.ne.u32 %p, %laneid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra SKIP_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.{ty} [%ptr_out], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "SKIP_{name}:").map_err(|e| e.to_string())?;
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

    fn make_template(op: ReduceOp, ty: PtxType, broadcast: bool) -> WarpReduceTemplate {
        WarpReduceTemplate::new(WarpReduceConfig { op, ty, broadcast })
    }

    #[test]
    fn kernel_name_sum_f32() {
        let t = make_template(ReduceOp::Sum, PtxType::F32, false);
        assert_eq!(t.cfg.kernel_name(), "warp_reduce_sum_f32");
    }

    #[test]
    fn kernel_name_max_u32_broadcast() {
        let t = make_template(ReduceOp::Max, PtxType::U32, true);
        assert_eq!(t.cfg.kernel_name(), "warp_reduce_max_u32_bcast");
    }

    #[test]
    fn generate_sum_f32_sm80_contains_kernel() {
        let t = make_template(ReduceOp::Sum, PtxType::F32, false);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX generation failed");
        assert!(
            ptx.contains("warp_reduce_sum_f32"),
            "missing kernel name\n{ptx}"
        );
        assert!(ptx.contains(".target sm_80"), "missing sm target\n{ptx}");
    }

    #[test]
    fn generate_uses_shfl_sync_bfly() {
        let t = make_template(ReduceOp::Sum, PtxType::F32, false);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen failed");
        assert!(
            ptx.contains("shfl.sync.bfly.b32"),
            "must use warp shuffle\n{ptx}"
        );
    }

    #[test]
    fn generate_five_shuffle_rounds() {
        let t = make_template(ReduceOp::Sum, PtxType::F32, false);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen failed");
        // 5 different offsets: 16, 8, 4, 2, 1
        for offset in [16, 8, 4, 2, 1] {
            assert!(
                ptx.contains(&format!("mov.u32 %offset, {offset};")),
                "missing offset={offset}\n{ptx}"
            );
        }
    }

    #[test]
    fn generate_broadcast_contains_idx_shfl() {
        let t = make_template(ReduceOp::Sum, PtxType::F32, true);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen failed");
        assert!(
            ptx.contains("shfl.sync.idx.b32"),
            "broadcast requires idx shuffle\n{ptx}"
        );
    }

    #[test]
    fn generate_min_u32_uses_min_instr() {
        let t = make_template(ReduceOp::Min, PtxType::U32, false);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen failed");
        assert!(ptx.contains("min.u32"), "must use min.u32\n{ptx}");
    }

    #[test]
    fn generate_max_f32_uses_max_instr() {
        let t = make_template(ReduceOp::Max, PtxType::F32, false);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen failed");
        assert!(ptx.contains("max.f32"), "must use max.f32\n{ptx}");
    }

    #[test]
    fn generate_sum_f64_splits_into_32bit_shuffles() {
        let t = make_template(ReduceOp::Sum, PtxType::F64, false);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen failed");
        // 64-bit path must split into lo/hi 32-bit words
        assert!(ptx.contains("%lo"), "must have %lo register\n{ptx}");
        assert!(ptx.contains("%hi"), "must have %hi register\n{ptx}");
        assert!(
            ptx.contains("mov.b64"),
            "must reassemble 64-bit value\n{ptx}"
        );
    }

    #[test]
    fn generate_sm75_uses_correct_target() {
        let t = make_template(ReduceOp::Sum, PtxType::F32, false);
        let ptx = t.generate(SmVersion::Sm75).expect("PTX gen failed");
        assert!(ptx.contains(".target sm_75"), "wrong sm target\n{ptx}");
    }

    #[test]
    fn generate_product_f32_uses_mul() {
        let t = make_template(ReduceOp::Product, PtxType::F32, false);
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen failed");
        assert!(ptx.contains("mul.f32"), "must use mul.f32\n{ptx}");
    }

    #[test]
    fn config_equality() {
        let a = WarpReduceConfig {
            op: ReduceOp::Sum,
            ty: PtxType::F32,
            broadcast: false,
        };
        let b = WarpReduceConfig {
            op: ReduceOp::Sum,
            ty: PtxType::F32,
            broadcast: false,
        };
        assert_eq!(a, b);
    }
}
