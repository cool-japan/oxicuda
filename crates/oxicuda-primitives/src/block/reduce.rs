//! Block-level parallel reduction using shared memory and warp shuffles.
//!
//! A block reduction aggregates all values across an entire thread block.
//! The algorithm uses two stages:
//!
//! 1. **Intra-warp reduction**: each of the `blockDim.x / 32` warps reduces
//!    its 32 elements to a single value stored in shared memory.
//! 2. **Inter-warp reduction**: warp 0 reads the `blockDim.x / 32` partial
//!    sums from shared memory and performs a final warp reduction.
//!
//! The result ends up in thread 0 of the block.  An optional `broadcast`
//! flag causes the result to be broadcast to all threads via another shuffle.
//!
//! # Supported block sizes
//!
//! Block size must be a power of 2 between 32 and 1024.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::block::reduce::{BlockReduceTemplate, BlockReduceConfig};
//! use oxicuda_primitives::ptx_helpers::ReduceOp;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = BlockReduceConfig {
//!     op: ReduceOp::Sum,
//!     ty: PtxType::F32,
//!     block_size: 256,
//!     broadcast: false,
//! };
//! let ptx = BlockReduceTemplate::new(cfg)
//!     .generate(SmVersion::Sm80)
//!     .expect("PTX gen failed");
//! assert!(ptx.contains("block_reduce_sum_f32_bs256"));
//! assert!(ptx.contains(".shared .align"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::ptx_helpers::{ReduceOp, ptx_header, ptx_type_str};

/// Maximum supported block size for block-level reductions.
pub const MAX_BLOCK_SIZE: u32 = 1024;

/// Configuration for a block-level reduction kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockReduceConfig {
    /// The combining operation.
    pub op: ReduceOp,
    /// Element type.
    pub ty: PtxType,
    /// Number of threads per block (must be power of 2, 32–1024).
    pub block_size: u32,
    /// If `true`, broadcast result to all threads in the block.
    pub broadcast: bool,
}

impl BlockReduceConfig {
    /// Validate that `block_size` is a power-of-2 in [32, 1024].
    ///
    /// # Errors
    ///
    /// Returns an error message if validation fails.
    pub fn validate(&self) -> Result<(), String> {
        if self.block_size < 32 || self.block_size > MAX_BLOCK_SIZE {
            return Err(format!(
                "block_size must be in [32, {MAX_BLOCK_SIZE}], got {}",
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
        let suffix = if self.broadcast { "_bcast" } else { "" };
        format!(
            "block_reduce_{}_{}_{}{suffix}",
            self.op.name(),
            ptx_type_str(self.ty),
            format_args!("bs{}", self.block_size),
        )
    }

    /// Number of warps in the block.
    #[must_use]
    pub fn num_warps(&self) -> u32 {
        self.block_size / 32
    }
}

/// PTX code generator for block-level reduction.
pub struct BlockReduceTemplate {
    /// Kernel configuration.
    pub cfg: BlockReduceConfig,
}

impl BlockReduceTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: BlockReduceConfig) -> Self {
        Self { cfg }
    }

    /// Generate the PTX source.
    ///
    /// # Errors
    ///
    /// Returns a string error if configuration validation fails or PTX
    /// generation encounters an issue.
    pub fn generate(&self, sm: SmVersion) -> Result<String, String> {
        self.cfg.validate()?;
        self.generate_inner(sm)
    }

    fn generate_inner(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let op = self.cfg.op.ptx_instr(self.cfg.ty);
        let num_warps = self.cfg.num_warps();
        let is_64bit = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        let elem_bytes = if is_64bit { 8u32 } else { 4u32 };
        let smem_bytes = num_warps * elem_bytes;

        let mut out = ptx_header(sm);

        // ── Shared memory declaration ────────────────────────────────────────
        writeln!(
            out,
            ".shared .align {elem_bytes} .{ty} smem_partial[{num_warps}];  \
             // one slot per warp"
        )
        .map_err(|e| e.to_string())?;

        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_result,\n    \
             .param .u64 param_input,\n    \
             .param .u32 param_n\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %val, %shfl;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u32    %tid, %n, %warpid, %laneid, %mask, %offset;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %ptr_in, %ptr_out, %addr, %smem_addr;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p, %q;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_out, [param_result];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %n,        [param_n];").map_err(|e| e.to_string())?;

        writeln!(out, "    mov.u32 %tid,    %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    shr.u32 %warpid, %tid, 5;    // warpid = tid / 32")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    and.b32 %laneid, %tid, 31;   // laneid = tid % 32")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32 %mask, 0xFFFFFFFF;").map_err(|e| e.to_string())?;

        // Load input element (identity if out of bounds)
        let identity = self.cfg.op.identity_literal::<f32>();
        let elem_bytes_usize = elem_bytes as usize;
        writeln!(out, "    setp.ge.u32 %p, %tid, %n;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u64  %addr, %tid, {elem_bytes_usize}, %ptr_in;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.global.{ty} %val, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;

        // ── Stage 1: Intra-warp reduction via butterfly shuffle ──────────────
        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(out, "    mov.u32 %offset, {offset};").map_err(|e| e.to_string())?;
            if is_64bit {
                writeln!(out, "    // 64-bit warp shuffle via lo/hi split")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    {{").map_err(|e| e.to_string())?;
                writeln!(out, "        .reg .u32 %lo64, %hi64, %sl, %sh;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "        .reg .{ty} %shfl64;").map_err(|e| e.to_string())?;
                writeln!(out, "        mov.b64 {{%lo64, %hi64}}, %val;")
                    .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sl, %lo64, %offset, 31, %mask;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sh, %hi64, %offset, 31, %mask;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "        mov.b64 %shfl64, {{%sl, %sh}};")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "        {op} %val, %val, %shfl64;").map_err(|e| e.to_string())?;
                writeln!(out, "    }}").map_err(|e| e.to_string())?;
            } else {
                writeln!(
                    out,
                    "    shfl.sync.bfly.b32 %shfl, %val, %offset, 31, %mask;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "    {op} %val, %val, %shfl;").map_err(|e| e.to_string())?;
            }
        }

        // ── Lane 0 of each warp writes its partial sum to shared memory ──────
        writeln!(out, "    setp.ne.u32 %p, %laneid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra AFTER_SMEM_WRITE_{name};").map_err(|e| e.to_string())?;
        // smem_partial[warpid] = val
        writeln!(out, "    mul.lo.u32  %offset, %warpid, {elem_bytes};")
            .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mov.u64     %smem_addr, smem_partial;  // base of shared array"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    cvt.u64.u32 %addr, %offset;  // zero-extend offset"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    add.u64     %smem_addr, %smem_addr, %addr;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.{ty} [%smem_addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "AFTER_SMEM_WRITE_{name}:").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;   // all warps done writing").map_err(|e| e.to_string())?;

        // ── Stage 2: Warp 0 reads all partial sums and reduces ───────────────
        writeln!(out, "    setp.ne.u32 %p, %warpid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra SKIP_WARP0_{name};").map_err(|e| e.to_string())?;

        // lane `laneid` (0..num_warps-1) loads smem_partial[laneid]
        writeln!(
            out,
            "    setp.ge.u32 %p, %laneid, {num_warps};  // lanes >= num_warps get identity"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    mul.lo.u32  %offset, %laneid, {elem_bytes};")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64     %smem_addr, smem_partial;").map_err(|e| e.to_string())?;
        writeln!(out, "    cvt.u64.u32 %addr, %offset;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u64     %smem_addr, %smem_addr, %addr;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.shared.{ty} %val, [%smem_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;

        // Final intra-warp reduction on the partial sums (≤32 warps)
        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(out, "    mov.u32 %offset, {offset};").map_err(|e| e.to_string())?;
            if is_64bit {
                writeln!(out, "    {{").map_err(|e| e.to_string())?;
                writeln!(out, "        .reg .u32 %lo64, %hi64, %sl, %sh;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "        .reg .{ty} %shfl64;").map_err(|e| e.to_string())?;
                writeln!(out, "        mov.b64 {{%lo64, %hi64}}, %val;")
                    .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sl, %lo64, %offset, 31, %mask;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sh, %hi64, %offset, 31, %mask;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "        mov.b64 %shfl64, {{%sl, %sh}};")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "        {op} %val, %val, %shfl64;").map_err(|e| e.to_string())?;
                writeln!(out, "    }}").map_err(|e| e.to_string())?;
            } else {
                writeln!(
                    out,
                    "    shfl.sync.bfly.b32 %shfl, %val, %offset, 31, %mask;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "    {op} %val, %val, %shfl;").map_err(|e| e.to_string())?;
            }
        }

        // Thread 0 writes result
        writeln!(out, "    setp.ne.u32 %p, %laneid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra SKIP_WARP0_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.{ty} [%ptr_out], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "SKIP_WARP0_{name}:").map_err(|e| e.to_string())?;

        // Optional broadcast: write result to smem and let all threads read it
        if self.cfg.broadcast {
            writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
            writeln!(out, "    mov.u64 %smem_addr, smem_partial;").map_err(|e| e.to_string())?;
            // Warp 0, lane 0 already wrote result to smem_partial[0]
            // (we need to store it there explicitly)
            // Re-store: lane 0 of warp 0 stores val at smem_partial[0]
            writeln!(out, "    setp.ne.u32 %p, %tid, 0;").map_err(|e| e.to_string())?;
            writeln!(out, "    @%p bra BCAST_READ_{name};").map_err(|e| e.to_string())?;
            writeln!(out, "    st.shared.{ty} [%smem_addr], %val;").map_err(|e| e.to_string())?;
            writeln!(out, "BCAST_READ_{name}:").map_err(|e| e.to_string())?;
            writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
            writeln!(out, "    ld.shared.{ty} %val, [%smem_addr];").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "    mad.lo.u64  %addr, %tid, {elem_bytes_usize}, %ptr_out;"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "    st.global.{ty} [%addr], %val;").map_err(|e| e.to_string())?;
        }

        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;
        let _ = smem_bytes;

        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    fn cfg(op: ReduceOp, ty: PtxType, bs: u32, bc: bool) -> BlockReduceConfig {
        BlockReduceConfig {
            op,
            ty,
            block_size: bs,
            broadcast: bc,
        }
    }

    #[test]
    fn kernel_name_sum_f32_bs256() {
        let c = cfg(ReduceOp::Sum, PtxType::F32, 256, false);
        assert_eq!(c.kernel_name(), "block_reduce_sum_f32_bs256");
    }

    #[test]
    fn kernel_name_max_u32_bs512_bcast() {
        let c = cfg(ReduceOp::Max, PtxType::U32, 512, true);
        assert_eq!(c.kernel_name(), "block_reduce_max_u32_bs512_bcast");
    }

    #[test]
    fn num_warps_256() {
        assert_eq!(cfg(ReduceOp::Sum, PtxType::F32, 256, false).num_warps(), 8);
    }

    #[test]
    fn num_warps_1024() {
        assert_eq!(
            cfg(ReduceOp::Sum, PtxType::F32, 1024, false).num_warps(),
            32
        );
    }

    #[test]
    fn validate_ok() {
        assert!(
            cfg(ReduceOp::Sum, PtxType::F32, 256, false)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn validate_non_power_of_two_fails() {
        assert!(
            cfg(ReduceOp::Sum, PtxType::F32, 200, false)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_too_small_fails() {
        assert!(
            cfg(ReduceOp::Sum, PtxType::F32, 16, false)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_too_large_fails() {
        assert!(
            cfg(ReduceOp::Sum, PtxType::F32, 2048, false)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn generate_sum_f32_bs256_has_kernel_name() {
        let t = BlockReduceTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, 256, false));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(
            ptx.contains("block_reduce_sum_f32_bs256"),
            "missing name\n{ptx}"
        );
    }

    #[test]
    fn generate_uses_shared_memory() {
        let t = BlockReduceTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, 256, false));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(ptx.contains(".shared"), "must have shared memory\n{ptx}");
        assert!(
            ptx.contains("smem_partial"),
            "must reference smem_partial\n{ptx}"
        );
    }

    #[test]
    fn generate_uses_bar_sync() {
        let t = BlockReduceTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, 256, false));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(ptx.contains("bar.sync"), "must synchronise threads\n{ptx}");
    }

    #[test]
    fn generate_uses_shfl_bfly() {
        let t = BlockReduceTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, 256, false));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(
            ptx.contains("shfl.sync.bfly.b32"),
            "must use warp shuffle\n{ptx}"
        );
    }

    #[test]
    fn generate_f64_uses_lo_hi_split() {
        let t = BlockReduceTemplate::new(cfg(ReduceOp::Sum, PtxType::F64, 256, false));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(ptx.contains("mov.b64"), "must reassemble 64-bit\n{ptx}");
    }

    #[test]
    fn generate_min_u32_uses_min_instr() {
        let t = BlockReduceTemplate::new(cfg(ReduceOp::Min, PtxType::U32, 128, false));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(ptx.contains("min.u32"), "must use min.u32\n{ptx}");
    }

    #[test]
    fn generate_broadcast_uses_second_bar_sync() {
        let t = BlockReduceTemplate::new(cfg(ReduceOp::Sum, PtxType::F32, 64, true));
        let ptx = t.generate(SmVersion::Sm80).expect("PTX gen");
        // broadcast requires at least 2 bar.sync calls
        let count = ptx.matches("bar.sync").count();
        assert!(
            count >= 2,
            "broadcast needs ≥2 bar.sync, got {count}\n{ptx}"
        );
    }
}
