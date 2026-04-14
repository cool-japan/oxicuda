//! Device-wide parallel reduction.
//!
//! Computes a single aggregate (sum, max, min, product, …) across all
//! elements of a device buffer.  Large inputs are processed in multiple
//! kernel launches:
//!
//! 1. **Pass 1** — each thread block reduces `block_size` elements to one
//!    partial result, written to a temporary device buffer.
//! 2. **Pass 2** — a single block reduces all partial results to the final
//!    scalar value.
//!
//! Both passes reuse the [`crate::block::BlockReduceTemplate`] PTX generator.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::device::reduce::{DeviceReduceConfig, DeviceReduceTemplate};
//! use oxicuda_primitives::ptx_helpers::ReduceOp;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = DeviceReduceConfig {
//!     op: ReduceOp::Sum,
//!     ty: PtxType::F32,
//!     block_size: 256,
//! };
//! let t = DeviceReduceTemplate::new(cfg);
//! let (pass1_ptx, pass2_ptx) = t.generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(pass1_ptx.contains("device_reduce_pass1_sum_f32_bs256"));
//! assert!(pass2_ptx.contains("device_reduce_pass2_sum_f32_bs256"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::ptx_helpers::{ReduceOp, ptx_header, ptx_type_str};

/// Default block size for device-wide reductions.
pub const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Configuration for a device-wide reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceReduceConfig {
    /// The combining operation.
    pub op: ReduceOp,
    /// Element type.
    pub ty: PtxType,
    /// Threads per block (power of 2, 32–1024).
    pub block_size: u32,
}

impl DeviceReduceConfig {
    /// Create a config with the default block size.
    #[must_use]
    pub fn new(op: ReduceOp, ty: PtxType) -> Self {
        Self {
            op,
            ty,
            block_size: DEFAULT_BLOCK_SIZE,
        }
    }

    /// Number of thread blocks required to reduce `n` elements in pass 1.
    #[must_use]
    pub fn num_blocks(&self, n: u64) -> u64 {
        n.div_ceil(self.block_size as u64)
    }

    /// Bytes of temporary storage required for `n` elements.
    #[must_use]
    pub fn temp_bytes(&self, n: u64) -> u64 {
        let elem = match self.ty {
            PtxType::F64 | PtxType::U64 | PtxType::S64 => 8,
            _ => 4,
        };
        self.num_blocks(n) * elem
    }

    /// Kernel name for pass 1.
    #[must_use]
    pub fn pass1_name(&self) -> String {
        format!(
            "device_reduce_pass1_{}_{}_bs{}",
            self.op.name(),
            ptx_type_str(self.ty),
            self.block_size
        )
    }

    /// Kernel name for pass 2.
    #[must_use]
    pub fn pass2_name(&self) -> String {
        format!(
            "device_reduce_pass2_{}_{}_bs{}",
            self.op.name(),
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

/// PTX code generator for device-wide reduction (two-pass).
pub struct DeviceReduceTemplate {
    /// Configuration.
    pub cfg: DeviceReduceConfig,
}

impl DeviceReduceTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: DeviceReduceConfig) -> Self {
        Self { cfg }
    }

    /// Generate PTX source for both reduction passes.
    ///
    /// Returns `(pass1_ptx, pass2_ptx)`.
    ///
    /// # Errors
    ///
    /// Returns a string error on generation failure.
    pub fn generate(&self, sm: SmVersion) -> Result<(String, String), String> {
        let pass1 = self.generate_pass1(sm)?;
        let pass2 = self.generate_pass2(sm)?;
        Ok((pass1, pass2))
    }

    // ── Pass 1: each block reduces block_size elements → partial sums ──────

    fn generate_pass1(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.pass1_name();
        let ty = ptx_type_str(self.cfg.ty);
        let op = self.cfg.op.ptx_instr(self.cfg.ty);
        let bs = self.cfg.block_size;
        let is_64bit = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        let elem_bytes: u32 = if is_64bit { 8 } else { 4 };
        let num_warps = bs / 32;

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
        writeln!(
            out,
            ".shared .align {elem_bytes} .{ty} pass1_smem[{num_warps}];"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_partials,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %val, %shfl;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u32    %tid, %bid, %warpid, %laneid, %mask, %offset;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %n, %global_tid;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %ptr_in, %ptr_out, %addr, %smem_addr;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_out, [param_partials];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %n,        [param_n];").map_err(|e| e.to_string())?;

        writeln!(out, "    mov.u32 %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32 %bid, %ctaid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    shr.u32 %warpid, %tid, 5;").map_err(|e| e.to_string())?;
        writeln!(out, "    and.b32 %laneid, %tid, 31;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32 %mask, 0xFFFFFFFF;").map_err(|e| e.to_string())?;

        // global thread index
        writeln!(out, "    mad.lo.u64 %global_tid, %bid, {bs}, %tid;")
            .map_err(|e| e.to_string())?;

        // Load with bounds check
        writeln!(out, "    setp.ge.u64 %p, %global_tid, %n;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u64  %addr, %global_tid, {elem_bytes}, %ptr_in;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.global.{ty} %val, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;

        // Warp shuffle butterfly reduction (5 rounds)
        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(out, "    mov.u32 %offset, {offset};").map_err(|e| e.to_string())?;
            if is_64bit {
                writeln!(out, "    {{").map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        .reg .u32 %lo, %hi, %sl, %sh; .reg .{ty} %shfl64;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "        mov.b64 {{%lo, %hi}}, %val;").map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sl, %lo, %offset, 31, %mask;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sh, %hi, %offset, 31, %mask;"
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

        // Lane 0 writes warp partial to shared memory
        writeln!(out, "    setp.ne.u32 %p, %laneid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra PASS1_SMEM_SKIP_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32 %offset, {elem_bytes};").map_err(|e| e.to_string())?;
        writeln!(out, "    mul.lo.u32 %offset, %warpid, %offset;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64 %smem_addr, pass1_smem;").map_err(|e| e.to_string())?;
        writeln!(out, "    cvt.u64.u32 %addr, %offset;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u64 %smem_addr, %smem_addr, %addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.{ty} [%smem_addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "PASS1_SMEM_SKIP_{name}:").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

        // Warp 0 reduces partial sums
        writeln!(out, "    setp.ne.u32 %p, %warpid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra PASS1_WARP0_SKIP_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ge.u32 %p, %laneid, {num_warps};").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32 %offset, {elem_bytes};").map_err(|e| e.to_string())?;
        writeln!(out, "    mul.lo.u32 %offset, %laneid, %offset;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64 %smem_addr, pass1_smem;").map_err(|e| e.to_string())?;
        writeln!(out, "    cvt.u64.u32 %addr, %offset;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u64 %smem_addr, %smem_addr, %addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.shared.{ty} %val, [%smem_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;

        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(out, "    mov.u32 %offset, {offset};").map_err(|e| e.to_string())?;
            if is_64bit {
                writeln!(out, "    {{").map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        .reg .u32 %lo, %hi, %sl, %sh; .reg .{ty} %shfl64;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "        mov.b64 {{%lo, %hi}}, %val;").map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sl, %lo, %offset, 31, %mask;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sh, %hi, %offset, 31, %mask;"
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

        // Lane 0 writes block partial to output
        writeln!(out, "    setp.ne.u32 %p, %laneid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra PASS1_WARP0_SKIP_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64 %addr, %bid, {elem_bytes}, %ptr_out;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.{ty} [%addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "PASS1_WARP0_SKIP_{name}:").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    // ── Pass 2: reduce the partial sums from pass 1 → final scalar ─────────

    fn generate_pass2(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.pass2_name();
        let ty = ptx_type_str(self.cfg.ty);
        let op = self.cfg.op.ptx_instr(self.cfg.ty);
        let bs = self.cfg.block_size;
        let is_64bit = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        let elem_bytes: u32 = if is_64bit { 8 } else { 4 };
        let num_warps = bs / 32;

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
        writeln!(
            out,
            ".shared .align {elem_bytes} .{ty} pass2_smem[{num_warps}];"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_result,\n    \
             .param .u64 param_partials,\n    \
             .param .u32 param_npartials\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %val, %shfl;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u32    %tid, %np, %warpid, %laneid, %mask, %offset;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %ptr_in, %ptr_out, %addr, %smem_addr;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_out, [param_result];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_partials];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %np,       [param_npartials];")
            .map_err(|e| e.to_string())?;

        writeln!(out, "    mov.u32 %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    shr.u32 %warpid, %tid, 5;").map_err(|e| e.to_string())?;
        writeln!(out, "    and.b32 %laneid, %tid, 31;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32 %mask, 0xFFFFFFFF;").map_err(|e| e.to_string())?;

        writeln!(out, "    setp.ge.u32 %p, %tid, %np;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64  %addr, %tid, {elem_bytes}, %ptr_in;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.global.{ty} %val, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;

        // Same two-stage warp reduce as pass 1
        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(out, "    mov.u32 %offset, {offset};").map_err(|e| e.to_string())?;
            if is_64bit {
                writeln!(out, "    {{").map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        .reg .u32 %lo, %hi, %sl, %sh; .reg .{ty} %shfl64;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "        mov.b64 {{%lo, %hi}}, %val;").map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sl, %lo, %offset, 31, %mask;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sh, %hi, %offset, 31, %mask;"
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

        writeln!(out, "    setp.ne.u32 %p, %laneid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra PASS2_SMEM_SKIP_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    mul.lo.u32 %offset, %warpid, {elem_bytes};")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64 %smem_addr, pass2_smem;").map_err(|e| e.to_string())?;
        writeln!(out, "    cvt.u64.u32 %addr, %offset;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u64 %smem_addr, %smem_addr, %addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.{ty} [%smem_addr], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "PASS2_SMEM_SKIP_{name}:").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

        writeln!(out, "    setp.ne.u32 %p, %warpid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra PASS2_DONE_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ge.u32 %p, %laneid, {num_warps};").map_err(|e| e.to_string())?;
        writeln!(out, "    mul.lo.u32 %offset, %laneid, {elem_bytes};")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64 %smem_addr, pass2_smem;").map_err(|e| e.to_string())?;
        writeln!(out, "    cvt.u64.u32 %addr, %offset;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u64 %smem_addr, %smem_addr, %addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    @!%p ld.shared.{ty} %val, [%smem_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  mov.{ty} %val, {identity};").map_err(|e| e.to_string())?;

        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(out, "    mov.u32 %offset, {offset};").map_err(|e| e.to_string())?;
            if is_64bit {
                writeln!(out, "    {{").map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        .reg .u32 %lo, %hi, %sl, %sh; .reg .{ty} %shfl64;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "        mov.b64 {{%lo, %hi}}, %val;").map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sl, %lo, %offset, 31, %mask;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "        shfl.sync.bfly.b32 %sh, %hi, %offset, 31, %mask;"
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

        writeln!(out, "    setp.ne.u32 %p, %laneid, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra PASS2_DONE_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.{ty} [%ptr_out], %val;").map_err(|e| e.to_string())?;
        writeln!(out, "PASS2_DONE_{name}:").map_err(|e| e.to_string())?;
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

    fn cfg(op: ReduceOp, ty: PtxType) -> DeviceReduceConfig {
        DeviceReduceConfig::new(op, ty)
    }

    #[test]
    fn num_blocks_multiple_of_bs() {
        let c = cfg(ReduceOp::Sum, PtxType::F32);
        assert_eq!(c.num_blocks(1024), 4); // 1024/256
    }

    #[test]
    fn num_blocks_partial_last() {
        let c = cfg(ReduceOp::Sum, PtxType::F32);
        assert_eq!(c.num_blocks(257), 2); // ceil(257/256)
    }

    #[test]
    fn temp_bytes_f32() {
        let c = cfg(ReduceOp::Sum, PtxType::F32);
        assert_eq!(c.temp_bytes(256), 4); // 1 block × 4 bytes
        assert_eq!(c.temp_bytes(512), 8); // 2 blocks × 4 bytes
    }

    #[test]
    fn temp_bytes_f64() {
        let c = cfg(ReduceOp::Sum, PtxType::F64);
        assert_eq!(c.temp_bytes(256), 8); // 1 block × 8 bytes
    }

    #[test]
    fn pass1_name_sum_f32() {
        let c = cfg(ReduceOp::Sum, PtxType::F32);
        assert!(c.pass1_name().contains("pass1_sum"), "{}", c.pass1_name());
    }

    #[test]
    fn pass2_name_sum_f32() {
        let c = cfg(ReduceOp::Sum, PtxType::F32);
        assert!(c.pass2_name().contains("pass2_sum"), "{}", c.pass2_name());
    }

    #[test]
    fn generate_sum_f32_both_passes() {
        let t = DeviceReduceTemplate::new(cfg(ReduceOp::Sum, PtxType::F32));
        let (p1, p2) = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(
            p1.contains("device_reduce_pass1"),
            "pass1 name missing\n{p1}"
        );
        assert!(
            p2.contains("device_reduce_pass2"),
            "pass2 name missing\n{p2}"
        );
    }

    #[test]
    fn generate_uses_bar_sync() {
        let t = DeviceReduceTemplate::new(cfg(ReduceOp::Max, PtxType::U32));
        let (p1, p2) = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(p1.contains("bar.sync"), "pass1 must sync\n{p1}");
        assert!(p2.contains("bar.sync"), "pass2 must sync\n{p2}");
    }

    #[test]
    fn generate_min_f32_uses_min_instr() {
        let t = DeviceReduceTemplate::new(cfg(ReduceOp::Min, PtxType::F32));
        let (p1, _) = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(p1.contains("min.f32"), "must use min.f32\n{p1}");
    }

    #[test]
    fn generate_f64_uses_lo_hi_split() {
        let t = DeviceReduceTemplate::new(cfg(ReduceOp::Sum, PtxType::F64));
        let (p1, _) = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(p1.contains("mov.b64"), "must split 64-bit\n{p1}");
    }

    #[test]
    fn generate_uses_shfl_bfly() {
        let t = DeviceReduceTemplate::new(cfg(ReduceOp::Sum, PtxType::F32));
        let (p1, _) = t.generate(SmVersion::Sm80).expect("PTX gen");
        assert!(
            p1.contains("shfl.sync.bfly.b32"),
            "must use warp shuffle\n{p1}"
        );
    }
}
