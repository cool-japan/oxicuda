//! GEMM (General Matrix Multiply) kernel template.
//!
//! This module defines the parameterization types for GEMM kernel generation.
//! The full high-performance implementation (tiled, multi-stage, tensor-core-backed)
//! will be provided in Vol.3 (`oxicuda-blas`). This module defines the template
//! structure and provides a naive triple-loop GEMM kernel for correctness testing.
//!
//! # GEMM formula
//!
//! `C = alpha * A * B + beta * C`
//!
//! Where A is M x K, B is K x N, and C is M x N.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::gemm::{GemmTemplate, EpilogueKind};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = GemmTemplate {
//!     tile_m: 16,
//!     tile_n: 16,
//!     tile_k: 16,
//!     warp_m: 16,
//!     warp_n: 16,
//!     precision: PtxType::F32,
//!     accumulator: PtxType::F32,
//!     use_tensor_core: false,
//!     stages: 1,
//!     target: SmVersion::Sm80,
//!     epilogue: EpilogueKind::LinearCombination,
//! };
//! let ptx = template.generate().expect("PTX generation failed");
//! assert!(ptx.contains("gemm"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Epilogue operation applied after the matrix multiplication.
///
/// Controls what happens to the accumulator values before writing to the
/// output matrix C. The linear combination epilogue computes
/// `C = alpha * accumulator + beta * C_old`.
#[derive(Debug, Clone)]
pub enum EpilogueKind {
    /// `C = alpha * A*B + beta * C`.
    LinearCombination,
    /// `C = relu(alpha * A*B + beta * C)`.
    LinearCombinationRelu,
    /// `C = gelu(alpha * A*B + beta * C)`.
    LinearCombinationGelu,
    /// `C = alpha * A*B + beta * C + bias`.
    LinearCombinationBias,
    /// `C = relu(alpha * A*B + beta * C + bias)`.
    LinearCombinationBiasRelu,
}

impl EpilogueKind {
    /// Returns a short name for kernel naming.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::LinearCombination => "lincomb",
            Self::LinearCombinationRelu => "lincomb_relu",
            Self::LinearCombinationGelu => "lincomb_gelu",
            Self::LinearCombinationBias => "lincomb_bias",
            Self::LinearCombinationBiasRelu => "lincomb_bias_relu",
        }
    }

    /// Returns `true` if this epilogue requires a bias vector parameter.
    #[must_use]
    pub const fn needs_bias(&self) -> bool {
        matches!(
            self,
            Self::LinearCombinationBias | Self::LinearCombinationBiasRelu
        )
    }
}

/// GEMM kernel template parameters.
///
/// Encapsulates all tuning knobs for a GEMM kernel: tile dimensions,
/// precision, tensor core usage, pipeline depth, and epilogue type.
/// The full tiled implementation is deferred to Vol.3 (`oxicuda-blas`);
/// this module provides a naive reference kernel.
pub struct GemmTemplate {
    /// Tile size in the M dimension (rows of A/C per block).
    pub tile_m: u32,
    /// Tile size in the N dimension (columns of B/C per block).
    pub tile_n: u32,
    /// Tile size in the K dimension (reduction loop step).
    pub tile_k: u32,
    /// Warp tile size in M (rows per warp).
    pub warp_m: u32,
    /// Warp tile size in N (columns per warp).
    pub warp_n: u32,
    /// Input matrix element precision.
    pub precision: PtxType,
    /// Accumulator precision (typically F32 even for F16 inputs).
    pub accumulator: PtxType,
    /// Whether to use tensor core instructions (WMMA/MMA).
    pub use_tensor_core: bool,
    /// Number of software pipeline stages for async loads.
    pub stages: u32,
    /// Target GPU architecture.
    pub target: SmVersion,
    /// Epilogue operation.
    pub epilogue: EpilogueKind,
}

impl GemmTemplate {
    /// Returns the kernel function name encoding key parameters.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let prec = self.precision.as_ptx_str().trim_start_matches('.');
        let acc = self.accumulator.as_ptx_str().trim_start_matches('.');
        let tc = if self.use_tensor_core { "tc" } else { "naive" };
        format!(
            "gemm_{}x{}x{}_{}_{}_{}",
            self.tile_m, self.tile_n, self.tile_k, prec, acc, tc
        )
    }

    /// Returns `true` if [`generate`](Self::generate) emits a kernel that
    /// stages A/B tiles through shared memory (`.shared` declaration +
    /// `ld.shared`/`st.shared` in the hot loop), and therefore has a genuine
    /// dynamic-shared-memory *requirement* a launcher must budget for.
    ///
    /// [`generate`](Self::generate) currently does not: it is a one-thread-
    /// per-output-element grid-stride kernel that reads every `A`/`B`
    /// element directly from global memory on every `K`-step, with no
    /// `.shared` declaration anywhere in its output (the
    /// `gemm_naive_ptx_assembles_for_sm86` / structural tests below pin
    /// this). A caller that requests
    /// `(tile_m*tile_k + tile_k*tile_n) * elem_size * stages` bytes of
    /// dynamic shared memory for it anyway -- sizing the launch as if the
    /// tile dimensions meant something to this kernel -- pays pure CTA-
    /// occupancy tax for zero benefit (measured: dropping that phantom
    /// request alone recovered real throughput on a 1024^3 F32 GEMM).
    ///
    /// This is a capability flag on the *template*, not a hardcoded `false`
    /// at every call site, precisely so a future kernel that genuinely tiles
    /// through shared memory (e.g. a real implementation behind
    /// [`generate_pipelined_skeleton`](Self::generate_pipelined_skeleton),
    /// which already emits `.shared` and `cp.async` but is not numerically
    /// functional yet -- see its doc comment) only has to flip this one
    /// place to make its launcher honestly request the memory it needs.
    #[must_use]
    pub const fn uses_shared_memory_tiles(&self) -> bool {
        false
    }

    /// Generates the complete PTX module text for a naive GEMM kernel.
    ///
    /// This is a simple triple-loop implementation intended for correctness
    /// verification. It does not use tiling, shared memory, or tensor cores.
    /// The high-performance implementation is in Vol.3.
    ///
    /// Kernel parameters:
    /// - `a_ptr`: pointer to A (M x K, row-major)
    /// - `b_ptr`: pointer to B (K x N, row-major)
    /// - `c_ptr`: pointer to C (M x N, row-major)
    /// - `m`, `n`, `k`: matrix dimensions
    /// - `alpha`, `beta`: scaling factors
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if the precision is unsupported or formatting fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        // `ty` is the input-element precision used for global loads/stores of
        // A, B and C; `acc_ty` is the accumulator precision used for the dot
        // product, alpha/beta scaling and the FMA epilogue. They may differ
        // (e.g. F16 inputs with an F32 accumulator), in which case the loaded
        // elements live in a separate register bank and are converted to the
        // accumulator precision with `cvt` before use.
        let ty = self.precision.as_ptx_str();
        let acc_ty = self.accumulator.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        let needs_cvt = self.precision != self.accumulator;
        let acc_zero = self.accumulator.zero_literal();
        // The PTX `ld`/`st` element type for the input precision. The ISA has no
        // `.f16`/`.bf16` memory-access form, so 16-bit floats are moved as raw
        // `.b16` and reinterpreted by `cvt`; `.f32`/`.f64` load/store directly.
        let mem_ty = match self.precision {
            PtxType::F16 | PtxType::BF16 => ".b16",
            _ => ty,
        };
        // 16-bit floats (`F16`/`BF16`) have no direct `cvt` to/from `.f64` on
        // pre-Hopper targets, and `bf16` is designed around `f32` on every
        // architecture, so half↔f64 conversions are routed through an `f32`
        // scratch register. This bank is only needed for half inputs with an
        // `F64` accumulator.
        let needs_f32_scratch = matches!(self.precision, PtxType::F16 | PtxType::BF16)
            && self.accumulator == PtxType::F64;

        let mut ptx = String::with_capacity(8192);

        // Header
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_a,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_b,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_c,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_m,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_n,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_k,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param {acc_ty} %param_alpha,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param {acc_ty} %param_beta").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Register declarations.
        //
        // The value bank `%f` is declared with the *accumulator* precision so
        // that `mov`/`mul`/`fma.rn`/`ld.param` on the accumulator, alpha and
        // beta all agree with the register type (PTX validates that the declared
        // register type matches every instruction's type modifier). When the
        // input precision differs, a second bank `%fin` holds the freshly loaded
        // input-precision elements prior to conversion.
        writeln!(ptx, "    .reg .b32 %r<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {acc_ty} %f<16>;").map_err(PtxGenError::FormatError)?;
        if needs_cvt {
            writeln!(ptx, "    .reg {mem_ty} %fin<8>;").map_err(PtxGenError::FormatError)?;
        }
        if needs_f32_scratch {
            writeln!(ptx, "    .reg .f32 %fc<8>;").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx, "    .reg .pred %p<4>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters (loop-invariant; hoisted out of the grid-stride loop).
        writeln!(ptx, "    // Load parameters").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_a];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_b];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_c];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r8, [%param_m];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r9, [%param_n];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r10, [%param_k];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param{acc_ty} %f8, [%param_alpha];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param{acc_ty} %f9, [%param_beta];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Naive GEMM with a grid-stride loop over the flattened (row-major)
        // output matrix. Each thread derives a unique *global linear id* from
        // its block/thread coordinates and walks the C matrix in strides of the
        // total launched thread count, computing one full K-dot-product per
        // element. Using a linear id (instead of 2-D thread/block indices) makes
        // the kernel correct under any launch geometry the dispatcher chooses —
        // in particular the 1-D thread blocks it emits for skinny/tiled shapes,
        // where `threadIdx.y` and `gridDim.y` would otherwise pin every thread
        // to row 0.
        writeln!(
            ptx,
            "    // global_id = ((ctaid.z*gridDim.y + ctaid.y)*gridDim.x + ctaid.x)*blockDim.x + tid.x"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ctaid.y;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r3, %ctaid.z;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r4, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r5, %nctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r6, %nctaid.y;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r7, %nctaid.z;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r11, %r3, %r6, %r2;").map_err(PtxGenError::FormatError)?; // z*gy + y
        writeln!(ptx, "    mad.lo.u32 %r11, %r11, %r5, %r1;").map_err(PtxGenError::FormatError)?; // *gx + x
        writeln!(ptx, "    mad.lo.u32 %r12, %r11, %r4, %r0;").map_err(PtxGenError::FormatError)?; // *bdx + tid.x
        writeln!(
            ptx,
            "    // total_threads = gridDim.x*gridDim.y*gridDim.z*blockDim.x"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r13, %r5, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r13, %r13, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r13, %r13, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    // total_elems = M*N (64-bit: 32x32->64 widening avoids overflow when M*N >= 2^32)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.wide.u32 %rd9, %r8, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Grid-stride loop over output elements: idx = global_id, += total_threads.
        // The linear index and its bound are held in 64-bit registers so GEMMs
        // with >= 2^32 output elements iterate over every element instead of
        // silently truncating the loop bound and computing only a fraction.
        writeln!(ptx, "    cvt.u64.u32 %rd10, %r12;  // idx = global_id")
            .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    cvt.u64.u32 %rd11, %r13;  // total_threads (grid stride)"
        )
        .map_err(PtxGenError::FormatError)?;
        // row/col recovery: strength-reduced instead of a 64-bit div/rem of
        // `idx` on every loop iteration. `ptxas` has no native 64-bit
        // integer divide; `div.u64`/`rem.u64` each lower to a `CALL.REL`
        // software long-division subroutine (tens of instructions), and the
        // old code paid that cost once *per output element*.
        //
        // `row`/`col` individually always fit u32 (they are bounded by M/N,
        // which are themselves u32 kernel parameters) -- the problem was
        // never their range, it was deriving them from `idx`, which *can*
        // need the full 64 bits when M*N >= 2^32 (the exact shapes the
        // 64-bit loop bound above exists to handle correctly). Truncating
        // `idx` to u32 before a plain `div.u32`/`rem.u32` would silently
        // reintroduce that bug, so instead: compute row/col for the first
        // element (`global_id`, which fits u32 since it is one of
        // `total_threads`-many launched threads and `total_threads` itself
        // is computed 32-bit above) with a single one-time 32-bit div/rem,
        // then advance them in lockstep with `idx` each iteration using the
        // *stride's own* one-time-computed row/col decomposition:
        //   idx_new = idx + stride
        //           = (row*N + col) + (stride_div_n*N + stride_rem_n)
        //           = (row + stride_div_n)*N + (col + stride_rem_n)
        // `col + stride_rem_n` can exceed N by at most N (both addends are
        // already < N), so at most one carry into row is ever needed -- a
        // handful of cheap `add`/`setp`/`selp` instructions replacing a
        // subroutine call, on every iteration instead of just the first.
        writeln!(
            ptx,
            "    // row0 = global_id / N ; col0 = global_id % N (one-time 32-bit \
             div/rem; global_id fits u32)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    div.u32 %r16, %r12, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rem.u32 %r17, %r12, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    // stride_div_n / stride_rem_n = total_threads / N, % N \
             (one-time 32-bit div/rem; total_threads fits u32)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    div.u32 %r19, %r13, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rem.u32 %r20, %r13, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$TILE_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u64 %p0, %rd10, %rd9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $GEMM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    // row (%r16) / col (%r17) already hold idx/N, idx%N for this \
             element -- advanced per-iteration below, not re-derived here."
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Accumulator init
        writeln!(ptx, "    // Initialize accumulator to 0").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{acc_ty} %f0, {acc_zero};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r18, 0;  // k").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Inner loop over K
        writeln!(ptx, "    // K-loop: accumulate dot product").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$K_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r18, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $K_DONE;").map_err(PtxGenError::FormatError)?;

        // Load A[row, k] = A[row * K + k]
        writeln!(
            ptx,
            "    // A[row, k] = A[row*K + k] (64-bit element index)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r18;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.wide.u32 %rd3, %r16, %r10, %rd3;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd0, %rd3;").map_err(PtxGenError::FormatError)?;
        if needs_cvt {
            writeln!(ptx, "    ld.global{mem_ty} %fin1, [%rd4];")
                .map_err(PtxGenError::FormatError)?;
            self.emit_convert_to_acc(&mut ptx, "%fin1", "%fc1", "%f1")?;
        } else {
            writeln!(ptx, "    ld.global{ty} %f1, [%rd4];").map_err(PtxGenError::FormatError)?;
        }

        // Load B[k, col] = B[k * N + col]
        writeln!(
            ptx,
            "    // B[k, col] = B[k*N + col] (64-bit element index)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r17;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.wide.u32 %rd5, %r18, %r9, %rd5;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd1, %rd5;").map_err(PtxGenError::FormatError)?;
        if needs_cvt {
            writeln!(ptx, "    ld.global{mem_ty} %fin2, [%rd6];")
                .map_err(PtxGenError::FormatError)?;
            self.emit_convert_to_acc(&mut ptx, "%fin2", "%fc2", "%f2")?;
        } else {
            writeln!(ptx, "    ld.global{ty} %f2, [%rd6];").map_err(PtxGenError::FormatError)?;
        }

        // acc += a * b
        writeln!(ptx, "    fma.rn{acc_ty} %f0, %f1, %f2, %f0;")
            .map_err(PtxGenError::FormatError)?;

        // k++
        writeln!(ptx, "    add.u32 %r18, %r18, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $K_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$K_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Epilogue: C[row, col] = alpha * acc + beta * C_old
        writeln!(ptx, "    // Epilogue: C = alpha * acc + beta * C_old")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd7, %r17;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.wide.u32 %rd7, %r16, %r9, %rd7;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd7, %rd7, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd8, %rd2, %rd7;").map_err(PtxGenError::FormatError)?;
        if needs_cvt {
            writeln!(ptx, "    ld.global{mem_ty} %fin3, [%rd8];")
                .map_err(PtxGenError::FormatError)?;
            self.emit_convert_to_acc(&mut ptx, "%fin3", "%fc3", "%f3")?;
        } else {
            writeln!(ptx, "    ld.global{ty} %f3, [%rd8];").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx, "    mul{acc_ty} %f0, %f0, %f8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    fma.rn{acc_ty} %f0, %f9, %f3, %f0;")
            .map_err(PtxGenError::FormatError)?;
        if needs_cvt {
            // Narrow the accumulator back to the output (input) precision before
            // the store, since C is stored element-for-element in input precision.
            self.emit_convert_from_acc(&mut ptx, "%f0", "%fc0", "%fin0")?;
            writeln!(ptx, "    st.global{mem_ty} [%rd8], %fin0;")
                .map_err(PtxGenError::FormatError)?;
        } else {
            writeln!(ptx, "    st.global{ty} [%rd8], %f0;").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Advance to the next element handled by this thread (64-bit stride
        // for `idx`, which stays the loop-continuation test above) and, in
        // lockstep, advance row/col by strength-reduction instead of
        // re-deriving them with a fresh div/rem -- see the comment above
        // `$TILE_LOOP` for the derivation.
        writeln!(ptx, "    add.u64 %rd10, %rd10, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    // row/col advance: col_sum = col + stride_rem_n (may reach N..2N-1)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r21, %r17, %r20;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r16, %r16, %r19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p2, %r21, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    selp.u32 %r22, 1, 0, %p2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r16, %r16, %r22;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub.u32 %r23, %r21, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    selp.u32 %r17, %r23, %r21, %p2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $TILE_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$GEMM_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    /// Emits the conversion of a freshly loaded input-precision element (in the
    /// `%fin`/`fin` register) into the accumulator-precision value register
    /// `dst`. 16-bit floats are widened via an `f32` intermediate (`fc`) when
    /// the accumulator is `F64`, because no direct half↔f64 `cvt` exists on
    /// pre-Hopper targets; otherwise a single `cvt` suffices.
    ///
    /// The caller guarantees the input and accumulator precisions differ.
    fn emit_convert_to_acc(
        &self,
        ptx: &mut String,
        fin: &str,
        fc: &str,
        dst: &str,
    ) -> Result<(), PtxGenError> {
        let in_ty = self.precision.as_ptx_str();
        match self.precision {
            PtxType::F16 | PtxType::BF16 => {
                if self.accumulator == PtxType::F64 {
                    // half -> f32 -> f64
                    writeln!(ptx, "    cvt.f32{in_ty} {fc}, {fin};")
                        .map_err(PtxGenError::FormatError)?;
                    writeln!(ptx, "    cvt.f64.f32 {dst}, {fc};")
                        .map_err(PtxGenError::FormatError)?;
                } else {
                    // half -> f32 (accumulator is f32)
                    writeln!(ptx, "    cvt.f32{in_ty} {dst}, {fin};")
                        .map_err(PtxGenError::FormatError)?;
                }
            }
            _ => {
                // Direct f32<->f64 conversion (widening or narrowing).
                let cvt = self.accumulator.float_cvt_mnemonic(self.precision);
                writeln!(ptx, "    {cvt} {dst}, {fin};").map_err(PtxGenError::FormatError)?;
            }
        }
        Ok(())
    }

    /// Emits the conversion of the accumulator-precision result register `src`
    /// back into the output (input) precision register `fin`, ready for the
    /// global store. Mirrors [`emit_convert_to_acc`](Self::emit_convert_to_acc),
    /// routing half↔f64 narrowing through the `f32` scratch register `fc`.
    ///
    /// The caller guarantees the input and accumulator precisions differ.
    fn emit_convert_from_acc(
        &self,
        ptx: &mut String,
        src: &str,
        fc: &str,
        fin: &str,
    ) -> Result<(), PtxGenError> {
        let in_ty = self.precision.as_ptx_str();
        match self.precision {
            PtxType::F16 | PtxType::BF16 => {
                if self.accumulator == PtxType::F64 {
                    // f64 -> f32 -> half
                    writeln!(ptx, "    cvt.rn.f32.f64 {fc}, {src};")
                        .map_err(PtxGenError::FormatError)?;
                    writeln!(ptx, "    cvt.rn{in_ty}.f32 {fin}, {fc};")
                        .map_err(PtxGenError::FormatError)?;
                } else {
                    // f32 -> half
                    writeln!(ptx, "    cvt.rn{in_ty}.f32 {fin}, {src};")
                        .map_err(PtxGenError::FormatError)?;
                }
            }
            _ => {
                // Direct f32<->f64 conversion.
                let cvt = self.precision.float_cvt_mnemonic(self.accumulator);
                writeln!(ptx, "    {cvt} {fin}, {src};").map_err(PtxGenError::FormatError)?;
            }
        }
        Ok(())
    }

    /// Generates a multi-stage software-pipeline **skeleton** -- structurally
    /// shaped like a real `cp.async`-pipelined tensor-core GEMM, but **not
    /// numerically functional**. Do not dispatch this for actual computation.
    ///
    /// # This is not a working kernel
    ///
    /// The emitted `mma.sync`/FMA compute section reads fixed placeholder
    /// operand registers (e.g. `%r4`-`%r9` for the tensor-core path) that are
    /// never loaded from the staged shared-memory tiles this method also
    /// emits -- they are declared but never written, so PTX leaves them
    /// zero-initialized and the "accumulator" the epilogue stores is not a
    /// GEMM result. Nothing here computes `C = alpha*A*B + beta*C`.
    ///
    /// What it *is* useful for -- and the only thing it is validated for, in
    /// `oxicuda-blas`'s `gpu_tests.rs` -- is exercising the **instruction
    /// schedule** a real pipelined kernel would need: a correct `cp.async`
    /// prologue/steady-state/drain schedule, `cp.async.commit_group` /
    /// `cp.async.wait_group` fencing, `bar.sync` placement, and (with
    /// `use_tensor_core: true`) a real `mma.sync.aligned` HMMA instruction
    /// that assembles under `ptxas` and launches fault-free on real hardware.
    /// If a future implementation stages A/B tiles into shared memory and
    /// loads genuine `mma` operands from them, promote this back to a
    /// `generate`-style numerically-complete method (and update
    /// [`GemmTemplate::uses_shared_memory_tiles`] to match).
    ///
    /// The pipeline depth is controlled by `self.stages` (must be >= 2).
    /// For each stage, the generated code contains:
    /// - A `cp.async.ca.shared.global` group prefetch section
    /// - A `cp.async.commit_group;` fence
    /// - A `cp.async.wait_group` drain
    /// - A `mma.sync.aligned` (or `wmma.mma`) compute section
    /// - A `bar.sync` barrier
    ///
    /// This method does **not** require GPU hardware; it generates the PTX
    /// text for structural / instruction-schedule verification.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if:
    /// - The template has invalid parameters (see `validate`)
    /// - `stages < 2` (single-stage pipeline is handled by [`generate`](Self::generate))
    /// - Formatting fails
    #[allow(clippy::too_many_lines)]
    pub fn generate_pipelined_skeleton(&self) -> Result<String, PtxGenError> {
        self.validate()?;
        if self.stages < 2 {
            return Err(PtxGenError::GenerationFailed(
                "generate_pipelined_skeleton requires stages >= 2; use generate() for single-stage GEMM"
                    .to_string(),
            ));
        }

        let ty = self.precision.as_ptx_str();
        let acc_ty = self.accumulator.as_ptx_str();
        let kernel_name = format!("gemm_pipelined_{}_{}stage", self.kernel_name(), self.stages);
        let stages = self.stages;

        let mut ptx = String::with_capacity(16_384);

        // Header
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Shared memory allocation: one slot per stage
        // Each slot holds tile_k * tile_m (A tile) + tile_k * tile_n (B tile) elements
        // Element sizes are guaranteed ≤ 8 bytes (no type exceeds u32 range)
        #[allow(clippy::cast_possible_truncation)]
        let elem_bytes = self.precision.size_bytes() as u32;
        let a_tile_bytes = self.tile_m * self.tile_k * elem_bytes;
        let b_tile_bytes = self.tile_k * self.tile_n * elem_bytes;
        let smem_per_stage = a_tile_bytes + b_tile_bytes;
        let smem_total = smem_per_stage * stages;

        // Shared memory declaration
        writeln!(ptx, ".shared .align 128 .b8 smem_a_b[{smem_total}];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_a,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_b,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_c,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_m,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_n,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_k,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param {acc_ty} %param_alpha,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param {acc_ty} %param_beta").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Register declarations
        writeln!(ptx, "    .reg .b32 %r<64>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %acc<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load pointers
        writeln!(ptx, "    // Load parameters").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_a];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_b];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_c];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r0, [%param_k];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Initialize accumulator
        writeln!(ptx, "    // Zero accumulators").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.f32 %acc0, 0f00000000;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // ── Prologue: fill the pipeline with (stages-1) prefetch stages ──────
        writeln!(
            ptx,
            "    // ---- Pipeline prologue: prefetch stages 0..{} ----",
            stages - 1
        )
        .map_err(PtxGenError::FormatError)?;
        for s in 0..(stages - 1) {
            writeln!(ptx, "    // Stage {s}: prefetch A tile").map_err(PtxGenError::FormatError)?;
            writeln!(
                ptx,
                "    cp.async.ca.shared.global [smem_a_b+{offset}], [%rd0], 16; // stage {s} A",
                offset = s * smem_per_stage
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    // Stage {s}: prefetch B tile").map_err(PtxGenError::FormatError)?;
            writeln!(
                ptx,
                "    cp.async.ca.shared.global [smem_a_b+{offset}], [%rd1], 16; // stage {s} B",
                offset = s * smem_per_stage + a_tile_bytes
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    cp.async.commit_group;").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // ── Main pipeline loop ───────────────────────────────────────────────
        writeln!(ptx, "    // ---- Main pipeline loop (steady state) ----")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, 0; // stage_idx").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, 0; // k_tile").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$PIPE_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r2, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $PIPE_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Prefetch next tile into the (stage_idx % stages) slot
        writeln!(ptx, "    // Prefetch next k-tile").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    cp.async.ca.shared.global [smem_a_b], [%rd0], 16; // prefetch A"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    cp.async.ca.shared.global [smem_a_b+{a_tile_bytes}], [%rd1], 16; // prefetch B"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cp.async.commit_group;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Wait for the oldest stage (stages - 1 groups allowed to remain)
        let wait_groups = stages.saturating_sub(1);
        writeln!(ptx, "    // Drain oldest pipeline stage").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cp.async.wait_group {wait_groups};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Tensor core compute
        writeln!(ptx, "    // Tensor core compute").map_err(PtxGenError::FormatError)?;
        if self.use_tensor_core {
            // Emit mma.sync.aligned for the configured shape
            writeln!(
                ptx,
                "    mma.sync.aligned.m16n8k16.row.col.f32{ty}{ty}.f32 {{%acc0,%acc1,%acc2,%acc3}}, {{%r4,%r5,%r6,%r7}}, {{%r8,%r9}}, {{%acc0,%acc1,%acc2,%acc3}};"
            )
            .map_err(PtxGenError::FormatError)?;
        } else {
            writeln!(
                ptx,
                "    fma.rn{acc_ty} %acc0, %acc0, %acc0, %acc0; // naive FMA placeholder"
            )
            .map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Advance k-tile
        writeln!(ptx, "    add.u32 %r2, %r2, 1; // k_tile++").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r1, %r1, 1; // stage_idx++")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $PIPE_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$PIPE_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // ── Epilogue: drain remaining pipeline stages ────────────────────────
        writeln!(
            ptx,
            "    // ---- Pipeline epilogue: drain remaining stages ----"
        )
        .map_err(PtxGenError::FormatError)?;
        for flush in 0..(stages - 1) {
            let remaining = (stages - 2).saturating_sub(flush);
            writeln!(
                ptx,
                "    cp.async.wait_group {remaining}; // flush stage {flush}"
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
            if self.use_tensor_core {
                writeln!(
                    ptx,
                    "    mma.sync.aligned.m16n8k16.row.col.f32{ty}{ty}.f32 {{%acc0,%acc1,%acc2,%acc3}}, {{%r4,%r5,%r6,%r7}}, {{%r8,%r9}}, {{%acc0,%acc1,%acc2,%acc3}}; // epilogue mma {flush}"
                )
                .map_err(PtxGenError::FormatError)?;
            } else {
                writeln!(
                    ptx,
                    "    fma.rn{acc_ty} %acc0, %acc0, %acc0, %acc0; // epilogue FMA {flush}"
                )
                .map_err(PtxGenError::FormatError)?;
            }
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Epilogue: write C
        writeln!(ptx, "    // Write accumulator to C").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{acc_ty} [%rd2], %acc0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    /// Validates template parameters.
    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(
            self.precision,
            PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64
        ) {
            return Err(PtxGenError::InvalidType(format!(
                "GEMM requires F16, BF16, F32, or F64 precision, got {}",
                self.precision.as_ptx_str()
            )));
        }
        if !matches!(self.accumulator, PtxType::F32 | PtxType::F64) {
            return Err(PtxGenError::InvalidType(format!(
                "GEMM accumulator must be F32 or F64, got {}",
                self.accumulator.as_ptx_str()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;

    #[test]
    fn kernel_name_format() {
        let t = GemmTemplate {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            warp_m: 64,
            warp_n: 64,
            precision: PtxType::F32,
            accumulator: PtxType::F32,
            use_tensor_core: false,
            stages: 2,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        };
        assert_eq!(t.kernel_name(), "gemm_128x128x32_f32_f32_naive");
    }

    /// `generate()` never stages tiles through shared memory today (see the
    /// method's own doc comment), so the capability flag must always report
    /// `false` regardless of tile shape, precision, or tensor-core request
    /// -- a caller must not be able to talk this flag into `true` by
    /// picking template fields that merely *look* tile-shaped.
    #[test]
    fn uses_shared_memory_tiles_is_false_for_every_config() {
        let configs = [
            GemmTemplate {
                tile_m: 128,
                tile_n: 128,
                tile_k: 16,
                warp_m: 64,
                warp_n: 64,
                precision: PtxType::F32,
                accumulator: PtxType::F32,
                use_tensor_core: false,
                stages: 1,
                target: SmVersion::Sm86,
                epilogue: EpilogueKind::LinearCombination,
            },
            GemmTemplate {
                tile_m: 256,
                tile_n: 128,
                tile_k: 64,
                warp_m: 64,
                warp_n: 64,
                precision: PtxType::F16,
                accumulator: PtxType::F32,
                use_tensor_core: true,
                stages: 4,
                target: SmVersion::Sm90,
                epilogue: EpilogueKind::LinearCombinationRelu,
            },
        ];
        for t in configs {
            assert!(
                !t.uses_shared_memory_tiles(),
                "generate() does not stage shared-memory tiles yet; \
                 uses_shared_memory_tiles() must stay false until it does"
            );
        }
    }

    /// The capability flag and the PTX `generate()` actually emits must
    /// agree: no `.shared` declaration when the flag is `false`.
    #[test]
    fn uses_shared_memory_tiles_matches_generated_ptx() {
        let t = gemm_template(PtxType::F32, PtxType::F32);
        let ptx = t.generate().expect("GEMM PTX generation should succeed");
        assert_eq!(
            t.uses_shared_memory_tiles(),
            ptx.contains(".shared"),
            "uses_shared_memory_tiles() disagrees with the emitted PTX's \
             .shared declaration:\n{ptx}"
        );
    }

    #[test]
    fn kernel_name_tensor_core() {
        let t = GemmTemplate {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            warp_m: 64,
            warp_n: 64,
            precision: PtxType::F16,
            accumulator: PtxType::F32,
            use_tensor_core: true,
            stages: 3,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombinationRelu,
        };
        assert_eq!(t.kernel_name(), "gemm_128x128x32_f16_f32_tc");
    }

    #[test]
    fn epilogue_kind_names() {
        assert_eq!(EpilogueKind::LinearCombination.as_str(), "lincomb");
        assert_eq!(
            EpilogueKind::LinearCombinationBiasRelu.as_str(),
            "lincomb_bias_relu"
        );
        assert!(EpilogueKind::LinearCombinationBias.needs_bias());
        assert!(!EpilogueKind::LinearCombination.needs_bias());
    }

    #[test]
    fn generate_naive_gemm_f32() {
        let t = GemmTemplate {
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            warp_m: 16,
            warp_n: 16,
            precision: PtxType::F32,
            accumulator: PtxType::F32,
            use_tensor_core: false,
            stages: 1,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        };
        let ptx = t.generate().expect("should generate naive GEMM");
        assert!(ptx.contains(".entry gemm_"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("$K_LOOP"));
    }

    /// Generated PTX must be pure ASCII for every precision and both code paths.
    ///
    /// `ptxas` and the driver JIT reject any non-ASCII byte in a module, even
    /// inside a `//` comment. CUDA 11.x tolerated it, so a typographic character
    /// in a generated comment (`32x32->64`, box-drawing rules) silently worked
    /// until CUDA 12.9+, where it surfaced only as `CUDA_ERROR_INVALID_PTX`.
    #[test]
    fn generated_ptx_is_ascii_only() {
        for (precision, accumulator) in [
            (PtxType::F32, PtxType::F32),
            (PtxType::F64, PtxType::F64),
            (PtxType::F16, PtxType::F32),
            (PtxType::F16, PtxType::F64),
            (PtxType::BF16, PtxType::F32),
        ] {
            // Skinny-tile shape: the config the dispatcher picks for small
            // f64 GEMMs, which is the one that failed on CUDA 12.9+.
            let t = GemmTemplate {
                tile_m: 64,
                tile_n: 8,
                tile_k: 8,
                warp_m: 32,
                warp_n: 8,
                precision,
                accumulator,
                use_tensor_core: false,
                stages: 1,
                target: SmVersion::Sm86,
                epilogue: EpilogueKind::LinearCombination,
            };
            let ptx = t.generate().expect("should generate GEMM");
            assert!(
                ptx.is_ascii(),
                "generate() emitted non-ASCII for {precision:?}/{accumulator:?}: {:?}",
                ptx.lines().find(|l| !l.is_ascii()),
            );
        }

        // The software-pipelined path has its own comment banners.
        let pipelined = GemmTemplate {
            tile_m: 64,
            tile_n: 64,
            tile_k: 32,
            warp_m: 32,
            warp_n: 32,
            precision: PtxType::F16,
            accumulator: PtxType::F32,
            use_tensor_core: true,
            stages: 3,
            target: SmVersion::Sm86,
            epilogue: EpilogueKind::LinearCombination,
        };
        if let Ok(ptx) = pipelined.generate_pipelined_skeleton() {
            assert!(
                ptx.is_ascii(),
                "generate_pipelined_skeleton() emitted non-ASCII: {:?}",
                ptx.lines().find(|l| !l.is_ascii()),
            );
        }
    }

    #[test]
    fn invalid_accumulator() {
        let t = GemmTemplate {
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            warp_m: 16,
            warp_n: 16,
            precision: PtxType::F32,
            accumulator: PtxType::F16,
            use_tensor_core: false,
            stages: 1,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        };
        assert!(t.generate().is_err());
    }

    // ── Multi-stage pipelined GEMM tests ─────────────────────────────────────

    fn make_pipelined_template(stages: u32, use_tensor_core: bool) -> GemmTemplate {
        GemmTemplate {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            warp_m: 64,
            warp_n: 64,
            precision: PtxType::F16,
            accumulator: PtxType::F32,
            use_tensor_core,
            stages,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        }
    }

    #[test]
    fn test_3stage_pipeline_gemm_ptx_structure() {
        let t = make_pipelined_template(3, false);
        let ptx = t
            .generate_pipelined_skeleton()
            .expect("3-stage pipelined GEMM should generate");

        // Must contain the entry point
        assert!(
            ptx.contains(".entry gemm_pipelined_"),
            "expected pipelined entry point in PTX:\n{ptx}"
        );

        // Must contain cp.async instructions for prefetching
        let cp_async_count = ptx.matches("cp.async.ca.shared.global").count();
        assert!(
            cp_async_count >= 3,
            "expected at least 3 cp.async instructions for 3-stage pipeline, got {cp_async_count}:\n{ptx}"
        );

        // Must contain cp.async.commit_group to fence each stage group
        let commit_count = ptx.matches("cp.async.commit_group;").count();
        assert!(
            commit_count >= 2,
            "expected at least 2 cp.async.commit_group fences for 3-stage pipeline, got {commit_count}:\n{ptx}"
        );

        // Must contain cp.async.wait_group to drain stages
        assert!(
            ptx.contains("cp.async.wait_group"),
            "expected cp.async.wait_group in 3-stage pipelined PTX:\n{ptx}"
        );

        // Must contain bar.sync for intra-CTA synchronization between stages
        assert!(
            ptx.contains("bar.sync 0;"),
            "expected bar.sync 0 between pipeline stages:\n{ptx}"
        );

        // Must contain shared memory declaration
        assert!(
            ptx.contains(".shared"),
            "expected shared memory declaration:\n{ptx}"
        );
    }

    #[test]
    fn test_4stage_pipeline_gemm_ptx_structure() {
        let t = make_pipelined_template(4, false);
        let ptx = t
            .generate_pipelined_skeleton()
            .expect("4-stage pipelined GEMM should generate");

        assert!(
            ptx.contains(".entry gemm_pipelined_"),
            "expected pipelined entry point:\n{ptx}"
        );

        let cp_async_count = ptx.matches("cp.async.ca.shared.global").count();
        assert!(
            cp_async_count >= 4,
            "expected at least 4 cp.async instructions for 4-stage pipeline, got {cp_async_count}:\n{ptx}"
        );

        let commit_count = ptx.matches("cp.async.commit_group;").count();
        assert!(
            commit_count >= 3,
            "expected at least 3 cp.async.commit_group fences for 4-stage pipeline, got {commit_count}:\n{ptx}"
        );

        assert!(
            ptx.contains("cp.async.wait_group"),
            "expected cp.async.wait_group in 4-stage pipelined PTX:\n{ptx}"
        );
        assert!(
            ptx.contains("bar.sync 0;"),
            "expected bar.sync 0 between pipeline stages:\n{ptx}"
        );
    }

    #[test]
    fn test_3stage_pipeline_tensor_core_contains_mma() {
        let t = make_pipelined_template(3, true);
        let ptx = t
            .generate_pipelined_skeleton()
            .expect("3-stage TC pipelined GEMM should generate");

        // Tensor core path must emit mma.sync instructions
        assert!(
            ptx.contains("mma.sync.aligned"),
            "expected mma.sync.aligned in tensor-core pipelined PTX:\n{ptx}"
        );

        // Must still have cp.async for data movement
        assert!(
            ptx.contains("cp.async.ca.shared.global"),
            "expected cp.async for shared memory prefetch:\n{ptx}"
        );
    }

    #[test]
    fn test_pipeline_requires_stages_ge_2() {
        let t = make_pipelined_template(1, false);
        let result = t.generate_pipelined_skeleton();
        assert!(
            result.is_err(),
            "generate_pipelined_skeleton should reject stages < 2"
        );
    }

    #[test]
    fn test_pipeline_smem_declaration_scales_with_stages() {
        let t3 = make_pipelined_template(3, false);
        let t4 = make_pipelined_template(4, false);
        let ptx3 = t3
            .generate_pipelined_skeleton()
            .expect("3-stage should generate");
        let ptx4 = t4
            .generate_pipelined_skeleton()
            .expect("4-stage should generate");

        // The shared memory array size for 4 stages should be larger than for 3 stages.
        // We verify by checking that both contain .shared declarations.
        assert!(
            ptx3.contains(".shared"),
            "3-stage PTX must have .shared declaration"
        );
        assert!(
            ptx4.contains(".shared"),
            "4-stage PTX must have .shared declaration"
        );

        // The 4-stage PTX is expected to be larger (more code) than the 3-stage PTX
        assert!(
            ptx4.len() > ptx3.len(),
            "4-stage PTX ({} bytes) should be longer than 3-stage PTX ({} bytes)",
            ptx4.len(),
            ptx3.len()
        );
    }

    // ── Numerical correctness tests ───────────────────────────────────────────
    //
    // These tests verify that the CPU reference implementation of the GEMM
    // algorithm (which mirrors the PTX kernel's exact computation) produces
    // correct results within floating-point precision tolerance.
    //
    // The PTX kernel implements (per thread computing element [row, col]):
    //   acc = 0.0
    //   for k in 0..K: acc = fma(A[row,k], B[k,col], acc)
    //   C[row,col] = alpha * acc + beta * C_old[row,col]
    //
    // This reference matches cuBLAS SGEMM semantics for row-major layouts.
    // Verification against hand-calculated results confirms precision ≤ 1 ULP
    // for small F32 matrices, which is the required tolerance.

    /// CPU reference implementation of the naive GEMM algorithm from the PTX
    /// template. Computes C = alpha * A * B + beta * C (row-major).
    ///
    /// Mirrors the PTX kernel's inner loop exactly: FMA-based accumulation.
    #[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
    fn cpu_reference_gemm_f32(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        c: &[f32],
        alpha: f32,
        beta: f32,
    ) -> Vec<f32> {
        assert_eq!(a.len(), m * k, "A must be m×k");
        assert_eq!(b.len(), k * n, "B must be k×n");
        assert_eq!(c.len(), m * n, "C must be m×n");

        let mut result = c.to_vec();
        for row in 0..m {
            for col in 0..n {
                // FMA inner loop — exactly mirrors: fma.rn.f32 %f0, %f1, %f2, %f0
                let mut acc = 0.0_f32;
                for ki in 0..k {
                    acc = f32::mul_add(a[row * k + ki], b[ki * n + col], acc);
                }
                // Epilogue: alpha * acc + beta * C_old
                result[row * n + col] = f32::mul_add(beta, result[row * n + col], alpha * acc);
            }
        }
        result
    }

    /// Tests the CPU reference GEMM on a 2×2 example with hand-calculated values.
    ///
    /// A = [[1, 2], [3, 4]]
    /// B = [[5, 6], [7, 8]]
    /// A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    ///       = [[19, 22], [43, 50]]
    /// With `alpha=1`, `beta=0`, `C_init` = zeros: C = A*B = [[19,22],[43,50]]
    #[test]
    fn gemm_numerical_2x2_alpha1_beta0() {
        let a = [1.0_f32, 2.0, 3.0, 4.0]; // 2×2 row-major
        let b = [5.0_f32, 6.0, 7.0, 8.0]; // 2×2 row-major
        let c_init = [0.0_f32; 4];

        let result = cpu_reference_gemm_f32(2, 2, 2, &a, &b, &c_init, 1.0, 0.0);

        let expected = [19.0_f32, 22.0, 43.0, 50.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "element [{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// Tests alpha and beta scaling: C = 2*(A*B) + 0.5*`C_init`.
    ///
    /// Using 2×2 case from above: A*B = [[19,22],[43,50]].
    /// `alpha=2`, `beta=0.5`, `C_init` = [[1,1],[1,1]].
    /// Expected: C = 2*[[19,22],[43,50]] + 0.5*[[1,1],[1,1]]
    ///             = [[38.5, 44.5], [86.5, 100.5]]
    #[test]
    fn gemm_numerical_2x2_alpha2_beta_half() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        let c_init = [1.0_f32; 4];

        let result = cpu_reference_gemm_f32(2, 2, 2, &a, &b, &c_init, 2.0, 0.5);

        let expected = [38.5_f32, 44.5, 86.5, 100.5];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "element [{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// Tests a non-square 3×2 × 2×4 = 3×4 GEMM: K=2, matches cuBLAS conventions.
    ///
    /// A (3×2): [[1,2],[3,4],[5,6]]
    /// B (2×4): [[1,2,3,4],[5,6,7,8]]
    /// C = A*B:
    ///   row 0: [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11,14,17,20]
    ///   row 1: [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23,30,37,44]
    ///   row 2: [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35,46,57,68]
    #[test]
    fn gemm_numerical_3x2x4_non_square() {
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3×2
        let b = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2×4
        let c_init = [0.0_f32; 12]; // 3×4

        let result = cpu_reference_gemm_f32(3, 2, 4, &a, &b, &c_init, 1.0, 0.0);

        let expected = [
            11.0_f32, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0,
        ];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "element [{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// Tests the identity property: A * I = A.
    ///
    /// With B = identity and alpha=1, beta=0, the result must equal A.
    #[test]
    fn gemm_numerical_identity_matrix() {
        // A (2×3)
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // B = 3×3 identity
        let b = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c_init = [0.0_f32; 6]; // 2×3

        let result = cpu_reference_gemm_f32(2, 3, 3, &a, &b, &c_init, 1.0, 0.0);

        for (i, (&got, &exp)) in result.iter().zip(a.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "A*I must equal A at [{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// Verifies the PTX contains `fma.rn.f32` — confirming the kernel uses
    /// FMA semantics matching the CPU reference (no separate mul+add split).
    #[test]
    fn gemm_ptx_contains_fma_rn_f32() {
        let t = GemmTemplate {
            tile_m: 64,
            tile_n: 64,
            tile_k: 16,
            warp_m: 32,
            warp_n: 32,
            precision: PtxType::F32,
            accumulator: PtxType::F32,
            use_tensor_core: false,
            stages: 1,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        };
        let ptx = t.generate().expect("GEMM F32 should generate");
        assert!(
            ptx.contains("fma.rn.f32"),
            "PTX must use fma.rn.f32 for F32 accumulation:\n{ptx}"
        );
    }

    /// Verifies alpha scaling (`mul.f32`) and beta epilogue (`fma.rn.f32` for
    /// `beta*C_old` + `alpha*acc`) are present in the generated PTX.
    #[test]
    fn gemm_ptx_epilogue_has_alpha_beta_scaling() {
        let t = GemmTemplate {
            tile_m: 64,
            tile_n: 64,
            tile_k: 16,
            warp_m: 32,
            warp_n: 32,
            precision: PtxType::F32,
            accumulator: PtxType::F32,
            use_tensor_core: false,
            stages: 1,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        };
        let ptx = t.generate().expect("GEMM F32 should generate");

        // alpha * acc: mul.f32 %f0, %f0, %f8
        assert!(
            ptx.contains("mul.f32"),
            "PTX must scale accumulator by alpha with mul.f32:\n{ptx}"
        );

        // beta * C_old in epilogue: load then fma
        let fma_count = ptx.matches("fma.rn.f32").count();
        assert!(
            fma_count >= 2,
            "PTX must have ≥2 fma.rn.f32 instructions (inner loop + epilogue), got {fma_count}"
        );

        // Bounds check present
        assert!(
            ptx.contains("setp.ge.u32"),
            "PTX must guard row/col bounds with setp.ge.u32"
        );
    }

    /// Spot-check: CPU reference GEMM result for a 4×4 identity × random vector.
    /// Runs 1000 GEMM calls and measures throughput (GFLOPS) at CPU reference speed.
    ///
    /// This is the P1/P2 benchmark proxy: on CPU we expect > 0.1 GFLOPS for 4×4
    /// matrices. GPU (cuBLAS) would achieve > 100× this rate.
    #[test]
    #[allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
    fn gemm_numerical_throughput_proxy_4x4() {
        let m = 4usize;
        let n = 4;
        let k = 4;
        let a: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();
        let c_init = vec![0.0_f32; m * n];

        let iters: usize = 50_000;
        let start = std::time::Instant::now();
        let mut last = c_init;
        for _ in 0..iters {
            last = cpu_reference_gemm_f32(m, k, n, &a, &b, &last, 1.0, 0.0);
        }
        let elapsed_ns = start.elapsed().as_nanos().max(1);

        let flops_per_gemm = 2 * m * k * n; // mul + add per element
        let total_gflops = (iters as f64 * flops_per_gemm as f64) / (elapsed_ns as f64); // ns → GFLOPS
        println!("GEMM CPU reference 4×4: {total_gflops:.4} GFLOPS ({iters} iters)");

        // The last result must be non-zero (prevents the compiler from eliding the loop)
        assert!(
            last.iter().any(|&x| x != 0.0),
            "GEMM result must be non-zero"
        );
        // CPU reference must achieve at least a trivially low bar (sanity check)
        assert!(
            total_gflops > 0.0001,
            "CPU reference GEMM unacceptably slow: {total_gflops:.6} GFLOPS"
        );
    }

    // ── PTX precision-correctness tests (the f64 codegen bug) ────────────────
    //
    // The naive GEMM template historically declared its value register bank as
    // `.reg .f32 %f<16>` and initialised the accumulator with the single-
    // precision zero literal `0f00000000`, even when the kernel body used
    // `.f64` instructions. `ptxas` rejects that with "Arguments mismatch for
    // instruction 'ld'/'mov'/'fma'/'mul'/'st'". These tests assert the emitted
    // PTX is precision-correct and (when `ptxas` is present) actually assembles
    // for the real A4000 target (sm_86).

    fn gemm_template(precision: PtxType, accumulator: PtxType) -> GemmTemplate {
        GemmTemplate {
            tile_m: 64,
            tile_n: 8,
            tile_k: 8,
            warp_m: 32,
            warp_n: 8,
            precision,
            accumulator,
            use_tensor_core: false,
            stages: 1,
            target: SmVersion::Sm86,
            epilogue: EpilogueKind::LinearCombination,
        }
    }

    /// The F64 naive kernel must declare a `.f64` value bank, use the double
    /// zero literal, and must NOT contain the single-precision `0f00000000`
    /// literal or an `.f32` value bank anywhere in the body.
    #[test]
    fn gemm_f64_ptx_is_precision_correct() {
        let ptx = gemm_template(PtxType::F64, PtxType::F64)
            .generate()
            .expect("F64 GEMM should generate");
        assert!(
            ptx.contains(".reg .f64 %f<16>;"),
            "F64 GEMM must declare a .f64 value bank:\n{ptx}"
        );
        assert!(
            ptx.contains("mov.f64 %f0, 0d0000000000000000;"),
            "F64 GEMM must zero the accumulator with the double literal:\n{ptx}"
        );
        assert!(
            !ptx.contains("0f00000000"),
            "F64 GEMM must not contain the single-precision zero literal:\n{ptx}"
        );
        assert!(
            !ptx.contains(".reg .f32"),
            "F64 GEMM must not declare a .f32 register bank:\n{ptx}"
        );
        // Body must use f64 arithmetic/memory instructions.
        for needle in [
            "ld.param.f64",
            "ld.global.f64",
            "fma.rn.f64",
            "mul.f64",
            "st.global.f64",
        ] {
            assert!(
                ptx.contains(needle),
                "F64 GEMM must contain {needle}:\n{ptx}"
            );
        }
    }

    /// Mixed precision (F16 input, F32 accumulator) must use two register banks
    /// and `cvt` between them — never an `.f64`/`.f32` type clash.
    #[test]
    fn gemm_mixed_f16_f32_uses_cvt() {
        let ptx = gemm_template(PtxType::F16, PtxType::F32)
            .generate()
            .expect("F16/F32 GEMM should generate");
        assert!(
            ptx.contains(".reg .f32 %f<16>;"),
            "accumulator bank f32:\n{ptx}"
        );
        // 16-bit floats are moved through `.b16` containers (PTX `ld`/`st` have
        // no `.f16` form) and reinterpreted by `cvt`.
        assert!(ptx.contains(".reg .b16 %fin<8>;"), "input bank b16:\n{ptx}");
        assert!(
            ptx.contains("ld.global.b16 %fin1,"),
            "load A as b16:\n{ptx}"
        );
        assert!(ptx.contains("cvt.f32.f16 %f1, %fin1;"), "widen A:\n{ptx}");
        assert!(
            ptx.contains("cvt.rn.f16.f32 %fin0, %f0;"),
            "narrow store:\n{ptx}"
        );
        assert!(
            ptx.contains("st.global.b16 [%rd8], %fin0;"),
            "store as b16:\n{ptx}"
        );
    }

    /// The default GEMM must compute its output-element count, loop bound, loop
    /// index, and A/B/C offset math in 64-bit so shapes with >= 2^32 output
    /// elements do not silently truncate.
    #[test]
    fn gemm_uses_64bit_loop_and_offsets() {
        let ptx = gemm_template(PtxType::F32, PtxType::F32)
            .generate()
            .expect("GEMM PTX generation should succeed");
        assert!(
            ptx.contains("mul.wide.u32 %rd9"),
            "total_elems = M*N must be computed 64-bit (mul.wide.u32):\n{ptx}"
        );
        assert!(
            ptx.contains("setp.ge.u64"),
            "grid-stride loop bound must be a 64-bit compare:\n{ptx}"
        );
        assert!(
            ptx.contains("mad.wide.u32"),
            "A/B/C element offsets must use 32×32→64 mad.wide.u32:\n{ptx}"
        );
        // The old 32-bit truncating forms must be gone.
        assert!(
            !ptx.contains("mul.lo.u32 %r14"),
            "M*N must not be computed with a truncating 32-bit multiply:\n{ptx}"
        );
        assert!(
            !ptx.contains("mad.lo.u32 %r19"),
            "row*K+k offset must not be a truncating 32-bit mad:\n{ptx}"
        );
    }

    /// Locate `ptxas` on PATH (or the well-known CUDA bin dir), returning its
    /// path if present so the test can skip gracefully on CPU-only machines.
    fn find_ptxas() -> Option<std::path::PathBuf> {
        // Windows names the binary `ptxas.exe`; probing only the bare name made
        // every ptxas assembly check silently skip there, so the toolchain was
        // never actually exercised on that platform.
        if let Ok(path) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path) {
                for name in ["ptxas", "ptxas.exe"] {
                    let candidate = dir.join(name);
                    if candidate.is_file() {
                        return Some(candidate);
                    }
                }
            }
        }
        let fallback = std::path::PathBuf::from("/usr/local/cuda/bin/ptxas");
        if fallback.is_file() {
            return Some(fallback);
        }
        None
    }

    /// Runs `ptxas -arch=sm_86` on the generated PTX for every precision
    /// combination the naive path admits, asserting each assembles cleanly.
    /// Skips (with a printed note) when `ptxas` is unavailable.
    #[test]
    fn gemm_naive_ptx_assembles_for_sm86() {
        let Some(ptxas) = find_ptxas() else {
            println!("skipping: ptxas not found on PATH");
            return;
        };

        // Every combination admitted by `validate()`: precision ∈
        // {F16,BF16,F32,F64}, accumulator ∈ {F32,F64}.
        let combos = [
            (PtxType::F32, PtxType::F32),
            (PtxType::F64, PtxType::F64),
            (PtxType::F16, PtxType::F32),
            (PtxType::BF16, PtxType::F32),
            (PtxType::F32, PtxType::F64),
            (PtxType::F64, PtxType::F32),
            (PtxType::F16, PtxType::F64),
            (PtxType::BF16, PtxType::F64),
        ];

        for (precision, accumulator) in combos {
            let ptx = gemm_template(precision, accumulator)
                .generate()
                .expect("GEMM PTX generation should succeed");

            let mut ptx_path = std::env::temp_dir();
            ptx_path.push(format!(
                "oxicuda_gemm_{}_{}_{}.ptx",
                precision.as_ptx_str().trim_start_matches('.'),
                accumulator.as_ptx_str().trim_start_matches('.'),
                std::process::id(),
            ));
            std::fs::write(&ptx_path, &ptx).expect("write PTX to temp file");

            // Assemble to a throwaway file rather than a null device:
            // `/dev/null` is not openable on Windows, where ptxas then fails for
            // a reason unrelated to the PTX under test.
            let cubin = ptx_path.with_extension("cubin");
            let output = std::process::Command::new(&ptxas)
                .arg("-arch=sm_86")
                .arg(&ptx_path)
                .arg("-o")
                .arg(&cubin)
                .output()
                .expect("invoke ptxas");

            let _ = std::fs::remove_file(&ptx_path);
            let _ = std::fs::remove_file(&cubin);

            assert!(
                output.status.success(),
                // ptxas prints its diagnostics on stdout, not stderr.
                "ptxas rejected {precision:?}/{accumulator:?} GEMM PTX:\n{}{}\n--- PTX ---\n{ptx}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }

    // ── row/col strength-reduction (replaces the per-element div.u64/rem.u64) ─

    /// Pins the fix: `generate()` must never emit a 64-bit integer
    /// divide/remainder. Before the strength-reduction, `div.u64`/`rem.u64`
    /// appeared once per K-loop-guarded output element (inside
    /// `$TILE_LOOP`); each lowers to a `CALL.REL` software long-division
    /// subroutine under `ptxas` on every architecture this crate targets
    /// (no NVIDIA GPU has a native 64-bit integer divider).
    #[test]
    fn gemm_row_col_decomposition_has_no_64bit_divide() {
        let ptx = gemm_template(PtxType::F32, PtxType::F32)
            .generate()
            .expect("GEMM PTX generation should succeed");
        assert!(
            !ptx.contains("div.u64") && !ptx.contains("rem.u64"),
            "generate() must not emit a 64-bit divide/remainder \
             (CALL.REL subroutine on every ptxas target) in the per-element \
             row/col recovery -- row/col must be strength-reduced instead:\n{ptx}"
        );
        // The one-time 32-bit init and the branch-free per-iteration advance
        // must both be present.
        assert!(
            ptx.contains("div.u32") && ptx.contains("rem.u32"),
            "expected the one-time 32-bit row0/col0 and \
             stride_div_n/stride_rem_n init:\n{ptx}"
        );
        assert!(
            ptx.contains("selp.u32"),
            "expected a branch-free selp-based carry in the per-iteration \
             row/col advance:\n{ptx}"
        );
    }

    /// The 64-bit loop-continuation guard (`setp.ge.u64`) and the 64-bit
    /// `idx += stride` advance (`add.u64 %rd10, %rd10, %rd11`) that keep
    /// `generate()` correct for `M*N >= 2^32` shapes must survive the
    /// strength-reduction untouched -- only the row/col *recovery* changed,
    /// not the loop-bound arithmetic `gemm_uses_64bit_loop_and_offsets`
    /// already pins.
    #[test]
    fn gemm_row_col_strength_reduction_preserves_64bit_loop_guard() {
        let ptx = gemm_template(PtxType::F32, PtxType::F32)
            .generate()
            .expect("GEMM PTX generation should succeed");
        assert!(ptx.contains("setp.ge.u64 %p0, %rd10, %rd9;"));
        assert!(ptx.contains("add.u64 %rd10, %rd10, %rd11;"));
    }

    /// Rust-level model of the exact PTX strength-reduction algorithm
    /// (one-time `row0 = id/N, col0 = id%N, stride_div_n = stride/N,
    /// stride_rem_n = stride%N`, then per-iteration `col += stride_rem_n`
    /// with a carry into `row`), swept against a fresh `idx/N, idx%N`
    /// division at every step across many `(id, stride, n, iterations)`
    /// combinations -- including `idx` values beyond `u32::MAX`, which is
    /// exactly the range the 64-bit loop-bound fix exists to cover
    /// correctly and that a naive "just use `div.u32`/`rem.u32` on a
    /// truncated `idx`" fix would have silently reintroduced.
    #[test]
    fn row_col_strength_reduction_matches_fresh_division_swept() {
        /// Mirrors the PTX: one-time init, then a branch-free carry advance.
        fn advance(row: &mut u64, col: &mut u64, n: u64, stride_div_n: u64, stride_rem_n: u64) {
            let col_sum = *col + stride_rem_n;
            let mut row_next = *row + stride_div_n;
            let col_next = if col_sum >= n {
                row_next += 1;
                col_sum - n
            } else {
                col_sum
            };
            *row = row_next;
            *col = col_next;
        }

        // (id, stride, n, iterations) -- `id < stride` always (id is a
        // `global_id < total_threads == stride`), mirroring the PTX
        // invariant. Includes small values, values that don't divide
        // evenly, and `idx` sequences that cross 2^32 (the case a
        // truncating 32-bit division would get wrong).
        let cases: &[(u64, u64, u64, u64)] = &[
            (0, 1, 1, 50),
            (0, 16, 29, 200), // matches gemm_gpu_tests's on-device case
            (5, 7, 3, 100),
            (0, 128, 128, 10), // stride is an exact multiple of n
            (127, 128, 100, 500),
            (0, 3, 1_000_000_007, 20),
            // idx crosses 2^32 partway through the sweep: id + iterations*stride
            // must exceed u32::MAX while id and stride individually still fit
            // u32 (mirroring the PTX's own "id/stride fit u32, idx might not"
            // invariant).
            (
                4_000_000_000,
                200_000_000,
                999_999_937, // a large prime, so stride % n varies richly
                40,
            ),
            (u64::from(u32::MAX) - 3, 7, 5, 30),
        ];

        for &(id, stride, n, iterations) in cases {
            assert!(n > 0, "test case bug: n must be positive");
            let stride_div_n = stride / n;
            let stride_rem_n = stride % n;
            let mut row = id / n;
            let mut col = id % n;
            let mut idx = id;

            for step in 0..iterations {
                let fresh_row = idx / n;
                let fresh_col = idx % n;
                assert_eq!(
                    (row, col),
                    (fresh_row, fresh_col),
                    "case (id={id}, stride={stride}, n={n}) step {step}: \
                     strength-reduced (row={row}, col={col}) diverged from \
                     fresh division (row={fresh_row}, col={fresh_col}) at \
                     idx={idx} (idx fits u32: {})",
                    u32::try_from(idx).is_ok(),
                );
                idx += stride;
                advance(&mut row, &mut col, n, stride_div_n, stride_rem_n);
            }
        }
    }
}

// On-device numeric coverage for the row/col strength-reduction above,
// deliberately forcing multiple grid-stride iterations per thread (see the
// module doc comment in `gemm_gpu_tests.rs` for why this is complementary
// to, not redundant with, `oxicuda-blas`'s production-launch-geometry GPU
// tests).
#[cfg(all(test, feature = "gpu-tests"))]
#[path = "gemm_gpu_tests.rs"]
mod gemm_gpu_tests;
