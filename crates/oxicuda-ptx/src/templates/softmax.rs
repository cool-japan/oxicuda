//! Numerically stable softmax kernel template.
//!
//! Generates PTX kernels for computing row-wise softmax using the numerically
//! stable three-pass algorithm:
//!
//! 1. **Find row maximum**: `m = max(x[0], x[1], ..., x[N-1])`
//! 2. **Exponentiate and sum**: `s = sum(exp(x[i] - m))`
//! 3. **Normalize**: `y[i] = exp(x[i] - m) / s`
//!
//! The implementation strategy depends on the row size:
//!
//! - **`row_size` <= 32**: Warp shuffle reduction (no shared memory needed)
//! - **`row_size` <= 1024**: Shared memory block reduction
//! - **`row_size` > 1024**: Multi-block (currently unsupported; falls back to error)
//!
//! This module currently implements the warp shuffle variant for small rows.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::softmax::SoftmaxTemplate;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = SoftmaxTemplate {
//!     precision: PtxType::F32,
//!     target: SmVersion::Sm80,
//!     row_size: 32,
//! };
//! let ptx = template.generate().expect("PTX generation failed");
//! assert!(ptx.contains("softmax"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Template for generating numerically stable softmax PTX kernels.
///
/// Each block processes one row of the input matrix. The row size determines
/// the reduction strategy (warp shuffle vs shared memory).
pub struct SoftmaxTemplate {
    /// The data precision.
    pub precision: PtxType,
    /// The target GPU architecture.
    pub target: SmVersion,
    /// Number of elements per row. Must be <= 1024 for current implementation.
    pub row_size: u32,
}

impl SoftmaxTemplate {
    /// Returns the kernel function name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        format!("softmax_{type_str}_r{}", self.row_size)
    }

    /// Generates the complete PTX module text for the softmax kernel.
    ///
    /// Kernel parameters:
    /// - `input`: pointer to input matrix (`batch_size` x `row_size`, row-major)
    /// - `output`: pointer to output matrix (same shape)
    /// - `batch_size`: number of rows
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if:
    /// - The precision is not a supported float type
    /// - The row size exceeds the supported limit (> 1024)
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        if self.row_size <= 32 {
            self.generate_warp_shuffle()
        } else {
            self.generate_shared_memory()
        }
    }

    /// Validates template parameters.
    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(
            self.precision,
            PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64
        ) {
            return Err(PtxGenError::InvalidType(format!(
                "softmax requires F16, BF16, F32, or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        if self.row_size == 0 {
            return Err(PtxGenError::GenerationFailed(
                "row_size must be > 0".to_string(),
            ));
        }
        if self.row_size > 1024 {
            return Err(PtxGenError::GenerationFailed(format!(
                "row_size {} exceeds current limit of 1024; multi-block softmax not yet implemented",
                self.row_size
            )));
        }
        Ok(())
    }

    /// Generates a warp-shuffle-based softmax for `row_size` <= 32.
    ///
    /// One warp processes one row. Each thread handles one element.
    /// The three-pass reduction (max, exp+sum, normalize) uses
    /// `shfl.sync.down.b32` for intra-warp communication.
    #[allow(clippy::too_many_lines)]
    fn generate_warp_shuffle(&self) -> Result<String, PtxGenError> {
        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        let neg_inf = match self.precision {
            PtxType::F64 => "0dFFF0000000000000",
            _ => "0fFF800000",
        };

        let mut ptx = String::with_capacity(4096);

        // Header
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Declarations
        writeln!(ptx, "    .reg .b32 %r<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<4>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread indexing: warp_id = global_tid / 32 => row index
        //                  lane_id = global_tid % 32 => element within row
        writeln!(ptx, "    // Compute row and lane indices").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;").map_err(PtxGenError::FormatError)?;
        // r4 = warp_id (row), r5 = lane_id (element)
        writeln!(ptx, "    shr.u32 %r4, %r3, 5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    and.b32 %r5, %r3, 31;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check: row < batch_size
        writeln!(ptx, "    ld.param.u32 %r6, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r4, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $SM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load element (or -INF for out-of-row lanes)
        let row_size = self.row_size;
        writeln!(ptx, "    // Load element").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.u32 %p1, %r5, {row_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f0, {neg_inf};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p1 bra $SKIP_LOAD_SM;").map_err(PtxGenError::FormatError)?;

        // Compute address: input[row * row_size + lane]
        writeln!(ptx, "    mad.lo.u32 %r7, %r4, {row_size}, %r5;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd1, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd1, %rd1, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd2, %rd0, %rd1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f0, [%rd2];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$SKIP_LOAD_SM:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 1: find row max via warp shuffle
        writeln!(ptx, "    // Pass 1: row-wise max reduction").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f1, %f0;").map_err(PtxGenError::FormatError)?;
        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(
                ptx,
                "    shfl.sync.down.b32 %f2, %f1, {offset}, 31, 0xFFFFFFFF;"
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    max{ty} %f1, %f1, %f2;").map_err(PtxGenError::FormatError)?;
        }
        // Broadcast max from lane 0 to all lanes
        writeln!(ptx, "    shfl.sync.idx.b32 %f1, %f1, 0, 31, 0xFFFFFFFF;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 2: exp(x - max) and sum
        writeln!(ptx, "    // Pass 2: exp(x - max) and sum reduction")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub{ty} %f3, %f0, %f1;").map_err(PtxGenError::FormatError)?;
        // exp via 2^(x * log2(e)): log2(e) = 0f3FB8AA3B
        writeln!(ptx, "    mul{ty} %f3, %f3, 0f3FB8AA3B;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f3, %f3;").map_err(PtxGenError::FormatError)?;

        // Out-of-row lanes contribute 0 to the sum
        writeln!(ptx, "    @!%p1 mov{ty} %f3, 0f00000000;").map_err(PtxGenError::FormatError)?;

        // Sum reduction
        writeln!(ptx, "    mov{ty} %f4, %f3;").map_err(PtxGenError::FormatError)?;
        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(
                ptx,
                "    shfl.sync.down.b32 %f5, %f4, {offset}, 31, 0xFFFFFFFF;"
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    add{ty} %f4, %f4, %f5;").map_err(PtxGenError::FormatError)?;
        }
        // Broadcast sum from lane 0
        writeln!(ptx, "    shfl.sync.idx.b32 %f4, %f4, 0, 31, 0xFFFFFFFF;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 3: normalize and store
        writeln!(ptx, "    // Pass 3: normalize and store").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p1 bra $SM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rcp.approx{ty} %f6, %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f7, %f3, %f6;").map_err(PtxGenError::FormatError)?;

        // Store to output
        writeln!(ptx, "    ld.param.u64 %rd3, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r8, %r4, {row_size}, %r5;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd4, %r8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd4, %rd4, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd5, %rd3, %rd4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd5], %f7;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$SM_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    /// Generates a shared-memory-based softmax for 32 < `row_size` <= 1024.
    ///
    /// One block processes one row, with threads loading multiple elements.
    #[allow(clippy::too_many_lines)]
    fn generate_shared_memory(&self) -> Result<String, PtxGenError> {
        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        let neg_inf = match self.precision {
            PtxType::F64 => "0dFFF0000000000000",
            _ => "0fFF800000",
        };
        let row_size = self.row_size;
        // Use min(row_size, 256) threads per block, power-of-2
        let block_size = self.row_size.next_power_of_two().min(256);
        let smem_bytes = (block_size as usize) * byte_size;

        let mut ptx = String::with_capacity(4096);

        // Header
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    .maxntid {block_size}, 1, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b32 %r<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<12>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<4>;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    .shared .align {} .b8 smem_softmax[{}];",
            byte_size.max(4),
            smem_bytes
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Each block handles one row: row = blockIdx.x
        writeln!(ptx, "    // Block per row").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r2, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r1, %r2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $SM_BLK_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Base address for this row
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd1, %r1;").map_err(PtxGenError::FormatError)?;
        let row_bytes = row_size as usize * byte_size;
        writeln!(ptx, "    mul.lo.u64 %rd1, %rd1, {row_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd2, %rd0, %rd1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 1: Each thread finds its local max, then reduce via shared mem
        writeln!(ptx, "    // Pass 1: find row max").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f0, {neg_inf};").map_err(PtxGenError::FormatError)?;
        // Thread i handles elements i, i+blockSize, i+2*blockSize, ...
        writeln!(ptx, "    mov.u32 %r3, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$MAX_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r3, {row_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $MAX_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    max{ty} %f0, %f0, %f1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r3, %r3, {block_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $MAX_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$MAX_DONE:").map_err(PtxGenError::FormatError)?;

        // Store to shared memory and reduce
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u64 %rd6, smem_softmax;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd7, %rd6, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.shared{ty} [%rd7], %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;

        // Tree reduction for max in shared memory
        let mut stride = block_size / 2;
        while stride > 0 {
            writeln!(ptx, "    setp.lt.u32 %p2, %r0, {stride};")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    @!%p2 bra $SKIP_MAX_{stride};").map_err(PtxGenError::FormatError)?;
            let partner_off = stride as usize * byte_size;
            writeln!(ptx, "    ld.shared{ty} %f2, [%rd7+{partner_off}];")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    ld.shared{ty} %f3, [%rd7];").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    max{ty} %f3, %f3, %f2;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    st.shared{ty} [%rd7], %f3;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "$SKIP_MAX_{stride}:").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
            stride /= 2;
        }

        // Broadcast max from shared memory position 0
        writeln!(ptx, "    ld.shared{ty} %f4, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 2: exp(x - max) and partial sum
        writeln!(ptx, "    // Pass 2: exp(x - max) and sum").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f5, 0f00000000;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r3, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$EXP_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r3, {row_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $EXP_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f6, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub{ty} %f6, %f6, %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f6, %f6, 0f3FB8AA3B;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f6, %f6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add{ty} %f5, %f5, %f6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r3, %r3, {block_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $EXP_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$EXP_DONE:").map_err(PtxGenError::FormatError)?;

        // Reduce sum via shared memory
        writeln!(ptx, "    st.shared{ty} [%rd7], %f5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        stride = block_size / 2;
        while stride > 0 {
            writeln!(ptx, "    setp.lt.u32 %p2, %r0, {stride};")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    @!%p2 bra $SKIP_SUM_{stride};").map_err(PtxGenError::FormatError)?;
            let partner_off = stride as usize * byte_size;
            writeln!(ptx, "    ld.shared{ty} %f7, [%rd7+{partner_off}];")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    ld.shared{ty} %f8, [%rd7];").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    add{ty} %f8, %f8, %f7;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    st.shared{ty} [%rd7], %f8;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "$SKIP_SUM_{stride}:").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
            stride /= 2;
        }

        // Broadcast sum
        writeln!(ptx, "    ld.shared{ty} %f9, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rcp.approx{ty} %f10, %f9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 3: normalize and store
        writeln!(ptx, "    // Pass 3: normalize and store").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd8, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd9, %rd8, %rd1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r3, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$NORM_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r3, {row_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $SM_BLK_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f11, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub{ty} %f11, %f11, %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f11, %f11, 0f3FB8AA3B;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f11, %f11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f11, %f11, %f10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd10, %rd9, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd10], %f11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r3, %r3, {block_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $NORM_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$SM_BLK_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;

    #[test]
    fn kernel_name_format() {
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 32,
        };
        assert_eq!(t.kernel_name(), "softmax_f32_r32");
    }

    #[test]
    fn invalid_precision() {
        let t = SoftmaxTemplate {
            precision: PtxType::U32,
            target: SmVersion::Sm80,
            row_size: 32,
        };
        assert!(t.generate().is_err());
    }

    #[test]
    fn too_large_row() {
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 2048,
        };
        assert!(t.generate().is_err());
    }

    #[test]
    fn zero_row() {
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 0,
        };
        assert!(t.generate().is_err());
    }

    #[test]
    fn generate_warp_shuffle_softmax() {
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 32,
        };
        let ptx = t.generate().expect("should generate warp shuffle softmax");
        assert!(ptx.contains(".entry softmax_f32_r32"));
        assert!(ptx.contains("shfl.sync.down"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
    }

    #[test]
    fn generate_shared_mem_softmax() {
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 256,
        };
        let ptx = t.generate().expect("should generate shared mem softmax");
        assert!(ptx.contains(".entry softmax_f32_r256"));
        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("bar.sync 0"));
    }

    #[test]
    fn generate_non_power_of_2_row() {
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 100,
        };
        let ptx = t.generate().expect("should handle non-power-of-2 rows");
        assert!(ptx.contains(".entry softmax_f32_r100"));
    }
}
