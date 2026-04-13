//! Simplified FlashAttention-style kernel template.
//!
//! Generates PTX kernels for computing scaled dot-product attention using
//! tiled matrix multiplication with online softmax. The kernel operates on
//! a single attention head and processes sequences in tiles to reduce
//! shared memory pressure.
//!
//! **Algorithm outline (per tile)**:
//!
//! 1. Load a tile of Q into shared memory (`block_seq` x `head_dim`)
//! 2. For each K/V tile along the sequence dimension:
//!    a. Load K tile into shared memory, compute S = Q * K^T (tiled GEMM)
//!    b. Apply causal mask if enabled (set future positions to -inf)
//!    c. Online softmax: track running max and sum per row
//!    d. Load V tile into shared memory, accumulate P * V into output
//! 3. Final rescale of the accumulated output by the softmax denominator
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::attention::AttentionTemplate;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = AttentionTemplate::new(PtxType::F32, 64, 64, false);
//! let ptx = template.generate(SmVersion::Sm80).expect("PTX generation failed");
//! assert!(ptx.contains("attention"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Template for generating scaled dot-product attention PTX kernels.
///
/// Produces a simplified FlashAttention-style kernel that:
/// - Tiles Q, K, V through shared memory
/// - Uses online softmax (running max + sum) for numerical stability
/// - Optionally applies a causal mask
///
/// Each thread block processes one tile of Q rows (`block_seq` rows) for
/// one attention head. The kernel iterates over K/V tiles along the
/// key/value sequence length.
pub struct AttentionTemplate {
    /// Data precision (F32 or F64).
    pub precision: PtxType,
    /// Dimension of each attention head (typically 64 or 128).
    pub head_dim: u32,
    /// Number of query rows processed per block (tile height).
    pub block_seq: u32,
    /// Whether to apply causal masking (future tokens masked to -inf).
    pub causal: bool,
}

impl AttentionTemplate {
    /// Creates a new attention template with the given parameters.
    #[must_use]
    pub const fn new(precision: PtxType, head_dim: u32, block_seq: u32, causal: bool) -> Self {
        Self {
            precision,
            head_dim,
            block_seq,
            causal,
        }
    }

    /// Sets the precision type. Returns `self` for chaining.
    #[must_use]
    pub const fn with_precision(mut self, precision: PtxType) -> Self {
        self.precision = precision;
        self
    }

    /// Sets the head dimension. Returns `self` for chaining.
    #[must_use]
    pub const fn with_head_dim(mut self, head_dim: u32) -> Self {
        self.head_dim = head_dim;
        self
    }

    /// Sets the block sequence tile size. Returns `self` for chaining.
    #[must_use]
    pub const fn with_block_seq(mut self, block_seq: u32) -> Self {
        self.block_seq = block_seq;
        self
    }

    /// Enables or disables causal masking. Returns `self` for chaining.
    #[must_use]
    pub const fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Returns the kernel function name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        let causal_tag = if self.causal { "_causal" } else { "" };
        format!(
            "attention_{type_str}_hd{}_bs{}{}",
            self.head_dim, self.block_seq, causal_tag
        )
    }

    /// Validates template parameters before code generation.
    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(self.precision, PtxType::F32 | PtxType::F64) {
            return Err(PtxGenError::InvalidType(format!(
                "attention requires F32 or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }

        if self.head_dim == 0 || !self.head_dim.is_power_of_two() {
            return Err(PtxGenError::GenerationFailed(format!(
                "head_dim must be a power of 2 > 0, got {}",
                self.head_dim
            )));
        }

        if self.head_dim > 256 {
            return Err(PtxGenError::GenerationFailed(format!(
                "head_dim {} exceeds maximum of 256",
                self.head_dim
            )));
        }

        if self.block_seq == 0 || !self.block_seq.is_power_of_two() {
            return Err(PtxGenError::GenerationFailed(format!(
                "block_seq must be a power of 2 > 0, got {}",
                self.block_seq
            )));
        }

        if self.block_seq > 256 {
            return Err(PtxGenError::GenerationFailed(format!(
                "block_seq {} exceeds maximum of 256",
                self.block_seq
            )));
        }

        Ok(())
    }

    /// Generates the complete PTX module text for the attention kernel.
    ///
    /// The generated kernel expects the following parameters:
    /// - `Q`: pointer to query matrix (`seq_len` x `head_dim`, row-major)
    /// - `K`: pointer to key matrix (`seq_len` x `head_dim`, row-major)
    /// - `V`: pointer to value matrix (`seq_len` x `head_dim`, row-major)
    /// - `output`: pointer to output matrix (`seq_len` x `head_dim`, row-major)
    /// - `seq_len`: total sequence length
    /// - `scale`: softmax scaling factor (typically `1/sqrt(head_dim)`)
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if validation fails or PTX formatting fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate(&self, sm: SmVersion) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        let block_seq = self.block_seq;
        let head_dim = self.head_dim;

        let neg_inf = match self.precision {
            PtxType::F64 => "0dFFF0000000000000",
            _ => "0fFF800000",
        };
        let zero_lit = match self.precision {
            PtxType::F64 => "0d0000000000000000",
            _ => "0f00000000",
        };

        // Shared memory sizes
        let q_smem_bytes = (block_seq as usize) * (head_dim as usize) * byte_size;
        let kv_smem_bytes = (block_seq as usize) * (head_dim as usize) * byte_size;
        let scores_smem_bytes = (block_seq as usize) * (block_seq as usize) * byte_size;
        let total_smem = q_smem_bytes + kv_smem_bytes + scores_smem_bytes;

        let mut ptx = String::with_capacity(8192);

        // Header
        writeln!(ptx, ".version {}", sm.ptx_version()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", sm.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_Q,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_K,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_V,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_seq_len,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param {ty} %param_scale").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Directives and register declarations
        writeln!(ptx, "    .maxntid {block_seq}, 1, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b32 %r<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<24>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    .shared .align {} .b8 smem_attn[{}];",
            byte_size.max(4),
            total_smem
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread and block indexing
        // tid.x = thread within the Q-tile row (0..block_seq-1)
        // ctaid.x = which Q-tile block along the sequence
        writeln!(ptx, "    // Thread and block indexing").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    // Load kernel parameters").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_Q];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_K];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_V];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd3, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r2, [%param_seq_len];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param{ty} %f0, [%param_scale];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute the global query row for this thread
        // q_row = ctaid.x * block_seq + tid.x
        writeln!(ptx, "    // Compute query row index").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, {block_seq}, %r0;")
            .map_err(PtxGenError::FormatError)?;
        // Bounds check: q_row < seq_len
        writeln!(ptx, "    setp.ge.u32 %p0, %r3, %r2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $ATTN_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Shared memory layout:
        // [0..q_smem_bytes): Q tile
        // [q_smem_bytes..q_smem_bytes+kv_smem_bytes): K/V tile
        // [q_smem_bytes+kv_smem_bytes..total_smem): Scores tile
        writeln!(ptx, "    // Shared memory base pointers").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u64 %rd4, smem_attn;").map_err(PtxGenError::FormatError)?;
        let kv_smem_offset = q_smem_bytes;
        let scores_smem_offset = q_smem_bytes + kv_smem_bytes;
        writeln!(ptx, "    add.u64 %rd5, %rd4, {kv_smem_offset};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd4, {scores_smem_offset};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load Q tile: each thread loads one row of Q into shared memory
        // Q_row address = Q + q_row * head_dim * byte_size
        let row_bytes = (head_dim as usize) * byte_size;
        writeln!(ptx, "    // Load Q tile into shared memory").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd7, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd7, %rd7, {row_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd8, %rd0, %rd7;").map_err(PtxGenError::FormatError)?;
        // Copy head_dim elements from global to shared memory
        writeln!(ptx, "    cvt.u64.u32 %rd9, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd9, %rd9, {row_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd10, %rd4, %rd9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r4, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$LOAD_Q_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r4, {head_dim};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $LOAD_Q_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd11, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd12, %rd8, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd12];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd13, %rd10, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.shared{ty} [%rd13], %f1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r4, %r4, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $LOAD_Q_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$LOAD_Q_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Initialize online softmax accumulators:
        // f1 = running max (initialized to -inf)
        // f2 = running sum of exp (initialized to 0)
        // Output accumulator is kept in registers for each head_dim element
        // (simplified: we use a loop to accumulate into shared memory later)
        writeln!(ptx, "    // Initialize online softmax state")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f2, {neg_inf};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f3, {zero_lit};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Initialize output accumulator to zero (re-use scores region as output accum)
        // Each thread accumulates head_dim values; store in global output directly
        // First zero-out the output row
        writeln!(ptx, "    // Zero output accumulator row").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd14, %rd3, %rd7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r5, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$ZERO_OUT_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p2, %r5, {head_dim};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $ZERO_OUT_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd15, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd15, %rd15, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd16, %rd14, %rd15;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd16], {zero_lit};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r5, %r5, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $ZERO_OUT_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$ZERO_OUT_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Main loop: iterate over K/V tiles
        // r6 = kv_tile_start (0, block_seq, 2*block_seq, ...)
        writeln!(ptx, "    // Main K/V tile loop").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r6, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$KV_TILE_LOOP:").map_err(PtxGenError::FormatError)?;

        // Determine the upper bound for KV iteration
        if self.causal {
            // For causal: only iterate up to (ctaid.x + 1) * block_seq
            writeln!(
                ptx,
                "    // Causal: limit KV tiles to current Q tile position"
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    add.u32 %r7, %r1, 1;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    mul.lo.u32 %r7, %r7, {block_seq};")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    min.u32 %r7, %r7, %r2;").map_err(PtxGenError::FormatError)?;
        } else {
            writeln!(ptx, "    mov.u32 %r7, %r2;").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx, "    setp.ge.u32 %p3, %r6, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p3 bra $KV_TILE_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load K tile row (using tid.x to cooperatively load)
        // K_row = K + (kv_tile_start + tid.x) * head_dim * byte_size
        writeln!(ptx, "    // Load K tile row into shared memory")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r8, %r6, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.u32 %p4, %r8, %r2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p4 bra $SKIP_LOAD_K;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd17, %r8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd17, %rd17, {row_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd18, %rd1, %rd17;").map_err(PtxGenError::FormatError)?;
        // Store into K shared memory region
        writeln!(ptx, "    cvt.u64.u32 %rd19, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd19, %rd19, {row_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd20, %rd5, %rd19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r9, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$LOAD_K_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p5, %r9, {head_dim};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p5 bra $LOAD_K_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd21, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd21, %rd21, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd22, %rd18, %rd21;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f4, [%rd22];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd23, %rd20, %rd21;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.shared{ty} [%rd23], %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r9, %r9, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $LOAD_K_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$LOAD_K_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$SKIP_LOAD_K:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute Q * K^T score for this thread's Q row vs each K row in the tile
        // score[j] = sum_d(Q[row][d] * K[kv_tile_start+j][d]) * scale
        // Loop over j (K rows in tile)
        writeln!(ptx, "    // Compute Q * K^T scores for this Q row")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r10, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$SCORE_J_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p4, %r10, {block_seq};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 bra $SCORE_J_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Check if k_row = kv_tile_start + j is within seq_len
        writeln!(ptx, "    add.u32 %r11, %r6, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p5, %r11, %r2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p5 mov{ty} %f5, {neg_inf};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p5 bra $STORE_SCORE;").map_err(PtxGenError::FormatError)?;

        // Dot product: sum over head_dim
        writeln!(ptx, "    mov{ty} %f5, {zero_lit};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r12, 0;").map_err(PtxGenError::FormatError)?;
        // Q base for this thread: smem_attn + tid.x * row_bytes
        // K base for j-th row: smem_kv + j * row_bytes
        writeln!(ptx, "    cvt.u64.u32 %rd19, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd19, %rd19, {row_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd20, %rd5, %rd19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$DOT_D_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p6, %r12, {head_dim};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p6 bra $DOT_D_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd21, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd21, %rd21, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        // Load Q[tid.x][d] from Q shared mem
        writeln!(ptx, "    add.u64 %rd22, %rd10, %rd21;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.shared{ty} %f6, [%rd22];").map_err(PtxGenError::FormatError)?;
        // Load K[j][d] from K shared mem
        writeln!(ptx, "    add.u64 %rd23, %rd20, %rd21;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.shared{ty} %f7, [%rd23];").map_err(PtxGenError::FormatError)?;
        // Accumulate
        writeln!(ptx, "    fma{ty} %f5, %f6, %f7, %f5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r12, %r12, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $DOT_D_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$DOT_D_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Apply scale
        writeln!(ptx, "    mul{ty} %f5, %f5, %f0;").map_err(PtxGenError::FormatError)?;

        // Apply causal mask
        if self.causal {
            writeln!(
                ptx,
                "    // Apply causal mask: if k_pos > q_pos, set to -inf"
            )
            .map_err(PtxGenError::FormatError)?;
            // q_pos = r3 (q_row), k_pos = r11 (kv_tile_start + j)
            writeln!(ptx, "    setp.gt.u32 %p7, %r11, %r3;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    @%p7 mov{ty} %f5, {neg_inf};").map_err(PtxGenError::FormatError)?;
        }

        // Store score to scores shared memory
        writeln!(ptx, "$STORE_SCORE:").map_err(PtxGenError::FormatError)?;
        // scores_addr = scores_smem + tid.x * block_seq * byte_size + j * byte_size
        let tile_score_row_bytes = (block_seq as usize) * byte_size;
        writeln!(ptx, "    cvt.u64.u32 %rd19, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd19, %rd19, {tile_score_row_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd20, %rd6, %rd19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd21, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd21, %rd21, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd22, %rd20, %rd21;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.shared{ty} [%rd22], %f5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    add.u32 %r10, %r10, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $SCORE_J_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$SCORE_J_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Online softmax update for this tile
        // For each j in tile: update running max and sum, then accumulate P*V
        writeln!(
            ptx,
            "    // Online softmax: update max and sum, accumulate P*V"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r10, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$ONLINE_SM_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p4, %r10, {block_seq};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 bra $ONLINE_SM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Check bounds
        writeln!(ptx, "    add.u32 %r11, %r6, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p5, %r11, %r2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p5 bra $ONLINE_SM_NEXT;").map_err(PtxGenError::FormatError)?;

        // Load score
        writeln!(ptx, "    cvt.u64.u32 %rd19, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd19, %rd19, {tile_score_row_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd20, %rd6, %rd19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd21, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd21, %rd21, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd22, %rd20, %rd21;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.shared{ty} %f8, [%rd22];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Online softmax step:
        // new_max = max(old_max, score)
        // correction = exp(old_max - new_max)
        // new_sum = old_sum * correction + exp(score - new_max)
        // rescale existing output by correction
        writeln!(ptx, "    // Update online softmax state").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    max{ty} %f9, %f2, %f8;").map_err(PtxGenError::FormatError)?;
        // correction = exp(old_max - new_max): use ex2(x * log2(e))
        writeln!(ptx, "    sub{ty} %f10, %f2, %f9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f10, %f10, 0f3FB8AA3B;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f10, %f10;").map_err(PtxGenError::FormatError)?;
        // p_ij = exp(score - new_max)
        writeln!(ptx, "    sub{ty} %f11, %f8, %f9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f11, %f11, 0f3FB8AA3B;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f11, %f11;").map_err(PtxGenError::FormatError)?;
        // new_sum = old_sum * correction + p_ij
        writeln!(ptx, "    fma{ty} %f3, %f3, %f10, %f11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Rescale existing output accumulator by correction and add p_ij * V[k_row]
        writeln!(ptx, "    // Rescale output and accumulate P*V")
            .map_err(PtxGenError::FormatError)?;
        // V row address
        writeln!(ptx, "    cvt.u64.u32 %rd17, %r11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd17, %rd17, {row_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd18, %rd2, %rd17;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r12, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$PV_ACCUM_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p6, %r12, {head_dim};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p6 bra $PV_ACCUM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd19, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd19, %rd19, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        // Load current output[q_row][d]
        writeln!(ptx, "    add.u64 %rd20, %rd14, %rd19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f12, [%rd20];").map_err(PtxGenError::FormatError)?;
        // Rescale by correction
        writeln!(ptx, "    mul{ty} %f12, %f12, %f10;").map_err(PtxGenError::FormatError)?;
        // Load V[k_row][d]
        writeln!(ptx, "    add.u64 %rd21, %rd18, %rd19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f13, [%rd21];").map_err(PtxGenError::FormatError)?;
        // Accumulate: output += p_ij * V[d]
        writeln!(ptx, "    fma{ty} %f12, %f11, %f13, %f12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd20], %f12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r12, %r12, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $PV_ACCUM_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$PV_ACCUM_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Update running max
        writeln!(ptx, "    mov{ty} %f2, %f9;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$ONLINE_SM_NEXT:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r10, %r10, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $ONLINE_SM_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$ONLINE_SM_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Advance to next KV tile
        writeln!(ptx, "    add.u32 %r6, %r6, {block_seq};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $KV_TILE_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$KV_TILE_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Final normalization: divide output by softmax sum
        writeln!(ptx, "    // Final normalization: output /= sum")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rcp.approx{ty} %f14, %f3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r12, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$FINAL_NORM_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p4, %r12, {head_dim};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 bra $ATTN_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd19, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd19, %rd19, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd20, %rd14, %rd19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f15, [%rd20];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f15, %f15, %f14;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd20], %f15;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r12, %r12, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $FINAL_NORM_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$ATTN_DONE:").map_err(PtxGenError::FormatError)?;
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
    fn kernel_name_non_causal() {
        let t = AttentionTemplate::new(PtxType::F32, 64, 64, false);
        assert_eq!(t.kernel_name(), "attention_f32_hd64_bs64");
    }

    #[test]
    fn kernel_name_causal() {
        let t = AttentionTemplate::new(PtxType::F32, 128, 64, true);
        assert_eq!(t.kernel_name(), "attention_f32_hd128_bs64_causal");
    }

    #[test]
    fn kernel_name_f64() {
        let t = AttentionTemplate::new(PtxType::F64, 64, 32, false);
        assert_eq!(t.kernel_name(), "attention_f64_hd64_bs32");
    }

    #[test]
    fn invalid_precision_u32() {
        let t = AttentionTemplate::new(PtxType::U32, 64, 64, false);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_precision_f16() {
        let t = AttentionTemplate::new(PtxType::F16, 64, 64, false);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_head_dim_zero() {
        let t = AttentionTemplate::new(PtxType::F32, 0, 64, false);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_head_dim_not_power_of_two() {
        let t = AttentionTemplate::new(PtxType::F32, 48, 64, false);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_head_dim_too_large() {
        let t = AttentionTemplate::new(PtxType::F32, 512, 64, false);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_block_seq_zero() {
        let t = AttentionTemplate::new(PtxType::F32, 64, 0, false);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_block_seq_not_power_of_two() {
        let t = AttentionTemplate::new(PtxType::F32, 64, 48, false);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn generate_basic_f32() {
        let t = AttentionTemplate::new(PtxType::F32, 64, 64, false);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate attention kernel");
        assert!(ptx.contains(".entry attention_f32_hd64_bs64"));
        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("bar.sync 0"));
        assert!(ptx.contains("fma.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
    }

    #[test]
    fn generate_causal_f32() {
        let t = AttentionTemplate::new(PtxType::F32, 64, 64, true);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate causal attention");
        assert!(ptx.contains("attention_f32_hd64_bs64_causal"));
        // Causal mask: compare k_pos > q_pos
        assert!(ptx.contains("setp.gt.u32"));
        assert!(ptx.contains("causal"));
    }

    #[test]
    fn generate_f64() {
        let t = AttentionTemplate::new(PtxType::F64, 64, 32, false);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate f64 attention");
        assert!(ptx.contains("attention_f64_hd64_bs32"));
        assert!(ptx.contains("fma.f64"));
    }

    #[test]
    fn generate_small_head_dim() {
        let t = AttentionTemplate::new(PtxType::F32, 32, 32, false);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate with small head_dim");
        assert!(ptx.contains("attention_f32_hd32_bs32"));
    }

    #[test]
    fn generate_large_head_dim() {
        let t = AttentionTemplate::new(PtxType::F32, 256, 64, false);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate with head_dim=256");
        assert!(ptx.contains("attention_f32_hd256_bs64"));
    }

    #[test]
    fn builder_pattern() {
        let t = AttentionTemplate::new(PtxType::F32, 64, 64, false)
            .with_precision(PtxType::F64)
            .with_head_dim(128)
            .with_block_seq(32)
            .with_causal(true);
        assert_eq!(t.kernel_name(), "attention_f64_hd128_bs32_causal");
    }

    #[test]
    fn generate_contains_scale_param() {
        let t = AttentionTemplate::new(PtxType::F32, 64, 64, false);
        let ptx = t.generate(SmVersion::Sm80).expect("should generate");
        assert!(ptx.contains("%param_scale"));
    }

    #[test]
    fn generate_contains_online_softmax() {
        let t = AttentionTemplate::new(PtxType::F32, 64, 64, false);
        let ptx = t.generate(SmVersion::Sm80).expect("should generate");
        // Online softmax uses running max and sum with correction factor
        assert!(ptx.contains("online softmax") || ptx.contains("Online softmax"));
    }

    #[test]
    fn generate_different_sm_versions() {
        let t = AttentionTemplate::new(PtxType::F32, 64, 64, false);
        let ptx_75 = t
            .generate(SmVersion::Sm75)
            .expect("should generate for Sm75");
        let ptx_90 = t
            .generate(SmVersion::Sm90)
            .expect("should generate for Sm90");
        assert!(ptx_75.contains("sm_75"));
        assert!(ptx_90.contains("sm_90"));
    }

    #[test]
    fn generate_shared_memory_declared() {
        let t = AttentionTemplate::new(PtxType::F32, 64, 64, false);
        let ptx = t.generate(SmVersion::Sm80).expect("should generate");
        assert!(ptx.contains("smem_attn"));
    }

    #[test]
    fn invalid_block_seq_too_large() {
        let t = AttentionTemplate::new(PtxType::F32, 64, 512, false);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }
}
