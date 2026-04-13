//! Mixture of Experts (`MoE`) kernel templates.
//!
//! Generates PTX kernels for the four phases of `MoE` execution:
//!
//! 1. **Top-k gating**: Softmax over expert logits, select top-k experts per token,
//!    compute normalized gating weights.
//! 2. **Permute**: Scatter tokens to expert bins based on selected expert indices,
//!    tracking per-expert counts and enforcing capacity limits.
//! 3. **Expert GEMM**: Grouped matrix multiplication where each expert processes
//!    its assigned tokens using expert-specific weight matrices.
//! 4. **Unpermute**: Gather expert outputs back to original token order, applying
//!    gating weights and accumulating contributions from top-k experts.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::moe::MoETemplate;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = MoETemplate {
//!     num_experts: 8,
//!     top_k: 2,
//!     hidden_dim: 4096,
//!     expert_dim: 14336,
//!     capacity_factor: 1.25,
//!     sm_version: SmVersion::Sm80,
//!     float_type: PtxType::F32,
//! };
//! let ptx = template.generate_topk_gating().expect("gating kernel");
//! assert!(ptx.contains("topk_gating"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Configuration for Mixture of Experts kernel generation.
///
/// Encapsulates all parameters needed to generate the four `MoE` phase kernels:
/// gating, permutation, grouped GEMM, and unpermutation. The `float_type`
/// determines whether `F32` or `F16` arithmetic is used throughout.
#[derive(Debug, Clone)]
pub struct MoETemplate {
    /// Number of experts in the `MoE` layer (e.g., 8, 16, 64).
    pub num_experts: u32,
    /// Number of experts selected per token (e.g., 1 or 2).
    pub top_k: u32,
    /// Hidden dimension of the model (input/output size per token).
    pub hidden_dim: u32,
    /// Expert feed-forward dimension (intermediate size).
    pub expert_dim: u32,
    /// Capacity factor controlling max tokens per expert (e.g., 1.25).
    /// The per-expert capacity is `ceil(capacity_factor * batch_size * top_k / num_experts)`.
    pub capacity_factor: f32,
    /// Target GPU architecture.
    pub sm_version: SmVersion,
    /// Floating-point type for computation (F32 or F16).
    pub float_type: PtxType,
}

impl MoETemplate {
    /// Returns a type suffix string for kernel naming (e.g., `"f32"`, `"f16"`).
    const fn type_suffix(&self) -> &'static str {
        match self.float_type {
            PtxType::F16 => "f16",
            PtxType::F64 => "f64",
            _ => "f32",
        }
    }

    /// Returns the PTX type string for this template's float type.
    const fn ty(&self) -> &'static str {
        self.float_type.as_ptx_str()
    }

    /// Returns the byte size of the float type.
    const fn byte_size(&self) -> usize {
        self.float_type.size_bytes()
    }

    /// Returns the zero literal for the configured float type.
    const fn zero_lit(&self) -> &'static str {
        match self.float_type {
            PtxType::F64 => "0d0000000000000000",
            _ => "0f00000000",
        }
    }

    /// Returns the negative infinity literal for the configured float type.
    const fn neg_inf(&self) -> &'static str {
        match self.float_type {
            PtxType::F64 => "0dFFF0000000000000",
            _ => "0fFF800000",
        }
    }

    /// Validates the template configuration.
    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(self.float_type, PtxType::F16 | PtxType::F32 | PtxType::F64) {
            return Err(PtxGenError::InvalidType(format!(
                "MoE requires F16, F32, or F64, got {}",
                self.float_type.as_ptx_str()
            )));
        }
        if self.num_experts == 0 {
            return Err(PtxGenError::GenerationFailed(
                "num_experts must be > 0".to_string(),
            ));
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(PtxGenError::GenerationFailed(format!(
                "top_k must be in [1, num_experts={}], got {}",
                self.num_experts, self.top_k
            )));
        }
        if self.hidden_dim == 0 {
            return Err(PtxGenError::GenerationFailed(
                "hidden_dim must be > 0".to_string(),
            ));
        }
        if self.expert_dim == 0 {
            return Err(PtxGenError::GenerationFailed(
                "expert_dim must be > 0".to_string(),
            ));
        }
        if self.capacity_factor <= 0.0 || !self.capacity_factor.is_finite() {
            return Err(PtxGenError::GenerationFailed(format!(
                "capacity_factor must be a positive finite value, got {}",
                self.capacity_factor
            )));
        }
        Ok(())
    }

    /// Writes the standard PTX header (version, target, address size).
    fn write_header(&self, ptx: &mut String) -> Result<(), PtxGenError> {
        writeln!(ptx, ".version {}", self.sm_version.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.sm_version.as_ptx_str())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Phase 1: Top-k Gating Kernel
    // -----------------------------------------------------------------------

    /// Returns the kernel name for the top-k gating kernel.
    #[must_use]
    pub fn topk_gating_kernel_name(&self) -> String {
        format!(
            "topk_gating_{}_{}e_top{}",
            self.type_suffix(),
            self.num_experts,
            self.top_k
        )
    }

    /// Generates a PTX kernel for top-k expert selection with softmax gating.
    ///
    /// The kernel performs per-token softmax over expert logits, then selects
    /// the top-k experts with the highest probability. Gating weights are
    /// the normalized probabilities of the selected experts.
    ///
    /// **Parameters:**
    /// - `logits`: pointer to expert logits `[batch_size, num_experts]`
    /// - `expert_indices`: output pointer `[batch_size, top_k]` (u32)
    /// - `expert_weights`: output pointer `[batch_size, top_k]` (float)
    /// - `batch_size`: number of tokens
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if validation fails or formatting fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate_topk_gating(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.ty();
        let byte_size = self.byte_size();
        let kernel_name = self.topk_gating_kernel_name();
        let num_experts = self.num_experts;
        let top_k = self.top_k;
        let neg_inf = self.neg_inf();
        let zero_lit = self.zero_lit();

        let mut ptx = String::with_capacity(8192);
        self.write_header(&mut ptx)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_logits,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_expert_indices,")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_expert_weights,")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Register declarations
        writeln!(ptx, "    .reg .b32 %r<48>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<24>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread indexing: one thread per token
        writeln!(ptx, "    // Compute global thread index (token index)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check
        writeln!(ptx, "    ld.param.u32 %r4, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r3, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $GATING_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load base pointers
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_logits];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_expert_indices];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_expert_weights];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute logits base address for this token: logits + token_idx * num_experts * byte_size
        let row_bytes = (num_experts as usize) * byte_size;
        writeln!(ptx, "    // Compute logits row address for token %r3")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {row_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd0, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 1: Find maximum logit for numerical stability
        writeln!(
            ptx,
            "    // Pass 1: Find max logit across {num_experts} experts"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f0, {neg_inf};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r5, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$GATING_MAX_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r5, {num_experts};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $GATING_MAX_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd4, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    max{ty} %f0, %f0, %f1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r5, %r5, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $GATING_MAX_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$GATING_MAX_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 2: Compute exp(logit - max) and sum for softmax denominator
        writeln!(ptx, "    // Pass 2: Compute exp(logit_i - max) and sum")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f2, {zero_lit};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r5, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$GATING_EXP_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r5, {num_experts};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $GATING_EXP_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd4, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f3, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub{ty} %f3, %f3, %f0;").map_err(PtxGenError::FormatError)?;
        // Use ex2 approximation for exp: exp(x) = 2^(x * log2(e))
        // log2(e) ~= 1.4426950408889634
        let log2e = match self.float_type {
            PtxType::F64 => "0d3FF71547652B82FE",
            _ => "0f3FB8AA3B",
        };
        writeln!(ptx, "    mul{ty} %f4, %f3, {log2e};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f3, %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add{ty} %f2, %f2, %f3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r5, %r5, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $GATING_EXP_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$GATING_EXP_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute reciprocal of sum for normalization
        writeln!(ptx, "    // Compute 1/sum for normalization")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rcp.approx{ty} %f5, %f2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 3: Top-k selection via iterative argmax
        // For each k in 0..top_k:
        //   - Find expert with highest softmax prob (excluding already selected)
        //   - Store expert index and normalized weight
        //   - Mark expert as used by setting its prob to -inf
        writeln!(ptx, "    // Pass 3: Top-k selection (iterative argmax)")
            .map_err(PtxGenError::FormatError)?;

        // Use r10 as the top-k iteration counter
        writeln!(ptx, "    mov.u32 %r10, 0;").map_err(PtxGenError::FormatError)?;

        // Track selected expert weight sum for renormalization
        writeln!(ptx, "    mov{ty} %f20, {zero_lit};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$TOPK_ITER_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p2, %r10, {top_k};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $TOPK_ITER_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Find argmax across experts
        writeln!(ptx, "    // Find argmax among {num_experts} experts")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f6, {neg_inf};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r11, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r12, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$TOPK_ARGMAX_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p3, %r12, {num_experts};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p3 bra $TOPK_ARGMAX_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd7, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd7, %rd7, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd8, %rd4, %rd7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f7, [%rd8];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.gt{ty} %p4, %f7, %f6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 mov{ty} %f6, %f7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 mov.u32 %r11, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r12, %r12, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $TOPK_ARGMAX_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$TOPK_ARGMAX_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute softmax probability for this expert: exp(logit - max) / sum
        writeln!(ptx, "    // Compute softmax prob for selected expert")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub{ty} %f8, %f6, %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f9, %f8, {log2e};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f8, %f9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f8, %f8, %f5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Accumulate weight sum for renormalization
        writeln!(ptx, "    add{ty} %f20, %f20, %f8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Store expert index: expert_indices[token * top_k + k]
        let out_idx_elem_bytes = 4_usize; // u32
        let out_weight_elem_bytes = byte_size;
        writeln!(ptx, "    // Store expert index and weight").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd9, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd9, %rd9, {top_k};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd10, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd9, %rd9, %rd10;").map_err(PtxGenError::FormatError)?;

        // Store expert index
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd9, {out_idx_elem_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd12, %rd1, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global.u32 [%rd12], %r11;").map_err(PtxGenError::FormatError)?;

        // Store expert weight (will be renormalized in a final pass)
        writeln!(ptx, "    mul.lo.u64 %rd13, %rd9, {out_weight_elem_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd14, %rd2, %rd13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd14], %f8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Mark this expert as used: store -inf at its logit position
        writeln!(
            ptx,
            "    // Mark selected expert as used (set logit to -inf)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd15, %r11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd15, %rd15, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd16, %rd4, %rd15;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd16], {neg_inf};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Advance top-k counter
        writeln!(ptx, "    add.u32 %r10, %r10, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $TOPK_ITER_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$TOPK_ITER_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 4: Renormalize weights so they sum to 1
        writeln!(ptx, "    // Pass 4: Renormalize top-k weights to sum to 1")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rcp.approx{ty} %f21, %f20;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r13, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$RENORM_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p5, %r13, {top_k};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p5 bra $RENORM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd17, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd17, %rd17, {top_k};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd18, %r13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd17, %rd17, %rd18;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd19, %rd17, {out_weight_elem_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd20, %rd2, %rd19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f22, [%rd20];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f22, %f22, %f21;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd20], %f22;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r13, %r13, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $RENORM_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$RENORM_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$GATING_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    // -----------------------------------------------------------------------
    // Phase 2: Permute Kernel
    // -----------------------------------------------------------------------

    /// Returns the kernel name for the permute kernel.
    #[must_use]
    pub fn permute_kernel_name(&self) -> String {
        format!(
            "moe_permute_{}_{}e_top{}",
            self.type_suffix(),
            self.num_experts,
            self.top_k
        )
    }

    /// Generates a PTX kernel for token permutation into expert bins.
    ///
    /// Each token is scattered to its assigned expert's bin. Per-expert token
    /// counts are tracked atomically, and a capacity limit is enforced so that
    /// no expert receives more than `capacity` tokens.
    ///
    /// **Parameters:**
    /// - `expert_indices`: input `[batch_size, top_k]` (u32)
    /// - `token_ids`: output `[num_experts, capacity]` mapping expert slot to source token (u32)
    /// - `expert_counts`: output `[num_experts]` atomic counters (u32)
    /// - `batch_size`: number of tokens
    /// - `capacity`: max tokens per expert
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if validation fails or formatting fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate_permute(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let kernel_name = self.permute_kernel_name();
        let num_experts = self.num_experts;
        let top_k = self.top_k;

        let mut ptx = String::with_capacity(4096);
        self.write_header(&mut ptx)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_expert_indices,")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_token_ids,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_expert_counts,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_capacity").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    .reg .b32 %r<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread index = token index
        writeln!(ptx, "    // Global thread index = token index")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check
        writeln!(ptx, "    ld.param.u32 %r4, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r3, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $PERMUTE_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load base pointers and capacity
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_expert_indices];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_token_ids];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_expert_counts];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r5, [%param_capacity];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // For each of the top_k expert assignments for this token
        writeln!(ptx, "    // Process {top_k} expert assignment(s) per token")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r6, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$PERMUTE_K_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r6, {top_k};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $PERMUTE_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load expert index: expert_indices[token * top_k + k]
        writeln!(ptx, "    // Load expert index for (token, k)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r7, %r3, {top_k}, %r6;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, 4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd0, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global.u32 %r8, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Validate expert index
        writeln!(ptx, "    setp.ge.u32 %p2, %r8, {num_experts};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $PERMUTE_K_NEXT;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Atomically increment expert count: slot = atomicAdd(&expert_counts[expert], 1)
        writeln!(ptx, "    // Atomic increment of expert count")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, 4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd2, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    atom.global.add.u32 %r9, [%rd6], 1;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Capacity check: if slot >= capacity, skip (token dropped)
        writeln!(ptx, "    // Enforce capacity limit").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p3, %r9, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p3 bra $PERMUTE_K_NEXT;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Store token id: token_ids[expert * capacity + slot] = token_index
        writeln!(ptx, "    // Store token index into expert bin")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r10, %r8, %r5, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd7, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd7, %rd7, 4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd8, %rd1, %rd7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global.u32 [%rd8], %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$PERMUTE_K_NEXT:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r6, %r6, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $PERMUTE_K_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$PERMUTE_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    // -----------------------------------------------------------------------
    // Phase 3: Expert GEMM Kernel
    // -----------------------------------------------------------------------

    /// Returns the kernel name for the expert GEMM kernel.
    #[must_use]
    pub fn expert_gemm_kernel_name(&self) -> String {
        format!(
            "moe_expert_gemm_{}_{}e_{}x{}",
            self.type_suffix(),
            self.num_experts,
            self.hidden_dim,
            self.expert_dim
        )
    }

    /// Generates a PTX kernel for grouped GEMM across experts.
    ///
    /// Each expert has its own weight matrix `W[expert, expert_dim, hidden_dim]`.
    /// For each expert, the kernel multiplies the expert's assigned tokens by
    /// its weight matrix. Shared memory tiling is used for efficiency.
    ///
    /// **Parameters:**
    /// - `input`: pointer to token embeddings `[total_tokens, hidden_dim]`
    /// - `weights`: pointer to expert weights `[num_experts, expert_dim, hidden_dim]`
    /// - `output`: pointer to results `[total_tokens, expert_dim]`
    /// - `token_ids`: permuted token indices `[num_experts, capacity]` (u32)
    /// - `expert_counts`: actual tokens per expert `[num_experts]` (u32)
    /// - `capacity`: max tokens per expert
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if validation fails or formatting fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate_expert_gemm(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.ty();
        let byte_size = self.byte_size();
        let kernel_name = self.expert_gemm_kernel_name();
        let num_experts = self.num_experts;
        let hidden_dim = self.hidden_dim;
        let expert_dim = self.expert_dim;
        let zero_lit = self.zero_lit();

        // Tile size for shared memory tiling
        let tile_size: u32 = 16;
        let tile_bytes = (tile_size as usize) * (tile_size as usize) * byte_size;
        let total_smem = tile_bytes * 2; // Two tiles: A tile and B tile

        let mut ptx = String::with_capacity(8192);
        self.write_header(&mut ptx)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_weights,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_token_ids,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_expert_counts,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_capacity").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Register and shared memory declarations
        writeln!(ptx, "    .reg .b32 %r<48>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<24>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    .shared .align {} .b8 smem_gemm[{}];",
            byte_size.max(4),
            total_smem
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread/block layout:
        // blockIdx.z = expert_id
        // blockIdx.y = tile row in expert_dim
        // blockIdx.x = token tile within this expert's assigned tokens
        // threadIdx.x = column within tile, threadIdx.y = row within tile
        writeln!(ptx, "    // Thread and block indexing").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %tid.y;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r3, %ctaid.y;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r4, %ctaid.z;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Validate expert_id < num_experts
        writeln!(ptx, "    setp.ge.u32 %p0, %r4, {num_experts};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $GEMM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_weights];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd3, [%param_token_ids];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd4, [%param_expert_counts];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r5, [%param_capacity];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load expert token count: expert_counts[expert_id]
        writeln!(ptx, "    // Load number of tokens assigned to this expert")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, 4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd4, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global.u32 %r6, [%rd6];").map_err(PtxGenError::FormatError)?;
        // Clamp to capacity
        writeln!(ptx, "    min.u32 %r6, %r6, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute global row/col this thread is responsible for
        // token_tile_idx = blockIdx.x * tile_size + threadIdx.y => local token index
        // expert_dim_idx = blockIdx.y * tile_size + threadIdx.x => expert_dim column
        writeln!(ptx, "    // Compute output coordinates").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r7, %r2, {tile_size}, %r1;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r8, %r3, {tile_size}, %r0;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check: local token index < expert_count, expert_dim_idx < expert_dim
        writeln!(ptx, "    setp.ge.u32 %p1, %r7, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p2, %r8, {expert_dim};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    or.pred %p3, %p1, %p2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Resolve actual token index from token_ids: token_ids[expert * capacity + local_idx]
        writeln!(
            ptx,
            "    // Resolve global token index from permutation table"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r9, %r4, %r5, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd7, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd7, %rd7, 4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd8, %rd3, %rd7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r10, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p1 ld.global.u32 %r10, [%rd8];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Shared memory base pointers
        writeln!(ptx, "    mov.u64 %rd9, smem_gemm;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd10, %rd9, {tile_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Initialize accumulator
        writeln!(ptx, "    mov{ty} %f0, {zero_lit};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Tiled GEMM: iterate over hidden_dim in chunks of tile_size
        // For each tile: load A[token, hidden_chunk] and B[expert, expert_dim_idx, hidden_chunk]
        let hidden_dim_val = hidden_dim;
        writeln!(
            ptx,
            "    // Tiled reduction over hidden_dim ({hidden_dim_val})"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r11, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$GEMM_K_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p4, %r11, {hidden_dim_val};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 bra $GEMM_K_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load A tile element: input[global_token_id, k_offset + threadIdx.x]
        let hidden_bytes = (hidden_dim as usize) * byte_size;
        writeln!(ptx, "    // Load input element for tiled multiply")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r12, %r11, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.u32 %p5, %r12, {hidden_dim_val};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f1, {zero_lit};").map_err(PtxGenError::FormatError)?;
        // Compute address: input + global_token_id * hidden_dim * byte_size + (k + tid.x) * byte_size
        writeln!(ptx, "    cvt.u64.u32 %rd11, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {hidden_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd12, %rd0, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd13, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd13, %rd13, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd14, %rd12, %rd13;").map_err(PtxGenError::FormatError)?;
        // Conditional load (only if within bounds and token is valid)
        writeln!(ptx, "    and.pred %p6, %p5, !%p3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p6 ld.global{ty} %f1, [%rd14];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Store to shared memory A tile: smem_A[tid.y * tile + tid.x]
        writeln!(ptx, "    // Store to shared memory tile A").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r13, %r1, {tile_size}, %r0;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd15, %r13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd15, %rd15, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd16, %rd9, %rd15;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.shared{ty} [%rd16], %f1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load B tile element: weights[expert * expert_dim * hidden_dim + expert_dim_idx * hidden_dim + k_offset + tid.y]
        let expert_weight_size = (expert_dim as usize) * (hidden_dim as usize) * byte_size;
        writeln!(ptx, "    // Load weight element for tiled multiply")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r14, %r11, %r1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.u32 %p5, %r14, {hidden_dim_val};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f2, {zero_lit};").map_err(PtxGenError::FormatError)?;
        // Compute address: weights + expert_id * expert_dim * hidden_dim * byte_size
        //                          + expert_dim_idx * hidden_dim * byte_size
        //                          + (k_offset + tid.y) * byte_size
        writeln!(ptx, "    cvt.u64.u32 %rd17, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd17, %rd17, {expert_weight_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd18, %rd1, %rd17;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd19, %r8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd19, %rd19, {hidden_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd20, %rd18, %rd19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd21, %r14;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd21, %rd21, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd22, %rd20, %rd21;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    and.pred %p7, %p5, !%p2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p7 ld.global{ty} %f2, [%rd22];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Store to shared memory B tile: smem_B[tid.y * tile + tid.x]
        writeln!(ptx, "    // Store to shared memory tile B").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r15, %r1, {tile_size}, %r0;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd23, %r15;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd23, %rd23, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd24, %rd10, %rd23;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.shared{ty} [%rd24], %f2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Synchronize
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Dot product over tile: accumulate A[tid.y, t] * B[t, tid.x] for t in 0..tile_size
        writeln!(ptx, "    // Inner product over shared memory tile")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r16, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$GEMM_TILE_DOT:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p4, %r16, {tile_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 bra $GEMM_TILE_DOT_DONE;").map_err(PtxGenError::FormatError)?;

        // Load A[tid.y, t] from smem
        writeln!(ptx, "    mad.lo.u32 %r17, %r1, {tile_size}, %r16;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd25, %r17;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd25, %rd25, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd26, %rd9, %rd25;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.shared{ty} %f3, [%rd26];").map_err(PtxGenError::FormatError)?;

        // Load B[t, tid.x] from smem
        writeln!(ptx, "    mad.lo.u32 %r18, %r16, {tile_size}, %r0;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd27, %r18;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd27, %rd27, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd28, %rd10, %rd27;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.shared{ty} %f4, [%rd28];").map_err(PtxGenError::FormatError)?;

        // FMA: acc += a * b
        writeln!(ptx, "    fma.rn{ty} %f0, %f3, %f4, %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r16, %r16, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $GEMM_TILE_DOT;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$GEMM_TILE_DOT_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Sync before next tile
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r11, %r11, {tile_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $GEMM_K_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$GEMM_K_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Write result if within bounds
        // output[global_token_id * expert_dim + expert_dim_idx]
        let expert_dim_bytes = (expert_dim as usize) * byte_size;
        writeln!(ptx, "    // Write result to output").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p3 bra $GEMM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd29, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd29, %rd29, {expert_dim_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd30, %rd2, %rd29;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd31, %r8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd31, %rd31, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd30, %rd30, %rd31;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd30], %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$GEMM_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    // -----------------------------------------------------------------------
    // Phase 4: Unpermute Kernel
    // -----------------------------------------------------------------------

    /// Returns the kernel name for the unpermute kernel.
    #[must_use]
    pub fn unpermute_kernel_name(&self) -> String {
        format!(
            "moe_unpermute_{}_{}e_top{}",
            self.type_suffix(),
            self.num_experts,
            self.top_k
        )
    }

    /// Generates a PTX kernel for result unpermutation with gating weight application.
    ///
    /// Gathers expert outputs back to original token order. For each token, the
    /// outputs from its top-k experts are weighted by the corresponding gating
    /// weights and accumulated to produce the final output.
    ///
    /// **Parameters:**
    /// - `expert_output`: input `[total_tokens, expert_dim]` (expert results)
    /// - `output`: final output `[batch_size, expert_dim]`
    /// - `expert_indices`: `[batch_size, top_k]` (u32)
    /// - `expert_weights`: `[batch_size, top_k]` (float)
    /// - `token_ids`: permuted mapping `[num_experts, capacity]` (u32)
    /// - `expert_counts`: `[num_experts]` (u32)
    /// - `batch_size`: number of tokens
    /// - `capacity`: max tokens per expert
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if validation fails or formatting fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate_unpermute(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.ty();
        let byte_size = self.byte_size();
        let kernel_name = self.unpermute_kernel_name();
        let num_experts = self.num_experts;
        let top_k = self.top_k;
        let expert_dim = self.expert_dim;
        let zero_lit = self.zero_lit();

        let mut ptx = String::with_capacity(8192);
        self.write_header(&mut ptx)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_expert_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_expert_indices,")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_expert_weights,")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_token_ids,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_expert_counts,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_capacity").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    .reg .b32 %r<48>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<24>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread layout: each thread handles one (token, dim_element) pair
        // blockIdx.x * blockDim.x + threadIdx.x = global_idx
        // token_idx = global_idx / expert_dim
        // dim_idx = global_idx % expert_dim
        writeln!(ptx, "    // Compute token and dimension indices")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // token_idx = r3 / expert_dim, dim_idx = r3 % expert_dim
        writeln!(ptx, "    div.u32 %r4, %r3, {expert_dim};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rem.u32 %r5, %r3, {expert_dim};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check
        writeln!(ptx, "    ld.param.u32 %r6, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r4, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $UNPERM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_expert_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_expert_indices];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd3, [%param_expert_weights];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd4, [%param_token_ids];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd5, [%param_expert_counts];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r7, [%param_capacity];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Initialize accumulator for this output element
        writeln!(ptx, "    mov{ty} %f0, {zero_lit};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Loop over top_k experts for this token
        writeln!(
            ptx,
            "    // Accumulate weighted outputs from {top_k} expert(s)"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r8, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$UNPERM_K_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r8, {top_k};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $UNPERM_K_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load expert_index and expert_weight for (token, k)
        let weight_elem_bytes = byte_size;
        writeln!(ptx, "    // Load expert index and weight for (token, k)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r9, %r4, {top_k}, %r8;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd6, %r9;").map_err(PtxGenError::FormatError)?;
        // Load expert index
        writeln!(ptx, "    mul.lo.u64 %rd7, %rd6, 4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd8, %rd2, %rd7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global.u32 %r10, [%rd8];").map_err(PtxGenError::FormatError)?;
        // Load gating weight
        writeln!(ptx, "    mul.lo.u64 %rd9, %rd6, {weight_elem_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd10, %rd3, %rd9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd10];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Validate expert index
        writeln!(ptx, "    setp.ge.u32 %p2, %r10, {num_experts};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $UNPERM_K_NEXT;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Find this token's position in the expert's bin by scanning token_ids
        // We need to find slot s such that token_ids[expert * capacity + s] == token_idx
        // and s < expert_counts[expert]
        writeln!(ptx, "    // Load expert count and search for token slot")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd11, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, 4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd12, %rd5, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global.u32 %r11, [%rd12];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    min.u32 %r11, %r11, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Linear scan for token position within expert's bin
        writeln!(ptx, "    // Linear scan for token in expert bin")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r12, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r13, 0xFFFFFFFF;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$UNPERM_SCAN_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p3, %r12, %r11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p3 bra $UNPERM_SCAN_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r14, %r10, %r7, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd13, %r14;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd13, %rd13, 4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd14, %rd4, %rd13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global.u32 %r15, [%rd14];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.eq.u32 %p4, %r15, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 mov.u32 %r13, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p4 bra $UNPERM_SCAN_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r12, %r12, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $UNPERM_SCAN_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$UNPERM_SCAN_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // If token not found (r13 == 0xFFFFFFFF), skip
        writeln!(ptx, "    setp.eq.u32 %p5, %r13, 0xFFFFFFFF;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p5 bra $UNPERM_K_NEXT;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load expert output: expert_output[global_token_slot * expert_dim + dim_idx]
        // global_token_slot is the position in the flattened expert output
        // which is stored at index (expert * capacity + slot) in the permuted order
        let expert_dim_bytes = (expert_dim as usize) * byte_size;
        writeln!(ptx, "    // Load expert output and apply gating weight")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r16, %r10, %r7, %r13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd15, %r16;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd15, %rd15, {expert_dim_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd16, %rd0, %rd15;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd17, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd17, %rd17, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd18, %rd16, %rd17;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f2, [%rd18];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Apply gating weight: acc += weight * expert_output
        writeln!(ptx, "    fma.rn{ty} %f0, %f1, %f2, %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$UNPERM_K_NEXT:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r8, %r8, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $UNPERM_K_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$UNPERM_K_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Store final accumulated output: output[token_idx * expert_dim + dim_idx]
        writeln!(ptx, "    // Store weighted sum to output").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd19, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd19, %rd19, {expert_dim_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd20, %rd1, %rd19;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd21, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd21, %rd21, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd22, %rd20, %rd21;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd22], %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$UNPERM_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;
    use crate::ir::PtxType;

    fn default_template() -> MoETemplate {
        MoETemplate {
            num_experts: 8,
            top_k: 2,
            hidden_dim: 4096,
            expert_dim: 14336,
            capacity_factor: 1.25,
            sm_version: SmVersion::Sm80,
            float_type: PtxType::F32,
        }
    }

    // -- Kernel name tests --

    #[test]
    fn topk_gating_kernel_name() {
        let t = default_template();
        assert_eq!(t.topk_gating_kernel_name(), "topk_gating_f32_8e_top2");
    }

    #[test]
    fn permute_kernel_name() {
        let t = default_template();
        assert_eq!(t.permute_kernel_name(), "moe_permute_f32_8e_top2");
    }

    #[test]
    fn expert_gemm_kernel_name() {
        let t = default_template();
        assert_eq!(
            t.expert_gemm_kernel_name(),
            "moe_expert_gemm_f32_8e_4096x14336"
        );
    }

    #[test]
    fn unpermute_kernel_name() {
        let t = default_template();
        assert_eq!(t.unpermute_kernel_name(), "moe_unpermute_f32_8e_top2");
    }

    // -- Gating kernel tests --

    #[test]
    fn generate_topk_gating_f32() {
        let t = default_template();
        let ptx = t
            .generate_topk_gating()
            .expect("should generate gating kernel");
        assert!(ptx.contains(".entry topk_gating_f32_8e_top2"));
        assert!(ptx.contains("ex2.approx"));
        assert!(ptx.contains("rcp.approx"));
        assert!(ptx.contains("max.f32"));
        assert!(ptx.contains("st.global.u32"));
        assert!(ptx.contains("st.global.f32"));
        assert!(ptx.contains(".version 7.0"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn generate_topk_gating_f16() {
        let t = MoETemplate {
            float_type: PtxType::F16,
            ..default_template()
        };
        let ptx = t
            .generate_topk_gating()
            .expect("should generate f16 gating");
        assert!(ptx.contains("topk_gating_f16_8e_top2"));
        assert!(ptx.contains(".f16"));
    }

    #[test]
    fn generate_topk_gating_top1() {
        let t = MoETemplate {
            top_k: 1,
            ..default_template()
        };
        let ptx = t
            .generate_topk_gating()
            .expect("should generate top-1 gating");
        assert!(ptx.contains("topk_gating_f32_8e_top1"));
    }

    // -- Permute kernel tests --

    #[test]
    fn generate_permute_f32() {
        let t = default_template();
        let ptx = t
            .generate_permute()
            .expect("should generate permute kernel");
        assert!(ptx.contains(".entry moe_permute_f32_8e_top2"));
        assert!(ptx.contains("atom.global.add.u32"));
        assert!(ptx.contains("setp.ge.u32")); // capacity check
        assert!(ptx.contains("st.global.u32"));
    }

    #[test]
    fn generate_permute_capacity_enforcement() {
        let t = default_template();
        let ptx = t
            .generate_permute()
            .expect("should generate permute kernel");
        // Verify capacity check: slot >= capacity leads to skip
        assert!(ptx.contains("@%p3 bra $PERMUTE_K_NEXT"));
        // Verify atomic increment for expert count
        assert!(ptx.contains("atom.global.add.u32"));
    }

    // -- Expert GEMM kernel tests --

    #[test]
    fn generate_expert_gemm_f32() {
        let t = default_template();
        let ptx = t
            .generate_expert_gemm()
            .expect("should generate expert gemm kernel");
        assert!(ptx.contains(".entry moe_expert_gemm_f32_8e_4096x14336"));
        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("bar.sync 0"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("ld.shared.f32"));
        assert!(ptx.contains("st.shared.f32"));
    }

    #[test]
    fn generate_expert_gemm_uses_tiling() {
        let t = default_template();
        let ptx = t
            .generate_expert_gemm()
            .expect("should generate expert gemm");
        // Verify tiled loop structure
        assert!(ptx.contains("$GEMM_K_LOOP"));
        assert!(ptx.contains("$GEMM_K_DONE"));
        assert!(ptx.contains("$GEMM_TILE_DOT"));
    }

    // -- Unpermute kernel tests --

    #[test]
    fn generate_unpermute_f32() {
        let t = default_template();
        let ptx = t
            .generate_unpermute()
            .expect("should generate unpermute kernel");
        assert!(ptx.contains(".entry moe_unpermute_f32_8e_top2"));
        assert!(ptx.contains("fma.rn.f32")); // gating weight application
        assert!(ptx.contains("$UNPERM_K_LOOP")); // top-k accumulation loop
        assert!(ptx.contains("st.global.f32")); // final output store
    }

    #[test]
    fn generate_unpermute_gating_weights() {
        let t = default_template();
        let ptx = t.generate_unpermute().expect("should generate unpermute");
        // Verify gating weight is loaded and applied via FMA
        assert!(ptx.contains("ld.global.f32 %f1"));
        assert!(ptx.contains("fma.rn.f32 %f0, %f1, %f2, %f0"));
    }

    // -- Validation tests --

    #[test]
    fn invalid_float_type() {
        let t = MoETemplate {
            float_type: PtxType::U32,
            ..default_template()
        };
        assert!(t.generate_topk_gating().is_err());
        assert!(t.generate_permute().is_err());
        assert!(t.generate_expert_gemm().is_err());
        assert!(t.generate_unpermute().is_err());
    }

    #[test]
    fn invalid_num_experts_zero() {
        let t = MoETemplate {
            num_experts: 0,
            ..default_template()
        };
        assert!(t.generate_topk_gating().is_err());
    }

    #[test]
    fn invalid_top_k_exceeds_experts() {
        let t = MoETemplate {
            top_k: 16,
            num_experts: 8,
            ..default_template()
        };
        assert!(t.generate_topk_gating().is_err());
    }

    #[test]
    fn invalid_hidden_dim_zero() {
        let t = MoETemplate {
            hidden_dim: 0,
            ..default_template()
        };
        assert!(t.generate_expert_gemm().is_err());
    }

    #[test]
    fn invalid_capacity_factor() {
        let t = MoETemplate {
            capacity_factor: -1.0,
            ..default_template()
        };
        assert!(t.generate_permute().is_err());

        let t2 = MoETemplate {
            capacity_factor: f32::INFINITY,
            ..default_template()
        };
        assert!(t2.generate_permute().is_err());
    }

    // -- Architecture variation tests --

    #[test]
    fn generate_for_sm90() {
        let t = MoETemplate {
            sm_version: SmVersion::Sm90,
            ..default_template()
        };
        let ptx = t
            .generate_topk_gating()
            .expect("should generate for Hopper");
        assert!(ptx.contains(".target sm_90"));
        assert!(ptx.contains(".version 8.0"));
    }

    #[test]
    fn generate_16_experts_top1() {
        let t = MoETemplate {
            num_experts: 16,
            top_k: 1,
            ..default_template()
        };
        let ptx = t.generate_topk_gating().expect("should generate 16e top1");
        assert!(ptx.contains("topk_gating_f32_16e_top1"));
    }
}
