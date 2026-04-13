//! Batch normalization kernel templates.
//!
//! Generates PTX kernels for batch normalization in both training and inference
//! modes. The kernels operate on NCHW-format tensors where N is the batch size,
//! C is the number of channels, and each channel has `spatial_size` = H*W elements.
//!
//! **Training mode**:
//! 1. Compute per-channel mean via parallel reduction: `mean_c = (1/N*HW) * sum(x)`
//! 2. Compute per-channel variance via parallel reduction: `var_c = (1/N*HW) * sum((x - mean)^2)`
//! 3. Normalize: `y = gamma * (x - mean) / sqrt(var + eps) + beta`
//!
//! **Inference mode**:
//! Uses precomputed running mean and variance (no reduction needed):
//! `y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta`
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::batch_norm::{BatchNormTemplate, BnMode};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = BatchNormTemplate::new(PtxType::F32, BnMode::Inference, 64, 1024, 1e-5, 256);
//! let ptx = template.generate(SmVersion::Sm80).expect("PTX generation failed");
//! assert!(ptx.contains("batch_norm"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Batch normalization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BnMode {
    /// Training mode: compute mean and variance from the input batch.
    Training,
    /// Inference mode: use precomputed running mean and variance.
    Inference,
}

impl BnMode {
    /// Returns a short string tag for kernel naming.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Training => "train",
            Self::Inference => "infer",
        }
    }
}

/// Template for generating batch normalization PTX kernels.
///
/// Each thread block handles one channel. For training mode, threads
/// cooperatively reduce over the spatial dimension to compute mean and variance.
/// For inference mode, threads directly apply the normalization transform using
/// precomputed statistics.
pub struct BatchNormTemplate {
    /// Data precision (F32 or F64).
    pub precision: PtxType,
    /// Training or inference mode.
    pub mode: BnMode,
    /// Number of channels (C dimension).
    pub channels: u32,
    /// Spatial size per channel per sample (H * W).
    pub spatial_size: u32,
    /// Epsilon for numerical stability in the denominator.
    pub epsilon: f32,
    /// Threads per block (must be power of 2, >= 32).
    pub block_size: u32,
}

impl BatchNormTemplate {
    /// Creates a new batch normalization template.
    #[must_use]
    pub const fn new(
        precision: PtxType,
        mode: BnMode,
        channels: u32,
        spatial_size: u32,
        epsilon: f32,
        block_size: u32,
    ) -> Self {
        Self {
            precision,
            mode,
            channels,
            spatial_size,
            epsilon,
            block_size,
        }
    }

    /// Sets the precision type. Returns `self` for chaining.
    #[must_use]
    pub const fn with_precision(mut self, precision: PtxType) -> Self {
        self.precision = precision;
        self
    }

    /// Sets the mode. Returns `self` for chaining.
    #[must_use]
    pub const fn with_mode(mut self, mode: BnMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the number of channels. Returns `self` for chaining.
    #[must_use]
    pub const fn with_channels(mut self, channels: u32) -> Self {
        self.channels = channels;
        self
    }

    /// Sets the spatial size. Returns `self` for chaining.
    #[must_use]
    pub const fn with_spatial_size(mut self, spatial_size: u32) -> Self {
        self.spatial_size = spatial_size;
        self
    }

    /// Sets the epsilon value. Returns `self` for chaining.
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Sets the block size. Returns `self` for chaining.
    #[must_use]
    pub const fn with_block_size(mut self, block_size: u32) -> Self {
        self.block_size = block_size;
        self
    }

    /// Returns the kernel function name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        format!(
            "batch_norm_{}_{}_c{}_s{}_bs{}",
            self.mode.as_str(),
            type_str,
            self.channels,
            self.spatial_size,
            self.block_size,
        )
    }

    /// Validates template parameters before code generation.
    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(self.precision, PtxType::F32 | PtxType::F64) {
            return Err(PtxGenError::InvalidType(format!(
                "batch_norm requires F32 or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }

        if self.block_size < 32 || !self.block_size.is_power_of_two() {
            return Err(PtxGenError::GenerationFailed(format!(
                "block_size must be a power of 2 >= 32, got {}",
                self.block_size
            )));
        }

        if self.channels == 0 {
            return Err(PtxGenError::GenerationFailed(
                "channels must be > 0".to_string(),
            ));
        }

        if self.spatial_size == 0 {
            return Err(PtxGenError::GenerationFailed(
                "spatial_size must be > 0".to_string(),
            ));
        }

        if self.epsilon <= 0.0 {
            return Err(PtxGenError::GenerationFailed(format!(
                "epsilon must be > 0, got {}",
                self.epsilon
            )));
        }

        if self.block_size > 1024 {
            return Err(PtxGenError::GenerationFailed(format!(
                "block_size {} exceeds maximum of 1024",
                self.block_size
            )));
        }

        Ok(())
    }

    /// Generates the complete PTX module text for the batch normalization kernel.
    ///
    /// **Training mode parameters**:
    /// - `input`: pointer to input tensor (N * C * `spatial_size` elements)
    /// - `output`: pointer to output tensor (same shape)
    /// - `gamma`: pointer to per-channel scale (C elements)
    /// - `beta`: pointer to per-channel shift (C elements)
    /// - `batch_count`: number of samples in the batch (N)
    ///
    /// **Inference mode parameters**:
    /// - `input`: pointer to input tensor
    /// - `output`: pointer to output tensor
    /// - `gamma`: pointer to per-channel scale
    /// - `beta`: pointer to per-channel shift
    /// - `running_mean`: pointer to per-channel running mean (C elements)
    /// - `running_var`: pointer to per-channel running variance (C elements)
    /// - `batch_count`: number of samples in the batch (N)
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if validation fails or PTX formatting fails.
    pub fn generate(&self, sm: SmVersion) -> Result<String, PtxGenError> {
        self.validate()?;

        match self.mode {
            BnMode::Training => self.generate_training(sm),
            BnMode::Inference => self.generate_inference(sm),
        }
    }

    /// Generates training-mode batch normalization kernel.
    ///
    /// Uses two-pass approach:
    /// 1. Parallel reduction to compute mean
    /// 2. Parallel reduction to compute variance
    /// 3. Normalize with gamma/beta
    #[allow(clippy::too_many_lines)]
    fn generate_training(&self, sm: SmVersion) -> Result<String, PtxGenError> {
        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        let block_size = self.block_size;
        let spatial_size = self.spatial_size;
        let smem_bytes = (block_size as usize) * byte_size;

        let zero_lit = match self.precision {
            PtxType::F64 => "0d0000000000000000",
            _ => "0f00000000",
        };

        // Epsilon as hex literal
        let eps_hex = format!("0f{:08X}", self.epsilon.to_bits());

        let mut ptx = String::with_capacity(8192);

        // Header
        writeln!(ptx, ".version {}", sm.ptx_version()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", sm.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_gamma,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_beta,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_count").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Declarations
        writeln!(ptx, "    .maxntid {block_size}, 1, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b32 %r<24>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<20>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<24>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    .shared .align {} .b8 smem_bn[{}];",
            byte_size.max(4),
            smem_bytes
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread/block indexing
        // Each block handles one channel: channel_idx = ctaid.x
        writeln!(ptx, "    // Thread and block indexing").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    // Load parameters").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_gamma];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd3, [%param_beta];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r2, [%param_batch_count];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute total elements for this channel across batch: N * spatial_size
        writeln!(
            ptx,
            "    // Compute total elements per channel across batch"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r3, %r2, {spatial_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Shared memory base
        writeln!(ptx, "    mov.u64 %rd4, smem_bn;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd4, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // =====================================================
        // PASS 1: Compute mean via parallel reduction
        // Each thread accumulates partial sum over its assigned elements
        // Element layout: input[n * C * spatial_size + channel * spatial_size + s]
        // =====================================================
        let channel_stride = spatial_size as usize * byte_size;
        let sample_stride_elems = self.channels as usize * spatial_size as usize;
        let sample_stride_bytes = sample_stride_elems * byte_size;

        writeln!(ptx, "    // Pass 1: Compute channel mean via reduction")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f0, {zero_lit};").map_err(PtxGenError::FormatError)?;

        // Base address for this channel: input + channel_idx * spatial_size * byte_size
        writeln!(ptx, "    cvt.u64.u32 %rd7, %r1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd7, %rd7, {channel_stride};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd8, %rd0, %rd7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Loop over batch samples, and within each sample over spatial elements
        // Thread i handles spatial elements i, i+block_size, i+2*block_size, ...
        // For each batch sample n: base = input + n * C * spatial * byte_size + ch * spatial * byte_size
        writeln!(ptx, "    mov.u32 %r4, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$MEAN_BATCH_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r4, %r2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $MEAN_BATCH_DONE;").map_err(PtxGenError::FormatError)?;

        // sample_base = rd8 + r4 * sample_stride_bytes
        writeln!(ptx, "    cvt.u64.u32 %rd9, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd9, %rd9, {sample_stride_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd10, %rd8, %rd9;").map_err(PtxGenError::FormatError)?;

        // Inner loop: spatial elements
        writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$MEAN_SPATIAL_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r5, {spatial_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $MEAN_SPATIAL_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd11, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd12, %rd10, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd12];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add{ty} %f0, %f0, %f1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $MEAN_SPATIAL_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$MEAN_SPATIAL_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r4, %r4, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $MEAN_BATCH_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$MEAN_BATCH_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Store partial sum to shared memory and reduce
        writeln!(ptx, "    st.shared{ty} [%rd6], %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;

        // Tree reduction for sum
        self.emit_shared_reduction(&mut ptx, ty, byte_size, "add", "MEAN_RED")?;

        // Thread 0 computes mean = sum / total_elements
        // Broadcast mean from shared memory position 0
        writeln!(ptx, "    ld.shared{ty} %f2, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.rn{ty}.u32 %f3, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    div.rn{ty} %f2, %f2, %f3;").map_err(PtxGenError::FormatError)?;
        // f2 = channel mean
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // =====================================================
        // PASS 2: Compute variance via parallel reduction
        // var = (1/N) * sum((x - mean)^2)
        // =====================================================
        writeln!(ptx, "    // Pass 2: Compute channel variance via reduction")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f4, {zero_lit};").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    mov.u32 %r4, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$VAR_BATCH_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r4, %r2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $VAR_BATCH_DONE;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    cvt.u64.u32 %rd9, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd9, %rd9, {sample_stride_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd10, %rd8, %rd9;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$VAR_SPATIAL_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r5, {spatial_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $VAR_SPATIAL_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd11, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd12, %rd10, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f5, [%rd12];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub{ty} %f5, %f5, %f2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    fma{ty} %f4, %f5, %f5, %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $VAR_SPATIAL_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$VAR_SPATIAL_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r4, %r4, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $VAR_BATCH_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$VAR_BATCH_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Reduce variance partial sums
        writeln!(ptx, "    st.shared{ty} [%rd6], %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;

        self.emit_shared_reduction(&mut ptx, ty, byte_size, "add", "VAR_RED")?;

        // Compute variance and inverse stddev
        writeln!(ptx, "    ld.shared{ty} %f6, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    div.rn{ty} %f6, %f6, %f3;").map_err(PtxGenError::FormatError)?;
        // f6 = variance
        // inv_stddev = 1 / sqrt(var + eps)
        writeln!(ptx, "    add{ty} %f7, %f6, {eps_hex};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sqrt.rn{ty} %f7, %f7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rcp.approx{ty} %f7, %f7;").map_err(PtxGenError::FormatError)?;
        // f7 = inv_stddev
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load gamma and beta for this channel
        writeln!(ptx, "    // Load gamma and beta for channel")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd13, %r1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd13, %rd13, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd14, %rd2, %rd13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f8, [%rd14];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd15, %rd3, %rd13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f9, [%rd15];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // =====================================================
        // PASS 3: Normalize and write output
        // y = gamma * (x - mean) * inv_stddev + beta
        // =====================================================
        writeln!(ptx, "    // Pass 3: Normalize and write output")
            .map_err(PtxGenError::FormatError)?;
        // Output channel base
        writeln!(ptx, "    add.u64 %rd16, %rd1, %rd7;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    mov.u32 %r4, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$NORM_BATCH_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r4, %r2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $BN_DONE;").map_err(PtxGenError::FormatError)?;

        // Input sample base
        writeln!(ptx, "    cvt.u64.u32 %rd9, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd9, %rd9, {sample_stride_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd10, %rd8, %rd9;").map_err(PtxGenError::FormatError)?;
        // Output sample base
        writeln!(ptx, "    add.u64 %rd17, %rd16, %rd9;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$NORM_SPATIAL_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r5, {spatial_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $NORM_SPATIAL_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd11, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        // Load input element
        writeln!(ptx, "    add.u64 %rd12, %rd10, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f10, [%rd12];").map_err(PtxGenError::FormatError)?;
        // Normalize: (x - mean) * inv_stddev
        writeln!(ptx, "    sub{ty} %f10, %f10, %f2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f10, %f10, %f7;").map_err(PtxGenError::FormatError)?;
        // Apply gamma and beta: y = gamma * normalized + beta
        writeln!(ptx, "    fma{ty} %f10, %f8, %f10, %f9;").map_err(PtxGenError::FormatError)?;
        // Store output
        writeln!(ptx, "    add.u64 %rd18, %rd17, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd18], %f10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $NORM_SPATIAL_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$NORM_SPATIAL_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r4, %r4, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $NORM_BATCH_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$BN_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    /// Generates inference-mode batch normalization kernel.
    ///
    /// Uses precomputed running mean and variance -- no reductions needed.
    #[allow(clippy::too_many_lines)]
    fn generate_inference(&self, sm: SmVersion) -> Result<String, PtxGenError> {
        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        let block_size = self.block_size;
        let spatial_size = self.spatial_size;

        // Epsilon as hex literal
        let eps_hex = format!("0f{:08X}", self.epsilon.to_bits());

        let mut ptx = String::with_capacity(4096);

        // Header
        writeln!(ptx, ".version {}", sm.ptx_version()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", sm.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Kernel signature (includes running_mean and running_var)
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_gamma,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_beta,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_running_mean,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_running_var,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_count").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Declarations
        writeln!(ptx, "    .maxntid {block_size}, 1, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b32 %r<20>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<20>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<4>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread/block indexing
        writeln!(ptx, "    // Thread and block indexing").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    // Load parameters").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_gamma];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd3, [%param_beta];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd4, [%param_running_mean];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd5, [%param_running_var];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r2, [%param_batch_count];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load running mean and variance for this channel
        writeln!(
            ptx,
            "    // Load running stats, gamma, beta for this channel"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd6, %r1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd6, %rd6, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd7, %rd4, %rd6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f0, [%rd7];").map_err(PtxGenError::FormatError)?;
        // f0 = running_mean
        writeln!(ptx, "    add.u64 %rd8, %rd5, %rd6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd8];").map_err(PtxGenError::FormatError)?;
        // f1 = running_var
        writeln!(ptx, "    add.u64 %rd9, %rd2, %rd6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f2, [%rd9];").map_err(PtxGenError::FormatError)?;
        // f2 = gamma
        writeln!(ptx, "    add.u64 %rd10, %rd3, %rd6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f3, [%rd10];").map_err(PtxGenError::FormatError)?;
        // f3 = beta
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Precompute inv_stddev = 1 / sqrt(running_var + eps)
        writeln!(ptx, "    // Compute inv_stddev = 1/sqrt(var + eps)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add{ty} %f4, %f1, {eps_hex};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sqrt.rn{ty} %f4, %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rcp.approx{ty} %f4, %f4;").map_err(PtxGenError::FormatError)?;
        // f4 = inv_stddev
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute channel base address
        let channel_stride = spatial_size as usize * byte_size;
        let sample_stride_elems = self.channels as usize * spatial_size as usize;
        let sample_stride_bytes = sample_stride_elems * byte_size;

        writeln!(ptx, "    // Compute channel base address").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd11, %r1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {channel_stride};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd12, %rd0, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd13, %rd1, %rd11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Apply normalization: y = gamma * (x - mean) * inv_stddev + beta
        writeln!(
            ptx,
            "    // Apply normalization across batch and spatial dims"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r3, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$INF_BATCH_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r3, %r2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $BN_INF_DONE;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    cvt.u64.u32 %rd14, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd14, %rd14, {sample_stride_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd15, %rd12, %rd14;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd16, %rd13, %rd14;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    mov.u32 %r4, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$INF_SPATIAL_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r4, {spatial_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $INF_SPATIAL_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd17, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd17, %rd17, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        // Load input
        writeln!(ptx, "    add.u64 %rd18, %rd15, %rd17;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f5, [%rd18];").map_err(PtxGenError::FormatError)?;
        // Normalize: (x - mean) * inv_stddev
        writeln!(ptx, "    sub{ty} %f5, %f5, %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f5, %f5, %f4;").map_err(PtxGenError::FormatError)?;
        // Scale and shift: gamma * norm + beta
        writeln!(ptx, "    fma{ty} %f5, %f2, %f5, %f3;").map_err(PtxGenError::FormatError)?;
        // Store output
        writeln!(ptx, "    add.u64 %rd19, %rd16, %rd17;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd19], %f5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r4, %r4, {block_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $INF_SPATIAL_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$INF_SPATIAL_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r3, %r3, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $INF_BATCH_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$BN_INF_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    /// Emits a shared memory tree reduction with the given combine operation.
    /// Assumes data is already in shared memory at `[%rd6]` (thread's slot).
    fn emit_shared_reduction(
        &self,
        ptx: &mut String,
        ty: &str,
        byte_size: usize,
        combine_op: &str,
        label_prefix: &str,
    ) -> Result<(), PtxGenError> {
        let mut stride = self.block_size / 2;
        while stride > 0 {
            writeln!(ptx, "    setp.lt.u32 %p2, %r0, {stride};")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    @!%p2 bra $SKIP_{label_prefix}_{stride};")
                .map_err(PtxGenError::FormatError)?;
            let partner_off = stride as usize * byte_size;
            writeln!(ptx, "    ld.shared{ty} %f11, [%rd6+{partner_off}];")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    ld.shared{ty} %f12, [%rd6];").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    {combine_op}{ty} %f12, %f12, %f11;")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    st.shared{ty} [%rd6], %f12;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "$SKIP_{label_prefix}_{stride}:").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
            stride /= 2;
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;

    #[test]
    fn kernel_name_training() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 64, 1024, 1e-5, 256);
        assert_eq!(t.kernel_name(), "batch_norm_train_f32_c64_s1024_bs256");
    }

    #[test]
    fn kernel_name_inference() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Inference, 128, 256, 1e-5, 128);
        assert_eq!(t.kernel_name(), "batch_norm_infer_f32_c128_s256_bs128");
    }

    #[test]
    fn kernel_name_f64() {
        let t = BatchNormTemplate::new(PtxType::F64, BnMode::Training, 32, 512, 1e-5, 256);
        assert_eq!(t.kernel_name(), "batch_norm_train_f64_c32_s512_bs256");
    }

    #[test]
    fn invalid_precision_u32() {
        let t = BatchNormTemplate::new(PtxType::U32, BnMode::Training, 64, 1024, 1e-5, 256);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_precision_f16() {
        let t = BatchNormTemplate::new(PtxType::F16, BnMode::Training, 64, 1024, 1e-5, 256);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_block_size_not_pow2() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 64, 1024, 1e-5, 100);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_block_size_too_small() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 64, 1024, 1e-5, 16);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_block_size_too_large() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 64, 1024, 1e-5, 2048);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_channels_zero() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 0, 1024, 1e-5, 256);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_spatial_size_zero() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 64, 0, 1e-5, 256);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_epsilon_zero() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 64, 1024, 0.0, 256);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn invalid_epsilon_negative() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 64, 1024, -1e-5, 256);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn generate_training_f32() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 64, 1024, 1e-5, 256);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate training BN");
        assert!(ptx.contains(".entry batch_norm_train_f32_c64_s1024_bs256"));
        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("bar.sync 0"));
        assert!(ptx.contains("sqrt.rn.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
        assert!(ptx.contains("fma.f32"));
        assert!(ptx.contains("%param_gamma"));
        assert!(ptx.contains("%param_beta"));
    }

    #[test]
    fn generate_inference_f32() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Inference, 64, 1024, 1e-5, 256);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate inference BN");
        assert!(ptx.contains(".entry batch_norm_infer_f32_c64_s1024_bs256"));
        assert!(ptx.contains("%param_running_mean"));
        assert!(ptx.contains("%param_running_var"));
        assert!(ptx.contains("sqrt.rn.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
        assert!(ptx.contains("fma.f32"));
    }

    #[test]
    fn generate_training_f64() {
        let t = BatchNormTemplate::new(PtxType::F64, BnMode::Training, 32, 512, 1e-5, 128);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate f64 training BN");
        assert!(ptx.contains("batch_norm_train_f64"));
        assert!(ptx.contains("fma.f64"));
    }

    #[test]
    fn generate_inference_f64() {
        let t = BatchNormTemplate::new(PtxType::F64, BnMode::Inference, 32, 512, 1e-5, 128);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate f64 inference BN");
        assert!(ptx.contains("batch_norm_infer_f64"));
    }

    #[test]
    fn generate_small_block() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 16, 64, 1e-5, 32);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("should generate with block_size=32");
        assert!(ptx.contains("batch_norm_train_f32_c16_s64_bs32"));
    }

    #[test]
    fn generate_different_sm_versions() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 64, 1024, 1e-5, 256);
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
    fn builder_pattern() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 64, 1024, 1e-5, 256)
            .with_precision(PtxType::F64)
            .with_mode(BnMode::Inference)
            .with_channels(128)
            .with_spatial_size(512)
            .with_epsilon(1e-6)
            .with_block_size(128);
        assert_eq!(t.kernel_name(), "batch_norm_infer_f64_c128_s512_bs128");
    }

    #[test]
    fn bn_mode_as_str() {
        assert_eq!(BnMode::Training.as_str(), "train");
        assert_eq!(BnMode::Inference.as_str(), "infer");
    }

    #[test]
    fn training_has_reduction_phases() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Training, 64, 1024, 1e-5, 256);
        let ptx = t.generate(SmVersion::Sm80).expect("should generate");
        // Must have mean and variance computation phases
        assert!(ptx.contains("Pass 1") || ptx.contains("mean"));
        assert!(ptx.contains("Pass 2") || ptx.contains("variance"));
        assert!(ptx.contains("Pass 3") || ptx.contains("Normalize"));
    }

    #[test]
    fn inference_no_shared_memory() {
        let t = BatchNormTemplate::new(PtxType::F32, BnMode::Inference, 64, 1024, 1e-5, 256);
        let ptx = t.generate(SmVersion::Sm80).expect("should generate");
        // Inference mode should not use shared memory reductions
        assert!(!ptx.contains(".shared"));
    }
}
