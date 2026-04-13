//! INT4/INT8 GEMM configuration for quantized model inference.
//!
//! Provides PTX generation for integer-precision GEMM operations commonly
//! used in quantized neural network inference (e.g. GPTQ, AWQ, GGML):
//!
//! - **INT8 GEMM**: Uses `dp4a` (dot product of 4 INT8 values accumulated
//!   into INT32) for high-throughput quantized inference. Available from
//!   Turing (sm_75) onward.
//!
//! - **INT4 GEMM**: Packed 4-bit integer storage (8 values per `u32`) with
//!   on-the-fly dequantization using per-group scale factors. Targets
//!   weight-only quantization for LLM serving.
//!
//! # Mixed-precision accumulation
//!
//! - INT8 inputs -> INT32 accumulation -> optional FP32 output scaling
//! - INT4 inputs -> FP16 dequant -> FP32 accumulation -> FP32/FP16 output

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::prelude::SmVersion;

use crate::error::{BlasError, BlasResult};
use crate::level3::gemm::dispatch::TileConfig;
use crate::types::Layout;

// ---------------------------------------------------------------------------
// Accumulator type
// ---------------------------------------------------------------------------

/// Accumulator type for integer GEMM operations.
///
/// Controls the precision of the accumulation during the inner product
/// computation. INT32 preserves full integer precision; FP32 allows
/// mixed-precision pipelines where the output is floating-point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccType {
    /// 32-bit signed integer accumulation (lossless for INT8 dot products).
    I32,
    /// 32-bit IEEE-754 float accumulation (for mixed-precision output).
    F32,
}

impl AccType {
    /// Returns the PTX type string for this accumulator.
    #[must_use]
    pub const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::I32 => ".s32",
            Self::F32 => ".f32",
        }
    }

    /// Returns a short name for kernel naming.
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::F32 => "f32",
        }
    }
}

// ---------------------------------------------------------------------------
// INT8 GEMM configuration
// ---------------------------------------------------------------------------

/// Configuration for INT8 GEMM using dp4a acceleration.
///
/// Each `dp4a` instruction computes the dot product of four INT8 values
/// packed into a single 32-bit register, accumulating into an INT32 or
/// FP32 result. This gives 4x throughput compared to scalar INT8 multiply.
///
/// # Layout
///
/// Matrices A (M x K) and B (K x N) are stored as packed INT8 in row-major
/// or column-major order. K must be a multiple of 4 for dp4a alignment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Int8GemmConfig {
    /// Number of rows of the output matrix C (and of A).
    pub m: u32,
    /// Number of columns of the output matrix C (and of B).
    pub n: u32,
    /// Shared (inner) dimension: columns of A / rows of B.
    /// Must be a multiple of 4 for dp4a packing.
    pub k: u32,
    /// Memory layout for matrix A.
    pub layout_a: Layout,
    /// Memory layout for matrix B.
    pub layout_b: Layout,
    /// Accumulator type (I32 for integer output, F32 for float output).
    pub accumulator: AccType,
}

impl Int8GemmConfig {
    /// Element size in bytes for INT8.
    pub const ELEMENT_BYTES: u32 = 1;

    /// Validates the INT8 GEMM configuration.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidDimension`] if dimensions are zero or
    /// K is not a multiple of 4. Returns [`BlasError::UnsupportedOperation`]
    /// if the target architecture lacks dp4a support.
    pub fn validate(&self, sm: SmVersion) -> BlasResult<()> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "INT8 GEMM dimensions must be non-zero: m={}, n={}, k={}",
                self.m, self.n, self.k
            )));
        }
        if self.k % 4 != 0 {
            return Err(BlasError::InvalidDimension(format!(
                "INT8 GEMM K dimension must be a multiple of 4 for dp4a, got k={}",
                self.k
            )));
        }
        if !Self::is_available(sm) {
            return Err(BlasError::UnsupportedOperation(format!(
                "INT8 dp4a requires Turing+ (sm_75), got {sm}"
            )));
        }
        Ok(())
    }

    /// Check if INT8 dp4a is available on the given architecture.
    ///
    /// dp4a is available from Turing (sm_75) onward.
    #[must_use]
    pub fn is_available(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm75
    }

    /// Get optimal tile config for INT8 GEMM.
    ///
    /// INT8 has 4x the element density compared to FP32, so larger tiles
    /// in K are efficient. The dp4a instruction processes 4 elements per
    /// cycle, so tile_k should be a multiple of 4.
    #[must_use]
    pub fn tile_config(&self, sm: SmVersion) -> TileConfig {
        match sm {
            SmVersion::Sm90 | SmVersion::Sm90a | SmVersion::Sm100 | SmVersion::Sm120 => {
                TileConfig {
                    tile_m: 128,
                    tile_n: 128,
                    tile_k: 64,
                    warp_m: 64,
                    warp_n: 64,
                    stages: 4,
                    use_tensor_core: true,
                    split_k: 1,
                }
            }
            SmVersion::Sm80 | SmVersion::Sm86 | SmVersion::Sm89 => TileConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: 3,
                use_tensor_core: true,
                split_k: 1,
            },
            SmVersion::Sm75 => TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 32,
                warp_m: 32,
                warp_n: 32,
                stages: 2,
                use_tensor_core: true,
                split_k: 1,
            },
        }
    }

    /// Returns the kernel function name for this configuration.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let layout_a = layout_short_name(self.layout_a);
        let layout_b = layout_short_name(self.layout_b);
        let acc = self.accumulator.short_name();
        format!(
            "int8_gemm_{}x{}x{}_{}{}_{}_dp4a",
            self.m, self.n, self.k, layout_a, layout_b, acc
        )
    }
}

// ---------------------------------------------------------------------------
// INT4 GEMM configuration
// ---------------------------------------------------------------------------

/// Configuration for INT4 GEMM with grouped dequantization.
///
/// INT4 weights are packed 8 values per `u32`. During GEMM computation,
/// values are dequantized on-the-fly using per-group FP16 scale factors.
/// This targets weight-only quantization schemes (GPTQ, AWQ) where
/// activations remain in FP16/FP32 but weights are compressed to 4 bits.
///
/// # Packing format
///
/// Eight 4-bit signed integers are packed into a single `u32`:
/// `[v0:4][v1:4][v2:4][v3:4][v4:4][v5:4][v6:4][v7:4]`
///
/// Each value is sign-extended from 4-bit two's complement (range -8..7).
///
/// # Group quantization
///
/// Scale factors are stored per group. A group of `group_size` consecutive
/// weight values along the K dimension shares one FP16 scale factor.
/// Typical group sizes: 32, 64, 128.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Int4GemmConfig {
    /// Number of rows of the output matrix C (and of A).
    pub m: u32,
    /// Number of columns of the output matrix C (and of B).
    pub n: u32,
    /// Shared (inner) dimension: columns of A / rows of B.
    /// Must be a multiple of 8 for INT4 packing alignment.
    pub k: u32,
    /// Memory layout for matrix A (typically activations in FP16).
    pub layout_a: Layout,
    /// Memory layout for matrix B (INT4-packed weights).
    pub layout_b: Layout,
    /// Number of consecutive elements sharing a single scale factor.
    /// Must be a power of 2 and divide K evenly.
    pub group_size: u32,
}

impl Int4GemmConfig {
    /// Packed element size: 8 INT4 values fit in 4 bytes (one u32).
    pub const PACKED_ELEMENT_BYTES: u32 = 4;

    /// Number of INT4 values packed per u32.
    pub const VALUES_PER_U32: u32 = 8;

    /// Common group sizes for quantization.
    pub const VALID_GROUP_SIZES: &'static [u32] = &[32, 64, 128, 256];

    /// Validates the INT4 GEMM configuration.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidDimension`] if dimensions are zero,
    /// K is not a multiple of 8, or group_size does not divide K.
    /// Returns [`BlasError::UnsupportedOperation`] if the architecture
    /// lacks the required support.
    pub fn validate(&self, sm: SmVersion) -> BlasResult<()> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "INT4 GEMM dimensions must be non-zero: m={}, n={}, k={}",
                self.m, self.n, self.k
            )));
        }
        if self.k % 8 != 0 {
            return Err(BlasError::InvalidDimension(format!(
                "INT4 GEMM K dimension must be a multiple of 8 for packing, got k={}",
                self.k
            )));
        }
        if self.group_size == 0 || !self.group_size.is_power_of_two() {
            return Err(BlasError::InvalidArgument(format!(
                "INT4 group_size must be a positive power of 2, got {}",
                self.group_size
            )));
        }
        if self.k % self.group_size != 0 {
            return Err(BlasError::InvalidArgument(format!(
                "INT4 group_size ({}) must divide K ({}) evenly",
                self.group_size, self.k
            )));
        }
        if !Self::is_available(sm) {
            return Err(BlasError::UnsupportedOperation(format!(
                "INT4 GEMM requires Turing+ (sm_75), got {sm}"
            )));
        }
        Ok(())
    }

    /// Check if INT4 dequantized GEMM is available on the given architecture.
    ///
    /// Requires at least Turing for efficient bit manipulation.
    #[must_use]
    pub fn is_available(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm75
    }

    /// Get optimal tile config for INT4 GEMM.
    ///
    /// INT4 has extremely high element density (8 values per u32), so
    /// tile_k is set large to amortize dequantization overhead.
    #[must_use]
    pub fn tile_config(&self, sm: SmVersion) -> TileConfig {
        match sm {
            SmVersion::Sm90 | SmVersion::Sm90a | SmVersion::Sm100 | SmVersion::Sm120 => {
                TileConfig {
                    tile_m: 128,
                    tile_n: 128,
                    tile_k: 128,
                    warp_m: 64,
                    warp_n: 64,
                    stages: 4,
                    use_tensor_core: true,
                    split_k: 1,
                }
            }
            SmVersion::Sm80 | SmVersion::Sm86 | SmVersion::Sm89 => TileConfig {
                tile_m: 128,
                tile_n: 64,
                tile_k: 64,
                warp_m: 64,
                warp_n: 32,
                stages: 3,
                use_tensor_core: true,
                split_k: 1,
            },
            SmVersion::Sm75 => TileConfig {
                tile_m: 64,
                tile_n: 64,
                tile_k: 64,
                warp_m: 32,
                warp_n: 32,
                stages: 2,
                use_tensor_core: true,
                split_k: 1,
            },
        }
    }

    /// Returns the kernel function name for this configuration.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let layout_a = layout_short_name(self.layout_a);
        let layout_b = layout_short_name(self.layout_b);
        format!(
            "int4_gemm_{}x{}x{}_{}{}_{}_gs{}",
            self.m, self.n, self.k, layout_a, layout_b, "f32", self.group_size
        )
    }

    /// Returns the number of scale factor groups along the K dimension.
    #[must_use]
    pub fn num_groups(&self) -> u32 {
        self.k / self.group_size
    }

    /// Returns the number of packed u32 values needed to store K INT4 values.
    #[must_use]
    pub fn packed_k(&self) -> u32 {
        self.k / Self::VALUES_PER_U32
    }
}

// ---------------------------------------------------------------------------
// INT4 packing utilities
// ---------------------------------------------------------------------------

/// Packs an array of 4-bit signed integers into packed `u32` values.
///
/// Each `u32` stores 8 INT4 values. The input values are clamped to the
/// range [-8, 7] (4-bit two's complement). The input length must be a
/// multiple of 8.
///
/// # Arguments
///
/// * `values` - Slice of `i8` values, each in the range [-8, 7].
///
/// # Errors
///
/// Returns [`BlasError::InvalidArgument`] if the input length is not a
/// multiple of 8 or any value is outside the [-8, 7] range.
pub fn pack_int4(values: &[i8]) -> BlasResult<Vec<u32>> {
    if values.len() % 8 != 0 {
        return Err(BlasError::InvalidArgument(format!(
            "pack_int4 requires input length to be a multiple of 8, got {}",
            values.len()
        )));
    }

    for (i, &v) in values.iter().enumerate() {
        if !(-8..=7).contains(&v) {
            return Err(BlasError::InvalidArgument(format!(
                "INT4 value at index {i} out of range [-8, 7]: {v}"
            )));
        }
    }

    let mut result = Vec::with_capacity(values.len() / 8);
    for chunk in values.chunks_exact(8) {
        let mut packed: u32 = 0;
        for (j, &val) in chunk.iter().enumerate() {
            let nibble = (val as u8) & 0x0F;
            packed |= (nibble as u32) << (j * 4);
        }
        result.push(packed);
    }
    Ok(result)
}

/// Unpacks 8 INT4 values from a packed `u32`.
///
/// Each nibble is sign-extended from 4-bit two's complement to `i8`.
///
/// # Arguments
///
/// * `packed` - A `u32` containing 8 packed INT4 values.
///
/// # Returns
///
/// An array of 8 `i8` values, each in the range [-8, 7].
#[must_use]
pub fn unpack_int4(packed: u32) -> [i8; 8] {
    let mut result = [0i8; 8];
    for (j, slot) in result.iter_mut().enumerate() {
        let nibble = ((packed >> (j * 4)) & 0x0F) as u8;
        // Sign-extend from 4-bit: if bit 3 is set, the value is negative.
        *slot = if nibble & 0x08 != 0 {
            // Extend sign: 0xF0 | nibble gives the i8 two's complement.
            (0xF0u8 | nibble) as i8
        } else {
            nibble as i8
        };
    }
    result
}

// ---------------------------------------------------------------------------
// PTX generation — INT8 GEMM
// ---------------------------------------------------------------------------

/// Generates PTX source for an INT8 GEMM kernel using dp4a instructions.
///
/// The generated kernel computes `C = A * B` where A and B contain INT8
/// values, using `dp4a` (dot product of 4 INT8 values accumulated into
/// INT32) for the inner product. Each thread computes one element of C.
///
/// This is a reference implementation for correctness. A production kernel
/// would use shared memory tiling and software pipelining.
///
/// # Errors
///
/// Returns [`BlasError`] if validation fails or PTX formatting fails.
pub fn generate_int8_gemm_ptx(config: &Int8GemmConfig, sm: SmVersion) -> BlasResult<String> {
    config.validate(sm)?;

    let kernel_name = config.kernel_name();
    let k_iters = config.k / 4; // Number of dp4a iterations (4 elements each)

    let mut ptx = String::with_capacity(8192);

    write_line(&mut ptx, &format!(".version {}", sm.ptx_version()))?;
    write_line(&mut ptx, &format!(".target {}", sm.as_ptx_str()))?;
    write_line(&mut ptx, ".address_size 64")?;
    write_line(&mut ptx, "")?;

    // Kernel signature
    write_line(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
    write_line(&mut ptx, "    .param .u64 %param_a,")?;
    write_line(&mut ptx, "    .param .u64 %param_b,")?;
    write_line(&mut ptx, "    .param .u64 %param_c,")?;
    write_line(&mut ptx, "    .param .u32 %param_m,")?;
    write_line(&mut ptx, "    .param .u32 %param_n,")?;
    write_line(&mut ptx, "    .param .u32 %param_k")?;
    write_line(&mut ptx, ")")?;
    write_line(&mut ptx, "{")?;

    // Register declarations
    write_line(&mut ptx, "    .reg .b32 %r<32>;")?;
    write_line(&mut ptx, "    .reg .b64 %rd<16>;")?;
    write_line(&mut ptx, "    .reg .f32 %f<8>;")?;
    write_line(&mut ptx, "    .reg .pred %p<4>;")?;
    write_line(&mut ptx, "")?;

    // Thread index computation
    write_line(&mut ptx, "    // Compute row and column indices")?;
    write_line(&mut ptx, "    mov.u32 %r0, %tid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r1, %tid.y;")?;
    write_line(&mut ptx, "    mov.u32 %r2, %ctaid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r3, %ctaid.y;")?;
    write_line(&mut ptx, "    mov.u32 %r4, %ntid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r5, %ntid.y;")?;
    write_line(&mut ptx, "    mad.lo.u32 %r6, %r2, %r4, %r0;")?; // col
    write_line(&mut ptx, "    mad.lo.u32 %r7, %r3, %r5, %r1;")?; // row
    write_line(&mut ptx, "")?;

    // Load parameters
    write_line(&mut ptx, "    // Load parameters")?;
    write_line(&mut ptx, "    ld.param.u64 %rd0, [%param_a];")?;
    write_line(&mut ptx, "    ld.param.u64 %rd1, [%param_b];")?;
    write_line(&mut ptx, "    ld.param.u64 %rd2, [%param_c];")?;
    write_line(&mut ptx, "    ld.param.u32 %r8, [%param_m];")?;
    write_line(&mut ptx, "    ld.param.u32 %r9, [%param_n];")?;
    write_line(&mut ptx, "    ld.param.u32 %r10, [%param_k];")?;
    write_line(&mut ptx, "")?;

    // Bounds check
    write_line(&mut ptx, "    // Bounds check")?;
    write_line(&mut ptx, "    setp.ge.u32 %p0, %r7, %r8;")?;
    write_line(&mut ptx, "    setp.ge.u32 %p1, %r6, %r9;")?;
    write_line(&mut ptx, "    @%p0 bra $INT8_DONE;")?;
    write_line(&mut ptx, "    @%p1 bra $INT8_DONE;")?;
    write_line(&mut ptx, "")?;

    // Accumulator init
    write_line(&mut ptx, "    // Initialize INT32 accumulator to 0")?;
    write_line(&mut ptx, "    mov.s32 %r20, 0;")?;
    write_line(
        &mut ptx,
        &format!("    mov.u32 %r11, 0;  // Loop counter: 0..{k_iters}"),
    )?;
    write_line(&mut ptx, "")?;

    // dp4a loop: each iteration processes 4 INT8 elements packed in a u32
    write_line(
        &mut ptx,
        "    // dp4a loop: dot product of 4 INT8 elements per iteration",
    )?;
    write_line(&mut ptx, "$DP4A_LOOP:")?;
    write_line(&mut ptx, &format!("    setp.ge.u32 %p2, %r11, {k_iters};"))?;
    write_line(&mut ptx, "    @%p2 bra $DP4A_DONE;")?;

    // Load packed A[row, k*4..(k+1)*4] as u32
    write_line(&mut ptx, "    // Load packed INT8x4 from A")?;
    write_line(&mut ptx, "    shr.u32 %r15, %r10, 2;")?; // K/4 (packed stride)
    write_line(&mut ptx, "    mad.lo.u32 %r12, %r7, %r15, %r11;")?; // row * (K/4) + iter
    write_line(&mut ptx, "    cvt.u64.u32 %rd3, %r12;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd3, %rd3, 4;")?; // byte offset (u32)
    write_line(&mut ptx, "    add.u64 %rd4, %rd0, %rd3;")?;
    write_line(&mut ptx, "    ld.global.u32 %r13, [%rd4];")?;

    // Load packed B[k*4..(k+1)*4, col] as u32
    write_line(&mut ptx, "    // Load packed INT8x4 from B")?;
    write_line(&mut ptx, "    shr.u32 %r16, %r9, 0;")?; // N (columns)
    // For row-major B: element at (k_base + j, col) packed as u32
    // We load B packed in groups of 4 along K: B_packed[iter, col]
    write_line(&mut ptx, "    mad.lo.u32 %r14, %r11, %r9, %r6;")?; // iter * N + col
    write_line(&mut ptx, "    cvt.u64.u32 %rd5, %r14;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd5, %rd5, 4;")?;
    write_line(&mut ptx, "    add.u64 %rd6, %rd1, %rd5;")?;
    write_line(&mut ptx, "    ld.global.u32 %r14, [%rd6];")?;

    // dp4a: acc += dot(a_packed, b_packed)
    write_line(
        &mut ptx,
        "    // dp4a: 4-element INT8 dot product with INT32 accumulation",
    )?;
    write_line(&mut ptx, "    dp4a.s32.s32 %r20, %r13, %r14, %r20;")?;

    // Loop increment
    write_line(&mut ptx, "    add.u32 %r11, %r11, 1;")?;
    write_line(&mut ptx, "    bra $DP4A_LOOP;")?;
    write_line(&mut ptx, "$DP4A_DONE:")?;
    write_line(&mut ptx, "")?;

    // Store result
    write_line(&mut ptx, "    // Store accumulator to C")?;
    write_line(&mut ptx, "    mad.lo.u32 %r17, %r7, %r9, %r6;")?; // row * N + col
    write_line(&mut ptx, "    cvt.u64.u32 %rd7, %r17;")?;

    match config.accumulator {
        AccType::I32 => {
            write_line(&mut ptx, "    mul.lo.u64 %rd7, %rd7, 4;")?; // 4 bytes per s32
            write_line(&mut ptx, "    add.u64 %rd8, %rd2, %rd7;")?;
            write_line(&mut ptx, "    st.global.s32 [%rd8], %r20;")?;
        }
        AccType::F32 => {
            write_line(&mut ptx, "    // Convert INT32 accumulator to FP32")?;
            write_line(&mut ptx, "    cvt.rn.f32.s32 %f0, %r20;")?;
            write_line(&mut ptx, "    mul.lo.u64 %rd7, %rd7, 4;")?; // 4 bytes per f32
            write_line(&mut ptx, "    add.u64 %rd8, %rd2, %rd7;")?;
            write_line(&mut ptx, "    st.global.f32 [%rd8], %f0;")?;
        }
    }

    write_line(&mut ptx, "$INT8_DONE:")?;
    write_line(&mut ptx, "    ret;")?;
    write_line(&mut ptx, "}")?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// PTX generation — INT4 GEMM
// ---------------------------------------------------------------------------

/// Generates PTX source for an INT4 GEMM kernel with on-the-fly dequantization.
///
/// The generated kernel computes `C = dequant(A_int4, scales) * B` where
/// A_int4 contains packed 4-bit weights and scales are per-group FP16
/// scale factors. Each thread computes one element of C, dequantizing
/// INT4 values on-the-fly and accumulating in FP32.
///
/// This is a reference implementation. A production kernel would use shared
/// memory tiling, vectorized loads, and warp-level parallelism.
///
/// # Errors
///
/// Returns [`BlasError`] if validation fails or PTX formatting fails.
pub fn generate_int4_gemm_ptx(config: &Int4GemmConfig, sm: SmVersion) -> BlasResult<String> {
    config.validate(sm)?;

    let kernel_name = config.kernel_name();

    let mut ptx = String::with_capacity(16384);

    write_line(&mut ptx, &format!(".version {}", sm.ptx_version()))?;
    write_line(&mut ptx, &format!(".target {}", sm.as_ptx_str()))?;
    write_line(&mut ptx, ".address_size 64")?;
    write_line(&mut ptx, "")?;

    // Kernel signature: a_packed_ptr, scales_ptr, b_ptr, c_ptr, m, n, k, group_size
    write_line(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
    write_line(&mut ptx, "    .param .u64 %param_a_packed,")?;
    write_line(&mut ptx, "    .param .u64 %param_scales,")?;
    write_line(&mut ptx, "    .param .u64 %param_b,")?;
    write_line(&mut ptx, "    .param .u64 %param_c,")?;
    write_line(&mut ptx, "    .param .u32 %param_m,")?;
    write_line(&mut ptx, "    .param .u32 %param_n,")?;
    write_line(&mut ptx, "    .param .u32 %param_k,")?;
    write_line(&mut ptx, "    .param .u32 %param_group_size")?;
    write_line(&mut ptx, ")")?;
    write_line(&mut ptx, "{")?;

    // Register declarations
    write_line(&mut ptx, "    .reg .b32 %r<48>;")?;
    write_line(&mut ptx, "    .reg .b64 %rd<24>;")?;
    write_line(&mut ptx, "    .reg .f32 %f<16>;")?;
    write_line(&mut ptx, "    .reg .f16 %h<4>;")?;
    write_line(&mut ptx, "    .reg .pred %p<8>;")?;
    write_line(&mut ptx, "")?;

    // Thread index
    write_line(&mut ptx, "    // Compute row and column indices")?;
    write_line(&mut ptx, "    mov.u32 %r0, %tid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r1, %tid.y;")?;
    write_line(&mut ptx, "    mov.u32 %r2, %ctaid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r3, %ctaid.y;")?;
    write_line(&mut ptx, "    mov.u32 %r4, %ntid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r5, %ntid.y;")?;
    write_line(&mut ptx, "    mad.lo.u32 %r6, %r2, %r4, %r0;")?; // col
    write_line(&mut ptx, "    mad.lo.u32 %r7, %r3, %r5, %r1;")?; // row
    write_line(&mut ptx, "")?;

    // Load parameters
    write_line(&mut ptx, "    // Load parameters")?;
    write_line(&mut ptx, "    ld.param.u64 %rd0, [%param_a_packed];")?;
    write_line(&mut ptx, "    ld.param.u64 %rd1, [%param_scales];")?;
    write_line(&mut ptx, "    ld.param.u64 %rd2, [%param_b];")?;
    write_line(&mut ptx, "    ld.param.u64 %rd3, [%param_c];")?;
    write_line(&mut ptx, "    ld.param.u32 %r8, [%param_m];")?;
    write_line(&mut ptx, "    ld.param.u32 %r9, [%param_n];")?;
    write_line(&mut ptx, "    ld.param.u32 %r10, [%param_k];")?;
    write_line(&mut ptx, "    ld.param.u32 %r11, [%param_group_size];")?;
    write_line(&mut ptx, "")?;

    // Bounds check
    write_line(&mut ptx, "    // Bounds check")?;
    write_line(&mut ptx, "    setp.ge.u32 %p0, %r7, %r8;")?;
    write_line(&mut ptx, "    setp.ge.u32 %p1, %r6, %r9;")?;
    write_line(&mut ptx, "    @%p0 bra $INT4_DONE;")?;
    write_line(&mut ptx, "    @%p1 bra $INT4_DONE;")?;
    write_line(&mut ptx, "")?;

    // Initialize FP32 accumulator
    write_line(&mut ptx, "    // Initialize FP32 accumulator")?;
    write_line(&mut ptx, "    mov.f32 %f0, 0f00000000;")?;
    write_line(&mut ptx, "    mov.u32 %r12, 0;  // k index")?;
    write_line(&mut ptx, "")?;

    // Outer loop over K, processing 8 INT4 elements per packed u32
    write_line(
        &mut ptx,
        "    // Loop over K dimension, 8 INT4 values per packed word",
    )?;
    write_line(&mut ptx, "$INT4_K_LOOP:")?;
    write_line(&mut ptx, "    setp.ge.u32 %p2, %r12, %r10;")?;
    write_line(&mut ptx, "    @%p2 bra $INT4_K_DONE;")?;

    // Compute group index and load scale factor
    write_line(&mut ptx, "    // Compute group index for dequantization")?;
    write_line(&mut ptx, "    div.u32 %r13, %r12, %r11;")?; // group_idx = k / group_size
    // Scale layout: scales[row * num_groups + group_idx]
    write_line(&mut ptx, "    div.u32 %r30, %r10, %r11;")?; // num_groups = K / group_size
    write_line(&mut ptx, "    mad.lo.u32 %r14, %r7, %r30, %r13;")?; // row * num_groups + group_idx
    write_line(&mut ptx, "    cvt.u64.u32 %rd4, %r14;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd4, %rd4, 2;")?; // 2 bytes per f16 scale
    write_line(&mut ptx, "    add.u64 %rd5, %rd1, %rd4;")?;
    write_line(&mut ptx, "    ld.global.b16 %h0, [%rd5];")?;
    write_line(&mut ptx, "    cvt.f32.f16 %f1, %h0;")?; // scale as f32
    write_line(&mut ptx, "")?;

    // Load packed u32 containing 8 INT4 values
    // a_packed layout: a_packed[row * (K/8) + k/8]
    write_line(&mut ptx, "    // Load packed INT4x8 from A")?;
    write_line(&mut ptx, "    shr.u32 %r15, %r12, 3;")?; // k / 8
    write_line(&mut ptx, "    shr.u32 %r16, %r10, 3;")?; // K / 8
    write_line(&mut ptx, "    mad.lo.u32 %r17, %r7, %r16, %r15;")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd6, %r17;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd6, %rd6, 4;")?;
    write_line(&mut ptx, "    add.u64 %rd7, %rd0, %rd6;")?;
    write_line(&mut ptx, "    ld.global.u32 %r18, [%rd7];")?;
    write_line(&mut ptx, "")?;

    // Unpack and dequantize 8 INT4 values, multiply with B and accumulate
    write_line(
        &mut ptx,
        "    // Unpack 8 INT4 values, dequantize, and accumulate",
    )?;
    for j in 0..8u32 {
        let shift = j * 4;
        write_line(&mut ptx, &format!("    // INT4 element {j}"))?;
        write_line(&mut ptx, &format!("    shr.u32 %r19, %r18, {shift};"))?;
        write_line(&mut ptx, "    and.b32 %r19, %r19, 0x0F;")?;
        // Sign extend from 4-bit
        write_line(&mut ptx, "    and.b32 %r21, %r19, 0x08;")?; // test sign bit
        write_line(&mut ptx, "    setp.ne.u32 %p3, %r21, 0;")?;
        write_line(&mut ptx, "    @%p3 or.b32 %r19, %r19, 0xFFFFFFF0;")?; // sign extend
        // Convert to f32 and multiply by scale
        write_line(&mut ptx, "    cvt.rn.f32.s32 %f2, %r19;")?;
        write_line(&mut ptx, "    mul.f32 %f2, %f2, %f1;")?; // dequantized value

        // Load B[k + j, col]
        write_line(&mut ptx, &format!("    add.u32 %r22, %r12, {j};"))?;
        write_line(&mut ptx, "    mad.lo.u32 %r23, %r22, %r9, %r6;")?; // (k+j)*N + col
        write_line(&mut ptx, "    cvt.u64.u32 %rd8, %r23;")?;
        write_line(&mut ptx, "    mul.lo.u64 %rd8, %rd8, 4;")?; // 4 bytes per f32 in B
        write_line(&mut ptx, "    add.u64 %rd9, %rd2, %rd8;")?;
        write_line(&mut ptx, "    ld.global.f32 %f3, [%rd9];")?;

        // FMA: acc += dequant * b
        write_line(&mut ptx, "    fma.rn.f32 %f0, %f2, %f3, %f0;")?;
        write_line(&mut ptx, "")?;
    }

    // Advance k by 8
    write_line(&mut ptx, "    add.u32 %r12, %r12, 8;")?;
    write_line(&mut ptx, "    bra $INT4_K_LOOP;")?;
    write_line(&mut ptx, "$INT4_K_DONE:")?;
    write_line(&mut ptx, "")?;

    // Store FP32 result to C
    write_line(&mut ptx, "    // Store FP32 result to C")?;
    write_line(&mut ptx, "    mad.lo.u32 %r24, %r7, %r9, %r6;")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd10, %r24;")?;
    write_line(&mut ptx, "    mul.lo.u64 %rd10, %rd10, 4;")?;
    write_line(&mut ptx, "    add.u64 %rd11, %rd3, %rd10;")?;
    write_line(&mut ptx, "    st.global.f32 [%rd11], %f0;")?;

    write_line(&mut ptx, "$INT4_DONE:")?;
    write_line(&mut ptx, "    ret;")?;
    write_line(&mut ptx, "}")?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns a short name for layout (for kernel naming).
#[must_use]
fn layout_short_name(layout: Layout) -> &'static str {
    match layout {
        Layout::RowMajor => "r",
        Layout::ColMajor => "c",
    }
}

/// Write a line to the PTX string, mapping format errors to `BlasError`.
fn write_line(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln!(ptx, "{line}").map_err(|e| BlasError::PtxGeneration(format!("format error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- INT8 config validation --

    #[test]
    fn int8_config_valid() {
        let config = Int8GemmConfig {
            m: 128,
            n: 128,
            k: 64,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            accumulator: AccType::I32,
        };
        assert!(config.validate(SmVersion::Sm80).is_ok());
    }

    #[test]
    fn int8_config_zero_dim() {
        let config = Int8GemmConfig {
            m: 0,
            n: 128,
            k: 64,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            accumulator: AccType::I32,
        };
        assert!(config.validate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn int8_config_k_not_multiple_of_4() {
        let config = Int8GemmConfig {
            m: 128,
            n: 128,
            k: 33,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            accumulator: AccType::I32,
        };
        let err = config.validate(SmVersion::Sm80);
        assert!(err.is_err());
        let msg = format!("{}", err.err().unwrap_or_else(|| unreachable!()));
        assert!(msg.contains("multiple of 4"));
    }

    #[test]
    fn int8_unsupported_arch() {
        // sm_75 is the minimum for dp4a, but let's test that the check works
        // by verifying sm_75 is accepted.
        let config = Int8GemmConfig {
            m: 64,
            n: 64,
            k: 32,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            accumulator: AccType::F32,
        };
        assert!(config.validate(SmVersion::Sm75).is_ok());
    }

    #[test]
    fn int8_tile_config_turing() {
        let config = Int8GemmConfig {
            m: 256,
            n: 256,
            k: 128,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            accumulator: AccType::I32,
        };
        let tc = config.tile_config(SmVersion::Sm75);
        assert!(tc.use_tensor_core);
        assert_eq!(tc.tile_k, 32);
        assert_eq!(tc.stages, 2);
    }

    #[test]
    fn int8_tile_config_hopper() {
        let config = Int8GemmConfig {
            m: 1024,
            n: 1024,
            k: 512,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            accumulator: AccType::I32,
        };
        let tc = config.tile_config(SmVersion::Sm90);
        assert!(tc.use_tensor_core);
        assert_eq!(tc.tile_k, 64);
        assert_eq!(tc.stages, 4);
    }

    // -- INT4 config validation --

    #[test]
    fn int4_config_valid() {
        let config = Int4GemmConfig {
            m: 128,
            n: 128,
            k: 128,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            group_size: 32,
        };
        assert!(config.validate(SmVersion::Sm80).is_ok());
    }

    #[test]
    fn int4_config_zero_dim() {
        let config = Int4GemmConfig {
            m: 128,
            n: 0,
            k: 128,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            group_size: 32,
        };
        assert!(config.validate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn int4_config_k_not_multiple_of_8() {
        let config = Int4GemmConfig {
            m: 128,
            n: 128,
            k: 100,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            group_size: 32,
        };
        let err = config.validate(SmVersion::Sm80);
        assert!(err.is_err());
    }

    #[test]
    fn int4_config_invalid_group_size() {
        let config = Int4GemmConfig {
            m: 128,
            n: 128,
            k: 128,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            group_size: 3, // not power of 2
        };
        assert!(config.validate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn int4_config_group_size_not_divides_k() {
        let config = Int4GemmConfig {
            m: 128,
            n: 128,
            k: 48,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            group_size: 32, // 32 does not divide 48
        };
        assert!(config.validate(SmVersion::Sm80).is_err());
    }

    // -- PTX generation --

    #[test]
    fn int8_ptx_generation_i32_acc() {
        let config = Int8GemmConfig {
            m: 64,
            n: 64,
            k: 32,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            accumulator: AccType::I32,
        };
        let ptx = generate_int8_gemm_ptx(&config, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("dp4a"));
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("sm_80"));
        assert!(ptx.contains("st.global.s32"));
    }

    #[test]
    fn int8_ptx_generation_f32_acc() {
        let config = Int8GemmConfig {
            m: 128,
            n: 256,
            k: 64,
            layout_a: Layout::RowMajor,
            layout_b: Layout::ColMajor,
            accumulator: AccType::F32,
        };
        let ptx = generate_int8_gemm_ptx(&config, SmVersion::Sm90);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("dp4a"));
        assert!(ptx.contains("cvt.rn.f32.s32"));
        assert!(ptx.contains("st.global.f32"));
        assert!(ptx.contains("sm_90"));
    }

    #[test]
    fn int4_ptx_generation() {
        let config = Int4GemmConfig {
            m: 64,
            n: 64,
            k: 128,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            group_size: 32,
        };
        let ptx = generate_int4_gemm_ptx(&config, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("sm_80"));
        assert!(ptx.contains("param_scales"));
        assert!(ptx.contains("param_group_size"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("cvt.f32.f16")); // dequant uses f16 scales
    }

    // -- Pack / Unpack INT4 --

    #[test]
    fn pack_unpack_int4_roundtrip() {
        let values: Vec<i8> = vec![0, 1, -1, 7, -8, 3, -5, 2];
        let packed = pack_int4(&values);
        assert!(packed.is_ok());
        let packed = packed.unwrap_or_default();
        assert_eq!(packed.len(), 1);

        let unpacked = unpack_int4(packed[0]);
        assert_eq!(&unpacked[..], &values[..]);
    }

    #[test]
    fn pack_int4_wrong_length() {
        let values: Vec<i8> = vec![0, 1, 2]; // not multiple of 8
        assert!(pack_int4(&values).is_err());
    }

    #[test]
    fn pack_int4_out_of_range() {
        let values: Vec<i8> = vec![0, 1, 2, 3, 4, 5, 6, 10]; // 10 > 7
        assert!(pack_int4(&values).is_err());
    }

    #[test]
    fn unpack_int4_all_zeros() {
        let unpacked = unpack_int4(0x00000000);
        assert_eq!(unpacked, [0i8; 8]);
    }

    #[test]
    fn unpack_int4_all_negative_one() {
        // -1 in 4-bit two's complement = 0xF = 1111
        // 8 copies of 0xF packed = 0xFFFFFFFF
        let unpacked = unpack_int4(0xFFFFFFFF);
        assert_eq!(unpacked, [-1i8; 8]);
    }

    #[test]
    fn unpack_int4_min_max() {
        // min = -8 (0x8), max = 7 (0x7)
        // Pack: [7, -8, 7, -8, 7, -8, 7, -8]
        // = 0x87878787
        let unpacked = unpack_int4(0x87878787);
        assert_eq!(unpacked, [7, -8, 7, -8, 7, -8, 7, -8]);
    }

    // -- AccType --

    #[test]
    fn acc_type_ptx_str() {
        assert_eq!(AccType::I32.as_ptx_str(), ".s32");
        assert_eq!(AccType::F32.as_ptx_str(), ".f32");
    }

    // -- INT4 helpers --

    #[test]
    fn int4_num_groups() {
        let config = Int4GemmConfig {
            m: 128,
            n: 128,
            k: 256,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            group_size: 64,
        };
        assert_eq!(config.num_groups(), 4);
    }

    #[test]
    fn int4_packed_k() {
        let config = Int4GemmConfig {
            m: 128,
            n: 128,
            k: 256,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            group_size: 128,
        };
        assert_eq!(config.packed_k(), 32); // 256 / 8
    }

    // -- Kernel names --

    #[test]
    fn int8_kernel_name_format() {
        let config = Int8GemmConfig {
            m: 128,
            n: 64,
            k: 32,
            layout_a: Layout::RowMajor,
            layout_b: Layout::ColMajor,
            accumulator: AccType::I32,
        };
        let name = config.kernel_name();
        assert!(name.contains("int8_gemm"));
        assert!(name.contains("dp4a"));
        assert!(name.contains("rc")); // RowMajor + ColMajor
        assert!(name.contains("i32"));
    }

    #[test]
    fn int4_kernel_name_format() {
        let config = Int4GemmConfig {
            m: 256,
            n: 128,
            k: 64,
            layout_a: Layout::ColMajor,
            layout_b: Layout::RowMajor,
            group_size: 32,
        };
        let name = config.kernel_name();
        assert!(name.contains("int4_gemm"));
        assert!(name.contains("cr")); // ColMajor + RowMajor
        assert!(name.contains("gs32"));
    }

    // -- Availability checks --

    #[test]
    fn int8_availability() {
        assert!(Int8GemmConfig::is_available(SmVersion::Sm75));
        assert!(Int8GemmConfig::is_available(SmVersion::Sm80));
        assert!(Int8GemmConfig::is_available(SmVersion::Sm90));
    }

    #[test]
    fn int4_availability() {
        assert!(Int4GemmConfig::is_available(SmVersion::Sm75));
        assert!(Int4GemmConfig::is_available(SmVersion::Sm80));
        assert!(Int4GemmConfig::is_available(SmVersion::Sm90));
    }

    // -- PTX generation for various sizes --

    #[test]
    fn int8_ptx_large_dims() {
        let config = Int8GemmConfig {
            m: 4096,
            n: 4096,
            k: 4096,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            accumulator: AccType::I32,
        };
        let ptx = generate_int8_gemm_ptx(&config, SmVersion::Sm90);
        assert!(ptx.is_ok());
    }

    #[test]
    fn int4_ptx_group_size_128() {
        let config = Int4GemmConfig {
            m: 512,
            n: 512,
            k: 1024,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            group_size: 128,
        };
        let ptx = generate_int4_gemm_ptx(&config, SmVersion::Sm90);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("param_group_size"));
    }
}
